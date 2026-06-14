#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explanation Card S1 : 정상 reference bank FULL 빌더 (정상 72명)

기준:
  - docs/explanation_card_plan_v1.md (S1)
  - reports/explanation_cards/s1_reference_bank_preflight_v1.md
  - reports/explanation_cards/s1_full_preflight_v1.md
  - smoke 스크립트 build_explanation_card_s1_reference_bank_smoke.py (보존; 본 파일은 별도)

설계 결정:
  - 통계 bank 로직은 smoke 검증본을 그대로 동결(roi>=0.50, 동일 통계).
  - 보강은 대표 crop '선정/표시' 레이어에만 적용(crop 후보 roi>=0.70 + 보수적 air penalty + mask contour).
  - 대상 = v2 normal score 존재 정상 72명(val+test). 362 전체 금지.
  - join key = safe_id. stage2_holdout/lesion 접근 금지.

가드:
  - 플래그 없으면 BLOCKED. --selftest/--dry-run/--plan-full-only 는 read-only.
  - --run-full 은 --confirm-generate 동반 필요. DONE.json/잔여 산출물 있으면 BLOCKED. --overwrite 없음.
  - full run 외 다른 실행모드 없음. 본 단계에서 --run-full --confirm-generate 미실행.
"""

import argparse
import csv
import inspect
import json
import math
import os
import sys
from datetime import datetime

# ----------------------------------------------------------------------------
# 확정 상수
# ----------------------------------------------------------------------------
MODEL_TAG = "padim_v2_roi0_0"
FRAME = "roi_0_0"
MASK_TAG = "refined_roi_v4_20_modeB_all_v1"
THRESHOLD_P95 = 14.0921

ALLOWED_GROUPS = ("val", "test")

# 통계 bank 필터 (smoke 동결)
STATS_ROI_MIN = 0.50

# 대표 crop 보강 (crop 레이어 한정)
CROP_ROI_MIN = 0.70                 # 대표 crop 후보 roi 하한 (통계엔 미적용)
AIR_DENSE_TARGET = 0.05             # dense_frac_hu_gt_minus500 이 이 값 미만이면 air penalty
CENTRAL_AIR_W = 3.0                 # 중심 bin air penalty 가중(혈관 기대 -> air 강하게 페널티)
PERIPHERAL_AIR_W = 1.0             # 말초 bin 약하게(정상 air 특성 보존)
LOW_ROI_W = 1.0                    # roi 낮을수록 penalty
KEEP_PER_PATIENT_PER_BIN = 2        # 패스 중 환자/ bin 당 후보 보관 상한(메모리 bound)
CROP_N_PER_BIN = 5
CROP_M_PER_PATIENT = 1              # bin별 환자당 최대 대표 crop

BIN_LOW_SAMPLE = 100
LUNG_WINDOW_CENTER = -600.0
LUNG_WINDOW_WIDTH = 1500.0
CONTOUR_RGB = (0, 255, 0)           # v4 mask contour 색

POSITION_BINS = (
    "upper_central", "upper_peripheral",
    "middle_central", "middle_peripheral",
    "lower_central", "lower_peripheral",
)

FORBIDDEN_PATH_TOKENS = ("stage2_holdout", "holdout", "lesion")

PLANNED_ARTIFACTS = (
    "reference_stats_by_position_bin.csv",
    "reference_crop_manifest.csv",
    "reference_crops/<position_bin>/*.png",
    "runtime_summary.json",
    "errors.csv",
    "DONE.json",
)

STATS_FIELDS = (
    "position_bin", "valid_patch_count", "patient_count",
    "p90_hu_p5", "p90_hu_p50", "p90_hu_p95",
    "p99_hu_p5", "p99_hu_p50", "p99_hu_p95",
    "mean_hu_p5", "mean_hu_p50", "mean_hu_p95",
    "dense_frac_hu_gt_minus500_p5", "dense_frac_hu_gt_minus500_p50", "dense_frac_hu_gt_minus500_p95",
    "dense_frac_hu_gt_minus300_p5", "dense_frac_hu_gt_minus300_p50", "dense_frac_hu_gt_minus300_p95",
    "roi_patch_ratio_p50", "low_sample_flag",
)

MANIFEST_FIELDS = (
    "safe_id", "patient_id", "group", "position_bin", "slice_index",
    "y0", "x0", "y1", "x1", "padim_score", "roi_0_0_patch_ratio",
    "dense_frac_hu_gt_minus500", "mean_hu",
    "crop_png_path", "selection_reason",
)

ERROR_FIELDS = ("safe_id", "patient_id", "stage", "detail")
ERROR_STAGES = ("load", "shape", "parse", "bbox", "empty_roi", "png", "other")

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_BLOCKED = 2

# ----------------------------------------------------------------------------
# 경로
# ----------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NORMAL_SCORE_DIR = os.path.join(
    REPO, "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/normal_by_patient")
V4_NORMAL_MASK_DIR = os.path.join(
    REPO, "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1/normal")
THRESHOLD_JSON = os.path.join(
    REPO, "outputs/position-aware-padim-v1/evaluation/normal_v2_roi0_0/normal_v2_threshold.json")
CT_HU_SOURCE_ROOT = (
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")

BANK_ROOT = os.path.join(
    REPO, "outputs/position-aware-padim-v1/visualizations/candidate_cards/reference_bank_v1")
FULL_OUT = os.path.join(BANK_ROOT, "full")


# ----------------------------------------------------------------------------
# 가드
# ----------------------------------------------------------------------------
def safe_path(path):
    low = str(path).replace("\\", "/").lower()
    for tok in FORBIDDEN_PATH_TOKENS:
        if tok in low:
            raise RuntimeError("FORBIDDEN path token '%s' in: %s" % (tok, path))
    return path


def is_file(path):
    return os.path.isfile(safe_path(path))


def is_dir(path):
    return os.path.isdir(safe_path(path))


# ----------------------------------------------------------------------------
# read-only 헬퍼
# ----------------------------------------------------------------------------
def list_normal_score_csvs():
    if not is_dir(NORMAL_SCORE_DIR):
        return []
    return [os.path.join(NORMAL_SCORE_DIR, n)
            for n in sorted(os.listdir(safe_path(NORMAL_SCORE_DIR))) if n.lower().endswith(".csv")]


def read_score_header(csv_path):
    with open(safe_path(csv_path), "r", encoding="utf-8-sig", newline="") as f:
        return [h.strip() for h in next(csv.reader(f), [])]


def read_first_row_meta(csv_path):
    with open(safe_path(csv_path), "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        row = next(reader, None)
    if not row:
        return None
    return {"patient_id": (row.get("patient_id") or "").strip(),
            "safe_id": (row.get("safe_id") or "").strip(),
            "group": (row.get("group") or "").strip()}


def iter_score_rows(csv_path):
    with open(safe_path(csv_path), "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        for row in reader:
            yield row


def resolve_mask_path(safe_id):
    return os.path.join(V4_NORMAL_MASK_DIR, safe_id, "refined_roi.npy")


def resolve_ct_meta_path(safe_id):
    return os.path.join(CT_HU_SOURCE_ROOT, safe_id, "meta.json")


def resolve_ct_hu_path(safe_id):
    return os.path.join(CT_HU_SOURCE_ROOT, safe_id, "ct_hu.npy")


# ----------------------------------------------------------------------------
# 대상 정상 환자 (전체 72)
# ----------------------------------------------------------------------------
def collect_eligible_normals():
    csvs = list_normal_score_csvs()
    audit = {"score_csv_count": len(csvs), "blocked": False, "block_reason": None}
    eligible = []
    for path in csvs:
        meta = read_first_row_meta(path)
        if meta is None:
            continue
        grp = meta["group"].lower()
        if ("lesion" in grp) or ("holdout" in grp):
            audit["blocked"] = True
            audit["block_reason"] = "non-normal group: %s (%s)" % (grp, os.path.basename(path))
            return [], audit
        if grp not in ALLOWED_GROUPS:
            continue
        sid = meta["safe_id"]
        if not sid:
            continue
        rec = {"safe_id": sid, "patient_id": meta["patient_id"], "group": grp, "score_csv": path,
               "mask_exists": is_file(resolve_mask_path(sid)),
               "ct_meta_exists": is_file(resolve_ct_meta_path(sid))}
        if rec["mask_exists"] and rec["ct_meta_exists"]:
            eligible.append(rec)
    eligible.sort(key=lambda r: (ALLOWED_GROUPS.index(r["group"]), r["safe_id"]))
    audit["eligible_count"] = len(eligible)
    return eligible, audit


def resolve_full_targets():
    """전체 eligible 정상(72)을 그대로 대상으로 반환 (smoke 처럼 3명으로 자르지 않음)."""
    eligible, audit = collect_eligible_normals()
    if audit.get("blocked"):
        return [], audit
    audit["full_target_count"] = len(eligible)
    return eligible, audit


# ----------------------------------------------------------------------------
# 순수 계산 (selftest 대상)
# ----------------------------------------------------------------------------
def compute_patch_hu_stats(hu_patch, roi_patch):
    """통계 bank 동결 로직 (smoke 동일)."""
    import numpy as np
    mask = np.asarray(roi_patch).astype(bool)
    vals = np.asarray(hu_patch)[mask]
    if vals.size == 0:
        return None
    out = {"p90_hu": float(np.percentile(vals, 90)),
           "p99_hu": float(np.percentile(vals, 99)),
           "mean_hu": float(np.mean(vals)),
           "std_hu": float(np.std(vals)),
           "n_vox": int(vals.size)}
    out["dense_frac_hu_gt_minus500"] = float(np.mean(vals > -500.0))
    out["dense_frac_hu_gt_minus300"] = float(np.mean(vals > -300.0))
    return out


def _is_valid_score(s):
    return (s is not None) and (not math.isnan(s)) and (not math.isinf(s))


def air_penalty(position_bin, dense_frac500):
    """순수 air patch 보수적 penalty. dense_frac500 이 target 미만일 때만, 말초는 약하게."""
    air_term = max(0.0, AIR_DENSE_TARGET - float(dense_frac500))
    w = CENTRAL_AIR_W if position_bin.endswith("central") else PERIPHERAL_AIR_W
    return w * air_term


def provisional_crop_key(position_bin, dense_frac500, roi_ratio):
    """median-독립 후보 정렬키(낮을수록 우선). 환자/ bin 당 top-K 보관에 사용."""
    return air_penalty(position_bin, dense_frac500) + LOW_ROI_W * (1.0 - float(roi_ratio))


def select_representative_full(crop_pool, bin_medians,
                               n_per_bin=CROP_N_PER_BIN, m_per_patient=CROP_M_PER_PATIENT):
    """
    crop_pool: bin -> list of record dict
      (keys: safe_id, patient_id, group, position_bin, slice_index, y0,x0,y1,x1,
             padim_score, roi_0_0_patch_ratio, dense_frac_hu_gt_minus500, mean_hu)
    full selection_score = |score-median| + air_penalty + LOW_ROI_W*(1-roi). 낮을수록 우선.
    환자당 m, bin당 n.
    """
    selected = {}
    for b in crop_pool:
        med = bin_medians.get(b)
        recs = crop_pool[b]

        def full_score(r):
            md = abs(r["padim_score"] - med) if med is not None else 0.0
            ap = air_penalty(b, r["dense_frac_hu_gt_minus500"])
            return md + ap + LOW_ROI_W * (1.0 - float(r["roi_0_0_patch_ratio"]))

        ordered = sorted(recs, key=full_score)
        chosen, per_pat = [], {}
        for r in ordered:
            pid = r["patient_id"]
            if per_pat.get(pid, 0) >= m_per_patient:
                continue
            md = abs(r["padim_score"] - med) if med is not None else float("nan")
            ap = air_penalty(b, r["dense_frac_hu_gt_minus500"])
            r2 = dict(r)
            r2["selection_reason"] = ("median_dist=%.4f|air_pen=%.3f|roi=%.2f|sel=%.4f"
                                      % (md, ap, float(r["roi_0_0_patch_ratio"]), full_score(r)))
            chosen.append(r2)
            per_pat[pid] = per_pat.get(pid, 0) + 1
            if len(chosen) >= n_per_bin:
                break
        selected[b] = chosen
    return selected


def _percentiles_5_50_95(values):
    import numpy as np
    if values is None or len(values) == 0:
        return (None, None, None)
    a = np.asarray(values, dtype=float)
    return (float(np.percentile(a, 5)), float(np.percentile(a, 50)), float(np.percentile(a, 95)))


def _window_to_uint8(ct_patch, center=LUNG_WINDOW_CENTER, width=LUNG_WINDOW_WIDTH):
    import numpy as np
    lo, hi = center - width / 2.0, center + width / 2.0
    arr = np.clip(np.asarray(ct_patch, dtype=float), lo, hi)
    return ((arr - lo) / (hi - lo) * 255.0).astype("uint8")


def _mask_contour(mask):
    """4-이웃 erosion 기반 경계(numpy만). 내부가 빈 경계 mask 반환."""
    import numpy as np
    m = np.asarray(mask).astype(bool)
    er = m.copy()
    er[1:, :] &= m[:-1, :]
    er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]
    er[:, :-1] &= m[:, 1:]
    er[0, :] = False; er[-1, :] = False; er[:, 0] = False; er[:, -1] = False
    return m & ~er


def _crop_rgb_with_contour(ct_patch, mask_patch):
    """lung-window 회색 crop + v4 mask contour overlay RGB(uint8)."""
    import numpy as np
    g = _window_to_uint8(ct_patch)
    rgb = np.stack([g, g, g], axis=-1)
    cont = _mask_contour(mask_patch)
    rgb[cont] = np.asarray(CONTOUR_RGB, dtype="uint8")
    return rgb


# ----------------------------------------------------------------------------
# FULL 생성 (--run-full --confirm-generate; 본 단계 미실행)
# ----------------------------------------------------------------------------
def _generate_full_reference_bank(out_dir):
    import numpy as np

    done_path = os.path.join(out_dir, "DONE.json")
    stats_path = os.path.join(out_dir, "reference_stats_by_position_bin.csv")
    manifest_path = os.path.join(out_dir, "reference_crop_manifest.csv")
    errors_path = os.path.join(out_dir, "errors.csv")
    runtime_path = os.path.join(out_dir, "runtime_summary.json")
    crops_root = os.path.join(out_dir, "reference_crops")

    # 가드: DONE 또는 잔여 산출물(미완료 이전 실행) -> BLOCKED (덮어쓰기/삭제 금지)
    if os.path.exists(safe_path(done_path)):
        sys.stderr.write("[BLOCKED] full DONE.json 존재: %s\n  새 버전 경로(reference_bank_v2/) 사용.\n" % done_path)
        return EXIT_BLOCKED
    if os.path.isdir(safe_path(out_dir)):
        leftovers = [p for p in (stats_path, manifest_path, runtime_path, errors_path) if os.path.exists(p)]
        if os.path.isdir(crops_root) and any(os.scandir(safe_path(crops_root))):
            leftovers.append(crops_root)
        if leftovers:
            sys.stderr.write("[BLOCKED] 미완료 이전 산출물 존재: %s\n  삭제/덮어쓰기 금지 -> 새 버전 경로 사용.\n" % leftovers)
            return EXIT_BLOCKED

    started_at = datetime.now().isoformat(timespec="seconds")
    os.makedirs(safe_path(crops_root), exist_ok=True)

    targets, audit = resolve_full_targets()
    if audit.get("blocked"):
        sys.stderr.write("[BLOCKED] %s\n" % audit.get("block_reason"))
        return EXIT_BLOCKED
    if not targets:
        sys.stderr.write("[BLOCKED] full 대상 0명\n")
        return EXIT_BLOCKED

    errors = []
    # 통계용: bin -> metric -> [np.float32 chunk per patient]
    metrics = ["p90_hu", "p99_hu", "mean_hu",
               "dense_frac_hu_gt_minus500", "dense_frac_hu_gt_minus300", "roi_ratio", "padim_score"]
    bin_chunks = {b: {m: [] for m in metrics} for b in POSITION_BINS}
    bin_patients = {b: set() for b in POSITION_BINS}
    # crop 후보: bin -> patient -> [records] (top-K by provisional key)
    crop_pool_pp = {b: {} for b in POSITION_BINS}
    sid_paths = {}  # safe_id -> (ct_path, mask_path) (대표 crop PNG 재로드용)
    valid_total = 0
    skipped_total = 0

    for t in targets:
        sid, pid, grp = t["safe_id"], t["patient_id"], t["group"]
        ct_path, mask_path = resolve_ct_hu_path(sid), resolve_mask_path(sid)
        sid_paths[sid] = (ct_path, mask_path)
        if not is_file(ct_path) or not is_file(mask_path):
            errors.append({"safe_id": sid, "patient_id": pid, "stage": "load", "detail": "ct/mask npy missing"})
            continue
        ct = np.load(safe_path(ct_path), mmap_mode="r")
        mask = np.load(safe_path(mask_path), mmap_mode="r")
        if ct.shape != mask.shape:
            errors.append({"safe_id": sid, "patient_id": pid, "stage": "shape",
                           "detail": "ct%s!=mask%s" % (ct.shape, mask.shape)})
            del ct, mask
            continue
        Z, H, W = ct.shape
        local = {b: {m: [] for m in metrics} for b in POSITION_BINS}  # 환자 단위 누적

        for row in iter_score_rows(t["score_csv"]):
            pbin = (row.get("position_bin") or "").strip()
            if pbin not in POSITION_BINS:
                skipped_total += 1
                continue
            try:
                score = float(row.get("padim_score"))
                roi_ratio = float(row.get("roi_0_0_patch_ratio"))
                y0 = int(float(row.get("y0"))); x0 = int(float(row.get("x0")))
                y1 = int(float(row.get("y1"))); x1 = int(float(row.get("x1")))
                zidx = int(float(row.get("slice_index")))
            except (TypeError, ValueError):
                errors.append({"safe_id": sid, "patient_id": pid, "stage": "parse", "detail": "bad numeric"})
                skipped_total += 1
                continue
            if not _is_valid_score(score) or roi_ratio < STATS_ROI_MIN:
                skipped_total += 1
                continue
            if not (0 <= zidx < Z and 0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W):
                errors.append({"safe_id": sid, "patient_id": pid, "stage": "bbox",
                               "detail": "z=%d y=%d:%d x=%d:%d shape=%s" % (zidx, y0, y1, x0, x1, ct.shape)})
                skipped_total += 1
                continue
            st = compute_patch_hu_stats(np.asarray(ct[zidx, y0:y1, x0:x1]),
                                        np.asarray(mask[zidx, y0:y1, x0:x1]))
            if st is None:
                errors.append({"safe_id": sid, "patient_id": pid, "stage": "empty_roi",
                               "detail": "v4 mask empty z=%d y=%d:%d x=%d:%d" % (zidx, y0, y1, x0, x1)})
                skipped_total += 1
                continue
            valid_total += 1
            local[pbin]["p90_hu"].append(st["p90_hu"])
            local[pbin]["p99_hu"].append(st["p99_hu"])
            local[pbin]["mean_hu"].append(st["mean_hu"])
            local[pbin]["dense_frac_hu_gt_minus500"].append(st["dense_frac_hu_gt_minus500"])
            local[pbin]["dense_frac_hu_gt_minus300"].append(st["dense_frac_hu_gt_minus300"])
            local[pbin]["roi_ratio"].append(roi_ratio)
            local[pbin]["padim_score"].append(score)
            bin_patients[pbin].add(sid)
            # 대표 crop 후보 (roi>=0.70) : 환자/ bin 당 top-K provisional 보관
            if roi_ratio >= CROP_ROI_MIN:
                rec = {"safe_id": sid, "patient_id": pid, "group": grp, "position_bin": pbin,
                       "slice_index": zidx, "y0": y0, "x0": x0, "y1": y1, "x1": x1,
                       "padim_score": score, "roi_0_0_patch_ratio": roi_ratio,
                       "dense_frac_hu_gt_minus500": st["dense_frac_hu_gt_minus500"], "mean_hu": st["mean_hu"],
                       "_pkey": provisional_crop_key(pbin, st["dense_frac_hu_gt_minus500"], roi_ratio)}
                lst = crop_pool_pp[pbin].setdefault(pid, [])
                lst.append(rec)
                if len(lst) > KEEP_PER_PATIENT_PER_BIN:
                    lst.sort(key=lambda r: r["_pkey"])
                    del lst[KEEP_PER_PATIENT_PER_BIN:]

        # 환자 누적 -> 전역 chunk (float32) 후 해제 (메모리 bound)
        for b in POSITION_BINS:
            for m in metrics:
                if local[b][m]:
                    bin_chunks[b][m].append(np.asarray(local[b][m], dtype=np.float32))
        del ct, mask, local

    # ---- 통계 CSV ----
    bin_arrays = {b: {m: (np.concatenate(bin_chunks[b][m]) if bin_chunks[b][m] else np.array([], dtype=np.float32))
                      for m in metrics} for b in POSITION_BINS}
    with open(safe_path(stats_path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STATS_FIELDS)
        w.writeheader()
        for b in POSITION_BINS:
            arr = bin_arrays[b]
            n = int(arr["p90_hu"].size)
            p90 = _percentiles_5_50_95(arr["p90_hu"])
            p99 = _percentiles_5_50_95(arr["p99_hu"])
            mhu = _percentiles_5_50_95(arr["mean_hu"])
            d500 = _percentiles_5_50_95(arr["dense_frac_hu_gt_minus500"])
            d300 = _percentiles_5_50_95(arr["dense_frac_hu_gt_minus300"])
            roi_p50 = _percentiles_5_50_95(arr["roi_ratio"])[1] if n else None
            w.writerow({"position_bin": b, "valid_patch_count": n, "patient_count": len(bin_patients[b]),
                        "p90_hu_p5": p90[0], "p90_hu_p50": p90[1], "p90_hu_p95": p90[2],
                        "p99_hu_p5": p99[0], "p99_hu_p50": p99[1], "p99_hu_p95": p99[2],
                        "mean_hu_p5": mhu[0], "mean_hu_p50": mhu[1], "mean_hu_p95": mhu[2],
                        "dense_frac_hu_gt_minus500_p5": d500[0], "dense_frac_hu_gt_minus500_p50": d500[1], "dense_frac_hu_gt_minus500_p95": d500[2],
                        "dense_frac_hu_gt_minus300_p5": d300[0], "dense_frac_hu_gt_minus300_p50": d300[1], "dense_frac_hu_gt_minus300_p95": d300[2],
                        "roi_patch_ratio_p50": roi_p50, "low_sample_flag": bool(n < BIN_LOW_SAMPLE)})

    # ---- 대표 crop 선정 + PNG(contour) + manifest ----
    bin_medians = {b: (float(np.median(bin_arrays[b]["padim_score"])) if bin_arrays[b]["padim_score"].size else None)
                   for b in POSITION_BINS}
    crop_pool = {b: [r for recs in crop_pool_pp[b].values() for r in recs] for b in POSITION_BINS}
    selected = select_representative_full(crop_pool, bin_medians)

    created_pngs = []
    with open(safe_path(manifest_path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for b in POSITION_BINS:
            bin_dir = os.path.join(crops_root, b)
            for r in selected.get(b, []):
                png_rel = os.path.join("reference_crops", b,
                                       "%s__z%d__y%d_x%d.png" % (r["safe_id"], r["slice_index"], r["y0"], r["x0"]))
                png_abs = os.path.join(out_dir, png_rel)
                try:
                    os.makedirs(safe_path(bin_dir), exist_ok=True)
                    ct_path, mask_path = sid_paths[r["safe_id"]]
                    ct = np.load(safe_path(ct_path), mmap_mode="r")
                    mask = np.load(safe_path(mask_path), mmap_mode="r")
                    rgb = _crop_rgb_with_contour(np.asarray(ct[r["slice_index"], r["y0"]:r["y1"], r["x0"]:r["x1"]]),
                                                 np.asarray(mask[r["slice_index"], r["y0"]:r["y1"], r["x0"]:r["x1"]]))
                    del ct, mask
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    plt.imsave(safe_path(png_abs), rgb)
                    created_pngs.append(png_abs)
                except Exception as e:
                    errors.append({"safe_id": r["safe_id"], "patient_id": r["patient_id"], "stage": "png", "detail": "%s" % e})
                    png_rel = ""
                w.writerow({"safe_id": r["safe_id"], "patient_id": r["patient_id"], "group": r["group"],
                            "position_bin": b, "slice_index": r["slice_index"],
                            "y0": r["y0"], "x0": r["x0"], "y1": r["y1"], "x1": r["x1"],
                            "padim_score": r["padim_score"], "roi_0_0_patch_ratio": r["roi_0_0_patch_ratio"],
                            "dense_frac_hu_gt_minus500": r["dense_frac_hu_gt_minus500"], "mean_hu": r["mean_hu"],
                            "crop_png_path": png_rel, "selection_reason": r.get("selection_reason", "")})

    # ---- errors.csv ----
    with open(safe_path(errors_path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ERROR_FIELDS)
        w.writeheader()
        for e in errors:
            w.writerow(e)

    finished_at = datetime.now().isoformat(timespec="seconds")
    created_files = [stats_path, manifest_path, errors_path] + created_pngs + [runtime_path]
    summary = {
        "mode": "run-full", "model_tag": MODEL_TAG, "mask_tag": MASK_TAG, "frame": FRAME,
        "threshold_p95": THRESHOLD_P95, "bank_source": "v2_normal_score_72 (val+test, full)",
        "normal_patient_count": len(targets), "safe_ids": [t["safe_id"] for t in targets],
        "position_bins": list(POSITION_BINS),
        "valid_patch_count_total": valid_total, "skipped_patch_count_total": skipped_total,
        "error_count": len(errors), "created_files": created_files,
        "stage2_holdout_accessed": False, "lesion_accessed": False, "done": True,
        "started_at": started_at, "finished_at": finished_at,
        "representative_crop_filter": {"roi_min": CROP_ROI_MIN, "air_dense_target": AIR_DENSE_TARGET,
                                       "central_air_w": CENTRAL_AIR_W, "peripheral_air_w": PERIPHERAL_AIR_W,
                                       "low_roi_w": LOW_ROI_W, "keep_per_patient_per_bin": KEEP_PER_PATIENT_PER_BIN,
                                       "n_per_bin": CROP_N_PER_BIN, "m_per_patient": CROP_M_PER_PATIENT,
                                       "contour_overlay": True},
        "stats_filter": {"roi_min": STATS_ROI_MIN, "position_bins": list(POSITION_BINS),
                         "exclude_nan_inf_score": True, "require_v4_mask_nonempty": True,
                         "bin_low_sample": BIN_LOW_SAMPLE},
    }
    with open(safe_path(runtime_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(safe_path(done_path), "w", encoding="utf-8") as f:
        json.dump({"done": True, "error_count": len(errors),
                   "created_artifact_count": len(created_files) + 1,
                   "stage2_holdout_accessed": False, "lesion_accessed": False}, f, ensure_ascii=False, indent=2)

    print("[run-full] 완료. patients=%d valid=%d skipped=%d errors=%d crops=%d -> %s"
          % (len(targets), valid_total, skipped_total, len(errors), len(created_pngs), out_dir))
    return EXIT_OK


# ----------------------------------------------------------------------------
# 모드: dry-run / plan-full-only / selftest / run-full
# ----------------------------------------------------------------------------
def mode_dry_run():
    print("[MODE] --dry-run (입력 read-only 점검 + 출력 계획)")
    checks = []

    def chk(label, ok, detail=""):
        checks.append(bool(ok)); print("  [%s] %s %s" % ("OK" if ok else "MISS", label, detail))

    csvs = list_normal_score_csvs()
    chk("normal score dir", is_dir(NORMAL_SCORE_DIR), "(csv=%d)" % len(csvs))
    if csvs:
        header = read_score_header(csvs[0])
        need = ["safe_id", "group", "position_bin", "padim_score", "roi_0_0_patch_ratio",
                "slice_index", "y0", "x0", "y1", "x1"]
        miss = [c for c in need if c not in header]
        chk("score header 필수컬럼", not miss, ("missing=%s" % miss) if miss else "(all present)")
    chk("v4 normal mask dir", is_dir(V4_NORMAL_MASK_DIR))
    chk("threshold json", is_file(THRESHOLD_JSON), "(p95=%s)" % THRESHOLD_P95)
    chk("CT HU source root", is_dir(CT_HU_SOURCE_ROOT))
    full_done = os.path.exists(safe_path(os.path.join(FULL_OUT, "DONE.json")))
    chk("full DONE.json 부재", not full_done, "(존재시 BLOCKED)")

    print("\n[OUTPUT PLAN] full ->", FULL_OUT, "(미생성)")
    for a in PLANNED_ARTIFACTS:
        print("    ", os.path.join(FULL_OUT, a))
    print("\n[STATS FILTER] roi>=%.2f, bin 6, score 유효, v4 mask nonempty" % STATS_ROI_MIN)
    print("[CROP FILTER] roi>=%.2f, air_target=%.2f, central_w=%.1f periph_w=%.1f, N=%d/bin M=%d/patient, contour overlay"
          % (CROP_ROI_MIN, AIR_DENSE_TARGET, CENTRAL_AIR_W, PERIPHERAL_AIR_W, CROP_N_PER_BIN, CROP_M_PER_PATIENT))
    ok = all(checks)
    print("\n[DRY-RUN RESULT]", "OK" if ok else "NEEDS_FIX")
    return EXIT_OK if ok else EXIT_FAIL


def mode_plan_full_only():
    print("[MODE] --plan-full-only (정상 72명 계획만. npy 미로드)")
    targets, audit = resolve_full_targets()
    if audit.get("blocked"):
        print("  [BLOCKED]", audit.get("block_reason")); return EXIT_BLOCKED
    print("  score CSV 총:", audit["score_csv_count"], "| eligible(full 대상):", audit.get("full_target_count"))
    grp = {}
    for t in targets:
        grp[t["group"]] = grp.get(t["group"], 0) + 1
    print("  group 분포:", grp)
    bad = [t for t in targets if ("lesion" in t["group"]) or ("holdout" in t["group"])]
    print("  [HOLDOUT CHECK] lesion/holdout 대상 =", len(bad), "(must be 0)")
    print("  대상 샘플(처음 5):", [t["safe_id"] for t in targets[:5]])
    print("\n[OUTPUT PLAN] full ->", FULL_OUT, "(미생성)")
    if len(targets) == 0:
        return EXIT_FAIL
    return EXIT_OK if not bad else EXIT_BLOCKED


def mode_selftest():
    print("[MODE] --selftest (순수 로직 + 소스 정적 검토)")
    results = []

    def expect(name, cond):
        results.append(bool(cond)); print("  [%s] %s" % ("PASS" if cond else "FAIL", name))

    # forbidden guard
    g_ok = True
    for p in ("a/stage2_holdout/x", "b/holdout.csv", "c/lesion/y.npy"):
        try:
            safe_path(p); g_ok = False
        except RuntimeError:
            pass
    expect("forbidden guard blocks holdout/lesion", g_ok)
    try:
        safe_path("x/normal/normal004__abc/refined_roi.npy"); n_ok = True
    except RuntimeError:
        n_ok = False
    expect("forbidden guard allows normal", n_ok)

    expect("position_bin == 6", len(POSITION_BINS) == 6)
    expect("stats filter roi>=0.50", STATS_ROI_MIN == 0.50)
    expect("crop filter roi>=0.70", CROP_ROI_MIN == 0.70)
    expect("low_sample_flag 기준 100", BIN_LOW_SAMPLE == 100)
    expect("crop N=5/bin", CROP_N_PER_BIN == 5)
    expect("crop M=1/patient", CROP_M_PER_PATIENT == 1)

    # air_penalty: central > peripheral, dense>=target -> 0
    ap_c = air_penalty("upper_central", 0.0)
    ap_p = air_penalty("upper_peripheral", 0.0)
    expect("air_penalty central>peripheral", ap_c > ap_p > 0)
    expect("air_penalty dense>=target ->0", air_penalty("upper_central", 0.10) == 0.0)

    # representative selection: N/M, air 영향
    recs = []
    for i in range(8):
        recs.append({"safe_id": "p%d" % i, "patient_id": "p%d" % i, "group": "val",
                     "position_bin": "middle_central", "slice_index": 10, "y0": 0, "x0": 0, "y1": 32, "x1": 32,
                     "padim_score": 9.0, "roi_0_0_patch_ratio": 0.9, "dense_frac_hu_gt_minus500": 0.2, "mean_hu": -700})
    # 같은 환자 중복 2개(M=1 확인용)
    recs.append({"safe_id": "p0", "patient_id": "p0", "group": "val", "position_bin": "middle_central",
                 "slice_index": 11, "y0": 0, "x0": 0, "y1": 32, "x1": 32, "padim_score": 9.0,
                 "roi_0_0_patch_ratio": 0.95, "dense_frac_hu_gt_minus500": 0.3, "mean_hu": -650})
    # air-only 1개(후순위 확인): dense 0, central penalty 큼
    recs.append({"safe_id": "pAir", "patient_id": "pAir", "group": "val", "position_bin": "middle_central",
                 "slice_index": 12, "y0": 0, "x0": 0, "y1": 32, "x1": 32, "padim_score": 9.0,
                 "roi_0_0_patch_ratio": 0.99, "dense_frac_hu_gt_minus500": 0.0, "mean_hu": -900})
    sel = select_representative_full({"middle_central": recs}, {"middle_central": 9.0})["middle_central"]
    expect("N/bin cap == 5", len(sel) == 5)
    expect("M/patient == 1", all(sum(1 for r in sel if r["patient_id"] == pid) <= 1
                                 for pid in set(r["patient_id"] for r in sel)))
    expect("air-only patch 후순위(미선택)", all(r["patient_id"] != "pAir" for r in sel))
    expect("selection_reason 기록", all("selection_reason" in r for r in sel))

    # HU stats + dense 2종
    try:
        import numpy as np
        hu = np.array([[-1000, -400], [-200, 50]], float); roi = np.ones((2, 2), "uint8")
        st = compute_patch_hu_stats(hu, roi)
        ok = (st and abs(st["dense_frac_hu_gt_minus500"] - 0.75) < 1e-9
              and abs(st["dense_frac_hu_gt_minus300"] - 0.50) < 1e-9 and st["n_vox"] == 4)
        ok = ok and (compute_patch_hu_stats(hu, np.zeros_like(roi)) is None)
        expect("HU dense 2종 + empty->None", ok)
        # mask contour: 4x4 채움 -> 가장자리만 True, 내부 False
        m = np.ones((4, 4), bool)
        cont = _mask_contour(m)
        expect("mask contour 경계만(내부 비움)", bool(cont[0, 0]) and (not bool(cont[1, 1])) and bool(cont.any()))
        rgb = _crop_rgb_with_contour(np.full((4, 4), -700.0), m)
        expect("contour overlay RGB(H,W,3) uint8", rgb.shape == (4, 4, 3) and rgb.dtype.name == "uint8")
    except Exception as e:
        expect("numpy 계산", False); print("    detail:", e)

    # full targets = 전체(3명 고정 아님)
    src_resolve = inspect.getsource(resolve_full_targets)
    expect("resolve_full_targets 전체 반환(3 고정 아님)",
           ("[:3]" not in src_resolve) and ("SMOKE_PATIENT_IDS" not in src_resolve) and ("eligible" in src_resolve))
    expect("safe_id join 사용",
           ("resolve_mask_path(sid)" in inspect.getsource(collect_eligible_normals))
           and ("safe_id" in inspect.getsource(resolve_mask_path)))

    # 소스 정적: placeholder/실제연결/mmap/DONE 가드/run-full confirm
    src_run = inspect.getsource(mode_run_full)
    src_gen = inspect.getsource(_generate_full_reference_bank)
    expect("run-full -> 실제 생성함수 연결", "_generate_full_reference_bank(FULL_OUT)" in src_run)
    expect("run-full confirm 없으면 BLOCKED", "confirm_generate" in src_run and "EXIT_BLOCKED" in src_run)
    expect("run-full 단순 print 아님", "return _generate_full_reference_bank(FULL_OUT)" in src_run)
    expect("placeholder 없음", not any(b in src_gen or b in src_run for b in ("placeholder", "구현 자리", "호출되지 않")))
    expect("mmap_mode='r' 사용", 'mmap_mode="r"' in src_gen)
    expect("DONE 존재시 BLOCKED 로직", "DONE.json 존재" in src_gen and "EXIT_BLOCKED" in src_gen)
    expect("mask contour 함수 호출", "_crop_rgb_with_contour(" in src_gen)
    expect("전체 산출물 기록", all(a in src_gen for a in ("reference_stats_by_position_bin.csv",
            "reference_crop_manifest.csv", "runtime_summary.json", "errors.csv", "DONE.json")))

    n_pass = sum(1 for ok in results if ok)
    print("\n[SELFTEST] %d/%d PASS" % (n_pass, len(results)))
    return EXIT_OK if n_pass == len(results) else EXIT_FAIL


def mode_run_full(confirm_generate):
    if not confirm_generate:
        sys.stderr.write("[BLOCKED] --run-full 은 --confirm-generate 동반 + 사용자 승인 필요.\n")
        return EXIT_BLOCKED
    return _generate_full_reference_bank(FULL_OUT)


def build_parser():
    p = argparse.ArgumentParser(description="Explanation Card S1 정상 reference bank FULL 빌더 (가드 필수).")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plan-full-only", action="store_true")
    p.add_argument("--run-full", action="store_true")
    p.add_argument("--confirm-generate", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest:
        return mode_selftest()
    if args.dry_run:
        return mode_dry_run()
    if args.plan_full_only:
        return mode_plan_full_only()
    if args.run_full:
        return mode_run_full(args.confirm_generate)
    sys.stderr.write(
        "[BLOCKED] 가드 플래그가 필요합니다.\n"
        "  허용: --selftest | --dry-run | --plan-full-only\n"
        "  (--run-full 은 --confirm-generate + 승인 필요)\n")
    return EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main())
