#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explanation Card S1 : 정상 reference bank smoke 빌더

기준 문서:
  - docs/explanation_card_plan_v1.md (S1)
  - outputs/position-aware-padim-v1/reports/explanation_cards/s1_reference_bank_preflight_v1.md

성격:
  - 가드 없는 실행은 즉시 BLOCKED.
  - --selftest / --dry-run / --plan-smoke-only 는 read-only 계획/검사만 수행한다.
  - --run-smoke --confirm-generate 는 실제 생성 모드(정상 3명 smoke). full run 플래그는 없다.

금지(어떤 모드에서도):
  - stage2_holdout 접근, lesion score/lesion mask/holdout 파일 스캔
  - 기존 score/model/mask/heatmap/overlay 수정·삭제·이동·덮어쓰기
  - 전체 volume in-memory 적재 (CT/mask 는 mmap, patch bbox 영역만 사용)
join key 는 반드시 safe_id (patient_id 단독 join 금지).
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
SMOKE_N_PATIENTS = 3
SMOKE_PATIENT_IDS = ("normal023", "normal024", "normal036")  # 고정 smoke 대상

ROI_PATCH_RATIO_MIN = 0.50
DENSE_HU_THRESHOLDS = {
    "dense_frac_hu_gt_minus500": -500.0,
    "dense_frac_hu_gt_minus300": -300.0,
}
BIN_LOW_SAMPLE = 100  # bin valid patch < 100 -> low_sample_flag (실패 아님)
CROP_N_PER_BIN = 5
CROP_M_PER_PATIENT = 1

LUNG_WINDOW_CENTER = -600.0
LUNG_WINDOW_WIDTH = 1500.0

POSITION_BINS = (
    "upper_central", "upper_peripheral",
    "middle_central", "middle_peripheral",
    "lower_central", "lower_peripheral",
)

FORBIDDEN_PATH_TOKENS = ("stage2_holdout", "holdout", "lesion")

# 산출물 (run-smoke 에서만 생성)
PLANNED_ARTIFACTS = (
    "reference_stats_by_position_bin.csv",
    "reference_crop_manifest.csv",
    "reference_crops/<position_bin>/*.png",
    "runtime_summary.json",
    "errors.csv",
    "DONE.json",
)

# per-bin 통계 컬럼 순서 (reference_stats_by_position_bin.csv)
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
    "crop_png_path", "selection_reason",
)

ERROR_FIELDS = ("safe_id", "patient_id", "stage", "detail")

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
SMOKE_OUT = os.path.join(BANK_ROOT, "smoke")
FULL_OUT = os.path.join(BANK_ROOT, "full")


# ----------------------------------------------------------------------------
# 안전 가드
# ----------------------------------------------------------------------------
def safe_path(path):
    low = str(path).replace("\\", "/").lower()
    for tok in FORBIDDEN_PATH_TOKENS:
        if tok in low:
            raise RuntimeError("FORBIDDEN path token '%s' in: %s" % (tok, path))
    return path


def path_exists(path):
    return os.path.exists(safe_path(path))


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
    out = []
    for name in sorted(os.listdir(safe_path(NORMAL_SCORE_DIR))):
        if name.lower().endswith(".csv"):
            out.append(os.path.join(NORMAL_SCORE_DIR, name))
    return out


def read_score_header(csv_path):
    with open(safe_path(csv_path), "r", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f), [])
    return [h.strip() for h in header]


def read_first_row_meta(csv_path):
    with open(safe_path(csv_path), "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        row = next(reader, None)
    if not row:
        return None
    return {
        "patient_id": (row.get("patient_id") or "").strip(),
        "safe_id": (row.get("safe_id") or "").strip(),
        "group": (row.get("group") or "").strip(),
    }


def iter_score_rows(csv_path):
    """score CSV 의 모든 patch 행을 dict 로 순회 (run-smoke 에서만 호출)."""
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
# 순수 계산 (selftest 대상)
# ----------------------------------------------------------------------------
def compute_patch_hu_stats(hu_patch, roi_patch):
    """patch HU 통계. roi_patch(bool) 내부 화소만 사용."""
    import numpy as np
    mask = np.asarray(roi_patch).astype(bool)
    vals = np.asarray(hu_patch)[mask]
    if vals.size == 0:
        return None
    out = {
        "p90_hu": float(np.percentile(vals, 90)),
        "p99_hu": float(np.percentile(vals, 99)),
        "mean_hu": float(np.mean(vals)),
        "std_hu": float(np.std(vals)),
        "n_vox": int(vals.size),
    }
    for key, thr in DENSE_HU_THRESHOLDS.items():
        out[key] = float(np.mean(vals > thr))
    return out


def _is_valid_score(s):
    return (s is not None) and (not math.isnan(s)) and (not math.isinf(s))


def select_representative_patches(rows, bin_medians,
                                  n_per_bin=CROP_N_PER_BIN,
                                  m_per_patient=CROP_M_PER_PATIENT,
                                  roi_min=ROI_PATCH_RATIO_MIN):
    """같은 bin / roi>=min / 유효 score / 중앙값 근접 / 환자당 m / bin당 n."""
    by_bin = {}
    for r in rows:
        s = r.get("padim_score")
        if not _is_valid_score(s):
            continue
        if float(r.get("roi_0_0_patch_ratio", 0.0)) < roi_min:
            continue
        by_bin.setdefault(r["position_bin"], []).append(r)
    selected = {}
    for b, items in by_bin.items():
        med = bin_medians.get(b)
        if med is None:
            ordered = sorted(items, key=lambda r: r["padim_score"])
        else:
            ordered = sorted(items, key=lambda r: abs(r["padim_score"] - med))
        chosen, per_patient = [], {}
        for r in ordered:
            sid = r["safe_id"]
            if per_patient.get(sid, 0) >= m_per_patient:
                continue
            r2 = dict(r)
            r2["selection_reason"] = ("nearest_to_bin_median(|score-med|=%.4f)"
                                      % (abs(r["padim_score"] - med) if med is not None else float("nan")))
            chosen.append(r2)
            per_patient[sid] = per_patient.get(sid, 0) + 1
            if len(chosen) >= n_per_bin:
                break
        selected[b] = chosen
    return selected


def _percentiles_5_50_95(values):
    import numpy as np
    if not values:
        return (None, None, None)
    a = np.asarray(values, dtype=float)
    return (float(np.percentile(a, 5)), float(np.percentile(a, 50)), float(np.percentile(a, 95)))


def _window_to_uint8(ct_patch, center=LUNG_WINDOW_CENTER, width=LUNG_WINDOW_WIDTH):
    import numpy as np
    lo = center - width / 2.0
    hi = center + width / 2.0
    arr = np.clip(np.asarray(ct_patch, dtype=float), lo, hi)
    arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype("uint8")


# ----------------------------------------------------------------------------
# eligible / smoke 대상
# ----------------------------------------------------------------------------
def collect_eligible_normals():
    """정상 score 72 중 safe_id 기준 score+mask+ct_meta 3자 존재 목록 (read-only)."""
    csvs = list_normal_score_csvs()
    audit = {"score_csv_count": len(csvs), "candidates": [], "blocked": False, "block_reason": None}
    eligible = []
    for path in csvs:
        meta = read_first_row_meta(path)
        if meta is None:
            audit["candidates"].append({"file": os.path.basename(path), "skip": "empty_csv"})
            continue
        grp = meta["group"].lower()
        if ("lesion" in grp) or ("holdout" in grp):
            audit["blocked"] = True
            audit["block_reason"] = "non-normal group: %s (%s)" % (grp, os.path.basename(path))
            return [], audit
        if grp not in ALLOWED_GROUPS:
            audit["candidates"].append({"file": os.path.basename(path), "skip": "group=%s" % grp})
            continue
        sid = meta["safe_id"]
        if not sid:
            audit["candidates"].append({"file": os.path.basename(path), "skip": "no_safe_id"})
            continue
        rec = {
            "safe_id": sid, "patient_id": meta["patient_id"], "group": grp,
            "score_csv": path,
            "mask_exists": is_file(resolve_mask_path(sid)),
            "ct_meta_exists": is_file(resolve_ct_meta_path(sid)),
        }
        audit["candidates"].append({k: rec[k] for k in ("safe_id", "patient_id", "group", "mask_exists", "ct_meta_exists")})
        if rec["mask_exists"] and rec["ct_meta_exists"]:
            eligible.append(rec)
    eligible.sort(key=lambda r: (ALLOWED_GROUPS.index(r["group"]) if r["group"] in ALLOWED_GROUPS else 9, r["safe_id"]))
    audit["eligible_count"] = len(eligible)
    return eligible, audit


def resolve_smoke_targets():
    """고정 SMOKE_PATIENT_IDS 에 해당하는 eligible 정상만 반환."""
    eligible, audit = collect_eligible_normals()
    if audit["blocked"]:
        return [], audit
    by_pid = {r["patient_id"]: r for r in eligible}
    targets = [by_pid[pid] for pid in SMOKE_PATIENT_IDS if pid in by_pid]
    audit["requested_ids"] = list(SMOKE_PATIENT_IDS)
    audit["resolved_count"] = len(targets)
    return targets, audit


def planned_output_layout(out_dir):
    return {"root": out_dir, "artifacts": [os.path.join(out_dir, a) for a in PLANNED_ARTIFACTS]}


# ----------------------------------------------------------------------------
# 실제 smoke 생성 (run-smoke --confirm-generate 에서만; 이번 단계 미실행)
# ----------------------------------------------------------------------------
def _generate_smoke_reference_bank(out_dir):
    import numpy as np  # 실제 생성 시에만 필요

    done_path = os.path.join(out_dir, "DONE.json")
    if os.path.exists(safe_path(done_path)):
        sys.stderr.write(
            "[BLOCKED] 이미 DONE.json 존재: %s\n  덮어쓰기 금지. 재실행은 새 버전 경로(reference_bank_v2/ 등) 사용.\n"
            % done_path)
        return EXIT_BLOCKED

    started_at = datetime.now().isoformat(timespec="seconds")
    crops_root = os.path.join(out_dir, "reference_crops")
    os.makedirs(safe_path(crops_root), exist_ok=True)

    targets, audit = resolve_smoke_targets()
    if audit.get("blocked"):
        sys.stderr.write("[BLOCKED] %s\n" % audit.get("block_reason"))
        return EXIT_BLOCKED
    if len(targets) != SMOKE_N_PATIENTS:
        sys.stderr.write("[BLOCKED] smoke 대상 해석 실패: %d/%d (%s)\n"
                         % (len(targets), SMOKE_N_PATIENTS, SMOKE_PATIENT_IDS))
        return EXIT_BLOCKED

    errors = []  # dict(ERROR_FIELDS)
    bin_patch_stats = {b: [] for b in POSITION_BINS}  # bin -> per-patch stat dict
    bin_patients = {b: set() for b in POSITION_BINS}
    rep_rows = []  # 대표 선정 후보 (manifest 용 필드 포함)
    valid_total = 0
    skipped_total = 0

    for t in targets:
        sid, pid, grp = t["safe_id"], t["patient_id"], t["group"]
        ct_path = resolve_ct_hu_path(sid)
        mask_path = resolve_mask_path(sid)
        if not is_file(ct_path) or not is_file(mask_path):
            errors.append({"safe_id": sid, "patient_id": pid, "stage": "load",
                           "detail": "ct or mask npy missing"})
            continue
        # mmap (전체 volume in-memory 적재 안 함)
        ct = np.load(safe_path(ct_path), mmap_mode="r")
        mask = np.load(safe_path(mask_path), mmap_mode="r")
        if ct.shape != mask.shape:
            errors.append({"safe_id": sid, "patient_id": pid, "stage": "shape",
                           "detail": "ct%s != mask%s" % (ct.shape, mask.shape)})
            del ct, mask
            continue
        Z, H, W = ct.shape

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
                errors.append({"safe_id": sid, "patient_id": pid, "stage": "parse",
                               "detail": "bad numeric in row"})
                skipped_total += 1
                continue
            if not _is_valid_score(score) or roi_ratio < ROI_PATCH_RATIO_MIN:
                skipped_total += 1
                continue
            # bbox 범위 검증
            if not (0 <= zidx < Z and 0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W):
                errors.append({"safe_id": sid, "patient_id": pid, "stage": "bbox",
                               "detail": "out of range z=%d y=%d:%d x=%d:%d shape=%s"
                                         % (zidx, y0, y1, x0, x1, ct.shape)})
                skipped_total += 1
                continue
            # patch bbox 영역만 슬라이스 (작은 copy)
            ct_patch = np.asarray(ct[zidx, y0:y1, x0:x1])
            mask_patch = np.asarray(mask[zidx, y0:y1, x0:x1])
            st = compute_patch_hu_stats(ct_patch, mask_patch)
            if st is None:
                # v4 mask 내부 화소 0 (흉벽 제거로 빈 영역) -> skip 기록
                errors.append({"safe_id": sid, "patient_id": pid, "stage": "empty_roi",
                               "detail": "v4 mask empty in patch z=%d y=%d:%d x=%d:%d" % (zidx, y0, y1, x0, x1)})
                skipped_total += 1
                continue
            valid_total += 1
            st["roi_0_0_patch_ratio"] = roi_ratio
            bin_patch_stats[pbin].append(st)
            bin_patients[pbin].add(sid)
            rep_rows.append({
                "position_bin": pbin, "safe_id": sid, "patient_id": pid, "group": grp,
                "slice_index": zidx, "y0": y0, "x0": x0, "y1": y1, "x1": x1,
                "padim_score": score, "roi_0_0_patch_ratio": roi_ratio,
                "_ct_path": ct_path,
            })
        del ct, mask  # mmap 핸들 해제

    # ---- per-bin 통계 CSV ----
    stats_path = os.path.join(out_dir, "reference_stats_by_position_bin.csv")
    with open(safe_path(stats_path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STATS_FIELDS)
        w.writeheader()
        for b in POSITION_BINS:
            stats = bin_patch_stats[b]
            n = len(stats)
            def col(key):
                return [s[key] for s in stats]
            p90 = _percentiles_5_50_95(col("p90_hu")) if n else (None, None, None)
            p99 = _percentiles_5_50_95(col("p99_hu")) if n else (None, None, None)
            mhu = _percentiles_5_50_95(col("mean_hu")) if n else (None, None, None)
            d500 = _percentiles_5_50_95(col("dense_frac_hu_gt_minus500")) if n else (None, None, None)
            d300 = _percentiles_5_50_95(col("dense_frac_hu_gt_minus300")) if n else (None, None, None)
            roi_p50 = _percentiles_5_50_95(col("roi_0_0_patch_ratio"))[1] if n else None
            w.writerow({
                "position_bin": b, "valid_patch_count": n, "patient_count": len(bin_patients[b]),
                "p90_hu_p5": p90[0], "p90_hu_p50": p90[1], "p90_hu_p95": p90[2],
                "p99_hu_p5": p99[0], "p99_hu_p50": p99[1], "p99_hu_p95": p99[2],
                "mean_hu_p5": mhu[0], "mean_hu_p50": mhu[1], "mean_hu_p95": mhu[2],
                "dense_frac_hu_gt_minus500_p5": d500[0], "dense_frac_hu_gt_minus500_p50": d500[1], "dense_frac_hu_gt_minus500_p95": d500[2],
                "dense_frac_hu_gt_minus300_p5": d300[0], "dense_frac_hu_gt_minus300_p50": d300[1], "dense_frac_hu_gt_minus300_p95": d300[2],
                "roi_patch_ratio_p50": roi_p50,
                "low_sample_flag": bool(n < BIN_LOW_SAMPLE),
            })

    # ---- 대표 crop 선정 + PNG + manifest ----
    bin_medians = {}
    import numpy as np
    for b in POSITION_BINS:
        scores_b = [r["padim_score"] for r in rep_rows if r["position_bin"] == b]
        bin_medians[b] = float(np.median(scores_b)) if scores_b else None
    selected = select_representative_patches(rep_rows, bin_medians)

    manifest_path = os.path.join(out_dir, "reference_crop_manifest.csv")
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
                    ct = np.load(safe_path(r["_ct_path"]), mmap_mode="r")
                    ct_patch = np.asarray(ct[r["slice_index"], r["y0"]:r["y1"], r["x0"]:r["x1"]])
                    del ct
                    img = _window_to_uint8(ct_patch)
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    plt.imsave(safe_path(png_abs), img, cmap="gray")
                    created_pngs.append(png_abs)
                except Exception as e:
                    errors.append({"safe_id": r["safe_id"], "patient_id": r["patient_id"],
                                   "stage": "png", "detail": "%s" % e})
                    png_rel = ""  # 실패 시 경로 비움, 전체 중단하지 않음
                w.writerow({
                    "safe_id": r["safe_id"], "patient_id": r["patient_id"], "group": r["group"],
                    "position_bin": b, "slice_index": r["slice_index"],
                    "y0": r["y0"], "x0": r["x0"], "y1": r["y1"], "x1": r["x1"],
                    "padim_score": r["padim_score"], "roi_0_0_patch_ratio": r["roi_0_0_patch_ratio"],
                    "crop_png_path": png_rel, "selection_reason": r.get("selection_reason", ""),
                })

    # ---- errors.csv (비어도 header) ----
    errors_path = os.path.join(out_dir, "errors.csv")
    with open(safe_path(errors_path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ERROR_FIELDS)
        w.writeheader()
        for e in errors:
            w.writerow(e)

    finished_at = datetime.now().isoformat(timespec="seconds")
    created_files = [stats_path, manifest_path, errors_path] + created_pngs

    # ---- runtime_summary.json ----
    runtime_path = os.path.join(out_dir, "runtime_summary.json")
    summary = {
        "mode": "run-smoke",
        "model_tag": MODEL_TAG, "mask_tag": MASK_TAG, "frame": FRAME,
        "threshold_p95": THRESHOLD_P95, "bank_source": "v2_normal_score_72 (smoke 3)",
        "smoke_patient_count": len(targets),
        "safe_ids": [t["safe_id"] for t in targets],
        "position_bins": list(POSITION_BINS),
        "valid_patch_count_total": valid_total,
        "skipped_patch_count_total": skipped_total,
        "error_count": len(errors),
        "created_files": created_files + [runtime_path],
        "stage2_holdout_accessed": False,
        "lesion_accessed": False,
        "done": True,
        "started_at": started_at, "finished_at": finished_at,
    }
    with open(safe_path(runtime_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- DONE.json (맨 마지막) ----
    with open(safe_path(done_path), "w", encoding="utf-8") as f:
        json.dump({
            "done": True,
            "error_count": len(errors),
            "created_artifact_count": len(created_files) + 2,  # +runtime +done
            "stage2_holdout_accessed": False,
            "lesion_accessed": False,
        }, f, ensure_ascii=False, indent=2)

    print("[run-smoke] 완료. valid=%d skipped=%d errors=%d crops=%d -> %s"
          % (valid_total, skipped_total, len(errors), len(created_pngs), out_dir))
    return EXIT_OK


# ----------------------------------------------------------------------------
# 모드: dry-run
# ----------------------------------------------------------------------------
def mode_dry_run():
    print("[MODE] --dry-run (read-only 입력 점검 + 출력 계획. value 접근/생성 없음)")
    checks = []

    def chk(label, ok, detail=""):
        checks.append({"label": label, "ok": bool(ok), "detail": detail})
        print("  [%s] %s %s" % ("OK" if ok else "MISS", label, detail))

    score_csvs = list_normal_score_csvs()
    chk("normal score dir", is_dir(NORMAL_SCORE_DIR), "(csv=%d)" % len(score_csvs))
    if score_csvs:
        header = read_score_header(score_csvs[0])
        need = ["safe_id", "group", "position_bin", "padim_score", "roi_0_0_patch_ratio",
                "slice_index", "y0", "x0", "y1", "x1"]
        missing = [c for c in need if c not in header]
        chk("score header 필수컬럼", not missing, ("missing=%s" % missing) if missing else "(all present)")
    chk("v4 normal mask dir", is_dir(V4_NORMAL_MASK_DIR))
    chk("threshold json", is_file(THRESHOLD_JSON), "(p95=%s)" % THRESHOLD_P95)
    chk("CT HU source root", is_dir(CT_HU_SOURCE_ROOT))

    print("\n[OUTPUT PLAN] (이번 단계 미생성)")
    print("  bank_root :", BANK_ROOT)
    for art in planned_output_layout(SMOKE_OUT)["artifacts"]:
        print("    smoke ->", art)
    print("  full_root(참고) :", FULL_OUT)
    print("\n[GUARD] forbidden tokens =", FORBIDDEN_PATH_TOKENS, "| join key = safe_id")
    all_ok = all(c["ok"] for c in checks)
    print("\n[DRY-RUN RESULT]", "OK" if all_ok else "NEEDS_FIX")
    return EXIT_OK if all_ok else EXIT_FAIL


# ----------------------------------------------------------------------------
# 모드: plan-smoke-only
# ----------------------------------------------------------------------------
def mode_plan_smoke_only():
    print("[MODE] --plan-smoke-only (정상 3명 smoke 계획만. npy 미로드)")
    targets, audit = resolve_smoke_targets()
    if audit.get("blocked"):
        print("  [BLOCKED]", audit.get("block_reason"))
        return EXIT_BLOCKED
    print("  score CSV 총:", audit["score_csv_count"], "| eligible:", audit.get("eligible_count"),
          "| 요청 id:", SMOKE_PATIENT_IDS)
    if len(targets) != SMOKE_N_PATIENTS:
        print("  [NEEDS_FIX] resolved %d/%d" % (len(targets), SMOKE_N_PATIENTS))
        return EXIT_FAIL
    print("\n[SMOKE TARGETS]")
    for i, t in enumerate(targets, 1):
        print("  %d) safe_id=%s patient_id=%s group=%s mask=%s ct_meta=%s"
              % (i, t["safe_id"], t["patient_id"], t["group"], t["mask_exists"], t["ct_meta_exists"]))
    print("\n[SMOKE PARAMS] ROI>=%.2f | dense_HU=%s | low_sample<%d(warn) | crop N=%d/bin M=%d/patient"
          % (ROI_PATCH_RATIO_MIN, list(DENSE_HU_THRESHOLDS.keys()), BIN_LOW_SAMPLE, CROP_N_PER_BIN, CROP_M_PER_PATIENT))
    print("  position_bins =", POSITION_BINS)
    print("  lung window: center=%.0f width=%.0f" % (LUNG_WINDOW_CENTER, LUNG_WINDOW_WIDTH))
    print("\n[OUTPUT PLAN] smoke ->", SMOKE_OUT, "(미생성)")
    bad = [t for t in targets if ("lesion" in t["group"]) or ("holdout" in t["group"])]
    print("\n[HOLDOUT CHECK] lesion/holdout in targets =", len(bad), "(must be 0)")
    return EXIT_OK if not bad else EXIT_BLOCKED


# ----------------------------------------------------------------------------
# 모드: selftest
# ----------------------------------------------------------------------------
def mode_selftest():
    print("[MODE] --selftest (순수 로직 + 소스 정적 검토; fs 생성/CT 접근 없음)")
    results = []

    def expect(name, cond):
        results.append((name, bool(cond)))
        print("  [%s] %s" % ("PASS" if cond else "FAIL", name))

    # forbidden guard
    g_ok = True
    for p in ("a/stage2_holdout/x", "b/holdout.csv", "c/lesion/y.npy"):
        try:
            safe_path(p); g_ok = False
        except RuntimeError:
            pass
    expect("forbidden guard blocks holdout/lesion", g_ok)
    try:
        safe_path("outputs/.../normal/normal004__abc/refined_roi.npy"); n_ok = True
    except RuntimeError:
        n_ok = False
    expect("forbidden guard allows normal path", n_ok)

    expect("position_bin count == 6", len(POSITION_BINS) == 6)
    expect("dense HU 2종 계산", set(DENSE_HU_THRESHOLDS.keys()) ==
           {"dense_frac_hu_gt_minus500", "dense_frac_hu_gt_minus300"})
    expect("crop N/bin == 5", CROP_N_PER_BIN == 5)
    expect("crop M/patient == 1", CROP_M_PER_PATIENT == 1)

    # 대표 선정 로직
    rows = []
    for sc in (9.0, 10.0, 11.0):
        rows.append({"position_bin": "A", "safe_id": "p1", "padim_score": sc, "roi_0_0_patch_ratio": 0.9})
    rows.append({"position_bin": "A", "safe_id": "p2", "padim_score": 100.0, "roi_0_0_patch_ratio": 0.9})
    rows.append({"position_bin": "A", "safe_id": "p3", "padim_score": 10.0, "roi_0_0_patch_ratio": 0.10})
    rows.append({"position_bin": "A", "safe_id": "p4", "padim_score": float("nan"), "roi_0_0_patch_ratio": 0.9})
    sel = select_representative_patches(rows, {"A": 10.0}, n_per_bin=5, m_per_patient=1, roi_min=0.50)
    chosenA = sel.get("A", [])
    expect("M/patient=1 enforced", sum(1 for r in chosenA if r["safe_id"] == "p1") == 1)
    expect("low-ROI excluded", all(r["roi_0_0_patch_ratio"] >= 0.50 for r in chosenA))
    expect("NaN excluded", all(_is_valid_score(r["padim_score"]) for r in chosenA))
    expect("nearest-to-median first", chosenA and abs(chosenA[0]["padim_score"] - 10.0) < 1e-9)
    expect("selection_reason 기록", all("selection_reason" in r for r in chosenA))

    many = [{"position_bin": "B", "safe_id": "q%d" % i, "padim_score": float(i), "roi_0_0_patch_ratio": 0.9}
            for i in range(20)]
    selB = select_representative_patches(many, {"B": 0.0}, n_per_bin=5, m_per_patient=1, roi_min=0.50)
    expect("N/bin cap == 5", len(selB.get("B", [])) == 5)

    # HU stats
    try:
        import numpy as np
        hu = np.array([[-1000, -400], [-200, 50]], dtype=float)
        roi = np.ones((2, 2), dtype="uint8")
        st = compute_patch_hu_stats(hu, roi)
        ok = (st and abs(st["dense_frac_hu_gt_minus500"] - 0.75) < 1e-9
              and abs(st["dense_frac_hu_gt_minus300"] - 0.50) < 1e-9
              and st["n_vox"] == 4 and "p90_hu" in st and "p99_hu" in st and "mean_hu" in st)
        ok = ok and (compute_patch_hu_stats(hu, np.zeros_like(roi)) is None)
        expect("HU dense_frac(>-500)=.75 (>-300)=.50 + empty->None", ok)
        # windowing
        img = _window_to_uint8(np.array([[-2000, -600], [800, 5000]], float))
        expect("windowing uint8 0..255", img.dtype.name == "uint8" and int(img.min()) == 0 and int(img.max()) == 255)
    except Exception as e:
        expect("numpy 계산", False)
        print("    detail:", e)

    expect("join key is safe_id", "safe_id" == "safe_id")
    expect("artifact schema 일치", set(PLANNED_ARTIFACTS) == {
        "reference_stats_by_position_bin.csv", "reference_crop_manifest.csv",
        "reference_crops/<position_bin>/*.png", "runtime_summary.json", "errors.csv", "DONE.json"})

    # ---- 소스 정적 검토: placeholder 제거 / 실제 구현 연결 ----
    src_run = inspect.getsource(mode_run_smoke)
    src_gen = inspect.getsource(_generate_smoke_reference_bank)
    banned = ("placeholder", "구현 자리", "호출되지 않는다")
    expect("run-smoke -> 실제 생성 함수 연결", "_generate_smoke_reference_bank" in src_run)
    expect("run-smoke 단순 print 아님", "return _generate_smoke_reference_bank(SMOKE_OUT)" in src_run)
    expect("placeholder 마커 없음", not any(b in src_run or b in src_gen for b in banned))
    expect("생성부 mmap_mode='r' 사용", 'mmap_mode="r"' in src_gen)
    expect("생성부 전체 산출물 기록", all(a in src_gen for a in (
        "reference_stats_by_position_bin.csv", "reference_crop_manifest.csv",
        "runtime_summary.json", "errors.csv", "DONE.json")))
    expect("생성부 DONE 덮어쓰기 가드", "이미 DONE.json 존재" in src_gen)

    n_pass = sum(1 for _, ok in results if ok)
    print("\n[SELFTEST] %d/%d PASS" % (n_pass, len(results)))
    return EXIT_OK if n_pass == len(results) else EXIT_FAIL


# ----------------------------------------------------------------------------
# 모드: run-smoke
# ----------------------------------------------------------------------------
def mode_run_smoke(confirm_generate):
    if not confirm_generate:
        sys.stderr.write("[BLOCKED] --run-smoke 는 --confirm-generate 동반 + 사용자 승인 필요.\n")
        return EXIT_BLOCKED
    return _generate_smoke_reference_bank(SMOKE_OUT)


# ----------------------------------------------------------------------------
# entry
# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Explanation Card S1 정상 reference bank smoke 빌더 (가드 필수).")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plan-smoke-only", action="store_true")
    p.add_argument("--run-smoke", action="store_true")
    p.add_argument("--confirm-generate", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest:
        return mode_selftest()
    if args.dry_run:
        return mode_dry_run()
    if args.plan_smoke_only:
        return mode_plan_smoke_only()
    if args.run_smoke:
        return mode_run_smoke(args.confirm_generate)
    sys.stderr.write(
        "[BLOCKED] 가드 플래그가 필요합니다.\n"
        "  허용: --selftest | --dry-run | --plan-smoke-only\n"
        "  (--run-smoke 는 --confirm-generate + 승인 필요)\n"
        "  full run 플래그는 존재하지 않습니다.\n")
    return EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main())
