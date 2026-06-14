"""
p_b_s2h_stage2_holdout_scoring.py

목적: stage2_holdout 154명에 대해 EfficientNet-B0 PaDiM scoring.
     p_b11_lesion_stage1_dev_scoring.py 로직 그대로, 대상 환자만 stage2_holdout으로 교체.

고정 조건:
  - distribution npz: p_b7 학습 결과 (변경 금지)
  - v4_20 ROI ratio threshold: 0.5
  - threshold JSON: p_b9 read-only (재계산 금지)
  - stage2_holdout eval-only (method tuning 금지)
  - resume: score CSV 존재 시 skip

금지:
  - metrics / AUROC·AUPRC / threshold 재계산
  - stage1_dev score CSV 수정
  - model 재학습
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


# ── fast scoring (cov_inv 미리 캐시 + 배치 Mahalanobis) ─────────────────────
def build_cov_inv_cache(model):
    """모든 분포의 cov_inv를 한 번만 계산해서 캐시."""
    cache = {}
    all_keys = list(model.stats.keys())
    for key in all_keys:
        s = model.stats[key]
        if s.get("count", 0) < 2:
            continue
        cov = s["cov"]
        try:
            cache[key] = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cache[key] = np.linalg.pinv(cov)
    return cache


def resolve_cov_inv(pb, cov_inv_cache, model):
    """position_bin → fallback 순서로 cov_inv 반환."""
    for key in [pb,
                model._position_bin_to_z_level(pb),
                model.GLOBAL_KEY]:
        if key in cov_inv_cache:
            return cov_inv_cache[key], model.stats[key]["mean"]
    raise RuntimeError(f"cov_inv 없음: {pb}")


def fast_score_patient(model, feat, patient_data, cov_inv_cache):
    """
    score_patient 대체: cov_inv 캐시 + 슬라이스별 배치 Mahalanobis.
    반환값 형식은 score_patient와 동일 (patch_df + padim_score 컬럼).
    """
    import pandas as pd
    from position_aware_padim.preprocessing import preprocess_ct_slice

    indices = model.selected_feature_indices
    ct_hu   = patient_data["ct_hu"]
    patch_df = patient_data["patch_df"].reset_index(drop=True)
    scores  = np.full(len(patch_df), np.nan)

    for z_value, group in patch_df.groupby("local_z"):
        z = int(z_value)
        if z < 0 or z >= ct_hu.shape[0]:
            continue

        pos_list = list(group.index)
        patch_coords = [(int(r.y0), int(r.x0), int(r.y1), int(r.x1))
                        for r in group.itertuples(index=False)]
        pbs = [str(r.position_bin) for r in group.itertuples(index=False)]

        slice_2d = np.asarray(ct_hu[z], dtype=np.float32)
        preprocessed = preprocess_ct_slice(slice_2d)
        feats_full = feat.extract_patch_features(preprocessed, patch_coords)  # (M, 448)
        feats = feats_full[:, indices].astype(np.float64)                      # (M, 100)
        del feats_full, slice_2d, preprocessed

        # position_bin별 그룹화 → 배치 Mahalanobis
        from collections import defaultdict
        pb_groups = defaultdict(list)
        for local_i, pb in enumerate(pbs):
            pb_groups[pb].append((local_i, pos_list[local_i]))

        for pb, items in pb_groups.items():
            try:
                cov_inv, mean = resolve_cov_inv(pb, cov_inv_cache, model)
            except RuntimeError:
                continue
            local_idxs = [x[0] for x in items]
            df_idxs    = [x[1] for x in items]
            diff = feats[local_idxs] - mean          # (K, 100)
            # batched: dist_sq[i] = diff[i] @ cov_inv @ diff[i]
            dist_sq = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
            scores[df_idxs] = np.sqrt(np.maximum(0.0, dist_sq))

        del feats

    result = patient_data["patch_df"].copy()
    result["padim_score"] = scores
    return result

PROJ_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT  = Path(__file__).resolve().parents[1]
SRC_DIR   = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

BACKBONE            = "efficientnet_b0"
RAW_FEATURE_DIM     = 144
REDUCED_FEATURE_DIM = 100
MASK_TYPE           = "roi_0_0"
SCRIPT_NAME         = "p_b_s2h_stage2_holdout_scoring.py"
EXPECTED_N          = 154
EXPECTED_STAGE      = "stage2_holdout"
EXPECTED_GROUPS     = {"NSCLC": 124, "MSD_Lung": 30}
JOIN_KEY            = "patient_id"

V4_20_PATCH_RATIO_THRESHOLD = 0.5
V4_20_LESION_ROOT = PROJ_ROOT / "outputs" / "mip-postprocess-research-v1" / "masks" / "refined_roi_v4_20_modeB_all_v1" / "lesion"

# P-B9 고정 threshold (read-only, 재계산 금지)
EXPECTED_P95 = 13.231265
EXPECTED_P99 = 15.472385
THRESH_TOL = 1e-4

MODEL_NPZ        = EXP_ROOT / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
SELECTED_INDICES = EXP_ROOT / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
THRESH_JSON      = EXP_ROOT / "outputs" / "evaluation" / "normal_val_thresholds" / "normal_val_threshold.json"

LESION_SPLIT    = PROJ_ROOT / "outputs" / "second-stage-lesion-refiner-v1" / "splits" / "lesion_stage_split_v1_balanced.csv"
LESION_ROOT     = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1")
LESION_MANIFEST = LESION_ROOT / "manifests" / "patient_manifest.csv"
LESION_VOLUMES  = LESION_ROOT / "volumes_npy"
GT_MASK_FILE    = "lesion_mask_roi_0_0.npy"

SCORE_DIR  = EXP_ROOT / "outputs" / "scores" / "stage2_holdout_by_patient"
EVAL_DIR   = EXP_ROOT / "outputs" / "evaluation" / "stage2_holdout_scoring"
REPORT_DIR = EXP_ROOT / "outputs" / "reports" / "stage2_holdout"
SCORING_SUMMARY_JSON = EVAL_DIR / "stage2_holdout_scoring_summary.json"
SCORING_SUMMARY_CSV  = EVAL_DIR / "stage2_holdout_scoring_summary.csv"
REPORT_MD   = REPORT_DIR / "p_b_s2h_stage2_holdout_scoring.md"
REPORT_JSON = REPORT_DIR / "p_b_s2h_stage2_holdout_scoring.json"
RUNTIME_CSV = REPORT_DIR / "p_b_s2h_runtime_summary.csv"
PATCH_FILTER_CSV = REPORT_DIR / "p_b_s2h_patch_filtering_summary.csv"
ERROR_CSV   = REPORT_DIR / "error.csv"

P_B10_JSON = EXP_ROOT / "outputs" / "reports" / "normal_test" / "p_b10_normal_test_sanity.json"
P_B9_JSON  = EXP_ROOT / "outputs" / "reports" / "normal_val" / "p_b9_normal_val_threshold.json"
P_B8_JSON  = EXP_ROOT / "outputs" / "reports" / "full" / "p_b8_distribution_validation.json"


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def abort(msg, code=2):
    print(f"[ABORT] {msg}")
    sys.exit(code)


def record_error(pid, etype, emsg, where):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    wh = not ERROR_CSV.exists() or ERROR_CSV.stat().st_size == 0
    with open(ERROR_CSV, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "error_type", "error_msg", "where"])
        if wh:
            w.writeheader()
        w.writerow({"patient_id": pid, "error_type": etype, "error_msg": emsg, "where": where})


def main():
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{SCRIPT_NAME}] 시작: {ts}\n")

    # ── G1~3: 선행 verdict ───────────────────────────────────────────────
    p_b10 = json.load(open(P_B10_JSON, encoding="utf-8")) if P_B10_JSON.exists() else None
    p_b9  = json.load(open(P_B9_JSON,  encoding="utf-8")) if P_B9_JSON.exists()  else None
    p_b8  = json.load(open(P_B8_JSON,  encoding="utf-8")) if P_B8_JSON.exists()  else None
    if not (p_b10 and p_b10.get("verdict") == "통과"): abort("P-B10 verdict != 통과")
    if not (p_b9  and p_b9.get("verdict")  == "통과"): abort("P-B9 verdict != 통과")
    if not (p_b8  and p_b8.get("verdict")  == "통과"): abort("P-B8 verdict != 통과")
    print("[G1-3] P-B10/P-B9/P-B8 통과 ✅")

    # ── G4: distribution npz ─────────────────────────────────────────────
    if not MODEL_NPZ.exists(): abort(f"distribution npz 없음: {MODEL_NPZ}")
    print("[G4] distribution npz 존재 ✅")

    # ── G5: selected index ───────────────────────────────────────────────
    sidx = np.load(str(SELECTED_INDICES))
    if not (sidx.shape == (REDUCED_FEATURE_DIM,) and len(np.unique(sidx)) == REDUCED_FEATURE_DIM
            and sidx.min() >= 0 and sidx.max() <= 143):
        abort(f"selected index 검증 실패: {sidx.shape}, [{sidx.min()},{sidx.max()}]")
    print(f"[G5] selected index OK: range=[{int(sidx.min())},{int(sidx.max())}] ✅")

    # ── G6: threshold read-only (재계산 금지) ────────────────────────────
    if not THRESH_JSON.exists(): abort(f"threshold JSON 없음: {THRESH_JSON}")
    thr_mtime_before = THRESH_JSON.stat().st_mtime
    thr = json.load(open(THRESH_JSON, encoding="utf-8"))
    p95 = float(thr["threshold_p95"]); p99 = float(thr["threshold_p99"])
    if abs(p95 - EXPECTED_P95) > THRESH_TOL or abs(p99 - EXPECTED_P99) > THRESH_TOL:
        abort(f"threshold 불일치: p95={p95}, p99={p99}")
    print(f"[G6] threshold read-only: p95={p95:.6f}, p99={p99:.6f} (재계산 안 함) ✅")

    # ── G7: lesion split stage2_holdout 154명 + 구성 ─────────────────────
    if not LESION_SPLIT.exists(): abort(f"lesion split 없음: {LESION_SPLIT}")
    split_rows = list(csv.DictReader(open(LESION_SPLIT, encoding="utf-8-sig")))
    s2h_rows = [r for r in split_rows if r["stage_split"] == EXPECTED_STAGE]
    if len(s2h_rows) != EXPECTED_N:
        abort(f"stage2_holdout {len(s2h_rows)}≠{EXPECTED_N}")
    groups = {}
    for r in s2h_rows:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
    if groups != EXPECTED_GROUPS:
        abort(f"group 구성 불일치: {groups} (기대 {EXPECTED_GROUPS})")
    print(f"[G7] stage2_holdout {len(s2h_rows)}명 (NSCLC {groups.get('NSCLC')} / MSD_Lung {groups.get('MSD_Lung')}) ✅")

    # ── G8: stage1_dev contamination 0 (혼입 금지) ───────────────────────
    s2h_ids = [r[JOIN_KEY].strip() for r in s2h_rows]
    stage1_dev_ids = {r[JOIN_KEY].strip() for r in split_rows if r["stage_split"] == "stage1_dev"}
    contam = set(s2h_ids) & stage1_dev_ids
    if contam:
        abort(f"stage2_holdout에 stage1_dev 혼입 {len(contam)}명 — 즉시 중단")
    print(f"[G8] stage1_dev contamination = 0 ✅")

    s2h_safe = {r[JOIN_KEY].strip(): r["safe_id"].strip() for r in s2h_rows}

    # ── G9: lesion root roi_0_0 조건 ─────────────────────────────────────
    if "roi0_0_ts_lung_raw_no_dilate" not in str(LESION_ROOT) or "model_roi" in str(LESION_ROOT):
        abort(f"lesion root 조건 불일치: {LESION_ROOT}")
    if not LESION_ROOT.exists(): abort(f"lesion root 없음: {LESION_ROOT}")
    if not LESION_MANIFEST.exists(): abort(f"lesion manifest 없음: {LESION_MANIFEST}")
    print("[G9] lesion root roi_0_0 조건 ✅")

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dist_sha = sha256_of(MODEL_NPZ)
    print(f"[p_b_s2h] distribution sha256: {dist_sha[:16]}...")

    # ── 모델 / 추출기 / loader ───────────────────────────────────────────
    from position_aware_padim.data_loader import DataLoader
    from position_aware_padim.path_resolver import PathResolver
    from position_aware_padim.padim_model import PaDiMModel
    from position_aware_padim.feature_extractor_effnet_b0_scaffold import FeatureExtractorEffNetB0

    model = PaDiMModel(selected_feature_indices_path=str(SELECTED_INDICES),
                       feature_dim=REDUCED_FEATURE_DIM, eps=1e-5)
    model.load(str(MODEL_NPZ))
    feat = FeatureExtractorEffNetB0()
    print(f"[p_b_s2h] device: {feat.device}")

    # cov_inv 캐시 빌드 (position_bin별 한 번만 계산)
    cov_inv_cache = build_cov_inv_cache(model)
    print(f"[p_b_s2h] cov_inv 캐시 완료: {len(cov_inv_cache)}개 분포")

    import torch
    gpu_avail = (feat.device == "cuda")
    if gpu_avail:
        torch.cuda.reset_peak_memory_stats()

    path_resolver = PathResolver(str(LESION_MANIFEST), str(LESION_ROOT))
    loader = DataLoader(str(LESION_MANIFEST), path_resolver, str(ERROR_CSV), use_mmap=True)

    # ── scoring 루프 (resume) ────────────────────────────────────────────
    n_scored = n_skipped = n_failed = 0
    failed = []
    per_patient_rows = []
    ct_missing = roi_missing = mask_missing = shape_mismatch = 0
    start = time.time()

    for i, pid in enumerate(s2h_ids, 1):
        safe_id = s2h_safe[pid]
        score_path = SCORE_DIR / f"{pid}.csv"
        if score_path.exists():
            n_skipped += 1
            print(f"  [SKIP] ({i}/{EXPECTED_N}) {pid}: 이미 존재 (resume)")
            continue

        data = loader.load_patient_data(pid, mask_type=MASK_TYPE)
        if data is None:
            n_failed += 1; failed.append(pid); ct_missing += 1
            record_error(pid, "load_failed", "load_patient_data None", "ct/patch")
            print(f"  [FAIL] ({i}/{EXPECTED_N}) {pid}: 로드 실패")
            continue
        ct_hu = data["ct_hu"]

        roi_path = V4_20_LESION_ROOT / safe_id / "refined_roi.npy"
        if not roi_path.exists():
            n_failed += 1; failed.append(pid); roi_missing += 1
            record_error(pid, "roi_missing", str(roi_path), "v4_20_roi")
            print(f"  [FAIL] ({i}/{EXPECTED_N}) {pid}: v4_20 ROI 없음")
            continue
        refined_roi = np.load(str(roi_path), mmap_mode='r')

        # GT mask: 존재/shape만 확인 (value 미로드)
        gt_path = LESION_VOLUMES / safe_id / GT_MASK_FILE
        if not gt_path.exists():
            mask_missing += 1
            record_error(pid, "gt_mask_missing", str(gt_path), "gt_mask")
        gt_shape = np.load(str(gt_path), mmap_mode='r').shape if gt_path.exists() else None

        # shape 일치
        if refined_roi.shape != ct_hu.shape or (gt_shape is not None and gt_shape != ct_hu.shape):
            shape_mismatch += 1; n_failed += 1; failed.append(pid)
            record_error(pid, "shape_mismatch",
                         f"ct={ct_hu.shape} roi={refined_roi.shape} gt={gt_shape}", "shape")
            print(f"  [FAIL] ({i}/{EXPECTED_N}) {pid}: shape mismatch")
            continue

        # v4_20 patch 재필터링
        patch_df = data["patch_df"]
        n_before = len(patch_df)
        keep = []
        for r in patch_df.itertuples(index=False):
            zz = int(r.local_z)
            if zz < 0 or zz >= refined_roi.shape[0]:
                keep.append(False); continue
            sub = np.asarray(refined_roi[zz, int(r.y0):int(r.y1), int(r.x0):int(r.x1)])
            ratio = float(sub.mean()) if sub.size > 0 else 0.0
            keep.append(ratio >= V4_20_PATCH_RATIO_THRESHOLD)
        keep = np.array(keep, dtype=bool)
        patch_df_v4 = patch_df[keep].reset_index(drop=True)
        n_after = len(patch_df_v4)

        data["mask"] = np.asarray(refined_roi)
        data["patch_df"] = patch_df_v4

        try:
            scored = fast_score_patient(model, feat, data, cov_inv_cache)
        except Exception as exc:
            n_failed += 1; failed.append(pid)
            record_error(pid, "score_error", str(exc), "fast_score_patient")
            print(f"  [FAIL] ({i}/{EXPECTED_N}) {pid}: score 오류 {exc}")
            continue

        scored.to_csv(score_path, index=False, encoding="utf-8-sig")
        n_scored += 1
        s = scored["padim_score"].to_numpy(dtype=np.float64)
        fs = s[np.isfinite(s)]
        ex95 = int((fs > p95).sum()); ex99 = int((fs > p99).sum())
        group_val = [r["group"] for r in s2h_rows if r[JOIN_KEY].strip() == pid]
        per_patient_rows.append({
            "patient_id": pid, "safe_id": safe_id,
            "group": group_val[0] if group_val else "unknown",
            "patch_before": n_before, "patch_after": n_after,
            "patch_removed": n_before - n_after,
            "removed_ratio": round((n_before - n_after) / n_before, 6) if n_before else 0.0,
            "scored_patches": int(s.size), "nan": int(np.isnan(s).sum()), "inf": int(np.isinf(s).sum()),
            "exceed_p95": ex95, "exceed_p99": ex99,
        })
        print(f"  [OK]   ({i}/{EXPECTED_N}) {pid}: patch {n_before:,}→{n_after:,}, scored={s.size:,}, ex95={ex95}")

    elapsed = time.time() - start
    peak_gpu_gb = (torch.cuda.max_memory_allocated() / 1e9) if gpu_avail else 0.0

    # ── 전체 score 집계 (신규 + resume 기존 CSV) ─────────────────────────
    import pandas as pd
    all_scores = []
    csv_count = 0
    for pid in s2h_ids:
        sp = SCORE_DIR / f"{pid}.csv"
        if not sp.exists():
            continue
        csv_count += 1
        df = pd.read_csv(sp, encoding="utf-8-sig", usecols=lambda c: c == "padim_score")
        s = df["padim_score"].to_numpy(dtype=np.float64)
        all_scores.append(s)
    scores = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float64)
    n_total = int(scores.size)
    n_nan = int(np.isnan(scores).sum()); n_inf = int(np.isinf(scores).sum())
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        abort("유효 score 0개")

    total_before = sum(r["patch_before"] for r in per_patient_rows)
    total_after  = sum(r["patch_after"]  for r in per_patient_rows)

    exceed95 = int((finite > p95).sum()); exceed99 = int((finite > p99).sum())
    rate95 = exceed95 / finite.size; rate99 = exceed99 / finite.size
    s_min = float(np.min(finite)); s_max = float(np.max(finite))
    s_mean = float(np.mean(finite)); s_std = float(np.std(finite)); s_median = float(np.median(finite))

    thr_mtime_after  = THRESH_JSON.stat().st_mtime
    thr_unchanged    = (thr_mtime_after == thr_mtime_before)

    print(f"\n[scoring] score CSV={csv_count}/{EXPECTED_N}, total scored patch={n_total:,}, NaN={n_nan}, Inf={n_inf}")
    print(f"[scoring] p95 초과: {exceed95:,} ({rate95*100:.3f}%) / p99 초과: {exceed99:,} ({rate99*100:.3f}%)")
    print(f"[scoring] threshold mtime 불변: {thr_unchanged}")

    verdict = ("통과" if (csv_count == EXPECTED_N and shape_mismatch == 0
                          and n_nan == 0 and n_inf == 0 and thr_unchanged) else "부분통과")

    # ── 저장 ─────────────────────────────────────────────────────────────
    summary = {
        "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        "official_roi_source": "refined_roi_v4_20_modeB_all_v1",
        "verdict": verdict, "created": ts,
        "n_stage2_holdout": csv_count, "n_scored_this_run": n_scored,
        "n_skipped_resume": n_skipped, "n_failed": n_failed, "failed_patients": failed,
        "nsclc": groups.get("NSCLC"), "msd_lung": groups.get("MSD_Lung"),
        "stage1_dev_contamination": len(contam),
        "ct_missing": ct_missing, "roi_missing": roi_missing, "mask_missing": mask_missing,
        "shape_mismatch": shape_mismatch,
        "total_scored_patches": n_total,
        "total_patch_before_this_run": total_before, "total_patch_after_this_run": total_after,
        "n_nan": n_nan, "n_inf": n_inf,
        "score_min": s_min, "score_max": s_max, "score_mean": s_mean,
        "score_std": s_std, "score_median": s_median,
        "threshold_p95": p95, "threshold_p99": p99,
        "threshold_recalculated": False, "threshold_json_mtime_unchanged": thr_unchanged,
        "exceed_p95": exceed95, "rate_exceed_p95": round(rate95, 6),
        "exceed_p99": exceed99, "rate_exceed_p99": round(rate99, 6),
        "label_source": "lesion patch CSV (has_lesion_patch). GT mask는 shape 검증만.",
        "distribution_sha256": dist_sha,
    }
    with open(SCORING_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(SCORING_SUMMARY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k in ["n_stage2_holdout", "total_scored_patches", "score_min", "score_max",
                  "score_mean", "score_std", "score_median", "threshold_p95", "threshold_p99",
                  "exceed_p95", "rate_exceed_p95", "exceed_p99", "rate_exceed_p99"]:
            w.writerow([k, summary[k]])

    if per_patient_rows:
        with open(PATCH_FILTER_CSV, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["patient_id", "safe_id", "group", "patch_before", "patch_after",
                                              "patch_removed", "removed_ratio", "scored_patches",
                                              "nan", "inf", "exceed_p95", "exceed_p99"])
            w.writeheader()
            w.writerows(per_patient_rows)

    with open(RUNTIME_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "script", "metric", "value"])
        for k, v in [("n_stage2_holdout", csv_count), ("n_scored_this_run", n_scored),
                     ("n_skipped_resume", n_skipped), ("total_scored_patches", n_total),
                     ("rate_exceed_p95", round(rate95, 6)), ("rate_exceed_p99", round(rate99, 6)),
                     ("elapsed_seconds", round(elapsed, 2)), ("peak_gpu_gb", round(peak_gpu_gb, 3))]:
            w.writerow([ts, SCRIPT_NAME, k, v])

    report = dict(summary)
    report["step"] = "p_b_s2h"
    report["elapsed_seconds"] = round(elapsed, 2)
    report["peak_gpu_gb"] = round(peak_gpu_gb, 3)
    report["normal_val_test_rerun"] = False
    report["safety"] = {
        "metrics_calculated": False, "auroc_auprc_computed": False, "dice_recall_computed": False,
        "threshold_recalculated": False, "normal_val_rerun": False, "normal_test_rerun": False,
        "stage2_holdout_accessed": True, "stage2_holdout_eval_only": True,
        "model_roi_used": False, "e_drive_used": False,
        "existing_stage1_results_modified": False,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# p_b_s2h EfficientNet-B0 Stage2_holdout Scoring\n",
        f"**판정: {verdict}**\n",
        f"- 생성일시: {ts}",
        f"- branch: efficientnet_b0_imagenet_chestwall_removed_roi_v1 / ROI: refined_roi_v4_20_modeB_all_v1\n",
        "## 처리\n",
        f"- stage2_holdout score CSV: {csv_count}/{EXPECTED_N} (이번 실행 신규 {n_scored}, resume skip {n_skipped}, 실패 {n_failed})",
        f"- NSCLC {groups.get('NSCLC')} / MSD_Lung {groups.get('MSD_Lung')}",
        f"- stage1_dev contamination: {len(contam)}",
        f"- CT/ROI/GT mask 누락: {ct_missing}/{roi_missing}/{mask_missing}, shape mismatch: {shape_mismatch}",
        f"- total scored patch: {n_total:,}",
        f"- (이번 실행) patch before→after: {total_before:,} → {total_after:,}",
        f"- NaN/Inf: {n_nan}/{n_inf}\n",
        "## score 통계\n",
        "| 지표 | 값 |",
        "|------|----|",
        f"| min | {s_min:.6f} |",
        f"| max | {s_max:.6f} |",
        f"| mean | {s_mean:.6f} |",
        f"| std | {s_std:.6f} |",
        f"| median | {s_median:.6f} |\n",
        "## threshold exceedance (P-B9 고정, 재계산 없음)\n",
        "| threshold | 값 | 초과 patch | 초과율 |",
        "|-----------|----|-----------|--------|",
        f"| p95 | {p95:.6f} | {exceed95:,} | {rate95*100:.3f}% |",
        f"| p99 | {p99:.6f} | {exceed99:,} | {rate99*100:.3f}% |\n",
        f"- threshold JSON mtime 불변: **{thr_unchanged}**\n",
        "## 가드레일\n",
        "- stage2_holdout eval-only (method tuning 금지)",
        "- distribution npz / threshold: read-only (stage1_dev 학습 결과 그대로)",
        "- metrics / AUROC·AUPRC / threshold 재계산: 안 함",
        "- stage1_dev score CSV 수정: 안 함",
    ]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n=== p_b_s2h 완료: {verdict} ===")
    print(f"[보고서] {REPORT_DIR}")
    return 0 if verdict != "실패" else 1


if __name__ == "__main__":
    sys.exit(main())
