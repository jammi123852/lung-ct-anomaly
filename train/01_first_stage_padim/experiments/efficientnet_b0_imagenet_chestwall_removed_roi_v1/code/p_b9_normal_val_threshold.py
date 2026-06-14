"""
P-B9: v4_20 ROI EfficientNet-B0 normal val threshold

v4_20 흉벽 제거 ROI branch full distribution으로 normal_val 36명 scoring + p95/p99 threshold 계산.
- CT: roi_0_0 ct_hu.npy (C드라이브, v4_20와 동일 좌표계)
- ROI: refined_roi_v4_20_modeB_all_v1/normal/<safe_id>/refined_roi.npy (v4_20 lock)
- patch 재필터링: v4_20 ROI ratio >= 0.5
- threshold: v4_20 branch 전용 신규 계산 (기존 roi_0_0 p95=13.240479 재사용 금지)

금지: normal test scoring / lesion scoring / metrics / AUROC·AUPRC / stage2_holdout
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

PROJ_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT  = Path(__file__).resolve().parents[1]
ROI0_BRANCH = PROJ_ROOT / "experiments" / "efficientnet_b0_imagenet_v1"
SRC_DIR   = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

BACKBONE            = "efficientnet_b0"
RAW_FEATURE_DIM     = 144
REDUCED_FEATURE_DIM = 100
MASK_TYPE           = "roi_0_0"
PATHS_CONFIG        = "paths.local.v2_roi0_0.yaml"
SCRIPT_NAME         = "p_b9_normal_val_threshold.py"
EXPECTED_VAL_N      = 36

V4_20_PATCH_RATIO_THRESHOLD = 0.5
V4_20_NORMAL_ROOT = PROJ_ROOT / "outputs" / "mip-postprocess-research-v1" / "masks" / "refined_roi_v4_20_modeB_all_v1" / "normal"

# 참고용(재사용 금지)
ROI0_P95 = 13.240479
ROI0_P99 = 15.332286

MODEL_NPZ        = EXP_ROOT / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
SELECTED_INDICES = EXP_ROOT / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
EXPECTED_GLOBAL_COUNT = 11356415

SCORE_DIR   = EXP_ROOT / "outputs" / "scores" / "normal_val_by_patient"
THRESH_DIR  = EXP_ROOT / "outputs" / "evaluation" / "normal_val_thresholds"
THRESH_JSON = THRESH_DIR / "normal_val_threshold.json"
THRESH_CSV  = THRESH_DIR / "normal_val_threshold.csv"
REPORT_DIR  = EXP_ROOT / "outputs" / "reports" / "normal_val"
REPORT_MD   = REPORT_DIR / "p_b9_normal_val_threshold.md"
REPORT_JSON = REPORT_DIR / "p_b9_normal_val_threshold.json"
RUNTIME_CSV = REPORT_DIR / "p_b9_runtime_summary.csv"
PATCH_FILTER_CSV = REPORT_DIR / "p_b9_patch_filtering_summary.csv"

SPLIT_JSON = PROJ_ROOT / "outputs" / "position-aware-padim-v1" / "splits" / "normal_v1.json"
P_B8_JSON  = EXP_ROOT / "outputs" / "reports" / "full" / "p_b8_distribution_validation.json"
P_B7_JSON  = EXP_ROOT / "outputs" / "reports" / "full" / "p_b7_full_train.json"


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def abort(msg, code=2):
    print(f"[ABORT] {msg}")
    sys.exit(code)


def main():
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{SCRIPT_NAME}] 시작: {ts}\n")

    # ── 가드 1~2: P-B8/P-B7 verdict ──────────────────────────────────────
    p_b8 = json.load(open(P_B8_JSON, encoding="utf-8")) if P_B8_JSON.exists() else None
    p_b7 = json.load(open(P_B7_JSON, encoding="utf-8")) if P_B7_JSON.exists() else None
    if not (p_b8 and p_b8.get("verdict") == "통과"):
        abort(f"P-B8 verdict != 통과: {p_b8.get('verdict') if p_b8 else None}")
    if not (p_b7 and p_b7.get("verdict") == "통과"):
        abort(f"P-B7 verdict != 통과: {p_b7.get('verdict') if p_b7 else None}")
    print(f"[guard1-2] P-B8={p_b8.get('verdict')}, P-B7={p_b7.get('verdict')} ✅")

    # ── 가드 3~4: distribution npz / global_count ────────────────────────
    if not MODEL_NPZ.exists():
        abort(f"distribution npz 없음: {MODEL_NPZ}")
    z = np.load(str(MODEL_NPZ), allow_pickle=True)
    global_count = int(z["global_pure_lung_count"])
    if global_count != EXPECTED_GLOBAL_COUNT:
        abort(f"global_count {global_count} != {EXPECTED_GLOBAL_COUNT}")
    print(f"[guard3-4] distribution npz OK, global_count={global_count} ✅")

    # ── 가드 5: selected index ───────────────────────────────────────────
    sidx = np.load(str(SELECTED_INDICES))
    if not (sidx.shape == (REDUCED_FEATURE_DIM,) and len(np.unique(sidx)) == REDUCED_FEATURE_DIM
            and sidx.min() >= 0 and sidx.max() <= 143):
        abort(f"selected index 검증 실패: shape={sidx.shape}, range=[{sidx.min()},{sidx.max()}]")
    print(f"[guard5] selected index OK: shape={sidx.shape}, range=[{int(sidx.min())},{int(sidx.max())}] ✅")

    # ── 가드 6: normal_val split 36 ──────────────────────────────────────
    split_data = json.load(open(SPLIT_JSON, encoding="utf-8"))
    val_patients = list(split_data["val"])
    p2s = split_data.get("patient_to_safe_id", {})
    if len(val_patients) != EXPECTED_VAL_N:
        abort(f"normal_val {len(val_patients)}≠{EXPECTED_VAL_N}")
    print(f"[guard6] normal_val {len(val_patients)}명 ✅")

    # ── 가드 7: 출력 collision (덮어쓰기 금지) ───────────────────────────
    if THRESH_JSON.exists():
        abort(f"threshold JSON 이미 존재 (덮어쓰기 금지): {THRESH_JSON}")
    if SCORE_DIR.exists() and any(SCORE_DIR.glob("*.csv")):
        abort(f"normal_val score CSV 이미 존재 (덮어쓰기 금지): {SCORE_DIR}")

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    THRESH_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dist_sha = sha256_of(MODEL_NPZ)
    print(f"[P-B9] distribution sha256: {dist_sha[:16]}...")

    # ── 모델 / 추출기 ────────────────────────────────────────────────────
    from position_aware_padim.config_manager import ConfigManager
    from position_aware_padim.data_loader import DataLoader
    from position_aware_padim.path_resolver import PathResolver
    from position_aware_padim.padim_model import PaDiMModel
    from position_aware_padim.feature_extractor_effnet_b0_scaffold import FeatureExtractorEffNetB0

    cfg = ConfigManager(str(PROJ_ROOT))
    cfg.load_config(paths_yaml=PATHS_CONFIG)
    normal_training_ready = cfg.get("paths", "normal_training_ready", "")
    manifest_path = Path(normal_training_ready) / "manifests" / "patient_manifest.csv"

    model = PaDiMModel(
        selected_feature_indices_path=str(SELECTED_INDICES),
        feature_dim=REDUCED_FEATURE_DIM, eps=1e-5,
    )
    model.load(str(MODEL_NPZ))
    print(f"[P-B9] distribution 로드 완료: position_bin 수={len(model.stats)}")

    feat = FeatureExtractorEffNetB0()
    print(f"[P-B9] device: {feat.device}")

    import torch
    gpu_avail = (feat.device == "cuda")
    if gpu_avail:
        torch.cuda.reset_peak_memory_stats()

    path_resolver = PathResolver(str(manifest_path), normal_training_ready)
    loader = DataLoader(str(manifest_path), path_resolver, str(REPORT_DIR / "error.csv"), use_mmap=True)

    # ── val scoring (v4_20 ROI 교체 + patch 재필터링) ────────────────────
    all_scores = []
    n_csv = n_failed = 0
    missing = []
    shape_mismatch = 0
    per_patient_rows = []
    start = time.time()

    for i, pid in enumerate(val_patients, 1):
        safe_id = p2s.get(pid, pid)
        data = loader.load_patient_data(pid, mask_type=MASK_TYPE)
        if data is None:
            n_failed += 1; missing.append(pid)
            print(f"  [FAIL] ({i}/{EXPECTED_VAL_N}) {pid}: 로드 실패")
            continue
        ct_hu = data["ct_hu"]
        roi_path = V4_20_NORMAL_ROOT / safe_id / "refined_roi.npy"
        if not roi_path.exists():
            n_failed += 1; missing.append(pid)
            print(f"  [FAIL] ({i}/{EXPECTED_VAL_N}) {pid}: v4_20 ROI 없음")
            continue
        refined_roi = np.load(str(roi_path), mmap_mode='r')
        if refined_roi.shape != ct_hu.shape:
            shape_mismatch += 1; n_failed += 1; missing.append(pid)
            print(f"  [FAIL] ({i}/{EXPECTED_VAL_N}) {pid}: shape mismatch")
            continue

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

        scored = model.score_patient(feat, data)
        out_csv = SCORE_DIR / f"{pid}.csv"
        scored.to_csv(out_csv, index=False, encoding="utf-8-sig")
        n_csv += 1
        s = scored["padim_score"].to_numpy(dtype=np.float64)
        all_scores.append(s)
        per_patient_rows.append({
            "patient_id": pid, "safe_id": safe_id,
            "patch_before": n_before, "patch_after": n_after,
            "patch_removed": n_before - n_after,
            "removed_ratio": round((n_before - n_after) / n_before, 6) if n_before else 0.0,
            "scored_patches": int(s.size),
            "nan": int(np.isnan(s).sum()), "inf": int(np.isinf(s).sum()),
        })
        print(f"  [OK]   ({i}/{EXPECTED_VAL_N}) {pid}: patch {n_before:,}→{n_after:,}, "
              f"scored={s.size:,}, nan={int(np.isnan(s).sum())}")

    elapsed = time.time() - start
    peak_gpu_gb = (torch.cuda.max_memory_allocated() / 1e9) if gpu_avail else 0.0

    # ── threshold 계산 ───────────────────────────────────────────────────
    scores = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float64)
    n_total = int(scores.size)
    n_nan = int(np.isnan(scores).sum())
    n_inf = int(np.isinf(scores).sum())
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        abort("유효 score 0개")

    p95 = float(np.percentile(finite, 95))
    p99 = float(np.percentile(finite, 99))
    s_min = float(np.min(finite)); s_max = float(np.max(finite))
    s_mean = float(np.mean(finite)); s_std = float(np.std(finite))
    s_median = float(np.median(finite))

    total_before = sum(r["patch_before"] for r in per_patient_rows)
    total_after = sum(r["patch_after"] for r in per_patient_rows)

    print(f"\n[threshold] total scored patch={n_total:,}, NaN={n_nan}, Inf={n_inf}")
    print(f"[threshold] p95={p95:.6f}, p99={p99:.6f}")
    print(f"[threshold] min={s_min:.4f}, max={s_max:.4f}, mean={s_mean:.4f}, std={s_std:.4f}, median={s_median:.4f}")
    print(f"[threshold] (참고) 기존 roi_0_0 p95={ROI0_P95}, p99={ROI0_P99} — 재사용 안 함")

    verdict = "통과" if (n_csv == EXPECTED_VAL_N and shape_mismatch == 0 and n_nan == 0 and n_inf == 0) else "부분통과"

    # ── threshold JSON/CSV ───────────────────────────────────────────────
    dist_sha_full = dist_sha
    thr_obj = {
        "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        "official_roi_source": "refined_roi_v4_20_modeB_all_v1",
        "threshold_p95": p95,
        "threshold_p99": p99,
        "v4_20_branch_specific": True,
        "roi0_threshold_reused": False,
        "roi0_reference_only": {"p95": ROI0_P95, "p99": ROI0_P99},
        "n_val_patients": n_csv,
        "val_n_patches": n_total,
        "n_nan": n_nan, "n_inf": n_inf,
        "score_min": s_min, "score_max": s_max, "score_mean": s_mean,
        "score_std": s_std, "score_median": s_median,
        "distribution_npz": str(MODEL_NPZ),
        "distribution_sha256": dist_sha_full,
        "selected_indices": str(SELECTED_INDICES),
        "patch_ratio_threshold": V4_20_PATCH_RATIO_THRESHOLD,
        "created": ts,
    }
    with open(THRESH_JSON, "w", encoding="utf-8") as f:
        json.dump(thr_obj, f, ensure_ascii=False, indent=2)
    with open(THRESH_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k in ["threshold_p95", "threshold_p99", "n_val_patients", "val_n_patches",
                  "score_min", "score_max", "score_mean", "score_std", "score_median",
                  "n_nan", "n_inf"]:
            w.writerow([k, thr_obj[k]])

    # patch filtering summary
    with open(PATCH_FILTER_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "safe_id", "patch_before", "patch_after",
                                          "patch_removed", "removed_ratio", "scored_patches", "nan", "inf"])
        w.writeheader()
        w.writerows(per_patient_rows)

    # runtime
    with open(RUNTIME_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "script", "metric", "value"])
        for k, v in [("n_val_patients", n_csv), ("n_failed", n_failed),
                     ("total_scored_patches", n_total), ("total_patch_before", total_before),
                     ("total_patch_after", total_after), ("p95", p95), ("p99", p99),
                     ("elapsed_seconds", round(elapsed, 2)), ("peak_gpu_gb", round(peak_gpu_gb, 3))]:
            w.writerow([ts, SCRIPT_NAME, k, v])

    # report JSON
    report = {
        "step": "P-B9", "verdict": verdict, "timestamp": ts,
        "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        "official_roi_source": "refined_roi_v4_20_modeB_all_v1",
        "n_val_patients_processed": n_csv, "n_failed": n_failed,
        "missing": missing, "shape_mismatch": shape_mismatch,
        "total_scored_patches": n_total,
        "total_patch_before": total_before, "total_patch_after": total_after,
        "total_patch_removed": total_before - total_after,
        "total_removed_ratio": round((total_before - total_after) / total_before, 6) if total_before else 0.0,
        "n_nan": n_nan, "n_inf": n_inf,
        "score_min": s_min, "score_max": s_max, "score_mean": s_mean,
        "score_std": s_std, "score_median": s_median,
        "threshold_p95": p95, "threshold_p99": p99,
        "v4_20_branch_specific_threshold": True,
        "roi0_threshold_reused": False,
        "roi0_reference_only": {"p95": ROI0_P95, "p99": ROI0_P99},
        "distribution_npz": str(MODEL_NPZ), "distribution_sha256": dist_sha_full,
        "selected_indices": str(SELECTED_INDICES),
        "elapsed_seconds": round(elapsed, 2), "peak_gpu_gb": round(peak_gpu_gb, 3),
        "safety": {
            "normal_test_scoring": False, "lesion_scoring": False,
            "metrics_calculated": False, "auroc_auprc_computed": False,
            "stage2_holdout_accessed": False, "model_roi_used": False,
            "e_drive_used": False, "lesion_file_used": False,
            "training": False, "existing_results_modified": False,
        },
        "next_step_p_b10_normal_test_sanity_feasible": (verdict == "통과"),
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = [
        "# P-B9 v4_20 ROI EfficientNet-B0 Normal Val Threshold\n",
        f"**판정: {verdict}**\n",
        f"- 생성일시: {ts}",
        f"- branch: efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        f"- ROI source: refined_roi_v4_20_modeB_all_v1\n",
        "## 처리\n",
        f"- normal_val 처리: {n_csv}/{EXPECTED_VAL_N} (실패 {n_failed})",
        f"- shape mismatch: {shape_mismatch}",
        f"- total scored patch: {n_total:,}",
        f"- patch before→after: {total_before:,} → {total_after:,} (제거 {total_before-total_after:,})",
        f"- NaN/Inf: {n_nan}/{n_inf}\n",
        "## score 통계\n",
        "| 지표 | 값 |",
        "|------|----|",
        f"| min | {s_min:.6f} |",
        f"| max | {s_max:.6f} |",
        f"| mean | {s_mean:.6f} |",
        f"| std | {s_std:.6f} |",
        f"| median | {s_median:.6f} |\n",
        "## threshold (v4_20 branch 전용 신규)\n",
        "| 지표 | 값 |",
        "|------|----|",
        f"| **p95** | **{p95:.6f}** |",
        f"| **p99** | **{p99:.6f}** |\n",
        f"- ⚠ 기존 roi_0_0 threshold(p95={ROI0_P95}, p99={ROI0_P99})는 **참고만**, 재사용 안 함",
        f"- 이 threshold는 **v4_20 흉벽 제거 ROI branch 전용**\n",
        "## 사용 artifact\n",
        f"- distribution: `{MODEL_NPZ}` (sha256 `{dist_sha_full[:16]}...`)",
        f"- selected index: `{SELECTED_INDICES}`",
        f"- patch ratio threshold: {V4_20_PATCH_RATIO_THRESHOLD}\n",
        "## 실행\n",
        f"- elapsed: {round(elapsed,2)}초, peak GPU: {round(peak_gpu_gb,3)} GB\n",
        "## 미실행 / 미사용 확인\n",
        "- normal test scoring / lesion scoring / metrics / AUROC·AUPRC: 미실행",
        "- stage2_holdout / model_roi / E드라이브 / lesion 파일: 미접근·미사용",
        "- 기존 roi_0_0 threshold 재사용: 안 함",
        "- 기존 P-B1~P-B8 / roi_0_0 branch 결과: 무수정\n",
        "## 다음 단계\n",
        f"- P-B10 normal test sanity 진행 가능: {verdict == '통과'}",
    ]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n=== P-B9 완료: {verdict} ===")
    print(f"[보고서] {REPORT_DIR}")
    print(f"[threshold] {THRESH_JSON}")
    return 0 if verdict != "실패" else 1


if __name__ == "__main__":
    sys.exit(main())
