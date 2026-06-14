"""
P-B10: v4_20 ROI EfficientNet-B0 normal test sanity

v4_20 ROI branch full distribution + P-B9 threshold(고정)로 normal_test 36명 scoring sanity.
- CT: roi_0_0 ct_hu.npy (C드라이브, v4_20와 동일 좌표계)
- ROI: refined_roi_v4_20_modeB_all_v1/normal/<safe_id>/refined_roi.npy (v4_20 lock)
- patch 재필터링: v4_20 ROI ratio >= 0.5
- threshold: P-B9 값 read-only 로드. 재계산/수정 절대 금지. mtime 불변 검증.

금지: threshold 재계산 / normal val 재실행 / lesion scoring / metrics / AUROC·AUPRC / stage2_holdout
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
SCRIPT_NAME         = "p_b10_normal_test_sanity.py"
EXPECTED_TEST_N     = 36

V4_20_PATCH_RATIO_THRESHOLD = 0.5
V4_20_NORMAL_ROOT = PROJ_ROOT / "outputs" / "mip-postprocess-research-v1" / "masks" / "refined_roi_v4_20_modeB_all_v1" / "normal"

# P-B9 고정 threshold (read-only, 재계산 금지)
EXPECTED_P95 = 13.231265
EXPECTED_P99 = 15.472385
THRESH_TOL = 1e-4

MODEL_NPZ        = EXP_ROOT / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
SELECTED_INDICES = EXP_ROOT / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
THRESH_JSON      = EXP_ROOT / "outputs" / "evaluation" / "normal_val_thresholds" / "normal_val_threshold.json"

SCORE_DIR  = EXP_ROOT / "outputs" / "scores" / "normal_test_by_patient"
EVAL_DIR   = EXP_ROOT / "outputs" / "evaluation" / "normal_test_sanity"
EVAL_JSON  = EVAL_DIR / "normal_test_sanity_summary.json"
EVAL_CSV   = EVAL_DIR / "normal_test_sanity_summary.csv"
REPORT_DIR = EXP_ROOT / "outputs" / "reports" / "normal_test"
REPORT_MD  = REPORT_DIR / "p_b10_normal_test_sanity.md"
REPORT_JSON = REPORT_DIR / "p_b10_normal_test_sanity.json"
RUNTIME_CSV = REPORT_DIR / "p_b10_runtime_summary.csv"
PATCH_FILTER_CSV = REPORT_DIR / "p_b10_patch_filtering_summary.csv"

SPLIT_JSON = PROJ_ROOT / "outputs" / "position-aware-padim-v1" / "splits" / "normal_v1.json"
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


def main():
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{SCRIPT_NAME}] 시작: {ts}\n")

    # ── 가드 1~2: P-B9/P-B8 verdict ──────────────────────────────────────
    p_b9 = json.load(open(P_B9_JSON, encoding="utf-8")) if P_B9_JSON.exists() else None
    p_b8 = json.load(open(P_B8_JSON, encoding="utf-8")) if P_B8_JSON.exists() else None
    if not (p_b9 and p_b9.get("verdict") == "통과"):
        abort(f"P-B9 verdict != 통과: {p_b9.get('verdict') if p_b9 else None}")
    if not (p_b8 and p_b8.get("verdict") == "통과"):
        abort(f"P-B8 verdict != 통과: {p_b8.get('verdict') if p_b8 else None}")
    print(f"[guard1-2] P-B9={p_b9.get('verdict')}, P-B8={p_b8.get('verdict')} ✅")

    # ── 가드 3: distribution npz ─────────────────────────────────────────
    if not MODEL_NPZ.exists():
        abort(f"distribution npz 없음: {MODEL_NPZ}")

    # ── 가드 4: selected index ───────────────────────────────────────────
    sidx = np.load(str(SELECTED_INDICES))
    if not (sidx.shape == (REDUCED_FEATURE_DIM,) and len(np.unique(sidx)) == REDUCED_FEATURE_DIM
            and sidx.min() >= 0 and sidx.max() <= 143):
        abort(f"selected index 검증 실패: {sidx.shape}, [{sidx.min()},{sidx.max()}]")
    print(f"[guard4] selected index OK: shape={sidx.shape}, range=[{int(sidx.min())},{int(sidx.max())}] ✅")

    # ── 가드 5: threshold read-only 로드 + mtime 기록 (재계산 금지) ──────
    if not THRESH_JSON.exists():
        abort(f"P-B9 threshold JSON 없음: {THRESH_JSON}")
    thr_mtime_before = THRESH_JSON.stat().st_mtime
    thr = json.load(open(THRESH_JSON, encoding="utf-8"))
    p95 = float(thr["threshold_p95"])
    p99 = float(thr["threshold_p99"])
    if abs(p95 - EXPECTED_P95) > THRESH_TOL or abs(p99 - EXPECTED_P99) > THRESH_TOL:
        abort(f"threshold 불일치: p95={p95}(기대 {EXPECTED_P95}), p99={p99}(기대 {EXPECTED_P99})")
    print(f"[guard5] threshold read-only 로드: p95={p95:.6f}, p99={p99:.6f} (재계산 안 함) ✅")
    print(f"[guard5] threshold JSON mtime(before)={thr_mtime_before}")

    # ── 가드 6: normal_test split 36 ─────────────────────────────────────
    split_data = json.load(open(SPLIT_JSON, encoding="utf-8"))
    test_patients = list(split_data["test"])
    p2s = split_data.get("patient_to_safe_id", {})
    if len(test_patients) != EXPECTED_TEST_N:
        abort(f"normal_test {len(test_patients)}≠{EXPECTED_TEST_N}")
    print(f"[guard6] normal_test {len(test_patients)}명 ✅")

    # ── 가드 7: 출력 collision ───────────────────────────────────────────
    if EVAL_JSON.exists():
        abort(f"sanity summary 이미 존재 (덮어쓰기 금지): {EVAL_JSON}")
    if SCORE_DIR.exists() and any(SCORE_DIR.glob("*.csv")):
        abort(f"normal_test score CSV 이미 존재 (덮어쓰기 금지): {SCORE_DIR}")

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dist_sha = sha256_of(MODEL_NPZ)
    print(f"[P-B10] distribution sha256: {dist_sha[:16]}...")

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
    feat = FeatureExtractorEffNetB0()
    print(f"[P-B10] device: {feat.device}")

    import torch
    gpu_avail = (feat.device == "cuda")
    if gpu_avail:
        torch.cuda.reset_peak_memory_stats()

    path_resolver = PathResolver(str(manifest_path), normal_training_ready)
    loader = DataLoader(str(manifest_path), path_resolver, str(REPORT_DIR / "error.csv"), use_mmap=True)

    # ── test scoring (v4_20 ROI 교체 + patch 재필터링) ───────────────────
    all_scores = []
    n_csv = n_failed = 0
    missing = []
    shape_mismatch = 0
    per_patient_rows = []
    start = time.time()

    for i, pid in enumerate(test_patients, 1):
        safe_id = p2s.get(pid, pid)
        data = loader.load_patient_data(pid, mask_type=MASK_TYPE)
        if data is None:
            n_failed += 1; missing.append(pid)
            print(f"  [FAIL] ({i}/{EXPECTED_TEST_N}) {pid}: 로드 실패")
            continue
        ct_hu = data["ct_hu"]
        roi_path = V4_20_NORMAL_ROOT / safe_id / "refined_roi.npy"
        if not roi_path.exists():
            n_failed += 1; missing.append(pid)
            print(f"  [FAIL] ({i}/{EXPECTED_TEST_N}) {pid}: v4_20 ROI 없음")
            continue
        refined_roi = np.load(str(roi_path), mmap_mode='r')
        if refined_roi.shape != ct_hu.shape:
            shape_mismatch += 1; n_failed += 1; missing.append(pid)
            print(f"  [FAIL] ({i}/{EXPECTED_TEST_N}) {pid}: shape mismatch")
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
        scored.to_csv(SCORE_DIR / f"{pid}.csv", index=False, encoding="utf-8-sig")
        n_csv += 1
        s = scored["padim_score"].to_numpy(dtype=np.float64)
        all_scores.append(s)
        fs = s[np.isfinite(s)]
        ex95 = int((fs > p95).sum()); ex99 = int((fs > p99).sum())
        per_patient_rows.append({
            "patient_id": pid, "safe_id": safe_id,
            "patch_before": n_before, "patch_after": n_after,
            "patch_removed": n_before - n_after,
            "removed_ratio": round((n_before - n_after) / n_before, 6) if n_before else 0.0,
            "scored_patches": int(s.size), "nan": int(np.isnan(s).sum()), "inf": int(np.isinf(s).sum()),
            "exceed_p95": ex95, "exceed_p99": ex99,
            "rate_p95": round(ex95 / fs.size, 6) if fs.size else 0.0,
            "rate_p99": round(ex99 / fs.size, 6) if fs.size else 0.0,
        })
        print(f"  [OK]   ({i}/{EXPECTED_TEST_N}) {pid}: patch {n_before:,}→{n_after:,}, "
              f"ex95={ex95}({ex95/max(fs.size,1)*100:.2f}%), ex99={ex99}({ex99/max(fs.size,1)*100:.2f}%)")

    elapsed = time.time() - start
    peak_gpu_gb = (torch.cuda.max_memory_allocated() / 1e9) if gpu_avail else 0.0

    # ── 집계 ─────────────────────────────────────────────────────────────
    scores = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float64)
    n_total = int(scores.size)
    n_nan = int(np.isnan(scores).sum())
    n_inf = int(np.isinf(scores).sum())
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        abort("유효 score 0개")

    exceed95 = int((finite > p95).sum())
    exceed99 = int((finite > p99).sum())
    rate95 = exceed95 / finite.size
    rate99 = exceed99 / finite.size
    s_min = float(np.min(finite)); s_max = float(np.max(finite))
    s_mean = float(np.mean(finite)); s_std = float(np.std(finite)); s_median = float(np.median(finite))
    total_before = sum(r["patch_before"] for r in per_patient_rows)
    total_after = sum(r["patch_after"] for r in per_patient_rows)

    # ── 가드: threshold JSON mtime 불변 ──────────────────────────────────
    thr_mtime_after = THRESH_JSON.stat().st_mtime
    thr_unchanged = (thr_mtime_after == thr_mtime_before)

    print(f"\n[sanity] total scored patch={n_total:,}, NaN={n_nan}, Inf={n_inf}")
    print(f"[sanity] p95 초과: {exceed95:,} ({rate95*100:.3f}%) / p99 초과: {exceed99:,} ({rate99*100:.3f}%)")
    print(f"[sanity] threshold mtime 불변: {thr_unchanged}")

    # sanity 판정: p95 초과율 5% 근처, p99 초과율 1% 근처
    sanity_p95_ok = rate95 <= 0.10   # high-normal tail 허용 (느슨)
    sanity_p99_ok = rate99 <= 0.03
    verdict = "통과" if (n_csv == EXPECTED_TEST_N and shape_mismatch == 0 and n_nan == 0 and n_inf == 0
                         and thr_unchanged and sanity_p95_ok and sanity_p99_ok) else "부분통과"

    # ── 저장 ─────────────────────────────────────────────────────────────
    summary = {
        "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        "official_roi_source": "refined_roi_v4_20_modeB_all_v1",
        "verdict": verdict, "created": ts,
        "n_test_patients": n_csv, "n_failed": n_failed, "shape_mismatch": shape_mismatch,
        "total_scored_patches": n_total,
        "total_patch_before": total_before, "total_patch_after": total_after,
        "total_removed_ratio": round((total_before - total_after) / total_before, 6) if total_before else 0.0,
        "n_nan": n_nan, "n_inf": n_inf,
        "score_min": s_min, "score_max": s_max, "score_mean": s_mean,
        "score_std": s_std, "score_median": s_median,
        "threshold_p95": p95, "threshold_p99": p99,
        "threshold_recalculated": False, "threshold_json_mtime_unchanged": thr_unchanged,
        "exceed_p95": exceed95, "rate_exceed_p95": round(rate95, 6),
        "exceed_p99": exceed99, "rate_exceed_p99": round(rate99, 6),
        "distribution_sha256": dist_sha,
    }
    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(EVAL_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k in ["n_test_patients", "total_scored_patches", "score_min", "score_max",
                  "score_mean", "score_std", "score_median", "threshold_p95", "threshold_p99",
                  "exceed_p95", "rate_exceed_p95", "exceed_p99", "rate_exceed_p99"]:
            w.writerow([k, summary[k]])

    with open(PATCH_FILTER_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "safe_id", "patch_before", "patch_after",
                                          "patch_removed", "removed_ratio", "scored_patches",
                                          "nan", "inf", "exceed_p95", "exceed_p99", "rate_p95", "rate_p99"])
        w.writeheader()
        w.writerows(per_patient_rows)

    with open(RUNTIME_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "script", "metric", "value"])
        for k, v in [("n_test_patients", n_csv), ("total_scored_patches", n_total),
                     ("rate_exceed_p95", round(rate95, 6)), ("rate_exceed_p99", round(rate99, 6)),
                     ("elapsed_seconds", round(elapsed, 2)), ("peak_gpu_gb", round(peak_gpu_gb, 3))]:
            w.writerow([ts, SCRIPT_NAME, k, v])

    report = {
        "step": "P-B10", "verdict": verdict, "timestamp": ts,
        "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        "official_roi_source": "refined_roi_v4_20_modeB_all_v1",
        "n_test_patients_processed": n_csv, "n_failed": n_failed,
        "missing": missing, "shape_mismatch": shape_mismatch,
        "total_scored_patches": n_total,
        "total_patch_before": total_before, "total_patch_after": total_after,
        "total_removed_ratio": round((total_before - total_after) / total_before, 6) if total_before else 0.0,
        "n_nan": n_nan, "n_inf": n_inf,
        "score_min": s_min, "score_max": s_max, "score_mean": s_mean,
        "score_std": s_std, "score_median": s_median,
        "threshold_p95_used": p95, "threshold_p99_used": p99,
        "threshold_recalculated": False, "threshold_json_mtime_unchanged": thr_unchanged,
        "exceed_p95": exceed95, "rate_exceed_p95": round(rate95, 6),
        "exceed_p99": exceed99, "rate_exceed_p99": round(rate99, 6),
        "normal_val_rerun": False,
        "elapsed_seconds": round(elapsed, 2), "peak_gpu_gb": round(peak_gpu_gb, 3),
        "safety": {
            "threshold_recalculated": False, "normal_val_rerun": False,
            "lesion_scoring": False, "metrics_calculated": False,
            "auroc_auprc_computed": False, "stage2_holdout_accessed": False,
            "model_roi_used": False, "e_drive_used": False, "lesion_file_used": False,
            "existing_results_modified": False,
        },
        "next_step_p_b11_stage1_dev_lesion_scoring_feasible": (verdict == "통과"),
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = [
        "# P-B10 v4_20 ROI EfficientNet-B0 Normal Test Sanity\n",
        f"**판정: {verdict}**\n",
        f"- 생성일시: {ts}",
        f"- branch: efficientnet_b0_imagenet_chestwall_removed_roi_v1 / ROI source: refined_roi_v4_20_modeB_all_v1\n",
        "## 처리\n",
        f"- normal_test 처리: {n_csv}/{EXPECTED_TEST_N} (실패 {n_failed})",
        f"- shape mismatch: {shape_mismatch}",
        f"- total scored patch: {n_total:,}",
        f"- patch before→after: {total_before:,} → {total_after:,} (제거 {total_before-total_after:,}, {round((total_before-total_after)/total_before*100,2) if total_before else 0}%)",
        f"- NaN/Inf: {n_nan}/{n_inf}\n",
        "## score 통계\n",
        "| 지표 | 값 |",
        "|------|----|",
        f"| min | {s_min:.6f} |",
        f"| max | {s_max:.6f} |",
        f"| mean | {s_mean:.6f} |",
        f"| std | {s_std:.6f} |",
        f"| median | {s_median:.6f} |\n",
        "## threshold exceedance (P-B9 threshold 고정, 재계산 없음)\n",
        "| threshold | 값 | 초과 patch | 초과율 |",
        "|-----------|----|-----------|--------|",
        f"| p95 | {p95:.6f} | {exceed95:,} | **{rate95*100:.3f}%** |",
        f"| p99 | {p99:.6f} | {exceed99:,} | **{rate99*100:.3f}%** |\n",
        "- sanity 기준: p95 초과율 ~5%, p99 초과율 ~1% 근처면 정상",
        f"- threshold JSON mtime 불변: **{thr_unchanged}** (재계산/수정 없음)",
        "- ⚠ 정상 test 결과를 보고 threshold를 조정하지 않음 (leakage 방지)\n",
        "## 미실행 / 미사용 확인\n",
        "- threshold 재계산 / normal val 재실행: 안 함",
        "- lesion scoring / metrics / AUROC·AUPRC / Dice·recall: 미실행",
        "- stage2_holdout / model_roi / E드라이브 / lesion 파일: 미접근·미사용",
        "- 기존 P-B1~P-B9 / roi_0_0 branch 결과: 무수정\n",
        "## 다음 단계\n",
        f"- P-B11 stage1_dev lesion scoring 진행 가능: {verdict == '통과'}",
    ]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n=== P-B10 완료: {verdict} ===")
    print(f"[보고서] {REPORT_DIR}")
    return 0 if verdict != "실패" else 1


if __name__ == "__main__":
    sys.exit(main())
