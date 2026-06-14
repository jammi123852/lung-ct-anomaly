"""
P-B7: v4_20 ROI EfficientNet-B0 normal_train 290명 full train

v4_20 흉벽 제거 ROI branch의 정상 분포 full train.
- CT: roi_0_0 ct_hu.npy (C드라이브, v4_20 refined ROI와 동일 좌표계)
- ROI: refined_roi_v4_20_modeB_all_v1/normal/<safe_id>/refined_roi.npy (v4_20 lock)
- patch 재필터링: v4_20 ROI ratio >= 0.5 (기존 roi_0_0 threshold 동일)
- 출력: outputs/models/distributions/position_bin_stats.npz (full, smoke와 분리)

실행: --full-run 플래그 필수. 없으면 dry-check만 수행 후 중단.
금지: val·test scoring / lesion scoring / threshold / metrics / stage2_holdout / model_roi / E드라이브 / lesion 파일
"""
from __future__ import annotations

import argparse
import csv
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
SCRIPT_NAME         = "p_b7_full_train.py"
EXPECTED_TRAIN_N    = 290

V4_20_PATCH_RATIO_THRESHOLD = 0.5
V4_20_NORMAL_ROOT = PROJ_ROOT / "outputs" / "mip-postprocess-research-v1" / "masks" / "refined_roi_v4_20_modeB_all_v1" / "normal"
NORMAL_CT_ROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")

# 출력 (full, smoke와 분리)
SELECTED_INDICES_PATH = EXP_ROOT / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
SELECTED_INDICES_SRC  = ROI0_BRANCH / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
FULL_NPZ              = EXP_ROOT / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
SMOKE_NPZ_L1          = EXP_ROOT / "outputs" / "smoke" / "train_limit1" / "position_bin_stats.npz"
SMOKE_NPZ_L5          = EXP_ROOT / "outputs" / "smoke" / "train_limit5" / "position_bin_stats.npz"
REPORTS_DIR           = EXP_ROOT / "outputs" / "reports" / "full"
ERROR_CSV             = REPORTS_DIR / "error.csv"
RUNTIME_CSV           = REPORTS_DIR / "p_b7_runtime_summary.csv"
PATCH_FILTER_CSV      = REPORTS_DIR / "p_b7_patch_filtering_summary.csv"
FULL_LOG              = REPORTS_DIR / "full_train.log"

SPLIT_JSON  = PROJ_ROOT / "outputs" / "position-aware-padim-v1" / "splits" / "normal_v1.json"
P_B6_JSON   = REPORTS_DIR / "p_b6_full_train_preflight.json"
P_B5_JSON   = EXP_ROOT / "outputs" / "reports" / "smoke" / "p_b5_train_smoke_limit5.json"
P_B4_JSON   = EXP_ROOT / "outputs" / "reports" / "smoke" / "p_b4_train_smoke_limit1.json"
P_B3_JSON   = EXP_ROOT / "outputs" / "reports" / "p_b3_lesion_safety_validation" / "p_b3_lesion_safety_validation.json"
P_B2_6_JSON = EXP_ROOT / "outputs" / "reports" / "p_b2_6_v4_20_source_lock" / "p_b2_6_v4_20_source_lock.json"

RUNTIME_COLUMNS = ["timestamp", "script", "metric", "value"]
PATCH_FILTER_COLUMNS = ["patient_id", "safe_id", "ct_shape", "roi_shape", "shape_match",
                        "roi_voxel", "patch_before", "patch_after", "patch_removed", "removed_ratio"]


def log_line(msg, do_log):
    print(msg)
    if do_log:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(FULL_LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def run_guards(do_log):
    """G1~G24 입력/경로/가드 검증. dry-check와 full-run 공통. (forward/training 없음)"""
    result = {"issues": [], "abort": False}

    def fail(msg, hard=True):
        result["issues"].append(msg)
        if hard:
            result["abort"] = True

    # G2~G6: 선행 verdict
    p_b6 = json.load(open(P_B6_JSON, encoding="utf-8")) if P_B6_JSON.exists() else None
    p_b5 = json.load(open(P_B5_JSON, encoding="utf-8")) if P_B5_JSON.exists() else None
    p_b4 = json.load(open(P_B4_JSON, encoding="utf-8")) if P_B4_JSON.exists() else None
    p_b3 = json.load(open(P_B3_JSON, encoding="utf-8")) if P_B3_JSON.exists() else None
    p_b26 = json.load(open(P_B2_6_JSON, encoding="utf-8")) if P_B2_6_JSON.exists() else None

    if not (p_b6 and p_b6.get("verdict") == "통과"):
        fail("G2: P-B6 verdict != 통과")
    if not (p_b5 and p_b5.get("verdict") == "통과"):
        fail("G3: P-B5 verdict != 통과")
    if not (p_b4 and p_b4.get("verdict") == "통과"):
        fail("G4: P-B4 verdict != 통과")
    if not (p_b3 and p_b3.get("verdict") in ("통과", "부분통과")):
        fail("G5: P-B3 verdict not in (통과,부분통과)")
    if not (p_b26 and p_b26.get("user_correction_applied", {}).get("official_roi_source") == "refined_roi_v4_20_modeB_all_v1"):
        fail("G6: P-B2.6 source lock != v4_20")
    log_line(f"[G2-6] 선행 verdict: P-B6={p_b6.get('verdict') if p_b6 else None}, "
             f"P-B5={p_b5.get('verdict') if p_b5 else None}, P-B4={p_b4.get('verdict') if p_b4 else None}, "
             f"P-B3={p_b3.get('verdict') if p_b3 else None}, "
             f"P-B2.6 lock={'v4_20' if (p_b26 and not result['abort']) else '?'}", do_log)

    # G7: selected index
    sidx_info = {}
    if SELECTED_INDICES_PATH.exists():
        sidx = np.load(str(SELECTED_INDICES_PATH))
        ok = (sidx.shape == (REDUCED_FEATURE_DIM,) and len(np.unique(sidx)) == REDUCED_FEATURE_DIM
              and sidx.min() >= 0 and sidx.max() <= 143)
        sidx_info = {"shape": list(sidx.shape), "unique": int(len(np.unique(sidx))),
                     "range": [int(sidx.min()), int(sidx.max())]}
        if not ok:
            fail("G7: selected index 검증 실패")
    else:
        fail("G7: selected index 파일 없음")
    log_line(f"[G7] selected index: {sidx_info}", do_log)

    # G8~G9: train split 290 + normal004 test
    split_data = json.load(open(SPLIT_JSON, encoding="utf-8"))
    train_patients = list(split_data["train"])
    test_patients  = list(split_data.get("test", []))
    p2s = split_data.get("patient_to_safe_id", {})
    if len(train_patients) != EXPECTED_TRAIN_N:
        fail(f"G8: train split {len(train_patients)}≠{EXPECTED_TRAIN_N}")
    normal004_in_test = "normal004" in test_patients
    log_line(f"[G8] train split {len(train_patients)}명, [G9] normal004 in test={normal004_in_test} "
             f"(train 제외 정상)", do_log)

    # G10~G11: CT/ROI 존재 + shape
    ct_missing, roi_missing, shape_mismatch = [], [], []
    for pid in train_patients:
        safe_id = p2s.get(pid, pid)
        ct_path  = NORMAL_CT_ROOT / safe_id / "ct_hu.npy"
        roi_path = V4_20_NORMAL_ROOT / safe_id / "refined_roi.npy"
        ce, re = ct_path.exists(), roi_path.exists()
        if not ce: ct_missing.append(pid)
        if not re: roi_missing.append(pid)
        if ce and re:
            cs = np.load(str(ct_path), mmap_mode='r').shape
            rs = np.load(str(roi_path), mmap_mode='r').shape
            if cs != rs:
                shape_mismatch.append(pid)
    if ct_missing: fail(f"G10: CT 누락 {len(ct_missing)}")
    if roi_missing: fail(f"G10: ROI 누락 {len(roi_missing)}")
    if shape_mismatch: fail(f"G11: shape mismatch {len(shape_mismatch)}")
    log_line(f"[G10-11] CT 누락={len(ct_missing)}, ROI 누락={len(roi_missing)}, shape mismatch={len(shape_mismatch)}", do_log)

    # G12~G13: full output collision / smoke-full 분리
    if FULL_NPZ.exists():
        fail("G12: full output npz 이미 존재 (덮어쓰기 금지)")
    if FULL_NPZ == SMOKE_NPZ_L1 or FULL_NPZ == SMOKE_NPZ_L5:
        fail("G13: smoke/full 경로 충돌")
    log_line(f"[G12-13] full npz 존재={FULL_NPZ.exists()}, smoke/full 분리=True", do_log)

    # G14: smoke 보존
    smoke_ok = SMOKE_NPZ_L1.exists() and SMOKE_NPZ_L5.exists()
    log_line(f"[G14] smoke limit1/5 보존: {smoke_ok}", do_log)

    # G17~G20: 미사용 선언 (코드 경로상 보장)
    log_line("[G17-20] model_roi/E드라이브/lesion/stage2_holdout 미사용 (코드 경로상 미참조)", do_log)

    result.update({
        "p_b6": p_b6, "p_b5": p_b5, "p_b4": p_b4, "p_b3": p_b3, "p_b26": p_b26,
        "sidx_info": sidx_info,
        "train_patients": train_patients, "p2s": p2s,
        "normal004_in_test": normal004_in_test,
        "ct_missing": ct_missing, "roi_missing": roi_missing, "shape_mismatch": shape_mismatch,
        "smoke_preserved": smoke_ok,
    })
    return result


def record_runtime(rows):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_CSV.exists() or RUNTIME_CSV.stat().st_size == 0
    with open(RUNTIME_CSV, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def save_patch_filter_csv(rows):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PATCH_FILTER_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PATCH_FILTER_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def run_full_train(guard_result):
    """실제 full train. --full-run일 때만 호출."""
    train_patients = guard_result["train_patients"]
    p2s = guard_result["p2s"]

    from position_aware_padim.config_manager import ConfigManager
    from position_aware_padim.data_loader import DataLoader
    from position_aware_padim.path_resolver import PathResolver
    from position_aware_padim.padim_model import PaDiMModel
    from position_aware_padim.feature_extractor_effnet_b0_scaffold import FeatureExtractorEffNetB0

    cfg = ConfigManager(str(PROJ_ROOT))
    cfg.load_config(paths_yaml=PATHS_CONFIG)
    normal_training_ready = cfg.get("paths", "normal_training_ready", "")
    manifest_path = Path(normal_training_ready) / "manifests" / "patient_manifest.csv"

    path_resolver = PathResolver(str(manifest_path), normal_training_ready)
    loader = DataLoader(str(manifest_path), path_resolver, str(ERROR_CSV), use_mmap=True)

    log_line("[full] FeatureExtractorEffNetB0 초기화...", True)
    feature_extractor = FeatureExtractorEffNetB0()
    assert feature_extractor.raw_feature_dim == RAW_FEATURE_DIM
    log_line(f"[full] device={feature_extractor.device}, raw_feature_dim={feature_extractor.raw_feature_dim}", True)

    import torch
    gpu_avail = (feature_extractor.device == "cuda")
    if gpu_avail:
        torch.cuda.reset_peak_memory_stats()

    model = PaDiMModel(
        selected_feature_indices_path=str(SELECTED_INDICES_PATH),
        feature_dim=REDUCED_FEATURE_DIM, eps=1e-5,
    )

    per_patient_rows = []
    shape_mismatch_count = 0

    def patient_stream():
        nonlocal shape_mismatch_count
        for idx, pid in enumerate(train_patients):
            safe_id = p2s.get(pid, pid)
            data = loader.load_patient_data(pid, mask_type=MASK_TYPE)
            if data is None:
                log_line(f"  [SKIP] {pid}: 로드 실패", True)
                continue
            ct_hu = data["ct_hu"]
            roi_path = V4_20_NORMAL_ROOT / safe_id / "refined_roi.npy"
            if not roi_path.exists():
                log_line(f"  [SKIP] {pid}: v4_20 ROI 없음", True)
                continue
            refined_roi = np.load(str(roi_path), mmap_mode='r')
            if refined_roi.shape != ct_hu.shape:
                shape_mismatch_count += 1
                log_line(f"  [SKIP] {pid}: shape mismatch", True)
                continue

            roi_voxel = int(np.asarray(refined_roi).sum())
            patch_df = data["patch_df"]
            n_before = len(patch_df)
            keep = []
            for r in patch_df.itertuples(index=False):
                z = int(r.local_z)
                if z < 0 or z >= refined_roi.shape[0]:
                    keep.append(False); continue
                sub = np.asarray(refined_roi[z, int(r.y0):int(r.y1), int(r.x0):int(r.x1)])
                ratio = float(sub.mean()) if sub.size > 0 else 0.0
                keep.append(ratio >= V4_20_PATCH_RATIO_THRESHOLD)
            keep = np.array(keep, dtype=bool)
            patch_df_v4 = patch_df[keep].reset_index(drop=True)
            n_after = len(patch_df_v4)
            removed = n_before - n_after
            per_patient_rows.append({
                "patient_id": pid, "safe_id": safe_id,
                "ct_shape": str(tuple(ct_hu.shape)), "roi_shape": str(tuple(refined_roi.shape)),
                "shape_match": True, "roi_voxel": roi_voxel,
                "patch_before": n_before, "patch_after": n_after,
                "patch_removed": removed,
                "removed_ratio": round(removed / n_before, 6) if n_before > 0 else 0.0,
            })
            if (idx + 1) % 20 == 0 or idx == 0:
                log_line(f"  [{idx+1}/{len(train_patients)}] {pid}: patch {n_before:,}→{n_after:,}", True)

            data["mask"] = np.asarray(refined_roi)
            data["patch_df"] = patch_df_v4
            yield data

    start = time.time()
    log_line("\n[full] full train 시작...", True)
    model.train(feature_extractor, patient_stream(), split=None)
    elapsed = time.time() - start

    peak_gpu_gb = 0.0
    if gpu_avail:
        peak_gpu_gb = torch.cuda.max_memory_allocated() / 1e9

    summary = model.train_summary
    assert model.stats, "stats 비어있음"
    FULL_NPZ.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(FULL_NPZ))
    log_line(f"[full] 저장: {FULL_NPZ}", True)

    n_bins_with_data = total_nan = total_inf = 0
    sample_mean_shape = sample_cov_shape = None
    for key in model._all_keys():
        s = model.stats.get(key, {})
        if s.get("count", 0) > 0 and "mean" in s:
            mean = s["mean"]; cov = s.get("cov", np.array([]))
            total_nan += int(np.isnan(mean).sum()) + (int(np.isnan(cov).sum()) if cov.size else 0)
            total_inf += int(np.isinf(mean).sum()) + (int(np.isinf(cov).sum()) if cov.size else 0)
            n_bins_with_data += 1
            if sample_mean_shape is None:
                sample_mean_shape = mean.shape; sample_cov_shape = cov.shape

    assert sample_mean_shape == (REDUCED_FEATURE_DIM,)
    assert sample_cov_shape == (REDUCED_FEATURE_DIM, REDUCED_FEATURE_DIM)
    assert total_nan == 0 and total_inf == 0

    ok_rows = [r for r in per_patient_rows if r["patch_before"] is not None]
    total_before = sum(r["patch_before"] for r in ok_rows)
    total_after  = sum(r["patch_after"] for r in ok_rows)
    save_patch_filter_csv(per_patient_rows)

    ts = datetime.now().isoformat(timespec="seconds")
    record_runtime([
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_processed", "value": summary["n_patients_success"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "total_patch_before", "value": total_before},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "total_patch_after", "value": total_after},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patches_used", "value": summary["n_patches_used"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patches_skipped", "value": summary["n_patches_skipped"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "position_bins_with_data", "value": n_bins_with_data},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "elapsed_seconds", "value": round(elapsed, 2)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "peak_gpu_gb", "value": round(peak_gpu_gb, 3)},
    ])

    verdict = "통과" if (shape_mismatch_count == 0 and summary["n_patients_success"] == EXPECTED_TRAIN_N) else "부분통과"
    report = {
        "step": "P-B7", "verdict": verdict, "timestamp": ts,
        "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        "backbone": BACKBONE, "official_roi_source": "refined_roi_v4_20_modeB_all_v1",
        "n_patients_processed": summary["n_patients_success"],
        "n_patients_failed": summary["n_patients_failed"],
        "shape_mismatch": shape_mismatch_count,
        "total_patch_before": total_before, "total_patch_after": total_after,
        "total_patch_removed": total_before - total_after,
        "total_removed_ratio": round((total_before - total_after) / total_before, 6) if total_before else 0.0,
        "raw_feature_dim": RAW_FEATURE_DIM, "selected_feature_dim": REDUCED_FEATURE_DIM,
        "mean_shape": list(sample_mean_shape), "cov_shape": list(sample_cov_shape),
        "position_bins_with_data": n_bins_with_data,
        "n_patches_used": summary["n_patches_used"], "n_patches_skipped": summary["n_patches_skipped"],
        "total_nan": total_nan, "total_inf": total_inf,
        "elapsed_seconds": round(elapsed, 2), "peak_gpu_gb": round(peak_gpu_gb, 3),
        "output_npz": str(FULL_NPZ),
        "safety": {
            "val_test_scoring": False, "lesion_scoring": False,
            "threshold_calculated": False, "metrics_calculated": False,
            "stage2_holdout_accessed": False, "model_roi_used": False,
            "e_drive_used": False, "lesion_file_used": False,
            "existing_results_modified": False,
        },
        "next_step_p_b8_distribution_validation_feasible": (verdict == "통과"),
    }
    with open(REPORTS_DIR / "p_b7_full_train.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    md = [
        "# P-B7 v4_20 ROI EfficientNet-B0 Full Train\n",
        f"**판정: {verdict}**\n",
        f"- 생성일시: {ts}",
        f"- 처리 환자: {summary['n_patients_success']}/{EXPECTED_TRAIN_N}",
        f"- total patch: {total_before:,} → {total_after:,} (제거 {total_before-total_after:,})",
        f"- used patch: {summary['n_patches_used']:,} / skipped: {summary['n_patches_skipped']:,}",
        f"- mean/cov shape: {sample_mean_shape} / {sample_cov_shape}",
        f"- position_bin with data: {n_bins_with_data}, NaN/Inf: {total_nan}/{total_inf}",
        f"- elapsed: {round(elapsed,2)}초, peak GPU: {round(peak_gpu_gb,3)} GB",
        f"- output: {FULL_NPZ}\n",
        "## 다음 단계\n",
        f"- P-B8 distribution validation 가능: {verdict == '통과'}",
    ]
    with open(REPORTS_DIR / "p_b7_full_train.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    log_line(f"\n=== P-B7 full train 완료: {verdict} ===", True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run", action="store_true", help="실제 full train 실행")
    args = parser.parse_args()

    print(f"[{SCRIPT_NAME}] {'FULL-RUN' if args.full_run else 'DRY-CHECK'} 모드, {datetime.now().isoformat(timespec='seconds')}\n")

    # 공통 가드 (forward/training 없음)
    gr = run_guards(do_log=args.full_run)

    # ── G1: --full-run 없으면 dry-check 보고서 작성 후 중단 ───────────────
    if not args.full_run:
        verdict = "실패" if gr["abort"] else ("부분통과" if (gr["ct_missing"] or gr["roi_missing"] or gr["shape_mismatch"] or FULL_NPZ.exists()) else "통과")
        print(f"\n[dry-check] G1: --full-run 미입력 → full train 실행 안 함")
        print(f"[dry-check] 판정: {verdict}")
        for i in gr["issues"]:
            print(f"  ⚠ {i}")

        p_b7b_cmd = ("source ~/ai_env/bin/activate && python "
                     "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/code/p_b7_full_train.py --full-run")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        drycheck = {
            "stage": "P-B7a_full_train_script_drycheck",
            "created": datetime.now().isoformat(timespec="seconds"),
            "verdict": verdict,
            "script_path": str(EXP_ROOT / "code" / SCRIPT_NAME),
            "mode": "dry-check (--full-run 미입력)",
            "scope": {
                "full_train_executed": False, "model_forward": False,
                "feature_extraction": False, "padim_update": False,
                "val_test_scoring": False, "lesion_scoring": False,
                "threshold_calculated": False, "metrics_calculated": False,
                "stage2_holdout_accessed": False, "model_roi_used": False,
                "e_drive_used": False, "lesion_file_used": False,
                "existing_results_modified": False,
            },
            "input_validation": {
                "p_b6_verdict": gr["p_b6"].get("verdict") if gr["p_b6"] else None,
                "p_b5_verdict": gr["p_b5"].get("verdict") if gr["p_b5"] else None,
                "p_b4_verdict": gr["p_b4"].get("verdict") if gr["p_b4"] else None,
                "p_b3_verdict": gr["p_b3"].get("verdict") if gr["p_b3"] else None,
                "p_b2_6_source_locked": gr["p_b26"] is not None and gr["p_b26"].get("user_correction_applied", {}).get("official_roi_source") == "refined_roi_v4_20_modeB_all_v1",
                "official_roi_source": "refined_roi_v4_20_modeB_all_v1",
            },
            "selected_index": gr["sidx_info"],
            "normal_train_count": len(gr["train_patients"]),
            "normal004_in_test": gr["normal004_in_test"],
            "normal004_note": "normal004는 test split 소속 → train 290에 없음 (정상)",
            "ct_missing": len(gr["ct_missing"]),
            "roi_missing": len(gr["roi_missing"]),
            "shape_mismatch": len(gr["shape_mismatch"]),
            "full_output_collision": FULL_NPZ.exists(),
            "smoke_full_distinct": FULL_NPZ != SMOKE_NPZ_L1 and FULL_NPZ != SMOKE_NPZ_L5,
            "smoke_preserved": gr["smoke_preserved"],
            "expected_full_outputs": [
                str(FULL_NPZ),
                str(REPORTS_DIR / "p_b7_full_train.md"),
                str(REPORTS_DIR / "p_b7_full_train.json"),
                str(RUNTIME_CSV),
                str(PATCH_FILTER_CSV),
                str(FULL_LOG),
            ],
            "p_b7b_run_command": p_b7b_cmd,
            "p_b7b_full_train_feasible": (verdict == "통과"),
            "issues": gr["issues"],
        }
        with open(REPORTS_DIR / "p_b7a_full_train_script_drycheck.json", "w", encoding="utf-8") as f:
            json.dump(drycheck, f, ensure_ascii=False, indent=2)

        md = [
            "# P-B7a v4_20 Full Train Script Dry-Check\n",
            f"**판정: {verdict}**\n",
            f"- 생성일시: {drycheck['created']}",
            f"- 스크립트: `{drycheck['script_path']}`",
            "- 모드: dry-check (--full-run 미입력 → full train 실행 안 함)\n",
            "## 입력 검증\n",
            "| 단계 | verdict |",
            "|------|---------|",
            f"| P-B6 | {drycheck['input_validation']['p_b6_verdict']} |",
            f"| P-B5 | {drycheck['input_validation']['p_b5_verdict']} |",
            f"| P-B4 | {drycheck['input_validation']['p_b4_verdict']} |",
            f"| P-B3 | {drycheck['input_validation']['p_b3_verdict']} |",
            f"| P-B2.6 source lock | {'v4_20 ✅' if drycheck['input_validation']['p_b2_6_source_locked'] else 'NG'} |\n",
            f"- selected index: {gr['sidx_info']}",
            f"- normal train: {len(gr['train_patients'])}명",
            f"- normal004 in test: {gr['normal004_in_test']} (train 제외 정상)\n",
            "## 존재/경로 검증\n",
            "| 항목 | 결과 |",
            "|------|------|",
            f"| CT 누락 | {len(gr['ct_missing'])} |",
            f"| v4_20 ROI 누락 | {len(gr['roi_missing'])} |",
            f"| shape mismatch | {len(gr['shape_mismatch'])} |",
            f"| full output collision | {FULL_NPZ.exists()} |",
            f"| smoke/full 경로 분리 | {drycheck['smoke_full_distinct']} |",
            f"| smoke limit1/5 보존 | {gr['smoke_preserved']} |\n",
            "## 미실행 / 미사용 확인\n",
            "- full train / forward / feature extraction / PaDiM update: 미실행",
            "- val/test scoring / lesion scoring / threshold / metrics: 미실행",
            "- model_roi / E드라이브 / lesion 파일 / stage2_holdout: 미사용",
            "- 기존 roi_0_0 / P-B1~P-B6 결과: 무수정, smoke 보존\n",
            "## full train 생성 예정 파일 (P-B7b)\n",
        ]
        for o in drycheck["expected_full_outputs"]:
            md.append(f"- `{o}`")
        md += [
            "",
            "## P-B7b 실행 명령 초안\n",
            "```bash",
            p_b7b_cmd,
            "```\n",
            "## 미결 사항\n",
        ]
        for i in gr["issues"]:
            md.append(f"- ⚠ {i}")
        if not gr["issues"]:
            md.append("- 없음")
        md += [
            "",
            "## 최종 판정\n",
            f"- **{verdict}**",
            f"- P-B7b full train 실행 가능: **{verdict == '통과'}**",
        ]
        with open(REPORTS_DIR / "p_b7a_full_train_script_drycheck.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")
        print(f"\n[dry-check] 보고서 저장: {REPORTS_DIR}")
        sys.exit(0 if verdict != "실패" else 1)

    # ── --full-run: 가드 abort면 중단 ────────────────────────────────────
    if gr["abort"]:
        print("[full-run][ABORT] 가드 실패로 중단")
        for i in gr["issues"]:
            print(f"  ⚠ {i}")
        sys.exit(1)
    run_full_train(gr)


if __name__ == "__main__":
    main()
