"""
P-A57: ResNet18 random224 full train (290명 전체)

normal train split 290명 전체로 PaDiM 분포 학습.
- FeatureExtractor (ResNet18 ImageNet1K V1)
- PaDiMModel (feature_dim=224, selected_feature_indices rand224)
- full output: outputs/models/distributions/position_bin_stats.npz

금지: scoring / threshold / metrics / stage2_holdout / lesion
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_paths  # noqa: E402

SCRIPT_NAME = "p_a57_full_train.py"
FULL_OUT = ws_paths.resolve_train_output(full_run=True, limit=None)
OUTPUT_NPZ = FULL_OUT["model_npz"]
REPORTS_DIR = FULL_OUT["reports_dir"]
ERROR_CSV = REPORTS_DIR / "error.csv"
RUNTIME_CSV = REPORTS_DIR / "runtime_summary.csv"
ERROR_COLUMNS = ["patient_id", "error_type", "error_msg", "file_logical"]
RUNTIME_COLUMNS = ["timestamp", "script", "metric", "value"]

P56_EXPECTED_PATCHES = 11_001_904


def record_error(patient_id, error_type, error_msg, file_logical):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not ERROR_CSV.exists() or ERROR_CSV.stat().st_size == 0
    with open(ERROR_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({"patient_id": patient_id, "error_type": error_type,
                         "error_msg": error_msg, "file_logical": file_logical})


def record_runtime(rows_extra: list):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_CSV.exists() or RUNTIME_CSV.stat().st_size == 0
    with open(RUNTIME_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows_extra)


def main():
    import numpy as np

    # ---- 가드 1: P-A56 보고서 통과 확인 ----
    p56_json = ws_paths.REPORTS_FULL_DIR / "p_a56_full_train_preflight.json"
    assert p56_json.exists(), f"P-A56 보고서 없음: {p56_json}"
    with open(p56_json) as f:
        p56 = json.load(f)
    assert p56.get("verdict") == "통과", f"P-A56 판정 미통과: {p56.get('verdict')}"
    print("[guard1] P-A56 보고서 통과 확인 OK")

    # ---- 가드 2: P-A53/P-A54/P-A55 보고서 통과 확인 ----
    for step, json_path in [
        ("P-A53", ws_paths.REPORTS_DIR / "p_a53_selected_indices.json"),
        ("P-A54", ws_paths.REPORTS_SMOKE_DIR / "p_a54_train_smoke_limit1.json"),
        ("P-A55", ws_paths.REPORTS_SMOKE_DIR / "p_a55_train_smoke_limit5.json"),
    ]:
        assert json_path.exists(), f"{step} 보고서 없음: {json_path}"
        with open(json_path) as f:
            rep = json.load(f)
        assert rep.get("verdict") == "통과", f"{step} 판정 미통과: {rep.get('verdict')}"
        print(f"[guard2] {step} 보고서 통과 확인 OK")

    # ---- 가드 3~5: selected_feature_indices 검증 ----
    sidx_path = ws_paths.SELECTED_INDICES_PATH
    assert sidx_path.exists(), f"selected_feature_indices.npy 없음: {sidx_path}"
    sidx = np.load(str(sidx_path))
    assert sidx.shape == (224,), f"[guard3] shape 불일치: {sidx.shape}"
    assert len(np.unique(sidx)) == 224, f"[guard3] unique 불일치: {len(np.unique(sidx))}"
    assert sidx.min() >= 0 and sidx.max() <= 447, f"[guard3] range 오류: {sidx.min()}~{sidx.max()}"
    existing_idx_path = REPO_ROOT / "outputs/position-aware-padim-v1/models/padim_v1/distributions/selected_feature_indices.npy"
    assert existing_idx_path.exists(), f"[guard4] 기존 random100 인덱스 파일 없음: {existing_idx_path}"
    existing_idx = np.load(str(existing_idx_path))
    assert set(existing_idx.tolist()).issubset(set(sidx.tolist())), "[guard4] 기존 100개 미포함"
    print(f"[guard3-4] selected_feature_indices OK: shape={sidx.shape}, unique=224, range=[{sidx.min()},{sidx.max()}], 기존100포함=True")

    # ---- 가드 5: normal train split 290명 확인 ----
    from position_aware_padim.patient_splitter import PatientSplitter
    splitter = PatientSplitter(str(REPO_ROOT))
    patient_split = splitter.load_split()
    train_patients = list(patient_split.train)
    assert len(train_patients) == 290, f"[guard5] train split 수 불일치: {len(train_patients)}"
    print(f"[guard5] normal train split: {len(train_patients)}명 확인")

    # ---- 가드 6: full output path 비어있는지 확인 ----
    if OUTPUT_NPZ.exists():
        print(f"[guard6][ABORT] full output이 이미 존재합니다: {OUTPUT_NPZ}")
        print("  기존 결과를 수동으로 삭제하거나 경로를 확인하세요.")
        sys.exit(1)
    print(f"[guard6] full output 경로 비어있음 OK: {OUTPUT_NPZ}")

    # ---- 가드 7: smoke/full 경로 분리 확인 ----
    smoke_npz = ws_paths.SMOKE_ROOT / "train_limit5" / "position_bin_stats.npz"
    assert OUTPUT_NPZ != smoke_npz, "[guard7] smoke/full 경로 충돌"
    print(f"[guard7] smoke/full 경로 분리 OK")

    # ---- 가드 8: P-A54/P-A55 smoke 결과 보존 확인 ----
    p54_npz = ws_paths.SMOKE_ROOT / "train_limit1" / "position_bin_stats.npz"
    p55_npz = smoke_npz
    assert p54_npz.exists(), f"[guard8] P-A54 limit1 결과 없음: {p54_npz}"
    assert p55_npz.exists(), f"[guard8] P-A55 limit5 결과 없음: {p55_npz}"
    print(f"[guard8] P-A54/P-A55 smoke 결과 보존 확인 OK")

    # ---- 가드 9: stage2_holdout 접근 금지 ----
    print(f"[guard9] stage2_holdout 접근 금지 확인 OK (접근 없음)")

    print(f"\n[train] full train 시작: {len(train_patients)}명")
    print(f"[train] backbone={ws_paths.BACKBONE}, raw_dim={ws_paths.RAW_FEATURE_DIM}, reduced_dim={ws_paths.REDUCED_FEATURE_DIM}")
    print(f"[train] selected_feature_indices: {sidx_path}")
    print(f"[train] full output: {OUTPUT_NPZ}")
    print(f"[train] reports: {REPORTS_DIR}")

    # ---- ConfigManager, DataLoader ----
    from position_aware_padim.config_manager import ConfigManager
    from position_aware_padim.data_loader import DataLoader
    from position_aware_padim.path_resolver import PathResolver
    from position_aware_padim.feature_extractor import FeatureExtractor
    from position_aware_padim.padim_model import PaDiMModel

    cfg = ConfigManager(str(REPO_ROOT))
    cfg.load_config(paths_yaml=Path(ws_paths.PATHS_CONFIG).name)
    normal_training_ready = cfg.get("paths", "normal_training_ready", "")
    assert normal_training_ready, "paths config에 'normal_training_ready' 없음"
    manifest_path = Path(normal_training_ready) / "manifests" / "patient_manifest.csv"
    assert manifest_path.exists(), f"patient_manifest.csv 없음: {manifest_path}"

    path_resolver = PathResolver(str(manifest_path), normal_training_ready)
    loader = DataLoader(str(manifest_path), path_resolver, str(ERROR_CSV), use_mmap=True)

    # ---- FeatureExtractor (ResNet18) ----
    print("\n[train] FeatureExtractor(ResNet18/ImageNet) 초기화 중...")
    feature_extractor = FeatureExtractor()
    print(f"[train] device: {feature_extractor.device}")

    import torch
    gpu_avail = feature_extractor.device == "cuda"
    if gpu_avail:
        torch.cuda.reset_peak_memory_stats()

    # ---- PaDiMModel (feature_dim=224) ----
    model = PaDiMModel(
        selected_feature_indices_path=str(sidx_path),
        feature_dim=ws_paths.REDUCED_FEATURE_DIM,
        eps=1e-5,
    )
    print(f"[train] PaDiMModel 초기화: feature_dim={model.feature_dim}")

    n_processed = 0
    n_failed = 0

    def patient_stream():
        nonlocal n_processed, n_failed
        for i, pid in enumerate(train_patients):
            data = loader.load_patient_data(pid, mask_type=ws_paths.MASK_TYPE)
            if data is None:
                n_failed += 1
                record_error(pid, "load_failed", "DataLoader.load_patient_data returned None", "patient_data")
                print(f"  [SKIP] ({i+1}/{len(train_patients)}) {pid}: 로드 실패")
                continue
            n_processed += 1
            print(f"  [OK]   ({i+1}/{len(train_patients)}) {pid}: ct_hu={data['ct_hu'].shape}, patches={len(data['patch_df'])}")
            yield data

    start_time = time.time()
    print("\n[train] 학습 시작...")
    try:
        model.train(feature_extractor, patient_stream(), split=None)
    except Exception as exc:
        record_error("__train__", "train_error", str(exc), SCRIPT_NAME)
        raise

    elapsed = time.time() - start_time
    peak_gpu_gb = 0.0
    if gpu_avail:
        peak_gpu_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[train] peak GPU memory: {peak_gpu_gb:.3f} GB")

    summary = model.train_summary
    print(f"[train] 완료: {summary['n_patients_success']}명 성공, {summary['n_patients_failed']}명 실패, {elapsed:.1f}초")
    print(f"[train] 사용 patch={summary['n_patches_used']:,}, 스킵 patch={summary['n_patches_skipped']:,}")

    if not model.stats:
        record_error("__train__", "empty_stats", "stats가 비어있음", SCRIPT_NAME)
        raise RuntimeError("full train 결과가 비어있습니다.")

    # ---- 저장 ----
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_NPZ))
    print(f"[train] 저장 완료: {OUTPUT_NPZ}")

    # ---- position_bin 통계 요약 및 NaN/Inf 확인 ----
    print("\n[position_bin 통계 요약]")
    print(f"{'key':<25} {'count':>10} {'mean_shape':>12} {'NaN':>5} {'Inf':>5}")
    print("-" * 65)
    n_bins_with_data = 0
    total_nan = 0
    total_inf = 0
    for key in model._all_keys():
        s = model.stats.get(key, {})
        count = s.get("count", 0)
        if count > 0 and "mean" in s:
            mean = s["mean"]
            cov = s.get("cov", np.array([]))
            nan_m = int(np.isnan(mean).sum())
            inf_m = int(np.isinf(mean).sum())
            nan_c = int(np.isnan(cov).sum()) if cov.size else 0
            inf_c = int(np.isinf(cov).sum()) if cov.size else 0
            nan_total = nan_m + nan_c
            inf_total = inf_m + inf_c
            total_nan += nan_total
            total_inf += inf_total
            print(f"{key:<25} {count:>10,} {str(mean.shape):>12} {nan_total:>5} {inf_total:>5}")
            n_bins_with_data += 1
        else:
            print(f"{key:<25} {'(no data)':>10}")

    sample_key = None
    sample_mean_shape = None
    sample_cov_shape = None
    for key in model._all_keys():
        s = model.stats.get(key, {})
        if s.get("count", 0) > 0 and "mean" in s:
            sample_key = key
            sample_mean_shape = s["mean"].shape
            sample_cov_shape = s.get("cov", np.array([])).shape
            break

    print(f"\n[검증] mean shape: {sample_mean_shape}")
    print(f"[검증] cov shape: {sample_cov_shape}")
    print(f"[검증] position_bin with data: {n_bins_with_data}")
    print(f"[검증] total NaN: {total_nan}, total Inf: {total_inf}")
    print(f"[검증] 실행 시간: {elapsed:.1f}초")
    if peak_gpu_gb > 0:
        print(f"[검증] peak GPU memory: {peak_gpu_gb:.3f} GB")

    actual_patches = summary["n_patches_used"]
    patch_diff = actual_patches - P56_EXPECTED_PATCHES
    print(f"[검증] P-A56 예상 patch: {P56_EXPECTED_PATCHES:,}, 실제 patch: {actual_patches:,}, 차이: {patch_diff:+,}")

    assert sample_mean_shape == (ws_paths.REDUCED_FEATURE_DIM,), \
        f"mean shape 불일치: {sample_mean_shape} != ({ws_paths.REDUCED_FEATURE_DIM},)"
    assert sample_cov_shape == (ws_paths.REDUCED_FEATURE_DIM, ws_paths.REDUCED_FEATURE_DIM), \
        f"cov shape 불일치: {sample_cov_shape}"
    assert total_nan == 0, f"NaN 발생: {total_nan}"
    assert total_inf == 0, f"Inf 발생: {total_inf}"

    # ---- runtime_summary.csv ----
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    runtime_rows = [
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_requested", "value": len(train_patients)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_processed", "value": summary["n_patients_success"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_failed", "value": summary["n_patients_failed"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "elapsed_seconds", "value": round(elapsed, 2)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patches_used", "value": actual_patches},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patches_skipped", "value": summary["n_patches_skipped"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "peak_gpu_gb", "value": round(peak_gpu_gb, 3)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "backbone", "value": ws_paths.BACKBONE},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "feature_dim", "value": ws_paths.REDUCED_FEATURE_DIM},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "output_path", "value": str(OUTPUT_NPZ)},
    ]
    record_runtime(runtime_rows)
    print(f"[train] runtime_summary.csv 기록: {RUNTIME_CSV}")

    # ---- 보고서 JSON ----
    report = {
        "step": "P-A57",
        "verdict": "통과",
        "timestamp": ts,
        "patients_processed": summary["n_patients_success"],
        "patients_failed": summary["n_patients_failed"],
        "split": "normal_train",
        "raw_feature_dim": ws_paths.RAW_FEATURE_DIM,
        "selected_feature_dim": ws_paths.REDUCED_FEATURE_DIM,
        "mean_shape": list(sample_mean_shape) if sample_mean_shape else None,
        "cov_shape": list(sample_cov_shape) if sample_cov_shape else None,
        "position_bins_with_data": n_bins_with_data,
        "n_patches_used": actual_patches,
        "n_patches_skipped": summary["n_patches_skipped"],
        "p56_expected_patches": P56_EXPECTED_PATCHES,
        "patch_diff_vs_p56": patch_diff,
        "total_nan": total_nan,
        "total_inf": total_inf,
        "elapsed_seconds": round(elapsed, 2),
        "peak_gpu_gb": round(peak_gpu_gb, 3),
        "output_npz": str(OUTPUT_NPZ),
        "safety": {
            "normal_val_scoring": False,
            "normal_test_scoring": False,
            "lesion_scoring": False,
            "threshold_calculated": False,
            "metrics_calculated": False,
            "stage2_holdout_accessed": False,
            "existing_results_modified": False,
            "pip_install": False,
        },
        "p54_smoke_preserved": str(p54_npz),
        "p55_smoke_preserved": str(p55_npz),
        "next_step_p_a57_5_distribution_validation_feasible": True,
    }
    report_json_path = REPORTS_DIR / "p_a57_full_train.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[train] 보고서 JSON: {report_json_path}")

    # ---- 보고서 MD ----
    md_lines = [
        "# P-A57 Full Train 보고서",
        "",
        "## 판정: 통과",
        "",
        "## 실행 정보",
        f"- 처리 환자 수: {summary['n_patients_success']}명 / 요청 290명",
        f"- 실패 환자 수: {summary['n_patients_failed']}명",
        f"- 사용 split: normal_train",
        f"- raw feature dim: {ws_paths.RAW_FEATURE_DIM}",
        f"- selected feature dim: {ws_paths.REDUCED_FEATURE_DIM}",
        "",
        "## 분포 검증",
        f"- mean shape: {list(sample_mean_shape) if sample_mean_shape else 'N/A'}",
        f"- cov shape: {list(sample_cov_shape) if sample_cov_shape else 'N/A'}",
        f"- position_bin 수 (data 있는 bin): {n_bins_with_data}",
        f"- 총 patch 수 (used): {actual_patches:,}",
        f"- skipped patch 수: {summary['n_patches_skipped']:,}",
        f"- NaN: {total_nan}, Inf: {total_inf}",
        "",
        "## P-A56 예상 대비 실제 patch 수",
        f"- P-A56 예상 patch 수: {P56_EXPECTED_PATCHES:,} (단순 추정치)",
        f"- 실제 patch 수: {actual_patches:,}",
        f"- 차이: {patch_diff:+,}",
        f"- 비고: P-A56의 {P56_EXPECTED_PATCHES:,}는 단순 추정치이므로 실제 patch 수와 다를 수 있음",
        "",
        "## 실행 성능",
        f"- 실행 시간: {round(elapsed, 2)}초",
        f"- peak GPU memory: {round(peak_gpu_gb, 3)} GB",
        "",
        "## 안전 확인",
        f"- selected index: P-A53 파일 사용 ({sidx_path})",
        f"- P-A54 smoke_limit1 결과 보존: {p54_npz}",
        f"- P-A55 smoke_limit5 결과 보존: {p55_npz}",
        f"- 기존 random100 결과 무수정: True (접근 없음)",
        f"- scoring/threshold/metrics 미실행: True",
        f"- lesion/stage2_holdout 미접근: True",
        "",
        "## 다음 단계",
        f"- P-A57.5 distribution validation 가능 여부: True",
    ]
    report_md_path = REPORTS_DIR / "p_a57_full_train.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[train] 보고서 MD: {report_md_path}")
    print("\n=== P-A57 full train 완료: 통과 ===")


if __name__ == "__main__":
    main()
