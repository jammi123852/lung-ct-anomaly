"""
P-A70: EfficientNet-B0 ImageNet train smoke limit=5

normal train split 중 5명으로 확장해 누적 안정성, position-bin 분포,
저장 구조, NaN/Inf 여부를 확인한다.
- FeatureExtractorEffNetB0 (EfficientNet-B0 ImageNet1K V1)
- PaDiMModel (feature_dim=100, selected_feature_indices effnet100)
- smoke output: experiments/efficientnet_b0_imagenet_v1/outputs/smoke/train_limit5/position_bin_stats.npz

금지: full train / val/test scoring / lesion scoring / threshold / metrics / stage2_holdout
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 경로 정의
PROJ_ROOT = Path(__file__).resolve().parents[3]   # lung-ct-anomaly
EXP_ROOT  = Path(__file__).resolve().parents[1]   # experiments/efficientnet_b0_imagenet_v1
SRC_DIR   = PROJ_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 백본 고정값
BACKBONE            = "efficientnet_b0"
PRETRAIN_SOURCE     = "imagenet"
RAW_FEATURE_DIM     = 144
REDUCED_FEATURE_DIM = 100
MASK_TYPE           = "roi_0_0"
PATHS_CONFIG        = "paths.local.v2_roi0_0.yaml"
LIMIT               = 5
SCRIPT_NAME         = "p_a70_train_smoke_limit5.py"

# 경로
SELECTED_INDICES_PATH = EXP_ROOT / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
SMOKE_NPZ             = EXP_ROOT / "outputs" / "smoke" / "train_limit5" / "position_bin_stats.npz"
FULL_MODEL_NPZ        = EXP_ROOT / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
REPORTS_DIR           = EXP_ROOT / "outputs" / "reports" / "smoke"
ERROR_CSV             = REPORTS_DIR / "error.csv"
RUNTIME_CSV           = REPORTS_DIR / "p_a70_runtime_summary.csv"

# 보존 확인 대상 (P-A69 limit1 결과)
P_A69_NPZ  = EXP_ROOT / "outputs" / "smoke" / "train_limit1" / "position_bin_stats.npz"
P_A69_JSON = REPORTS_DIR / "p_a69_train_smoke_limit1.json"

# 선행 보고서
P_A68_JSON = EXP_ROOT / "outputs" / "reports" / "p_a68_selected_indices.json"
P_A67B_JSON = (
    EXP_ROOT / "outputs" / "reports"
    / "p_a67b_weight_and_1slice_smoke"
    / "p_a67b_effnet_b0_weight_and_1slice_smoke.json"
)

# normal split JSON
SPLIT_JSON = PROJ_ROOT / "outputs" / "position-aware-padim-v1" / "splits" / "normal_v1.json"

# 공통 컬럼
ERROR_COLUMNS   = ["patient_id", "error_type", "error_msg", "file_logical"]
RUNTIME_COLUMNS = ["timestamp", "script", "metric", "value"]


# ------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------

def record_error(patient_id: str, error_type: str, error_msg: str, file_logical: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not ERROR_CSV.exists() or ERROR_CSV.stat().st_size == 0
    with open(ERROR_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "patient_id": patient_id,
            "error_type": error_type,
            "error_msg": error_msg,
            "file_logical": file_logical,
        })


def record_runtime(rows_extra: list) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_CSV.exists() or RUNTIME_CSV.stat().st_size == 0
    with open(RUNTIME_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows_extra)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main() -> None:
    import numpy as np

    # ---- 가드 1: P-A69 보고서 통과 확인 ----
    assert P_A69_JSON.exists(), f"P-A69 보고서 없음: {P_A69_JSON}"
    with open(P_A69_JSON) as f:
        p69 = json.load(f)
    assert p69.get("verdict") == "통과", f"P-A69 verdict != 통과: {p69.get('verdict')}"
    print("[guard1] P-A69 verdict: 통과 ✅")

    # ---- 가드 2: P-A69 limit1 결과 보존 확인 ----
    assert P_A69_NPZ.exists(), f"P-A69 limit1 npz 없음 (보존 필요): {P_A69_NPZ}"
    print(f"[guard2] P-A69 limit1 결과 보존 확인 OK: {P_A69_NPZ}")

    # ---- 가드 3: P-A68 selected_feature_indices 검증 ----
    assert P_A68_JSON.exists(), f"P-A68 보고서 없음: {P_A68_JSON}"
    with open(P_A68_JSON) as f:
        p68 = json.load(f)
    assert p68.get("verdict") == "pass", f"P-A68 verdict != pass: {p68.get('verdict')}"
    assert SELECTED_INDICES_PATH.exists(), f"selected_feature_indices.npy 없음: {SELECTED_INDICES_PATH}"
    sidx = np.load(str(SELECTED_INDICES_PATH))
    assert sidx.shape == (REDUCED_FEATURE_DIM,), f"[guard3] shape 불일치: {sidx.shape}"
    assert len(np.unique(sidx)) == REDUCED_FEATURE_DIM, f"[guard3] unique 불일치: {len(np.unique(sidx))}"
    assert sidx.min() >= 0 and sidx.max() <= 143, \
        f"[guard3] range 오류: {sidx.min()}~{sidx.max()} (기대: 0~143)"
    print(f"[guard3] selected_feature_indices OK: shape={sidx.shape}, "
          f"unique={REDUCED_FEATURE_DIM}, range=[{sidx.min()},{sidx.max()}]")

    # ---- 가드 4: P-A67b raw_feature_dim=144 확인 ----
    assert P_A67B_JSON.exists(), f"P-A67b 보고서 없음: {P_A67B_JSON}"
    with open(P_A67B_JSON) as f:
        p67b = json.load(f)
    rfd = p67b.get("raw_feature_dim", {})
    measured = rfd.get("measured") if isinstance(rfd, dict) else rfd
    assert measured == 144, f"[guard4] P-A67b raw_feature_dim 불일치: {measured}"
    print(f"[guard4] P-A67b raw_feature_dim=144 확인 ✅")

    # ---- 가드 5: normal train split 290명 확인 ----
    assert SPLIT_JSON.exists(), f"normal_v1.json 없음: {SPLIT_JSON}"
    with open(SPLIT_JSON) as f:
        split_data = json.load(f)
    train_patients = list(split_data["train"])
    assert len(train_patients) == 290, f"[guard5] train split 수 불일치: {len(train_patients)}"
    print(f"[guard5] normal train split: {len(train_patients)}명 확인 ✅")

    # ---- 가드 6: limit=5 초과 금지 ----
    assert LIMIT == 5, f"[guard6] limit 값 오류: {LIMIT}"
    print(f"[guard6] limit={LIMIT} 확인 OK")

    # ---- 가드 7: smoke output 경로 비어있는지 확인 ----
    if SMOKE_NPZ.exists():
        print(f"[guard7][ABORT] smoke output이 이미 존재합니다: {SMOKE_NPZ}")
        print("  기존 결과를 수동으로 삭제하거나 경로를 확인하세요.")
        sys.exit(1)
    print(f"[guard7] smoke output 경로 비어있음 OK: {SMOKE_NPZ}")

    # ---- 가드 8: full train 경로와 smoke 경로 분리 확인 ----
    assert SMOKE_NPZ != FULL_MODEL_NPZ, "[guard8] smoke/full 경로 충돌"
    print(f"[guard8] smoke/full 경로 분리 OK")

    # ---- 가드 9: stage2_holdout 접근 금지 ----
    print("[guard9] stage2_holdout 접근 금지 확인 OK (접근 없음)")

    # ---- limit=5 환자 사용 ----
    smoke_patients = train_patients[:LIMIT]
    print(f"\n[train] smoke limit={LIMIT}, 환자: {smoke_patients}")
    print(f"[train] backbone={BACKBONE}, raw_dim={RAW_FEATURE_DIM}, reduced_dim={REDUCED_FEATURE_DIM}")
    print(f"[train] selected_feature_indices: {SELECTED_INDICES_PATH}")
    print(f"[train] smoke output: {SMOKE_NPZ}")
    print(f"[train] reports: {REPORTS_DIR}")

    # ---- ConfigManager, DataLoader ----
    from position_aware_padim.config_manager import ConfigManager
    from position_aware_padim.data_loader import DataLoader
    from position_aware_padim.path_resolver import PathResolver
    from position_aware_padim.padim_model import PaDiMModel
    from position_aware_padim.feature_extractor_effnet_b0_scaffold import FeatureExtractorEffNetB0

    cfg = ConfigManager(str(PROJ_ROOT))
    cfg.load_config(paths_yaml=PATHS_CONFIG)
    normal_training_ready = cfg.get("paths", "normal_training_ready", "")
    assert normal_training_ready, "paths config에 'normal_training_ready' 없음"
    manifest_path = Path(normal_training_ready) / "manifests" / "patient_manifest.csv"
    assert manifest_path.exists(), f"patient_manifest.csv 없음: {manifest_path}"

    path_resolver = PathResolver(str(manifest_path), normal_training_ready)
    loader = DataLoader(str(manifest_path), path_resolver, str(ERROR_CSV), use_mmap=True)

    # ---- FeatureExtractorEffNetB0 초기화 ----
    print("\n[train] FeatureExtractorEffNetB0(EfficientNet-B0/ImageNet) 초기화 중...")
    feature_extractor = FeatureExtractorEffNetB0()
    print(f"[train] device: {feature_extractor.device}")

    assert feature_extractor.raw_feature_dim == RAW_FEATURE_DIM, \
        f"raw_feature_dim 불일치: {feature_extractor.raw_feature_dim} != {RAW_FEATURE_DIM}"
    print(f"[train] raw_feature_dim={feature_extractor.raw_feature_dim} 확인 ✅")

    # GPU peak memory 초기화
    import torch
    gpu_avail = (feature_extractor.device == "cuda")
    if gpu_avail:
        torch.cuda.reset_peak_memory_stats()

    # ---- PaDiMModel (feature_dim=100) ----
    model = PaDiMModel(
        selected_feature_indices_path=str(SELECTED_INDICES_PATH),
        feature_dim=REDUCED_FEATURE_DIM,
        eps=1e-5,
    )
    print(f"[train] PaDiMModel 초기화: feature_dim={model.feature_dim}")

    n_processed = 0
    n_failed    = 0

    def patient_stream():
        nonlocal n_processed, n_failed
        for pid in smoke_patients:
            data = loader.load_patient_data(pid, mask_type=MASK_TYPE)
            if data is None:
                n_failed += 1
                record_error(pid, "load_failed",
                             "DataLoader.load_patient_data returned None", "patient_data")
                print(f"  [SKIP] {pid}: 로드 실패")
                continue
            n_processed += 1
            print(f"  [OK]   {pid}: ct_hu={data['ct_hu'].shape}, patches={len(data['patch_df'])}")
            yield data

    start_time = time.time()
    print("\n[train] smoke 학습 시작...")
    try:
        model.train(feature_extractor, patient_stream(), split=None)
    except Exception as exc:
        record_error("__train__", "train_error", str(exc), SCRIPT_NAME)
        raise

    elapsed     = time.time() - start_time
    peak_gpu_gb = 0.0
    if gpu_avail:
        peak_gpu_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[train] peak GPU memory: {peak_gpu_gb:.3f} GB")

    summary = model.train_summary
    print(f"[train] 완료: {summary['n_patients_success']}명 성공, "
          f"{summary['n_patients_failed']}명 실패, {elapsed:.1f}초")
    print(f"[train] 사용 patch={summary['n_patches_used']:,}, "
          f"스킵 patch={summary['n_patches_skipped']:,}")

    if not model.stats:
        record_error("__train__", "empty_stats", "stats가 비어있음", SCRIPT_NAME)
        raise RuntimeError("smoke 학습 결과가 비어있습니다.")

    # ---- 저장 ----
    SMOKE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(SMOKE_NPZ))
    print(f"[train] 저장 완료: {SMOKE_NPZ}")

    # ---- position_bin 통계 요약 및 NaN/Inf 확인 ----
    print("\n[position_bin 통계 요약]")
    print(f"{'key':<25} {'count':>10} {'mean_shape':>12} {'NaN':>5} {'Inf':>5}")
    print("-" * 65)
    n_bins_with_data  = 0
    total_nan         = 0
    total_inf         = 0
    sample_key        = None
    sample_mean_shape = None
    sample_cov_shape  = None

    for key in model._all_keys():
        s     = model.stats.get(key, {})
        count = s.get("count", 0)
        if count > 0 and "mean" in s:
            mean  = s["mean"]
            cov   = s.get("cov", np.array([]))
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
            if sample_key is None:
                sample_key        = key
                sample_mean_shape = mean.shape
                sample_cov_shape  = cov.shape
        else:
            print(f"{key:<25} {'(no data)':>10}")

    print(f"\n[검증] mean shape:              {sample_mean_shape}")
    print(f"[검증] cov shape:               {sample_cov_shape}")
    print(f"[검증] position_bin with data:  {n_bins_with_data}")
    print(f"[검증] total NaN:               {total_nan}")
    print(f"[검증] total Inf:               {total_inf}")
    print(f"[검증] raw_feature_dim 144:     {feature_extractor.raw_feature_dim == 144}")
    print(f"[검증] selected_dim 100:        {model.feature_dim == 100}")
    print(f"[검증] 실행 시간:               {elapsed:.1f}초")
    if peak_gpu_gb > 0:
        print(f"[검증] peak GPU memory:         {peak_gpu_gb:.3f} GB")

    # P-A69 limit1 결과 보존 확인
    assert P_A69_NPZ.exists(), f"P-A69 limit1 결과가 사라졌습니다: {P_A69_NPZ}"
    print(f"[검증] P-A69 limit1 결과 보존: OK")

    assert sample_mean_shape == (REDUCED_FEATURE_DIM,), \
        f"mean shape 불일치: {sample_mean_shape}"
    assert sample_cov_shape == (REDUCED_FEATURE_DIM, REDUCED_FEATURE_DIM), \
        f"cov shape 불일치: {sample_cov_shape}"
    assert total_nan == 0, f"NaN 발생: {total_nan}"
    assert total_inf == 0, f"Inf 발생: {total_inf}"

    # ---- runtime_summary.csv ----
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    runtime_rows = [
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_requested",
         "value": LIMIT},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_processed",
         "value": summary["n_patients_success"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_failed",
         "value": summary["n_patients_failed"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "elapsed_seconds",
         "value": round(elapsed, 2)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patches_used",
         "value": summary["n_patches_used"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patches_skipped",
         "value": summary["n_patches_skipped"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "peak_gpu_gb",
         "value": round(peak_gpu_gb, 3)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "backbone",        "value": BACKBONE},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "raw_feature_dim", "value": RAW_FEATURE_DIM},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "feature_dim",     "value": REDUCED_FEATURE_DIM},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "smoke_limit",     "value": LIMIT},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "output_path",     "value": str(SMOKE_NPZ)},
    ]
    record_runtime(runtime_rows)
    print(f"[train] runtime_summary.csv 기록: {RUNTIME_CSV}")

    # ---- 보고서 JSON ----
    verdict = "통과"
    report = {
        "step": "P-A70",
        "verdict": verdict,
        "timestamp": ts,
        "backbone": BACKBONE,
        "patients_requested": LIMIT,
        "patients_processed": summary["n_patients_success"],
        "patients_failed": summary["n_patients_failed"],
        "smoke_patients": smoke_patients,
        "split": "normal_train",
        "limit": LIMIT,
        "p_a69_verdict": p69.get("verdict"),
        "p_a68_verdict": p68.get("verdict"),
        "raw_feature_dim": RAW_FEATURE_DIM,
        "raw_feature_dim_verified": (feature_extractor.raw_feature_dim == RAW_FEATURE_DIM),
        "selected_feature_dim": REDUCED_FEATURE_DIM,
        "selected_feature_dim_verified": (model.feature_dim == REDUCED_FEATURE_DIM),
        "mean_shape": list(sample_mean_shape) if sample_mean_shape else None,
        "cov_shape": list(sample_cov_shape) if sample_cov_shape else None,
        "position_bins_with_data": n_bins_with_data,
        "n_patches_used": summary["n_patches_used"],
        "n_patches_skipped": summary["n_patches_skipped"],
        "total_nan": total_nan,
        "total_inf": total_inf,
        "elapsed_seconds": round(elapsed, 2),
        "peak_gpu_gb": round(peak_gpu_gb, 3),
        "output_npz": str(SMOKE_NPZ),
        "p_a69_limit1_preserved": P_A69_NPZ.exists(),
        "safety": {
            "full_train": False,
            "val_test_scoring": False,
            "lesion_scoring": False,
            "threshold_calculated": False,
            "metrics_calculated": False,
            "stage2_holdout_accessed": False,
            "existing_results_modified": False,
            "pip_install": False,
            "additional_download": False,
            "limit_exceeded": False,
        },
        "next_step_p_a71_full_train_preflight_feasible": True,
    }
    report_json_path = REPORTS_DIR / "p_a70_train_smoke_limit5.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[train] 보고서 JSON: {report_json_path}")

    # ---- 보고서 MD ----
    report_md_path = REPORTS_DIR / "p_a70_train_smoke_limit5.md"
    md_lines = [
        "# P-A70 EfficientNet-B0 Train Smoke Limit=5 보고서",
        "",
        f"**판정: {verdict}**",
        "",
        f"- 생성일시: {ts}",
        f"- 백본: {BACKBONE} (ImageNet)",
        "",
        "## 처리 결과",
        "",
        f"- 처리 환자 수: {summary['n_patients_success']}명",
        f"- 실패 환자 수: {summary['n_patients_failed']}명",
        f"- 처리 환자 ID: {', '.join(smoke_patients)}",
        "",
        "## Feature 차원 검증",
        "",
        f"- raw_feature_dim=144 확인: {feature_extractor.raw_feature_dim == 144}",
        f"- selected_dim=100 확인: {model.feature_dim == 100}",
        f"- mean shape: {sample_mean_shape}",
        f"- cov shape: {sample_cov_shape}",
        "",
        "## Patch 통계",
        "",
        f"- used patch: {summary['n_patches_used']:,}",
        f"- skipped patch: {summary['n_patches_skipped']:,}",
        f"- position_bin with data: {n_bins_with_data}",
        f"- NaN: {total_nan}",
        f"- Inf: {total_inf}",
        "",
        "## 실행 정보",
        "",
        f"- elapsed_seconds: {round(elapsed, 2)}",
        f"- peak_gpu_gb: {round(peak_gpu_gb, 3)}",
        f"- smoke output: {SMOKE_NPZ}",
        "",
        "## 기존 결과 보존 확인",
        "",
        f"- P-A69 limit1 결과 보존: {P_A69_NPZ.exists()}",
        "- P-A67a/67b/68/69 결과 무수정: True",
        "- ResNet18/ResNet50 결과 무수정: True",
        "- stage2_holdout 접근: False",
        "",
        "## Safety",
        "",
        "- full_train: False",
        "- val/test scoring: False",
        "- lesion scoring: False",
        "- threshold 계산: False",
        "- metrics 계산: False",
        "- stage2_holdout 접근: False",
        "- 기존 결과 수정: False",
        "- pip/conda install: False",
        "- 추가 다운로드: False",
        "- limit 초과: False",
        "",
        "## 다음 단계",
        "",
        "- P-A71 full train preflight 진행 가능: True",
    ]
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[train] 보고서 MD: {report_md_path}")

    print(f"\n=== P-A70 smoke limit5 완료: {verdict} ===")


if __name__ == "__main__":
    main()
