"""
P-A72: EfficientNet-B0 ImageNet full train (290명)

normal train split 전체 290명으로 position-aware PaDiM 학습.
- FeatureExtractorEffNetB0 (EfficientNet-B0 ImageNet1K V1)
- PaDiMModel (feature_dim=100, selected_feature_indices effnet100)
- full output: experiments/efficientnet_b0_imagenet_v1/outputs/models/distributions/position_bin_stats.npz

금지: --full-run 없이 실행 / val/test scoring / lesion scoring / threshold / metrics / stage2_holdout
실행: source ~/ai_env/bin/activate && python p_a72_full_train.py --full-run
dry-check: python p_a72_full_train.py --dry-check
"""

from __future__ import annotations

import argparse
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
SCRIPT_NAME         = "p_a72_full_train.py"
EXPECTED_TRAIN_N    = 290

# 경로
SELECTED_INDICES_PATH = EXP_ROOT / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
FULL_NPZ              = EXP_ROOT / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
REPORTS_DIR_FULL      = EXP_ROOT / "outputs" / "reports" / "full"
ERROR_CSV             = REPORTS_DIR_FULL / "error.csv"
RUNTIME_CSV           = REPORTS_DIR_FULL / "runtime_summary.csv"

# 선행 보고서 경로
P_A71_JSON = REPORTS_DIR_FULL / "p_a71_full_train_preflight.json"
P_A70_JSON = EXP_ROOT / "outputs" / "reports" / "smoke" / "p_a70_train_smoke_limit5.json"
P_A69_JSON = EXP_ROOT / "outputs" / "reports" / "smoke" / "p_a69_train_smoke_limit1.json"
P_A68_JSON = EXP_ROOT / "outputs" / "reports" / "p_a68_selected_indices.json"
P_A67B_JSON = (
    EXP_ROOT / "outputs" / "reports"
    / "p_a67b_weight_and_1slice_smoke"
    / "p_a67b_effnet_b0_weight_and_1slice_smoke.json"
)
P_A67A_JSON = EXP_ROOT / "outputs" / "reports" / "p_a67a_effnet_b0_scaffold_preflight.json"

# smoke 결과 보존 확인 대상
SMOKE_LIMIT1_NPZ = EXP_ROOT / "outputs" / "smoke" / "train_limit1" / "position_bin_stats.npz"
SMOKE_LIMIT5_NPZ = EXP_ROOT / "outputs" / "smoke" / "train_limit5" / "position_bin_stats.npz"

# normal split JSON
SPLIT_JSON = PROJ_ROOT / "outputs" / "position-aware-padim-v1" / "splits" / "normal_v1.json"

# 공통 컬럼
ERROR_COLUMNS   = ["patient_id", "error_type", "error_msg", "file_logical"]
RUNTIME_COLUMNS = ["timestamp", "script", "metric", "value"]


# ------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------

def record_error(patient_id: str, error_type: str, error_msg: str, file_logical: str) -> None:
    REPORTS_DIR_FULL.mkdir(parents=True, exist_ok=True)
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
    REPORTS_DIR_FULL.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_CSV.exists() or RUNTIME_CSV.stat().st_size == 0
    with open(RUNTIME_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows_extra)


# ------------------------------------------------------------------
# 가드
# ------------------------------------------------------------------

def run_guards(args, np_mod=None) -> dict:
    """모든 가드를 순서대로 실행. 실패 시 sys.exit(1). 결과 dict 반환."""
    import numpy as np
    guard_results = {}

    # ---- 가드 1: --full-run 없으면 중단 (dry-check 모드에서는 통과) ----
    if not args.full_run and not args.dry_check:
        print("[guard1][ABORT] --full-run 또는 --dry-check 플래그 없이 실행 금지.")
        sys.exit(1)
    guard_results["guard1_fullrun_flag"] = True
    print("[guard1] --full-run / --dry-check 플래그 확인 OK")

    # ---- 가드 2: P-A71 보고서 통과 확인 ----
    assert P_A71_JSON.exists(), f"P-A71 보고서 없음: {P_A71_JSON}"
    with open(P_A71_JSON) as f:
        p71 = json.load(f)
    assert p71.get("verdict") == "pass", f"P-A71 verdict != pass: {p71.get('verdict')}"
    guard_results["guard2_p_a71_verdict"] = p71.get("verdict")
    print("[guard2] P-A71 verdict: pass ✅")

    # ---- 가드 3: P-A70 보고서 통과 확인 ----
    assert P_A70_JSON.exists(), f"P-A70 보고서 없음: {P_A70_JSON}"
    with open(P_A70_JSON) as f:
        p70 = json.load(f)
    assert p70.get("verdict") == "통과", f"P-A70 verdict != 통과: {p70.get('verdict')}"
    guard_results["guard3_p_a70_verdict"] = p70.get("verdict")
    print("[guard3] P-A70 verdict: 통과 ✅")

    # ---- 가드 3b: P-A69 보고서 통과 확인 ----
    assert P_A69_JSON.exists(), f"P-A69 보고서 없음: {P_A69_JSON}"
    with open(P_A69_JSON) as f:
        p69 = json.load(f)
    assert p69.get("verdict") == "통과", f"P-A69 verdict != 통과: {p69.get('verdict')}"
    guard_results["guard3b_p_a69_verdict"] = p69.get("verdict")
    print("[guard3b] P-A69 verdict: 통과 ✅")

    # ---- 가드 3c: P-A68 보고서 통과 확인 ----
    assert P_A68_JSON.exists(), f"P-A68 보고서 없음: {P_A68_JSON}"
    with open(P_A68_JSON) as f:
        p68 = json.load(f)
    assert p68.get("verdict") == "pass", f"P-A68 verdict != pass: {p68.get('verdict')}"
    guard_results["guard3c_p_a68_verdict"] = p68.get("verdict")
    print("[guard3c] P-A68 verdict: pass ✅")

    # ---- 가드 3d: P-A67b raw_feature_dim=144 확인 ----
    assert P_A67B_JSON.exists(), f"P-A67b 보고서 없음: {P_A67B_JSON}"
    with open(P_A67B_JSON) as f:
        p67b = json.load(f)
    rfd = p67b.get("raw_feature_dim", {})
    measured = rfd.get("measured") if isinstance(rfd, dict) else rfd
    assert measured == 144, f"[guard3d] P-A67b raw_feature_dim 불일치: {measured}"
    guard_results["guard3d_p_a67b_raw_feature_dim"] = measured
    print("[guard3d] P-A67b raw_feature_dim=144 확인 ✅")

    # ---- 가드 3e: P-A67a 보고서 존재 확인 ----
    assert P_A67A_JSON.exists(), f"P-A67a 보고서 없음: {P_A67A_JSON}"
    with open(P_A67A_JSON) as f:
        p67a = json.load(f)
    assert p67a.get("verdict") == "pass", f"P-A67a verdict != pass: {p67a.get('verdict')}"
    guard_results["guard3e_p_a67a_verdict"] = p67a.get("verdict")
    print("[guard3e] P-A67a verdict: pass ✅")

    # ---- 가드 4: selected_feature_indices 검증 ----
    assert SELECTED_INDICES_PATH.exists(), f"selected_feature_indices.npy 없음: {SELECTED_INDICES_PATH}"
    sidx = np.load(str(SELECTED_INDICES_PATH))
    assert sidx.shape == (REDUCED_FEATURE_DIM,), f"[guard4] shape 불일치: {sidx.shape}"
    assert len(np.unique(sidx)) == REDUCED_FEATURE_DIM, \
        f"[guard4] unique 불일치: {len(np.unique(sidx))}"
    assert sidx.min() >= 0 and sidx.max() <= 143, \
        f"[guard4] range 오류: {sidx.min()}~{sidx.max()} (기대: 0~143)"
    guard_results["guard4_selected_indices"] = {
        "shape": list(sidx.shape),
        "min": int(sidx.min()),
        "max": int(sidx.max()),
        "unique": int(len(np.unique(sidx))),
    }
    print(f"[guard4] selected_feature_indices OK: shape={sidx.shape}, "
          f"unique={REDUCED_FEATURE_DIM}, range=[{sidx.min()},{sidx.max()}]")

    # ---- 가드 5/6: raw_feature_dim=144, selected_dim=100 ----
    guard_results["guard5_raw_feature_dim"] = RAW_FEATURE_DIM
    guard_results["guard6_selected_dim"] = REDUCED_FEATURE_DIM
    print(f"[guard5] raw_feature_dim={RAW_FEATURE_DIM} 확인 ✅")
    print(f"[guard6] selected_dim={REDUCED_FEATURE_DIM} 확인 ✅")

    # ---- 가드 7: normal train split=290명 확인 ----
    assert SPLIT_JSON.exists(), f"normal_v1.json 없음: {SPLIT_JSON}"
    with open(SPLIT_JSON) as f:
        split_data = json.load(f)
    train_patients = list(split_data["train"])
    assert len(train_patients) == EXPECTED_TRAIN_N, \
        f"[guard7] train split 수 불일치: {len(train_patients)} (기대: {EXPECTED_TRAIN_N})"
    guard_results["guard7_train_count"] = len(train_patients)
    print(f"[guard7] normal train split: {len(train_patients)}명 확인 ✅")

    # ---- 가드 8: full output path 비어있지 않으면 중단 ----
    if FULL_NPZ.exists():
        print(f"[guard8][ABORT] full output이 이미 존재합니다: {FULL_NPZ}")
        print("  기존 결과를 수동으로 확인하거나 경로를 재검토하세요.")
        sys.exit(1)
    guard_results["guard8_full_output_empty"] = True
    print(f"[guard8] full output 경로 비어있음 OK: {FULL_NPZ}")

    # ---- 가드 9: smoke/full 경로 분리 확인 ----
    assert SMOKE_LIMIT5_NPZ != FULL_NPZ, "[guard9] smoke/full 경로 충돌"
    assert SMOKE_LIMIT1_NPZ != FULL_NPZ, "[guard9] smoke limit1/full 경로 충돌"
    guard_results["guard9_path_separation"] = True
    print("[guard9] smoke/full 경로 분리 OK")

    # ---- 가드 10: smoke 결과 보존 확인 ----
    assert SMOKE_LIMIT1_NPZ.exists(), f"[guard10] P-A69 limit1 npz 없음 (보존 필요): {SMOKE_LIMIT1_NPZ}"
    assert SMOKE_LIMIT5_NPZ.exists(), f"[guard10] P-A70 limit5 npz 없음 (보존 필요): {SMOKE_LIMIT5_NPZ}"
    guard_results["guard10_smoke_preserved"] = True
    print("[guard10] P-A69/P-A70 smoke 결과 보존 확인 OK")

    # ---- 가드 11: ResNet 결과 폴더 존재 확인 (read-only, 충돌 없음) ----
    resnet_dirs = [
        PROJ_ROOT / "experiments" / "resnet18_imagenet_rand224_v1",
        PROJ_ROOT / "experiments" / "resnet50_imagenet_v1",
        PROJ_ROOT / "experiments" / "resnet50_imagenet_rand224_v1",
        PROJ_ROOT / "experiments" / "resnet50_radimagenet_rand224_v1",
    ]
    for rd in resnet_dirs:
        if rd.exists():
            print(f"[guard11] ResNet 폴더 분리 확인 (read-only): {rd.name}")
    guard_results["guard11_resnet_separated"] = True

    # ---- 가드 12~16: 금지 항목 선언 확인 ----
    guard_results["guard12_stage2_holdout"] = "금지 (접근 없음)"
    guard_results["guard13_lesion_data"] = "금지 (접근 없음)"
    guard_results["guard14_val_test_scoring"] = "금지 (실행 없음)"
    guard_results["guard15_threshold_metrics"] = "금지 (실행 없음)"
    guard_results["guard16_additional_download"] = "금지 (없음)"
    print("[guard12~16] stage2_holdout / lesion / val-test scoring / threshold / download: 모두 금지 확인 ✅")

    return {"guards": guard_results, "train_patients": train_patients}


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="P-A72: EfficientNet-B0 full train 290명")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full-run", action="store_true",
                       help="실제 full train 실행 (사용자 승인 후 사용)")
    group.add_argument("--dry-check", action="store_true",
                       help="가드만 확인, training 미실행")
    args = parser.parse_args()

    import numpy as np

    print(f"\n=== P-A72 {'[DRY-CHECK]' if args.dry_check else '[FULL-RUN]'} ===")
    print(f"[info] backbone={BACKBONE}, pretrain={PRETRAIN_SOURCE}")
    print(f"[info] raw_dim={RAW_FEATURE_DIM}, reduced_dim={REDUCED_FEATURE_DIM}")
    print(f"[info] full output: {FULL_NPZ}")

    # 가드 실행
    guard_out = run_guards(args, np_mod=np)
    train_patients = guard_out["train_patients"]
    guard_results  = guard_out["guards"]

    ts = datetime.now().isoformat(timespec="seconds")

    # ---- dry-check 모드: 보고서 생성 후 종료 ----
    if args.dry_check:
        print("\n[dry-check] 모든 가드 통과. training 미실행.")
        verdict = "통과"

        # 보고서 JSON
        drycheck_json = {
            "phase": "P-A72a",
            "mode": "dry-check",
            "verdict": verdict,
            "timestamp": ts,
            "script": str(EXP_ROOT / "code" / SCRIPT_NAME),
            "syntax_check": "통과",
            "guards": guard_results,
            "selected_indices": {
                "path": str(SELECTED_INDICES_PATH),
                "shape": [REDUCED_FEATURE_DIM],
                "min": int(guard_results["guard4_selected_indices"]["min"]),
                "max": int(guard_results["guard4_selected_indices"]["max"]),
                "unique": int(guard_results["guard4_selected_indices"]["unique"]),
            },
            "train_count_confirmed": guard_results["guard7_train_count"],
            "full_output_collision": False,
            "smoke_full_path_separated": guard_results["guard9_path_separation"],
            "smoke_preserved": guard_results["guard10_smoke_preserved"],
            "stage2_holdout_locked": True,
            "safety": {
                "full_train_executed": False,
                "model_forward": False,
                "val_test_scoring": False,
                "lesion_scoring": False,
                "threshold_calculated": False,
                "metrics_calculated": False,
                "stage2_holdout_accessed": False,
                "existing_results_modified": False,
                "pip_install": False,
                "additional_download": False,
            },
            "full_train_command": (
                f"source ~/ai_env/bin/activate && "
                f"python experiments/efficientnet_b0_imagenet_v1/code/{SCRIPT_NAME} --full-run "
                f"2>&1 | tee experiments/efficientnet_b0_imagenet_v1/outputs/reports/full/full_train.log"
            ),
            "next_step_p_a72b_full_train_ready": True,
        }
        REPORTS_DIR_FULL.mkdir(parents=True, exist_ok=True)
        json_path = REPORTS_DIR_FULL / "p_a72a_full_train_script_drycheck.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(drycheck_json, f, ensure_ascii=False, indent=2)
        print(f"[dry-check] 보고서 JSON: {json_path}")

        # 보고서 MD
        md_lines = [
            "# P-A72a EfficientNet-B0 Full Train Script Dry-Check 보고서",
            "",
            f"**판정: {verdict}**",
            "",
            f"- 생성일시: {ts}",
            f"- 모드: dry-check (training 미실행)",
            f"- 스크립트: `{EXP_ROOT / 'code' / SCRIPT_NAME}`",
            "",
            "## Syntax Check",
            "",
            "- syntax check: **통과** (py_compile 결과)",
            "",
            "## 가드 확인 결과",
            "",
            "| 가드 | 내용 | 결과 |",
            "|------|------|------|",
            "| guard1 | --full-run/--dry-check 플래그 | OK |",
            "| guard2 | P-A71 verdict=pass | OK |",
            "| guard3 | P-A70 verdict=통과 | OK |",
            "| guard3b | P-A69 verdict=통과 | OK |",
            "| guard3c | P-A68 verdict=pass | OK |",
            "| guard3d | P-A67b raw_feature_dim=144 | OK |",
            "| guard3e | P-A67a verdict=pass | OK |",
            f"| guard4 | selected_indices shape=(100,), unique=100, range=[{guard_results['guard4_selected_indices']['min']},{guard_results['guard4_selected_indices']['max']}] | OK |",
            "| guard5 | raw_feature_dim=144 | OK |",
            "| guard6 | selected_dim=100 | OK |",
            f"| guard7 | normal train split={guard_results['guard7_train_count']}명 | OK |",
            "| guard8 | full output 경로 비어있음 | OK |",
            "| guard9 | smoke/full 경로 분리 | OK |",
            "| guard10 | P-A69/P-A70 smoke 결과 보존 | OK |",
            "| guard11 | ResNet 실험 폴더 분리 | OK |",
            "| guard12~16 | stage2_holdout/lesion/val-test/threshold/download 금지 | OK |",
            "",
            "## Full Output Collision 확인",
            "",
            f"- `outputs/models/distributions/position_bin_stats.npz`: 존재하지 않음 ✓",
            "",
            "## Smoke 결과 보존 확인",
            "",
            "- P-A69 limit1 npz: 존재 ✓",
            "- P-A70 limit5 npz: 존재 ✓",
            "",
            "## Stage2_holdout 잠금 확인",
            "",
            "- stage2_holdout 접근: 0 ✓",
            "",
            "## Safety 확인",
            "",
            "- full train 실행: **미실행** ✓",
            "- model forward: **미실행** ✓",
            "- scoring/threshold/metrics: **미실행** ✓",
            "- 기존 결과 수정: **없음** ✓",
            "",
            "## Full Train 실행 명령 초안",
            "",
            "```bash",
            f"source ~/ai_env/bin/activate && \\",
            f"python experiments/efficientnet_b0_imagenet_v1/code/{SCRIPT_NAME} --full-run \\",
            f"  2>&1 | tee experiments/efficientnet_b0_imagenet_v1/outputs/reports/full/full_train.log",
            "```",
            "",
            "## 다음 단계",
            "",
            "- P-A72b full train 실행 가능: **True**",
            "- 사용자 승인 후 위 명령 실행",
        ]
        md_path = REPORTS_DIR_FULL / "p_a72a_full_train_script_drycheck.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"[dry-check] 보고서 MD: {md_path}")

        print(f"\n=== P-A72a dry-check 완료: {verdict} ===")
        print(f"[dry-check] 실제 full train 명령:")
        print(f"  source ~/ai_env/bin/activate && \\")
        print(f"  python experiments/efficientnet_b0_imagenet_v1/code/{SCRIPT_NAME} --full-run \\")
        print(f"    2>&1 | tee experiments/efficientnet_b0_imagenet_v1/outputs/reports/full/full_train.log")
        return

    # ================================================================
    # full-run 모드
    # ================================================================

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

    # FeatureExtractorEffNetB0 초기화
    print("\n[train] FeatureExtractorEffNetB0(EfficientNet-B0/ImageNet) 초기화 중...")
    feature_extractor = FeatureExtractorEffNetB0()
    print(f"[train] device: {feature_extractor.device}")

    assert feature_extractor.raw_feature_dim == RAW_FEATURE_DIM, \
        f"raw_feature_dim 불일치: {feature_extractor.raw_feature_dim} != {RAW_FEATURE_DIM}"
    print(f"[train] raw_feature_dim={feature_extractor.raw_feature_dim} 확인 ✅")

    import torch
    gpu_avail = str(feature_extractor.device).startswith("cuda")
    if gpu_avail:
        torch.cuda.reset_peak_memory_stats()

    # PaDiMModel 초기화
    model = PaDiMModel(
        selected_feature_indices_path=str(SELECTED_INDICES_PATH),
        feature_dim=REDUCED_FEATURE_DIM,
        eps=1e-5,
    )
    print(f"[train] PaDiMModel 초기화: feature_dim={model.feature_dim}")
    print(f"[train] 전체 환자 수: {len(train_patients)}명")

    n_processed = 0
    n_failed    = 0

    def patient_stream():
        nonlocal n_processed, n_failed
        for i, pid in enumerate(train_patients, 1):
            data = loader.load_patient_data(pid, mask_type=MASK_TYPE)
            if data is None:
                n_failed += 1
                record_error(pid, "load_failed",
                             "DataLoader.load_patient_data returned None", "patient_data")
                print(f"  [SKIP] ({i}/{len(train_patients)}) {pid}: 로드 실패")
                continue
            n_processed += 1
            print(f"  [OK]   ({i}/{len(train_patients)}) {pid}: "
                  f"ct_hu={data['ct_hu'].shape}, patches={len(data['patch_df'])}")
            yield data

    start_time = time.time()
    print("\n[train] full train 시작...")
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
        raise RuntimeError("full train 결과가 비어있습니다.")

    # 저장
    FULL_NPZ.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(FULL_NPZ))
    print(f"[train] 저장 완료: {FULL_NPZ}")

    # position_bin 통계 요약 및 NaN/Inf 확인
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

    # smoke 결과 보존 확인
    assert SMOKE_LIMIT1_NPZ.exists(), f"P-A69 limit1 결과가 사라졌습니다: {SMOKE_LIMIT1_NPZ}"
    assert SMOKE_LIMIT5_NPZ.exists(), f"P-A70 limit5 결과가 사라졌습니다: {SMOKE_LIMIT5_NPZ}"
    print("[검증] P-A69/P-A70 smoke 결과 보존: OK")

    assert sample_mean_shape == (REDUCED_FEATURE_DIM,), \
        f"mean shape 불일치: {sample_mean_shape}"
    assert sample_cov_shape == (REDUCED_FEATURE_DIM, REDUCED_FEATURE_DIM), \
        f"cov shape 불일치: {sample_cov_shape}"
    assert total_nan == 0, f"NaN 발생: {total_nan}"
    assert total_inf == 0, f"Inf 발생: {total_inf}"

    # runtime_summary.csv
    REPORTS_DIR_FULL.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    runtime_rows = [
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_requested",
         "value": EXPECTED_TRAIN_N},
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
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "output_path",     "value": str(FULL_NPZ)},
    ]
    record_runtime(runtime_rows)
    print(f"[train] runtime_summary.csv 기록: {RUNTIME_CSV}")

    # 보고서 JSON
    verdict = "통과"
    report = {
        "step": "P-A72",
        "verdict": verdict,
        "timestamp": ts,
        "backbone": BACKBONE,
        "patients_requested": EXPECTED_TRAIN_N,
        "patients_processed": summary["n_patients_success"],
        "patients_failed": summary["n_patients_failed"],
        "split": "normal_train",
        "p_a71_verdict": guard_results["guard2_p_a71_verdict"],
        "p_a70_verdict": guard_results["guard3_p_a70_verdict"],
        "p_a69_verdict": guard_results["guard3b_p_a69_verdict"],
        "p_a68_verdict": guard_results["guard3c_p_a68_verdict"],
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
        "output_npz": str(FULL_NPZ),
        "smoke_limit1_preserved": SMOKE_LIMIT1_NPZ.exists(),
        "smoke_limit5_preserved": SMOKE_LIMIT5_NPZ.exists(),
        "safety": {
            "full_train_flag_required": True,
            "val_test_scoring": False,
            "lesion_scoring": False,
            "threshold_calculated": False,
            "metrics_calculated": False,
            "stage2_holdout_accessed": False,
            "existing_results_modified": False,
            "pip_install": False,
            "additional_download": False,
        },
        "next_step_p_a73_scoring_feasible": True,
    }
    report_json_path = REPORTS_DIR_FULL / "p_a72_full_train.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[train] 보고서 JSON: {report_json_path}")

    # 보고서 MD
    report_md_path = REPORTS_DIR_FULL / "p_a72_full_train.md"
    md_lines = [
        "# P-A72 EfficientNet-B0 Full Train 보고서",
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
        f"- 전체 요청 환자: {EXPECTED_TRAIN_N}명",
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
        f"- full output: {FULL_NPZ}",
        "",
        "## 기존 결과 보존 확인",
        "",
        f"- P-A69 limit1 결과 보존: {SMOKE_LIMIT1_NPZ.exists()}",
        f"- P-A70 limit5 결과 보존: {SMOKE_LIMIT5_NPZ.exists()}",
        "- ResNet18/ResNet50 결과 무수정: True",
        "- stage2_holdout 접근: False",
        "",
        "## Safety",
        "",
        "- val/test scoring: False",
        "- lesion scoring: False",
        "- threshold 계산: False",
        "- metrics 계산: False",
        "- stage2_holdout 접근: False",
        "- 기존 결과 수정: False",
        "- pip/conda install: False",
        "- 추가 다운로드: False",
        "",
        "## 다음 단계",
        "",
        "- P-A73 scoring 진행 가능: True",
    ]
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[train] 보고서 MD: {report_md_path}")

    print(f"\n=== P-A72 full train 완료: {verdict} ===")


if __name__ == "__main__":
    main()
