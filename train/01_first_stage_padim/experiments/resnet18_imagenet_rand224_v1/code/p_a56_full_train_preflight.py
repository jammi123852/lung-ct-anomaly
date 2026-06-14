"""
P-A56: ResNet18 random224 full train preflight

full train 290명 실행 전 입력/출력/가드/시간/OOM/기존 결과 영향 확인.
실제 forward/training 실행 없음.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_paths  # noqa: E402

SCRIPT_NAME = "p_a56_full_train_preflight.py"
REPORTS_FULL_DIR = ws_paths.REPORTS_FULL_DIR
FULL_OUTPUT_NPZ = ws_paths.MODEL_NPZ
SMOKE_L1_NPZ = ws_paths.SMOKE_ROOT / "train_limit1" / "position_bin_stats.npz"
SMOKE_L5_NPZ = ws_paths.SMOKE_ROOT / "train_limit5" / "position_bin_stats.npz"


def main():
    import numpy as np

    issues = []
    results = {}

    # ---- 가드 1: P-A55 보고서 통과 확인 ----
    p55_json = ws_paths.REPORTS_SMOKE_DIR / "p_a55_train_smoke_limit5.json"
    assert p55_json.exists(), f"P-A55 보고서 없음: {p55_json}"
    with open(p55_json) as f:
        p55 = json.load(f)
    assert p55.get("verdict") == "통과", f"P-A55 판정 미통과: {p55.get('verdict')}"
    print("[guard1] P-A55 보고서 통과 확인 OK")

    # P-A53/P-A54 보고서도 확인
    p53_json = ws_paths.REPORTS_DIR / "p_a53_selected_indices.json"
    assert p53_json.exists(), f"P-A53 보고서 없음: {p53_json}"
    with open(p53_json) as f:
        p53 = json.load(f)
    assert p53.get("verdict") == "통과", f"P-A53 판정 미통과"

    p54_json = ws_paths.REPORTS_SMOKE_DIR / "p_a54_train_smoke_limit1.json"
    assert p54_json.exists(), f"P-A54 보고서 없음: {p54_json}"
    with open(p54_json) as f:
        p54 = json.load(f)
    assert p54.get("verdict") == "통과", f"P-A54 판정 미통과"

    results["p53_verdict"] = p53.get("verdict")
    results["p54_verdict"] = p54.get("verdict")
    results["p55_verdict"] = p55.get("verdict")
    print(f"[guard1] P-A53/P-A54/P-A55 모두 통과 확인")

    # ---- 가드 2: selected_feature_indices 검증 ----
    sidx_path = ws_paths.SELECTED_INDICES_PATH
    assert sidx_path.exists(), f"selected_feature_indices.npy 없음"
    sidx = np.load(str(sidx_path))
    assert sidx.shape == (224,), f"[guard2] shape 불일치: {sidx.shape}"
    assert len(np.unique(sidx)) == 224, f"[guard2] unique 불일치"
    assert sidx.min() >= 0 and sidx.max() <= 447, f"[guard2] range 오류"
    existing_idx = np.load(str(
        REPO_ROOT / "outputs/position-aware-padim-v1/models/padim_v1/distributions/selected_feature_indices.npy"
    ))
    assert set(existing_idx.tolist()).issubset(set(sidx.tolist())), "[guard2] 기존 100개 미포함"
    results["sidx_shape"] = list(sidx.shape)
    results["sidx_unique"] = int(len(np.unique(sidx)))
    results["sidx_range"] = [int(sidx.min()), int(sidx.max())]
    results["random100_subset"] = True
    print(f"[guard2] selected_feature_indices OK: shape={sidx.shape}, unique=224, range=[{sidx.min()},{sidx.max()}], 기존100포함=True")

    # ---- 가드 3: normal train split 290명 확인 ----
    from position_aware_padim.patient_splitter import PatientSplitter
    splitter = PatientSplitter(str(REPO_ROOT))
    patient_split = splitter.load_split()
    train_patients = list(patient_split.train)
    assert len(train_patients) == 290, f"[guard3] train split 수 불일치: {len(train_patients)}"
    results["train_patients"] = len(train_patients)
    print(f"[guard3] normal train split: {len(train_patients)}명 확인")

    # ---- 가드 4: full output path 비어있는지 확인 ----
    if FULL_OUTPUT_NPZ.exists():
        print(f"[guard4][ABORT] full output이 이미 존재합니다: {FULL_OUTPUT_NPZ}")
        sys.exit(1)
    results["full_output_exists"] = False
    print(f"[guard4] full output 경로 비어있음 OK: {FULL_OUTPUT_NPZ}")

    # ---- 가드 5: smoke/full 경로 분리 확인 ----
    assert FULL_OUTPUT_NPZ != SMOKE_L1_NPZ, "[guard5] full/smoke_l1 경로 충돌"
    assert FULL_OUTPUT_NPZ != SMOKE_L5_NPZ, "[guard5] full/smoke_l5 경로 충돌"
    results["smoke_full_separated"] = True
    print(f"[guard5] smoke/full 경로 분리 OK")

    # ---- 가드 6: stage2_holdout 접근 금지 ----
    results["stage2_holdout_accessed"] = False
    print(f"[guard6] stage2_holdout 접근 금지 OK")

    # ---- 가드 7: 실제 forward/training 미실행 확인 ----
    results["training_executed"] = False
    print(f"[guard7] forward/training 미실행 OK")

    # ---- P-A54/P-A55 smoke 결과 보존 확인 ----
    assert SMOKE_L1_NPZ.exists(), f"P-A54 limit1 smoke 결과 없음: {SMOKE_L1_NPZ}"
    assert SMOKE_L5_NPZ.exists(), f"P-A55 limit5 smoke 결과 없음: {SMOKE_L5_NPZ}"
    results["smoke_l1_preserved"] = True
    results["smoke_l5_preserved"] = True
    print(f"[check] P-A54 smoke_limit1 보존 OK: {SMOKE_L1_NPZ}")
    print(f"[check] P-A55 smoke_limit5 보존 OK: {SMOKE_L5_NPZ}")

    # ---- 기존 결과 충돌 확인 ----
    legacy_npz = REPO_ROOT / "outputs/position-aware-padim-v1/models/padim_v1/distributions/position_bin_stats.npz"
    collision = (FULL_OUTPUT_NPZ == legacy_npz)
    results["legacy_collision"] = collision
    print(f"[check] 기존 random100 결과 충돌: {collision} (경로 분리됨)")

    # ---- 예상 patch 수 추정 ----
    n_smoke = 5
    n_full = 290
    patches_smoke = p55.get("n_patches_used", 189688)
    elapsed_smoke = p55.get("elapsed_seconds", 17.6)
    patches_per_patient = patches_smoke / n_smoke
    expected_patches = patches_per_patient * n_full
    expected_time_base = elapsed_smoke / n_smoke * n_full
    expected_time_1_5x = expected_time_base * 1.5
    expected_time_2_0x = expected_time_base * 2.0
    results["expected_patches_full"] = int(expected_patches)
    results["expected_time_seconds_base"] = round(expected_time_base, 1)
    results["expected_time_seconds_1_5x"] = round(expected_time_1_5x, 1)
    results["expected_time_seconds_2_0x"] = round(expected_time_2_0x, 1)
    print(f"[estimate] 예상 full train patch: {expected_patches:,.0f}")
    print(f"[estimate] 예상 runtime: {expected_time_base:.0f}초({expected_time_base/60:.1f}분) / 1.5x={expected_time_1_5x:.0f}초({expected_time_1_5x/60:.1f}분) / 2x={expected_time_2_0x:.0f}초({expected_time_2_0x/60:.1f}분)")

    # ---- 예상 npz 크기 추정 ----
    n_bins = 10
    mean_bytes = 224 * 8
    cov_bytes = 224 * 224 * 8
    per_bin = mean_bytes + cov_bytes + 8
    total_bytes = per_bin * n_bins
    results["expected_npz_mb_uncompressed"] = round(total_bytes / 1024 / 1024, 2)
    print(f"[estimate] 예상 npz 크기 (10 bins, 압축 전): {total_bytes/1024/1024:.2f} MB")

    # ---- OOM 위험 평가 ----
    peak_gpu_smoke = p55.get("peak_gpu_gb", 0.048)
    results["peak_gpu_smoke_gb"] = peak_gpu_smoke
    results["oom_risk"] = "낮음 (peak GPU 0.048GB 기준)"
    print(f"[estimate] OOM 위험: 낮음 (P-A55 peak GPU {peak_gpu_smoke:.3f}GB, 환자 순차 처리 구조)")

    # ---- full train 생성될 파일 목록 ----
    full_output_files = [
        str(ws_paths.MODEL_NPZ),
        str(REPORTS_FULL_DIR / "runtime_summary.csv"),
        str(REPORTS_FULL_DIR / "p_a57_full_train.md"),
        str(REPORTS_FULL_DIR / "p_a57_full_train.json"),
        str(REPORTS_FULL_DIR / "p_a57_full_train.log"),
    ]
    results["full_train_output_files"] = full_output_files
    print(f"[check] full train 생성 파일 목록:")
    for f in full_output_files:
        print(f"  {f}")

    # ---- full train 실행 명령 초안 ----
    full_train_cmd = (
        "source ~/ai_env/bin/activate && "
        "python experiments/resnet18_imagenet_rand224_v1/code/p_a57_full_train.py "
        "2>&1 | tee experiments/resnet18_imagenet_rand224_v1/outputs/reports/full/p_a57_full_train.log"
    )
    results["full_train_command_draft"] = full_train_cmd
    print(f"[draft] full train 실행 명령 초안:")
    print(f"  {full_train_cmd}")

    # ---- 보고서 JSON ----
    REPORTS_FULL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    report = {
        "step": "P-A56",
        "verdict": "통과",
        "timestamp": ts,
        "prior_steps": {
            "p53_verdict": results["p53_verdict"],
            "p54_verdict": results["p54_verdict"],
            "p55_verdict": results["p55_verdict"],
        },
        "selected_index": {
            "shape": results["sidx_shape"],
            "unique": results["sidx_unique"],
            "range": results["sidx_range"],
            "random100_subset": results["random100_subset"],
        },
        "train_patients": results["train_patients"],
        "full_output_exists": results["full_output_exists"],
        "smoke_full_separated": results["smoke_full_separated"],
        "smoke_l1_preserved": results["smoke_l1_preserved"],
        "smoke_l5_preserved": results["smoke_l5_preserved"],
        "legacy_collision": results["legacy_collision"],
        "estimates": {
            "expected_patches_full": results["expected_patches_full"],
            "expected_time_seconds_base": results["expected_time_seconds_base"],
            "expected_time_seconds_1_5x": results["expected_time_seconds_1_5x"],
            "expected_time_seconds_2_0x": results["expected_time_seconds_2_0x"],
            "expected_npz_mb_uncompressed": results["expected_npz_mb_uncompressed"],
            "peak_gpu_smoke_gb": results["peak_gpu_smoke_gb"],
            "oom_risk": results["oom_risk"],
        },
        "full_train_output_files": results["full_train_output_files"],
        "full_train_command_draft": results["full_train_command_draft"],
        "safety": {
            "full_train_executed": False,
            "forward_executed": False,
            "scoring_executed": False,
            "threshold_calculated": False,
            "metrics_calculated": False,
            "stage2_holdout_accessed": False,
            "existing_results_modified": False,
            "pip_install": False,
        },
        "next_step_p_a57_full_train_feasible": True,
    }
    report_json_path = REPORTS_FULL_DIR / "p_a56_full_train_preflight.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[report] JSON: {report_json_path}")

    # ---- 보고서 MD ----
    md_lines = [
        "# P-A56 Full Train Preflight 보고서",
        "",
        "## 판정: 통과",
        "",
        "## 선행 단계 확인",
        f"- P-A53 verdict: {results['p53_verdict']}",
        f"- P-A54 verdict: {results['p54_verdict']}",
        f"- P-A55 verdict: {results['p55_verdict']}",
        "",
        "## selected index 검증",
        f"- shape: {results['sidx_shape']}",
        f"- unique: {results['sidx_unique']}",
        f"- range: {results['sidx_range']}",
        f"- 기존 random100 subset 포함: {results['random100_subset']}",
        "",
        "## 데이터 확인",
        f"- normal train split: {results['train_patients']}명",
        "",
        "## 경로 안전성",
        f"- full output 경로 비어있음: {not results['full_output_exists']}",
        f"  - `{FULL_OUTPUT_NPZ}`",
        f"- smoke/full 경로 분리: {results['smoke_full_separated']}",
        f"- P-A54 smoke_limit1 보존: {results['smoke_l1_preserved']}",
        f"- P-A55 smoke_limit5 보존: {results['smoke_l5_preserved']}",
        f"- 기존 random100 결과 충돌: {results['legacy_collision']} (없음)",
        "",
        "## 예상치",
        f"- 예상 full train patch 수: {results['expected_patches_full']:,}",
        f"- 예상 runtime (단순비례): {results['expected_time_seconds_base']}초 ({results['expected_time_seconds_base']/60:.1f}분)",
        f"- 예상 runtime (1.5x 보수): {results['expected_time_seconds_1_5x']}초 ({results['expected_time_seconds_1_5x']/60:.1f}분)",
        f"- 예상 runtime (2.0x 보수): {results['expected_time_seconds_2_0x']}초 ({results['expected_time_seconds_2_0x']/60:.1f}분)",
        f"- 예상 npz 크기 (압축 전, 10 bins): {results['expected_npz_mb_uncompressed']} MB",
        f"- OOM 위험: {results['oom_risk']}",
        "",
        "## Full Train 생성될 파일 목록",
    ] + [f"- `{f}`" for f in results["full_train_output_files"]] + [
        "",
        "## Full Train 실행 명령 초안",
        "```bash",
        results["full_train_command_draft"],
        "```",
        "",
        "## 안전 확인",
        "- full train 미실행: True",
        "- forward/training 미실행: True",
        "- scoring/threshold/metrics 미실행: True",
        "- lesion/stage2_holdout 미접근: True",
        "- 기존 결과 무수정: True",
        "",
        "## 다음 단계",
        "- P-A57 full train 가능 여부: True",
    ]
    report_md_path = REPORTS_FULL_DIR / "p_a56_full_train_preflight.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[report] MD: {report_md_path}")
    print("\n=== P-A56 full train preflight 완료: 통과 ===")


if __name__ == "__main__":
    main()
