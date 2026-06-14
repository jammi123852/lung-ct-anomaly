"""
P-B6: v4_20 ROI EfficientNet-B0 full train preflight (read-only)

normal_train 290명 full train 실행 전 입력/출력/가드/시간/OOM/경로충돌/resume preflight.
- 실제 forward/training/feature extraction 없음.
- v4_20 normal ROI 290 + roi_0_0 CT 290 존재/shape 확인만.
- 예상 patch/시간/npz/OOM 산정 (기존 roi_0_0 full train 실측 기준).

금지: full train 실행 / forward / feature extraction / scoring / threshold / metrics / stage2_holdout
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT  = Path(__file__).resolve().parents[1]
ROI0_BRANCH = PROJ_ROOT / "experiments" / "efficientnet_b0_imagenet_v1"

REDUCED_FEATURE_DIM = 100
V4_20_PATCH_RATIO_THRESHOLD = 0.5

V4_20_NORMAL_ROOT = PROJ_ROOT / "outputs" / "mip-postprocess-research-v1" / "masks" / "refined_roi_v4_20_modeB_all_v1" / "normal"
NORMAL_CT_ROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")

SELECTED_INDICES_PATH = EXP_ROOT / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
SPLIT_JSON = PROJ_ROOT / "outputs" / "position-aware-padim-v1" / "splits" / "normal_v1.json"

# 입력 보고서
P_B5_JSON   = EXP_ROOT / "outputs" / "reports" / "smoke" / "p_b5_train_smoke_limit5.json"
P_B4_JSON   = EXP_ROOT / "outputs" / "reports" / "smoke" / "p_b4_train_smoke_limit1.json"
P_B3_JSON   = EXP_ROOT / "outputs" / "reports" / "p_b3_lesion_safety_validation" / "p_b3_lesion_safety_validation.json"
P_B2_6_JSON = EXP_ROOT / "outputs" / "reports" / "p_b2_6_v4_20_source_lock" / "p_b2_6_v4_20_source_lock.json"

# 기존 roi_0_0 full train 실측 (P-A72)
ROI0_FULL_JSON = ROI0_BRANCH / "outputs" / "reports" / "full" / "p_a72_full_train.json"

# full train 예상 출력 (P-B7)
FULL_NPZ        = EXP_ROOT / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
FULL_REPORTS    = EXP_ROOT / "outputs" / "reports" / "full"
SMOKE_NPZ_L1    = EXP_ROOT / "outputs" / "smoke" / "train_limit1" / "position_bin_stats.npz"
SMOKE_NPZ_L5    = EXP_ROOT / "outputs" / "smoke" / "train_limit5" / "position_bin_stats.npz"

# P-B6 출력
REPORT_DIR = FULL_REPORTS
SCRIPT_NAME = "p_b6_full_train_preflight.py"


def main():
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{SCRIPT_NAME}] 시작: {ts}\n")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    issues = []
    abort = False

    # ── 가드 1~4: 선행 단계 verdict ──────────────────────────────────────
    p_b5 = json.load(open(P_B5_JSON, encoding="utf-8")) if P_B5_JSON.exists() else None
    p_b4 = json.load(open(P_B4_JSON, encoding="utf-8")) if P_B4_JSON.exists() else None
    p_b3 = json.load(open(P_B3_JSON, encoding="utf-8")) if P_B3_JSON.exists() else None
    p_b26 = json.load(open(P_B2_6_JSON, encoding="utf-8")) if P_B2_6_JSON.exists() else None

    g1 = p_b5 is not None and p_b5.get("verdict") == "통과"
    g2 = p_b4 is not None and p_b4.get("verdict") == "통과"
    g3 = p_b3 is not None and p_b3.get("verdict") in ("통과", "부분통과")
    g4 = p_b26 is not None and p_b26.get("user_correction_applied", {}).get("official_roi_source") == "refined_roi_v4_20_modeB_all_v1"
    print(f"[guard1] P-B5 통과: {g1}")
    print(f"[guard2] P-B4 통과: {g2}")
    print(f"[guard3] P-B3 통과/부분통과: {g3}")
    print(f"[guard4] P-B2.6 source lock v4_20: {g4}")
    if not (g1 and g2 and g3 and g4):
        issues.append("선행 단계(P-B5/P-B4/P-B3/P-B2.6) verdict 불충족")
        abort = True

    # ── 가드 5: selected index ───────────────────────────────────────────
    g5 = False
    sidx_info = {}
    if SELECTED_INDICES_PATH.exists():
        sidx = np.load(str(SELECTED_INDICES_PATH))
        g5 = (sidx.shape == (REDUCED_FEATURE_DIM,) and len(np.unique(sidx)) == REDUCED_FEATURE_DIM
              and sidx.min() >= 0 and sidx.max() <= 143)
        sidx_info = {"shape": list(sidx.shape), "unique": int(len(np.unique(sidx))),
                     "range": [int(sidx.min()), int(sidx.max())]}
    print(f"[guard5] selected index 검증: {g5} {sidx_info}")
    if not g5:
        issues.append("selected index 검증 실패")
        abort = True

    # ── 가드 6: normal train split 290 + normal004 test 확인 ─────────────
    split_data = json.load(open(SPLIT_JSON, encoding="utf-8"))
    train_patients = list(split_data["train"])
    test_patients  = list(split_data.get("test", []))
    p2s = split_data.get("patient_to_safe_id", {})
    g6 = len(train_patients) == 290
    normal004_in_test = "normal004" in test_patients
    normal004_in_train = "normal004" in train_patients
    print(f"[guard6] train split 290: {g6} (실제 {len(train_patients)})")
    print(f"[guard6] normal004 in test: {normal004_in_test}, in train: {normal004_in_train}")
    if not g6:
        issues.append(f"train split {len(train_patients)}≠290")
        abort = True

    # ── 6~8: normal train 290명 CT/ROI 존재 + lightweight shape check ────
    ct_missing = []
    roi_missing = []
    shape_mismatch = []
    shape_checked = 0
    roi_voxel_samples = []   # 일부 voxel 합 (추정 보조, 앞 10명만)

    for i, pid in enumerate(train_patients):
        safe_id = p2s.get(pid, pid)
        ct_path  = NORMAL_CT_ROOT / safe_id / "ct_hu.npy"
        roi_path = V4_20_NORMAL_ROOT / safe_id / "refined_roi.npy"
        ct_exists = ct_path.exists()
        roi_exists = roi_path.exists()
        if not ct_exists:
            ct_missing.append(pid)
        if not roi_exists:
            roi_missing.append(pid)
        if ct_exists and roi_exists:
            # mmap header만 읽어 shape 비교 (forward 아님)
            ct_shape  = np.load(str(ct_path),  mmap_mode='r').shape
            roi_shape = np.load(str(roi_path), mmap_mode='r').shape
            shape_checked += 1
            if ct_shape != roi_shape:
                shape_mismatch.append({"patient_id": pid, "ct": str(ct_shape), "roi": str(roi_shape)})
            if i < 10:
                roi_voxel_samples.append(int(np.asarray(np.load(str(roi_path), mmap_mode='r')).sum()))

    print(f"\n[존재] CT 누락: {len(ct_missing)}, ROI 누락: {len(roi_missing)}, "
          f"shape check: {shape_checked}/290, shape mismatch: {len(shape_mismatch)}")
    if ct_missing or roi_missing:
        issues.append(f"CT 누락 {len(ct_missing)} / ROI 누락 {len(roi_missing)}")
    if shape_mismatch:
        issues.append(f"CT/ROI shape mismatch {len(shape_mismatch)}건")

    # ── 가드 9~10: full output path 분리/collision ───────────────────────
    full_npz_exists = FULL_NPZ.exists()
    smoke_full_distinct = (FULL_NPZ != SMOKE_NPZ_L1 and FULL_NPZ != SMOKE_NPZ_L5)
    smoke_preserved = SMOKE_NPZ_L1.exists() and SMOKE_NPZ_L5.exists()
    print(f"[guard9] full npz 이미 존재: {full_npz_exists} (존재 시 P-B7에서 abort)")
    print(f"[guard10] smoke/full 경로 분리: {smoke_full_distinct}, smoke 보존: {smoke_preserved}")
    if full_npz_exists:
        issues.append("full output npz 이미 존재 (P-B7 실행 시 덮어쓰기 금지로 중단)")
    if not smoke_full_distinct:
        issues.append("smoke/full 경로 충돌")
        abort = True

    # 기존 branch collision (다른 branch 경로이므로 분리)
    roi0_full_npz = ROI0_BRANCH / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
    branch_path_isolated = (FULL_NPZ != roi0_full_npz)
    print(f"[guard11] 기존 roi_0_0 branch와 경로 분리: {branch_path_isolated}")

    # ── 예상 산정 (기존 roi_0_0 full train 실측 기준) ────────────────────
    roi0_full = json.load(open(ROI0_FULL_JSON, encoding="utf-8")) if ROI0_FULL_JSON.exists() else {}
    roi0_patches = roi0_full.get("n_patches_used")          # 12,130,820
    roi0_elapsed = roi0_full.get("elapsed_seconds")          # 1088.79
    roi0_peak_gpu = roi0_full.get("peak_gpu_gb")             # 0.059

    p_b5_removed_ratio = p_b5.get("total_removed_ratio") if p_b5 else None  # 0.0666
    est = {}
    if roi0_patches and p_b5_removed_ratio is not None:
        est_used = int(round(roi0_patches * (1 - p_b5_removed_ratio)))
        est["roi0_full_patches"] = roi0_patches
        est["v4_removed_ratio_from_p_b5"] = p_b5_removed_ratio
        est["estimated_v4_used_patches"] = est_used
        est["estimated_v4_removed_patches"] = roi0_patches - est_used
    if roi0_elapsed:
        # patch 약간 감소하나 slice forward가 주 비용 → 거의 동일~약간 감소. 보수 1.5×/2× 병기.
        est["roi0_full_elapsed_sec"] = roi0_elapsed
        est["estimated_elapsed_sec_baseline"] = round(roi0_elapsed * (1 - (p_b5_removed_ratio or 0)), 1)
        est["estimated_elapsed_sec_conservative_1_5x"] = round(roi0_elapsed * 1.5, 1)
        est["estimated_elapsed_sec_conservative_2x"] = round(roi0_elapsed * 2.0, 1)
        est["estimated_elapsed_readable_baseline"] = f"{roi0_elapsed/60:.1f}분"
    # npz 크기: 구조 동일 (mean 100 + cov 100x100, position_bin 10개) → 기존과 거의 동일
    roi0_npz_size = roi0_full_npz.stat().st_size if roi0_full_npz.exists() else None
    est["roi0_full_npz_bytes"] = roi0_npz_size
    est["estimated_v4_npz_bytes"] = roi0_npz_size   # 구조 동일
    est["estimated_v4_npz_readable"] = f"{(roi0_npz_size or 0)/1024:.0f} KB"
    # OOM: streaming, smoke peak 0.059GB와 동일
    est["estimated_peak_gpu_gb"] = roi0_peak_gpu or p_b5.get("peak_gpu_gb")
    est["oom_risk"] = "낮음 (streaming 누적, peak GPU≈0.059GB, 전체 patch 메모리 적재 없음)"

    print(f"\n[추정] v4 used patches ≈ {est.get('estimated_v4_used_patches'):,}" if est.get('estimated_v4_used_patches') else "")
    print(f"[추정] elapsed baseline ≈ {est.get('estimated_elapsed_sec_baseline')}초 "
          f"(보수 1.5× {est.get('estimated_elapsed_sec_conservative_1_5x')}초)")
    print(f"[추정] npz ≈ {est.get('estimated_v4_npz_readable')}, peak GPU ≈ {est.get('estimated_peak_gpu_gb')}GB")

    # ── 판정 ──────────────────────────────────────────────────────────────
    if abort:
        verdict = "실패"
    elif ct_missing or roi_missing or shape_mismatch or full_npz_exists:
        verdict = "부분통과"
    else:
        verdict = "통과"
    print(f"\n[판정] {verdict}")
    for i in issues:
        print(f"  ⚠ {i}")

    p_b7_can_proceed = (verdict == "통과")

    full_train_cmd = ("source ~/ai_env/bin/activate && python "
                      "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/code/p_b7_full_train.py")

    expected_outputs = [
        str(FULL_NPZ),
        str(FULL_REPORTS / "p_b7_full_train.md"),
        str(FULL_REPORTS / "p_b7_full_train.json"),
        str(FULL_REPORTS / "p_b7_runtime_summary.csv"),
        str(FULL_REPORTS / "p_b7_patch_filtering_summary.csv"),
        str(FULL_REPORTS / "full_train.log"),
    ]

    report = {
        "stage": "P-B6_full_train_preflight",
        "created": ts,
        "verdict": verdict,
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
            "p_b5_verdict": p_b5.get("verdict") if p_b5 else None,
            "p_b4_verdict": p_b4.get("verdict") if p_b4 else None,
            "p_b3_verdict": p_b3.get("verdict") if p_b3 else None,
            "p_b2_6_source_locked": g4,
            "official_roi_source": "refined_roi_v4_20_modeB_all_v1",
        },
        "selected_index": sidx_info,
        "normal_train_split": {
            "count": len(train_patients), "expected": 290, "match": g6,
            "normal004_in_train": normal004_in_train,
            "normal004_in_test": normal004_in_test,
            "normal004_note": "normal004는 test split 소속 → train split(290)에 없음. train smoke 앞 5명에서 자연 제외(정상).",
        },
        "ct_roi_existence": {
            "ct_missing_count": len(ct_missing),
            "roi_missing_count": len(roi_missing),
            "ct_missing_sample": ct_missing[:5],
            "roi_missing_sample": roi_missing[:5],
            "shape_checked": shape_checked,
            "shape_mismatch_count": len(shape_mismatch),
            "shape_mismatch_sample": shape_mismatch[:5],
            "roi_voxel_first10_sample": roi_voxel_samples,
        },
        "output_path": {
            "full_npz": str(FULL_NPZ),
            "full_npz_exists": full_npz_exists,
            "smoke_full_distinct": smoke_full_distinct,
            "smoke_l1_preserved": SMOKE_NPZ_L1.exists(),
            "smoke_l5_preserved": SMOKE_NPZ_L5.exists(),
            "branch_path_isolated_from_roi0": branch_path_isolated,
        },
        "estimates": est,
        "expected_outputs": expected_outputs,
        "full_train_command_draft": full_train_cmd,
        "p_b7_full_train_feasible": p_b7_can_proceed,
        "issues": issues,
    }
    with open(REPORT_DIR / "p_b6_full_train_preflight.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── MD ───────────────────────────────────────────────────────────────
    md = [
        "# P-B6 v4_20 ROI EfficientNet-B0 Full Train Preflight\n",
        f"**판정: {verdict}**\n",
        f"- 생성일시: {ts}",
        f"- branch: efficientnet_b0_imagenet_chestwall_removed_roi_v1\n",
        "## 0. 범위\n",
        "- full train **미실행** / forward·feature extraction **없음** / read-only preflight\n",
        "## 1. 선행 단계 입력 검증\n",
        "| 단계 | verdict |",
        "|------|---------|",
        f"| P-B5 | {p_b5.get('verdict') if p_b5 else None} |",
        f"| P-B4 | {p_b4.get('verdict') if p_b4 else None} |",
        f"| P-B3 | {p_b3.get('verdict') if p_b3 else None} |",
        f"| P-B2.6 source lock | {'v4_20 ✅' if g4 else 'NG'} |\n",
        "## 2. selected index\n",
        f"- {sidx_info}\n",
        "## 3. normal train split\n",
        f"- train 290명: {g6}",
        f"- normal004 in test: {normal004_in_test} / in train: {normal004_in_train}",
        "- → normal004는 test split 소속이라 train 290에 없음. P-B5 smoke 앞 5명(001,002,003,005,006)에서 자연 제외(정상).\n",
        "## 4. CT/ROI 존재 + shape check (290명)\n",
        "| 항목 | 결과 |",
        "|------|------|",
        f"| CT 누락 | {len(ct_missing)} |",
        f"| v4_20 ROI 누락 | {len(roi_missing)} |",
        f"| shape check 수행 | {shape_checked}/290 |",
        f"| CT/ROI shape mismatch | {len(shape_mismatch)} |\n",
        "## 5. 출력 경로 분리 / collision\n",
        "| 항목 | 결과 |",
        "|------|------|",
        f"| full npz 이미 존재 | {full_npz_exists} |",
        f"| smoke/full 경로 분리 | {smoke_full_distinct} |",
        f"| smoke limit1 보존 | {SMOKE_NPZ_L1.exists()} |",
        f"| smoke limit5 보존 | {SMOKE_NPZ_L5.exists()} |",
        f"| 기존 roi_0_0 branch와 경로 분리 | {branch_path_isolated} |\n",
        "## 6. full train 예상 산정 (기존 roi_0_0 full train 실측 기준)\n",
        "| 지표 | 값 |",
        "|------|----|",
        f"| 기존 roi_0_0 full patch | {est.get('roi0_full_patches'):,} |" if est.get('roi0_full_patches') else "| 기존 patch | N/A |",
        f"| P-B5 제거율 적용 | {est.get('v4_removed_ratio_from_p_b5')} |",
        f"| **예상 v4 used patch** | **{est.get('estimated_v4_used_patches'):,}** |" if est.get('estimated_v4_used_patches') else "",
        f"| 예상 v4 제거 patch | {est.get('estimated_v4_removed_patches'):,} |" if est.get('estimated_v4_removed_patches') else "",
        f"| 기존 full elapsed | {est.get('roi0_full_elapsed_sec')}초 ({est.get('estimated_elapsed_readable_baseline')}) |",
        f"| 예상 elapsed (baseline) | {est.get('estimated_elapsed_sec_baseline')}초 |",
        f"| 예상 elapsed (보수 1.5×) | {est.get('estimated_elapsed_sec_conservative_1_5x')}초 |",
        f"| 예상 elapsed (보수 2×) | {est.get('estimated_elapsed_sec_conservative_2x')}초 |",
        f"| 예상 npz 크기 | {est.get('estimated_v4_npz_readable')} (구조 동일) |",
        f"| 예상 peak GPU | {est.get('estimated_peak_gpu_gb')} GB |",
        f"| OOM 위험 | {est.get('oom_risk')} |\n",
        "## 7. 생성될 파일 (P-B7)\n",
    ]
    for o in expected_outputs:
        md.append(f"- `{o}`")
    md += [
        "",
        "## 8. full train 실행 명령 초안 (P-B7, 이번 단계 미실행)\n",
        "```bash",
        full_train_cmd,
        "```\n",
        "## 9. 미사용 / 무수정 확인\n",
        "- model_roi.npy / E드라이브 / lesion 파일 / stage2_holdout: 미사용",
        "- full train / forward / feature extraction / scoring / threshold / metrics: 미실행",
        "- 기존 roi_0_0 / EfficientNet-B0 / P-B1~P-B5 결과: 무수정",
        "- P-B4 limit1 / P-B5 limit5 smoke 결과: 보존\n",
        "## 10. 미결 사항\n",
    ]
    for i in issues:
        md.append(f"- ⚠ {i}")
    if not issues:
        md.append("- 없음")
    md += [
        "",
        "## 11. 최종 판정\n",
        f"- **{verdict}**",
        f"- P-B7 full train 진행 가능: **{p_b7_can_proceed}**",
    ]
    with open(REPORT_DIR / "p_b6_full_train_preflight.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n[저장] {REPORT_DIR}")
    print(f"[완료] 판정: {verdict}, P-B7 가능: {p_b7_can_proceed}")
    return 0 if verdict != "실패" else 1


if __name__ == "__main__":
    sys.exit(main())
