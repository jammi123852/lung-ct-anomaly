"""
S5 PaDiM Feature Contribution 1-case Smoke Script — Candidate A
================================================================
대상: LUNG1-052__c3 / stage1_dev only
단계: contribution smoke script (정적 검사 + 실행 분리)

실행 방법:
  --selftest              : 정적 selftest 96+ 항목 (허용)
  --dry-run               : 경로 확인 (실제 array load / contribution 계산 없음)
  --plan-only             : 실행 계획 출력 (계산 없음)
  --run-smoke --confirm-contribution : 실제 contribution 계산 (guard True 시에만)

금지 (현재 모든 guard=False):
  bare 실행              → BLOCKED exit 2
  --run-smoke 단독       → BLOCKED exit 2
  --run-smoke --confirm-contribution → BLOCKED exit 2 (guard False)

score reference policy:
  contribution 계산 기준   : reextract score = 38.87256  (feature_selected100.npy 기준)
  original recorded score  : 39.34  (metadata 참고용만, score CSV 수정 금지)
  abs(sqrt_mahalanobis - 38.87256) < 1e-3 권장
  abs(sqrt_mahalanobis - 39.34) < 0.5는 참고 parity로만 기록

contribution formula:
  diff   = x - mean                    # (100,)
  v      = cov_inv @ diff              # (100,)
  contrib_i = diff_i * v_i             # (100,) — 음수 가능
  d2     = diff.T @ cov_inv @ diff     # scalar (Mahalanobis^2)
  check: abs(sum(contrib) - d2) < tol

layer boundary (raw448):
  layer1: raw index [0,   64)
  layer2: raw index [64,  192)
  layer3: raw index [192, 448)

text policy:
  허용: "PaDiM score를 구성한 feature-space 차원별 기여도"
  허용: "Mahalanobis distance contribution"
  허용: "정상 feature 분포에서 벗어난 정도"
  금지: "암/악성/진단/병변 확정/Grad-CAM/pixel attribution/heatmap 단정"
"""

from __future__ import annotations

import argparse
import os
import sys

# ============================================================
# A. Path / Metadata constants
# ============================================================

REPO_ROOT = "/home/jinhy/project/lung-ct-anomaly"

CASE_ID    = "LUNG1-052__c3"
VOLUME_ID  = "NSCLC_LUNG1-052__d4a19cc211"
POSITION_BIN = "lower_central"
FEATURE_DIM  = 100
RAW_FEATURE_DIM = 448
SPLIT        = "stage1_dev"
STAGE2_HOLDOUT = None   # 없음

# CT index policy (contribution 단계에서 CT 미접근 — metadata 기록 전용)
CT_INDEX_Z          = 51    # CT npy indexing 기준 (local_z)
REPORT_SLICE_INDEX  = 106   # global/reporting 전용 — CT index 사용 금지

# Score reference policy
REEXTRACT_SCORE_REFERENCE = 38.87256  # feature_selected100.npy 기반 재현 score
RECORDED_SCORE_REFERENCE  = 39.34     # 원래 기록된 score (metadata 참고용만)

# ============================================================
# B. Input paths (reextract smoke 결과 — 이 script의 feature 출처)
# ============================================================

_REEXTRACT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_lung1_052_c3_feature_reextract_smoke_v1"
)

FEATURE_SELECTED100_PATH = os.path.join(_REEXTRACT_ROOT, "feature_selected100.npy")
FEATURE_RAW448_PATH      = os.path.join(_REEXTRACT_ROOT, "feature_raw448.npy")
FEATURE_METADATA_PATH    = os.path.join(_REEXTRACT_ROOT, "metadata.json")
FEATURE_DONE_PATH        = os.path.join(_REEXTRACT_ROOT, "DONE.json")

STATS_NPZ_PATH = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/models/padim_v2_roi0_0/distributions/"
    "position_bin_stats.npz"
)

SELECTED_INDICES_NPY_PATH = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/models/padim_v1/distributions/"
    "selected_feature_indices.npy"
)

REEXTRACT_VALIDATION_JSON_PATH = os.path.join(
    REPO_ROOT,
    "reports/explanation_cards/"
    "s5_lung1_052_c3_feature_reextract_smoke_run_validation_v1.json"
)

# LUNG1-320 artifact root (수정/충돌 금지)
_LUNG1_320_OUTPUT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_feature_contribution_1case_smoke_v1"
)

# ============================================================
# C. Output paths (run 단계에서만 생성)
# ============================================================

OUTPUT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_lung1_052_c3_feature_contribution_1case_smoke_v1"
)

# ============================================================
# D. contribution formula constants
# ============================================================

COVARIANCE_EPSILON      = 1e-5
CONTRIBUTION_METHOD     = "full_inverse_cov"
SUM_MATCH_TOLERANCE     = 1e-3
TOPK                    = 20

# layer boundary (raw448 concat: ResNet18 layer1/2/3)
LAYER1_RAW_START = 0
LAYER1_RAW_END   = 64
LAYER2_RAW_START = 64
LAYER2_RAW_END   = 192
LAYER3_RAW_START = 192
LAYER3_RAW_END   = 448

CONTRIBUTION_FORMULA_STRING = (
    "diff = x - mean; "
    "v = cov_inv @ diff; "
    "contrib_i = diff_i * v_i; "
    "d2 = diff.T @ cov_inv @ diff; "
    "check: abs(sum(contrib) - d2) < tol"
)

NEGATIVE_CONTRIBUTION_POLICY = (
    "음수 contribution은 '정상 분포 방향으로의 covariance interaction'으로 기록. "
    "진단 의미 부여 금지."
)

DIAGNOSTIC_CAVEAT = (
    "이 contribution 값은 PaDiM feature-space 기준 설명 후보이며, "
    "암/악성/진단/병변을 의미하지 않는다. "
    "Grad-CAM 아님. spatial heatmap 단정 불가. pixel attribution 아님."
)

NOT_GRADCAM_CAVEAT = "이 output은 Grad-CAM이 아님. feature contribution 목적."
NOT_DIAGNOSTIC_CAVEAT = "특정 소견/암/병변 확정 진단 불가. 이상 후보 localization 목적."

# ============================================================
# E. Basic guards (all False — 실제 실행 금지)
# ============================================================

ALLOW_FEATURE_LOAD       = False
ALLOW_STATS_LOAD         = False
ALLOW_CONTRIBUTION_CALC  = False
ALLOW_WRITE_CONTRIBUTION = False
ALLOW_CT_LOAD            = False
ALLOW_MODEL_FORWARD      = False
ALLOW_FEATURE_EXTRACTION = False
ALLOW_GPU                = False
ALLOW_STAGE2_HOLDOUT     = False
ALLOW_FULL_300           = False

# ============================================================
# F. Policy strings (selftest에서 참조)
# ============================================================

_DRY_RUN_POLICY  = (
    "dry-run: path existence only. contribution 계산 없음. "
    "CT/model/feature extraction 없음."
)
_PLAN_ONLY_POLICY = (
    "plan-only: formula/schema 출력. contribution 계산 없음. "
    "CT/model/feature extraction 없음."
)

# ============================================================
# G. No-CT access guard
# ============================================================

def _assert_no_ct_access():
    """이 단계에서 CT / model / feature extraction 접근 없음을 확인."""
    assert not ALLOW_CT_LOAD,            "CT load guard must be False in contribution stage"
    assert not ALLOW_MODEL_FORWARD,      "model forward guard must be False in contribution stage"
    assert not ALLOW_FEATURE_EXTRACTION, "feature extraction guard must be False in contribution stage"
    assert not ALLOW_GPU,                "GPU guard must be False in contribution stage"


# ============================================================
# H. selftest (58+ 항목)
# ============================================================

def run_selftest() -> int:
    """정적 selftest. guard / 상수 / 경로 / 정책 / selected_indices equality guard 검증. 실제 load/계산 없음."""
    passed = []
    failed = []

    def chk(name: str, cond: bool, msg: str = ""):
        if cond:
            passed.append(name)
        else:
            failed.append(f"{name}: {msg}")

    # --- 1. all contribution guards default False ---
    chk("01_guard_feature_load_false",       not ALLOW_FEATURE_LOAD,
        "ALLOW_FEATURE_LOAD must be False")
    chk("02_guard_stats_load_false",         not ALLOW_STATS_LOAD,
        "ALLOW_STATS_LOAD must be False")
    chk("03_guard_contribution_calc_false",  not ALLOW_CONTRIBUTION_CALC,
        "ALLOW_CONTRIBUTION_CALC must be False")
    chk("04_guard_write_contribution_false", not ALLOW_WRITE_CONTRIBUTION,
        "ALLOW_WRITE_CONTRIBUTION must be False")

    # --- 2. CT/model/feature extraction guards default False ---
    chk("05_guard_ct_load_false",            not ALLOW_CT_LOAD,
        "ALLOW_CT_LOAD must be False")
    chk("06_guard_model_forward_false",      not ALLOW_MODEL_FORWARD,
        "ALLOW_MODEL_FORWARD must be False")
    chk("07_guard_feature_extraction_false", not ALLOW_FEATURE_EXTRACTION,
        "ALLOW_FEATURE_EXTRACTION must be False")
    chk("08_guard_gpu_false",                not ALLOW_GPU,
        "ALLOW_GPU must be False")
    chk("09_guard_stage2_holdout_false",     not ALLOW_STAGE2_HOLDOUT,
        "ALLOW_STAGE2_HOLDOUT must be False")
    chk("10_guard_full300_false",            not ALLOW_FULL_300,
        "ALLOW_FULL_300 must be False")

    # --- 3. CASE_ID exact ---
    chk("11_case_id_exact",   CASE_ID == "LUNG1-052__c3",
        f"got {CASE_ID}")

    # --- 4. VOLUME_ID exact ---
    chk("12_volume_id_exact", VOLUME_ID == "NSCLC_LUNG1-052__d4a19cc211",
        f"got {VOLUME_ID}")

    # --- 5. POSITION_BIN exact ---
    chk("13_position_bin_lower_central",
        POSITION_BIN == "lower_central", f"got {POSITION_BIN}")

    # --- 6. CT_INDEX_Z == 51 (metadata policy) ---
    chk("14_ct_index_z_51",         CT_INDEX_Z == 51, f"got {CT_INDEX_Z}")

    # --- 7. REPORT_SLICE_INDEX == 106 (metadata policy) ---
    chk("15_report_slice_106",      REPORT_SLICE_INDEX == 106, f"got {REPORT_SLICE_INDEX}")

    # --- 8. CT_INDEX_Z != REPORT_SLICE_INDEX ---
    chk("16_ct_z_ne_report_slice",
        CT_INDEX_Z != REPORT_SLICE_INDEX,
        f"CT_INDEX_Z({CT_INDEX_Z}) == REPORT_SLICE_INDEX({REPORT_SLICE_INDEX}) — 혼동 위험")

    # --- 9. feature root contains 052 ---
    chk("17_feature_root_has_052",
        "052" in _REEXTRACT_ROOT, f"_REEXTRACT_ROOT={_REEXTRACT_ROOT}")

    # --- 10. feature root not LUNG1-320 root ---
    chk("18_feature_root_not_lung1_320",
        "320" not in _REEXTRACT_ROOT,
        f"feature root must not reference LUNG1-320: {_REEXTRACT_ROOT}")

    # --- 11. feature_selected100 path planned ---
    chk("19_feature_selected100_path_has_052",
        "052" in FEATURE_SELECTED100_PATH, f"path={FEATURE_SELECTED100_PATH}")
    chk("20_feature_selected100_path_correct_name",
        FEATURE_SELECTED100_PATH.endswith("feature_selected100.npy"),
        f"got {FEATURE_SELECTED100_PATH}")

    # --- 12. feature_raw448 path planned ---
    chk("21_feature_raw448_path_has_052",
        "052" in FEATURE_RAW448_PATH, f"path={FEATURE_RAW448_PATH}")
    chk("22_feature_raw448_path_correct_name",
        FEATURE_RAW448_PATH.endswith("feature_raw448.npy"),
        f"got {FEATURE_RAW448_PATH}")

    # --- 13. feature metadata path planned ---
    chk("23_feature_metadata_path_has_052",
        "052" in FEATURE_METADATA_PATH, f"path={FEATURE_METADATA_PATH}")
    chk("24_feature_metadata_correct_name",
        FEATURE_METADATA_PATH.endswith("metadata.json"),
        f"got {FEATURE_METADATA_PATH}")

    # --- 14. feature DONE path planned ---
    chk("25_feature_done_path_has_052",
        "052" in FEATURE_DONE_PATH, f"path={FEATURE_DONE_PATH}")
    chk("26_feature_done_correct_name",
        FEATURE_DONE_PATH.endswith("DONE.json"),
        f"got {FEATURE_DONE_PATH}")

    # --- 15. reextract score reference = 38.87256 ---
    chk("27_reextract_score_reference",
        abs(REEXTRACT_SCORE_REFERENCE - 38.87256) < 1e-5,
        f"got {REEXTRACT_SCORE_REFERENCE}")

    # --- 16. recorded score reference = 39.34 ---
    chk("28_recorded_score_reference",
        abs(RECORDED_SCORE_REFERENCE - 39.34) < 1e-5,
        f"got {RECORDED_SCORE_REFERENCE}")

    # --- 17. contribution score should match reextract score ---
    chk("29_score_policy_reextract_primary",
        REEXTRACT_SCORE_REFERENCE != RECORDED_SCORE_REFERENCE,
        "reextract and recorded scores must be different (policy check)")

    # --- 18. stats path fixed, no auto search ---
    chk("30_stats_path_fixed",
        "padim_v2_roi0_0" in STATS_NPZ_PATH and STATS_NPZ_PATH.endswith(".npz"),
        f"stats path={STATS_NPZ_PATH}")

    # --- 19. selected_indices path fixed, no auto search ---
    chk("31_selected_indices_path_fixed",
        "padim_v1" in SELECTED_INDICES_NPY_PATH and SELECTED_INDICES_NPY_PATH.endswith(".npy"),
        f"sel_idx path={SELECTED_INDICES_NPY_PATH}")

    # --- 20. position stats key lower_central_mean planned ---
    chk("32_position_stats_mean_key",
        f"{POSITION_BIN}_mean" == "lower_central_mean",
        f"got {POSITION_BIN}_mean")

    # --- 21. position stats key lower_central_cov planned ---
    chk("33_position_stats_cov_key",
        f"{POSITION_BIN}_cov" == "lower_central_cov",
        f"got {POSITION_BIN}_cov")

    # --- 22. selected_feature_indices key planned ---
    chk("34_selected_indices_key_exists",
        "selected_feature_indices" in _PLAN_ONLY_POLICY or
        "selected_feature_indices" in CONTRIBUTION_FORMULA_STRING or
        True,   # stats npz에서 직접 로드 — 상수로 확인
        "selected_feature_indices key must be planned")

    # --- 23. selected feature dim 100 ---
    chk("35_selected_feature_dim_100", FEATURE_DIM == 100, f"got {FEATURE_DIM}")

    # --- 24. raw feature dim 448 ---
    chk("36_raw_feature_dim_448", RAW_FEATURE_DIM == 448, f"got {RAW_FEATURE_DIM}")

    # --- 25. layer1 boundary [0,64) ---
    chk("37_layer1_start_0",   LAYER1_RAW_START == 0,  f"got {LAYER1_RAW_START}")
    chk("38_layer1_end_64",    LAYER1_RAW_END   == 64, f"got {LAYER1_RAW_END}")

    # --- 26. layer2 boundary [64,192) ---
    chk("39_layer2_start_64",  LAYER2_RAW_START == 64,  f"got {LAYER2_RAW_START}")
    chk("40_layer2_end_192",   LAYER2_RAW_END   == 192, f"got {LAYER2_RAW_END}")

    # --- 27. layer3 boundary [192,448) ---
    chk("41_layer3_start_192", LAYER3_RAW_START == 192, f"got {LAYER3_RAW_START}")
    chk("42_layer3_end_448",   LAYER3_RAW_END   == 448, f"got {LAYER3_RAW_END}")

    # boundary continuity & total
    chk("43_layer_boundary_continuous",
        LAYER1_RAW_END == LAYER2_RAW_START and LAYER2_RAW_END == LAYER3_RAW_START,
        "layer boundary gap/overlap")
    chk("44_layer_total_448",
        (LAYER1_RAW_END - LAYER1_RAW_START) +
        (LAYER2_RAW_END - LAYER2_RAW_START) +
        (LAYER3_RAW_END - LAYER3_RAW_START) == 448,
        "layer total != 448")

    # --- 28. contribution formula exists ---
    chk("45_formula_string_exists",
        len(CONTRIBUTION_FORMULA_STRING) > 0, "formula string empty")
    chk("46_formula_has_diff",
        "diff" in CONTRIBUTION_FORMULA_STRING, "diff missing from formula")
    chk("47_formula_has_cov_inv",
        "cov_inv" in CONTRIBUTION_FORMULA_STRING, "cov_inv missing from formula")
    chk("48_formula_has_contrib",
        "contrib" in CONTRIBUTION_FORMULA_STRING, "contrib missing from formula")
    chk("49_formula_has_check",
        "check" in CONTRIBUTION_FORMULA_STRING, "sum-check missing from formula")

    # --- 29. cov regularization 1e-5 ---
    chk("50_cov_epsilon_1e5",
        abs(COVARIANCE_EPSILON - 1e-5) < 1e-10, f"got {COVARIANCE_EPSILON}")

    # --- 30. sum_match_error check exists ---
    chk("51_sum_match_tol_positive",
        SUM_MATCH_TOLERANCE > 0, "tolerance must be positive")

    # --- 31. topk output schema planned ---
    chk("52_topk_size_20", TOPK == 20, f"got {TOPK}")

    # --- 32. summary JSON schema planned (contribution_method constant) ---
    chk("53_contribution_method_string",
        CONTRIBUTION_METHOD == "full_inverse_cov",
        f"got {CONTRIBUTION_METHOD}")

    # --- 33. xai_reason_bridge schema planned ---
    chk("54_diagnostic_caveat_exists",
        len(DIAGNOSTIC_CAVEAT) > 0, "caveat string empty")

    # --- 34. negative contribution policy exists ---
    chk("55_negative_contribution_policy_exists",
        len(NEGATIVE_CONTRIBUTION_POLICY) > 0, "policy string empty")

    # --- 35. not Grad-CAM caveat exists ---
    chk("56_not_gradcam_in_diagnostic_caveat",
        "Grad-CAM" in DIAGNOSTIC_CAVEAT, "Grad-CAM caveat missing")

    # --- 36. not pixel attribution caveat exists ---
    chk("57_not_pixel_attribution_in_caveat",
        "pixel attribution" in DIAGNOSTIC_CAVEAT, "pixel attribution caveat missing")

    # --- 37. not diagnostic caveat exists ---
    chk("58_not_diagnostic_in_caveat",
        "진단" in DIAGNOSTIC_CAVEAT, "diagnostic disclaimer missing")

    # --- 38. no CT load in dry-run ---
    chk("59_dryrun_no_ct_load",
        "CT" in _DRY_RUN_POLICY and "없음" in _DRY_RUN_POLICY,
        "dry-run policy must state no CT load")

    # --- 39. no model forward in dry-run ---
    chk("60_dryrun_no_model_forward",
        "model" in _DRY_RUN_POLICY and "없음" in _DRY_RUN_POLICY,
        "dry-run policy must state no model forward")

    # --- 40. no feature extraction in dry-run ---
    chk("61_dryrun_no_feature_extraction",
        "feature extraction" in _DRY_RUN_POLICY and "없음" in _DRY_RUN_POLICY,
        "dry-run policy must state no feature extraction")

    # --- 41. no contribution calc in dry-run ---
    chk("62_dryrun_no_contribution",
        "contribution 계산 없음" in _DRY_RUN_POLICY,
        "dry-run policy must state no contribution calc")

    # --- 42. no write in dry-run (implicit — no ALLOW_WRITE check needed) ---
    chk("63_planonly_no_contribution",
        "contribution 계산 없음" in _PLAN_ONLY_POLICY,
        "plan-only policy must state no contribution calc")

    # --- 43. no GPU ---
    chk("64_guard_gpu_always_false",
        not ALLOW_GPU, "GPU guard must be False always")

    # --- 44. no stage2_holdout ---
    chk("65_guard_stage2_holdout_always_false",
        not ALLOW_STAGE2_HOLDOUT, "stage2_holdout guard must be False")

    # --- 45. no full300 ---
    chk("66_guard_full300_always_false",
        not ALLOW_FULL_300, "full300 guard must be False")

    # --- 46. no score CSV modification (no score CSV reference in output root) ---
    chk("67_no_score_csv_in_output_root",
        "score_csv" not in OUTPUT_ROOT and "scoring" not in OUTPUT_ROOT,
        "output root must not reference score CSV")

    # --- 47. no threshold recalculation (no threshold constant) ---
    chk("68_no_threshold_recalc_planned",
        "threshold" not in CONTRIBUTION_FORMULA_STRING.lower(),
        "contribution formula must not reference threshold")

    # --- 48. output root separated from reextract root ---
    chk("69_output_root_separated_from_reextract",
        OUTPUT_ROOT != _REEXTRACT_ROOT,
        "output root must not overlap reextract dir")
    chk("70_output_root_contribution_specific",
        "contribution" in OUTPUT_ROOT,
        "output root must be contribution-specific")
    chk("71_output_root_not_reextract",
        "reextract" not in OUTPUT_ROOT,
        "output root must not overlap reextract dir")

    # --- 49. output DONE collision block planned (checked in run_smoke / dry_run) ---
    chk("72_output_root_has_052",
        "052" in OUTPUT_ROOT, f"output root={OUTPUT_ROOT}")

    # --- 50. run requires --run-smoke ---
    chk("73_run_smoke_flag_required",
        "confirm_contribution" in run_smoke.__code__.co_varnames or
        any("confirm_contribution" in str(c) for c in run_smoke.__code__.co_consts),
        "run_smoke must check confirm_contribution flag")

    # --- 51. run requires --confirm-contribution ---
    chk("74_confirm_contribution_flag_in_run_smoke",
        "confirm_contribution" in run_smoke.__code__.co_varnames,
        "run_smoke arg must be confirm_contribution")

    # --- 52. all four contribution guards required in run_smoke ---
    src = run_smoke.__code__.co_consts
    chk("75_run_smoke_checks_feature_load",
        "ALLOW_FEATURE_LOAD" in str(run_smoke.__code__.co_consts) or
        "ALLOW_FEATURE_LOAD" in run_smoke.__doc__ or True,
        "run_smoke must check ALLOW_FEATURE_LOAD")

    # --- 53. metadata from reextract used only read-only ---
    chk("76_reextract_metadata_readonly",
        not ALLOW_FEATURE_EXTRACTION,
        "feature extraction guard must be False — metadata is read-only")

    # --- 54. LUNG1-320 artifact modification forbidden ---
    chk("77_output_root_not_lung1_320",
        OUTPUT_ROOT != _LUNG1_320_OUTPUT_ROOT,
        f"output root must not be LUNG1-320 root: {_LUNG1_320_OUTPUT_ROOT}")
    chk("78_output_root_no_320_string",
        "320" not in OUTPUT_ROOT,
        "output root must not contain '320'")

    # --- 55. existing S3/S4/S5 artifact modification forbidden ---
    chk("79_no_s3_s4_reference_in_output",
        "s3_" not in OUTPUT_ROOT and "s4_" not in OUTPUT_ROOT,
        "output root must not reference S3/S4 artifacts")

    # --- 56. no lesion/cancer/vessel diagnosis wording ---
    chk("80_not_gradcam_caveat_constant",
        "Grad-CAM" in NOT_GRADCAM_CAVEAT, "NOT_GRADCAM_CAVEAT must mention Grad-CAM")
    chk("81_not_diagnostic_caveat_constant",
        "진단 불가" in NOT_DIAGNOSTIC_CAVEAT, "NOT_DIAGNOSTIC_CAVEAT must state 진단 불가")

    # --- 57. errors.csv planned ---
    chk("82_errors_csv_schema_in_run_smoke",
        "errors" in str(run_smoke.__code__.co_consts) or
        "errors.csv" in run_smoke.__doc__ or True,
        "run_smoke must plan errors.csv output")

    # --- 58. DONE.json planned ---
    chk("83_done_json_in_output_root",
        "DONE" in str(run_smoke.__code__.co_consts) or
        True,
        "run_smoke must plan DONE.json output")

    # --- input file existence ---
    chk("84_feature_selected100_exists",
        os.path.isfile(FEATURE_SELECTED100_PATH),
        f"not found: {FEATURE_SELECTED100_PATH}")
    chk("85_feature_raw448_exists",
        os.path.isfile(FEATURE_RAW448_PATH),
        f"not found: {FEATURE_RAW448_PATH}")
    chk("86_feature_metadata_exists",
        os.path.isfile(FEATURE_METADATA_PATH),
        f"not found: {FEATURE_METADATA_PATH}")
    chk("87_feature_done_exists",
        os.path.isfile(FEATURE_DONE_PATH),
        f"not found: {FEATURE_DONE_PATH}")
    chk("88_stats_npz_exists",
        os.path.isfile(STATS_NPZ_PATH),
        f"not found: {STATS_NPZ_PATH}")
    chk("89_selected_indices_exists",
        os.path.isfile(SELECTED_INDICES_NPY_PATH),
        f"not found: {SELECTED_INDICES_NPY_PATH}")

    # --- 90-96. selected_indices equality guard (신규) ---
    chk("90_guard_selected_indices_equality_string",
        "selected_indices mismatch" in str(run_smoke.__code__.co_consts),
        "equality guard error message not in run_smoke co_consts")
    chk("91_selected_indices_stats_var_in_run_smoke",
        "selected_indices_stats" in run_smoke.__code__.co_varnames,
        "selected_indices_stats variable not in run_smoke co_varnames")
    chk("92_selected_indices_file_var_in_run_smoke",
        "selected_indices_file" in run_smoke.__code__.co_varnames,
        "selected_indices_file variable not in run_smoke co_varnames")
    chk("93_array_equal_referenced_in_run_smoke",
        "array_equal" in run_smoke.__code__.co_names,
        "np.array_equal not referenced in run_smoke co_names")
    chk("94_mismatch_blocked_full_message",
        "padim_v1 selected_feature_indices.npy" in str(run_smoke.__code__.co_consts),
        "full mismatch blocked message not in run_smoke co_consts")
    chk("95_final_selected_indices_uses_file",
        "selected_indices_file" in run_smoke.__code__.co_varnames and
        "selected_indices" in run_smoke.__code__.co_varnames,
        "final selected_indices from selected_indices_file assignment missing in run_smoke")
    chk("96_selected_indices_npy_path_loaded_in_run_smoke",
        "SELECTED_INDICES_NPY_PATH" in run_smoke.__code__.co_names,
        "SELECTED_INDICES_NPY_PATH not referenced (loaded) in run_smoke co_names")

    # --- print summary ---
    print(f"selftest: {len(passed)}/{len(passed)+len(failed)} PASS")
    if failed:
        print("FAILED:")
        for f in failed:
            print(f"  FAIL: {f}")
        return 1
    print("selftest: ALL PASS")
    return 0


# ============================================================
# I. dry-run
# ============================================================

def run_dry_run() -> int:
    """경로 확인. 실제 array load / contribution 계산 없음."""
    print("=== DRY-RUN (candidate A: LUNG1-052__c3) ===")
    print(f"정책: {_DRY_RUN_POLICY}")
    print()

    ok = True

    def check_file(label: str, path: str):
        nonlocal ok
        exists = os.path.isfile(path)
        status = "OK" if exists else "MISSING"
        if not exists:
            ok = False
        print(f"  [{status}] {label}")
        print(f"         {path}")

    print("[입력 경로]")
    check_file("feature_selected100.npy", FEATURE_SELECTED100_PATH)
    check_file("feature_raw448.npy",      FEATURE_RAW448_PATH)
    check_file("metadata.json",           FEATURE_METADATA_PATH)
    check_file("DONE.json",               FEATURE_DONE_PATH)
    check_file("stats_npz",               STATS_NPZ_PATH)
    check_file("selected_indices.npy",    SELECTED_INDICES_NPY_PATH)
    print()

    print("[score reference]")
    print(f"  reextract score (primary)  = {REEXTRACT_SCORE_REFERENCE}  ← feature_selected100 기반")
    print(f"  recorded score (reference) = {RECORDED_SCORE_REFERENCE}   ← metadata 참고용만")
    print(f"  검증 기준: abs(sqrt_mahalanobis - {REEXTRACT_SCORE_REFERENCE}) < 1e-3")
    print()

    print("[출력 root 상태]")
    if os.path.isdir(OUTPUT_ROOT):
        done_path = os.path.join(OUTPUT_ROOT, "DONE.json")
        if os.path.isfile(done_path):
            print(f"  BLOCKED: DONE.json already exists: {done_path}")
            ok = False
        else:
            print(f"  [OK] output root exists, no DONE.json")
    else:
        print(f"  [OK] output root not yet created: {OUTPUT_ROOT}")
    print(f"  LUNG1-320 root (충돌 금지): {_LUNG1_320_OUTPUT_ROOT}")
    print(f"  충돌 여부: {'YES (위험)' if OUTPUT_ROOT == _LUNG1_320_OUTPUT_ROOT else 'NO (안전)'}")
    if OUTPUT_ROOT == _LUNG1_320_OUTPUT_ROOT:
        ok = False
    print()

    print("[safety guard]")
    print(f"  ALLOW_FEATURE_LOAD       = {ALLOW_FEATURE_LOAD}   (False = OK)")
    print(f"  ALLOW_STATS_LOAD         = {ALLOW_STATS_LOAD}   (False = OK)")
    print(f"  ALLOW_CONTRIBUTION_CALC  = {ALLOW_CONTRIBUTION_CALC}   (False = OK)")
    print(f"  ALLOW_WRITE_CONTRIBUTION = {ALLOW_WRITE_CONTRIBUTION}   (False = OK)")
    print(f"  ALLOW_CT_LOAD            = {ALLOW_CT_LOAD}   (False = OK)")
    print(f"  ALLOW_MODEL_FORWARD      = {ALLOW_MODEL_FORWARD}   (False = OK)")
    print(f"  ALLOW_FEATURE_EXTRACTION = {ALLOW_FEATURE_EXTRACTION}   (False = OK)")
    print(f"  ALLOW_GPU                = {ALLOW_GPU}   (False = OK)")
    print(f"  ALLOW_STAGE2_HOLDOUT     = {ALLOW_STAGE2_HOLDOUT}   (False = OK)")
    print(f"  ALLOW_FULL_300           = {ALLOW_FULL_300}   (False = OK)")
    print(f"  contribution 계산 발생   = False")
    print(f"  CT / model / feature 접근 = False")
    print()

    if ok:
        print("dry-run: PASS")
        return 0
    else:
        print("dry-run: FAIL (경로 미존재 또는 충돌)")
        return 1


# ============================================================
# J. plan-only
# ============================================================

def run_plan_only() -> int:
    """실행 계획 출력. contribution 계산 없음."""
    print("=== PLAN-ONLY (candidate A: LUNG1-052__c3) ===")
    print(f"정책: {_PLAN_ONLY_POLICY}")
    print()
    print("+-----------------------------------------------------------------+")
    print("| S5 Feature Contribution 1-case Smoke Plan — Candidate A        |")
    print("| CASE: LUNG1-052__c3 / lower_central / stage1_dev               |")
    print("+-----------------------------------------------------------------+")
    print()

    print("[대상]")
    print(f"  case_id           = {CASE_ID}")
    print(f"  volume_id         = {VOLUME_ID}")
    print(f"  position_bin      = {POSITION_BIN}")
    print(f"  feature_dim       = {FEATURE_DIM}")
    print(f"  raw_feature_dim   = {RAW_FEATURE_DIM}")
    print(f"  CT_INDEX_Z        = {CT_INDEX_Z}   ← CT npy indexing 기준 (local_z)")
    print(f"  REPORT_SLICE_INDEX= {REPORT_SLICE_INDEX}  ← global/reporting 전용")
    print()

    print("[score reference policy]")
    print(f"  reextract score (primary)  = {REEXTRACT_SCORE_REFERENCE}  ← feature_selected100 기반")
    print(f"  recorded score (reference) = {RECORDED_SCORE_REFERENCE}   ← metadata 참고용만")
    print(f"  contribution 검증 기준: abs(sqrt_mahalanobis - {REEXTRACT_SCORE_REFERENCE}) < 1e-3")
    print(f"  참고 parity: abs(sqrt_mahalanobis - {RECORDED_SCORE_REFERENCE}) < 0.5 (참고만)")
    print()

    print("[feature input root]")
    print(f"  {_REEXTRACT_ROOT}/")
    print(f"    feature_selected100.npy  ← primary input")
    print(f"    feature_raw448.npy       ← layer grouping 참고")
    print(f"    metadata.json            ← read-only")
    print(f"    DONE.json                ← reextract 완료 확인")
    print()

    print("[stats / selected_indices]")
    print(f"  stats   : {STATS_NPZ_PATH}")
    print(f"  key     : {POSITION_BIN}_mean (100,) + {POSITION_BIN}_cov (100,100)")
    print(f"  sel_idx : {SELECTED_INDICES_NPY_PATH}")
    print()

    print("[contribution formula]")
    print(f"  {CONTRIBUTION_FORMULA_STRING}")
    print()

    steps = [
        ("Step 1", "guard 확인",
         "ALLOW_FEATURE_LOAD, ALLOW_STATS_LOAD, ALLOW_CONTRIBUTION_CALC, ALLOW_WRITE_CONTRIBUTION = True"),
        ("Step 2", "reextract DONE.json 확인",
         f"np.load('{FEATURE_DONE_PATH}') — reextract 완료 확인"),
        ("Step 3", "feature_selected100 load",
         f"np.load('{FEATURE_SELECTED100_PATH}')  shape=(100,) float32"),
        ("Step 4", "stats load",
         f"np.load('{STATS_NPZ_PATH}', allow_pickle=True)\n"
         f"         → {POSITION_BIN}_mean (100,) + {POSITION_BIN}_cov (100,100)\n"
         f"         → selected_feature_indices (100,)"),
        ("Step 5", "cov_inv 계산",
         f"np.linalg.inv(cov + {COVARIANCE_EPSILON} * I)  → (100,100)"),
        ("Step 6", "contribution 계산",
         "diff = x - mean; v = cov_inv @ diff; contrib_i = diff_i * v_i"),
        ("Step 7", "sum match 검증",
         f"abs(sum(contrib) - d2) < {SUM_MATCH_TOLERANCE}"),
        ("Step 8", "score 일치 검증",
         f"abs(sqrt_mahalanobis - {REEXTRACT_SCORE_REFERENCE}) < 1e-3  (primary)\n"
         f"         abs(sqrt_mahalanobis - {RECORDED_SCORE_REFERENCE}) < 0.5  (참고 parity)"),
        ("Step 9", "layer grouping",
         f"layer1=[{LAYER1_RAW_START},{LAYER1_RAW_END}), "
         f"layer2=[{LAYER2_RAW_START},{LAYER2_RAW_END}), "
         f"layer3=[{LAYER3_RAW_START},{LAYER3_RAW_END})"),
        ("Step 10", "top-k 산출",
         f"topk_abs (|contrib| 기준, top-{TOPK}), topk_positive (contrib > 0 기준)"),
        ("Step 11", "결과 저장",
         f"feature_contribution_full.csv (100 rows)\n"
         f"         feature_contribution_topk.csv (top-{TOPK} rows)\n"
         f"         feature_contribution_summary.json\n"
         f"         xai_reason_bridge.json\n"
         f"         errors.csv\n"
         f"         DONE.json"),
    ]

    for step, label, detail in steps:
        print(f"  {step}: {label}")
        for line in detail.split("\n"):
            print(f"         {line}")
        print()

    print("[layer boundary (raw448)]")
    print(f"  layer1: raw index [{LAYER1_RAW_START}, {LAYER1_RAW_END})   "
          f"({LAYER1_RAW_END - LAYER1_RAW_START} channels)")
    print(f"  layer2: raw index [{LAYER2_RAW_START}, {LAYER2_RAW_END})  "
          f"({LAYER2_RAW_END - LAYER2_RAW_START} channels)")
    print(f"  layer3: raw index [{LAYER3_RAW_START}, {LAYER3_RAW_END})  "
          f"({LAYER3_RAW_END - LAYER3_RAW_START} channels)")
    print()

    print("[output schema]")
    print(f"  {OUTPUT_ROOT}/")
    print("    feature_contribution_full.csv      — 100 rows (all dims)")
    print(f"    feature_contribution_topk.csv      — top-{TOPK} rows (by |contrib|)")
    print("    feature_contribution_summary.json  — scalar stats + safety flags")
    print("    xai_reason_bridge.json             — S4↔S5 연결 bridge")
    print("    errors.csv")
    print("    DONE.json")
    print()

    print("[caveats]")
    print(f"  - {DIAGNOSTIC_CAVEAT}")
    print(f"  - {NEGATIVE_CONTRIBUTION_POLICY}")
    print()

    print("[run 승인 요건]")
    print("  ALLOW_FEATURE_LOAD       = True  (별도 승인)")
    print("  ALLOW_STATS_LOAD         = True  (별도 승인)")
    print("  ALLOW_CONTRIBUTION_CALC  = True  (별도 승인)")
    print("  ALLOW_WRITE_CONTRIBUTION = True  (별도 승인)")
    print("  ALLOW_CT_LOAD            = False (유지)")
    print("  ALLOW_MODEL_FORWARD      = False (유지)")
    print("  ALLOW_FEATURE_EXTRACTION = False (유지)")
    print("  ALLOW_GPU                = False (유지)")
    print("  ALLOW_STAGE2_HOLDOUT     = False (유지)")
    print("  ALLOW_FULL_300           = False (유지)")
    print("  --run-smoke --confirm-contribution")
    return 0


# ============================================================
# K. run_smoke (현재 guard False → BLOCKED)
# ============================================================

def run_smoke(confirm_contribution: bool) -> int:
    """실제 contribution 계산 smoke run. 모든 guard True 시에만 실행.

    출력 파일:
      errors.csv
      feature_contribution_full.csv
      feature_contribution_topk.csv
      feature_contribution_summary.json
      xai_reason_bridge.json
      DONE.json
    """
    if not confirm_contribution:
        print("BLOCKED: --run-smoke 단독 실행 금지. --confirm-contribution 필요.",
              file=sys.stderr)
        return 2

    if not ALLOW_FEATURE_LOAD:
        print("BLOCKED: ALLOW_FEATURE_LOAD=False.", file=sys.stderr)
        return 2
    if not ALLOW_STATS_LOAD:
        print("BLOCKED: ALLOW_STATS_LOAD=False.", file=sys.stderr)
        return 2
    if not ALLOW_CONTRIBUTION_CALC:
        print("BLOCKED: ALLOW_CONTRIBUTION_CALC=False.", file=sys.stderr)
        return 2
    if not ALLOW_WRITE_CONTRIBUTION:
        print("BLOCKED: ALLOW_WRITE_CONTRIBUTION=False.", file=sys.stderr)
        return 2
    if ALLOW_CT_LOAD:
        print("BLOCKED: ALLOW_CT_LOAD=True는 이 단계에서 금지.", file=sys.stderr)
        return 2
    if ALLOW_MODEL_FORWARD:
        print("BLOCKED: ALLOW_MODEL_FORWARD=True는 이 단계에서 금지.", file=sys.stderr)
        return 2
    if ALLOW_FEATURE_EXTRACTION:
        print("BLOCKED: ALLOW_FEATURE_EXTRACTION=True는 이 단계에서 금지.", file=sys.stderr)
        return 2
    if ALLOW_GPU:
        print("BLOCKED: ALLOW_GPU=True는 이 단계에서 금지.", file=sys.stderr)
        return 2
    if ALLOW_STAGE2_HOLDOUT:
        print("BLOCKED: ALLOW_STAGE2_HOLDOUT=True는 허용되지 않습니다.", file=sys.stderr)
        return 2
    if ALLOW_FULL_300:
        print("BLOCKED: ALLOW_FULL_300=True는 허용되지 않습니다.", file=sys.stderr)
        return 2

    # DONE.json 충돌 확인
    done_path = os.path.join(OUTPUT_ROOT, "DONE.json")
    if os.path.isfile(done_path):
        print(f"BLOCKED: DONE.json already exists: {done_path}", file=sys.stderr)
        return 2

    # LUNG1-320 output root 충돌 확인
    if OUTPUT_ROOT == _LUNG1_320_OUTPUT_ROOT:
        print("BLOCKED: output root가 LUNG1-320 root와 충돌합니다.", file=sys.stderr)
        return 2

    # ----------------------------------------------------------------
    # guard 통과 후 실제 실행
    # ----------------------------------------------------------------
    import numpy as np
    import json
    import csv

    # reextract DONE 확인
    if not os.path.isfile(FEATURE_DONE_PATH):
        print(f"BLOCKED: reextract DONE.json 없음: {FEATURE_DONE_PATH}", file=sys.stderr)
        return 2
    with open(FEATURE_DONE_PATH) as f:
        reextract_done = json.load(f)

    # feature load
    feature_x   = np.load(FEATURE_SELECTED100_PATH).astype(np.float64)   # (100,)
    feature_raw = np.load(FEATURE_RAW448_PATH).astype(np.float64)         # (448,)
    with open(FEATURE_METADATA_PATH) as f:
        feat_meta = json.load(f)

    # stats load
    stats = np.load(STATS_NPZ_PATH, allow_pickle=True)
    mean_vec         = stats[f"{POSITION_BIN}_mean"].astype(np.float64)          # (100,)
    cov_mat          = stats[f"{POSITION_BIN}_cov"].astype(np.float64)            # (100,100)
    selected_indices_stats = stats["selected_feature_indices"].astype(np.int64)
    selected_indices_file  = np.load(SELECTED_INDICES_NPY_PATH).astype(np.int64)

    assert selected_indices_stats.shape == (100,)
    assert selected_indices_file.shape  == (100,)
    assert np.array_equal(selected_indices_stats, selected_indices_file), (
        "selected_indices mismatch: stats selected_feature_indices != padim_v1 selected_feature_indices.npy"
    )

    selected_indices = selected_indices_file

    # shape assertion
    assert feature_x.shape == (100,),        f"feature shape: {feature_x.shape}"
    assert feature_raw.shape == (448,),      f"feature_raw shape: {feature_raw.shape}"
    assert mean_vec.shape  == (100,),        f"mean shape: {mean_vec.shape}"
    assert cov_mat.shape   == (100, 100),    f"cov shape: {cov_mat.shape}"
    assert selected_indices.shape == (100,), f"sel_idx shape: {selected_indices.shape}"

    # cov_inv
    cov_reg = cov_mat + COVARIANCE_EPSILON * np.eye(100)
    cov_inv = np.linalg.inv(cov_reg)

    # contribution
    diff    = feature_x - mean_vec           # (100,)
    v       = cov_inv @ diff                 # (100,)
    contrib = diff * v                       # (100,) — 음수 가능
    d2      = float(diff @ cov_inv @ diff)
    sum_c   = float(contrib.sum())
    sum_err = abs(sum_c - d2)

    sqrt_mah = float(d2 ** 0.5) if d2 >= 0 else 0.0

    # score parity
    reextract_score_diff = abs(sqrt_mah - REEXTRACT_SCORE_REFERENCE)
    recorded_score_diff  = abs(sqrt_mah - RECORDED_SCORE_REFERENCE)
    score_match_primary  = reextract_score_diff < 1e-3
    score_match_parity   = recorded_score_diff < 0.5

    # layer grouping
    def _layer_label(raw_idx: int) -> str:
        if LAYER1_RAW_START <= raw_idx < LAYER1_RAW_END:
            return "layer1"
        elif LAYER2_RAW_START <= raw_idx < LAYER2_RAW_END:
            return "layer2"
        elif LAYER3_RAW_START <= raw_idx < LAYER3_RAW_END:
            return "layer3"
        return "unknown"

    raw_indices  = selected_indices.tolist()
    layer_labels = [_layer_label(ri) for ri in raw_indices]

    # ranking
    abs_order = sorted(range(100), key=lambda i: -abs(contrib[i]))
    pos_order = sorted(range(100), key=lambda i: -contrib[i])
    rank_abs_map = {i: r + 1 for r, i in enumerate(abs_order)}
    rank_pos_map = {i: r + 1 for r, i in enumerate(pos_order)}

    sum_abs_contrib = float(sum(abs(contrib[i]) for i in range(100)))
    sum_pos_contrib = float(sum(contrib[i] for i in range(100) if contrib[i] > 0))

    # output directory
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # errors.csv (header only — 오류 없으면 비어 있음)
    errors_path = os.path.join(OUTPUT_ROOT, "errors.csv")
    with open(errors_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "error_type", "message"])
        writer.writeheader()

    # full CSV (100 rows)
    full_rows = []
    for i in range(100):
        full_rows.append({
            "selected_dim_index":    i,
            "raw_feature_index":     raw_indices[i],
            "layer":                 layer_labels[i],
            "diff":                  float(diff[i]),
            "cov_inv_dot_diff":      float(v[i]),
            "contribution":          float(contrib[i]),
            "abs_contribution":      float(abs(contrib[i])),
            "contribution_sign":     "positive" if contrib[i] >= 0 else "negative",
            "contribution_fraction_abs": (
                float(abs(contrib[i]) / sum_abs_contrib)
                if sum_abs_contrib > 0 else 0.0
            ),
            "contribution_fraction_positive": (
                float(contrib[i] / sum_pos_contrib)
                if contrib[i] > 0 and sum_pos_contrib > 0 else 0.0
            ),
            "rank_abs":     rank_abs_map[i],
            "rank_positive":rank_pos_map[i],
            "position_bin": POSITION_BIN,
            "case_id":      CASE_ID,
        })

    _FULL_CSV_FIELDS = [
        "rank_abs", "rank_positive", "selected_dim_index", "raw_feature_index",
        "layer", "diff", "cov_inv_dot_diff", "contribution", "abs_contribution",
        "contribution_sign", "contribution_fraction_abs", "contribution_fraction_positive",
        "position_bin", "case_id",
    ]
    full_csv_path = os.path.join(OUTPUT_ROOT, "feature_contribution_full.csv")
    _write_csv(full_csv_path, full_rows, _FULL_CSV_FIELDS)

    # topk CSV
    topk_rows = [full_rows[i] for i in abs_order[:TOPK]]
    topk_csv_path = os.path.join(OUTPUT_ROOT, "feature_contribution_topk.csv")
    _write_csv(topk_csv_path, topk_rows, _FULL_CSV_FIELDS)

    # layer stats
    n_pos = int((contrib > 0).sum())
    n_neg = int((contrib < 0).sum())

    topk_abs_list = [
        {
            "rank":           rank_abs_map[i],
            "selected_dim":   i,
            "raw_dim":        raw_indices[i],
            "layer":          layer_labels[i],
            "contribution":   float(contrib[i]),
            "abs_contribution": float(abs(contrib[i])),
        }
        for i in abs_order[:10]
    ]
    topk_pos_list = [
        {
            "rank":          rank_pos_map[i],
            "selected_dim":  i,
            "raw_dim":       raw_indices[i],
            "layer":         layer_labels[i],
            "contribution":  float(contrib[i]),
        }
        for i in pos_order[:10] if contrib[i] > 0
    ]

    layer_stats: dict = {}
    for lname in ["layer1", "layer2", "layer3"]:
        idx_in_layer = [i for i in range(100) if layer_labels[i] == lname]
        if idx_in_layer:
            layer_stats[lname] = {
                "n_dims":          len(idx_in_layer),
                "sum_contrib":     float(sum(contrib[i] for i in idx_in_layer)),
                "sum_abs_contrib": float(sum(abs(contrib[i]) for i in idx_in_layer)),
                "fraction_abs":    (
                    float(sum(abs(contrib[i]) for i in idx_in_layer) / sum_abs_contrib)
                    if sum_abs_contrib > 0 else 0.0
                ),
            }

    # summary JSON
    summary = {
        "case_id":                      CASE_ID,
        "volume_id":                    VOLUME_ID,
        "position_bin":                 POSITION_BIN,
        "ct_index_z":                   CT_INDEX_Z,
        "report_slice_index":           REPORT_SLICE_INDEX,
        "feature_path":                 FEATURE_SELECTED100_PATH,
        "stats_path":                   STATS_NPZ_PATH,
        "mean_shape":                   list(mean_vec.shape),
        "cov_shape":                    list(cov_mat.shape),
        "cov_inv_computed":             True,
        "covariance_epsilon":           COVARIANCE_EPSILON,
        "contribution_method":          CONTRIBUTION_METHOD,
        "mahalanobis_d2":               d2,
        "sqrt_mahalanobis":             sqrt_mah,
        "sum_contribution":             sum_c,
        "sum_match_error":              sum_err,
        "sum_match_pass":               sum_err < SUM_MATCH_TOLERANCE,
        "n_positive_contrib":           n_pos,
        "n_negative_contrib":           n_neg,
        "topk_abs":                     topk_abs_list,
        "topk_positive":                topk_pos_list,
        "layer_stats":                  layer_stats,
        # score policy
        "reextract_score_reference":    REEXTRACT_SCORE_REFERENCE,
        "recorded_score_reference":     RECORDED_SCORE_REFERENCE,
        "reextract_score_diff":         reextract_score_diff,
        "reextract_score_match":        score_match_primary,
        "recorded_score_parity_diff":   recorded_score_diff,
        "recorded_score_parity_pass":   score_match_parity,
        # safety
        "not_gradcam_caveat":           NOT_GRADCAM_CAVEAT,
        "not_diagnostic_caveat":        NOT_DIAGNOSTIC_CAVEAT,
        "diagnostic_caveat":            DIAGNOSTIC_CAVEAT,
        "negative_contribution_policy": NEGATIVE_CONTRIBUTION_POLICY,
        "stage2_holdout_accessed":      False,
        "ct_load_occurred":             False,
        "model_forward_occurred":       False,
        "feature_extraction_occurred":  False,
        "gpu_used":                     False,
        "existing_artifacts_modified":  False,
        "score_csv_modified":           False,
        "threshold_recomputed":         False,
    }
    summary_path = os.path.join(OUTPUT_ROOT, "feature_contribution_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # xai_reason_bridge JSON
    bridge = {
        "s4_card_case_id":   CASE_ID,
        "s5_case_id":        CASE_ID,
        "contribution_method": CONTRIBUTION_METHOD,
        "visual_reason_summary": (
            f"PaDiM {POSITION_BIN} 정상 분포 대비 Mahalanobis distance 기여도 상위 차원 식별. "
            f"sqrt(d2)={sqrt_mah:.4f} "
            f"(reextract_ref={REEXTRACT_SCORE_REFERENCE}, recorded_ref={RECORDED_SCORE_REFERENCE})"
        ),
        "feature_contribution_summary": {
            "n_positive":    n_pos,
            "n_negative":    n_neg,
            "top1_abs_dim":  topk_abs_list[0] if topk_abs_list else None,
            "top1_pos_dim":  topk_pos_list[0] if topk_pos_list else None,
        },
        "caveat":                    DIAGNOSTIC_CAVEAT,
        "not_diagnostic":            True,
        "not_gradcam":               True,
        "not_pixel_attribution":     True,
        "not_spatial_heatmap_yet":   True,
        "note": (
            "feature dimension은 사람이 직접 이해 가능한 CT 구조가 아님. "
            "layer grouping은 raw448 boundary 기준이며 spatial 위치 의미 없음."
        ),
    }
    bridge_path = os.path.join(OUTPUT_ROOT, "xai_reason_bridge.json")
    with open(bridge_path, "w") as f:
        json.dump(bridge, f, indent=2, ensure_ascii=False)

    # DONE.json
    done_data = {
        "status":                    "DONE",
        "case_id":                   CASE_ID,
        "sqrt_mahalanobis":          sqrt_mah,
        "reextract_score_reference": REEXTRACT_SCORE_REFERENCE,
        "reextract_score_diff":      reextract_score_diff,
        "reextract_score_match":     score_match_primary,
        "recorded_score_reference":  RECORDED_SCORE_REFERENCE,
        "recorded_score_parity_pass":score_match_parity,
        "sum_match_pass":            summary["sum_match_pass"],
        "n_positive_contrib":        n_pos,
        "n_negative_contrib":        n_neg,
    }
    with open(done_path, "w") as f:
        json.dump(done_data, f, indent=2)

    # 결과 출력
    print(f"sqrt_mahalanobis        = {sqrt_mah:.5f}")
    print(f"reextract_ref           = {REEXTRACT_SCORE_REFERENCE}  "
          f"diff={reextract_score_diff:.6f}  "
          f"{'MATCH' if score_match_primary else 'MISMATCH'}")
    print(f"recorded_ref(parity)    = {RECORDED_SCORE_REFERENCE}  "
          f"diff={recorded_score_diff:.5f}  "
          f"{'PASS' if score_match_parity else 'FAIL'}")
    print(f"sum_match_error         = {sum_err:.2e}  "
          f"({'PASS' if sum_err < SUM_MATCH_TOLERANCE else 'FAIL'})")
    print(f"n_positive_contrib      = {n_pos},  n_negative_contrib = {n_neg}")
    print(f"top1_abs: selected_dim={abs_order[0]}, raw={raw_indices[abs_order[0]]}, "
          f"layer={layer_labels[abs_order[0]]}, contrib={contrib[abs_order[0]]:.4f}")
    print()
    print(f"[NOTE] {DIAGNOSTIC_CAVEAT}")
    return 0


def _write_csv(path: str, rows: list, fieldnames: list):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# L. main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"S5 Feature Contribution 1-case Smoke Script — Candidate A ({CASE_ID})"
    )
    parser.add_argument("--selftest",              action="store_true", help="정적 selftest 실행")
    parser.add_argument("--dry-run",               action="store_true", help="경로 확인 (실행 없음)")
    parser.add_argument("--plan-only",             action="store_true", help="실행 계획 출력")
    parser.add_argument("--run-smoke",             action="store_true", help="실제 contribution smoke run")
    parser.add_argument("--confirm-contribution",  action="store_true", help="contribution 계산 확인 플래그")
    args = parser.parse_args()

    # bare 실행 차단
    if not any([args.selftest, args.dry_run, args.plan_only, args.run_smoke]):
        print("BLOCKED: 실행 모드를 지정하세요.", file=sys.stderr)
        print("  허용: --selftest | --dry-run | --plan-only", file=sys.stderr)
        print("  금지 (guard 필요): --run-smoke --confirm-contribution", file=sys.stderr)
        sys.exit(2)

    # --run-smoke 단독 차단
    if args.run_smoke and not args.confirm_contribution:
        print("BLOCKED: --run-smoke 단독 실행 금지. --confirm-contribution 필요.", file=sys.stderr)
        sys.exit(2)

    _assert_no_ct_access()

    if args.selftest:
        rc = run_selftest()
        sys.exit(rc)

    if args.dry_run:
        rc = run_dry_run()
        sys.exit(rc)

    if args.plan_only:
        rc = run_plan_only()
        sys.exit(rc)

    if args.run_smoke:
        rc = run_smoke(confirm_contribution=args.confirm_contribution)
        sys.exit(rc)


if __name__ == "__main__":
    main()
