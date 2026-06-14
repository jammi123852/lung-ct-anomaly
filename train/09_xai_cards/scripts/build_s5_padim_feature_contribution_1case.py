"""
S5 PaDiM Feature Contribution 1-case Smoke Script
==================================================
대상: LUNG1-320__c2 / stage1_dev only
단계: contribution smoke script (정적 검사 + 실행 분리)

실행 방법:
  --selftest              : 정적 selftest 37+ 항목 (허용)
  --dry-run               : 경로 확인 (실제 array load / contribution 계산 없음)
  --plan-only             : 실행 계획 출력 (계산 없음)
  --run-smoke --confirm-contribution : 실제 contribution 계산 (guard True 시에만)

금지 (현재 모든 guard=False):
  bare 실행              → BLOCKED exit 2
  --run-smoke 단독       → BLOCKED exit 2
  --run-smoke --confirm-contribution → BLOCKED exit 2 (guard False)

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
  금지: "암/악성/진단/병변 확정/Grad-CAM/heatmap 단정"
"""

from __future__ import annotations

import argparse
import os
import sys

# ============================================================
# A. Path / Metadata constants
# ============================================================

REPO_ROOT = "/home/jinhy/project/lung-ct-anomaly"

CASE_ID        = "LUNG1-320__c2"
POSITION_BIN   = "lower_peripheral"
FEATURE_DIM    = 100
SPLIT          = "stage1_dev"

# ============================================================
# B. Input paths
# ============================================================

_REEXTRACT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_feature_reextract_1case_smoke_v1"
)

FEATURE_SELECTED100_PATH = os.path.join(
    _REEXTRACT_ROOT, "s5_feature_reextract_1case_feature_selected100.npy"
)
FEATURE_RAW448_PATH = os.path.join(
    _REEXTRACT_ROOT, "s5_feature_reextract_1case_feature_raw448.npy"
)
FEATURE_METADATA_PATH = os.path.join(
    _REEXTRACT_ROOT, "s5_feature_reextract_1case_metadata.json"
)

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
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_feature_reextract_1case_smoke_run_validation_v1.json"
)

# ============================================================
# C. Output paths (run 단계에서만 생성)
# ============================================================

OUTPUT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_feature_contribution_1case_smoke_v1"
)

# ============================================================
# D. contribution formula constants
# ============================================================

COVARIANCE_EPSILON     = 1e-5    # cov_inv 계산 시 regularization
CONTRIBUTION_METHOD    = "full_inverse_cov"  # full inverse-cov decomposition
SUM_MATCH_TOLERANCE    = 1e-3    # abs(sum_contrib - d2) < tolerance

# layer boundary (raw448 concat 구조)
LAYER1_RAW_START       = 0
LAYER1_RAW_END         = 64
LAYER2_RAW_START       = 64
LAYER2_RAW_END         = 192
LAYER3_RAW_START       = 192
LAYER3_RAW_END         = 448

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
    "Grad-CAM 아님. spatial heatmap 단정 불가."
)

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
# F. Local z guard (contribution 단계 — CT 접근 없음)
# ============================================================

def _assert_no_ct_access():
    """이 단계에서 CT / model / feature extraction 접근 없음을 확인."""
    assert not ALLOW_CT_LOAD,            "CT load guard must be False"
    assert not ALLOW_MODEL_FORWARD,      "model forward guard must be False"
    assert not ALLOW_FEATURE_EXTRACTION, "feature extraction guard must be False"
    assert not ALLOW_GPU,                "GPU guard must be False"

# ============================================================
# G. selftest (37+ 항목)
# ============================================================

def run_selftest() -> int:
    """정적 selftest. 모든 guard / 상수 / 경로 / 설계 검증. 실제 load/계산 없음."""
    passed = []
    failed = []

    def chk(name: str, cond: bool, msg: str = ""):
        if cond:
            passed.append(name)
        else:
            failed.append(f"{name}: {msg}")

    # 1. all guards default False
    chk("01_guard_feature_load_false",       not ALLOW_FEATURE_LOAD,       "ALLOW_FEATURE_LOAD must be False")
    chk("02_guard_stats_load_false",         not ALLOW_STATS_LOAD,         "ALLOW_STATS_LOAD must be False")
    chk("03_guard_contribution_calc_false",  not ALLOW_CONTRIBUTION_CALC,  "ALLOW_CONTRIBUTION_CALC must be False")
    chk("04_guard_write_contribution_false", not ALLOW_WRITE_CONTRIBUTION, "ALLOW_WRITE_CONTRIBUTION must be False")
    chk("05_guard_ct_load_false",            not ALLOW_CT_LOAD,            "ALLOW_CT_LOAD must be False")
    chk("06_guard_model_forward_false",      not ALLOW_MODEL_FORWARD,      "ALLOW_MODEL_FORWARD must be False")
    chk("07_guard_feature_extraction_false", not ALLOW_FEATURE_EXTRACTION, "ALLOW_FEATURE_EXTRACTION must be False")
    chk("08_guard_gpu_false",                not ALLOW_GPU,                "ALLOW_GPU must be False")
    chk("09_guard_stage2_holdout_false",     not ALLOW_STAGE2_HOLDOUT,     "ALLOW_STAGE2_HOLDOUT must be False")
    chk("10_guard_full300_false",            not ALLOW_FULL_300,           "ALLOW_FULL_300 must be False")

    # 2. case_id exact
    chk("11_case_id_exact",   CASE_ID == "LUNG1-320__c2",        f"got {CASE_ID}")

    # 3. input paths exist
    chk("12_feature_selected100_exists",
        os.path.isfile(FEATURE_SELECTED100_PATH),
        f"not found: {FEATURE_SELECTED100_PATH}")
    chk("13_feature_metadata_exists",
        os.path.isfile(FEATURE_METADATA_PATH),
        f"not found: {FEATURE_METADATA_PATH}")
    chk("14_stats_npz_exists",
        os.path.isfile(STATS_NPZ_PATH),
        f"not found: {STATS_NPZ_PATH}")
    chk("15_selected_indices_exists",
        os.path.isfile(SELECTED_INDICES_NPY_PATH),
        f"not found: {SELECTED_INDICES_NPY_PATH}")
    chk("16_reextract_validation_exists",
        os.path.isfile(REEXTRACT_VALIDATION_JSON_PATH),
        f"not found: {REEXTRACT_VALIDATION_JSON_PATH}")

    # 4. position_bin
    chk("17_position_bin_lower_peripheral",
        POSITION_BIN == "lower_peripheral", f"got {POSITION_BIN}")

    # 5. dims
    chk("18_feature_dim_100",    FEATURE_DIM == 100,           f"got {FEATURE_DIM}")
    chk("19_layer1_raw_end_64",  LAYER1_RAW_END == 64,         f"got {LAYER1_RAW_END}")
    chk("20_layer2_raw_end_192", LAYER2_RAW_END == 192,        f"got {LAYER2_RAW_END}")
    chk("21_layer3_raw_end_448", LAYER3_RAW_END == 448,        f"got {LAYER3_RAW_END}")
    chk("22_layer_boundary_coverage",
        LAYER1_RAW_END == LAYER2_RAW_START and
        LAYER2_RAW_END == LAYER3_RAW_START,
        "layer boundary gap/overlap")
    chk("23_layer_total_448",
        LAYER1_RAW_END + (LAYER2_RAW_END - LAYER2_RAW_START) +
        (LAYER3_RAW_END - LAYER3_RAW_START) == 448,
        "layer total != 448")

    # 6. contribution formula
    chk("24_formula_string_exists",
        len(CONTRIBUTION_FORMULA_STRING) > 0, "formula string empty")
    chk("25_formula_has_diff",       "diff" in CONTRIBUTION_FORMULA_STRING,
        "diff missing from formula")
    chk("26_formula_has_cov_inv",    "cov_inv" in CONTRIBUTION_FORMULA_STRING,
        "cov_inv missing from formula")
    chk("27_formula_has_contrib",    "contrib" in CONTRIBUTION_FORMULA_STRING,
        "contrib missing from formula")
    chk("28_formula_has_check",      "check" in CONTRIBUTION_FORMULA_STRING,
        "sum-check missing from formula")
    chk("29_full_inverse_cov_method",
        CONTRIBUTION_METHOD == "full_inverse_cov", f"got {CONTRIBUTION_METHOD}")

    # 7. sum match tolerance
    chk("30_sum_match_tol_positive", SUM_MATCH_TOLERANCE > 0,
        "tolerance must be positive")

    # 8. negative contribution policy
    chk("31_negative_contribution_policy",
        len(NEGATIVE_CONTRIBUTION_POLICY) > 0, "policy string empty")
    chk("32_negative_policy_no_diagnosis",
        "진단" not in NEGATIVE_CONTRIBUTION_POLICY.replace("진단 의미 부여 금지", ""),
        "negative policy contains diagnostic language")

    # 9. diagnostic caveat
    chk("33_diagnostic_caveat_exists",
        len(DIAGNOSTIC_CAVEAT) > 0, "caveat string empty")
    chk("34_not_gradcam_in_caveat",
        "Grad-CAM" in DIAGNOSTIC_CAVEAT, "Grad-CAM caveat missing")
    chk("35_not_spatial_heatmap_in_caveat",
        "spatial heatmap" in DIAGNOSTIC_CAVEAT, "spatial heatmap caveat missing")
    chk("36_not_diagnostic_in_caveat",
        "진단" in DIAGNOSTIC_CAVEAT, "diagnostic disclaimer missing")

    # 10. output root separation
    chk("37_output_root_separated",
        "s5_feature_contribution_1case_smoke_v1" in OUTPUT_ROOT,
        "output root should be contribution-specific")
    chk("38_output_root_not_reextract",
        "reextract" not in OUTPUT_ROOT, "output root must not overlap reextract dir")

    # 11. covariance epsilon
    chk("39_cov_epsilon_1e5",
        abs(COVARIANCE_EPSILON - 1e-5) < 1e-10, f"got {COVARIANCE_EPSILON}")

    # 12. run requires correct flags
    chk("40_run_requires_confirm_flag",
        "confirm_contribution" in run_smoke.__code__.co_varnames or
        "confirm_contribution" in str(run_smoke.__code__.co_consts),
        "run_smoke must check confirm_contribution flag")

    # 13. dry-run does not calculate contribution
    chk("41_dryrun_no_contribution",
        "contribution 계산 없음" in _DRY_RUN_POLICY,
        "dry-run policy must state no contribution calc")

    # 14. plan-only does not calculate contribution
    chk("42_planonly_no_contribution",
        "contribution 계산 없음" in _PLAN_ONLY_POLICY,
        "plan-only policy must state no contribution calc")

    # 15. feature raw448 path exists
    chk("43_feature_raw448_exists",
        os.path.isfile(FEATURE_RAW448_PATH),
        f"not found: {FEATURE_RAW448_PATH}")

    # print summary
    print(f"selftest: {len(passed)}/{len(passed)+len(failed)} PASS")
    if failed:
        print("FAILED:")
        for f in failed:
            print(f"  FAIL: {f}")
        return 1
    print("selftest: ALL PASS")
    return 0


# dry-run / plan-only policy strings (selftest에서 참조)
_DRY_RUN_POLICY  = "dry-run: path existence only. contribution 계산 없음. CT/model/feature extraction 없음."
_PLAN_ONLY_POLICY = "plan-only: formula/schema 출력. contribution 계산 없음. CT/model/feature extraction 없음."

# ============================================================
# H. dry-run
# ============================================================

def run_dry_run() -> int:
    """경로 확인. 실제 array load / contribution 계산 없음."""
    print(f"=== DRY-RUN ===")
    print(f"정책: {_DRY_RUN_POLICY}")
    print()

    ok = True

    def check_path(label: str, path: str):
        nonlocal ok
        exists = os.path.isfile(path)
        status = "OK" if exists else "MISSING"
        if not exists:
            ok = False
        print(f"  [{status}] {label}")
        print(f"         {path}")

    print("[입력 경로]")
    check_path("feature_selected100.npy", FEATURE_SELECTED100_PATH)
    check_path("feature_raw448.npy",      FEATURE_RAW448_PATH)
    check_path("feature_metadata.json",   FEATURE_METADATA_PATH)
    check_path("stats_npz",               STATS_NPZ_PATH)
    check_path("selected_indices.npy",    SELECTED_INDICES_NPY_PATH)
    check_path("reextract_validation.json", REEXTRACT_VALIDATION_JSON_PATH)
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
    print()

    print("[safety guard]")
    print(f"  ALLOW_CT_LOAD            = {ALLOW_CT_LOAD}   (False = OK)")
    print(f"  ALLOW_MODEL_FORWARD      = {ALLOW_MODEL_FORWARD}   (False = OK)")
    print(f"  ALLOW_FEATURE_EXTRACTION = {ALLOW_FEATURE_EXTRACTION}   (False = OK)")
    print(f"  ALLOW_GPU                = {ALLOW_GPU}   (False = OK)")
    print(f"  ALLOW_STAGE2_HOLDOUT     = {ALLOW_STAGE2_HOLDOUT}   (False = OK)")
    print(f"  ALLOW_CONTRIBUTION_CALC  = {ALLOW_CONTRIBUTION_CALC}   (False = OK)")
    print(f"  contribution 계산 발생   = False")
    print(f"  CT / model 접근          = False")
    print()

    if ok:
        print("dry-run: PASS")
        return 0
    else:
        print("dry-run: FAIL (경로 미존재 또는 DONE.json 충돌)")
        return 1


# ============================================================
# I. plan-only
# ============================================================

def run_plan_only() -> int:
    """실행 계획 출력. contribution 계산 없음."""
    print(f"=== PLAN-ONLY ===")
    print(f"정책: {_PLAN_ONLY_POLICY}")
    print()
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│ S5 Feature Contribution 1-case Smoke Plan                       │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()

    print(f"[대상]")
    print(f"  case_id      = {CASE_ID}")
    print(f"  position_bin = {POSITION_BIN}")
    print(f"  feature_dim  = {FEATURE_DIM}")
    print()

    print("[contribution formula]")
    print(f"  {CONTRIBUTION_FORMULA_STRING}")
    print()

    print("[단계별 계획]")
    steps = [
        ("Step 1", "guard 확인",
         "ALLOW_FEATURE_LOAD, ALLOW_STATS_LOAD, ALLOW_CONTRIBUTION_CALC, ALLOW_WRITE_CONTRIBUTION = True"),
        ("Step 2", "feature_selected100 load",
         f"np.load('{FEATURE_SELECTED100_PATH}')  shape=(100,) float32"),
        ("Step 3", "stats load",
         f"np.load('{STATS_NPZ_PATH}')  → lower_peripheral_mean (100,), lower_peripheral_cov (100,100)"),
        ("Step 4", "cov_inv 계산",
         f"np.linalg.inv(cov + {COVARIANCE_EPSILON} * I)  → (100,100)"),
        ("Step 5", "contribution 계산",
         "diff = x - mean; v = cov_inv @ diff; contrib_i = diff_i * v_i"),
        ("Step 6", "sum match 검증",
         f"abs(sum(contrib) - d2) < {SUM_MATCH_TOLERANCE}"),
        ("Step 7", "layer grouping",
         f"layer1=[0,{LAYER1_RAW_END}), layer2=[{LAYER2_RAW_START},{LAYER2_RAW_END}), layer3=[{LAYER3_RAW_START},{LAYER3_RAW_END})"),
        ("Step 8", "top-k 산출",
         "topk_abs (|contrib| 기준), topk_positive (contrib > 0 기준)"),
        ("Step 9", "결과 저장",
         f"feature_contribution_topk.csv, feature_contribution_full.csv, "
         f"feature_contribution_summary.json, xai_reason_bridge.json, DONE.json"),
    ]
    for step, label, detail in steps:
        print(f"  {step}: {label}")
        print(f"         {detail}")
        print()

    print("[caveats]")
    print(f"  - {DIAGNOSTIC_CAVEAT}")
    print(f"  - {NEGATIVE_CONTRIBUTION_POLICY}")
    print()

    print("[layer boundary (raw448)]")
    print(f"  layer1: raw index [{LAYER1_RAW_START}, {LAYER1_RAW_END})   (64 channels)")
    print(f"  layer2: raw index [{LAYER2_RAW_START}, {LAYER2_RAW_END})  (128 channels)")
    print(f"  layer3: raw index [{LAYER3_RAW_START}, {LAYER3_RAW_END})  (256 channels)")
    print()

    print("[output schema]")
    print("  feature_contribution_topk.csv      — top-k rows")
    print("  feature_contribution_full.csv       — 100 rows (all dims)")
    print("  feature_contribution_summary.json   — scalar stats + safety flags")
    print("  xai_reason_bridge.json              — S4↔S5 연결 bridge")
    print("  errors.csv")
    print("  DONE.json")
    print()

    print("[run 승인 요건]")
    print("  ALLOW_FEATURE_LOAD       = True")
    print("  ALLOW_STATS_LOAD         = True")
    print("  ALLOW_CONTRIBUTION_CALC  = True")
    print("  ALLOW_WRITE_CONTRIBUTION = True")
    print("  ALLOW_CT_LOAD            = False  (유지)")
    print("  ALLOW_MODEL_FORWARD      = False  (유지)")
    print("  ALLOW_FEATURE_EXTRACTION = False  (유지)")
    print("  ALLOW_GPU                = False  (유지)")
    print("  --run-smoke --confirm-contribution")
    return 0


# ============================================================
# J. run_smoke (현재 guard False → BLOCKED)
# ============================================================

def run_smoke(confirm_contribution: bool) -> int:
    """실제 contribution 계산 smoke run. guard True 시에만 실행."""
    if not confirm_contribution:
        print("BLOCKED: --run-smoke 단독 실행 금지. --confirm-contribution 필요.", file=sys.stderr)
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

    # ----------------------------------------------------------------
    # guard 통과 후 실제 실행 로직
    # ----------------------------------------------------------------
    import numpy as np
    import json
    import csv

    # --- load ---
    feature_x = np.load(FEATURE_SELECTED100_PATH).astype(np.float64)    # (100,)
    feature_raw = np.load(FEATURE_RAW448_PATH).astype(np.float64)       # (448,)
    with open(FEATURE_METADATA_PATH) as f:
        feat_meta = json.load(f)

    stats = np.load(STATS_NPZ_PATH, allow_pickle=True)
    mean_vec = stats[f"{POSITION_BIN}_mean"].astype(np.float64)          # (100,)
    cov_mat  = stats[f"{POSITION_BIN}_cov"].astype(np.float64)           # (100,100)
    selected_indices = stats["selected_feature_indices"].astype(np.int64) # (100,)

    assert feature_x.shape == (100,), f"feature shape: {feature_x.shape}"
    assert mean_vec.shape  == (100,), f"mean shape: {mean_vec.shape}"
    assert cov_mat.shape   == (100, 100), f"cov shape: {cov_mat.shape}"
    assert selected_indices.shape == (100,), f"sel_idx shape: {selected_indices.shape}"

    # --- cov_inv ---
    cov_reg  = cov_mat + COVARIANCE_EPSILON * np.eye(100)
    cov_inv  = np.linalg.inv(cov_reg)

    # --- contribution ---
    diff     = feature_x - mean_vec                     # (100,)
    v        = cov_inv @ diff                           # (100,)
    contrib  = diff * v                                 # (100,) — 음수 가능
    d2       = float(diff @ cov_inv @ diff)
    sum_c    = float(contrib.sum())
    sum_err  = abs(sum_c - d2)

    sqrt_mah = float(d2 ** 0.5) if d2 >= 0 else 0.0
    score_expected = feat_meta.get("expected_score", None)
    score_recomputed = sqrt_mah

    # --- layer grouping ---
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

    # --- abs / positive ordering ---
    abs_order  = sorted(range(100), key=lambda i: -abs(contrib[i]))
    pos_order  = sorted(range(100), key=lambda i: -contrib[i])

    rank_abs_map  = {i: r+1 for r, i in enumerate(abs_order)}
    rank_pos_map  = {i: r+1 for r, i in enumerate(pos_order)}

    sum_abs_contrib = sum(abs(contrib[i]) for i in range(100))
    sum_pos_contrib = sum(contrib[i] for i in range(100) if contrib[i] > 0)

    # --- output ---
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # full CSV (100 rows)
    full_rows = []
    for i in range(100):
        full_rows.append({
            "selected_dim_index":        i,
            "raw_feature_index":         raw_indices[i],
            "layer":                     layer_labels[i],
            "diff":                      float(diff[i]),
            "cov_inv_dot_diff":          float(v[i]),
            "contribution":              float(contrib[i]),
            "abs_contribution":          float(abs(contrib[i])),
            "contribution_sign":         "positive" if contrib[i] >= 0 else "negative",
            "contribution_fraction_abs": float(abs(contrib[i]) / sum_abs_contrib) if sum_abs_contrib > 0 else 0.0,
            "contribution_fraction_positive": (float(contrib[i] / sum_pos_contrib)
                                               if contrib[i] > 0 and sum_pos_contrib > 0 else 0.0),
            "rank_abs":                  rank_abs_map[i],
            "rank_positive":             rank_pos_map[i],
            "position_bin":              POSITION_BIN,
            "case_id":                   CASE_ID,
        })

    full_csv_path = os.path.join(OUTPUT_ROOT, "feature_contribution_full.csv")
    _write_csv(full_csv_path, full_rows,
               ["rank_abs","rank_positive","selected_dim_index","raw_feature_index",
                "layer","diff","cov_inv_dot_diff","contribution","abs_contribution",
                "contribution_sign","contribution_fraction_abs","contribution_fraction_positive",
                "position_bin","case_id"])

    # topk CSV (top-20 by abs)
    TOPK = 20
    topk_rows = [full_rows[i] for i in abs_order[:TOPK]]
    topk_csv_path = os.path.join(OUTPUT_ROOT, "feature_contribution_topk.csv")
    _write_csv(topk_csv_path, topk_rows,
               ["rank_abs","rank_positive","selected_dim_index","raw_feature_index",
                "layer","diff","cov_inv_dot_diff","contribution","abs_contribution",
                "contribution_sign","contribution_fraction_abs","contribution_fraction_positive",
                "position_bin","case_id"])

    # summary JSON
    n_pos = int((contrib > 0).sum())
    n_neg = int((contrib < 0).sum())
    topk_abs_list = [
        {"rank": rank_abs_map[i], "selected_dim": i, "raw_dim": raw_indices[i],
         "layer": layer_labels[i], "contribution": float(contrib[i]),
         "abs_contribution": float(abs(contrib[i]))}
        for i in abs_order[:10]
    ]
    topk_pos_list = [
        {"rank": rank_pos_map[i], "selected_dim": i, "raw_dim": raw_indices[i],
         "layer": layer_labels[i], "contribution": float(contrib[i])}
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
                "fraction_abs":    float(sum(abs(contrib[i]) for i in idx_in_layer) / sum_abs_contrib)
                                   if sum_abs_contrib > 0 else 0.0,
            }

    summary = {
        "case_id":                      CASE_ID,
        "position_bin":                 POSITION_BIN,
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
        "score_expected":               score_expected,
        "score_recomputed_from_feature": score_recomputed,
        "score_diff":                   abs(score_recomputed - score_expected) if score_expected is not None else None,
        "diagnostic_guard_passed":      True,
        "diagnostic_caveat":            DIAGNOSTIC_CAVEAT,
        "negative_contribution_policy": NEGATIVE_CONTRIBUTION_POLICY,
        "stage2_holdout_accessed":      False,
        "ct_load_occurred":             False,
        "model_forward_occurred":       False,
        "feature_extraction_occurred":  False,
        "existing_artifacts_modified":  False,
    }
    summary_path = os.path.join(OUTPUT_ROOT, "feature_contribution_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # xai_reason_bridge
    bridge = {
        "s4_card_case_id":            CASE_ID,
        "s5_case_id":                 CASE_ID,
        "contribution_method":        CONTRIBUTION_METHOD,
        "visual_reason_summary":      (
            f"PaDiM lower_peripheral 정상 분포 대비 Mahalanobis distance 기여도 상위 차원 식별. "
            f"sqrt(d2)={sqrt_mah:.4f} (expected≈{score_expected})"
        ),
        "feature_contribution_summary": {
            "n_positive": n_pos,
            "n_negative": n_neg,
            "top1_abs_dim": topk_abs_list[0] if topk_abs_list else None,
            "top1_pos_dim": topk_pos_list[0] if topk_pos_list else None,
        },
        "caveat":           DIAGNOSTIC_CAVEAT,
        "not_diagnostic":   True,
        "not_gradcam":      True,
        "not_spatial_heatmap_yet": True,
        "note": (
            "feature dimension은 사람이 직접 이해 가능한 CT 구조가 아님. "
            "layer grouping은 raw448 boundary 기준이며 spatial 위치 의미 없음."
        ),
    }
    bridge_path = os.path.join(OUTPUT_ROOT, "xai_reason_bridge.json")
    with open(bridge_path, "w") as f:
        json.dump(bridge, f, indent=2, ensure_ascii=False)

    # errors.csv (empty — header only)
    errors_path = os.path.join(OUTPUT_ROOT, "errors.csv")
    with open(errors_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id","error_type","message"])
        writer.writeheader()

    # DONE.json
    done = {
        "status":           "DONE",
        "case_id":          CASE_ID,
        "sqrt_mahalanobis": sqrt_mah,
        "sum_match_pass":   summary["sum_match_pass"],
        "n_positive_contrib": n_pos,
        "n_negative_contrib": n_neg,
    }
    with open(done_path, "w") as f:
        json.dump(done, f, indent=2)

    # 결과 출력
    print(f"sqrt_mahalanobis  = {sqrt_mah:.5f}  (expected ≈ {score_expected})")
    print(f"sum_match_error   = {sum_err:.2e}  ({'PASS' if sum_err < SUM_MATCH_TOLERANCE else 'FAIL'})")
    print(f"n_positive_contrib = {n_pos},  n_negative_contrib = {n_neg}")
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
# K. main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="S5 Feature Contribution 1-case Smoke Script"
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
