"""
Phase 8.3 stage2_holdout scoring smoke preflight

목적:
  stage2_holdout dedicated 6ch crop manifest를 v1/v1 RD4AD scoring pipeline에
  연결할 수 있는지 사전 검증한다.

금지 사항 (절대 금지):
  - model forward 금지
  - scoring 금지
  - metric / threshold / p95 / p99 / hit-rate 계산 금지
  - training 금지
  - checkpoint 생성 금지
  - full scoring 금지
  - stage2_holdout score CSV 생성 금지
  - crop / manifest 수정 금지
  - 기존 Phase 6/7/8 output 수정 금지
  - torch.load / checkpoint load 금지
  - 전체 npz 로드 금지
  - pip / conda install 금지
  - 외부 다운로드 금지

참고 파일 (read-only):
  scripts/phase7_1_v1v1_scoring_smoke.py
  scripts/phase7_2_v1v1_stage1_dev_full_scoring.py
  scripts/train_rd4ad_2p5d_normal.py
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────
# Project root
# ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────
# Input paths
# ──────────────────────────────────────────────────────────
MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/datasets"
    / "s6a_stage2_holdout_filtered_manifest_v1.csv"
)
CROP_ROOT = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/crops_stage2_holdout_6ch_dedicated_v1"
)
PHASE8_2G_JSON = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_2g_full_crop_output_validation_v1"
    / "phase8_2g_full_crop_output_validation_v1.json"
)
PHASE8_2F_DONE_JSON = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_2f_stage2_dedicated_6ch_crop_generation_v1"
    / "phase8_2f_stage2_dedicated_6ch_crop_generation_DONE.json"
)
PHASE8_2F_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_2f_stage2_dedicated_6ch_crop_generation_v1"
)
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/models"
    / "rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/best_val_loss.pt"
)
PHASE6_2B_JSON = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase6_2b_s6a_model_forward_smoke_v1"
    / "phase6_2b_s6a_model_forward_smoke_v1.json"
)
PHASE7_1_SCRIPT = PROJECT_ROOT / "scripts/phase7_1_v1v1_scoring_smoke.py"
PHASE7_2_SCRIPT = PROJECT_ROOT / "scripts/phase7_2_v1v1_stage1_dev_full_scoring.py"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/train_rd4ad_2p5d_normal.py"

# ──────────────────────────────────────────────────────────
# Expected values (stage2_holdout 기준)
# ──────────────────────────────────────────────────────────
EXPECTED_ROW_COUNT = 143735
EXPECTED_PATIENT_COUNT = 154
EXPECTED_POSITIVE_COUNT = 51335
EXPECTED_HARD_NEG_COUNT = 92400
EXPECTED_ERROR_COUNT = 0
EXPECTED_STAGE_SPLIT = "stage2_holdout"

REQUIRED_MANIFEST_COLUMNS = [
    "row_id",
    "patient_id",
    "safe_id",
    "npz_path",
    "label",
    "sampling_label",
    "stage_split",
    "asset_scope",
    "contamination_check_status",
    "approval_required_before_scoring",
    "crop_shape",
    "input_channels",
    "crop_size",
]

CANONICAL_27_COLUMNS = [
    "crop_id",
    "patient_id",
    "npz_path",
    "label",
    "sampling_label",
    "stage_split",
    "model_tag",
    "checkpoint_path",
    "crop_score_l1_mean",
    "crop_score_l1_max",
    "crop_score_mse_mean",
    "channel_0_l1_mean",
    "channel_1_l1_mean",
    "channel_2_l1_mean",
    "channel_3_l1_mean",
    "channel_4_l1_mean",
    "channel_5_l1_mean",
    "lung_channels_l1_mean",
    "mediastinal_channels_l1_mean",
    "input_min",
    "input_max",
    "recon_min",
    "recon_max",
    "error_min",
    "error_max",
    "has_nan",
    "has_inf",
]

# ──────────────────────────────────────────────────────────
# Output paths
# ──────────────────────────────────────────────────────────
OUT_DIR = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_3_stage2_scoring_smoke_preflight_v1"
)
OUT_CSV_NAME = "phase8_3_stage2_scoring_smoke_preflight_v1.csv"
OUT_JSON_NAME = "phase8_3_stage2_scoring_smoke_preflight_v1.json"
OUT_MD_NAME = "phase8_3_stage2_scoring_smoke_preflight_report_v1.md"

# ──────────────────────────────────────────────────────────
# Smoke / full scoring output path candidates (설계만, 생성 금지)
# ──────────────────────────────────────────────────────────
SMOKE_SCORE_ROOT_CANDIDATE = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/scores"
    / "phase8_3_stage2_scoring_smoke_v1"
)
SMOKE_REVIEW_ROOT_CANDIDATE = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_3_stage2_scoring_smoke_v1"
)
FULL_SCORE_ROOT_CANDIDATE = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/scores"
    / "phase8_4_stage2_full_scoring_v1"
)
PHASE7_2_SCORE_ROOT = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/scores"
    / "phase7_2_v1v1_stage1_dev_full_scoring_v1"
)


# ──────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 8.3 stage2_holdout scoring smoke preflight. "
            "--run 플래그가 없으면 실행되지 않습니다."
        )
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="[필수] preflight 실행 플래그.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# Output guard
# ──────────────────────────────────────────────────────────
def check_output_guard():
    if OUT_DIR.exists():
        print(f"[BLOCKED] output root already exists: {OUT_DIR}")
        print("[BLOCKED] BLOCKED_OUTPUT_CONFLICT: 기존 output root가 존재합니다. 삭제 후 재실행하세요.")
        sys.exit(1)


def check_file_guard(path: Path):
    if path.exists():
        print(f"[BLOCKED] output file already exists: {path}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────
# Section A: prior phase readiness
# ──────────────────────────────────────────────────────────
def run_section_a(manifest_df: pd.DataFrame, phase8_2g_data: dict) -> list:
    rows = []

    def make_row(item, expected, observed, status, note=""):
        return {
            "section": "A",
            "item": item,
            "check_item": item,
            "expected": str(expected),
            "observed": str(observed),
            "status": status,
            "note": note,
            "path": "",
            "planned_value": "",
            "rationale": "",
            "blocker": "",
            "next_required_action": "",
            "script_path": "",
            "output_type": "",
            "exists": "",
        }

    # A-1: Phase 8.2F DONE marker
    done_exists = PHASE8_2F_DONE_JSON.exists()
    rows.append(make_row(
        "Phase8_2F_DONE_marker",
        "exists",
        "exists" if done_exists else "missing",
        "PASS" if done_exists else "FAIL",
        str(PHASE8_2F_DONE_JSON) if done_exists else f"not found: {PHASE8_2F_DONE_JSON}",
    ))

    # A-2: Phase 8.2G readiness
    g_readiness = phase8_2g_data.get("readiness_for_phase8_3", "")
    expected_readiness = "READY_FOR_PHASE8_3_STAGE2_SCORING_SMOKE_PREFLIGHT"
    rows.append(make_row(
        "Phase8_2G_readiness",
        expected_readiness,
        g_readiness,
        "PASS" if g_readiness == expected_readiness else "FAIL",
    ))

    # A-3: manifest row count (실제 집계)
    actual_rows = len(manifest_df)
    rows.append(make_row(
        "manifest_row_count",
        EXPECTED_ROW_COUNT,
        actual_rows,
        "PASS" if actual_rows == EXPECTED_ROW_COUNT else "FAIL",
    ))

    # A-4: patient count
    actual_patients = manifest_df["patient_id"].nunique()
    rows.append(make_row(
        "patient_count",
        EXPECTED_PATIENT_COUNT,
        actual_patients,
        "PASS" if actual_patients == EXPECTED_PATIENT_COUNT else "FAIL",
    ))

    # A-5: positive count
    actual_pos = int((manifest_df["sampling_label"] == "positive").sum())
    rows.append(make_row(
        "positive_count",
        EXPECTED_POSITIVE_COUNT,
        actual_pos,
        "PASS" if actual_pos == EXPECTED_POSITIVE_COUNT else "FAIL",
    ))

    # A-6: hard_negative count
    actual_hn = int((manifest_df["sampling_label"] == "hard_negative").sum())
    rows.append(make_row(
        "hard_negative_count",
        EXPECTED_HARD_NEG_COUNT,
        actual_hn,
        "PASS" if actual_hn == EXPECTED_HARD_NEG_COUNT else "FAIL",
    ))

    # A-7: error count (Phase 8.2G JSON 근거)
    phase8f_summary = phase8_2g_data.get("phase8f_summary", {})
    actual_err = phase8f_summary.get("total_error", -1)
    rows.append(make_row(
        "error_count",
        EXPECTED_ERROR_COUNT,
        actual_err,
        "PASS" if actual_err == EXPECTED_ERROR_COUNT else "FAIL",
        "Phase 8.2G JSON phase8f_summary.total_error 기준",
    ))

    return rows


# ──────────────────────────────────────────────────────────
# Section B: scoring input manifest validation
# ──────────────────────────────────────────────────────────
def run_section_b(manifest_df: pd.DataFrame, phase8_2g_data: dict) -> list:
    rows = []

    def make_row(check_item, expected, observed, status, note=""):
        return {
            "section": "B",
            "item": check_item,
            "check_item": check_item,
            "expected": str(expected),
            "observed": str(observed),
            "status": status,
            "note": note,
            "path": "",
            "planned_value": "",
            "rationale": "",
            "blocker": "",
            "next_required_action": "",
            "script_path": "",
            "output_type": "",
            "exists": "",
        }

    # B-1: 필수 컬럼 13개
    actual_cols = set(manifest_df.columns.tolist())
    missing_cols = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in actual_cols]
    rows.append(make_row(
        "required_columns_13",
        "all_present",
        "all_present" if not missing_cols else f"missing:{missing_cols}",
        "PASS" if not missing_cols else "FAIL",
        f"required={REQUIRED_MANIFEST_COLUMNS}",
    ))

    # B-2: stage_split unique = stage2_holdout
    stage_splits = manifest_df["stage_split"].unique().tolist()
    rows.append(make_row(
        "stage_split_unique",
        "stage2_holdout",
        str(stage_splits),
        "PASS" if stage_splits == [EXPECTED_STAGE_SPLIT] else "FAIL",
    ))

    # B-3: approval_required_before_scoring=True 행 수
    if "approval_required_before_scoring" in manifest_df.columns:
        approval_true_count = int(
            manifest_df["approval_required_before_scoring"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
            .sum()
        )
        rows.append(make_row(
            "approval_required_before_scoring_true_count",
            "0 (preferred)",
            approval_true_count,
            "PASS" if approval_true_count == 0 else "WARN",
            "0이면 전행 scoring 가능, 0보다 크면 해소 후 scoring 진행",
        ))
    else:
        rows.append(make_row(
            "approval_required_before_scoring_true_count",
            "column_exists",
            "column_missing",
            "FAIL",
        ))

    # B-4: npz_path exists count (Phase 8.2G JSON 근거, 전체 npz 로드 금지)
    phase8f_summary = phase8_2g_data.get("phase8f_summary", {})
    npz_exists_count = phase8f_summary.get("npz_path_exists_count", -1)
    npz_missing_count = phase8f_summary.get("npz_path_missing_count", -1)
    rows.append(make_row(
        "npz_path_exists_count",
        EXPECTED_ROW_COUNT,
        npz_exists_count,
        "PASS" if npz_exists_count == EXPECTED_ROW_COUNT else "FAIL",
        f"Phase 8.2G JSON 근거 (npz_path_missing={npz_missing_count}). 이번 preflight에서 전체 npz 로드 금지.",
    ))

    return rows


# ──────────────────────────────────────────────────────────
# Section C: model/config/checkpoint readiness
# ──────────────────────────────────────────────────────────
def run_section_c(phase6_2b_data: dict) -> list:
    rows = []

    def make_row(item, path, expected, observed, status, note=""):
        return {
            "section": "C",
            "item": item,
            "check_item": item,
            "expected": str(expected),
            "observed": str(observed),
            "status": status,
            "note": note,
            "path": str(path),
            "planned_value": "",
            "rationale": "",
            "blocker": "",
            "next_required_action": "",
            "script_path": "",
            "output_type": "",
            "exists": "",
        }

    # C-1: checkpoint 존재
    ckpt_exists = CHECKPOINT_PATH.exists()
    rows.append(make_row(
        "checkpoint_exists",
        CHECKPOINT_PATH,
        "exists",
        "exists" if ckpt_exists else "missing",
        "PASS" if ckpt_exists else "FAIL",
    ))

    # C-2: checkpoint 0-byte 아님
    if ckpt_exists:
        ckpt_size = CHECKPOINT_PATH.stat().st_size
        rows.append(make_row(
            "checkpoint_size_nonzero",
            CHECKPOINT_PATH,
            ">0",
            ckpt_size,
            "PASS" if ckpt_size > 0 else "FAIL",
            f"size={ckpt_size} bytes",
        ))
    else:
        rows.append(make_row(
            "checkpoint_size_nonzero",
            CHECKPOINT_PATH,
            ">0",
            "N/A (file missing)",
            "SKIP",
        ))

    # C-3: model class 확인 (train_rd4ad_2p5d_normal.py read-only)
    model_class = "UNKNOWN"
    if TRAIN_SCRIPT.exists():
        train_text = TRAIN_SCRIPT.read_text(encoding="utf-8")
        m = re.search(r"class\s+(ConvAutoencoder\w+)\s*\(nn\.Module\)", train_text)
        if m:
            model_class = m.group(1)
    rows.append(make_row(
        "model_class",
        TRAIN_SCRIPT,
        "ConvAutoencoder2p5D",
        model_class,
        "PASS" if model_class == "ConvAutoencoder2p5D" else "FAIL",
        "train_rd4ad_2p5d_normal.py read-only 확인",
    ))

    # C-4: input_channels=6 (Phase 6.2b 근거 + train script)
    phase6_shapes = phase6_2b_data.get("input_shapes", [])
    input_channels_ok = False
    observed_channels = "N/A"
    if phase6_shapes:
        # shape = [N, C, H, W]
        ch = phase6_shapes[0][1] if len(phase6_shapes[0]) >= 2 else -1
        observed_channels = ch
        input_channels_ok = (ch == 6)
    rows.append(make_row(
        "input_channels",
        "N/A",
        6,
        observed_channels,
        "PASS" if input_channels_ok else "FAIL",
        "Phase 6.2b JSON input_shapes 근거 (torch.load 금지)",
    ))

    # C-5: expected shape=(6,96,96) 확인 (Phase 6.2b 근거)
    shape_ok = False
    observed_shape = "N/A"
    if phase6_shapes:
        s = tuple(phase6_shapes[0][1:]) if len(phase6_shapes[0]) >= 4 else ()
        observed_shape = str(s)
        shape_ok = (s == (6, 96, 96))
    rows.append(make_row(
        "expected_input_shape",
        "N/A",
        "(6,96,96)",
        observed_shape,
        "PASS" if shape_ok else "FAIL",
        "Phase 6.2b JSON input_shapes[0][1:] 기준",
    ))

    # C-6: Phase 6.2b checkpoint_loaded=true 확인
    ckpt_loaded = phase6_2b_data.get("checkpoint_loaded", False)
    rows.append(make_row(
        "phase6_2b_checkpoint_loaded",
        PHASE6_2B_JSON,
        True,
        ckpt_loaded,
        "PASS" if ckpt_loaded is True else "FAIL",
        "Phase 6.2b smoke test에서 checkpoint load 통과 여부",
    ))

    # C-7: no torch.load this preflight (명시적 확인)
    rows.append(make_row(
        "no_torch_load_this_preflight",
        "N/A",
        "torch.load 금지",
        "CONFIRMED",
        "PASS",
        "이번 preflight에서는 torch.load/checkpoint load 완전 금지",
    ))

    return rows


# ──────────────────────────────────────────────────────────
# Section D: scoring script compatibility
# ──────────────────────────────────────────────────────────
def run_section_d() -> list:
    rows = []

    def make_row(script_path, check_item, expected, observed, status, note=""):
        return {
            "section": "D",
            "item": check_item,
            "check_item": check_item,
            "expected": str(expected),
            "observed": str(observed),
            "status": status,
            "note": note,
            "path": str(script_path),
            "planned_value": "",
            "rationale": "",
            "blocker": "",
            "next_required_action": "",
            "script_path": str(script_path),
            "output_type": "",
            "exists": "",
        }

    # Phase 7.2 script 분석
    p72_text = ""
    p72_exists = PHASE7_2_SCRIPT.exists()
    if p72_exists:
        p72_text = PHASE7_2_SCRIPT.read_text(encoding="utf-8")

    # D-1: Phase 7.2 script 존재
    rows.append(make_row(
        PHASE7_2_SCRIPT,
        "phase7_2_script_exists",
        "exists",
        "exists" if p72_exists else "missing",
        "PASS" if p72_exists else "FAIL",
    ))

    # D-2: hardcoded stage1_dev manifest 경로 있는지
    has_hardcoded_manifest = "phase6_1b_s6a_stage1_dev_filtered_manifest_v1.csv" in p72_text
    rows.append(make_row(
        PHASE7_2_SCRIPT,
        "hardcoded_stage1_dev_manifest",
        "present (DEFAULT_MANIFEST)",
        "YES" if has_hardcoded_manifest else "NO",
        "WARN" if has_hardcoded_manifest else "PASS",
        "DEFAULT_MANIFEST은 하드코딩이지만 --manifest 인자로 override 가능",
    ))

    # D-3: --manifest 인자 override 가능 여부
    has_manifest_arg = '"--manifest"' in p72_text or "'--manifest'" in p72_text
    rows.append(make_row(
        PHASE7_2_SCRIPT,
        "manifest_arg_override_possible",
        "True",
        str(has_manifest_arg),
        "PASS" if has_manifest_arg else "FAIL",
        "--manifest 인자로 stage2_holdout manifest 경로 override 가능 여부",
    ))

    # D-4: hardcoded output root 있는지
    has_hardcoded_output = "phase7_2_v1v1_stage1_dev_full_scoring_v1" in p72_text
    rows.append(make_row(
        PHASE7_2_SCRIPT,
        "hardcoded_output_root",
        "present (DEFAULT_OUTPUT_ROOT)",
        "YES" if has_hardcoded_output else "NO",
        "WARN" if has_hardcoded_output else "PASS",
        "DEFAULT_OUTPUT_ROOT는 하드코딩이지만 --output-root 인자로 override 가능",
    ))

    # D-5: manifest safety 검증 상수 (EXPECTED_MANIFEST_ROWS = 129437) stage2_holdout과 불일치
    has_hardcoded_rows_129437 = "EXPECTED_MANIFEST_ROWS = 129437" in p72_text
    rows.append(make_row(
        PHASE7_2_SCRIPT,
        "manifest_safety_constants_stage1_dev_hardcoded",
        "stage2_holdout 기준으로 재정의 필요",
        "YES: EXPECTED_MANIFEST_ROWS=129437" if has_hardcoded_rows_129437 else "NO",
        "WARN" if has_hardcoded_rows_129437 else "PASS",
        (
            "verify_manifest_safety()가 EXPECTED_MANIFEST_ROWS=129437, "
            "EXPECTED_UNIQUE_PATIENTS=152, EXPECTED_STAGE2_HOLDOUT_ROWS=0으로 "
            "하드코딩됨. stage2_holdout manifest(143735행/154명)는 이 검증에서 실패함. "
            "→ stage2_holdout 전용 scoring script 필요."
        ),
    ))

    # D-6: --run-full 플래그 필요 (stage2_holdout smoke에서는 별도 플래그 필요)
    has_run_full = "--run-full" in p72_text
    rows.append(make_row(
        PHASE7_2_SCRIPT,
        "run_flag",
        "--run-full 플래그 필요",
        "--run-full" if has_run_full else "N/A",
        "INFO",
        "stage2_holdout 전용 script에서는 --run-smoke 같은 별도 플래그 설계 권장",
    ))

    # D-7: canonical 27-column schema 유지 가능 여부
    has_canonical = "CANONICAL_COLUMNS" in p72_text and "27" in p72_text
    rows.append(make_row(
        PHASE7_2_SCRIPT,
        "canonical_27col_schema_available",
        "CANONICAL_COLUMNS 정의됨",
        "YES" if has_canonical else "PARTIAL",
        "PASS" if has_canonical else "WARN",
        (
            "stage2_holdout 전용 script에서 동일 CANONICAL_COLUMNS 정의 재사용 가능. "
            "canonical schema 유지 가능."
        ),
    ))

    # D-8: stage2_holdout 전용 scoring smoke script 필요 여부
    rows.append(make_row(
        PHASE7_2_SCRIPT,
        "dedicated_stage2_scoring_script_needed",
        "True",
        "True",
        "PASS",
        (
            "이유: verify_manifest_safety() 내 stage1_dev 기준 상수 하드코딩으로 "
            "override만으로는 stage2_holdout 검증 실패. "
            "권장 파일명: scripts/phase8_3b_stage2_scoring_smoke.py"
        ),
    ))

    # Phase 7.1 script 존재 확인
    p71_exists = PHASE7_1_SCRIPT.exists()
    rows.append(make_row(
        PHASE7_1_SCRIPT,
        "phase7_1_script_exists",
        "exists",
        "exists" if p71_exists else "missing",
        "PASS" if p71_exists else "WARN",
        "Phase 7.1 smoke script: 구조 참조용",
    ))

    return rows


# ──────────────────────────────────────────────────────────
# Section E: smoke scoring design
# ──────────────────────────────────────────────────────────
def run_section_e() -> list:
    rows = []

    def make_row(item, planned_value, rationale, status, note=""):
        return {
            "section": "E",
            "item": item,
            "check_item": item,
            "expected": "",
            "observed": "",
            "status": status,
            "note": note,
            "path": "",
            "planned_value": str(planned_value),
            "rationale": rationale,
            "blocker": "",
            "next_required_action": "",
            "script_path": "",
            "output_type": "",
            "exists": "",
        }

    rows.append(make_row(
        "smoke_target",
        "--max-crops 64 (positive/hard_negative 각 32)",
        "빠른 pipeline 연결 검증 + positive/hard_negative 둘 다 포함 확인",
        "PLAN",
    ))
    rows.append(make_row(
        "smoke_max_patients_alternative",
        "--max-patients 1~3",
        "환자 단위로 smoke 실행할 경우 대안",
        "PLAN",
    ))
    rows.append(make_row(
        "smoke_script_name",
        "scripts/phase8_3b_stage2_scoring_smoke.py",
        "stage2_holdout 전용 smoke script. phase7_2 참조 구조 재사용.",
        "PLAN",
    ))
    rows.append(make_row(
        "smoke_score_output_root",
        str(SMOKE_SCORE_ROOT_CANDIDATE),
        "stage1_dev score output과 분리된 dedicated root",
        "PLAN",
    ))
    rows.append(make_row(
        "smoke_review_output_root",
        str(SMOKE_REVIEW_ROOT_CANDIDATE),
        "smoke run 검토 보고서 저장 위치",
        "PLAN",
    ))
    rows.append(make_row(
        "full_score_output_root_design_only",
        str(FULL_SCORE_ROOT_CANDIDATE),
        "full scoring용 output root 설계. 이번 preflight에서 생성 금지.",
        "DESIGN_ONLY",
        "phase8_3b smoke 통과 후 별도 승인으로 진행",
    ))
    rows.append(make_row(
        "model_tag",
        "rd4ad_2p5d_normal_mw_fixed96_v1",
        "Phase 7.2와 동일한 model tag 유지",
        "PLAN",
    ))
    rows.append(make_row(
        "canonical_schema",
        f"{len(CANONICAL_27_COLUMNS)}-column schema",
        "Phase 7.2 CANONICAL_COLUMNS 동일 재사용",
        "PLAN",
    ))

    return rows


# ──────────────────────────────────────────────────────────
# Section F: output collision check
# ──────────────────────────────────────────────────────────
def run_section_f() -> list:
    rows = []

    def make_row(output_type, path, exists, status, note=""):
        return {
            "section": "F",
            "item": output_type,
            "check_item": output_type,
            "expected": "",
            "observed": "",
            "status": status,
            "note": note,
            "path": str(path),
            "planned_value": "",
            "rationale": "",
            "blocker": "",
            "next_required_action": "",
            "script_path": "",
            "output_type": output_type,
            "exists": str(exists),
        }

    # F-1: phase8_3 smoke score output 후보
    smoke_score_exists = SMOKE_SCORE_ROOT_CANDIDATE.exists()
    rows.append(make_row(
        "phase8_3_smoke_score_output",
        SMOKE_SCORE_ROOT_CANDIDATE,
        smoke_score_exists,
        "BLOCKED_OUTPUT_CONFLICT" if smoke_score_exists else "PASS",
        "이미 존재하면 BLOCKED" if smoke_score_exists else "충돌 없음",
    ))

    # F-2: phase8_3 smoke review output 후보
    smoke_review_exists = SMOKE_REVIEW_ROOT_CANDIDATE.exists()
    rows.append(make_row(
        "phase8_3_smoke_review_output",
        SMOKE_REVIEW_ROOT_CANDIDATE,
        smoke_review_exists,
        "WARN" if smoke_review_exists else "PASS",
        "이미 존재하면 smoke 실행 전 확인 필요" if smoke_review_exists else "충돌 없음",
    ))

    # F-3: phase8_4 full score output 후보 (note만, 생성 금지)
    full_score_exists = FULL_SCORE_ROOT_CANDIDATE.exists()
    rows.append(make_row(
        "phase8_4_full_score_output_candidate",
        FULL_SCORE_ROOT_CANDIDATE,
        full_score_exists,
        "NOTE" if full_score_exists else "PASS",
        "이미 존재함. 이번 preflight에서 생성 금지." if full_score_exists else "미존재. 이번 preflight에서 생성 금지.",
    ))

    # F-4: Phase 7.2 score output (수정·삭제·덮어쓰기 금지)
    p72_score_exists = PHASE7_2_SCORE_ROOT.exists()
    rows.append(make_row(
        "phase7_2_score_output_untouched",
        PHASE7_2_SCORE_ROOT,
        p72_score_exists,
        "PASS",
        f"존재={p72_score_exists}. 이번 preflight에서 수정·삭제·덮어쓰기 금지. 확인됨.",
    ))

    return rows


# ──────────────────────────────────────────────────────────
# Section G: readiness decision
# ──────────────────────────────────────────────────────────
def run_section_g(all_rows: list) -> tuple:
    """returns (readiness_str, blockers_list, section_g_rows)"""
    # 전체 FAIL / BLOCKED_OUTPUT_CONFLICT 수집
    blockers = []
    for r in all_rows:
        sec = r.get("section", "")
        st = r.get("status", "")
        item = r.get("item", r.get("check_item", ""))
        if st == "FAIL":
            blockers.append(f"Section {sec} / {item}: FAIL")
        elif st == "BLOCKED_OUTPUT_CONFLICT":
            blockers.append(f"Section {sec} / {item}: BLOCKED_OUTPUT_CONFLICT")

    # readiness 판정
    has_collision = any("BLOCKED_OUTPUT_CONFLICT" in b for b in blockers)
    has_manifest_fail = any("Section A" in b or "Section B" in b for b in blockers)
    has_model_fail = any("Section C" in b for b in blockers)
    has_schema_fail = any("canonical_27col" in b for b in blockers)

    if has_collision:
        readiness = "BLOCKED_OUTPUT_CONFLICT"
    elif has_manifest_fail:
        readiness = "BLOCKED_STAGE2_MANIFEST_INVALID"
    elif has_model_fail:
        readiness = "BLOCKED_MODEL_CONFIG_CHECKPOINT_MISSING"
    elif has_schema_fail:
        readiness = "BLOCKED_UNCLEAR_SCORE_SCHEMA"
    elif blockers:
        readiness = "BLOCKED_SCORING_SCRIPT_INCOMPATIBLE"
    else:
        readiness = "READY_FOR_PHASE8_3B_STAGE2_SCORING_SMOKE_SCRIPT"

    blocker_str = "; ".join(blockers) if blockers else ""
    next_action = (
        "Phase 8.3B stage2 scoring smoke script 작성 및 실행 전 검토 진행"
        if readiness == "READY_FOR_PHASE8_3B_STAGE2_SCORING_SMOKE_SCRIPT"
        else f"blocker 해소 필요: {blocker_str}"
    )

    g_rows = [{
        "section": "G",
        "item": "readiness_decision",
        "check_item": "readiness_decision",
        "expected": "READY_FOR_PHASE8_3B_STAGE2_SCORING_SMOKE_SCRIPT",
        "observed": readiness,
        "status": "PASS" if "READY" in readiness else "BLOCKED",
        "note": f"blockers={len(blockers)}",
        "path": "",
        "planned_value": "",
        "rationale": "",
        "blocker": blocker_str,
        "next_required_action": next_action,
        "script_path": "",
        "output_type": "",
        "exists": "",
    }]

    return readiness, blockers, g_rows


# ──────────────────────────────────────────────────────────
# CSV builder
# ──────────────────────────────────────────────────────────
UNIFIED_COLUMNS = [
    "section",
    "item",
    "check_item",
    "expected",
    "observed",
    "status",
    "note",
    "path",
    "planned_value",
    "rationale",
    "blocker",
    "next_required_action",
    "script_path",
    "output_type",
    "exists",
]


def build_csv(all_rows: list) -> pd.DataFrame:
    df = pd.DataFrame(all_rows, columns=UNIFIED_COLUMNS)
    return df


# ──────────────────────────────────────────────────────────
# JSON builder
# ──────────────────────────────────────────────────────────
def build_json(
    manifest_df: pd.DataFrame,
    sec_a: list,
    sec_b: list,
    sec_c: list,
    sec_d: list,
    sec_e: list,
    sec_f: list,
    readiness: str,
    blockers: list,
) -> dict:
    def to_dict_list(rows):
        return [{k: v for k, v in r.items()} for r in rows]

    return {
        "input_paths": {
            "manifest": str(MANIFEST_PATH),
            "crop_root": str(CROP_ROOT),
            "phase8_2g_json": str(PHASE8_2G_JSON),
            "phase8_2f_done_json": str(PHASE8_2F_DONE_JSON),
            "checkpoint": str(CHECKPOINT_PATH),
            "phase6_2b_json": str(PHASE6_2B_JSON),
            "phase7_1_script": str(PHASE7_1_SCRIPT),
            "phase7_2_script": str(PHASE7_2_SCRIPT),
            "train_script": str(TRAIN_SCRIPT),
        },
        "prior_phase_readiness": to_dict_list(sec_a),
        "scoring_input_manifest_validation": to_dict_list(sec_b),
        "model_config_checkpoint_readiness": to_dict_list(sec_c),
        "scoring_script_compatibility": to_dict_list(sec_d),
        "smoke_scoring_design": to_dict_list(sec_e),
        "output_collision_check": to_dict_list(sec_f),
        "readiness_for_phase8_3b": readiness,
        "blockers": blockers,
        "notes": {
            "preflight_only": True,
            "no_model_forward": True,
            "no_scoring": True,
            "no_metric_calculation": True,
            "no_threshold": True,
            "no_training": True,
            "no_checkpoint_creation": True,
            "no_existing_output_modification": True,
        },
    }


# ──────────────────────────────────────────────────────────
# MD report builder
# ──────────────────────────────────────────────────────────
def build_md(
    sec_a: list,
    sec_b: list,
    sec_c: list,
    sec_d: list,
    sec_e: list,
    sec_f: list,
    readiness: str,
    blockers: list,
    phase8_2g_data: dict,
) -> str:
    def sec_table(rows: list, cols: list) -> str:
        header = " | ".join(cols)
        sep = " | ".join(["---"] * len(cols))
        lines = [f"| {header} |", f"| {sep} |"]
        for r in rows:
            cells = " | ".join(str(r.get(c, "")) for c in cols)
            lines.append(f"| {cells} |")
        return "\n".join(lines)

    readiness_status = "PASS" if "READY" in readiness else "BLOCKED"

    md = f"""# Phase 8.3 stage2_holdout scoring smoke preflight

## 1. Phase 8.3 목적

stage2_holdout dedicated 6ch crop manifest를 v1/v1 RD4AD scoring pipeline에
연결할 수 있는지 사전 검증한다.

- model forward / scoring / metric / threshold / training 모두 금지
- 이번 단계는 preflight only
- 실제 scoring smoke 실행은 이번 preflight 통과 후 별도 승인

---

## 2. Phase 8.2F / 8.2G 결과 요약

| 항목 | 값 |
|---|---|
| Phase 8.2F DONE marker | {PHASE8_2F_DONE_JSON} |
| Phase 8.2G readiness | {phase8_2g_data.get('readiness_for_phase8_3', 'N/A')} |
| manifest row count | {phase8_2g_data.get('phase8f_summary', {}).get('npz_path_exists_count', 'N/A')} |
| patient count | {phase8_2g_data.get('phase8f_summary', {}).get('patient_count', 'N/A')} |
| positive | {phase8_2g_data.get('phase8f_summary', {}).get('positive_count', 'N/A')} |
| hard_negative | {phase8_2g_data.get('phase8f_summary', {}).get('hard_negative_count', 'N/A')} |
| error | {phase8_2g_data.get('phase8f_summary', {}).get('total_error', 'N/A')} |

---

## 3. scoring input manifest validation

{sec_table(sec_b, ['check_item', 'expected', 'observed', 'status', 'note'])}

---

## 4. model/config/checkpoint readiness

{sec_table(sec_c, ['item', 'path', 'expected', 'observed', 'status', 'note'])}

---

## 5. scoring script compatibility

{sec_table(sec_d, ['script_path', 'check_item', 'expected', 'observed', 'status', 'note'])}

---

## 6. smoke scoring design

{sec_table(sec_e, ['item', 'planned_value', 'rationale', 'status', 'note'])}

---

## 7. output collision check

{sec_table(sec_f, ['output_type', 'path', 'exists', 'status', 'note'])}

---

## 8. readiness 판정

**{readiness}** ({readiness_status})

blockers ({len(blockers)}개):
""" + "\n".join(f"- {b}" for b in blockers) + (
        "\n\n(blocker 없음)" if not blockers else ""
    ) + f"""

---

## 9. 다음 단계

"""

    if "READY" in readiness:
        md += """- Phase 8.3B stage2 scoring smoke script 작성 및 실행 전 검토를 진행한다.
- 권장 파일명: `scripts/phase8_3b_stage2_scoring_smoke.py`
- smoke 대상: `--max-crops 64` 또는 `--max-patients 1~3` (positive/hard_negative 둘 다 포함)
- smoke output root: `outputs/second-stage-lesion-refiner-v1/scores/phase8_3_stage2_scoring_smoke_v1/`
- 실제 smoking 실행은 별도 승인 후 진행한다.
"""
    else:
        md += f"- BLOCKED 상태. 아래 blocker를 해소한 후 이 preflight를 재실행한다.\n"
        for b in blockers:
            md += f"  - {b}\n"

    md += """
---

## 10. 금지 사항

- model forward 금지
- scoring 금지
- metric / threshold / p95 / p99 / hit-rate 계산 금지
- training 금지
- checkpoint 생성 금지
- full scoring 금지
- stage2_holdout score CSV 생성 금지
- crop / manifest 수정 금지
- 기존 Phase 6/7/8 output 수정 금지
- torch.load / checkpoint load 금지
- 전체 npz 로드 금지
- pip / conda install 금지
- 외부 다운로드 금지
"""
    return md


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if not args.run:
        print("[ERROR] --run 플래그가 필요합니다.")
        print(
            "사용법: python scripts/phase8_3_stage2_scoring_smoke_preflight.py --run"
        )
        sys.exit(1)

    print("[INFO] Phase 8.3 stage2_holdout scoring smoke preflight 시작")

    # output guard (최우선)
    check_output_guard()

    # manifest 로드
    print(f"[INFO] Loading manifest: {MANIFEST_PATH}")
    if not MANIFEST_PATH.exists():
        print(f"[ERROR] manifest not found: {MANIFEST_PATH}")
        sys.exit(1)
    manifest_df = pd.read_csv(MANIFEST_PATH)
    print(f"[INFO] Manifest loaded: {len(manifest_df)} rows")

    # Phase 8.2G JSON 로드
    print(f"[INFO] Loading Phase 8.2G JSON: {PHASE8_2G_JSON}")
    if not PHASE8_2G_JSON.exists():
        print(f"[ERROR] Phase 8.2G JSON not found: {PHASE8_2G_JSON}")
        sys.exit(1)
    with open(PHASE8_2G_JSON, encoding="utf-8") as f:
        phase8_2g_data = json.load(f)

    # Phase 6.2b JSON 로드
    print(f"[INFO] Loading Phase 6.2b JSON: {PHASE6_2B_JSON}")
    if not PHASE6_2B_JSON.exists():
        print(f"[ERROR] Phase 6.2b JSON not found: {PHASE6_2B_JSON}")
        sys.exit(1)
    with open(PHASE6_2B_JSON, encoding="utf-8") as f:
        phase6_2b_data = json.load(f)

    # 각 섹션 실행
    print("[INFO] Section A: prior phase readiness")
    sec_a = run_section_a(manifest_df, phase8_2g_data)

    print("[INFO] Section B: scoring input manifest validation")
    sec_b = run_section_b(manifest_df, phase8_2g_data)

    print("[INFO] Section C: model/config/checkpoint readiness")
    sec_c = run_section_c(phase6_2b_data)

    print("[INFO] Section D: scoring script compatibility")
    sec_d = run_section_d()

    print("[INFO] Section E: smoke scoring design")
    sec_e = run_section_e()

    print("[INFO] Section F: output collision check")
    sec_f = run_section_f()

    all_rows = sec_a + sec_b + sec_c + sec_d + sec_e + sec_f

    # Section G: readiness decision
    print("[INFO] Section G: readiness decision")
    readiness, blockers, sec_g = run_section_g(all_rows)
    all_rows += sec_g

    print(f"[INFO] readiness_for_phase8_3b = {readiness}")
    if blockers:
        for b in blockers:
            print(f"[WARN] blocker: {b}")

    # output 생성
    print(f"[INFO] Creating output root: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    # CSV 저장
    csv_path = OUT_DIR / OUT_CSV_NAME
    check_file_guard(csv_path)
    df = build_csv(all_rows)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] CSV saved: {csv_path}")

    # JSON 저장
    json_path = OUT_DIR / OUT_JSON_NAME
    check_file_guard(json_path)
    json_data = build_json(
        manifest_df, sec_a, sec_b, sec_c, sec_d, sec_e, sec_f, readiness, blockers
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] JSON saved: {json_path}")

    # MD 저장
    md_path = OUT_DIR / OUT_MD_NAME
    check_file_guard(md_path)
    md_text = build_md(sec_a, sec_b, sec_c, sec_d, sec_e, sec_f, readiness, blockers, phase8_2g_data)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"[INFO] MD saved: {md_path}")

    print(f"\n[DONE] Phase 8.3 preflight 완료")
    print(f"  output root : {OUT_DIR}")
    print(f"  CSV         : {csv_path}")
    print(f"  JSON        : {json_path}")
    print(f"  MD          : {md_path}")
    print(f"  readiness   : {readiness}")
    print(f"  blockers    : {len(blockers)}개")


if __name__ == "__main__":
    main()
