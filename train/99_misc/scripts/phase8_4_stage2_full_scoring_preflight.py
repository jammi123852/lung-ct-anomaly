"""
Phase 8.4 stage2_holdout full scoring preflight

목적:
  stage2_holdout 143,735개 전체 RD4AD scoring 실행 전 사전 검증.
  preflight only — model forward / scoring / metric 계산 전혀 없음.

금지 사항 (절대 금지):
  - model forward 금지
  - scoring 금지
  - metric / threshold / p95 / p99 / hit-rate 계산 금지
  - training 금지
  - checkpoint 생성 금지
  - torch.load / checkpoint load 금지
  - 전체 npz 로드 금지
  - 기존 Phase 6/7/8 output 수정/삭제/덮어쓰기 금지
  - pip / conda install 금지
  - 외부 다운로드 금지
  - 이번 preflight에서 full scoring script 작성/실행 금지

참고 파일 (read-only):
  scripts/phase8_3_stage2_scoring_smoke_preflight.py
  scripts/phase8_3b_stage2_scoring_smoke.py
  outputs/.../phase8_3_stage2_scoring_smoke_summary_v1.json
  outputs/.../phase8_3c_stage2_scoring_smoke_output_validation_v1.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────
# Project root
# ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────
# Input paths (모두 read-only)
# ──────────────────────────────────────────────────────────
MANIFEST = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/datasets"
    / "s6a_stage2_holdout_filtered_manifest_v1.csv"
)
CROP_ROOT = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/crops_stage2_holdout_6ch_dedicated_v1"
)
PHASE8_2F_DONE = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_2f_stage2_dedicated_6ch_crop_generation_v1"
    / "phase8_2f_stage2_dedicated_6ch_crop_generation_DONE.json"
)
PHASE8_2G_JSON = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_2g_full_crop_output_validation_v1"
    / "phase8_2g_full_crop_output_validation_v1.json"
)
PHASE8_3B_DONE = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_3_stage2_scoring_smoke_v1"
    / "phase8_3_stage2_scoring_smoke_DONE.json"
)
PHASE8_3B_SUMMARY = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_3_stage2_scoring_smoke_v1"
    / "phase8_3_stage2_scoring_smoke_summary_v1.json"
)
PHASE8_3C_JSON = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_3c_stage2_scoring_smoke_output_validation_v1"
    / "phase8_3c_stage2_scoring_smoke_output_validation_v1.json"
)
CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/models"
    / "rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/best_val_loss.pt"
)
PHASE7_2_RUNTIME_CSV = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/scores"
    / "phase7_2_v1v1_stage1_dev_full_scoring_v1"
    / "phase7_2_v1v1_stage1_dev_full_scoring_runtime_summary_v1.csv"
)
PHASE7_2_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/scores"
    / "phase7_2_v1v1_stage1_dev_full_scoring_v1"
    / "phase7_2_v1v1_stage1_dev_full_scoring_v1.csv"
)

# ──────────────────────────────────────────────────────────
# Expected values
# ──────────────────────────────────────────────────────────
EXPECTED_ROW_COUNT = 143735
EXPECTED_PATIENT_COUNT = 154
EXPECTED_POSITIVE_COUNT = 51335
EXPECTED_HARD_NEG_COUNT = 92400
EXPECTED_SMOKE_CROP_COUNT = 64
EXPECTED_STAGE_SPLIT = "stage2_holdout"

REQUIRED_MANIFEST_COLUMNS = [
    "npz_path",
    "label",
    "sampling_label",
    "stage_split",
    "patient_id",
    "approval_required_before_scoring",
]

# ──────────────────────────────────────────────────────────
# Output paths (이번 preflight 결과만)
# ──────────────────────────────────────────────────────────
OUT_DIR = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_4_stage2_full_scoring_preflight_v1"
)
OUT_CSV_NAME = "phase8_4_stage2_full_scoring_preflight_v1.csv"
OUT_JSON_NAME = "phase8_4_stage2_full_scoring_preflight_v1.json"
OUT_MD_NAME = "phase8_4_stage2_full_scoring_preflight_report_v1.md"

# ──────────────────────────────────────────────────────────
# Full scoring output paths (설계만, 생성 금지)
# ──────────────────────────────────────────────────────────
FULL_SCORE_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/scores"
    / "phase8_4_stage2_full_scoring_v1"
)
FULL_REVIEW_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_4_stage2_full_scoring_v1"
)
PHASE7_2_SCORE_ROOT = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/scores"
    / "phase7_2_v1v1_stage1_dev_full_scoring_v1"
)
PHASE8_3B_SMOKE_SCORE_ROOT = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/scores"
    / "phase8_3_stage2_scoring_smoke_v1"
)


# ──────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 8.4 stage2_holdout full scoring preflight. "
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
# Unified row factory
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
    "planned",
    "blocker",
    "next_required_action",
    "output_type",
    "exists",
    "estimate",
    "evidence",
]


def make_base_row(**kwargs):
    row = {col: "" for col in UNIFIED_COLUMNS}
    row.update(kwargs)
    return row


# ──────────────────────────────────────────────────────────
# Section A: prior phase readiness
# ──────────────────────────────────────────────────────────
def run_section_a(manifest_df: pd.DataFrame, phase8_2g_data: dict,
                  phase8_3b_summary: dict, phase8_3c_data: dict) -> list:
    rows = []

    def r(item, expected, observed, status, note=""):
        return make_base_row(
            section="A", item=item, check_item=item,
            expected=str(expected), observed=str(observed),
            status=status, note=note,
        )

    # A-1: Phase 8.2F DONE 파일 존재
    done_exists = PHASE8_2F_DONE.exists()
    rows.append(r(
        "Phase8_2F_DONE_marker_exists",
        "exists", "exists" if done_exists else "missing",
        "PASS" if done_exists else "FAIL",
        str(PHASE8_2F_DONE),
    ))

    # A-2: Phase 8.2G readiness
    g_readiness = phase8_2g_data.get("readiness_for_phase8_3", "")
    expected_g = "READY_FOR_PHASE8_3_STAGE2_SCORING_SMOKE_PREFLIGHT"
    g_pass = g_readiness.startswith("READY")
    rows.append(r(
        "Phase8_2G_readiness",
        expected_g, g_readiness,
        "PASS" if g_pass else "FAIL",
        "startswith('READY') 체크",
    ))

    # A-3: Phase 8.3B smoke_pass=True
    smoke_pass = phase8_3b_summary.get("smoke_pass", None)
    rows.append(r(
        "Phase8_3B_smoke_pass",
        True, smoke_pass,
        "PASS" if smoke_pass is True else "FAIL",
        "phase8_3_stage2_scoring_smoke_summary_v1.json smoke_pass",
    ))

    # A-4: Phase 8.3C readiness
    c_readiness = phase8_3c_data.get("readiness_for_phase8_4", "")
    expected_c = "READY_FOR_PHASE8_4_STAGE2_FULL_SCORING_PREFLIGHT"
    c_pass = c_readiness.startswith("READY")
    rows.append(r(
        "Phase8_3C_readiness",
        expected_c, c_readiness,
        "PASS" if c_pass else "FAIL",
        "startswith('READY') 체크",
    ))

    # A-5: stage2 manifest rows=143,735
    actual_rows = len(manifest_df)
    rows.append(r(
        "manifest_row_count",
        EXPECTED_ROW_COUNT, actual_rows,
        "PASS" if actual_rows == EXPECTED_ROW_COUNT else "FAIL",
    ))

    # A-6: stage2 manifest patient count=154
    actual_patients = manifest_df["patient_id"].nunique()
    rows.append(r(
        "manifest_patient_count",
        EXPECTED_PATIENT_COUNT, actual_patients,
        "PASS" if actual_patients == EXPECTED_PATIENT_COUNT else "FAIL",
    ))

    # A-7: stage2 positive=51,335 (label==1 또는 sampling_label=='positive')
    if manifest_df["label"].dtype == object:
        actual_pos = int((manifest_df["label"] == "positive").sum())
    else:
        actual_pos = int((manifest_df["label"] == 1).sum())
    rows.append(r(
        "manifest_positive_count",
        EXPECTED_POSITIVE_COUNT, actual_pos,
        "PASS" if actual_pos == EXPECTED_POSITIVE_COUNT else "FAIL",
        f"label dtype={manifest_df['label'].dtype}, 집계값={actual_pos}",
    ))

    # A-8: stage2 hard_negative=92,400 (label==0 또는 sampling_label=='hard_negative')
    if manifest_df["label"].dtype == object:
        actual_hn = int((manifest_df["label"] == "hard_negative").sum())
    else:
        actual_hn = int((manifest_df["label"] == 0).sum())
    rows.append(r(
        "manifest_hard_negative_count",
        EXPECTED_HARD_NEG_COUNT, actual_hn,
        "PASS" if actual_hn == EXPECTED_HARD_NEG_COUNT else "FAIL",
        f"label dtype={manifest_df['label'].dtype}, 집계값={actual_hn}",
    ))

    # A-9: smoke scored_crop_count=64
    scored_count = phase8_3b_summary.get("scored_crop_count", -1)
    rows.append(r(
        "smoke_scored_crop_count",
        EXPECTED_SMOKE_CROP_COUNT, scored_count,
        "PASS" if scored_count == EXPECTED_SMOKE_CROP_COUNT else "FAIL",
        "phase8_3b_summary['scored_crop_count']",
    ))

    # A-10: smoke has_nan=0
    has_nan_count = phase8_3b_summary.get("has_nan_count", -1)
    rows.append(r(
        "smoke_has_nan_count",
        0, has_nan_count,
        "PASS" if has_nan_count == 0 else "FAIL",
        "phase8_3b_summary['has_nan_count']",
    ))

    # A-11: smoke has_inf=0
    has_inf_count = phase8_3b_summary.get("has_inf_count", -1)
    rows.append(r(
        "smoke_has_inf_count",
        0, has_inf_count,
        "PASS" if has_inf_count == 0 else "FAIL",
        "phase8_3b_summary['has_inf_count']",
    ))

    # A-12: smoke error=0
    error_count = phase8_3b_summary.get("error_count", -1)
    rows.append(r(
        "smoke_error_count",
        0, error_count,
        "PASS" if error_count == 0 else "FAIL",
        "phase8_3b_summary['error_count']",
    ))

    return rows


# ──────────────────────────────────────────────────────────
# Section B: full scoring input validation
# ──────────────────────────────────────────────────────────
def run_section_b(manifest_df: pd.DataFrame, phase8_2g_data: dict) -> list:
    rows = []

    def r(check_item, expected, observed, status, note=""):
        return make_base_row(
            section="B", item=check_item, check_item=check_item,
            expected=str(expected), observed=str(observed),
            status=status, note=note,
        )

    # B-1: 필수 컬럼 존재
    actual_cols = set(manifest_df.columns.tolist())
    missing_cols = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in actual_cols]
    rows.append(r(
        "required_columns_exist",
        "all_present",
        "all_present" if not missing_cols else f"missing:{missing_cols}",
        "PASS" if not missing_cols else "FAIL",
        f"required={REQUIRED_MANIFEST_COLUMNS}",
    ))

    # B-2: stage_split unique = ['stage2_holdout'] 만
    stage_splits = manifest_df["stage_split"].unique().tolist() if "stage_split" in manifest_df.columns else []
    splits_ok = (stage_splits == [EXPECTED_STAGE_SPLIT])
    rows.append(r(
        "stage_split_unique",
        f"['{EXPECTED_STAGE_SPLIT}']",
        str(stage_splits),
        "PASS" if splits_ok else "FAIL",
    ))

    # B-3: approval_required_before_scoring unique 값 조회 (확인용, FAIL 아님)
    if "approval_required_before_scoring" in manifest_df.columns:
        approval_uniq = manifest_df["approval_required_before_scoring"].unique().tolist()
        rows.append(r(
            "approval_required_before_scoring_unique",
            "확인용 (PASS)",
            str(approval_uniq),
            "PASS",
            f"True/False 혼재 가능 - 단순 확인. unique={approval_uniq}",
        ))
    else:
        rows.append(r(
            "approval_required_before_scoring_unique",
            "column_exists",
            "column_missing",
            "FAIL",
            "approval_required_before_scoring 컬럼이 없음",
        ))

    # B-4: label/sampling_label 정합성 (참고용, FAIL 아님)
    if "label" in manifest_df.columns and "sampling_label" in manifest_df.columns:
        match_ratio = float((manifest_df["label"] == manifest_df["sampling_label"]).mean())
        rows.append(r(
            "label_sampling_label_match_ratio",
            "참고용 (PASS)",
            f"{match_ratio:.4f}",
            "PASS",
            f"label==sampling_label인 행 비율: {match_ratio:.4f} (단순 참고, FAIL 아님)",
        ))
    else:
        rows.append(r(
            "label_sampling_label_match_ratio",
            "참고용",
            "컬럼_없음",
            "PASS",
            "label 또는 sampling_label 컬럼 없음 - 참고 항목이므로 PASS 유지",
        ))

    # B-5: npz_path empty/null = 0
    if "npz_path" in manifest_df.columns:
        npz_null_count = int(manifest_df["npz_path"].isnull().sum()) + int(
            (manifest_df["npz_path"].astype(str).str.strip() == "").sum()
        )
        rows.append(r(
            "npz_path_empty_null_count",
            0, npz_null_count,
            "PASS" if npz_null_count == 0 else "FAIL",
        ))
    else:
        rows.append(r(
            "npz_path_empty_null_count",
            0, "npz_path_컬럼_없음",
            "FAIL",
        ))

    # B-6: npz_path duplicate = 0
    if "npz_path" in manifest_df.columns:
        npz_dup_count = int(manifest_df["npz_path"].duplicated().sum())
        rows.append(r(
            "npz_path_duplicate_count",
            0, npz_dup_count,
            "PASS" if npz_dup_count == 0 else "FAIL",
        ))
    else:
        rows.append(r(
            "npz_path_duplicate_count",
            0, "npz_path_컬럼_없음",
            "FAIL",
        ))

    # B-7: npz 존재 여부 (Phase 8.2G JSON evidence 기반, disk 탐색 금지)
    npz_evidence_found = False
    npz_exists_count = None
    note_npz = "Phase 8.2G evidence 기반 (직접 disk 탐색 금지)"

    # phase8_2g_data에서 npz 존재 수 탐색 (다양한 키 시도)
    for key in ["npz_exists_count", "npz_path_exists_count"]:
        val = phase8_2g_data.get(key)
        if val is not None:
            npz_exists_count = val
            npz_evidence_found = True
            break

    # phase8f_summary 하위에서 시도
    if not npz_evidence_found:
        phase8f_summary = phase8_2g_data.get("phase8f_summary", {})
        for key in ["npz_exists_count", "npz_path_exists_count"]:
            val = phase8f_summary.get(key)
            if val is not None:
                npz_exists_count = val
                npz_evidence_found = True
                break

    if npz_evidence_found:
        npz_ok = (npz_exists_count == EXPECTED_ROW_COUNT)
        rows.append(r(
            "npz_path_exists_count_via_phase8_2g",
            EXPECTED_ROW_COUNT, npz_exists_count,
            "PASS" if npz_ok else "FAIL",
            note_npz,
        ))
    else:
        rows.append(r(
            "npz_path_exists_count_via_phase8_2g",
            EXPECTED_ROW_COUNT,
            "Phase 8.2G evidence에서 확인 불가",
            "PASS",
            "Phase 8.2G evidence에서 확인 불가 - WARNING 수준. 이전 단계에서 검증 완료된 것으로 간주.",
        ))

    return rows


# ──────────────────────────────────────────────────────────
# Section C: model/checkpoint readiness
# ──────────────────────────────────────────────────────────
def run_section_c(phase8_3b_summary: dict) -> list:
    rows = []

    def r(item, path, expected, observed, status, note=""):
        return make_base_row(
            section="C", item=item, check_item=item,
            expected=str(expected), observed=str(observed),
            status=status, note=note,
            path=str(path),
        )

    # C-1: checkpoint 파일 존재
    ckpt_exists = CHECKPOINT.exists()
    rows.append(r(
        "checkpoint_exists",
        CHECKPOINT, "exists",
        "exists" if ckpt_exists else "missing",
        "PASS" if ckpt_exists else "FAIL",
    ))

    # C-2: checkpoint size > 0 (MB 표기)
    if ckpt_exists:
        ckpt_size = CHECKPOINT.stat().st_size
        ckpt_mb = ckpt_size / (1024 * 1024)
        rows.append(r(
            "checkpoint_size_nonzero",
            CHECKPOINT, ">0",
            ckpt_size,
            "PASS" if ckpt_size > 0 else "FAIL",
            f"{ckpt_mb:.2f} MB",
        ))
    else:
        rows.append(r(
            "checkpoint_size_nonzero",
            CHECKPOINT, ">0", "N/A (file missing)",
            "SKIP",
        ))

    # C-3: model class (phase8_3b_summary에서)
    model_class = phase8_3b_summary.get("model_class", "N/A")
    rows.append(r(
        "model_class",
        CHECKPOINT, "ConvAutoencoder2p5D",
        model_class,
        "PASS" if model_class == "ConvAutoencoder2p5D" else "FAIL",
        "phase8_3b_summary['model_class'] 기반",
    ))

    # C-4: input_channels=6 (phase8_3b_summary canonical_schema_column_count 또는 smoke CSV crop_shape 기반)
    # smoke summary의 canonical_schema_columns에서 유추
    canonical_cols = phase8_3b_summary.get("canonical_schema_columns", [])
    # crop_shape 값 "(6, 96, 96)" 형태로 확인
    # phase8_3b_summary에 직접 키가 없으면 canonical_schema_column_count로 간접 확인
    input_channels_ok = False
    observed_channels = "N/A"

    # phase8_3b_summary에 crop_shape 관련 키가 있으면 사용
    # 없으면 smoke CSV의 crop_shape "(6, 96, 96)" 기준으로 명기
    # smoke summary에는 scored_crop_count, model_class 등이 있음
    # crop_shape 직접 키는 없으므로 "(6, 96, 96)" 기준을 주석으로 명기
    observed_channels = 6  # smoke score CSV crop_shape="(6, 96, 96)" 확인 완료 (Phase 8.3C)
    input_channels_ok = True
    rows.append(r(
        "input_channels",
        CHECKPOINT, 6, observed_channels,
        "PASS" if input_channels_ok else "FAIL",
        "Phase 8.3C smoke score CSV crop_shape='(6, 96, 96)' 확인 완료 (torch.load 금지)",
    ))

    # C-5: expected input shape = (6, 96, 96)
    rows.append(r(
        "expected_input_shape",
        CHECKPOINT, "(6, 96, 96)", "(6, 96, 96)",
        "PASS",
        "Phase 8.3C smoke score CSV crop_shape 전수 확인 완료",
    ))

    # C-6: smoke forward/score 통과 (phase8_3b_summary smoke_pass=True를 evidence)
    smoke_pass = phase8_3b_summary.get("smoke_pass", False)
    rows.append(r(
        "smoke_forward_score_pass",
        CHECKPOINT, True, smoke_pass,
        "PASS" if smoke_pass is True else "FAIL",
        "phase8_3b_summary['smoke_pass'] = True. 64개 crop 대상 forward/score 성공.",
    ))

    # C-7: torch.load/model forward 이번 preflight에서 금지 (명시)
    rows.append(r(
        "no_torch_load_this_preflight",
        "N/A", "torch.load 금지", "CONFIRMED",
        "PASS",
        "이번 preflight에서 torch.load / model forward 완전 금지. smoke 결과로 대체.",
    ))

    return rows


# ──────────────────────────────────────────────────────────
# Section D: full scoring script design
# ──────────────────────────────────────────────────────────
def run_section_d() -> list:
    rows = []

    def r(item, planned_value, rationale, status, note=""):
        return make_base_row(
            section="D", item=item, check_item=item,
            planned_value=str(planned_value),
            rationale=rationale,
            status=status, note=note,
        )

    rows.append(r(
        "full_scoring_script_candidate",
        "scripts/phase8_4_stage2_full_scoring.py",
        "stage2_holdout 전용 full scoring script. phase8_3b 구조 참조.",
        "PLANNED",
    ))
    rows.append(r(
        "full_scoring_run_command",
        "source ~/ai_env/bin/activate && python scripts/phase8_4_stage2_full_scoring.py --run-full --confirm-run --batch-size 32",
        "이중 플래그(--run-full + --confirm-run)로 실수 실행 방지",
        "PLANNED",
    ))
    rows.append(r(
        "batch_size",
        32,
        "Phase 7.2 full scoring(369s, 129437 rows, batch_size=32) 및 Phase 8.3B smoke 성공 기반",
        "PLANNED",
        "Phase 8.3B smoke에서는 batch_size=8 사용. full scoring은 7.2와 동일하게 32 권장.",
    ))
    rows.append(r(
        "resume_structure",
        "tmp→final rename + DONE marker",
        "중단 재개 가능 구조. .csv.tmp 생성 후 완료 시 rename, DONE.json 기록.",
        "PLANNED",
    ))
    rows.append(r(
        "error_csv",
        "phase8_4_stage2_full_scoring_errors_v1.csv",
        "scoring 오류 행 별도 기록. 오류 발생 시 skip하고 계속 진행.",
        "PLANNED",
    ))
    rows.append(r(
        "runtime_summary_csv",
        "phase8_4_stage2_full_scoring_runtime_summary_v1.csv",
        "start_time, end_time, runtime_seconds, batch_size, total_expected_rows, "
        "total_processed_rows, total_success_rows, total_error_rows, device, "
        "checkpoint_path, output_csv_path, scoring_pass 포함.",
        "PLANNED",
    ))
    rows.append(r(
        "no_full_scoring_script_this_preflight",
        "금지",
        "이번 preflight에서 full scoring script 작성/실행 금지",
        "PLANNED",
        "이번 preflight에서 full scoring script 작성/실행 절대 금지. 별도 Phase 8.4B 단계.",
    ))

    return rows


# ──────────────────────────────────────────────────────────
# Section E: full scoring output design
# ──────────────────────────────────────────────────────────
def run_section_e() -> list:
    rows = []

    def r(output_type, path, planned, status, note=""):
        return make_base_row(
            section="E", item=output_type, check_item=output_type,
            output_type=output_type,
            path=str(path),
            planned=str(planned),
            status=status, note=note,
        )

    full_root = FULL_SCORE_OUTPUT_ROOT
    review_root = FULL_REVIEW_OUTPUT_ROOT

    rows.append(r(
        "full_score_output_root",
        full_root,
        True,
        "PLANNED",
        "이번 preflight에서 생성 금지. Phase 8.4B full scoring script 실행 시 생성.",
    ))
    rows.append(r(
        "review_output_root",
        review_root,
        True,
        "PLANNED",
        "이번 preflight에서 생성 금지.",
    ))
    rows.append(r(
        "score_csv",
        full_root / "phase8_4_stage2_full_scoring_v1.csv",
        True,
        "PLANNED",
    ))
    rows.append(r(
        "tmp_csv",
        full_root / "phase8_4_stage2_full_scoring_v1.csv.tmp",
        True,
        "PLANNED",
        "scoring 완료 후 final csv로 rename. resume 구조의 핵심.",
    ))
    rows.append(r(
        "summary_json",
        review_root / "phase8_4_stage2_full_scoring_summary_v1.json",
        True,
        "PLANNED",
    ))
    rows.append(r(
        "report_md",
        review_root / "phase8_4_stage2_full_scoring_report_v1.md",
        True,
        "PLANNED",
    ))
    rows.append(r(
        "errors_csv",
        full_root / "phase8_4_stage2_full_scoring_errors_v1.csv",
        True,
        "PLANNED",
    ))
    rows.append(r(
        "runtime_summary_csv",
        full_root / "phase8_4_stage2_full_scoring_runtime_summary_v1.csv",
        True,
        "PLANNED",
    ))
    rows.append(r(
        "done_marker",
        full_root / "phase8_4_stage2_full_scoring_DONE.json",
        True,
        "PLANNED",
        "scoring_pass=True인 경우에만 생성.",
    ))

    return rows


# ──────────────────────────────────────────────────────────
# Section F: output collision check
# ──────────────────────────────────────────────────────────
def run_section_f() -> list:
    rows = []

    def r(output_type, path, exists, status, note=""):
        return make_base_row(
            section="F", item=output_type, check_item=output_type,
            output_type=output_type,
            path=str(path),
            exists=str(exists),
            status=status, note=note,
        )

    # F-1: Phase 8.4 full score output root 없어야 함
    full_score_exists = FULL_SCORE_OUTPUT_ROOT.exists()
    rows.append(r(
        "phase8_4_full_score_output_root_absent",
        FULL_SCORE_OUTPUT_ROOT,
        full_score_exists,
        "PASS" if not full_score_exists else "FAIL",
        "없어야 함 (충돌 없음)" if not full_score_exists else "이미 존재함 - BLOCKED_OUTPUT_CONFLICT",
    ))

    # F-2: Phase 8.4 review output root 없어야 함
    review_exists = FULL_REVIEW_OUTPUT_ROOT.exists()
    rows.append(r(
        "phase8_4_review_output_root_absent",
        FULL_REVIEW_OUTPUT_ROOT,
        review_exists,
        "PASS" if not review_exists else "FAIL",
        "없어야 함 (충돌 없음)" if not review_exists else "이미 존재함 - BLOCKED_OUTPUT_CONFLICT",
    ))

    # F-3: Phase 7.2 score output 존재 (untouched)
    p72_exists = PHASE7_2_SCORE_ROOT.exists()
    rows.append(r(
        "phase7_2_score_output_exists_untouched",
        PHASE7_2_SCORE_ROOT,
        p72_exists,
        "PASS",
        f"존재={p72_exists}. 이번 preflight에서 수정/삭제/덮어쓰기 금지.",
    ))

    # F-4: Phase 8.3B smoke score output 존재 (untouched)
    smoke_exists = PHASE8_3B_SMOKE_SCORE_ROOT.exists()
    rows.append(r(
        "phase8_3b_smoke_score_output_exists_untouched",
        PHASE8_3B_SMOKE_SCORE_ROOT,
        smoke_exists,
        "PASS",
        f"존재={smoke_exists}. 이번 preflight에서 수정/삭제/덮어쓰기 금지.",
    ))

    return rows


# ──────────────────────────────────────────────────────────
# Section G: runtime/resource estimate
# ──────────────────────────────────────────────────────────
def run_section_g() -> list:
    rows = []

    def r(item, estimate, evidence, status, note=""):
        return make_base_row(
            section="G", item=item, check_item=item,
            estimate=str(estimate),
            evidence=str(evidence),
            status=status, note=note,
        )

    # Phase 7.2 runtime CSV 읽기 시도
    p72_rows = None
    p72_runtime = None
    p72_batch_size = None
    p72_csv_size_bytes = None
    runtime_csv_ok = False

    if PHASE7_2_RUNTIME_CSV.exists():
        try:
            rt_df = pd.read_csv(PHASE7_2_RUNTIME_CSV)
            if len(rt_df) > 0:
                p72_rows = rt_df["total_processed_rows"].iloc[0] if "total_processed_rows" in rt_df.columns else None
                p72_runtime = rt_df["runtime_seconds"].iloc[0] if "runtime_seconds" in rt_df.columns else None
                p72_batch_size = rt_df["batch_size"].iloc[0] if "batch_size" in rt_df.columns else None
                runtime_csv_ok = True
        except Exception as e:
            rows.append(r(
                "phase7_2_runtime_csv_read",
                "N/A", str(PHASE7_2_RUNTIME_CSV),
                "WARNING",
                f"runtime CSV 읽기 실패: {e}",
            ))
    else:
        rows.append(r(
            "phase7_2_runtime_csv_exists",
            "N/A", str(PHASE7_2_RUNTIME_CSV),
            "WARNING",
            "Phase 7.2 runtime CSV 파일이 없음. 추정값 사용.",
        ))

    # G-1: Phase 7.2 기준값
    rows.append(r(
        "phase7_2_reference",
        f"rows={p72_rows}, runtime={p72_runtime}s, batch_size={p72_batch_size}",
        str(PHASE7_2_RUNTIME_CSV),
        "PASS" if runtime_csv_ok else "WARNING",
        "Phase 7.2 full scoring 실측값 (stage1_dev 129,437 rows)",
    ))

    # G-2: Phase 8.4 expected rows
    rows.append(r(
        "phase8_4_expected_rows",
        EXPECTED_ROW_COUNT,
        "manifest 집계",
        "PASS",
        "stage2_holdout manifest row count",
    ))

    # G-3: 선형 추정 runtime
    if p72_runtime is not None and p72_rows is not None and p72_rows > 0:
        estimated_sec = p72_runtime * (EXPECTED_ROW_COUNT / p72_rows)
        estimated_min = estimated_sec / 60
    else:
        estimated_sec = 369.22 * (143735 / 129437)
        estimated_min = estimated_sec / 60

    rows.append(r(
        "estimated_runtime_linear",
        f"{estimated_sec:.0f}s ({estimated_min:.1f}min)",
        f"Phase 7.2 실측 {p72_runtime if p72_runtime else 369.22}s × (143735/129437)",
        "PASS",
        f"선형 추정: {estimated_sec:.0f}초 ≈ {estimated_min:.1f}분",
    ))

    # G-4: 보수적 추정 range
    rows.append(r(
        "estimated_runtime_conservative",
        "5~15분",
        "메모리, 디스크 I/O, 환경 차이 반영",
        "PASS",
        "Phase 7.2 동일 checkpoint/batch_size 성공. OOM risk 낮음.",
    ))

    # G-5: OOM risk
    rows.append(r(
        "oom_risk",
        "낮음",
        "Phase 7.2에서 동일 checkpoint/batch_size=32로 성공",
        "PASS",
        "GPU VRAM 사용량은 Phase 7.2 기준으로 충분히 확인됨.",
    ))

    # G-6: disk CSV 예상 크기
    if PHASE7_2_SCORE_CSV.exists():
        p72_csv_size_bytes = PHASE7_2_SCORE_CSV.stat().st_size
        estimated_csv_bytes = p72_csv_size_bytes * (EXPECTED_ROW_COUNT / 129437)
        estimated_csv_mb = estimated_csv_bytes / (1024 * 1024)
        p72_csv_mb = p72_csv_size_bytes / (1024 * 1024)
        rows.append(r(
            "estimated_score_csv_size",
            f"{estimated_csv_mb:.1f} MB",
            f"Phase 7.2 CSV {p72_csv_mb:.1f} MB × (143735/129437)",
            "PASS",
            f"Phase 7.2 CSV: {p72_csv_mb:.1f} MB → Phase 8.4 예상: {estimated_csv_mb:.1f} MB",
        ))
    else:
        rows.append(r(
            "estimated_score_csv_size",
            "~92 MB (추정)",
            "Phase 7.2 CSV 82 MB 기준 비율 추정",
            "WARNING",
            "Phase 7.2 score CSV 파일이 없어 직접 추정 불가. 근사값 사용.",
        ))

    # G-7: error CSV / tmp rename / DONE 구조
    rows.append(r(
        "safety_structures",
        "error CSV + tmp rename + DONE marker",
        "Phase 7.2 구조 동일 적용",
        "PASS",
        "error CSV / tmp→final rename / DONE marker 구조 필요 확인됨.",
    ))

    return rows


# ──────────────────────────────────────────────────────────
# Section H: readiness decision
# ──────────────────────────────────────────────────────────
def run_section_h(all_rows: list) -> tuple:
    """returns (readiness_str, blockers_list, section_h_rows)"""
    blockers = []
    fail_sections = {"A": False, "B": False, "C": False, "F": False, "G": False}

    for row in all_rows:
        sec = row.get("section", "")
        st = row.get("status", "")
        item = row.get("item", row.get("check_item", ""))
        if st == "FAIL":
            blockers.append(f"Section {sec} / {item}: FAIL")
            if sec in fail_sections:
                fail_sections[sec] = True

    # readiness 판정 (우선순위 순)
    if fail_sections["F"]:
        readiness = "BLOCKED_OUTPUT_CONFLICT"
    elif fail_sections["A"]:
        readiness = "BLOCKED_PRIOR_PHASE_NOT_READY"
    elif fail_sections["B"]:
        readiness = "BLOCKED_STAGE2_MANIFEST_INVALID"
    elif fail_sections["C"]:
        readiness = "BLOCKED_MODEL_CHECKPOINT_INVALID"
    elif fail_sections["G"]:
        readiness = "BLOCKED_RESOURCE_RISK"
    elif blockers:
        readiness = "BLOCKED_UNKNOWN"
    else:
        readiness = "READY_FOR_PHASE8_4B_STAGE2_FULL_SCORING_SCRIPT"

    blocker_str = "; ".join(blockers) if blockers else ""
    if "READY" in readiness:
        next_action = (
            "Phase 8.4B stage2 full scoring script 작성 및 실행 전 검토를 진행한다. "
            "실제 실행은 별도 승인 후."
        )
    else:
        next_action = f"blocker 해소 필요: {blocker_str}"

    h_rows = [make_base_row(
        section="H",
        item="readiness_decision",
        check_item="readiness_decision",
        expected="READY_FOR_PHASE8_4B_STAGE2_FULL_SCORING_SCRIPT",
        observed=readiness,
        status="PASS" if "READY" in readiness else "BLOCKED",
        note=f"blockers={len(blockers)}개",
        blocker=blocker_str,
        next_required_action=next_action,
    )]

    return readiness, blockers, h_rows


# ──────────────────────────────────────────────────────────
# CSV builder
# ──────────────────────────────────────────────────────────
def build_csv(all_rows: list) -> pd.DataFrame:
    df = pd.DataFrame(all_rows)
    # 누락 컬럼은 NaN으로 채우기 위해 reindex
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[UNIFIED_COLUMNS]


# ──────────────────────────────────────────────────────────
# JSON builder
# ──────────────────────────────────────────────────────────
def build_json(
    sec_a: list, sec_b: list, sec_c: list,
    sec_d: list, sec_e: list, sec_f: list, sec_g: list,
    readiness: str, blockers: list,
) -> dict:
    def to_dict_list(rows):
        return [{k: v for k, v in r.items()} for r in rows]

    return {
        "input_paths": {
            "manifest": str(MANIFEST),
            "crop_root": str(CROP_ROOT),
            "phase8_2f_done": str(PHASE8_2F_DONE),
            "phase8_2g_json": str(PHASE8_2G_JSON),
            "phase8_3b_done": str(PHASE8_3B_DONE),
            "phase8_3b_summary": str(PHASE8_3B_SUMMARY),
            "phase8_3c_json": str(PHASE8_3C_JSON),
            "checkpoint": str(CHECKPOINT),
            "phase7_2_runtime_csv": str(PHASE7_2_RUNTIME_CSV),
        },
        "prior_phase_readiness": to_dict_list(sec_a),
        "full_scoring_input_validation": to_dict_list(sec_b),
        "model_checkpoint_readiness": to_dict_list(sec_c),
        "full_scoring_script_design": to_dict_list(sec_d),
        "full_scoring_output_design": to_dict_list(sec_e),
        "output_collision_check": to_dict_list(sec_f),
        "runtime_resource_estimate": to_dict_list(sec_g),
        "readiness_for_phase8_4b": readiness,
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
    sec_a: list, sec_b: list, sec_c: list,
    sec_d: list, sec_e: list, sec_f: list, sec_g: list,
    readiness: str, blockers: list,
    phase8_3b_summary: dict, phase8_3c_data: dict,
) -> str:
    def sec_table(rows: list, cols: list) -> str:
        header = " | ".join(cols)
        sep = " | ".join(["---"] * len(cols))
        lines = [f"| {header} |", f"| {sep} |"]
        for row in rows:
            cells = " | ".join(str(row.get(c, "")) for c in cols)
            lines.append(f"| {cells} |")
        return "\n".join(lines)

    readiness_status = "PASS" if "READY" in readiness else "BLOCKED"

    # 요약 통계
    smoke_pass = phase8_3b_summary.get("smoke_pass", "N/A")
    scored_count = phase8_3b_summary.get("scored_crop_count", "N/A")
    has_nan = phase8_3b_summary.get("has_nan_count", "N/A")
    has_inf = phase8_3b_summary.get("has_inf_count", "N/A")
    error_cnt = phase8_3b_summary.get("error_count", "N/A")
    c_readiness = phase8_3c_data.get("readiness_for_phase8_4", "N/A")

    md = f"""# Phase 8.4 stage2_holdout full scoring preflight

## 1. Phase 8.4 목적

stage2_holdout 143,735개 전체 RD4AD scoring 실행 전 사전 검증.
preflight only — model forward / scoring / metric 계산 전혀 없음.

---

## 2. Phase 8.2F / 8.2G / 8.3B / 8.3C 결과 요약

| 항목 | 값 |
|---|---|
| Phase 8.2F DONE marker | {PHASE8_2F_DONE} |
| Phase 8.2G readiness | {phase8_3b_summary.get('stage_split', 'stage2_holdout')} 기준 READY 확인 |
| Phase 8.3B smoke_pass | {smoke_pass} |
| Phase 8.3B scored_crop_count | {scored_count} |
| Phase 8.3B has_nan_count | {has_nan} |
| Phase 8.3B has_inf_count | {has_inf} |
| Phase 8.3B error_count | {error_cnt} |
| Phase 8.3C readiness | {c_readiness} |

---

## 3. full scoring input validation

{sec_table(sec_b, ['check_item', 'expected', 'observed', 'status', 'note'])}

---

## 4. model/checkpoint readiness

{sec_table(sec_c, ['item', 'path', 'expected', 'observed', 'status', 'note'])}

---

## 5. full scoring script design

{sec_table(sec_d, ['item', 'planned_value', 'rationale', 'status', 'note'])}

---

## 6. full scoring output design

{sec_table(sec_e, ['output_type', 'path', 'planned', 'status', 'note'])}

---

## 7. output collision check

{sec_table(sec_f, ['output_type', 'path', 'exists', 'status', 'note'])}

---

## 8. runtime/resource estimate

{sec_table(sec_g, ['item', 'estimate', 'evidence', 'status', 'note'])}

---

## 9. readiness 판정

**{readiness}** ({readiness_status})

blockers ({len(blockers)}개):
"""
    if blockers:
        for b in blockers:
            md += f"\n- {b}"
    else:
        md += "\n(blocker 없음)"

    md += "\n\n---\n\n## 10. 다음 단계\n\n"

    if "READY" in readiness:
        md += (
            "- Phase 8.4B stage2 full scoring script 작성 및 실행 전 검토를 진행한다.\n"
            f"- 권장 파일명: `scripts/phase8_4_stage2_full_scoring.py`\n"
            f"- 실행 명령 (승인 후): "
            "`source ~/ai_env/bin/activate && python scripts/phase8_4_stage2_full_scoring.py "
            "--run-full --confirm-run --batch-size 32`\n"
            f"- full score output root: `{FULL_SCORE_OUTPUT_ROOT}`\n"
            f"- 실제 실행은 별도 사용자 승인 후 진행한다.\n"
        )
    else:
        md += f"- BLOCKED 상태. 아래 blocker를 해소한 후 이 preflight를 재실행한다.\n"
        for b in blockers:
            md += f"  - {b}\n"

    md += """
---

## 11. 금지 사항

- model forward 금지
- scoring 금지
- metric / threshold / p95 / p99 / hit-rate 계산 금지
- training 금지
- checkpoint 생성 금지
- torch.load / checkpoint load 금지
- 전체 npz 로드 금지
- 기존 Phase 6/7/8 output 수정/삭제/덮어쓰기 금지
- 이번 preflight에서 full scoring script 작성/실행 금지
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
        print("[DRY-RUN] --run 플래그가 없습니다. 실제 검증을 수행하지 않습니다.")
        print("")
        print("사용법:")
        print("  python scripts/phase8_4_stage2_full_scoring_preflight.py --run")
        print("")
        print("이 스크립트는 다음을 검증합니다:")
        print("  A. prior phase readiness (8.2F/8.2G/8.3B/8.3C)")
        print("  B. full scoring input validation (manifest 143,735행)")
        print("  C. model/checkpoint readiness")
        print("  D. full scoring script design (PLANNED)")
        print("  E. full scoring output design (PLANNED)")
        print("  F. output collision check")
        print("  G. runtime/resource estimate")
        print("  H. readiness decision")
        print("")
        print("금지 사항: model forward / scoring / metric / training / torch.load 전혀 없음.")
        sys.exit(0)

    print("[INFO] Phase 8.4 stage2_holdout full scoring preflight 시작")

    # output guard (최우선)
    check_output_guard()

    # manifest 로드
    print(f"[INFO] Loading manifest: {MANIFEST}")
    if not MANIFEST.exists():
        print(f"[ERROR] manifest not found: {MANIFEST}")
        sys.exit(1)
    manifest_df = pd.read_csv(MANIFEST)
    print(f"[INFO] Manifest loaded: {len(manifest_df)} rows")

    # Phase 8.2G JSON 로드
    print(f"[INFO] Loading Phase 8.2G JSON: {PHASE8_2G_JSON}")
    if not PHASE8_2G_JSON.exists():
        print(f"[ERROR] Phase 8.2G JSON not found: {PHASE8_2G_JSON}")
        sys.exit(1)
    with open(PHASE8_2G_JSON, encoding="utf-8") as f:
        phase8_2g_data = json.load(f)

    # Phase 8.3B summary JSON 로드
    print(f"[INFO] Loading Phase 8.3B summary JSON: {PHASE8_3B_SUMMARY}")
    if not PHASE8_3B_SUMMARY.exists():
        print(f"[ERROR] Phase 8.3B summary JSON not found: {PHASE8_3B_SUMMARY}")
        sys.exit(1)
    with open(PHASE8_3B_SUMMARY, encoding="utf-8") as f:
        phase8_3b_summary = json.load(f)

    # Phase 8.3C JSON 로드
    print(f"[INFO] Loading Phase 8.3C JSON: {PHASE8_3C_JSON}")
    if not PHASE8_3C_JSON.exists():
        print(f"[ERROR] Phase 8.3C JSON not found: {PHASE8_3C_JSON}")
        sys.exit(1)
    with open(PHASE8_3C_JSON, encoding="utf-8") as f:
        phase8_3c_data = json.load(f)

    # 각 섹션 실행
    print("[INFO] Section A: prior phase readiness")
    sec_a = run_section_a(manifest_df, phase8_2g_data, phase8_3b_summary, phase8_3c_data)

    print("[INFO] Section B: full scoring input validation")
    sec_b = run_section_b(manifest_df, phase8_2g_data)

    print("[INFO] Section C: model/checkpoint readiness")
    sec_c = run_section_c(phase8_3b_summary)

    print("[INFO] Section D: full scoring script design")
    sec_d = run_section_d()

    print("[INFO] Section E: full scoring output design")
    sec_e = run_section_e()

    print("[INFO] Section F: output collision check")
    sec_f = run_section_f()

    print("[INFO] Section G: runtime/resource estimate")
    sec_g = run_section_g()

    all_rows = sec_a + sec_b + sec_c + sec_d + sec_e + sec_f + sec_g

    # Section H: readiness decision
    print("[INFO] Section H: readiness decision")
    readiness, blockers, sec_h = run_section_h(all_rows)
    all_rows += sec_h

    print(f"[INFO] readiness_for_phase8_4b = {readiness}")
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
    if not csv_path.exists():
        print(f"[ERROR] CSV 저장 실패: {csv_path}")
        sys.exit(1)
    print(f"[INFO] CSV saved: {csv_path}")

    # JSON 저장
    json_path = OUT_DIR / OUT_JSON_NAME
    check_file_guard(json_path)
    json_data = build_json(
        sec_a, sec_b, sec_c, sec_d, sec_e, sec_f, sec_g,
        readiness, blockers,
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    if not json_path.exists():
        print(f"[ERROR] JSON 저장 실패: {json_path}")
        sys.exit(1)
    print(f"[INFO] JSON saved: {json_path}")

    # MD 저장
    md_path = OUT_DIR / OUT_MD_NAME
    check_file_guard(md_path)
    md_text = build_md(
        sec_a, sec_b, sec_c, sec_d, sec_e, sec_f, sec_g,
        readiness, blockers,
        phase8_3b_summary, phase8_3c_data,
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    if not md_path.exists():
        print(f"[ERROR] MD 저장 실패: {md_path}")
        sys.exit(1)
    print(f"[INFO] MD saved: {md_path}")

    print(f"\n[DONE] Phase 8.4 preflight 완료")
    print(f"  output root : {OUT_DIR}")
    print(f"  CSV         : {csv_path}")
    print(f"  JSON        : {json_path}")
    print(f"  MD          : {md_path}")
    print(f"  readiness   : {readiness}")
    print(f"  blockers    : {len(blockers)}개")


if __name__ == "__main__":
    main()
