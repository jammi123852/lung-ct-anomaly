#!/usr/bin/env python3
"""
Phase 5.93 Expanded Hard Negative Candidate Sample Dry-Run Script

Purpose:
    Using the confirmed normal split source (CAND01) and normal score CSV source (SRC01b),
    perform a read-only sample candidate extraction dry-run on a small number of normal patients.

Safety Constraints:
    - --dry-run flag is REQUIRED. Without it, this script exits immediately.
    - No model forward, no score recalculation, no threshold finalization.
    - No crop manifest, training dataset, or split CSV modification.
    - No CT/ROI/mask npy loading.
    - No stage2_holdout or v2 access.
    - SRC02 (s6a_stage1_train_val_split.csv) is forbidden.
    - --split train is blocked: SRC01b score CSV contains val+test only.
    - candidate CSV/JSON/MD are review-only, NOT a hard negative manifest.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd


# === Project-relative paths ===
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

PHASE592B_JSON_DEFAULT = (
    _PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase5_92b_split_source_disambiguation_preflight_v1"
    / "phase5_92b_split_source_disambiguation_preflight_v1.json"
)
PHASE592B_CSV_DEFAULT = (
    _PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase5_92b_split_source_disambiguation_preflight_v1"
    / "phase5_92b_split_source_disambiguation_preflight_v1.csv"
)
PHASE591_JSON_DEFAULT = (
    _PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase5_91_expanded_hard_negative_candidate_pool_preflight_plan_v1"
    / "phase5_91_expanded_hard_negative_candidate_pool_preflight_plan_v1.json"
)
CAND01_SPLIT_CSV_DEFAULT = (
    _PROJECT_ROOT / "data/normal_training_ready/manifests/train_val_test_split.csv"
)
OUTPUT_ROOT = (
    _PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase5_93_expanded_candidate_sample_dry_run_v1"
)
OUTPUT_CSV = OUTPUT_ROOT / "phase5_93_expanded_candidate_sample_dry_run_v1.csv"
OUTPUT_JSON = OUTPUT_ROOT / "phase5_93_expanded_candidate_sample_dry_run_v1.json"
OUTPUT_MD = OUTPUT_ROOT / "phase5_93_expanded_candidate_sample_dry_run_report_v1.md"

SCORE_CSV_REQUIRED_COLUMNS = [
    "patient_id", "local_z", "y0", "x0", "y1", "x1",
    "patch_size", "patch_stride", "padim_score", "roi_0_0_patch_ratio",
]

SPLIT_CSV_REQUIRED_COLUMNS = ["patient_id", "split"]

# SRC02 forbidden indicator
SRC02_FILENAME = "s6a_stage1_train_val_split.csv"

# Segment-level forbidden keywords.
# "second-stage-lesion-refiner-v1" is whitelisted as the project folder name.
FORBIDDEN_SEGMENTS = frozenset({
    "NSCLC", "MSD_Lung", "LUNG1", "stage2_holdout", "v2", "v2v2",
    "crops_lesion", "lesion_by_patient", "NSCLC_MSD", "MSD_lung",
})

PROJECT_FOLDER_WHITELIST = frozenset({
    "second-stage-lesion-refiner-v1",
})


def guard_forbidden_path(path_str: str, context: str = "") -> None:
    """Segment-based forbidden path guard.
    Avoids false positives from the project folder name 'second-stage-lesion-refiner-v1'.
    """
    p = pathlib.Path(path_str)

    if p.name == SRC02_FILENAME or "s6a_stage1_train_val_split" in p.name:
        sys.exit(
            f"[GUARD] SRC02 forbidden path detected in {context}: {path_str}\n"
            f"        SRC02 is LESION_SPLIT_NOT_COMPATIBLE. Use is forbidden."
        )

    for seg in p.parts:
        if seg in PROJECT_FOLDER_WHITELIST:
            continue
        if seg in FORBIDDEN_SEGMENTS:
            sys.exit(
                f"[GUARD] Forbidden path segment '{seg}' detected in {context}: {path_str}\n"
                f"        This path may reference lesion patients, holdout, or v2 data."
            )


def validate_score_csv_columns(df: pd.DataFrame, csv_path: str) -> None:
    missing = [c for c in SCORE_CSV_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(
            f"[VALIDATION] Score CSV missing required columns: {missing}\n"
            f"             File: {csv_path}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5.93: Expanded Hard Negative Candidate Sample Dry-Run.\n"
            "--dry-run is REQUIRED. Script exits immediately without it.\n"
            "--split train is blocked: SRC01b score CSV contains val+test only."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="REQUIRED. Enables dry-run mode (read-only review-only candidate extraction).",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help=(
            "Split to sample from. Only 'val' or 'test' are allowed. "
            "'train' is blocked: SRC01b score CSV contains val+test only. "
            "Default: val."
        ),
    )
    parser.add_argument(
        "--sample-patients",
        type=int,
        default=3,
        metavar="N",
        help="Number of patients to sample. Max 3. Default: 3.",
    )
    parser.add_argument(
        "--phase592b-json",
        type=str,
        default=str(PHASE592B_JSON_DEFAULT),
        help="Path to Phase 5.92b disambiguation JSON (used to extract SRC01b score root).",
    )
    parser.add_argument(
        "--phase591-json",
        type=str,
        default=str(PHASE591_JSON_DEFAULT),
        help="Path to Phase 5.91 plan JSON.",
    )
    parser.add_argument(
        "--cand01-split-csv",
        type=str,
        default=str(CAND01_SPLIT_CSV_DEFAULT),
        help="Path to CAND01 normal split CSV (train_val_test_split.csv).",
    )

    args = parser.parse_args()

    # ==================================================
    # GUARD 1: --dry-run 필수
    # ==================================================
    if not args.dry_run:
        print("[GUARD] --dry-run flag is required. This script will not run without it.")
        print(
            "        Usage: python phase5_93_expanded_hard_negative_candidate_sample_dry_run.py"
            " --dry-run [--split val|test] [--sample-patients N]"
        )
        sys.exit(1)

    # ==================================================
    # GUARD 2: sample-patients 범위 제한
    # ==================================================
    if args.sample_patients > 3:
        sys.exit(
            f"[GUARD] --sample-patients cannot exceed 3. Got: {args.sample_patients}\n"
            f"        This is a dry-run script for small-scale review only."
        )
    if args.sample_patients < 1:
        sys.exit(
            f"[GUARD] --sample-patients must be at least 1. Got: {args.sample_patients}"
        )

    # ==================================================
    # test split warning 기록
    # ==================================================
    test_split_warnings = []
    if args.split == "test":
        test_split_warnings = [
            "test split is allowed only for read-only sample dry-run review",
            "no threshold tuning",
            "no hard negative final selection",
            "no performance conclusion",
        ]
        for w in test_split_warnings:
            print(f"[WARNING] {w}")

    print(f"\n[INFO] Phase 5.93 dry-run starting.")
    print(f"[INFO] split={args.split}, sample_patients={args.sample_patients}")

    # ==================================================
    # STEP 1: Phase 5.92b JSON 읽기 → SRC01b score root 추출
    # ==================================================
    phase592b_json_path = pathlib.Path(args.phase592b_json)
    if not phase592b_json_path.exists():
        sys.exit(f"[VALIDATION] Phase 5.92b JSON not found: {phase592b_json_path}")

    with open(phase592b_json_path, "r", encoding="utf-8") as f:
        phase592b_data = json.load(f)

    src01b_info = phase592b_data.get("src01b_patient_check", {})
    score_root_str = src01b_info.get("source_path", "")
    if not score_root_str:
        sys.exit(
            "[VALIDATION] Cannot find src01b_patient_check.source_path in Phase 5.92b JSON.\n"
            "             Do not guess the path. Check the Phase 5.92b JSON structure."
        )

    print(f"[INFO] SRC01b score root from Phase 5.92b JSON: {score_root_str}")

    # ==================================================
    # GUARD 3: score root forbidden path check
    # ==================================================
    guard_forbidden_path(score_root_str, "SRC01b score root")

    score_root = pathlib.Path(score_root_str)
    if not score_root.exists():
        sys.exit(f"[VALIDATION] SRC01b score root does not exist: {score_root}")

    # ==================================================
    # STEP 2: Phase 5.92b CSV 존재 확인
    # ==================================================
    phase592b_csv_path = pathlib.Path(PHASE592B_CSV_DEFAULT)
    if not phase592b_csv_path.exists():
        sys.exit(f"[VALIDATION] Phase 5.92b CSV not found: {phase592b_csv_path}")
    print(f"[INFO] Phase 5.92b CSV exists: {phase592b_csv_path}")

    # ==================================================
    # STEP 3: score CSV 파일 수 확인
    # ==================================================
    score_csv_files = sorted(score_root.glob("*.csv"))
    n_score_files = len(score_csv_files)
    print(f"[INFO] Score CSV files found: {n_score_files}")

    if n_score_files == 0:
        sys.exit(f"[VALIDATION] No score CSV files found in: {score_root}")

    expected_n = src01b_info.get("n_score_files", 72)
    if n_score_files != expected_n:
        print(
            f"[WARNING] Expected {expected_n} score CSV files (Phase 5.92b JSON), "
            f"found {n_score_files}. Proceeding with actual count."
        )

    # ==================================================
    # STEP 4: CAND01 split CSV 검증
    # ==================================================
    cand01_csv_path = pathlib.Path(args.cand01_split_csv)
    guard_forbidden_path(str(cand01_csv_path), "CAND01 split CSV")

    if not cand01_csv_path.exists():
        sys.exit(f"[VALIDATION] CAND01 split CSV not found: {cand01_csv_path}")

    split_df = pd.read_csv(cand01_csv_path)
    n_split_rows = len(split_df)
    print(f"[INFO] CAND01 split CSV rows: {n_split_rows}")

    missing_split_cols = [c for c in SPLIT_CSV_REQUIRED_COLUMNS if c not in split_df.columns]
    if missing_split_cols:
        sys.exit(
            f"[VALIDATION] CAND01 split CSV missing required columns: {missing_split_cols}"
        )

    split_values = split_df["split"].value_counts().to_dict()
    print(f"[INFO] CAND01 split values: {split_values}")

    # ==================================================
    # STEP 5: SRC01b patient_id와 CAND01 join 가능성 확인
    # ==================================================
    score_patient_ids = {f.stem for f in score_csv_files}
    cand01_patient_ids = set(split_df["patient_id"].tolist())
    joinable = score_patient_ids & cand01_patient_ids
    print(f"[INFO] Joinable patient_ids (SRC01b ∩ CAND01): {len(joinable)}")

    if len(joinable) == 0:
        sys.exit(
            "[VALIDATION] No joinable patient_ids between SRC01b score CSVs and CAND01 split CSV.\n"
            "             Check patient_id naming consistency."
        )

    # ==================================================
    # STEP 6: 선택된 split에서 sample patient 선정
    # ==================================================
    target_df = split_df[split_df["split"] == args.split]
    target_patients = target_df["patient_id"].tolist()
    valid_patients = [p for p in target_patients if p in score_patient_ids]

    print(f"[INFO] Target split='{args.split}': {len(target_patients)} patients in CAND01")
    print(f"[INFO] Valid patients (with matching score CSV): {len(valid_patients)}")

    if len(valid_patients) == 0:
        sys.exit(
            f"[VALIDATION] No valid patients found for split='{args.split}' with matching score CSV.\n"
            f"             SRC01b contains val+test score CSV only. train is blocked."
        )

    sample_patients = valid_patients[: args.sample_patients]
    print(f"[INFO] Sampled {len(sample_patients)} patients: {sample_patients}")

    # ==================================================
    # STEP 7: score CSV 컬럼 검증 (첫 번째 sample patient로 probe)
    # ==================================================
    first_score_csv = score_root / f"{sample_patients[0]}.csv"
    if not first_score_csv.exists():
        sys.exit(
            f"[VALIDATION] Score CSV not found for first sample patient: {first_score_csv}"
        )

    probe_df = pd.read_csv(first_score_csv)
    validate_score_csv_columns(probe_df, str(first_score_csv))
    print(f"[INFO] Score CSV column validation passed: {first_score_csv.name}")

    # ==================================================
    # STEP 8: candidate extraction (read-only, sample-local)
    # ==================================================
    candidate_rows = []

    for patient_id in sample_patients:
        score_csv_path = score_root / f"{patient_id}.csv"
        guard_forbidden_path(str(score_csv_path), f"score CSV for {patient_id}")

        if not score_csv_path.exists():
            print(f"[WARNING] Score CSV not found for {patient_id}, skipping.")
            continue

        df = pd.read_csv(score_csv_path)
        validate_score_csv_columns(df, str(score_csv_path))

        # 최소 ROI 포함 조건: roi_0_0_patch_ratio > 0
        # roi_0_0_patch_ratio는 pure_lung_patch_ratio와 동일하다고 단정하지 않음
        df_roi = df[df["roi_0_0_patch_ratio"] > 0].copy()
        if len(df_roi) == 0:
            print(
                f"[WARNING] No patches with roi_0_0_patch_ratio > 0 for {patient_id}. "
                f"Using all patches."
            )
            df_roi = df.copy()

        # sample-local p99 (not global threshold)
        p99_val = np.percentile(df_roi["padim_score"].values, 99)
        df_cand = df_roi[df_roi["padim_score"] >= p99_val].copy()

        if len(df_cand) == 0:
            df_cand = df_roi.nlargest(1, "padim_score").copy()

        df_cand = df_cand.sort_values("padim_score", ascending=False).reset_index(drop=True)
        df_cand["candidate_rank_in_patient"] = df_cand.index + 1

        df_cand["section"] = "phase5_93_sample_dry_run"
        df_cand["split"] = args.split
        df_cand["candidate_score_percentile_mode"] = "sample_local_not_global"
        df_cand["candidate_selection_rule"] = (
            "padim_score_p99_sample_local | roi_0_0_patch_ratio_gt_0"
        )
        df_cand["candidate_status"] = (
            "sample_review_candidate_only"
            "|not_training_ready"
            "|not_hard_negative_final"
            "|requires_visual_review"
        )
        df_cand["candidate_class_hint"] = "unknown_requires_visual_review"
        df_cand["risk_note"] = (
            "roi_0_0_patch_ratio is NOT confirmed to be pure_lung_patch_ratio. "
            "Used as minimum ROI inclusion filter only. "
            "candidate_class_hint is unknown; visual review required before any label assignment."
        )
        df_cand["source_score_csv"] = str(score_csv_path)
        df_cand["source_split_csv"] = str(cand01_csv_path)
        df_cand["limitation"] = (
            "sample_local_not_global"
            " | dry_run_only"
            " | not_hard_negative_final"
            " | threshold_not_finalized"
            " | requires_visual_review"
        )

        output_cols = [
            "section", "patient_id", "split", "local_z",
            "y0", "x0", "y1", "x1", "patch_size", "patch_stride",
            "padim_score", "roi_0_0_patch_ratio",
            "candidate_rank_in_patient", "candidate_score_percentile_mode",
            "candidate_selection_rule", "candidate_status", "candidate_class_hint",
            "risk_note", "source_score_csv", "source_split_csv", "limitation",
        ]
        candidate_rows.append(df_cand[output_cols])

    if not candidate_rows:
        sys.exit("[VALIDATION] No candidate rows extracted from sample patients.")

    candidate_df = pd.concat(candidate_rows, ignore_index=True)
    print(f"[INFO] Total candidate rows (review-only): {len(candidate_df)}")

    # ==================================================
    # STEP 9: output guard (output root 생성 전 최종 확인)
    # ==================================================
    if OUTPUT_ROOT.exists():
        sys.exit(
            f"[GUARD] Output root already exists: {OUTPUT_ROOT}\n"
            f"        Remove it manually before re-running."
        )
    for out_file in [OUTPUT_CSV, OUTPUT_JSON, OUTPUT_MD]:
        if out_file.exists():
            sys.exit(
                f"[GUARD] Output file already exists: {out_file}\n"
                f"        Remove it manually before re-running."
            )

    # ==================================================
    # STEP 10: output root 생성 (모든 검증 완료 이후)
    # ==================================================
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] Output root created: {OUTPUT_ROOT}")

    # ==================================================
    # STEP 11: CSV 저장 (직전 재확인)
    # ==================================================
    if OUTPUT_CSV.exists():
        sys.exit(f"[GUARD] Output CSV already appeared before write: {OUTPUT_CSV}")
    candidate_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Candidate CSV saved: {OUTPUT_CSV}")

    # ==================================================
    # STEP 12: JSON 저장
    # ==================================================
    phase591_json_path = pathlib.Path(args.phase591_json)
    phase591_exists = phase591_json_path.exists()

    src02_path = (
        phase592b_data.get("src02_identity_check", {}).get("source_path", "N/A")
    )

    json_output = {
        "phase": "phase5_93",
        "dry_run_only": True,
        "sample_local_not_global": True,
        "no_candidate_finalization": True,
        "no_true_hard_negative_manifest": True,
        "no_training_dataset_modification": True,
        "no_crop_manifest_modification": True,
        "no_model_forward": True,
        "no_score_recalculation": True,
        "threshold_not_finalized": True,
        "lesion_conclusion_forbidden": True,
        "hard_negative_not_finalized": True,
        "stage2_holdout_unused": True,
        "v2_unused": True,
        "src02_forbidden": True,
        "train_split_blocked": True,
        "src01b_val_test_only": True,
        "run_config": {
            "split": args.split,
            "sample_patients": args.sample_patients,
            "sampled_patient_ids": sample_patients,
            "test_split_warnings": (
                test_split_warnings if args.split == "test" else []
            ),
        },
        "input_paths": {
            "phase592b_json": str(phase592b_json_path),
            "phase592b_csv": str(phase592b_csv_path),
            "phase591_json": str(phase591_json_path),
            "cand01_split_csv": str(cand01_csv_path),
            "src01b_score_root": score_root_str,
        },
        "input_validation": {
            "phase592b_json_exists": True,
            "phase592b_csv_exists": True,
            "phase591_json_exists": phase591_exists,
            "score_root_exists": True,
            "n_score_files": n_score_files,
            "cand01_csv_exists": True,
            "cand01_row_count": n_split_rows,
            "cand01_split_values": split_values,
            "joinable_patient_count": len(joinable),
            "valid_patients_in_split": len(valid_patients),
            "sampled_patient_count": len(sample_patients),
            "score_csv_required_columns": SCORE_CSV_REQUIRED_COLUMNS,
            "score_csv_column_validation": "passed",
        },
        "candidate_extraction": {
            "method": "padim_score p99 sample-local | roi_0_0_patch_ratio > 0",
            "total_candidate_rows": len(candidate_df),
            "roi_note": (
                "roi_0_0_patch_ratio is NOT confirmed to be pure_lung_patch_ratio. "
                "Used as minimum ROI inclusion filter only."
            ),
            "percentile_note": (
                "p99 is computed per-patient within the sample, not globally. "
                "This is sample_local_not_global."
            ),
        },
        "output_paths": {
            "csv": str(OUTPUT_CSV),
            "json": str(OUTPUT_JSON),
            "md": str(OUTPUT_MD),
        },
        "notes": {
            "src01b_val_test_only": (
                "SRC01b contains val+test score CSV only (72 files). "
                "train split patients (290) have no matching SRC01b score CSV. "
                "train is blocked for Phase 5.93 sample dry-run."
            ),
            "train_split_blocked": (
                "--split train is blocked at argparse choices level. "
                "Choices are restricted to ['val', 'test'] only. "
                "No automatic fallback is implemented."
            ),
            "candidate_class_hint": (
                "unknown_requires_visual_review. "
                "No CT/PNG loaded. No structural label confirmed in this phase."
            ),
            "candidate_status": (
                "sample_review_candidate_only|not_training_ready"
                "|not_hard_negative_final|requires_visual_review"
            ),
            "roi_0_0_patch_ratio_note": (
                "roi_0_0_patch_ratio is used as a proxy for ROI inclusion. "
                "Not confirmed to be identical to pure_lung_patch_ratio."
            ),
            "dry_run_only": (
                "This output is for review only. Not a hard negative manifest."
            ),
            "sample_local_not_global": (
                "Percentile thresholds are computed per-patient within the sample. "
                "Cannot be used as global thresholds."
            ),
            "src02_forbidden": f"SRC02 path ({src02_path}) is LESION_SPLIT_NOT_COMPATIBLE. Forbidden.",
        },
        "limitations": [
            "Sample-local p99 is not a global threshold — cannot finalize threshold.",
            "roi_0_0_patch_ratio may differ from pure_lung_patch_ratio — used only as minimum ROI filter.",
            "candidate_class_hint is unknown — CT/PNG visual review required.",
            "Only val or test split available (SRC01b score CSV does not include train patients).",
            "stage2_holdout and v2 remain sealed.",
            "Hard negative finalization requires visual review and explicit user approval.",
        ],
        "future_command_example": (
            "python scripts/phase5_93_expanded_hard_negative_candidate_sample_dry_run.py"
            " --dry-run --split val --sample-patients 3"
        ),
    }

    if OUTPUT_JSON.exists():
        sys.exit(f"[GUARD] Output JSON already appeared before write: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"[INFO] JSON saved: {OUTPUT_JSON}")

    # ==================================================
    # STEP 13: MD report 저장
    # ==================================================
    src02_reason = (
        "NSCLC/MSD_Lung lesion patient 기반 stage1 학습용 split. "
        "normal* 환자 없음. LUNG1-* patient_id 패턴. "
        "n_positive/n_hard_negative 컬럼 존재. "
        "Phase 5.92b에서 LESION_SPLIT_NOT_COMPATIBLE 확정."
    )

    md_lines = [
        "# Phase 5.93 Expanded Hard Negative Candidate Sample Dry-Run Report",
        "",
        "## Phase 5.93 목적",
        "",
        "Phase 5.92b에서 확정된 normal split source(CAND01)와",
        "normal score CSV source(SRC01b)를 사용하여,",
        "소수 normal patient 대상으로 read-only candidate extraction dry-run을 수행한다.",
        "",
        "이 단계의 output은 **review-only 후보 table**이며, hard negative manifest가 아니다.",
        "",
        "## 사용한 Normal Split Source: CAND01",
        "",
        f"- 경로: `{str(cand01_csv_path)}`",
        f"- row 수: {n_split_rows}",
        f"- split 구성: {split_values}",
        "- Phase 5.92b에서 NORMAL_SPLIT_COMPATIBLE으로 확인됨",
        "- next_session_handoff.md에서 source of truth로 명시됨",
        "",
        "## SRC02 사용 금지 이유",
        "",
        f"- 경로: `{src02_path}`",
        "- 판정: LESION_SPLIT_NOT_COMPATIBLE",
        f"- 이유: {src02_reason}",
        "- Phase 5.93에서 참조 불가. argparse guard로 차단됨.",
        "",
        "## train split 차단 이유",
        "",
        "- SRC01b normal_by_patient score CSV는 val+test 72명만 포함.",
        "- train 환자(290명)에 대한 SRC01b score CSV 없음.",
        "- `--split train`을 허용하면 score join 결과가 0명이 됨.",
        "- argparse choices를 `['val', 'test']`로 제한하여 train을 차단.",
        "- **자동 fallback 없음.** train 요청 시 argparse 단계에서 즉시 오류.",
        "",
        "## 실행 설정",
        "",
        f"- split: `{args.split}`",
        f"- sample_patients: {args.sample_patients}",
        f"- sampled patient IDs: {sample_patients}",
    ]

    if args.split == "test":
        md_lines += [
            "",
            "## test split 사용 경고",
            "",
        ]
        for w in test_split_warnings:
            md_lines.append(f"- {w}")

    md_lines += [
        "",
        "## Sample Patient 목록",
        "",
    ]
    for i, pid in enumerate(sample_patients, 1):
        md_lines.append(f"{i}. `{pid}`")

    md_lines += [
        "",
        "## Candidate Extraction 결과",
        "",
        f"- 총 candidate rows (review-only): {len(candidate_df)}",
        "- 방법: padim_score sample-local p99 + roi_0_0_patch_ratio > 0",
        "",
        "**이 결과는 review-only candidate table이며, hard negative manifest가 아님.**",
        "",
        "## Candidate Status",
        "",
        "모든 후보의 candidate_status:",
        "",
        "- `sample_review_candidate_only`",
        "- `not_training_ready`",
        "- `not_hard_negative_final`",
        "- `requires_visual_review`",
        "",
        "## Threshold 확정 아님",
        "",
        "- sample-local p99는 글로벌 threshold가 아님.",
        "- threshold 확정은 이 단계에서 불가.",
        "- 후보 ranking 기준으로만 사용 (sample_local_not_global 표시됨).",
        "",
        "## Hard Negative 최종 채택 아님",
        "",
        "- 이 단계의 output은 review candidate table이며 hard negative manifest가 아님.",
        "- 최종 채택에는 visual review 및 사용자 명시적 승인 필요.",
        "",
        "## roi_0_0_patch_ratio 주의사항",
        "",
        "- roi_0_0_patch_ratio는 pure_lung_patch_ratio의 대체 후보로 사용.",
        "- 동일한 의미라고 단정하지 않음.",
        "- 최소 ROI inclusion filter 역할만 수행.",
        "",
        "## candidate_class_hint",
        "",
        "- 기본값: `unknown_requires_visual_review`",
        "- CT/PNG를 로드하지 않으므로 구조물 label 확정 불가.",
        "- vessel/pleural 등을 score/좌표만으로 단정하지 않음.",
        "",
        "## 다음 단계",
        "",
        "- **visual review pack preflight (Phase 5.94)**",
        "  - sample candidate에 대한 PNG 시각 검토 pack 구성",
        "  - PNG 생성 전 preflight 먼저 수행",
        "- 또는 **expanded candidate source refinement**",
        "  - candidate selection rule 조정 후 재 dry-run",
        "",
        "## 제한 사항",
        "",
        "- sample-local p99는 global threshold가 아님 — threshold 확정에 사용 불가.",
        "- roi_0_0_patch_ratio ≠ pure_lung_patch_ratio (확정 불가).",
        "- candidate_class_hint는 unknown — CT/PNG visual review 필요.",
        "- SRC01b score CSV는 val+test only (train 환자 score 없음, train 차단됨).",
        "- stage2_holdout 및 v2 봉인 유지.",
        "- hard negative 최종 채택은 별도 visual review 및 사용자 승인 필요.",
        "",
        "## Safety Checklist",
        "",
        "- [x] --dry-run required",
        "- [x] --split train blocked at argparse choices level (choices=['val','test'])",
        "- [x] SRC02 forbidden (guard_forbidden_path)",
        "- [x] forbidden path segment guard (segment-level, not substring)",
        "- [x] second-stage-lesion-refiner-v1 오탐 차단 방지 (PROJECT_FOLDER_WHITELIST)",
        "- [x] stage2_holdout unused",
        "- [x] v2 unused",
        "- [x] No model forward",
        "- [x] No score recalculation",
        "- [x] No threshold finalization",
        "- [x] No hard negative finalization",
        "- [x] No CT/ROI/mask npy loaded",
        "- [x] No crop manifest modified",
        "- [x] No training dataset modified",
        "- [x] No split CSV modified",
        "- [x] candidate_class_hint = unknown_requires_visual_review",
        "- [x] output guard: output root checked before creation",
        "- [x] output root created only after all validations passed",
        "- [x] pre-write existence re-check for CSV/JSON/MD",
        "- [x] candidate_status = review-only (4 flags)",
        "- [x] sample_local_not_global marked on all candidates",
        "- [x] SRC01b val+test only fact recorded in MD/JSON notes",
        "- [x] train automatic fallback: NOT implemented",
    ]

    md_content = "\n".join(md_lines)

    if OUTPUT_MD.exists():
        sys.exit(f"[GUARD] Output MD already appeared before write: {OUTPUT_MD}")
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[INFO] MD report saved: {OUTPUT_MD}")

    # ==================================================
    # FINAL: 완료 보고
    # ==================================================
    print("\n[INFO] Phase 5.93 dry-run complete.")
    print(f"[INFO] CSV:  {OUTPUT_CSV}")
    print(f"[INFO] JSON: {OUTPUT_JSON}")
    print(f"[INFO] MD:   {OUTPUT_MD}")
    print("\n[NOTE] This output is review-only. Not a hard negative manifest.")
    print("[NOTE] Future command example:")
    print(
        "       python scripts/phase5_93_expanded_hard_negative_candidate_sample_dry_run.py"
        " --dry-run --split val --sample-patients 3"
    )


if __name__ == "__main__":
    main()
