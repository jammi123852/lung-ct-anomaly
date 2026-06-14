#!/usr/bin/env python3
"""Phase 8.0: stage2_holdout final evaluation preflight — read-only safety check only.

stage2_holdout scoring / model forward / metric 계산은 수행하지 않는다.
파일 존재 여부 및 split inventory 확인만 허용.
"""

import argparse
import csv
import fnmatch
import json
import sys
from pathlib import Path

import pandas as pd

# ── project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SL_ROOT = PROJECT_ROOT / "outputs" / "second-stage-lesion-refiner-v1"

# ── 입력 경로 (read-only) ──────────────────────────────────────────────────────
SPLIT_CSV = SL_ROOT / "splits" / "lesion_stage_split_v1.csv"
ORIGINAL_REQUEST_SPLIT_PATH = "splits/lesion_stage_split_v1.csv"

PHASE7_7_DIR = SL_ROOT / "evaluation" / "phase7_7_v1v1_final_performance_closure_v1"
PHASE7_7_JSON = PHASE7_7_DIR / "phase7_7_v1v1_final_performance_closure_v1.json"

PHASE6_1B_DIR = (
    SL_ROOT / "review_annotations" / "phase6_1b_s6a_stage1_dev_filtered_manifest_v1"
)
EXCLUDED_ROWS_CSV = PHASE6_1B_DIR / "phase6_1b_s6a_stage2_holdout_excluded_rows_v1.csv"
STAGE1_DEV_MANIFEST_CSV = PHASE6_1B_DIR / "phase6_1b_s6a_stage1_dev_filtered_manifest_v1.csv"

# ── 출력 경로 ──────────────────────────────────────────────────────────────────
OUT_DIR = (
    SL_ROOT / "review_annotations" / "phase8_0_stage2_holdout_final_eval_preflight_v1"
)
OUT_CSV = OUT_DIR / "phase8_0_stage2_holdout_final_eval_preflight_v1.csv"
OUT_JSON = OUT_DIR / "phase8_0_stage2_holdout_final_eval_preflight_v1.json"
OUT_MD = OUT_DIR / "phase8_0_stage2_holdout_final_eval_preflight_report_v1.md"

# ── asset 후보 경로 ────────────────────────────────────────────────────────────
MIXED_CROP_ROOT = SL_ROOT / "crops_s6a_6ch_full"
DEDICATED_MANIFEST_PHASE8_1 = (
    SL_ROOT / "review_annotations" / "phase8_1_stage2_holdout_manifest_preflight_v1"
)
DEDICATED_DATASETS_DIR = SL_ROOT / "datasets" / "stage2_holdout"
DEDICATED_MANIFEST_CSV_1 = SL_ROOT / "datasets" / "s6a_stage2_holdout_manifest.csv"
DEDICATED_MANIFEST_CSV_2 = SL_ROOT / "datasets" / "s6a_stage2_holdout_filtered_manifest.csv"
SCORES_ROOT = SL_ROOT / "scores"


# ── guard ──────────────────────────────────────────────────────────────────────

def guard_run_flag(run: bool) -> None:
    if not run:
        print("Dry mode: pass --run to execute")
        sys.exit(0)


def guard_output_collision() -> None:
    if OUT_DIR.exists():
        print(f"[ABORT] output root already exists: {OUT_DIR}")
        print("기존 output root가 존재합니다. 덮어쓰기 금지.")
        sys.exit(1)
    for p in [OUT_CSV, OUT_JSON, OUT_MD]:
        if p.exists():
            print(f"[ABORT] output file already exists: {p}")
            sys.exit(1)


# ── 데이터 로드 (read-only) ────────────────────────────────────────────────────

def load_phase7_7_json() -> dict | None:
    if not PHASE7_7_JSON.exists():
        return None
    with open(PHASE7_7_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split_csv() -> pd.DataFrame | None:
    if not SPLIT_CSV.exists():
        return None
    return pd.read_csv(SPLIT_CSV, dtype=str)


def load_excluded_rows_csv() -> pd.DataFrame | None:
    if not EXCLUDED_ROWS_CSV.exists():
        return None
    return pd.read_csv(EXCLUDED_ROWS_CSV, dtype=str)


# ── scores 1단계 glob (recursive 금지) ────────────────────────────────────────

def glob_scores_1level(pattern: str) -> list[str]:
    if not SCORES_ROOT.exists():
        return []
    return [p.name for p in SCORES_ROOT.iterdir() if fnmatch.fnmatch(p.name, pattern)]


# ── Section A ──────────────────────────────────────────────────────────────────

def build_section_a(p77: dict | None) -> list[dict]:
    rows = []
    if p77 is None:
        rows.append({
            "section": "A", "item": "phase7_7_json",
            "status": "MISSING", "evidence": str(PHASE7_7_JSON), "note": "phase7_7 JSON 없음",
        })
        return rows

    cl = p77.get("crop_level_metrics", {})
    l1 = cl.get("crop_score_l1_mean", {})
    mse = cl.get("crop_score_mse_mean", {})
    pm = p77.get("patient_level_metrics_majority_rule", {})
    p_l1 = pm.get("crop_score_l1_mean", {})
    p_mse = pm.get("crop_score_mse_mean", {})
    stage2_st = p77.get("evaluation_completeness_status", {}).get("stage2_holdout_status", "UNKNOWN")

    rows += [
        {"section": "A", "item": "stage1_dev_performance_closure",
         "status": p77.get("final_performance_closure_status", "UNKNOWN"),
         "evidence": str(PHASE7_7_JSON.relative_to(PROJECT_ROOT)),
         "note": "phase7_7 JSON final_performance_closure_status"},
        {"section": "A", "item": "crop_score_l1_mean_auroc",
         "status": str(l1.get("auroc", "N/A")),
         "evidence": str(PHASE7_7_JSON.relative_to(PROJECT_ROOT)), "note": "crop-level, stage1_dev filtered"},
        {"section": "A", "item": "crop_score_l1_mean_auprc",
         "status": str(l1.get("auprc", "N/A")),
         "evidence": str(PHASE7_7_JSON.relative_to(PROJECT_ROOT)), "note": ""},
        {"section": "A", "item": "crop_score_mse_mean_auroc",
         "status": str(mse.get("auroc", "N/A")),
         "evidence": str(PHASE7_7_JSON.relative_to(PROJECT_ROOT)), "note": ""},
        {"section": "A", "item": "crop_score_mse_mean_auprc",
         "status": str(mse.get("auprc", "N/A")),
         "evidence": str(PHASE7_7_JSON.relative_to(PROJECT_ROOT)), "note": ""},
        {"section": "A", "item": "positive_prevalence",
         "status": str(cl.get("positive_prevalence", "N/A")),
         "evidence": str(PHASE7_7_JSON.relative_to(PROJECT_ROOT)), "note": ""},
        {"section": "A", "item": "patient_mean_l1_auroc",
         "status": str(p_l1.get("patient_mean", "N/A")),
         "evidence": str(PHASE7_7_JSON.relative_to(PROJECT_ROOT)), "note": "majority_label1 rule 조건부"},
        {"section": "A", "item": "patient_mean_mse_auroc",
         "status": str(p_mse.get("patient_mean", "N/A")),
         "evidence": str(PHASE7_7_JSON.relative_to(PROJECT_ROOT)), "note": "majority_label1 rule 조건부"},
        {"section": "A", "item": "stage2_holdout_locked",
         "status": stage2_st,
         "evidence": str(PHASE7_7_JSON.relative_to(PROJECT_ROOT)), "note": "stage2_holdout locked/unused 상태"},
    ]
    return rows


# ── Section B ──────────────────────────────────────────────────────────────────

def build_section_b(split_df: pd.DataFrame | None) -> list[dict]:
    rows = []
    split_exists = SPLIT_CSV.exists()

    rows.append({
        "section": "B", "item": "split_csv_exists",
        "expected": "TRUE", "observed": str(split_exists).upper(),
        "status": "PASS" if split_exists else "FAIL",
        "note": str(SPLIT_CSV.relative_to(PROJECT_ROOT)),
    })
    rows.append({
        "section": "B", "item": "split_path_corrected",
        "expected": str(SPLIT_CSV.relative_to(PROJECT_ROOT)),
        "observed": ORIGINAL_REQUEST_SPLIT_PATH,
        "status": "CORRECTED",
        "note": "요청서 표기 경로와 실제 경로가 달랐음. 실제 존재 경로 기준으로 preflight 수행.",
    })

    if split_df is not None and "stage_split" in split_df.columns:
        holdout_count = int((split_df["stage_split"] == "stage2_holdout").sum())
        dev_count = int((split_df["stage_split"] == "stage1_dev").sum())
        total = len(split_df)
        rows += [
            {"section": "B", "item": "stage2_holdout_patient_count",
             "expected": "154", "observed": str(holdout_count),
             "status": "PASS" if holdout_count == 154 else "CHECK",
             "note": f"실제 count={holdout_count}"},
            {"section": "B", "item": "stage1_dev_patient_count",
             "expected": "-", "observed": str(dev_count), "status": "INFO", "note": ""},
            {"section": "B", "item": "split_total_rows",
             "expected": "-", "observed": str(total), "status": "INFO", "note": ""},
        ]
    else:
        col_list = list(split_df.columns) if split_df is not None else []
        rows.append({
            "section": "B", "item": "split_csv_columns",
            "expected": "stage_split column required", "observed": str(col_list),
            "status": "FAIL", "note": "stage_split 컬럼 없음",
        })
    return rows


# ── Section C ──────────────────────────────────────────────────────────────────

def build_section_c() -> list[dict]:
    rows = []

    def asset_row(asset_type: str, path: Path, allowed: str, forbidden: str, note: str = "") -> dict:
        return {
            "section": "C", "asset_type": asset_type,
            "candidate_path": str(path.relative_to(PROJECT_ROOT)),
            "exists": str(path.exists()).upper(),
            "allowed_use": allowed,
            "forbidden_use": forbidden,
            "note": note,
        }

    rows.append(asset_row(
        "mixed_crop_root", MIXED_CROP_ROOT,
        "기존 S6-A crop root 후보 확인 (1단계 stat만)",
        "npz 로드 금지 / recursive scan 금지 / READY 판정 근거로 사용 금지",
        "mixed/original asset — stage1_dev+stage2_holdout 혼재 가능",
    ))
    rows.append(asset_row(
        "dedicated_manifest_phase8_1", DEDICATED_MANIFEST_PHASE8_1,
        "없으면 MISSING으로 기록",
        "내용 로드 금지 / scoring 입력 금지",
        "dedicated stage2_holdout manifest preflight 결과",
    ))
    rows.append(asset_row(
        "dedicated_manifest_datasets_dir", DEDICATED_DATASETS_DIR,
        "없으면 MISSING으로 기록",
        "내용 로드 금지",
        "",
    ))
    rows.append(asset_row(
        "dedicated_manifest_csv_1", DEDICATED_MANIFEST_CSV_1,
        "없으면 MISSING으로 기록",
        "내용 로드 금지 / scoring 입력 금지",
        "",
    ))
    rows.append(asset_row(
        "dedicated_manifest_csv_2", DEDICATED_MANIFEST_CSV_2,
        "없으면 MISSING으로 기록",
        "내용 로드 금지 / scoring 입력 금지",
        "",
    ))

    # scores 1단계 glob
    for asset_type, pattern in [
        ("score_output_phase8", "phase8_*"),
        ("score_output_holdout", "stage2_holdout*"),
        ("score_output_holdout2", "*holdout*"),
    ]:
        matched = glob_scores_1level(pattern)
        rows.append({
            "section": "C", "asset_type": asset_type,
            "candidate_path": str(SCORES_ROOT.relative_to(PROJECT_ROOT)) + f"/{pattern}",
            "exists": str(bool(matched)).upper(),
            "allowed_use": "존재 여부 확인만",
            "forbidden_use": "내용 로드 금지 / 분석 금지",
            "note": f"matches: {matched}",
        })
    return rows


# ── Section D ──────────────────────────────────────────────────────────────────

def build_section_d(
    split_df: pd.DataFrame | None,
    excluded_df: pd.DataFrame | None,
) -> list[dict]:
    rows = []
    exc_exists = EXCLUDED_ROWS_CSV.exists()

    rows.append({
        "section": "D", "check_item": "excluded_rows_file_exists",
        "status": "PASS" if exc_exists else "FAIL",
        "evidence": str(EXCLUDED_ROWS_CSV.relative_to(PROJECT_ROOT)),
        "required_action": "" if exc_exists else "Phase 6.1b excluded rows 파일 확인 필요",
    })

    if excluded_df is not None:
        exc_count = len(excluded_df)
        rows.append({
            "section": "D", "check_item": "excluded_rows_count",
            "status": str(exc_count),
            "evidence": f"crop-level row count={exc_count}",
            "required_action": "",
        })

        if (
            split_df is not None
            and "stage_split" in split_df.columns
            and "patient_id" in split_df.columns
            and "patient_id" in excluded_df.columns
        ):
            holdout_pids = set(
                split_df.loc[split_df["stage_split"] == "stage2_holdout", "patient_id"].astype(str)
            )
            exc_pids = set(excluded_df["patient_id"].astype(str).unique())
            overlap = holdout_pids & exc_pids
            rows.append({
                "section": "D", "check_item": "excluded_patient_ids_match_split",
                "status": "OVERLAP_FOUND" if overlap else "NO_OVERLAP",
                "evidence": (
                    f"overlap={len(overlap)} / holdout_split={len(holdout_pids)} / "
                    f"exc_unique_patients={len(exc_pids)}"
                ),
                "required_action": (
                    "leakage audit evidence — excluded rows가 stage2_holdout 분리 근거임"
                    if overlap else "overlap 없음, 추가 확인 필요"
                ),
            })
        else:
            missing_cols = []
            if split_df is None:
                missing_cols.append("split_df is None")
            elif "patient_id" not in split_df.columns:
                missing_cols.append("split patient_id column missing")
            if excluded_df is not None and "patient_id" not in excluded_df.columns:
                missing_cols.append(f"excluded patient_id column missing (cols={list(excluded_df.columns)})")
            rows.append({
                "section": "D", "check_item": "excluded_patient_ids_match_split",
                "status": "SKIP",
                "evidence": "; ".join(missing_cols),
                "required_action": "컬럼 구조 확인 필요",
            })
    else:
        rows += [
            {"section": "D", "check_item": "excluded_rows_count",
             "status": "MISSING", "evidence": str(EXCLUDED_ROWS_CSV), "required_action": "파일 없음"},
            {"section": "D", "check_item": "excluded_patient_ids_match_split",
             "status": "MISSING", "evidence": str(EXCLUDED_ROWS_CSV), "required_action": "파일 없음"},
        ]

    rows.append({
        "section": "D", "check_item": "stage1_dev_manifest_not_reused",
        "status": "PASS",
        "evidence": (
            f"stage1_dev filtered manifest: "
            f"{STAGE1_DEV_MANIFEST_CSV.relative_to(PROJECT_ROOT)}. "
            "이 스크립트는 stage2_holdout 평가에 재사용하지 않음."
        ),
        "required_action": "",
    })
    rows.append({
        "section": "D", "check_item": "original_s6a_index_not_used",
        "status": "PASS",
        "evidence": "원본 S6-A index (1222 rows)를 로드하거나 평가 입력으로 사용하지 않음.",
        "required_action": "",
    })
    rows.append({
        "section": "D", "check_item": "contamination_history_noted",
        "status": "NOTED",
        "evidence": (
            "원본 S6-A dataset index 오염 이력: LUNG1-295, LUNG1-415 (1,222 rows). "
            "stage1_dev 평가에는 Phase 6.1b filtered shadow manifest만 사용."
        ),
        "required_action": "",
    })
    return rows


# ── Section E ──────────────────────────────────────────────────────────────────

def build_section_e(
    sec_c: list[dict],
    sec_d: list[dict],
) -> tuple[list[dict], str, str, str]:
    c_exists = {r["asset_type"]: r["exists"] for r in sec_c}

    dedicated_manifest_exists = any(
        c_exists.get(k) == "TRUE"
        for k in [
            "dedicated_manifest_phase8_1",
            "dedicated_manifest_datasets_dir",
            "dedicated_manifest_csv_1",
            "dedicated_manifest_csv_2",
        ]
    )
    # dedicated crop root 후보는 아직 없음
    dedicated_crop_root_exists = False

    d_status = {r["check_item"]: r["status"] for r in sec_d}
    leakage_all_pass = all(
        d_status.get(k, "FAIL") in ("PASS", "NOTED", "OVERLAP_FOUND")
        for k in [
            "excluded_rows_file_exists",
            "stage1_dev_manifest_not_reused",
            "original_s6a_index_not_used",
            "contamination_history_noted",
        ]
    )

    rows = [
        {
            "section": "E", "item": "dedicated_manifest_available",
            "status": str(dedicated_manifest_exists).upper(),
            "blocker": "" if dedicated_manifest_exists else "dedicated stage2_holdout filtered manifest 없음",
            "next_required_action": (
                "" if dedicated_manifest_exists
                else "Phase 8.1: dedicated stage2_holdout manifest 생성 preflight 필요"
            ),
        },
        {
            "section": "E", "item": "dedicated_crop_root_available",
            "status": str(dedicated_crop_root_exists).upper(),
            "blocker": (
                "" if dedicated_crop_root_exists
                else "dedicated stage2_holdout crop root 없음 (mixed crops_s6a_6ch_full/ 만 존재)"
            ),
            "next_required_action": (
                "" if dedicated_crop_root_exists
                else "dedicated stage2_holdout crop 생성 필요 (smoke → full)"
            ),
        },
        {
            "section": "E", "item": "leakage_safety_all_pass",
            "status": str(leakage_all_pass).upper(),
            "blocker": "" if leakage_all_pass else "leakage safety check 미통과 항목 있음",
            "next_required_action": "" if leakage_all_pass else "Section D 재확인",
        },
    ]

    if not leakage_all_pass:
        readiness = "BLOCKED_LEAKAGE_RISK"
        blocker = "leakage safety check 미통과"
        next_action = "Section D 전항목 PASS 확인 후 재진행"
    elif not dedicated_manifest_exists or not dedicated_crop_root_exists:
        readiness = "BLOCKED_MISSING_STAGE2_ASSETS"
        parts = []
        if not dedicated_manifest_exists:
            parts.append("dedicated manifest 없음")
        if not dedicated_crop_root_exists:
            parts.append("dedicated crop root 없음")
        blocker = "; ".join(parts)
        next_action = (
            "Phase 8.1: dedicated stage2_holdout filtered manifest 및 crop 생성 preflight 설계 필요"
        )
    else:
        readiness = "READY_FOR_STAGE2_HOLDOUT_SCORING_SMOKE"
        blocker = ""
        next_action = "Phase 8.1: stage2_holdout scoring smoke 설계 진행 가능"

    rows += [
        {
            "section": "E", "item": "readiness_for_stage2_holdout_eval",
            "status": readiness,
            "blocker": blocker,
            "next_required_action": next_action,
        },
        {
            "section": "E", "item": "next_step",
            "status": next_action,
            "blocker": blocker,
            "next_required_action": next_action,
        },
    ]
    return rows, readiness, blocker, next_action


# ── 출력 저장 ──────────────────────────────────────────────────────────────────

def write_csv(all_rows: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(
        ["section", "item", "status", "evidence", "note",
         "expected", "observed",
         "asset_type", "candidate_path", "exists", "allowed_use", "forbidden_use",
         "check_item", "required_action",
         "blocker", "next_required_action"]
    ))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def write_json(
    split_df: pd.DataFrame | None,
    sec_c: list[dict],
    sec_d: list[dict],
    readiness: str,
    blocker: str,
    next_action: str,
    p77: dict | None,
) -> None:
    holdout_count = 0
    if split_df is not None and "stage_split" in split_df.columns:
        holdout_count = int((split_df["stage_split"] == "stage2_holdout").sum())

    cl = (p77 or {}).get("crop_level_metrics", {})
    l1 = cl.get("crop_score_l1_mean", {})
    mse = cl.get("crop_score_mse_mean", {})
    pm = (p77 or {}).get("patient_level_metrics_majority_rule", {})

    c_map = {r["asset_type"]: {"exists": r["exists"], "path": r["candidate_path"]} for r in sec_c}
    d_map = {
        r["check_item"]: {"status": r["status"], "evidence": r.get("evidence", "")}
        for r in sec_d
    }

    data = {
        "input_paths": {
            "split_csv": str(SPLIT_CSV.relative_to(PROJECT_ROOT)),
            "phase7_7_json": str(PHASE7_7_JSON.relative_to(PROJECT_ROOT)),
            "excluded_rows_csv": str(EXCLUDED_ROWS_CSV.relative_to(PROJECT_ROOT)),
            "stage1_dev_manifest_csv": str(STAGE1_DEV_MANIFEST_CSV.relative_to(PROJECT_ROOT)),
        },
        "current_stage1_dev_performance_status": {
            "final_performance_closure_status": (
                p77.get("final_performance_closure_status", "N/A") if p77 else "N/A"
            ),
            "crop_score_l1_mean_auroc": l1.get("auroc"),
            "crop_score_l1_mean_auprc": l1.get("auprc"),
            "crop_score_mse_mean_auroc": mse.get("auroc"),
            "crop_score_mse_mean_auprc": mse.get("auprc"),
            "positive_prevalence": cl.get("positive_prevalence"),
            "patient_mean_l1_auroc": pm.get("crop_score_l1_mean", {}).get("patient_mean"),
            "patient_mean_mse_auroc": pm.get("crop_score_mse_mean", {}).get("patient_mean"),
        },
        "corrected_split_path": str(SPLIT_CSV.relative_to(PROJECT_ROOT)),
        "original_request_split_path": ORIGINAL_REQUEST_SPLIT_PATH,
        "split_path_correction_note": (
            "요청서 표기 경로와 실제 경로가 달랐음. 실제 존재 경로 기준으로 preflight 수행함."
        ),
        "stage2_holdout_patient_count": holdout_count,
        "stage2_holdout_manifest_candidates": {
            k: c_map.get(k, {})
            for k in [
                "dedicated_manifest_phase8_1",
                "dedicated_manifest_datasets_dir",
                "dedicated_manifest_csv_1",
                "dedicated_manifest_csv_2",
            ]
        },
        "stage2_holdout_crop_root_candidates": {"mixed_crop_root": c_map.get("mixed_crop_root", {})},
        "stage2_holdout_score_output_candidates": {
            k: c_map.get(k, {})
            for k in ["score_output_phase8", "score_output_holdout", "score_output_holdout2"]
        },
        "asset_existence_summary": {r["asset_type"]: r["exists"] for r in sec_c},
        "leakage_safety_checks": d_map,
        "readiness_for_stage2_holdout_eval": readiness,
        "blockers": [blocker] if blocker else [],
        "required_next_steps": [next_action] if next_action else [],
        "notes": {
            "preflight_only": True,
            "no_stage2_holdout_loading": True,
            "no_model_forward": True,
            "no_scoring": True,
            "no_metric_calculation": True,
            "no_threshold": True,
            "no_training": True,
            "no_v2v2": True,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_md(
    split_df: pd.DataFrame | None,
    sec_c: list[dict],
    sec_d: list[dict],
    readiness: str,
    blocker: str,
    next_action: str,
    p77: dict | None,
) -> None:
    holdout_count = 0
    stage1_dev_count = 0
    if split_df is not None and "stage_split" in split_df.columns:
        holdout_count = int((split_df["stage_split"] == "stage2_holdout").sum())
        stage1_dev_count = int((split_df["stage_split"] == "stage1_dev").sum())

    cl = (p77 or {}).get("crop_level_metrics", {})
    l1 = cl.get("crop_score_l1_mean", {})
    mse = cl.get("crop_score_mse_mean", {})
    pm = (p77 or {}).get("patient_level_metrics_majority_rule", {})

    lines = [
        "# Phase 8.0: stage2_holdout Final Evaluation Preflight",
        "",
        "## 1. Phase 8.0 목적",
        "",
        "Position-Aware PaDiM / Second-Stage Lesion Refiner v1의 stage2_holdout 최종 평가 실행 전 안전 점검 문서를 생성한다.",
        "",
        "이번 Phase 8.0은 preflight only이며, stage2_holdout 데이터 내용 분석이나 scoring은 수행하지 않는다.",
        "",
        "## 2. split 경로 보정 기록",
        "",
        f"- 요청서 표기 경로: `{ORIGINAL_REQUEST_SPLIT_PATH}`",
        f"- 실제 사용 경로: `{SPLIT_CSV.relative_to(PROJECT_ROOT)}`",
        "- 요청서 표기 경로와 실제 경로가 달랐음. 실제 존재 경로 기준으로 preflight를 수행했음.",
        "",
        "## 3. 현재 v1/v1 stage1_dev 성능평가 종료 상태",
        "",
    ]
    if p77:
        stage2_st = p77.get("evaluation_completeness_status", {}).get("stage2_holdout_status", "N/A")
        lines += [
            f"- final_performance_closure_status: **{p77.get('final_performance_closure_status', 'N/A')}**",
            f"- crop_score_l1_mean AUROC / AUPRC: {l1.get('auroc')} / {l1.get('auprc')}",
            f"- crop_score_mse_mean AUROC / AUPRC: {mse.get('auroc')} / {mse.get('auprc')}",
            f"- positive_prevalence: {cl.get('positive_prevalence')}",
            f"- patient_mean AUROC (l1, majority_rule): {pm.get('crop_score_l1_mean', {}).get('patient_mean')}",
            f"- patient_mean AUROC (mse, majority_rule): {pm.get('crop_score_mse_mean', {}).get('patient_mean')}",
            f"- stage2_holdout_status: {stage2_st}",
        ]
    else:
        lines.append("- phase7_7 JSON을 찾을 수 없습니다.")

    lines += [
        "",
        "## 4. stage2_holdout split 확인",
        "",
        f"- split CSV 경로: `{SPLIT_CSV.relative_to(PROJECT_ROOT)}`",
        f"- stage2_holdout patient count: **{holdout_count}** (예상 154)",
        f"- stage1_dev patient count: {stage1_dev_count}",
        "- stage2_holdout row 154건 확인은 split inventory 확인일 뿐, 데이터 내용 분석이나 평가 실행이 아님.",
        "- stage2_holdout은 계속 locked / unused 상태임.",
        "",
        "## 5. stage2_holdout 입력 자산 존재 여부",
        "",
    ]
    for r in sec_c:
        exists_label = "EXISTS" if r["exists"] == "TRUE" else "MISSING"
        lines.append(f"- [{r['asset_type']}] `{r['candidate_path']}`: **{exists_label}**")
        if r.get("note"):
            lines.append(f"  - {r['note']}")

    lines += [
        "",
        "> `crops_s6a_6ch_full/` 존재는 stage2_holdout 평가 READY를 의미하지 않는다.",
        "",
        "## 6. stage1_dev / stage2_holdout 분리 안전성",
        "",
    ]
    for r in sec_d:
        lines.append(f"- **{r['check_item']}**: {r['status']}")
        if r.get("evidence"):
            lines.append(f"  - evidence: {r['evidence']}")
        if r.get("required_action"):
            lines.append(f"  - required_action: {r['required_action']}")

    lines += [
        "",
        "> Phase 6.1b excluded rows는 leakage audit evidence이지, final evaluation manifest가 아니다.",
        "",
        "## 7. stage2_holdout 평가 전 필요한 자산 목록",
        "",
        "현재 MISSING 또는 준비 미완료 자산:",
    ]
    for r in sec_c:
        if r["exists"] == "FALSE":
            lines.append(f"- `{r['candidate_path']}`")

    lines += [
        "",
        "stage2_holdout 최종 평가를 위해서는 dedicated stage2_holdout filtered manifest와 전용 scoring smoke 설계가 필요하다.",
        "",
        "## 8. readiness 판정",
        "",
        f"**{readiness}**",
    ]
    if blocker:
        lines.append(f"- Blocker: {blocker}")

    lines += [
        "",
        "## 9. 다음 단계",
        "",
        f"- {next_action}",
        "",
        "## 10. 금지 사항",
        "",
    ]
    for item in [
        "stage2_holdout crop 실제 로드",
        "stage2_holdout CT/ROI/mask npy 로드",
        "model forward",
        "scoring",
        "metric 계산",
        "threshold/p95/p99/hit-rate 계산",
        "training",
        "checkpoint 생성",
        "score CSV 수정",
        "filtered manifest 수정",
        "원본 S6-A index (1222 rows) 사용",
        "crop 파일 수정/삭제/이동",
        "기존 Phase 6/7 output 삭제/이동/덮어쓰기",
        "v2/v2v2 접근",
        "NSCLC/MSD root 내용 접근",
        "split CSV 수정/복사/symlink 생성",
        "pip/conda install",
        "외부 다운로드",
    ]:
        lines.append(f"- {item} 금지")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8.0 stage2_holdout final evaluation preflight"
    )
    parser.add_argument("--run", action="store_true", help="실제 실행 (없으면 dry mode)")
    parsed = parser.parse_args()

    guard_run_flag(parsed.run)
    guard_output_collision()

    print("[Phase 8.0] stage2_holdout final evaluation preflight 시작")

    print("  - phase7_7 JSON 로드...")
    p77 = load_phase7_7_json()

    print("  - split CSV 로드 (read-only)...")
    split_df = load_split_csv()

    print("  - excluded rows CSV 로드 (read-only)...")
    excluded_df = load_excluded_rows_csv()

    print("  - Section A: current_status...")
    sec_a = build_section_a(p77)

    print("  - Section B: split inventory...")
    sec_b = build_section_b(split_df)

    print("  - Section C: asset existence...")
    sec_c = build_section_c()

    print("  - Section D: leakage safety checks...")
    sec_d = build_section_d(split_df, excluded_df)

    print("  - Section E: readiness decision...")
    sec_e, readiness, blocker, next_action = build_section_e(sec_c, sec_d)

    all_rows = sec_a + sec_b + sec_c + sec_d + sec_e

    try:
        OUT_DIR.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"[ABORT] output root already exists (race condition): {OUT_DIR}")
        sys.exit(1)

    print("  - CSV 저장...")
    write_csv(all_rows)

    print("  - JSON 저장...")
    write_json(split_df, sec_c, sec_d, readiness, blocker, next_action, p77)

    print("  - MD 보고서 저장...")
    write_md(split_df, sec_c, sec_d, readiness, blocker, next_action, p77)

    print(f"\n[완료] readiness: {readiness}")
    print(f"  output root : {OUT_DIR}")
    print(f"  CSV         : {OUT_CSV}")
    print(f"  JSON        : {OUT_JSON}")
    print(f"  MD          : {OUT_MD}")
    if blocker:
        print(f"  blocker     : {blocker}")
    print(f"  next step   : {next_action}")


if __name__ == "__main__":
    main()
