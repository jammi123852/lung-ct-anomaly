"""
Phase 8.3C: Stage2 Scoring Smoke Output Validation

목적:
  Phase 8.3B smoke scoring 산출물을 검증하여
  Phase 8.4 full scoring preflight로 넘어갈 수 있는지 판단한다.
  read-only validation only. model forward/scoring/metric 계산 전혀 없음.

실행:
  python scripts/phase8_3c_stage2_scoring_smoke_output_validation.py          # dry-run
  python scripts/phase8_3c_stage2_scoring_smoke_output_validation.py --run    # 실제 검증 수행
"""

import argparse
import json
import os
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

# torch/model import 절대 금지 (read-only validation only)

# ---------------------------------------------------------------------------
# 경로 정의
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent

SMOKE_SCORE_CSV = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/scores/phase8_3_stage2_scoring_smoke_v1/phase8_3_stage2_scoring_smoke_v1.csv"
REVIEW_ROOT = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_3_stage2_scoring_smoke_v1"
SUMMARY_JSON = REVIEW_ROOT / "phase8_3_stage2_scoring_smoke_summary_v1.json"
ERRORS_CSV = REVIEW_ROOT / "phase8_3_stage2_scoring_smoke_errors_v1.csv"
RUNTIME_CSV = REVIEW_ROOT / "phase8_3_stage2_scoring_smoke_runtime_summary_v1.csv"
DONE_MARKER = REVIEW_ROOT / "phase8_3_stage2_scoring_smoke_DONE.json"
SMOKE_REPORT_MD = REVIEW_ROOT / "phase8_3_stage2_scoring_smoke_report_v1.md"

STAGE2_MANIFEST = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_filtered_manifest_v1.csv"

OUT_DIR = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_3c_stage2_scoring_smoke_output_validation_v1"
OUT_CSV = OUT_DIR / "phase8_3c_stage2_scoring_smoke_output_validation_v1.csv"
OUT_JSON = OUT_DIR / "phase8_3c_stage2_scoring_smoke_output_validation_v1.json"
OUT_MD = OUT_DIR / "phase8_3c_stage2_scoring_smoke_output_validation_report_v1.md"

# Phase 8.4 full scoring output root (존재하면 안 됨)
PHASE84_FULL_SCORING_ROOT = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/scores/phase8_4_stage2_full_scoring_v1"
# Phase 7.2 score output (존재해야 함 = untouched)
PHASE72_SCORE_ROOT = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/scores/phase7_2_v1v1_stage1_dev_full_scoring_v1"

# ---------------------------------------------------------------------------
# canonical 컬럼 목록
# ---------------------------------------------------------------------------
CANONICAL_COLS = [
    "crop_id", "patient_id", "npz_path", "label", "sampling_label",
    "stage_split", "model_tag", "checkpoint_path",
    "crop_score_l1_mean", "crop_score_l1_max", "crop_score_mse_mean",
    "channel_0_l1_mean", "channel_1_l1_mean", "channel_2_l1_mean",
    "channel_3_l1_mean", "channel_4_l1_mean", "channel_5_l1_mean",
    "lung_channels_l1_mean", "mediastinal_channels_l1_mean",
    "input_min", "input_max", "recon_min", "recon_max",
    "error_min", "error_max", "has_nan", "has_inf",
]

SMOKE_EXTRA_COLS = [
    "row_id", "crop_shape",
    "input_nan_count", "input_inf_count",
    "recon_nan_count", "recon_inf_count",
    "error_nan_count", "error_inf_count",
    "score_nan_count", "score_inf_count",
    "smoke_status", "issue", "note",
]

SCORE_VALUE_COLS = [
    "crop_score_l1_mean", "crop_score_l1_max", "crop_score_mse_mean",
    "channel_0_l1_mean", "channel_1_l1_mean", "channel_2_l1_mean",
    "channel_3_l1_mean", "channel_4_l1_mean", "channel_5_l1_mean",
    "lung_channels_l1_mean", "mediastinal_channels_l1_mean",
]

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _is_finite_val(v):
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def _round6(v):
    try:
        return round(float(v), 6)
    except Exception:
        return None


def _norm_crop_shape(s):
    """'(6, 96, 96)' 또는 '(6,96,96)' 계열을 정규화해서 비교용 튜플 반환."""
    try:
        cleaned = str(s).replace(" ", "")
        # (6,96,96) → (6, 96, 96) 형태로 파싱
        inner = cleaned.strip("()").split(",")
        return tuple(int(x) for x in inner)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Section A: artifact existence
# ---------------------------------------------------------------------------

def section_a_artifact_existence():
    artifacts = [
        ("smoke_score_csv", SMOKE_SCORE_CSV),
        ("summary_json", SUMMARY_JSON),
        ("errors_csv", ERRORS_CSV),
        ("runtime_csv", RUNTIME_CSV),
        ("done_marker", DONE_MARKER),
        ("smoke_report_md", SMOKE_REPORT_MD),
    ]
    rows = []
    for item, path in artifacts:
        exists = path.exists()
        rows.append({
            "section": "A",
            "item": item,
            "path": str(path),
            "exists": exists,
            "status": "PASS" if exists else "FAIL",
            "note": "" if exists else "파일 없음",
        })
    return rows


# ---------------------------------------------------------------------------
# Section B: summary / DONE consistency
# ---------------------------------------------------------------------------

def section_b_summary_done_consistency():
    rows = []

    # summary JSON 로드
    summary = None
    if SUMMARY_JSON.exists():
        with open(SUMMARY_JSON, encoding="utf-8") as f:
            summary = json.load(f)

    # DONE marker 로드
    done = None
    if DONE_MARKER.exists():
        with open(DONE_MARKER, encoding="utf-8") as f:
            done = json.load(f)

    def _check(check_item, expected, observed, note=""):
        status = "PASS" if str(observed) == str(expected) else "FAIL"
        return {
            "section": "B",
            "check_item": check_item,
            "expected": expected,
            "observed": observed,
            "status": status,
            "note": note,
        }

    bool_checks = [
        ("smoke_pass", True),
        ("metric_calculation_executed", False),
        ("threshold_calculated", False),
        ("training_executed", False),
        ("backward_executed", False),
        ("optimizer_step_executed", False),
        ("checkpoint_created", False),
        ("full_scoring_executed", False),
        ("approval_gate_interpretation_corrected", True),
    ]
    int_checks = [
        ("scored_crop_count", 64),
        ("smoke_positive_count", 32),
        ("smoke_hard_negative_count", 32),
        ("has_nan_count", 0),
        ("has_inf_count", 0),
        ("error_count", 0),
    ]

    if summary is None:
        for item, _ in bool_checks + int_checks:
            rows.append({
                "section": "B", "check_item": item,
                "expected": "N/A", "observed": "summary_json_missing",
                "status": "FAIL", "note": "summary JSON 로드 실패",
            })
    else:
        for item, expected in bool_checks:
            obs = summary.get(item, "KEY_MISSING")
            rows.append(_check(item, expected, obs))
        for item, expected in int_checks:
            obs = summary.get(item, "KEY_MISSING")
            rows.append(_check(item, expected, obs))

    # DONE marker smoke_pass 비교
    if done is None:
        rows.append({
            "section": "B", "check_item": "done_marker_smoke_pass_vs_summary",
            "expected": "True", "observed": "done_marker_missing",
            "status": "FAIL", "note": "DONE marker 로드 실패",
        })
    else:
        done_sp = done.get("smoke_pass", "KEY_MISSING")
        if summary is not None:
            summary_sp = summary.get("smoke_pass", "KEY_MISSING")
            match = (str(done_sp) == str(summary_sp))
            rows.append({
                "section": "B",
                "check_item": "done_marker_smoke_pass_vs_summary",
                "expected": f"summary.smoke_pass={summary_sp}",
                "observed": f"done.smoke_pass={done_sp}",
                "status": "PASS" if match else "FAIL",
                "note": "" if match else "DONE marker와 summary JSON smoke_pass 불일치",
            })
        else:
            rows.append({
                "section": "B",
                "check_item": "done_marker_smoke_pass_vs_summary",
                "expected": "True",
                "observed": str(done_sp),
                "status": "PASS" if str(done_sp) == "True" else "FAIL",
                "note": "summary JSON 없어서 DONE marker 단독 확인",
            })

    return rows


# ---------------------------------------------------------------------------
# Section C: score CSV schema validation
# ---------------------------------------------------------------------------

def section_c_score_csv_schema():
    rows = []
    if not SMOKE_SCORE_CSV.exists():
        for col in CANONICAL_COLS:
            rows.append({
                "section": "C", "column_name": col,
                "required": "canonical", "exists": False,
                "status": "FAIL", "note": "score CSV 없음",
            })
        for col in SMOKE_EXTRA_COLS:
            rows.append({
                "section": "C", "column_name": col,
                "required": "smoke_extra", "exists": False,
                "status": "FAIL", "note": "score CSV 없음",
            })
        return rows

    df = pd.read_csv(SMOKE_SCORE_CSV, nrows=0)
    actual_cols = set(df.columns.tolist())

    for col in CANONICAL_COLS:
        exists = col in actual_cols
        rows.append({
            "section": "C", "column_name": col,
            "required": "canonical", "exists": exists,
            "status": "PASS" if exists else "FAIL",
            "note": "" if exists else "컬럼 없음",
        })
    for col in SMOKE_EXTRA_COLS:
        exists = col in actual_cols
        rows.append({
            "section": "C", "column_name": col,
            "required": "smoke_extra", "exists": exists,
            "status": "PASS" if exists else "FAIL",
            "note": "" if exists else "컬럼 없음",
        })
    return rows


# ---------------------------------------------------------------------------
# Section D: score CSV content validation
# ---------------------------------------------------------------------------

def section_d_score_csv_content():
    rows = []

    if not SMOKE_SCORE_CSV.exists():
        rows.append({
            "section": "D", "check_item": "score_csv_load",
            "expected": "file_exists", "observed": "missing",
            "status": "FAIL", "note": "score CSV 없음",
        })
        return rows

    df = pd.read_csv(SMOKE_SCORE_CSV)

    def _chk(check_item, expected, observed, note=""):
        status = "PASS" if str(observed) == str(expected) else "FAIL"
        return {
            "section": "D",
            "check_item": check_item,
            "expected": str(expected),
            "observed": str(observed),
            "status": status,
            "note": note,
        }

    # row count
    rows.append(_chk("row_count", 64, len(df)))

    # positive count (label 컬럼 기준, 값이 'positive' 또는 1)
    if "label" in df.columns:
        # label 컬럼 값 확인: 'positive' 또는 int 1
        lv = df["label"].astype(str).str.lower()
        pos_cnt = int((lv == "positive").sum())
        # int 라벨의 경우 (1)
        if pos_cnt == 0:
            try:
                pos_cnt = int((df["label"] == 1).sum())
            except Exception:
                pass
        rows.append(_chk("positive_count", 32, pos_cnt))
    else:
        rows.append({
            "section": "D", "check_item": "positive_count",
            "expected": 32, "observed": "label_col_missing",
            "status": "FAIL", "note": "label 컬럼 없음",
        })

    # hard_negative count (label 또는 sampling_label)
    hn_cnt = 0
    if "label" in df.columns:
        lv = df["label"].astype(str).str.lower()
        hn_cnt = int((lv == "hard_negative").sum())
    if hn_cnt == 0 and "sampling_label" in df.columns:
        slv = df["sampling_label"].astype(str).str.lower()
        hn_cnt = int((slv == "hard_negative").sum())
    rows.append(_chk("hard_negative_count", 32, hn_cnt))

    # stage_split unique
    if "stage_split" in df.columns:
        uniq = sorted(df["stage_split"].dropna().unique().tolist())
        expected_uniq = ["stage2_holdout"]
        rows.append(_chk("stage_split_unique", str(expected_uniq), str(uniq)))
    else:
        rows.append({
            "section": "D", "check_item": "stage_split_unique",
            "expected": "['stage2_holdout']", "observed": "col_missing",
            "status": "FAIL", "note": "stage_split 컬럼 없음",
        })

    # model_tag unique
    if "model_tag" in df.columns:
        uniq = sorted(df["model_tag"].dropna().unique().tolist())
        expected_uniq = ["rd4ad_2p5d_normal_mw_fixed96_v1"]
        rows.append(_chk("model_tag_unique", str(expected_uniq), str(uniq)))
    else:
        rows.append({
            "section": "D", "check_item": "model_tag_unique",
            "expected": "['rd4ad_2p5d_normal_mw_fixed96_v1']", "observed": "col_missing",
            "status": "FAIL", "note": "model_tag 컬럼 없음",
        })

    # smoke_status unique
    if "smoke_status" in df.columns:
        uniq = sorted(df["smoke_status"].dropna().unique().tolist())
        expected_uniq = ["PASS"]
        rows.append(_chk("smoke_status_unique", str(expected_uniq), str(uniq)))
    else:
        rows.append({
            "section": "D", "check_item": "smoke_status_unique",
            "expected": "['PASS']", "observed": "col_missing",
            "status": "FAIL", "note": "smoke_status 컬럼 없음",
        })

    # has_nan count (has_nan != 0인 행 수)
    if "has_nan" in df.columns:
        nan_rows = int((df["has_nan"] != 0).sum())
        rows.append(_chk("has_nan_row_count", 0, nan_rows))
    else:
        rows.append({
            "section": "D", "check_item": "has_nan_row_count",
            "expected": 0, "observed": "col_missing",
            "status": "FAIL", "note": "has_nan 컬럼 없음",
        })

    # has_inf count
    if "has_inf" in df.columns:
        inf_rows = int((df["has_inf"] != 0).sum())
        rows.append(_chk("has_inf_row_count", 0, inf_rows))
    else:
        rows.append({
            "section": "D", "check_item": "has_inf_row_count",
            "expected": 0, "observed": "col_missing",
            "status": "FAIL", "note": "has_inf 컬럼 없음",
        })

    # NaN/Inf count 컬럼 합계
    for col in [
        "input_nan_count", "input_inf_count",
        "recon_nan_count", "recon_inf_count",
        "error_nan_count", "error_inf_count",
        "score_nan_count", "score_inf_count",
    ]:
        if col in df.columns:
            total = int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
            rows.append(_chk(f"{col}_sum", 0, total))
        else:
            rows.append({
                "section": "D", "check_item": f"{col}_sum",
                "expected": 0, "observed": "col_missing",
                "status": "FAIL", "note": f"{col} 컬럼 없음",
            })

    # crop_shape: 전 행이 (6, 96, 96) 계열인지
    if "crop_shape" in df.columns:
        target = (6, 96, 96)
        bad = df["crop_shape"].apply(lambda s: _norm_crop_shape(s) != target)
        bad_cnt = int(bad.sum())
        rows.append(_chk("crop_shape_all_6_96_96", 0, bad_cnt,
                         note="" if bad_cnt == 0 else f"비정상 shape 행: {bad_cnt}"))
    else:
        rows.append({
            "section": "D", "check_item": "crop_shape_all_6_96_96",
            "expected": 0, "observed": "col_missing",
            "status": "FAIL", "note": "crop_shape 컬럼 없음",
        })

    # npz_path empty/null
    if "npz_path" in df.columns:
        empty_cnt = int(df["npz_path"].isnull().sum() + (df["npz_path"].astype(str).str.strip() == "").sum())
        rows.append(_chk("npz_path_empty_null", 0, empty_cnt))

        # npz_path duplicate
        dup_cnt = int(df["npz_path"].duplicated().sum())
        rows.append(_chk("npz_path_duplicate", 0, dup_cnt))

        # npz_path exists on disk
        exists_cnt = int(df["npz_path"].apply(lambda p: os.path.exists(str(p))).sum())
        rows.append(_chk("npz_path_exists_on_disk", 64, exists_cnt,
                         note=f"존재하지 않는 파일: {64 - exists_cnt}개" if exists_cnt != 64 else ""))

        # stage2 manifest 교집합
        manifest_npz_set = set()
        if STAGE2_MANIFEST.exists():
            mdf = pd.read_csv(STAGE2_MANIFEST)
            # npz_path 컬럼 우선, 없으면 crop_npz_path
            if "npz_path" in mdf.columns:
                manifest_npz_set = set(mdf["npz_path"].dropna().astype(str).tolist())
            elif "crop_npz_path" in mdf.columns:
                manifest_npz_set = set(mdf["crop_npz_path"].dropna().astype(str).tolist())

        if manifest_npz_set:
            score_npz_set = set(df["npz_path"].dropna().astype(str).tolist())
            not_in_manifest = score_npz_set - manifest_npz_set
            rows.append(_chk(
                "npz_path_all_in_stage2_manifest",
                0,
                len(not_in_manifest),
                note="" if not not_in_manifest else f"manifest에 없는 경로: {len(not_in_manifest)}개",
            ))
        else:
            rows.append({
                "section": "D", "check_item": "npz_path_all_in_stage2_manifest",
                "expected": 0, "observed": "manifest_load_failed",
                "status": "FAIL" if not STAGE2_MANIFEST.exists() else "WARNING",
                "note": "stage2 manifest 로드 실패 또는 npz_path 컬럼 없음",
            })
    else:
        for item in ["npz_path_empty_null", "npz_path_duplicate",
                     "npz_path_exists_on_disk", "npz_path_all_in_stage2_manifest"]:
            rows.append({
                "section": "D", "check_item": item,
                "expected": "N/A", "observed": "col_missing",
                "status": "FAIL", "note": "npz_path 컬럼 없음",
            })

    return rows


# ---------------------------------------------------------------------------
# Section E: score value validation
# ---------------------------------------------------------------------------

def section_e_score_value_validation():
    rows = []

    if not SMOKE_SCORE_CSV.exists():
        rows.append({
            "section": "E", "check_item": "score_csv_load",
            "expected": "file_exists", "observed": "missing",
            "status": "FAIL", "note": "score CSV 없음",
        })
        return rows

    df = pd.read_csv(SMOKE_SCORE_CSV)

    def _chk(check_item, expected, observed, note=""):
        status = "PASS" if str(observed) == str(expected) else "FAIL"
        return {
            "section": "E",
            "check_item": check_item,
            "expected": str(expected),
            "observed": str(observed),
            "status": status,
            "note": note,
        }

    # 각 score 컬럼 finite 확인
    for col in SCORE_VALUE_COLS:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            non_finite = int((~series.apply(lambda x: _is_finite_val(x))).sum())
            rows.append(_chk(f"{col}_all_finite", 0, non_finite,
                             note="" if non_finite == 0 else f"non-finite 행: {non_finite}"))
        else:
            rows.append({
                "section": "E", "check_item": f"{col}_all_finite",
                "expected": 0, "observed": "col_missing",
                "status": "FAIL", "note": f"{col} 컬럼 없음",
            })

    # summary JSON과 score 통계 비교
    summary = None
    if SUMMARY_JSON.exists():
        with open(SUMMARY_JSON, encoding="utf-8") as f:
            summary = json.load(f)

    for csv_col, json_min, json_max, json_mean in [
        ("crop_score_l1_mean", "score_l1_min", "score_l1_max", "score_l1_mean"),
        ("crop_score_mse_mean", "score_mse_min", "score_mse_max", "score_mse_mean"),
    ]:
        if csv_col in df.columns and summary is not None:
            series = pd.to_numeric(df[csv_col], errors="coerce")
            calc_min = _round6(series.min())
            calc_max = _round6(series.max())
            calc_mean = _round6(series.mean())
            exp_min = _round6(summary.get(json_min))
            exp_max = _round6(summary.get(json_max))
            exp_mean = _round6(summary.get(json_mean))

            rows.append(_chk(f"{csv_col}_min_vs_summary",
                             exp_min, calc_min,
                             note=f"summary.{json_min}={exp_min}"))
            rows.append(_chk(f"{csv_col}_max_vs_summary",
                             exp_max, calc_max,
                             note=f"summary.{json_max}={exp_max}"))
            rows.append(_chk(f"{csv_col}_mean_vs_summary",
                             exp_mean, calc_mean,
                             note=f"summary.{json_mean}={exp_mean}"))
        elif summary is None:
            rows.append({
                "section": "E", "check_item": f"{csv_col}_vs_summary",
                "expected": "N/A", "observed": "summary_missing",
                "status": "FAIL", "note": "summary JSON 없음",
            })
        else:
            rows.append({
                "section": "E", "check_item": f"{csv_col}_vs_summary",
                "expected": "N/A", "observed": "col_missing",
                "status": "FAIL", "note": f"{csv_col} 컬럼 없음",
            })

    # input/recon/error 범위 확인
    range_checks = [
        ("input_min", ">=", 0),
        ("input_max", "<=", 1),
        ("recon_min", ">=", 0),
        ("recon_max", "<=", 1),
        ("error_min", ">=", 0),
    ]
    for col, op, bound in range_checks:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if op == ">=":
                bad = int((series < bound).sum())
            else:
                bad = int((series > bound).sum())
            rows.append(_chk(f"{col}_{op.replace('>', 'le').replace('<', 'ge')}_{bound}_all_rows",
                             0, bad,
                             note="" if bad == 0 else f"범위 위반 행: {bad}"))
        else:
            rows.append({
                "section": "E",
                "check_item": f"{col}_{op}_{bound}_all_rows",
                "expected": 0, "observed": "col_missing",
                "status": "FAIL", "note": f"{col} 컬럼 없음",
            })

    # error_max finite
    if "error_max" in df.columns:
        series = pd.to_numeric(df["error_max"], errors="coerce")
        non_finite = int((~series.apply(lambda x: _is_finite_val(x))).sum())
        rows.append(_chk("error_max_all_finite", 0, non_finite))
    else:
        rows.append({
            "section": "E", "check_item": "error_max_all_finite",
            "expected": 0, "observed": "col_missing",
            "status": "FAIL", "note": "error_max 컬럼 없음",
        })

    return rows


# ---------------------------------------------------------------------------
# Section F: error / runtime validation
# ---------------------------------------------------------------------------

def section_f_error_runtime_validation():
    rows = []

    # errors CSV row count
    if ERRORS_CSV.exists():
        edf = pd.read_csv(ERRORS_CSV)
        err_rows = len(edf)
        rows.append({
            "section": "F", "check_item": "errors_csv_row_count",
            "expected": 0, "observed": err_rows,
            "status": "PASS" if err_rows == 0 else "FAIL",
            "note": "" if err_rows == 0 else f"error 행 {err_rows}개 존재",
        })
    else:
        rows.append({
            "section": "F", "check_item": "errors_csv_row_count",
            "expected": 0, "observed": "file_missing",
            "status": "FAIL", "note": "errors CSV 없음",
        })

    # runtime CSV
    if RUNTIME_CSV.exists():
        rdf = pd.read_csv(RUNTIME_CSV)
        row_count = len(rdf)
        rows.append({
            "section": "F", "check_item": "runtime_csv_row_count",
            "expected": 1, "observed": row_count,
            "status": "PASS" if row_count == 1 else "FAIL",
            "note": "",
        })

        runtime_checks = [
            ("smoke_pass", "True"),
            ("total_processed_rows", "64"),
            ("total_error_rows", "0"),
            ("batch_size", "8"),
        ]
        for col, exp in runtime_checks:
            if col in rdf.columns:
                obs = str(rdf[col].iloc[0]) if row_count >= 1 else "empty"
                rows.append({
                    "section": "F", "check_item": f"runtime_{col}",
                    "expected": exp, "observed": obs,
                    "status": "PASS" if obs == exp else "FAIL",
                    "note": "",
                })
            else:
                rows.append({
                    "section": "F", "check_item": f"runtime_{col}",
                    "expected": exp, "observed": "col_missing",
                    "status": "WARNING", "note": f"{col} 컬럼 없음",
                })

        # output_csv_path → SMOKE_SCORE_CSV 일치 확인
        if "output_csv_path" in rdf.columns and row_count >= 1:
            obs_path = str(rdf["output_csv_path"].iloc[0])
            exp_path = str(SMOKE_SCORE_CSV)
            match = (obs_path == exp_path)
            rows.append({
                "section": "F", "check_item": "runtime_output_csv_path_matches_smoke_score_csv",
                "expected": exp_path, "observed": obs_path,
                "status": "PASS" if match else "FAIL",
                "note": "" if match else "경로 불일치",
            })
        else:
            rows.append({
                "section": "F", "check_item": "runtime_output_csv_path_matches_smoke_score_csv",
                "expected": str(SMOKE_SCORE_CSV), "observed": "col_missing",
                "status": "WARNING", "note": "output_csv_path 컬럼 없음",
            })
    else:
        rows.append({
            "section": "F", "check_item": "runtime_csv_load",
            "expected": "file_exists", "observed": "missing",
            "status": "FAIL", "note": "runtime CSV 없음",
        })

    return rows


# ---------------------------------------------------------------------------
# Section G: output collision / full scoring safety
# ---------------------------------------------------------------------------

def section_g_output_collision():
    rows = []

    # Phase 8.4 full scoring output root 없어야 PASS
    p84_exists = PHASE84_FULL_SCORING_ROOT.exists()
    rows.append({
        "section": "G",
        "item": "phase8_4_full_scoring_root_absent",
        "path": str(PHASE84_FULL_SCORING_ROOT),
        "expected": "absent",
        "observed": "exists" if p84_exists else "absent",
        "status": "PASS" if not p84_exists else "FAIL",
        "note": "Phase 8.4 full scoring output이 이미 존재함" if p84_exists else "",
    })

    # Phase 7.2 score output 존재해야 PASS (untouched)
    p72_exists = PHASE72_SCORE_ROOT.exists()
    rows.append({
        "section": "G",
        "item": "phase7_2_score_root_untouched",
        "path": str(PHASE72_SCORE_ROOT),
        "expected": "exists",
        "observed": "exists" if p72_exists else "absent",
        "status": "PASS" if p72_exists else "FAIL",
        "note": "" if p72_exists else "Phase 7.2 score output이 없음 (삭제 의심)",
    })

    # smoke score CSV read-only 명기
    rows.append({
        "section": "G",
        "item": "smoke_score_csv_not_modified",
        "path": str(SMOKE_SCORE_CSV),
        "expected": "read_only",
        "observed": "read_only",
        "status": "PASS",
        "note": "이번 validation 스크립트는 smoke score CSV를 수정하지 않음 (코드 상 read-only)",
    })

    return rows


# ---------------------------------------------------------------------------
# Section H: readiness decision
# ---------------------------------------------------------------------------

def section_h_readiness(a_rows, b_rows, c_rows, d_rows, e_rows, f_rows, g_rows):
    def _has_fail(rows):
        return any(r.get("status") == "FAIL" for r in rows)

    blocker_map = {
        "A": ("BLOCKED_SMOKE_SCORE_ARTIFACT_MISSING", _has_fail(a_rows)),
        "B": ("BLOCKED_SMOKE_SUMMARY_DONE_MISMATCH", _has_fail(b_rows)),
        "C": ("BLOCKED_SMOKE_SCORE_SCHEMA_MISMATCH", _has_fail(c_rows)),
        "D": ("BLOCKED_SMOKE_SCORE_CONTENT_MISMATCH", _has_fail(d_rows)),
        "E": ("BLOCKED_SMOKE_SCORE_VALUE_INVALID", _has_fail(e_rows)),
        "F": ("BLOCKED_SMOKE_ERROR_RUNTIME_INVALID", _has_fail(f_rows)),
        "G": ("BLOCKED_FULL_SCORING_OUTPUT_CONFLICT", _has_fail(g_rows)),
    }

    blockers = []
    for sec, (code, is_fail) in blocker_map.items():
        if is_fail:
            blockers.append(code)

    if not blockers:
        readiness = "READY_FOR_PHASE8_4_STAGE2_FULL_SCORING_PREFLIGHT"
    else:
        readiness = " | ".join(blockers)

    section_labels = {
        "A": "artifact_existence",
        "B": "summary_done_consistency",
        "C": "score_schema_validation",
        "D": "score_content_validation",
        "E": "score_value_validation",
        "F": "error_runtime_validation",
        "G": "output_collision_safety",
    }

    rows = []
    for sec, (code, is_fail) in blocker_map.items():
        rows.append({
            "section": "H",
            "item": section_labels[sec],
            "status": "FAIL" if is_fail else "PASS",
            "blocker": code if is_fail else "",
            "next_required_action": (
                "blocker 해소 후 Phase 8.3C 재실행" if is_fail
                else "PASS — 다음 섹션 통과"
            ),
        })

    rows.append({
        "section": "H",
        "item": "READINESS_DECISION",
        "status": "PASS" if not blockers else "FAIL",
        "blocker": " | ".join(blockers) if blockers else "",
        "next_required_action": (
            "Phase 8.4 full scoring preflight 진행"
            if not blockers
            else "위 blocker 해소 필요"
        ),
    })

    return rows, readiness, blockers


# ---------------------------------------------------------------------------
# MD report
# ---------------------------------------------------------------------------

def _section_fail_count(rows):
    return sum(1 for r in rows if r.get("status") == "FAIL")


def _section_pass_count(rows):
    return sum(1 for r in rows if r.get("status") == "PASS")


def build_md_report(
    a_rows, b_rows, c_rows, d_rows, e_rows, f_rows, g_rows, h_rows,
    readiness, blockers,
):
    lines = []

    lines.append("# Phase 8.3C Stage2 Scoring Smoke Output Validation Report")
    lines.append("")

    # 1. 목적
    lines.append("## 1. Phase 8.3C 목적")
    lines.append("")
    lines.append("Phase 8.3B smoke scoring 산출물을 검증하여 Phase 8.4 full scoring preflight로 넘어갈 수 있는지 판단한다.")
    lines.append("본 검증은 read-only validation only. model forward / scoring / metric 계산 전혀 없음.")
    lines.append("")

    # 2. Phase 8.3B smoke 결과 요약
    lines.append("## 2. Phase 8.3B Smoke 결과 요약")
    lines.append("")
    if SUMMARY_JSON.exists():
        with open(SUMMARY_JSON, encoding="utf-8") as f:
            summary = json.load(f)
        lines.append(f"- smoke_pass: {summary.get('smoke_pass')}")
        lines.append(f"- scored_crop_count: {summary.get('scored_crop_count')}")
        lines.append(f"- smoke_positive_count: {summary.get('smoke_positive_count')}")
        lines.append(f"- smoke_hard_negative_count: {summary.get('smoke_hard_negative_count')}")
        lines.append(f"- has_nan_count: {summary.get('has_nan_count')}")
        lines.append(f"- has_inf_count: {summary.get('has_inf_count')}")
        lines.append(f"- error_count: {summary.get('error_count')}")
        lines.append(f"- metric_calculation_executed: {summary.get('metric_calculation_executed')}")
        lines.append(f"- full_scoring_executed: {summary.get('full_scoring_executed')}")
    else:
        lines.append("- summary JSON 없음 (로드 실패)")
    lines.append("")

    # 3–9. 섹션 요약
    sections = [
        ("3. Artifact Existence (Section A)", a_rows, ["section", "item", "path", "exists", "status", "note"]),
        ("4. Summary / DONE Consistency (Section B)", b_rows, ["section", "check_item", "expected", "observed", "status", "note"]),
        ("5. Score CSV Schema Validation (Section C)", c_rows, ["section", "column_name", "required", "exists", "status", "note"]),
        ("6. Score CSV Content Validation (Section D)", d_rows, ["section", "check_item", "expected", "observed", "status", "note"]),
        ("7. Score Value Validation (Section E)", e_rows, ["section", "check_item", "expected", "observed", "status", "note"]),
        ("8. Error / Runtime Validation (Section F)", f_rows, ["section", "check_item", "expected", "observed", "status", "note"]),
        ("9. Output Collision / Full Scoring Safety (Section G)", g_rows, ["section", "item", "path", "expected", "observed", "status", "note"]),
    ]
    for title, rows, cols in sections:
        lines.append(f"## {title}")
        lines.append("")
        fail_n = _section_fail_count(rows)
        pass_n = _section_pass_count(rows)
        lines.append(f"PASS: {pass_n} / FAIL: {fail_n}")
        lines.append("")
        # 테이블
        avail_cols = [c for c in cols if any(c in r for r in rows)]
        if avail_cols:
            lines.append("| " + " | ".join(avail_cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(avail_cols)) + " |")
            for r in rows:
                cells = [str(r.get(c, "")) for c in avail_cols]
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # 10. readiness 판정
    lines.append("## 10. Readiness 판정")
    lines.append("")
    lines.append(f"**{readiness}**")
    lines.append("")
    if blockers:
        lines.append("### Blockers")
        for b in blockers:
            lines.append(f"- {b}")
        lines.append("")

    # 11. 다음 단계
    lines.append("## 11. 다음 단계")
    lines.append("")
    if not blockers:
        lines.append("- Phase 8.4 full scoring preflight 진행")
        lines.append("- Phase 8.4에서 full manifest 전체 대상 scoring 사전 확인 수행")
    else:
        lines.append("- 위 blocker 항목 해소 후 Phase 8.3C validation 재실행")
        lines.append("- 재실행 전 OUT_DIR 삭제 필요 (output guard 적용됨)")
    lines.append("")

    # 12. 금지 사항
    lines.append("## 12. 금지 사항")
    lines.append("")
    lines.append("- model forward / scoring 실행 금지")
    lines.append("- metric 계산 금지")
    lines.append("- threshold 계산 금지")
    lines.append("- 학습 (training) 실행 금지")
    lines.append("- backward / optimizer step 실행 금지")
    lines.append("- checkpoint 생성 금지")
    lines.append("- full scoring 실행 금지")
    lines.append("- 기존 smoke score CSV 수정 금지")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 8.3C: Stage2 Scoring Smoke Output Validation"
    )
    parser.add_argument("--run", action="store_true", help="실제 검증 수행 (없으면 dry-run)")
    args = parser.parse_args()

    if not args.run:
        print("=" * 70)
        print("[DRY-RUN] Phase 8.3C Stage2 Scoring Smoke Output Validation")
        print("=" * 70)
        print()
        print("실제 검증을 수행하려면 --run 플래그를 추가하세요.")
        print()
        print("입력 경로 (모두 read-only):")
        print(f"  SMOKE_SCORE_CSV : {SMOKE_SCORE_CSV}")
        print(f"  SUMMARY_JSON    : {SUMMARY_JSON}")
        print(f"  ERRORS_CSV      : {ERRORS_CSV}")
        print(f"  RUNTIME_CSV     : {RUNTIME_CSV}")
        print(f"  DONE_MARKER     : {DONE_MARKER}")
        print(f"  SMOKE_REPORT_MD : {SMOKE_REPORT_MD}")
        print(f"  STAGE2_MANIFEST : {STAGE2_MANIFEST}")
        print()
        print("출력 경로:")
        print(f"  OUT_DIR : {OUT_DIR}")
        print(f"  OUT_CSV : {OUT_CSV}")
        print(f"  OUT_JSON: {OUT_JSON}")
        print(f"  OUT_MD  : {OUT_MD}")
        print()
        print("[주의] OUT_DIR이 이미 존재하면 실행 즉시 종료됩니다.")
        print()
        print("검증 항목: A(artifact) B(summary/DONE) C(schema) D(content) E(value) F(error/runtime) G(collision) H(readiness)")
        sys.exit(0)

    # output guard
    if OUT_DIR.exists():
        print(f"[ERROR] OUT_DIR이 이미 존재합니다. 중복 실행 방지를 위해 종료합니다.")
        print(f"  {OUT_DIR}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] OUT_DIR 생성: {OUT_DIR}")

    print("[Phase 8.3C] 검증 시작...")

    # 각 섹션 실행
    print("  Section A: artifact existence...")
    a_rows = section_a_artifact_existence()

    print("  Section B: summary/DONE consistency...")
    b_rows = section_b_summary_done_consistency()

    print("  Section C: score CSV schema...")
    c_rows = section_c_score_csv_schema()

    print("  Section D: score CSV content...")
    d_rows = section_d_score_csv_content()

    print("  Section E: score value validation...")
    e_rows = section_e_score_value_validation()

    print("  Section F: error/runtime validation...")
    f_rows = section_f_error_runtime_validation()

    print("  Section G: output collision / full scoring safety...")
    g_rows = section_g_output_collision()

    print("  Section H: readiness decision...")
    h_rows, readiness, blockers = section_h_readiness(
        a_rows, b_rows, c_rows, d_rows, e_rows, f_rows, g_rows
    )

    # ---------------------------------------------------------------------------
    # CSV 저장 (모든 섹션 concat)
    # ---------------------------------------------------------------------------
    all_dfs = []
    for rows in [a_rows, b_rows, c_rows, d_rows, e_rows, f_rows, g_rows, h_rows]:
        if rows:
            all_dfs.append(pd.DataFrame(rows))
    combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)

    # 저장 직전 파일 재검증
    if OUT_CSV.exists():
        print(f"[ERROR] OUT_CSV가 이미 존재합니다. 중단합니다: {OUT_CSV}")
        sys.exit(1)
    combined_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"  [저장] {OUT_CSV}")

    # ---------------------------------------------------------------------------
    # JSON 저장
    # ---------------------------------------------------------------------------
    out_json_data = {
        "input_paths": {
            "SMOKE_SCORE_CSV": str(SMOKE_SCORE_CSV),
            "SUMMARY_JSON": str(SUMMARY_JSON),
            "ERRORS_CSV": str(ERRORS_CSV),
            "RUNTIME_CSV": str(RUNTIME_CSV),
            "DONE_MARKER": str(DONE_MARKER),
            "SMOKE_REPORT_MD": str(SMOKE_REPORT_MD),
            "STAGE2_MANIFEST": str(STAGE2_MANIFEST),
        },
        "artifact_existence": a_rows,
        "summary_done_consistency": b_rows,
        "score_csv_schema_validation": c_rows,
        "score_csv_content_validation": d_rows,
        "score_value_validation": e_rows,
        "error_runtime_validation": f_rows,
        "output_collision_full_scoring_safety": g_rows,
        "readiness_for_phase8_4": readiness,
        "blockers": blockers,
        "notes": {
            "validation_only": True,
            "no_model_forward": True,
            "no_scoring_rerun": True,
            "no_metric_calculation": True,
            "no_threshold": True,
            "no_training": True,
            "no_full_scoring": True,
            "no_existing_output_modification": True,
        },
    }

    if OUT_JSON.exists():
        print(f"[ERROR] OUT_JSON이 이미 존재합니다. 중단합니다: {OUT_JSON}")
        sys.exit(1)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_json_data, f, ensure_ascii=False, indent=2)
    print(f"  [저장] {OUT_JSON}")

    # ---------------------------------------------------------------------------
    # MD report 저장
    # ---------------------------------------------------------------------------
    md_content = build_md_report(
        a_rows, b_rows, c_rows, d_rows, e_rows, f_rows, g_rows, h_rows,
        readiness, blockers,
    )

    if OUT_MD.exists():
        print(f"[ERROR] OUT_MD가 이미 존재합니다. 중단합니다: {OUT_MD}")
        sys.exit(1)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  [저장] {OUT_MD}")

    # ---------------------------------------------------------------------------
    # 콘솔 최종 출력
    # ---------------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"[Phase 8.3C] Readiness 판정:")
    print(f"  {readiness}")
    if blockers:
        print()
        print("  Blockers:")
        for b in blockers:
            print(f"    - {b}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
