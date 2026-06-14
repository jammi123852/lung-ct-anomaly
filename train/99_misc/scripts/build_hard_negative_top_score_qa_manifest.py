#!/usr/bin/env python3
"""
Phase 5.38: Hard Negative Top-Score QA Manifest 생성 script

이 script는 hard_negative_scores_v1.csv를 read-only로 읽어
QA 검토 우선순위 manifest를 생성합니다.

주의:
- threshold 확정 아님
- 병변 성능 결론 금지
- stage2_holdout 미사용
- v2 미사용
- 원본 volume 미접근
- crop 복사 없음
- PNG/overlay 생성 없음
"""

import argparse
import json
import sys
import os
import pandas as pd
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

EXPECTED_ROW_COUNT = 7700
EXPECTED_PATIENT_COUNT = 154

REQUIRED_COLUMNS = [
    "patient_id",
    "safe_id",
    "crop_id",
    "crop_path",
    "crop_score_l1_mean",
    "crop_score_l1_max",
    "crop_score_mse_mean",
    "threshold_exceed_val_p90",
    "threshold_exceed_val_p95",
    "threshold_exceed_val_p99",
]

OPTIONAL_COLUMNS = [
    "source_candidate_id",
    "rd4ad_label",
    "binary_label",
    "original_candidate_role",
    "padim_score_mean",
    "padim_score_max",
    "large_bbox_flag",
    "zero_lc_patient_flag",
    "weak_case_flag",
]

OUTPUT_MANIFEST_COLUMNS = [
    "patient_id",
    "safe_id",
    "crop_id",
    "crop_path",
    "crop_score_l1_mean",
    "crop_score_l1_max",
    "crop_score_mse_mean",
    "threshold_exceed_val_p90",
    "threshold_exceed_val_p95",
    "threshold_exceed_val_p99",
    "padim_score_mean",
    "padim_score_max",
    "large_bbox_flag",
    "rd4ad_label",
    "binary_label",
    "original_candidate_role",
    "qa_group",
    "qa_priority",
    "suggested_review_reason",
    "manual_label_placeholder",
]

QA_PRIORITY_MAP = {
    "keep_high_priority": 1,
    "review_group": 2,
    "low_priority": 3,
    "defer": 4,
}

RUNTIME_SUMMARY_KEY = "phase5_38_hard_negative_top_score_qa_manifest"


# ---------------------------------------------------------------------------
# 안전장치 함수
# ---------------------------------------------------------------------------

def preflight_check(score_csv_path: str, summary_json_path: str, df: pd.DataFrame):
    """입력 파일 및 데이터 기본 안전장치 확인."""
    errors = []
    warnings = []

    # 금지 경로 확인 (대소문자 무관)
    _BLOCKED_PATH_KEYWORDS = ["stage2_holdout", "holdout", "v2"]
    for path_str, label in [(score_csv_path, "score CSV"), (summary_json_path, "summary JSON")]:
        path_lower = path_str.lower()
        for kw in _BLOCKED_PATH_KEYWORDS:
            if kw in path_lower:
                errors.append(f"[BLOCKED] {label} 경로에 '{kw}' 포함 — 사용 금지: {path_str}")

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    # crop_path 값 확인 (확장자 제한)
    if "crop_path" in df.columns:
        crop_paths = df["crop_path"].dropna().astype(str)
        # 비어있는 값 제외 후 검사
        non_empty = crop_paths[crop_paths.str.strip() != ""]
        # .nii/.nii.gz 원본 volume 경로 차단
        nii_mask = non_empty.str.contains(r"\.nii(?:\.gz)?$", regex=True, case=False)
        if nii_mask.any():
            bad_paths = non_empty[nii_mask].head(3).tolist()
            errors.append(
                f"[BLOCKED] crop_path에 .nii 또는 .nii.gz 경로 포함 — 원본 volume 접근 금지. 예시: {bad_paths}"
            )
        # v2 경로 차단 (전체 crop_path 검사)
        v2_mask = non_empty.str.lower().str.contains("v2", na=False)
        if v2_mask.any():
            v2_samples = non_empty[v2_mask].head(5).tolist()
            errors.append(
                f"[BLOCKED] crop_path에 'v2' 포함 경로 발견 ({v2_mask.sum()}건) — v2 volume source는 현재 Phase 5.38에서 사용하지 않음. 예시: {v2_samples}"
            )
        # .npz 이외 확장자는 중단 (crop_path는 .npz만 허용)
        non_npz = non_empty[~non_empty.str.endswith(".npz")]
        if not non_npz.empty:
            non_npz_sample = non_npz.head(3).tolist()
            errors.append(
                f"[BLOCKED] crop_path 중 .npz 이외 확장자 포함 ({len(non_npz)}건) — crop_path는 .npz만 허용. 예시: {non_npz_sample}"
            )

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    # row 수 확인
    if len(df) != EXPECTED_ROW_COUNT:
        warnings.append(
            f"[WARNING] CSV row 수 불일치. 기대={EXPECTED_ROW_COUNT}, 실제={len(df)}"
        )

    # patient 수 확인
    if "patient_id" in df.columns:
        n_patients = df["patient_id"].nunique()
        if n_patients != EXPECTED_PATIENT_COUNT:
            warnings.append(
                f"[WARNING] patient 수 불일치. 기대={EXPECTED_PATIENT_COUNT}, 실제={n_patients}"
            )

    for w in warnings:
        print(w)

    return warnings


def check_required_columns(df: pd.DataFrame):
    """필수 컬럼 존재 여부 확인. 없으면 SystemExit."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[ERROR] 필수 컬럼 누락: {missing}", file=sys.stderr)
        sys.exit(1)


def resolve_optional_columns(df: pd.DataFrame):
    """optional 컬럼 present/missing 분류, 없는 컬럼은 NaN으로 추가."""
    present = []
    missing = []
    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            present.append(col)
        else:
            missing.append(col)
            df[col] = float("nan")
    return present, missing


# ---------------------------------------------------------------------------
# truthy 판정 헬퍼
# ---------------------------------------------------------------------------

def to_bool_series(series: pd.Series) -> pd.Series:
    """다양한 형태의 truthy 값을 bool로 변환."""
    def _to_bool(v):
        if pd.isna(v):
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        s = str(v).strip().lower()
        return s in ("true", "1", "yes")

    return series.apply(_to_bool)


# ---------------------------------------------------------------------------
# QA group 생성
# ---------------------------------------------------------------------------

def build_qa_groups(
    df: pd.DataFrame,
    summary: dict,
    top_k_crops: int,
    top_k_patients: int,
    top_k_per_patient: int,
    large_bbox_limit: int,
    padim_high_quantile: float,
    include_p90_exceed: bool,
    include_p95_exceed: bool,
    optional_missing: list,
) -> pd.DataFrame:
    """
    각 row에 qa_group (list), qa_priority, suggested_review_reason을 부여.
    반환: 원본 df에 컬럼 추가된 복사본.
    """
    df = df.copy()

    val_threshold_p90 = summary.get("val_threshold_p90", None)

    # qa_group, reason 초기화 (list 형태)
    df["_qa_groups"] = [[] for _ in range(len(df))]
    df["_qa_reasons"] = [[] for _ in range(len(df))]

    # --- HN-p95-exceed ---
    if include_p95_exceed:
        p95_mask = to_bool_series(df["threshold_exceed_val_p95"])
        for idx in df.index[p95_mask]:
            df.at[idx, "_qa_groups"].append("HN-p95-exceed")
            df.at[idx, "_qa_reasons"].append("p95_exceed_high_reconstruction_score")

    # --- HN-p90-exceed (p95 False인 것만) ---
    if include_p90_exceed:
        p90_mask = to_bool_series(df["threshold_exceed_val_p90"])
        p95_mask = to_bool_series(df["threshold_exceed_val_p95"])
        p90_only_mask = p90_mask & ~p95_mask
        for idx in df.index[p90_only_mask]:
            df.at[idx, "_qa_groups"].append("HN-p90-exceed")
            df.at[idx, "_qa_reasons"].append("p90_exceed_candidate")

    # --- HN-top10-crops ---
    top_crops_idx = df.nlargest(top_k_crops, "crop_score_l1_mean").index
    for idx in top_crops_idx:
        df.at[idx, "_qa_groups"].append("HN-top10-crops")
        df.at[idx, "_qa_reasons"].append("top_score_crop")

    # --- HN-top10-patients ---
    # summary top_score_patients 활용, 없으면 CSV에서 계산
    top_patient_ids = None
    if "top_score_patients" in summary and isinstance(summary["top_score_patients"], list):
        top_patient_ids = [
            item["patient_id"]
            for item in summary["top_score_patients"][:top_k_patients]
            if "patient_id" in item
        ]
    if not top_patient_ids:
        # CSV에서 환자별 max score로 계산
        patient_max = df.groupby("patient_id")["crop_score_l1_mean"].max()
        top_patient_ids = patient_max.nlargest(top_k_patients).index.tolist()

    for pid in top_patient_ids:
        patient_df = df[df["patient_id"] == pid]
        top_per_patient_idx = patient_df.nlargest(top_k_per_patient, "crop_score_l1_mean").index
        for idx in top_per_patient_idx:
            df.at[idx, "_qa_groups"].append("HN-top10-patients")
            df.at[idx, "_qa_reasons"].append("top_score_patient_crop")

    # --- HN-large-bbox ---
    if "large_bbox_flag" not in optional_missing:
        large_bbox_bool = to_bool_series(df["large_bbox_flag"])
        large_bbox_df = df[large_bbox_bool]
        if not large_bbox_df.empty:
            top_large_bbox_idx = large_bbox_df.nlargest(large_bbox_limit, "crop_score_l1_mean").index
            for idx in top_large_bbox_idx:
                df.at[idx, "_qa_groups"].append("HN-large-bbox")
                df.at[idx, "_qa_reasons"].append("large_bbox_candidate")
    # large_bbox_flag 없으면 skip (summary에서는 missing으로 기록됨)

    # --- HN-padim-high-rd4ad-low ---
    padim_missing = "padim_score_mean" in optional_missing and "padim_score_max" in optional_missing
    if not padim_missing and val_threshold_p90 is not None:
        # padim_score_mean 또는 padim_score_max 유효한 컬럼 선택
        if "padim_score_mean" not in optional_missing:
            padim_col = "padim_score_mean"
        else:
            padim_col = "padim_score_max"

        padim_series = pd.to_numeric(df[padim_col], errors="coerce")
        valid_padim = padim_series.dropna()
        if not valid_padim.empty:
            padim_threshold = valid_padim.quantile(padim_high_quantile)
            padim_high_mask = padim_series >= padim_threshold
            rd4ad_low_mask = df["crop_score_l1_mean"] < val_threshold_p90
            combo_mask = padim_high_mask & rd4ad_low_mask
            for idx in df.index[combo_mask]:
                df.at[idx, "_qa_groups"].append("HN-padim-high-rd4ad-low")
                df.at[idx, "_qa_reasons"].append("padim_high_rd4ad_low_disagreement")
    # padim 컬럼 없으면 skip

    return df


# ---------------------------------------------------------------------------
# qa_priority 결정
# ---------------------------------------------------------------------------

def assign_qa_priority(groups: list) -> tuple:
    """
    qa_group 리스트를 받아 (priority_label, priority_num) 반환.
    """
    if not groups:
        return "defer", QA_PRIORITY_MAP["defer"]

    high_groups = {"HN-p95-exceed", "HN-top10-crops"}
    review_groups = {"HN-p90-exceed", "HN-top10-patients", "HN-large-bbox"}
    low_groups = {"HN-padim-high-rd4ad-low"}

    group_set = set(groups)

    if group_set & high_groups:
        return "keep_high_priority", QA_PRIORITY_MAP["keep_high_priority"]
    elif group_set & review_groups:
        return "review_group", QA_PRIORITY_MAP["review_group"]
    elif group_set & low_groups:
        return "low_priority", QA_PRIORITY_MAP["low_priority"]
    else:
        return "defer", QA_PRIORITY_MAP["defer"]


# ---------------------------------------------------------------------------
# 중복 제거
# ---------------------------------------------------------------------------

def deduplicate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    crop_id 기준 중복 제거.
    crop_id 없거나 NaN이면 patient_id + crop_path 기준 보조 중복 제거.
    중복 시 suggested_review_reason, qa_group 합치기, qa_priority는 낮은 숫자 유지.
    """

    def merge_groups(series):
        all_groups = []
        for val in series:
            if isinstance(val, list):
                all_groups.extend(val)
            elif isinstance(val, str) and val:
                all_groups.extend([g.strip() for g in val.split(";") if g.strip()])
        return list(dict.fromkeys(all_groups))  # 순서 유지 중복 제거

    def merge_reasons(series):
        all_reasons = []
        for val in series:
            if isinstance(val, list):
                all_reasons.extend(val)
            elif isinstance(val, str) and val:
                all_reasons.extend([r.strip() for r in val.split(";") if r.strip()])
        return list(dict.fromkeys(all_reasons))

    def min_priority(series):
        nums = [v for v in series if isinstance(v, (int, float)) and not pd.isna(v)]
        return min(nums) if nums else QA_PRIORITY_MAP["defer"]

    # crop_id 유효성 확인
    has_crop_id = "crop_id" in df.columns and df["crop_id"].notna().any()

    if has_crop_id:
        key_col = "crop_id"
        valid_mask = df[key_col].notna() & (df[key_col].astype(str).str.strip() != "")
        df_valid = df[valid_mask].copy()
        df_invalid = df[~valid_mask].copy()
    else:
        key_col = None
        df_valid = df.copy()
        df_invalid = pd.DataFrame(columns=df.columns)

    if key_col and not df_valid.empty:
        # 그룹별 병합
        agg_dict = {}
        for col in df_valid.columns:
            if col == "_qa_groups":
                agg_dict[col] = merge_groups
            elif col == "_qa_reasons":
                agg_dict[col] = merge_reasons
            elif col == "_qa_priority_num":
                agg_dict[col] = min_priority
            elif col == "_qa_priority_label":
                agg_dict[col] = "first"
            else:
                agg_dict[col] = "first"

        df_valid = df_valid.groupby(key_col, as_index=False, sort=False).agg(agg_dict)

        # priority label 재결정 (num 기반)
        def num_to_label(n):
            for label, num in QA_PRIORITY_MAP.items():
                if num == n:
                    return label
            return "defer"

        df_valid["_qa_priority_label"] = df_valid["_qa_priority_num"].apply(num_to_label)

    # invalid rows: patient_id + crop_path 기준 보조 중복 제거
    if not df_invalid.empty:
        # 보조 키
        df_invalid["_dedup_key"] = (
            df_invalid["patient_id"].astype(str) + "||" + df_invalid["crop_path"].astype(str)
        )
        agg_dict2 = {}
        for col in df_invalid.columns:
            if col == "_dedup_key":
                continue
            elif col == "_qa_groups":
                agg_dict2[col] = merge_groups
            elif col == "_qa_reasons":
                agg_dict2[col] = merge_reasons
            elif col == "_qa_priority_num":
                agg_dict2[col] = min_priority
            elif col == "_qa_priority_label":
                agg_dict2[col] = "first"
            else:
                agg_dict2[col] = "first"

        df_invalid = df_invalid.groupby("_dedup_key", as_index=False, sort=False).agg(agg_dict2)
        df_invalid = df_invalid.drop(columns=["_dedup_key"], errors="ignore")

        df_valid["_qa_priority_label"] = df_valid["_qa_priority_num"].apply(num_to_label)

    result = pd.concat([df_valid, df_invalid], ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# max-total 제한 적용
# ---------------------------------------------------------------------------

def apply_max_total(df: pd.DataFrame, max_total: int) -> pd.DataFrame:
    """
    max_total 초과 시 우선순위 순으로 자르기.
    keep_high_priority(1) > review_group(2) > low_priority(3) > defer(4)
    """
    if len(df) <= max_total:
        return df

    result_parts = []
    remaining = max_total

    for priority_num in [1, 2, 3, 4]:
        part = df[df["_qa_priority_num"] == priority_num]
        if part.empty:
            continue
        if len(part) <= remaining:
            result_parts.append(part)
            remaining -= len(part)
        else:
            # score 높은 순으로 자르기
            part_sorted = part.sort_values("crop_score_l1_mean", ascending=False)
            result_parts.append(part_sorted.head(remaining))
            remaining = 0
            break

    if not result_parts:
        return df.head(0)

    return pd.concat(result_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 출력 manifest DataFrame 정리
# ---------------------------------------------------------------------------

def finalize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """_qa_groups, _qa_reasons, _qa_priority_num/label 을 출력 컬럼으로 변환."""
    df = df.copy()

    # qa_group: 세미콜론 연결 문자열
    df["qa_group"] = df["_qa_groups"].apply(
        lambda lst: ";".join(lst) if isinstance(lst, list) else str(lst)
    )

    # suggested_review_reason: 세미콜론 연결 문자열
    df["suggested_review_reason"] = df["_qa_reasons"].apply(
        lambda lst: ";".join(lst) if isinstance(lst, list) else str(lst)
    )

    # qa_priority: 레이블 문자열
    df["qa_priority"] = df.get("_qa_priority_label", "defer")

    # manual_label_placeholder
    df["manual_label_placeholder"] = ""

    # 출력 컬럼만 선택 (없는 컬럼은 빈 문자열)
    for col in OUTPUT_MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[OUTPUT_MANIFEST_COLUMNS]


# ---------------------------------------------------------------------------
# summary JSON 생성
# ---------------------------------------------------------------------------

def build_summary_json(
    score_csv_path: str,
    summary_json_path: str,
    df_input: pd.DataFrame,
    df_selected: pd.DataFrame,
    optional_present: list,
    optional_missing: list,
    args,
) -> dict:
    n_by_group = {}
    for _, row in df_selected.iterrows():
        groups = row.get("qa_group", "")
        if isinstance(groups, str):
            for g in groups.split(";"):
                g = g.strip()
                if g:
                    n_by_group[g] = n_by_group.get(g, 0) + 1

    n_by_priority = {}
    for _, row in df_selected.iterrows():
        p = row.get("qa_priority", "defer")
        n_by_priority[p] = n_by_priority.get(p, 0) + 1

    n_p90 = int(to_bool_series(df_input["threshold_exceed_val_p90"]).sum()) if "threshold_exceed_val_p90" in df_input.columns else 0
    n_p95 = int(to_bool_series(df_input["threshold_exceed_val_p95"]).sum()) if "threshold_exceed_val_p95" in df_input.columns else 0
    n_p99 = int(to_bool_series(df_input["threshold_exceed_val_p99"]).sum()) if "threshold_exceed_val_p99" in df_input.columns else 0

    return {
        "input_score_csv": score_csv_path,
        "input_score_summary": summary_json_path,
        "n_input_rows": len(df_input),
        "n_input_patients": int(df_input["patient_id"].nunique()) if "patient_id" in df_input.columns else -1,
        "n_selected_rows": len(df_selected),
        "n_selected_patients": int(df_selected["patient_id"].nunique()) if "patient_id" in df_selected.columns else -1,
        "n_p90_exceed_input": n_p90,
        "n_p95_exceed_input": n_p95,
        "n_p99_exceed_input": n_p99,
        "n_selected_by_group": n_by_group,
        "n_selected_by_priority": n_by_priority,
        "optional_columns_present": optional_present,
        "optional_columns_missing": optional_missing,
        "max_total": args.max_total,
        "top_k_crops": args.top_k_crops,
        "top_k_patients": args.top_k_patients,
        "top_k_per_patient": args.top_k_per_patient,
        "note": {
            "qa_manifest_purpose": "QA manifest는 검토 후보 축소용",
            "threshold": "threshold 확정 아님",
            "lesion_conclusion": "병변 성능 결론 금지",
            "crop_copy": "crop 복사 없음",
            "png_overlay": "PNG/overlay 생성 없음",
            "stage2_holdout": "stage2_holdout 미사용",
            "v2": "v2 미사용",
            "original_volume": "원본 volume 미사용",
        },
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# review plan MD 생성
# ---------------------------------------------------------------------------

def build_review_plan_md(
    df_selected: pd.DataFrame,
    summary_dict: dict,
    optional_missing: list,
) -> str:
    n_by_group = summary_dict.get("n_selected_by_group", {})
    n_by_priority = summary_dict.get("n_selected_by_priority", {})
    n_selected = summary_dict.get("n_selected_rows", 0)
    n_patients = summary_dict.get("n_selected_patients", 0)
    ts = summary_dict.get("timestamp", "")

    group_lines = "\n".join(
        f"  - {g}: {c}건" for g, c in sorted(n_by_group.items())
    )
    priority_lines = "\n".join(
        f"  - {p}: {c}건" for p, c in sorted(n_by_priority.items(), key=lambda x: QA_PRIORITY_MAP.get(x[0], 99))
    )

    missing_note = ""
    if optional_missing:
        missing_note = f"\n- optional 컬럼 없음 (skip됨): {', '.join(optional_missing)}"

    md = f"""# Phase 5.38 Hard Negative Top-Score QA Review Plan

생성일시: {ts}

---

## QA 목적

- hard_negative_scores_v1.csv 에서 이상 후보 축소 및 우선 검토 대상 선정
- 각 crop의 시각적 확인 후 manual_label_placeholder 기입
- **threshold 확정 아님** — 이 manifest는 검토 후보 범위를 줄이기 위한 도구임
- **병변 성능 결론 금지** — 검토 결과를 바탕으로 단정적 성능 평가 금지

---

## 후보군별 수

선택된 전체 crop: {n_selected}건 / {n_patients}명 환자

### QA Group별
{group_lines}

### 우선순위별
{priority_lines}
{missing_note}

---

## 우선 검토 순서

1. **keep_high_priority** — p95 초과 또는 top score crops ({n_by_priority.get('keep_high_priority', 0)}건)
2. **review_group** — p90 초과, top patient crop, large bbox ({n_by_priority.get('review_group', 0)}건)
3. **low_priority** — padim high / rd4ad low 불일치 ({n_by_priority.get('low_priority', 0)}건)
4. **defer** — 정보 부족 또는 규칙 외 ({n_by_priority.get('defer', 0)}건)

---

## Manual Label 후보 목록

manual_label_placeholder 컬럼에 아래 레이블 중 하나를 기입:

| 레이블 | 설명 |
|--------|------|
| vessel_branch | 혈관 분지 구조 |
| elongated_vessel | 길게 뻗은 혈관 |
| pleural_wall | 흉막/벽 구조 |
| bronchus_air_boundary | 기관지-공기 경계 |
| nodule_suspect | 결절 의심 |
| large_bbox_structure | 큰 bbox 구조물 |
| fragmented_small_objects | 작은 파편 구조물 |
| irregular_large_object | 불규칙 큰 구조물 |
| unclear | 판단 불가 |

---

## 주의사항

- **threshold 확정 아님**: p90/p95/p99 초과 수치는 검토용이며 최종 threshold 결정이 아님
- **병변 성능 결론 금지**: 이 manifest만으로 병변 탐지 성능을 평가하지 않음
- crop 복사 없음, PNG/overlay 생성 없음
- stage2_holdout 미사용, v2 미사용, 원본 volume 미접근
"""
    return md


# ---------------------------------------------------------------------------
# runtime summary append
# ---------------------------------------------------------------------------

def append_runtime_summary(runtime_summary_path: str, n_selected: int, output_root: str):
    """runtime_summary_v1.json에 이번 실행 기록 append."""
    summary_path = Path(runtime_summary_path)
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    data[RUNTIME_SUMMARY_KEY] = {
        "timestamp": datetime.now().isoformat(),
        "n_selected": n_selected,
        "output_root": str(output_root),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] runtime summary append 완료: {runtime_summary_path}")


# ---------------------------------------------------------------------------
# 메인 처리 함수
# ---------------------------------------------------------------------------

def run_selection(args, df: pd.DataFrame, summary: dict, optional_present: list, optional_missing: list):
    """
    QA group 부여 -> priority 결정 -> 중복 제거 -> max-total 제한 순으로 처리.
    반환: 최종 선택 DataFrame (내부 컬럼 포함)
    """
    include_p90 = not args.no_p90_exceed
    include_p95 = not args.no_p95_exceed

    df_work = build_qa_groups(
        df=df,
        summary=summary,
        top_k_crops=args.top_k_crops,
        top_k_patients=args.top_k_patients,
        top_k_per_patient=args.top_k_per_patient,
        large_bbox_limit=args.large_bbox_limit,
        padim_high_quantile=args.padim_high_quantile,
        include_p90_exceed=include_p90,
        include_p95_exceed=include_p95,
        optional_missing=optional_missing,
    )

    # qa_group 없는 row는 제외 (아무 그룹에도 속하지 않으면 불필요)
    has_group_mask = df_work["_qa_groups"].apply(lambda lst: len(lst) > 0)
    df_candidates = df_work[has_group_mask].copy()

    # priority 결정
    priority_results = df_candidates["_qa_groups"].apply(assign_qa_priority)
    df_candidates["_qa_priority_label"] = priority_results.apply(lambda x: x[0])
    df_candidates["_qa_priority_num"] = priority_results.apply(lambda x: x[1])

    # 중복 제거
    df_dedup = deduplicate_candidates(df_candidates)

    # max-total 제한
    df_final = apply_max_total(df_dedup, args.max_total)

    return df_final


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 5.38: Hard Negative Top-Score QA Manifest 생성"
    )
    parser.add_argument(
        "--score-csv",
        default=(
            "outputs/second-stage-lesion-refiner-v1/evaluation/"
            "rd4ad_2p5d_normal_mw_fixed96_v1/hard_negative_scores_v1/"
            "hard_negative_scores_v1.csv"
        ),
        help="입력 hard negative score CSV 경로",
    )
    parser.add_argument(
        "--score-summary",
        default=(
            "outputs/second-stage-lesion-refiner-v1/evaluation/"
            "rd4ad_2p5d_normal_mw_fixed96_v1/hard_negative_scores_v1/"
            "hard_negative_score_summary_v1.json"
        ),
        help="입력 hard negative score summary JSON 경로",
    )
    parser.add_argument(
        "--output-root",
        default=(
            "outputs/second-stage-lesion-refiner-v1/evaluation/"
            "rd4ad_2p5d_normal_mw_fixed96_v1/hard_negative_top_score_qa_v1/"
        ),
        help="출력 루트 디렉토리 경로",
    )
    parser.add_argument("--max-total", type=int, default=150, help="최대 manifest 수 (기본 150)")
    parser.add_argument("--top-k-crops", type=int, default=10, help="top score crops 수 (기본 10)")
    parser.add_argument("--top-k-patients", type=int, default=10, help="top score patients 수 (기본 10)")
    parser.add_argument("--top-k-per-patient", type=int, default=3, help="환자당 선택 crops 수 (기본 3)")
    parser.add_argument("--no-p90-exceed", action="store_true", help="p90 초과 후보 포함 비활성화")
    parser.add_argument("--no-p95-exceed", action="store_true", help="p95 초과 후보 포함 비활성화")
    parser.add_argument("--large-bbox-limit", type=int, default=30, help="large bbox 후보 최대 수 (기본 30)")
    parser.add_argument("--padim-high-quantile", type=float, default=0.95, help="padim high 분위수 (기본 0.95)")
    parser.add_argument("--preflight-only", action="store_true", help="입력 파일 확인만 (파일 생성 없음)")
    parser.add_argument("--dry-run", action="store_true", help="후보 선택까지만 수행 (파일 생성 없음)")
    parser.add_argument("--force", action="store_true", help="output root 충돌 시 강제 덮어쓰기")
    parser.add_argument("--no-runtime-append", action="store_true", help="runtime summary append 비활성화")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    score_csv_path = str(args.score_csv)
    summary_json_path = str(args.score_summary)
    output_root = Path(args.output_root)

    # --- output_root 금지 경로 확인 ---
    _output_root_lower = str(args.output_root).lower()
    for _kw in ["stage2_holdout", "holdout", "v2"]:
        if _kw in _output_root_lower:
            print(
                f"[BLOCKED] output_root 경로에 '{_kw}' 포함 — 사용 금지: {args.output_root}",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- 입력 파일 존재 확인 ---
    if not os.path.isfile(score_csv_path):
        print(f"[ERROR] score CSV 파일 없음: {score_csv_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(summary_json_path):
        print(f"[ERROR] summary JSON 파일 없음: {summary_json_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] score CSV 로드: {score_csv_path}")
    df = pd.read_csv(score_csv_path)
    print(f"[INFO] 로드 완료 — rows={len(df)}, columns={len(df.columns)}")

    print(f"[INFO] summary JSON 로드: {summary_json_path}")
    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    print(f"[INFO] summary JSON 로드 완료")

    # --- preflight 체크 ---
    warnings = preflight_check(score_csv_path, summary_json_path, df)

    # --- 필수 컬럼 확인 ---
    check_required_columns(df)
    print(f"[INFO] 필수 컬럼 확인 완료")

    # --- optional 컬럼 처리 ---
    optional_present, optional_missing = resolve_optional_columns(df)
    print(f"[INFO] optional 컬럼 present={optional_present}")
    print(f"[INFO] optional 컬럼 missing={optional_missing}")

    # --- p90/p95/p99 exceed count 출력 ---
    n_p90 = int(to_bool_series(df["threshold_exceed_val_p90"]).sum())
    n_p95 = int(to_bool_series(df["threshold_exceed_val_p95"]).sum())
    n_p99 = int(to_bool_series(df["threshold_exceed_val_p99"]).sum())
    print(f"[INFO] exceed count — p90={n_p90}, p95={n_p95}, p99={n_p99}")

    # --- preflight-only 종료 ---
    if args.preflight_only:
        # output 충돌은 경고만
        if output_root.exists():
            print(f"[WARNING] output root 이미 존재: {output_root}")
        print("[INFO] preflight-only 완료. 파일 생성 없음.")
        return

    # --- output 충돌 확인 (full run 전용) ---
    if not args.dry_run:
        if output_root.exists() and not args.force:
            print(
                f"[ERROR] output root 이미 존재: {output_root}\n"
                "       덮어쓰려면 --force 플래그를 사용하세요.",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- 후보 선택 ---
    print("[INFO] QA 후보 선택 시작")
    df_final_internal = run_selection(args, df, summary, optional_present, optional_missing)
    n_selected = len(df_final_internal)
    n_selected_patients = int(df_final_internal["patient_id"].nunique()) if "patient_id" in df_final_internal.columns else -1

    print(f"[INFO] 선택 완료 — 총 {n_selected}건, {n_selected_patients}명 환자")

    # group별 count 출력
    group_counts = {}
    for _, row in df_final_internal.iterrows():
        for g in row.get("_qa_groups", []):
            group_counts[g] = group_counts.get(g, 0) + 1
    print("[INFO] group별 count:")
    for g, c in sorted(group_counts.items()):
        print(f"       {g}: {c}")

    # priority별 count 출력
    priority_counts = {}
    for _, row in df_final_internal.iterrows():
        p = row.get("_qa_priority_label", "defer")
        priority_counts[p] = priority_counts.get(p, 0) + 1
    print("[INFO] priority별 count:")
    for p, c in sorted(priority_counts.items(), key=lambda x: QA_PRIORITY_MAP.get(x[0], 99)):
        print(f"       {p}: {c}")

    # --- dry-run 종료 ---
    if args.dry_run:
        print("\n[DRY-RUN] 상위 5개 예시:")
        df_preview = finalize_manifest(df_final_internal)
        preview_cols = ["crop_id", "patient_id", "crop_score_l1_mean", "qa_group", "qa_priority", "suggested_review_reason"]
        preview_cols_exist = [c for c in preview_cols if c in df_preview.columns]
        print(df_preview[preview_cols_exist].head(5).to_string(index=False))
        print("[INFO] dry-run 완료. 파일 생성 없음.")
        return

    # --- manifest 정리 ---
    df_manifest = finalize_manifest(df_final_internal)

    # --- summary dict 생성 ---
    summary_dict = build_summary_json(
        score_csv_path=score_csv_path,
        summary_json_path=summary_json_path,
        df_input=df,
        df_selected=df_manifest,
        optional_present=optional_present,
        optional_missing=optional_missing,
        args=args,
    )

    # --- review plan MD 생성 ---
    review_plan_md = build_review_plan_md(
        df_selected=df_manifest,
        summary_dict=summary_dict,
        optional_missing=optional_missing,
    )

    # --- 출력 저장 ---
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "hard_negative_top_score_qa_manifest_v1.csv"
    summary_out_path = output_root / "hard_negative_top_score_qa_summary_v1.json"
    review_plan_path = output_root / "hard_negative_top_score_qa_review_plan_v1.md"

    df_manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"[INFO] manifest 저장 완료: {manifest_path}")

    with open(summary_out_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
    print(f"[INFO] summary JSON 저장 완료: {summary_out_path}")

    with open(review_plan_path, "w", encoding="utf-8") as f:
        f.write(review_plan_md)
    print(f"[INFO] review plan MD 저장 완료: {review_plan_path}")

    # --- runtime summary append ---
    if not args.no_runtime_append:
        runtime_summary_path = (
            "outputs/second-stage-lesion-refiner-v1/evaluation/"
            "rd4ad_2p5d_normal_mw_fixed96_v1/runtime_summary_v1.json"
        )
        append_runtime_summary(runtime_summary_path, n_selected, str(output_root))

    print("[INFO] 전체 완료.")


if __name__ == "__main__":
    main()
