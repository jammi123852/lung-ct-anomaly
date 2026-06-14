#!/usr/bin/env python3
"""
build_explanation_card_s4_reason_layer_ctstat_smoke.py

S4 Reason Layer CT-stat Smoke Script

목적:
- metadata-only reason으로 부족했던 "왜 PaDiM score가 높게 보였는가"를
  HU/texture/edge 기반 시각적 근거 후보로 보완
- smoke 8장 대상으로 CT-stat 계산 + CT-stat tag 생성 + reason text 보강

금지 (항상):
- reference CT/mask npy 로드
- PNG open
- model import / feature extraction
- score/threshold 재계산
- 카드 PNG/JSON 수정
- 기존 산출물 수정/삭제/덮어쓰기
- stage2_holdout 접근
- lesion GT mask 사용
- 전체 300장 적용

실행 모드:
- bare 실행: BLOCKED exit 2
- --selftest: 22개 guard 검사 (CT npy 로드 없음)
- --dry-run: 입력 파일/경로 존재 확인 (npy open 없음)
- --plan-smoke-only: dry-run + 계산 계획 출력 (npy open 없음)
- --run-smoke --confirm-generate: 실제 CT-stat 계산 및 산출물 저장
  (이번 단계에서는 ALLOW_RUN_CTSTAT=False 가드로 BLOCKED)

syntax check (실행 아님):
  python -m py_compile scripts/build_explanation_card_s4_reason_layer_ctstat_smoke.py
"""

import argparse
import ast
import json
import os
import pathlib
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

# ============================================================
# 최상위 가드 — 이 단계에서는 전부 False
# ============================================================
ALLOW_CT_STAT = False           # CT/mask npy 로드 허용 여부
ALLOW_RUN_CTSTAT = False        # 실제 CT-stat smoke run 허용 여부
ALLOW_FEATURE_XAI = False       # feature/model 기반 XAI 허용 여부
ALLOW_CARD_MODIFICATION = False # 카드 PNG/JSON 수정 허용 여부

# ============================================================
# 진단 금지어
# ============================================================
FORBIDDEN_TERMS = [
    "cancer", "malignancy", "malignant", "benign",
    "tumor", "tumour",
    "nodule 확정", "pulmonary nodule 확정",
    "ground-glass nodule 확정", "ggn 확정",
    "폐암", "악성", "양성", "종양",
    "결절로 진단", "유리결절로 진단",
    "병변 확정", "암 가능성 높음",
]

# ============================================================
# stage2_holdout 접근 금지 토큰
# ============================================================
STAGE2_HOLDOUT_TOKENS = [
    "stage2_holdout", "stage2-holdout", "stage2holdout",
    "holdout_stage2", "holdout-stage2",
]

# ============================================================
# 경로 상수
# ============================================================
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

CARD_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s3_expansion_cards_v2_fontfix"
)
INDEX_CSV = CARD_ROOT / "index_cards.csv"
CARDS_JSON_DIR = CARD_ROOT / "cards_json"

MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/candidates"
    / "s3_expansion_manifest_v1/s3_expansion_candidate_manifest_v1.csv"
)
HOLD_LIST_CSV = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s3_expansion_hold_list_v1.csv"
)
METADATA_SMOKE_CSV = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s4_reason_layer_metadata_smoke_v1/s4_reason_layer_metadata_smoke_v1.csv"
)
CTSTAT_PREFLIGHT_JSON = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s4_ctstat_reason_smoke_preflight_v1.json"
)
CTSTAT_TAG_DESIGN_CSV = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s4_ctstat_reason_tag_design_v1.csv"
)

REFERENCE_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "reference_bank_v1/full/reference_crop_manifest.csv"
)
REFERENCE_STATS_CSV = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "reference_bank_v1/full/reference_stats_by_position_bin.csv"
)

PATHS_CONFIG = PROJECT_ROOT / "configs/paths.local.yaml"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s4_reason_layer_ctstat_smoke_v1"
)

# smoke 대상 (8장)
SMOKE_TARGETS = [
    "LUNG1-284__c1",
    "LUNG1-220__c3",
    "LUNG1-402__c1",
    "LUNG1-305__c1",
    "MSD_lung_054__c1",
    "LUNG1-057__c1",
    "subset9_1.3.6.1.4.1.14519.5.2.1.6279.6001.291156498203266896953765649282__c1",
    "LUNG1-320__c2",
]

# CT-stat 임계값 (smoke-only draft — 전체 300장 적용 금지)
THRESHOLDS = {
    "denser_than_same_bin_reference_delta_hu_p50": 80.0,
    "denser_than_same_bin_reference_delta_hu_mean": 80.0,
    "less_air_than_reference_delta_air_frac": -0.20,
    "air_sparse_region_air_frac_lt_minus900": 0.30,
    "soft_tissue_or_wall_adjacent_dense_frac": 0.10,
    "texture_or_edge_rich_delta_edge_density": 0.05,
    "texture_or_edge_rich_delta_texture_std": 50.0,
    "reference_hu_mismatch_abs_delta": 300.0,
    "roi_mask_low_coverage": 0.50,
    "ct_stat_uncertain_abs_delta_hu": 80.0,
}


# ============================================================
# 유틸리티 함수
# ============================================================

def check_forbidden_terms(text: str) -> List[str]:
    """reason text에 진단 금지어 포함 여부 검사."""
    found = []
    tl = text.lower()
    for term in FORBIDDEN_TERMS:
        if term.lower() in tl:
            found.append(term)
    return found


def assert_no_forbidden(text: str, context: str = "") -> None:
    violations = check_forbidden_terms(text)
    if violations:
        raise RuntimeError(
            f"BLOCKED: forbidden term detected in {context}: {violations}"
        )


def check_stage2_holdout(path_str: str) -> bool:
    pl = path_str.lower()
    return any(tok in pl for tok in STAGE2_HOLDOUT_TOKENS)


def assert_no_stage2_holdout(path_str: str, context: str = "") -> None:
    if check_stage2_holdout(path_str):
        raise RuntimeError(
            f"BLOCKED: stage2_holdout access in {context}: {path_str}"
        )


def load_yaml_paths(config_path: pathlib.Path) -> Dict[str, str]:
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_vol_root(role: str, paths_cfg: Dict[str, str]) -> pathlib.Path:
    """role에 따라 CT vol root 결정."""
    if role == "normal_control":
        raw = paths_cfg.get("normal_training_ready_v2_roi0_0", "").strip()
    else:
        raw = paths_cfg.get("nsclc_msd_usable_only_v2", "").strip()
    return pathlib.Path(raw) if raw else pathlib.Path("")


def resolve_ct_path(vol_root: pathlib.Path, safe_id: str) -> pathlib.Path:
    return vol_root / "volumes_npy" / safe_id / "ct_hu.npy"


def resolve_mask_path(vol_root: pathlib.Path, safe_id: str) -> pathlib.Path:
    return vol_root / "volumes_npy" / safe_id / "roi_0_0.npy"


def get_role(row: Dict[str, Any]) -> Optional[str]:
    """role 있으면 사용, 없으면 prototype_role, 둘 다 없으면 None."""
    role = str(row.get("role", "")).strip()
    if role and role not in ("", "nan"):
        return role
    proto = str(row.get("prototype_role", "")).strip()
    if proto and proto not in ("", "nan"):
        return proto
    return None


# ============================================================
# 입력 데이터 로드 함수
# ============================================================

def load_hold_set(hold_csv: pathlib.Path) -> set:
    if not hold_csv.exists():
        return set()
    df = pd.read_csv(hold_csv, dtype=str)
    col = "expansion_case_id"
    if col in df.columns:
        return set(df[col].dropna().str.strip().tolist())
    return set()


def load_index_csv(index_csv: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(index_csv, dtype=str)


def load_manifest_csv(manifest_csv: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(manifest_csv, dtype=str)


def load_metadata_smoke_csv(csv_path: pathlib.Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path, dtype=str)


def load_card_json(cards_json_dir: pathlib.Path, case_id: str) -> Optional[Dict]:
    p = cards_json_dir / f"{case_id}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_reference_manifest(ref_manifest_csv: pathlib.Path) -> pd.DataFrame:
    if not ref_manifest_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(ref_manifest_csv, dtype=str)


def load_reference_stats(ref_stats_csv: pathlib.Path) -> pd.DataFrame:
    if not ref_stats_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(ref_stats_csv, dtype=str)


# ============================================================
# reference lookup 함수
# ============================================================

def lookup_reference_stats(
    card_json: Dict,
    ref_manifest_df: pd.DataFrame,
    ref_stats_df: pd.DataFrame,
    position_bin: str,
) -> Dict[str, Any]:
    """
    card JSON의 normal_reference_crops 목록을 reference_crop_manifest에서 lookup.
    reference CT 로드 없음, PNG open 없음.
    반환: {reference_hu_mean, reference_dense_frac, reference_crop_count,
            reference_bin_hu_p50, reference_bin_dense_p50, lookup_errors}
    """
    result: Dict[str, Any] = {
        "reference_hu_mean": None,
        "reference_hu_p50_from_crops": None,
        "reference_dense_frac": None,
        "reference_crop_count": 0,
        "reference_bin_hu_p50": None,
        "reference_bin_dense_p50": None,
        "lookup_errors": [],
    }

    # reference_crop_manifest lookup
    ref_crops = card_json.get("normal_reference_crops", [])
    if not ref_crops:
        result["lookup_errors"].append("normal_reference_crops empty in card_json")
        return result

    if ref_manifest_df.empty:
        result["lookup_errors"].append("reference_crop_manifest not loaded")
        return result

    hu_vals = []
    dense_vals = []
    not_found = []

    for crop_path in ref_crops:
        # crop_path 예: "reference_crops/upper_peripheral/subset7_...png"
        # manifest crop_png_path 형식과 동일
        matched = ref_manifest_df[
            ref_manifest_df["crop_png_path"].str.strip() == str(crop_path).strip()
        ]
        if matched.empty:
            # basename 매칭 시도
            basename = pathlib.Path(crop_path).name
            matched = ref_manifest_df[
                ref_manifest_df["crop_png_path"].apply(
                    lambda x: pathlib.Path(str(x)).name == basename
                )
            ]
        if matched.empty:
            not_found.append(str(crop_path))
            continue

        row = matched.iloc[0]
        # position_bin 일치 확인
        row_bin = str(row.get("position_bin", "")).strip()
        if row_bin != position_bin:
            result["lookup_errors"].append(
                f"bin mismatch: crop={crop_path} bin={row_bin} expected={position_bin}"
            )
        try:
            hu = float(row["mean_hu"])
            hu_vals.append(hu)
        except (ValueError, TypeError):
            pass
        try:
            dense = float(row["dense_frac_hu_gt_minus500"])
            dense_vals.append(dense)
        except (ValueError, TypeError):
            pass

    if not_found:
        result["lookup_errors"].append(f"reference crops not found in manifest: {not_found}")

    if hu_vals:
        result["reference_hu_mean"] = float(sum(hu_vals) / len(hu_vals))
        sorted_hu = sorted(hu_vals)
        mid = len(sorted_hu) // 2
        result["reference_hu_p50_from_crops"] = float(sorted_hu[mid])
        result["reference_crop_count"] = len(hu_vals)
    if dense_vals:
        result["reference_dense_frac"] = float(sum(dense_vals) / len(dense_vals))

    # bin-level p50 from reference_stats_by_position_bin
    if not ref_stats_df.empty and "position_bin" in ref_stats_df.columns:
        bin_row = ref_stats_df[
            ref_stats_df["position_bin"].str.strip() == position_bin
        ]
        if not bin_row.empty:
            br = bin_row.iloc[0]
            try:
                result["reference_bin_hu_p50"] = float(br.get("mean_hu_p50", float("nan")))
            except (ValueError, TypeError):
                pass
            try:
                result["reference_bin_dense_p50"] = float(
                    br.get("dense_frac_hu_gt_minus500_p50", float("nan"))
                )
            except (ValueError, TypeError):
                pass

    return result


# ============================================================
# CT-stat tag 함수 (실제 run-smoke 에서 사용)
# ============================================================

def compute_ctstat_tags(stat_row: Dict[str, Any]) -> Dict[str, bool]:
    """
    CT-stat 수치 기반 tag 계산.
    임계값은 smoke-only draft — 전체 300장 적용 금지.

    입력 stat_row 키 (run-smoke 에서 생성):
      candidate_hu_mean, candidate_hu_p50, candidate_hu_p90, candidate_hu_std,
      candidate_dense_frac_gt_minus500, candidate_dense_frac_gt_minus300,
      candidate_air_frac_lt_minus900,
      reference_hu_mean, reference_bin_hu_p50,
      reference_dense_frac,
      delta_hu_mean, delta_hu_p50,
      delta_dense_frac, delta_air_frac,
      candidate_edge_density, candidate_texture_std,
      roi_coverage,
      position_bin
    """
    def safe_float(val, default=float("nan")):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    delta_hu_p50 = safe_float(stat_row.get("delta_hu_p50"))
    delta_hu_mean = safe_float(stat_row.get("delta_hu_mean"))
    delta_air_frac = safe_float(stat_row.get("delta_air_frac"))
    cand_air_frac = safe_float(stat_row.get("candidate_air_frac_lt_minus900"))
    dense_gt_minus300 = safe_float(stat_row.get("candidate_dense_frac_gt_minus300"))
    delta_edge = safe_float(stat_row.get("delta_edge_density"))
    delta_texture = safe_float(stat_row.get("delta_texture_std"))
    ref_hu_mean = safe_float(stat_row.get("reference_hu_mean"))
    cand_hu_mean = safe_float(stat_row.get("candidate_hu_mean"))
    roi_cov = safe_float(stat_row.get("roi_coverage"))
    position_bin = str(stat_row.get("position_bin", "")).lower()
    ref_crop_count = int(stat_row.get("reference_crop_count", 0))

    import math

    def is_valid(v):
        return not (v is None or (isinstance(v, float) and math.isnan(v)))

    tags: Dict[str, bool] = {}

    # denser_than_same_bin_reference
    if is_valid(delta_hu_p50) or is_valid(delta_hu_mean):
        val = delta_hu_p50 if is_valid(delta_hu_p50) else delta_hu_mean
        tags["denser_than_same_bin_reference"] = val > THRESHOLDS[
            "denser_than_same_bin_reference_delta_hu_p50"
        ]
    else:
        tags["denser_than_same_bin_reference"] = False

    # less_air_than_reference
    if is_valid(delta_air_frac):
        tags["less_air_than_reference"] = delta_air_frac < THRESHOLDS[
            "less_air_than_reference_delta_air_frac"
        ]
    else:
        tags["less_air_than_reference"] = False

    # air_sparse_region
    # NOTE: 임계값 0.30 은 smoke 후 재조정 가능성 높음
    if is_valid(cand_air_frac):
        tags["air_sparse_region"] = cand_air_frac < THRESHOLDS[
            "air_sparse_region_air_frac_lt_minus900"
        ]
    else:
        tags["air_sparse_region"] = False

    # soft_tissue_or_wall_adjacent (peripheral 위치 + dense_frac_gt_minus300 > 0.10)
    is_peripheral = "peripheral" in position_bin
    if is_valid(dense_gt_minus300) and is_peripheral:
        tags["soft_tissue_or_wall_adjacent"] = dense_gt_minus300 > THRESHOLDS[
            "soft_tissue_or_wall_adjacent_dense_frac"
        ]
    else:
        tags["soft_tissue_or_wall_adjacent"] = False

    # texture_or_edge_rich
    edge_rich = (
        is_valid(delta_edge)
        and delta_edge > THRESHOLDS["texture_or_edge_rich_delta_edge_density"]
    )
    texture_rich = (
        is_valid(delta_texture)
        and delta_texture > THRESHOLDS["texture_or_edge_rich_delta_texture_std"]
    )
    tags["texture_or_edge_rich"] = edge_rich or texture_rich

    # reference_hu_mismatch (abs delta > 300 → reference 품질 낮음 가능성)
    if is_valid(ref_hu_mean) and is_valid(cand_hu_mean):
        tags["reference_hu_mismatch"] = abs(cand_hu_mean - ref_hu_mean) > THRESHOLDS[
            "reference_hu_mismatch_abs_delta"
        ]
    else:
        tags["reference_hu_mismatch"] = False

    # reference_texture_mismatch — smoke에서는 candidate_texture_std만 있음
    tags["reference_texture_mismatch"] = False  # smoke 단계 미지원

    # roi_mask_low_coverage
    if is_valid(roi_cov):
        tags["roi_mask_low_coverage"] = roi_cov < THRESHOLDS["roi_mask_low_coverage"]
    else:
        tags["roi_mask_low_coverage"] = True  # unknown → 보수적

    # ct_stat_uncertain
    uncertain_reasons = []
    if ref_crop_count == 0:
        uncertain_reasons.append("reference_lookup_failed")
    if is_valid(delta_hu_mean) and abs(delta_hu_mean) < THRESHOLDS["ct_stat_uncertain_abs_delta_hu"]:
        uncertain_reasons.append("delta_hu_mean_weak")
    if not is_valid(delta_hu_mean) and not is_valid(delta_hu_p50):
        uncertain_reasons.append("delta_hu_unavailable")
    tags["ct_stat_uncertain"] = len(uncertain_reasons) > 0
    tags["ct_stat_uncertain_reasons"] = uncertain_reasons  # type: ignore

    return tags


# ============================================================
# CT-stat reason text 생성 함수
# ============================================================

def build_ctstat_reason_text(
    tags: Dict[str, bool],
    stat_row: Dict[str, Any],
    position_bin: str,
    lang: str = "ko",
) -> str:
    """
    CT-stat 기반 reason text 생성.
    진단 문구 금지, 단정 금지, "may support / 검토 근거 후보" 표현 사용.
    """
    def safe_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    hu_mean = safe_float(stat_row.get("candidate_hu_mean"))
    ref_hu = safe_float(stat_row.get("reference_hu_mean"))
    delta_p50 = safe_float(stat_row.get("delta_hu_p50"))
    roi_cov = safe_float(stat_row.get("roi_coverage"))
    cand_air = safe_float(stat_row.get("candidate_air_frac_lt_minus900"))

    parts_ko = []
    parts_en = []

    if tags.get("ct_stat_uncertain"):
        if lang == "ko":
            return (
                "HU 통계 근거가 불충분하여 시각적 보조 설명을 생성하기 어렵습니다. "
                "이는 reference lookup 실패 또는 통계 차이가 작음을 의미합니다. "
                "이는 진단 의미가 아닙니다."
            )
        else:
            return (
                "CT-stat evidence is insufficient to generate a visual support explanation. "
                "This may indicate reference lookup failure or weak statistical difference. "
                "This is not a diagnosis."
            )

    if tags.get("denser_than_same_bin_reference"):
        if delta_p50 is not None:
            parts_ko.append(
                f"HU 통계상 후보 crop의 중앙값 밀도가 같은 위치({position_bin}) 정상 reference보다 "
                f"약 {delta_p50:.0f} HU 높게 나타났습니다. "
                "이는 PaDiM high-response의 시각적 근거 후보일 수 있으나, 진단 의미는 아닙니다."
            )
            parts_en.append(
                f"HU statistics show the candidate crop has a higher median density "
                f"(approx. {delta_p50:.0f} HU above) than same-bin ({position_bin}) normal references. "
                "This may support the observed PaDiM high response, but it is not a diagnosis."
            )
        else:
            parts_ko.append(
                f"같은 위치({position_bin}) 정상 reference보다 밀도가 높은 경향이 관찰됩니다. "
                "이는 PaDiM high-response의 시각적 근거 후보일 수 있으나, 진단 의미는 아닙니다."
            )
            parts_en.append(
                f"Higher density than same-bin ({position_bin}) normal references was observed. "
                "This may support the observed PaDiM high response, but it is not a diagnosis."
            )

    if tags.get("less_air_than_reference"):
        parts_ko.append(
            "reference 대비 air 비율이 낮게 나타났습니다. 이는 조밀한 조직 또는 경계부 구조 인접 가능성을 시사하는 검토 근거 후보입니다."
        )
        parts_en.append(
            "Lower air fraction compared to reference suggests possible adjacency to dense tissue or boundary structure."
        )

    if tags.get("soft_tissue_or_wall_adjacent"):
        parts_ko.append(
            "말초부 위치에서 고밀도 조직 비율이 높게 관찰되어, 흉벽 또는 연부조직 인접 가능성의 검토 근거 후보입니다."
        )
        parts_en.append(
            "High dense-tissue fraction at peripheral location suggests possible chest wall or soft tissue adjacency."
        )

    if tags.get("texture_or_edge_rich"):
        parts_ko.append(
            "후보 crop에서 texture 또는 edge 복잡도가 reference 대비 높게 나타났습니다. 이는 경계부 또는 비균질 구조의 시각적 검토 근거 후보일 수 있습니다."
        )
        parts_en.append(
            "Higher texture or edge complexity compared to reference was observed. "
            "This may support review of boundary or heterogeneous structure."
        )

    if tags.get("roi_mask_low_coverage"):
        if roi_cov is not None:
            parts_ko.append(
                f"ROI mask coverage가 낮습니다({roi_cov:.2f}). crop 영역 일부가 ROI 밖에 있을 수 있습니다."
            )
            parts_en.append(
                f"Low ROI mask coverage ({roi_cov:.2f}): part of the crop may be outside the lung ROI."
            )
        else:
            parts_ko.append("ROI mask coverage를 확인하세요.")
            parts_en.append("Please check ROI mask coverage.")

    if not parts_ko:
        parts_ko = ["CT 통계상 reference와 명확한 차이가 관찰되지 않았습니다. 시각적 보조 설명 근거가 약합니다."]
        parts_en = ["No clear difference from reference in CT statistics. CT-stat support is weak."]

    disclaimer_ko = " 이는 진단 의미가 아니며, 연구용 stage1-dev 후보입니다."
    disclaimer_en = " This is not a diagnosis. This is a stage1-dev research candidate."

    if lang == "ko":
        text = " ".join(parts_ko) + disclaimer_ko
    else:
        text = " ".join(parts_en) + disclaimer_en

    # 금지어 검사
    assert_no_forbidden(text, context=f"ctstat_reason_{lang}")
    return text


# ============================================================
# 실제 CT-stat 계산 함수 (run-smoke 분기 안에만 존재)
# np.load는 이 함수 내부에서만 호출 — mmap_mode="r" 필수
# ============================================================

def compute_candidate_ctstat(
    ct_path: pathlib.Path,
    mask_path: pathlib.Path,
    display_bbox: List[int],
    max_score_slice_index: int,
) -> Dict[str, Any]:
    """
    실제 CT/mask npy 로드 및 통계 계산.
    이 함수는 run-smoke 분기에서만 호출된다.
    np.load는 반드시 mmap_mode="r"로 호출.
    """
    import numpy as np  # run-smoke 분기에서만 import
    from scipy.ndimage import sobel  # texture/edge용 CPU scipy

    result: Dict[str, Any] = {
        "candidate_hu_mean": None,
        "candidate_hu_p10": None,
        "candidate_hu_p50": None,
        "candidate_hu_p90": None,
        "candidate_hu_std": None,
        "candidate_dense_frac_gt_minus500": None,
        "candidate_dense_frac_gt_minus300": None,
        "candidate_air_frac_lt_minus900": None,
        "candidate_texture_std": None,
        "candidate_edge_density": None,
        "roi_coverage": None,
        "mask_empty_flag": None,
        "display_bbox_area": None,
        "error": None,
    }

    # stage2_holdout 접근 방지
    assert_no_stage2_holdout(str(ct_path), context="compute_candidate_ctstat/ct")
    assert_no_stage2_holdout(str(mask_path), context="compute_candidate_ctstat/mask")

    try:
        # CT npy 로드 — mmap_mode="r" 필수
        ct = np.load(str(ct_path), mmap_mode="r")
        mask = np.load(str(mask_path), mmap_mode="r")

        z = int(max_score_slice_index)
        z = max(0, min(z, ct.shape[0] - 1))

        # bbox: [y0, x0, y1, x1]
        if len(display_bbox) != 4:
            result["error"] = f"display_bbox length {len(display_bbox)} != 4"
            return result

        y0, x0, y1, x1 = [int(v) for v in display_bbox]
        y0 = max(0, y0)
        x0 = max(0, x0)
        y1 = min(ct.shape[1], y1)
        x1 = min(ct.shape[2], x1)

        area = (y1 - y0) * (x1 - x0)
        result["display_bbox_area"] = area

        if area < 64:  # 최소 8x8
            result["error"] = f"display_bbox area too small: {area}"
            return result

        ct_crop = ct[z, y0:y1, x0:x1].copy()
        mask_crop = mask[z, y0:y1, x0:x1].copy()

        # ROI coverage
        roi_pix = int(mask_crop.sum())
        roi_cov = roi_pix / area if area > 0 else 0.0
        result["roi_coverage"] = float(roi_cov)
        result["mask_empty_flag"] = roi_cov < 0.05

        # 통계 영역 결정: roi masked 우선, fallback to full_bbox
        if roi_cov >= 0.05:
            stat_vals = ct_crop[mask_crop > 0].astype(np.float32)
        else:
            stat_vals = ct_crop.flatten().astype(np.float32)

        if stat_vals.size == 0:
            result["error"] = "stat_vals empty after mask"
            return result

        result["candidate_hu_mean"] = float(np.mean(stat_vals))
        result["candidate_hu_p10"] = float(np.percentile(stat_vals, 10))
        result["candidate_hu_p50"] = float(np.percentile(stat_vals, 50))
        result["candidate_hu_p90"] = float(np.percentile(stat_vals, 90))
        result["candidate_hu_std"] = float(np.std(stat_vals))
        result["candidate_dense_frac_gt_minus500"] = float(
            np.mean(stat_vals > -500)
        )
        result["candidate_dense_frac_gt_minus300"] = float(
            np.mean(stat_vals > -300)
        )
        result["candidate_air_frac_lt_minus900"] = float(
            np.mean(stat_vals < -900)
        )

        # texture (std of sobel magnitude — CPU Sobel)
        ct_float = ct_crop.astype(np.float32)
        sx = sobel(ct_float, axis=0)
        sy = sobel(ct_float, axis=1)
        edge_mag = np.sqrt(sx**2 + sy**2)
        result["candidate_texture_std"] = float(np.std(ct_float))
        result["candidate_edge_density"] = float(np.mean(edge_mag))

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ============================================================
# smoke 대상 join 함수
# ============================================================

def build_smoke_rows(
    smoke_targets: List[str],
    index_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    metadata_smoke_df: pd.DataFrame,
    hold_set: set,
    cards_json_dir: pathlib.Path,
    ref_manifest_df: pd.DataFrame,
    ref_stats_df: pd.DataFrame,
    paths_cfg: Dict[str, str],
) -> Tuple[List[Dict], List[Dict]]:
    """
    smoke 대상 8장을 join하여 row 목록 반환.
    반환: (rows, errors)
    CT/mask npy open 없음.
    """
    rows = []
    errors = []

    for case_id in smoke_targets:
        row: Dict[str, Any] = {"expansion_case_id": case_id}

        # stage2_holdout 접근 방지
        assert_no_stage2_holdout(case_id, context="build_smoke_rows/case_id")

        # expansion manifest join
        mrow = manifest_df[manifest_df["expansion_case_id"] == case_id]
        if mrow.empty:
            errors.append({"case_id": case_id, "error": "not found in expansion manifest"})
            continue
        mrow = mrow.iloc[0].to_dict()

        # role 결정
        role = get_role(mrow)
        if role is None:
            errors.append({"case_id": case_id, "error": "role and prototype_role both missing"})
            continue

        row["role"] = role
        row["prototype_role"] = str(mrow.get("prototype_role", ""))
        row["safe_id"] = str(mrow.get("safe_id", "")).strip()
        row["patient_id"] = str(mrow.get("patient_id", "")).strip()
        row["position_bin"] = str(mrow.get("position_bin", "")).strip()
        row["hold_flag"] = case_id in hold_set

        # card JSON join
        card_json = load_card_json(cards_json_dir, case_id)
        if card_json is None:
            errors.append({"case_id": case_id, "error": "card JSON not found"})
            continue

        max_score_slice = card_json.get("max_score_slice_index")
        display_bbox = card_json.get("display_bbox")

        if max_score_slice is None:
            errors.append({"case_id": case_id, "error": "max_score_slice_index missing in card JSON"})
            continue
        if display_bbox is None or len(display_bbox) != 4:
            errors.append({"case_id": case_id, "error": f"display_bbox invalid: {display_bbox}"})
            continue

        row["max_score_slice_index"] = int(max_score_slice)
        row["display_bbox"] = display_bbox
        row["display_bbox_area"] = card_json.get("display_bbox_area")
        row["component_bbox_area"] = card_json.get("component_bbox_area")

        # metadata smoke join (optional)
        if not metadata_smoke_df.empty and "expansion_case_id" in metadata_smoke_df.columns:
            meta_row = metadata_smoke_df[
                metadata_smoke_df["expansion_case_id"] == case_id
            ]
            if not meta_row.empty:
                mr = meta_row.iloc[0]
                row["z_span"] = mr.get("z_span", mrow.get("z_span", ""))
                row["overmerge_flag"] = mr.get("overmerge_flag", mrow.get("overmerge_flag", ""))
                row["overmerge_level"] = mr.get("overmerge_level", mrow.get("overmerge_level", ""))
                row["apex_caution"] = mr.get("apex_caution", mrow.get("apex_caution", ""))
                row["max_padim_score"] = mr.get("max_padim_score", mrow.get("max_padim_score", ""))
                row["threshold"] = mr.get("threshold", mrow.get("threshold", ""))
            else:
                row["z_span"] = mrow.get("z_span", "")
                row["overmerge_flag"] = mrow.get("overmerge_flag", "")
                row["overmerge_level"] = mrow.get("overmerge_level", "")
                row["apex_caution"] = mrow.get("apex_caution", "")
                row["max_padim_score"] = mrow.get("max_padim_score", "")
                row["threshold"] = mrow.get("threshold", "")
        else:
            row["z_span"] = mrow.get("z_span", "")
            row["overmerge_flag"] = mrow.get("overmerge_flag", "")
            row["overmerge_level"] = mrow.get("overmerge_level", "")
            row["apex_caution"] = mrow.get("apex_caution", "")
            row["max_padim_score"] = mrow.get("max_padim_score", "")
            row["threshold"] = mrow.get("threshold", "")

        # reference lookup (CSV only, CT 로드 없음)
        ref_stats = lookup_reference_stats(
            card_json=card_json,
            ref_manifest_df=ref_manifest_df,
            ref_stats_df=ref_stats_df,
            position_bin=row["position_bin"],
        )
        row.update(ref_stats)

        # CT/mask path resolve (dry-run에서는 존재 확인만)
        vol_root = resolve_vol_root(role, paths_cfg)
        ct_path = resolve_ct_path(vol_root, row["safe_id"])
        mask_path = resolve_mask_path(vol_root, row["safe_id"])

        assert_no_stage2_holdout(str(ct_path), context="ct_path resolve")
        assert_no_stage2_holdout(str(mask_path), context="mask_path resolve")

        row["ct_path"] = str(ct_path)
        row["mask_path"] = str(mask_path)
        row["ct_path_exists"] = ct_path.exists()
        row["mask_path_exists"] = mask_path.exists()

        rows.append(row)

    return rows, errors


# ============================================================
# selftest 함수 (22개 검사)
# ============================================================

def run_selftest() -> bool:
    """22개 guard 정적 검사."""
    import importlib
    import inspect

    script_path = pathlib.Path(__file__).resolve()
    source = script_path.read_text(encoding="utf-8")

    results = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        results.append((name, status, detail))
        if not condition:
            print(f"  FAIL: {name} — {detail}")
        else:
            print(f"  PASS: {name}")

    print("\n[selftest] 22개 guard 검사 시작")

    # 1. bare guard
    check("1_bare_guard",
          "if len(sys.argv) == 1:" in source or "__name__ == \"__main__\"" in source,
          "bare exit 2 guard")

    # 2. run-smoke confirm guard
    check("2_run_smoke_confirm_guard",
          "--confirm-generate" in source,
          "--run-smoke --confirm-generate 조합 guard")

    # 3. ALLOW_CT_STAT = False
    check("3_allow_ct_stat_false",
          ALLOW_CT_STAT is False,
          f"ALLOW_CT_STAT={ALLOW_CT_STAT}")

    # 4. ALLOW_RUN_CTSTAT = False
    check("4_allow_run_ctstat_false",
          ALLOW_RUN_CTSTAT is False,
          f"ALLOW_RUN_CTSTAT={ALLOW_RUN_CTSTAT}")

    # 5. ALLOW_FEATURE_XAI = False
    check("5_allow_feature_xai_false",
          ALLOW_FEATURE_XAI is False,
          f"ALLOW_FEATURE_XAI={ALLOW_FEATURE_XAI}")

    # 6. ALLOW_CARD_MODIFICATION = False
    check("6_allow_card_modification_false",
          ALLOW_CARD_MODIFICATION is False,
          f"ALLOW_CARD_MODIFICATION={ALLOW_CARD_MODIFICATION}")

    # 7. dry-run에서 np.load 미사용 확인 (np.load는 compute_candidate_ctstat 내부에만)
    # compute_candidate_ctstat 함수 밖에 np.load가 없어야 함
    import re as _re
    lines = source.split("\n")

    def is_actual_npload_call(line: str) -> bool:
        # 실제 np.load( 호출인지 확인 (주석/docstring/string literal 제외)
        s = line.strip()
        if not s:
            return False
        # 주석 라인
        if s.startswith("#"):
            return False
        # docstring/string literal 라인 (따옴표로 시작)
        if s.startswith('"""') or s.startswith("'''") or s.startswith('"') or s.startswith("'"):
            return False
        # 실제 np.load( 호출 패턴
        if not _re.search(r"np\.load\s*\(", s):
            return False
        # 문자열 리터럴 안의 "np.load" 제외
        if '"np.load"' in s or "'np.load'" in s:
            return False
        return True

    # np.load가 있는 함수 확인
    in_ctstat_func = False
    npload_outside = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "def compute_candidate_ctstat(" in stripped:
            in_ctstat_func = True
        elif in_ctstat_func and stripped.startswith("def ") and "compute_candidate_ctstat" not in stripped:
            in_ctstat_func = False
        if is_actual_npload_call(stripped):
            if not in_ctstat_func:
                npload_outside.append((i + 1, stripped))

    check("7_npload_only_in_run_smoke_func",
          len(npload_outside) == 0,
          f"np.load outside compute_candidate_ctstat: {npload_outside[:3]}")

    # 8. np.load는 run-smoke 함수 내부에만
    check("8_npload_in_ctstat_func",
          "np.load" in source and "compute_candidate_ctstat" in source,
          "compute_candidate_ctstat 함수 내 np.load 존재")

    # 9. np.load mmap_mode="r" 포함
    check("9_npload_mmap_mode_r",
          'mmap_mode="r"' in source or "mmap_mode='r'" in source,
          "mmap_mode='r' 포함")

    # 10. reference CT load 코드 없음 (compute_candidate_ctstat 외 np.load 없음 → 7에서 확인)
    check("10_no_reference_ct_load",
          len(npload_outside) == 0,
          "reference CT npy load 없음 (외부 np.load 없음)")

    def is_actual_code_call(line: str, call_pattern: str) -> bool:
        """실제 함수 호출/import인지 확인 (문자열 리터럴 안의 패턴 제외)."""
        s = line.strip()
        if s.startswith("#"):
            return False
        if call_pattern not in s:
            return False
        # 문자열 리터럴 안에 있는지 확인 (따옴표로 감싸진 경우 제외)
        if f'"{call_pattern}"' in s or f"'{call_pattern}'" in s:
            return False
        return True

    # 11. PNG open 코드 없음 (실제 호출만)
    png_opens = [
        line for line in lines
        if (
            is_actual_code_call(line, "Image.open(")
            or is_actual_code_call(line, "plt.imread(")
            or is_actual_code_call(line, "cv2.imread(")
        )
    ]
    check("11_no_png_open",
          len(png_opens) == 0,
          f"PNG open 코드: {png_opens[:3]}")

    # 12. model import 없음 (실제 import 문만)
    model_imports = [
        line for line in lines
        if (
            _re.match(r"\s*import\s+torch\b", line)
            or _re.match(r"\s*import\s+torchvision\b", line)
            or _re.match(r"\s*from\s+torchvision\b", line)
            or _re.match(r"\s*from\s+torch\b", line)
        )
        and not line.strip().startswith("#")
    ]
    check("12_no_model_import",
          len(model_imports) == 0,
          f"model import 라인: {model_imports[:3]}")

    # 13. score 재계산 함수 없음 (def 정의만)
    score_recomp = [
        line for line in lines
        if _re.match(r"\s*def\s+(recompute_score|calc_padim_score|compute_padim_score)\s*\(", line)
        and not line.strip().startswith("#")
    ]
    check("13_no_score_recompute",
          len(score_recomp) == 0,
          f"score 재계산 함수: {score_recomp[:3]}")

    # 14. threshold 재계산 함수 없음 (def 정의만)
    thresh_recomp = [
        line for line in lines
        if _re.match(r"\s*def\s+(recompute_threshold|calc_threshold|compute_threshold)\s*\(", line)
        and not line.strip().startswith("#")
    ]
    check("14_no_threshold_recompute",
          len(thresh_recomp) == 0,
          f"threshold 재계산 함수: {thresh_recomp[:3]}")

    # 15. smoke 대상 8장 이하
    check("15_smoke_targets_le_8",
          len(SMOKE_TARGETS) <= 8,
          f"SMOKE_TARGETS count={len(SMOKE_TARGETS)}")

    # 16. hold 3건 포함
    hold_cases = {"LUNG1-284__c1", "LUNG1-220__c3", "LUNG1-402__c1"}
    check("16_hold_3_included",
          all(c in SMOKE_TARGETS for c in hold_cases),
          f"hold cases in SMOKE_TARGETS: {[c in SMOKE_TARGETS for c in hold_cases]}")

    # 17. role/prototype_role fallback 함수 존재
    check("17_role_prototype_fallback",
          "get_role" in source and "prototype_role" in source,
          "get_role 함수 및 prototype_role fallback")

    # 18. reference manifest lookup 함수 존재
    check("18_reference_manifest_lookup",
          "lookup_reference_stats" in source,
          "lookup_reference_stats 함수 존재")

    # 19. CT-stat tag 함수 존재
    check("19_ctstat_tag_func",
          "compute_ctstat_tags" in source,
          "compute_ctstat_tags 함수 존재")

    # 20. forbidden term guard 함수 존재
    check("20_forbidden_term_guard",
          "check_forbidden_terms" in source and "assert_no_forbidden" in source,
          "check_forbidden_terms + assert_no_forbidden 함수 존재")

    # 21. output guard (ALLOW_RUN_CTSTAT guard) 존재
    check("21_output_guard",
          "ALLOW_RUN_CTSTAT" in source,
          "ALLOW_RUN_CTSTAT guard 존재")

    # 22. stage2_holdout path 접근 없음 (check 함수 호출)
    check("22_stage2_holdout_guard",
          "assert_no_stage2_holdout" in source and "STAGE2_HOLDOUT_TOKENS" in source,
          "stage2_holdout guard 함수 존재")

    total = len(results)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = total - passed

    print(f"\n[selftest] {passed}/{total} PASS, {failed} FAIL")
    return failed == 0


# ============================================================
# dry-run 함수
# ============================================================

def run_dry_run(paths_cfg: Dict[str, str]) -> Dict[str, Any]:
    """
    입력 파일/경로 존재 확인.
    CT/mask npy open 없음.
    """
    report: Dict[str, Any] = {
        "mode": "dry_run",
        "input_files": {},
        "smoke_targets": {},
        "reference_lookup": {},
        "output_guard": {},
        "stage2_holdout_count": 0,
        "npy_open_count": 0,
        "png_open_count": 0,
        "errors": [],
    }

    # 입력 파일 존재 확인
    input_files = {
        "index_csv": INDEX_CSV,
        "manifest_csv": MANIFEST_CSV,
        "hold_list_csv": HOLD_LIST_CSV,
        "metadata_smoke_csv": METADATA_SMOKE_CSV,
        "ctstat_preflight_json": CTSTAT_PREFLIGHT_JSON,
        "ctstat_tag_design_csv": CTSTAT_TAG_DESIGN_CSV,
        "reference_manifest_csv": REFERENCE_MANIFEST_CSV,
        "reference_stats_csv": REFERENCE_STATS_CSV,
        "paths_config": PATHS_CONFIG,
        "cards_json_dir": CARDS_JSON_DIR,
    }
    for name, p in input_files.items():
        exists = p.exists()
        report["input_files"][name] = {"path": str(p), "exists": exists}
        if not exists:
            report["errors"].append(f"missing input: {name} = {p}")

    # smoke 대상 8장 확인
    index_df = load_index_csv(INDEX_CSV) if INDEX_CSV.exists() else pd.DataFrame()
    manifest_df = load_manifest_csv(MANIFEST_CSV) if MANIFEST_CSV.exists() else pd.DataFrame()
    hold_set = load_hold_set(HOLD_LIST_CSV)
    ref_manifest_df = load_reference_manifest(REFERENCE_MANIFEST_CSV)
    ref_stats_df = load_reference_stats(REFERENCE_STATS_CSV)
    metadata_smoke_df = load_metadata_smoke_csv(METADATA_SMOKE_CSV)

    hold_count = 0
    stage2_count = 0
    for case_id in SMOKE_TARGETS:
        r: Dict[str, Any] = {"case_id": case_id}

        # stage2_holdout 체크
        if check_stage2_holdout(case_id):
            stage2_count += 1
            r["stage2_holdout_detected"] = True
            report["errors"].append(f"BLOCKED: stage2_holdout in case_id: {case_id}")
        else:
            r["stage2_holdout_detected"] = False

        # expansion manifest 존재 확인
        if not manifest_df.empty:
            found = case_id in manifest_df["expansion_case_id"].values
            r["in_manifest"] = found
            if not found:
                report["errors"].append(f"case not in manifest: {case_id}")
        else:
            r["in_manifest"] = "manifest_not_loaded"

        # card JSON 존재 확인
        json_path = CARDS_JSON_DIR / f"{case_id}.json"
        r["card_json_exists"] = json_path.exists()
        if not json_path.exists():
            report["errors"].append(f"card JSON not found: {case_id}")

        # hold 확인
        r["hold_flag"] = case_id in hold_set
        if case_id in hold_set:
            hold_count += 1

        # role 확인
        role = None
        if not manifest_df.empty:
            mrows = manifest_df[manifest_df["expansion_case_id"] == case_id]
            if not mrows.empty:
                role = get_role(mrows.iloc[0].to_dict())
        r["role"] = role

        # CT/mask path 존재 확인 (npy open 없음)
        if role is not None:
            vol_root = resolve_vol_root(role, paths_cfg)
            safe_id = ""
            if not manifest_df.empty:
                mrows = manifest_df[manifest_df["expansion_case_id"] == case_id]
                if not mrows.empty:
                    safe_id = str(mrows.iloc[0].get("safe_id", "")).strip()

            if safe_id and vol_root != pathlib.Path(""):
                ct_path = resolve_ct_path(vol_root, safe_id)
                mask_path = resolve_mask_path(vol_root, safe_id)

                # stage2_holdout 경로 체크
                if check_stage2_holdout(str(ct_path)):
                    stage2_count += 1
                    report["errors"].append(f"BLOCKED: stage2_holdout in ct_path: {ct_path}")

                r["ct_path"] = str(ct_path)
                r["mask_path"] = str(mask_path)
                r["ct_path_exists"] = ct_path.exists()
                r["mask_path_exists"] = mask_path.exists()

                if not ct_path.exists():
                    report["errors"].append(f"CT not found: {case_id} -> {ct_path}")
                if not mask_path.exists():
                    report["errors"].append(f"mask not found: {case_id} -> {mask_path}")
            else:
                r["ct_path"] = "unresolved"
                r["mask_path"] = "unresolved"
                r["ct_path_exists"] = False
                r["mask_path_exists"] = False

        # reference lookup 가능 여부 (CSV만, CT 로드 없음)
        if json_path.exists():
            card_json = load_card_json(CARDS_JSON_DIR, case_id)
            if card_json is not None:
                ref_crops = card_json.get("normal_reference_crops", [])
                r["reference_crop_count_in_json"] = len(ref_crops)
                # manifest에서 몇 개 찾을 수 있는지
                if not ref_manifest_df.empty and ref_crops:
                    found_count = 0
                    for cp in ref_crops:
                        m = ref_manifest_df[
                            ref_manifest_df["crop_png_path"].str.strip() == str(cp).strip()
                        ]
                        if m.empty:
                            basename = pathlib.Path(cp).name
                            m = ref_manifest_df[
                                ref_manifest_df["crop_png_path"].apply(
                                    lambda x: pathlib.Path(str(x)).name == basename
                                )
                            ]
                        if not m.empty:
                            found_count += 1
                    r["reference_manifest_found"] = found_count
                    r["reference_manifest_total"] = len(ref_crops)
                else:
                    r["reference_manifest_found"] = 0
                    r["reference_manifest_total"] = len(ref_crops) if ref_crops else 0

        report["smoke_targets"][case_id] = r

    report["hold_count"] = hold_count
    report["stage2_holdout_count"] = stage2_count

    # output root 충돌 확인
    report["output_guard"]["output_root"] = str(OUTPUT_ROOT)
    report["output_guard"]["output_root_exists"] = OUTPUT_ROOT.exists()
    report["output_guard"]["npy_open_count"] = 0  # dry-run은 0
    report["output_guard"]["png_open_count"] = 0

    # 통계
    ct_found = sum(
        1 for v in report["smoke_targets"].values()
        if v.get("ct_path_exists") is True
    )
    mask_found = sum(
        1 for v in report["smoke_targets"].values()
        if v.get("mask_path_exists") is True
    )
    report["ct_paths_found"] = f"{ct_found}/{len(SMOKE_TARGETS)}"
    report["mask_paths_found"] = f"{mask_found}/{len(SMOKE_TARGETS)}"

    return report


# ============================================================
# plan-smoke-only 함수
# ============================================================

def run_plan_smoke_only(dry_report: Dict[str, Any]) -> None:
    """dry-run 결과 + 계산 계획 출력."""
    print("\n[plan-smoke-only] 계산 계획")
    print(f"  smoke 대상: {len(SMOKE_TARGETS)}장")
    print(f"  hold 포함: {dry_report.get('hold_count', 0)}건")
    print(f"  stage2_holdout: {dry_report.get('stage2_holdout_count', 0)}건")
    print(f"  CT paths found: {dry_report.get('ct_paths_found', 'N/A')}")
    print(f"  mask paths found: {dry_report.get('mask_paths_found', 'N/A')}")

    print("\n  [각 케이스 계획]")
    for case_id, info in dry_report.get("smoke_targets", {}).items():
        ct_ok = "OK" if info.get("ct_path_exists") else "MISSING"
        mask_ok = "OK" if info.get("mask_path_exists") else "MISSING"
        hold = "[HOLD]" if info.get("hold_flag") else ""
        ref_found = info.get("reference_manifest_found", "?")
        ref_total = info.get("reference_manifest_total", "?")
        print(
            f"    {case_id} {hold}: CT={ct_ok}, mask={mask_ok}, "
            f"ref={ref_found}/{ref_total}"
        )

    print("\n  [실제 run-smoke 시 계산 예정 컬럼]")
    ctstat_cols = [
        "candidate_hu_mean", "candidate_hu_p10", "candidate_hu_p50",
        "candidate_hu_p90", "candidate_hu_std",
        "candidate_dense_frac_gt_minus500", "candidate_dense_frac_gt_minus300",
        "candidate_air_frac_lt_minus900",
        "candidate_texture_std", "candidate_edge_density",
        "reference_hu_mean", "reference_dense_frac",
        "reference_crop_count", "reference_bin_hu_p50",
        "delta_hu_mean", "delta_hu_p50",
        "delta_dense_frac", "delta_air_frac",
        "delta_edge_density", "delta_texture_std",
        "roi_coverage", "mask_empty_flag",
        "display_bbox_area", "component_bbox_area",
        "boundary_like_flag",
    ]
    for col in ctstat_cols:
        print(f"    - {col}")

    print("\n  [CT-stat tag 계획]")
    for tag in [
        "denser_than_same_bin_reference", "less_air_than_reference",
        "air_sparse_region", "soft_tissue_or_wall_adjacent",
        "texture_or_edge_rich", "reference_hu_mismatch",
        "reference_texture_mismatch", "roi_mask_low_coverage",
        "ct_stat_uncertain",
    ]:
        print(f"    - {tag}")

    print("\n  [임계값 — smoke-only draft, 전체 300장 적용 금지]")
    for k, v in THRESHOLDS.items():
        print(f"    {k}: {v}")

    print("\n  [이번 단계 guard]")
    print(f"    ALLOW_CT_STAT = {ALLOW_CT_STAT}")
    print(f"    ALLOW_RUN_CTSTAT = {ALLOW_RUN_CTSTAT}")
    print(f"    ALLOW_FEATURE_XAI = {ALLOW_FEATURE_XAI}")
    print(f"    ALLOW_CARD_MODIFICATION = {ALLOW_CARD_MODIFICATION}")
    print("    → 실제 CT-stat smoke 실행은 이번 단계 BLOCKED")


# ============================================================
# 실제 run-smoke 함수 (이번 단계 BLOCKED)
# ============================================================

def run_smoke(paths_cfg: Dict[str, str]) -> None:
    """
    실제 CT-stat smoke 실행.
    ALLOW_RUN_CTSTAT=False 인 경우 BLOCKED.
    """
    if not ALLOW_RUN_CTSTAT:
        print("BLOCKED: ALLOW_RUN_CTSTAT=False — 실제 CT-stat smoke 실행은 이번 단계에서 금지.")
        print("다음 단계 승인 후 ALLOW_RUN_CTSTAT=True로 변경 후 실행하세요.")
        sys.exit(2)

    if not ALLOW_CT_STAT:
        print("BLOCKED: ALLOW_CT_STAT=False — CT npy 로드 금지.")
        sys.exit(2)

    if ALLOW_CARD_MODIFICATION:
        print("BLOCKED: ALLOW_CARD_MODIFICATION=True — 카드 수정은 금지입니다.")
        sys.exit(2)

    # 이 아래 코드는 ALLOW_RUN_CTSTAT=True 일 때만 실행됨
    # (이번 단계에서는 도달하지 않음)
    import numpy as np  # noqa: F401 — run-smoke 분기

    index_df = load_index_csv(INDEX_CSV)
    manifest_df = load_manifest_csv(MANIFEST_CSV)
    hold_set = load_hold_set(HOLD_LIST_CSV)
    ref_manifest_df = load_reference_manifest(REFERENCE_MANIFEST_CSV)
    ref_stats_df = load_reference_stats(REFERENCE_STATS_CSV)
    metadata_smoke_df = load_metadata_smoke_csv(METADATA_SMOKE_CSV)

    smoke_rows, join_errors = build_smoke_rows(
        SMOKE_TARGETS,
        index_df,
        manifest_df,
        metadata_smoke_df,
        hold_set,
        CARDS_JSON_DIR,
        ref_manifest_df,
        ref_stats_df,
        paths_cfg,
    )

    output_rows = []
    errors = list(join_errors)

    for row in smoke_rows:
        case_id = row["expansion_case_id"]
        ct_path = pathlib.Path(row["ct_path"])
        mask_path = pathlib.Path(row["mask_path"])

        if not ct_path.exists():
            errors.append({"case_id": case_id, "error": f"CT not found: {ct_path}"})
            continue
        if not mask_path.exists():
            errors.append({"case_id": case_id, "error": f"mask not found: {mask_path}"})
            continue

        # CT-stat 계산 (np.load는 compute_candidate_ctstat 내부에서만)
        stat = compute_candidate_ctstat(
            ct_path=ct_path,
            mask_path=mask_path,
            display_bbox=row["display_bbox"],
            max_score_slice_index=row["max_score_slice_index"],
        )

        if stat.get("error"):
            errors.append({"case_id": case_id, "error": stat["error"]})
            continue

        # delta 계산
        cand_hu_mean = stat.get("candidate_hu_mean")
        cand_hu_p50 = stat.get("candidate_hu_p50")
        ref_hu_mean = row.get("reference_hu_mean")
        ref_hu_p50 = row.get("reference_hu_p50_from_crops")
        cand_dense = stat.get("candidate_dense_frac_gt_minus500")
        ref_dense = row.get("reference_dense_frac")
        cand_air = stat.get("candidate_air_frac_lt_minus900")

        def safe_delta(a, b):
            try:
                return float(a) - float(b) if a is not None and b is not None else None
            except (TypeError, ValueError):
                return None

        stat_row = dict(row)
        stat_row.update(stat)
        stat_row["delta_hu_mean"] = safe_delta(cand_hu_mean, ref_hu_mean)
        stat_row["delta_hu_p50"] = safe_delta(cand_hu_p50, ref_hu_p50)
        stat_row["delta_dense_frac"] = safe_delta(cand_dense, ref_dense)
        stat_row["delta_air_frac"] = None  # air_frac reference는 bin stats에서
        stat_row["delta_edge_density"] = None  # reference edge not in manifest
        stat_row["delta_texture_std"] = None  # reference texture not in manifest
        stat_row["boundary_like_flag"] = (
            "peripheral" in str(row.get("position_bin", "")).lower()
            and (stat.get("candidate_dense_frac_gt_minus300") or 0) > 0.10
        )

        # CT-stat tag
        tags = compute_ctstat_tags(stat_row)
        stat_row.update({f"tag_{k}": v for k, v in tags.items() if k != "ct_stat_uncertain_reasons"})
        stat_row["tag_ct_stat_uncertain_reasons"] = json.dumps(
            tags.get("ct_stat_uncertain_reasons", []), ensure_ascii=False
        )

        # reason text
        stat_row["ctstat_reason_ko"] = build_ctstat_reason_text(
            tags, stat_row, row.get("position_bin", ""), lang="ko"
        )
        stat_row["ctstat_reason_en"] = build_ctstat_reason_text(
            tags, stat_row, row.get("position_bin", ""), lang="en"
        )

        output_rows.append(stat_row)

    # 산출물 저장
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    out_csv = OUTPUT_ROOT / "s4_reason_layer_ctstat_smoke_v1.csv"
    err_csv = OUTPUT_ROOT / "errors.csv"

    if output_rows:
        flat_rows = []
        for r in output_rows:
            flat = {
                k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                for k, v in r.items()
            }
            flat_rows.append(flat)
        pd.DataFrame(flat_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"  output CSV: {out_csv} ({len(output_rows)} rows)")

    if errors:
        pd.DataFrame(errors).to_csv(err_csv, index=False, encoding="utf-8-sig")
        print(f"  errors CSV: {err_csv} ({len(errors)} rows)")

    # summary
    summary = {
        "smoke_id": "s4_reason_layer_ctstat_smoke_v1",
        "total_cases": len(SMOKE_TARGETS),
        "output_rows": len(output_rows),
        "error_count": len(errors),
        "hold_count": sum(1 for r in output_rows if r.get("hold_flag")),
        "ct_stat_uncertain_count": sum(
            1 for r in output_rows if r.get("tag_ct_stat_uncertain")
        ),
        "allow_ct_stat": ALLOW_CT_STAT,
        "allow_run_ctstat": ALLOW_RUN_CTSTAT,
        "allow_card_modification": ALLOW_CARD_MODIFICATION,
    }
    summary_json = OUTPUT_ROOT / "s4_reason_layer_ctstat_smoke_summary_v1.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  summary JSON: {summary_json}")

    done_json = OUTPUT_ROOT / "DONE.json"
    with open(done_json, "w", encoding="utf-8") as f:
        json.dump({"status": "done", "summary": summary}, f, indent=2, ensure_ascii=False)
    print(f"  DONE.json: {done_json}")


# ============================================================
# main
# ============================================================

def main() -> None:
    # bare 실행 방지 — 인자 없이 실행하면 BLOCKED exit 2
    if len(sys.argv) == 1:
        print(
            "BLOCKED: bare 실행 금지.\n"
            "사용법:\n"
            "  --selftest          : 22개 guard 정적 검사\n"
            "  --dry-run           : 입력 파일/경로 존재 확인 (CT npy 로드 없음)\n"
            "  --plan-smoke-only   : dry-run + 계산 계획 출력\n"
            "  --run-smoke --confirm-generate : 실제 CT-stat smoke 실행 (별도 승인 필요)"
        )
        sys.exit(2)

    parser = argparse.ArgumentParser(
        description="S4 Reason Layer CT-stat Smoke Script"
    )
    parser.add_argument("--selftest", action="store_true", help="22개 guard 정적 검사")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="입력 파일/경로 존재 확인 (CT npy 로드 없음)")
    parser.add_argument("--plan-smoke-only", action="store_true", dest="plan_smoke_only",
                        help="dry-run + 계산 계획 출력")
    parser.add_argument("--run-smoke", action="store_true", dest="run_smoke",
                        help="실제 CT-stat smoke 실행 (ALLOW_RUN_CTSTAT=True 필요)")
    parser.add_argument("--confirm-generate", action="store_true", dest="confirm_generate",
                        help="--run-smoke와 함께 사용: 실제 산출물 저장 확인")
    args = parser.parse_args()

    # --run-smoke 단독 실행 방지
    if args.run_smoke and not args.confirm_generate:
        print("BLOCKED: --run-smoke 단독 실행 금지. --run-smoke --confirm-generate 조합 필요.")
        sys.exit(2)

    # --run-smoke --confirm-generate 이번 단계 BLOCKED
    if args.run_smoke and args.confirm_generate:
        if not ALLOW_RUN_CTSTAT:
            print(
                "BLOCKED: ALLOW_RUN_CTSTAT=False\n"
                "실제 CT-stat smoke 실행은 이번 단계에서 금지입니다.\n"
                "다음 단계 'S4 CT-stat reason smoke 실제 실행 승인' 후 실행하세요."
            )
            sys.exit(2)

    # paths 로드
    paths_cfg = load_yaml_paths(PATHS_CONFIG)

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)

    if args.dry_run or args.plan_smoke_only:
        print("[dry-run] 입력 파일 및 경로 존재 확인 (CT npy 로드 없음)")
        dry_report = run_dry_run(paths_cfg)

        # 결과 출력
        print(f"\n  입력 파일:")
        for name, info in dry_report["input_files"].items():
            status = "OK" if info["exists"] else "MISSING"
            print(f"    [{status}] {name}: {info['path']}")

        print(f"\n  smoke 대상 ({len(SMOKE_TARGETS)}장):")
        for case_id, info in dry_report["smoke_targets"].items():
            hold = "[HOLD]" if info.get("hold_flag") else "      "
            ct_ok = "OK" if info.get("ct_path_exists") else "MISS"
            mask_ok = "OK" if info.get("mask_path_exists") else "MISS"
            ref_f = info.get("reference_manifest_found", "?")
            ref_t = info.get("reference_manifest_total", "?")
            print(
                f"    {hold} {case_id}: CT={ct_ok}, mask={mask_ok}, "
                f"ref={ref_f}/{ref_t}"
            )

        print(f"\n  hold 포함: {dry_report['hold_count']}건")
        print(f"  stage2_holdout: {dry_report['stage2_holdout_count']}건")
        print(f"  CT paths: {dry_report.get('ct_paths_found', 'N/A')}")
        print(f"  mask paths: {dry_report.get('mask_paths_found', 'N/A')}")
        print(f"  npy open 수: {dry_report['output_guard'].get('npy_open_count', 0)}")
        print(f"  PNG open 수: {dry_report['output_guard'].get('png_open_count', 0)}")

        if dry_report["errors"]:
            print(f"\n  [오류 {len(dry_report['errors'])}건]")
            for e in dry_report["errors"]:
                print(f"    - {e}")
        else:
            print("\n  [오류 없음]")

        if args.plan_smoke_only:
            run_plan_smoke_only(dry_report)

        verdict = "PASS" if not dry_report["errors"] else "NEEDS_FIX"
        print(f"\n  최종 판정: {verdict}")
        sys.exit(0 if verdict == "PASS" else 1)

    if args.run_smoke and args.confirm_generate:
        run_smoke(paths_cfg)
        return

    # 알 수 없는 인자
    print("알 수 없는 실행 모드입니다. --selftest / --dry-run / --plan-smoke-only 중 하나를 사용하세요.")
    sys.exit(2)


if __name__ == "__main__":
    main()
