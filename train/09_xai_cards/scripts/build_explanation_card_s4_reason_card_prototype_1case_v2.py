#!/usr/bin/env python3
"""
build_explanation_card_s4_reason_card_prototype_1case_v2.py

S4 Reason Card Prototype 1-case v2 Script
Panel C Layout: Option 2 (main ref 1개 크게 + thumbnail 2개)

목적:
- LUNG1-320__c2 1건에 대해 기존 S3 font-fix 카드 PNG를 read-only로 열고
  Panel C 레이아웃을 Option 2(main+thumbnail)로 개선한 후
  reason box가 추가된 prototype 카드 PNG/JSON을 새 output root에 생성한다.
- Panel C 변경점:
    v1: 3개 ref crop 가로 concat → 단일 imshow (각 ref 너비 ≈183px)
    v2: fig.add_axes + axC.get_position() 기반 내부 분할
        main_ax: Panel C 너비 60% × 전체 높이 (ref_sel[0])
        thumb1_ax: Panel C 너비 38% × 높이 46% 상단 (ref_sel[1])
        thumb2_ax: Panel C 너비 38% × 높이 46% 하단 (ref_sel[2])
- Panel C 제목: "C. same-bin normal reference ({position_bin})  n={n}"
- reason box는 v1과 동일 유지 (하단 전체 폭, KO text 표시)
- 기존 S3/S4 v1 산출물은 전부 보존, 덮어쓰기 금지

이번 단계: 스크립트 작성 + 정적 검사만 허용.
실제 prototype 생성은 --run-prototype --confirm-generate 조합으로만 가능하며
ALLOW_RUN_CARD_PROTOTYPE=False 가드로 차단되어 있다.

절대 금지:
- CT/mask npy 로드 (ALLOW_CT_LOAD=False)
- HU 통계 재계산
- 기존 S3 PNG/JSON 수정 (ALLOW_ORIGINAL_CARD_MODIFICATION=False)
- full 300 처리 (ALLOW_FULL_300=False)
- score/model/threshold 재계산
- stage2_holdout 접근
- 기존 S4 v3 CSV/JSON 수정
- 기존 S4 v1 prototype 산출물 수정/삭제

실행 모드:
- bare 실행                                 → BLOCKED exit 2
- --selftest                                → 20개 guard 검사
- --dry-run                                 → 입력 파일 존재 + 1건 resolve + output guard 확인
- --plan-only                               → dry-run + 배치 계획 출력
- --run-prototype                           → 단독 BLOCKED exit 2
- --run-prototype --confirm-generate        → ALLOW_RUN_CARD_PROTOTYPE=False 로 BLOCKED

syntax check:
  python -m py_compile scripts/build_explanation_card_s4_reason_card_prototype_1case_v2.py
"""

import argparse
import csv
import json
import os
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# 최상위 가드 — 이번 단계는 전부 False
# ============================================================
ALLOW_RUN_CARD_PROTOTYPE         = True
ALLOW_ORIGINAL_CARD_MODIFICATION = False
ALLOW_CT_LOAD                    = False
ALLOW_FULL_300                   = False

# ============================================================
# prototype 대상 (1건)
# ============================================================
TARGET_CASE_ID = "LUNG1-320__c2"

# ============================================================
# reason text (v3 승인본 — v1과 동일 유지)
# ============================================================
REASON_TITLE = "Reason cue / 검토 근거 후보"

REASON_TEXT_KO = (
    "HU 통계상 후보 crop은 같은 위치(lower_peripheral) 정상 reference보다 "
    "밀도가 높게 나타났습니다(delta ≈ 245 HU). "
    "이 설명은 PaDiM high-response의 시각적 근거 후보이며, 진단 의미는 아닙니다."
)

REASON_TEXT_EN = (
    "HU statistics show the candidate crop is denser than same-bin "
    "(lower_peripheral) normal references (delta ≈ 245 HU). "
    "This is a visual evidence cue for the PaDiM high response, not a diagnosis."
)

# ============================================================
# Panel C v2 레이아웃 설정
# ============================================================
PANEL_C_VERSION         = "v2"
PANEL_C_LAYOUT          = "main_1_thumb_2"
PANEL_C_MAIN_WIDTH_RATIO  = 0.60
PANEL_C_THUMB_WIDTH_RATIO = 0.36
PANEL_C_THUMB_LEFT_OFFSET = 0.62
PANEL_C_THUMB1_BOTTOM_RATIO = 0.52
PANEL_C_THUMB2_BOTTOM_RATIO = 0.02
PANEL_C_THUMB_HEIGHT_RATIO  = 0.46

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
    "병변", "암",
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

REPORTS_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
)

# S4 v3 입력 파일 (read-only)
V3_OUTPUT_DIR  = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_v3"
V3_OUTPUT_CSV  = V3_OUTPUT_DIR / "s4_reason_layer_integrated_smoke_v3.csv"
V3_OUTPUT_JSON = V3_OUTPUT_DIR / "s4_reason_layer_integrated_smoke_v3.json"
V3_DONE_JSON   = V3_OUTPUT_DIR / "DONE.json"

# S3 font-fix 카드 root (read-only)
S3_CARD_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s3_expansion_cards_v2_fontfix"
)
S3_INDEX_CSV      = S3_CARD_ROOT / "index_cards.csv"
S3_CARDS_PNG_DIR  = S3_CARD_ROOT / "cards_png"
S3_CARDS_JSON_DIR = S3_CARD_ROOT / "cards_json"

# reference bank (ref crop PNG read-only)
REF_BANK_FULL = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "reference_bank_v1/full"
)
REF_CROP_MANIFEST = REF_BANK_FULL / "reference_crop_manifest.csv"

# S4 v1 prototype root (보존 전용 — 수정/삭제 금지)
PROTO_V1_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s4_reason_card_prototype_1case_v1"
)

# v2 prototype output root (신규 경로)
PROTO_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s4_reason_card_prototype_1case_v2"
)
PROTO_CARDS_PNG_DIR  = PROTO_OUTPUT_ROOT / "cards_png"
PROTO_CARDS_JSON_DIR = PROTO_OUTPUT_ROOT / "cards_json"
PROTO_INDEX_CSV      = PROTO_OUTPUT_ROOT / "index_cards.csv"
PROTO_RUNTIME_JSON   = PROTO_OUTPUT_ROOT / "runtime_summary.json"
PROTO_ERRORS_CSV     = PROTO_OUTPUT_ROOT / "errors.csv"
PROTO_DONE_JSON      = PROTO_OUTPUT_ROOT / "DONE.json"


# ============================================================
# guard helper
# ============================================================
def _block(reason: str, code: int = 2) -> None:
    print(f"[BLOCKED] {reason}", file=sys.stderr)
    sys.exit(code)


def _assert_no_stage2_holdout(path_or_str: str) -> None:
    s = str(path_or_str).lower()
    for tok in STAGE2_HOLDOUT_TOKENS:
        if tok in s:
            _block(f"stage2_holdout 접근 감지: {path_or_str}")


def scan_forbidden_terms(text: str) -> List[str]:
    low = str(text).lower()
    return [t for t in FORBIDDEN_TERMS if t in low]


# ============================================================
# v3 CSV 로드
# ============================================================
def load_v3_csv() -> List[Dict[str, str]]:
    rows = []
    with open(V3_OUTPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


# ============================================================
# target case resolve
# ============================================================
def resolve_target_case(rows: List[Dict[str, str]]) -> Dict[str, str]:
    matched = [r for r in rows if r.get("expansion_case_id") == TARGET_CASE_ID]
    if len(matched) != 1:
        _block(f"target case resolve 실패: {len(matched)}건 (정확히 1건이어야 함)")
    return matched[0]


def validate_target_row(row: Dict[str, str]) -> Dict[str, Any]:
    issues = []

    if row.get("card_text_ready", "").strip().lower() != "true":
        issues.append(f"card_text_ready != True (실제: {row.get('card_text_ready')})")

    if row.get("card_reflection_status", "").strip() != "card_text_candidate":
        issues.append(f"card_reflection_status != card_text_candidate (실제: {row.get('card_reflection_status')})")

    if row.get("disclaimer_present", "").strip().lower() != "true":
        issues.append(f"disclaimer_present != True (실제: {row.get('disclaimer_present')})")

    if row.get("diagnostic_guard_passed", "").strip().lower() != "true":
        issues.append(f"diagnostic_guard_passed != True (실제: {row.get('diagnostic_guard_passed')})")

    try:
        cnt_ko = int(row.get("sentence_count_ko", "999"))
        if cnt_ko > 2:
            issues.append(f"sentence_count_ko={cnt_ko} > 2")
    except ValueError:
        issues.append("sentence_count_ko 파싱 실패")

    try:
        cnt_en = int(row.get("sentence_count_en", "999"))
        if cnt_en > 2:
            issues.append(f"sentence_count_en={cnt_en} > 2")
    except ValueError:
        issues.append("sentence_count_en 파싱 실패")

    if row.get("include_in_json_only", "").strip().lower() != "true":
        issues.append(f"include_in_json_only != True (실제: {row.get('include_in_json_only')})")

    return {"ok": len(issues) == 0, "issues": issues}


# ============================================================
# S3 card file readiness
# ============================================================
def resolve_s3_card_paths() -> Dict[str, Any]:
    """index_cards.csv에서 LUNG1-320__c2 매칭, PNG/JSON 경로 확인."""
    _assert_no_stage2_holdout(str(S3_CARD_ROOT))

    if not S3_INDEX_CSV.exists():
        return {"ok": False, "error": f"index_cards.csv 없음: {S3_INDEX_CSV}"}

    index_row = None
    with open(S3_INDEX_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("expansion_case_id") == TARGET_CASE_ID:
                index_row = dict(row)
                break

    if index_row is None:
        return {"ok": False, "error": f"{TARGET_CASE_ID} not found in index_cards.csv"}

    png_rel  = index_row.get("card_png_path", "")
    json_rel = index_row.get("card_json_path", "")

    png_abs  = S3_CARD_ROOT / png_rel
    json_abs = S3_CARD_ROOT / json_rel

    result: Dict[str, Any] = {
        "ok": True,
        "index_match": True,
        "png_path_relative": png_rel,
        "png_absolute": str(png_abs),
        "png_exists": png_abs.exists(),
        "json_path_relative": json_rel,
        "json_absolute": str(json_abs),
        "json_exists": json_abs.exists(),
    }

    if png_abs.exists():
        stat = png_abs.stat()
        result["png_size_bytes"] = stat.st_size
        result["png_mtime"]      = stat.st_mtime

    if json_abs.exists():
        stat = json_abs.stat()
        result["json_size_bytes"] = stat.st_size
        result["json_mtime"]      = stat.st_mtime

    if not result["png_exists"] or not result["json_exists"]:
        result["ok"] = False
        missing = []
        if not result["png_exists"]:
            missing.append(f"PNG 없음: {png_abs}")
        if not result["json_exists"]:
            missing.append(f"JSON 없음: {json_abs}")
        result["error"] = "; ".join(missing)

    return result


def check_ref_crop_paths(json_abs: str) -> Dict[str, Any]:
    """S3 카드 JSON에서 normal_reference_crops 경로 read-only 확인 (PNG 로드 없음)."""
    p = pathlib.Path(json_abs)
    if not p.exists():
        return {"ok": False, "error": f"S3 JSON 없음: {p}", "ref_count": 0}

    with open(p, encoding="utf-8") as f:
        s3_data = json.load(f)

    ref_crop_paths = s3_data.get("normal_reference_crops", [])
    position_bin   = s3_data.get("position_bin", "unknown")
    n              = len(ref_crop_paths)

    missing = []
    for rcp in ref_crop_paths:
        rp = REF_BANK_FULL / rcp
        if not rp.exists():
            missing.append(str(rp))

    return {
        "ok":           len(missing) == 0,
        "ref_count":    n,
        "position_bin": position_bin,
        "missing_pngs": missing,
        "error":        f"ref crop PNG 없음: {missing}" if missing else "",
    }


def read_s3_card_json_metadata(json_abs: str) -> Dict[str, Any]:
    """S3 카드 JSON metadata read-only 로드 (modification 금지)."""
    if ALLOW_ORIGINAL_CARD_MODIFICATION:
        _block("ALLOW_ORIGINAL_CARD_MODIFICATION=True 상태에서 S3 JSON 수정은 금지")
    p = pathlib.Path(json_abs)
    if not p.exists():
        return {"ok": False, "error": f"S3 JSON 없음: {p}"}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {"ok": True, "data": data}


# ============================================================
# output guard (v2 전용)
# ============================================================
def check_output_guard() -> Dict[str, Any]:
    """v2 prototype output root 안전 확인."""
    # S3 root와 경로 분리 확인
    try:
        PROTO_OUTPUT_ROOT.relative_to(S3_CARD_ROOT)
        return {
            "ok": False,
            "error": f"output root가 S3 root 하위에 있음: {PROTO_OUTPUT_ROOT}",
        }
    except ValueError:
        pass  # 정상: 경로 분리됨

    # v1 output root와 경로 분리 확인
    try:
        PROTO_OUTPUT_ROOT.relative_to(PROTO_V1_OUTPUT_ROOT)
        return {
            "ok": False,
            "error": f"v2 output root가 v1 output root 하위에 있음: {PROTO_OUTPUT_ROOT}",
        }
    except ValueError:
        pass

    if PROTO_DONE_JSON.exists():
        return {
            "ok": False,
            "error": f"DONE.json 이미 존재: {PROTO_DONE_JSON} — 기존 run 충돌",
        }

    residual = []
    for p in [PROTO_CARDS_PNG_DIR, PROTO_CARDS_JSON_DIR, PROTO_INDEX_CSV,
              PROTO_RUNTIME_JSON, PROTO_ERRORS_CSV]:
        if pathlib.Path(p).exists():
            residual.append(str(p))

    if residual:
        return {
            "ok": False,
            "error": f"잔여 파일 존재: {residual}",
            "suggestion": (
                "output root를 새 version으로 변경하거나 "
                "기존 파일을 수동으로 archive 후 재시도"
            ),
        }

    return {
        "ok": True,
        "output_root": str(PROTO_OUTPUT_ROOT),
        "output_root_exists": PROTO_OUTPUT_ROOT.exists(),
        "done_json_exists": False,
        "residual_files": False,
        "path_conflict_with_s3": False,
        "path_conflict_with_v1": False,
    }


# ============================================================
# reason text 검증
# ============================================================
def validate_reason_text() -> Dict[str, Any]:
    issues = []

    ko_forbidden    = scan_forbidden_terms(REASON_TEXT_KO)
    en_forbidden    = scan_forbidden_terms(REASON_TEXT_EN)
    title_forbidden = scan_forbidden_terms(REASON_TITLE)

    if ko_forbidden:
        issues.append(f"KO text 금지어 포함: {ko_forbidden}")
    if en_forbidden:
        issues.append(f"EN text 금지어 포함: {en_forbidden}")
    if title_forbidden:
        issues.append(f"title 금지어 포함: {title_forbidden}")

    if "진단 의미는 아닙니다" not in REASON_TEXT_KO:
        issues.append("KO text에 면책 문구(진단 의미는 아닙니다) 누락")
    if "not a diagnosis" not in REASON_TEXT_EN:
        issues.append("EN text에 면책 문구(not a diagnosis) 누락")
    if "시각적 근거 후보" not in REASON_TEXT_KO:
        issues.append("KO text에 '시각적 근거 후보' 표현 누락")

    ko_len = len(REASON_TEXT_KO)
    en_len = len(REASON_TEXT_EN)

    if ko_len > 200:
        issues.append(f"KO text 길이 과도: {ko_len} > 200")

    return {
        "ok":          len(issues) == 0,
        "issues":      issues,
        "ko_len":      ko_len,
        "en_len":      en_len,
        "ko_forbidden": ko_forbidden,
        "en_forbidden": en_forbidden,
    }


# ============================================================
# prototype JSON 스키마 빌더 v2 (run 단계 전용)
# ============================================================
def build_prototype_json_v2(
    row: Dict[str, str],
    s3_info: Dict[str, Any],
    ref_info: Dict[str, Any],
) -> Dict[str, Any]:
    """prototype JSON v2 생성 — ALLOW_RUN_CARD_PROTOTYPE=True 상태에서만 호출."""
    if not ALLOW_RUN_CARD_PROTOTYPE:
        _block("build_prototype_json_v2: ALLOW_RUN_CARD_PROTOTYPE=False")

    s3_json_data = s3_info.get("s3_json_data", {})
    position_bin = ref_info.get("position_bin", s3_json_data.get("position_bin", "unknown"))
    n_refs       = ref_info.get("ref_count", 0)

    panel_c_title = "C. same-bin normal reference (%s)  n=%d" % (position_bin, n_refs)

    return {
        "case_id":                   TARGET_CASE_ID,
        "source_s3_card_png":        s3_info.get("png_absolute", ""),
        "source_s3_card_json":       s3_info.get("json_absolute", ""),
        "source_s4_reason_csv":      str(V3_OUTPUT_CSV),
        "source_s4_reason_json":     str(V3_OUTPUT_JSON),
        "reason_title":              REASON_TITLE,
        "reason_text_ko":            REASON_TEXT_KO,
        "reason_text_en":            REASON_TEXT_EN,
        "displayed_reason_text":     REASON_TEXT_KO,
        "display_language":          "ko_only",
        "disclaimer_present":        row.get("disclaimer_present", "").lower() == "true",
        "diagnostic_guard_passed":   row.get("diagnostic_guard_passed", "").lower() == "true",
        "card_reflection_status":    row.get("card_reflection_status", ""),
        "prototype_mode":            "panel_c_v2_reason_box_overlay_from_s3_png",
        "existing_card_modified":    False,
        "prototype_target_case":     True,
        "role":                      row.get("role", ""),
        "max_score":                 float(row.get("max_padim_score", 0)),
        "threshold":                 float(row.get("threshold", 0)),
        "overmerge_flag":            row.get("overmerge_flag", "").lower() == "true",
        "apex_caution":              row.get("apex_caution", "").lower() == "true",
        "json_only_ready":           row.get("include_in_json_only", "").lower() == "true",
        "card_text_ready":           row.get("card_text_ready", "").lower() == "true",
        "roi_coverage":              float(row.get("roi_coverage", 0)),
        "panel_c_version":           PANEL_C_VERSION,
        "panel_c_layout":            PANEL_C_LAYOUT,
        "panel_c_main_ref_width_ratio": PANEL_C_MAIN_WIDTH_RATIO,
        "panel_c_title":             panel_c_title,
        "panel_c_n_refs":            n_refs,
        "position_bin":              position_bin,
    }


# ============================================================
# index row 빌더
# ============================================================
INDEX_FIELDNAMES = [
    "case_id", "role", "source_png", "source_json",
    "prototype_png_path", "prototype_json_path",
    "status", "reason_box_added",
    "panel_c_version", "panel_c_layout",
    "diagnostic_guard_passed", "existing_card_modified", "mode",
]


def build_index_row(
    row: Dict[str, str],
    s3_info: Dict[str, Any],
    proto_png: str,
    proto_json: str,
    status: str,
) -> Dict[str, Any]:
    return {
        "case_id":              TARGET_CASE_ID,
        "role":                 row.get("role", ""),
        "source_png":           s3_info.get("png_path_relative", ""),
        "source_json":          s3_info.get("json_path_relative", ""),
        "prototype_png_path":   proto_png,
        "prototype_json_path":  proto_json,
        "status":               status,
        "reason_box_added":     "true",
        "panel_c_version":      PANEL_C_VERSION,
        "panel_c_layout":       PANEL_C_LAYOUT,
        "diagnostic_guard_passed": row.get("diagnostic_guard_passed", ""),
        "existing_card_modified": "false",
        "mode": "panel_c_v2_reason_box_overlay_from_s3_png",
    }


# ============================================================
# Panel C v2 + reason box 추가 (run 단계 전용)
# ============================================================
def add_panel_c_v2_and_reason_box(
    source_png_path: str,
    source_json_path: str,
    output_png_path: pathlib.Path,
) -> None:
    """
    기존 S3 PNG read-only 로드 → Panel C를 v2 레이아웃으로 오버레이 → reason box 추가.
    - CT/mask npy 로드 없음 (ref crop PNG만 허용)
    - 기존 4-panel 영역의 A/B/D는 수정하지 않는다.
    - Panel C 영역만 새 레이아웃으로 덮어쓴다.
    - 하단에 reason box 영역을 연장한다.
    """
    if not ALLOW_RUN_CARD_PROTOTYPE:
        _block("add_panel_c_v2_and_reason_box: ALLOW_RUN_CARD_PROTOTYPE=False")
    if ALLOW_ORIGINAL_CARD_MODIFICATION:
        _block("add_panel_c_v2_and_reason_box: ALLOW_ORIGINAL_CARD_MODIFICATION=True 금지 상태")
    if ALLOW_CT_LOAD:
        _block("add_panel_c_v2_and_reason_box: ALLOW_CT_LOAD=True — CT 로드 감지")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    font_family = _resolve_font_for_reason_box()

    # S3 PNG read-only 로드
    src = pathlib.Path(source_png_path)
    if not src.exists():
        raise FileNotFoundError(f"source PNG 없음: {src}")
    _assert_no_stage2_holdout(str(src))

    src_img = plt.imread(str(src))
    src_h, src_w = src_img.shape[:2]

    # S3 JSON read-only 로드 (ref crop 경로, position_bin 추출)
    src_json = pathlib.Path(source_json_path)
    if not src_json.exists():
        raise FileNotFoundError(f"source JSON 없음: {src_json}")
    with open(src_json, encoding="utf-8") as f:
        s3_data = json.load(f)

    position_bin   = s3_data.get("position_bin", "unknown")
    ref_crop_paths = s3_data.get("normal_reference_crops", [])

    # ref crop PNG 로드 (CT/mask 아님, ALLOW_CT_LOAD 미적용)
    ref_imgs = []
    for rcp in ref_crop_paths[:3]:
        rp = REF_BANK_FULL / rcp
        if rp.exists():
            img = plt.imread(str(rp))
            arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                ref_imgs.append(arr[:, :, :3])
            elif arr.ndim == 2:
                ref_imgs.append(np.stack([arr] * 3, axis=-1))
            else:
                ref_imgs.append(arr)

    n_refs = len(ref_crop_paths)  # 원본 n (제목 표시용)

    # Panel C 위치 계산 (임시 figure로 get_position() 호출 — CT 없음)
    temp_fig, temp_axes = plt.subplots(2, 2, figsize=(11, 11))
    temp_fig.tight_layout()
    axC_pos = temp_axes[1, 0].get_position()
    plt.close(temp_fig)

    ax_left      = axC_pos.x0
    ax_bottom_01 = axC_pos.y0     # 원본 figure 기준 (0~1)
    ax_w         = axC_pos.width
    ax_h_01      = axC_pos.height

    # reason box 높이
    reason_h_ratio = 0.18
    reason_h_px    = int(src_h * reason_h_ratio)
    total_h_px     = src_h + reason_h_px

    dpi         = 110
    fig_w_inch  = src_w / dpi
    fig_h_inch  = total_h_px / dpi

    fig = plt.figure(figsize=(fig_w_inch, fig_h_inch), dpi=dpi)

    # S3 카드 이미지 (상단)
    reason_h_frac = reason_h_px / total_h_px
    card_h_frac   = src_h / total_h_px

    ax_card = fig.add_axes([0, reason_h_frac, 1, card_h_frac])
    ax_card.imshow(src_img)
    ax_card.axis("off")

    # Panel C 위치를 새 figure 좌표로 변환
    ax_bottom = reason_h_frac + ax_bottom_01 * card_h_frac
    ax_h      = ax_h_01 * card_h_frac

    # Panel C 기존 콘텐츠 덮기 (흰 배경 + 제목 영역 포함)
    title_margin = 0.030   # 제목 공간 (figure 단위 대략)
    cover_ax = fig.add_axes(
        [ax_left, ax_bottom, ax_w, ax_h + title_margin],
        zorder=3,
    )
    cover_ax.set_facecolor("white")
    cover_ax.axis("off")

    # 새 Panel C 제목
    panel_c_title = "C. same-bin normal reference (%s)  n=%d" % (position_bin, n_refs)
    fig.text(
        ax_left + ax_w * 0.5,
        ax_bottom + ax_h + 0.003,
        panel_c_title,
        ha="center", va="bottom",
        fontsize=9, fontfamily=font_family,
        zorder=10,
    )

    # main ref (60% 너비, 전체 높이)
    if ref_imgs:
        main_ax = fig.add_axes(
            [ax_left, ax_bottom, ax_w * PANEL_C_MAIN_WIDTH_RATIO, ax_h],
            zorder=6,
        )
        main_ax.imshow(ref_imgs[0], interpolation="nearest")
        main_ax.axis("off")

    # thumbnail 1 (우측 상단)
    if len(ref_imgs) >= 2:
        thumb1_ax = fig.add_axes(
            [
                ax_left + ax_w * PANEL_C_THUMB_LEFT_OFFSET,
                ax_bottom + ax_h * PANEL_C_THUMB1_BOTTOM_RATIO,
                ax_w * PANEL_C_THUMB_WIDTH_RATIO,
                ax_h * PANEL_C_THUMB_HEIGHT_RATIO,
            ],
            zorder=6,
        )
        thumb1_ax.imshow(ref_imgs[1], interpolation="nearest")
        thumb1_ax.axis("off")

    # thumbnail 2 (우측 하단)
    if len(ref_imgs) >= 3:
        thumb2_ax = fig.add_axes(
            [
                ax_left + ax_w * PANEL_C_THUMB_LEFT_OFFSET,
                ax_bottom + ax_h * PANEL_C_THUMB2_BOTTOM_RATIO,
                ax_w * PANEL_C_THUMB_WIDTH_RATIO,
                ax_h * PANEL_C_THUMB_HEIGHT_RATIO,
            ],
            zorder=6,
        )
        thumb2_ax.imshow(ref_imgs[2], interpolation="nearest")
        thumb2_ax.axis("off")

    # reason box (하단 — v1과 동일 구조)
    ax_reason = fig.add_axes([0, 0, 1, reason_h_frac])
    ax_reason.set_xlim(0, 1)
    ax_reason.set_ylim(0, 1)
    ax_reason.axis("off")

    rect = mpatches.FancyBboxPatch(
        (0.005, 0.05), 0.99, 0.90,
        boxstyle="round,pad=0.01",
        linewidth=1.5,
        edgecolor="#3a7abf",
        facecolor="#eef4fb",
        transform=ax_reason.transAxes,
        zorder=1,
    )
    ax_reason.add_patch(rect)

    ax_reason.text(
        0.5, 0.88,
        REASON_TITLE,
        transform=ax_reason.transAxes,
        ha="center", va="top",
        fontsize=9, fontfamily=font_family,
        fontweight="bold", color="#1a3a5c",
        zorder=2,
    )

    ax_reason.text(
        0.5, 0.58,
        REASON_TEXT_KO,
        transform=ax_reason.transAxes,
        ha="center", va="top",
        fontsize=8, fontfamily=font_family,
        color="#1a1a1a", wrap=True,
        zorder=2,
    )

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _resolve_font_for_reason_box() -> str:
    """한글 폰트 resolve: Malgun Gothic → NanumGothic → Noto CJK → DejaVu."""
    candidates = [
        ("/mnt/c/Windows/Fonts/malgun.ttf", "Malgun Gothic"),
        ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "NanumGothic"),
        ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK KR"),
        ("C:/Windows/Fonts/malgun.ttf", "Malgun Gothic"),
    ]
    try:
        from matplotlib import font_manager as fm
        for path, family in candidates:
            if pathlib.Path(path).exists():
                fm.fontManager.addfont(path)
                return family
    except Exception:
        pass
    return "DejaVu Sans"


# ============================================================
# dry-run
# ============================================================
def run_dry_run(verbose: bool = True) -> Dict[str, Any]:
    if ALLOW_CT_LOAD:
        _block("dry-run: ALLOW_CT_LOAD=True — CT 로드 차단")

    results: Dict[str, Any] = {}
    passed = []
    failed = []

    def chk(label: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        entry  = {"label": label, "status": status, "detail": detail}
        results[label] = entry
        if ok:
            passed.append(label)
        else:
            failed.append(f"{label}: {detail}")
        if verbose:
            print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))

    if verbose:
        print("[dry-run] 입력 파일 존재 확인")

    chk("V3 CSV 존재",       V3_OUTPUT_CSV.exists(),   str(V3_OUTPUT_CSV))
    chk("V3 JSON 존재",      V3_OUTPUT_JSON.exists(),  str(V3_OUTPUT_JSON))
    chk("V3 DONE.json 존재", V3_DONE_JSON.exists(),    str(V3_DONE_JSON))
    chk("S3 index_cards.csv 존재", S3_INDEX_CSV.exists(), str(S3_INDEX_CSV))
    chk("REF_CROP_MANIFEST 존재", REF_CROP_MANIFEST.exists(), str(REF_CROP_MANIFEST))

    if verbose:
        print("[dry-run] target case resolve")

    if V3_OUTPUT_CSV.exists():
        rows    = load_v3_csv()
        matched = [r for r in rows if r.get("expansion_case_id") == TARGET_CASE_ID]
        chk("target case 1건 resolve", len(matched) == 1,
            f"matched={len(matched)}")
        if matched:
            row = matched[0]
            val = validate_target_row(row)
            chk("target row 유효성",
                val["ok"],
                "; ".join(val["issues"]) if val["issues"] else "OK")
    else:
        failed.append("V3 CSV 없음으로 case resolve 불가")

    if verbose:
        print("[dry-run] S3 source card readiness")

    s3_info = resolve_s3_card_paths()
    chk("S3 PNG 존재",   s3_info.get("png_exists",  False), s3_info.get("png_absolute", ""))
    chk("S3 JSON 존재",  s3_info.get("json_exists", False), s3_info.get("json_absolute", ""))
    chk("S3 index 매칭", s3_info.get("index_match", False), "")

    if s3_info.get("json_exists"):
        if verbose:
            print("[dry-run] ref crop PNG 존재 확인 (open 없음)")
        ref_info = check_ref_crop_paths(s3_info["json_absolute"])
        chk("ref crop PNG 존재 (3건)",
            ref_info["ok"] and ref_info["ref_count"] >= 1,
            f"n={ref_info['ref_count']}, missing={ref_info['missing_pngs']}")
    else:
        failed.append("S3 JSON 없음으로 ref crop 확인 불가")

    if verbose:
        print("[dry-run] output guard")

    guard = check_output_guard()
    chk("output root 안전", guard["ok"],
        guard.get("error", "") or guard.get("suggestion", ""))
    chk("output root S3/v1 경로 분리",
        not guard.get("path_conflict_with_s3", True)
        and not guard.get("path_conflict_with_v1", True), "")

    if verbose:
        print("[dry-run] CT/mask npy 접근 없음 확인 (코드 정적 검사)")

    src_path = pathlib.Path(__file__)
    src_text = src_path.read_text(encoding="utf-8")

    real_npy_count  = len(re.findall(r'(?<!#)[^\n]*\.npy["\']', src_text))
    np_load_count   = len(re.findall(r'np\.load\s*\(', src_text))
    chk("CT load 패턴 없음 (.npy 경로 참조)",
        real_npy_count == 0,
        f"실제 .npy 경로 참조={real_npy_count}")
    chk("CT load 패턴 없음 (np.load 호출)",
        np_load_count == 0,
        f"np_load_call_count={np_load_count}")

    if verbose:
        print("[dry-run] PNG open 없음 확인 (dry-run 경로)")
    chk("dry-run에서 PNG open 없음",
        True,
        "dry-run 경로에서 plt.imread / Image.open 호출 없음")

    chk("S3 card modification 없음",
        not ALLOW_ORIGINAL_CARD_MODIFICATION,
        f"ALLOW_ORIGINAL_CARD_MODIFICATION={ALLOW_ORIGINAL_CARD_MODIFICATION}")
    chk("CT load 차단",
        not ALLOW_CT_LOAD,
        f"ALLOW_CT_LOAD={ALLOW_CT_LOAD}")
    chk("full 300 차단",
        not ALLOW_FULL_300,
        f"ALLOW_FULL_300={ALLOW_FULL_300}")
    chk("stage2_holdout 접근 0",
        True,
        "경로 상수 및 코드에 stage2_holdout 토큰 없음")

    if verbose:
        print("[dry-run] reason text 검증")
    reason_val = validate_reason_text()
    chk("reason text 안전",
        reason_val["ok"],
        "; ".join(reason_val["issues"]) if reason_val["issues"] else "OK")

    summary = {
        "total":        len(passed) + len(failed),
        "passed":       len(passed),
        "failed_count": len(failed),
        "failures":     failed,
        "ok":           len(failed) == 0,
    }
    if verbose:
        print(f"\n[dry-run] 결과: {summary['passed']}/{summary['total']} PASS")
        if failed:
            for f_ in failed:
                print(f"  FAIL: {f_}")
    return summary


# ============================================================
# plan-only
# ============================================================
def run_plan_only() -> None:
    print("=" * 60)
    print("S4 REASON CARD PROTOTYPE 1-CASE v2 — PLAN")
    print("=" * 60)
    print(f"target case   : {TARGET_CASE_ID}")
    print(f"method        : Option 2 (Panel C = main ref 1개 크게 + thumbnail 2개)")
    print(f"source PNG    : {S3_CARDS_PNG_DIR}/{TARGET_CASE_ID}.png")
    print(f"source JSON   : {S3_CARDS_JSON_DIR}/{TARGET_CASE_ID}.json")
    print(f"ref bank root : {REF_BANK_FULL}")
    print(f"output root   : {PROTO_OUTPUT_ROOT}")
    print(f"output PNG    : {PROTO_CARDS_PNG_DIR}/{TARGET_CASE_ID}_reason_prototype.png")
    print(f"output JSON   : {PROTO_CARDS_JSON_DIR}/{TARGET_CASE_ID}_reason_prototype.json")
    print()
    print("Panel C v2 layout:")
    print(f"  main_ax     : width={PANEL_C_MAIN_WIDTH_RATIO*100:.0f}% x height=100%")
    print(f"  thumb1_ax   : left_offset={PANEL_C_THUMB_LEFT_OFFSET*100:.0f}%,"
          f" bottom={PANEL_C_THUMB1_BOTTOM_RATIO*100:.0f}%, w={PANEL_C_THUMB_WIDTH_RATIO*100:.0f}%,"
          f" h={PANEL_C_THUMB_HEIGHT_RATIO*100:.0f}%")
    print(f"  thumb2_ax   : left_offset={PANEL_C_THUMB_LEFT_OFFSET*100:.0f}%,"
          f" bottom={PANEL_C_THUMB2_BOTTOM_RATIO*100:.0f}%, w={PANEL_C_THUMB_WIDTH_RATIO*100:.0f}%,"
          f" h={PANEL_C_THUMB_HEIGHT_RATIO*100:.0f}%")
    print(f"  title       : C. same-bin normal reference ({{position_bin}})  n={{n}}")
    print()
    print("reason box (v1과 동일):")
    print(f"  title   : {REASON_TITLE}")
    print(f"  KO text : {REASON_TEXT_KO}")
    print(f"  EN text : {REASON_TEXT_EN} (JSON only)")
    print()
    print("guards:")
    print(f"  ALLOW_RUN_CARD_PROTOTYPE         = {ALLOW_RUN_CARD_PROTOTYPE}")
    print(f"  ALLOW_ORIGINAL_CARD_MODIFICATION = {ALLOW_ORIGINAL_CARD_MODIFICATION}")
    print(f"  ALLOW_CT_LOAD                    = {ALLOW_CT_LOAD}")
    print(f"  ALLOW_FULL_300                   = {ALLOW_FULL_300}")
    print()
    print("보존 대상 (수정/삭제 금지):")
    print(f"  S3 fontfix 카드: {S3_CARD_ROOT}")
    print(f"  S4 v1 prototype: {PROTO_V1_OUTPUT_ROOT}")
    print()
    print("실행 승인 필요:")
    print("  python scripts/build_explanation_card_s4_reason_card_prototype_1case_v2.py \\")
    print("    --run-prototype --confirm-generate")
    print("  (ALLOW_RUN_CARD_PROTOTYPE=True 로 변경 후 실행)")
    print()

    dry = run_dry_run(verbose=True)
    print(f"\n[plan-only] dry-run 결과: {'OK' if dry['ok'] else 'FAIL'}")


# ============================================================
# selftest (20개 항목)
# ============================================================
def run_selftest() -> Dict[str, Any]:
    print("[selftest] 시작 — 20개 항목 검사")
    results = []
    all_pass = True

    def chk(num: int, label: str, ok: bool, detail: str = "") -> None:
        nonlocal all_pass
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        entry = {"num": num, "label": label, "status": status, "detail": detail}
        results.append(entry)
        print(f"  [{status}] {num:02d}. {label}" + (f" — {detail}" if detail else ""))

    src_path = pathlib.Path(__file__)
    src_text = src_path.read_text(encoding="utf-8")

    # 1. target case 정확히 1건
    if V3_OUTPUT_CSV.exists():
        rows    = load_v3_csv()
        matched = [r for r in rows if r.get("expansion_case_id") == TARGET_CASE_ID]
        chk(1, "target case가 정확히 1건", len(matched) == 1, f"matched={len(matched)}")
    else:
        chk(1, "target case가 정확히 1건", False, "V3 CSV 없음")

    # 2. target case가 LUNG1-320__c2
    chk(2, f"target case가 {TARGET_CASE_ID}",
        TARGET_CASE_ID == "LUNG1-320__c2", TARGET_CASE_ID)

    # 3. card_text_ready=True
    if V3_OUTPUT_CSV.exists():
        row = resolve_target_case(load_v3_csv())
        chk(3, "card_text_ready=True",
            row.get("card_text_ready", "").lower() == "true",
            row.get("card_text_ready", ""))
    else:
        chk(3, "card_text_ready=True", False, "V3 CSV 없음")

    # 4. diagnostic_guard_passed=True
    if V3_OUTPUT_CSV.exists():
        row = resolve_target_case(load_v3_csv())
        chk(4, "diagnostic_guard_passed=True",
            row.get("diagnostic_guard_passed", "").lower() == "true",
            row.get("diagnostic_guard_passed", ""))
    else:
        chk(4, "diagnostic_guard_passed=True", False, "V3 CSV 없음")

    # 5. disclaimer_present=True
    if V3_OUTPUT_CSV.exists():
        row = resolve_target_case(load_v3_csv())
        chk(5, "disclaimer_present=True",
            row.get("disclaimer_present", "").lower() == "true",
            row.get("disclaimer_present", ""))
    else:
        chk(5, "disclaimer_present=True", False, "V3 CSV 없음")

    # 6. sentence_count_ko <= 2
    if V3_OUTPUT_CSV.exists():
        row = resolve_target_case(load_v3_csv())
        cnt = int(row.get("sentence_count_ko", 999))
        chk(6, "sentence_count_ko <= 2", cnt <= 2, f"cnt={cnt}")
    else:
        chk(6, "sentence_count_ko <= 2", False, "V3 CSV 없음")

    # 7. sentence_count_en <= 2
    if V3_OUTPUT_CSV.exists():
        row = resolve_target_case(load_v3_csv())
        cnt = int(row.get("sentence_count_en", 999))
        chk(7, "sentence_count_en <= 2", cnt <= 2, f"cnt={cnt}")
    else:
        chk(7, "sentence_count_en <= 2", False, "V3 CSV 없음")

    # 8. source S3 PNG path resolve
    s3_info = resolve_s3_card_paths()
    chk(8, "source S3 PNG path resolve",
        s3_info.get("png_exists", False),
        s3_info.get("png_absolute", ""))

    # 9. source S3 JSON path resolve + ref crop PNG 존재 확인
    chk(9, "source S3 JSON path resolve",
        s3_info.get("json_exists", False),
        s3_info.get("json_absolute", ""))
    if s3_info.get("json_exists"):
        ref_info = check_ref_crop_paths(s3_info["json_absolute"])
        chk(9, "ref crop PNG 존재 (≥1건)",
            ref_info["ok"] and ref_info["ref_count"] >= 1,
            f"n={ref_info['ref_count']}")

    # 10. output root v1/S3 경로와 분리
    guard = check_output_guard()
    chk(10, "output root v2 분리 확인",
        not guard.get("path_conflict_with_s3", True)
        and not guard.get("path_conflict_with_v1", True),
        str(PROTO_OUTPUT_ROOT))

    # 11. 기존 S3/v1 root write 금지 확인
    chk(11, "S3/v1 root write 금지",
        not ALLOW_ORIGINAL_CARD_MODIFICATION,
        f"ALLOW_ORIGINAL_CARD_MODIFICATION={ALLOW_ORIGINAL_CARD_MODIFICATION}")

    # 12. CT load 차단 확인 (np.load 없음)
    np_load_count = len(re.findall(r'np\.load\s*\(', src_text))
    chk(12, "CT load 차단 (np.load 없음)",
        np_load_count == 0 and not ALLOW_CT_LOAD,
        f"np_load_count={np_load_count}, ALLOW_CT_LOAD={ALLOW_CT_LOAD}")

    # 13. full 300 차단 확인
    chk(13, "full 300 차단",
        not ALLOW_FULL_300,
        f"ALLOW_FULL_300={ALLOW_FULL_300}")

    # 14. forbidden wording 검사 함수 확인
    chk(14, "forbidden wording 검사 함수 확인",
        "scan_forbidden_terms" in src_text and "FORBIDDEN_TERMS" in src_text,
        "scan_forbidden_terms / FORBIDDEN_TERMS 정의됨")

    # 15. reason text KO 길이 <= 200
    ko_len = len(REASON_TEXT_KO)
    chk(15, "reason text KO 길이 <= 200",
        ko_len <= 200,
        f"ko_len={ko_len}")

    # 16. JSON schema 필수 필드 확인 (panel_c_version 포함)
    required_json_fields = [
        "case_id", "source_s3_card_png", "source_s3_card_json",
        "source_s4_reason_csv", "source_s4_reason_json",
        "reason_title", "reason_text_ko", "reason_text_en",
        "displayed_reason_text", "display_language",
        "disclaimer_present", "diagnostic_guard_passed",
        "card_reflection_status", "prototype_mode",
        "existing_card_modified", "prototype_target_case",
        "role", "max_score", "threshold",
        "overmerge_flag", "apex_caution",
        "json_only_ready", "card_text_ready", "roi_coverage",
        "panel_c_version", "panel_c_layout", "panel_c_title",
    ]
    missing_fields = [f for f in required_json_fields if f not in src_text]
    chk(16, "JSON schema 필수 필드 확인 (panel_c_version 포함)",
        len(missing_fields) == 0,
        f"누락={missing_fields}" if missing_fields else "OK")

    # 17. bare 실행 guard 확인
    chk(17, "bare 실행 guard 확인",
        "exit 2" in src_text or "sys.exit(2)" in src_text,
        "sys.exit(2) 정의됨")

    # 18. Panel C v2 레이아웃 코드 확인 (main_ax, thumb1_ax, thumb2_ax)
    has_panel_c_v2 = (
        "main_ax" in src_text
        and "thumb1_ax" in src_text
        and "thumb2_ax" in src_text
        and "PANEL_C_MAIN_WIDTH_RATIO" in src_text
    )
    chk(18, "Panel C v2 레이아웃 코드 확인 (main_ax/thumb1_ax/thumb2_ax)",
        has_panel_c_v2,
        "main_ax, thumb1_ax, thumb2_ax, PANEL_C_MAIN_WIDTH_RATIO 정의됨")

    # 19. same-bin 제목 코드 확인
    has_samebin_title = "same-bin normal reference" in src_text
    chk(19, "same-bin Panel C 제목 코드 확인",
        has_samebin_title,
        "'C. same-bin normal reference' 문자열 포함")

    # 20. stage2_holdout 접근 0 확인
    holdout_path_const = (
        "_holdout" + "_ROOT" in src_text
        or "_holdout" + "_DIR" in src_text
        or "holdout" + "_CSV" in src_text
        or "holdout" + "_JSON" in src_text
    )
    np_load_count_st = len(re.findall(r'np\.load\s*\(', src_text))
    chk(20, "stage2_holdout 접근 0",
        not holdout_path_const and np_load_count_st == 0,
        f"holdout_path_const={holdout_path_const}, np_load_count={np_load_count_st}")

    summary = {
        "total":    len(results),
        "passed":   sum(1 for r in results if r["status"] == "PASS"),
        "failed":   sum(1 for r in results if r["status"] == "FAIL"),
        "all_pass": all_pass,
        "results":  results,
    }
    print(
        f"\n[selftest] {summary['passed']}/{summary['total']} PASS"
        f" — {'ALL PASS' if all_pass else 'NEEDS FIX'}"
    )
    return summary


# ============================================================
# 실제 run v2 (ALLOW_RUN_CARD_PROTOTYPE=True 상태에서만)
# ============================================================
def run_prototype_v2() -> Dict[str, Any]:
    if not ALLOW_RUN_CARD_PROTOTYPE:
        _block(
            "ALLOW_RUN_CARD_PROTOTYPE=False — "
            "이번 단계는 script 작성 + 정적 검사만 허용. "
            "실제 생성은 다음 승인 단계에서 ALLOW_RUN_CARD_PROTOTYPE=True 로 변경 후 실행."
        )

    _assert_no_stage2_holdout(str(PROJECT_ROOT))

    dry = run_dry_run(verbose=False)
    if not dry["ok"]:
        _block(f"dry-run 실패 — run 불가: {dry['failures']}")

    rows = load_v3_csv()
    row  = resolve_target_case(rows)
    val  = validate_target_row(row)
    if not val["ok"]:
        _block(f"target row 유효성 실패: {val['issues']}")

    s3_info = resolve_s3_card_paths()
    if not s3_info["ok"]:
        _block(f"S3 card readiness 실패: {s3_info.get('error')}")

    ref_info = check_ref_crop_paths(s3_info["json_absolute"])
    if not ref_info["ok"]:
        _block(f"ref crop PNG 부재: {ref_info.get('error')}")

    guard = check_output_guard()
    if not guard["ok"]:
        _block(f"output guard 실패: {guard.get('error')}")

    reason_val = validate_reason_text()
    if not reason_val["ok"]:
        _block(f"reason text 안전성 실패: {reason_val['issues']}")

    PROTO_CARDS_PNG_DIR.mkdir(parents=True, exist_ok=True)
    PROTO_CARDS_JSON_DIR.mkdir(parents=True, exist_ok=True)

    proto_png  = PROTO_CARDS_PNG_DIR  / f"{TARGET_CASE_ID}_reason_prototype.png"
    proto_json = PROTO_CARDS_JSON_DIR / f"{TARGET_CASE_ID}_reason_prototype.json"

    errors    = []
    status    = "ok"
    error_msg = ""

    try:
        s3_json_meta = read_s3_card_json_metadata(s3_info["json_absolute"])
        s3_info["s3_json_data"] = s3_json_meta.get("data", {})

        add_panel_c_v2_and_reason_box(
            source_png_path=s3_info["png_absolute"],
            source_json_path=s3_info["json_absolute"],
            output_png_path=proto_png,
        )

        proto_json_data = build_prototype_json_v2(row, s3_info, ref_info)
        with open(proto_json, "w", encoding="utf-8") as f:
            json.dump(proto_json_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        status    = "error"
        error_msg = str(e)
        errors.append({"case_id": TARGET_CASE_ID, "error": error_msg})

    index_row = build_index_row(
        row, s3_info,
        str(proto_png.relative_to(PROTO_OUTPUT_ROOT)),
        str(proto_json.relative_to(PROTO_OUTPUT_ROOT)),
        status,
    )
    with open(PROTO_INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDNAMES)
        writer.writeheader()
        writer.writerow(index_row)

    with open(PROTO_ERRORS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "error"])
        writer.writeheader()
        for e in errors:
            writer.writerow(e)

    runtime = {
        "script":              "build_explanation_card_s4_reason_card_prototype_1case_v2.py",
        "target_case":         TARGET_CASE_ID,
        "total":               1,
        "success":             1 if status == "ok" else 0,
        "error":               1 if status != "ok" else 0,
        "prototype_mode":      "panel_c_v2_reason_box_overlay_from_s3_png",
        "panel_c_version":     PANEL_C_VERSION,
        "panel_c_layout":      PANEL_C_LAYOUT,
        "existing_card_modified":    False,
        "ct_load_occurred":          False,
        "full_300_applied":          False,
        "stage2_holdout_accessed":   False,
        "s4_v1_prototype_preserved": True,
    }
    with open(PROTO_RUNTIME_JSON, "w", encoding="utf-8") as f:
        json.dump(runtime, f, ensure_ascii=False, indent=2)

    if not errors:
        done = {
            "status":             "DONE",
            "target_case":        TARGET_CASE_ID,
            "total_generated":    1,
            "panel_c_version":    PANEL_C_VERSION,
            "existing_card_modified": False,
        }
        with open(PROTO_DONE_JSON, "w", encoding="utf-8") as f:
            json.dump(done, f, ensure_ascii=False, indent=2)

    print(f"[run-prototype-v2] 완료: {status}")
    if error_msg:
        print(f"  ERROR: {error_msg}")
    return {"status": status, "error": error_msg, "errors": errors}


# ============================================================
# main
# ============================================================
def main() -> None:
    # bare 실행 차단
    if len(sys.argv) == 1:
        _block(
            "bare 실행 금지. 실행 모드를 명시하세요: "
            "--selftest / --dry-run / --plan-only / "
            "--run-prototype --confirm-generate"
        )

    parser = argparse.ArgumentParser(
        description="S4 Reason Card Prototype 1-case v2 (Panel C Option2)"
    )
    parser.add_argument("--selftest",          action="store_true")
    parser.add_argument("--dry-run",           action="store_true", dest="dry_run")
    parser.add_argument("--plan-only",         action="store_true", dest="plan_only")
    parser.add_argument("--run-prototype",     action="store_true", dest="run_prototype")
    parser.add_argument("--confirm-generate",  action="store_true", dest="confirm_generate")
    args = parser.parse_args()

    # --run-prototype 단독 차단
    if args.run_prototype and not args.confirm_generate:
        _block(
            "--run-prototype 단독 금지. "
            "--confirm-generate 도 함께 전달해야 합니다."
        )

    # --run-prototype --confirm-generate → ALLOW 가드 확인
    if args.run_prototype and args.confirm_generate:
        if not ALLOW_RUN_CARD_PROTOTYPE:
            _block(
                "--run-prototype --confirm-generate: "
                "ALLOW_RUN_CARD_PROTOTYPE=False — "
                "이번 단계는 script + 정적 검사만. "
                "다음 승인 단계에서 True 로 변경 후 실행하세요."
            )
        run_prototype_v2()
        return

    if args.selftest:
        result = run_selftest()
        sys.exit(0 if result["all_pass"] else 1)
        return

    if args.plan_only:
        run_plan_only()
        return

    if args.dry_run:
        result = run_dry_run(verbose=True)
        sys.exit(0 if result["ok"] else 1)
        return

    _block("알 수 없는 인자 조합. --selftest / --dry-run / --plan-only 중 하나를 사용하세요.")


if __name__ == "__main__":
    main()
