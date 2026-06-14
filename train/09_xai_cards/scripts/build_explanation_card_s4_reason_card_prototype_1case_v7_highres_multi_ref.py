#!/usr/bin/env python3
"""
build_explanation_card_s4_reason_card_prototype_1case_v7_highres_multi_ref.py

S4 Reason Card Prototype 1-case v7 — High-res multi-reference + font fix

목적:
- LUNG1-320__c2 1건에 대해 v6 눈검증 실패 사항 수정 후 재prototype.
- v6 실패 원인:
    1. 한글 glyph □ 깨짐 (DejaVu Sans fallback — 한글 미지원)
    2. normal reference 1개 → 비교 설명력 약함
    3. reason text가 HU 수치 중심, 시각 차이 묘사 부족

v7 핵심 변경:
  1. 한글 font fix: Malgun Gothic → /mnt/c/Windows/Fonts/malgun.ttf 직접 등록
     DejaVu fallback 완전 금지 — 한글 폰트 없으면 BLOCKED
  2. normal reference를 3개로 확장 (matched 1 + additional 2)
     모두 원본 CT slice에서 직접 96×96 crop
  3. reason text 강화: 공기 음영/연부조직·혈관성 구조 시각 차이 중심
  4. v7 output root를 새로 사용 (v1~v6 모두 보존)

v6 대비 layout 변경:
  Row 1: Candidate 96→320px (좌) | Matched normal 96→320px (우)  [동일]
  Row 2: Additional normal1 96→180px (좌) | Additional normal2 96→180px (중)
         | A whole-slice context 190px (우)                       [변경: 2개 추가]
  Row 3: Reason box KO (전체 폭)                                   [텍스트 강화]

CT path resolve 정책 (run 단계):
  - 정상 CT: NORMAL_PATIENT_MANIFEST → ct_hu_npy 컬럼
  - Candidate CT: NSCLC_PATIENT_MANIFEST → ct_hu_npy 컬럼
  - Windows E:\\ → /mnt/e/..., C:\\ → /mnt/c/... 자동 변환
  - 이번 정적 단계에서는 path existence 확인만 (실제 mmap 금지)

절대 금지:
  - 실제 CT npy load (ALLOW_CT_LOAD=False)
  - 실제 prototype PNG/JSON 생성 (ALLOW_RUN_CARD_PROTOTYPE=False)
  - 기존 S3 card PNG/JSON 수정 (ALLOW_ORIGINAL_CARD_MODIFICATION=False)
  - full 300 처리 (ALLOW_FULL_300=False)
  - score/model/threshold 재계산
  - stage2_holdout 접근
  - 기존 v1~v6 prototype 수정/삭제/덮어쓰기
  - 32×32 reference PNG 확대 방식 카드 사용 (USE_32PX_REF_PNG=False)
  - old Panel C 카드 포함
  - S3 PNG full card paste

실행 모드:
  bare 실행                                  → BLOCKED exit 2
  --selftest                                 → selftest items 검사
  --dry-run                                  → 입력 파일 존재 + output guard 확인
  --plan-only                                → dry-run + 배치 계획 출력
  --run-prototype                            → 단독 BLOCKED exit 2
  --run-prototype --confirm-generate         → ALLOW_RUN_CARD_PROTOTYPE=False이면 BLOCKED

syntax check:
  python -m py_compile scripts/build_explanation_card_s4_reason_card_prototype_1case_v7_highres_multi_ref.py
"""

import argparse
import csv
import json
import os
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ============================================================
# 최상위 가드
# ============================================================
ALLOW_RUN_CARD_PROTOTYPE         = False   # True 변경은 사용자 승인 필수
ALLOW_ORIGINAL_CARD_MODIFICATION = False   # 항구 False
ALLOW_CT_LOAD                    = False   # run 단계 승인 전 False. read-only mmap만 허용
ALLOW_FULL_300                   = False   # 항구 False

# ============================================================
# v7 핵심 전략 식별자
# ============================================================
FULL_CARD_PASTE_STRATEGY     = "DISABLED"   # S3 PNG 전체 paste — v4부터 폐기
SHOW_OLD_REF2_REF3_ON_CARD   = False        # v5 방식 Ref2/Ref3 표시 금지
USE_32PX_REF_PNG             = False        # 32×32 PNG 확대 방식 완전 폐기
USE_ORIGINAL_CT_FOR_REF      = True         # v7: 원본 CT slice에서 직접 crop
USE_ORIGINAL_CT_FOR_CAND     = True         # v7: candidate도 원본 CT에서 crop
ADDITIONAL_REFS_DISPLAYED_ON_CARD = True    # v7: additional normals 카드 표시
OLD_PANEL_C_USED             = False        # v7: old Panel C 완전 제거

# ============================================================
# prototype 대상 (1건)
# ============================================================
TARGET_CASE_ID      = "LUNG1-320__c2"
TARGET_PATIENT_ID   = "LUNG1-320"
TARGET_SAFE_ID      = "NSCLC_LUNG1-320__95de24d86f"

# ============================================================
# reason text v7 — 시각 차이 중심, 면책문구 필수
# ============================================================
REASON_TITLE = "Reason cue / 검토 근거 후보"

REASON_TEXT_KO = (
    "같은 lower_peripheral 위치의 정상 CT crop들과 같은 크기/윈도우로 비교했을 때, "
    "후보 crop은 공기 음영보다 밝은 연부조직/혈관성 구조가 더 넓게 보였고 "
    "HU 밀도도 약 +245 HU 높게 측정되었습니다. "
    "이는 PaDiM high-response를 해석하기 위한 시각적 참고 단서이며, 진단 의미는 아닙니다."
)

REASON_TEXT_EN = (
    "Compared with same-bin normal CT crops in the lower_peripheral region "
    "using the same crop size and window, the candidate crop showed a broader "
    "bright soft-tissue or vessel-like structure and an HU density difference "
    "of about +245 HU. "
    "This is a visual reference cue for interpreting the PaDiM high response, "
    "not a diagnosis."
)

# ============================================================
# v7 layout 설정
# ============================================================
LAYOUT_VERSION    = "highres_multi_reference_v1"
PROTOTYPE_VERSION = "v7"

CANDIDATE_LABEL    = "Candidate — original CT crop (96×96→320px)"
MAIN_REF_LABEL     = "Matched normal — original CT crop\n(96×96→320px, lower_peripheral)"
ADDREF1_LABEL      = "Normal example 1\n(96×96→180px)"
ADDREF2_LABEL      = "Normal example 2\n(96×96→180px)"
PANEL_A_LABEL      = "A. Whole slice (위치 확인)"
COMPARISON_SECTION_LABEL = "Candidate vs same-bin normal references (original CT, 동일 조건)"

# ============================================================
# crop 설계 — candidate
# ============================================================
CANDIDATE_SLICE_Z    = 140
CANDIDATE_CROP_Y0    = 288
CANDIDATE_CROP_X0    = 176
CANDIDATE_CROP_Y1    = 384
CANDIDATE_CROP_X1    = 272
CANDIDATE_CROP_SIZE_H = CANDIDATE_CROP_Y1 - CANDIDATE_CROP_Y0   # 96
CANDIDATE_CROP_SIZE_W = CANDIDATE_CROP_X1 - CANDIDATE_CROP_X0   # 96

# ============================================================
# crop 설계 — matched normal reference (subset4...d3e03b0ce0)
# ============================================================
NORMAL_REF_SAFE_ID    = (
    "subset4_1.3.6.1.4.1.14519.5.2.1.6279.6001."
    "200837896655745926888305239398__d3e03b0ce0"
)
NORMAL_REF_PATIENT_ID = (
    "subset4_1.3.6.1.4.1.14519.5.2.1.6279.6001."
    "200837896655745926888305239398"
)
NORMAL_REF_SLICE_Z    = 179
NORMAL_REF_ORIG_Y0    = 240
NORMAL_REF_ORIG_X0    = 368
NORMAL_REF_ORIG_Y1    = 272
NORMAL_REF_ORIG_X1    = 400
NORMAL_REF_CENTER_Y   = (NORMAL_REF_ORIG_Y0 + NORMAL_REF_ORIG_Y1) // 2   # 256
NORMAL_REF_CENTER_X   = (NORMAL_REF_ORIG_X0 + NORMAL_REF_ORIG_X1) // 2   # 384
NORMAL_REF_CROP_Y0    = NORMAL_REF_CENTER_Y - 48   # 208
NORMAL_REF_CROP_X0    = NORMAL_REF_CENTER_X - 48   # 336
NORMAL_REF_CROP_Y1    = NORMAL_REF_CENTER_Y + 48   # 304
NORMAL_REF_CROP_X1    = NORMAL_REF_CENTER_X + 48   # 432
NORMAL_REF_CROP_SIZE_H = NORMAL_REF_CROP_Y1 - NORMAL_REF_CROP_Y0   # 96
NORMAL_REF_CROP_SIZE_W = NORMAL_REF_CROP_X1 - NORMAL_REF_CROP_X0   # 96

# ============================================================
# crop 설계 — additional normal ref 1 (subset9...9e5f73ce9b)
# ============================================================
ADDREF1_SAFE_ID    = (
    "subset9_1.3.6.1.4.1.14519.5.2.1.6279.6001."
    "330043769832606379655473292782__9e5f73ce9b"
)
ADDREF1_SLICE_Z    = 165
ADDREF1_ORIG_Y0    = 272
ADDREF1_ORIG_X0    = 80
ADDREF1_ORIG_Y1    = 304
ADDREF1_ORIG_X1    = 112
ADDREF1_CENTER_Y   = (ADDREF1_ORIG_Y0 + ADDREF1_ORIG_Y1) // 2   # 288
ADDREF1_CENTER_X   = (ADDREF1_ORIG_X0 + ADDREF1_ORIG_X1) // 2   # 96
ADDREF1_CROP_Y0    = ADDREF1_CENTER_Y - 48   # 240
ADDREF1_CROP_X0    = ADDREF1_CENTER_X - 48   # 48
ADDREF1_CROP_Y1    = ADDREF1_CENTER_Y + 48   # 336
ADDREF1_CROP_X1    = ADDREF1_CENTER_X + 48   # 144
ADDREF1_CROP_SIZE_H = ADDREF1_CROP_Y1 - ADDREF1_CROP_Y0   # 96
ADDREF1_CROP_SIZE_W = ADDREF1_CROP_X1 - ADDREF1_CROP_X0   # 96

# ============================================================
# crop 설계 — additional normal ref 2 (subset3...e46a38ea0a)
# ============================================================
ADDREF2_SAFE_ID    = (
    "subset3_1.3.6.1.4.1.14519.5.2.1.6279.6001."
    "268589491017129166376960414534__e46a38ea0a"
)
ADDREF2_SLICE_Z    = 178
ADDREF2_ORIG_Y0    = 288
ADDREF2_ORIG_X0    = 112
ADDREF2_ORIG_Y1    = 320
ADDREF2_ORIG_X1    = 144
ADDREF2_CENTER_Y   = (ADDREF2_ORIG_Y0 + ADDREF2_ORIG_Y1) // 2   # 304
ADDREF2_CENTER_X   = (ADDREF2_ORIG_X0 + ADDREF2_ORIG_X1) // 2   # 128
ADDREF2_CROP_Y0    = ADDREF2_CENTER_Y - 48   # 256
ADDREF2_CROP_X0    = ADDREF2_CENTER_X - 48   # 80
ADDREF2_CROP_Y1    = ADDREF2_CENTER_Y + 48   # 352
ADDREF2_CROP_X1    = ADDREF2_CENTER_X + 48   # 176
ADDREF2_CROP_SIZE_H = ADDREF2_CROP_Y1 - ADDREF2_CROP_Y0   # 96
ADDREF2_CROP_SIZE_W = ADDREF2_CROP_X1 - ADDREF2_CROP_X0   # 96

# ============================================================
# display size
# ============================================================
CROP_SOURCE_SIZE      = 96    # 원본 CT crop size (모든 ref 동일)
DISPLAY_SIZE          = 320   # candidate + matched normal display (LANCZOS upscale)
ADD_DISPLAY_SIZE      = 180   # additional normals display size
CONTEXT_DISPLAY_SIZE  = 190   # A whole-slice context

# ============================================================
# HU window — S3와 동일 lung window
# ============================================================
HU_WIN_CENTER = -600
HU_WIN_WIDTH  = 1500
HU_WIN_LOW    = HU_WIN_CENTER - HU_WIN_WIDTH // 2   # -1350
HU_WIN_HIGH   = HU_WIN_CENTER + HU_WIN_WIDTH // 2   # 150
WINDOWING_NOTE = (
    "center=-600HU, width=1500HU — S3 card 생성 파라미터와 동일. "
    "candidate/normal/additional refs 모두 동일 window 적용."
)

# ============================================================
# canvas 크기
# ============================================================
CANVAS_WIDTH_PX = 1210
CANVAS_DPI      = 110
MARGIN          = 20
LABEL_H         = 24

ROW1_H_PX = DISPLAY_SIZE + LABEL_H              # 344
ROW2_H_PX = ADD_DISPLAY_SIZE + LABEL_H          # 204
ROW3_H_PX = 190                                  # reason box
CANVAS_HEIGHT_ESTIMATE_PX = (
    ROW1_H_PX + ROW2_H_PX + ROW3_H_PX + MARGIN * 5
)  # ~848

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
    "원인이다", "원인으로 판단",
    "확진",
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

V3_OUTPUT_DIR  = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_v3"
V3_OUTPUT_CSV  = V3_OUTPUT_DIR / "s4_reason_layer_integrated_smoke_v3.csv"
V3_OUTPUT_JSON = V3_OUTPUT_DIR / "s4_reason_layer_integrated_smoke_v3.json"
V3_DONE_JSON   = V3_OUTPUT_DIR / "DONE.json"

S3_CARD_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s3_expansion_cards_v2_fontfix"
)
S3_CARDS_PNG_DIR  = S3_CARD_ROOT / "cards_png"
S3_CARDS_JSON_DIR = S3_CARD_ROOT / "cards_json"

REF_BANK_FULL = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "reference_bank_v1/full"
)
REF_CROP_MANIFEST = REF_BANK_FULL / "reference_crop_manifest.csv"

# ── patient manifest ──────────────────────────────────────────
NORMAL_PATIENT_MANIFEST = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/"
    "manifests/patient_manifest.csv"
)
NSCLC_PATIENT_MANIFEST = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/"
    "manifests/patient_manifest.csv"
)

# ── CT npy 경로 (manifest resolve 결과) ──────────────────────
CANDIDATE_CT_NPY = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/"
    "volumes_npy/NSCLC_LUNG1-320__95de24d86f/ct_hu.npy"
)
NORMAL_REF_CT_NPY = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/"
    "volumes_npy/"
    "subset4_1.3.6.1.4.1.14519.5.2.1.6279.6001."
    "200837896655745926888305239398__d3e03b0ce0/ct_hu.npy"
)
ADDREF1_CT_NPY = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/"
    "volumes_npy/"
    "subset9_1.3.6.1.4.1.14519.5.2.1.6279.6001."
    "330043769832606379655473292782__9e5f73ce9b/ct_hu.npy"
)
ADDREF2_CT_NPY = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/"
    "volumes_npy/"
    "subset3_1.3.6.1.4.1.14519.5.2.1.6279.6001."
    "268589491017129166376960414534__e46a38ea0a/ct_hu.npy"
)

# ── 보존 전용 경로 (수정/삭제 금지) ──────────────────────────
CARDS_VIS_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
)
PROTO_V1_OUTPUT_ROOT = CARDS_VIS_ROOT / "s4_reason_card_prototype_1case_v1"
PROTO_V2_OUTPUT_ROOT = CARDS_VIS_ROOT / "s4_reason_card_prototype_1case_v2"
PROTO_V3_OUTPUT_ROOT = CARDS_VIS_ROOT / "s4_reason_card_prototype_1case_v3"
PROTO_V4_OUTPUT_ROOT = CARDS_VIS_ROOT / "s4_reason_card_prototype_1case_v4"
PROTO_V5_OUTPUT_ROOT = CARDS_VIS_ROOT / "s4_reason_card_prototype_1case_v5"
PROTO_V6_OUTPUT_ROOT = CARDS_VIS_ROOT / "s4_reason_card_prototype_1case_v6_highres_ref"

# ── v7 출력 경로 ─────────────────────────────────────────────
PROTO_OUTPUT_ROOT    = CARDS_VIS_ROOT / "s4_reason_card_prototype_1case_v7_highres_multi_ref"
PROTO_CARDS_PNG_DIR  = PROTO_OUTPUT_ROOT / "cards_png"
PROTO_CARDS_JSON_DIR = PROTO_OUTPUT_ROOT / "cards_json"
PROTO_INDEX_CSV      = PROTO_OUTPUT_ROOT / "index_cards.csv"
PROTO_RUNTIME_JSON   = PROTO_OUTPUT_ROOT / "runtime_summary.json"
PROTO_ERRORS_CSV     = PROTO_OUTPUT_ROOT / "errors.csv"
PROTO_DONE_JSON      = PROTO_OUTPUT_ROOT / "DONE.json"


# ============================================================
# 한글 font resolve (DejaVu fallback 금지)
# ============================================================
_KOREAN_FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/malgun.ttf",
    "/mnt/c/Windows/Fonts/Hancom Gothic Regular.ttf",
    "/mnt/c/Windows/Fonts/HANDotum.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def resolve_korean_font() -> Optional[str]:
    """Malgun Gothic 또는 대체 한글 폰트 경로 resolve.
    DejaVu fallback 금지 — 반환값이 None이면 run에서 BLOCKED."""
    for path in _KOREAN_FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


SELECTED_FONT_PATH   = resolve_korean_font()
SELECTED_FONT_FAMILY = (
    os.path.basename(SELECTED_FONT_PATH) if SELECTED_FONT_PATH else "NOT_FOUND"
)


# ============================================================
# 유틸
# ============================================================
def _block(reason: str, code: int = 2) -> None:
    print(f"[BLOCKED] {reason}", file=sys.stderr)
    sys.exit(code)


def _wsl_path(win_path: str) -> str:
    """Windows drive path → WSL /mnt/... 변환."""
    if win_path.startswith(("E:\\", "e:\\")):
        return "/mnt/e/" + win_path[3:].replace("\\", "/")
    if win_path.startswith(("C:\\", "c:\\")):
        return "/mnt/c/" + win_path[3:].replace("\\", "/")
    return win_path


def scan_forbidden_terms(text: str) -> List[str]:
    hits = []
    lower = text.lower()
    for term in FORBIDDEN_TERMS:
        if term.lower() in lower:
            hits.append(term)
    return hits


def resolve_normal_ct_path_from_manifest(safe_id: str) -> Optional[str]:
    """NORMAL_PATIENT_MANIFEST에서 safe_id의 ct_hu_npy 경로 resolve."""
    if not NORMAL_PATIENT_MANIFEST.exists():
        return None
    with open(NORMAL_PATIENT_MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("safe_id", "") == safe_id:
                ct_path = row.get("ct_hu_npy", "")
                if ct_path:
                    return _wsl_path(ct_path)
    return None


def resolve_candidate_ct_path_from_manifest(safe_id: str) -> Optional[str]:
    """NSCLC_PATIENT_MANIFEST에서 safe_id의 ct_hu_npy 경로 resolve."""
    if not NSCLC_PATIENT_MANIFEST.exists():
        return None
    with open(NSCLC_PATIENT_MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("safe_id", "") == safe_id:
                ct_path = row.get("ct_hu_npy", "")
                if ct_path:
                    return _wsl_path(ct_path)
    return None


# ============================================================
# 검사 함수
# ============================================================
def _assert_no_stage2_holdout(path: pathlib.Path) -> None:
    for tok in STAGE2_HOLDOUT_TOKENS:
        if tok in str(path):
            _block(f"stage2_holdout 경로 접근 금지: {path}")


def check_output_guard() -> Dict[str, Any]:
    """v7 output root가 기존 경로와 충돌하지 않는지 확인."""
    prior_roots = [
        PROTO_V1_OUTPUT_ROOT,
        PROTO_V2_OUTPUT_ROOT,
        PROTO_V3_OUTPUT_ROOT,
        PROTO_V4_OUTPUT_ROOT,
        PROTO_V5_OUTPUT_ROOT,
        PROTO_V6_OUTPUT_ROOT,
        S3_CARD_ROOT,
    ]
    _assert_no_stage2_holdout(PROTO_OUTPUT_ROOT)

    for pr in prior_roots:
        if PROTO_OUTPUT_ROOT == pr:
            return {"ok": False, "error": f"v7 output root가 기존 경로와 동일: {pr.name}"}
        if str(PROTO_OUTPUT_ROOT).startswith(str(pr) + "/"):
            return {"ok": False, "error": f"v7 output root가 {pr.name} 하위임"}
        if str(pr).startswith(str(PROTO_OUTPUT_ROOT) + "/"):
            return {"ok": False, "error": f"기존 경로 {pr.name}가 v7 output 하위임"}

    if PROTO_DONE_JSON.exists():
        return {"ok": False, "error": f"DONE.json 이미 존재: {PROTO_DONE_JSON}"}

    return {"ok": True, "v7_root": PROTO_OUTPUT_ROOT.name}


def validate_crop_sizes() -> Dict[str, Any]:
    """모든 crop size가 96×96인지 확인."""
    issues = []
    results: Dict[str, Any] = {}

    for label, h, w in [
        ("candidate",   CANDIDATE_CROP_SIZE_H,   CANDIDATE_CROP_SIZE_W),
        ("normal_ref",  NORMAL_REF_CROP_SIZE_H,  NORMAL_REF_CROP_SIZE_W),
        ("addref1",     ADDREF1_CROP_SIZE_H,      ADDREF1_CROP_SIZE_W),
        ("addref2",     ADDREF2_CROP_SIZE_H,      ADDREF2_CROP_SIZE_W),
    ]:
        ok = (h == 96 and w == 96)
        if not ok:
            issues.append(f"{label} crop size {h}×{w} ≠ 96×96")
        results[label] = {"h": h, "w": w, "ok": ok}

    same_size = all(results[k]["ok"] for k in results)
    results["all_same_96x96"] = same_size

    return {"ok": len(issues) == 0, "issues": issues, "details": results}


def validate_display_sizes() -> Dict[str, Any]:
    """display size 기준 확인."""
    issues = []
    if DISPLAY_SIZE != 320:
        issues.append(f"DISPLAY_SIZE={DISPLAY_SIZE} ≠ 320")
    if ADD_DISPLAY_SIZE < 160:
        issues.append(f"ADD_DISPLAY_SIZE={ADD_DISPLAY_SIZE} < 160 (너무 작음)")
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "main_display": DISPLAY_SIZE,
        "add_display": ADD_DISPLAY_SIZE,
        "context_display": CONTEXT_DISPLAY_SIZE,
    }


def validate_reason_text() -> Dict[str, Any]:
    """reason text v7 검사."""
    issues = []

    has_disclaimer = "진단 의미는 아닙니다" in REASON_TEXT_KO
    if not has_disclaimer:
        issues.append("'진단 의미는 아닙니다' 없음")

    has_visual_cue = "시각적 참고 단서" in REASON_TEXT_KO
    if not has_visual_cue:
        issues.append("'시각적 참고 단서' 없음")

    has_245hu = "약 +245 HU" in REASON_TEXT_KO
    if not has_245hu:
        issues.append("'약 +245 HU' 없음")

    ko_hits = scan_forbidden_terms(REASON_TEXT_KO)
    en_hits = scan_forbidden_terms(REASON_TEXT_EN)
    if ko_hits:
        issues.append(f"KO 금지어: {ko_hits}")
    if en_hits:
        issues.append(f"EN 금지어: {en_hits}")

    ko_sentences = [s.strip() for s in REASON_TEXT_KO.split(".") if s.strip()]
    if len(ko_sentences) < 2:
        issues.append(f"KO 문장 수 부족: {len(ko_sentences)}문장 (2문장 이상 필요)")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "has_disclaimer": has_disclaimer,
        "has_visual_cue": has_visual_cue,
        "has_245hu": has_245hu,
        "ko_forbidden": ko_hits,
        "en_forbidden": en_hits,
        "ko_len": len(REASON_TEXT_KO),
        "ko_sentence_count": len(ko_sentences),
    }


def validate_font() -> Dict[str, Any]:
    """한글 font 유효성 확인."""
    issues = []

    if SELECTED_FONT_PATH is None:
        issues.append(
            "한글 폰트 없음 — Malgun Gothic(/mnt/c/Windows/Fonts/malgun.ttf) 확인 필요"
        )
        return {"ok": False, "issues": issues, "path": None, "family": "NOT_FOUND"}

    if "dejavusans" in SELECTED_FONT_FAMILY.lower().replace(" ", ""):
        issues.append("DejaVu fallback 금지 — 한글 폰트 지정 필요")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "path": SELECTED_FONT_PATH,
        "family": SELECTED_FONT_FAMILY,
    }


def build_v7_canvas_design() -> Dict[str, Any]:
    """v7 카드 레이아웃 설계 정보 반환 (실제 생성 없음, CT load 없음)."""
    design = {
        "prototype_version": PROTOTYPE_VERSION,
        "layout_version": LAYOUT_VERSION,
        "target_case_id": TARGET_CASE_ID,
        "canvas_width_px": CANVAS_WIDTH_PX,
        "canvas_height_estimate_px": CANVAS_HEIGHT_ESTIMATE_PX,
        "font": {
            "selected_font_path": SELECTED_FONT_PATH,
            "selected_font_family": SELECTED_FONT_FAMILY,
            "dejavu_fallback_allowed": False,
        },
        "rows": [
            {
                "row": 1,
                "label": "Main comparison (Candidate vs Matched normal)",
                "height_px": ROW1_H_PX,
                "left": {
                    "label": CANDIDATE_LABEL,
                    "source": "original_candidate_CT_crop",
                    "slice_z": CANDIDATE_SLICE_Z,
                    "crop_bbox_yxyx": [
                        CANDIDATE_CROP_Y0, CANDIDATE_CROP_X0,
                        CANDIDATE_CROP_Y1, CANDIDATE_CROP_X1,
                    ],
                    "crop_size_hw": [CANDIDATE_CROP_SIZE_H, CANDIDATE_CROP_SIZE_W],
                    "display_size_px": DISPLAY_SIZE,
                },
                "right": {
                    "label": MAIN_REF_LABEL,
                    "source": "original_normal_CT_crop",
                    "slice_z": NORMAL_REF_SLICE_Z,
                    "crop_bbox_yxyx": [
                        NORMAL_REF_CROP_Y0, NORMAL_REF_CROP_X0,
                        NORMAL_REF_CROP_Y1, NORMAL_REF_CROP_X1,
                    ],
                    "crop_size_hw": [NORMAL_REF_CROP_SIZE_H, NORMAL_REF_CROP_SIZE_W],
                    "display_size_px": DISPLAY_SIZE,
                },
            },
            {
                "row": 2,
                "label": "Additional normal examples + context",
                "height_px": ROW2_H_PX,
                "left": {
                    "label": ADDREF1_LABEL,
                    "source": "original_normal_CT_crop",
                    "slice_z": ADDREF1_SLICE_Z,
                    "crop_bbox_yxyx": [
                        ADDREF1_CROP_Y0, ADDREF1_CROP_X0,
                        ADDREF1_CROP_Y1, ADDREF1_CROP_X1,
                    ],
                    "crop_size_hw": [ADDREF1_CROP_SIZE_H, ADDREF1_CROP_SIZE_W],
                    "display_size_px": ADD_DISPLAY_SIZE,
                },
                "center": {
                    "label": ADDREF2_LABEL,
                    "source": "original_normal_CT_crop",
                    "slice_z": ADDREF2_SLICE_Z,
                    "crop_bbox_yxyx": [
                        ADDREF2_CROP_Y0, ADDREF2_CROP_X0,
                        ADDREF2_CROP_Y1, ADDREF2_CROP_X1,
                    ],
                    "crop_size_hw": [ADDREF2_CROP_SIZE_H, ADDREF2_CROP_SIZE_W],
                    "display_size_px": ADD_DISPLAY_SIZE,
                },
                "right": {
                    "label": PANEL_A_LABEL,
                    "source": "S3_PNG_A_panel_crop",
                    "display_size_px": CONTEXT_DISPLAY_SIZE,
                },
            },
            {
                "row": 3,
                "label": "Reason box",
                "height_px": ROW3_H_PX,
                "title": REASON_TITLE,
                "reason_text_ko": REASON_TEXT_KO,
                "display_language": "KO",
                "disclaimer_present": True,
                "visual_cue_present": True,
                "font": SELECTED_FONT_FAMILY,
            },
        ],
        "windowing": {
            "hu_win_center": HU_WIN_CENTER,
            "hu_win_width": HU_WIN_WIDTH,
            "hu_win_low": HU_WIN_LOW,
            "hu_win_high": HU_WIN_HIGH,
            "applied_to": ["candidate", "normal_ref", "addref1", "addref2"],
            "note": WINDOWING_NOTE,
        },
        "disabled_strategies": {
            "FULL_CARD_PASTE_STRATEGY": FULL_CARD_PASTE_STRATEGY,
            "USE_32PX_REF_PNG": USE_32PX_REF_PNG,
            "OLD_PANEL_C_USED": OLD_PANEL_C_USED,
            "SHOW_OLD_REF2_REF3_ON_CARD": SHOW_OLD_REF2_REF3_ON_CARD,
        },
        "guards": {
            "ALLOW_RUN_CARD_PROTOTYPE": ALLOW_RUN_CARD_PROTOTYPE,
            "ALLOW_CT_LOAD": ALLOW_CT_LOAD,
            "ALLOW_ORIGINAL_CARD_MODIFICATION": ALLOW_ORIGINAL_CARD_MODIFICATION,
            "ALLOW_FULL_300": ALLOW_FULL_300,
        },
    }
    return design


# ============================================================
# v7 실제 생성 함수군 (run 단계 전용 — ALLOW_CT_LOAD guard 아래에서만 호출)
# ============================================================

def load_ct_mmap(path: pathlib.Path) -> np.ndarray:
    """read-only mmap으로 CT npy 로드. ALLOW_CT_LOAD=True 아래에서만 호출."""
    if not ALLOW_CT_LOAD:
        _block("load_ct_mmap: ALLOW_CT_LOAD=False — CT load 금지")
    _assert_no_stage2_holdout(path)
    ct = np.load(str(path), mmap_mode="r")
    return ct


def clamp_bbox(
    y0: int, x0: int, y1: int, x1: int,
    shape_hw: Tuple[int, int],
    crop_size: int = CROP_SOURCE_SIZE,
) -> Tuple[int, int, int, int, List[str]]:
    """bbox를 slice 경계 내로 clip. (cy0, cx0, cy1, cx1, warnings) 반환."""
    H, W = shape_hw
    warns: List[str] = []
    cy0 = int(np.clip(y0, 0, H))
    cx0 = int(np.clip(x0, 0, W))
    cy1 = int(np.clip(y1, 0, H))
    cx1 = int(np.clip(x1, 0, W))
    if cy0 != y0 or cx0 != x0 or cy1 != y1 or cx1 != x1:
        warns.append(
            f"bbox clamped: [{y0},{x0},{y1},{x1}] → [{cy0},{cx0},{cy1},{cx1}]"
            f" (slice H={H},W={W})"
        )
    crop_h = cy1 - cy0
    crop_w = cx1 - cx0
    if crop_h != crop_size or crop_w != crop_size:
        warns.append(
            f"crop size after clamp: {crop_h}×{crop_w} ≠ {crop_size}×{crop_size}"
        )
    return cy0, cx0, cy1, cx1, warns


def window_hu(
    crop: np.ndarray,
    center: float = HU_WIN_CENTER,
    width: float = HU_WIN_WIDTH,
) -> np.ndarray:
    """HU windowing → [0,255] uint8. HU_WIN_CENTER/HU_WIN_WIDTH 기본값으로 양측 동일 적용."""
    low = center - width / 2.0
    high = center + width / 2.0
    norm = (np.asarray(crop, dtype=np.float32) - low) / (high - low)
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def extract_ct_crop(
    ct_vol: np.ndarray,
    slice_z: int,
    y0: int, x0: int, y1: int, x1: int,
    label: str = "crop",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """CT volume에서 특정 slice의 crop 추출. clamp_bbox 적용."""
    H = ct_vol.shape[1]
    W = ct_vol.shape[2]
    cy0, cx0, cy1, cx1, bbox_warns = clamp_bbox(y0, x0, y1, x1, (H, W))
    crop = np.array(ct_vol[slice_z, cy0:cy1, cx0:cx1])
    meta: Dict[str, Any] = {
        "label": label,
        "slice_z": slice_z,
        "ct_shape": list(ct_vol.shape),
        "original_bbox_yxyx": [y0, x0, y1, x1],
        "clamped_bbox_yxyx": [cy0, cx0, cy1, cx1],
        "crop_shape": list(crop.shape),
        "bbox_warnings": bbox_warns,
        "hu_win_center": HU_WIN_CENTER,
        "hu_win_width": HU_WIN_WIDTH,
        "hu_win_low": HU_WIN_LOW,
        "hu_win_high": HU_WIN_HIGH,
    }
    return crop, meta


def crop_s3_png_panel(
    s3_png_path: pathlib.Path,
    panel: str = "A",
    target_size: int = CONTEXT_DISPLAY_SIZE,
) -> Any:
    """S3 PNG에서 A panel을 crop하여 PIL Image로 반환 (read-only)."""
    from PIL import Image as _PILImage
    img = _PILImage.open(str(s3_png_path))
    W, H = img.size
    half_w = W // 2
    half_h = H // 2
    if panel == "A":
        region = img.crop((0, 0, half_w, half_h))
    elif panel == "D":
        region = img.crop((half_w, half_h, W, H))
    else:
        raise ValueError(f"panel must be 'A' or 'D', got {panel!r}")
    return region.resize((target_size, target_size), _PILImage.LANCZOS)


def _load_korean_font(size: int) -> Any:
    """한글 폰트 로드. SELECTED_FONT_PATH 없으면 BLOCKED."""
    from PIL import ImageFont as _IFont
    if not SELECTED_FONT_PATH:
        _block(
            "_load_korean_font: 한글 폰트 없음 — Malgun Gothic 필요. "
            "DejaVu fallback 금지."
        )
    if "dejavu" in SELECTED_FONT_FAMILY.lower():
        _block(
            "_load_korean_font: DejaVu fallback 금지 — 한글 폰트 필요"
        )
    return _IFont.truetype(SELECTED_FONT_PATH, size)


def _draw_wrapped_text_ko(
    draw: Any,
    text: str,
    font: Any,
    x: int,
    y: int,
    max_width: int,
    line_spacing: int = 20,
    fill: Tuple[int, int, int] = (50, 50, 50),
) -> None:
    """한글 포함 텍스트 자동 줄바꿈. 글자/어절 단위로 분리."""
    # 어절(공백) 단위로 분리, 각 어절은 한글 포함 가능
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        test = (current + " " + word).strip() if current else word
        try:
            bbox = font.getbbox(test)
            w = bbox[2] - bbox[0]
        except AttributeError:
            w = len(test) * 14  # 한글 추정 너비
        if w > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_spacing), line, fill=fill, font=font)


def build_v7_card_image(
    cand_u8: np.ndarray,
    ref_u8: np.ndarray,
    add1_u8: np.ndarray,
    add2_u8: np.ndarray,
    s3_png_path: pathlib.Path,
) -> Any:
    """v7 canvas 생성 (PIL Image 반환).

    Row 1: Candidate 320px (좌) | Matched normal 320px (우)
    Row 2: Additional normal1 180px (좌) | Additional normal2 180px (중) | A context 190px (우)
    Row 3: Reason box (전체 폭, 한글 font)
    """
    from PIL import Image as _PILImage, ImageDraw as _IDraw

    # 한글 폰트 로드 (DejaVu fallback 금지)
    font_label  = _load_korean_font(12)
    font_reason = _load_korean_font(13)
    font_title  = _load_korean_font(15)

    # Row 1 이미지 (grayscale → RGB)
    cand_img = _PILImage.fromarray(np.stack([cand_u8] * 3, axis=-1)).resize(
        (DISPLAY_SIZE, DISPLAY_SIZE), _PILImage.LANCZOS
    )
    ref_img = _PILImage.fromarray(np.stack([ref_u8] * 3, axis=-1)).resize(
        (DISPLAY_SIZE, DISPLAY_SIZE), _PILImage.LANCZOS
    )

    # Row 2 이미지
    add1_img = _PILImage.fromarray(np.stack([add1_u8] * 3, axis=-1)).resize(
        (ADD_DISPLAY_SIZE, ADD_DISPLAY_SIZE), _PILImage.LANCZOS
    )
    add2_img = _PILImage.fromarray(np.stack([add2_u8] * 3, axis=-1)).resize(
        (ADD_DISPLAY_SIZE, ADD_DISPLAY_SIZE), _PILImage.LANCZOS
    )

    # Row 2 context (S3 PNG A panel read-only crop)
    ctx_a = crop_s3_png_panel(s3_png_path, "A", CONTEXT_DISPLAY_SIZE)

    # canvas 크기 계산
    row1_h = DISPLAY_SIZE + LABEL_H
    row2_h = ADD_DISPLAY_SIZE + LABEL_H
    row3_h = ROW3_H_PX

    canvas_w = CANVAS_WIDTH_PX
    canvas_h = row1_h + row2_h + row3_h + MARGIN * 5
    canvas = _PILImage.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))
    draw = _IDraw.Draw(canvas)

    # ── Row 1 배치 ───────────────────────────────────────────
    y_row1 = MARGIN

    # section label
    draw.text(
        (MARGIN, y_row1),
        COMPARISON_SECTION_LABEL,
        fill=(60, 60, 60), font=font_label,
    )

    # candidate (좌)
    cand_y = y_row1 + LABEL_H
    draw.text((MARGIN, cand_y), CANDIDATE_LABEL, fill=(30, 30, 30), font=font_label)
    canvas.paste(cand_img, (MARGIN, cand_y + LABEL_H))

    # matched normal (우)
    ref_x = MARGIN + DISPLAY_SIZE + MARGIN * 2
    draw.text(
        (ref_x, cand_y),
        MAIN_REF_LABEL.replace("\n", " "),
        fill=(30, 30, 30), font=font_label,
    )
    canvas.paste(ref_img, (ref_x, cand_y + LABEL_H))

    # ── Row 2 배치 ───────────────────────────────────────────
    y_row2 = y_row1 + row1_h + MARGIN * 2

    # additional normal 1 (좌)
    add1_x = MARGIN
    draw.text(
        (add1_x, y_row2),
        ADDREF1_LABEL.replace("\n", " "),
        fill=(30, 30, 30), font=font_label,
    )
    canvas.paste(add1_img, (add1_x, y_row2 + LABEL_H))

    # additional normal 2 (중)
    add2_x = MARGIN + ADD_DISPLAY_SIZE + MARGIN
    draw.text(
        (add2_x, y_row2),
        ADDREF2_LABEL.replace("\n", " "),
        fill=(30, 30, 30), font=font_label,
    )
    canvas.paste(add2_img, (add2_x, y_row2 + LABEL_H))

    # A context (우)
    ctx_x = MARGIN + ADD_DISPLAY_SIZE * 2 + MARGIN * 2
    draw.text((ctx_x, y_row2), PANEL_A_LABEL, fill=(30, 30, 30), font=font_label)
    canvas.paste(ctx_a, (ctx_x, y_row2 + LABEL_H))

    # ── Row 3 reason box ─────────────────────────────────────
    y_row3 = y_row2 + row2_h + MARGIN
    box_x0, box_y0 = MARGIN, y_row3
    box_x1 = canvas_w - MARGIN
    box_y1 = y_row3 + row3_h
    draw.rectangle(
        (box_x0, box_y0, box_x1, box_y1),
        outline=(100, 100, 100), width=1,
    )
    draw.text(
        (box_x0 + 8, box_y0 + 6),
        REASON_TITLE,
        fill=(30, 30, 30), font=font_title,
    )
    _draw_wrapped_text_ko(
        draw, REASON_TEXT_KO, font_reason,
        box_x0 + 8, box_y0 + 30,
        box_x1 - box_x0 - 16,
        line_spacing=22,
    )

    return canvas


def build_prototype_json(
    cand_meta: Dict[str, Any],
    ref_meta: Dict[str, Any],
    add1_meta: Dict[str, Any],
    add2_meta: Dict[str, Any],
    s3_json_path: pathlib.Path,
    card_png_path: pathlib.Path,
    card_json_path: pathlib.Path,
) -> Dict[str, Any]:
    """v7 prototype JSON 스키마 생성."""
    s3_data: Dict[str, Any] = {}
    if s3_json_path.exists():
        with open(s3_json_path, encoding="utf-8") as f:
            s3_data = json.load(f)

    same_crop_size = (
        cand_meta["crop_shape"] == ref_meta["crop_shape"]
        == add1_meta["crop_shape"] == add2_meta["crop_shape"]
    )
    same_window = (
        cand_meta["hu_win_center"] == ref_meta["hu_win_center"]
        == add1_meta["hu_win_center"] == add2_meta["hu_win_center"]
        and cand_meta["hu_win_width"] == ref_meta["hu_win_width"]
        == add1_meta["hu_win_width"] == add2_meta["hu_win_width"]
    )

    warnings_all: List[str] = []
    for meta in [cand_meta, ref_meta, add1_meta, add2_meta]:
        warnings_all.extend(meta.get("bbox_warnings", []))

    ko_forbidden = scan_forbidden_terms(REASON_TEXT_KO)
    en_forbidden = scan_forbidden_terms(REASON_TEXT_EN)
    diagnostic_guard = (len(ko_forbidden) == 0 and len(en_forbidden) == 0)

    proto_json: Dict[str, Any] = {
        "case_id": TARGET_CASE_ID,
        "prototype_version": PROTOTYPE_VERSION,
        "layout_version": LAYOUT_VERSION,
        "selected_font_family": SELECTED_FONT_FAMILY,
        "selected_font_path": SELECTED_FONT_PATH,
        "source_s3_card_png": str(S3_CARDS_PNG_DIR / f"{TARGET_CASE_ID}.png"),
        "source_s3_card_json": str(s3_json_path),
        "source_s4_reason_csv": str(V3_OUTPUT_CSV),
        "source_s4_reason_json": str(V3_OUTPUT_JSON),
        "candidate_ct_npy": str(CANDIDATE_CT_NPY),
        "normal_ref_ct_npy": str(NORMAL_REF_CT_NPY),
        "additional_ref_ct_npys": [
            str(ADDREF1_CT_NPY),
            str(ADDREF2_CT_NPY),
        ],
        "candidate_slice_z": cand_meta["slice_z"],
        "normal_ref_slice_z": ref_meta["slice_z"],
        "additional_ref_slice_zs": [
            add1_meta["slice_z"],
            add2_meta["slice_z"],
        ],
        "candidate_crop_bbox_yxyx": cand_meta["clamped_bbox_yxyx"],
        "normal_ref_crop_bbox_yxyx": ref_meta["clamped_bbox_yxyx"],
        "additional_ref_crop_bboxes_yxyx": [
            add1_meta["clamped_bbox_yxyx"],
            add2_meta["clamped_bbox_yxyx"],
        ],
        "candidate_crop_shape": cand_meta["crop_shape"],
        "normal_ref_crop_shape": ref_meta["crop_shape"],
        "additional_ref_crop_shapes": [
            add1_meta["crop_shape"],
            add2_meta["crop_shape"],
        ],
        "candidate_display_size": [DISPLAY_SIZE, DISPLAY_SIZE],
        "normal_ref_display_size": [DISPLAY_SIZE, DISPLAY_SIZE],
        "additional_ref_display_sizes": [
            [ADD_DISPLAY_SIZE, ADD_DISPLAY_SIZE],
            [ADD_DISPLAY_SIZE, ADD_DISPLAY_SIZE],
        ],
        "hu_window_center": HU_WIN_CENTER,
        "hu_window_width": HU_WIN_WIDTH,
        "hu_window_low": HU_WIN_LOW,
        "hu_window_high": HU_WIN_HIGH,
        "same_crop_size": same_crop_size,
        "same_window_applied": same_window,
        "normal_reference_source_type": "original_ct_crop",
        "use_32px_ref_png": USE_32PX_REF_PNG,
        "additional_refs_displayed_on_card": ADDITIONAL_REFS_DISPLAYED_ON_CARD,
        "old_panel_c_used": OLD_PANEL_C_USED,
        "full_s3_card_paste_used": (FULL_CARD_PASTE_STRATEGY != "DISABLED"),
        "context_row_added": True,
        "reason_box_added": True,
        "reason_text_ko": REASON_TEXT_KO,
        "reason_text_en": REASON_TEXT_EN,
        "display_language": "ko_only",
        "disclaimer_present": "진단 의미는 아닙니다" in REASON_TEXT_KO,
        "diagnostic_guard_passed": diagnostic_guard,
        "existing_card_modified": False,
        "ct_load_occurred": True,
        "ct_load_mode": "read_only_mmap",
        "full_300_applied": ALLOW_FULL_300,
        "stage2_holdout_accessed": False,
        "output_card_png": str(card_png_path),
        "output_card_json": str(card_json_path),
        "warnings": warnings_all,
        "windowing_note": WINDOWING_NOTE,
    }
    return proto_json


def write_prototype_outputs(
    card_img: Any,
    proto_json: Dict[str, Any],
) -> None:
    """prototype PNG/JSON/index/runtime/DONE/errors 파일 생성."""
    PROTO_CARDS_PNG_DIR.mkdir(parents=True, exist_ok=True)
    PROTO_CARDS_JSON_DIR.mkdir(parents=True, exist_ok=True)

    png_out  = PROTO_CARDS_PNG_DIR  / f"{TARGET_CASE_ID}_reason_prototype.png"
    json_out = PROTO_CARDS_JSON_DIR / f"{TARGET_CASE_ID}_reason_prototype.json"

    card_img.save(str(png_out), format="PNG", dpi=(CANVAS_DPI, CANVAS_DPI))
    print(f"  [SAVED] PNG: {png_out}")

    with open(str(json_out), "w", encoding="utf-8") as f:
        json.dump(proto_json, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] JSON: {json_out}")

    index_row = {
        "case_id": TARGET_CASE_ID,
        "prototype_version": PROTOTYPE_VERSION,
        "layout_version": LAYOUT_VERSION,
        "png_path": str(png_out),
        "json_path": str(json_out),
        "selected_font_family": SELECTED_FONT_FAMILY,
        "same_crop_size": proto_json.get("same_crop_size"),
        "same_window_applied": proto_json.get("same_window_applied"),
        "ct_load_mode": proto_json.get("ct_load_mode"),
        "additional_refs_displayed": ADDITIONAL_REFS_DISPLAYED_ON_CARD,
        "full_300_applied": False,
        "stage2_holdout_accessed": False,
    }
    with open(str(PROTO_INDEX_CSV), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_row.keys()))
        writer.writeheader()
        writer.writerow(index_row)
    print(f"  [SAVED] index: {PROTO_INDEX_CSV}")

    runtime = {
        "script": "build_explanation_card_s4_reason_card_prototype_1case_v7_highres_multi_ref.py",
        "prototype_version": PROTOTYPE_VERSION,
        "case_id": TARGET_CASE_ID,
        "n_generated": 1,
        "n_errors": 0,
        "selected_font_family": SELECTED_FONT_FAMILY,
        "ct_load_mode": "read_only_mmap",
        "n_ct_loaded": 4,
        "full_300_applied": False,
        "stage2_holdout_accessed": False,
        "existing_card_modified": False,
    }
    with open(str(PROTO_RUNTIME_JSON), "w", encoding="utf-8") as f:
        json.dump(runtime, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] runtime: {PROTO_RUNTIME_JSON}")

    with open(str(PROTO_ERRORS_CSV), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "error", "detail"])
        writer.writeheader()
    print(f"  [SAVED] errors: {PROTO_ERRORS_CSV}")

    done = {
        "prototype_version": PROTOTYPE_VERSION,
        "case_id": TARGET_CASE_ID,
        "n_generated": 1,
        "n_errors": 0,
        "full_300_applied": False,
        "stage2_holdout_accessed": False,
        "existing_card_modified": False,
        "selected_font_family": SELECTED_FONT_FAMILY,
    }
    with open(str(PROTO_DONE_JSON), "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] DONE: {PROTO_DONE_JSON}")


# ============================================================
# prototype 생성 (run 단계 — 현재 차단)
# ============================================================
def run_prototype() -> None:
    """실제 v7 카드 생성 (ALLOW_RUN_CARD_PROTOTYPE + ALLOW_CT_LOAD 둘 다 True 필요)."""
    if not ALLOW_RUN_CARD_PROTOTYPE:
        _block("run_prototype: ALLOW_RUN_CARD_PROTOTYPE=False — 사용자 승인 후 변경 필요")
    if not ALLOW_CT_LOAD:
        _block("run_prototype: ALLOW_CT_LOAD=False — CT load 별도 승인 필요 (read-only mmap만 허용)")
    if ALLOW_ORIGINAL_CARD_MODIFICATION:
        _block("run_prototype: ALLOW_ORIGINAL_CARD_MODIFICATION=True — 항구 False여야 함")
    if ALLOW_FULL_300:
        _block("run_prototype: ALLOW_FULL_300=True — 항구 False여야 함")

    # 한글 폰트 guard
    font_res = validate_font()
    if not font_res["ok"]:
        _block(f"run_prototype: 한글 폰트 오류 — {font_res['issues']}")

    # output guard
    guard = check_output_guard()
    if not guard["ok"]:
        _block(f"run_prototype: output guard FAIL — {guard.get('error')}")

    print(f"[run_prototype] v7 prototype 생성 시작: {TARGET_CASE_ID}")
    print(f"  [FONT] 사용 폰트: {SELECTED_FONT_FAMILY} ({SELECTED_FONT_PATH})")

    # CT 로드 (read-only mmap — 4건)
    print("  [CT LOAD] candidate CT ...")
    cand_ct = load_ct_mmap(CANDIDATE_CT_NPY)
    print(f"  [CT LOAD] candidate shape={cand_ct.shape}")

    print("  [CT LOAD] matched normal reference CT ...")
    ref_ct = load_ct_mmap(NORMAL_REF_CT_NPY)
    print(f"  [CT LOAD] normal ref shape={ref_ct.shape}")

    print("  [CT LOAD] additional normal ref 1 CT ...")
    add1_ct = load_ct_mmap(ADDREF1_CT_NPY)
    print(f"  [CT LOAD] addref1 shape={add1_ct.shape}")

    print("  [CT LOAD] additional normal ref 2 CT ...")
    add2_ct = load_ct_mmap(ADDREF2_CT_NPY)
    print(f"  [CT LOAD] addref2 shape={add2_ct.shape}")

    # crop 추출
    cand_crop, cand_meta = extract_ct_crop(
        cand_ct, CANDIDATE_SLICE_Z,
        CANDIDATE_CROP_Y0, CANDIDATE_CROP_X0,
        CANDIDATE_CROP_Y1, CANDIDATE_CROP_X1,
        label="candidate",
    )
    ref_crop, ref_meta = extract_ct_crop(
        ref_ct, NORMAL_REF_SLICE_Z,
        NORMAL_REF_CROP_Y0, NORMAL_REF_CROP_X0,
        NORMAL_REF_CROP_Y1, NORMAL_REF_CROP_X1,
        label="normal_ref",
    )
    add1_crop, add1_meta = extract_ct_crop(
        add1_ct, ADDREF1_SLICE_Z,
        ADDREF1_CROP_Y0, ADDREF1_CROP_X0,
        ADDREF1_CROP_Y1, ADDREF1_CROP_X1,
        label="addref1",
    )
    add2_crop, add2_meta = extract_ct_crop(
        add2_ct, ADDREF2_SLICE_Z,
        ADDREF2_CROP_Y0, ADDREF2_CROP_X0,
        ADDREF2_CROP_Y1, ADDREF2_CROP_X1,
        label="addref2",
    )

    for meta in [cand_meta, ref_meta, add1_meta, add2_meta]:
        if meta.get("bbox_warnings"):
            print(f"  [WARN] {meta['label']} bbox: {meta['bbox_warnings']}")

    # HU windowing (동일 파라미터 전체 적용)
    cand_u8  = window_hu(cand_crop)
    ref_u8   = window_hu(ref_crop)
    add1_u8  = window_hu(add1_crop)
    add2_u8  = window_hu(add2_crop)
    print(f"  [WINDOW] candidate u8: shape={cand_u8.shape}, range=[{cand_u8.min()},{cand_u8.max()}]")
    print(f"  [WINDOW] normal ref u8: shape={ref_u8.shape}, range=[{ref_u8.min()},{ref_u8.max()}]")
    print(f"  [WINDOW] addref1 u8: shape={add1_u8.shape}, range=[{add1_u8.min()},{add1_u8.max()}]")
    print(f"  [WINDOW] addref2 u8: shape={add2_u8.shape}, range=[{add2_u8.min()},{add2_u8.max()}]")

    # S3 PNG 경로 (read-only)
    s3_png_path  = S3_CARDS_PNG_DIR  / f"{TARGET_CASE_ID}.png"
    s3_json_path = S3_CARDS_JSON_DIR / f"{TARGET_CASE_ID}.json"

    # v7 canvas 생성
    print("  [CANVAS] v7 canvas 생성 ...")
    card_img = build_v7_card_image(cand_u8, ref_u8, add1_u8, add2_u8, s3_png_path)
    print(f"  [CANVAS] 완료: {card_img.size}")

    # prototype JSON
    png_out  = PROTO_CARDS_PNG_DIR  / f"{TARGET_CASE_ID}_reason_prototype.png"
    json_out = PROTO_CARDS_JSON_DIR / f"{TARGET_CASE_ID}_reason_prototype.json"
    proto_json = build_prototype_json(
        cand_meta, ref_meta, add1_meta, add2_meta,
        s3_json_path, png_out, json_out,
    )

    if not proto_json["diagnostic_guard_passed"]:
        _block("run_prototype: diagnostic_guard_passed=False")

    write_prototype_outputs(card_img, proto_json)
    print(f"[run_prototype] 완료: {TARGET_CASE_ID}")


# ============================================================
# selftest (24항목)
# ============================================================
def run_selftest() -> bool:
    """v7 selftest — guard, font, 설계 검사."""
    import inspect

    results: List[Dict[str, Any]] = []
    all_pass = True

    def _check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal all_pass
        if not passed:
            all_pass = False
        status = "PASS" if passed else "FAIL"
        results.append({"name": name, "status": status, "detail": detail})
        mark = "[PASS]" if passed else "[FAIL]"
        print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))

    print("=== selftest (v7 high-res multi-reference) ===")

    # 01. guard 기본값
    _check("01. ALLOW_RUN_CARD_PROTOTYPE=False",
           not ALLOW_RUN_CARD_PROTOTYPE,
           f"value={ALLOW_RUN_CARD_PROTOTYPE}")
    _check("02. ALLOW_CT_LOAD=False",
           not ALLOW_CT_LOAD,
           f"value={ALLOW_CT_LOAD}")
    _check("03. ALLOW_ORIGINAL_CARD_MODIFICATION=False",
           not ALLOW_ORIGINAL_CARD_MODIFICATION,
           f"value={ALLOW_ORIGINAL_CARD_MODIFICATION}")
    _check("04. ALLOW_FULL_300=False",
           not ALLOW_FULL_300,
           f"value={ALLOW_FULL_300}")

    # 05. Malgun font path exists
    font_res = validate_font()
    _check("05. 한글 폰트 path exists (Malgun 우선)",
           font_res["ok"],
           font_res.get("path", "NOT_FOUND") or "NOT_FOUND")

    # 06. DejaVu fallback 금지
    no_dejavu = (
        SELECTED_FONT_PATH is not None
        and "dejavu" not in SELECTED_FONT_FAMILY.lower()
    )
    _check("06. DejaVu fallback 금지",
           no_dejavu,
           f"family={SELECTED_FONT_FAMILY}")

    # 07. selected_font_family JSON field 준비
    _check("07. SELECTED_FONT_FAMILY 상수 존재",
           isinstance(SELECTED_FONT_FAMILY, str) and len(SELECTED_FONT_FAMILY) > 0,
           f"SELECTED_FONT_FAMILY={SELECTED_FONT_FAMILY}")

    # 08. no old Panel C
    _check("08. OLD_PANEL_C_USED=False",
           not OLD_PANEL_C_USED,
           f"OLD_PANEL_C_USED={OLD_PANEL_C_USED}")

    # 09. no full S3 card paste
    _check("09. FULL_CARD_PASTE_STRATEGY=DISABLED",
           FULL_CARD_PASTE_STRATEGY == "DISABLED",
           f"value={FULL_CARD_PASTE_STRATEGY}")

    # 10. use_32px_ref_png=false
    _check("10. USE_32PX_REF_PNG=False",
           not USE_32PX_REF_PNG,
           f"value={USE_32PX_REF_PNG}")

    # 11. candidate crop size 96×96
    crop_res = validate_crop_sizes()
    cand_detail = crop_res["details"].get("candidate", {})
    _check("11. candidate crop size=96×96",
           cand_detail.get("ok", False),
           f"h={cand_detail.get('h')}×w={cand_detail.get('w')}")

    # 12. matched normal crop size 96×96
    nref_detail = crop_res["details"].get("normal_ref", {})
    _check("12. matched normal crop size=96×96",
           nref_detail.get("ok", False),
           f"h={nref_detail.get('h')}×w={nref_detail.get('w')}")

    # 13. additional refs original_ct_crop source
    _check("13. additional refs source=original_ct_crop",
           USE_ORIGINAL_CT_FOR_REF and ADDITIONAL_REFS_DISPLAYED_ON_CARD,
           f"USE_ORIGINAL_CT_FOR_REF={USE_ORIGINAL_CT_FOR_REF}, displayed={ADDITIONAL_REFS_DISPLAYED_ON_CARD}")

    # 14. candidate/normal display size 320×320
    size_res = validate_display_sizes()
    _check("14. candidate/normal display size=320px",
           size_res["ok"] and DISPLAY_SIZE == 320,
           f"DISPLAY_SIZE={DISPLAY_SIZE}")

    # 15. additional display size >=160
    _check("15. additional display size>=160px",
           ADD_DISPLAY_SIZE >= 160,
           f"ADD_DISPLAY_SIZE={ADD_DISPLAY_SIZE}")

    # 16. same HU window (HU_WIN_CENTER/WIDTH 상수 동일)
    _check("16. same HU window 상수",
           HU_WIN_CENTER == -600 and HU_WIN_WIDTH == 1500,
           f"center={HU_WIN_CENTER}, width={HU_WIN_WIDTH}")

    # 17. reason text '진단 의미는 아닙니다'
    text_res = validate_reason_text()
    _check("17. reason text '진단 의미는 아닙니다'",
           text_res["has_disclaimer"],
           "포함됨" if text_res["has_disclaimer"] else "없음")

    # 18. reason text '시각적 참고 단서'
    _check("18. reason text '시각적 참고 단서'",
           text_res["has_visual_cue"],
           "포함됨" if text_res["has_visual_cue"] else "없음")

    # 19. reason text '약 +245 HU'
    _check("19. reason text '약 +245 HU'",
           text_res.get("has_245hu", False),
           "포함됨" if text_res.get("has_245hu") else "없음")

    # 20. forbidden term hit 0
    _check("20. 금지어 hit 0",
           len(text_res["ko_forbidden"]) == 0 and len(text_res["en_forbidden"]) == 0,
           f"KO={text_res['ko_forbidden']}, EN={text_res['en_forbidden']}")

    # 21. np.load는 ALLOW_CT_LOAD guard 아래에서만
    load_src = inspect.getsource(load_ct_mmap)
    _check("21. np.load는 ALLOW_CT_LOAD guard 아래에서만",
           "ALLOW_CT_LOAD" in load_src and "np.load" in load_src,
           "guard 확인됨")

    # 22. np.load에 mmap_mode='r' 전용
    has_mmap_r = "mmap_mode=\"r\"" in load_src or "mmap_mode='r'" in load_src
    _check("22. np.load mmap_mode='r' 전용",
           has_mmap_r,
           "mmap_mode='r' 확인됨" if has_mmap_r else "mmap_mode 없음")

    # 23. full 300 loop 없음 (run_prototype 소스)
    run_src = inspect.getsource(run_prototype)
    full300_loop = any(
        "ALLOW_FULL_300" in ln and "for " in ln
        for ln in run_src.split("\n")
    )
    _check("23. full 300 loop 없음",
           not full300_loop,
           "ALLOW_FULL_300 guard만 있고 loop 없음" if not full300_loop else "full 300 loop 발견")

    # 24. stage2_holdout intersection 0
    s2_in_output = any(tok in str(PROTO_OUTPUT_ROOT) for tok in STAGE2_HOLDOUT_TOKENS)
    design_src = inspect.getsource(build_v7_canvas_design)
    s2_in_design = any(tok in design_src for tok in STAGE2_HOLDOUT_TOKENS)
    _check("24. stage2_holdout intersection 0",
           not s2_in_output and not s2_in_design,
           "stage2_holdout 없음" if (not s2_in_output and not s2_in_design) else "stage2_holdout 토큰 발견")

    # 추가 항목
    _check("K1. output root v1~v6와 분리",
           PROTO_OUTPUT_ROOT.name == "s4_reason_card_prototype_1case_v7_highres_multi_ref",
           PROTO_OUTPUT_ROOT.name)

    guard_res = check_output_guard()
    _check("K2. output guard (DONE.json 없음 + 경로 충돌 없음)",
           guard_res["ok"],
           guard_res.get("error", f"safe: {PROTO_OUTPUT_ROOT.name}"))

    _check("K3. candidate CT npy path exists",
           CANDIDATE_CT_NPY.exists(),
           str(CANDIDATE_CT_NPY))

    _check("K4. matched normal CT npy path exists",
           NORMAL_REF_CT_NPY.exists(),
           str(NORMAL_REF_CT_NPY))

    _check("K5. addref1 CT npy path exists",
           ADDREF1_CT_NPY.exists(),
           str(ADDREF1_CT_NPY))

    _check("K6. addref2 CT npy path exists",
           ADDREF2_CT_NPY.exists(),
           str(ADDREF2_CT_NPY))

    _check("K7. addref1 crop size 96×96",
           crop_res["details"].get("addref1", {}).get("ok", False),
           str(crop_res["details"].get("addref1", {})))

    _check("K8. addref2 crop size 96×96",
           crop_res["details"].get("addref2", {}).get("ok", False),
           str(crop_res["details"].get("addref2", {})))

    _check("K9. v1~v6 prototype 보존 확인",
           all([
               PROTO_V1_OUTPUT_ROOT.exists(),
               PROTO_V2_OUTPUT_ROOT.exists(),
               PROTO_V3_OUTPUT_ROOT.exists(),
               PROTO_V4_OUTPUT_ROOT.exists(),
               PROTO_V5_OUTPUT_ROOT.exists(),
               PROTO_V6_OUTPUT_ROOT.exists(),
           ]),
           "v1~v6 존재")

    _check("K10. JSON 필수 필드 selected_font_family 준비",
           "selected_font_family" in inspect.getsource(build_prototype_json),
           "필드 준비됨")

    _check("K11. JSON 필수 필드 additional_ref_ct_npys 준비",
           "additional_ref_ct_npys" in inspect.getsource(build_prototype_json),
           "필드 준비됨")

    _check("K12. run_prototype에 NotImplementedError 없음",
           "NotImplementedError" not in run_src,
           "NotImplementedError 없음" if "NotImplementedError" not in run_src else "NotImplementedError 발견")

    _check("K13. clamp_bbox 함수 존재",
           callable(globals().get("clamp_bbox")),
           "존재" if callable(globals().get("clamp_bbox")) else "없음")

    _check("K14. window_hu 함수가 HU_WIN_CENTER/HU_WIN_WIDTH 기본값 사용",
           "HU_WIN_CENTER" in inspect.getsource(window_hu),
           "HU_WIN_CENTER 참조 확인됨")

    _check("K15. _draw_wrapped_text_ko 함수 존재",
           callable(globals().get("_draw_wrapped_text_ko")),
           "존재" if callable(globals().get("_draw_wrapped_text_ko")) else "없음")

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n=== selftest 결과: PASS={n_pass}/{len(results)} FAIL={n_fail}/{len(results)} ===")

    if n_fail == 0:
        print("[PASS] selftest 전체 통과")
    else:
        print("[FAIL] selftest 일부 실패 — 위 FAIL 항목 확인 필요")

    return n_fail == 0


# ============================================================
# dry-run
# ============================================================
def run_dry_run() -> bool:
    """입력 파일 존재 + output guard + font + resolve 확인 (CT load 없음)."""
    print("=== dry-run (v7 high-res multi-reference) ===")
    issues: List[str] = []

    if ALLOW_RUN_CARD_PROTOTYPE:
        issues.append("ALLOW_RUN_CARD_PROTOTYPE=True — 이 단계에서는 False여야 함")
    if ALLOW_CT_LOAD:
        issues.append("ALLOW_CT_LOAD=True — 이 단계에서는 금지")

    # font
    font_res = validate_font()
    if font_res["ok"]:
        print(f"  [OK] 한글 폰트: {font_res['family']} ({font_res['path']})")
    else:
        for i in font_res["issues"]:
            issues.append(f"font: {i}")
            print(f"  [WARN] font 이슈: {i}")

    # S4 v3 CSV/JSON
    for path, name in [(V3_OUTPUT_CSV, "V3 CSV"), (V3_OUTPUT_JSON, "V3 JSON")]:
        if path.exists():
            print(f"  [OK] {name}: {path.name}")
        else:
            issues.append(f"{name} 없음: {path}")

    # S3 source
    s3_png  = S3_CARDS_PNG_DIR  / f"{TARGET_CASE_ID}.png"
    s3_json = S3_CARDS_JSON_DIR / f"{TARGET_CASE_ID}.json"
    for path, name in [(s3_png, "S3 PNG"), (s3_json, "S3 JSON")]:
        if path.exists():
            print(f"  [OK] {name}: {path.name}")
        else:
            issues.append(f"{name} 없음: {path}")

    # reference manifest
    if REF_CROP_MANIFEST.exists():
        print(f"  [OK] REF_CROP_MANIFEST: {REF_CROP_MANIFEST.name}")
    else:
        issues.append(f"REF_CROP_MANIFEST 없음: {REF_CROP_MANIFEST}")

    # CT npy paths (4건)
    for ct_path, label in [
        (CANDIDATE_CT_NPY,  "candidate CT"),
        (NORMAL_REF_CT_NPY, "matched normal ref CT"),
        (ADDREF1_CT_NPY,    "addref1 CT"),
        (ADDREF2_CT_NPY,    "addref2 CT"),
    ]:
        if ct_path.exists():
            print(f"  [OK] {label}: {ct_path.parent.name}/ct_hu.npy")
        else:
            issues.append(f"{label} npy 없음: {ct_path}")
            print(f"  [WARN] {label} 없음: {ct_path}")

    # patient manifests
    for m_path, name in [
        (NORMAL_PATIENT_MANIFEST, "normal patient manifest"),
        (NSCLC_PATIENT_MANIFEST,  "NSCLC patient manifest"),
    ]:
        if m_path.exists():
            print(f"  [OK] {name}: {m_path.name}")
        else:
            issues.append(f"{name} 없음: {m_path}")

    # crop size 확인
    crop_res = validate_crop_sizes()
    if crop_res["ok"]:
        print("  [OK] crop size: 전체 96×96 동일")
    else:
        for i in crop_res["issues"]:
            issues.append(i)

    # display size 확인
    size_res = validate_display_sizes()
    if size_res["ok"]:
        print(f"  [OK] display size: main={DISPLAY_SIZE}px, add={ADD_DISPLAY_SIZE}px, ctx={CONTEXT_DISPLAY_SIZE}px")
    else:
        for i in size_res["issues"]:
            issues.append(i)

    # output guard
    guard = check_output_guard()
    if guard["ok"]:
        print(f"  [OK] output guard: v7 output root 충돌 없음 ({PROTO_OUTPUT_ROOT.name})")
    else:
        issues.append(f"output guard FAIL: {guard.get('error')}")

    # reason text
    text_res = validate_reason_text()
    if text_res["ok"]:
        print(f"  [OK] reason text: {text_res['ko_len']}자, 면책문구+시각적참고단서+245HU 포함, 금지어 없음")
    else:
        for i in text_res["issues"]:
            issues.append(i)

    print(f"  [NOTE] windowing: center={HU_WIN_CENTER}HU width={HU_WIN_WIDTH}HU")
    print(f"  [NOTE] additional refs source=original_ct_crop, displayed={ADDITIONAL_REFS_DISPLAYED_ON_CARD}")

    if issues:
        print(f"\n[FAIL] dry-run 이슈 {len(issues)}건:")
        for i in issues:
            print(f"  - {i}")
        return False
    else:
        print("\n[PASS] dry-run 전체 통과")
        return True


# ============================================================
# plan-only
# ============================================================
def run_plan_only() -> None:
    """dry-run + v7 배치 계획 출력."""
    dry_ok = run_dry_run()

    print("\n=== plan-only: v7 high-res multi-reference layout 계획 ===")
    print(f"  대상 case      : {TARGET_CASE_ID}")
    print(f"  prototype 버전 : {PROTOTYPE_VERSION}")
    print(f"  layout 버전    : {LAYOUT_VERSION}")
    print(f"  v6 실패 원인   : 한글 □ 깨짐 / normal reference 1개 / reason text 미흡")
    print(f"  v7 핵심 변경   : Malgun font fix / additional normals 2개 / text 강화")
    print()
    print(f"  [Font] {SELECTED_FONT_FAMILY} ({SELECTED_FONT_PATH})")
    print()
    print(f"  [Row 1] Main comparison (candidate 320px vs matched normal 320px)")
    print(f"    좌: Candidate {TARGET_SAFE_ID}")
    print(f"      z={CANDIDATE_SLICE_Z}, crop [{CANDIDATE_CROP_Y0},{CANDIDATE_CROP_X0},"
          f"{CANDIDATE_CROP_Y1},{CANDIDATE_CROP_X1}] = {CANDIDATE_CROP_SIZE_H}×{CANDIDATE_CROP_SIZE_W}px")
    print(f"      display: {DISPLAY_SIZE}×{DISPLAY_SIZE}px (LANCZOS)")
    print(f"    우: Matched normal {NORMAL_REF_SAFE_ID[:40]}...")
    print(f"      z={NORMAL_REF_SLICE_Z}, crop [{NORMAL_REF_CROP_Y0},{NORMAL_REF_CROP_X0},"
          f"{NORMAL_REF_CROP_Y1},{NORMAL_REF_CROP_X1}] = {NORMAL_REF_CROP_SIZE_H}×{NORMAL_REF_CROP_SIZE_W}px")
    print(f"      display: {DISPLAY_SIZE}×{DISPLAY_SIZE}px")
    print()
    print(f"  [Row 2] Additional examples (180px each) + context (190px)")
    print(f"    좌: Addref1 {ADDREF1_SAFE_ID[:35]}...")
    print(f"      z={ADDREF1_SLICE_Z}, crop [{ADDREF1_CROP_Y0},{ADDREF1_CROP_X0},"
          f"{ADDREF1_CROP_Y1},{ADDREF1_CROP_X1}] → display {ADD_DISPLAY_SIZE}px")
    print(f"    중: Addref2 {ADDREF2_SAFE_ID[:35]}...")
    print(f"      z={ADDREF2_SLICE_Z}, crop [{ADDREF2_CROP_Y0},{ADDREF2_CROP_X0},"
          f"{ADDREF2_CROP_Y1},{ADDREF2_CROP_X1}] → display {ADD_DISPLAY_SIZE}px")
    print(f"    우: A whole-slice context {CONTEXT_DISPLAY_SIZE}px — S3 PNG crop")
    print()
    print(f"  [Row 3] Reason box (KO text, 전체 폭, 한글 font)")
    print(f"    Title: {REASON_TITLE}")
    print(f"    KO ({len(REASON_TEXT_KO)}자): {REASON_TEXT_KO[:80]}...")
    print()
    print(f"  canvas 추정: {CANVAS_WIDTH_PX}×{CANVAS_HEIGHT_ESTIMATE_PX}px")
    print(f"  Output root: {PROTO_OUTPUT_ROOT}")
    print(f"  Output PNG : {PROTO_CARDS_PNG_DIR / f'{TARGET_CASE_ID}_reason_prototype.png'}")
    print(f"  Output JSON: {PROTO_CARDS_JSON_DIR / f'{TARGET_CASE_ID}_reason_prototype.json'}")
    print()
    print(f"  windowing: center={HU_WIN_CENTER}HU width={HU_WIN_WIDTH}HU")
    print(f"             applied to: candidate / normal_ref / addref1 / addref2")
    print(f"  CT load: read-only mmap (np.load mmap_mode='r') — run 단계 승인 후")
    print(f"  CT load 건수: 4건 (candidate + matched + add1 + add2)")
    print(f"  stage2_holdout 접근: 0")
    print(f"  full 300: false")
    print()

    if not dry_ok:
        print("[WARN] dry-run 이슈 있음 — plan-only 결과 참고용")
    else:
        print("[PASS] plan-only 완료")


# ============================================================
# main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="S4 Reason Card Prototype 1-case v7 — high-res multi-reference + font fix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--selftest",
                        action="store_true",
                        help="selftest 실행 (guard/font/설계 검사, 생성 없음)")
    parser.add_argument("--dry-run",
                        action="store_true",
                        dest="dry_run",
                        help="입력 파일 존재 + output guard + font 확인 (생성 없음)")
    parser.add_argument("--plan-only",
                        action="store_true",
                        dest="plan_only",
                        help="dry-run + v7 배치 계획 출력 (생성 없음)")
    parser.add_argument("--run-prototype",
                        action="store_true",
                        dest="run_prototype",
                        help="실제 v7 prototype 생성 (ALLOW_RUN_CARD_PROTOTYPE + ALLOW_CT_LOAD=True 필요)")
    parser.add_argument("--confirm-generate",
                        action="store_true",
                        dest="confirm_generate",
                        help="--run-prototype 와 함께 사용 시 실제 생성 확인")

    args = parser.parse_args()

    # bare 실행 차단
    if not any([args.selftest, args.dry_run, args.plan_only, args.run_prototype]):
        _block(
            "bare 실행 차단: --selftest / --dry-run / --plan-only / --run-prototype 중 하나를 지정하세요."
        )

    # --run-prototype 단독 차단
    if args.run_prototype and not args.confirm_generate:
        _block("--run-prototype 단독 실행 차단: --confirm-generate 를 함께 지정하세요.")

    # --run-prototype --confirm-generate: guard 체크
    if args.run_prototype and args.confirm_generate:
        if not ALLOW_RUN_CARD_PROTOTYPE:
            _block(
                "--run-prototype --confirm-generate: ALLOW_RUN_CARD_PROTOTYPE=False — "
                "사용자 승인 후 스크립트 내 ALLOW_RUN_CARD_PROTOTYPE=True 로 변경 후 재실행"
            )
        run_prototype()
        return

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)

    if args.plan_only:
        run_plan_only()
        sys.exit(0)

    if args.dry_run:
        ok = run_dry_run()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
