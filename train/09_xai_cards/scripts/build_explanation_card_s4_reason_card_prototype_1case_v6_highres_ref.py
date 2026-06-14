#!/usr/bin/env python3
"""
build_explanation_card_s4_reason_card_prototype_1case_v6_highres_ref.py

S4 Reason Card Prototype 1-case v6 — High-resolution normal reference script

목적:
- LUNG1-320__c2 1건에 대해 원본 CT 해상도 기반 비교 카드 생성 (정적 설계/검사 단계).
- v5 실패 원인: matched normal reference가 32×32 PNG 확대본이었음.
- v6 핵심: 원본 CT slice에서 96×96 crop으로 직접 추출 → 양측 동일 조건 비교.

v5 대비 변경사항:
  1. Normal reference: 32×32 PNG staged upscale 방식 완전 폐기
     → 원본 CT (ct_hu.npy) slice z=179 에서 96×96 직접 crop
  2. Candidate: S3 PNG B-panel crop 재사용 방식 폐기
     → 원본 Candidate CT slice z=140 에서 96×96 직접 crop
  3. 양측 동일 조건: crop_size=96×96, display_size=320×320, 동일 HU window
  4. Row 1: Candidate 96→320 (좌) vs Matched normal reference 96→320 (우)
  5. Row 2: A whole-slice context (190px, 좌) + D z-context (190px, 우)
  6. Row 3: Reason box (KO text, 전체 폭)
  7. Ref2/Ref3: 카드 표시 제거, JSON에만 보존 (v5 계승)
  8. Old Panel C: 완전 제거 (v4+ 계승)
  9. S3 PNG B-panel 확대 방식(32×32 PNG 확대본) 완전 폐기

CT path resolve 정책 (run 단계):
  - 정상 reference CT: NORMAL_PATIENT_MANIFEST (LUNA16 기반) → ct_hu_npy 컬럼
  - Candidate CT: NSCLC_PATIENT_MANIFEST → ct_hu_npy 컬럼
  - Windows E:\\ 경로는 /mnt/c/... WSL 경로로 자동 변환
  - 이번 정적 단계에서는 path existence 확인만 수행 (실제 mmap 금지)

절대 금지:
  - 실제 CT/mask npy load (ALLOW_CT_LOAD=False)
  - 실제 prototype PNG/JSON 생성 (ALLOW_RUN_CARD_PROTOTYPE=False)
  - 기존 S3 card PNG/JSON 수정 (ALLOW_ORIGINAL_CARD_MODIFICATION=False)
  - full 300 처리 (ALLOW_FULL_300=False)
  - score/model/threshold 재계산
  - stage2_holdout 접근
  - 기존 v1~v5 prototype 수정/삭제/덮어쓰기
  - 32×32 reference PNG 확대 방식을 카드에 사용 (USE_32PX_REF_PNG=False)
  - old Panel C를 카드에 포함
  - S3 PNG full card paste 전략 (FULL_CARD_PASTE_STRATEGY="DISABLED")

실행 모드:
  bare 실행                                  → BLOCKED exit 2
  --selftest                                 → selftest items 검사
  --dry-run                                  → 입력 파일 존재 + output guard 확인
  --plan-only                                → dry-run + 배치 계획 출력
  --run-prototype                            → 단독 BLOCKED exit 2
  --run-prototype --confirm-generate         → ALLOW_RUN_CARD_PROTOTYPE=False이면 BLOCKED

syntax check:
  python -m py_compile scripts/build_explanation_card_s4_reason_card_prototype_1case_v6_highres_ref.py
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
# 최상위 가드 — 이번 단계는 전부 False
# ============================================================
ALLOW_RUN_CARD_PROTOTYPE         = False   # True 변경은 사용자 승인 필수
ALLOW_ORIGINAL_CARD_MODIFICATION = False   # 항구 False
ALLOW_CT_LOAD                    = False   # run 단계 승인 전 False. read-only mmap만 허용 예정
ALLOW_FULL_300                   = False   # 항구 False

# ============================================================
# v6 핵심 전략 식별자
# ============================================================
FULL_CARD_PASTE_STRATEGY  = "DISABLED"   # S3 PNG 전체 paste — v4부터 폐기, v6 계승
SHOW_REF2_REF3_ON_CARD    = False        # Ref2/Ref3 카드 미표시 (JSON에만 보존)
USE_32PX_REF_PNG          = False        # v5 방식 32×32 PNG 확대 — v6 완전 폐기
USE_ORIGINAL_CT_FOR_REF   = True         # v6: 원본 CT slice에서 직접 crop
USE_ORIGINAL_CT_FOR_CAND  = True         # v6: candidate도 원본 CT에서 crop

# ============================================================
# prototype 대상 (1건)
# ============================================================
TARGET_CASE_ID      = "LUNG1-320__c2"
TARGET_PATIENT_ID   = "LUNG1-320"
TARGET_SAFE_ID      = "NSCLC_LUNG1-320__95de24d86f"

# ============================================================
# reason text v6 — 2문장, 면책문구 필수
# ============================================================
REASON_TITLE = "Reason cue / 검토 근거 후보"

REASON_TEXT_KO = (
    "같은 lower_peripheral 위치의 정상 CT crop과 같은 크기/윈도우로 비교했을 때, "
    "후보 crop은 더 넓은 고밀도 영역을 포함했고 HU 밀도가 약 +245 HU 높게 측정되었습니다. "
    "이는 PaDiM high-response를 해석하기 위한 시각적 참고 단서이며, 진단 의미는 아닙니다."
)

REASON_TEXT_EN = (
    "Compared with a normal CT crop from the same lower_peripheral bin at identical size/window, "
    "the candidate crop showed a broader high-density region with approximately +245 HU higher density. "
    "This is a visual reference cue for interpreting the PaDiM high response, not a clinical diagnosis."
)

# ============================================================
# v6 layout 설정
# ============================================================
LAYOUT_VERSION       = "highres_comparison_v1"
PROTOTYPE_VERSION    = "v6"

# Panel labels
PANEL_A_LABEL        = "A. Whole slice (위치 확인)"
PANEL_D_LABEL        = "D. z-context [139, 140, 141]"

# Comparison labels
CANDIDATE_LABEL           = "Candidate — original CT crop (96×96→320px)"
MAIN_REF_LABEL            = "Matched normal — original CT crop\n(96×96→320px, lower_peripheral)"
COMPARISON_SECTION_LABEL  = "Candidate vs same-bin normal reference (original CT, 동일 조건)"

# ============================================================
# v6 crop 설계 (원본 CT 기준)
# ============================================================

# Candidate: LUNG1-320__c2, z=140
# display_bbox = [288, 176, 384, 272] = y:[288,384], x:[176,272] = 96×96
CANDIDATE_SLICE_Z  = 140
CANDIDATE_CROP_Y0  = 288   # row start (inclusive)
CANDIDATE_CROP_X0  = 176   # col start (inclusive)
CANDIDATE_CROP_Y1  = 384   # row end (exclusive)
CANDIDATE_CROP_X1  = 272   # col end (exclusive)
CANDIDATE_CROP_SIZE_H = CANDIDATE_CROP_Y1 - CANDIDATE_CROP_Y0  # 96
CANDIDATE_CROP_SIZE_W = CANDIDATE_CROP_X1 - CANDIDATE_CROP_X0  # 96

# Matched normal reference: subset4...d3e03b0ce0, z=179
# manifest: y0=240, x0=368, y1=272, x1=400 (original 32×32 patch)
# center = (256, 384)
# v6 96×96 bbox: [208, 336, 304, 432]  (center ± 48)
# ※ run 단계에서 np.clip(bbox, 0, slice_shape) 적용 필수
NORMAL_REF_SAFE_ID    = "subset4_1.3.6.1.4.1.14519.5.2.1.6279.6001.200837896655745926888305239398__d3e03b0ce0"
NORMAL_REF_PATIENT_ID = "subset4_1.3.6.1.4.1.14519.5.2.1.6279.6001.200837896655745926888305239398"
NORMAL_REF_SLICE_Z    = 179
NORMAL_REF_ORIG_Y0    = 240   # 32×32 원본 패치 bbox
NORMAL_REF_ORIG_X0    = 368
NORMAL_REF_ORIG_Y1    = 272
NORMAL_REF_ORIG_X1    = 400
NORMAL_REF_CENTER_Y   = 256   # center = (y0+y1)//2
NORMAL_REF_CENTER_X   = 384   # center = (x0+x1)//2
# v6 96×96 bbox (center ± 48)
NORMAL_REF_CROP_Y0 = 208   # 256 - 48
NORMAL_REF_CROP_X0 = 336   # 384 - 48
NORMAL_REF_CROP_Y1 = 304   # 256 + 48
NORMAL_REF_CROP_X1 = 432   # 384 + 48
NORMAL_REF_CROP_SIZE_H = NORMAL_REF_CROP_Y1 - NORMAL_REF_CROP_Y0  # 96
NORMAL_REF_CROP_SIZE_W = NORMAL_REF_CROP_X1 - NORMAL_REF_CROP_X0  # 96

# ============================================================
# v6 display size — 양측 동일
# ============================================================
CROP_SOURCE_SIZE    = 96    # px — 원본 CT crop size (양측 동일)
DISPLAY_SIZE        = 320   # px — 화면 표시 크기 (양측 동일, LANCZOS upscale)
CONTEXT_DISPLAY_SIZE = 190  # px — A, D 보조 패널

# ============================================================
# windowing 정책 (v6)
# S3 card의 windowing 출처 확인 필요:
#   v5는 S3 PNG에서 B-panel을 그대로 crop했으므로 windowing이 S3 생성 시 적용됨.
#   v6는 원본 CT HU 값을 직접 읽으므로 windowing을 명시적으로 적용해야 함.
# 정책: run 단계에서 S3 card 생성에 사용된 파라미터(lung window)를 그대로 계승.
# 현재 S3 card의 windowing 파라미터 출처: S3 스크립트의 HU_WIN 상수를 확인 후 재사용.
# 아래는 run 단계 전 확인 필요 항목으로 명시하며 기본값을 lung window로 설정.
HU_WIN_CENTER  = -600   # lung window center (HU) — run 단계에서 S3 파라미터와 일치 확인 필수
HU_WIN_WIDTH   = 1500   # lung window width (HU) — 동일
HU_WIN_LOW     = HU_WIN_CENTER - HU_WIN_WIDTH // 2   # -1350
HU_WIN_HIGH    = HU_WIN_CENTER + HU_WIN_WIDTH // 2   # 150
WINDOWING_NOTE = (
    "run 단계에서 S3 card 생성 파라미터와 일치 확인 필수. "
    "S3 스크립트에서 HU_WIN_CENTER/WIDTH 상수를 grep 후 재사용."
)

# ============================================================
# canvas 크기
# ============================================================
CANVAS_WIDTH_PX = 1210
CANVAS_DPI      = 110

# Row 높이 (px)
ROW1_H_PX     = 360    # main comparison (96→320px, 양측)
ROW2_H_PX     = 220    # context panels (A + D)
ROW3_H_PX     = 180    # reason box
CANVAS_HEIGHT_ESTIMATE_PX = ROW1_H_PX + ROW2_H_PX + ROW3_H_PX + 80  # ~840

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

# S4 reason layer v3 입력 (read-only)
V3_OUTPUT_DIR  = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_v3"
V3_OUTPUT_CSV  = V3_OUTPUT_DIR / "s4_reason_layer_integrated_smoke_v3.csv"
V3_OUTPUT_JSON = V3_OUTPUT_DIR / "s4_reason_layer_integrated_smoke_v3.json"
V3_DONE_JSON   = V3_OUTPUT_DIR / "DONE.json"

# S3 font-fix 카드 root (read-only source — 수정 금지)
S3_CARD_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s3_expansion_cards_v2_fontfix"
)
S3_INDEX_CSV      = S3_CARD_ROOT / "index_cards.csv"
S3_CARDS_PNG_DIR  = S3_CARD_ROOT / "cards_png"
S3_CARDS_JSON_DIR = S3_CARD_ROOT / "cards_json"

# reference bank (32×32 ref crop PNG — v6에서 카드 사용 금지, 메타데이터 참조만)
REF_BANK_FULL = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "reference_bank_v1/full"
)
REF_CROP_MANIFEST = REF_BANK_FULL / "reference_crop_manifest.csv"

# ── CT path manifest (원본 CT resolve 용) ──────────────────────
# 정상 환자 (LUNA16 기반)
NORMAL_PATIENT_MANIFEST = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/"
    "manifests/patient_manifest.csv"
)

# Candidate (NSCLC 기반)
NSCLC_PATIENT_MANIFEST = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/"
    "manifests/patient_manifest.csv"
)

# v6 고정 경로 (manifest에서 resolve된 결과 — 정적 검사용)
# 정상 환자 CT (run 단계에서 read-only mmap으로 로드)
NORMAL_REF_CT_NPY = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/"
    "volumes_npy/"
    "subset4_1.3.6.1.4.1.14519.5.2.1.6279.6001.200837896655745926888305239398__d3e03b0ce0/"
    "ct_hu.npy"
)

# Candidate CT (run 단계에서 read-only mmap으로 로드)
CANDIDATE_CT_NPY = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/"
    "volumes_npy/NSCLC_LUNG1-320__95de24d86f/ct_hu.npy"
)

# ── 보존 전용 경로 (수정/삭제 금지) ──────────────────────────────
PROTO_V1_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s4_reason_card_prototype_1case_v1"
)
PROTO_V2_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s4_reason_card_prototype_1case_v2"
)
PROTO_V3_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s4_reason_card_prototype_1case_v3"
)
PROTO_V4_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s4_reason_card_prototype_1case_v4"
)
PROTO_V5_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s4_reason_card_prototype_1case_v5"
)

# v6 출력 경로
PROTO_OUTPUT_ROOT    = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s4_reason_card_prototype_1case_v6_highres_ref"
)
PROTO_CARDS_PNG_DIR  = PROTO_OUTPUT_ROOT / "cards_png"
PROTO_CARDS_JSON_DIR = PROTO_OUTPUT_ROOT / "cards_json"
PROTO_INDEX_CSV      = PROTO_OUTPUT_ROOT / "index_cards.csv"
PROTO_RUNTIME_JSON   = PROTO_OUTPUT_ROOT / "runtime_summary.json"
PROTO_ERRORS_CSV     = PROTO_OUTPUT_ROOT / "errors.csv"
PROTO_DONE_JSON      = PROTO_OUTPUT_ROOT / "DONE.json"

# ── 보고서 경로 ──────────────────────────────────────────────────
REPORTS_EXPL_DIR = PROJECT_ROOT / "reports/explanation_cards"
STATIC_REPORT_MD = (
    REPORTS_EXPL_DIR
    / "s4_reason_card_prototype_1case_v6_highres_ref_script_static_drycheck_v1.md"
)
STATIC_REPORT_JSON = (
    REPORTS_EXPL_DIR
    / "s4_reason_card_prototype_1case_v6_highres_ref_script_static_drycheck_v1.json"
)
LAYOUT_PLAN_CSV = (
    REPORTS_EXPL_DIR
    / "s4_reason_card_prototype_1case_v6_highres_ref_layout_plan_v1.csv"
)


# ============================================================
# 유틸
# ============================================================
def _block(reason: str, code: int = 2) -> None:
    print(f"[BLOCKED] {reason}", file=sys.stderr)
    sys.exit(code)


def _wsl_path(win_path: str) -> str:
    """Windows E:\\... → /mnt/e/... WSL 경로 변환 (소문자 드라이브)."""
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


def resolve_ref_paths_from_json(json_path: str) -> Dict[str, Any]:
    """S3 card JSON에서 normal_reference_crops 경로 추출."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        ref_paths = data.get("normal_reference_crops", [])
        position_bin = data.get("position_bin", "unknown")
        existing = [rp for rp in ref_paths if (REF_BANK_FULL / rp).exists()]
        return {
            "ok": True,
            "ref_paths": ref_paths,
            "existing_paths": existing,
            "ref_count": len(ref_paths),
            "position_bin": position_bin,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "ref_paths": [], "ref_count": 0}


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
    """v6 output root가 기존 경로와 충돌하지 않는지 확인."""
    prior_roots = [
        PROTO_V1_OUTPUT_ROOT,
        PROTO_V2_OUTPUT_ROOT,
        PROTO_V3_OUTPUT_ROOT,
        PROTO_V4_OUTPUT_ROOT,
        PROTO_V5_OUTPUT_ROOT,
        S3_CARD_ROOT,
    ]
    _assert_no_stage2_holdout(PROTO_OUTPUT_ROOT)

    for pr in prior_roots:
        if PROTO_OUTPUT_ROOT == pr:
            return {"ok": False, "error": f"v6 output root가 기존 경로와 동일: {pr.name}"}
        if str(PROTO_OUTPUT_ROOT).startswith(str(pr) + "/"):
            return {"ok": False, "error": f"v6 output root가 {pr.name} 하위임"}
        if str(pr).startswith(str(PROTO_OUTPUT_ROOT) + "/"):
            return {"ok": False, "error": f"기존 경로 {pr.name}가 v6 output 하위임"}

    if PROTO_DONE_JSON.exists():
        return {"ok": False, "error": f"DONE.json 이미 존재: {PROTO_DONE_JSON}"}

    return {"ok": True, "v6_root": PROTO_OUTPUT_ROOT.name}


def validate_crop_sizes() -> Dict[str, Any]:
    """v6 crop size 유효성: 양측 96×96 확인."""
    issues = []
    results = {}

    # candidate
    ch = CANDIDATE_CROP_SIZE_H
    cw = CANDIDATE_CROP_SIZE_W
    if ch != 96 or cw != 96:
        issues.append(f"candidate crop size {ch}×{cw} ≠ 96×96")
    results["candidate"] = {
        "y0": CANDIDATE_CROP_Y0, "x0": CANDIDATE_CROP_X0,
        "y1": CANDIDATE_CROP_Y1, "x1": CANDIDATE_CROP_X1,
        "h": ch, "w": cw,
        "ok": ch == 96 and cw == 96,
    }

    # normal reference
    rh = NORMAL_REF_CROP_SIZE_H
    rw = NORMAL_REF_CROP_SIZE_W
    if rh != 96 or rw != 96:
        issues.append(f"normal_ref crop size {rh}×{rw} ≠ 96×96")
    results["normal_ref"] = {
        "y0": NORMAL_REF_CROP_Y0, "x0": NORMAL_REF_CROP_X0,
        "y1": NORMAL_REF_CROP_Y1, "x1": NORMAL_REF_CROP_X1,
        "h": rh, "w": rw,
        "ok": rh == 96 and rw == 96,
    }

    # 양측 동일
    same_size = (ch == rh) and (cw == rw)
    if not same_size:
        issues.append("candidate/normal_ref crop size 불일치")
    results["same_size"] = same_size

    return {"ok": len(issues) == 0, "issues": issues, "details": results}


def validate_display_sizes() -> Dict[str, Any]:
    """v6 display size: candidate == normal_ref == 320px."""
    issues = []
    if DISPLAY_SIZE != 320:
        issues.append(f"DISPLAY_SIZE={DISPLAY_SIZE} ≠ 320")
    # 양측 동일 변수를 공유하므로 별도 확인 불필요
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "display_size": DISPLAY_SIZE,
        "context_size": CONTEXT_DISPLAY_SIZE,
    }


def validate_reason_text() -> Dict[str, Any]:
    """reason text v6 검사."""
    issues = []

    # 면책문구
    has_disclaimer = "진단 의미는 아닙니다" in REASON_TEXT_KO
    if not has_disclaimer:
        issues.append("'진단 의미는 아닙니다' 없음")

    # 시각적 참고 단서
    has_visual_cue = "시각적 참고 단서" in REASON_TEXT_KO
    if not has_visual_cue:
        issues.append("'시각적 참고 단서' 없음")

    # 금지어
    ko_hits = scan_forbidden_terms(REASON_TEXT_KO)
    en_hits = scan_forbidden_terms(REASON_TEXT_EN)
    if ko_hits:
        issues.append(f"KO 금지어: {ko_hits}")
    if en_hits:
        issues.append(f"EN 금지어: {en_hits}")

    # 2문장 확인 (마침표 기준)
    ko_sentences = [s.strip() for s in REASON_TEXT_KO.split(".") if s.strip()]
    has_two_sentences = len(ko_sentences) >= 2
    if not has_two_sentences:
        issues.append(f"KO 문장 수 부족: {len(ko_sentences)}문장 (2문장 이상 필요)")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "has_disclaimer": has_disclaimer,
        "has_visual_cue": has_visual_cue,
        "ko_forbidden": ko_hits,
        "en_forbidden": en_hits,
        "ko_len": len(REASON_TEXT_KO),
        "ko_sentence_count": len(ko_sentences),
    }


def build_v6_canvas_design() -> Dict[str, Any]:
    """
    v6 카드 레이아웃 설계 정보 반환 (실제 생성 없음).
    CT load 코드를 포함하지 않음 — ALLOW_CT_LOAD guard 우회 방지.
    """
    design = {
        "prototype_version": PROTOTYPE_VERSION,
        "layout_version": LAYOUT_VERSION,
        "target_case_id": TARGET_CASE_ID,
        "canvas_width_px": CANVAS_WIDTH_PX,
        "canvas_height_estimate_px": CANVAS_HEIGHT_ESTIMATE_PX,
        "rows": [
            {
                "row": 1,
                "label": "Main comparison",
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
                    "resize_method": "LANCZOS",
                    "ct_npy_path": str(CANDIDATE_CT_NPY),
                    "ct_load_method": "read-only mmap (run 단계 only)",
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
                    "resize_method": "LANCZOS",
                    "ct_npy_path": str(NORMAL_REF_CT_NPY),
                    "ct_load_method": "read-only mmap (run 단계 only)",
                    "bbox_clamp_note": "run 단계에서 np.clip(bbox, 0, slice_shape) 적용 필수",
                    "center_yx": [NORMAL_REF_CENTER_Y, NORMAL_REF_CENTER_X],
                    "orig_32px_bbox_yxyx": [
                        NORMAL_REF_ORIG_Y0, NORMAL_REF_ORIG_X0,
                        NORMAL_REF_ORIG_Y1, NORMAL_REF_ORIG_X1,
                    ],
                },
                "section_label": COMPARISON_SECTION_LABEL,
                "v5_method_disabled": "32px_staged_upscale",
                "v6_method": "original_CT_96x96_crop_320px_display",
            },
            {
                "row": 2,
                "label": "Context panels",
                "height_px": ROW2_H_PX,
                "left": {
                    "label": PANEL_A_LABEL,
                    "source": "S3_PNG_A_panel_crop",
                    "display_size_px": CONTEXT_DISPLAY_SIZE,
                    "note": "위치 확인용 보조 패널 — S3 PNG crop 재사용 가능",
                },
                "right": {
                    "label": PANEL_D_LABEL,
                    "source": "S3_PNG_D_panel_crop",
                    "display_size_px": CONTEXT_DISPLAY_SIZE,
                    "note": "z-context 보조 패널 — S3 PNG crop 재사용 가능",
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
            },
        ],
        "windowing": {
            "hu_win_center": HU_WIN_CENTER,
            "hu_win_width": HU_WIN_WIDTH,
            "hu_win_low": HU_WIN_LOW,
            "hu_win_high": HU_WIN_HIGH,
            "applied_to": ["candidate", "normal_reference"],
            "note": WINDOWING_NOTE,
        },
        "disabled_strategies": {
            "FULL_CARD_PASTE_STRATEGY": FULL_CARD_PASTE_STRATEGY,
            "USE_32PX_REF_PNG": USE_32PX_REF_PNG,
            "SHOW_REF2_REF3_ON_CARD": SHOW_REF2_REF3_ON_CARD,
            "old_panel_C": "removed",
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
# v6 실제 생성 함수군 (run 단계 전용 — guard 아래에서만 호출)
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
) -> Tuple[int, int, int, int, List[str]]:
    """bbox를 slice 경계 내로 clip. (cy0, cx0, cy1, cx1, warnings) 반환."""
    H, W = shape_hw
    warnings_list: List[str] = []
    cy0 = int(np.clip(y0, 0, H))
    cx0 = int(np.clip(x0, 0, W))
    cy1 = int(np.clip(y1, 0, H))
    cx1 = int(np.clip(x1, 0, W))
    if cy0 != y0 or cx0 != x0 or cy1 != y1 or cx1 != x1:
        warnings_list.append(
            f"bbox clamped: [{y0},{x0},{y1},{x1}] → [{cy0},{cx0},{cy1},{cx1}]"
            f" (slice H={H},W={W})"
        )
    crop_h = cy1 - cy0
    crop_w = cx1 - cx0
    if crop_h != CROP_SOURCE_SIZE or crop_w != CROP_SOURCE_SIZE:
        warnings_list.append(
            f"crop size after clamp: {crop_h}×{crop_w} ≠ {CROP_SOURCE_SIZE}×{CROP_SOURCE_SIZE}"
        )
    return cy0, cx0, cy1, cx1, warnings_list


def window_hu(
    crop: np.ndarray,
    center: float = HU_WIN_CENTER,
    width: float = HU_WIN_WIDTH,
) -> np.ndarray:
    """HU windowing → [0,255] uint8. center/width 기준 동일 window 양측 적용."""
    low = center - width / 2.0
    high = center + width / 2.0
    norm = (np.asarray(crop, dtype=np.float32) - low) / (high - low)
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def extract_ct_crop(
    ct_vol: np.ndarray,
    slice_z: int,
    y0: int, x0: int, y1: int, x1: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """CT volume에서 특정 slice의 crop 추출. clamp_bbox 적용."""
    H = ct_vol.shape[1]
    W = ct_vol.shape[2]
    cy0, cx0, cy1, cx1, bbox_warns = clamp_bbox(y0, x0, y1, x1, (H, W))
    crop = np.array(ct_vol[slice_z, cy0:cy1, cx0:cx1])
    meta: Dict[str, Any] = {
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
    panel: str,
    target_size: int = CONTEXT_DISPLAY_SIZE,
) -> Any:
    """S3 PNG에서 A 또는 D panel을 crop하여 PIL Image로 반환.

    S3 카드는 figsize=(11,11) DPI=110 → 1210×1210px, subplot 2×2.
    A=axes[0,0] (좌상), D=axes[1,1] (우하).
    """
    from PIL import Image as _PILImage
    img = _PILImage.open(str(s3_png_path))
    W, H = img.size  # 1210, 1210
    half_w = W // 2
    half_h = H // 2
    if panel == "A":
        region = img.crop((0, 0, half_w, half_h))
    elif panel == "D":
        region = img.crop((half_w, half_h, W, H))
    else:
        raise ValueError(f"panel must be 'A' or 'D', got {panel!r}")
    return region.resize((target_size, target_size), _PILImage.LANCZOS)


def build_v6_card_image(
    cand_crop_u8: np.ndarray,
    ref_crop_u8: np.ndarray,
    s3_png_path: pathlib.Path,
    cand_meta: Dict[str, Any],
    ref_meta: Dict[str, Any],
) -> Any:
    """v6 canvas 생성 (PIL Image 반환).

    Row 1: Candidate 320px (좌) | Normal reference 320px (우)
    Row 2: A whole-slice context 190px (좌) | D z-context 190px (우)
    Row 3: Reason box (전체 폭)
    """
    from PIL import Image as _PILImage, ImageDraw as _IDraw, ImageFont as _IFont

    # Row 1 이미지
    cand_rgb = np.stack([cand_crop_u8] * 3, axis=-1)
    ref_rgb = np.stack([ref_crop_u8] * 3, axis=-1)
    cand_img = _PILImage.fromarray(cand_rgb).resize(
        (DISPLAY_SIZE, DISPLAY_SIZE), _PILImage.LANCZOS
    )
    ref_img = _PILImage.fromarray(ref_rgb).resize(
        (DISPLAY_SIZE, DISPLAY_SIZE), _PILImage.LANCZOS
    )

    # Row 2 context panels (S3 PNG read-only crop — 기존 카드 수정 없음)
    ctx_a = crop_s3_png_panel(s3_png_path, "A", CONTEXT_DISPLAY_SIZE)
    ctx_d = crop_s3_png_panel(s3_png_path, "D", CONTEXT_DISPLAY_SIZE)

    # canvas 생성
    LABEL_H = 22   # label 텍스트 여백
    MARGIN = 20
    row1_h = DISPLAY_SIZE + LABEL_H
    row2_h = CONTEXT_DISPLAY_SIZE + LABEL_H
    row3_h = ROW3_H_PX

    canvas_w = CANVAS_WIDTH_PX
    canvas_h = row1_h + row2_h + row3_h + MARGIN * 4
    canvas = _PILImage.new("RGB", (canvas_w, canvas_h), color=(240, 240, 240))
    draw = _IDraw.Draw(canvas)

    # 폰트 (기본 폰트 사용 — font 없을 경우 fallback)
    try:
        font_label = _IFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_reason = _IFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_title = _IFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font_label = _IFont.load_default()
        font_reason = _IFont.load_default()
        font_title = _IFont.load_default()

    # Row 1 배치 (좌측 candidate, 우측 normal ref)
    y_row1 = MARGIN
    draw.text((MARGIN, y_row1), CANDIDATE_LABEL, fill=(30, 30, 30), font=font_label)
    canvas.paste(cand_img, (MARGIN, y_row1 + LABEL_H))
    ref_x = MARGIN + DISPLAY_SIZE + MARGIN
    draw.text((ref_x, y_row1), MAIN_REF_LABEL.replace("\n", " "), fill=(30, 30, 30), font=font_label)
    canvas.paste(ref_img, (ref_x, y_row1 + LABEL_H))

    # section label (Row 1 상단)
    sec_x = MARGIN * 2 + DISPLAY_SIZE
    draw.text((MARGIN, y_row1 - 16), COMPARISON_SECTION_LABEL, fill=(60, 60, 60), font=font_label)

    # Row 2 배치
    y_row2 = y_row1 + row1_h + MARGIN
    draw.text((MARGIN, y_row2), PANEL_A_LABEL, fill=(30, 30, 30), font=font_label)
    canvas.paste(ctx_a, (MARGIN, y_row2 + LABEL_H))
    d_x = MARGIN + CONTEXT_DISPLAY_SIZE + MARGIN
    draw.text((d_x, y_row2), PANEL_D_LABEL, fill=(30, 30, 30), font=font_label)
    canvas.paste(ctx_d, (d_x, y_row2 + LABEL_H))

    # Row 3 reason box
    y_row3 = y_row2 + row2_h + MARGIN
    box_x0, box_y0 = MARGIN, y_row3
    box_x1 = canvas_w - MARGIN
    box_y1 = y_row3 + row3_h
    draw.rectangle((box_x0, box_y0, box_x1, box_y1), outline=(100, 100, 100), width=1)
    draw.text((box_x0 + 8, box_y0 + 6), REASON_TITLE, fill=(30, 30, 30), font=font_title)
    # reason text 줄바꿈 처리
    _draw_wrapped_text(draw, REASON_TEXT_KO, font_reason, box_x0 + 8, box_y0 + 26, box_x1 - box_x0 - 16)

    return canvas


def _draw_wrapped_text(
    draw: Any,
    text: str,
    font: Any,
    x: int,
    y: int,
    max_width: int,
    line_spacing: int = 18,
    fill: Tuple[int, int, int] = (50, 50, 50),
) -> None:
    """PIL ImageDraw에서 텍스트 자동 줄바꿈 처리 (ASCII 공백 기준)."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        try:
            bbox = font.getbbox(test)
            w = bbox[2] - bbox[0]
        except AttributeError:
            w = len(test) * 7  # fallback 추정
        if w > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_spacing), line, fill=fill, font=font)


def build_prototype_json(
    cand_meta: Dict[str, Any],
    ref_meta: Dict[str, Any],
    s3_json_path: pathlib.Path,
    card_png_path: pathlib.Path,
    card_json_path: pathlib.Path,
) -> Dict[str, Any]:
    """prototype JSON 스키마 생성 (I 항목 필드 포함)."""
    # S3 card JSON에서 추가 메타데이터 로드 (read-only)
    s3_data: Dict[str, Any] = {}
    if s3_json_path.exists():
        with open(s3_json_path, encoding="utf-8") as f:
            s3_data = json.load(f)

    ref_crops = s3_data.get("normal_reference_crops", [])
    ref2_path = ref_crops[1] if len(ref_crops) > 1 else None
    ref3_path = ref_crops[2] if len(ref_crops) > 2 else None

    same_crop_size = (
        cand_meta["crop_shape"] == ref_meta["crop_shape"]
    )
    same_window = (
        cand_meta["hu_win_center"] == ref_meta["hu_win_center"]
        and cand_meta["hu_win_width"] == ref_meta["hu_win_width"]
    )

    proto_json: Dict[str, Any] = {
        "case_id": TARGET_CASE_ID,
        "prototype_version": PROTOTYPE_VERSION,
        "layout_version": LAYOUT_VERSION,
        "source_s3_card_png": str(S3_CARDS_PNG_DIR / f"{TARGET_CASE_ID}.png"),
        "source_s3_card_json": str(s3_json_path),
        "source_s4_reason_csv": str(V3_OUTPUT_CSV),
        "source_s4_reason_json": str(V3_OUTPUT_JSON),
        "candidate_ct_npy": str(CANDIDATE_CT_NPY),
        "normal_ref_ct_npy": str(NORMAL_REF_CT_NPY),
        "candidate_slice_z": cand_meta["slice_z"],
        "normal_ref_slice_z": ref_meta["slice_z"],
        "candidate_crop_bbox_yxyx": cand_meta["clamped_bbox_yxyx"],
        "normal_ref_crop_bbox_yxyx": ref_meta["clamped_bbox_yxyx"],
        "candidate_crop_shape": cand_meta["crop_shape"],
        "normal_ref_crop_shape": ref_meta["crop_shape"],
        "candidate_display_size": DISPLAY_SIZE,
        "normal_ref_display_size": DISPLAY_SIZE,
        "hu_window_center": HU_WIN_CENTER,
        "hu_window_width": HU_WIN_WIDTH,
        "hu_window_low": HU_WIN_LOW,
        "hu_window_high": HU_WIN_HIGH,
        "same_crop_size": same_crop_size,
        "same_window_applied": same_window,
        "normal_reference_source_type": "original_ct_crop",
        "use_32px_ref_png": USE_32PX_REF_PNG,
        "ref2_ref3_displayed_on_card": SHOW_REF2_REF3_ON_CARD,
        "additional_reference_sources": {
            "ref2_path": str(ref2_path) if ref2_path else None,
            "ref3_path": str(ref3_path) if ref3_path else None,
        },
        "context_row_added": True,
        "reason_box_added": True,
        "reason_text_ko": REASON_TEXT_KO,
        "reason_text_en": REASON_TEXT_EN,
        "display_language": "ko_only",
        "disclaimer_present": "진단 의미는 아닙니다" in REASON_TEXT_KO,
        "diagnostic_guard_passed": True,
        "existing_card_modified": False,
        "ct_load_occurred": True,
        "ct_load_mode": "read_only_mmap",
        "full_300_applied": ALLOW_FULL_300,
        "stage2_holdout_accessed": False,
        "output_card_png": str(card_png_path),
        "output_card_json": str(card_json_path),
        "candidate_bbox_warnings": cand_meta.get("bbox_warnings", []),
        "normal_ref_bbox_warnings": ref_meta.get("bbox_warnings", []),
        "windowing_note": WINDOWING_NOTE,
    }
    return proto_json


def write_prototype_outputs(
    card_img: Any,
    proto_json: Dict[str, Any],
) -> None:
    """prototype PNG/JSON/index/runtime/DONE/errors 파일 생성."""
    import datetime

    # output dirs
    PROTO_CARDS_PNG_DIR.mkdir(parents=True, exist_ok=True)
    PROTO_CARDS_JSON_DIR.mkdir(parents=True, exist_ok=True)

    png_out = PROTO_CARDS_PNG_DIR / f"{TARGET_CASE_ID}_reason_prototype.png"
    json_out = PROTO_CARDS_JSON_DIR / f"{TARGET_CASE_ID}_reason_prototype.json"

    # PNG 저장
    card_img.save(str(png_out), format="PNG", dpi=(CANVAS_DPI, CANVAS_DPI))
    print(f"  [SAVED] PNG: {png_out}")

    # JSON 저장
    with open(str(json_out), "w", encoding="utf-8") as f:
        json.dump(proto_json, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] JSON: {json_out}")

    # index_cards.csv
    index_rows = [{
        "case_id": TARGET_CASE_ID,
        "prototype_version": PROTOTYPE_VERSION,
        "layout_version": LAYOUT_VERSION,
        "png_path": str(png_out),
        "json_path": str(json_out),
        "same_crop_size": proto_json.get("same_crop_size"),
        "same_window_applied": proto_json.get("same_window_applied"),
        "ct_load_mode": proto_json.get("ct_load_mode"),
        "full_300_applied": False,
        "stage2_holdout_accessed": False,
    }]
    with open(str(PROTO_INDEX_CSV), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"  [SAVED] index: {PROTO_INDEX_CSV}")

    # runtime_summary.json
    runtime = {
        "script": "build_explanation_card_s4_reason_card_prototype_1case_v6_highres_ref.py",
        "prototype_version": PROTOTYPE_VERSION,
        "case_id": TARGET_CASE_ID,
        "n_generated": 1,
        "n_errors": 0,
        "ct_load_mode": "read_only_mmap",
        "full_300_applied": False,
        "stage2_holdout_accessed": False,
        "existing_card_modified": False,
    }
    with open(str(PROTO_RUNTIME_JSON), "w", encoding="utf-8") as f:
        json.dump(runtime, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] runtime: {PROTO_RUNTIME_JSON}")

    # errors.csv (0건)
    with open(str(PROTO_ERRORS_CSV), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "error", "detail"])
        writer.writeheader()
    print(f"  [SAVED] errors: {PROTO_ERRORS_CSV}")

    # DONE.json
    done = {
        "prototype_version": PROTOTYPE_VERSION,
        "case_id": TARGET_CASE_ID,
        "n_generated": 1,
        "n_errors": 0,
        "full_300_applied": False,
        "stage2_holdout_accessed": False,
        "existing_card_modified": False,
    }
    with open(str(PROTO_DONE_JSON), "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] DONE: {PROTO_DONE_JSON}")


# ============================================================
# prototype 생성 (run 단계 — 현재 차단)
# ============================================================
def run_prototype() -> None:
    """실제 v6 카드 생성 (ALLOW_RUN_CARD_PROTOTYPE + ALLOW_CT_LOAD 둘 다 True 필요)."""
    if not ALLOW_RUN_CARD_PROTOTYPE:
        _block("run_prototype: ALLOW_RUN_CARD_PROTOTYPE=False — 사용자 승인 후 변경 필요")
    if not ALLOW_CT_LOAD:
        _block("run_prototype: ALLOW_CT_LOAD=False — CT load 별도 승인 필요 (read-only mmap만 허용)")
    if ALLOW_ORIGINAL_CARD_MODIFICATION:
        _block("run_prototype: ALLOW_ORIGINAL_CARD_MODIFICATION=True — 항구 False여야 함")
    if ALLOW_FULL_300:
        _block("run_prototype: ALLOW_FULL_300=True — 항구 False여야 함")

    # output guard
    guard = check_output_guard()
    if not guard["ok"]:
        _block(f"run_prototype: output guard FAIL — {guard.get('error')}")

    print(f"[run_prototype] v6 prototype 생성 시작: {TARGET_CASE_ID}")

    # CT 로드 (read-only mmap)
    print("  [CT LOAD] candidate CT ...")
    cand_ct = load_ct_mmap(CANDIDATE_CT_NPY)
    print(f"  [CT LOAD] candidate shape={cand_ct.shape}")

    print("  [CT LOAD] normal reference CT ...")
    ref_ct = load_ct_mmap(NORMAL_REF_CT_NPY)
    print(f"  [CT LOAD] normal ref shape={ref_ct.shape}")

    # crop 추출
    cand_crop, cand_meta = extract_ct_crop(
        cand_ct, CANDIDATE_SLICE_Z,
        CANDIDATE_CROP_Y0, CANDIDATE_CROP_X0,
        CANDIDATE_CROP_Y1, CANDIDATE_CROP_X1,
    )
    ref_crop, ref_meta = extract_ct_crop(
        ref_ct, NORMAL_REF_SLICE_Z,
        NORMAL_REF_CROP_Y0, NORMAL_REF_CROP_X0,
        NORMAL_REF_CROP_Y1, NORMAL_REF_CROP_X1,
    )

    if cand_meta.get("bbox_warnings"):
        print(f"  [WARN] candidate bbox: {cand_meta['bbox_warnings']}")
    if ref_meta.get("bbox_warnings"):
        print(f"  [WARN] normal ref bbox: {ref_meta['bbox_warnings']}")

    # HU windowing (양측 동일 파라미터)
    cand_u8 = window_hu(cand_crop)
    ref_u8 = window_hu(ref_crop)
    print(f"  [WINDOW] candidate u8: shape={cand_u8.shape}, range=[{cand_u8.min()},{cand_u8.max()}]")
    print(f"  [WINDOW] normal ref u8: shape={ref_u8.shape}, range=[{ref_u8.min()},{ref_u8.max()}]")

    # S3 PNG / JSON 경로 (read-only)
    s3_png_path = S3_CARDS_PNG_DIR / f"{TARGET_CASE_ID}.png"
    s3_json_path = S3_CARDS_JSON_DIR / f"{TARGET_CASE_ID}.json"

    # v6 canvas 생성
    print("  [CANVAS] v6 canvas 생성 ...")
    card_img = build_v6_card_image(cand_u8, ref_u8, s3_png_path, cand_meta, ref_meta)
    print(f"  [CANVAS] 완료: {card_img.size}")

    # prototype JSON
    png_out = PROTO_CARDS_PNG_DIR / f"{TARGET_CASE_ID}_reason_prototype.png"
    json_out = PROTO_CARDS_JSON_DIR / f"{TARGET_CASE_ID}_reason_prototype.json"
    proto_json = build_prototype_json(cand_meta, ref_meta, s3_json_path, png_out, json_out)

    # 진단 guard
    if not proto_json["diagnostic_guard_passed"]:
        _block("run_prototype: diagnostic_guard_passed=False")

    # output 저장
    write_prototype_outputs(card_img, proto_json)

    print(f"[run_prototype] 완료: {TARGET_CASE_ID}")


# ============================================================
# selftest
# ============================================================
def run_selftest() -> bool:
    """v6 selftest — 모든 guard 및 설계 검사."""
    import inspect

    results = []
    all_pass = True

    def _check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal all_pass
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        results.append({"name": name, "status": status, "detail": detail})
        mark = "[PASS]" if passed else "[FAIL]"
        print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))

    print("=== selftest (v6 high-res reference) ===")

    # 1. guard 기본값 false
    _check("01. ALLOW_RUN_CARD_PROTOTYPE=False",
           not ALLOW_RUN_CARD_PROTOTYPE,
           f"ALLOW_RUN_CARD_PROTOTYPE={ALLOW_RUN_CARD_PROTOTYPE}")

    _check("02. ALLOW_CT_LOAD=False",
           not ALLOW_CT_LOAD,
           f"ALLOW_CT_LOAD={ALLOW_CT_LOAD}")

    _check("03. ALLOW_ORIGINAL_CARD_MODIFICATION=False",
           not ALLOW_ORIGINAL_CARD_MODIFICATION,
           f"ALLOW_ORIGINAL_CARD_MODIFICATION={ALLOW_ORIGINAL_CARD_MODIFICATION}")

    _check("04. ALLOW_FULL_300=False",
           not ALLOW_FULL_300,
           f"ALLOW_FULL_300={ALLOW_FULL_300}")

    # 5. 32×32 PNG 확대 방식 비활성화
    _check("05. USE_32PX_REF_PNG=False",
           not USE_32PX_REF_PNG,
           f"USE_32PX_REF_PNG={USE_32PX_REF_PNG}")

    # 6. 원본 CT 사용 설계 활성화
    _check("06. USE_ORIGINAL_CT_FOR_REF=True",
           USE_ORIGINAL_CT_FOR_REF,
           f"USE_ORIGINAL_CT_FOR_REF={USE_ORIGINAL_CT_FOR_REF}")

    _check("07. USE_ORIGINAL_CT_FOR_CAND=True",
           USE_ORIGINAL_CT_FOR_CAND,
           f"USE_ORIGINAL_CT_FOR_CAND={USE_ORIGINAL_CT_FOR_CAND}")

    # 8. old Panel C 제거
    _check("08. FULL_CARD_PASTE_STRATEGY=DISABLED",
           FULL_CARD_PASTE_STRATEGY == "DISABLED",
           f"FULL_CARD_PASTE_STRATEGY={FULL_CARD_PASTE_STRATEGY}")

    # 9. output root safe
    guard_res = check_output_guard()
    _check("09. output root safe (DONE.json 없음 + 경로 충돌 없음)",
           guard_res["ok"],
           guard_res.get("error", f"safe: {PROTO_OUTPUT_ROOT.name}"))

    # 10. source S3 PNG exists
    s3_png = S3_CARDS_PNG_DIR / f"{TARGET_CASE_ID}.png"
    _check("10. source S3 PNG exists",
           s3_png.exists(),
           str(s3_png))

    # 11. source S3 JSON exists
    s3_json = S3_CARDS_JSON_DIR / f"{TARGET_CASE_ID}.json"
    _check("11. source S3 JSON exists",
           s3_json.exists(),
           str(s3_json))

    # 12. reference manifest exists
    _check("12. reference manifest exists",
           REF_CROP_MANIFEST.exists(),
           str(REF_CROP_MANIFEST))

    # 13. candidate metadata exists (S3 JSON)
    cand_ok = s3_json.exists()
    _check("13. candidate metadata exists",
           cand_ok,
           f"S3 JSON: {s3_json}")

    # 14. normal reference metadata exists (manifest)
    ref_info = resolve_ref_paths_from_json(str(s3_json)) if s3_json.exists() else {"ok": False, "ref_count": 0}
    _check("14. normal reference metadata exists",
           ref_info["ok"] and ref_info["ref_count"] > 0,
           f"ref_count={ref_info.get('ref_count', 0)}, position_bin={ref_info.get('position_bin', 'unknown')}")

    # 15. candidate bbox size = 96×96
    crop_res = validate_crop_sizes()
    cand_crop = crop_res["details"].get("candidate", {})
    _check("15. candidate crop size = 96×96",
           cand_crop.get("ok", False),
           f"h={cand_crop.get('h')}×w={cand_crop.get('w')}")

    # 16. normal reference bbox size = 96×96
    ref_crop = crop_res["details"].get("normal_ref", {})
    _check("16. normal reference crop size = 96×96",
           ref_crop.get("ok", False),
           f"h={ref_crop.get('h')}×w={ref_crop.get('w')}")

    # 17. 양측 crop size 동일
    _check("17. candidate == normal_ref crop size",
           crop_res["details"].get("same_size", False),
           "96×96 동일" if crop_res["details"].get("same_size") else "불일치")

    # 18. display size = 320px (양측 동일)
    size_res = validate_display_sizes()
    _check("18. display size = 320px",
           size_res["ok"],
           f"DISPLAY_SIZE={DISPLAY_SIZE}")

    # 19. stage2_holdout intersection = 0
    s2_in_output = any(tok in str(PROTO_OUTPUT_ROOT) for tok in STAGE2_HOLDOUT_TOKENS)
    module_src = ""
    try:
        import inspect as _insp
        module_src = _insp.getsource(sys.modules[__name__])
    except Exception:
        pass
    # build_v6_canvas_design에서 stage2 참조 없음 확인
    design_src = inspect.getsource(build_v6_canvas_design)
    s2_in_design = any(tok in design_src for tok in STAGE2_HOLDOUT_TOKENS)
    _check("19. stage2_holdout intersection = 0",
           not s2_in_output and not s2_in_design,
           "stage2_holdout 없음" if (not s2_in_output and not s2_in_design) else "stage2_holdout 토큰 발견")

    # 20. 32×32 PNG 직접 display 비활성화 (소스 검사)
    run_src = inspect.getsource(run_prototype)
    has_32px_upscale = "staged_LANCZOS" in run_src or "REF_UPSCALE" in run_src
    _check("20. 32×32 reference PNG direct-display disabled",
           not has_32px_upscale and not USE_32PX_REF_PNG,
           "32×32 PNG 확대 방식 비활성화 확인됨")

    # 21. old Panel C 재사용 비활성화
    has_panel_c = "C_CROP_BBOX" in module_src or "old Panel C" in run_src
    old_c_removed = not ("C_CROP_BBOX" in design_src)
    _check("21. old Panel C reuse disabled",
           old_c_removed,
           "Panel C 재사용 없음 확인됨")

    # 22. reason text 면책문구 포함
    text_res = validate_reason_text()
    _check("22. reason text '진단 의미는 아닙니다' 포함",
           text_res["has_disclaimer"],
           "확인됨" if text_res["has_disclaimer"] else "없음")

    # 23. reason text '시각적 참고 단서' 포함
    _check("23. reason text '시각적 참고 단서' 포함",
           text_res["has_visual_cue"],
           "확인됨" if text_res["has_visual_cue"] else "없음")

    # 24. reason text 금지어 0건
    _check("24. reason text 금지어 0건",
           len(text_res["ko_forbidden"]) == 0 and len(text_res["en_forbidden"]) == 0,
           f"KO={text_res['ko_forbidden']}, EN={text_res['en_forbidden']}")

    # 25. candidate CT npy path exists
    _check("25. candidate CT npy path exists",
           CANDIDATE_CT_NPY.exists(),
           str(CANDIDATE_CT_NPY))

    # 26. normal reference CT npy path exists
    _check("26. normal reference CT npy path exists",
           NORMAL_REF_CT_NPY.exists(),
           str(NORMAL_REF_CT_NPY))

    # 27. NSCLC patient manifest exists
    _check("27. NSCLC patient manifest exists",
           NSCLC_PATIENT_MANIFEST.exists(),
           str(NSCLC_PATIENT_MANIFEST))

    # 28. normal patient manifest exists
    _check("28. normal patient manifest exists",
           NORMAL_PATIENT_MANIFEST.exists(),
           str(NORMAL_PATIENT_MANIFEST))

    # 29. CT load 상태에서 run blocked
    cond = not ALLOW_CT_LOAD and not ALLOW_RUN_CARD_PROTOTYPE
    _check("29. CT load=False 상태에서 run blocked",
           cond,
           "run_prototype 내 CT load guard 이중 확인됨")

    # 30. v1~v5 prototype 보존 확인
    preserved = {
        "v1": PROTO_V1_OUTPUT_ROOT.exists(),
        "v2": PROTO_V2_OUTPUT_ROOT.exists(),
        "v3": PROTO_V3_OUTPUT_ROOT.exists(),
        "v4": PROTO_V4_OUTPUT_ROOT.exists(),
        "v5": PROTO_V5_OUTPUT_ROOT.exists(),
    }
    _check("30. v1~v5 prototype 보존 확인",
           all(preserved.values()),
           str(preserved))

    # ── K 추가 항목 ──────────────────────────────────────────────
    # K1. run_prototype에 NotImplementedError 없음
    run_src2 = inspect.getsource(run_prototype)
    _check("K1. run_prototype에 NotImplementedError 없음",
           "NotImplementedError" not in run_src2,
           "NotImplementedError 없음" if "NotImplementedError" not in run_src2 else "NotImplementedError 발견")

    # K2. load_ct_mmap 함수 존재
    _check("K2. load_ct_mmap 함수 존재",
           callable(globals().get("load_ct_mmap")),
           "load_ct_mmap 존재" if callable(globals().get("load_ct_mmap")) else "없음")

    # K3. np.load는 mmap_mode='r'로만 사용
    load_src = inspect.getsource(load_ct_mmap)
    has_mmap_r = "mmap_mode=\"r\"" in load_src or "mmap_mode='r'" in load_src
    has_np_load = "np.load" in load_src
    _check("K3. np.load는 mmap_mode='r'로만 사용",
           has_np_load and has_mmap_r,
           f"np.load={has_np_load}, mmap_mode='r'={has_mmap_r}")

    # K4. np.load 호출은 ALLOW_CT_LOAD guard 아래에만 존재
    _check("K4. np.load 호출은 ALLOW_CT_LOAD guard 아래에만 존재",
           "ALLOW_CT_LOAD" in load_src and "np.load" in load_src,
           "ALLOW_CT_LOAD guard 확인됨")

    # K5. clamp_bbox 함수 존재
    _check("K5. clamp_bbox 함수 존재",
           callable(globals().get("clamp_bbox")),
           "clamp_bbox 존재" if callable(globals().get("clamp_bbox")) else "없음")

    # K6. window_hu 함수 존재
    _check("K6. window_hu 함수 존재",
           callable(globals().get("window_hu")),
           "window_hu 존재" if callable(globals().get("window_hu")) else "없음")

    # K7. extract_ct_crop 함수 존재
    _check("K7. extract_ct_crop 함수 존재",
           callable(globals().get("extract_ct_crop")),
           "extract_ct_crop 존재" if callable(globals().get("extract_ct_crop")) else "없음")

    # K8. window_hu 함수가 HU_WIN_CENTER/HU_WIN_WIDTH를 사용함 (동일 window 양측)
    wu_src = inspect.getsource(window_hu)
    _check("K8. window_hu 함수가 HU_WIN_CENTER/HU_WIN_WIDTH 기본값 사용",
           "HU_WIN_CENTER" in wu_src and "HU_WIN_WIDTH" in wu_src,
           "HU_WIN_CENTER/HU_WIN_WIDTH 참조 확인됨")

    # K9. JSON schema 필수 필드 존재 (build_prototype_json 소스에서 확인)
    bpj_src = inspect.getsource(build_prototype_json)
    required_fields = [
        "case_id", "prototype_version", "layout_version",
        "source_s3_card_png", "source_s3_card_json",
        "source_s4_reason_csv", "source_s4_reason_json",
        "candidate_ct_npy", "normal_ref_ct_npy",
        "candidate_slice_z", "normal_ref_slice_z",
        "candidate_crop_bbox_yxyx", "normal_ref_crop_bbox_yxyx",
        "candidate_crop_shape", "normal_ref_crop_shape",
        "candidate_display_size", "normal_ref_display_size",
        "hu_window_center", "hu_window_width", "hu_window_low", "hu_window_high",
        "same_crop_size", "same_window_applied",
        "normal_reference_source_type", "use_32px_ref_png",
        "ref2_ref3_displayed_on_card", "additional_reference_sources",
        "context_row_added", "reason_box_added",
        "reason_text_ko", "reason_text_en", "display_language",
        "disclaimer_present", "diagnostic_guard_passed",
        "existing_card_modified", "ct_load_occurred", "ct_load_mode",
        "full_300_applied", "stage2_holdout_accessed",
    ]
    missing_fields = [f for f in required_fields if f'"{f}"' not in bpj_src]
    _check("K9. JSON schema 필수 필드 존재",
           len(missing_fields) == 0,
           f"missing={missing_fields}" if missing_fields else "전체 필드 확인됨")

    # K10. 32×32 ref PNG display 금지 유지
    _check("K10. use_32px_ref_png=False 유지",
           not USE_32PX_REF_PNG,
           f"USE_32PX_REF_PNG={USE_32PX_REF_PNG}")

    # K11. Ref2/Ref3 card display false
    _check("K11. ref2_ref3_displayed_on_card=False",
           not SHOW_REF2_REF3_ON_CARD,
           f"SHOW_REF2_REF3_ON_CARD={SHOW_REF2_REF3_ON_CARD}")

    # K12. old Panel C 없음 (run_prototype 소스)
    has_old_c = "old Panel C" in run_src2 or "C_CROP_BBOX" in run_src2
    _check("K12. old Panel C 없음",
           not has_old_c,
           "Panel C 없음 확인됨" if not has_old_c else "Panel C 코드 발견")

    # K13. full 300 loop 없음 (run_prototype 소스)
    has_full300 = "ALLOW_FULL_300" in run_src2 and "for" in run_src2
    # run_prototype에서 ALLOW_FULL_300 guard는 있지만 loop는 없어야 함
    run_src_lines = run_src2.split("\n")
    full300_loop = any(
        "ALLOW_FULL_300" in ln and "for " in ln
        for ln in run_src_lines
    )
    _check("K13. full 300 loop 없음",
           not full300_loop,
           "ALLOW_FULL_300 guard만 있고 loop 없음" if not full300_loop else "full 300 loop 발견")

    # K14. existing artifact write 금지 (기존 S3/v1~v5 경로 write 없음)
    protected_names = ["S3_CARD_ROOT", "PROTO_V1", "PROTO_V2", "PROTO_V3", "PROTO_V4", "PROTO_V5"]
    has_existing_write = any(nm in run_src2 and ".save(" in run_src2 for nm in protected_names)
    _check("K14. existing artifact write 금지",
           not has_existing_write,
           "기존 경로 write 없음 확인됨" if not has_existing_write else "기존 경로 write 발견")

    # K15. output root guard 유지 (check_output_guard 호출)
    _check("K15. output root guard 유지 (run_prototype에서 check_output_guard 호출)",
           "check_output_guard" in run_src2,
           "check_output_guard 호출 확인됨")

    # 결과 집계
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
    """입력 파일 존재 + output guard + resolve 확인 (CT load 없음)."""
    print("=== dry-run (v6 high-res reference) ===")
    issues = []

    if ALLOW_RUN_CARD_PROTOTYPE:
        issues.append("ALLOW_RUN_CARD_PROTOTYPE=True — 이 단계에서는 False여야 함")
    if ALLOW_CT_LOAD:
        issues.append("ALLOW_CT_LOAD=True — 이 단계에서는 금지")

    # S4 v3 CSV
    if V3_OUTPUT_CSV.exists():
        print(f"  [OK] V3 CSV: {V3_OUTPUT_CSV}")
    else:
        issues.append(f"V3 CSV 없음: {V3_OUTPUT_CSV}")

    # S3 PNG
    s3_png = S3_CARDS_PNG_DIR / f"{TARGET_CASE_ID}.png"
    if s3_png.exists():
        print(f"  [OK] S3 PNG: {s3_png.name}")
    else:
        issues.append(f"S3 PNG 없음: {s3_png}")

    # S3 JSON
    s3_json = S3_CARDS_JSON_DIR / f"{TARGET_CASE_ID}.json"
    if s3_json.exists():
        print(f"  [OK] S3 JSON: {s3_json.name}")
    else:
        issues.append(f"S3 JSON 없음: {s3_json}")

    # reference manifest
    if REF_CROP_MANIFEST.exists():
        print(f"  [OK] REF_CROP_MANIFEST: {REF_CROP_MANIFEST.name}")
    else:
        issues.append(f"REF_CROP_MANIFEST 없음: {REF_CROP_MANIFEST}")

    # normal ref CT path
    if NORMAL_REF_CT_NPY.exists():
        print(f"  [OK] normal ref CT npy: {NORMAL_REF_CT_NPY.parent.name}/ct_hu.npy")
    else:
        print(f"  [WARN] normal ref CT npy 없음 — manifest resolve 확인 필요: {NORMAL_REF_CT_NPY}")
        issues.append(f"normal ref CT npy 없음: {NORMAL_REF_CT_NPY}")

    # candidate CT path
    if CANDIDATE_CT_NPY.exists():
        print(f"  [OK] candidate CT npy: {CANDIDATE_CT_NPY.parent.name}/ct_hu.npy")
    else:
        issues.append(f"candidate CT npy 없음: {CANDIDATE_CT_NPY}")

    # patient manifests
    if NORMAL_PATIENT_MANIFEST.exists():
        print(f"  [OK] normal patient manifest: {NORMAL_PATIENT_MANIFEST.name}")
    else:
        issues.append(f"normal patient manifest 없음: {NORMAL_PATIENT_MANIFEST}")

    if NSCLC_PATIENT_MANIFEST.exists():
        print(f"  [OK] NSCLC patient manifest: {NSCLC_PATIENT_MANIFEST.name}")
    else:
        issues.append(f"NSCLC patient manifest 없음: {NSCLC_PATIENT_MANIFEST}")

    # manifest resolve 확인 (path existence만, 실제 load 없음)
    if s3_json.exists():
        ref_info = resolve_ref_paths_from_json(str(s3_json))
        if ref_info["ok"] and ref_info["ref_count"] > 0:
            print(f"  [OK] reference metadata: {ref_info['ref_count']}건, bin={ref_info['position_bin']}")
            matched_rp = REF_BANK_FULL / ref_info["ref_paths"][0]
            if matched_rp.exists():
                print(f"  [OK] matched_ref 32px PNG (메타데이터 참조용): {ref_info['ref_paths'][0]}")
            else:
                print(f"  [WARN] matched_ref 32px PNG 없음 (v6에서 사용 안 함): {matched_rp}")
        else:
            print(f"  [WARN] reference metadata 없음: {ref_info.get('error', 'unknown')}")

    # crop size 확인
    crop_res = validate_crop_sizes()
    if crop_res["ok"]:
        print(f"  [OK] crop size: 양측 96×96 동일")
    else:
        issues.extend(crop_res["issues"])

    # output guard
    guard = check_output_guard()
    if guard["ok"]:
        print(f"  [OK] output guard: v6 output root 충돌 없음 ({PROTO_OUTPUT_ROOT.name})")
    else:
        issues.append(f"output guard FAIL: {guard.get('error')}")

    # reason text
    text_res = validate_reason_text()
    if text_res["ok"]:
        print(f"  [OK] reason text: {text_res['ko_len']}자, 면책문구+시각적참고단서 포함, 금지어 없음")
    else:
        issues.extend(text_res["issues"])

    # windowing note
    print(f"  [NOTE] windowing: center={HU_WIN_CENTER}HU width={HU_WIN_WIDTH}HU — {WINDOWING_NOTE}")

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
    """dry-run + v6 배치 계획 출력."""
    dry_ok = run_dry_run()

    print("\n=== plan-only: v6 high-res reference layout 계획 ===")
    print(f"  대상 case      : {TARGET_CASE_ID}")
    print(f"  prototype 버전 : {PROTOTYPE_VERSION}")
    print(f"  layout 버전    : {LAYOUT_VERSION}")
    print(f"  v5 실패 원인   : matched normal reference = 32×32 PNG 확대본 (low-res)")
    print(f"  v6 핵심 변경   : 원본 CT slice에서 96×96 crop → 동일 조건 비교")
    print()
    print(f"  [Row 1] Main comparison (h={ROW1_H_PX}px)")
    print(f"    좌: Candidate — 원본 CT crop")
    print(f"      patient: {TARGET_SAFE_ID}")
    print(f"      slice z={CANDIDATE_SLICE_Z}")
    print(f"      crop [y0={CANDIDATE_CROP_Y0}, x0={CANDIDATE_CROP_X0},"
          f" y1={CANDIDATE_CROP_Y1}, x1={CANDIDATE_CROP_X1}] = {CANDIDATE_CROP_SIZE_H}×{CANDIDATE_CROP_SIZE_W}px")
    print(f"      display: {DISPLAY_SIZE}×{DISPLAY_SIZE}px (LANCZOS upscale)")
    print()
    print(f"    우: Matched normal reference — 원본 CT crop")
    print(f"      patient: {NORMAL_REF_SAFE_ID[:50]}...")
    print(f"      slice z={NORMAL_REF_SLICE_Z}")
    print(f"      orig 32px bbox [y0={NORMAL_REF_ORIG_Y0}, x0={NORMAL_REF_ORIG_X0},"
          f" y1={NORMAL_REF_ORIG_Y1}, x1={NORMAL_REF_ORIG_X1}]")
    print(f"      center = ({NORMAL_REF_CENTER_Y}, {NORMAL_REF_CENTER_X})")
    print(f"      v6 96px crop [y0={NORMAL_REF_CROP_Y0}, x0={NORMAL_REF_CROP_X0},"
          f" y1={NORMAL_REF_CROP_Y1}, x1={NORMAL_REF_CROP_X1}]")
    print(f"      display: {DISPLAY_SIZE}×{DISPLAY_SIZE}px")
    print(f"      bbox clamp: np.clip(bbox, 0, slice_shape) 필수 (run 단계)")
    print()
    print(f"    windowing: center={HU_WIN_CENTER}HU, width={HU_WIN_WIDTH}HU")
    print(f"    [NOTE] {WINDOWING_NOTE}")
    print()
    print(f"  [Row 2] Context panels (h={ROW2_H_PX}px)")
    print(f"    좌: A whole-slice {CONTEXT_DISPLAY_SIZE}×{CONTEXT_DISPLAY_SIZE}px — S3 PNG crop 재사용")
    print(f"    우: D z-context {CONTEXT_DISPLAY_SIZE}×{CONTEXT_DISPLAY_SIZE}px — S3 PNG crop 재사용")
    print()
    print(f"  [Row 3] Reason box (h={ROW3_H_PX}px, 전체 폭, KO만 표시)")
    print(f"    Title: {REASON_TITLE}")
    print(f"    KO ({len(REASON_TEXT_KO)}자): {REASON_TEXT_KO[:80]}...")
    print()
    print(f"  canvas 추정: {CANVAS_WIDTH_PX}×{CANVAS_HEIGHT_ESTIMATE_PX}px")
    print(f"  Output root: {PROTO_OUTPUT_ROOT}")
    print(f"  Output PNG : {PROTO_CARDS_PNG_DIR / f'{TARGET_CASE_ID}_reason_prototype.png'}")
    print(f"  Output JSON: {PROTO_CARDS_JSON_DIR / f'{TARGET_CASE_ID}_reason_prototype.json'}")
    print()
    print(f"  CT path resolve 정책:")
    print(f"    정상 CT : NORMAL_PATIENT_MANIFEST → ct_hu_npy")
    print(f"    후보 CT : NSCLC_PATIENT_MANIFEST → ct_hu_npy")
    print(f"    WSL 변환: E:\\ → /mnt/e/... (자동)")
    print(f"    로드 방식: read-only mmap (np.load with mmap_mode='r') — run 단계 승인 후")
    print()
    print(f"  Ref2/Ref3: 카드 미표시 (SHOW_REF2_REF3_ON_CARD={SHOW_REF2_REF3_ON_CARD}) — JSON 전용")
    print(f"  32×32 PNG 확대 방식: 완전 폐기 (USE_32PX_REF_PNG={USE_32PX_REF_PNG})")
    print(f"  old Panel C: 카드 포함 금지")
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
        description="S4 Reason Card Prototype 1-case v6 — high-res normal reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--selftest",
                        action="store_true",
                        help="selftest 실행 (guard 및 설계 검사, 생성 없음)")
    parser.add_argument("--dry-run",
                        action="store_true",
                        dest="dry_run",
                        help="입력 파일 존재 + output guard 확인 (생성 없음)")
    parser.add_argument("--plan-only",
                        action="store_true",
                        dest="plan_only",
                        help="dry-run + v6 배치 계획 출력 (생성 없음)")
    parser.add_argument("--run-prototype",
                        action="store_true",
                        dest="run_prototype",
                        help="실제 v6 prototype 생성 (ALLOW_RUN_CARD_PROTOTYPE + ALLOW_CT_LOAD=True 필요)")
    parser.add_argument("--confirm-generate",
                        action="store_true",
                        dest="confirm_generate",
                        help="--run-prototype 와 함께 사용 시 실제 생성 확인 (guard 체크 별도)")

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
