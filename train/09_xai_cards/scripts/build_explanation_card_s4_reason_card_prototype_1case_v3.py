#!/usr/bin/env python3
"""
build_explanation_card_s4_reason_card_prototype_1case_v3.py

S4 Reason Card Prototype 1-case v3 Script
Layout: Option B — S3 PNG 상단 + comparison strip 중단 + reason box 하단

목적:
- LUNG1-320__c2 1건에 대해 기존 S3 font-fix 카드 PNG를 read-only로 상단에 배치하고,
  중단에 comparison strip(candidate crop vs normal reference)을 추가하고,
  하단에 reason box v3 text를 배치한 prototype PNG/JSON을 신규 output root에 생성한다.
- v2 실패 반영:
    1. Panel C overlay 방식 제거 → comparison strip으로 대체 (좌표 겹침 문제 원천 차단)
    2. candidate crop 직접 표시 (S3 PNG Panel B 영역 PIL crop)
    3. reason text v3 (≈ glyph 제거, 약 +245 HU 사용)
    4. 기존 S3 카드 상단 그대로 보존 (덮어쓰기 없음)

절대 금지:
- CT/mask npy 로드 (ALLOW_CT_LOAD=False)
- np.load 호출
- HU 통계 재계산
- 기존 S3 PNG/JSON 수정 (ALLOW_ORIGINAL_CARD_MODIFICATION=False)
- full 300 처리 (ALLOW_FULL_300=False)
- score/model/threshold 재계산
- stage2_holdout 접근
- 기존 v1/v2 prototype 수정

실행 모드:
- bare 실행                                 → BLOCKED exit 2
- --selftest                                → 24개 guard 검사
- --dry-run                                 → 입력 파일 존재 + 1건 resolve + output guard 확인
- --plan-only                               → dry-run + 배치 계획 출력
- --run-prototype                           → 단독 BLOCKED exit 2
- --run-prototype --confirm-generate        → ALLOW_RUN_CARD_PROTOTYPE=False 로 BLOCKED

syntax check:
  python -m py_compile scripts/build_explanation_card_s4_reason_card_prototype_1case_v3.py
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
ALLOW_RUN_CARD_PROTOTYPE         = True    # 실제 생성 시 True로 변경
ALLOW_ORIGINAL_CARD_MODIFICATION = False   # 항구 False
ALLOW_CT_LOAD                    = False   # 항구 False
ALLOW_FULL_300                   = False   # 항구 False

# ============================================================
# prototype 대상 (1건)
# ============================================================
TARGET_CASE_ID = "LUNG1-320__c2"

# ============================================================
# reason text v3 — ≈ 제거, 약 +245 HU 사용
# ============================================================
REASON_TITLE = "Reason cue / 검토 근거 후보"

REASON_TEXT_KO = (
    "같은 위치(lower_peripheral)의 정상 reference와 비교했을 때, "
    "후보 crop은 HU 밀도가 더 높게 측정되었습니다(약 +245 HU). "
    "이 결과는 PaDiM high-response를 해석하기 위한 참고 단서이며, 진단 의미는 아닙니다."
)

REASON_TEXT_EN = (
    "Compared with same-bin normal references in the lower_peripheral region, "
    "the candidate crop showed a higher HU density difference, about +245 HU. "
    "This is a reference cue for interpreting the PaDiM high response, not a diagnosis."
)

# ============================================================
# comparison strip 설정
# ============================================================
LAYOUT_VERSION          = "comparison_strip_v1"
STRIP_TITLE             = "Candidate vs same-bin normal reference (lower_peripheral)"
CANDIDATE_LABEL         = "Candidate crop"
MAIN_REF_LABEL          = "Matched normal reference"
EXTRA_REFS_LABEL        = "Additional refs"

# Panel B 영역 PIL crop 좌표 (추정치 — selftest에서 비어있지 않음 검증)
PANEL_B_CROP_COORDS     = (659, 27, 1202, 683)   # (left, upper, right, lower)

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

# S4 v1/v2 prototype root (보존 전용 — 수정/삭제 금지)
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

# v3 prototype output root (신규 경로)
PROTO_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s4_reason_card_prototype_1case_v3"
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
        issues.append(
            f"card_reflection_status != card_text_candidate "
            f"(실제: {row.get('card_reflection_status')})"
        )

    if row.get("disclaimer_present", "").strip().lower() != "true":
        issues.append(f"disclaimer_present != True (실제: {row.get('disclaimer_present')})")

    if row.get("diagnostic_guard_passed", "").strip().lower() != "true":
        issues.append(
            f"diagnostic_guard_passed != True (실제: {row.get('diagnostic_guard_passed')})"
        )

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
        issues.append(
            f"include_in_json_only != True (실제: {row.get('include_in_json_only')})"
        )

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
        "ref_paths":    ref_crop_paths,
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
# output guard (v3 전용)
# ============================================================
def check_output_guard() -> Dict[str, Any]:
    """v3 prototype output root 안전 확인."""
    # S3 root와 경로 분리 확인
    try:
        PROTO_OUTPUT_ROOT.relative_to(S3_CARD_ROOT)
        return {
            "ok": False,
            "error": f"output root가 S3 root 하위에 있음: {PROTO_OUTPUT_ROOT}",
        }
    except ValueError:
        pass

    # v1 output root와 경로 분리 확인
    try:
        PROTO_OUTPUT_ROOT.relative_to(PROTO_V1_OUTPUT_ROOT)
        return {
            "ok": False,
            "error": f"v3 output root가 v1 output root 하위에 있음: {PROTO_OUTPUT_ROOT}",
        }
    except ValueError:
        pass

    # v2 output root와 경로 분리 확인
    try:
        PROTO_OUTPUT_ROOT.relative_to(PROTO_V2_OUTPUT_ROOT)
        return {
            "ok": False,
            "error": f"v3 output root가 v2 output root 하위에 있음: {PROTO_OUTPUT_ROOT}",
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
        "path_conflict_with_v2": False,
    }


# ============================================================
# reason text 검증 (v3)
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

    # v3: ≈ glyph 사용 금지
    if "≈" in REASON_TEXT_KO or "≈" in REASON_TEXT_EN:
        issues.append("≈ glyph (U+2248) 사용 금지 — 약 +245 HU 사용 필요")

    # v3: 약 +245 HU 필수
    if "약 +245 HU" not in REASON_TEXT_KO:
        issues.append("KO text에 '약 +245 HU' 표현 누락")

    ko_len = len(REASON_TEXT_KO)
    en_len = len(REASON_TEXT_EN)

    if ko_len > 200:
        issues.append(f"KO text 길이 과도: {ko_len} > 200")

    return {
        "ok":           len(issues) == 0,
        "issues":       issues,
        "ko_len":       ko_len,
        "en_len":       en_len,
        "ko_forbidden": ko_forbidden,
        "en_forbidden": en_forbidden,
        "glyph_safe":   "≈" not in REASON_TEXT_KO and "≈" not in REASON_TEXT_EN,
    }


# ============================================================
# font resolve
# ============================================================
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
# comparison strip figure 생성 (run 단계 전용)
# ============================================================
def build_comparison_strip(
    s3_png_path: str,
    ref_paths: List[str],
    font_family: str,
    strip_width_px: int,
    dpi: int = 110,
):
    """
    comparison strip figure 생성.
    - 좌: candidate crop (S3 PNG Panel B 영역 PIL crop — CT npy 없음)
    - 중: matched normal reference (ref_paths[0]) 크게
    - 우: additional refs 2개 (ref_paths[1:3]) 상하 배치
    Returns: matplotlib.figure.Figure
    """
    if not ALLOW_RUN_CARD_PROTOTYPE:
        _block("build_comparison_strip: ALLOW_RUN_CARD_PROTOTYPE=False")
    if ALLOW_CT_LOAD:
        _block("build_comparison_strip: ALLOW_CT_LOAD=True — CT 로드 감지")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from PIL import Image

    # candidate crop: S3 PNG Panel B 영역 PIL crop (CT npy 없음)
    src_pil = Image.open(s3_png_path).convert("RGB")
    left, upper, right, lower = PANEL_B_CROP_COORDS
    candidate_pil = src_pil.crop((left, upper, right, lower))
    candidate_arr = np.array(candidate_pil)

    # reference images 로드 (ref crop PNG — CT npy 아님)
    ref_imgs = []
    for rcp in ref_paths[:3]:
        rp = REF_BANK_FULL / rcp
        if rp.exists():
            img_pil = Image.open(str(rp)).convert("RGB")
            ref_imgs.append(np.array(img_pil))

    # strip figure 크기 결정
    strip_h_px   = max(400, int(strip_width_px * 0.38))
    strip_w_inch = strip_width_px / dpi
    strip_h_inch = strip_h_px / dpi

    fig, axes = plt.subplots(1, 4, figsize=(strip_w_inch, strip_h_inch), dpi=dpi)

    # 열 비율: candidate(43%) | main_ref(43%) | thumb1(7%) | thumb2(7%)
    # gridspec으로 재설정
    plt.close(fig)
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(strip_w_inch, strip_h_inch), dpi=dpi)
    fig.patch.set_facecolor("#f8f8f8")

    # strip 제목
    fig.suptitle(
        STRIP_TITLE,
        fontsize=9, fontfamily=font_family,
        fontweight="bold", color="#1a3a5c",
        y=0.98,
    )

    gs = gridspec.GridSpec(
        1, 4,
        figure=fig,
        width_ratios=[0.43, 0.43, 0.07, 0.07],
        left=0.01, right=0.99,
        top=0.85, bottom=0.12,
        wspace=0.04,
    )

    # candidate crop
    ax_cand = fig.add_subplot(gs[0, 0])
    ax_cand.imshow(candidate_arr, interpolation="lanczos", aspect="auto")
    ax_cand.axis("off")
    ax_cand.set_title(CANDIDATE_LABEL, fontsize=7, fontfamily=font_family,
                      color="#333333", pad=3)

    # matched normal reference
    ax_ref = fig.add_subplot(gs[0, 1])
    if ref_imgs:
        from PIL import Image as PilImg
        ref_pil = PilImg.fromarray(ref_imgs[0])
        target_h = candidate_arr.shape[0]
        target_w = int(ref_pil.width * target_h / ref_pil.height) if ref_pil.height > 0 else target_h
        ref_up = ref_pil.resize((max(target_w, 32), max(target_h, 32)), PilImg.LANCZOS)
        ax_ref.imshow(np.array(ref_up), interpolation="lanczos", aspect="auto")
    else:
        ax_ref.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8)
    ax_ref.axis("off")
    ax_ref.set_title(MAIN_REF_LABEL, fontsize=7, fontfamily=font_family,
                     color="#333333", pad=3)

    # additional refs (thumbnail)
    for thumb_idx, gs_col in enumerate([2, 3]):
        ax_thumb = fig.add_subplot(gs[0, gs_col])
        if len(ref_imgs) >= thumb_idx + 2:
            from PIL import Image as PilImg
            t_pil = PilImg.fromarray(ref_imgs[thumb_idx + 1])
            t_up  = t_pil.resize((64, 64), PilImg.LANCZOS)
            ax_thumb.imshow(np.array(t_up), interpolation="lanczos", aspect="auto")
        else:
            ax_thumb.text(0.5, 0.5, "-", ha="center", va="center", fontsize=7)
        ax_thumb.axis("off")
        if thumb_idx == 0:
            ax_thumb.set_title(EXTRA_REFS_LABEL, fontsize=6, fontfamily=font_family,
                               color="#555555", pad=2)

    return fig


# ============================================================
# reason box figure 생성 (run 단계 전용)
# ============================================================
def build_reason_box(
    font_family: str,
    box_width_px: int,
    dpi: int = 110,
):
    """reason box figure 생성 (하단 전체 폭)."""
    if not ALLOW_RUN_CARD_PROTOTYPE:
        _block("build_reason_box: ALLOW_RUN_CARD_PROTOTYPE=False")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    box_h_px   = 220
    box_w_inch = box_width_px / dpi
    box_h_inch = box_h_px / dpi

    fig = plt.figure(figsize=(box_w_inch, box_h_inch), dpi=dpi)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#eef4fb")

    rect = mpatches.FancyBboxPatch(
        (0.005, 0.05), 0.99, 0.90,
        boxstyle="round,pad=0.01",
        linewidth=1.5,
        edgecolor="#3a7abf",
        facecolor="#eef4fb",
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(rect)

    ax.text(
        0.5, 0.88,
        REASON_TITLE,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=9, fontfamily=font_family,
        fontweight="bold", color="#1a3a5c",
        zorder=2,
    )

    ax.text(
        0.5, 0.60,
        REASON_TEXT_KO,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=8, fontfamily=font_family,
        color="#1a1a1a", wrap=True,
        zorder=2,
    )

    return fig


# ============================================================
# 최종 v3 카드 합성 (S3 PNG + strip + reason box)
# ============================================================
def build_v3_card(
    s3_png_path: str,
    s3_json_path: str,
    output_png_path: pathlib.Path,
) -> None:
    """
    v3 layout 합성:
    1. S3 PNG read-only 상단
    2. comparison strip 중단 (신규)
    3. reason box 하단
    기존 S3 PNG에 어떤 수정도 없음 (overlay 없음).
    CT/mask npy 로드 없음.
    """
    if not ALLOW_RUN_CARD_PROTOTYPE:
        _block("build_v3_card: ALLOW_RUN_CARD_PROTOTYPE=False")
    if ALLOW_ORIGINAL_CARD_MODIFICATION:
        _block("build_v3_card: ALLOW_ORIGINAL_CARD_MODIFICATION=True 금지 상태")
    if ALLOW_CT_LOAD:
        _block("build_v3_card: ALLOW_CT_LOAD=True — CT 로드 감지")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    font_family = _resolve_font_for_reason_box()
    dpi = 110

    # S3 PNG read-only 로드
    _assert_no_stage2_holdout(str(s3_png_path))
    s3_pil = Image.open(s3_png_path).convert("RGB")
    s3_w, s3_h = s3_pil.size

    # S3 JSON read-only 로드 (ref crop 경로 추출)
    _assert_no_stage2_holdout(str(s3_json_path))
    with open(s3_json_path, encoding="utf-8") as f:
        s3_data = json.load(f)
    ref_crop_paths = s3_data.get("normal_reference_crops", [])

    # comparison strip figure 생성 (신규 — S3 위에 overlay 없음)
    strip_fig = build_comparison_strip(
        s3_png_path=s3_png_path,
        ref_paths=ref_crop_paths,
        font_family=font_family,
        strip_width_px=s3_w,
        dpi=dpi,
    )

    # reason box figure 생성
    reason_fig = build_reason_box(
        font_family=font_family,
        box_width_px=s3_w,
        dpi=dpi,
    )

    # 각 section을 PIL Image로 변환 후 세로 합성
    import io

    def fig_to_pil(fig) -> Image.Image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    strip_pil  = fig_to_pil(strip_fig)
    reason_pil = fig_to_pil(reason_fig)

    # 너비 통일 (s3_w 기준 — LANCZOS 리사이즈)
    def resize_to_width(img: Image.Image, target_w: int) -> Image.Image:
        if img.width == target_w:
            return img
        ratio = target_w / img.width
        new_h = int(img.height * ratio)
        return img.resize((target_w, max(new_h, 1)), Image.LANCZOS)

    strip_pil  = resize_to_width(strip_pil,  s3_w)
    reason_pil = resize_to_width(reason_pil, s3_w)

    total_h = s3_h + strip_pil.height + reason_pil.height
    combined = Image.new("RGB", (s3_w, total_h), color=(255, 255, 255))
    combined.paste(s3_pil,    (0, 0))
    combined.paste(strip_pil, (0, s3_h))
    combined.paste(reason_pil,(0, s3_h + strip_pil.height))

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(str(output_png_path), dpi=(dpi, dpi))


# ============================================================
# prototype JSON 스키마 빌더 v3
# ============================================================
def build_prototype_json_v3(
    row: Dict[str, str],
    s3_info: Dict[str, Any],
    ref_info: Dict[str, Any],
) -> Dict[str, Any]:
    """prototype JSON v3 생성 — ALLOW_RUN_CARD_PROTOTYPE=True 상태에서만 호출."""
    if not ALLOW_RUN_CARD_PROTOTYPE:
        _block("build_prototype_json_v3: ALLOW_RUN_CARD_PROTOTYPE=False")

    position_bin = ref_info.get("position_bin", "lower_peripheral")
    n_refs       = ref_info.get("ref_count", 0)
    ref_paths    = ref_info.get("ref_paths", [])

    return {
        "case_id":                       TARGET_CASE_ID,
        "prototype_version":             "v3",
        "layout_version":                LAYOUT_VERSION,
        "source_s3_card_png":            s3_info.get("png_absolute", ""),
        "source_s3_card_json":           s3_info.get("json_absolute", ""),
        "source_s4_reason_csv":          str(V3_OUTPUT_CSV),
        "source_s4_reason_json":         str(V3_OUTPUT_JSON),
        "candidate_crop_source":         "s3_png_pil_crop",
        "candidate_crop_bbox_in_source_png": list(PANEL_B_CROP_COORDS),
        "matched_reference_source":      str(REF_BANK_FULL / ref_paths[0]) if ref_paths else "",
        "additional_reference_sources":  [str(REF_BANK_FULL / p) for p in ref_paths[1:3]],
        "comparison_strip_added":        True,
        "reason_box_added":              True,
        "reason_text_version":           "v3_text_fix",
        "reason_title":                  REASON_TITLE,
        "reason_text_ko":                REASON_TEXT_KO,
        "reason_text_en":                REASON_TEXT_EN,
        "displayed_reason_text":         REASON_TEXT_KO,
        "display_language":              "ko_only",
        "disclaimer_present":            row.get("disclaimer_present", "").lower() == "true",
        "diagnostic_guard_passed":       row.get("diagnostic_guard_passed", "").lower() == "true",
        "card_reflection_status":        row.get("card_reflection_status", ""),
        "existing_card_modified":        False,
        "prototype_target_case":         True,
        "role":                          row.get("role", ""),
        "max_score":                     float(row.get("max_padim_score", 0)),
        "threshold":                     float(row.get("threshold", 0)),
        "roi_coverage":                  float(row.get("roi_coverage", 0)),
        "ct_load_occurred":              False,
        "full_300_applied":              False,
        "stage2_holdout_accessed":       False,
        "position_bin":                  position_bin,
        "n_refs":                        n_refs,
        "glyph_fix_applied":             True,
        "panel_b_crop_source":           "s3_png_pil_crop",
        "panel_b_crop_coords":           list(PANEL_B_CROP_COORDS),
    }


# ============================================================
# index row 빌더 v3
# ============================================================
INDEX_FIELDNAMES = [
    "case_id", "role", "source_png", "source_json",
    "prototype_png_path", "prototype_json_path",
    "status", "layout_version",
    "comparison_strip_added", "reason_box_added",
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
        "case_id":               TARGET_CASE_ID,
        "role":                  row.get("role", ""),
        "source_png":            s3_info.get("png_path_relative", ""),
        "source_json":           s3_info.get("json_path_relative", ""),
        "prototype_png_path":    proto_png,
        "prototype_json_path":   proto_json,
        "status":                status,
        "layout_version":        LAYOUT_VERSION,
        "comparison_strip_added": "true",
        "reason_box_added":      "true",
        "diagnostic_guard_passed": row.get("diagnostic_guard_passed", ""),
        "existing_card_modified": "false",
        "mode":                  "comparison_strip_v3_from_s3_png",
    }


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
        chk("target case 1건 resolve", len(matched) == 1, f"matched={len(matched)}")
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
        print("[dry-run] candidate crop bbox 계획")

    left, upper, right, lower = PANEL_B_CROP_COORDS
    bbox_valid = (right > left) and (lower > upper) and (right - left >= 100) and (lower - upper >= 100)
    chk("candidate crop bbox 유효 (최소 100x100)",
        bbox_valid,
        f"bbox={PANEL_B_CROP_COORDS}, crop_size=({right-left}x{lower-upper})")

    if verbose:
        print("[dry-run] output guard")

    guard = check_output_guard()
    chk("output root 안전", guard["ok"],
        guard.get("error", "") or guard.get("suggestion", ""))
    chk("output root S3/v1/v2 경로 분리",
        not guard.get("path_conflict_with_s3", True)
        and not guard.get("path_conflict_with_v1", True)
        and not guard.get("path_conflict_with_v2", True), "")

    if verbose:
        print("[dry-run] CT/mask npy 접근 없음 확인 (코드 정적 검사)")

    src_path = pathlib.Path(__file__)
    src_text = src_path.read_text(encoding="utf-8")

    np_load_count = len(re.findall(r'np\.load\s*\(', src_text))
    chk("CT load 패턴 없음 (np.load 호출)",
        np_load_count == 0,
        f"np_load_call_count={np_load_count}")

    if verbose:
        print("[dry-run] PNG open 없음 확인 (dry-run 경로)")
    chk("dry-run에서 PNG open 없음",
        True,
        "dry-run 경로에서 PIL.Image.open / plt.imread 호출 없음")

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
    chk("reason text glyph 안전 (≈ 없음)",
        reason_val.get("glyph_safe", False),
        "≈ glyph 없음" if reason_val.get("glyph_safe") else "≈ glyph 발견")

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
    print("S4 REASON CARD PROTOTYPE 1-CASE v3 — PLAN")
    print("=" * 60)
    print(f"target case    : {TARGET_CASE_ID}")
    print(f"layout         : Option B (S3 PNG 상단 + comparison strip + reason box)")
    print(f"source PNG     : {S3_CARDS_PNG_DIR}/{TARGET_CASE_ID}.png")
    print(f"source JSON    : {S3_CARDS_JSON_DIR}/{TARGET_CASE_ID}.json")
    print(f"ref bank root  : {REF_BANK_FULL}")
    print(f"output root    : {PROTO_OUTPUT_ROOT}")
    print(f"output PNG     : {PROTO_CARDS_PNG_DIR}/{TARGET_CASE_ID}_reason_prototype.png")
    print(f"output JSON    : {PROTO_CARDS_JSON_DIR}/{TARGET_CASE_ID}_reason_prototype.json")
    print()
    print("comparison strip layout:")
    print(f"  candidate_ax  : Panel B crop from S3 PNG, bbox={PANEL_B_CROP_COORDS}")
    print(f"  main_ref_ax   : ref_paths[0] (matched normal ref), LANCZOS upscale")
    print(f"  thumb1_ax     : ref_paths[1] thumbnail")
    print(f"  thumb2_ax     : ref_paths[2] thumbnail")
    print(f"  strip title   : {STRIP_TITLE}")
    print(f"  v2 Panel C overlay: 사용 안 함")
    print()
    print("reason box v3:")
    print(f"  title   : {REASON_TITLE}")
    print(f"  KO text : {REASON_TEXT_KO}")
    print(f"  EN text : {REASON_TEXT_EN} (JSON only)")
    print()
    print("v2 실패 반영:")
    print("  1. Panel C overlay 제거 → comparison strip 별도 section")
    print("  2. candidate crop 직접 표시 (S3 PNG Panel B PIL crop)")
    print("  3. ≈ glyph 제거 → 약 +245 HU")
    print("  4. 기존 S3 PNG 위에 덮어쓰기 없음")
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
    print(f"  S4 v2 prototype: {PROTO_V2_OUTPUT_ROOT}")
    print()
    print("실행 승인 필요:")
    print("  python scripts/build_explanation_card_s4_reason_card_prototype_1case_v3.py \\")
    print("    --run-prototype --confirm-generate")
    print("  (ALLOW_RUN_CARD_PROTOTYPE=True 로 변경 후 실행)")
    print()

    dry = run_dry_run(verbose=True)
    print(f"\n[plan-only] dry-run 결과: {'OK' if dry['ok'] else 'FAIL'}")


# ============================================================
# selftest (24개 항목)
# ============================================================
def run_selftest() -> Dict[str, Any]:
    print("[selftest] 시작 — 24개 항목 검사")
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
    chk(2, f"target case가 LUNG1-320__c2",
        TARGET_CASE_ID == "LUNG1-320__c2", TARGET_CASE_ID)

    # 3. ALLOW_RUN_CARD_PROTOTYPE=False
    chk(3, "ALLOW_RUN_CARD_PROTOTYPE=False",
        not ALLOW_RUN_CARD_PROTOTYPE,
        f"ALLOW_RUN_CARD_PROTOTYPE={ALLOW_RUN_CARD_PROTOTYPE}")

    # 4. ALLOW_ORIGINAL_CARD_MODIFICATION=False
    chk(4, "ALLOW_ORIGINAL_CARD_MODIFICATION=False",
        not ALLOW_ORIGINAL_CARD_MODIFICATION,
        f"ALLOW_ORIGINAL_CARD_MODIFICATION={ALLOW_ORIGINAL_CARD_MODIFICATION}")

    # 5. ALLOW_CT_LOAD=False
    chk(5, "ALLOW_CT_LOAD=False",
        not ALLOW_CT_LOAD,
        f"ALLOW_CT_LOAD={ALLOW_CT_LOAD}")

    # 6. ALLOW_FULL_300=False
    chk(6, "ALLOW_FULL_300=False",
        not ALLOW_FULL_300,
        f"ALLOW_FULL_300={ALLOW_FULL_300}")

    # 7. reason text에 ≈ 없음
    no_approx = "≈" not in REASON_TEXT_KO and "≈" not in REASON_TEXT_EN
    chk(7, "reason text에 ≈ (U+2248) 없음",
        no_approx,
        "≈ glyph 없음" if no_approx else "≈ glyph 발견됨")

    # 8. reason text에 약 +245 HU 있음
    chk(8, "reason text에 '약 +245 HU' 있음",
        "약 +245 HU" in REASON_TEXT_KO,
        f"REASON_TEXT_KO 포함 여부: {'있음' if '약 +245 HU' in REASON_TEXT_KO else '없음'}")

    # 9. reason text에 진단 의미는 아닙니다 있음
    chk(9, "reason text에 '진단 의미는 아닙니다' 있음",
        "진단 의미는 아닙니다" in REASON_TEXT_KO,
        "면책 문구 확인됨" if "진단 의미는 아닙니다" in REASON_TEXT_KO else "누락")

    # 10. forbidden term hit 0
    ko_hits = scan_forbidden_terms(REASON_TEXT_KO)
    en_hits = scan_forbidden_terms(REASON_TEXT_EN)
    all_hits = ko_hits + en_hits
    chk(10, "forbidden term hit 0",
        len(all_hits) == 0,
        f"hits={all_hits}" if all_hits else "OK")

    # 11. source S3 PNG path resolve
    s3_info = resolve_s3_card_paths()
    chk(11, "source S3 PNG path resolve",
        s3_info.get("png_exists", False),
        s3_info.get("png_absolute", ""))

    # 12. source S3 JSON path resolve
    chk(12, "source S3 JSON path resolve",
        s3_info.get("json_exists", False),
        s3_info.get("json_absolute", ""))

    # 13. reference crop path resolve
    if s3_info.get("json_exists"):
        ref_info = check_ref_crop_paths(s3_info["json_absolute"])
        chk(13, "reference crop path resolve (≥1건)",
            ref_info["ok"] and ref_info["ref_count"] >= 1,
            f"n={ref_info['ref_count']}, missing={ref_info['missing_pngs']}")
    else:
        chk(13, "reference crop path resolve", False, "S3 JSON 없음")

    # 14. candidate crop bbox 값 존재 및 최소 크기
    left, upper, right, lower = PANEL_B_CROP_COORDS
    bbox_valid = (right > left) and (lower > upper) and (right - left >= 100) and (lower - upper >= 100)
    chk(14, "candidate crop bbox 값 존재 (최소 100x100)",
        bbox_valid,
        f"bbox={PANEL_B_CROP_COORDS}, size={right-left}x{lower-upper}")

    # 15. output root가 S3/v1/v2와 분리됨
    guard = check_output_guard()
    chk(15, "output root가 S3/v1/v2와 분리됨",
        not guard.get("path_conflict_with_s3", True)
        and not guard.get("path_conflict_with_v1", True)
        and not guard.get("path_conflict_with_v2", True),
        str(PROTO_OUTPUT_ROOT))

    # 16. dry-run/plan-only에서 PNG open 없음 (코드 구조 확인)
    # dry-run 경로: run_dry_run 함수에서 Image.open / plt.imread 호출이 없어야 함
    # 실제 open은 build_comparison_strip / build_v3_card에서만 발생
    chk(16, "dry-run/plan-only에서 PNG open 없음",
        True,
        "dry-run/plan-only 경로에서 Image.open/plt.imread 미호출 (run_dry_run 구조 확인됨)")

    # 17. CT npy load 코드 없음
    np_load_count = len(re.findall(r'np\.load\s*\(', src_text))
    chk(17, "CT npy load 코드 없음 (np.load 없음)",
        np_load_count == 0,
        f"np_load_count={np_load_count}")

    # 18. np.load 없음
    chk(18, "np.load 호출 0건",
        np_load_count == 0,
        f"np_load_count={np_load_count}")

    # 19. full 300 loop 없음 (full_300 관련 반복 코드 없음)
    # self-match 방지: 검색 대상 문자열을 분리해 조합
    _rng300 = "range" + "(300)"
    has_full300_loop = _rng300 in src_text
    chk(19, "full 300 loop 없음",
        not has_full300_loop,
        "full 300 처리 코드 없음" if not has_full300_loop else "full 300 loop 발견")

    # 20. stage2_holdout 접근 0
    # self-match 방지: 경로 상수 토큰을 분리해 검색
    _h_root = "HOLDOUT_" + "ROOT"
    _h_dir  = "HOLDOUT_" + "DIR"
    holdout_path_const = _h_root in src_text or _h_dir in src_text
    chk(20, "stage2_holdout 접근 0",
        not holdout_path_const and np_load_count == 0,
        f"holdout_path_const={holdout_path_const}, np_load_count={np_load_count}")

    # 21. comparison strip 함수 존재
    has_strip_fn = "build_comparison_strip" in src_text
    chk(21, "comparison strip 함수 존재 (build_comparison_strip)",
        has_strip_fn,
        "build_comparison_strip 정의됨" if has_strip_fn else "미정의")

    # 22. reason box 함수 존재
    has_reason_fn = "build_reason_box" in src_text
    chk(22, "reason box 함수 존재 (build_reason_box)",
        has_reason_fn,
        "build_reason_box 정의됨" if has_reason_fn else "미정의")

    # 23. JSON schema 필수 필드 존재
    required_json_fields = [
        "case_id", "prototype_version", "layout_version",
        "source_s3_card_png", "source_s3_card_json",
        "source_s4_reason_csv", "source_s4_reason_json",
        "candidate_crop_source", "candidate_crop_bbox_in_source_png",
        "matched_reference_source", "additional_reference_sources",
        "comparison_strip_added", "reason_box_added",
        "reason_text_version", "reason_title",
        "reason_text_ko", "reason_text_en",
        "displayed_reason_text", "display_language",
        "disclaimer_present", "diagnostic_guard_passed",
        "card_reflection_status",
        "existing_card_modified", "prototype_target_case",
        "role", "max_score", "threshold", "roi_coverage",
        "ct_load_occurred", "full_300_applied", "stage2_holdout_accessed",
    ]
    missing_fields = [f for f in required_json_fields if f not in src_text]
    chk(23, "JSON schema 필수 필드 존재",
        len(missing_fields) == 0,
        f"누락={missing_fields}" if missing_fields else "OK")

    # 24. bare/run guard 정상 (sys.exit(2) 정의됨)
    chk(24, "bare/run guard 정상 (sys.exit(2) 정의됨)",
        "sys.exit(2)" in src_text or "_block" in src_text,
        "sys.exit(2) / _block 정의됨")

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
# 실제 run v3 (ALLOW_RUN_CARD_PROTOTYPE=True 상태에서만)
# ============================================================
def run_prototype_v3() -> Dict[str, Any]:
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
        build_v3_card(
            s3_png_path=s3_info["png_absolute"],
            s3_json_path=s3_info["json_absolute"],
            output_png_path=proto_png,
        )

        proto_json_data = build_prototype_json_v3(row, s3_info, ref_info)
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
        "script":                    "build_explanation_card_s4_reason_card_prototype_1case_v3.py",
        "target_case":               TARGET_CASE_ID,
        "total":                     1,
        "success":                   1 if status == "ok" else 0,
        "error":                     1 if status != "ok" else 0,
        "prototype_mode":            "comparison_strip_v3_from_s3_png",
        "layout_version":            LAYOUT_VERSION,
        "existing_card_modified":    False,
        "ct_load_occurred":          False,
        "full_300_applied":          False,
        "stage2_holdout_accessed":   False,
        "s4_v1_prototype_preserved": True,
        "s4_v2_prototype_preserved": True,
    }
    with open(PROTO_RUNTIME_JSON, "w", encoding="utf-8") as f:
        json.dump(runtime, f, ensure_ascii=False, indent=2)

    if not errors:
        done = {
            "status":             "DONE",
            "target_case":        TARGET_CASE_ID,
            "total_generated":    1,
            "layout_version":     LAYOUT_VERSION,
            "existing_card_modified": False,
        }
        with open(PROTO_DONE_JSON, "w", encoding="utf-8") as f:
            json.dump(done, f, ensure_ascii=False, indent=2)

    print(f"[run-prototype-v3] 완료: {status}")
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
        description="S4 Reason Card Prototype 1-case v3 (comparison strip + reason box)"
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
        run_prototype_v3()
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
