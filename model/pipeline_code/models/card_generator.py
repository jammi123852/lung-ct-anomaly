"""
card_generator.py

/analyze_volume 결과를 받아서 dynamic reference card 데이터 생성.
- Panel 1: 후보 crop 이미지 + 정상 3명 reference
- Panel 2: 3x3 score grid
- Panel 3: 위치/점수 요약
- Panel 4: 임상 판독 보조 텍스트 (clinical_readable_v1 기준)
- Panel 5: NSCLC 보조 분류기 결과 (P-C-NORMAL24j)
"""
import os
import csv
import math
import base64
from io import BytesIO
from PIL import Image

BANK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reference_bank",
    "dynamic_normal_reference_bank_three_patients_v1"
)
BANK_INDEX = os.path.join(BANK_DIR, "dynamic_reference_slice_index.csv")

_ref_rows = None


def _load_ref_index():
    global _ref_rows
    if _ref_rows is not None:
        return _ref_rows
    rows = []
    if not os.path.exists(BANK_INDEX):
        _ref_rows = []
        return _ref_rows
    with open(BANK_INDEX, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    _ref_rows = rows
    return rows


def _distance(lung_z_pct, y_pct, x_pct, side, row):
    rz = float(row["lung_z_pct"])
    by0 = float(row["lung_bbox_y0"]); by1 = float(row["lung_bbox_y1"])
    bx0 = float(row["lung_bbox_x0"]); bx1 = float(row["lung_bbox_x1"])
    rcy = float(row["lung_center_y"]); rcx = float(row["lung_center_x"])
    ry = (rcy - by0) / max(by1 - by0, 1)
    rx = (rcx - bx0) / max(bx1 - bx0, 1)
    return math.sqrt((lung_z_pct - rz) ** 2 + (y_pct - ry) ** 2 + (x_pct - rx) ** 2)


def select_normal_refs(lung_z_pct: float, y_pct: float, x_pct: float, side: str):
    rows = _load_ref_index()
    result = []
    for alias in ["normal_patient_1", "normal_patient_2", "normal_patient_3"]:
        cands = [r for r in rows if r["patient_alias"] == alias
                 and r["valid_lung_slice"] == "True"
                 and r["slice_quality"] != "low"]
        if not cands:
            cands = [r for r in rows if r["patient_alias"] == alias
                     and r["valid_lung_slice"] == "True"]
        if not cands:
            continue
        best = min(cands, key=lambda r: _distance(lung_z_pct, y_pct, x_pct, side, r))
        result.append({
            "alias": alias,
            "row": best,
            "distance": round(_distance(lung_z_pct, y_pct, x_pct, side, best), 6),
        })
    return result


def _img_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Panel 4 임상 판독 보조 텍스트 헬퍼 ────────────────────────────────────────

def _score_interpretation(rd4ad_score: float) -> dict:
    if rd4ad_score >= 0.45:
        return {
            "title": f"뚜렷한 이상 패턴 / Distinct anomaly pattern (score {rd4ad_score:.4f})",
            "body":  "정상 대비 재구성 오차가 높은 편입니다. 추가 판독 확인 권장.",
        }
    elif rd4ad_score >= 0.35:
        return {
            "title": f"경계 수준 이상 패턴 / Borderline anomaly (score {rd4ad_score:.4f})",
            "body":  "정상 대비 다소 이질적인 패턴입니다. 단독 판정 근거로는 불충분.",
        }
    else:
        return {
            "title": f"낮은 수준 이상 패턴 / Low-level anomaly (score {rd4ad_score:.4f})",
            "body":  "정상 범주 경계 수준입니다. 추가 context 확인 권장.",
        }


_POSITION_CONTEXT = {
    "upper_peripheral": "상부 말초 영역으로, 흉막 인접 구조·말초 혈관·섬유화/반흔 변화와 함께 해석해야 합니다.",
    "lower_peripheral": "하부 말초 영역으로, 횡격막·흉막·기저부 구조에 의한 오탐 가능성을 주의하세요.",
    "lower_central":    "하부 중심부로, 폐문·혈관 인접 구조와의 겹침 가능성을 확인해야 합니다.",
    "upper_central":    "상부 중심부로, 상부 혈관·기관지 주변 구조와의 겹침 가능성을 고려해야 합니다.",
    "middle_peripheral":"중부 말초 영역으로, 흉막 인접 구조와 주변 혈관에 주의하세요.",
    "middle_central":   "중부 중심부로, 폐문·혈관·기관지 주변 구조와의 겹침 가능성을 고려해야 합니다.",
}


def _zpct_warning(lung_z_pct: float) -> str:
    if lung_z_pct is None:
        return "폐 z 위치 정보 없음 (unknown)."
    if lung_z_pct < 0.08:
        return f"⚠ 폐저부/횡격막 인접 (z_pct={lung_z_pct:.3f}) — 오탐 가능성 높음."
    elif lung_z_pct > 0.92:
        return f"⚠ 폐첨부 인접 (z_pct={lung_z_pct:.3f}) — 경계 오탐 가능성 있음."
    else:
        return f"폐 중간 구역 (z_pct={lung_z_pct:.3f}) — 위치 기반 오탐 가능성 낮음."


def _build_panel4(position_bin: str, rd4ad_score: float, lung_z_pct: float) -> dict:
    interp = _score_interpretation(rd4ad_score)
    pos_ctx = _POSITION_CONTEXT.get(
        position_bin,
        f"{position_bin} 영역 — 해당 구역 임상 맥락 확인 필요."
    )
    return {
        "key_finding": {
            "title": interp["title"],
            "body":  interp["body"],
            "context": "",
        },
        "location_context": {
            "position":  pos_ctx,
            "z_warning": _zpct_warning(lung_z_pct),
        },
        "disclaimer": (
            "연구용 보조 설명 | 진단 아님 | NSCLC 확률 아님 | "
            "Not diagnostic | not a replacement for radiologist review."
        ),
    }


def _build_panel5(nsclc_prob, nsclc_label: str) -> dict:
    """NSCLC 보조 분류기 Panel 5 데이터"""
    if nsclc_prob is None:
        return {
            "available": False,
            "reason": "NSCLC 추론 실패 또는 데이터 없음",
            "disclaimer": "연구용 보조 증거 | 진단 아님 | Shortcut risk 미제거",
        }

    prob_pct = round(nsclc_prob * 100, 1)

    if nsclc_prob >= 0.7:
        level = "high"
        interpretation = f"NSCLC-lesion-like 보조 점수 높음 ({prob_pct}%). 추가 맥락 확인 권장."
        color = "red"
    elif nsclc_prob >= 0.5:
        level = "borderline"
        interpretation = f"NSCLC-lesion-like 경계 수준 ({prob_pct}%). 단독 판정 근거 불충분."
        color = "yellow"
    elif nsclc_prob >= 0.3:
        level = "low_nsclc"
        interpretation = f"Normal-like 우세 (NSCLC-like {prob_pct}%). 낮은 수준의 NSCLC 신호."
        color = "blue"
    else:
        level = "normal_like"
        interpretation = f"Normal-like ({prob_pct}%). NSCLC-lesion 패턴과 거리 멈."
        color = "green"

    return {
        "available": True,
        "prob":          nsclc_prob,
        "prob_pct":      prob_pct,
        "label":         nsclc_label,
        "level":         level,
        "color":         color,
        "interpretation": interpretation,
        "model_info":    "P-C-NORMAL24j | EfficientNet-B0 + scalar fusion | Patient AUROC=0.9898",
        "performance":   "Sensitivity=0.9918, Specificity=0.4167 (threshold=0.5) — high sensitivity, low specificity",
        "caution": (
            "특이도 41.7%: 정상 폐에서도 양성 판정 多. "
            "이상 점수와 독립적 증거로만 사용. 이상 점수에 합산 금지."
        ),
        "disclaimer": (
            "연구용 보조 증거 | 진단 아님 | NSCLC 확률 아님 | "
            "Shortcut risk(SR-HU, SR-CONTEXT) 미제거 | "
            "Not a replacement for radiologist review."
        ),
    }


# ── 메인 함수 ──────────────────────────────────────────────────────────────────

def generate_card_data(
    slice_index: int,
    total_slices: int,
    top_patch: dict,
    all_patches: list,
    ct_crop_b64: str = None,
    heatmap_ct_crop_b64: str = None,
    nsclc_prob: float = None,
    nsclc_label: str = None,
    gradcam_base64: str = None,
    heatmap_type: str = None,
    lung_z_pct: float = None,
    y_pct: float = None,
    x_pct: float = None,
    # 하위 호환 (더 이상 사용 안 함)
    dicom_pixel: list = None,
) -> dict:
    """
    slice_index  : 현재 슬라이스 z (0-based)
    total_slices : 전체 슬라이스 수
    top_patch    : anomaly_patches 중 score 가장 높은 것
    all_patches  : anomaly_patches 전체 리스트
    ct_crop_b64  : 백엔드 lung window 128×128 CT crop base64 (권장)
    heatmap_type : "gradcam" | "rd4ad"
    """
    pos = top_patch["position"]
    y0, x0, y1, x1 = pos["y0"], pos["x0"], pos["y1"], pos["x1"]
    cy = (y0 + y1) / 2
    cx = (x0 + x1) / 2

    # 폐 마스크 기반 상대좌표 (main.py에서 전달). 없으면 폴백
    if lung_z_pct is None:
        lung_z_pct = round(slice_index / max(total_slices - 1, 1), 4)
    if y_pct is None:
        y_pct = round(cy / 512, 4)
    if x_pct is None:
        x_pct = round(cx / 512, 4)
    side  = "left" if cx < 256 else "right"

    position_bin = _position_bin(cy, cx)

    # 3x3 패치 응답 맵
    patch_stride = 16
    patch_dict   = {
        (p["position"]["y0"], p["position"]["x0"]): p["score"]
        for p in all_patches
    }
    ys = [y0 - patch_stride, y0, y0 + patch_stride]
    xs = [x0 - patch_stride, x0, x0 + patch_stride]
    patch_map = {}
    for yy in ys:
        for xx in xs:
            v = patch_dict.get((yy, xx))
            patch_map[f"{yy},{xx}"] = round(v, 4) if v is not None else None

    # 정상 reference 매칭
    refs       = select_normal_refs(lung_z_pct, y_pct, x_pct, side)
    ref_images = []
    # 후보 crop(main.py)과 동일하게 "폐 bbox 높이의 일정 비율"을 FOV로 사용 → 줌 정규화 (A)
    OUT_SIZE = 256
    LUNG_CROP_FRAC = 0.70   # main.py 후보 crop과 반드시 동일 상수
    for ref in refs:
        png_path = os.path.join(BANK_DIR, ref["row"]["png_path"].replace("/", os.sep))
        if os.path.exists(png_path):
            row = ref["row"]
            # ref의 폐 bbox에 candidate y_pct/x_pct 역산 → 동일 상대위치 절대좌표
            by0 = float(row["lung_bbox_y0"]); by1 = float(row["lung_bbox_y1"])
            bx0 = float(row["lung_bbox_x0"]); bx1 = float(row["lung_bbox_x1"])
            crop_cy = int(by0 + y_pct * (by1 - by0))
            crop_cx = int(bx0 + x_pct * (bx1 - bx0))
            img = Image.open(png_path).convert("L")
            IW, IH = img.size
            # 폐 높이 비율로 crop half 결정 (후보와 동일 척도)
            _half = max(48, int(LUNG_CROP_FRAC * max(by1 - by0, 1) / 2))
            crop_y0 = max(0, crop_cy - _half); crop_y1 = min(IH, crop_cy + _half)
            crop_x0 = max(0, crop_cx - _half); crop_x1 = min(IW, crop_cx + _half)
            ref_crop = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            _side = 2 * _half
            if ref_crop.size != (_side, _side):
                padded = Image.new("L", (_side, _side), 0)
                padded.paste(ref_crop, (_half - (crop_cx - crop_x0), _half - (crop_cy - crop_y0)))
                ref_crop = padded
            ref_crop = ref_crop.resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)
            print(f"[DBG-REF] alias={ref['alias']} img={img.size} lung_h={by1-by0} half={_half} fov={_side}px->256 crop_cy={crop_cy} crop_cx={crop_cx}", flush=True)
            ref_images.append({
                "alias":      ref["alias"],
                "lung_z_pct": float(row["lung_z_pct"]),
                "distance":   ref["distance"],
                "image_base64": _img_to_base64(ref_crop),
            })

    # 후보 크롭: 백엔드 직접 생성(lung window 보장) 우선, 없으면 None
    candidate_crop_b64 = ct_crop_b64

    rd4ad_score = float(top_patch.get("score", 0.0))

    return {
        "slice_index":   slice_index,
        "lung_z_pct":    lung_z_pct,
        "position_bin":  position_bin,
        "candidate": {
            "y0": y0, "x0": x0, "y1": y1, "x1": x1,
            "score":       round(rd4ad_score, 4),
            "padim_score": round(float(top_patch.get("padim_score", 0.0)), 4),
            "side":        side,
            "crop_base64": candidate_crop_b64,
        },
        "patch_3x3":  patch_map,
        "normal_refs": ref_images,
        "panel4":  _build_panel4(position_bin, rd4ad_score, lung_z_pct),
        "panel5":  _build_panel5(nsclc_prob, nsclc_label or "unavailable"),
        "gradcam_base64":  gradcam_base64,
        "heatmap_ct_crop_b64": heatmap_ct_crop_b64,  # 히트맵과 동일 96px FOV CT (B)
        "heatmap_type":    heatmap_type,   # "gradcam" | "rd4ad" | None
    }


def _position_bin(cy: float, cx: float, H: float = 512, W: float = 512) -> str:
    zone   = "upper"  if cy < H / 3 else ("middle" if cy < 2 * H / 3 else "lower")
    region = "central" if abs(cx - W / 2) < W * 0.2 else "peripheral"
    return f"{zone}_{region}"
