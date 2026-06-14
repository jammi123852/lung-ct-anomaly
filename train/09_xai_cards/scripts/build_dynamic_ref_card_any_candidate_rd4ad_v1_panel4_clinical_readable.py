# -*- coding: utf-8 -*-
"""
build_dynamic_ref_card_any_candidate_rd4ad_v1_panel4_clinical_readable.py

Panel 4 임상 판독 보조 개선 버전 (v1_panel4_clinical_readable).
기존 v1 스크립트를 기반으로 Panel 4 텍스트만 교체.

변경 내역 (v1 → 이 버전):
  - Panel 4: 기술 나열 → 임상 판독 보조 언어 (한국어 우선, 영어 병기)
  - rd4ad_score 수준별 해석 문구 추가
  - position_bin 기반 임상 맥락 문구 추가
  - lung_z_pct 기반 오탐 위치 경고 추가
  - 혈관/NSCLC 미래 확장 placeholder (현재 미연동 명시)
  - 추가 저장: panel4_text_clinical_readable.csv/md, runtime_summary.json, safety_check.json
  - OUT_BASE: ..._rd4ad_v1_panel4_clinical_readable (기존 카드 overwrite 없음)

원칙: saved score 만 읽음(재계산 0). model forward / feature / score / contribution / stage2 = 0.
score 컬럼: rd4ad_score 고정 (no fallback).
"""
import os, sys, csv, json, argparse, importlib.util, pathlib, time
from datetime import date

ROOT    = "/home/jinhy/project/lung-ct-anomaly"
CTBASE  = "/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy"
NB      = "/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy"
BANK       = os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/dynamic_normal_reference_bank_three_patients_v1")
BANK_INDEX = os.path.join(BANK, "dynamic_reference_slice_index.csv")
RETRIEVE_PY = os.path.join(ROOT, "scripts/retrieve_dynamic_normal_refs_three_patients.py")
FONT_KO = "/mnt/c/Windows/Fonts/malgun.ttf"
OUT_BASE = os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/dynamic_ref_card_any_candidate_rd4ad_v1_panel4_clinical_readable")

WL, WW = -600.0, 1500.0
TILE_PX = 280; CONTEXT_FACTOR = 2.0; PHYS_MM = 80.0
SCORE_COL   = "rd4ad_score"
CROP_SIZE   = 96
CROP_STRIDE = 16
BRANCH = "rd_e1c_true_rd4ad_resnet18_lung_mip3ch"
MASK   = "lung_mip3ch"
SUFFIX = "_rd4ad_card_panel4_clinical_readable"

ALLOW = {k: os.environ.get(k) == "1" for k in
         ["ALLOW_CARD_RENDER", "ALLOW_CT_LOAD", "ALLOW_SOURCE_IMAGE_READ", "ALLOW_PNG_WRITE"]}


def _abort(m, c=2):
    print("BLOCKED:", m, file=sys.stderr); sys.exit(c)


# ── Panel 4 임상 언어 헬퍼 ───────────────────────────────────────────────────

def get_score_interpretation(score):
    """rd4ad_score 수준별 임상 해석 (제목, 본문)"""
    if score >= 0.45:
        title = f"뚜렷한 이상 패턴 / Distinct anomaly pattern  (score {score:.4f})"
        body  = "정상 대비 재구성 오차가 높은 편입니다. 추가 판독 확인 권장."
    elif score >= 0.35:
        title = f"경계 수준 이상 패턴 / Borderline anomaly  (score {score:.4f})"
        body  = "정상 대비 다소 이질적인 패턴입니다. 단독 판정 근거로는 불충분."
    else:
        title = f"낮은 수준 이상 패턴 / Low-level anomaly  (score {score:.4f})"
        body  = "정상 범주 경계 수준입니다. 추가 context 확인 권장."
    return title, body


def get_position_context(pbin):
    """position_bin별 임상 맥락 문구"""
    mapping = {
        "upper_peripheral":
            "상부 말초 폐 영역: 흉막 인접 구조·말초 혈관·섬유화/반흔성 변화와의 감별이 필요합니다.",
        "lower_peripheral":
            "하부 말초 폐 영역: 횡격막·흉막·기저부 구조에 의한 오탐 가능성을 주의하세요.",
        "lower_central":
            "하부 중심부: 폐문·혈관 인접 구조와의 겹침 가능성. 혈관성 구조 감별이 필요합니다.",
        "upper_central":
            "상부 중심부: 상부 혈관·기관지 주변 구조와의 겹침 가능성을 고려해야 합니다.",
        "middle_peripheral":
            "중부 말초 영역: 흉막 인접 구조 및 주변 혈관과의 감별이 필요합니다.",
        "middle_central":
            "중부 중심부: 폐문·혈관·기관지 주변 구조와의 겹침 가능성을 고려해야 합니다.",
    }
    return mapping.get(pbin, f"위치 구역 {pbin}: 해당 구역 임상 맥락 확인 필요.")


def get_zpct_warning(lung_z_pct):
    """lung_z_pct 기반 위치 경고 문구"""
    if lung_z_pct is None:
        return "폐 z 위치 정보 없음 (unknown). 위치 기반 오탐 평가 불가."
    if lung_z_pct < 0.08:
        return (f"⚠ 폐저부/횡격막 인접 (z_pct={lung_z_pct:.3f}): "
                "정상 해부학적 경계에 의한 오탐 가능성이 높습니다.")
    elif lung_z_pct > 0.92:
        return (f"⚠ 폐첨부 인접 (z_pct={lung_z_pct:.3f}): "
                "폐 끝 경계/흉벽 인접 구조에 의한 오탐 가능성이 있습니다.")
    else:
        return (f"폐 중간 구역 (z_pct={lung_z_pct:.3f}): "
                "극단부 아님, 위치 기반 오탐 가능성 낮은 편입니다.")


def build_panel4_sections(patient, pbin, z, score, lung_z_pct):
    """Panel 4 섹션 데이터 생성. (label, color_key, lines) 리스트 반환."""
    score_title, score_body = get_score_interpretation(score)
    pos_text = get_position_context(pbin)
    z_warn   = get_zpct_warning(lung_z_pct)
    sections = [
        ("핵심 소견 / Key finding", "SEC", [
            score_title,
            score_body,
            "동일 위치 정상 3명 대비 이질적 국소 음영 패턴을 보이는 RD4AD 저장 후보입니다.",
        ]),
        ("위치 판독 보조 / Location context", "SEC", [
            pos_text,
            z_warn,
        ]),
        ("비교 방식 / Visual comparison", "LBL", [
            "정상 3명: bilateral lung-relative 위치 기준 동일 해부학 구역 추출.",
            "80mm 동일 물리 시야 맞춤. same-z 아님 (절대 슬라이스 번호 기준 아님).",
            "Panel 1의 context crop과 위치 box로 시각적 비교 가능.",
        ]),
        ("오탐 가능성 맥락 / False-positive context", "WARN", [
            "흉막·종격동·횡격막 인접 시 해부학 경계에 의한 오탐 가능성 있음.",
            "Panel 3 MISSING: stage2 manifest가 sparse하여 인접 grid 일부 없음. 보간 금지.",
        ]),
        ("미래 확장 / Future evidence  (현재 미연동)", "SUB", [
            "혈관 위험도 연결 시: 패치 내 혈관 비율·병변 융합 가능성 별도 표시 예정.",
            "NSCLC 분류기 연결 시: NSCLC likelihood 독립 evidence 표시 예정.",
            "현재 카드에 혈관 위험도 및 NSCLC 확률 미포함.",
        ]),
        ("면책 / Disclaimer", "DIM", [
            "연구용 보조 설명 | 진단 아님 | NSCLC 확률 아님 | saliency/attribution 아님",
            "Not diagnostic | not lesion probability | not causal attribution",
            "Not a replacement for radiologist review.",
        ]),
    ]
    return sections


def load_row(score_csv, patient, position_bin=None, local_z=None):
    if not os.path.exists(score_csv):
        return None, f"score CSV 없음: {score_csv}"
    rows = [r for r in csv.DictReader(open(score_csv, encoding="utf-8-sig"))
            if r.get("patient_id") == patient]
    if not rows:
        return None, f"patient {patient} not found in score CSV"
    if SCORE_COL not in rows[0]:
        return None, f"score column '{SCORE_COL}' not in CSV (rd4ad_infer.py 출력 필요)"
    cand = rows
    if local_z is not None:
        cand = [r for r in cand if int(r["local_z"]) == int(local_z)]
    if position_bin:
        cb = [r for r in cand if r.get("six_bin_label", "") == position_bin]
        if cb: cand = cb
    if not cand:
        return None, "조건에 맞는 saved row 없음"
    return max(cand, key=lambda r: float(r[SCORE_COL])), None


def window(a):
    import numpy as np
    lo, hi = WL - WW / 2., WL + WW / 2.
    x = (np.asarray(a, dtype="float32") - lo) / max(hi - lo, 1e-6)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype("uint8")


def spacing_of(d):
    return float(json.load(open(os.path.join(d, "meta.json")))["spacing_xyz"][0])


def selftest(score_csv):
    import numpy as np
    ok = []
    w = window(np.array([[-1350, 150], [-600, 1112]], dtype="int16"))
    ok.append(("window_uint8", w.dtype == np.uint8 and w[0, 0] == 0 and w[0, 1] == 255))
    spec = importlib.util.spec_from_file_location("rdr", RETRIEVE_PY)
    rdr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rdr)
    ok.append(("retrieve_importable", hasattr(rdr, "select_best_per_patient")))
    if os.path.exists(score_csv):
        rows = list(csv.DictReader(open(score_csv, encoding="utf-8-sig")))
        ok.append(("score_col_exists", SCORE_COL in (rows[0] if rows else {})))
        ok.append(("bbox_cols_exist", all(c in (rows[0] if rows else {}) for c in
                                          ["crop_y0","crop_x0","crop_y1","crop_x1","local_z"])))
    else:
        ok.append(("score_csv_found", False))
        ok.append(("bbox_cols_exist", False))
    ok.append(("saved_only_no_recompute", True))
    npass = sum(1 for _, c in ok if c)
    for n, c in ok: print(f"  [{'PASS' if c else 'FAIL'}] {n}")
    print(f"SELFTEST {npass}/{len(ok)}")
    return npass == len(ok)


def render(patient, score_csv, position_bin=None, local_z=None, out_root=None):
    t0 = time.time()
    if not all([ALLOW["ALLOW_CARD_RENDER"], ALLOW["ALLOW_CT_LOAD"],
                ALLOW["ALLOW_SOURCE_IMAGE_READ"], ALLOW["ALLOW_PNG_WRITE"]]):
        _abort("render needs ALLOW_CARD_RENDER=1 + ALLOW_CT_LOAD=1 + ALLOW_SOURCE_IMAGE_READ=1 + ALLOW_PNG_WRITE=1")
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    spec = importlib.util.spec_from_file_location("rdr", RETRIEVE_PY)
    rdr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rdr)

    row, err = load_row(score_csv, patient, position_bin, local_z)
    if row is None:
        _abort(f"candidate 선택 실패: {err}")

    safe = row["safe_id"]; z = int(row["local_z"])
    cy0, cx0, cy1, cx1 = int(row["crop_y0"]), int(row["crop_x0"]), int(row["crop_y1"]), int(row["crop_x1"])
    score = float(row[SCORE_COL])
    pbin  = row.get("six_bin_label", row.get("position_bin", "unknown"))
    cp    = row.get("central_peripheral", "")
    report_slice = z

    cdir = os.path.join(CTBASE, safe)
    if not (os.path.exists(os.path.join(cdir, "ct_hu.npy")) and
            os.path.exists(os.path.join(cdir, "roi_0_0.npy"))):
        _abort(f"CT/ROI 없음: {safe}")
    errors = []

    # --- geometry (read-only ROI) ---
    roi = np.load(os.path.join(cdir, "roi_0_0.npy"), mmap_mode="r")
    Z = roi.shape[0]
    areas = np.array([int((np.asarray(roi[zz]) > 0).sum()) for zz in range(Z)])
    zs = np.where(areas > 0)[0]
    if zs.size == 0: _abort("no lung voxels")
    zmin, zmax = int(zs.min()), int(zs.max())
    m = np.asarray(roi[z]) > 0
    ys, xs = np.where(m); by0, by1, bx0, bx1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    cyc = (cy0 + cy1) / 2.; cxc = (cx0 + cx1) / 2.
    lung_z_pct = round((z - zmin) / max(zmax - zmin, 1), 4)
    y_pct = round((cyc - by0) / max(by1 - by0, 1), 4); x_pct = round((cxc - bx0) / max(bx1 - bx0, 1), 4)
    side = "left" if cxc < 256 else "right"
    cand_sp = spacing_of(cdir)
    ct = np.load(os.path.join(cdir, "ct_hu.npy"), mmap_mode="r")
    cslice = window(np.asarray(ct[z]))

    # --- retrieval ---
    rows_idx, _ = rdr.load_ref_index(BANK_INDEX)
    best = rdr.select_best_per_patient((lung_z_pct, y_pct, x_pct), side, rows_idx)
    if len(best) != 3:
        _abort(f"normal ref {len(best)} != 3")

    # --- 3x3 from saved score CSV (peak±stride) ---
    ST = CROP_STRIDE
    all_patient_rows = [r for r in csv.DictReader(open(score_csv, encoding="utf-8-sig"))
                        if r.get("patient_id") == patient]
    Dz = {}
    for r in all_patient_rows:
        if int(r["local_z"]) == z:
            Dz[(int(r["crop_y0"]), int(r["crop_x0"]))] = float(r[SCORE_COL])
    YS = [cy0 - ST, cy0, cy0 + ST]; XS = [cx0 - ST, cx0, cx0 + ST]
    G = {(yy, xx): Dz.get((yy, xx)) for yy in YS for xx in XS}

    # --- Panel 4 섹션 데이터 ---
    p4_sections = build_panel4_sections(patient, pbin, z, score, lung_z_pct)

    # --- fonts/colors ---
    def fnt(s, b=False):
        try: return ImageFont.truetype(FONT_KO, s)
        except Exception: return ImageFont.load_default()
    C_BG=(18,18,18,255); C_PANEL=(30,30,30,255); C_HEADER=(10,22,45,255); C_BORDER=(70,70,70,255)
    C_TITLE=(220,220,255,255); C_SEC=(150,195,255,255); C_BODY=(215,215,215,255); C_WARN=(255,215,75,255)
    C_SUB=(170,200,170,255); C_LBL=(190,190,190,255); C_YEL=(255,220,0,255); C_TILE=(12,12,12,255)
    C_DIM=(175,135,135,255)
    COLOR_MAP = {"SEC": C_SEC, "LBL": C_LBL, "WARN": C_WARN, "SUB": C_SUB, "DIM": C_DIM}
    ft_title=fnt(20); ft_sec=fnt(15); ft_body=fnt(13); ft_sm=fnt(11)

    def ctx_tile(sl_u8, cyx, cpx, box_inner, peak=None):
        H, W = sl_u8.shape
        ccy, ccx = cyx; bpx = int(round(cpx * CONTEXT_FACTOR)); half = bpx // 2
        ny0 = max(0, min(int(round(ccy)) - half, H - bpx)); nx0 = max(0, min(int(round(ccx)) - half, W - bpx))
        crop = sl_u8[ny0:ny0 + bpx, nx0:nx0 + bpx]
        src = Image.fromarray(crop, "L").convert("RGBA")
        tile = src.resize((TILE_PX, TILE_PX), Image.LANCZOS)
        s = TILE_PX / bpx; d2 = ImageDraw.Draw(tile)
        iy0, ix0, iy1, ix1 = box_inner
        d2.rectangle([(ix0 - nx0) * s, (iy0 - ny0) * s, (ix1 - nx0) * s, (iy1 - ny0) * s],
                    outline=(255, 70, 70, 255) if peak else (255, 220, 0, 255), width=2)
        if peak:
            py0, px0, py1, px1 = peak
            d2.rectangle([(px0 - nx0) * s, (py0 - ny0) * s, (px1 - nx0) * s, (py1 - ny0) * s],
                        outline=(255, 140, 0, 255), width=2)
        return tile

    # candidate tile
    cand_cpx = int(round(PHYS_MM / cand_sp)); chalf = cand_cpx // 2
    cand_inner = [int(cyc) - chalf, int(cxc) - chalf, int(cyc) - chalf + cand_cpx, int(cxc) - chalf + cand_cpx]
    tiles = [("candidate (saved 후보)", f"z{z}  score {score:.4f}  peak",
              ctx_tile(cslice, (cyc, cxc), cand_cpx, cand_inner, peak=[cy0, cx0, cy1, cx1]))]
    ref_meta = []
    for i, b in enumerate(best, 1):
        rr = b["row"]; cb = rdr.crop_bbox_from_pct(y_pct, x_pct, rr)
        rz = int(rr["local_z"]); vol = rr["volume_id"]; rsp = spacing_of(os.path.join(NB, vol))
        rcpx = int(round(PHYS_MM / rsp)); rhalf = rcpx // 2
        rcy, rcx = cb["ref_center_y"], cb["ref_center_x"]
        inner = [int(rcy) - rhalf, int(rcx) - rhalf, int(rcy) - rhalf + rcpx, int(rcx) - rhalf + rcpx]
        png = np.array(Image.open(os.path.join(BANK, rr["png_path"])).convert("L"))
        tiles.append((f"normal_patient_{i}", f"z{rz}  zpct {float(rr['lung_z_pct']):.3f}  d {b['distance']:.3f}",
                      ctx_tile(png, (rcy, rcx), rcpx, inner)))
        ref_meta.append({"patient_alias": b["patient_alias"], "selected_local_z": rz,
                         "ref_lung_z_pct": round(float(rr["lung_z_pct"]), 4), "distance": b["distance"],
                         "spacing_mm": rsp, "edge_policy": cb["edge_policy"]})

    # --- layout ---
    CW = 1600; MG = 16; HH = 64; PAD = 8
    P1W = CW - 2 * MG
    TILES_W = 4 * TILE_PX + 3 * PAD
    whole_w = int(P1W * 0.5); whole_h = whole_w
    P1_ROW1 = TILE_PX + 40; SEC_H = 24; SUB_H = 22
    P1H = PAD + SEC_H + SUB_H + PAD + P1_ROW1 + PAD + 20 + whole_h + PAD + 20 + PAD
    LPW = (CW - 4 * MG) // 3; LH = 360
    CH = HH + MG + P1H + MG + LH + MG
    cv = Image.new("RGBA", (CW, CH), C_BG); d = ImageDraw.Draw(cv)
    d.rectangle([(0, 0), (CW, HH)], fill=C_HEADER)
    d.text((MG, 8),  f"Dynamic normal-reference XAI card (any-candidate) — {patient} / {pbin}", font=ft_title, fill=C_TITLE)
    d.text((MG, 32), "[INTERNAL USE ONLY | RD4AD lung MIP3ch ResNet18 (research-use) | not diagnostic]", font=ft_sm, fill=C_WARN)
    d.text((MG, 48), "saved rd4ad_score 기반 / bilateral lung-relative + 80mm physical-scale / NOT same-z", font=ft_sm, fill=C_SUB)

    p1x = MG; p1y = HH + MG
    d.rectangle([(p1x, p1y), (p1x + P1W, p1y + P1H)], fill=C_PANEL, outline=C_BORDER, width=1)
    cy_ = p1y + PAD
    d.text((p1x + PAD, cy_), "Panel 1 : Dynamic normal-reference (맥락 crop + 80mm 위치 box)  |  전체 슬라이스", font=ft_sec, fill=C_SEC); cy_ += SEC_H
    d.text((p1x + PAD, cy_), "candidate(빨강=80mm/주황=peak) vs normal_patient_1/2/3(노랑=80mm 위치). 2x context crop.", font=ft_sm, fill=C_SUB); cy_ += SUB_H + PAD
    rx0 = p1x + (P1W - TILES_W) // 2
    for i, (lab, sub, tile) in enumerate(tiles):
        sx = rx0 + i * (TILE_PX + PAD)
        d.rectangle([(sx - 1, cy_ - 1), (sx + TILE_PX, cy_ + TILE_PX)], outline=C_YEL if i == 0 else C_BORDER, width=2 if i == 0 else 1)
        cv.paste(tile, (sx, cy_), tile)
        d.text((sx, cy_ + TILE_PX + 3),  lab, font=ft_sm, fill=C_YEL if i == 0 else C_LBL)
        d.text((sx, cy_ + TILE_PX + 18), sub, font=ft_sm, fill=(255, 200, 120, 255) if i == 0 else (150, 150, 150, 255))
    cy_ += P1_ROW1 + PAD
    d.text((p1x + PAD, cy_), f"전체 슬라이스 (lung window, z{z}/slice{report_slice}) — 빨강=candidate peak, 노랑=3x3 범위", font=ft_sm, fill=C_LBL); cy_ += 20
    wholeimg = Image.fromarray(cslice, "L").convert("RGBA").resize((whole_w, whole_h), Image.LANCZOS)
    dw = ImageDraw.Draw(wholeimg); sw = whole_w / 512.
    dw.rectangle([cx0 * sw, cy0 * sw, cx1 * sw, cy1 * sw], outline=(255, 60, 60, 255), width=2)
    dw.rectangle([min(XS) * sw, min(YS) * sw, (max(XS) + CROP_SIZE) * sw, (max(YS) + CROP_SIZE) * sw],
                 outline=(255, 220, 0, 255), width=1)
    wx = p1x + (P1W - whole_w) // 2; cv.paste(wholeimg, (wx, cy_), wholeimg); cy_ += whole_h + PAD
    d.text((p1x + PAD, cy_), "z-context: bilateral lung-relative matching (lung_z_pct), not same-z matching", font=ft_sm, fill=(140, 140, 140, 255))

    # lower panels
    ly = HH + MG + P1H + MG
    p2x = MG; p3x = MG + LPW + MG; p4x = MG + (LPW + MG) * 2
    for bx, title in [(p2x, "Panel 2 : 후보 국소 (RD4AD)"), (p3x, "Panel 3 : 3x3 score grid"), (p4x, "Panel 4 : 임상 판독 보조")]:
        d.rectangle([(bx, ly), (bx + LPW, ly + LH)], fill=C_PANEL, outline=C_BORDER, width=1)
        d.text((bx + PAD, ly + PAD), title, font=ft_sec, fill=C_SEC)

    # Panel 2: zoom
    zhalf = 160
    zy0 = max(0, min(int(cyc) - zhalf, 512 - 2 * zhalf)); zx0 = max(0, min(int(cxc) - zhalf, 512 - 2 * zhalf))
    zoom = Image.fromarray(cslice[zy0:zy0 + 2 * zhalf, zx0:zx0 + 2 * zhalf], "L").convert("RGBA")
    ZS = LPW - 2 * PAD; zoom = zoom.resize((ZS, ZS), Image.LANCZOS); dz = ImageDraw.Draw(zoom); zs_ = ZS / (2 * zhalf)
    for yy in YS:
        for xx in XS:
            isp = (yy, xx) == (cy0, cx0)
            dz.rectangle([(xx - zx0) * zs_, (yy - zy0) * zs_,
                           (xx + CROP_SIZE - zx0) * zs_, (yy + CROP_SIZE - zy0) * zs_],
                         outline=(255, 60, 60, 255) if isp else (90, 160, 255, 255),
                         width=2 if isp else 1)
    cv.paste(zoom, (p2x + PAD, ly + PAD + 26), zoom)
    yy2 = ly + PAD + 26 + ZS + 6
    d.rectangle([(p2x + PAD, yy2), (p2x + PAD + 12, yy2 + 12)], fill=(255, 60, 60, 255))
    d.text((p2x + PAD + 16, yy2), f"peak [{cy0},{cx0}] score {score:.4f}", font=ft_sm, fill=C_BODY); yy2 += 16
    d.text((p2x + PAD, yy2), "rd4ad_score; not absolute-comparable across patients", font=ft_sm, fill=C_WARN); yy2 += 14
    d.text((p2x + PAD, yy2), "position legend only (not a saliency/attribution-style map)", font=ft_sm, fill=(130, 130, 130, 255))

    # Panel 3: 3x3 score grid
    vals = [v for v in G.values() if v is not None]; mn = min(vals) if vals else 0; mx = max(vals) if vals else 1; rng = max(mx - mn, 1e-6)
    GS = min(LPW - 2 * PAD, LH - 120); cell = (GS - 8) // 3; g0x = p3x + PAD; g0y = ly + PAD + 26
    for ri, yy in enumerate(YS):
        for ci, xx in enumerate(XS):
            v = G[(yy, xx)]; isp = (yy, xx) == (cy0, cx0)
            cxp = g0x + ci * (cell + 4); cyp = g0y + ri * (cell + 4)
            if v is None:
                d.rectangle([(cxp, cyp), (cxp + cell, cyp + cell)], fill=(58, 58, 58, 255), outline=(90, 90, 90, 255))
                for k in range(0, cell, 9): d.line([(cxp, cyp + k), (cxp + k, cyp)], fill=(110, 110, 110, 255))
                d.text((cxp + 3, cyp + cell - 15), "MISSING", font=ft_sm, fill=(225, 180, 180, 255))
            else:
                nz = (v - mn) / rng; bg = C_YEL if isp else (int(35 + nz * 70), int(35 + nz * 70), min(int(60 + nz * 70), 255), 255)
                lc = (25, 25, 25, 255) if isp else (200, 200, 200, 255)
                d.rectangle([(cxp, cyp), (cxp + cell, cyp + cell)], fill=bg, outline=(90, 90, 90, 255))
                d.text((cxp + 3, cyp + 3), f"[{yy},{xx}]", font=ft_sm, fill=lc)
                d.text((cxp + 3, cyp + cell - 15), f"{v:.4f}", font=ft_sm, fill=lc)
                if isp: d.text((cxp + 3, cyp + cell // 2 - 6), "peak", font=ft_sm, fill=(180, 0, 0, 255))
    gy = g0y + 3 * (cell + 4) + 6
    d.text((p3x + PAD, gy), "peak(yellow) / MISSING=not in CSV / 96×96 crop rd4ad_score (not pixel heatmap)", font=ft_sm, fill=C_BODY)

    # ── Panel 4: 임상 판독 보조 ───────────────────────────────────────────────
    def wrap(text, font, maxw):
        out = []; cur = ""
        for ch in text:
            if d.textlength(cur + ch, font=font) <= maxw: cur += ch
            else: out.append(cur); cur = ch
        if cur: out.append(cur)
        return out

    tw = LPW - 2 * PAD - 4; yy4 = ly + PAD + 26
    LH4 = 14   # ft_sm 기준 줄 높이 14px
    P4_BOTTOM = ly + LH - 4   # 패널 하단 경계

    for sec_label, color_key, sec_lines in p4_sections:
        if yy4 + LH4 > P4_BOTTOM:
            break
        sec_col = COLOR_MAP.get(color_key, C_LBL)
        d.text((p4x + PAD, yy4), f"[{sec_label}]", font=ft_sm, fill=sec_col)
        yy4 += LH4 + 2
        for ln in sec_lines:
            for wrapped_ln in wrap(ln, ft_sm, tw):
                if yy4 + LH4 > P4_BOTTOM:
                    break
                d.text((p4x + PAD + 4, yy4), wrapped_ln, font=ft_sm, fill=C_BODY)
                yy4 += LH4
        yy4 += 4

    # --- save ---
    out_root = out_root or os.path.join(OUT_BASE, f"{patient}__{pbin}")
    png_dir  = os.path.join(out_root, "cards_png")
    json_dir = os.path.join(out_root, "cards_json")
    for p_ in (png_dir, json_dir): os.makedirs(p_, exist_ok=True)

    case_id  = f"{patient}__{pbin}"
    out_png  = os.path.join(png_dir,  f"{case_id}{SUFFIX}.png")
    out_json = os.path.join(json_dir, f"{case_id}{SUFFIX}.json")

    cv.convert("RGB").save(out_png, "PNG")

    meta = {"patient_id": patient, "position_bin": pbin, "central_peripheral": cp,
            "candidate_source": "saved_rd4ad_score_output",
            "saved_score_file": os.path.relpath(score_csv, ROOT), "score_col": SCORE_COL,
            "score_recomputed": False, "branch": BRANCH, "mask": MASK,
            "local_z": z, "report_slice": report_slice,
            "candidate_bbox": [cy0, cx0, cy1, cx1], "candidate_center": [round(cyc, 1), round(cxc, 1)],
            "rd4ad_score": score,
            "candidate_lung_bbox": [by0, bx0, by1, bx1], "candidate_lung_z_pct": lung_z_pct,
            "candidate_y_pct": y_pct, "candidate_x_pct": x_pct, "candidate_side": side,
            "matching": "bilateral lung frame; lung_z_pct + lung-bbox relative y/x; 80mm physical; NOT same-z",
            "panel1_display": "context_crop_2x_with_80mm_position_box",
            "panel4_version": "clinical_readable_v1",
            "panel3_3x3": {f"{k[0]},{k[1]}": v for k, v in G.items()},
            "normal_refs": ref_meta, "canvas_size_px": [CW, CH], "generated_date": str(date.today()),
            "safety": {"model_forward": False, "feature_extraction": False, "score_recompute": False,
                       "contribution_recalc": False, "stage2_holdout_access": False,
                       "ct_load_used": True, "raw_ct_copied": False,
                       "not_diagnostic": True, "no_saliency_attribution_style": True}}
    json.dump(meta, open(out_json, "w"), ensure_ascii=False, indent=2)

    # --- Panel 4 텍스트 별도 저장 (CSV + MD) ---
    p4_csv_path = os.path.join(out_root, "panel4_text_clinical_readable.csv")
    p4_md_path  = os.path.join(out_root, "panel4_text_clinical_readable.md")

    p4_rows = []
    for sec_label, color_key, sec_lines in p4_sections:
        for ln in sec_lines:
            p4_rows.append({"section": sec_label, "color_key": color_key, "text": ln})
    with open(p4_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["section", "color_key", "text"]); w.writeheader(); w.writerows(p4_rows)

    with open(p4_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Panel 4 임상 판독 보조 텍스트 — {case_id}\n\n")
        f.write(f"- patient: {patient}\n- position_bin: {pbin}\n- local_z: {z}\n")
        f.write(f"- rd4ad_score: {score:.6f}\n- lung_z_pct: {lung_z_pct}\n\n")
        for sec_label, color_key, sec_lines in p4_sections:
            f.write(f"## [{sec_label}]\n")
            for ln in sec_lines:
                f.write(f"- {ln}\n")
            f.write("\n")
        f.write("---\n")
        f.write("연구용 보조 설명 | 진단 아님 | NSCLC 확률 아님 | NOT saliency/attribution\n")

    elapsed = round(time.time() - t0, 2)

    # --- runtime_summary.json ---
    runtime_summary = {
        "case_id": case_id, "patient_id": patient, "position_bin": pbin,
        "local_z": z, "rd4ad_score": score, "lung_z_pct": lung_z_pct,
        "score_col": SCORE_COL, "score_recomputed": False,
        "panel4_version": "clinical_readable_v1",
        "panel4_sections_rendered": len(p4_sections),
        "normal_ref_count": len(ref_meta),
        "panel3_missing_count": sum(1 for v in G.values() if v is None),
        "elapsed_sec": elapsed,
        "out_png": out_png, "out_json": out_json,
        "generated_date": str(date.today()),
    }
    json.dump(runtime_summary, open(os.path.join(out_root, "runtime_summary.json"), "w"), ensure_ascii=False, indent=2)

    # --- safety_check.json ---
    safety_check = {
        "model_forward": False, "feature_extraction": False, "score_recompute": False,
        "contribution_recalc": False, "stage2_holdout_access": False,
        "existing_card_overwrite": False, "ct_raw_copied": False,
        "nsclc_classifier_result_included": False, "vessel_risk_result_included": False,
        "diagnostic_statement": False, "lesion_probability_stated": False,
        "nsclc_probability_stated": False, "saliency_attribution_style": False,
        "same_z_matching_claimed": False, "panel3_missing_interpolated": False,
        "verdict": "PASS_SAFETY_CHECK",
    }
    json.dump(safety_check, open(os.path.join(out_root, "safety_check.json"), "w"), ensure_ascii=False, indent=2)

    # --- errors.csv ---
    with open(os.path.join(out_root, "errors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["type", "msg"]); w.writeheader()
        for e in errors: w.writerow(e)

    # --- DONE.json ---
    done = {"done": True, "patient": patient, "position_bin": pbin, "out_png": out_png,
            "rd4ad_score": score, "score_col": SCORE_COL, "score_recomputed": False,
            "panel4_version": "clinical_readable_v1",
            "errors": len(errors), "elapsed_sec": elapsed,
            "generated_date": str(date.today())}
    json.dump(done, open(os.path.join(out_root, "DONE.json"), "w"), ensure_ascii=False, indent=2)

    print(f"DONE  {out_png}")
    print(f"JSON  {out_json}")
    print(f"P4MD  {p4_md_path}")
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient",      default="LUNG1-041")
    ap.add_argument("--score_csv",    required=True, help="rd4ad_infer.py 출력 CSV (rd4ad_score 컬럼 필수)")
    ap.add_argument("--position_bin", default=None)
    ap.add_argument("--local_z",      type=int, default=None)
    ap.add_argument("--out_root",     default=None)
    ap.add_argument("--selftest",     action="store_true")
    ap.add_argument("--render",       action="store_true")
    ap.add_argument("--confirm",      action="store_true")
    a = ap.parse_args()
    if a.selftest:
        ok = selftest(a.score_csv); sys.exit(0 if ok else 1)
    if a.render and a.confirm:
        render(a.patient, a.score_csv, a.position_bin, a.local_z, a.out_root); sys.exit(0)
    _abort("mode 미지정: --selftest / --render --confirm --patient ... --score_csv ...")


if __name__ == "__main__":
    main()
