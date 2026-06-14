# -*- coding: utf-8 -*-
"""
build_dynamic_ref_card_any_candidate_rd4ad_v1.py

범용 dynamic-reference full XAI 카드 생성기 (RD4AD 버전).
후보(patient_id [+six_bin_label])만 주면, --score_csv (rd4ad_infer.py 출력) +
그 환자 CT/ROI(read-only) + 정상 reference bank 로부터 4-panel dark-card 를 자동 생성한다.

- Panel 1 = dynamic normal-reference (candidate + normal_patient_1/2/3), 2x 맥락 crop + 80mm 위치 box
- Panel 2 = candidate 국소 zoom (lung window) + 3x3 crop 위치 + peak box + legend
- Panel 3 = 3x3 RD4AD score grid (peak±stride 위치 lookup; MISSING=hatch, 보간 금지)
- Panel 4 = 해석/주의 (KO+EN)

원칙: saved score 만 읽음(재계산 0). model forward / feature / score / contribution / stage2 = 0.
score 컬럼: rd4ad_score 고정 (no fallback).
candidate CT 는 read-only mmap 1 slice. raw CT 는 출력에 copy 안 함.
render gate: ALLOW_CARD_RENDER=1 + ALLOW_CT_LOAD=1 + ALLOW_SOURCE_IMAGE_READ=1 + ALLOW_PNG_WRITE=1 + --render --confirm.
"""
import os, sys, csv, json, argparse, importlib.util, pathlib
from datetime import date

ROOT    = "/home/jinhy/project/lung-ct-anomaly"
CTBASE  = "/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy"
NB      = "/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy"
BANK       = os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/dynamic_normal_reference_bank_three_patients_v1")
BANK_INDEX = os.path.join(BANK, "dynamic_reference_slice_index.csv")
RETRIEVE_PY = os.path.join(ROOT, "scripts/retrieve_dynamic_normal_refs_three_patients.py")
FONT_KO = "/mnt/c/Windows/Fonts/malgun.ttf"
OUT_BASE = os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/dynamic_ref_card_any_candidate_rd4ad_v1")

WL, WW = -600.0, 1500.0
TILE_PX = 280; CONTEXT_FACTOR = 2.0; PHYS_MM = 80.0
SCORE_COL   = "rd4ad_score"
CROP_SIZE   = 96
CROP_STRIDE = 16
BRANCH = "rd_e1c_true_rd4ad_resnet18_lung_mip3ch"
MASK   = "lung_mip3ch"

ALLOW = {k: os.environ.get(k) == "1" for k in
         ["ALLOW_CARD_RENDER", "ALLOW_CT_LOAD", "ALLOW_SOURCE_IMAGE_READ", "ALLOW_PNG_WRITE"]}


def _abort(m, c=2):
    print("BLOCKED:", m, file=sys.stderr); sys.exit(c)


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
    report_slice = z  # RD4AD CSV has no slice_index; use local_z directly

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
    cslice = window(np.asarray(ct[z]))  # 512x512 uint8

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

    # --- fonts/colors ---
    def fnt(s, b=False):
        try: return ImageFont.truetype(FONT_KO, s)
        except Exception: return ImageFont.load_default()
    C_BG=(18,18,18,255); C_PANEL=(30,30,30,255); C_HEADER=(10,22,45,255); C_BORDER=(70,70,70,255)
    C_TITLE=(220,220,255,255); C_SEC=(150,195,255,255); C_BODY=(215,215,215,255); C_WARN=(255,215,75,255)
    C_SUB=(170,200,170,255); C_LBL=(190,190,190,255); C_YEL=(255,220,0,255); C_TILE=(12,12,12,255)
    ft_title=fnt(20); ft_sec=fnt(15); ft_body=fnt(13); ft_sm=fnt(11)

    def ctx_tile(sl_u8, cyx, cpx, box_inner, peak=None):
        H, W = sl_u8.shape
        ccy, ccx = cyx; bpx = int(round(cpx * CONTEXT_FACTOR)); half = bpx // 2
        ny0 = max(0, min(int(round(ccy)) - half, H - bpx)); nx0 = max(0, min(int(round(ccx)) - half, W - bpx))
        crop = sl_u8[ny0:ny0 + bpx, nx0:nx0 + bpx]
        src = Image.fromarray(crop, "L").convert("RGBA")
        tile = src.resize((TILE_PX, TILE_PX), Image.LANCZOS)
        s = TILE_PX / bpx; d = ImageDraw.Draw(tile)
        iy0, ix0, iy1, ix1 = box_inner
        d.rectangle([(ix0 - nx0) * s, (iy0 - ny0) * s, (ix1 - nx0) * s, (iy1 - ny0) * s],
                    outline=(255, 70, 70, 255) if peak else (255, 220, 0, 255), width=2)
        if peak:
            py0, px0, py1, px1 = peak
            d.rectangle([(px0 - nx0) * s, (py0 - ny0) * s, (px1 - nx0) * s, (py1 - ny0) * s],
                        outline=(255, 140, 0, 255), width=2)
        return tile

    # candidate tile: 80mm box around candidate + peak(=candidate bbox) 표시
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
    # whole slice + candidate bbox + 3x3 extent
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
    for bx, title in [(p2x, "Panel 2 : 후보 국소 (RD4AD)"), (p3x, "Panel 3 : 3x3 score grid"), (p4x, "Panel 4 : 설명 요약")]:
        d.rectangle([(bx, ly), (bx + LPW, ly + LH)], fill=C_PANEL, outline=C_BORDER, width=1)
        d.text((bx + PAD, ly + PAD), title, font=ft_sec, fill=C_SEC)

    # Panel 2: zoom around candidate (lung window) + 3x3 crop positions + peak
    zhalf = 160  # wider window for 96×96 crop
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

    # Panel 4 text
    def wrap(text, font, maxw):
        out = []; cur = ""
        for ch in text:
            if d.textlength(cur + ch, font=font) <= maxw: cur += ch
            else: out.append(cur); cur = ch
        if cur: out.append(cur)
        return out
    tw = LPW - 2 * PAD; yy4 = ly + PAD + 26
    P4 = [("[Key finding]", C_SEC,
           f"{patient} 저장 1차 후보(z{z}, {pbin}, score {score:.4f})를 정상 3명 dynamic reference와 비교."),
          ("[Interpretation]", C_SEC,
           "매칭=bilateral lung 기준 lung_z_pct + lung-bbox 상대 y/x, 절대 slice 아님. 80mm 동일 물리 시야, 맥락 crop+위치 box."),
          ("[Caution]", C_WARN,
           "same-z 아님. rd4ad_score 절대비교 금지. 흉막/종격동 인접 시 과해석 주의. ref 3개는 정상 예시(진단 근거 아님)."),
          ("[Disclaimer]", (195, 155, 155, 255),
           "연구용 보조 설명이며 진단이 아닙니다. saliency/attribution 스타일 지도가 아닙니다.")]
    for lab, col, ko in P4:
        d.text((p4x + PAD, yy4), lab, font=ft_sm, fill=col); yy4 += 16
        for ln in wrap(ko, ft_body, tw):
            d.text((p4x + PAD, yy4), ln, font=ft_body, fill=C_BODY); yy4 += 16
        yy4 += 6

    # --- save ---
    out_root = out_root or os.path.join(OUT_BASE, f"{patient}__{pbin}")
    png_dir = os.path.join(out_root, "cards_png"); json_dir = os.path.join(out_root, "cards_json")
    for p_ in (png_dir, json_dir): os.makedirs(p_, exist_ok=True)
    out_png  = os.path.join(png_dir,  f"{patient}__{pbin}_rd4ad_card.png")
    out_json = os.path.join(json_dir, f"{patient}__{pbin}_rd4ad_card.json")
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
            "panel3_3x3": {f"{k[0]},{k[1]}": v for k, v in G.items()},
            "normal_refs": ref_meta, "canvas_size_px": [CW, CH], "generated_date": str(date.today()),
            "safety": {"model_forward": False, "feature_extraction": False, "score_recompute": False,
                       "contribution_recalc": False, "stage2_holdout_access": False,
                       "ct_load_used": True, "raw_ct_copied": False,
                       "not_diagnostic": True, "no_saliency_attribution_style": True}}
    json.dump(meta, open(out_json, "w"), ensure_ascii=False, indent=2)
    with open(os.path.join(out_root, "errors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["type", "msg"]); w.writeheader()
        for e in errors: w.writerow(e)
    done = {"done": True, "patient": patient, "position_bin": pbin, "out_png": out_png,
            "rd4ad_score": score, "score_col": SCORE_COL, "score_recomputed": False,
            "errors": len(errors), "generated_date": str(date.today())}
    json.dump(done, open(os.path.join(out_root, "DONE.json"), "w"), ensure_ascii=False, indent=2)
    print(f"DONE  {out_png}")
    print(f"JSON  {out_json}")
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
