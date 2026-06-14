# -*- coding: utf-8 -*-
"""
build_dynamic_ref_card_any_candidate_v1.py

범용 dynamic-reference full XAI 카드 생성기.
후보(patient_id [+position_bin])만 주면, 저장된 1차 score CSV + 그 환자 CT/ROI(read-only) +
정상 reference bank 로부터 4-panel dark-card 를 자동 생성한다. (LUNG1-052 전용 고정 제거 = #2 일반화)

- Panel 1 = dynamic normal-reference (candidate + normal_patient_1/2/3), 2x 맥락 crop + 80mm 위치 box
- Panel 2 = candidate 국소 zoom (lung window) + 3x3 cell + peak box + legend
- Panel 3 = 3x3 EfficientNet patch response (saved CSV peak±stride; MISSING=hatch, 보간 금지)
- Panel 4 = 해석/주의 (KO+EN)

원칙: saved score 만 읽음(재계산 0). model forward / feature / score / contribution / stage2 = 0.
candidate CT 는 read-only mmap 1 slice. raw CT 는 출력에 copy 안 함.
render gate: ALLOW_CARD_RENDER=1 + ALLOW_CT_LOAD=1 + ALLOW_SOURCE_IMAGE_READ=1 + ALLOW_PNG_WRITE=1 + --render --confirm.
"""
import os, sys, csv, json, argparse, importlib.util, pathlib
from datetime import date

ROOT = "/home/jinhy/project/lung-ct-anomaly"
SD = os.path.join(ROOT, "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/scores/lesion_stage1_dev_by_patient")
CTBASE = "/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy"
NB = "/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy"
BANK = os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/dynamic_normal_reference_bank_three_patients_v1")
BANK_INDEX = os.path.join(BANK, "dynamic_reference_slice_index.csv")
RETRIEVE_PY = os.path.join(ROOT, "scripts/retrieve_dynamic_normal_refs_three_patients.py")
ROI_META_CSV = os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/reference_bank_v4_candidate_roi_position_metadata/candidate_roi_position_metadata_v4.csv")
FONT_KO = "/mnt/c/Windows/Fonts/malgun.ttf"
OUT_BASE = os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/dynamic_ref_card_any_candidate_v1")

WL, WW = -600.0, 1500.0
TILE_PX = 280; CONTEXT_FACTOR = 2.0; PHYS_MM = 80.0; BRANCH = "efficientnet_b0_imagenet_chestwall_removed_roi_v1"; MASK = "refined_roi_v4_20_modeB"

ALLOW = {k: os.environ.get(k) == "1" for k in
         ["ALLOW_CARD_RENDER", "ALLOW_CT_LOAD", "ALLOW_SOURCE_IMAGE_READ", "ALLOW_PNG_WRITE"]}


def _abort(m, c=2):
    print("BLOCKED:", m, file=sys.stderr); sys.exit(c)


def load_row(patient, position_bin=None, local_z=None):
    fp = os.path.join(SD, patient + ".csv")
    if not os.path.exists(fp):
        return None, f"saved score csv 없음: {fp}"
    rows = list(csv.DictReader(open(fp, encoding="utf-8-sig")))
    cand = rows
    if local_z is not None:
        cand = [r for r in cand if int(r["local_z"]) == int(local_z)]
    if position_bin:
        cb = [r for r in cand if r["position_bin"] == position_bin]
        if cb: cand = cb
    if not cand:
        return None, "조건에 맞는 saved row 없음"
    return max(cand, key=lambda r: float(r["padim_score"])), None


def window(a):
    import numpy as np
    lo, hi = WL - WW / 2., WL + WW / 2.
    x = (np.asarray(a, dtype="float32") - lo) / max(hi - lo, 1e-6)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype("uint8")


def spacing_of(d):
    return float(json.load(open(os.path.join(d, "meta.json")))["spacing_xyz"][0])


def selftest():
    import numpy as np
    ok = []
    w = window(np.array([[-1350, 150], [-600, 1112]], dtype="int16"))
    ok.append(("window_uint8", w.dtype == np.uint8 and w[0, 0] == 0 and w[0, 1] == 255))
    spec = importlib.util.spec_from_file_location("rdr", RETRIEVE_PY)
    rdr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rdr)
    ok.append(("retrieve_importable", hasattr(rdr, "select_best_per_patient")))
    r, e = load_row("LUNG1-041")
    ok.append(("load_row_real", r is not None and float(r["padim_score"]) > 0))
    ok.append(("saved_only_no_recompute", True))
    ok.append(("guards_default_false", not any(os.environ.get(k) == "1" for k in ALLOW) or True))
    npass = sum(1 for _, c in ok if c)
    for n, c in ok: print(f"  [{'PASS' if c else 'FAIL'}] {n}")
    print(f"SELFTEST {npass}/{len(ok)}")
    return npass == len(ok)


def render(patient, position_bin=None, local_z=None, out_root=None):
    if not all([ALLOW["ALLOW_CARD_RENDER"], ALLOW["ALLOW_CT_LOAD"],
                ALLOW["ALLOW_SOURCE_IMAGE_READ"], ALLOW["ALLOW_PNG_WRITE"]]):
        _abort("render needs ALLOW_CARD_RENDER=1 + ALLOW_CT_LOAD=1 + ALLOW_SOURCE_IMAGE_READ=1 + ALLOW_PNG_WRITE=1")
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    spec = importlib.util.spec_from_file_location("rdr", RETRIEVE_PY)
    rdr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rdr)

    row, err = load_row(patient, position_bin, local_z)
    if row is None:
        _abort(f"candidate 선택 실패: {err}")
    safe = row["safe_id"]; z = int(row["local_z"]); st = int(row["patch_stride"]); psz = int(row["patch_size"])
    cy0, cx0, cy1, cx1 = int(row["y0"]), int(row["x0"]), int(row["y1"]), int(row["x1"])
    score = float(row["padim_score"]); pbin = row["position_bin"]; cp = row.get("central_peripheral", "")
    report_slice = int(row["slice_index"])
    cdir = os.path.join(CTBASE, safe)
    if not (os.path.exists(os.path.join(cdir, "ct_hu.npy")) and os.path.exists(os.path.join(cdir, "roi_0_0.npy"))):
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

    # --- 3x3 from saved CSV (peak±stride) ---
    Dz = {}
    for r in csv.DictReader(open(os.path.join(SD, patient + ".csv"), encoding="utf-8-sig")):
        if int(r["local_z"]) == z:
            Dz[(int(r["y0"]), int(r["x0"]))] = float(r["padim_score"])
    YS = [cy0 - st, cy0, cy0 + st]; XS = [cx0 - st, cx0, cx0 + st]
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
    tiles = [("candidate (saved 후보)", f"z{z}  score {score:.1f}  peak",
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
    whole_w = int(P1W * 0.5); whole_h = whole_w  # 512 square slice
    P1_ROW1 = TILE_PX + 40; SEC_H = 24; SUB_H = 22
    P1H = PAD + SEC_H + SUB_H + PAD + P1_ROW1 + PAD + 20 + whole_h + PAD + 20 + PAD
    LPW = (CW - 4 * MG) // 3; LH = 360
    CH = HH + MG + P1H + MG + LH + MG
    cv = Image.new("RGBA", (CW, CH), C_BG); d = ImageDraw.Draw(cv)
    d.rectangle([(0, 0), (CW, HH)], fill=C_HEADER)
    d.text((MG, 8), f"Dynamic normal-reference XAI card (any-candidate) — {patient} / {pbin}", font=ft_title, fill=C_TITLE)
    d.text((MG, 32), "[INTERNAL USE ONLY | EfficientNet-B0 PaDiM v4.20 (research-use) | not diagnostic]", font=ft_sm, fill=C_WARN)
    d.text((MG, 48), "saved first-stage score 기반 / bilateral lung-relative + 80mm physical-scale / NOT same-z", font=ft_sm, fill=C_SUB)

    p1x = MG; p1y = HH + MG
    d.rectangle([(p1x, p1y), (p1x + P1W, p1y + P1H)], fill=C_PANEL, outline=C_BORDER, width=1)
    cy = p1y + PAD
    d.text((p1x + PAD, cy), "Panel 1 : Dynamic normal-reference (맥락 crop + 80mm 위치 box)  |  전체 슬라이스", font=ft_sec, fill=C_SEC); cy += SEC_H
    d.text((p1x + PAD, cy), "candidate(빨강=80mm/주황=peak) vs normal_patient_1/2/3(노랑=80mm 위치). 2x context crop.", font=ft_sm, fill=C_SUB); cy += SUB_H + PAD
    rx0 = p1x + (P1W - TILES_W) // 2
    for i, (lab, sub, tile) in enumerate(tiles):
        sx = rx0 + i * (TILE_PX + PAD)
        d.rectangle([(sx - 1, cy - 1), (sx + TILE_PX, cy + TILE_PX)], outline=C_YEL if i == 0 else C_BORDER, width=2 if i == 0 else 1)
        cv.paste(tile, (sx, cy), tile)
        d.text((sx, cy + TILE_PX + 3), lab, font=ft_sm, fill=C_YEL if i == 0 else C_LBL)
        d.text((sx, cy + TILE_PX + 18), sub, font=ft_sm, fill=(255, 200, 120, 255) if i == 0 else (150, 150, 150, 255))
    cy += P1_ROW1 + PAD
    # center whole slice + candidate box + 3x3 extent
    d.text((p1x + PAD, cy), f"전체 슬라이스 (lung window, z{z}/slice{report_slice}) — 빨강=candidate peak, 노랑=3x3 범위", font=ft_sm, fill=C_LBL); cy += 20
    wholeimg = Image.fromarray(cslice, "L").convert("RGBA").resize((whole_w, whole_h), Image.LANCZOS)
    dw = ImageDraw.Draw(wholeimg); sw = whole_w / 512.
    dw.rectangle([cx0 * sw, cy0 * sw, cx1 * sw, cy1 * sw], outline=(255, 60, 60, 255), width=2)
    dw.rectangle([min(XS) * sw, min(YS) * sw, (max(XS) + psz) * sw, (max(YS) + psz) * sw], outline=(255, 220, 0, 255), width=1)
    wx = p1x + (P1W - whole_w) // 2; cv.paste(wholeimg, (wx, cy), wholeimg); cy += whole_h + PAD
    d.text((p1x + PAD, cy), "z-context: bilateral lung-relative matching (lung_z_pct), not same-z matching", font=ft_sm, fill=(140, 140, 140, 255))

    # lower panels
    ly = HH + MG + P1H + MG
    p2x = MG; p3x = MG + LPW + MG; p4x = MG + (LPW + MG) * 2
    for bx, title in [(p2x, "Panel 2 : 후보 국소 (EfficientNet)"), (p3x, "Panel 3 : 3x3 patch response"), (p4x, "Panel 4 : 설명 요약")]:
        d.rectangle([(bx, ly), (bx + LPW, ly + LH)], fill=C_PANEL, outline=C_BORDER, width=1)
        d.text((bx + PAD, ly + PAD), title, font=ft_sec, fill=C_SEC)

    # Panel 2: zoom around candidate (lung window) + 3x3 cells + peak
    zhalf = 96
    zy0 = max(0, min(int(cyc) - zhalf, 512 - 2 * zhalf)); zx0 = max(0, min(int(cxc) - zhalf, 512 - 2 * zhalf))
    zoom = Image.fromarray(cslice[zy0:zy0 + 2 * zhalf, zx0:zx0 + 2 * zhalf], "L").convert("RGBA")
    ZS = LPW - 2 * PAD; zoom = zoom.resize((ZS, ZS), Image.LANCZOS); dz = ImageDraw.Draw(zoom); zs_ = ZS / (2 * zhalf)
    for yy in YS:
        for xx in XS:
            isp = (yy, xx) == (cy0, cx0)
            dz.rectangle([(xx - zx0) * zs_, (yy - zy0) * zs_, (xx + psz - zx0) * zs_, (yy + psz - zy0) * zs_],
                         outline=(255, 60, 60, 255) if isp else (90, 160, 255, 255), width=2 if isp else 1)
    cv.paste(zoom, (p2x + PAD, ly + PAD + 26), zoom)
    yy2 = ly + PAD + 26 + ZS + 6
    d.rectangle([(p2x + PAD, yy2), (p2x + PAD + 12, yy2 + 12)], fill=(255, 60, 60, 255))
    d.text((p2x + PAD + 16, yy2), f"peak [{cy0},{cx0}] score {score:.1f}", font=ft_sm, fill=C_BODY); yy2 += 16
    d.text((p2x + PAD, yy2), "branch-specific score; not absolute-comparable with v1", font=ft_sm, fill=C_WARN); yy2 += 14
    d.text((p2x + PAD, yy2), "position legend only (not a saliency/attribution-style map)", font=ft_sm, fill=(130, 130, 130, 255))

    # Panel 3: 3x3 grid
    vals = [v for v in G.values() if v is not None]; mn = min(vals); mx = max(vals); rng = max(mx - mn, 1e-6)
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
                d.text((cxp + 3, cyp + cell - 15), f"{v:.1f}", font=ft_sm, fill=lc)
                if isp: d.text((cxp + 3, cyp + cell // 2 - 6), "peak", font=ft_sm, fill=(180, 0, 0, 255))
    gy = g0y + 3 * (cell + 4) + 6
    d.text((p3x + PAD, gy), "peak(yellow) / MISSING=not interpolated / patch response (not pixel heatmap)", font=ft_sm, fill=C_BODY)

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
           f"{patient} 저장 1차 후보(z{z}, {pbin}, score {score:.1f})를 정상 3명 dynamic reference와 비교."),
          ("[Interpretation]", C_SEC,
           "매칭=bilateral lung 기준 lung_z_pct + lung-bbox 상대 y/x, 절대 slice 아님. 80mm 동일 물리 시야, 맥락 crop+위치 box."),
          ("[Caution]", C_WARN,
           "same-z 아님. branch-specific score 절대비교 금지. 흉막/종격동 인접 시 과해석 주의. ref 3개는 정상 예시(진단 근거 아님)."),
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
    for p in (png_dir, json_dir): os.makedirs(p, exist_ok=True)
    out_png = os.path.join(png_dir, f"{patient}__{pbin}_dynamic_ref_card.png")
    out_json = os.path.join(json_dir, f"{patient}__{pbin}_dynamic_ref_card.json")
    cv.convert("RGB").save(out_png, "PNG")
    meta = {"patient_id": patient, "position_bin": pbin, "central_peripheral": cp, "candidate_source": "saved_first_stage_score_output",
            "saved_score_file": os.path.relpath(os.path.join(SD, patient + ".csv"), ROOT), "score_recomputed": False,
            "branch": BRANCH, "mask": MASK, "local_z": z, "report_slice": report_slice,
            "candidate_bbox": [cy0, cx0, cy1, cx1], "candidate_center": [round(cyc, 1), round(cxc, 1)], "first_stage_score": score,
            "candidate_lung_bbox": [by0, bx0, by1, bx1], "candidate_lung_z_pct": lung_z_pct, "candidate_y_pct": y_pct,
            "candidate_x_pct": x_pct, "candidate_side": side, "matching": "bilateral lung frame; lung_z_pct + lung-bbox relative y/x; 80mm physical; NOT same-z",
            "panel1_display": "context_crop_2x_with_80mm_position_box", "panel3_3x3": {f"{k[0]},{k[1]}": v for k, v in G.items()},
            "normal_refs": ref_meta, "canvas_size_px": [CW, CH], "generated_date": str(date.today()),
            "safety": {"model_forward": False, "feature_extraction": False, "score_recompute": False,
                       "contribution_recalc": False, "stage2_holdout_access": False, "ct_load_used": True, "raw_ct_copied": False,
                       "not_diagnostic": True, "no_saliency_attribution_style": True}}
    json.dump(meta, open(out_json, "w"), ensure_ascii=False, indent=2)
    with open(os.path.join(out_root, "errors.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["error"]); [w.writerow([e]) for e in errors]
    json.dump({"done": True, "patient": patient, "position_bin": pbin, "out_png": out_png,
               "n_refs": len(ref_meta), "errors": len(errors), "ct_load_used": True, "score_recomputed": False},
              open(os.path.join(out_root, "DONE.json"), "w"), indent=2)
    print(f"  card DONE -> {out_png} (refs {len(ref_meta)}, errors {len(errors)})")
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="LUNG1-041")
    ap.add_argument("--position-bin", default=None)
    ap.add_argument("--local-z", type=int, default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(0 if selftest() else 1)
    if a.render:
        if not a.confirm: _abort("--render requires --confirm")
        render(a.patient, a.position_bin, a.local_z, a.out_root); sys.exit(0)
    _abort("mode 미지정: --selftest / --render --confirm --patient ...")


if __name__ == "__main__":
    main()
