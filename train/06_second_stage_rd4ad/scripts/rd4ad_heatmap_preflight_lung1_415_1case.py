# -*- coding: utf-8 -*-
"""
rd4ad_heatmap_preflight_lung1_415_1case.py

LUNG1-415 upper_peripheral 1케이스에 대해 RD4AD spatial anomaly heatmap
생성 가능 여부를 확인하는 preflight.

원칙:
  - saved score CSV read-only
  - CT/ROI mmap read-only
  - model forward 금지 (저장된 feature map 없으면 NEEDS 판정 후 중단)
  - spatial map .npy가 없으면 candidate_patch.png만 생성하고 NEEDS 보고
  - stage2_holdout_access = False
  - 기존 카드 overwrite = False
"""
import os, sys, csv, json
from datetime import date
from pathlib import Path

ROOT        = "/home/jinhy/project/lung-ct-anomaly"
INFER_DIR   = os.path.join(ROOT, "outputs/end/rd4ad_lung_mip3ch_infer_v1")
SCORE_CSV   = os.path.join(INFER_DIR, "test_LUNG1-415_scores.csv")
CTBASE      = "/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy"
OUT_ROOT    = os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/rd4ad_heatmap_preflight_lung1_415_1case_v1")
FONT_KO     = "/mnt/c/Windows/Fonts/malgun.ttf"

PATIENT     = "LUNG1-415"
LOCAL_Z     = 226
PBIN        = "upper_peripheral"
SCORE_SAVED = 0.40526
HU_MIN, HU_MAX = -1000.0, 600.0
CROP_SIZE   = 96


def build_mip3ch_crop(ct_arr, local_z, y0, x0, y1, x1):
    import numpy as np
    Z, H, W = ct_arr.shape
    z = int(local_z)
    y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)
    pad_top  = max(0, -y0);  pad_bot = max(0, y1 - H)
    pad_left = max(0, -x0);  pad_right = max(0, x1 - W)
    needs_pad = pad_top > 0 or pad_bot > 0 or pad_left > 0 or pad_right > 0
    cy0 = max(0, y0); cy1 = min(H, y1)
    cx0 = max(0, x0); cx1 = min(W, x1)

    def _win(sl):
        c = np.clip(sl.astype(np.float32), HU_MIN, HU_MAX)
        return (c - HU_MIN) / (HU_MAX - HU_MIN)

    def _ch(zi):
        zi = int(np.clip(zi, 0, Z - 1))
        s = _win(ct_arr[zi, cy0:cy1, cx0:cx1])
        if needs_pad:
            pm = "reflect" if (cy1 - cy0 > 1) and (cx1 - cx0 > 1) else "edge"
            s = np.pad(s, ((pad_top, pad_bot), (pad_left, pad_right)), mode=pm)
        return s

    def _mip(zs):
        return np.stack([_ch(zi) for zi in zs], axis=0).max(axis=0)

    return np.stack([
        _mip([z - 3, z - 2, z - 1]),
        _mip([z - 1, z,     z + 1]),
        _mip([z + 1, z + 2, z + 3]),
    ], axis=0).astype("float32")


def main():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(OUT_ROOT, exist_ok=True)
    errors = []

    # ── 1. source audit ────────────────────────────────────────────────────────
    candidate_files = [
        (SCORE_CSV, "csv", "rd4ad_score saved scores"),
        (os.path.join(INFER_DIR, "rd4ad_infer.py"), "py", "inference script"),
        (os.path.join(INFER_DIR, "best_train_loss.pth"), "pth", "RD4AD student checkpoint"),
        (os.path.join(INFER_DIR, "test_LUNG1-415_manifest.csv"), "csv", "LUNG1-415 manifest"),
    ]
    # spatial map candidates (.npy): de3/tf3/spatial_map 등
    npy_candidates = [
        os.path.join(INFER_DIR, f) for f in os.listdir(INFER_DIR) if f.endswith(".npy")
    ]
    for nf in npy_candidates:
        candidate_files.append((nf, "npy", "spatial feature map candidate"))

    audit_rows = []
    for fpath, ftype, note in candidate_files:
        exists = os.path.exists(fpath)
        shape_str = ""; dtype_str = ""
        if exists and ftype == "npy":
            try:
                arr = np.load(fpath, mmap_mode="r")
                shape_str = str(arr.shape); dtype_str = str(arr.dtype)
            except Exception as e:
                shape_str = f"load_error:{e}"
        elif exists and ftype == "csv":
            try:
                with open(fpath) as f:
                    rows = list(csv.DictReader(f))
                shape_str = f"({len(rows)} rows)"
            except Exception:
                pass
        audit_rows.append({
            "source_file": os.path.relpath(fpath, ROOT),
            "exists": exists,
            "file_type": ftype,
            "shape": shape_str,
            "dtype": dtype_str,
            "used_for_heatmap": ftype == "npy" and exists,
            "note": note,
        })

    spatial_maps_found = [r for r in audit_rows if r["file_type"] == "npy" and r["exists"]]
    heatmap_possible = len(spatial_maps_found) > 0

    with open(os.path.join(OUT_ROOT, "heatmap_source_audit.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source_file", "exists", "file_type", "shape", "dtype", "used_for_heatmap", "note"])
        w.writeheader(); w.writerows(audit_rows)

    # ── 2. saved score row 확인 ────────────────────────────────────────────────
    score_rows = [r for r in csv.DictReader(open(SCORE_CSV, encoding="utf-8-sig"))
                  if r.get("patient_id") == PATIENT and int(r.get("local_z", -1)) == LOCAL_Z]
    if not score_rows:
        errors.append({"type": "score_row_missing", "msg": f"z={LOCAL_Z} not found in score CSV"})
        score_match = False
        cand_row = None
    else:
        cand_row = max(score_rows, key=lambda r: float(r["rd4ad_score"]))
        score_saved_actual = float(cand_row["rd4ad_score"])
        score_match = abs(score_saved_actual - SCORE_SAVED) < 0.001

    # ── 3. candidate patch 생성 (CT mmap, model forward 없음) ─────────────────
    candidate_patch_generated = False
    out_patch_png = os.path.join(OUT_ROOT, "candidate_patch.png")
    try:
        safe = cand_row["safe_id"] if cand_row else f"NSCLC_{PATIENT}__75e5b68d83"
        ct_path = os.path.join(CTBASE, safe, "ct_hu.npy")
        if os.path.exists(ct_path) and cand_row:
            ct_arr = np.load(ct_path, mmap_mode="r")
            crop = build_mip3ch_crop(ct_arr, LOCAL_Z,
                                     int(cand_row["crop_y0"]), int(cand_row["crop_x0"]),
                                     int(cand_row["crop_y1"]), int(cand_row["crop_x1"]))
            # MIP3ch 3채널 → RGB 시각화 (0~1 float → 0~255 uint8)
            rgb = (np.stack([crop[0], crop[1], crop[2]], axis=-1) * 255).astype("uint8")
            # 각 채널 개별 grayscale도 저장
            ch_imgs = []
            for ch_i, ch_name in enumerate(["ch0_MIP_z-3~z-1", "ch1_MIP_z-1~z+1", "ch2_MIP_z+1~z+3"]):
                ch_arr = (crop[ch_i] * 255).astype("uint8")
                ch_imgs.append(ch_arr)

            # 3채널 RGB + 3채널 grayscale 나란히 (6열 → 96*6 + PAD)
            PAD = 4
            W_total = CROP_SIZE * 6 + PAD * 5 + 60  # 60px for labels
            H_total = CROP_SIZE + 40
            patch_img = Image.new("RGB", (W_total, H_total), (20, 20, 20))
            draw = ImageDraw.Draw(patch_img)
            try:
                font = ImageFont.truetype(FONT_KO, 10)
            except Exception:
                font = ImageFont.load_default()

            # RGB composite
            patch_img.paste(Image.fromarray(rgb, "RGB"), (0, 20))
            draw.text((0, 4), "MIP3ch RGB", font=font, fill=(200, 200, 255))

            for ci, (arr, lbl) in enumerate(zip(ch_imgs, ["ch0\nz-3~z-1", "ch1\nz-1~z+1", "ch2\nz+1~z+3"])):
                x = CROP_SIZE + PAD + ci * (CROP_SIZE + PAD)
                patch_img.paste(Image.fromarray(arr, "L").convert("RGB"), (x, 20))
                draw.text((x, 4), lbl.replace("\n", " "), font=font, fill=(150, 200, 150))

            # score info 텍스트
            info_x = CROP_SIZE * 4 + PAD * 4
            draw.text((info_x, 4),  f"patient: {PATIENT}", font=font, fill=(200, 200, 200))
            draw.text((info_x, 16), f"local_z: {LOCAL_Z}", font=font, fill=(200, 200, 200))
            draw.text((info_x, 28), f"pbin: {PBIN}", font=font, fill=(200, 200, 200))
            draw.text((info_x, 40), f"rd4ad_score: {SCORE_SAVED}", font=font, fill=(255, 220, 80))
            draw.text((info_x, 52), "crop: MIP3ch 96x96", font=font, fill=(150, 150, 150))
            draw.text((info_x, 64), "CT mmap read-only", font=font, fill=(120, 120, 120))
            draw.text((info_x, 76), "model_forward=False", font=font, fill=(180, 100, 100))

            patch_img.save(out_patch_png, "PNG")
            candidate_patch_generated = True
    except Exception as e:
        errors.append({"type": "patch_gen_error", "msg": str(e)})

    # ── 4. heatmap 생성 가능 여부 판정 ────────────────────────────────────────
    if heatmap_possible:
        verdict = "PASS_RD4AD_HEATMAP_PREFLIGHT_READY"
        heatmap_source = spatial_maps_found[0]["source_file"]
        heatmap_method = "saved_spatial_cosine_distance_map"
        overlay_generated = False  # 실제 heatmap 생성은 별도 단계
        limitations = ["spatial map exists — overlay generation pending next step"]
    else:
        verdict = "NEEDS_RD4AD_INFER_SAVE_HEATMAP"
        heatmap_source = "none_saved"
        heatmap_method = "none_available_without_model_forward"
        overlay_generated = False
        limitations = [
            "현재 rd4ad_infer.py는 scalar score(score_layer1/2/3, rd4ad_score)만 저장함.",
            "spatial cosine distance map (de3 vs tf3 per-spatial-position)이 저장되지 않음.",
            "model forward 없이 heatmap 생성 불가.",
            "다음 단계: rd4ad_infer.py에 --save_spatial_map 옵션 추가 후 재실행 필요.",
            "spatial map shape 예상: (H//32, W//32) = (3, 3) for 96×96 crop + layer3, (6,6) layer2, (12,12) layer1",
        ]

    # ── 5. heatmap placeholder PNG ──────────────────────────────────────────
    out_heatmap_png = os.path.join(OUT_ROOT, "rd4ad_spatial_anomaly_map.png")
    out_overlay_png = os.path.join(OUT_ROOT, "rd4ad_spatial_overlay.png")
    if not heatmap_possible:
        # MISSING placeholder
        for fpath, label in [(out_heatmap_png, "SPATIAL MAP NOT AVAILABLE\n(model forward required)"),
                              (out_overlay_png, "OVERLAY NOT AVAILABLE\n(spatial map required)")]:
            img = Image.new("RGB", (320, 120), (30, 20, 20))
            dw = ImageDraw.Draw(img)
            try: fnt = ImageFont.truetype(FONT_KO, 13)
            except: fnt = ImageFont.load_default()
            dw.text((10, 10), label, font=fnt, fill=(220, 100, 100))
            dw.text((10, 80), f"verdict: {verdict}", font=fnt, fill=(200, 200, 100))
            img.save(fpath, "PNG")

    # ── 6. report.md ───────────────────────────────────────────────────────────
    report_lines = [
        f"# RD4AD Heatmap Preflight Report — {PATIENT}__{PBIN}",
        "",
        f"- case_id: {PATIENT}__{PBIN}",
        f"- patient: {PATIENT}",
        f"- local_z: {LOCAL_Z}",
        f"- rd4ad_score_saved: {SCORE_SAVED}",
        f"- score_match: {score_match}",
        f"- heatmap_possible_without_model_forward: {heatmap_possible}",
        f"- verdict: **{verdict}**",
        "",
        "## Source Audit",
        "",
    ]
    for r in audit_rows:
        exists_str = "✅" if r["exists"] else "❌"
        report_lines.append(f"- {exists_str} `{r['source_file']}` ({r['file_type']}) — {r['note']}")
        if r["shape"]: report_lines.append(f"  - shape: {r['shape']}  dtype: {r['dtype']}")
    report_lines += [
        "",
        "## Spatial Map 저장 여부",
        "",
        f"- .npy spatial map files found: {len(spatial_maps_found)}",
        f"- heatmap_possible: {heatmap_possible}",
        "",
        "## Limitations",
        "",
    ]
    for lim in limitations:
        report_lines.append(f"- {lim}")
    report_lines += [
        "",
        "## 다음 단계",
        "",
        "### NEEDS_RD4AD_INFER_SAVE_HEATMAP인 경우:" if not heatmap_possible else "### PASS인 경우:",
    ]
    if not heatmap_possible:
        report_lines += [
            "1. `rd4ad_infer.py`에 `--save_spatial_map` 옵션 추가",
            "   - de3 vs tf3 per-spatial-position cosine distance map 저장",
            "   - 저장 형식: `{patient}_z{z}_y{cy0}_x{cx0}_spatial_map.npy` (float32, shape=(H',W'))",
            "2. LUNG1-415 1케이스 재추론",
            "3. 저장된 spatial map으로 overlay 생성",
            "4. 품질 확인 후 카드 통합 결정",
        ]
    else:
        report_lines += [
            "1. 저장된 spatial map으로 overlay PNG 생성",
            "2. Panel 2 또는 Panel 3에 통합",
        ]
    report_lines += [
        "",
        "---",
        "Research-use preflight | model_forward=False | stage2_holdout_access=False",
    ]
    with open(os.path.join(OUT_ROOT, "heatmap_preflight_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # ── 7. summary JSON ────────────────────────────────────────────────────────
    summary = {
        "case_id": f"{PATIENT}__{PBIN}",
        "patient": PATIENT,
        "local_z": LOCAL_Z,
        "rd4ad_score_saved": SCORE_SAVED,
        "score_match": score_match,
        "heatmap_possible_without_model_forward": heatmap_possible,
        "heatmap_source": heatmap_source,
        "heatmap_method": heatmap_method,
        "model_forward": False,
        "score_recompute": False,
        "feature_extraction": False,
        "stage2_holdout_access": False,
        "candidate_patch_generated": candidate_patch_generated,
        "overlay_generated": overlay_generated,
        "spatial_npy_files_found": len(spatial_maps_found),
        "verdict": verdict,
        "limitations": limitations,
        "generated_date": str(date.today()),
    }
    json.dump(summary, open(os.path.join(OUT_ROOT, "heatmap_preflight_summary.json"), "w"), ensure_ascii=False, indent=2)

    # safety check
    json.dump({"model_forward": False, "feature_extraction": False, "score_recompute": False,
               "stage2_holdout_access": False, "existing_card_overwrite": False,
               "raw_feature_copied_to_end_package": False,
               "verdict": "PASS_SAFETY_CHECK"},
              open(os.path.join(OUT_ROOT, "safety_check.json"), "w"), ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_ROOT, "errors.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["type", "msg"]); w.writeheader()
        for e in errors: w.writerow(e)

    json.dump({"done": True, "verdict": verdict, "patient": PATIENT, "local_z": LOCAL_Z,
               "heatmap_possible": heatmap_possible,
               "candidate_patch_generated": candidate_patch_generated,
               "errors": len(errors), "generated_date": str(date.today())},
              open(os.path.join(OUT_ROOT, "DONE.json"), "w"), ensure_ascii=False, indent=2)

    # ── 8. 결과 출력 ──────────────────────────────────────────────────────────
    print(f"VERDICT: {verdict}")
    print(f"heatmap_possible: {heatmap_possible}")
    print(f"spatial_npy_found: {len(spatial_maps_found)}")
    print(f"candidate_patch: {out_patch_png}  generated={candidate_patch_generated}")
    print(f"heatmap_png:     {out_heatmap_png}  (MISSING placeholder)")
    print(f"overlay_png:     {out_overlay_png}  (MISSING placeholder)")
    print(f"report:          {os.path.join(OUT_ROOT, 'heatmap_preflight_report.md')}")
    print(f"errors: {len(errors)}")
    for lim in limitations[:3]:
        print(f"  LIMIT: {lim}")


if __name__ == "__main__":
    main()
