#!/usr/bin/env python3
"""
make_b0_ct_context_review_pack.py

B0 CT-context visual review pack 생성:
  - contact sheet 3개 (normal_fp, lesion_safety, all_33)
  - review guide MD
  - label template CSV (33행, ct_context 컬럼 빈 값)
  - review pack metadata JSON

금지: CT/ROI/mask npy 로드, score CSV 로드, scoring/metric/suppression,
      기존 PNG 수정, current_state.md 수정, stage2_holdout 접근,
      원인 확정, score adjustment, threshold 변경
"""

import sys
import json
import math
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path("qa/dev_safe_mixed_error_visual_qa")
PANELS_DIR = BASE_DIR / "b0_ct_context_panels"
TARGETS_CSV = BASE_DIR / "b0_ct_context_shape_checked_targets.csv"
FILLED_GPT_CSV = BASE_DIR / "b0_vessel_pleura_visual_review_labels_template_filled_by_gpt.csv"

OUTPUT_CONTACT_NORMAL = BASE_DIR / "b0_ct_context_contact_sheet_normal_fp.png"
OUTPUT_CONTACT_LESION = BASE_DIR / "b0_ct_context_contact_sheet_lesion_safety.png"
OUTPUT_CONTACT_ALL = BASE_DIR / "b0_ct_context_contact_sheet_all_33.png"
OUTPUT_GUIDE = BASE_DIR / "b0_ct_context_review_guide.md"
OUTPUT_TEMPLATE = BASE_DIR / "b0_ct_context_review_labels_template.csv"
OUTPUT_JSON = BASE_DIR / "b0_ct_context_review_pack.json"

REQUIRED_TOTAL = 33
REQUIRED_NORMAL_FP = 9
REQUIRED_LESION_SAFETY = 24

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
]


def abort(msg):
    print(f"[ABORT] {msg}", file=sys.stderr)
    sys.exit(1)


def load_font(size):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def check_no_existing_outputs():
    for p in [OUTPUT_CONTACT_NORMAL, OUTPUT_CONTACT_LESION, OUTPUT_CONTACT_ALL,
              OUTPUT_GUIDE, OUTPUT_TEMPLATE, OUTPUT_JSON]:
        if p.exists():
            abort(f"output collision — {p} 이미 존재. 덮어쓰기 금지.")


def parse_review_id_from_stem(stem):
    # B0CTX_1_R004_normal_fp_bridge → parts[2] = 'R004'
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[2]
    return None


def validate_inputs(targets_df, png_files):
    if len(targets_df) != REQUIRED_TOTAL:
        abort(f"targets CSV {len(targets_df)}행 (expected {REQUIRED_TOTAL})")

    if len(png_files) != REQUIRED_TOTAL:
        abort(f"PNG {len(png_files)}장 (expected {REQUIRED_TOTAL})")

    zero_bytes = [p for p in png_files if p.stat().st_size == 0]
    if zero_bytes:
        abort(f"0바이트 PNG {len(zero_bytes)}건: {[p.name for p in zero_bytes]}")

    normal_fp_pngs = [p for p in png_files if "normal_fp_bridge" in p.name]
    lesion_safety_pngs = [p for p in png_files if "lesion_safety_bridge" in p.name]

    if len(normal_fp_pngs) != REQUIRED_NORMAL_FP:
        abort(f"normal_fp_bridge PNG {len(normal_fp_pngs)}장 (expected {REQUIRED_NORMAL_FP})")
    if len(lesion_safety_pngs) != REQUIRED_LESION_SAFETY:
        abort(f"lesion_safety_bridge PNG {len(lesion_safety_pngs)}장 (expected {REQUIRED_LESION_SAFETY})")

    if "stage2_holdout_flag" in targets_df.columns:
        holdout = targets_df["stage2_holdout_flag"].sum()
        if holdout != 0:
            abort(f"stage2_holdout_flag 비0 {holdout}건")

    png_review_ids = set()
    for p in png_files:
        rid = parse_review_id_from_stem(p.stem)
        if rid:
            png_review_ids.add(rid)

    csv_review_ids = set(targets_df["review_id"])
    if png_review_ids != csv_review_ids:
        missing = csv_review_ids - png_review_ids
        extra = png_review_ids - csv_review_ids
        abort(f"PNG-CSV review_id 불일치. 누락: {missing}, 초과: {extra}")

    print(f"[GUARD] 입력 검증 통과: total={REQUIRED_TOTAL}, "
          f"normal_fp={len(normal_fp_pngs)}, lesion_safety={len(lesion_safety_pngs)}")
    return normal_fp_pngs, lesion_safety_pngs


def record_mtimes(png_files):
    return {p: p.stat().st_mtime for p in png_files}


def verify_mtimes(png_files, original_mtimes):
    changed = [p.name for p in png_files if p.stat().st_mtime != original_mtimes[p]]
    if changed:
        abort(f"기존 PNG mtime 변화 감지: {changed}")
    print(f"[GUARD] 기존 PNG {len(png_files)}장 mtime 변화 없음 확인")


def make_tile_label(t_row, gpt_row, short=False):
    ct_context_id = t_row.get("ct_context_id", "")
    review_id = t_row.get("review_id", "")
    patient_id = str(t_row.get("patient_id", ""))
    patient_short = patient_id[-18:] if len(patient_id) > 18 else patient_id

    b0_visual_label = ""
    candidate_score = ""
    if gpt_row:
        b0_visual_label = gpt_row.get("b0_visual_label", "")
        raw_score = gpt_row.get("candidate_score", "")
        try:
            candidate_score = f"{float(raw_score):.2f}"
        except (TypeError, ValueError):
            candidate_score = str(raw_score)

    if short:
        return (f"{ct_context_id}|{review_id}\n"
                f"{patient_short}\n"
                f"{b0_visual_label}|sc={candidate_score}")
    return (f"{ct_context_id} | {review_id} | {patient_short}\n"
            f"{b0_visual_label} | score={candidate_score} | marker=center-only")


def make_contact_sheet(rows_data, output_path, n_cols, tile_img_w, title_text):
    n_total = len(rows_data)
    n_grid_rows = math.ceil(n_total / n_cols)

    sample_img = Image.open(rows_data[0]["png_path"])
    orig_w, orig_h = sample_img.size
    sample_img.close()

    tile_img_h = int(tile_img_w * orig_h / orig_w)
    header_h = 72
    tile_h = tile_img_h + header_h

    title_h = 44
    sheet_w = n_cols * tile_img_w
    sheet_h = title_h + n_grid_rows * tile_h

    sheet = Image.new("RGB", (sheet_w, sheet_h), (28, 28, 36))
    draw = ImageDraw.Draw(sheet)

    font_title = load_font(16)
    font_label = load_font(12)

    draw.text((8, 10), title_text, fill=(230, 230, 255), font=font_title)

    for idx, row_data in enumerate(rows_data):
        col = idx % n_cols
        row_i = idx // n_cols

        x = col * tile_img_w
        y = title_h + row_i * tile_h

        # 헤더 배경
        draw.rectangle([x, y, x + tile_img_w - 1, y + header_h - 2], fill=(44, 44, 66))

        # 헤더 텍스트
        label_text = row_data["label"]
        draw.text((x + 4, y + 4), label_text, fill=(200, 220, 255), font=font_label)

        # PNG 로드 및 resize
        img = Image.open(row_data["png_path"]).convert("RGB")
        img_resized = img.resize((tile_img_w, tile_img_h), Image.LANCZOS)
        img.close()
        sheet.paste(img_resized, (x, y + header_h))
        img_resized.close()

        # 테두리
        draw.rectangle([x, y, x + tile_img_w - 1, y + tile_h - 1], outline=(90, 90, 110))

    sheet.save(str(output_path))
    size_kb = output_path.stat().st_size // 1024
    print(f"[OK] {output_path.name}: {n_total}장, {n_cols}열 ({sheet_w}x{sheet_h}px, {size_kb}KB)")


def make_review_guide(output_path):
    content = """# B0 CT-context visual review guide

## 1. Scope
- dev_safe only
- B0 CT-context panel 33 PNGs
- normal FP 9
- lesion safety 24
- stage2_holdout 0
- existing PNG review only
- no CT/ROI/mask reload
- no score adjustment
- no suppression applied

## 2. How to read the panel
- each PNG contains z-2, z-1, z, z+1, z+2 context
- green contour = ROI
- red contour = lesion mask when lesion_safety
- yellow/orange center marker = candidate center
- patch extent is unknown; marker is center-only
- no patch rectangle should be interpreted as exact lesion overlap

## 3. Normal FP review questions
- Does the marker align with vessel-like structure?
- Does the marker align with pleura/chest-wall boundary?
- Does it align with hilar/mediastinal high-contrast structure?
- Is it diaphragm/base-related?
- Is it mixed or unclear?
- Do not conclude final cause.

## 4. Lesion safety review questions
- Is the lesion near vessel/pleura/hilar structure?
- Does the marker overlap or touch lesion contour?
- Would a vessel/pleura suppression rule risk suppressing the lesion candidate?
- Is CT context enough, or is local zoom/MIP needed?
- Do not approve suppression from this review alone.

## 5. Label options
### ct_context_label
- vessel_like_context_candidate
- pleura_or_chest_wall_context_candidate
- hilar_or_mediastinal_context_candidate
- diaphragm_or_base_context_candidate
- mixed_high_contrast_context_candidate
- lesion_near_vessel_or_pleura_context_candidate
- lesion_protect_context
- unclear

### ct_context_action
- keep_for_rule_design_review
- needs_local_zoom
- needs_mip_or_vessel_context
- reference_only
- exclude_from_b0

### confidence
- high
- medium
- low

## 6. Safety rules
- no rule selected from normal FP only
- lesion safety must be reviewed together
- if lesion safety overlap is high, suppression remains blocked
- no score adjustment
- no threshold retuning
- no stage2_holdout
- no full_retrospective

## 7. Next step
- human fills `b0_ct_context_review_labels_template.csv`
- then summarize CT-context labels
- only then decide whether local zoom/MIP is needed or B0 branch should be stopped/continued
"""
    output_path.write_text(content, encoding="utf-8")
    print(f"[OK] {output_path.name}")


def make_label_template(targets_df, gpt_df, png_dir, output_path):
    gpt_map = gpt_df.set_index("review_id").to_dict("index")

    rows = []
    for _, row in targets_df.iterrows():
        rid = row["review_id"]
        gpt_row = gpt_map.get(rid, {})

        png_candidates = list(png_dir.glob(f"B0CTX_*_{rid}_*.png"))
        png_path = str(png_candidates[0]) if png_candidates else ""

        group = row.get("group", "")
        if "normal_fp" in str(group):
            cs_ref = "b0_ct_context_contact_sheet_normal_fp.png"
        else:
            cs_ref = "b0_ct_context_contact_sheet_lesion_safety.png"

        rows.append({
            "ct_context_id": row.get("ct_context_id", ""),
            "review_id": rid,
            "patient_id": row.get("patient_id", ""),
            "group": group,
            "source_group": row.get("source_group", ""),
            "safety_role": row.get("safety_role", ""),
            "b0_visual_label": gpt_row.get("b0_visual_label", ""),
            "b0_confidence": gpt_row.get("b0_confidence", ""),
            "b0_action_recommendation": gpt_row.get("b0_action_recommendation", ""),
            "candidate_score": gpt_row.get("candidate_score", ""),
            "candidate_local_z": row.get("candidate_local_z", ""),
            "candidate_patch_index": row.get("candidate_patch_index", ""),
            "existing_panel_png_path": png_path,
            "contact_sheet_reference": cs_ref,
            "marker_type": "center_only",
            "patch_extent_status": "unknown",
            "ct_context_label": "",
            "ct_context_confidence": "",
            "ct_context_action": "",
            "ct_context_note": "",
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(str(output_path), index=False)

    # ct_context 컬럼이 전부 빈 값인지 확인
    for col in ["ct_context_label", "ct_context_confidence", "ct_context_action", "ct_context_note"]:
        non_empty = df_out[col].dropna().astype(str).str.strip().ne("").sum()
        if non_empty > 0:
            abort(f"[GUARD] {col} 컬럼에 빈 값이 아닌 항목이 있음: {non_empty}건")

    print(f"[OK] {output_path.name}: {len(df_out)}행, ct_context 컬럼 전부 빈 값 확인")
    return len(df_out)


def make_review_pack_json(label_rows, output_path):
    pack = {
        "input_files": {
            "targets_csv": str(TARGETS_CSV),
            "filled_gpt_csv": str(FILLED_GPT_CSV),
            "panels_dir": str(PANELS_DIR),
        },
        "output_files": [
            str(OUTPUT_CONTACT_NORMAL),
            str(OUTPUT_CONTACT_LESION),
            str(OUTPUT_CONTACT_ALL),
            str(OUTPUT_GUIDE),
            str(OUTPUT_TEMPLATE),
            str(OUTPUT_JSON),
        ],
        "validation_result": "PASS",
        "normal_fp_png_count": REQUIRED_NORMAL_FP,
        "lesion_safety_png_count": REQUIRED_LESION_SAFETY,
        "total_png_count": REQUIRED_TOTAL,
        "label_template_rows": label_rows,
        "safety_constraints": [
            "no CT/ROI/mask npy load",
            "no score CSV reload",
            "no scoring/metric/suppression",
            "no existing PNG modification",
            "no current_state.md modification",
            "no stage2_holdout access",
            "no cause determination",
            "no score adjustment",
            "no threshold change",
        ],
        "decisions_not_made": [
            "FP root cause (vessel/pleura/hilar) — requires human visual review",
            "vessel/pleura suppression rule selection",
            "lesion safety overlap assessment",
            "B0 branch continue/stop decision",
        ],
    }
    output_path.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {output_path.name}")


def main():
    print("[INFO] B0 CT-context review pack 생성 시작")

    # 출력 collision 확인
    check_no_existing_outputs()

    # 데이터 로드
    if not TARGETS_CSV.exists():
        abort(f"targets CSV 없음: {TARGETS_CSV}")
    if not FILLED_GPT_CSV.exists():
        abort(f"filled_gpt CSV 없음: {FILLED_GPT_CSV}")
    if not PANELS_DIR.exists():
        abort(f"panels dir 없음: {PANELS_DIR}")

    targets_df = pd.read_csv(str(TARGETS_CSV))
    gpt_df = pd.read_csv(str(FILLED_GPT_CSV))
    png_files = sorted(PANELS_DIR.glob("B0CTX_*.png"),
                       key=lambda x: int(x.stem.split("_")[1]))

    # 입력 검증
    normal_fp_pngs, lesion_safety_pngs = validate_inputs(targets_df, png_files)

    # mtime 기록
    original_mtimes = record_mtimes(png_files)

    # gpt join map
    gpt_map = gpt_df.set_index("review_id").to_dict("index")

    def build_rows_data(pngs):
        result = []
        for p in sorted(pngs, key=lambda x: int(x.stem.split("_")[1])):
            rid = parse_review_id_from_stem(p.stem)
            t_rows = targets_df[targets_df["review_id"] == rid]
            if t_rows.empty:
                abort(f"review_id {rid} not found in targets CSV")
            t_row = t_rows.iloc[0].to_dict()
            gpt_row = gpt_map.get(rid)
            short = len(pngs) >= 20
            result.append({
                "png_path": p,
                "label": make_tile_label(t_row, gpt_row, short=short),
            })
        return result

    # contact sheet 생성
    normal_rows = build_rows_data(normal_fp_pngs)
    lesion_rows = build_rows_data(lesion_safety_pngs)
    all_rows = build_rows_data(png_files)

    make_contact_sheet(
        normal_rows, OUTPUT_CONTACT_NORMAL,
        n_cols=3, tile_img_w=660,
        title_text=(f"B0 CT-context | Normal FP ({REQUIRED_NORMAL_FP}) "
                    "| marker=center-only | no cause determination"),
    )
    make_contact_sheet(
        lesion_rows, OUTPUT_CONTACT_LESION,
        n_cols=4, tile_img_w=494,
        title_text=(f"B0 CT-context | Lesion Safety ({REQUIRED_LESION_SAFETY}) "
                    "| marker=center-only | no suppression decision"),
    )
    make_contact_sheet(
        all_rows, OUTPUT_CONTACT_ALL,
        n_cols=5, tile_img_w=396,
        title_text=(f"B0 CT-context | All 33 Overview "
                    "| see normal/lesion sheets for detail"),
    )

    # review guide
    make_review_guide(OUTPUT_GUIDE)

    # label template
    label_rows = make_label_template(targets_df, gpt_df, PANELS_DIR, OUTPUT_TEMPLATE)

    # mtime 확인 (contact sheet 생성 후)
    verify_mtimes(png_files, original_mtimes)

    # review pack JSON
    make_review_pack_json(label_rows, OUTPUT_JSON)

    print()
    print("[SUMMARY] 생성 파일:")
    for p in [OUTPUT_CONTACT_NORMAL, OUTPUT_CONTACT_LESION, OUTPUT_CONTACT_ALL,
              OUTPUT_GUIDE, OUTPUT_TEMPLATE, OUTPUT_JSON]:
        if p.exists():
            size = p.stat().st_size
            unit = "KB" if size >= 1024 else "B"
            size_display = f"{size // 1024}{unit}" if size >= 1024 else f"{size}{unit}"
            print(f"  {p.name}: {size_display}")
        else:
            print(f"  [MISSING] {p.name}")
    print("[DONE]")


if __name__ == "__main__":
    main()
