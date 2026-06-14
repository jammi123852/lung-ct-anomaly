"""
make_dev_safe_mixed_error_review_pack.py

Generates review pack from existing QA PNGs:
  - contact_sheet_normal_fp.png
  - contact_sheet_lesion_fn_p99.png
  - contact_sheet_all_54.png
  - review_guide.md
  - review_labels_template.csv

Usage:
  python scripts/make_dev_safe_mixed_error_review_pack.py --run

Safety:
  - Reads existing PNGs as read-only (resize in memory only, mtime unchanged).
  - Aborts if any output file already exists (no overwrite).
  - Does NOT load CT/ROI/mask npy, score CSVs, model, or checkpoints.
  - Does NOT recalculate scores or metrics.
  - Does NOT assign cause labels automatically.
"""

import argparse
import os
import re
import sys

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/home/jinhy/project/lung-ct-anomaly"
QA_DIR = os.path.join(PROJECT_ROOT, "qa", "dev_safe_mixed_error_visual_qa")
MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "mixed_cohort_patient_metrics",
    "dev_safe_v1_visual_qa_manifest.csv",
)

OUTPUT_FILES = {
    "contact_normal": os.path.join(QA_DIR, "contact_sheet_normal_fp.png"),
    "contact_lesion": os.path.join(QA_DIR, "contact_sheet_lesion_fn_p99.png"),
    "contact_all": os.path.join(QA_DIR, "contact_sheet_all_54.png"),
    "guide": os.path.join(QA_DIR, "review_guide.md"),
    "template": os.path.join(QA_DIR, "review_labels_template.csv"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_candidate_reason(note: str) -> str:
    """Extract abbreviated candidate_reason from the note field."""
    if pd.isna(note):
        return ""
    m = re.search(r"candidate_reason=([^\s;\"]+)", str(note))
    if not m:
        return ""
    raw = m.group(1)
    # Abbreviate to short labels
    abbrev = {
        "near_patient_p95_high_score": "near_p95",
        "top1pct_representative": "top1pct",
        "max_patch_outlier_check": "max",
        "lesion_positive_low_score": "low",
        "lesion_positive_high_score": "high",
    }
    return abbrev.get(raw, raw)


def short_patient_id(pid: str) -> str:
    """Shorten long patient IDs for tile label display."""
    if len(pid) <= 20:
        return pid
    # Take first 8 chars + '...' + last 6 chars
    return pid[:8] + "..." + pid[-6:]


def get_font(size: int):
    """Return a PIL font; fall back to default if no TTF found."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size)
        except Exception:
            return ImageFont.load_default()


def make_tile(row: pd.Series, png_path: str, tile_w: int, tile_h: int, label_h: int, font) -> Image.Image:
    """
    Load PNG, resize to tile_w x (tile_h - label_h), paste onto white canvas,
    draw text label at bottom.
    Original PNG file is read-only; mtime is not modified.
    """
    img_h = tile_h - label_h
    try:
        img = Image.open(png_path).convert("RGB")
        img = img.resize((tile_w, img_h), Image.LANCZOS)
    except Exception as e:
        img = Image.new("RGB", (tile_w, img_h), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        draw.text((4, 4), f"LOAD ERR\n{str(e)[:30]}", fill=(180, 0, 0), font=font)

    canvas = Image.new("RGB", (tile_w, tile_h), color=(255, 255, 255))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    # Draw separator line
    draw.line([(0, img_h), (tile_w - 1, img_h)], fill=(160, 160, 160), width=1)

    cand_reason = parse_candidate_reason(row.get("note", ""))
    score_val = row.get("candidate_score", "")
    try:
        score_val = f"{float(score_val):.2f}"
    except Exception:
        score_val = str(score_val)

    visual_reason = str(row.get("visual_reason_to_check", ""))
    # Truncate long visual_reason
    if len(visual_reason) > 28:
        visual_reason = visual_reason[:26] + ".."

    pid_short = short_patient_id(str(row.get("patient_id", "")))
    review_id = str(row.get("review_id", ""))
    thr = str(row.get("threshold_level", ""))

    lines = [
        f"{review_id}  {pid_short}",
        f"score:{score_val}  [{cand_reason}]  thr:{thr}",
        f"{visual_reason}",
    ]

    y = img_h + 2
    line_h = label_h // 3
    for line in lines:
        draw.text((2, y), line, fill=(20, 20, 20), font=font)
        y += line_h

    return canvas


def check_collision():
    """Abort if any output file already exists."""
    found = [p for p in OUTPUT_FILES.values() if os.path.exists(p)]
    if found:
        print("[ABORT] The following output files already exist (no overwrite):")
        for f in found:
            print(f"  {f}")
        sys.exit(1)


def validate_inputs(df: pd.DataFrame):
    """Validate manifest and PNG files. Abort if any check fails."""
    errors = []

    # Check PNG count
    png_files = sorted(
        [f for f in os.listdir(QA_DIR) if f.endswith(".png")]
    )
    png_rids = [os.path.splitext(f)[0] for f in png_files]
    expected_rids = [f"R{i:03d}" for i in range(1, 55)]
    if sorted(png_rids) != sorted(expected_rids):
        errors.append(f"PNG files mismatch. Found: {png_rids}")

    # Check 0-byte PNGs
    zero_bytes = [f for f in png_files if os.path.getsize(os.path.join(QA_DIR, f)) == 0]
    if zero_bytes:
        errors.append(f"0-byte PNGs found: {zero_bytes}")

    # Check manifest row count
    if len(df) != 54:
        errors.append(f"Manifest row count = {len(df)}, expected 54")

    # Check review_id <-> PNG 1:1
    manifest_rids = set(df["review_id"].tolist())
    png_rid_set = set(expected_rids)
    if manifest_rids != png_rid_set:
        errors.append(f"review_id/PNG mismatch. manifest only: {manifest_rids - png_rid_set}, png only: {png_rid_set - manifest_rids}")

    # Check group counts
    normal_count = (df["group"] == "normal_fp").sum()
    lesion_count = (df["group"] == "lesion_fn_p99").sum()
    if normal_count != 30:
        errors.append(f"normal_fp count = {normal_count}, expected 30")
    if lesion_count != 24:
        errors.append(f"lesion_fn_p99 count = {lesion_count}, expected 24")

    # Check stage2_holdout_flag all 0
    if (df["stage2_holdout_flag"] != 0).any():
        errors.append("stage2_holdout_flag has non-zero value(s)")

    # Check no resnet50 path strings in manifest
    manifest_str = df.to_string()
    if "resnet50" in manifest_str.lower() or "ResNet50" in manifest_str:
        errors.append("ResNet50 path string found in manifest")

    if errors:
        print("[ABORT] Input validation failed:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    print(f"[OK] Input validation passed: 54 PNG, 0 zero-bytes, 54 manifest rows, "
          f"normal_fp={normal_count}, lesion_fn_p99={lesion_count}, "
          f"stage2_holdout_flag all 0, no ResNet50 path")


# ---------------------------------------------------------------------------
# Contact sheet builders
# ---------------------------------------------------------------------------

def build_contact_sheet_normal_fp(df: pd.DataFrame, out_path: str):
    """
    normal_fp 30 tiles: group by patient_id, 3 tiles per row.
    """
    sub = df[df["group"] == "normal_fp"].copy()
    # Sort by patient_id then review_id to keep 3-per-patient grouping stable
    sub = sub.sort_values(["patient_id", "review_id"]).reset_index(drop=True)

    tile_w = 300
    img_h_per_tile = 300
    label_h = 54  # 3 lines x 18px
    tile_h = img_h_per_tile + label_h
    cols = 3
    rows = (len(sub) + cols - 1) // cols  # 10 rows for 30 tiles

    font = get_font(11)
    gap = 2
    sheet_w = cols * tile_w + (cols + 1) * gap
    sheet_h = rows * tile_h + (rows + 1) * gap

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(100, 100, 100))

    for idx, row in sub.iterrows():
        rid = row["review_id"]
        png_path = os.path.join(QA_DIR, f"{rid}.png")
        tile = make_tile(row, png_path, tile_w, tile_h, label_h, font)
        col_pos = idx % cols
        row_pos = idx // cols
        x = gap + col_pos * (tile_w + gap)
        y = gap + row_pos * (tile_h + gap)
        sheet.paste(tile, (x, y))

    # Title bar at top
    # We shift everything down 30px for title
    title_h = 30
    final = Image.new("RGB", (sheet_w, sheet_h + title_h), color=(50, 50, 80))
    draw = ImageDraw.Draw(final)
    title_font = get_font(14)
    draw.text((gap, 6), "normal_fp contact sheet  (dev_safe_v1 QA)  30 tiles / 10 patients / 3 each", fill=(220, 220, 255), font=title_font)
    final.paste(sheet, (0, title_h))

    final.save(out_path)
    print(f"[OK] Saved: {out_path}  size={final.size}")


def build_contact_sheet_lesion_fn_p99(df: pd.DataFrame, out_path: str):
    """
    lesion_fn_p99 24 tiles: 2 tiles per patient (low/high), arranged 2-per-row.
    """
    sub = df[df["group"] == "lesion_fn_p99"].copy()
    # Sort: patient then low first (low score = smaller candidate_score)
    sub = sub.sort_values(["patient_id", "candidate_score"]).reset_index(drop=True)

    tile_w = 340
    img_h_per_tile = 340
    label_h = 60  # 3 lines
    tile_h = img_h_per_tile + label_h
    cols = 2  # low | high per row
    rows = len(sub) // cols  # 12 rows

    font = get_font(11)
    gap = 2
    sheet_w = cols * tile_w + (cols + 1) * gap
    sheet_h = rows * tile_h + (rows + 1) * gap

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(100, 80, 80))

    for idx, row in sub.iterrows():
        rid = row["review_id"]
        png_path = os.path.join(QA_DIR, f"{rid}.png")
        tile = make_tile(row, png_path, tile_w, tile_h, label_h, font)
        col_pos = idx % cols
        row_pos = idx // cols
        x = gap + col_pos * (tile_w + gap)
        y = gap + row_pos * (tile_h + gap)
        sheet.paste(tile, (x, y))

    # Title bar
    title_h = 30
    final = Image.new("RGB", (sheet_w, sheet_h + title_h), color=(80, 50, 50))
    draw = ImageDraw.Draw(final)
    title_font = get_font(14)
    draw.text((gap, 6), "lesion_fn_p99 contact sheet  (dev_safe_v1 QA)  24 tiles / 12 patients / low+high per row  thr=p99", fill=(255, 210, 210), font=title_font)
    final.paste(sheet, (0, title_h))

    final.save(out_path)
    print(f"[OK] Saved: {out_path}  size={final.size}")


def build_contact_sheet_all_54(df: pd.DataFrame, out_path: str):
    """
    All 54 tiles: overview, 9 per row.
    Tiles are small (overview use only).
    """
    sub = df.sort_values("review_id").reset_index(drop=True)

    tile_w = 160
    img_h_per_tile = 160
    label_h = 28
    tile_h = img_h_per_tile + label_h
    cols = 9
    rows = (len(sub) + cols - 1) // cols  # 6 rows

    font = get_font(9)
    gap = 2
    sheet_w = cols * tile_w + (cols + 1) * gap
    sheet_h = rows * tile_h + (rows + 1) * gap

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(80, 80, 80))

    for idx, row in sub.iterrows():
        rid = row["review_id"]
        png_path = os.path.join(QA_DIR, f"{rid}.png")

        img_h_tile = tile_h - label_h
        try:
            img = Image.open(png_path).convert("RGB")
            img = img.resize((tile_w, img_h_tile), Image.LANCZOS)
        except Exception:
            img = Image.new("RGB", (tile_w, img_h_tile), color=(180, 180, 180))

        canvas = Image.new("RGB", (tile_w, tile_h), color=(255, 255, 255))
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.line([(0, img_h_tile), (tile_w - 1, img_h_tile)], fill=(160, 160, 160), width=1)
        grp_short = "FP" if row.get("group") == "normal_fp" else "FN"
        try:
            score_str = f"{float(row.get('candidate_score', '')):.1f}"
        except Exception:
            score_str = str(row.get("candidate_score", ""))
        label_line = f"{rid} {grp_short} {score_str}"
        draw.text((2, img_h_tile + 2), label_line, fill=(20, 20, 20), font=font)

        col_pos = idx % cols
        row_pos = idx // cols
        x = gap + col_pos * (tile_w + gap)
        y = gap + row_pos * (tile_h + gap)
        sheet.paste(canvas, (x, y))

    # Title bar
    title_h = 36
    final = Image.new("RGB", (sheet_w, sheet_h + title_h), color=(50, 50, 50))
    draw = ImageDraw.Draw(final)
    title_font = get_font(13)
    draw.text((gap, 4), "ALL 54 overview  (dev_safe_v1 QA)  [NOTE: overview only - too small for detail judgment]", fill=(240, 240, 180), font=title_font)
    draw.text((gap, 20), "For detail review: use contact_sheet_normal_fp.png / contact_sheet_lesion_fn_p99.png / original PNG files", fill=(200, 200, 200), font=get_font(10))
    final.paste(sheet, (0, title_h))

    final.save(out_path)
    print(f"[OK] Saved: {out_path}  size={final.size}")


# ---------------------------------------------------------------------------
# review_guide.md
# ---------------------------------------------------------------------------

GUIDE_TEXT = """\
# dev_safe mixed error visual QA review guide

## 1. Scope
- dev_safe only
- normal FP 10 patients / 30 PNG
- lesion FN p99 12 patients / 24 PNG
- stage2_holdout 0
- generated from existing PNG only
- no CT/ROI/mask reload
- no scoring/model/training

## 2. How to review normal FP
- Green ROI contour in the PNG
- Blue patch marker showing the candidate patch location
- No red lesion contour (these are normal patients)
- Observation focus (not a conclusion):
  - vessel / hilar / pleural / chest_wall / bronchus / diaphragm / base / artifact
- Do NOT conclude cause at this stage. Record in review_labels_template.csv after visual inspection.

## 3. How to review lesion FN
- Red lesion contour shows labeled lesion location
- Blue marker shows candidate patch location
- Compare low-score and high-score candidates from the same patient (two tiles per row)
- Observation focus (not a conclusion):
  - small lesion
  - lesion only partially inside ROI
  - low contrast between lesion and background
  - nearby normal high-contrast structure scored higher
  - mismatch between marker and lesion contour
- Do NOT conclude cause at this stage. Record in review_labels_template.csv after visual inspection.

## 4. Contact sheets
- normal_fp sheet:   qa/dev_safe_mixed_error_visual_qa/contact_sheet_normal_fp.png
- lesion_fn_p99 sheet: qa/dev_safe_mixed_error_visual_qa/contact_sheet_lesion_fn_p99.png
- all_54 sheet (overview): qa/dev_safe_mixed_error_visual_qa/contact_sheet_all_54.png
  NOTE: all_54 is an overview only. Tiles are small and not suitable for detail judgment.
  For detail review: use the normal/fn split sheets or the original PNG files in this folder.

## 5. Review label options
- normal_fp_candidate_type (fill in human_label column):
  - vessel_like
  - pleura_or_chest_wall
  - hilar_or_mediastinal
  - airway_or_bronchus
  - diaphragm_or_base
  - artifact_or_noise
  - unclear

- lesion_fn_candidate_type (fill in human_label column):
  - small_lesion
  - low_contrast_lesion
  - partial_roi_coverage
  - lesion_marker_mismatch
  - non_lesion_high_score_elsewhere
  - unclear

- human_confidence options: high / medium / low
- final_use options: include / exclude / needs_second_view

## 6. Do not conclude yet
- This guide is a review preparation document, not a cause-determination document.
- current_state.md update should be done AFTER human review labels are filled in.
- Filling in review_labels_template.csv is the next step.
"""


def build_review_guide(out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(GUIDE_TEXT)
    print(f"[OK] Saved: {out_path}")


# ---------------------------------------------------------------------------
# review_labels_template.csv
# ---------------------------------------------------------------------------

def build_review_labels_template(df: pd.DataFrame, out_path: str):
    """
    Build template CSV with all manifest metadata columns plus empty human columns.
    human_label, human_confidence, human_note, final_use are intentionally empty.
    suggested_review_focus is observation guidance only (not a cause judgment).
    """
    rows = []
    for _, row in df.iterrows():
        rid = row["review_id"]
        group = row["group"]
        pid = row["patient_id"]
        cand_reason = parse_candidate_reason(row.get("note", ""))

        # suggested_review_focus: observation guidance only
        if group == "normal_fp":
            if cand_reason == "near_p95":
                focus = "check vessel/pleura/hilar/airway/diaphragm/artifact (near p95 score: suspected normal high-score structure, confirm visually)"
            elif cand_reason == "top1pct":
                focus = "check vessel/pleura/hilar/airway/diaphragm/artifact (top-1pct representative: confirm what structure scored high)"
            elif cand_reason == "max":
                focus = "check vessel/pleura/hilar/airway/diaphragm/artifact (max outlier patch: confirm if artifact or anatomical structure)"
            else:
                focus = "check vessel/pleura/hilar/airway/diaphragm/artifact"
        else:  # lesion_fn_p99
            if cand_reason == "low":
                focus = "check small/low_contrast/partial_roi/marker_mismatch (low-score lesion patch: confirm if lesion is present and why score is low)"
            elif cand_reason == "high":
                focus = "check non_lesion_high_score/partial_roi/marker_mismatch (high-score lesion patch: confirm if marker aligns with lesion contour)"
            else:
                focus = "check small/low_contrast/partial_roi/marker_mismatch/non_lesion_high_score"

        png_path = f"qa/dev_safe_mixed_error_visual_qa/{rid}.png"

        rows.append({
            "review_id": rid,
            "patient_id": pid,
            "group": group,
            "error_type": row.get("error_type", ""),
            "candidate_score": row.get("candidate_score", ""),
            "candidate_local_z": row.get("candidate_local_z", ""),
            "candidate_patch_index": row.get("candidate_patch_index", ""),
            "visual_reason_to_check": row.get("visual_reason_to_check", ""),
            "png_path": png_path,
            "suggested_review_focus": focus,
            "human_label": "",
            "human_confidence": "",
            "human_note": "",
            "final_use": "",
        })

    out_df = pd.DataFrame(rows, columns=[
        "review_id", "patient_id", "group", "error_type",
        "candidate_score", "candidate_local_z", "candidate_patch_index",
        "visual_reason_to_check", "png_path", "suggested_review_focus",
        "human_label", "human_confidence", "human_note", "final_use",
    ])
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Saved: {out_path}  rows={len(out_df)}")

    # Verify human columns are empty
    for col in ["human_label", "human_confidence", "human_note", "final_use"]:
        non_empty = out_df[col].dropna().astype(str).str.strip().ne("").sum()
        if non_empty > 0:
            print(f"[WARN] Column '{col}' has {non_empty} non-empty values — check logic")
        else:
            print(f"[OK] Column '{col}' is fully empty (correct)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate dev_safe mixed error review pack")
    parser.add_argument("--run", action="store_true", help="Actually generate output files")
    args = parser.parse_args()

    if not args.run:
        print("Dry-run mode. Use --run to generate output files.")
        print("Output files that would be created:")
        for k, v in OUTPUT_FILES.items():
            print(f"  {k}: {v}")
        return

    print("=== make_dev_safe_mixed_error_review_pack.py ===")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"QA dir       : {QA_DIR}")
    print(f"Manifest     : {MANIFEST_PATH}")
    print()

    # Step 1: Collision check
    print("[Step 1] Collision check ...")
    check_collision()
    print("[OK] No collision detected")
    print()

    # Step 2: Load manifest
    print("[Step 2] Loading manifest ...")
    df = pd.read_csv(MANIFEST_PATH)
    print(f"[OK] Manifest loaded: {len(df)} rows")
    print()

    # Step 3: Input validation
    print("[Step 3] Input validation ...")
    validate_inputs(df)
    print()

    # Step 4: Build contact sheet normal_fp
    print("[Step 4] Building contact_sheet_normal_fp.png ...")
    build_contact_sheet_normal_fp(df, OUTPUT_FILES["contact_normal"])
    print()

    # Step 5: Build contact sheet lesion_fn_p99
    print("[Step 5] Building contact_sheet_lesion_fn_p99.png ...")
    build_contact_sheet_lesion_fn_p99(df, OUTPUT_FILES["contact_lesion"])
    print()

    # Step 6: Build contact sheet all 54
    print("[Step 6] Building contact_sheet_all_54.png ...")
    build_contact_sheet_all_54(df, OUTPUT_FILES["contact_all"])
    print()

    # Step 7: Build review guide
    print("[Step 7] Building review_guide.md ...")
    build_review_guide(OUTPUT_FILES["guide"])
    print()

    # Step 8: Build review labels template
    print("[Step 8] Building review_labels_template.csv ...")
    build_review_labels_template(df, OUTPUT_FILES["template"])
    print()

    print("=== All done ===")
    print("Generated files:")
    for k, v in OUTPUT_FILES.items():
        if os.path.exists(v):
            size = os.path.getsize(v)
            print(f"  {v}  ({size:,} bytes)")
        else:
            print(f"  MISSING: {v}")


if __name__ == "__main__":
    main()
