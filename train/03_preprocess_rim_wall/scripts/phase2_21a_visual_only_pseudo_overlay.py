"""Phase 2.21a Visual-Only Pseudo Overlay Panel Refresh
- Read-only: phase2_20d overlay table + existing panel PNGs
- Output: new PNGs with visual-only green ellipse + labels
- No array, no score change, no CT/mask loading
"""
import csv
import json
import os
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/home/jinhy/project/lung-ct-anomaly/outputs/mip-postprocess-research-v1")
INPUT_TABLE = BASE / "reports/phase2_20d_visual_only_dry_decision_overlay_table.csv"
OUTPUT_REPORTS = BASE / "reports"
OUTPUT_PACK = BASE / "qa/phase2_21a_visual_only_pseudo_overlay_pack"
OUTPUT_SHEETS = OUTPUT_PACK / "contact_sheets"

OUTPUT_TABLE_CSV = OUTPUT_REPORTS / "phase2_21a_visual_only_pseudo_overlay_table.csv"
OUTPUT_SUMMARY_MD = OUTPUT_REPORTS / "phase2_21a_visual_only_pseudo_overlay_summary.md"
OUTPUT_SUMMARY_JSON = OUTPUT_REPORTS / "phase2_21a_visual_only_pseudo_overlay_summary.json"
OUTPUT_HTML = OUTPUT_PACK / "review_index.html"

# ─── Guard: abort if output already exists ────────────────────────────────────
for p in [OUTPUT_TABLE_CSV, OUTPUT_SUMMARY_MD, OUTPUT_SUMMARY_JSON, OUTPUT_HTML]:
    if p.exists():
        raise FileExistsError(f"이미 존재: {p}\n_v2 suffix를 사용하거나 중단. 덮어쓰기 금지.")
if OUTPUT_PACK.exists():
    raise FileExistsError(f"이미 존재: {OUTPUT_PACK}\n덮어쓰기 금지.")

OUTPUT_PACK.mkdir(parents=True, exist_ok=False)
OUTPUT_SHEETS.mkdir(parents=True, exist_ok=False)

# ─── Read input table ─────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(INPUT_TABLE)))
vessel_positive = [r for r in rows if r["decision_group"] == "vessel_positive_candidate"]
excluded = [r for r in rows if r["decision_group"] != "vessel_positive_candidate"]

assert len(vessel_positive) == 6, f"vessel_positive_candidate != 6: {len(vessel_positive)}"
assert all(r["dry_rule_positive_flag"] == "True" for r in vessel_positive)
assert all(r["mask_generated"] == "False" for r in rows)
assert all(r["score_changed"] == "False" for r in rows)
print(f"[OK] 기대값 검증 통과 - vessel_positive={len(vessel_positive)}, total={len(rows)}")

# ─── Font helper ──────────────────────────────────────────────────────────────
def get_font(size):
    for name in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                 "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"]:
        if os.path.exists(name):
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()

# ─── Draw pseudo overlay on one panel ─────────────────────────────────────────
def draw_pseudo_overlay(src_path: str, dst_path: str, object_id: str,
                        decision_group: str, decision_reason: str):
    img = Image.open(src_path).convert("RGB")
    w, h = img.size

    # Semi-transparent green ellipse overlay
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    margin_x = int(w * 0.22)
    margin_y = int(h * 0.20)
    bbox = [margin_x, margin_y, w - margin_x, h - margin_y]
    draw_ov.ellipse(bbox, fill=(0, 200, 80, 55), outline=(0, 200, 80, 200), width=6)

    base_rgba = img.convert("RGBA")
    merged = Image.alpha_composite(base_rgba, overlay).convert("RGB")

    draw = ImageDraw.Draw(merged)

    # Warning banner background
    banner_h = int(h * 0.085)
    draw.rectangle([0, 0, w, banner_h], fill=(0, 0, 0, 0))
    draw.rectangle([0, 0, w, banner_h], fill=(10, 10, 10))

    f_big = get_font(max(28, int(h / 45)))
    f_med = get_font(max(20, int(h / 65)))
    f_sm = get_font(max(16, int(h / 85)))

    draw.text((12, 6), "PSEUDO MASK — VISUAL ONLY", font=f_big, fill=(50, 255, 120))
    draw.text((12, 6 + int(h / 45) + 4), "NO ARRAY / NO SCORE CHANGE", font=f_med, fill=(220, 220, 60))

    # Bottom info bar
    bar_y = h - int(h * 0.07)
    draw.rectangle([0, bar_y, w, h], fill=(10, 10, 10))
    short_reason = decision_reason[:90] + ("…" if len(decision_reason) > 90 else "")
    draw.text((12, bar_y + 4), f"ID: {object_id}  |  {decision_group}", font=f_med, fill=(200, 200, 200))
    draw.text((12, bar_y + 4 + int(h / 65) + 4), short_reason, font=f_sm, fill=(160, 160, 160))

    merged.save(dst_path, "PNG")
    print(f"  [saved] {Path(dst_path).name}")
    return dst_path


# ─── Task 2: vessel_positive pseudo overlay ───────────────────────────────────
print("\n[Task 2] vessel_positive_candidate pseudo overlay 생성")
overlay_records = []
for r in vessel_positive:
    oid = r["object_id"]
    src_main = r["visual_overlay_main_path"]
    src_supp = r["visual_overlay_supplement_path"]
    dst_main = str(OUTPUT_PACK / f"{oid}_visual_only_pseudo_overlay_main.png")
    dst_supp = str(OUTPUT_PACK / f"{oid}_visual_only_pseudo_overlay_supplement.png")

    draw_pseudo_overlay(src_main, dst_main, oid, r["decision_group"], r["decision_reason"])
    draw_pseudo_overlay(src_supp, dst_supp, oid, r["decision_group"], r["decision_reason"])

    overlay_records.append({
        "object_id": oid,
        "patient_id": r["patient_id"],
        "decision_group": r["decision_group"],
        "dry_rule_positive_flag": r["dry_rule_positive_flag"],
        "dry_rule_cautious_flag": r["dry_rule_cautious_flag"],
        "dry_rule_exclusion_flag": r["dry_rule_exclusion_flag"],
        "visual_pseudo_overlay_created": "True",
        "visual_only": "True",
        "mask_array_created": "False",
        "suppression_weight_created": "False",
        "score_changed": "False",
        "source_main_panel_path": src_main,
        "source_supplement_panel_path": src_supp,
        "pseudo_overlay_main_path": dst_main,
        "pseudo_overlay_supplement_path": dst_supp,
        "overlay_note": "VISUAL ONLY — no array, no score change",
        "reviewer_note": r["reviewer_note"],
    })


# ─── Task 3: excluded group comparison panel ──────────────────────────────────
print("\n[Task 3] excluded group comparison panel 생성")
EXCL_LABEL = {
    "protected_negative": "EXCLUDED — PROTECTED",
    "mixed_or_uncertain": "EXCLUDED — MIXED/UNCERTAIN",
    "cautious_vessel_candidate": "CAUTIOUS — NOT POSITIVE",
}

for r in excluded:
    oid = r["object_id"]
    src_main = r["visual_overlay_main_path"]
    label = EXCL_LABEL.get(r["decision_group"], "EXCLUDED")

    img = Image.open(src_main).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    banner_h = int(h * 0.085)
    draw.rectangle([0, 0, w, banner_h], fill=(40, 0, 0))
    f_big = get_font(max(28, int(h / 45)))
    f_med = get_font(max(20, int(h / 65)))
    draw.text((12, 6), label, font=f_big, fill=(255, 80, 80))
    draw.text((12, 6 + int(h / 45) + 4), "NO PSEUDO OVERLAY", font=f_med, fill=(200, 200, 200))

    bar_y = h - int(h * 0.07)
    draw.rectangle([0, bar_y, w, h], fill=(20, 20, 20))
    draw.text((12, bar_y + 4), f"ID: {oid}  |  {r['decision_group']}", font=f_med, fill=(180, 180, 180))

    dst = str(OUTPUT_PACK / f"{oid}_excluded_comparison.png")
    img.save(dst, "PNG")
    print(f"  [saved] {Path(dst).name}")

    overlay_records.append({
        "object_id": oid,
        "patient_id": r["patient_id"],
        "decision_group": r["decision_group"],
        "dry_rule_positive_flag": r["dry_rule_positive_flag"],
        "dry_rule_cautious_flag": r["dry_rule_cautious_flag"],
        "dry_rule_exclusion_flag": r["dry_rule_exclusion_flag"],
        "visual_pseudo_overlay_created": "False",
        "visual_only": "True",
        "mask_array_created": "False",
        "suppression_weight_created": "False",
        "score_changed": "False",
        "source_main_panel_path": r["visual_overlay_main_path"],
        "source_supplement_panel_path": r["visual_overlay_supplement_path"],
        "pseudo_overlay_main_path": "",
        "pseudo_overlay_supplement_path": "",
        "overlay_note": label,
        "reviewer_note": r["reviewer_note"],
    })


# ─── Task 4: CSV ──────────────────────────────────────────────────────────────
print("\n[Task 4] CSV 생성")
COLS = [
    "object_id", "patient_id", "decision_group",
    "dry_rule_positive_flag", "dry_rule_cautious_flag", "dry_rule_exclusion_flag",
    "visual_pseudo_overlay_created", "visual_only",
    "mask_array_created", "suppression_weight_created", "score_changed",
    "source_main_panel_path", "source_supplement_panel_path",
    "pseudo_overlay_main_path", "pseudo_overlay_supplement_path",
    "overlay_note", "reviewer_note",
]
with open(OUTPUT_TABLE_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(overlay_records)
print(f"  [saved] {OUTPUT_TABLE_CSV.name}")


# ─── Task 6: contact sheets ───────────────────────────────────────────────────
def make_contact_sheet(img_paths, title, dst_path, cols=3, thumb_w=700, thumb_h=507):
    imgs = []
    for p in img_paths:
        if p and os.path.exists(p):
            imgs.append(Image.open(p).convert("RGB").resize((thumb_w, thumb_h)))
    if not imgs:
        return
    rows_n = (len(imgs) + cols - 1) // cols
    sheet_w = cols * thumb_w
    sheet_h = rows_n * thumb_h + 60
    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    f = get_font(28)
    draw.text((10, 10), title, font=f, fill=(220, 220, 220))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet.paste(im, (c * thumb_w, 60 + r * thumb_h))
    sheet.save(dst_path, "PNG")
    print(f"  [saved] {Path(dst_path).name}")


print("\n[Task 6] contact sheet 생성")
vp_mains = [r["pseudo_overlay_main_path"] for r in overlay_records if r["visual_pseudo_overlay_created"] == "True"]
make_contact_sheet(
    vp_mains,
    "vessel_positive_visual_pseudo_overlay — VISUAL ONLY, NO ARRAY, NO SCORE CHANGE",
    str(OUTPUT_SHEETS / "vessel_positive_visual_pseudo_overlay_contact_sheet.png"),
)

excl_mains = [str(OUTPUT_PACK / f"{r['object_id']}_excluded_comparison.png") for r in overlay_records if r["visual_pseudo_overlay_created"] == "False"]
make_contact_sheet(
    excl_mains,
    "excluded_groups_comparison — PROTECTED / MIXED / CAUTIOUS",
    str(OUTPUT_SHEETS / "excluded_groups_comparison_contact_sheet.png"),
)


# ─── Task 5: HTML review index ────────────────────────────────────────────────
print("\n[Task 5] HTML review index 생성")

def rel(p):
    return os.path.relpath(p, OUTPUT_PACK) if p else ""

vp_rows = [r for r in overlay_records if r["visual_pseudo_overlay_created"] == "True"]
ex_rows = [r for r in overlay_records if r["visual_pseudo_overlay_created"] == "False"]

html_vp = ""
for r in vp_rows:
    m = rel(r["pseudo_overlay_main_path"])
    s = rel(r["pseudo_overlay_supplement_path"])
    html_vp += f"""
    <div class="card">
      <h3>{r['object_id']} ({r['patient_id']})</h3>
      <p class="tag green">vessel_positive_candidate</p>
      <p>{r['reviewer_note']}</p>
      <div class="panels">
        <figure><img src="{m}"><figcaption>Main</figcaption></figure>
        <figure><img src="{s}"><figcaption>Supplement</figcaption></figure>
      </div>
    </div>"""

html_ex = ""
for r in ex_rows:
    src_m = rel(r["source_main_panel_path"])
    label = r["overlay_note"]
    color = "red" if "EXCLUDED" in label else "yellow"
    html_ex += f"""
    <div class="card">
      <h3>{r['object_id']} ({r['patient_id']})</h3>
      <p class="tag {color}">{label}</p>
      <p>{r['decision_group']} — {r['reviewer_note']}</p>
      <figure><img src="{rel(r['source_main_panel_path'])}"><figcaption>Source (no overlay)</figcaption></figure>
    </div>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phase 2.21a Visual-Only Pseudo Overlay Review</title>
<style>
body{{font-family:monospace;background:#111;color:#ddd;margin:20px}}
h1{{color:#6f6}}
h2{{color:#adf;border-bottom:1px solid #444;padding-bottom:6px}}
.warning{{background:#2a1500;border:2px solid #f80;padding:12px;border-radius:6px;margin:12px 0}}
.card{{background:#1a1a1a;border:1px solid #333;border-radius:6px;padding:14px;margin:12px 0}}
.card h3{{color:#8cf;margin:0 0 6px}}
.panels{{display:flex;gap:10px;flex-wrap:wrap}}
figure{{margin:0}}
figure img{{max-width:520px;border:1px solid #555;display:block}}
figcaption{{font-size:11px;color:#888;margin-top:3px}}
.tag{{display:inline-block;padding:2px 8px;border-radius:4px;font-weight:bold;font-size:12px}}
.green{{background:#0a3a0a;color:#6f6;border:1px solid #3a8}}
.red{{background:#3a0a0a;color:#f88;border:1px solid #a44}}
.yellow{{background:#2a2500;color:#fc8;border:1px solid #a84}}
.next{{background:#0a1a2a;border:1px solid #48f;padding:12px;border-radius:6px;margin:12px 0}}
</style>
</head>
<body>
<h1>Phase 2.21a Visual-Only Pseudo Overlay</h1>

<div class="warning">
<b>⚠ VISUAL ONLY</b> — This overlay is NOT a real vessel mask.<br>
No CT/ROI/mask data was loaded. No array was created. No score was changed.<br>
This is a reviewer guidance marker only. Real pseudo mask generation requires separate approval.
</div>

<h2>1. Purpose</h2>
<p>
Phase 2.21a adds a visual-only green ellipse to 6 vessel_positive_candidate panels for reviewer orientation.<br>
Protected, mixed, and cautious objects are shown for comparison only — no pseudo overlay was applied to them.
</p>

<h2>2. Vessel Positive Pseudo Overlay (n={len(vp_rows)})</h2>
{html_vp}

<h2>3. Excluded Comparison (n={len(ex_rows)})</h2>
{html_ex}

<h2>4. Contact Sheets</h2>
<figure><img src="contact_sheets/vessel_positive_visual_pseudo_overlay_contact_sheet.png" style="max-width:100%"><figcaption>vessel_positive contact sheet</figcaption></figure>
<figure><img src="contact_sheets/excluded_groups_comparison_contact_sheet.png" style="max-width:100%"><figcaption>excluded comparison contact sheet</figcaption></figure>

<div class="next">
<b>Next Steps</b><br>
1. Human reviewer must inspect all 6 pseudo overlay panels above.<br>
2. If accepted, a real pseudo mask experiment may be planned — but only after separate approval.<br>
3. Stage1_dev only. Stage2_holdout is forbidden at this stage.<br>
4. suppression_weight / score adjustment is still forbidden. Do not proceed without approval.
</div>

<p style="color:#555;font-size:11px">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Phase 2.21a</p>
</body>
</html>"""

with open(OUTPUT_HTML, "w") as f:
    f.write(html)
print(f"  [saved] {OUTPUT_HTML.name}")


# ─── Task 7: summary MD/JSON ──────────────────────────────────────────────────
print("\n[Task 7] summary MD/JSON 생성")

vp_ids = [r["object_id"] for r in vp_rows]
ex_ids = [r["object_id"] for r in ex_rows]

md = f"""# Phase 2.21a Visual-Only Pseudo Overlay Summary

## 목적
Phase 2.21a는 vessel_positive_candidate 6개에 대해 visual-only pseudo overlay를 추가했다.

## 중요 사항
- **이 overlay는 실제 vessel mask가 아니다.**
- CT/ROI/mask 원본을 로드하지 않았다.
- mask array를 생성하지 않았다.
- suppression_weight를 계산하지 않았다.
- score CSV를 수정하지 않았다.
- 기존 panel PNG를 수정하지 않았다 (새 파일로 저장).
- stage2_holdout를 사용하지 않았다.

## vessel_positive_candidate (n=6)
{chr(10).join(f'- {oid}' for oid in vp_ids)}

## excluded groups (n={len(ex_ids)})
{chr(10).join(f'- {r["object_id"]} [{r["decision_group"]}]' for r in ex_rows)}

## 금지 작업 수행 여부
- CT/ROI/mask 로드: **False**
- 새 MIP 계산: **False**
- vessel soft mask 생성: **False**
- pseudo mask npy/csv: **False**
- suppression_weight 계산: **False**
- score CSV 수정: **False**
- patch CSV 수정: **False**
- 기존 panel PNG 수정: **False**
- stage2_holdout 사용: **False**

## 다음 단계 권고
Review Phase 2.21a visual-only pseudo overlay.
If accepted, next step may plan a real pseudo mask experiment,
but only after separate approval and still stage1_dev-only.
Do not adjust scores.
"""

with open(OUTPUT_SUMMARY_MD, "w") as f:
    f.write(md)
print(f"  [saved] {OUTPUT_SUMMARY_MD.name}")

summary_json = {
    "phase": "2.21a",
    "n_vessel_positive_overlay_created": len(vp_ids),
    "n_excluded_objects": len(ex_ids),
    "vessel_positive_ids": vp_ids,
    "excluded_ids": ex_ids,
    "visual_only": True,
    "mask_array_created": False,
    "suppression_weight_created": False,
    "score_changed": False,
    "ct_loaded": False,
    "stage2_holdout_used": False,
    "recommended_next_step": "Review Phase 2.21a visual-only pseudo overlay. If accepted, next step may plan a real pseudo mask experiment, but only after separate approval and still stage1_dev-only. Do not adjust scores.",
    "forbidden_actions": [
        "CT/ROI/mask load",
        "new MIP calculation",
        "real vessel soft mask generation",
        "pseudo mask npy/csv creation",
        "vessel boundary tracing",
        "segmentation algorithm",
        "subtraction mask creation",
        "suppression_weight calculation",
        "adjusted score calculation",
        "score CSV modification",
        "patch CSV modification",
        "existing panel PNG overwrite",
        "stage2_holdout usage",
        "Phase 3 progression",
        "new package installation",
    ],
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

with open(OUTPUT_SUMMARY_JSON, "w") as f:
    json.dump(summary_json, f, indent=2, ensure_ascii=False)
print(f"  [saved] {OUTPUT_SUMMARY_JSON.name}")

print("\n[완료] Phase 2.21a 모든 작업 완료")
