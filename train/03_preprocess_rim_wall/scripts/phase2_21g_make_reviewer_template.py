"""
Phase 2.21g - Object-Level Reviewer Label Template 생성 스크립트
입력: 21d/21e objects CSV, panel png 경로
출력: CSV, guide MD, summary JSON, review_index HTML
"""

import csv
import json
import os
from pathlib import Path

BASE = Path("/home/jinhy/project/lung-ct-anomaly/outputs/mip-postprocess-research-v1")
REPORTS = BASE / "reports"
QA_21D = BASE / "qa" / "phase2_21d_axis_corrected_cor_sag_mip_review" / "panels"
QA_21E = BASE / "qa" / "phase2_21e_inner_roi_margin_sensitivity_review" / "panels"
OUT_DIR = BASE / "qa" / "phase2_21g_object_level_reviewer_label_template"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 입력 CSV 경로 ──────────────────────────────────────────────────────────────
CSV_21D = REPORTS / "phase2_21d_axis_corrected_cor_sag_mip_review_objects.csv"
CSV_21E = REPORTS / "phase2_21e_inner_roi_margin_sensitivity_objects.csv"

# ── 작업 1: 21d/21e 병합 ───────────────────────────────────────────────────────
COLS_21D = [
    "object_id", "patient_id", "safe_id",
    "z_min", "z_max",
    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
    "object_eq_diameter_mm", "object_depth_mm",
    "cor_primary_mm_21d", "sag_primary_mm_21d",
    "cor_secondary_mm_list_21d", "sag_secondary_mm_list_21d",
]

COLS_21E = [
    "centroid_removed_by_inner1_proxy", "centroid_removed_by_inner2_proxy",
    "bbox_roi_original_fill_ratio", "bbox_inner1_fill_ratio", "bbox_inner2_fill_ratio",
    "bbox_margin_loss_inner1_proxy", "bbox_margin_loss_inner2_proxy",
    "min_distance_to_roi_boundary_px",
    "safety_risk_if_inner1_flag", "safety_risk_if_inner2_flag",
    "safety_risk_if_inner1_reason", "safety_risk_if_inner2_reason",
    "boundary_contact_hint", "inner_roi_sensitive_hint",
    "lesion_safety_risk_hint", "candidate_for_margin_review",
    "reviewer_priority",
]

REVIEWER_LABEL_COLS = [
    "reviewer_vessel_like_flag",
    "reviewer_blob_like_flag",
    "reviewer_mixed_contact_flag",
    "reviewer_boundary_risk_flag",
    "reviewer_overmerge_flag",
    "reviewer_too_small_object_flag",
    "reviewer_too_large_object_flag",
    "reviewer_protected_flag",
    "reviewer_uncertain_flag",
    "reviewer_best_coronal_thickness_mm",
    "reviewer_best_sagittal_thickness_mm",
    "reviewer_final_group",
    "reviewer_note",
]

PANEL_COLS = [
    "phase2_21d_main_panel_path",
    "phase2_21d_supplement_panel_path",
    "phase2_21d_zoom_panel_path",
    "phase2_21e_margin_panel_path",
]


def load_csv_as_dict(path, key_col="object_id"):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row[key_col]] = row
    return rows


rows_21d = load_csv_as_dict(CSV_21D)
rows_21e = load_csv_as_dict(CSV_21E)

object_ids = list(rows_21d.keys())
assert set(object_ids) == set(rows_21e.keys()), "21d/21e object_id 불일치"

merged = []
for oid in object_ids:
    r = {}
    for c in COLS_21D:
        r[c] = rows_21d[oid].get(c, "")
    for c in COLS_21E:
        r[c] = rows_21e[oid].get(c, "")

    r["phase2_21d_main_panel_path"]       = str(QA_21D / f"{oid}_main.png")
    r["phase2_21d_supplement_panel_path"] = str(QA_21D / f"{oid}_supplement.png")
    r["phase2_21d_zoom_panel_path"]       = str(QA_21D / f"{oid}_zoom.png")
    r["phase2_21e_margin_panel_path"]     = str(QA_21E / f"{oid}_margin.png")

    for c in REVIEWER_LABEL_COLS:
        r[c] = ""

    merged.append(r)

# ── 작업 3: 정렬 ───────────────────────────────────────────────────────────────
def sort_key(r):
    pri = 0 if r.get("reviewer_priority", "") == "high" else 1
    risk = 0 if (r.get("safety_risk_if_inner1_flag", "") == "True" or
                 r.get("safety_risk_if_inner2_flag", "") == "True") else 1
    try:
        dist = float(r.get("min_distance_to_roi_boundary_px", 9999))
    except (ValueError, TypeError):
        dist = 9999.0
    try:
        loss = -float(r.get("bbox_margin_loss_inner2_proxy", 0))
    except (ValueError, TypeError):
        loss = 0.0
    return (pri, risk, dist, loss, r.get("patient_id", ""), r.get("object_id", ""))

merged.sort(key=sort_key)

# ── 작업 1 출력: label template CSV ───────────────────────────────────────────
all_cols = COLS_21D + COLS_21E + PANEL_COLS + REVIEWER_LABEL_COLS
OUT_CSV = REPORTS / "phase2_21g_object_level_reviewer_label_template.csv"
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(merged)

print(f"[OK] CSV 저장: {OUT_CSV}")

# ── 작업 4: HTML review_index ─────────────────────────────────────────────────
def rel_panel(abs_path):
    """HTML에서 panels/ 상대 경로로 변환"""
    p = Path(abs_path)
    return f"../phase2_21d_axis_corrected_cor_sag_mip_review/panels/{p.name}" if "_main" in p.name or "_supplement" in p.name or "_zoom" in p.name else f"../phase2_21e_inner_roi_margin_sensitivity_review/panels/{p.name}"

rows_html = []
for r in merged:
    oid = r["object_id"]
    pid = r["patient_id"]
    pri = r.get("reviewer_priority", "")
    dist = r.get("min_distance_to_roi_boundary_px", "")
    inner1_rem = r.get("centroid_removed_by_inner1_proxy", "")
    inner2_rem = r.get("centroid_removed_by_inner2_proxy", "")
    risk1 = r.get("safety_risk_if_inner1_flag", "")
    risk2 = r.get("safety_risk_if_inner2_flag", "")

    main_rel   = f"../phase2_21d_axis_corrected_cor_sag_mip_review/panels/{oid}_main.png"
    zoom_rel   = f"../phase2_21d_axis_corrected_cor_sag_mip_review/panels/{oid}_zoom.png"
    suppl_rel  = f"../phase2_21d_axis_corrected_cor_sag_mip_review/panels/{oid}_supplement.png"
    margin_rel = f"../phase2_21e_inner_roi_margin_sensitivity_review/panels/{oid}_margin.png"

    risk1_style = ' style="background:#ffe0e0"' if risk1 == "True" else ""
    risk2_style = ' style="background:#ffe0e0"' if risk2 == "True" else ""
    pri_style   = ' style="font-weight:bold;color:#c00"' if pri == "high" else ""

    rows_html.append(f"""
  <tr>
    <td><code>{oid}</code></td>
    <td>{pid}</td>
    <td{pri_style}>{pri}</td>
    <td>{dist}</td>
    <td>{inner1_rem}</td>
    <td>{inner2_rem}</td>
    <td{risk1_style}>{risk1}</td>
    <td{risk2_style}>{risk2}</td>
    <td><a href="{main_rel}" target="_blank"><img src="{main_rel}" width="160" loading="lazy"></a></td>
    <td><a href="{zoom_rel}" target="_blank"><img src="{zoom_rel}" width="160" loading="lazy"></a></td>
    <td><a href="{margin_rel}" target="_blank"><img src="{margin_rel}" width="160" loading="lazy"></a></td>
    <td><a href="{suppl_rel}" target="_blank">supplement</a></td>
  </tr>""")

html_body = "\n".join(rows_html)
html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Phase 2.21g — Object Reviewer Label Template</title>
<style>
  body {{ font-family: monospace; font-size: 12px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 6px; vertical-align: top; }}
  th {{ background: #f0f0f0; }}
  .warn {{ background: #fff3cd; border: 2px solid #f0ad4e; padding: 12px; margin-bottom: 16px; }}
  .warn h2 {{ margin: 0 0 8px 0; color: #c00; }}
</style>
</head>
<body>
<div class="warn">
  <h2>⚠ REVIEW TEMPLATE ONLY — 자동 라벨 없음</h2>
  <ul>
    <li><strong>NO VESSEL MASK</strong> — vessel mask가 생성되지 않았습니다.</li>
    <li><strong>NO PSEUDO MASK</strong> — pseudo mask가 생성되지 않았습니다.</li>
    <li><strong>NO SCORE CHANGE</strong> — score CSV가 수정되지 않았습니다.</li>
    <li><strong>INNER ROI IS NOT A HARD FILTER</strong> — inner ROI는 boundary sensitivity feature로만 사용합니다.</li>
    <li>이 HTML은 reviewer가 panel을 보면서 수동으로 라벨을 채우기 위한 참고 화면입니다.</li>
  </ul>
</div>
<h1>Phase 2.21g — Object-Level Reviewer Label Template</h1>
<p>총 {len(merged)}개 object / 정렬 기준: reviewer_priority &gt; safety_risk &gt; boundary_dist &gt; margin_loss</p>
<table>
  <thead>
    <tr>
      <th>object_id</th>
      <th>patient_id</th>
      <th>priority</th>
      <th>min_dist_boundary_px</th>
      <th>inner1_removed</th>
      <th>inner2_removed</th>
      <th>safety_risk_inner1</th>
      <th>safety_risk_inner2</th>
      <th>21d main panel</th>
      <th>21d zoom panel</th>
      <th>21e margin panel</th>
      <th>21d supplement</th>
    </tr>
  </thead>
  <tbody>
{html_body}
  </tbody>
</table>
</body>
</html>
"""

OUT_HTML = OUT_DIR / "review_index.html"
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"[OK] HTML 저장: {OUT_HTML}")

# ── 작업 6: summary JSON ───────────────────────────────────────────────────────
summary = {
    "phase": "2.21g",
    "n_objects": len(merged),
    "n_patients": 7,
    "input_21d_available": True,
    "input_21e_available": True,
    "label_columns_initialized_blank": True,
    "auto_final_label_created": False,
    "vessel_mask_generated": False,
    "pseudo_mask_generated": False,
    "suppression_weight_generated": False,
    "adjusted_score_generated": False,
    "score_csv_modified": False,
    "stage2_holdout_used": False,
    "inner_roi_hard_filter_used": False,
    "sort_order": [
        "reviewer_priority == high",
        "safety_risk_if_inner1_flag or safety_risk_if_inner2_flag",
        "min_distance_to_roi_boundary_px asc",
        "bbox_margin_loss_inner2_proxy desc",
        "patient_id / object_id",
    ],
    "sort_is_review_order_only_not_judgment": True,
    "recommended_next_step": "Fill reviewer labels manually, then create a dry-rule prototype from reviewer labels only.",
}

OUT_JSON = REPORTS / "phase2_21g_object_level_reviewer_label_summary.json"
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"[OK] JSON 저장: {OUT_JSON}")
print("=== 완료 ===")
print(f"  CSV  : {OUT_CSV}")
print(f"  JSON : {OUT_JSON}")
print(f"  HTML : {OUT_HTML}")
