#!/usr/bin/env python3
"""
Phase 2.19f: Remaining Object Vessel Candidate Review Pack
Read-only: 기존 Phase 2.19 CSV와 PNG 사용
No new MIP computation, no CT/ROI/mask loading
"""
import io
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from datetime import datetime

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE = Path("/home/jinhy/project/lung-ct-anomaly/outputs/mip-postprocess-research-v1")
REPORTS = BASE / "reports"
QA_BASE = BASE / "qa"
OUT_QA = QA_BASE / "phase2_19f_remaining_vessel_candidate_review"
PHASE219_QA = QA_BASE / "phase2_19_v2_adaptive_mip_thickness_qa"

THICKNESS_QA_CSV = REPORTS / "phase2_19_v2_adaptive_mip_thickness_qa.csv"
SLAB_LEVEL_CSV   = REPORTS / "phase2_19_v2_adaptive_mip_slab_level.csv"
OBJECT_LABEL_CSV = REPORTS / "phase2_19e_manual_review_labels_object_level.csv"
COMPONENT_LABEL_CSV = REPORTS / "phase2_19e_manual_review_labels_component_level.csv"

OUT_CSV  = REPORTS / "phase2_19f_remaining_vessel_candidate_review.csv"
OUT_HTML = OUT_QA  / "review_index.html"
OUT_MD   = REPORTS / "phase2_19f_remaining_vessel_candidate_review_report.md"

ALREADY_REVIEWED = {
    "p218obj_LUNG1_004_o001",
    "p218obj_LUNG1_001_o001",
    "p218obj_LUNG1_001_o002",
    "p218obj_LUNG1_020_o003",
    "p218obj_LUNG1_035_o001",
}

PANEL_TOP_N = 8


def safe_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix, v = path.stem, path.suffix, 2
    while True:
        new_path = path.parent / f"{stem}_v{v}{suffix}"
        if not new_path.exists():
            return new_path
        v += 1


def load_img(p):
    if p and Path(p).exists():
        return np.array(Image.open(p).convert("RGB"))
    return None


def get_crop(obj_dir: Path, axis: str, thickness: str, window: str = "raw"):
    t = "1s" if thickness == "1slice" else thickness
    candidate = obj_dir / f"{axis}_{t}_c_{window}_crop.png"
    return candidate if candidate.exists() else None


def compute_priority_score(row: pd.Series):
    score = 0.0
    reasons = []

    overlap     = float(row.get("object_lesion_overlap_max", 1.0))
    vessel_cont = bool(row.get("vessel_continuity_hint", False))
    tubular     = bool(row.get("tubular_growth_hint", False))
    mixed       = bool(row.get("mixed_contact_hint", False))
    gradient    = float(row.get("object_gradient_profile_score", 0.0))
    flat_op     = float(row.get("object_flat_opacity_score", 1.0))
    diameter    = float(row.get("object_equivalent_diameter_mm", 30.0))
    small_p     = bool(row.get("small_lesion_protection_hint", False))
    ggo_p       = bool(row.get("ggo_like_protection_hint", False))

    if vessel_cont:
        score += 2.0
        reasons.append("vessel_continuity=True(+2)")
    if tubular:
        score += 2.0
        reasons.append("tubular_growth=True(+2)")
    if not mixed:
        score += 1.0
        reasons.append("mixed_contact=False(+1)")

    overlap_bonus = max(0.0, 1.0 - 2.0 * overlap)
    score += overlap_bonus
    if overlap_bonus > 0:
        reasons.append(f"low_overlap({overlap:.2f},+{overlap_bonus:.2f})")

    score += gradient
    reasons.append(f"gradient({gradient:.2f})")
    score += (1.0 - flat_op)
    reasons.append(f"opacity({flat_op:.2f})")

    if overlap >= 0.5:
        score -= 3.0
        reasons.append(f"high_overlap({overlap:.2f},-3)")
    if small_p:
        score -= 2.0
        reasons.append("small_lesion_protect(-2)")
    if ggo_p:
        score -= 2.0
        reasons.append("ggo_like_protect(-2)")
    if diameter > 50:
        score -= 2.0
        reasons.append(f"very_large_diam({diameter:.0f}mm,-2)")
    elif diameter > 30:
        score -= 1.0
        reasons.append(f"large_diam({diameter:.0f}mm,-1)")
    if flat_op > 0.7:
        score -= 1.0
        reasons.append(f"high_flat_opacity({flat_op:.2f},-1)")

    return round(score, 3), "; ".join(reasons)


def build_main_panel(obj_dir: Path, row: pd.Series, rank: int, score: float, reason: str):
    views = {
        "axial_1s":      get_crop(obj_dir, "axial",    "1slice"),
        "axial_3mm":     get_crop(obj_dir, "axial",    "3mm"),
        "axial_5mm":     get_crop(obj_dir, "axial",    "5mm"),
        "axial_10mm":    get_crop(obj_dir, "axial",    "10mm"),
        "coronal_3mm":   get_crop(obj_dir, "coronal",  "3mm"),
        "coronal_5mm":   get_crop(obj_dir, "coronal",  "5mm"),
        "coronal_10mm":  get_crop(obj_dir, "coronal",  "10mm"),
        "sagittal_3mm":  get_crop(obj_dir, "sagittal", "3mm"),
        "sagittal_5mm":  get_crop(obj_dir, "sagittal", "5mm"),
        "sagittal_10mm": get_crop(obj_dir, "sagittal", "10mm"),
    }
    imgs = {k: load_img(v) for k, v in views.items()}

    fig = plt.figure(figsize=(20, 18), facecolor="#1a1a2e")
    gs = gridspec.GridSpec(5, 4, figure=fig,
                           hspace=0.35, wspace=0.08,
                           top=0.96, bottom=0.03, left=0.02, right=0.98)

    ax_t = fig.add_subplot(gs[0, :])
    ax_t.set_facecolor("#16213e")
    ax_t.axis("off")
    oid     = row["object_id"]
    pid     = row["patient_id"]
    diam    = float(row.get("object_equivalent_diameter_mm", 0))
    overlap = float(row.get("object_lesion_overlap_max", 0))
    hints = []
    if row.get("vessel_continuity_hint"):   hints.append("VesselCont")
    if row.get("tubular_growth_hint"):      hints.append("Tubular")
    if row.get("mixed_contact_hint"):       hints.append("MixedContact")
    if row.get("small_lesion_protection_hint"): hints.append("SmallProtect")
    if row.get("ggo_like_protection_hint"): hints.append("GGOProtect")
    hint_str = " | ".join(hints) if hints else "None"
    ax_t.text(0.5, 0.5,
              (f"Rank #{rank}  |  {oid}  |  Patient: {pid}\n"
               f"Priority Score: {score:.3f}  |  Diam: {diam:.1f}mm  |  LesionOverlap: {overlap:.2f}\n"
               f"Hints: {hint_str}\n"
               f"Reason: {reason[:120]}"),
              transform=ax_t.transAxes, color="white",
              fontsize=8.5, va="center", ha="center", fontfamily="monospace")

    def show(ax, key, label):
        img = imgs.get(key)
        ax.axis("off")
        ax.set_facecolor("#0f3460")
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "N/A", color="gray",
                    transform=ax.transAxes, ha="center", va="center")
        ax.set_title(label, color="#e2e2e2", fontsize=7.5, pad=3)

    show(fig.add_subplot(gs[1, 0]), "axial_1s",   "Axial 1slice\n(raw crop)")
    show(fig.add_subplot(gs[1, 1]), "axial_3mm",  "Axial 3mm\n(raw crop)")
    show(fig.add_subplot(gs[1, 2]), "axial_5mm",  "Axial 5mm\n(raw crop)")
    show(fig.add_subplot(gs[1, 3]), "axial_10mm", "Axial 10mm\n(raw crop)")

    show(fig.add_subplot(gs[2, 0]), "coronal_3mm",  "Coronal 3mm\n(raw crop)")
    show(fig.add_subplot(gs[2, 1]), "coronal_5mm",  "Coronal 5mm\n(raw crop)")
    show(fig.add_subplot(gs[2, 2]), "coronal_10mm", "Coronal 10mm\n(raw crop)")
    ax_e = fig.add_subplot(gs[2, 3])
    ax_e.axis("off")
    ax_e.set_facecolor("#1a1a2e")

    show(fig.add_subplot(gs[3, 0]), "sagittal_3mm",  "Sagittal 3mm\n(raw crop)")
    show(fig.add_subplot(gs[3, 1]), "sagittal_5mm",  "Sagittal 5mm\n(raw crop)")
    show(fig.add_subplot(gs[3, 2]), "sagittal_10mm", "Sagittal 10mm\n(raw crop)")
    ax_e2 = fig.add_subplot(gs[3, 3])
    ax_e2.axis("off")
    ax_e2.set_facecolor("#1a1a2e")

    ax_g = fig.add_subplot(gs[4, :])
    ax_g.set_facecolor("#16213e")
    ax_g.axis("off")
    ax_g.text(0.02, 0.5,
              ("VESSEL CANDIDATE GUIDE:\n"
               "  • Tubular across coronal/sagittal → vessel candidate\n"
               "  • Compact across all views → lesion/blob protection\n"
               "  • Core + branch separation → mixed contact\n"
               "  • Faint/flat low-HU → GGO-like protection candidate"),
              transform=ax_g.transAxes, color="#a8dadc",
              fontsize=8, va="center", fontfamily="monospace")

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
    buf.seek(0)
    arr = np.array(Image.open(buf))
    plt.close(fig)
    return arr


def build_supplement_panel(obj_dir: Path, row: pd.Series, rank: int):
    keys = {
        "axial_masked":    obj_dir / "axial_5mm_c_masked_crop.png",
        "coronal_masked":  obj_dir / "coronal_5mm_c_masked_crop.png",
        "sagittal_masked": obj_dir / "sagittal_5mm_c_masked_crop.png",
        "axial_medi":      obj_dir / "axial_5mm_c_medi_crop.png",
        "coronal_medi":    obj_dir / "coronal_5mm_c_medi_crop.png",
        "sagittal_medi":   obj_dir / "sagittal_5mm_c_medi_crop.png",
        "axial_narrow":    obj_dir / "axial_5mm_c_narrow_crop.png",
        "coronal_narrow":  obj_dir / "coronal_5mm_c_narrow_crop.png",
        "sagittal_narrow": obj_dir / "sagittal_5mm_c_narrow_crop.png",
    }
    imgs = {k: load_img(v) for k, v in keys.items()}

    fig = plt.figure(figsize=(15, 14), facecolor="#1a1a2e")
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.35, wspace=0.08,
                           top=0.94, bottom=0.03, left=0.02, right=0.98)

    ax_t = fig.add_subplot(gs[0, :])
    ax_t.set_facecolor("#16213e")
    ax_t.axis("off")
    ax_t.text(0.5, 0.5,
              f"Supplement Panel — Rank #{rank}  |  {row['object_id']}  |  5mm center slab",
              transform=ax_t.transAxes, color="white",
              fontsize=9, va="center", ha="center", fontfamily="monospace")

    def show(ax, key, label):
        img = imgs.get(key)
        ax.axis("off")
        ax.set_facecolor("#0f3460")
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "N/A", color="gray",
                    transform=ax.transAxes, ha="center", va="center")
        ax.set_title(label, color="#e2e2e2", fontsize=7.5, pad=3)

    show(fig.add_subplot(gs[1, 0]), "axial_masked",   "Axial 5mm\n(masked)")
    show(fig.add_subplot(gs[1, 1]), "coronal_masked",  "Coronal 5mm\n(masked)")
    show(fig.add_subplot(gs[1, 2]), "sagittal_masked", "Sagittal 5mm\n(masked)")
    show(fig.add_subplot(gs[2, 0]), "axial_medi",      "Axial 5mm\n(mediastinal)")
    show(fig.add_subplot(gs[2, 1]), "coronal_medi",    "Coronal 5mm\n(mediastinal)")
    show(fig.add_subplot(gs[2, 2]), "sagittal_medi",   "Sagittal 5mm\n(mediastinal)")
    show(fig.add_subplot(gs[3, 0]), "axial_narrow",    "Axial 5mm\n(narrow lung)")
    show(fig.add_subplot(gs[3, 1]), "coronal_narrow",  "Coronal 5mm\n(narrow lung)")
    show(fig.add_subplot(gs[3, 2]), "sagittal_narrow", "Sagittal 5mm\n(narrow lung)")

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
    buf.seek(0)
    arr = np.array(Image.open(buf))
    plt.close(fig)
    return arr


def write_html(df_remain, out_rows, html_path: Path):
    rows_html = ""
    for r in out_rows:
        oid = r["object_id"]
        main_name = Path(r["annotated_main_panel_path"]).name if r["annotated_main_panel_path"] else ""
        supp_name = Path(r["annotated_supplement_panel_path"]).name if r["annotated_supplement_panel_path"] else ""
        main_link = f'<a href="{main_name}">main</a>' if main_name else "—"
        supp_link = f'<a href="{supp_name}">supp</a>' if supp_name else "—"
        panel_note = "(panel)" if r["review_priority_rank"] <= PANEL_TOP_N else "(index only)"
        rows_html += (
            f"<tr>"
            f"<td>{r['review_priority_rank']}</td>"
            f"<td>{oid}</td>"
            f"<td>{r['patient_id']}</td>"
            f"<td>{float(r['object_equivalent_diameter_mm']):.1f}mm</td>"
            f"<td>{float(r['object_lesion_overlap_max']):.2f}</td>"
            f"<td>{r['vessel_continuity_hint']}</td>"
            f"<td>{r['tubular_growth_hint']}</td>"
            f"<td>{r['vessel_review_priority_score']:.3f}</td>"
            f"<td>{main_link} {supp_link} {panel_note}</td>"
            f"</tr>\n"
        )

    html_path.write_text(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phase 2.19f: Remaining Vessel Candidate Review</title>
<style>
body {{ font-family: monospace; background: #1a1a2e; color: #e2e2e2; margin: 24px; }}
h1, h2 {{ color: #a8dadc; }}
table {{ border-collapse: collapse; width: 100%; }}
th {{ background: #16213e; color: #a8dadc; padding: 6px 10px; text-align: left; }}
td {{ border-bottom: 1px solid #333; padding: 5px 10px; }}
tr:hover {{ background: #16213e; }}
a {{ color: #4cc9f0; }}
.note {{ background: #0f3460; border-left: 4px solid #e63946; padding: 10px; margin: 12px 0; }}
</style>
</head>
<body>
<h1>Phase 2.19f: Remaining Object Vessel Candidate Review Pack</h1>
<div class="note">
  <b>Context:</b> As of Phase 2.19e, only 1 vessel candidate was confirmed (p218obj_LUNG1_020_o003).
  This pack covers the remaining {len(out_rows)} objects not yet manually reviewed,
  sorted by vessel review priority score.<br>
  <b>This is NOT automatic classification.</b> Reviewer labels must be filled in manually.<br>
  No new MIP computation. No CT/ROI/mask loading.
  Read-only use of existing Phase 2.19 PNG slabs.
</div>
<h2>Remaining Objects — Vessel Review Priority Ranking</h2>
<table>
  <tr>
    <th>Rank</th><th>Object ID</th><th>Patient</th>
    <th>Diameter</th><th>LesionOverlap</th>
    <th>VesselCont</th><th>Tubular</th>
    <th>Priority Score</th><th>Panels</th>
  </tr>
  {rows_html}
</table>
<h2>Reviewer Label Guide</h2>
<ul>
  <li><b>reviewer_label_vessel_branch</b>: Clear branching vessel structure visible</li>
  <li><b>reviewer_label_vessel_tubular</b>: Tubular shape confirmed across views</li>
  <li><b>reviewer_label_lesion_blob</b>: Solid compact lesion-like blob</li>
  <li><b>reviewer_label_mixed_contact</b>: Mixed lesion+vessel contact region</li>
  <li><b>reviewer_label_ggo_like</b>: Faint, flat, low-HU region (GGO-like)</li>
  <li><b>reviewer_label_non_vessel_blob</b>: Non-vessel, non-lesion structure</li>
  <li><b>reviewer_label_uncertain</b>: Cannot determine from visual review</li>
</ul>
<p style="color:#888; font-size:0.85em;">
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
  Phase 2.19f | Auto-classification: None
</p>
</body>
</html>""", encoding="utf-8")


def write_report(df_remain, out_rows, md_path: Path, csv_path: Path, html_path: Path):
    top5_str = "\n".join(
        f"  - Rank {r['review_priority_rank']}: {r['object_id']} "
        f"(score={r['vessel_review_priority_score']:.3f})"
        for r in out_rows if r["review_priority_rank"] <= 5
    )
    md_path.write_text(f"""# Phase 2.19f: Remaining Object Vessel Candidate Review Pack

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Phase 2.19e Label Summary

- Phase 2.19e manual review 완료. vessel candidate 확정 1개: `p218obj_LUNG1_020_o003`
- 나머지 4개는 lesion/blob 또는 보호 대상으로 판정됨

## 이미 검토한 object 제외 (5개)

- p218obj_LUNG1_004_o001
- p218obj_LUNG1_001_o001
- p218obj_LUNG1_001_o002
- p218obj_LUNG1_020_o003
- p218obj_LUNG1_035_o001

## 대상 object

- 전체 object 수: 16
- 이미 검토한 수: 5
- 남은 object 수: {len(df_remain)}

## panel 생성

- panel 생성 대상: 우선순위 상위 {PANEL_TOP_N}개
- 나머지 {len(df_remain) - PANEL_TOP_N}개: CSV/HTML index에만 포함

## vessel review priority 기준

올라가는 조건:
- vessel_continuity_hint=True (+2)
- tubular_growth_hint=True (+2)
- mixed_contact_hint=False (+1)
- object_lesion_overlap_max 낮을수록 가산
- object_gradient_profile_score 높을수록 가산
- object_flat_opacity_score 낮을수록 가산

내려가는 조건:
- object_lesion_overlap_max >= 0.5 (-3)
- small_lesion_protection_hint=True (-2)
- ggo_like_protection_hint=True (-2)
- object_equivalent_diameter_mm > 50mm (-2), > 30mm (-1)
- object_flat_opacity_score > 0.7 (-1)

## review priority 상위 5개

{top5_str}

## reviewer label

- 모든 reviewer label 컬럼은 공란
- 자동 판정 없음
- 사람이 직접 panel을 보고 레이블 작성 필요

## 출력 파일

- CSV: {csv_path}
- HTML: {html_path}

## 작업 준수 사항 확인

- 새 MIP 계산: 없음 ✓
- CT/ROI/mask 원본 로드: 없음 ✓
- 새 component 재검출: 없음 ✓
- 기존 PNG 수정/삭제/이동/덮어쓰기: 없음 ✓
- vessel soft mask 생성: 없음 ✓
- subtraction mask 생성: 없음 ✓
- suppression_weight 계산: 없음 ✓
- adjusted score 계산: 없음 ✓
- Phase 3 진행: 없음 ✓
- outputs/mip-postprocess-research-v1/ 밖 생성: 없음 ✓
""", encoding="utf-8")


def main():
    df_all    = pd.read_csv(THICKNESS_QA_CSV)
    df_labels = pd.read_csv(OBJECT_LABEL_CSV)
    print(f"[INFO] Total objects: {len(df_all)}")
    print(f"[INFO] Already reviewed: {len(ALREADY_REVIEWED)}")

    df_remain = df_all[~df_all["object_id"].isin(ALREADY_REVIEWED)].copy()
    df_remain = df_remain.reset_index(drop=True)
    print(f"[INFO] Remaining objects: {len(df_remain)}")

    scores, reasons = [], []
    for _, row in df_remain.iterrows():
        s, r = compute_priority_score(row)
        scores.append(s)
        reasons.append(r)
    df_remain["vessel_review_priority_score"] = scores
    df_remain["vessel_review_priority_reason"] = reasons
    df_remain = df_remain.sort_values("vessel_review_priority_score", ascending=False).reset_index(drop=True)
    df_remain["review_priority_rank"] = df_remain.index + 1

    print("\n[INFO] Priority ranking:")
    for _, row in df_remain.iterrows():
        print(f"  Rank {int(row['review_priority_rank']):2d}: {row['object_id']}  "
              f"score={row['vessel_review_priority_score']:.3f}")

    OUT_QA.mkdir(parents=True, exist_ok=True)

    panel_targets = df_remain[df_remain["review_priority_rank"] <= PANEL_TOP_N]
    main_panel_paths = {}
    supp_panel_paths = {}

    for _, row in panel_targets.iterrows():
        oid   = row["object_id"]
        sid   = row["safe_id"]
        rank  = int(row["review_priority_rank"])
        score = float(row["vessel_review_priority_score"])
        reason = str(row["vessel_review_priority_reason"])

        obj_dir = PHASE219_QA / f"{sid}_{oid}"
        if not obj_dir.exists():
            print(f"[WARN] obj_dir not found: {obj_dir}")
            main_panel_paths[oid] = ""
            supp_panel_paths[oid] = ""
            continue

        print(f"[PANEL] Rank {rank}: {oid}")

        main_arr = build_main_panel(obj_dir, row, rank, score, reason)
        main_out = safe_output_path(OUT_QA / f"{oid}_rank{rank:02d}_main.png")
        if main_arr is not None:
            Image.fromarray(main_arr).save(main_out)
            main_panel_paths[oid] = str(main_out)
        else:
            main_panel_paths[oid] = ""

        supp_arr = build_supplement_panel(obj_dir, row, rank)
        supp_out = safe_output_path(OUT_QA / f"{oid}_rank{rank:02d}_supplement.png")
        if supp_arr is not None:
            Image.fromarray(supp_arr).save(supp_out)
            supp_panel_paths[oid] = str(supp_out)
        else:
            supp_panel_paths[oid] = ""

    out_rows = []
    for _, row in df_remain.iterrows():
        oid = row["object_id"]
        out_rows.append({
            "review_priority_rank":           int(row["review_priority_rank"]),
            "object_id":                      oid,
            "patient_id":                     row["patient_id"],
            "safe_id":                        row["safe_id"],
            "z_min":                          row["z_min"],
            "z_max":                          row["z_max"],
            "object_size_group":              row["object_size_group"],
            "object_equivalent_diameter_mm":  row["object_equivalent_diameter_mm"],
            "object_lesion_overlap_max":      row["object_lesion_overlap_max"],
            "object_lung_window_mean_hu":     row["object_lung_window_mean_hu"],
            "object_flat_opacity_score":      row["object_flat_opacity_score"],
            "object_gradient_profile_score":  row["object_gradient_profile_score"],
            "small_lesion_protection_hint":   row["small_lesion_protection_hint"],
            "ggo_like_protection_hint":       row["ggo_like_protection_hint"],
            "mixed_contact_hint":             row["mixed_contact_hint"],
            "vessel_continuity_hint":         row["vessel_continuity_hint"],
            "tubular_growth_hint":            row["tubular_growth_hint"],
            "vessel_review_priority_score":   row["vessel_review_priority_score"],
            "vessel_review_priority_reason":  row["vessel_review_priority_reason"],
            "annotated_main_panel_path":      main_panel_paths.get(oid, ""),
            "annotated_supplement_panel_path": supp_panel_paths.get(oid, ""),
            "reviewer_label_vessel_branch":   "",
            "reviewer_label_vessel_tubular":  "",
            "reviewer_label_lesion_blob":     "",
            "reviewer_label_mixed_contact":   "",
            "reviewer_label_ggo_like":        "",
            "reviewer_label_non_vessel_blob": "",
            "reviewer_label_uncertain":       "",
            "reviewer_note":                  "",
        })

    out_csv_path = safe_output_path(OUT_CSV)
    pd.DataFrame(out_rows).to_csv(out_csv_path, index=False)
    print(f"\n[CSV] Saved: {out_csv_path}")

    html_path = safe_output_path(OUT_HTML)
    write_html(df_remain, out_rows, html_path)
    print(f"[HTML] Saved: {html_path}")

    md_path = safe_output_path(OUT_MD)
    write_report(df_remain, out_rows, md_path, out_csv_path, html_path)
    print(f"[MD] Saved: {md_path}")

    print("\n" + "=" * 60)
    print("Phase 2.19f 실행 결과 보고")
    print("=" * 60)
    print(f"1. exit code: 0 (오류 없음)")
    print(f"2. 전체 object 수: {len(df_all)}")
    print(f"3. 이미 검토한 object 제외 수: {len(ALREADY_REVIEWED)}")
    print(f"4. 남은 object 수: {len(df_remain)}")
    print(f"5. panel 생성 object 수: {len(panel_targets)}")
    print(f"6. CSV 경로: {out_csv_path}")
    print(f"7. HTML 경로: {html_path}")
    print(f"8. report MD 경로: {md_path}")
    print(f"9. review priority 상위 5개:")
    for r in out_rows[:5]:
        print(f"   Rank {r['review_priority_rank']}: {r['object_id']} "
              f"(score={r['vessel_review_priority_score']:.3f})")
    print(f"10. reviewer label 공란: ✓")
    print(f"11. 새 MIP 계산 없음: ✓")
    print(f"12. CT/ROI/mask 로드 없음: ✓")
    print(f"13. vessel soft mask/subtraction/suppression/Phase 3 없음: ✓")
    print(f"14. outputs/mip-postprocess-research-v1/ 밖 생성 없음: ✓")


if __name__ == "__main__":
    main()
