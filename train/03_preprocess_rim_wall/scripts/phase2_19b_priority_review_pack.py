"""
Phase 2.19b Priority Object Visual Review Pack 생성 스크립트
- 새 계산 없음, CT/mask 원본 로드 없음
- Phase 2.19 결과 read-only 사용
- CSV/MD/HTML/PNG contact sheet만 생성
"""

import os
import csv
import shutil
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 경로 설정 ──────────────────────────────────────────────────────────────
BASE = Path("/home/jinhy/project/lung-ct-anomaly/outputs/mip-postprocess-research-v1")
SRC_CSV = BASE / "reports/phase2_19_v2_adaptive_mip_thickness_qa.csv"
OUT_PACK = BASE / "qa/phase2_19b_priority_object_review_pack"
OUT_PANELS = OUT_PACK / "object_panels"
OUT_CS = OUT_PACK / "contact_sheets"
OUT_TABLES = OUT_PACK / "tables"
OUT_CSV = BASE / "reports/phase2_19b_priority_object_review_pack.csv"
OUT_GUIDE = BASE / "reports/phase2_19b_priority_object_review_guide.md"
OUT_HTML = OUT_PACK / "review_index.html"
OUT_REPORT = BASE / "reports/phase2_19b_priority_object_review_pack_report.md"

PRIORITY_ORDER = [
    "p218obj_LUNG1_004_o001",
    "p218obj_LUNG1_001_o001",
    "p218obj_LUNG1_001_o002",
    "p218obj_LUNG1_020_o003",
    "p218obj_LUNG1_035_o001",
]

EXTRACT_COLS = [
    "object_id", "patient_id", "safe_id", "z_min", "z_max",
    "object_size_group", "object_equivalent_diameter_mm",
    "object_lesion_overlap_max", "object_lung_window_mean_hu",
    "object_flat_opacity_score", "object_gradient_profile_score",
    "small_lesion_protection_hint", "ggo_like_protection_hint",
    "mixed_contact_hint", "vessel_continuity_hint", "tubular_growth_hint",
    "adaptive_mip_qa_panel_path", "adaptive_mip_supplement_path",
]

REVIEWER_COLS = [
    "reviewer_label_lesion_blob",
    "reviewer_label_vessel_branch",
    "reviewer_label_vessel_tubular",
    "reviewer_label_mixed_contact",
    "reviewer_label_ggo_like",
    "reviewer_label_non_vessel_blob",
    "reviewer_label_uncertain",
    "reviewer_note",
]


# ── 작업 0: 출력 폴더 생성 ──────────────────────────────────────────────────
def setup_dirs():
    if OUT_PACK.exists() and any(OUT_PACK.iterdir()):
        raise RuntimeError(
            f"출력 폴더가 비어있지 않습니다: {OUT_PACK}\n"
            "_v2 suffix 사용 또는 수동 정리 후 재실행 필요"
        )
    for d in [OUT_PANELS, OUT_CS, OUT_TABLES]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[OK] 출력 폴더 생성: {OUT_PACK}")


# ── 작업 2: priority object 데이터 추출 ────────────────────────────────────
def load_priority_rows():
    rows_by_id = {}
    with open(SRC_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["object_id"] in PRIORITY_ORDER:
                rows_by_id[row["object_id"]] = row

    ordered = []
    for oid in PRIORITY_ORDER:
        if oid not in rows_by_id:
            raise RuntimeError(f"object_id 없음: {oid}")
        ordered.append(rows_by_id[oid])
    print(f"[OK] priority object {len(ordered)}개 추출 완료")
    return ordered


# ── 작업 3: priority review CSV 생성 ───────────────────────────────────────
def write_priority_csv(rows):
    out_cols = ["review_order"] + EXTRACT_COLS + REVIEWER_COLS
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            out_row = {"review_order": i}
            for col in EXTRACT_COLS:
                out_row[col] = row.get(col, "")
            for col in REVIEWER_COLS:
                out_row[col] = ""
            writer.writerow(out_row)
    print(f"[OK] priority review CSV: {OUT_CSV}")


# ── 작업 4: review guide MD 생성 ───────────────────────────────────────────
def write_guide_md():
    content = """# Phase 2.19b Priority Object Review Guide

## 이 문서의 목적

이번 review pack은 자동 판정이 아니라 사람 검토용이다.
reviewer가 직접 panel을 보고 `reviewer_label_*` 컬럼을 채운다.

## panel 두께별 확인 포인트

| 두께 | 확인 목적 |
|------|-----------|
| 10mm | 혈관 흐름 연속성 확인용 |
| 5mm / 3mm | 작은 결절 보호 여부 확인용 |
| 1slice | 실제 단면 형태 확인용 |

## 판정 기준

- **lesion_blob 가능성**: axial에서 blob이 남고 coronal/sagittal에서도 꽉 찬 덩어리면 lesion_blob 가능성
- **vessel_branch 가능성**: axial에서 둥글어도 coronal/sagittal에서 선형으로 이어지면 vessel_branch 가능성
- **mixed_contact 가능성**: 10mm에서는 붙어 보이고 3mm/1slice에서 core와 branch가 나뉘면 mixed_contact 가능성
- **ggo_like / faint opacity**: HU -700~-300 근처에서 flat_opacity_score가 높으면 ggo_like 또는 faint opacity 보호 후보로 본다

## 특이 object 주의 사항

- **LUNG1-035_o001**: eq_diam 62.6mm로 큰 object — 과병합 여부를 먼저 확인한다
- **LUNG1-020_o003**: lesion_overlap 0이지만 HU 약 -400으로 보고됨 — GGO-like 또는 faint opacity 여부를 확인한다

## 검토 순서

| review_order | object_id | 우선 확인 이유 |
|---|---|---|
| 1 | p218obj_LUNG1_004_o001 | lesion_overlap=1.0, small_lesion_protection |
| 2 | p218obj_LUNG1_001_o001 | lesion_overlap=1.0, small_lesion_protection |
| 3 | p218obj_LUNG1_001_o002 | lesion_overlap=1.0, small_lesion_protection |
| 4 | p218obj_LUNG1_020_o003 | lesion_overlap=0, HU≈-401, mixed+tubular hint |
| 5 | p218obj_LUNG1_035_o001 | large(62.6mm), mixed_contact, 과병합 확인 필요 |
"""
    with open(OUT_GUIDE, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] review guide MD: {OUT_GUIDE}")


# ── 작업 5: contact sheet PNG 생성 ─────────────────────────────────────────
def load_image_safe(path):
    p = Path(path)
    if not p.exists():
        return None
    return Image.open(p)


def make_contact_sheet(images, labels, out_path, title, cols=2):
    valid = [(img, lbl) for img, lbl in zip(images, labels) if img is not None]
    if not valid:
        print(f"[SKIP] 이미지 없어 contact sheet 미생성: {out_path}")
        return

    n = len(valid)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle(title, fontsize=13, y=0.98)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx < len(valid):
                img, lbl = valid[idx]
                ax.imshow(img)
                ax.set_title(lbl, fontsize=9)
                ax.axis("off")
                idx += 1
            else:
                ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] contact sheet: {out_path}")


def write_contact_sheets(rows):
    main_imgs, supp_imgs, labels = [], [], []
    for row in rows:
        labels.append(row["object_id"])
        main_imgs.append(load_image_safe(row["adaptive_mip_qa_panel_path"]))
        supp_imgs.append(load_image_safe(row["adaptive_mip_supplement_path"]))

    make_contact_sheet(
        main_imgs, labels,
        OUT_CS / "priority_top5_main_panels.png",
        "Priority Top5 — Main QA Panels",
        cols=2,
    )
    make_contact_sheet(
        supp_imgs, labels,
        OUT_CS / "priority_top5_supplement_panels.png",
        "Priority Top5 — Supplement Panels",
        cols=2,
    )

    # summary: main + supp 나란히 (object당 1행)
    fig, axes = plt.subplots(len(rows), 2, figsize=(18, len(rows) * 5))
    fig.suptitle("Priority Top5 — Summary (Main | Supplement)", fontsize=13, y=0.99)
    for i, (row, main, supp) in enumerate(zip(rows, main_imgs, supp_imgs)):
        lbl = row["object_id"]
        ax_m = axes[i][0]
        ax_s = axes[i][1]
        if main:
            ax_m.imshow(main)
        ax_m.set_title(f"{lbl}\n(main)", fontsize=8)
        ax_m.axis("off")
        if supp:
            ax_s.imshow(supp)
        ax_s.set_title(f"{lbl}\n(supp)", fontsize=8)
        ax_s.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUT_CS / "priority_top5_summary.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] contact sheet: {OUT_CS}/priority_top5_summary.png")


# ── 작업 6: index HTML 생성 ─────────────────────────────────────────────────
def write_index_html(rows):
    qa_dir = Path("/home/jinhy/project/lung-ct-anomaly/outputs/mip-postprocess-research-v1/qa/phase2_19_v2_adaptive_mip_thickness_qa")

    def rel(p):
        try:
            return os.path.relpath(p, OUT_PACK)
        except Exception:
            return str(p)

    rows_html = ""
    for i, row in enumerate(rows, start=1):
        main_rel = rel(row["adaptive_mip_qa_panel_path"])
        supp_rel = rel(row["adaptive_mip_supplement_path"])
        hints = ", ".join(
            k for k in ["small_lesion_protection_hint", "ggo_like_protection_hint",
                         "mixed_contact_hint", "vessel_continuity_hint", "tubular_growth_hint"]
            if row.get(k, "").strip().lower() == "true"
        ) or "—"
        rows_html += f"""
        <tr>
          <td>{i}</td>
          <td><b>{row['object_id']}</b></td>
          <td>{row['patient_id']}</td>
          <td>{row['object_size_group']}</td>
          <td>{float(row['object_equivalent_diameter_mm']):.1f}</td>
          <td>{float(row['object_lesion_overlap_max']):.2f}</td>
          <td>{float(row['object_lung_window_mean_hu']):.1f}</td>
          <td>{float(row['object_flat_opacity_score']):.4f}</td>
          <td>{float(row['object_gradient_profile_score']):.4f}</td>
          <td>{hints}</td>
          <td><a href="{main_rel}" target="_blank">main panel</a></td>
          <td><a href="{supp_rel}" target="_blank">supplement</a></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Phase 2.19b Priority Object Review Pack</title>
<style>
  body {{ font-family: sans-serif; font-size: 13px; margin: 20px; }}
  h1 {{ font-size: 18px; }}
  h2 {{ font-size: 15px; margin-top: 28px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 5px 8px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  a {{ color: #1a6eb0; }}
  .note {{ background: #fffbe6; border-left: 4px solid #f0c040; padding: 8px 12px; margin: 12px 0; }}
</style>
</head>
<body>
<h1>Phase 2.19b Priority Object Review Pack</h1>
<div class="note">
  이 페이지는 자동 판정이 아닌 사람 검토용입니다.<br>
  panel을 확인한 뒤 <code>phase2_19b_priority_object_review_pack.csv</code>의
  <code>reviewer_label_*</code> 컬럼을 직접 채워주세요.
</div>

<h2>검토 순서 및 핵심 수치</h2>
<table>
  <thead>
    <tr>
      <th>#</th><th>object_id</th><th>patient</th><th>size_group</th>
      <th>eq_diam_mm</th><th>lesion_overlap</th><th>mean_HU</th>
      <th>flat_opacity</th><th>gradient</th><th>active hints</th>
      <th>main panel</th><th>supplement</th>
    </tr>
  </thead>
  <tbody>{rows_html}
  </tbody>
</table>

<h2>Contact Sheets</h2>
<ul>
  <li><a href="contact_sheets/priority_top5_main_panels.png" target="_blank">priority_top5_main_panels.png</a></li>
  <li><a href="contact_sheets/priority_top5_supplement_panels.png" target="_blank">priority_top5_supplement_panels.png</a></li>
  <li><a href="contact_sheets/priority_top5_summary.png" target="_blank">priority_top5_summary.png</a></li>
</ul>

<h2>관련 파일</h2>
<ul>
  <li><a href="../../reports/phase2_19b_priority_object_review_pack.csv">priority_object_review_pack.csv</a></li>
  <li><a href="../../reports/phase2_19b_priority_object_review_guide.md">review_guide.md</a></li>
  <li><a href="../../reports/phase2_19b_priority_object_review_pack_report.md">report.md</a></li>
</ul>
</body>
</html>
"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] index HTML: {OUT_HTML}")


# ── 작업 7: report MD 생성 ──────────────────────────────────────────────────
def write_report_md(rows):
    generated = [
        str(OUT_CSV),
        str(OUT_GUIDE),
        str(OUT_HTML),
        str(OUT_CS / "priority_top5_main_panels.png"),
        str(OUT_CS / "priority_top5_supplement_panels.png"),
        str(OUT_CS / "priority_top5_summary.png"),
        str(OUT_REPORT),
    ]

    obj_lines = "\n".join(
        f"| {i} | {r['object_id']} | {r['patient_id']} | "
        f"{r['object_size_group']} | {float(r['object_equivalent_diameter_mm']):.1f} | "
        f"{float(r['object_lesion_overlap_max']):.2f} | {float(r['object_lung_window_mean_hu']):.1f} |"
        for i, r in enumerate(rows, 1)
    )

    content = f"""# Phase 2.19b Priority Object Visual Review Pack — 실행 보고서

## Phase 2.19 실행 결과 요약

Phase 2.19 v2 Adaptive MIP Thickness QA 정상 완료.
입력 CSV: `{SRC_CSV.name}`

## Priority Object 5개 선정 이유

| # | object_id | patient | size_group | eq_diam_mm | lesion_overlap | mean_HU |
|---|-----------|---------|------------|------------|----------------|---------|
{obj_lines}

- **LUNG1-004_o001, LUNG1-001_o001/o002**: lesion_overlap=1.0 + small_lesion_protection — 실제 결절 후보 최우선
- **LUNG1-020_o003**: lesion_overlap=0이지만 HU≈-401 (GGO-like/faint opacity 여부 확인 필요)
- **LUNG1-035_o001**: eq_diam=62.6mm (large) + mixed_contact — 과병합 여부 확인 필요

## 생성 파일 목록

{chr(10).join(f'- {p}' for p in generated)}

## 준수 사항 확인

| 항목 | 확인 |
|------|------|
| 원본 Phase 2.19 파일 read-only 사용 | ✅ |
| 새 MIP 계산 없음 | ✅ |
| CT/ROI/mask 원본 로드 없음 | ✅ |
| vessel soft mask 생성 없음 | ✅ |
| subtraction mask 생성 없음 | ✅ |
| suppression_weight 계산 없음 | ✅ |
| adjusted score 계산 없음 | ✅ |
| Phase 3 진행 없음 | ✅ |
| reviewer label 공란 | ✅ |
| outputs/mip-postprocess-research-v1/ 밖 생성 없음 | ✅ |
"""
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] report MD: {OUT_REPORT}")


# ── 메인 ────────────────────────────────────────────────────────────────────
def main():
    print("=== Phase 2.19b Priority Object Review Pack 시작 ===")
    setup_dirs()
    rows = load_priority_rows()
    write_priority_csv(rows)
    write_guide_md()
    write_contact_sheets(rows)
    write_index_html(rows)
    write_report_md(rows)
    print("=== 완료 ===")


if __name__ == "__main__":
    main()
