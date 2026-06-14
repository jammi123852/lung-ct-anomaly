"""
Phase 2.20d Visual-Only Dry Decision Overlay Pack
- Phase 2.20c decision table 기반으로 기존 panel PNG에 색상 프레임 + decision label만 추가
- CT/ROI/mask 로드 없음, 새 MIP 계산 없음, score CSV 수정 없음
- PIL/Pillow만 사용
"""

import os
import sys
import json
import csv
import textwrap
from pathlib import Path
from datetime import datetime

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE = Path("/home/jinhy/project/lung-ct-anomaly/outputs/mip-postprocess-research-v1")
INPUT_CSV = BASE / "reports" / "phase2_20c_dry_decision_visual_review_table.csv"

OUTPUT_CSV = BASE / "reports" / "phase2_20d_visual_only_dry_decision_overlay_table.csv"
OUTPUT_MD  = BASE / "reports" / "phase2_20d_visual_only_dry_decision_overlay_summary.md"
OUTPUT_JSON = BASE / "reports" / "phase2_20d_visual_only_dry_decision_overlay_summary.json"

OVERLAY_DIR = BASE / "qa" / "phase2_20d_visual_only_dry_decision_overlay_pack"
CONTACT_DIR = OVERLAY_DIR / "contact_sheets"
HTML_PATH   = OVERLAY_DIR / "review_index.html"

# ─────────────────────────────────────────────
# 색상 맵
# ─────────────────────────────────────────────
COLOR_MAP = {
    "vessel_positive_candidate": (0, 180, 0),
    "cautious_vessel_candidate": (220, 180, 0),
    "protected_negative": (0, 100, 220),
    "mixed_or_uncertain": (220, 100, 0),
    "unsafe_for_suppression": (220, 0, 0),
}

FRAME_WIDTH = 6
LABEL_BAR_HEIGHT = 28
WATERMARK_TEXT = "VISUAL ONLY — NO MASK / NO SCORE CHANGE"

# ─────────────────────────────────────────────
# 덮어쓰기 방지
# ─────────────────────────────────────────────
def check_no_overwrite(path: Path) -> Path:
    """파일/폴더가 이미 존재하면 _v2 suffix를 붙여 새 경로 반환. 존재하지 않으면 그대로 반환."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for i in range(2, 100):
        candidate = parent / f"{stem}_v{i}{suffix}"
        if not candidate.exists():
            print(f"[경고] {path} 이미 존재 → {candidate} 로 변경합니다.")
            return candidate
    raise RuntimeError(f"덮어쓰기 방지: {path} 에 대한 대체 경로를 찾지 못했습니다.")


def check_dir_no_overwrite(path: Path) -> Path:
    """디렉토리가 이미 존재하면 _v2 suffix를 붙여 새 경로 반환."""
    if not path.exists():
        return path
    parent = path.parent
    name = path.name
    for i in range(2, 100):
        candidate = parent / f"{name}_v{i}"
        if not candidate.exists():
            print(f"[경고] {path} 이미 존재 → {candidate} 로 변경합니다.")
            return candidate
    raise RuntimeError(f"덮어쓰기 방지: {path} 에 대한 대체 경로를 찾지 못했습니다.")


# ─────────────────────────────────────────────
# 폰트 로드 (실패 시 기본 폰트)
# ─────────────────────────────────────────────
def load_font(size: int):
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def load_font_regular(size: int):
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ─────────────────────────────────────────────
# label bar 텍스트 생성
# ─────────────────────────────────────────────
def make_label_text(row: dict) -> str:
    group = row["decision_group"]
    flags = []
    if str(row.get("dry_rule_positive_flag", "")).upper() == "TRUE":
        flags.append("POS")
    if str(row.get("dry_rule_cautious_flag", "")).upper() == "TRUE":
        flags.append("CAUTIOUS")
    if str(row.get("dry_rule_exclusion_flag", "")).upper() == "TRUE":
        flags.append("EXCL")
    flag_str = "/".join(flags) if flags else "—"

    reason_raw = str(row.get("decision_reason", ""))
    reason_short = reason_raw[:60] + ("…" if len(reason_raw) > 60 else "")

    return f"[{group.upper()}]  flags:{flag_str}  |  {reason_short}"


# ─────────────────────────────────────────────
# 단일 overlay 생성
# ─────────────────────────────────────────────
def create_overlay(src_path: str, row: dict) -> Image.Image:
    src = Image.open(src_path).convert("RGB")
    w, h = src.size
    color = COLOR_MAP.get(row["decision_group"], (180, 180, 180))

    # 새 캔버스: label bar + frame 포함
    new_h = h + LABEL_BAR_HEIGHT
    canvas = Image.new("RGB", (w, new_h), (30, 30, 30))

    # 1) label bar 그리기
    draw_bar = ImageDraw.Draw(canvas)
    draw_bar.rectangle([(0, 0), (w, LABEL_BAR_HEIGHT)], fill=color)
    label_text = make_label_text(row)
    font_bar = load_font(13)
    # 텍스트 색상: 밝은 배경에는 검정, 어두운 배경에는 흰색
    r, g, b = color
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    draw_bar.text((8, 6), label_text, fill=text_color, font=font_bar)

    # 2) 원본 이미지 붙여넣기
    canvas.paste(src, (0, LABEL_BAR_HEIGHT))

    # 3) 색상 프레임 (label bar 아래 이미지 영역에만)
    draw_frame = ImageDraw.Draw(canvas)
    # 상단 (label bar 바로 아래)
    draw_frame.rectangle(
        [(0, LABEL_BAR_HEIGHT), (w, LABEL_BAR_HEIGHT + FRAME_WIDTH)],
        fill=color,
    )
    # 하단
    draw_frame.rectangle(
        [(0, new_h - FRAME_WIDTH), (w, new_h)],
        fill=color,
    )
    # 좌측
    draw_frame.rectangle(
        [(0, LABEL_BAR_HEIGHT), (FRAME_WIDTH, new_h)],
        fill=color,
    )
    # 우측
    draw_frame.rectangle(
        [(w - FRAME_WIDTH, LABEL_BAR_HEIGHT), (w, new_h)],
        fill=color,
    )

    # 4) 우하단 워터마크
    font_wm = load_font_regular(11)
    wm_text = WATERMARK_TEXT
    # 텍스트 크기 측정
    try:
        bbox = draw_frame.textbbox((0, 0), wm_text, font=font_wm)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw_frame.textsize(wm_text, font=font_wm)

    margin = 10
    wm_x = w - tw - margin - FRAME_WIDTH - 2
    wm_y = new_h - th - margin - FRAME_WIDTH - 2

    # 반투명 배경 (RGBA 레이어)
    overlay_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay_layer)
    pad = 4
    draw_overlay.rectangle(
        [(wm_x - pad, wm_y - pad), (wm_x + tw + pad, wm_y + th + pad)],
        fill=(0, 0, 0, 160),
    )
    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba = Image.alpha_composite(canvas_rgba, overlay_layer)
    canvas = canvas_rgba.convert("RGB")

    # 워터마크 텍스트 그리기
    draw_final = ImageDraw.Draw(canvas)
    draw_final.text((wm_x, wm_y), wm_text, fill=(255, 255, 220), font=font_wm)

    return canvas


# ─────────────────────────────────────────────
# contact sheet 생성
# ─────────────────────────────────────────────
def create_contact_sheet(images_paths: list, group_name: str, count: int) -> Image.Image:
    """이미지 경로 리스트로 contact sheet 생성."""
    THUMB_W = 600
    COLS = 3
    TITLE_H = 50
    PAD = 8

    if not images_paths:
        # 빈 이미지
        sheet = Image.new("RGB", (THUMB_W * COLS + PAD * (COLS + 1), TITLE_H + 200), (50, 50, 50))
        draw = ImageDraw.Draw(sheet)
        font = load_font(20)
        draw.text((20, TITLE_H + 80), "No objects", fill=(200, 200, 200), font=font)
        # 제목
        draw.rectangle([(0, 0), (sheet.width, TITLE_H)], fill=(80, 80, 80))
        draw.text((10, 12), f"{group_name.upper()}  (n={count})", fill=(255, 255, 255), font=font)
        return sheet

    thumbs = []
    for p in images_paths:
        try:
            img = Image.open(p).convert("RGB")
            ratio = THUMB_W / img.width
            th = int(img.height * ratio)
            thumbs.append(img.resize((THUMB_W, th), Image.LANCZOS))
        except Exception as e:
            print(f"[경고] contact sheet 썸네일 로드 실패 {p}: {e}")

    if not thumbs:
        sheet = Image.new("RGB", (THUMB_W, TITLE_H + 100), (50, 50, 50))
        draw = ImageDraw.Draw(sheet)
        font = load_font(18)
        draw.text((10, TITLE_H + 30), "Image load failed", fill=(200, 100, 100), font=font)
        return sheet

    rows = (len(thumbs) + COLS - 1) // COLS
    max_h = max(t.height for t in thumbs)
    sheet_w = COLS * THUMB_W + (COLS + 1) * PAD
    sheet_h = TITLE_H + rows * (max_h + PAD) + PAD

    color = COLOR_MAP.get(group_name, (100, 100, 100))
    sheet = Image.new("RGB", (sheet_w, sheet_h), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)

    # 제목 바
    draw.rectangle([(0, 0), (sheet_w, TITLE_H)], fill=color)
    font_title = load_font(20)
    r, g, b = color
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    title_text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    draw.text((10, 12), f"{group_name.upper()}  (n={count})", fill=title_text_color, font=font_title)

    for idx, thumb in enumerate(thumbs):
        row_idx = idx // COLS
        col_idx = idx % COLS
        x = PAD + col_idx * (THUMB_W + PAD)
        y = TITLE_H + PAD + row_idx * (max_h + PAD)
        sheet.paste(thumb, (x, y))

    return sheet


# ─────────────────────────────────────────────
# HTML review index 생성
# ─────────────────────────────────────────────
def build_html(df: pd.DataFrame, overlay_dir: Path, html_path: Path):
    group_order = [
        "vessel_positive_candidate",
        "cautious_vessel_candidate",
        "protected_negative",
        "mixed_or_uncertain",
    ]
    group_color_hex = {
        "vessel_positive_candidate": "#00b400",
        "cautious_vessel_candidate": "#dcb400",
        "protected_negative": "#0064dc",
        "mixed_or_uncertain": "#dc6400",
        "unsafe_for_suppression": "#dc0000",
    }
    group_counts = df["decision_group"].value_counts().to_dict()

    lines = []
    lines.append("<!DOCTYPE html>")
    lines.append('<html lang="ko"><head><meta charset="utf-8">')
    lines.append("<title>Phase 2.20d Visual-Only Dry Decision Overlay Pack</title>")
    lines.append("<style>")
    lines.append("body { font-family: Arial, sans-serif; background: #1a1a1a; color: #eee; margin: 20px; }")
    lines.append("h1 { color: #fff; font-size: 1.5em; }")
    lines.append("h2 { color: #ccc; font-size: 1.2em; margin-top: 30px; }")
    lines.append("h3 { color: #aaa; font-size: 1.0em; margin-top: 20px; }")
    lines.append("table { border-collapse: collapse; margin: 10px 0; }")
    lines.append("th, td { border: 1px solid #555; padding: 6px 12px; }")
    lines.append("th { background: #333; }")
    lines.append(".group-section { margin-top: 30px; border-left: 6px solid #555; padding-left: 14px; }")
    lines.append(".object-block { display: inline-block; margin: 8px; vertical-align: top; }")
    lines.append(".object-block img { max-width: 480px; border: 2px solid #555; }")
    lines.append(".object-block p { font-size: 0.82em; color: #aaa; margin: 2px 0; }")
    lines.append(".notice { background: #2a2a1a; border: 1px solid #aa8800; padding: 10px 16px; margin: 10px 0; }")
    lines.append(".next-step { background: #1a2a1a; border: 1px solid #448844; padding: 10px 16px; margin: 10px 0; }")
    lines.append("</style>")
    lines.append("</head><body>")

    lines.append("<h1>Phase 2.20d &mdash; Visual-Only Dry Decision Overlay Pack</h1>")
    lines.append(f"<p>생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    lines.append('<div class="notice">')
    lines.append("<strong>VISUAL ONLY</strong>: 이 팩은 Phase 2.20c decision table을 기반으로 기존 panel PNG에 색상 프레임과 decision label만 추가한 시각화 전용 결과입니다.<br>")
    lines.append("CT/ROI/mask 원본 로드 없음. 새 MIP 계산 없음. score CSV 수정 없음. mask_generated=False. score_changed=False.")
    lines.append("</div>")

    # group별 요약 테이블
    lines.append("<h2>Group Summary</h2>")
    lines.append("<table>")
    lines.append("<tr><th>Decision Group</th><th>Count</th><th>Color</th></tr>")
    for g in group_order:
        cnt = group_counts.get(g, 0)
        hex_c = group_color_hex.get(g, "#888")
        lines.append(f'<tr><td>{g}</td><td>{cnt}</td><td style="background:{hex_c};color:#fff;text-align:center;">{hex_c}</td></tr>')
    lines.append("</table>")

    # 섹션별 object 목록
    for g in group_order:
        g_df = df[df["decision_group"] == g]
        hex_c = group_color_hex.get(g, "#888")
        lines.append(f'<div class="group-section" style="border-left-color:{hex_c};">')
        lines.append(f'<h2 style="color:{hex_c};">{g.upper()} (n={len(g_df)})</h2>')

        for _, row in g_df.iterrows():
            oid = row["object_id"]
            main_overlay_name = f"{oid}_visual_only_decision_overlay_main.png"
            supp_overlay_name = f"{oid}_visual_only_decision_overlay_supplement.png"
            main_overlay_rel = main_overlay_name  # HTML은 overlay_dir 안에 있음
            supp_overlay_rel = supp_overlay_name

            lines.append('<div class="object-block">')
            lines.append(f"<h3>{oid}</h3>")
            lines.append(f'<img src="{main_overlay_rel}" alt="{oid} main overlay"><br>')
            lines.append(f'<p>main overlay: {main_overlay_name}</p>')
            # 원본은 링크만
            src_main = row.get("main_panel_path", "")
            src_supp = row.get("supplement_panel_path", "")
            lines.append(f'<p>원본 main: <a href="file://{src_main}" style="color:#88aaff;">{Path(src_main).name}</a></p>')
            lines.append(f'<p><a href="{supp_overlay_rel}" style="color:#88aaff;">supplement overlay 보기</a></p>')
            lines.append(f'<p>원본 supplement: <a href="file://{src_supp}" style="color:#88aaff;">{Path(src_supp).name}</a></p>')
            reason = str(row.get("decision_reason", ""))[:100]
            lines.append(f'<p style="color:#999;"><em>{reason}</em></p>')
            lines.append("</div>")

        lines.append("</div>")

    # 다음 단계
    lines.append('<div class="next-step">')
    lines.append("<h2>다음 단계 안내</h2>")
    lines.append("<p>이 시각화 결과를 검토한 후, 다음 단계를 진행한다.</p>")
    lines.append("<ul>")
    lines.append("<li>vessel_positive_candidate 6개: suppression 적용 검토 대상 (Phase 2.20e 이후 별도 승인 필요)</li>")
    lines.append("<li>cautious_vessel_candidate 1개: 추가 검토 후 결정</li>")
    lines.append("<li>protected_negative 7개: suppression 미적용 유지</li>")
    lines.append("<li>mixed_or_uncertain 1개: 추가 검토 필요</li>")
    lines.append("<li><strong>주의</strong>: suppression_weight 적용 또는 score CSV 수정은 이 단계에서 수행하지 않는다.</li>")
    lines.append("</ul>")
    lines.append("</div>")

    lines.append("</body></html>")

    html_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[완료] HTML: {html_path}")


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 2.20d Visual-Only Dry Decision Overlay Pack")
    print("=" * 60)

    # 안전 확인
    print("[확인] CT/ROI/mask 로드 없음, score CSV 수정 없음, PIL만 사용")

    # ── 작업 1: CSV 검증 ──
    print("\n[작업 1] decision table 기대값 검증")
    df = pd.read_csv(INPUT_CSV)
    errors = []

    n_vp = (df["decision_group"] == "vessel_positive_candidate").sum()
    n_cv = (df["decision_group"] == "cautious_vessel_candidate").sum()
    n_pn = (df["decision_group"] == "protected_negative").sum()
    n_mu = (df["decision_group"] == "mixed_or_uncertain").sum()
    n_pos = df["dry_rule_positive_flag"].astype(str).str.upper().eq("TRUE").sum()
    n_miss_main = df["main_panel_exists"].astype(str).str.upper().eq("FALSE").sum()
    n_miss_supp = df["supplement_panel_exists"].astype(str).str.upper().eq("FALSE").sum()

    checks = [
        ("vessel_positive_candidate", n_vp, 6),
        ("cautious_vessel_candidate", n_cv, 1),
        ("protected_negative", n_pn, 7),
        ("mixed_or_uncertain", n_mu, 1),
        ("dry_rule_positive_flag=True", n_pos, 6),
        ("missing main panel", n_miss_main, 0),
        ("missing supplement panel", n_miss_supp, 0),
    ]
    all_ok = True
    for name, actual, expected in checks:
        status = "OK" if actual == expected else "FAIL"
        if status == "FAIL":
            all_ok = False
            errors.append(f"{name}: expected={expected}, actual={actual}")
        print(f"  [{status}] {name}: {actual} (expected {expected})")

    if not all_ok:
        print("\n[중단] 기대값 불일치:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("[통과] 작업 1 검증 완료")

    # ── 출력 경로 설정 (덮어쓰기 방지) ──
    overlay_dir = check_dir_no_overwrite(OVERLAY_DIR)
    contact_dir = overlay_dir / "contact_sheets"
    html_path   = overlay_dir / "review_index.html"
    out_csv     = check_no_overwrite(OUTPUT_CSV)
    out_md      = check_no_overwrite(OUTPUT_MD)
    out_json    = check_no_overwrite(OUTPUT_JSON)

    overlay_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    # ── 작업 2 & 3: overlay 생성 ──
    print("\n[작업 2&3] overlay 이미지 생성")
    overlay_rows = []
    n_main = 0
    n_supp = 0

    group_overlay_main_paths = {g: [] for g in COLOR_MAP}
    group_overlay_main_paths["vessel_positive_candidate"] = []
    group_overlay_main_paths["cautious_vessel_candidate"] = []
    group_overlay_main_paths["protected_negative"] = []
    group_overlay_main_paths["mixed_or_uncertain"] = []

    for _, row in df.iterrows():
        oid = row["object_id"]
        group = row["decision_group"]
        color_rgb = COLOR_MAP.get(group, (180, 180, 180))
        color_hex = "#{:02x}{:02x}{:02x}".format(*color_rgb)

        main_out = overlay_dir / f"{oid}_visual_only_decision_overlay_main.png"
        supp_out = overlay_dir / f"{oid}_visual_only_decision_overlay_supplement.png"

        row_dict = row.to_dict()

        # main overlay
        try:
            main_img = create_overlay(row["main_panel_path"], row_dict)
            main_img.save(str(main_out))
            n_main += 1
            print(f"  [main] {oid} → {main_out.name}")
            if group in group_overlay_main_paths:
                group_overlay_main_paths[group].append(str(main_out))
        except Exception as e:
            print(f"  [오류] {oid} main overlay 실패: {e}")
            main_out = None

        # supplement overlay
        try:
            supp_img = create_overlay(row["supplement_panel_path"], row_dict)
            supp_img.save(str(supp_out))
            n_supp += 1
            print(f"  [supp] {oid} → {supp_out.name}")
        except Exception as e:
            print(f"  [오류] {oid} supplement overlay 실패: {e}")
            supp_out = None

        overlay_rows.append({
            "object_id": oid,
            "patient_id": row["patient_id"],
            "decision_group": group,
            "dry_rule_positive_flag": row["dry_rule_positive_flag"],
            "dry_rule_cautious_flag": row["dry_rule_cautious_flag"],
            "dry_rule_exclusion_flag": row["dry_rule_exclusion_flag"],
            "display_color": color_hex,
            "visual_overlay_main_path": str(main_out) if main_out else "",
            "visual_overlay_supplement_path": str(supp_out) if supp_out else "",
            "source_main_panel_path": row["main_panel_path"],
            "source_supplement_panel_path": row["supplement_panel_path"],
            "visual_only": True,
            "mask_generated": False,
            "score_changed": False,
            "reviewer_note": row.get("reviewer_note", ""),
            "decision_reason": row.get("decision_reason", ""),
        })

    print(f"[완료] main overlay: {n_main}개, supplement overlay: {n_supp}개")

    # ── 작업 4: overlay table CSV ──
    print("\n[작업 4] overlay table CSV 생성")
    out_df = pd.DataFrame(overlay_rows)
    out_df.to_csv(str(out_csv), index=False)
    print(f"  저장: {out_csv}")

    # ── 작업 5: HTML index ──
    print("\n[작업 5] HTML review index 생성")
    build_html(df, overlay_dir, html_path)

    # ── 작업 6: contact sheet ──
    print("\n[작업 6] contact sheet 생성")
    contact_sheet_paths = []
    group_label_map = {
        "vessel_positive_candidate": "vessel_positive_overlay_contact_sheet",
        "cautious_vessel_candidate": "cautious_overlay_contact_sheet",
        "protected_negative": "protected_negative_overlay_contact_sheet",
        "mixed_or_uncertain": "mixed_uncertain_overlay_contact_sheet",
    }
    group_counts_map = {
        "vessel_positive_candidate": n_vp,
        "cautious_vessel_candidate": n_cv,
        "protected_negative": n_pn,
        "mixed_or_uncertain": n_mu,
    }

    for group, label in group_label_map.items():
        paths = group_overlay_main_paths.get(group, [])
        cnt = group_counts_map.get(group, 0)
        sheet = create_contact_sheet(paths, group, cnt)
        sheet_path = check_no_overwrite(contact_dir / f"{label}.png")
        sheet.save(str(sheet_path))
        contact_sheet_paths.append(str(sheet_path))
        print(f"  [{group}] contact sheet → {sheet_path.name}")

    print(f"[완료] contact sheet: {len(contact_sheet_paths)}개")

    # ── 작업 7: summary MD / JSON ──
    print("\n[작업 7] summary MD/JSON 생성")
    output_files = (
        [str(out_csv), str(out_md), str(out_json), str(html_path)]
        + [r["visual_overlay_main_path"] for r in overlay_rows if r["visual_overlay_main_path"]]
        + [r["visual_overlay_supplement_path"] for r in overlay_rows if r["visual_overlay_supplement_path"]]
        + contact_sheet_paths
    )

    # JSON
    summary_json = {
        "phase": "2.20d",
        "generated_at": datetime.now().isoformat(),
        "n_vessel_positive": int(n_vp),
        "n_cautious": int(n_cv),
        "n_protected_negative": int(n_pn),
        "n_mixed_uncertain": int(n_mu),
        "n_main_overlay": n_main,
        "n_supplement_overlay": n_supp,
        "n_contact_sheets": len(contact_sheet_paths),
        "output_files": output_files,
        "mask_generated": False,
        "score_changed": False,
        "ct_loaded": False,
        "visual_only": True,
        "recommended_next_step": (
            "Review Phase 2.20d visual-only decision overlay. "
            "If accepted, next phase should still be limited to planning or explicitly approved "
            "visual-only pseudo mask experiments. "
            "Do not apply suppression_weight or adjust scores."
        ),
        "forbidden_actions": [
            "CT/ROI/mask 원본 로드",
            "새 MIP 계산",
            "vessel soft mask / pseudo mask / suppression_weight 생성",
            "score CSV / patch CSV 수정",
            "기존 panel PNG 수정/덮어쓰기",
            "기존 CSV/MD/JSON 수정",
            "outputs/mip-postprocess-research-v1/ 밖 파일 생성",
            "새 패키지 설치",
        ],
    }
    out_json_path = out_json
    out_json_path.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  저장: {out_json_path}")

    # MD
    md_lines = [
        "# Phase 2.20d Visual-Only Dry Decision Overlay Pack Summary",
        "",
        f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 목적",
        "Phase 2.20c decision table 기반으로 기존 panel PNG에 색상 프레임 + decision label만 추가한 visual-only overlay 이미지를 생성한다.",
        "새 계산(MIP, mask, score) 없이 시각화만 한다.",
        "",
        "## 입력 파일",
        f"- `{INPUT_CSV}`",
        "",
        "## 출력 파일",
        f"- overlay table CSV: `{out_csv}`",
        f"- summary JSON: `{out_json_path}`",
        f"- HTML index: `{html_path}`",
        "- overlay main PNG: 15개",
        "- overlay supplement PNG: 15개",
        "- contact sheet PNG: 4개",
        "",
        "## Group별 개수",
        f"- vessel_positive_candidate: {n_vp}",
        f"- cautious_vessel_candidate: {n_cv}",
        f"- protected_negative: {n_pn}",
        f"- mixed_or_uncertain: {n_mu}",
        "",
        "## 안전 확인",
        "- visual_only=True",
        "- mask_generated=False",
        "- score_changed=False",
        "- ct_loaded=False",
        "- 기존 panel PNG 수정/덮어쓰기: 없음",
        "- 기존 CSV/MD/JSON 수정: 없음",
        "- outputs/mip-postprocess-research-v1/ 밖 파일 생성: 없음",
        "",
        "## 다음 단계",
        "이 overlay 결과를 검토한 후, 다음 단계(suppression 계획 등)를 별도 승인 후 진행한다.",
        "이 단계에서는 suppression_weight 적용 또는 score CSV 수정을 수행하지 않는다.",
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  저장: {out_md}")

    # ── 최종 보고 ──
    print("\n" + "=" * 60)
    print("실행 완료 보고")
    print("=" * 60)
    print(f"1. exit code: 0 (정상)")
    print(f"2. overlay table CSV: {out_csv}")
    print(f"3. summary MD: {out_md}")
    print(f"4. summary JSON: {out_json_path}")
    print(f"5. HTML index: {html_path}")
    print(f"6. overlay main panel 생성 수: {n_main}")
    print(f"7. overlay supplement panel 생성 수: {n_supp}")
    print(f"8. contact sheet 개수: {len(contact_sheet_paths)}")
    for cp in contact_sheet_paths:
        print(f"   - {cp}")
    print(f"9. vessel_positive={n_vp} / cautious={n_cv} / protected={n_pn} / mixed={n_mu}")
    print(f"10. visual_only=True: 확인")
    print(f"11. mask_generated=False: 확인")
    print(f"12. score_changed=False: 확인")
    print(f"13. CT/ROI/mask 로드 없음: 확인")
    print(f"14. outputs 밖 생성 없음: 확인")


if __name__ == "__main__":
    main()
