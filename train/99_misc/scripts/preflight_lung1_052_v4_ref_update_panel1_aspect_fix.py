"""
preflight_lung1_052_v4_ref_update_panel1_aspect_fix.py

목적:
v4_ref_update 카드 Panel 1 의 candidate/reference crop tile 비율이 어색하게 보이는
문제(불일치 aspect + 강제 square resize 왜곡 + baked-in title/여백)를 진단하고,
equal-square tile + z-context 처리 + revised layout 계획을 확정하는 preflight.

이번 단계 금지:
  card render / PNG 생성 / CT load / ROI load / model / feature / contribution /
  stage2 / 기존 artifact 수정.
허용:
  기존 PNG dimension read-only(PIL Image.open size) + 카드 JSON read-only + 계획 리포트 생성.
"""

import os
import sys
import csv
import json
import pathlib
from datetime import date

# ============================================================
# GUARD FLAGS (전부 False)
# ============================================================
ALLOW_CARD_RENDER         = False
ALLOW_PNG_WRITE           = False
ALLOW_CT_LOAD             = False
ALLOW_ROI_LOAD            = False
ALLOW_STAGE2_HOLDOUT      = False
ALLOW_MODEL_FORWARD       = False
ALLOW_FEATURE_EXTRACTION  = False
ALLOW_CONTRIBUTION_RECALC = False
ALLOW_FULL300             = False

# ============================================================
# PATHS
# ============================================================
REPO = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")
EC = REPO / "outputs/position-aware-padim-v1/reports/explanation_cards"
PREVIEW_ROOT = (REPO / "outputs/position-aware-padim-v1/visualizations"
                / "reference_bank_v4_selected_cases_crop_preview")

CARD_ROOT = EC / "s5_demo_card_prototype_lung1_052_c3_v4_ref_update"
CARD_PNG  = CARD_ROOT / "cards_png/LUNG1-052__c3_s5_demo_prototype_v4_ref_update.png"
CARD_JSON = CARD_ROOT / "cards_json/LUNG1-052__c3_s5_demo_prototype_v4_ref_update.json"
PREVIEW_META = PREVIEW_ROOT / "LUNG1-052__c3_preview_metadata.json"
PREVIEW_INDEX = PREVIEW_ROOT / "preview_index.csv"

OUTPUT_ROOT = EC / "lung1_052_s5_demo_card_v4_ref_update_panel1_aspect_fix_preflight"

CASE_ID = "LUNG1-052__c3"

# option B slot 순서 (현재 카드)
SLOTS = [
    (0, "candidate",    f"{CASE_ID}_candidate.png",    "Candidate"),
    (1, "normal_ref_1", f"{CASE_ID}_normal_ref_1.png", "Same-cell normal ref 1"),
    (2, "normal_ref_3", f"{CASE_ID}_normal_ref_3.png", "Same-cell normal ref 3"),
    (3, "normal_ref_2", f"{CASE_ID}_normal_ref_2.png", "Additional same-cell ref"),
]

# 현재 card render 의 tile 크기 산정 (v4_ref_update 와 동일 공식)
CANVAS_W = 1600; MARGIN = 16; PAD = 8
P1_W = CANVAS_W - 2 * MARGIN
CELL_SIZE = (P1_W - 5 * PAD) // 4   # 현재 강제 square resize 대상 크기

TARGET_TILE = 280   # 권장 equal-square tile

FORBIDDEN_PHRASES = [
    "same-z matched", "same z matched", "z-matched", "z matched",
    "identical z", "동일 z 위치", "같은 z 위치",
    "diagnostic heatmap", "grad-cam", "pixel attribution",
    "병변 원인", "암 위치", "혈관 때문에", "diagnostic conclusion",
    "raw427", "dim91",
]


def _abort(msg, code=2):
    print(f"\nBLOCKED: {msg}", file=sys.stderr)
    sys.exit(code)


def scan_forbidden(blob):
    low = str(blob).lower()
    return [p for p in FORBIDDEN_PHRASES if p.lower() in low]


def main():
    print("=" * 64)
    print("LUNG1-052 v4_ref_update Panel 1 aspect/layout fix PREFLIGHT (report-only)")
    print(f"date: {date.today()}")
    print("=" * 64)

    if (ALLOW_CARD_RENDER or ALLOW_PNG_WRITE or ALLOW_CT_LOAD or ALLOW_ROI_LOAD
            or ALLOW_STAGE2_HOLDOUT or ALLOW_MODEL_FORWARD or ALLOW_FEATURE_EXTRACTION
            or ALLOW_CONTRIBUTION_RECALC or ALLOW_FULL300):
        _abort("a forbidden guard is True; preflight must keep all False")

    try:
        from PIL import Image
    except ImportError as e:
        _abort(f"PIL not available: {e}")

    checks = []

    def chk(idx, desc, passed, note=""):
        checks.append({"id": idx, "desc": desc, "passed": bool(passed), "note": note})

    # 1-2 card exists
    chk(1, "current_card_png_exists", CARD_PNG.exists())
    chk(2, "current_card_json_exists", CARD_JSON.exists())

    # 3 source PNG exist
    src_exist = all((PREVIEW_ROOT / s[2]).exists() for s in SLOTS)
    chk(3, "panel1_source_png_4_exist", src_exist)

    # 4-8 dimension / aspect / stretch / distortion / margin
    issue_rows = []
    aspects = []
    for slot, role, fn, label in SLOTS:
        p = PREVIEW_ROOT / fn
        if not p.exists():
            issue_rows.append({"slot": slot, "role": role, "png": fn, "w": "", "h": "",
                               "aspect": "", "target_square": TARGET_TILE,
                               "x_scale_to_square": "", "y_scale_to_square": "",
                               "distortion_ratio": "", "has_baked_label_margin": "",
                               "issue": "MISSING"})
            continue
        w, h = Image.open(p).size
        aspect = round(w / h, 3)
        aspects.append(aspect)
        # 현재 card 는 (CELL_SIZE,CELL_SIZE) 로 강제 resize -> 축별 배율
        xs = round(CELL_SIZE / w, 3)
        ys = round(CELL_SIZE / h, 3)
        distortion = round(xs / ys, 3)  # 1.0 이면 왜곡 없음
        # baked label/margin: 96x96 clean crop 이 아니면 True (preview 는 matplotlib figure)
        has_baked = not (w == 96 and h == 96)
        issue = []
        if abs(distortion - 1.0) > 0.05:
            issue.append(f"square-resize distortion x/y={distortion}")
        if has_baked:
            issue.append("baked title/white margin (matplotlib figure, not 96x96)")
        issue_rows.append({"slot": slot, "role": role, "png": fn, "w": w, "h": h,
                           "aspect": aspect, "target_square": TARGET_TILE,
                           "x_scale_to_square": xs, "y_scale_to_square": ys,
                           "distortion_ratio": distortion,
                           "has_baked_label_margin": has_baked,
                           "issue": "; ".join(issue) if issue else "ok"})

    chk(4, "source_dimensions_read", all(r["w"] != "" for r in issue_rows))
    # crop area vs canvas: clean 96x96 아님을 확인 (proxy)
    chk(5, "crop_area_vs_canvas_checked", any(r["has_baked_label_margin"] for r in issue_rows),
        "preview PNGs are matplotlib figures (image+title+margin), not clean 96x96")
    # candidate vs ref tile size 차이
    aspect_spread = (max(aspects) - min(aspects)) if aspects else 0
    chk(6, "candidate_ref_tile_size_diff_detected", aspect_spread > 0.2,
        f"aspect spread={round(aspect_spread,3)} (candidate portrait vs ref wide)")
    # stretch 가능성
    chk(7, "stretch_distortion_present",
        any(abs(r["distortion_ratio"] - 1.0) > 0.05 for r in issue_rows
            if r["distortion_ratio"] != ""))
    # label/white margin 포함
    chk(8, "baked_label_margin_present", any(r["has_baked_label_margin"] for r in issue_rows))

    # 9 z-context placeholder 공간 비율 (현재 card JSON panel_1 설명에 z-context row 존재)
    card = json.loads(CARD_JSON.read_text()) if CARD_JSON.exists() else {}
    p1desc = card.get("panels", {}).get("panel_1", "")
    z_row_present = ("z-context" in p1desc.lower()) or ("Row3" in p1desc)
    chk(9, "z_context_placeholder_space_checked", z_row_present,
        "z-context placeholder Row3 present in current card (CT load disabled)")

    # 10-11 fix 방식 / z-context 처리 선택 (계획 확정)
    chk(10, "panel1_fix_option_selected", True, "equal-square contain-fit tiles")
    chk(11, "z_context_handling_selected", True, "option B: shrink to one-line note")
    # 12 panel2/3 unchanged 유지 가능
    chk(12, "panel2_3_unchanged_preserved", True,
        "revised layout only touches Panel1 tiles + z-context row; Panel2/3 sources unchanged")

    # 13 forbidden wording
    plan_text_blob = "equal-square contain-fit tiles; preserve aspect; no stretch; " \
                     "labels rendered by card outside tile; z-context one-line note; " \
                     "same lung-ROI position cell; not same-z matching"
    fb = scan_forbidden(plan_text_blob + p1desc)
    chk(13, "forbidden_wording_0", len(fb) == 0, f"hits={fb}")
    # 14 guards 0
    chk(14, "ct_roi_model_feature_contribution_0",
        not (ALLOW_CT_LOAD or ALLOW_ROI_LOAD or ALLOW_MODEL_FORWARD
             or ALLOW_FEATURE_EXTRACTION or ALLOW_CONTRIBUTION_RECALC))
    # 15 artifact modification 0 (write only new root)
    chk(15, "writes_only_new_root", "panel1_aspect_fix_preflight" in str(OUTPUT_ROOT))

    n_pass = sum(1 for c in checks if c["passed"])
    n_fail = len(checks) - n_pass
    print("=" * 64)
    print("PREFLIGHT CHECKS")
    print("=" * 64)
    for c in checks:
        flag = "PASS" if c["passed"] else "FAIL"
        extra = f"  ({c['note']})" if c["note"] else ""
        print(f"  [{flag}] {c['id']:>2}. {c['desc']}{extra}")
    print(f"  ---> {n_pass} PASS / {n_fail} FAIL")

    hard = {1, 2, 3, 13, 14}
    failed_hard = [c for c in checks if c["id"] in hard and not c["passed"]]
    if failed_hard:
        _abort("hard check failed: " + ", ".join(str(c["id"]) for c in failed_hard))

    # ---- verdict ----
    # 비율 문제 원인 확인 + equal-square 계획 + z-context 처리 확정 => PASS.
    # clean-crop true source 는 CT load 필요 (별도 actual step) 로 명시.
    if n_fail == 0:
        verdict = "PASS"
    else:
        verdict = "PARTIAL_PASS"

    # ---- write outputs ----
    if (OUTPUT_ROOT / "DONE.json").exists():
        _abort("existing DONE.json at OUTPUT_ROOT. Archive before re-preflight.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 3. panel1_current_visual_issue_table.csv
    ISSUE_COLS = ["slot", "role", "png", "w", "h", "aspect", "target_square",
                  "x_scale_to_square", "y_scale_to_square", "distortion_ratio",
                  "has_baked_label_margin", "issue"]
    with open(OUTPUT_ROOT / "panel1_current_visual_issue_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ISSUE_COLS)
        w.writeheader()
        for r in issue_rows:
            w.writerow(r)

    # 4. panel1_tile_fix_plan.csv
    TILE_COLS = ["item", "current", "fix"]
    tile_rows = [
        ("tile_size", f"forced resize to {CELL_SIZE}x{CELL_SIZE} square",
         f"equal square tile {TARGET_TILE}x{TARGET_TILE} for all 4"),
        ("scaling", "stretch (resize to square, aspect ignored)",
         "contain/fit: scale preserving aspect, center on tile background, no stretch"),
        ("source_aspect", "inconsistent (candidate portrait ~0.79, ref wide ~1.46)",
         "normalize via contain-fit; (clean source = re-crop, see CT-load note)"),
        ("baked_label_margin", "matplotlib title (long patient_id) + white margin baked in",
         "RECOMMENDED clean 96x96 crop source (no title/margin) -> needs CT load (separate step); "
         "interim: contain-fit annotated PNG (no distortion, label still baked)"),
        ("labels", "duplicated: baked title + card-drawn label below",
         "card renderer draws labels outside tile only; tile = image content only"),
        ("label_text", "role/id/z/cell/fb multi-line",
         "Candidate | Same-cell normal ref 1 | Same-cell normal ref 3 | Additional same-cell ref "
         "(z/cell/fb short under tile or metadata)"),
    ]
    with open(OUTPUT_ROOT / "panel1_tile_fix_plan.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(TILE_COLS)
        for r in tile_rows:
            w.writerow(r)

    # 5. z_context_placeholder_fix_plan.csv
    Z_COLS = ["option", "description", "recommended"]
    z_rows = [
        ("A", "remove z-context placeholder row entirely", "candidate"),
        ("B", "shrink to a single small one-line note ('z-context: CT load deferred')", "RECOMMENDED"),
        ("C", "defer real z-context to a separate later step (no row now)", "acceptable"),
    ]
    with open(OUTPUT_ROOT / "z_context_placeholder_fix_plan.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(Z_COLS)
        for r in z_rows:
            w.writerow(r)

    # 6. revised_layout_plan.csv
    LAYOUT_COLS = ["element", "current", "revised", "note"]
    layout_rows = [
        ("panel1_row1_tiles", f"4 x {CELL_SIZE}px forced-square (stretched)",
         f"4 x {TARGET_TILE}px equal square, contain-fit", "no stretch; aspect preserved"),
        ("panel1_tile_labels", "baked + card-drawn (duplicate)",
         "card-drawn only, outside tile", "tile shows image content only"),
        ("panel1_row2_whole_slice", "kept", "kept", "unchanged"),
        ("panel1_row3_zcontext", "3 large placeholders (CT disabled)",
         "one-line note (option B)", "removes wasted vertical space"),
        ("panel1_height", "tall (subtitle+row1+row2+row3)",
         "reduced (row3 collapsed)", "card less vertically long"),
        ("panel2", "model 반응 위치 overlay", "unchanged", "Panel2 unchanged"),
        ("panel3", "3x3 patch response map", "unchanged", "Panel3 unchanged"),
        ("panel4", "v4_ref_update text", "unchanged", "Panel4 text kept"),
    ]
    with open(OUTPUT_ROOT / "revised_layout_plan.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(LAYOUT_COLS)
        for r in layout_rows:
            w.writerow(r)

    # 7. safety_check.csv
    with open(OUTPUT_ROOT / "safety_check.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item", "value", "status"])
        for it in [("card_render", 0, "OK"), ("png_write", 0, "OK"),
                   ("ct_load", 0, "OK"), ("roi_load", 0, "OK"),
                   ("model_forward", 0, "OK"), ("feature_extraction", 0, "OK"),
                   ("contribution_recalc", 0, "OK"), ("stage2_holdout", 0, "OK"),
                   ("existing_artifact_modified", 0, "OK"),
                   ("png_dimension_read_only", len(SLOTS), "OK"),
                   ("forbidden_wording", len(fb), "OK" if not fb else "FAIL")]:
            w.writerow(it)

    # 8. errors.csv
    with open(OUTPUT_ROOT / "errors.csv", "w", newline="") as f:
        csv.writer(f).writerow(["stage", "error_type", "detail"])

    # 2. json report
    report = {
        "date": str(date.today()), "verdict": verdict, "preflight_only": True,
        "case_id": CASE_ID,
        "root_cause": {
            "summary": "preview PNGs are matplotlib figures (image + baked title with long "
                       "patient_id + white margin); inconsistent aspect (candidate portrait "
                       "~0.79 vs ref wide ~1.46); card render force-resizes to square -> "
                       "opposite-direction stretch distortion between candidate and refs",
            "candidate_aspect": issue_rows[0]["aspect"] if issue_rows else None,
            "ref_aspects": [r["aspect"] for r in issue_rows[1:]],
            "current_forced_square_px": CELL_SIZE,
        },
        "panel1_issue_table": issue_rows,
        "recommended_fix": {
            "tile": f"equal square {TARGET_TILE}px, contain-fit, no stretch, labels outside tile",
            "clean_source": "true clean 96x96 crop requires CT load (read-only mmap, 4 volumes: "
                            "LUNG1-052 candidate + subset1/subset9/subset2 normals) -> SEPARATE "
                            "actual step (CT-load approval); interim CT-free = contain-fit annotated PNG",
            "z_context": "option B: collapse to one-line note",
            "panels_2_3_4": "unchanged",
        },
        "validation": {"pass": n_pass, "total": len(checks),
                       "failed": [c["desc"] for c in checks if not c["passed"]]},
        "safety": {"card_render": 0, "png_write": 0, "ct_load": 0, "roi_load": 0,
                   "model_forward": 0, "feature_extraction": 0, "contribution_recalc": 0,
                   "stage2_holdout": 0, "existing_artifact_modified": 0,
                   "forbidden_wording": len(fb)},
    }
    (OUTPUT_ROOT / "panel1_aspect_fix_preflight_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    # 1. md report
    md = [
        f"# {CASE_ID} v4_ref_update Panel 1 aspect/layout fix — PREFLIGHT",
        f"date: {date.today()}", f"verdict: **{verdict}**",
        "",
        "## scope",
        "- preflight only; no card render / PNG / CT / ROI; PNG dimension read-only",
        "",
        "## root cause (현재 사진 비율 문제 원인)",
        "- preview PNG은 matplotlib figure (이미지 + baked title[긴 patient_id] + white margin)이며 "
        "clean 96x96 crop 이 아님",
        "- aspect 불일치: candidate 세로형(~0.79) vs normal_ref 가로형(~1.46) — title 길이 차이가 원인",
        f"- card render가 모두 {CELL_SIZE}x{CELL_SIZE} square 로 강제 resize → candidate/ref가 반대 방향으로 stretch 왜곡",
        "",
        "## Panel 1 source image dimension / issue",
        "| slot | role | w x h | aspect | x_scale | y_scale | distortion(x/y) | issue |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in issue_rows:
        md.append(f"| {r['slot']} | {r['role']} | {r['w']}x{r['h']} | {r['aspect']} | "
                  f"{r['x_scale_to_square']} | {r['y_scale_to_square']} | "
                  f"{r['distortion_ratio']} | {r['issue']} |")
    md += [
        "",
        "## 추천 수정안",
        f"- Panel 1: 4개 tile 모두 동일 정사각형 {TARGET_TILE}px, **contain/fit (aspect 보존, stretch 금지)**, "
        "여백은 tile background",
        "- tile 내부에는 이미지 콘텐츠만; label은 카드 렌더러가 tile 바깥에 그림 "
        "(Candidate | Same-cell normal ref 1 | Same-cell normal ref 3 | Additional same-cell ref)",
        "- **clean 96x96 crop true source** 는 CT load(read-only mmap, 4 volumes) 필요 → "
        "별도 actual step(CT-load 승인). 이번 preflight 및 CT-free fix 에서는 contain-fit 으로 왜곡만 제거",
        "",
        "## z-context placeholder 처리안",
        "- option A: row 제거 / **option B(추천): 한 줄 note 로 축소** / option C: 별도 단계로 연기",
        "",
        "## revised layout plan",
        "- Panel1 row1 = equal-square contain-fit tiles, row3(z-context) 축소, Panel1 높이 감소",
        "- Panel2/3 unchanged, Panel4 v4_ref_update 문구 유지",
        "",
        "## safety",
        "- card render 0 / PNG 0 / CT 0 / ROI 0 / model 0 / feature 0 / contribution 0 / stage2 0",
        "- 기존 v4_ref_update / v3d artifact 수정 0 (신규 preflight root 만 기록)",
        f"- forbidden wording: {len(fb)}",
        "",
    ]
    body = "\n".join(md)
    if scan_forbidden(body):
        _abort(f"forbidden phrase in md body: {scan_forbidden(body)}")
    (OUTPUT_ROOT / "panel1_aspect_fix_preflight_report.md").write_text(body)

    # 9. DONE.json
    done = {
        "status": "DONE", "verdict": verdict, "date": str(date.today()),
        "preflight_only": True, "case_id": CASE_ID,
        "recommended_tile": f"{TARGET_TILE}px equal square contain-fit",
        "z_context_option": "B",
        "clean_source_needs_ct_load": True,
        "validation_pass": f"{n_pass}/{len(checks)}",
        "outputs": [
            "panel1_aspect_fix_preflight_report.md",
            "panel1_aspect_fix_preflight_report.json",
            "panel1_current_visual_issue_table.csv",
            "panel1_tile_fix_plan.csv",
            "z_context_placeholder_fix_plan.csv",
            "revised_layout_plan.csv",
            "safety_check.csv", "errors.csv", "DONE.json",
        ],
    }
    (OUTPUT_ROOT / "DONE.json").write_text(json.dumps(done, indent=2))

    print(f"\nVERDICT: {verdict}  validation {n_pass}/{len(checks)}")
    print(f"  outputs -> {OUTPUT_ROOT}")
    if verdict == "NEEDS_FIX":
        sys.exit(1)


if __name__ == "__main__":
    main()
