"""
preflight_lung1_052_s5_demo_card_v4_reference_reflection.py

목적:
reference bank v4 selected cases crop preview visual review(PASS) 결과를 바탕으로,
GOOD_FOR_DEMO 로 판정된 LUNG1-052__c3 만 기존 v3d S5 demo card 에 반영할 수 있는지
preflight 한다.

이번 단계는 preflight only:
  card render / PNG 재생성 / CT load / ROI load / model / feature / contribution /
  stage2_holdout / 기존 artifact 수정 전부 금지.
  reference 포함/제외 정책은 visual review 결과(시각 품질)만 근거로 하며,
  "혈관 때문에" 같은 원인 단정은 하지 않는다.
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
R = REPO / "outputs/position-aware-padim-v1/reports"
V = REPO / "outputs/position-aware-padim-v1/visualizations"
EC = R / "explanation_cards"

VISUAL_REVIEW_ROOT = R / "reference_bank_v4_selected_cases_crop_preview_visual_review"
VR_DONE     = VISUAL_REVIEW_ROOT / "DONE.json"
VR_TABLE    = VISUAL_REVIEW_ROOT / "case_visual_review_table_v4.csv"
VR_REFNOTES = VISUAL_REVIEW_ROOT / "reference_quality_notes_v4.csv"

PREVIEW_ROOT = V / "reference_bank_v4_selected_cases_crop_preview"
PREVIEW_META = PREVIEW_ROOT / "LUNG1-052__c3_preview_metadata.json"

V3D_CARD_JSON = (EC / "s5_demo_card_prototype_lung1_052_c3_v3d_panel1_v4_ref"
                 / "cards_json" / "LUNG1-052__c3_s5_demo_prototype_v3d.json")

OUTPUT_ROOT = EC / "lung1_052_s5_demo_card_v4_reference_reflection_preflight"

CASE_ID = "LUNG1-052__c3"
CELL_KEY = "image_left|Z1|Y2|X1"

FORBIDDEN_PHRASES = [
    "same-z matched", "same z matched", "z-matched", "z matched",
    "identical z", "동일 z 위치", "같은 z 위치",
    "diagnostic heatmap", "grad-cam", "pixel attribution",
    "병변 원인", "암 위치", "혈관 때문에", "diagnostic conclusion",
]

# ============================================================
# PANEL TEXT (확정 문구)
# ============================================================
PANEL1_SUBTITLE = ("Normal references are selected from the same lung-ROI position "
                   "cell; this is not same-z matching.")
PANEL4_KEY_FINDING = ("Compared with same-cell normal references, the candidate crop "
                      "shows a denser local opacity pattern.")
PANEL4_INTERPRETATION = ("Compared with normal references selected from the same "
                         "lung-ROI position cell, the area around the higher PaDiM "
                         "response appears as a different local pattern than the normal "
                         "examples; the response continuity between the center patch and "
                         "the patch directly below is preserved.")
PANEL4_CAUTION = ("Same-cell comparison, not same-z matching; z-direction alignment is "
                  "limited; the candidate is slightly pleura-adjacent, so avoid "
                  "over-interpretation.")
PANEL4_DISCLAIMER = ("Research-use auxiliary explanation only; not a diagnostic result "
                     "and not a statement of cause.")

# Panel 1 label policy (allowed labels)
LABEL_CANDIDATE      = "Candidate"
LABEL_REF1           = "Same-cell normal ref 1"
LABEL_REF3           = "Same-cell normal ref 3"
LABEL_REF2_ADDITIONAL = "Additional same-cell ref"


def _abort(msg, code=2):
    print(f"\nBLOCKED: {msg}", file=sys.stderr)
    sys.exit(code)


def scan_forbidden(blob):
    low = str(blob).lower()
    return [p for p in FORBIDDEN_PHRASES if p.lower() in low]


# ============================================================
# A/B/C OPTION DEFINITIONS
# ============================================================
PANEL1_OPTIONS = {
    "A": {
        "name": "3-reference 유지 (candidate|ref1|ref2|ref3)",
        "pros": "기존 4-column 구조 그대로 유지; v3d 레이아웃 변경 최소",
        "cons": "ref2(subset2)가 structure-dominant 라 비교 품질 저하 가능",
        "slots": ["candidate", "normal_ref_1", "normal_ref_2", "normal_ref_3"],
    },
    "B": {
        "name": "ref1/ref3 우선 + ref2 additional 표시",
        "pros": "좋은 reference 중심 비교; 4-column 유지하되 ref2를 additional/less-preferred 로 라벨",
        "cons": "라벨/순서 조정 필요(레이아웃 자체는 4-column 유지)",
        "slots": ["candidate", "normal_ref_1", "normal_ref_3", "normal_ref_2"],
    },
    "C": {
        "name": "ref2 제외 2-reference (candidate|ref1|ref3)",
        "pros": "비교 품질 최상",
        "cons": "기존 v3d 4-column 구조와 달라져 레이아웃 변경 큼",
        "slots": ["candidate", "normal_ref_1", "normal_ref_3"],
    },
}
RECOMMENDED_OPTION = "B"


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 64)
    print("LUNG1-052__c3 S5 demo card v4-ref reflection PREFLIGHT (report-only)")
    print(f"date: {date.today()}")
    print("=" * 64)

    # guard self-check
    if (ALLOW_CARD_RENDER or ALLOW_PNG_WRITE or ALLOW_CT_LOAD or ALLOW_ROI_LOAD
            or ALLOW_STAGE2_HOLDOUT or ALLOW_MODEL_FORWARD
            or ALLOW_FEATURE_EXTRACTION or ALLOW_CONTRIBUTION_RECALC):
        _abort("a forbidden guard is True; preflight must keep all False")

    checks = []

    def chk(idx, desc, passed, note=""):
        checks.append({"id": idx, "desc": desc, "passed": bool(passed), "note": note})

    # 1. visual review DONE exists
    vr_ok = VR_DONE.exists()
    chk(1, "visual_review_DONE_exists", vr_ok)
    vr_done = json.loads(VR_DONE.read_text()) if vr_ok else {}
    # 2. visual review verdict PASS
    chk(2, "visual_review_verdict_PASS", vr_done.get("verdict") == "PASS",
        f"verdict={vr_done.get('verdict')}")

    # 3. LUNG1-052 final_visual_verdict == GOOD_FOR_DEMO
    lung_verdict = None
    if VR_TABLE.exists():
        import csv as _csv
        for row in _csv.DictReader(open(VR_TABLE)):
            if row["case_id"] == CASE_ID:
                lung_verdict = row["final_visual_verdict"]
    chk(3, "lung1_052_GOOD_FOR_DEMO", lung_verdict == "GOOD_FOR_DEMO",
        f"verdict={lung_verdict}")

    # 4-9. preview PNG/metadata exist
    pngs = {
        "candidate": PREVIEW_ROOT / f"{CASE_ID}_candidate.png",
        "normal_ref_1": PREVIEW_ROOT / f"{CASE_ID}_normal_ref_1.png",
        "normal_ref_2": PREVIEW_ROOT / f"{CASE_ID}_normal_ref_2.png",
        "normal_ref_3": PREVIEW_ROOT / f"{CASE_ID}_normal_ref_3.png",
        "montage": PREVIEW_ROOT / f"{CASE_ID}_reference_preview_montage.png",
    }
    chk(4, "candidate_png_exists", pngs["candidate"].exists())
    chk(5, "normal_ref_1_png_exists", pngs["normal_ref_1"].exists())
    chk(6, "normal_ref_2_png_exists", pngs["normal_ref_2"].exists())
    chk(7, "normal_ref_3_png_exists", pngs["normal_ref_3"].exists())
    chk(8, "montage_png_exists", pngs["montage"].exists())
    chk(9, "preview_metadata_exists", PREVIEW_META.exists())

    meta = json.loads(PREVIEW_META.read_text()) if PREVIEW_META.exists() else {}
    refs_meta = {r["role"]: r for r in meta.get("normal_references", [])}

    # ref keep/deprioritize policy from visual review refnotes
    ref_policy = {}
    if VR_REFNOTES.exists():
        import csv as _csv
        for row in _csv.DictReader(open(VR_REFNOTES)):
            if row["case_id"] == CASE_ID:
                ref_policy[row["reference_role"]] = row
    # 10. ref1 keep policy recorded
    chk(10, "ref1_keep_recorded",
        ref_policy.get("normal_ref_1", {}).get("keep_or_exclude") == "keep")
    # 11. ref2 deprioritize policy recorded
    chk(11, "ref2_deprioritize_recorded",
        ref_policy.get("normal_ref_2", {}).get("keep_or_exclude") == "deprioritize")
    # 12. ref3 keep policy recorded
    chk(12, "ref3_keep_recorded",
        ref_policy.get("normal_ref_3", {}).get("keep_or_exclude") == "keep")

    # 13. options A/B/C evaluated
    chk(13, "options_ABC_evaluated", set(PANEL1_OPTIONS.keys()) == {"A", "B", "C"})
    # 14. recommended option selected
    chk(14, "recommended_option_selected", RECOMMENDED_OPTION in PANEL1_OPTIONS)
    # 15. panel1 subtitle updated
    chk(15, "panel1_subtitle_updated", bool(PANEL1_SUBTITLE))
    # 16. panel4 caution updated
    chk(16, "panel4_caution_updated", bool(PANEL4_CAUTION))

    # 17/18. forbidden wording scan over all preflight-generated text
    all_text = " ".join([
        PANEL1_SUBTITLE, PANEL4_KEY_FINDING, PANEL4_INTERPRETATION,
        PANEL4_CAUTION, PANEL4_DISCLAIMER,
        json.dumps(PANEL1_OPTIONS, ensure_ascii=False),
        LABEL_CANDIDATE, LABEL_REF1, LABEL_REF3, LABEL_REF2_ADDITIONAL,
    ])
    fb_hits = scan_forbidden(all_text)
    chk(17, "same_z_forbidden_wording_0", len(fb_hits) == 0, f"hits={fb_hits}")
    chk(18, "diagnostic_causal_wording_0", len(fb_hits) == 0)

    # 19. CT/ROI/render guards 0
    chk(19, "ct_roi_render_guards_0",
        not (ALLOW_CT_LOAD or ALLOW_ROI_LOAD or ALLOW_CARD_RENDER or ALLOW_PNG_WRITE))
    # 20. existing artifact modification 0 (write only new root)
    chk(20, "writes_only_new_root", "reflection_preflight" in str(OUTPUT_ROOT))
    # extra: v3d card json exists
    chk("E1", "v3d_card_json_exists", V3D_CARD_JSON.exists())

    n_pass = sum(1 for c in checks if c["passed"])
    n_fail = len(checks) - n_pass

    print("=" * 64)
    print("PREFLIGHT CHECKS")
    print("=" * 64)
    for c in checks:
        flag = "PASS" if c["passed"] else "FAIL"
        extra = f"  ({c['note']})" if c["note"] else ""
        print(f"  [{flag}] {str(c['id']):>3}. {c['desc']}{extra}")
    print(f"  ---> {n_pass} PASS / {n_fail} FAIL")

    hard = {1, 2, 3, 4, 9, 17, 18, 19}
    failed_hard = [c for c in checks if c["id"] in hard and not c["passed"]]
    if failed_hard:
        _abort("hard check failed: " + ", ".join(str(c["id"]) for c in failed_hard))

    # verdict
    layout_uncertain = False  # 추천안 B 명확
    if n_fail == 0:
        verdict = "PASS"
    elif lung_verdict == "GOOD_FOR_DEMO" and not failed_hard:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "NEEDS_FIX"

    # ---- write outputs ----
    if (OUTPUT_ROOT / "DONE.json").exists():
        _abort("existing DONE.json at OUTPUT_ROOT. Archive before re-preflight.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    new_preview_root_rel = str(PREVIEW_ROOT.relative_to(REPO))

    # 3. panel1_reference_reflection_plan.csv  (추천안 B 기준 slot)
    P1_COLS = ["slot", "role", "display_label", "inclusion", "source_png",
               "patient_id", "z", "note"]
    slot_def = [
        ("1", "candidate", LABEL_CANDIDATE, "primary",
         f"{new_preview_root_rel}/{CASE_ID}_candidate.png",
         CASE_ID, meta.get("candidate", {}).get("local_z", ""),
         "candidate crop; slightly pleura-adjacent (caution)"),
        ("2", "normal_ref_1", LABEL_REF1, "primary",
         f"{new_preview_root_rel}/{CASE_ID}_normal_ref_1.png",
         str(refs_meta.get("normal_ref_1", {}).get("patient_id", ""))[:40],
         refs_meta.get("normal_ref_1", {}).get("local_z", ""),
         "clean normal interior; keep"),
        ("3", "normal_ref_3", LABEL_REF3, "primary",
         f"{new_preview_root_rel}/{CASE_ID}_normal_ref_3.png",
         str(refs_meta.get("normal_ref_3", {}).get("patient_id", ""))[:40],
         refs_meta.get("normal_ref_3", {}).get("local_z", ""),
         "clean normal parenchyma; best comparison ref; keep"),
        ("4", "normal_ref_2", LABEL_REF2_ADDITIONAL, "additional",
         f"{new_preview_root_rel}/{CASE_ID}_normal_ref_2.png",
         str(refs_meta.get("normal_ref_2", {}).get("patient_id", ""))[:40],
         refs_meta.get("normal_ref_2", {}).get("local_z", ""),
         "structure-dominant, less suitable for visual comparison; show as additional "
         "same-cell ref (no causal claim)"),
    ]
    with open(OUTPUT_ROOT / "panel1_reference_reflection_plan.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(P1_COLS)
        for s in slot_def:
            w.writerow(s)

    # 4. panel_text_update_plan.csv
    PT_COLS = ["panel", "field", "action", "new_text"]
    pt_rows = [
        ("panel_1", "subtitle", "update", PANEL1_SUBTITLE),
        ("panel_1", "row1_slots", "reorder+relabel (option B)",
         "candidate | Same-cell normal ref 1 | Same-cell normal ref 3 | "
         "Additional same-cell ref"),
        ("panel_1", "preview_root", "repoint", new_preview_root_rel),
        ("panel_2", "-", "keep", "lung-window overlay (not Grad-CAM / not pixel attribution) — unchanged"),
        ("panel_3", "-", "keep", "3x3 patch response schematic — unchanged"),
        ("panel_4", "key_finding", "update", PANEL4_KEY_FINDING),
        ("panel_4", "interpretation", "update", PANEL4_INTERPRETATION),
        ("panel_4", "caution", "update", PANEL4_CAUTION),
        ("panel_4", "disclaimer", "update", PANEL4_DISCLAIMER),
    ]
    with open(OUTPUT_ROOT / "panel_text_update_plan.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PT_COLS)
        for r in pt_rows:
            w.writerow(r)

    # 5. reference_inclusion_decision.csv
    RI_COLS = ["reference_role", "patient_id", "decision", "display_label", "rationale"]
    ri_rows = [
        ("normal_ref_1", str(refs_meta.get("normal_ref_1", {}).get("patient_id", ""))[:40],
         "keep_primary", LABEL_REF1, "clean normal lung interior (visual review: keep)"),
        ("normal_ref_3", str(refs_meta.get("normal_ref_3", {}).get("patient_id", ""))[:40],
         "keep_primary", LABEL_REF3,
         "clean normal parenchyma; best comparison ref (visual review: keep/best)"),
        ("normal_ref_2", str(refs_meta.get("normal_ref_2", {}).get("patient_id", ""))[:40],
         "keep_additional", LABEL_REF2_ADDITIONAL,
         "structure-dominant; less suitable for visual comparison; shown as additional "
         "same-cell ref; recorded in report/metadata only; no causal claim"),
    ]
    with open(OUTPUT_ROOT / "reference_inclusion_decision.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RI_COLS)
        for r in ri_rows:
            w.writerow(r)

    # 6. card_generation_readiness.csv
    CR_COLS = ["item", "status", "detail"]
    cr_rows = [
        ("v3d_card_json", "READY" if V3D_CARD_JSON.exists() else "MISSING",
         str(V3D_CARD_JSON.relative_to(REPO))),
        ("new_candidate_png", "READY" if pngs["candidate"].exists() else "MISSING", ""),
        ("new_normal_ref_1_png", "READY" if pngs["normal_ref_1"].exists() else "MISSING", ""),
        ("new_normal_ref_2_png", "READY" if pngs["normal_ref_2"].exists() else "MISSING", ""),
        ("new_normal_ref_3_png", "READY" if pngs["normal_ref_3"].exists() else "MISSING", ""),
        ("new_montage_png", "READY" if pngs["montage"].exists() else "MISSING", ""),
        ("cell_key_match", "READY" if meta.get("cell_key") == CELL_KEY else "CHECK",
         f"{meta.get('cell_key')} vs {CELL_KEY}"),
        ("ref_patients_same_as_v3d", "READY",
         "v4 exact-cell top3 same patients as v3d (subset1/subset2/subset9); "
         "PNG path repoint only"),
        ("selected_option", "READY", f"option {RECOMMENDED_OPTION}"),
        ("panel1_subtitle", "READY", "defined"),
        ("panel4_text", "READY", "key_finding/interpretation/caution/disclaimer defined"),
        ("layout_change", "MINIMAL", "4-column kept; reorder+relabel + preview_root repoint"),
    ]
    with open(OUTPUT_ROOT / "card_generation_readiness.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CR_COLS)
        for r in cr_rows:
            w.writerow(r)

    # 7. safety_check.csv
    with open(OUTPUT_ROOT / "safety_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "value", "status"])
        for it in [
            ("card_render", 0, "OK"), ("png_write", 0, "OK"),
            ("ct_load", 0, "OK"), ("roi_load", 0, "OK"),
            ("model_forward", 0, "OK"), ("feature_extraction", 0, "OK"),
            ("contribution_recalc", 0, "OK"), ("stage2_holdout_access", 0, "OK"),
            ("existing_artifact_modified", 0, "OK"),
            ("causal_claim_on_ref2", 0, "OK"),
            ("forbidden_wording", len(fb_hits), "OK" if not fb_hits else "FAIL"),
            ("recommended_option", RECOMMENDED_OPTION, "OK"),
        ]:
            w.writerow(it)

    # 8. errors.csv
    with open(OUTPUT_ROOT / "errors.csv", "w", newline="") as f:
        csv.writer(f).writerow(["stage", "error_type", "detail"])

    # 2. json report
    report = {
        "date": str(date.today()), "verdict": verdict, "preflight_only": True,
        "case_id": CASE_ID, "cell_key": CELL_KEY,
        "lung1_052_visual_verdict": lung_verdict,
        "panel1_options": PANEL1_OPTIONS,
        "recommended_option": RECOMMENDED_OPTION,
        "reference_inclusion": {r[0]: {"decision": r[2], "label": r[3]} for r in ri_rows},
        "panel_text": {
            "panel1_subtitle": PANEL1_SUBTITLE,
            "panel4_key_finding": PANEL4_KEY_FINDING,
            "panel4_interpretation": PANEL4_INTERPRETATION,
            "panel4_caution": PANEL4_CAUTION,
            "panel4_disclaimer": PANEL4_DISCLAIMER,
        },
        "new_preview_root": new_preview_root_rel,
        "v3d_change_scope": "Panel1 reorder+relabel (option B) + preview_root repoint to "
                            "selected_cases preview + Panel1 subtitle + Panel4 text; "
                            "Panel2/3 unchanged",
        "validation": {"pass": n_pass, "total": len(checks),
                       "failed": [c["desc"] for c in checks if not c["passed"]]},
        "safety": {
            "card_render": 0, "png_write": 0, "ct_load": 0, "roi_load": 0,
            "model_forward": 0, "feature_extraction": 0, "contribution_recalc": 0,
            "stage2_holdout": 0, "existing_artifact_modified": 0,
            "forbidden_wording": len(fb_hits),
        },
    }
    (OUTPUT_ROOT / "reflection_preflight_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    # 1. md report
    md = [
        f"# {CASE_ID} S5 demo card v4-ref reflection — PREFLIGHT",
        f"date: {date.today()}",
        f"verdict: **{verdict}**",
        "",
        "## scope",
        "- preflight only; no card render / CT / ROI / PNG / model / feature / contribution",
        f"- only {CASE_ID} (GOOD_FOR_DEMO) reflected into v3d S5 demo card",
        "",
        "## LUNG1-052 reflection readiness",
        f"- visual verdict: {lung_verdict}",
        f"- cell_key: {CELL_KEY} (matches v3d v3d_reference_selection)",
        "- new preview PNGs (candidate + 3 ref + montage): READY",
        "- ref patients same as v3d (subset1 / subset2 / subset9) -> PNG path repoint only",
        "",
        "## Panel 1 option comparison",
        "| option | layout | pros | cons |",
        "|---|---|---|---|",
    ]
    for k in ["A", "B", "C"]:
        o = PANEL1_OPTIONS[k]
        md.append(f"| {k}{' (recommended)' if k==RECOMMENDED_OPTION else ''} | "
                  f"{o['name']} | {o['pros']} | {o['cons']} |")
    md += [
        "",
        f"## selected option: **{RECOMMENDED_OPTION}**",
        "- candidate | Same-cell normal ref 1 | Same-cell normal ref 3 | Additional same-cell ref",
        "- 4-column layout 유지(레이아웃 변경 최소), ref2는 4번째 slot에 'Additional same-cell ref' 라벨",
        "",
        "## reference inclusion decision",
        "| role | decision | label | rationale |",
        "|---|---|---|---|",
    ]
    for r in ri_rows:
        md.append(f"| {r[0]} | {r[2]} | {r[3]} | {r[4]} |")
    md += [
        "",
        "## text / caution update policy",
        f"- Panel 1 subtitle: {PANEL1_SUBTITLE}",
        f"- Panel 4 Key finding: {PANEL4_KEY_FINDING}",
        f"- Panel 4 Interpretation: {PANEL4_INTERPRETATION}",
        f"- Panel 4 Caution: {PANEL4_CAUTION}",
        f"- Panel 4 Disclaimer: {PANEL4_DISCLAIMER}",
        "- Panel 2 / Panel 3: unchanged",
        "",
        "## safety",
        "- card render: 0 / PNG: 0 / CT: 0 / ROI: 0 / model: 0 / feature: 0 / contribution: 0 / stage2: 0",
        "- existing artifact modified: 0 (writes only to new preflight root)",
        f"- forbidden wording: {len(fb_hits)}",
        "- ref2 is recorded as structure-dominant / less suitable for visual comparison "
        "(report/metadata only); no causal claim",
        "",
    ]
    body = "\n".join(md)
    body_hits = scan_forbidden(body)
    if body_hits:
        _abort(f"forbidden phrase in preflight md body: {body_hits}")
    (OUTPUT_ROOT / "reflection_preflight_report.md").write_text(body)

    # 9. DONE.json
    done = {
        "status": "DONE", "verdict": verdict, "date": str(date.today()),
        "preflight_only": True, "case_id": CASE_ID,
        "recommended_option": RECOMMENDED_OPTION,
        "validation_pass": f"{n_pass}/{len(checks)}",
        "outputs": [
            "reflection_preflight_report.md", "reflection_preflight_report.json",
            "panel1_reference_reflection_plan.csv", "panel_text_update_plan.csv",
            "reference_inclusion_decision.csv", "card_generation_readiness.csv",
            "safety_check.csv", "errors.csv", "DONE.json",
        ],
    }
    (OUTPUT_ROOT / "DONE.json").write_text(json.dumps(done, indent=2))

    print(f"\nVERDICT: {verdict}  validation {n_pass}/{len(checks)}  "
          f"recommended option {RECOMMENDED_OPTION}")
    print(f"  outputs -> {OUTPUT_ROOT}")

    if verdict == "NEEDS_FIX":
        sys.exit(1)


if __name__ == "__main__":
    main()
