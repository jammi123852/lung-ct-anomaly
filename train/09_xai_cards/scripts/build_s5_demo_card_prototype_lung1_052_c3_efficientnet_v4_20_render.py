#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EfficientNet PaDiM v4.20 전용 LUNG1-052__c3 S5 card — RENDER-CAPABLE.

static-only 버전(build_..._efficientnet_v4_20_static.py)을 기반으로, 실제 draw/read/write 함수를 구현한다.
단, 모든 IO(source image read / card render / PNG write)는 guard가 True일 때만 동작한다.
guard 기본값은 전부 False이므로 본 단계(implementation + static-drycheck)에서는 어떤 파일도 read/write 하지 않는다.

기존 v1 card script/PNG/JSON, static-only script, reference bank outputs, v4_ref_update_aspectfix는 절대 수정하지 않는다.
candidate_position_policy = v1_location_fixed (Panel1 재사용, EfficientNet peak는 Panel2/3에만).
"""
import os, sys, json, csv, argparse

# ===== GUARDS (전부 False) =====
ALLOW_CARD_RENDER                    = False
ALLOW_SOURCE_IMAGE_READ              = False
ALLOW_PNG_WRITE                      = False
ALLOW_CT_LOAD                        = False
ALLOW_MODEL_FORWARD                  = False
ALLOW_FEATURE_EXTRACTION             = False
ALLOW_SCORE_RECOMPUTE                = False
ALLOW_STAGE2_HOLDOUT_ACCESS          = False
ALLOW_EXISTING_ARTIFACT_MODIFICATION = False
ALLOW_MAIN_RENAME                    = False

GUARDS = {
    "ALLOW_CARD_RENDER": ALLOW_CARD_RENDER, "ALLOW_SOURCE_IMAGE_READ": ALLOW_SOURCE_IMAGE_READ,
    "ALLOW_PNG_WRITE": ALLOW_PNG_WRITE, "ALLOW_CT_LOAD": ALLOW_CT_LOAD,
    "ALLOW_MODEL_FORWARD": ALLOW_MODEL_FORWARD, "ALLOW_FEATURE_EXTRACTION": ALLOW_FEATURE_EXTRACTION,
    "ALLOW_SCORE_RECOMPUTE": ALLOW_SCORE_RECOMPUTE, "ALLOW_STAGE2_HOLDOUT_ACCESS": ALLOW_STAGE2_HOLDOUT_ACCESS,
    "ALLOW_EXISTING_ARTIFACT_MODIFICATION": ALLOW_EXISTING_ARTIFACT_MODIFICATION, "ALLOW_MAIN_RENAME": ALLOW_MAIN_RENAME,
}
TURN_ON_FOR_RENDER = ["ALLOW_CARD_RENDER", "ALLOW_SOURCE_IMAGE_READ", "ALLOW_PNG_WRITE"]
REMAIN_FALSE = ["ALLOW_CT_LOAD", "ALLOW_MODEL_FORWARD", "ALLOW_FEATURE_EXTRACTION", "ALLOW_SCORE_RECOMPUTE",
                "ALLOW_STAGE2_HOLDOUT_ACCESS", "ALLOW_EXISTING_ARTIFACT_MODIFICATION", "ALLOW_MAIN_RENAME"]

ROOT = "/home/jinhy/project/lung-ct-anomaly"
PREVIEW = os.path.join(ROOT, "outputs/position-aware-padim-v1/visualizations/reference_bank_v4_selected_cases_crop_preview")
OUTPUT_ROOT = os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/efficientnet_s5_card_lung1_052_c3_v4_20")
V1_CARD_ROOT = os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/explanation_cards/s5_demo_card_prototype_lung1_052_c3_v4_ref_update")

CASE = {
    "case_id": "LUNG1-052__c3", "local_z": 51, "report_slice": 106,
    "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1", "mask": "refined_roi_v4_20_modeB",
    "card_version": "efficientnet_v4_20", "candidate_position_policy": "v1_location_fixed",
}
PANEL1_TILES = [
    ("candidate",    "LUNG1-052__c3_candidate.png",   "Candidate location"),
    ("normal_ref_1", "LUNG1-052__c3_normal_ref_1.png", "Same-cell normal ref 1"),
    ("normal_ref_3", "LUNG1-052__c3_normal_ref_3.png", "Same-cell normal ref 3"),
    ("normal_ref_2", "LUNG1-052__c3_normal_ref_2.png", "Additional same-cell ref 2"),
]
EFF_PEAK   = {"bbox": [288, 112, 320, 144], "score": 21.6854}
V1_PATCH4  = {"bbox": [288, 128, 320, 160], "eff_score": 17.9909}
V1_PATCH7  = {"bbox": [320, 128, 352, 160], "eff_score": 18.2798}
EFF_3X3 = {  # (y0,x0) -> score, None = MISSING
    (288,112):21.6854, (304,112):19.2811, (288,96):18.9322, (288,128):17.9909,
    (304,128):17.7807, (272,112):15.5562, (272,128):14.5310, (272,96):14.3818, (304,96):None,
}
ALLOWED_DISCLAIMERS = [
    "Same-cell normal references are matched by lung-ROI position cell, not by identical z-slice.",
    "Scores are branch-specific and should not be compared as absolute values.",
    "This is a research-use auxiliary explanation, not a diagnostic conclusion.",
    "Panel 1 keeps the previous candidate-centered view to preserve the same-cell reference comparison.",
]
PANEL_TEXT = {
    "panel1_note": "Panel 1 keeps the previous candidate-centered view to preserve the same-cell reference comparison.",
    "panel2_title": "EfficientNet-B0 PaDiM response overview",
    "panel3_title": "EfficientNet patch response map (3x3)",
    "panel4": ("Using EfficientNet-B0 PaDiM features, the response peak is slightly shifted left from the "
               "previous v1-centered patch, while a downward response pattern is partially preserved. "
               "Same-cell normal references are matched by lung-ROI position cell, not by identical z-slice. "
               "Scores are branch-specific and should not be compared as absolute values. "
               "This is a research-use auxiliary explanation, not a diagnostic conclusion."),
}
FORBIDDEN_TERMS = ["diagnostic heatmap","grad-cam","gradcam","pixel attribution","cancer probability",
    "lesion cause","same-z matched","identical z","혈관 때문에 병변","암 위치 확정","진단 확정"]

JSON_REQUIRED_FIELDS = [
    "case_id","branch","mask","card_version","candidate_position_policy","local_z","report_slice",
    "efficientnet_peak_bbox","efficientnet_peak_score","v1_patch4_bbox","v1_patch4_efficientnet_score",
    "v1_patch7_bbox","v1_patch7_efficientnet_score","score_scale_note","same_cell_reference_note",
    "not_diagnostic","not_gradcam","not_pixel_attribution","stage2_holdout_used","model_forward_used",
    "feature_extraction_used","score_recomputed","ct_load_used","source_image_read_used","png_write_used",
]

# =========================================================================
# guard 위반 시 즉시 차단하는 헬퍼
def _require(guard_name, value):
    if not value:
        raise PermissionError(f"GUARD_BLOCKED: {guard_name} is False")

# ---- IO / draw 함수 (구현하되 guard True일 때만 실제 동작) ----
def read_source_tile(path):
    """source PNG tile read. ALLOW_SOURCE_IMAGE_READ True 필요."""
    _require("ALLOW_SOURCE_IMAGE_READ", ALLOW_SOURCE_IMAGE_READ)
    from PIL import Image
    return Image.open(path).convert("RGB")

def draw_panel1(ax_list):
    """Panel1: candidate + same-cell refs tiles 배치 (slot order). ALLOW_CARD_RENDER True 필요."""
    _require("ALLOW_CARD_RENDER", ALLOW_CARD_RENDER)
    import numpy as np
    for ax, (role, fname, label) in zip(ax_list, PANEL1_TILES):
        img = read_source_tile(os.path.join(PREVIEW, fname))
        ax.imshow(np.asarray(img)); ax.set_title(label, fontsize=8); ax.axis("off")

def draw_panel2(ax):
    """Panel2: v1 grid vs EfficientNet peak schematic overlay. ALLOW_CARD_RENDER True 필요."""
    _require("ALLOW_CARD_RENDER", ALLOW_CARD_RENDER)
    import matplotlib.patches as mpatches
    ax.set_title(PANEL_TEXT["panel2_title"], fontsize=8)
    ax.set_xlim(96, 192); ax.set_ylim(352, 256); ax.set_aspect("equal")
    for d, color, lab in [(V1_PATCH4,"tab:blue","v1 patch4 loc"),(V1_PATCH7,"tab:cyan","v1 patch7 loc"),
                          (EFF_PEAK,"tab:red","EfficientNet peak")]:
        y0,x0,y1,x1 = d["bbox"]
        ax.add_patch(mpatches.Rectangle((x0,y0), x1-x0, y1-y0, fill=False, edgecolor=color, lw=2, label=lab))
    ax.legend(fontsize=6, loc="lower right")
    ax.text(0.5,-0.12,"branch-specific score, not absolute comparable with v1",
            transform=ax.transAxes, ha="center", fontsize=6)

def draw_panel3(ax):
    """Panel3: EfficientNet 3x3 patch map. MISSING은 hatched/gray. 보간 금지. ALLOW_CARD_RENDER True 필요."""
    _require("ALLOW_CARD_RENDER", ALLOW_CARD_RENDER)
    import numpy as np, matplotlib.patches as mpatches
    ax.set_title(PANEL_TEXT["panel3_title"], fontsize=8)
    ys = [272,288,304]; xs = [96,112,128]
    vals = [EFF_3X3[(y,x)] for y in ys for x in xs if EFF_3X3[(y,x)] is not None]
    vmin, vmax = min(vals), max(vals)
    for i,y in enumerate(ys):
        for j,x in enumerate(xs):
            v = EFF_3X3[(y,x)]
            if v is None:
                ax.add_patch(mpatches.Rectangle((j,2-i),1,1, facecolor="lightgray", hatch="///", edgecolor="k"))
                ax.text(j+0.5, 2-i+0.5, "MISSING", ha="center", va="center", fontsize=5)
            else:
                t = (v-vmin)/(vmax-vmin+1e-9)
                ax.add_patch(mpatches.Rectangle((j,2-i),1,1, facecolor=(1,1-t*0.7,1-t*0.7), edgecolor="k"))
                ax.text(j+0.5, 2-i+0.5, f"{v:.1f}", ha="center", va="center", fontsize=6)
            if (y,x)==(288,112): ax.text(j+0.5,2-i+0.85,"peak",ha="center",fontsize=5,color="red")
    ax.set_xlim(0,3); ax.set_ylim(0,3); ax.axis("off")

def draw_panel4(ax):
    """Panel4: text box. ALLOW_CARD_RENDER True 필요."""
    _require("ALLOW_CARD_RENDER", ALLOW_CARD_RENDER)
    ax.axis("off")
    ax.text(0.02,0.98, PANEL_TEXT["panel4"], transform=ax.transAxes, va="top", ha="left",
            fontsize=6.5, wrap=True)

def build_metadata():
    """JSON metadata dict (IO 없음 — 항상 호출 가능)."""
    return {
        "case_id": CASE["case_id"], "branch": CASE["branch"], "mask": CASE["mask"],
        "card_version": CASE["card_version"], "candidate_position_policy": CASE["candidate_position_policy"],
        "local_z": CASE["local_z"], "report_slice": CASE["report_slice"],
        "efficientnet_peak_bbox": EFF_PEAK["bbox"], "efficientnet_peak_score": EFF_PEAK["score"],
        "v1_patch4_bbox": V1_PATCH4["bbox"], "v1_patch4_efficientnet_score": V1_PATCH4["eff_score"],
        "v1_patch7_bbox": V1_PATCH7["bbox"], "v1_patch7_efficientnet_score": V1_PATCH7["eff_score"],
        "score_scale_note": "branch-specific, not absolute comparable",
        "same_cell_reference_note": "not same-z matching",
        "not_diagnostic": True, "not_gradcam": True, "not_pixel_attribution": True,
        "stage2_holdout_used": False, "model_forward_used": False, "feature_extraction_used": False,
        "score_recomputed": False, "ct_load_used": False,
        "source_image_read_used": bool(ALLOW_SOURCE_IMAGE_READ), "png_write_used": bool(ALLOW_PNG_WRITE),
    }

def render_card():
    """전체 카드 render + 저장. guard 다중 요구. 본 단계에서는 호출되지 않음."""
    _require("ALLOW_CARD_RENDER", ALLOW_CARD_RENDER)
    _require("ALLOW_SOURCE_IMAGE_READ", ALLOW_SOURCE_IMAGE_READ)
    _require("ALLOW_PNG_WRITE", ALLOW_PNG_WRITE)
    if os.path.isdir(OUTPUT_ROOT):
        raise FileExistsError(f"COLLISION_BLOCKED: {OUTPUT_ROOT} already exists")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(OUTPUT_ROOT, "cards_png"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "cards_json"), exist_ok=True)
    fig = plt.figure(figsize=(12,8))
    p1 = [fig.add_subplot(4,4,k) for k in (1,2,3,4)]
    draw_panel1(p1)
    draw_panel2(fig.add_subplot(2,2,3))
    draw_panel3(fig.add_subplot(4,4,11))
    draw_panel4(fig.add_subplot(2,2,4))
    png = os.path.join(OUTPUT_ROOT, "cards_png", "LUNG1-052__c3_efficientnet_v4_20.png")
    save_png(fig, png)
    save_json(os.path.join(OUTPUT_ROOT, "cards_json", "LUNG1-052__c3_efficientnet_v4_20.json"), build_metadata())
    save_index_csv(os.path.join(OUTPUT_ROOT, "index_cards_efficientnet_v4_20.csv"), png)
    save_runtime_summary(os.path.join(OUTPUT_ROOT, "runtime_summary_efficientnet_v4_20.json"))
    save_done(os.path.join(OUTPUT_ROOT, "DONE.json"), png)
    return png

def save_png(fig, path):
    _require("ALLOW_PNG_WRITE", ALLOW_PNG_WRITE)
    fig.savefig(path, dpi=150, bbox_inches="tight")

def save_json(path, obj):
    _require("ALLOW_PNG_WRITE", ALLOW_PNG_WRITE)
    with open(path, "w") as f: json.dump(obj, f, ensure_ascii=False, indent=2)

def save_index_csv(path, png):
    _require("ALLOW_PNG_WRITE", ALLOW_PNG_WRITE)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case_id","output_png","card_version","candidate_position_policy"])
        w.writerow([CASE["case_id"], png, CASE["card_version"], CASE["candidate_position_policy"]])

def save_runtime_summary(path):
    _require("ALLOW_PNG_WRITE", ALLOW_PNG_WRITE)
    with open(path, "w") as f:
        json.dump({"guards": GUARDS, "ct_load_used": False, "source_image_read_used": True,
                   "png_write_used": True}, f, ensure_ascii=False, indent=2)

def save_done(path, png):
    _require("ALLOW_PNG_WRITE", ALLOW_PNG_WRITE)
    with open(path, "w") as f:
        json.dump({"step": "efficientnet_s5_card_actual_render", "status": "DONE",
                   "output_png": png, "card_version": CASE["card_version"]}, f, ensure_ascii=False, indent=2)

# ---- forbidden scan (negation-aware) ----
def scan_forbidden():
    text = "\n".join([PANEL_TEXT["panel1_note"], PANEL_TEXT["panel2_title"], PANEL_TEXT["panel3_title"],
                      PANEL_TEXT["panel4"]] + [t[2] for t in PANEL1_TILES])
    for d in ALLOWED_DISCLAIMERS: text = text.replace(d, " ")
    low = text.lower()
    return {t: low.count(t.lower()) for t in FORBIDDEN_TERMS}

# =========================================================================
def cmd_selftest():
    notes = []
    assert all(v is False for v in GUARDS.values()); notes.append("guards all False OK")
    assert CASE["candidate_position_policy"] == "v1_location_fixed"; notes.append("candidate v1 fixed OK")
    assert [t[0] for t in PANEL1_TILES] == ["candidate","normal_ref_1","normal_ref_3","normal_ref_2"]; notes.append("Panel1 slot order OK")
    assert EFF_PEAK["bbox"][1] == 112 and V1_PATCH4["bbox"][1] == 128 and 112 < 128; notes.append("peak x=112 < v1 patch4 x=128 OK")
    miss = [k for k,v in EFF_3X3.items() if v is None]
    assert miss == [(304,96)]; notes.append("exactly one MISSING [304,96] OK")
    assert sum(scan_forbidden().values()) == 0; notes.append("forbidden 0 OK")
    md = build_metadata(); assert all(f in md for f in JSON_REQUIRED_FIELDS); notes.append(f"JSON {len(JSON_REQUIRED_FIELDS)} fields OK")
    assert md["source_image_read_used"] is False and md["png_write_used"] is False; notes.append("md IO flags False OK")
    return True, notes

def cmd_dry_run():
    rows = [{"asset": t[0], "path_exists": os.path.isfile(os.path.join(PREVIEW, t[1])), "read_now": False} for t in PANEL1_TILES]
    return {"panel1_assets": rows, "output_root_exists": os.path.isdir(OUTPUT_ROOT),
            "v1_card_root_present_unmodified": os.path.isdir(V1_CARD_ROOT),
            "ct_load": False, "source_image_read": False, "card_render": False, "png_write": False}

def cmd_plan_only():
    return {"case": CASE, "panel1_tiles": [t[0] for t in PANEL1_TILES],
            "panel2": {"peak": EFF_PEAK, "v1_patch4": V1_PATCH4, "v1_patch7": V1_PATCH7},
            "panel3_3x3": {f"{k[0]},{k[1]}": v for k,v in EFF_3X3.items()},
            "panel4_text": PANEL_TEXT["panel4"], "output_root": OUTPUT_ROOT, "actual_render_now": False}

def run_prototype(confirmed):
    if not confirmed:
        print("BLOCKED: --run-prototype requires --confirm-generate", file=sys.stderr); return 2
    hard = [ALLOW_CARD_RENDER, ALLOW_SOURCE_IMAGE_READ, ALLOW_PNG_WRITE]
    if not all(hard):
        print("BLOCKED: render guards are False (no render/read/write in this step)", file=sys.stderr); return 2
    png = render_card()
    print(json.dumps({"render": "DONE", "png": png}, ensure_ascii=False)); return 0

def main(argv):
    ap = argparse.ArgumentParser(add_help=True)
    for m in ["selftest","dry-run","plan-only","static-drycheck","run-prototype","confirm-generate"]:
        ap.add_argument("--"+m, action="store_true")
    if len(argv) == 0:
        print("BLOCKED: no args. 모드: --selftest/--dry-run/--plan-only/--static-drycheck", file=sys.stderr); return 2
    a = ap.parse_args(argv)
    if getattr(a, "run_prototype"): return run_prototype(getattr(a, "confirm_generate"))
    if a.selftest:
        ok, notes = cmd_selftest(); print(json.dumps({"selftest":"PASS" if ok else "FAIL","notes":notes},ensure_ascii=False,indent=2)); return 0 if ok else 1
    if getattr(a, "dry_run"):
        print(json.dumps({"dry_run":"PASS","status":cmd_dry_run()},ensure_ascii=False,indent=2)); return 0
    if getattr(a, "plan_only"):
        print(json.dumps({"plan_only":"PASS","plan":cmd_plan_only()},ensure_ascii=False,indent=2)); return 0
    if getattr(a, "static_drycheck"):
        ok, notes = cmd_selftest()
        out = {"static_drycheck":"PASS" if ok else "FAIL","guards":GUARDS,"selftest_notes":notes,
               "dry_run":cmd_dry_run(),"forbidden_scan":scan_forbidden(),"plan":cmd_plan_only()}
        print(json.dumps(out,ensure_ascii=False,indent=2)); return 0 if ok else 1
    print("BLOCKED: unknown mode", file=sys.stderr); return 2

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
