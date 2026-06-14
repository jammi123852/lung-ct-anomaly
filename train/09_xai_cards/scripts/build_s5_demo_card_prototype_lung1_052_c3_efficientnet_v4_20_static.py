#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EfficientNet PaDiM v4.20 전용 LUNG1-052__c3 S5 card — STATIC (script + static-drycheck only).

이 스크립트는 기존 v1 카드 스크립트(build_s5_demo_card_prototype_lung1_052_c3_v4_ref_update*.py)를
직접 수정하지 않는다. copy/new 방식의 독립 스크립트다.

이번 버전은 실제 render/CT load/PNG write를 하지 않는다(모든 guard False).
actual render는 별도 승인 단계에서만 허용된다.

결정: candidate 위치 = v1 위치 유지(Panel 1 same-cell reference 재사용).
      EfficientNet peak shift는 Panel 2/3 에서만 표시.

모드:
  (no args)                              -> BLOCKED exit 2
  --selftest                             -> 내부 로직 테스트(IO 없음)
  --dry-run                              -> render/CT/PNG 없이 경로/스키마/텍스트 점검
  --plan-only                            -> 계획된 카드 구조 출력
  --static-drycheck                      -> 전체 정적검사
  --run-prototype                        -> guard False면 BLOCKED exit 2
  --run-prototype --confirm-generate     -> guard False면 BLOCKED exit 2
"""
import os, sys, json, argparse

# ===== GUARDS (전부 False) =====
ALLOW_CARD_RENDER                  = False
ALLOW_CT_LOAD                      = False
ALLOW_PNG_WRITE                    = False
ALLOW_SOURCE_IMAGE_READ            = False
ALLOW_MODEL_FORWARD                = False
ALLOW_FEATURE_EXTRACTION           = False
ALLOW_SCORE_RECOMPUTE              = False
ALLOW_STAGE2_HOLDOUT_ACCESS        = False
ALLOW_EXISTING_ARTIFACT_MODIFICATION = False
ALLOW_MAIN_RENAME                  = False

GUARDS = {
    "ALLOW_CARD_RENDER": ALLOW_CARD_RENDER,
    "ALLOW_CT_LOAD": ALLOW_CT_LOAD,
    "ALLOW_PNG_WRITE": ALLOW_PNG_WRITE,
    "ALLOW_SOURCE_IMAGE_READ": ALLOW_SOURCE_IMAGE_READ,
    "ALLOW_MODEL_FORWARD": ALLOW_MODEL_FORWARD,
    "ALLOW_FEATURE_EXTRACTION": ALLOW_FEATURE_EXTRACTION,
    "ALLOW_SCORE_RECOMPUTE": ALLOW_SCORE_RECOMPUTE,
    "ALLOW_STAGE2_HOLDOUT_ACCESS": ALLOW_STAGE2_HOLDOUT_ACCESS,
    "ALLOW_EXISTING_ARTIFACT_MODIFICATION": ALLOW_EXISTING_ARTIFACT_MODIFICATION,
    "ALLOW_MAIN_RENAME": ALLOW_MAIN_RENAME,
}

ROOT = "/home/jinhy/project/lung-ct-anomaly"
EFF  = os.path.join(ROOT, "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1")

CASE = {
    "case_id": "LUNG1-052__c3", "local_z": 51, "report_slice": 106,
    "candidate_position": "v1_location_fixed",
    "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
    "mask": "refined_roi_v4_20_modeB", "version": "efficientnet_v4_20",
}

# EfficientNet saved score (LUNG1-052 local_z=51) — preflight에서 확인된 값(하드코딩 plan; static에서 재계산 안 함)
EFF_PATCH = {
    "v1_patch4_same_loc": {"bbox": [288,128,320,160], "eff_score": 17.9909, "v1_score": 38.872562},
    "v1_patch7_same_loc": {"bbox": [320,128,352,160], "eff_score": 18.2798, "v1_score": 36.61247},
    "efficientnet_peak":  {"bbox": [288,112,320,144], "eff_score": 21.6854, "v1_score": None, "bin": "lower_peripheral"},
}
# Panel 3 3x3 (peak 주변), [304,96]=MISSING(v4_20 흉벽제거)
EFF_3X3 = {
    "[288,112]": 21.6854, "[304,112]": 19.2811, "[288,96]": 18.9322, "[288,128]": 17.9909,
    "[304,128]": 17.7807, "[272,112]": 15.5562, "[272,128]": 14.5310, "[272,96]": 14.3818,
    "[304,96]": None,  # MISSING
}

# 재사용(reuse) 소스 — 기존 v1 카드 자산 (read 계획만; 이번 단계 미read)
REUSE_SOURCES = {
    "v1_card_json": os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/explanation_cards/s5_demo_card_prototype_lung1_052_c3_v4_ref_update/cards_json/LUNG1-052__c3_s5_demo_prototype_v4_ref_update.json"),
    "ref_preview_root": os.path.join(ROOT, "outputs/position-aware-padim-v1/visualizations/reference_bank_v4_selected_cases_crop_preview"),
    "candidate_png": "LUNG1-052__c3_candidate.png",
    "normal_ref_1": "LUNG1-052__c3_normal_ref_1.png",
    "normal_ref_3": "LUNG1-052__c3_normal_ref_3.png",
    "normal_ref_2": "LUNG1-052__c3_normal_ref_2.png",
}
ORIGINAL_V1_SCRIPT = os.path.join(ROOT, "scripts/build_s5_demo_card_prototype_lung1_052_c3_v4_ref_update.py")  # 수정 금지

# 신규 output root (이번 단계 미생성)
ACTUAL_OUTPUT_ROOT = os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/efficientnet_s5_card_lung1_052_c3_v4_20")

SLOT_ORDER = ["candidate", "normal_ref_1", "normal_ref_3", "normal_ref_2"]

# ===== 카드 텍스트 =====
ALLOWED_DISCLAIMERS = [
    "Same-cell normal references are matched by lung-ROI position cell, not by identical z-slice.",
    "Scores are branch-specific and should not be compared as absolute values.",
    "This is a research-use auxiliary explanation, not a diagnostic conclusion.",
    "Panel 1 keeps the previous candidate-centered view to preserve the same-cell reference comparison.",
]
PANEL_TEXT = {
    "panel1_subtitle": "Candidate location with same-cell normal references (v4 reference bank).",
    "panel1_note": "Panel 1 keeps the previous candidate-centered view to preserve the same-cell reference comparison.",
    "panel2": "EfficientNet-B0 PaDiM response overview. The response peak is near the candidate region.",
    "panel3": "EfficientNet-specific patch response map (3x3) around the response peak.",
    "panel4": ("Using EfficientNet-B0 PaDiM features, the response peak is slightly shifted left from the "
               "previous v1-centered patch, while a downward response pattern is partially preserved. "
               "Same-cell normal references are matched by lung-ROI position cell, not by identical z-slice. "
               "Scores are branch-specific and should not be compared as absolute values. "
               "This is a research-use auxiliary explanation, not a diagnostic conclusion."),
    "panel1_labels": ["Candidate location", "Same-cell normal ref 1", "Same-cell normal ref 3", "Additional same-cell ref 2"],
}

FORBIDDEN_TERMS = [
    "diagnostic heatmap", "grad-cam", "gradcam", "pixel attribution", "cancer probability",
    "lesion cause", "same-z matched", "identical z", "혈관 때문에 병변", "암 위치 확정", "진단 확정",
]


def all_card_text():
    parts = [PANEL_TEXT["panel1_subtitle"], PANEL_TEXT["panel1_note"], PANEL_TEXT["panel2"],
             PANEL_TEXT["panel3"], PANEL_TEXT["panel4"]] + PANEL_TEXT["panel1_labels"]
    return "\n".join(parts)


def scan_forbidden():
    """negation-aware: 허용 disclaimer 문장 제거 후 금지어 스캔."""
    text = all_card_text()
    for d in ALLOWED_DISCLAIMERS:
        text = text.replace(d, " ")
    low = text.lower()
    counts = {}
    for t in FORBIDDEN_TERMS:
        counts[t] = low.count(t.lower())
    return counts


def cmd_selftest():
    ok = True; notes = []
    assert all(v is False for v in GUARDS.values()), "guard not all False"
    notes.append("guards all False OK")
    assert CASE["candidate_position"] == "v1_location_fixed"
    notes.append("candidate 위치 v1 고정 OK")
    assert SLOT_ORDER == ["candidate", "normal_ref_1", "normal_ref_3", "normal_ref_2"]
    notes.append("Panel1 slot order OK")
    # EfficientNet peak가 v1 patch4보다 좌측(x 작음)
    assert EFF_PATCH["efficientnet_peak"]["bbox"][1] < EFF_PATCH["v1_patch4_same_loc"]["bbox"][1]
    notes.append("peak x=-16px 좌측 OK")
    # forbidden scan == 0 (disclaimer 제외 후)
    c = scan_forbidden()
    assert sum(c.values()) == 0, f"forbidden found: {c}"
    notes.append("forbidden wording 0 (disclaimer 제외 후) OK")
    # 3x3 missing 표기
    assert EFF_3X3["[304,96]"] is None
    notes.append("3x3 [304,96] MISSING 표기 OK")
    return ok, notes


def cmd_dry_run():
    rows = []
    # 재사용 소스 존재만 점검(이미지 read 안 함)
    for k in ["candidate_png", "normal_ref_1", "normal_ref_3", "normal_ref_2"]:
        p = os.path.join(REUSE_SOURCES["ref_preview_root"], REUSE_SOURCES[k])
        rows.append({"asset": k, "path_exists": os.path.isfile(p), "read_now": False})
    v1json_exists = os.path.isfile(REUSE_SOURCES["v1_card_json"])
    eff_score_dir = os.path.isdir(os.path.join(EFF, "outputs/scores/lesion_stage1_dev_by_patient"))
    orig_unmodified = os.path.isfile(ORIGINAL_V1_SCRIPT)  # 존재 확인(수정 안 함)
    actual_root_exists = os.path.isdir(ACTUAL_OUTPUT_ROOT)
    return {
        "panel1_assets": rows,
        "v1_card_json_exists_reference_only": v1json_exists,
        "efficientnet_score_dir_exists": eff_score_dir,
        "original_v1_script_present_unmodified": orig_unmodified,
        "actual_output_root_exists_should_be_false": actual_root_exists,
        "ct_load": False, "png_write": False, "card_render": False, "source_image_read": False,
    }


def cmd_plan_only():
    return {
        "case": CASE,
        "panels": {
            "panel_1": {"reuse": True, "candidate_position": "v1_location_fixed",
                        "slots": SLOT_ORDER, "labels": PANEL_TEXT["panel1_labels"],
                        "note": PANEL_TEXT["panel1_note"]},
            "panel_2": {"rebuild": True, "highlight": "efficientnet_peak [288,112]",
                        "removed": "v1 patch4/patch7 absolute scores", "text": PANEL_TEXT["panel2"]},
            "panel_3": {"rebuild": True, "map": "EfficientNet 3x3 saved padim_score",
                        "missing": "[304,96]", "no_interpolation": True, "cells": EFF_3X3},
            "panel_4": {"rewrite": True, "text": PANEL_TEXT["panel4"]},
        },
        "efficientnet_patch": EFF_PATCH,
        "actual_output_root": ACTUAL_OUTPUT_ROOT,
        "actual_render_now": False,
    }


def run_prototype(confirmed):
    if not confirmed:
        print("BLOCKED: --run-prototype requires --confirm-generate", file=sys.stderr); return 2
    hard = [ALLOW_CARD_RENDER, ALLOW_PNG_WRITE, ALLOW_SOURCE_IMAGE_READ]
    if not all(hard):
        print("BLOCKED: render guards are False (no render/PNG write in this step)", file=sys.stderr); return 2
    print("ERROR: actual render는 별도 승인 단계 전용. 본 static 버전에서 비활성.", file=sys.stderr); return 2


def main(argv):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--static-drycheck", action="store_true")
    ap.add_argument("--run-prototype", action="store_true")
    ap.add_argument("--confirm-generate", action="store_true")
    if len(argv) == 0:
        print("BLOCKED: no args. 모드: --selftest/--dry-run/--plan-only/--static-drycheck", file=sys.stderr)
        return 2
    args = ap.parse_args(argv)

    if args.run_prototype:
        return run_prototype(args.confirm_generate)
    if args.selftest:
        ok, notes = cmd_selftest()
        print(json.dumps({"selftest": "PASS" if ok else "FAIL", "notes": notes}, ensure_ascii=False, indent=2))
        return 0 if ok else 1
    if args.dry_run:
        print(json.dumps({"dry_run": "PASS", "status": cmd_dry_run()}, ensure_ascii=False, indent=2))
        return 0
    if args.plan_only:
        print(json.dumps({"plan_only": "PASS", "plan": cmd_plan_only()}, ensure_ascii=False, indent=2))
        return 0
    if args.static_drycheck:
        ok, notes = cmd_selftest()
        out = {"static_drycheck": "PASS" if ok else "FAIL", "guards": GUARDS,
               "selftest_notes": notes, "dry_run": cmd_dry_run(),
               "forbidden_scan": scan_forbidden(), "plan": cmd_plan_only()}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if ok else 1
    print("BLOCKED: unknown mode", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
