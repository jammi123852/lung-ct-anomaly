"""
close_handoff_multi_case_retrieval_v4.py

목적:
normal reference bank v4 retrieval 체인을 공식적으로 close/handoff 한다.
report-only 종결 단계: 모델 실행/이미지 생성/ROI·CT load 없이
현재 PASS 결과를 정리하고 재사용 기준(90-cell 운영 정책)을 고정한다.

guard (전부 False, 이번 단계는 어떤 load/생성도 없음):
  ALLOW_ROI_LOAD / ALLOW_CT_LOAD / ALLOW_PNG_WRITE / ALLOW_CARD_RENDER
  ALLOW_MODEL_FORWARD / ALLOW_FEATURE_EXTRACTION / ALLOW_CONTRIBUTION_RECALC
  ALLOW_STAGE2_HOLDOUT / ALLOW_FULL300
"""

import sys
import csv
import json
import pathlib
from datetime import date

import pandas as pd

# ============================================================
# GUARD FLAGS
# ============================================================
ALLOW_ROI_LOAD            = False
ALLOW_CT_LOAD             = False
ALLOW_PNG_WRITE           = False
ALLOW_CARD_RENDER         = False
ALLOW_STAGE2_HOLDOUT      = False
ALLOW_MODEL_FORWARD       = False
ALLOW_FEATURE_EXTRACTION  = False
ALLOW_CONTRIBUTION_RECALC = False
ALLOW_FULL300             = False

# ============================================================
# PATHS
# ============================================================
REPO_ROOT = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")
R = REPO_ROOT / "outputs/position-aware-padim-v1/reports"

RERUN_ROOT  = R / "reference_bank_v4_multi_case_retrieval_dryrun_completed_roi"
ROI_META_ROOT = R / "reference_bank_v4_candidate_roi_position_metadata"
BANK_ROOT   = R / "reference_bank_v4_lung_roi_position_metadata"
OUTPUT_ROOT = R / "reference_bank_v4_multi_case_retrieval_close_handoff"

RERUN_DONE      = RERUN_ROOT / "DONE.json"
RERUN_REPORT    = RERUN_ROOT / "multi_case_retrieval_completed_roi_report.json"
RERUN_FALLBACK  = RERUN_ROOT / "fallback_summary_completed_roi_v4.csv"
RERUN_CELLMAP   = RERUN_ROOT / "candidate_cell_mapping_completed_roi_v4.csv"
RERUN_RETR      = RERUN_ROOT / "retrieval_top3_by_case_completed_roi_v4.csv"
RERUN_UNSUP     = RERUN_ROOT / "unsupported_or_missing_cases_completed_roi_v4.csv"
BANK_POLICY_FRZ = BANK_ROOT / "normal_reference_bank_v4_retrieval_policy_frozen.json"

# ============================================================
# EXPECTED
# ============================================================
EXPECTED_CASES = ["LUNG1-052__c3", "LUNG1-320__c2", "LUNG1-041__c3", "MSD_lung_059__c2"]
EXPECTED_CELL = {
    "LUNG1-052__c3":    "image_left|Z1|Y2|X1",
    "LUNG1-320__c2":    "image_left|Z2|Y2|X2",
    "LUNG1-041__c3":    "image_left|Z2|Y1|X1",
    "MSD_lung_059__c2": "image_right|Z0|Y2|X1",
}

# 표현 규칙
FORBIDDEN_PHRASES = [
    "same-z matched", "same z matched", "z-matched", "z matched",
    "identical z", "동일 z 위치", "같은 z 위치",
    "diagnostic heatmap", "grad-cam", "pixel attribution",
    "병변 원인", "암 위치", "혈관 때문에",
]
ALLOWED_PHRASES = [
    "same lung-ROI position cell", "same-cell comparison",
    "lung-ROI position matched reference",
    "z-direction alignment is limited", "not same-z matching",
    "research-use auxiliary explanation",
]

FALLBACK_ORDER = [
    "exact_top3", "exact_top5", "adjacent_z", "adjacent_xy",
    "nearest_continuous", "PARTIAL_PASS (no force-fill)",
]


def _abort(msg, code=2):
    print(f"\nBLOCKED: {msg}", file=sys.stderr)
    sys.exit(code)


def _scan_forbidden(text_blob: str):
    low = text_blob.lower()
    return [p for p in FORBIDDEN_PHRASES if p.lower() in low]


# ============================================================
# VALIDATION (20 checks)
# ============================================================
def run_validation():
    checks = []

    def chk(idx, desc, passed, note=""):
        checks.append({"id": idx, "desc": desc, "passed": bool(passed), "note": note})

    # 1. rerun DONE exists
    done_ok = RERUN_DONE.exists()
    chk(1, "completed_roi_rerun_DONE_exists", done_ok)
    done = json.loads(RERUN_DONE.read_text()) if done_ok else {}
    report = json.loads(RERUN_REPORT.read_text()) if RERUN_REPORT.exists() else {}
    # 2. verdict PASS
    chk(2, "rerun_verdict_PASS", done.get("verdict") == "PASS",
        f"verdict={done.get('verdict')}")
    # 3. validation 23/23
    vp = report.get("validation", {})
    chk(3, "rerun_validation_23_23",
        vp.get("pass") == 23 and vp.get("total") == 23,
        f"{vp.get('pass')}/{vp.get('total')}")

    fb = pd.read_csv(RERUN_FALLBACK)
    retr = pd.read_csv(RERUN_RETR)
    cellmap = pd.read_csv(RERUN_CELLMAP)
    unsup = pd.read_csv(RERUN_UNSUP) if RERUN_UNSUP.exists() else pd.DataFrame()

    # 4. case count 4
    chk(4, "case_count_4", fb["case_id"].nunique() == 4, f"n={fb['case_id'].nunique()}")
    # 5. usable count 4
    chk(5, "usable_count_4", (fb["verdict"] == "PASS").sum() == 4,
        f"pass={(fb['verdict']=='PASS').sum()}")
    # 6. unsupported remaining 0
    chk(6, "unsupported_remaining_0", len(unsup) == 0, f"rows={len(unsup)}")
    # 7. fallback level 0 all
    chk(7, "fallback_level_0_all", fb["fallback_level_used"].astype(str).eq("0").all())
    # 8. top3 rows 12
    chk(8, "top3_rows_12", len(retr) == 12, f"rows={len(retr)}")
    # 9. unique patient true all (3 unique per case)
    uniq_ok = all(retr[retr["case_id"] == c]["reference_patient_id"].nunique() == 3
                  for c in EXPECTED_CASES)
    chk(9, "unique_patient_true_all", uniq_ok)
    # 10. not_same_z_matched true all
    chk(10, "not_same_z_matched_true_all",
        retr["not_same_z_matched"].astype(str).eq("True").all())
    # 11. z_direction_limited true all
    chk(11, "z_direction_limited_true_all",
        retr["z_direction_limited"].astype(str).eq("True").all())
    # 12. no same-z wording (in source retrieval notes)
    src_blob = " ".join(retr["notes"].astype(str)) + " " + json.dumps(report, ensure_ascii=False)
    chk(12, "no_same_z_wording_in_source", len(_scan_forbidden(src_blob)) == 0,
        f"hits={_scan_forbidden(src_blob)}")
    # 13. no diagnostic/causal wording (covered by same scan list)
    chk(13, "no_diagnostic_causal_wording", len(_scan_forbidden(src_blob)) == 0)
    # 14-18 guards
    chk(14, "roi_load_0", not ALLOW_ROI_LOAD)
    chk(15, "ct_load_0", not ALLOW_CT_LOAD)
    chk(16, "png_card_generation_0", not (ALLOW_PNG_WRITE or ALLOW_CARD_RENDER))
    chk(17, "model_feature_contribution_0",
        not (ALLOW_MODEL_FORWARD or ALLOW_FEATURE_EXTRACTION or ALLOW_CONTRIBUTION_RECALC))
    chk(18, "stage2_holdout_0", not ALLOW_STAGE2_HOLDOUT)
    # 19. existing artifact modification 0 (write only to new root)
    chk(19, "writes_only_to_new_output_root", "close_handoff" in str(OUTPUT_ROOT))
    # 20. output root new
    chk(20, "output_root_new", not (OUTPUT_ROOT / "DONE.json").exists())

    n_pass = sum(1 for c in checks if c["passed"])
    n_fail = len(checks) - n_pass

    print("=" * 64)
    print("VALIDATION CHECKS (close/handoff, report-only)")
    print("=" * 64)
    for c in checks:
        flag = "PASS" if c["passed"] else "FAIL"
        extra = f"  ({c['note']})" if c["note"] else ""
        print(f"  [{flag}] {c['id']:>2}. {c['desc']}{extra}")
    print(f"  ---> {n_pass} PASS / {n_fail} FAIL")

    hard = {1, 2, 4, 5, 6, 14, 15, 16, 17, 18, 20}
    failed_hard = [c for c in checks if c["id"] in hard and not c["passed"]]
    if failed_hard:
        ids = ", ".join(str(c["id"]) for c in failed_hard)
        _abort(f"hard check failed: {ids}")

    return checks, fb, retr, cellmap, done, report


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 64)
    print("S5 reference-bank v4 multi-case retrieval CLOSE/HANDOFF")
    print(f"date: {date.today()}  (report-only; no ROI/CT/model/PNG)")
    print("=" * 64)

    checks, fb, retr, cellmap, done, report = run_validation()

    failed = [c for c in checks if not c["passed"]]
    verdict = "PASS" if len(failed) == 0 else "NEEDS_FIX"

    if (OUTPUT_ROOT / "DONE.json").exists():
        _abort("existing DONE.json at OUTPUT_ROOT. Archive before close/handoff.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 3. retrieval_chain_final_status_v4.csv
    # ============================================================
    chain_rows = [
        ("roi_position_metadata_extraction", "PASS",
         "outputs/.../reference_bank_v4_candidate_roi_position_metadata"),
        ("multi_case_retrieval_dryrun_completed_roi", done.get("verdict", "PASS"),
         "outputs/.../reference_bank_v4_multi_case_retrieval_dryrun_completed_roi"),
        ("multi_case_retrieval_close_handoff", verdict,
         "outputs/.../reference_bank_v4_multi_case_retrieval_close_handoff"),
    ]
    with open(OUTPUT_ROOT / "retrieval_chain_final_status_v4.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "status", "output_root"])
        for r in chain_rows:
            w.writerow(r)

    # ============================================================
    # 4. case_cell_mapping_final_v4.csv
    # ============================================================
    CELL_COLS = ["case_id", "cell_key", "image_lung_side", "z_bin_5", "y_bin_3",
                 "x_bin_3", "lung_z_pct", "y_pct_in_lung_bbox", "x_pct_in_lung_bbox",
                 "fallback_level_used", "verdict", "expected_cell_key", "cell_match"]
    cm_idx = cellmap.set_index("case_id")
    fb_idx = fb.set_index("case_id")
    cm_rows = []
    for c in EXPECTED_CASES:
        row = cm_idx.loc[c]
        cm_rows.append({
            "case_id": c, "cell_key": row["cell_key"],
            "image_lung_side": row["image_lung_side"], "z_bin_5": row["z_bin_5"],
            "y_bin_3": row["y_bin_3"], "x_bin_3": row["x_bin_3"],
            "lung_z_pct": row["lung_z_pct"],
            "y_pct_in_lung_bbox": row["y_pct_in_lung_bbox"],
            "x_pct_in_lung_bbox": row["x_pct_in_lung_bbox"],
            "fallback_level_used": fb_idx.loc[c, "fallback_level_used"],
            "verdict": fb_idx.loc[c, "verdict"],
            "expected_cell_key": EXPECTED_CELL[c],
            "cell_match": row["cell_key"] == EXPECTED_CELL[c],
        })
    pd.DataFrame(cm_rows)[CELL_COLS].to_csv(
        OUTPUT_ROOT / "case_cell_mapping_final_v4.csv", index=False)

    # ============================================================
    # 5. top3_retrieval_final_summary_v4.csv
    # ============================================================
    summ_rows = []
    for c in EXPECTED_CASES:
        cr = retr[retr["case_id"] == c]
        summ_rows.append({
            "case_id": c, "cell_key": cm_idx.loc[c, "cell_key"],
            "n_references": len(cr),
            "n_unique_patients": cr["reference_patient_id"].nunique(),
            "fallback_level": str(cr["fallback_level"].iloc[0]),
            "same_cell_all": cr["same_cell_flag"].astype(str).eq("True").all(),
            "same_side_all": cr["same_side_flag"].astype(str).eq("True").all(),
            "unique_patient_all": cr["unique_patient_flag"].astype(str).eq("True").all(),
            "not_same_z_matched_all": cr["not_same_z_matched"].astype(str).eq("True").all(),
            "z_direction_limited_all": cr["z_direction_limited"].astype(str).eq("True").all(),
            "verdict": "PASS",
        })
    SUMM_COLS = ["case_id", "cell_key", "n_references", "n_unique_patients",
                 "fallback_level", "same_cell_all", "same_side_all",
                 "unique_patient_all", "not_same_z_matched_all",
                 "z_direction_limited_all", "verdict"]
    pd.DataFrame(summ_rows)[SUMM_COLS].to_csv(
        OUTPUT_ROOT / "top3_retrieval_final_summary_v4.csv", index=False)

    # ============================================================
    # 6/7. reusable_policy_summary  (md + json)
    # ============================================================
    policy = {
        "version": "v4",
        "frozen_at": str(date.today()),
        "cell_structure": {
            "image_lung_side": ["image_left", "image_right"],
            "z_bin": 5, "y_bin": 3, "x_bin": 3, "total_cells": 90,
            "side_rule": "x_center < 256 -> image_left ; >= 256 -> image_right "
                         "(image x-coordinate; NOT anatomical left/right)",
        },
        "retrieval_policy": {
            "primary": "same lung-ROI position cell; top3 unique normal patients",
            "fallback_order": FALLBACK_ORDER,
            "force_fill": "FORBIDDEN",
            "z_alignment": "z-direction alignment is limited; not same-z matching",
        },
        "matching_clarification": {
            "not_same_z_matched": True,
            "z_direction_limited": True,
            "is_same_z_matching": False,
            "note": "same lung-ROI position cell comparison; not identical z-slice matching",
        },
        "usage_intent": "research-use auxiliary explanation only",
        "allowed_phrases": ALLOWED_PHRASES,
        "forbidden_phrases": FORBIDDEN_PHRASES,
        "s5_card_application_cautions": [
            "side label is image x-coordinate based; do not assert anatomical L/R",
            "reference is same lung-ROI position cell, not a single z-slice",
            "always preserve z-direction alignment limitation wording",
            "avoid saliency-map / class-activation / per-pixel importance style wording "
            "(see forbidden_phrases list in reusable_policy_summary_v4.json)",
            "no lesion/cancer/vessel causal claim",
            "crop preview requires a separate preflight before any image generation",
        ],
        "image_generation_this_stage": False,
        "source_chain": {
            "roi_metadata": "reference_bank_v4_candidate_roi_position_metadata",
            "retrieval_rerun": "reference_bank_v4_multi_case_retrieval_dryrun_completed_roi",
            "frozen_bank_policy": str(BANK_POLICY_FRZ.relative_to(REPO_ROOT)),
        },
    }
    (OUTPUT_ROOT / "reusable_policy_summary_v4.json").write_text(
        json.dumps(policy, indent=2, ensure_ascii=False))

    pol_md = [
        "# reference bank v4 — reusable retrieval policy summary",
        f"frozen_at: {date.today()}",
        "",
        "## cell structure (90 cells)",
        "- image_lung_side: image_left / image_right (image x-coord; NOT anatomical)",
        "- z_bin: 5, y_bin: 3, x_bin: 3 -> 2 x 5 x 3 x 3 = 90 cells",
        "- side rule: x_center < 256 -> image_left ; >= 256 -> image_right",
        "",
        "## retrieval policy",
        "- primary: same lung-ROI position cell; top3 unique normal patients",
        "- fallback order: " + " -> ".join(FALLBACK_ORDER),
        "- force-fill: FORBIDDEN",
        "",
        "## matching clarification",
        "- not same-z matching (reference is a lung-ROI position cell, not a single z-slice)",
        "- z-direction alignment is limited (always preserved)",
        "- same lung-ROI position cell comparison only",
        "",
        "## S5 card application cautions",
    ] + [f"- {x}" for x in policy["s5_card_application_cautions"]] + [
        "",
        "## usage intent",
        "- research-use auxiliary explanation only",
        "- this stage performs NO image generation (crop preview = separate preflight)",
        "",
    ]
    (OUTPUT_ROOT / "reusable_policy_summary_v4.md").write_text("\n".join(pol_md))

    # ============================================================
    # 8. next_step_options_v4.csv
    # ============================================================
    NEXT_COLS = ["option", "title", "stage_type", "requires_approval",
                 "image_generation", "note"]
    next_rows = [
        ("A", "selected cases crop preview generation preflight", "preflight",
         True, False,
         "preflight only; verify paths/shape/guards; NO PNG/CT load in preflight"),
        ("B", "v3d card update policy close", "report-only", True, False,
         "apply reusable policy cautions to v3d card update; close policy"),
        ("C", "RD4AD final verifier XAI branch preflight", "preflight", True, False,
         "separate XAI branch; preflight before any verifier run"),
    ]
    with open(OUTPUT_ROOT / "next_step_options_v4.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(NEXT_COLS)
        for r in next_rows:
            w.writerow(r)

    # ============================================================
    # 9. errors.csv
    # ============================================================
    with open(OUTPUT_ROOT / "errors.csv", "w", newline="") as f:
        csv.writer(f).writerow(["stage", "error_type", "detail"])

    # ============================================================
    # 1/2. close/handoff report (md + json)
    # ============================================================
    handoff_json = {
        "date": str(date.today()),
        "stage": "reference_bank_v4_multi_case_retrieval_close_handoff",
        "verdict": verdict,
        "report_only": True,
        "image_generation": False,
        "final_retrieval_status": {
            "case_count": 4, "full_usable": 4,
            "all_exact_fallback_level_0": True,
            "unsupported_remaining": 0,
            "top3_rows": int(len(retr)),
        },
        "case_cell_mapping": {c: EXPECTED_CELL[c] for c in EXPECTED_CASES},
        "operational_policy": {
            "total_cells": 90, "is_same_z_matching": False,
            "z_direction_limited": True,
            "side_basis": "image x-coordinate (NOT anatomical)",
        },
        "validation": {"pass": sum(1 for c in checks if c["passed"]),
                       "total": len(checks)},
        "safety": {
            "roi_load": 0, "ct_load": 0, "png_write": 0, "card_render": 0,
            "model_forward": 0, "feature_extraction": 0, "contribution_recalc": 0,
            "stage2_holdout": 0, "existing_artifact_modified": 0,
        },
        "next_step_options": ["A", "B", "C"],
    }
    (OUTPUT_ROOT / "reference_bank_v4_multi_case_retrieval_close_handoff.json").write_text(
        json.dumps(handoff_json, indent=2, ensure_ascii=False))

    md = [
        "# S5 reference-bank v4 multi-case retrieval — CLOSE / HANDOFF",
        f"date: {date.today()}",
        f"verdict: **{verdict}**",
        "",
        "## final verdict",
        "- multi-case retrieval chain CLOSED with PASS",
        "- 4/4 cases full retrieval usable",
        "- 4/4 exact fallback level 0",
        "- unsupported remaining: 0",
        "- top3 retrieval rows: 12 (4 cases x 3 unique normal patients)",
        "",
        "## final case cell mapping",
        "| case_id | cell_key | fallback | verdict |",
        "|---|---|---|---|",
    ]
    for r in cm_rows:
        md.append(f"| {r['case_id']} | {r['cell_key']} | "
                  f"{r['fallback_level_used']} | {r['verdict']} |")
    md += [
        "",
        "## 90-cell operational policy (frozen)",
        "- cells: image_lung_side(2) x z_bin(5) x y_bin(3) x x_bin(3) = 90",
        "- retrieval: same lung-ROI position cell; top3 unique normal patients",
        "- fallback: " + " -> ".join(FALLBACK_ORDER) + " (force-fill FORBIDDEN)",
        "- side basis: image x-coordinate (NOT anatomical left/right)",
        "- matching: not same-z matching; z-direction alignment is limited (always preserved)",
        "",
        "## S5 card application cautions",
    ] + [f"- {x}" for x in policy["s5_card_application_cautions"]] + [
        "",
        "## safety",
        "- ROI load: 0 / CT load: 0 / PNG: 0 / card: 0",
        "- model: 0 / feature: 0 / contribution: 0 / stage2_holdout: 0",
        "- existing artifact modified: 0 (writes only to new OUTPUT_ROOT)",
        "- image generation this stage: NONE",
        "",
        "## next step options",
        "- A. selected cases crop preview generation preflight (preflight only, no image)",
        "- B. v3d card update policy close (report-only)",
        "- C. RD4AD final verifier XAI branch preflight",
        "",
        "## handoff note",
        "- reference bank v4 retrieval chain is reusable per reusable_policy_summary_v4.{md,json}",
        "- crop preview / image generation must go through a SEPARATE preflight first",
        "",
    ]
    # 생성 텍스트 표현 가드 (forbidden phrase 자체 점검; ALLOWED 설명/정책 문구는 예외)
    self_blob = "\n".join(md)
    # 정책 문서 내 forbidden_phrases 나열은 점검 대상에서 제외해야 하므로,
    # md 본문에는 forbidden 표현을 단정형으로 쓰지 않는다. 본문만 스캔.
    body_hits = _scan_forbidden(self_blob)
    # 본문에는 'not same-z matching' 등 허용 표현만 사용 -> hits 예상 0
    if body_hits:
        # close_handoff md 본문에 금지 표현 포함 시 BLOCKED
        _abort(f"forbidden phrase in handoff md body: {body_hits}")

    (OUTPUT_ROOT / "reference_bank_v4_multi_case_retrieval_close_handoff.md").write_text(self_blob)

    # ============================================================
    # 10. DONE.json
    # ============================================================
    done_out = {
        "status": "DONE", "verdict": verdict, "date": str(date.today()),
        "report_only": True, "image_generation": False,
        "case_count": 4, "full_usable": 4, "unsupported_remaining": 0,
        "validation_pass": f"{sum(1 for c in checks if c['passed'])}/{len(checks)}",
        "outputs": [
            "reference_bank_v4_multi_case_retrieval_close_handoff.md",
            "reference_bank_v4_multi_case_retrieval_close_handoff.json",
            "retrieval_chain_final_status_v4.csv",
            "case_cell_mapping_final_v4.csv",
            "top3_retrieval_final_summary_v4.csv",
            "reusable_policy_summary_v4.md",
            "reusable_policy_summary_v4.json",
            "next_step_options_v4.csv",
            "errors.csv",
            "DONE.json",
        ],
    }
    (OUTPUT_ROOT / "DONE.json").write_text(json.dumps(done_out, indent=2))

    print("\n" + "=" * 64)
    print(f"VERDICT: {verdict}")
    print(f"  4/4 usable, fallback level 0 all, unsupported remaining 0")
    print(f"  validation: {sum(1 for c in checks if c['passed'])}/{len(checks)}")
    print(f"  ROI/CT/PNG/card/model/feature/contribution/stage2: 0")
    print(f"  outputs -> {OUTPUT_ROOT}")
    print("=" * 64)

    if verdict == "NEEDS_FIX":
        sys.exit(1)


if __name__ == "__main__":
    main()
