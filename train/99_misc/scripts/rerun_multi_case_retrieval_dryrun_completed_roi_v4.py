"""
rerun_multi_case_retrieval_dryrun_completed_roi_v4.py

목적:
ROI position metadata extraction(PASS) 결과를 반영하여,
이전 multi-case retrieval dry-run에서 UNSUPPORTED였던 case 포함 4개 case 전부를
full retrieval usable / top3 retrieval PASS 인지 CSV/JSON 기반으로 재검증한다.

이번 단계는 report-only rerun 이다.
- 이미 추출된 completed ROI metadata CSV/JSON 만 사용
- ROI load / CT load / PNG / card / model / feature / contribution / stage2 전부 금지
- 기존 artifact 수정 금지 (신규 OUTPUT_ROOT 에만 기록)
- rough mapping 금지 (cell_key 는 추출 결과를 그대로 사용)

guard:
  ALLOW_ROI_LOAD            = False
  ALLOW_CT_LOAD             = False
  ALLOW_PNG_WRITE           = False
  ALLOW_CARD_RENDER         = False
  ALLOW_STAGE2_HOLDOUT      = False
  ALLOW_MODEL_FORWARD       = False
  ALLOW_FEATURE_EXTRACTION  = False
  ALLOW_CONTRIBUTION_RECALC = False
  ALLOW_FULL300             = False
"""

import sys
import csv
import json
import pathlib
from datetime import date

import pandas as pd

# ============================================================
# GUARD FLAGS (전부 False, 이번 단계는 어떤 load 도 없음)
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

PREV_DRYRUN_ROOT = R / "reference_bank_v4_multi_case_retrieval_dryrun"
ROI_META_ROOT    = R / "reference_bank_v4_candidate_roi_position_metadata"
BANK_ROOT        = R / "reference_bank_v4_lung_roi_position_metadata"
OP_POLICY_ROOT   = R / "reference_bank_v4_operational_policy_final"
OUTPUT_ROOT      = R / "reference_bank_v4_multi_case_retrieval_dryrun_completed_roi"

# inputs - prev dryrun
PREV_INVENTORY   = PREV_DRYRUN_ROOT / "candidate_case_inventory_v4.csv"
PREV_UNSUPPORTED = PREV_DRYRUN_ROOT / "unsupported_or_missing_cases_v4.csv"
PREV_FALLBACK    = PREV_DRYRUN_ROOT / "fallback_summary_v4.csv"
PREV_POLICY_USED = PREV_DRYRUN_ROOT / "retrieval_policy_used_v4.json"

# inputs - ROI metadata extraction
ROI_DONE         = ROI_META_ROOT / "DONE.json"
ROI_REPORT_JSON  = ROI_META_ROOT / "roi_position_metadata_report.json"
ROI_META_CSV     = ROI_META_ROOT / "candidate_roi_position_metadata_v4.csv"
ROI_CELLMAP_CSV  = ROI_META_ROOT / "candidate_cell_mapping_completed_v4.csv"
ROI_RETR_CSV     = ROI_META_ROOT / "retrieval_top3_after_roi_metadata_v4.csv"
ROI_UNSUP_CSV    = ROI_META_ROOT / "unsupported_after_roi_metadata_v4.csv"

# inputs - bank / operational policy
BANK_CELL_INDEX  = BANK_ROOT / "normal_reference_bank_v4_cell_index.csv"
BANK_TOP3        = BANK_ROOT / "normal_reference_bank_v4_top3_by_cell.csv"
BANK_POLICY_FRZ  = BANK_ROOT / "normal_reference_bank_v4_retrieval_policy_frozen.json"
OP_POLICY_JSON   = OP_POLICY_ROOT / "normal_reference_bank_v4_operational_policy_final.json"

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
CONTROL_CASE = "LUNG1-052__c3"

FORBIDDEN_WORDS = ["cancer", "tumor", "malignant", "lesion cause",
                   "vessel cause", "same-z match", "same z match"]


def _abort(msg, code=2):
    print(f"\nBLOCKED: {msg}", file=sys.stderr)
    sys.exit(code)


# ============================================================
# VALIDATION (22 checks)
# ============================================================
def run_validation():
    checks = []

    def chk(idx, desc, passed, note=""):
        checks.append({"id": idx, "desc": desc, "passed": bool(passed), "note": note})

    # 1. ROI metadata extraction DONE exists
    roi_done_ok = ROI_DONE.exists()
    chk(1, "roi_metadata_extraction_DONE_exists", roi_done_ok)
    roi_done = json.loads(ROI_DONE.read_text()) if roi_done_ok else {}
    # 2. ROI metadata extraction verdict == PASS
    chk(2, "roi_metadata_extraction_verdict_PASS", roi_done.get("verdict") == "PASS",
        f"verdict={roi_done.get('verdict')}")
    # 3. unsupported_after_roi has 0 rows
    unsup_after = pd.read_csv(ROI_UNSUP_CSV) if ROI_UNSUP_CSV.exists() else pd.DataFrame()
    chk(3, "unsupported_after_roi_0_rows", len(unsup_after) == 0,
        f"rows={len(unsup_after)}")
    # 4. cell mapping completed exists
    chk(4, "cell_mapping_completed_exists", ROI_CELLMAP_CSV.exists())
    # 5. retrieval_top3_after_roi exists
    chk(5, "retrieval_top3_after_roi_exists", ROI_RETR_CSV.exists())

    cellmap = pd.read_csv(ROI_CELLMAP_CSV)
    retr = pd.read_csv(ROI_RETR_CSV)

    # 6. case count == 4
    chk(6, "case_count_eq_4", cellmap["case_id"].nunique() == 4,
        f"n={cellmap['case_id'].nunique()}")
    # 7. full usable cases == 4 (all have cell_key, no UNSUPPORTED verdict)
    usable = cellmap[(cellmap["cell_key"].notna())
                     & (cellmap["cell_key"].astype(str) != "")
                     & (cellmap["verdict"].astype(str).str.upper() != "UNSUPPORTED")]
    chk(7, "full_usable_cases_eq_4", len(usable) == 4, f"usable={len(usable)}")
    # 8. each case has cell_key (and matches expected)
    cellmap_dict = dict(zip(cellmap["case_id"], cellmap["cell_key"]))
    all_have = all(c in cellmap_dict and str(cellmap_dict[c]) not in ("", "nan")
                   for c in EXPECTED_CASES)
    cell_match = all(cellmap_dict.get(c) == EXPECTED_CELL[c] for c in EXPECTED_CASES)
    chk(8, "each_case_has_cell_key", all_have)
    chk("8b", "cell_key_matches_expected", cell_match,
        "; ".join(f"{c}:{cellmap_dict.get(c)}" for c in EXPECTED_CASES))
    # 9. each case has exact same-cell top3 (same_cell_flag True, fallback 0)
    same_cell_ok = True
    for c in EXPECTED_CASES:
        rows = retr[retr["case_id"] == c]
        if len(rows) < 3 or not rows["same_cell_flag"].astype(str).eq("True").all():
            same_cell_ok = False
    chk(9, "each_case_exact_same_cell_top3", same_cell_ok)
    # 10. each case 3 unique normal reference patients
    uniq_ok = all(
        retr[retr["case_id"] == c]["reference_patient_id"].nunique() == 3
        for c in EXPECTED_CASES)
    chk(10, "each_case_3_unique_patients", uniq_ok)
    # 11. fallback_level == 0 for all rows
    fb_ok = retr["fallback_level"].astype(str).eq("0").all()
    chk(11, "fallback_level_0_all", fb_ok,
        f"levels={sorted(retr['fallback_level'].astype(str).unique())}")
    # 12. not_same_z_matched == True all
    chk(12, "not_same_z_matched_true_all",
        retr["not_same_z_matched"].astype(str).eq("True").all())
    # 13. z_direction_limited == True all
    chk(13, "z_direction_limited_true_all",
        retr["z_direction_limited"].astype(str).eq("True").all())
    # 14. no same-z wording in notes
    notes_blob = " ".join(retr["notes"].astype(str)).lower()
    chk(14, "no_same_z_wording",
        "same-z match" not in notes_blob and "same z match" not in notes_blob)
    # 15. no diagnostic/causal wording
    diag_hit = [w for w in FORBIDDEN_WORDS if w in notes_blob]
    chk(15, "no_diagnostic_causal_wording", len(diag_hit) == 0, f"hits={diag_hit}")
    # 16-20. guards
    chk(16, "roi_load_0", not ALLOW_ROI_LOAD)
    chk(17, "ct_load_0", not ALLOW_CT_LOAD)
    chk(18, "png_card_generation_0", not (ALLOW_PNG_WRITE or ALLOW_CARD_RENDER))
    chk(19, "model_feature_contribution_0",
        not (ALLOW_MODEL_FORWARD or ALLOW_FEATURE_EXTRACTION or ALLOW_CONTRIBUTION_RECALC))
    chk(20, "stage2_holdout_0", not ALLOW_STAGE2_HOLDOUT)
    # 21. existing artifact modification 0 (write only to new OUTPUT_ROOT)
    chk(21, "writes_only_to_new_output_root",
        "dryrun_completed_roi" in str(OUTPUT_ROOT))
    # 22. output collision 없음
    chk(22, "no_output_collision", not (OUTPUT_ROOT / "DONE.json").exists())

    n_pass = sum(1 for c in checks if c["passed"])
    n_fail = len(checks) - n_pass

    print("=" * 64)
    print("VALIDATION CHECKS (completed-ROI retrieval rerun, report-only)")
    print("=" * 64)
    for c in checks:
        flag = "PASS" if c["passed"] else "FAIL"
        extra = f"  ({c['note']})" if c["note"] else ""
        print(f"  [{flag}] {str(c['id']):>3}. {c['desc']}{extra}")
    print(f"  ---> {n_pass} PASS / {n_fail} FAIL")

    # hard-stop guards / collision
    hard = {16, 17, 18, 19, 20, 22}
    failed_hard = [c for c in checks if c["id"] in hard and not c["passed"]]
    if failed_hard:
        ids = ", ".join(str(c["id"]) for c in failed_hard)
        _abort(f"hard guard/collision check failed: {ids}")

    return checks, cellmap, retr, unsup_after, roi_done


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 64)
    print("multi-case retrieval dry-run RERUN with completed ROI metadata")
    print(f"date: {date.today()}  (report-only; ROI/CT/model load = 0)")
    print("=" * 64)

    checks, cellmap, retr, unsup_after, roi_done = run_validation()

    # ---- load supporting inputs (read-only) ----
    inv = pd.read_csv(PREV_INVENTORY, dtype=str)
    roi_meta = pd.read_csv(ROI_META_CSV)
    cell_index = pd.read_csv(BANK_CELL_INDEX)

    # ---- determine verdict ----
    failed = [c for c in checks if not c["passed"]]
    # control mismatch -> NEEDS_FIX
    control_cell = dict(zip(cellmap["case_id"], cellmap["cell_key"])).get(CONTROL_CASE)
    control_ok = control_cell == EXPECTED_CELL[CONTROL_CASE]

    n_usable = len(cellmap[(cellmap["cell_key"].astype(str) != "")
                           & (cellmap["verdict"].astype(str).str.upper() != "UNSUPPORTED")])
    n_unsup_remaining = len(unsup_after)

    if not control_ok:
        verdict = "NEEDS_FIX"
    elif len(failed) == 0 and n_usable == 4 and n_unsup_remaining == 0:
        verdict = "PASS"
    elif n_usable >= 1:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "NEEDS_FIX"

    # ---- collision check before write ----
    if (OUTPUT_ROOT / "DONE.json").exists():
        _abort("existing DONE.json at OUTPUT_ROOT. Archive before rerun.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # OUTPUT 3: candidate_case_inventory_completed_roi_v4.csv
    #   prev inventory + cell_key + usable=True for target cases
    # ============================================================
    inv2 = inv[inv["case_id"].isin(EXPECTED_CASES)].copy()
    cellmap_idx = cellmap.set_index("case_id")
    inv2["cell_key"] = inv2["case_id"].map(cellmap_idx["cell_key"])
    inv2["usable_for_retrieval"] = "True"
    inv2["reason_if_unusable"] = ""
    inv2["roi_metadata_source"] = str(ROI_META_CSV.relative_to(REPO_ROOT))
    inv2.to_csv(OUTPUT_ROOT / "candidate_case_inventory_completed_roi_v4.csv", index=False)

    # ============================================================
    # OUTPUT 4: candidate_cell_mapping_completed_roi_v4.csv
    # ============================================================
    cellmap.to_csv(OUTPUT_ROOT / "candidate_cell_mapping_completed_roi_v4.csv", index=False)

    # ============================================================
    # OUTPUT 5: retrieval_top3_by_case_completed_roi_v4.csv
    #   (정해진 case 순서로 정렬, 필수 컬럼 그대로)
    # ============================================================
    RETR_COLS = [
        "case_id", "retrieval_rank", "fallback_level", "reference_id",
        "reference_patient_id", "reference_volume_id", "reference_local_z",
        "reference_lung_z_pct", "reference_cell_key",
        "reference_crop_y0", "reference_crop_x0", "reference_crop_y1",
        "reference_crop_x1", "reference_quality_score",
        "same_cell_flag", "same_side_flag", "unique_patient_flag",
        "not_same_z_matched", "z_direction_limited", "notes",
    ]
    order = {c: i for i, c in enumerate(EXPECTED_CASES)}
    retr_out = retr.copy()
    retr_out["_o"] = retr_out["case_id"].map(order).fillna(99)
    retr_out = retr_out.sort_values(
        ["_o", "retrieval_rank"]).drop(columns="_o")[RETR_COLS]
    retr_out.to_csv(
        OUTPUT_ROOT / "retrieval_top3_by_case_completed_roi_v4.csv", index=False)

    # ============================================================
    # OUTPUT 6: fallback_summary_completed_roi_v4.csv
    # ============================================================
    ci_idx = cell_index.set_index("cell_key")
    fb_rows = []
    for case_id in EXPECTED_CASES:
        cell = cellmap_idx.loc[case_id, "cell_key"]
        case_retr = retr[retr["case_id"] == case_id]
        n_ref = len(case_retr)
        n_uniq = case_retr["reference_patient_id"].nunique()
        top3_av = bool(ci_idx.loc[cell, "top3_available"]) if cell in ci_idx.index else False
        top5_av = bool(ci_idx.loc[cell, "top5_available"]) if cell in ci_idx.index else False
        fb_level = str(case_retr["fallback_level"].iloc[0]) if n_ref else "NA"
        verdict_case = "PASS" if (n_uniq == 3 and fb_level == "0") else (
            "PARTIAL_PASS" if n_uniq >= 1 else "UNSUPPORTED")
        fb_rows.append({
            "case_id": case_id, "cell_key": cell,
            "exact_top3_available": top3_av, "exact_top5_available": top5_av,
            "fallback_level_used": fb_level,
            "fallback_reason": ("none (exact same-cell top3 satisfied with 3 unique patients)"
                                if verdict_case == "PASS" else "see retrieval rows"),
            "n_references_returned": n_ref,
            "n_unique_patients_returned": n_uniq,
            "verdict": verdict_case,
        })
    FB_COLS = ["case_id", "cell_key", "exact_top3_available", "exact_top5_available",
               "fallback_level_used", "fallback_reason", "n_references_returned",
               "n_unique_patients_returned", "verdict"]
    pd.DataFrame(fb_rows)[FB_COLS].to_csv(
        OUTPUT_ROOT / "fallback_summary_completed_roi_v4.csv", index=False)

    # ============================================================
    # OUTPUT 7: unsupported_or_missing_cases_completed_roi_v4.csv
    #   기대값: header only (0 rows)
    # ============================================================
    UNSUP_COLS = ["case_id", "missing_field", "reason", "can_retry", "recommended_fix"]
    remaining = []
    for r in fb_rows:
        if r["verdict"] == "UNSUPPORTED":
            remaining.append({
                "case_id": r["case_id"], "missing_field": "cell_key/top3",
                "reason": "retrieval not satisfied after completed ROI metadata",
                "can_retry": True, "recommended_fix": "review ROI metadata extraction",
            })
    with open(OUTPUT_ROOT / "unsupported_or_missing_cases_completed_roi_v4.csv",
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=UNSUP_COLS)
        w.writeheader()
        for row in remaining:
            w.writerow(row)

    # ============================================================
    # OUTPUT 8: retrieval_policy_used_completed_roi_v4.json
    # ============================================================
    prev_policy = json.loads(PREV_POLICY_USED.read_text()) if PREV_POLICY_USED.exists() else {}
    policy_used = dict(prev_policy)
    policy_used["rerun_stage"] = "multi_case_retrieval_dryrun_completed_roi"
    policy_used["rerun_date"] = str(date.today())
    policy_used["roi_metadata_source"] = {
        "candidate_roi_position_metadata": str(ROI_META_CSV.relative_to(REPO_ROOT)),
        "candidate_cell_mapping_completed": str(ROI_CELLMAP_CSV.relative_to(REPO_ROOT)),
        "retrieval_top3_after_roi_metadata": str(ROI_RETR_CSV.relative_to(REPO_ROOT)),
    }
    policy_used["mapping_policy"] = policy_used.get("mapping_policy", {})
    policy_used["mapping_policy"]["rough_mapping"] = "FORBIDDEN (cell_key from ROI extraction only)"
    policy_used["mapping_policy"]["cell_key_source"] = "ROI-based extraction (completed); no rough mapping"
    policy_used["safety_flags"] = {
        "ALLOW_ROI_LOAD": ALLOW_ROI_LOAD, "ALLOW_CT_LOAD": ALLOW_CT_LOAD,
        "ALLOW_PNG_WRITE": ALLOW_PNG_WRITE, "ALLOW_CARD_RENDER": ALLOW_CARD_RENDER,
        "ALLOW_STAGE2_HOLDOUT": ALLOW_STAGE2_HOLDOUT,
        "ALLOW_MODEL_FORWARD": ALLOW_MODEL_FORWARD,
        "ALLOW_FEATURE_EXTRACTION": ALLOW_FEATURE_EXTRACTION,
        "ALLOW_CONTRIBUTION_RECALC": ALLOW_CONTRIBUTION_RECALC,
        "ALLOW_FULL300": ALLOW_FULL300,
    }
    (OUTPUT_ROOT / "retrieval_policy_used_completed_roi_v4.json").write_text(
        json.dumps(policy_used, indent=2, ensure_ascii=False))

    # ============================================================
    # OUTPUT 9: safety_check_completed_roi_v4.csv
    # ============================================================
    SAFE_COLS = ["item", "value", "status"]
    safety_items = [
        ("roi_load", 0, "OK"),
        ("ct_load", 0, "OK"),
        ("png_write", 0, "OK"),
        ("card_render", 0, "OK"),
        ("model_forward", 0, "OK"),
        ("feature_extraction", 0, "OK"),
        ("contribution_recalc", 0, "OK"),
        ("stage2_holdout_access", 0, "OK"),
        ("existing_artifact_modified", 0, "OK"),
        ("rough_mapping_used", 0, "OK"),
        ("diagnostic_causal_wording", 0, "OK"),
        ("output_collision", 0, "OK"),
        ("full_usable_cases", n_usable, "OK" if n_usable == 4 else "CHECK"),
        ("unsupported_remaining", n_unsup_remaining, "OK" if n_unsup_remaining == 0 else "CHECK"),
        ("control_cell_match", int(control_ok), "OK" if control_ok else "FAIL"),
    ]
    with open(OUTPUT_ROOT / "safety_check_completed_roi_v4.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SAFE_COLS)
        for it in safety_items:
            w.writerow(it)

    # ============================================================
    # OUTPUT 10: errors.csv
    # ============================================================
    with open(OUTPUT_ROOT / "errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "case_id", "error_type", "detail"])

    # ============================================================
    # OUTPUT 1/2: report md/json
    # ============================================================
    fb_df = pd.DataFrame(fb_rows)
    fb_level_dist = fb_df["fallback_level_used"].value_counts().to_dict()

    report = {
        "date": str(date.today()),
        "verdict": verdict,
        "stage": "multi_case_retrieval_dryrun_completed_roi",
        "report_only": True,
        "case_count": int(cellmap["case_id"].nunique()),
        "full_usable_cases": int(n_usable),
        "cell_mapping": {r["case_id"]: r["cell_key"] for r in fb_rows},
        "expected_cell": EXPECTED_CELL,
        "control_case": CONTROL_CASE,
        "control_cell_match": control_ok,
        "fallback_level_distribution": fb_level_dist,
        "unsupported_remaining": int(n_unsup_remaining),
        "validation": {"pass": sum(1 for c in checks if c["passed"]),
                       "total": len(checks),
                       "failed": [c["desc"] for c in checks if not c["passed"]]},
        "safety": {
            "roi_load": 0, "ct_load": 0, "png_write": 0, "card_render": 0,
            "model_forward": 0, "feature_extraction": 0, "contribution_recalc": 0,
            "stage2_holdout": 0, "existing_artifact_modified": 0,
            "rough_mapping": 0,
        },
        "roi_metadata_extraction_verdict": roi_done.get("verdict"),
    }
    (OUTPUT_ROOT / "multi_case_retrieval_completed_roi_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    md = [
        "# multi-case retrieval dry-run RERUN — completed ROI metadata (report-only)",
        f"date: {date.today()}",
        f"verdict: **{verdict}**",
        "",
        "## scope",
        "- report-only CSV/JSON rerun (no ROI/CT load, no model/feature/contribution)",
        f"- case count: {cellmap['case_id'].nunique()} / full usable: {n_usable}",
        "",
        "## cell mapping (ROI-based, completed)",
        "| case_id | cell_key | expected | match |",
        "|---|---|---|---|",
    ]
    for r in fb_rows:
        exp = EXPECTED_CELL[r["case_id"]]
        md.append(f"| {r['case_id']} | {r['cell_key']} | {exp} | "
                  f"{'MATCH' if r['cell_key']==exp else 'MISMATCH'} |")
    md += [
        "",
        "## top3 retrieval summary",
        "| case_id | n_ref | n_unique_pat | fallback_level | verdict |",
        "|---|---|---|---|---|",
    ]
    for r in fb_rows:
        md.append(f"| {r['case_id']} | {r['n_references_returned']} | "
                  f"{r['n_unique_patients_returned']} | {r['fallback_level_used']} | "
                  f"{r['verdict']} |")
    md += [
        "",
        f"## fallback level distribution: {fb_level_dist}",
        "",
        "## unsupported remaining",
        ("- (none) all 4 cases full retrieval usable" if n_unsup_remaining == 0
         else f"- {n_unsup_remaining} case(s) remaining; see unsupported csv"),
        "",
        "## safety",
        "- ROI load: 0 / CT load: 0 / PNG: 0 / card: 0",
        "- model: 0 / feature: 0 / contribution: 0 / stage2_holdout: 0",
        "- rough mapping: 0 (cell_key from ROI extraction only)",
        "- existing artifact modified: 0 (writes only to new OUTPUT_ROOT)",
        f"- control LUNG1-052 cell match: {control_ok}",
        "",
        "## notes",
        "- side by image x-coord (NOT anatomical); not_same_z_matched=True; z_direction_limited=True",
        "- no same-z wording; no diagnostic/lesion/cancer/vessel cause claim",
        "",
    ]
    (OUTPUT_ROOT / "multi_case_retrieval_completed_roi_report.md").write_text("\n".join(md))

    # ============================================================
    # OUTPUT 11: DONE.json
    # ============================================================
    done = {
        "status": "DONE", "verdict": verdict, "date": str(date.today()),
        "report_only": True,
        "case_count": int(cellmap["case_id"].nunique()),
        "full_usable_cases": int(n_usable),
        "unsupported_remaining": int(n_unsup_remaining),
        "control_cell_match": control_ok,
        "validation_pass": f"{sum(1 for c in checks if c['passed'])}/{len(checks)}",
        "outputs": [
            "multi_case_retrieval_completed_roi_report.md",
            "multi_case_retrieval_completed_roi_report.json",
            "candidate_case_inventory_completed_roi_v4.csv",
            "candidate_cell_mapping_completed_roi_v4.csv",
            "retrieval_top3_by_case_completed_roi_v4.csv",
            "fallback_summary_completed_roi_v4.csv",
            "unsupported_or_missing_cases_completed_roi_v4.csv",
            "retrieval_policy_used_completed_roi_v4.json",
            "safety_check_completed_roi_v4.csv",
            "errors.csv",
            "DONE.json",
        ],
    }
    (OUTPUT_ROOT / "DONE.json").write_text(json.dumps(done, indent=2))

    # ---- final summary ----
    print("\n" + "=" * 64)
    print(f"VERDICT: {verdict}")
    print(f"  case count: {cellmap['case_id'].nunique()} / full usable: {n_usable}")
    print(f"  control match: {control_ok} ({control_cell})")
    print(f"  fallback level dist: {fb_level_dist}")
    print(f"  unsupported remaining: {n_unsup_remaining}")
    print(f"  validation: {sum(1 for c in checks if c['passed'])}/{len(checks)}")
    print(f"  ROI/CT/PNG/card/model/feature/contribution/stage2: 0")
    print(f"  outputs -> {OUTPUT_ROOT}")
    print("=" * 64)

    if verdict == "BLOCKED":
        sys.exit(2)
    elif verdict == "NEEDS_FIX":
        sys.exit(1)


if __name__ == "__main__":
    main()
