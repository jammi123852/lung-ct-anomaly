"""
preflight_selected_cases_crop_preview_v4.py

목적:
reference bank v4 retrieval close/handoff 결과를 바탕으로,
선택된 4개 S5 case 의 candidate crop + same-cell normal reference top3 preview 를
실제 생성할 수 있는지 사전 검증(preflight)한다.

이번 단계는 preflight only.
- PNG 생성 / CT load / ROI load / card render 전부 금지
- 경로 존재 확인(os.path.exists, stat) + crop bounds 산술 점검만 수행
- 기존 artifact 수정 금지 (신규 OUTPUT_ROOT 에만 기록)
- rough mapping 금지 / 병변·암·혈관 원인 단정 금지

guard (전부 False):
  ALLOW_CT_LOAD / ALLOW_ROI_LOAD / ALLOW_PNG_WRITE / ALLOW_CARD_RENDER
  ALLOW_STAGE2_HOLDOUT / ALLOW_MODEL_FORWARD / ALLOW_FEATURE_EXTRACTION
  ALLOW_CONTRIBUTION_RECALC / ALLOW_FULL300
"""

import os
import sys
import csv
import json
import pathlib
from datetime import date

import pandas as pd

# ============================================================
# GUARD FLAGS
# ============================================================
ALLOW_CT_LOAD             = False
ALLOW_ROI_LOAD            = False
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

HANDOFF_ROOT = R / "reference_bank_v4_multi_case_retrieval_close_handoff"
RERUN_ROOT   = R / "reference_bank_v4_multi_case_retrieval_dryrun_completed_roi"
BANK_ROOT    = R / "reference_bank_v4_lung_roi_position_metadata"
DRYRUN_ROOT  = R / "reference_bank_v4_multi_case_retrieval_dryrun"
OUTPUT_ROOT  = R / "reference_bank_v4_selected_cases_crop_preview_preflight"

HANDOFF_DONE   = HANDOFF_ROOT / "DONE.json"
HANDOFF_JSON   = HANDOFF_ROOT / "reference_bank_v4_multi_case_retrieval_close_handoff.json"
RERUN_DONE     = RERUN_ROOT / "DONE.json"
RERUN_RETR     = RERUN_ROOT / "retrieval_top3_by_case_completed_roi_v4.csv"
RERUN_CELLMAP  = RERUN_ROOT / "candidate_cell_mapping_completed_roi_v4.csv"
RERUN_FALLBACK = RERUN_ROOT / "fallback_summary_completed_roi_v4.csv"
BANK_TOP3      = BANK_ROOT / "normal_reference_bank_v4_top3_by_cell.csv"
CAND_INVENTORY = DRYRUN_ROOT / "candidate_case_inventory_v4.csv"

# candidate volume root (NSCLC/MSD test-ready)
CAND_VOLUME_ROOT = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")

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
CROP_SIZE = 96
IMG_H = IMG_W = 512

FORBIDDEN_PHRASES = [
    "same-z matched", "same z matched", "z-matched", "z matched",
    "identical z", "동일 z 위치", "같은 z 위치",
    "diagnostic heatmap", "grad-cam", "pixel attribution",
    "병변 원인", "암 위치", "혈관 때문에",
]
# actual 단계 PNG 정책 (이번 preflight 에서는 생성 0)
PREVIEW_NAMING = [
    "{case_id}_candidate.png",
    "{case_id}_normal_ref_1.png",
    "{case_id}_normal_ref_2.png",
    "{case_id}_normal_ref_3.png",
    "{case_id}_reference_preview_montage.png",
]


def _abort(msg, code=2):
    print(f"\nBLOCKED: {msg}", file=sys.stderr)
    sys.exit(code)


def _scan_forbidden(blob):
    low = blob.lower()
    return [p for p in FORBIDDEN_PHRASES if p.lower() in low]


def crop_in_bounds(y0, x0, y1, x1):
    return (y0 >= 0 and x0 >= 0 and y1 <= IMG_H and x1 <= IMG_W
            and (y1 - y0) == CROP_SIZE and (x1 - x0) == CROP_SIZE)


# ============================================================
# VALIDATION (25 checks)
# ============================================================
def run_validation(cand_plan, ref_plan):
    checks = []

    def chk(idx, desc, passed, note=""):
        checks.append({"id": idx, "desc": desc, "passed": bool(passed), "note": note})

    # 1. close/handoff DONE exists
    h_ok = HANDOFF_DONE.exists()
    chk(1, "handoff_DONE_exists", h_ok)
    h_done = json.loads(HANDOFF_DONE.read_text()) if h_ok else {}
    # 2. handoff verdict PASS
    chk(2, "handoff_verdict_PASS", h_done.get("verdict") == "PASS",
        f"verdict={h_done.get('verdict')}")
    # 3. completed ROI rerun DONE exists
    r_ok = RERUN_DONE.exists()
    chk(3, "completed_roi_rerun_DONE_exists", r_ok)
    r_done = json.loads(RERUN_DONE.read_text()) if r_ok else {}
    # 4. 4/4 case usable (rerun DONE key = full_usable_cases)
    full_usable = r_done.get("full_usable_cases", r_done.get("full_usable"))
    chk(4, "usable_4_4", full_usable == 4, f"full_usable_cases={full_usable}")
    # 5. unsupported remaining 0
    chk(5, "unsupported_remaining_0", r_done.get("unsupported_remaining") == 0)
    # 6. each case has candidate crop bbox
    chk(6, "each_case_candidate_bbox",
        all(c["crop_y0"] != "" and c["crop_x0"] != "" for c in cand_plan))
    # 7. each case has candidate ct_path
    chk(7, "each_case_candidate_ct_path",
        all(bool(c["ct_path"]) for c in cand_plan))
    # 8. each case candidate ct_path exists
    chk(8, "each_case_candidate_ct_exists",
        all(c["ct_exists"] for c in cand_plan),
        f"exists={sum(c['ct_exists'] for c in cand_plan)}/4")
    # 9. each case top3 retrieval rows exist (3 per case)
    per_case = {c: 0 for c in EXPECTED_CASES}
    for r in ref_plan:
        per_case[r["case_id"]] = per_case.get(r["case_id"], 0) + 1
    chk(9, "each_case_top3_rows", all(per_case[c] == 3 for c in EXPECTED_CASES),
        f"{per_case}")
    # 10. each case top3 unique patients
    uniq_ok = True
    for c in EXPECTED_CASES:
        pats = [r["reference_patient_id"] for r in ref_plan if r["case_id"] == c]
        if len(set(pats)) != 3:
            uniq_ok = False
    chk(10, "each_case_3_unique_patients", uniq_ok)
    # 11. all top3 ct_path exists
    chk(11, "all_top3_ct_exists", all(r["ct_exists"] for r in ref_plan),
        f"exists={sum(r['ct_exists'] for r in ref_plan)}/{len(ref_plan)}")
    # 12. all crop_size == 96
    sizes_ok = all(c["crop_size"] == CROP_SIZE for c in cand_plan) and \
               all(r["crop_size"] == CROP_SIZE for r in ref_plan)
    chk(12, "all_crop_size_96", sizes_ok)
    # 13. all crop bounds valid
    bounds_ok = all(c["crop_in_bounds"] for c in cand_plan) and \
                all(r["crop_in_bounds"] for r in ref_plan)
    chk(13, "all_crop_bounds_valid", bounds_ok)
    # 14. all stage2_holdout false
    chk(14, "stage2_holdout_false_all",
        all(not c["stage2_holdout_flag"] for c in cand_plan))
    # 15. fallback_level 0 for all
    chk(15, "fallback_level_0_all",
        all(str(r["fallback_level"]) == "0" for r in ref_plan))
    # 16. not_same_z_matched true
    chk(16, "not_same_z_matched_true_all",
        all(str(r["not_same_z_matched"]) == "True" for r in ref_plan))
    # 17. z_direction_limited true
    chk(17, "z_direction_limited_true_all",
        all(str(r["z_direction_limited"]) == "True" for r in ref_plan))
    # 18/19. no forbidden wording in SOURCE inputs
    src_blob = json.dumps(h_done) + json.dumps(r_done)
    for r in ref_plan:
        src_blob += " " + str(r.get("notes", ""))
    chk(18, "no_same_z_wording_in_source", len(_scan_forbidden(src_blob)) == 0,
        f"hits={_scan_forbidden(src_blob)}")
    chk(19, "no_diagnostic_causal_wording", len(_scan_forbidden(src_blob)) == 0)
    # 20-23 guards
    chk(20, "ct_load_0", not ALLOW_CT_LOAD)
    chk(21, "roi_load_0", not ALLOW_ROI_LOAD)
    chk(22, "png_card_0", not (ALLOW_PNG_WRITE or ALLOW_CARD_RENDER))
    chk(23, "model_feature_contribution_0",
        not (ALLOW_MODEL_FORWARD or ALLOW_FEATURE_EXTRACTION or ALLOW_CONTRIBUTION_RECALC))
    # 24. existing artifact modification 0 (write only new root)
    chk(24, "writes_only_new_root", "crop_preview_preflight" in str(OUTPUT_ROOT))
    # 25. output collision 없음
    chk(25, "no_output_collision", not (OUTPUT_ROOT / "DONE.json").exists())
    # extra: cell_key matches expected (NEEDS_FIX guard)
    chk("E1", "cell_key_matches_expected",
        all(c["cell_key"] == EXPECTED_CELL[c["case_id"]] for c in cand_plan))

    n_pass = sum(1 for c in checks if c["passed"])
    n_fail = len(checks) - n_pass
    print("=" * 64)
    print("PREFLIGHT VALIDATION (selected cases crop preview, preflight-only)")
    print("=" * 64)
    for c in checks:
        flag = "PASS" if c["passed"] else "FAIL"
        extra = f"  ({c['note']})" if c["note"] else ""
        print(f"  [{flag}] {str(c['id']):>3}. {c['desc']}{extra}")
    print(f"  ---> {n_pass} PASS / {n_fail} FAIL")

    hard = {1, 2, 3, 20, 21, 22, 23, 25}
    failed_hard = [c for c in checks if c["id"] in hard and not c["passed"]]
    if failed_hard:
        ids = ", ".join(str(c["id"]) for c in failed_hard)
        _abort(f"hard guard/collision check failed: {ids}")

    return checks


# ============================================================
# BUILD PLANS
# ============================================================
def build_plans():
    inv = pd.read_csv(CAND_INVENTORY, dtype=str)
    retr = pd.read_csv(RERUN_RETR)
    cellmap = pd.read_csv(RERUN_CELLMAP)
    bank_top3 = pd.read_csv(BANK_TOP3)
    bank_ct = dict(zip(bank_top3["reference_id"], bank_top3["ct_path"]))
    bank_roi = dict(zip(bank_top3["reference_id"], bank_top3["roi_path"]))
    cell_idx = dict(zip(cellmap["case_id"], cellmap["cell_key"]))

    cand_plan, ref_plan = [], []

    for case_id in EXPECTED_CASES:
        irow = inv[inv["case_id"] == case_id].iloc[0]
        volume_id = irow["volume_id"]
        source = irow["source"]
        local_z = int(irow["local_z"])
        y0 = int(irow["crop_y0"]); x0 = int(irow["crop_x0"])
        y1 = int(irow["crop_y1"]); x1 = int(irow["crop_x1"])
        ct_path = str(CAND_VOLUME_ROOT / volume_id / "ct_hu.npy")
        ct_exists = os.path.exists(ct_path)
        stage2 = str(irow.get("stage2_holdout_flag", "False")) == "True"
        cell_key = cell_idx.get(case_id, "")
        inb = crop_in_bounds(y0, x0, y1, x1)
        cand_plan.append({
            "case_id": case_id, "role": "candidate",
            "patient_id": irow["patient_id"], "volume_id": volume_id,
            "source": source, "cell_key": cell_key, "ct_path": ct_path,
            "local_z": local_z,
            "crop_y0": y0, "crop_x0": x0, "crop_y1": y1, "crop_x1": x1,
            "crop_size": CROP_SIZE, "crop_in_bounds": inb, "ct_exists": ct_exists,
            "stage2_holdout_flag": stage2,
            "fallback_level": 0, "reference_quality_score": "",
            "not_same_z_matched": "True", "z_direction_limited": "True",
            "planned_png_name": f"{case_id}_candidate.png",
            "notes": ("MSD source" if source == "MSD_Lung" else "")
                     + ("; " if source == "MSD_Lung" else "")
                     + "candidate crop; same-cell comparison, not same-z matching",
        })

        case_retr = retr[retr["case_id"] == case_id].sort_values("retrieval_rank")
        for _, rr in case_retr.iterrows():
            rank = int(rr["retrieval_rank"])
            rid = rr["reference_id"]
            ref_ct = bank_ct.get(rid, "")
            ref_ct_exists = os.path.exists(ref_ct) if ref_ct else False
            ry0 = int(rr["reference_crop_y0"]); rx0 = int(rr["reference_crop_x0"])
            ry1 = int(rr["reference_crop_y1"]); rx1 = int(rr["reference_crop_x1"])
            rinb = crop_in_bounds(ry0, rx0, ry1, rx1)
            ref_plan.append({
                "case_id": case_id, "role": f"normal_ref_{rank}",
                "retrieval_rank": rank, "reference_id": rid,
                "reference_patient_id": rr["reference_patient_id"],
                "reference_volume_id": rr["reference_volume_id"],
                "reference_cell_key": rr["reference_cell_key"],
                "ct_path": ref_ct, "ct_exists": ref_ct_exists,
                "reference_local_z": int(rr["reference_local_z"]),
                "reference_crop_y0": ry0, "reference_crop_x0": rx0,
                "reference_crop_y1": ry1, "reference_crop_x1": rx1,
                "crop_size": CROP_SIZE, "crop_in_bounds": rinb,
                "unique_patient_flag": str(rr["unique_patient_flag"]),
                "same_cell_flag": str(rr["same_cell_flag"]),
                "same_side_flag": str(rr["same_side_flag"]),
                "fallback_level": int(rr["fallback_level"]),
                "reference_quality_score": float(rr["reference_quality_score"]),
                "not_same_z_matched": str(rr["not_same_z_matched"]),
                "z_direction_limited": str(rr["z_direction_limited"]),
                "planned_png_name": f"{case_id}_normal_ref_{rank}.png",
            })

    return cand_plan, ref_plan


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 64)
    print("selected cases crop preview PREFLIGHT (preflight-only; no CT/ROI/PNG)")
    print(f"date: {date.today()}")
    print("=" * 64)

    cand_plan, ref_plan = build_plans()
    checks = run_validation(cand_plan, ref_plan)

    failed = [c for c in checks if not c["passed"]]
    # readiness counts
    cand_ready = sum(1 for c in cand_plan if c["ct_exists"] and c["crop_in_bounds"]
                     and not c["stage2_holdout_flag"])
    ref_ready = sum(1 for r in ref_plan if r["ct_exists"] and r["crop_in_bounds"])
    cell_ok = all(c["cell_key"] == EXPECTED_CELL[c["case_id"]] for c in cand_plan)

    if not cell_ok:
        verdict = "NEEDS_FIX"
    elif len(failed) == 0 and cand_ready == 4 and ref_ready == 12:
        verdict = "PASS"
    elif cand_ready >= 1 or ref_ready >= 1:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "NEEDS_FIX"

    # collision before write
    if (OUTPUT_ROOT / "DONE.json").exists():
        _abort("existing DONE.json at OUTPUT_ROOT. Archive before preflight.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ---- 3. selected_cases_crop_preview_plan_v4.csv (candidate + refs unified) ----
    PLAN_COLS = ["case_id", "role", "patient_id", "volume_id", "source", "cell_key",
                 "ct_path", "local_z", "crop_y0", "crop_x0", "crop_y1", "crop_x1",
                 "crop_size", "crop_in_bounds", "ct_exists", "stage2_holdout_flag",
                 "fallback_level", "reference_quality_score",
                 "not_same_z_matched", "z_direction_limited",
                 "planned_png_name", "notes"]
    plan_rows = []
    for c in cand_plan:
        plan_rows.append({k: c.get(k, "") for k in PLAN_COLS})
    for r in ref_plan:
        plan_rows.append({
            "case_id": r["case_id"], "role": r["role"],
            "patient_id": r["reference_patient_id"],
            "volume_id": r["reference_volume_id"], "source": "NSCLC_Normal_LUNA16",
            "cell_key": r["reference_cell_key"], "ct_path": r["ct_path"],
            "local_z": r["reference_local_z"],
            "crop_y0": r["reference_crop_y0"], "crop_x0": r["reference_crop_x0"],
            "crop_y1": r["reference_crop_y1"], "crop_x1": r["reference_crop_x1"],
            "crop_size": r["crop_size"], "crop_in_bounds": r["crop_in_bounds"],
            "ct_exists": r["ct_exists"], "stage2_holdout_flag": False,
            "fallback_level": r["fallback_level"],
            "reference_quality_score": r["reference_quality_score"],
            "not_same_z_matched": r["not_same_z_matched"],
            "z_direction_limited": r["z_direction_limited"],
            "planned_png_name": r["planned_png_name"],
            "notes": "normal reference crop; same lung-ROI position cell; not same-z matching",
        })
    # case 순서 정렬
    order = {c: i for i, c in enumerate(EXPECTED_CASES)}
    role_order = {"candidate": 0, "normal_ref_1": 1, "normal_ref_2": 2, "normal_ref_3": 3}
    plan_df = pd.DataFrame(plan_rows)[PLAN_COLS]
    plan_df["_o"] = plan_df["case_id"].map(order)
    plan_df["_r"] = plan_df["role"].map(role_order)
    plan_df = plan_df.sort_values(["_o", "_r"]).drop(columns=["_o", "_r"])
    plan_df.to_csv(OUTPUT_ROOT / "selected_cases_crop_preview_plan_v4.csv", index=False)

    # ---- 4. candidate_crop_readiness_v4.csv ----
    CAND_COLS = ["case_id", "ct_path", "ct_exists", "local_z", "crop_y0", "crop_x0",
                 "crop_y1", "crop_x1", "crop_size", "crop_in_bounds",
                 "readiness_status", "reason_if_not_ready"]
    cand_ready_rows = []
    for c in cand_plan:
        ready = c["ct_exists"] and c["crop_in_bounds"] and not c["stage2_holdout_flag"]
        reason = ""
        if not c["ct_exists"]:
            reason = "ct_path missing"
        elif not c["crop_in_bounds"]:
            reason = "crop out of bounds / size!=96"
        elif c["stage2_holdout_flag"]:
            reason = "stage2_holdout_flag True"
        cand_ready_rows.append({
            "case_id": c["case_id"], "ct_path": c["ct_path"], "ct_exists": c["ct_exists"],
            "local_z": c["local_z"], "crop_y0": c["crop_y0"], "crop_x0": c["crop_x0"],
            "crop_y1": c["crop_y1"], "crop_x1": c["crop_x1"], "crop_size": c["crop_size"],
            "crop_in_bounds": c["crop_in_bounds"],
            "readiness_status": "READY" if ready else "NOT_READY",
            "reason_if_not_ready": reason,
        })
    pd.DataFrame(cand_ready_rows)[CAND_COLS].to_csv(
        OUTPUT_ROOT / "candidate_crop_readiness_v4.csv", index=False)

    # ---- 5. normal_reference_crop_readiness_v4.csv ----
    REF_COLS = ["case_id", "retrieval_rank", "reference_id", "reference_patient_id",
                "reference_volume_id", "reference_cell_key", "ct_path", "ct_exists",
                "reference_local_z", "reference_crop_y0", "reference_crop_x0",
                "reference_crop_y1", "reference_crop_x1", "crop_size", "crop_in_bounds",
                "unique_patient_flag", "same_cell_flag", "same_side_flag",
                "readiness_status", "reason_if_not_ready"]
    ref_ready_rows = []
    for r in ref_plan:
        ready = r["ct_exists"] and r["crop_in_bounds"]
        reason = "" if ready else ("ct_path missing" if not r["ct_exists"]
                                   else "crop out of bounds / size!=96")
        ref_ready_rows.append({
            "case_id": r["case_id"], "retrieval_rank": r["retrieval_rank"],
            "reference_id": r["reference_id"],
            "reference_patient_id": r["reference_patient_id"],
            "reference_volume_id": r["reference_volume_id"],
            "reference_cell_key": r["reference_cell_key"],
            "ct_path": r["ct_path"], "ct_exists": r["ct_exists"],
            "reference_local_z": r["reference_local_z"],
            "reference_crop_y0": r["reference_crop_y0"],
            "reference_crop_x0": r["reference_crop_x0"],
            "reference_crop_y1": r["reference_crop_y1"],
            "reference_crop_x1": r["reference_crop_x1"],
            "crop_size": r["crop_size"], "crop_in_bounds": r["crop_in_bounds"],
            "unique_patient_flag": r["unique_patient_flag"],
            "same_cell_flag": r["same_cell_flag"], "same_side_flag": r["same_side_flag"],
            "readiness_status": "READY" if ready else "NOT_READY",
            "reason_if_not_ready": reason,
        })
    pd.DataFrame(ref_ready_rows)[REF_COLS].to_csv(
        OUTPUT_ROOT / "normal_reference_crop_readiness_v4.csv", index=False)

    # ---- 6/7. preview_generation_policy (md + json) ----
    planned_png = {
        "per_case": ["candidate(1)", "normal_ref(3)", "montage(1)"],
        "per_case_count": 5,
        "total_cases": 4,
        "max_total_png": 20,
        "this_preflight_png": 0,
    }
    policy = {
        "version": "v4",
        "stage": "selected_cases_crop_preview_preflight",
        "this_stage_png_generation": 0,
        "actual_stage_allowed_png": {
            "candidate_per_case": 1, "normal_ref_per_case": 3,
            "montage_per_case": 1, "max_total": 20,
        },
        "preview_naming_rule": PREVIEW_NAMING,
        "display_policy": {
            "crop_size": "96x96 (all crops identical display size)",
            "window": "lung window",
            "label_fields": ["role", "z (local_z)", "cell_key", "fallback_level"],
            "caution_text": "same-cell comparison, not same-z matching",
            "forbidden": "no diagnostic/causal expression; "
                         "see forbidden_phrases list",
        },
        "forbidden_phrases": FORBIDDEN_PHRASES,
        "matching_clarification": {
            "not_same_z_matched": True, "z_direction_limited": True,
            "is_same_z_matching": False,
            "basis": "same lung-ROI position cell (image x-coord side; NOT anatomical)",
        },
        "msd_note": "MSD_lung_059__c2 is MSD source; record source; "
                    "no diagnosis/cancer/lesion causal claim",
        "source_inputs": {
            "handoff": str(HANDOFF_ROOT.relative_to(REPO_ROOT)),
            "rerun": str(RERUN_ROOT.relative_to(REPO_ROOT)),
            "bank_top3": str(BANK_TOP3.relative_to(REPO_ROOT)),
        },
    }
    (OUTPUT_ROOT / "preview_generation_policy_v4.json").write_text(
        json.dumps(policy, indent=2, ensure_ascii=False))

    pol_md = [
        "# selected cases crop preview — generation policy (v4)",
        f"date: {date.today()}",
        "",
        "## this stage",
        "- preflight only; PNG generation in this stage = 0",
        "",
        "## actual stage allowed PNG (separate approval)",
        "- per case: 1 candidate + 3 normal_ref + 1 montage = 5",
        "- 4 cases x 5 = max 20 PNG",
        "",
        "## preview naming rule",
    ] + [f"- {n}" for n in PREVIEW_NAMING] + [
        "",
        "## display policy",
        "- all crops 96x96, identical display size, lung window",
        "- label: role / z (local_z) / cell_key / fallback_level",
        "- caution text: \"same-cell comparison, not same-z matching\"",
        "- no diagnostic/causal expression (see forbidden_phrases in policy json)",
        "",
        "## matching clarification",
        "- not same-z matching; z-direction alignment is limited (always preserved)",
        "- basis: same lung-ROI position cell; side by image x-coordinate (NOT anatomical)",
        "",
        "## MSD note",
        "- MSD_lung_059__c2 is MSD source; record source; no diagnosis/causal claim",
        "",
    ]
    (OUTPUT_ROOT / "preview_generation_policy_v4.md").write_text("\n".join(pol_md))

    # ---- 8. safety_check_v4.csv ----
    with open(OUTPUT_ROOT / "safety_check_v4.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "value", "status"])
        items = [
            ("ct_load", 0, "OK"), ("roi_load", 0, "OK"),
            ("png_write", 0, "OK"), ("card_render", 0, "OK"),
            ("model_forward", 0, "OK"), ("feature_extraction", 0, "OK"),
            ("contribution_recalc", 0, "OK"), ("stage2_holdout_access", 0, "OK"),
            ("existing_artifact_modified", 0, "OK"), ("rough_mapping_used", 0, "OK"),
            ("diagnostic_causal_wording", 0, "OK"), ("output_collision", 0, "OK"),
            ("candidate_ready", cand_ready, "OK" if cand_ready == 4 else "CHECK"),
            ("normal_ref_ready", ref_ready, "OK" if ref_ready == 12 else "CHECK"),
            ("cell_key_match", int(cell_ok), "OK" if cell_ok else "FAIL"),
            ("this_stage_png", 0, "OK"),
        ]
        for it in items:
            w.writerow(it)

    # ---- 9. errors.csv ----
    with open(OUTPUT_ROOT / "errors.csv", "w", newline="") as f:
        csv.writer(f).writerow(["stage", "case_id", "error_type", "detail"])

    # ---- 1/2. preflight report (md + json) ----
    report = {
        "date": str(date.today()), "verdict": verdict,
        "stage": "selected_cases_crop_preview_preflight", "preflight_only": True,
        "this_stage_png_generation": 0,
        "case_readiness": {
            "candidate_ready": cand_ready, "candidate_total": 4,
            "normal_ref_ready": ref_ready, "normal_ref_total": 12,
        },
        "cell_key_match": cell_ok,
        "planned_png_actual_stage": planned_png,
        "validation": {"pass": sum(1 for c in checks if c["passed"]),
                       "total": len(checks),
                       "failed": [c["desc"] for c in checks if not c["passed"]]},
        "safety": {
            "ct_load": 0, "roi_load": 0, "png_write": 0, "card_render": 0,
            "model_forward": 0, "feature_extraction": 0, "contribution_recalc": 0,
            "stage2_holdout": 0, "existing_artifact_modified": 0, "rough_mapping": 0,
        },
    }
    (OUTPUT_ROOT / "selected_cases_crop_preview_preflight_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    md = [
        "# selected cases crop preview — PREFLIGHT report (v4)",
        f"date: {date.today()}",
        f"verdict: **{verdict}**",
        "",
        "## scope",
        "- preflight only; PNG/CT/ROI/card generation = 0",
        "- path existence + crop bounds validation only",
        "",
        "## case readiness",
        f"- candidate ready: {cand_ready}/4",
        f"- normal reference ready: {ref_ready}/12",
        f"- cell_key match (expected): {cell_ok}",
        "",
        "## candidate crop readiness",
        "| case_id | ct_exists | local_z | crop(y0,x0,y1,x1) | in_bounds | status |",
        "|---|---|---|---|---|---|",
    ]
    for c in cand_ready_rows:
        md.append(f"| {c['case_id']} | {c['ct_exists']} | {c['local_z']} | "
                  f"({c['crop_y0']},{c['crop_x0']},{c['crop_y1']},{c['crop_x1']}) | "
                  f"{c['crop_in_bounds']} | {c['readiness_status']} |")
    md += [
        "",
        "## normal reference crop readiness (per case top3)",
        "| case_id | rank | ref_patient | ct_exists | in_bounds | status |",
        "|---|---|---|---|---|---|",
    ]
    for r in ref_ready_rows:
        md.append(f"| {r['case_id']} | {r['retrieval_rank']} | "
                  f"{str(r['reference_patient_id'])[:24]} | {r['ct_exists']} | "
                  f"{r['crop_in_bounds']} | {r['readiness_status']} |")
    md += [
        "",
        "## planned PNG (actual stage, separate approval)",
        "- per case: 1 candidate + 3 normal_ref + 1 montage = 5",
        "- 4 cases x 5 = max 20 PNG (this preflight: 0 PNG)",
        "",
        "## safety",
        "- CT load: 0 / ROI load: 0 / PNG: 0 / card: 0",
        "- model: 0 / feature: 0 / contribution: 0 / stage2_holdout: 0",
        "- existing artifact modified: 0 (writes only to new OUTPUT_ROOT)",
        "- rough mapping: 0; no diagnostic/causal wording",
        "",
        "## notes",
        "- same-cell comparison, not same-z matching; z-direction alignment is limited",
        "- side by image x-coordinate (NOT anatomical); MSD_lung_059 is MSD source",
        "",
    ]
    self_blob = "\n".join(md)
    hits = _scan_forbidden(self_blob)
    if hits:
        _abort(f"forbidden phrase in preflight md body: {hits}")
    (OUTPUT_ROOT / "selected_cases_crop_preview_preflight_report.md").write_text(self_blob)

    # ---- 10. DONE.json ----
    done = {
        "status": "DONE", "verdict": verdict, "date": str(date.today()),
        "preflight_only": True, "this_stage_png_generation": 0,
        "candidate_ready": cand_ready, "normal_ref_ready": ref_ready,
        "cell_key_match": cell_ok,
        "validation_pass": f"{sum(1 for c in checks if c['passed'])}/{len(checks)}",
        "outputs": [
            "selected_cases_crop_preview_preflight_report.md",
            "selected_cases_crop_preview_preflight_report.json",
            "selected_cases_crop_preview_plan_v4.csv",
            "candidate_crop_readiness_v4.csv",
            "normal_reference_crop_readiness_v4.csv",
            "preview_generation_policy_v4.md",
            "preview_generation_policy_v4.json",
            "safety_check_v4.csv",
            "errors.csv",
            "DONE.json",
        ],
    }
    (OUTPUT_ROOT / "DONE.json").write_text(json.dumps(done, indent=2))

    print("\n" + "=" * 64)
    print(f"VERDICT: {verdict}")
    print(f"  candidate ready: {cand_ready}/4 | normal_ref ready: {ref_ready}/12")
    print(f"  cell_key match: {cell_ok}")
    print(f"  planned PNG (actual stage): max 20 | this preflight: 0")
    print(f"  validation: {sum(1 for c in checks if c['passed'])}/{len(checks)}")
    print(f"  CT/ROI/PNG/card/model/feature/contribution/stage2: 0")
    print(f"  outputs -> {OUTPUT_ROOT}")
    print("=" * 64)

    if verdict == "NEEDS_FIX":
        sys.exit(1)


if __name__ == "__main__":
    main()
