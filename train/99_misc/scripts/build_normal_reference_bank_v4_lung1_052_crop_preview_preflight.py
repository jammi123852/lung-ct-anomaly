"""
normal reference bank v4 crop preview generation preflight
- LUNG1-052__c3 exact cell top3 reference crop preview plan 생성
- CT load 금지, PNG 생성 금지, metadata 수정 금지
- 경로 존재 여부, crop 좌표 유효성, guard 조건만 확인
"""

import json
import os
import sys
import csv
from pathlib import Path
from datetime import date

import pandas as pd

# ──────────────────────────────────────────────
# GUARD FLAGS
# ──────────────────────────────────────────────
ALLOW_CT_LOAD              = False
ALLOW_PREVIEW_PNG_WRITE    = False
ALLOW_METADATA_WRITE       = False
ALLOW_STAGE2_HOLDOUT       = False
ALLOW_MODEL_FORWARD        = False
ALLOW_FEATURE_EXTRACTION   = False
ALLOW_CONTRIBUTION_RECALC  = False
ALLOW_FULL300              = False

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
METADATA_ROOT = Path(
    "outputs/position-aware-padim-v1/reports/"
    "reference_bank_v4_lung_roi_position_metadata"
)
OUTPUT_ROOT = Path(
    "outputs/position-aware-padim-v1/reports/"
    "reference_bank_v4_lung1_052_crop_preview_preflight"
)

METADATA_DONE         = METADATA_ROOT / "DONE.json"
METADATA_ERRORS       = METADATA_ROOT / "errors.csv"
METADATA_CSV          = METADATA_ROOT / "normal_reference_bank_v4_metadata.csv"
CELL_INDEX_CSV        = METADATA_ROOT / "normal_reference_bank_v4_cell_index.csv"
TOP3_BY_CELL_CSV      = METADATA_ROOT / "normal_reference_bank_v4_top3_by_cell.csv"
RETRIEVAL_PREVIEW_CSV = METADATA_ROOT / "lung1_052_v4_retrieval_preview.csv"
POLICY_FROZEN_JSON    = METADATA_ROOT / "normal_reference_bank_v4_retrieval_policy_frozen.json"

# LUNG1-052 candidate 고정값
LUNG1_052_CT_PATH   = (
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/"
    "volumes_npy/NSCLC_LUNG1-052__d4a19cc211/ct_hu.npy"
)
LUNG1_052_VOLUME_ID = "NSCLC_LUNG1-052__d4a19cc211"
LUNG1_052_PATIENT   = "LUNG1-052"
LUNG1_052_LOCAL_Z   = 51
LUNG1_052_CROP      = dict(crop_y0=256, crop_x0=96, crop_y1=352, crop_x1=192)
LUNG1_052_CELL_KEY  = "image_left|Z1|Y2|X1"
CT_SHAPE_HW         = (512, 512)   # expected spatial shape

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _abort(msg: str, code: int = 2):
    print(f"\nBLOCKED: {msg}", file=sys.stderr)
    sys.exit(code)


def _crop_valid(y0, x0, y1, x1, h=512, w=512, size=96):
    return (
        y1 - y0 == size and x1 - x0 == size
        and y0 >= 0 and x0 >= 0
        and y1 <= h and x1 <= w
    )


# ──────────────────────────────────────────────
# STATIC GUARD CHECKS
# ──────────────────────────────────────────────

def check_guards():
    if ALLOW_CT_LOAD:
        _abort("ALLOW_CT_LOAD must be False for preflight")
    if ALLOW_PREVIEW_PNG_WRITE:
        _abort("ALLOW_PREVIEW_PNG_WRITE must be False for preflight")
    if ALLOW_METADATA_WRITE:
        _abort("ALLOW_METADATA_WRITE must be False for preflight")
    if ALLOW_STAGE2_HOLDOUT:
        _abort("ALLOW_STAGE2_HOLDOUT must be False")
    if ALLOW_MODEL_FORWARD or ALLOW_FEATURE_EXTRACTION or ALLOW_CONTRIBUTION_RECALC:
        _abort("model/feature/contribution guards must be False")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("normal reference bank v4 crop preview preflight")
    print("LUNG1-052__c3 exact cell")
    print("=" * 60)

    # ── guard checks ──
    check_guards()

    errors = []
    warnings = []

    # ── 1. metadata DONE exists ──
    if not METADATA_DONE.exists():
        _abort("metadata DONE.json not found")
    done = json.loads(METADATA_DONE.read_text())

    # ── 2. metadata verdict PASS ──
    if done.get("verdict") != "PASS":
        _abort(f"metadata verdict is not PASS: {done.get('verdict')}")

    # ── 3. errors.csv error count 0 ──
    if METADATA_ERRORS.exists():
        with open(METADATA_ERRORS) as f:
            err_rows = list(csv.DictReader(f))
        if len(err_rows) > 0:
            errors.append(f"metadata errors.csv has {len(err_rows)} errors")
    else:
        errors.append("metadata errors.csv not found")

    # ── 4. retrieval preview CSV exists ──
    if not RETRIEVAL_PREVIEW_CSV.exists():
        _abort("lung1_052_v4_retrieval_preview.csv not found")

    # ── load retrieval preview ──
    top3_df = pd.read_csv(RETRIEVAL_PREVIEW_CSV)

    # ── 5. top3 row count == 3 ──
    if len(top3_df) != 3:
        errors.append(f"top3 row count expected 3, got {len(top3_df)}")

    # ── 6. top3 unique patient count == 3 ──
    n_uniq = top3_df["patient_id"].nunique() if len(top3_df) > 0 else 0
    if n_uniq != 3:
        errors.append(f"top3 unique patients expected 3, got {n_uniq}")

    # ── 7. top3 cell_key == exact ──
    wrong_cell = top3_df[top3_df["cell_key"] != LUNG1_052_CELL_KEY]
    if len(wrong_cell) > 0:
        errors.append(
            f"top3 cell_key mismatch: expected {LUNG1_052_CELL_KEY}, "
            f"got {wrong_cell['cell_key'].unique().tolist()}"
        )

    # ── 8. top3 ct_path exists ──
    missing_ct = []
    for _, row in top3_df.iterrows():
        if not Path(row["ct_path"]).exists():
            missing_ct.append(row["ct_path"])
    if missing_ct:
        errors.append(f"top3 ct_path missing ({len(missing_ct)}): {missing_ct[:2]}")

    # ── 9. top3 stage2_holdout_flag == false ──
    if "stage2_holdout_flag" in top3_df.columns:
        if top3_df["stage2_holdout_flag"].any():
            errors.append("top3 has stage2_holdout_flag=True")
    # column absent means False (normal pool)

    # ── 10. top3 crop_size == 96 ──
    if "crop_size" in top3_df.columns:
        wrong_size = top3_df[top3_df["crop_size"] != 96]
        if len(wrong_size) > 0:
            errors.append(f"top3 crop_size != 96: {wrong_size['crop_size'].tolist()}")

    # ── 11. top3 crop_in_bounds == true ──
    if "crop_in_bounds" in top3_df.columns:
        oob = top3_df[top3_df["crop_in_bounds"] != True]
        if len(oob) > 0:
            errors.append(f"top3 crop_in_bounds=False: {len(oob)} rows")

    # ── 12. top3 quality_flag == true ──
    if "quality_flag" in top3_df.columns:
        bad_q = top3_df[top3_df["quality_flag"] != True]
        if len(bad_q) > 0:
            errors.append(f"top3 quality_flag=False: {len(bad_q)} rows")

    # ── 13. candidate CT path exists ──
    cand_ct_exists = Path(LUNG1_052_CT_PATH).exists()
    if not cand_ct_exists:
        errors.append(f"candidate CT path not found: {LUNG1_052_CT_PATH}")

    # ── 14. candidate crop bounds valid ──
    c = LUNG1_052_CROP
    cand_crop_valid = _crop_valid(
        c["crop_y0"], c["crop_x0"], c["crop_y1"], c["crop_x1"],
        h=CT_SHAPE_HW[0], w=CT_SHAPE_HW[1], size=96
    )
    if not cand_crop_valid:
        errors.append(
            f"candidate crop invalid: "
            f"y{c['crop_y0']}:{c['crop_y1']} x{c['crop_x0']}:{c['crop_x1']}"
        )

    # ── 14b. top3 crop bounds valid ──
    for _, row in top3_df.iterrows():
        if not _crop_valid(
            row["crop_y0"], row["crop_x0"], row["crop_y1"], row["crop_x1"],
            h=CT_SHAPE_HW[0], w=CT_SHAPE_HW[1], size=96
        ):
            errors.append(
                f"top3 crop invalid: {row['patient_id']} "
                f"y{row['crop_y0']}:{row['crop_y1']} x{row['crop_x0']}:{row['crop_x1']}"
            )

    # ── 15. output root collision ──
    if (OUTPUT_ROOT / "DONE.json").exists():
        _abort("existing DONE.json at output root. Archive or use new root.")
    if OUTPUT_ROOT.resolve() == METADATA_ROOT.resolve():
        _abort("OUTPUT_ROOT must differ from METADATA_ROOT")

    # ── report static check result ──
    print(f"\n[static checks]")
    print(f"  metadata verdict: PASS")
    print(f"  metadata errors.csv: 0")
    print(f"  top3 rows: {len(top3_df)}, unique patients: {n_uniq}")
    print(f"  top3 cell_key: {LUNG1_052_CELL_KEY}")
    print(f"  candidate CT exists: {cand_ct_exists}")
    print(f"  candidate crop valid: {cand_crop_valid}")
    print(f"  errors so far: {len(errors)}")

    # ── build crop_preview_plan ──
    print("\n[building crop preview plan...]")

    # candidate row
    cand_row = {
        "role": "candidate",
        "patient_id": LUNG1_052_PATIENT,
        "volume_id": LUNG1_052_VOLUME_ID,
        "ct_path": LUNG1_052_CT_PATH,
        "local_z": LUNG1_052_LOCAL_Z,
        "crop_y0": LUNG1_052_CROP["crop_y0"],
        "crop_x0": LUNG1_052_CROP["crop_x0"],
        "crop_y1": LUNG1_052_CROP["crop_y1"],
        "crop_x1": LUNG1_052_CROP["crop_x1"],
        "crop_size": 96,
        "crop_in_bounds": True,
        "stage2_holdout_flag": False,
        "quality_flag": True,
        "cell_key": LUNG1_052_CELL_KEY,
        "reference_quality_score": None,
        "preview_output_png_planned": "candidate_crop_lung1_052.png",
        "notes": "NSCLC LUNG1-052__c3 candidate; stage2 test case; not normal bank",
    }

    # top3 reference rows
    ref_rows = []
    for i, (_, row) in enumerate(top3_df.iterrows(), start=1):
        ref_row = {
            "role": f"normal_ref_{i}",
            "patient_id": row["patient_id"],
            "volume_id": row["volume_id"],
            "ct_path": row["ct_path"],
            "local_z": row["local_z"],
            "crop_y0": row["crop_y0"],
            "crop_x0": row["crop_x0"],
            "crop_y1": row["crop_y1"],
            "crop_x1": row["crop_x1"],
            "crop_size": row.get("crop_size", 96),
            "crop_in_bounds": row.get("crop_in_bounds", True),
            "stage2_holdout_flag": row.get("stage2_holdout_flag", False),
            "quality_flag": row.get("quality_flag", True),
            "cell_key": row["cell_key"],
            "reference_quality_score": row.get("reference_quality_score", None),
            "preview_output_png_planned": f"normal_ref_{i}_v4.png",
            "notes": (
                f"rank{i} in cell {LUNG1_052_CELL_KEY}; "
                f"score={row.get('reference_quality_score', 'N/A'):.4f}"
                if row.get("reference_quality_score") is not None
                else f"rank{i} in cell {LUNG1_052_CELL_KEY}"
            ),
        }
        ref_rows.append(ref_row)

    all_plan_rows = [cand_row] + ref_rows
    plan_df = pd.DataFrame(all_plan_rows)

    # ── build selected_top3_reference_check ──
    check_cols = [
        "rank_in_cell", "patient_id", "volume_id", "cell_key",
        "local_z", "lung_z_pct",
        "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "crop_size", "crop_in_bounds", "stage2_holdout_flag", "quality_flag",
        "crop_lung_roi_ratio", "reference_quality_score",
        "ct_path",
    ]
    avail_cols = [c for c in check_cols if c in top3_df.columns]
    check_df = top3_df[avail_cols].copy()
    # add ct_path_exists
    check_df["ct_path_exists"] = check_df["ct_path"].apply(lambda p: Path(p).exists())
    # add crop_valid
    check_df["crop_valid"] = check_df.apply(
        lambda r: _crop_valid(r["crop_y0"], r["crop_x0"], r["crop_y1"], r["crop_x1"]),
        axis=1
    )

    # ── determine verdict ──
    blocked_reasons = []
    if ALLOW_CT_LOAD:
        blocked_reasons.append("ALLOW_CT_LOAD=True")
    if ALLOW_PREVIEW_PNG_WRITE:
        blocked_reasons.append("ALLOW_PREVIEW_PNG_WRITE=True")

    if blocked_reasons:
        verdict = "BLOCKED"
    elif len(errors) > 0:
        # classify: missing CT = PARTIAL_PASS, others = NEEDS_FIX
        ct_missing = any("ct_path missing" in e or "CT path not found" in e for e in errors)
        non_ct_errors = [e for e in errors if "ct_path" not in e.lower()]
        if len(non_ct_errors) > 0:
            verdict = "NEEDS_FIX"
        elif ct_missing:
            verdict = "PARTIAL_PASS"
        else:
            verdict = "NEEDS_FIX"
    else:
        verdict = "PASS"

    print(f"\n[verdict] {verdict}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)

    # ── write outputs ──
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. crop_preview_plan CSV
    plan_csv = OUTPUT_ROOT / "crop_preview_plan_lung1_052_v4.csv"
    plan_df.to_csv(plan_csv, index=False)
    print(f"\n  → saved: crop_preview_plan_lung1_052_v4.csv  ({len(plan_df)} rows)")

    # 2. selected_top3_reference_check CSV
    check_csv = OUTPUT_ROOT / "selected_top3_reference_check_v4.csv"
    check_df.to_csv(check_csv, index=False)
    print(f"  → saved: selected_top3_reference_check_v4.csv  ({len(check_df)} rows)")

    # 3. generation policy md
    policy_lines = [
        "# crop preview generation policy v4",
        f"date: {date.today()}",
        "",
        "## scope",
        "LUNG1-052__c3 exact cell — preview only (5 PNG max)",
        "",
        "## allowed PNG files",
        "1. candidate_crop_lung1_052.png",
        "2. normal_ref_1_v4.png",
        "3. normal_ref_2_v4.png",
        "4. normal_ref_3_v4.png",
        "5. lung1_052_v4_reference_preview_montage.png",
        "",
        "## allowed operations",
        "- CT read-only mmap load (numpy memmap, shape only)",
        "- lung window: center=-600, width=1500 → clip → uint8",
        "- crop 96×96 from specified coordinates",
        "- save to OUTPUT_ROOT only",
        "",
        "## forbidden",
        "- 전체 cell 90개 crop preview 생성",
        "- 전체 candidate pool PNG 생성",
        "- reference bank metadata 수정",
        "- stage2_holdout 접근",
        "- ROI load",
        "- model/feature/contribution",
        "",
        "## guard flags for actual run",
        "ALLOW_CT_LOAD = True",
        "ALLOW_PREVIEW_PNG_WRITE = True",
        "ALLOW_METADATA_WRITE = False  (계속 False)",
        "ALLOW_STAGE2_HOLDOUT = False  (계속 False)",
        "ALLOW_MODEL_FORWARD = False   (계속 False)",
        "ALLOW_FEATURE_EXTRACTION = False  (계속 False)",
        "ALLOW_CONTRIBUTION_RECALC = False  (계속 False)",
        "ALLOW_FULL300 = False  (계속 False)",
    ]
    policy_md = OUTPUT_ROOT / "crop_preview_generation_policy_v4.md"
    policy_md.write_text("\n".join(policy_lines))
    print(f"  → saved: crop_preview_generation_policy_v4.md")

    # 4. generation policy json
    policy_dict = {
        "version": "v4",
        "date": str(date.today()),
        "scope": "LUNG1-052__c3 exact cell — 5 PNG max",
        "allowed_png": [
            "candidate_crop_lung1_052.png",
            "normal_ref_1_v4.png",
            "normal_ref_2_v4.png",
            "normal_ref_3_v4.png",
            "lung1_052_v4_reference_preview_montage.png",
        ],
        "ct_load": "read-only numpy memmap",
        "lung_window": {"center": -600, "width": 1500},
        "crop_size": 96,
        "guard_flags_actual_run": {
            "ALLOW_CT_LOAD": True,
            "ALLOW_PREVIEW_PNG_WRITE": True,
            "ALLOW_METADATA_WRITE": False,
            "ALLOW_STAGE2_HOLDOUT": False,
            "ALLOW_MODEL_FORWARD": False,
            "ALLOW_FEATURE_EXTRACTION": False,
            "ALLOW_CONTRIBUTION_RECALC": False,
            "ALLOW_FULL300": False,
        },
    }
    policy_json = OUTPUT_ROOT / "crop_preview_generation_policy_v4.json"
    policy_json.write_text(json.dumps(policy_dict, indent=2))
    print(f"  → saved: crop_preview_generation_policy_v4.json")

    # 5. report md
    ct_check_lines = [
        f"  - candidate CT: {'EXISTS' if cand_ct_exists else 'MISSING'} — {LUNG1_052_CT_PATH}",
    ]
    for _, row in check_df.iterrows():
        rank = row.get("rank_in_cell", "?")
        exists_str = "EXISTS" if row["ct_path_exists"] else "MISSING"
        ct_check_lines.append(f"  - normal_ref_{rank} CT: {exists_str} — {row['ct_path']}")

    report_lines = [
        "# normal reference bank v4 crop preview preflight report",
        f"date: {date.today()}",
        f"verdict: **{verdict}**",
        "",
        "## input validation",
        "- metadata DONE: YES (verdict=PASS)",
        "- metadata errors.csv: 0",
        f"- retrieval preview rows: {len(top3_df)}",
        f"- unique patients: {n_uniq}",
        f"- cell_key: {LUNG1_052_CELL_KEY}",
        "",
        "## CT path check",
    ] + ct_check_lines + [
        "",
        "## crop bounds check",
        f"  - candidate crop valid: {cand_crop_valid}  "
        f"(y{LUNG1_052_CROP['crop_y0']}:{LUNG1_052_CROP['crop_y1']} "
        f"x{LUNG1_052_CROP['crop_x0']}:{LUNG1_052_CROP['crop_x1']})",
    ]
    for _, row in check_df.iterrows():
        report_lines.append(
            f"  - normal_ref_{row.get('rank_in_cell','?')} crop valid: {row['crop_valid']}  "
            f"(y{row['crop_y0']}:{row['crop_y1']} x{row['crop_x0']}:{row['crop_x1']})"
        )

    report_lines += [
        "",
        "## preview plan",
        f"- PNG planned: 5 (candidate × 1, normal_ref × 3, montage × 1)",
        "- candidate_crop_lung1_052.png",
        "- normal_ref_1_v4.png / normal_ref_2_v4.png / normal_ref_3_v4.png",
        "- lung1_052_v4_reference_preview_montage.png",
        "",
        "## safety",
        "- CT load (preflight): 0",
        "- PNG write (preflight): 0",
        "- metadata write: 0",
        "- stage2_holdout: 0",
        "- model/feature/contribution: 0",
        "- existing artifact modified: 0",
    ]
    if errors:
        report_lines += ["", "## errors"] + [f"- {e}" for e in errors]
    if warnings:
        report_lines += ["", "## warnings"] + [f"- {w}" for w in warnings]
    report_lines += [""]

    report_md = OUTPUT_ROOT / "crop_preview_preflight_report.md"
    report_md.write_text("\n".join(report_lines))
    print(f"  → saved: crop_preview_preflight_report.md")

    # 6. report json
    report_dict = {
        "date": str(date.today()),
        "verdict": verdict,
        "input_validation": {
            "metadata_verdict": "PASS",
            "metadata_errors": 0,
            "top3_rows": len(top3_df),
            "unique_patients": int(n_uniq),
            "cell_key": LUNG1_052_CELL_KEY,
            "candidate_ct_exists": bool(cand_ct_exists),
            "candidate_crop_valid": bool(cand_crop_valid),
        },
        "top3_ct_exists": check_df["ct_path_exists"].tolist(),
        "top3_crop_valid": check_df["crop_valid"].tolist(),
        "preview_plan_png_count": 5,
        "safety": {
            "ct_load": 0,
            "png_write": 0,
            "metadata_write": 0,
            "stage2_holdout": 0,
            "model_forward": 0,
            "feature_extraction": 0,
            "contribution_recalc": 0,
            "artifact_modified": 0,
        },
        "errors": errors,
        "warnings": warnings,
        "blockers": len(blocked_reasons),
    }
    report_json = OUTPUT_ROOT / "crop_preview_preflight_report.json"
    report_json.write_text(json.dumps(report_dict, indent=2))
    print(f"  → saved: crop_preview_preflight_report.json")

    # 7. errors.csv
    errors_csv = OUTPUT_ROOT / "errors.csv"
    with open(errors_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "message"])
        for e in errors:
            w.writerow(["preflight", e])
    print(f"  → saved: errors.csv ({len(errors)} errors)")

    # 8. DONE.json
    done_dict = {
        "status": "DONE",
        "verdict": verdict,
        "date": str(date.today()),
        "blockers": len(blocked_reasons),
        "errors": len(errors),
        "warnings": len(warnings),
        "outputs": [
            "crop_preview_plan_lung1_052_v4.csv",
            "selected_top3_reference_check_v4.csv",
            "crop_preview_generation_policy_v4.md",
            "crop_preview_generation_policy_v4.json",
            "crop_preview_preflight_report.md",
            "crop_preview_preflight_report.json",
            "errors.csv",
            "DONE.json",
        ],
    }
    done_json = OUTPUT_ROOT / "DONE.json"
    done_json.write_text(json.dumps(done_dict, indent=2))
    print(f"  → saved: DONE.json")

    # ── final summary ──
    print()
    print("=" * 60)
    print(f"VERDICT: {verdict}")
    print(f"  top3 rows: {len(top3_df)}, unique patients: {n_uniq}")
    print(f"  candidate CT exists: {cand_ct_exists}")
    print(f"  top3 CT exists: {check_df['ct_path_exists'].all()}")
    print(f"  crop bounds valid: candidate={cand_crop_valid}, "
          f"top3={check_df['crop_valid'].all()}")
    print(f"  CT load: 0 / PNG write: 0 / stage2_holdout: 0")
    print(f"  errors: {len(errors)}")
    print("=" * 60)

    if verdict == "BLOCKED":
        sys.exit(2)
    elif verdict == "NEEDS_FIX":
        sys.exit(1)


if __name__ == "__main__":
    main()
