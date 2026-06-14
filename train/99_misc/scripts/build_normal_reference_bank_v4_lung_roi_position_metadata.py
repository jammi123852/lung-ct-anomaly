"""
normal reference bank v4 actual metadata generation
- preflight DONE/PASS 검증 후 quality candidate filtering
- cell별 ranking (unique patient 우선)
- metadata CSV/JSON, cell index, top3_by_cell, retrieval preview 생성
- CT/ROI load 금지, PNG 생성 금지, model/feature 금지
"""

import json
import os
import sys
import csv
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# GUARD FLAGS
# ──────────────────────────────────────────────
ALLOW_METADATA_WRITE       = True
ALLOW_CROP_PNG_WRITE       = False
ALLOW_CT_LOAD              = False
ALLOW_ROI_LOAD             = False
ALLOW_STAGE2_HOLDOUT       = False
ALLOW_MODEL_FORWARD        = False
ALLOW_FEATURE_EXTRACTION   = False
ALLOW_CONTRIBUTION_RECALC  = False
ALLOW_FULL300              = False

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
PREFLIGHT_ROOT = Path(
    "outputs/position-aware-padim-v1/reports/"
    "reference_bank_v4_lung_roi_position_preflight"
)
OUTPUT_ROOT = Path(
    "outputs/position-aware-padim-v1/reports/"
    "reference_bank_v4_lung_roi_position_metadata"
)

PREFLIGHT_DONE        = PREFLIGHT_ROOT / "DONE.json"
PREFLIGHT_ERRORS      = PREFLIGHT_ROOT / "errors.csv"
CANDIDATE_POOL_CSV    = PREFLIGHT_ROOT / "reference_bank_candidate_pool_v4.csv"
CELL_COVERAGE_CSV     = PREFLIGHT_ROOT / "cell_coverage_summary_v4.csv"
RETRIEVAL_POLICY_JSON = PREFLIGHT_ROOT / "retrieval_policy_v4.json"
LUNG1_052_PREVIEW_JSON = PREFLIGHT_ROOT / "candidate_lung1_052_cell_mapping_preview_v4.json"

# ──────────────────────────────────────────────
# QUALITY FILTER THRESHOLDS  (from retrieval_policy)
# ──────────────────────────────────────────────
QUALITY_CROP_SIZE      = 96
QUALITY_CROP_IN_BOUNDS = True
QUALITY_ROI_RATIO_MIN  = 0.15
QUALITY_ROI_AREA_MIN   = 300

# ──────────────────────────────────────────────
# STATIC PREFLIGHT CHECKS
# ──────────────────────────────────────────────

def _abort(msg: str, code: int = 2):
    print(f"\nBLOCKED: {msg}", file=sys.stderr)
    sys.exit(code)


def run_static_checks():
    errors = []

    # 1. preflight DONE exists
    if not PREFLIGHT_DONE.exists():
        _abort("preflight DONE.json not found")
    done = json.loads(PREFLIGHT_DONE.read_text())

    # 2. verdict PASS
    if done.get("verdict") != "PASS":
        _abort(f"preflight verdict is not PASS: {done.get('verdict')}")

    # 3. errors.csv error count == 0
    if PREFLIGHT_ERRORS.exists():
        with open(PREFLIGHT_ERRORS) as f:
            rows = list(csv.DictReader(f))
        if len(rows) > 0:
            errors.append(f"preflight errors.csv has {len(rows)} errors")
    else:
        errors.append("preflight errors.csv not found")

    # 4. candidate pool exists
    if not CANDIDATE_POOL_CSV.exists():
        _abort("reference_bank_candidate_pool_v4.csv not found")

    # 5. cell coverage exists
    if not CELL_COVERAGE_CSV.exists():
        _abort("cell_coverage_summary_v4.csv not found")

    # 6. retrieval policy exists
    if not RETRIEVAL_POLICY_JSON.exists():
        _abort("retrieval_policy_v4.json not found")
    policy = json.loads(RETRIEVAL_POLICY_JSON.read_text())

    # 7. PASS_TOP3 90/90
    cov_df = pd.read_csv(CELL_COVERAGE_CSV)
    pass_top3 = (cov_df["coverage_status"] == "PASS_TOP3").sum()
    total_cells = len(cov_df)
    if pass_top3 != 90 or total_cells != 90:
        errors.append(f"PASS_TOP3 expected 90/90, got {pass_top3}/{total_cells}")

    # 8. LUNG1-052 top3_available
    if LUNG1_052_PREVIEW_JSON.exists():
        preview = json.loads(LUNG1_052_PREVIEW_JSON.read_text())
        if not preview.get("top3_available"):
            errors.append("LUNG1-052 top3_available is not true")
    else:
        errors.append("candidate_lung1_052_cell_mapping_preview_v4.json not found")

    # 9. output collision check
    if (OUTPUT_ROOT / "DONE.json").exists():
        _abort(
            "existing DONE.json found at output root. "
            "Archive or use a new output root."
        )

    # 10. output root separate from preflight root
    if OUTPUT_ROOT.resolve() == PREFLIGHT_ROOT.resolve():
        _abort("OUTPUT_ROOT must differ from PREFLIGHT_ROOT")

    # 11. guard: PNG write
    if ALLOW_CROP_PNG_WRITE:
        _abort("ALLOW_CROP_PNG_WRITE must be False for this stage")

    # 12. guard: CT/ROI load
    if ALLOW_CT_LOAD:
        _abort("ALLOW_CT_LOAD must be False for this stage")
    if ALLOW_ROI_LOAD:
        _abort("ALLOW_ROI_LOAD must be False for this stage")

    # 13. guard: stage2 holdout
    if ALLOW_STAGE2_HOLDOUT:
        _abort("ALLOW_STAGE2_HOLDOUT must be False for this stage")

    # 14. guard: model/feature
    if ALLOW_MODEL_FORWARD or ALLOW_FEATURE_EXTRACTION or ALLOW_CONTRIBUTION_RECALC:
        _abort("model/feature/contribution guards must be False for this stage")

    if errors:
        for e in errors:
            print(f"  CHECK FAIL: {e}", file=sys.stderr)
        _abort(f"{len(errors)} static check(s) failed")

    print("[static checks] ALL PASS")
    return policy, cov_df, preview if LUNG1_052_PREVIEW_JSON.exists() else {}


# ──────────────────────────────────────────────
# QUALITY SCORE FORMULA
# ──────────────────────────────────────────────

def compute_quality_score(df: pd.DataFrame) -> pd.Series:
    """
    reference_quality_score = 0.60 * crop_lung_roi_ratio
                            + 0.20 * roi_area_normalized
                            + 0.10 * center_validity
                            + 0.10 * cell_center_balance
    center_validity   = 1.0 (always, after quality filter crop_in_bounds==True)
    roi_area_normalized = min-max normalised roi_area_at_slice over quality pool
    cell_center_balance = 1 - mean(|y_pct - cell_y_center|, |x_pct - cell_x_center|)
    """
    roi_min = df["roi_area_at_slice"].min()
    roi_max = df["roi_area_at_slice"].max()
    roi_norm = (df["roi_area_at_slice"] - roi_min) / max(roi_max - roi_min, 1e-6)

    center_validity = pd.Series(1.0, index=df.index)

    # cell center in pct: Y0→0.167, Y1→0.5, Y2→0.833; same for X
    bin_center = {"Y0": 0.167, "Y1": 0.5, "Y2": 0.833,
                  "X0": 0.167, "X1": 0.5, "X2": 0.833}
    y_cell_center = df["y_bin_3"].map(bin_center)
    x_cell_center = df["x_bin_3"].map(bin_center)
    y_dev = (df["y_pct_in_lung_bbox"] - y_cell_center).abs()
    x_dev = (df["x_pct_in_lung_bbox"] - x_cell_center).abs()
    cell_center_balance = 1.0 - (y_dev + x_dev) / 2.0

    score = (
        0.60 * df["crop_lung_roi_ratio"]
        + 0.20 * roi_norm
        + 0.10 * center_validity
        + 0.10 * cell_center_balance
    )
    return score.clip(0.0, 1.0)


# ──────────────────────────────────────────────
# TOP3 UNIQUE PATIENT SELECTION
# ──────────────────────────────────────────────

def select_top_unique_patients(cell_df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    cell 내에서 unique patient top-N 선발:
    1. reference_quality_score 내림차순 정렬
    2. 환자당 가장 높은 score 1개만 유지
    3. 상위 N개 선택
    """
    best_per_patient = (
        cell_df
        .sort_values("reference_quality_score", ascending=False)
        .drop_duplicates(subset=["patient_id"], keep="first")
    )
    return best_per_patient.head(n).reset_index(drop=True)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("normal reference bank v4 metadata generation")
    print("=" * 60)

    # ── static checks ──
    policy, cov_df, lung1052_preview = run_static_checks()

    # ── load candidate pool ──
    print("\n[1/7] Loading candidate pool...")
    pool = pd.read_csv(CANDIDATE_POOL_CSV)
    print(f"  loaded: {len(pool):,} rows")

    # ── quality filter ──
    print("\n[2/7] Applying quality filter...")
    q = pool[
        (pool["crop_size"] == QUALITY_CROP_SIZE) &
        (pool["crop_in_bounds"] == QUALITY_CROP_IN_BOUNDS) &
        (pool["crop_lung_roi_ratio"] >= QUALITY_ROI_RATIO_MIN) &
        (pool["roi_area_at_slice"] >= QUALITY_ROI_AREA_MIN) &
        (pool["quality_flag"] == True)
    ].copy()
    print(f"  quality candidates: {len(q):,}")

    if len(q) == 0:
        _abort("quality candidates == 0 after filter")

    # ── validate guard columns ──
    # stage2_holdout_flag: not in pool → add as False
    if "stage2_holdout_flag" not in q.columns:
        q["stage2_holdout_flag"] = False
    else:
        if q["stage2_holdout_flag"].any():
            _abort("stage2_holdout_flag=True found in quality candidates")

    # add source_preflight
    q["source_preflight"] = str(PREFLIGHT_ROOT)

    # validate crop_size only 96
    if set(q["crop_size"].unique()) != {96}:
        _abort(f"crop_size values other than 96 found: {q['crop_size'].unique()}")

    # ── compute quality score ──
    print("\n[3/7] Computing reference_quality_score...")
    q["reference_quality_score"] = compute_quality_score(q)

    # ── build metadata CSV ──
    META_COLS = [
        "reference_id", "patient_id", "volume_id", "image_lung_side",
        "local_z", "lung_z_pct", "z_bin_5",
        "y_center", "x_center",
        "y_pct_in_lung_bbox", "x_pct_in_lung_bbox",
        "y_bin_3", "x_bin_3",
        "cell_key",
        "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "crop_size", "crop_in_bounds", "crop_lung_roi_ratio",
        "roi_area_at_slice", "quality_flag",
        "ct_path", "roi_path",
        "source_preflight", "stage2_holdout_flag",
        "reference_quality_score",
    ]
    # cell_key: compose if missing
    if "cell_key" not in q.columns:
        q["cell_key"] = (
            q["image_lung_side"] + "|" + q["z_bin_5"] + "|"
            + q["y_bin_3"] + "|" + q["x_bin_3"]
        )

    meta_df = q[META_COLS].copy()

    # ── build cell index ──
    print("\n[4/7] Building cell index...")
    cell_stats = []
    for cell_key, cell_df in q.groupby("cell_key"):
        parts = cell_key.split("|")
        n_refs = len(cell_df)
        n_uniq = cell_df["patient_id"].nunique()
        cell_stats.append({
            "cell_key": cell_key,
            "image_lung_side": parts[0],
            "z_bin_5": parts[1],
            "y_bin_3": parts[2],
            "x_bin_3": parts[3],
            "n_references": n_refs,
            "n_unique_patients": n_uniq,
            "top3_available": n_uniq >= 3,
            "top5_available": n_uniq >= 5,
            "coverage_status": "PASS_TOP3" if n_uniq >= 3 else (
                "PARTIAL" if n_uniq >= 1 else "EMPTY"
            ),
        })

    cell_index_df = pd.DataFrame(cell_stats)
    top3_cells = (cell_index_df["top3_available"] == True).sum()
    top5_cells = (cell_index_df["top5_available"] == True).sum()
    total_cells_idx = len(cell_index_df)
    print(f"  cells with top3: {top3_cells}/{total_cells_idx}")
    print(f"  cells with top5: {top5_cells}/{total_cells_idx}")

    # ── build top3_by_cell ──
    print("\n[5/7] Building top3_by_cell...")
    TOP3_COLS = [
        "cell_key", "rank_in_cell", "reference_id", "patient_id", "volume_id",
        "local_z", "lung_z_pct", "y_center", "x_center",
        "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "crop_lung_roi_ratio", "reference_quality_score",
        "ct_path", "roi_path",
    ]
    top3_rows = []
    for cell_key, cell_df in q.groupby("cell_key"):
        selected = select_top_unique_patients(cell_df, n=3)
        for rank_idx, row in selected.iterrows():
            entry = {col: row[col] for col in TOP3_COLS if col != "cell_key" and col != "rank_in_cell"}
            entry["cell_key"] = cell_key
            entry["rank_in_cell"] = rank_idx + 1
            top3_rows.append(entry)

    top3_df = pd.DataFrame(top3_rows)[TOP3_COLS]
    print(f"  top3 entries total: {len(top3_df):,}")

    # ── LUNG1-052 retrieval preview ──
    print("\n[6/7] Generating LUNG1-052 retrieval preview...")
    lung1052_cell = lung1052_preview.get("cell_key", "image_left|Z1|Y2|X1")
    if lung1052_cell in q["cell_key"].values:
        lung1052_cell_df = q[q["cell_key"] == lung1052_cell].copy()
        lung1052_top3 = select_top_unique_patients(lung1052_cell_df, n=3)
        lung1052_top3["rank_in_cell"] = range(1, len(lung1052_top3) + 1)

        # verify preview conditions
        preview_ok = True
        if len(lung1052_top3) < 3:
            print(f"  WARNING: LUNG1-052 top3 only {len(lung1052_top3)} candidates")
            preview_ok = False
        if lung1052_top3["patient_id"].nunique() < len(lung1052_top3):
            print("  WARNING: LUNG1-052 top3 has duplicate patients")
            preview_ok = False
        if lung1052_top3["stage2_holdout_flag"].any():
            print("  WARNING: LUNG1-052 top3 has stage2_holdout_flag=True")
            preview_ok = False
        if not (lung1052_top3["crop_in_bounds"] == True).all():
            print("  WARNING: LUNG1-052 top3 has crop_in_bounds=False")
            preview_ok = False
        if not (lung1052_top3["quality_flag"] == True).all():
            print("  WARNING: LUNG1-052 top3 has quality_flag=False")
            preview_ok = False

        preview_verdict = "PASS" if preview_ok else "WARNING"
        print(f"  LUNG1-052 retrieval preview: {preview_verdict}")
        print(f"  cell: {lung1052_cell}, n_top3={len(lung1052_top3)}")
        for _, r in lung1052_top3.iterrows():
            print(f"    rank{r['rank_in_cell']}: {r['patient_id']}  score={r['reference_quality_score']:.4f}")
    else:
        print(f"  WARNING: cell {lung1052_cell} not found in quality candidates")
        lung1052_top3 = pd.DataFrame()
        preview_verdict = "WARNING"

    # ── write outputs ──
    if not ALLOW_METADATA_WRITE:
        _abort("ALLOW_METADATA_WRITE=False — dry-run mode, no files written")

    print("\n[7/7] Writing outputs...")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. metadata CSV
    out_meta_csv = OUTPUT_ROOT / "normal_reference_bank_v4_metadata.csv"
    meta_df.to_csv(out_meta_csv, index=False)
    print(f"  → saved: normal_reference_bank_v4_metadata.csv  ({len(meta_df):,} rows)")

    # 2. metadata JSON (records orient, limit memory: write in chunks via csv path)
    out_meta_json = OUTPUT_ROOT / "normal_reference_bank_v4_metadata.json"
    meta_df.to_json(out_meta_json, orient="records", indent=2)
    print(f"  → saved: normal_reference_bank_v4_metadata.json")

    # 3. cell index CSV
    out_ci_csv = OUTPUT_ROOT / "normal_reference_bank_v4_cell_index.csv"
    cell_index_df.to_csv(out_ci_csv, index=False)
    print(f"  → saved: normal_reference_bank_v4_cell_index.csv  ({len(cell_index_df)} cells)")

    # 4. cell index JSON
    out_ci_json = OUTPUT_ROOT / "normal_reference_bank_v4_cell_index.json"
    cell_index_df.to_json(out_ci_json, orient="records", indent=2)
    print(f"  → saved: normal_reference_bank_v4_cell_index.json")

    # 5. top3_by_cell CSV
    out_top3_csv = OUTPUT_ROOT / "normal_reference_bank_v4_top3_by_cell.csv"
    top3_df.to_csv(out_top3_csv, index=False)
    print(f"  → saved: normal_reference_bank_v4_top3_by_cell.csv  ({len(top3_df):,} entries)")

    # 6. retrieval policy frozen
    out_policy = OUTPUT_ROOT / "normal_reference_bank_v4_retrieval_policy_frozen.json"
    policy_frozen = dict(policy)
    policy_frozen["frozen_at"] = str(date.today())
    policy_frozen["source_preflight"] = str(PREFLIGHT_ROOT)
    policy_frozen["quality_score_formula"] = (
        "0.60*crop_lung_roi_ratio + 0.20*roi_area_normalized "
        "+ 0.10*center_validity + 0.10*cell_center_balance"
    )
    out_policy.write_text(json.dumps(policy_frozen, indent=2))
    print(f"  → saved: normal_reference_bank_v4_retrieval_policy_frozen.json")

    # 7. LUNG1-052 retrieval preview CSV
    out_lung1052 = OUTPUT_ROOT / "lung1_052_v4_retrieval_preview.csv"
    if not lung1052_top3.empty:
        lung1052_top3.to_csv(out_lung1052, index=False)
    else:
        pd.DataFrame().to_csv(out_lung1052, index=False)
    print(f"  → saved: lung1_052_v4_retrieval_preview.csv")

    # ── determine verdict ──
    needs_fix_reasons = []
    blocked_reasons = []

    # duplicate patient check in top3
    dup_cells = top3_df.groupby("cell_key")["patient_id"].apply(
        lambda x: x.nunique() < len(x)
    )
    if dup_cells.any():
        needs_fix_reasons.append(
            f"duplicate patients in top3: {dup_cells.sum()} cells"
        )

    # stage2_holdout must be False
    if meta_df["stage2_holdout_flag"].any():
        blocked_reasons.append("stage2_holdout_flag=True found in metadata")

    # crop_size must be 96 only
    if set(meta_df["crop_size"].unique()) != {96}:
        needs_fix_reasons.append(
            f"crop_size values: {set(meta_df['crop_size'].unique())}"
        )

    # PNG/CT/ROI guard confirmation
    if ALLOW_CROP_PNG_WRITE:
        blocked_reasons.append("PNG guard violated")
    if ALLOW_CT_LOAD or ALLOW_ROI_LOAD:
        blocked_reasons.append("CT/ROI load guard violated")

    if blocked_reasons:
        verdict = "BLOCKED"
    elif needs_fix_reasons:
        verdict = "NEEDS_FIX"
    elif top3_cells < 90 or preview_verdict == "WARNING":
        verdict = "PARTIAL_PASS"
    else:
        verdict = "PASS"

    # 8. report md
    report_lines = [
        "# normal reference bank v4 metadata generation report",
        f"date: {date.today()}",
        f"verdict: **{verdict}**",
        "",
        "## input validation",
        f"- preflight DONE: YES (verdict=PASS)",
        f"- preflight errors: 0",
        f"- candidate pool rows: {len(pool):,}",
        f"- quality candidates: {len(q):,}",
        f"- cells: {total_cells_idx}/90",
        f"- PASS_TOP3 cells: {top3_cells}/{total_cells_idx}",
        "",
        "## output files",
        f"- normal_reference_bank_v4_metadata.csv: {len(meta_df):,} rows",
        f"- normal_reference_bank_v4_cell_index.csv: {total_cells_idx} cells",
        f"- normal_reference_bank_v4_top3_by_cell.csv: {len(top3_df):,} entries",
        f"- LUNG1-052 retrieval preview: {preview_verdict}",
        "",
        "## quality score formula",
        "reference_quality_score =",
        "  0.60 * crop_lung_roi_ratio",
        "  + 0.20 * roi_area_normalized  (min-max over quality pool)",
        "  + 0.10 * center_validity  (1.0; all crop_in_bounds=True)",
        "  + 0.10 * cell_center_balance  (1 - mean absolute deviation from bin center)",
        "",
        "## safety",
        "- PNG write: 0",
        "- CT load: 0",
        "- ROI load: 0",
        "- stage2_holdout access: 0",
        "- model/feature/contribution: 0",
        f"- existing artifact modified: 0",
    ]
    if blocked_reasons:
        report_lines += ["", "## BLOCKED"] + [f"- {r}" for r in blocked_reasons]
    if needs_fix_reasons:
        report_lines += ["", "## NEEDS_FIX"] + [f"- {r}" for r in needs_fix_reasons]
    report_lines += [""]

    out_report_md = OUTPUT_ROOT / "metadata_generation_report.md"
    out_report_md.write_text("\n".join(report_lines))
    print(f"  → saved: metadata_generation_report.md")

    # 9. report json
    report_dict = {
        "date": str(date.today()),
        "verdict": verdict,
        "input_validation": {
            "preflight_verdict": "PASS",
            "preflight_errors": 0,
            "candidate_pool_rows": len(pool),
            "quality_candidates": len(q),
            "cells": total_cells_idx,
            "top3_cells": int(top3_cells),
            "top5_cells": int(top5_cells),
            "lung1052_preview": preview_verdict,
        },
        "outputs": {
            "metadata_rows": len(meta_df),
            "cell_index_rows": total_cells_idx,
            "top3_rows": len(top3_df),
        },
        "safety": {
            "png_write": 0,
            "ct_load": 0,
            "roi_load": 0,
            "stage2_holdout": 0,
            "model_forward": 0,
            "feature_extraction": 0,
            "contribution_recalc": 0,
            "artifact_modified": 0,
        },
        "blockers": len(blocked_reasons),
        "warnings": len(needs_fix_reasons),
        "blocked_reasons": blocked_reasons,
        "fix_reasons": needs_fix_reasons,
    }
    out_report_json = OUTPUT_ROOT / "metadata_generation_report.json"
    out_report_json.write_text(json.dumps(report_dict, indent=2))
    print(f"  → saved: metadata_generation_report.json")

    # 10. errors.csv
    out_errors = OUTPUT_ROOT / "errors.csv"
    with open(out_errors, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "message"])
    print(f"  → saved: errors.csv (0 errors)")

    # 11. DONE.json
    done_dict = {
        "status": "DONE",
        "verdict": verdict,
        "date": str(date.today()),
        "blockers": len(blocked_reasons),
        "warnings": len(needs_fix_reasons),
        "outputs": [
            "normal_reference_bank_v4_metadata.csv",
            "normal_reference_bank_v4_metadata.json",
            "normal_reference_bank_v4_cell_index.csv",
            "normal_reference_bank_v4_cell_index.json",
            "normal_reference_bank_v4_top3_by_cell.csv",
            "normal_reference_bank_v4_retrieval_policy_frozen.json",
            "lung1_052_v4_retrieval_preview.csv",
            "metadata_generation_report.md",
            "metadata_generation_report.json",
            "errors.csv",
            "DONE.json",
        ],
    }
    out_done = OUTPUT_ROOT / "DONE.json"
    out_done.write_text(json.dumps(done_dict, indent=2))
    print(f"  → saved: DONE.json")

    # ── final summary ──
    print()
    print("=" * 60)
    print(f"VERDICT: {verdict}")
    print(f"  quality candidates: {len(q):,}")
    print(f"  cells top3: {top3_cells}/{total_cells_idx}")
    print(f"  top5 cells: {top5_cells}/{total_cells_idx}")
    print(f"  LUNG1-052 preview: {preview_verdict}")
    print(f"  PNG write: 0 / CT load: 0 / ROI load: 0")
    print(f"  stage2_holdout: 0 / model/feature: 0")
    print("=" * 60)

    if verdict == "BLOCKED":
        sys.exit(2)
    elif verdict == "NEEDS_FIX":
        sys.exit(1)


if __name__ == "__main__":
    main()
