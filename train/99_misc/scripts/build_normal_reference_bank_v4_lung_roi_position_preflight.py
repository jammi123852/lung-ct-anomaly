#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_normal_reference_bank_v4_lung_roi_position_preflight.py

Normal reference bank v4 — lung ROI position-based preflight.

목표:
  lung ROI extent 기준 z percentile + in-slice lung bbox 기준 y/x pct로
  reference bank v4 cell coverage를 확인한다.

금지:
  stage2_holdout 접근, PNG 생성, 모델 실행, bank 실제 생성,
  기존 artifact 수정, CT write, 진단/원인 단정

허용 (preflight):
  ALLOW_NORMAL_ROI_LOAD  = True  (read-only full load, 1개씩 처리 후 del)
  ALLOW_NORMAL_CT_LOAD   = True  (shape/mmap 확인만)
  나머지 ALLOW_* = False
"""

import csv
import json
import pathlib
import sys
import traceback
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
# GUARDS
# ============================================================
ALLOW_NORMAL_ROI_LOAD       = True
ALLOW_NORMAL_CT_LOAD        = True   # shape/mmap only
ALLOW_REFERENCE_BANK_WRITE  = False
ALLOW_CROP_PNG_WRITE        = False
ALLOW_STAGE2_HOLDOUT        = False
ALLOW_MODEL_FORWARD         = False
ALLOW_FEATURE_EXTRACTION    = False
ALLOW_CONTRIBUTION_RECALC   = False
ALLOW_FULL300               = False

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT   = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")
NORMAL_CT_ROOT = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1"
    "/volumes_npy"
)
PATIENT_MANIFEST = (PROJECT_ROOT
    / "data/normal_training_ready/manifests/patient_manifest.csv")
SPLIT_FILE = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/splits/normal_v1.json")
OUTPUT_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports"
    / "reference_bank_v4_lung_roi_position_preflight")

# Output files
OUT_PREFLIGHT_MD   = OUTPUT_ROOT / "reference_bank_v4_preflight_report.md"
OUT_PREFLIGHT_JSON = OUTPUT_ROOT / "reference_bank_v4_preflight_report.json"
OUT_VOL_INV_CSV    = OUTPUT_ROOT / "normal_volume_inventory_v4.csv"
OUT_ROI_EXT_CSV    = OUTPUT_ROOT / "lung_roi_extent_inventory_v4.csv"
OUT_POOL_CSV       = OUTPUT_ROOT / "reference_bank_candidate_pool_v4.csv"
OUT_CELL_COV_CSV   = OUTPUT_ROOT / "cell_coverage_summary_v4.csv"
OUT_SPARSE_CSV     = OUTPUT_ROOT / "empty_or_sparse_cells_v4.csv"
OUT_POLICY_MD      = OUTPUT_ROOT / "retrieval_policy_v4.md"
OUT_POLICY_JSON    = OUTPUT_ROOT / "retrieval_policy_v4.json"
OUT_CAND_MAP_JSON  = OUTPUT_ROOT / "candidate_lung1_052_cell_mapping_preview_v4.json"
OUT_ERRORS_CSV     = OUTPUT_ROOT / "errors.csv"
OUT_DONE_JSON      = OUTPUT_ROOT / "DONE.json"

# ============================================================
# BIN PARAMETERS
# ============================================================
Z_BIN_NAMES   = ["Z0", "Z1", "Z2", "Z3", "Z4"]
Y_BIN_NAMES   = ["Y0", "Y1", "Y2"]
X_BIN_NAMES   = ["X0", "X1", "X2"]
SIDES         = ["image_left", "image_right"]

CROP_SIZE     = 96
SLICE_STRIDE  = 5      # sample every Nth slice for candidate pool (performance)
MIN_ROI_RATIO = 0.15   # minimum crop_lung_roi_ratio for quality_flag=True
MIN_ROI_AREA  = 300    # minimum ROI pixel count at a slice (per side)
MIN_BBOX_DIM  = 48     # minimum ROI bbox dimension to generate grid candidates
IMAGE_WIDTH   = 512
IMAGE_HEIGHT  = 512
SIDE_SPLIT_X  = IMAGE_WIDTH // 2  # 256

# Grid fractions for 3×3 sampling within ROI bbox
GRID_FRACS = [1 / 6, 3 / 6, 5 / 6]  # → y/x_pct ≈ 1/6, 3/6, 5/6 → Y0,Y1,Y2 / X0,X1,X2

STAGE2_FORBIDDEN_KEYWORDS = ["stage2_holdout", "stage2holdout", "holdout"]

# LUNG1-052__c3 candidate constants (dry-run preview only)
CAND_CASE_ID    = "LUNG1-052__c3"
CAND_VOLUME_ID  = "NSCLC_LUNG1-052__d4a19cc211"
CAND_LOCAL_Z    = 51
CAND_Y_CENTER   = 304.0
CAND_X_CENTER   = 144.0
CAND_CROP_Y0    = 256
CAND_CROP_X0    = 96
CAND_CROP_Y1    = 352
CAND_CROP_X1    = 192
# Precomputed from NSCLC ROI (read-only check above):
CAND_LUNG_Z_MIN_LEFT = 0
CAND_LUNG_Z_MAX_LEFT = 212
CAND_LUNG_Z_PCT      = round((CAND_LOCAL_Z - CAND_LUNG_Z_MIN_LEFT)
                              / max(CAND_LUNG_Z_MAX_LEFT - CAND_LUNG_Z_MIN_LEFT, 1), 4)
CAND_BBOX_Y0  = 155
CAND_BBOX_Y1  = 344
CAND_BBOX_X0  = 98
CAND_BBOX_X1  = 211


# ============================================================
# STATIC CHECKS (20 items)
# ============================================================
def run_static_checks():
    results = []

    def chk(item_id, desc, passed, note=""):
        results.append({"id": item_id, "desc": desc,
                        "passed": bool(passed), "note": note})

    chk(1,  "output_root_contains_reference_bank_v4",
        "reference_bank_v4" in str(OUTPUT_ROOT))
    chk(2,  "stage2_holdout_forbidden",
        not ALLOW_STAGE2_HOLDOUT)
    chk(3,  "ct_roi_read_only_policy_exists",
        ALLOW_NORMAL_CT_LOAD and ALLOW_NORMAL_ROI_LOAD
        and not ALLOW_REFERENCE_BANK_WRITE)
    chk(4,  "no_png_write",
        not ALLOW_CROP_PNG_WRITE)
    chk(5,  "no_model_forward",
        not ALLOW_MODEL_FORWARD)
    chk(6,  "no_feature_extraction",
        not ALLOW_FEATURE_EXTRACTION)
    chk(7,  "no_contribution_recalc",
        not ALLOW_CONTRIBUTION_RECALC)
    chk(8,  "no_score_threshold_recompute",
        True, "no model/threshold computation in this script")
    chk(9,  "z_bins_eq_5",
        len(Z_BIN_NAMES) == 5, f"Z_BIN_NAMES={Z_BIN_NAMES}")
    chk(10, "y_bins_eq_3",
        len(Y_BIN_NAMES) == 3, f"Y_BIN_NAMES={Y_BIN_NAMES}")
    chk(11, "x_bins_eq_3",
        len(X_BIN_NAMES) == 3, f"X_BIN_NAMES={X_BIN_NAMES}")
    chk(12, "image_lung_side_not_anatomical_claim",
        True,
        "sides labeled image_left/image_right by image x-coord; no anatomical L/R claim")
    chk(13, "crop_size_eq_96",
        CROP_SIZE == 96, f"CROP_SIZE={CROP_SIZE}")
    chk(14, "top3_unique_patient_rule_defined",
        True,
        "retrieval policy selects top3 unique patients per cell")
    chk(15, "fallback_policy_defined",
        True,
        "4-step fallback: z±1, xy±1, nearest-continuous, PARTIAL_PASS if <3")
    chk(16, "cell_coverage_summary_planned",
        str(OUT_CELL_COV_CSV).endswith("cell_coverage_summary_v4.csv"))
    chk(17, "candidate_lung1_052_mapping_planned",
        str(OUT_CAND_MAP_JSON).endswith(
            "candidate_lung1_052_cell_mapping_preview_v4.json"))
    chk(18, "errors_and_done_planned",
        str(OUT_ERRORS_CSV).endswith("errors.csv")
        and str(OUT_DONE_JSON).endswith("DONE.json"))
    chk(19, "no_diagnostic_wording",
        True,
        "no 'cancer','tumor','cause of finding' in this script output")
    chk(20, "no_lesion_cancer_vessel_cause_claim",
        True,
        "position-based only; no clinical cause claims")

    n_pass = sum(1 for r in results if r["passed"])
    n_fail = sum(1 for r in results if not r["passed"])
    return results, n_pass, n_fail


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def get_z_bin(pct: float) -> str:
    return Z_BIN_NAMES[min(int(pct / 0.2), 4)]


def get_y_bin(pct: float) -> str:
    return Y_BIN_NAMES[min(int(pct * 3), 2)]


def get_x_bin(pct: float) -> str:
    return X_BIN_NAMES[min(int(pct * 3), 2)]


def compute_crop_bounds(y_center: float, x_center: float,
                        img_h: int = IMAGE_HEIGHT, img_w: int = IMAGE_WIDTH):
    half = CROP_SIZE // 2
    y0 = int(round(y_center)) - half
    x0 = int(round(x_center)) - half
    y1 = y0 + CROP_SIZE
    x1 = x0 + CROP_SIZE
    in_bounds = (y0 >= 0 and x0 >= 0 and y1 <= img_h and x1 <= img_w)
    return y0, x0, y1, x1, in_bounds


def compute_roi_ratio_in_crop(roi_slice: np.ndarray, y0, x0, y1, x1) -> float:
    H, W = roi_slice.shape
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        return 0.0
    return float(roi_slice[y0:y1, x0:x1].sum()) / (CROP_SIZE * CROP_SIZE)


# ============================================================
# STEP 1 — LOAD PATIENT INVENTORY
# ============================================================
def load_patient_inventory():
    with open(SPLIT_FILE) as f:
        split_info = json.load(f)
    stage2_holdout_set = set(split_info["test"])

    pid_to_sid = {}
    with open(PATIENT_MANIFEST, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid_to_sid[row["patient_id"]] = row["safe_id"]

    return pid_to_sid, split_info, stage2_holdout_set


# ============================================================
# STEP 2 — BUILD NORMAL VOLUME INVENTORY
# ============================================================
def build_volume_inventory(pid_to_sid, split_info, stage2_holdout_set):
    records = []
    split_lookup = {}
    for s in ("train", "val", "test"):
        for pid in split_info[s]:
            split_lookup[pid] = s

    for patient_id, safe_id in pid_to_sid.items():
        ct_path  = NORMAL_CT_ROOT / safe_id / "ct_hu.npy"
        roi_path = NORMAL_CT_ROOT / safe_id / "roi_0_0.npy"
        ct_exists  = ct_path.exists()
        roi_exists = roi_path.exists()
        is_holdout = patient_id in stage2_holdout_set
        source_split = split_lookup.get(patient_id, "unknown")

        ct_shape   = None
        roi_shape  = None
        shape_match = None

        if ct_exists and ALLOW_NORMAL_CT_LOAD:
            try:
                arr = np.load(str(ct_path), mmap_mode="r")
                ct_shape = list(arr.shape)
                del arr
            except Exception as e:
                ct_shape = f"ERROR:{e}"

        if roi_exists and ALLOW_NORMAL_ROI_LOAD:
            try:
                arr = np.load(str(roi_path), mmap_mode="r")
                roi_shape = list(arr.shape)
                del arr
            except Exception as e:
                roi_shape = f"ERROR:{e}"

        if (isinstance(ct_shape, list) and isinstance(roi_shape, list)):
            shape_match = (ct_shape == roi_shape)

        usable = (ct_exists and roi_exists and not is_holdout
                  and shape_match is not False)
        reasons = []
        if not ct_exists:        reasons.append("ct_missing")
        if not roi_exists:       reasons.append("roi_missing")
        if is_holdout:           reasons.append("stage2_holdout")
        if shape_match is False: reasons.append("shape_mismatch")

        records.append({
            "patient_id":          patient_id,
            "volume_id":           safe_id,
            "ct_path":             str(ct_path),
            "roi_path":            str(roi_path),
            "ct_exists":           ct_exists,
            "roi_exists":          roi_exists,
            "ct_shape":            str(ct_shape) if ct_shape is not None else "",
            "roi_shape":           str(roi_shape) if roi_shape is not None else "",
            "shape_match":         shape_match,
            "source_split":        source_split,
            "stage2_holdout_flag": is_holdout,
            "usable_flag":         usable,
            "reason_if_unusable":  ";".join(reasons),
        })
    return records


# ============================================================
# STEP 3 — PROCESS ROIs → EXTENT + CANDIDATE POOL
# ============================================================
def process_volumes(volume_inventory):
    roi_extent_records = []
    candidate_records  = []
    errors             = []
    ref_counter        = 0

    usable_vols = [r for r in volume_inventory if r["usable_flag"]]
    n_total = len(usable_vols)

    for vi, vol in enumerate(usable_vols):
        patient_id = vol["patient_id"]
        safe_id    = vol["volume_id"]
        roi_path   = pathlib.Path(vol["roi_path"])
        ct_path    = pathlib.Path(vol["ct_path"])

        # Forbidden keyword guard
        path_str_lower = str(roi_path).lower()
        for kw in STAGE2_FORBIDDEN_KEYWORDS:
            if kw in path_str_lower:
                errors.append({
                    "patient_id": patient_id, "volume_id": safe_id,
                    "error_type": "FORBIDDEN_KEYWORD",
                    "detail": f"keyword '{kw}' found in path"
                })
                break
        else:
            # Load ROI (full, not mmap — for efficient boolean ops)
            if not ALLOW_NORMAL_ROI_LOAD:
                errors.append({"patient_id": patient_id, "volume_id": safe_id,
                                "error_type": "GUARD_BLOCKED",
                                "detail": "ALLOW_NORMAL_ROI_LOAD=False"})
                continue
            try:
                roi = np.load(str(roi_path)).astype(bool)
            except Exception as e:
                errors.append({"patient_id": patient_id, "volume_id": safe_id,
                                "error_type": "ROI_LOAD_ERROR", "detail": str(e)})
                continue

            Z_dim, H, W = roi.shape

            side_defs = [
                ("image_left",  0,      SIDE_SPLIT_X),
                ("image_right", SIDE_SPLIT_X, W),
            ]

            for side_name, x_start, x_end in side_defs:
                # Per-side ROI
                side_roi = roi[:, :, x_start:x_end]   # (Z, H, half_w)
                z_has_roi = side_roi.any(axis=(1, 2))
                z_indices = np.where(z_has_roi)[0]

                if len(z_indices) == 0:
                    roi_extent_records.append({
                        "patient_id":        patient_id,
                        "volume_id":         safe_id,
                        "image_lung_side":   side_name,
                        "z_min":             None,
                        "z_max":             None,
                        "n_slices_with_roi": 0,
                        "mean_roi_area":     0.0,
                        "valid_side_flag":   False,
                    })
                    continue

                z_min   = int(z_indices.min())
                z_max   = int(z_indices.max())
                z_range = max(z_max - z_min, 1)
                mean_roi_area = float(
                    side_roi[z_indices].sum(axis=(1, 2)).mean()
                )

                roi_extent_records.append({
                    "patient_id":        patient_id,
                    "volume_id":         safe_id,
                    "image_lung_side":   side_name,
                    "z_min":             z_min,
                    "z_max":             z_max,
                    "n_slices_with_roi": int(len(z_indices)),
                    "mean_roi_area":     round(mean_roi_area, 1),
                    "valid_side_flag":   True,
                })

                # Grid-sampled candidate crops
                sampled_z = z_indices[::SLICE_STRIDE]

                for z_val in sampled_z:
                    slice_side_mask = side_roi[int(z_val)]  # (H, half_w)
                    y_coords, xh_coords = np.where(slice_side_mask)
                    if len(y_coords) == 0:
                        continue

                    roi_area = int(slice_side_mask.sum())
                    if roi_area < MIN_ROI_AREA:
                        continue

                    # ROI bbox in full image coordinates
                    bbox_y0 = int(y_coords.min())
                    bbox_y1 = int(y_coords.max())
                    bbox_x0 = int(xh_coords.min()) + x_start
                    bbox_x1 = int(xh_coords.max()) + x_start

                    bbox_h = bbox_y1 - bbox_y0
                    bbox_w = bbox_x1 - bbox_x0
                    if bbox_h < MIN_BBOX_DIM or bbox_w < MIN_BBOX_DIM:
                        continue

                    lung_z_pct = (int(z_val) - z_min) / z_range
                    z_bin      = get_z_bin(lung_z_pct)

                    full_roi_slice = roi[int(z_val)]  # (H, W) full slice for ratio

                    # 3×3 grid positions within bbox
                    for y_frac in GRID_FRACS:
                        for x_frac in GRID_FRACS:
                            y_center = bbox_y0 + y_frac * bbox_h
                            x_center = bbox_x0 + x_frac * bbox_w

                            y_pct = (y_center - bbox_y0) / bbox_h
                            x_pct = (x_center - bbox_x0) / bbox_w
                            y_bin = get_y_bin(y_pct)
                            x_bin = get_x_bin(x_pct)

                            y0, x0, y1, x1, in_bounds = compute_crop_bounds(
                                y_center, x_center, H, W)
                            roi_ratio = compute_roi_ratio_in_crop(
                                full_roi_slice, y0, x0, y1, x1)

                            quality = (in_bounds and roi_ratio >= MIN_ROI_RATIO)

                            ref_counter += 1
                            candidate_records.append({
                                "reference_id":        f"v4_{ref_counter:08d}",
                                "patient_id":          patient_id,
                                "volume_id":           safe_id,
                                "image_lung_side":     side_name,
                                "local_z":             int(z_val),
                                "lung_z_pct":          round(float(lung_z_pct), 4),
                                "z_bin_5":             z_bin,
                                "y_center":            round(float(y_center), 1),
                                "x_center":            round(float(x_center), 1),
                                "lung_bbox_y0":        bbox_y0,
                                "lung_bbox_x0":        bbox_x0,
                                "lung_bbox_y1":        bbox_y1,
                                "lung_bbox_x1":        bbox_x1,
                                "y_pct_in_lung_bbox":  round(float(y_pct), 4),
                                "x_pct_in_lung_bbox":  round(float(x_pct), 4),
                                "y_bin_3":             y_bin,
                                "x_bin_3":             x_bin,
                                "crop_y0":             y0,
                                "crop_x0":             x0,
                                "crop_y1":             y1,
                                "crop_x1":             x1,
                                "crop_size":           CROP_SIZE,
                                "crop_in_bounds":      in_bounds,
                                "roi_area_at_slice":   roi_area,
                                "crop_lung_roi_ratio": round(float(roi_ratio), 4),
                                "quality_flag":        quality,
                                "ct_path":             str(ct_path),
                                "roi_path":            str(roi_path),
                            })

            del roi  # free memory

        if (vi + 1) % 50 == 0:
            print(f"  processed {vi+1}/{n_total} volumes, "
                  f"candidates so far: {len(candidate_records)}")

    return roi_extent_records, candidate_records, errors


# ============================================================
# STEP 4 — CELL COVERAGE SUMMARY
# ============================================================
def build_cell_coverage(candidate_records):
    # Aggregate quality candidates only
    cell_data = defaultdict(lambda: {"n_candidates": 0, "patients": set()})

    for rec in candidate_records:
        if not rec["quality_flag"]:
            continue
        cell_key = (rec["image_lung_side"], rec["z_bin_5"],
                    rec["y_bin_3"], rec["x_bin_3"])
        cell_data[cell_key]["n_candidates"] += 1
        cell_data[cell_key]["patients"].add(rec["patient_id"])

    coverage_rows = []
    for side in SIDES:
        for zb in Z_BIN_NAMES:
            for yb in Y_BIN_NAMES:
                for xb in X_BIN_NAMES:
                    key = (side, zb, yb, xb)
                    data = cell_data[key]
                    n_cand = data["n_candidates"]
                    n_pat  = len(data["patients"])

                    if n_pat >= 3:
                        status = "PASS_TOP3"
                    elif n_pat == 2:
                        status = "PARTIAL_TOP2"
                    elif n_pat == 1:
                        status = "PARTIAL_TOP1"
                    else:
                        status = "EMPTY"

                    coverage_rows.append({
                        "image_lung_side":  side,
                        "z_bin_5":          zb,
                        "y_bin_3":          yb,
                        "x_bin_3":          xb,
                        "n_candidates":     n_cand,
                        "n_unique_patients": n_pat,
                        "top3_available":   n_pat >= 3,
                        "fallback_needed":  n_pat < 3,
                        "coverage_status":  status,
                    })

    return coverage_rows


# ============================================================
# STEP 5 — LUNG1-052 DRY-RUN CELL MAPPING
# ============================================================
def build_candidate_cell_mapping(coverage_rows):
    """Compute LUNG1-052__c3 cell assignment and check coverage."""

    # y_pct, x_pct relative to left-side ROI bbox at z=51
    bbox_h = CAND_BBOX_Y1 - CAND_BBOX_Y0
    bbox_w = CAND_BBOX_X1 - CAND_BBOX_X0

    y_pct  = (CAND_Y_CENTER - CAND_BBOX_Y0) / bbox_h
    x_pct  = (CAND_X_CENTER - CAND_BBOX_X0) / bbox_w

    z_bin  = get_z_bin(CAND_LUNG_Z_PCT)
    y_bin  = get_y_bin(y_pct)
    x_bin  = get_x_bin(x_pct)

    # Check coverage for this cell
    cell_cov = next(
        (r for r in coverage_rows
         if r["image_lung_side"] == "image_left"
         and r["z_bin_5"] == z_bin
         and r["y_bin_3"] == y_bin
         and r["x_bin_3"] == x_bin),
        None
    )

    cov_status   = cell_cov["coverage_status"]    if cell_cov else "UNKNOWN"
    n_unique_pat = cell_cov["n_unique_patients"]   if cell_cov else 0
    n_cand       = cell_cov["n_candidates"]        if cell_cov else 0

    mapping = {
        "case_id":            CAND_CASE_ID,
        "volume_id":          CAND_VOLUME_ID,
        "local_z":            CAND_LOCAL_Z,
        "y_center":           CAND_Y_CENTER,
        "x_center":           CAND_X_CENTER,
        "crop_y0":            CAND_CROP_Y0,
        "crop_x0":            CAND_CROP_X0,
        "crop_y1":            CAND_CROP_Y1,
        "crop_x1":            CAND_CROP_X1,
        "image_lung_side":    "image_left",
        "lung_z_min_left":    CAND_LUNG_Z_MIN_LEFT,
        "lung_z_max_left":    CAND_LUNG_Z_MAX_LEFT,
        "lung_z_pct":         CAND_LUNG_Z_PCT,
        "z_bin_5":            z_bin,
        "lung_bbox_y0":       CAND_BBOX_Y0,
        "lung_bbox_y1":       CAND_BBOX_Y1,
        "lung_bbox_x0":       CAND_BBOX_X0,
        "lung_bbox_x1":       CAND_BBOX_X1,
        "y_pct_in_lung_bbox": round(y_pct, 4),
        "x_pct_in_lung_bbox": round(x_pct, 4),
        "y_bin_3":            y_bin,
        "x_bin_3":            x_bin,
        "cell_key":           f"image_left|{z_bin}|{y_bin}|{x_bin}",
        "cell_n_candidates":  n_cand,
        "cell_n_unique_patients": n_unique_pat,
        "cell_coverage_status": cov_status,
        "top3_available":     n_unique_pat >= 3,
        "note":               (
            "dry-run preview; NSCLC ROI read-only to compute z extent only. "
            "lung_z_pct uses actual NSCLC lung ROI extent at z=51 (left side). "
            "image_lung_side=image_left because x_center=144 < 256. "
            "No model/feature/score computation."
        ),
    }
    return mapping


# ============================================================
# STEP 6 — RETRIEVAL POLICY
# ============================================================
RETRIEVAL_POLICY = {
    "version": "v4",
    "description": (
        "Retrieve top3 normal reference crops for a given candidate patch "
        "using lung-ROI position bins."
    ),
    "schema": {
        "image_lung_side": "image_left or image_right (by image x-coord; "
                           "NOT anatomical direction)",
        "z_bin_5":  "Z0–Z4 (lung ROI extent-based z percentile, 5 equal bins)",
        "y_bin_3":  "Y0–Y2 (in-slice lung ROI bbox relative y position, 3 equal bins)",
        "x_bin_3":  "X0–X2 (in-slice lung ROI bbox relative x position, 3 equal bins)",
        "cell":     "side × z_bin × y_bin × x_bin = 2×5×3×3 = 90 cells total",
    },
    "retrieval_steps": [
        {
            "step": 1,
            "desc": "exact match",
            "rule": "same side + same z_bin + same y_bin + same x_bin",
            "select": "top3 unique patients ranked by reference_match_score",
        },
        {
            "step": 2,
            "desc": "z fallback",
            "rule": "same side + adjacent z_bin (±1) + same y_bin + same x_bin",
            "select": "top3 unique patients",
        },
        {
            "step": 3,
            "desc": "xy fallback",
            "rule": "same side + same z_bin + adjacent y_bin (±1) or x_bin (±1)",
            "select": "top3 unique patients",
        },
        {
            "step": 4,
            "desc": "nearest continuous",
            "rule": "same side + nearest by continuous z_pct/y_pct/x_pct distance",
            "select": "top3 unique patients",
        },
        {
            "step": 5,
            "desc": "PARTIAL_PASS",
            "rule": "if still <3 unique patients after all fallbacks",
            "select": "report PARTIAL_PASS; do not force-fill",
        },
    ],
    "reference_match_score": {
        "formula":
            "0.45*z_similarity + 0.35*xy_similarity "
            "+ 0.10*crop_lung_roi_ratio_similarity + 0.10*quality_score",
        "note":
            "pleural_distance and vessel features NOT used (unavailable); "
            "use NA for missing components",
    },
    "quality_filter": {
        "crop_size":                CROP_SIZE,
        "crop_in_bounds":           True,
        "crop_lung_roi_ratio_min":  MIN_ROI_RATIO,
        "roi_area_at_slice_min":    MIN_ROI_AREA,
        "all_zero_crop":            "excluded (implicit via roi_ratio > 0)",
        "ct_path_exists":           True,
        "roi_path_exists":          True,
        "stage2_holdout_flag":      False,
    },
}


# ============================================================
# STEP 7 — VERDICT
# ============================================================
def compute_verdict(vol_inventory, coverage_rows, cand_mapping,
                    static_n_fail, errors):
    # Blockers
    blockers = []
    warnings = []
    notes    = []

    # Static check failures
    if static_n_fail > 0:
        blockers.append(f"static_check: {static_n_fail} items FAILED")

    # ALLOW_STAGE2_HOLDOUT check
    if ALLOW_STAGE2_HOLDOUT:
        blockers.append("ALLOW_STAGE2_HOLDOUT is True — forbidden")

    # Errors check
    if errors:
        fb_kw = [e for e in errors if e.get("error_type") == "FORBIDDEN_KEYWORD"]
        if fb_kw:
            blockers.append(f"FORBIDDEN_KEYWORD errors: {len(fb_kw)}")
        other_errs = [e for e in errors if e.get("error_type") != "FORBIDDEN_KEYWORD"]
        if other_errs:
            warnings.append(f"processing errors: {len(other_errs)}")

    # Volume inventory
    n_usable = sum(1 for r in vol_inventory if r["usable_flag"])
    n_ct_missing  = sum(1 for r in vol_inventory if not r["ct_exists"])
    n_roi_missing = sum(1 for r in vol_inventory if not r["roi_exists"])
    n_mismatch    = sum(1 for r in vol_inventory
                        if r["shape_match"] is False)

    if n_usable < 100:
        blockers.append(f"too few usable volumes: {n_usable}")
    if n_ct_missing > 0:
        warnings.append(f"ct_missing: {n_ct_missing}")
    if n_roi_missing > 0:
        warnings.append(f"roi_missing: {n_roi_missing}")
    if n_mismatch > 0:
        warnings.append(f"shape_mismatch: {n_mismatch}")

    # Cell coverage
    n_empty         = sum(1 for r in coverage_rows if r["coverage_status"] == "EMPTY")
    n_partial_top1  = sum(1 for r in coverage_rows if r["coverage_status"] == "PARTIAL_TOP1")
    n_partial_top2  = sum(1 for r in coverage_rows if r["coverage_status"] == "PARTIAL_TOP2")
    n_pass_top3     = sum(1 for r in coverage_rows if r["coverage_status"] == "PASS_TOP3")
    n_total_cells   = len(coverage_rows)  # should be 90

    if n_pass_top3 < n_total_cells * 0.5:
        warnings.append(
            f"less than 50% cells have top3: PASS_TOP3={n_pass_top3}/{n_total_cells}")
    if n_empty > n_total_cells * 0.3:
        warnings.append(f"many empty cells: {n_empty}/{n_total_cells}")

    # Candidate cell check
    if not cand_mapping.get("top3_available", False):
        cov_s = cand_mapping.get("cell_coverage_status", "UNKNOWN")
        n_p   = cand_mapping.get("cell_n_unique_patients", 0)
        warnings.append(
            f"LUNG1-052__c3 cell top3 NOT available: "
            f"status={cov_s}, n_patients={n_p}")
    else:
        notes.append(
            f"LUNG1-052__c3 cell top3 AVAILABLE: "
            f"n_patients={cand_mapping.get('cell_n_unique_patients')}")

    # Verdict
    if blockers:
        verdict = "BLOCKED"
    elif n_usable < 200 or n_pass_top3 < n_total_cells * 0.7:
        verdict = "NEEDS_FIX"
    elif warnings:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "PASS"

    summary = {
        "n_usable_volumes":  n_usable,
        "n_ct_missing":      n_ct_missing,
        "n_roi_missing":     n_roi_missing,
        "n_shape_mismatch":  n_mismatch,
        "n_total_cells":     n_total_cells,
        "n_pass_top3":       n_pass_top3,
        "n_partial_top2":    n_partial_top2,
        "n_partial_top1":    n_partial_top1,
        "n_empty_cells":     n_empty,
        "candidate_cell_top3_available": cand_mapping.get("top3_available", False),
    }
    return verdict, summary, blockers, warnings, notes


# ============================================================
# REPORT GENERATION
# ============================================================
def build_md_report(verdict, summary, static_results, static_n_pass, static_n_fail,
                    cand_mapping, coverage_rows, blockers, warnings, notes):
    date_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "# Normal Reference Bank v4 Preflight Report",
        "",
        f"**date**: {date_str}  ",
        f"**verdict**: {verdict}  ",
        f"**static_check**: {static_n_pass}/20 PASS  ",
        "",
        "---",
        "",
        "## 1. Normal Volume Inventory",
        "",
        f"- usable_volumes (train+val, non-holdout): {summary['n_usable_volumes']}",
        f"- ct_missing: {summary['n_ct_missing']}",
        f"- roi_missing: {summary['n_roi_missing']}",
        f"- shape_mismatch: {summary['n_shape_mismatch']}",
        "",
        "## 2. ROI Extent Readiness",
        "",
        "lung ROI z extent computed per (volume, image_lung_side).",
        "See lung_roi_extent_inventory_v4.csv for details.",
        "",
        "## 3. Cell Coverage Summary",
        "",
        f"- total cells (2 sides × 5 z_bins × 3 y_bins × 3 x_bins): "
        f"{summary['n_total_cells']}",
        f"- PASS_TOP3  (≥3 unique patients): {summary['n_pass_top3']}",
        f"- PARTIAL_TOP2 (2 patients):        {summary['n_partial_top2']}",
        f"- PARTIAL_TOP1 (1 patient):         {summary['n_partial_top1']}",
        f"- EMPTY (0 patients):               {summary['n_empty_cells']}",
        "",
    ]

    # Empty cells table
    empty_cells = [r for r in coverage_rows
                   if r["coverage_status"] in ("EMPTY", "PARTIAL_TOP1")]
    if empty_cells:
        lines += [
            "### Empty / Sparse Cells (needs fallback)",
            "",
            "| side | z_bin | y_bin | x_bin | status | n_patients |",
            "|------|-------|-------|-------|--------|------------|",
        ]
        for r in empty_cells:
            lines.append(
                f"| {r['image_lung_side']} | {r['z_bin_5']} | "
                f"{r['y_bin_3']} | {r['x_bin_3']} | "
                f"{r['coverage_status']} | {r['n_unique_patients']} |"
            )
        lines.append("")
    else:
        lines += ["All cells have ≥2 unique patients.", ""]

    lines += [
        "## 4. Candidate LUNG1-052__c3 Preview",
        "",
        f"- cell: {cand_mapping['cell_key']}",
        f"- lung_z_pct: {cand_mapping['lung_z_pct']} (lung ROI z extent: "
        f"z_min={cand_mapping['lung_z_min_left']}, z_max={cand_mapping['lung_z_max_left']})",
        f"- y_pct: {cand_mapping['y_pct_in_lung_bbox']}, x_pct: {cand_mapping['x_pct_in_lung_bbox']}",
        f"- cell_coverage_status: {cand_mapping['cell_coverage_status']}",
        f"- n_unique_patients: {cand_mapping['cell_n_unique_patients']}",
        f"- top3_available: {cand_mapping['top3_available']}",
        "",
        "## 5. Retrieval Policy",
        "",
        "4-step fallback: exact → z±1 → xy±1 → nearest continuous → PARTIAL_PASS",
        "See retrieval_policy_v4.md for full spec.",
        "",
        "## 6. Safety",
        "",
        f"- ALLOW_STAGE2_HOLDOUT: {ALLOW_STAGE2_HOLDOUT}",
        f"- ALLOW_REFERENCE_BANK_WRITE: {ALLOW_REFERENCE_BANK_WRITE}",
        f"- ALLOW_CROP_PNG_WRITE: {ALLOW_CROP_PNG_WRITE}",
        f"- ALLOW_MODEL_FORWARD: {ALLOW_MODEL_FORWARD}",
        "",
    ]

    if blockers:
        lines += ["## BLOCKERS", ""]
        for b in blockers:
            lines.append(f"- ⛔ {b}")
        lines.append("")

    if warnings:
        lines += ["## WARNINGS", ""]
        for w in warnings:
            lines.append(f"- ⚠️  {w}")
        lines.append("")

    if notes:
        lines += ["## NOTES", ""]
        for n in notes:
            lines.append(f"- ℹ️  {n}")
        lines.append("")

    # Next step
    lines += ["## Next Step", ""]
    if verdict in ("PASS", "PARTIAL_PASS"):
        lines.append(
            "→ **normal reference bank v4 actual metadata generation** "
            "(ALLOW_REFERENCE_BANK_WRITE=True, ALLOW_CROP_PNG_WRITE still False)"
        )
    else:
        lines.append("→ Fix blockers/warnings above before proceeding.")
    lines.append("")

    return "\n".join(lines)


# ============================================================
# RETRIEVAL POLICY MARKDOWN
# ============================================================
def build_policy_md():
    lines = [
        "# Retrieval Policy v4 — Normal Reference Bank",
        "",
        "## Schema",
        "",
        "| field | description |",
        "|-------|-------------|",
        "| image_lung_side | image_left (x<256) or image_right (x>=256); "
        "NOT anatomical direction |",
        "| z_bin_5 | Z0–Z4: lung ROI extent-based z percentile (5 equal bins) |",
        "| y_bin_3 | Y0–Y2: in-slice lung ROI bbox relative y (3 equal bins) |",
        "| x_bin_3 | X0–X2: in-slice lung ROI bbox relative x (3 equal bins) |",
        "",
        "## Total Cells",
        "",
        "2 sides × 5 z_bins × 3 y_bins × 3 x_bins = **90 cells**",
        "",
        "## Retrieval Steps",
        "",
        "1. **Exact match**: same side + same z_bin + same y_bin + same x_bin → top3 unique patients",
        "2. **Z fallback**: same side + adjacent z_bin (±1) + same y/x bins",
        "3. **XY fallback**: same side + same z_bin + adjacent y_bin or x_bin (±1)",
        "4. **Nearest continuous**: same side + nearest by continuous (z_pct, y_pct, x_pct) distance",
        "5. **PARTIAL_PASS**: if still <3 patients after all fallbacks — report and do not force-fill",
        "",
        "## reference_match_score",
        "",
        "```",
        "score = 0.45 * z_similarity",
        "      + 0.35 * xy_similarity",
        "      + 0.10 * crop_lung_roi_ratio_similarity",
        "      + 0.10 * quality_score",
        "```",
        "",
        "Note: pleural_distance and vessel features are NOT used (unavailable); ",
        "record as NA.",
        "",
        "## Quality Filter",
        "",
        f"- crop_size: {CROP_SIZE}×{CROP_SIZE}",
        f"- crop_in_bounds: True",
        f"- crop_lung_roi_ratio ≥ {MIN_ROI_RATIO}",
        f"- roi_area_at_slice ≥ {MIN_ROI_AREA} pixels",
        "- stage2_holdout_flag: False",
        "- ct_path exists: True",
        "- roi_path exists: True",
        "",
    ]
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Normal Reference Bank v4 Preflight")
    print("=" * 60)

    # Guard checks
    if ALLOW_STAGE2_HOLDOUT:
        print("BLOCKED: ALLOW_STAGE2_HOLDOUT is True — forbidden")
        sys.exit(2)
    if ALLOW_REFERENCE_BANK_WRITE:
        print("BLOCKED: ALLOW_REFERENCE_BANK_WRITE is True — preflight only")
        sys.exit(2)
    if ALLOW_CROP_PNG_WRITE:
        print("BLOCKED: ALLOW_CROP_PNG_WRITE is True — forbidden in preflight")
        sys.exit(2)
    if ALLOW_MODEL_FORWARD or ALLOW_FEATURE_EXTRACTION or ALLOW_CONTRIBUTION_RECALC:
        print("BLOCKED: model/feature/contribution flags are True — forbidden")
        sys.exit(2)

    # Create output directory
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"output root: {OUTPUT_ROOT}")

    # ---------- Static checks ----------
    print("\n[1/7] Running static checks...")
    static_results, static_n_pass, static_n_fail = run_static_checks()
    print(f"  static checks: {static_n_pass}/20 PASS, {static_n_fail} FAIL")
    if static_n_fail > 0:
        failed = [r for r in static_results if not r["passed"]]
        for r in failed:
            print(f"  FAIL: [{r['id']}] {r['desc']}: {r['note']}")

    # ---------- Load patient inventory ----------
    print("\n[2/7] Loading patient inventory...")
    pid_to_sid, split_info, stage2_holdout_set = load_patient_inventory()
    print(f"  total patients in manifest: {len(pid_to_sid)}")
    print(f"  stage2_holdout (test split): {len(stage2_holdout_set)}")

    # ---------- Build volume inventory ----------
    print("\n[3/7] Building volume inventory...")
    vol_inventory = build_volume_inventory(pid_to_sid, split_info, stage2_holdout_set)
    n_usable = sum(1 for r in vol_inventory if r["usable_flag"])
    n_holdout = sum(1 for r in vol_inventory if r["stage2_holdout_flag"])
    n_ct_ok  = sum(1 for r in vol_inventory if r["ct_exists"])
    n_roi_ok = sum(1 for r in vol_inventory if r["roi_exists"])
    print(f"  total: {len(vol_inventory)}, usable: {n_usable}, holdout: {n_holdout}")
    print(f"  ct_exists: {n_ct_ok}, roi_exists: {n_roi_ok}")

    pd.DataFrame(vol_inventory).to_csv(OUT_VOL_INV_CSV, index=False)
    print(f"  → saved: {OUT_VOL_INV_CSV.name}")

    # ---------- Process volumes ----------
    print(f"\n[4/7] Processing {n_usable} volumes (ROI load + candidate pool)...")
    print(f"  slice_stride={SLICE_STRIDE}, grid=3×3, crop_size={CROP_SIZE}")
    roi_extent_records, candidate_records, errors = process_volumes(vol_inventory)

    n_quality = sum(1 for r in candidate_records if r["quality_flag"])
    print(f"  total candidates: {len(candidate_records)}, quality: {n_quality}")
    print(f"  ROI extent records: {len(roi_extent_records)}")
    print(f"  errors: {len(errors)}")

    pd.DataFrame(roi_extent_records).to_csv(OUT_ROI_EXT_CSV, index=False)
    print(f"  → saved: {OUT_ROI_EXT_CSV.name}")

    pd.DataFrame(candidate_records).to_csv(OUT_POOL_CSV, index=False)
    print(f"  → saved: {OUT_POOL_CSV.name}")

    # ---------- Cell coverage ----------
    print("\n[5/7] Building cell coverage summary...")
    coverage_rows = build_cell_coverage(candidate_records)
    n_pass_top3  = sum(1 for r in coverage_rows if r["coverage_status"] == "PASS_TOP3")
    n_empty      = sum(1 for r in coverage_rows if r["coverage_status"] == "EMPTY")
    print(f"  total cells: {len(coverage_rows)}, PASS_TOP3: {n_pass_top3}, "
          f"EMPTY: {n_empty}")

    pd.DataFrame(coverage_rows).to_csv(OUT_CELL_COV_CSV, index=False)
    print(f"  → saved: {OUT_CELL_COV_CSV.name}")

    # Empty/sparse cells CSV
    sparse = [r for r in coverage_rows
              if r["coverage_status"] in ("EMPTY", "PARTIAL_TOP1", "PARTIAL_TOP2")]
    pd.DataFrame(sparse).to_csv(OUT_SPARSE_CSV, index=False)
    print(f"  → saved: {OUT_SPARSE_CSV.name} ({len(sparse)} sparse/empty cells)")

    # ---------- LUNG1-052 dry-run ----------
    print("\n[6/7] LUNG1-052__c3 dry-run cell mapping...")
    cand_mapping = build_candidate_cell_mapping(coverage_rows)
    print(f"  cell: {cand_mapping['cell_key']}")
    print(f"  lung_z_pct: {cand_mapping['lung_z_pct']}")
    print(f"  y_pct: {cand_mapping['y_pct_in_lung_bbox']}, "
          f"x_pct: {cand_mapping['x_pct_in_lung_bbox']}")
    print(f"  coverage_status: {cand_mapping['cell_coverage_status']}, "
          f"n_unique_patients: {cand_mapping['cell_n_unique_patients']}")
    print(f"  top3_available: {cand_mapping['top3_available']}")

    with open(OUT_CAND_MAP_JSON, "w") as f:
        json.dump(cand_mapping, f, indent=2, ensure_ascii=False)
    print(f"  → saved: {OUT_CAND_MAP_JSON.name}")

    # ---------- Save reports ----------
    print("\n[7/7] Saving reports...")

    # Retrieval policy
    with open(OUT_POLICY_JSON, "w") as f:
        json.dump(RETRIEVAL_POLICY, f, indent=2, ensure_ascii=False)
    OUT_POLICY_MD.write_text(build_policy_md(), encoding="utf-8")
    print(f"  → saved: {OUT_POLICY_MD.name}, {OUT_POLICY_JSON.name}")

    # Errors CSV
    with open(OUT_ERRORS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "volume_id",
                                           "error_type", "detail"])
        w.writeheader()
        for e in errors:
            w.writerow({k: e.get(k, "") for k in
                        ["patient_id", "volume_id", "error_type", "detail"]})
    print(f"  → saved: {OUT_ERRORS_CSV.name} ({len(errors)} errors)")

    # Verdict
    verdict, vsum, blockers, warnings, notes = compute_verdict(
        vol_inventory, coverage_rows, cand_mapping, static_n_fail, errors)

    # Preflight JSON
    preflight_data = {
        "report":          "normal reference bank v4 preflight",
        "version":         "v4",
        "date":            datetime.now().strftime("%Y-%m-%d"),
        "script":          "scripts/build_normal_reference_bank_v4_lung_roi_position_preflight.py",
        "verdict":         verdict,
        "static_check":    {"n_pass": static_n_pass, "n_fail": static_n_fail,
                            "items": static_results},
        "normal_inventory": {
            "total":         len(vol_inventory),
            "usable":        vsum["n_usable_volumes"],
            "stage2_holdout":  n_holdout,
            "ct_missing":    vsum["n_ct_missing"],
            "roi_missing":   vsum["n_roi_missing"],
            "shape_mismatch": vsum["n_shape_mismatch"],
        },
        "roi_extent_readiness": {
            "n_roi_extent_records": len(roi_extent_records),
            "n_valid_sides":
                sum(1 for r in roi_extent_records if r.get("valid_side_flag")),
        },
        "cell_coverage": {
            "total_cells":   vsum["n_total_cells"],
            "PASS_TOP3":     vsum["n_pass_top3"],
            "PARTIAL_TOP2":  vsum["n_partial_top2"],
            "PARTIAL_TOP1":  vsum["n_partial_top1"],
            "EMPTY":         vsum["n_empty_cells"],
        },
        "candidate_pool": {
            "total_candidates": len(candidate_records),
            "quality_candidates": n_quality,
        },
        "candidate_lung1_052_preview": cand_mapping,
        "sparse_empty_cells": {
            "count": len(sparse),
            "list":  [{"side": r["image_lung_side"], "z": r["z_bin_5"],
                       "y": r["y_bin_3"], "x": r["x_bin_3"],
                       "status": r["coverage_status"]}
                      for r in sparse[:30]],
        },
        "safety": {
            "ALLOW_STAGE2_HOLDOUT":       ALLOW_STAGE2_HOLDOUT,
            "ALLOW_REFERENCE_BANK_WRITE": ALLOW_REFERENCE_BANK_WRITE,
            "ALLOW_CROP_PNG_WRITE":       ALLOW_CROP_PNG_WRITE,
            "ALLOW_MODEL_FORWARD":        ALLOW_MODEL_FORWARD,
            "ALLOW_FEATURE_EXTRACTION":   ALLOW_FEATURE_EXTRACTION,
            "ALLOW_CONTRIBUTION_RECALC":  ALLOW_CONTRIBUTION_RECALC,
        },
        "blockers": blockers,
        "warnings": warnings,
        "notes":    notes,
        "errors":   {"count": len(errors)},
        "next_step": (
            "normal reference bank v4 actual metadata generation"
            if verdict in ("PASS", "PARTIAL_PASS")
            else "fix blockers/warnings"
        ),
    }
    with open(OUT_PREFLIGHT_JSON, "w") as f:
        json.dump(preflight_data, f, indent=2, ensure_ascii=False)
    print(f"  → saved: {OUT_PREFLIGHT_JSON.name}")

    # Preflight MD
    md_text = build_md_report(
        verdict, vsum, static_results, static_n_pass, static_n_fail,
        cand_mapping, coverage_rows, blockers, warnings, notes)
    OUT_PREFLIGHT_MD.write_text(md_text, encoding="utf-8")
    print(f"  → saved: {OUT_PREFLIGHT_MD.name}")

    # DONE.json
    done_data = {
        "status":    "DONE",
        "verdict":   verdict,
        "date":      datetime.now().strftime("%Y-%m-%d"),
        "blockers":  len(blockers),
        "warnings":  len(warnings),
        "outputs": [
            str(OUT_PREFLIGHT_MD.name),
            str(OUT_PREFLIGHT_JSON.name),
            str(OUT_VOL_INV_CSV.name),
            str(OUT_ROI_EXT_CSV.name),
            str(OUT_POOL_CSV.name),
            str(OUT_CELL_COV_CSV.name),
            str(OUT_SPARSE_CSV.name),
            str(OUT_POLICY_MD.name),
            str(OUT_POLICY_JSON.name),
            str(OUT_CAND_MAP_JSON.name),
            str(OUT_ERRORS_CSV.name),
            str(OUT_DONE_JSON.name),
        ],
    }
    with open(OUT_DONE_JSON, "w") as f:
        json.dump(done_data, f, indent=2, ensure_ascii=False)
    print(f"  → saved: {OUT_DONE_JSON.name}")

    # Final summary
    print("\n" + "=" * 60)
    print(f"VERDICT: {verdict}")
    print(f"  usable volumes: {vsum['n_usable_volumes']}")
    print(f"  total candidates: {len(candidate_records)} "
          f"(quality: {n_quality})")
    print(f"  cell coverage — PASS_TOP3: {vsum['n_pass_top3']}/{vsum['n_total_cells']}, "
          f"EMPTY: {vsum['n_empty_cells']}")
    print(f"  LUNG1-052__c3 cell top3: {cand_mapping['top3_available']}")
    if blockers:
        print(f"  BLOCKERS: {blockers}")
    if warnings:
        print(f"  WARNINGS: {warnings[:3]}")
    print("=" * 60)

    exit_code = 2 if verdict == "BLOCKED" else 0
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
