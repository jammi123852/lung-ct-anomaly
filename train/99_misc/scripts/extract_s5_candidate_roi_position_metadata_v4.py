"""
extract_s5_candidate_roi_position_metadata_v4.py

목적:
normal reference bank v4 multi-case retrieval dry-run에서 UNSUPPORTED 처리된 case의
ROI 기반 위치 metadata(lung_z_pct, y/x_pct_in_lung_bbox, z/y/x bin, cell_key)를
read-only 로 추출하여 cell mapping + top3 retrieval 가능 여부를 확정한다.

이번 단계는 report-only metadata extraction 이다.
- roi_0_0.npy read-only load 만 허용 (ALLOW_ROI_LOAD=True 일 때)
- CT load / PNG 생성 / card render / model forward / feature / contribution / stage2 금지
- 기존 artifact 수정 금지 (새 OUTPUT_ROOT 에만 기록)
- rough mapping 금지 (ROI pct 미산출 시 UNSUPPORTED/ WARNING 으로만 기록)

산출식은 reference bank v4 preflight 스크립트
  scripts/build_normal_reference_bank_v4_lung_roi_position_preflight.py
와 동일한 정의를 그대로 사용한다 (동일 기준 재계산).

guard:
  ALLOW_ROI_LOAD            = False (env ALLOW_ROI_LOAD=1 일 때만 True)
  ALLOW_CT_LOAD             = False
  ALLOW_PNG_WRITE           = False
  ALLOW_CARD_RENDER         = False
  ALLOW_STAGE2_HOLDOUT      = False
  ALLOW_MODEL_FORWARD       = False
  ALLOW_FEATURE_EXTRACTION  = False
  ALLOW_CONTRIBUTION_RECALC = False
  ALLOW_FULL300             = False

dry-run (ALLOW_ROI_LOAD=False) -> plan-only, BLOCKED exit 2, 파일 미기록.
actual  (ALLOW_ROI_LOAD=True)  -> roi_0_0 read-only 추출, CSV/JSON/MD 기록.
"""

import os
import sys
import csv
import json
import pathlib
from datetime import date

import numpy as np
import pandas as pd

# ============================================================
# GUARD FLAGS
# ============================================================
ALLOW_ROI_LOAD            = False   # report-only ROI read-only load (env override only)
ALLOW_CT_LOAD             = False
ALLOW_PNG_WRITE           = False
ALLOW_CARD_RENDER         = False
ALLOW_STAGE2_HOLDOUT      = False
ALLOW_MODEL_FORWARD       = False
ALLOW_FEATURE_EXTRACTION  = False
ALLOW_CONTRIBUTION_RECALC = False
ALLOW_FULL300             = False

# env override: actual report-only extraction 승인 시에만 True 로 올림 (내릴 수는 없음)
if os.environ.get("ALLOW_ROI_LOAD") == "1":
    ALLOW_ROI_LOAD = True

# ============================================================
# PATHS
# ============================================================
REPO_ROOT = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")

DRYRUN_ROOT = REPO_ROOT / (
    "outputs/position-aware-padim-v1/reports/"
    "reference_bank_v4_multi_case_retrieval_dryrun"
)
BANK_ROOT = REPO_ROOT / (
    "outputs/position-aware-padim-v1/reports/"
    "reference_bank_v4_lung_roi_position_metadata"
)
OUTPUT_ROOT = REPO_ROOT / (
    "outputs/position-aware-padim-v1/reports/"
    "reference_bank_v4_candidate_roi_position_metadata"
)

CANDIDATE_INVENTORY_CSV = DRYRUN_ROOT / "candidate_case_inventory_v4.csv"
UNSUPPORTED_INPUT_CSV   = DRYRUN_ROOT / "unsupported_or_missing_cases_v4.csv"

BANK_CELL_INDEX_CSV  = BANK_ROOT / "normal_reference_bank_v4_cell_index.csv"
BANK_TOP3_CSV        = BANK_ROOT / "normal_reference_bank_v4_top3_by_cell.csv"
BANK_POLICY_JSON     = BANK_ROOT / "normal_reference_bank_v4_retrieval_policy_frozen.json"

# ============================================================
# CONSTANTS (reference bank v4 preflight 과 동일 정의)
# ============================================================
IMAGE_WIDTH  = 512
IMAGE_HEIGHT = 512
SIDE_SPLIT_X = IMAGE_WIDTH // 2  # 256

Z_BIN_NAMES = ["Z0", "Z1", "Z2", "Z3", "Z4"]
Y_BIN_NAMES = ["Y0", "Y1", "Y2"]
X_BIN_NAMES = ["X0", "X1", "X2"]
SIDES       = ["image_left", "image_right"]

MIN_ROI_AREA = 300   # per-side slice ROI 최소 픽셀 (참고/경고용)
MIN_BBOX_DIM = 48    # bbox 최소 변 길이 (참고/경고용)

# 대상 case (3 unsupported + 1 control)
TARGET_CASES = [
    "LUNG1-320__c2",
    "LUNG1-041__c3",
    "MSD_lung_059__c2",
    "LUNG1-052__c3",   # control (기존 cell image_left|Z1|Y2|X1 read-only sanity check)
]
CONTROL_CASE_ID       = "LUNG1-052__c3"
CONTROL_EXPECTED_CELL = "image_left|Z1|Y2|X1"

# 후보 volume roi_0_0 경로 (read-only)
VOLUME_NPY_ROOT = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy"
)


# ============================================================
# BIN / SIDE HELPERS (preflight 와 동일)
# ============================================================
def get_z_bin(pct: float) -> str:
    return Z_BIN_NAMES[min(int(pct / 0.2), 4)]


def get_y_bin(pct: float) -> str:
    return Y_BIN_NAMES[min(int(pct * 3), 2)]


def get_x_bin(pct: float) -> str:
    return X_BIN_NAMES[min(int(pct * 3), 2)]


def image_lung_side_from_x(x_center: float) -> str:
    # image x-coord 기준. 해부학적 좌우라고 단정하지 않는다.
    return "image_left" if x_center < SIDE_SPLIT_X else "image_right"


# ============================================================
# STATIC CHECKS
# ============================================================
def _abort(msg: str, code: int = 2):
    print(f"\nBLOCKED: {msg}", file=sys.stderr)
    sys.exit(code)


def run_static_checks():
    """report-only 안전 정적 점검. 실패 시 BLOCKED exit 2."""
    checks = []

    def chk(idx, desc, passed, note=""):
        checks.append({"id": idx, "desc": desc, "passed": bool(passed), "note": note})

    # 1. output root is new (still allowed to exist empty; DONE collision은 write 직전에 별도 검사)
    chk(1, "output_root_name_is_candidate_roi_position_metadata",
        "reference_bank_v4_candidate_roi_position_metadata" in str(OUTPUT_ROOT))
    # 2. unsupported input exists
    chk(2, "unsupported_input_exists", UNSUPPORTED_INPUT_CSV.exists(),
        str(UNSUPPORTED_INPUT_CSV))
    # 3. candidate inventory exists
    chk(3, "candidate_inventory_exists", CANDIDATE_INVENTORY_CSV.exists(),
        str(CANDIDATE_INVENTORY_CSV))
    # 4. v4 reference bank metadata exists (cell index + top3 + policy)
    chk(4, "v4_reference_bank_metadata_exists",
        BANK_CELL_INDEX_CSV.exists() and BANK_TOP3_CSV.exists()
        and BANK_POLICY_JSON.exists())
    # 5. ROI load guard exists (constant defined)
    chk(5, "roi_load_guard_exists", "ALLOW_ROI_LOAD" in globals())
    # 6. CT load false
    chk(6, "ct_load_false", not ALLOW_CT_LOAD)
    # 7. PNG write false
    chk(7, "png_write_false", not ALLOW_PNG_WRITE)
    # 8. card render false
    chk(8, "card_render_false", not ALLOW_CARD_RENDER)
    # 9. model/feature/contribution false
    chk(9, "model_feature_contribution_false",
        not (ALLOW_MODEL_FORWARD or ALLOW_FEATURE_EXTRACTION
             or ALLOW_CONTRIBUTION_RECALC))
    # 10. stage2_holdout false
    chk(10, "stage2_holdout_false", not ALLOW_STAGE2_HOLDOUT)
    # 11. full300 false
    chk(11, "full300_false", not ALLOW_FULL300)
    # 12. z/y/x bin counts
    chk(12, "bins_5_3_3",
        len(Z_BIN_NAMES) == 5 and len(Y_BIN_NAMES) == 3 and len(X_BIN_NAMES) == 3)
    # 13. side split = 256 (non-anatomical)
    chk(13, "side_split_256_non_anatomical", SIDE_SPLIT_X == 256)

    n_pass = sum(1 for c in checks if c["passed"])
    n_fail = len(checks) - n_pass

    print("=" * 64)
    print("STATIC CHECKS (report-only ROI position metadata extraction)")
    print("=" * 64)
    for c in checks:
        flag = "PASS" if c["passed"] else "FAIL"
        extra = f"  ({c['note']})" if c["note"] else ""
        print(f"  [{flag}] {c['id']:>2}. {c['desc']}{extra}")
    print(f"  ---> {n_pass} PASS / {n_fail} FAIL")

    # hard-stop conditions
    must_pass = {1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13}
    failed_must = [c for c in checks if c["id"] in must_pass and not c["passed"]]
    if failed_must:
        ids = ", ".join(str(c["id"]) for c in failed_must)
        _abort(f"required static check(s) failed: {ids}")

    return checks


# ============================================================
# ROI METADATA EXTRACTION (single case)
# ============================================================
def compute_side_extent_and_bbox(roi_bool, side_name, local_z):
    """
    roi_bool: (Z, H, W) bool array
    반환: dict with side_z_min/max, n_slices_with_roi, slice bbox, area, valid flags
    preflight 와 동일 산출.
    """
    Z, H, W = roi_bool.shape
    if side_name == "image_left":
        x_start, x_end = 0, SIDE_SPLIT_X
    else:
        x_start, x_end = SIDE_SPLIT_X, W

    side_roi = roi_bool[:, :, x_start:x_end]          # (Z, H, half_w)
    z_has_roi = side_roi.any(axis=(1, 2))
    z_indices = np.where(z_has_roi)[0]

    out = {
        "x_start": x_start, "x_end": x_end,
        "side_z_min": None, "side_z_max": None,
        "side_n_slices_with_roi": int(len(z_indices)),
        "slice_side_roi_area": None,
        "lung_bbox_y0": None, "lung_bbox_x0": None,
        "lung_bbox_y1": None, "lung_bbox_x1": None,
        "side_valid": bool(len(z_indices) > 0),
        "slice_has_roi": False,
    }
    if len(z_indices) == 0:
        return out

    z_min = int(z_indices.min())
    z_max = int(z_indices.max())
    out["side_z_min"] = z_min
    out["side_z_max"] = z_max

    # slice bbox at local_z (해당 slice 가 side ROI 가 있는 경우)
    if 0 <= local_z < Z:
        slice_side_mask = side_roi[int(local_z)]          # (H, half_w)
        ys, xhs = np.where(slice_side_mask)
        if len(ys) > 0:
            out["slice_has_roi"] = True
            out["slice_side_roi_area"] = int(slice_side_mask.sum())
            out["lung_bbox_y0"] = int(ys.min())
            out["lung_bbox_y1"] = int(ys.max())
            out["lung_bbox_x0"] = int(xhs.min()) + x_start
            out["lung_bbox_x1"] = int(xhs.max()) + x_start
    return out


def extract_case_metadata(cand_row):
    """
    cand_row: dict(candidate_case_inventory_v4.csv row)
    반환: (metadata_dict, readiness_dict, error_or_None)
    """
    case_id    = cand_row["case_id"]
    patient_id = cand_row["patient_id"]
    volume_id  = cand_row["volume_id"]
    source     = cand_row["source"]
    local_z    = int(cand_row["local_z"])
    report_slice_index = int(cand_row["report_slice_index"])
    crop_y0 = int(cand_row["crop_y0"]); crop_x0 = int(cand_row["crop_x0"])
    crop_y1 = int(cand_row["crop_y1"]); crop_x1 = int(cand_row["crop_x1"])
    crop_center_y = float(cand_row["crop_center_y"])
    crop_center_x = float(cand_row["crop_center_x"])
    stage2_flag = str(cand_row.get("stage2_holdout_flag", "False")) == "True"

    side = image_lung_side_from_x(crop_center_x)
    roi_path = VOLUME_NPY_ROOT / volume_id / "roi_0_0.npy"

    meta = {
        "case_id": case_id, "patient_id": patient_id, "volume_id": volume_id,
        "source": source, "local_z": local_z,
        "report_slice_index": report_slice_index,
        "crop_y0": crop_y0, "crop_x0": crop_x0,
        "crop_y1": crop_y1, "crop_x1": crop_x1,
        "crop_center_y": crop_center_y, "crop_center_x": crop_center_x,
        "image_lung_side": side, "roi_path": str(roi_path),
        "roi_exists": roi_path.exists(), "roi_shape": "",
        "stage2_holdout_flag": stage2_flag,
        "side_z_min": "", "side_z_max": "", "side_n_slices_with_roi": "",
        "lung_z_pct": "", "z_bin_5": "",
        "slice_side_roi_area": "",
        "lung_bbox_y0": "", "lung_bbox_x0": "",
        "lung_bbox_y1": "", "lung_bbox_x1": "",
        "y_pct_in_lung_bbox": "", "x_pct_in_lung_bbox": "",
        "y_bin_3": "", "x_bin_3": "", "cell_key": "",
        "mapping_status": "", "mapping_warning": "", "notes": "",
    }
    readiness = {
        "case_id": case_id, "roi_path": str(roi_path),
        "roi_exists": roi_path.exists(),
        "roi_load_attempted": False, "roi_load_success": False,
        "roi_shape": "", "stage2_holdout_flag": stage2_flag,
        "reason_if_unusable": "",
    }
    error = None

    # stage2 guard
    if stage2_flag:
        meta["mapping_status"] = "UNSUPPORTED"
        meta["mapping_warning"] = "stage2_holdout_flag=True (forbidden)"
        readiness["reason_if_unusable"] = "stage2_holdout_flag=True"
        error = {"case_id": case_id, "stage": "guard",
                 "error_type": "STAGE2_HOLDOUT", "detail": "stage2_holdout_flag=True"}
        return meta, readiness, error

    if not roi_path.exists():
        meta["mapping_status"] = "UNSUPPORTED"
        meta["mapping_warning"] = "roi_0_0.npy not found"
        readiness["reason_if_unusable"] = "roi path missing"
        error = {"case_id": case_id, "stage": "roi_load",
                 "error_type": "ROI_MISSING", "detail": str(roi_path)}
        return meta, readiness, error

    # ROI read-only load
    readiness["roi_load_attempted"] = True
    try:
        roi = np.load(str(roi_path)).astype(bool)
    except Exception as e:
        meta["mapping_status"] = "UNSUPPORTED"
        meta["mapping_warning"] = f"ROI load error: {e}"
        readiness["reason_if_unusable"] = f"load error: {e}"
        error = {"case_id": case_id, "stage": "roi_load",
                 "error_type": "ROI_LOAD_ERROR", "detail": str(e)}
        return meta, readiness, error

    readiness["roi_load_success"] = True
    shape_str = "x".join(str(s) for s in roi.shape)
    readiness["roi_shape"] = shape_str
    meta["roi_shape"] = shape_str

    Z, H, W = roi.shape
    warnings = []

    if H != IMAGE_HEIGHT or W != IMAGE_WIDTH:
        warnings.append(f"ROI shape {shape_str} != expected H/W {IMAGE_HEIGHT}/{IMAGE_WIDTH}")
    if not (0 <= local_z < Z):
        meta["mapping_status"] = "UNSUPPORTED"
        meta["mapping_warning"] = f"local_z {local_z} out of volume z-range [0,{Z-1}]"
        readiness["reason_if_unusable"] = "local_z out of z-range"
        del roi
        error = {"case_id": case_id, "stage": "extract",
                 "error_type": "LOCAL_Z_OUT_OF_RANGE",
                 "detail": f"local_z={local_z}, Z={Z}"}
        return meta, readiness, error

    ext = compute_side_extent_and_bbox(roi, side, local_z)
    del roi  # free memory per-case

    meta["side_z_min"] = ext["side_z_min"]
    meta["side_z_max"] = ext["side_z_max"]
    meta["side_n_slices_with_roi"] = ext["side_n_slices_with_roi"]

    if not ext["side_valid"]:
        meta["mapping_status"] = "UNSUPPORTED"
        meta["mapping_warning"] = f"{side} has no ROI in this volume"
        readiness["reason_if_unusable"] = f"{side} side has no ROI"
        error = {"case_id": case_id, "stage": "extract",
                 "error_type": "SIDE_NO_ROI", "detail": side}
        return meta, readiness, error

    z_min, z_max = ext["side_z_min"], ext["side_z_max"]
    z_range = max(z_max - z_min, 1)
    lung_z_pct = (local_z - z_min) / z_range
    if not (0.0 <= lung_z_pct <= 1.0):
        warnings.append(
            f"local_z {local_z} outside {side} z-extent [{z_min},{z_max}] "
            f"(lung_z_pct={lung_z_pct:.4f}); clamped for binning")
    lung_z_pct_clamped = min(max(lung_z_pct, 0.0), 1.0)
    z_bin = get_z_bin(lung_z_pct_clamped)
    meta["lung_z_pct"] = round(float(lung_z_pct), 4)
    meta["z_bin_5"] = z_bin

    if not ext["slice_has_roi"]:
        meta["mapping_status"] = "UNSUPPORTED"
        meta["mapping_warning"] = (
            f"slice local_z={local_z} has no {side} ROI bbox; "
            f"y/x pct not derivable")
        readiness["reason_if_unusable"] = "slice has no side ROI bbox"
        error = {"case_id": case_id, "stage": "extract",
                 "error_type": "SLICE_NO_SIDE_ROI",
                 "detail": f"z={local_z}, side={side}"}
        return meta, readiness, error

    bbox_y0 = ext["lung_bbox_y0"]; bbox_y1 = ext["lung_bbox_y1"]
    bbox_x0 = ext["lung_bbox_x0"]; bbox_x1 = ext["lung_bbox_x1"]
    meta["slice_side_roi_area"] = ext["slice_side_roi_area"]
    meta["lung_bbox_y0"] = bbox_y0; meta["lung_bbox_x0"] = bbox_x0
    meta["lung_bbox_y1"] = bbox_y1; meta["lung_bbox_x1"] = bbox_x1

    bbox_h = max(bbox_y1 - bbox_y0, 1)
    bbox_w = max(bbox_x1 - bbox_x0, 1)
    y_pct = (crop_center_y - bbox_y0) / bbox_h
    x_pct = (crop_center_x - bbox_x0) / bbox_w

    if (bbox_y1 - bbox_y0) < MIN_BBOX_DIM or (bbox_x1 - bbox_x0) < MIN_BBOX_DIM:
        warnings.append(
            f"slice side bbox small (h={bbox_y1-bbox_y0}, w={bbox_x1-bbox_x0} "
            f"< MIN_BBOX_DIM {MIN_BBOX_DIM})")
    if ext["slice_side_roi_area"] < MIN_ROI_AREA:
        warnings.append(
            f"slice side roi_area {ext['slice_side_roi_area']} < MIN_ROI_AREA {MIN_ROI_AREA}")
    if not (0.0 <= y_pct <= 1.0):
        warnings.append(f"y_pct {y_pct:.4f} outside [0,1] (crop center outside side bbox); clamped")
    if not (0.0 <= x_pct <= 1.0):
        warnings.append(f"x_pct {x_pct:.4f} outside [0,1] (crop center outside side bbox); clamped")

    y_pct_clamped = min(max(y_pct, 0.0), 1.0)
    x_pct_clamped = min(max(x_pct, 0.0), 1.0)
    y_bin = get_y_bin(y_pct_clamped)
    x_bin = get_x_bin(x_pct_clamped)
    meta["y_pct_in_lung_bbox"] = round(float(y_pct), 4)
    meta["x_pct_in_lung_bbox"] = round(float(x_pct), 4)
    meta["y_bin_3"] = y_bin
    meta["x_bin_3"] = x_bin

    cell_key = f"{side}|{z_bin}|{y_bin}|{x_bin}"
    meta["cell_key"] = cell_key

    pct_in_range = (0.0 <= lung_z_pct <= 1.0
                    and 0.0 <= y_pct <= 1.0 and 0.0 <= x_pct <= 1.0)
    if pct_in_range and not warnings:
        meta["mapping_status"] = "COMPLETED"
    else:
        meta["mapping_status"] = "WARNING"
    meta["mapping_warning"] = "; ".join(warnings) if warnings else ""

    notes = ["side by image x-coord (NOT anatomical)",
             "local_z==report_slice_index for this case"
             if local_z == report_slice_index else
             f"local_z({local_z}) != report_slice_index({report_slice_index})"]
    if case_id == CONTROL_CASE_ID:
        match = (cell_key == CONTROL_EXPECTED_CELL)
        notes.append(
            f"CONTROL sanity: recomputed={cell_key} "
            f"expected={CONTROL_EXPECTED_CELL} -> {'MATCH' if match else 'MISMATCH'}")
    meta["notes"] = "; ".join(notes)

    return meta, readiness, None


# ============================================================
# RETRIEVAL (exact cell lookup from frozen bank)
# ============================================================
def build_retrieval_rows(meta, cell_index_df, top3_df):
    """meta(cell_key 확정) -> retrieval_top3 rows + mapping_completed row"""
    case_id = meta["case_id"]
    cell_key = meta["cell_key"]
    local_z = meta["local_z"]

    retr_rows = []
    mapping_row = {
        "case_id": case_id, "cell_key": cell_key,
        "image_lung_side": meta["image_lung_side"],
        "z_bin_5": meta["z_bin_5"], "y_bin_3": meta["y_bin_3"],
        "x_bin_3": meta["x_bin_3"],
        "lung_z_pct": meta["lung_z_pct"],
        "y_pct_in_lung_bbox": meta["y_pct_in_lung_bbox"],
        "x_pct_in_lung_bbox": meta["x_pct_in_lung_bbox"],
        "cell_top3_available": "", "cell_top5_available": "",
        "fallback_level_required": "", "mapping_confidence": "",
        "verdict": "",
    }

    if not cell_key or meta["mapping_status"] == "UNSUPPORTED":
        mapping_row["verdict"] = "UNSUPPORTED"
        mapping_row["fallback_level_required"] = "NA"
        mapping_row["mapping_confidence"] = "low"
        retr_rows.append({
            "case_id": case_id, "retrieval_rank": "NA", "fallback_level": "NA",
            "reference_id": "UNSUPPORTED", "reference_patient_id": "NA",
            "reference_volume_id": "NA", "reference_local_z": "NA",
            "reference_lung_z_pct": "NA", "reference_cell_key": "UNSUPPORTED",
            "reference_crop_y0": "NA", "reference_crop_x0": "NA",
            "reference_crop_y1": "NA", "reference_crop_x1": "NA",
            "reference_quality_score": "NA",
            "same_cell_flag": "NA", "same_side_flag": "NA",
            "unique_patient_flag": "NA", "not_same_z_matched": "True",
            "z_direction_limited": "True",
            "notes": "cell_key not derivable; no retrieval performed",
        })
        return retr_rows, mapping_row

    # cell index lookup
    ci = cell_index_df[cell_index_df["cell_key"] == cell_key]
    top3_avail = bool(ci["top3_available"].iloc[0]) if len(ci) else False
    top5_avail = bool(ci["top5_available"].iloc[0]) if len(ci) else False
    mapping_row["cell_top3_available"] = top3_avail
    mapping_row["cell_top5_available"] = top5_avail

    cell_rows = top3_df[top3_df["cell_key"] == cell_key].sort_values("rank_in_cell")
    n_found = len(cell_rows)

    if n_found >= 3 and top3_avail:
        fallback_level = 0
    else:
        # 90/90 PASS_TOP3 이므로 정상적으로 도달하지 않음. 도달 시 fallback 필요 표시만.
        fallback_level = 5  # PARTIAL (no force-fill)

    mapping_row["fallback_level_required"] = fallback_level

    patients_seen = set()
    for _, r in cell_rows.iterrows():
        rank = int(r["rank_in_cell"])
        ref_z = int(r["local_z"])
        abs_dz = abs(ref_z - local_z)
        uniq = r["patient_id"] not in patients_seen
        patients_seen.add(r["patient_id"])
        note = (f"same lung-ROI position cell (exact); "
                f"abs_delta_z vs candidate(z={local_z})={abs_dz}")
        if abs_dz > 5:
            note += "; z-direction alignment limited"
        retr_rows.append({
            "case_id": case_id, "retrieval_rank": rank,
            "fallback_level": fallback_level,
            "reference_id": r["reference_id"],
            "reference_patient_id": r["patient_id"],
            "reference_volume_id": r["volume_id"],
            "reference_local_z": ref_z,
            "reference_lung_z_pct": r["lung_z_pct"],
            "reference_cell_key": cell_key,
            "reference_crop_y0": int(r["crop_y0"]),
            "reference_crop_x0": int(r["crop_x0"]),
            "reference_crop_y1": int(r["crop_y1"]),
            "reference_crop_x1": int(r["crop_x1"]),
            "reference_quality_score": round(float(r["reference_quality_score"]), 4),
            "same_cell_flag": True, "same_side_flag": True,
            "unique_patient_flag": uniq,
            "not_same_z_matched": True, "z_direction_limited": True,
            "notes": note,
        })

    # mapping confidence / verdict
    if meta["mapping_status"] == "COMPLETED" and fallback_level == 0:
        mapping_row["mapping_confidence"] = "high"
        mapping_row["verdict"] = "COMPLETED"
    elif meta["mapping_status"] == "WARNING":
        mapping_row["mapping_confidence"] = "medium"
        mapping_row["verdict"] = "COMPLETED_WITH_WARNING"
    else:
        mapping_row["mapping_confidence"] = "medium"
        mapping_row["verdict"] = "PARTIAL"

    return retr_rows, mapping_row


# ============================================================
# PLAN-ONLY (dry-run) PRINT
# ============================================================
def print_plan(inv_df):
    print("\n" + "=" * 64)
    print("PLAN-ONLY (ALLOW_ROI_LOAD=False) — no ROI load, no file write")
    print("=" * 64)
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"target cases: {TARGET_CASES}")
    print("\n각 case 에 대해 (승인 후) 수행될 read-only 작업:")
    for case_id in TARGET_CASES:
        rows = inv_df[inv_df["case_id"] == case_id]
        if len(rows) == 0:
            print(f"  - {case_id}: (inventory 에 없음)")
            continue
        r = rows.iloc[0]
        side = image_lung_side_from_x(float(r["crop_center_x"]))
        roi_path = VOLUME_NPY_ROOT / r["volume_id"] / "roi_0_0.npy"
        exists = "OK" if roi_path.exists() else "MISSING"
        print(f"  - {case_id}: vol={r['volume_id']}")
        print(f"      local_z={r['local_z']} crop_center=({r['crop_center_y']},"
              f"{r['crop_center_x']}) side={side}")
        print(f"      roi={roi_path} [{exists}]")
    print("\n계산 정책: lung_z_pct=(local_z-z_min)/max(z_max-z_min,1); "
          "z_bin=int(pct/0.2); y/x_bin=int(pct*3); cell=side|z|y|x")
    print("생성 예정 파일(승인 후): 9개 (report md/json, 6 csv, errors.csv, DONE.json)")
    print("\nALLOW_ROI_LOAD=False -> BLOCKED (plan-only). 실제 추출은 "
          "ALLOW_ROI_LOAD=1 환경변수로 승인 실행.")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 64)
    print("S5 candidate ROI position metadata extraction (report-only)")
    print(f"date: {date.today()}")
    print(f"ALLOW_ROI_LOAD = {ALLOW_ROI_LOAD}")
    print("=" * 64)

    static_checks = run_static_checks()

    inv_df = pd.read_csv(CANDIDATE_INVENTORY_CSV, dtype=str)

    # ---- dry-run / guard gate ----
    if not ALLOW_ROI_LOAD:
        print_plan(inv_df)
        _abort("ALLOW_ROI_LOAD=False (report-only plan-only). "
               "Set env ALLOW_ROI_LOAD=1 to perform actual read-only extraction.")

    # ---- collision check (write 직전) ----
    if (OUTPUT_ROOT / "DONE.json").exists():
        _abort("existing DONE.json at OUTPUT_ROOT. Archive/remove before re-extracting.")

    # ---- load frozen bank ----
    cell_index_df = pd.read_csv(BANK_CELL_INDEX_CSV)
    top3_df = pd.read_csv(BANK_TOP3_CSV)

    # ---- per-case extraction ----
    meta_rows, readiness_rows, retr_rows, mapping_rows, error_rows = [], [], [], [], []
    unsupported_rows = []

    for case_id in TARGET_CASES:
        rows = inv_df[inv_df["case_id"] == case_id]
        if len(rows) == 0:
            error_rows.append({"case_id": case_id, "stage": "inventory",
                               "error_type": "CASE_NOT_FOUND",
                               "detail": "not in candidate_case_inventory_v4.csv"})
            continue
        cand_row = rows.iloc[0].to_dict()
        meta, readiness, err = extract_case_metadata(cand_row)
        meta_rows.append(meta)
        readiness_rows.append(readiness)
        if err is not None:
            error_rows.append(err)

        retr, mapping = build_retrieval_rows(meta, cell_index_df, top3_df)
        retr_rows.extend(retr)
        mapping_rows.append(mapping)

        if meta["mapping_status"] == "UNSUPPORTED":
            unsupported_rows.append({
                "case_id": case_id,
                "missing_field": "cell_key" if not meta["cell_key"] else "",
                "reason": meta["mapping_warning"],
                "image_lung_side": meta["image_lung_side"],
                "roi_exists": meta["roi_exists"],
                "can_retry": False,
                "note": "ROI loaded but mapping not derivable",
            })

        print(f"  [{case_id}] side={meta['image_lung_side']} "
              f"cell={meta['cell_key'] or 'NA'} status={meta['mapping_status']}")
        if meta["mapping_warning"]:
            print(f"      warning: {meta['mapping_warning']}")

    # ---- verdict ----
    n_target_unsup = sum(1 for c in TARGET_CASES if c != CONTROL_CASE_ID)
    n_done = sum(1 for m in meta_rows
                 if m["case_id"] != CONTROL_CASE_ID
                 and m["mapping_status"] in ("COMPLETED", "WARNING")
                 and m["cell_key"])
    n_blocked = sum(1 for e in error_rows
                    if e.get("error_type") == "STAGE2_HOLDOUT")

    # control match
    control_meta = next((m for m in meta_rows if m["case_id"] == CONTROL_CASE_ID), None)
    control_match = (control_meta is not None
                     and control_meta["cell_key"] == CONTROL_EXPECTED_CELL)

    if n_blocked > 0:
        verdict = "BLOCKED"
    elif n_done == n_target_unsup and (control_meta is None or control_match):
        # 모든 unsupported 가 cell_key 확정 + control 일치
        all_completed = all(
            m["mapping_status"] == "COMPLETED"
            for m in meta_rows if m["case_id"] != CONTROL_CASE_ID and m["cell_key"])
        verdict = "PASS" if all_completed else "PARTIAL_PASS"
    elif n_done >= 1:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "NEEDS_FIX"

    if control_meta is not None and not control_match:
        verdict = "NEEDS_FIX"  # control sanity mismatch = 산출식 불일치 위험

    # ---- write outputs ----
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    META_COLS = [
        "case_id", "patient_id", "volume_id", "source", "local_z",
        "report_slice_index", "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "crop_center_y", "crop_center_x", "image_lung_side", "roi_path",
        "roi_exists", "roi_shape", "stage2_holdout_flag",
        "side_z_min", "side_z_max", "side_n_slices_with_roi",
        "lung_z_pct", "z_bin_5", "slice_side_roi_area",
        "lung_bbox_y0", "lung_bbox_x0", "lung_bbox_y1", "lung_bbox_x1",
        "y_pct_in_lung_bbox", "x_pct_in_lung_bbox", "y_bin_3", "x_bin_3",
        "cell_key", "mapping_status", "mapping_warning", "notes",
    ]
    pd.DataFrame(meta_rows)[META_COLS].to_csv(
        OUTPUT_ROOT / "candidate_roi_position_metadata_v4.csv", index=False)

    MAP_COLS = [
        "case_id", "cell_key", "image_lung_side", "z_bin_5", "y_bin_3", "x_bin_3",
        "lung_z_pct", "y_pct_in_lung_bbox", "x_pct_in_lung_bbox",
        "cell_top3_available", "cell_top5_available",
        "fallback_level_required", "mapping_confidence", "verdict",
    ]
    pd.DataFrame(mapping_rows)[MAP_COLS].to_csv(
        OUTPUT_ROOT / "candidate_cell_mapping_completed_v4.csv", index=False)

    RETR_COLS = [
        "case_id", "retrieval_rank", "fallback_level", "reference_id",
        "reference_patient_id", "reference_volume_id", "reference_local_z",
        "reference_lung_z_pct", "reference_cell_key",
        "reference_crop_y0", "reference_crop_x0", "reference_crop_y1",
        "reference_crop_x1", "reference_quality_score",
        "same_cell_flag", "same_side_flag", "unique_patient_flag",
        "not_same_z_matched", "z_direction_limited", "notes",
    ]
    pd.DataFrame(retr_rows)[RETR_COLS].to_csv(
        OUTPUT_ROOT / "retrieval_top3_after_roi_metadata_v4.csv", index=False)

    READY_COLS = [
        "case_id", "roi_path", "roi_exists", "roi_load_attempted",
        "roi_load_success", "roi_shape", "stage2_holdout_flag",
        "reason_if_unusable",
    ]
    pd.DataFrame(readiness_rows)[READY_COLS].to_csv(
        OUTPUT_ROOT / "roi_path_readiness_v4.csv", index=False)

    UNSUP_COLS = ["case_id", "missing_field", "reason", "image_lung_side",
                  "roi_exists", "can_retry", "note"]
    if unsupported_rows:
        pd.DataFrame(unsupported_rows)[UNSUP_COLS].to_csv(
            OUTPUT_ROOT / "unsupported_after_roi_metadata_v4.csv", index=False)
    else:
        with open(OUTPUT_ROOT / "unsupported_after_roi_metadata_v4.csv", "w",
                  newline="") as f:
            csv.writer(f).writerow(UNSUP_COLS)

    # errors.csv
    with open(OUTPUT_ROOT / "errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "stage", "error_type", "detail"])
        for e in error_rows:
            w.writerow([e.get("case_id", ""), e.get("stage", ""),
                        e.get("error_type", ""), e.get("detail", "")])

    # ---- report json ----
    case_summ = []
    for m in meta_rows:
        case_summ.append({
            "case_id": m["case_id"], "image_lung_side": m["image_lung_side"],
            "lung_z_pct": m["lung_z_pct"], "cell_key": m["cell_key"],
            "mapping_status": m["mapping_status"],
            "warning": m["mapping_warning"],
        })
    report = {
        "date": str(date.today()),
        "verdict": verdict,
        "allow_roi_load": ALLOW_ROI_LOAD,
        "target_cases": TARGET_CASES,
        "control_case": CONTROL_CASE_ID,
        "control_expected_cell": CONTROL_EXPECTED_CELL,
        "control_recomputed_cell": control_meta["cell_key"] if control_meta else None,
        "control_match": control_match,
        "n_unsupported_targets": n_target_unsup,
        "n_cell_mapping_done": n_done,
        "cases": case_summ,
        "safety": {
            "ct_load": 0, "png_write": 0, "card_render": 0,
            "model_forward": 0, "feature_extraction": 0,
            "contribution_recalc": 0, "stage2_holdout_access": 0,
            "roi_load_readonly": int(sum(1 for r in readiness_rows
                                         if r["roi_load_success"])),
            "existing_artifact_modified": 0,
        },
        "static_checks_pass": sum(1 for c in static_checks if c["passed"]),
        "static_checks_total": len(static_checks),
        "errors": len(error_rows),
    }
    (OUTPUT_ROOT / "roi_position_metadata_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    # ---- report md ----
    md = [
        "# candidate ROI position metadata extraction report (v4, report-only)",
        f"date: {date.today()}",
        f"verdict: **{verdict}**",
        "",
        "## scope",
        "- report-only ROI read-only metadata extraction",
        "- CT load / PNG / card / model / feature / contribution / stage2: 0 (all forbidden)",
        f"- target cases: {', '.join(TARGET_CASES)}",
        f"- control: {CONTROL_CASE_ID} (expected {CONTROL_EXPECTED_CELL})",
        "",
        "## case-by-case cell mapping",
        "| case_id | side | lung_z_pct | cell_key | status |",
        "|---|---|---|---|---|",
    ]
    for m in meta_rows:
        md.append(f"| {m['case_id']} | {m['image_lung_side']} | "
                  f"{m['lung_z_pct']} | {m['cell_key'] or 'NA'} | "
                  f"{m['mapping_status']} |")
    md += [
        "",
        f"- control match (recomputed == expected): **{control_match}** "
        f"(recomputed={control_meta['cell_key'] if control_meta else 'NA'})",
        "",
        "## top3 retrieval (exact cell, fallback_level=0)",
        "| case_id | rank | reference_patient | ref_local_z | ref_cell |",
        "|---|---|---|---|---|",
    ]
    for r in retr_rows:
        if r["reference_id"] == "UNSUPPORTED":
            md.append(f"| {r['case_id']} | NA | UNSUPPORTED | NA | NA |")
        else:
            md.append(f"| {r['case_id']} | {r['retrieval_rank']} | "
                      f"{r['reference_patient_id']} | {r['reference_local_z']} | "
                      f"{r['reference_cell_key']} |")
    md += [
        "",
        "## unsupported remaining",
    ]
    if unsupported_rows:
        for u in unsupported_rows:
            md.append(f"- {u['case_id']}: {u['reason']}")
    else:
        md.append("- (none) all target unsupported cases now mapped")
    md += [
        "",
        "## safety",
        "- CT load: 0 / PNG: 0 / card: 0 / model: 0 / feature: 0 / contribution: 0",
        "- stage2_holdout access: 0",
        f"- roi_0_0 read-only loads: {report['safety']['roi_load_readonly']}",
        "- existing artifact modified: 0 (writes only to new OUTPUT_ROOT)",
        "",
        "## calculation policy (identical to reference bank v4 preflight)",
        "- image_lung_side: crop_center_x<256 -> image_left else image_right (image coord, NOT anatomical)",
        "- lung_z_pct = (local_z - side_z_min) / max(side_z_max - side_z_min, 1)",
        "- z_bin_5 = Z[min(int(pct/0.2),4)]; y_bin_3/x_bin_3 = [min(int(pct*3),2)]",
        "- bbox = per-slice side ROI y/x min~max (x offset +x_start)",
        "- cell_key = image_lung_side|z_bin_5|y_bin_3|x_bin_3",
        "- fallback policy: exact top3 (level0) -> z+-1 -> xy+-1 -> nearest -> PARTIAL (no force-fill)",
        "",
        "## notes",
        "- local_z == report_slice_index for these cases (preserved as separate columns)",
        "- z-direction alignment limited; not_same_z_matched=True",
        "- no diagnostic / lesion / cancer / vessel cause claim",
        "",
    ]
    (OUTPUT_ROOT / "roi_position_metadata_report.md").write_text("\n".join(md))

    # ---- DONE.json ----
    done = {
        "status": "DONE", "verdict": verdict, "date": str(date.today()),
        "allow_roi_load": ALLOW_ROI_LOAD,
        "n_cell_mapping_done": n_done,
        "n_unsupported_targets": n_target_unsup,
        "control_match": control_match,
        "errors": len(error_rows),
        "outputs": [
            "roi_position_metadata_report.md",
            "roi_position_metadata_report.json",
            "candidate_roi_position_metadata_v4.csv",
            "candidate_cell_mapping_completed_v4.csv",
            "retrieval_top3_after_roi_metadata_v4.csv",
            "unsupported_after_roi_metadata_v4.csv",
            "roi_path_readiness_v4.csv",
            "errors.csv",
            "DONE.json",
        ],
    }
    (OUTPUT_ROOT / "DONE.json").write_text(json.dumps(done, indent=2))

    # ---- final summary ----
    print("\n" + "=" * 64)
    print(f"VERDICT: {verdict}")
    print(f"  cell mapping done (targets): {n_done}/{n_target_unsup}")
    print(f"  control match: {control_match} "
          f"(recomputed={control_meta['cell_key'] if control_meta else 'NA'})")
    print(f"  errors: {len(error_rows)}")
    print(f"  CT/PNG/card/model/feature/contribution/stage2: 0")
    print(f"  outputs -> {OUTPUT_ROOT}")
    print("=" * 64)

    if verdict == "BLOCKED":
        sys.exit(2)
    elif verdict == "NEEDS_FIX":
        sys.exit(1)


if __name__ == "__main__":
    main()
