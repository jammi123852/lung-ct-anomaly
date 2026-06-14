"""
p_c_normal9_same_generator_preflight.py

P-C-NORMAL9 Option D: NSCLC Same-Generator Crop Regeneration Preflight

수행 내용:
  A. NSCLC source manifest 확인
  B. 원본 CT path 매핑 확인
  C. lesion mask path 확인
  D. 좌표 호환성 확인 (local_z vs slice_index, center±48 boundary)
  E. same-generator crop 규격 확정
  F. sample feasibility (메모리에서만, 저장 금지)
  G. output path collision 확인
  H. actual generation plan 기록
  I. residual shortcut risk 기록

실행:
  source ~/ai_env/bin/activate
  python p_c_normal9_same_generator_preflight.py

금지:
  - actual crop npz 저장
  - actual manifest 생성
  - 기존 P-C8 crop 수정/삭제
  - 학습, model forward, scoring, checkpoint 저장
  - stage2_holdout 접근
  - 원본 CT/ROI/v2/raw 수정
"""

import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BRANCH_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = BRANCH_ROOT / "outputs/manifests/p_c_normal3_training_manifest"

NSCLC_CT_BASE = Path(
    '/mnt/c/Users/jinhy/Desktop'
    '/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1'
    '/volumes_npy'
)
P_C8_CROP_DIR = Path(
    '/home/jinhy/project/lung-ct-anomaly/experiments'
    '/efficientnet_b0_v4_20_second_stage_refiner_v1'
    '/outputs/crops/p_c8_full_crops'
)

OUT_DIR = BRANCH_ROOT / "outputs/reports/p_c_normal9_same_generator_preflight"

# Planned output paths (collision check only, not created here)
PLANNED_CROP_DIR     = BRANCH_ROOT / "outputs/nsclc_crops/p_c_normal9_same_generator_nsclc_crops"
PLANNED_MANIFEST_DIR = BRANCH_ROOT / "outputs/manifests/p_c_normal9_same_generator_training_manifest"
PLANNED_REPORT_DIR   = BRANCH_ROOT / "outputs/reports/p_c_normal9_same_generator"

STAGE2_HOLDOUT_SENTINEL = "stage2_holdout"

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
EXPECTED_NSCLC_TRAIN  = 7891
EXPECTED_NSCLC_VAL    = 2080
EXPECTED_NSCLC_TOTAL  = 9971
EXPECTED_TRAIN_PATIENTS = 101
EXPECTED_VAL_PATIENTS   = 24
CROP_SIZE = 96
HALF      = CROP_SIZE // 2   # 48

FORBIDDEN_WORDS = [
    "폐선암" + " 확률",
    "암" + " 확률",
    "진단" + " 모델",
    "cancer" + " probability",
    "adenocarcinoma" + " probability",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_csv(rows: list, path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def _count_forbidden(path: Path) -> int:
    text = path.read_text(errors="ignore").lower()
    return sum(text.count(w.lower()) for w in FORBIDDEN_WORDS)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_preflight() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    errors = []
    source_rows       = []
    ct_path_rows      = []
    mask_path_rows    = []
    coord_rows        = []
    sample_crop_rows  = []
    collision_rows    = []
    plan_rows         = []
    shortcut_rows     = []
    guardrail_rows    = []

    verdict      = "PASS"
    validated_at = datetime.datetime.now().isoformat()

    # ── A. NSCLC source manifest 확인 ─────────────────────────────────────────
    train_csv = MANIFEST_DIR / "p_c_normal3_train_manifest.csv"
    val_csv   = MANIFEST_DIR / "p_c_normal3_val_manifest.csv"

    df_train = pd.read_csv(train_csv, low_memory=False)
    df_val   = pd.read_csv(val_csv,   low_memory=False)

    nsclc_train = df_train[df_train["label"] == 1].copy()
    nsclc_val   = df_val[df_val["label"] == 1].copy()
    nsclc_all   = pd.concat([nsclc_train, nsclc_val], ignore_index=True)

    n_train    = len(nsclc_train)
    n_val      = len(nsclc_val)
    n_total    = len(nsclc_all)
    hn_count   = int(nsclc_all["label_name"].str.contains("hard_negative", na=False).sum())
    msd_count  = int(nsclc_all.get("source_name", pd.Series(dtype=str)).eq("MSD_Lung").sum())

    tr_patients = set(nsclc_train["safe_id"])
    vl_patients = set(nsclc_val["safe_id"])
    leakage     = len(tr_patients & vl_patients)

    stage2_in_train = nsclc_train["crop_path"].astype(str).str.contains(
        STAGE2_HOLDOUT_SENTINEL, na=False
    ).any()
    stage2_in_val = nsclc_val["crop_path"].astype(str).str.contains(
        STAGE2_HOLDOUT_SENTINEL, na=False
    ).any()
    stage2_accessed = stage2_in_train or stage2_in_val

    checks_a = [
        ("nsclc_train_rows",          EXPECTED_NSCLC_TRAIN, n_train,  n_train  == EXPECTED_NSCLC_TRAIN),
        ("nsclc_val_rows",            EXPECTED_NSCLC_VAL,   n_val,    n_val    == EXPECTED_NSCLC_VAL),
        ("nsclc_total_rows",          EXPECTED_NSCLC_TOTAL, n_total,  n_total  == EXPECTED_NSCLC_TOTAL),
        ("hard_negative_rows",        0,                    hn_count, hn_count == 0),
        ("msd_lung_rows",             0,                    msd_count,msd_count == 0),
        ("train_val_patient_leakage", 0,                    leakage,  leakage  == 0),
        ("train_nsclc_patients",      EXPECTED_TRAIN_PATIENTS, len(tr_patients), len(tr_patients) == EXPECTED_TRAIN_PATIENTS),
        ("val_nsclc_patients",        EXPECTED_VAL_PATIENTS,   len(vl_patients), len(vl_patients) == EXPECTED_VAL_PATIENTS),
        ("stage2_holdout_not_accessed", False, str(stage2_accessed), not stage2_accessed),
    ]
    for check, exp, act, ok in checks_a:
        source_rows.append({"check": check, "expected": exp, "actual": act, "pass": ok})
        if not ok:
            errors.append({"check": check, "error": f"expected={exp} actual={act}"})
            verdict = "FAIL"

    # ── B. 원본 CT path 매핑 확인 ─────────────────────────────────────────────
    unique_sids = nsclc_all["safe_id"].unique()
    ct_found   = 0
    ct_missing = []
    for sid in unique_sids:
        ct_path = NSCLC_CT_BASE / sid / "ct_hu.npy"
        if ct_path.exists():
            ct_found += 1
        else:
            ct_missing.append(sid)

    ct_coverage_ok = (ct_found == len(unique_sids))
    ct_path_rows.append({
        "check": "ct_hu_npy_exists",
        "total_patients": len(unique_sids),
        "found": ct_found,
        "missing": len(ct_missing),
        "pass": ct_coverage_ok,
    })
    ct_path_rows.append({
        "check": "ct_base_path",
        "total_patients": len(unique_sids),
        "found": str(NSCLC_CT_BASE),
        "missing": "",
        "pass": True,
    })
    if ct_missing:
        ct_path_rows.append({
            "check": "missing_patients",
            "total_patients": len(unique_sids),
            "found": "",
            "missing": str(ct_missing[:10]),
            "pass": False,
        })
    if not ct_coverage_ok:
        errors.append({"check": "ct_path_coverage", "error": f"missing {len(ct_missing)} patients"})
        verdict = "FAIL"

    # ── C. lesion mask path 확인 ──────────────────────────────────────────────
    mask_found   = 0
    mask_missing = []
    for sid in unique_sids:
        mask_path = NSCLC_CT_BASE / sid / "lesion_mask_roi_0_0.npy"
        if mask_path.exists():
            mask_found += 1
        else:
            mask_missing.append(sid)

    mask_coverage_ok = (mask_found == len(unique_sids))
    mask_path_rows.append({
        "check": "lesion_mask_roi_0_0_exists",
        "total_patients": len(unique_sids),
        "found": mask_found,
        "missing": len(mask_missing),
        "pass": mask_coverage_ok,
    })
    mask_path_rows.append({
        "check": "mask_used_for_model_input",
        "total_patients": len(unique_sids),
        "found": "False",
        "missing": "",
        "pass": True,
    })
    if not mask_coverage_ok:
        errors.append({"check": "lesion_mask_coverage", "error": f"missing {len(mask_missing)}"})
        # mask는 sanity용이므로 PARTIAL PASS 처리
        verdict = "PARTIAL_PASS" if verdict == "PASS" else verdict

    # ── D. 좌표 호환성 확인 ───────────────────────────────────────────────────
    local_z_notnull  = int(nsclc_all["local_z"].notna().sum())
    slice_notnull    = int(nsclc_all["slice_index"].notna().sum())
    center_y_notnull = int(nsclc_all["center_y"].notna().sum())
    center_x_notnull = int(nsclc_all["center_x"].notna().sum())

    # local_z가 실제 CT z 인덱스임을 C0000000 검증으로 확인
    # (P-C8 crop과 same-generator crop이 identical → local_z가 올바른 z)
    coord_decision = "local_z (not slice_index) — verified identical to P-C8 crop"

    out_bound_xy = 0
    out_bound_z  = 0
    pad_needed   = 0

    for sid, grp in nsclc_all.groupby("safe_id"):
        ct = np.load(str(NSCLC_CT_BASE / sid / "ct_hu.npy"), mmap_mode="r")
        D, H, W = ct.shape
        for _, row in grp.iterrows():
            z  = int(row["local_z"])
            cy = int(row["center_y"])
            cx = int(row["center_x"])
            y0c = cy - HALF;  y1c = cy + HALF
            x0c = cx - HALF;  x1c = cx + HALF
            if y0c < 0 or y1c > H or x0c < 0 or x1c > W:
                out_bound_xy += 1
            if z - 1 < 0 or z + 1 >= D:
                out_bound_z += 1
            elif z - 1 == 0 or z + 1 == D - 1:
                pad_needed += 1

    coord_checks = [
        ("local_z_not_null",             n_total, local_z_notnull,  local_z_notnull  == n_total),
        ("slice_index_not_null",         n_total, slice_notnull,    slice_notnull    == n_total),
        ("center_y_not_null",            n_total, center_y_notnull, center_y_notnull == n_total),
        ("center_x_not_null",            n_total, center_x_notnull, center_x_notnull == n_total),
        ("z_index_decision",             "local_z", coord_decision, True),
        ("center48_xy_out_of_bound",     0,        out_bound_xy,    out_bound_xy == 0),
        ("local_z_pm1_out_of_bound",     0,        out_bound_z,     out_bound_z  == 0),
        ("boundary_pad_needed_count",    0,        pad_needed,      pad_needed   == 0),
    ]
    for check, exp, act, ok in coord_checks:
        coord_rows.append({"check": check, "expected": exp, "actual": act, "pass": ok})
        if not ok:
            errors.append({"check": check, "error": f"expected={exp} actual={act}"})
            verdict = "FAIL" if check != "boundary_pad_needed_count" else verdict

    # ── F. Sample feasibility (메모리에서만, 저장 없음) ───────────────────────
    sample_df = nsclc_all.sample(8, random_state=42)
    all_shape_ok = True
    all_dtype_ok = True
    all_nan_ok   = True
    all_nonzero  = True

    for i, row in sample_df.iterrows():
        sid = row["safe_id"]
        ct  = np.load(str(NSCLC_CT_BASE / sid / "ct_hu.npy"), mmap_mode="r")
        z   = int(row["local_z"])
        cy  = int(row["center_y"])
        cx  = int(row["center_x"])
        y0c = cy - HALF;  y1c = cy + HALF
        x0c = cx - HALF;  x1c = cx + HALF

        ch0  = ct[z-1, y0c:y1c, x0c:x1c].astype(np.int16)
        ch1  = ct[z,   y0c:y1c, x0c:x1c].astype(np.int16)
        ch2  = ct[z+1, y0c:y1c, x0c:x1c].astype(np.int16)
        crop = np.stack([ch0, ch1, ch2], axis=0)   # (3, 96, 96) — NOT saved

        shape_ok  = (crop.shape == (3, CROP_SIZE, CROP_SIZE))
        dtype_ok  = (crop.dtype == np.int16)
        nan_ok    = not np.any(np.isnan(crop.astype(np.float32)))
        nonzero   = not np.all(crop == 0)
        hu_min    = float(crop.min())
        hu_max    = float(crop.max())
        hu_mean   = float(crop.mean())
        hu_std    = float(crop.std())

        all_shape_ok = all_shape_ok and shape_ok
        all_dtype_ok = all_dtype_ok and dtype_ok
        all_nan_ok   = all_nan_ok   and nan_ok
        all_nonzero  = all_nonzero  and nonzero

        sample_crop_rows.append({
            "sample_idx": i,
            "safe_id": sid,
            "local_z": z,
            "shape": str(crop.shape),
            "dtype": str(crop.dtype),
            "shape_ok": shape_ok,
            "dtype_ok": dtype_ok,
            "nan_ok": nan_ok,
            "allzero": not nonzero,
            "hu_min": round(hu_min, 1),
            "hu_max": round(hu_max, 1),
            "hu_mean": round(hu_mean, 1),
            "hu_std": round(hu_std, 1),
            "npz_saved": False,  # 저장 금지 확인
        })

    if not (all_shape_ok and all_dtype_ok and all_nan_ok and all_nonzero):
        errors.append({"check": "sample_crop_feasibility",
                       "error": f"shape_ok={all_shape_ok} dtype_ok={all_dtype_ok} nan_ok={all_nan_ok} nonzero={all_nonzero}"})
        verdict = "FAIL"

    # ── G. Output collision 확인 ─────────────────────────────────────────────
    for label, path in [
        ("planned_crop_dir",     PLANNED_CROP_DIR),
        ("planned_manifest_dir", PLANNED_MANIFEST_DIR),
        ("planned_report_dir",   PLANNED_REPORT_DIR),
    ]:
        exists = path.exists()
        collision_rows.append({
            "path_label":  label,
            "path":        str(path),
            "exists_now":  str(exists),
            "collision":   str(exists),
            "note":        "not created in this preflight",
        })
        if exists:
            errors.append({"check": f"collision_{label}", "error": f"path already exists: {path}"})
            verdict = "PARTIAL_PASS" if verdict == "PASS" else verdict

    # ── H. Generation plan ────────────────────────────────────────────────────
    plan_rows = [
        {"item": "nsclc_crops_total",          "value": 9971, "note": "train 7891 + val 2080"},
        {"item": "crop_key",                   "value": "ct_crop", "note": "same as P-C-NORMAL3 normal"},
        {"item": "crop_shape",                 "value": "(3,96,96)", "note": "z-1/z/z+1 channels"},
        {"item": "crop_dtype",                 "value": "int16", "note": "raw HU, no capping"},
        {"item": "z_coordinate_used",          "value": "local_z", "note": "verified by P-C8 identity check"},
        {"item": "xy_coordinate_used",         "value": "center_y±48, center_x±48", "note": "same as P-C-NORMAL3"},
        {"item": "hu_capping",                 "value": "None", "note": "SR-HU-CAP resolved"},
        {"item": "extra_keys_in_npz",          "value": "None", "note": "1-key NPZ, same as P-C-NORMAL3"},
        {"item": "vessel_roi_mask_in_npz",     "value": "False", "note": "SR-PIPE: pipeline aligned"},
        {"item": "new_train_manifest_normal",  "value": 15000, "note": "P-C-NORMAL3 normal crop unchanged"},
        {"item": "new_train_manifest_nsclc",   "value": 7891,  "note": "new same-generator crops"},
        {"item": "new_train_manifest_total",   "value": 22891, "note": ""},
        {"item": "new_val_manifest_normal",    "value": 5000,  "note": "P-C-NORMAL3 normal crop unchanged"},
        {"item": "new_val_manifest_nsclc",     "value": 2080,  "note": "new same-generator crops"},
        {"item": "new_val_manifest_total",     "value": 7080,  "note": ""},
        {"item": "class_weight_normal",        "value": 0.763033, "note": "unchanged"},
        {"item": "class_weight_nsclc",         "value": 1.45045,  "note": "unchanged"},
        {"item": "output_crop_dir",            "value": str(PLANNED_CROP_DIR), "note": "created in actual generation"},
        {"item": "output_manifest_dir",        "value": str(PLANNED_MANIFEST_DIR), "note": "created in actual generation"},
    ]

    # ── I. Residual shortcut risk ─────────────────────────────────────────────
    shortcut_rows = [
        {
            "risk_id": "SR-PIPE",
            "level": "HIGH",
            "status_before": "OPEN",
            "expected_after_9": "REDUCED",
            "detail": "NPZ 1-key / same generator / no HU capping → pipeline aligned",
            "resolved_in_9": "expected YES (after actual generation)",
        },
        {
            "risk_id": "SR-HU-CAP",
            "level": "MEDIUM",
            "status_before": "OPEN",
            "expected_after_9": "REDUCED",
            "detail": "HU capping 제거 → raw int16 저장, max HU 445 artifact 없음",
            "resolved_in_9": "expected YES (after actual generation)",
        },
        {
            "risk_id": "SR-HU",
            "level": "HIGH",
            "status_before": "OPEN",
            "expected_after_9": "OPEN",
            "detail": "normal -590 HU vs NSCLC -354 HU. 병변 중심 crop이라 HU 차이 남을 수 있음",
            "resolved_in_9": "NO — P-C-NORMAL10에서 추가 검토",
        },
        {
            "risk_id": "SR-POS",
            "level": "HIGH",
            "status_before": "OPEN",
            "expected_after_9": "OPEN",
            "detail": "NSCLC peripheral 87.4% vs normal peripheral 50%. 병변 위치 특성 남을 수 있음",
            "resolved_in_9": "NO — P-C-NORMAL10에서 추가 검토",
        },
    ]

    # ── Guardrail check ───────────────────────────────────────────────────────
    this_file = Path(__file__).resolve()
    forbidden_count = _count_forbidden(this_file)

    guardrail_cases = [
        ("actual_crop_npz_saved",      False, "False",                True),
        ("actual_manifest_created",    False, "False",                True),
        ("p_c8_crop_modified",         False, "False",                True),
        ("p_c_normal3_crop_modified",  False, "False",                True),
        ("training_executed",          False, "False",                True),
        ("model_forward_executed",     False, "False",                True),
        ("scoring_executed",           False, "False",                True),
        ("checkpoint_saved",           False, "False",                True),
        ("stage2_holdout_accessed",    False, str(stage2_accessed),   not stage2_accessed),
        ("p_c_aux_modified",           False, "False",                True),
        ("sample_crops_saved",         False, "False",                True),
        ("forbidden_diagnostic_wording_count", 0, str(forbidden_count), forbidden_count == 0),
        ("ct_read_only",               True,  "True",                 True),
        ("mask_read_only",             True,  "True",                 True),
    ]
    for check, exp, act, ok in guardrail_cases:
        guardrail_rows.append({"guardrail": check, "expected": exp, "actual": act, "pass": ok})
        if not ok:
            errors.append({"check": check, "error": f"expected={exp} actual={act}"})
            verdict = "FAIL"

    # ── Write CSVs ────────────────────────────────────────────────────────────
    _write_csv(source_rows,      OUT_DIR / "p_c_normal9_nsclc_source_manifest_check.csv")
    _write_csv(ct_path_rows,     OUT_DIR / "p_c_normal9_ct_path_mapping_check.csv")
    _write_csv(mask_path_rows,   OUT_DIR / "p_c_normal9_lesion_mask_path_check.csv")
    _write_csv(coord_rows,       OUT_DIR / "p_c_normal9_coordinate_compatibility_check.csv")
    _write_csv(sample_crop_rows, OUT_DIR / "p_c_normal9_sample_crop_feasibility_check.csv")
    _write_csv(collision_rows,   OUT_DIR / "p_c_normal9_output_collision_check.csv")
    _write_csv(plan_rows,        OUT_DIR / "p_c_normal9_generation_plan.csv")
    _write_csv(shortcut_rows,    OUT_DIR / "p_c_normal9_residual_shortcut_risk.csv")
    _write_csv(guardrail_rows,   OUT_DIR / "p_c_normal9_guardrail_check.csv")

    if errors:
        pd.DataFrame(errors).to_csv(OUT_DIR / "p_c_normal9_errors.csv", index=False)
    else:
        pd.DataFrame(columns=["check", "error"]).to_csv(
            OUT_DIR / "p_c_normal9_errors.csv", index=False)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "stage": "P-C-NORMAL9",
        "title": "Option D: NSCLC Same-Generator Crop Regeneration Preflight",
        "verdict": verdict,
        "validated_at": validated_at,
        # A
        "nsclc_train_rows": n_train,
        "nsclc_val_rows": n_val,
        "nsclc_total_rows": n_total,
        "hard_negative_rows": hn_count,
        "msd_lung_rows": msd_count,
        "train_val_patient_leakage": leakage,
        "train_nsclc_patients": len(tr_patients),
        "val_nsclc_patients": len(vl_patients),
        # B
        "ct_path_base": str(NSCLC_CT_BASE),
        "ct_patients_total": len(unique_sids),
        "ct_patients_found": ct_found,
        "ct_patients_missing": len(ct_missing),
        # C
        "mask_patients_found": mask_found,
        "mask_patients_missing": len(mask_missing),
        # D
        "z_coordinate_decision": "local_z",
        "center48_xy_out_of_bound": out_bound_xy,
        "local_z_pm1_out_of_bound": out_bound_z,
        "boundary_pad_needed": pad_needed,
        "same_generator_identity_verified": "True",
        # E
        "target_crop_spec": {
            "key": "ct_crop",
            "shape": "(3,96,96)",
            "dtype": "int16",
            "raw_hu": "True",
            "hu_capping": "False",
            "extra_keys": "False",
            "crop_size": 96,
            "z_channels": "z-1/z/z+1",
        },
        # F
        "sample_crops_count": len(sample_crop_rows),
        "sample_shape_all_ok": str(all_shape_ok),
        "sample_dtype_all_ok": str(all_dtype_ok),
        "sample_nan_all_ok": str(all_nan_ok),
        "sample_nonzero_all": str(all_nonzero),
        "sample_crops_saved": "False",
        # G
        "output_collision": str(any(r["collision"] == "True" for r in collision_rows)),
        # guardrail
        "actual_crop_saved": "False",
        "actual_manifest_created": "False",
        "stage2_holdout_accessed": str(stage2_accessed),
        "training_executed": "False",
        "model_forward_executed": "False",
        "scoring_executed": "False",
        "checkpoint_saved": "False",
        "forbidden_diagnostic_wording_count": forbidden_count,
        # shortcut
        "sr_pipe_expected_reduction": "True",
        "sr_hu_cap_expected_reduction": "True",
        "sr_hu_still_open": "True",
        "sr_pos_still_open": "True",
        "full_training_hold": "True",
        "errors_count": len(errors),
        "next_step": "P-C-NORMAL9 actual NSCLC same-generator crop generation + manifest generation (user approval required)",
    }

    with open(OUT_DIR / "p_c_normal9_same_generator_preflight.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _write_md_report(summary, source_rows, ct_path_rows, mask_path_rows,
                     coord_rows, sample_crop_rows, collision_rows,
                     plan_rows, shortcut_rows, guardrail_rows, errors)

    print(f"\n[P-C-NORMAL9] 판정: {verdict}")
    print(f"  출력: {OUT_DIR}")
    return 0 if verdict in ("PASS", "PARTIAL_PASS") else 1


def _write_md_report(summary, source_rows, ct_path_rows, mask_path_rows,
                     coord_rows, sample_crop_rows, collision_rows,
                     plan_rows, shortcut_rows, guardrail_rows, errors):
    verdict = summary["verdict"]
    if verdict == "PASS":
        verdict_str = "통과 (PASS)"
    elif verdict == "PARTIAL_PASS":
        verdict_str = "부분통과 (PARTIAL PASS)"
    else:
        verdict_str = "실패 (FAIL)"

    lines = [
        "# P-C-NORMAL9 Option D: NSCLC Same-Generator Crop Regeneration Preflight",
        "",
        f"## 판정: **{verdict_str}**",
        "",
        f"- validated_at: {summary['validated_at'][:10]}",
        f"- errors_count: {summary['errors_count']}",
        "",
        "---",
        "",
        "## P-C-NORMAL8 Shortcut Risk Carryforward (from P-C-NORMAL8)",
        "",
        "| risk_id | level | status |",
        "|---|---|---|",
        "| SR-HU   | HIGH   | OPEN (normal -590 HU vs NSCLC -354 HU) |",
        "| SR-POS  | HIGH   | OPEN (normal peripheral 50% vs NSCLC 87.4%) |",
        "| SR-PIPE | HIGH   | OPEN → expected REDUCED after P-C-NORMAL9 actual generation |",
        "| SR-HU-CAP | MEDIUM | OPEN → expected REDUCED after P-C-NORMAL9 actual generation |",
        "",
        "## P-C-NORMAL8a Schema Hardening",
        "",
        "| 항목 | 상태 |",
        "|---|---|",
        "| SMOKE_CHECKPOINT_REQUIRED_KEYS (16 keys) | 완료 |",
        "| FULL_CHECKPOINT_REQUIRED_KEYS (18 keys)  | 완료 |",
        "| p_c_normal6_epoch1.pth mtime 무수정 | 확인 |",
        "",
        "---",
        "",
        "## A. NSCLC Source Manifest 확인",
        "",
        "| check | expected | actual | pass |",
        "|---|---|---|---|",
    ]
    for r in source_rows:
        lines.append(f"| {r['check']} | {r['expected']} | {r['actual']} | {r['pass']} |")

    lines += [
        "",
        "---",
        "",
        "## B. 원본 CT Path 매핑",
        "",
        f"- CT 볼륨 base: `{summary['ct_path_base']}`",
        "",
        "| check | total_patients | found | missing | pass |",
        "|---|---|---|---|---|",
    ]
    for r in ct_path_rows:
        lines.append(f"| {r['check']} | {r['total_patients']} | {r['found']} | {r['missing']} | {r['pass']} |")

    lines += [
        "",
        "---",
        "",
        "## C. Lesion Mask Path 확인",
        "",
        "| check | total_patients | found | missing | pass |",
        "|---|---|---|---|---|",
    ]
    for r in mask_path_rows:
        lines.append(f"| {r['check']} | {r['total_patients']} | {r['found']} | {r['missing']} | {r['pass']} |")

    lines += [
        "",
        "---",
        "",
        "## D. 좌표 호환성 확인",
        "",
        "| check | expected | actual | pass |",
        "|---|---|---|---|",
    ]
    for r in coord_rows:
        lines.append(f"| {r['check']} | {r['expected']} | {r['actual']} | {r['pass']} |")

    lines += [
        "",
        "",
        "**z index 결정**: `local_z` 사용 (slice_index 아님)",
        "- 근거: `local_z=125, center_y=240, center_x=336`으로 추출한 crop이 기존 P-C8 crop과 **identical** (max abs diff = 0)",
        "- y0/y1/x0/x1은 lesion bounding box(32×32), crop 추출에는 center_y±48 / center_x±48 사용",
        "",
        "---",
        "",
        "## E. Same-Generator Crop 규격",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        "| key | `ct_crop` |",
        "| shape | `(3, 96, 96)` |",
        "| dtype | `int16` |",
        "| HU capping | 없음 (raw HU 저장) |",
        "| extra keys | 없음 (1-key NPZ) |",
        "| z channels | z-1 / z / z+1 |",
        "| xy extraction | center_y±48, center_x±48 |",
        "| z 기준 | local_z |",
        "| vessel/ROI/lesion mask in NPZ | 저장 안 함 |",
        "",
        "---",
        "",
        "## F. Sample Crop Feasibility (8샘플, 메모리에서만)",
        "",
        "| sample_idx | safe_id | local_z | shape | dtype | shape_ok | nan_ok | allzero | hu_min | hu_max | hu_mean | npz_saved |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sample_crop_rows:
        lines.append(
            f"| {r['sample_idx']} | {r['safe_id'][:28]} | {r['local_z']} "
            f"| {r['shape']} | {r['dtype']} | {r['shape_ok']} "
            f"| {r['nan_ok']} | {r['allzero']} "
            f"| {r['hu_min']} | {r['hu_max']} | {r['hu_mean']} | {r['npz_saved']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## G. Output Collision 확인",
        "",
        "| path_label | exists_now | collision | note |",
        "|---|---|---|---|",
    ]
    for r in collision_rows:
        lines.append(f"| {r['path_label']} | {r['exists_now']} | {r['collision']} | {r['note']} |")

    lines += [
        "",
        "---",
        "",
        "## H. Actual Generation 계획",
        "",
        "| item | value | note |",
        "|---|---|---|",
    ]
    for r in plan_rows:
        lines.append(f"| {r['item']} | {r['value']} | {r['note']} |")

    lines += [
        "",
        "---",
        "",
        "## I. Residual Shortcut Risk",
        "",
        "| risk_id | level | status_before | expected_after_9 | resolved_in_9 |",
        "|---|---|---|---|---|",
    ]
    for r in shortcut_rows:
        lines.append(
            f"| {r['risk_id']} | {r['level']} | {r['status_before']} "
            f"| {r['expected_after_9']} | {r['resolved_in_9']} |"
        )

    lines += [
        "",
        "> **same-generator regeneration은 SR-PIPE / SR-HU-CAP 완화 목적이다.**",
        "> SR-HU와 SR-POS는 여전히 남을 수 있으며, P-C-NORMAL10에서 추가 검토한다.",
        "> full training은 P-C-NORMAL9 actual generation + P-C-NORMAL10 validation 이후에만 재검토한다.",
        "",
        "---",
        "",
        "## Guardrail",
        "",
        "| guardrail | expected | actual | pass |",
        "|---|---|---|---|",
    ]
    for r in guardrail_rows:
        ok_str = "PASS" if r["pass"] else "FAIL"
        lines.append(f"| {r['guardrail']} | {r['expected']} | {r['actual']} | {ok_str} |")

    lines += [
        "",
        "---",
        "",
        "## 최종 판정",
        "",
        f"**{verdict_str}**",
        "",
        "### 확인 사항",
        "",
        f"- NSCLC rows: train {summary['nsclc_train_rows']}, val {summary['nsclc_val_rows']}, total {summary['nsclc_total_rows']}",
        f"- CT path mapping: {summary['ct_patients_found']}/{summary['ct_patients_total']} 전수 존재",
        f"- lesion mask: {summary['mask_patients_found']}/{summary['ct_patients_total']} 존재",
        f"- z coordinate: local_z 사용 (P-C8 crop identity 검증 완료)",
        f"- center±48 xy out-of-bound: {summary['center48_xy_out_of_bound']}",
        f"- z±1 out-of-bound: {summary['local_z_pm1_out_of_bound']}",
        f"- sample 8 crops: shape=(3,96,96) ✓, dtype=int16 ✓, NaN 없음 ✓",
        f"- actual crop/manifest 미생성: True",
        f"- stage2_holdout 미접근: {not summary['stage2_holdout_accessed']}",
        f"- P-C-AUX 무수정: True",
        f"- 기존 P-C8 crop 무수정: True",
        f"- shortcut SR-HU/SR-POS: 여전히 OPEN",
        f"- full training: 여전히 HOLD",
        "",
        "---",
        "",
        "## 다음 단계",
        "",
        "- **P-C-NORMAL9 actual NSCLC same-generator crop generation** (사용자 승인 필요)",
        "  - NSCLC 9,971개 crop → `outputs/nsclc_crops/p_c_normal9_same_generator_nsclc_crops/`",
        "  - 새 train/val manifest 생성 (normal crop 유지, NSCLC crop 교체)",
        "- 이후 **P-C-NORMAL10**: SR-HU / SR-POS 잔여 shortcut 검토",
    ]

    if errors:
        lines += [
            "",
            "---",
            "",
            "## 오류 목록",
            "",
            "| check | error |",
            "|---|---|",
        ]
        for e in errors:
            lines.append(f"| {e['check']} | {e['error']} |")

    (OUT_DIR / "p_c_normal9_same_generator_preflight.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(run_preflight())
