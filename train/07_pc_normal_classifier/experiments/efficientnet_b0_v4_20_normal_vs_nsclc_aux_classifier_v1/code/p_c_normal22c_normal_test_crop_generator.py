"""
P-C-NORMAL22c: Normal-Test Crop Generator
Branch: efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1

목적:
  n_c11 normal_test 36명의 21,600 좌표로 final_test용 (3,96,96) int16 crops 생성.
  label=0, split=final_test, source_split=normal_test

모드:
  --dry-check        : input/schema/feasibility 확인만 (actual 저장 없음)
  --run-generate --confirm-final-test --confirm-no-prediction --confirm-no-threshold
                     : actual crop + manifest 생성 (P-C-NORMAL22d+ 별도 승인 필요)

안전 규칙:
  ALLOW_REAL_GENERATION=False → --run-generate exit 2
  stage2_holdout 접근 금지
  model forward / training / threshold 계산 금지
  기존 결과 수정 금지
"""

import argparse
import csv
import json
import os
import py_compile
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Safety flag — P-C-NORMAL22d+ 별도 승인 후 True로 변경
# ─────────────────────────────────────────────────────────────────────────────
ALLOW_REAL_GENERATION = True

LEAKAGE_PATIENTS = frozenset()  # normal script: leakage 없음 (NSCLC only)
STAGE2_HOLDOUT_ACCESS_FORBIDDEN = True

GENERATOR_VERSION = "p_c_normal22c_normal_generator_v1"
CROP_SIZE = 96
HALF_CROP = 48
LABEL_NORMAL = 0

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path('/home/jinhy/project/lung-ct-anomaly')
BRANCH_ROOT = (
    PROJECT_ROOT
    / 'experiments'
    / 'efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1'
)

NORMAL_CT_BASE = Path(
    '/mnt/c/Users/jinhy/Desktop/'
    'Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/'
    'volumes_npy'
)

N_C11_MANIFEST = (
    PROJECT_ROOT
    / 'experiments'
    / 'normal_only_second_stage_refiner_v1'
    / 'outputs'
    / 'manifests'
    / 'n_c11_normal_test_crop_manifest'
    / 'n_c11_normal_test_crop_manifest.csv'
)

CROP_OUTPUT_DIR = (
    PROJECT_ROOT
    / 'outputs'
    / 'test_crops'
    / 'p_c_normal22_final_baseline_test_crops'
    / 'normal_test'
)
MANIFEST_OUTPUT_DIR = (
    PROJECT_ROOT
    / 'outputs'
    / 'manifests'
    / 'p_c_normal22_final_baseline_test_manifest'
)
DRYCHECK_OUTPUT_DIR = (
    PROJECT_ROOT
    / 'outputs'
    / 'reports'
    / 'p_c_normal22c_final_test_crop_generator_drycheck'
)

EXPECTED_PATIENTS = 36
EXPECTED_TOTAL_COORDS = 21600
EXPECTED_CROPS_PER_PATIENT = 600
EXPECTED_POSITION_BINS = {
    'upper_central', 'upper_peripheral',
    'middle_central', 'middle_peripheral',
    'lower_central', 'lower_peripheral',
}

GUARDRAIL = {
    'stage2_holdout_accessed': False,
    'crop_generated': False,
    'manifest_generated': False,
    'model_forward_run': False,
    'prediction_export_run': False,
    'training_run': False,
    'backward_run': False,
    'optimizer_step': False,
    'checkpoint_saved': False,
    'threshold_computed': False,
    'existing_outputs_modified': False,
    'forbidden_diagnostic_wording_count': 0,
}

START_TIME = datetime.now()


# ─────────────────────────────────────────────────────────────────────────────
# Crop extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_crop(ct_vol: np.ndarray, local_z: float, center_y: float, center_x: float) -> np.ndarray:
    """
    (3, 96, 96) int16 crop.
    z channels: z-1 / z / z+1 with edge clamping.
    xy: center±48 with zero-padding if OOB.
    """
    z_dim, y_dim, x_dim = ct_vol.shape
    lz = int(round(local_z))
    cy = int(round(center_y))
    cx = int(round(center_x))

    z_indices = [
        max(0, lz - 1),
        max(0, min(z_dim - 1, lz)),
        min(z_dim - 1, lz + 1),
    ]

    y0 = cy - HALF_CROP
    y1 = cy + HALF_CROP
    x0 = cx - HALF_CROP
    x1 = cx + HALF_CROP

    y_pad_before = max(0, -y0)
    y_pad_after  = max(0, y1 - y_dim)
    x_pad_before = max(0, -x0)
    x_pad_after  = max(0, x1 - x_dim)

    y0c = max(0, y0)
    y1c = min(y_dim, y1)
    x0c = max(0, x0)
    x1c = min(x_dim, x1)

    channels = []
    for zi in z_indices:
        sl = ct_vol[zi, y0c:y1c, x0c:x1c]
        if any([y_pad_before, y_pad_after, x_pad_before, x_pad_after]):
            sl = np.pad(
                sl,
                ((y_pad_before, y_pad_after), (x_pad_before, x_pad_after)),
                mode='constant',
                constant_values=0,
            )
        channels.append(sl)

    crop = np.stack(channels, axis=0)
    assert crop.shape == (3, CROP_SIZE, CROP_SIZE), f"Bad shape: {crop.shape}"
    return crop.astype(np.int16)


# ─────────────────────────────────────────────────────────────────────────────
# Dry-check
# ─────────────────────────────────────────────────────────────────────────────

def run_dry_check():
    print(f"[22c-NORMAL DRY-CHECK] Start: {START_TIME.isoformat()}")
    DRYCHECK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    errors = []
    results = {}

    # ── 1. Script compile check ───────────────────────────────────────────────
    print("[1/8] Script compile check...")
    this_script = Path(__file__)
    compile_ok = False
    try:
        py_compile.compile(str(this_script), doraise=True)
        compile_ok = True
        print("  PASS: compile ok")
    except py_compile.PyCompileError as e:
        errors.append({'check': 'compile', 'item': str(this_script), 'error': str(e)})
        print(f"  FAIL: {e}")

    results['compile_check'] = {'status': 'PASS' if compile_ok else 'FAIL', 'script': str(this_script)}

    # ── 2. Manifest load & schema check ──────────────────────────────────────
    print("[2/8] n_c11 manifest load...")
    manifest_ok = False
    df = None
    if not N_C11_MANIFEST.exists():
        errors.append({'check': 'manifest_exists', 'item': str(N_C11_MANIFEST), 'error': 'not found'})
        print(f"  FAIL: manifest not found: {N_C11_MANIFEST}")
    else:
        df = pd.read_csv(N_C11_MANIFEST, low_memory=False)
        required_cols = [
            'normal_test_candidate_id', 'patient_id', 'safe_id', 'split',
            'local_z', 'slice_index', 'center_y', 'center_x',
            'position_bin', 'z_level',
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            errors.append({'check': 'manifest_schema', 'item': str(N_C11_MANIFEST), 'error': f'missing cols: {missing}'})
            print(f"  FAIL: missing columns: {missing}")
        else:
            manifest_ok = True
            print(f"  PASS: {len(df)} rows, columns ok")

    results['manifest_check'] = {
        'path': str(N_C11_MANIFEST),
        'exists': N_C11_MANIFEST.exists(),
        'rows': len(df) if df is not None else 0,
        'status': 'PASS' if manifest_ok else 'FAIL',
    }

    # ── 3. Patient count & crop count check ──────────────────────────────────
    print("[3/8] Patient / coordinate count check...")
    count_ok = False
    if df is not None:
        actual_patients = df['patient_id'].nunique()
        actual_coords = len(df)
        patient_crops = df.groupby('patient_id').size()

        count_ok = (
            actual_patients == EXPECTED_PATIENTS
            and actual_coords == EXPECTED_TOTAL_COORDS
            and (patient_crops == EXPECTED_CROPS_PER_PATIENT).all()
        )

        print(f"  patients: {actual_patients} (expected {EXPECTED_PATIENTS})")
        print(f"  coords: {actual_coords} (expected {EXPECTED_TOTAL_COORDS})")
        crops_ok = (patient_crops == EXPECTED_CROPS_PER_PATIENT).all()
        print(f"  per-patient 600: {'OK' if crops_ok else 'MISMATCH'}")

        if not count_ok:
            errors.append({'check': 'count', 'item': 'normal_test', 'error': f'patients={actual_patients}, coords={actual_coords}'})
    else:
        actual_patients = 0
        actual_coords = 0

    results['count_check'] = {
        'patients': actual_patients if df is not None else 0,
        'expected_patients': EXPECTED_PATIENTS,
        'total_coords': actual_coords if df is not None else 0,
        'expected_coords': EXPECTED_TOTAL_COORDS,
        'per_patient_ok': count_ok,
        'status': 'PASS' if count_ok else 'FAIL',
    }

    # ── 4. Position bin check ─────────────────────────────────────────────────
    print("[4/8] Position bin check...")
    bin_ok = False
    bin_dist = {}
    if df is not None:
        actual_bins = set(df['position_bin'].unique())
        bin_dist = df['position_bin'].value_counts().to_dict()
        bin_ok = EXPECTED_POSITION_BINS.issubset(actual_bins)
        print(f"  bins found: {sorted(actual_bins)}")
        print(f"  distribution: {bin_dist}")
        if not bin_ok:
            missing_bins = EXPECTED_POSITION_BINS - actual_bins
            errors.append({'check': 'position_bins', 'item': 'normal_test', 'error': f'missing bins: {missing_bins}'})

    results['position_bin_check'] = {
        'bins_found': sorted(actual_bins) if df is not None else [],
        'expected_bins': sorted(EXPECTED_POSITION_BINS),
        'distribution': bin_dist,
        'status': 'PASS' if bin_ok else 'FAIL',
    }

    # ── 5. CT file existence check ────────────────────────────────────────────
    print("[5/8] CT file existence check...")
    ct_ok = False
    ct_missing = []
    if df is not None:
        safe_ids = df['safe_id'].unique()
        for sid in safe_ids:
            ct_path = NORMAL_CT_BASE / sid / 'ct_hu.npy'
            if not ct_path.exists():
                ct_missing.append(str(ct_path))

        ct_ok = len(ct_missing) == 0
        print(f"  CT checked: {len(safe_ids)}, missing: {len(ct_missing)}")
        if ct_missing:
            for m in ct_missing[:5]:
                errors.append({'check': 'ct_exists', 'item': m, 'error': 'file not found'})

    results['ct_existence_check'] = {
        'safe_ids_checked': len(safe_ids) if df is not None else 0,
        'missing_count': len(ct_missing),
        'missing_examples': ct_missing[:3],
        'status': 'PASS' if ct_ok else 'FAIL',
    }

    # ── 6. z / xy boundary check ──────────────────────────────────────────────
    print("[6/8] z boundary & xy boundary check (sample)...")
    boundary_ok = False
    boundary_issues = []
    if df is not None and ct_ok:
        sample_rows = df.sample(min(20, len(df)), random_state=42)
        for _, row in sample_rows.iterrows():
            sid = row['safe_id']
            ct_path = NORMAL_CT_BASE / sid / 'ct_hu.npy'
            try:
                ct_vol = np.load(str(ct_path), mmap_mode='r')
                z_dim, y_dim, x_dim = ct_vol.shape
                lz = int(round(row['local_z']))
                cy = int(round(row['center_y']))
                cx = int(round(row['center_x']))

                z_ok = 0 <= lz < z_dim
                y_range_ok = True  # zero-padding handles OOB
                x_range_ok = True

                z_minus1 = max(0, lz - 1)
                z_plus1  = min(z_dim - 1, lz + 1)

                if not z_ok:
                    boundary_issues.append({
                        'safe_id': sid,
                        'local_z': lz,
                        'z_dim': z_dim,
                        'issue': 'z out of range',
                    })
                    errors.append({'check': 'z_boundary', 'item': sid, 'error': f'local_z={lz} z_dim={z_dim}'})

            except Exception as e:
                boundary_issues.append({'safe_id': sid, 'issue': str(e)})

        boundary_ok = len(boundary_issues) == 0
        print(f"  boundary issues: {len(boundary_issues)}")

    results['boundary_check'] = {
        'sample_size': 20,
        'issues_found': len(boundary_issues),
        'issues_examples': boundary_issues[:3],
        'status': 'PASS' if boundary_ok else 'FAIL',
    }

    # ── 7. Crop feasibility check (5~10 crops, in-memory only) ────────────────
    print("[7/8] Crop feasibility check (in-memory, no save)...")
    feasibility_ok = False
    feasibility_results = []
    GUARDRAIL['crop_generated'] = False

    if df is not None and ct_ok:
        sample_crops = df.sample(min(10, len(df)), random_state=99)
        loaded_vols = {}

        for _, row in sample_crops.iterrows():
            sid = row['safe_id']
            ct_path = NORMAL_CT_BASE / sid / 'ct_hu.npy'
            try:
                if sid not in loaded_vols:
                    loaded_vols[sid] = np.load(str(ct_path))

                ct_vol = loaded_vols[sid]
                crop = extract_crop(ct_vol, row['local_z'], row['center_y'], row['center_x'])

                shape_ok   = crop.shape == (3, CROP_SIZE, CROP_SIZE)
                dtype_ok   = crop.dtype == np.int16
                nan_ok     = not np.any(np.isnan(crop.astype(np.float32)))
                inf_ok     = not np.any(np.isinf(crop.astype(np.float32)))
                nonzero_ok = np.any(crop != 0)

                feasibility_results.append({
                    'candidate_id': row['normal_test_candidate_id'],
                    'shape': str(crop.shape),
                    'dtype': str(crop.dtype),
                    'shape_ok': shape_ok,
                    'dtype_ok': dtype_ok,
                    'nan_ok': nan_ok,
                    'inf_ok': inf_ok,
                    'nonzero_ok': nonzero_ok,
                    'all_pass': all([shape_ok, dtype_ok, nan_ok, inf_ok, nonzero_ok]),
                })

            except Exception as e:
                feasibility_results.append({
                    'candidate_id': row.get('normal_test_candidate_id', '?'),
                    'error': str(e),
                    'all_pass': False,
                })
                errors.append({'check': 'crop_feasibility', 'item': str(row.get('normal_test_candidate_id', '?')), 'error': str(e)})

        feasibility_ok = all(r.get('all_pass', False) for r in feasibility_results)
        pass_count = sum(1 for r in feasibility_results if r.get('all_pass', False))
        print(f"  feasibility samples: {len(feasibility_results)}, pass: {pass_count}/{len(feasibility_results)}")
        print("  (no npz saved — in-memory only)")

    results['crop_feasibility'] = {
        'samples_checked': len(feasibility_results),
        'pass_count': sum(1 for r in feasibility_results if r.get('all_pass', False)),
        'results': feasibility_results,
        'status': 'PASS' if feasibility_ok else 'FAIL',
        'note': 'in-memory only, no npz saved',
    }

    # ── 8. Output collision check ─────────────────────────────────────────────
    print("[8/8] Output collision check...")
    crop_dir_exists    = CROP_OUTPUT_DIR.exists()
    manifest_dir_exists = MANIFEST_OUTPUT_DIR.exists()
    collision_found    = crop_dir_exists or manifest_dir_exists

    results['output_collision'] = {
        'crop_dir': str(CROP_OUTPUT_DIR),
        'crop_dir_exists': crop_dir_exists,
        'manifest_dir': str(MANIFEST_OUTPUT_DIR),
        'manifest_dir_exists': manifest_dir_exists,
        'collision_found': collision_found,
        'status': 'CLEAN' if not collision_found else 'COLLISION',
    }
    if collision_found:
        print(f"  WARNING: output dir already exists (collision possible)")
    else:
        print(f"  CLEAN: no collision")

    # ── Manifest schema plan ──────────────────────────────────────────────────
    manifest_schema = {
        'required_columns': [
            'row_index', 'final_candidate_id', 'patient_id', 'safe_id',
            'split', 'source_split', 'source_name', 'crop_path',
            'label', 'label_name', 'position_bin', 'z_level',
            'local_z', 'slice_index', 'center_y', 'center_x',
            'sample_weight', 'stage2_holdout_flag', 'hard_negative_flag',
            'msd_lung_flag', 'leakage_excluded_flag', 'generator_version',
            'crop_shape', 'crop_dtype',
        ],
        'optional_metadata': [
            'crop_lung_roi_ratio', 'lung_z_percentile',
            'crop_hu_mean', 'crop_hu_std',
            'dense_frac_gt_minus500', 'air_frac_lt_minus800',
        ],
        'label': LABEL_NORMAL,
        'label_name': 'normal',
        'split': 'final_test',
        'source_split': 'normal_test',
        'source_name': 'n_c11_normal_test',
        'stage2_holdout_flag': False,
        'hard_negative_flag': False,
        'msd_lung_flag': False,
        'leakage_excluded_flag': False,
        'generator_version': GENERATOR_VERSION,
        'crop_shape': '(3,96,96)',
        'crop_dtype': 'int16',
    }
    results['manifest_schema_plan'] = manifest_schema

    # ── Guardrail final record ─────────────────────────────────────────────────
    assert not GUARDRAIL['crop_generated'],   "crop_generated should remain False in dry-check"
    assert not GUARDRAIL['manifest_generated'], "manifest_generated should remain False"
    assert not GUARDRAIL['model_forward_run'],  "model_forward_run should remain False"
    assert not GUARDRAIL['training_run'],       "training_run should remain False"
    results['guardrail'] = {**GUARDRAIL, 'stage2_holdout_unlocked_for_final_test': False}

    # ── Overall verdict ──────────────────────────────────────────────────────
    all_checks = [
        compile_ok, manifest_ok, count_ok, bin_ok,
        ct_ok, boundary_ok, feasibility_ok, not collision_found,
    ]
    passed = sum(all_checks)
    total  = len(all_checks)

    if passed == total:
        decision = 'READY_FOR_FINAL_TEST_CROP_GENERATION'
    elif compile_ok and manifest_ok:
        decision = 'NEEDS_NORMAL_SCRIPT_FIX'
    else:
        decision = 'FAIL'

    results['decision'] = decision
    results['checks_passed'] = f"{passed}/{total}"
    results['error_count'] = len(errors)

    print(f"\n[DECISION] {decision} ({passed}/{total} checks passed)")

    # ── Write reports ──────────────────────────────────────────────────────────
    _write_reports(results, errors, feasibility_results)
    print(f"[22c-NORMAL DRY-CHECK] Done.")
    return decision


def _write_reports(results, errors, feasibility_results):
    out = DRYCHECK_OUTPUT_DIR

    # compile check CSV
    with open(out / 'p_c_normal22c_script_compile_check.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['script', 'status'])
        w.writerow([results['compile_check']['script'], results['compile_check']['status']])

    # normal test drycheck CSV
    with open(out / 'p_c_normal22c_normal_test_drycheck.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['check', 'expected', 'actual', 'status'])
        cc = results['count_check']
        w.writerow(['patients', cc['expected_patients'], cc['patients'], cc['status']])
        w.writerow(['total_coords', cc['expected_coords'], cc['total_coords'], cc['status']])
        bc = results['boundary_check']
        w.writerow(['boundary_issues', 0, bc['issues_found'], bc['status']])
        fc = results['crop_feasibility']
        w.writerow(['feasibility_pass', fc['samples_checked'], fc['pass_count'], fc['status']])

    # errors CSV
    with open(out / 'p_c_normal22c_errors.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['check', 'item', 'error'])
        w.writeheader()
        w.writerows(errors)

    # output collision CSV
    oc = results['output_collision']
    with open(out / 'p_c_normal22c_output_collision_check.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'exists', 'collision'])
        w.writerow([oc['crop_dir'], oc['crop_dir_exists'], oc['collision_found']])
        w.writerow([oc['manifest_dir'], oc['manifest_dir_exists'], oc['collision_found']])

    # manifest schema CSV
    ms = results['manifest_schema_plan']
    with open(out / 'p_c_normal22c_manifest_schema_check.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['column', 'required', 'plan_value'])
        for col in ms['required_columns']:
            w.writerow([col, True, ''])
        for col in ms['optional_metadata']:
            w.writerow([col, False, 'metadata_only'])

    # guardrail CSV
    with open(out / 'p_c_normal22c_guardrail_check.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['guardrail_item', 'expected', 'actual', 'pass'])
        gr = results['guardrail']
        expected_false = [
            'crop_generated', 'manifest_generated', 'model_forward_run',
            'prediction_export_run', 'training_run', 'backward_run',
            'optimizer_step', 'checkpoint_saved', 'threshold_computed',
            'existing_outputs_modified',
        ]
        for item in expected_false:
            w.writerow([item, False, gr[item], not gr[item]])
        w.writerow(['forbidden_diagnostic_wording_count', 0, gr['forbidden_diagnostic_wording_count'], gr['forbidden_diagnostic_wording_count'] == 0])

    # Summary JSON (normal script only — full report written by drycheck runner)
    summary = {
        'script': 'p_c_normal22c_normal_test_crop_generator.py',
        'date': START_TIME.isoformat(),
        'decision': results['decision'],
        'checks_passed': results['checks_passed'],
        'error_count': results['error_count'],
        'normal_patients': results['count_check']['patients'],
        'normal_coords': results['count_check']['total_coords'],
        'guardrail': results['guardrail'],
    }
    with open(out / 'p_c_normal22c_normal_script_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[Reports] Written to: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Post-run hard assertion (P-C-NORMAL22d2 hardening)
# ─────────────────────────────────────────────────────────────────────────────

def _post_run_hard_assertion_normal(saved, error_rows, manifest_rows, integrity_log):
    """
    Normal generation loop 종료 후 manifest fragment 저장 전 호출.
    하나라도 실패하면 manifest 저장 없이 exit 2.
    all_zero는 반드시 failure에 포함.
    """
    failures = []

    # 1. saved == 21,600
    if saved != EXPECTED_TOTAL_COORDS:
        failures.append(f"saved={saved} != expected={EXPECTED_TOTAL_COORDS}")

    # 2. len(error_rows) == 0
    if len(error_rows) != 0:
        failures.append(f"error_rows={len(error_rows)} != 0")

    # 3. len(manifest_rows) == 21,600
    if len(manifest_rows) != EXPECTED_TOTAL_COORDS:
        failures.append(f"manifest_rows={len(manifest_rows)} != {EXPECTED_TOTAL_COORDS}")

    # 4. integrity_log rows == 21,600
    if len(integrity_log) != EXPECTED_TOTAL_COORDS:
        failures.append(f"integrity_log={len(integrity_log)} != {EXPECTED_TOTAL_COORDS}")

    # 5. all shape_ok=True
    bad_shape = [r for r in integrity_log if not r.get('shape_ok', False)]
    if bad_shape:
        failures.append(f"shape_ok=False count={len(bad_shape)}")

    # 6. all dtype_ok=True
    bad_dtype = [r for r in integrity_log if not r.get('dtype_ok', False)]
    if bad_dtype:
        failures.append(f"dtype_ok=False count={len(bad_dtype)}")

    # 7. all has_nan=False
    has_nan_rows = [r for r in integrity_log if r.get('has_nan', True)]
    if has_nan_rows:
        failures.append(f"has_nan=True count={len(has_nan_rows)}")

    # 8. all has_inf=False
    has_inf_rows = [r for r in integrity_log if r.get('has_inf', True)]
    if has_inf_rows:
        failures.append(f"has_inf=True count={len(has_inf_rows)}")

    # 9. all all_zero=False (hard failure — all_zero는 반드시 failure)
    all_zero_rows = [r for r in integrity_log if r.get('all_zero', False)]
    if all_zero_rows:
        failures.append(f"all_zero=True count={len(all_zero_rows)} (hard failure)")

    # 10. patient count == 36
    actual_patients = len(set(r['patient_id'] for r in manifest_rows))
    if actual_patients != EXPECTED_PATIENTS:
        failures.append(f"patient_count={actual_patients} != {EXPECTED_PATIENTS}")

    # 11. label only 0
    bad_label = [r for r in manifest_rows if r.get('label') != LABEL_NORMAL]
    if bad_label:
        failures.append(f"label!=0 count={len(bad_label)}")

    # 12. stage2_holdout_flag all False
    bad_holdout = [r for r in manifest_rows if r.get('stage2_holdout_flag') is not False]
    if bad_holdout:
        failures.append(f"stage2_holdout_flag!=False count={len(bad_holdout)}")

    # 13. hard_negative_flag all False
    bad_hn = [r for r in manifest_rows if r.get('hard_negative_flag') is not False]
    if bad_hn:
        failures.append(f"hard_negative_flag!=False count={len(bad_hn)}")

    # 14. msd_lung_flag all False
    bad_msd = [r for r in manifest_rows if r.get('msd_lung_flag') is not False]
    if bad_msd:
        failures.append(f"msd_lung_flag!=False count={len(bad_msd)}")

    # 15. crop_path unique == 21,600
    crop_paths = [r['crop_path'] for r in manifest_rows]
    if len(set(crop_paths)) != EXPECTED_TOTAL_COORDS:
        failures.append(f"crop_path_unique={len(set(crop_paths))} != {EXPECTED_TOTAL_COORDS}")

    # 16. final_candidate_id unique == 21,600
    candidate_ids = [r['final_candidate_id'] for r in manifest_rows]
    if len(set(candidate_ids)) != EXPECTED_TOTAL_COORDS:
        failures.append(f"final_candidate_id_unique={len(set(candidate_ids))} != {EXPECTED_TOTAL_COORDS}")

    if failures:
        print("[POST-RUN ASSERTION FAIL] Normal generator post-run hard check FAILED:")
        for fail in failures:
            print(f"  - {fail}")
        print("[ABORT] manifest fragment 저장 중단 (failed 상태). DONE/combine으로 넘어가지 않음.")
        sys.exit(2)

    print(f"[POST-RUN ASSERTION PASS] All {len(manifest_rows)} normal crops passed all hard assertions.")


# ─────────────────────────────────────────────────────────────────────────────
# Actual generation (guarded)
# ─────────────────────────────────────────────────────────────────────────────

def run_generate(confirm_final_test: bool, confirm_no_prediction: bool, confirm_no_threshold: bool):
    """
    Actual crop generation — normal_test 36 patients → 21,600 (3,96,96) int16 crops.
    ALLOW_REAL_GENERATION=True + 세 가지 confirm 플래그 모두 필요.
    P-C-NORMAL22e 별도 승인 전까지 ALLOW_REAL_GENERATION=False 유지.
    """
    if not ALLOW_REAL_GENERATION:
        print("[ABORT] ALLOW_REAL_GENERATION=False. P-C-NORMAL22e 별도 승인 후 True로 변경 필요.")
        sys.exit(2)

    if not (confirm_final_test and confirm_no_prediction and confirm_no_threshold):
        print("[ABORT] --confirm-final-test --confirm-no-prediction --confirm-no-threshold 세 가지 모두 필요.")
        sys.exit(2)

    run_start = datetime.now()
    print(f"[22d-NORMAL GENERATE] Start: {run_start.isoformat()}")

    # ── Output collision hard blocker ─────────────────────────────────────────
    COMBINED_MANIFEST_CSV = MANIFEST_OUTPUT_DIR / 'p_c_normal22_final_test_manifest.csv'
    NORMAL_FRAGMENT_CSV   = MANIFEST_OUTPUT_DIR / 'p_c_normal22_final_test_normal_manifest.csv'
    for col_label, col_path in [
        ('crop_dir_normal_test',     CROP_OUTPUT_DIR),
        ('normal_manifest_fragment', NORMAL_FRAGMENT_CSV),
        ('combined_manifest',        COMBINED_MANIFEST_CSV),
    ]:
        if col_path.exists():
            print(f"[ABORT] Output collision — {col_label}: {col_path}")
            print("  기존 출력 보호. 수동 확인 후 제거해야 재실행 가능.")
            sys.exit(2)
    print("[COLLISION-OK] No collision detected. Proceeding.")

    # ── stage2_holdout 접근 금지 guard ────────────────────────────────────────
    if STAGE2_HOLDOUT_ACCESS_FORBIDDEN:
        manifest_path_lower = str(N_C11_MANIFEST).lower()
        if 'stage2' in manifest_path_lower or 'holdout' in manifest_path_lower:
            print("[ABORT] N_C11_MANIFEST path references stage2/holdout — forbidden.")
            sys.exit(2)

    # ── Manifest load & count guard ───────────────────────────────────────────
    if not N_C11_MANIFEST.exists():
        print(f"[ABORT] n_c11 manifest not found: {N_C11_MANIFEST}")
        sys.exit(2)

    df = pd.read_csv(N_C11_MANIFEST, low_memory=False)
    actual_patients = df['patient_id'].nunique()
    actual_coords   = len(df)
    if actual_patients != EXPECTED_PATIENTS or actual_coords != EXPECTED_TOTAL_COORDS:
        print(f"[ABORT] Manifest count mismatch: patients={actual_patients}/{EXPECTED_PATIENTS}, "
              f"coords={actual_coords}/{EXPECTED_TOTAL_COORDS}")
        sys.exit(2)
    print(f"[MANIFEST] {actual_patients} patients | {actual_coords} coords")

    # ── Create output directories ─────────────────────────────────────────────
    CROP_OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    MANIFEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DIRS] crop: {CROP_OUTPUT_DIR}")

    # ── Crop generation loop ──────────────────────────────────────────────────
    manifest_rows = []
    integrity_log = []
    error_rows    = []
    loaded_vols   = {}
    total         = len(df)
    saved         = 0

    for row_idx, (_, row) in enumerate(df.iterrows()):
        sid        = row['safe_id']
        patient_id = row['patient_id']
        ct_path    = NORMAL_CT_BASE / sid / 'ct_hu.npy'

        try:
            if sid not in loaded_vols:
                if not ct_path.exists():
                    raise FileNotFoundError(f"CT not found: {ct_path}")
                loaded_vols[sid] = np.load(str(ct_path))
            ct_vol = loaded_vols[sid]

            crop = extract_crop(
                ct_vol,
                float(row['local_z']),
                float(row['center_y']),
                float(row['center_x']),
            )

            shape_ok = crop.shape == (3, CROP_SIZE, CROP_SIZE)
            dtype_ok = crop.dtype == np.int16
            has_nan  = bool(np.any(np.isnan(crop.astype(np.float32))))
            has_inf  = bool(np.any(np.isinf(crop.astype(np.float32))))
            all_zero = bool(not np.any(crop != 0))

            integrity_log.append({
                'row_index':    row_idx,
                'patient_id':   patient_id,
                'safe_id':      sid,
                'shape':        str(crop.shape),
                'dtype':        str(crop.dtype),
                'shape_ok':     shape_ok,
                'dtype_ok':     dtype_ok,
                'has_nan':      has_nan,
                'has_inf':      has_inf,
                'all_zero':     all_zero,
                'integrity_ok': shape_ok and dtype_ok and not has_nan and not has_inf,
            })

            if not (shape_ok and dtype_ok):
                error_rows.append({
                    'row_index':  row_idx,
                    'patient_id': patient_id,
                    'safe_id':    sid,
                    'error':      f'shape={crop.shape} dtype={crop.dtype}',
                })
                continue

            # 1-key npz, key=ct_crop
            candidate_id  = row['normal_test_candidate_id']
            patient_dir   = CROP_OUTPUT_DIR / sid
            patient_dir.mkdir(parents=True, exist_ok=True)
            crop_path_abs = patient_dir / f"{candidate_id}.npz"
            np.savez_compressed(str(crop_path_abs), ct_crop=crop)
            GUARDRAIL['crop_generated'] = True
            saved += 1

            manifest_rows.append({
                'row_index':             row_idx,
                'final_candidate_id':    candidate_id,
                'patient_id':            patient_id,
                'safe_id':               sid,
                'split':                 'final_test',
                'source_split':          'normal_test',
                'source_name':           'n_c11_normal_test',
                'crop_path':             str(crop_path_abs),
                'label':                 LABEL_NORMAL,
                'label_name':            'normal',
                'position_bin':          row['position_bin'],
                'z_level':               row['z_level'],
                'local_z':               row['local_z'],
                'slice_index':           row['slice_index'],
                'center_y':              row['center_y'],
                'center_x':              row['center_x'],
                'sample_weight':         1.0,
                'stage2_holdout_flag':   False,
                'hard_negative_flag':    False,
                'msd_lung_flag':         False,
                'leakage_excluded_flag': False,
                'generator_version':     GENERATOR_VERSION,
                'crop_shape':            '(3,96,96)',
                'crop_dtype':            'int16',
            })

        except Exception as exc:
            error_rows.append({
                'row_index':  row_idx,
                'patient_id': patient_id,
                'safe_id':    sid,
                'error':      str(exc),
            })

        if (row_idx + 1) % 2000 == 0:
            print(f"  [{row_idx + 1}/{total}] saved={saved} errors={len(error_rows)}")

    print(f"[GENERATE] saved={saved}/{total} errors={len(error_rows)}")

    # ── Post-run hard assertion (P-C-NORMAL22d2) — manifest 저장 전 필수 통과 ──
    _post_run_hard_assertion_normal(saved, error_rows, manifest_rows, integrity_log)

    # ── Save normal manifest fragment ─────────────────────────────────────────
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(str(NORMAL_FRAGMENT_CSV), index=False)
    GUARDRAIL['manifest_generated'] = True
    print(f"[MANIFEST] fragment → {NORMAL_FRAGMENT_CSV} ({len(df_manifest)} rows)")

    # ── Save integrity log & errors ───────────────────────────────────────────
    pd.DataFrame(integrity_log).to_csv(
        str(MANIFEST_OUTPUT_DIR / 'p_c_normal22d_normal_integrity_log.csv'), index=False
    )
    pd.DataFrame(error_rows).to_csv(
        str(MANIFEST_OUTPUT_DIR / 'p_c_normal22_errors.csv'), index=False
    )

    elapsed = (datetime.now() - run_start).total_seconds()
    print(f"[22d-NORMAL GENERATE] Done. elapsed={elapsed:.1f}s | saved={saved} | errors={len(error_rows)}")
    return saved, len(error_rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='P-C-NORMAL22c Normal Test Crop Generator')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-check', action='store_true')
    mode.add_argument('--run-generate', action='store_true')
    parser.add_argument('--confirm-final-test',    action='store_true')
    parser.add_argument('--confirm-no-prediction', action='store_true')
    parser.add_argument('--confirm-no-threshold',  action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_generate:
        if not ALLOW_REAL_GENERATION:
            print("[ABORT] ALLOW_REAL_GENERATION=False. bare --run-generate 불가.")
            sys.exit(2)
        run_generate(
            args.confirm_final_test,
            args.confirm_no_prediction,
            args.confirm_no_threshold,
        )
    elif args.dry_check:
        decision = run_dry_check()
        sys.exit(0 if decision == 'READY_FOR_FINAL_TEST_CROP_GENERATION' else 1)
    else:
        print("[ABORT] --dry-check 또는 --run-generate 필요.")
        sys.exit(2)


if __name__ == '__main__':
    main()
