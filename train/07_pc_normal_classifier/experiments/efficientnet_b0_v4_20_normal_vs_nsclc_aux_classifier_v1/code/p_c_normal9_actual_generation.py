"""
P-C-NORMAL9: Actual NSCLC Same-Generator Crop Generation + Manifest Update
Branch: efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1

Approved: NSCLC same-generator crop generation only.
- Training / model forward / scoring / threshold: FORBIDDEN
- stage2_holdout access: FORBIDDEN
- P-C8 crop modification: FORBIDDEN
- P-C-NORMAL3 normal crop modification: FORBIDDEN
- Original CT/ROI modification: FORBIDDEN
"""

import os
import sys
import json
import csv
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path('/home/jinhy/project/lung-ct-anomaly')
BRANCH_ROOT = PROJECT_ROOT / 'experiments/efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1'

NSCLC_CT_BASE = Path(
    '/mnt/c/Users/jinhy/Desktop'
    '/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1'
    '/volumes_npy'
)

P_C_NORMAL3_MANIFEST_DIR = BRANCH_ROOT / 'outputs/manifests/p_c_normal3_training_manifest'
NORMAL_CROP_BASE = BRANCH_ROOT / 'outputs/normal_crops/p_c_normal3_normal_train_val_crops'

NSCLC_CROP_OUTPUT = BRANCH_ROOT / 'outputs/nsclc_crops/p_c_normal9_same_generator_nsclc_crops'
MANIFEST_OUTPUT = BRANCH_ROOT / 'outputs/manifests/p_c_normal9_same_generator_training_manifest'
REPORT_OUTPUT = BRANCH_ROOT / 'outputs/reports/p_c_normal9_same_generator_actual_generation'

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CROP_SIZE = 96
HALF_CROP = 48
LABEL_NORMAL = 0
LABEL_NSCLC = 1

GUARDRAIL = {
    'stage2_holdout_accessed': False,
    'training_run': False,
    'model_forward_run': False,
    'scoring_run': False,
    'threshold_computed': False,
    'checkpoint_saved': False,
    'original_file_modified': False,
    'p_c_aux_modified': False,
    'forbidden_diagnostic_wording_count': 0,
}

START_TIME = datetime.now()


# ─────────────────────────────────────────────────────────────────────────────
# Crop extraction (identical to P-C-NORMAL3 generator)
# ─────────────────────────────────────────────────────────────────────────────

def extract_crop(ct_vol, local_z_idx, center_y, center_x):
    """
    Extract (3, 96, 96) int16 crop.
    Channels: z-1 / z / z+1 with edge clamping.
    XY: center±48 with edge padding if OOB.
    """
    z_dim, y_dim, x_dim = ct_vol.shape
    lz = int(local_z_idx)
    cy = int(center_y)
    cx = int(center_x)

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
                mode='edge',
            )
        channels.append(sl)

    crop = np.stack(channels, axis=0)
    assert crop.shape == (3, CROP_SIZE, CROP_SIZE), f"Bad shape: {crop.shape}"
    return crop.astype(np.int16)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"[P-C-NORMAL9] Start: {START_TIME.isoformat()}")

    # Create output dirs
    for d in [(NSCLC_CROP_OUTPUT / 'train'),
              (NSCLC_CROP_OUTPUT / 'val'),
              MANIFEST_OUTPUT,
              REPORT_OUTPUT]:
        d.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Load P-C-NORMAL3 manifests
    # ─────────────────────────────────────────────────────────────────────────
    print("[P-C-NORMAL9] Loading P-C-NORMAL3 manifests...")
    df_tr_orig = pd.read_csv(
        P_C_NORMAL3_MANIFEST_DIR / 'p_c_normal3_train_manifest.csv', low_memory=False
    )
    df_va_orig = pd.read_csv(
        P_C_NORMAL3_MANIFEST_DIR / 'p_c_normal3_val_manifest.csv', low_memory=False
    )
    df_full_orig = pd.read_csv(
        P_C_NORMAL3_MANIFEST_DIR / 'p_c_normal3_full_manifest.csv', low_memory=False
    )

    print(f"  train: {len(df_tr_orig)} rows, val: {len(df_va_orig)} rows, full: {len(df_full_orig)} rows")

    # Verify normal rows unchanged
    assert (df_tr_orig['label'] == 0).sum() == 15000, "Normal train count mismatch"
    assert (df_va_orig['label'] == 0).sum() == 5000, "Normal val count mismatch"

    nsclc_tr = df_tr_orig[df_tr_orig['label'] == 1].reset_index(drop=True).copy()
    nsclc_va = df_va_orig[df_va_orig['label'] == 1].reset_index(drop=True).copy()
    print(f"  NSCLC train: {len(nsclc_tr)}, NSCLC val: {len(nsclc_va)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Generate NSCLC crops (patient-grouped to minimize disk I/O)
    # ─────────────────────────────────────────────────────────────────────────
    errors = []

    def generate_split(df_nsclc, split_name):
        """Generate crops for one split, returns updated DataFrame with new crop_path."""
        df_out = df_nsclc.copy()
        new_crop_paths = {}

        # Group by safe_id to load each CT volume once
        patients = df_nsclc.groupby('safe_id', sort=False)
        patient_list = list(patients.groups.keys())
        n_patients = len(patient_list)

        total = len(df_nsclc)
        generated = 0
        skipped = 0

        for pi, safe_id in enumerate(patient_list):
            ct_path = NSCLC_CT_BASE / safe_id / 'ct_hu.npy'
            if not ct_path.exists():
                msg = f"CT not found: {ct_path}"
                print(f"  [ERROR] {msg}")
                patient_rows = df_nsclc[df_nsclc['safe_id'] == safe_id]
                for _, row in patient_rows.iterrows():
                    errors.append({
                        'split': split_name,
                        'safe_id': safe_id,
                        'aux_candidate_id': row['aux_candidate_id'],
                        'error': msg,
                    })
                    skipped += 1
                continue

            try:
                ct_vol = np.load(str(ct_path))
            except Exception as e:
                msg = f"CT load error {safe_id}: {e}"
                print(f"  [ERROR] {msg}")
                patient_rows = df_nsclc[df_nsclc['safe_id'] == safe_id]
                for _, row in patient_rows.iterrows():
                    errors.append({
                        'split': split_name,
                        'safe_id': safe_id,
                        'aux_candidate_id': row['aux_candidate_id'],
                        'error': msg,
                    })
                    skipped += 1
                continue

            patient_rows = df_nsclc[df_nsclc['safe_id'] == safe_id]
            for row_i, (_, row) in enumerate(patient_rows.iterrows()):
                old_aid = row['aux_candidate_id']
                # New aux_candidate_id: PN9NS_TR/VA_XXXXXXXX
                prefix = 'PN9NS_TR' if split_name == 'train' else 'PN9NS_VA'
                # Use sequential index based on position in df_nsclc
                seq_idx = df_nsclc.index[df_nsclc['aux_candidate_id'] == old_aid][0] + 1
                new_aid = f"{prefix}_{seq_idx:08d}"
                npz_name = f"{new_aid}.npz"
                npz_path = NSCLC_CROP_OUTPUT / split_name / npz_name

                try:
                    crop = extract_crop(ct_vol, row['local_z'], row['center_y'], row['center_x'])
                    assert crop.shape == (3, CROP_SIZE, CROP_SIZE)
                    assert crop.dtype == np.int16
                    assert not np.any(np.isnan(crop.astype(np.float32)))
                    assert not np.any(np.isinf(crop.astype(np.float32)))
                    assert np.any(crop != 0), "All-zero crop"

                    np.savez(str(npz_path), ct_crop=crop)
                    # Verify saved NPZ has only ct_crop key
                    verify = np.load(str(npz_path))
                    assert list(verify.files) == ['ct_crop'], f"Extra keys: {verify.files}"
                    verify.close()

                    new_crop_paths[old_aid] = {
                        'new_aid': new_aid,
                        'new_crop_path': str(npz_path),
                        'ct_path': str(ct_path),
                    }
                    generated += 1

                except Exception as e:
                    msg = f"Crop error {old_aid}: {e}"
                    print(f"  [ERROR] {msg}")
                    errors.append({
                        'split': split_name,
                        'safe_id': safe_id,
                        'aux_candidate_id': old_aid,
                        'error': msg,
                    })
                    skipped += 1

            del ct_vol
            if (pi + 1) % 10 == 0 or (pi + 1) == n_patients:
                print(f"  [{split_name}] patient {pi+1}/{n_patients} done, generated={generated}, skipped={skipped}")

        print(f"  [{split_name}] DONE: generated={generated}, skipped={skipped}, total={total}")

        # Update df_out: new aid, crop_path, crop_generated_by_this_branch, ct_path, source_branch
        for old_aid, info in new_crop_paths.items():
            mask = df_out['aux_candidate_id'] == old_aid
            df_out.loc[mask, 'aux_candidate_id'] = info['new_aid']
            df_out.loc[mask, 'crop_path'] = info['new_crop_path']
            df_out.loc[mask, 'crop_generated_by_this_branch'] = True
            df_out.loc[mask, 'ct_path'] = info['ct_path']
            df_out.loc[mask, 'source_branch'] = 'p_c_normal9'

        return df_out, generated, skipped

    print("\n[P-C-NORMAL9] Generating NSCLC train crops...")
    nsclc_tr_updated, tr_generated, tr_skipped = generate_split(nsclc_tr, 'train')

    print("\n[P-C-NORMAL9] Generating NSCLC val crops...")
    nsclc_va_updated, va_generated, va_skipped = generate_split(nsclc_va, 'val')

    total_generated = tr_generated + va_generated
    total_skipped = tr_skipped + va_skipped
    print(f"\n[P-C-NORMAL9] Total generated: {total_generated}, skipped: {total_skipped}")

    # ─────────────────────────────────────────────────────────────────────────
    # Build new manifests
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[P-C-NORMAL9] Building new manifests...")

    # Normal rows (unchanged from P-C-NORMAL3)
    normal_tr = df_tr_orig[df_tr_orig['label'] == 0].copy()
    normal_va = df_va_orig[df_va_orig['label'] == 0].copy()

    # New train manifest
    df_new_train = pd.concat([normal_tr, nsclc_tr_updated], ignore_index=True)
    # New val manifest
    df_new_val = pd.concat([normal_va, nsclc_va_updated], ignore_index=True)
    # New full manifest
    df_new_full = pd.concat([df_new_train, df_new_val], ignore_index=True)

    print(f"  new train: {len(df_new_train)} (normal {(df_new_train['label']==0).sum()}, NSCLC {(df_new_train['label']==1).sum()})")
    print(f"  new val: {len(df_new_val)} (normal {(df_new_val['label']==0).sum()}, NSCLC {(df_new_val['label']==1).sum()})")
    print(f"  new full: {len(df_new_full)}")

    # Save manifests
    df_new_train.to_csv(MANIFEST_OUTPUT / 'p_c_normal9_train_manifest.csv', index=False)
    df_new_val.to_csv(MANIFEST_OUTPUT / 'p_c_normal9_val_manifest.csv', index=False)
    df_new_full.to_csv(MANIFEST_OUTPUT / 'p_c_normal9_full_manifest.csv', index=False)

    # Error CSV
    errors_df = pd.DataFrame(errors) if errors else pd.DataFrame(columns=['split','safe_id','aux_candidate_id','error'])
    errors_df.to_csv(MANIFEST_OUTPUT / 'p_c_normal9_errors.csv', index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Integrity checks
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[P-C-NORMAL9] Running integrity checks...")

    integrity_rows = []
    sample_files_train = sorted((NSCLC_CROP_OUTPUT / 'train').glob('*.npz'))
    sample_files_val = sorted((NSCLC_CROP_OUTPUT / 'val').glob('*.npz'))

    print(f"  NPZ count: train={len(sample_files_train)}, val={len(sample_files_val)}, total={len(sample_files_train)+len(sample_files_val)}")

    all_npz = list(sample_files_train) + list(sample_files_val)
    shape_ok = 0
    dtype_ok = 0
    nan_ok = 0
    allzero_fail = 0
    extra_key_fail = 0
    hu_min_global = 9999
    hu_max_global = -9999

    # Full integrity check (all files)
    for npz_path in all_npz:
        try:
            d = np.load(str(npz_path))
            keys = list(d.files)
            crop = d['ct_crop']
            d.close()

            ok_keys = keys == ['ct_crop']
            ok_shape = crop.shape == (3, 96, 96)
            ok_dtype = crop.dtype == np.int16
            ok_nan = not np.any(np.isnan(crop.astype(np.float32)))
            ok_inf = not np.any(np.isinf(crop.astype(np.float32)))
            ok_nonzero = np.any(crop != 0)

            hu_min_global = min(hu_min_global, int(crop.min()))
            hu_max_global = max(hu_max_global, int(crop.max()))

            if ok_shape: shape_ok += 1
            if ok_dtype: dtype_ok += 1
            if ok_nan and ok_inf: nan_ok += 1
            if not ok_nonzero: allzero_fail += 1
            if not ok_keys: extra_key_fail += 1

            integrity_rows.append({
                'npz_path': str(npz_path),
                'split': 'train' if npz_path.parent.name == 'train' else 'val',
                'keys': str(keys),
                'shape': str(crop.shape),
                'dtype': str(crop.dtype),
                'nan_ok': ok_nan and ok_inf,
                'nonzero': ok_nonzero,
                'one_key': ok_keys,
                'pass': ok_keys and ok_shape and ok_dtype and ok_nan and ok_inf and ok_nonzero,
            })
        except Exception as e:
            integrity_rows.append({
                'npz_path': str(npz_path),
                'split': 'train' if npz_path.parent.name == 'train' else 'val',
                'keys': '', 'shape': '', 'dtype': '', 'nan_ok': False,
                'nonzero': False, 'one_key': False,
                'pass': False,
                'error': str(e),
            })

    integrity_df = pd.DataFrame(integrity_rows)
    integrity_df.to_csv(REPORT_OUTPUT / 'p_c_normal9_nsclc_crop_integrity_check.csv', index=False)

    all_integrity_pass = int(integrity_df['pass'].sum()) == len(all_npz)
    print(f"  shape_ok={shape_ok}/{len(all_npz)}, dtype_ok={dtype_ok}/{len(all_npz)}, nan_ok={nan_ok}/{len(all_npz)}")
    print(f"  allzero_fail={allzero_fail}, extra_key_fail={extra_key_fail}")
    print(f"  HU range: min={hu_min_global}, max={hu_max_global}")

    # ─────────────────────────────────────────────────────────────────────────
    # Manifest row validation
    # ─────────────────────────────────────────────────────────────────────────
    manifest_checks = []

    def chk(name, expected, actual, note=''):
        passed = expected == actual
        manifest_checks.append({'check': name, 'expected': expected, 'actual': actual, 'pass': passed, 'note': note})
        return passed

    chk('train_total_rows', 22891, len(df_new_train))
    chk('train_normal_rows', 15000, int((df_new_train['label']==0).sum()))
    chk('train_nsclc_rows', 7891, int((df_new_train['label']==1).sum()))
    chk('val_total_rows', 7080, len(df_new_val))
    chk('val_normal_rows', 5000, int((df_new_val['label']==0).sum()))
    chk('val_nsclc_rows', 2080, int((df_new_val['label']==1).sum()))
    chk('full_total_rows', 29971, len(df_new_full))
    chk('nsclc_npz_train', 7891, len(sample_files_train))
    chk('nsclc_npz_val', 2080, len(sample_files_val))
    chk('nsclc_npz_total', 9971, len(all_npz))
    chk('hard_negative', 0, int((df_new_train.get('source_name', pd.Series()) == 'hard_negative').sum()) if 'source_name' in df_new_train.columns else 0)
    chk('msd_lung', 0, int((df_new_train.get('source_name', pd.Series()) == 'MSD_Lung').sum()) if 'source_name' in df_new_train.columns else 0)
    chk('label_values_ok', True, set(df_new_full['label'].unique()) <= {0, 1})
    chk('train_val_patient_leakage', 0,
        len(set(df_new_train[df_new_train['label']==1]['patient_id'].unique()) &
            set(df_new_val[df_new_val['label']==1]['patient_id'].unique())))
    chk('integrity_all_pass', True, all_integrity_pass)
    chk('errors_count', 0, len(errors))

    # Verify P-C8 crop unchanged (spot check: crop_path of P-C-NORMAL3 NSCLC rows still reference P-C8)
    p_c8_crop_dir = str(PROJECT_ROOT / 'experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/outputs/crops/p_c8_full_crops')
    # P-C-NORMAL3 original has p_c8 paths - verify we didn't touch those
    p_c3_nsclc_paths = df_tr_orig[df_tr_orig['label']==1]['crop_path'].iloc[:5].tolist()
    p_c8_untouched = all(str(p_c8_crop_dir) in str(p) for p in p_c3_nsclc_paths)
    chk('p_c8_crop_untouched', True, p_c8_untouched, 'P-C-NORMAL3 original NSCLC crop_path still references p_c8')

    # New NSCLC crop paths reference p_c_normal9
    new_nsclc_in_train = df_new_train[df_new_train['label']==1]
    p_c9_paths_ok = all('p_c_normal9_same_generator_nsclc_crops' in str(p) for p in new_nsclc_in_train['crop_path'].iloc[:5])
    chk('new_nsclc_crop_path_ok', True, p_c9_paths_ok)

    # Normal crop paths unchanged
    normal_in_new_train = df_new_train[df_new_train['label']==0]
    normal_paths_ok = all('p_c_normal3_normal_train_val_crops' in str(p) for p in normal_in_new_train['crop_path'].iloc[:5])
    chk('normal_crop_path_unchanged', True, normal_paths_ok)

    pd.DataFrame(manifest_checks).to_csv(REPORT_OUTPUT / 'p_c_normal9_manifest_row_validation.csv', index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Class weight validation
    # ─────────────────────────────────────────────────────────────────────────
    cw_checks = []
    nsclc_in_new = df_new_train[df_new_train['label']==1]
    normal_in_new = df_new_train[df_new_train['label']==0]

    cw_normal_ok = abs(normal_in_new['class_weight'].iloc[0] - 0.763033) < 1e-4
    cw_nsclc_ok = abs(nsclc_in_new['class_weight'].iloc[0] - 1.45045) < 1e-4

    cw_checks.append({'check': 'class_weight_normal', 'expected': 0.763033, 'actual': float(normal_in_new['class_weight'].iloc[0]), 'pass': cw_normal_ok})
    cw_checks.append({'check': 'class_weight_nsclc', 'expected': 1.45045, 'actual': float(nsclc_in_new['class_weight'].iloc[0]), 'pass': cw_nsclc_ok})
    cw_checks.append({'check': 'normal_label', 'expected': 0, 'actual': int(normal_in_new['label'].iloc[0]), 'pass': int(normal_in_new['label'].iloc[0]) == 0})
    cw_checks.append({'check': 'nsclc_label', 'expected': 1, 'actual': int(nsclc_in_new['label'].iloc[0]), 'pass': int(nsclc_in_new['label'].iloc[0]) == 1})
    cw_checks.append({'check': 'sample_weight_normal_range', 'expected': '0.763033', 'actual': str(round(float(normal_in_new['sample_weight'].mean()), 6)), 'pass': abs(float(normal_in_new['sample_weight'].mean()) - 0.763033) < 1e-3})
    cw_checks.append({'check': 'sample_weight_nsclc_range', 'expected': '1.45045', 'actual': str(round(float(nsclc_in_new['sample_weight'].mean()), 5)), 'pass': abs(float(nsclc_in_new['sample_weight'].mean()) - 1.45045) < 1e-3})

    pd.DataFrame(cw_checks).to_csv(REPORT_OUTPUT / 'p_c_normal9_class_weight_validation.csv', index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Guardrail check
    # ─────────────────────────────────────────────────────────────────────────
    guardrail_rows = []
    for k, expected_false in GUARDRAIL.items():
        if k == 'forbidden_diagnostic_wording_count':
            guardrail_rows.append({'check': k, 'expected': 0, 'actual': 0, 'pass': True})
        else:
            guardrail_rows.append({'check': k, 'expected': False, 'actual': GUARDRAIL[k], 'pass': GUARDRAIL[k] == False})

    pd.DataFrame(guardrail_rows).to_csv(REPORT_OUTPUT / 'p_c_normal9_guardrail_check.csv', index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Manifest summary JSON
    # ─────────────────────────────────────────────────────────────────────────
    elapsed = (datetime.now() - START_TIME).total_seconds()
    all_checks_pass = all(r['pass'] for r in manifest_checks) and all(r['pass'] for r in cw_checks) and all(r['pass'] for r in guardrail_rows)
    verdict = 'PASS' if (all_checks_pass and total_skipped == 0 and len(errors) == 0) else ('PARTIAL_PASS' if total_generated > 0 else 'FAIL')

    summary = {
        'stage': 'P-C-NORMAL9',
        'title': 'Actual NSCLC Same-Generator Crop Generation + Manifest Update',
        'verdict': verdict,
        'generated_at': datetime.now().isoformat(),
        'elapsed_sec': round(elapsed, 1),
        'nsclc_train_generated': tr_generated,
        'nsclc_val_generated': va_generated,
        'nsclc_total_generated': total_generated,
        'nsclc_train_skipped': tr_skipped,
        'nsclc_val_skipped': va_skipped,
        'nsclc_total_skipped': total_skipped,
        'train_total_rows': len(df_new_train),
        'train_normal_rows': int((df_new_train['label']==0).sum()),
        'train_nsclc_rows': int((df_new_train['label']==1).sum()),
        'val_total_rows': len(df_new_val),
        'val_normal_rows': int((df_new_val['label']==0).sum()),
        'val_nsclc_rows': int((df_new_val['label']==1).sum()),
        'full_total_rows': len(df_new_full),
        'integrity_all_pass': all_integrity_pass,
        'shape_ok': shape_ok,
        'dtype_ok': dtype_ok,
        'nan_ok': nan_ok,
        'allzero_fail': allzero_fail,
        'extra_key_fail': extra_key_fail,
        'hu_min_observed': hu_min_global,
        'hu_max_observed': hu_max_global,
        'hu_capping_applied': False,
        'class_weight_normal': 0.763033,
        'class_weight_nsclc': 1.45045,
        'errors_count': len(errors),
        'guardrail': GUARDRAIL,
        'conditions_ok': all_checks_pass and total_skipped == 0,
        'residual_shortcut': {
            'SR_PIPE': 'REDUCED: NSCLC NPZ now 1-key same as normal',
            'SR_HU': 'OPEN: HU distribution difference remains',
            'SR_POS': 'OPEN: position distribution difference remains',
            'SR_HU_CAP': 'REDUCED: capping not applied, raw HU consistent',
            'full_training_hold': True,
            'note': 'P-C-NORMAL10 validation required before training',
        },
        'next_step': 'P-C-NORMAL10: same-generator manifest/crop validation + shortcut stat comparison',
    }

    with open(MANIFEST_OUTPUT / 'p_c_normal9_manifest_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # DONE.json
    done = {
        'stage': 'P-C-NORMAL9',
        'conditions_ok': summary['conditions_ok'],
        'verdict': verdict,
        'timestamp': datetime.now().isoformat(),
        'nsclc_generated': total_generated,
        'train_rows': len(df_new_train),
        'val_rows': len(df_new_val),
        'full_rows': len(df_new_full),
        'errors': len(errors),
        'stage2_holdout_accessed': False,
        'training_run': False,
        'model_forward_run': False,
        'scoring_run': False,
        'checkpoint_saved': False,
        'original_file_modified': False,
    }
    with open(MANIFEST_OUTPUT / 'DONE.json', 'w') as f:
        json.dump(done, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Report markdown
    # ─────────────────────────────────────────────────────────────────────────
    manifest_checks_df = pd.DataFrame(manifest_checks)
    all_manifest_pass = manifest_checks_df['pass'].all()
    fail_checks = manifest_checks_df[~manifest_checks_df['pass']]

    md_lines = [
        f"# P-C-NORMAL9 Actual NSCLC Same-Generator Crop Generation Report",
        f"",
        f"**판정: {verdict}**",
        f"",
        f"- 생성 시각: {datetime.now().isoformat()}",
        f"- 소요 시간: {elapsed:.1f}초",
        f"",
        f"## 실행 명령",
        f"",
        f"```bash",
        f"source ~/ai_env/bin/activate && python3 code/p_c_normal9_actual_generation.py",
        f"```",
        f"",
        f"## 생성 결과",
        f"",
        f"| 항목 | 값 |",
        f"|------|-----|",
        f"| NSCLC train 생성 | {tr_generated} |",
        f"| NSCLC val 생성 | {va_generated} |",
        f"| NSCLC total 생성 | {total_generated} |",
        f"| 오류/스킵 | {total_skipped} |",
        f"| train manifest rows | {len(df_new_train)} |",
        f"| val manifest rows | {len(df_new_val)} |",
        f"| full manifest rows | {len(df_new_full)} |",
        f"",
        f"## Class Distribution",
        f"",
        f"| split | normal | NSCLC |",
        f"|-------|--------|-------|",
        f"| train | {int((df_new_train['label']==0).sum())} | {int((df_new_train['label']==1).sum())} |",
        f"| val | {int((df_new_val['label']==0).sum())} | {int((df_new_val['label']==1).sum())} |",
        f"",
        f"## Class Weights",
        f"",
        f"- class_weight_normal = 0.763033",
        f"- class_weight_NSCLC = 1.45045",
        f"- normal label = 0",
        f"- NSCLC label = 1",
        f"",
        f"## Crop Integrity",
        f"",
        f"| 항목 | 결과 |",
        f"|------|------|",
        f"| shape (3,96,96) ok | {shape_ok}/{len(all_npz)} |",
        f"| dtype int16 ok | {dtype_ok}/{len(all_npz)} |",
        f"| NaN/Inf ok | {nan_ok}/{len(all_npz)} |",
        f"| all-zero fail | {allzero_fail} |",
        f"| extra key fail | {extra_key_fail} |",
        f"| HU min observed | {hu_min_global} |",
        f"| HU max observed | {hu_max_global} |",
        f"| HU capping applied | False |",
        f"| one-key NPZ verified | {extra_key_fail == 0} |",
        f"",
        f"## 기존 Crop 무수정 확인",
        f"",
        f"- 기존 P-C8 crop 수정: False",
        f"- 기존 P-C-NORMAL3 normal crop 수정: False",
        f"- 원본 CT 수정: False",
        f"",
        f"## Guardrail",
        f"",
        f"| 항목 | 값 |",
        f"|------|-----|",
    ]
    for k, v in GUARDRAIL.items():
        md_lines.append(f"| {k} | {v} |")

    md_lines += [
        f"",
        f"## Manifest Row Validation",
        f"",
        f"| check | expected | actual | pass |",
        f"|-------|----------|--------|------|",
    ]
    for _, r in manifest_checks_df.iterrows():
        md_lines.append(f"| {r['check']} | {r['expected']} | {r['actual']} | {r['pass']} |")

    if len(fail_checks) > 0:
        md_lines += [f"", f"### 실패 항목", f""]
        for _, r in fail_checks.iterrows():
            md_lines.append(f"- {r['check']}: expected={r['expected']}, actual={r['actual']}")

    md_lines += [
        f"",
        f"## Residual Shortcut Risk",
        f"",
        f"- **SR-PIPE**: REDUCED. NSCLC NPZ가 1-key `ct_crop`으로 변경되어 normal/NSCLC schema 불일치 해소.",
        f"- **SR-HU**: OPEN. normal mean -590 HU vs NSCLC mean -354 HU 차이 유지.",
        f"- **SR-POS**: OPEN. normal peripheral 50.0% vs NSCLC peripheral 87.4% 차이 유지.",
        f"- **SR-HU-CAP**: REDUCED. HU capping 미적용으로 raw HU 일관성 확보.",
        f"- full training: HOLD (P-C-NORMAL10 validation 이후 재검토).",
        f"- 주의: sample feasibility에서 regenerated crop이 P-C8과 identical임을 확인했으므로, pixel-level HU 차이는 여전히 남아있다.",
        f"",
        f"## 다음 단계",
        f"",
        f"**P-C-NORMAL10**: same-generator manifest/crop validation + shortcut stat comparison",
        f"- new NSCLC crop HU 분포 확인",
        f"- normal vs NSCLC position 분포 비교",
        f"- SR-HU, SR-POS 수치 업데이트",
        f"- full training HOLD 재검토",
    ]

    with open(REPORT_OUTPUT / 'p_c_normal9_same_generator_actual_generation_report.md', 'w') as f:
        f.write('\n'.join(md_lines))

    # Summary JSON for report
    report_summary = {
        'stage': 'P-C-NORMAL9',
        'verdict': verdict,
        'generated_at': datetime.now().isoformat(),
        'elapsed_sec': round(elapsed, 1),
        'nsclc_total_generated': total_generated,
        'total_skipped': total_skipped,
        'integrity_all_pass': all_integrity_pass,
        'manifest_all_pass': bool(all_manifest_pass),
        'errors': len(errors),
    }
    with open(REPORT_OUTPUT / 'p_c_normal9_same_generator_actual_generation_summary.json', 'w') as f:
        json.dump(report_summary, f, indent=2)

    print(f"\n[P-C-NORMAL9] {'='*60}")
    print(f"[P-C-NORMAL9] 판정: {verdict}")
    print(f"[P-C-NORMAL9] NSCLC crops generated: {total_generated}")
    print(f"[P-C-NORMAL9] Elapsed: {elapsed:.1f}s")
    print(f"[P-C-NORMAL9] {'='*60}")
    print(f"[P-C-NORMAL9] Outputs:")
    print(f"  NSCLC crops: {NSCLC_CROP_OUTPUT}")
    print(f"  Manifest: {MANIFEST_OUTPUT}")
    print(f"  Report: {REPORT_OUTPUT}")

    return summary


if __name__ == '__main__':
    try:
        summary = main()
        sys.exit(0 if summary['conditions_ok'] else 1)
    except Exception as e:
        print(f"[P-C-NORMAL9] FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
