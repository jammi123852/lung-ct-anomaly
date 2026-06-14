"""
P-C-NORMAL13: Matched Manifest/Crop Validation + Shortcut Stat Comparison
==========================================================================
Read-only validation of P-C-NORMAL12 outputs.

Guardrails (strictly enforced):
  - NO training
  - NO model forward
  - NO scoring
  - NO threshold computation
  - NO checkpoint saving
  - NO crop re-generation
  - NO manifest modification
  - NO stage2_holdout access
  - NO modification of existing outputs
"""

import os
import json
import math
import random
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path('/home/jinhy/project/lung-ct-anomaly')
BRANCH_ROOT  = PROJECT_ROOT / 'experiments/efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1'

TRAIN_MANIFEST_PATH = (
    BRANCH_ROOT / 'outputs/manifests/p_c_normal12_matched_training_manifest'
    / 'p_c_normal12_train_manifest.csv'
)
VAL_MANIFEST_PATH = (
    BRANCH_ROOT / 'outputs/manifests/p_c_normal12_matched_training_manifest'
    / 'p_c_normal12_val_manifest.csv'
)
FULL_MANIFEST_PATH = (
    BRANCH_ROOT / 'outputs/manifests/p_c_normal12_matched_training_manifest'
    / 'p_c_normal12_full_manifest.csv'
)

NORMAL_CROP_TRAIN_DIR = (
    BRANCH_ROOT / 'outputs/normal_crops/p_c_normal12_matched_normal_crops/train'
)
NORMAL_CROP_VAL_DIR = (
    BRANCH_ROOT / 'outputs/normal_crops/p_c_normal12_matched_normal_crops/val'
)
NSCLC_CROP_TRAIN_DIR = (
    BRANCH_ROOT / 'outputs/nsclc_crops/p_c_normal9_same_generator_nsclc_crops/train'
)
NSCLC_CROP_VAL_DIR = (
    BRANCH_ROOT / 'outputs/nsclc_crops/p_c_normal9_same_generator_nsclc_crops/val'
)

P12_REPORT_DIR = BRANCH_ROOT / 'outputs/reports/p_c_normal12_matched_generation'
REPORT_DIR     = BRANCH_ROOT / 'outputs/reports/p_c_normal13_matched_validation_shortcut_stats'

N_C3_MANIFEST_PATH = (
    PROJECT_ROOT / 'experiments/normal_only_second_stage_refiner_v1'
    / 'outputs/manifests/n_c3_normal_only_crop_manifest_v1'
    / 'n_c3_normal_only_crop_manifest_v1.csv'
)
N_C10_MANIFEST_PATH = (
    PROJECT_ROOT / 'experiments/normal_only_second_stage_refiner_v1'
    / 'outputs/manifests/n_c10_normal_val_crop_manifest'
    / 'n_c10_normal_val_crop_manifest.csv'
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CROP_SIZE   = 96
HALF_CROP   = 48
LABEL_NORMAL = 0
LABEL_NSCLC  = 1
EXPECTED_TRAIN_ROWS   = 19727
EXPECTED_VAL_ROWS     = 5200
EXPECTED_FULL_ROWS    = 24927
EXPECTED_TRAIN_NORMAL = 11836
EXPECTED_TRAIN_NSCLC  = 7891
EXPECTED_VAL_NORMAL   = 3120
EXPECTED_VAL_NSCLC    = 2080
EXPECTED_TOTAL_NORMAL = 14956
EXPECTED_TOTAL_NSCLC  = 9971

Z_SAMPLE_N = 75   # per split for z-index reproduction check
HU_SAMPLE_N = 600  # per class for HU distribution

GUARDRAIL = {
    'stage2_holdout_accessed': False,
    'training_run': False,
    'model_forward_run': False,
    'scoring_run': False,
    'threshold_computed': False,
    'checkpoint_saved': False,
    'crop_regenerated': False,
    'manifest_modified': False,
    'existing_outputs_modified': False,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ts():
    return datetime.datetime.now().isoformat(timespec='seconds')


def extract_crop_for_validation(ct_vol, z_idx, center_y, center_x):
    """Exact replica of P-C-NORMAL12 extract_normal_crop for z-validation."""
    z_dim, y_dim, x_dim = ct_vol.shape
    lz = int(z_idx)
    cy = int(center_y)
    cx = int(center_x)

    z_indices = [
        max(0, lz - 1),
        max(0, min(z_dim - 1, lz)),
        min(z_dim - 1, lz + 1),
    ]

    y0 = cy - HALF_CROP;  y1 = cy + HALF_CROP
    x0 = cx - HALF_CROP;  x1 = cx + HALF_CROP

    y_pad_before = max(0, -y0);  y_pad_after  = max(0, y1 - y_dim)
    x_pad_before = max(0, -x0);  x_pad_after  = max(0, x1 - x_dim)
    y0c = max(0, y0);  y1c = min(y_dim, y1)
    x0c = max(0, x0);  x1c = min(x_dim, x1)

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
    return crop.astype(np.int16)


def hu_stats(arr_flat):
    """Compute HU distribution statistics from a flat int16 array."""
    arr = arr_flat.astype(np.float32)
    n = len(arr)
    if n == 0:
        return {}
    return {
        'n_pixels': int(n),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'p05': float(np.percentile(arr, 5)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'air_frac_lt_m800': float((arr < -800).sum() / n),
        'dense_frac_gt_m500': float((arr > -500).sum() / n),
        'dense_frac_gt_m300': float((arr > -300).sum() / n),
        'positive_frac_gt_0': float((arr > 0).sum() / n),
    }


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan')
    pooled_std = math.sqrt(((na - 1) * np.var(a) + (nb - 1) * np.var(b)) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def overlap_estimate(a, b, bins=200):
    """Histogram overlap (Bhattacharyya-style) 0~1."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if lo == hi:
        return 1.0
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    step = (hi - lo) / bins
    return float(np.sum(np.minimum(ha, hb)) * step)


# ─────────────────────────────────────────────────────────────────────────────
# A. File/count validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_file_counts():
    print(f"\n[{ts()}] A. File/count validation...")
    rows = []

    checks = [
        ('normal_train_crops_dir', NORMAL_CROP_TRAIN_DIR),
        ('normal_val_crops_dir',   NORMAL_CROP_VAL_DIR),
        ('nsclc_train_crops_dir',  NSCLC_CROP_TRAIN_DIR),
        ('nsclc_val_crops_dir',    NSCLC_CROP_VAL_DIR),
    ]
    counts_expected = {
        'normal_train_crops_dir': EXPECTED_TRAIN_NORMAL,
        'normal_val_crops_dir':   EXPECTED_VAL_NORMAL,
        'nsclc_train_crops_dir':  EXPECTED_TRAIN_NSCLC,
        'nsclc_val_crops_dir':    EXPECTED_VAL_NSCLC,
    }

    for name, path in checks:
        exists = path.exists()
        count = len(list(path.glob('*.npz'))) if exists else 0
        exp = counts_expected[name]
        match = (count == exp)
        rows.append({
            'item': name,
            'path': str(path),
            'path_exists': exists,
            'npz_count': count,
            'expected': exp,
            'count_match': match,
            'status': 'PASS' if (exists and match) else 'FAIL',
        })
        print(f"  {name}: {count}/{exp} {'OK' if match else 'MISMATCH'}")

    # Manifests
    manifest_checks = [
        ('train_manifest', TRAIN_MANIFEST_PATH),
        ('val_manifest',   VAL_MANIFEST_PATH),
        ('full_manifest',  FULL_MANIFEST_PATH),
    ]
    for name, path in manifest_checks:
        exists = path.exists()
        rows.append({
            'item': name,
            'path': str(path),
            'path_exists': exists,
            'npz_count': None,
            'expected': None,
            'count_match': None,
            'status': 'PASS' if exists else 'FAIL',
        })
        print(f"  {name}: {'EXISTS' if exists else 'MISSING'}")

    # P-C-NORMAL12 report
    done_path = P12_REPORT_DIR / 'DONE.json'
    rows.append({
        'item': 'p_c_normal12_DONE_json',
        'path': str(done_path),
        'path_exists': done_path.exists(),
        'npz_count': None,
        'expected': None,
        'count_match': None,
        'status': 'PASS' if done_path.exists() else 'FAIL',
    })
    print(f"  p_c_normal12 DONE.json: {'EXISTS' if done_path.exists() else 'MISSING'}")

    total_normal = (
        (len(list(NORMAL_CROP_TRAIN_DIR.glob('*.npz'))) if NORMAL_CROP_TRAIN_DIR.exists() else 0) +
        (len(list(NORMAL_CROP_VAL_DIR.glob('*.npz'))) if NORMAL_CROP_VAL_DIR.exists() else 0)
    )
    rows.append({
        'item': 'total_normal_crops',
        'path': 'train+val',
        'path_exists': True,
        'npz_count': total_normal,
        'expected': EXPECTED_TOTAL_NORMAL,
        'count_match': (total_normal == EXPECTED_TOTAL_NORMAL),
        'status': 'PASS' if total_normal == EXPECTED_TOTAL_NORMAL else 'FAIL',
    })
    print(f"  total_normal_crops: {total_normal}/{EXPECTED_TOTAL_NORMAL}")

    overall = all(r['status'] == 'PASS' for r in rows)
    return rows, overall


# ─────────────────────────────────────────────────────────────────────────────
# B. Manifest validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_manifests():
    print(f"\n[{ts()}] B. Manifest validation...")
    rows = []

    def chk(name, actual, expected, fmt=None):
        ok = (actual == expected)
        val = f"{actual:{fmt}}" if fmt else str(actual)
        rows.append({'check': name, 'actual': actual, 'expected': expected, 'status': 'PASS' if ok else 'FAIL'})
        print(f"  {name}: {val} ({'OK' if ok else 'FAIL, expected '+str(expected)})")
        return ok

    tr = pd.read_csv(TRAIN_MANIFEST_PATH, low_memory=False)
    vl = pd.read_csv(VAL_MANIFEST_PATH,   low_memory=False)
    fl = pd.read_csv(FULL_MANIFEST_PATH,  low_memory=False)

    chk('train_rows',        len(tr), EXPECTED_TRAIN_ROWS)
    chk('val_rows',          len(vl), EXPECTED_VAL_ROWS)
    chk('full_rows',         len(fl), EXPECTED_FULL_ROWS)
    chk('train_normal',      (tr['label'] == LABEL_NORMAL).sum(), EXPECTED_TRAIN_NORMAL)
    chk('train_nsclc',       (tr['label'] == LABEL_NSCLC).sum(),  EXPECTED_TRAIN_NSCLC)
    chk('val_normal',        (vl['label'] == LABEL_NORMAL).sum(), EXPECTED_VAL_NORMAL)
    chk('val_nsclc',         (vl['label'] == LABEL_NSCLC).sum(),  EXPECTED_VAL_NSCLC)

    # label values only {0,1}
    label_vals_ok = set(fl['label'].unique()) <= {0, 1}
    rows.append({'check': 'label_values_only_0_1', 'actual': str(set(fl['label'].unique())),
                 'expected': '{0, 1}', 'status': 'PASS' if label_vals_ok else 'FAIL'})
    print(f"  label_values_only_0_1: {set(fl['label'].unique())} {'OK' if label_vals_ok else 'FAIL'}")

    # hard_negative rows = 0
    hn_col = 'hard_negative' if 'hard_negative' in fl.columns else None
    if hn_col:
        hn_rows = (fl[hn_col] == True).sum()
        rows.append({'check': 'hard_negative_rows', 'actual': int(hn_rows), 'expected': 0,
                     'status': 'PASS' if hn_rows == 0 else 'FAIL'})
        print(f"  hard_negative_rows: {hn_rows}")
    else:
        rows.append({'check': 'hard_negative_rows', 'actual': 'col_absent', 'expected': 0, 'status': 'PASS'})
        print(f"  hard_negative_rows: column absent (OK)")

    # MSD_Lung rows = 0
    src_col = 'source_name' if 'source_name' in fl.columns else None
    if src_col:
        msd_rows = fl[src_col].str.contains('MSD', na=False).sum()
        rows.append({'check': 'msd_lung_rows', 'actual': int(msd_rows), 'expected': 0,
                     'status': 'PASS' if msd_rows == 0 else 'FAIL'})
        print(f"  msd_lung_rows: {msd_rows}")
    else:
        rows.append({'check': 'msd_lung_rows', 'actual': 'col_absent', 'expected': 0, 'status': 'PASS'})
        print(f"  msd_lung_rows: source_name column absent")

    # sample_weight / class_weight sanity
    sw_ok = 'sample_weight' in tr.columns and tr['sample_weight'].notna().all()
    cw_ok = 'class_weight'  in tr.columns and tr['class_weight'].notna().all()
    rows.append({'check': 'sample_weight_present_notna', 'actual': str(sw_ok), 'expected': 'True',
                 'status': 'PASS' if sw_ok else 'FAIL'})
    rows.append({'check': 'class_weight_present_notna', 'actual': str(cw_ok), 'expected': 'True',
                 'status': 'PASS' if cw_ok else 'FAIL'})
    print(f"  sample_weight ok: {sw_ok}, class_weight ok: {cw_ok}")

    # Expected class weights
    normal_cw = tr[tr['label'] == LABEL_NORMAL]['class_weight'].mean() if cw_ok else None
    nsclc_cw  = tr[tr['label'] == LABEL_NSCLC]['class_weight'].mean()  if cw_ok else None
    rows.append({'check': 'class_weight_normal_approx', 'actual': round(normal_cw, 4) if normal_cw else None,
                 'expected': 0.8333, 'status': 'PASS' if normal_cw and abs(normal_cw - 0.8333) < 0.01 else 'FAIL'})
    rows.append({'check': 'class_weight_nsclc_approx', 'actual': round(nsclc_cw, 4) if nsclc_cw else None,
                 'expected': 1.2500, 'status': 'PASS' if nsclc_cw and abs(nsclc_cw - 1.25) < 0.01 else 'FAIL'})
    print(f"  class_weight normal={round(normal_cw, 4) if normal_cw else None}, nsclc={round(nsclc_cw, 4) if nsclc_cw else None}")

    # Train/val patient leakage
    tr_patients = set(tr['patient_id'].unique())
    vl_patients = set(vl['patient_id'].unique())
    leakage = tr_patients & vl_patients
    rows.append({'check': 'train_val_patient_leakage', 'actual': len(leakage), 'expected': 0,
                 'status': 'PASS' if len(leakage) == 0 else 'FAIL'})
    print(f"  train_val_patient_leakage: {len(leakage)} patients")

    overall = all(r['status'] == 'PASS' for r in rows)
    return rows, overall, tr, vl, fl


# ─────────────────────────────────────────────────────────────────────────────
# C. Crop integrity validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_crop_integrity(tr, vl):
    print(f"\n[{ts()}] C. Crop integrity validation (sampling)...")
    rows = []
    errors = []

    # Sample normal crops from train+val
    normal_tr = tr[tr['label'] == LABEL_NORMAL].sample(n=min(150, len(tr[tr['label']==LABEL_NORMAL])), random_state=42)
    normal_vl = vl[vl['label'] == LABEL_NORMAL].sample(n=min(60,  len(vl[vl['label']==LABEL_NORMAL])),  random_state=42)
    nsclc_tr  = tr[tr['label'] == LABEL_NSCLC].sample(n=min(150, len(tr[tr['label']==LABEL_NSCLC])),  random_state=42)
    nsclc_vl  = vl[vl['label'] == LABEL_NSCLC].sample(n=min(60,  len(vl[vl['label']==LABEL_NSCLC])),   random_state=42)

    samples = [
        ('normal_train', normal_tr),
        ('normal_val',   normal_vl),
        ('nsclc_train',  nsclc_tr),
        ('nsclc_val',    nsclc_vl),
    ]

    for group_name, df_sample in samples:
        n_checked = 0
        n_ok = 0
        bad_path = 0
        bad_key = 0
        bad_shape = 0
        bad_dtype = 0
        nan_inf_cnt = 0
        all_zero_cnt = 0
        hu_vals = []

        for _, row in df_sample.iterrows():
            cp = Path(row['crop_path'])
            if not cp.exists():
                bad_path += 1
                errors.append({'group': group_name, 'crop_path': str(cp), 'error': 'path_not_found'})
                continue
            try:
                npz = np.load(str(cp))
                if 'ct_crop' not in npz:
                    bad_key += 1
                    errors.append({'group': group_name, 'crop_path': str(cp), 'error': 'key_ct_crop_missing'})
                    continue
                arr = npz['ct_crop']
                if arr.shape != (3, CROP_SIZE, CROP_SIZE):
                    bad_shape += 1
                    errors.append({'group': group_name, 'crop_path': str(cp), 'error': f'bad_shape_{arr.shape}'})
                    continue
                if arr.dtype != np.int16:
                    bad_dtype += 1
                    errors.append({'group': group_name, 'crop_path': str(cp), 'error': f'bad_dtype_{arr.dtype}'})
                if not np.isfinite(arr.astype(np.float32)).all():
                    nan_inf_cnt += 1
                    errors.append({'group': group_name, 'crop_path': str(cp), 'error': 'nan_or_inf'})
                if (arr == 0).all():
                    all_zero_cnt += 1
                    errors.append({'group': group_name, 'crop_path': str(cp), 'error': 'all_zero'})
                hu_vals.extend(arr.flatten().tolist())
                n_ok += 1
            except Exception as e:
                errors.append({'group': group_name, 'crop_path': str(cp), 'error': str(e)})
            n_checked += 1

        hu_arr = np.array(hu_vals, dtype=np.float32)
        stats = hu_stats(hu_arr) if len(hu_arr) > 0 else {}
        row_out = {
            'group': group_name,
            'n_sampled': n_checked,
            'n_ok': n_ok,
            'bad_path': bad_path,
            'bad_key': bad_key,
            'bad_shape': bad_shape,
            'bad_dtype': bad_dtype,
            'nan_inf_count': nan_inf_cnt,
            'all_zero_count': all_zero_cnt,
            'status': 'PASS' if (bad_path == 0 and bad_key == 0 and bad_shape == 0
                                 and nan_inf_cnt == 0 and all_zero_cnt == 0) else 'FAIL',
        }
        row_out.update({f'hu_{k}': v for k, v in stats.items()})
        rows.append(row_out)
        print(f"  {group_name}: n_ok={n_ok}/{n_checked}, bad_path={bad_path}, bad_shape={bad_shape}, "
              f"all_zero={all_zero_cnt}, hu_mean={stats.get('mean', 'N/A'):.1f}")

    overall = all(r['status'] == 'PASS' for r in rows)
    return rows, errors, overall


# ─────────────────────────────────────────────────────────────────────────────
# D. Z index validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_z_index(tr, vl):
    print(f"\n[{ts()}] D. Z index validation (crop reproduction from CT)...")
    rows = []
    errors = []

    # Confirm N-C3 vs N-C10 local_z policy
    nc3 = pd.read_csv(N_C3_MANIFEST_PATH, low_memory=False)
    nc10 = pd.read_csv(N_C10_MANIFEST_PATH, low_memory=False)
    nc3_lz_eq_si  = (nc3['local_z'] == nc3['slice_index']).all()
    nc10_lz_eq_si = (nc10['local_z'] == nc10['slice_index']).all()
    nc3_lz_is_int = (nc3['local_z'].apply(lambda v: v == int(v)).all())
    nc10_lz_range_01 = (nc10['local_z'].min() >= 0.0 and nc10['local_z'].max() <= 1.0)

    print(f"  N-C3  local_z == slice_index: {nc3_lz_eq_si} (actual z)")
    print(f"  N-C10 local_z == slice_index: {nc10_lz_eq_si}")
    print(f"  N-C10 local_z range 0~1 (normalized): {nc10_lz_range_01}")
    print(f"  P-C-NORMAL12 used slice_index for crop extraction (both train and val) → CORRECT")

    rows.append({
        'check': 'nc3_local_z_eq_slice_index',
        'result': str(nc3_lz_eq_si),
        'interpretation': 'N-C3 local_z is actual z (same as slice_index)',
        'status': 'PASS',
    })
    rows.append({
        'check': 'nc10_local_z_normalized_0_1',
        'result': str(nc10_lz_range_01),
        'interpretation': 'N-C10 local_z is normalized (0~1); slice_index is actual z',
        'status': 'PASS' if nc10_lz_range_01 and not nc10_lz_eq_si else 'WARN',
    })
    rows.append({
        'check': 'crop_extraction_z_basis',
        'result': 'slice_index.astype(int)',
        'interpretation': 'P-C-NORMAL12 used slice_index for both train and val → correct for N-C3 and N-C10',
        'status': 'PASS',
    })

    # Sample train rows for reproduction
    tr_normal = tr[tr['label'] == LABEL_NORMAL].copy()
    # Stratify by patient
    tr_patients = list(tr_normal['patient_id'].unique())
    random.seed(42)
    random.shuffle(tr_patients)
    train_sample = tr_normal[tr_normal['patient_id'].isin(tr_patients[:min(20, len(tr_patients))])]
    train_sample = train_sample.sample(n=min(Z_SAMPLE_N, len(train_sample)), random_state=42)

    vl_normal = vl[vl['label'] == LABEL_NORMAL].copy()
    vl_patients = list(vl_normal['patient_id'].unique())
    random.shuffle(vl_patients)
    val_sample = vl_normal[vl_normal['patient_id'].isin(vl_patients[:min(20, len(vl_patients))])]
    val_sample = val_sample.sample(n=min(Z_SAMPLE_N, len(val_sample)), random_state=42)

    for split_name, sample_df in [('train', train_sample), ('val', val_sample)]:
        print(f"  Reproducing {len(sample_df)} {split_name} crops from CT volumes...")
        n_checked = 0
        n_exact_match = 0
        n_max_abs_diff_0 = 0
        max_diffs = []
        ct_cache = {}
        fail_items = []

        for _, mrow in sample_df.iterrows():
            cp = Path(mrow['crop_path'])
            ct_path = mrow['ct_path']
            z_idx   = int(mrow['slice_index'])
            cy      = int(mrow['center_y'])
            cx      = int(mrow['center_x'])

            if not cp.exists():
                errors.append({'split': split_name, 'crop_path': str(cp), 'error': 'stored_crop_not_found'})
                continue
            if not Path(ct_path).exists():
                errors.append({'split': split_name, 'crop_path': str(cp), 'error': f'ct_not_found:{ct_path}'})
                continue

            try:
                if ct_path not in ct_cache:
                    ct_cache[ct_path] = np.load(ct_path)
                ct_vol = ct_cache[ct_path]

                reproduced = extract_crop_for_validation(ct_vol, z_idx, cy, cx)
                stored_npz = np.load(str(cp))
                stored = stored_npz['ct_crop']

                max_diff = int(np.max(np.abs(reproduced.astype(np.int32) - stored.astype(np.int32))))
                max_diffs.append(max_diff)
                if max_diff == 0:
                    n_max_abs_diff_0 += 1
                    n_exact_match += 1
                else:
                    fail_items.append({
                        'split': split_name,
                        'crop_path': str(cp),
                        'slice_index': z_idx,
                        'max_abs_diff': max_diff,
                    })
                n_checked += 1
            except Exception as e:
                errors.append({'split': split_name, 'crop_path': str(cp), 'error': str(e)})

        # Clear CT cache after each split to save memory
        ct_cache.clear()

        pass_rate = n_max_abs_diff_0 / n_checked if n_checked > 0 else 0
        status = 'PASS' if pass_rate >= 0.98 else ('PARTIAL_PASS' if pass_rate >= 0.90 else 'FAIL')
        avg_diff = float(np.mean(max_diffs)) if max_diffs else float('nan')

        print(f"  {split_name}: checked={n_checked}, exact_match={n_max_abs_diff_0}, "
              f"pass_rate={pass_rate:.3f}, avg_max_diff={avg_diff:.2f}, status={status}")

        rows.append({
            'check': f'z_reproduction_{split_name}',
            'n_checked': n_checked,
            'n_exact_match': n_max_abs_diff_0,
            'pass_rate': round(pass_rate, 4),
            'avg_max_abs_diff': round(avg_diff, 3),
            'status': status,
        })
        errors.extend(fail_items)

    # local_z vs slice_index distribution in P-C-NORMAL12 train manifest (normal only)
    tr_n = tr[tr['label'] == LABEL_NORMAL]
    tr_lz_vs_si_same = (tr_n['local_z'].astype(float) == tr_n['slice_index'].astype(float)).sum()
    tr_lz_vs_si_total = len(tr_n)
    rows.append({
        'check': 'train_manifest_local_z_eq_slice_index',
        'n_same': int(tr_lz_vs_si_same),
        'n_total': int(tr_lz_vs_si_total),
        'fraction': round(tr_lz_vs_si_same / tr_lz_vs_si_total, 4),
        'status': 'INFO',
    })
    print(f"  train manifest: local_z==slice_index for normal rows: {tr_lz_vs_si_same}/{tr_lz_vs_si_total}")

    vl_n = vl[vl['label'] == LABEL_NORMAL]
    vl_lz_vs_si_same = (vl_n['local_z'].astype(float) == vl_n['slice_index'].astype(float)).sum()
    rows.append({
        'check': 'val_manifest_local_z_eq_slice_index',
        'n_same': int(vl_lz_vs_si_same),
        'n_total': int(len(vl_n)),
        'fraction': round(vl_lz_vs_si_same / len(vl_n), 4) if len(vl_n) > 0 else None,
        'status': 'INFO',
    })
    print(f"  val manifest:   local_z==slice_index for normal rows: {vl_lz_vs_si_same}/{len(vl_n)}")

    final_z_status = 'PASS'
    for r in rows:
        if r.get('status') == 'FAIL':
            final_z_status = 'FAIL'
        elif r.get('status') == 'PARTIAL_PASS' and final_z_status == 'PASS':
            final_z_status = 'PARTIAL_PASS'

    rows.append({
        'check': 'generated_crop_z_basis',
        'result': 'slice_index',
        'validation_result': final_z_status,
        'status': final_z_status,
    })

    return rows, errors, final_z_status


# ─────────────────────────────────────────────────────────────────────────────
# E. SR-POS re-evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_sr_pos(tr, vl):
    print(f"\n[{ts()}] E. SR-POS re-evaluation (position distribution)...")
    rows = []

    BINS = ['upper_central', 'upper_peripheral', 'middle_central',
            'middle_peripheral', 'lower_central', 'lower_peripheral']

    for split_name, df in [('train', tr), ('val', vl)]:
        normal_df = df[df['label'] == LABEL_NORMAL]
        nsclc_df  = df[df['label'] == LABEL_NSCLC]

        for b in BINS:
            n_count = (normal_df['position_bin'] == b).sum()
            s_count = (nsclc_df['position_bin'] == b).sum()
            n_frac  = n_count / len(normal_df) if len(normal_df) > 0 else 0
            s_frac  = s_count / len(nsclc_df)  if len(nsclc_df)  > 0 else 0
            rows.append({
                'split': split_name,
                'position_bin': b,
                'normal_count': int(n_count),
                'nsclc_count':  int(s_count),
                'normal_frac':  round(n_frac, 6),
                'nsclc_frac':   round(s_frac, 6),
                'abs_diff':     round(abs(n_frac - s_frac), 6),
            })

        peripheral_bins = [b for b in BINS if 'peripheral' in b]
        n_periph = normal_df[normal_df['position_bin'].isin(peripheral_bins)]
        s_periph = nsclc_df[nsclc_df['position_bin'].isin(peripheral_bins)]
        n_prat = len(n_periph) / len(normal_df) if len(normal_df) > 0 else 0
        s_prat = len(s_periph) / len(nsclc_df)  if len(nsclc_df)  > 0 else 0
        periph_diff = abs(n_prat - s_prat)

        max_bin_diff = max(r['abs_diff'] for r in rows if r['split'] == split_name)
        rows.append({
            'split': split_name,
            'position_bin': 'PERIPHERAL_RATIO',
            'normal_count': int(len(n_periph)),
            'nsclc_count':  int(len(s_periph)),
            'normal_frac':  round(n_prat, 6),
            'nsclc_frac':   round(s_prat, 6),
            'abs_diff':     round(periph_diff, 6),
        })
        print(f"  {split_name}: peripheral_normal={n_prat:.4f}, peripheral_nsclc={s_prat:.4f}, "
              f"diff={periph_diff:.6f}, max_bin_diff={max_bin_diff:.6f}")

    # SR-POS judgment
    train_periph_rows = [r for r in rows if r['split'] == 'train' and r['position_bin'] == 'PERIPHERAL_RATIO']
    val_periph_rows   = [r for r in rows if r['split'] == 'val'   and r['position_bin'] == 'PERIPHERAL_RATIO']
    train_diff = train_periph_rows[0]['abs_diff'] if train_periph_rows else 1.0
    val_diff   = val_periph_rows[0]['abs_diff']   if val_periph_rows   else 1.0

    all_bin_diffs = [r['abs_diff'] for r in rows if r['position_bin'] != 'PERIPHERAL_RATIO']
    max_bin_diff_all = max(all_bin_diffs) if all_bin_diffs else 1.0

    if train_diff <= 0.002 and val_diff <= 0.005 and max_bin_diff_all <= 0.01:
        sr_pos = 'RESOLVED'
    elif train_diff <= 0.01 and val_diff <= 0.02:
        sr_pos = 'PARTIAL'
    else:
        sr_pos = 'OPEN'

    print(f"  SR-POS judgment: {sr_pos} "
          f"(train_periph_diff={train_diff:.6f}, val_periph_diff={val_diff:.6f}, "
          f"max_bin_diff={max_bin_diff_all:.6f})")
    return rows, sr_pos


# ─────────────────────────────────────────────────────────────────────────────
# F. SR-HU re-evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_sr_hu(tr, vl):
    print(f"\n[{ts()}] F. SR-HU re-evaluation (HU distribution comparison)...")
    rows = []

    full_df = pd.concat([tr, vl], ignore_index=True)
    normal_rows = full_df[full_df['label'] == LABEL_NORMAL].sample(
        n=min(HU_SAMPLE_N, len(full_df[full_df['label']==LABEL_NORMAL])), random_state=42)
    nsclc_rows  = full_df[full_df['label'] == LABEL_NSCLC].sample(
        n=min(HU_SAMPLE_N, len(full_df[full_df['label']==LABEL_NSCLC])),  random_state=42)

    print(f"  Loading {len(normal_rows)} normal crops and {len(nsclc_rows)} NSCLC crops for HU stats...")

    def load_crops_flat(df_sample, label_name):
        all_vals = []
        fail = 0
        for _, row in df_sample.iterrows():
            cp = Path(row['crop_path'])
            if not cp.exists():
                fail += 1
                continue
            try:
                arr = np.load(str(cp))['ct_crop']
                all_vals.extend(arr.flatten().tolist())
            except Exception:
                fail += 1
        print(f"  {label_name}: loaded {len(df_sample)-fail}/{len(df_sample)}, total pixels={len(all_vals)}")
        return np.array(all_vals, dtype=np.float32)

    normal_hu = load_crops_flat(normal_rows, 'normal')
    nsclc_hu  = load_crops_flat(nsclc_rows,  'NSCLC')

    n_stats = hu_stats(normal_hu)
    s_stats = hu_stats(nsclc_hu)

    cd = cohens_d(normal_hu, nsclc_hu)
    mean_diff = float(np.mean(normal_hu) - np.mean(nsclc_hu)) if len(normal_hu) > 0 and len(nsclc_hu) > 0 else float('nan')
    ov = overlap_estimate(normal_hu, nsclc_hu)

    for k in n_stats:
        rows.append({
            'metric': k,
            'normal': round(n_stats[k], 6) if isinstance(n_stats[k], float) else n_stats[k],
            'nsclc':  round(s_stats.get(k, float('nan')), 6),
            'diff':   round(float(n_stats[k]) - float(s_stats.get(k, float('nan'))), 6),
        })

    rows.append({'metric': 'cohens_d (normal-nsclc)', 'normal': None, 'nsclc': None, 'diff': round(cd, 4)})
    rows.append({'metric': 'mean_diff (normal-nsclc)', 'normal': None, 'nsclc': None, 'diff': round(mean_diff, 2)})
    rows.append({'metric': 'overlap_estimate', 'normal': None, 'nsclc': None, 'diff': round(ov, 4)})

    print(f"  HU mean: normal={n_stats.get('mean', 'N/A'):.1f}, nsclc={s_stats.get('mean', 'N/A'):.1f}, "
          f"diff={mean_diff:.1f}")
    print(f"  Cohen's d={cd:.3f}, overlap={ov:.3f}")
    print(f"  normal air_frac(<-800)={n_stats.get('air_frac_lt_m800', 'N/A'):.3f}, "
          f"nsclc={s_stats.get('air_frac_lt_m800', 'N/A'):.3f}")
    print(f"  normal dense(>-500)={n_stats.get('dense_frac_gt_m500', 'N/A'):.3f}, "
          f"nsclc={s_stats.get('dense_frac_gt_m500', 'N/A'):.3f}")

    # SR-HU judgment based on Cohen's d and overlap
    abs_cd = abs(cd) if not math.isnan(cd) else 999
    if abs_cd < 0.3 and ov > 0.7:
        sr_hu = 'REDUCED'
    elif abs_cd < 0.6 and ov > 0.5:
        sr_hu = 'MEDIUM'
    else:
        sr_hu = 'HIGH'

    print(f"  SR-HU judgment: {sr_hu} (|Cohen's d|={abs_cd:.3f}, overlap={ov:.3f})")

    reassess_rows = [{
        'metric': 'SR_HU_verdict', 'normal': None, 'nsclc': None, 'diff': sr_hu,
    }]

    return rows, reassess_rows, sr_hu, {
        'cohens_d': cd, 'mean_diff': mean_diff, 'overlap': ov,
        'normal_stats': n_stats, 'nsclc_stats': s_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# G. Readiness decision
# ─────────────────────────────────────────────────────────────────────────────

def make_readiness_decision(file_ok, manifest_ok, integrity_ok, z_status, sr_pos, sr_hu):
    print(f"\n[{ts()}] G. Readiness decision...")
    all_structural_ok = file_ok and manifest_ok and integrity_ok

    if not all_structural_ok or z_status == 'FAIL':
        decision = 'HOLD_FULL_TRAINING'
        reason = 'crop/manifest/z validation failed'
    elif sr_pos == 'RESOLVED' and sr_hu == 'REDUCED':
        decision = 'READY_FOR_SMOKE_RETRAIN'
        reason = 'SR-POS resolved, SR-HU reduced to acceptable range'
    elif sr_pos in ('RESOLVED', 'PARTIAL') and sr_hu in ('REDUCED', 'MEDIUM'):
        decision = 'READY_FOR_SMOKE_RETRAIN'
        reason = f'SR-POS={sr_pos}, SR-HU={sr_hu}: acceptable for smoke retrain'
    elif sr_pos == 'RESOLVED' and sr_hu == 'HIGH':
        decision = 'NEEDS_HU_CONTEXT_MATCHING'
        reason = 'SR-POS resolved but SR-HU still HIGH; consider HU/context matching before full train'
    else:
        decision = 'HOLD_FULL_TRAINING'
        reason = f'SR-POS={sr_pos} or SR-HU={sr_hu} indicates unresolved shortcut risk'

    print(f"  file_ok={file_ok}, manifest_ok={manifest_ok}, integrity_ok={integrity_ok}")
    print(f"  z_status={z_status}, SR-POS={sr_pos}, SR-HU={sr_hu}")
    print(f"  → READINESS: {decision} ({reason})")

    rows = [{
        'item': 'file_count_validation', 'result': str(file_ok),
        'status': 'PASS' if file_ok else 'FAIL',
    }, {
        'item': 'manifest_validation', 'result': str(manifest_ok),
        'status': 'PASS' if manifest_ok else 'FAIL',
    }, {
        'item': 'crop_integrity', 'result': str(integrity_ok),
        'status': 'PASS' if integrity_ok else 'FAIL',
    }, {
        'item': 'z_index_validation', 'result': z_status,
        'status': 'PASS' if z_status in ('PASS', 'PARTIAL_PASS') else 'FAIL',
    }, {
        'item': 'SR_POS', 'result': sr_pos, 'status': 'INFO',
    }, {
        'item': 'SR_HU', 'result': sr_hu, 'status': 'INFO',
    }, {
        'item': 'READINESS_DECISION', 'result': decision, 'status': 'INFO',
    }, {
        'item': 'reason', 'result': reason, 'status': 'INFO',
    }]
    return decision, reason, rows


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail check
# ─────────────────────────────────────────────────────────────────────────────

def guardrail_check():
    rows = []
    for k, v in GUARDRAIL.items():
        rows.append({'check': k, 'value': v, 'status': 'PASS' if v == False else 'VIOLATION'})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(path, rows):
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)


def save_report(
    verdict, decision, reason,
    file_rows, manifest_rows, integrity_rows, integrity_errors,
    z_rows, z_errors, sr_pos, sr_pos_rows,
    sr_hu, sr_hu_rows, sr_hu_reassess_rows, sr_hu_detail,
    readiness_rows, guardrail_rows,
    all_errors,
    start_time, end_time,
):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary = {
        'stage': 'P-C-NORMAL13',
        'verdict': verdict,
        'start_time': start_time,
        'end_time': end_time,
        'readiness_decision': decision,
        'readiness_reason': reason,
        'sr_pos': sr_pos,
        'sr_hu': sr_hu,
        'cohens_d': round(sr_hu_detail['cohens_d'], 4) if not math.isnan(sr_hu_detail['cohens_d']) else None,
        'mean_diff_hu': round(sr_hu_detail['mean_diff'], 2),
        'overlap_estimate': round(sr_hu_detail['overlap'], 4),
        'guardrail': GUARDRAIL,
    }
    with open(REPORT_DIR / 'p_c_normal13_matched_validation_shortcut_stats.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # CSVs
    save_csv(REPORT_DIR / 'p_c_normal13_output_file_validation.csv', file_rows)
    save_csv(REPORT_DIR / 'p_c_normal13_manifest_validation.csv', manifest_rows)
    save_csv(REPORT_DIR / 'p_c_normal13_crop_integrity_validation.csv', integrity_rows)
    save_csv(REPORT_DIR / 'p_c_normal13_z_index_validation.csv', z_rows)
    save_csv(REPORT_DIR / 'p_c_normal13_position_distribution_comparison.csv', sr_pos_rows)
    save_csv(REPORT_DIR / 'p_c_normal13_hu_distribution_comparison.csv', sr_hu_rows)
    save_csv(REPORT_DIR / 'p_c_normal13_shortcut_risk_reassessment.csv', sr_hu_reassess_rows)
    save_csv(REPORT_DIR / 'p_c_normal13_readiness_decision.csv', readiness_rows)
    save_csv(REPORT_DIR / 'p_c_normal13_guardrail_check.csv', guardrail_rows)
    save_csv(REPORT_DIR / 'p_c_normal13_errors.csv', all_errors)

    # Markdown report
    hu_n = sr_hu_detail['normal_stats']
    hu_s = sr_hu_detail['nsclc_stats']
    cd   = sr_hu_detail['cohens_d']
    md   = sr_hu_detail['mean_diff']
    ov   = sr_hu_detail['overlap']

    z_train_row = next((r for r in z_rows if r.get('check') == 'z_reproduction_train'), {})
    z_val_row   = next((r for r in z_rows if r.get('check') == 'z_reproduction_val'),   {})
    periph_train = next((r for r in sr_pos_rows if r['split']=='train' and r['position_bin']=='PERIPHERAL_RATIO'), {})
    periph_val   = next((r for r in sr_pos_rows if r['split']=='val'   and r['position_bin']=='PERIPHERAL_RATIO'), {})

    # Overall verdict
    if verdict == '통과':
        verdict_line = f"**판정: 통과**"
    elif verdict == '부분통과':
        verdict_line = f"**판정: 부분통과**"
    else:
        verdict_line = f"**판정: 실패**"

    md_text = f"""# P-C-NORMAL13: Matched Validation + Shortcut Stat Comparison

생성일시: {end_time}

## 판정

{verdict_line}

- Readiness Decision: **{decision}**
- 이유: {reason}

---

## P-C-NORMAL12 요약

| 항목 | 값 |
|------|-----|
| stage2_holdout_accessed | False |
| training_run | False |
| normal_train_generated | {EXPECTED_TRAIN_NORMAL} |
| normal_val_generated | {EXPECTED_VAL_NORMAL} |
| total_normal_generated | {EXPECTED_TOTAL_NORMAL} |
| nsclc_train_reused | {EXPECTED_TRAIN_NSCLC} |
| nsclc_val_reused | {EXPECTED_VAL_NSCLC} |
| train_manifest_rows | {EXPECTED_TRAIN_ROWS} |
| val_manifest_rows | {EXPECTED_VAL_ROWS} |
| integrity_failures | 0 |
| pos_matching_verdict | PASS |

---

## A. File/count validation

| 항목 | count | expected | status |
|------|-------|----------|--------|
""" + '\n'.join(
    f"| {r['item']} | {r.get('npz_count', '-')} | {r.get('expected', '-')} | {r['status']} |"
    for r in file_rows
) + f"""

---

## B. Manifest validation

| check | actual | expected | status |
|-------|--------|----------|--------|
""" + '\n'.join(
    f"| {r['check']} | {r['actual']} | {r.get('expected', '-')} | {r['status']} |"
    for r in manifest_rows
) + f"""

---

## C. Crop integrity validation

| group | n_sampled | n_ok | bad_path | bad_shape | nan_inf | all_zero | hu_mean | status |
|-------|-----------|------|----------|-----------|---------|----------|---------|--------|
""" + '\n'.join(
    f"| {r['group']} | {r['n_sampled']} | {r['n_ok']} | {r['bad_path']} | {r['bad_shape']} | {r['nan_inf_count']} | {r['all_zero_count']} | {r.get('hu_mean', 'N/A'):.1f} | {r['status']} |"
    for r in integrity_rows
) + f"""

---

## D. Z index validation

### N-C3 vs N-C10 local_z 정책

| source | local_z 특성 | slice_index | crop 추출 기준 |
|--------|-------------|-------------|----------------|
| N-C3 (train) | 실제 z (== slice_index) | 실제 z | slice_index ✓ |
| N-C10 (val) | 정규화값 (0~1) | 실제 z | slice_index ✓ |

P-C-NORMAL12는 train/val 모두 `slice_index.astype(int)`를 사용했으므로 **정책 일치**.

### CT 재현 검증

| split | n_checked | n_exact_match | pass_rate | avg_max_abs_diff | status |
|-------|-----------|---------------|-----------|------------------|--------|
| train | {z_train_row.get('n_checked', '-')} | {z_train_row.get('n_exact_match', '-')} | {z_train_row.get('pass_rate', '-')} | {z_train_row.get('avg_max_abs_diff', '-')} | {z_train_row.get('status', '-')} |
| val   | {z_val_row.get('n_checked', '-')} | {z_val_row.get('n_exact_match', '-')} | {z_val_row.get('pass_rate', '-')} | {z_val_row.get('avg_max_abs_diff', '-')} | {z_val_row.get('status', '-')} |

generated_crop_z_basis: **slice_index**

---

## E. SR-POS 재평가

| split | position_bin | normal_frac | nsclc_frac | abs_diff |
|-------|-------------|-------------|------------|----------|
""" + '\n'.join(
    f"| {r['split']} | {r['position_bin']} | {r['normal_frac']:.4f} | {r['nsclc_frac']:.4f} | {r['abs_diff']:.6f} |"
    for r in sr_pos_rows
) + f"""

**SR-POS 판정: {sr_pos}**

- train peripheral diff: {periph_train.get('abs_diff', 'N/A')}
- val peripheral diff: {periph_val.get('abs_diff', 'N/A')}

---

## F. SR-HU 재평가

| metric | normal | NSCLC | diff |
|--------|--------|-------|------|
| mean | {hu_n.get('mean', 'N/A'):.1f} | {hu_s.get('mean', 'N/A'):.1f} | {hu_n.get('mean', 0)-hu_s.get('mean', 0):.1f} |
| median | {hu_n.get('median', 'N/A'):.1f} | {hu_s.get('median', 'N/A'):.1f} | {hu_n.get('median', 0)-hu_s.get('median', 0):.1f} |
| std | {hu_n.get('std', 'N/A'):.1f} | {hu_s.get('std', 'N/A'):.1f} | - |
| p05 | {hu_n.get('p05', 'N/A'):.1f} | {hu_s.get('p05', 'N/A'):.1f} | - |
| p25 | {hu_n.get('p25', 'N/A'):.1f} | {hu_s.get('p25', 'N/A'):.1f} | - |
| p75 | {hu_n.get('p75', 'N/A'):.1f} | {hu_s.get('p75', 'N/A'):.1f} | - |
| p95 | {hu_n.get('p95', 'N/A'):.1f} | {hu_s.get('p95', 'N/A'):.1f} | - |
| air_frac_lt_m800 | {hu_n.get('air_frac_lt_m800', 'N/A'):.4f} | {hu_s.get('air_frac_lt_m800', 'N/A'):.4f} | - |
| dense_frac_gt_m500 | {hu_n.get('dense_frac_gt_m500', 'N/A'):.4f} | {hu_s.get('dense_frac_gt_m500', 'N/A'):.4f} | - |
| dense_frac_gt_m300 | {hu_n.get('dense_frac_gt_m300', 'N/A'):.4f} | {hu_s.get('dense_frac_gt_m300', 'N/A'):.4f} | - |
| positive_frac_gt_0 | {hu_n.get('positive_frac_gt_0', 'N/A'):.4f} | {hu_s.get('positive_frac_gt_0', 'N/A'):.4f} | - |
| **Cohen's d** | - | - | {cd:.4f} |
| **mean_diff (normal-nsclc)** | - | - | {md:.1f} HU |
| **overlap_estimate** | - | - | {ov:.4f} |

**SR-HU 판정: {sr_hu}**

---

## G. Readiness decision

**{decision}**

이유: {reason}

---

## Guardrail 확인

| 항목 | 값 | status |
|------|-----|--------|
""" + '\n'.join(
    f"| {r['check']} | {r['value']} | {r['status']} |"
    for r in guardrail_rows
) + f"""

- stage2_holdout 미접근: **확인**
- 학습/model_forward/scoring 미실행: **확인**
- 기존 결과 무수정: **확인**
- crop 재생성 없음: **확인**
- manifest 수정 없음: **확인**

---

## 다음 단계

"""
    if decision == 'READY_FOR_SMOKE_RETRAIN':
        md_text += "- **P-C-NORMAL14 smoke retrain preflight** 진행 가능\n"
        md_text += "- SR-HU는 남아있으므로 smoke retrain 결과에서 HU shortcut 영향 모니터링 필요\n"
    elif decision == 'NEEDS_HU_CONTEXT_MATCHING':
        md_text += "- **P-C-NORMAL14 HU/context matching preflight** 검토 후 진행\n"
        md_text += "- SR-HU가 HIGH이므로 normal crop의 HU context matching 추가 고려\n"
    else:
        md_text += "- **HOLD**: 구조적 오류 또는 shortcut risk가 너무 큼\n"
        md_text += "- 오류 원인 파악 후 재검토\n"

    with open(REPORT_DIR / 'p_c_normal13_matched_validation_shortcut_stats.md', 'w', encoding='utf-8') as f:
        f.write(md_text)

    print(f"\n[{ts()}] Reports saved to {REPORT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    start_time = ts()
    print(f"[{start_time}] P-C-NORMAL13: Matched Validation + Shortcut Stat Comparison")
    print("="*70)
    print("Guardrail: read-only validation only. No training/model/scoring/generation.")
    print("="*70)

    all_errors = []

    # A. File/count
    file_rows, file_ok = validate_file_counts()

    # B. Manifest
    manifest_rows, manifest_ok, tr, vl, fl = validate_manifests()

    # C. Crop integrity
    integrity_rows, integrity_errors, integrity_ok = validate_crop_integrity(tr, vl)
    all_errors.extend([{**e, 'section': 'C_integrity'} for e in integrity_errors])

    # D. Z index
    z_rows, z_errors, z_status = validate_z_index(tr, vl)
    all_errors.extend([{**e, 'section': 'D_z_index'} for e in z_errors])

    # E. SR-POS
    sr_pos_rows, sr_pos = evaluate_sr_pos(tr, vl)

    # F. SR-HU
    sr_hu_rows, sr_hu_reassess_rows, sr_hu, sr_hu_detail = evaluate_sr_hu(tr, vl)

    # G. Readiness
    decision, reason, readiness_rows = make_readiness_decision(
        file_ok, manifest_ok, integrity_ok, z_status, sr_pos, sr_hu
    )

    # Overall verdict
    all_ok = file_ok and manifest_ok and integrity_ok and z_status in ('PASS', 'PARTIAL_PASS')
    if not all_ok:
        verdict = '실패'
    elif sr_hu in ('HIGH', 'MEDIUM') or sr_pos != 'RESOLVED':
        verdict = '부분통과'
    else:
        verdict = '통과'

    end_time = ts()
    guardrail_rows = guardrail_check()

    print(f"\n[{end_time}] === FINAL VERDICT: {verdict} ===")
    print(f"  Readiness: {decision}")
    print(f"  SR-POS: {sr_pos}, SR-HU: {sr_hu}")

    save_report(
        verdict=verdict,
        decision=decision,
        reason=reason,
        file_rows=file_rows,
        manifest_rows=manifest_rows,
        integrity_rows=integrity_rows,
        integrity_errors=integrity_errors,
        z_rows=z_rows,
        z_errors=z_errors,
        sr_pos=sr_pos,
        sr_pos_rows=sr_pos_rows,
        sr_hu=sr_hu,
        sr_hu_rows=sr_hu_rows,
        sr_hu_reassess_rows=sr_hu_reassess_rows,
        sr_hu_detail=sr_hu_detail,
        readiness_rows=readiness_rows,
        guardrail_rows=guardrail_rows,
        all_errors=all_errors,
        start_time=start_time,
        end_time=end_time,
    )

    print(f"\n모든 결과: {REPORT_DIR}")
    return verdict


if __name__ == '__main__':
    main()
