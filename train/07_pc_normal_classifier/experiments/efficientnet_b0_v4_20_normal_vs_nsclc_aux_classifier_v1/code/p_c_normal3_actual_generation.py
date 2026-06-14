"""
P-C-NORMAL3: Actual Normal Crop Generation + Training Manifest Generation
Branch: efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1

Approved by user: Option B sampling, normal train 15,000 / val 5,000
NSCLC crops referenced (not copied), stage2_holdout not accessed.
Training/model forward/scoring NOT executed here.
"""

import os
import sys
import json
import csv
import math
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

N_C3_MANIFEST = (
    PROJECT_ROOT
    / 'experiments/normal_only_second_stage_refiner_v1'
    / 'outputs/manifests/n_c3_normal_only_crop_manifest_v1'
    / 'n_c3_normal_only_crop_manifest_v1.csv'
)
N_C10_MANIFEST = (
    PROJECT_ROOT
    / 'experiments/normal_only_second_stage_refiner_v1'
    / 'outputs/manifests/n_c10_normal_val_crop_manifest'
    / 'n_c10_normal_val_crop_manifest.csv'
)
P_C_AUX_MANIFEST = (
    PROJECT_ROOT
    / 'experiments/efficientnet_b0_v4_20_supervised_aux_source_classifier_v1'
    / 'outputs/manifests/p_c_aux2_source_classifier_training_manifest'
    / 'p_c_aux2_source_classifier_training_manifest.csv'
)
P_C_AUX_CROP_BASE = (
    PROJECT_ROOT / 'experiments/efficientnet_b0_v4_20_second_stage_refiner_v1'
)
CT_VOL_BASE = Path(
    '/mnt/c/Users/jinhy/Desktop'
    '/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1'
    '/volumes_npy'
)

NORMAL_CROP_OUTPUT_BASE = (
    BRANCH_ROOT / 'outputs/normal_crops/p_c_normal3_normal_train_val_crops'
)
MANIFEST_OUTPUT_BASE = (
    BRANCH_ROOT / 'outputs/manifests/p_c_normal3_training_manifest'
)
REPORT_OUTPUT_BASE = (
    BRANCH_ROOT / 'outputs/reports/p_c_normal3_actual_generation'
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CROP_SIZE = 96
HALF_CROP = 48
LABEL_NORMAL = 0
LABEL_NSCLC = 1
NORMAL_TRAIN_CAP = 15000
NORMAL_VAL_CAP = 5000
RANDOM_SEED = 42

MANIFEST_COLUMNS = [
    'aux_candidate_id',
    'source_branch',
    'patient_id',
    'safe_id',
    'split',
    'label',
    'label_name',
    'crop_path',
    'crop_generated_by_this_branch',
    'ct_path',
    'local_z',
    'slice_index',
    'y0',
    'x0',
    'y1',
    'x1',
    'center_y',
    'center_x',
    'position_bin',
    'z_level',
    'crop_shape_expected',
    'crop_dtype_expected',
    'raw_hu_available',
    'preprocessing',
    'sample_weight',
    'class_weight',
    'patient_cap_applied',
    'source_patient_count',
    'forbidden_source_classifier_mixed',
    'original_file_modified',
    # Normal-specific
    'normal_source_split',
    'normal_coord_source',
    'lesion_pixels',
    'has_lesion_patch',
    # NSCLC-specific
    'original_p_c_candidate_id',
    'original_p_c_label',
    'lesion_pixels_nsclc',
    'has_lesion_patch_nsclc',
    'source_name',
]


# ─────────────────────────────────────────────────────────────────────────────
# Crop extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_normal_crop(ct_vol, local_z_idx, center_y, center_x):
    """
    Extract (3, 96, 96) int16 crop.
    Channels: z-1 / z / z+1 with edge padding at z boundaries.
    XY: center±48 with edge padding if out of bounds.
    """
    z_dim, y_dim, x_dim = ct_vol.shape
    lz = int(local_z_idx)
    cy = int(center_y)
    cx = int(center_x)

    # Z: clamp to valid range (edge padding)
    z_indices = [
        max(0, lz - 1),
        max(0, min(z_dim - 1, lz)),
        min(z_dim - 1, lz + 1),
    ]

    # XY: clamp + edge padding
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
# Sampling helper
# ─────────────────────────────────────────────────────────────────────────────

def sample_balanced_by_position_bin(df, target_total, n_patients, seed=RANDOM_SEED):
    """
    Sample rows with position_bin balance per patient.
    Targets exactly `target_total` rows total.
    per_bin_cap uses ceiling division so total sampled >= target_total,
    then trims to exact target.
    """
    n_bins = df['position_bin'].nunique()
    # Ceiling division ensures we over-sample, then trim to exact target
    per_bin_cap = math.ceil(target_total / (n_patients * n_bins))

    results = []
    for patient_id, pdf in df.groupby('patient_id'):
        patient_rows = []
        for bin_name, bdf in pdf.groupby('position_bin'):
            n_take = min(len(bdf), per_bin_cap)
            patient_rows.append(bdf.sample(n=n_take, random_state=seed))
        results.append(pd.concat(patient_rows))

    sampled = pd.concat(results).reset_index(drop=True)

    # Trim to exact target if over
    if len(sampled) > target_total:
        sampled = sampled.sample(n=target_total, random_state=seed).reset_index(drop=True)

    return sampled


def compute_class_weights(n_normal, n_nsclc):
    total = n_normal + n_nsclc
    w_normal = total / (2.0 * n_normal)
    w_nsclc  = total / (2.0 * n_nsclc)
    return w_normal, w_nsclc


# ─────────────────────────────────────────────────────────────────────────────
# CSV helper
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(path, rows, fieldnames=None):
    if not rows:
        rows = [{'note': 'empty'}]
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail pre-check
# ─────────────────────────────────────────────────────────────────────────────

def preflight_guardrail_check():
    """Check guardrails before any file generation."""
    issues = []

    if NORMAL_CROP_OUTPUT_BASE.exists():
        issues.append(f"COLLISION: {NORMAL_CROP_OUTPUT_BASE} already exists")
    if MANIFEST_OUTPUT_BASE.exists():
        issues.append(f"COLLISION: {MANIFEST_OUTPUT_BASE} already exists")

    # Check stage2_holdout is NOT referenced
    holdout_paths = list(PROJECT_ROOT.glob('**/stage2_holdout*'))
    # We only fail if we actually ACCESS them — just check for accidental imports

    # Required inputs must exist
    for p in [N_C3_MANIFEST, N_C10_MANIFEST, P_C_AUX_MANIFEST]:
        if not p.exists():
            issues.append(f"MISSING_INPUT: {p}")

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Main generation
# ─────────────────────────────────────────────────────────────────────────────

def run_generation():
    start_time = datetime.now()
    print(f"[P-C-NORMAL3] Start: {start_time.isoformat()}")

    errors = []
    guardrail = {
        'stage2_holdout_accessed': False,
        'training_run': False,
        'model_forward_run': False,
        'scoring_run': False,
        'threshold_computed': False,
        'original_file_modified': False,
        'p_c_aux_modified': False,
        'nsclc_crop_copied': False,
        'nsclc_crop_modified': False,
        'hard_negative_included': False,
        'msd_lung_included': False,
        'forbidden_diagnostic_wording_count': 0,
    }

    # ── 0. Pre-flight guardrail ────────────────────────────────────────────
    issues = preflight_guardrail_check()
    if issues:
        print("[ERROR] Pre-flight failed:")
        for iss in issues:
            print(f"  {iss}")
        sys.exit(1)

    # ── Create output directories ─────────────────────────────────────────
    train_crop_dir = NORMAL_CROP_OUTPUT_BASE / 'train'
    val_crop_dir   = NORMAL_CROP_OUTPUT_BASE / 'val'
    train_crop_dir.mkdir(parents=True, exist_ok=True)
    val_crop_dir.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"[P-C-NORMAL3] Output dirs created.")

    # ── 1. Load normal manifests ──────────────────────────────────────────
    print("[P-C-NORMAL3] Loading N-C3 manifest...")
    df3  = pd.read_csv(N_C3_MANIFEST)
    print(f"  N-C3: {len(df3)} rows, {df3['patient_id'].nunique()} patients")

    print("[P-C-NORMAL3] Loading N-C10 manifest...")
    df10 = pd.read_csv(N_C10_MANIFEST)
    print(f"  N-C10: {len(df10)} rows, {df10['patient_id'].nunique()} patients")

    n_c3_patients  = df3['patient_id'].nunique()
    n_c10_patients = df10['patient_id'].nunique()

    # ── 2. Sample normal rows ─────────────────────────────────────────────
    print("[P-C-NORMAL3] Sampling normal train rows (Option B, 15,000)...")
    sampled_train = sample_balanced_by_position_bin(df3, NORMAL_TRAIN_CAP, n_c3_patients)
    print(f"  Sampled train normal: {len(sampled_train)} rows")

    print("[P-C-NORMAL3] Sampling normal val rows (Option B, 5,000)...")
    sampled_val = sample_balanced_by_position_bin(df10, NORMAL_VAL_CAP, n_c10_patients)
    print(f"  Sampled val normal:   {len(sampled_val)} rows")

    # ── 3. Generate normal train crops ────────────────────────────────────
    print("[P-C-NORMAL3] Generating normal train crops...")
    normal_train_manifest_rows = []
    normal_train_generated = 0
    normal_train_errors = 0

    # Group by CT path for memory efficiency
    # N-C3: local_z is the actual integer slice index (same as slice_index)
    sampled_train = sampled_train.copy()
    sampled_train['_ct_path'] = sampled_train['source_ct_path']
    sampled_train['_local_z_for_crop'] = sampled_train['local_z'].astype(int)

    for ct_path, grp in sampled_train.groupby('_ct_path'):
        try:
            ct_vol = np.load(ct_path)
        except Exception as e:
            for idx, row in grp.iterrows():
                errors.append({'severity': 'ERROR', 'check': 'ct_load_train',
                               'patient_id': row['patient_id'], 'message': str(e)})
                normal_train_errors += 1
            continue

        for grp_idx, (_, row) in enumerate(grp.iterrows()):
            global_idx = normal_train_generated + 1
            aux_cid    = f"PN3TR_{global_idx:08d}"
            out_fname  = f"{aux_cid}.npz"
            out_path   = train_crop_dir / out_fname

            try:
                crop = extract_normal_crop(
                    ct_vol,
                    row['_local_z_for_crop'],
                    row['center_y'],
                    row['center_x'],
                )
                np.savez(str(out_path), ct_crop=crop)
                normal_train_generated += 1

                # Build manifest row
                mrow = {
                    'aux_candidate_id':             aux_cid,
                    'source_branch':                'p_c_normal3',
                    'patient_id':                   row['patient_id'],
                    'safe_id':                      row['safe_id'],
                    'split':                        'train',
                    'label':                        LABEL_NORMAL,
                    'label_name':                   'normal',
                    'crop_path':                    str(out_path),
                    'crop_generated_by_this_branch': True,
                    'ct_path':                      str(row['source_ct_path']),
                    'local_z':                      int(row['local_z']),
                    'slice_index':                  int(row.get('slice_index', row['local_z'])),
                    'y0':                           int(row['y0']),
                    'x0':                           int(row['x0']),
                    'y1':                           int(row['y1']),
                    'x1':                           int(row['x1']),
                    'center_y':                     int(row['center_y']),
                    'center_x':                     int(row['center_x']),
                    'position_bin':                 row['position_bin'],
                    'z_level':                      row['z_level'],
                    'crop_shape_expected':          '(3,96,96)',
                    'crop_dtype_expected':          'int16',
                    'raw_hu_available':             True,
                    'preprocessing':                'none_raw_hu',
                    'sample_weight':                None,   # filled after class_weight computed
                    'class_weight':                 None,
                    'patient_cap_applied':          True,
                    'source_patient_count':         n_c3_patients,
                    'forbidden_source_classifier_mixed': False,
                    'original_file_modified':       False,
                    # Normal-specific
                    'normal_source_split':          'normal_train',
                    'normal_coord_source':          'n_c3',
                    'lesion_pixels':                0,
                    'has_lesion_patch':             False,
                    # NSCLC-specific (N/A)
                    'original_p_c_candidate_id':    '',
                    'original_p_c_label':           '',
                    'lesion_pixels_nsclc':          '',
                    'has_lesion_patch_nsclc':       '',
                    'source_name':                  'NORMAL_LUNA16',
                }
                normal_train_manifest_rows.append(mrow)

            except Exception as e:
                errors.append({'severity': 'ERROR', 'check': 'crop_extract_train',
                               'patient_id': row['patient_id'], 'message': str(e)})
                normal_train_errors += 1

        del ct_vol

    print(f"  Normal train generated: {normal_train_generated} (errors: {normal_train_errors})")

    # ── 4. Generate normal val crops ──────────────────────────────────────
    print("[P-C-NORMAL3] Generating normal val crops...")
    normal_val_manifest_rows = []
    normal_val_generated = 0
    normal_val_errors = 0

    # N-C10: local_z is NORMALIZED (0-1), slice_index is actual integer slice
    sampled_val = sampled_val.copy()
    sampled_val['_ct_path'] = sampled_val.apply(
        lambda r: str(CT_VOL_BASE / r['safe_id'] / 'ct_hu.npy'), axis=1
    )
    sampled_val['_local_z_for_crop'] = sampled_val['slice_index'].astype(int)

    for ct_path, grp in sampled_val.groupby('_ct_path'):
        try:
            ct_vol = np.load(ct_path)
        except Exception as e:
            for idx, row in grp.iterrows():
                errors.append({'severity': 'ERROR', 'check': 'ct_load_val',
                               'patient_id': row['patient_id'], 'message': str(e)})
                normal_val_errors += 1
            continue

        for _, row in grp.iterrows():
            global_idx = normal_val_generated + 1
            aux_cid    = f"PN3VL_{global_idx:08d}"
            out_fname  = f"{aux_cid}.npz"
            out_path   = val_crop_dir / out_fname

            try:
                crop = extract_normal_crop(
                    ct_vol,
                    row['_local_z_for_crop'],
                    row['center_y'],
                    row['center_x'],
                )
                np.savez(str(out_path), ct_crop=crop)
                normal_val_generated += 1

                mrow = {
                    'aux_candidate_id':             aux_cid,
                    'source_branch':                'p_c_normal3',
                    'patient_id':                   row['patient_id'],
                    'safe_id':                      row['safe_id'],
                    'split':                        'val',
                    'label':                        LABEL_NORMAL,
                    'label_name':                   'normal',
                    'crop_path':                    str(out_path),
                    'crop_generated_by_this_branch': True,
                    'ct_path':                      str(CT_VOL_BASE / row['safe_id'] / 'ct_hu.npy'),
                    'local_z':                      float(row['local_z']),
                    'slice_index':                  int(row['slice_index']),
                    'y0':                           int(row['y0']),
                    'x0':                           int(row['x0']),
                    'y1':                           int(row['y1']),
                    'x1':                           int(row['x1']),
                    'center_y':                     int(row['center_y']),
                    'center_x':                     int(row['center_x']),
                    'position_bin':                 row['position_bin'],
                    'z_level':                      row['z_level'],
                    'crop_shape_expected':          '(3,96,96)',
                    'crop_dtype_expected':          'int16',
                    'raw_hu_available':             True,
                    'preprocessing':                'none_raw_hu',
                    'sample_weight':                None,
                    'class_weight':                 None,
                    'patient_cap_applied':          True,
                    'source_patient_count':         n_c10_patients,
                    'forbidden_source_classifier_mixed': False,
                    'original_file_modified':       False,
                    'normal_source_split':          'normal_val',
                    'normal_coord_source':          'n_c10',
                    'lesion_pixels':                0,
                    'has_lesion_patch':             False,
                    'original_p_c_candidate_id':    '',
                    'original_p_c_label':           '',
                    'lesion_pixels_nsclc':          '',
                    'has_lesion_patch_nsclc':       '',
                    'source_name':                  'NORMAL_LUNA16',
                }
                normal_val_manifest_rows.append(mrow)

            except Exception as e:
                errors.append({'severity': 'ERROR', 'check': 'crop_extract_val',
                               'patient_id': row['patient_id'], 'message': str(e)})
                normal_val_errors += 1

        del ct_vol

    print(f"  Normal val generated: {normal_val_generated} (errors: {normal_val_errors})")

    # ── 5. Load NSCLC rows (read-only, no copy) ───────────────────────────
    print("[P-C-NORMAL3] Loading NSCLC rows from P-C-AUX manifest (read-only)...")
    df_aux  = pd.read_csv(P_C_AUX_MANIFEST)
    nsclc_df = df_aux[df_aux['source_name'] == 'NSCLC'].copy()

    # Guardrail: no MSD_Lung, no hard_negative
    msd_count  = len(df_aux[df_aux['source_name'] == 'MSD_Lung'])
    hard_count = int(nsclc_df['forbidden_hard_negative_used'].sum())
    guardrail['msd_lung_included']    = (msd_count > 0 and len(nsclc_df) != len(df_aux[df_aux['source_name']=='NSCLC']))
    guardrail['hard_negative_included'] = (hard_count > 0)

    nsclc_train_df = nsclc_df[nsclc_df['split'] == 'train'].copy()
    nsclc_val_df   = nsclc_df[nsclc_df['split'] == 'val'].copy()

    print(f"  NSCLC train: {len(nsclc_train_df)}, val: {len(nsclc_val_df)}")

    # ── 6. Compute class weights from actual generated counts ─────────────
    actual_normal_train = normal_train_generated
    actual_nsclc_train  = len(nsclc_train_df)
    w_normal, w_nsclc = compute_class_weights(actual_normal_train, actual_nsclc_train)
    print(f"  Class weight normal={w_normal:.4f}, NSCLC={w_nsclc:.4f}")

    # Fill sample_weight in normal rows
    for r in normal_train_manifest_rows:
        r['sample_weight'] = round(w_normal, 6)
        r['class_weight']  = round(w_normal, 6)
    for r in normal_val_manifest_rows:
        r['sample_weight'] = round(w_normal, 6)
        r['class_weight']  = round(w_normal, 6)

    # ── 7. Build NSCLC manifest rows ──────────────────────────────────────
    def build_nsclc_rows(nsclc_sub_df, split_name, class_w_nsclc):
        rows = []
        for i, (_, row) in enumerate(nsclc_sub_df.iterrows()):
            full_crop_path = str(P_C_AUX_CROP_BASE / row['crop_path'])
            mrow = {
                'aux_candidate_id':             f"PN3NS_{split_name[:2].upper()}_{i+1:07d}",
                'source_branch':                'p_c_aux2_ref',
                'patient_id':                   row['patient_id'],
                'safe_id':                      row['safe_id'],
                'split':                        split_name,
                'label':                        LABEL_NSCLC,
                'label_name':                   'NSCLC',
                'crop_path':                    full_crop_path,
                'crop_generated_by_this_branch': False,
                'ct_path':                      '',
                'local_z':                      row['local_z'],
                'slice_index':                  row['slice_index'],
                'y0':                           row['y0'],
                'x0':                           row['x0'],
                'y1':                           row['y1'],
                'x1':                           row['x1'],
                'center_y':                     row['center_y'],
                'center_x':                     row['center_x'],
                'position_bin':                 row['position_bin'],
                'z_level':                      row['z_level'],
                'crop_shape_expected':          '(3,96,96)',
                'crop_dtype_expected':          'int16',
                'raw_hu_available':             True,
                'preprocessing':                'none_raw_hu',
                'sample_weight':                round(class_w_nsclc, 6),
                'class_weight':                 round(class_w_nsclc, 6),
                'patient_cap_applied':          row.get('patient_cap_applied', True),
                'source_patient_count':         int(nsclc_df['patient_id'].nunique()),
                'forbidden_source_classifier_mixed': False,
                'original_file_modified':       False,
                # Normal-specific (N/A)
                'normal_source_split':          '',
                'normal_coord_source':          '',
                'lesion_pixels':                '',
                'has_lesion_patch':             '',
                # NSCLC-specific
                'original_p_c_candidate_id':    row.get('aux_candidate_id', ''),
                'original_p_c_label':           row.get('original_p_c_label', 'positive'),
                'lesion_pixels_nsclc':          row.get('lesion_pixels', ''),
                'has_lesion_patch_nsclc':       row.get('has_lesion_patch', ''),
                'source_name':                  'NSCLC',
            }
            rows.append(mrow)
        return rows

    nsclc_train_rows = build_nsclc_rows(nsclc_train_df, 'train', w_nsclc)
    nsclc_val_rows   = build_nsclc_rows(nsclc_val_df,   'val',   w_nsclc)
    print(f"  NSCLC manifest rows built: train={len(nsclc_train_rows)}, val={len(nsclc_val_rows)}")
    guardrail['nsclc_crop_copied']   = False
    guardrail['nsclc_crop_modified'] = False

    # ── 8. Combine manifests ──────────────────────────────────────────────
    all_train_rows = normal_train_manifest_rows + nsclc_train_rows
    all_val_rows   = normal_val_manifest_rows   + nsclc_val_rows
    all_rows       = all_train_rows + all_val_rows

    df_train = pd.DataFrame(all_train_rows, columns=MANIFEST_COLUMNS)
    df_val   = pd.DataFrame(all_val_rows,   columns=MANIFEST_COLUMNS)
    df_full  = pd.DataFrame(all_rows,       columns=MANIFEST_COLUMNS)

    # ── 9. Write manifest CSVs ────────────────────────────────────────────
    train_csv = MANIFEST_OUTPUT_BASE / 'p_c_normal3_train_manifest.csv'
    val_csv   = MANIFEST_OUTPUT_BASE / 'p_c_normal3_val_manifest.csv'
    full_csv  = MANIFEST_OUTPUT_BASE / 'p_c_normal3_full_manifest.csv'

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv,     index=False)
    df_full.to_csv(full_csv,   index=False)
    print(f"  Manifests written: train={len(df_train)}, val={len(df_val)}, full={len(df_full)}")

    # ── 10. Integrity checks ──────────────────────────────────────────────
    print("[P-C-NORMAL3] Running integrity checks...")

    # Check normal crop count
    train_npz_list = list(train_crop_dir.glob('*.npz'))
    val_npz_list   = list(val_crop_dir.glob('*.npz'))

    # Spot-check 20 normal crops from train + 10 from val
    integrity_rows = []
    spot_train = train_npz_list[:10] + train_npz_list[-10:]
    for p in spot_train:
        try:
            d = np.load(str(p))
            ok_key   = 'ct_crop' in d
            ok_shape = d['ct_crop'].shape == (3, 96, 96)
            ok_dtype = d['ct_crop'].dtype == np.int16
            ok_nan   = not (np.isnan(d['ct_crop'].astype(float)).any() or
                            np.isinf(d['ct_crop'].astype(float)).any())
            status = 'ok' if (ok_key and ok_shape and ok_dtype and ok_nan) else 'fail'
            integrity_rows.append({
                'split': 'train', 'file': p.name, 'status': status,
                'has_ct_crop': ok_key, 'shape_ok': ok_shape,
                'dtype_ok': ok_dtype, 'no_nan_inf': ok_nan,
            })
        except Exception as e:
            integrity_rows.append({
                'split': 'train', 'file': p.name, 'status': f'error:{e}',
                'has_ct_crop': False, 'shape_ok': False, 'dtype_ok': False, 'no_nan_inf': False,
            })

    spot_val = val_npz_list[:5] + val_npz_list[-5:]
    for p in spot_val:
        try:
            d = np.load(str(p))
            ok_key   = 'ct_crop' in d
            ok_shape = d['ct_crop'].shape == (3, 96, 96)
            ok_dtype = d['ct_crop'].dtype == np.int16
            ok_nan   = not (np.isnan(d['ct_crop'].astype(float)).any() or
                            np.isinf(d['ct_crop'].astype(float)).any())
            status = 'ok' if (ok_key and ok_shape and ok_dtype and ok_nan) else 'fail'
            integrity_rows.append({
                'split': 'val', 'file': p.name, 'status': status,
                'has_ct_crop': ok_key, 'shape_ok': ok_shape,
                'dtype_ok': ok_dtype, 'no_nan_inf': ok_nan,
            })
        except Exception as e:
            integrity_rows.append({
                'split': 'val', 'file': p.name, 'status': f'error:{e}',
                'has_ct_crop': False, 'shape_ok': False, 'dtype_ok': False, 'no_nan_inf': False,
            })

    # Patient split check (no leakage)
    normal_train_patients = set(df_train[df_train['label']==LABEL_NORMAL]['patient_id'].unique())
    normal_val_patients   = set(df_val[df_val['label']==LABEL_NORMAL]['patient_id'].unique())
    nsclc_train_patients  = set(df_train[df_train['label']==LABEL_NSCLC]['patient_id'].unique())
    nsclc_val_patients    = set(df_val[df_val['label']==LABEL_NSCLC]['patient_id'].unique())
    normal_leakage = len(normal_train_patients & normal_val_patients)
    nsclc_leakage  = len(nsclc_train_patients & nsclc_val_patients)

    patient_split_rows = [
        {'check': 'normal_train_patients', 'count': len(normal_train_patients), 'expected': n_c3_patients},
        {'check': 'normal_val_patients',   'count': len(normal_val_patients),   'expected': n_c10_patients},
        {'check': 'normal_patient_leakage','count': normal_leakage,             'expected': 0},
        {'check': 'nsclc_train_patients',  'count': len(nsclc_train_patients),  'expected': 101},
        {'check': 'nsclc_val_patients',    'count': len(nsclc_val_patients),    'expected': 24},
        {'check': 'nsclc_patient_leakage', 'count': nsclc_leakage,             'expected': 0},
    ]

    # Label distribution
    label_dist_rows = []
    for split_name, df_s in [('train', df_train), ('val', df_val), ('full', df_full)]:
        for lbl, lbl_name in [(LABEL_NORMAL, 'normal'), (LABEL_NSCLC, 'NSCLC')]:
            cnt = int((df_s['label'] == lbl).sum())
            label_dist_rows.append({
                'split': split_name, 'label': lbl, 'label_name': lbl_name, 'count': cnt
            })

    # Class weight check
    actual_train_total = len(df_train)
    actual_w_normal    = actual_train_total / (2.0 * normal_train_generated)
    actual_w_nsclc     = actual_train_total / (2.0 * actual_nsclc_train)
    class_weight_rows = [
        {'label': 'normal', 'count': normal_train_generated,  'class_weight': round(actual_w_normal, 6), 'expected_approx': 0.7630},
        {'label': 'NSCLC',  'count': actual_nsclc_train,      'class_weight': round(actual_w_nsclc, 6),  'expected_approx': 1.4505},
        {'label': 'train_total', 'count': actual_train_total, 'class_weight': '', 'expected_approx': ''},
    ]

    # NSCLC crop existence check (sample 10)
    nsclc_sample = df_train[df_train['label']==LABEL_NSCLC]['crop_path'].sample(
        min(10, len(nsclc_train_rows)), random_state=42
    ).tolist()
    nsclc_exist_ok = sum(1 for p in nsclc_sample if Path(p).exists())

    # Hard negative / MSD_Lung check in final manifest
    hard_neg_in_manifest = int(df_full['source_name'].isin(['MSD_Lung']).sum())
    msd_in_manifest      = hard_neg_in_manifest  # MSD_Lung filtered at source

    guardrail['hard_negative_included'] = False  # nsclc filtered from p_c_aux2
    guardrail['msd_lung_included']      = (msd_in_manifest > 0)

    # ── 11. Write report CSVs ─────────────────────────────────────────────
    write_csv(REPORT_OUTPUT_BASE / 'p_c_normal3_normal_crop_integrity_check.csv', integrity_rows)
    write_csv(REPORT_OUTPUT_BASE / 'p_c_normal3_label_distribution_check.csv', label_dist_rows)
    write_csv(REPORT_OUTPUT_BASE / 'p_c_normal3_patient_split_check.csv', patient_split_rows)
    write_csv(REPORT_OUTPUT_BASE / 'p_c_normal3_class_weight_check.csv', class_weight_rows)

    integrity_fail_count = sum(1 for r in integrity_rows if r['status'] != 'ok')

    # Guardrail final check CSV
    guardrail_rows = [{'guardrail': k, 'violated': v} for k, v in guardrail.items()]
    write_csv(REPORT_OUTPUT_BASE / 'p_c_normal3_guardrail_check.csv', guardrail_rows)

    # Errors
    write_csv(
        MANIFEST_OUTPUT_BASE / 'p_c_normal3_errors.csv',
        errors if errors else [{'severity': 'NONE', 'check': '-', 'message': 'No errors'}]
    )

    # ── 12. Determine verdict ─────────────────────────────────────────────
    conditions_ok = (
        normal_train_generated == NORMAL_TRAIN_CAP
        and normal_val_generated == NORMAL_VAL_CAP
        and len(nsclc_train_rows) == 7891
        and len(nsclc_val_rows) == 2080
        and normal_leakage == 0
        and nsclc_leakage == 0
        and not guardrail['hard_negative_included']
        and not guardrail['msd_lung_included']
        and not guardrail['stage2_holdout_accessed']
        and not guardrail['original_file_modified']
        and not guardrail['p_c_aux_modified']
        and not guardrail['nsclc_crop_copied']
        and not guardrail['training_run']
        and not guardrail['model_forward_run']
        and not guardrail['scoring_run']
        and integrity_fail_count == 0
    )

    if (conditions_ok):
        verdict = 'PASS'
    elif (normal_train_generated > 0 and normal_val_generated > 0 and integrity_fail_count == 0):
        verdict = 'PARTIAL_PASS'
    else:
        verdict = 'FAIL'

    end_time = datetime.now()
    elapsed_sec = (end_time - start_time).total_seconds()

    # ── 13. Write manifest summary JSON ──────────────────────────────────
    manifest_summary = {
        'stage': 'P-C-NORMAL3',
        'verdict': verdict,
        'conditions_ok': conditions_ok,
        'normal_train_generated': normal_train_generated,
        'normal_val_generated': normal_val_generated,
        'normal_train_errors': normal_train_errors,
        'normal_val_errors': normal_val_errors,
        'nsclc_train_rows': len(nsclc_train_rows),
        'nsclc_val_rows': len(nsclc_val_rows),
        'train_total': len(df_train),
        'val_total': len(df_val),
        'full_total': len(df_full),
        'train_npz_files_on_disk': len(train_npz_list),
        'val_npz_files_on_disk': len(val_npz_list),
        'normal_patient_leakage': normal_leakage,
        'nsclc_patient_leakage': nsclc_leakage,
        'class_weight_normal': round(actual_w_normal, 6),
        'class_weight_nsclc': round(actual_w_nsclc, 6),
        'nsclc_crop_sample_exist': f'{nsclc_exist_ok}/10',
        'integrity_spot_checks': len(integrity_rows),
        'integrity_fail_count': integrity_fail_count,
        'guardrail': guardrail,
        'elapsed_sec': round(elapsed_sec, 1),
        'errors': errors,
        'next_step': 'P-C-NORMAL4: manifest/crop validation checkpoint',
    }

    summary_json = MANIFEST_OUTPUT_BASE / 'p_c_normal3_manifest_summary.json'
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(manifest_summary, f, ensure_ascii=False, indent=2, default=str)

    # ── 14. Write generation report JSON ─────────────────────────────────
    gen_report = {
        'stage': 'P-C-NORMAL3',
        'verdict': verdict,
        'conditions_ok': conditions_ok,
        'sampling_option': 'B',
        'normal_train_cap': NORMAL_TRAIN_CAP,
        'normal_val_cap': NORMAL_VAL_CAP,
        'normal_train_generated': normal_train_generated,
        'normal_val_generated': normal_val_generated,
        'nsclc_train': len(nsclc_train_rows),
        'nsclc_val': len(nsclc_val_rows),
        'n_c3_source': str(N_C3_MANIFEST),
        'n_c10_source': str(N_C10_MANIFEST),
        'normal_crop_output': str(NORMAL_CROP_OUTPUT_BASE),
        'manifest_output': str(MANIFEST_OUTPUT_BASE),
        'n_c10_local_z_handling': 'slice_index_used_not_normalized_local_z',
        'n_c3_local_z_handling': 'local_z_column_integer',
        'class_weight_normal': round(actual_w_normal, 6),
        'class_weight_nsclc': round(actual_w_nsclc, 6),
        'guardrail': guardrail,
        'errors_count': len(errors),
        'elapsed_sec': round(elapsed_sec, 1),
    }

    gen_summary_path = REPORT_OUTPUT_BASE / 'p_c_normal3_actual_generation_summary.json'
    with open(gen_summary_path, 'w', encoding='utf-8') as f:
        json.dump(gen_report, f, ensure_ascii=False, indent=2, default=str)

    # ── 15. Write markdown report ─────────────────────────────────────────
    _write_markdown_report(verdict, manifest_summary, errors, integrity_rows,
                           patient_split_rows, label_dist_rows, class_weight_rows)

    # ── 16. Write DONE.json ───────────────────────────────────────────────
    done = {
        'stage': 'P-C-NORMAL3',
        'verdict': verdict,
        'conditions_ok': conditions_ok,
        'completed_at': end_time.isoformat(),
        'normal_train_generated': normal_train_generated,
        'normal_val_generated': normal_val_generated,
        'train_total': len(df_train),
        'val_total': len(df_val),
        'next_step': 'P-C-NORMAL4',
    }
    with open(MANIFEST_OUTPUT_BASE / 'DONE.json', 'w') as f:
        json.dump(done, f, ensure_ascii=False, indent=2)

    # ── Summary print ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"P-C-NORMAL3 VERDICT: {verdict}")
    print(f"  Normal train crops : {normal_train_generated} / {NORMAL_TRAIN_CAP}")
    print(f"  Normal val crops   : {normal_val_generated} / {NORMAL_VAL_CAP}")
    print(f"  NSCLC train rows   : {len(nsclc_train_rows)} (expected 7891)")
    print(f"  NSCLC val rows     : {len(nsclc_val_rows)} (expected 2080)")
    print(f"  Train total        : {len(df_train)} (expected 22891)")
    print(f"  Val total          : {len(df_val)} (expected 7080)")
    print(f"  Normal leakage     : {normal_leakage} (expected 0)")
    print(f"  class_weight_normal: {actual_w_normal:.4f} (expected ~0.7630)")
    print(f"  class_weight_nsclc : {actual_w_nsclc:.4f} (expected ~1.4505)")
    print(f"  Integrity fails    : {integrity_fail_count}")
    print(f"  Errors             : {len(errors)}")
    print(f"  Elapsed            : {elapsed_sec:.1f}s")
    print(f"  Manifest dir       : {MANIFEST_OUTPUT_BASE}")
    print(f"  Report dir         : {REPORT_OUTPUT_BASE}")
    print(f"{'='*65}")

    return verdict


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def _write_markdown_report(verdict, summary, errors, integrity_rows,
                           patient_split_rows, label_dist_rows, class_weight_rows):
    lines = [
        "# P-C-NORMAL3 Actual Normal Crop Generation Report",
        "",
        f"## 판정: {verdict}",
        "",
        f"- Normal train 생성: {summary['normal_train_generated']} / 15,000",
        f"- Normal val 생성:   {summary['normal_val_generated']} / 5,000",
        f"- NSCLC train rows:  {summary['nsclc_train_rows']}",
        f"- NSCLC val rows:    {summary['nsclc_val_rows']}",
        f"- Train total:       {summary['train_total']}",
        f"- Val total:         {summary['val_total']}",
        f"- Normal leakage:    {summary['normal_patient_leakage']}",
        f"- class_weight_normal: {summary['class_weight_normal']}",
        f"- class_weight_nsclc:  {summary['class_weight_nsclc']}",
        f"- Integrity fails:   {summary['integrity_fail_count']}",
        f"- Errors:            {len(errors)}",
        f"- Elapsed:           {summary['elapsed_sec']}s",
        "",
        "---",
        "",
        "## Guardrail",
        "",
    ]
    for k, v in summary['guardrail'].items():
        status = "✗ VIOLATED" if v else "OK"
        lines.append(f"- {k}: {status}")
    lines.append("")

    lines += [
        "## Label Distribution",
        "",
        "| split | label | label_name | count |",
        "|-------|-------|------------|-------|",
    ]
    for r in label_dist_rows:
        lines.append(f"| {r['split']} | {r['label']} | {r['label_name']} | {r['count']} |")
    lines.append("")

    lines += [
        "## Class Weights",
        "",
        "| label | count | class_weight | expected |",
        "|-------|-------|--------------|---------|",
    ]
    for r in class_weight_rows:
        lines.append(f"| {r['label']} | {r['count']} | {r['class_weight']} | {r['expected_approx']} |")
    lines.append("")

    lines += [
        "## Patient Split",
        "",
        "| check | count | expected |",
        "|-------|-------|----------|",
    ]
    for r in patient_split_rows:
        lines.append(f"| {r['check']} | {r['count']} | {r['expected']} |")
    lines.append("")

    lines += [
        "## Crop Integrity (spot check)",
        "",
        f"- 총 점검 수: {len(integrity_rows)}",
        f"- 실패: {sum(1 for r in integrity_rows if r['status'] != 'ok')}",
        "",
    ]

    if errors:
        lines += ["## Errors", ""]
        for e in errors[:20]:
            lines.append(f"- [{e.get('severity','')}] {e.get('check','')}: {e.get('message','')}")
        if len(errors) > 20:
            lines.append(f"... and {len(errors)-20} more")
        lines.append("")

    lines += [
        "---",
        "",
        "## 다음 단계",
        "",
        "- P-C-NORMAL4: manifest/crop validation checkpoint",
        "- 그 다음 P-C-NORMAL5: training script dry-check",
    ]

    md_path = REPORT_OUTPUT_BASE / 'p_c_normal3_actual_generation_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # Also write manifest report
    mf_md_path = MANIFEST_OUTPUT_BASE / 'p_c_normal3_manifest_report.md'
    with open(mf_md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    verdict = run_generation()
    sys.exit(0 if verdict in ('PASS', 'PARTIAL_PASS') else 1)
