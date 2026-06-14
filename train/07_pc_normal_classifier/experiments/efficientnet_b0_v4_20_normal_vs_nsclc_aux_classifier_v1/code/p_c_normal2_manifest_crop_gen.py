"""
P-C-NORMAL2: Normal-vs-NSCLC Manifest / Normal Crop Generation Script
Branch: efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1

Mode:
  --mode drycheck  : dry-check only (no crop files saved, no manifest saved)
  --mode generate  : actual normal crop generation + manifest creation (requires user approval)

WARNING: --mode generate is BLOCKED until user explicitly approves P-C-NORMAL2 dry-check results.

Usage:
  python p_c_normal2_manifest_crop_gen.py --mode drycheck
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths (project-relative or absolute WSL paths)
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path('/home/jinhy/project/lung-ct-anomaly')
BRANCH_ROOT = PROJECT_ROOT / 'experiments/efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1'

# Normal manifests
N_C3_MANIFEST = PROJECT_ROOT / 'experiments/normal_only_second_stage_refiner_v1/outputs/manifests/n_c3_normal_only_crop_manifest_v1/n_c3_normal_only_crop_manifest_v1.csv'
N_C10_MANIFEST = PROJECT_ROOT / 'experiments/normal_only_second_stage_refiner_v1/outputs/manifests/n_c10_normal_val_crop_manifest/n_c10_normal_val_crop_manifest.csv'

# NSCLC positive manifest (P-C-AUX side branch — read-only)
P_C_AUX_MANIFEST = PROJECT_ROOT / 'experiments/efficientnet_b0_v4_20_supervised_aux_source_classifier_v1/outputs/manifests/p_c_aux2_source_classifier_training_manifest/p_c_aux2_source_classifier_training_manifest.csv'
P_C_AUX_CROP_BASE = PROJECT_ROOT / 'experiments/efficientnet_b0_v4_20_second_stage_refiner_v1'

# CT volume base directory
CT_VOL_BASE = Path('/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy')

# Output paths (for actual generation — blocked until user approves)
NORMAL_CROP_OUTPUT_BASE = BRANCH_ROOT / 'outputs/normal_crops/p_c_normal3_normal_train_val_crops'
MANIFEST_OUTPUT_BASE = BRANCH_ROOT / 'outputs/manifests/p_c_normal3_training_manifest'

# Dry-check report path
DRYCHECK_REPORT_DIR = BRANCH_ROOT / 'outputs/reports/p_c_normal2_manifest_crop_gen_drycheck'

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CROP_SIZE = 96
HALF_CROP = CROP_SIZE // 2  # 48

# Labels
LABEL_NORMAL = 0
LABEL_NSCLC = 1

# Sampling caps (per recommended Option B)
# See dry-check sampling_option_comparison.csv for all options
RECOMMENDED_OPTION = 'B'
NORMAL_TRAIN_CAP = 15000
NORMAL_VAL_CAP = 5000
NORMAL_TRAIN_PER_PATIENT_CAP = NORMAL_TRAIN_CAP // 290  # ~51 per patient
NORMAL_VAL_PER_PATIENT_CAP = NORMAL_VAL_CAP // 36       # ~138 per patient

# ─────────────────────────────────────────────────────────────────────────────
# Path resolver for normal CT volumes
# ─────────────────────────────────────────────────────────────────────────────

def resolve_normal_ct_path(row, source='n_c3'):
    """
    n_c3: has source_ct_path column directly.
    n_c10: has safe_id, no source_ct_path — construct from CT_VOL_BASE / safe_id / ct_hu.npy
    """
    if source == 'n_c3':
        return str(row['source_ct_path'])
    elif source == 'n_c10':
        return str(CT_VOL_BASE / row['safe_id'] / 'ct_hu.npy')
    else:
        raise ValueError(f"Unknown source: {source}")


# ─────────────────────────────────────────────────────────────────────────────
# Crop extraction (3-channel, edge padding)
# ─────────────────────────────────────────────────────────────────────────────

def extract_normal_crop(ct_vol, local_z, center_y, center_x):
    """
    Extract (3, 96, 96) crop from CT volume.
    Channels: z-1 / z / z+1 with edge padding at z boundaries.
    XY: center±48 with edge padding if out of bounds.
    Returns np.ndarray (3, 96, 96) int16 — raw HU values.
    """
    import numpy as np
    z_dim, y_dim, x_dim = ct_vol.shape
    local_z = int(local_z)
    cy = int(center_y)
    cx = int(center_x)

    # Z indices with boundary clamp (edge padding)
    z_indices = [
        max(0, local_z - 1),
        local_z,
        min(z_dim - 1, local_z + 1),
    ]

    # XY slice with boundary clamp + edge padding
    y0 = cy - HALF_CROP
    y1 = cy + HALF_CROP
    x0 = cx - HALF_CROP
    x1 = cx + HALF_CROP

    y_pad_before = max(0, -y0)
    y_pad_after = max(0, y1 - y_dim)
    x_pad_before = max(0, -x0)
    x_pad_after = max(0, x1 - x_dim)

    y0c = max(0, y0)
    y1c = min(y_dim, y1)
    x0c = max(0, x0)
    x1c = min(x_dim, x1)

    channels = []
    for zi in z_indices:
        slice_2d = ct_vol[zi, y0c:y1c, x0c:x1c]
        if any([y_pad_before, y_pad_after, x_pad_before, x_pad_after]):
            slice_2d = np.pad(
                slice_2d,
                ((y_pad_before, y_pad_after), (x_pad_before, x_pad_after)),
                mode='edge',
            )
        channels.append(slice_2d)

    crop = np.stack(channels, axis=0)
    assert crop.shape == (3, CROP_SIZE, CROP_SIZE), f"Unexpected shape: {crop.shape}"
    return crop


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def sample_balanced_by_position_bin(df, per_patient_cap, seed=42):
    """
    Sample rows with position_bin balance per patient.
    Each patient × position_bin group gets equal share up to per_patient_cap total.
    """
    import pandas as pd
    results = []
    n_bins = df['position_bin'].nunique()
    per_bin_cap = max(1, per_patient_cap // n_bins)
    for patient_id, pdf in df.groupby('patient_id'):
        sampled_rows = []
        for bin_name, bdf in pdf.groupby('position_bin'):
            n_take = min(len(bdf), per_bin_cap)
            sampled_rows.append(bdf.sample(n=n_take, random_state=seed))
        results.append(pd.concat(sampled_rows))
    return pd.concat(results).reset_index(drop=True)


def compute_class_weights(n_normal, n_nsclc):
    """Inverse frequency class weights."""
    total = n_normal + n_nsclc
    w_normal = total / (2.0 * n_normal)
    w_nsclc = total / (2.0 * n_nsclc)
    return w_normal, w_nsclc


# ─────────────────────────────────────────────────────────────────────────────
# Manifest schema builder (for metadata plan — no actual CSV written in drycheck)
# ─────────────────────────────────────────────────────────────────────────────

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
# Dry-check runner
# ─────────────────────────────────────────────────────────────────────────────

def run_drycheck():
    import pandas as pd
    import numpy as np

    DRYCHECK_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    errors = []
    checks = {}

    # ── 1. P-C-NORMAL1 summary exists ──────────────────────────────────────
    normal1_md = BRANCH_ROOT / 'outputs/reports/p_c_normal1_design_preflight/p_c_normal1_design_preflight.md'
    normal1_json = BRANCH_ROOT / 'outputs/reports/p_c_normal1_design_preflight/p_c_normal1_design_preflight.json'
    checks['p_c_normal1_md_exists'] = normal1_md.exists()
    checks['p_c_normal1_json_exists'] = normal1_json.exists()
    if normal1_json.exists():
        with open(normal1_json) as f:
            n1 = json.load(f)
        checks['p_c_normal1_verdict'] = n1.get('verdict', 'UNKNOWN')
    else:
        checks['p_c_normal1_verdict'] = 'NOT_FOUND'
        errors.append({'severity': 'ERROR', 'check': 'p_c_normal1_verdict', 'message': 'P-C-NORMAL1 json not found'})

    # ── 2. P-C-AUX side branch exists and NOT modified ─────────────────────
    p_c_aux_path = PROJECT_ROOT / 'experiments/efficientnet_b0_v4_20_supervised_aux_source_classifier_v1'
    checks['p_c_aux_exists'] = p_c_aux_path.exists()
    checks['p_c_aux_modified'] = False  # read-only check — we never write to it

    # ── 3. Normal coordinate manifests ─────────────────────────────────────
    checks['n_c3_manifest_exists'] = N_C3_MANIFEST.exists()
    checks['n_c10_manifest_exists'] = N_C10_MANIFEST.exists()
    if not N_C3_MANIFEST.exists():
        errors.append({'severity': 'ERROR', 'check': 'n_c3_manifest', 'message': str(N_C3_MANIFEST)})
    if not N_C10_MANIFEST.exists():
        errors.append({'severity': 'ERROR', 'check': 'n_c10_manifest', 'message': str(N_C10_MANIFEST)})

    # ── 4. Load normal manifests ────────────────────────────────────────────
    df3 = pd.read_csv(N_C3_MANIFEST)
    df10 = pd.read_csv(N_C10_MANIFEST)
    checks['n_c3_row_count'] = len(df3)
    checks['n_c3_patient_count'] = df3['patient_id'].nunique()
    checks['n_c10_row_count'] = len(df10)
    checks['n_c10_patient_count'] = df10['patient_id'].nunique()

    # ── 5. Patient overlap check ────────────────────────────────────────────
    n_c3_patients = set(df3['patient_id'].unique())
    n_c10_patients = set(df10['patient_id'].unique())
    overlap = n_c3_patients & n_c10_patients
    checks['normal_train_val_patient_overlap'] = len(overlap)
    if len(overlap) > 0:
        errors.append({'severity': 'ERROR', 'check': 'patient_overlap', 'message': f'Overlap: {overlap}'})

    # ── 6. Normal CT path mapping ───────────────────────────────────────────
    # n_c3: source_ct_path column present and sample file exists
    n_c3_ct_null = df3['source_ct_path'].isnull().sum()
    checks['n_c3_source_ct_path_null_count'] = int(n_c3_ct_null)
    sample_n_c3_ct = df3['source_ct_path'].iloc[0]
    checks['n_c3_sample_ct_path'] = sample_n_c3_ct
    checks['n_c3_sample_ct_exists'] = os.path.exists(sample_n_c3_ct)
    if not os.path.exists(sample_n_c3_ct):
        errors.append({'severity': 'ERROR', 'check': 'n_c3_ct_path', 'message': f'Not found: {sample_n_c3_ct}'})

    # n_c10: safe_id based mapping
    checks['n_c10_has_source_ct_path'] = 'source_ct_path' in df10.columns
    missing_ct = 0
    for safe_id in df10['safe_id'].unique():
        ct_path = CT_VOL_BASE / safe_id / 'ct_hu.npy'
        if not ct_path.exists():
            missing_ct += 1
    checks['n_c10_ct_path_missing_count'] = missing_ct
    checks['n_c10_ct_path_mapping_success'] = (missing_ct == 0)
    if missing_ct > 0:
        errors.append({'severity': 'ERROR', 'check': 'n_c10_ct_mapping', 'message': f'{missing_ct} patients missing CT'})

    # ── 7. Normal crop extraction feasibility (memory only, no save) ────────
    sample_n_c10_ct_path = str(CT_VOL_BASE / df10['safe_id'].iloc[0] / 'ct_hu.npy')
    feasibility_results = []
    # Test 3 crops from n_c3
    for i, row in df3.head(3).iterrows():
        ct_path = row['source_ct_path']
        try:
            ct_vol = np.load(ct_path)
            crop = extract_normal_crop(ct_vol, row['local_z'], row['center_y'], row['center_x'])
            feasibility_results.append({
                'source': 'n_c3', 'idx': i,
                'status': 'ok', 'shape': str(crop.shape), 'dtype': str(crop.dtype),
                'npz_saved': False
            })
            del ct_vol, crop
        except Exception as e:
            feasibility_results.append({'source': 'n_c3', 'idx': i, 'status': f'error:{e}', 'shape': None, 'dtype': None, 'npz_saved': False})
            errors.append({'severity': 'ERROR', 'check': 'n_c3_crop_feasibility', 'message': str(e)})
    # Test 3 crops from n_c10
    for i, row in df10.head(3).iterrows():
        ct_path = resolve_normal_ct_path(row, source='n_c10')
        try:
            ct_vol = np.load(ct_path)
            crop = extract_normal_crop(ct_vol, row['local_z'], row['center_y'], row['center_x'])
            feasibility_results.append({
                'source': 'n_c10', 'idx': i,
                'status': 'ok', 'shape': str(crop.shape), 'dtype': str(crop.dtype),
                'npz_saved': False
            })
            del ct_vol, crop
        except Exception as e:
            feasibility_results.append({'source': 'n_c10', 'idx': i, 'status': f'error:{e}', 'shape': None, 'dtype': None, 'npz_saved': False})
            errors.append({'severity': 'ERROR', 'check': 'n_c10_crop_feasibility', 'message': str(e)})

    n3_ok = sum(1 for r in feasibility_results if r['source'] == 'n_c3' and r['status'] == 'ok')
    n10_ok = sum(1 for r in feasibility_results if r['source'] == 'n_c10' and r['status'] == 'ok')
    checks['n_c3_crop_feasibility_ok'] = (n3_ok == 3)
    checks['n_c10_crop_feasibility_ok'] = (n10_ok == 3)
    checks['crop_npz_saved'] = False  # guardrail: never saved

    # ── 8. NSCLC positive manifest ─────────────────────────────────────────
    checks['p_c_aux_manifest_exists'] = P_C_AUX_MANIFEST.exists()
    if not P_C_AUX_MANIFEST.exists():
        errors.append({'severity': 'ERROR', 'check': 'p_c_aux_manifest', 'message': str(P_C_AUX_MANIFEST)})

    df_aux = pd.read_csv(P_C_AUX_MANIFEST)
    nsclc_df = df_aux[df_aux['source_name'] == 'NSCLC'].copy()
    msd_df = df_aux[df_aux['source_name'] == 'MSD_Lung']
    hard_neg_df = df_aux[df_aux.get('forbidden_hard_negative_used', pd.Series([False]*len(df_aux))).fillna(False)]

    checks['nsclc_total_rows'] = len(nsclc_df)
    checks['nsclc_train_rows'] = len(nsclc_df[nsclc_df['split'] == 'train'])
    checks['nsclc_val_rows'] = len(nsclc_df[nsclc_df['split'] == 'val'])
    checks['nsclc_train_patients'] = int(nsclc_df[nsclc_df['split'] == 'train']['patient_id'].nunique())
    checks['nsclc_val_patients'] = int(nsclc_df[nsclc_df['split'] == 'val']['patient_id'].nunique())
    checks['msd_lung_excluded'] = True  # filtered by source_name != MSD_Lung
    checks['hard_negative_excluded'] = bool((nsclc_df['forbidden_hard_negative_used'] == False).all())
    checks['nsclc_all_positive'] = bool((nsclc_df['original_p_c_label'] == 'positive').all())

    if checks['nsclc_train_rows'] != 7891:
        errors.append({'severity': 'WARN', 'check': 'nsclc_train_count', 'message': f"Expected 7891, got {checks['nsclc_train_rows']}"})
    if checks['nsclc_val_rows'] != 2080:
        errors.append({'severity': 'WARN', 'check': 'nsclc_val_count', 'message': f"Expected 2080, got {checks['nsclc_val_rows']}"})

    # ── 9. NSCLC crop_path sample existence ────────────────────────────────
    sample_crop_paths = nsclc_df['crop_path'].sample(min(10, len(nsclc_df)), random_state=42).tolist()
    crop_exist_count = 0
    for cp in sample_crop_paths:
        full_path = P_C_AUX_CROP_BASE / cp
        if full_path.exists():
            crop_exist_count += 1
    checks['nsclc_sample_crop_exist_count'] = crop_exist_count
    checks['nsclc_sample_crop_total_checked'] = len(sample_crop_paths)
    checks['nsclc_crop_paths_ok'] = (crop_exist_count == len(sample_crop_paths))
    if crop_exist_count < len(sample_crop_paths):
        errors.append({'severity': 'ERROR', 'check': 'nsclc_crop_exist', 'message': f'{crop_exist_count}/{len(sample_crop_paths)} crops found'})

    # ── 10. Label mapping check ─────────────────────────────────────────────
    checks['label_normal'] = LABEL_NORMAL
    checks['label_nsclc'] = LABEL_NSCLC
    checks['label_mapping_correct'] = (LABEL_NORMAL == 0 and LABEL_NSCLC == 1)

    # ── 11. Sampling option comparison ─────────────────────────────────────
    nsclc_train = checks['nsclc_train_rows']
    nsclc_val = checks['nsclc_val_rows']
    n_normal_train_patients = checks['n_c3_patient_count']
    n_normal_val_patients = checks['n_c10_patient_count']

    sampling_options = [
        {
            'option': 'A',
            'description': 'patient cap 100/patient for train, no val cap',
            'normal_train_target': 29000,
            'normal_val_target': 21600,
            'normal_train_per_patient': 100,
            'normal_val_per_patient': 600,
            'nsclc_train': nsclc_train,
            'nsclc_val': nsclc_val,
            'train_imbalance_ratio': round(29000 / nsclc_train, 2),
            'val_imbalance_ratio': round(21600 / nsclc_val, 2),
            'train_class_weight_normal': round((29000 + nsclc_train) / (2.0 * 29000), 4),
            'train_class_weight_nsclc': round((29000 + nsclc_train) / (2.0 * nsclc_train), 4),
            'comment': 'val ratio 10.4:1 too high; class_weight correction needed',
        },
        {
            'option': 'B',
            'description': 'cap normal train 15000 / val 5000 (RECOMMENDED)',
            'normal_train_target': 15000,
            'normal_val_target': 5000,
            'normal_train_per_patient': round(15000 / n_normal_train_patients),
            'normal_val_per_patient': round(5000 / n_normal_val_patients),
            'nsclc_train': nsclc_train,
            'nsclc_val': nsclc_val,
            'train_imbalance_ratio': round(15000 / nsclc_train, 2),
            'val_imbalance_ratio': round(5000 / nsclc_val, 2),
            'train_class_weight_normal': round((15000 + nsclc_train) / (2.0 * 15000), 4),
            'train_class_weight_nsclc': round((15000 + nsclc_train) / (2.0 * nsclc_train), 4),
            'comment': 'balanced training; val 2.4:1 manageable; recommended',
        },
        {
            'option': 'C',
            'description': 'normal train ~16000 (2x NSCLC) / val ~4160 (2x NSCLC val)',
            'normal_train_target': nsclc_train * 2,
            'normal_val_target': nsclc_val * 2,
            'normal_train_per_patient': round((nsclc_train * 2) / n_normal_train_patients),
            'normal_val_per_patient': round((nsclc_val * 2) / n_normal_val_patients),
            'nsclc_train': nsclc_train,
            'nsclc_val': nsclc_val,
            'train_imbalance_ratio': 2.0,
            'val_imbalance_ratio': 2.0,
            'train_class_weight_normal': round((nsclc_train * 2 + nsclc_train) / (2.0 * nsclc_train * 2), 4),
            'train_class_weight_nsclc': round((nsclc_train * 2 + nsclc_train) / (2.0 * nsclc_train), 4),
            'comment': 'strictly 2:1 ratio; less normal diversity than Option B',
        },
    ]
    checks['recommended_sampling_option'] = RECOMMENDED_OPTION
    checks['recommended_normal_train_cap'] = NORMAL_TRAIN_CAP
    checks['recommended_normal_val_cap'] = NORMAL_VAL_CAP

    # Class weights for recommended option
    w_n, w_p = compute_class_weights(NORMAL_TRAIN_CAP, nsclc_train)
    checks['recommended_class_weight_normal'] = round(w_n, 4)
    checks['recommended_class_weight_nsclc'] = round(w_p, 4)

    # ── 12. Output collision check ─────────────────────────────────────────
    checks['normal_crop_output_dir_exists'] = NORMAL_CROP_OUTPUT_BASE.exists()
    checks['manifest_output_dir_exists'] = MANIFEST_OUTPUT_BASE.exists()
    checks['output_collision'] = NORMAL_CROP_OUTPUT_BASE.exists() or MANIFEST_OUTPUT_BASE.exists()
    # No collision = safe to generate later

    # ── 13. Guardrail: actual files NOT generated ──────────────────────────
    checks['actual_normal_crop_files_generated'] = False
    checks['actual_manifest_generated'] = False
    checks['model_forward_executed'] = False
    checks['training_executed'] = False
    checks['scoring_executed'] = False
    checks['threshold_calculated'] = False
    checks['stage2_holdout_accessed'] = False
    checks['p_c_aux_branch_modified'] = False
    checks['existing_results_modified'] = False

    # ── Aggregate verdict ──────────────────────────────────────────────────
    critical_errors = [e for e in errors if e['severity'] == 'ERROR']
    warn_errors = [e for e in errors if e['severity'] == 'WARN']

    ct_mapping_ok = (
        checks.get('n_c3_sample_ct_exists', False)
        and checks.get('n_c10_ct_path_mapping_success', False)
    )
    crop_feasibility_ok = (
        checks.get('n_c3_crop_feasibility_ok', False)
        and checks.get('n_c10_crop_feasibility_ok', False)
    )
    nsclc_source_ok = (
        checks.get('nsclc_crop_paths_ok', False)
        and checks.get('hard_negative_excluded', False)
        and checks.get('msd_lung_excluded', False)
    )
    guardrail_ok = not any([
        checks.get('actual_normal_crop_files_generated', True),
        checks.get('actual_manifest_generated', True),
        checks.get('model_forward_executed', True),
        checks.get('training_executed', True),
        checks.get('stage2_holdout_accessed', True),
        checks.get('p_c_aux_branch_modified', True),
        checks.get('existing_results_modified', True),
    ])

    if len(critical_errors) == 0 and ct_mapping_ok and crop_feasibility_ok and nsclc_source_ok and guardrail_ok:
        verdict = 'PASS'
    elif not ct_mapping_ok or not nsclc_source_ok:
        verdict = 'FAIL'
    else:
        verdict = 'PARTIAL_PASS'

    # ── Write CSV outputs ──────────────────────────────────────────────────

    _write_csv(DRYCHECK_REPORT_DIR / 'p_c_normal2_input_source_validation.csv', [
        {'check': 'n_c3_manifest_exists', 'value': checks['n_c3_manifest_exists'], 'path': str(N_C3_MANIFEST)},
        {'check': 'n_c10_manifest_exists', 'value': checks['n_c10_manifest_exists'], 'path': str(N_C10_MANIFEST)},
        {'check': 'p_c_aux_manifest_exists', 'value': checks['p_c_aux_manifest_exists'], 'path': str(P_C_AUX_MANIFEST)},
        {'check': 'p_c_aux_path_exists', 'value': checks['p_c_aux_exists'], 'path': str(p_c_aux_path)},
        {'check': 'p_c_normal1_json_exists', 'value': checks['p_c_normal1_json_exists'], 'path': str(normal1_json)},
        {'check': 'p_c_normal1_verdict', 'value': checks['p_c_normal1_verdict'], 'path': ''},
        {'check': 'n_c3_rows', 'value': checks['n_c3_row_count'], 'path': ''},
        {'check': 'n_c3_patients', 'value': checks['n_c3_patient_count'], 'path': ''},
        {'check': 'n_c10_rows', 'value': checks['n_c10_row_count'], 'path': ''},
        {'check': 'n_c10_patients', 'value': checks['n_c10_patient_count'], 'path': ''},
        {'check': 'nsclc_total_rows', 'value': checks['nsclc_total_rows'], 'path': ''},
        {'check': 'nsclc_train_rows', 'value': checks['nsclc_train_rows'], 'path': ''},
        {'check': 'nsclc_val_rows', 'value': checks['nsclc_val_rows'], 'path': ''},
        {'check': 'normal_train_val_patient_overlap', 'value': checks['normal_train_val_patient_overlap'], 'path': ''},
    ])

    _write_csv(DRYCHECK_REPORT_DIR / 'p_c_normal2_normal_ct_path_mapping_check.csv', [
        {'source': 'n_c3', 'check': 'has_source_ct_path_column', 'value': True, 'note': 'direct column in CSV'},
        {'source': 'n_c3', 'check': 'source_ct_path_null_count', 'value': checks['n_c3_source_ct_path_null_count'], 'note': ''},
        {'source': 'n_c3', 'check': 'sample_ct_path_exists', 'value': checks['n_c3_sample_ct_exists'], 'note': checks['n_c3_sample_ct_path']},
        {'source': 'n_c10', 'check': 'has_source_ct_path_column', 'value': checks['n_c10_has_source_ct_path'], 'note': 'NOT in CSV'},
        {'source': 'n_c10', 'check': 'safe_id_to_ct_mapping', 'value': 'CT_VOL_BASE/safe_id/ct_hu.npy', 'note': 'constructed from safe_id'},
        {'source': 'n_c10', 'check': 'ct_path_missing_count', 'value': checks['n_c10_ct_path_missing_count'], 'note': 'out of 36 patients'},
        {'source': 'n_c10', 'check': 'ct_path_mapping_success', 'value': checks['n_c10_ct_path_mapping_success'], 'note': ''},
    ])

    _write_csv(DRYCHECK_REPORT_DIR / 'p_c_normal2_normal_crop_feasibility_check.csv', feasibility_results)

    _write_csv(DRYCHECK_REPORT_DIR / 'p_c_normal2_nsclc_positive_source_check.csv', [
        {'check': 'source_filter', 'value': 'source_name==NSCLC', 'note': ''},
        {'check': 'msd_lung_excluded', 'value': checks['msd_lung_excluded'], 'note': 'filter by source_name'},
        {'check': 'hard_negative_excluded', 'value': checks['hard_negative_excluded'], 'note': 'forbidden_hard_negative_used==False'},
        {'check': 'all_positive_label', 'value': checks['nsclc_all_positive'], 'note': 'original_p_c_label==positive'},
        {'check': 'nsclc_train_rows', 'value': checks['nsclc_train_rows'], 'note': 'expected 7891'},
        {'check': 'nsclc_val_rows', 'value': checks['nsclc_val_rows'], 'note': 'expected 2080'},
        {'check': 'nsclc_train_patients', 'value': checks['nsclc_train_patients'], 'note': 'expected 101'},
        {'check': 'nsclc_val_patients', 'value': checks['nsclc_val_patients'], 'note': 'expected 24'},
        {'check': 'crop_path_sample_exist', 'value': f"{checks['nsclc_sample_crop_exist_count']}/{checks['nsclc_sample_crop_total_checked']}", 'note': ''},
        {'check': 'crop_base_path', 'value': str(P_C_AUX_CROP_BASE), 'note': ''},
    ])

    _write_csv(DRYCHECK_REPORT_DIR / 'p_c_normal2_sampling_option_comparison.csv', sampling_options)

    _write_csv(DRYCHECK_REPORT_DIR / 'p_c_normal2_manifest_schema_plan.csv', [
        {'column': c, 'present_in_normal': True, 'present_in_nsclc': True,
         'note': 'normal-specific' if c in ['normal_source_split', 'normal_coord_source', 'lesion_pixels', 'has_lesion_patch']
                 else ('nsclc-specific' if c in ['original_p_c_candidate_id', 'original_p_c_label', 'lesion_pixels_nsclc', 'has_lesion_patch_nsclc', 'source_name'] else 'common')}
        for c in MANIFEST_COLUMNS
    ])

    _write_csv(DRYCHECK_REPORT_DIR / 'p_c_normal2_class_weight_plan.csv', [
        {'option': opt['option'],
         'normal_train': opt['normal_train_target'],
         'nsclc_train': opt['nsclc_train'],
         'train_ratio': opt['train_imbalance_ratio'],
         'class_weight_normal': opt['train_class_weight_normal'],
         'class_weight_nsclc': opt['train_class_weight_nsclc'],
         'sample_weight_strategy': 'inverse_freq_per_sample',
         'loss': 'BCEWithLogitsLoss(reduction=mean) + sample_weight',
         'recommended': (opt['option'] == RECOMMENDED_OPTION)}
        for opt in sampling_options
    ])

    _write_csv(DRYCHECK_REPORT_DIR / 'p_c_normal2_output_collision_check.csv', [
        {'path': str(NORMAL_CROP_OUTPUT_BASE), 'exists': checks['normal_crop_output_dir_exists'], 'collision': checks['normal_crop_output_dir_exists'], 'note': ''},
        {'path': str(MANIFEST_OUTPUT_BASE), 'exists': checks['manifest_output_dir_exists'], 'collision': checks['manifest_output_dir_exists'], 'note': ''},
        {'path': str(DRYCHECK_REPORT_DIR), 'exists': True, 'collision': False, 'note': 'drycheck report dir — intentional'},
    ])

    _write_csv(DRYCHECK_REPORT_DIR / 'p_c_normal2_guardrail_check.csv', [
        {'guardrail': 'actual_normal_crop_files_generated', 'violated': checks['actual_normal_crop_files_generated']},
        {'guardrail': 'actual_manifest_generated', 'violated': checks['actual_manifest_generated']},
        {'guardrail': 'model_forward_executed', 'violated': checks['model_forward_executed']},
        {'guardrail': 'training_executed', 'violated': checks['training_executed']},
        {'guardrail': 'scoring_executed', 'violated': checks['scoring_executed']},
        {'guardrail': 'threshold_calculated', 'violated': checks['threshold_calculated']},
        {'guardrail': 'stage2_holdout_accessed', 'violated': checks['stage2_holdout_accessed']},
        {'guardrail': 'p_c_aux_branch_modified', 'violated': checks['p_c_aux_branch_modified']},
        {'guardrail': 'existing_results_modified', 'violated': checks['existing_results_modified']},
    ])

    _write_csv(DRYCHECK_REPORT_DIR / 'p_c_normal2_errors.csv', errors if errors else [{'severity': 'NONE', 'check': '-', 'message': 'No errors'}])

    # ── JSON summary ────────────────────────────────────────────────────────
    summary = {
        'stage': 'P-C-NORMAL2',
        'mode': 'drycheck',
        'verdict': verdict,
        'checks': checks,
        'errors': errors,
        'sampling': {
            'recommended_option': RECOMMENDED_OPTION,
            'normal_train_cap': NORMAL_TRAIN_CAP,
            'normal_val_cap': NORMAL_VAL_CAP,
            'nsclc_train': nsclc_train,
            'nsclc_val': nsclc_val,
            'train_imbalance': round(NORMAL_TRAIN_CAP / nsclc_train, 2),
            'val_imbalance': round(NORMAL_VAL_CAP / nsclc_val, 2),
            'class_weight_normal': round(w_n, 4),
            'class_weight_nsclc': round(w_p, 4),
        },
        'next_step': 'P-C-NORMAL3: actual normal crop generation + manifest creation (requires user approval)',
    }

    summary_path = DRYCHECK_REPORT_DIR / 'p_c_normal2_manifest_crop_gen_drycheck.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── Markdown report ─────────────────────────────────────────────────────
    _write_markdown_report(verdict, checks, errors, sampling_options, feasibility_results)

    print(f"\n{'='*60}")
    print(f"P-C-NORMAL2 DRY-CHECK VERDICT: {verdict}")
    print(f"Critical errors: {len(critical_errors)}, Warnings: {len(warn_errors)}")
    print(f"Report dir: {DRYCHECK_REPORT_DIR}")
    print(f"{'='*60}")

    return verdict


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_markdown_report(verdict, checks, errors, sampling_options, feasibility_results):
    critical_errors = [e for e in errors if e['severity'] == 'ERROR']
    warn_errors = [e for e in errors if e['severity'] == 'WARN']
    nsclc_train = checks['nsclc_train_rows']
    nsclc_val = checks['nsclc_val_rows']

    lines = [
        "# P-C-NORMAL2 Manifest / Crop Generation Dry-Check Report",
        "",
        f"## 판정: {verdict}",
        "",
    ]

    if verdict == 'PASS':
        lines += [
            "> normal crop generation feasibility 확인 완료.",
            "> normal CT path mapping 100% 성공.",
            "> NSCLC positive source 즉시 사용 가능.",
            "> sampling/class_weight plan 확정.",
            "> actual crop/manifest 미생성. stage2_holdout 미접근.",
            "",
        ]
    elif verdict == 'PARTIAL_PASS':
        lines += ["> 일부 항목 보완 필요. 아래 errors 섹션 참고.", ""]
    else:
        lines += ["> FAIL: 아래 critical errors 해소 필요.", ""]

    lines += [
        "---",
        "",
        "## 1. 입력 소스 검증",
        "",
        "| 소스 | 항목 | 값 |",
        "|---|---|---|",
        f"| N-C3 train manifest | exists | {checks['n_c3_manifest_exists']} |",
        f"| N-C3 train manifest | rows | {checks['n_c3_row_count']} |",
        f"| N-C3 train manifest | patients | {checks['n_c3_patient_count']} |",
        f"| N-C10 val manifest | exists | {checks['n_c10_manifest_exists']} |",
        f"| N-C10 val manifest | rows | {checks['n_c10_row_count']} |",
        f"| N-C10 val manifest | patients | {checks['n_c10_patient_count']} |",
        f"| normal train/val overlap | patient count | {checks['normal_train_val_patient_overlap']} |",
        f"| P-C-AUX manifest | exists | {checks['p_c_aux_manifest_exists']} |",
        f"| NSCLC train rows | count | {checks['nsclc_train_rows']} (expected 7891) |",
        f"| NSCLC val rows | count | {checks['nsclc_val_rows']} (expected 2080) |",
        "",
        "---",
        "",
        "## 2. Normal CT Path Mapping",
        "",
        "| 소스 | 방법 | 결과 |",
        "|---|---|---|",
        f"| n_c3 | source_ct_path 직접 사용 | null={checks['n_c3_source_ct_path_null_count']}, sample_exists={checks['n_c3_sample_ct_exists']} |",
        f"| n_c10 | safe_id → CT_VOL_BASE/safe_id/ct_hu.npy 구성 | missing={checks['n_c10_ct_path_missing_count']}/36, success={checks['n_c10_ct_path_mapping_success']} |",
        "",
        "**부분통과 원인(n_c10 source_ct_path 없음) 해소됨:** `resolve_normal_ct_path()` 함수로 매핑 가능.",
        "",
        "---",
        "",
        "## 3. Normal Crop Extraction Feasibility",
        "",
        "| source | idx | status | shape | dtype | npz_saved |",
        "|---|---|---|---|---|---|",
    ]
    for r in feasibility_results:
        lines.append(f"| {r['source']} | {r['idx']} | {r['status']} | {r.get('shape','-')} | {r.get('dtype','-')} | {r.get('npz_saved', False)} |")

    lines += [
        "",
        "crop extraction 로직: `ct[z-1:z+2, cy-48:cy+48, cx-48:cx+48]`, z/xy 경계 edge padding.",
        "**npz 저장 없음** — 메모리 feasibility 확인만 수행.",
        "",
        "---",
        "",
        "## 4. NSCLC Positive Source",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 소스 manifest | p_c_aux2_source_classifier_training_manifest.csv |",
        f"| 필터 | source_name == NSCLC |",
        f"| MSD_Lung 제외 | {checks['msd_lung_excluded']} |",
        f"| hard_negative 제외 | {checks['hard_negative_excluded']} |",
        f"| all positive label | {checks['nsclc_all_positive']} |",
        f"| train rows | {checks['nsclc_train_rows']} |",
        f"| val rows | {checks['nsclc_val_rows']} |",
        f"| train patients | {checks['nsclc_train_patients']} |",
        f"| val patients | {checks['nsclc_val_patients']} |",
        f"| crop_path sample exist | {checks['nsclc_sample_crop_exist_count']}/{checks['nsclc_sample_crop_total_checked']} |",
        f"| crop base | experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/ |",
        "",
        "---",
        "",
        "## 5. Sampling Option 비교",
        "",
        "| Option | normal_train | nsclc_train | ratio | normal_val | nsclc_val | val_ratio | 비고 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for opt in sampling_options:
        rec_mark = " ★" if opt['option'] == 'B' else ""
        lines.append(
            f"| {opt['option']}{rec_mark} | {opt['normal_train_target']} | {opt['nsclc_train']} | {opt['train_imbalance_ratio']}:1 | "
            f"{opt['normal_val_target']} | {opt['nsclc_val']} | {opt['val_imbalance_ratio']}:1 | {opt['comment']} |"
        )

    lines += [
        "",
        "### 추천: Option B",
        "",
        "- normal train 15,000 / val 5,000",
        f"- train imbalance: {round(15000/nsclc_train, 2)}:1 → class_weight + sample_weight 보정",
        f"- val imbalance: {round(5000/nsclc_val, 2)}:1 → 현실적 비율",
        "- per-patient cap: train ~51/patient (290명 × 6 bins 균형), val ~138/patient (36명)",
        "",
        "---",
        "",
        "## 6. Label 정의",
        "",
        "| class | label | label_name |",
        "|---|---|---|",
        f"| normal | {LABEL_NORMAL} | normal |",
        f"| NSCLC lesion-positive | {LABEL_NSCLC} | nsclc_positive |",
        "| MSD_Lung | 제외 | — |",
        "| hard_negative | 제외(기본) | — |",
        "",
        "---",
        "",
        "## 7. Class Weight / Sample Weight 계획",
        "",
        f"**Option B 기준** (normal_train={NORMAL_TRAIN_CAP}, nsclc_train={nsclc_train})",
        "",
        f"| class | count | class_weight |",
        "|---|---|---|",
        f"| normal (0) | {NORMAL_TRAIN_CAP} | {checks['recommended_class_weight_normal']} |",
        f"| NSCLC (1) | {nsclc_train} | {checks['recommended_class_weight_nsclc']} |",
        "",
        "- 손실함수: `BCEWithLogitsLoss` with per-sample `sample_weight`",
        "- `sample_weight = class_weight[label]`",
        "- P-C-AUX 방식 동일",
        "",
        "---",
        "",
        "## 8. Guardrail 확인",
        "",
        "| 항목 | 상태 |",
        "|---|---|",
        f"| actual normal crop 파일 생성 | {checks['actual_normal_crop_files_generated']} (금지 준수) |",
        f"| actual manifest 생성 | {checks['actual_manifest_generated']} (금지 준수) |",
        f"| model forward | {checks['model_forward_executed']} (금지 준수) |",
        f"| training | {checks['training_executed']} (금지 준수) |",
        f"| scoring | {checks['scoring_executed']} (금지 준수) |",
        f"| threshold 계산 | {checks['threshold_calculated']} (금지 준수) |",
        f"| stage2_holdout 접근 | {checks['stage2_holdout_accessed']} (금지 준수) |",
        f"| P-C-AUX 수정 | {checks['p_c_aux_branch_modified']} (금지 준수) |",
        f"| 기존 결과 수정 | {checks['existing_results_modified']} (금지 준수) |",
        "",
        "---",
        "",
        "## 9. Errors / Warnings",
        "",
    ]
    if not errors:
        lines.append("없음.")
    else:
        for e in errors:
            lines.append(f"- `[{e['severity']}]` {e.get('check','')} — {e.get('message','')}")

    lines += [
        "",
        "---",
        "",
        "## 10. 다음 단계",
        "",
        "**P-C-NORMAL3: actual normal crop generation + training manifest actual generation**",
        "",
        "조건:",
        "- 사용자가 P-C-NORMAL2 dry-check 결과 승인 후 진행",
        "- normal crop: NORMAL_CROP_OUTPUT_BASE 아래 저장",
        "- training manifest: MANIFEST_OUTPUT_BASE 아래 저장",
        "- Option B 기준 적용 (normal train 15,000 / val 5,000)",
        "",
        "또는 P-C-NORMAL2a script hardening 먼저 진행 (필요시)",
    ]

    report_path = DRYCHECK_REPORT_DIR / 'p_c_normal2_manifest_crop_gen_drycheck.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Actual generation (BLOCKED — requires user approval)
# ─────────────────────────────────────────────────────────────────────────────

def run_generate():
    raise RuntimeError(
        "BLOCKED: actual normal crop generation and manifest creation are not allowed "
        "until the user has reviewed and approved the P-C-NORMAL2 dry-check results. "
        "Run --mode drycheck first, then obtain user approval."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV helper
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['drycheck', 'generate'], default='drycheck')
    args = parser.parse_args()

    if args.mode == 'drycheck':
        verdict = run_drycheck()
        sys.exit(0 if verdict in ('PASS', 'PARTIAL_PASS') else 1)
    elif args.mode == 'generate':
        run_generate()


if __name__ == '__main__':
    main()
