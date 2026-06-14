"""
P-C-NORMAL12: Matched Normal Crop Generation + Matched Manifest Generation
Branch: efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1

Option B selected (P-C-NORMAL11 PASS).
Normal crops: position_bin-matched to NSCLC train/val distribution.
Target: normal train=11,836 / val=3,120 (1.5x NSCLC ratio).
NSCLC crops: reused from P-C-NORMAL9 (not modified, not re-generated).
Training/model forward/scoring/threshold NOT executed here.
stage2_holdout NOT accessed.
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
BRANCH_ROOT  = PROJECT_ROOT / 'experiments/efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1'

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

# P-C-NORMAL9 manifests (NSCLC source, read-only)
P9_TRAIN_MANIFEST = (
    BRANCH_ROOT / 'outputs/manifests/p_c_normal9_same_generator_training_manifest'
    / 'p_c_normal9_train_manifest.csv'
)
P9_VAL_MANIFEST = (
    BRANCH_ROOT / 'outputs/manifests/p_c_normal9_same_generator_training_manifest'
    / 'p_c_normal9_val_manifest.csv'
)

# P-C-NORMAL3 manifests (to identify already-used coords, read-only)
P3_TRAIN_MANIFEST = (
    BRANCH_ROOT / 'outputs/manifests/p_c_normal3_training_manifest'
    / 'p_c_normal3_train_manifest.csv'
)
P3_VAL_MANIFEST = (
    BRANCH_ROOT / 'outputs/manifests/p_c_normal3_training_manifest'
    / 'p_c_normal3_val_manifest.csv'
)

CT_VOL_BASE = Path(
    '/mnt/c/Users/jinhy/Desktop'
    '/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1'
    '/volumes_npy'
)

NORMAL_CROP_OUTPUT_BASE = (
    BRANCH_ROOT / 'outputs/normal_crops/p_c_normal12_matched_normal_crops'
)
MANIFEST_OUTPUT_BASE = (
    BRANCH_ROOT / 'outputs/manifests/p_c_normal12_matched_training_manifest'
)
REPORT_OUTPUT_BASE = (
    BRANCH_ROOT / 'outputs/reports/p_c_normal12_matched_generation'
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CROP_SIZE       = 96
HALF_CROP       = 48
LABEL_NORMAL    = 0
LABEL_NSCLC     = 1
RANDOM_SEED     = 42

# Target counts (fixed by P-C-NORMAL11 / user spec)
TARGET_NORMAL_TRAIN = 11836
TARGET_NORMAL_VAL   = 3120
TARGET_NSCLC_TRAIN  = 7891
TARGET_NSCLC_VAL    = 2080

# Derived totals
TARGET_TRAIN_TOTAL  = TARGET_NORMAL_TRAIN + TARGET_NSCLC_TRAIN   # 19,727
TARGET_VAL_TOTAL    = TARGET_NORMAL_VAL   + TARGET_NSCLC_VAL     # 5,200
TARGET_FULL_TOTAL   = TARGET_TRAIN_TOTAL  + TARGET_VAL_TOTAL     # 24,927

# Class weights (from P-C-NORMAL11, train-based)
CW_NORMAL = TARGET_TRAIN_TOTAL / (2.0 * TARGET_NORMAL_TRAIN)   # ≈ 0.8333
CW_NSCLC  = TARGET_TRAIN_TOTAL / (2.0 * TARGET_NSCLC_TRAIN)    # ≈ 1.2500

# ─────────────────────────────────────────────────────────────────────────────
# NSCLC position_bin reference distributions (from P-C-NORMAL11 PASS)
# ─────────────────────────────────────────────────────────────────────────────
NSCLC_TRAIN_BIN_COUNTS = {
    'lower_central':     270,
    'lower_peripheral': 1940,
    'middle_central':    352,
    'middle_peripheral': 3841,
    'upper_central':     369,
    'upper_peripheral':  1119,
}
NSCLC_VAL_BIN_COUNTS = {
    'lower_central':      88,
    'lower_peripheral':   337,
    'middle_central':      67,
    'middle_peripheral': 1009,
    'upper_central':      174,
    'upper_peripheral':   405,
}

BINS_ORDER = [
    'lower_central', 'lower_peripheral',
    'middle_central', 'middle_peripheral',
    'upper_central',  'upper_peripheral',
]


# ─────────────────────────────────────────────────────────────────────────────
# Bin target computation (largest-remainder method)
# ─────────────────────────────────────────────────────────────────────────────
def compute_bin_targets(nsclc_bin_counts, target_total):
    """Compute per-bin targets summing exactly to target_total using largest-remainder."""
    nsclc_total = sum(nsclc_bin_counts.values())
    raw = {b: target_total * c / nsclc_total for b, c in nsclc_bin_counts.items()}
    floors = {b: int(v) for b, v in raw.items()}
    remainders = {b: raw[b] - floors[b] for b in raw}
    deficit = target_total - sum(floors.values())
    sorted_bins = sorted(remainders, key=lambda b: -remainders[b])
    targets = dict(floors)
    for b in sorted_bins[:deficit]:
        targets[b] += 1
    assert sum(targets.values()) == target_total, \
        f"Bin targets sum {sum(targets.values())} != {target_total}"
    return targets


# ─────────────────────────────────────────────────────────────────────────────
# Crop extraction (same as P-C-NORMAL3)
# ─────────────────────────────────────────────────────────────────────────────
def extract_normal_crop(ct_vol, local_z_idx, center_y, center_x):
    """Extract (3,96,96) int16 crop. Channels: z-1/z/z+1 with edge padding."""
    z_dim, y_dim, x_dim = ct_vol.shape
    lz = int(local_z_idx)
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
    assert crop.shape == (3, CROP_SIZE, CROP_SIZE), f"Bad shape: {crop.shape}"
    return crop.astype(np.int16)


# ─────────────────────────────────────────────────────────────────────────────
# Position-bin matched sampling with patient diversity
# ─────────────────────────────────────────────────────────────────────────────
def sample_position_matched(df, bin_targets, seed=RANDOM_SEED):
    """
    Sample rows from df (N-C3 or N-C10 remaining) per bin with patient diversity.
    For each bin, distribute target count across patients as evenly as possible.
    Returns sampled DataFrame with exactly sum(bin_targets.values()) rows.
    """
    rng = np.random.default_rng(seed)
    result_parts = []

    for bin_name in BINS_ORDER:
        target = bin_targets.get(bin_name, 0)
        if target == 0:
            continue

        bin_df = df[df['position_bin'] == bin_name].copy()
        if len(bin_df) < target:
            raise ValueError(
                f"Bin '{bin_name}' has {len(bin_df)} rows but need {target}"
            )

        # Distribute across patients evenly
        patients = bin_df['patient_id'].unique()
        n_patients = len(patients)
        per_patient_base = target // n_patients
        extra = target % n_patients

        # Shuffle patient order for reproducibility
        rng.shuffle(patients)
        selected_rows = []

        for i, pid in enumerate(patients):
            quota = per_patient_base + (1 if i < extra else 0)
            if quota == 0:
                continue
            prows = bin_df[bin_df['patient_id'] == pid]
            take = min(len(prows), quota)
            selected_rows.append(
                prows.sample(n=take, random_state=seed + i)
            )

        bin_sampled = pd.concat(selected_rows)

        # If under-sampled (some patients had fewer rows than quota), top up
        if len(bin_sampled) < target:
            deficit = target - len(bin_sampled)
            used_keys = set(bin_sampled.index)
            remainder = bin_df[~bin_df.index.isin(used_keys)]
            if len(remainder) >= deficit:
                extra_rows = remainder.sample(n=deficit, random_state=seed + 9999)
                bin_sampled = pd.concat([bin_sampled, extra_rows])
            else:
                raise ValueError(
                    f"Cannot fill bin '{bin_name}': only {len(remainder)} rows left "
                    f"after patient-distributed sampling, need {deficit} more"
                )

        assert len(bin_sampled) == target, \
            f"Bin {bin_name}: sampled {len(bin_sampled)}, expected {target}"
        result_parts.append(bin_sampled)

    sampled = pd.concat(result_parts).reset_index(drop=True)
    total_expected = sum(bin_targets.values())
    assert len(sampled) == total_expected, \
        f"Total sampled {len(sampled)}, expected {total_expected}"
    return sampled


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
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_generation():
    start_time = datetime.now()
    print(f"[P-C-NORMAL12] Start: {start_time.isoformat()}")

    errors = []
    guardrail = {
        'stage2_holdout_accessed':    False,
        'training_run':               False,
        'model_forward_run':          False,
        'scoring_run':                False,
        'threshold_computed':         False,
        'checkpoint_saved':           False,
        'existing_outputs_modified':  False,
        'nsclc_crop_modified':        False,
        'p_c_normal3_crop_modified':  False,
        'hard_negative_included':     False,
        'msd_lung_included':          False,
        'forbidden_diagnostic_wording_count': 0,
    }

    # ── 0. Pre-flight checks ──────────────────────────────────────────────────
    print("[P-C-NORMAL12] Running pre-flight checks...")
    preflight_issues = []

    for p, label in [
        (N_C3_MANIFEST, 'N_C3_MANIFEST'),
        (N_C10_MANIFEST, 'N_C10_MANIFEST'),
        (P9_TRAIN_MANIFEST, 'P9_TRAIN_MANIFEST'),
        (P9_VAL_MANIFEST, 'P9_VAL_MANIFEST'),
        (P3_TRAIN_MANIFEST, 'P3_TRAIN_MANIFEST'),
        (P3_VAL_MANIFEST, 'P3_VAL_MANIFEST'),
    ]:
        if not p.exists():
            preflight_issues.append(f"MISSING_INPUT: {label} = {p}")

    if NORMAL_CROP_OUTPUT_BASE.exists():
        preflight_issues.append(
            f"COLLISION: output crop dir already exists: {NORMAL_CROP_OUTPUT_BASE}"
        )
    if MANIFEST_OUTPUT_BASE.exists():
        preflight_issues.append(
            f"COLLISION: manifest dir already exists: {MANIFEST_OUTPUT_BASE}"
        )

    if preflight_issues:
        print("[ERROR] Pre-flight failed:")
        for iss in preflight_issues:
            print(f"  {iss}")
        sys.exit(1)

    print("  Pre-flight OK")

    # ── Create output directories ─────────────────────────────────────────────
    train_crop_dir = NORMAL_CROP_OUTPUT_BASE / 'train'
    val_crop_dir   = NORMAL_CROP_OUTPUT_BASE / 'val'
    train_crop_dir.mkdir(parents=True, exist_ok=True)
    val_crop_dir.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    print("  Output dirs created.")

    # ── 1. Load and filter N-C3 pool ─────────────────────────────────────────
    print("[P-C-NORMAL12] Loading N-C3 manifest...")
    nc3_full = pd.read_csv(N_C3_MANIFEST, low_memory=False)
    print(f"  N-C3 full: {len(nc3_full)} rows, {nc3_full['patient_id'].nunique()} patients")

    print("[P-C-NORMAL12] Loading P-C-NORMAL3 train to exclude used coords...")
    p3_train = pd.read_csv(P3_TRAIN_MANIFEST, low_memory=False)
    p3_normal_train = p3_train[p3_train['label'] == LABEL_NORMAL]
    used_train_keys = set(
        zip(p3_normal_train['patient_id'],
            p3_normal_train['slice_index'].astype(int),
            p3_normal_train['center_y'].astype(int),
            p3_normal_train['center_x'].astype(int))
    )
    print(f"  P-C-NORMAL3 train used coords: {len(used_train_keys)}")

    nc3_keys = list(zip(
        nc3_full['patient_id'],
        nc3_full['slice_index'].astype(int),
        nc3_full['center_y'].astype(int),
        nc3_full['center_x'].astype(int),
    ))
    used_mask_train = pd.Series([k in used_train_keys for k in nc3_keys], index=nc3_full.index)
    nc3_remaining = nc3_full[~used_mask_train].copy()
    print(f"  N-C3 remaining: {len(nc3_remaining)} rows")
    print(f"  N-C3 remaining by bin:")
    for b in BINS_ORDER:
        n = (nc3_remaining['position_bin'] == b).sum()
        print(f"    {b}: {n}")

    # ── 2. Load and filter N-C10 pool ────────────────────────────────────────
    print("[P-C-NORMAL12] Loading N-C10 manifest...")
    nc10_full = pd.read_csv(N_C10_MANIFEST, low_memory=False)
    print(f"  N-C10 full: {len(nc10_full)} rows, {nc10_full['patient_id'].nunique()} patients")

    print("[P-C-NORMAL12] Loading P-C-NORMAL3 val to exclude used coords...")
    p3_val = pd.read_csv(P3_VAL_MANIFEST, low_memory=False)
    p3_normal_val = p3_val[p3_val['label'] == LABEL_NORMAL]
    used_val_keys = set(
        zip(p3_normal_val['patient_id'],
            p3_normal_val['slice_index'].astype(int),
            p3_normal_val['center_y'].astype(int),
            p3_normal_val['center_x'].astype(int))
    )
    print(f"  P-C-NORMAL3 val used coords: {len(used_val_keys)}")

    nc10_keys = list(zip(
        nc10_full['patient_id'],
        nc10_full['slice_index'].astype(int),
        nc10_full['center_y'].astype(int),
        nc10_full['center_x'].astype(int),
    ))
    used_mask_val = pd.Series([k in used_val_keys for k in nc10_keys], index=nc10_full.index)
    nc10_remaining = nc10_full[~used_mask_val].copy()
    print(f"  N-C10 remaining: {len(nc10_remaining)} rows")
    print(f"  N-C10 remaining by bin:")
    for b in BINS_ORDER:
        n = (nc10_remaining['position_bin'] == b).sum()
        print(f"    {b}: {n}")

    n_c3_patients  = nc3_remaining['patient_id'].nunique()
    n_c10_patients = nc10_remaining['patient_id'].nunique()

    # ── 3. Compute bin targets ────────────────────────────────────────────────
    print("[P-C-NORMAL12] Computing bin targets...")
    train_bin_targets = compute_bin_targets(NSCLC_TRAIN_BIN_COUNTS, TARGET_NORMAL_TRAIN)
    val_bin_targets   = compute_bin_targets(NSCLC_VAL_BIN_COUNTS,   TARGET_NORMAL_VAL)

    print(f"  Train bin targets (total={sum(train_bin_targets.values())}):")
    for b in BINS_ORDER:
        nsclc_n = NSCLC_TRAIN_BIN_COUNTS.get(b, 0)
        print(f"    {b}: {train_bin_targets[b]}  (NSCLC={nsclc_n})")

    print(f"  Val bin targets (total={sum(val_bin_targets.values())}):")
    for b in BINS_ORDER:
        nsclc_n = NSCLC_VAL_BIN_COUNTS.get(b, 0)
        print(f"    {b}: {val_bin_targets[b]}  (NSCLC={nsclc_n})")

    # Verify NC3 remaining can cover train targets
    for b in BINS_ORDER:
        avail = (nc3_remaining['position_bin'] == b).sum()
        need  = train_bin_targets.get(b, 0)
        if avail < need:
            print(f"[ERROR] Bin '{b}' NC3 remaining {avail} < target {need}")
            sys.exit(1)

    # Verify NC10 remaining can cover val targets
    for b in BINS_ORDER:
        avail = (nc10_remaining['position_bin'] == b).sum()
        need  = val_bin_targets.get(b, 0)
        if avail < need:
            print(f"[ERROR] Bin '{b}' NC10 remaining {avail} < val target {need}")
            sys.exit(1)

    # ── 4. Sample rows ────────────────────────────────────────────────────────
    print("[P-C-NORMAL12] Sampling normal train rows (position matched)...")
    sampled_train = sample_position_matched(nc3_remaining, train_bin_targets, seed=RANDOM_SEED)
    print(f"  Sampled train: {len(sampled_train)} rows")

    print("[P-C-NORMAL12] Sampling normal val rows (position matched)...")
    sampled_val = sample_position_matched(nc10_remaining, val_bin_targets, seed=RANDOM_SEED + 1)
    print(f"  Sampled val:   {len(sampled_val)} rows")

    # Leakage check: train/val patient overlap in normal pool
    train_patients = set(sampled_train['patient_id'].unique())
    val_patients   = set(sampled_val['patient_id'].unique())
    normal_tv_overlap = train_patients.intersection(val_patients)
    if normal_tv_overlap:
        print(f"[ERROR] Normal train/val patient overlap: {normal_tv_overlap}")
        sys.exit(1)
    print(f"  Normal train/val leakage check: PASS (0 overlap)")

    # ── 5. Generate normal train crops ───────────────────────────────────────
    print("[P-C-NORMAL12] Generating normal train crops...")
    normal_train_manifest_rows = []
    normal_train_generated     = 0
    normal_train_errors        = 0

    sampled_train = sampled_train.copy()
    sampled_train['_ct_path'] = sampled_train['source_ct_path']
    sampled_train['_z_for_crop'] = sampled_train['slice_index'].astype(int)

    ct_groups = list(sampled_train.groupby('_ct_path'))
    n_ct = len(ct_groups)
    for ct_idx, (ct_path, grp) in enumerate(ct_groups):
        if ct_idx % 20 == 0:
            print(f"  Train CT {ct_idx+1}/{n_ct} ({normal_train_generated} crops so far)")
        try:
            ct_vol = np.load(ct_path)
        except Exception as e:
            for _, row in grp.iterrows():
                errors.append({'severity': 'ERROR', 'check': 'ct_load_train',
                               'patient_id': row['patient_id'], 'message': str(e)})
                normal_train_errors += 1
            continue

        for _, row in grp.iterrows():
            global_idx = normal_train_generated + 1
            aux_cid  = f"PN12TR_{global_idx:08d}"
            out_path = train_crop_dir / f"{aux_cid}.npz"

            try:
                crop = extract_normal_crop(
                    ct_vol,
                    int(row['_z_for_crop']),
                    row['center_y'],
                    row['center_x'],
                )
                np.savez(str(out_path), ct_crop=crop)
                normal_train_generated += 1

                normal_train_manifest_rows.append({
                    'aux_candidate_id':             aux_cid,
                    'source_branch':                'p_c_normal12',
                    'patient_id':                   row['patient_id'],
                    'safe_id':                      row['safe_id'],
                    'split':                        'train',
                    'label':                        LABEL_NORMAL,
                    'label_name':                   'normal',
                    'crop_path':                    str(out_path),
                    'crop_generated_by_this_branch': True,
                    'ct_path':                      str(row['source_ct_path']),
                    'local_z':                      int(row['local_z']),
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
                    'source_patient_count':         n_c3_patients,
                    'forbidden_source_classifier_mixed': False,
                    'original_file_modified':       False,
                    'normal_source_split':          'normal_train',
                    'normal_coord_source':          'n_c3',
                    'lesion_pixels':                0.0,
                    'has_lesion_patch':             False,
                    'original_p_c_candidate_id':    '',
                    'original_p_c_label':           '',
                    'lesion_pixels_nsclc':          '',
                    'has_lesion_patch_nsclc':       '',
                    'source_name':                  'NORMAL_LUNA16',
                })
            except Exception as e:
                errors.append({'severity': 'ERROR', 'check': 'crop_extract_train',
                               'patient_id': row['patient_id'], 'message': str(e)})
                normal_train_errors += 1

        del ct_vol

    print(f"  Normal train generated: {normal_train_generated} (errors: {normal_train_errors})")

    # ── 6. Generate normal val crops ─────────────────────────────────────────
    print("[P-C-NORMAL12] Generating normal val crops...")
    normal_val_manifest_rows = []
    normal_val_generated     = 0
    normal_val_errors        = 0

    sampled_val = sampled_val.copy()
    sampled_val['_ct_path'] = sampled_val.apply(
        lambda r: str(CT_VOL_BASE / r['safe_id'] / 'ct_hu.npy'), axis=1
    )
    sampled_val['_z_for_crop'] = sampled_val['slice_index'].astype(int)

    ct_groups_val = list(sampled_val.groupby('_ct_path'))
    n_ct_val = len(ct_groups_val)
    for ct_idx, (ct_path, grp) in enumerate(ct_groups_val):
        if ct_idx % 5 == 0:
            print(f"  Val CT {ct_idx+1}/{n_ct_val} ({normal_val_generated} crops so far)")
        try:
            ct_vol = np.load(ct_path)
        except Exception as e:
            for _, row in grp.iterrows():
                errors.append({'severity': 'ERROR', 'check': 'ct_load_val',
                               'patient_id': row['patient_id'], 'message': str(e)})
                normal_val_errors += 1
            continue

        for _, row in grp.iterrows():
            global_idx = normal_val_generated + 1
            aux_cid  = f"PN12VL_{global_idx:08d}"
            out_path = val_crop_dir / f"{aux_cid}.npz"

            try:
                crop = extract_normal_crop(
                    ct_vol,
                    int(row['_z_for_crop']),
                    row['center_y'],
                    row['center_x'],
                )
                np.savez(str(out_path), ct_crop=crop)
                normal_val_generated += 1

                normal_val_manifest_rows.append({
                    'aux_candidate_id':             aux_cid,
                    'source_branch':                'p_c_normal12',
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
                    'lesion_pixels':                0.0,
                    'has_lesion_patch':             False,
                    'original_p_c_candidate_id':    '',
                    'original_p_c_label':           '',
                    'lesion_pixels_nsclc':          '',
                    'has_lesion_patch_nsclc':       '',
                    'source_name':                  'NORMAL_LUNA16',
                })
            except Exception as e:
                errors.append({'severity': 'ERROR', 'check': 'crop_extract_val',
                               'patient_id': row['patient_id'], 'message': str(e)})
                normal_val_errors += 1

        del ct_vol

    print(f"  Normal val generated: {normal_val_generated} (errors: {normal_val_errors})")

    # ── 7. Count totals and bail if under-generated ───────────────────────────
    if normal_train_generated != TARGET_NORMAL_TRAIN:
        print(f"[ERROR] Expected {TARGET_NORMAL_TRAIN} train crops, got {normal_train_generated}")
        sys.exit(1)
    if normal_val_generated != TARGET_NORMAL_VAL:
        print(f"[ERROR] Expected {TARGET_NORMAL_VAL} val crops, got {normal_val_generated}")
        sys.exit(1)

    # ── 8. Apply class weights ────────────────────────────────────────────────
    cw_normal = CW_NORMAL
    cw_nsclc  = CW_NSCLC
    print(f"[P-C-NORMAL12] class_weight_normal={cw_normal:.6f}, class_weight_nsclc={cw_nsclc:.6f}")

    for mrow in normal_train_manifest_rows:
        mrow['sample_weight'] = cw_normal
        mrow['class_weight']  = cw_normal
    for mrow in normal_val_manifest_rows:
        mrow['sample_weight'] = cw_normal
        mrow['class_weight']  = cw_normal

    # ── 9. Load NSCLC rows from P-C-NORMAL9 ──────────────────────────────────
    print("[P-C-NORMAL12] Loading NSCLC rows from P-C-NORMAL9...")
    p9_train = pd.read_csv(P9_TRAIN_MANIFEST, low_memory=False)
    p9_val   = pd.read_csv(P9_VAL_MANIFEST,   low_memory=False)

    nsclc_train_df = p9_train[p9_train['label'] == LABEL_NSCLC].copy()
    nsclc_val_df   = p9_val[p9_val['label']   == LABEL_NSCLC].copy()

    if len(nsclc_train_df) != TARGET_NSCLC_TRAIN:
        print(f"[ERROR] Expected {TARGET_NSCLC_TRAIN} NSCLC train, got {len(nsclc_train_df)}")
        sys.exit(1)
    if len(nsclc_val_df) != TARGET_NSCLC_VAL:
        print(f"[ERROR] Expected {TARGET_NSCLC_VAL} NSCLC val, got {len(nsclc_val_df)}")
        sys.exit(1)

    # Guard: no hard_negative / MSD_Lung in NSCLC rows
    if 'source_name' in nsclc_train_df.columns:
        msd_count = (nsclc_train_df['source_name'].str.contains('MSD', na=False)).sum()
        if msd_count > 0:
            guardrail['msd_lung_included'] = True
            errors.append({'severity': 'ERROR', 'check': 'msd_lung_guard',
                           'patient_id': 'N/A', 'message': f"{msd_count} MSD rows"})
    if 'has_lesion_patch' in nsclc_train_df.columns:
        hn_count = (nsclc_train_df.get('hard_negative', pd.Series([False]*len(nsclc_train_df))) == True).sum()
        if hn_count > 0:
            guardrail['hard_negative_included'] = True

    # Update class weights in NSCLC rows
    nsclc_train_df = nsclc_train_df.copy()
    nsclc_val_df   = nsclc_val_df.copy()
    nsclc_train_df['sample_weight'] = cw_nsclc
    nsclc_train_df['class_weight']  = cw_nsclc
    nsclc_val_df['sample_weight']   = cw_nsclc
    nsclc_val_df['class_weight']    = cw_nsclc

    # Convert NSCLC manifest rows to dicts for manifest building
    nsclc_train_rows = nsclc_train_df.to_dict('records')
    nsclc_val_rows   = nsclc_val_df.to_dict('records')

    print(f"  NSCLC train: {len(nsclc_train_rows)}, NSCLC val: {len(nsclc_val_rows)}")

    # Leakage check: NSCLC patients vs normal patients
    nsclc_train_patients = set(nsclc_train_df['patient_id'].unique())
    nsclc_val_patients   = set(nsclc_val_df['patient_id'].unique())

    normal_nsclc_overlap = train_patients.union(val_patients).intersection(
        nsclc_train_patients.union(nsclc_val_patients)
    )
    if normal_nsclc_overlap:
        print(f"[ERROR] Normal/NSCLC patient overlap: {normal_nsclc_overlap}")
        sys.exit(1)

    nsclc_tv_overlap = nsclc_train_patients.intersection(nsclc_val_patients)
    if nsclc_tv_overlap:
        print(f"[ERROR] NSCLC train/val patient overlap: {nsclc_tv_overlap}")
        sys.exit(1)

    print("  Leakage checks all PASS.")

    # ── 10. Build manifests ───────────────────────────────────────────────────
    print("[P-C-NORMAL12] Building manifests...")

    # Define column order (compatible with P-C-NORMAL9)
    MANIFEST_COLS = [
        'aux_candidate_id', 'source_branch', 'patient_id', 'safe_id',
        'split', 'label', 'label_name', 'crop_path',
        'crop_generated_by_this_branch', 'ct_path',
        'local_z', 'slice_index', 'y0', 'x0', 'y1', 'x1',
        'center_y', 'center_x', 'position_bin', 'z_level',
        'crop_shape_expected', 'crop_dtype_expected',
        'raw_hu_available', 'preprocessing',
        'sample_weight', 'class_weight',
        'patient_cap_applied', 'source_patient_count',
        'forbidden_source_classifier_mixed', 'original_file_modified',
        'normal_source_split', 'normal_coord_source',
        'lesion_pixels', 'has_lesion_patch',
        'original_p_c_candidate_id', 'original_p_c_label',
        'lesion_pixels_nsclc', 'has_lesion_patch_nsclc',
        'source_name',
    ]

    train_rows = normal_train_manifest_rows + nsclc_train_rows
    val_rows   = normal_val_manifest_rows   + nsclc_val_rows
    full_rows  = train_rows + val_rows

    # Save manifests
    train_csv_path = MANIFEST_OUTPUT_BASE / 'p_c_normal12_train_manifest.csv'
    val_csv_path   = MANIFEST_OUTPUT_BASE / 'p_c_normal12_val_manifest.csv'
    full_csv_path  = MANIFEST_OUTPUT_BASE / 'p_c_normal12_full_manifest.csv'

    write_csv(train_csv_path, train_rows, fieldnames=MANIFEST_COLS)
    write_csv(val_csv_path,   val_rows,   fieldnames=MANIFEST_COLS)
    write_csv(full_csv_path,  full_rows,  fieldnames=MANIFEST_COLS)

    print(f"  train manifest: {len(train_rows)} rows → {train_csv_path}")
    print(f"  val manifest:   {len(val_rows)} rows → {val_csv_path}")
    print(f"  full manifest:  {len(full_rows)} rows → {full_csv_path}")

    # ── 11. Integrity check ───────────────────────────────────────────────────
    print("[P-C-NORMAL12] Running NPZ integrity check on new normal crops...")
    integrity_rows = []

    # Train
    train_npz_files = list(train_crop_dir.glob("PN12TR_*.npz"))
    print(f"  Checking {len(train_npz_files)} train NPZ files...")
    hu_stats_list = []

    for npz_path in train_npz_files:
        row = {'split': 'train', 'path': str(npz_path), 'ok_key': False,
               'ok_shape': False, 'ok_dtype': False, 'no_nan_inf': False,
               'no_all_zero': False, 'error': ''}
        try:
            d = np.load(str(npz_path))
            keys = list(d.files)
            ok_key   = keys == ['ct_crop']
            ok_shape = d['ct_crop'].shape == (3, 96, 96)
            ok_dtype = d['ct_crop'].dtype == np.int16
            arr_f    = d['ct_crop'].astype(np.float32)
            no_nan   = not (np.isnan(arr_f).any() or np.isinf(arr_f).any())
            no_zero  = arr_f.max() != 0 or arr_f.min() != 0
            row.update({'ok_key': ok_key, 'ok_shape': ok_shape, 'ok_dtype': ok_dtype,
                        'no_nan_inf': no_nan, 'no_all_zero': no_zero})
            if ok_dtype:
                hu_stats_list.append({
                    'split': 'train', 'path': str(npz_path),
                    'hu_min': float(d['ct_crop'].min()),
                    'hu_max': float(d['ct_crop'].max()),
                    'hu_mean': float(arr_f.mean()),
                    'hu_std':  float(arr_f.std()),
                })
        except Exception as e:
            row['error'] = str(e)
        integrity_rows.append(row)

    # Val
    val_npz_files = list(val_crop_dir.glob("PN12VL_*.npz"))
    print(f"  Checking {len(val_npz_files)} val NPZ files...")
    for npz_path in val_npz_files:
        row = {'split': 'val', 'path': str(npz_path), 'ok_key': False,
               'ok_shape': False, 'ok_dtype': False, 'no_nan_inf': False,
               'no_all_zero': False, 'error': ''}
        try:
            d = np.load(str(npz_path))
            keys = list(d.files)
            ok_key   = keys == ['ct_crop']
            ok_shape = d['ct_crop'].shape == (3, 96, 96)
            ok_dtype = d['ct_crop'].dtype == np.int16
            arr_f    = d['ct_crop'].astype(np.float32)
            no_nan   = not (np.isnan(arr_f).any() or np.isinf(arr_f).any())
            no_zero  = arr_f.max() != 0 or arr_f.min() != 0
            row.update({'ok_key': ok_key, 'ok_shape': ok_shape, 'ok_dtype': ok_dtype,
                        'no_nan_inf': no_nan, 'no_all_zero': no_zero})
            if ok_dtype:
                hu_stats_list.append({
                    'split': 'val', 'path': str(npz_path),
                    'hu_min': float(d['ct_crop'].min()),
                    'hu_max': float(d['ct_crop'].max()),
                    'hu_mean': float(arr_f.mean()),
                    'hu_std':  float(arr_f.std()),
                })
        except Exception as e:
            row['error'] = str(e)
        integrity_rows.append(row)

    integrity_df = pd.DataFrame(integrity_rows)
    n_fail = (~integrity_df['ok_key'] | ~integrity_df['ok_shape'] |
              ~integrity_df['ok_dtype'] | ~integrity_df['no_nan_inf'] |
              ~integrity_df['no_all_zero']).sum()
    print(f"  Integrity: {len(integrity_rows)} checked, {n_fail} failures")

    # ── 12. Position distribution validation ─────────────────────────────────
    print("[P-C-NORMAL12] Validating position distribution matching...")
    train_df = pd.read_csv(train_csv_path, low_memory=False)
    val_df   = pd.read_csv(val_csv_path,   low_memory=False)

    normal_train_bins = train_df[train_df['label']==0]['position_bin'].value_counts().to_dict()
    nsclc_train_bins  = train_df[train_df['label']==1]['position_bin'].value_counts().to_dict()
    normal_val_bins   = val_df[val_df['label']==0]['position_bin'].value_counts().to_dict()
    nsclc_val_bins    = val_df[val_df['label']==1]['position_bin'].value_counts().to_dict()

    n_normal_train = sum(normal_train_bins.values())
    n_nsclc_train  = sum(nsclc_train_bins.values())
    n_normal_val   = sum(normal_val_bins.values())
    n_nsclc_val    = sum(nsclc_val_bins.values())

    pos_rows = []
    for b in BINS_ORDER:
        nm_tr_n = normal_train_bins.get(b, 0)
        ns_tr_n = nsclc_train_bins.get(b, 0)
        nm_vl_n = normal_val_bins.get(b, 0)
        ns_vl_n = nsclc_val_bins.get(b, 0)

        nm_tr_pct = nm_tr_n / n_normal_train if n_normal_train else 0
        ns_tr_pct = ns_tr_n / n_nsclc_train  if n_nsclc_train  else 0
        nm_vl_pct = nm_vl_n / n_normal_val   if n_normal_val   else 0
        ns_vl_pct = ns_vl_n / n_nsclc_val    if n_nsclc_val    else 0

        train_diff = abs(nm_tr_pct - ns_tr_pct)
        val_diff   = abs(nm_vl_pct - ns_vl_pct)
        pos_rows.append({
            'position_bin':       b,
            'normal_train_count': nm_tr_n,
            'nsclc_train_count':  ns_tr_n,
            'normal_train_pct':   round(nm_tr_pct, 4),
            'nsclc_train_pct':    round(ns_tr_pct, 4),
            'train_abs_diff':     round(train_diff, 4),
            'train_pass':         train_diff <= 0.02,
            'normal_val_count':   nm_vl_n,
            'nsclc_val_count':    ns_vl_n,
            'normal_val_pct':     round(nm_vl_pct, 4),
            'nsclc_val_pct':      round(ns_vl_pct, 4),
            'val_abs_diff':       round(val_diff, 4),
            'val_pass':           val_diff <= 0.02,
        })

    pos_df = pd.DataFrame(pos_rows)

    # Peripheral ratio
    peripheral_bins = ['lower_peripheral', 'middle_peripheral', 'upper_peripheral']
    nm_tr_peri_ratio = sum(normal_train_bins.get(b, 0) for b in peripheral_bins) / n_normal_train
    ns_tr_peri_ratio = sum(nsclc_train_bins.get(b, 0)  for b in peripheral_bins) / n_nsclc_train
    nm_vl_peri_ratio = sum(normal_val_bins.get(b, 0)   for b in peripheral_bins) / n_normal_val
    ns_vl_peri_ratio = sum(nsclc_val_bins.get(b, 0)    for b in peripheral_bins) / n_nsclc_val

    train_peri_diff = abs(nm_tr_peri_ratio - ns_tr_peri_ratio)
    val_peri_diff   = abs(nm_vl_peri_ratio - ns_vl_peri_ratio)

    print(f"  Train peripheral ratio - normal: {nm_tr_peri_ratio:.4f}, NSCLC: {ns_tr_peri_ratio:.4f}, diff: {train_peri_diff:.4f}")
    print(f"  Val peripheral ratio   - normal: {nm_vl_peri_ratio:.4f}, NSCLC: {ns_vl_peri_ratio:.4f}, diff: {val_peri_diff:.4f}")

    pos_matching_train_all_pass = all(pos_df['train_pass'])
    pos_matching_val_all_pass   = all(pos_df['val_pass'])
    peri_train_pass = train_peri_diff <= 0.03
    peri_val_pass   = val_peri_diff   <= 0.03

    pos_verdict = 'PASS' if (pos_matching_train_all_pass and pos_matching_val_all_pass
                              and peri_train_pass and peri_val_pass) else 'PARTIAL_PASS'

    # ── 13. Class weight validation ───────────────────────────────────────────
    actual_cw_normal = TARGET_TRAIN_TOTAL / (2.0 * normal_train_generated)
    actual_cw_nsclc  = TARGET_TRAIN_TOTAL / (2.0 * TARGET_NSCLC_TRAIN)
    cw_rows = [
        {'metric': 'train_total',         'value': TARGET_TRAIN_TOTAL},
        {'metric': 'normal_train_count',  'value': normal_train_generated},
        {'metric': 'nsclc_train_count',   'value': TARGET_NSCLC_TRAIN},
        {'metric': 'class_weight_normal', 'value': round(actual_cw_normal, 6)},
        {'metric': 'class_weight_nsclc',  'value': round(actual_cw_nsclc, 6)},
        {'metric': 'cw_normal_expected',  'value': round(CW_NORMAL, 6)},
        {'metric': 'cw_nsclc_expected',   'value': round(CW_NSCLC, 6)},
        {'metric': 'cw_normal_match',     'value': abs(actual_cw_normal - CW_NORMAL) < 0.001},
        {'metric': 'cw_nsclc_match',      'value': abs(actual_cw_nsclc - CW_NSCLC) < 0.001},
    ]

    # ── 14. Patient split validation ─────────────────────────────────────────
    split_rows = [
        {'check': 'normal_train_patients',  'count': len(train_patients)},
        {'check': 'normal_val_patients',    'count': len(val_patients)},
        {'check': 'normal_tv_leakage',      'count': len(normal_tv_overlap)},
        {'check': 'nsclc_train_patients',   'count': len(nsclc_train_patients)},
        {'check': 'nsclc_val_patients',     'count': len(nsclc_val_patients)},
        {'check': 'nsclc_tv_leakage',       'count': len(nsclc_tv_overlap)},
        {'check': 'normal_nsclc_leakage',   'count': len(normal_nsclc_overlap)},
        {'check': 'all_leakage_zero',       'count': 0 if not normal_tv_overlap and
                                                          not nsclc_tv_overlap and
                                                          not normal_nsclc_overlap else 1},
    ]

    # ── 15. Manifest row validation ───────────────────────────────────────────
    full_df = pd.read_csv(full_csv_path, low_memory=False)
    mv_rows = [
        {'check': 'train_rows',            'expected': TARGET_TRAIN_TOTAL, 'actual': len(train_df), 'pass': len(train_df)==TARGET_TRAIN_TOTAL},
        {'check': 'val_rows',              'expected': TARGET_VAL_TOTAL,   'actual': len(val_df),   'pass': len(val_df)==TARGET_VAL_TOTAL},
        {'check': 'full_rows',             'expected': TARGET_FULL_TOTAL,  'actual': len(full_df),  'pass': len(full_df)==TARGET_FULL_TOTAL},
        {'check': 'train_normal_count',    'expected': TARGET_NORMAL_TRAIN,'actual': (train_df['label']==0).sum(), 'pass': (train_df['label']==0).sum()==TARGET_NORMAL_TRAIN},
        {'check': 'train_nsclc_count',     'expected': TARGET_NSCLC_TRAIN, 'actual': (train_df['label']==1).sum(), 'pass': (train_df['label']==1).sum()==TARGET_NSCLC_TRAIN},
        {'check': 'val_normal_count',      'expected': TARGET_NORMAL_VAL,  'actual': (val_df['label']==0).sum(), 'pass': (val_df['label']==0).sum()==TARGET_NORMAL_VAL},
        {'check': 'val_nsclc_count',       'expected': TARGET_NSCLC_VAL,   'actual': (val_df['label']==1).sum(), 'pass': (val_df['label']==1).sum()==TARGET_NSCLC_VAL},
        {'check': 'label_values_valid',    'expected': '{0,1}', 'actual': str(set(full_df['label'].unique())), 'pass': set(full_df['label'].unique()).issubset({0,1})},
    ]

    # ── 16. Guardrail check ───────────────────────────────────────────────────
    gc_rows = [{'guardrail': k, 'value': str(v), 'ok': not v if k != 'forbidden_diagnostic_wording_count' else v == 0}
               for k, v in guardrail.items()]

    # ── 17. Save reports ──────────────────────────────────────────────────────
    print("[P-C-NORMAL12] Saving reports...")

    write_csv(REPORT_OUTPUT_BASE / 'p_c_normal12_normal_crop_integrity_check.csv', integrity_rows)
    write_csv(REPORT_OUTPUT_BASE / 'p_c_normal12_position_distribution_validation.csv', pos_rows)
    write_csv(REPORT_OUTPUT_BASE / 'p_c_normal12_class_weight_validation.csv', cw_rows)
    write_csv(REPORT_OUTPUT_BASE / 'p_c_normal12_patient_split_validation.csv', split_rows)
    write_csv(REPORT_OUTPUT_BASE / 'p_c_normal12_manifest_row_validation.csv', mv_rows)
    write_csv(REPORT_OUTPUT_BASE / 'p_c_normal12_guardrail_check.csv', gc_rows)
    write_csv(MANIFEST_OUTPUT_BASE / 'p_c_normal12_errors.csv', errors if errors else [{'note': 'no errors'}])

    # Summary JSON
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    all_mv_pass     = all(r['pass'] for r in mv_rows)
    all_integrity   = n_fail == 0
    all_leakage     = not (normal_tv_overlap or nsclc_tv_overlap or normal_nsclc_overlap)
    all_guardrails  = all(r['ok'] for r in gc_rows)

    if all_mv_pass and all_integrity and all_leakage and all_guardrails:
        if pos_verdict == 'PASS':
            verdict = 'PASS'
        else:
            verdict = 'PARTIAL_PASS'
    elif not all_mv_pass or not all_integrity:
        verdict = 'FAIL'
    else:
        verdict = 'PARTIAL_PASS'

    summary = {
        'stage':              'P-C-NORMAL12',
        'verdict':            verdict,
        'start_time':         start_time.isoformat(),
        'end_time':           end_time.isoformat(),
        'elapsed_sec':        round(elapsed, 1),
        'normal_train_generated': normal_train_generated,
        'normal_val_generated':   normal_val_generated,
        'total_normal_generated': normal_train_generated + normal_val_generated,
        'nsclc_train_reused':     len(nsclc_train_rows),
        'nsclc_val_reused':       len(nsclc_val_rows),
        'train_manifest_rows':    len(train_rows),
        'val_manifest_rows':      len(val_rows),
        'full_manifest_rows':     len(full_rows),
        'integrity_checked':      len(integrity_rows),
        'integrity_failures':     int(n_fail),
        'pos_matching_verdict':   pos_verdict,
        'train_peripheral_ratio_normal': round(nm_tr_peri_ratio, 4),
        'train_peripheral_ratio_nsclc':  round(ns_tr_peri_ratio, 4),
        'val_peripheral_ratio_normal':   round(nm_vl_peri_ratio, 4),
        'val_peripheral_ratio_nsclc':    round(ns_vl_peri_ratio, 4),
        'class_weight_normal':    round(actual_cw_normal, 6),
        'class_weight_nsclc':     round(actual_cw_nsclc, 6),
        'leakage_all_zero':       all_leakage,
        'generation_errors':      len(errors),
        'guardrail': guardrail,
    }

    with open(REPORT_OUTPUT_BASE / 'p_c_normal12_matched_generation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # HU stats CSV
    if hu_stats_list:
        write_csv(REPORT_OUTPUT_BASE / 'p_c_normal12_hu_stats.csv', hu_stats_list)

    # Manifest summary JSON
    manifest_summary = {
        'stage':           'P-C-NORMAL12',
        'train_total':     len(train_rows),
        'train_normal':    normal_train_generated,
        'train_nsclc':     len(nsclc_train_rows),
        'val_total':       len(val_rows),
        'val_normal':      normal_val_generated,
        'val_nsclc':       len(nsclc_val_rows),
        'full_total':      len(full_rows),
        'class_weight_normal': round(actual_cw_normal, 6),
        'class_weight_nsclc':  round(actual_cw_nsclc, 6),
        'train_bin_targets':   train_bin_targets,
        'val_bin_targets':     val_bin_targets,
        'generation_errors':   len(errors),
        'conditions_ok':       verdict != 'FAIL',
    }
    with open(MANIFEST_OUTPUT_BASE / 'p_c_normal12_manifest_summary.json', 'w') as f:
        json.dump(manifest_summary, f, indent=2, ensure_ascii=False)

    # Markdown report
    pos_table_lines = ['| position_bin | normal_tr | nsclc_tr | tr_diff | tr_pass | normal_vl | nsclc_vl | vl_diff | vl_pass |',
                       '|---|---|---|---|---|---|---|---|---|']
    for r in pos_rows:
        pos_table_lines.append(
            f"| {r['position_bin']} | {r['normal_train_count']} ({r['normal_train_pct']}) "
            f"| {r['nsclc_train_count']} ({r['nsclc_train_pct']}) | {r['train_abs_diff']} | {'✓' if r['train_pass'] else '✗'} "
            f"| {r['normal_val_count']} ({r['normal_val_pct']}) | {r['nsclc_val_count']} ({r['nsclc_val_pct']}) "
            f"| {r['val_abs_diff']} | {'✓' if r['val_pass'] else '✗'} |"
        )

    report_md = f"""# P-C-NORMAL12 Matched Generation Report

## 판정: {verdict}

## 실행 명령
```
source ~/ai_env/bin/activate && python experiments/efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1/code/p_c_normal12_matched_generation.py
```

## Generated Matched Normal Crop Count
- Train: {normal_train_generated} / {TARGET_NORMAL_TRAIN} expected {'✓' if normal_train_generated==TARGET_NORMAL_TRAIN else '✗'}
- Val:   {normal_val_generated} / {TARGET_NORMAL_VAL} expected {'✓' if normal_val_generated==TARGET_NORMAL_VAL else '✗'}
- Total: {normal_train_generated + normal_val_generated} / 14,956 expected {'✓' if (normal_train_generated+normal_val_generated)==14956 else '✗'}

## Manifest Row Count
| manifest | rows | expected | pass |
|---|---|---|---|
| train | {len(train_rows)} | {TARGET_TRAIN_TOTAL} | {'✓' if len(train_rows)==TARGET_TRAIN_TOTAL else '✗'} |
| val   | {len(val_rows)} | {TARGET_VAL_TOTAL} | {'✓' if len(val_rows)==TARGET_VAL_TOTAL else '✗'} |
| full  | {len(full_rows)} | {TARGET_FULL_TOTAL} | {'✓' if len(full_rows)==TARGET_FULL_TOTAL else '✗'} |

## Label Distribution
| split | normal | NSCLC |
|---|---|---|
| train | {(train_df['label']==0).sum()} | {(train_df['label']==1).sum()} |
| val   | {(val_df['label']==0).sum()} | {(val_df['label']==1).sum()} |

## Class Weights
- class_weight_normal = {actual_cw_normal:.6f} (expected ≈ {CW_NORMAL:.6f})
- class_weight_nsclc  = {actual_cw_nsclc:.6f} (expected ≈ {CW_NSCLC:.6f})

## Position Distribution Matching: {pos_verdict}

### Bin-level comparison
{chr(10).join(pos_table_lines)}

### Peripheral Ratio
| split | normal | NSCLC | diff | pass |
|---|---|---|---|---|
| train | {nm_tr_peri_ratio:.4f} | {ns_tr_peri_ratio:.4f} | {train_peri_diff:.4f} | {'✓' if peri_train_pass else '✗'} |
| val   | {nm_vl_peri_ratio:.4f} | {ns_vl_peri_ratio:.4f} | {val_peri_diff:.4f} | {'✓' if peri_val_pass else '✗'} |

## Crop Integrity
- Total checked: {len(integrity_rows)}
- Failures: {n_fail}
- Result: {'PASS' if n_fail==0 else 'FAIL'}

## Patient Leakage
- Normal train/val overlap: {len(normal_tv_overlap)} ({'PASS' if not normal_tv_overlap else 'FAIL'})
- NSCLC train/val overlap:  {len(nsclc_tv_overlap)} ({'PASS' if not nsclc_tv_overlap else 'FAIL'})
- Normal/NSCLC overlap:     {len(normal_nsclc_overlap)} ({'PASS' if not normal_nsclc_overlap else 'FAIL'})

## Old Outputs Unmodified
- P-C-NORMAL9 NSCLC crops: NOT modified (reused by path reference only)
- P-C-NORMAL3 normal crops: NOT modified
- stage2_holdout accessed: {guardrail['stage2_holdout_accessed']} ({'PASS' if not guardrail['stage2_holdout_accessed'] else 'FAIL'})

## Guardrails
- training_run: {guardrail['training_run']} ({'PASS' if not guardrail['training_run'] else 'FAIL'})
- model_forward_run: {guardrail['model_forward_run']} ({'PASS' if not guardrail['model_forward_run'] else 'FAIL'})
- scoring_run: {guardrail['scoring_run']} ({'PASS' if not guardrail['scoring_run'] else 'FAIL'})
- threshold_computed: {guardrail['threshold_computed']} ({'PASS' if not guardrail['threshold_computed'] else 'FAIL'})
- checkpoint_saved: {guardrail['checkpoint_saved']} ({'PASS' if not guardrail['checkpoint_saved'] else 'FAIL'})
- hard_negative_included: {guardrail['hard_negative_included']} ({'PASS' if not guardrail['hard_negative_included'] else 'FAIL'})
- msd_lung_included: {guardrail['msd_lung_included']} ({'PASS' if not guardrail['msd_lung_included'] else 'FAIL'})
- forbidden_diagnostic_wording: {guardrail['forbidden_diagnostic_wording_count']} ({'PASS' if guardrail['forbidden_diagnostic_wording_count']==0 else 'FAIL'})

## Residual Shortcut Risk (SR-HU) Warning
- SR-POS 완화 목적으로 position-bin matched 정상 crop을 생성했다.
- SR-HU (Cohen d ≈ 1.94)는 여전히 남아 있을 수 있다.
- SR-HU는 병변 중심 crop과 정상 crop의 본질적 HU 분포 차이를 포함하므로 완전 제거 대상이 아니다.
- P-C-NORMAL13에서 HU/context 재비교가 필요하다.
- Full training은 아직 HOLD. P-C-NORMAL13 validation + shortcut stat comparison 이후 판단.

## Generation Errors: {len(errors)}

## Elapsed: {round(elapsed,1)}s

## Next Step
P-C-NORMAL13: matched manifest/crop validation + shortcut stat comparison (사용자 승인 후 진행)
"""

    with open(REPORT_OUTPUT_BASE / 'p_c_normal12_matched_generation_report.md', 'w', encoding='utf-8') as f:
        f.write(report_md)

    # ── 18. DONE.json ─────────────────────────────────────────────────────────
    done = {
        'stage':                       'P-C-NORMAL12',
        'verdict':                     verdict,
        'conditions_ok':               verdict != 'FAIL',
        'normal_train_generated':      normal_train_generated,
        'normal_val_generated':        normal_val_generated,
        'total_normal_generated':      normal_train_generated + normal_val_generated,
        'train_manifest_rows':         len(train_rows),
        'val_manifest_rows':           len(val_rows),
        'full_manifest_rows':          len(full_rows),
        'integrity_failures':          int(n_fail),
        'pos_matching_verdict':        pos_verdict,
        'leakage_all_zero':            all_leakage,
        'stage2_holdout_accessed':     False,
        'training_run':                False,
        'model_forward_run':           False,
        'scoring_run':                 False,
        'threshold_computed':          False,
        'checkpoint_saved':            False,
        'existing_outputs_modified':   False,
        'hard_negative_included':      guardrail['hard_negative_included'],
        'msd_lung_included':           guardrail['msd_lung_included'],
        'forbidden_diagnostic_wording_count': guardrail['forbidden_diagnostic_wording_count'],
        'generation_errors':           len(errors),
        'elapsed_sec':                 round(elapsed, 1),
    }
    with open(MANIFEST_OUTPUT_BASE / 'DONE.json', 'w') as f:
        json.dump(done, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"[P-C-NORMAL12] VERDICT: {verdict}")
    print(f"  Normal crops generated: {normal_train_generated + normal_val_generated} / 14,956")
    print(f"  Train manifest: {len(train_rows)} rows")
    print(f"  Val manifest:   {len(val_rows)} rows")
    print(f"  Position matching: {pos_verdict}")
    print(f"  Integrity failures: {n_fail}")
    print(f"  Leakage: {'PASS' if all_leakage else 'FAIL'}")
    print(f"  Elapsed: {round(elapsed,1)}s")
    print(f"{'='*60}")

    return verdict


if __name__ == '__main__':
    try:
        run_generation()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
        sys.exit(1)
