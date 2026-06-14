"""
P-C-NORMAL22c: NSCLC Holdout Crop Generator (clean stage2_holdout)
Branch: efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1

목적:
  clean stage2_holdout NSCLC LUNG1-only 122명 (leakage 2명 hard exclude, MSD_lung 30명 source 제외)의
  sampling_label==positive rows로 final_test용 (3,96,96) int16 crops 생성.
  label=1, split=final_test, source_split=stage2_holdout

  stage2_holdout unlock 승인 목적: final independent test baseline만.
  개발/튜닝/threshold 최적화 사용 금지.

P-C-NORMAL22e0c 패치: LUNG1-only source filter 적용.
  MSD_lung 30명 / 6590 rows 제외 (normal-vs-NSCLC binary label 체계 밖).
  corrected target: LUNG1-only 122명 / 44723 positive rows / combined 66323.

모드:
  --dry-check        : input/schema/feasibility 확인만 (actual 저장 없음)
  --run-generate --confirm-final-test --confirm-no-prediction --confirm-no-threshold
                     : actual crop + manifest 생성 (P-C-NORMAL22d/22e 별도 승인 필요)

안전 규칙:
  ALLOW_REAL_GENERATION=False → --run-generate exit 2
  LUNG1-295 / LUNG1-415 hard exclusion (leakage) → output plan 포함 시 exit 2
  model forward / training / threshold 계산 금지
  기존 6ch crop 재사용 금지
  P-C8/P-C-AUX/P-B/P-C/N-C/RD 결과 수정 금지
  진단명/cancer_probability/adenocarcinoma_probability 표현 금지
"""

import argparse
import csv
import json
import os
import py_compile
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Safety flags
# ─────────────────────────────────────────────────────────────────────────────
ALLOW_REAL_GENERATION = False  # P-C-NORMAL22e2 완료 — 원복

# P-C-NORMAL22e3 완료 — ALLOW_REAL_COMBINE 원복
ALLOW_REAL_COMBINE = False

LEAKAGE_PATIENTS = frozenset(['LUNG1-295', 'LUNG1-415'])
STAGE2_HOLDOUT_UNLOCK_PURPOSE = 'final_independent_test_baseline_only'
TUNING_AFTER_HOLDOUT_FORBIDDEN = True

GENERATOR_VERSION = "p_c_normal22c_nsclc_holdout_generator_v1"
CROP_SIZE = 96
HALF_CROP = 48
LABEL_NSCLC = 1

EXPECTED_HOLDOUT_SOURCE_PATIENTS = 154  # s6a manifest 전체 환자 수
EXPECTED_LEAKAGE_EXCLUDED = 2
EXPECTED_MSD_EXCLUDED_PATIENTS = 30    # binary label 체계 밖 — 제외 필수 (P-C-NORMAL22e0c)
EXPECTED_MSD_EXCLUDED_ROWS = 6590      # MSD_lung positive rows 합계 — 제외 필수
EXPECTED_CLEAN_PATIENTS = 122          # LUNG1-only (leakage 2명 + MSD_lung 30명 제외)
EXPECTED_NSCLC_POSITIVE_ROWS = 44723   # LUNG1-only positive rows (corrected from 51313)
EXPECTED_NORMAL_ROWS = 21600           # normal test rows
EXPECTED_COMBINED_ROWS = 66323         # 21600 + 44723

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path('/home/jinhy/project/lung-ct-anomaly')
BRANCH_ROOT = (
    PROJECT_ROOT
    / 'experiments'
    / 'efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1'
)

NSCLC_CT_BASE = Path(
    '/mnt/c/Users/jinhy/Desktop/'
    'NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/'
    'volumes_npy'
)

S6A_HOLDOUT_MANIFEST = (
    PROJECT_ROOT
    / 'outputs'
    / 'second-stage-lesion-refiner-v1'
    / 'datasets'
    / 's6a_stage2_holdout_candidate_coordinate_manifest_v1.csv'
)

CROP_OUTPUT_DIR = (
    PROJECT_ROOT
    / 'outputs'
    / 'test_crops'
    / 'p_c_normal22_final_baseline_test_crops'
    / 'stage2_holdout_nsclc'
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

# NSCLC CT / ROI / lesion filename pattern
CT_FILENAME      = 'ct_hu.npy'
ROI_FILENAME     = 'roi_0_0.npy'
LESION_FILENAME  = 'lesion_mask_roi_0_0.npy'

GUARDRAIL = {
    'stage2_holdout_unlocked_for_final_test': True,
    'tuning_after_holdout_forbidden': True,
    'leakage_hard_excluded': True,
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
    'reused_6ch_crops': False,
    'forbidden_diagnostic_wording_count': 0,
}

START_TIME = datetime.now()


# ─────────────────────────────────────────────────────────────────────────────
# Leakage guard
# ─────────────────────────────────────────────────────────────────────────────

def check_leakage_guard(patient_ids):
    """LUNG1-295 / LUNG1-415가 output plan에 포함되면 exit 2."""
    leaked = [p for p in patient_ids if p in LEAKAGE_PATIENTS]
    if leaked:
        print(f"[ABORT] Leakage patients found in output plan: {leaked}")
        print("  LUNG1-295 / LUNG1-415는 반드시 제외해야 합니다.")
        sys.exit(2)


# ─────────────────────────────────────────────────────────────────────────────
# Crop extraction (P-C-NORMAL9 same-generator 방식)
# ─────────────────────────────────────────────────────────────────────────────

def extract_crop(ct_vol: np.ndarray, local_z: float, center_y: float, center_x: float) -> np.ndarray:
    """
    (3, 96, 96) int16 crop.
    z channels: z-1 / z / z+1 with edge clamping.
    xy: center±48 with zero-padding if OOB.
    P-C-NORMAL9 same-generator 방식 (기존 6ch crop 재사용 금지).
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
    print(f"[22c-NSCLC DRY-CHECK] Start: {START_TIME.isoformat()}")
    DRYCHECK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    errors = []
    results = {}

    # ── 1. Script compile check ───────────────────────────────────────────────
    print("[1/9] Script compile check...")
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

    # ── 2. s6a manifest load & schema check ──────────────────────────────────
    print("[2/9] s6a holdout manifest load...")
    manifest_ok = False
    df_raw = None
    if not S6A_HOLDOUT_MANIFEST.exists():
        errors.append({'check': 'manifest_exists', 'item': str(S6A_HOLDOUT_MANIFEST), 'error': 'not found'})
        print(f"  FAIL: manifest not found: {S6A_HOLDOUT_MANIFEST}")
    else:
        df_raw = pd.read_csv(S6A_HOLDOUT_MANIFEST, low_memory=False)
        required_cols = [
            'patient_id', 'safe_id', 'local_z', 'y0', 'x0', 'y1', 'x1',
            'sampling_label', 'stage_split', 'slice_index', 'position_bin', 'z_level',
        ]
        missing = [c for c in required_cols if c not in df_raw.columns]
        if missing:
            errors.append({'check': 'manifest_schema', 'item': str(S6A_HOLDOUT_MANIFEST), 'error': f'missing cols: {missing}'})
            print(f"  FAIL: missing columns: {missing}")
        else:
            manifest_ok = True
            print(f"  PASS: {len(df_raw)} rows, columns ok")
            # Confirm center_y/x NOT in raw manifest (need derivation)
            has_center_y = 'center_y' in df_raw.columns
            has_center_x = 'center_x' in df_raw.columns
            print(f"  center_y in manifest: {has_center_y} (expected False or needs verification)")
            print(f"  center_x in manifest: {has_center_x} (expected False or needs verification)")

    results['manifest_check'] = {
        'path': str(S6A_HOLDOUT_MANIFEST),
        'exists': S6A_HOLDOUT_MANIFEST.exists(),
        'rows': len(df_raw) if df_raw is not None else 0,
        'status': 'PASS' if manifest_ok else 'FAIL',
    }

    # ── 3. center_y / center_x derivation & bbox check ────────────────────────
    print("[3/9] center_y/x derivation & bbox size check...")
    center_ok = False
    bbox_ok = False
    df = None
    if df_raw is not None:
        # Derive center_y / center_x
        df_raw = df_raw.copy()
        df_raw['center_y'] = (df_raw['y0'] + df_raw['y1']) / 2.0
        df_raw['center_x'] = (df_raw['x0'] + df_raw['x1']) / 2.0
        center_ok = True
        print(f"  center_y derived: (y0+y1)/2  ✓")
        print(f"  center_x derived: (x0+x1)/2  ✓")

        # bbox size check: expect 32x32
        bbox_h = (df_raw['y1'] - df_raw['y0']).unique()
        bbox_w = (df_raw['x1'] - df_raw['x0']).unique()
        bbox_ok = set(bbox_h) == {32} and set(bbox_w) == {32}
        print(f"  bbox height values: {sorted(set(bbox_h))} (expected: {{32}})")
        print(f"  bbox width values:  {sorted(set(bbox_w))} (expected: {{32}})")
        if not bbox_ok:
            errors.append({'check': 'bbox_size', 'item': 's6a', 'error': f'h={set(bbox_h)} w={set(bbox_w)}'})
        df = df_raw

    results['center_derivation'] = {
        'policy': 'center_y=(y0+y1)/2, center_x=(x0+x1)/2',
        'center_ok': center_ok,
        'bbox_ok': bbox_ok,
        'expected_bbox': '32x32',
        'status': 'PASS' if (center_ok and bbox_ok) else ('PARTIAL' if center_ok else 'FAIL'),
    }

    # ── 4. Leakage exclusion check ────────────────────────────────────────────
    print("[4/9] Leakage exclusion check...")
    leakage_ok = False
    leakage_info = {}
    df_clean = None
    if df is not None:
        all_patients = set(df['patient_id'].unique())
        leakage_found = all_patients & LEAKAGE_PATIENTS
        leakage_info = {
            'total_patients_in_manifest': len(all_patients),
            'leakage_patients_expected': sorted(LEAKAGE_PATIENTS),
            'leakage_patients_found_in_manifest': sorted(leakage_found),
        }

        # Hard exclude leakage
        df_after_leakage = df[~df['patient_id'].isin(LEAKAGE_PATIENTS)].copy()
        leakage_info['excluded_rows'] = len(df) - len(df_after_leakage)

        # Hard guard: ensure leakage not in clean set
        check_leakage_guard(df_after_leakage['patient_id'].unique())

        # ── LUNG1-only source filter (P-C-NORMAL22e0c patch) ─────────────────
        msd_patients_set = set(p for p in df_after_leakage['patient_id'].unique()
                               if not str(p).startswith('LUNG1'))
        msd_lung_patients = len(msd_patients_set)
        msd_lung_rows = int((~df_after_leakage['patient_id'].str.startswith('LUNG1')).sum())
        df_clean = df_after_leakage[df_after_leakage['patient_id'].str.startswith('LUNG1')].copy()
        clean_patients = df_clean['patient_id'].nunique()

        leakage_info['clean_patients_after_exclusion'] = clean_patients
        leakage_info['msd_lung_patients_excluded'] = msd_lung_patients
        leakage_info['msd_lung_rows_excluded'] = msd_lung_rows

        leakage_ok = (
            len(leakage_found) == EXPECTED_LEAKAGE_EXCLUDED
            and msd_lung_patients == EXPECTED_MSD_EXCLUDED_PATIENTS
            and clean_patients == EXPECTED_CLEAN_PATIENTS
        )

        print(f"  Total patients in manifest: {len(all_patients)}")
        print(f"  Leakage found: {sorted(leakage_found)}")
        print(f"  MSD_lung patients excluded: {msd_lung_patients} (expected {EXPECTED_MSD_EXCLUDED_PATIENTS})")
        print(f"  Clean LUNG1-only patients:  {clean_patients} (expected {EXPECTED_CLEAN_PATIENTS})")

        if not leakage_ok:
            errors.append({'check': 'leakage_exclusion', 'item': 's6a',
                           'error': f'clean_patients={clean_patients}, msd_excluded={msd_lung_patients}'})

    results['leakage_exclusion'] = {**leakage_info, 'status': 'PASS' if leakage_ok else 'FAIL'}

    # ── 5. Positive rows, patient count, MSD absent check ────────────────────
    print("[5/9] Positive rows & patient count & MSD absent check...")
    count_ok = False
    msd_absent_ok = False
    positive_dist = {}
    if df_clean is not None:
        df_positive = df_clean[df_clean['sampling_label'] == 'positive'].copy()
        actual_patients = df_positive['patient_id'].nunique()
        actual_rows = len(df_positive)
        positive_dist = df_positive['position_bin'].value_counts().to_dict()

        # MSD absent guard — LUNG1-only 필터 후에도 잔존 확인
        msd_in_positive = [p for p in df_positive['patient_id'].unique()
                           if not str(p).startswith('LUNG1')]
        msd_absent_ok = len(msd_in_positive) == 0
        if not msd_absent_ok:
            errors.append({'check': 'msd_absent', 'item': 's6a',
                           'error': f'msd_in_positive={msd_in_positive[:5]}'})

        count_ok = (
            msd_absent_ok
            and actual_patients == EXPECTED_CLEAN_PATIENTS
            and actual_rows == EXPECTED_NSCLC_POSITIVE_ROWS
        )

        print(f"  Clean LUNG1-only patients: {actual_patients} (expected {EXPECTED_CLEAN_PATIENTS})")
        print(f"  Positive rows:             {actual_rows} (expected {EXPECTED_NSCLC_POSITIVE_ROWS})")
        print(f"  MSD_lung absent:           {msd_absent_ok}")
        print(f"  Position bin dist: {positive_dist}")

        if not count_ok:
            errors.append({'check': 'positive_count', 'item': 's6a',
                           'error': f'patients={actual_patients}, rows={actual_rows}, msd_absent={msd_absent_ok}'})
    else:
        df_positive = None
        actual_patients = 0
        actual_rows = 0
        msd_absent_ok = False

    results['count_check'] = {
        'clean_patients': actual_patients,
        'expected_patients': EXPECTED_CLEAN_PATIENTS,
        'positive_rows': actual_rows,
        'expected_positive_rows': EXPECTED_NSCLC_POSITIVE_ROWS,
        'msd_absent_ok': msd_absent_ok,
        'position_bin_distribution': positive_dist,
        'status': 'PASS' if count_ok else 'FAIL',
    }

    # ── 6. CT / ROI / lesion file existence check ─────────────────────────────
    print("[6/9] CT / ROI / lesion file existence check...")
    files_ok = False
    ct_missing = []
    roi_missing = []
    lesion_missing = []
    if df_positive is not None:
        safe_ids = df_positive['safe_id'].unique()
        for sid in safe_ids:
            base = NSCLC_CT_BASE / sid
            if not (base / CT_FILENAME).exists():
                ct_missing.append(sid)
            if not (base / ROI_FILENAME).exists():
                roi_missing.append(sid)
            if not (base / LESION_FILENAME).exists():
                lesion_missing.append(sid)

        files_ok = not ct_missing and not roi_missing and not lesion_missing
        print(f"  CT:     {len(safe_ids) - len(ct_missing)}/{len(safe_ids)}")
        print(f"  ROI:    {len(safe_ids) - len(roi_missing)}/{len(safe_ids)}")
        print(f"  lesion: {len(safe_ids) - len(lesion_missing)}/{len(safe_ids)}")

        for s in ct_missing[:3]:
            errors.append({'check': 'ct_exists', 'item': s, 'error': 'ct_hu.npy not found'})
        for s in roi_missing[:3]:
            errors.append({'check': 'roi_exists', 'item': s, 'error': 'roi_0_0.npy not found'})
        for s in lesion_missing[:3]:
            errors.append({'check': 'lesion_exists', 'item': s, 'error': 'lesion_mask_roi_0_0.npy not found'})
    else:
        safe_ids = []

    results['file_existence'] = {
        'safe_ids_checked': len(safe_ids),
        'ct_ok': len(safe_ids) - len(ct_missing),
        'roi_ok': len(safe_ids) - len(roi_missing),
        'lesion_ok': len(safe_ids) - len(lesion_missing),
        'ct_missing': ct_missing[:5],
        'roi_missing': roi_missing[:5],
        'lesion_missing': lesion_missing[:5],
        'status': 'PASS' if files_ok else 'FAIL',
    }

    # ── 7. z / xy boundary check ──────────────────────────────────────────────
    print("[7/9] z & xy boundary check (sample)...")
    boundary_ok = False
    boundary_issues = []
    if df_positive is not None and files_ok:
        sample_rows = df_positive.sample(min(20, len(df_positive)), random_state=42)
        for _, row in sample_rows.iterrows():
            sid = row['safe_id']
            ct_path = NSCLC_CT_BASE / sid / CT_FILENAME
            try:
                ct_vol = np.load(str(ct_path), mmap_mode='r')
                z_dim, y_dim, x_dim = ct_vol.shape
                lz = int(round(row['local_z']))
                cy = row['center_y']
                cx = row['center_x']

                z_ok = 0 <= lz < z_dim
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

    # ── 8. Crop feasibility check (5~10 crops, in-memory only) ────────────────
    print("[8/9] Crop feasibility check (in-memory, no save)...")
    feasibility_ok = False
    feasibility_results = []
    GUARDRAIL['crop_generated'] = False

    if df_positive is not None and files_ok:
        sample_crops = df_positive.sample(min(10, len(df_positive)), random_state=77)
        loaded_vols = {}

        for i, (_, row) in enumerate(sample_crops.iterrows()):
            sid = row['safe_id']
            ct_path = NSCLC_CT_BASE / sid / CT_FILENAME
            try:
                if sid not in loaded_vols:
                    loaded_vols[sid] = np.load(str(ct_path))

                ct_vol = loaded_vols[sid]
                lz     = row['local_z']
                cy     = row['center_y']
                cx     = row['center_x']

                crop = extract_crop(ct_vol, lz, cy, cx)

                shape_ok   = crop.shape == (3, CROP_SIZE, CROP_SIZE)
                dtype_ok   = crop.dtype == np.int16
                nan_ok     = not np.any(np.isnan(crop.astype(np.float32)))
                inf_ok     = not np.any(np.isinf(crop.astype(np.float32)))
                nonzero_ok = np.any(crop != 0)

                # Confirm this is a positive (lesion-positive) crop
                is_positive = row['sampling_label'] == 'positive'

                feasibility_results.append({
                    'row_id': row.get('row_id', i),
                    'patient_id': row['patient_id'],
                    'sampling_label': row['sampling_label'],
                    'shape': str(crop.shape),
                    'dtype': str(crop.dtype),
                    'shape_ok': shape_ok,
                    'dtype_ok': dtype_ok,
                    'nan_ok': nan_ok,
                    'inf_ok': inf_ok,
                    'nonzero_ok': nonzero_ok,
                    'is_positive_crop': is_positive,
                    'all_pass': all([shape_ok, dtype_ok, nan_ok, inf_ok, nonzero_ok, is_positive]),
                })

            except Exception as e:
                feasibility_results.append({
                    'row_id': row.get('row_id', i),
                    'patient_id': row.get('patient_id', '?'),
                    'error': str(e),
                    'all_pass': False,
                })
                errors.append({'check': 'crop_feasibility', 'item': str(row.get('patient_id', '?')), 'error': str(e)})

        feasibility_ok = all(r.get('all_pass', False) for r in feasibility_results)
        pass_count = sum(1 for r in feasibility_results if r.get('all_pass', False))
        print(f"  feasibility samples: {len(feasibility_results)}, pass: {pass_count}/{len(feasibility_results)}")
        print("  (no npz saved — in-memory only)")

    results['crop_feasibility'] = {
        'samples_checked': len(feasibility_results),
        'pass_count': sum(1 for r in feasibility_results if r.get('all_pass', False)),
        'all_positive_only': all(r.get('is_positive_crop', False) for r in feasibility_results if 'is_positive_crop' in r),
        'results': feasibility_results,
        'status': 'PASS' if feasibility_ok else 'FAIL',
        'note': 'in-memory only, no npz saved, lesion-positive only',
    }

    # ── 9. Output collision check ─────────────────────────────────────────────
    print("[9/9] Output collision check...")
    crop_dir_exists     = CROP_OUTPUT_DIR.exists()
    manifest_dir_exists = MANIFEST_OUTPUT_DIR.exists()
    collision_found     = crop_dir_exists or manifest_dir_exists

    results['output_collision'] = {
        'crop_dir': str(CROP_OUTPUT_DIR),
        'crop_dir_exists': crop_dir_exists,
        'manifest_dir': str(MANIFEST_OUTPUT_DIR),
        'manifest_dir_exists': manifest_dir_exists,
        'collision_found': collision_found,
        'status': 'CLEAN' if not collision_found else 'COLLISION',
    }
    if collision_found:
        print("  WARNING: output dir already exists (collision possible)")
    else:
        print("  CLEAN: no collision")

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
        'label': LABEL_NSCLC,
        'label_name': 'nsclc',
        'split': 'final_test',
        'source_split': 'stage2_holdout',
        'source_name': 's6a_stage2_holdout_lung1_nsclc',
        'stage2_holdout_flag': True,
        'hard_negative_flag': False,
        'msd_lung_flag': False,
        'leakage_excluded_flag': False,
        'generator_version': GENERATOR_VERSION,
        'crop_shape': '(3,96,96)',
        'crop_dtype': 'int16',
        'center_y_source': '(y0+y1)/2 derived from s6a bbox',
        'center_x_source': '(x0+x1)/2 derived from s6a bbox',
        'optional_metadata_note': '모델 입력 아님 — 해석/비교용 metadata',
    }
    results['manifest_schema_plan'] = manifest_schema

    # ── Guardrail final record ─────────────────────────────────────────────────
    assert not GUARDRAIL['crop_generated'],            "crop_generated should remain False"
    assert not GUARDRAIL['manifest_generated'],        "manifest_generated should remain False"
    assert not GUARDRAIL['model_forward_run'],         "model_forward_run should remain False"
    assert not GUARDRAIL['training_run'],              "training_run should remain False"
    assert not GUARDRAIL['reused_6ch_crops'],          "reused_6ch_crops should remain False"
    assert GUARDRAIL['leakage_hard_excluded'],         "leakage_hard_excluded should remain True"
    results['guardrail'] = dict(GUARDRAIL)

    # ── Overall verdict ──────────────────────────────────────────────────────
    all_checks = [
        compile_ok,
        manifest_ok,
        center_ok and bbox_ok,
        leakage_ok,
        count_ok,
        files_ok,
        boundary_ok,
        feasibility_ok,
        not collision_found,
    ]
    passed = sum(all_checks)
    total  = len(all_checks)

    if passed == total:
        decision = 'READY_FOR_FINAL_TEST_CROP_GENERATION'
    elif compile_ok and manifest_ok and leakage_ok:
        decision = 'NEEDS_NSCLC_SCRIPT_FIX'
    elif compile_ok and not leakage_ok:
        decision = 'NEEDS_LEAKAGE_GUARD_FIX'
    else:
        decision = 'FAIL'

    results['decision'] = decision
    results['checks_passed'] = f"{passed}/{total}"
    results['error_count'] = len(errors)

    print(f"\n[DECISION] {decision} ({passed}/{total} checks passed)")

    _write_reports(results, errors, feasibility_results)
    print(f"[22c-NSCLC DRY-CHECK] Done.")
    return decision


def _write_reports(results, errors, feasibility_results):
    out = DRYCHECK_OUTPUT_DIR

    # NSCLC holdout drycheck CSV
    with open(out / 'p_c_normal22c_nsclc_holdout_drycheck.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['check', 'expected', 'actual', 'status'])
        lc = results['leakage_exclusion']
        w.writerow(['source_patients', EXPECTED_HOLDOUT_SOURCE_PATIENTS, lc.get('total_patients_in_manifest', '?'), lc['status']])
        w.writerow(['excluded_leakage', EXPECTED_LEAKAGE_EXCLUDED, lc.get('excluded_rows', '?'), lc['status']])
        w.writerow(['clean_patients', EXPECTED_CLEAN_PATIENTS, lc.get('clean_patients_after_exclusion', '?'), lc['status']])
        cc = results['count_check']
        w.writerow(['positive_rows', EXPECTED_NSCLC_POSITIVE_ROWS, cc['positive_rows'], cc['status']])
        cd = results['center_derivation']
        w.writerow(['center_y_derived', True, cd['center_ok'], 'PASS' if cd['center_ok'] else 'FAIL'])
        w.writerow(['bbox_32x32', True, cd['bbox_ok'], 'PASS' if cd['bbox_ok'] else 'FAIL'])
        fe = results['file_existence']
        w.writerow(['ct_available', EXPECTED_CLEAN_PATIENTS, fe['ct_ok'], fe['status']])
        w.writerow(['roi_available', EXPECTED_CLEAN_PATIENTS, fe['roi_ok'], fe['status']])
        w.writerow(['lesion_available', EXPECTED_CLEAN_PATIENTS, fe['lesion_ok'], fe['status']])
        bc = results['boundary_check']
        w.writerow(['boundary_issues', 0, bc['issues_found'], bc['status']])
        fc = results['crop_feasibility']
        w.writerow(['feasibility_pass', fc['samples_checked'], fc['pass_count'], fc['status']])
        w.writerow(['all_positive_only', True, fc['all_positive_only'], 'PASS' if fc['all_positive_only'] else 'FAIL'])

    # Leakage exclusion check CSV
    with open(out / 'p_c_normal22c_leakage_exclusion_check.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['patient_id', 'in_manifest', 'excluded', 'status'])
        lc = results['leakage_exclusion']
        for p in sorted(LEAKAGE_PATIENTS):
            found = p in lc.get('leakage_patients_found_in_manifest', [])
            w.writerow([p, found, True, 'PASS' if found else 'WARN_NOT_IN_MANIFEST'])

    # errors CSV (append mode — normal script may have written already)
    mode = 'a' if (out / 'p_c_normal22c_errors.csv').exists() else 'w'
    with open(out / 'p_c_normal22c_errors.csv', mode, newline='') as f:
        w = csv.DictWriter(f, fieldnames=['check', 'item', 'error'])
        if mode == 'w':
            w.writeheader()
        w.writerows(errors)

    # output collision CSV (append)
    oc = results['output_collision']
    collision_file = out / 'p_c_normal22c_output_collision_check.csv'
    mode = 'a' if collision_file.exists() else 'w'
    with open(collision_file, mode, newline='') as f:
        w = csv.writer(f)
        if mode == 'w':
            w.writerow(['path', 'exists', 'collision'])
        w.writerow([oc['crop_dir'], oc['crop_dir_exists'], oc['collision_found']])

    # manifest schema CSV (write if not exists)
    schema_file = out / 'p_c_normal22c_manifest_schema_check.csv'
    ms = results['manifest_schema_plan']
    if not schema_file.exists():
        with open(schema_file, 'w', newline='') as f:
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
            'existing_outputs_modified', 'reused_6ch_crops',
        ]
        for item in expected_false:
            w.writerow([item, False, gr.get(item, False), not gr.get(item, False)])
        expected_true = ['leakage_hard_excluded', 'stage2_holdout_unlocked_for_final_test']
        for item in expected_true:
            w.writerow([item, True, gr.get(item, False), gr.get(item, False)])
        w.writerow(['forbidden_diagnostic_wording_count', 0, gr.get('forbidden_diagnostic_wording_count', 0), gr.get('forbidden_diagnostic_wording_count', 0) == 0])

    # Summary JSON (NSCLC script only)
    summary = {
        'script': 'p_c_normal22c_nsclc_holdout_crop_generator.py',
        'date': START_TIME.isoformat(),
        'decision': results['decision'],
        'checks_passed': results['checks_passed'],
        'error_count': results['error_count'],
        'leakage_excluded': sorted(LEAKAGE_PATIENTS),
        'clean_nsclc_patients': results['leakage_exclusion'].get('clean_patients_after_exclusion', 0),
        'positive_rows': results['count_check'].get('positive_rows', 0),
        'guardrail': results['guardrail'],
    }
    with open(out / 'p_c_normal22c_nsclc_script_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[Reports] Written to: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Post-run hard assertion (P-C-NORMAL22d2 hardening)
# ─────────────────────────────────────────────────────────────────────────────

def _post_run_hard_assertion_nsclc(saved, error_rows, manifest_rows, integrity_log):
    """
    NSCLC generation loop 종료 후 manifest fragment 저장 전 호출.
    하나라도 실패하면 manifest 저장 없이 exit 2.
    all_zero는 반드시 failure. leakage patient 하나라도 있으면 즉시 abort.
    """
    failures = []

    # 1. saved == 51,313
    if saved != EXPECTED_NSCLC_POSITIVE_ROWS:
        failures.append(f"saved={saved} != expected={EXPECTED_NSCLC_POSITIVE_ROWS}")

    # 2. len(error_rows) == 0
    if len(error_rows) != 0:
        failures.append(f"error_rows={len(error_rows)} != 0")

    # 3. len(manifest_rows) == 51,313
    if len(manifest_rows) != EXPECTED_NSCLC_POSITIVE_ROWS:
        failures.append(f"manifest_rows={len(manifest_rows)} != {EXPECTED_NSCLC_POSITIVE_ROWS}")

    # 4. integrity_log rows == 51,313
    if len(integrity_log) != EXPECTED_NSCLC_POSITIVE_ROWS:
        failures.append(f"integrity_log={len(integrity_log)} != {EXPECTED_NSCLC_POSITIVE_ROWS}")

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

    # 9. all all_zero=False (hard failure)
    all_zero_rows = [r for r in integrity_log if r.get('all_zero', False)]
    if all_zero_rows:
        failures.append(f"all_zero=True count={len(all_zero_rows)} (hard failure)")

    # 10. patient count == 152
    actual_patients = len(set(r['patient_id'] for r in manifest_rows))
    if actual_patients != EXPECTED_CLEAN_PATIENTS:
        failures.append(f"patient_count={actual_patients} != {EXPECTED_CLEAN_PATIENTS}")

    # 11. LUNG1-295 absent (leakage hard check — 즉시 abort)
    lung1_295_rows = [r for r in manifest_rows if r.get('patient_id') == 'LUNG1-295']
    if lung1_295_rows:
        print("[ABORT] LEAKAGE: LUNG1-295 found in manifest_rows — immediate abort")
        sys.exit(2)

    # 12. LUNG1-415 absent (leakage hard check — 즉시 abort)
    lung1_415_rows = [r for r in manifest_rows if r.get('patient_id') == 'LUNG1-415']
    if lung1_415_rows:
        print("[ABORT] LEAKAGE: LUNG1-415 found in manifest_rows — immediate abort")
        sys.exit(2)

    # 12b. MSD_lung absent guard (source guard — exit 2)
    msd_in_manifest = [r for r in manifest_rows
                       if not str(r.get('patient_id', '')).startswith('LUNG1')]
    if msd_in_manifest:
        print(f"[ABORT] MSD_lung found in manifest_rows: "
              f"{[r['patient_id'] for r in msd_in_manifest[:5]]}")
        print("  MSD_lung은 NSCLC binary label 체계 밖 — manifest 저장 중단.")
        sys.exit(2)

    # 13. label only 1
    bad_label = [r for r in manifest_rows if r.get('label') != LABEL_NSCLC]
    if bad_label:
        failures.append(f"label!=1 count={len(bad_label)}")

    # 14. stage2_holdout_flag all True
    bad_holdout = [r for r in manifest_rows if r.get('stage2_holdout_flag') is not True]
    if bad_holdout:
        failures.append(f"stage2_holdout_flag!=True count={len(bad_holdout)}")

    # 15. hard_negative_flag all False
    bad_hn = [r for r in manifest_rows if r.get('hard_negative_flag') is not False]
    if bad_hn:
        failures.append(f"hard_negative_flag!=False count={len(bad_hn)}")

    # 16. msd_lung_flag all False
    bad_msd = [r for r in manifest_rows if r.get('msd_lung_flag') is not False]
    if bad_msd:
        failures.append(f"msd_lung_flag!=False count={len(bad_msd)}")

    # 17. leakage_excluded_flag all False
    bad_leakage_flag = [r for r in manifest_rows if r.get('leakage_excluded_flag') is not False]
    if bad_leakage_flag:
        failures.append(f"leakage_excluded_flag!=False count={len(bad_leakage_flag)}")

    # 18. crop_path unique == 51,313
    crop_paths = [r['crop_path'] for r in manifest_rows]
    if len(set(crop_paths)) != EXPECTED_NSCLC_POSITIVE_ROWS:
        failures.append(f"crop_path_unique={len(set(crop_paths))} != {EXPECTED_NSCLC_POSITIVE_ROWS}")

    # 19. final_candidate_id unique == 51,313
    candidate_ids = [r['final_candidate_id'] for r in manifest_rows]
    if len(set(candidate_ids)) != EXPECTED_NSCLC_POSITIVE_ROWS:
        failures.append(f"final_candidate_id_unique={len(set(candidate_ids))} != {EXPECTED_NSCLC_POSITIVE_ROWS}")

    if failures:
        print("[POST-RUN ASSERTION FAIL] NSCLC generator post-run hard check FAILED:")
        for fail in failures:
            print(f"  - {fail}")
        print("[ABORT] manifest fragment 저장 중단 (failed 상태). DONE/combine으로 넘어가지 않음.")
        sys.exit(2)

    print(f"[POST-RUN ASSERTION PASS] All {len(manifest_rows)} NSCLC crops passed all hard assertions.")


# ─────────────────────────────────────────────────────────────────────────────
# Actual generation (guarded)
# ─────────────────────────────────────────────────────────────────────────────

def run_generate(confirm_final_test: bool, confirm_no_prediction: bool, confirm_no_threshold: bool):
    """
    Actual crop generation — clean stage2_holdout LUNG1-only 122 patients → 44,723 (3,96,96) int16 crops.
    LUNG1-295 / LUNG1-415 hard exclude. MSD_lung 30명 source filter 제외 (binary label 체계 밖).
    ALLOW_REAL_GENERATION=True + 세 가지 confirm 플래그 모두 필요.
    P-C-NORMAL22e2 별도 승인 전까지 ALLOW_REAL_GENERATION=False 유지.
    기존 6ch crop 재사용 금지.
    """
    if not ALLOW_REAL_GENERATION:
        print("[ABORT] ALLOW_REAL_GENERATION=False. P-C-NORMAL22e 별도 승인 후 True로 변경 필요.")
        sys.exit(2)

    if not (confirm_final_test and confirm_no_prediction and confirm_no_threshold):
        print("[ABORT] --confirm-final-test --confirm-no-prediction --confirm-no-threshold 세 가지 모두 필요.")
        sys.exit(2)

    run_start = datetime.now()
    print(f"[22d-NSCLC GENERATE] Start: {run_start.isoformat()}")

    # ── Output collision hard blocker ─────────────────────────────────────────
    COMBINED_MANIFEST_CSV = MANIFEST_OUTPUT_DIR / 'p_c_normal22_final_test_manifest.csv'
    NSCLC_FRAGMENT_CSV    = MANIFEST_OUTPUT_DIR / 'p_c_normal22_final_test_nsclc_manifest.csv'
    for col_label, col_path in [
        ('crop_dir_stage2_holdout_nsclc', CROP_OUTPUT_DIR),
        ('nsclc_manifest_fragment',       NSCLC_FRAGMENT_CSV),
        ('combined_manifest',             COMBINED_MANIFEST_CSV),
    ]:
        if col_path.exists():
            print(f"[ABORT] Output collision — {col_label}: {col_path}")
            print("  기존 출력 보호. 수동 확인 후 제거해야 재실행 가능.")
            sys.exit(2)
    print("[COLLISION-OK] No collision detected. Proceeding.")

    # ── Manifest load ─────────────────────────────────────────────────────────
    if not S6A_HOLDOUT_MANIFEST.exists():
        print(f"[ABORT] s6a manifest not found: {S6A_HOLDOUT_MANIFEST}")
        sys.exit(2)

    df_raw = pd.read_csv(S6A_HOLDOUT_MANIFEST, low_memory=False)

    # stage2_holdout 행만 사용
    if 'stage_split' in df_raw.columns:
        df_stage2 = df_raw[df_raw['stage_split'] == 'stage2_holdout'].copy()
    else:
        df_stage2 = df_raw.copy()
    print(f"[MANIFEST] stage2_holdout rows: {len(df_stage2)}")

    # leakage 확인 (로그용)
    found_leakage = set(df_stage2['patient_id'].unique()) & LEAKAGE_PATIENTS
    print(f"[LEAKAGE] Found in manifest (to be excluded): {sorted(found_leakage)}")

    # hard exclude leakage
    df_after_leakage = df_stage2[~df_stage2['patient_id'].isin(LEAKAGE_PATIENTS)].copy()

    # excluded patient가 output plan에 있으면 즉시 exit 2
    check_leakage_guard(df_after_leakage['patient_id'].unique())

    # ── LUNG1-only source filter (P-C-NORMAL22e0c patch) ─────────────────────
    msd_patients_before = [p for p in df_after_leakage['patient_id'].unique()
                           if not str(p).startswith('LUNG1')]
    msd_count = len(msd_patients_before)
    msd_rows_count = int((~df_after_leakage['patient_id'].str.startswith('LUNG1')).sum())
    if msd_count != EXPECTED_MSD_EXCLUDED_PATIENTS:
        print(f"[ABORT] MSD_lung patient count unexpected: {msd_count}/{EXPECTED_MSD_EXCLUDED_PATIENTS}")
        sys.exit(2)
    df_clean = df_after_leakage[df_after_leakage['patient_id'].str.startswith('LUNG1')].copy()
    print(f"[LUNG1-ONLY] MSD_lung excluded: {msd_count} patients / {msd_rows_count} rows")
    print(f"[LUNG1-ONLY] Clean LUNG1-only patients: {df_clean['patient_id'].nunique()}")

    clean_patients = df_clean['patient_id'].nunique()
    if clean_patients != EXPECTED_CLEAN_PATIENTS:
        print(f"[ABORT] Clean patient count mismatch: {clean_patients}/{EXPECTED_CLEAN_PATIENTS}")
        sys.exit(2)

    # ── Positive rows filter ──────────────────────────────────────────────────
    df_positive = df_clean[df_clean['sampling_label'] == 'positive'].copy()
    actual_rows = len(df_positive)
    if actual_rows != EXPECTED_NSCLC_POSITIVE_ROWS:
        print(f"[ABORT] Positive row count mismatch: {actual_rows}/{EXPECTED_NSCLC_POSITIVE_ROWS}")
        sys.exit(2)

    # ── MSD_lung absent guard ─────────────────────────────────────────────────
    msd_in_positive = [p for p in df_positive['patient_id'].unique()
                       if not str(p).startswith('LUNG1')]
    if msd_in_positive:
        print(f"[ABORT] MSD_lung found in positive rows: {msd_in_positive[:5]}")
        print("  MSD_lung은 NSCLC binary label 체계 밖 — generation 중단.")
        sys.exit(2)
    print("[MSD-GUARD] MSD_lung absent from positive rows: OK")

    # ── Derive center_y / center_x ────────────────────────────────────────────
    df_positive = df_positive.copy()
    df_positive['center_y'] = (df_positive['y0'] + df_positive['y1']) / 2.0
    df_positive['center_x'] = (df_positive['x0'] + df_positive['x1']) / 2.0
    print("[CENTER] center_y=(y0+y1)/2, center_x=(x0+x1)/2")

    # output plan에 leakage 없음 재확인
    check_leakage_guard(df_positive['patient_id'].unique())
    print(f"[GENERATE] {actual_rows} positive rows | {clean_patients} clean patients")

    # ── Create output directories ─────────────────────────────────────────────
    CROP_OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    MANIFEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DIRS] crop: {CROP_OUTPUT_DIR}")

    # ── Crop generation loop ──────────────────────────────────────────────────
    manifest_rows = []
    integrity_log = []
    error_rows    = []
    loaded_vols   = {}
    total         = len(df_positive)
    saved         = 0

    for row_idx, (_, row) in enumerate(df_positive.iterrows()):
        sid        = row['safe_id']
        patient_id = row['patient_id']
        ct_path    = NSCLC_CT_BASE / sid / CT_FILENAME

        try:
            if sid not in loaded_vols:
                if not ct_path.exists():
                    raise FileNotFoundError(f"CT not found: {ct_path}")
                loaded_vols[sid] = np.load(str(ct_path))
            ct_vol = loaded_vols[sid]

            lz = float(row['local_z'])
            cy = float(row['center_y'])
            cx = float(row['center_x'])

            crop = extract_crop(ct_vol, lz, cy, cx)

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

            row_id        = row.get('row_id', row_idx)
            crop_filename = f"nsclc_{sid}_{row_id}.npz"
            patient_dir   = CROP_OUTPUT_DIR / sid
            patient_dir.mkdir(parents=True, exist_ok=True)
            crop_path_abs = patient_dir / crop_filename
            np.savez_compressed(str(crop_path_abs), ct_crop=crop)  # 1-key npz
            GUARDRAIL['crop_generated'] = True
            saved += 1

            manifest_rows.append({
                'row_index':             row_idx,
                'final_candidate_id':    f"nsclc_{sid}_{row_id}",
                'patient_id':            patient_id,
                'safe_id':               sid,
                'split':                 'final_test',
                'source_split':          'stage2_holdout',
                'source_name':           's6a_stage2_holdout_lung1_nsclc',
                'crop_path':             str(crop_path_abs),
                'label':                 LABEL_NSCLC,
                'label_name':            'nsclc',
                'position_bin':          row['position_bin'],
                'z_level':               row['z_level'],
                'local_z':               row['local_z'],
                'slice_index':           row['slice_index'],
                'center_y':              row['center_y'],
                'center_x':              row['center_x'],
                'sample_weight':         1.0,
                'stage2_holdout_flag':   True,
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

        if (row_idx + 1) % 5000 == 0:
            print(f"  [{row_idx + 1}/{total}] saved={saved} errors={len(error_rows)}")

    print(f"[GENERATE] saved={saved}/{total} errors={len(error_rows)}")

    # ── Post-run hard assertion (P-C-NORMAL22d2) — manifest 저장 전 필수 통과 ──
    _post_run_hard_assertion_nsclc(saved, error_rows, manifest_rows, integrity_log)

    # ── Save NSCLC manifest fragment ──────────────────────────────────────────
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(str(NSCLC_FRAGMENT_CSV), index=False)
    GUARDRAIL['manifest_generated'] = True
    print(f"[MANIFEST] fragment → {NSCLC_FRAGMENT_CSV} ({len(df_manifest)} rows)")

    # ── Save integrity log ────────────────────────────────────────────────────
    pd.DataFrame(integrity_log).to_csv(
        str(MANIFEST_OUTPUT_DIR / 'p_c_normal22d_nsclc_integrity_log.csv'), index=False
    )

    # ── Save errors (append if normal errors already exists) ──────────────────
    errors_path = MANIFEST_OUTPUT_DIR / 'p_c_normal22_errors.csv'
    mode = 'a' if errors_path.exists() else 'w'
    pd.DataFrame(error_rows).to_csv(str(errors_path), index=False,
                                    mode=mode, header=(mode == 'w'))

    elapsed = (datetime.now() - run_start).total_seconds()
    print(f"[22d-NSCLC GENERATE] Done. elapsed={elapsed:.1f}s | saved={saved} | errors={len(error_rows)}")
    return saved, len(error_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Manifest combiner (Option A) — --combine 모드로 실행
# normal + nsclc fragments → combined manifest + summary + report + DONE.json
# ─────────────────────────────────────────────────────────────────────────────

def run_combine_manifest(confirm_final_test: bool = False,
                         confirm_no_prediction: bool = False,
                         confirm_no_threshold: bool = False):
    """
    normal + nsclc manifest fragments를 combined final manifest로 합친다.
    P-C-NORMAL22d2 hardening:
      - ALLOW_REAL_COMBINE=True + 세 가지 confirm 플래그 모두 필요.
      - combined manifest / DONE.json이 이미 있으면 exit 2.
      - hard validation 통과 후에만 출력 저장.
      - prediction / model_forward / threshold 없음.
    """
    # ── Guard 1: ALLOW_REAL_COMBINE flag ─────────────────────────────────────
    if not ALLOW_REAL_COMBINE:
        print("[ABORT] ALLOW_REAL_COMBINE=False. P-C-NORMAL22e 별도 승인 후 True로 변경 필요.")
        print("  --combine은 manifest actual generation이므로 사용자 승인 필수.")
        sys.exit(2)

    # ── Guard 2: 세 가지 confirm 플래그 모두 필요 ────────────────────────────
    if not (confirm_final_test and confirm_no_prediction and confirm_no_threshold):
        print("[ABORT] --combine 실행에는 --confirm-final-test --confirm-no-prediction --confirm-no-threshold 모두 필요.")
        sys.exit(2)

    combine_start = datetime.now()
    print(f"[22e-COMBINE] Start: {combine_start.isoformat()}")

    NORMAL_FRAGMENT_CSV   = MANIFEST_OUTPUT_DIR / 'p_c_normal22_final_test_normal_manifest.csv'
    NSCLC_FRAGMENT_CSV    = MANIFEST_OUTPUT_DIR / 'p_c_normal22_final_test_nsclc_manifest.csv'
    COMBINED_MANIFEST_CSV = MANIFEST_OUTPUT_DIR / 'p_c_normal22_final_test_manifest.csv'
    SUMMARY_JSON          = MANIFEST_OUTPUT_DIR / 'p_c_normal22_final_test_manifest_summary.json'
    REPORT_MD             = MANIFEST_OUTPUT_DIR / 'p_c_normal22_final_test_manifest_report.md'
    ERRORS_CSV            = MANIFEST_OUTPUT_DIR / 'p_c_normal22_errors.csv'
    DONE_JSON             = MANIFEST_OUTPUT_DIR / 'DONE.json'

    EXPECTED_TOTAL    = 66323   # 21600 + 44723 (LUNG1-only, P-C-NORMAL22e0c patch)
    EXPECTED_NORMAL   = 21600
    EXPECTED_NSCLC    = 44723   # LUNG1-only corrected (was 51313 incl MSD_lung 6590 rows)
    EXPECTED_NORM_PAT = 36
    EXPECTED_NSCLC_PAT = 122    # LUNG1-only corrected (was 152 incl MSD_lung 30 patients)

    # ── Collision check ────────────────────────────────────────────────────────
    for col_label, col_path in [
        ('combined_manifest', COMBINED_MANIFEST_CSV),
        ('DONE_json',         DONE_JSON),
    ]:
        if col_path.exists():
            print(f"[ABORT] Combine collision — {col_label}: {col_path}")
            sys.exit(2)

    # ── Fragment existence check ───────────────────────────────────────────────
    for frag_label, frag_path in [
        ('normal_fragment', NORMAL_FRAGMENT_CSV),
        ('nsclc_fragment',  NSCLC_FRAGMENT_CSV),
    ]:
        if not frag_path.exists():
            print(f"[ABORT] Fragment not found — {frag_label}: {frag_path}")
            print("  --run-generate 를 먼저 실행하세요.")
            sys.exit(2)

    df_normal = pd.read_csv(str(NORMAL_FRAGMENT_CSV), low_memory=False)
    df_nsclc  = pd.read_csv(str(NSCLC_FRAGMENT_CSV),  low_memory=False)

    # ── Pre-combine hard validation ────────────────────────────────────────────
    combine_failures = []

    if len(df_normal) != EXPECTED_NORMAL:
        combine_failures.append(f"normal_rows={len(df_normal)} != {EXPECTED_NORMAL}")
    if len(df_nsclc) != EXPECTED_NSCLC:
        combine_failures.append(f"nsclc_rows={len(df_nsclc)} != {EXPECTED_NSCLC}")

    norm_patients = int(df_normal['patient_id'].nunique())
    nsclc_patients = int(df_nsclc['patient_id'].nunique())
    if norm_patients != EXPECTED_NORM_PAT:
        combine_failures.append(f"normal_patients={norm_patients} != {EXPECTED_NORM_PAT}")
    if nsclc_patients != EXPECTED_NSCLC_PAT:
        combine_failures.append(f"nsclc_patients={nsclc_patients} != {EXPECTED_NSCLC_PAT}")

    # label distribution
    if set(df_normal['label'].unique()) != {0}:
        combine_failures.append(f"normal label not only 0: {df_normal['label'].unique()}")
    if set(df_nsclc['label'].unique()) != {1}:
        combine_failures.append(f"nsclc label not only 1: {df_nsclc['label'].unique()}")

    # source_split distribution
    if set(df_normal['source_split'].unique()) != {'normal_test'}:
        combine_failures.append(f"normal source_split unexpected: {df_normal['source_split'].unique()}")
    if set(df_nsclc['source_split'].unique()) != {'stage2_holdout'}:
        combine_failures.append(f"nsclc source_split unexpected: {df_nsclc['source_split'].unique()}")

    # leakage hard check on nsclc fragment
    leakage_in_nsclc = [p for p in df_nsclc['patient_id'].unique() if p in LEAKAGE_PATIENTS]
    if leakage_in_nsclc:
        print(f"[ABORT] LEAKAGE patients in nsclc fragment: {leakage_in_nsclc}")
        sys.exit(2)
    # leakage hard check via function
    check_leakage_guard(df_nsclc['patient_id'].unique())

    # MSD_lung absent guard on nsclc fragment (P-C-NORMAL22e0c patch)
    msd_in_nsclc = [p for p in df_nsclc['patient_id'].unique()
                    if not str(p).startswith('LUNG1')]
    if msd_in_nsclc:
        print(f"[ABORT] MSD_lung found in nsclc fragment: {msd_in_nsclc[:5]}")
        print("  MSD_lung은 NSCLC binary label 체계 밖 — combine 중단.")
        sys.exit(2)
    print(f"[MSD-GUARD] MSD_lung absent from nsclc fragment: OK")

    # flag checks on nsclc
    if df_nsclc['hard_negative_flag'].any():
        combine_failures.append("nsclc hard_negative_flag has True")
    if df_nsclc['msd_lung_flag'].any():
        combine_failures.append("nsclc msd_lung_flag has True")

    # flag checks on normal
    if df_normal['hard_negative_flag'].any():
        combine_failures.append("normal hard_negative_flag has True")
    if df_normal['msd_lung_flag'].any():
        combine_failures.append("normal msd_lung_flag has True")

    # crop_path uniqueness
    all_crop_paths = list(df_normal['crop_path']) + list(df_nsclc['crop_path'])
    if len(set(all_crop_paths)) != EXPECTED_TOTAL:
        combine_failures.append(f"crop_path_unique={len(set(all_crop_paths))} != {EXPECTED_TOTAL}")

    # final_candidate_id uniqueness
    all_ids = list(df_normal['final_candidate_id']) + list(df_nsclc['final_candidate_id'])
    if len(set(all_ids)) != EXPECTED_TOTAL:
        combine_failures.append(f"final_candidate_id_unique={len(set(all_ids))} != {EXPECTED_TOTAL}")

    # schema columns check
    required_columns = [
        'row_index', 'final_candidate_id', 'patient_id', 'safe_id',
        'split', 'source_split', 'source_name', 'crop_path',
        'label', 'label_name', 'position_bin', 'z_level',
        'local_z', 'slice_index', 'center_y', 'center_x',
        'sample_weight', 'stage2_holdout_flag', 'hard_negative_flag',
        'msd_lung_flag', 'leakage_excluded_flag', 'generator_version',
        'crop_shape', 'crop_dtype',
    ]
    for col in required_columns:
        if col not in df_normal.columns:
            combine_failures.append(f"normal fragment missing column: {col}")
        if col not in df_nsclc.columns:
            combine_failures.append(f"nsclc fragment missing column: {col}")

    if combine_failures:
        print("[COMBINE HARD VALIDATION FAIL]:")
        for fail in combine_failures:
            print(f"  - {fail}")
        print("[ABORT] combined manifest / DONE.json 저장 중단.")
        sys.exit(2)

    print("[COMBINE HARD VALIDATION PASS] All pre-combine checks passed.")

    # ── Build combined manifest ────────────────────────────────────────────────
    df_combined = pd.concat([df_normal, df_nsclc], ignore_index=True)
    df_combined['row_index'] = range(len(df_combined))

    # ── Post-combine hard validation ──────────────────────────────────────────
    post_failures = []
    if len(df_combined) != EXPECTED_TOTAL:
        post_failures.append(f"combined_rows={len(df_combined)} != {EXPECTED_TOTAL}")

    label_dist = df_combined['label'].value_counts().to_dict()
    if label_dist.get(0, 0) != EXPECTED_NORMAL:
        post_failures.append(f"label_0_count={label_dist.get(0,0)} != {EXPECTED_NORMAL}")
    if label_dist.get(1, 0) != EXPECTED_NSCLC:
        post_failures.append(f"label_1_count={label_dist.get(1,0)} != {EXPECTED_NSCLC}")

    source_dist = df_combined['source_split'].value_counts().to_dict()
    if source_dist.get('normal_test', 0) != EXPECTED_NORMAL:
        post_failures.append(f"normal_test_rows={source_dist.get('normal_test',0)} != {EXPECTED_NORMAL}")
    if source_dist.get('stage2_holdout', 0) != EXPECTED_NSCLC:
        post_failures.append(f"stage2_holdout_rows={source_dist.get('stage2_holdout',0)} != {EXPECTED_NSCLC}")

    leakage_in_combined = [p for p in df_combined['patient_id'].unique() if p in LEAKAGE_PATIENTS]
    if leakage_in_combined:
        print(f"[ABORT] LEAKAGE in combined manifest: {leakage_in_combined}")
        sys.exit(2)

    # MSD_lung absent guard on combined manifest (P-C-NORMAL22e3 fix: msd_lung_flag 컬럼 기반)
    # 이전: patient_id prefix 기반 → LUNA subset 정상환자(subset0_/subset1_)를 오탐
    # 수정: msd_lung_flag 컬럼이 authoritative source of truth
    msd_in_combined = df_combined[df_combined['msd_lung_flag'] == True]['patient_id'].unique().tolist()
    if msd_in_combined:
        print(f"[ABORT] MSD_lung in combined manifest: {msd_in_combined[:5]}")
        print("  MSD_lung은 NSCLC binary label 체계 밖 — combined manifest 저장 중단.")
        sys.exit(2)

    if post_failures:
        print("[POST-COMBINE HARD VALIDATION FAIL]:")
        for fail in post_failures:
            print(f"  - {fail}")
        print("[ABORT] combined manifest / DONE.json 저장 중단.")
        sys.exit(2)

    print("[POST-COMBINE VALIDATION PASS] All post-combine checks passed.")

    # ── Save combined manifest ─────────────────────────────────────────────────
    df_combined.to_csv(str(COMBINED_MANIFEST_CSV), index=False)
    print(f"[COMBINE] Combined manifest → {COMBINED_MANIFEST_CSV} ({len(df_combined)} rows)")

    summary = {
        'generator':            'p_c_normal22e_combine',
        'date':                 combine_start.isoformat(),
        'total_rows':           len(df_combined),
        'normal_rows':          len(df_normal),
        'nsclc_rows':           len(df_nsclc),
        'normal_patients':      norm_patients,
        'nsclc_patients':       nsclc_patients,
        'expected_total_rows':  EXPECTED_TOTAL,
        'expected_normal_rows': EXPECTED_NORMAL,
        'expected_nsclc_rows':  EXPECTED_NSCLC,
        'label_distribution':   {int(k): int(v) for k, v in label_dist.items()},
        'split_distribution':   source_dist,
        'leakage_excluded':     sorted(LEAKAGE_PATIENTS),
        'manifest_validation_pass': True,
        'guardrail': {
            'prediction_export_run': False,
            'model_forward_run':     False,
            'threshold_computed':    False,
            'training_run':          False,
        },
    }
    with open(str(SUMMARY_JSON), 'w') as _f:
        json.dump(summary, _f, indent=2)

    with open(str(REPORT_MD), 'w') as _f:
        _f.write("# P-C-NORMAL22 Final Baseline Test Manifest Report\n\n")
        _f.write(f"- **date**: {combine_start.isoformat()}\n")
        _f.write(f"- **total_rows**: {len(df_combined)}\n")
        _f.write(f"  - normal: {len(df_normal)}\n")
        _f.write(f"  - nsclc: {len(df_nsclc)}\n")
        _f.write(f"- **normal_patients**: {norm_patients}\n")
        _f.write(f"- **nsclc_patients (clean)**: {nsclc_patients}\n")
        _f.write(f"- **leakage_excluded**: {sorted(LEAKAGE_PATIENTS)}\n")
        _f.write("- **prediction**: NONE\n")
        _f.write("- **model_forward**: NONE\n")
        _f.write("- **threshold**: NONE\n\n")
        _f.write("## Label Distribution\n\n")
        for label_val, cnt in label_dist.items():
            _f.write(f"- label={label_val}: {cnt}\n")

    if not ERRORS_CSV.exists():
        pd.DataFrame(columns=['row_index', 'patient_id', 'safe_id', 'error']).to_csv(
            str(ERRORS_CSV), index=False
        )

    # ── DONE.json (P-C-NORMAL22d2 강화 스키마) ──────────────────────────────
    with open(str(DONE_JSON), 'w') as _f:
        json.dump({
            'stage':                              'P-C-NORMAL22e',
            'status':                             'done',
            'date':                               combine_start.isoformat(),
            'conditions_ok':                      True,
            'total_rows':                         len(df_combined),
            'normal_rows':                        len(df_normal),
            'nsclc_rows':                         len(df_nsclc),
            'normal_patients':                    norm_patients,
            'nsclc_patients':                     nsclc_patients,
            'nsclc_source':                       'LUNG1_only',
            'leakage_excluded':                   sorted(LEAKAGE_PATIENTS),
            'msd_lung_excluded':                  True,
            'msd_lung_excluded_patients':         EXPECTED_MSD_EXCLUDED_PATIENTS,
            'msd_lung_excluded_rows':             EXPECTED_MSD_EXCLUDED_ROWS,
            'hard_negative_count':                0,
            'msd_lung_count':                     0,
            'stage2_holdout_used_for_final_test_only': True,
            'model_forward_run':                  False,
            'prediction_export_run':              False,
            'threshold_computed':                 False,
            'training_run':                       False,
            'crop_generation_errors':             0,
            'manifest_validation_pass':           True,
        }, _f, indent=2)

    elapsed = (datetime.now() - combine_start).total_seconds()
    print(f"[22e-COMBINE] Done. elapsed={elapsed:.1f}s | rows={len(df_combined)}")
    return len(df_combined)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='P-C-NORMAL22c NSCLC Holdout Crop Generator')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-check',    action='store_true')
    mode.add_argument('--run-generate', action='store_true')
    mode.add_argument('--combine',      action='store_true',
                      help='Combine normal + nsclc manifest fragments (run after both --run-generate)')
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
    elif args.combine:
        # P-C-NORMAL22d2: ALLOW_REAL_COMBINE=False → exit 2, confirm flags 필요
        run_combine_manifest(
            confirm_final_test=args.confirm_final_test,
            confirm_no_prediction=args.confirm_no_prediction,
            confirm_no_threshold=args.confirm_no_threshold,
        )
        sys.exit(0)
    elif args.dry_check:
        decision = run_dry_check()
        sys.exit(0 if decision == 'READY_FOR_FINAL_TEST_CROP_GENERATION' else 1)
    else:
        print("[ABORT] --dry-check / --run-generate / --combine 중 하나 필요.")
        sys.exit(2)


if __name__ == '__main__':
    main()
