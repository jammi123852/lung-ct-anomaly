#!/usr/bin/env python3
"""
S3 expansion manifest creation - manifest only, no card generation.
Scope: review_balanced_n20_l80_top3
Forbidden: card generation, PNG/JSON creation, CT/mask load, score recomputation,
           model forward, threshold recomputation, stage2_holdout access.
"""
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

BASE = Path('/home/jinhy/project/lung-ct-anomaly')
OUTROOT = BASE / 'outputs/position-aware-padim-v1'

# Input paths
COMP_CSV = OUTROOT / 'candidates/padim_v2_roi0_0_explanation_candidates_v1/component_candidates.csv'
PATIENT_CSV = OUTROOT / 'candidates/padim_v2_roi0_0_explanation_candidates_v1/patient_candidate_summary.csv'
SPLIT_CSV = BASE / 'outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv'
SCORE_NORMAL_DIR = OUTROOT / 'scores/padim_v2_roi0_0/normal_by_patient'
SCORE_LESION_DIR = OUTROOT / 'scores/padim_v2_roi0_0/lesion_v2_by_patient'

# Output paths
MANIFEST_ROOT = OUTROOT / 'candidates/s3_expansion_manifest_v1'
REPORT_ROOT = OUTROOT / 'reports/explanation_cards'

# Guard: output root must not exist yet
if MANIFEST_ROOT.exists():
    print(f"BLOCKED: manifest root already exists: {MANIFEST_ROOT}")
    sys.exit(1)

# Constants
THRESHOLD = 14.0921
THRESHOLD_TYPE = 'p95'
NORMAL_TARGET = 20
LESION_TARGET = 80
TOP_K = 3

# Prototype patient IDs (from s3_prototype_candidate_manifest_v1.csv)
PROTO_NORMALS = {
    'subset9_1.3.6.1.4.1.14519.5.2.1.6279.6001.291156498203266896953765649282',
    'subset8_1.3.6.1.4.1.14519.5.2.1.6279.6001.175318131822744218104175746898',
    'normal004',
}
PROTO_LESIONS = {'MSD_lung_054', 'MSD_lung_071', 'LUNG1-402', 'LUNG1-057', 'LUNG1-284'}
PROTO_ALL = PROTO_NORMALS | PROTO_LESIONS

FORBIDDEN_TOKENS = ('stage2_holdout', 'holdout')

def assert_no_holdout(values, context):
    for v in values:
        s = str(v)
        for tok in FORBIDDEN_TOKENS:
            if tok in s:
                raise ValueError(f"HOLDOUT VIOLATION [{context}]: {v}")

started_at = datetime.utcnow().isoformat() + 'Z'
errors = []

print("=== S3 Expansion Manifest v1 ===")
print(f"Started: {started_at}")

# === 1. Load patient summary ===
patients = pd.read_csv(PATIENT_CSV)
assert_no_holdout(patients['patient_id'], 'patient_summary')
assert all(patients['stage_split_safety_flag'].isin(['normal', 'stage1_dev'])), \
    "Unexpected stage_split_safety_flag"

normals_df = patients[patients['label'] == 'normal'].copy()
lesions_df = patients[patients['label'] != 'normal'].copy()
print(f"[1] Patients loaded: {len(normals_df)} normal, {len(lesions_df)} lesion")

# === 2. Build holdout denylist ===
split_df = pd.read_csv(SPLIT_CSV)
holdout_set = set(split_df[split_df['stage_split'] == 'stage2_holdout']['patient_id'])
stage1_dev_set = set(split_df[split_df['stage_split'] == 'stage1_dev']['patient_id'])
holdout_intersection = set(patients['patient_id']) & holdout_set
assert len(holdout_intersection) == 0, f"HOLDOUT VIOLATION: {holdout_intersection}"
print(f"[2] Holdout denylist: {len(holdout_set)} patients — intersection with S2=0 PASS")

# === 3. Select 20 normal patients ===
# Prototype normals first (3명), then top 17 by max_padim_score from remaining
proto_n = normals_df[normals_df['patient_id'].isin(PROTO_NORMALS)]
remaining_n = normals_df[~normals_df['patient_id'].isin(PROTO_NORMALS)] \
    .sort_values('max_padim_score', ascending=False)
fill_n = remaining_n.head(NORMAL_TARGET - len(proto_n))
selected_normals = pd.concat([proto_n, fill_n]).drop_duplicates(subset='patient_id')
assert len(selected_normals) == NORMAL_TARGET, \
    f"Normal count mismatch: expected {NORMAL_TARGET}, got {len(selected_normals)}"
print(f"[3] Normal selected: {len(selected_normals)}")
print(f"    Prototype normals included: {sorted(selected_normals[selected_normals['patient_id'].isin(PROTO_NORMALS)]['patient_id'].tolist())[:3]}")

# === 4. Select 80 lesion patients ===
# Prototype lesions first (5명), then top 75 by max_padim_score from remaining
proto_l = lesions_df[lesions_df['patient_id'].isin(PROTO_LESIONS)]
remaining_l = lesions_df[~lesions_df['patient_id'].isin(PROTO_LESIONS)] \
    .sort_values('max_padim_score', ascending=False)
fill_l = remaining_l.head(LESION_TARGET - len(proto_l))
selected_lesions = pd.concat([proto_l, fill_l]).drop_duplicates(subset='patient_id')
assert len(selected_lesions) == LESION_TARGET, \
    f"Lesion count mismatch: expected {LESION_TARGET}, got {len(selected_lesions)}"
assert_no_holdout(selected_lesions['patient_id'], 'lesion selection')
assert all(selected_lesions['stage_split_safety_flag'] == 'stage1_dev'), \
    "Non-stage1_dev in lesion selection"
print(f"[4] Lesion selected: {len(selected_lesions)}")
proto_incl = sorted(selected_lesions[selected_lesions['patient_id'].isin(PROTO_LESIONS)]['patient_id'].tolist())
print(f"    Prototype lesions included: {proto_incl}")

# upper_peripheral check
up_peri = selected_lesions[selected_lesions['top_component_position_bin'] == 'upper_peripheral']
print(f"    Upper_peripheral top-component: {len(up_peri)} patients: {up_peri['patient_id'].tolist()}")

# === 5. Load component candidates for selected patients (chunked) ===
selected_pids = set(selected_normals['patient_id']) | set(selected_lesions['patient_id'])
print(f"[5] Loading components for {len(selected_pids)} patients...")
chunks = []
for chunk in pd.read_csv(COMP_CSV, chunksize=5000):
    filt = chunk[chunk['patient_id'].isin(selected_pids)]
    if len(filt) > 0:
        chunks.append(filt)
comps = pd.concat(chunks).reset_index(drop=True)
print(f"    Components loaded: {len(comps)} rows")

# Verify all selected patients have components
pids_in_comps = set(comps['patient_id'].unique())
missing_comp = selected_pids - pids_in_comps
if missing_comp:
    print(f"    WARNING: {len(missing_comp)} patients have no components in CSV: {missing_comp}")
    for pid in missing_comp:
        errors.append({'patient_id': pid, 'error_type': 'no_component_found', 'message': 'No component in component_candidates.csv'})

# === 6. Get z_ratio lookup from score CSVs (read-only, existing pre-computed scores) ===
print(f"[6] Building z_ratio lookup from score CSVs...")
slice_z_lookup = {}  # (patient_id, slice_index) -> z_ratio
z_ratio_source = 'score_csv'
z_missing_count = 0

for pid in sorted(selected_pids):
    is_normal = pid in set(selected_normals['patient_id'])
    score_path = (SCORE_NORMAL_DIR if is_normal else SCORE_LESION_DIR) / f"{pid}.csv"
    if score_path.exists():
        try:
            sc = pd.read_csv(score_path, usecols=['slice_index', 'z_ratio'])
            for si, zr in sc[['slice_index', 'z_ratio']].drop_duplicates('slice_index').itertuples(index=False):
                slice_z_lookup[(pid, int(si))] = float(zr)
        except Exception as e:
            z_missing_count += 1
            errors.append({'patient_id': pid, 'error_type': 'score_csv_read_error', 'message': str(e)})
    else:
        z_missing_count += 1
        errors.append({'patient_id': pid, 'error_type': 'score_csv_not_found',
                       'message': f'Not found: {score_path}'})

if z_missing_count > 0:
    z_ratio_source = 'score_csv_with_fallback'
print(f"    z_ratio lookup entries: {len(slice_z_lookup)}, missing patients: {z_missing_count}")

# === 7. Select top-K per patient ===
print(f"[7] Selecting top{TOP_K} components per patient...")
top_k_list = []
for pid in selected_pids:
    pcomps = comps[comps['patient_id'] == pid].sort_values('rank_in_patient').head(TOP_K)
    if len(pcomps) < TOP_K:
        print(f"    WARNING: {pid} has only {len(pcomps)} components (expected {TOP_K})")
        errors.append({'patient_id': pid, 'error_type': 'less_than_topK',
                       'message': f'Only {len(pcomps)} components'})
    top_k_list.append(pcomps)

manifest = pd.concat(top_k_list).reset_index(drop=True)
print(f"    Manifest rows: {len(manifest)}")

# === 8. Compute derived columns ===
manifest = manifest.copy()

# component_bbox_area
manifest['component_bbox_area'] = (manifest['y1'] - manifest['y0']) * \
                                    (manifest['x1'] - manifest['x0'])

# overmerge_flag and level
def get_overmerge(row):
    a, z, p = row['component_bbox_area'], row['z_span'], row['patch_count']
    if a >= 100000 or z >= 50 or p >= 1000:
        return True, 'extreme_union'
    elif a >= 25000 or z >= 30 or p >= 500:
        return True, 'large_union'
    return False, 'none'

om_results = manifest.apply(lambda r: pd.Series(get_overmerge(r)), axis=1)
manifest['overmerge_flag'] = om_results[0]
manifest['overmerge_level'] = om_results[1]

def make_overmerge_reason(r):
    parts = []
    if r['component_bbox_area'] >= 25000:
        parts.append(f"bbox_area={int(r['component_bbox_area'])}")
    if r['z_span'] >= 30:
        parts.append(f"z_span={int(r['z_span'])}")
    if r['patch_count'] >= 500:
        parts.append(f"patch_count={int(r['patch_count'])}")
    return '|'.join(parts) if parts else 'none'

manifest['overmerge_reason'] = manifest.apply(make_overmerge_reason, axis=1)

# z_ratio from lookup (fallback: estimate from slice_index_max)
def get_z_ratio(r):
    key = (r['patient_id'], int(r['max_score_slice_index']))
    if key in slice_z_lookup:
        return slice_z_lookup[key]
    # Fallback: estimate
    max_known = max(int(r['slice_index_max']), int(r['max_score_slice_index']))
    return float(r['max_score_slice_index']) / max(max_known, 1)

manifest['z_ratio'] = manifest.apply(get_z_ratio, axis=1)

# apex_caution: z_ratio > 0.85 AND upper_peripheral
manifest['apex_caution'] = (
    (manifest['position_bin'] == 'upper_peripheral') &
    (manifest['z_ratio'] > 0.85)
)
manifest['apex_caution_reason'] = manifest.apply(
    lambda r: ('upper_peripheral_z_ratio_gt_0.85'
               if r['apex_caution'] else 'none'), axis=1)

# role, prototype_role, metadata
manifest['role'] = manifest['label'].apply(
    lambda l: 'normal_control' if l == 'normal' else 'lesion_candidate')
manifest['prototype_role'] = manifest['patient_id'].apply(
    lambda pid: 'prototype' if pid in PROTO_ALL else 'expansion')
manifest['expansion_case_id'] = manifest.apply(
    lambda r: f"{r['patient_id']}__c{int(r['rank_in_patient'])}", axis=1)
manifest['expansion_scope'] = 'review_balanced_n20_l80_top3'
manifest['internal_use_only'] = True
manifest['font_fix_required_before_external_share'] = True
manifest['threshold'] = THRESHOLD
manifest['threshold_type'] = THRESHOLD_TYPE
manifest['selection_rule'] = manifest.apply(
    lambda r: ('rank_top3_patch_count_ge2_or_zspan_ge1'
               if r['patch_count'] >= 2 or r['z_span'] >= 1 else 'rank_top3'), axis=1)
manifest['selection_reason'] = manifest.apply(
    lambda r: f"rank{int(r['rank_in_patient'])}_score_{r['max_padim_score']:.2f}", axis=1)
manifest['source_component_csv'] = str(COMP_CSV)
manifest['source_candidate_root'] = str(
    OUTROOT / 'candidates/padim_v2_roi0_0_explanation_candidates_v1')

# === 9. Final column order ===
output_cols = [
    'expansion_case_id', 'expansion_scope', 'internal_use_only',
    'font_fix_required_before_external_share',
    'group', 'patient_id', 'safe_id', 'prototype_role', 'role', 'label',
    'component_id', 'rank_in_patient', 'position_bin',
    'slice_index_min', 'slice_index_max', 'max_score_slice_index', 'z_span', 'z_ratio',
    'y0', 'x0', 'y1', 'x1',
    'patch_count', 'max_padim_score', 'mean_padim_score',
    'threshold', 'threshold_type',
    'roi_0_0_patch_ratio_mean', 'central_peripheral',
    'central_distance_ratio_mean', 'left_right_metadata',
    'component_bbox_area', 'overmerge_flag', 'overmerge_level', 'overmerge_reason',
    'apex_caution', 'apex_caution_reason',
    'selection_rule', 'selection_reason',
    'stage_split_safety_flag',
    'source_component_csv', 'source_candidate_root',
]
final = manifest[output_cols].copy()

# === 10. Static validation ===
print("[10] Running static validation...")
assert_no_holdout(final['patient_id'], 'final manifest')
assert_no_holdout(final['stage_split_safety_flag'], 'stage_split_flags')
assert_no_holdout(final['source_component_csv'], 'source paths')

n_total = len(final)
n_normal = int((final['label'] == 'normal').sum())
n_lesion = int((final['label'] != 'normal').sum())
n_p_normal = int(final[final['label'] == 'normal']['patient_id'].nunique())
n_p_lesion = int(final[final['label'] != 'normal']['patient_id'].nunique())
n_overmerge = int(final['overmerge_flag'].sum())
n_extreme = int((final['overmerge_level'] == 'extreme_union').sum())
n_large = int((final['overmerge_level'] == 'large_union').sum())
n_apex = int(final['apex_caution'].sum())

assert n_total <= 300, f"Too many cards: {n_total}"
assert n_p_normal == NORMAL_TARGET, f"Normal patients: {n_p_normal} != {NORMAL_TARGET}"
assert n_p_lesion == LESION_TARGET, f"Lesion patients: {n_p_lesion} != {LESION_TARGET}"
assert set(final['threshold'].unique()) == {THRESHOLD}
assert (final['y0'] < final['y1']).all(), "y0 >= y1 violation"
assert (final['x0'] < final['x1']).all(), "x0 >= x1 violation"
assert final['max_padim_score'].isna().sum() == 0, "NaN in max_padim_score"
dup_comps = int(final['component_id'].duplicated().sum())
assert dup_comps == 0, f"Duplicate component_ids: {dup_comps}"
assert set(final['internal_use_only'].unique()) == {True}
assert set(final['font_fix_required_before_external_share'].unique()) == {True}
valid_roles = {'normal_control', 'lesion_candidate'}
assert set(final['role'].unique()).issubset(valid_roles), \
    f"Invalid roles: {set(final['role'].unique()) - valid_roles}"

# position_bin distribution
bin_dist = final['position_bin'].value_counts().to_dict()
# overmerge_level distribution
level_dist = final['overmerge_level'].value_counts().to_dict()
# score stats
score_stats = {
    'min': float(final['max_padim_score'].min()),
    'median': float(final['max_padim_score'].median()),
    'max': float(final['max_padim_score'].max()),
    'mean': float(final['max_padim_score'].mean()),
}
# patch_count stats
patch_stats = {
    'min': int(final['patch_count'].min()),
    'median': int(final['patch_count'].median()),
    'max': int(final['patch_count'].max()),
}
# z_span stats
zspan_stats = {
    'min': int(final['z_span'].min()),
    'median': int(final['z_span'].median()),
    'max': int(final['z_span'].max()),
}

print(f"    Validation PASS")
print(f"    Total: {n_total} cards ({n_normal} normal / {n_lesion} lesion)")
print(f"    Patients: {n_p_normal} normal / {n_p_lesion} lesion")
print(f"    Overmerge: {n_overmerge} ({n_overmerge/n_total*100:.1f}%) | extreme={n_extreme} | large={n_large}")
print(f"    Apex caution: {n_apex} ({n_apex/n_total*100:.1f}%)")
print(f"    Position bin: {bin_dist}")
print(f"    Score: min={score_stats['min']:.2f} median={score_stats['median']:.2f} max={score_stats['max']:.2f}")

# === 11. Write outputs ===
print("[11] Writing outputs...")
MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

final.to_csv(MANIFEST_ROOT / 's3_expansion_candidate_manifest_v1.csv', index=False)
print(f"    Written: s3_expansion_candidate_manifest_v1.csv ({n_total} rows)")

# Patient summary
ps_rows = []
for pid, grp in final.groupby('patient_id'):
    ps_rows.append({
        'expansion_patient_id': pid,
        'prototype_role': grp['prototype_role'].iloc[0],
        'group': grp['group'].iloc[0],
        'patient_id': pid,
        'safe_id': grp['safe_id'].iloc[0],
        'label': grp['label'].iloc[0],
        'n_selected_components': len(grp),
        'selected_component_ids': '|'.join(grp['component_id'].tolist()),
        'top_selected_score': float(grp['max_padim_score'].max()),
        'position_bins_selected': '|'.join(sorted(grp['position_bin'].unique().tolist())),
        'n_overmerge_flag': int(grp['overmerge_flag'].sum()),
        'n_extreme_union': int((grp['overmerge_level'] == 'extreme_union').sum()),
        'n_apex_caution': int(grp['apex_caution'].sum()),
        'selection_reason': grp['selection_reason'].iloc[0],
        'stage_split_safety_flag': grp['stage_split_safety_flag'].iloc[0],
    })
ps_df = pd.DataFrame(ps_rows)
ps_df.to_csv(MANIFEST_ROOT / 's3_expansion_patient_summary_v1.csv', index=False)
print(f"    Written: s3_expansion_patient_summary_v1.csv ({len(ps_df)} rows)")

finished_at = datetime.utcnow().isoformat() + 'Z'

# runtime_summary.json
runtime_summary = {
    'report': 'S3 expansion manifest v1 - manifest only, no card generation',
    'mode': 's3_expansion_manifest_v1',
    'expansion_scope': 'review_balanced_n20_l80_top3',
    'normal_target_patients': NORMAL_TARGET,
    'lesion_target_patients': LESION_TARGET,
    'top_k': TOP_K,
    'n_selected_patients': n_p_normal + n_p_lesion,
    'n_normal_patients': n_p_normal,
    'n_lesion_patients': n_p_lesion,
    'n_selected_components': n_total,
    'n_normal_cards': n_normal,
    'n_lesion_cards': n_lesion,
    'n_overmerge_flag': n_overmerge,
    'n_extreme_union': n_extreme,
    'n_large_union': n_large,
    'n_apex_caution': n_apex,
    'threshold': THRESHOLD,
    'threshold_type': THRESHOLD_TYPE,
    'internal_use_only': True,
    'font_fix_required_before_external_share': True,
    'holdout_intersection': 0,
    'stage2_holdout_accessed': False,
    'score_recomputed': False,
    'threshold_recomputed': False,
    'model_forward': False,
    'ct_mask_loaded': False,
    'existing_artifacts_modified': False,
    'total_slice_count_source': z_ratio_source,
    'z_ratio_missing_patients': z_missing_count,
    'overmerge_rate': round(n_overmerge / n_total, 4),
    'apex_caution_rate': round(n_apex / n_total, 4),
    'position_bin_distribution': bin_dist,
    'overmerge_level_distribution': level_dist,
    'score_stats': score_stats,
    'patch_count_stats': patch_stats,
    'z_span_stats': zspan_stats,
    'prototype_patients_included': sorted(
        ps_df[ps_df['prototype_role'] == 'prototype']['patient_id'].tolist()),
    'errors_count': len(errors),
    'done': True,
    'started_at': started_at,
    'finished_at': finished_at,
}
with open(MANIFEST_ROOT / 'runtime_summary.json', 'w', encoding='utf-8') as f:
    json.dump(runtime_summary, f, indent=2, ensure_ascii=False)
print("    Written: runtime_summary.json")

# errors.csv
with open(MANIFEST_ROOT / 'errors.csv', 'w', newline='', encoding='utf-8') as f:
    f.write('patient_id,error_type,message\n')
    for e in errors:
        msg = str(e['message']).replace(',', ';').replace('\n', ' ')
        f.write(f"{e['patient_id']},{e['error_type']},{msg}\n")
print(f"    Written: errors.csv ({len(errors)} entries)")

# DONE.json
with open(MANIFEST_ROOT / 'DONE.json', 'w') as f:
    json.dump({'done': True, 'finished_at': finished_at, 'manifest_rows': n_total}, f, indent=2)
print("    Written: DONE.json")

print(f"\n=== MANIFEST COMPLETE ===")
print(f"Cards: {n_total} | Normal: {n_normal} ({n_p_normal}명) | Lesion: {n_lesion} ({n_p_lesion}명)")
print(f"Overmerge: {n_overmerge}/{n_total} ({n_overmerge/n_total*100:.1f}%) | Extreme: {n_extreme}")
print(f"Apex caution: {n_apex}/{n_total} ({n_apex/n_total*100:.1f}%)")
print(f"z_ratio source: {z_ratio_source}")
print(f"Errors: {len(errors)}")
print(f"Output: {MANIFEST_ROOT}")
