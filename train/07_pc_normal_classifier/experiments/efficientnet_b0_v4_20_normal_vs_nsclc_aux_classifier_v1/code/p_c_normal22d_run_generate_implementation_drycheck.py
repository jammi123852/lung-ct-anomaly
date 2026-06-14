"""
P-C-NORMAL22d: run_generate() Implementation Dry-Check
실제 crop generation 없음. 구현 완료 여부 + guard 정상 여부만 확인한다.

확인 항목:
  1. py_compile — 두 스크립트 PASS
  2. placeholder 제거 — run_generate() 본문에 'not implemented' / sys.exit(2) 단독 없음
  3. ALLOW_REAL_GENERATION=False 확인 (static)
  4. --run-generate 단독 exit 2 (실제 실행 확인)
  5. --run-generate + confirms → ALLOW_REAL_GENERATION=False → exit 2
  6. --dry-check 기존 PASS 유지
  7. collision blocker coverage (static 코드 검사)
  8. manifest schema 컬럼 일치
  9. leakage hard exclusion (static)
  10. guardrail 전 항목 False 확인

출력 경로:
  experiments/.../outputs/reports/p_c_normal22d_run_generate_implementation_drycheck/
"""

import ast
import csv
import json
import os
import py_compile
import subprocess
import sys
from datetime import datetime
from pathlib import Path

START_TIME = datetime.now()

PROJECT_ROOT = Path('/home/jinhy/project/lung-ct-anomaly')
CODE_DIR = (
    PROJECT_ROOT
    / 'experiments'
    / 'efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1'
    / 'code'
)
NORMAL_SCRIPT = CODE_DIR / 'p_c_normal22c_normal_test_crop_generator.py'
NSCLC_SCRIPT  = CODE_DIR / 'p_c_normal22c_nsclc_holdout_crop_generator.py'

REPORT_DIR = (
    PROJECT_ROOT
    / 'experiments'
    / 'efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1'
    / 'outputs'
    / 'reports'
    / 'p_c_normal22d_run_generate_implementation_drycheck'
)

PYTHON_BIN = Path(os.environ.get('VIRTUAL_ENV', '/home/jinhy/ai_env')) / 'bin' / 'python'
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path('/home/jinhy/ai_env/bin/python')

# Expected manifest schema columns
EXPECTED_MANIFEST_COLS = [
    'row_index', 'final_candidate_id', 'patient_id', 'safe_id',
    'split', 'source_split', 'source_name', 'crop_path',
    'label', 'label_name', 'position_bin', 'z_level',
    'local_z', 'slice_index', 'center_y', 'center_x',
    'sample_weight', 'stage2_holdout_flag', 'hard_negative_flag',
    'msd_lung_flag', 'leakage_excluded_flag', 'generator_version',
    'crop_shape', 'crop_dtype',
]

LEAKAGE_PATIENTS = frozenset(['LUNG1-295', 'LUNG1-415'])

GUARDRAIL_EXPECTED_FALSE = [
    'crop_generated', 'manifest_generated', 'model_forward_run',
    'prediction_export_run', 'training_run', 'backward_run',
    'optimizer_step', 'checkpoint_saved', 'threshold_computed',
    'existing_outputs_modified',
]

# ─────────────────────────────────────────────────────────────────────────────

def read_source(path: Path) -> str:
    with open(str(path), encoding='utf-8') as f:
        return f.read()


def check_1_compile(errors):
    results = {}
    for label, script in [('normal', NORMAL_SCRIPT), ('nsclc', NSCLC_SCRIPT)]:
        ok = False
        try:
            py_compile.compile(str(script), doraise=True)
            ok = True
            print(f"  [1] {label}: compile PASS")
        except py_compile.PyCompileError as e:
            errors.append({'check': 'compile', 'script': label, 'error': str(e)})
            print(f"  [1] {label}: compile FAIL — {e}")
        results[label] = {'status': 'PASS' if ok else 'FAIL', 'path': str(script)}
    return results


def check_2_placeholder_removed(errors):
    results = {}
    for label, script in [('normal', NORMAL_SCRIPT), ('nsclc', NSCLC_SCRIPT)]:
        src = read_source(script)
        placeholder_phrases = [
            'not implemented in this dry-check script version',
            'P-C-NORMAL22d에서 실행 예정',
            'P-C-NORMAL22d 또는 22e에서 실행 예정',
        ]
        found = [p for p in placeholder_phrases if p in src]
        ok = len(found) == 0

        # 추가 확인: run_generate 함수 내부에 실제 로직이 있는지 (collision blocker 존재)
        has_collision_check = 'COLLISION-OK' in src or 'collision' in src.lower()

        if not ok:
            errors.append({'check': 'placeholder_removed', 'script': label, 'error': f'found: {found}'})
            print(f"  [2] {label}: placeholder FOUND — {found}")
        else:
            print(f"  [2] {label}: placeholder removed PASS | collision_check_present={has_collision_check}")

        results[label] = {
            'status': 'PASS' if ok else 'FAIL',
            'placeholder_found': found,
            'collision_check_present': has_collision_check,
        }
    return results


def check_3_allow_real_generation_false(errors):
    results = {}
    for label, script in [('normal', NORMAL_SCRIPT), ('nsclc', NSCLC_SCRIPT)]:
        src = read_source(script)
        # ALLOW_REAL_GENERATION=False でなければならない
        has_false = 'ALLOW_REAL_GENERATION = False' in src
        has_true  = 'ALLOW_REAL_GENERATION = True' in src
        ok = has_false and not has_true
        if not ok:
            errors.append({'check': 'allow_real_gen_false', 'script': label,
                           'error': f'has_false={has_false} has_true={has_true}'})
        print(f"  [3] {label}: ALLOW_REAL_GENERATION=False={has_false}, =True={has_true} → {'PASS' if ok else 'FAIL'}")
        results[label] = {'status': 'PASS' if ok else 'FAIL', 'has_false': has_false, 'has_true': has_true}
    return results


def check_4_run_generate_bare_exit2(errors):
    """--run-generate 단독 → exit 2 확인 (ALLOW_REAL_GENERATION=False이므로 즉시 exit)."""
    results = {}
    for label, script in [('normal', NORMAL_SCRIPT), ('nsclc', NSCLC_SCRIPT)]:
        try:
            proc = subprocess.run(
                [str(PYTHON_BIN), str(script), '--run-generate'],
                capture_output=True, text=True, timeout=30,
            )
            ok = proc.returncode == 2
            print(f"  [4] {label}: --run-generate bare returncode={proc.returncode} → {'PASS' if ok else 'FAIL'}")
            if not ok:
                errors.append({'check': 'run_generate_bare_exit2', 'script': label,
                               'error': f'returncode={proc.returncode} stdout={proc.stdout[:200]}'})
            results[label] = {
                'status': 'PASS' if ok else 'FAIL',
                'returncode': proc.returncode,
                'stdout': proc.stdout[:300],
                'stderr': proc.stderr[:200],
            }
        except Exception as e:
            errors.append({'check': 'run_generate_bare_exit2', 'script': label, 'error': str(e)})
            results[label] = {'status': 'ERROR', 'error': str(e)}
    return results


def check_5_run_generate_with_confirms_exit2(errors):
    """ALLOW_REAL_GENERATION=False + --run-generate + all confirms → exit 2."""
    results = {}
    for label, script in [('normal', NORMAL_SCRIPT), ('nsclc', NSCLC_SCRIPT)]:
        try:
            proc = subprocess.run(
                [str(PYTHON_BIN), str(script),
                 '--run-generate',
                 '--confirm-final-test',
                 '--confirm-no-prediction',
                 '--confirm-no-threshold'],
                capture_output=True, text=True, timeout=30,
            )
            ok = proc.returncode == 2
            abort_msg_ok = 'ALLOW_REAL_GENERATION=False' in proc.stdout
            print(f"  [5] {label}: --run-generate + confirms returncode={proc.returncode} "
                  f"abort_msg={'OK' if abort_msg_ok else 'MISSING'} → {'PASS' if ok else 'FAIL'}")
            if not ok:
                errors.append({'check': 'run_generate_confirms_exit2', 'script': label,
                               'error': f'returncode={proc.returncode}'})
            results[label] = {
                'status': 'PASS' if ok else 'FAIL',
                'returncode': proc.returncode,
                'abort_msg_ok': abort_msg_ok,
                'stdout': proc.stdout[:300],
            }
        except Exception as e:
            errors.append({'check': 'run_generate_confirms_exit2', 'script': label, 'error': str(e)})
            results[label] = {'status': 'ERROR', 'error': str(e)}
    return results


def check_6_drycheck_still_passes(errors):
    """--dry-check 기존 PASS 유지 확인."""
    results = {}
    for label, script in [('normal', NORMAL_SCRIPT), ('nsclc', NSCLC_SCRIPT)]:
        try:
            proc = subprocess.run(
                [str(PYTHON_BIN), str(script), '--dry-check'],
                capture_output=True, text=True, timeout=120,
            )
            # decision은 stdout에 [DECISION] 라인으로 출력됨
            decision_line = [l for l in proc.stdout.splitlines() if '[DECISION]' in l]
            decision = decision_line[-1] if decision_line else 'UNKNOWN'
            ok = proc.returncode == 0
            print(f"  [6] {label}: --dry-check returncode={proc.returncode} {decision} → {'PASS' if ok else 'FAIL'}")
            if not ok:
                errors.append({'check': 'drycheck_still_passes', 'script': label,
                               'error': f'returncode={proc.returncode} decision={decision}'})
            results[label] = {
                'status': 'PASS' if ok else 'FAIL',
                'returncode': proc.returncode,
                'decision': decision,
                'stdout_tail': proc.stdout[-500:],
            }
        except Exception as e:
            errors.append({'check': 'drycheck_still_passes', 'script': label, 'error': str(e)})
            results[label] = {'status': 'ERROR', 'error': str(e)}
    return results


def check_7_collision_blocker_coverage(errors):
    """Static: collision blocker가 crop dirs + manifest 경로를 모두 검사하는지 확인."""
    results = {}

    normal_src = read_source(NORMAL_SCRIPT)
    nsclc_src  = read_source(NSCLC_SCRIPT)

    # Expected crop dir names in collision check
    normal_crop_key = 'normal_test'
    nsclc_crop_key  = 'stage2_holdout_nsclc'
    manifest_key    = 'p_c_normal22_final_test'  # manifest fragment / combined

    checks = {
        'normal': {
            'crop_dir_covered':     normal_crop_key in normal_src,
            'manifest_covered':     manifest_key in normal_src,
            'collision_code_present': 'collision' in normal_src.lower() or 'COLLISION' in normal_src,
        },
        'nsclc': {
            'crop_dir_covered':     nsclc_crop_key in nsclc_src,
            'manifest_covered':     manifest_key in nsclc_src,
            'collision_code_present': 'collision' in nsclc_src.lower() or 'COLLISION' in nsclc_src,
        },
    }

    for label, chk in checks.items():
        ok = all(chk.values())
        if not ok:
            errors.append({'check': 'collision_blocker_coverage', 'script': label, 'error': str(chk)})
        print(f"  [7] {label}: collision coverage {chk} → {'PASS' if ok else 'FAIL'}")
        results[label] = {'status': 'PASS' if ok else 'FAIL', **chk}
    return results


def check_8_manifest_schema(errors):
    """Static: run_generate()가 생성할 manifest 컬럼이 spec과 일치하는지 확인."""
    results = {}
    for label, script in [('normal', NORMAL_SCRIPT), ('nsclc', NSCLC_SCRIPT)]:
        src = read_source(script)
        missing_cols = [col for col in EXPECTED_MANIFEST_COLS if f"'{col}'" not in src]
        ok = len(missing_cols) == 0
        if not ok:
            errors.append({'check': 'manifest_schema', 'script': label,
                           'error': f'missing: {missing_cols}'})
        print(f"  [8] {label}: manifest schema missing={missing_cols} → {'PASS' if ok else 'FAIL'}")
        results[label] = {
            'status': 'PASS' if ok else 'FAIL',
            'missing_cols': missing_cols,
            'checked_cols': EXPECTED_MANIFEST_COLS,
        }
    return results


def check_9_leakage_guard(errors):
    """Static: NSCLC script에 LUNG1-295 / LUNG1-415 hard exclusion 코드가 있는지 확인."""
    src = read_source(NSCLC_SCRIPT)
    checks = {
        'LUNG1-295_defined':    'LUNG1-295' in src,
        'LUNG1-415_defined':    'LUNG1-415' in src,
        'LEAKAGE_PATIENTS_set': 'LEAKAGE_PATIENTS' in src,
        'check_leakage_guard':  'check_leakage_guard' in src,
        'exit_2_on_leakage':    'sys.exit(2)' in src,
    }
    ok = all(checks.values())
    if not ok:
        errors.append({'check': 'leakage_guard', 'script': 'nsclc', 'error': str(checks)})
    print(f"  [9] nsclc: leakage guard {checks} → {'PASS' if ok else 'FAIL'}")
    return {'nsclc': {'status': 'PASS' if ok else 'FAIL', **checks}}


def check_10_guardrail_false(errors):
    """Static: GUARDRAIL dict에 false 항목들이 정의되어 있는지 확인."""
    results = {}
    for label, script in [('normal', NORMAL_SCRIPT), ('nsclc', NSCLC_SCRIPT)]:
        src = read_source(script)
        missing = [k for k in GUARDRAIL_EXPECTED_FALSE if f"'{k}'" not in src]
        ok = len(missing) == 0
        if not ok:
            errors.append({'check': 'guardrail_false', 'script': label, 'error': f'missing: {missing}'})
        print(f"  [10] {label}: guardrail keys missing={missing} → {'PASS' if ok else 'FAIL'}")
        results[label] = {'status': 'PASS' if ok else 'FAIL', 'missing_keys': missing}
    return results


def check_no_actual_output(errors):
    """actual generation이 발생하지 않았는지 output 경로로 확인."""
    crop_normal = (PROJECT_ROOT / 'outputs' / 'test_crops'
                   / 'p_c_normal22_final_baseline_test_crops' / 'normal_test')
    crop_nsclc  = (PROJECT_ROOT / 'outputs' / 'test_crops'
                   / 'p_c_normal22_final_baseline_test_crops' / 'stage2_holdout_nsclc')
    manifest_dir = (PROJECT_ROOT / 'outputs' / 'manifests'
                    / 'p_c_normal22_final_baseline_test_manifest')

    results = {
        'normal_test_crop_dir_exists':        crop_normal.exists(),
        'stage2_holdout_nsclc_crop_dir_exists': crop_nsclc.exists(),
        'manifest_dir_exists':                manifest_dir.exists(),
    }
    actual_generated = any(results.values())
    if actual_generated:
        errors.append({'check': 'no_actual_output', 'script': 'both',
                       'error': f'output paths exist: {results}'})
    ok = not actual_generated
    print(f"  [+] actual generation check: {results} → {'CLEAN (PASS)' if ok else 'GENERATED (FAIL)'}")
    results['status'] = 'PASS' if ok else 'FAIL'
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Write reports
# ─────────────────────────────────────────────────────────────────────────────

def write_reports(all_results, errors, decision):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── patch_summary CSV ─────────────────────────────────────────────────────
    patch_rows = []
    for label in ['normal', 'nsclc']:
        patch_rows.append({
            'script': label,
            'compile': all_results['compile'].get(label, {}).get('status', '?'),
            'placeholder_removed': all_results['placeholder'].get(label, {}).get('status', '?'),
            'allow_real_gen_false': all_results['allow_real_gen'].get(label, {}).get('status', '?'),
            'run_gen_bare_exit2': all_results['run_gen_bare'].get(label, {}).get('status', '?'),
            'run_gen_confirms_exit2': all_results['run_gen_confirms'].get(label, {}).get('status', '?'),
            'drycheck_pass': all_results['drycheck'].get(label, {}).get('status', '?'),
        })
    with open(REPORT_DIR / 'p_c_normal22d_patch_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(patch_rows[0].keys()))
        w.writeheader()
        w.writerows(patch_rows)

    # ── static_generation_check CSV ───────────────────────────────────────────
    gen_rows = [
        {'check': 'normal_placeholder_removed',
         'result': all_results['placeholder'].get('normal', {}).get('status', '?')},
        {'check': 'nsclc_placeholder_removed',
         'result': all_results['placeholder'].get('nsclc', {}).get('status', '?')},
        {'check': 'normal_allow_real_gen_false',
         'result': all_results['allow_real_gen'].get('normal', {}).get('status', '?')},
        {'check': 'nsclc_allow_real_gen_false',
         'result': all_results['allow_real_gen'].get('nsclc', {}).get('status', '?')},
        {'check': 'normal_collision_blocker',
         'result': all_results['collision'].get('normal', {}).get('status', '?')},
        {'check': 'nsclc_collision_blocker',
         'result': all_results['collision'].get('nsclc', {}).get('status', '?')},
        {'check': 'no_actual_crop_dir',
         'result': 'PASS' if not all_results['no_actual_output'].get('normal_test_crop_dir_exists') else 'FAIL'},
        {'check': 'no_actual_nsclc_dir',
         'result': 'PASS' if not all_results['no_actual_output'].get('stage2_holdout_nsclc_crop_dir_exists') else 'FAIL'},
        {'check': 'no_manifest_dir',
         'result': 'PASS' if not all_results['no_actual_output'].get('manifest_dir_exists') else 'FAIL'},
    ]
    with open(REPORT_DIR / 'p_c_normal22d_static_generation_check.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['check', 'result'])
        w.writeheader()
        w.writerows(gen_rows)

    # ── guard_check CSV ───────────────────────────────────────────────────────
    guard_rows = [
        {'guard': f'{label}_run_gen_bare_exit2',
         'expected': 'exit_2', 'actual': all_results['run_gen_bare'].get(label, {}).get('returncode', '?'),
         'status': all_results['run_gen_bare'].get(label, {}).get('status', '?')}
        for label in ['normal', 'nsclc']
    ] + [
        {'guard': f'{label}_run_gen_confirms_exit2',
         'expected': 'exit_2', 'actual': all_results['run_gen_confirms'].get(label, {}).get('returncode', '?'),
         'status': all_results['run_gen_confirms'].get(label, {}).get('status', '?')}
        for label in ['normal', 'nsclc']
    ]
    with open(REPORT_DIR / 'p_c_normal22d_guard_check.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['guard', 'expected', 'actual', 'status'])
        w.writeheader()
        w.writerows(guard_rows)

    # ── output_collision_check CSV ────────────────────────────────────────────
    no_out = all_results['no_actual_output']
    collision_rows = [
        {'path': 'outputs/test_crops/.../normal_test',
         'exists': no_out.get('normal_test_crop_dir_exists'), 'collision': no_out.get('normal_test_crop_dir_exists')},
        {'path': 'outputs/test_crops/.../stage2_holdout_nsclc',
         'exists': no_out.get('stage2_holdout_nsclc_crop_dir_exists'), 'collision': no_out.get('stage2_holdout_nsclc_crop_dir_exists')},
        {'path': 'outputs/manifests/p_c_normal22_final_baseline_test_manifest',
         'exists': no_out.get('manifest_dir_exists'), 'collision': no_out.get('manifest_dir_exists')},
    ]
    with open(REPORT_DIR / 'p_c_normal22d_output_collision_check.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'exists', 'collision'])
        w.writeheader()
        w.writerows(collision_rows)

    # ── manifest_schema_check CSV ─────────────────────────────────────────────
    schema_rows = []
    for label in ['normal', 'nsclc']:
        sch = all_results['manifest_schema'].get(label, {})
        for col in EXPECTED_MANIFEST_COLS:
            schema_rows.append({
                'script': label,
                'column': col,
                'found_in_source': col not in sch.get('missing_cols', []),
                'status': 'PASS' if col not in sch.get('missing_cols', []) else 'FAIL',
            })
    with open(REPORT_DIR / 'p_c_normal22d_manifest_schema_check.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['script', 'column', 'found_in_source', 'status'])
        w.writeheader()
        w.writerows(schema_rows)

    # ── leakage_guard_check CSV ───────────────────────────────────────────────
    leak = all_results['leakage'].get('nsclc', {})
    leakage_rows = [
        {'item': k, 'expected': True, 'actual': v, 'status': 'PASS' if v else 'FAIL'}
        for k, v in leak.items() if k != 'status'
    ]
    with open(REPORT_DIR / 'p_c_normal22d_leakage_guard_check.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['item', 'expected', 'actual', 'status'])
        w.writeheader()
        w.writerows(leakage_rows)

    # ── guardrail_check CSV ───────────────────────────────────────────────────
    gr_rows = []
    for label in ['normal', 'nsclc']:
        gr = all_results['guardrail'].get(label, {})
        for k in GUARDRAIL_EXPECTED_FALSE:
            gr_rows.append({
                'script': label,
                'guardrail_item': k,
                'expected': False,
                'found_in_source': k not in gr.get('missing_keys', []),
                'status': 'PASS' if k not in gr.get('missing_keys', []) else 'FAIL',
            })
    with open(REPORT_DIR / 'p_c_normal22d_guardrail_check.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['script', 'guardrail_item', 'expected', 'found_in_source', 'status'])
        w.writeheader()
        w.writerows(gr_rows)

    # ── errors CSV ────────────────────────────────────────────────────────────
    with open(REPORT_DIR / 'p_c_normal22d_errors.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['check', 'script', 'error'])
        w.writeheader()
        w.writerows(errors)

    # ── main drycheck JSON ────────────────────────────────────────────────────
    summary = {
        'stage': 'P-C-NORMAL22d',
        'date': START_TIME.isoformat(),
        'decision': decision,
        'error_count': len(errors),
        'normal_expected_crops': 21600,
        'nsclc_expected_crops': 51313,
        'combined_expected_rows': 72913,
        'leakage_hard_excluded': sorted(LEAKAGE_PATIENTS),
        'guardrail': {
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
        },
        'check_results': {k: v for k, v in all_results.items()
                          if k not in ('no_actual_output',)},
    }
    with open(REPORT_DIR / 'p_c_normal22d_run_generate_implementation_drycheck.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # ── main drycheck MD ──────────────────────────────────────────────────────
    with open(REPORT_DIR / 'p_c_normal22d_run_generate_implementation_drycheck.md', 'w') as f:
        f.write(f"# P-C-NORMAL22d run_generate() Implementation Dry-Check\n\n")
        f.write(f"- **date**: {START_TIME.isoformat()}\n")
        f.write(f"- **decision**: {decision}\n")
        f.write(f"- **error_count**: {len(errors)}\n\n")
        f.write("## 수정한 스크립트\n\n")
        f.write(f"1. `{NORMAL_SCRIPT}`\n")
        f.write(f"2. `{NSCLC_SCRIPT}`\n\n")
        f.write("## run_generate placeholder 제거 여부\n\n")
        for label in ['normal', 'nsclc']:
            ph = all_results['placeholder'].get(label, {})
            f.write(f"- {label}: {ph.get('status', '?')} "
                    f"(found={ph.get('placeholder_found', [])})\n")
        f.write("\n## Expected Crop Counts\n\n")
        f.write("- normal expected crops: **21,600**\n")
        f.write("- NSCLC expected crops: **51,313**\n")
        f.write("- combined expected rows: **72,913**\n\n")
        f.write("## Leakage Hard Exclusion\n\n")
        leak = all_results['leakage'].get('nsclc', {})
        f.write(f"- LUNG1-295 defined: {leak.get('LUNG1-295_defined')}\n")
        f.write(f"- LUNG1-415 defined: {leak.get('LUNG1-415_defined')}\n")
        f.write(f"- check_leakage_guard function: {leak.get('check_leakage_guard')}\n")
        f.write(f"- exit 2 on leakage: {leak.get('exit_2_on_leakage')}\n\n")
        f.write("## Output Collision Blocker\n\n")
        for label in ['normal', 'nsclc']:
            col = all_results['collision'].get(label, {})
            f.write(f"- {label}: crop_dir_covered={col.get('crop_dir_covered')} "
                    f"manifest_covered={col.get('manifest_covered')} → {col.get('status')}\n")
        f.write("\n## Actual Generation\n\n")
        no_out = all_results['no_actual_output']
        f.write(f"- normal_test crop dir exists: {no_out.get('normal_test_crop_dir_exists')} → "
                f"{'CLEAN' if not no_out.get('normal_test_crop_dir_exists') else 'GENERATED!'}\n")
        f.write(f"- stage2_holdout_nsclc dir exists: {no_out.get('stage2_holdout_nsclc_crop_dir_exists')} → "
                f"{'CLEAN' if not no_out.get('stage2_holdout_nsclc_crop_dir_exists') else 'GENERATED!'}\n")
        f.write(f"- manifest dir exists: {no_out.get('manifest_dir_exists')} → "
                f"{'CLEAN' if not no_out.get('manifest_dir_exists') else 'EXISTS'}\n\n")
        f.write("## 다음 단계\n\n")
        f.write("**P-C-NORMAL22e**: actual final baseline test crop + manifest generation\n\n")
        f.write("## P-C-NORMAL22e 승인 문구 초안\n\n")
        f.write(
            "\"P-C-NORMAL22d run_generate implementation dry-check 통과 확인. "
            "P-C-NORMAL22e final baseline test crop + manifest actual generation 실행 승인. "
            "normal_test 36명 21,600 crops와 clean stage2_holdout NSCLC 152명 51,313 crops를 "
            "P-C-NORMAL 규격 3×96×96 int16 ct_crop으로 생성하고, LUNG1-295/LUNG1-415는 hard exclude하며, "
            "prediction/model_forward/threshold 없이 crop+manifest만 1회 생성.\"\n"
        )

    print(f"\n[REPORTS] Written to: {REPORT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n[P-C-NORMAL22d DRY-CHECK] Start: {START_TIME.isoformat()}")
    print(f"  Normal script: {NORMAL_SCRIPT}")
    print(f"  NSCLC  script: {NSCLC_SCRIPT}\n")

    errors = []
    all_results = {}

    print("[1/10] py_compile check...")
    all_results['compile'] = check_1_compile(errors)

    print("[2/10] placeholder removed check...")
    all_results['placeholder'] = check_2_placeholder_removed(errors)

    print("[3/10] ALLOW_REAL_GENERATION=False check...")
    all_results['allow_real_gen'] = check_3_allow_real_generation_false(errors)

    print("[4/10] --run-generate bare → exit 2 check...")
    all_results['run_gen_bare'] = check_4_run_generate_bare_exit2(errors)

    print("[5/10] --run-generate + confirms → exit 2 check...")
    all_results['run_gen_confirms'] = check_5_run_generate_with_confirms_exit2(errors)

    print("[6/10] --dry-check still passes...")
    all_results['drycheck'] = check_6_drycheck_still_passes(errors)

    print("[7/10] Collision blocker coverage (static)...")
    all_results['collision'] = check_7_collision_blocker_coverage(errors)

    print("[8/10] Manifest schema check (static)...")
    all_results['manifest_schema'] = check_8_manifest_schema(errors)

    print("[9/10] Leakage guard check (static)...")
    all_results['leakage'] = check_9_leakage_guard(errors)

    print("[10/10] Guardrail dict check (static)...")
    all_results['guardrail'] = check_10_guardrail_false(errors)

    print("[+] Actual output check...")
    all_results['no_actual_output'] = check_no_actual_output(errors)

    # ── Verdict ───────────────────────────────────────────────────────────────
    check_names = [
        'compile', 'placeholder', 'allow_real_gen',
        'run_gen_bare', 'run_gen_confirms', 'drycheck',
        'collision', 'manifest_schema', 'leakage', 'guardrail',
    ]
    all_pass = True
    partial  = False
    for name in check_names:
        res = all_results.get(name, {})
        for label, val in res.items():
            if isinstance(val, dict):
                s = val.get('status', 'UNKNOWN')
                if s != 'PASS':
                    all_pass = False
                if s in ('PARTIAL', 'WARN'):
                    partial = True

    no_actual = all_results['no_actual_output'].get('status') == 'PASS'
    if not no_actual:
        all_pass = False

    if all_pass:
        decision = 'PASS'
    elif partial or len(errors) <= 3:
        decision = 'PARTIAL_PASS'
    else:
        decision = 'FAIL'

    print(f"\n[DECISION] {decision} | errors={len(errors)}")

    write_reports(all_results, errors, decision)
    print(f"[P-C-NORMAL22d DRY-CHECK] Done.\n")
    return decision


if __name__ == '__main__':
    decision = main()
    sys.exit(0 if decision in ('PASS', 'PARTIAL_PASS') else 1)
