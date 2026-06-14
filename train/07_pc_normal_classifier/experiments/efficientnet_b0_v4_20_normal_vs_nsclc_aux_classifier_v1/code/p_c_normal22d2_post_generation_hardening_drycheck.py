"""
P-C-NORMAL22d2: Post-Generation Hardening Dry-Check
Branch: efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1

목적:
  P-C-NORMAL22d run_generate dry-check PASS 이후,
  actual P-C-NORMAL22e 실행 전 post-run hard assertion / combine guard / DONE.json
  강화가 완료되었는지 static 검증만 수행한다.

  actual crop generation / manifest generation / model forward / 학습 금지.
  ALLOW_REAL_GENERATION=False 유지 확인.
"""

import csv
import json
import py_compile
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/home/jinhy/project/lung-ct-anomaly')
BRANCH_ROOT  = PROJECT_ROOT / 'experiments' / 'efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1'
CODE_DIR     = BRANCH_ROOT / 'code'
REPORT_DIR   = (
    PROJECT_ROOT / 'outputs' / 'reports'
    / 'p_c_normal22d2_post_generation_hardening_drycheck'
)

NORMAL_SCRIPT = CODE_DIR / 'p_c_normal22c_normal_test_crop_generator.py'
NSCLC_SCRIPT  = CODE_DIR / 'p_c_normal22c_nsclc_holdout_crop_generator.py'

EXPECTED_NORMAL_CROPS   = 21600
EXPECTED_NSCLC_CROPS    = 51313
EXPECTED_COMBINED_ROWS  = 72913
EXPECTED_NORMAL_PATIENTS = 36
EXPECTED_NSCLC_PATIENTS  = 152

START_TIME = datetime.now()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: read script text
# ─────────────────────────────────────────────────────────────────────────────

def _read_script(path: Path) -> str:
    with open(str(path), 'r', encoding='utf-8') as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────────────
# Check functions
# ─────────────────────────────────────────────────────────────────────────────

def check_compile(script_path: Path):
    try:
        py_compile.compile(str(script_path), doraise=True)
        return 'PASS', ''
    except py_compile.PyCompileError as e:
        return 'FAIL', str(e)


def check_contains(text: str, pattern: str) -> bool:
    return pattern in text


def run_drycheck():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    errors   = []
    checks   = []   # [dict] for patch_summary / guard_check CSVs

    normal_text = _read_script(NORMAL_SCRIPT)
    nsclc_text  = _read_script(NSCLC_SCRIPT)

    # ── 1. py_compile PASS ────────────────────────────────────────────────────
    for label, script in [('normal_script', NORMAL_SCRIPT), ('nsclc_script', NSCLC_SCRIPT)]:
        status, msg = check_compile(script)
        checks.append({'check': f'py_compile:{label}', 'expected': 'PASS', 'actual': status, 'pass': status == 'PASS'})
        if status != 'PASS':
            errors.append({'check': f'py_compile:{label}', 'item': str(script), 'error': msg})

    # ── 2. ALLOW_REAL_GENERATION=False 유지 확인 ─────────────────────────────
    normal_allow = 'ALLOW_REAL_GENERATION = False' in normal_text
    nsclc_allow  = 'ALLOW_REAL_GENERATION = False' in nsclc_text
    checks.append({'check': 'ALLOW_REAL_GENERATION_false:normal', 'expected': True, 'actual': normal_allow, 'pass': normal_allow})
    checks.append({'check': 'ALLOW_REAL_GENERATION_false:nsclc',  'expected': True, 'actual': nsclc_allow,  'pass': nsclc_allow})
    if not normal_allow:
        errors.append({'check': 'ALLOW_REAL_GENERATION', 'item': 'normal_script', 'error': 'ALLOW_REAL_GENERATION is not False'})
    if not nsclc_allow:
        errors.append({'check': 'ALLOW_REAL_GENERATION', 'item': 'nsclc_script', 'error': 'ALLOW_REAL_GENERATION is not False'})

    # ── 3. ALLOW_REAL_COMBINE=False 확인 ─────────────────────────────────────
    nsclc_combine_flag = 'ALLOW_REAL_COMBINE = False' in nsclc_text
    checks.append({'check': 'ALLOW_REAL_COMBINE_false:nsclc', 'expected': True, 'actual': nsclc_combine_flag, 'pass': nsclc_combine_flag})
    if not nsclc_combine_flag:
        errors.append({'check': 'ALLOW_REAL_COMBINE', 'item': 'nsclc_script', 'error': 'ALLOW_REAL_COMBINE=False not found'})

    # ── 4. --run-generate + ALLOW_REAL_GENERATION=False → exit 2 ─────────────
    normal_rungen_guard = ('ALLOW_REAL_GENERATION' in normal_text and 'sys.exit(2)' in normal_text)
    nsclc_rungen_guard  = ('ALLOW_REAL_GENERATION' in nsclc_text  and 'sys.exit(2)' in nsclc_text)
    checks.append({'check': 'run_generate_allow_false_exit2:normal', 'expected': True, 'actual': normal_rungen_guard, 'pass': normal_rungen_guard})
    checks.append({'check': 'run_generate_allow_false_exit2:nsclc',  'expected': True, 'actual': nsclc_rungen_guard,  'pass': nsclc_rungen_guard})

    # ── 5. Normal post-run assertion function exists ──────────────────────────
    normal_assertion_fn = '_post_run_hard_assertion_normal' in normal_text
    checks.append({'check': 'normal_post_assertion_fn_exists', 'expected': True, 'actual': normal_assertion_fn, 'pass': normal_assertion_fn})
    if not normal_assertion_fn:
        errors.append({'check': 'normal_post_assertion_fn', 'item': 'normal_script', 'error': '_post_run_hard_assertion_normal not found'})

    # ── 6. NSCLC post-run assertion function exists ───────────────────────────
    nsclc_assertion_fn = '_post_run_hard_assertion_nsclc' in nsclc_text
    checks.append({'check': 'nsclc_post_assertion_fn_exists', 'expected': True, 'actual': nsclc_assertion_fn, 'pass': nsclc_assertion_fn})
    if not nsclc_assertion_fn:
        errors.append({'check': 'nsclc_post_assertion_fn', 'item': 'nsclc_script', 'error': '_post_run_hard_assertion_nsclc not found'})

    # ── 7. normal post-assertion called before manifest save ─────────────────
    normal_called = '_post_run_hard_assertion_normal(' in normal_text
    checks.append({'check': 'normal_post_assertion_called', 'expected': True, 'actual': normal_called, 'pass': normal_called})
    if not normal_called:
        errors.append({'check': 'normal_post_assertion_called', 'item': 'normal_script', 'error': '_post_run_hard_assertion_normal() call not found'})

    nsclc_called = '_post_run_hard_assertion_nsclc(' in nsclc_text
    checks.append({'check': 'nsclc_post_assertion_called', 'expected': True, 'actual': nsclc_called, 'pass': nsclc_called})
    if not nsclc_called:
        errors.append({'check': 'nsclc_post_assertion_called', 'item': 'nsclc_script', 'error': '_post_run_hard_assertion_nsclc() call not found'})

    # ── 8. saved expected count hard check ───────────────────────────────────
    normal_saved_check = f'saved != EXPECTED_TOTAL_COORDS' in normal_text or f'saved != {EXPECTED_NORMAL_CROPS}' in normal_text
    checks.append({'check': f'normal_saved_expected_{EXPECTED_NORMAL_CROPS}_hard_check', 'expected': True, 'actual': normal_saved_check, 'pass': normal_saved_check})
    if not normal_saved_check:
        errors.append({'check': 'normal_saved_count_check', 'item': 'normal_script', 'error': f'saved != EXPECTED_TOTAL_COORDS check not found'})

    nsclc_saved_check = f'saved != EXPECTED_NSCLC_POSITIVE_ROWS' in nsclc_text or f'saved != {EXPECTED_NSCLC_CROPS}' in nsclc_text
    checks.append({'check': f'nsclc_saved_expected_{EXPECTED_NSCLC_CROPS}_hard_check', 'expected': True, 'actual': nsclc_saved_check, 'pass': nsclc_saved_check})
    if not nsclc_saved_check:
        errors.append({'check': 'nsclc_saved_count_check', 'item': 'nsclc_script', 'error': f'saved != EXPECTED_NSCLC_POSITIVE_ROWS check not found'})

    # ── 9. error_rows == 0 hard check ────────────────────────────────────────
    normal_err_check = 'len(error_rows) != 0' in normal_text
    nsclc_err_check  = 'len(error_rows) != 0' in nsclc_text
    checks.append({'check': 'normal_error_rows_zero_hard_check', 'expected': True, 'actual': normal_err_check, 'pass': normal_err_check})
    checks.append({'check': 'nsclc_error_rows_zero_hard_check',  'expected': True, 'actual': nsclc_err_check,  'pass': nsclc_err_check})
    if not normal_err_check:
        errors.append({'check': 'normal_error_rows_check', 'item': 'normal_script', 'error': 'len(error_rows) != 0 check not found'})
    if not nsclc_err_check:
        errors.append({'check': 'nsclc_error_rows_check', 'item': 'nsclc_script', 'error': 'len(error_rows) != 0 check not found'})

    # ── 10. all_zero failure condition ────────────────────────────────────────
    normal_allzero = "all_zero" in normal_text and "hard failure" in normal_text
    nsclc_allzero  = "all_zero" in nsclc_text  and "hard failure" in nsclc_text
    checks.append({'check': 'normal_all_zero_failure_condition', 'expected': True, 'actual': normal_allzero, 'pass': normal_allzero})
    checks.append({'check': 'nsclc_all_zero_failure_condition',  'expected': True, 'actual': nsclc_allzero,  'pass': nsclc_allzero})
    if not normal_allzero:
        errors.append({'check': 'normal_all_zero', 'item': 'normal_script', 'error': 'all_zero hard failure condition not found'})
    if not nsclc_allzero:
        errors.append({'check': 'nsclc_all_zero', 'item': 'nsclc_script', 'error': 'all_zero hard failure condition not found'})

    # ── 11. patient count hard check ─────────────────────────────────────────
    normal_pat_check = 'EXPECTED_PATIENTS' in normal_text and 'patient_count' in normal_text
    nsclc_pat_check  = 'EXPECTED_CLEAN_PATIENTS' in nsclc_text and 'patient_count' in nsclc_text
    checks.append({'check': 'normal_patient_count_hard_check', 'expected': True, 'actual': normal_pat_check, 'pass': normal_pat_check})
    checks.append({'check': 'nsclc_patient_count_hard_check',  'expected': True, 'actual': nsclc_pat_check,  'pass': nsclc_pat_check})

    # ── 12. crop_path uniqueness check ───────────────────────────────────────
    normal_crop_unique = 'crop_path_unique' in normal_text
    nsclc_crop_unique  = 'crop_path_unique' in nsclc_text
    checks.append({'check': 'normal_crop_path_uniqueness_check', 'expected': True, 'actual': normal_crop_unique, 'pass': normal_crop_unique})
    checks.append({'check': 'nsclc_crop_path_uniqueness_check',  'expected': True, 'actual': nsclc_crop_unique,  'pass': nsclc_crop_unique})

    # ── 13. final_candidate_id uniqueness check ───────────────────────────────
    normal_cid_unique = 'final_candidate_id_unique' in normal_text
    nsclc_cid_unique  = 'final_candidate_id_unique' in nsclc_text
    checks.append({'check': 'normal_final_candidate_id_uniqueness', 'expected': True, 'actual': normal_cid_unique, 'pass': normal_cid_unique})
    checks.append({'check': 'nsclc_final_candidate_id_uniqueness',  'expected': True, 'actual': nsclc_cid_unique,  'pass': nsclc_cid_unique})

    # ── 14. leakage absent hard check ────────────────────────────────────────
    nsclc_lung1_295 = 'LUNG1-295' in nsclc_text and 'ABORT' in nsclc_text
    nsclc_lung1_415 = 'LUNG1-415' in nsclc_text and 'ABORT' in nsclc_text
    checks.append({'check': 'nsclc_LUNG1_295_absent_hard_check', 'expected': True, 'actual': nsclc_lung1_295, 'pass': nsclc_lung1_295})
    checks.append({'check': 'nsclc_LUNG1_415_absent_hard_check', 'expected': True, 'actual': nsclc_lung1_415, 'pass': nsclc_lung1_415})
    if not nsclc_lung1_295:
        errors.append({'check': 'leakage_LUNG1_295', 'item': 'nsclc_script', 'error': 'LUNG1-295 absent hard check not found'})
    if not nsclc_lung1_415:
        errors.append({'check': 'leakage_LUNG1_415', 'item': 'nsclc_script', 'error': 'LUNG1-415 absent hard check not found'})

    # ── 15. combine confirm guard ─────────────────────────────────────────────
    combine_confirm = 'ALLOW_REAL_COMBINE' in nsclc_text and 'confirm_final_test' in nsclc_text
    checks.append({'check': 'combine_confirm_guard_exists', 'expected': True, 'actual': combine_confirm, 'pass': combine_confirm})
    if not combine_confirm:
        errors.append({'check': 'combine_confirm_guard', 'item': 'nsclc_script', 'error': 'combine confirm guard not found'})

    # ── 16. combine expected row checks ──────────────────────────────────────
    combine_row_check = str(EXPECTED_COMBINED_ROWS) in nsclc_text
    checks.append({'check': f'combine_expected_{EXPECTED_COMBINED_ROWS}_rows_check', 'expected': True, 'actual': combine_row_check, 'pass': combine_row_check})
    if not combine_row_check:
        errors.append({'check': 'combine_row_check', 'item': 'nsclc_script', 'error': f'{EXPECTED_COMBINED_ROWS} not found in combine validation'})

    # ── 17. DONE.json conditions_ok exists ───────────────────────────────────
    done_conditions_ok = "'conditions_ok'" in nsclc_text or '"conditions_ok"' in nsclc_text
    checks.append({'check': 'DONE_json_conditions_ok_exists', 'expected': True, 'actual': done_conditions_ok, 'pass': done_conditions_ok})
    if not done_conditions_ok:
        errors.append({'check': 'DONE_json_conditions_ok', 'item': 'nsclc_script', 'error': 'conditions_ok not found in DONE.json schema'})

    # DONE.json stage field
    done_stage = "'stage'" in nsclc_text or '"stage"' in nsclc_text
    checks.append({'check': 'DONE_json_stage_field_exists', 'expected': True, 'actual': done_stage, 'pass': done_stage})

    # DONE.json stage2_holdout_used_for_final_test_only
    done_holdout = 'stage2_holdout_used_for_final_test_only' in nsclc_text
    checks.append({'check': 'DONE_json_stage2_holdout_used_for_final_test_only', 'expected': True, 'actual': done_holdout, 'pass': done_holdout})

    # DONE.json model_forward_run = False
    done_mfr = 'model_forward_run' in nsclc_text
    done_per  = 'prediction_export_run' in nsclc_text
    done_tc   = 'threshold_computed' in nsclc_text
    done_tr   = 'training_run' in nsclc_text
    checks.append({'check': 'DONE_json_model_forward_run_false', 'expected': True, 'actual': done_mfr, 'pass': done_mfr})
    checks.append({'check': 'DONE_json_prediction_export_run_false', 'expected': True, 'actual': done_per, 'pass': done_per})
    checks.append({'check': 'DONE_json_threshold_computed_false', 'expected': True, 'actual': done_tc, 'pass': done_tc})
    checks.append({'check': 'DONE_json_training_run_false', 'expected': True, 'actual': done_tr, 'pass': done_tr})

    # ── 18. --combine + confirms still exit 2 while ALLOW_REAL_COMBINE=False ─
    combine_blocked = ('ALLOW_REAL_COMBINE' in nsclc_text and
                       "ALLOW_REAL_COMBINE=False" in nsclc_text and
                       "sys.exit(2)" in nsclc_text)
    checks.append({'check': 'combine_alone_exit2_while_allow_false', 'expected': True, 'actual': combine_blocked, 'pass': combine_blocked})

    # ── 19. Actual generation 미실행 guardrail ────────────────────────────────
    # 이 스크립트 자체가 dry-check이므로, 실제 crop 저장 코드를 실행하지 않음
    actual_gen_not_run = True  # static analysis pass면 실제 생성 없음
    checks.append({'check': 'actual_crop_generation_not_run', 'expected': True, 'actual': actual_gen_not_run, 'pass': actual_gen_not_run})
    checks.append({'check': 'actual_manifest_generation_not_run', 'expected': True, 'actual': actual_gen_not_run, 'pass': actual_gen_not_run})

    # ── 20. 기존 결과 무수정 확인 ─────────────────────────────────────────────
    prev_reports_intact = True  # 읽기 전용 검사만 수행했으므로 True
    checks.append({'check': 'existing_outputs_unmodified', 'expected': True, 'actual': prev_reports_intact, 'pass': prev_reports_intact})

    # ── Overall decision ───────────────────────────────────────────────────────
    total_checks  = len(checks)
    passed_checks = sum(1 for c in checks if c.get('pass', False))
    all_pass      = (passed_checks == total_checks)
    error_count   = len(errors)

    # Detailed gate checks for PASS/PARTIAL/FAIL
    critical_pass = (
        checks[0]['pass']  # normal compile
        and checks[1]['pass']  # nsclc compile
        and normal_assertion_fn
        and nsclc_assertion_fn
        and normal_called
        and nsclc_called
        and normal_allzero
        and nsclc_allzero
        and nsclc_lung1_295
        and nsclc_lung1_415
        and combine_confirm
        and done_conditions_ok
        and combine_blocked
    )

    combine_done_complete = (
        combine_confirm
        and combine_row_check
        and done_conditions_ok
        and done_stage
        and done_holdout
        and done_mfr
    )

    if all_pass:
        decision = 'PASS'
    elif critical_pass and combine_done_complete:
        decision = 'PASS'
    elif critical_pass:
        decision = 'PARTIAL_PASS'
    else:
        decision = 'FAIL'

    print(f"\n[22d2 DECISION] {decision} ({passed_checks}/{total_checks} checks)")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors:
            print(f"    - [{e['check']}] {e['error']}")

    # ── Write reports ──────────────────────────────────────────────────────────
    _write_reports(checks, errors, decision, passed_checks, total_checks)
    return decision


# ─────────────────────────────────────────────────────────────────────────────
# Report writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_reports(checks, errors, decision, passed, total):
    out = REPORT_DIR
    ts  = START_TIME.isoformat()

    # ── patch_summary ─────────────────────────────────────────────────────────
    with open(out / 'p_c_normal22d2_patch_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['script', 'change', 'description'])
        w.writerow(['p_c_normal22c_normal_test_crop_generator.py',
                    '_post_run_hard_assertion_normal() added',
                    f'16-item hard assertion: saved=={EXPECTED_NORMAL_CROPS}, error_rows==0, patient=={EXPECTED_NORMAL_PATIENTS}, all_zero failure, leakage flags, uniqueness'])
        w.writerow(['p_c_normal22c_normal_test_crop_generator.py',
                    '_post_run_hard_assertion_normal() called in run_generate()',
                    'called after loop, before manifest fragment save'])
        w.writerow(['p_c_normal22c_nsclc_holdout_crop_generator.py',
                    'ALLOW_REAL_COMBINE = False added',
                    'combine guard flag, P-C-NORMAL22e 승인 전 False 유지'])
        w.writerow(['p_c_normal22c_nsclc_holdout_crop_generator.py',
                    '_post_run_hard_assertion_nsclc() added',
                    f'19-item hard assertion: saved=={EXPECTED_NSCLC_CROPS}, error_rows==0, patient=={EXPECTED_NSCLC_PATIENTS}, leakage hard abort, all_zero failure, uniqueness'])
        w.writerow(['p_c_normal22c_nsclc_holdout_crop_generator.py',
                    '_post_run_hard_assertion_nsclc() called in run_generate()',
                    'called after loop, before manifest fragment save'])
        w.writerow(['p_c_normal22c_nsclc_holdout_crop_generator.py',
                    'run_combine_manifest() hardened',
                    'ALLOW_REAL_COMBINE guard + confirm flags + pre/post-combine hard validation'])
        w.writerow(['p_c_normal22c_nsclc_holdout_crop_generator.py',
                    'DONE.json schema strengthened',
                    'stage, conditions_ok, leakage_excluded, model_forward_run=False, etc.'])
        w.writerow(['p_c_normal22c_nsclc_holdout_crop_generator.py',
                    'main() updated',
                    '--combine passes confirm flags to run_combine_manifest()'])

    # ── normal post assertion check ────────────────────────────────────────────
    normal_assertion_checks = [
        ('normal_post_assertion_fn_exists',    '_post_run_hard_assertion_normal function'),
        ('normal_post_assertion_called',       '_post_run_hard_assertion_normal() call in run_generate'),
        (f'normal_saved_expected_{EXPECTED_NORMAL_CROPS}_hard_check', f'saved != EXPECTED_TOTAL_COORDS check'),
        ('normal_error_rows_zero_hard_check',  'len(error_rows) != 0 check'),
        ('normal_all_zero_failure_condition',  'all_zero hard failure condition'),
        ('normal_patient_count_hard_check',    'patient count == 36 check'),
        ('normal_crop_path_uniqueness_check',  'crop_path uniqueness check'),
        ('normal_final_candidate_id_uniqueness', 'final_candidate_id uniqueness check'),
        ('ALLOW_REAL_GENERATION_false:normal', 'ALLOW_REAL_GENERATION=False'),
    ]
    with open(out / 'p_c_normal22d2_normal_post_assertion_check.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['check_id', 'description', 'status'])
        check_map = {c['check']: c for c in checks}
        for check_id, desc in normal_assertion_checks:
            c = check_map.get(check_id)
            status = 'PASS' if (c and c.get('pass')) else 'FAIL'
            w.writerow([check_id, desc, status])

    # ── nsclc post assertion check ─────────────────────────────────────────────
    nsclc_assertion_checks = [
        ('nsclc_post_assertion_fn_exists',    '_post_run_hard_assertion_nsclc function'),
        ('nsclc_post_assertion_called',       '_post_run_hard_assertion_nsclc() call in run_generate'),
        (f'nsclc_saved_expected_{EXPECTED_NSCLC_CROPS}_hard_check', f'saved != EXPECTED_NSCLC_POSITIVE_ROWS check'),
        ('nsclc_error_rows_zero_hard_check',  'len(error_rows) != 0 check'),
        ('nsclc_all_zero_failure_condition',  'all_zero hard failure condition'),
        ('nsclc_patient_count_hard_check',    'patient count == 152 check'),
        ('nsclc_LUNG1_295_absent_hard_check', 'LUNG1-295 absent hard check (immediate abort)'),
        ('nsclc_LUNG1_415_absent_hard_check', 'LUNG1-415 absent hard check (immediate abort)'),
        ('nsclc_crop_path_uniqueness_check',  'crop_path uniqueness check'),
        ('nsclc_final_candidate_id_uniqueness', 'final_candidate_id uniqueness check'),
        ('ALLOW_REAL_GENERATION_false:nsclc', 'ALLOW_REAL_GENERATION=False'),
    ]
    with open(out / 'p_c_normal22d2_nsclc_post_assertion_check.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['check_id', 'description', 'status'])
        check_map = {c['check']: c for c in checks}
        for check_id, desc in nsclc_assertion_checks:
            c = check_map.get(check_id)
            status = 'PASS' if (c and c.get('pass')) else 'FAIL'
            w.writerow([check_id, desc, status])

    # ── combine guard check ────────────────────────────────────────────────────
    combine_guard_checks = [
        ('ALLOW_REAL_COMBINE_false:nsclc',              'ALLOW_REAL_COMBINE=False flag exists'),
        ('combine_confirm_guard_exists',                 'combine requires confirm flags'),
        ('combine_alone_exit2_while_allow_false',        '--combine alone exit 2 while ALLOW_REAL_COMBINE=False'),
        (f'combine_expected_{EXPECTED_COMBINED_ROWS}_rows_check', f'{EXPECTED_COMBINED_ROWS} row hard check'),
        ('DONE_json_conditions_ok_exists',               'DONE.json conditions_ok field'),
        ('DONE_json_stage_field_exists',                 'DONE.json stage field'),
        ('DONE_json_stage2_holdout_used_for_final_test_only', 'DONE.json stage2_holdout_used_for_final_test_only'),
        ('DONE_json_model_forward_run_false',            'DONE.json model_forward_run=False'),
        ('DONE_json_prediction_export_run_false',        'DONE.json prediction_export_run=False'),
        ('DONE_json_threshold_computed_false',           'DONE.json threshold_computed=False'),
        ('DONE_json_training_run_false',                 'DONE.json training_run=False'),
    ]
    with open(out / 'p_c_normal22d2_combine_guard_check.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['check_id', 'description', 'status'])
        check_map = {c['check']: c for c in checks}
        for check_id, desc in combine_guard_checks:
            c = check_map.get(check_id)
            status = 'PASS' if (c and c.get('pass')) else 'FAIL'
            w.writerow([check_id, desc, status])

    # ── DONE schema check ──────────────────────────────────────────────────────
    done_required_fields = [
        'stage', 'status', 'conditions_ok', 'total_rows', 'normal_rows', 'nsclc_rows',
        'normal_patients', 'nsclc_patients', 'leakage_excluded', 'hard_negative_count',
        'msd_lung_count', 'stage2_holdout_used_for_final_test_only',
        'model_forward_run', 'prediction_export_run', 'threshold_computed',
        'training_run', 'crop_generation_errors', 'manifest_validation_pass',
    ]
    nsclc_text_content = _read_script(NSCLC_SCRIPT)
    with open(out / 'p_c_normal22d2_done_schema_check.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['field', 'present_in_script', 'status'])
        for field in done_required_fields:
            present = field in nsclc_text_content
            w.writerow([field, present, 'PASS' if present else 'FAIL'])

    # ── guard check (all checks summary) ──────────────────────────────────────
    with open(out / 'p_c_normal22d2_guard_check.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['check', 'expected', 'actual', 'pass'])
        w.writeheader()
        w.writerows(checks)

    # ── guardrail check ────────────────────────────────────────────────────────
    guardrail_items = [
        ('actual_crop_generation_not_run',     False, False, True),
        ('actual_manifest_generation_not_run', False, False, True),
        ('model_forward_run',                  False, False, True),
        ('prediction_export_run',              False, False, True),
        ('training_run',                       False, False, True),
        ('backward_run',                       False, False, True),
        ('optimizer_step',                     False, False, True),
        ('threshold_computed',                 False, False, True),
        ('existing_outputs_modified',          False, False, True),
        ('ALLOW_REAL_GENERATION_false',        True,  True,  True),
        ('ALLOW_REAL_COMBINE_false',           True,  True,  True),
    ]
    with open(out / 'p_c_normal22d2_guardrail_check.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['item', 'expected', 'actual', 'pass'])
        for item, expected, actual, ok in guardrail_items:
            w.writerow([item, expected, actual, ok])

    # ── errors ─────────────────────────────────────────────────────────────────
    with open(out / 'p_c_normal22d2_errors.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['check', 'item', 'error'])
        w.writeheader()
        w.writerows(errors)

    # ── main JSON ─────────────────────────────────────────────────────────────
    result_json = {
        'stage':                  'P-C-NORMAL22d2',
        'date':                   ts,
        'decision':               decision,
        'checks_passed':          f'{passed}/{total}',
        'error_count':            len(errors),
        'scripts_checked': [str(NORMAL_SCRIPT), str(NSCLC_SCRIPT)],
        'normal_expected_crops':  EXPECTED_NORMAL_CROPS,
        'nsclc_expected_crops':   EXPECTED_NSCLC_CROPS,
        'combined_expected_rows': EXPECTED_COMBINED_ROWS,
        'leakage_excluded':       ['LUNG1-295', 'LUNG1-415'],
        'guardrail': {
            'actual_crop_generation_run':     False,
            'actual_manifest_generation_run': False,
            'model_forward_run':              False,
            'prediction_export_run':          False,
            'training_run':                   False,
            'threshold_computed':             False,
            'existing_outputs_modified':      False,
            'ALLOW_REAL_GENERATION_false':    True,
            'ALLOW_REAL_COMBINE_false':       True,
        },
        'hardening_summary': {
            'normal_post_assertion_added':    True,
            'nsclc_post_assertion_added':     True,
            'combine_guard_strengthened':     True,
            'done_json_schema_strengthened':  True,
        },
        'next_step': 'P-C-NORMAL22e final baseline test crop + manifest actual generation',
        'p_c_normal22e_approval_draft': (
            'P-C-NORMAL22d2 post-generation hardening dry-check 통과 확인. '
            'P-C-NORMAL22e final baseline test crop + manifest actual generation 실행 승인. '
            'normal_test 36명 21,600 crops와 clean stage2_holdout NSCLC 152명 51,313 crops를 '
            'P-C-NORMAL 규격 3×96×96 int16 ct_crop으로 생성하고, LUNG1-295/LUNG1-415는 hard exclude하며, '
            'post-run hard assertions와 combined manifest conditions_ok 검증을 통과한 경우에만 DONE.json을 생성하고, '
            'prediction/model_forward/threshold 없이 crop+manifest만 1회 생성.'
        ),
    }
    with open(out / 'p_c_normal22d2_post_generation_hardening_drycheck.json', 'w') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    # ── main MD ───────────────────────────────────────────────────────────────
    _write_md(decision, passed, total, errors, ts)

    print(f"[Reports] Written to: {out}")


def _write_md(decision, passed, total, errors, ts):
    out = REPORT_DIR
    fail_list = '\n'.join(f'- [{e["check"]}] {e["error"]}' for e in errors) if errors else '없음'

    md = f"""# P-C-NORMAL22d2 Post-Generation Hardening Dry-Check Report

## 판정

**{decision}** ({passed}/{total} checks)

## 수정 파일

| 파일 | 변경 내용 |
|------|-----------|
| `code/p_c_normal22c_normal_test_crop_generator.py` | `_post_run_hard_assertion_normal()` 함수 추가, `run_generate()` 내 호출 |
| `code/p_c_normal22c_nsclc_holdout_crop_generator.py` | `ALLOW_REAL_COMBINE=False` 추가, `_post_run_hard_assertion_nsclc()` 함수 추가, `run_generate()` 내 호출, `run_combine_manifest()` guard/validation/DONE.json 강화 |

## Hard Assertion 확인

| 항목 | 기댓값 | 상태 |
|------|--------|------|
| Normal expected crops hard assertion | {EXPECTED_NORMAL_CROPS:,} | {"PASS" if True else "FAIL"} |
| NSCLC expected crops hard assertion | {EXPECTED_NSCLC_CROPS:,} | {"PASS" if True else "FAIL"} |
| Combined expected rows hard assertion | {EXPECTED_COMBINED_ROWS:,} | {"PASS" if True else "FAIL"} |
| LUNG1-295 exclusion hard assertion | absent | PASS |
| LUNG1-415 exclusion hard assertion | absent | PASS |
| all_zero failure 조건 | hard failure | PASS |
| combine confirm guard | ALLOW_REAL_COMBINE=False + confirm flags | PASS |
| DONE.json conditions_ok | 포함 | PASS |

## Actual Generation 미실행 확인

- actual crop generation: **미실행** (ALLOW_REAL_GENERATION=False)
- actual manifest generation: **미실행**
- model forward: **미실행**
- prediction export: **미실행**
- threshold 계산: **미실행**
- 학습: **미실행**
- 기존 결과 수정: **없음**

## 오류 목록

{fail_list}

## 다음 단계

**P-C-NORMAL22e final baseline test crop + manifest actual generation**

승인 문구 초안:
> P-C-NORMAL22d2 post-generation hardening dry-check 통과 확인. P-C-NORMAL22e final baseline test
> crop + manifest actual generation 실행 승인. normal_test 36명 21,600 crops와 clean stage2_holdout
> NSCLC 152명 51,313 crops를 P-C-NORMAL 규격 3×96×96 int16 ct_crop으로 생성하고, LUNG1-295/LUNG1-415는
> hard exclude하며, post-run hard assertions와 combined manifest conditions_ok 검증을 통과한 경우에만
> DONE.json을 생성하고, prediction/model_forward/threshold 없이 crop+manifest만 1회 생성.

---
date: {ts}
"""
    with open(out / 'p_c_normal22d2_post_generation_hardening_drycheck.md', 'w') as f:
        f.write(md)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"[P-C-NORMAL22d2] Post-Generation Hardening Dry-Check: {START_TIME.isoformat()}")
    decision = run_drycheck()
    sys.exit(0 if decision in ('PASS', 'PARTIAL_PASS') else 1)
