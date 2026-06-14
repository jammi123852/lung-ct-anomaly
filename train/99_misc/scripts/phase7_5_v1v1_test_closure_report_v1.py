"""
Phase 7.5 v1/v1 test closure report
- Phase 6.4 ~ Phase 7.4 결과 종합
- closure_status: CLOSED_STAGE1_DEV_CROP_LEVEL
- 새 metric 계산 / scoring 재실행 / threshold 계산 금지
"""

import csv
import json
import pathlib
import sys

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BASE = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1"

PHASE64_JSON = BASE / "review_annotations/phase6_4_v1v1_second_stage_design_freeze_v1/phase6_4_v1v1_second_stage_design_freeze_v1.json"
PHASE64_CSV  = BASE / "review_annotations/phase6_4_v1v1_second_stage_design_freeze_v1/phase6_4_v1v1_second_stage_design_freeze_v1.csv"
PHASE64_MD   = BASE / "review_annotations/phase6_4_v1v1_second_stage_design_freeze_v1/phase6_4_v1v1_second_stage_design_freeze_report_v1.md"

PHASE72_JSON = BASE / "scores/phase7_2_v1v1_stage1_dev_full_scoring_v1/phase7_2_v1v1_stage1_dev_full_scoring_summary_v1.json"
PHASE72_MD   = BASE / "scores/phase7_2_v1v1_stage1_dev_full_scoring_v1/phase7_2_v1v1_stage1_dev_full_scoring_report_v1.md"
PHASE72_DONE = BASE / "scores/phase7_2_v1v1_stage1_dev_full_scoring_v1/phase7_2_v1v1_stage1_dev_full_scoring_DONE.json"

PHASE73_CSV  = BASE / "review_annotations/phase7_3_v1v1_metric_calculation_preflight_v1/phase7_3_v1v1_metric_calculation_preflight_v1.csv"
PHASE73_JSON = BASE / "review_annotations/phase7_3_v1v1_metric_calculation_preflight_v1/phase7_3_v1v1_metric_calculation_preflight_v1.json"
PHASE73_MD   = BASE / "review_annotations/phase7_3_v1v1_metric_calculation_preflight_v1/phase7_3_v1v1_metric_calculation_preflight_report_v1.md"

PHASE74_CSV  = BASE / "evaluation/phase7_4_v1v1_crop_level_metrics_v1/phase7_4_v1v1_crop_level_metrics_v1.csv"
PHASE74_JSON = BASE / "evaluation/phase7_4_v1v1_crop_level_metrics_v1/phase7_4_v1v1_crop_level_metrics_v1.json"
PHASE74_MD   = BASE / "evaluation/phase7_4_v1v1_crop_level_metrics_v1/phase7_4_v1v1_crop_level_metrics_report_v1.md"

PHASE75_DIR = BASE / "review_annotations/phase7_5_v1v1_test_closure_report_v1"
OUT_CSV = PHASE75_DIR / "phase7_5_v1v1_test_closure_report_v1.csv"
OUT_JSON = PHASE75_DIR / "phase7_5_v1v1_test_closure_report_v1.json"
OUT_MD   = PHASE75_DIR / "phase7_5_v1v1_test_closure_report_v1.md"

# ---------------------------------------------------------------------------
# output overwrite guard
# ---------------------------------------------------------------------------
def check_output_guard():
    if PHASE75_DIR.exists():
        print(f"[ERROR] output root already exists: {PHASE75_DIR}")
        sys.exit(1)
    for p in [OUT_CSV, OUT_JSON, OUT_MD]:
        if p.exists():
            print(f"[ERROR] output file already exists: {p}")
            sys.exit(1)


def recheck_before_save(path):
    if path.exists():
        print(f"[ERROR] output file already exists before save: {path}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# JSON 로드 헬퍼
# ---------------------------------------------------------------------------
def load_json(path):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# 입력 파일 전체 존재 확인
# ---------------------------------------------------------------------------
REQUIRED_INPUT_FILES = [
    PHASE64_CSV, PHASE64_JSON, PHASE64_MD,
    PHASE72_JSON, PHASE72_MD, PHASE72_DONE,
    PHASE73_CSV, PHASE73_JSON, PHASE73_MD,
    PHASE74_CSV, PHASE74_JSON, PHASE74_MD,
]


def check_input_files():
    missing = [p for p in REQUIRED_INPUT_FILES if not p.exists()]
    if missing:
        for p in missing:
            print(f"[ERROR] required input not found: {p}", file=sys.stderr)
        sys.exit(1)
    print(f"[check] input files: {len(REQUIRED_INPUT_FILES)}/{len(REQUIRED_INPUT_FILES)} PASS")


# ---------------------------------------------------------------------------
# sanity check
# ---------------------------------------------------------------------------
def check_sanity(p64, p72_done, p73, p74):
    errors = []

    # Phase 6.4
    dfs = (p64 or {}).get("design_freeze_status")
    if dfs != "FROZEN":
        errors.append(f"Phase 6.4 design_freeze_status expected=FROZEN, got={dfs}")
    mpc = (p64 or {}).get("minimal_pipeline_connectivity_status")
    if mpc != "PASS":
        errors.append(f"Phase 6.4 minimal_pipeline_connectivity_status expected=PASS, got={mpc}")

    # Phase 7.2 DONE
    if (p72_done or {}).get("scoring_pass") is not True:
        errors.append(f"Phase 7.2 scoring_pass expected=True, got={(p72_done or {}).get('scoring_pass')}")
    if (p72_done or {}).get("output_csv_row_count") != 129437:
        errors.append(f"Phase 7.2 output_csv_row_count expected=129437, got={(p72_done or {}).get('output_csv_row_count')}")
    if (p72_done or {}).get("error_count") != 0:
        errors.append(f"Phase 7.2 error_count expected=0, got={(p72_done or {}).get('error_count')}")
    if (p72_done or {}).get("metric_calculation_executed") is not False:
        errors.append("Phase 7.2 metric_calculation_executed expected=False")
    if (p72_done or {}).get("threshold_calculated") is not False:
        errors.append("Phase 7.2 threshold_calculated expected=False")
    if (p72_done or {}).get("training_executed") is not False:
        errors.append("Phase 7.2 training_executed expected=False")

    # Phase 7.3
    r = (p73 or {}).get("readiness_for_phase7_4")
    if r != "READY_FOR_PHASE7_4_METRIC_CALCULATION":
        errors.append(
            f"Phase 7.3 readiness expected=READY_FOR_PHASE7_4_METRIC_CALCULATION, got={r}"
        )
    # report_consistency_warning=True는 허용 (limitation으로만 기록)

    # Phase 7.4
    if (p74 or {}).get("metric_pass") is not True:
        errors.append(f"Phase 7.4 metric_pass expected=True, got={(p74 or {}).get('metric_pass')}")
    for key in [
        "crop_level_auroc_primary", "crop_level_auprc_primary",
        "crop_level_auroc_secondary", "crop_level_auprc_secondary",
    ]:
        if (p74 or {}).get(key) is None:
            errors.append(f"Phase 7.4 {key} is None or missing")
    if (p74 or {}).get("threshold_calculated") is not False:
        errors.append("Phase 7.4 threshold_calculated expected=False")
    if (p74 or {}).get("patient_level_metric_calculated") is not False:
        errors.append("Phase 7.4 patient_level_metric_calculated expected=False")
    if (p74 or {}).get("training_executed") is not False:
        errors.append("Phase 7.4 training_executed expected=False")

    if errors:
        print("[ERROR] sanity check failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    print("[check] sanity check: PASS")


# ---------------------------------------------------------------------------
# Section A: closure decision
# ---------------------------------------------------------------------------
def run_section_a(p64, p72_done, p73, p74):
    rows = []

    def add(item, status, evidence, limitation=""):
        rows.append({
            "section": "A",
            "item": item,
            "status": status,
            "evidence": evidence,
            "limitation": limitation,
        })

    # pipeline design
    dfs = (p64 or {}).get("design_freeze_status", "N/A")
    add("v1v1_pipeline_design",
        "CLOSED" if dfs == "FROZEN" else f"UNKNOWN({dfs})",
        f"Phase 6.4 design_freeze_status={dfs}")

    # stage1_dev crop-level test
    p74_pass = (p74 or {}).get("metric_pass", False)
    add("stage1_dev_crop_level_test",
        "CLOSED_STAGE1_DEV_CROP_LEVEL" if p74_pass else "INCOMPLETE",
        "Phase 7.4 metric_pass=" + str(p74_pass))

    # minimal pipeline connectivity
    mpc = (p64 or {}).get("minimal_pipeline_connectivity_status", "N/A")
    add("minimal_pipeline_connectivity",
        "PASS" if mpc == "PASS" else f"UNKNOWN({mpc})",
        f"Phase 6.4 minimal_pipeline_connectivity_status={mpc}")

    # full scoring
    sp = (p72_done or {}).get("scoring_pass", False)
    rc = (p72_done or {}).get("output_csv_row_count", "N/A")
    ec = (p72_done or {}).get("error_count", "N/A")
    add("full_scoring",
        "PASS" if sp else "FAIL",
        f"Phase 7.2 scoring_pass={sp}, row_count={rc}, error_count={ec}")

    # crop-level metric
    auroc_p = (p74 or {}).get("crop_level_auroc_primary")
    add("crop_level_metric_calculation",
        "PASS" if p74_pass else "FAIL",
        f"Phase 7.4 auroc_primary={auroc_p}, metric_pass={p74_pass}")

    # stage2_holdout
    add("stage2_holdout_evaluation",
        "NOT_DONE / LOCKED",
        "stage2_holdout sealed; not accessed in any phase",
        "holdout data must not be used until separate approval")

    # patient-level metric
    add("patient_level_metric",
        "NOT_DONE / REQUIRES_LABEL_AGGREGATION_STRATEGY",
        "Phase 7.3: all 152 patients have mixed labels (0 and 1)",
        "aggregation strategy (e.g. patient=positive if any crop=1) must be defined first")

    # threshold
    add("threshold_selection",
        "NOT_DONE / FORBIDDEN_UNTIL_SEPARATE_APPROVAL",
        "Phase 7.4 threshold_calculated=False",
        "threshold policy must be defined and approved separately")

    # hard negative finalization
    add("hard_negative_finalization",
        "NOT_DONE / FORBIDDEN_UNTIL_SEPARATE_APPROVAL",
        "hard_negative candidates not finalized",
        "requires separate approval")

    # training
    te = (p72_done or {}).get("training_executed", False)
    add("training_or_retraining",
        "NOT_DONE",
        f"Phase 7.2 training_executed={te}; no training in any phase")

    return rows


# ---------------------------------------------------------------------------
# Section B: completed phases
# ---------------------------------------------------------------------------
def run_section_b(p64, p72_done, p73, p74):
    rows = []

    def add(phase, result, key_numbers, output_path, note=""):
        rows.append({
            "section": "B",
            "phase": phase,
            "result": result,
            "key_numbers": key_numbers,
            "output_path": output_path,
            "note": note,
        })

    # Phase 6.4
    dfs = (p64 or {}).get("design_freeze_status", "N/A")
    mpc = (p64 or {}).get("minimal_pipeline_connectivity_status", "N/A")
    add("phase6_4",
        f"design_freeze={dfs}, connectivity={mpc}",
        f"design_freeze_status={dfs}; minimal_pipeline_connectivity_status={mpc}",
        str(PHASE64_JSON),
        "second-stage pipeline design frozen")

    # Phase 7.2
    sp = (p72_done or {}).get("scoring_pass", "N/A")
    rc = (p72_done or {}).get("output_csv_row_count", "N/A")
    ec = (p72_done or {}).get("error_count", "N/A")
    add("phase7_2",
        "full_scoring_PASS" if sp else "full_scoring_FAIL",
        f"row_count={rc}, error_count={ec}, scoring_pass={sp}",
        str(PHASE72_DONE),
        "129437 crops scored; 27 canonical columns; no NaN/Inf")

    # Phase 7.3
    rd = (p73 or {}).get("readiness_for_phase7_4", "N/A")
    ld = (p73 or {}).get("label_distribution", {})
    prev = (p73 or {}).get("positive_prevalence")
    if prev is None and ld:
        n1 = ld.get("1", 0)
        n0 = ld.get("0", 0)
        prev = round(n1 / (n0 + n1), 4) if (n0 + n1) > 0 else "N/A"
    add("phase7_3",
        rd,
        f"label=0:{ld.get('0','N/A')}, label=1:{ld.get('1','N/A')}, prevalence={prev}",
        str(PHASE73_JSON),
        "binary label confirmed; crop-level AUROC/AUPRC feasible")

    # Phase 7.4
    auroc_p = (p74 or {}).get("crop_level_auroc_primary", "N/A")
    auprc_p = (p74 or {}).get("crop_level_auprc_primary", "N/A")
    auroc_s = (p74 or {}).get("crop_level_auroc_secondary", "N/A")
    auprc_s = (p74 or {}).get("crop_level_auprc_secondary", "N/A")
    backend = (p74 or {}).get("metric_backend", "N/A")
    add("phase7_4",
        "crop_level_metric_PASS",
        f"AUROC_primary={auroc_p}, AUPRC_primary={auprc_p}, "
        f"AUROC_secondary={auroc_s}, AUPRC_secondary={auprc_s}",
        str(PHASE74_JSON),
        f"metric_backend={backend}; threshold/patient-level not calculated")

    return rows


# ---------------------------------------------------------------------------
# Section C: final metrics
# ---------------------------------------------------------------------------
def run_section_c(p74):
    rows = []
    interp = "stage1_dev filtered crop-level; not final; stage2_holdout not evaluated"

    def add(metric_name, score_column, value, direction, scope):
        rows.append({
            "section": "C",
            "metric_name": metric_name,
            "score_column": score_column,
            "value": value,
            "direction": direction,
            "scope": scope,
            "interpretation_limit": interp,
        })

    p = p74 or {}
    add("crop_level_auroc_primary",
        "crop_score_l1_mean",
        p.get("crop_level_auroc_primary", "N/A"),
        "higher_better",
        "stage1_dev_crop_level")
    add("crop_level_auprc_primary",
        "crop_score_l1_mean",
        p.get("crop_level_auprc_primary", "N/A"),
        "higher_better",
        "stage1_dev_crop_level")
    add("crop_level_auroc_secondary",
        "crop_score_mse_mean",
        p.get("crop_level_auroc_secondary", "N/A"),
        "higher_better",
        "stage1_dev_crop_level")
    add("crop_level_auprc_secondary",
        "crop_score_mse_mean",
        p.get("crop_level_auprc_secondary", "N/A"),
        "higher_better",
        "stage1_dev_crop_level")

    return rows


# ---------------------------------------------------------------------------
# Section D: not done / forbidden
# ---------------------------------------------------------------------------
def run_section_d():
    rows = []

    def add(item, status, reason, required_before_use):
        rows.append({
            "section": "D",
            "item": item,
            "status": status,
            "reason": reason,
            "required_before_use": required_before_use,
        })

    add("patient_level_metric",
        "NOT_DONE",
        "모든 환자가 label 0/1 혼재; aggregation strategy 미정",
        "patient-level label 집계 전략 정의 및 별도 승인")
    add("threshold_selection",
        "FORBIDDEN_UNTIL_APPROVAL",
        "threshold policy 미결정",
        "threshold policy preflight 및 별도 승인")
    add("p95_p99",
        "FORBIDDEN_UNTIL_APPROVAL",
        "threshold 관련 통계; threshold policy 승인 전 금지",
        "threshold policy 승인 후 진행 가능")
    add("hit_rate",
        "FORBIDDEN_UNTIL_APPROVAL",
        "threshold 기반 지표; threshold 미확정",
        "threshold 확정 후 진행 가능")
    add("stage2_holdout_evaluation",
        "LOCKED",
        "holdout sealed; stage1_dev 테스트 종료 후에도 별도 승인 필요",
        "stage2_holdout evaluation preflight 및 별도 승인")
    add("hard_negative_finalization",
        "FORBIDDEN_UNTIL_APPROVAL",
        "hard negative 후보 분석 완료, 최종 채택 미결정",
        "hard negative finalization policy 및 별도 승인")
    add("training_retraining",
        "NOT_DONE",
        "현재 파이프라인은 평가 전용",
        "별도 훈련 계획 및 승인")

    return rows


# ---------------------------------------------------------------------------
# Section E: next options
# ---------------------------------------------------------------------------
def run_section_e():
    rows = []

    def add(option_id, option_name, description, approval_required, recommendation):
        rows.append({
            "section": "E",
            "option_id": option_id,
            "option_name": option_name,
            "description": description,
            "approval_required": str(approval_required),
            "recommendation": recommendation,
        })

    add("A", "patient_level_metric_design",
        "patient-level label 집계 전략 정의 후 AUROC/AUPRC 계산",
        True, "별도 승인 필요")
    add("B", "threshold_policy_preflight",
        "threshold 기준 설계 및 p95/p99 계산 방식 결정",
        True, "별도 승인 필요")
    add("C", "stage2_holdout_final_evaluation_preflight",
        "stage2_holdout 봉인 해제 전 preflight 설계",
        True, "별도 승인 필요")
    add("D", "handoff_document_update",
        "현재 결과(Phase 6.4~7.4)를 handoff 문서에 반영",
        False, "RECOMMENDED_FIRST")

    return rows


# ---------------------------------------------------------------------------
# CSV 저장
# ---------------------------------------------------------------------------
def save_csv(a_rows, b_rows, c_rows, d_rows, e_rows):
    all_fields = sorted({
        "section", "item", "status", "evidence", "limitation",
        "phase", "result", "key_numbers", "output_path", "note",
        "metric_name", "score_column", "value", "direction", "scope",
        "interpretation_limit",
        "reason", "required_before_use",
        "option_id", "option_name", "description", "approval_required",
        "recommendation",
    })

    all_rows = []
    for r in a_rows + b_rows + c_rows + d_rows + e_rows:
        all_rows.append({f: r.get(f, "") for f in all_fields})

    recheck_before_save(OUT_CSV)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_fields))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[CSV] saved: {OUT_CSV}  ({len(all_rows)} rows)")


# ---------------------------------------------------------------------------
# JSON 저장
# ---------------------------------------------------------------------------
def save_json(p64, p72_done, p73, p74):
    ld = (p73 or {}).get("label_distribution", {})
    n1 = ld.get("1", 0)
    n0 = ld.get("0", 0)
    prev73 = round(n1 / (n0 + n1), 4) if (n0 + n1) > 0 else None

    auroc_p = (p74 or {}).get("crop_level_auroc_primary")
    auprc_p = (p74 or {}).get("crop_level_auprc_primary")
    auroc_s = (p74 or {}).get("crop_level_auroc_secondary")
    auprc_s = (p74 or {}).get("crop_level_auprc_secondary")
    prev74 = (p74 or {}).get("positive_prevalence")

    output = {
        "closure_status": "CLOSED_STAGE1_DEV_CROP_LEVEL",
        "design_status": "CLOSED",
        "test_status": "CLOSED_STAGE1_DEV_CROP_LEVEL",
        "phase6_4_summary": {
            "design_freeze_status": (p64 or {}).get("design_freeze_status", "N/A"),
            "minimal_pipeline_connectivity_status":
                (p64 or {}).get("minimal_pipeline_connectivity_status", "N/A"),
        },
        "phase7_2_summary": {
            "scoring_pass": (p72_done or {}).get("scoring_pass"),
            "output_csv_row_count": (p72_done or {}).get("output_csv_row_count"),
            "error_count": (p72_done or {}).get("error_count"),
            "metric_calculation_executed":
                (p72_done or {}).get("metric_calculation_executed"),
            "threshold_calculated": (p72_done or {}).get("threshold_calculated"),
            "training_executed": (p72_done or {}).get("training_executed"),
        },
        "phase7_3_summary": {
            "readiness": (p73 or {}).get("readiness_for_phase7_4"),
            "label_distribution": ld,
            "positive_prevalence": prev73,
            "report_consistency_warning":
                (p73 or {}).get("report_consistency_warning"),
        },
        "phase7_4_summary": {
            "crop_level_auroc_primary": auroc_p,
            "crop_level_auprc_primary": auprc_p,
            "crop_level_auroc_secondary": auroc_s,
            "crop_level_auprc_secondary": auprc_s,
            "positive_prevalence": prev74,
            "metric_backend": (p74 or {}).get("metric_backend"),
            "metric_pass": (p74 or {}).get("metric_pass"),
        },
        "final_crop_level_metrics": {
            "crop_level_auroc_primary": auroc_p,
            "crop_level_auprc_primary": auprc_p,
            "crop_level_auroc_secondary": auroc_s,
            "crop_level_auprc_secondary": auprc_s,
            "positive_prevalence": prev74 if prev74 is not None else prev73,
        },
        "completed_items": [
            "v1v1_pipeline_design",
            "stage1_dev_crop_level_test",
            "minimal_pipeline_connectivity",
            "full_scoring",
            "crop_level_metric_calculation",
        ],
        "not_done_items": [
            "patient_level_metric",
            "threshold_selection",
            "p95_p99",
            "hit_rate",
            "stage2_holdout_evaluation",
            "hard_negative_finalization",
            "training_retraining",
        ],
        "forbidden_until_separate_approval": [
            "threshold_selection",
            "p95_p99",
            "hit_rate",
            "stage2_holdout_evaluation",
            "hard_negative_finalization",
        ],
        "next_options": [
            "A: patient_level_metric_design",
            "B: threshold_policy_preflight",
            "C: stage2_holdout_final_evaluation_preflight",
            "D: handoff_document_update (RECOMMENDED_FIRST)",
        ],
        "recommended_next_step": (
            "Phase 7.5 D handoff 최신화 권장. "
            "patient-level/threshold/stage2_holdout은 별도 승인 후 진행."
        ),
        "limitations": [
            "stage1_dev filtered crop-level 결과이며 stage2_holdout 최종 검증이 아니다.",
            "patient-level metric 미계산.",
            "threshold 기반 성능 평가 미수행.",
            "hard negative 최종 채택 미결정.",
            "병변 탐지 성능의 최종 결론이 아니다.",
            "원본 S6-A index에는 stage2_holdout 2명 포함 이력 있음; filtered shadow manifest만 사용.",
            "metric_backend=numpy_fallback; sklearn 미설치.",
        ],
        "notes": {
            "closure_report_only": True,
            "no_new_scoring": True,
            "no_new_metric_calculation": True,
            "no_threshold": True,
            "no_training": True,
            "no_stage2_holdout": True,
            "no_v2": True,
        },
    }

    recheck_before_save(OUT_JSON)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[JSON] saved: {OUT_JSON}")
    return output


# ---------------------------------------------------------------------------
# MD report 저장
# ---------------------------------------------------------------------------
def save_md(jdata, c_rows):
    lines = []

    def h2(t):
        lines.append(f"\n## {t}\n")

    def li(t):
        lines.append(f"- {t}")

    lines.append("# Phase 7.5 v1/v1 test closure report\n")

    h2("1. Phase 7.5 목적")
    li("v1/v1 기반 2차 파이프라인 stage1_dev filtered crop-level 테스트를 종료한다.")
    li("Phase 6.4 design freeze부터 Phase 7.4 crop-level metric까지의 결과를 정리한다.")
    li("closure report only — 새 scoring, metric 재계산, threshold, training 없음.")
    li("입력 파일 전체 존재 확인: PASS "
       "(Phase 6.4 × 3, Phase 7.2 × 3, Phase 7.3 × 3, Phase 7.4 × 3 = 12파일)")

    h2("2. 최종 판정")
    lines.append(f"\n**closure_status: {jdata['closure_status']}**\n")
    lines.append(f"**design_status: {jdata['design_status']}**\n")
    lines.append(f"**test_status: {jdata['test_status']}**\n")

    h2("3. 완료된 파이프라인 범위")
    lines.append("")
    lines.append("| phase | result | key_numbers |")
    lines.append("|-------|--------|-------------|")
    p64s = jdata["phase6_4_summary"]
    lines.append(
        f"| Phase 6.4 | design_freeze={p64s['design_freeze_status']}, "
        f"connectivity={p64s['minimal_pipeline_connectivity_status']} | — |"
    )
    p72s = jdata["phase7_2_summary"]
    lines.append(
        f"| Phase 7.2 | full_scoring_PASS | "
        f"row_count={p72s['output_csv_row_count']}, error={p72s['error_count']} |"
    )
    p73s = jdata["phase7_3_summary"]
    lines.append(
        f"| Phase 7.3 | {p73s['readiness']} | "
        f"label=0:{p73s['label_distribution'].get('0','N/A')}, "
        f"label=1:{p73s['label_distribution'].get('1','N/A')}, "
        f"prevalence={p73s['positive_prevalence']} |"
    )
    p74s = jdata["phase7_4_summary"]
    lines.append(
        f"| Phase 7.4 | crop_level_metric_PASS | "
        f"AUROC_p={p74s['crop_level_auroc_primary']}, "
        f"AUPRC_p={p74s['crop_level_auprc_primary']} |"
    )

    h2("4. 최종 crop-level metric 결과")
    lines.append("")
    lines.append("| metric_name | score_column | value | direction | scope |")
    lines.append("|-------------|-------------|-------|-----------|-------|")
    for r in c_rows:
        lines.append(
            f"| {r['metric_name']} | {r['score_column']} | {r['value']} "
            f"| {r['direction']} | {r['scope']} |"
        )
    fm = jdata["final_crop_level_metrics"]
    li(f"positive prevalence = {fm['positive_prevalence']}")
    li(f"metric_backend = {p74s['metric_backend']}")

    h2("5. 해석 범위")
    for lim in jdata["limitations"]:
        li(lim)

    h2("6. 원본 S6-A index contamination 주의")
    li("원본 S6-A index는 stage2_holdout 2명(LUNG1-295, LUNG1-415)을 포함한 이력이 있다.")
    li("Phase 6.1b 이후에는 filtered shadow manifest만 사용해야 한다.")
    li("원본 index를 다시 사용하거나 수정하는 것은 금지된다.")

    h2("7. 아직 하지 않은 것")
    not_done = [
        "patient-level metric — aggregation strategy 미정",
        "threshold 선택 — policy 미결정",
        "p95/p99 계산",
        "hit-rate 계산",
        "stage2_holdout evaluation — holdout 봉인 유지",
        "hard negative final manifest 확정",
        "training / retraining",
    ]
    for item in not_done:
        li(item)

    h2("8. 다음 선택지")
    li("**A** patient-level metric design — 별도 승인 필요")
    li("**B** threshold policy preflight — 별도 승인 필요")
    li("**C** stage2_holdout final evaluation preflight — 별도 승인 필요")
    li("**D** handoff 문서 최신화 — 승인 불필요 **(RECOMMENDED_FIRST)**")

    h2("9. 추천")
    li("지금은 **D handoff 최신화**를 먼저 진행한다.")
    li("patient-level metric / threshold / stage2_holdout은 별도 승인 후 진행한다.")

    h2("10. 금지 사항 확인 목록")
    forbidden = [
        "새 metric 계산 금지",
        "scoring 재실행 금지",
        "threshold 계산 금지",
        "p95/p99 계산 금지",
        "patient-level metric 계산 금지",
        "model forward 금지",
        "training 금지",
        "checkpoint 생성 금지",
        "hard negative 최종 채택 금지",
        "score CSV 수정 금지",
        "filtered manifest 수정 금지",
        "stage2_holdout 접근 금지",
        "v2/v2v2 접근 금지",
        "기존 Phase 6/7 output 수정 금지",
    ]
    for item in forbidden:
        li(f"[확인] {item}")

    recheck_before_save(OUT_MD)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[MD] saved: {OUT_MD}")


# ---------------------------------------------------------------------------
# 콘솔 보고 (보고 형식 1~22번)
# ---------------------------------------------------------------------------
def print_report(jdata):
    print("\n" + "=" * 70)
    print("Phase 7.5 test closure report 보고")
    print("=" * 70)

    def p(n, label, val):
        print(f"  {n:2}. {label}: {val}")

    fm = jdata["final_crop_level_metrics"]
    p74s = jdata["phase7_4_summary"]

    p(1,  "output root", str(PHASE75_DIR))
    p(2,  "생성 CSV 경로", str(OUT_CSV))
    p(3,  "생성 JSON 경로", str(OUT_JSON))
    p(4,  "생성 MD 경로", str(OUT_MD))
    p(5,  "반영한 Phase 범위", "Phase 6.4 ~ Phase 7.4")
    p(6,  "closure_status", jdata["closure_status"])
    p(7,  "design_status", jdata["design_status"])
    p(8,  "test_status", jdata["test_status"])
    p(9,  "final primary AUROC/AUPRC",
      f"AUROC={fm['crop_level_auroc_primary']}, AUPRC={fm['crop_level_auprc_primary']}")
    p(10, "final secondary AUROC/AUPRC",
      f"AUROC={fm['crop_level_auroc_secondary']}, AUPRC={fm['crop_level_auprc_secondary']}")
    p(11, "positive prevalence", fm["positive_prevalence"])
    p(12, "해석 제한 반영 여부", "TRUE (MD section 5)")
    p(13, "not-done 목록 반영 여부", "TRUE (MD section 7 / JSON not_done_items)")
    p(14, "forbidden 목록 반영 여부", "TRUE (MD section 10 / JSON forbidden_until_separate_approval)")
    p(15, "새 metric 계산 없음 확인", "TRUE")
    p(16, "scoring 재실행 없음 확인", "TRUE")
    p(17, "threshold 계산 없음 확인", "TRUE")
    p(18, "patient-level metric 계산 없음 확인", "TRUE")
    p(19, "training 없음 확인", "TRUE")
    p(20, "stage2_holdout/v2/v2v2 미접근 확인", "TRUE")
    p(21, "기존 파일 미수정 확인", "TRUE (read-only inputs)")
    p(22, "다음 단계 제안", jdata["recommended_next_step"])

    print("=" * 70)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("[Phase 7.5] test closure report 시작")

    # 1. 입력 파일 전체 존재 확인 (output root 생성 전)
    check_input_files()

    # 2. output guard
    check_output_guard()

    # 3. 입력 JSON 로드
    print("[load] Phase 6.4 JSON")
    p64 = load_json(PHASE64_JSON)
    print("[load] Phase 7.2 DONE marker")
    p72_done = load_json(PHASE72_DONE)
    print("[load] Phase 7.3 JSON")
    p73 = load_json(PHASE73_JSON)
    print("[load] Phase 7.4 JSON")
    p74 = load_json(PHASE74_JSON)

    for name, obj, path in [
        ("Phase 6.4 JSON", p64, PHASE64_JSON),
        ("Phase 7.2 DONE", p72_done, PHASE72_DONE),
        ("Phase 7.3 JSON", p73, PHASE73_JSON),
        ("Phase 7.4 JSON", p74, PHASE74_JSON),
    ]:
        if obj is None:
            print(f"[ERROR] {name} load failed: {path}", file=sys.stderr)
            sys.exit(1)

    # 4. sanity check (output root 생성 전)
    check_sanity(p64, p72_done, p73, p74)

    # 5. output root 생성 (모든 검증 통과 후)
    PHASE75_DIR.mkdir(parents=True, exist_ok=False)

    # 섹션 실행
    a_rows = run_section_a(p64, p72_done, p73, p74)
    b_rows = run_section_b(p64, p72_done, p73, p74)
    c_rows = run_section_c(p74)
    d_rows = run_section_d()
    e_rows = run_section_e()

    # 저장
    save_csv(a_rows, b_rows, c_rows, d_rows, e_rows)
    jdata = save_json(p64, p72_done, p73, p74)
    save_md(jdata, c_rows)
    print_report(jdata)

    print("\n[Phase 7.5] test closure report 완료")


if __name__ == "__main__":
    main()
