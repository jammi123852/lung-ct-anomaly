"""
Phase 7.7 v1/v1 final performance closure report
- v1/v1 기반 2차 모델 성능평가 최종 종료
- Phase 7.4 crop-level + Phase 7.6 patient/top-k 결과 종합
- 새 metric 계산 / scoring 재실행 / threshold 계산 / model forward / training 금지
- stage2_holdout / v2 / v2v2 봉인 유지
"""

import argparse
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

PHASE74_JSON = (
    BASE / "evaluation/phase7_4_v1v1_crop_level_metrics_v1"
    / "phase7_4_v1v1_crop_level_metrics_v1.json"
)
PHASE74_MD = (
    BASE / "evaluation/phase7_4_v1v1_crop_level_metrics_v1"
    / "phase7_4_v1v1_crop_level_metrics_report_v1.md"
)
PHASE75_JSON = (
    BASE / "review_annotations/phase7_5_v1v1_test_closure_report_v1"
    / "phase7_5_v1v1_test_closure_report_v1.json"
)
PHASE75_MD = (
    BASE / "review_annotations/phase7_5_v1v1_test_closure_report_v1"
    / "phase7_5_v1v1_test_closure_report_v1.md"
)
PHASE76_CSV = (
    BASE / "evaluation/phase7_6_v1v1_eval_completeness_patient_topk_v1"
    / "phase7_6_v1v1_eval_completeness_patient_topk_v1.csv"
)
PHASE76_JSON = (
    BASE / "evaluation/phase7_6_v1v1_eval_completeness_patient_topk_v1"
    / "phase7_6_v1v1_eval_completeness_patient_topk_v1.json"
)
PHASE76_MD = (
    BASE / "evaluation/phase7_6_v1v1_eval_completeness_patient_topk_v1"
    / "phase7_6_v1v1_eval_completeness_patient_topk_report_v1.md"
)

OUT_DIR = (
    BASE / "evaluation/phase7_7_v1v1_final_performance_closure_v1"
)
OUT_CSV = OUT_DIR / "phase7_7_v1v1_final_performance_closure_v1.csv"
OUT_JSON = OUT_DIR / "phase7_7_v1v1_final_performance_closure_v1.json"
OUT_MD = OUT_DIR / "phase7_7_v1v1_final_performance_closure_report_v1.md"

# ---------------------------------------------------------------------------
# Hard-coded 결과 (Phase 7.2/7.4/7.6 확인 완료값, 재계산 금지)
# ---------------------------------------------------------------------------
PHASE72_SUMMARY = {
    "scoring_pass": True,
    "row_count": 129437,
    "has_nan": 0,
    "has_inf": 0,
    "failed_crop": 0,
    "runtime_sec": 369.2,
    "batch_size": 32,
}

CROP_METRICS = {
    "crop_score_l1_mean": {"auroc": 0.649008, "auprc": 0.397344},
    "crop_score_mse_mean": {"auroc": 0.625855, "auprc": 0.381264},
}
POSITIVE_PREVALENCE = 0.3363
METRIC_BACKEND = "numpy_fallback"

PATIENT_FEASIBILITY = {
    "patient_positive_if_any_label1": {
        "status": "NOT_APPLICABLE",
        "positive_patient_count": 152,
        "negative_patient_count": 0,
        "reason": "negative patient=0, 전원 positive 판정",
    },
    "patient_positive_if_majority_label1": {
        "status": "FEASIBLE",
        "positive_patient_count": 24,
        "negative_patient_count": 128,
        "reason": "양쪽 class 존재, AUROC/AUPRC 계산 가능",
    },
    "patient_positive_if_positive_ratio_above_0": {
        "status": "NOT_APPLICABLE",
        "positive_patient_count": 152,
        "negative_patient_count": 0,
        "reason": "negative patient=0, 전원 positive 판정",
    },
}

PATIENT_AUROC_MAJORITY = {
    "crop_score_l1_mean": {
        "patient_mean": 0.6152,
        "patient_top10_percent_mean": 0.5866,
        "patient_top5_percent_mean": 0.5703,
        "patient_max": 0.5547,
        "patient_top1_percent_mean": 0.5488,
    },
    "crop_score_mse_mean": {
        "patient_mean": 0.5677,
        "patient_top10_percent_mean": 0.5355,
        "patient_top5_percent_mean": 0.5280,
        "patient_max": 0.5332,
        "patient_top1_percent_mean": 0.5251,
    },
}

TOPK_RANKING_ROW_COUNT = 16

# ---------------------------------------------------------------------------
# 최종 판정 상수
# ---------------------------------------------------------------------------
FINAL_STATUS = "CLOSED_STAGE1_DEV_PERFORMANCE_EVAL"
EVAL_STATUS = {
    "crop_level_metrics_status": "DONE",
    "patient_level_metrics_status": "DONE_CONDITIONAL_MAJORITY_RULE_ONLY",
    "topk_ranking_status": "DONE_DIAGNOSTIC_ONLY",
    "threshold_metrics_status": "NOT_DONE_REQUIRES_THRESHOLD_POLICY",
    "stage2_holdout_status": "LOCKED_NOT_EVALUATED",
    "training_status": "NOT_DONE",
    "hard_negative_finalization_status": "NOT_DONE",
}


# ---------------------------------------------------------------------------
# Output guard
# ---------------------------------------------------------------------------
def check_output_guard():
    if OUT_DIR.exists():
        print(f"[ERROR] output root already exists: {OUT_DIR}")
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
# Section A: final closure decision
# ---------------------------------------------------------------------------
def build_section_a():
    items = [
        (
            "final_performance_closure_status",
            "CLOSED",
            "Phase 7.2/7.4/7.5/7.6 완료 + READY_FOR_PHASE7_7 판정",
            "stage1_dev filtered 결과에 한정",
        ),
        (
            "crop_level_metrics_status",
            "DONE",
            "Phase 7.4 crop AUROC l1=0.649008, mse=0.625855",
            "stage1_dev crop-level, threshold 기반 아님",
        ),
        (
            "patient_level_metrics_status",
            "DONE_CONDITIONAL",
            "majority_label1 rule만 feasible, any_label1/positive_ratio_above_0 NOT_APPLICABLE",
            "majority_label1 rule 조건부 결과",
        ),
        (
            "topk_ranking_status",
            "DONE_DIAGNOSTIC",
            "Phase 7.6 top-k 16 rows, threshold-free diagnostic",
            "threshold 확정 아님",
        ),
        (
            "threshold_metrics_status",
            "NOT_DONE",
            "threshold policy 미결정",
            "별도 승인 필요",
        ),
        (
            "stage2_holdout_status",
            "LOCKED",
            "stage2_holdout 봉인 유지",
            "별도 승인 후에만 접근 가능",
        ),
        (
            "training_status",
            "NOT_DONE",
            "이번 단계에서 수행하지 않음",
            "별도 결정 필요",
        ),
        (
            "hard_negative_finalization_status",
            "NOT_DONE",
            "이번 단계에서 수행하지 않음",
            "별도 결정 필요",
        ),
    ]
    return [
        {
            "section": "A_final_closure_decision",
            "item": item,
            "status": status,
            "evidence": evidence,
            "limitation": limitation,
        }
        for item, status, evidence, limitation in items
    ]


# ---------------------------------------------------------------------------
# Section B: crop-level final metrics
# ---------------------------------------------------------------------------
def build_section_b():
    rows = [
        {
            "section": "B_crop_level_final_metrics",
            "metric_name": "positive_prevalence",
            "score_column": "—",
            "value": str(POSITIVE_PREVALENCE),
            "direction": "—",
            "scope": "stage1_dev crop",
            "interpretation_limit": "stage1_dev filtered, stage2_holdout 아님",
        }
    ]
    for sc, v in CROP_METRICS.items():
        rows.append({
            "section": "B_crop_level_final_metrics",
            "metric_name": "crop_level_auroc",
            "score_column": sc,
            "value": str(v["auroc"]),
            "direction": "↑",
            "scope": "stage1_dev crop",
            "interpretation_limit": "threshold 기반 아님, stage2_holdout 아님",
        })
        rows.append({
            "section": "B_crop_level_final_metrics",
            "metric_name": "crop_level_auprc",
            "score_column": sc,
            "value": str(v["auprc"]),
            "direction": "↑",
            "scope": "stage1_dev crop",
            "interpretation_limit": "threshold 기반 아님, stage2_holdout 아님",
        })
    return rows


# ---------------------------------------------------------------------------
# Section C: patient-level feasibility
# ---------------------------------------------------------------------------
def build_section_c():
    return [
        {
            "section": "C_patient_level_feasibility",
            "patient_label_rule": rule,
            "status": info["status"],
            "positive_patient_count": info["positive_patient_count"],
            "negative_patient_count": info["negative_patient_count"],
            "reason": info["reason"],
            "interpretation_limit": "stage1_dev filtered, majority_label1만 feasible",
        }
        for rule, info in PATIENT_FEASIBILITY.items()
    ]


# ---------------------------------------------------------------------------
# Section D: patient-level AUROC summary (majority_label1 only)
# ---------------------------------------------------------------------------
def build_section_d():
    rows = []
    for sc, agg_vals in PATIENT_AUROC_MAJORITY.items():
        for agg, auroc in agg_vals.items():
            rows.append({
                "section": "D_patient_level_auroc_summary",
                "score_column": sc,
                "aggregation": agg,
                "patient_label_rule": "patient_positive_if_majority_label1",
                "auroc": str(auroc),
                "status": "DONE",
                "note": "majority_label1 조건부 결과, stage1_dev filtered",
            })
    return rows


# ---------------------------------------------------------------------------
# Section E: top-k diagnostic summary
# ---------------------------------------------------------------------------
def build_section_e():
    return [
        {
            "section": "E_topk_diagnostic_summary",
            "item": "topk_ranking_rows",
            "status": "DONE",
            "evidence": f"Phase 7.6 top-k ranking {TOPK_RANKING_ROW_COUNT} rows 생성",
            "interpretation_limit": (
                "threshold-free diagnostic metric, "
                "threshold 확정 아님, p95/p99 계산 아님"
            ),
        },
        {
            "section": "E_topk_diagnostic_summary",
            "item": "topk_scope",
            "status": "DONE",
            "evidence": (
                "global level + per_patient_mean, "
                "top-1%/5%/10%/20%, l1_mean/mse_mean"
            ),
            "interpretation_limit": (
                "ranking diagnostic only, threshold 적용 결과 아님"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Section F: not done / locked
# ---------------------------------------------------------------------------
def build_section_f():
    items = [
        (
            "threshold_p95_p99_hit_rate",
            "NOT_DONE",
            "threshold policy 미결정",
            "threshold policy preflight 별도 승인 후 진행",
        ),
        (
            "stage2_holdout_evaluation",
            "LOCKED",
            "stage2_holdout 봉인 유지",
            "stage2_holdout preflight 별도 승인 후 진행",
        ),
        (
            "training_retraining",
            "NOT_DONE",
            "이번 단계 미수행",
            "별도 결정 후 진행",
        ),
        (
            "hard_negative_final_manifest",
            "NOT_DONE",
            "이번 단계 미수행",
            "별도 결정 후 진행",
        ),
        (
            "suppression_mask_roi_modification",
            "NOT_DONE",
            "이번 단계 미수행",
            "별도 승인 후 진행",
        ),
    ]
    return [
        {
            "section": "F_not_done_locked",
            "item": item,
            "status": status,
            "reason": reason,
            "required_before_use": required,
        }
        for item, status, reason, required in items
    ]


# ---------------------------------------------------------------------------
# Section G: next options
# ---------------------------------------------------------------------------
def build_section_g():
    options = [
        (
            "A",
            "threshold_policy_preflight",
            "p95/p99 기반 threshold 결정 preflight 설계 및 승인",
            "별도 승인 필요",
            "OPTIONAL",
        ),
        (
            "B",
            "stage2_holdout_final_evaluation_preflight",
            "stage2_holdout 최종 평가 preflight 설계 및 승인",
            "별도 승인 필요",
            "OPTIONAL",
        ),
        (
            "C",
            "hard_negative_final_manifest_design",
            "hard negative 최종 manifest 설계",
            "별도 결정 필요",
            "OPTIONAL",
        ),
        (
            "D",
            "handoff_document_update",
            "handoff 문서에 v1/v1 성능평가 결과 반영",
            "승인 불필요",
            "RECOMMENDED_FIRST",
        ),
    ]
    return [
        {
            "section": "G_next_options",
            "option_id": opt_id,
            "option_name": name,
            "description": desc,
            "approval_required": approval,
            "recommendation": rec,
        }
        for opt_id, name, desc, approval, rec in options
    ]


# ---------------------------------------------------------------------------
# CSV 출력
# ---------------------------------------------------------------------------
def write_csv(all_sections):
    recheck_before_save(OUT_CSV)
    all_rows = []
    for section_rows in all_sections:
        all_rows.extend(section_rows)

    all_keys = []
    for r in all_rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k, "") for k in all_keys})

    print(f"[CSV] {OUT_CSV} ({len(all_rows)} rows)")


# ---------------------------------------------------------------------------
# JSON 출력
# ---------------------------------------------------------------------------
def write_json(phase74_data, phase75_data, phase76_data):
    recheck_before_save(OUT_JSON)

    output = {
        "final_performance_closure_status": FINAL_STATUS,
        "input_paths": {
            "phase7_4_json": str(PHASE74_JSON),
            "phase7_4_md": str(PHASE74_MD),
            "phase7_5_json": str(PHASE75_JSON),
            "phase7_5_md": str(PHASE75_MD),
            "phase7_6_csv": str(PHASE76_CSV),
            "phase7_6_json": str(PHASE76_JSON),
            "phase7_6_md": str(PHASE76_MD),
        },
        "crop_level_metrics": {
            "positive_prevalence": POSITIVE_PREVALENCE,
            "metric_backend": METRIC_BACKEND,
            "crop_score_l1_mean": CROP_METRICS["crop_score_l1_mean"],
            "crop_score_mse_mean": CROP_METRICS["crop_score_mse_mean"],
        },
        "phase7_2_scoring_summary": PHASE72_SUMMARY,
        "patient_level_feasibility": [
            {
                "patient_label_rule": rule,
                "status": info["status"],
                "positive_patient_count": info["positive_patient_count"],
                "negative_patient_count": info["negative_patient_count"],
                "reason": info["reason"],
            }
            for rule, info in PATIENT_FEASIBILITY.items()
        ],
        "patient_level_metrics_majority_rule": PATIENT_AUROC_MAJORITY,
        "topk_ranking_summary": {
            "row_count": TOPK_RANKING_ROW_COUNT,
            "scope": (
                "global + per_patient_mean, "
                "top-1%/5%/10%/20%, l1_mean/mse_mean"
            ),
            "note": "threshold-free diagnostic metric, threshold 확정 아님",
        },
        "evaluation_completeness_status": EVAL_STATUS,
        "not_done_items": [
            "threshold_p95_p99_hit_rate",
            "training_retraining",
            "hard_negative_final_manifest",
            "suppression_mask_roi_modification",
        ],
        "locked_items": ["stage2_holdout_evaluation"],
        "forbidden_until_separate_approval": [
            "threshold 계산",
            "p95/p99 계산",
            "threshold-based hit-rate",
            "stage2_holdout 접근",
            "v2/v2v2 접근",
            "model forward",
            "training",
            "scoring 재실행",
            "NSCLC/MSD root 접근",
        ],
        "final_interpretation_limits": [
            "본 결과는 stage1_dev filtered crop-level 범위에 한정된다.",
            "stage2_holdout 최종 검증 결과가 아니다.",
            "threshold 기반 성능 평가가 수행되지 않았다.",
            (
                "patient-level은 majority_label1 rule 조건부 결과이며 "
                "도메인 아티팩트 포함 가능성이 있다."
            ),
            "top-k ranking은 threshold 확정이 아닌 ranking diagnostic metric이다.",
            "병변 성능 최종 결론이 아니다.",
        ],
        "recommended_next_step": "D. handoff 문서 최신화 (v1/v1 성능평가 결과 반영)",
        "notes": {
            "final_closure_report_only": True,
            "no_new_metric_calculation": True,
            "no_scoring_reexecution": True,
            "no_model_forward": True,
            "no_training": True,
            "no_threshold": True,
            "no_p95_p99": True,
            "no_stage2_holdout": True,
            "no_v2": True,
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[JSON] {OUT_JSON}")


# ---------------------------------------------------------------------------
# MD report 출력
# ---------------------------------------------------------------------------
def write_md():
    recheck_before_save(OUT_MD)

    lines = []
    lines.append("# Phase 7.7 v1/v1 Final Performance Closure Report")
    lines.append("")

    lines.append("## 1. Phase 7.7 목적")
    lines.append("")
    lines.append("- v1/v1 기반 2차 모델 성능평가를 여기서 최종 종료한다.")
    lines.append("- Phase 7.4 crop-level metric과 Phase 7.6 patient/top-k diagnostic 결과를 종합한다.")
    lines.append("- 새 metric 계산, scoring 재실행, threshold 계산, model forward, training은 수행하지 않는다.")
    lines.append("- stage2_holdout / v2 / v2v2는 계속 봉인한다.")
    lines.append("")

    lines.append("## 2. 최종 판정")
    lines.append("")
    lines.append(f"**{FINAL_STATUS}**")
    lines.append("")
    lines.append("| 평가 항목 | 상태 |")
    lines.append("|---|---|")
    for k, v in EVAL_STATUS.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## 3. 전체 평가 범위 (완료 현황)")
    lines.append("")
    phases = [
        ("Phase 6.4", "design freeze", "완료"),
        ("Phase 7.2", "full scoring (129,437 rows, has_nan=0, has_inf=0, runtime=369.2s)", "완료"),
        ("Phase 7.4", "crop-level AUROC/AUPRC", "완료"),
        ("Phase 7.5", "stage1_dev crop-level test closure", "완료"),
        ("Phase 7.6", "evaluation completeness + patient/top-k", "완료"),
        ("Phase 7.7", "final performance closure report", "현재"),
    ]
    lines.append("| Phase | 내용 | 상태 |")
    lines.append("|---|---|---|")
    for ph, content, status in phases:
        lines.append(f"| {ph} | {content} | {status} |")
    lines.append("")

    lines.append("## 4. crop-level metric 결과")
    lines.append("")
    lines.append(f"- positive prevalence: {POSITIVE_PREVALENCE}")
    lines.append(f"- metric backend: {METRIC_BACKEND}")
    lines.append("")
    lines.append("| score_column | AUROC(↑) | AUPRC(↑) |")
    lines.append("|---|---|---|")
    for sc, v in CROP_METRICS.items():
        lines.append(f"| {sc} | {v['auroc']} | {v['auprc']} |")
    lines.append("")

    lines.append("## 5. patient-level feasibility 결과")
    lines.append("")
    lines.append("| patient_label_rule | pos_patients | neg_patients | 판정 |")
    lines.append("|---|---|---|---|")
    for rule, info in PATIENT_FEASIBILITY.items():
        lines.append(
            f"| {rule} | {info['positive_patient_count']} "
            f"| {info['negative_patient_count']} | {info['status']} |"
        )
    lines.append("")
    lines.append("- any_label1: NOT_APPLICABLE — negative patient=0, 전원 positive 판정")
    lines.append("- positive_ratio_above_0: NOT_APPLICABLE — negative patient=0, 전원 positive 판정")
    lines.append("- majority_label1: FEASIBLE — pos=24, neg=128")
    lines.append("")

    lines.append("## 6. majority_label1 기준 patient-level AUROC")
    lines.append("")
    lines.append("| score_column | aggregation | AUROC |")
    lines.append("|---|---|---|")
    for sc, agg_vals in PATIENT_AUROC_MAJORITY.items():
        for agg, auroc in agg_vals.items():
            lines.append(f"| {sc} | {agg} | {auroc} |")
    lines.append("")
    lines.append(
        "> **주의**: majority_label1 rule 조건부 결과. "
        "도메인 아티팩트(stage1_dev 특성) 포함 가능성 있음."
    )
    lines.append("")

    lines.append("## 7. top-k ranking diagnostic 요약")
    lines.append("")
    lines.append(f"- Phase 7.6에서 생성된 top-k ranking rows: {TOPK_RANKING_ROW_COUNT}")
    lines.append("- 범위: global level + per_patient_mean, top-1%/5%/10%/20%, l1_mean/mse_mean 2개 score")
    lines.append("")
    lines.append(
        "> **주의**: threshold-free diagnostic metric이다. "
        "threshold 확정이 아니며, p95/p99 계산이 아니다."
    )
    lines.append("")

    lines.append("## 8. 아직 하지 않은 것")
    lines.append("")
    not_done = [
        "threshold 결정 및 threshold-based hit-rate 계산",
        "p95/p99 기반 threshold 정책 수립",
        "stage2_holdout 최종 평가",
        "hard negative final manifest 확정",
        "training / retraining",
        "suppression / mask / ROI 수정",
    ]
    for item in not_done:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 9. 해석 제한")
    lines.append("")
    limits = [
        "본 결과는 stage1_dev filtered crop-level 범위에 한정된다.",
        "stage2_holdout 최종 검증 결과가 아니다.",
        "threshold 기반 성능 평가가 수행되지 않았다.",
        "patient-level은 majority_label1 rule 조건부 결과이며, 도메인 아티팩트 포함 가능성이 있다.",
        "top-k ranking은 threshold 확정이 아닌 ranking diagnostic metric이다.",
        "병변 성능 최종 결론이 아니다.",
    ]
    for limit in limits:
        lines.append(f"- {limit}")
    lines.append("")

    lines.append("## 10. 다음 선택지")
    lines.append("")
    next_options = [
        ("A", "threshold policy preflight",
         "p95/p99 기반 threshold 결정 preflight 설계 및 승인", "별도 승인 필요"),
        ("B", "stage2_holdout final evaluation preflight",
         "stage2_holdout 최종 평가 preflight 설계 및 승인", "별도 승인 필요"),
        ("C", "hard negative final manifest design",
         "hard negative 최종 manifest 설계", "별도 결정 필요"),
        ("D", "handoff 문서 최신화",
         "handoff 문서에 v1/v1 성능평가 결과 반영", "승인 불필요"),
    ]
    lines.append("| 선택지 | 이름 | 설명 | 필요 절차 |")
    lines.append("|---|---|---|---|")
    for opt_id, name, desc, approval in next_options:
        lines.append(f"| {opt_id} | {name} | {desc} | {approval} |")
    lines.append("")

    lines.append("## 11. 추천")
    lines.append("")
    lines.append(
        "1. **먼저 D. handoff 문서 최신화** "
        "— v1/v1 성능평가 결과를 handoff 문서에 반영한다."
    )
    lines.append(
        "2. 그 다음 **A. threshold policy preflight** 또는 "
        "**B. stage2_holdout final evaluation preflight**는 별도 승인 후 진행한다."
    )
    lines.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[MD] {OUT_MD}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 7.7 final performance closure report"
    )
    parser.add_argument("--run", action="store_true", help="실제 실행 (없으면 dry-run)")
    args = parser.parse_args()

    if not args.run:
        print("dry-run: --run 플래그를 붙여 실행하세요.")
        print(f"  PHASE74_JSON: {PHASE74_JSON}")
        print(f"  PHASE74_MD:   {PHASE74_MD}")
        print(f"  PHASE75_JSON: {PHASE75_JSON}")
        print(f"  PHASE75_MD:   {PHASE75_MD}")
        print(f"  PHASE76_CSV:  {PHASE76_CSV}")
        print(f"  PHASE76_JSON: {PHASE76_JSON}")
        print(f"  PHASE76_MD:   {PHASE76_MD}")
        print(f"  output:       {OUT_DIR}")
        return

    # 입력 파일 존재 확인
    for p in [PHASE74_JSON, PHASE74_MD, PHASE75_JSON, PHASE75_MD,
              PHASE76_CSV, PHASE76_JSON, PHASE76_MD]:
        if not p.exists():
            print(f"[ERROR] input file not found: {p}")
            sys.exit(1)

    # output guard (입력 검증 직후, 파일 읽기 전)
    check_output_guard()

    # 입력 JSON 로드 (검증용)
    with open(PHASE74_JSON, encoding="utf-8") as f:
        phase74_data = json.load(f)
    with open(PHASE75_JSON, encoding="utf-8") as f:
        phase75_data = json.load(f)
    with open(PHASE76_JSON, encoding="utf-8") as f:
        phase76_data = json.load(f)

    print("[로드] 입력 파일 JSON 3개 로드 완료")

    # Section 구성
    section_a = build_section_a()
    section_b = build_section_b()
    section_c = build_section_c()
    section_d = build_section_d()
    section_e = build_section_e()
    section_f = build_section_f()
    section_g = build_section_g()

    all_sections = [
        section_a, section_b, section_c, section_d,
        section_e, section_f, section_g,
    ]

    # 출력
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    write_csv(all_sections)
    write_json(phase74_data, phase75_data, phase76_data)
    write_md()

    print(f"\n=== Phase 7.7 완료 ===")
    print(f"최종 판정: {FINAL_STATUS}")
    print(f"출력 경로: {OUT_DIR}")


if __name__ == "__main__":
    main()
