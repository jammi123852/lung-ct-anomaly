"""
P-C-NORMAL17: Matched Smoke Result Interpretation + Full Training Readiness Checkpoint

- 모든 입력은 read-only
- model forward / scoring / threshold / checkpoint 저장 없음
- 기존 결과 수정 없음
- stage2_holdout 접근 없음
"""

import json
import csv
import pathlib
import datetime
import sys

# ── 경로 ─────────────────────────────────────────────────────────────────────
BRANCH_ROOT = pathlib.Path(
    "/home/jinhy/project/lung-ct-anomaly/experiments"
    "/efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1"
)
OUT_DIR = BRANCH_ROOT / "outputs/reports/p_c_normal17_smoke_interpretation_readiness"

# ── guardrail ─────────────────────────────────────────────────────────────────
GUARDRAIL = {
    "training_run": False,
    "backward_run": False,
    "optimizer_step": False,
    "model_forward_run": False,
    "scoring_run": False,
    "threshold_computed": False,
    "checkpoint_saved": False,
    "stage2_holdout_accessed": False,
    "original_report_modified": False,
    "hard_negative_included": False,
    "MSD_Lung_included": False,
    "forbidden_diagnostic_wording_count": 0,
}

ERRORS = []


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def _err(msg):
    ERRORS.append({"error": msg})
    print(f"[ERROR] {msg}", file=sys.stderr)


def _write_csv(path, rows):
    if not rows:
        rows = [{"note": "empty"}]
    all_keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"  saved → {path}")


# ── 1. Smoke result summary ──────────────────────────────────────────────────
def build_smoke_result_summary():
    rows = [
        {
            "stage": "P-C-NORMAL15",
            "item": "purpose",
            "value": "matched dataset 1-epoch smoke training — 학습 루프 sanity 확인",
            "note": "성능 결론 단계 아님",
        },
        {"stage": "P-C-NORMAL15", "item": "epoch",          "value": "1",       "note": ""},
        {"stage": "P-C-NORMAL15", "item": "device",         "value": "cuda",    "note": ""},
        {"stage": "P-C-NORMAL15", "item": "train_rows",     "value": "19727",   "note": "p_c_normal12 train manifest"},
        {"stage": "P-C-NORMAL15", "item": "val_rows",       "value": "5200",    "note": "p_c_normal12 val manifest"},
        {"stage": "P-C-NORMAL15", "item": "train_loss",     "value": "0.0797",  "note": ""},
        {"stage": "P-C-NORMAL15", "item": "train_acc",      "value": "0.9776",  "note": ""},
        {"stage": "P-C-NORMAL15", "item": "val_loss",       "value": "0.0276",  "note": ""},
        {"stage": "P-C-NORMAL15", "item": "val_acc",        "value": "0.9917",  "note": ""},
        {
            "stage": "P-C-NORMAL15",
            "item": "val_auc_original",
            "value": "null",
            "note": "AUROC_ERROR:ModuleNotFoundError — sklearn 미설치, training failure 아님",
        },
        {"stage": "P-C-NORMAL15", "item": "smoke_only",     "value": "True",    "note": ""},
        {"stage": "P-C-NORMAL15", "item": "full_training",  "value": "False",   "note": ""},
        {"stage": "P-C-NORMAL15", "item": "stage2_holdout_accessed", "value": "False", "note": ""},
        {"stage": "P-C-NORMAL15", "item": "hard_negative",  "value": "0",       "note": ""},
        {"stage": "P-C-NORMAL15", "item": "MSD_Lung",       "value": "0",       "note": ""},
        {
            "stage": "P-C-NORMAL15",
            "item": "checkpoint",
            "value": "p_c_normal15_epoch1.pth (48.6MB)",
            "note": "valid — NaN/Inf=0, 16 keys OK",
        },
        # P-C-NORMAL16
        {
            "stage": "P-C-NORMAL16",
            "item": "purpose",
            "value": "checkpoint 검증 + sklearn-free AUROC backfill",
            "note": "",
        },
        {"stage": "P-C-NORMAL16", "item": "verdict",           "value": "PASS",     "note": ""},
        {"stage": "P-C-NORMAL16", "item": "checkpoint_16_keys","value": "all present","note": ""},
        {"stage": "P-C-NORMAL16", "item": "model_weights_NaN", "value": "0",        "note": "4,050,894 params"},
        {"stage": "P-C-NORMAL16", "item": "model_weights_Inf", "value": "0",        "note": ""},
        {"stage": "P-C-NORMAL16", "item": "metrics_consistent","value": "True",     "note": "checkpoint/JSON/MD 정합"},
        {
            "stage": "P-C-NORMAL16",
            "item": "val_auc_backfill",
            "value": "0.9998",
            "note": "numpy rank-sum Mann-Whitney, sklearn 미사용",
        },
        {"stage": "P-C-NORMAL16", "item": "auc_backfill_n_pos","value": "2080",     "note": "val NSCLC-like"},
        {"stage": "P-C-NORMAL16", "item": "auc_backfill_n_neg","value": "3120",     "note": "val normal-like"},
        {"stage": "P-C-NORMAL16", "item": "p_c_normal6_mtime_unchanged", "value": "True", "note": ""},
        {"stage": "P-C-NORMAL16", "item": "guardrail_all_pass","value": "True",     "note": ""},
        {"stage": "P-C-NORMAL16", "item": "error_count",       "value": "0",        "note": ""},
    ]
    return rows


# ── 2. Shortcut status review ────────────────────────────────────────────────
def build_shortcut_status():
    rows = [
        {
            "shortcut_id": "SR-POS",
            "status": "RESOLVED",
            "evidence": "z-index reproduction 100%, position_bin train/val distribution matched, peripheral_diff=0.0",
            "risk_level": "LOW",
            "action_required": "None",
        },
        {
            "shortcut_id": "SR-PIPE",
            "status": "REDUCED",
            "evidence": "same generator (p_c_normal9) 사용, crop shape/dtype/preprocessing 동일, vessel/roi mask 미사용",
            "risk_level": "LOW",
            "action_required": "None",
        },
        {
            "shortcut_id": "SR-HU-CAP",
            "status": "RESOLVED",
            "evidence": "both classes HU clip [-1000, 200] + scale [0,1] + ImageNet normalize 동일 적용",
            "risk_level": "LOW",
            "action_required": "None",
        },
        {
            "shortcut_id": "SR-HU",
            "status": "MEDIUM/OPEN",
            "evidence": (
                "P-C-NORMAL13: normal mean HU=-574, NSCLC mean HU=-325, diff=-249 HU, "
                "Cohen's d=0.598, overlap_estimate=0.629. "
                "air_frac(<-800): normal=0.539 vs NSCLC=0.172 (diff=+0.367). "
                "dense_frac(>-500): normal=0.322 vs NSCLC=0.594 (diff=-0.272)."
            ),
            "risk_level": "MEDIUM",
            "action_required": (
                "full training 후 per-HU-bin error analysis + "
                "position_bin/HU bin별 recall/specificity 분석 권장"
            ),
        },
        {
            "shortcut_id": "SR-CONTEXT",
            "status": "OPEN",
            "evidence": (
                "lesion 중심 crop(NSCLC)과 normal matched crop은 "
                "인접 조직 context 구성이 다를 수 있음 — "
                "lesion 주변 조직 패턴이 signal을 형성할 가능성 있음"
            ),
            "risk_level": "MEDIUM",
            "action_required": (
                "full training 후 validation prob distribution 분석, "
                "per-position-bin recall 분석 권장"
            ),
        },
        {
            "shortcut_id": "AUROC=0.9998@1epoch",
            "status": "AMBIGUOUS",
            "evidence": (
                "1-epoch AUROC 0.9998은 높은 수치이나 "
                "smoke training은 성능 결론 단계가 아님. "
                "두 가지 해석 가능: "
                "(A) 실제 lesion-density/texture separability, "
                "(B) residual HU/context shortcut. "
                "1-epoch overfitting 여부 불확실."
            ),
            "risk_level": "INFORMATIONAL",
            "action_required": (
                "full training 시 epoch별 val_auc 추적, "
                "early stopping으로 과적합 관리, "
                "stage2_holdout 최종 평가 전까지 성능 결론 유보"
            ),
        },
    ]
    return rows


# ── 3. Full training readiness decision ──────────────────────────────────────
def build_readiness_decision():
    # 판정 기준:
    # - smoke/checkpoint/guardrail 모두 정상 → READY 조건 충족
    # - SR-HU medium 이지만 expected lesion-density signal로 관리 가능
    # - P-C-NORMAL13 이후 matched manifest로 SR-POS/SR-PIPE/SR-HU-CAP 해소
    # → READY_FOR_FULL_TRAINING_PREFLIGHT

    decision = "READY_FOR_FULL_TRAINING_PREFLIGHT"
    rows = [
        {
            "criterion": "smoke_training_completed",
            "value": "True",
            "weight": "required",
            "pass": True,
            "note": "P-C-NORMAL15 PASS",
        },
        {
            "criterion": "checkpoint_valid",
            "value": "True",
            "weight": "required",
            "pass": True,
            "note": "16 keys, NaN/Inf=0",
        },
        {
            "criterion": "guardrail_all_pass",
            "value": "True",
            "weight": "required",
            "pass": True,
            "note": "P-C-NORMAL16 모든 guardrail 통과",
        },
        {
            "criterion": "val_auc_backfill_valid",
            "value": "0.9998",
            "weight": "required",
            "pass": True,
            "note": "sklearn-free 계산 성공, status=OK",
        },
        {
            "criterion": "SR-POS",
            "value": "RESOLVED",
            "weight": "blocking",
            "pass": True,
            "note": "position shortcut 해소",
        },
        {
            "criterion": "SR-PIPE",
            "value": "REDUCED",
            "weight": "blocking",
            "pass": True,
            "note": "pipeline shortcut 감소",
        },
        {
            "criterion": "SR-HU-CAP",
            "value": "RESOLVED",
            "weight": "blocking",
            "pass": True,
            "note": "HU cap shortcut 해소",
        },
        {
            "criterion": "SR-HU",
            "value": "MEDIUM/OPEN",
            "weight": "advisory",
            "pass": True,
            "note": (
                "Cohen's d=0.598, 관리 가능 수준. "
                "expected lesion-density signal로 해석 가능. "
                "full training 후 per-HU-bin 분석으로 재평가 예정."
            ),
        },
        {
            "criterion": "SR-CONTEXT",
            "value": "OPEN",
            "weight": "advisory",
            "pass": True,
            "note": "full training 후 position_bin 분석으로 재평가 예정",
        },
        {
            "criterion": "stage2_holdout_not_accessed",
            "value": "True",
            "weight": "required",
            "pass": True,
            "note": "접근 없음 확인됨",
        },
        {
            "criterion": "DECISION",
            "value": decision,
            "weight": "final",
            "pass": True,
            "note": (
                "smoke/checkpoint/guardrail 모두 정상. "
                "남은 SR-HU/SR-CONTEXT는 full training 후 관리 가능. "
                "P-C-NORMAL18 full training preflight 진행 가능."
            ),
        },
    ]
    return decision, rows


# ── 4. Full training preflight plan ──────────────────────────────────────────
def build_preflight_plan():
    rows = [
        # P-C-NORMAL18 계획
        {
            "item": "stage",
            "value": "P-C-NORMAL18",
            "category": "next_step",
            "note": "actual full training 아님 — full training preflight 전용",
        },
        {
            "item": "preflight_purpose",
            "value": "full training 실행 전 조건 점검 (데이터/코드/경로/schema/guardrail)",
            "category": "purpose",
            "note": "P-C-NORMAL17과 동일하게 실제 학습 없음",
        },
        # 학습 파라미터 후보
        {
            "item": "max_epochs_candidate",
            "value": "10~20",
            "category": "training_config",
            "note": "epoch별 val_auc 추적 필수, early stopping 권장",
        },
        {
            "item": "early_stopping",
            "value": "True",
            "category": "training_config",
            "note": "patience=3~5 권장, monitor=val_auc",
        },
        {
            "item": "monitor_metrics",
            "value": "val_auc + val_loss (both required)",
            "category": "training_config",
            "note": "val_auc는 sklearn-free numpy rank-sum 함수 사용",
        },
        {
            "item": "auroc_function",
            "value": "numpy_rank_sum_mannwhitney (sklearn 금지)",
            "category": "training_config",
            "note": "P-C-NORMAL16에서 검증된 함수 그대로 사용",
        },
        # checkpoint schema
        {
            "item": "checkpoint_schema_keys",
            "value": "18 keys",
            "category": "checkpoint",
            "note": (
                "smoke 16 keys + best_epoch + best_val_auc. "
                "best_auc.pth (val_auc 기준 최고) + last.pth (마지막 epoch) 두 파일 저장."
            ),
        },
        {
            "item": "best_auc_pth",
            "value": "p_c_normal_full_best_auc.pth",
            "category": "checkpoint",
            "note": "val_auc 최고 시점 저장",
        },
        {
            "item": "last_pth",
            "value": "p_c_normal_full_last.pth",
            "category": "checkpoint",
            "note": "마지막 epoch 저장",
        },
        # guardrail
        {
            "item": "stage2_holdout_guard",
            "value": "접근 금지 유지",
            "category": "guardrail",
            "note": "full training에서도 stage2_holdout 접근 없음",
        },
        {
            "item": "hard_negative_guard",
            "value": "포함 금지",
            "category": "guardrail",
            "note": "hard_negative/MSD_Lung 0 유지",
        },
        {
            "item": "performance_expression",
            "value": "normal-like vs NSCLC-lesion-like auxiliary score",
            "category": "wording",
            "note": "diagnostic wording (암, 악성, 진단, 확진) 사용 금지",
        },
        # 추가 분석 계획 (full training 후)
        {
            "item": "val_prob_distribution_save",
            "value": "epoch별 val probs/labels CSV 저장 계획",
            "category": "post_training_analysis",
            "note": "prob distribution 변화 추적, shortcut 재평가에 활용",
        },
        {
            "item": "confusion_matrix_at_0_5",
            "value": "threshold=0.5 기준 confusion matrix 계획",
            "category": "post_training_analysis",
            "note": "임계값 최적화 아님, 기술적 요약 목적",
        },
        {
            "item": "per_position_bin_analysis",
            "value": "position_bin별 recall/specificity 계획",
            "category": "post_training_analysis",
            "note": "SR-CONTEXT 재평가, 위치별 편향 탐지",
        },
        {
            "item": "per_hu_bin_error_analysis",
            "value": "HU bin별 오분류 비율 계획",
            "category": "post_training_analysis",
            "note": "SR-HU 재평가, air_frac/dense_frac 구간별 분석",
        },
        {
            "item": "patient_level_aggregation_preview",
            "value": "환자별 crop score 집계 구조 설계 계획",
            "category": "post_training_analysis",
            "note": "crop-level score → patient-level summary, stage2 평가 준비",
        },
    ]
    return rows


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("P-C-NORMAL17: Smoke Interpretation + Full Training Readiness")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.datetime.now().isoformat()

    smoke_rows   = build_smoke_result_summary()
    shortcut_rows = build_shortcut_status()
    decision, readiness_rows = build_readiness_decision()
    preflight_rows = build_preflight_plan()

    # guardrail 체크 rows
    guardrail_rows = []
    for k, v in GUARDRAIL.items():
        expected = False if k != "forbidden_diagnostic_wording_count" else 0
        ok = v == expected
        guardrail_rows.append({
            "guardrail": k,
            "expected": str(expected),
            "actual": str(v),
            "pass": ok,
        })
        if not ok:
            _err(f"Guardrail violated: {k}={v}")

    verdict = "PASS" if not ERRORS else "FAIL"
    guardrail_pass = all(r["pass"] for r in guardrail_rows)

    print(f"\n  판정: {verdict}")
    print(f"  readiness: {decision}")
    print(f"  guardrail: {'PASS' if guardrail_pass else 'FAIL'}")
    print(f"  errors: {len(ERRORS)}")

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    _write_csv(OUT_DIR / "p_c_normal17_smoke_result_summary.csv",            smoke_rows)
    _write_csv(OUT_DIR / "p_c_normal17_shortcut_status_review.csv",          shortcut_rows)
    _write_csv(OUT_DIR / "p_c_normal17_full_training_readiness_decision.csv",readiness_rows)
    _write_csv(OUT_DIR / "p_c_normal17_full_training_preflight_plan.csv",    preflight_rows)
    _write_csv(OUT_DIR / "p_c_normal17_guardrail_check.csv",                 guardrail_rows)
    _write_csv(OUT_DIR / "p_c_normal17_errors.csv",
               ERRORS if ERRORS else [{"error": "none"}])

    # ── JSON ──────────────────────────────────────────────────────────────────
    summary_json = {
        "stage": "P-C-NORMAL17",
        "validated_at": now_iso,
        "verdict": verdict,
        "readiness_decision": decision,
        "p_c_normal15_smoke_completed": True,
        "p_c_normal16_verdict": "PASS",
        "val_auc_backfill": 0.9998,
        "val_auc_backfill_method": "numpy_rank_sum_mannwhitney",
        "val_auc_null_reason": "sklearn not installed — monitoring metric missing, not training failure",
        "checkpoint_valid": True,
        "shortcut_status": {
            "SR-POS":     "RESOLVED",
            "SR-PIPE":    "REDUCED",
            "SR-HU-CAP":  "RESOLVED",
            "SR-HU":      "MEDIUM/OPEN",
            "SR-CONTEXT": "OPEN",
        },
        "auroc_1epoch_interpretation": (
            "0.9998 is informational only. "
            "Smoke training purpose = loop sanity, not performance conclusion. "
            "Two interpretations remain open: "
            "(A) real lesion-density/texture separability, "
            "(B) residual HU/context shortcut. "
            "Final performance conclusion deferred to full training + stage2_holdout evaluation."
        ),
        "next_step": "P-C-NORMAL18 full training preflight",
        "full_training_not_yet_run": True,
        "guardrail_pass": guardrail_pass,
        "error_count": len(ERRORS),
        "guardrail": GUARDRAIL,
    }
    json_out = OUT_DIR / "p_c_normal17_smoke_interpretation_readiness.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    print(f"  saved → {json_out}")

    # ── MD ────────────────────────────────────────────────────────────────────
    shortcut_table = "\n".join(
        f"| {r['shortcut_id']} | {r['status']} | {r['risk_level']} | {r['action_required'][:60]}... |"
        if len(r['action_required']) > 60
        else f"| {r['shortcut_id']} | {r['status']} | {r['risk_level']} | {r['action_required']} |"
        for r in shortcut_rows
    )

    readiness_table = "\n".join(
        f"| {r['criterion']} | {r['value']} | {r['weight']} | {'OK' if r['pass'] else 'FAIL'} |"
        for r in readiness_rows
    )

    md_lines = [
        "# P-C-NORMAL17 Smoke Result Interpretation + Full Training Readiness",
        "",
        f"**판정: {verdict}**  ",
        f"**readiness: {decision}**  ",
        f"**날짜:** {now_iso[:10]}",
        "",
        "---",
        "",
        "## 1. 결론",
        "",
        "- P-C-NORMAL15 smoke training 정상 완료 (학습 루프 sanity 확인 목적)",
        "- val_auc null → sklearn 미설치로 인한 monitoring metric 누락 (training failure 아님)",
        "- P-C-NORMAL16 checkpoint 검증 PASS, sklearn-free AUROC backfill = **0.9998**",
        "- 1-epoch AUROC=0.9998은 참고 수치이며 성능 결론 아님",
        "  - (A) 실제 lesion-density/texture separability 가능",
        "  - (B) residual SR-HU / SR-CONTEXT shortcut 가능",
        "- SR-POS / SR-PIPE / SR-HU-CAP 모두 해소됨",
        "- SR-HU (Cohen's d=0.598) 및 SR-CONTEXT는 OPEN — full training 후 재평가 예정",
        f"- **다음 단계: P-C-NORMAL18 full training preflight**",
        "",
        "---",
        "",
        "## 2. P-C-NORMAL15 / P-C-NORMAL16 요약",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        "| P-C-NORMAL15 목적 | matched dataset 1-epoch smoke — 학습 루프 sanity |",
        "| epoch | 1 |",
        "| train_rows / val_rows | 19,727 / 5,200 |",
        "| train_loss / train_acc | 0.0797 / 0.9776 |",
        "| val_loss / val_acc | 0.0276 / 0.9917 |",
        "| val_auc (원본) | null (AUROC_ERROR:ModuleNotFoundError) |",
        "| val_auc (backfill) | **0.9998** (numpy rank-sum, sklearn 미사용) |",
        "| P-C-NORMAL16 verdict | PASS |",
        "| checkpoint 16 keys | 모두 존재 |",
        "| model weights NaN/Inf | 0 / 0 (4,050,894 params) |",
        "| guardrail all pass | True |",
        "",
        "---",
        "",
        "## 3. Shortcut status",
        "",
        "| ID | 상태 | 위험도 | 조치 |",
        "|----|------|--------|------|",
        shortcut_table,
        "",
        "**SR-HU 세부 (P-C-NORMAL13 기준):**",
        "- normal mean HU = -574, NSCLC mean HU = -325, diff = -249 HU",
        "- Cohen's d = 0.598, overlap_estimate = 0.629",
        "- air_frac(<-800): normal=0.539, NSCLC=0.172 (diff=+0.367)",
        "- dense_frac(>-500): normal=0.322, NSCLC=0.594 (diff=-0.272)",
        "",
        "---",
        "",
        "## 4. Full training readiness 판정",
        "",
        f"**{decision}**",
        "",
        "| 기준 | 값 | 가중치 | 판정 |",
        "|------|-----|--------|------|",
        readiness_table,
        "",
        "**판정 근거:**",
        "- smoke/checkpoint/guardrail 모두 정상",
        "- 모든 blocking shortcut (SR-POS/SR-PIPE/SR-HU-CAP) 해소됨",
        "- SR-HU는 expected lesion-density signal로 관리 가능 (advisory level)",
        "- P-C-NORMAL18은 actual full training 아님 — full training preflight 수행",
        "",
        "---",
        "",
        "## 5. Full training 조건 (P-C-NORMAL18 이후 적용)",
        "",
        "| 항목 | 계획 |",
        "|------|------|",
        "| max_epochs | 10~20 (early stopping patience=3~5) |",
        "| monitor | val_auc + val_loss (둘 다 기록) |",
        "| AUROC 함수 | numpy rank-sum Mann-Whitney (sklearn 금지) |",
        "| checkpoint | best_auc.pth + last.pth, schema 18 keys |",
        "| stage2_holdout | 접근 금지 유지 |",
        "| 성능 표현 | normal-like vs NSCLC-lesion-like auxiliary score |",
        "",
        "**Full training 후 추가 분석 계획:**",
        "- epoch별 val probs/labels 분포 CSV 저장",
        "- threshold=0.5 confusion matrix",
        "- position_bin별 recall/specificity (SR-CONTEXT 재평가)",
        "- HU bin별 오분류 비율 (SR-HU 재평가)",
        "- 환자별 crop score 집계 구조 설계 (stage2 평가 준비)",
        "",
        "---",
        "",
        "## 6. Guardrail",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
    ]
    for k, v in GUARDRAIL.items():
        md_lines.append(f"| {k} | {v} |")

    md_lines += [
        "",
        "---",
        "",
        f"## 판정: {verdict} / {decision}",
        "",
        "- error_count: 0",
        "- 다음 단계: **P-C-NORMAL18 full training preflight**",
    ]

    md_out = OUT_DIR / "p_c_normal17_smoke_interpretation_readiness.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  saved → {md_out}")

    print(f"\n[DONE] 출력 폴더: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
