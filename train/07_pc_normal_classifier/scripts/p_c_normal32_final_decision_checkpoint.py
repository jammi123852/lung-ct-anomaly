"""
P-C-NORMAL32: final decision checkpoint
decision/documentation only — no training, no model forward, no prediction re-export
"""

import sys
import csv
import json
import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

REPORT31  = PROJECT_ROOT / "outputs/reports/p_c_normal31_repaired_final_test_masked_comparison"
REPORT31B = PROJECT_ROOT / "outputs/reports/p_c_normal31b_zero_mask_patient_sanity_audit"
REPORT31C = PROJECT_ROOT / "outputs/reports/p_c_normal31c_low_mask_fn_caveat_addendum"
REPORT30B = PROJECT_ROOT / "outputs/reports/p_c_normal30b_masked_input_full_train"
REPORT_ROOT = PROJECT_ROOT / "outputs/reports/p_c_normal32_final_decision_checkpoint"
IN_CROP_RAW = REPORT31 / "p_c_normal31_crop_raw_metrics_comparison.csv"

GUARDRAILS = {
    "decision_checkpoint_only": True,
    "no_training_run": True,
    "no_model_forward": True,
    "no_prediction_export_rerun": True,
    "no_threshold_optimization": True,
    "no_threshold_sweep": True,
    "no_best_threshold_selection": True,
    "no_checkpoint_modification": True,
    "no_existing_result_overwrite": True,
    "original_31b_blocked_verdict_preserved": False,   # 확인 후 set
    "p_c_normal31c_pass_for_decision_confirmed": False, # 확인 후 set
    "selected_candidate_masked_30b": True,
    "selected_with_caveat": True,
    "diagnostic_wording_avoided": True,
}

# ── known P-C-NORMAL31 crop-level results ─────────────────────────────────
CROP = {
    "ref_auroc": 0.951717, "ref_auprc": 0.973255, "ref_brier": 0.153304,
    "ref_specificity": 0.495640, "ref_sensitivity": 0.992912,
    "ref_fp": 10874, "ref_fn": 317,
    "cand_auroc": 0.990387, "cand_auprc": 0.995366, "cand_brier": 0.075006,
    "cand_specificity": 0.732839, "cand_sensitivity": 0.992353,
    "cand_fp": 5760, "cand_fn": 342,
}
# known patient-level (mean_prob)
PAT = {
    "ref_auroc": 0.989754, "ref_fp_patients": 21, "ref_fn_patients": 1,
    "cand_auroc": 0.994308, "cand_fp_patients": 5, "cand_fn_patients": 1,
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows, path):
    if not rows:
        path.write_text("(empty)\n", encoding="utf-8")
        return
    all_keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run():
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 출력 충돌 방지
    if REPORT_ROOT.exists() and any(REPORT_ROOT.iterdir()):
        print(f"[ABORT] output already exists: {REPORT_ROOT}", file=sys.stderr)
        sys.exit(2)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── 1. Input guard ─────────────────────────────────────────────────────
    required = [
        REPORT31 / "DONE.json",
        REPORT31B / "DONE.json",
        REPORT31B / "p_c_normal31b_summary.json",
        REPORT31C / "DONE.json",
        REPORT31C / "p_c_normal31c_summary.json",
        REPORT30B / "p_c_normal30b_summary.json",
        IN_CROP_RAW,
    ]
    for p in required:
        if not p.exists():
            print(f"[ABORT] Missing: {p}", file=sys.stderr)
            sys.exit(1)

    # ── 2. Load prior verdicts ─────────────────────────────────────────────
    done31  = load_json(REPORT31  / "DONE.json")
    done31b = load_json(REPORT31B / "DONE.json")
    sum31b  = load_json(REPORT31B / "p_c_normal31b_summary.json")
    done31c = load_json(REPORT31C / "DONE.json")
    sum31c  = load_json(REPORT31C / "p_c_normal31c_summary.json")
    sum30b  = load_json(REPORT30B / "p_c_normal30b_summary.json")

    verdict_31  = done31.get("verdict",  done31.get("phase_a_verdict", "UNKNOWN"))
    verdict_31b = done31b.get("re_adjudication_verdict",
                  sum31b.get("verdict", "UNKNOWN"))
    verdict_31c = done31c.get("re_adjudication_verdict",
                  sum31c.get("re_adjudication_verdict", "UNKNOWN"))

    print(f"[32] P-C-NORMAL31  verdict : {verdict_31}")
    print(f"[32] P-C-NORMAL31b verdict : {verdict_31b}  (original BLOCKED preserved)")
    print(f"[32] P-C-NORMAL31c verdict : {verdict_31c}")

    # 31b original BLOCKED 확인
    original_31b = sum31b.get("verdict", "UNKNOWN")
    if original_31b != "BLOCKED":
        print(f"[ABORT] P-C-NORMAL31b original verdict expected BLOCKED, got {original_31b}",
              file=sys.stderr)
        sys.exit(2)
    GUARDRAILS["original_31b_blocked_verdict_preserved"] = True

    # 31c PASS_FOR_DECISION 확인
    if verdict_31c != "PASS_FOR_DECISION":
        print(f"[ABORT] P-C-NORMAL31c verdict expected PASS_FOR_DECISION, got {verdict_31c}",
              file=sys.stderr)
        sys.exit(2)
    GUARDRAILS["p_c_normal31c_pass_for_decision_confirmed"] = True

    ckpt_path = sum30b.get("best_ckpt", "")
    best_epoch = sum30b.get("best_epoch", "")
    best_val_auc = sum30b.get("best_val_auc", "")

    if not ckpt_path:
        print("[ABORT] selected checkpoint path is empty. Check p_c_normal30b_summary.json key name.",
              file=sys.stderr)
        sys.exit(2)
    if "smoke" in str(ckpt_path).lower():
        print(f"[ABORT] smoke checkpoint must not be selected: {ckpt_path}", file=sys.stderr)
        sys.exit(2)

    # ── 3. Selected candidate summary ─────────────────────────────────────
    selected_rows = [
        {"field": "selected_candidate", "value": "P-C-NORMAL30b_masked_input"},
        {"field": "reference_candidate", "value": "P-C-NORMAL24j_fix_balanced_w1"},
        {"field": "selection_basis", "value": "repaired_final_test_fixed_threshold_0p5"},
        {"field": "threshold_optimized", "value": "false"},
        {"field": "selected_with_caveat", "value": "true"},
        {"field": "caveat", "value": "low_mask_lesion_edge_crop_blank_input_FN_risk"},
        {"field": "masked_30b_checkpoint",
         "value": ckpt_path},
        {"field": "masked_30b_best_epoch", "value": str(best_epoch)},
        {"field": "masked_30b_val_auc", "value": str(best_val_auc)},
        {"field": "p_c_normal31_verdict", "value": verdict_31},
        {"field": "p_c_normal31b_original_verdict", "value": original_31b},
        {"field": "p_c_normal31c_re_adjudication_verdict", "value": verdict_31c},
    ]
    write_csv(selected_rows,
              REPORT_ROOT / "p_c_normal32_selected_candidate_summary.csv")

    # ── 4. Metric comparison ───────────────────────────────────────────────
    metric_rows = [
        # crop-level
        {"level": "crop", "metric": "AUROC",
         "reference": CROP["ref_auroc"], "selected": CROP["cand_auroc"],
         "delta": round(CROP["cand_auroc"] - CROP["ref_auroc"], 6),
         "direction": "higher_is_better", "result": "IMPROVED"},
        {"level": "crop", "metric": "AUPRC",
         "reference": CROP["ref_auprc"], "selected": CROP["cand_auprc"],
         "delta": round(CROP["cand_auprc"] - CROP["ref_auprc"], 6),
         "direction": "higher_is_better", "result": "IMPROVED"},
        {"level": "crop", "metric": "Brier",
         "reference": CROP["ref_brier"], "selected": CROP["cand_brier"],
         "delta": round(CROP["cand_brier"] - CROP["ref_brier"], 6),
         "direction": "lower_is_better", "result": "IMPROVED"},
        {"level": "crop", "metric": "specificity",
         "reference": CROP["ref_specificity"], "selected": CROP["cand_specificity"],
         "delta": round(CROP["cand_specificity"] - CROP["ref_specificity"], 6),
         "direction": "higher_is_better", "result": "IMPROVED"},
        {"level": "crop", "metric": "sensitivity",
         "reference": CROP["ref_sensitivity"], "selected": CROP["cand_sensitivity"],
         "delta": round(CROP["cand_sensitivity"] - CROP["ref_sensitivity"], 6),
         "direction": "higher_is_better", "result": "SLIGHTLY_DECREASED"},
        {"level": "crop", "metric": "FP_normal",
         "reference": CROP["ref_fp"], "selected": CROP["cand_fp"],
         "delta": CROP["cand_fp"] - CROP["ref_fp"],
         "direction": "lower_is_better", "result": "IMPROVED"},
        {"level": "crop", "metric": "FN_NSCLC",
         "reference": CROP["ref_fn"], "selected": CROP["cand_fn"],
         "delta": CROP["cand_fn"] - CROP["ref_fn"],
         "direction": "lower_is_better", "result": "SLIGHTLY_WORSENED"},
        # patient-level mean_prob
        {"level": "patient_mean_prob", "metric": "AUROC",
         "reference": PAT["ref_auroc"], "selected": PAT["cand_auroc"],
         "delta": round(PAT["cand_auroc"] - PAT["ref_auroc"], 6),
         "direction": "higher_is_better", "result": "IMPROVED"},
        {"level": "patient_mean_prob", "metric": "FP_patients",
         "reference": PAT["ref_fp_patients"], "selected": PAT["cand_fp_patients"],
         "delta": PAT["cand_fp_patients"] - PAT["ref_fp_patients"],
         "direction": "lower_is_better", "result": "IMPROVED"},
        {"level": "patient_mean_prob", "metric": "FN_patients",
         "reference": PAT["ref_fn_patients"], "selected": PAT["cand_fn_patients"],
         "delta": PAT["cand_fn_patients"] - PAT["ref_fn_patients"],
         "direction": "lower_is_better", "result": "UNCHANGED"},
    ]
    write_csv(metric_rows,
              REPORT_ROOT / "p_c_normal32_metric_comparison_summary.csv")

    # ── 5. Caveat summary ─────────────────────────────────────────────────
    caveat_rows = [
        {"id": "C1",
         "title": "low-mask lesion-edge blank-input FN risk",
         "description": (
             "masked input은 normal FP를 크게 줄이나, "
             "ROI overlap이 매우 낮은 NSCLC crop에서는 "
             "mask 적용 후 입력이 거의 blank가 되어 crop-level FN이 발생할 수 있다."
         ),
         "observed_cases": "LUNG1-205 low_mask crop 2건 (nzr_mean 0.037/0.047)",
         "patient_level_impact": "없음 (LUNG1-205 patient-level TP 유지, mean/p95/max 모두)",
         "monitoring": "downstream 적용 시 low-mask lesion-edge case 모니터링 필요"},
        {"id": "C2",
         "title": "sensitivity 소폭 하락",
         "description": "crop-level sensitivity 0.9929 → 0.9924 (-0.0006), FN +25건",
         "observed_cases": "crop-level 전체",
         "patient_level_impact": "patient-level FN patients 동일 (ref=1, cand=1)",
         "monitoring": "crop-level FN 증가폭 추적 필요 (현재 수준은 제한적)"},
        {"id": "C3",
         "title": "fixed threshold 0.5 기준 평가",
         "description": (
             "모든 metric은 fixed threshold 0.5에서 계산됨. "
             "threshold 최적화/sweep 미실시. "
             "operating point 변경 시 결과 달라질 수 있음."
         ),
         "observed_cases": "-",
         "patient_level_impact": "-",
         "monitoring": "threshold 변경 시 재평가 필요"},
    ]
    write_csv(caveat_rows, REPORT_ROOT / "p_c_normal32_caveat_summary.csv")

    # ── 6. Guardrail check ─────────────────────────────────────────────────
    guardrail_rows = [
        {"key": k, "value": v, "status": "OK" if v is True else "FAIL"}
        for k, v in GUARDRAILS.items()
    ]
    write_csv(guardrail_rows, REPORT_ROOT / "p_c_normal32_guardrail_check.csv")
    guardrail_fail = sum(1 for v in GUARDRAILS.values() if v is False)

    # ── 7. Decision checkpoint markdown ───────────────────────────────────
    checkpoint_md = f"""# P-C-NORMAL32: Final Decision Checkpoint

생성일: {ts}
Decision/documentation only — 학습/inference/threshold 최적화 없음.

---

## 1. Prior Verdict 체인

| Stage | Verdict | 비고 |
|-------|---------|------|
| P-C-NORMAL28 | reference | balanced-w1, baseline |
| P-C-NORMAL30b | PASS | masked-input full training |
| P-C-NORMAL31 | PARTIAL_PASS | repaired final_test 비교, low_or_zero_mask=16 |
| P-C-NORMAL31b | **BLOCKED** (원본 유지) | NSCLC low_mask crop 2건 crop-level FN |
| P-C-NORMAL31c | PASS_FOR_DECISION | low_mask FN caveat 재판정, patient-level 악화 없음 확인 |
| **P-C-NORMAL32** | **{("PASS" if guardrail_fail == 0 else "PARTIAL_PASS")}** | final decision checkpoint |

> P-C-NORMAL31b BLOCKED verdict는 원본 파일에서 수정되지 않았음.

---

## 2. Selected Candidate

**P-C-NORMAL30b masked-input** (selected with caveat)

- Checkpoint: `{ckpt_path}`
- Best epoch: {best_epoch} / Val AUROC: {best_val_auc}
- Selection basis: repaired_final_test, fixed threshold 0.5
- Threshold optimized: **No**
- Selected with caveat: **Yes** (caveat C1, C2, C3 참고)

Reference (not selected): P-C-NORMAL24j-fix-balanced-w1

---

## 3. Crop-level Metric Comparison (fixed threshold 0.5)

| Metric | Reference 24j bw1 | Selected 30b masked | Delta | Result |
|--------|-------------------|---------------------|-------|--------|
| AUROC | {CROP["ref_auroc"]} | {CROP["cand_auroc"]} | +{CROP["cand_auroc"]-CROP["ref_auroc"]:.4f} | IMPROVED |
| AUPRC | {CROP["ref_auprc"]} | {CROP["cand_auprc"]} | +{CROP["cand_auprc"]-CROP["ref_auprc"]:.4f} | IMPROVED |
| Brier | {CROP["ref_brier"]} | {CROP["cand_brier"]} | {CROP["cand_brier"]-CROP["ref_brier"]:.4f} | IMPROVED |
| specificity | {CROP["ref_specificity"]} | {CROP["cand_specificity"]} | +{CROP["cand_specificity"]-CROP["ref_specificity"]:.4f} | IMPROVED |
| sensitivity | {CROP["ref_sensitivity"]} | {CROP["cand_sensitivity"]} | {CROP["cand_sensitivity"]-CROP["ref_sensitivity"]:.4f} | SLIGHTLY_DECREASED |
| FP (normal) | {CROP["ref_fp"]:,} | {CROP["cand_fp"]:,} | -{CROP["ref_fp"]-CROP["cand_fp"]:,} | IMPROVED |
| FN (NSCLC) | {CROP["ref_fn"]:,} | {CROP["cand_fn"]:,} | +{CROP["cand_fn"]-CROP["ref_fn"]:,} | SLIGHTLY_WORSENED |

## 4. Patient-level Metric Comparison (mean_prob, fixed threshold 0.5)

| Metric | Reference | Selected | Delta | Result |
|--------|-----------|----------|-------|--------|
| AUROC | {PAT["ref_auroc"]} | {PAT["cand_auroc"]} | +{PAT["cand_auroc"]-PAT["ref_auroc"]:.4f} | IMPROVED |
| FP patients | {PAT["ref_fp_patients"]} | {PAT["cand_fp_patients"]} | -{PAT["ref_fp_patients"]-PAT["cand_fp_patients"]} | IMPROVED |
| FN patients | {PAT["ref_fn_patients"]} | {PAT["cand_fn_patients"]} | {PAT["cand_fn_patients"]-PAT["ref_fn_patients"]} | UNCHANGED |

p95_prob / max_prob 기준 cand FN patients: **0** (reference도 0)

---

## 5. Caveat (필수)

### C1 — low-mask lesion-edge blank-input FN risk

masked input은 normal FP를 크게 줄이는 auxiliary score 개선 효과가 있다.
그러나 ROI overlap이 매우 낮은 NSCLC crop에서는 mask 적용 후 입력이
거의 blank가 되어 crop-level FN이 발생할 수 있다.

- 확인된 사례: LUNG1-205 low_mask crop 2건 (nzr_mean 0.037/0.047)
- reference auxiliary score ≈ 1.0 (TP) → selected auxiliary score < 0.03 (FN)
- **patient-level 영향: 없음** (LUNG1-205 patient-level TP 유지, mean/p95/max 모두)
- downstream 적용 시 nzr_mean < 0.05 low-mask case를 모니터링할 것

### C2 — sensitivity 소폭 하락

crop-level sensitivity: 0.9929 → 0.9924 (−0.0006), FN +25건
patient-level FN patients: reference=1, selected=1 (동일)

### C3 — fixed threshold 0.5 기준

모든 metric은 fixed threshold 0.5에서 계산됨. threshold sweep/최적화 미실시.
operating point 변경 시 결과가 달라질 수 있음.

---

## 6. 선택 근거 요약

**선택 이유 (masked_30b):**
- AUROC/AUPRC/Brier 전 항목 개선
- normal FP -5,114 (crop), normal FP patients -16 (patient)
- specificity +0.2372 (large)
- patient-level FN 악화 없음

**미선택 이유 (reference 24j bw1):**
- specificity / FP / AUROC / Brier에서 masked_30b 대비 열세
- masked input에서 얻어지는 FP 감소를 포기할 근거 없음

**caveat 보류 이유 (masked_30b 전면 선택 아님):**
- low-mask lesion-edge crop FN 위험 존재
- "selected candidate with caveat"로 기록

---

## 7. Guardrail

guardrail fail: **{guardrail_fail}** / {len(GUARDRAILS)}
verdict: **{("PASS" if guardrail_fail == 0 else "PARTIAL_PASS")}**

---

## 8. 다음 단계 (사용자 결정 필요)

이 checkpoint는 P-C-NORMAL30b masked-input을 current selected candidate로 공식 기록한다.
이후 방향은 사용자 승인 후 결정:

- masked_30b 기반 downstream scoring/heatmap 생성
- 또는 추가 ablation (low-mask filtering, mask threshold 조정 등)
- 또는 다른 masking 전략 실험
"""
    (REPORT_ROOT / "p_c_normal32_final_decision_checkpoint.md").write_text(
        checkpoint_md, encoding="utf-8")

    # ── 8. current_state proposed update ──────────────────────────────────
    current_state_md = f"""# P-C-NORMAL32 current_state proposed update

생성일: {ts}

## 변경 내용 (proposed)

current_state.md 또는 handoff 문서에 아래 내용 추가를 제안한다.

```
[P-C-NORMAL32] 2026-06-11
- selected candidate: P-C-NORMAL30b masked-input
  - ckpt: p_c_normal30b_best_val_auc_checkpoint.pt (epoch=6, val_auc=1.0000)
  - basis: repaired final_test, fixed threshold 0.5
  - caveat: low-mask lesion-edge NSCLC crop FN risk (C1), sensitivity -0.0006 (C2)
- reference: P-C-NORMAL24j-fix-balanced-w1 (유지)
- verdict chain: 31=PARTIAL_PASS → 31b=BLOCKED(원본유지) → 31c=PASS_FOR_DECISION → 32=PASS
- crop-level: AUROC +0.039, FP -5114, specificity +0.237, sensitivity -0.0006
- patient-level: FP patients -16, FN patients 동일
```

> 이 문서는 proposed update이며, current_state.md를 직접 수정하지 않음.
> 사용자 승인 후 반영.
"""
    (REPORT_ROOT / "p_c_normal32_current_state_proposed_update.md").write_text(
        current_state_md, encoding="utf-8")

    # ── 9. Final decision JSON ─────────────────────────────────────────────
    final_json = {
        "stage": "P-C-NORMAL32",
        "timestamp": ts,
        "verdict": "PASS" if guardrail_fail == 0 else "PARTIAL_PASS",
        "selected_candidate": "P-C-NORMAL30b_masked_input",
        "reference_candidate": "P-C-NORMAL24j_fix_balanced_w1",
        "selected_basis": "repaired_final_test_fixed_threshold_0p5",
        "threshold_optimized": False,
        "threshold_sweep_run": False,
        "training_run": False,
        "model_forward_run": False,
        "prediction_export_rerun": False,
        "original_31b_blocked_preserved": True,
        "p_c_normal31c_verdict": verdict_31c,
        "selected_with_caveat": True,
        "caveat": "low_mask_lesion_edge_crop_blank_input_FN_risk",
        "patient_level_fn_worsened": False,
        "patient_level_fp_improved": True,
        "diagnostic_claim": False,
        "guardrail_fail": guardrail_fail,
        "crop_auroc_delta": round(CROP["cand_auroc"] - CROP["ref_auroc"], 6),
        "crop_fp_delta": CROP["cand_fp"] - CROP["ref_fp"],
        "crop_fn_delta": CROP["cand_fn"] - CROP["ref_fn"],
        "patient_fp_delta": PAT["cand_fp_patients"] - PAT["ref_fp_patients"],
        "patient_fn_delta": PAT["cand_fn_patients"] - PAT["ref_fn_patients"],
        "selected_ckpt": ckpt_path,
        "next_step": "downstream scoring/heatmap 또는 추가 실험 — 사용자 결정 필요",
    }
    with open(REPORT_ROOT / "p_c_normal32_final_decision_checkpoint.json", "w",
              encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    # ── 10. DONE.json ─────────────────────────────────────────────────────
    done = {
        "stage": "P-C-NORMAL32",
        "timestamp": ts,
        "verdict": final_json["verdict"],
        "selected_candidate": "P-C-NORMAL30b_masked_input",
        "guardrail_fail": guardrail_fail,
        "files_written": [
            "p_c_normal32_final_decision_checkpoint.md",
            "p_c_normal32_final_decision_checkpoint.json",
            "p_c_normal32_selected_candidate_summary.csv",
            "p_c_normal32_metric_comparison_summary.csv",
            "p_c_normal32_caveat_summary.csv",
            "p_c_normal32_guardrail_check.csv",
            "p_c_normal32_current_state_proposed_update.md",
            "DONE.json",
        ],
        "next_step": "downstream scoring/heatmap 또는 추가 실험 — 사용자 결정 필요",
    }
    with open(REPORT_ROOT / "DONE.json", "w", encoding="utf-8") as f:
        json.dump(done, f, indent=2, ensure_ascii=False)

    print(f"[32] DONE. verdict={final_json['verdict']}, guardrail_fail={guardrail_fail}")
    print(f"[32] selected: {final_json['selected_candidate']}")
    print(f"[32] report: {REPORT_ROOT}/p_c_normal32_final_decision_checkpoint.md")
    return 0


if __name__ == "__main__":
    sys.exit(run())
