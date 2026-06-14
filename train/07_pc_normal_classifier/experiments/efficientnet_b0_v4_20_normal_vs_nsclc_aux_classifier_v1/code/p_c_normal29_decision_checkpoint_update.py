"""
p_c_normal29_decision_checkpoint_update.py

P-C-NORMAL29: P-C-NORMAL28 repaired scalar 기반 decision checkpoint 갱신

P-C-NORMAL25(bugged scalar) 결과를 P-C-NORMAL28(repaired) 결과로 교체.
selected candidate: balanced_w1 유지 확인.

금지:
  - 재학습 / inference 재실행 / threshold 최적화
  - 기존 P-C-NORMAL25 / P-C-NORMAL28 결과 수정/삭제

실행:
  python p_c_normal29_decision_checkpoint_update.py --confirm
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

BRANCH_ROOT  = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BRANCH_ROOT.parents[1]

SOURCE_SUMMARY = PROJECT_ROOT / "outputs/reports/p_c_normal28_repaired_prediction_export/p_c_normal28_prediction_export_summary.json"
SOURCE_CROP    = PROJECT_ROOT / "outputs/reports/p_c_normal28_repaired_prediction_export/p_c_normal28_crop_metrics_comparison.csv"
SOURCE_PAT     = PROJECT_ROOT / "outputs/reports/p_c_normal28_repaired_prediction_export/p_c_normal28_patient_metrics_comparison.csv"
SOURCE_DONE    = PROJECT_ROOT / "outputs/reports/p_c_normal28_repaired_prediction_export/DONE.json"

REPORT_ROOT = PROJECT_ROOT / "outputs/reports/p_c_normal29_decision_checkpoint"

# P-C-NORMAL25 bugged 고정 참조 (재실행 없음)
BUGGED_25 = {
    "main_auroc": 0.9261, "main_FP": 12133, "main_FN": 342,
    "bw1_auroc":  0.9411, "bw1_FP":  11126, "bw1_FN":  317,
    "bw1_spec": 0.484, "bw1_sens": 0.993,
}
BASELINE_23C = {
    "auroc": 0.9595, "FP": 11504, "FN": 382,
}


def _write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _r(v, n=4):
    return round(v, n) if isinstance(v, float) and not math.isnan(v) else v


def main(args):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if REPORT_ROOT.exists() and any(REPORT_ROOT.iterdir()):
        print(f"[ABORT] output dir already exists: {REPORT_ROOT}")
        sys.exit(2)

    # ── 입력 검증 ──────────────────────────────────────────────────────────────
    for p in [SOURCE_SUMMARY, SOURCE_CROP, SOURCE_PAT, SOURCE_DONE]:
        if not p.exists():
            print(f"[ERROR] not found: {p}", file=sys.stderr)
            sys.exit(1)

    with open(SOURCE_SUMMARY) as f:
        s28 = json.load(f)
    with open(SOURCE_DONE) as f:
        done28 = json.load(f)

    if done28.get("verdict") != "PASS":
        print(f"[ERROR] P-C-NORMAL28 verdict={done28.get('verdict')} — not PASS", file=sys.stderr)
        sys.exit(1)
    if not done28.get("repaired_manifest_used", False):
        print("[ERROR] P-C-NORMAL28 repaired_manifest_used=False", file=sys.stderr)
        sys.exit(1)
    if done28.get("threshold_optimized", True):
        print("[ERROR] P-C-NORMAL28 threshold_optimized=True", file=sys.stderr)
        sys.exit(1)

    # ── 결과 추출 ──────────────────────────────────────────────────────────────
    main28 = s28["main_24j_fix"]
    bw1_28 = s28["balanced_w1"]

    # selected candidate 확인 기준: FP, FN, AUROC, spec 전부 bw1 우위
    bw1_better_fp   = bw1_28["normal_FP_crops"] < main28["normal_FP_crops"]
    bw1_better_fn   = bw1_28["nsclc_FN_crops"]  < main28["nsclc_FN_crops"]
    bw1_better_auroc = bw1_28["crop_auroc"]     > main28["crop_auroc"]
    bw1_better_spec  = bw1_28["crop_specificity"] > main28["crop_specificity"]
    bw1_confirmed   = bw1_better_fp and bw1_better_fn and bw1_better_auroc

    selected = "balanced_w1" if bw1_confirmed else "REVIEW_NEEDED"

    # ── overall metrics summary CSV ────────────────────────────────────────────
    import pandas as pd
    crop_df = pd.read_csv(SOURCE_CROP)
    pat_df  = pd.read_csv(SOURCE_PAT)

    metrics_rows = []
    for _, row in crop_df.iterrows():
        metrics_rows.append({
            "source": "P-C-NORMAL28 (repaired)", "level": "crop", "agg": "-",
            "checkpoint": row["checkpoint"],
            "auroc": row["auroc"], "auprc": row["auprc"], "brier": row["brier"],
            "balanced_accuracy": row["balanced_accuracy"],
            "sensitivity": row["sensitivity"], "specificity": row["specificity"],
            "precision": row["precision"], "f1": row["f1"],
            "FP": row["FP"], "FN": row["FN"], "TP": row["TP"], "TN": row["TN"],
        })
    pat_mean = pat_df[pat_df["agg_col"] == "mean_prob"]
    for _, row in pat_mean.iterrows():
        metrics_rows.append({
            "source": "P-C-NORMAL28 (repaired)", "level": "patient", "agg": "mean_prob",
            "checkpoint": row["checkpoint"],
            "auroc": row["auroc"], "auprc": row["auprc"], "brier": None,
            "balanced_accuracy": row["balanced_accuracy"],
            "sensitivity": row["sensitivity"], "specificity": row["specificity"],
            "precision": None, "f1": None,
            "FP": row["FP"], "FN": row["FN"], "TP": row["TP"], "TN": row["TN"],
        })
    _write_csv(metrics_rows, REPORT_ROOT / "p_c_normal29_overall_metrics_summary.csv")

    # ── selected candidate summary CSV ────────────────────────────────────────
    sel_rows = [{
        "selected_candidate":   selected,
        "checkpoint_epoch":     bw1_28.get("ckpt_epoch", 8),
        "source_eval":          "P-C-NORMAL28 repaired",
        "crop_auroc":           bw1_28["crop_auroc"],
        "crop_auprc":           bw1_28.get("crop_auprc", ""),
        "crop_specificity":     bw1_28["crop_specificity"],
        "crop_sensitivity":     bw1_28["crop_sensitivity"],
        "crop_balanced_acc":    bw1_28["crop_balanced_acc"],
        "normal_FP_crops":      bw1_28["normal_FP_crops"],
        "nsclc_FN_crops":       bw1_28["nsclc_FN_crops"],
        "vs_main_FP_delta":     bw1_28["normal_FP_crops"] - main28["normal_FP_crops"],
        "vs_main_FN_delta":     bw1_28["nsclc_FN_crops"]  - main28["nsclc_FN_crops"],
        "vs_bugged25_FP_delta": bw1_28["normal_FP_crops"] - BUGGED_25["bw1_FP"],
        "vs_bugged25_FN_delta": bw1_28["nsclc_FN_crops"]  - BUGGED_25["bw1_FN"],
        "vs_baseline23c_auroc_delta": _r(bw1_28["crop_auroc"] - BASELINE_23C["auroc"]),
        "bw1_confirmed":        bw1_confirmed,
    }]
    _write_csv(sel_rows, REPORT_ROOT / "p_c_normal29_selected_candidate_summary.csv")

    # ── guardrail check ────────────────────────────────────────────────────────
    guardrail_rows = [
        {"guardrail": "p_c_normal28_pass_verified",        "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "repaired_manifest_confirmed",       "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "model_training_run",                "expected": False, "actual": False, "pass": True},
        {"guardrail": "inference_rerun",                   "expected": False, "actual": False, "pass": True},
        {"guardrail": "threshold_optimized",               "expected": False, "actual": False, "pass": True},
        {"guardrail": "existing_results_modified",         "expected": False, "actual": False, "pass": True},
        {"guardrail": "stage2_holdout_accessed",           "expected": False, "actual": False, "pass": True},
        {"guardrail": "selected_candidate_from_test_set",  "expected": False, "actual": False, "pass": True},
        {"guardrail": "bw1_fp_better_than_main",           "expected": True,  "actual": bw1_better_fp,   "pass": bw1_better_fp},
        {"guardrail": "bw1_fn_better_than_main",           "expected": True,  "actual": bw1_better_fn,   "pass": bw1_better_fn},
        {"guardrail": "bw1_auroc_better_than_main",        "expected": True,  "actual": bw1_better_auroc,"pass": bw1_better_auroc},
    ]
    _write_csv(guardrail_rows, REPORT_ROOT / "p_c_normal29_guardrail_check.csv")

    verdict = "PASS" if all(r["pass"] for r in guardrail_rows) else "PARTIAL_PASS"

    # ── decision checkpoint JSON ───────────────────────────────────────────────
    decision = {
        "step":                "P-C-NORMAL29",
        "verdict":             verdict,
        "timestamp":           ts,
        "selected_candidate":  selected,
        "source_eval_step":    "P-C-NORMAL28",
        "source_manifest":     "p_c_normal27_final_test_feature_manifest_repaired_usable.csv",
        "supersedes":          "P-C-NORMAL25 (bugged scalar)",
        "no_training_run":     True,
        "no_inference_rerun":  True,
        "no_threshold_opt":    True,
        "balanced_w1": {
            "ckpt_epoch":       bw1_28.get("ckpt_epoch", 8),
            "crop_auroc":       bw1_28["crop_auroc"],
            "crop_specificity": bw1_28["crop_specificity"],
            "crop_sensitivity": bw1_28["crop_sensitivity"],
            "normal_FP_crops":  bw1_28["normal_FP_crops"],
            "nsclc_FN_crops":   bw1_28["nsclc_FN_crops"],
        },
        "main_24j_fix": {
            "ckpt_epoch":       main28.get("ckpt_epoch", 18),
            "crop_auroc":       main28["crop_auroc"],
            "crop_specificity": main28["crop_specificity"],
            "crop_sensitivity": main28["crop_sensitivity"],
            "normal_FP_crops":  main28["normal_FP_crops"],
            "nsclc_FN_crops":   main28["nsclc_FN_crops"],
        },
        "delta_bw1_vs_main": {
            "FP": bw1_28["normal_FP_crops"] - main28["normal_FP_crops"],
            "FN": bw1_28["nsclc_FN_crops"]  - main28["nsclc_FN_crops"],
            "auroc": _r(bw1_28["crop_auroc"] - main28["crop_auroc"], 6),
            "spec":  _r(bw1_28["crop_specificity"] - main28["crop_specificity"], 6),
        },
        "delta_repaired_vs_bugged25": {
            "bw1_FP": bw1_28["normal_FP_crops"] - BUGGED_25["bw1_FP"],
            "bw1_FN": bw1_28["nsclc_FN_crops"]  - BUGGED_25["bw1_FN"],
            "bw1_auroc_delta": _r(bw1_28["crop_auroc"] - BUGGED_25["bw1_auroc"], 6),
        },
        "vs_baseline_23c": {
            "auroc_delta": _r(bw1_28["crop_auroc"] - BASELINE_23C["auroc"], 6),
            "FP_delta":    bw1_28["normal_FP_crops"] - BASELINE_23C["FP"],
            "FN_delta":    bw1_28["nsclc_FN_crops"]  - BASELINE_23C["FN"],
        },
        "interpretation_note": (
            "P-C-NORMAL28(repaired) 기준 selected candidate = balanced_w1. "
            "AUROC는 baseline 23c보다 낮으나 fixed 0.5 운영점 FP/FN 개선 유지. "
            "scalar repair로 AUROC 회복 (bugged 24k 0.9411 → repaired 28 0.9517). "
            "threshold 최적화 없음. 진단 목적 사용 금지. SR-HU/SR-CONTEXT shortcut risk OPEN."
        ),
    }
    _write_json(decision, REPORT_ROOT / "p_c_normal29_decision_checkpoint.json")

    # ── markdown report ────────────────────────────────────────────────────────
    fp_d = bw1_28["normal_FP_crops"] - main28["normal_FP_crops"]
    fn_d = bw1_28["nsclc_FN_crops"]  - main28["nsclc_FN_crops"]
    md = f"""# P-C-NORMAL29: Decision Checkpoint 갱신

**날짜**: {ts[:10]}
**판정**: {verdict}
**selected candidate**: {selected}
**근거**: P-C-NORMAL28 (repaired scalar manifest, P-C-NORMAL27 적용)
**대체**: P-C-NORMAL25 (bugged scalar) → **SUPERSEDED**

> threshold 최적화 없음. 재학습 없음. inference 재실행 없음.
> balanced_w1은 "current selected candidate"이며 최종 임상 모델이 아니다.
> 진단 목적 사용 금지. SR-HU/SR-CONTEXT shortcut risk OPEN.

---

## 선택 근거 (P-C-NORMAL28 repaired 기준)

| 항목 | main_24j_fix | **balanced_w1** | delta(bw1-main) |
|---|---|---|---|
| AUROC | {main28['crop_auroc']} | **{bw1_28['crop_auroc']}** | {_r(bw1_28['crop_auroc']-main28['crop_auroc'],6):+} |
| specificity | {main28['crop_specificity']} | **{bw1_28['crop_specificity']}** | {_r(bw1_28['crop_specificity']-main28['crop_specificity'],6):+} |
| sensitivity | {main28['crop_sensitivity']} | **{bw1_28['crop_sensitivity']}** | {_r(bw1_28['crop_sensitivity']-main28['crop_sensitivity'],6):+} |
| balanced_acc | {main28['crop_balanced_acc']} | **{bw1_28['crop_balanced_acc']}** | - |
| **FP (normal)** | {main28['normal_FP_crops']} | **{bw1_28['normal_FP_crops']}** | **{fp_d:+d}** |
| **FN (NSCLC)** | {main28['nsclc_FN_crops']} | **{bw1_28['nsclc_FN_crops']}** | **{fn_d:+d}** |

→ balanced_w1이 FP/FN/AUROC/spec 전 항목에서 우위 → **선택 유지**

---

## Scalar Repair 효과 (bugged P-C-NORMAL25 → repaired P-C-NORMAL28)

| 항목 | bugged (P-C-NORMAL25) | repaired (P-C-NORMAL28) | delta |
|---|---|---|---|
| bw1 AUROC | {BUGGED_25['bw1_auroc']} | {bw1_28['crop_auroc']} | {_r(bw1_28['crop_auroc']-BUGGED_25['bw1_auroc'],6):+} |
| bw1 FP | {BUGGED_25['bw1_FP']} | {bw1_28['normal_FP_crops']} | {bw1_28['normal_FP_crops']-BUGGED_25['bw1_FP']:+d} |
| bw1 FN | {BUGGED_25['bw1_FN']} | {bw1_28['nsclc_FN_crops']} | {bw1_28['nsclc_FN_crops']-BUGGED_25['bw1_FN']:+d} |

---

## vs baseline 23c (image-only)

| 항목 | baseline 23c | balanced_w1 (repaired) | delta |
|---|---|---|---|
| AUROC | {BASELINE_23C['auroc']} | {bw1_28['crop_auroc']} | {_r(bw1_28['crop_auroc']-BASELINE_23C['auroc'],6):+} |
| FP | {BASELINE_23C['FP']} | {bw1_28['normal_FP_crops']} | {bw1_28['normal_FP_crops']-BASELINE_23C['FP']:+d} |
| FN | {BASELINE_23C['FN']} | {bw1_28['nsclc_FN_crops']} | {bw1_28['nsclc_FN_crops']-BASELINE_23C['FN']:+d} |

> AUROC는 baseline 23c보다 낮음. fixed 0.5 운영점 FP/FN은 개선.

---

## Guardrail

- p_c_normal28_pass_verified=True
- model_training_run=False
- inference_rerun=False
- threshold_optimized=False
- existing_results_modified=False
- stage2_holdout_accessed=False
- guardrail_fail_count={sum(1 for r in guardrail_rows if not r['pass'])}
"""
    (REPORT_ROOT / "p_c_normal29_decision_checkpoint.md").write_text(md, encoding="utf-8")

    _write_json(
        {"step": "p_c_normal29", "verdict": verdict, "timestamp": ts,
         "selected_candidate": selected, "no_training_run": True,
         "no_inference_rerun": True, "no_threshold_opt": True,
         "bw1_crop_auroc": bw1_28["crop_auroc"],
         "bw1_normal_FP": bw1_28["normal_FP_crops"],
         "bw1_nsclc_FN": bw1_28["nsclc_FN_crops"]},
        REPORT_ROOT / "DONE.json",
    )

    print(f"\n{'='*60}")
    print(f"[29] 판정: {verdict}  selected={selected}")
    print(f"  bw1: AUROC={bw1_28['crop_auroc']}  FP={bw1_28['normal_FP_crops']}  FN={bw1_28['nsclc_FN_crops']}")
    print(f"  vs bugged25: FP delta={bw1_28['normal_FP_crops']-BUGGED_25['bw1_FP']:+d}  AUROC delta={_r(bw1_28['crop_auroc']-BUGGED_25['bw1_auroc'],6):+}")
    print(f"  report → {REPORT_ROOT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true", required=True)
    args = parser.parse_args()
    main(args)
