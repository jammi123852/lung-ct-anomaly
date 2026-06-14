"""
P-C-NORMAL33: selected candidate handoff / spec package
documentation only — no training, no model forward, no scoring, no heatmap
"""

import sys
import csv
import json
import datetime
from pathlib import Path

PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

# ── Input paths ────────────────────────────────────────────────────────────
IN_32_JSON  = PROJECT_ROOT / "outputs/reports/p_c_normal32_final_decision_checkpoint/p_c_normal32_final_decision_checkpoint.json"
IN_30B_SUM  = PROJECT_ROOT / "outputs/reports/p_c_normal30b_masked_input_full_train/p_c_normal30b_summary.json"
IN_29B_MAN  = PROJECT_ROOT / "outputs/reports/p_c_normal29b_crop_level_mask_generation/p_c_normal29b_mask_manifest.csv"
IN_SCALAR   = PROJECT_ROOT / "outputs/reports/p_c_normal24h_fix_zroi_scalar_fusion_drycheck/p_c_normal24h_fix_scalar_normalization_stats.json"
IN_32_DONE  = PROJECT_ROOT / "outputs/reports/p_c_normal32_final_decision_checkpoint/DONE.json"
IN_31_DONE  = PROJECT_ROOT / "outputs/reports/p_c_normal31_repaired_final_test_masked_comparison/DONE.json"
IN_31C_SUM  = PROJECT_ROOT / "outputs/reports/p_c_normal31c_low_mask_fn_caveat_addendum/p_c_normal31c_summary.json"

REPORT_ROOT = PROJECT_ROOT / "outputs/reports/p_c_normal33_selected_candidate_handoff_package"

GUARDRAILS = {
    "documentation_only": True,
    "no_training_run": True,
    "no_model_forward": True,
    "no_prediction_export_rerun": True,
    "no_downstream_scoring_run": True,
    "no_heatmap_run": True,
    "no_threshold_optimization": True,
    "no_threshold_sweep": True,
    "no_best_threshold_selection": True,
    "no_checkpoint_modification": True,
    "no_existing_result_overwrite": True,
    "selected_candidate_masked_30b_confirmed": False,
    "selected_checkpoint_not_smoke": False,
    "selected_with_caveat": True,
    "low_mask_caveat_recorded": False,
    "diagnostic_wording_avoided": True,
}


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows, path):
    if not rows:
        path.write_text("(empty)\n", encoding="utf-8")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
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
    for p in [IN_32_JSON, IN_30B_SUM, IN_29B_MAN, IN_SCALAR,
              IN_32_DONE, IN_31_DONE, IN_31C_SUM]:
        if not p.exists():
            print(f"[ABORT] Missing: {p}", file=sys.stderr)
            sys.exit(1)

    # ── 2. Load prior results ──────────────────────────────────────────────
    dec32   = load_json(IN_32_JSON)
    sum30b  = load_json(IN_30B_SUM)
    scalar  = load_json(IN_SCALAR)
    sum31c  = load_json(IN_31C_SUM)

    # P-C-NORMAL32 PASS 확인
    if dec32.get("verdict") != "PASS":
        print(f"[ABORT] P-C-NORMAL32 verdict expected PASS, got {dec32.get('verdict')}", file=sys.stderr)
        sys.exit(2)

    selected_candidate = dec32.get("selected_candidate", "")
    if "30b" not in selected_candidate.lower():
        print(f"[ABORT] unexpected selected_candidate: {selected_candidate}", file=sys.stderr)
        sys.exit(2)
    GUARDRAILS["selected_candidate_masked_30b_confirmed"] = True

    ckpt_path = dec32.get("selected_ckpt", sum30b.get("best_ckpt", ""))
    if not ckpt_path:
        print("[ABORT] checkpoint path empty", file=sys.stderr)
        sys.exit(2)
    if "smoke" in str(ckpt_path).lower():
        print(f"[ABORT] smoke checkpoint selected: {ckpt_path}", file=sys.stderr)
        sys.exit(2)
    GUARDRAILS["selected_checkpoint_not_smoke"] = True

    # scalar norm stats
    lzp_mean = scalar["features"]["lung_z_percentile"]["mean"]
    lzp_std  = scalar["features"]["lung_z_percentile"]["std"]
    clrr_mean = scalar["features"]["crop_lung_roi_ratio"]["mean"]
    clrr_std  = scalar["features"]["crop_lung_roi_ratio"]["std"]

    print(f"[33] selected: {selected_candidate}")
    print(f"[33] ckpt: {ckpt_path}")
    print(f"[33] scalar: lzp mean={lzp_mean:.6f} std={lzp_std:.6f} / clrr mean={clrr_mean:.6f} std={clrr_std:.6f}")

    # ── 3. Selected candidate summary CSV ─────────────────────────────────
    cand_rows = [
        {"field": "selected_candidate",        "value": selected_candidate},
        {"field": "selected_checkpoint",        "value": ckpt_path},
        {"field": "selected_checkpoint_is_smoke","value": "false"},
        {"field": "best_epoch",                 "value": sum30b.get("best_epoch", "")},
        {"field": "best_val_auc",               "value": sum30b.get("best_val_auc", "")},
        {"field": "reference_candidate",        "value": dec32.get("reference_candidate", "")},
        {"field": "selected_basis",             "value": "P-C-NORMAL32 decision checkpoint"},
        {"field": "eval_set",                   "value": "repaired_final_test (normal_test=21560 / stage2_holdout=44723)"},
        {"field": "threshold_default",          "value": "0.5"},
        {"field": "threshold_optimized",        "value": "false"},
        {"field": "selected_with_caveat",       "value": "true"},
        {"field": "caveat",                     "value": "low_mask_lesion_edge_crop_blank_input_FN_risk"},
        {"field": "diagnostic_claim",           "value": "false"},
        {"field": "train_stage2_excluded",      "value": "true"},
        {"field": "balanced_sampling",          "value": "1:1 downsampling (balanced_w1)"},
        {"field": "p_c_normal32_verdict",       "value": dec32.get("verdict", "")},
        {"field": "p_c_normal31c_verdict",      "value": sum31c.get("re_adjudication_verdict", "")},
        {"field": "original_31b_blocked_preserved", "value": "true"},
        {"field": "crop_AUROC_delta",           "value": dec32.get("crop_auroc_delta", "")},
        {"field": "crop_FP_delta",              "value": dec32.get("crop_fp_delta", "")},
        {"field": "crop_FN_delta",              "value": dec32.get("crop_fn_delta", "")},
        {"field": "patient_FP_delta",           "value": dec32.get("patient_fp_delta", "")},
        {"field": "patient_FN_delta",           "value": dec32.get("patient_fn_delta", "")},
    ]
    write_csv(cand_rows,
              REPORT_ROOT / "p_c_normal33_selected_candidate_summary.csv")

    # ── 4. Input schema CSV ────────────────────────────────────────────────
    schema_rows = [
        # CT crop
        {"input": "ct_crop",   "key": "shape",   "value": "3×96×96",
         "note": "channel = z-1 / z / z+1"},
        {"input": "ct_crop",   "key": "HU_clip", "value": "-1000 to 200",
         "note": "HU clip before normalize"},
        {"input": "ct_crop",   "key": "normalize","value": "[0, 1] then ImageNet mean/std",
         "note": "after mask multiply"},
        {"input": "ct_crop",   "key": "ImageNet_mean","value": "[0.485, 0.456, 0.406]",
         "note": "applied per channel after masking"},
        {"input": "ct_crop",   "key": "ImageNet_std", "value": "[0.229, 0.224, 0.225]",
         "note": "applied per channel after masking"},
        # mask
        {"input": "mask_3ch",  "key": "shape",   "value": "3×96×96",
         "note": "same shape as ct_crop"},
        {"input": "mask_3ch",  "key": "dtype",   "value": "uint8 or bool",
         "note": "1=lung ROI, 0=background"},
        {"input": "mask_3ch",  "key": "apply",   "value": "ct_crop * mask_3ch",
         "note": "image only — scalar/label/weight 제외"},
        {"input": "mask_3ch",  "key": "source",  "value": "ROI npy via P-C-NORMAL29b spec",
         "note": "z-1/z/z+1 각 채널별 slice crop"},
        # scalar
        {"input": "lung_z_percentile",      "key": "mean", "value": str(round(lzp_mean, 8)),
         "note": "P-C-NORMAL24h-fix normalization"},
        {"input": "lung_z_percentile",      "key": "std",  "value": str(round(lzp_std, 8)),
         "note": "P-C-NORMAL24h-fix normalization"},
        {"input": "crop_lung_roi_ratio",    "key": "mean", "value": str(round(clrr_mean, 8)),
         "note": "P-C-NORMAL24h-fix normalization"},
        {"input": "crop_lung_roi_ratio",    "key": "std",  "value": str(round(clrr_std, 8)),
         "note": "P-C-NORMAL24h-fix normalization"},
        {"input": "crop_lung_roi_ratio",    "key": "denominator",
         "value": "bbox_h * bbox_w (fixed, 24g-fix)",
         "note": "old 24h was 1/9 scaled (wrong) — 24h-fix corrects this"},
        # model
        {"input": "model",     "key": "architecture", "value": "EfficientNet-B0 + scalar fusion",
         "note": "img branch 1280-dim + scalar branch 2→32→16 → head 1296→64→1"},
        {"input": "model",     "key": "loss",         "value": "BCEWithLogitsLoss",
         "note": "sample_weight=1.0 (balanced_w1)"},
    ]
    write_csv(schema_rows, REPORT_ROOT / "p_c_normal33_input_schema.csv")

    # ── 5. Mask generation spec CSV ───────────────────────────────────────
    mask_spec_rows = [
        {"item": "roi_source",         "spec": "ROI npy from P-C-NORMAL22 / safe_cut pipeline"},
        {"item": "crop_center",        "spec": "canonical_volume_z / center_y / center_x"},
        {"item": "crop_size",          "spec": "96×96 px (center_y±48, center_x±48)"},
        {"item": "channel_mapping",    "spec": "ch0=z-1, ch1=z, ch2=z+1"},
        {"item": "z_boundary",         "spec": "nearest-repeat (z < 0 → z=0, z >= max → z=max-1)"},
        {"item": "mask_dtype",         "spec": "uint8 (0/1)"},
        {"item": "spatial_pad",        "spec": "zero-pad if ROI smaller than 96×96"},
        {"item": "zero_mask_criterion","spec": "mask_3ch.sum() == 0 → zero_mask=True"},
        {"item": "low_mask_criterion", "spec": "nzr_mean < 0.05 and not zero_mask → low_mask=True"},
        {"item": "zero_mask_action",   "spec": "audit 필수, PARTIAL_PASS 판정"},
        {"item": "low_mask_action",    "spec": "audit 필수, caveat flag 기록, downstream monitoring 필요"},
        {"item": "low_mask_caveat_case","spec": "LUNG1-205 crop 2건 (nzr_mean 0.037/0.047) — reference TP → masked_30b crop-level FN"},
        {"item": "low_mask_patient_impact","spec": "LUNG1-205 patient-level TP 유지 (mean/p95/max)"},
        {"item": "mask_apply_target",  "spec": "ct_crop only; scalar/label/sample_weight 제외"},
        {"item": "ref_implementation", "spec": "P-C-NORMAL29b crop-level mask generation"},
    ]
    write_csv(mask_spec_rows, REPORT_ROOT / "p_c_normal33_mask_generation_spec.csv")

    # ── 6. Downstream output contract CSV ─────────────────────────────────
    contract_cols = [
        ("patient_id",             "str",   "환자 ID"),
        ("safe_id",                "str",   "safe anonymized ID"),
        ("crop_path",              "str",   "npz 파일 경로"),
        ("mask_path",              "str",   "mask npz 경로"),
        ("source_split",           "str",   "normal_test / stage2_holdout / etc"),
        ("canonical_volume_z",     "float", "CT volume z 좌표 (mm or index)"),
        ("local_z",                "float", "slice 내 local z 인덱스"),
        ("center_y",               "int",   "crop center y"),
        ("center_x",               "int",   "crop center x"),
        ("position_bin",           "str",   "upper/lower × central/peripheral"),
        ("lung_z_percentile_raw",  "float", "정규화 전 raw scalar"),
        ("crop_lung_roi_ratio_raw","float", "정규화 전 raw scalar"),
        ("lung_z_percentile_norm", "float", "24h-fix 정규화 후"),
        ("crop_lung_roi_ratio_norm","float","24h-fix 정규화 후"),
        ("mask_nonzero_ratio_mean","float", "mask nzr (3채널 평균)"),
        ("low_mask_flag",          "bool",  "nzr_mean < 0.05"),
        ("zero_mask_flag",         "bool",  "mask_3ch.sum() == 0"),
        ("logit",                  "float", "모델 raw logit"),
        ("prob",                   "float", "sigmoid(logit)"),
        ("pred_at_0p5",            "int",   "1 if prob >= 0.5 else 0"),
        ("model_name",             "str",   "masked_30b"),
        ("checkpoint_path",        "str",   "절대 경로"),
        ("masked_input_used",      "bool",  "True"),
        ("threshold_used",         "float", "0.5"),
        ("threshold_optimized",    "bool",  "False"),
        ("caveat_flag",            "str",   "none / low_mask / zero_mask"),
    ]
    contract_rows = [
        {"column": c, "dtype": d, "description": desc}
        for c, d, desc in contract_cols
    ]
    write_csv(contract_rows,
              REPORT_ROOT / "p_c_normal33_downstream_output_contract.csv")

    # ── 7. Caveat & monitoring rules CSV ──────────────────────────────────
    GUARDRAILS["low_mask_caveat_recorded"] = True
    caveat_rows = [
        {"rule_id": "CAV-01",
         "category": "low_mask_fn_risk",
         "description": (
             "masked input은 normal FP를 크게 줄이지만, "
             "ROI overlap이 매우 낮은 NSCLC crop에서 입력이 거의 blank가 되어 "
             "crop-level FN이 발생할 수 있다."
         ),
         "observed": "LUNG1-205 low_mask crop 2건 (nzr_mean 0.037/0.047)",
         "patient_level_impact": "없음 (LUNG1-205 patient-level TP 유지)",
         "action": "nzr_mean < 0.05 crop은 low_mask_flag=True로 별도 monitoring table 분리"},
        {"rule_id": "CAV-02",
         "category": "sensitivity_slight_decrease",
         "description": "crop-level sensitivity 0.9929 → 0.9924 (−0.0006), FN +25건",
         "observed": "crop-level 전체 (final_test 66,283 crops)",
         "patient_level_impact": "patient FN patients ref=1 cand=1 (동일)",
         "action": "crop-level FN 증가폭 downstream에서 추적"},
        {"rule_id": "CAV-03",
         "category": "fixed_threshold_only",
         "description": "모든 metric은 fixed threshold 0.5 기준. threshold 최적화/sweep 미실시.",
         "observed": "-",
         "patient_level_impact": "-",
         "action": "operating point 변경 시 전체 재평가 필요"},
        {"rule_id": "MON-01",
         "category": "low_mask_monitoring",
         "description": "scoring 시 nzr_mean < 0.05 crop을 low_mask_flag=True로 기록하고 별도 table 분리",
         "observed": "-",
         "patient_level_impact": "-",
         "action": "low_mask crop score는 참고값으로만 사용"},
        {"rule_id": "MON-02",
         "category": "patient_aggregation",
         "description": "patient-level 집계는 mean_prob / p95_prob / max_prob 모두 기록",
         "observed": "-",
         "patient_level_impact": "-",
         "action": "mean_prob 기본, p95/max 보조"},
        {"rule_id": "USE-01",
         "category": "terminology",
         "description": "score는 'NSCLC-lesion-like auxiliary score'로만 표현. 진단/암 확률 표현 금지.",
         "observed": "-",
         "patient_level_impact": "-",
         "action": "보고서/카드에서 prohibited_terms 사전 검사"},
    ]
    write_csv(caveat_rows,
              REPORT_ROOT / "p_c_normal33_caveat_and_monitoring_rules.csv")

    # ── 8. Guardrail CSV ──────────────────────────────────────────────────
    guardrail_rows = [
        {"key": k, "value": v, "status": "OK" if v is True else "FAIL"}
        for k, v in GUARDRAILS.items()
    ]
    write_csv(guardrail_rows, REPORT_ROOT / "p_c_normal33_guardrail_check.csv")
    guardrail_fail = sum(1 for v in GUARDRAILS.values() if v is False)
    verdict = "PASS" if guardrail_fail == 0 else "PARTIAL_PASS"

    # ── 9. Handoff markdown ───────────────────────────────────────────────
    handoff_md = f"""# P-C-NORMAL33: Selected Candidate Handoff / Spec Package

생성일: {ts}
Documentation only — 학습/inference/scoring/heatmap 없음.

---

## 1. Selected Candidate

| 항목 | 값 |
|------|----|
| selected candidate | **P-C-NORMAL30b masked-input** |
| checkpoint | `{ckpt_path}` |
| smoke checkpoint | No |
| epoch | {sum30b.get("best_epoch", "")} |
| val AUROC | {sum30b.get("best_val_auc", "")} |
| reference candidate | P-C-NORMAL24j-fix balanced-w1 |
| selection basis | P-C-NORMAL32 decision checkpoint |
| threshold default | 0.5 (fixed, not optimized) |
| selected with caveat | Yes |
| diagnostic claim | No |

### Verdict chain

| Stage | Verdict |
|-------|---------|
| P-C-NORMAL31 | PARTIAL_PASS (low_or_zero_mask=16) |
| P-C-NORMAL31b | BLOCKED (원본 보존) |
| P-C-NORMAL31c | PASS_FOR_DECISION |
| P-C-NORMAL32 | PASS |
| **P-C-NORMAL33** | **{verdict}** |

---

## 2. Eval set 구성

- eval: repaired_final_test
  - normal_test: 21,560 crops (정상)
  - stage2_holdout: 44,723 crops (NSCLC)
  - 합계: 66,283 crops / 158 patients
- train/val: P-C-NORMAL24g-fix balanced_w1 (stage2_holdout 제외)
  - train 15,782 / val 4,160 (1:1 balanced downsampling)

---

## 3. 핵심 성능 (reference 대비)

| Metric | Reference | Selected | Delta |
|--------|-----------|----------|-------|
| crop AUROC | 0.9517 | 0.9904 | +0.0387 |
| crop AUPRC | 0.9733 | 0.9954 | +0.0221 |
| crop Brier | 0.1533 | 0.0750 | -0.0783 |
| crop Accuracy | 0.8312 | 0.9079 | +0.0767 |
| crop Precision | 0.8033 | 0.8851 | +0.0818 |
| crop Recall | 0.9929 | 0.9924 | -0.0006 |
| crop F1 | 0.8881 | 0.9357 | +0.0476 |
| crop specificity | 0.4956 | 0.7328 | +0.2372 |
| crop FP | 10,874 | 5,760 | -5,114 |
| crop FN | 317 | 342 | +25 |
| patient AUROC (mean) | 0.9898 | 0.9943 | +0.0046 |
| patient FP patients | 21 | 5 | -16 |
| patient FN patients | 1 | 1 | 0 |

---

## 4. Input Schema

### CT Crop
- shape: 3×96×96 (ch0=z-1, ch1=z, ch2=z+1)
- HU clip: −1000 to 200 → normalize [0,1]
- masked input: `ct_crop * mask_3ch` → ImageNet mean/std
- ImageNet mean: [0.485, 0.456, 0.406]
- ImageNet std: [0.229, 0.224, 0.225]

### Mask
- shape: 3×96×96, dtype=uint8
- source: ROI npy → center_y±48, center_x±48 crop, z-1/z/z+1 채널
- z boundary: nearest-repeat
- apply to image only (scalar/label 제외)
- low_mask 기준: nzr_mean < 0.05 → caveat flag

### Scalar Features (P-C-NORMAL24h-fix)
| Feature | mean | std |
|---------|------|-----|
| lung_z_percentile | {lzp_mean:.8f} | {lzp_std:.8f} |
| crop_lung_roi_ratio | {clrr_mean:.8f} | {clrr_std:.8f} |

> 주의: crop_lung_roi_ratio는 bbox_h×bbox_w 분모 기준 (24g-fix). 구 24h stats는 폐기.

### Model Architecture
- EfficientNet-B0 image branch (1280-dim)
- Scalar branch: 2→32→16
- Fusion head: 1296→64→1, BCEWithLogitsLoss
- sample_weight = 1.0 (balanced_w1)

---

## 5. Mask Generation Spec

1. ROI npy 로드 (safe_cut pipeline)
2. center_y±48, center_x±48 기준 96×96 crop
3. z-1/z/z+1 채널별 ROI slice crop
4. z boundary: nearest-repeat
5. zero_mask: mask_3ch.sum()==0 → audit/PARTIAL_PASS
6. low_mask: nzr_mean<0.05 → caveat_flag, monitoring table 분리
7. 구현 참조: P-C-NORMAL29b

---

## 6. Downstream Output Contract (주요 컬럼)

patient_id / safe_id / crop_path / mask_path / source_split /
canonical_volume_z / local_z / center_y / center_x / position_bin /
lung_z_percentile_raw / crop_lung_roi_ratio_raw /
lung_z_percentile_norm / crop_lung_roi_ratio_norm /
mask_nonzero_ratio_mean / low_mask_flag / zero_mask_flag /
logit / prob / pred_at_0p5 /
model_name / checkpoint_path / masked_input_used /
threshold_used / threshold_optimized / caveat_flag

---

## 7. Caveat & Monitoring Rules

| Rule | 내용 |
|------|------|
| CAV-01 | low_mask NSCLC crop에서 blank-input FN 위험. nzr_mean<0.05 → monitoring |
| CAV-02 | crop-level sensitivity −0.0006. patient FN 동일. |
| CAV-03 | fixed threshold 0.5만 기본 reporting. sweep 미실시. |
| MON-01 | nzr_mean<0.05 crop → low_mask_flag=True, 별도 table 분리 |
| MON-02 | patient 집계: mean/p95/max prob 모두 기록 |
| USE-01 | score = NSCLC-lesion-like auxiliary score. 진단/암 확률 표현 금지. |

### 금지 표현
암 확률 / 폐선암 확률 / cancer probability / diagnostic probability /
clinical diagnosis / cancer diagnosis / malignancy probability

### 허용 표현
auxiliary score / NSCLC-lesion-like score /
normal-like vs NSCLC-lesion-like classifier /
selected candidate / candidate scoring / fixed threshold operating point

---

## 8. guardrail fail: {guardrail_fail} / {len(GUARDRAILS)}

---

## 9. 다음 단계 (사용자 결정 필요)

→ `p_c_normal33_next_step_options.md` 참고
"""
    (REPORT_ROOT / "p_c_normal33_selected_candidate_handoff.md").write_text(
        handoff_md, encoding="utf-8")

    # ── 10. Next step options ─────────────────────────────────────────────
    next_md = f"""# P-C-NORMAL33 next step options

생성일: {ts}
이 문서는 가능한 다음 단계를 나열한다. 실행은 사용자 승인 후 진행.

---

## Option A — masked_30b 기반 전체 scoring

selected candidate (masked_30b)로 전체 데이터셋 또는 특정 cohort에 대해
crop-level NSCLC-lesion-like auxiliary score를 생성한다.

- 입력: crop npz + mask npz + scalar
- 출력: output contract CSV (prob / pred_at_0p5 / caveat_flag 포함)
- 주의: low_mask crop 별도 monitoring

## Option B — slice/patient-level heatmap 생성

crop-level prob → slice별 공간 집계 → heatmap PNG 생성

- 입력: Option A 결과 CSV
- 출력: slice heatmap PNG, patient-level ranking

## Option C — XAI 설명 카드 생성

selected candidate score를 기반으로 LUNG1 candidate에 대한 설명 카드 생성

- 참조: EfficientNet S5 card 체인 (기존 XAI 파이프라인)
- 입력: Option A/B 결과
- 주의: 진단 표현 금지, auxiliary score로만 표현

## Option D — 추가 실험 (masking 전략 개선)

- low-mask filtering: nzr_mean < 0.05 crop masking 제외 또는 별도 처리
- mask threshold 조정 실험
- 또는 다른 masking 전략

---

현재 권장: **Option A → Option B → Option C** 순서
단, 각 단계 실행 전 사용자 승인 필요.
"""
    (REPORT_ROOT / "p_c_normal33_next_step_options.md").write_text(
        next_md, encoding="utf-8")

    # ── 11. Handoff JSON ──────────────────────────────────────────────────
    handoff_json = {
        "stage": "P-C-NORMAL33",
        "timestamp": ts,
        "verdict": verdict,
        "package_type": "selected_candidate_handoff",
        "selected_candidate": selected_candidate,
        "selected_checkpoint": ckpt_path,
        "selected_checkpoint_is_smoke": False,
        "reference_candidate": dec32.get("reference_candidate", ""),
        "selected_basis": "P-C-NORMAL32",
        "selected_with_caveat": True,
        "caveat": "low_mask_lesion_edge_crop_blank_input_FN_risk",
        "fixed_threshold_default": 0.5,
        "threshold_optimized": False,
        "training_run": False,
        "model_forward_run": False,
        "prediction_export_rerun": False,
        "downstream_scoring_run": False,
        "heatmap_run": False,
        "diagnostic_claim": False,
        "scalar_norm_source": "P-C-NORMAL24h-fix",
        "lzp_mean": lzp_mean, "lzp_std": lzp_std,
        "clrr_mean": clrr_mean, "clrr_std": clrr_std,
        "guardrail_fail": guardrail_fail,
        "files_written": [
            "p_c_normal33_selected_candidate_handoff.md",
            "p_c_normal33_selected_candidate_handoff.json",
            "p_c_normal33_selected_candidate_summary.csv",
            "p_c_normal33_input_schema.csv",
            "p_c_normal33_mask_generation_spec.csv",
            "p_c_normal33_downstream_output_contract.csv",
            "p_c_normal33_caveat_and_monitoring_rules.csv",
            "p_c_normal33_guardrail_check.csv",
            "p_c_normal33_next_step_options.md",
            "DONE.json",
        ],
        "next_step": "Option A/B/C/D — 사용자 결정 필요",
    }
    with open(REPORT_ROOT / "p_c_normal33_selected_candidate_handoff.json",
              "w", encoding="utf-8") as f:
        json.dump(handoff_json, f, indent=2, ensure_ascii=False)

    # ── 12. DONE.json ─────────────────────────────────────────────────────
    done = {
        "stage": "P-C-NORMAL33",
        "timestamp": ts,
        "verdict": verdict,
        "guardrail_fail": guardrail_fail,
        "selected_candidate": selected_candidate,
        "next_step": "Option A/B/C/D — 사용자 결정 필요",
    }
    with open(REPORT_ROOT / "DONE.json", "w", encoding="utf-8") as f:
        json.dump(done, f, indent=2, ensure_ascii=False)

    print(f"[33] DONE. verdict={verdict}, guardrail_fail={guardrail_fail}")
    print(f"[33] package: {REPORT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
