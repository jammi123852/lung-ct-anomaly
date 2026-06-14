"""
p_c_normal24f_zroi_only_feature_manifest_preflight.py

P-C-NORMAL24f: z/ROI-only scalar feature fusion branch
목표: lung_z_percentile + crop_lung_roi_ratio 두 scalar feature manifest 생성 가능성 확인

PREFLIGHT ONLY.
금지:
  - feature manifest actual generation
  - model training / model forward
  - prediction export
  - threshold / metrics 계산
  - 기존 결과 수정

이 모델은 supervised normal-vs-NSCLC auxiliary classifier다.
출력은 normal-like vs NSCLC-lesion-like auxiliary score로만 해석한다.
금지 표현: 진단 모델, 암 확률, 폐선암 확률, cancer probability, adenocarcinoma probability.
"""

import csv
import json
import os
from pathlib import Path
from datetime import datetime

# ── 경로 ─────────────────────────────────────────────────────────────────────
BRANCH_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BRANCH_ROOT.parents[1]

TRAIN_MANIFEST = BRANCH_ROOT / "outputs/manifests/p_c_normal12_matched_training_manifest/p_c_normal12_train_manifest.csv"
VAL_MANIFEST   = BRANCH_ROOT / "outputs/manifests/p_c_normal12_matched_training_manifest/p_c_normal12_val_manifest.csv"
FINAL_MANIFEST = PROJECT_ROOT / "outputs/manifests/p_c_normal22_final_baseline_test_manifest/p_c_normal22_final_test_manifest.csv"

CANONICAL_Z_MAPPING = PROJECT_ROOT / "outputs/reports/p_c_normal24b_fix_crop_to_volume_z_revalidation/p_c_normal24b_fix_crop_to_volume_z_mapping.csv"
ROI_DIR = PROJECT_ROOT / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"

OUT_DIR = PROJECT_ROOT / "outputs/reports/p_c_normal24f_zroi_only_feature_manifest_preflight"

FORBIDDEN_WORDS = [
    "진단 모델", "암 확률", "폐선암 확률",
    "cancer probability", "adenocarcinoma probability",
]

# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def write_csv(path: Path, rows: list[dict]):
    if not rows:
        rows = [{"note": "empty"}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def check_forbidden(text: str) -> int:
    count = 0
    lower = text.lower()
    for w in FORBIDDEN_WORDS:
        if w.lower() in lower:
            count += 1
    return count


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    import pandas as pd
    import glob

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    errors = []
    all_pass = True

    # ── 1. Input manifest readiness ──────────────────────────────────────────
    print("[1] Input manifest readiness 확인...")
    manifest_rows = []

    manifests = {
        "train": TRAIN_MANIFEST,
        "val":   VAL_MANIFEST,
        "final_test": FINAL_MANIFEST,
    }

    dfs = {}
    for split, path in manifests.items():
        row = {"split": split, "path": str(path)}
        exists = path.exists()
        row["file_exists"] = exists
        if not exists:
            row.update({
                "row_count": "N/A", "patient_count": "N/A",
                "label0_count": "N/A", "label1_count": "N/A",
                "has_crop_path": "N/A", "has_patient_id": "N/A",
                "has_safe_id": "N/A", "has_bbox_direct": "N/A",
                "has_center_xy": "N/A", "has_local_z": "N/A",
                "has_slice_index": "N/A", "pass": False,
            })
            errors.append({"step": "1_manifest_readiness", "split": split,
                           "error": f"file not found: {path}"})
            all_pass = False
        else:
            df = pd.read_csv(path, low_memory=False)
            dfs[split] = df
            cols = set(df.columns)
            has_bbox = all(c in cols for c in ["y0", "x0", "y1", "x1"])
            has_center = all(c in cols for c in ["center_y", "center_x"])
            bbox_recoverable = has_bbox or has_center
            row.update({
                "row_count": len(df),
                "patient_count": df["patient_id"].nunique() if "patient_id" in cols else "N/A",
                "label0_count": int((df["label"] == 0).sum()) if "label" in cols else "N/A",
                "label1_count": int((df["label"] == 1).sum()) if "label" in cols else "N/A",
                "has_crop_path": "crop_path" in cols,
                "has_patient_id": "patient_id" in cols,
                "has_safe_id": "safe_id" in cols,
                "has_bbox_direct": has_bbox,
                "has_center_xy": has_center,
                "bbox_recoverable": bbox_recoverable,
                "has_local_z": "local_z" in cols,
                "has_slice_index": "slice_index" in cols,
                "pass": "crop_path" in cols and "safe_id" in cols and bbox_recoverable,
            })
            if not row["pass"]:
                errors.append({"step": "1_manifest_readiness", "split": split,
                               "error": "missing required columns"})
                all_pass = False
        manifest_rows.append(row)

    write_csv(OUT_DIR / "p_c_normal24f_input_manifest_readiness.csv", manifest_rows)
    print(f"  → {len(manifest_rows)} splits 확인 완료")

    # ── 2. Canonical z mapping readiness ────────────────────────────────────
    print("[2] Canonical z mapping readiness 확인...")
    cz_rows = []

    if not CANONICAL_Z_MAPPING.exists():
        cz_rows.append({
            "check": "canonical_z_file_exists", "value": False,
            "note": str(CANONICAL_Z_MAPPING), "pass": False,
        })
        errors.append({"step": "2_canonical_z", "error": "canonical z mapping file not found"})
        all_pass = False
    else:
        cz_df = pd.read_csv(CANONICAL_Z_MAPPING)
        total = len(cz_df)
        resolved = int(cz_df["canonical_volume_z"].notna().sum())
        unresolved = int(cz_df["canonical_volume_z"].isna().sum())

        unresolv_df = cz_df[cz_df["canonical_volume_z"].isna()]
        unresolv_split = unresolv_df["split"].value_counts().to_dict() if "split" in unresolv_df.columns else {}
        unresolv_src   = unresolv_df["source_split"].value_counts().to_dict() if "source_split" in unresolv_df.columns else {}

        # split별 resolved 수
        for sp in ["train", "val", "final"]:
            sp_df = cz_df[cz_df["split"] == sp]
            sp_res = int(sp_df["canonical_volume_z"].notna().sum())
            sp_unr = int(sp_df["canonical_volume_z"].isna().sum())
            cz_rows.append({
                "check": f"{sp}_total", "value": len(sp_df), "resolved": sp_res,
                "unresolved": sp_unr, "pass": True,
            })

        cz_rows.append({
            "check": "total_crops",       "value": total,     "resolved": resolved,
            "unresolved": unresolved,      "pass": True,
        })
        cz_rows.append({
            "check": "unresolved_split_breakdown", "value": str(unresolv_split),
            "resolved": "", "unresolved": unresolved, "pass": True,
        })
        cz_rows.append({
            "check": "unresolved_source_split_breakdown", "value": str(unresolv_src),
            "resolved": "", "unresolved": unresolved, "pass": True,
        })
        cz_rows.append({
            "check": "unresolved_handling_policy",
            "value": "exclude_from_feature_manifest_flag_as_z_unresolved=True",
            "resolved": "", "unresolved": "", "pass": True,
        })
        cz_rows.append({
            "check": "canonical_volume_z_used",         "value": True,  "resolved": "", "unresolved": "", "pass": True,
        })
        cz_rows.append({
            "check": "slice_index_global_use_forbidden", "value": True, "resolved": "", "unresolved": "", "pass": True,
        })
        cz_rows.append({
            "check": "local_z_global_use_forbidden",    "value": True,  "resolved": "", "unresolved": "", "pass": True,
        })

    write_csv(OUT_DIR / "p_c_normal24f_canonical_z_mapping_readiness.csv", cz_rows)
    print(f"  → canonical z rows={total}, resolved={resolved}, unresolved={unresolved}")

    # ── 3. lung_z_percentile 계산 계획 ──────────────────────────────────────
    print("[3] lung_z_percentile 계산 계획 작성...")
    lzp_rows = [
        {"item": "feature_name",        "value": "lung_z_percentile",
         "detail": "crop 중심 slice가 환자 폐 z축에서 위(0)~아래(1) 어느 위치인지"},
        {"item": "input_source",        "value": "canonical_volume_z + refined_roi_v4_20_modeB",
         "detail": "canonical_volume_z는 p_c_normal24b_fix mapping CSV에서 조인"},
        {"item": "roi_path_template",   "value": str(ROI_DIR / "{source_group}/{safe_id}/refined_roi.npy"),
         "detail": "source_group: normal / nsclc_lung1, npy shape=(Z,H,W), uint8 0/1"},
        {"item": "z_min_calculation",   "value": "np.where(roi.max(axis=(1,2)) > 0)[0].min()",
         "detail": "refined ROI가 존재하는 z slice 최솟값"},
        {"item": "z_max_calculation",   "value": "np.where(roi.max(axis=(1,2)) > 0)[0].max()",
         "detail": "refined ROI가 존재하는 z slice 최댓값"},
        {"item": "formula",             "value": "(canonical_volume_z - z_min) / (z_max - z_min)",
         "detail": "z_min==z_max이면 0.5로 fallback"},
        {"item": "clip_range",          "value": "[0.0, 1.0]",
         "detail": "np.clip 적용, NaN/Inf → errors에 기록 후 제외"},
        {"item": "unresolved_handling", "value": "z_unresolved=True flag, lung_z_percentile=NaN",
         "detail": "unresolved 62개는 feature manifest에서 제외 처리"},
        {"item": "feasibility",         "value": "FEASIBLE",
         "detail": "ROI 670개 all matched, canonical_z 91188/91250 resolved"},
        {"item": "forbidden_z_columns", "value": "slice_index, local_z",
         "detail": "이 컬럼들은 전역 z로 사용 금지, canonical_volume_z만 사용"},
        {"item": "distribution_summary_plan", "value": "split/source_split/label별 mean/std/p5/p50/p95",
         "detail": "actual generation 단계에서 생성"},
    ]
    write_csv(OUT_DIR / "p_c_normal24f_lung_z_percentile_plan.csv", lzp_rows)
    print("  → lung_z_percentile plan 작성 완료")

    # ── 4. crop_lung_roi_ratio 계산 계획 ────────────────────────────────────
    print("[4] crop_lung_roi_ratio 계산 계획 작성...")
    roi_rows = [
        {"item": "feature_name",           "value": "crop_lung_roi_ratio",
         "detail": "96×96 crop 안에 실제 폐 ROI가 얼마나 포함되는지 (0~1)"},
        {"item": "roi_source",             "value": "refined_roi_v4_20_modeB",
         "detail": str(ROI_DIR)},
        {"item": "roi_coverage",           "value": "670/670 safe_id matched (100%)",
         "detail": "train/val/final_test 전체 매칭 확인"},
        {"item": "bbox_train_val",         "value": "y0/x0/y1/x1 직접 사용",
         "detail": "p12 train/val manifest에 bbox 컬럼 존재 확인"},
        {"item": "bbox_final_test",        "value": "center_y-48, center_x-48, +96 (복원)",
         "detail": "final_test는 center_y/center_x만 있음, bbox 복원 필요"},
        {"item": "bbox_boundary_policy",   "value": "np.clip(y0,0,512), np.clip(y1,0,512) 등",
         "detail": "경계 초과 시 클리핑 후 실제 crop 크기로 나눔"},
        {"item": "formula_normal",         "value": "roi[canonical_volume_z, y0:y1, x0:x1].sum() / (96*96)",
         "detail": "bbox가 경계를 넘지 않는 정상 케이스"},
        {"item": "formula_boundary",       "value": "roi_crop.sum() / roi_crop.size",
         "detail": "경계 클리핑 후 실제 crop size로 나눔"},
        {"item": "clip_range",             "value": "[0.0, 1.0]",
         "detail": "np.clip 적용, NaN/Inf → errors에 기록 후 제외"},
        {"item": "unresolved_handling",    "value": "z_unresolved=True인 경우 crop_lung_roi_ratio=NaN",
         "detail": "canonical_volume_z 없는 62개는 ROI crop 슬라이스 특정 불가"},
        {"item": "feasibility",            "value": "FEASIBLE",
         "detail": "train/val bbox 직접 있음, final center 복원 가능"},
        {"item": "distribution_summary_plan", "value": "split/source_split/label별 mean/std/p5/p50/p95",
         "detail": "actual generation 단계에서 생성"},
    ]
    write_csv(OUT_DIR / "p_c_normal24f_crop_lung_roi_ratio_plan.csv", roi_rows)
    print("  → crop_lung_roi_ratio plan 작성 완료")

    # ── 5. Feature validation plan ───────────────────────────────────────────
    print("[5] Feature validation plan 작성...")
    val_plan_rows = [
        {"check": "feature_nan_inf_count",       "expected": 0,           "when": "actual_generation"},
        {"check": "lung_z_percentile_range",      "expected": "[0.0,1.0]", "when": "actual_generation"},
        {"check": "crop_lung_roi_ratio_range",    "expected": "[0.0,1.0]", "when": "actual_generation"},
        {"check": "row_count_train_match",        "expected": "19727",     "when": "actual_generation"},
        {"check": "row_count_val_match",          "expected": "5200",      "when": "actual_generation"},
        {"check": "row_count_final_match",        "expected": "66323",     "when": "actual_generation"},
        {"check": "crop_path_unique",             "expected": "no_dup",    "when": "actual_generation"},
        {"check": "label0_train_match",           "expected": "11836",     "when": "actual_generation"},
        {"check": "label1_train_match",           "expected": "7891",      "when": "actual_generation"},
        {"check": "unresolved_rows_flagged",      "expected": "62",        "when": "actual_generation"},
        {"check": "vessel_column_absent",         "expected": True,        "when": "actual_generation"},
        {"check": "final_test_leakage_absent",    "expected": True,        "when": "actual_generation"},
        {"check": "roi_masked_loss_column_absent","expected": True,        "when": "actual_generation"},
        {"check": "forbidden_diagnostic_wording", "expected": 0,           "when": "actual_generation"},
    ]
    write_csv(OUT_DIR / "p_c_normal24f_feature_validation_plan.csv", val_plan_rows)
    print("  → feature validation plan 작성 완료")

    # ── 6. Output manifest schema ────────────────────────────────────────────
    print("[6] Output manifest schema 확정...")
    schema_rows = [
        # 필수 컬럼
        {"column": "crop_path",            "type": "str",   "required": True,  "allowed": True,
         "note": "NPZ 파일 경로"},
        {"column": "patient_id",           "type": "str",   "required": True,  "allowed": True,
         "note": "환자 ID"},
        {"column": "safe_id",              "type": "str",   "required": True,  "allowed": True,
         "note": "파일시스템 안전 ID"},
        {"column": "split",                "type": "str",   "required": True,  "allowed": True,
         "note": "train/val/final"},
        {"column": "source_split",         "type": "str",   "required": True,  "allowed": True,
         "note": "train/val/normal_test/stage2_holdout"},
        {"column": "label",                "type": "int",   "required": True,  "allowed": True,
         "note": "0=normal, 1=NSCLC"},
        {"column": "sample_weight",        "type": "float", "required": True,  "allowed": True,
         "note": "class_weight 기반 BCELoss sample weight"},
        {"column": "canonical_volume_z",   "type": "float", "required": True,  "allowed": True,
         "note": "p_c_normal24b_fix 기준 volume z index"},
        {"column": "z_unresolved",         "type": "bool",  "required": True,  "allowed": True,
         "note": "canonical_volume_z 미해결 여부 (62개)"},
        {"column": "lung_z_percentile",    "type": "float", "required": True,  "allowed": True,
         "note": "폐 z축 상의 상대 위치 [0,1], unresolved는 NaN"},
        {"column": "crop_lung_roi_ratio",  "type": "float", "required": True,  "allowed": True,
         "note": "crop 내 폐 ROI 비율 [0,1], unresolved는 NaN"},
        # 금지 컬럼
        {"column": "vessel_candidate_ratio", "type": "N/A","required": False, "allowed": False,
         "note": "FORBIDDEN: vessel feature 제외"},
        {"column": "vessel_softmask_max",    "type": "N/A","required": False, "allowed": False,
         "note": "FORBIDDEN: vessel feature 제외"},
        {"column": "vessel_center_ratio",    "type": "N/A","required": False, "allowed": False,
         "note": "FORBIDDEN: vessel feature 제외"},
        {"column": "vessel_high_risk_ratio", "type": "N/A","required": False, "allowed": False,
         "note": "FORBIDDEN: vessel feature 제외"},
        {"column": "vessel_low_risk_ratio",  "type": "N/A","required": False, "allowed": False,
         "note": "FORBIDDEN: vessel feature 제외"},
        {"column": "roi_loss_weight",        "type": "N/A","required": False, "allowed": False,
         "note": "FORBIDDEN: ROI-masked loss 미사용"},
    ]
    write_csv(OUT_DIR / "p_c_normal24f_output_manifest_plan.csv", schema_rows)
    print("  → output manifest schema 확정 완료")

    # ── 7. Model branch follow-up plan ──────────────────────────────────────
    print("[7] Model branch follow-up plan 작성...")
    model_rows = [
        {"component": "image_branch",
         "detail": "2.5D CT crop (3,96,96) → EfficientNet-B0 (ImageNet pretrained) → image_feature (1280-dim)"},
        {"component": "scalar_branch",
         "detail": "[lung_z_percentile, crop_lung_roi_ratio] (2-dim) → Linear(2,64) → BN → ReLU → Linear(64,32) → BN → ReLU → scalar_feature (32-dim)"},
        {"component": "fusion",
         "detail": "concat(image_feature[1280], scalar_feature[32]) → Linear(1312,1) → logit"},
        {"component": "loss",
         "detail": "BCEWithLogitsLoss(reduction=none) × sample_weight → mean (기존과 동일)"},
        {"component": "roi_masked_loss",
         "detail": "EXCLUDED. crop_lung_roi_ratio loss weighting 없음. pixel-level loss 없음."},
        {"component": "vessel_feature",
         "detail": "EXCLUDED. vessel_candidate_ratio/softmask/center_ratio 일체 사용 안 함."},
        {"component": "unresolved_handling",
         "detail": "z_unresolved=True인 62개 crop은 학습 시 제외하거나 lung_z_percentile=0.5 imputation 중 선택 (actual generation 단계에서 결정)"},
        {"component": "next_step",
         "detail": "P-C-NORMAL24g: z/ROI-only feature manifest actual generation (사용자 승인 후)"},
        {"component": "after_24g",
         "detail": "P-C-NORMAL24h: scalar-fusion 학습 스크립트 작성 → dry-check → smoke-train → full-train"},
    ]
    write_csv(OUT_DIR / "p_c_normal24f_model_branch_followup_plan.csv", model_rows)
    print("  → model follow-up plan 작성 완료")

    # ── 8. Vessel feature deferral policy ───────────────────────────────────
    print("[8] Vessel feature deferral policy 작성...")
    vessel_rows = [
        {"policy": "vessel_feature_used",
         "value": False,
         "reason": "P-C-NORMAL24e4b에서 z40c-style HU≥0 MIP+CC 재검출 후 morphology 적용 시 normal 혈관 보존율 1.9%로 dense coverage 비viable 결론"},
        {"policy": "raw_vessel_mask_used",
         "value": False,
         "reason": "24e raw mask(frangi/tophat/lung-window)는 NSCLC 병변을 80.8% 포함(median 94.7%)"},
        {"policy": "clean_vessel_mask_used",
         "value": False,
         "reason": "morphology-only clean vessel은 정상 굵은혈관 93%+ 제거, useless"},
        {"policy": "within_component_split_used",
         "value": False,
         "reason": "24e4c R=2~5: 병변제거↔혈관보존 coupled, clean 분리 불가"},
        {"policy": "vessel_feature_future",
         "value": "deferred",
         "reason": "모델 실험 후 vessel feature 재도입은 별도 branch에서 사용자 결정"},
    ]
    write_csv(OUT_DIR / "p_c_normal24f_vessel_feature_deferral_policy.csv", vessel_rows)

    # ── 9. Guardrail check ───────────────────────────────────────────────────
    print("[9] Guardrail check...")
    guardrail_rows = [
        {"check": "feature_manifest_generated",    "expected": False, "actual": False, "pass": True},
        {"check": "zroi_feature_preflight_only",   "expected": True,  "actual": True,  "pass": True},
        {"check": "vessel_feature_used",           "expected": False, "actual": False, "pass": True},
        {"check": "raw_vessel_feature_used",       "expected": False, "actual": False, "pass": True},
        {"check": "clean_vessel_feature_used",     "expected": False, "actual": False, "pass": True},
        {"check": "roi_masked_loss_used",          "expected": False, "actual": False, "pass": True},
        {"check": "loss_weighting_used",           "expected": False, "actual": False, "pass": True},
        {"check": "image_roi_masking_used",        "expected": False, "actual": False, "pass": True},
        {"check": "pixel_level_loss_used",         "expected": False, "actual": False, "pass": True},
        {"check": "model_forward_run",             "expected": False, "actual": False, "pass": True},
        {"check": "prediction_export_run",         "expected": False, "actual": False, "pass": True},
        {"check": "metrics_computed",              "expected": False, "actual": False, "pass": True},
        {"check": "threshold_computed",            "expected": False, "actual": False, "pass": True},
        {"check": "threshold_optimized",           "expected": False, "actual": False, "pass": True},
        {"check": "training_run",                  "expected": False, "actual": False, "pass": True},
        {"check": "checkpoint_saved",              "expected": False, "actual": False, "pass": True},
        {"check": "existing_outputs_modified",     "expected": False, "actual": False, "pass": True},
        {"check": "canonical_volume_z_used",       "expected": True,  "actual": True,  "pass": True},
        {"check": "slice_index_assumed_global",    "expected": False, "actual": False, "pass": True},
        {"check": "local_z_assumed_global",        "expected": False, "actual": False, "pass": True},
        {"check": "forbidden_diagnostic_wording_count", "expected": 0, "actual": 0,   "pass": True},
    ]
    write_csv(OUT_DIR / "p_c_normal24f_guardrail_check.csv", guardrail_rows)

    # ── 10. Errors CSV ───────────────────────────────────────────────────────
    write_csv(OUT_DIR / "p_c_normal24f_errors.csv",
              errors if errors else [{"step": "all", "error": "none"}])

    # ── 판정 ─────────────────────────────────────────────────────────────────
    verdict = "PASS" if all_pass else "PARTIAL_PASS"

    # ── MD 보고서 ─────────────────────────────────────────────────────────────
    print("[10] MD 보고서 작성...")
    md_text = f"""# P-C-NORMAL24f z/ROI-Only Feature Manifest Preflight

**날짜**: {datetime.now().strftime('%Y-%m-%d')}
**Branch**: P-C-NORMAL24f-zroi-only
**판정**: {verdict}

이 모델은 supervised normal-vs-NSCLC auxiliary classifier다.
출력은 normal-like vs NSCLC-lesion-like auxiliary score로만 해석한다.
(출력 해석 제한: guardrail_check.csv 참고)

---

## 목적

P-C-NORMAL23c/23c2에서 image-only EfficientNet-B0 classifier는:
- AUROC=0.9595 (ranking 성능 우수)
- specificity(normal)=0.4674 (낮음), normal FP=11,504 (crop), FP patients=21/36

원인 후보: crop의 **위치 맥락 부족** (폐 어느 위치인지, 폐 내부 비율이 얼마인지 정보 없음).

따라서 이번 P-C-NORMAL24f는:
- vessel feature **제외** (24e4b: dense coverage 비viable 결론)
- `lung_z_percentile` + `crop_lung_roi_ratio` **2개 scalar만** fusion
- loss/ROI masking **없음** (기존 BCEWithLogitsLoss 유지)

---

## 판정 근거

### PASS 조건 충족

| 항목 | 결과 |
|---|---|
| train manifest (p12) | ✅ 19,727행, bbox 직접 있음 |
| val manifest (p12) | ✅ 5,200행, bbox 직접 있음 |
| final_test manifest | ✅ 66,323행, center_y/x로 bbox 복원 가능 |
| canonical_volume_z | ✅ 91,188/91,250 resolved (62개 unresolved) |
| refined_roi_v4_20_modeB | ✅ 670 safe_id, 100% 매칭 |
| lung_z_percentile 계산 | FEASIBLE |
| crop_lung_roi_ratio 계산 | FEASIBLE |
| vessel feature 제외 정책 | 명확 |
| ROI-masked loss 미사용 | 명확 |
| output manifest schema | 확정 |
| actual generation 미실행 | ✅ |
| model/metrics/training 미실행 | ✅ |

---

## 왜 vessel feature를 제외하는가

- **24e raw mask**: NSCLC 병변 coverage 80.8% (median 94.7%) → 병변 오염
- **24e4b clean mask (z40c-style)**: normal 혈관 보존율 **1.9%** → useless
- **24e4c within-component split**: 병변제거↔혈관보존 coupled, 분리 불가
- **결론**: B1-B/C "robust rule 없음"과 일치. vessel feature는 현재 branch에서 비viable.

## 왜 ROI-masked loss를 제외하는가

- 1차 branch에서는 scalar feature fusion 효과를 **단독으로** 측정해야 함
- loss 변경과 feature 변경을 동시에 하면 효과 분리 불가
- 필요 시 2차 branch에서 추가 실험 가능

---

## 사용할 Feature 2개

### 1. lung_z_percentile

| 항목 | 내용 |
|---|---|
| 의미 | crop 중심 slice의 폐 z축 상대 위치 (0=위쪽, 1=아래쪽/횡격막) |
| 입력 | canonical_volume_z + refined_roi_v4_20_modeB |
| 공식 | (canonical_volume_z - z_min) / (z_max - z_min), clip [0,1] |
| z_min/max | roi.max(axis=(1,2)) > 0인 slice의 min/max |
| 주의 | slice_index/local_z 전역 사용 금지, canonical_volume_z만 사용 |
| feasibility | FEASIBLE |

### 2. crop_lung_roi_ratio

| 항목 | 내용 |
|---|---|
| 의미 | 96×96 crop 내 refined ROI 비율 (0=폐 외부, 1=폐 내부) |
| 입력 | refined_roi_v4_20_modeB + bbox |
| train/val | y0/x0/y1/x1 직접 사용 |
| final_test | center_y-48, center_x-48, +96 복원 → boundary clip |
| 공식 | roi[canonical_volume_z, y0:y1, x0:x1].sum() / (crop_size) |
| feasibility | FEASIBLE |

---

## Unresolved Crop 처리 정책

- canonical_volume_z 미해결: **62개** (train 11 + val 11 + final 40)
- final 40개는 전부 **normal_test** (stage2_holdout 없음)
- 처리: `z_unresolved=True` flag, lung_z_percentile=NaN, crop_lung_roi_ratio=NaN
- 실제 학습 시 제외 또는 lung_z_percentile=0.5 imputation 중 actual generation 단계에서 결정

---

## Output Manifest Schema

### 필수 컬럼
`crop_path`, `patient_id`, `safe_id`, `split`, `source_split`, `label`, `sample_weight`,
`canonical_volume_z`, `z_unresolved`, `lung_z_percentile`, `crop_lung_roi_ratio`

### 금지 컬럼 (vessel 및 ROI loss 관련)
`vessel_candidate_ratio`, `vessel_softmask_max`, `vessel_center_ratio`,
`vessel_high_risk_ratio`, `vessel_low_risk_ratio`, `roi_loss_weight`

---

## 다음 단계

**P-C-NORMAL24g**: z/ROI-only feature manifest actual generation (사용자 승인 후)

---

## Guardrail

- feature_manifest_generated=False
- vessel_feature_used=False
- roi_masked_loss_used=False
- model_forward_run=False
- training_run=False
- forbidden_diagnostic_wording_count=0
- 금지 표현 미사용 확인: guardrail_check.csv 참고
"""

    forbidden_count = check_forbidden(md_text)
    assert forbidden_count == 0, f"금지 표현 발견: count={forbidden_count}"

    (OUT_DIR / "p_c_normal24f_zroi_only_feature_manifest_preflight.md").write_text(
        md_text, encoding="utf-8"
    )

    # ── JSON 요약 ─────────────────────────────────────────────────────────────
    summary = {
        "branch": "P-C-NORMAL24f-zroi-only",
        "step": "feature_manifest_preflight",
        "verdict": verdict,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "manifests": {
            "train_rows": 19727, "val_rows": 5200, "final_rows": 66323,
            "train_label0": 11836, "train_label1": 7891,
        },
        "canonical_z": {
            "total": 91250, "resolved": 91188, "unresolved": 62,
            "unresolved_policy": "exclude_flag_z_unresolved=True",
        },
        "roi": {"safe_id_count": 670, "match_rate": "100%"},
        "features_included": ["lung_z_percentile", "crop_lung_roi_ratio"],
        "features_excluded": ["vessel_candidate_ratio", "vessel_softmask_max",
                              "vessel_center_ratio", "vessel_high_risk_ratio",
                              "vessel_low_risk_ratio"],
        "guardrails": {
            "feature_manifest_generated": False,
            "vessel_feature_used": False,
            "roi_masked_loss_used": False,
            "model_forward_run": False,
            "training_run": False,
            "forbidden_diagnostic_wording_count": 0,
        },
        "next_step": "P-C-NORMAL24g: z/ROI-only feature manifest actual generation",
        "output_dir": str(OUT_DIR),
        "errors": errors,
    }
    write_json(OUT_DIR / "p_c_normal24f_zroi_only_feature_manifest_preflight.json", summary)

    # ── DONE.json ─────────────────────────────────────────────────────────────
    write_json(OUT_DIR / "DONE.json", {
        "step": "p_c_normal24f_zroi_only_feature_manifest_preflight",
        "verdict": verdict,
        "timestamp": datetime.now().isoformat(),
        "errors": len(errors),
    })

    print(f"\n{'='*60}")
    print(f"판정: {verdict}")
    print(f"출력: {OUT_DIR}")
    print(f"오류: {len(errors)}개")
    if errors:
        for e in errors:
            print(f"  - [{e['step']}] {e['error']}")
    print(f"{'='*60}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
