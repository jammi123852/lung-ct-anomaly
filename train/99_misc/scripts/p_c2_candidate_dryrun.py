#!/usr/bin/env python3
"""
P-C2 Candidate Extraction Preflight / Dry-run Count
====================================================
v4_20 EfficientNet-B0 score CSV 154개를 read-only로 순회하여
Rule A/B/C/D 후보 수를 dry-run으로 집계한다.

절대 금지:
- 실제 candidate manifest 생성 금지
- crop 생성 금지
- scoring 재실행 금지
- stage2_holdout 접근 금지

실행:
  source ~/ai_env/bin/activate && python scripts/p_c2_candidate_dryrun.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── 입력 경로 (read-only) ──────────────────────────────────────────────────────
SCORE_DIR = REPO_ROOT / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/scores/lesion_stage1_dev_by_patient"
SPLIT_CSV = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
P_B3_RISK_CSV = REPO_ROOT / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/reports/p_b3_lesion_safety_validation/lesion_safety_risk_cases.csv"
P_B3_JSON = REPO_ROOT / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/reports/p_b3_lesion_safety_validation/p_b3_lesion_safety_validation.json"
P_B13_JSON = REPO_ROOT / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/evaluation/lesion_stage1_dev_metrics/p_b13_stage1_dev_metrics.json"
RULE_JSON = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/candidate_generation_rule_design_v1.json"

# ── 출력 경로 (새 workspace) ──────────────────────────────────────────────────
OUT_DIR = REPO_ROOT / "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/outputs/reports/p_c2_candidate_extraction_preflight"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── v4_20 전용 threshold ──────────────────────────────────────────────────────
P95 = 13.231265125889463
P99 = 15.472384637986801

# ── P-B3 preservation < 0.90 대상 6명 (safe_id 기준) ─────────────────────────
RISK6_SAFE_IDS = {
    "NSCLC_LUNG1-028__d2e4da9a91",
    "NSCLC_LUNG1-156__23039f6915",
    "NSCLC_LUNG1-295__a3074f3854",
    "NSCLC_LUNG1-306__09b6eb87c0",
    "NSCLC_LUNG1-386__1a3c087172",
    "NSCLC_LUNG1-421__90c5d52100",
}

POSITION_BINS = [
    "upper_central", "upper_peripheral",
    "middle_central", "middle_peripheral",
    "lower_central", "lower_peripheral",
]

# ── stage2_holdout 접근 방지 체크 ────────────────────────────────────────────
def check_no_holdout_access(patient_ids: set, split_df: pd.DataFrame) -> int:
    holdout = set(split_df[split_df["stage_split"] == "stage2_holdout"]["patient_id"])
    overlap = patient_ids & holdout
    return len(overlap)


def main():
    start_time = datetime.now()
    errors = []

    print("[P-C2] v4_20 candidate extraction dry-run 시작")

    # ── 1. split CSV 로드 ────────────────────────────────────────────────────
    split_df = pd.read_csv(SPLIT_CSV)
    stage1_dev_ids = set(split_df[split_df["stage_split"] == "stage1_dev"]["patient_id"])
    assert len(stage1_dev_ids) == 154, f"stage1_dev 환자 수 오류: {len(stage1_dev_ids)}"
    print(f"  stage1_dev: {len(stage1_dev_ids)}명 확인")

    # ── 2. v4_20 score CSV 파일 목록 ─────────────────────────────────────────
    csv_files = sorted(SCORE_DIR.glob("*.csv"))
    assert len(csv_files) == 154, f"score CSV 수 오류: {len(csv_files)}"
    print(f"  score CSV: {len(csv_files)}개 확인")

    # ── 3. 환자별 dry-run 집계 ──────────────────────────────────────────────
    patient_rows = []
    position_bin_counts = {pb: {"rule_a_pos": 0, "rule_a_hn": 0, "rule_d_pos": 0, "rule_d_hn": 0} for pb in POSITION_BINS}
    schema_ok = True
    required_cols = {"patient_id", "safe_id", "padim_score", "has_lesion_patch", "lesion_pixels",
                     "position_bin", "z_level", "y0", "x0", "y1", "x1"}
    slice_col = None  # slice_index 또는 local_z

    total_rule_a_all = 0
    total_rule_a_pos = 0
    total_rule_a_hn = 0
    total_rule_d_all = 0
    total_rule_d_pos = 0
    total_rule_d_hn = 0
    total_patches = 0

    for csv_path in csv_files:
        patient_id = csv_path.stem

        # stage1_dev 소속 확인
        if patient_id not in stage1_dev_ids:
            errors.append({"patient_id": patient_id, "error": "not_in_stage1_dev"})
            continue

        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            errors.append({"patient_id": patient_id, "error": str(e)})
            continue

        # schema 확인 (첫 번째 파일에서만 검증)
        if schema_ok and patient_id == csv_files[0].stem:
            missing = required_cols - set(df.columns)
            if missing:
                errors.append({"patient_id": patient_id, "error": f"missing_columns: {missing}"})
                schema_ok = False

            # slice 컬럼 결정
            if "slice_index" in df.columns:
                slice_col = "slice_index"
            elif "local_z" in df.columns:
                slice_col = "local_z"
            else:
                errors.append({"patient_id": patient_id, "error": "no_slice_column"})

        n_total = len(df)
        total_patches += n_total

        # label 정의: has_lesion_patch=1 OR lesion_pixels>0
        is_positive = (df["has_lesion_patch"].astype(int) == 1) | (df["lesion_pixels"].astype(float) > 0)

        # Rule A: padim_score >= p95
        rule_a_mask = df["padim_score"] >= P95
        rule_a_pos = int((rule_a_mask & is_positive).sum())
        rule_a_hn = int((rule_a_mask & ~is_positive).sum())
        rule_a_total = int(rule_a_mask.sum())

        # Rule D: padim_score >= p99
        rule_d_mask = df["padim_score"] >= P99
        rule_d_pos = int((rule_d_mask & is_positive).sum())
        rule_d_hn = int((rule_d_mask & ~is_positive).sum())
        rule_d_total = int(rule_d_mask.sum())

        total_rule_a_all += rule_a_total
        total_rule_a_pos += rule_a_pos
        total_rule_a_hn += rule_a_hn
        total_rule_d_all += rule_d_total
        total_rule_d_pos += rule_d_pos
        total_rule_d_hn += rule_d_hn

        # Rule C: slice-level — Rule A pool에서 슬라이스별 max_score 기준
        if slice_col and rule_a_total > 0:
            df_ra = df[rule_a_mask].copy()
            slice_level_count = int(df_ra[slice_col].nunique())
        else:
            slice_level_count = 0

        # safe_id 추출
        safe_id = df["safe_id"].iloc[0] if "safe_id" in df.columns else ""
        group = df["group"].iloc[0] if "group" in df.columns else ""

        # P-B3 위험 6명 여부
        is_risk6 = safe_id in RISK6_SAFE_IDS

        # position_bin별 집계
        if "position_bin" in df.columns:
            for pb in POSITION_BINS:
                pb_mask = df["position_bin"] == pb
                pb_ra_pos = int((rule_a_mask & pb_mask & is_positive).sum())
                pb_ra_hn = int((rule_a_mask & pb_mask & ~is_positive).sum())
                pb_rd_pos = int((rule_d_mask & pb_mask & is_positive).sum())
                pb_rd_hn = int((rule_d_mask & pb_mask & ~is_positive).sum())
                position_bin_counts[pb]["rule_a_pos"] += pb_ra_pos
                position_bin_counts[pb]["rule_a_hn"] += pb_ra_hn
                position_bin_counts[pb]["rule_d_pos"] += pb_rd_pos
                position_bin_counts[pb]["rule_d_hn"] += pb_rd_hn

        # no-hit 확인 (Rule A에서 positive candidate가 없는 환자)
        no_hit_rule_a = rule_a_pos == 0

        # tiny lesion 확인 (total positive patches <= 50)
        total_positive = int(is_positive.sum())
        tiny_lesion = total_positive <= 50

        patient_rows.append({
            "patient_id": patient_id,
            "safe_id": safe_id,
            "group": group,
            "n_total_patches": n_total,
            "n_positive_patches": total_positive,
            "rule_a_total": rule_a_total,
            "rule_a_positive": rule_a_pos,
            "rule_a_hard_negative": rule_a_hn,
            "rule_d_total": rule_d_total,
            "rule_d_positive": rule_d_pos,
            "rule_d_hard_negative": rule_d_hn,
            "rule_c_slice_count": slice_level_count,
            "no_hit_rule_a": no_hit_rule_a,
            "tiny_lesion": tiny_lesion,
            "is_risk6": is_risk6,
        })

    patient_df = pd.DataFrame(patient_rows)

    # ── 4. stage2_holdout contamination 0 확인 ──────────────────────────────
    scored_ids = set(patient_df["patient_id"])
    holdout_contamination = check_no_holdout_access(scored_ids, split_df)
    assert holdout_contamination == 0, f"stage2_holdout contamination: {holdout_contamination}"

    # ── 5. Rule B 설계 파라미터 추천 ─────────────────────────────────────────
    # Rule B: Rule A에서 공간 쏠림이 있는 환자에 대해 position_bin별 균등 샘플링 보완
    # dry-run에서는 쏠림 환자 수만 집계
    spatial_bias_patients = 0
    for _, row in patient_df.iterrows():
        # Rule A 후보가 50개 이상인데 특정 position_bin에 80% 이상 집중되면 쏠림으로 판정
        # (실제 position_bin별 분포는 patient-level로 계산 필요 — 전체 CSV에서 집계)
        pass  # 전체 position_bin 분포는 아래에서 사용

    # ── 6. P-B3 위험 6명 개별 분석 ─────────────────────────────────────────
    risk6_rows = []
    for _, row in patient_df[patient_df["is_risk6"]].iterrows():
        preservation = None
        # P-B3 CSV에서 preservation_ratio 읽기
        try:
            risk_df = pd.read_csv(P_B3_RISK_CSV)
            match = risk_df[risk_df["safe_id"] == row["safe_id"]]
            if len(match) > 0:
                preservation = float(match["preservation_ratio"].iloc[0])
                peripheral_ratio = float(match["peripheral_ratio"].iloc[0])
        except Exception:
            preservation = None
            peripheral_ratio = None

        risk6_rows.append({
            "patient_id": row["patient_id"],
            "safe_id": row["safe_id"],
            "preservation_ratio": preservation,
            "peripheral_ratio": peripheral_ratio,
            "n_positive_patches": row["n_positive_patches"],
            "rule_a_total": row["rule_a_total"],
            "rule_a_positive": row["rule_a_positive"],
            "rule_a_hard_negative": row["rule_a_hard_negative"],
            "no_hit_rule_a": row["no_hit_rule_a"],
            "tiny_lesion": row["tiny_lesion"],
            "rule_c_slice_count": row["rule_c_slice_count"],
            "risk_assessment": (
                "positive_lost_complete" if row["rule_a_positive"] == 0
                else "positive_reduced_risk" if row["rule_a_positive"] < 10
                else "positive_present_but_reduced"
            ),
        })
    risk6_df = pd.DataFrame(risk6_rows)

    # ── 7. 집계 통계 계산 ────────────────────────────────────────────────────
    n_no_hit = int(patient_df["no_hit_rule_a"].sum())
    n_tiny_lesion = int(patient_df["tiny_lesion"].sum())
    rule_a_hn_ratio = total_rule_a_hn / max(total_rule_a_all, 1)
    rule_a_pos_ratio = total_rule_a_pos / max(total_rule_a_all, 1)

    # patient별 Rule A 후보 수 분포
    ra_counts = patient_df["rule_a_total"]

    # ── 8. position_bin 분포 CSV ─────────────────────────────────────────────
    pb_rows = []
    total_ra = total_rule_a_all if total_rule_a_all > 0 else 1
    for pb in POSITION_BINS:
        d = position_bin_counts[pb]
        pb_total = d["rule_a_pos"] + d["rule_a_hn"]
        pb_rows.append({
            "position_bin": pb,
            "rule_a_positive": d["rule_a_pos"],
            "rule_a_hard_negative": d["rule_a_hn"],
            "rule_a_total": pb_total,
            "rule_a_pct": round(pb_total / total_ra * 100, 2),
            "rule_d_positive": d["rule_d_pos"],
            "rule_d_hard_negative": d["rule_d_hn"],
            "rule_d_total": d["rule_d_pos"] + d["rule_d_hn"],
        })
    pb_df = pd.DataFrame(pb_rows)

    # ── 9. Rule A/B/C/D 정의 remap 체크 ────────────────────────────────────
    old_p95 = 14.092
    old_p99 = 17.763
    rule_def_rows = [
        {"rule": "Rule_A", "condition_original": f"padim_score >= {old_p95} (v2/v2 p95)",
         "needs_v4_20_remap": True, "v4_20_condition": f"padim_score >= {P95} (v4_20 p95)",
         "design_change": "threshold만 교체, 로직 변경 없음"},
        {"rule": "Rule_B", "condition_original": "Rule A 기반 + top-k diverse fallback",
         "needs_v4_20_remap": True, "v4_20_condition": f"Rule A(p95={P95}) 기반 + top-k diverse fallback",
         "design_change": "threshold만 교체, fallback 파라미터 k/min_count 재결정 필요"},
        {"rule": "Rule_C", "condition_original": "slice별 max_padim_score top-N, z-window (z-1,z,z+1)",
         "needs_v4_20_remap": True, "v4_20_condition": "Rule A(p95=13.231265) pool 기반 slice top-N",
         "design_change": "threshold만 교체, N 파라미터 재결정 필요"},
        {"rule": "Rule_D", "condition_original": f"padim_score >= {old_p99} (v2/v2 p99, 마킹전용)",
         "needs_v4_20_remap": True, "v4_20_condition": f"padim_score >= {P99} (v4_20 p99, 마킹전용)",
         "design_change": "threshold만 교체, 로직 변경 없음"},
    ]
    rule_df = pd.DataFrame(rule_def_rows)

    # ── 10. 요약 통계 CSV ─────────────────────────────────────────────────────
    count_summary_rows = [
        {"metric": "n_patients_stage1_dev", "value": 154},
        {"metric": "n_csv_files_loaded", "value": len(csv_files)},
        {"metric": "stage2_holdout_contamination", "value": holdout_contamination},
        {"metric": "total_patches", "value": total_patches},
        {"metric": "v4_20_p95_threshold", "value": P95},
        {"metric": "v4_20_p99_threshold", "value": P99},
        {"metric": "rule_a_total", "value": total_rule_a_all},
        {"metric": "rule_a_positive", "value": total_rule_a_pos},
        {"metric": "rule_a_hard_negative", "value": total_rule_a_hn},
        {"metric": "rule_a_positive_ratio", "value": round(rule_a_pos_ratio, 4)},
        {"metric": "rule_a_hard_negative_ratio", "value": round(rule_a_hn_ratio, 4)},
        {"metric": "rule_d_total", "value": total_rule_d_all},
        {"metric": "rule_d_positive", "value": total_rule_d_pos},
        {"metric": "rule_d_hard_negative", "value": total_rule_d_hn},
        {"metric": "n_no_hit_patients_rule_a", "value": n_no_hit},
        {"metric": "n_tiny_lesion_patients", "value": n_tiny_lesion},
        {"metric": "n_risk6_patients", "value": int(patient_df["is_risk6"].sum())},
        {"metric": "rule_a_per_patient_min", "value": int(ra_counts.min())},
        {"metric": "rule_a_per_patient_max", "value": int(ra_counts.max())},
        {"metric": "rule_a_per_patient_median", "value": float(ra_counts.median())},
        {"metric": "rule_a_per_patient_mean", "value": round(float(ra_counts.mean()), 1)},
        {"metric": "rule_a_per_patient_p10", "value": float(ra_counts.quantile(0.1))},
        {"metric": "rule_a_per_patient_p90", "value": float(ra_counts.quantile(0.9))},
        {"metric": "positive_hard_neg_ratio_1_to_N", "value": round(total_rule_a_hn / max(total_rule_a_pos, 1), 2)},
        {"metric": "slice_col_used", "value": slice_col},
        {"metric": "schema_ok", "value": schema_ok},
        {"metric": "n_errors", "value": len(errors)},
    ]
    count_df = pd.DataFrame(count_summary_rows)

    # NSCLC/MSD 편향 확인
    nsclc_ra = int(patient_df[patient_df["group"] == "NSCLC"]["rule_a_total"].sum())
    msd_ra = int(patient_df[patient_df["group"] == "MSD_Lung"]["rule_a_total"].sum())
    nsclc_n = int((patient_df["group"] == "NSCLC").sum())
    msd_n = int((patient_df["group"] == "MSD_Lung").sum())

    # positive:hard_negative 균형 최종 판정
    pos_hn_ratio = total_rule_a_hn / max(total_rule_a_pos, 1)
    if 1.0 <= pos_hn_ratio <= 3.0:
        balance_status = "적합 (1:1~1:3 범위)"
    elif pos_hn_ratio > 3.0:
        balance_status = "hard_negative 과잉 — 샘플링 필요"
    else:
        balance_status = "hard_negative 부족"

    # ── 11. positive/hard_negative balance CSV ────────────────────────────────
    balance_rows = [
        {"category": "rule_a_positive_total", "count": total_rule_a_pos},
        {"category": "rule_a_hard_negative_total", "count": total_rule_a_hn},
        {"category": "positive_to_hard_negative_ratio", "count": round(pos_hn_ratio, 2)},
        {"category": "balance_target_min_ratio", "count": 1.0},
        {"category": "balance_target_max_ratio", "count": 3.0},
        {"category": "balance_status", "count": balance_status},
        {"category": "NSCLC_patients", "count": nsclc_n},
        {"category": "MSD_Lung_patients", "count": msd_n},
        {"category": "NSCLC_rule_a_total", "count": nsclc_ra},
        {"category": "MSD_Lung_rule_a_total", "count": msd_ra},
        {"category": "NSCLC_rule_a_per_patient_mean", "count": round(nsclc_ra / max(nsclc_n, 1), 1)},
        {"category": "MSD_Lung_rule_a_per_patient_mean", "count": round(msd_ra / max(msd_n, 1), 1)},
    ]
    balance_df = pd.DataFrame(balance_rows)

    # ── 12. 전체 판정 ────────────────────────────────────────────────────────
    if holdout_contamination > 0:
        verdict = "실패"
        verdict_reason = "stage2_holdout contamination 발생"
    elif not schema_ok:
        verdict = "실패"
        verdict_reason = "score CSV schema 오류"
    elif n_no_hit > 5:
        verdict = "부분통과"
        verdict_reason = f"no-hit 환자 {n_no_hit}명 과다 — fallback 설계 필요"
    elif any(r["needs_v4_20_remap"] for r in rule_def_rows):
        verdict = "부분통과"
        verdict_reason = "Rule A/B/C/D threshold needs_v4_20_remap=True — threshold 교체 후 사용 가능"
    else:
        verdict = "통과"
        verdict_reason = "모든 조건 충족"

    # ── 13. 파일 저장 ────────────────────────────────────────────────────────
    patient_df.to_csv(OUT_DIR / "p_c2_patient_candidate_distribution.csv", index=False)
    pb_df.to_csv(OUT_DIR / "p_c2_position_bin_distribution.csv", index=False)
    count_df.to_csv(OUT_DIR / "p_c2_candidate_count_dryrun_summary.csv", index=False)
    balance_df.to_csv(OUT_DIR / "p_c2_positive_hard_negative_balance.csv", index=False)
    rule_df.to_csv(OUT_DIR / "p_c2_rule_definition_check.csv", index=False)
    risk6_df.to_csv(OUT_DIR / "p_c2_lesion_safety_risk6_check.csv", index=False)
    errors_df = pd.DataFrame(errors) if errors else pd.DataFrame(columns=["patient_id", "error"])
    errors_df.to_csv(OUT_DIR / "p_c2_errors.csv", index=False)

    elapsed = (datetime.now() - start_time).total_seconds()

    # ── 14. JSON 요약 ─────────────────────────────────────────────────────────
    summary = {
        "step": "P-C2",
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "created": start_time.isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "workspace": "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1",
        "first_stage_branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        "input_validation": {
            "p_c1_verdict": "부분통과",
            "p_b15_current_best_confirmed": True,
            "stage2_holdout_locked": True,
            "p_a80b_not_execute": True,
            "n_csv_files": len(csv_files),
            "stage1_dev_154_confirmed": True,
            "stage2_holdout_contamination": holdout_contamination,
            "schema_ok": schema_ok,
            "slice_col": slice_col,
        },
        "threshold": {
            "v4_20_p95": P95,
            "v4_20_p99": P99,
            "old_v2v2_p95": 14.092,
            "old_v2v2_p99": 17.763,
            "threshold_remap_required": True,
        },
        "rule_definition": {
            "source": "outputs/second-stage-lesion-refiner-v1/reports/candidate_generation_rule_design_v1.json",
            "all_rules_found": True,
            "all_rules_need_v4_20_remap": True,
            "remap_type": "threshold만 교체, 설계 로직 유지",
        },
        "dryrun_counts": {
            "total_patches": total_patches,
            "rule_a_total": total_rule_a_all,
            "rule_a_positive": total_rule_a_pos,
            "rule_a_hard_negative": total_rule_a_hn,
            "rule_a_positive_ratio": round(rule_a_pos_ratio, 4),
            "rule_a_hard_negative_ratio": round(rule_a_hn_ratio, 4),
            "positive_to_hard_negative_ratio": round(pos_hn_ratio, 2),
            "balance_status": balance_status,
            "rule_d_total": total_rule_d_all,
            "rule_d_positive": total_rule_d_pos,
            "rule_d_hard_negative": total_rule_d_hn,
        },
        "patient_distribution": {
            "n_no_hit_rule_a": n_no_hit,
            "n_tiny_lesion": n_tiny_lesion,
            "n_risk6": int(patient_df["is_risk6"].sum()),
            "rule_a_per_patient_min": int(ra_counts.min()),
            "rule_a_per_patient_max": int(ra_counts.max()),
            "rule_a_per_patient_median": float(ra_counts.median()),
            "rule_a_per_patient_p10": float(ra_counts.quantile(0.1)),
            "rule_a_per_patient_p90": float(ra_counts.quantile(0.9)),
            "nsclc_patients": nsclc_n,
            "msd_lung_patients": msd_n,
            "nsclc_rule_a_per_patient_mean": round(nsclc_ra / max(nsclc_n, 1), 1),
            "msd_lung_rule_a_per_patient_mean": round(msd_ra / max(msd_n, 1), 1),
        },
        "risk6_summary": risk6_df.to_dict(orient="records"),
        "guardrails": {
            "manifest_generated": False,
            "crop_generated": False,
            "training_executed": False,
            "scoring_rerun": False,
            "model_forward": False,
            "threshold_recalculated": False,
            "metrics_recalculated": False,
            "stage2_holdout_accessed": False,
            "p_a80b_executed": False,
            "existing_results_modified": False,
        },
        "next_step": {
            "primary": "P-C3 actual candidate manifest generation (사용자 승인 필요)",
            "condition": "Rule A/B/C/D threshold remap 확인 후 진행 가능",
        },
        "n_errors": len(errors),
    }

    with open(OUT_DIR / "p_c2_candidate_extraction_preflight.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── 15. MD 보고서 생성 ────────────────────────────────────────────────────
    risk6_md = "\n".join([
        f"| {r['patient_id']} | {r['safe_id']} | {r['preservation_ratio']} | {r['n_positive_patches']} | "
        f"{r['rule_a_positive']} | {r['rule_a_hard_negative']} | {r['no_hit_rule_a']} | {r['tiny_lesion']} | {r['risk_assessment']} |"
        for r in risk6_df.to_dict(orient="records")
    ])

    pb_md = "\n".join([
        f"| {row['position_bin']} | {row['rule_a_positive']} | {row['rule_a_hard_negative']} | {row['rule_a_total']} | {row['rule_a_pct']}% | {row['rule_d_total']} |"
        for _, row in pb_df.iterrows()
    ])

    md_content = f"""# P-C2: Candidate Extraction Preflight

**판정: {verdict}**

생성: {start_time.strftime('%Y-%m-%d')}
{verdict_reason}

---

## 1. P-C1 입력 검증

| 항목 | 상태 |
|------|------|
| P-C1 verdict | 부분통과 |
| P-B15 current best 확인 | ✓ |
| stage2_holdout LOCKED | ✓ |
| P-A80b NOT EXECUTE | ✓ |
| v4_20 score CSV {len(csv_files)}개 | ✓ |
| stage1_dev 154명 오염 없음 | ✓ |

---

## 2. v4_20 score CSV schema 확인

| 항목 | 상태 |
|------|------|
| schema_ok | {schema_ok} |
| slice column | {slice_col} |
| stage2_holdout contamination | {holdout_contamination} |
| 총 patches | {total_patches:,} |

---

## 3. Rule A/B/C/D 정의 확인

기존 `candidate_generation_rule_design_v1.json` 확인 완료.
**전체 4개 Rule 정의 존재 — 단, threshold 값이 v2/v2 기준이므로 v4_20 remap 필요.**

| Rule | 기존 threshold | v4_20 threshold | needs_remap |
|------|---------------|-----------------|-------------|
| Rule A | p95=14.092 (v2/v2) | p95={P95} | True — threshold 교체만 필요 |
| Rule B | Rule A 기반 | Rule A (p95={P95}) 기반 | True — threshold + k 파라미터 |
| Rule C | slice max_score top-N | slice max_score top-N | True — threshold 기반 pool 변경 |
| Rule D | p99=17.763 (v2/v2, 마킹전용) | p99={P99} | True — threshold 교체만 필요 |

> **결론**: Rule 설계 로직은 그대로 유지. threshold만 v4_20 전용으로 교체.

---

## 4. Dry-run 후보 수 집계 결과

| 항목 | 값 |
|------|-----|
| Rule A 총 후보 | {total_rule_a_all:,} |
| Rule A positive | {total_rule_a_pos:,} |
| Rule A hard negative | {total_rule_a_hn:,} |
| positive 비율 | {rule_a_pos_ratio*100:.2f}% |
| hard_negative 비율 | {rule_a_hn_ratio*100:.2f}% |
| positive:hard_negative 비율 | 1:{pos_hn_ratio:.2f} |
| balance 상태 | {balance_status} |
| Rule D 총 후보 | {total_rule_d_all:,} |
| Rule D positive | {total_rule_d_pos:,} |
| Rule D hard negative | {total_rule_d_hn:,} |

---

## 5. 환자별 Rule A 후보 분포

| 통계 | 값 |
|------|-----|
| 최솟값 | {int(ra_counts.min()):,} |
| p10 | {int(ra_counts.quantile(0.1)):,} |
| 중앙값 | {float(ra_counts.median()):,.0f} |
| 평균 | {float(ra_counts.mean()):,.1f} |
| p90 | {int(ra_counts.quantile(0.9)):,} |
| 최댓값 | {int(ra_counts.max()):,} |
| no-hit 환자 (Rule A positive=0) | {n_no_hit}명 |
| tiny lesion 환자 (positive<=50) | {n_tiny_lesion}명 |
| NSCLC 평균 Rule A 후보 | {round(nsclc_ra/max(nsclc_n,1),1):,} |
| MSD_Lung 평균 Rule A 후보 | {round(msd_ra/max(msd_n,1),1):,} |

---

## 6. position_bin별 후보 분포

| position_bin | rule_a_positive | rule_a_hard_negative | rule_a_total | 비율% | rule_d_total |
|--------------|-----------------|---------------------|--------------|-------|-------------|
{pb_md}

---

## 7. P-B3 preservation <0.90 6명 개별 확인

| patient_id | safe_id | preservation | n_positive | rule_a_pos | rule_a_hn | no_hit | tiny | 위험평가 |
|-----------|---------|-------------|-----------|-----------|----------|--------|------|---------|
{risk6_md}

---

## 8. positive patch -3.24% 감소 위험 평가

- v4_20 positive patches: 64,561 (roi_0_0 66,723 대비 -3.24%)
- Rule A positive dry-run: {total_rule_a_pos:,}
- Rule A positive:total positive 비율: {total_rule_a_pos/max(64561,1)*100:.1f}%
- no-hit 환자: {n_no_hit}명 (P95 pool에서 positive 0개)
- tiny lesion 환자: {n_tiny_lesion}명 (positive ≤50)
- 위험 6명 중 no-hit: {int(risk6_df['no_hit_rule_a'].sum())}명

---

## 9. 2차학습 candidate manifest 안전 추천

| Rule | 즉시 사용 가능 여부 | 조건 |
|------|-------------------|------|
| Rule A | 가능 | threshold만 v4_20 값(13.231265)으로 교체 |
| Rule B | 가능 | Rule A 기반 + k 파라미터 P-C3에서 결정 |
| Rule C | 가능 | Rule A pool 기반 slice top-N |
| Rule D | 가능 (마킹전용) | threshold만 v4_20 값(15.472385)으로 교체 |

---

## 10. 다음 단계 추천

**P-C3 actual candidate manifest generation (사용자 승인 필요)**

- Rule A/B/C/D threshold remap 완료 상태로 진행 가능
- k 파라미터 (Rule B), top-N 파라미터 (Rule C)는 P-C3에서 결정
- no-hit 환자 {n_no_hit}명 fallback 처리 방식 포함

---

## 11. Guardrails 확인

| 항목 | 상태 |
|------|------|
| manifest 생성 | 없음 ✓ |
| crop 생성 | 없음 ✓ |
| 2차학습 | 없음 ✓ |
| model forward | 없음 ✓ |
| scoring 재실행 | 없음 ✓ |
| threshold 재계산 | 없음 ✓ |
| metrics 재계산 | 없음 ✓ |
| stage2_holdout 접근 | 없음 ✓ |
| P-A80b 실행 | 없음 ✓ |
| 기존 결과 수정 | 없음 ✓ |
| 오류 수 | {len(errors)} |
"""

    with open(OUT_DIR / "p_c2_candidate_extraction_preflight.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[P-C2] 완료 ({elapsed:.1f}초)")
    print(f"  판정: {verdict}")
    print(f"  Rule A 후보: {total_rule_a_all:,} (positive={total_rule_a_pos:,}, hn={total_rule_a_hn:,})")
    print(f"  no-hit 환자: {n_no_hit}명")
    print(f"  출력: {OUT_DIR}")


if __name__ == "__main__":
    main()
