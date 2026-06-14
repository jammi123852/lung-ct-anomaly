"""
Phase 8.6C: per-scan proxy sensitivity / FP-burden analysis
- true FROC 계산 아님
- lesion-level sensitivity 계산 아님
- proxy-only analysis (PROXY_ONLY_APPROVED from Phase 8.6B)
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── 경로 정의 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/scores"
    / "phase8_4_stage2_full_scoring_v1/phase8_4_stage2_full_scoring_v1.csv"
)
PHASE86B_SUMMARY = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_6b_proxy_protocol_review_v1/phase8_6b_proxy_protocol_review_summary.json"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_6c_per_scan_proxy_analysis_v1"
)

SCORE_COLUMNS = {
    "primary": "mediastinal_channels_l1_mean",
    "comparison": ["crop_score_l1_mean", "crop_score_mse_mean"],
}
ALL_SCORE_COLS = [SCORE_COLUMNS["primary"]] + SCORE_COLUMNS["comparison"]

PROXY_NOTE = "proxy_only_not_true_froc"
THRESHOLD_GRID_N = 100  # quantile grid 개수

# ── output 디렉토리 collision guard ───────────────────────────────────────────
if OUTPUT_ROOT.exists():
    existing = list(OUTPUT_ROOT.iterdir())
    if existing:
        print(f"[ERROR] 출력 폴더가 이미 존재하고 파일이 있습니다: {OUTPUT_ROOT}")
        print("기존 output 덮어쓰기 금지 규칙에 따라 중단합니다.")
        sys.exit(1)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Phase 8.6B 사전 확인 ──────────────────────────────────────────────────────
with open(PHASE86B_SUMMARY) as f:
    b_summary = json.load(f)

assert b_summary["phase"] == "8.6B_proxy_protocol_review", "Phase 8.6B summary 불일치"
assert b_summary["phase_8_6b_verdict"] == "PROXY_ONLY_APPROVED", (
    f"Phase 8.6B verdict가 PROXY_ONLY_APPROVED가 아님: {b_summary['phase_8_6b_verdict']}"
)
assert b_summary["true_froc_calculated"] is False
assert b_summary["lesion_unique_id_available"] is False
assert b_summary["lesion_gt_count_available"] is False

print("[OK] Phase 8.6B PROXY_ONLY_APPROVED 확인")

# ── score CSV 로드 ─────────────────────────────────────────────────────────────
df = pd.read_csv(SCORE_CSV)
assert len(df) == 143735, f"score CSV 행 수 불일치: {len(df)}"
assert "sampling_label" in df.columns
assert "patient_id" in df.columns
for col in ALL_SCORE_COLS:
    assert col in df.columns, f"score column 없음: {col}"

# NaN/Inf 확인
for col in ALL_SCORE_COLS:
    assert not df[col].isna().any(), f"{col} NaN 있음"
    assert not np.isinf(df[col]).any(), f"{col} Inf 있음"

n_positive = (df["sampling_label"] == "positive").sum()
n_hn = (df["sampling_label"] == "hard_negative").sum()
n_patients = df["patient_id"].nunique()

print(f"[OK] score CSV 로드: {len(df)} rows, {n_patients} scans")
print(f"     positive crops: {n_positive}, hard_negative crops: {n_hn}")
print(f"[NOTE] lesion_patch_ratio 컬럼은 score CSV에 없음")
print(f"       Phase 8.6A에서 positive=51335 == lesion_patch_ratio>0=51335 확인됨")
print(f"       proxy hit 정의: sampling_label == positive 로 대체 사용")

pos_df = df[df["sampling_label"] == "positive"].copy()
hn_df = df[df["sampling_label"] == "hard_negative"].copy()

# ── 1. scan_level_proxy_summary.csv ───────────────────────────────────────────
print("\n[1/4] scan_level_proxy_summary 계산 중...")

pos_grp = pos_df.groupby("patient_id")[SCORE_COLUMNS["primary"]].agg(
    positive_score_max="max",
    positive_score_mean="mean",
    positive_crop_count="count",
).reset_index()
pos_grp_all = {}
for col in ALL_SCORE_COLS:
    pos_grp_all[col] = pos_df.groupby("patient_id")[col].agg(
        **{f"pos_{col}_max": "max", f"pos_{col}_mean": "mean"}
    ).reset_index()

hn_grp = hn_df.groupby("patient_id")[SCORE_COLUMNS["primary"]].agg(
    hard_negative_score_max="max",
    hard_negative_score_mean="mean",
    hard_negative_crop_count="count",
).reset_index()

scan_summary = pos_grp.merge(hn_grp, on="patient_id", how="outer")
scan_summary = scan_summary.rename(columns={"patient_id": "scan_id"})
# column 순서 정리
scan_summary = scan_summary[
    [
        "scan_id",
        "positive_crop_count",
        "hard_negative_crop_count",
        "positive_score_max",
        "hard_negative_score_max",
        "positive_score_mean",
        "hard_negative_score_mean",
    ]
]

out_scan = OUTPUT_ROOT / "scan_level_proxy_summary.csv"
scan_summary.to_csv(out_scan, index=False)
print(f"    저장: {out_scan.name}  ({len(scan_summary)} rows)")

# ── 2. threshold_proxy_curve.csv ──────────────────────────────────────────────
print("\n[2/4] threshold_proxy_curve 계산 중...")

positive_scans = set(pos_df["patient_id"].unique())
n_positive_scans = len(positive_scans)

curve_rows = []
for col in ALL_SCORE_COLS:
    all_scores = df[col].values
    thresholds = np.quantile(all_scores, np.linspace(0.0, 1.0, THRESHOLD_GRID_N + 1))
    thresholds = np.unique(thresholds)

    col_pos = pos_df.groupby("patient_id")[col].max()
    col_hn_counts = hn_df.groupby("patient_id")[col]

    for t in thresholds:
        # scan-level proxy hit: positive scan 중 max-score >= threshold
        n_hits = (col_pos >= t).sum()
        proxy_sensitivity = n_hits / n_positive_scans if n_positive_scans > 0 else 0.0

        # FP crop burden per scan
        fp_counts = col_hn_counts.apply(lambda x: (x >= t).sum())
        mean_fp = float(fp_counts.mean())
        median_fp = float(fp_counts.median())
        scans_with_fp = int((fp_counts > 0).sum())
        fp_scan_rate = scans_with_fp / n_patients if n_patients > 0 else 0.0

        curve_rows.append(
            {
                "score_column": col,
                "threshold": float(t),
                "scan_proxy_sensitivity": float(proxy_sensitivity),
                "mean_fp_crop_burden_per_scan": mean_fp,
                "median_fp_crop_burden_per_scan": median_fp,
                "scans_with_any_fp_proxy": scans_with_fp,
                "fp_proxy_scan_rate": fp_scan_rate,
                "note": PROXY_NOTE,
            }
        )

curve_df = pd.DataFrame(curve_rows)
out_curve = OUTPUT_ROOT / "threshold_proxy_curve.csv"
curve_df.to_csv(out_curve, index=False)
print(f"    저장: {out_curve.name}  ({len(curve_df)} rows, {len(ALL_SCORE_COLS)} columns)")

# ── 3. score_column_proxy_comparison.csv ─────────────────────────────────────
print("\n[3/4] score_column_proxy_comparison 계산 중...")

comp_rows = []
for col in ALL_SCORE_COLS:
    pos_scores = pos_df[col]
    hn_scores = hn_df[col]
    pos_summary = {
        "mean": round(float(pos_scores.mean()), 6),
        "std": round(float(pos_scores.std()), 6),
        "p25": round(float(pos_scores.quantile(0.25)), 6),
        "median": round(float(pos_scores.median()), 6),
        "p75": round(float(pos_scores.quantile(0.75)), 6),
        "max": round(float(pos_scores.max()), 6),
    }
    hn_summary = {
        "mean": round(float(hn_scores.mean()), 6),
        "std": round(float(hn_scores.std()), 6),
        "p25": round(float(hn_scores.quantile(0.25)), 6),
        "median": round(float(hn_scores.median()), 6),
        "p75": round(float(hn_scores.quantile(0.75)), 6),
        "max": round(float(hn_scores.max()), 6),
    }
    # separation: (pos_mean - hn_mean) / (pos_std + hn_std)
    sep = (pos_scores.mean() - hn_scores.mean()) / (pos_scores.std() + hn_scores.std() + 1e-9)
    comp_rows.append(
        {
            "score_column": col,
            "positive_scan_score_summary": json.dumps(pos_summary),
            "hard_negative_scan_score_summary": json.dumps(hn_summary),
            "separation_summary": round(float(sep), 6),
            "interpretation_limit": (
                "proxy only: positive=sampling_label==positive, "
                "not lesion_level_sensitivity, not true_froc"
            ),
        }
    )

comp_df = pd.DataFrame(comp_rows)
out_comp = OUTPUT_ROOT / "score_column_proxy_comparison.csv"
comp_df.to_csv(out_comp, index=False)
print(f"    저장: {out_comp.name}  ({len(comp_df)} rows)")

# ── 4. summary JSON ────────────────────────────────────────────────────────────
print("\n[4/4] summary JSON 및 report 저장 중...")

# 각 score column의 proxy sensitivity @ fixed FP burden 계산
sensitivity_at_fp = {}
for col in ALL_SCORE_COLS:
    col_curve = curve_df[curve_df["score_column"] == col].sort_values("threshold")
    sens_at = {}
    for fp_target in [1, 2, 3, 5, 10]:
        candidates = col_curve[col_curve["mean_fp_crop_burden_per_scan"] <= fp_target]
        if len(candidates) > 0:
            sens_at[f"sens_at_fp{fp_target}"] = round(
                float(candidates["scan_proxy_sensitivity"].max()), 4
            )
        else:
            sens_at[f"sens_at_fp{fp_target}"] = None
    sensitivity_at_fp[col] = sens_at

summary = {
    "phase": "8.6C",
    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "analysis_type": "per_scan_proxy_sensitivity_fp_burden",
    "true_froc_calculated": False,
    "lesion_level_sensitivity_calculated": False,
    "lesion_unique_id_available": False,
    "lesion_gt_count_available": False,
    "patient_id_used_as_scan_id": True,
    "threshold_recommendation_made": False,
    "cutoff_selected": False,
    "model_forward_executed": False,
    "training_executed": False,
    "checkpoint_created": False,
    "score_csv_modified": False,
    "stage2_holdout_crop_npz_reloaded": False,
    "nms_executed": False,
    "primary_score_column": SCORE_COLUMNS["primary"],
    "comparison_score_columns": SCORE_COLUMNS["comparison"],
    "proxy_analysis_executed": True,
    "score_csv_rows_confirmed": int(len(df)),
    "n_scans": int(n_patients),
    "n_positive_scans": int(n_positive_scans),
    "positive_crop_count": int(n_positive),
    "hard_negative_crop_count": int(n_hn),
    "threshold_grid_n": THRESHOLD_GRID_N,
    "proxy_sensitivity_at_fixed_fp_burden": sensitivity_at_fp,
    "proxy_hit_definition": (
        "scan_level: positive crop (sampling_label==positive) max-score >= threshold. "
        "lesion_patch_ratio not available in score CSV; "
        "confirmed equal to positive label in Phase 8.6A (51335 == 51335)."
    ),
    "fp_burden_proxy_definition": (
        "hard_negative crop count with score >= threshold per scan. "
        "NOT lesion/object-level FP count. "
        "Crop duplication may cause overestimation."
    ),
    "forbidden_operations_confirmed_not_executed": [
        "true_FROC_calculation",
        "lesion_level_sensitivity_calculation",
        "threshold_cutoff_recommendation",
        "score_csv_modification",
        "existing_output_modification_deletion",
        "model_forward",
        "training_backward_optimizer_step",
        "checkpoint_creation",
        "stage2_holdout_crop_npz_reload",
        "v2_v2v2_access",
        "adjusted_score_generation",
        "candidate_suppression_application",
        "NMS_execution",
    ],
    "output_files": [
        "scan_level_proxy_summary.csv",
        "threshold_proxy_curve.csv",
        "score_column_proxy_comparison.csv",
        "phase8_6c_per_scan_proxy_analysis_summary.json",
        "phase8_6c_per_scan_proxy_analysis_report.md",
    ],
    "final_status": "PASS_PROXY_ONLY",
}

out_json = OUTPUT_ROOT / "phase8_6c_per_scan_proxy_analysis_summary.json"
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"    저장: {out_json.name}")

# ── 5. report MD ───────────────────────────────────────────────────────────────
sep_info = "\n".join(
    f"- {r['score_column']}: separation={r['separation_summary']:.4f}"
    for _, r in comp_df.iterrows()
)
fp_sens_lines = []
for col, vals in sensitivity_at_fp.items():
    line = f"**{col}**  "
    for k, v in vals.items():
        line += f"  {k}={v}"
    fp_sens_lines.append(line)

report = f"""# Phase 8.6C: Per-Scan Proxy Sensitivity / FP-Burden Analysis Report

**생성 시각:** {summary['timestamp']}
**분석 유형:** per-scan proxy sensitivity / FP-burden analysis
**판정:** `PASS_PROXY_ONLY`
**true FROC 계산 여부:** False

---

## 1. 이 분석이 True FROC가 아닌 이유

- `lesion_unique_id` 컬럼 없음 → lesion-level NMS 불가, 병변 단위 hit 정의 불가
- `lesion_gt_count_per_scan` 컬럼 없음 → sensitivity 분모(병변 GT 수) 정의 불가
- 따라서 표준 FROC(sensitivity vs FP/scan by lesion) 계산 불가
- **이 결과를 "FROC"로 단독 표기하면 안 됨**

## 2. Lesion-Level Sensitivity가 아닌 이유

- 같은 병변 주변 crop 다수(환자당 positive crop 중위수 176개) → 어느 crop이 같은 병변인지 알 수 없음
- scan-level proxy hit = scan 내 positive crop 중 최고 score ≥ threshold인지 여부만 판단
- **병변을 찾았다는 의미가 아니라 스캔에 신호가 있다는 의미임**

## 3. Scan-Level Proxy Hit 정의

```
positive_candidate = sampling_label == positive
  (score CSV에 lesion_patch_ratio 없음;
   Phase 8.6A에서 positive == lesion_patch_ratio>0 51335개 일치 확인)
scan_proxy_hit(scan_id, threshold) =
    max(primary_score) over positive crops in scan >= threshold
proxy_sensitivity = #(scans with proxy hit) / #(total positive scans)
```

## 4. FP Crop Burden Proxy 정의

```
hard_negative_candidate = sampling_label == hard_negative
fp_crop_burden(scan_id, threshold) = count(hard_negative crops with score >= threshold)
```

- **lesion/object 단위 FP count가 아님**
- 같은 구조물 주변에 FP crop 다수 존재 시 과대계산 가능
- NMS 없이 raw crop count 기준

## 5. Threshold Sweep 운용 방침

- 목적: 탐색/곡선 생성 전용 — **cutoff 선택 금지**
- threshold grid: {THRESHOLD_GRID_N} quantile 포인트
- **이 sweep 결과로 stage2_holdout threshold를 선택하면 holdout 오염**

## 6. Proxy Sensitivity at Fixed FP Burden

{chr(10).join(fp_sens_lines)}

> 주의: 위 수치는 "scan 내 신호 탐지 proxy"이며 "병변 발견율"이 아님

## 7. Score Column Separation 요약

{sep_info}

> separation = (pos_mean - hn_mean) / (pos_std + hn_std). 높을수록 score가 positive/negative를 잘 구분함.

## 8. 결과 해석 제한

| 해석 | 허용 여부 |
|------|-----------|
| "이 모델은 FROC 기준 sensitivity X% 달성" | **금지** |
| "lesion을 X% 발견" | **금지** |
| "threshold T에서 FP/scan이 N개" | 조건부 허용 (crop 중복 명시) |
| "scan 단위 proxy hit이 X%" | 허용 (proxy 명시 조건) |
| "threshold 선택" | **금지** |
| "다발성 병변 환자 성능" | **금지** (GT 없음) |

## 9. 출력 파일

- `scan_level_proxy_summary.csv`: 스캔별 max/mean score (positive / hard_negative)
- `threshold_proxy_curve.csv`: threshold × score_column × proxy sensitivity / FP burden
- `score_column_proxy_comparison.csv`: 3개 score column 분리도 비교
- `phase8_6c_per_scan_proxy_analysis_summary.json`: 실행 상태 요약

## 10. 다음 단계

- **GT lesion ID / lesion count 확보 시:** true FROC 재설계 (Phase 8.6D 등)
- **현재 proxy 결과 활용:** 제한적 보조 분석으로만 사용 가능
  - "판독 보조 시스템이 스캔의 XX%에서 관련 신호를 탐지했다" 수준의 표현만 허용
  - sensitivity, FROC, FP/lesion 표현 금지
"""

out_report = OUTPUT_ROOT / "phase8_6c_per_scan_proxy_analysis_report.md"
out_report.write_text(report, encoding="utf-8")
print(f"    저장: {out_report.name}")

# ── 최종 확인 ──────────────────────────────────────────────────────────────────
print("\n[검증] 출력 파일 존재 확인:")
for fname in summary["output_files"]:
    p = OUTPUT_ROOT / fname
    exists = p.exists()
    print(f"    {'OK' if exists else 'MISSING'} {fname}")
    assert exists, f"출력 파일 없음: {fname}"

print(f"\n[완료] Phase 8.6C PASS_PROXY_ONLY")
print(f"       출력 위치: {OUTPUT_ROOT}")
print(f"       true_froc_calculated = False")
print(f"       lesion_level_sensitivity_calculated = False")
print(f"       threshold_recommendation_made = False")
print(f"       stage2_holdout_crop_npz_reloaded = False")
