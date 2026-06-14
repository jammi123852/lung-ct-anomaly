"""
analyze_stage1_dev_v1v2_vs_v2v2.py
stage1_dev 기준 v1/v2 vs v2/v2 top-k coverage 하락 원인 진단.
read-only 분석. 기존 score/evaluation/reports 미수정.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "outputs/second-stage-lesion-refiner-v1/reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 입력 경로 ──────────────────────────────────────────────
SPLIT_CSV      = REPO / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
SCR_V1V2       = REPO / "outputs/position-aware-padim-v1/evaluation/lesion_subset_v2/per_patient_screening.csv"
SCR_V2V2       = REPO / "outputs/position-aware-padim-v1/evaluation/lesion_subset_v2_model_v2/per_patient_screening.csv"
HIT_V1V2       = REPO / "outputs/position-aware-padim-v1/reports/lesion_hit_overlap_by_patient_v2.csv"
HIT_V2V2       = REPO / "outputs/position-aware-padim-v1/reports_v2_roi0_0_lesion/lesion_hit_overlap_by_patient.csv"
SCR_SUM_V1V2   = REPO / "outputs/position-aware-padim-v1/evaluation/lesion_subset_v2/screening_analysis_summary.json"
SCR_SUM_V2V2   = REPO / "outputs/position-aware-padim-v1/evaluation/lesion_subset_v2_model_v2/screening_analysis_summary.json"

# ── 데이터 로드 ────────────────────────────────────────────
split   = pd.read_csv(SPLIT_CSV)[["patient_id", "stage_split", "group"]]
scr_v1  = pd.read_csv(SCR_V1V2)
scr_v2  = pd.read_csv(SCR_V2V2)
hit_v1  = pd.read_csv(HIT_V1V2)
hit_v2  = pd.read_csv(HIT_V2V2)

# stage_split 조인 (screening CSV에는 없으므로 split CSV로 보완)
def add_split(df, split_df):
    if "stage_split" not in df.columns:
        df = df.merge(split_df[["patient_id","stage_split"]], on="patient_id", how="left")
    return df

scr_v1 = add_split(scr_v1, split)
scr_v2 = add_split(scr_v2, split)

# ── stage1_dev 필터링 ──────────────────────────────────────
def dev(df): return df[df["stage_split"] == "stage1_dev"].copy()

scr_v1_d = dev(scr_v1)
scr_v2_d = dev(scr_v2)
hit_v1_d = dev(hit_v1)
hit_v2_d = dev(hit_v2)

n_dev = len(scr_v2_d)
print(f"stage1_dev 환자 수: {n_dev}")

# ── A. stage1_dev 핵심 지표 비교 ──────────────────────────
def pct(v): return f"{v:.4f}" if not np.isnan(v) else "NaN"

def hit_rate(df, col="patient_hit"):
    if col in df.columns:
        return df[col].mean()
    return np.nan

def recall_stats(df, col):
    s = df[col].dropna()
    return s.mean(), s.median()

# screening 기준 비교
metrics_a = []
for tag, sd, hd in [("v1v2", scr_v1_d, hit_v1_d), ("v2v2", scr_v2_d, hit_v2_d)]:
    hr = (hd["patient_hit"] == True).mean() if "patient_hit" in hd.columns else (hd["patient_patch_recall"] > 0).mean()
    pr_m, pr_med = recall_stats(sd, "recall_p95")
    sr_m, sr_med = recall_stats(sd, "slice_cov_p95")
    cov_m = sd["slice_cov_p95"].dropna().mean()
    no_hit = hd[hd["patient_patch_recall"] == 0]["patient_id"].tolist()
    fp_m = sd["fp_ratio_p95"].dropna().mean()
    n_pos_m = sd["n_positive_p95"].dropna().mean()
    metrics_a.append({
        "model": tag,
        "n_stage1_dev": n_dev,
        "patient_hit_rate_p95": hr,
        "n_no_hit": len(no_hit),
        "no_hit_patients": no_hit,
        "lesion_patch_recall_mean_p95": pr_m,
        "lesion_patch_recall_median_p95": pr_med,
        "lesion_slice_recall_mean_p95": sr_m,
        "lesion_slice_recall_median_p95": sr_med,
        "patient_coverage_mean_p95": cov_m,
        "fp_ratio_mean_p95": fp_m,
        "n_positive_mean_p95": n_pos_m,
    })

# ── B. top-k coverage 하락 원인 진단 ──────────────────────
# screening summary에서 전체 topk는 있으나 stage1_dev 단위로는 없음
# → per_patient 수준에서 fp_ratio, n_positive, lesion_patch_count로 대리 진단

# per-patient 머지 (v1v2/v2v2 나란히)
merged = scr_v1_d[["patient_id","group","stage_split","lesion_patch_total","n_patch",
                    "recall_p95","slice_cov_p95","n_positive_p95","n_fp_p95","fp_ratio_p95"]].copy()
merged = merged.rename(columns={
    "recall_p95":"recall_p95_v1v2","slice_cov_p95":"slice_cov_p95_v1v2",
    "n_positive_p95":"n_pos_v1v2","n_fp_p95":"n_fp_v1v2","fp_ratio_p95":"fp_ratio_v1v2"
})
v2_cols = scr_v2_d[["patient_id","recall_p95","slice_cov_p95","n_positive_p95",
                     "n_fp_p95","fp_ratio_p95"]].rename(columns={
    "recall_p95":"recall_p95_v2v2","slice_cov_p95":"slice_cov_p95_v2v2",
    "n_positive_p95":"n_pos_v2v2","n_fp_p95":"n_fp_v2v2","fp_ratio_p95":"fp_ratio_v2v2"
})
merged = merged.merge(v2_cols, on="patient_id", how="outer")
merged["recall_diff"] = merged["recall_p95_v2v2"] - merged["recall_p95_v1v2"]
merged["fp_ratio_diff"] = merged["fp_ratio_v2v2"] - merged["fp_ratio_v1v2"]
merged["lesion_size_bin"] = pd.cut(merged["lesion_patch_total"],
    bins=[0,50,200,500,9999], labels=["tiny(≤50)","small(51-200)","medium(201-500)","large(>500)"])

# 원인 진단 집계
diag = {}

# (1) fp_ratio 변화: v2/v2에서 FP가 더 많은지
diag["fp_ratio_mean_v1v2"] = merged["fp_ratio_v1v2"].mean()
diag["fp_ratio_mean_v2v2"] = merged["fp_ratio_v2v2"].mean()
diag["fp_ratio_diff_mean"] = merged["fp_ratio_diff"].mean()

# (2) n_positive 비교: v2/v2에서 positive patch 수 자체가 적은지
diag["n_positive_mean_v1v2"] = merged["n_pos_v1v2"].mean()
diag["n_positive_mean_v2v2"] = merged["n_pos_v2v2"].mean()

# (3) 작은 lesion에서 더 심한지
diag["recall_diff_by_lesion_size"] = merged.groupby("lesion_size_bin", observed=True)["recall_diff"].mean().to_dict()

# (4) NSCLC/MSD 그룹 차이
diag["recall_v2v2_by_group"] = merged.groupby("group")["recall_p95_v2v2"].mean().to_dict()
diag["recall_v1v2_by_group"] = merged.groupby("group")["recall_p95_v1v2"].mean().to_dict()
diag["fp_ratio_v2v2_by_group"] = merged.groupby("group")["fp_ratio_v2v2"].mean().to_dict()

# (5) v2/v2에서 positive는 많은데 lesion recall이 낮은 환자 (positive patch가 FP에 집중)
hi_pos_lo_rec = merged[(merged["n_pos_v2v2"] > merged["n_pos_v2v2"].median()) &
                        (merged["recall_p95_v2v2"] < 0.3)]
diag["hi_positive_lo_recall_count_v2v2"] = len(hi_pos_lo_rec)
diag["hi_positive_lo_recall_patients_v2v2"] = hi_pos_lo_rec["patient_id"].tolist()

# ── C. low recall top10 (stage1_dev) ─────────────────────
low_v1 = scr_v1_d.nsmallest(10, "recall_p95")[["patient_id","group","lesion_patch_total","recall_p95","fp_ratio_p95"]]
low_v2 = scr_v2_d.nsmallest(10, "recall_p95")[["patient_id","group","lesion_patch_total","recall_p95","fp_ratio_p95"]]

# ── D. 2차 candidate 관점 요약 ────────────────────────────
with open(SCR_SUM_V1V2) as f: ss_v1 = json.load(f)
with open(SCR_SUM_V2V2) as f: ss_v2 = json.load(f)
topk_v1 = {m["threshold_mode"]: m for m in ss_v1["metrics"]}
topk_v2 = {m["threshold_mode"]: m for m in ss_v2["metrics"]}

candidate_analysis = {
    "topk_coverage_overall": {
        "v1v2_p95": {"top10": topk_v1["normal_val_p95"]["topk10_coverage"],
                     "top30": topk_v1["normal_val_p95"]["topk30_coverage"],
                     "top50": topk_v1["normal_val_p95"]["topk50_coverage"]},
        "v2v2_p95": {"top10": topk_v2["normal_val_p95"]["topk10_coverage"],
                     "top30": topk_v2["normal_val_p95"]["topk30_coverage"],
                     "top50": topk_v2["normal_val_p95"]["topk50_coverage"]},
    },
    "concern": "v2/v2 top-10 coverage 0.263: 10개 후보만 사용 시 4명 중 3명에서 병변 미포함 위험",
    "recommendation_stage1_dev": [
        "top-k 단독 사용 위험: top-10은 너무 좁음, top-50도 54.5%에 그침",
        "threshold 기반 후보(p95 patient hit rate 99.7%)와 top-k 병행 필요",
        "no-hit 방지 fallback: score 상위 N% 또는 top-100 보장 규칙 고려",
        "tiny lesion(≤50 patch) 환자는 별도 처리 필요 가능성 있음"
    ],
    "stage2_holdout_sealed": True,
    "note": "stage1_dev 기준 분석. stage2_holdout 개별 환자 기반 규칙 변경 금지."
}

# ── 저장 ─────────────────────────────────────────────────
out_csv = OUT_DIR / "stage1_dev_v1v2_vs_v2v2_candidate_diagnostic.csv"
merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

summary = {
    "purpose": "stage1_dev 기준 v1/v2 vs v2/v2 top-k coverage 하락 원인 진단",
    "note": "read-only 분석. 기존 score/evaluation/reports 미수정. stage2_holdout 봉인.",
    "stage1_dev_comparison": metrics_a,
    "topk_coverage_diagnosis": diag,
    "candidate_analysis": candidate_analysis,
    "low_recall_top10_v1v2": low_v1.to_dict(orient="records"),
    "low_recall_top10_v2v2": low_v2.to_dict(orient="records"),
}
out_json = OUT_DIR / "stage1_dev_v1v2_vs_v2v2_candidate_diagnostic_summary.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

print(f"[OK] {out_csv}")
print(f"[OK] {out_json}")

# ── 콘솔 요약 ────────────────────────────────────────────
print("\n=== stage1_dev 비교 ===")
for m in metrics_a:
    print(f"  [{m['model']}] hit_rate={m['patient_hit_rate_p95']:.4f}  no_hit={m['n_no_hit']}"
          f"  patch_recall={m['lesion_patch_recall_mean_p95']:.4f}"
          f"  slice_recall={m['lesion_slice_recall_mean_p95']:.4f}"
          f"  fp_ratio={m['fp_ratio_mean_p95']:.4f}"
          f"  n_pos_mean={m['n_positive_mean_p95']:.1f}")
print(f"\n=== top-k coverage (전체 308명) ===")
for k in ["top10","top30","top50"]:
    print(f"  {k}: v1/v2={topk_v1['normal_val_p95'][f'topk{k[3:]}_coverage']:.4f}  "
          f"v2/v2={topk_v2['normal_val_p95'][f'topk{k[3:]}_coverage']:.4f}")
print(f"\n=== 원인 진단 ===")
print(f"  fp_ratio 평균: v1/v2={diag['fp_ratio_mean_v1v2']:.4f}  v2/v2={diag['fp_ratio_mean_v2v2']:.4f}  diff={diag['fp_ratio_diff_mean']:+.4f}")
print(f"  n_positive 평균: v1/v2={diag['n_positive_mean_v1v2']:.1f}  v2/v2={diag['n_positive_mean_v2v2']:.1f}")
print(f"  그룹별 recall_p95:")
for g in diag["recall_v2v2_by_group"]:
    print(f"    {g}: v1/v2={diag['recall_v1v2_by_group'].get(g,'?'):.4f}  v2/v2={diag['recall_v2v2_by_group'][g]:.4f}")
print(f"  병변 크기별 recall_diff (v2v2-v1v2):")
for sz, d in diag["recall_diff_by_lesion_size"].items():
    print(f"    {sz}: {d:+.4f}")
print(f"  positive 많은데 recall 낮은 환자(v2/v2): {diag['hi_positive_lo_recall_count_v2v2']}명")
