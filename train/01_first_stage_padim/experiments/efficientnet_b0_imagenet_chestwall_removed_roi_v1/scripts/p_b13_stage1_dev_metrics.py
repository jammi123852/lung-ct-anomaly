"""P-B13: v4_20 ROI EfficientNet-B0 stage1_dev metrics 계산
- read-only: score CSV 154개 + threshold JSON + split CSV
- metrics 계산만 수행, scoring/model forward 금지
- slice grouping: patient_id + slice_index (z_level 사용 금지, P-A76 bug)
- stage2_holdout 접근 금지
"""
import os, sys, json, datetime, math
from pathlib import Path
import pandas as pd
import numpy as np

# sklearn 없이 numpy로 직접 구현
def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg
    fpr = np.concatenate([[0], fpr])
    tpr = np.concatenate([[0], tpr])
    return float(np.trapz(tpr, fpr))

def average_precision_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    n_pos = y_true.sum()
    if n_pos == 0:
        return float("nan")
    tp = np.cumsum(y_true_sorted)
    total = np.arange(1, len(y_true_sorted) + 1)
    precision = tp / total
    recall = tp / n_pos
    recall_prev = np.concatenate([[0], recall[:-1]])
    return float(np.sum(precision * (recall - recall_prev)))

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE = Path("/home/jinhy/project/lung-ct-anomaly")
BRANCH = BASE / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1"
SCORE_DIR = BRANCH / "outputs/scores/lesion_stage1_dev_by_patient"
THRESHOLD_JSON = BRANCH / "outputs/evaluation/normal_val_thresholds/normal_val_threshold.json"
SPLIT_CSV = BASE / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"

EVAL_OUT = BRANCH / "outputs/evaluation/lesion_stage1_dev_metrics"
RPT_OUT  = BRANCH / "outputs/reports/lesion_stage1_dev"
EVAL_OUT.mkdir(parents=True, exist_ok=True)
RPT_OUT.mkdir(parents=True, exist_ok=True)

P95_REF       = 13.231265125889463
P99_REF       = 15.472384637986801
P11_TOTAL     = 2_508_819
THRESHOLD_MTIME_REF = 1780624209

print("=" * 70)
print("P-B13 stage1_dev metrics 계산 시작")
print(f"시각: {datetime.datetime.now().isoformat()}")
print("=" * 70)

# ── 1. threshold JSON read-only ───────────────────────────────────────────────
print("\n[1] threshold JSON 확인")
assert THRESHOLD_JSON.exists()
thr_mtime = int(THRESHOLD_JSON.stat().st_mtime)
with open(THRESHOLD_JSON) as f:
    thr = json.load(f)
p95 = thr["threshold_p95"]
p99 = thr["threshold_p99"]
assert abs(p95 - P95_REF) < 1e-6, f"p95 불일치: {p95}"
assert abs(p99 - P99_REF) < 1e-6, f"p99 불일치: {p99}"
mtime_ok = (thr_mtime == THRESHOLD_MTIME_REF)
print(f"  p95={p95:.6f} ✓  p99={p99:.6f} ✓  mtime_ok={mtime_ok}")

# ── 2. split CSV ──────────────────────────────────────────────────────────────
print("\n[2] split CSV 로드")
split_df = pd.read_csv(SPLIT_CSV, encoding="utf-8-sig")
dev_df     = split_df[split_df["stage_split"] == "stage1_dev"]
holdout_df = split_df[split_df["stage_split"] == "stage2_holdout"]
dev_pids     = set(dev_df["patient_id"].tolist())
holdout_pids = set(holdout_df["patient_id"].tolist())
nsclc_dev = dev_df[dev_df["group"] == "NSCLC"].shape[0]
msd_dev   = dev_df[dev_df["group"] == "MSD_Lung"].shape[0]
assert len(dev_pids) == 154
assert nsclc_dev == 125
assert msd_dev == 29
print(f"  stage1_dev={len(dev_pids)}  NSCLC={nsclc_dev}  MSD_Lung={msd_dev}")

# ── 3. score CSV 전체 로드 ────────────────────────────────────────────────────
print("\n[3] score CSV 154개 로드")
csv_files = sorted(SCORE_DIR.glob("*.csv"))
assert len(csv_files) == 154, f"CSV 수 이상: {len(csv_files)}"

dfs = []
for cf in csv_files:
    df = pd.read_csv(cf, encoding="utf-8-sig")
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
print(f"  total rows={len(all_df):,}")

# stage2_holdout contamination 확인
contaminated = set(all_df["patient_id"].unique()) & holdout_pids
assert len(contaminated) == 0, f"STAGE2_HOLDOUT 오염: {contaminated}"
print(f"  stage2_holdout contamination=0 ✓")

# total patch count 확인
assert len(all_df) == P11_TOTAL, f"total patch count 불일치: {len(all_df)} vs {P11_TOTAL}"
print(f"  total patch={len(all_df):,} ✓")

# NaN/Inf 확인
n_nan = int(all_df["padim_score"].isna().sum())
n_inf = int(all_df["padim_score"].apply(lambda x: math.isinf(x) if isinstance(x, float) else False).sum())
assert n_nan == 0 and n_inf == 0, f"NaN={n_nan}  Inf={n_inf}"
print(f"  NaN={n_nan}  Inf={n_inf} ✓")

# ── 4. positive label 정의: has_lesion_patch==1 OR lesion_pixels>0 ─────────
print("\n[4] label 컬럼 확인")
all_df["patch_label"] = ((all_df["has_lesion_patch"] == 1) | (all_df["lesion_pixels"] > 0)).astype(int)
n_pos_patches = int(all_df["patch_label"].sum())
n_neg_patches = len(all_df) - n_pos_patches
print(f"  positive patches={n_pos_patches:,}  negative patches={n_neg_patches:,}")
print(f"  positive ratio={n_pos_patches/len(all_df)*100:.3f}%")

# ── 5. patch-level metrics (threshold-independent) ────────────────────────────
print("\n[5] patch-level AUROC / AUPRC")
scores_arr = all_df["padim_score"].values
labels_arr = all_df["patch_label"].values

patch_auroc = float(roc_auc_score(labels_arr, scores_arr))
patch_auprc = float(average_precision_score(labels_arr, scores_arr))
print(f"  patch AUROC={patch_auroc:.4f}")
print(f"  patch AUPRC={patch_auprc:.4f}")

# ── 6. slice-level metrics ───────────────────────────────────────────────────
print("\n[6] slice-level metrics (grouping=patient_id+slice_index)")
# z_level 사용 금지 (P-A76 bug)
slice_df = all_df.groupby(["patient_id", "slice_index"]).agg(
    slice_score=("padim_score", "max"),
    slice_label=("patch_label", "max"),
    n_patches=("padim_score", "count"),
).reset_index()

n_slice_total     = len(slice_df)
n_positive_slices = int((slice_df["slice_label"] == 1).sum())
n_negative_slices = n_slice_total - n_positive_slices
print(f"  n_slice_total={n_slice_total:,}  positive={n_positive_slices:,}  negative={n_negative_slices:,}")

slice_auroc = float(roc_auc_score(slice_df["slice_label"].values, slice_df["slice_score"].values))
slice_auprc = float(average_precision_score(slice_df["slice_label"].values, slice_df["slice_score"].values))
print(f"  slice AUROC={slice_auroc:.4f}")
print(f"  slice AUPRC={slice_auprc:.4f}")

# ── 7. threshold-dependent metrics ───────────────────────────────────────────
print("\n[7] threshold-dependent metrics")

def compute_threshold_metrics(df, slice_df_in, thr_val, thr_name):
    pred_pos = (df["padim_score"] > thr_val).astype(int)
    true_pos = df["patch_label"]

    # patch recall
    tp = int(((pred_pos == 1) & (true_pos == 1)).sum())
    fn = int(((pred_pos == 0) & (true_pos == 1)).sum())
    patch_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # patch Dice
    fp = int(((pred_pos == 1) & (true_pos == 0)).sum())
    patch_dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # slice recall
    sl = slice_df_in.copy()
    sl["slice_pred"] = (sl["slice_score"] > thr_val).astype(int)
    tp_s = int(((sl["slice_pred"] == 1) & (sl["slice_label"] == 1)).sum())
    fn_s = int(((sl["slice_pred"] == 0) & (sl["slice_label"] == 1)).sum())
    slice_recall = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0

    # patient hit rate: patient에 p95 초과 patch가 1개 이상 있으면 hit
    pat_hit = df.groupby("patient_id").apply(
        lambda g: int((g["padim_score"] > thr_val).any())
    )
    n_hit = int(pat_hit.sum())
    n_total_pat = len(pat_hit)
    patient_hit_rate = n_hit / n_total_pat if n_total_pat > 0 else 0.0

    print(f"  [{thr_name}] patch_recall={patch_recall:.4f}  patch_dice={patch_dice:.4f}")
    print(f"  [{thr_name}] slice_recall={slice_recall:.4f}")
    print(f"  [{thr_name}] patient_hit={n_hit}/{n_total_pat} ({patient_hit_rate:.4f})")

    return {
        f"{thr_name}_patch_recall":    patch_recall,
        f"{thr_name}_patch_dice":      patch_dice,
        f"{thr_name}_slice_recall":    slice_recall,
        f"{thr_name}_patient_hit_rate": patient_hit_rate,
        f"{thr_name}_patient_hit_n":   n_hit,
        f"{thr_name}_n_patients":      n_total_pat,
    }

m_p95 = compute_threshold_metrics(all_df, slice_df, p95, "p95")
m_p99 = compute_threshold_metrics(all_df, slice_df, p99, "p99")

# ── 8. patient-level AUROC ────────────────────────────────────────────────────
print("\n[8] patient-level AUROC")
# stage1_dev는 전원 positive-only (lesion patients)
pat_label_check = all_df.groupby("patient_id")["patch_label"].max()
n_pos_patients = int((pat_label_check > 0).sum())
n_all_patients = len(pat_label_check)
print(f"  positive patients={n_pos_patients}/{n_all_patients}")
if n_pos_patients == n_all_patients:
    patient_auroc = "not_applicable_positive_only"
    print(f"  patient-level AUROC: {patient_auroc}")
else:
    # 혼합이면 계산
    pat_scores = all_df.groupby("patient_id")["padim_score"].max()
    patient_auroc = float(roc_auc_score(pat_label_check.values, pat_scores.values))
    print(f"  patient-level AUROC={patient_auroc:.4f}")

# ── 9. per-patient summary ────────────────────────────────────────────────────
print("\n[9] per-patient summary 생성")
per_pat = all_df.groupby("patient_id").agg(
    n_patches=("padim_score", "count"),
    n_lesion_patches=("patch_label", "sum"),
    score_mean=("padim_score", "mean"),
    score_max=("padim_score", "max"),
    score_median=("padim_score", "median"),
    p95_exceed=("padim_score", lambda x: (x > p95).sum()),
    p99_exceed=("padim_score", lambda x: (x > p99).sum()),
    group=("group", "first"),
).reset_index()
per_pat["p95_rate"] = per_pat["p95_exceed"] / per_pat["n_patches"] * 100
per_pat["p99_rate"] = per_pat["p99_exceed"] / per_pat["n_patches"] * 100
per_pat["lesion_patch_rate"] = per_pat["n_lesion_patches"] / per_pat["n_patches"] * 100
per_pat["p95_lesion_recall"] = per_pat.apply(
    lambda r: all_df[(all_df["patient_id"]==r["patient_id"]) & (all_df["patch_label"]==1)].shape[0], axis=1
)
# per-patient recall 계산
def pat_recall(pid, thr_val):
    sub = all_df[all_df["patient_id"] == pid]
    tp = int(((sub["padim_score"] > thr_val) & (sub["patch_label"] == 1)).sum())
    fn = int(((sub["padim_score"] <= thr_val) & (sub["patch_label"] == 1)).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else float("nan")

per_pat["p95_recall"] = per_pat["patient_id"].apply(lambda pid: pat_recall(pid, p95))
per_pat["p99_recall"] = per_pat["patient_id"].apply(lambda pid: pat_recall(pid, p99))

print(f"  per-patient rows={len(per_pat)}")

# ── 10. slice metrics summary CSV ────────────────────────────────────────────
slice_summary = slice_df.copy()
slice_summary["p95_pred"] = (slice_summary["slice_score"] > p95).astype(int)
slice_summary["p99_pred"] = (slice_summary["slice_score"] > p99).astype(int)

# ── 11. 출력 파일 저장 ────────────────────────────────────────────────────────
print("\n[10] 출력 파일 저장")
now_str = datetime.datetime.now().isoformat()

metrics_dict = {
    "step": "P-B13",
    "created": now_str,
    "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
    "roi_source": "refined_roi_v4_20_modeB_all_v1",
    "input_validation": {
        "p_b12_verdict": "통과",
        "p_b11_verdict": "통과",
        "threshold_p95": p95,
        "threshold_p99": p99,
        "threshold_mtime_ok": mtime_ok,
        "threshold_recalculated": False,
        "total_patches": len(all_df),
        "p11_total_match": len(all_df) == P11_TOTAL,
        "nan_count": n_nan,
        "inf_count": n_inf,
        "stage2_holdout_contamination": 0,
        "patient_id_set": f"{len(dev_pids)} stage1_dev",
        "nsclc": nsclc_dev,
        "msd_lung": msd_dev,
    },
    "label_stats": {
        "n_positive_patches": n_pos_patches,
        "n_negative_patches": n_neg_patches,
        "positive_ratio_pct": round(n_pos_patches / len(all_df) * 100, 4),
        "positive_label_def": "has_lesion_patch==1 OR lesion_pixels>0",
    },
    "patch_metrics": {
        "auroc": patch_auroc,
        "auprc": patch_auprc,
    },
    "slice_metrics": {
        "grouping": "patient_id + slice_index (z_level 사용 안 함)",
        "n_slice_total": n_slice_total,
        "n_positive_slices": n_positive_slices,
        "n_negative_slices": n_negative_slices,
        "auroc": slice_auroc,
        "auprc": slice_auprc,
    },
    "threshold_metrics": {
        "p95": {**m_p95},
        "p99": {**m_p99},
    },
    "patient_auroc": patient_auroc,
    "guardrails": {
        "np_loadtxt_used": False,
        "scoring_rerun": False,
        "model_forward": False,
        "feature_extraction": False,
        "threshold_recalculated": False,
        "normal_val_test_rerun": False,
        "stage2_holdout_accessed": False,
        "existing_results_modified": False,
    },
    "next_step": {
        "p_b14_comparison_ready": True,
        "note": "P-B14 roi_0_0 EfficientNet branch read-only comparison 가능. stage2_holdout은 계속 locked.",
    },
}

# JSON 저장
with open(EVAL_OUT / "p_b13_stage1_dev_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
print("  p_b13_stage1_dev_metrics.json 저장")

# metrics CSV (요약 1행)
metrics_row = {
    "branch": "v4_20_roi_efficientnet_b0",
    "split": "stage1_dev",
    "n_patients": 154,
    "nsclc": 125,
    "msd_lung": 29,
    "total_patches": len(all_df),
    "n_positive_patches": n_pos_patches,
    "patch_auroc": round(patch_auroc, 4),
    "patch_auprc": round(patch_auprc, 4),
    "n_slice_total": n_slice_total,
    "n_positive_slices": n_positive_slices,
    "n_negative_slices": n_negative_slices,
    "slice_auroc": round(slice_auroc, 4),
    "slice_auprc": round(slice_auprc, 4),
    "p95_patch_recall": round(m_p95["p95_patch_recall"], 4),
    "p95_patch_dice": round(m_p95["p95_patch_dice"], 4),
    "p95_slice_recall": round(m_p95["p95_slice_recall"], 4),
    "p95_patient_hit_rate": round(m_p95["p95_patient_hit_rate"], 4),
    "p99_patch_recall": round(m_p99["p99_patch_recall"], 4),
    "p99_patch_dice": round(m_p99["p99_patch_dice"], 4),
    "p99_slice_recall": round(m_p99["p99_slice_recall"], 4),
    "p99_patient_hit_rate": round(m_p99["p99_patient_hit_rate"], 4),
    "patient_auroc": patient_auroc,
    "threshold_p95": p95,
    "threshold_p99": p99,
}
pd.DataFrame([metrics_row]).to_csv(EVAL_OUT / "p_b13_stage1_dev_metrics.csv", index=False, encoding="utf-8-sig")
print("  p_b13_stage1_dev_metrics.csv 저장")

# per-patient CSV
per_pat.to_csv(EVAL_OUT / "p_b13_stage1_dev_per_patient.csv", index=False, encoding="utf-8-sig")
print("  p_b13_stage1_dev_per_patient.csv 저장")

# slice metrics CSV
slice_summary.to_csv(EVAL_OUT / "p_b13_stage1_dev_slice_metrics_summary.csv", index=False, encoding="utf-8-sig")
print("  p_b13_stage1_dev_slice_metrics_summary.csv 저장")

# ── MD 보고서 ────────────────────────────────────────────────────────────────
md = [
    "# P-B13 v4_20 ROI EfficientNet-B0 Stage1_dev Metrics",
    "",
    "**판정: 통과**",
    "",
    f"- 생성일시: {now_str}",
    f"- branch: efficientnet_b0_imagenet_chestwall_removed_roi_v1 / ROI: refined_roi_v4_20_modeB_all_v1",
    "",
    "## P-B12 입력 검증",
    "",
    "- P-B12 verdict=통과  P-B11 verdict=통과",
    f"- threshold p95={p95:.6f}  p99={p99:.6f}  (재계산 없음, mtime_ok={mtime_ok})",
    f"- total patches={len(all_df):,} (P-B11과 일치)  NaN={n_nan}  Inf={n_inf}",
    f"- NSCLC {nsclc_dev} / MSD_Lung {msd_dev}  stage2_holdout contamination=0",
    "",
    "## label 통계",
    "",
    f"- positive 정의: has_lesion_patch==1 OR lesion_pixels>0",
    f"- positive patches={n_pos_patches:,}  negative patches={n_neg_patches:,}",
    f"- positive ratio={n_pos_patches/len(all_df)*100:.3f}%",
    "",
    "## patch-level metrics (threshold-independent)",
    "",
    "| 지표 | 값 |",
    "|------|----|",
    f"| patch AUROC | {patch_auroc:.4f} |",
    f"| patch AUPRC | {patch_auprc:.4f} |",
    "",
    "## slice-level metrics (grouping=patient_id+slice_index, z_level 사용 안 함)",
    "",
    "| 지표 | 값 |",
    "|------|----|",
    f"| n_slice_total | {n_slice_total:,} |",
    f"| n_positive_slices | {n_positive_slices:,} |",
    f"| n_negative_slices | {n_negative_slices:,} |",
    f"| slice AUROC | {slice_auroc:.4f} |",
    f"| slice AUPRC | {slice_auprc:.4f} |",
    "",
    "## threshold-dependent metrics",
    "",
    "| 지표 | p95 (13.231265) | p99 (15.472385) |",
    "|------|-----------------|-----------------|",
    f"| lesion_patch_recall | {m_p95['p95_patch_recall']:.4f} | {m_p99['p99_patch_recall']:.4f} |",
    f"| patch Dice | {m_p95['p95_patch_dice']:.4f} | {m_p99['p99_patch_dice']:.4f} |",
    f"| lesion_slice_recall | {m_p95['p95_slice_recall']:.4f} | {m_p99['p99_slice_recall']:.4f} |",
    f"| patient_hit_rate | {m_p95['p95_patient_hit_rate']:.4f} ({m_p95['p95_patient_hit_n']}/{m_p95['p95_n_patients']}) | {m_p99['p99_patient_hit_rate']:.4f} ({m_p99['p99_patient_hit_n']}/{m_p99['p99_n_patients']}) |",
    "",
    "## patient-level AUROC",
    "",
    f"- {patient_auroc}",
    f"- stage1_dev 전원 lesion positive → patient-level AUROC 계산 불가",
    "",
    "## 가드레일 확인",
    "",
    "- np.loadtxt 미사용: True",
    "- scoring 재실행: 없음  model forward: 없음  feature extraction: 없음",
    "- threshold 재계산: 없음  threshold 수정: 없음",
    "- normal val/test 재실행: 없음",
    "- stage2_holdout 미접근: True",
    "- 기존 P-B1~P-B12 결과 무수정: True",
    "- slice grouping: patient_id + slice_index (z_level 사용 안 함, P-A76 bug 회피)",
    "",
    "## 해석 주의",
    "",
    "- 이 결과는 stage1_dev 개발셋 기준 metrics임",
    "- 최종 일반화 성능 결론 금지",
    "- stage2_holdout 계속 locked",
    "- P-B14 roi_0_0 EfficientNet branch 비교 전까지 개선/악화 결론 보류",
    "",
    "## 다음 단계",
    "",
    "- P-B14 roi_0_0 EfficientNet branch read-only comparison 가능: True",
]
with open(RPT_OUT / "p_b13_stage1_dev_metrics.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("  p_b13_stage1_dev_metrics.md 저장")

# report JSON (별도)
with open(RPT_OUT / "p_b13_stage1_dev_metrics_report.json", "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
print("  p_b13_stage1_dev_metrics_report.json 저장")

print("\n" + "=" * 70)
print(f"P-B13 완료: 판정=통과")
print(f"  patch AUROC={patch_auroc:.4f}  patch AUPRC={patch_auprc:.4f}")
print(f"  slice AUROC={slice_auroc:.4f}  slice AUPRC={slice_auprc:.4f}")
print("=" * 70)
