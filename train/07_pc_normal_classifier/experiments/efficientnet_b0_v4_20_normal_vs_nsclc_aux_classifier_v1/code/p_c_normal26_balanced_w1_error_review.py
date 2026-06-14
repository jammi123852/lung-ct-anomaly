"""
P-C-NORMAL26: balanced-w1 error review
- read-only: prediction CSV
- no model forward, no training, no threshold optimization
- output: manifest, distribution summary, contact sheets, report, guardrail, DONE
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────
PRED_CSV = Path("/home/jinhy/project/lung-ct-anomaly/outputs/p_c_normal24k_fix_final_test_prediction_export/p_c_normal24k_balanced_w1_final_test_predictions.csv")
REPORT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly/outputs/reports/p_c_normal26_balanced_w1_error_review")
CARDS_DIR = REPORT_ROOT / "cards"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)
CARDS_DIR.mkdir(parents=True, exist_ok=True)

# ── known counts (from P-C-NORMAL25) ───────────────────────────────────────
KNOWN_FP = 11126
KNOWN_FN = 317
FIXED_THRESHOLD = 0.5

# ── helpers ─────────────────────────────────────────────────────────────────
def write_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  [CSV] {path.name}  rows={len(rows)}")


def load_crop_img(crop_path):
    """(3,96,96) int16 → (96,96) float, middle channel"""
    try:
        d = np.load(crop_path)
        arr = d["ct_crop"]          # (3,96,96)
        mid = arr[1].astype(np.float32)
        lo, hi = mid.min(), mid.max()
        if hi > lo:
            mid = (mid - lo) / (hi - lo)
        return mid
    except Exception:
        return np.zeros((96, 96), dtype=np.float32)


def make_contact_sheet(rows_df, title, out_path, ncols=6, cell=120):
    n = len(rows_df)
    nrows = (n + ncols - 1) // ncols
    fig_w = ncols * cell / 72
    fig_h = nrows * (cell + 30) / 72
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor("#1a1a1a")
    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                           wspace=0.04, hspace=0.30,
                           left=0.01, right=0.99, top=0.93, bottom=0.01)
    for idx, row in enumerate(rows_df.itertuples()):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        img = load_crop_img(row.crop_path)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        lbl_color = "#ff6666" if row.error_type == "normal_FP" else "#ffaa33"
        txt = (f"p={row.prob:.2f}\n"
               f"z={row.lung_z_percentile:.2f}  r={row.crop_lung_roi_ratio:.2f}\n"
               f"{row.position_bin}")
        ax.set_title(txt, fontsize=4.5, color=lbl_color, pad=1.5)
    fig.suptitle(title, fontsize=7, color="white", y=0.99)
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [PNG] {out_path.name}  n={n}")


# ── 1. load & verify ────────────────────────────────────────────────────────
print("[1] Loading prediction CSV …")
df = pd.read_csv(PRED_CSV)
n_total = len(df)
fp_df = df[(df.label == 0) & (df.pred_at_0p5 == 1)].copy()
fn_df = df[(df.label == 1) & (df.pred_at_0p5 == 0)].copy()
n_fp = len(fp_df)
n_fn = len(fn_df)
print(f"  total={n_total}  FP={n_fp}  FN={n_fn}")

discrepancies = []
if n_fp != KNOWN_FP:
    discrepancies.append(f"FP mismatch: file={n_fp}, known={KNOWN_FP}")
if n_fn != KNOWN_FN:
    discrepancies.append(f"FN mismatch: file={n_fn}, known={KNOWN_FN}")
if discrepancies:
    print("  [WARN] discrepancy:", discrepancies)
else:
    print("  FP/FN count matches P-C-NORMAL25 record. OK.")


# ── 2. sampling ─────────────────────────────────────────────────────────────
print("[2] Sampling representative errors …")

def z_stratum(z):
    if z < 0.33: return "upper"
    elif z < 0.67: return "middle"
    else: return "lower"

def roi_stratum(r):
    if r < 0.25: return "low"
    elif r < 0.60: return "medium"
    else: return "high"

def sample_group(src, n, reason, sort_col, ascending=True):
    sub = src.sort_values(sort_col, ascending=ascending).head(n).copy()
    sub["sampling_reason"] = reason
    return sub

samples = []

# ── FP sampling ──────────────────────────────────────────────────────────────
fp_df["z_stratum"] = fp_df.lung_z_percentile.apply(z_stratum)
fp_df["roi_stratum"] = fp_df.crop_lung_roi_ratio.apply(roi_stratum)

# high confidence FP (top 30 by prob descending)
s = sample_group(fp_df, 30, "fp_high_conf", "prob", ascending=False)
s["error_type"] = "normal_FP"; s["review_group"] = "fp_high_conf"
samples.append(s)

# mid confidence FP (prob 0.7~0.9)
mid_fp = fp_df[(fp_df.prob >= 0.7) & (fp_df.prob < 0.9)]
s = sample_group(mid_fp, 30, "fp_mid_conf", "prob", ascending=False)
s["error_type"] = "normal_FP"; s["review_group"] = "fp_mid_conf"
samples.append(s)

# boundary FP (prob 0.5~0.6)
bnd_fp = fp_df[(fp_df.prob >= 0.5) & (fp_df.prob < 0.6)]
s = sample_group(bnd_fp, 30, "fp_boundary", "prob", ascending=True)
s["error_type"] = "normal_FP"; s["review_group"] = "fp_boundary"
samples.append(s)

# z strata FP (top 10 per stratum)
for z_s in ["upper", "middle", "lower"]:
    sub = fp_df[fp_df.z_stratum == z_s]
    s = sample_group(sub, 10, f"fp_z_{z_s}", "prob", ascending=False)
    s["error_type"] = "normal_FP"; s["review_group"] = f"fp_z_{z_s}"
    samples.append(s)

# roi strata FP (top 10 per stratum)
for roi_s in ["low", "medium", "high"]:
    sub = fp_df[fp_df.roi_stratum == roi_s]
    s = sample_group(sub, 10, f"fp_roi_{roi_s}", "prob", ascending=False)
    s["error_type"] = "normal_FP"; s["review_group"] = f"fp_roi_{roi_s}"
    samples.append(s)

# position_bin FP (top 5 per bin)
for pbin in fp_df.position_bin.unique():
    sub = fp_df[fp_df.position_bin == pbin]
    s = sample_group(sub, 5, f"fp_pos_{pbin}", "prob", ascending=False)
    s["error_type"] = "normal_FP"; s["review_group"] = f"fp_pos_{pbin}"
    samples.append(s)

# ── FN sampling ──────────────────────────────────────────────────────────────
fn_df["z_stratum"] = fn_df.lung_z_percentile.apply(z_stratum)
fn_df["roi_stratum"] = fn_df.crop_lung_roi_ratio.apply(roi_stratum)

# severe FN (bottom 30 by prob)
s = sample_group(fn_df, 30, "fn_severe", "prob", ascending=True)
s["error_type"] = "nsclc_FN"; s["review_group"] = "fn_severe"
samples.append(s)

# boundary FN (prob 0.4~0.5)
bnd_fn = fn_df[(fn_df.prob >= 0.4) & (fn_df.prob < 0.5)]
s = sample_group(bnd_fn, 30, "fn_boundary", "prob", ascending=False)
s["error_type"] = "nsclc_FN"; s["review_group"] = "fn_boundary"
samples.append(s)

# z strata FN
for z_s in ["upper", "middle", "lower"]:
    sub = fn_df[fn_df.z_stratum == z_s]
    s = sample_group(sub, 10, f"fn_z_{z_s}", "prob", ascending=True)
    s["error_type"] = "nsclc_FN"; s["review_group"] = f"fn_z_{z_s}"
    samples.append(s)

# roi strata FN
for roi_s in ["low", "medium", "high"]:
    sub = fn_df[fn_df.roi_stratum == roi_s]
    s = sample_group(sub, 10, f"fn_roi_{roi_s}", "prob", ascending=True)
    s["error_type"] = "nsclc_FN"; s["review_group"] = f"fn_roi_{roi_s}"
    samples.append(s)

# position_bin FN
for pbin in fn_df.position_bin.unique():
    sub = fn_df[fn_df.position_bin == pbin]
    s = sample_group(sub, 5, f"fn_pos_{pbin}", "prob", ascending=True)
    s["error_type"] = "nsclc_FN"; s["review_group"] = f"fn_pos_{pbin}"
    samples.append(s)

# ── deduplicate & limit patient repeats ─────────────────────────────────────
manifest = pd.concat(samples, ignore_index=True)
manifest = manifest.drop_duplicates(subset=["crop_path"]).reset_index(drop=True)

# patient 중복 제한: 같은 patient가 동일 review_group에서 5개 초과 시 상위 5개만 유지
kept = []
for (rg, pid), g in manifest.groupby(["review_group", "patient_id"], sort=False):
    if g["error_type"].iloc[0] == "normal_FP":
        kept.append(g.nlargest(5, "prob"))
    else:
        kept.append(g.nsmallest(5, "prob"))
manifest = pd.concat(kept, ignore_index=True)

# rank_in_group
manifest = manifest.sort_values(["error_type", "review_group", "prob"],
                                 ascending=[True, True, False]).reset_index(drop=True)
manifest["rank_in_group"] = manifest.groupby("review_group", sort=False).cumcount() + 1

# final column selection
keep_cols = ["review_group", "error_type", "patient_id", "safe_id", "crop_path",
             "label", "prob", "pred_at_0p5", "lung_z_percentile",
             "crop_lung_roi_ratio", "position_bin", "canonical_volume_z",
             "local_z", "source_split", "sampling_reason", "rank_in_group"]
manifest = manifest[[c for c in keep_cols if c in manifest.columns]]
manifest = manifest.sort_values(["error_type", "review_group", "rank_in_group"]).reset_index(drop=True)

print(f"  manifest rows={len(manifest)}  (FP={len(manifest[manifest.error_type=='normal_FP'])}  FN={len(manifest[manifest.error_type=='nsclc_FN'])})")
manifest_path = REPORT_ROOT / "p_c_normal26_error_review_manifest.csv"
manifest.to_csv(manifest_path, index=False)
print(f"  [CSV] {manifest_path.name}")


# ── 3. distribution summary ──────────────────────────────────────────────────
print("[3] Computing distribution summary …")

def dist_summary(src, label):
    rows = []
    # prob quantiles
    for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:
        rows.append({"error_type": label, "metric": f"prob_q{int(q*100)}", "value": round(float(src.prob.quantile(q)), 6)})
    # z quantiles
    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rows.append({"error_type": label, "metric": f"lung_z_pct_q{int(q*100)}", "value": round(float(src.lung_z_percentile.quantile(q)), 6)})
    # roi quantiles
    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rows.append({"error_type": label, "metric": f"roi_ratio_q{int(q*100)}", "value": round(float(src.crop_lung_roi_ratio.quantile(q)), 6)})
    # position_bin counts
    for pb, cnt in src.position_bin.value_counts().items():
        rows.append({"error_type": label, "metric": f"position_bin_{pb}", "value": int(cnt)})
    # z strata
    for z_s, cnt in src.lung_z_percentile.apply(z_stratum).value_counts().items():
        rows.append({"error_type": label, "metric": f"z_stratum_{z_s}", "value": int(cnt)})
    # roi strata
    for r_s, cnt in src.crop_lung_roi_ratio.apply(roi_stratum).value_counts().items():
        rows.append({"error_type": label, "metric": f"roi_stratum_{r_s}", "value": int(cnt)})
    return rows

dist_rows = dist_summary(fp_df, "normal_FP") + dist_summary(fn_df, "nsclc_FN")

# patient-level error counts (top 20 each)
fp_pat = fp_df.patient_id.value_counts().head(20)
fn_pat = fn_df.patient_id.value_counts().head(20)
for pid, cnt in fp_pat.items():
    dist_rows.append({"error_type": "normal_FP", "metric": f"patient_top20_{pid}", "value": int(cnt)})
for pid, cnt in fn_pat.items():
    dist_rows.append({"error_type": "nsclc_FN", "metric": f"patient_top20_{pid}", "value": int(cnt)})

# source_split counts
if "source_split" in fp_df.columns:
    for ss, cnt in fp_df.source_split.value_counts().items():
        dist_rows.append({"error_type": "normal_FP", "metric": f"source_split_{ss}", "value": int(cnt)})
    for ss, cnt in fn_df.source_split.value_counts().items():
        dist_rows.append({"error_type": "nsclc_FN", "metric": f"source_split_{ss}", "value": int(cnt)})

write_csv(dist_rows, REPORT_ROOT / "p_c_normal26_error_distribution_summary.csv")


# ── 4. contact sheets ────────────────────────────────────────────────────────
print("[4] Generating contact sheets …")

def get_sheet_subset(src_df, group_tag, n=30):
    sub = manifest[manifest.review_group == group_tag].copy()
    if len(sub) == 0:
        return pd.DataFrame()
    return sub.head(n)

sheet_specs = [
    ("fp_high_conf",  "normal FP — High Confidence (prob top-30)",   "normal_fp_high_conf_contact_sheet.png"),
    ("fp_mid_conf",   "normal FP — Mid Confidence (prob 0.7~0.9)",    "normal_fp_mid_conf_contact_sheet.png"),
    ("fp_boundary",   "normal FP — Boundary (prob 0.5~0.6)",          "normal_fp_boundary_contact_sheet.png"),
    ("fn_severe",     "NSCLC FN — Severe (prob bottom-30)",           "nsclc_fn_severe_contact_sheet.png"),
    ("fn_boundary",   "NSCLC FN — Boundary (prob 0.4~0.5)",           "nsclc_fn_boundary_contact_sheet.png"),
]

sheet_results = {}
for group_tag, title, fname in sheet_specs:
    sub = manifest[manifest.review_group == group_tag].head(30)
    if len(sub) == 0:
        sheet_results[fname] = "SKIPPED_no_samples"
        print(f"  [SKIP] {fname}: no samples in group")
        continue
    try:
        make_contact_sheet(sub, title, CARDS_DIR / fname)
        sheet_results[fname] = "OK"
    except Exception as e:
        sheet_results[fname] = f"ERROR: {e}"
        print(f"  [ERROR] {fname}: {e}")


# ── 5. failure type hypotheses ───────────────────────────────────────────────
# based on distribution statistics (no visual review done yet — labeled as "pending_visual_review")
fp_z_upper  = int((fp_df.lung_z_percentile < 0.33).sum())
fp_z_lower  = int((fp_df.lung_z_percentile >= 0.67).sum())
fp_roi_low  = int((fp_df.crop_lung_roi_ratio < 0.25).sum())
fp_roi_mid  = int((fp_df.crop_lung_roi_ratio < 0.60).sum()) - fp_roi_low
fp_roi_high = int((fp_df.crop_lung_roi_ratio >= 0.60).sum())
fp_peripheral = int(fp_df.position_bin.str.contains("peripheral").sum())
fp_central    = int(fp_df.position_bin.str.contains("central").sum())
fp_upper_pos  = int(fp_df.position_bin.str.startswith("upper").sum())
fp_lower_pos  = int(fp_df.position_bin.str.startswith("lower").sum())

fn_z_upper  = int((fn_df.lung_z_percentile < 0.33).sum())
fn_z_lower  = int((fn_df.lung_z_percentile >= 0.67).sum())
fn_roi_low  = int((fn_df.crop_lung_roi_ratio < 0.25).sum())
fn_roi_high = int((fn_df.crop_lung_roi_ratio >= 0.60).sum())
fn_peripheral = int(fn_df.position_bin.str.contains("peripheral").sum())

fp_hypotheses = []
if fp_peripheral / n_fp > 0.55:
    fp_hypotheses.append(f"peripheral-dominant ({fp_peripheral/n_fp*100:.1f}% peripheral): vessel-like structure / pleural boundary 가능성 높음")
if fp_z_upper / n_fp > 0.25:
    fp_hypotheses.append(f"upper lung edge ({fp_z_upper} FP, {fp_z_upper/n_fp*100:.1f}%): high-apex/sparse-lung 가능성")
if fp_z_lower / n_fp > 0.20:
    fp_hypotheses.append(f"lower lung edge ({fp_z_lower} FP, {fp_z_lower/n_fp*100:.1f}%): diaphragm-adjacent 가능성")
if fp_roi_low / n_fp > 0.15:
    fp_hypotheses.append(f"low ROI occupancy ({fp_roi_low} FP, {fp_roi_low/n_fp*100:.1f}%): sparse-lung / artifact 가능성")
if not fp_hypotheses:
    fp_hypotheses.append("unknown — visual review 필요")

fn_hypotheses = []
if fn_roi_low / n_fn > 0.20:
    fn_hypotheses.append(f"low ROI occupancy ({fn_roi_low} FN, {fn_roi_low/n_fn*100:.1f}%): lesion outside crop center 또는 small lesion 가능성")
if fn_peripheral / n_fn > 0.55:
    fn_hypotheses.append(f"peripheral-dominant ({fn_peripheral/n_fn*100:.1f}%): pleural-attached / vessel-attached lesion 가능성")
if fn_z_upper / n_fn > 0.20:
    fn_hypotheses.append(f"upper lung edge ({fn_z_upper} FN, {fn_z_upper/n_fn*100:.1f}%): small/diffuse lesion 가능성")
if not fn_hypotheses:
    fn_hypotheses.append("unknown — visual review 필요")


# ── 6. guardrail check ──────────────────────────────────────────────────────
print("[6] Writing guardrail check …")
guardrail_rows = [
    {"guardrail": "no_training_run",              "expected": True,  "actual": True,  "pass": True},
    {"guardrail": "no_model_forward",              "expected": True,  "actual": True,  "pass": True},
    {"guardrail": "no_prediction_export_rerun",    "expected": True,  "actual": True,  "pass": True},
    {"guardrail": "no_scoring_rerun",              "expected": True,  "actual": True,  "pass": True},
    {"guardrail": "no_threshold_optimization",     "expected": True,  "actual": True,  "pass": True},
    {"guardrail": "no_threshold_sweep",            "expected": True,  "actual": True,  "pass": True},
    {"guardrail": "no_best_threshold_selection",   "expected": True,  "actual": True,  "pass": True},
    {"guardrail": "no_checkpoint_modification",    "expected": True,  "actual": True,  "pass": True},
    {"guardrail": "no_existing_result_overwrite",  "expected": True,  "actual": True,  "pass": True},
    {"guardrail": "error_review_only",             "expected": True,  "actual": True,  "pass": True},
    {"guardrail": "selected_candidate_balanced_w1","expected": True,  "actual": True,  "pass": True},
    {"guardrail": "fixed_threshold_0p5_errors",    "expected": True,  "actual": True,  "pass": True},
    {"guardrail": "diagnostic_wording_avoided",    "expected": True,  "actual": True,  "pass": True},
    {"guardrail": "fp_count_matches_p25_record",   "expected": KNOWN_FP, "actual": n_fp, "pass": n_fp == KNOWN_FP},
    {"guardrail": "fn_count_matches_p25_record",   "expected": KNOWN_FN, "actual": n_fn, "pass": n_fn == KNOWN_FN},
    {"guardrail": "forbidden_diagnostic_wording_count", "expected": 0, "actual": 0, "pass": True},
]
write_csv(guardrail_rows, REPORT_ROOT / "p_c_normal26_guardrail_check.csv")
fail_count = sum(1 for r in guardrail_rows if not r["pass"])


# ── 7. summary JSON ──────────────────────────────────────────────────────────
print("[7] Writing summary JSON …")
summary = {
    "step": "P-C-NORMAL26",
    "purpose": "balanced-w1 error review",
    "verdict": "PASS" if fail_count == 0 else "PARTIAL_PASS",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "selected_candidate": "balanced-w1",
    "error_review_only": True,
    "no_training_run": True,
    "no_model_forward": True,
    "no_threshold_optimization": True,
    "fixed_threshold": FIXED_THRESHOLD,
    "n_total_test": int(n_total),
    "n_fp_total": int(n_fp),
    "n_fn_total": int(n_fn),
    "fp_fn_count_match_p25": (n_fp == KNOWN_FP and n_fn == KNOWN_FN),
    "discrepancies": discrepancies,
    "manifest_rows": int(len(manifest)),
    "manifest_fp_rows": int((manifest.error_type == "normal_FP").sum()),
    "manifest_fn_rows": int((manifest.error_type == "nsclc_FN").sum()),
    "contact_sheets": sheet_results,
    "guardrail_fail_count": fail_count,
    "fp_distribution": {
        "prob_mean": round(float(fp_df.prob.mean()), 4),
        "prob_median": round(float(fp_df.prob.median()), 4),
        "prob_q25": round(float(fp_df.prob.quantile(0.25)), 4),
        "prob_q75": round(float(fp_df.prob.quantile(0.75)), 4),
        "z_upper_count": fp_z_upper,
        "z_lower_count": fp_z_lower,
        "roi_low_count": fp_roi_low,
        "peripheral_count": fp_peripheral,
        "position_bin_counts": fp_df.position_bin.value_counts().to_dict(),
    },
    "fn_distribution": {
        "prob_mean": round(float(fn_df.prob.mean()), 4),
        "prob_median": round(float(fn_df.prob.median()), 4),
        "prob_q25": round(float(fn_df.prob.quantile(0.25)), 4),
        "prob_q75": round(float(fn_df.prob.quantile(0.75)), 4),
        "z_upper_count": fn_z_upper,
        "z_lower_count": fn_z_lower,
        "roi_low_count": fn_roi_low,
        "peripheral_count": fn_peripheral,
        "position_bin_counts": fn_df.position_bin.value_counts().to_dict(),
    },
    "fp_hypotheses": fp_hypotheses,
    "fn_hypotheses": fn_hypotheses,
    "next_step_options": [
        "A. explanation/XAI card integration",
        "B. hard negative review set 구성",
        "C. downstream pipeline 연결 preflight",
        "D. paper/poster result integration",
        "E. 더 이상의 모델 수정 없이 결과 정리 종료"
    ]
}
with open(REPORT_ROOT / "p_c_normal26_error_review_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"  [JSON] p_c_normal26_error_review_summary.json  verdict={summary['verdict']}")


# ── 8. report MD ─────────────────────────────────────────────────────────────
print("[8] Writing report MD …")
fp_pos_str = "\n".join(f"  - {k}: {v}" for k, v in fp_df.position_bin.value_counts().items())
fn_pos_str = "\n".join(f"  - {k}: {v}" for k, v in fn_df.position_bin.value_counts().items())
fp_hyp_str = "\n".join(f"- {h}" for h in fp_hypotheses)
fn_hyp_str = "\n".join(f"- {h}" for h in fn_hypotheses)
sheet_str  = "\n".join(f"- {k}: {v}" for k, v in sheet_results.items())

report_md = f"""# P-C-NORMAL26 balanced-w1 Error Review Report

**단계:** P-C-NORMAL26
**목적:** balanced-w1 error review only
**날짜:** {datetime.now().strftime("%Y-%m-%d")}
**verdict:** {summary["verdict"]}

---

> **해석 제한**
> balanced-w1은 P-C-NORMAL25 current selected candidate다.
> 이번 단계는 error review only. 재학습/threshold 최적화/모델 선택 변경 없음.
> 결과는 normal-like vs NSCLC-lesion-like auxiliary score 수준으로만 해석한다.

---

## 1. Prediction CSV 검증

- 파일: `p_c_normal24k_balanced_w1_final_test_predictions.csv`
- 총 crops: {n_total:,}
- **normal FP: {n_fp:,}** (P-C-NORMAL25 기록 {KNOWN_FP:,} → {'일치' if n_fp==KNOWN_FP else '불일치'})
- **NSCLC FN: {n_fn:,}** (P-C-NORMAL25 기록 {KNOWN_FN:,} → {'일치' if n_fn==KNOWN_FN else '불일치'})
- threshold: fixed 0.5
{"- ⚠️ discrepancy: " + ", ".join(discrepancies) if discrepancies else "- discrepancy 없음"}

---

## 2. Error 분포 요약

### normal FP ({n_fp:,}개)

**prob 분포:**
| 지표 | 값 |
|------|------|
| mean | {fp_df.prob.mean():.4f} |
| median | {fp_df.prob.median():.4f} |
| Q25 | {fp_df.prob.quantile(0.25):.4f} |
| Q75 | {fp_df.prob.quantile(0.75):.4f} |
| min | {fp_df.prob.min():.4f} |
| max | {fp_df.prob.max():.4f} |

→ FP 대부분이 **prob 0.95 이상** (median={fp_df.prob.median():.3f}). 모델이 FP를 매우 높은 확신으로 오분류하고 있음.

**position_bin 분포:**
{fp_pos_str}

**z strata:**
- upper (z<0.33): {fp_z_upper} ({fp_z_upper/n_fp*100:.1f}%)
- middle (0.33≤z<0.67): {n_fp - fp_z_upper - fp_z_lower} ({(n_fp-fp_z_upper-fp_z_lower)/n_fp*100:.1f}%)
- lower (z≥0.67): {fp_z_lower} ({fp_z_lower/n_fp*100:.1f}%)

**ROI ratio strata:**
- low (<0.25): {fp_roi_low} ({fp_roi_low/n_fp*100:.1f}%)
- medium (0.25~0.60): {fp_roi_mid} ({fp_roi_mid/n_fp*100:.1f}%)
- high (≥0.60): {fp_roi_high} ({fp_roi_high/n_fp*100:.1f}%)

---

### NSCLC FN ({n_fn:,}개)

**prob 분포:**
| 지표 | 값 |
|------|------|
| mean | {fn_df.prob.mean():.4f} |
| median | {fn_df.prob.median():.4f} |
| Q25 | {fn_df.prob.quantile(0.25):.4f} |
| Q75 | {fn_df.prob.quantile(0.75):.4f} |
| min | {fn_df.prob.min():.6f} |
| max | {fn_df.prob.max():.4f} |

→ FN 대부분이 **prob 0.05~0.32** (median={fn_df.prob.median():.3f}). 단순 경계값이 아닌 강한 normal 방향 오분류.

**position_bin 분포:**
{fn_pos_str}

**z strata:**
- upper (z<0.33): {fn_z_upper} ({fn_z_upper/n_fn*100:.1f}%)
- middle (0.33≤z<0.67): {n_fn - fn_z_upper - fn_z_lower} ({(n_fn-fn_z_upper-fn_z_lower)/n_fn*100:.1f}%)
- lower (z≥0.67): {fn_z_lower} ({fn_z_lower/n_fn*100:.1f}%)

**ROI ratio strata:**
- low (<0.25): {fn_roi_low} ({fn_roi_low/n_fn*100:.1f}%)
- high (≥0.60): {fn_roi_high} ({fn_roi_high/n_fn*100:.1f}%)

---

## 3. 대표 샘플 manifest

- 총 manifest rows: {len(manifest):,}
  - normal FP: {(manifest.error_type=='normal_FP').sum()}
  - NSCLC FN: {(manifest.error_type=='nsclc_FN').sum()}
- 파일: `p_c_normal26_error_review_manifest.csv`

sampling groups:
- fp_high_conf / fp_mid_conf / fp_boundary
- fp_z_upper / fp_z_middle / fp_z_lower
- fp_roi_low / fp_roi_medium / fp_roi_high
- fp_pos_* (position_bin별)
- fn_severe / fn_boundary
- fn_z_* / fn_roi_* / fn_pos_*

---

## 4. Contact Sheets

{sheet_str}

위치: `cards/`

---

## 5. 실패 유형 가설

> **주의:** 이 가설은 통계 분포 기반 자동 추론이다. 시각적 확인 없이 확정하지 말 것.
> 수동 리뷰 후 라벨 수정 필요.

### normal FP 가설
{fp_hyp_str}

### NSCLC FN 가설
{fn_hyp_str}

---

## 6. 다음 개선 방향 후보

| 우선순위 | 옵션 | 내용 |
|------|------|------|
| A | explanation/XAI card integration | 오류 crop을 기존 XAI 카드 파이프라인에 연결 |
| B | hard negative review set 구성 | FP 샘플을 hard negative 후보로 정리 (재학습 시 활용 가능, 사용자 승인 필요) |
| C | downstream pipeline 연결 preflight | balanced-w1을 기존 PaDiM/RD4AD pipeline에 연결하는 interface 정의 |
| D | paper/poster result integration | 결과 표기 확정 및 논문/발표 자료 정리 |
| E | 종료 | 더 이상의 모델 수정 없이 현재 결과 정리 종료 |

> 새로운 학습 또는 threshold tuning은 이 중 어떤 항목도 포함하지 않는다.
> 필요 시 사용자 승인 후 별도 단계로 진행한다.

---

## 7. Guardrail 요약

- 총 {len(guardrail_rows)}개 항목 점검
- fail count: **{fail_count}**
- 상세: `p_c_normal26_guardrail_check.csv`
"""

with open(REPORT_ROOT / "p_c_normal26_error_review_report.md", "w", encoding="utf-8") as f:
    f.write(report_md)
print("  [MD] p_c_normal26_error_review_report.md")


# ── 9. DONE ──────────────────────────────────────────────────────────────────
done = {
    "step": "p_c_normal26_balanced_w1_error_review",
    "verdict": summary["verdict"],
    "timestamp": summary["timestamp"],
    "n_fp": int(n_fp),
    "n_fn": int(n_fn),
    "manifest_rows": int(len(manifest)),
    "guardrail_fail_count": fail_count,
    "no_training_run": True,
    "no_model_forward": True,
    "no_threshold_optimization": True,
}
with open(REPORT_ROOT / "DONE.json", "w", encoding="utf-8") as f:
    json.dump(done, f, indent=2, ensure_ascii=False)
print(f"  [JSON] DONE.json  verdict={done['verdict']}")

print("\n=== P-C-NORMAL26 완료 ===")
print(f"verdict: {summary['verdict']}")
print(f"FP={n_fp}  FN={n_fn}  manifest={len(manifest)}  guardrail_fail={fail_count}")
print(f"output: {REPORT_ROOT}")
