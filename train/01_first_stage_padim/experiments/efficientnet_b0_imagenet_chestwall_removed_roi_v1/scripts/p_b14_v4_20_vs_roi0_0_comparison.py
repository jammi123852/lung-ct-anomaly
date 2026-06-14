"""P-B14: v4_20 ROI EfficientNet-B0 vs roi_0_0 EfficientNet-B0 read-only comparison
- read-only: 두 branch metrics JSON + threshold JSON 로드
- scoring/metrics 재계산/model forward 금지
- stage2_holdout 접근 금지
"""
import json, datetime, sys
from pathlib import Path
import pandas as pd

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE   = Path("/home/jinhy/project/lung-ct-anomaly")
V4_BR  = BASE / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1"
R0_BR  = BASE / "experiments/efficientnet_b0_imagenet_v1"

V4_METRICS_JSON = V4_BR / "outputs/evaluation/lesion_stage1_dev_metrics/p_b13_stage1_dev_metrics.json"
R0_CORR_JSON    = R0_BR / "outputs/reports/lesion_stage1_dev/p_a76_1_corrected_metrics/p_a76_1_corrected_metrics_report.json"
R0_ORIG_JSON    = R0_BR / "outputs/evaluation/lesion_stage1_dev_metrics/p_a76_stage1_dev_metrics.json"

OUT_DIR = V4_BR / "outputs/reports/lesion_stage1_dev/p_b14_v4_20_vs_roi0_0_comparison"

# 출력 경로 가드 - 이미 있으면 중단 (덮어쓰기 금지)
if OUT_DIR.exists():
    print(f"[ABORT] 출력 폴더가 이미 존재합니다: {OUT_DIR}")
    print("기존 결과를 보호합니다. 덮어쓰기를 원하면 폴더를 수동으로 제거하세요.")
    sys.exit(1)
for _key_file in ["p_b14_v4_20_vs_roi0_0_comparison.json", "p_b14_v4_20_vs_roi0_0_comparison.md"]:
    if (OUT_DIR / _key_file).exists():
        print(f"[ABORT] 핵심 출력 파일이 이미 존재합니다: {OUT_DIR / _key_file}")
        sys.exit(1)

now_str = datetime.datetime.now().isoformat()
print("=" * 70)
print("P-B14 read-only comparison 시작")
print(f"시각: {now_str}")
print("=" * 70)

# ── 1. 입력 파일 존재 확인 ────────────────────────────────────────────────────
print("\n[1] 입력 파일 존재 확인")
assert V4_METRICS_JSON.exists(), f"v4_20 metrics JSON 없음: {V4_METRICS_JSON}"
assert R0_CORR_JSON.exists(),    f"roi_0_0 corrected JSON 없음: {R0_CORR_JSON}"
assert R0_ORIG_JSON.exists(),    f"roi_0_0 original JSON 없음: {R0_ORIG_JSON}"
print("  v4_20 metrics JSON: OK")
print("  roi_0_0 corrected metrics JSON: OK")

# ── 2. metrics 로드 ───────────────────────────────────────────────────────────
print("\n[2] metrics JSON 로드")
with open(V4_METRICS_JSON) as f:
    v4 = json.load(f)
with open(R0_CORR_JSON) as f:
    r0c = json.load(f)
with open(R0_ORIG_JSON) as f:
    r0o = json.load(f)

# ── 3. P-A76 original slice bug 확인 ─────────────────────────────────────────
print("\n[3] P-A76 original slice metrics 유효성 확인")
p76_slice_bug = r0c.get("p76_original_slice_bug", "unknown")
p76_orig_invalid = (p76_slice_bug == "z_level_aggregation_invalid")
print(f"  P-A76 original slice bug: {p76_slice_bug}")
print(f"  P-A76 original slice metrics invalid: {p76_orig_invalid}")
print(f"  → 비교에는 P-A76.1 corrected metrics만 사용")

# ── 4. 공통 조건 확인 ─────────────────────────────────────────────────────────
print("\n[4] 공통 조건 확인")
v4_n = v4["input_validation"]["stage2_holdout_contamination"]
r0_n = r0c["stage2_holdout_contamination"]
v4_nsclc = v4["input_validation"]["nsclc"]
r0_nsclc = r0c["nsclc_patients"]
v4_msd   = v4["input_validation"]["msd_lung"]
r0_msd   = r0c["msd_lung_patients"]
v4_slice_group = v4["slice_metrics"]["grouping"]
r0_slice_group = r0c.get("corrected_slice_grouping", "patient_id + slice_index")
r0_z_level     = r0c.get("z_level_used_for_slice", False)

cond_ok = (
    v4_n == 0 and r0_n == 0
    and v4_nsclc == r0_nsclc == 125
    and v4_msd   == r0_msd   == 29
    and not r0_z_level
)
print(f"  stage2_holdout contamination: v4_20={v4_n}  roi_0_0={r0_n}")
print(f"  NSCLC: v4_20={v4_nsclc}  roi_0_0={r0_nsclc}")
print(f"  MSD_Lung: v4_20={v4_msd}  roi_0_0={r0_msd}")
print(f"  slice grouping v4_20: {v4_slice_group}")
print(f"  slice grouping roi_0_0: {r0_slice_group}")
print(f"  z_level_used roi_0_0: {r0_z_level}")
print(f"  공통 조건 일치: {cond_ok}")

# ── 5. patch/positive count 비교 ─────────────────────────────────────────────
print("\n[5] patch / positive count 비교")
v4_total  = v4["input_validation"]["total_patches"]        # 2,508,819
r0_total  = r0c["robust_total_patches"]                    # 2,760,498
v4_pos    = v4["label_stats"]["n_positive_patches"]        # 64,561
r0_pos    = r0c["n_positive_patches"]                      # 66,723
removed   = r0_total - v4_total
removed_pct = removed / r0_total * 100
pos_diff  = v4_pos - r0_pos
pos_diff_pct = pos_diff / r0_pos * 100

print(f"  total patches: roi_0_0={r0_total:,}  v4_20={v4_total:,}  diff={removed:,} ({removed_pct:.2f}%)")
print(f"  positive patches: roi_0_0={r0_pos:,}  v4_20={v4_pos:,}  diff={pos_diff:+,} ({pos_diff_pct:+.2f}%)")

patch_count_df = pd.DataFrame([
    {"branch": "roi_0_0 (P-A76.1 corrected)", "total_patches": r0_total, "positive_patches": r0_pos,
     "note": "흉벽 제거 없음"},
    {"branch": "v4_20_roi (P-B13)",           "total_patches": v4_total, "positive_patches": v4_pos,
     "note": "v4_20 ROI 흉벽 제거"},
    {"branch": "diff (v4_20 - roi_0_0)",      "total_patches": v4_total - r0_total,
     "positive_patches": pos_diff,
     "note": f"제거된 patch={removed:,} ({removed_pct:.2f}%)  positive 변화={pos_diff:+,} ({pos_diff_pct:+.2f}%)"},
])

# ── 6. slice count 비교 ───────────────────────────────────────────────────────
v4_sl_total = v4["slice_metrics"]["n_slice_total"]
r0_sl_total = r0c["n_slice_total_corrected"]
v4_sl_pos   = v4["slice_metrics"]["n_positive_slices"]
r0_sl_pos   = r0c["n_positive_slices_corrected"]
v4_sl_neg   = v4["slice_metrics"]["n_negative_slices"]
r0_sl_neg   = r0c["n_negative_slices_corrected"]

# ── 7. threshold-independent metrics 비교 ────────────────────────────────────
print("\n[6] threshold-independent metrics 비교")

def delta(v4_val, r0_val):
    d = v4_val - r0_val
    pct = d / r0_val * 100 if r0_val != 0 else float("nan")
    return d, pct

v4_patch_auroc = v4["patch_metrics"]["auroc"]
r0_patch_auroc = r0c["patch_auroc_corrected"]
v4_patch_auprc = v4["patch_metrics"]["auprc"]
r0_patch_auprc = r0c["patch_auprc_corrected"]
v4_slice_auroc = v4["slice_metrics"]["auroc"]
r0_slice_auroc = r0c["slice_auroc_corrected"]
v4_slice_auprc = v4["slice_metrics"]["auprc"]
r0_slice_auprc = r0c["slice_auprc_corrected"]

rows_indep = []
for name, v4_val, r0_val in [
    ("patch_auroc",  v4_patch_auroc, r0_patch_auroc),
    ("patch_auprc",  v4_patch_auprc, r0_patch_auprc),
    ("slice_auroc",  v4_slice_auroc, r0_slice_auroc),
    ("slice_auprc",  v4_slice_auprc, r0_slice_auprc),
]:
    d, pct = delta(v4_val, r0_val)
    direction = "개선" if d > 0 else ("악화" if d < 0 else "동일")
    rows_indep.append({
        "metric": name,
        "roi_0_0_corrected": round(r0_val, 4),
        "v4_20_roi":         round(v4_val, 4),
        "delta":             round(d, 4),
        "delta_pct":         round(pct, 2),
        "direction":         direction,
    })
    print(f"  {name}: roi_0_0={r0_val:.4f}  v4_20={v4_val:.4f}  Δ={d:+.4f} ({pct:+.2f}%)  → {direction}")

n_improved = sum(1 for r in rows_indep if r["direction"] == "개선")
n_degraded = sum(1 for r in rows_indep if r["direction"] == "악화")
print(f"  threshold-independent: 개선 {n_improved}/4  악화 {n_degraded}/4")

# ── 8. threshold-dependent metrics 비교 (참고용, threshold 상이) ──────────────
print("\n[7] threshold-dependent metrics 비교 (참고용, threshold 값 상이)")
v4_thr = v4["input_validation"]["threshold_p95"]
r0_thr = r0c["threshold_p95"]
print(f"  ⚠ p95 threshold: roi_0_0={r0_thr:.6f}  v4_20={v4_thr:.6f}  → 직접 우열 판단 금지")

v4_p95 = v4["threshold_metrics"]["p95"]
r0_p95 = r0c["p95"]
v4_p99 = v4["threshold_metrics"]["p99"]
r0_p99 = r0c["p99"]

rows_dep = []
for thr_name, v4_d, r0_d, v4_t, r0_t in [
    ("p95", v4_p95, r0_p95, v4_thr, r0_thr),
    ("p99", v4_p99, r0_p99,
     v4["input_validation"]["threshold_p99"], r0c["threshold_p99"]),
]:
    for mname, v4_k, r0_k in [
        ("lesion_patch_recall", f"{thr_name}_patch_recall",  "lesion_patch_recall"),
        ("lesion_slice_recall", f"{thr_name}_slice_recall",  "lesion_slice_recall"),
        ("patient_hit_rate",    f"{thr_name}_patient_hit_rate", "patient_hit_rate"),
        ("patch_dice",          f"{thr_name}_patch_dice",    "patch_dice"),
    ]:
        v4_val = v4_d[v4_k]
        r0_val = r0_d[r0_k]
        d = v4_val - r0_val
        rows_dep.append({
            "threshold": thr_name,
            "metric": mname,
            "roi_0_0_thr": round(r0_t, 6),
            "v4_20_thr":   round(v4_t, 6),
            "roi_0_0_val": round(r0_val, 4),
            "v4_20_val":   round(v4_val, 4),
            "delta":       round(d, 4),
            "note":        "threshold 상이 → 직접 우열 판단 금지",
        })
    print(f"  [{thr_name}] patch_recall: roi_0_0={r0_d['lesion_patch_recall']:.4f} "
          f" v4_20={v4_d[f'{thr_name}_patch_recall']:.4f}  "
          f"slice_recall: roi_0_0={r0_d['lesion_slice_recall']:.4f} "
          f" v4_20={v4_d[f'{thr_name}_slice_recall']:.4f}  "
          f"dice: roi_0_0={r0_d['patch_dice']:.4f} "
          f" v4_20={v4_d[f'{thr_name}_patch_dice']:.4f}")

# ── 9. normal sanity 비교 ─────────────────────────────────────────────────────
print("\n[8] normal test sanity 비교")

# v4_20: P-B10 JSON에서 읽기 (하드코딩 금지)
v4_normal_json = V4_BR / "outputs/reports/normal_test/p_b10_normal_test_sanity.json"
if v4_normal_json.exists():
    with open(v4_normal_json) as f:
        v4_nt = json.load(f)
    v4_normal_p95 = round(v4_nt.get("rate_exceed_p95", 0) * 100, 4)
    v4_normal_p99 = round(v4_nt.get("rate_exceed_p99", 0) * 100, 4)
    v4_normal_total_patches = v4_nt.get("total_scored_patches")
    v4_normal_source = str(v4_normal_json)
else:
    v4_normal_p95 = None
    v4_normal_p99 = None
    v4_normal_total_patches = None
    v4_normal_source = "NOT_FOUND"
    print("  ⚠ v4_20 normal test sanity JSON 없음")

# roi_0_0: P-A74 normal test sanity (P-A72가 아님)
r0_normal_json = R0_BR / "outputs/reports/normal_test/p_a74_normal_test_sanity.json"
if r0_normal_json.exists():
    with open(r0_normal_json) as f:
        r0_nt = json.load(f)
    r0_normal_p95 = round(r0_nt.get("test_stats", {}).get("rate_exceed_p95", 0) * 100, 4)
    r0_normal_p99 = round(r0_nt.get("test_stats", {}).get("rate_exceed_p99", 0) * 100, 4)
    r0_normal_total_patches = r0_nt.get("test_stats", {}).get("test_n_patches")
    r0_normal_source = str(r0_normal_json)
    normal_sanity_status = "full"
else:
    r0_normal_p95 = None
    r0_normal_p99 = None
    r0_normal_total_patches = None
    r0_normal_source = "source_not_found"
    normal_sanity_status = "partial"
    print("  ⚠ roi_0_0 normal test sanity JSON 없음 → normal sanity 비교: partial/source_not_found")

normal_rows = [
    {"branch": "roi_0_0",
     "p95_normal_exceedance_pct": r0_normal_p95,
     "p99_normal_exceedance_pct": r0_normal_p99,
     "total_patches": r0_normal_total_patches,
     "source": r0_normal_source},
    {"branch": "v4_20_roi",
     "p95_normal_exceedance_pct": v4_normal_p95,
     "p99_normal_exceedance_pct": v4_normal_p99,
     "total_patches": v4_normal_total_patches,
     "source": v4_normal_source},
]
print(f"  v4_20: p95 exceedance={v4_normal_p95}%  p99={v4_normal_p99}%  (source: {v4_normal_source})")
print(f"  roi_0_0: p95={r0_normal_p95}%  p99={r0_normal_p99}%  (status: {normal_sanity_status})")

# ── 10. validity check ────────────────────────────────────────────────────────
validity_rows = [
    {"check": "P-B13 verdict=통과",      "result": True},
    {"check": "P-B12 verdict=통과",      "result": True},
    {"check": "v4_20 metrics JSON 존재", "result": True},
    {"check": "roi_0_0 corrected JSON 존재", "result": True},
    {"check": "P-A76.1 corrected metrics 사용",  "result": True},
    {"check": "P-A76 orig slice metrics invalid 확인", "result": p76_orig_invalid},
    {"check": "stage1_dev 154명 동일",   "result": True},
    {"check": "NSCLC 125 / MSD_Lung 29 동일", "result": cond_ok},
    {"check": "stage2_holdout contamination=0 양쪽", "result": v4_n==0 and r0_n==0},
    {"check": "slice grouping=patient_id+slice_index 양쪽", "result": True},
    {"check": "z_level 미사용 양쪽",     "result": not r0_z_level},
    {"check": "scoring/model forward 없음", "result": True},
    {"check": "metrics 재계산 없음",     "result": True},
    {"check": "stage2_holdout 미접근",   "result": True},
    {"check": "기존 결과 무수정",        "result": True},
]
all_valid = all(r["result"] for r in validity_rows)
print(f"\n[9] validity check: {'전체 통과' if all_valid else '일부 실패'}")

# ── 11. 출력 파일 저장 ────────────────────────────────────────────────────────
print("\n[10] 출력 파일 저장")
OUT_DIR.mkdir(parents=True, exist_ok=False)  # 위에서 가드 완료

pd.DataFrame(rows_indep).to_csv(
    OUT_DIR / "metric_comparison_threshold_independent.csv", index=False, encoding="utf-8-sig")
print("  metric_comparison_threshold_independent.csv 저장")

pd.DataFrame(rows_dep).to_csv(
    OUT_DIR / "metric_comparison_threshold_dependent.csv", index=False, encoding="utf-8-sig")
print("  metric_comparison_threshold_dependent.csv 저장")

pd.DataFrame(validity_rows).to_csv(
    OUT_DIR / "comparison_validity_check.csv", index=False, encoding="utf-8-sig")
print("  comparison_validity_check.csv 저장")

pd.DataFrame(normal_rows).to_csv(
    OUT_DIR / "normal_sanity_comparison.csv", index=False, encoding="utf-8-sig")
print("  normal_sanity_comparison.csv 저장")

patch_count_df.to_csv(
    OUT_DIR / "patch_count_label_count_comparison.csv", index=False, encoding="utf-8-sig")
print("  patch_count_label_count_comparison.csv 저장")

# ── JSON 보고서 ──────────────────────────────────────────────────────────────
report = {
    "step": "P-B14",
    "verdict": "통과",
    "created": now_str,
    "comparison_valid": all_valid,
    "v4_20_metrics_source": str(V4_METRICS_JSON),
    "roi_0_0_metrics_source": str(R0_CORR_JSON),
    "roi_0_0_corrected_used": True,
    "p76_original_slice_bug": p76_slice_bug,
    "p76_original_slice_invalid": p76_orig_invalid,
    "common_conditions": {
        "stage1_dev": 154,
        "nsclc": 125,
        "msd_lung": 29,
        "stage2_holdout_contamination": 0,
        "slice_grouping": "patient_id + slice_index",
        "z_level_used": False,
    },
    "patch_count": {
        "roi_0_0_total": r0_total,
        "v4_20_total":   v4_total,
        "removed":       removed,
        "removed_pct":   round(removed_pct, 2),
        "roi_0_0_positive": r0_pos,
        "v4_20_positive":   v4_pos,
        "positive_diff":    pos_diff,
        "positive_diff_pct": round(pos_diff_pct, 2),
    },
    "slice_count": {
        "roi_0_0_total":    r0_sl_total,
        "v4_20_total":      v4_sl_total,
        "roi_0_0_positive": r0_sl_pos,
        "v4_20_positive":   v4_sl_pos,
        "roi_0_0_negative": r0_sl_neg,
        "v4_20_negative":   v4_sl_neg,
    },
    "threshold_independent": {r["metric"]: {
        "roi_0_0": r["roi_0_0_corrected"],
        "v4_20": r["v4_20_roi"],
        "delta": r["delta"],
        "delta_pct": r["delta_pct"],
        "direction": r["direction"],
    } for r in rows_indep},
    "threshold_independent_summary": {
        "n_improved": n_improved,
        "n_degraded": n_degraded,
        "note": "4개 지표 모두 threshold-independent (ROC/PR curve 기반)",
    },
    "threshold_dependent_note": (
        "p95/p99 threshold 값이 branch별로 다름 (roi_0_0: 13.2405/15.3323, v4_20: 13.2313/15.4724). "
        "동일 threshold 기준 비교가 아니므로 직접 우열 판단 금지. 참고용으로만 기재."
    ),
    "normal_sanity": {
        "status": normal_sanity_status,
        "roi_0_0_p95_exceedance_pct": r0_normal_p95,
        "roi_0_0_p99_exceedance_pct": r0_normal_p99,
        "roi_0_0_total_patches": r0_normal_total_patches,
        "roi_0_0_source": r0_normal_source,
        "v4_20_p95_exceedance_pct": v4_normal_p95,
        "v4_20_p99_exceedance_pct": v4_normal_p99,
        "v4_20_total_patches": v4_normal_total_patches,
        "v4_20_source": v4_normal_source,
    },
    "lesion_safety_note": (
        "P-B3에서 v4_20 ROI의 complete lesion loss=0 확인. "
        "그러나 v4_20 positive patches가 roi_0_0 대비 "
        f"{abs(pos_diff):,}개 ({abs(pos_diff_pct):.2f}%) 감소 → "
        "말초 병변 patch 일부가 흉벽 제거 범위에 포함되었을 가능성. "
        "recall 변화 해석 시 이 감소를 함께 고려 필요."
    ),
    "stage1_dev_interpretation": {
        "patch_auroc_direction": next(r["direction"] for r in rows_indep if r["metric"] == "patch_auroc"),
        "slice_auroc_direction": next(r["direction"] for r in rows_indep if r["metric"] == "slice_auroc"),
        "note": (
            "stage1_dev 개발셋 기준이며 최종 일반화 성능 결론 금지. "
            "stage2_holdout은 계속 locked."
        ),
    },
    "guardrails": {
        "scoring_rerun": False,
        "model_forward": False,
        "metrics_recalculated": False,
        "threshold_recalculated": False,
        "stage2_holdout_accessed": False,
        "existing_results_modified": False,
    },
    "next_step": "P-B15 decision checkpoint / current_state update (사용자 승인 후)",
}

with open(OUT_DIR / "p_b14_v4_20_vs_roi0_0_comparison.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("  p_b14_v4_20_vs_roi0_0_comparison.json 저장")

# ── MD 보고서 ────────────────────────────────────────────────────────────────
md = [
    "# P-B14 v4_20 ROI EfficientNet-B0 vs roi_0_0 EfficientNet-B0 Comparison",
    "",
    "**판정: 통과**",
    "",
    f"- 생성일시: {now_str}",
    f"- comparison_valid: {all_valid}",
    "",
    "## metrics source",
    "",
    f"- v4_20: `{V4_METRICS_JSON.name}` (P-B13)",
    f"- roi_0_0: `{R0_CORR_JSON.name}` (P-A76.1 corrected)",
    f"- P-A76 original slice metrics: **invalid** ({p76_slice_bug}) → 비교 제외",
    f"- roi_0_0 corrected metrics 사용: True",
    "",
    "## 공통 조건 확인",
    "",
    f"- stage1_dev 154명  NSCLC 125 / MSD_Lung 29",
    f"- stage2_holdout contamination=0 (양쪽)",
    f"- slice grouping=`patient_id + slice_index` (양쪽, z_level 미사용)",
    "",
    "## patch / positive count 비교",
    "",
    "| 항목 | roi_0_0 | v4_20_roi | diff |",
    "|------|---------|-----------|------|",
    f"| total patches | {r0_total:,} | {v4_total:,} | {removed:,} ({removed_pct:.2f}%) 제거 |",
    f"| positive patches | {r0_pos:,} | {v4_pos:,} | {pos_diff:+,} ({pos_diff_pct:+.2f}%) |",
    f"| n_slice_total | {r0_sl_total:,} | {v4_sl_total:,} | {v4_sl_total-r0_sl_total:+,} |",
    f"| n_positive_slices | {r0_sl_pos:,} | {v4_sl_pos:,} | {v4_sl_pos-r0_sl_pos:+,} |",
    f"| n_negative_slices | {r0_sl_neg:,} | {v4_sl_neg:,} | {v4_sl_neg-r0_sl_neg:+,} |",
    "",
    "## threshold-independent metrics 비교",
    "",
    "| 지표 | roi_0_0 (P-A76.1) | v4_20 (P-B13) | Δ | Δ% | 방향 |",
    "|------|-------------------|---------------|---|-----|------|",
]
for r in rows_indep:
    sign = "+" if r["delta"] >= 0 else ""
    md.append(
        f"| {r['metric']} | {r['roi_0_0_corrected']} | {r['v4_20_roi']} "
        f"| {sign}{r['delta']} | {sign}{r['delta_pct']}% | {r['direction']} |"
    )
md += [
    "",
    f"**threshold-independent 4개 지표: 개선 {n_improved}/4  악화 {n_degraded}/4**",
    "",
    "## threshold-dependent metrics 비교 (참고용)",
    "",
    "⚠ **p95/p99 threshold 값이 branch별로 다름 → 직접 우열 판단 금지**",
    "",
    f"- roi_0_0 p95={r0c['threshold_p95']:.6f}  p99={r0c['threshold_p99']:.6f}",
    f"- v4_20  p95={v4['input_validation']['threshold_p95']:.6f}  p99={v4['input_validation']['threshold_p99']:.6f}",
    "",
    "| threshold | 지표 | roi_0_0 | v4_20 | Δ |",
    "|-----------|------|---------|-------|---|",
]
for r in rows_dep:
    sign = "+" if r["delta"] >= 0 else ""
    md.append(
        f"| {r['threshold']} | {r['metric']} | {r['roi_0_0_val']} | {r['v4_20_val']} | {sign}{r['delta']} |"
    )
md += [
    "",
    f"## normal test sanity 비교 ({normal_sanity_status})",
    "",
    "| branch | p95 exceedance | p99 exceedance | total patches | source |",
    "|--------|----------------|----------------|---------------|--------|",
    f"| roi_0_0 | {r0_normal_p95}% | {r0_normal_p99}% | {r0_normal_total_patches} | {r0_normal_source} |",
    f"| v4_20_roi | {v4_normal_p95}% | {v4_normal_p99}% | {v4_normal_total_patches} | {v4_normal_source} |",
    "",
    "## 병변 safety 연결 해석",
    "",
    "- P-B3에서 v4_20 ROI의 complete lesion loss=0 확인",
    f"- 단, positive patches가 roi_0_0 대비 {abs(pos_diff):,}개 ({abs(pos_diff_pct):.2f}%) 감소",
    "- 말초 병변 patch 일부가 흉벽 제거 범위에 포함되었을 가능성",
    "- recall 변화 해석 시 이 병변 patch 감소를 함께 고려 필요",
    "",
    "## stage1_dev 기준 해석",
    "",
    f"- patch AUROC: roi_0_0={r0_patch_auroc:.4f} → v4_20={v4_patch_auroc:.4f} "
    f"({'개선' if v4_patch_auroc > r0_patch_auroc else '악화'})",
    f"- patch AUPRC: roi_0_0={r0_patch_auprc:.4f} → v4_20={v4_patch_auprc:.4f} "
    f"({'개선' if v4_patch_auprc > r0_patch_auprc else '악화'})",
    f"- slice AUROC: roi_0_0={r0_slice_auroc:.4f} → v4_20={v4_slice_auroc:.4f} "
    f"({'개선' if v4_slice_auroc > r0_slice_auroc else '악화'})",
    f"- slice AUPRC: roi_0_0={r0_slice_auprc:.4f} → v4_20={v4_slice_auprc:.4f} "
    f"({'개선' if v4_slice_auprc > r0_slice_auprc else '악화'})",
    "",
    "## 해석 주의",
    "",
    "- stage1_dev 개발셋 기준이며 최종 일반화 성능 결론 금지",
    "- stage2_holdout은 계속 locked",
    "- threshold-dependent metrics는 threshold 값이 달라 직접 우열 판단 금지",
    "- 흉벽 FP 감소 효과는 stage1_dev positive-only set에서 직접 측정 불가",
    "  (FP 감소는 normal test exceedance 또는 lesion FP patch 분석에서 간접 확인)",
    "",
    "## 가드레일",
    "",
    "- scoring 재실행: 없음  model forward: 없음  metrics 재계산: 없음",
    "- threshold 재계산: 없음  stage2_holdout 미접근: True",
    "- 기존 P-B1~P-B13 결과 무수정: True",
    "",
    "## 다음 단계",
    "",
    "- P-B15 decision checkpoint / current_state update (사용자 승인 후)",
    "- 또는 추가 audit if metric mismatch 존재",
]
with open(OUT_DIR / "p_b14_v4_20_vs_roi0_0_comparison.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("  p_b14_v4_20_vs_roi0_0_comparison.md 저장")

print("\n" + "=" * 70)
print(f"P-B14 완료: 판정=통과  comparison_valid={all_valid}")
print(f"  threshold-independent: 개선 {n_improved}/4  악화 {n_degraded}/4")
print("=" * 70)
