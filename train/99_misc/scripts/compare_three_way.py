"""
v1/v1, v1/v2, v2/v2 3-way 비교 스크립트
read-only 입력, 신규 CSV/JSON/MD만 생성
"""
import json
import csv
import os
from pathlib import Path

BASE = Path("outputs/position-aware-padim-v1")
OUT_DIR = BASE / "reports_v2_roi0_0_lesion"

# ── 입력 파일 경로 ──────────────────────────────────────────────────────────
SOURCES = {
    "v1v1": {
        "p95_summary": BASE / "evaluation/lesion_subset/lesion_eval_p95_fast_summary.json",
        "p99_summary": BASE / "evaluation/lesion_subset/lesion_eval_p99_fast_summary.json",
        "screening":   BASE / "evaluation/lesion_subset/screening_analysis_summary.json",
        "hit_overlap": BASE / "reports/lesion_hit_overlap_summary.json",
    },
    "v1v2": {
        "p95_summary": BASE / "evaluation/lesion_subset_v2/lesion_eval_v2_p95_fast_summary.json",
        "p99_summary": BASE / "evaluation/lesion_subset_v2/lesion_eval_v2_p99_fast_summary.json",
        "screening":   BASE / "evaluation/lesion_subset_v2/screening_analysis_summary.json",
        "hit_overlap": BASE / "reports/lesion_hit_overlap_summary_v2.json",
    },
    "v2v2": {
        "p95_summary": BASE / "evaluation/lesion_subset_v2_model_v2/lesion_eval_p95_fast_summary.json",
        "p99_summary": BASE / "evaluation/lesion_subset_v2_model_v2/lesion_eval_p99_fast_summary.json",
        "screening":   BASE / "evaluation/lesion_subset_v2_model_v2/screening_analysis_summary.json",
        "hit_overlap": BASE / "reports_v2_roi0_0_lesion/lesion_hit_overlap_summary.json",
    },
}

def load_json(path):
    if not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def fmt(v, digits=4):
    if v is None:
        return "missing"
    if isinstance(v, float):
        return round(v, digits)
    return v

def screening_by_mode(screening_data, mode):
    if screening_data is None:
        return {}
    for m in screening_data.get("metrics", []):
        if m.get("threshold_mode") == mode:
            return m
    return {}

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
data = {}
for key, paths in SOURCES.items():
    data[key] = {k: load_json(v) for k, v in paths.items()}

# 파일 존재 여부 확인
print("=== 입력 파일 존재 여부 ===")
for combo, paths in SOURCES.items():
    for k, p in paths.items():
        status = "OK" if Path(p).exists() else "MISSING"
        print(f"  [{combo}][{k}] {status}: {p}")

# ── A. fast metrics 수집 ─────────────────────────────────────────────────────
def get_fast_metrics(combo):
    p95 = data[combo]["p95_summary"]
    p99 = data[combo]["p99_summary"]
    if p95 is None and p99 is None:
        return {}
    # AUROC/AUPRC는 threshold 무관하므로 p95에서 가져옴
    base = p95 or p99
    return {
        "patch_auroc":   fmt(base.get("patch_auroc")),
        "patch_auprc":   fmt(base.get("patch_auprc")),
        "slice_auroc":   fmt(base.get("slice_auroc")),
        "slice_auprc":   fmt(base.get("slice_auprc")),
        "patient_auroc": base.get("patient_auroc_status", "not_applicable"),
        "p95_threshold": fmt(p95.get("threshold_value") if p95 else None),
        "p95_patch_dice": fmt(p95.get("patch_dice") if p95 else None),
        "p95_patch_iou":  fmt(p95.get("patch_iou") if p95 else None),
        "p99_threshold": fmt(p99.get("threshold_value") if p99 else None),
        "p99_patch_dice": fmt(p99.get("patch_dice") if p99 else None),
        "p99_patch_iou":  fmt(p99.get("patch_iou") if p99 else None),
        "lesion_patch_total": p95.get("patch_positive") if p95 else None,
        "lesion_slice_total": p95.get("slice_positive") if p95 else None,
    }

# ── B. screening 수집 ────────────────────────────────────────────────────────
def get_screening(combo):
    s = data[combo]["screening"]
    p95m = screening_by_mode(s, "normal_val_p95")
    p99m = screening_by_mode(s, "normal_val_p99")
    return {
        "p95_lesion_patch_recall":   fmt(p95m.get("lesion_patch_recall")),
        "p95_lesion_slice_recall":   fmt(p95m.get("lesion_slice_recall")),
        "p95_patient_coverage_mean": fmt(p95m.get("patient_coverage_mean")),
        "p95_patient_hit_rate":      fmt(p95m.get("patient_hit_rate")),
        "p95_topk10_coverage":       fmt(p95m.get("topk10_coverage")),
        "p95_topk30_coverage":       fmt(p95m.get("topk30_coverage")),
        "p95_topk50_coverage":       fmt(p95m.get("topk50_coverage")),
        "p99_lesion_patch_recall":   fmt(p99m.get("lesion_patch_recall")),
        "p99_lesion_slice_recall":   fmt(p99m.get("lesion_slice_recall")),
        "p99_patient_coverage_mean": fmt(p99m.get("patient_coverage_mean")),
        "p99_patient_hit_rate":      fmt(p99m.get("patient_hit_rate")),
    }

# ── C. hit overlap 수집 ──────────────────────────────────────────────────────
def get_hit_overlap(combo):
    h = data[combo]["hit_overlap"]
    if h is None:
        return {}
    no_hit = [p.get("patient_id") for p in h.get("no_hit_patients", [])]
    low10 = [
        f"{p.get('patient_id')}({p.get('patient_patch_recall', 'N/A')})"
        for p in h.get("lowest_patient_patch_recall_top10", [])
    ]
    missed10 = [
        f"{p.get('patient_id')}({int(p.get('missed_lesion_slice_count', 0))})"
        for p in h.get("most_missed_lesion_slice_top10", [])
    ]
    return {
        "p95_threshold":               fmt(h.get("p95_threshold")),
        "patient_hit_rate":            fmt(h.get("patient_hit_rate")),
        "n_patient_no_hit":            h.get("n_patient_no_hit", 0),
        "no_hit_patients":             "; ".join(no_hit) if no_hit else "없음",
        "micro_lesion_patch_recall":   fmt(h.get("micro_lesion_patch_recall")),
        "micro_lesion_slice_recall":   fmt(h.get("micro_lesion_slice_recall")),
        "patient_patch_recall_mean":   fmt(h.get("patient_patch_recall_mean")),
        "patient_patch_recall_median": fmt(h.get("patient_patch_recall_median")),
        "continuous_hit_ratio_mean":   fmt(h.get("continuous_hit_ratio_mean")),
        "continuous_hit_ratio_median": fmt(h.get("continuous_hit_ratio_median")),
        "lowest_patch_recall_top10":   "; ".join(low10),
        "most_missed_slice_top10":     "; ".join(missed10),
    }

combos = ["v1v1", "v1v2", "v2v2"]
fm  = {c: get_fast_metrics(c) for c in combos}
sc  = {c: get_screening(c)    for c in combos}
ho  = {c: get_hit_overlap(c)  for c in combos}

# ── CSV 출력 ─────────────────────────────────────────────────────────────────
csv_path = OUT_DIR / "v1v1_v1v2_v2v2_three_way_comparison.csv"

rows = []

def add_row(section, metric, v1v1, v1v2, v2v2):
    rows.append({
        "section": section,
        "metric": metric,
        "v1v1": v1v1,
        "v1v2": v1v2,
        "v2v2": v2v2,
    })

# A. fast metrics
add_row("A.fast_metrics", "patch_auroc",        fm["v1v1"].get("patch_auroc"),   fm["v1v2"].get("patch_auroc"),   fm["v2v2"].get("patch_auroc"))
add_row("A.fast_metrics", "patch_auprc",        fm["v1v1"].get("patch_auprc"),   fm["v1v2"].get("patch_auprc"),   fm["v2v2"].get("patch_auprc"))
add_row("A.fast_metrics", "slice_auroc",        fm["v1v1"].get("slice_auroc"),   fm["v1v2"].get("slice_auroc"),   fm["v2v2"].get("slice_auroc"))
add_row("A.fast_metrics", "slice_auprc",        fm["v1v1"].get("slice_auprc"),   fm["v1v2"].get("slice_auprc"),   fm["v2v2"].get("slice_auprc"))
add_row("A.fast_metrics", "patient_auroc",      fm["v1v1"].get("patient_auroc"), fm["v1v2"].get("patient_auroc"), fm["v2v2"].get("patient_auroc"))
add_row("A.fast_metrics", "p95_threshold",      fm["v1v1"].get("p95_threshold"), fm["v1v2"].get("p95_threshold"), fm["v2v2"].get("p95_threshold"))
add_row("A.fast_metrics", "p95_patch_dice",     fm["v1v1"].get("p95_patch_dice"),fm["v1v2"].get("p95_patch_dice"),fm["v2v2"].get("p95_patch_dice"))
add_row("A.fast_metrics", "p95_patch_iou",      fm["v1v1"].get("p95_patch_iou"), fm["v1v2"].get("p95_patch_iou"), fm["v2v2"].get("p95_patch_iou"))
add_row("A.fast_metrics", "p99_threshold",      fm["v1v1"].get("p99_threshold"), fm["v1v2"].get("p99_threshold"), fm["v2v2"].get("p99_threshold"))
add_row("A.fast_metrics", "p99_patch_dice",     fm["v1v1"].get("p99_patch_dice"),fm["v1v2"].get("p99_patch_dice"),fm["v2v2"].get("p99_patch_dice"))
add_row("A.fast_metrics", "p99_patch_iou",      fm["v1v1"].get("p99_patch_iou"), fm["v1v2"].get("p99_patch_iou"), fm["v2v2"].get("p99_patch_iou"))
add_row("A.fast_metrics", "lesion_patch_total", fm["v1v1"].get("lesion_patch_total"), fm["v1v2"].get("lesion_patch_total"), fm["v2v2"].get("lesion_patch_total"))
add_row("A.fast_metrics", "lesion_slice_total", fm["v1v1"].get("lesion_slice_total"), fm["v1v2"].get("lesion_slice_total"), fm["v2v2"].get("lesion_slice_total"))

# B. screening
add_row("B.screening", "p95_lesion_patch_recall",   sc["v1v1"].get("p95_lesion_patch_recall"),   sc["v1v2"].get("p95_lesion_patch_recall"),   sc["v2v2"].get("p95_lesion_patch_recall"))
add_row("B.screening", "p95_lesion_slice_recall",   sc["v1v1"].get("p95_lesion_slice_recall"),   sc["v1v2"].get("p95_lesion_slice_recall"),   sc["v2v2"].get("p95_lesion_slice_recall"))
add_row("B.screening", "p95_patient_coverage_mean", sc["v1v1"].get("p95_patient_coverage_mean"), sc["v1v2"].get("p95_patient_coverage_mean"), sc["v2v2"].get("p95_patient_coverage_mean"))
add_row("B.screening", "p95_patient_hit_rate",      sc["v1v1"].get("p95_patient_hit_rate"),      sc["v1v2"].get("p95_patient_hit_rate"),      sc["v2v2"].get("p95_patient_hit_rate"))
add_row("B.screening", "p95_topk10_coverage",       sc["v1v1"].get("p95_topk10_coverage"),       sc["v1v2"].get("p95_topk10_coverage"),       sc["v2v2"].get("p95_topk10_coverage"))
add_row("B.screening", "p95_topk30_coverage",       sc["v1v1"].get("p95_topk30_coverage"),       sc["v1v2"].get("p95_topk30_coverage"),       sc["v2v2"].get("p95_topk30_coverage"))
add_row("B.screening", "p95_topk50_coverage",       sc["v1v1"].get("p95_topk50_coverage"),       sc["v1v2"].get("p95_topk50_coverage"),       sc["v2v2"].get("p95_topk50_coverage"))
add_row("B.screening", "p99_lesion_patch_recall",   sc["v1v1"].get("p99_lesion_patch_recall"),   sc["v1v2"].get("p99_lesion_patch_recall"),   sc["v2v2"].get("p99_lesion_patch_recall"))
add_row("B.screening", "p99_lesion_slice_recall",   sc["v1v1"].get("p99_lesion_slice_recall"),   sc["v1v2"].get("p99_lesion_slice_recall"),   sc["v2v2"].get("p99_lesion_slice_recall"))
add_row("B.screening", "p99_patient_coverage_mean", sc["v1v1"].get("p99_patient_coverage_mean"), sc["v1v2"].get("p99_patient_coverage_mean"), sc["v2v2"].get("p99_patient_coverage_mean"))
add_row("B.screening", "p99_patient_hit_rate",      sc["v1v1"].get("p99_patient_hit_rate"),      sc["v1v2"].get("p99_patient_hit_rate"),      sc["v2v2"].get("p99_patient_hit_rate"))

# C. hit overlap
add_row("C.hit_overlap", "p95_threshold",               ho["v1v1"].get("p95_threshold"),               ho["v1v2"].get("p95_threshold"),               ho["v2v2"].get("p95_threshold"))
add_row("C.hit_overlap", "patient_hit_rate",            ho["v1v1"].get("patient_hit_rate"),            ho["v1v2"].get("patient_hit_rate"),            ho["v2v2"].get("patient_hit_rate"))
add_row("C.hit_overlap", "n_patient_no_hit",            ho["v1v1"].get("n_patient_no_hit"),            ho["v1v2"].get("n_patient_no_hit"),            ho["v2v2"].get("n_patient_no_hit"))
add_row("C.hit_overlap", "no_hit_patients",             ho["v1v1"].get("no_hit_patients"),             ho["v1v2"].get("no_hit_patients"),             ho["v2v2"].get("no_hit_patients"))
add_row("C.hit_overlap", "micro_lesion_patch_recall",   ho["v1v1"].get("micro_lesion_patch_recall"),   ho["v1v2"].get("micro_lesion_patch_recall"),   ho["v2v2"].get("micro_lesion_patch_recall"))
add_row("C.hit_overlap", "micro_lesion_slice_recall",   ho["v1v1"].get("micro_lesion_slice_recall"),   ho["v1v2"].get("micro_lesion_slice_recall"),   ho["v2v2"].get("micro_lesion_slice_recall"))
add_row("C.hit_overlap", "patient_patch_recall_mean",   ho["v1v1"].get("patient_patch_recall_mean"),   ho["v1v2"].get("patient_patch_recall_mean"),   ho["v2v2"].get("patient_patch_recall_mean"))
add_row("C.hit_overlap", "patient_patch_recall_median", ho["v1v1"].get("patient_patch_recall_median"), ho["v1v2"].get("patient_patch_recall_median"), ho["v2v2"].get("patient_patch_recall_median"))
add_row("C.hit_overlap", "continuous_hit_ratio_mean",   ho["v1v1"].get("continuous_hit_ratio_mean"),   ho["v1v2"].get("continuous_hit_ratio_mean"),   ho["v2v2"].get("continuous_hit_ratio_mean"))
add_row("C.hit_overlap", "continuous_hit_ratio_median", ho["v1v1"].get("continuous_hit_ratio_median"), ho["v1v2"].get("continuous_hit_ratio_median"), ho["v2v2"].get("continuous_hit_ratio_median"))
add_row("C.hit_overlap", "lowest_patch_recall_top10",   ho["v1v1"].get("lowest_patch_recall_top10"),   ho["v1v2"].get("lowest_patch_recall_top10"),   ho["v2v2"].get("lowest_patch_recall_top10"))
add_row("C.hit_overlap", "most_missed_slice_top10",     ho["v1v1"].get("most_missed_slice_top10"),     ho["v1v2"].get("most_missed_slice_top10"),     ho["v2v2"].get("most_missed_slice_top10"))

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["section", "metric", "v1v1", "v1v2", "v2v2"])
    writer.writeheader()
    writer.writerows(rows)
print(f"CSV 저장: {csv_path}")

# ── JSON 출력 ─────────────────────────────────────────────────────────────────
json_path = OUT_DIR / "v1v1_v1v2_v2v2_three_way_comparison.json"

# 변화 요약 계산 (숫자인 경우만)
def diff(a, b):
    try:
        return round(float(b) - float(a), 4)
    except (TypeError, ValueError):
        return "N/A"

comparison = {
    "purpose": "v1/v1, v1/v2, v2/v2 3-way 비교. 1차 스크리닝 경향 파악용. 최종 성능 결론 아님.",
    "note": "read-only 분석. 기존 score/evaluation/reports 미수정.",
    "caution": {
        "testset_diff": "v1/v1은 v1 전처리 기반(lesion_patch_total=216250), v1/v2·v2/v2는 v2 roi_0_0 기반(145715). v1/v1 recall은 직접 비교 불가.",
        "threshold_diff": "v1/v1·v1/v2는 v1 모델 기반 threshold(p95=14.377, p99=18.673), v2/v2는 v2 모델 기반(p95=14.092, p99=17.763). Dice/IoU/recall 비교 시 threshold 차이 영향 혼재.",
        "v2v2_dataset_profile": "reports_v2_roi0_0_lesion/lesion_hit_overlap_summary.json의 dataset_profile='v1_model_roi'는 메타데이터 오기입으로 추정. threshold=14.092(v2 기반)이며 이전 v1v2_vs_v2v2_comparison에서 v2/v2로 확인됨.",
    },
    "comparison_condition": {
        "v1v1": {"model": "v1 PaDiM", "testset": "v1 병변 308명", "threshold_source": "v1 normal val p95/p99"},
        "v1v2": {"model": "v1 PaDiM", "testset": "v2 roi_0_0 병변 308명", "threshold_source": "v1 normal val p95/p99"},
        "v2v2": {"model": "v2 PaDiM (roi_0_0)", "testset": "v2 roi_0_0 병변 308명", "threshold_source": "v2 normal val p95/p99"},
    },
    "A_fast_metrics": {k: {"v1v1": fm["v1v1"].get(k), "v1v2": fm["v1v2"].get(k), "v2v2": fm["v2v2"].get(k)} for k in [
        "patch_auroc","patch_auprc","slice_auroc","slice_auprc","patient_auroc",
        "p95_threshold","p95_patch_dice","p95_patch_iou",
        "p99_threshold","p99_patch_dice","p99_patch_iou",
        "lesion_patch_total","lesion_slice_total",
    ]},
    "B_screening": {k: {"v1v1": sc["v1v1"].get(k), "v1v2": sc["v1v2"].get(k), "v2v2": sc["v2v2"].get(k)} for k in [
        "p95_lesion_patch_recall","p95_lesion_slice_recall","p95_patient_coverage_mean","p95_patient_hit_rate",
        "p95_topk10_coverage","p95_topk30_coverage","p95_topk50_coverage",
        "p99_lesion_patch_recall","p99_lesion_slice_recall","p99_patient_coverage_mean","p99_patient_hit_rate",
    ]},
    "C_hit_overlap": {k: {"v1v1": ho["v1v1"].get(k), "v1v2": ho["v1v2"].get(k), "v2v2": ho["v2v2"].get(k)} for k in [
        "p95_threshold","patient_hit_rate","n_patient_no_hit","no_hit_patients",
        "micro_lesion_patch_recall","micro_lesion_slice_recall",
        "patient_patch_recall_mean","patient_patch_recall_median",
        "continuous_hit_ratio_mean","continuous_hit_ratio_median",
        "lowest_patch_recall_top10","most_missed_slice_top10",
    ]},
    "change_v1v1_to_v1v2": {
        "interpretation": "테스트셋/ROI 전처리 변화 영향(v1→v2 roi_0_0). 모델 동일.",
        "patch_auroc_diff":              diff(fm["v1v1"].get("patch_auroc"),              fm["v1v2"].get("patch_auroc")),
        "patch_auprc_diff":              diff(fm["v1v1"].get("patch_auprc"),              fm["v1v2"].get("patch_auprc")),
        "slice_auroc_diff":              diff(fm["v1v1"].get("slice_auroc"),              fm["v1v2"].get("slice_auroc")),
        "slice_auprc_diff":              diff(fm["v1v1"].get("slice_auprc"),              fm["v1v2"].get("slice_auprc")),
        "p95_patch_dice_diff":           diff(fm["v1v1"].get("p95_patch_dice"),           fm["v1v2"].get("p95_patch_dice")),
        "p95_lesion_patch_recall_diff":  diff(sc["v1v1"].get("p95_lesion_patch_recall"),  sc["v1v2"].get("p95_lesion_patch_recall")),
        "p95_lesion_slice_recall_diff":  diff(sc["v1v1"].get("p95_lesion_slice_recall"),  sc["v1v2"].get("p95_lesion_slice_recall")),
        "p95_patient_hit_rate_diff":     diff(sc["v1v1"].get("p95_patient_hit_rate"),     sc["v1v2"].get("p95_patient_hit_rate")),
        "p95_topk10_coverage_diff":      diff(sc["v1v1"].get("p95_topk10_coverage"),      sc["v1v2"].get("p95_topk10_coverage")),
        "p95_topk30_coverage_diff":      diff(sc["v1v1"].get("p95_topk30_coverage"),      sc["v1v2"].get("p95_topk30_coverage")),
    },
    "change_v1v2_to_v2v2": {
        "interpretation": "v2 정상 학습셋 + v2 PaDiM 학습 변화 영향. 테스트셋 동일.",
        "patch_auroc_diff":              diff(fm["v1v2"].get("patch_auroc"),              fm["v2v2"].get("patch_auroc")),
        "patch_auprc_diff":              diff(fm["v1v2"].get("patch_auprc"),              fm["v2v2"].get("patch_auprc")),
        "slice_auroc_diff":              diff(fm["v1v2"].get("slice_auroc"),              fm["v2v2"].get("slice_auroc")),
        "slice_auprc_diff":              diff(fm["v1v2"].get("slice_auprc"),              fm["v2v2"].get("slice_auprc")),
        "p95_patch_dice_diff":           diff(fm["v1v2"].get("p95_patch_dice"),           fm["v2v2"].get("p95_patch_dice")),
        "p95_lesion_patch_recall_diff":  diff(sc["v1v2"].get("p95_lesion_patch_recall"),  sc["v2v2"].get("p95_lesion_patch_recall")),
        "p95_lesion_slice_recall_diff":  diff(sc["v1v2"].get("p95_lesion_slice_recall"),  sc["v2v2"].get("p95_lesion_slice_recall")),
        "p95_patient_hit_rate_diff":     diff(sc["v1v2"].get("p95_patient_hit_rate"),     sc["v2v2"].get("p95_patient_hit_rate")),
        "p95_topk10_coverage_diff":      diff(sc["v1v2"].get("p95_topk10_coverage"),      sc["v2v2"].get("p95_topk10_coverage")),
        "p95_topk30_coverage_diff":      diff(sc["v1v2"].get("p95_topk30_coverage"),      sc["v2v2"].get("p95_topk30_coverage")),
        "patient_hit_rate_diff":         diff(ho["v1v2"].get("patient_hit_rate"),         ho["v2v2"].get("patient_hit_rate")),
        "n_no_hit_diff":                 diff(ho["v1v2"].get("n_patient_no_hit"),         ho["v2v2"].get("n_patient_no_hit")),
    },
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(comparison, f, ensure_ascii=False, indent=2)
print(f"JSON 저장: {json_path}")

# ── MD 출력 ──────────────────────────────────────────────────────────────────
md_path = OUT_DIR / "v1v1_v1v2_v2v2_three_way_comparison.md"

def pct(v):
    if v is None or v == "missing":
        return "missing"
    try:
        return f"{float(v)*100:.2f}%"
    except (TypeError, ValueError):
        return str(v)

md_lines = [
    "# v1/v1 · v1/v2 · v2/v2 3-way 비교",
    "",
    "> **목적**: 2.5D RD4AD 2차 모델 학습 전 1차 PaDiM 후보 생성 경향 파악. 최종 성능 결론 아님.",
    "> **주의**:",
    "> - v1/v1의 테스트셋은 v1 전처리 기반(lesion_patch_total=216,250), v1/v2·v2/v2는 v2 roi_0_0 기반(145,715). v1/v1 recall은 직접 수치 비교 불가.",
    "> - v1/v1·v1/v2는 v1 모델 기반 threshold(p95=14.377, p99=18.673), v2/v2는 v2 모델 기반(p95=14.092, p99=17.763). Dice/IoU/recall 비교 시 threshold 차이 혼재.",
    "> - v2/v2 hit overlap summary의 `dataset_profile='v1_model_roi'`는 메타데이터 오기입으로 추정. threshold=14.092(v2 기반)이며 이전 비교에서 v2/v2로 확인됨.",
    "",
    "---",
    "",
    "## A. fast metrics",
    "",
    "| 지표 | v1/v1 | v1/v2 | v2/v2 |",
    "|------|-------|-------|-------|",
    f"| patch AUROC | {fm['v1v1'].get('patch_auroc')} | {fm['v1v2'].get('patch_auroc')} | {fm['v2v2'].get('patch_auroc')} |",
    f"| patch AUPRC | {fm['v1v1'].get('patch_auprc')} | {fm['v1v2'].get('patch_auprc')} | {fm['v2v2'].get('patch_auprc')} |",
    f"| slice AUROC | {fm['v1v1'].get('slice_auroc')} | {fm['v1v2'].get('slice_auroc')} | {fm['v2v2'].get('slice_auroc')} |",
    f"| slice AUPRC | {fm['v1v1'].get('slice_auprc')} | {fm['v1v2'].get('slice_auprc')} | {fm['v2v2'].get('slice_auprc')} |",
    f"| patient AUROC | {fm['v1v1'].get('patient_auroc')} | {fm['v1v2'].get('patient_auroc')} | {fm['v2v2'].get('patient_auroc')} |",
    f"| p95 threshold | {fm['v1v1'].get('p95_threshold')} | {fm['v1v2'].get('p95_threshold')} | {fm['v2v2'].get('p95_threshold')} |",
    f"| p95 patch Dice | {fm['v1v1'].get('p95_patch_dice')} | {fm['v1v2'].get('p95_patch_dice')} | {fm['v2v2'].get('p95_patch_dice')} |",
    f"| p95 patch IoU | {fm['v1v1'].get('p95_patch_iou')} | {fm['v1v2'].get('p95_patch_iou')} | {fm['v2v2'].get('p95_patch_iou')} |",
    f"| p99 threshold | {fm['v1v1'].get('p99_threshold')} | {fm['v1v2'].get('p99_threshold')} | {fm['v2v2'].get('p99_threshold')} |",
    f"| p99 patch Dice | {fm['v1v1'].get('p99_patch_dice')} | {fm['v1v2'].get('p99_patch_dice')} | {fm['v2v2'].get('p99_patch_dice')} |",
    f"| p99 patch IoU | {fm['v1v1'].get('p99_patch_iou')} | {fm['v1v2'].get('p99_patch_iou')} | {fm['v2v2'].get('p99_patch_iou')} |",
    f"| lesion_patch_total | {fm['v1v1'].get('lesion_patch_total')} | {fm['v1v2'].get('lesion_patch_total')} | {fm['v2v2'].get('lesion_patch_total')} |",
    f"| lesion_slice_total | {fm['v1v1'].get('lesion_slice_total')} | {fm['v1v2'].get('lesion_slice_total')} | {fm['v2v2'].get('lesion_slice_total')} |",
    "",
    "---",
    "",
    "## B. screening analysis",
    "",
    "| 지표 | v1/v1 | v1/v2 | v2/v2 |",
    "|------|-------|-------|-------|",
    f"| p95 lesion_patch_recall | {pct(sc['v1v1'].get('p95_lesion_patch_recall'))} | {pct(sc['v1v2'].get('p95_lesion_patch_recall'))} | {pct(sc['v2v2'].get('p95_lesion_patch_recall'))} |",
    f"| p95 lesion_slice_recall | {pct(sc['v1v1'].get('p95_lesion_slice_recall'))} | {pct(sc['v1v2'].get('p95_lesion_slice_recall'))} | {pct(sc['v2v2'].get('p95_lesion_slice_recall'))} |",
    f"| p95 patient_coverage_mean | {pct(sc['v1v1'].get('p95_patient_coverage_mean'))} | {pct(sc['v1v2'].get('p95_patient_coverage_mean'))} | {pct(sc['v2v2'].get('p95_patient_coverage_mean'))} |",
    f"| p95 patient_hit_rate | {pct(sc['v1v1'].get('p95_patient_hit_rate'))} | {pct(sc['v1v2'].get('p95_patient_hit_rate'))} | {pct(sc['v2v2'].get('p95_patient_hit_rate'))} |",
    f"| p95 top-10 coverage | {pct(sc['v1v1'].get('p95_topk10_coverage'))} | {pct(sc['v1v2'].get('p95_topk10_coverage'))} | {pct(sc['v2v2'].get('p95_topk10_coverage'))} |",
    f"| p95 top-30 coverage | {pct(sc['v1v1'].get('p95_topk30_coverage'))} | {pct(sc['v1v2'].get('p95_topk30_coverage'))} | {pct(sc['v2v2'].get('p95_topk30_coverage'))} |",
    f"| p95 top-50 coverage | {pct(sc['v1v1'].get('p95_topk50_coverage'))} | {pct(sc['v1v2'].get('p95_topk50_coverage'))} | {pct(sc['v2v2'].get('p95_topk50_coverage'))} |",
    f"| p99 lesion_patch_recall | {pct(sc['v1v1'].get('p99_lesion_patch_recall'))} | {pct(sc['v1v2'].get('p99_lesion_patch_recall'))} | {pct(sc['v2v2'].get('p99_lesion_patch_recall'))} |",
    f"| p99 lesion_slice_recall | {pct(sc['v1v1'].get('p99_lesion_slice_recall'))} | {pct(sc['v1v2'].get('p99_lesion_slice_recall'))} | {pct(sc['v2v2'].get('p99_lesion_slice_recall'))} |",
    f"| p99 patient_coverage_mean | {pct(sc['v1v1'].get('p99_patient_coverage_mean'))} | {pct(sc['v1v2'].get('p99_patient_coverage_mean'))} | {pct(sc['v2v2'].get('p99_patient_coverage_mean'))} |",
    f"| p99 patient_hit_rate | {pct(sc['v1v1'].get('p99_patient_hit_rate'))} | {pct(sc['v1v2'].get('p99_patient_hit_rate'))} | {pct(sc['v2v2'].get('p99_patient_hit_rate'))} |",
    "",
    "---",
    "",
    "## C. hit overlap (p95 기준)",
    "",
    "| 지표 | v1/v1 | v1/v2 | v2/v2 |",
    "|------|-------|-------|-------|",
    f"| p95 threshold | {ho['v1v1'].get('p95_threshold')} | {ho['v1v2'].get('p95_threshold')} | {ho['v2v2'].get('p95_threshold')} |",
    f"| patient_hit_rate | {pct(ho['v1v1'].get('patient_hit_rate'))} | {pct(ho['v1v2'].get('patient_hit_rate'))} | {pct(ho['v2v2'].get('patient_hit_rate'))} |",
    f"| n_patient_no_hit | {ho['v1v1'].get('n_patient_no_hit')} | {ho['v1v2'].get('n_patient_no_hit')} | {ho['v2v2'].get('n_patient_no_hit')} |",
    f"| no_hit_patients | {ho['v1v1'].get('no_hit_patients')} | {ho['v1v2'].get('no_hit_patients')} | {ho['v2v2'].get('no_hit_patients')} |",
    f"| micro_lesion_patch_recall | {pct(ho['v1v1'].get('micro_lesion_patch_recall'))} | {pct(ho['v1v2'].get('micro_lesion_patch_recall'))} | {pct(ho['v2v2'].get('micro_lesion_patch_recall'))} |",
    f"| micro_lesion_slice_recall | {pct(ho['v1v1'].get('micro_lesion_slice_recall'))} | {pct(ho['v1v2'].get('micro_lesion_slice_recall'))} | {pct(ho['v2v2'].get('micro_lesion_slice_recall'))} |",
    f"| patient_patch_recall_mean | {pct(ho['v1v1'].get('patient_patch_recall_mean'))} | {pct(ho['v1v2'].get('patient_patch_recall_mean'))} | {pct(ho['v2v2'].get('patient_patch_recall_mean'))} |",
    f"| patient_patch_recall_median | {pct(ho['v1v1'].get('patient_patch_recall_median'))} | {pct(ho['v1v2'].get('patient_patch_recall_median'))} | {pct(ho['v2v2'].get('patient_patch_recall_median'))} |",
    f"| continuous_hit_ratio_mean | {pct(ho['v1v1'].get('continuous_hit_ratio_mean'))} | {pct(ho['v1v2'].get('continuous_hit_ratio_mean'))} | {pct(ho['v2v2'].get('continuous_hit_ratio_mean'))} |",
    f"| continuous_hit_ratio_median | {pct(ho['v1v1'].get('continuous_hit_ratio_median'))} | {pct(ho['v1v2'].get('continuous_hit_ratio_median'))} | {pct(ho['v2v2'].get('continuous_hit_ratio_median'))} |",
    "",
    "### lowest patch recall top10",
    "",
    "**v1/v1**",
    ho["v1v1"].get("lowest_patch_recall_top10", "missing"),
    "",
    "**v1/v2**",
    ho["v1v2"].get("lowest_patch_recall_top10", "missing"),
    "",
    "**v2/v2**",
    ho["v2v2"].get("lowest_patch_recall_top10", "missing"),
    "",
    "### most missed lesion slice top10",
    "",
    "**v1/v1**",
    ho["v1v1"].get("most_missed_slice_top10", "missing"),
    "",
    "**v1/v2**",
    ho["v1v2"].get("most_missed_slice_top10", "missing"),
    "",
    "**v2/v2**",
    ho["v2v2"].get("most_missed_slice_top10", "missing"),
    "",
    "---",
    "",
    "## 변화 요약",
    "",
    "### v1/v1 → v1/v2 (테스트셋/ROI 전처리 변화, 모델 동일)",
    "",
    f"- patch AUROC: {fm['v1v1'].get('patch_auroc')} → {fm['v1v2'].get('patch_auroc')} (Δ{diff(fm['v1v1'].get('patch_auroc'), fm['v1v2'].get('patch_auroc'))})",
    f"- slice AUROC: {fm['v1v1'].get('slice_auroc')} → {fm['v1v2'].get('slice_auroc')} (Δ{diff(fm['v1v1'].get('slice_auroc'), fm['v1v2'].get('slice_auroc'))})",
    f"- p95 lesion_patch_recall: {pct(sc['v1v1'].get('p95_lesion_patch_recall'))} → {pct(sc['v1v2'].get('p95_lesion_patch_recall'))} (**테스트셋 크기 다름, 직접 비교 주의**)",
    f"- p95 lesion_slice_recall: {pct(sc['v1v1'].get('p95_lesion_slice_recall'))} → {pct(sc['v1v2'].get('p95_lesion_slice_recall'))} (**테스트셋 크기 다름**)",
    f"- p95 patient_hit_rate: {pct(sc['v1v1'].get('p95_patient_hit_rate'))} → {pct(sc['v1v2'].get('p95_patient_hit_rate'))}",
    f"- p95 top-10 coverage: {pct(sc['v1v1'].get('p95_topk10_coverage'))} → {pct(sc['v1v2'].get('p95_topk10_coverage'))} (Δ{diff(sc['v1v1'].get('p95_topk10_coverage'), sc['v1v2'].get('p95_topk10_coverage'))})",
    f"- p95 top-30 coverage: {pct(sc['v1v1'].get('p95_topk30_coverage'))} → {pct(sc['v1v2'].get('p95_topk30_coverage'))} (Δ{diff(sc['v1v1'].get('p95_topk30_coverage'), sc['v1v2'].get('p95_topk30_coverage'))})",
    "",
    "### v1/v2 → v2/v2 (v2 정상 학습셋 + v2 PaDiM 학습 변화, 테스트셋 동일)",
    "",
    f"- patch AUROC: {fm['v1v2'].get('patch_auroc')} → {fm['v2v2'].get('patch_auroc')} (Δ{diff(fm['v1v2'].get('patch_auroc'), fm['v2v2'].get('patch_auroc'))})",
    f"- slice AUROC: {fm['v1v2'].get('slice_auroc')} → {fm['v2v2'].get('slice_auroc')} (Δ{diff(fm['v1v2'].get('slice_auroc'), fm['v2v2'].get('slice_auroc'))})",
    f"- p95 lesion_patch_recall: {pct(sc['v1v2'].get('p95_lesion_patch_recall'))} → {pct(sc['v2v2'].get('p95_lesion_patch_recall'))} (Δ{diff(sc['v1v2'].get('p95_lesion_patch_recall'), sc['v2v2'].get('p95_lesion_patch_recall'))})",
    f"- p95 lesion_slice_recall: {pct(sc['v1v2'].get('p95_lesion_slice_recall'))} → {pct(sc['v2v2'].get('p95_lesion_slice_recall'))} (Δ{diff(sc['v1v2'].get('p95_lesion_slice_recall'), sc['v2v2'].get('p95_lesion_slice_recall'))})",
    f"- p95 patient_hit_rate: {pct(sc['v1v2'].get('p95_patient_hit_rate'))} → {pct(sc['v2v2'].get('p95_patient_hit_rate'))} (Δ{diff(sc['v1v2'].get('p95_patient_hit_rate'), sc['v2v2'].get('p95_patient_hit_rate'))})",
    f"- p95 top-10 coverage: {pct(sc['v1v2'].get('p95_topk10_coverage'))} → {pct(sc['v2v2'].get('p95_topk10_coverage'))} (Δ{diff(sc['v1v2'].get('p95_topk10_coverage'), sc['v2v2'].get('p95_topk10_coverage'))})",
    f"- p95 top-30 coverage: {pct(sc['v1v2'].get('p95_topk30_coverage'))} → {pct(sc['v2v2'].get('p95_topk30_coverage'))} (Δ{diff(sc['v1v2'].get('p95_topk30_coverage'), sc['v2v2'].get('p95_topk30_coverage'))})",
    f"- no-hit 환자 수: {ho['v1v2'].get('n_patient_no_hit')}명 → {ho['v2v2'].get('n_patient_no_hit')}명",
    "",
    "---",
    "",
    "## 가장 좋은 지표 / 나빠진 지표",
    "",
    "### v1/v2 기준 (v1/v1은 테스트셋 달라 제외)",
    "",
    "**v1/v2가 v2/v2보다 좋은 지표**",
    f"- patch AUROC: {fm['v1v2'].get('patch_auroc')} > {fm['v2v2'].get('patch_auroc')}",
    f"- patch AUPRC: {fm['v1v2'].get('patch_auprc')} > {fm['v2v2'].get('patch_auprc')}",
    f"- slice AUROC: {fm['v1v2'].get('slice_auroc')} > {fm['v2v2'].get('slice_auroc')}",
    f"- slice AUPRC: {fm['v1v2'].get('slice_auprc')} > {fm['v2v2'].get('slice_auprc')}",
    f"- p95 patch Dice/IoU: {fm['v1v2'].get('p95_patch_dice')}/{fm['v1v2'].get('p95_patch_iou')} > {fm['v2v2'].get('p95_patch_dice')}/{fm['v2v2'].get('p95_patch_iou')}",
    f"- p95 top-k coverage 전반 (top10: {pct(sc['v1v2'].get('p95_topk10_coverage'))} vs {pct(sc['v2v2'].get('p95_topk10_coverage'))})",
    "",
    "**v2/v2가 v1/v2보다 좋은 지표**",
    f"- p95 patient_hit_rate: {pct(sc['v1v2'].get('p95_patient_hit_rate'))} → {pct(sc['v2v2'].get('p95_patient_hit_rate'))} (no-hit 2명→1명)",
    f"- p99 lesion_slice_recall: {pct(sc['v1v2'].get('p99_lesion_slice_recall'))} → {pct(sc['v2v2'].get('p99_lesion_slice_recall'))}",
    f"- p99 patient_coverage_mean: {pct(sc['v1v2'].get('p99_patient_coverage_mean'))} → {pct(sc['v2v2'].get('p99_patient_coverage_mean'))}",
    "",
    "---",
    "",
    "## 다음 단계 판정",
    "",
    "**2.5D RD4AD 학습 설계로 넘어가도 되는지**: 판정 유보 (사용자 확인 필요)",
    "",
    "- 3-way 비교 완료. 기존 결과 미수정 확인.",
    "- v1/v2가 patch/slice 정밀 지표와 top-k coverage에서 v2/v2 대비 전반적으로 높음.",
    "- v2/v2는 patient hit rate(no-hit 감소) 및 p99 slice recall에서 소폭 우세.",
    "- 어느 조합을 2차 모델 입력으로 쓸지는 사용자 판단이 필요함.",
    "- stage2_holdout 최종 성능과 섞어 해석하지 않음.",
    "",
    "> **생성 일자**: 2026-05-24",
    "> **분석 방법**: read-only JSON 로드, 재실행 없음",
]

with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"MD 저장: {md_path}")
print("완료.")
