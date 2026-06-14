"""
topn_analysis_sample.py
(current_review.md 승인 기준으로 실행)

기존 ratio_adjusted_score_sample_diagnostic.csv를 read-only로 사용해
top-N 기준 분석 + 완화 variant 비교를 수행한다.

- full-run 금지
- 기존 sample 파일 덮어쓰기 금지
- scoring/학습/crop/npz 재실행 금지
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = REPO_ROOT / "outputs/position-aware-padim-v1/reports_v2_roi0_0_lesion/ratio_adjusted_score_sample_diagnostic.csv"
OUT_DIR = REPO_ROOT / "outputs/position-aware-padim-v1/reports_v2_roi0_0_lesion"

OUT_FILES = {
    "csv": OUT_DIR / "ratio_adjusted_score_sample_topn_diagnostic.csv",
    "json": OUT_DIR / "ratio_adjusted_score_sample_topn_diagnostic.json",
    "md": OUT_DIR / "ratio_adjusted_score_sample_topn_diagnostic.md",
}

THRESHOLDS = {
    "v1v2": {"p95": 14.377350028011772, "p99": 18.672782302362954},
    "v2v2": {"p95": 14.092057666455288, "p99": 17.763281310708145},
}

WEAK_PATIENTS = ["LUNG1-156", "LUNG1-415", "MSD_lung_071", "MSD_lung_096", "MSD_lung_079"]

BASE_VARIANTS = [
    "score_original",
    "score_roi_weighted",
    "score_valid950_weighted",
    "score_valid970_weighted",
]
SOFT_VARIANTS = [
    "score_valid950_pow025",
    "score_valid950_floor025",
    "score_valid950_soft",
]
ALL_VARIANTS = BASE_VARIANTS + SOFT_VARIANTS


def check_conflicts() -> None:
    for k, f in OUT_FILES.items():
        if f.exists():
            print(f"[중단] 이미 존재: {f}")
            print("  삭제 후 재실행하세요.")
            sys.exit(1)


def is_lesion(df: pd.DataFrame) -> pd.Series:
    if "patch_label" in df.columns:
        return df["patch_label"] == 1
    return df["lesion_patch_ratio"] > 0


def add_soft_variants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    v = df["valid_ratio_roi_air950"].clip(0, 1)
    df["score_valid950_pow025"] = df["padim_score"] * (v ** 0.25)
    df["score_valid950_floor025"] = df["padim_score"] * np.sqrt(v.clip(lower=0.25))
    df["score_valid950_soft"] = df["padim_score"] * (0.7 + 0.3 * np.sqrt(v))
    return df


def compute_topn_metrics(df: pd.DataFrame, score_col: str, n_topn: int,
                         model_type: str, thr_label: str) -> dict:
    lesion_flag = is_lesion(df)
    if n_topn > 0:
        cutoff = df[score_col].nlargest(n_topn).min()
        selected = df[score_col] >= cutoff
    else:
        selected = pd.Series(False, index=df.index)

    n_total = len(df)
    n_selected = int(selected.sum())
    n_lesion_total = int(lesion_flag.sum())
    n_sel_lesion = int((selected & lesion_flag).sum())
    n_fp = n_selected - n_sel_lesion

    # lesion slice recall
    lesion_slices = set(df[lesion_flag][["patient_id", "local_z"]].apply(tuple, axis=1))
    selected_slices = set(df[selected][["patient_id", "local_z"]].apply(tuple, axis=1))
    hit_lesion_slices = lesion_slices & selected_slices
    slice_recall = len(hit_lesion_slices) / len(lesion_slices) if lesion_slices else 0.0

    # patient hit rate
    lesion_patients = df[lesion_flag]["patient_id"].unique()
    hit_patients = df[selected & lesion_flag]["patient_id"].unique()
    no_hit = [p for p in lesion_patients if p not in hit_patients]
    patient_hit_rate = len(hit_patients) / len(lesion_patients) if len(lesion_patients) > 0 else 0.0

    # patient top-k
    patient_topk = {}
    for k in [10, 30, 50, 100, 500]:
        n_hit = 0
        n_with_lesion = 0
        for pid in lesion_patients:
            sub_p = df[df["patient_id"] == pid]
            lf = is_lesion(sub_p)
            if lf.sum() == 0:
                continue
            n_with_lesion += 1
            ranked = sub_p[score_col].rank(ascending=False, method="first")
            if ranked[lf].min() <= k:
                n_hit += 1
        patient_topk[f"patient_top{k}_hit_rate"] = n_hit / n_with_lesion if n_with_lesion > 0 else 0.0

    return {
        "model_type": model_type,
        "score_col": score_col,
        "topn_label": thr_label,
        "n_topn": n_topn,
        "n_total": n_total,
        "n_selected": n_selected,
        "n_lesion_total": n_lesion_total,
        "n_sel_lesion": n_sel_lesion,
        "n_fp": n_fp,
        "positive_ratio": n_sel_lesion / n_selected if n_selected > 0 else 0.0,
        "fp_ratio": n_fp / n_selected if n_selected > 0 else 0.0,
        "lesion_patch_recall": n_sel_lesion / n_lesion_total if n_lesion_total > 0 else 0.0,
        "lesion_slice_recall": slice_recall,
        "patient_hit_rate": patient_hit_rate,
        "n_nohit": len(no_hit),
        "nohit_patients": no_hit,
        **patient_topk,
    }


def compute_stratified_topn(df: pd.DataFrame, score_col: str, n_topn: int,
                             strat_col: str) -> list[dict]:
    records = []
    if strat_col not in df.columns:
        return records
    for gval in sorted(df[strat_col].dropna().unique()):
        sub = df[df[strat_col] == gval]
        if len(sub) == 0:
            continue
        lesion_flag = is_lesion(sub)
        if n_topn > 0:
            cutoff = sub[score_col].nlargest(n_topn).min()
            selected = sub[score_col] >= cutoff
        else:
            selected = pd.Series(False, index=sub.index)
        n_total = len(sub)
        n_selected = int(selected.sum())
        n_lesion = int(lesion_flag.sum())
        n_sel_les = int((selected & lesion_flag).sum())
        n_fp = n_selected - n_sel_les
        records.append({
            "strat_col": strat_col,
            "group": str(gval),
            "score_col": score_col,
            "n_topn": n_topn,
            "n_total": n_total,
            "n_selected": n_selected,
            "n_lesion": n_lesion,
            "n_sel_lesion": n_sel_les,
            "n_fp": n_fp,
            "positive_ratio": n_sel_les / n_selected if n_selected > 0 else 0.0,
            "fp_ratio": n_fp / n_selected if n_selected > 0 else 0.0,
            "lesion_recall": n_sel_les / n_lesion if n_lesion > 0 else 0.0,
        })
    return records


def compute_weak_topn(df: pd.DataFrame, score_col: str, orig_n95: int,
                      model_type: str) -> list[dict]:
    records = []
    sub_model = df[df["model_type"] == model_type]
    for pid in WEAK_PATIENTS:
        sub_p = sub_model[sub_model["patient_id"] == pid]
        if sub_p.empty:
            continue
        lf = is_lesion(sub_p)
        ranked = sub_p[score_col].rank(ascending=False, method="first")
        lesion_ranks = ranked[lf]
        orig_ranked = sub_p["score_original"].rank(ascending=False, method="first")
        orig_lesion_ranks = orig_ranked[lf]
        best_rank = int(lesion_ranks.min()) if len(lesion_ranks) > 0 else None
        orig_best = int(orig_lesion_ranks.min()) if len(orig_lesion_ranks) > 0 else None
        records.append({
            "model_type": model_type,
            "patient_id": pid,
            "score_col": score_col,
            "lesion_best_rank": best_rank,
            "orig_best_rank": orig_best,
            "rank_change": (best_rank - orig_best) if (best_rank and orig_best) else None,
            "lesion_in_top_n95": int((lesion_ranks <= orig_n95).sum()) if len(lesion_ranks) > 0 else 0,
            "n_lesion_patch": int(lf.sum()),
        })
    return records


def build_md(all_metrics: list[dict], strat_records: list[dict],
             weak_records: list[dict]) -> str:
    lines = ["# Top-N 기준 Score Variant 비교 [SAMPLE]\n"]
    lines.append("> fixed threshold 비교 보완: original p95/p99 선택 개수 N과 동일한 개수를 각 variant에서 선택해 공정 비교\n")

    # 전체 지표 테이블
    lines.append("## 전체 지표 (model별 topN)\n")
    lines.append("| model | score_col | topN | N | pos_ratio | fp_ratio | les_recall | slice_recall | hit_rate | nohit | top50_hit |")
    lines.append("|-------|-----------|------|---|-----------|---------|-----------|-------------|---------|-------|-----------|")
    for m in all_metrics:
        lines.append(
            f"| {m['model_type']} | {m['score_col']} | {m['topn_label']} "
            f"| {m['n_selected']:,} | {m['positive_ratio']:.4f} | {m['fp_ratio']:.4f} "
            f"| {m['lesion_patch_recall']:.4f} | {m['lesion_slice_recall']:.4f} "
            f"| {m['patient_hit_rate']:.4f} | {m['n_nohit']} | {m.get('patient_top50_hit_rate', 0):.4f} |"
        )

    # patient top-k
    lines.append("\n## 환자별 Top-k Hit Rate\n")
    lines.append("| model | score_col | topN | top10 | top30 | top50 | top100 | top500 |")
    lines.append("|-------|-----------|------|-------|-------|-------|--------|--------|")
    for m in all_metrics:
        lines.append(
            f"| {m['model_type']} | {m['score_col']} | {m['topn_label']} "
            f"| {m.get('patient_top10_hit_rate',0):.4f} | {m.get('patient_top30_hit_rate',0):.4f} "
            f"| {m.get('patient_top50_hit_rate',0):.4f} | {m.get('patient_top100_hit_rate',0):.4f} "
            f"| {m.get('patient_top500_hit_rate',0):.4f} |"
        )

    # Stratified (중요 그룹만 요약)
    if strat_records:
        lines.append("\n## Stratified 결과 요약\n")
        lines.append("| strat_col | group | score_col | N | fp_ratio | les_recall |")
        lines.append("|-----------|-------|-----------|---|---------|-----------|")
        for r in strat_records[:80]:
            lines.append(
                f"| {r['strat_col']} | {r['group']} | {r['score_col']} "
                f"| {r['n_selected']:,} | {r['fp_ratio']:.4f} | {r['lesion_recall']:.4f} |"
            )

    # Weak patient
    lines.append("\n## Weak Patient Rank 변화\n")
    lines.append("| model | patient | score_col | best_rank | orig_rank | rank_change | in_topN95 |")
    lines.append("|-------|---------|-----------|-----------|-----------|-------------|-----------|")
    for w in weak_records:
        rc = w['rank_change']
        rc_str = f"{rc:+d}" if rc is not None else "N/A"
        lines.append(
            f"| {w['model_type']} | {w['patient_id']} | {w['score_col']} "
            f"| {w['lesion_best_rank']} | {w['orig_best_rank']} | {rc_str} | {w['lesion_in_top_n95']} |"
        )

    lines.append("\n## 해석 기준\n")
    lines.append("- adjusted topN에서 FP 비율 감소, lesion recall/patient hit 유지 → 보조 ranking으로 가치 있음")
    lines.append("- lesion recall 감소 또는 weak 환자 rank 악화 → 후보 삭제용 사용 금지")
    lines.append("- pow025/floor025/soft 중 가장 안전한 variant를 full-run 전 판단")

    return "\n".join(lines)


def main():
    check_conflicts()

    print(f"=== 기존 sample CSV 로드 ===")
    df = pd.read_csv(SAMPLE_CSV, encoding="utf-8-sig")
    print(f"  row: {len(df):,}, patient: {df['patient_id'].nunique()}, model: {df['model_type'].unique()}")

    print("\n=== 완화 variant 계산 ===")
    df = add_soft_variants(df)
    print(f"  추가된 variant: {SOFT_VARIANTS}")

    all_metrics = []
    strat_all = []
    weak_all = []

    for model_type in df["model_type"].unique():
        sub = df[df["model_type"] == model_type].copy()
        thr = THRESHOLDS[model_type]

        # original p95/p99 선택 개수 계산
        n95 = int((sub["score_original"] >= thr["p95"]).sum())
        n99 = int((sub["score_original"] >= thr["p99"]).sum())
        print(f"\n[{model_type}] original p95 N={n95:,}, p99 N={n99:,}")

        for sv in ALL_VARIANTS:
            for thr_label, n_topn in [("topN_p95", n95), ("topN_p99", n99)]:
                m = compute_topn_metrics(sub, sv, n_topn, model_type, thr_label)
                all_metrics.append(m)

        # stratified (v1v2만 대표로, central_peripheral / position_bin / valid_ratio group / air_ratio group)
        if model_type == "v1v2":
            def valid_bin(s):
                return pd.cut(s, bins=[-0.001, 0.05, 0.3, 1.001],
                              labels=["valid_low", "valid_mid", "valid_high"]).astype(str)
            def air_bin(s):
                return pd.cut(s, bins=[-0.001, 0.3, 0.7, 1.001],
                              labels=["air_low", "air_mid", "air_high"]).astype(str)

            sub = sub.copy()
            sub["valid_bin_950"] = valid_bin(sub["valid_ratio_roi_air950"])
            sub["air_bin_950"] = air_bin(sub["air_ratio_950"])

            for sv in ALL_VARIANTS:
                for sc in ["central_peripheral", "position_bin", "valid_bin_950", "air_bin_950"]:
                    recs = compute_stratified_topn(sub, sv, n95, sc)
                    for r in recs:
                        r["model_type"] = model_type
                    strat_all.extend(recs)

        # weak patient
        for sv in ALL_VARIANTS:
            weak_all.extend(compute_weak_topn(sub, sv, n95, model_type))

    print(f"\n=== 결과 ===")
    print(f"  지표 조합 수: {len(all_metrics)}")
    print(f"  stratified 행 수: {len(strat_all)}")
    print(f"  weak patient 행 수: {len(weak_all)}")

    # 저장
    print("\n=== 출력 파일 저장 ===")
    pd.DataFrame(all_metrics).to_csv(OUT_FILES["csv"], index=False)
    print(f"  저장: {OUT_FILES['csv']}")

    def safe_json(obj):
        if isinstance(obj, dict):
            return {k: safe_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_json(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj

    with open(OUT_FILES["json"], "w", encoding="utf-8") as f:
        json.dump({
            "all_metrics": safe_json(all_metrics),
            "stratified": safe_json(strat_all),
            "weak_patients": safe_json(weak_all),
        }, f, ensure_ascii=False, indent=2)
    print(f"  저장: {OUT_FILES['json']}")

    md_text = build_md(all_metrics, strat_all, weak_all)
    OUT_FILES["md"].write_text(md_text, encoding="utf-8")
    print(f"  저장: {OUT_FILES['md']}")

    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
