"""
generate_rule_b_candidates.py
Rule B: p95 + diverse fallback candidate manifest-only dry-run
stage1_dev 154명 대상, 실제 crop/npy/PNG 생성 없음
후보 선택 기준: padim_score, slice_index, position_bin, z_level (label-free)
patch_label, lesion_overlap은 평가 지표 계산에만 사용
grid_position_bin 컬럼 없음 -> position_bin(6종) + z_level 조합으로 다양성 구현
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

RULE_A_MANIFEST  = REPO / "outputs/second-stage-lesion-refiner-v1/candidates/rule_a_p95_stage1_dev_candidate_manifest_dryrun.csv"
STAGE_SPLIT      = REPO / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
THRESHOLD_JSON   = REPO / "outputs/position-aware-padim-v1/evaluation/normal_v2_roi0_0/normal_v2_threshold.json"
SCREENING_CSV    = REPO / "outputs/position-aware-padim-v1/evaluation/lesion_subset_v2_model_v2/per_patient_screening.csv"
RULE_A_SUMMARY   = REPO / "outputs/second-stage-lesion-refiner-v1/reports/rule_a_p95_stage1_dev_candidate_summary.json"

OUT_MANIFEST     = REPO / "outputs/second-stage-lesion-refiner-v1/candidates/rule_b_stage1_dev_candidate_manifest_dryrun.csv"
OUT_SUMMARY_CSV  = REPO / "outputs/second-stage-lesion-refiner-v1/reports/rule_b_stage1_dev_candidate_summary.csv"
OUT_SUMMARY_JSON = REPO / "outputs/second-stage-lesion-refiner-v1/reports/rule_b_stage1_dev_candidate_summary.json"
OUT_SUMMARY_MD   = REPO / "outputs/second-stage-lesion-refiner-v1/reports/rule_b_stage1_dev_candidate_summary.md"

REQUIRED_COLS = [
    "patient_id", "stage_split", "padim_score", "slice_index",
    "position_bin", "z_level", "patch_label", "lesion_overlap", "group",
    "y0", "x0", "y1", "x1",
]


# ── guard ─────────────────────────────────────────────────────────────────────

def guard_output_files_absent():
    for f in [OUT_MANIFEST, OUT_SUMMARY_CSV, OUT_SUMMARY_JSON, OUT_SUMMARY_MD]:
        if f.exists():
            sys.exit(f"[GUARD] 출력 파일이 이미 존재합니다: {f}")

def guard_no_holdout(df: pd.DataFrame):
    if (df["stage_split"] == "stage2_holdout").any():
        sys.exit("[GUARD] stage2_holdout 데이터가 포함되어 있습니다. 중단.")

def guard_required_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"[GUARD] 필수 컬럼 없음: {missing}")

def guard_no_nan_inf(df: pd.DataFrame):
    if df["padim_score"].isna().any():
        sys.exit("[GUARD] padim_score에 NaN이 있습니다.")
    if np.isinf(df["padim_score"].values).any():
        sys.exit("[GUARD] padim_score에 Inf가 있습니다.")

def guard_stage1_dev_count(stage1_ids: list):
    if len(stage1_ids) != 154:
        sys.exit(f"[GUARD] stage1_dev 환자 수가 154명이 아닙니다: {len(stage1_ids)}")


# ── variant helpers ───────────────────────────────────────────────────────────
# 후보 선택에는 padim_score, slice_index, position_bin, z_level만 사용
# patch_label, lesion_overlap 컬럼은 이 함수들에서 절대 사용하지 않음

def _top_slices_top_patches(patient_df: pd.DataFrame, n_slices: int, n_patches: int) -> pd.DataFrame:
    """환자별로 slice_score(max padim) 기준 top n_slices, 각 slice에서 top n_patches 선택."""
    slice_max = patient_df.groupby("slice_index")["padim_score"].transform("max")
    patient_df = patient_df.copy()
    patient_df["_slice_score"] = slice_max

    top_slice_scores = (
        patient_df.groupby("slice_index")["padim_score"]
        .max()
        .nlargest(n_slices)
    )
    top_slice_idx = set(top_slice_scores.index)

    filtered = patient_df[patient_df["slice_index"].isin(top_slice_idx)].copy()
    filtered = filtered.sort_values("padim_score", ascending=False)
    selected = filtered.groupby("slice_index", sort=False).head(n_patches)
    selected = selected.drop(columns=["_slice_score"])
    return selected.reset_index(drop=True)


def apply_b1(df: pd.DataFrame) -> pd.DataFrame:
    """B1: 환자별 top30 slice × top5 patch. 최대 ~150/환자"""
    parts = []
    for pid, pdf in df.groupby("patient_id", sort=False):
        sel = _top_slices_top_patches(pdf, n_slices=30, n_patches=5)
        sel["rule_b_variant"] = "B1_slice_top30_patch5"
        parts.append(sel)
    return pd.concat(parts, ignore_index=True)


def apply_b2(df: pd.DataFrame) -> pd.DataFrame:
    """B2: 환자별 top50 slice × top5 patch. 최대 ~250/환자"""
    parts = []
    for pid, pdf in df.groupby("patient_id", sort=False):
        sel = _top_slices_top_patches(pdf, n_slices=50, n_patches=5)
        sel["rule_b_variant"] = "B2_slice_top50_patch5"
        parts.append(sel)
    return pd.concat(parts, ignore_index=True)


def apply_b3(df: pd.DataFrame) -> pd.DataFrame:
    """B3: 환자별 top50 slice × top10 patch. 최대 ~500/환자"""
    parts = []
    for pid, pdf in df.groupby("patient_id", sort=False):
        sel = _top_slices_top_patches(pdf, n_slices=50, n_patches=10)
        sel["rule_b_variant"] = "B3_slice_top50_patch10"
        parts.append(sel)
    return pd.concat(parts, ignore_index=True)


def apply_b4(df: pd.DataFrame) -> pd.DataFrame:
    """B4: 환자별 top200 diverse
    position_bin(6종)별 균등 배분 -> 부족분 전체 top 보충
    label 사용 금지
    """
    CAP = 200
    parts = []
    all_bins = df["position_bin"].unique()
    n_bins = len(all_bins)
    per_bin = max(1, CAP // n_bins)

    for pid, pdf in df.groupby("patient_id", sort=False):
        pdf_sorted = pdf.sort_values("padim_score", ascending=False)
        picked_idx: set = set()

        for pb in all_bins:
            bin_df = pdf_sorted[pdf_sorted["position_bin"] == pb]
            picked_idx.update(bin_df.head(per_bin).index.tolist())

        if len(picked_idx) < CAP:
            remaining = pdf_sorted[~pdf_sorted.index.isin(picked_idx)]
            fill = remaining.head(CAP - len(picked_idx))
            picked_idx.update(fill.index.tolist())

        sel = pdf[pdf.index.isin(picked_idx)].copy()
        if len(sel) > CAP:
            sel = sel.nlargest(CAP, "padim_score")
        sel["rule_b_variant"] = "B4_diverse_top200"
        parts.append(sel)
    return pd.concat(parts, ignore_index=True)


def apply_b5(df: pd.DataFrame) -> pd.DataFrame:
    """B5: p95 + diverse fallback
    1단계: top50 slice × top5 patch (pdf.index 보존)
    2단계: z_level별 MIN_PER_Z 미달 시 해당 z_level에서 추가
    환자당 max 500. label 사용 금지
    _top_slices_top_patches는 reset_index를 하므로 B5에서는 인라인 구현으로 pdf.index 직접 보존
    """
    MAX_CAP = 500
    Z_LEVELS = ["upper", "middle", "lower"]
    MIN_PER_Z = 10

    parts = []
    for pid, pdf in df.groupby("patient_id", sort=False):
        # 1단계: top50 slice × top5 patch (원본 pdf.index 보존)
        slice_max = pdf.groupby("slice_index")["padim_score"].max()
        top_slices = set(slice_max.nlargest(50).index)
        filtered = pdf[pdf["slice_index"].isin(top_slices)].sort_values("padim_score", ascending=False)
        base = filtered.groupby("slice_index", sort=False).head(5)  # pdf.index 공간 유지
        picked_idx: set = set(base.index.tolist())

        # 2단계: z_level 다양성 보완 (pdf.index 기준)
        z_counts = base["z_level"].value_counts()
        for z in Z_LEVELS:
            cur = z_counts.get(z, 0)
            if cur < MIN_PER_Z:
                need = MIN_PER_Z - cur
                pool = pdf[
                    (pdf["z_level"] == z) & (~pdf.index.isin(picked_idx))
                ].nlargest(need, "padim_score")
                picked_idx.update(pool.index.tolist())

        sel = pdf[pdf.index.isin(picked_idx)].copy()
        if len(sel) > MAX_CAP:
            sel = sel.nlargest(MAX_CAP, "padim_score")
        sel["rule_b_variant"] = "B5_p95_plus_diverse_fallback"
        parts.append(sel)
    return pd.concat(parts, ignore_index=True)


# ── 지표 계산 ──────────────────────────────────────────────────────────────────

def compute_metrics(
    variant_df: pd.DataFrame,
    all_stage1_ids: list,
    lesion_size_map: dict,
    lesion_slice_counts: dict,
    variant_name: str,
) -> dict:
    """patch_label, lesion_overlap은 평가 지표 계산에만 사용."""
    n_total = len(variant_df)
    per_patient = variant_df.groupby("patient_id").size()

    pos_mask = variant_df["lesion_overlap"].astype(bool)
    n_pos = int(pos_mask.sum())
    pos_ratio = n_pos / n_total if n_total > 0 else 0.0
    fp_ratio = 1.0 - pos_ratio

    patients_with_overlap = set(variant_df.loc[pos_mask, "patient_id"].unique())
    n_hit_patients = len(patients_with_overlap)
    no_hit_patients = sorted(set(all_stage1_ids) - patients_with_overlap)
    lung1_415_hit = "LUNG1-415" not in no_hit_patients

    # lesion slice hit rate
    hit_slices = (
        variant_df.loc[pos_mask]
        .groupby("patient_id")["slice_index"]
        .nunique()
    )
    total_lesion_slices = sum(lesion_slice_counts.get(pid, 0) for pid in all_stage1_ids)
    total_hit_slices = int(hit_slices.sum())
    slice_hit_rate = total_hit_slices / total_lesion_slices if total_lesion_slices > 0 else 0.0

    # group별
    group_stats: dict = {}
    for grp, gdf in variant_df.groupby("group"):
        g_total = len(gdf)
        g_pos = int(gdf["lesion_overlap"].astype(bool).sum())
        group_stats[grp] = {
            "n_patients": int(gdf["patient_id"].nunique()),
            "n_candidates": g_total,
            "positive_count": g_pos,
            "positive_ratio": g_pos / g_total if g_total > 0 else 0.0,
            "fp_ratio": 1 - g_pos / g_total if g_total > 0 else 0.0,
        }

    # lesion_size별
    size_summary: dict = {}
    for sz, pids in lesion_size_map.items():
        n_pat = len(pids)
        n_hit = sum(1 for p in pids if p in patients_with_overlap)
        size_summary[sz] = {
            "n_patients": n_pat,
            "n_with_lesion_overlap": n_hit,
            "hit_rate": n_hit / n_pat if n_pat > 0 else 0.0,
        }

    pos_bin_dist = variant_df["position_bin"].value_counts().to_dict()
    z_level_dist = variant_df["z_level"].value_counts().to_dict()
    too_many = bool(per_patient.max() > 1000) if len(per_patient) > 0 else False

    return {
        "variant": variant_name,
        "total_candidates": int(n_total),
        "n_patients": int(len(per_patient)),
        "candidates_per_patient": {
            "min": int(per_patient.min()),
            "median": float(per_patient.median()),
            "mean": float(per_patient.mean()),
            "max": int(per_patient.max()),
        },
        "positive_candidates": n_pos,
        "positive_ratio": pos_ratio,
        "fp_ratio": fp_ratio,
        "n_patients_with_lesion_overlap": n_hit_patients,
        "no_hit_patients": no_hit_patients,
        "lung1_415_hit": lung1_415_hit,
        "lesion_slice_hit_rate": slice_hit_rate,
        "group_summary": group_stats,
        "lesion_size_summary": size_summary,
        "position_bin_distribution": {k: int(v) for k, v in pos_bin_dist.items()},
        "z_level_distribution": {k: int(v) for k, v in z_level_dist.items()},
        "too_many_candidates_flag": too_many,
        "note_label_free": (
            "후보 선택 기준: padim_score, slice_index, position_bin, z_level만 사용. "
            "patch_label, lesion_overlap은 평가 지표 계산에만 사용."
        ),
    }


# ── summary MD ────────────────────────────────────────────────────────────────

def write_summary_md(results: list, rule_a_summary: dict) -> str:
    lines = [
        "# Rule B stage1_dev Candidate Dry-run Summary",
        "",
        "- 생성일: 2026-05-23",
        "- stage1_dev 154명 대상",
        f"- Rule A 기준선: 총 {rule_a_summary['total_candidates']:,}개, "
          f"FP {rule_a_summary['fp_ratio']:.4f}, "
          f"slice_hit_rate {rule_a_summary['slice_hit_rate']:.4f}",
        "- **후보 선택 기준: padim_score, slice_index, position_bin, z_level (label-free)**",
        "- patch_label, lesion_overlap은 평가 지표 계산에만 사용",
        "- stage2_holdout 봉인 준수",
        "- grid_position_bin 컬럼 없음 → position_bin(6종) + z_level 조합으로 다양성 구현",
        "",
        "## Variant 비교표",
        "",
        "| Variant | 총 후보 | mean/환자 | max/환자 | positive 수 | positive율 | FP율 | no-hit 수 | LUNG1-415 | slice hit rate |",
        "|---------|---------|-----------|---------|------------|------------|------|-----------|-----------|---------------|",
    ]

    for r in results:
        cpp = r["candidates_per_patient"]
        lines.append(
            f"| {r['variant']} "
            f"| {r['total_candidates']:,} "
            f"| {cpp['mean']:.0f} "
            f"| {cpp['max']:,} "
            f"| {r['positive_candidates']:,} "
            f"| {r['positive_ratio']:.4f} "
            f"| {r['fp_ratio']:.4f} "
            f"| {len(r['no_hit_patients'])} "
            f"| {'O' if r['lung1_415_hit'] else 'X'} "
            f"| {r['lesion_slice_hit_rate']:.4f} |"
        )

    lines += [
        "",
        "## Rule A 기준선",
        f"- 총 후보: {rule_a_summary['total_candidates']:,}",
        f"- positive율: {rule_a_summary['positive_ratio']:.4f}",
        f"- FP율: {rule_a_summary['fp_ratio']:.4f}",
        f"- slice hit rate: {rule_a_summary['slice_hit_rate']:.4f}",
        f"- no-hit 환자: {rule_a_summary['no_hit_patients']}",
        "",
    ]

    for r in results:
        cpp = r["candidates_per_patient"]
        lines += [
            f"## {r['variant']}",
            f"- 총 후보: {r['total_candidates']:,}",
            f"- 환자별 min/median/mean/max: "
              f"{cpp['min']} / {cpp['median']:.0f} / {cpp['mean']:.0f} / {cpp['max']}",
            f"- positive: {r['positive_candidates']:,} ({r['positive_ratio']:.4f})",
            f"- FP율: {r['fp_ratio']:.4f}",
            f"- lesion-overlap 보유 환자: {r['n_patients_with_lesion_overlap']}/154",
            f"- no-hit 환자: {r['no_hit_patients'] if r['no_hit_patients'] else '없음'}",
            f"- LUNG1-415 hit: {'O' if r['lung1_415_hit'] else 'X'}",
            f"- slice hit rate: {r['lesion_slice_hit_rate']:.4f}",
            f"- 후보 수 과다 경고: {'있음' if r['too_many_candidates_flag'] else '없음'}",
            "",
            "**NSCLC/MSD별:**",
        ]
        for grp, gs in r["group_summary"].items():
            lines.append(
                f"  - {grp}: n={gs['n_patients']}, "
                f"positive율={gs['positive_ratio']:.4f}, FP율={gs['fp_ratio']:.4f}"
            )
        lines += ["", "**lesion_size별 hit rate:**"]
        for sz, ss in r["lesion_size_summary"].items():
            lines.append(
                f"  - {sz}: {ss['n_with_lesion_overlap']}/{ss['n_patients']} "
                f"({ss['hit_rate']:.3f})"
            )
        lines += ["", "**position_bin 분포:**"]
        for pb, cnt in sorted(r["position_bin_distribution"].items(), key=lambda x: -x[1]):
            lines.append(f"  - {pb}: {cnt:,}")
        lines += ["", "**z_level 분포:**"]
        for zl, cnt in sorted(r["z_level_distribution"].items(), key=lambda x: -x[1]):
            lines.append(f"  - {zl}: {cnt:,}")
        lines.append("")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("[1/8] Guard: 출력 파일 존재 여부 확인")
    guard_output_files_absent()

    print("[2/8] 입력 파일 로드")
    df = pd.read_csv(RULE_A_MANIFEST)
    split_df = pd.read_csv(STAGE_SPLIT)
    screening_df = pd.read_csv(SCREENING_CSV)
    with open(THRESHOLD_JSON) as f:
        threshold_info = json.load(f)
    with open(RULE_A_SUMMARY) as f:
        rule_a_summary = json.load(f)

    print("[3/8] Guard: 컬럼, NaN, holdout 확인")
    guard_required_columns(df)
    guard_no_nan_inf(df)
    guard_no_holdout(df)

    stage1_ids = split_df[split_df["stage_split"] == "stage1_dev"]["patient_id"].tolist()
    guard_stage1_dev_count(stage1_ids)
    stage1_id_set = set(stage1_ids)

    df = df[df["stage_split"] == "stage1_dev"].copy()
    df = df.reset_index(drop=True)
    print(f"    stage1_dev 로드: {df['patient_id'].nunique()}명, {len(df):,}행")

    # lesion_size_map 구성
    screening_s1 = screening_df[screening_df["patient_id"].isin(stage1_id_set)].copy()
    screening_s1["lesion_size_bin"] = pd.cut(
        screening_s1["lesion_patch_total"],
        bins=[0, 50, 200, 500, 9_999_999],
        labels=["tiny(≤50)", "small(51-200)", "medium(201-500)", "large(>500)"],
    )
    lesion_size_map = {
        str(k): v
        for k, v in screening_s1.groupby("lesion_size_bin", observed=True)["patient_id"]
        .apply(list).to_dict().items()
    }

    # Rule A manifest에서 lesion_overlap=True인 slice 수 (환자별)
    lesion_slice_counts = (
        df[df["lesion_overlap"].astype(bool)]
        .groupby("patient_id")["slice_index"]
        .nunique()
        .to_dict()
    )

    print("[4/8] Variant 생성 중")
    variants = [
        ("B1_slice_top30_patch5",        apply_b1),
        ("B2_slice_top50_patch5",        apply_b2),
        ("B3_slice_top50_patch10",       apply_b3),
        ("B4_diverse_top200",            apply_b4),
        ("B5_p95_plus_diverse_fallback", apply_b5),
    ]

    all_results = []
    all_manifest_parts = []

    for vname, vfunc in variants:
        print(f"    variant: {vname} ...", end="", flush=True)
        vdf = vfunc(df)
        all_manifest_parts.append(vdf)
        metrics = compute_metrics(vdf, stage1_ids, lesion_size_map, lesion_slice_counts, vname)
        all_results.append(metrics)
        print(
            f" 총 후보 {metrics['total_candidates']:,}, "
            f"FP {metrics['fp_ratio']:.4f}, "
            f"no-hit {len(metrics['no_hit_patients'])}, "
            f"LUNG1-415 {'O' if metrics['lung1_415_hit'] else 'X'}"
        )

    print("[5/8] Manifest CSV 저장")
    manifest_df = pd.concat(all_manifest_parts, ignore_index=True)
    manifest_df.to_csv(OUT_MANIFEST, index=False)
    print(f"    저장: {OUT_MANIFEST} ({len(manifest_df):,}행)")

    print("[6/8] Summary CSV 저장")
    rows = []
    for r in all_results:
        cpp = r["candidates_per_patient"]
        gs_nsclc = r["group_summary"].get("NSCLC", {})
        gs_msd = r["group_summary"].get("MSD_Lung", {})
        rows.append({
            "variant": r["variant"],
            "total_candidates": r["total_candidates"],
            "cand_per_patient_min": cpp["min"],
            "cand_per_patient_median": cpp["median"],
            "cand_per_patient_mean": cpp["mean"],
            "cand_per_patient_max": cpp["max"],
            "positive_candidates": r["positive_candidates"],
            "positive_ratio": r["positive_ratio"],
            "fp_ratio": r["fp_ratio"],
            "n_patients_with_lesion_overlap": r["n_patients_with_lesion_overlap"],
            "n_no_hit_patients": len(r["no_hit_patients"]),
            "no_hit_patients": ";".join(r["no_hit_patients"]),
            "lung1_415_hit": r["lung1_415_hit"],
            "lesion_slice_hit_rate": r["lesion_slice_hit_rate"],
            "nsclc_positive_ratio": gs_nsclc.get("positive_ratio", ""),
            "nsclc_fp_ratio": gs_nsclc.get("fp_ratio", ""),
            "msd_positive_ratio": gs_msd.get("positive_ratio", ""),
            "msd_fp_ratio": gs_msd.get("fp_ratio", ""),
            "too_many_candidates_flag": r["too_many_candidates_flag"],
        })
    pd.DataFrame(rows).to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"    저장: {OUT_SUMMARY_CSV}")

    print("[7/8] Summary JSON 저장")
    summary_json = {
        "generated": "2026-05-23",
        "rule": "rule_b",
        "stage1_dev_n_patients": 154,
        "stage2_holdout_sealed": True,
        "existing_results_modified": False,
        "label_free_selection": True,
        "note": (
            "후보 선택 기준: padim_score, slice_index, position_bin, z_level. "
            "patch_label/lesion_overlap은 평가용만 사용."
        ),
        "grid_position_bin_note": (
            "grid_position_bin 컬럼 없음 -> position_bin(6종) + z_level 조합으로 다양성 구현"
        ),
        "threshold_p95": threshold_info.get("threshold_p95"),
        "rule_a_baseline": {
            "total_candidates": rule_a_summary["total_candidates"],
            "positive_ratio": rule_a_summary["positive_ratio"],
            "fp_ratio": rule_a_summary["fp_ratio"],
            "slice_hit_rate": rule_a_summary["slice_hit_rate"],
            "no_hit_patients": rule_a_summary["no_hit_patients"],
        },
        "variants": all_results,
    }
    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    print(f"    저장: {OUT_SUMMARY_JSON}")

    print("[8/8] Summary MD 저장")
    md_text = write_summary_md(all_results, rule_a_summary)
    OUT_SUMMARY_MD.write_text(md_text, encoding="utf-8")
    print(f"    저장: {OUT_SUMMARY_MD}")

    print("\n완료. 생성 파일:")
    for f in [OUT_MANIFEST, OUT_SUMMARY_CSV, OUT_SUMMARY_JSON, OUT_SUMMARY_MD]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
