"""
generate_rule_c_candidates.py
Rule C: p95 후보 slice 최대 유지 + slice 내 patch 수만 감소 candidate
- stage1_dev 154명만 사용
- Rule A manifest 입력 (원본 score CSV 재로드 없음)
- crop/npy/PNG 생성 없음
- 기존 score/evaluation/reports 미수정
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# ── 입력 경로 ──────────────────────────────────────────────────────────────
RULE_A_MANIFEST  = REPO / "outputs/second-stage-lesion-refiner-v1/candidates/rule_a_p95_stage1_dev_candidate_manifest_dryrun.csv"
RULE_A_SUMMARY   = REPO / "outputs/second-stage-lesion-refiner-v1/reports/rule_a_p95_stage1_dev_candidate_summary.json"
STAGE_SPLIT      = REPO / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
SCREENING_CSV    = REPO / "outputs/position-aware-padim-v1/evaluation/lesion_subset_v2_model_v2/per_patient_screening.csv"

# ── 출력 경로 ──────────────────────────────────────────────────────────────
OUT_CAND_DIR     = REPO / "outputs/second-stage-lesion-refiner-v1/candidates"
OUT_REPORT_DIR   = REPO / "outputs/second-stage-lesion-refiner-v1/reports"
OUT_MANIFEST     = OUT_CAND_DIR / "rule_c_stage1_dev_candidate_manifest_dryrun.csv"
OUT_SUMMARY_CSV  = OUT_REPORT_DIR / "rule_c_stage1_dev_candidate_summary.csv"
OUT_SUMMARY_JSON = OUT_REPORT_DIR / "rule_c_stage1_dev_candidate_summary.json"
OUT_SUMMARY_MD   = OUT_REPORT_DIR / "rule_c_stage1_dev_candidate_summary.md"

IMAGE_SIZE = 512

# ── Guard 1: 출력 파일 이미 있으면 중단 ────────────────────────────────────
for p in [OUT_MANIFEST, OUT_SUMMARY_CSV, OUT_SUMMARY_JSON, OUT_SUMMARY_MD]:
    if p.exists():
        print(f"[ABORT] 출력 파일이 이미 존재합니다: {p}")
        sys.exit(1)


# ── derived_grid_position_bin 생성 ────────────────────────────────────────
def make_grid_position_bin(df: pd.DataFrame, image_size: int = IMAGE_SIZE) -> pd.Series:
    """y_center, x_center 기준으로 3×3 grid bin 생성 (image_size=512 기준)."""
    y_center = (df["y0"] + df["y1"]) / 2
    x_center = (df["x0"] + df["x1"]) / 2

    boundary = image_size / 3  # 170.666...

    y_labels = pd.cut(
        y_center,
        bins=[0, boundary, boundary * 2, image_size],
        labels=["top", "middle", "bottom"],
        include_lowest=True,
    ).astype(str)

    x_labels = pd.cut(
        x_center,
        bins=[0, boundary, boundary * 2, image_size],
        labels=["left", "center", "right"],
        include_lowest=True,
    ).astype(str)

    return y_labels + "_" + x_labels


# ── coordinate guard 확인 ─────────────────────────────────────────────────
def check_coordinates(df: pd.DataFrame) -> dict:
    """좌표 guard 확인 결과 반환. 512 초과 시 즉시 중단."""
    max_y1 = int(df["y1"].max())
    max_x1 = int(df["x1"].max())
    patch_size_y_unique = sorted(df.eval("y1 - y0").unique().tolist())
    patch_size_x_unique = sorted(df.eval("x1 - x0").unique().tolist())

    violations = []
    if (df["y0"] < 0).any():
        violations.append("y0 < 0")
    if (df["x0"] < 0).any():
        violations.append("x0 < 0")
    if (df["y1"] > IMAGE_SIZE).any():
        violations.append(f"y1 > {IMAGE_SIZE}")
    if (df["x1"] > IMAGE_SIZE).any():
        violations.append(f"x1 > {IMAGE_SIZE}")
    if (df["y1"] <= df["y0"]).any():
        violations.append("y1 <= y0")
    if (df["x1"] <= df["x0"]).any():
        violations.append("x1 <= x0")

    if violations:
        print(f"[ABORT] coordinate guard 실패: {violations}")
        sys.exit(1)

    return {
        "image_size_assumption": IMAGE_SIZE,
        "max_y1": max_y1,
        "max_x1": max_x1,
        "patch_size_y_unique": patch_size_y_unique,
        "patch_size_x_unique": patch_size_x_unique,
        "coordinate_guard_passed": True,
    }


# ── variant 함수 ──────────────────────────────────────────────────────────
# 후보 선택 사용 가능: patient_id, slice_index/local_z, padim_score,
#                      y0/x0/y1/x1, position_bin, z_level, central_peripheral,
#                      derived_grid_position_bin
# 후보 선택 사용 금지: patch_label, lesion_overlap, lesion_pixels,
#                      lesion_patch_ratio, has_lesion_patch

def apply_c1(df: pd.DataFrame) -> pd.DataFrame:
    """C1: 모든 p95 slice 유지, 각 slice 내 padim_score 상위 1개 patch 선택."""
    sel = (
        df.sort_values("padim_score", ascending=False)
          .groupby(["patient_id", "slice_index"], sort=False)
          .head(1)
          .copy()
    )
    sel["rule_c_variant"] = "C1_all_p95_slices_top1_patch"
    return sel.reset_index(drop=True)


def apply_c2(df: pd.DataFrame) -> pd.DataFrame:
    """C2: 모든 p95 slice 유지, 각 slice 내 padim_score 상위 3개 patch 선택."""
    sel = (
        df.sort_values("padim_score", ascending=False)
          .groupby(["patient_id", "slice_index"], sort=False)
          .head(3)
          .copy()
    )
    sel["rule_c_variant"] = "C2_all_p95_slices_top3_patch"
    return sel.reset_index(drop=True)


def apply_c3(df: pd.DataFrame) -> pd.DataFrame:
    """C3: 모든 p95 slice 유지, 각 slice 내 padim_score 상위 5개 patch 선택."""
    sel = (
        df.sort_values("padim_score", ascending=False)
          .groupby(["patient_id", "slice_index"], sort=False)
          .head(5)
          .copy()
    )
    sel["rule_c_variant"] = "C3_all_p95_slices_top5_patch"
    return sel.reset_index(drop=True)


def apply_c4(df: pd.DataFrame) -> pd.DataFrame:
    """C4: 모든 p95 slice 유지, 각 slice 내 padim_score 상위 10개 patch 선택."""
    sel = (
        df.sort_values("padim_score", ascending=False)
          .groupby(["patient_id", "slice_index"], sort=False)
          .head(10)
          .copy()
    )
    sel["rule_c_variant"] = "C4_all_p95_slices_top10_patch"
    return sel.reset_index(drop=True)


def apply_c5(df: pd.DataFrame) -> pd.DataFrame:
    """C5: 모든 p95 slice 유지, slice × grid bin별 padim_score 상위 1개 patch 선택."""
    sel = (
        df.sort_values("padim_score", ascending=False)
          .groupby(["patient_id", "slice_index", "derived_grid_position_bin"], sort=False)
          .head(1)
          .copy()
    )
    sel["rule_c_variant"] = "C5_all_p95_slices_grid_top1_each"
    return sel.reset_index(drop=True)


def apply_c6(df: pd.DataFrame) -> pd.DataFrame:
    """C6: 모든 p95 slice 유지, slice × grid bin별 padim_score 상위 2개 patch 선택."""
    sel = (
        df.sort_values("padim_score", ascending=False)
          .groupby(["patient_id", "slice_index", "derived_grid_position_bin"], sort=False)
          .head(2)
          .copy()
    )
    sel["rule_c_variant"] = "C6_all_p95_slices_grid_top2_each"
    return sel.reset_index(drop=True)


# ── 지표 계산 ──────────────────────────────────────────────────────────────
def compute_metrics(
    variant_df: pd.DataFrame,
    all_stage1_ids: list,
    lesion_size_map: dict,
    total_lesion_slices: int,
    rule_a_total: int,
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
    total_hit_slices = int(hit_slices.sum())
    slice_hit_rate = total_hit_slices / total_lesion_slices if total_lesion_slices > 0 else 0.0

    # group별 요약
    group_stats: dict = {}
    for grp, gdf in variant_df.groupby("group"):
        g_total = len(gdf)
        g_pos = int(gdf["lesion_overlap"].astype(bool).sum())
        group_stats[str(grp)] = {
            "n_patients": int(gdf["patient_id"].nunique()),
            "n_candidates": g_total,
            "positive_count": g_pos,
            "positive_ratio": g_pos / g_total if g_total > 0 else 0.0,
            "fp_ratio": 1 - g_pos / g_total if g_total > 0 else 0.0,
        }

    # lesion_size별 요약
    size_summary: dict = {}
    for sz, pids in lesion_size_map.items():
        n_pat = len(pids)
        n_hit = sum(1 for p in pids if p in patients_with_overlap)
        size_summary[str(sz)] = {
            "n_patients": n_pat,
            "n_with_lesion_overlap": n_hit,
            "hit_rate": n_hit / n_pat if n_pat > 0 else 0.0,
        }

    pos_bin_dist = variant_df["position_bin"].value_counts().to_dict()
    z_level_dist = variant_df["z_level"].value_counts().to_dict()
    grid_bin_dist = variant_df["derived_grid_position_bin"].value_counts().to_dict()

    reduction_rate = 1.0 - (n_total / rule_a_total) if rule_a_total > 0 else 0.0

    return {
        "variant": variant_name,
        "total_candidates": n_total,
        "rule_a_total_candidates": rule_a_total,
        "reduction_rate_vs_rule_a": float(reduction_rate),
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
        "derived_grid_position_bin_distribution": {k: int(v) for k, v in grid_bin_dist.items()},
        "note_label_free": (
            "후보 선택 기준: padim_score, slice_index, derived_grid_position_bin 등. "
            "patch_label, lesion_overlap, lesion_pixels 등은 평가 지표 계산에만 사용."
        ),
    }


# ── summary MD ─────────────────────────────────────────────────────────────
def write_summary_md(results: list, rule_a_summary: dict, coord_guard: dict) -> str:
    lines = [
        "# Rule C stage1_dev Candidate Dry-run Summary",
        "",
        "- 생성일: 2026-05-24",
        "- stage1_dev 154명 대상",
        f"- Rule A 기준선: 총 {rule_a_summary['total_candidates']:,}개, "
          f"FP {rule_a_summary['fp_ratio']:.4f}, "
          f"slice_hit_rate {rule_a_summary['slice_hit_rate']:.4f}",
        "- **목적: p95 후보 slice 최대 유지 + slice 내 patch 수만 감소**",
        "- **후보 선택 기준: padim_score, slice_index, derived_grid_position_bin (label-free)**",
        "- patch_label, lesion_overlap 등은 평가 지표 계산에만 사용",
        "- stage2_holdout 봉인 준수",
        "",
        "## Coordinate Guard",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        f"| image_size_assumption | {coord_guard['image_size_assumption']} |",
        f"| max_y1 | {coord_guard['max_y1']} |",
        f"| max_x1 | {coord_guard['max_x1']} |",
        f"| patch_size_y unique | {coord_guard['patch_size_y_unique']} |",
        f"| patch_size_x unique | {coord_guard['patch_size_x_unique']} |",
        f"| coordinate_guard_passed | {coord_guard['coordinate_guard_passed']} |",
        "",
        "## Variant 비교표",
        "",
        "| Variant | 총 후보 | 감소율 | mean/환자 | max/환자 | positive 수 | positive율 | FP율 | no-hit 수 | LUNG1-415 | slice hit rate |",
        "|---------|---------|--------|-----------|---------|------------|------------|------|-----------|-----------|---------------|",
    ]

    for r in results:
        cpp = r["candidates_per_patient"]
        lines.append(
            f"| {r['variant']} "
            f"| {r['total_candidates']:,} "
            f"| {r['reduction_rate_vs_rule_a']:.4f} "
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
            f"- Rule A 대비 감소율: {r['reduction_rate_vs_rule_a']:.4f}",
            f"- 환자별 min/median/mean/max: "
              f"{cpp['min']} / {cpp['median']:.0f} / {cpp['mean']:.0f} / {cpp['max']}",
            f"- positive: {r['positive_candidates']:,} ({r['positive_ratio']:.4f})",
            f"- FP율: {r['fp_ratio']:.4f}",
            f"- lesion-overlap 보유 환자: {r['n_patients_with_lesion_overlap']}/154",
            f"- no-hit 환자: {r['no_hit_patients'] if r['no_hit_patients'] else '없음'}",
            f"- LUNG1-415 hit: {'O' if r['lung1_415_hit'] else 'X'}",
            f"- slice hit rate: {r['lesion_slice_hit_rate']:.4f}",
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
        lines += ["", "**derived_grid_position_bin 분포:**"]
        for gb, cnt in sorted(r["derived_grid_position_bin_distribution"].items(), key=lambda x: -x[1]):
            lines.append(f"  - {gb}: {cnt:,}")
        lines.append("")

    lines += [
        "---",
        "",
        "*생성일: 2026-05-24 | stage1_dev 전용 | stage2_holdout 봉인 | 기존 결과 미수정*",
    ]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────
def main():
    print("[1/8] Guard: 출력 파일 존재 여부 확인 (완료)")

    print("[2/8] 입력 파일 로드")
    df = pd.read_csv(RULE_A_MANIFEST)
    split_df = pd.read_csv(STAGE_SPLIT)
    screening_df = pd.read_csv(SCREENING_CSV)
    with open(RULE_A_SUMMARY) as f:
        rule_a_summary = json.load(f)
    print(f"    Rule A manifest: {len(df):,}행")

    print("[3/8] Guard: stage1_dev 확인 및 stage2_holdout 봉인")
    if (df["stage_split"] == "stage2_holdout").any():
        print("[ABORT] stage2_holdout 데이터가 포함되어 있습니다. 중단.")
        sys.exit(1)

    stage1_ids = split_df[split_df["stage_split"] == "stage1_dev"]["patient_id"].tolist()
    if len(stage1_ids) != 154:
        print(f"[ABORT] stage1_dev 환자 수가 154명이 아닙니다: {len(stage1_ids)}")
        sys.exit(1)

    df = df[df["stage_split"] == "stage1_dev"].copy().reset_index(drop=True)
    print(f"    stage1_dev: {df['patient_id'].nunique()}명, {len(df):,}행")
    print(f"    [OK] stage2_holdout 봉인 준수")

    print("[4/8] derived_grid_position_bin 생성 (image_size=512)")
    df["derived_grid_position_bin"] = make_grid_position_bin(df)
    print(f"    grid bin 종류({df['derived_grid_position_bin'].nunique()}개): "
          f"{sorted(df['derived_grid_position_bin'].unique())}")

    print("[5/8] coordinate guard 확인")
    coord_guard = check_coordinates(df)
    print(f"    [OK] coordinate_guard_passed: {coord_guard['coordinate_guard_passed']}")
    print(f"    max_y1={coord_guard['max_y1']}, max_x1={coord_guard['max_x1']}")
    print(f"    patch_size_y unique={coord_guard['patch_size_y_unique']}")
    print(f"    patch_size_x unique={coord_guard['patch_size_x_unique']}")

    # lesion_size_map 구성
    stage1_id_set = set(stage1_ids)
    screening_s1 = screening_df[screening_df["patient_id"].isin(stage1_id_set)].copy()
    if "lesion_patch_total" in screening_s1.columns:
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
    else:
        lesion_size_map = {}
        print("    [INFO] lesion_patch_total 컬럼 없음 - lesion_size 분석 생략")

    # 전체 lesion slice 수 (Rule A 기준, 평가용)
    lesion_slice_count = (
        df[df["lesion_overlap"].astype(bool)]
        .groupby(["patient_id", "slice_index"])
        .size()
        .shape[0]
    )
    rule_a_total = len(df)
    print(f"    전체 lesion slice 수 (Rule A 기준): {lesion_slice_count:,}")
    print(f"    Rule A 전체 후보 수: {rule_a_total:,}")

    print("[6/8] C1~C6 variant 생성")
    variants = [
        ("C1_all_p95_slices_top1_patch",     apply_c1),
        ("C2_all_p95_slices_top3_patch",     apply_c2),
        ("C3_all_p95_slices_top5_patch",     apply_c3),
        ("C4_all_p95_slices_top10_patch",    apply_c4),
        ("C5_all_p95_slices_grid_top1_each", apply_c5),
        ("C6_all_p95_slices_grid_top2_each", apply_c6),
    ]

    all_results = []
    all_manifest_parts = []

    for vname, vfunc in variants:
        print(f"    variant: {vname} ...", end="", flush=True)
        vdf = vfunc(df)
        all_manifest_parts.append(vdf)
        metrics = compute_metrics(
            vdf, stage1_ids, lesion_size_map, lesion_slice_count, rule_a_total, vname
        )
        all_results.append(metrics)
        print(
            f" 총 후보 {metrics['total_candidates']:,}, "
            f"감소율 {metrics['reduction_rate_vs_rule_a']:.4f}, "
            f"FP {metrics['fp_ratio']:.4f}, "
            f"no-hit {len(metrics['no_hit_patients'])}, "
            f"LUNG1-415 {'O' if metrics['lung1_415_hit'] else 'X'}, "
            f"slice_hit {metrics['lesion_slice_hit_rate']:.4f}"
        )

    print("[7/8] 출력 파일 저장")
    OUT_CAND_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.concat(all_manifest_parts, ignore_index=True)
    manifest_df.to_csv(OUT_MANIFEST, index=False, encoding="utf-8-sig")
    print(f"    manifest: {OUT_MANIFEST} ({len(manifest_df):,}행)")

    # summary CSV
    rows = []
    for r in all_results:
        cpp = r["candidates_per_patient"]
        gs_nsclc = r["group_summary"].get("NSCLC", {})
        gs_msd   = r["group_summary"].get("MSD_Lung", {})
        rows.append({
            "variant":                      r["variant"],
            "total_candidates":             r["total_candidates"],
            "reduction_rate_vs_rule_a":     r["reduction_rate_vs_rule_a"],
            "cand_per_patient_min":         cpp["min"],
            "cand_per_patient_median":      cpp["median"],
            "cand_per_patient_mean":        cpp["mean"],
            "cand_per_patient_max":         cpp["max"],
            "positive_candidates":          r["positive_candidates"],
            "positive_ratio":               r["positive_ratio"],
            "fp_ratio":                     r["fp_ratio"],
            "n_patients_with_lesion_overlap": r["n_patients_with_lesion_overlap"],
            "n_no_hit_patients":            len(r["no_hit_patients"]),
            "no_hit_patients":              ";".join(r["no_hit_patients"]),
            "lung1_415_hit":                r["lung1_415_hit"],
            "lesion_slice_hit_rate":        r["lesion_slice_hit_rate"],
            "nsclc_positive_ratio":         gs_nsclc.get("positive_ratio", ""),
            "nsclc_fp_ratio":               gs_nsclc.get("fp_ratio", ""),
            "msd_positive_ratio":           gs_msd.get("positive_ratio", ""),
            "msd_fp_ratio":                 gs_msd.get("fp_ratio", ""),
        })
    pd.DataFrame(rows).to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"    summary CSV: {OUT_SUMMARY_CSV}")

    # summary JSON
    summary_json = {
        "generated": "2026-05-24",
        "rule": "rule_c",
        "stage1_dev_n_patients": 154,
        "stage2_holdout_sealed": True,
        "existing_results_modified": False,
        "label_free_selection": True,
        "note": (
            "목적: p95 후보 slice 최대 유지 + slice 내 patch 수만 감소. "
            "후보 선택 기준: padim_score, slice_index, derived_grid_position_bin. "
            "patch_label/lesion_overlap은 평가용만 사용."
        ),
        "coordinate_guard": coord_guard,
        "rule_a_baseline": {
            "total_candidates":  rule_a_summary["total_candidates"],
            "positive_ratio":    rule_a_summary["positive_ratio"],
            "fp_ratio":          rule_a_summary["fp_ratio"],
            "slice_hit_rate":    rule_a_summary["slice_hit_rate"],
            "no_hit_patients":   rule_a_summary["no_hit_patients"],
        },
        "variants": all_results,
    }
    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    print(f"    summary JSON: {OUT_SUMMARY_JSON}")

    # summary MD
    md_text = write_summary_md(all_results, rule_a_summary, coord_guard)
    OUT_SUMMARY_MD.write_text(md_text, encoding="utf-8")
    print(f"    summary MD: {OUT_SUMMARY_MD}")

    print("\n[8/8] 완료. 생성 파일:")
    for out_file in [OUT_MANIFEST, OUT_SUMMARY_CSV, OUT_SUMMARY_JSON, OUT_SUMMARY_MD]:
        print(f"  {out_file}")


if __name__ == "__main__":
    main()
