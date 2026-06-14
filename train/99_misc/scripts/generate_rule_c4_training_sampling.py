"""
generate_rule_c4_training_sampling.py
C4 candidate pool에서 학습용 sampling manifest dry-run
- S1/S2/S3/S4 4가지 sampling rule 비교
- stage1_dev 154명만 사용, stage2_holdout 봉인
- manifest only: crop/npy/PNG 생성 없음, 기존 결과 미수정
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# ── 입력 경로 ──────────────────────────────────────────────────────────────
RULE_C_MANIFEST  = REPO / "outputs/second-stage-lesion-refiner-v1/candidates/rule_c_stage1_dev_candidate_manifest_dryrun.csv"
STAGE_SPLIT      = REPO / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
SCREENING_CSV    = REPO / "outputs/position-aware-padim-v1/evaluation/lesion_subset_v2_model_v2/per_patient_screening.csv"

# ── 출력 경로 ──────────────────────────────────────────────────────────────
OUT_CAND_DIR     = REPO / "outputs/second-stage-lesion-refiner-v1/candidates"
OUT_REPORT_DIR   = REPO / "outputs/second-stage-lesion-refiner-v1/reports"
OUT_MANIFEST     = OUT_CAND_DIR / "rule_c4_training_sampling_manifest_dryrun.csv"
OUT_SUMMARY_CSV  = OUT_REPORT_DIR / "rule_c4_training_sampling_summary.csv"
OUT_SUMMARY_JSON = OUT_REPORT_DIR / "rule_c4_training_sampling_summary.json"
OUT_SUMMARY_MD   = OUT_REPORT_DIR / "rule_c4_training_sampling_summary.md"

# S4 상수
S4_HN_MULTIPLIER  = 3
S4_HN_CAP         = 300
S4_NO_POS_HN_CAP  = 100

# ── Guard 1: 출력 파일 존재 시 중단 ────────────────────────────────────────
for p in [OUT_MANIFEST, OUT_SUMMARY_CSV, OUT_SUMMARY_JSON, OUT_SUMMARY_MD]:
    if p.exists():
        print(f"[ABORT] 출력 파일이 이미 존재합니다: {p}")
        sys.exit(1)


# ── hard negative 선택 함수 ────────────────────────────────────────────────

def select_hn_with_diversity(neg_df: pd.DataFrame, n_target: int) -> pd.DataFrame:
    """S1 전용: position_bin × z_level 그룹별 균등 배분 후 padim_score 보충."""
    if n_target <= 0 or len(neg_df) == 0:
        return pd.DataFrame(columns=neg_df.columns)

    neg_df = neg_df.reset_index(drop=True)
    groups = neg_df.groupby(["position_bin", "z_level"], sort=False)
    n_groups = len(groups)
    base_per_group = max(1, n_target // n_groups)

    selected_idx: set[int] = set()
    for _, gdf in groups:
        k = min(base_per_group, len(gdf))
        top_idx = gdf.nlargest(k, "padim_score").index.tolist()
        selected_idx.update(top_idx)

    result = neg_df.loc[sorted(selected_idx)].copy()

    # 부족분 채우기: 미선택 후보에서 padim_score 상위
    if len(result) < n_target:
        remaining = neg_df[~neg_df.index.isin(selected_idx)]
        extra_n = n_target - len(result)
        extra = remaining.nlargest(extra_n, "padim_score")
        result = pd.concat([result, extra])

    # 초과분 trim
    if len(result) > n_target:
        result = result.nlargest(n_target, "padim_score")

    return result.reset_index(drop=True)


def select_hn_top_score(neg_df: pd.DataFrame, n_target: int) -> pd.DataFrame:
    """S2/S3/S4 전용: padim_score 높은 순으로 n_target개 선택."""
    if n_target <= 0 or len(neg_df) == 0:
        return pd.DataFrame(columns=neg_df.columns)
    return neg_df.nlargest(min(n_target, len(neg_df)), "padim_score").reset_index(drop=True)


# ── sampling 적용 ──────────────────────────────────────────────────────────

def apply_sampling(
    c4_df: pd.DataFrame,
    rule_name: str,
    all_stage1_ids: list,
    hn_multiplier: float | None = None,
    s4_mode: bool = False,
) -> pd.DataFrame:
    """환자별 positive + hard negative manifest 생성."""
    pos_mask = (c4_df["patch_label"] == 1) | (c4_df["lesion_overlap"].astype(bool))
    pos_df = c4_df[pos_mask].copy()
    neg_df = c4_df[~pos_mask].copy()

    result_parts = []

    for pid in all_stage1_ids:
        p_pos = pos_df[pos_df["patient_id"] == pid].copy()
        p_neg = neg_df[neg_df["patient_id"] == pid].copy()

        n_pos = len(p_pos)
        no_positive = (n_pos == 0)

        if s4_mode:
            if no_positive:
                n_hn = S4_NO_POS_HN_CAP
            else:
                n_hn = min(n_pos * S4_HN_MULTIPLIER, S4_HN_CAP)
        else:
            # S1/S2/S3: no-positive 환자는 hn도 0 (positive × N = 0)
            n_hn = int(n_pos * hn_multiplier) if not no_positive else 0

        # positive 행 구성
        if not no_positive:
            p_pos["sampling_label"] = "positive"
            p_pos["no_positive_patient"] = False
            result_parts.append(p_pos)

        # hard negative 행 구성
        if n_hn > 0:
            if rule_name == "S1_all_positive_hn1":
                p_hn = select_hn_with_diversity(p_neg, n_hn)
            else:
                p_hn = select_hn_top_score(p_neg, n_hn)

            if len(p_hn) > 0:
                p_hn["sampling_label"] = "hard_negative"
                p_hn["no_positive_patient"] = no_positive
                result_parts.append(p_hn)

    if not result_parts:
        return pd.DataFrame()

    out = pd.concat(result_parts, ignore_index=True)
    out["sampling_rule"] = rule_name
    return out


# ── 지표 계산 ──────────────────────────────────────────────────────────────

def compute_metrics(
    vdf: pd.DataFrame,
    rule_name: str,
    all_stage1_ids: list,
    lesion_size_map: dict,
    hn_multiplier: float | None,
    s4_mode: bool,
) -> dict:
    n_total = len(vdf)
    pos_mask = vdf["sampling_label"] == "positive"
    neg_mask = vdf["sampling_label"] == "hard_negative"
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    per_patient_total = vdf.groupby("patient_id").size()
    per_patient_pos   = vdf[pos_mask].groupby("patient_id").size()
    per_patient_neg   = vdf[neg_mask].groupby("patient_id").size()

    no_pos_patients = sorted(
        pid for pid in all_stage1_ids
        if pid not in per_patient_pos.index
    )

    def series_stats(s: pd.Series) -> dict:
        if len(s) == 0:
            return {"min": 0, "median": 0.0, "mean": 0.0, "max": 0}
        return {
            "min":    int(s.min()),
            "median": float(s.median()),
            "mean":   float(s.mean()),
            "max":    int(s.max()),
        }

    # group별 요약
    group_stats: dict = {}
    for grp, gdf in vdf.groupby("group"):
        g_pos = int((gdf["sampling_label"] == "positive").sum())
        g_neg = int((gdf["sampling_label"] == "hard_negative").sum())
        g_total = len(gdf)
        group_stats[str(grp)] = {
            "n_patients":      int(gdf["patient_id"].nunique()),
            "n_total":         g_total,
            "positive":        g_pos,
            "hard_negative":   g_neg,
            "positive_ratio":  g_pos / g_total if g_total > 0 else 0.0,
        }

    # lesion_size별 positive 보존율
    size_summary: dict = {}
    for sz, pids in lesion_size_map.items():
        n_pat = len(pids)
        n_with_pos = sum(1 for p in pids if p in per_patient_pos.index)
        size_summary[str(sz)] = {
            "n_patients":               n_pat,
            "n_with_positive":          n_with_pos,
            "positive_preserved_rate":  n_with_pos / n_pat if n_pat > 0 else 0.0,
        }

    pos_bin_dist  = vdf["position_bin"].value_counts().to_dict()
    z_level_dist  = vdf["z_level"].value_counts().to_dict()
    grid_bin_dist = vdf["derived_grid_position_bin"].value_counts().to_dict()

    result: dict = {
        "variant":                       rule_name,
        "total_candidates":              n_total,
        "positive_count":                n_pos,
        "hard_negative_count":           n_neg,
        "positive_ratio":                n_pos / n_total if n_total > 0 else 0.0,
        "positive_to_negative_ratio":    f"1:{n_neg/n_pos:.2f}" if n_pos > 0 else "N/A",
        "candidates_per_patient":        series_stats(per_patient_total),
        "positive_per_patient":          series_stats(per_patient_pos),
        "hard_negative_per_patient":     series_stats(per_patient_neg),
        "n_no_positive_patients":        len(no_pos_patients),
        "no_positive_patients":          no_pos_patients,
        "group_summary":                 group_stats,
        "lesion_size_positive_preservation": size_summary,
        "position_bin_distribution":     {k: int(v) for k, v in pos_bin_dist.items()},
        "z_level_distribution":          {k: int(v) for k, v in z_level_dist.items()},
        "derived_grid_position_bin_distribution": {k: int(v) for k, v in grid_bin_dist.items()},
    }

    if s4_mode:
        result["hard_negative_rule"]          = f"min(pos_count * {S4_HN_MULTIPLIER}, {S4_HN_CAP})"
        result["no_positive_patient_hn_cap"]  = S4_NO_POS_HN_CAP
    else:
        result["hard_negative_rule"] = f"positive × {int(hn_multiplier)}"

    return result


# ── summary MD ─────────────────────────────────────────────────────────────

def write_summary_md(results: list, c4_total: int, c4_pos: int, c4_neg: int) -> str:
    lines = [
        "# Rule C4 Training Sampling Manifest Dry-run Summary",
        "",
        "- 생성일: 2026-05-24",
        "- stage1_dev 154명 대상",
        f"- C4 전체 후보: {c4_total:,}개 (positive {c4_pos:,} / negative {c4_neg:,})",
        "- **목적: 학습용 positive / hard negative sampling 비교 (S1~S4)**",
        "- **manifest only: crop/npy/PNG 생성 없음**",
        "- stage2_holdout 봉인 준수",
        "",
        "## Sampling Rule 비교표",
        "",
        "| Rule | 총 후보 | positive | hard_neg | pos:neg | 환자별 mean | no-pos 환자 |",
        "|------|---------|----------|----------|---------|------------|------------|",
    ]
    for r in results:
        cpp = r["candidates_per_patient"]
        lines.append(
            f"| {r['variant']} "
            f"| {r['total_candidates']:,} "
            f"| {r['positive_count']:,} "
            f"| {r['hard_negative_count']:,} "
            f"| {r['positive_to_negative_ratio']} "
            f"| {cpp['mean']:.0f} "
            f"| {r['n_no_positive_patients']} |"
        )

    for r in results:
        cpp  = r["candidates_per_patient"]
        p_pp = r["positive_per_patient"]
        h_pp = r["hard_negative_per_patient"]
        lines += [
            "",
            f"## {r['variant']}",
            f"- hard_negative_rule: {r['hard_negative_rule']}",
        ]
        if "no_positive_patient_hn_cap" in r:
            lines.append(f"- no_positive_patient_hn_cap: {r['no_positive_patient_hn_cap']}")
        lines += [
            f"- 총 후보: {r['total_candidates']:,}",
            f"- positive: {r['positive_count']:,} | hard_negative: {r['hard_negative_count']:,}",
            f"- positive:negative = {r['positive_to_negative_ratio']}",
            f"- 환자별 총 후보 min/median/mean/max: {cpp['min']}/{cpp['median']:.0f}/{cpp['mean']:.0f}/{cpp['max']}",
            f"- 환자별 positive min/median/mean/max: {p_pp['min']}/{p_pp['median']:.0f}/{p_pp['mean']:.0f}/{p_pp['max']}",
            f"- 환자별 hard_neg min/median/mean/max: {h_pp['min']}/{h_pp['median']:.0f}/{h_pp['mean']:.0f}/{h_pp['max']}",
            f"- no-positive 환자 수: {r['n_no_positive_patients']}",
            f"- no-positive 환자 목록: {r['no_positive_patients'] if r['no_positive_patients'] else '없음'}",
            "",
            "**NSCLC/MSD별:**",
        ]
        for grp, gs in r["group_summary"].items():
            lines.append(
                f"  - {grp}: n={gs['n_patients']}, "
                f"positive={gs['positive']:,}, hard_neg={gs['hard_negative']:,}, "
                f"positive율={gs['positive_ratio']:.4f}"
            )
        lines += ["", "**lesion_size별 positive 보존율:**"]
        for sz, ss in r["lesion_size_positive_preservation"].items():
            lines.append(
                f"  - {sz}: {ss['n_with_positive']}/{ss['n_patients']} "
                f"({ss['positive_preserved_rate']:.3f})"
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
    df_all      = pd.read_csv(RULE_C_MANIFEST)
    split_df    = pd.read_csv(STAGE_SPLIT)
    screening_df = pd.read_csv(SCREENING_CSV)
    print(f"    Rule C manifest: {len(df_all):,}행, variant 종류: {df_all['rule_c_variant'].nunique()}")

    print("[3/8] C4 필터링 및 stage2_holdout 봉인 확인")
    c4_df = df_all[df_all["rule_c_variant"] == "C4_all_p95_slices_top10_patch"].copy().reset_index(drop=True)
    print(f"    C4 후보: {len(c4_df):,}행, {c4_df['patient_id'].nunique()}명")

    if (c4_df["stage_split"] == "stage2_holdout").any():
        print("[ABORT] stage2_holdout 데이터가 포함되어 있습니다. 중단.")
        sys.exit(1)

    stage1_ids = split_df[split_df["stage_split"] == "stage1_dev"]["patient_id"].tolist()
    if len(stage1_ids) != 154:
        print(f"[ABORT] stage1_dev 환자 수가 154명이 아닙니다: {len(stage1_ids)}")
        sys.exit(1)
    print(f"    [OK] stage2_holdout 봉인 준수, stage1_dev={len(stage1_ids)}명")

    print("[4/8] lesion_size_map 구성")
    stage1_id_set = set(stage1_ids)
    screening_s1  = screening_df[screening_df["patient_id"].isin(stage1_id_set)].copy()
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
        print(f"    lesion_size_map 구성 완료: {list(lesion_size_map.keys())}")
    else:
        lesion_size_map = {}
        print("    [INFO] lesion_patch_total 컬럼 없음 - lesion_size 분석 생략")

    print("[5/8] C4 positive/negative 기본 통계")
    pos_mask  = (c4_df["patch_label"] == 1) | (c4_df["lesion_overlap"].astype(bool))
    n_c4_pos  = int(pos_mask.sum())
    n_c4_neg  = int((~pos_mask).sum())
    print(f"    positive: {n_c4_pos:,}, negative: {n_c4_neg:,}, positive율: {n_c4_pos/len(c4_df):.4f}")

    print("[6/8] S1~S4 sampling 적용")
    sampling_configs = [
        ("S1_all_positive_hn1",  False, 1.0),
        ("S2_all_positive_hn2",  False, 2.0),
        ("S3_all_positive_hn3",  False, 3.0),
        ("S4_patient_balanced",  True,  None),
    ]

    all_results        = []
    all_manifest_parts = []

    for rule_name, s4_mode, hn_mult in sampling_configs:
        print(f"    {rule_name} ...", end="", flush=True)
        vdf = apply_sampling(c4_df, rule_name, stage1_ids, hn_multiplier=hn_mult, s4_mode=s4_mode)
        all_manifest_parts.append(vdf)
        metrics = compute_metrics(vdf, rule_name, stage1_ids, lesion_size_map, hn_mult, s4_mode)
        all_results.append(metrics)
        print(
            f" 총 {metrics['total_candidates']:,}, "
            f"pos {metrics['positive_count']:,}, "
            f"hn {metrics['hard_negative_count']:,}, "
            f"no-pos 환자 {metrics['n_no_positive_patients']}"
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
        cpp  = r["candidates_per_patient"]
        p_pp = r["positive_per_patient"]
        h_pp = r["hard_negative_per_patient"]
        gs_nsclc = r["group_summary"].get("NSCLC", {})
        gs_msd   = r["group_summary"].get("MSD_Lung", {})
        rows.append({
            "rule":                         r["variant"],
            "total_candidates":             r["total_candidates"],
            "positive_count":               r["positive_count"],
            "hard_negative_count":          r["hard_negative_count"],
            "positive_ratio":               r["positive_ratio"],
            "positive_to_negative_ratio":   r["positive_to_negative_ratio"],
            "hard_negative_rule":           r["hard_negative_rule"],
            "cand_per_patient_min":         cpp["min"],
            "cand_per_patient_median":      cpp["median"],
            "cand_per_patient_mean":        cpp["mean"],
            "cand_per_patient_max":         cpp["max"],
            "pos_per_patient_min":          p_pp["min"],
            "pos_per_patient_median":       p_pp["median"],
            "pos_per_patient_mean":         p_pp["mean"],
            "pos_per_patient_max":          p_pp["max"],
            "hn_per_patient_min":           h_pp["min"],
            "hn_per_patient_median":        h_pp["median"],
            "hn_per_patient_mean":          h_pp["mean"],
            "hn_per_patient_max":           h_pp["max"],
            "n_no_positive_patients":       r["n_no_positive_patients"],
            "no_positive_patients":         ";".join(r["no_positive_patients"]),
            "nsclc_positive":               gs_nsclc.get("positive", ""),
            "nsclc_positive_ratio":         gs_nsclc.get("positive_ratio", ""),
            "msd_positive":                 gs_msd.get("positive", ""),
            "msd_positive_ratio":           gs_msd.get("positive_ratio", ""),
        })
    pd.DataFrame(rows).to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"    summary CSV: {OUT_SUMMARY_CSV}")

    # summary JSON
    summary_json = {
        "generated":                "2026-05-24",
        "c4_pool":                  "C4_all_p95_slices_top10_patch",
        "c4_total_candidates":      len(c4_df),
        "c4_positive_count":        n_c4_pos,
        "c4_negative_count":        n_c4_neg,
        "stage1_dev_n_patients":    154,
        "stage2_holdout_sealed":    True,
        "existing_results_modified": False,
        "manifest_only":            True,
        "s4_config": {
            "hard_negative_rule":          f"min(pos_count * {S4_HN_MULTIPLIER}, {S4_HN_CAP})",
            "no_positive_patient_hn_cap":  S4_NO_POS_HN_CAP,
        },
        "variants": all_results,
    }
    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    print(f"    summary JSON: {OUT_SUMMARY_JSON}")

    # summary MD
    md_text = write_summary_md(all_results, len(c4_df), n_c4_pos, n_c4_neg)
    OUT_SUMMARY_MD.write_text(md_text, encoding="utf-8")
    print(f"    summary MD: {OUT_SUMMARY_MD}")

    print("\n[8/8] 완료. 생성 파일:")
    for out_file in [OUT_MANIFEST, OUT_SUMMARY_CSV, OUT_SUMMARY_JSON, OUT_SUMMARY_MD]:
        print(f"  {out_file}")


if __name__ == "__main__":
    main()
