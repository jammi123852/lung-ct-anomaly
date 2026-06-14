"""
rule_s6a_manifest_gen.py

GS2_slice_top30 후보 pool에서 S6-A_positive_all_hn_ratio2 sampling으로
최종 후보 manifest를 생성한다.

실행:
  --run 없이: preflight 체크만 수행 후 종료
  --run: 실제 manifest + summary 저장

절대 금지:
- crop/npz/PNG 생성 금지
- 모델 학습 / scoring 재실행 금지
- 기존 score/candidate/evaluation/crop 파일 수정/덮어쓰기 금지
- stage2_holdout 환자 분석 금지
- weak 환자 전용 예외 로직 금지
- lesion local_z 직접 후보 추가 금지
- 병변 mask를 후보 pool 생성에 사용 금지
- pip/conda install 금지
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

DIAG_CSV = REPO_ROOT / "outputs/position-aware-padim-v1/reports_v2_roi0_0_lesion/ratio_adjusted_score_full_diagnostic.csv"
STAGE_SPLIT_CSV = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
S6_SUMMARY_JSON = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/rule_s6_gs2_sampling_design_summary.json"

OUT_CAND_DIR = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/candidates"
OUT_RPT_DIR = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports"

OUT_MANIFEST = OUT_CAND_DIR / "rule_s6a_gs2_selected_candidate_manifest_dryrun.csv"
OUT_SUMMARY_CSV = OUT_RPT_DIR / "rule_s6a_gs2_selected_candidate_manifest_summary.csv"
OUT_SUMMARY_JSON = OUT_RPT_DIR / "rule_s6a_gs2_selected_candidate_manifest_summary.json"
OUT_SUMMARY_MD = OUT_RPT_DIR / "rule_s6a_gs2_selected_candidate_manifest_summary.md"

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
V2V2_P95_THRESHOLD = 14.092057666455288
DEDUP_KEYS = ["patient_id", "local_z", "y0", "x0", "y1", "x1"]
WEAK_PATIENTS = ["LUNG1-156", "LUNG1-415", "MSD_lung_071", "MSD_lung_096", "MSD_lung_079"]
CHUNKSIZE = 200_000
EXPLOSION_THRESHOLD = 2000
SAMPLING_RULE = "S6-A_positive_all_hn_ratio2"

REQUIRED_COLS = [
    "patient_id", "local_z", "y0", "x0", "y1", "x1",
    "model_type",
    "score_original", "score_valid950_weighted", "score_valid950_pow025", "score_valid950_soft",
    "lesion_patch_ratio", "position_bin", "z_level", "central_peripheral",
]
OPTIONAL_COLS = [
    "slice_index",
    "group", "dataset", "source",
    "roi_inside_ratio", "air_ratio_950", "air_ratio_970",
    "valid_ratio_roi_air950", "valid_ratio_roi_air970",
    "lesion_pixels", "patch_label", "lesion_overlap",
    "sampling_label",
]

# ---------------------------------------------------------------------------
# Step 0: Guard 체크
# ---------------------------------------------------------------------------
def guard_check() -> None:
    for f in [OUT_MANIFEST, OUT_SUMMARY_CSV, OUT_SUMMARY_JSON, OUT_SUMMARY_MD]:
        if f.exists():
            print(f"[중단] 출력 파일 이미 존재: {f}")
            print("  기존 파일을 삭제하거나 이름을 바꾼 후 재실행하세요.")
            sys.exit(1)

    missing = []
    for f in [DIAG_CSV, STAGE_SPLIT_CSV, S6_SUMMARY_JSON]:
        if not f.exists():
            missing.append(str(f))
    if missing:
        print("[중단] 입력 파일 없음:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print("[Guard] 모든 입력 파일 존재 확인 완료")


# ---------------------------------------------------------------------------
# Step 1: stage split 로드
# ---------------------------------------------------------------------------
def load_stage_split() -> tuple[set[str], set[str]]:
    df = pd.read_csv(STAGE_SPLIT_CSV, encoding="utf-8-sig")
    if "stage_split" not in df.columns:
        print(f"[중단] stage_split 컬럼 없음. 실제 컬럼: {list(df.columns)}")
        sys.exit(1)

    dev = set(df[df["stage_split"] == "stage1_dev"]["patient_id"].tolist())
    holdout = set(df[df["stage_split"] == "stage2_holdout"]["patient_id"].tolist())

    print(f"[Stage Split] stage1_dev: {len(dev)}명, stage2_holdout: {len(holdout)}명")
    if len(dev) != 154:
        print(f"  [경고] stage1_dev 기대 154명, 실제 {len(dev)}명")

    return dev, holdout


# ---------------------------------------------------------------------------
# Step 2: ratio_adjusted_score_full_diagnostic.csv chunk 로드
# ---------------------------------------------------------------------------
def load_diag_filtered(dev_patients: set[str], holdout_patients: set[str]) -> pd.DataFrame:
    print(f"\n[Step 2] {DIAG_CSV.name} chunk 로드 시작 (chunksize={CHUNKSIZE:,})")

    first_chunk = pd.read_csv(DIAG_CSV, nrows=1, encoding="utf-8-sig")
    missing_cols = [c for c in REQUIRED_COLS if c not in first_chunk.columns]
    if missing_cols:
        print(f"[중단] 필수 컬럼 누락: {missing_cols}")
        sys.exit(1)
    print("  필수 컬럼 확인 완료")

    actual_optional = [c for c in OPTIONAL_COLS if c in first_chunk.columns]
    load_cols = REQUIRED_COLS + actual_optional
    print(f"  로드할 optional 컬럼: {actual_optional}")

    chunks = []
    total_read = 0
    total_filtered = 0

    for i, chunk in enumerate(
        pd.read_csv(DIAG_CSV, chunksize=CHUNKSIZE, encoding="utf-8-sig", low_memory=False)
    ):
        total_read += len(chunk)
        filtered = chunk[
            (chunk["model_type"] == "v2v2") &
            (chunk["patient_id"].isin(dev_patients))
        ]
        if len(filtered) > 0:
            avail_cols = [c for c in load_cols if c in filtered.columns]
            chunks.append(filtered[avail_cols])
            total_filtered += len(filtered)
        if (i + 1) % 10 == 0:
            print(f"  chunk {i+1} 처리 중... 읽은 행: {total_read:,}, 필터된 행: {total_filtered:,}")

    print(f"  전체 읽은 행: {total_read:,}, 필터 후: {total_filtered:,}")

    if not chunks:
        print("[중단] 필터 후 데이터 없음")
        sys.exit(1)

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    before = len(df)
    df = df.drop_duplicates(subset=DEDUP_KEYS)
    after = len(df)
    print(f"  중복 제거: {before:,} → {after:,} (제거 {before - after:,})")

    holdout_in_data = set(df["patient_id"].unique()) & holdout_patients
    if holdout_in_data:
        print(f"[중단] stage2_holdout 환자가 데이터에 포함됨: {holdout_in_data}")
        sys.exit(1)
    print("  stage2_holdout 봉인 확인 완료")

    if after == 0:
        print("[중단] 중복 제거 후 0행")
        sys.exit(1)

    df["stage_split"] = "stage1_dev"

    print(f"  환자 수: {df['patient_id'].nunique()}")
    print(f"  model_type 값: {df['model_type'].unique().tolist()}")
    return df


# ---------------------------------------------------------------------------
# Step 3: Rank score 계산
# ---------------------------------------------------------------------------
def compute_rank_scores(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 3] Rank score 계산 시작")

    def rank_percentile(series: pd.Series) -> pd.Series:
        r = series.rank(method="min", ascending=False)
        n = len(series)
        if n <= 1:
            return pd.Series(1.0, index=series.index)
        return 1.0 - (r - 1) / (n - 1)

    df["patient_rank_original"] = df.groupby("patient_id")["score_original"].transform(rank_percentile)
    df["slice_rank_original"] = df.groupby(["patient_id", "local_z"])["score_original"].transform(rank_percentile)
    df["patient_rank_valid950"] = df.groupby("patient_id")["score_valid950_weighted"].transform(rank_percentile)
    df["slice_rank_valid950"] = df.groupby(["patient_id", "local_z"])["score_valid950_weighted"].transform(rank_percentile)

    df["composite_rank_v2"] = (
        0.4 * df["patient_rank_original"]
        + 0.3 * df["slice_rank_original"]
        + 0.2 * df["patient_rank_valid950"]
        + 0.1 * df["slice_rank_valid950"]
    )

    print("  rank 컬럼 4개 + composite_rank_v2 계산 완료")
    return df


# ---------------------------------------------------------------------------
# Step 4: GS2 pool 구성 (후보 pool 생성에 병변 label 미사용)
# ---------------------------------------------------------------------------
def build_gs2_mask(df: pd.DataFrame) -> pd.Series:
    print("\n[Step 4] GS2 pool 구성 시작")

    g0_mask = df["score_original"] >= V2V2_P95_THRESHOLD
    n_g0 = int(g0_mask.sum())
    print(f"  G0_original_p95 후보 수: {n_g0:,}")

    slice_top30_mask = pd.Series(False, index=df.index)
    for (pid, lz), sub in df.groupby(["patient_id", "local_z"]):
        top_idx = sub["composite_rank_v2"].nlargest(min(30, len(sub))).index
        slice_top30_mask.loc[top_idx] = True

    gs2_mask = g0_mask | slice_top30_mask
    n_gs2 = int(gs2_mask.sum())
    print(f"  GS2 pool 후보 수: {n_gs2:,} (G0 union slice top30)")

    expected_gs2 = 986701
    if n_gs2 != expected_gs2:
        print(f"  [경고] GS2 pool 기대 {expected_gs2:,}, 실제 {n_gs2:,}")

    return gs2_mask


# ---------------------------------------------------------------------------
# Step 5: Positive 판정 (sampling 단계에서만 사용)
# ---------------------------------------------------------------------------
def is_positive(df_sub: pd.DataFrame) -> pd.Series:
    flag = pd.Series(False, index=df_sub.index)
    if "lesion_patch_ratio" in df_sub.columns:
        flag = flag | (df_sub["lesion_patch_ratio"].fillna(0) > 0)
    if "patch_label" in df_sub.columns:
        flag = flag | (df_sub["patch_label"].fillna(0) == 1)
    if "lesion_overlap" in df_sub.columns:
        flag = flag | df_sub["lesion_overlap"].fillna(False).astype(bool)
    if "sampling_label" in df_sub.columns:
        flag = flag | (df_sub["sampling_label"].fillna("") == "positive")
    return flag


# ---------------------------------------------------------------------------
# Step 6: S6-A Sampling
# ---------------------------------------------------------------------------
def sample_s6a(df_pool: pd.DataFrame) -> pd.Series:
    """S6-A: positive 전부 + hard_negative ratio 2배, 환자별 cap=600."""
    hn_ratio = 2.0
    patient_hn_cap = 600

    pos_mask = is_positive(df_pool)
    pos_idx = set(df_pool[pos_mask].index.tolist())
    n_total_pos = len(pos_idx)
    target_total_hn = int(n_total_pos * hn_ratio)

    print(f"  S6-A: positive {n_total_pos:,}개, target_hn {target_total_hn:,}개 (ratio={hn_ratio}, cap={patient_hn_cap})")

    hn_df = df_pool[~pos_mask]

    per_patient_hn = []
    for pid, sub in hn_df.groupby("patient_id"):
        top_hn = sub.sort_values("composite_rank_v2", ascending=False).head(patient_hn_cap)
        per_patient_hn.append(top_hn)

    if per_patient_hn:
        all_hn = pd.concat(per_patient_hn)
        selected_hn_idx = set(
            all_hn.sort_values("composite_rank_v2", ascending=False)
            .head(target_total_hn)
            .index.tolist()
        )
    else:
        selected_hn_idx = set()

    selected_idx = pos_idx | selected_hn_idx
    return df_pool.index.isin(selected_idx)


# ---------------------------------------------------------------------------
# Step 7: sampling_label / sampling_rule 컬럼 부여
# ---------------------------------------------------------------------------
def assign_sampling_label(df_sampled: pd.DataFrame) -> pd.DataFrame:
    pos_mask = is_positive(df_sampled)
    df_out = df_sampled.copy()
    df_out["sampling_label"] = "hard_negative"
    df_out.loc[pos_mask, "sampling_label"] = "positive"
    df_out["sampling_rule"] = SAMPLING_RULE
    return df_out


# ---------------------------------------------------------------------------
# Step 8: 검증 지표 계산
# ---------------------------------------------------------------------------
def get_patient_size_bin(df_full: pd.DataFrame) -> pd.Series:
    pos_mask = df_full["lesion_patch_ratio"].fillna(0) > 0
    patient_lesion_counts = df_full[pos_mask].groupby("patient_id").size()
    if len(patient_lesion_counts) < 4:
        return pd.Series(dtype=str)
    q25 = patient_lesion_counts.quantile(0.25)
    q50 = patient_lesion_counts.quantile(0.50)
    q75 = patient_lesion_counts.quantile(0.75)

    def size_bin(val: float) -> str:
        if val <= q25:
            return "tiny"
        elif val <= q50:
            return "small"
        elif val <= q75:
            return "medium"
        else:
            return "large"

    return patient_lesion_counts.apply(size_bin)


def compute_manifest_metrics(
    df_sampled: pd.DataFrame,
    df_full: pd.DataFrame,
    patient_size_bin: pd.Series,
) -> dict:
    pos_mask_full = df_full["lesion_patch_ratio"].fillna(0) > 0
    n_lesion_total = int(pos_mask_full.sum())
    lesion_patients_full = set(df_full[pos_mask_full]["patient_id"].unique())

    pos_mask_s = df_sampled["sampling_label"] == "positive"
    n_candidates = len(df_sampled)
    n_positive = int(pos_mask_s.sum())
    n_hn = int((~pos_mask_s).sum())
    n_patients = df_sampled["patient_id"].nunique()

    per_patient = df_sampled.groupby("patient_id").size()
    per_patient_stats = {
        "min": int(per_patient.min()) if len(per_patient) > 0 else 0,
        "median": float(per_patient.median()) if len(per_patient) > 0 else 0.0,
        "mean": float(per_patient.mean()) if len(per_patient) > 0 else 0.0,
        "max": int(per_patient.max()) if len(per_patient) > 0 else 0,
    }
    explosion_patients = sorted(per_patient[per_patient > EXPLOSION_THRESHOLD].index.tolist())

    hit_patients = set(df_sampled[pos_mask_s]["patient_id"].unique())
    nohit_patients = sorted(lesion_patients_full - hit_patients)
    patient_hit_rate = float(len(hit_patients) / len(lesion_patients_full)) if lesion_patients_full else 0.0

    lesion_patch_recall = float(n_positive / n_lesion_total) if n_lesion_total > 0 else 0.0

    lung140_detail: dict = {}
    if "LUNG1-140" in df_sampled["patient_id"].values:
        lung140 = df_sampled[df_sampled["patient_id"] == "LUNG1-140"]
        lung140_pos = int((lung140["sampling_label"] == "positive").sum())
        lung140_hn = int((lung140["sampling_label"] == "hard_negative").sum())
        if lung140_pos > EXPLOSION_THRESHOLD:
            explosion_cause = "positive"
        elif lung140_hn > EXPLOSION_THRESHOLD:
            explosion_cause = "hard_negative"
        else:
            explosion_cause = "combined"
        lung140_detail = {
            "total": len(lung140),
            "positive": lung140_pos,
            "hard_negative": lung140_hn,
            "explosion_cause": explosion_cause,
        }

    weak_hit: dict = {}
    for wp in WEAK_PATIENTS:
        wp_lesion = df_full[
            (df_full["patient_id"] == wp) &
            (df_full["lesion_patch_ratio"].fillna(0) > 0)
        ]
        wp_s = df_sampled[df_sampled["patient_id"] == wp]
        wp_hits = int((wp_s["sampling_label"] == "positive").sum())
        weak_hit[wp] = bool(wp_hits > 0) if len(wp_lesion) > 0 else None

    size_recall: dict = {}
    for bin_name in ["tiny", "small", "medium", "large"]:
        bin_patients = set(patient_size_bin[patient_size_bin == bin_name].index)
        bin_lesion = df_full[
            (df_full["patient_id"].isin(bin_patients)) &
            (df_full["lesion_patch_ratio"].fillna(0) > 0)
        ]
        bin_s = df_sampled[df_sampled["patient_id"].isin(bin_patients)]
        n_bin_lesion = len(bin_lesion)
        n_bin_hit = int((bin_s["sampling_label"] == "positive").sum())
        size_recall[bin_name] = float(n_bin_hit / n_bin_lesion) if n_bin_lesion > 0 else None

    return {
        "sampling_rule": SAMPLING_RULE,
        "n_candidates": n_candidates,
        "n_positive": n_positive,
        "n_hard_negative": n_hn,
        "n_patients": n_patients,
        "n_lesion_total_full": n_lesion_total,
        "lesion_patch_recall": round(lesion_patch_recall, 6),
        "patient_hit_rate": round(patient_hit_rate, 6),
        "n_nohit": len(nohit_patients),
        "nohit_patients": nohit_patients,
        "weak_patients_hit": weak_hit,
        "patient_lesion_size_recall": size_recall,
        "per_patient_stats": per_patient_stats,
        "explosion_patients": explosion_patients,
        "lung1_140_detail": lung140_detail,
        "expected": {
            "n_candidates": 130659,
            "n_positive": 43553,
            "n_hard_negative": 87106,
            "patient_hit_rate": 1.0,
            "n_nohit": 0,
        },
    }


# ---------------------------------------------------------------------------
# Step 9: manifest 컬럼 순서 결정
# ---------------------------------------------------------------------------
def build_manifest_columns(df_sampled: pd.DataFrame) -> list[str]:
    base = ["patient_id", "local_z"]
    for c in ["slice_index"]:
        if c in df_sampled.columns:
            base.append(c)
    base += ["y0", "x0", "y1", "x1", "sampling_rule", "sampling_label", "model_type", "stage_split"]
    for c in ["group", "dataset", "source"]:
        if c in df_sampled.columns:
            base.append(c)
    base += [
        "score_original", "score_valid950_weighted", "score_valid950_soft",
        "composite_rank_v2", "patient_rank_original", "slice_rank_original",
        "patient_rank_valid950", "slice_rank_valid950",
        "lesion_patch_ratio",
    ]
    for c in ["lesion_pixels", "patch_label", "lesion_overlap"]:
        if c in df_sampled.columns:
            base.append(c)
    base += ["position_bin", "z_level", "central_peripheral"]
    for c in df_sampled.columns:
        if any(c.startswith(prefix) for prefix in ["roi_inside", "air_ratio", "valid_ratio"]):
            if c not in base:
                base.append(c)
    return [c for c in base if c in df_sampled.columns]


# ---------------------------------------------------------------------------
# Step 10: 출력 저장
# ---------------------------------------------------------------------------
def save_outputs(df_sampled: pd.DataFrame, metrics: dict) -> None:
    OUT_CAND_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RPT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_cols = build_manifest_columns(df_sampled)
    df_sampled[manifest_cols].to_csv(OUT_MANIFEST, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_MANIFEST} ({len(df_sampled):,}행)")

    m = metrics
    exp = m.get("expected", {})
    flat = {
        "sampling_rule": SAMPLING_RULE,
        "n_candidates": m["n_candidates"],
        "n_positive": m["n_positive"],
        "n_hard_negative": m["n_hard_negative"],
        "n_patients": m["n_patients"],
        "lesion_patch_recall": m["lesion_patch_recall"],
        "patient_hit_rate": m["patient_hit_rate"],
        "n_nohit": m["n_nohit"],
        "per_patient_min": m["per_patient_stats"]["min"],
        "per_patient_median": m["per_patient_stats"]["median"],
        "per_patient_mean": m["per_patient_stats"]["mean"],
        "per_patient_max": m["per_patient_stats"]["max"],
        "explosion_count": len(m["explosion_patients"]),
        "match_n_candidates": m["n_candidates"] == exp.get("n_candidates"),
        "match_n_positive": m["n_positive"] == exp.get("n_positive"),
        "match_n_hard_negative": m["n_hard_negative"] == exp.get("n_hard_negative"),
    }
    for wp in WEAK_PATIENTS:
        flat[f"weak_hit_{wp}"] = m["weak_patients_hit"].get(wp)
    for size in ["tiny", "small", "medium", "large"]:
        flat[f"size_recall_{size}"] = m["patient_lesion_size_recall"].get(size)
    d140 = m.get("lung1_140_detail", {})
    flat["lung1_140_total"] = d140.get("total")
    flat["lung1_140_positive"] = d140.get("positive")
    flat["lung1_140_hard_negative"] = d140.get("hard_negative")
    flat["lung1_140_explosion_cause"] = d140.get("explosion_cause")

    pd.DataFrame([flat]).to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_SUMMARY_CSV}")

    def safe_json(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: safe_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_json(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(safe_json(metrics), f, ensure_ascii=False, indent=2)
    print(f"저장: {OUT_SUMMARY_JSON}")

    match_n = m["n_candidates"] == exp.get("n_candidates")
    match_pos = m["n_positive"] == exp.get("n_positive")
    match_hn = m["n_hard_negative"] == exp.get("n_hard_negative")
    match_hit = m["patient_hit_rate"] == 1.0
    match_nohit = m["n_nohit"] == 0

    lines = ["# Rule S6-A Manifest 생성 보고서\n"]
    lines.append(f"**sampling_rule**: {SAMPLING_RULE}\n")
    lines.append("## 검증 결과\n")
    lines.append("| 항목 | 기대값 | 실제값 | 일치 |")
    lines.append("|------|--------|--------|------|")
    lines.append(f"| n_candidates | {exp.get('n_candidates', 'N/A'):,} | {m['n_candidates']:,} | {'OK' if match_n else 'MISMATCH'} |")
    lines.append(f"| n_positive | {exp.get('n_positive', 'N/A'):,} | {m['n_positive']:,} | {'OK' if match_pos else 'MISMATCH'} |")
    lines.append(f"| n_hard_negative | {exp.get('n_hard_negative', 'N/A'):,} | {m['n_hard_negative']:,} | {'OK' if match_hn else 'MISMATCH'} |")
    lines.append(f"| n_patients | 154 | {m['n_patients']} | {'OK' if m['n_patients'] == 154 else 'MISMATCH'} |")
    lines.append(f"| patient_hit_rate | 1.000000 | {m['patient_hit_rate']:.6f} | {'OK' if match_hit else 'MISMATCH'} |")
    lines.append(f"| n_nohit | 0 | {m['n_nohit']} | {'OK' if match_nohit else 'MISMATCH'} |\n")

    lines.append("## Weak Patient 회수\n")
    lines.append("| 환자 | hit |")
    lines.append("|------|-----|")
    for wp, hit in m["weak_patients_hit"].items():
        lines.append(f"| {wp} | {hit} |")

    lines.append("\n## 병변 크기별 Recall\n")
    lines.append("| size | recall |")
    lines.append("|------|--------|")
    for size, val in m["patient_lesion_size_recall"].items():
        v_str = f"{float(val):.6f}" if val is not None else "N/A"
        lines.append(f"| {size} | {v_str} |")

    lines.append("\n## 환자별 후보 수\n")
    pp = m["per_patient_stats"]
    lines.append(f"- min: {pp['min']}")
    lines.append(f"- median: {pp['median']:.1f}")
    lines.append(f"- mean: {pp['mean']:.1f}")
    lines.append(f"- max: {pp['max']}")

    lines.append(f"\n## 폭주 환자 (>{EXPLOSION_THRESHOLD}개)\n")
    ep = m["explosion_patients"]
    lines.append(f"- 폭주 환자 수: {len(ep)}")
    if ep:
        lines.append(f"- 목록: {ep}")

    lines.append("\n## LUNG1-140 상세\n")
    d = m.get("lung1_140_detail", {})
    if d:
        lines.append(f"- 총 후보 수: {d.get('total', 'N/A')}")
        lines.append(f"- positive: {d.get('positive', 'N/A')}")
        lines.append(f"- hard_negative: {d.get('hard_negative', 'N/A')}")
        lines.append(f"- 폭주 원인: {d.get('explosion_cause', 'N/A')}")
    else:
        lines.append("- LUNG1-140 데이터 없음")

    if m["n_nohit"] > 0:
        lines.append("\n## No-hit 환자 목록\n")
        for p in m["nohit_patients"]:
            lines.append(f"- {p}")

    OUT_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {OUT_SUMMARY_MD}")


# ---------------------------------------------------------------------------
# Preflight 보고
# ---------------------------------------------------------------------------
def preflight_report(dev_patients: set[str]) -> None:
    print("\n=== Preflight 보고 ===")
    diag_size_gb = DIAG_CSV.stat().st_size / 1e9
    print(f"입력 파일: {DIAG_CSV.name} ({diag_size_gb:.1f}GB)")
    print(f"stage1_dev 환자 수: {len(dev_patients)}명")
    print(f"\n생성 예정 파일:")
    print(f"  manifest : {OUT_MANIFEST}")
    print(f"  summary CSV : {OUT_SUMMARY_CSV}")
    print(f"  summary JSON: {OUT_SUMMARY_JSON}")
    print(f"  summary MD  : {OUT_SUMMARY_MD}")
    print(f"\n예상 manifest 행 수 : ~130,659행")
    print(f"예상 소요 시간       : 15~40분")
    print(f"메모리 위험          : 중간 (전체 df + rank 컬럼 + GS2 mask, ~4~6GB)")
    print(f"\nS6-A 정의:")
    print(f"  - GS2 pool : G0_original_p95 (score_original >= {V2V2_P95_THRESHOLD}) | slice top30")
    print(f"  - sampling : positive 전부 + hn x2, patient_hn_cap=600")
    print(f"  - 후보 pool 생성에 병변 label 미사용")
    print(f"  - sampling 단계에서만 lesion_patch_ratio 기반 positive 구분")
    print(f"\n기존 파일 영향:")
    print(f"  - 기존 score/candidate/evaluation/crop 파일 수정 없음")
    print(f"  - 신규 파일 4개만 생성 (manifest 1개 + summary 3개)")
    print(f"\n실행 명령:")
    print(f"  /home/jinhy/ai_env/bin/python3 scripts/rule_s6a_manifest_gen.py --run")
    print(f"\n[Preflight 완료] 실제 실행은 --run 플래그 추가 후 진행하세요.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rule S6-A Manifest Gen (--run 없으면 preflight만)"
    )
    parser.add_argument("--run", action="store_true", help="실제 manifest 생성 실행")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    guard_check()
    dev_patients, holdout_patients = load_stage_split()

    if not args.run:
        preflight_report(dev_patients)
        return

    df = load_diag_filtered(dev_patients, holdout_patients)
    df = compute_rank_scores(df)
    gc.collect()

    gs2_mask = build_gs2_mask(df)
    df_gs2 = df.loc[gs2_mask].copy()
    print(f"\n[GS2 pool] 총 {len(df_gs2):,}행, {df_gs2['patient_id'].nunique()}명")

    print("\n[Step 6] S6-A sampling 시작")
    sampled_mask = sample_s6a(df_gs2)
    df_sampled = df_gs2.loc[sampled_mask].copy()
    del df_gs2
    gc.collect()
    print(f"  sampled: {len(df_sampled):,}행")

    df_sampled = assign_sampling_label(df_sampled)

    patient_size_bin = get_patient_size_bin(df)
    metrics = compute_manifest_metrics(df_sampled, df, patient_size_bin)
    del df
    gc.collect()

    print(f"\n[검증]")
    print(f"  n_candidates    : {metrics['n_candidates']:,} (기대 130,659)")
    print(f"  n_positive      : {metrics['n_positive']:,} (기대 43,553)")
    print(f"  n_hard_negative : {metrics['n_hard_negative']:,} (기대 87,106)")
    print(f"  patient_hit_rate: {metrics['patient_hit_rate']:.6f} (기대 1.000000)")
    print(f"  n_nohit         : {metrics['n_nohit']} (기대 0)")
    d140 = metrics.get("lung1_140_detail", {})
    if d140:
        print(
            f"  LUNG1-140       : total={d140.get('total')}, "
            f"pos={d140.get('positive')}, hn={d140.get('hard_negative')}, "
            f"cause={d140.get('explosion_cause')}"
        )

    save_outputs(df_sampled, metrics)
    print(f"\n=== 완료 ===\n출력 디렉토리: {OUT_CAND_DIR}")


if __name__ == "__main__":
    main()
