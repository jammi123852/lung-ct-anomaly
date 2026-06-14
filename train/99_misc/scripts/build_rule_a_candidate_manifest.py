"""
build_rule_a_candidate_manifest.py
Rule A: p95 threshold candidate manifest-only dry-run
- stage1_dev 154명만 사용
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
SPLIT_CSV      = REPO / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
SCORE_DIR      = REPO / "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/lesion_v2_by_patient"
THRESHOLD_JSON = REPO / "outputs/position-aware-padim-v1/evaluation/normal_v2_roi0_0/normal_v2_threshold.json"
SCREENING_CSV  = REPO / "outputs/position-aware-padim-v1/evaluation/lesion_subset_v2_model_v2/per_patient_screening.csv"
HIT_CSV        = REPO / "outputs/position-aware-padim-v1/reports_v2_roi0_0_lesion/lesion_hit_overlap_by_patient.csv"

# ── 출력 경로 ──────────────────────────────────────────────────────────────
OUT_CAND_DIR   = REPO / "outputs/second-stage-lesion-refiner-v1/candidates"
OUT_REPORT_DIR = REPO / "outputs/second-stage-lesion-refiner-v1/reports"
OUT_CSV        = OUT_CAND_DIR / "rule_a_p95_stage1_dev_candidate_manifest_dryrun.csv"
OUT_JSON       = OUT_REPORT_DIR / "rule_a_p95_stage1_dev_candidate_summary.json"
OUT_MD         = OUT_REPORT_DIR / "rule_a_p95_stage1_dev_candidate_summary.md"

# ── Guard 1: 출력 파일 이미 있으면 중단 ────────────────────────────────────
for p in [OUT_CSV, OUT_JSON, OUT_MD]:
    if p.exists():
        print(f"[ABORT] 출력 파일이 이미 존재합니다: {p}")
        sys.exit(1)

# ── 입력 로드 ──────────────────────────────────────────────────────────────
split_df = pd.read_csv(SPLIT_CSV)

# ── Guard 2: stage1_dev 154명 확인 ─────────────────────────────────────────
stage1_dev_ids = split_df.loc[split_df["stage_split"] == "stage1_dev", "patient_id"].tolist()
stage2_holdout_ids = set(split_df.loc[split_df["stage_split"] == "stage2_holdout", "patient_id"].tolist())
if len(stage1_dev_ids) != 154:
    print(f"[ABORT] stage1_dev 환자 수가 154명이 아님: {len(stage1_dev_ids)}명")
    sys.exit(1)
print(f"[OK] stage1_dev: {len(stage1_dev_ids)}명")

# ── Guard 3: threshold_p95 읽기 확인 ──────────────────────────────────────
with open(THRESHOLD_JSON) as f:
    thr_data = json.load(f)
if "threshold_p95" not in thr_data:
    print(f"[ABORT] threshold_p95가 {THRESHOLD_JSON}에 없음")
    sys.exit(1)
threshold_p95 = float(thr_data["threshold_p95"])
print(f"[OK] threshold_p95 = {threshold_p95}")

# ── Guard 4: score CSV 누락 환자 확인 ─────────────────────────────────────
score_files = {p.stem: p for p in SCORE_DIR.glob("*.csv")}
missing = [pid for pid in stage1_dev_ids if pid not in score_files]
if missing:
    print(f"[ABORT] score CSV 누락 환자 {len(missing)}명: {missing[:5]}...")
    sys.exit(1)
print(f"[OK] score CSV 존재 확인: {len(stage1_dev_ids)}명")

# ── 수정1: patient_id / group / stage_split 강제 주입 ──────────────────────
# score CSV 내부 patient_id를 신뢰하지 않고 파일명 기준 pid 강제 설정
# group, stage_split은 split CSV 기준으로 강제 주입
group_map = split_df.set_index("patient_id")["group"].to_dict()
split_map = split_df.set_index("patient_id")["stage_split"].to_dict()

print("score CSV 로드 중...")
chunks = []
for pid in stage1_dev_ids:
    df = pd.read_csv(score_files[pid], low_memory=False)
    df["patient_id"] = pid                  # 파일명 기준 강제 주입
    df["group"] = group_map[pid]            # split CSV 기준 강제 주입
    df["stage_split"] = split_map[pid]      # split CSV 기준 강제 주입
    chunks.append(df)
all_scores = pd.concat(chunks, ignore_index=True)
print(f"  전체 patch 수: {len(all_scores):,}")

# ── Guard 5: padim_score NaN/Inf 확인 ─────────────────────────────────────
n_nan = all_scores["padim_score"].isna().sum()
n_inf = np.isinf(all_scores["padim_score"]).sum()
if n_nan > 0 or n_inf > 0:
    print(f"[ABORT] padim_score NaN={n_nan}, Inf={n_inf}")
    sys.exit(1)
print(f"[OK] padim_score NaN=0, Inf=0")

# ── Guard 6: patch_label 컬럼 존재 확인 ──────────────────────────────────
if "patch_label" not in all_scores.columns:
    print(f"[ABORT] patch_label 컬럼 없음")
    sys.exit(1)
print(f"[OK] patch_label 컬럼 존재")

# ── 수정2: 필수 컬럼 guard 강화 ───────────────────────────────────────────
# slice_index / local_z 중 최소 하나 이상 필요
slice_col_present = "slice_index" in all_scores.columns or "local_z" in all_scores.columns
REQUIRED_COLS = [
    "patient_id", "group", "stage_split",
    "y0", "x0", "y1", "x1",
    "padim_score", "patch_label",
    "lesion_pixels", "lesion_patch_ratio", "has_lesion_patch",
    "roi_0_0_patch_ratio",
    "position_bin", "z_level", "central_peripheral",
]
missing_req = [c for c in REQUIRED_COLS if c not in all_scores.columns]
if missing_req:
    print(f"[ABORT] 필수 컬럼 없음: {missing_req}")
    sys.exit(1)
if not slice_col_present:
    print(f"[ABORT] slice_index와 local_z 둘 다 없음")
    sys.exit(1)
print(f"[OK] 필수 컬럼 모두 존재")

# ── Rule A 필터링: padim_score >= threshold_p95 ─────────────────────────
cand = all_scores[all_scores["padim_score"] >= threshold_p95].copy()
print(f"[OK] Rule A candidate 수: {len(cand):,} / 전체 {len(all_scores):,}")

# ── 수정5: candidate 0개면 중단 ──────────────────────────────────────────
if len(cand) == 0:
    print("[ABORT] Rule A candidate가 0개입니다. CSV 저장하지 않음.")
    sys.exit(1)

# ── Guard 7: stage2_holdout 포함 여부 확인 ───────────────────────────────
holdout_in_cand = set(cand["patient_id"].unique()) & stage2_holdout_ids
if holdout_in_cand:
    print(f"[ABORT] stage2_holdout 환자가 candidate에 포함됨: {holdout_in_cand}")
    sys.exit(1)
# stage_split 컬럼이 stage1_dev만 있는지 이중 확인
bad_split = cand[cand["stage_split"] != "stage1_dev"]
if len(bad_split) > 0:
    print(f"[ABORT] stage1_dev가 아닌 행이 {len(bad_split)}개 포함됨: {bad_split['stage_split'].unique()}")
    sys.exit(1)
print(f"[OK] stage2_holdout 봉인 준수")

# ── 파생 컬럼 추가 ────────────────────────────────────────────────────────
cand["candidate_rule"] = "rule_a_p95"
cand["threshold_used"] = threshold_p95
cand["lesion_overlap"] = (cand["patch_label"] == 1)

# candidate_rank_in_patient: 환자별 padim_score 내림차순
cand = cand.sort_values(["patient_id", "padim_score"], ascending=[True, False])
cand["candidate_rank_in_patient"] = cand.groupby("patient_id").cumcount() + 1

# ── 수정3: candidate_rank_in_slice 안전 처리 ─────────────────────────────
if "slice_index" in cand.columns:
    rank_slice_col = "slice_index"
elif "local_z" in cand.columns:
    rank_slice_col = "local_z"
else:
    print("[ABORT] slice_index와 local_z 둘 다 없음")
    sys.exit(1)

cand = cand.sort_values(["patient_id", rank_slice_col, "padim_score"], ascending=[True, True, False])
cand["candidate_rank_in_slice"] = cand.groupby(["patient_id", rank_slice_col]).cumcount() + 1

# ── 컬럼 순서 정리 ────────────────────────────────────────────────────────
base_cols = [
    "patient_id", "group", "stage_split",
    "local_z", "slice_index",
    "y0", "x0", "y1", "x1",
    "padim_score", "patch_label",
    "lesion_pixels", "lesion_patch_ratio", "has_lesion_patch",
    "roi_0_0_patch_ratio",
    "position_bin", "z_level", "central_peripheral",
    "candidate_rule", "threshold_used",
    "lesion_overlap",
    "candidate_rank_in_patient", "candidate_rank_in_slice",
]
# grid_position_bin이 있으면 포함 (없어도 중단 안 함)
if "grid_position_bin" in cand.columns:
    insert_idx = base_cols.index("z_level")
    base_cols.insert(insert_idx, "grid_position_bin")

# 컬럼 누락 시 sys.exit(1) (필수 컬럼은 위에서 이미 확인했으나, local_z/slice_index 중 없는 것은 제외)
optional_missing = [c for c in base_cols if c not in cand.columns]
if optional_missing:
    # local_z 또는 slice_index 중 하나만 없는 경우는 허용
    non_optional_missing = [c for c in optional_missing if c not in ("local_z", "slice_index")]
    if non_optional_missing:
        print(f"[ABORT] 출력 컬럼 누락: {non_optional_missing}")
        sys.exit(1)
    print(f"[INFO] 출력에서 제외 (선택 컬럼): {optional_missing}")
    base_cols = [c for c in base_cols if c in cand.columns]

cand_out = cand[base_cols].copy()

# ── 저장 ─────────────────────────────────────────────────────────────────
OUT_CAND_DIR.mkdir(parents=True, exist_ok=True)
OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
cand_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"[OK] manifest 저장: {OUT_CSV}")

# ── 요약 통계 계산 ────────────────────────────────────────────────────────
n_stage1_dev = len(stage1_dev_ids)

# ── 수정4: 0명 환자 포함 - 154명 전체 기준으로 candidate 수 계산 ─────────
n_cand_per_patient = (
    cand_out.groupby("patient_id").size()
    .reindex(stage1_dev_ids, fill_value=0)
)
n_hit_per_patient = cand_out[cand_out["lesion_overlap"]].groupby("patient_id").size()

has_any_candidate = set(n_cand_per_patient[n_cand_per_patient > 0].index)
has_lesion_overlap = set(n_hit_per_patient.index)
no_candidate_patients = [pid for pid in stage1_dev_ids if pid not in has_any_candidate]
no_hit_patients = [pid for pid in stage1_dev_ids if pid not in has_lesion_overlap]

n_pos = cand_out["lesion_overlap"].sum()
n_total = len(cand_out)
positive_ratio = n_pos / n_total if n_total > 0 else 0.0
fp_ratio = 1.0 - positive_ratio

# slice hit rate
slice_key_col = "slice_index" if "slice_index" in all_scores.columns else "local_z"
all_slices_dev = all_scores[["patient_id", slice_key_col, "patch_label"]].copy()
all_slices_dev["has_lesion"] = (all_scores["patch_label"] == 1)
lesion_slices = all_slices_dev.groupby(["patient_id", slice_key_col])["has_lesion"].any()
lesion_slices = lesion_slices[lesion_slices].reset_index()[["patient_id", slice_key_col]]
lesion_slices["slice_key"] = list(zip(lesion_slices["patient_id"], lesion_slices[slice_key_col]))

cand_slices = cand_out[["patient_id", slice_key_col]].drop_duplicates()
cand_slices["slice_key"] = list(zip(cand_slices["patient_id"], cand_slices[slice_key_col]))
hit_slices = lesion_slices["slice_key"].isin(set(cand_slices["slice_key"]))
slice_hit_rate = hit_slices.sum() / len(lesion_slices) if len(lesion_slices) > 0 else 0.0

# position_bin 분포
pos_bin_dist = cand_out["position_bin"].value_counts(dropna=False).to_dict()
pos_bin_dist = {str(k): int(v) for k, v in pos_bin_dist.items()}

# z_level 분포
z_level_dist = cand_out["z_level"].value_counts(dropna=False).to_dict()
z_level_dist = {str(k): int(v) for k, v in z_level_dist.items()}

# NSCLC/MSD별 요약
group_summary = {}
for grp, gdf in cand_out.groupby("group"):
    gpids = gdf["patient_id"].unique()
    g_total = len(gdf)
    g_pos = gdf["lesion_overlap"].sum()
    group_summary[str(grp)] = {
        "n_patients": int(len(gpids)),
        "n_candidates": int(g_total),
        "positive_count": int(g_pos),
        "positive_ratio": float(g_pos / g_total) if g_total > 0 else 0.0,
        "fp_ratio": float(1 - g_pos / g_total) if g_total > 0 else 0.0,
        "cand_per_patient_mean": float(gdf.groupby("patient_id").size().mean()),
    }

# 병변 크기별 요약 (per_patient_screening 사용)
screening_df = pd.read_csv(SCREENING_CSV)
screening_s1 = screening_df[screening_df["patient_id"].isin(stage1_dev_ids)].copy()
if "lesion_patch_total" in screening_s1.columns:
    screening_s1["lesion_size_bin"] = pd.cut(
        screening_s1["lesion_patch_total"],
        bins=[0, 50, 200, 500, 9_999_999],
        labels=["tiny(≤50)", "small(51-200)", "medium(201-500)", "large(>500)"],
    )
    size_pids = screening_s1.groupby("lesion_size_bin", observed=True)["patient_id"].apply(list).to_dict()
    size_summary = {}
    for sz, pids in size_pids.items():
        sz_cand = cand_out[cand_out["patient_id"].isin(pids)]
        n_pat = len(pids)
        n_hit = sum(1 for p in pids if p in has_lesion_overlap)
        size_summary[str(sz)] = {
            "n_patients": n_pat,
            "n_with_lesion_overlap": n_hit,
            "hit_rate": float(n_hit / n_pat) if n_pat > 0 else 0.0,
            "n_candidates_mean": float(sz_cand.groupby("patient_id").size().mean()) if len(sz_cand) > 0 else 0.0,
        }
else:
    size_summary = {"note": "lesion_patch_total 컬럼 없음"}

# ── JSON 저장 ─────────────────────────────────────────────────────────────
summary = {
    "generated": "2026-05-23",
    "rule": "rule_a_p95",
    "threshold_used": threshold_p95,
    "stage1_dev_n_patients": n_stage1_dev,
    "n_patients_with_any_candidate": len(has_any_candidate),
    "n_patients_no_candidate": len(no_candidate_patients),
    "no_candidate_patients": sorted(no_candidate_patients),
    "n_patients_with_lesion_overlap": len(has_lesion_overlap),
    "no_hit_patients": sorted(no_hit_patients),
    "total_candidates": int(n_total),
    "positive_candidates": int(n_pos),
    "positive_ratio": float(positive_ratio),
    "fp_ratio": float(fp_ratio),
    "candidates_per_patient": {
        "min": int(n_cand_per_patient.min()),
        "median": float(n_cand_per_patient.median()),
        "mean": float(n_cand_per_patient.mean()),
        "max": int(n_cand_per_patient.max()),
        "note": "154명 전체 기준 (candidate 0명 환자 포함)",
    },
    "lesion_overlap_candidates_per_patient": {
        "min": int(n_hit_per_patient.min()) if len(n_hit_per_patient) > 0 else 0,
        "median": float(n_hit_per_patient.median()) if len(n_hit_per_patient) > 0 else 0.0,
        "mean": float(n_hit_per_patient.mean()) if len(n_hit_per_patient) > 0 else 0.0,
        "max": int(n_hit_per_patient.max()) if len(n_hit_per_patient) > 0 else 0,
    },
    "slice_hit_rate": float(slice_hit_rate),
    "n_lesion_slices_total": int(len(lesion_slices)),
    "position_bin_distribution": pos_bin_dist,
    "z_level_distribution": z_level_dist,
    "group_summary": group_summary,
    "lesion_size_summary": size_summary,
    "stage2_holdout_sealed": True,
    "existing_results_modified": False,
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"[OK] summary JSON 저장: {OUT_JSON}")

# ── MD 저장 ───────────────────────────────────────────────────────────────
lines = [
    "# Rule A (p95) Candidate Manifest Dry-run 요약",
    "",
    f"> **생성일**: 2026-05-23 | **rule**: rule_a_p95 | **threshold**: {threshold_p95:.6f}",
    f"> stage1_dev 전용. stage2_holdout 봉인. crop/PNG/모델학습 없음. 기존 결과 미수정.",
    "",
    "---",
    "",
    "## 1. 기본 통계",
    "",
    f"| 항목 | 값 |",
    f"|------|-----|",
    f"| stage1_dev 환자 수 | {n_stage1_dev} |",
    f"| candidate 1개 이상 환자 | {len(has_any_candidate)} |",
    f"| candidate 0개 환자 | {len(no_candidate_patients)} |",
    f"| lesion_overlap candidate 보유 환자 | {len(has_lesion_overlap)} |",
    f"| no-hit 환자 수 | {len(no_hit_patients)} |",
    f"| 전체 candidate 수 | {n_total:,} |",
    f"| positive candidate 수 (lesion_overlap=True) | {int(n_pos):,} |",
    f"| positive 비율 | {positive_ratio:.4f} |",
    f"| false positive 비율 | {fp_ratio:.4f} |",
    f"| slice 단위 hit rate | {slice_hit_rate:.4f} |",
    "",
    "## 2. 환자별 candidate 수 (154명 전체 기준, 0명 환자 포함)",
    "",
    f"| 통계 | 값 |",
    f"|------|-----|",
    f"| min | {int(n_cand_per_patient.min())} |",
    f"| median | {n_cand_per_patient.median():.1f} |",
    f"| mean | {n_cand_per_patient.mean():.1f} |",
    f"| max | {int(n_cand_per_patient.max())} |",
    "",
    "## 3. no-hit 환자 목록",
    "",
]
if no_hit_patients:
    for pid in sorted(no_hit_patients):
        lines.append(f"- {pid}")
else:
    lines.append("- (없음)")
lines += [
    "",
    "## 4. NSCLC/MSD별 요약",
    "",
    "| group | 환자 수 | candidate 수 | positive 비율 | fp 비율 | 환자당 평균 |",
    "|-------|---------|-------------|--------------|---------|------------|",
]
for grp, gs in sorted(group_summary.items()):
    lines.append(
        f"| {grp} | {gs['n_patients']} | {gs['n_candidates']:,} | "
        f"{gs['positive_ratio']:.4f} | {gs['fp_ratio']:.4f} | {gs['cand_per_patient_mean']:.1f} |"
    )
lines += [
    "",
    "## 5. position_bin 분포",
    "",
    "| position_bin | candidate 수 |",
    "|-------------|-------------|",
]
for pb, cnt in sorted(pos_bin_dist.items(), key=lambda x: -x[1]):
    lines.append(f"| {pb} | {cnt:,} |")
lines += [
    "",
    "## 6. z_level 분포",
    "",
    "| z_level | candidate 수 |",
    "|--------|-------------|",
]
for zl, cnt in sorted(z_level_dist.items(), key=lambda x: -x[1]):
    lines.append(f"| {zl} | {cnt:,} |")

if isinstance(size_summary, dict) and "note" not in size_summary:
    lines += [
        "",
        "## 7. 병변 크기별 요약",
        "",
        "| 크기 구분 | 환자 수 | lesion-hit 환자 수 | hit rate | 환자당 평균 candidate |",
        "|---------|--------|-----------------|---------|---------------------|",
    ]
    for sz, ss in size_summary.items():
        lines.append(
            f"| {sz} | {ss['n_patients']} | {ss['n_with_lesion_overlap']} | "
            f"{ss['hit_rate']:.4f} | {ss['n_candidates_mean']:.1f} |"
        )

lines += [
    "",
    "---",
    "",
    "*생성일: 2026-05-23 | stage1_dev 전용 | stage2_holdout 봉인 | 기존 결과 미수정*",
]

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"[OK] summary MD 저장: {OUT_MD}")

# ── 콘솔 요약 ────────────────────────────────────────────────────────────
print("\n=== Rule A p95 Candidate Manifest Dry-run 결과 ===")
print(f"  stage1_dev: {n_stage1_dev}명")
print(f"  전체 candidate: {n_total:,}")
print(f"  positive 비율: {positive_ratio:.4f}  fp 비율: {fp_ratio:.4f}")
print(f"  lesion-overlap 보유 환자: {len(has_lesion_overlap)}/{n_stage1_dev}")
print(f"  no-hit 환자: {sorted(no_hit_patients)}")
print(f"  slice hit rate: {slice_hit_rate:.4f}")
print(f"  환자당 candidate (154명 전체): min={int(n_cand_per_patient.min())} median={n_cand_per_patient.median():.1f} mean={n_cand_per_patient.mean():.1f} max={int(n_cand_per_patient.max())}")
print(f"  stage2_holdout 봉인: 준수")
