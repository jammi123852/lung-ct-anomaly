"""
build_rd4ad_train_sampling_manifest.py

RD4AD 학습용 normal_like 후보를 candidate manifest에서 별도 sampling manifest로 추출한다.
- rd4ad_label == normal_like만 대상
- lesion_candidate / ambiguous 제외
- 환자당 최대 50개, large_bbox 환자당 최대 5개
- max_padim_score 상위 우선 + score stratum 균형 샘플링
- 원본 candidate manifest 수정 없음
- stage2_holdout / 봉인 환자 차단
- v2 경로 접근 금지
- crop/overlay/학습/평가/score 재계산 없음
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# ── 상수 ──────────────────────────────────────────────────────────────────────
SEALED_PATIENTS: set[str] = {"LUNG1-089", "LUNG1-231", "LUNG1-372"}
WEAK_CASE_PATIENTS: set[str] = {"MSD_lung_043", "MSD_lung_079", "MSD_lung_096"}

REQUIRED_COLUMNS: list[str] = [
    "patient_id", "group", "stage_split", "safe_id", "candidate_id",
    "rd4ad_label", "binary_label", "max_padim_score", "mean_padim_score",
    "bbox_too_large", "z_center", "z_lo", "z_hi",
    "y0", "x0", "y1", "x1",
    "y0_crop", "x0_crop", "y1_crop", "x1_crop",
    "y0_fixed_crop", "x0_fixed_crop", "y1_fixed_crop", "x1_fixed_crop",
]


# ── preflight ─────────────────────────────────────────────────────────────────

def run_preflight(args: argparse.Namespace, df: pd.DataFrame, output_dir: Path) -> None:
    """실행 전 안전장치 검증. 문제 발견 시 sys.exit(1)."""

    # v2 경로 차단
    if "v2" in str(args.source_manifest).lower():
        print(f"[ABORT] source-manifest 경로에 v2가 포함되어 있습니다: {args.source_manifest}")
        sys.exit(1)

    # 필수 컬럼 확인
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"[ABORT] 필수 컬럼 누락: {missing_cols}")
        sys.exit(1)

    # rd4ad_label 컬럼 존재
    if "rd4ad_label" not in df.columns:
        print("[ABORT] rd4ad_label 컬럼이 없습니다.")
        sys.exit(1)

    # stage_split guard: stage1_dev만 허용
    if "stage_split" not in df.columns:
        print("[ABORT] stage_split 컬럼이 없습니다.")
        sys.exit(1)
    # NaN 차단
    if df["stage_split"].isna().any():
        nan_patients = df[df["stage_split"].isna()]["patient_id"].unique().tolist()[:5]
        print("[ABORT] stage_split에 NaN이 있습니다.")
        print(f"  해당 환자 예시 (최대 5명): {nan_patients}")
        sys.exit(1)
    # 빈 문자열/공백 차단
    blank_mask = df["stage_split"].astype(str).str.strip().eq("")
    if blank_mask.any():
        blank_patients = df[blank_mask]["patient_id"].unique().tolist()[:5]
        print("[ABORT] stage_split에 빈 문자열 또는 공백이 있습니다.")
        print(f"  해당 환자 예시 (최대 5명): {blank_patients}")
        sys.exit(1)
    # stage1_dev 외 값 차단
    unique_splits = set(df["stage_split"].astype(str).str.strip().unique().tolist())
    if unique_splits != {"stage1_dev"}:
        unexpected = sorted(unique_splits - {"stage1_dev"})
        sample_patients = (
            df[df["stage_split"].astype(str).str.strip().isin(unexpected)]["patient_id"]
            .unique()
            .tolist()[:5]
        )
        print(f"[ABORT] stage_split에 stage1_dev 외 값이 있습니다: {sorted(unique_splits)}")
        print(f"  해당 환자 예시 (최대 5명): {sample_patients}")
        sys.exit(1)

    # row 수 / 환자 수 hard guard
    n_rows = len(df)
    n_patients = df["patient_id"].nunique()
    print(f"[preflight] source row 수: {n_rows} (기준: {args.expected_rows:,})")
    print(f"[preflight] unique patient 수: {n_patients} (기준: {args.expected_patients})")
    if n_rows != args.expected_rows:
        print(f"[ABORT] source row 수 불일치: 실제 {n_rows}, 기준 {args.expected_rows}")
        sys.exit(1)
    if n_patients != args.expected_patients:
        print(f"[ABORT] unique patient 수 불일치: 실제 {n_patients}, 기준 {args.expected_patients}")
        sys.exit(1)

    # stage2_holdout 차단
    if "stage_split" in df.columns:
        holdout_patients = df[df["stage_split"] == "stage2_holdout"]["patient_id"].unique().tolist()
        if holdout_patients:
            print(f"[ABORT] stage2_holdout 환자가 포함되어 있습니다: {holdout_patients}")
            sys.exit(1)

    # 봉인 환자 차단
    present_patients = set(df["patient_id"].unique())
    sealed_found = present_patients & SEALED_PATIENTS
    if sealed_found:
        print(f"[ABORT] 봉인 환자가 포함되어 있습니다: {sorted(sealed_found)}")
        sys.exit(1)

    # normal_like 후보 1개 이상 확인
    n_normal_like = (df["rd4ad_label"] == "normal_like").sum()
    if n_normal_like == 0:
        print("[ABORT] normal_like 후보가 없습니다.")
        sys.exit(1)
    print(f"[preflight] normal_like 후보 수: {n_normal_like}")

    # output dir 충돌 확인
    if not args.dry_run and output_dir.exists() and not args.force:
        print(f"[ABORT] 출력 디렉토리가 이미 존재합니다: {output_dir}")
        print("  --force 옵션을 사용하면 덮어쓸 수 있습니다.")
        sys.exit(1)

    print("[preflight] 전체 통과")


# ── zero_lc 환자 계산 ─────────────────────────────────────────────────────────

def compute_zero_lc_patients(df: pd.DataFrame) -> list[str]:
    """lesion_candidate가 0개인 환자 목록을 반환한다."""
    lc_counts = (
        df[df["rd4ad_label"] == "lesion_candidate"]
        .groupby("patient_id")
        .size()
    )
    all_patients = set(df["patient_id"].unique())
    patients_with_lc = set(lc_counts.index)
    return sorted(all_patients - patients_with_lc)


# ── 환자별 sampling ───────────────────────────────────────────────────────────

def select_patient_samples(
    patient_df: pd.DataFrame,
    max_per_patient: int,
    top_score: int,
    max_large_bbox: int,
    seed: int,
) -> pd.DataFrame:
    """
    환자별 normal_like 후보를 score stratified sampling으로 선택한다.
    - 입력: 해당 환자의 normal_like 행만 포함된 DataFrame
    - 반환: 선택된 행에 추가 컬럼이 붙은 DataFrame
    """
    rng = random.Random(seed)
    df = patient_df.copy()

    # 1. 안정 정렬: max_padim_score 내림차순, 동점 시 candidate_id 오름차순
    df = df.sort_values(
        ["max_padim_score", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    # 2. score_rank_within_patient 부여 (1 = 최고)
    df["score_rank_within_patient"] = range(1, len(df) + 1)

    # 3. large_bbox 여부 확인
    df["_is_large"] = df["bbox_too_large"].astype(bool)

    # 4. top_candidates 분리 (상위 top_score개)
    top_pool = df.iloc[:min(top_score, len(df))].copy()
    rest_pool = df.iloc[min(top_score, len(df)):].copy()

    # 5. large_bbox cap 적용
    large_bbox_cap_applied = False
    selected_large_count = 0

    top_large = top_pool[top_pool["_is_large"]]
    top_non_large = top_pool[~top_pool["_is_large"]]

    if len(top_large) > max_large_bbox:
        # large_bbox를 cap까지만 허용, 나머지는 non-large로 대체
        top_large_trimmed = top_large.iloc[:max_large_bbox]
        # 대체 후보: rest_pool의 non-large + top_pool에서 밀린 large
        extra_non_large_needed = len(top_large) - max_large_bbox
        rest_non_large = rest_pool[~rest_pool["_is_large"]]
        extra_fill = rest_non_large.iloc[:extra_non_large_needed]
        top_selected = pd.concat([top_non_large, top_large_trimmed, extra_fill])
        top_selected = top_selected.sort_values(
            ["max_padim_score", "candidate_id"], ascending=[False, True]
        ).reset_index(drop=True)
        # rest_pool에서 이미 사용된 extra_fill 제거
        used_ids = set(extra_fill["candidate_id"].tolist())
        rest_pool = rest_pool[~rest_pool["candidate_id"].isin(used_ids)].copy()
        large_bbox_cap_applied = True
        selected_large_count = max_large_bbox
    else:
        top_selected = top_pool.copy()
        selected_large_count = int(top_large["_is_large"].sum())

    # selected에 top_priority stratum 표시
    top_selected = top_selected.copy()
    top_selected["score_stratum"] = "top_priority"
    top_selected["selected_reason"] = "top_score"
    top_selected["large_bbox_cap_applied"] = large_bbox_cap_applied

    selected_parts = [top_selected]
    quota = max_per_patient - len(top_selected)

    # 6. 남은 quota를 stratum 기반 균형 샘플링
    if quota > 0 and len(rest_pool) > 0:
        # large_bbox cap 남은 허용량
        remaining_large_cap = max_large_bbox - selected_large_count
        # rest_pool에서 large_bbox cap 적용
        rest_non_large = rest_pool[~rest_pool["_is_large"]].copy()
        rest_large = rest_pool[rest_pool["_is_large"]].copy()
        if len(rest_large) > remaining_large_cap:
            rest_large = rest_large.iloc[:remaining_large_cap]
        pool = pd.concat([rest_non_large, rest_large]).sort_values(
            ["max_padim_score", "candidate_id"], ascending=[False, True]
        ).reset_index(drop=True)

        if len(pool) == 0:
            pass
        elif len(pool) <= 3:
            # 후보가 너무 적으면 그냥 전체 선택
            pool_sel = pool.iloc[:quota].copy()
            pool_sel["score_stratum"] = "fallback"
            pool_sel["selected_reason"] = "fallback"
            pool_sel["large_bbox_cap_applied"] = large_bbox_cap_applied
            selected_parts.append(pool_sel)
        else:
            # stratum 분할: high / mid / low (3등분)
            n_pool = len(pool)
            boundary1 = n_pool // 3
            boundary2 = 2 * (n_pool // 3)
            high_pool = pool.iloc[:boundary1].copy()
            mid_pool = pool.iloc[boundary1:boundary2].copy()
            low_pool = pool.iloc[boundary2:].copy()

            stratum_quota_base = quota // 3
            stratum_remainder = quota % 3
            stratum_quotas = {
                "high": stratum_quota_base + (1 if stratum_remainder > 0 else 0),
                "mid": stratum_quota_base + (1 if stratum_remainder > 1 else 0),
                "low": stratum_quota_base,
            }

            stratum_pools = {"high": high_pool, "mid": mid_pool, "low": low_pool}
            stratum_selected: dict[str, pd.DataFrame] = {}
            leftover_quota = 0

            for sname, spool in stratum_pools.items():
                sq = stratum_quotas[sname]
                available = len(spool)
                if available >= sq:
                    chosen = spool.iloc[:sq].copy()
                else:
                    chosen = spool.copy()
                    leftover_quota += sq - available
                chosen["score_stratum"] = sname
                chosen["selected_reason"] = "stratum_fill"
                chosen["large_bbox_cap_applied"] = large_bbox_cap_applied
                stratum_selected[sname] = chosen

            # leftover fallback: 부족분을 다른 stratum에서 보충
            if leftover_quota > 0:
                fallback_candidates = []
                for sname, spool in stratum_pools.items():
                    already_chosen_ids = set(stratum_selected[sname]["candidate_id"].tolist())
                    remaining = spool[~spool["candidate_id"].isin(already_chosen_ids)]
                    fallback_candidates.append(remaining)
                fallback_pool = pd.concat(fallback_candidates).sort_values(
                    ["max_padim_score", "candidate_id"], ascending=[False, True]
                ).reset_index(drop=True)
                fallback_sel = fallback_pool.iloc[:leftover_quota].copy()
                fallback_sel["score_stratum"] = "fallback"
                fallback_sel["selected_reason"] = "fallback"
                fallback_sel["large_bbox_cap_applied"] = large_bbox_cap_applied
                if len(fallback_sel) > 0:
                    selected_parts.append(fallback_sel)

            for sname in ["high", "mid", "low"]:
                if len(stratum_selected[sname]) > 0:
                    selected_parts.append(stratum_selected[sname])

    # 7. 합치고 max_per_patient로 trim
    result = pd.concat(selected_parts).drop_duplicates(subset=["candidate_id"])
    result = result.sort_values(
        ["max_padim_score", "candidate_id"], ascending=[False, True]
    ).reset_index(drop=True)
    if len(result) > max_per_patient:
        result = result.iloc[:max_per_patient].copy()

    # 8. patient_sample_index 부여 (0-based)
    result["patient_sample_index"] = range(len(result))

    # 9. _is_large 임시 컬럼 제거
    result = result.drop(columns=["_is_large"])

    return result


# ── sampling manifest 빌드 ────────────────────────────────────────────────────

def build_sampling_manifest(
    df: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, dict]:
    """
    전체 환자에 대해 sampling을 수행하고 manifest DataFrame과 summary dict를 반환한다.
    """
    sampling_rule = f"top{args.top_score_per_patient}_stratified_seed{args.seed}"
    source_manifest_str = str(Path(args.source_manifest).resolve())

    # zero_lc 환자 계산
    zero_lc_patients = compute_zero_lc_patients(df)

    # normal_like 필터
    normal_like_df = df[df["rd4ad_label"] == "normal_like"].copy()
    n_normal_like_available = len(normal_like_df)
    n_excluded_lc = int((df["rd4ad_label"] == "lesion_candidate").sum())
    n_excluded_amb = int((df["rd4ad_label"] == "ambiguous").sum())

    all_selected_parts = []
    patient_sample_counts: dict[str, int] = {}
    stratum_counts: dict[str, int] = {
        "top_priority": 0, "high": 0, "mid": 0, "low": 0, "fallback": 0
    }
    total_large_selected = 0
    excluded_by_large_cap = 0

    for patient_id, pgroup in normal_like_df.groupby("patient_id", sort=True):
        selected = select_patient_samples(
            patient_df=pgroup,
            max_per_patient=args.max_per_patient,
            top_score=args.top_score_per_patient,
            max_large_bbox=args.max_large_bbox_per_patient,
            seed=args.seed,
        )

        # 추가 컬럼
        selected = selected.copy()
        selected["selected_for_rd4ad_train"] = True
        selected["sample_role"] = "rd4ad_train_normal_like"
        selected["sampling_rule"] = sampling_rule
        selected["sampling_seed"] = args.seed
        selected["source_manifest_path"] = source_manifest_str
        selected["sampling_output_tag"] = args.output_tag

        all_selected_parts.append(selected)
        patient_sample_counts[str(patient_id)] = len(selected)

        for sname in stratum_counts:
            if "score_stratum" in selected.columns:
                stratum_counts[sname] += int(
                    (selected["score_stratum"] == sname).sum()
                )

        if "bbox_too_large" in selected.columns:
            total_large_selected += int(selected["bbox_too_large"].astype(bool).sum())

        # excluded_by_large_bbox_cap: 해당 환자의 large_bbox 총 수 - 선택된 large_bbox 수
        all_large_for_patient = int(pgroup["bbox_too_large"].astype(bool).sum())
        patient_large_selected = int(selected["bbox_too_large"].astype(bool).sum())
        if all_large_for_patient > args.max_large_bbox_per_patient:
            excluded_by_large_cap += all_large_for_patient - patient_large_selected

    if len(all_selected_parts) == 0:
        result_df = pd.DataFrame()
    else:
        result_df = pd.concat(all_selected_parts, ignore_index=True)

    n_selected_total = len(result_df)
    n_selected_patients = result_df["patient_id"].nunique() if n_selected_total > 0 else 0
    counts_list = list(patient_sample_counts.values())
    count_arr = np.array(counts_list, dtype=float) if counts_list else np.array([0.0])

    selected_large_ratio = (
        round(total_large_selected / n_selected_total, 4)
        if n_selected_total > 0 else 0.0
    )

    # weak_case selected counts
    weak_case_selected: dict[str, int] = {}
    for wc in sorted(WEAK_CASE_PATIENTS):
        weak_case_selected[wc] = patient_sample_counts.get(wc, 0)

    # zero_lc selected counts
    zero_lc_selected: dict[str, int] = {}
    for zlc in zero_lc_patients:
        zero_lc_selected[zlc] = patient_sample_counts.get(zlc, 0)

    # group별 selected counts
    group_selected: dict[str, int] = {}
    if n_selected_total > 0 and "group" in result_df.columns:
        for grp, grp_df in result_df.groupby("group"):
            group_selected[str(grp)] = len(grp_df)

    summary = {
        "script": "build_rd4ad_train_sampling_manifest.py",
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source_manifest_path": source_manifest_str,
        "output_tag": args.output_tag,
        "sampling_rule": sampling_rule,
        "random_seed": args.seed,
        "max_per_patient": args.max_per_patient,
        "top_score_per_patient": args.top_score_per_patient,
        "max_large_bbox_per_patient": args.max_large_bbox_per_patient,
        "n_source_rows": len(df),
        "n_source_patients": int(df["patient_id"].nunique()),
        "n_normal_like_available": n_normal_like_available,
        "n_excluded_lesion_candidate": n_excluded_lc,
        "n_excluded_ambiguous": n_excluded_amb,
        "n_selected_total": n_selected_total,
        "n_selected_patients": n_selected_patients,
        "patient_sample_count_min": int(count_arr.min()),
        "patient_sample_count_median": float(round(float(np.median(count_arr)), 1)),
        "patient_sample_count_max": int(count_arr.max()),
        "selected_large_bbox_count": total_large_selected,
        "selected_large_bbox_ratio": selected_large_ratio,
        "excluded_by_large_bbox_cap": excluded_by_large_cap,
        "weak_case_selected_counts": weak_case_selected,
        "zero_lc_patient_ids": zero_lc_patients,
        "zero_lc_selected_counts": zero_lc_selected,
        "score_stratum_selected_counts": stratum_counts,
        "group_selected_counts": group_selected,
        "dry_run": args.dry_run,
        "note": [
            "RD4AD 학습용 sampling manifest이며 성능 우위 확정 아님",
            "lesion_candidate와 ambiguous는 RD4AD 학습에서 제외됨",
        ],
    }

    return result_df, summary


# ── argparse ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RD4AD 학습용 normal_like sampling manifest 생성"
    )
    parser.add_argument(
        "--source-manifest",
        default=str(
            REPO
            / "outputs/second-stage-lesion-refiner-v1/candidates"
            / "stage1_dev_fixed96_thr001_v1"
            / "candidate_manifest_stage1_dev_fixed96_thr001_v1.csv"
        ),
        help="입력 candidate manifest 경로",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO / "outputs/second-stage-lesion-refiner-v1/sampling"),
        help="출력 루트 디렉토리",
    )
    parser.add_argument(
        "--output-tag",
        default="rd4ad_train_normal_like_fixed96_thr001_v1",
        help="출력 태그 (하위 폴더명 및 파일명에 사용)",
    )
    parser.add_argument(
        "--max-per-patient",
        type=int,
        default=50,
        help="환자당 최대 선택 수",
    )
    parser.add_argument(
        "--top-score-per-patient",
        type=int,
        default=20,
        help="상위 score 우선 선택 수",
    )
    parser.add_argument(
        "--max-large-bbox-per-patient",
        type=int,
        default=5,
        help="환자당 large_bbox 최대 허용 수",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="dry-run 모드: 파일 생성 없이 예상 결과만 출력",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 출력이 존재해도 덮어쓰기 허용",
    )
    parser.add_argument(
        "--no-runtime-append",
        action="store_true",
        help="runtime_summary.csv 기록 생략",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=22379,
        help="source manifest row 수 hard guard 기준값 (default: 22379)",
    )
    parser.add_argument(
        "--expected-patients",
        type=int,
        default=154,
        help="source manifest unique patient 수 hard guard 기준값 (default: 154)",
    )
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    source_path = Path(args.source_manifest)
    output_root = Path(args.output_root)
    output_dir = output_root / args.output_tag
    out_csv_name = f"rd4ad_train_sampling_manifest_{args.output_tag}.csv"
    out_json_name = f"rd4ad_train_sampling_manifest_{args.output_tag}_summary.json"
    out_csv = output_dir / out_csv_name
    out_json = output_dir / out_json_name

    # 입력 파일 로드
    if not source_path.exists():
        print(f"[ABORT] source manifest 파일이 없습니다: {source_path}")
        sys.exit(1)

    print(f"[load] {source_path}")
    df = pd.read_csv(source_path)

    # preflight
    run_preflight(args, df, output_dir)

    # sampling
    print("[sampling] 환자별 score stratified sampling 시작...")
    result_df, summary = build_sampling_manifest(df, args)

    n_selected = summary["n_selected_total"]
    n_patients = summary["n_selected_patients"]
    cnt_min = summary["patient_sample_count_min"]
    cnt_med = summary["patient_sample_count_median"]
    cnt_max = summary["patient_sample_count_max"]

    print(f"[sampling] 선택된 후보 수: {n_selected} (환자 {n_patients}명)")
    print(f"[sampling] 환자별 선택 수: min={cnt_min}, median={cnt_med}, max={cnt_max}")
    print(f"[sampling] large_bbox 선택 수: {summary['selected_large_bbox_count']}")
    print(f"[sampling] large_bbox cap 제외 수: {summary['excluded_by_large_bbox_cap']}")
    print(f"[sampling] weak_case selected: {summary['weak_case_selected_counts']}")
    print(f"[sampling] zero_lc 환자 수: {len(summary['zero_lc_patient_ids'])}")
    print(f"[sampling] score stratum: {summary['score_stratum_selected_counts']}")
    print(f"[sampling] group: {summary['group_selected_counts']}")

    if args.dry_run:
        print("\n[dry-run] 파일 생성 없음. 위 예상 결과만 출력합니다.")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    # 실제 저장
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[save] sampling manifest CSV → {out_csv}")
    result_df.to_csv(out_csv, index=False)

    print(f"[save] summary JSON → {out_json}")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[done] sampling manifest 생성 완료.")
    print(f"  CSV : {out_csv}")
    print(f"  JSON: {out_json}")


if __name__ == "__main__":
    main()
