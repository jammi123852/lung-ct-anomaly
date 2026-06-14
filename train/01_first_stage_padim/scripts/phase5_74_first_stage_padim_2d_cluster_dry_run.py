"""
Phase 5.74 First-Stage PaDiM Suspicious Patch 2D Clustering Dry-Run Script

목적:
  - score CSV (by_patient/)에서 sample 환자 N명을 선택하여
    sample-local p99 threshold 기준으로 suspicious patch를 필터링하고
    2D clustering (z-adjacent merge 없음)으로 cluster summary를 생성한다.

제한:
  - --dry-run 없이 실행 금지
  - --sample-patients 최대 3
  - split 파일 없음 → 파일명 정렬 기준 앞 N개 선택
  - sample-local p99: 각 환자 개별 계산, global p99 사용 금지
  - stage2_holdout, v2 등 금지 경로 접근 금지
  - output root 생성은 모든 검증 이후에만 수행
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─── 경로 상수 ────────────────────────────────────────────────────────────────

SCORE_CSV_ROOT = Path(
    "outputs/position-aware-padim-v1/scores/padim_v1/by_patient"
)
OUTPUT_ROOT_BASE = Path(
    "outputs/second-stage-lesion-refiner-v1/review_annotations"
    "/first_stage_padim_cluster_review"
)

EXPECTED_CSV_COUNT = 72

REQUIRED_COLUMNS = [
    "patient_id", "safe_id", "label", "local_z",
    "y0", "x0", "y1", "x1",
    "patch_size", "patch_stride",
    "position_bin", "z_level", "z_ratio",
    "central_peripheral", "central_distance_ratio_mean",
    "left_right_metadata",
    "pure_lung_pixels", "organ_pixels",
    "pure_lung_patch_ratio", "organ_patch_ratio",
    "slice_pure_lung_ratio",
    "padim_score",
]

# 각 항목: (label, 매칭 함수 seg_lower -> bool)
# seg_lower = segment를 lower-case로 정규화한 값
def _make_forbidden_rules():
    rules = [
        ("stage2_holdout", lambda s: s == "stage2_holdout"),
        # v2 단독 segment, v2v2로 시작하는 segment, v2_ 로 시작하는 segment 차단
        # 단 "second-stage-lesion-refiner-v1" 등 v2가 suffix로만 있는 건 허용
        ("v2/v2v2.../v2_ segment", lambda s: s == "v2" or s.startswith("v2v2") or s.startswith("v2_")),
        ("lesion_by_patient", lambda s: s.startswith("lesion_by_patient")),
        ("crops_lesion", lambda s: s.startswith("crops_lesion")),
        ("hard_negative", lambda s: "hard_negative" in s),
        ("nsclc_msd", lambda s: "nsclc_msd" in s),
        ("msd_lung", lambda s: "msd_lung" in s),
    ]
    return rules

FORBIDDEN_RULES = _make_forbidden_rules()

# ─── Guard 함수 ───────────────────────────────────────────────────────────────

def check_forbidden(path_str: str) -> None:
    """경로 각 segment를 case-insensitive + substring 기준으로 검사."""
    parts = Path(path_str).parts
    for seg in parts:
        seg_lower = seg.lower()
        for label, rule in FORBIDDEN_RULES:
            if rule(seg_lower):
                sys.exit(f"[ABORT] 금지 경로 segment '{seg}' ({label}) in {path_str}")


# ─── argparse ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5.74 First-Stage PaDiM 2D Clustering Dry-Run"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="[필수] 이 flag 없이 실행 금지.",
    )
    parser.add_argument(
        "--sample-patients",
        type=int,
        default=3,
        help="선택할 sample 환자 수 (최대 3, 기본값: 3).",
    )
    parser.add_argument(
        "--score-percentile",
        type=float,
        default=99.0,
        help="suspicious patch 기준 백분위수 (기본값: 99.0).",
    )
    parser.add_argument(
        "--max-clusters-per-patient",
        type=int,
        default=3,
        help="환자당 review candidate로 지정할 최대 cluster 수 (기본값: 3).",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="phase5_74_2d_cluster_dry_run_v1",
        help="output 폴더 태그 (기본값: phase5_74_2d_cluster_dry_run_v1).",
    )
    parser.add_argument(
        "--max-patches-per-patient",
        type=int,
        default=None,
        help="p99 필터 이후 환자별 padim_score 내림차순 최대 N개 사용 (기본: None=제한 없음).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="[DISABLED] Phase 5.74에서는 사용 불가. 설정 시 즉시 중단.",
    )
    return parser.parse_args()


# ─── 2D Clustering ──────────────────────────────────────────────────────────

def compute_2d_clusters(df_patient_slice: pd.DataFrame, stride: int = 16) -> list:
    """
    같은 patient_id + 같은 local_z 내에서만 clustering 수행.
    z-adjacent merge 없음.

    연결 기준:
      - bbox IoU > 0
      - 또는 center distance < stride * 1.5

    연결된 patch를 union-find로 connected component로 묶는다.

    Returns:
        list of int: 각 patch에 대한 component label (루트 인덱스 기준)
    """
    patches = df_patient_slice[
        ["y0", "x0", "y1", "x1", "padim_score", "pure_lung_patch_ratio"]
    ].values
    n = len(patches)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def bbox_iou(p1, p2) -> float:
        iy0 = max(p1[0], p2[0])
        ix0 = max(p1[1], p2[1])
        iy1 = min(p1[2], p2[2])
        ix1 = min(p1[3], p2[3])
        inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
        if inter == 0:
            return 0.0
        a1 = (p1[2] - p1[0]) * (p1[3] - p1[1])
        a2 = (p2[2] - p2[0]) * (p2[3] - p2[1])
        union_area = a1 + a2 - inter
        if union_area <= 0:
            return 0.0
        return inter / union_area

    def center_dist(p1, p2) -> float:
        cy1 = (p1[0] + p1[2]) / 2
        cx1 = (p1[1] + p1[3]) / 2
        cy2 = (p2[0] + p2[2]) / 2
        cx2 = (p2[1] + p2[3]) / 2
        return math.sqrt((cy1 - cy2) ** 2 + (cx1 - cx2) ** 2)

    threshold_dist = stride * 1.5

    for i in range(n):
        for j in range(i + 1, n):
            if (
                bbox_iou(patches[i], patches[j]) > 0
                or center_dist(patches[i], patches[j]) < threshold_dist
            ):
                union(i, j)

    labels = [find(i) for i in range(n)]
    return labels


# ─── Cluster Summary 생성 ────────────────────────────────────────────────────

def build_cluster_summary(
    df_suspicious: pd.DataFrame,
    patient_id: str,
    max_clusters_per_patient: int,
    patch_stride: int = 16,
) -> pd.DataFrame:
    """
    patient_id 단위로 각 local_z 내 2D clustering 수행 후
    cluster summary DataFrame 생성.
    """
    records = []

    for local_z, df_z in df_suspicious.groupby("local_z"):
        df_z = df_z.reset_index(drop=True)
        cluster_labels = compute_2d_clusters(df_z, stride=patch_stride)
        df_z["_cluster_label"] = cluster_labels

        for comp_label in set(cluster_labels):
            df_c = df_z[df_z["_cluster_label"] == comp_label]

            scores = df_c["padim_score"].values
            sorted_scores = np.sort(scores)[::-1]

            top3_mean = float(np.mean(sorted_scores[:3]))
            top5_mean = float(np.mean(sorted_scores[:5]))
            mean_score = float(np.mean(scores))
            max_score = float(np.max(scores))

            # max score 패치 좌표
            idx_max = df_c["padim_score"].idxmax()
            rep_row = df_c.loc[idx_max]

            record = {
                "cluster_id": None,  # rank 계산 후 채움
                "patient_id": patient_id,
                "local_z": int(local_z),
                "n_patches": int(len(df_c)),
                "y0_min": int(df_c["y0"].min()),
                "x0_min": int(df_c["x0"].min()),
                "y1_max": int(df_c["y1"].max()),
                "x1_max": int(df_c["x1"].max()),
                "bbox_h": int(df_c["y1"].max() - df_c["y0"].min()),
                "bbox_w": int(df_c["x1"].max() - df_c["x0"].min()),
                "bbox_area": int(
                    (df_c["y1"].max() - df_c["y0"].min())
                    * (df_c["x1"].max() - df_c["x0"].min())
                ),
                "max_patch_score": max_score,
                "top3_mean_patch_score": top3_mean,
                "top5_mean_patch_score": top5_mean,
                "mean_patch_score": mean_score,
                "representative_y0": int(rep_row["y0"]),
                "representative_x0": int(rep_row["x0"]),
                "representative_y1": int(rep_row["y1"]),
                "representative_x1": int(rep_row["x1"]),
                "mean_pure_lung_patch_ratio": float(
                    df_c["pure_lung_patch_ratio"].mean()
                ),
                "min_pure_lung_patch_ratio": float(
                    df_c["pure_lung_patch_ratio"].min()
                ),
                "max_pure_lung_patch_ratio": float(
                    df_c["pure_lung_patch_ratio"].max()
                ),
                # rank/flag는 아래에서 채움
                "cluster_rank_in_patient": None,
                "review_candidate_flag": None,
                "notes": (
                    "sample-local p99, dry-run only, 2D clustering, no z-merge"
                ),
            }
            records.append(record)

    if not records:
        return pd.DataFrame()

    df_clusters = pd.DataFrame(records)

    # 환자 내 rank: top3_mean_patch_score 기준 내림차순 (1-indexed)
    df_clusters = df_clusters.sort_values(
        "top3_mean_patch_score", ascending=False
    ).reset_index(drop=True)
    df_clusters["cluster_rank_in_patient"] = df_clusters.index + 1
    df_clusters["review_candidate_flag"] = (
        df_clusters["cluster_rank_in_patient"] <= max_clusters_per_patient
    )

    # cluster_id 채우기 (rank 확정 후)
    df_clusters["cluster_id"] = df_clusters.apply(
        lambda row: (
            f"{row['patient_id']}_z{row['local_z']}_c{row['cluster_rank_in_patient']}"
        ),
        axis=1,
    )

    return df_clusters


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # (A) dry-run guard
    if not args.dry_run:
        sys.exit("[ABORT] --dry-run 없이 실행 금지.")

    # (B) sample 환자 수 제한
    if args.sample_patients > 3:
        sys.exit("[ABORT] --sample-patients 최대 3.")

    # (C) 금지 경로 guard
    check_forbidden(str(SCORE_CSV_ROOT))
    check_forbidden(str(OUTPUT_ROOT_BASE))

    output_root = OUTPUT_ROOT_BASE / args.output_tag
    check_forbidden(str(output_root))

    # (D) score CSV root 존재 확인
    if not SCORE_CSV_ROOT.exists():
        sys.exit(f"[ABORT] score CSV root 없음: {SCORE_CSV_ROOT}")

    # (F) --force 사용 차단
    if args.force:
        sys.exit("[ABORT] Phase 5.74 dry-run에서는 --force 사용 금지.")

    # (E) output root overwrite guard — --force 없이 항상 중단
    if output_root.exists():
        sys.exit(
            f"[ABORT] output root 이미 존재: {output_root}\n"
            "기존 결과를 삭제하거나 output_tag를 바꿔 새 경로를 사용하세요."
        )

    # ── 입력 검증 (output root 생성 전) ─────────────────────────────────────

    all_csvs = sorted(SCORE_CSV_ROOT.glob("*.csv"))
    actual_count = len(all_csvs)
    if actual_count != EXPECTED_CSV_COUNT:
        print(
            f"[WARNING] CSV 파일 수 예상 {EXPECTED_CSV_COUNT}개, 실제 {actual_count}개."
        )

    if actual_count == 0:
        sys.exit(f"[ABORT] {SCORE_CSV_ROOT} 에 CSV 파일이 없습니다.")

    # sample 환자 선택: 파일명 정렬 기준 앞 N개 (split 파일 없음)
    sample_csvs = all_csvs[: args.sample_patients]
    source_file_stems = [p.stem for p in sample_csvs]

    print(f"[INFO] sample 환자 ({args.sample_patients}명, stem 기준): {source_file_stems}")
    print(
        "[INFO] split 파일 없음 → 파일명 정렬 기준 앞 N개 선택. "
        "성능/threshold 판단에 사용 금지."
    )

    # 각 sample CSV 로드 및 검증
    patient_dfs: dict[str, pd.DataFrame] = {}
    sample_patient_ids: list[str] = []
    for csv_path in sample_csvs:
        source_file_stem = csv_path.stem
        df = pd.read_csv(csv_path)

        # patient_id unique 확인
        unique_pids = df["patient_id"].unique()
        if len(unique_pids) != 1:
            sys.exit(
                f"[ABORT] {csv_path.name}: patient_id가 1개가 아님: {unique_pids.tolist()}"
            )
        internal_pid = str(unique_pids[0])
        if internal_pid != source_file_stem:
            print(
                f"[WARNING] {csv_path.name}: 파일 stem({source_file_stem})과 "
                f"내부 patient_id({internal_pid}) 불일치. 내부 patient_id 사용."
            )

        pid = internal_pid

        # 필수 컬럼 확인
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            sys.exit(
                f"[ABORT] {source_file_stem}: 필수 컬럼 누락: {missing_cols}"
            )

        # patch_size == 32 확인 (y1-y0, x1-x0 기준)
        computed_h = (df["y1"] - df["y0"]).unique()
        computed_w = (df["x1"] - df["x0"]).unique()
        if not (len(computed_h) == 1 and int(computed_h[0]) == 32):
            sys.exit(
                f"[ABORT] {source_file_stem}: patch 높이(y1-y0)가 32 아님: {computed_h.tolist()}"
            )
        if not (len(computed_w) == 1 and int(computed_w[0]) == 32):
            sys.exit(
                f"[ABORT] {source_file_stem}: patch 너비(x1-x0)가 32 아님: {computed_w.tolist()}"
            )

        # patch_size 컬럼 값 확인
        unique_patch_sizes = df["patch_size"].unique()
        if not (len(unique_patch_sizes) == 1 and int(unique_patch_sizes[0]) == 32):
            sys.exit(
                f"[ABORT] {source_file_stem}: patch_size 컬럼 값이 32 아님: {unique_patch_sizes.tolist()}"
            )

        # patch_stride 컬럼 값 확인 (16이어야 함, 복수 값 금지)
        unique_strides_check = df["patch_stride"].unique()
        if len(unique_strides_check) != 1:
            sys.exit(
                f"[ABORT] {source_file_stem}: patch_stride 컬럼이 복수 값: {unique_strides_check.tolist()}"
            )
        if int(unique_strides_check[0]) != 16:
            sys.exit(
                f"[ABORT] {source_file_stem}: patch_stride 컬럼 값이 16 아님: {int(unique_strides_check[0])}"
            )

        # padim_score NaN/Inf 확인
        nan_count = df["padim_score"].isna().sum()
        inf_count = np.isinf(df["padim_score"]).sum()
        if nan_count > 0:
            sys.exit(f"[ABORT] {source_file_stem}: padim_score NaN {nan_count}개.")
        if inf_count > 0:
            sys.exit(f"[ABORT] {source_file_stem}: padim_score Inf {inf_count}개.")

        patient_dfs[pid] = df
        sample_patient_ids.append(pid)
        print(f"[INFO] {pid} (stem: {source_file_stem}): {len(df)}행 로드 완료.")

    # ── p99 suspicious patch 필터 (sample-local p99) ─────────────────────────

    suspicious_dfs: dict[str, pd.DataFrame] = {}
    p99_thresholds: dict[str, float] = {}

    for pid, df in patient_dfs.items():
        threshold = float(np.percentile(df["padim_score"].values, args.score_percentile))
        p99_thresholds[pid] = threshold
        df_sus = df[df["padim_score"] >= threshold].copy().reset_index(drop=True)
        suspicious_dfs[pid] = df_sus
        print(
            f"[INFO] {pid}: sample-local p{args.score_percentile:.0f} threshold={threshold:.6f}, "
            f"suspicious patches={len(df_sus)}"
        )

    print(
        "[INFO] sample-local p99 사용 중. global threshold 아님. "
        "최종 threshold 확정 전 단계."
    )

    # ── max-patches-per-patient 제한 ────────────────────────────────────────
    if args.max_patches_per_patient is not None:
        for pid in list(suspicious_dfs.keys()):
            df_sus = suspicious_dfs[pid]
            if len(df_sus) > args.max_patches_per_patient:
                df_sus = df_sus.nlargest(args.max_patches_per_patient, "padim_score").reset_index(drop=True)
                suspicious_dfs[pid] = df_sus
                print(f"[INFO] {pid}: max_patches_per_patient={args.max_patches_per_patient} 적용 후 {len(df_sus)}행.")

    # ── 2D clustering ────────────────────────────────────────────────────────

    all_cluster_dfs: list[pd.DataFrame] = []
    per_patient_summary: dict = {}

    for pid, df_sus in suspicious_dfs.items():
        # patch_stride: CSV에서 읽어 사용 (단일 값, 16 확정)
        unique_strides = df_sus["patch_stride"].unique()
        if len(unique_strides) != 1:
            sys.exit(f"[ABORT] {pid}: suspicious df patch_stride 복수 값: {unique_strides.tolist()}")
        patch_stride = int(unique_strides[0])
        if patch_stride != 16:
            sys.exit(f"[ABORT] {pid}: suspicious df patch_stride가 16 아님: {patch_stride}")

        df_cluster = build_cluster_summary(
            df_sus, pid, args.max_clusters_per_patient, patch_stride
        )

        if df_cluster.empty:
            print(f"[INFO] {pid}: suspicious patch 없음 또는 cluster 없음.")
            per_patient_summary[pid] = {
                "input_rows": len(patient_dfs[pid]),
                "suspicious_rows": len(df_sus),
                "cluster_count": 0,
                "top_clusters": [],
                "p99_threshold": p99_thresholds[pid],
            }
            continue

        all_cluster_dfs.append(df_cluster)

        top_clusters = (
            df_cluster[df_cluster["review_candidate_flag"]]
            .sort_values("cluster_rank_in_patient")[
                [
                    "cluster_id",
                    "local_z",
                    "n_patches",
                    "top3_mean_patch_score",
                    "max_patch_score",
                    "bbox_area",
                    "cluster_rank_in_patient",
                ]
            ]
            .to_dict(orient="records")
        )

        per_patient_summary[pid] = {
            "input_rows": len(patient_dfs[pid]),
            "suspicious_rows": len(df_sus),
            "cluster_count": int(len(df_cluster)),
            "top_clusters": top_clusters,
            "p99_threshold": p99_thresholds[pid],
        }

        print(
            f"[INFO] {pid}: cluster {len(df_cluster)}개 생성, "
            f"review candidate {df_cluster['review_candidate_flag'].sum()}개."
        )

    # 전체 통합 DataFrame
    if all_cluster_dfs:
        df_all_clusters = pd.concat(all_cluster_dfs, ignore_index=True)
    else:
        df_all_clusters = pd.DataFrame()

    total_input_rows = sum(len(d) for d in patient_dfs.values())
    total_suspicious_rows = sum(len(d) for d in suspicious_dfs.values())
    total_cluster_count = len(df_all_clusters)

    # ── Summary JSON/MD 내용 구성 ─────────────────────────────────────────────

    summary = {
        "sample_patient_count": args.sample_patients,
        "sample_patient_ids": sample_patient_ids,
        "source_file_stems": source_file_stems,
        "split_file_used": False,
        "sample_selection_method": "filename_sort_top_N",
        "total_input_patch_rows": total_input_rows,
        "score_percentile_used": args.score_percentile,
        "percentile_mode": "sample_local_not_global",
        "suspicious_patch_count_total": total_suspicious_rows,
        "cluster_count_total": total_cluster_count,
        "max_patches_per_patient_used": args.max_patches_per_patient,
        "per_patient": per_patient_summary,
        "notes": {
            "dry_run_only": True,
            "no_z_adjacent_merge": True,
            "threshold_not_finalized": True,
            "lesion_conclusion_forbidden": True,
            "stage2_holdout_unused": True,
            "v2_unused": True,
            "original_score_csv_unmodified": True,
            "split_file_warning": (
                "split 파일 없음. 파일명 순서 기준 sample 선택. "
                "성능/threshold 판단에 사용 금지."
            ),
            "percentile_warning": (
                "sample-local p99이며 global threshold 아님. "
                "threshold 확정 전 dry-run 탐색 결과."
            ),
        },
    }

    summary_md_lines = [
        "# Phase 5.74 First-Stage PaDiM 2D Clustering Dry-Run Summary",
        "",
        "## 경고",
        "- **split 파일 없음**: 파일명 순서 기준 sample 선택. 성능/threshold 판단에 사용 금지.",
        "- **sample-local p99**: 각 환자 개별 계산. global threshold 아님.",
        "- **dry-run only**: threshold 미확정. lesion 결론 도출 금지.",
        "- **z-adjacent merge 없음**: 동일 local_z 내 2D clustering만 수행.",
        "- **stage2_holdout/v2 미사용**: 금지 경로 접근 없음.",
        "- **원본 score CSV 수정 없음**.",
        "",
        "## 기본 정보",
        f"- sample 환자 수: {args.sample_patients}",
        f"- sample 환자 IDs: {', '.join(sample_patient_ids)}",
        f"- 파일 선택 방법: {summary['sample_selection_method']}",
        f"- score percentile: {args.score_percentile}",
        f"- percentile 방식: {summary['percentile_mode']}",
        f"- max clusters per patient: {args.max_clusters_per_patient}",
        "",
        "## 집계",
        f"- 전체 입력 patch 행: {total_input_rows}",
        f"- suspicious patch 합계: {total_suspicious_rows}",
        f"- cluster 합계: {total_cluster_count}",
        "",
        "## 환자별 요약",
    ]

    for pid, info in per_patient_summary.items():
        summary_md_lines.append(f"### {pid}")
        summary_md_lines.append(f"- 입력 행: {info['input_rows']}")
        summary_md_lines.append(f"- suspicious 행: {info['suspicious_rows']}")
        summary_md_lines.append(f"- cluster 수: {info['cluster_count']}")
        summary_md_lines.append(f"- p99 threshold: {info['p99_threshold']:.6f}")
        if info["top_clusters"]:
            summary_md_lines.append("- top clusters:")
            for tc in info["top_clusters"]:
                summary_md_lines.append(
                    f"  - {tc['cluster_id']}: z={tc['local_z']}, "
                    f"n_patches={tc['n_patches']}, "
                    f"top3_mean={tc['top3_mean_patch_score']:.6f}, "
                    f"max={tc['max_patch_score']:.6f}"
                )
        summary_md_lines.append("")

    summary_md = "\n".join(summary_md_lines)

    # ── 출력 파일 경로 변수 정의 (mkdir 전) ─────────────────────────────────
    clusters_csv_path = output_root / "phase5_74_2d_cluster_summary.csv"
    summary_json_path = output_root / "phase5_74_2d_cluster_summary.json"
    summary_md_path = output_root / "phase5_74_2d_cluster_summary.md"

    # ── 저장 전 최종 검증 ────────────────────────────────────────────────────
    if not df_all_clusters.empty:
        # cluster_id 중복 검증
        dup_count = df_all_clusters["cluster_id"].duplicated().sum()
        if dup_count > 0:
            sys.exit(f"[ABORT] cluster_id 중복 {dup_count}건 발견.")

        # review_candidate_flag 누락 검증
        flag_null = df_all_clusters["review_candidate_flag"].isna().sum()
        if flag_null > 0:
            sys.exit(f"[ABORT] review_candidate_flag 누락 {flag_null}건.")

        # 환자당 review_candidate_flag=True 개수 검증
        for pid_check, grp in df_all_clusters.groupby("patient_id"):
            true_count = grp["review_candidate_flag"].sum()
            if true_count > args.max_clusters_per_patient:
                sys.exit(
                    f"[ABORT] {pid_check}: review_candidate_flag=True {true_count}개 > "
                    f"max_clusters_per_patient={args.max_clusters_per_patient}"
                )

    # summary notes 필수 항목 검증
    required_notes = [
        "dry_run_only", "threshold_not_finalized",
        "lesion_conclusion_forbidden", "stage2_holdout_unused", "v2_unused",
    ]
    for note_key in required_notes:
        if not summary.get("notes", {}).get(note_key, False):
            sys.exit(f"[ABORT] summary notes.{note_key}가 True가 아님.")

    # ── output root 생성 (모든 검증 이후) ────────────────────────────────────

    # 저장 직전 출력 파일 존재 재확인
    for fpath in [clusters_csv_path, summary_json_path, summary_md_path]:
        if fpath.exists():
            sys.exit(f"[ABORT] 저장 직전 파일 이미 존재: {fpath}")

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] output root 생성: {output_root}")

    # clusters CSV 저장
    if not df_all_clusters.empty:
        df_all_clusters.to_csv(clusters_csv_path, index=False)
        print(f"[INFO] clusters CSV 저장: {clusters_csv_path}")
    else:
        print("[WARNING] cluster가 없어 CSV를 생성하지 않습니다.")

    # summary JSON 저장
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] summary JSON 저장: {summary_json_path}")

    # summary MD 저장
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"[INFO] summary MD 저장: {summary_md_path}")

    print("[DONE] dry-run 완료.")


if __name__ == "__main__":
    main()
