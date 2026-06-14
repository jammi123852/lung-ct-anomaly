#!/usr/bin/env python3
"""
Phase 5.77 Weak 3D Z-Adjacent Merge Dry-run
input : Phase 5.74 2D cluster CSV (542 rows, sample-local p99)
output: 3D cluster CSV / JSON / MD  (dry-run only)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── constants ──────────────────────────────────────────────────────────────
PATCH_STRIDE = 16
EXPECTED_ROW_COUNT = 542
EXPECTED_PATIENTS = {"normal004", "normal013", "normal014"}

ALLOWED_Z_GAPS = {1, 2}
ALLOWED_CD_MULTIPLIERS = {1.5, 2.0}
ALLOWED_MAX_CLUSTERS_RANGE = (1, 3)

REQUIRED_COLUMNS = [
    "cluster_id", "patient_id", "local_z", "n_patches",
    "y0_min", "x0_min", "y1_max", "x1_max",
    "bbox_h", "bbox_w", "bbox_area",
    "max_patch_score", "top3_mean_patch_score", "top5_mean_patch_score", "mean_patch_score",
    "representative_y0", "representative_x0", "representative_y1", "representative_x1",
    "mean_pure_lung_patch_ratio", "min_pure_lung_patch_ratio", "max_pure_lung_patch_ratio",
    "cluster_rank_in_patient", "review_candidate_flag",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_INPUT_CSV = _PROJECT_ROOT / (
    "outputs/second-stage-lesion-refiner-v1/review_annotations"
    "/first_stage_padim_cluster_review"
    "/phase5_74_2d_cluster_dry_run_v1/phase5_74_2d_cluster_summary.csv"
)
_INPUT_JSON = _PROJECT_ROOT / (
    "outputs/second-stage-lesion-refiner-v1/review_annotations"
    "/first_stage_padim_cluster_review"
    "/phase5_74_2d_cluster_dry_run_v1/phase5_74_2d_cluster_summary.json"
)
_OUTPUT_ROOT_BASE = _PROJECT_ROOT / (
    "outputs/second-stage-lesion-refiner-v1/review_annotations"
    "/first_stage_padim_cluster_review"
)

REQUIRED_NOTE_KEYS = [
    "dry_run_only",
    "weak_3d_merge_run",
    "no_clustering_full_run",
    "threshold_not_finalized",
    "lesion_conclusion_forbidden",
    "stage2_holdout_unused",
    "v2_unused",
    "original_phase5_74_files_unmodified",
]


# ── output tag validation ──────────────────────────────────────────────────
def _validate_output_tag(tag: str) -> None:
    if not tag:
        sys.exit("[ERROR] --output-tag must not be empty.")
    if Path(tag).is_absolute():
        sys.exit(f"[ERROR] --output-tag must not be an absolute path: {tag!r}")
    if "/" in tag or "\\" in tag:
        sys.exit(f"[ERROR] --output-tag must not contain '/' or '\\': {tag!r}")
    if ".." in tag:
        sys.exit(f"[ERROR] --output-tag must not contain '..': {tag!r}")
    parts = Path(tag).parts
    if len(parts) != 1:
        sys.exit(f"[ERROR] --output-tag must be a single path component, got {parts!r}")
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", tag):
        sys.exit(
            f"[ERROR] --output-tag contains invalid characters. "
            f"Only alphanumeric, underscore, hyphen allowed: {tag!r}"
        )


# ── path guard ─────────────────────────────────────────────────────────────
def _guard_path(p: Path) -> None:
    for part in p.parts:
        pl = part.lower()
        if (
            "stage2_holdout" in pl
            or pl == "v2"
            or pl.startswith("v2v2")
            or pl.startswith("v2_")
            or pl.startswith("lesion_by_patient")
            or pl.startswith("crops_lesion")
            or "hard_negative" in pl
            or "nsclc_msd" in pl
            or "msd_lung" in pl
        ):
            sys.exit(f"[ERROR] Forbidden path segment '{part}' in {p}")


# ── Union-Find ─────────────────────────────────────────────────────────────
class _UF:
    def __init__(self, n: int):
        self._p = list(range(n))
        self._r = [0] * n

    def find(self, x: int) -> int:
        while self._p[x] != x:
            self._p[x] = self._p[self._p[x]]
            x = self._p[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self._r[rx] < self._r[ry]:
            rx, ry = ry, rx
        self._p[ry] = rx
        if self._r[rx] == self._r[ry]:
            self._r[rx] += 1
        return True


# ── geometry helpers ───────────────────────────────────────────────────────
def _bbox_iou(a: dict, b: dict) -> float:
    iy0 = max(a["y0_min"], b["y0_min"])
    ix0 = max(a["x0_min"], b["x0_min"])
    iy1 = min(a["y1_max"], b["y1_max"])
    ix1 = min(a["x1_max"], b["x1_max"])
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    if inter == 0:
        return 0.0
    union = a["bbox_area"] + b["bbox_area"] - inter
    return inter / union if union > 0 else 0.0


def _center_dist(a: dict, b: dict) -> float:
    cx_a = (a["x0_min"] + a["x1_max"]) / 2
    cy_a = (a["y0_min"] + a["y1_max"]) / 2
    cx_b = (b["x0_min"] + b["x1_max"]) / 2
    cy_b = (b["y0_min"] + b["y1_max"]) / 2
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


# ── merge ──────────────────────────────────────────────────────────────────
def _run_merge(df: pd.DataFrame, z_gap: int, cd_mult: float, allow_same_z: bool):
    """Union-find based weak 3D merge.
    Returns (uf, total_edge_count, adjacent_z_edge_count, same_z_edge_count).
    기본(allow_same_z=False)은 1 <= z_diff <= z_gap인 adjacent-z edge만 생성한다.
    """
    threshold = PATCH_STRIDE * cd_mult
    recs = df.to_dict("records")
    uf = _UF(len(recs))

    by_patient: dict = {}
    for i, r in enumerate(recs):
        by_patient.setdefault(r["patient_id"], []).append(i)

    adjacent_z_edge_count = 0
    same_z_edge_count = 0

    for idxs in by_patient.values():
        for ii in range(len(idxs)):
            for jj in range(ii + 1, len(idxs)):
                i, j = idxs[ii], idxs[jj]
                a, b = recs[i], recs[j]
                z_diff = abs(int(a["local_z"]) - int(b["local_z"]))

                if z_diff == 0:
                    if not allow_same_z:
                        continue
                    if _bbox_iou(a, b) > 0 or _center_dist(a, b) < threshold:
                        uf.union(i, j)
                        same_z_edge_count += 1
                elif z_diff <= z_gap:
                    if _bbox_iou(a, b) > 0 or _center_dist(a, b) < threshold:
                        uf.union(i, j)
                        adjacent_z_edge_count += 1

    total_edge_count = adjacent_z_edge_count + same_z_edge_count
    return uf, total_edge_count, adjacent_z_edge_count, same_z_edge_count


# ── representative selection ───────────────────────────────────────────────
def _pick_representative(group: pd.DataFrame) -> pd.Series:
    z_mid = (group["local_z"].min() + group["local_z"].max()) / 2
    g = group.copy()
    g["_z_dist"] = (g["local_z"] - z_mid).abs()
    return g.sort_values(
        ["top3_mean_patch_score", "max_patch_score", "n_patches", "_z_dist"],
        ascending=[False, False, False, True],
    ).iloc[0]


# ── build 3D cluster summary ───────────────────────────────────────────────
def _build_3d_summary(df: pd.DataFrame, uf: _UF, max_per_patient: int) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df["_comp"] = [uf.find(i) for i in range(len(df))]

    rows = []
    pid_counter: dict = {}

    for comp in sorted(df["_comp"].unique()):
        g = df[df["_comp"] == comp]
        pid = str(g["patient_id"].iloc[0])
        pid_counter[pid] = pid_counter.get(pid, 0) + 1
        cid = f"{pid}_3d_c{pid_counter[pid]}"

        z_min = int(g["local_z"].min())
        z_max = int(g["local_z"].max())
        z_span = z_max - z_min + 1
        n_2d = len(g)
        n_patches_total = int(g["n_patches"].sum())

        y0 = int(g["y0_min"].min())
        x0 = int(g["x0_min"].min())
        y1 = int(g["y1_max"].max())
        x1 = int(g["x1_max"].max())
        bh = y1 - y0
        bw = x1 - x0
        ba = bh * bw

        max_score = float(g["max_patch_score"].max())

        # top3_mean_patch_score_3d: 2D cluster들의 top3_mean_patch_score 상위 3개 평균
        top3_values = sorted(g["top3_mean_patch_score"].tolist(), reverse=True)
        t3 = float(np.mean(top3_values[:3]))
        # top5_mean_patch_score_3d: 2D cluster들의 top3_mean_patch_score 상위 5개 평균
        t5 = float(np.mean(top3_values[:5]))

        # mean_patch_score_3d: n_patches 가중 평균 (weighted mean)
        n_patches_arr = g["n_patches"].values
        mean_scores_arr = g["mean_patch_score"].values
        mean_s = float(np.sum(mean_scores_arr * n_patches_arr) / np.sum(n_patches_arr))

        rep = _pick_representative(g)

        mean_lr = float(g["mean_pure_lung_patch_ratio"].mean())
        min_lr = float(g["min_pure_lung_patch_ratio"].min())
        max_lr = float(g["max_pure_lung_patch_ratio"].max())

        t3_scores = g["top3_mean_patch_score"].tolist()
        mx_scores = g["max_patch_score"].tolist()
        sr_t3 = (max(t3_scores) / max(min(t3_scores), 1e-8)) if len(t3_scores) >= 2 else 1.0
        sr_mx = (max(mx_scores) / max(min(mx_scores), 1e-8)) if len(mx_scores) >= 2 else 1.0

        om = bool(z_span > 3)
        lb = bool(ba > 9216)
        le = bool(bw > 128 or bh > 128)
        cm = bool(n_2d >= 5 and ba > 9216)

        reasons = []
        if om:
            reasons.append(f"z_span={z_span}>3")
        if lb:
            reasons.append(f"bbox_area={ba}>9216")
        if le:
            reasons.append(f"bbox_w={bw} or bbox_h={bh}>128")
        if cm:
            reasons.append(f"n_2d={n_2d}>=5 and large_bbox")

        rows.append({
            "cluster3d_id": cid,
            "patient_id": pid,
            "z_min": z_min,
            "z_max": z_max,
            "z_span": z_span,
            "n_2d_clusters": n_2d,
            "n_patches_total": n_patches_total,
            "y0_min": y0,
            "x0_min": x0,
            "y1_max": y1,
            "x1_max": x1,
            "bbox_h": bh,
            "bbox_w": bw,
            "bbox_area": ba,
            "max_patch_score": max_score,
            "top3_mean_patch_score_3d": round(t3, 6),
            "top5_mean_patch_score_3d": round(t5, 6),
            "mean_patch_score_3d": round(mean_s, 6),
            "representative_2d_cluster_id": str(rep["cluster_id"]),
            "representative_local_z": int(rep["local_z"]),
            "representative_y0": int(rep["representative_y0"]),
            "representative_x0": int(rep["representative_x0"]),
            "representative_y1": int(rep["representative_y1"]),
            "representative_x1": int(rep["representative_x1"]),
            "mean_pure_lung_patch_ratio": round(mean_lr, 6),
            "min_pure_lung_patch_ratio": round(min_lr, 6),
            "max_pure_lung_patch_ratio": round(max_lr, 6),
            "score_ratio_top3": round(sr_t3, 4),
            "score_ratio_max": round(sr_mx, 4),
            "high_score_ratio_flag": bool(sr_t3 > 2.0),
            "overmerge_flag": om,
            "large_bbox_overmerge_flag": lb,
            "large_extent_overmerge_flag": le,
            "complex_merge_flag": cm,
            "overmerge_reason": "; ".join(reasons),
            "cluster3d_rank_in_patient": -1,
            "review_candidate_flag": False,
            "notes": (
                "weak_3d_merge dry-run only; "
                "sample-local p99 input; threshold not finalized"
            ),
        })

    df_3d = pd.DataFrame(rows)

    for pid, grp_idx in df_3d.groupby("patient_id").groups.items():
        ranked = (
            df_3d.loc[grp_idx]
            .sort_values("top3_mean_patch_score_3d", ascending=False)
            .index
        )
        for rank, idx in enumerate(ranked, 1):
            df_3d.at[idx, "cluster3d_rank_in_patient"] = rank
            df_3d.at[idx, "review_candidate_flag"] = rank <= max_per_patient

    return df_3d


# ── summary JSON ───────────────────────────────────────────────────────────
def _build_summary(
    df_2d: pd.DataFrame,
    df_3d: pd.DataFrame,
    uf: _UF,
    total_edge_count: int,
    adjacent_z_edge_count: int,
    same_z_edge_count: int,
    args,
    output_root: Path,
) -> dict:
    n_2d = len(df_2d)
    n_3d = len(df_3d)
    total_suspicious = 775

    pid_2d = {
        pid: int((df_2d["patient_id"] == pid).sum())
        for pid in sorted(df_2d["patient_id"].unique())
    }
    pid_3d = {
        pid: int((df_3d["patient_id"] == pid).sum())
        for pid in sorted(df_3d["patient_id"].unique())
    }

    top_per_patient = {}
    for pid in sorted(df_3d["patient_id"].unique()):
        cands = df_3d[
            (df_3d["patient_id"] == pid) & df_3d["review_candidate_flag"]
        ].sort_values("cluster3d_rank_in_patient")
        top_per_patient[pid] = [
            {
                "cluster3d_id": r["cluster3d_id"],
                "representative_local_z": int(r["representative_local_z"]),
                "top3_mean_patch_score_3d": float(r["top3_mean_patch_score_3d"]),
            }
            for _, r in cands.iterrows()
        ]

    df_tmp = df_2d.copy().reset_index(drop=True)
    df_tmp["_comp"] = [uf.find(i) for i in range(len(df_tmp))]
    z158 = set(
        df_tmp[
            (df_tmp["patient_id"] == "normal013") & (df_tmp["local_z"] == 158)
        ]["_comp"].tolist()
    )
    z159 = set(
        df_tmp[
            (df_tmp["patient_id"] == "normal013") & (df_tmp["local_z"] == 159)
        ]["_comp"].tolist()
    )
    normal013_merged = bool(len(z158 & z159) > 0)

    return {
        "output_tag": args.output_tag,
        "output_tag_validated": True,
        "output_root": str(output_root),
        "input_2d_cluster_count": n_2d,
        "weak_3d_cluster_count": n_3d,
        "reduction_rate_from_2d_cluster": round(1 - n_3d / n_2d, 4) if n_2d else 0,
        "reduction_rate_from_suspicious_patch": round(1 - n_3d / total_suspicious, 4),
        "patient_2d_cluster_count": pid_2d,
        "patient_3d_cluster_count": pid_3d,
        "patient_top_1_to_3_3d_clusters": top_per_patient,
        "singleton_2d_clusters_input": int((df_2d["n_patches"] == 1).sum()),
        "total_merge_edge_count": total_edge_count,
        "adjacent_z_merge_edge_count": adjacent_z_edge_count,
        "same_z_remerge_edge_count": same_z_edge_count,
        "allow_same_z_remerge": args.allow_same_z_remerge,
        "overmerge_flag_count": int(df_3d["overmerge_flag"].sum()),
        "high_score_ratio_flag_count": int(df_3d["high_score_ratio_flag"].sum()),
        "large_bbox_overmerge_flag_count": int(df_3d["large_bbox_overmerge_flag"].sum()),
        "large_extent_overmerge_flag_count": int(df_3d["large_extent_overmerge_flag"].sum()),
        "complex_merge_flag_count": int(df_3d["complex_merge_flag"].sum()),
        "normal013_z158_z159_merged": normal013_merged,
        "z_gap": args.z_gap,
        "center_distance_multiplier": args.center_distance_multiplier,
        "max_clusters_per_patient": args.max_clusters_per_patient,
        "patch_stride": PATCH_STRIDE,
        "argument_guard_note": (
            "z_gap, center_distance_multiplier, max_clusters_per_patient 값은 "
            "dry-run 후보값이며 확정 기준이 아님"
        ),
        "score_definitions": {
            "top3_mean_patch_score_3d": "2D cluster들의 top3_mean_patch_score 상위 3개 평균",
            "top5_mean_patch_score_3d": "2D cluster들의 top3_mean_patch_score 상위 5개 평균",
            "mean_patch_score_3d": "n_patches 가중 평균 (weighted mean)",
            "score_ratio_top3": "2D cluster top3_mean_patch_score 기준 max/min 비율",
        },
        "notes": {
            "input_basis": "sample-local p99 2D cluster from Phase 5.74",
            "dry_run_only": True,
            "weak_3d_merge_run": True,
            "no_clustering_full_run": True,
            "threshold_not_finalized": True,
            "lesion_conclusion_forbidden": True,
            "stage2_holdout_unused": True,
            "v2_unused": True,
            "original_phase5_74_files_unmodified": True,
        },
    }


# ── MD report ──────────────────────────────────────────────────────────────
def _build_md(summary: dict) -> str:
    lines = [
        "# Phase 5.77 Weak 3D Cluster Summary",
        "",
        f"- output_tag: {summary['output_tag']}",
        f"- output_root: {summary['output_root']}",
        f"- input 2D cluster count: {summary['input_2d_cluster_count']}",
        f"- weak 3D cluster count: {summary['weak_3d_cluster_count']}",
        f"- reduction rate from 2D: {summary['reduction_rate_from_2d_cluster']:.1%}",
        f"- reduction rate from suspicious: {summary['reduction_rate_from_suspicious_patch']:.1%}",
        f"- z_gap: {summary['z_gap']}",
        f"- center_distance_multiplier: {summary['center_distance_multiplier']}",
        f"- patch_stride: {summary['patch_stride']}",
        f"- allow_same_z_remerge: {summary['allow_same_z_remerge']}",
        f"- total_merge_edge_count: {summary['total_merge_edge_count']}",
        f"- adjacent_z_merge_edge_count: {summary['adjacent_z_merge_edge_count']}",
        f"- same_z_remerge_edge_count: {summary['same_z_remerge_edge_count']}",
        f"- overmerge_flag_count: {summary['overmerge_flag_count']}",
        f"- high_score_ratio_flag_count: {summary['high_score_ratio_flag_count']}",
        f"- large_bbox_overmerge_flag_count: {summary['large_bbox_overmerge_flag_count']}",
        f"- large_extent_overmerge_flag_count: {summary['large_extent_overmerge_flag_count']}",
        f"- complex_merge_flag_count: {summary['complex_merge_flag_count']}",
        f"- normal013 z158~159 merged: {summary['normal013_z158_z159_merged']}",
        "",
        "## Patient Summary",
        "",
    ]
    for pid in sorted(summary["patient_2d_cluster_count"]):
        n2d = summary["patient_2d_cluster_count"][pid]
        n3d = summary["patient_3d_cluster_count"].get(pid, 0)
        lines.append(f"### {pid}")
        lines.append(f"- 2D clusters: {n2d}  →  3D clusters: {n3d}")
        for t in summary["patient_top_1_to_3_3d_clusters"].get(pid, []):
            lines.append(
                f"  - {t['cluster3d_id']} "
                f"| z={t['representative_local_z']} "
                f"| top3_score={t['top3_mean_patch_score_3d']:.4f}"
            )
        lines.append("")
    lines += [
        "## Score Definitions",
        "",
        "- top3_mean_patch_score_3d: 2D cluster들의 top3_mean_patch_score 상위 3개 평균",
        "- top5_mean_patch_score_3d: 2D cluster들의 top3_mean_patch_score 상위 5개 평균",
        "- mean_patch_score_3d: n_patches 가중 평균 (weighted mean)",
        "- score_ratio_top3: 2D cluster top3_mean_patch_score 기준 max/min 비율",
        "",
        "## Notes",
        "",
        "- 입력: sample-local p99 기반 Phase 5.74 2D cluster (global threshold 아님)",
        "- threshold 확정 아님",
        "- 병변 성능 결론 아님",
        "- stage2_holdout / v2 미사용",
        "- 기존 Phase 5.74 파일 미수정",
        f"- argument_guard_note: {summary['argument_guard_note']}",
    ]
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 5.77 Weak 3D Merge Dry-run")
    parser.add_argument("--dry-run", action="store_true", required=True,
                        help="Must be specified. Dry-run mode only.")
    parser.add_argument("--z-gap", type=int, default=1)
    parser.add_argument("--center-distance-multiplier", type=float, default=2.0)
    parser.add_argument("--max-clusters-per-patient", type=int, default=3)
    parser.add_argument("--output-tag", type=str, default="phase5_77_weak_3d_merge_dry_run_v1")
    parser.add_argument("--allow-same-z-remerge", action="store_true", default=False,
                        help="Allow merging clusters on the same z-slice. Default: False.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # guard: --force 즉시 중단
    if args.force:
        sys.exit("[ERROR] --force is not allowed.")

    # argument guard
    if args.z_gap not in ALLOWED_Z_GAPS:
        sys.exit(f"[ERROR] --z-gap must be one of {sorted(ALLOWED_Z_GAPS)}, got {args.z_gap}")
    if args.center_distance_multiplier not in ALLOWED_CD_MULTIPLIERS:
        sys.exit(
            f"[ERROR] --center-distance-multiplier must be one of "
            f"{sorted(ALLOWED_CD_MULTIPLIERS)}, got {args.center_distance_multiplier}"
        )
    lo, hi = ALLOWED_MAX_CLUSTERS_RANGE
    if not (lo <= args.max_clusters_per_patient <= hi):
        sys.exit(
            f"[ERROR] --max-clusters-per-patient must be {lo}~{hi}, "
            f"got {args.max_clusters_per_patient}"
        )

    # output_tag validation + output_root 구성
    _validate_output_tag(args.output_tag)
    output_root = _OUTPUT_ROOT_BASE / args.output_tag

    # path guards
    for p in (_INPUT_CSV, _INPUT_JSON, output_root):
        _guard_path(p)

    # ── input validation ──────────────────────────────────────────────────
    if not _INPUT_CSV.exists():
        sys.exit(f"[ERROR] Input CSV not found: {_INPUT_CSV}")
    if not _INPUT_JSON.exists():
        sys.exit(f"[ERROR] Input JSON not found: {_INPUT_JSON}")

    df = pd.read_csv(_INPUT_CSV)

    if len(df) != EXPECTED_ROW_COUNT:
        sys.exit(f"[ERROR] Expected {EXPECTED_ROW_COUNT} rows, got {len(df)}")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        sys.exit(f"[ERROR] Missing required columns: {missing_cols}")

    if df["cluster_id"].duplicated().any():
        sys.exit("[ERROR] Duplicate cluster_id found.")

    actual_patients = set(df["patient_id"].unique())
    if actual_patients != EXPECTED_PATIENTS:
        sys.exit(
            f"[ERROR] Expected patients {sorted(EXPECTED_PATIENTS)}, "
            f"got {sorted(actual_patients)}"
        )

    try:
        df["local_z"] = df["local_z"].astype(int)
        for col in ("y0_min", "x0_min", "y1_max", "x1_max"):
            df[col] = df[col].astype(int)
    except (ValueError, TypeError) as e:
        sys.exit(f"[ERROR] Integer conversion failed: {e}")

    for col in ("max_patch_score", "top3_mean_patch_score",
                "top5_mean_patch_score", "mean_patch_score"):
        if df[col].isna().any() or np.isinf(df[col].values).any():
            sys.exit(f"[ERROR] NaN or Inf in column: {col}")

    print(f"[INFO] Input validation passed: {len(df)} rows, "
          f"patients={sorted(actual_patients)}")

    # ── output guard ──────────────────────────────────────────────────────
    if output_root.exists():
        sys.exit(f"[ERROR] Output root already exists: {output_root}")

    out_csv = output_root / "phase5_77_weak_3d_cluster_summary.csv"
    out_json = output_root / "phase5_77_weak_3d_cluster_summary.json"
    out_md = output_root / "phase5_77_weak_3d_cluster_summary.md"

    for f in (out_csv, out_json, out_md):
        if f.exists():
            sys.exit(f"[ERROR] Output file already exists: {f}")

    # ── merge + build ─────────────────────────────────────────────────────
    print(
        f"[INFO] Running weak 3D merge: "
        f"z_gap={args.z_gap}, "
        f"center_distance_multiplier={args.center_distance_multiplier}, "
        f"allow_same_z_remerge={args.allow_same_z_remerge}"
    )
    uf, total_edge_count, adjacent_z_edge_count, same_z_edge_count = _run_merge(
        df, args.z_gap, args.center_distance_multiplier, args.allow_same_z_remerge
    )
    df_3d = _build_3d_summary(df, uf, args.max_clusters_per_patient)

    # ── final validation ──────────────────────────────────────────────────
    if len(df_3d) == 0:
        sys.exit("[ERROR] df_3d has 0 rows.")
    if df_3d["cluster3d_id"].duplicated().any():
        sys.exit("[ERROR] Duplicate cluster3d_id found.")
    if df_3d["review_candidate_flag"].isna().any():
        sys.exit("[ERROR] review_candidate_flag contains NaN.")
    for pid, grp in df_3d.groupby("patient_id"):
        n_rev = int(grp["review_candidate_flag"].sum())
        if n_rev > args.max_clusters_per_patient:
            sys.exit(
                f"[ERROR] {pid} has {n_rev} review candidates "
                f"> {args.max_clusters_per_patient}"
            )
    if not args.allow_same_z_remerge and same_z_edge_count != 0:
        sys.exit(
            f"[ERROR] same_z_remerge_edge_count={same_z_edge_count} "
            f"but allow_same_z_remerge=False"
        )

    # ── df_3d 컬럼 존재 검증 (summary 생성 직전) ─────────────────────────────
    _required_3d_flag_cols = [
        "large_bbox_overmerge_flag",
        "large_extent_overmerge_flag",
        "complex_merge_flag",
    ]
    for _col in _required_3d_flag_cols:
        if _col not in df_3d.columns:
            sys.exit(f"[ERROR] df_3d missing required column: {_col}")

    summary = _build_summary(
        df, df_3d, uf,
        total_edge_count, adjacent_z_edge_count, same_z_edge_count,
        args, output_root,
    )

    for k in REQUIRED_NOTE_KEYS:
        if not summary["notes"].get(k):
            sys.exit(f"[ERROR] Summary notes missing required key: {k}")

    # ── summary 신규 flag count key/값 검증 ───────────────────────────────────
    _required_summary_keys = [
        "large_bbox_overmerge_flag_count",
        "large_extent_overmerge_flag_count",
        "complex_merge_flag_count",
    ]
    for _key in _required_summary_keys:
        if _key not in summary:
            sys.exit(f"[ERROR] Summary missing required key: {_key}")
        _val = summary[_key]
        if not isinstance(_val, int) or _val < 0:
            sys.exit(
                f"[ERROR] Summary key '{_key}' must be a non-negative integer, "
                f"got {_val!r}"
            )

    # ── create output root (after all validation + computation) ───────────
    output_root.mkdir(parents=True, exist_ok=False)

    # post-mkdir existence recheck (before first write)
    for f in (out_csv, out_json, out_md):
        if f.exists():
            sys.exit(f"[ERROR] Output file unexpectedly exists after mkdir: {f}")

    # ── save ──────────────────────────────────────────────────────────────
    df_3d.to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    with open(out_md, "w", encoding="utf-8") as fp:
        fp.write(_build_md(summary))

    print("[INFO] Done.")
    print(f"  2D clusters              : {len(df)}  →  3D clusters : {len(df_3d)}")
    print(f"  reduction                : {summary['reduction_rate_from_2d_cluster']:.1%}")
    print(f"  total_merge_edges        : {total_edge_count}")
    print(f"  adjacent_z_merge_edges   : {adjacent_z_edge_count}")
    print(f"  same_z_remerge_edges     : {same_z_edge_count}")
    print(f"  normal013 z158~159 merged: {summary['normal013_z158_z159_merged']}")
    print(f"  overmerge flags          : {summary['overmerge_flag_count']}")
    print(f"  large_bbox_overmerge     : {summary['large_bbox_overmerge_flag_count']}")
    print(f"  large_extent_overmerge   : {summary['large_extent_overmerge_flag_count']}")
    print(f"  complex_merge_flags      : {summary['complex_merge_flag_count']}")
    print(f"  output                   : {output_root}")


if __name__ == "__main__":
    main()
