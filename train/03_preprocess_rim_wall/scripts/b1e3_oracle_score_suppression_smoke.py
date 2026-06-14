"""
B1-E3: Oracle Score Suppression Simulation
EfficientNet-B0 v4_20 ROI branch의 dev-only oracle suppression simulation.

이 스크립트는:
- 실제 score CSV 수정 없음 (preview output만 생성)
- threshold 재계산 없음
- GPU 사용 없음
- stage2_holdout 접근 금지
- 원본 파일 mtime 변경 금지

Policy 정의:
  Baseline   : adjusted_score = original_score
  Policy_A   : oracle_ratio >= 0.05 AND lesion_ov == 0 → score * 0.5
  Policy_B   : oracle_ratio >= 0.05 AND lesion_ov == 0 → score * 0.0
"""

import os
import sys
import csv
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─── ALLOW GUARD ───────────────────────────────────────────────────────────────
ALLOW_REAL_PROCESSING = True
# ───────────────────────────────────────────────────────────────────────────────

PROJ_ROOT = Path(__file__).resolve().parents[1]

B1E1_ROOT = (
    PROJ_ROOT / "outputs" / "position-aware-padim-v1"
    / "oracle_vessel_suppression_effb0_v4_20_dev_only"
    / "b1e1_oracle_mask_preflight_v1"
)
B1E2_ROOT = (
    PROJ_ROOT / "outputs" / "position-aware-padim-v1"
    / "oracle_vessel_suppression_effb0_v4_20_dev_only"
    / "b1e2_patch_oracle_overlap_dryrun_v1"
)
OUTPUT_ROOT = (
    PROJ_ROOT / "outputs" / "position-aware-padim-v1"
    / "oracle_vessel_suppression_effb0_v4_20_dev_only"
    / "b1e3_oracle_score_suppression_smoke_v1"
)
LESION_SPLIT_CSV = (
    PROJ_ROOT / "outputs" / "second-stage-lesion-refiner-v1"
    / "splits" / "lesion_stage_split_v1.csv"
)
THRESHOLD_JSON = (
    PROJ_ROOT / "experiments" / "efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs" / "evaluation" / "normal_val_thresholds" / "normal_val_threshold.json"
)

THRESHOLD_P95 = 13.231265
THRESHOLD_P99 = 15.472385
ORACLE_SUPPRESS_THRESHOLD = 0.05
LESION_RISK_PATIENTS = {"LUNG1-020"}

TOPK_LIST = [1, 5, 10, 20]


def abort(msg: str) -> None:
    print(f"[ABORT] {msg}", file=sys.stderr)
    sys.exit(2)


def mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else -1.0


def read_csv_dicts(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_oracle_mask(ct: np.ndarray, roi: np.ndarray, lesion):
    roi_bool = roi > 0
    hu_ge0 = roi_bool & (ct >= 0)
    if lesion is not None:
        les_bool = lesion > 0
        oracle = hu_ge0 & (~les_bool)
    else:
        les_bool = None
        oracle = hu_ge0
    return roi_bool, oracle, les_bool


def topk_survival(les_flags: list, scores: list, k: int) -> bool:
    """상위 k위 패치 중 lesion 패치가 1개라도 있으면 True."""
    if not any(les_flags):
        return None
    sorted_idx = sorted(range(len(scores)), key=lambda i: -scores[i])
    top_k_idx = set(sorted_idx[:k])
    for i, flag in enumerate(les_flags):
        if flag and i in top_k_idx:
            return True
    return False


def process_patient(row: dict) -> tuple:
    """
    Returns:
        patch_rows (list of dict): 전체 패치 행
        summary (dict): 환자 요약
        errors (list of str): 오류 메시지
    """
    pid  = row["patient_id"]
    sid  = row["safe_id"]
    role = row["role"]
    errors = []

    ct_path    = Path(row["ct_path"])
    roi_path   = Path(row["roi_path"])
    les_path   = Path(row["lesion_mask_path"]) if row.get("lesion_mask_path") else None
    score_path = Path(row["score_csv_path"])

    # ── 파일 로드 ─────────────────────────────────────────────────────────────
    try:
        ct  = np.load(str(ct_path),  mmap_mode="r")
        roi = np.load(str(roi_path), mmap_mode="r")
        les = np.load(str(les_path), mmap_mode="r") if les_path else None
    except Exception as e:
        errors.append(f"load: {e}")
        return [], _empty_summary(pid, sid, role), errors

    ct_z, ct_y, ct_x = ct.shape

    # ── oracle mask 계산 ─────────────────────────────────────────────────────
    ct_arr  = np.asarray(ct,  dtype=np.int16)
    roi_arr = np.asarray(roi, dtype=np.uint8)
    les_arr = np.asarray(les, dtype=np.uint8) if les is not None else None
    _, oracle, les_bool = compute_oracle_mask(ct_arr, roi_arr, les_arr)

    # ── score CSV 로드 ───────────────────────────────────────────────────────
    try:
        import pandas as pd
        df = pd.read_csv(str(score_path))
    except Exception as e:
        errors.append(f"score_csv: {e}")
        return [], _empty_summary(pid, sid, role), errors

    n_total = len(df)

    # ── patch 좌표 계산 ──────────────────────────────────────────────────────
    results = []
    z_cache_oracle = {}
    z_cache_les    = {}

    for _, pr in df.iterrows():
        z  = int(pr["local_z"])
        y0 = int(pr["y0"])
        x0 = int(pr["x0"])
        y1 = int(pr["y1"])
        x1 = int(pr["x1"])
        score = float(pr["padim_score"])

        coord_valid = (
            0 <= z < ct_z and
            0 <= y0 < y1 <= ct_y and
            0 <= x0 < x1 <= ct_x
        )

        oracle_ov = les_ov = 0
        patch_area = (y1 - y0) * (x1 - x0)

        if coord_valid:
            if z not in z_cache_oracle:
                z_cache_oracle[z] = oracle[z].copy()
                z_cache_les[z] = (les_bool[z].copy()
                                  if les_bool is not None else None)
            oracle_crop = z_cache_oracle[z][y0:y1, x0:x1]
            oracle_ov = int(oracle_crop.sum())
            if z_cache_les[z] is not None:
                les_crop = z_cache_les[z][y0:y1, x0:x1]
                les_ov = int(les_crop.sum())

        oracle_ratio = oracle_ov / patch_area if patch_area > 0 else 0.0
        les_ratio    = les_ov    / patch_area if patch_area > 0 else 0.0

        eligible = (oracle_ratio >= ORACLE_SUPPRESS_THRESHOLD) and (les_ov == 0)

        adj_a = score * 0.5 if eligible else score
        adj_b = 0.0         if eligible else score

        results.append({
            "z": z, "y0": y0, "x0": x0, "y1": y1, "x1": x1,
            "score": score,
            "oracle_ratio": oracle_ratio,
            "les_ov": les_ov,
            "les_ratio": les_ratio,
            "eligible": eligible,
            "adj_a": adj_a,
            "adj_b": adj_b,
        })

    # ── 환자 내 rank 계산 ────────────────────────────────────────────────────
    n = len(results)
    rank_before   = _rank_desc([r["score"] for r in results])
    rank_after_a  = _rank_desc([r["adj_a"] for r in results])
    rank_after_b  = _rank_desc([r["adj_b"] for r in results])

    # ── 환자 summary 집계 ────────────────────────────────────────────────────
    n_eligible = sum(1 for r in results if r["eligible"])
    eligible_ratio = round(n_eligible / n_total, 6) if n_total else 0.0

    scores_orig = [r["score"] for r in results]
    scores_a    = [r["adj_a"] for r in results]
    scores_b    = [r["adj_b"] for r in results]

    max_before = float(max(scores_orig)) if scores_orig else 0.0
    max_after_a = float(max(scores_a))   if scores_a    else 0.0
    max_after_b = float(max(scores_b))   if scores_b    else 0.0

    # 상위 10/50 중 eligible 수
    sorted_by_orig = sorted(range(n), key=lambda i: -results[i]["score"])
    top10_elig = sum(1 for i in sorted_by_orig[:10]  if results[i]["eligible"])
    top50_elig = sum(1 for i in sorted_by_orig[:50]  if results[i]["eligible"])

    les_flags = [r["les_ov"] > 0 for r in results]
    les_patch_count = sum(les_flags)

    lesion_top1_before = None
    lesion_top1_after_a = None
    lesion_top1_after_b = None
    if les_patch_count > 0:
        les_scores_orig = [r["score"] for r in results if r["les_ov"] > 0]
        les_scores_a    = [r["adj_a"] for r in results if r["les_ov"] > 0]
        les_scores_b    = [r["adj_b"] for r in results if r["les_ov"] > 0]
        lesion_top1_before  = round(float(max(les_scores_orig)), 6)
        lesion_top1_after_a = round(float(max(les_scores_a)),    6)
        lesion_top1_after_b = round(float(max(les_scores_b)),    6)

    surv_base = {}
    surv_a = {}
    surv_b = {}
    for k in TOPK_LIST:
        surv_base[k] = topk_survival(les_flags, scores_orig, k)
        surv_a[k] = topk_survival(les_flags, scores_a, k)
        surv_b[k] = topk_survival(les_flags, scores_b, k)

    lesion_risk_flag = pid in LESION_RISK_PATIENTS

    summary = {
        "patient_id": pid,
        "safe_id": sid,
        "role": role,
        "n_total_patches": n_total,
        "n_suppression_eligible": n_eligible,
        "eligible_patch_ratio": eligible_ratio,
        "max_score_before": round(max_before, 6),
        "max_score_after_policy_a": round(max_after_a, 6),
        "max_score_after_policy_b": round(max_after_b, 6),
        "top10_eligible_count": top10_elig,
        "top50_eligible_count": top50_elig,
        "lesion_patch_count": les_patch_count,
        "lesion_top1_score_before": lesion_top1_before,
        "lesion_top1_score_after_policy_a": lesion_top1_after_a,
        "lesion_top1_score_after_policy_b": lesion_top1_after_b,
        "lesion_topk_survival_k1_baseline":  surv_base[1],
        "lesion_topk_survival_k5_baseline":  surv_base[5],
        "lesion_topk_survival_k10_baseline": surv_base[10],
        "lesion_topk_survival_k20_baseline": surv_base[20],
        "lesion_topk_survival_k1_policy_a":  surv_a[1],
        "lesion_topk_survival_k5_policy_a":  surv_a[5],
        "lesion_topk_survival_k10_policy_a": surv_a[10],
        "lesion_topk_survival_k20_policy_a": surv_a[20],
        "lesion_topk_survival_k1_policy_b":  surv_b[1],
        "lesion_topk_survival_k5_policy_b":  surv_b[5],
        "lesion_topk_survival_k10_policy_b": surv_b[10],
        "lesion_topk_survival_k20_policy_b": surv_b[20],
        "lesion_risk_case_flag": lesion_risk_flag,
    }

    # ── patch_rows 전체 반환 ─────────────────────────────────────────────────
    patch_rows = []
    for i, r in enumerate(results):
        patch_rows.append({
            "patient_id": pid,
            "safe_id": sid,
            "role": role,
            "local_z": r["z"],
            "y0": r["y0"], "x0": r["x0"], "y1": r["y1"], "x1": r["x1"],
            "original_score": round(r["score"], 6),
            "oracle_like_vessel_overlap_ratio": round(r["oracle_ratio"], 6),
            "lesion_overlap_voxel_count": r["les_ov"],
            "lesion_overlap_ratio": round(r["les_ratio"], 6),
            "suppression_eligible": r["eligible"],
            "adjusted_score_policy_a": round(r["adj_a"], 6),
            "adjusted_score_policy_b": round(r["adj_b"], 6),
            "rank_before": rank_before[i],
            "rank_after_policy_a": rank_after_a[i],
            "rank_after_policy_b": rank_after_b[i],
        })

    return patch_rows, summary, errors


def _rank_desc(scores: list) -> list:
    """내림차순 rank (1-based). 동점은 같은 rank."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: -scores[i])
    rank = [0] * n
    cur_rank = 1
    for pos, idx in enumerate(order):
        if pos > 0 and scores[idx] == scores[order[pos - 1]]:
            rank[idx] = rank[order[pos - 1]]
        else:
            rank[idx] = cur_rank
        cur_rank += 1
    return rank


def _empty_summary(pid, sid, role) -> dict:
    d = {
        "patient_id": pid, "safe_id": sid, "role": role,
        "n_total_patches": 0, "n_suppression_eligible": 0,
        "eligible_patch_ratio": 0,
        "max_score_before": 0, "max_score_after_policy_a": 0, "max_score_after_policy_b": 0,
        "top10_eligible_count": 0, "top50_eligible_count": 0,
        "lesion_patch_count": 0,
        "lesion_top1_score_before": None,
        "lesion_top1_score_after_policy_a": None,
        "lesion_top1_score_after_policy_b": None,
        "lesion_risk_case_flag": pid in LESION_RISK_PATIENTS,
    }
    for k in TOPK_LIST:
        d[f"lesion_topk_survival_k{k}_baseline"] = None
        d[f"lesion_topk_survival_k{k}_policy_a"] = None
        d[f"lesion_topk_survival_k{k}_policy_b"] = None
    return d


def compute_policy_comparison(all_summaries: list, all_patch_rows: list) -> list:
    """3개 policy 비교 집계."""
    lesion_pids = {s["patient_id"] for s in all_summaries if s["lesion_patch_count"] > 0}
    normal_pids = {s["patient_id"] for s in all_summaries if s["role"] == "normal_control"}

    total_patches = sum(s["n_total_patches"] for s in all_summaries)
    n_elig = sum(s["n_suppression_eligible"] for s in all_summaries)
    elig_ratio = round(n_elig / total_patches, 6) if total_patches else 0.0

    rows = []
    for policy in ["baseline", "policy_a", "policy_b"]:
        if policy == "baseline":
            score_key = "original_score"
            adj_key   = "original_score"
            n_elig_pol = 0
            elig_ratio_pol = 0.0
        elif policy == "policy_a":
            score_key = "original_score"
            adj_key   = "adjusted_score_policy_a"
            n_elig_pol = n_elig
            elig_ratio_pol = elig_ratio
        else:
            score_key = "original_score"
            adj_key   = "adjusted_score_policy_b"
            n_elig_pol = n_elig
            elig_ratio_pol = elig_ratio

        # eligible 패치 평균 점수 감소
        elig_drops = []
        for pr in all_patch_rows:
            if pr["suppression_eligible"]:
                orig  = pr["original_score"]
                after = pr[adj_key]
                elig_drops.append(orig - after)
        mean_drop_elig = round(float(np.mean(elig_drops)), 6) if elig_drops else 0.0

        # normal 환자 최대 점수 감소 평균
        normal_drops = []
        for s in all_summaries:
            if s["patient_id"] in normal_pids:
                before = s["max_score_before"]
                if policy == "baseline":
                    after = before
                elif policy == "policy_a":
                    after = s["max_score_after_policy_a"]
                else:
                    after = s["max_score_after_policy_b"]
                normal_drops.append(before - after)
        mean_normal_drop = round(float(np.mean(normal_drops)), 6) if normal_drops else 0.0

        # lesion 환자 중 eligible이고 lesion이 아닌 high score 패치 점수감소 평균
        les_nonles_drops = []
        for pr in all_patch_rows:
            if pr["patient_id"] in lesion_pids and pr["suppression_eligible"]:
                if pr["lesion_overlap_voxel_count"] == 0:
                    orig  = pr["original_score"]
                    after = pr[adj_key]
                    les_nonles_drops.append(orig - after)
        mean_les_nonles = round(float(np.mean(les_nonles_drops)), 6) if les_nonles_drops else 0.0

        # lesion topk survival 평균 (lesion 환자만)
        surv_means = {}
        for k in TOPK_LIST:
            col = f"lesion_topk_survival_k{k}_policy_{policy[-1]}" if policy != "baseline" else None
            vals = []
            for s in all_summaries:
                if s["patient_id"] in lesion_pids:
                    if policy == "baseline":
                        v = s.get(f"lesion_topk_survival_k{k}_baseline")
                    else:
                        v = s.get(f"lesion_topk_survival_k{k}_policy_{policy[-1]}")
                    if v is not None:
                        vals.append(1.0 if v else 0.0)
            valid_vals = [v for v in vals if v is not None]
            surv_means[k] = round(float(np.mean(valid_vals)), 6) if valid_vals else None

        rows.append({
            "policy": policy,
            "n_eligible_patches": n_elig_pol,
            "eligible_patch_ratio": elig_ratio_pol,
            "mean_score_drop_eligible": mean_drop_elig,
            "normal_max_score_drop_mean": mean_normal_drop,
            "lesion_nonlesion_highscore_drop_mean": mean_les_nonles,
            "lesion_topk_survival_mean_k1":  surv_means[1],
            "lesion_topk_survival_mean_k5":  surv_means[5],
            "lesion_topk_survival_mean_k10": surv_means[10],
            "lesion_topk_survival_mean_k20": surv_means[20],
        })

    return rows


def build_report(all_summaries: list, policy_rows: list,
                 n_errors: int, elapsed: float,
                 b1e2_ge005_by_pid: dict = None,
                 b1e2_total_ge005: int = 0) -> str:
    lesion_pids = [s["patient_id"] for s in all_summaries if s["lesion_patch_count"] > 0]
    normal_pids = [s["patient_id"] for s in all_summaries if s["role"] == "normal_control"]

    # 결론 판정
    pol_a = next((r for r in policy_rows if r["policy"] == "policy_a"), {})
    pol_b = next((r for r in policy_rows if r["policy"] == "policy_b"), {})

    surv_a1 = pol_a.get("lesion_topk_survival_mean_k1")
    surv_b1 = pol_b.get("lesion_topk_survival_mean_k1")
    normal_drop_a = pol_a.get("normal_max_score_drop_mean", 0)
    normal_drop_b = pol_b.get("normal_max_score_drop_mean", 0)

    # 결론 기준:
    #   GO_NEXT: lesion k1 survival >= 0.8 AND normal_drop > 0.5
    #   CAUTION: lesion k1 survival >= 0.5 OR (lesion k20 survival >= 0.8 AND normal_drop > 0.1)
    #   NO_GO: lesion k1 survival < 0.5
    surv_a20 = pol_a.get("lesion_topk_survival_mean_k20")
    conclusion_a = _judge_conclusion(surv_a1, surv_a20, normal_drop_a)
    conclusion_b = _judge_conclusion(surv_b1,
                                     pol_b.get("lesion_topk_survival_mean_k20"),
                                     normal_drop_b)
    overall = conclusion_a if conclusion_a != "GO_NEXT" else (
        "GO_NEXT" if conclusion_b != "NO_GO" else "CAUTION"
    )

    lung020 = next((s for s in all_summaries if s["patient_id"] == "LUNG1-020"), None)

    lines = [
        "# B1-E3 Oracle Score Suppression Simulation Report",
        "",
        "**⚠️ 이 보고서는 dev-only simulation 결과입니다. 실제 score CSV는 수정되지 않았습니다.**",
        "",
        "## 1. 실험 개요",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        "| Branch | EfficientNet-B0 v4_20 ROI |",
        "| Fixed threshold p95 | 13.231265 |",
        "| Fixed threshold p99 | 15.472385 |",
        "| suppression threshold | oracle_ratio >= 0.05 AND lesion_ov == 0 |",
        f"| 처리 환자 수 | {len(all_summaries)} |",
        f"| 오류 수 | {n_errors} |",
        f"| 실행 시간 | {elapsed:.1f}초 |",
        "",
        "## 2. 주의사항",
        "",
        "- 이 단계는 **dev-only simulation**이며 실제 score 수정이 아닙니다.",
        "- oracle-like mask는 CT HU>=0 voxel 기반 추정이며, **true vessel ground truth가 아닙니다**.",
        "- FP 억제 효과 추정치만 제공하며, 임상 적용 판단에 사용할 수 없습니다.",
        "",
        "## 3. Suppression Eligible 패치 현황",
        "",
    ]

    for s in all_summaries:
        lines.append(
            f"- {s['patient_id']} ({s['role']}): "
            f"eligible={s['n_suppression_eligible']}/{s['n_total_patches']} "
            f"({s['eligible_patch_ratio']*100:.2f}%)"
        )

    lines += [
        "",
        "## 4. Policy_A vs Policy_B 비교",
        "",
        "| Policy | Eligible 패치 | 평균 점수감소 | Normal max 점수감소 | Lesion non-lesion 점수감소 |",
        "|--------|--------------|--------------|---------------------|--------------------------|",
    ]
    for r in policy_rows:
        lines.append(
            f"| {r['policy']} | {r['n_eligible_patches']} | "
            f"{r['mean_score_drop_eligible']} | "
            f"{r['normal_max_score_drop_mean']} | "
            f"{r['lesion_nonlesion_highscore_drop_mean']} |"
        )

    lines += [
        "",
        "## 5. Lesion Recall 보존 (topk survival, 환자 평균)",
        "",
        "| Policy | k=1 | k=5 | k=10 | k=20 |",
        "|--------|-----|-----|------|------|",
    ]
    for r in policy_rows:
        lines.append(
            f"| {r['policy']} | "
            f"{_fmt_surv(r['lesion_topk_survival_mean_k1'])} | "
            f"{_fmt_surv(r['lesion_topk_survival_mean_k5'])} | "
            f"{_fmt_surv(r['lesion_topk_survival_mean_k10'])} | "
            f"{_fmt_surv(r['lesion_topk_survival_mean_k20'])} |"
        )

    lines += [
        "",
        "### 환자별 Lesion Topk Survival (Baseline / Policy_A / Policy_B)",
        "",
        "| 환자 | base_k1 | base_k20 | A_k1 | A_k20 | B_k1 | B_k20 |",
        "|------|---------|----------|------|-------|------|-------|",
    ]
    for s in all_summaries:
        if s["lesion_patch_count"] > 0:
            lines.append(
                f"| {s['patient_id']} | "
                f"{_fmt_surv(s['lesion_topk_survival_k1_baseline'])} | "
                f"{_fmt_surv(s['lesion_topk_survival_k20_baseline'])} | "
                f"{_fmt_surv(s['lesion_topk_survival_k1_policy_a'])} | "
                f"{_fmt_surv(s['lesion_topk_survival_k20_policy_a'])} | "
                f"{_fmt_surv(s['lesion_topk_survival_k1_policy_b'])} | "
                f"{_fmt_surv(s['lesion_topk_survival_k20_policy_b'])} |"
            )

    lines += [
        "",
        "## 6. Normal/FP 환자 점수 감소",
        "",
        "| 환자 | max_before | max_after_A | max_after_B | drop_A | drop_B |",
        "|------|-----------|-------------|-------------|--------|--------|",
    ]
    for s in all_summaries:
        if s["patient_id"] in normal_pids or s["role"] in ("normal_control", "normal"):
            drop_a = round(s["max_score_before"] - s["max_score_after_policy_a"], 4)
            drop_b = round(s["max_score_before"] - s["max_score_after_policy_b"], 4)
            lines.append(
                f"| {s['patient_id']} | "
                f"{s['max_score_before']} | "
                f"{s['max_score_after_policy_a']} | "
                f"{s['max_score_after_policy_b']} | "
                f"{drop_a} | {drop_b} |"
            )

    lines += [
        "",
        "## 7. B1-E2 ge005 vs B1-E3 Eligible 대조",
        "",
        f"- B1-E2 전체 ge005 패치 수: {b1e2_total_ge005}",
        f"- B1-E3 전체 eligible 패치 수: {sum(s['n_suppression_eligible'] for s in all_summaries)}",
        f"- 조건: eligible <= ge005 ({'OK' if b1e2_ge005_by_pid else 'B1-E2 데이터 없음'})",
        "",
        "| 환자 | B1-E2 ge005 | B1-E3 eligible | OK? |",
        "|------|------------|----------------|-----|",
    ]
    if b1e2_ge005_by_pid:
        for s in all_summaries:
            ge005 = b1e2_ge005_by_pid.get(s["patient_id"], "N/A")
            elig  = s["n_suppression_eligible"]
            ok    = "OK" if ge005 == "N/A" or elig <= ge005 else "FAIL"
            lines.append(f"| {s['patient_id']} | {ge005} | {elig} | {ok} |")

    lines += ["", "## 8. LUNG1-020 Lesion Risk 해석", ""]
    if lung020:
        lines += [
            "LUNG1-020은 **lesion risk case**로 분류됩니다.",
            "",
            f"- lesion_patch_count: {lung020['lesion_patch_count']}",
            f"- n_suppression_eligible: {lung020['n_suppression_eligible']}",
            f"- eligible_patch_ratio: {lung020['eligible_patch_ratio']*100:.2f}%",
            f"- lesion_top1_score_before: {lung020['lesion_top1_score_before']}",
            f"- lesion_top1_score_after_A: {lung020['lesion_top1_score_after_policy_a']}",
            f"- lesion_top1_score_after_B: {lung020['lesion_top1_score_after_policy_b']}",
            f"- k1 survival (A): {_fmt_surv(lung020['lesion_topk_survival_k1_policy_a'])}",
            f"- k1 survival (B): {_fmt_surv(lung020['lesion_topk_survival_k1_policy_b'])}",
            "",
            "oracle-like mask와 lesion mask가 겹치는 영역이 존재하므로, "
            "실제 vessel suppression 적용 시 lesion 패치 일부가 누락될 위험이 있습니다. "
            "suppression_eligible 조건(lesion_ov == 0)으로 lesion 패치 자체는 억제되지 않으나, "
            "인접 패치의 순위 변동으로 lesion recall이 변화할 수 있습니다.",
        ]
    else:
        lines.append("LUNG1-020 데이터 없음.")

    lines += [
        "",
        "## 9. 결론",
        "",
        f"- **Policy_A 결론**: {conclusion_a}",
        f"- **Policy_B 결론**: {conclusion_b}",
        f"- **종합 결론**: **{overall}**",
        "",
        "| 판정 | 의미 |",
        "|------|------|",
        "| GO_NEXT | oracle suppression이 실질적으로 유의미 |",
        "| CAUTION | 일부 효과 있으나 lesion risk 존재 |",
        "| NO_GO | FP 감소보다 lesion 손상이 큼 |",
        "",
        "---",
        "",
        "**참고**: 이 결과는 dev-only simulation이며, "
        "oracle-like mask 기반 추정치입니다. "
        "실제 vessel suppression 효과는 true GT mask 기반 평가로만 확인 가능합니다.",
    ]

    return "\n".join(lines) + "\n"


def _judge_conclusion(surv_k1, surv_k20, normal_drop) -> str:
    if surv_k1 is None:
        return "CAUTION"
    if surv_k1 >= 0.8 and (normal_drop is not None and normal_drop > 0.5):
        return "GO_NEXT"
    elif surv_k1 >= 0.5 or (surv_k20 is not None and surv_k20 >= 0.8
                              and normal_drop is not None and normal_drop > 0.1):
        return "CAUTION"
    else:
        return "NO_GO"


def _fmt_surv(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return "True" if v else "False"
    return f"{v:.3f}"


def main() -> None:
    if not ALLOW_REAL_PROCESSING:
        abort("ALLOW_REAL_PROCESSING=False: guard 활성. "
              "True로 변경 후 실행하세요.")

    # ── 전제 조건 확인 (OUTPUT_ROOT 생성 전) ──────────────────────────────
    if not (B1E1_ROOT / "DONE").exists():
        abort(f"B1-E1 DONE 파일 없음: {B1E1_ROOT}")
    if not (B1E2_ROOT / "DONE").exists():
        abort(f"B1-E2 DONE 파일 없음: {B1E2_ROOT}")

    b1e1_targets_csv = B1E1_ROOT / "b1e1_oracle_mask_preflight_targets.csv"
    if not b1e1_targets_csv.exists():
        abort(f"B1-E1 targets CSV 없음: {b1e1_targets_csv}")
    if not LESION_SPLIT_CSV.exists():
        abort(f"LESION_SPLIT_CSV 없음: {LESION_SPLIT_CSV}")

    # ── stage2 holdout denylist ──────────────────────────────────────────────
    holdout_pids: set = set()
    holdout_sids: set = set()
    with open(LESION_SPLIT_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("stage_split", "") == "stage2_holdout":
                holdout_pids.add(row["patient_id"].strip())
                holdout_sids.add(row["safe_id"].strip())

    # ── B1-E1 targets 읽기 ──────────────────────────────────────────────────
    targets = read_csv_dicts(b1e1_targets_csv)

    for t in targets:
        if t["patient_id"] in holdout_pids or t["safe_id"] in holdout_sids:
            abort(f"stage2_holdout 교집합 발견: {t['patient_id']}")

    # ── OUTPUT_ROOT 생성 (입력 검증 완료 후) ─────────────────────────────────
    if OUTPUT_ROOT.exists():
        abort(f"output root 이미 존재: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    # ── mtime 스냅샷 (보호 대상 확장) ──────────────────────────────────────
    b1e2_patient_csv = B1E2_ROOT / "b1e2_patch_oracle_overlap_dryrun_patient_summary.csv"
    b1e2_thresh_csv  = B1E2_ROOT / "b1e2_patch_oracle_overlap_dryrun_threshold_table.csv"
    b1e2_report_md   = B1E2_ROOT / "b1e2_patch_oracle_overlap_dryrun_report.md"
    b1e2_summ_json   = B1E2_ROOT / "b1e2_patch_oracle_overlap_dryrun_summary.json"
    b1e2_done        = B1E2_ROOT / "DONE"

    protected_paths = [
        THRESHOLD_JSON,
        LESION_SPLIT_CSV,
        b1e1_targets_csv,
        b1e2_done,
        b1e2_summ_json,
        b1e2_patient_csv,
        b1e2_thresh_csv,
        b1e2_report_md,
    ]
    for t in targets:
        protected_paths.append(Path(t["score_csv_path"]))
        protected_paths.append(Path(t["ct_path"]))
        protected_paths.append(Path(t["roi_path"]))
        if t.get("lesion_mask_path"):
            protected_paths.append(Path(t["lesion_mask_path"]))

    mtime_before = {str(p): mtime(p) for p in protected_paths}

    # ── B1-E2 ge005 대조용 데이터 로드 ──────────────────────────────────────
    b1e2_ge005_by_pid: dict = {}
    b1e2_total_ge005 = 0
    try:
        b1e2_patient_rows = read_csv_dicts(b1e2_patient_csv)
        for r in b1e2_patient_rows:
            pid_b2 = r["patient_id"]
            n_ge005 = int(r.get("n_oracle_overlap_ge005", 0))
            b1e2_ge005_by_pid[pid_b2] = n_ge005
            b1e2_total_ge005 += n_ge005
    except Exception as e:
        print(f"[WARN] B1-E2 patient summary 읽기 실패: {e}", file=sys.stderr)

    # ── 환자별 처리 ──────────────────────────────────────────────────────────
    all_patch_rows: list = []
    all_summaries:  list = []
    all_errors:     list = []
    t0 = time.time()

    for t in targets:
        pid = t["patient_id"]
        print(f"  처리 중: {pid} ({t['role']}) ...", flush=True)
        t1 = time.time()
        patch_rows, summary, errors = process_patient(t)
        elapsed = time.time() - t1
        print(
            f"    완료: {elapsed:.1f}s, "
            f"eligible={summary['n_suppression_eligible']}/{summary['n_total_patches']}, "
            f"lesion={summary['lesion_patch_count']}",
            flush=True,
        )
        all_patch_rows.extend(patch_rows)
        all_summaries.append(summary)
        for e in errors:
            all_errors.append({"patient_id": pid, "stage": "process", "msg": e})

    total_elapsed = time.time() - t0

    # ── B1-E2 ge005 대조 검사 ────────────────────────────────────────────────
    total_eligible_tmp = sum(s["n_suppression_eligible"] for s in all_summaries)
    b1e2_check_failures = []
    if b1e2_total_ge005 > 0:
        if total_eligible_tmp > b1e2_total_ge005:
            b1e2_check_failures.append(
                f"전체: B1-E3 eligible({total_eligible_tmp}) > B1-E2 ge005({b1e2_total_ge005})"
            )
        for s in all_summaries:
            pid_s = s["patient_id"]
            b2_ge005 = b1e2_ge005_by_pid.get(pid_s)
            if b2_ge005 is not None and s["n_suppression_eligible"] > b2_ge005:
                b1e2_check_failures.append(
                    f"{pid_s}: eligible({s['n_suppression_eligible']}) > B1E2 ge005({b2_ge005})"
                )
    if b1e2_check_failures:
        for msg in b1e2_check_failures:
            print(f"[FAIL] B1-E2 대조 불일치: {msg}", file=sys.stderr)
        abort("B1-E3 eligible 수가 B1-E2 ge005 수보다 큼. overlap 재계산 불일치 가능성.")

    # ── mtime 사후 검증 ──────────────────────────────────────────────────────
    mtime_violations = []
    for ps, before in mtime_before.items():
        after = mtime(Path(ps))
        if before != after:
            mtime_violations.append(f"{ps}: {before} → {after}")
    if mtime_violations:
        abort("원본 파일 mtime 변경 감지:\n" + "\n".join(mtime_violations))

    # ── patch_preview CSV ────────────────────────────────────────────────────
    patch_fields = [
        "patient_id", "safe_id", "role",
        "local_z", "y0", "x0", "y1", "x1",
        "original_score",
        "oracle_like_vessel_overlap_ratio",
        "lesion_overlap_voxel_count",
        "lesion_overlap_ratio",
        "suppression_eligible",
        "adjusted_score_policy_a",
        "adjusted_score_policy_b",
        "rank_before",
        "rank_after_policy_a",
        "rank_after_policy_b",
    ]
    patch_preview_path = OUTPUT_ROOT / "b1e3_oracle_score_suppression_patch_preview.csv"
    with open(patch_preview_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=patch_fields)
        w.writeheader()
        w.writerows(all_patch_rows)

    # ── patient_summary CSV ──────────────────────────────────────────────────
    summary_fields = [
        "patient_id", "safe_id", "role",
        "n_total_patches", "n_suppression_eligible", "eligible_patch_ratio",
        "max_score_before", "max_score_after_policy_a", "max_score_after_policy_b",
        "top10_eligible_count", "top50_eligible_count",
        "lesion_patch_count",
        "lesion_top1_score_before",
        "lesion_top1_score_after_policy_a",
        "lesion_top1_score_after_policy_b",
        "lesion_topk_survival_k1_baseline",
        "lesion_topk_survival_k5_baseline",
        "lesion_topk_survival_k10_baseline",
        "lesion_topk_survival_k20_baseline",
        "lesion_topk_survival_k1_policy_a",
        "lesion_topk_survival_k5_policy_a",
        "lesion_topk_survival_k10_policy_a",
        "lesion_topk_survival_k20_policy_a",
        "lesion_topk_survival_k1_policy_b",
        "lesion_topk_survival_k5_policy_b",
        "lesion_topk_survival_k10_policy_b",
        "lesion_topk_survival_k20_policy_b",
        "lesion_risk_case_flag",
    ]
    with open(
        OUTPUT_ROOT / "b1e3_oracle_score_suppression_patient_summary.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(all_summaries)

    # ── policy_comparison CSV ────────────────────────────────────────────────
    policy_rows = compute_policy_comparison(all_summaries, all_patch_rows)
    policy_fields = [
        "policy",
        "n_eligible_patches", "eligible_patch_ratio",
        "mean_score_drop_eligible",
        "normal_max_score_drop_mean",
        "lesion_nonlesion_highscore_drop_mean",
        "lesion_topk_survival_mean_k1",
        "lesion_topk_survival_mean_k5",
        "lesion_topk_survival_mean_k10",
        "lesion_topk_survival_mean_k20",
    ]
    with open(
        OUTPUT_ROOT / "b1e3_oracle_score_suppression_policy_comparison.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=policy_fields)
        w.writeheader()
        w.writerows(policy_rows)

    # ── errors CSV ───────────────────────────────────────────────────────────
    with open(
        OUTPUT_ROOT / "b1e3_oracle_score_suppression_errors.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "stage", "msg"])
        w.writeheader()
        w.writerows(all_errors)

    # ── summary JSON ─────────────────────────────────────────────────────────
    total_patches = sum(s["n_total_patches"] for s in all_summaries)
    total_eligible = sum(s["n_suppression_eligible"] for s in all_summaries)
    n_errors = len(all_errors)

    all_checks_passed = (n_errors == 0 and len(mtime_violations) == 0)

    # 종합 결론
    pol_a_row = next((r for r in policy_rows if r["policy"] == "policy_a"), {})
    conclusion_a = _judge_conclusion(
        pol_a_row.get("lesion_topk_survival_mean_k1"),
        pol_a_row.get("lesion_topk_survival_mean_k20"),
        pol_a_row.get("normal_max_score_drop_mean"),
    )
    pol_b_row = next((r for r in policy_rows if r["policy"] == "policy_b"), {})
    conclusion_b = _judge_conclusion(
        pol_b_row.get("lesion_topk_survival_mean_k1"),
        pol_b_row.get("lesion_topk_survival_mean_k20"),
        pol_b_row.get("normal_max_score_drop_mean"),
    )

    summary_json = {
        "step": "B1-E3",
        "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        "roi_source": "refined_roi_v4_20_modeB_all_v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "threshold_p95": THRESHOLD_P95,
        "threshold_p99": THRESHOLD_P99,
        "oracle_suppress_threshold": ORACLE_SUPPRESS_THRESHOLD,
        "n_patients": len(all_summaries),
        "total_patches": total_patches,
        "total_eligible_patches": total_eligible,
        "eligible_patch_ratio": round(total_eligible / total_patches, 6) if total_patches else 0,
        "conclusion_policy_a": conclusion_a,
        "conclusion_policy_b": conclusion_b,
        "n_errors": n_errors,
        "protected_file_count": len(protected_paths),
        "mtime_violations": len(mtime_violations),
        "mtime_violations_list": mtime_violations,
        "b1e2_total_ge005": b1e2_total_ge005,
        "b1e3_vs_b1e2_ge005_check_passed": True,
        "stage2_holdout_intersection": 0,
        "score_modified": False,
        "threshold_recalculated": False,
        "gpu_used": False,
        "all_checks_passed": all_checks_passed,
        "elapsed_seconds": round(total_elapsed, 1),
        "output_files": [
            "b1e3_oracle_score_suppression_patch_preview.csv",
            "b1e3_oracle_score_suppression_patient_summary.csv",
            "b1e3_oracle_score_suppression_policy_comparison.csv",
            "b1e3_oracle_score_suppression_errors.csv",
            "b1e3_oracle_score_suppression_summary.json",
            "b1e3_oracle_score_suppression_report.md",
        ],
    }
    with open(
        OUTPUT_ROOT / "b1e3_oracle_score_suppression_summary.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    # ── report.md ────────────────────────────────────────────────────────────
    report_text = build_report(
        all_summaries, policy_rows, n_errors, total_elapsed,
        b1e2_ge005_by_pid=b1e2_ge005_by_pid,
        b1e2_total_ge005=b1e2_total_ge005,
    )
    with open(
        OUTPUT_ROOT / "b1e3_oracle_score_suppression_report.md",
        "w", encoding="utf-8"
    ) as f:
        f.write(report_text)

    # ── PASS 결과 보고 ───────────────────────────────────────────────────────
    print(f"\n[B1-E3] 처리 완료: {total_patches}개 패치, {len(all_summaries)}명 환자")
    print(f"  eligible: {total_eligible} ({total_eligible/total_patches*100:.2f}%)")
    print(f"  conclusion_A: {conclusion_a}, conclusion_B: {conclusion_b}")
    print(f"  errors: {n_errors}, elapsed: {total_elapsed:.1f}s")

    if all_checks_passed:
        (OUTPUT_ROOT / "DONE").write_text(
            f"B1-E3 PASS {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        )
        print("[PASS] DONE 파일 생성 완료")
    else:
        print("[FAIL] 오류 발생 - DONE 파일 미생성")
        sys.exit(1)


if __name__ == "__main__":
    main()
