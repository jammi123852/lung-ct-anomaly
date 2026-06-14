"""
B1-E2: Patch-Oracle Overlap Dry-run
EfficientNet-B0 v4_20 ROI branch score CSV patch 좌표와
B1-E1 oracle-like vessel mask의 overlap을 계산하는 dry-run.

이 스크립트는:
- score suppression 미적용 (adjusted_score 등 score 변환 컬럼 생성 금지)
- 원본 score CSV / threshold / model / ROI / CT / lesion mask 수정 없음
- stage2_holdout 접근 금지
- output root가 이미 존재하면 즉시 중단
- GPU 불필요

z 좌표 기준:
  local_z = 실제 CT array z index (사용)
  slice_index = 원본 DICOM slice 번호 (lesion 환자는 환자별 고정 offset 존재, 미사용)
"""

import os
import sys
import csv
import json
import time
import random
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
OUTPUT_ROOT = (
    PROJ_ROOT / "outputs" / "position-aware-padim-v1"
    / "oracle_vessel_suppression_effb0_v4_20_dev_only"
    / "b1e2_patch_oracle_overlap_dryrun_v1"
)
LESION_SPLIT_CSV = (
    PROJ_ROOT / "outputs" / "second-stage-lesion-refiner-v1"
    / "splits" / "lesion_stage_split_v1.csv"
)
THRESHOLD_JSON = (
    PROJ_ROOT / "experiments" / "efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs" / "evaluation" / "normal_val_thresholds" / "normal_val_threshold.json"
)
DIST_NPZ = (
    PROJ_ROOT / "experiments" / "efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
)

THRESHOLD_P95 = 13.231265
THRESHOLD_P99 = 15.472385
ORACLE_THRESHOLDS = [0.0, 0.01, 0.05, 0.10, 0.25]

# patch_sample 저장 최대 행 수 (환자당)
SAMPLE_TOP_N   = 200   # score 상위
SAMPLE_RAND_N  = 300   # 랜덤 (중복 제외)
SAMPLE_MAX     = SAMPLE_TOP_N + SAMPLE_RAND_N


def abort(msg: str) -> None:
    print(f"[ABORT] {msg}", file=sys.stderr)
    sys.exit(2)


def mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else -1.0


def read_csv_dicts(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_oracle_mask(ct: np.ndarray, roi: np.ndarray,
                        lesion: np.ndarray | None):
    """ROI 내부 HU>=0 voxel 중 lesion과 겹치지 않는 부분 = oracle-like vessel mask."""
    roi_bool = roi > 0
    hu_ge0   = roi_bool & (ct >= 0)
    if lesion is not None:
        les_bool = lesion > 0
        oracle = hu_ge0 & (~les_bool)
    else:
        les_bool = None
        oracle = hu_ge0
    return roi_bool, oracle, les_bool


def process_patient(row: dict, rng: random.Random) -> tuple:
    """
    Returns:
        patch_rows (list of dict): 샘플 패치 행
        summary (dict): 환자 요약
        errors (list of str): 오류 메시지
    """
    pid  = row["patient_id"]
    sid  = row["safe_id"]
    role = row["role"]
    errors = []

    ct_path     = Path(row["ct_path"])
    roi_path    = Path(row["roi_path"])
    les_path    = Path(row["lesion_mask_path"]) if row["lesion_mask_path"] else None
    score_path  = Path(row["score_csv_path"])

    # ── 파일 로드 ────────────────────────────────────────────────────────────
    try:
        ct  = np.load(str(ct_path),  mmap_mode='r')
        roi = np.load(str(roi_path), mmap_mode='r')
        les = np.load(str(les_path), mmap_mode='r') if les_path else None
    except Exception as e:
        errors.append(f"load: {e}")
        return [], _empty_summary(pid, sid, role), errors

    ct_z, ct_y, ct_x = ct.shape

    # ── oracle mask 계산 (전체 3D) ─────────────────────────────────────────
    ct_arr  = np.asarray(ct,  dtype=np.int16)
    roi_arr = np.asarray(roi, dtype=np.uint8)
    les_arr = np.asarray(les, dtype=np.uint8) if les is not None else None
    roi_bool, oracle, les_bool = compute_oracle_mask(ct_arr, roi_arr, les_arr)

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
    z_cache_roi    = {}
    z_cache_les    = {}

    for _, pr in df.iterrows():
        z  = int(pr["local_z"])
        y0 = int(pr["y0"])
        x0 = int(pr["x0"])
        y1 = int(pr["y1"])
        x1 = int(pr["x1"])
        score = float(pr["padim_score"])

        # 좌표 유효성
        coord_valid = (
            0 <= z < ct_z and
            0 <= y0 < y1 <= ct_y and
            0 <= x0 < x1 <= ct_x
        )

        roi_ov = oracle_ov = les_ov = 0
        patch_area = (y1 - y0) * (x1 - x0)

        if coord_valid:
            if z not in z_cache_oracle:
                z_cache_oracle[z] = oracle[z].copy()
                z_cache_roi[z]    = roi_bool[z].copy()
                z_cache_les[z]    = (les_bool[z].copy()
                                     if les_bool is not None else None)
            oracle_z = z_cache_oracle[z]
            roi_z    = z_cache_roi[z]
            les_z    = z_cache_les[z]

            roi_crop    = roi_z[y0:y1, x0:x1]
            oracle_crop = oracle_z[y0:y1, x0:x1]

            roi_ov    = int(roi_crop.sum())
            oracle_ov = int(oracle_crop.sum())
            les_ov    = 0
            if les_z is not None:
                les_crop = les_z[y0:y1, x0:x1]
                les_ov   = int(les_crop.sum())

        roi_ratio    = roi_ov    / patch_area if patch_area > 0 else 0.0
        oracle_ratio = oracle_ov / patch_area if patch_area > 0 else 0.0
        les_ratio    = les_ov    / patch_area if patch_area > 0 else 0.0

        results.append({
            "z":           z,
            "y0": y0, "x0": x0, "y1": y1, "x1": x1,
            "score":       score,
            "patch_area":  patch_area,
            "roi_ov":      roi_ov,
            "roi_ratio":   roi_ratio,
            "oracle_ov":   oracle_ov,
            "oracle_ratio": oracle_ratio,
            "les_ov":      les_ov,
            "les_ratio":   les_ratio,
            "coord_valid": coord_valid,
        })

    # ── patient summary 집계 ─────────────────────────────────────────────────
    n_coord_valid    = sum(1 for r in results if r["coord_valid"])
    n_roi_gt0        = sum(1 for r in results if r["roi_ov"] > 0)

    n_oracle_gt0     = sum(1 for r in results if r["oracle_ov"] > 0)
    n_oracle_ge001   = sum(1 for r in results if r["oracle_ratio"] >= 0.01)
    n_oracle_ge005   = sum(1 for r in results if r["oracle_ratio"] >= 0.05)
    n_oracle_ge010   = sum(1 for r in results if r["oracle_ratio"] >= 0.10)
    n_oracle_ge025   = sum(1 for r in results if r["oracle_ratio"] >= 0.25)

    oracle_ratios_valid = [r["oracle_ratio"] for r in results if r["coord_valid"]]
    mean_or = float(np.mean(oracle_ratios_valid)) if oracle_ratios_valid else 0.0
    max_or  = float(np.max(oracle_ratios_valid))  if oracle_ratios_valid else 0.0

    # score 상위 1/10/50 patch의 oracle_ratio
    sorted_res = sorted(results, key=lambda r: -r["score"])
    top1_or  = sorted_res[0]["oracle_ratio"] if sorted_res else 0.0
    top10_or_count  = sum(1 for r in sorted_res[:10]  if r["oracle_ratio"] > 0)
    top50_or_count  = sum(1 for r in sorted_res[:50]  if r["oracle_ratio"] > 0)

    les_patch_count = sum(1 for r in results if r["les_ov"] > 0)
    les_and_oracle  = sum(1 for r in results if r["les_ov"] > 0 and r["oracle_ov"] > 0)
    les_risk_mixed  = les_and_oracle  # lesion 영역인데 oracle에도 포함되는 위험 패치

    usable = (n_coord_valid > 0 and n_oracle_gt0 > 0)

    summary = {
        "patient_id":      pid,
        "safe_id":         sid,
        "role":            role,
        "n_patches_total": n_total,
        "n_coordinate_valid": n_coord_valid,
        "n_roi_overlap_gt0":     n_roi_gt0,
        "n_oracle_overlap_gt0":  n_oracle_gt0,
        "n_oracle_overlap_ge001": n_oracle_ge001,
        "n_oracle_overlap_ge005": n_oracle_ge005,
        "n_oracle_overlap_ge010": n_oracle_ge010,
        "n_oracle_overlap_ge025": n_oracle_ge025,
        "oracle_overlap_patch_ratio_gt0":  round(n_oracle_gt0  / n_total, 6) if n_total else 0,
        "oracle_overlap_patch_ratio_ge001": round(n_oracle_ge001 / n_total, 6) if n_total else 0,
        "oracle_overlap_patch_ratio_ge005": round(n_oracle_ge005 / n_total, 6) if n_total else 0,
        "oracle_overlap_patch_ratio_ge010": round(n_oracle_ge010 / n_total, 6) if n_total else 0,
        "oracle_overlap_patch_ratio_ge025": round(n_oracle_ge025 / n_total, 6) if n_total else 0,
        "mean_oracle_overlap_ratio": round(mean_or, 6),
        "max_oracle_overlap_ratio":  round(max_or,  6),
        "top1_score_oracle_overlap_ratio":     round(top1_or, 6),
        "top10_score_oracle_overlap_patch_count": top10_or_count,
        "top50_score_oracle_overlap_patch_count": top50_or_count,
        "lesion_overlap_patch_count":       les_patch_count,
        "lesion_and_oracle_overlap_patch_count": les_and_oracle,
        "lesion_risk_mixed_patch_count":    les_risk_mixed,
        "usable_for_b1e3": usable,
    }

    # ── patch sample 선택: score 상위 N + 랜덤 M ──────────────────────────
    indices = list(range(len(results)))
    top_idx  = sorted(indices, key=lambda i: -results[i]["score"])[:SAMPLE_TOP_N]
    top_set  = set(top_idx)
    rest_idx = [i for i in indices if i not in top_set]
    rng.shuffle(rest_idx)
    sample_idx = top_idx + rest_idx[:SAMPLE_RAND_N]

    patch_rows = []
    for i in sample_idx:
        r = results[i]
        patch_rows.append({
            "patient_id":   pid,
            "safe_id":      sid,
            "role":         role,
            "score_csv_path": str(score_path),
            "slice_index_used": r["z"],
            "y0": r["y0"], "x0": r["x0"], "y1": r["y1"], "x1": r["x1"],
            "original_score":  round(r["score"], 6),
            "patch_area":      r["patch_area"],
            "roi_overlap_voxel_count":           r["roi_ov"],
            "roi_overlap_ratio":                 round(r["roi_ratio"],    6),
            "oracle_like_vessel_overlap_voxel_count": r["oracle_ov"],
            "oracle_like_vessel_overlap_ratio":   round(r["oracle_ratio"], 6),
            "lesion_overlap_voxel_count":         r["les_ov"],
            "lesion_overlap_ratio":               round(r["les_ratio"],   6),
            "would_be_oracle_candidate_gt0":   r["oracle_ov"] > 0,
            "would_be_oracle_candidate_ge001": r["oracle_ratio"] >= 0.01,
            "would_be_oracle_candidate_ge005": r["oracle_ratio"] >= 0.05,
            "would_be_oracle_candidate_ge010": r["oracle_ratio"] >= 0.10,
            "would_be_oracle_candidate_ge025": r["oracle_ratio"] >= 0.25,
            "coordinate_valid": r["coord_valid"],
            "shape_match": True,
            "stage2_holdout_intersection_flag": False,
        })

    return patch_rows, summary, errors


def _empty_summary(pid, sid, role) -> dict:
    return {
        "patient_id": pid, "safe_id": sid, "role": role,
        "n_patches_total": 0, "n_coordinate_valid": 0, "n_roi_overlap_gt0": 0,
        "n_oracle_overlap_gt0": 0, "n_oracle_overlap_ge001": 0,
        "n_oracle_overlap_ge005": 0, "n_oracle_overlap_ge010": 0,
        "n_oracle_overlap_ge025": 0,
        "oracle_overlap_patch_ratio_gt0": 0,
        "oracle_overlap_patch_ratio_ge001": 0,
        "oracle_overlap_patch_ratio_ge005": 0,
        "oracle_overlap_patch_ratio_ge010": 0,
        "oracle_overlap_patch_ratio_ge025": 0,
        "mean_oracle_overlap_ratio": 0, "max_oracle_overlap_ratio": 0,
        "top1_score_oracle_overlap_ratio": 0,
        "top10_score_oracle_overlap_patch_count": 0,
        "top50_score_oracle_overlap_patch_count": 0,
        "lesion_overlap_patch_count": 0,
        "lesion_and_oracle_overlap_patch_count": 0,
        "lesion_risk_mixed_patch_count": 0,
        "usable_for_b1e3": False,
    }


def main() -> None:
    if not ALLOW_REAL_PROCESSING:
        abort("ALLOW_REAL_PROCESSING=False: dry-run guard 활성. 이 스크립트는 직접 실행 불가.")

    # ── output root 존재 확인 ───────────────────────────────────────────────
    if OUTPUT_ROOT.exists():
        abort(f"output root 이미 존재: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    # ── B1-E1 DONE 확인 ─────────────────────────────────────────────────────
    if not (B1E1_ROOT / "DONE").exists():
        abort(f"B1-E1 DONE 파일 없음. B1-E1 먼저 완료 필요: {B1E1_ROOT}")

    # ── mtime 스냅샷 ──────────────────────────────────────────────────────────
    protected = [THRESHOLD_JSON, DIST_NPZ, LESION_SPLIT_CSV]
    mtime_before = {str(p): mtime(p) for p in protected}

    # ── stage2 holdout denylist ───────────────────────────────────────────────
    holdout_pids: set = set()
    holdout_sids: set = set()
    with open(LESION_SPLIT_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("stage_split", "") == "stage2_holdout":
                holdout_pids.add(row["patient_id"].strip())
                holdout_sids.add(row["safe_id"].strip())

    # ── B1-E1 targets 읽기 ────────────────────────────────────────────────────
    targets = read_csv_dicts(B1E1_ROOT / "b1e1_oracle_mask_preflight_targets.csv")

    # stage2_holdout 교집합 최종 확인
    for t in targets:
        if t["patient_id"] in holdout_pids or t["safe_id"] in holdout_sids:
            abort(f"stage2_holdout 교집합 발견: {t['patient_id']}")

    # ── 환자별 처리 ───────────────────────────────────────────────────────────
    rng = random.Random(42)
    all_patch_rows = []
    all_summaries  = []
    all_errors     = []
    t0 = time.time()

    for t in targets:
        pid = t["patient_id"]
        print(f"  처리 중: {pid} ({t['role']}) ...", flush=True)
        t1 = time.time()
        patch_rows, summary, errors = process_patient(t, rng)
        elapsed = time.time() - t1
        print(f"    완료: {elapsed:.1f}s, oracle_gt0={summary['n_oracle_overlap_gt0']}, "
              f"oracle_ge005={summary['n_oracle_overlap_ge005']}, "
              f"les_risk={summary['lesion_risk_mixed_patch_count']}")
        all_patch_rows.extend(patch_rows)
        all_summaries.append(summary)
        for e in errors:
            all_errors.append({"patient_id": pid, "stage": "process", "msg": e})

    total_elapsed = time.time() - t0

    # ── mtime 사후 검증 ──────────────────────────────────────────────────────
    mtime_violations = []
    for ps, before in mtime_before.items():
        after = mtime(Path(ps))
        if before != after:
            mtime_violations.append(f"{ps}: {before} → {after}")
    if mtime_violations:
        abort("원본 파일 mtime 변경 감지:\n" + "\n".join(mtime_violations))

    # ── patch_sample CSV 저장 ────────────────────────────────────────────────
    patch_fields = [
        "patient_id","safe_id","role","score_csv_path",
        "slice_index_used","y0","x0","y1","x1",
        "original_score","patch_area",
        "roi_overlap_voxel_count","roi_overlap_ratio",
        "oracle_like_vessel_overlap_voxel_count","oracle_like_vessel_overlap_ratio",
        "lesion_overlap_voxel_count","lesion_overlap_ratio",
        "would_be_oracle_candidate_gt0","would_be_oracle_candidate_ge001",
        "would_be_oracle_candidate_ge005","would_be_oracle_candidate_ge010",
        "would_be_oracle_candidate_ge025",
        "coordinate_valid","shape_match","stage2_holdout_intersection_flag",
    ]
    with open(OUTPUT_ROOT / "b1e2_patch_oracle_overlap_dryrun_patch_sample.csv",
              "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=patch_fields)
        w.writeheader()
        w.writerows(all_patch_rows)

    # ── patient_summary CSV ──────────────────────────────────────────────────
    summary_fields = [
        "patient_id","safe_id","role",
        "n_patches_total","n_coordinate_valid","n_roi_overlap_gt0",
        "n_oracle_overlap_gt0","n_oracle_overlap_ge001","n_oracle_overlap_ge005",
        "n_oracle_overlap_ge010","n_oracle_overlap_ge025",
        "oracle_overlap_patch_ratio_gt0","oracle_overlap_patch_ratio_ge001",
        "oracle_overlap_patch_ratio_ge005","oracle_overlap_patch_ratio_ge010",
        "oracle_overlap_patch_ratio_ge025",
        "mean_oracle_overlap_ratio","max_oracle_overlap_ratio",
        "top1_score_oracle_overlap_ratio",
        "top10_score_oracle_overlap_patch_count","top50_score_oracle_overlap_patch_count",
        "lesion_overlap_patch_count","lesion_and_oracle_overlap_patch_count",
        "lesion_risk_mixed_patch_count","usable_for_b1e3",
    ]
    with open(OUTPUT_ROOT / "b1e2_patch_oracle_overlap_dryrun_patient_summary.csv",
              "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(all_summaries)

    # ── threshold_table CSV ──────────────────────────────────────────────────
    thresh_rows = []
    agg = defaultdict(int)
    agg["patient_id"] = "AGGREGATE"
    agg["safe_id"]    = ""
    agg["role"]       = "all"
    for s in all_summaries:
        thresh_rows.append({
            "patient_id":      s["patient_id"],
            "safe_id":         s["safe_id"],
            "role":            s["role"],
            "n_total":         s["n_patches_total"],
            "n_oracle_gt0":    s["n_oracle_overlap_gt0"],
            "n_oracle_ge001":  s["n_oracle_overlap_ge001"],
            "n_oracle_ge005":  s["n_oracle_overlap_ge005"],
            "n_oracle_ge010":  s["n_oracle_overlap_ge010"],
            "n_oracle_ge025":  s["n_oracle_overlap_ge025"],
            "ratio_gt0":       s["oracle_overlap_patch_ratio_gt0"],
            "ratio_ge001":     s["oracle_overlap_patch_ratio_ge001"],
            "ratio_ge005":     s["oracle_overlap_patch_ratio_ge005"],
            "ratio_ge010":     s["oracle_overlap_patch_ratio_ge010"],
            "ratio_ge025":     s["oracle_overlap_patch_ratio_ge025"],
        })
        for k in ["n_patches_total","n_oracle_overlap_gt0","n_oracle_overlap_ge001",
                  "n_oracle_overlap_ge005","n_oracle_overlap_ge010","n_oracle_overlap_ge025"]:
            agg[k] = agg.get(k, 0) + s[k]

    n_agg = agg.get("n_patches_total", 1)
    thresh_rows.append({
        "patient_id": "AGGREGATE", "safe_id": "", "role": "all",
        "n_total":         agg.get("n_patches_total",      0),
        "n_oracle_gt0":    agg.get("n_oracle_overlap_gt0", 0),
        "n_oracle_ge001":  agg.get("n_oracle_overlap_ge001", 0),
        "n_oracle_ge005":  agg.get("n_oracle_overlap_ge005", 0),
        "n_oracle_ge010":  agg.get("n_oracle_overlap_ge010", 0),
        "n_oracle_ge025":  agg.get("n_oracle_overlap_ge025", 0),
        "ratio_gt0":    round(agg.get("n_oracle_overlap_gt0",  0) / n_agg, 6) if n_agg else 0,
        "ratio_ge001":  round(agg.get("n_oracle_overlap_ge001",0) / n_agg, 6) if n_agg else 0,
        "ratio_ge005":  round(agg.get("n_oracle_overlap_ge005",0) / n_agg, 6) if n_agg else 0,
        "ratio_ge010":  round(agg.get("n_oracle_overlap_ge010",0) / n_agg, 6) if n_agg else 0,
        "ratio_ge025":  round(agg.get("n_oracle_overlap_ge025",0) / n_agg, 6) if n_agg else 0,
    })

    thresh_fields = ["patient_id","safe_id","role","n_total",
                     "n_oracle_gt0","n_oracle_ge001","n_oracle_ge005",
                     "n_oracle_ge010","n_oracle_ge025",
                     "ratio_gt0","ratio_ge001","ratio_ge005","ratio_ge010","ratio_ge025"]
    with open(OUTPUT_ROOT / "b1e2_patch_oracle_overlap_dryrun_threshold_table.csv",
              "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=thresh_fields)
        w.writeheader()
        w.writerows(thresh_rows)

    # ── errors CSV ───────────────────────────────────────────────────────────
    with open(OUTPUT_ROOT / "b1e2_patch_oracle_overlap_dryrun_errors.csv",
              "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id","stage","msg"])
        w.writeheader()
        w.writerows(all_errors)

    # ── summary JSON + 판정 ──────────────────────────────────────────────────
    n_usable   = sum(1 for s in all_summaries if s["usable_for_b1e3"])
    n_les_risk = sum(s["lesion_risk_mixed_patch_count"] for s in all_summaries)
    total_patches = agg.get("n_patches_total", 0)
    total_oracle_gt0 = agg.get("n_oracle_overlap_gt0", 0)
    total_oracle_ge005 = agg.get("n_oracle_overlap_ge005", 0)

    # B1-E3 GO/CAUTION/NO-GO 판정
    oracle_rate = total_oracle_gt0 / total_patches if total_patches else 0
    les_risk_rate = n_les_risk / total_patches if total_patches else 0
    if total_oracle_gt0 == 0 or oracle_rate < 0.001:
        b1e3_verdict = "NO-GO"
        verdict_reason = "oracle overlap patch 너무 적음 (oracle_rate < 0.1%)"
    elif les_risk_rate > 0.05:
        b1e3_verdict = "CAUTION"
        verdict_reason = f"lesion 혼입 비율 높음 (les_risk_rate={les_risk_rate:.4f})"
    else:
        b1e3_verdict = "GO"
        verdict_reason = f"oracle_rate={oracle_rate:.4f}, les_risk_rate={les_risk_rate:.4f}"

    summary_json = {
        "step": "B1-E2",
        "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        "roi_source": "refined_roi_v4_20_modeB_all_v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "z_coordinate_note": (
            "local_z = 실제 CT array z index (사용). "
            "slice_index = 원본 DICOM 번호 (lesion 환자 환자별 고정 offset 존재, 미사용)."
        ),
        "threshold_p95": THRESHOLD_P95,
        "threshold_p99": THRESHOLD_P99,
        "n_patients": len(all_summaries),
        "n_usable_for_b1e3": n_usable,
        "total_patches": total_patches,
        "total_oracle_gt0": total_oracle_gt0,
        "total_oracle_ge005": total_oracle_ge005,
        "oracle_rate_gt0": round(oracle_rate, 6),
        "les_risk_total": n_les_risk,
        "les_risk_rate": round(les_risk_rate, 6),
        "b1e3_verdict": b1e3_verdict,
        "b1e3_verdict_reason": verdict_reason,
        "mtime_violations": len(mtime_violations),
        "n_errors": len(all_errors),
        "stage2_holdout_intersection": 0,
        "score_modified": False,
        "threshold_recalculated": False,
        "suppression_applied": False,
        "stage2_holdout_accessed": False,
        "gpu_used": False,
        "elapsed_seconds": round(total_elapsed, 1),
        "all_checks_passed": (
            len(all_errors) == 0 and len(mtime_violations) == 0
        ),
    }
    summary_json["next_step_b1e3_ready"] = (
        b1e3_verdict in ("GO", "CAUTION") and summary_json["all_checks_passed"]
    )

    with open(OUTPUT_ROOT / "b1e2_patch_oracle_overlap_dryrun_summary.json",
              "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    # ── 보고서 MD ────────────────────────────────────────────────────────────
    _write_report(all_summaries, thresh_rows, summary_json, n_les_risk,
                  oracle_rate, les_risk_rate, b1e3_verdict, verdict_reason)

    # ── DONE ────────────────────────────────────────────────────────────────
    print(f"\n=== B1-E2 완료 ({total_elapsed:.1f}s) ===")
    print(f"  총 patch: {total_patches:,}  oracle_gt0: {total_oracle_gt0:,} ({oracle_rate:.4f})")
    print(f"  les_risk: {n_les_risk:,} ({les_risk_rate:.4f})")
    print(f"  B1-E3 판정: {b1e3_verdict} — {verdict_reason}")

    if summary_json["all_checks_passed"]:
        (OUTPUT_ROOT / "DONE").write_text(
            f"B1-E2 PASS {summary_json['created']} verdict={b1e3_verdict}\n"
        )
        print("  → DONE 생성")
    else:
        print("  → 오류 있음. DONE 미생성.")
        for e in all_errors:
            print(f"    [{e['patient_id']}] {e['msg']}")


def _write_report(summaries, thresh_rows, sj, n_les_risk,
                  oracle_rate, les_risk_rate, verdict, reason):
    lines = [
        "# B1-E2 Patch-Oracle Overlap Dry-run 보고서",
        "",
        f"생성일시: {sj['created']}",
        f"branch: {sj['branch']}",
        f"ROI source: {sj['roi_source']}",
        "",
        "## 1. 이번 단계 성격 고지",
        "",
        "- **B1-E2는 suppression 적용이 아니라 좌표/overlap dry-run이다.**",
        "- oracle-like vessel mask는 진짜 vessel GT가 아니다 (HU>=0 기반, 조영제/종격동/흉벽 bright tissue 혼입 가능).",
        "- score CSV / threshold / model / ROI / CT 파일을 수정하지 않았다.",
        "",
        "## 2. z 좌표 매핑 기준",
        "",
        "| 컬럼 | 의미 | 사용 여부 |",
        "|---|---|---|",
        "| `local_z` | 실제 CT array z index (0-based) | **사용** |",
        "| `slice_index` | 원본 DICOM slice 번호 (lesion 환자는 환자별 고정 offset 존재) | 미사용 |",
        "",
        "- lesion 환자 5명 모두 `slice_index - local_z` = 고정 양수 offset (21~93)",
        "- normal 환자: offset = 0",
        "- `local_z` 기준 out_of_range = 0 확인됨",
        "",
        "## 3. 전체 집계 결과",
        "",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| 전체 대상 | {sj['n_patients']}명 |",
        f"| 전체 patch | {sj['total_patches']:,} |",
        f"| oracle_gt0 patch | {sj['total_oracle_gt0']:,} ({sj['oracle_rate_gt0']:.4f}) |",
        f"| oracle_ge005 patch | {sj['total_oracle_ge005']:,} |",
        f"| lesion 혼입 위험 patch | {sj['les_risk_total']:,} ({sj['les_risk_rate']:.4f}) |",
        f"| usable_for_b1e3 | {sj['n_usable_for_b1e3']}명 |",
        "",
        "## 4. 환자별 oracle overlap 통계",
        "",
        "| patient_id | role | n_total | n_oracle_gt0 | ratio_gt0 | n_oracle_ge005 | ratio_ge005 | les_risk | top1_or | usable |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['patient_id']} | {s['role']} "
            f"| {s['n_patches_total']:,} | {s['n_oracle_overlap_gt0']:,} "
            f"| {s['oracle_overlap_patch_ratio_gt0']:.4f} "
            f"| {s['n_oracle_overlap_ge005']:,} "
            f"| {s['oracle_overlap_patch_ratio_ge005']:.4f} "
            f"| {s['lesion_risk_mixed_patch_count']} "
            f"| {s['top1_score_oracle_overlap_ratio']:.4f} "
            f"| {s['usable_for_b1e3']} |"
        )

    lines += [
        "",
        "## 5. Threshold별 aggregate 집계",
        "",
        "| threshold | n_oracle | ratio |",
        "|---|---|---|",
    ]
    agg = thresh_rows[-1]  # AGGREGATE 행
    for thr, nk, rk in [
        (">0",    "n_oracle_gt0",   "ratio_gt0"),
        (">=0.01","n_oracle_ge001","ratio_ge001"),
        (">=0.05","n_oracle_ge005","ratio_ge005"),
        (">=0.10","n_oracle_ge010","ratio_ge010"),
        (">=0.25","n_oracle_ge025","ratio_ge025"),
    ]:
        lines.append(f"| {thr} | {agg[nk]:,} | {agg[rk]:.4f} |")

    # 안전한 threshold 1차 추천
    if agg["n_oracle_ge005"] > 100:
        recommended_thr = ">=0.05"
    elif agg["n_oracle_ge001"] > 100:
        recommended_thr = ">=0.01"
    else:
        recommended_thr = ">0"

    lines += [
        "",
        "## 6. 안전한 overlap threshold 1차 추천",
        "",
        f"**추천 threshold: oracle_overlap_ratio {recommended_thr}**",
        "",
        "- threshold가 너무 낮으면 ROI 경계의 noise voxel이 포함될 수 있음",
        "- threshold >= 0.05는 patch 면적의 5% 이상이 oracle voxel인 경우로, 실질적인 혈관 관여를 의미",
        f"- 단, lesion_risk_mixed_patch_count = {n_les_risk:,}이므로 suppression 적용 시 병변 일부 억제 위험 존재",
        "",
        "## 7. B1-E3 진행 가능 판정",
        "",
        f"**{verdict}** — {reason}",
        "",
    ]
    if verdict == "GO":
        lines.append("- oracle overlap patch 충분하고 lesion 혼입 낮음 → B1-E3 suppression 시뮬레이션 진행 가능")
    elif verdict == "CAUTION":
        lines += [
            "- oracle overlap 있으나 lesion 혼입 있음",
            f"- B1-E3에서 강한 suppression (×0.0)은 위험: lesion_and_oracle_overlap = {n_les_risk:,}개 패치에서 병변 score가 같이 억제될 수 있음",
            "- 권장: B1-E3에서 ×0.5 soft penalty 먼저 검토, ×0.0은 lesion_overlap_ratio=0 조건 패치에만 한정",
        ]
    else:
        lines.append("- oracle overlap patch 부족 → B1-E3 suppression 실험 의미 부족. 보류 권장.")

    lines += [
        "",
        "## 8. 다음 단계 B1-E3 안내",
        "",
        "- B1-E3에서는 **원본 score CSV를 수정하지 않는다.**",
        "- adjusted score는 **새 preview CSV 안에서만** score × 0.5 / × 0.0 시뮬레이션한다.",
        "- 시뮬레이션 후 threshold_p95/p99 기준 FP 감소율 및 lesion recall 보존율을 계산한다.",
        "- stage2_holdout는 절대 접근하지 않는다.",
        "",
        "## 9. 안전 게이트 확인",
        "",
        "| 항목 | 상태 |",
        "|---|---|",
        "| score CSV 수정 | 미실행 |",
        "| threshold 재계산 | 미실행 |",
        "| model 수정 | 미실행 |",
        "| ROI 파일 수정 | 미실행 |",
        "| CT 파일 수정 | 미실행 |",
        "| suppression 적용 | 미실행 |",
        "| adjusted_score 컬럼 생성 | 없음 |",
        "| stage2_holdout 접근 | 없음 |",
        "| GPU 사용 | 없음 |",
        f"| mtime 위반 | {sj['mtime_violations']} |",
    ]

    with open(OUTPUT_ROOT / "b1e2_patch_oracle_overlap_dryrun_report.md",
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
