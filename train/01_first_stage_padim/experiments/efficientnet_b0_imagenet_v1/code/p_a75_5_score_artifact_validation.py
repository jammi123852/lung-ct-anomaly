"""
P-A75.5: EfficientNet-B0 stage1_dev score artifact validation (read-only).

- P-A75 score CSV 154개를 pandas/csv.reader로 read-only 검증.
- np.loadtxt 사용 금지.
- metrics 계산 금지, scoring/forward/training 금지.
- stage2_holdout 접근 금지.

실행:
  source ~/ai_env/bin/activate && python experiments/efficientnet_b0_imagenet_v1/code/p_a75_5_score_artifact_validation.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT  = Path(__file__).resolve().parents[1]

EXPECTED_STAGE  = "stage1_dev"
EXPECTED_N      = 154
EXPECTED_GROUPS = {"NSCLC": 125, "MSD_Lung": 29}
JOIN_KEY        = "patient_id"

# P-A75 보고서 기준값
P_A75_REPORTED_PATCHES = 2_760_497
P_A75_P95_EXCEED       = 488_216
P_A75_P99_EXCEED       = 174_823
P_A75_REPORTED_MEAN    = 11.3699
P_A75_REPORTED_STD     = 2.3417

# ---- 입력 (read-only) ----
SCORE_DIR      = EXP_ROOT / "outputs" / "scores" / "lesion_stage1_dev_by_patient"
P_A75_MD       = EXP_ROOT / "outputs" / "reports" / "lesion_stage1_dev" / "p_a75_lesion_stage1_dev_scoring.md"
P_A75_JSON     = EXP_ROOT / "outputs" / "reports" / "lesion_stage1_dev" / "p_a75_lesion_stage1_dev_scoring.json"
THRESH_JSON    = EXP_ROOT / "outputs" / "evaluation" / "normal_val_thresholds" / "normal_val_threshold.json"
LESION_SPLIT   = PROJ_ROOT / "outputs" / "second-stage-lesion-refiner-v1" / "splits" / "lesion_stage_split_v1_balanced.csv"

# reference (ResNet18 rand224, row count 비교용, read-only)
REF_SCORE_DIR  = PROJ_ROOT / "experiments" / "resnet18_imagenet_rand224_v1" / "outputs" / "scores" / "lesion_stage1_dev_by_patient"

# ---- 출력 경로 ----
OUT_DIR         = EXP_ROOT / "outputs" / "reports" / "lesion_stage1_dev" / "p_a75_5_score_artifact_validation"
SUMMARY_CSV     = OUT_DIR / "score_artifact_validation_summary.csv"
ROW_COUNT_CSV   = OUT_DIR / "score_artifact_patient_row_counts.csv"
COL_CHECK_CSV   = OUT_DIR / "score_artifact_column_check.csv"
REPORT_MD       = OUT_DIR / "p_a75_5_score_artifact_validation.md"
REPORT_JSON     = OUT_DIR / "p_a75_5_score_artifact_validation.json"

# 필수 컬럼
REQUIRED_COLS = {"patient_id", "slice_index", "y0", "x0", "y1", "x1",
                 "padim_score", "position_bin", "has_lesion_patch"}
OPTIONAL_COLS = {"safe_id", "label", "local_z", "z_level", "lesion_pixels",
                 "patch_label", "group"}


def abort(msg: str) -> None:
    print(f"[P-A75.5][ABORT] {msg}")
    sys.exit(1)


def run_guards(stage1_dev_ids: set, holdout_ids: set) -> dict:
    # G1: P-A75 보고서 통과 확인
    if not P_A75_MD.exists():
        abort(f"P-A75 보고서 없음: {P_A75_MD}")
    with open(P_A75_MD, encoding="utf-8") as f:
        md_text = f.read()
    if "판정: 통과" not in md_text:
        abort(f"P-A75 보고서가 통과 상태가 아님: {P_A75_MD}")
    print("[G1] P-A75 보고서 통과 확인 ✅")

    # P-A75 JSON read-only 로드
    if not P_A75_JSON.exists():
        abort(f"P-A75 JSON 없음: {P_A75_JSON}")
    with open(P_A75_JSON, encoding="utf-8") as f:
        p75 = json.load(f)

    # G2: score CSV 수 154개 확인
    csv_files = list(SCORE_DIR.glob("*.csv"))
    if len(csv_files) != EXPECTED_N:
        abort(f"score CSV 수가 {len(csv_files)} (기대 {EXPECTED_N})")
    print(f"[G2] score CSV {len(csv_files)}개 확인 ✅")

    # G5: stage2_holdout contamination 0 확인
    csv_pids = {f.stem for f in csv_files}
    holdout_in_score = csv_pids & holdout_ids
    if holdout_in_score:
        abort(f"stage2_holdout 환자가 score dir에 존재: {holdout_in_score}")
    print(f"[G5] stage2_holdout contamination 0 확인 ✅")

    # G8: np.loadtxt 사용 금지 — 이 스크립트에서는 pandas만 사용 (정책 확인)
    print("[G8] np.loadtxt 미사용 (pandas 기반) ✅")

    # G11: 출력 경로 기존 결과 없음 확인
    if REPORT_JSON.exists() or REPORT_MD.exists():
        abort(f"기존 P-A75.5 결과 존재 → 덮어쓰기 금지: {OUT_DIR}")
    print(f"[G11] 출력 경로 기존 결과 없음 확인 ✅")

    return p75


def main() -> None:
    start_time = time.time()

    # split 로드
    split_rows = list(csv.DictReader(open(LESION_SPLIT, encoding="utf-8-sig")))
    dev_rows     = [r for r in split_rows if r["stage_split"] == EXPECTED_STAGE]
    holdout_rows = [r for r in split_rows if r["stage_split"] == "stage2_holdout"]
    stage1_dev_ids = {r[JOIN_KEY] for r in dev_rows}
    holdout_ids    = {r[JOIN_KEY] for r in holdout_rows}
    pid2group      = {r[JOIN_KEY]: r["group"] for r in dev_rows}

    # 가드 실행
    p75 = run_guards(stage1_dev_ids, holdout_ids)

    # threshold read-only 로드
    thresh_mtime = os.path.getmtime(THRESH_JSON)
    with open(THRESH_JSON, encoding="utf-8") as f:
        th = json.load(f)
    p95 = float(th["threshold_p95"])
    p99 = float(th["threshold_p99"])
    print(f"[P-A75.5] threshold p95={p95:.6f}, p99={p99:.6f} (read-only)")

    print(f"\n[P-A75.5] 모든 가드 통과. score artifact validation 시작.")

    # patient_id set 일치 확인
    csv_files = sorted(SCORE_DIR.glob("*.csv"))
    csv_pids = {f.stem for f in csv_files}
    pid_match    = (csv_pids == stage1_dev_ids)
    missing_csv  = stage1_dev_ids - csv_pids
    extra_csv    = csv_pids - stage1_dev_ids
    print(f"[P-A75.5] patient_id set 일치: {pid_match} (누락={len(missing_csv)}, 초과={len(extra_csv)})")

    # NSCLC / MSD_Lung 분류
    group_counts = dict(Counter(pid2group.get(pid, "UNKNOWN") for pid in csv_pids))
    print(f"[P-A75.5] group_counts: {group_counts}")

    # reference row count (ResNet18 rand224, read-only)
    ref_available = REF_SCORE_DIR.exists() and any(REF_SCORE_DIR.glob("*.csv"))
    ref_row_counts: dict = {}
    if ref_available:
        for fp in REF_SCORE_DIR.glob("*.csv"):
            pid = fp.stem
            if pid in stage1_dev_ids:
                try:
                    df_ref = pd.read_csv(fp, encoding="utf-8-sig", usecols=[0])
                    ref_row_counts[pid] = len(df_ref)
                except Exception:
                    pass
        print(f"[P-A75.5] reference (ResNet18 rand224) row count 로드: {len(ref_row_counts)}명")
    else:
        print(f"[P-A75.5] reference score dir 없음 → 내부 정합만 확인")

    # 컬럼 체크 (첫 번째 CSV에서 확인)
    sample_path = sorted(csv_files)[0]
    sample_df = pd.read_csv(sample_path, encoding="utf-8-sig", nrows=0)
    actual_cols = set(sample_df.columns)
    required_present = REQUIRED_COLS & actual_cols
    required_missing = REQUIRED_COLS - actual_cols
    optional_present = OPTIONAL_COLS & actual_cols
    has_zlevel = "local_z" in actual_cols or "z_level" in actual_cols
    has_lesion_label = "has_lesion_patch" in actual_cols or "patch_label" in actual_cols
    col_check_ok = len(required_missing) == 0
    print(f"[P-A75.5] 컬럼 체크: required_present={len(required_present)}, missing={required_missing}")

    # ------------------------------------------------------------------
    # 환자별 pandas 기반 robust read
    # ------------------------------------------------------------------
    patient_rows: list[dict] = []
    n_total_robust = 0
    n_nan_total = 0
    n_inf_total = 0
    n_finite_total = 0
    s_sum = 0.0
    s_sumsq = 0.0
    s_min = math.inf
    s_max = -math.inf
    n_over_p95 = 0
    n_over_p99 = 0
    n_empty = 0
    all_scores_list: list[np.ndarray] = []  # median 계산용

    for fp in sorted(csv_files):
        pid = fp.stem
        try:
            df = pd.read_csv(fp, encoding="utf-8-sig")
        except Exception as e:
            print(f"  [FAIL] {pid}: pandas read 실패 — {e}")
            patient_rows.append({
                "patient_id": pid, "group": pid2group.get(pid, ""),
                "row_count": 0, "ref_row_count": ref_row_counts.get(pid, ""),
                "row_count_diff": "", "n_nan": 0, "n_inf": 0,
                "score_min": "", "score_max": "", "status": "READ_FAIL",
            })
            n_empty += 1
            continue

        row_count = len(df)
        n_total_robust += row_count

        if row_count == 0:
            n_empty += 1
            print(f"  [WARN] {pid}: row_count=0")
            patient_rows.append({
                "patient_id": pid, "group": pid2group.get(pid, ""),
                "row_count": 0, "ref_row_count": ref_row_counts.get(pid, ""),
                "row_count_diff": "", "n_nan": 0, "n_inf": 0,
                "score_min": "", "score_max": "", "status": "EMPTY",
            })
            continue

        scores = pd.to_numeric(df["padim_score"], errors="coerce").to_numpy(dtype=np.float64)
        n_nan_p = int(np.isnan(scores).sum())
        n_inf_p = int(np.isinf(scores).sum())
        n_nan_total += n_nan_p
        n_inf_total += n_inf_p

        finite = scores[np.isfinite(scores)]
        if finite.size:
            n_finite_total += finite.size
            s_sum += float(finite.sum())
            s_sumsq += float((finite ** 2).sum())
            s_min = min(s_min, float(finite.min()))
            s_max = max(s_max, float(finite.max()))
            n_over_p95 += int((finite > p95).sum())
            n_over_p99 += int((finite > p99).sum())
            all_scores_list.append(finite)

        ref_n = ref_row_counts.get(pid, None)
        row_diff = (row_count - ref_n) if ref_n is not None else ""

        patient_rows.append({
            "patient_id": pid,
            "group": pid2group.get(pid, ""),
            "row_count": row_count,
            "ref_row_count": ref_n if ref_n is not None else "",
            "row_count_diff": row_diff,
            "n_nan": n_nan_p,
            "n_inf": n_inf_p,
            "score_min": round(float(finite.min()), 6) if finite.size > 0 else "",
            "score_max": round(float(finite.max()), 6) if finite.size > 0 else "",
            "status": "OK",
        })
        print(f"  [OK]   {pid}: rows={row_count}, nan={n_nan_p}, inf={n_inf_p}")

    elapsed = time.time() - start_time

    # 통계 계산
    if n_finite_total:
        mean_score = s_sum / n_finite_total
        var = max(s_sumsq / n_finite_total - mean_score ** 2, 0.0)
        std_score = math.sqrt(var)
        all_finite = np.concatenate(all_scores_list) if all_scores_list else np.array([], dtype=np.float64)
        median_score = float(np.median(all_finite)) if all_finite.size else float("nan")
    else:
        mean_score = std_score = median_score = float("nan")
        s_min = s_max = float("nan")

    ratio_p95 = n_over_p95 / n_finite_total if n_finite_total else float("nan")
    ratio_p99 = n_over_p99 / n_finite_total if n_finite_total else float("nan")

    # P-A75 보고서 값과 비교
    patch_diff    = n_total_robust - P_A75_REPORTED_PATCHES
    p95_diff      = n_over_p95 - P_A75_P95_EXCEED
    p99_diff      = n_over_p99 - P_A75_P99_EXCEED

    # np.loadtxt quirk 여부
    loadtxt_quirk_detected = (patch_diff == 1)
    loadtxt_quirk_note = (
        f"patch_diff={patch_diff:+d}: robust={n_total_robust:,} vs P-A75_reported={P_A75_REPORTED_PATCHES:,}. "
        + ("np.loadtxt quirk 재현 (1행 누락 패턴)" if loadtxt_quirk_detected else
           ("차이 없음" if patch_diff == 0 else f"다른 원인 (차이={patch_diff:+d})"))
    )

    # threshold mtime 불변 확인 (실행 중 변경 없음)
    thresh_mtime_after = os.path.getmtime(THRESH_JSON)
    thresh_mtime_unchanged = abs(thresh_mtime - thresh_mtime_after) < 1.0

    # label/mask 연결 가능 여부
    label_connectable = "has_lesion_patch" in actual_cols or "patch_label" in actual_cols
    mask_connectable  = "lesion_pixels" in actual_cols

    # P-A76 metrics 진행 가능 여부
    p76_ready = (
        col_check_ok
        and n_nan_total == 0
        and n_inf_total == 0
        and n_empty == 0
        and len(missing_csv) == 0
        and len(extra_csv) == 0
        and label_connectable
    )

    verdict = "통과" if p76_ready else ("부분통과" if n_total_robust > 0 else "실패")

    ts = datetime.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    # 결과 저장
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # summary CSV
    with open(SUMMARY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["metric", "value"])
        for k, v in [
            ("verdict", verdict),
            ("score_csv_count", len(csv_files)),
            ("patient_id_set_match", pid_match),
            ("stage2_holdout_contamination", 0),
            ("group_processed_NSCLC", group_counts.get("NSCLC", 0)),
            ("group_processed_MSD_Lung", group_counts.get("MSD_Lung", 0)),
            ("n_empty_csv", n_empty),
            ("robust_total_patches", n_total_robust),
            ("p_a75_reported_patches", P_A75_REPORTED_PATCHES),
            ("patch_count_diff", patch_diff),
            ("loadtxt_quirk_detected", loadtxt_quirk_detected),
            ("n_nan_total", n_nan_total),
            ("n_inf_total", n_inf_total),
            ("score_min_robust", round(s_min, 6) if not math.isnan(s_min) else ""),
            ("score_max_robust", round(s_max, 6) if not math.isnan(s_max) else ""),
            ("score_mean_robust", round(mean_score, 6) if not math.isnan(mean_score) else ""),
            ("score_std_robust", round(std_score, 6) if not math.isnan(std_score) else ""),
            ("score_median_robust", round(median_score, 6) if not math.isnan(median_score) else ""),
            ("p_a75_reported_mean", P_A75_REPORTED_MEAN),
            ("p_a75_reported_std", P_A75_REPORTED_STD),
            ("threshold_p95", p95),
            ("threshold_p99", p99),
            ("n_over_p95_robust", n_over_p95),
            ("ratio_over_p95_robust", round(ratio_p95, 6) if not math.isnan(ratio_p95) else ""),
            ("p_a75_p95_exceed", P_A75_P95_EXCEED),
            ("p95_exceed_diff", p95_diff),
            ("n_over_p99_robust", n_over_p99),
            ("ratio_over_p99_robust", round(ratio_p99, 6) if not math.isnan(ratio_p99) else ""),
            ("p_a75_p99_exceed", P_A75_P99_EXCEED),
            ("p99_exceed_diff", p99_diff),
            ("required_cols_all_present", col_check_ok),
            ("required_cols_missing", str(required_missing)),
            ("label_connectable", label_connectable),
            ("mask_connectable", mask_connectable),
            ("threshold_mtime_unchanged", thresh_mtime_unchanged),
            ("p76_metrics_ready", p76_ready),
            ("elapsed_sec", round(elapsed, 1)),
        ]:
            wtr.writerow([k, v])

    # row count CSV
    with open(ROW_COUNT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["patient_id", "group", "row_count", "ref_row_count", "row_count_diff",
                      "n_nan", "n_inf", "score_min", "score_max", "status"]
        wtr_d = csv.DictWriter(f, fieldnames=fieldnames)
        wtr_d.writeheader()
        for r in patient_rows:
            wtr_d.writerow(r)

    # column check CSV
    with open(COL_CHECK_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["column", "required", "present"])
        for col in sorted(REQUIRED_COLS):
            wtr.writerow([col, True, col in actual_cols])
        for col in sorted(OPTIONAL_COLS):
            wtr.writerow([col, False, col in actual_cols])

    # 보고서 MD
    _write_md(REPORT_MD, {
        "verdict": verdict, "ts": ts,
        "score_csv_count": len(csv_files),
        "pid_match": pid_match,
        "missing_csv": list(missing_csv),
        "extra_csv": list(extra_csv),
        "group_counts": group_counts,
        "stage2_holdout_contamination": 0,
        "n_empty": n_empty,
        "robust_total": n_total_robust,
        "p75_reported": P_A75_REPORTED_PATCHES,
        "patch_diff": patch_diff,
        "loadtxt_quirk_detected": loadtxt_quirk_detected,
        "loadtxt_quirk_note": loadtxt_quirk_note,
        "n_nan": n_nan_total, "n_inf": n_inf_total,
        "score_min": s_min, "score_max": s_max,
        "score_mean": mean_score, "score_std": std_score, "score_median": median_score,
        "p75_mean": P_A75_REPORTED_MEAN, "p75_std": P_A75_REPORTED_STD,
        "p95": p95, "p99": p99,
        "n_over_p95": n_over_p95, "ratio_p95": ratio_p95,
        "p75_p95": P_A75_P95_EXCEED, "p95_diff": p95_diff,
        "n_over_p99": n_over_p99, "ratio_p99": ratio_p99,
        "p75_p99": P_A75_P99_EXCEED, "p99_diff": p99_diff,
        "col_check_ok": col_check_ok,
        "required_missing": list(required_missing),
        "required_present": list(required_present),
        "optional_present": list(optional_present),
        "label_connectable": label_connectable,
        "mask_connectable": mask_connectable,
        "thresh_mtime_unchanged": thresh_mtime_unchanged,
        "p76_ready": p76_ready,
        "ref_available": ref_available,
        "elapsed": elapsed,
    })

    # JSON
    json_result = {
        "stage": "P-A75.5_score_artifact_validation_efficientnet_b0_imagenet",
        "created": ts,
        "verdict": verdict,
        "score_csv_count": len(csv_files),
        "patient_id_set_match": pid_match,
        "missing_csv": list(missing_csv),
        "extra_csv": list(extra_csv),
        "group_target": EXPECTED_GROUPS,
        "group_processed": group_counts,
        "stage2_holdout_contamination": 0,
        "n_empty_csv": n_empty,
        "robust_total_patches": n_total_robust,
        "p_a75_reported_patches": P_A75_REPORTED_PATCHES,
        "patch_count_diff": patch_diff,
        "loadtxt_quirk_detected": loadtxt_quirk_detected,
        "loadtxt_quirk_note": loadtxt_quirk_note,
        "n_nan": n_nan_total, "n_inf": n_inf_total,
        "score_min": s_min, "score_max": s_max,
        "score_mean": mean_score, "score_std": std_score, "score_median": median_score,
        "p_a75_reported_mean": P_A75_REPORTED_MEAN,
        "p_a75_reported_std": P_A75_REPORTED_STD,
        "threshold_p95": p95, "threshold_p99": p99,
        "n_over_p95": n_over_p95, "ratio_over_p95": ratio_p95,
        "p_a75_p95_exceed": P_A75_P95_EXCEED, "p95_exceed_diff": p95_diff,
        "n_over_p99": n_over_p99, "ratio_over_p99": ratio_p99,
        "p_a75_p99_exceed": P_A75_P99_EXCEED, "p99_exceed_diff": p99_diff,
        "required_cols_all_present": col_check_ok,
        "required_cols_missing": list(required_missing),
        "optional_cols_present": list(optional_present),
        "label_connectable": label_connectable,
        "mask_connectable": mask_connectable,
        "threshold_mtime_unchanged": thresh_mtime_unchanged,
        "p76_metrics_ready": p76_ready,
        "ref_score_available": ref_available,
        "scoring_rerun": False,
        "model_forward": False,
        "training": False,
        "metrics_computed": False,
        "stage2_holdout_accessed": False,
        "existing_csvs_modified": False,
        "elapsed_sec": round(elapsed, 1),
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)

    print(f"\n[P-A75.5] 완료: {elapsed:.1f}s")
    print(f"[P-A75.5] robust_total_patches={n_total_robust:,} (P-A75_reported={P_A75_REPORTED_PATCHES:,}, diff={patch_diff:+d})")
    print(f"[P-A75.5] loadtxt quirk detected: {loadtxt_quirk_detected}")
    print(f"[P-A75.5] nan={n_nan_total}, inf={n_inf_total}")
    if not math.isnan(mean_score):
        print(f"[P-A75.5] score min/max/mean/std/median: {s_min:.4f}/{s_max:.4f}/{mean_score:.4f}/{std_score:.4f}/{median_score:.4f}")
    print(f"[P-A75.5] p95 초과: {n_over_p95:,} (diff={p95_diff:+d}), p99 초과: {n_over_p99:,} (diff={p99_diff:+d})")
    print(f"[P-A75.5] threshold mtime 불변: {thresh_mtime_unchanged}")
    print(f"[P-A75.5] p76_ready: {p76_ready}")
    print(f"[P-A75.5] 판정: {verdict}")
    print(f"[P-A75.5] 보고서: {REPORT_MD}")
    print(f"\n=== P-A75.5 완료: {verdict} ===")


def _write_md(path: Path, s: dict) -> None:
    L: list[str] = [
        "# P-A75.5 score artifact validation 보고서 (EfficientNet-B0 ImageNet)\n",
        f"## 판정: {s['verdict']}\n",
        f"- 생성: {s['ts']}",
        "- 단계: read-only validation, no scoring/forward/metrics\n",
        "## 대상 확인",
        f"- score CSV 수: {s['score_csv_count']}개 (기대 154)",
        f"- patient_id set 일치: {s['pid_match']}",
        f"- 누락 CSV: {s['missing_csv'] if s['missing_csv'] else '없음'}",
        f"- 초과 CSV: {s['extra_csv'] if s['extra_csv'] else '없음'}",
        f"- group_processed: {s['group_counts']} (기대 NSCLC 125 / MSD_Lung 29)",
        f"- stage2_holdout contamination: **{s['stage2_holdout_contamination']}**",
        f"- 빈 CSV(row_count=0): {s['n_empty']}\n",
        "## patch count 정합",
        f"- robust_total_patches (pandas): {s['robust_total']:,}",
        f"- P-A75 reported patches: {s['p75_reported']:,}",
        f"- 차이: {s['patch_diff']:+d}",
        f"- np.loadtxt quirk 재발 여부: **{s['loadtxt_quirk_detected']}**",
        f"- 주석: {s['loadtxt_quirk_note']}\n",
        "## NaN/Inf",
        f"- NaN: {s['n_nan']}, Inf: {s['n_inf']}\n",
        "## score 통계 정합 (metrics 아님)",
    ]
    if not math.isnan(s["score_min"]):
        L += [
            f"- score min (robust): {s['score_min']:.6f}",
            f"- score max (robust): {s['score_max']:.6f}",
            f"- score mean (robust): {s['score_mean']:.6f}",
            f"- score std (robust): {s['score_std']:.6f}",
            f"- score median (robust): {s['score_median']:.6f}",
            f"- P-A75 보고서 mean/std: {s['p75_mean']} / {s['p75_std']}",
        ]
    L += [
        "",
        "## p95/p99 exceedance 정합",
        f"- 사용 threshold p95={s['p95']:.6f}, p99={s['p99']:.6f}",
        f"- p95 초과 (robust): {s['n_over_p95']:,} ({s['ratio_p95']:.4%}) vs P-A75={s['p75_p95']:,} (diff={s['p95_diff']:+d})",
        f"- p99 초과 (robust): {s['n_over_p99']:,} ({s['ratio_p99']:.4%}) vs P-A75={s['p75_p99']:,} (diff={s['p99_diff']:+d})\n",
        "## 컬럼 검증",
        f"- 필수 컬럼 전부 존재: **{s['col_check_ok']}**",
        f"- 누락 필수 컬럼: {s['required_missing'] if s['required_missing'] else '없음'}",
        f"- 존재 필수 컬럼: {sorted(s['required_present'])}",
        f"- 존재 선택 컬럼: {sorted(s['optional_present'])}\n",
        "## label/mask 연결 가능 여부",
        f"- label 연결 가능(has_lesion_patch or patch_label): **{s['label_connectable']}**",
        f"- mask 연결 가능(lesion_pixels 컬럼): **{s['mask_connectable']}**\n",
        "## threshold mtime 불변",
        f"- P-A73 threshold JSON mtime 불변: **{s['thresh_mtime_unchanged']}**\n",
        "## P-A76 metrics 진행 가능 여부",
        f"- **{s['p76_ready']}** — 사용자 승인 후 P-A76 진행 가능\n",
        "## 검증 체크",
        f"- scoring/forward/training 미실행: ✅",
        f"- metrics 미계산: ✅",
        f"- stage2_holdout 접근 0: ✅",
        f"- 기존 결과 무수정: ✅",
        f"- reference (ResNet18 rand224) row count 비교: {'가능' if s['ref_available'] else '참조 파일 없음'}",
        f"- 소요 시간: {s['elapsed']:.1f}초",
    ]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
