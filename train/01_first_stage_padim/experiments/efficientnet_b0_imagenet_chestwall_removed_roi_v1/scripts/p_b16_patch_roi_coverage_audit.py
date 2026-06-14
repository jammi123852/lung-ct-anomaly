"""
P-B16: First-Stage PaDiM Patch ROI Coverage / Position-Bin Audit

read-only audit. 학습/feature extraction/scoring/threshold/metrics 금지.
stage2_holdout/lesion 접근 금지. 기존 결과 수정 금지.

분석:
- normal_train 290명 기준
- v4_20 refined ROI 기준으로 각 patch의 ROI coverage 계산
- position_bin별 / central-peripheral별 / z_level별 집계
- filtering 전(patch_csv 원본) vs filtering 후(v4_20 ratio >= 0.5) 비교
"""
from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 경로 설정
PROJ_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT  = Path(__file__).resolve().parents[1]

SPLIT_JSON   = PROJ_ROOT / "outputs" / "position-aware-padim-v1" / "splits" / "normal_v1.json"
V4_20_ROOT   = PROJ_ROOT / "outputs" / "mip-postprocess-research-v1" / "masks" / \
               "refined_roi_v4_20_modeB_all_v1" / "normal"
MANIFEST_CSV = Path("/mnt/c/Users/jinhy/Desktop/"
                    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/"
                    "manifests/patient_manifest.csv")

OUT_DIR = EXP_ROOT / "outputs" / "reports" / "p_b16_patch_roi_coverage_position_bin_audit"

V4_20_THRESHOLD      = 0.5
MAX_LOW_ROI_EXAMPLES = 500  # low_roi_examples CSV 최대 행 수

EXPECTED_TRAIN_N     = 290
POSITION_BINS        = [
    "upper_central", "upper_peripheral",
    "middle_central", "middle_peripheral",
    "lower_central",  "lower_peripheral",
]
Z_LEVELS = ["upper", "middle", "lower"]


# ─── 통계 누적 헬퍼 ────────────────────────────────────────────────────────────

def make_stats_accumulator():
    return {
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "vals_sample": [],  # 분위수 계산용 샘플 (최대 50만 개)
        # threshold별 카운트
        "cnt_lt10": 0,
        "cnt_lt25": 0,
        "cnt_lt50": 0,
        "cnt_ge90": 0,
        # filtering
        "cnt_before": 0,  # v4_20 filtering 전 (patch_csv 전체)
        "cnt_after": 0,   # v4_20 filtering 후 (ratio >= 0.5)
        "cnt_removed": 0,
        # 경계성 (0.5~0.6)
        "cnt_borderline": 0,
    }


SAMPLE_MAX = 500_000  # 분위수 계산을 위한 최대 샘플


def accumulate(acc, ratio, sample_cap=SAMPLE_MAX):
    acc["count"] += 1
    acc["sum"] += ratio
    acc["sum_sq"] += ratio * ratio
    if ratio < acc["min"]: acc["min"] = ratio
    if ratio > acc["max"]: acc["max"] = ratio
    if acc["count"] <= sample_cap:
        acc["vals_sample"].append(ratio)
    # threshold
    if ratio < 0.10: acc["cnt_lt10"] += 1
    if ratio < 0.25: acc["cnt_lt25"] += 1
    if ratio < 0.50: acc["cnt_lt50"] += 1
    if ratio >= 0.90: acc["cnt_ge90"] += 1
    # filtering
    acc["cnt_before"] += 1
    if ratio >= V4_20_THRESHOLD:
        acc["cnt_after"] += 1
    else:
        acc["cnt_removed"] += 1
    # borderline
    if V4_20_THRESHOLD <= ratio < 0.60:
        acc["cnt_borderline"] += 1


def finalize_stats(acc):
    n = acc["count"]
    if n == 0:
        return {
            "count": 0, "mean": None, "median": None, "std": None,
            "min": None, "max": None,
            "p1": None, "p5": None, "p25": None, "p75": None, "p95": None,
            "cnt_lt10": 0, "rate_lt10": None,
            "cnt_lt25": 0, "rate_lt25": None,
            "cnt_lt50": 0, "rate_lt50": None,
            "cnt_ge90": 0, "rate_ge90": None,
            "cnt_before": 0, "cnt_after": 0, "cnt_removed": 0,
            "removed_rate": None,
            "cnt_borderline": 0, "borderline_rate": None,
        }
    mean = acc["sum"] / n
    var  = max(0.0, acc["sum_sq"] / n - mean * mean)
    std  = var ** 0.5
    arr  = np.array(acc["vals_sample"], dtype=np.float32)
    pcts = np.percentile(arr, [1, 5, 25, 50, 75, 95]).tolist()
    return {
        "count": n,
        "mean": round(mean, 6),
        "median": round(float(pcts[3]), 6),
        "std": round(std, 6),
        "min": round(acc["min"], 6),
        "max": round(acc["max"], 6),
        "p1":  round(float(pcts[0]), 6),
        "p5":  round(float(pcts[1]), 6),
        "p25": round(float(pcts[2]), 6),
        "p75": round(float(pcts[4]), 6),
        "p95": round(float(pcts[5]), 6),
        "cnt_lt10":  acc["cnt_lt10"],  "rate_lt10":  round(acc["cnt_lt10"]  / n, 6),
        "cnt_lt25":  acc["cnt_lt25"],  "rate_lt25":  round(acc["cnt_lt25"]  / n, 6),
        "cnt_lt50":  acc["cnt_lt50"],  "rate_lt50":  round(acc["cnt_lt50"]  / n, 6),
        "cnt_ge90":  acc["cnt_ge90"],  "rate_ge90":  round(acc["cnt_ge90"]  / n, 6),
        "cnt_before":  acc["cnt_before"],
        "cnt_after":   acc["cnt_after"],
        "cnt_removed": acc["cnt_removed"],
        "removed_rate": round(acc["cnt_removed"] / n, 6),
        "cnt_borderline": acc["cnt_borderline"],
        "borderline_rate": round(acc["cnt_borderline"] / acc["cnt_after"], 6) if acc["cnt_after"] > 0 else 0.0,
    }


# ─── 핵심 처리 ─────────────────────────────────────────────────────────────────

def run_audit():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_start = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()

    errors = []
    guardrail = {
        "stage2_holdout_accessed": False,
        "model_roi_used": False,
        "e_drive_used": False,
        "lesion_accessed": False,
        "pc_supervised_used": False,
        "existing_results_modified": False,
        "training_done": False,
        "feature_extraction_done": False,
        "scoring_done": False,
    }

    # ── 입력 검증 ────────────────────────────────────────────────────────────
    input_sources = []
    if not SPLIT_JSON.exists():
        errors.append(f"SPLIT_JSON 없음: {SPLIT_JSON}")
    else:
        input_sources.append({"file": str(SPLIT_JSON), "status": "found", "note": "normal_train split"})

    if not MANIFEST_CSV.exists():
        errors.append(f"MANIFEST_CSV 없음: {MANIFEST_CSV}")
    else:
        input_sources.append({"file": str(MANIFEST_CSV), "status": "found", "note": "patient_manifest"})

    if errors:
        _save_errors(errors)
        print(f"[P-B16] 입력 파일 없음 → 중단: {errors}")
        return

    split_data = json.load(open(SPLIT_JSON, encoding="utf-8"))
    train_pids  = list(split_data["train"])
    p2s         = split_data.get("patient_to_safe_id", {})
    assert len(train_pids) == EXPECTED_TRAIN_N, f"train 수 {len(train_pids)} ≠ {EXPECTED_TRAIN_N}"

    manifest = pd.read_csv(MANIFEST_CSV, encoding="utf-8-sig")
    patch_csv_map = dict(zip(manifest["patient_id"], manifest["patch_csv"]))

    input_sources.append({"file": str(V4_20_ROOT), "status": "directory", "note": "v4_20 ROI root"})

    # ── 누적기 초기화 ────────────────────────────────────────────────────────
    # 전체
    acc_all_before = make_stats_accumulator()
    acc_all_after  = make_stats_accumulator()

    # position_bin 별
    acc_bin_before = {b: make_stats_accumulator() for b in POSITION_BINS}
    acc_bin_after  = {b: make_stats_accumulator() for b in POSITION_BINS}

    # central/peripheral 별
    acc_cp_before = {"central": make_stats_accumulator(), "peripheral": make_stats_accumulator()}
    acc_cp_after  = {"central": make_stats_accumulator(), "peripheral": make_stats_accumulator()}

    # z_level 별
    acc_zl_before = {z: make_stats_accumulator() for z in Z_LEVELS}
    acc_zl_after  = {z: make_stats_accumulator() for z in Z_LEVELS}

    # patient 레벨 요약
    patient_rows = []

    # low_roi examples (before: ratio < 0.25)
    low_roi_examples = []

    # ── patient 단위 streaming ────────────────────────────────────────────────
    n_processed = 0
    for idx, pid in enumerate(train_pids):
        safe_id = p2s.get(pid, pid)
        roi_path   = V4_20_ROOT / safe_id / "refined_roi.npy"
        csv_path_s = patch_csv_map.get(pid, "")
        csv_path   = Path(csv_path_s) if csv_path_s else None

        # 파일 존재 확인
        if not roi_path.exists():
            errors.append(f"{pid}: v4_20 ROI 없음 ({roi_path})")
            continue
        if csv_path is None or not csv_path.exists():
            errors.append(f"{pid}: patch_csv 없음 ({csv_path_s})")
            continue

        try:
            roi = np.load(str(roi_path), mmap_mode="r").astype(bool)
            pdf = pd.read_csv(str(csv_path), encoding="utf-8-sig",
                              usecols=["local_z", "y0", "x0", "y1", "x1",
                                       "position_bin", "z_level", "central_peripheral"])
        except Exception as exc:
            errors.append(f"{pid}: 로드 오류: {exc}")
            continue

        D, H, W = roi.shape
        patch_size = 32  # 고정 (manifest 확인됨)

        # patient 레벨 누적기
        p_acc_before = make_stats_accumulator()
        p_acc_after  = make_stats_accumulator()
        p_bin_before = {b: 0 for b in POSITION_BINS}
        p_bin_after  = {b: 0 for b in POSITION_BINS}

        for row in pdf.itertuples(index=False):
            z  = int(row.local_z)
            y0 = int(row.y0); x0 = int(row.x0)
            y1 = int(row.y1); x1 = int(row.x1)
            pbin = str(row.position_bin)
            zlev = str(row.z_level)
            cp   = str(row.central_peripheral)

            # 범위 클리핑
            if z < 0 or z >= D:
                errors.append(f"{pid}: z={z} out of range D={D}")
                continue
            ry0 = max(0, y0); ry1 = min(H, y1)
            rx0 = max(0, x0); rx1 = min(W, x1)
            sub = roi[z, ry0:ry1, rx0:rx1]
            area = (y1 - y0) * (x1 - x0)
            if area <= 0:
                continue
            roi_pixels = int(sub.sum())
            ratio = roi_pixels / area

            passed = (ratio >= V4_20_THRESHOLD)

            # 전역 before 집계
            accumulate(acc_all_before, ratio)
            if passed:
                accumulate(acc_all_after, ratio)

            # position_bin
            if pbin in acc_bin_before:
                accumulate(acc_bin_before[pbin], ratio)
                if passed:
                    accumulate(acc_bin_after[pbin], ratio)

            # central/peripheral
            if cp in acc_cp_before:
                accumulate(acc_cp_before[cp], ratio)
                if passed:
                    accumulate(acc_cp_after[cp], ratio)

            # z_level
            if zlev in acc_zl_before:
                accumulate(acc_zl_before[zlev], ratio)
                if passed:
                    accumulate(acc_zl_after[zlev], ratio)

            # patient
            accumulate(p_acc_before, ratio)
            if passed:
                accumulate(p_acc_after, ratio)
            p_bin_before[pbin] = p_bin_before.get(pbin, 0) + 1
            if passed:
                p_bin_after[pbin] = p_bin_after.get(pbin, 0) + 1

            # low_roi examples
            if ratio < 0.25 and len(low_roi_examples) < MAX_LOW_ROI_EXAMPLES:
                low_roi_examples.append({
                    "patient_id": pid, "safe_id": safe_id,
                    "local_z": z, "y0": y0, "x0": x0, "y1": y1, "x1": x1,
                    "position_bin": pbin, "z_level": zlev,
                    "central_peripheral": cp,
                    "roi_patch_ratio": round(ratio, 6),
                    "passed_v4_20_filter": passed,
                })

        # patient 레벨 요약 저장
        pb_s = finalize_stats(p_acc_before)
        pa_s = finalize_stats(p_acc_after)
        patient_rows.append({
            "patient_id": pid, "safe_id": safe_id,
            "before_count": pb_s["count"],
            "after_count":  pa_s["count"],
            "removed_count": pb_s["cnt_removed"],
            "removed_rate":  pb_s["removed_rate"],
            "mean_roi_before":   pb_s["mean"],
            "mean_roi_after":    pa_s["mean"],
            "median_roi_before": pb_s["median"],
            "median_roi_after":  pa_s["median"],
            "cnt_lt25_before":   pb_s["cnt_lt25"],
            "rate_lt25_before":  pb_s["rate_lt25"],
            "cnt_borderline_after": pa_s["cnt_borderline"],
            "borderline_rate_after": pa_s["borderline_rate"],
            **{f"before_{b}": p_bin_before.get(b, 0) for b in POSITION_BINS},
            **{f"after_{b}":  p_bin_after.get(b, 0)  for b in POSITION_BINS},
        })

        n_processed += 1
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{EXPECTED_TRAIN_N}] {pid}: "
                  f"before={p_acc_before['count']:,} "
                  f"after={p_acc_after['count']:,} "
                  f"elapsed={elapsed:.1f}s")

    elapsed_total = time.time() - t0
    print(f"\n[P-B16] 처리 완료: {n_processed}/{EXPECTED_TRAIN_N}명, elapsed={elapsed_total:.1f}s")

    # ── 통계 최종화 ──────────────────────────────────────────────────────────
    ts_end = datetime.now().isoformat(timespec="seconds")
    s_all_before = finalize_stats(acc_all_before)
    s_all_after  = finalize_stats(acc_all_after)

    s_bin_before = {b: finalize_stats(acc_bin_before[b]) for b in POSITION_BINS}
    s_bin_after  = {b: finalize_stats(acc_bin_after[b])  for b in POSITION_BINS}

    s_cp_before = {k: finalize_stats(v) for k, v in acc_cp_before.items()}
    s_cp_after  = {k: finalize_stats(v) for k, v in acc_cp_after.items()}

    s_zl_before = {k: finalize_stats(v) for k, v in acc_zl_before.items()}
    s_zl_after  = {k: finalize_stats(v) for k, v in acc_zl_after.items()}

    # ── CSV 저장 ─────────────────────────────────────────────────────────────

    # 1. overall summary
    _save_csv(OUT_DIR / "p_b16_patch_roi_coverage_overall_summary.csv", [
        {
            "split": "before_v4_20_filter",
            **s_all_before,
        },
        {
            "split": "after_v4_20_filter",
            **s_all_after,
        },
    ])

    # 2. by position_bin
    rows_bin = []
    for b in POSITION_BINS:
        sb = s_bin_before[b]; sa = s_bin_after[b]
        rows_bin.append({
            "position_bin": b, "split": "before",
            **sb,
        })
        rows_bin.append({
            "position_bin": b, "split": "after",
            **sa,
        })
    _save_csv(OUT_DIR / "p_b16_patch_roi_coverage_by_position_bin.csv", rows_bin)

    # 3. by central/peripheral
    rows_cp = []
    for k in ["central", "peripheral"]:
        rows_cp.append({"central_peripheral": k, "split": "before", **s_cp_before[k]})
        rows_cp.append({"central_peripheral": k, "split": "after",  **s_cp_after[k]})
    _save_csv(OUT_DIR / "p_b16_patch_roi_coverage_by_central_peripheral.csv", rows_cp)

    # 4. by z_level
    rows_zl = []
    for z in Z_LEVELS:
        rows_zl.append({"z_level": z, "split": "before", **s_zl_before[z]})
        rows_zl.append({"z_level": z, "split": "after",  **s_zl_after[z]})
    _save_csv(OUT_DIR / "p_b16_patch_roi_coverage_by_z_level.csv", rows_zl)

    # 5. filter before/after summary (bin 단위)
    rows_fa = []
    for b in POSITION_BINS:
        sb = s_bin_before[b]
        rows_fa.append({
            "position_bin": b,
            "before_count": sb["count"],
            "after_count":  sb["cnt_after"],
            "removed_count": sb["cnt_removed"],
            "removed_rate": sb["removed_rate"],
            "mean_roi_before": sb["mean"],
            "mean_roi_after": s_bin_after[b]["mean"],
            "borderline_rate_after": s_bin_after[b]["borderline_rate"],
        })
    _save_csv(OUT_DIR / "p_b16_patch_filter_before_after_summary.csv", rows_fa)

    # 6. low_roi examples
    _save_csv(OUT_DIR / "p_b16_low_roi_patch_examples.csv", low_roi_examples)

    # 7. patient level
    _save_csv(OUT_DIR / "p_b16_patient_level_summary.csv", patient_rows)

    # 8. input source validation
    _save_csv(OUT_DIR / "p_b16_input_source_validation.csv", input_sources)

    # 9. guardrail check
    _save_csv(OUT_DIR / "p_b16_guardrail_check.csv", [
        {"check": k, "status": ("PASS" if not v else "FAIL"), "value": str(v)}
        for k, v in guardrail.items()
    ])

    # 10. errors
    _save_errors(errors)

    # ── 판정 ─────────────────────────────────────────────────────────────────
    filtering_before_available = True  # patch_csv가 roi_0_0 >= 0.5 이후 상태 → 설명 필요
    any_guardrail_fail = any(guardrail.values())

    if any_guardrail_fail:
        verdict = "실패"
    elif n_processed < EXPECTED_TRAIN_N:
        verdict = "부분통과"
    else:
        verdict = "통과"

    # ── JSON 보고서 ───────────────────────────────────────────────────────────
    report = {
        "step": "P-B16",
        "verdict": verdict,
        "timestamp_start": ts_start,
        "timestamp_end": ts_end,
        "elapsed_seconds": round(elapsed_total, 2),
        "n_patients_processed": n_processed,
        "n_patients_expected": EXPECTED_TRAIN_N,
        "official_roi": "refined_roi_v4_20_modeB_all_v1",
        "v4_20_threshold": V4_20_THRESHOLD,
        "filtering_before_source": "patch_csv (roi_0_0 >= 0.5 기준 이미 필터링된 상태)",
        "filtering_before_note": (
            "patch_csv 자체가 roi_0_0 ratio >= 0.5로 생성된 좌표임. "
            "따라서 'filtering 전'은 v4_20 미적용 상태의 patch_csv 전체이며, "
            "'filtering 후'는 추가로 v4_20 ROI ratio >= 0.5를 통과한 subset임."
        ),
        "overall_before": s_all_before,
        "overall_after": s_all_after,
        "by_position_bin_before": s_bin_before,
        "by_position_bin_after": s_bin_after,
        "by_central_peripheral_before": s_cp_before,
        "by_central_peripheral_after": s_cp_after,
        "by_z_level_before": s_zl_before,
        "by_z_level_after": s_zl_after,
        "n_errors": len(errors),
        "n_low_roi_examples": len(low_roi_examples),
        "guardrail": guardrail,
        "next_step": "P-C supervised auxiliary branch revival preflight",
    }
    with open(OUT_DIR / "p_b16_patch_roi_coverage_position_bin_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)

    # ── Markdown 보고서 ───────────────────────────────────────────────────────
    _write_markdown(report, s_bin_before, s_bin_after, s_cp_before, s_cp_after,
                    s_zl_before, s_zl_after, s_all_before, s_all_after,
                    n_processed, errors, elapsed_total)

    print(f"\n=== P-B16 완료: {verdict} ===")
    print(f"출력 디렉토리: {OUT_DIR}")


# ─── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _save_csv(path, rows):
    if not rows:
        path.write_text("(empty)\n", encoding="utf-8-sig")
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _save_errors(errors):
    rows = [{"index": i, "message": e} for i, e in enumerate(errors)]
    _save_csv(OUT_DIR / "p_b16_errors.csv", rows)


def _json_default(o):
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    return str(o)


def _fmt(v):
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:.4f}"
    return str(v)


def _write_markdown(report, s_bin_before, s_bin_after, s_cp_before, s_cp_after,
                    s_zl_before, s_zl_after, s_all_before, s_all_after,
                    n_processed, errors, elapsed):
    verdict = report["verdict"]
    ts = report["timestamp_end"]
    lines = [
        "# P-B16 Patch ROI Coverage / Position-Bin Audit\n",
        f"**판정: {verdict}**\n",
        f"- 생성일시: {ts}",
        f"- 처리 환자: {n_processed}/{report['n_patients_expected']}",
        f"- 공식 ROI: `refined_roi_v4_20_modeB_all_v1`",
        f"- v4_20 threshold: {report['v4_20_threshold']}",
        f"- elapsed: {elapsed:.1f}s",
        f"- 오류 수: {len(errors)}",
        "",
        "## Filtering 소스 주의",
        f"> {report['filtering_before_note']}",
        "",
        "## 전체 통계 (filtering 전/후)",
        "",
        "| split | count | mean | median | std | p5 | p25 | p75 | p95 | lt25_rate | lt50_rate | ge90_rate | removed_rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for split_name, s in [("before_v4_20", s_all_before), ("after_v4_20", s_all_after)]:
        lines.append(
            f"| {split_name} | {s['count']:,} | {_fmt(s['mean'])} | {_fmt(s['median'])} | "
            f"{_fmt(s['std'])} | {_fmt(s['p5'])} | {_fmt(s['p25'])} | {_fmt(s['p75'])} | "
            f"{_fmt(s['p95'])} | {_fmt(s['rate_lt25'])} | {_fmt(s['rate_lt50'])} | "
            f"{_fmt(s['rate_ge90'])} | {_fmt(s['removed_rate'])} |"
        )

    lines += [
        "",
        "## Position-Bin별 ROI 포함률 (before v4_20 filter)",
        "",
        "| position_bin | count | mean | median | lt25_rate | lt50_rate | ge90_rate | removed_rate | borderline_after |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for b in ["upper_central", "upper_peripheral", "middle_central",
              "middle_peripheral", "lower_central", "lower_peripheral"]:
        sb = s_bin_before[b]; sa = s_bin_after[b]
        lines.append(
            f"| {b} | {sb['count']:,} | {_fmt(sb['mean'])} | {_fmt(sb['median'])} | "
            f"{_fmt(sb['rate_lt25'])} | {_fmt(sb['rate_lt50'])} | {_fmt(sb['rate_ge90'])} | "
            f"{_fmt(sb['removed_rate'])} | {_fmt(sa['borderline_rate'])} |"
        )

    lines += [
        "",
        "## Central vs Peripheral",
        "",
        "| group | split | count | mean | median | lt25_rate | lt50_rate | ge90_rate | removed_rate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in ["central", "peripheral"]:
        for split_name, s in [("before", s_cp_before[k]), ("after", s_cp_after[k])]:
            lines.append(
                f"| {k} | {split_name} | {s['count']:,} | {_fmt(s['mean'])} | {_fmt(s['median'])} | "
                f"{_fmt(s['rate_lt25'])} | {_fmt(s['rate_lt50'])} | {_fmt(s['rate_ge90'])} | "
                f"{_fmt(s['removed_rate'])} |"
            )

    lines += [
        "",
        "## Z-Level별",
        "",
        "| z_level | split | count | mean | median | lt25_rate | removed_rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for z in Z_LEVELS:
        for split_name, s in [("before", s_zl_before[z]), ("after", s_zl_after[z])]:
            lines.append(
                f"| {z} | {split_name} | {s['count']:,} | {_fmt(s['mean'])} | {_fmt(s['median'])} | "
                f"{_fmt(s['rate_lt25'])} | {_fmt(s['removed_rate'])} |"
            )

    lines += [
        "",
        "## Filtering 전후 제거율 (position_bin별)",
        "",
        "| position_bin | before | after | removed | removed_rate |",
        "|---|---|---|---|---|",
    ]
    for b in ["upper_central", "upper_peripheral", "middle_central",
              "middle_peripheral", "lower_central", "lower_peripheral"]:
        sb = s_bin_before[b]
        lines.append(
            f"| {b} | {sb['count']:,} | {sb['cnt_after']:,} | "
            f"{sb['cnt_removed']:,} | {_fmt(sb['removed_rate'])} |"
        )

    # 핵심 질문 답변
    lines += [
        "",
        "## 핵심 질문 답변",
        "",
        f"1. 전체 v4_20 ROI 포함률: before mean={_fmt(s_all_before['mean'])}, after mean={_fmt(s_all_after['mean'])}",
        f"2. central before mean: {_fmt(s_cp_before['central']['mean'])}",
        f"3. peripheral before mean: {_fmt(s_cp_before['peripheral']['mean'])}",
        f"4. filtering 제거율: {_fmt(s_all_before['removed_rate'])} "
        f"(제거 {s_all_before['cnt_removed']:,} / 전체 {s_all_before['count']:,})",
        f"5. filtering 후 경계성(0.5~0.6) patch: {_fmt(s_all_after['borderline_rate'])}",
        f"6. lower_peripheral removed_rate: {_fmt(s_bin_before['lower_peripheral']['removed_rate'])}",
        f"7. lower_peripheral lt25_rate(before): {_fmt(s_bin_before['lower_peripheral']['rate_lt25'])}",
        "",
        "## Guardrail",
        "",
    ]
    for k, v in report["guardrail"].items():
        status = "✓ PASS" if not v else "✗ FAIL"
        lines.append(f"- {k}: {status}")

    lines += [
        "",
        "## 다음 단계",
        "",
        "- P-C supervised auxiliary branch revival preflight",
        "",
        "## 분석 범위",
        "",
        "- normal_train 290명 (stage2_holdout 미접근)",
        "- normal_val/test 제외",
        "- lesion/stage1_dev 제외",
        "- P-C supervised artifact 미사용",
        "- 기존 결과 수정 없음",
    ]

    with open(OUT_DIR / "p_b16_patch_roi_coverage_position_bin_audit.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        run_audit()
    except Exception as e:
        traceback.print_exc()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        _save_errors([f"FATAL: {e}"])
        sys.exit(1)
