"""
P-A75: EfficientNet-B0 ImageNet PaDiM — lesion stage1_dev 154명 scoring.

- P-A73 threshold(p95/p99)를 read-only로 로드한다.
- lesion stage1_dev 154명(NSCLC 125 + MSD_Lung 29)만 scoring한다.
- stage2_holdout 접근 금지, metrics 계산 금지.
- streaming 집계로 전체 score를 메모리에 올리지 않는다.
- resume 지원: 이미 생성된 score CSV는 skip한다.

실행:
  source ~/ai_env/bin/activate && python experiments/efficientnet_b0_imagenet_v1/code/p_a75_lesion_stage1_dev_scoring.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT  = Path(__file__).resolve().parents[1]
SRC_DIR   = PROJ_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

BACKBONE            = "efficientnet_b0"
PRETRAIN_SOURCE     = "imagenet"
RAW_FEATURE_DIM     = 144
REDUCED_FEATURE_DIM = 100
MASK_TYPE           = "roi_0_0"
PATHS_CONFIG        = "paths.local.v2_roi0_0.yaml"
SCRIPT_NAME         = "p_a75_lesion_stage1_dev_scoring.py"
RUN_TAG             = "padim_efficientnet_b0_imagenet"
EXPECTED_STAGE      = "stage1_dev"
EXPECTED_N          = 154
EXPECTED_GROUPS     = {"NSCLC": 125, "MSD_Lung": 29}
EXPECTED_P95        = 13.240479
EXPECTED_P99        = 15.332286
THRESH_TOLERANCE    = 1e-4
JOIN_KEY            = "patient_id"

# ---- 입력 경로 ----
MODEL_NPZ        = EXP_ROOT / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
SELECTED_INDICES = EXP_ROOT / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
THRESH_JSON      = EXP_ROOT / "outputs" / "evaluation" / "normal_val_thresholds" / "normal_val_threshold.json"
LESION_SPLIT     = PROJ_ROOT / "outputs" / "second-stage-lesion-refiner-v1" / "splits" / "lesion_stage_split_v1_balanced.csv"
LESION_ROOT      = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1")
LESION_MANIFEST  = LESION_ROOT / "manifests" / "patient_manifest.csv"

P_A74_REPORT_MD  = EXP_ROOT / "outputs" / "reports" / "normal_test" / "p_a74_normal_test_sanity.md"

# ---- 출력 경로 ----
SCORE_DIR   = EXP_ROOT / "outputs" / "scores" / "lesion_stage1_dev_by_patient"
EVAL_DIR    = EXP_ROOT / "outputs" / "evaluation" / "lesion_stage1_dev_scoring"
REPORT_DIR  = EXP_ROOT / "outputs" / "reports" / "lesion_stage1_dev"
ERROR_CSV   = EVAL_DIR / "error.csv"

SCORING_SUMMARY_JSON = EVAL_DIR / "lesion_stage1_dev_scoring_summary.json"
SCORING_SUMMARY_CSV  = EVAL_DIR / "lesion_stage1_dev_scoring_summary.csv"
REPORT_MD            = REPORT_DIR / "p_a75_lesion_stage1_dev_scoring.md"
REPORT_JSON          = REPORT_DIR / "p_a75_lesion_stage1_dev_scoring.json"
RUNTIME_CSV          = REPORT_DIR / "p_a75_runtime_summary.csv"


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def abort(msg: str, code: int = 2):
    print(f"[P-A75][ABORT] {msg}")
    sys.exit(code)


def record_error(pid: str, error_type: str, error_msg: str, file_logical: str) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not ERROR_CSV.exists() or ERROR_CSV.stat().st_size == 0
    with open(ERROR_CSV, "a", encoding="utf-8-sig", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["patient_id", "error_type", "error_msg", "file_logical"])
        if write_header:
            wtr.writeheader()
        wtr.writerow({"patient_id": pid, "error_type": error_type,
                      "error_msg": error_msg, "file_logical": file_logical})


def run_guards():
    # G1: P-A74 보고서 통과 확인
    if not P_A74_REPORT_MD.exists():
        abort(f"P-A74 보고서 없음: {P_A74_REPORT_MD}")
    with open(P_A74_REPORT_MD, encoding="utf-8") as f:
        md_text = f.read()
    if "판정: 통과" not in md_text:
        abort(f"P-A74 보고서가 통과 상태가 아님: {P_A74_REPORT_MD}")
    print("[G1] P-A74 보고서 통과 확인 ✅")

    # G2: P-A73 threshold JSON 존재 확인
    if not THRESH_JSON.exists():
        abort(f"P-A73 threshold JSON 없음: {THRESH_JSON}")
    thresh_mtime_before = os.path.getmtime(THRESH_JSON)
    with open(THRESH_JSON, encoding="utf-8") as f:
        th = json.load(f)
    p95 = float(th["threshold_p95"])
    p99 = float(th["threshold_p99"])
    print(f"[G2] threshold JSON 로드: p95={p95:.6f}, p99={p99:.6f}")

    # G3: threshold 값 일치 확인
    if abs(p95 - EXPECTED_P95) > THRESH_TOLERANCE:
        abort(f"p95 threshold 불일치: {p95:.6f} (기대: {EXPECTED_P95})")
    if abs(p99 - EXPECTED_P99) > THRESH_TOLERANCE:
        abort(f"p99 threshold 불일치: {p99:.6f} (기대: {EXPECTED_P99})")
    print(f"[G3] threshold 값 일치 ✅: p95={p95:.6f}, p99={p99:.6f} (재계산 없음)")

    # G4: distribution npz 존재 확인
    if not MODEL_NPZ.exists():
        abort(f"distribution npz 없음: {MODEL_NPZ}")
    print(f"[G4] distribution 존재: {MODEL_NPZ}")

    # G5: selected_feature_indices 검증
    if not SELECTED_INDICES.exists():
        abort(f"selected_feature_indices.npy 없음: {SELECTED_INDICES}")
    idx = np.load(SELECTED_INDICES)
    if idx.shape != (REDUCED_FEATURE_DIM,):
        abort(f"selected_index shape 불일치: {idx.shape}")
    if len(set(idx.tolist())) != REDUCED_FEATURE_DIM:
        abort(f"selected_index unique 불일치: {len(set(idx.tolist()))}")
    if not ((idx >= 0).all() and (idx < RAW_FEATURE_DIM).all()):
        abort(f"selected_index range 불일치: min={int(idx.min())}, max={int(idx.max())}")
    print(f"[G5] selected_feature_indices OK: shape={idx.shape}, unique={REDUCED_FEATURE_DIM}, range=[{int(idx.min())},{int(idx.max())}]")

    # G6: lesion split stage1_dev 154명 확인
    if not LESION_SPLIT.exists():
        abort(f"lesion split CSV 없음: {LESION_SPLIT}")
    split_rows = list(csv.DictReader(open(LESION_SPLIT, encoding="utf-8-sig")))
    dev_rows = [r for r in split_rows if r["stage_split"] == EXPECTED_STAGE]
    if len(dev_rows) != EXPECTED_N:
        abort(f"stage1_dev 환자 수가 {len(dev_rows)} (기대 {EXPECTED_N})")
    print(f"[G6] stage1_dev: {len(dev_rows)}명 확인 ✅")

    # G7: NSCLC 125 / MSD_Lung 29 확인
    groups = dict(Counter(r["group"] for r in dev_rows))
    if groups != EXPECTED_GROUPS:
        abort(f"stage1_dev group 구성 불일치: {groups} (기대 {EXPECTED_GROUPS})")
    print(f"[G7] group 구성: {groups} ✅")

    # G8: stage2_holdout 혼입 0 확인
    holdout_ids = {r[JOIN_KEY] for r in split_rows if r["stage_split"] == "stage2_holdout"}
    dev_ids = [r[JOIN_KEY] for r in dev_rows]
    contaminated = [pid for pid in dev_ids if pid in holdout_ids]
    if contaminated:
        abort(f"stage2_holdout 혼입 감지: {contaminated}")
    print(f"[G8] stage2_holdout 혼입 0 확인 ✅")

    # G9: lesion root roi_0_0 조건 확인
    if "roi0_0_ts_lung_raw_no_dilate" not in str(LESION_ROOT):
        abort(f"lesion root 조건 불일치(roi0_0_ts_lung_raw_no_dilate 아님): {LESION_ROOT}")
    if "model_roi" in str(LESION_ROOT):
        abort(f"model_roi 경로 사용 금지: {LESION_ROOT}")
    if not LESION_ROOT.exists():
        abort(f"lesion root 경로 없음: {LESION_ROOT}")
    print(f"[G9] lesion root roi_0_0 조건 확인 ✅")

    # G10: lesion manifest 존재 및 join key=patient_id 확인
    if not LESION_MANIFEST.exists():
        abort(f"lesion manifest 없음: {LESION_MANIFEST}")
    man_ids = {r[JOIN_KEY].strip() for r in csv.DictReader(open(LESION_MANIFEST, encoding="utf-8-sig"))}
    missing_manifest = [pid for pid in dev_ids if pid not in man_ids]
    if missing_manifest:
        abort(f"manifest에 없는 stage1_dev 환자 {len(missing_manifest)}명: {missing_manifest[:5]}")
    print(f"[G10] join key=patient_id, manifest에 154명 전원 존재 확인 ✅")

    # G11: EfficientNet-B0 weight 존재 확인 (재다운로드 금지)
    import torch
    from torchvision.models import EfficientNet_B0_Weights
    wname = EfficientNet_B0_Weights.IMAGENET1K_V1.url.rsplit("/", 1)[-1]
    wpath = Path(torch.hub.get_dir()) / "checkpoints" / wname
    if not wpath.exists():
        abort(f"EfficientNet-B0 weight 없음(재다운로드 금지): {wpath}")
    print(f"[G11] EfficientNet-B0 weight 존재: {wpath}")

    # G12: 출력 경로 기존 결과 없음 확인 (덮어쓰기 금지)
    if REPORT_MD.exists() or REPORT_JSON.exists():
        abort(f"기존 P-A75 보고서 존재 → 덮어쓰기 금지: {REPORT_DIR}")
    if SCORING_SUMMARY_JSON.exists():
        abort(f"기존 scoring summary 존재 → 덮어쓰기 금지: {SCORING_SUMMARY_JSON}")
    existing_csvs = list(SCORE_DIR.glob("*.csv")) if SCORE_DIR.exists() else []
    if len(existing_csvs) == EXPECTED_N:
        abort(f"이미 {EXPECTED_N}개 score CSV 전부 존재 → full 완료 상태 (재실행 금지)")
    print(f"[G12] 출력 경로 기존 결과 없음 (기존 score CSV: {len(existing_csvs)}개, resume 가능) ✅")

    return idx, p95, p99, th, groups, dev_ids, dev_rows, wpath, thresh_mtime_before


def _padim_col_index_cached(score_path: Path, cache: dict) -> int:
    if "col" not in cache:
        with open(score_path, encoding="utf-8-sig") as f:
            header = f.readline().rstrip("\r\n").lstrip("﻿").split(",")
        cache["col"] = header.index("padim_score")
    return cache["col"]


def _to_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return float("nan")


def main() -> None:
    idx, p95, p99, th, groups, target, dev_rows, wpath, thresh_mtime_before = run_guards()

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[P-A75] 모든 가드 통과. lesion stage1_dev scoring 시작.")
    print(f"[P-A75] backbone={BACKBONE}, mask={MASK_TYPE}, p95={p95:.6f}, p99={p99:.6f}")
    print(f"[P-A75] lesion root: {LESION_ROOT}")
    print(f"[P-A75] score 저장: {SCORE_DIR}")

    from position_aware_padim.config_manager import ConfigManager
    from position_aware_padim.data_loader import DataLoader
    from position_aware_padim.path_resolver import PathResolver
    from position_aware_padim.padim_model import PaDiMModel
    from position_aware_padim.feature_extractor_effnet_b0_scaffold import FeatureExtractorEffNetB0

    model = PaDiMModel(
        selected_feature_indices_path=str(SELECTED_INDICES),
        feature_dim=REDUCED_FEATURE_DIM,
        eps=1e-5,
    )
    model.load(str(MODEL_NPZ))
    print(f"[P-A75] PaDiM 모델 로드 완료: position_bin 수={len(model.stats)}")

    feat = FeatureExtractorEffNetB0()
    print(f"[P-A75] device: {feat.device}")

    path_resolver = PathResolver(str(LESION_MANIFEST), str(LESION_ROOT))
    loader = DataLoader(
        str(LESION_MANIFEST),
        path_resolver,
        str(ERROR_CSV),
        use_mmap=True,
    )

    start_time = time.time()
    n_scored = n_skipped = n_failed = 0
    failed_patients: list[str] = []

    for i, pid in enumerate(target, 1):
        score_path = SCORE_DIR / f"{pid}.csv"
        if score_path.exists():
            n_skipped += 1
            print(f"  [SKIP] ({i}/{EXPECTED_N}) {pid}: 이미 존재 (resume)")
            continue
        data = loader.load_patient_data(pid, mask_type=MASK_TYPE)
        if data is None:
            n_failed += 1
            failed_patients.append(pid)
            print(f"  [FAIL] ({i}/{EXPECTED_N}) {pid}: 로드 실패 (error.csv 기록됨)")
            continue
        try:
            scored_df = model.score_patient(feat, data)
        except Exception as exc:
            n_failed += 1
            failed_patients.append(pid)
            record_error(pid, "score_error", str(exc), "padim_model.score_patient")
            print(f"  [FAIL] ({i}/{EXPECTED_N}) {pid}: 스코어링 오류 — {exc}")
            continue
        try:
            scored_df.to_csv(score_path, index=False, encoding="utf-8-sig")
            n_scored += 1
            print(f"  [OK]   ({i}/{EXPECTED_N}) {pid}: {len(scored_df)}개 patch 저장")
        except Exception as exc:
            n_failed += 1
            failed_patients.append(pid)
            record_error(pid, "save_error", str(exc), str(score_path))
            print(f"  [FAIL] ({i}/{EXPECTED_N}) {pid}: 저장 오류 — {exc}")

    elapsed = time.time() - start_time

    # ---- G13: threshold JSON mtime 불변 확인 ----
    thresh_mtime_after = os.path.getmtime(THRESH_JSON)
    thresh_unchanged = abs(thresh_mtime_before - thresh_mtime_after) < 1.0
    if not thresh_unchanged:
        abort(f"threshold JSON mtime이 변경됨! before={thresh_mtime_before}, after={thresh_mtime_after}")
    print(f"[G13] threshold JSON mtime 불변 확인 ✅")

    # ---- streaming 집계 (전체 메모리 적재 안 함) ----
    n_patch_total = 0
    n_nan = n_inf = 0
    n_finite = 0
    s_sum = 0.0
    s_sumsq = 0.0
    s_min = math.inf
    s_max = -math.inf
    n_over_p95 = 0
    n_over_p99 = 0
    n_csv = 0
    col_cache: dict = {}

    for pid in target:
        score_path = SCORE_DIR / f"{pid}.csv"
        if not score_path.exists():
            continue
        n_csv += 1
        col = _padim_col_index_cached(score_path, col_cache)
        arr = np.loadtxt(score_path, delimiter=",", skiprows=1,
                         usecols=col, dtype=str, encoding="utf-8-sig")
        arr = np.atleast_1d(arr)
        vals = np.array([_to_float(x) for x in arr], dtype=np.float64)
        n_patch_total += vals.size
        nan_mask = np.isnan(vals)
        inf_mask = np.isinf(vals)
        n_nan += int(nan_mask.sum())
        n_inf += int(inf_mask.sum())
        finite = vals[~(nan_mask | inf_mask)]
        if finite.size:
            n_finite += finite.size
            s_sum += float(finite.sum())
            s_sumsq += float((finite ** 2).sum())
            s_min = min(s_min, float(finite.min()))
            s_max = max(s_max, float(finite.max()))
            n_over_p95 += int((finite > p95).sum())
            n_over_p99 += int((finite > p99).sum())

    if n_finite:
        mean_score = s_sum / n_finite
        var = max(s_sumsq / n_finite - mean_score ** 2, 0.0)
        std_score = math.sqrt(var)
    else:
        mean_score = std_score = float("nan")
        s_min = s_max = float("nan")

    ratio_p95 = (n_over_p95 / n_finite) if n_finite else float("nan")
    ratio_p99 = (n_over_p99 / n_finite) if n_finite else float("nan")

    pid2group = {r[JOIN_KEY]: r["group"] for r in dev_rows}
    processed_groups = dict(Counter(
        pid2group[pid] for pid in target if (SCORE_DIR / f"{pid}.csv").exists()
    ))

    dist_sha = sha256_of(MODEL_NPZ)

    done = n_scored + n_skipped
    if done == EXPECTED_N and n_failed == 0:
        next_step = ("가능: stage1_dev 154명 전원 scoring 완료, 실패 0. "
                     "사용자 승인 시 P-A75.5 score artifact validation 진행 가능.")
    else:
        next_step = (f"보류: 완료 {done}/{EXPECTED_N}, 실패 {n_failed}. "
                     "실패 원인 확인 후 재판정 필요. P-A75.5 진행 전 사용자 승인 필요.")

    verdict = "통과" if n_failed == 0 and n_nan == 0 and n_inf == 0 else (
        "부분통과" if n_csv > 0 else "실패"
    )

    ts = datetime.now().isoformat(timespec="seconds")

    summary = {
        "stage": "P-A75_lesion_stage1_dev_scoring_efficientnet_b0_imagenet",
        "created": ts,
        "verdict": verdict,
        "backbone": BACKBONE,
        "pretrain_source": PRETRAIN_SOURCE,
        "run_tag": RUN_TAG,
        "scoring_backend": f"GPU ({feat.device}) — PaDiMModel.score_patient",
        "target_stage": EXPECTED_STAGE,
        "n_patients_target_total": EXPECTED_N,
        "n_patients_this_run": len(target),
        "n_patients_scored": n_scored,
        "n_patients_skipped_resume": n_skipped,
        "n_patients_failed": n_failed,
        "failed_patients": failed_patients,
        "group_target": EXPECTED_GROUPS,
        "group_processed": processed_groups,
        "n_stage2_holdout_in_target": 0,
        "join_key": JOIN_KEY,
        "model_roi_v1_used": False,
        "split_csv": str(LESION_SPLIT),
        "lesion_manifest": str(LESION_MANIFEST),
        "lesion_root": str(LESION_ROOT),
        "distribution_npz": str(MODEL_NPZ),
        "distribution_sha256": dist_sha,
        "selected_index_path": str(SELECTED_INDICES),
        "selected_index_shape": list(idx.shape),
        "selected_index_unique": int(len(set(idx.tolist()))),
        "selected_index_min": int(idx.min()),
        "selected_index_max": int(idx.max()),
        "threshold_json": str(THRESH_JSON),
        "threshold_p95": p95,
        "threshold_p99": p99,
        "threshold_recomputed": False,
        "threshold_json_mtime_unchanged": thresh_unchanged,
        "weight_file": str(wpath),
        "mask_type": MASK_TYPE,
        "n_score_csv": n_csv,
        "n_patch_total": n_patch_total,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "score_min": s_min,
        "score_max": s_max,
        "score_mean": mean_score,
        "score_std": std_score,
        "n_over_p95": n_over_p95,
        "ratio_over_p95": ratio_p95,
        "n_over_p99": n_over_p99,
        "ratio_over_p99": ratio_p99,
        "elapsed_seconds": round(elapsed, 2),
        "all_outputs_inside_workspace": True,
        "stage2_holdout_accessed": False,
        "full_308_scoring": False,
        "metrics_computed": False,
        "auroc_auprc_dice_recall_computed": False,
        "normal_val_test_rerun": False,
        "existing_results_modified": False,
        "next_step_p_a75_5": next_step,
    }

    # ---- scoring summary ----
    with open(SCORING_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(SCORING_SUMMARY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["metric", "value"])
        for k, v in [
            ("verdict", verdict),
            ("n_patients_scored", n_scored),
            ("n_patients_skipped_resume", n_skipped),
            ("n_patients_failed", n_failed),
            ("n_patch_total", n_patch_total),
            ("n_nan", n_nan),
            ("n_inf", n_inf),
            ("score_min", s_min),
            ("score_max", s_max),
            ("score_mean", mean_score),
            ("score_std", std_score),
            ("threshold_p95", p95),
            ("threshold_p99", p99),
            ("n_over_p95", n_over_p95),
            ("ratio_over_p95", ratio_p95),
            ("n_over_p99", n_over_p99),
            ("ratio_over_p99", ratio_p99),
        ]:
            wtr.writerow([k, v])

    # ---- 보고서 MD ----
    _write_md(REPORT_MD, summary)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- runtime summary ----
    with open(RUNTIME_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["stage", "n_patients", "n_failed", "n_patch_total",
                      "elapsed_sec", "p95_threshold", "p99_threshold",
                      "n_over_p95", "ratio_over_p95",
                      "n_over_p99", "ratio_over_p99",
                      "verdict", "created"])
        wtr.writerow([
            "P-A75", n_csv, n_failed, n_patch_total,
            round(elapsed, 1), round(p95, 6), round(p99, 6),
            n_over_p95, round(ratio_p95, 6) if not math.isnan(ratio_p95) else "nan",
            n_over_p99, round(ratio_p99, 6) if not math.isnan(ratio_p99) else "nan",
            verdict, ts,
        ])

    print()
    print(f"[P-A75] 완료: scored={n_scored}, skip={n_skipped}, fail={n_failed}, {elapsed:.1f}s")
    print(f"[P-A75] 전체 patch={n_patch_total:,}, nan={n_nan}, inf={n_inf}")
    if n_finite:
        print(f"[P-A75] score min/max/mean/std: {s_min:.4f}/{s_max:.4f}/{mean_score:.4f}/{std_score:.4f}")
    print(f"[P-A75] p95 초과: {n_over_p95:,} ({ratio_p95:.4%}), p99 초과: {n_over_p99:,} ({ratio_p99:.4%})")
    print(f"[P-A75] 판정: {verdict}")
    print(f"[P-A75] 보고서: {REPORT_MD}")
    print(f"\n=== P-A75 완료: {verdict} ===")


def _write_md(path: Path, s: dict) -> None:
    L: list[str] = []
    L.append("# P-A75 lesion stage1_dev scoring 보고서 (EfficientNet-B0 ImageNet)\n")
    L.append(f"## 판정: {s['verdict']}\n")
    L.append(f"- 생성: {s['created']}")
    L.append(f"- backbone: {s['backbone']} ({s['pretrain_source']})")
    L.append(f"- scoring backend: {s['scoring_backend']}\n")
    L.append("## 대상")
    L.append(f"- target stage: {s['target_stage']}")
    L.append(f"- 처리 환자 수: {s['n_patients_this_run']} / 전체 {s['n_patients_target_total']}")
    L.append(f"- NSCLC 125 / MSD_Lung 29 확인: group_target={s['group_target']}")
    L.append(f"- group_processed: {s['group_processed']}")
    L.append(f"- stage2_holdout 혼입 수: **{s['n_stage2_holdout_in_target']}**")
    L.append(f"- join key: **{s['join_key']}**")
    L.append(f"- model_roi_v1 사용: **{s['model_roi_v1_used']}**")
    L.append(f"- scored={s['n_patients_scored']}, skip(resume)={s['n_patients_skipped_resume']}, fail={s['n_patients_failed']}")
    L.append(f"- 실패 환자: {s['failed_patients']}\n")
    L.append("## 입력")
    L.append(f"- split CSV: `{s['split_csv']}`")
    L.append(f"- lesion manifest: `{s['lesion_manifest']}`")
    L.append(f"- lesion root: `{s['lesion_root']}`")
    L.append(f"- mask_type: {s['mask_type']}")
    L.append(f"- distribution sha256: `{s['distribution_sha256']}`")
    L.append(f"- selected index shape={s['selected_index_shape']} unique={s['selected_index_unique']}")
    L.append(f"- threshold p95={s['threshold_p95']} / p99={s['threshold_p99']} (재계산: {s['threshold_recomputed']})")
    L.append(f"- threshold JSON mtime 불변: **{s['threshold_json_mtime_unchanged']}**\n")
    L.append("## score 요약 (metrics 아님)")
    L.append(f"- 생성 score CSV: {s['n_score_csv']}개")
    L.append(f"- 전체 patch 수: {s['n_patch_total']:,}")
    L.append(f"- NaN: {s['n_nan']}, Inf: {s['n_inf']}")
    if isinstance(s["score_min"], float) and not math.isnan(s["score_min"]):
        L.append(f"- score min/max: {s['score_min']:.6f} / {s['score_max']:.6f}")
        L.append(f"- score mean/std: {s['score_mean']:.6f} / {s['score_std']:.6f}")
    L.append(f"- 사용한 p95 threshold: {s['threshold_p95']}")
    L.append(f"- 사용한 p99 threshold: {s['threshold_p99']}")
    L.append(f"- p95 초과 patch: {s['n_over_p95']:,} (비율 {s['ratio_over_p95']:.6f})")
    L.append(f"- p99 초과 patch: {s['n_over_p99']:,} (비율 {s['ratio_over_p99']:.6f})")
    L.append(f"- 소요 시간: {s['elapsed_seconds']}s\n")
    L.append("## 검증 체크")
    L.append(f"- P-A74 normal test sanity 통과 후 진행: ✅")
    L.append(f"- P-A73 threshold 재계산/수정 없음: ✅")
    L.append(f"- threshold JSON mtime 불변: **{s['threshold_json_mtime_unchanged']}**")
    L.append(f"- stage2_holdout 접근: {s['stage2_holdout_accessed']}")
    L.append(f"- 308명 전체 scoring: {s['full_308_scoring']}")
    L.append(f"- metrics 계산: {s['metrics_computed']}")
    L.append(f"- AUROC/AUPRC/Dice/recall 계산: {s['auroc_auprc_dice_recall_computed']}")
    L.append(f"- normal val/test 재실행: {s['normal_val_test_rerun']}")
    L.append(f"- 기존 결과 수정: {s['existing_results_modified']}\n")
    L.append(f"## 다음 단계(P-A75.5) 판정\n{s['next_step_p_a75_5']}")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
