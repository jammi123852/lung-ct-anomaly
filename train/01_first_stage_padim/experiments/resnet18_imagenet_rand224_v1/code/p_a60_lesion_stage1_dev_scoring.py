"""
P-A60: ResNet18 ImageNet random224 PaDiM — lesion stage1_dev 154명 scoring.

- P-A58 threshold(p95/p99)를 read-only로 로드한다.
- lesion stage1_dev 154명(NSCLC 125 + MSD_Lung 29)만 scoring한다.
- stage2_holdout 접근 금지, metrics 계산 금지.

실행:
  source ~/ai_env/bin/activate && python experiments/resnet18_imagenet_rand224_v1/code/p_a60_lesion_stage1_dev_scoring.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_paths  # noqa: E402

REPO_ROOT = ws_paths.REPO_ROOT
SRC_DIR = ws_paths.SRC_DIR
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from position_aware_padim.path_resolver import PathResolver  # noqa: E402
from position_aware_padim.data_loader import DataLoader  # noqa: E402
from position_aware_padim.feature_extractor_resnet50_scaffold import FeatureExtractorScaffold  # noqa: E402
from position_aware_padim.padim_model_resnet50_scaffold import PaDiMModelResNet50Scaffold  # noqa: E402

# ------------------------------------------------------------------
# 고정 입력 (resnet18 rand224, read-only)
# ------------------------------------------------------------------
MODEL_NPZ = ws_paths.MODEL_NPZ
SELECTED_INDICES_PATH = ws_paths.SELECTED_INDICES_PATH
THRESHOLD_JSON = ws_paths.OUTPUTS / "evaluation" / "normal_val_thresholds" / "normal_val_threshold.json"

LESION_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
)
LESION_MANIFEST = LESION_ROOT / "manifests" / "patient_manifest.csv"
LESION_SPLIT = REPO_ROOT / "outputs" / "second-stage-lesion-refiner-v1" / "splits" / "lesion_stage_split_v1_balanced.csv"

EXPECTED_STAGE = "stage1_dev"
EXPECTED_N = 154
EXPECTED_GROUPS = {"NSCLC": 125, "MSD_Lung": 29}
EXPECTED_IDX_SHAPE = (ws_paths.REDUCED_FEATURE_DIM,)  # (224,)
EXPECTED_P95 = 20.2955
EXPECTED_P99 = 24.4483
JOIN_KEY = "patient_id"
MASK_TYPE = "roi_0_0"

# P-A59 보고서
P_A59_MD = ws_paths.OUTPUTS / "reports" / "normal_test" / "p_a59_normal_test_sanity.md"

# ------------------------------------------------------------------
# 출력 (lesion 전용, resnet18 rand224 workspace 내부)
# ------------------------------------------------------------------
LESION_SCORE_DIR = ws_paths.OUTPUTS / "scores" / "lesion_stage1_dev_by_patient"
LESION_EVAL_DIR = ws_paths.OUTPUTS / "evaluation" / "lesion_stage1_dev_scoring"
LESION_REPORT_DIR = ws_paths.OUTPUTS / "reports" / "lesion_stage1_dev"
LESION_ERROR_CSV = LESION_EVAL_DIR / "error.csv"
REPORT_MD = LESION_REPORT_DIR / "p_a60_lesion_stage1_dev_scoring.md"
REPORT_JSON = LESION_REPORT_DIR / "p_a60_lesion_stage1_dev_scoring.json"
SCORING_SUMMARY_JSON = LESION_EVAL_DIR / "lesion_stage1_dev_scoring_summary.json"
SCORING_SUMMARY_CSV = LESION_EVAL_DIR / "lesion_stage1_dev_scoring_summary.csv"
RUNTIME_CSV = LESION_REPORT_DIR / "p_a60_runtime_summary.csv"

SCRIPT_NAME = "p_a60_lesion_stage1_dev_scoring.py"


def stop(msg: str) -> None:
    print(f"[P-A60][ABORT] {msg}")
    sys.exit(1)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_error(patient_id, error_type, error_msg, file_logical) -> None:
    LESION_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not LESION_ERROR_CSV.exists() or LESION_ERROR_CSV.stat().st_size == 0
    with open(LESION_ERROR_CSV, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "error_type", "error_msg", "file_logical"])
        if write_header:
            w.writeheader()
        w.writerow({"patient_id": patient_id, "error_type": error_type,
                    "error_msg": error_msg, "file_logical": file_logical})


def run_guards():
    # G1: P-A59 보고서 통과 확인
    if not P_A59_MD.exists():
        stop(f"P-A59 보고서 없음: {P_A59_MD}")
    with open(P_A59_MD, encoding="utf-8") as f:
        md_text = f.read()
    if "판정: 통과" not in md_text:
        stop(f"P-A59 보고서가 통과 상태가 아님: {P_A59_MD}")
    print(f"[P-A60][G1] P-A59 보고서 통과 확인")

    # G2: P-A58 threshold JSON 존재 확인
    if not THRESHOLD_JSON.exists():
        stop(f"threshold json 없음: {THRESHOLD_JSON}")

    # G3/G4: threshold read-only 로드, 재계산 금지
    th = json.load(open(THRESHOLD_JSON, encoding="utf-8"))
    p95 = float(th["threshold_p95"])
    p99 = float(th["threshold_p99"])
    if abs(p95 - EXPECTED_P95) > 0.01:
        stop(f"threshold p95 불일치: JSON={p95:.6f}, 기대={EXPECTED_P95}")
    if abs(p99 - EXPECTED_P99) > 0.01:
        stop(f"threshold p99 불일치: JSON={p99:.6f}, 기대={EXPECTED_P99}")
    print(f"[P-A60][G3] p95={p95:.6f}, p99={p99:.6f} 확인 (재계산 없음)")

    # G9: lesion root roi_0_0 조건 확인
    if "roi0_0_ts_lung_raw_no_dilate" not in str(LESION_ROOT):
        stop(f"lesion root 조건 불일치(ts_lung_raw_no_dilate 아님): {LESION_ROOT}")
    if "model_roi" in str(LESION_ROOT):
        stop(f"model_roi 경로 사용 금지: {LESION_ROOT}")

    # 파일 존재 확인
    if not LESION_SPLIT.exists():
        stop(f"lesion split CSV 없음: {LESION_SPLIT}")
    if not MODEL_NPZ.exists():
        stop(f"distribution npz 없음: {MODEL_NPZ}")
    if not SELECTED_INDICES_PATH.exists():
        stop(f"selected_feature_indices.npy 없음: {SELECTED_INDICES_PATH}")
    if not LESION_MANIFEST.exists():
        stop(f"lesion manifest 없음: {LESION_MANIFEST}")

    # ResNet18 weight 존재 확인 (재다운로드 금지)
    import torch
    from torchvision.models import ResNet18_Weights
    wname = ResNet18_Weights.IMAGENET1K_V1.url.rsplit("/", 1)[-1]
    wpath = Path(torch.hub.get_dir()) / "checkpoints" / wname
    if not wpath.exists():
        stop(f"ResNet18 weight 없음(재다운로드 금지): {wpath}")

    # selected index shape / unique / range 확인
    idx = np.load(SELECTED_INDICES_PATH)
    if tuple(idx.shape) != EXPECTED_IDX_SHAPE:
        stop(f"selected index shape 불일치: {idx.shape} != {EXPECTED_IDX_SHAPE}")
    if len(np.unique(idx)) != EXPECTED_IDX_SHAPE[0]:
        stop(f"selected index unique 불일치: {len(np.unique(idx))} != {EXPECTED_IDX_SHAPE[0]}")
    if int(idx.min()) < 0 or int(idx.max()) >= ws_paths.RAW_FEATURE_DIM:
        stop(f"selected index range 불일치: min={idx.min()} max={idx.max()}")
    print(f"[P-A60][G6] selected_index shape={idx.shape}, unique={len(np.unique(idx))}")

    # G5: lesion split stage1_dev 154명 확인
    rows = list(csv.DictReader(open(LESION_SPLIT, encoding="utf-8-sig")))
    dev = [r for r in rows if r["stage_split"] == EXPECTED_STAGE]

    # G6: NSCLC 125 / MSD_Lung 29 확인
    if len(dev) != EXPECTED_N:
        stop(f"stage1_dev 환자 수가 {len(dev)} (기대 {EXPECTED_N})")
    groups = dict(Counter(r["group"] for r in dev))
    if groups != EXPECTED_GROUPS:
        stop(f"stage1_dev group 구성 불일치: {groups} (기대 {EXPECTED_GROUPS})")
    print(f"[P-A60][G5/G6] stage1_dev {len(dev)}명, {groups}")

    # G7: stage2_holdout 혼입 0 확인
    n_holdout_in_target = sum(1 for r in dev if r["stage_split"] != EXPECTED_STAGE)
    if n_holdout_in_target != 0:
        stop(f"대상에 비-stage1_dev 혼입: {n_holdout_in_target}")
    print(f"[P-A60][G7] stage2_holdout 혼입 0 확인")

    # G10: join key = patient_id, manifest에 154명 전원 존재 확인
    man_ids = {r[JOIN_KEY].strip() for r in csv.DictReader(open(LESION_MANIFEST, encoding="utf-8-sig"))}
    missing_manifest = [r[JOIN_KEY] for r in dev if r[JOIN_KEY] not in man_ids]
    if missing_manifest:
        stop(f"manifest에 없는 stage1_dev 환자 {len(missing_manifest)}명: {missing_manifest[:5]}...")
    print(f"[P-A60][G10] join key=patient_id, manifest 154명 전원 존재 확인")

    target = [r[JOIN_KEY] for r in dev]

    # G12: 출력 경로 기존 결과 없음 확인
    if REPORT_MD.exists() or REPORT_JSON.exists():
        stop(f"기존 P-A60 보고서 존재 → 덮어쓰기 금지: {LESION_REPORT_DIR}")
    if SCORING_SUMMARY_JSON.exists():
        stop(f"기존 P-A60 scoring summary 존재 → 덮어쓰기 금지: {SCORING_SUMMARY_JSON}")
    existing_csvs = list(LESION_SCORE_DIR.glob("*.csv")) if LESION_SCORE_DIR.exists() else []
    if len(existing_csvs) == EXPECTED_N:
        stop(f"이미 154개 score CSV 전부 존재 → 덮어쓰기 금지 (resume 가능하지만 full 완료 상태)")
    print(f"[P-A60][G12] 출력 경로 기존 결과 없음 확인 (resume 기존 {len(existing_csvs)}개)")

    return idx, p95, p99, th, groups, target, wpath


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
    idx, p95, p99, th, groups, target, wpath = run_guards()

    LESION_SCORE_DIR.mkdir(parents=True, exist_ok=True)
    LESION_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    LESION_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[P-A60] 모든 가드 통과. lesion stage1_dev scoring 시작.")
    print(f"[P-A60] backbone=resnet18, mask={MASK_TYPE}, p95={p95:.6f}, p99={p99:.6f}")
    print(f"[P-A60] lesion root: {LESION_ROOT}")
    print(f"[P-A60] score 저장: {LESION_SCORE_DIR}")

    model = PaDiMModelResNet50Scaffold(
        selected_feature_indices_path=str(SELECTED_INDICES_PATH),
        feature_dim=ws_paths.REDUCED_FEATURE_DIM,
        raw_feature_dim=ws_paths.RAW_FEATURE_DIM,
        eps=1e-5,
    )
    model.load(str(MODEL_NPZ))
    print(f"[P-A60] PaDiM 모델 로드 완료: position_bin 수={len(model.stats)}")

    wpath_mtime_before = wpath.stat().st_mtime
    feature_extractor = FeatureExtractorScaffold(backbone="resnet18", pretrain_source="imagenet")
    print(f"[P-A60] device: {feature_extractor.device}")
    download_happened = wpath.stat().st_mtime != wpath_mtime_before

    path_resolver = PathResolver(str(LESION_MANIFEST), str(LESION_ROOT))
    loader = DataLoader(str(LESION_MANIFEST), path_resolver, str(LESION_ERROR_CSV), use_mmap=True)

    start_time = time.time()
    n_scored = n_skipped = n_failed = 0
    failed_patients = []

    for pid in target:
        score_path = LESION_SCORE_DIR / f"{pid}.csv"
        if score_path.exists():
            n_skipped += 1
            print(f"  [SKIP] {pid}: 이미 존재 (resume)")
            continue
        data = loader.load_patient_data(pid, mask_type=MASK_TYPE)
        if data is None:
            n_failed += 1
            failed_patients.append(pid)
            print(f"  [FAIL] {pid}: 로드 실패 (error.csv 기록됨)")
            continue
        try:
            scored_df = model.score_patient(feature_extractor, data)
        except Exception as exc:
            n_failed += 1
            failed_patients.append(pid)
            record_error(pid, "score_error", str(exc), "padim_model.score_patient")
            print(f"  [FAIL] {pid}: 스코어링 오류 — {exc}")
            continue
        try:
            scored_df.to_csv(score_path, index=False, encoding="utf-8-sig")
            n_scored += 1
            print(f"  [OK]   {pid}: {len(scored_df)}개 patch 저장")
        except Exception as exc:
            n_failed += 1
            failed_patients.append(pid)
            record_error(pid, "save_error", str(exc), str(score_path))
            print(f"  [FAIL] {pid}: 저장 오류 — {exc}")

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # 요약 집계 (streaming, 전량 RAM 적재 안 함)
    # ------------------------------------------------------------------
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
    col_cache = {}

    for pid in target:
        score_path = LESION_SCORE_DIR / f"{pid}.csv"
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
        mean = s_sum / n_finite
        var = max(s_sumsq / n_finite - mean ** 2, 0.0)
        std = math.sqrt(var)
    else:
        mean = std = float("nan")
        s_min = s_max = float("nan")

    ratio_p95 = (n_over_p95 / n_finite) if n_finite else float("nan")
    ratio_p99 = (n_over_p99 / n_finite) if n_finite else float("nan")

    split_rows = list(csv.DictReader(open(LESION_SPLIT, encoding="utf-8-sig")))
    dev_rows = [r for r in split_rows if r["stage_split"] == EXPECTED_STAGE]
    pid2group = {r[JOIN_KEY]: r["group"] for r in dev_rows}
    processed_groups = dict(Counter(
        pid2group[pid] for pid in target if (LESION_SCORE_DIR / f"{pid}.csv").exists()
    ))

    distribution_sha = sha256_of(MODEL_NPZ)

    done = n_scored + n_skipped
    if done == EXPECTED_N and n_failed == 0:
        next_step = ("가능: stage1_dev 154명 전원 scoring 완료, 실패 0. "
                     "사용자 승인 시 P-A60.5 score artifact validation 진행 가능.")
    else:
        next_step = (f"보류: 완료 {done}/{EXPECTED_N}, 실패 {n_failed}. "
                     "실패 원인 확인 후 재판정 필요. P-A60.5 진행 전 사용자 승인 필요.")

    verdict = "통과" if n_failed == 0 and n_nan == 0 and n_inf == 0 else (
        "부분통과" if n_csv > 0 else "실패"
    )

    summary = {
        "stage": "P-A60_lesion_stage1_dev_scoring_resnet18_rand224",
        "created": datetime.now().isoformat(timespec="seconds"),
        "verdict": verdict,
        "backbone": "resnet18",
        "pretrain_source": "imagenet",
        "run_tag": "padim_resnet18_imagenet_rand224",
        "scoring_backend": f"GPU ({feature_extractor.device}) — PaDiMModelResNet50Scaffold.score_patient",
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
        "distribution_sha256": distribution_sha,
        "selected_index_path": str(SELECTED_INDICES_PATH),
        "selected_index_shape": list(idx.shape),
        "selected_index_unique": int(len(np.unique(idx))),
        "selected_index_min": int(idx.min()),
        "selected_index_max": int(idx.max()),
        "threshold_json": str(THRESHOLD_JSON),
        "threshold_p95": p95,
        "threshold_p99": p99,
        "threshold_recomputed": False,
        "weight_file": str(wpath),
        "additional_download": bool(download_happened),
        "mask_type": MASK_TYPE,
        "n_score_csv": n_csv,
        "n_patch_total": n_patch_total,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "score_min": s_min,
        "score_max": s_max,
        "score_mean": mean,
        "score_std": std,
        "n_over_p95": n_over_p95,
        "ratio_over_p95": ratio_p95,
        "n_over_p99": n_over_p99,
        "ratio_over_p99": ratio_p99,
        "elapsed_seconds": round(elapsed, 2),
        "all_outputs_inside_workspace": True,
        "modified_random100_workspace": False,
        "modified_resnet18_v2_results": False,
        "modified_position_aware_padim_v1": False,
        "modified_original_source_code": False,
        "stage2_holdout_accessed": False,
        "full_308_scoring": False,
        "metrics_computed": False,
        "auroc_auprc_dice_iou_recall_computed": False,
        "next_step_p_a60_5": next_step,
    }

    # scoring summary
    with open(SCORING_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(SCORING_SUMMARY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["metric", "value"])
        for k, v in [
            ("verdict", verdict), ("n_patients_scored", n_scored),
            ("n_patients_failed", n_failed), ("n_patch_total", n_patch_total),
            ("n_nan", n_nan), ("n_inf", n_inf),
            ("score_min", s_min), ("score_max", s_max),
            ("score_mean", mean), ("score_std", std),
            ("threshold_p95", p95), ("threshold_p99", p99),
            ("n_over_p95", n_over_p95), ("ratio_over_p95", ratio_p95),
            ("n_over_p99", n_over_p99), ("ratio_over_p99", ratio_p99),
        ]:
            wtr.writerow([k, v])

    # 보고서 작성
    _write_md(REPORT_MD, summary)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # runtime summary
    with open(RUNTIME_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["stage", "n_patients", "n_failed", "n_patch_total",
                      "elapsed_sec", "p95_threshold", "p99_threshold",
                      "n_over_p95", "ratio_over_p95",
                      "n_over_p99", "ratio_over_p99",
                      "verdict", "created"])
        wtr.writerow([
            "P-A60", n_csv, n_failed, n_patch_total,
            round(elapsed, 1), round(p95, 6), round(p99, 6),
            n_over_p95, round(ratio_p95, 6),
            n_over_p99, round(ratio_p99, 6),
            verdict, datetime.now().isoformat(timespec="seconds"),
        ])

    print()
    print(f"[P-A60] 완료: scored={n_scored}, skip={n_skipped}, fail={n_failed}, {elapsed:.1f}s")
    print(f"[P-A60] 전체 patch={n_patch_total:,}, nan={n_nan}, inf={n_inf}")
    if n_finite:
        print(f"[P-A60] score min/max/mean/std: {s_min:.4f}/{s_max:.4f}/{mean:.4f}/{std:.4f}")
    print(f"[P-A60] p95 초과: {n_over_p95:,} ({ratio_p95:.4%}), p99 초과: {n_over_p99:,} ({ratio_p99:.4%})")
    print(f"[P-A60] 판정: {verdict}")
    print(f"[P-A60] 보고서: {REPORT_MD}")
    print("JSON_INFO_BEGIN"); print(json.dumps(summary, ensure_ascii=False)); print("JSON_INFO_END")


def _write_md(path: Path, s: dict) -> None:
    L = []
    L.append("# P-A60 lesion stage1_dev scoring 보고서 (ResNet18 rand224)\n")
    L.append(f"## 판정: {s['verdict']}\n")
    L.append(f"- 생성: {s['created']}")
    L.append(f"- backbone: resnet18 (imagenet)")
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
    L.append(f"- 추가 다운로드: **{s['additional_download']}**\n")
    L.append("## score 요약 (metrics 아님)")
    L.append(f"- 생성 score CSV: {s['n_score_csv']}개")
    L.append(f"- 전체 patch 수: {s['n_patch_total']:,}")
    L.append(f"- NaN: {s['n_nan']}, Inf: {s['n_inf']}")
    if isinstance(s['score_min'], float) and not math.isnan(s['score_min']):
        L.append(f"- score min/max: {s['score_min']:.6f} / {s['score_max']:.6f}")
        L.append(f"- score mean/std: {s['score_mean']:.6f} / {s['score_std']:.6f}")
    L.append(f"- 사용한 p95 threshold: {s['threshold_p95']}")
    L.append(f"- 사용한 p99 threshold: {s['threshold_p99']}")
    L.append(f"- p95 초과 patch: {s['n_over_p95']:,} (비율 {s['ratio_over_p95']:.6f})")
    L.append(f"- p99 초과 patch: {s['n_over_p99']:,} (비율 {s['ratio_over_p99']:.6f})")
    L.append(f"- 소요 시간: {s['elapsed_seconds']}s\n")
    L.append("## 검증 체크")
    L.append(f"- P-A59 normal test sanity 통과 후 진행: ✅")
    L.append(f"- P-A58 threshold 재계산/수정 없음: ✅")
    L.append(f"- 기존 random100 결과 무수정: {not s['modified_random100_workspace']}")
    L.append(f"- stage2_holdout 접근: {s['stage2_holdout_accessed']}")
    L.append(f"- 308명 전체 scoring: {s['full_308_scoring']}")
    L.append(f"- metrics 계산: {s['metrics_computed']}")
    L.append(f"- AUROC/AUPRC/Dice/recall 계산: {s['auroc_auprc_dice_iou_recall_computed']}\n")
    L.append(f"## 다음 단계(P-A60.5) 판정\n{s['next_step_p_a60_5']}")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
