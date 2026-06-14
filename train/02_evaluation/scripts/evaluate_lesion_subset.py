"""
evaluate_lesion_subset.py: Phase 8 병변 평가 스크립트 (smoke test 단계).

- configs/paths.local.yaml의 nsclc_msd_usable_only 병변 데이터셋을 평가한다.
- 정상 학습 모델(position_bin_stats.npz)을 그대로 load하여 patch PaDiM score를 계산한다.
- patch CSV의 lesion 컬럼으로 patch_label을 생성하고 Evaluator로 지표를 계산한다.

설계 기준 (정상 데이터와 분리):
- 병변 score 출력: outputs/.../scores/padim_v1/lesion_by_patient/{patient_id}.csv
- 평가 결과 출력: outputs/.../evaluation/lesion_subset/{output_tag}_metrics.csv
                  outputs/.../evaluation/lesion_subset/{output_tag}_summary.json

안전 가드:
- --limit 또는 --dry-run 없이 실행하면 중단 (전체 308명 자동 실행 차단).
- 이번 단계는 smoke test이며, 병변 성능 결론을 내리지 않는다.

정상 데이터용 PathResolver / DataLoader / PaDiMModel 동작은 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 프로젝트 루트를 sys.path에 추가 (src 하위 패키지 import용)
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from position_aware_padim.path_resolver import PathResolver
from position_aware_padim.data_loader import DataLoader
from position_aware_padim.feature_extractor import FeatureExtractor
from position_aware_padim.padim_model import PaDiMModel
from position_aware_padim.evaluator import Evaluator
from position_aware_padim.patient_splitter import PatientSplitter

REPORTS_DIR = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "reports"
ERROR_CSV = REPORTS_DIR / "error.csv"
RUNTIME_CSV = REPORTS_DIR / "runtime_summary.csv"
MODEL_NPZ = (
    REPO_ROOT / "outputs" / "position-aware-padim-v1"
    / "models" / "padim_v1" / "distributions" / "position_bin_stats.npz"
)
# 정상 score 출력 (threshold 계산용, read-only)
NORMAL_SCORE_DIR = (
    REPO_ROOT / "outputs" / "position-aware-padim-v1"
    / "scores" / "padim_v1" / "by_patient"
)
# 병변 score 출력 — v1 기본 경로 (v2는 런타임에 결정)
LESION_SCORE_DIR_V1 = (
    REPO_ROOT / "outputs" / "position-aware-padim-v1"
    / "scores" / "padim_v1" / "lesion_by_patient"
)
LESION_SCORE_DIR_V2 = (
    REPO_ROOT / "outputs" / "position-aware-padim-v1"
    / "scores" / "padim_v1" / "lesion_v2_by_patient"
)
# 평가 결과 출력 — v1 기본 경로 (v2는 런타임에 결정)
EVAL_DIR_V1 = (
    REPO_ROOT / "outputs" / "position-aware-padim-v1"
    / "evaluation" / "lesion_subset"
)
EVAL_DIR_V2 = (
    REPO_ROOT / "outputs" / "position-aware-padim-v1"
    / "evaluation" / "lesion_subset_v2"
)

ERROR_COLUMNS = ["patient_id", "error_type", "error_msg", "file_logical"]
RUNTIME_COLUMNS = ["timestamp", "script", "metric", "value"]
SCRIPT_NAME = "evaluate_lesion_subset.py"

# 지표 계산용 최소 컬럼 (전체 308명 평가 시 메모리 누적 방지).
# score CSV 저장은 원본 전체 컬럼을 유지하고, 메모리에 쌓는 집계용으로만 이 컬럼들을 사용한다.
METRIC_COLS = ["patient_id", "local_z", "padim_score", "patch_label"]


def record_error(patient_id: str, error_type: str, error_msg: str, file_logical: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not ERROR_CSV.exists() or ERROR_CSV.stat().st_size == 0
    with open(ERROR_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "patient_id": patient_id, "error_type": error_type,
            "error_msg": error_msg, "file_logical": file_logical,
        })


def record_runtime_rows(rows) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_CSV.exists() or RUNTIME_CSV.stat().st_size == 0
    with open(RUNTIME_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def load_lesion_manifest_rows(manifest_path: str):
    """병변 manifest를 utf-8-sig로 읽어 (patient_id, safe_id, group) 행 목록 반환."""
    rows = []
    with open(manifest_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("patient_id") or "").strip()
            if not pid:
                continue
            rows.append({
                "patient_id": pid,
                "safe_id": (row.get("safe_id") or "").strip(),
                "group": (row.get("group") or "").strip(),
            })
    return rows


def make_patch_label(df: pd.DataFrame, label_mode: str) -> np.ndarray:
    """patch CSV lesion 컬럼 기반 patch_label(0/1) 생성.

    any_pixel : has_lesion_patch==1 또는 lesion_pixels>0
    ratio_001 : lesion_patch_ratio>=0.01
    """
    if label_mode == "ratio_001":
        if "lesion_patch_ratio" not in df.columns:
            raise ValueError("label_mode=ratio_001인데 lesion_patch_ratio 컬럼이 없습니다.")
        return (df["lesion_patch_ratio"].astype(float) >= 0.01).astype(int).values

    # 기본: any_pixel
    has_col = df["has_lesion_patch"].astype(float) > 0 if "has_lesion_patch" in df.columns else False
    pix_col = df["lesion_pixels"].astype(float) > 0 if "lesion_pixels" in df.columns else False
    if has_col is False and pix_col is False:
        raise ValueError("any_pixel 모드인데 has_lesion_patch / lesion_pixels 컬럼이 모두 없습니다.")
    combined = np.zeros(len(df), dtype=int)
    if has_col is not False:
        combined = combined | has_col.astype(int).values
    if pix_col is not False:
        combined = combined | pix_col.astype(int).values
    return combined


def compute_threshold(threshold_mode: str, lesion_scores: np.ndarray, threshold_json: str | None = None):
    """threshold-mode에 따라 patch score threshold(float)를 계산한다.

    Returns
    -------
    (threshold, info) : (float | None, dict)
        threshold가 None이면 적용 안 함(Dice/IoU 생략).
        info에는 source / 사용 환자 수 / 상태(확인 필요 등)를 담는다.

    threshold_mode
    --------------
    none                  : None 반환 (적용 안 함)
    normal_val_p95/p99    : 정상 val 환자 by_patient score의 95/99 percentile.
                            threshold_json이 있으면 JSON에서 직접 읽음 (v2 전용).
    lesion_score_p95_debug: 이번 lesion 대상 score 내부 95 percentile (디버그 전용, 성능보고 금지)
    """
    info: dict = {"threshold_mode": threshold_mode}

    if threshold_mode == "none":
        info["status"] = "not_applied"
        return None, info

    if threshold_mode == "lesion_score_p95_debug":
        if lesion_scores is None or len(lesion_scores) == 0:
            info["status"] = "no_lesion_scores"
            return None, info
        thr = float(np.percentile(lesion_scores, 95))
        info["status"] = "applied_debug_only"
        info["source"] = "lesion_subset_internal_p95"
        info["n_scores"] = int(len(lesion_scores))
        info["warning"] = "디버그 전용 threshold. 성능 보고에 사용 금지."
        return thr, info

    if threshold_mode in ("normal_val_p95", "normal_val_p99"):
        pct = 95 if threshold_mode.endswith("p95") else 99

        # threshold_json이 있으면 JSON에서 직접 읽기 (v2 전용, NORMAL_SCORE_DIR 불필요)
        if threshold_json is not None:
            try:
                with open(threshold_json, encoding="utf-8") as f:
                    data = json.load(f)
                thr = float(data[f"threshold_p{pct}"])
                info["status"] = "applied"
                info["source"] = f"threshold_json_p{pct}"
                info["threshold_json"] = threshold_json
                return thr, info
            except Exception as exc:
                info["status"] = "확인 필요"
                info["reason"] = f"threshold_json 읽기 실패: {exc}"
                return None, info

        # 기존 방식: 정상 val 환자 목록에서 직접 계산 (v1)
        try:
            splitter = PatientSplitter(str(REPO_ROOT))
            split = splitter.load_split()
            val_pids = list(split.val)
        except Exception as exc:
            info["status"] = "확인 필요"
            info["reason"] = f"val 구분 실패: {exc}"
            return None, info
        if not val_pids:
            info["status"] = "확인 필요"
            info["reason"] = "val 환자 목록이 비어 있음"
            return None, info

        # val 환자 by_patient score 수집 (read-only)
        scores_list = []
        n_used = 0
        n_missing = 0
        for pid in val_pids:
            p = NORMAL_SCORE_DIR / f"{pid}.csv"
            if not p.exists():
                n_missing += 1
                continue
            try:
                col = pd.read_csv(p, encoding="utf-8-sig", usecols=["padim_score"])["padim_score"]
            except Exception:
                n_missing += 1
                continue
            col = col[~col.isna()]
            if len(col) > 0:
                scores_list.append(col.values)
                n_used += 1
        if not scores_list:
            info["status"] = "확인 필요"
            info["reason"] = "val score를 하나도 수집하지 못함"
            return None, info
        all_val = np.concatenate(scores_list)
        thr = float(np.percentile(all_val, pct))
        info["status"] = "applied"
        info["source"] = f"normal_val_p{pct}"
        info["n_val_patients_used"] = n_used
        info["n_val_patients_missing"] = n_missing
        info["n_val_scores"] = int(len(all_val))
        return thr, info

    info["status"] = "unknown_mode"
    return None, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 병변 평가 (smoke test 단계)")
    parser.add_argument("--limit", type=int, default=None,
                        help="처리할 최대 환자 수 (smoke test용). 전체 실행은 이번 단계에서 금지.")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="score 없이 데이터 로드/경로 확인만 수행")
    parser.add_argument("--output-tag", type=str, default="smoke_test",
                        help="출력 파일 접두사 (기본: smoke_test). 전체 실행 시 evaluation 등으로 구분")
    parser.add_argument("--threshold-mode", type=str, default="none",
                        choices=["none", "normal_val_p95", "normal_val_p99",
                                 "lesion_score_p95_debug"],
                        help="patch_dice/patch_iou용 threshold 모드. "
                             "none=미적용, normal_val_p95/p99=정상 val score percentile, "
                             "lesion_score_p95_debug=lesion 내부 percentile(디버그 전용)")
    parser.add_argument("--label-mode", type=str, default="any_pixel",
                        choices=["any_pixel", "ratio_001"],
                        help="patch_label 생성 기준 (기본: any_pixel)")
    parser.add_argument(
        "--dataset-profile",
        type=str,
        default="v1_model_roi",
        choices=["v1_model_roi", "v2_roi_0_0"],
        help=(
            "평가할 데이터셋 profile. "
            "v1_model_roi: 기존 model_roi 기반 데이터셋 (기본값). "
            "v2_roi_0_0: roi_0_0 기반 신규 데이터셋 (출력 경로 자동 분리)."
        ),
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        default=False,
        help=(
            "환자별 score CSV 생성까지만 수행하고 patch/slice/patient 지표 계산을 생략한다. "
            "전체 308명 scoring 후 compute_lesion_metrics_fast.py로 지표를 별도 계산할 때 사용."
        ),
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default=None,
        help="v2 model stats 경로 오버라이드 (기본: padim_v1/distributions/position_bin_stats.npz)",
    )
    parser.add_argument(
        "--score-dir",
        type=str,
        default=None,
        help="lesion score 출력 경로 오버라이드 (기본: dataset-profile 기반 자동 결정)",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=str,
        default=None,
        help="evaluation 출력 경로 오버라이드 (기본: dataset-profile 기반 자동 결정)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="reports 경로 오버라이드 (기본: reports)",
    )
    parser.add_argument(
        "--threshold-json",
        type=str,
        default=None,
        help=(
            "normal threshold JSON 경로 (예: evaluation/normal_v2_roi0_0/normal_v2_threshold.json). "
            "--threshold-mode normal_val_p95/p99 시 이 파일에서 threshold를 읽어 NORMAL_SCORE_DIR 대신 사용."
        ),
    )
    args = parser.parse_args()
    is_v2 = (args.dataset_profile == "v2_roi_0_0")

    # --- CLI 오버라이드: v2 model stats / reports 경로 ---
    global MODEL_NPZ, REPORTS_DIR, ERROR_CSV, RUNTIME_CSV
    if args.stats_path is not None:
        MODEL_NPZ = REPO_ROOT / args.stats_path
    if args.reports_dir is not None:
        REPORTS_DIR = REPO_ROOT / args.reports_dir
        ERROR_CSV = REPORTS_DIR / "error.csv"
        RUNTIME_CSV = REPORTS_DIR / "runtime_summary.csv"

    # --- 안전 가드: --limit 또는 --dry-run 필수 ---
    if args.limit is None and not args.dry_run:
        print(
            "[ERROR] 안전을 위해 --limit N 또는 --dry-run 중 하나를 명시해야 합니다.\n"
            "        이번 단계에서 전체 308명 평가는 금지입니다.\n"
            "예: python scripts/evaluate_lesion_subset.py --limit 3 --output-tag smoke_test --label-mode any_pixel"
        )
        sys.exit(1)

    start_time = time.time()

    # --- profile별 경로 결정 ---
    LESION_SCORE_DIR = LESION_SCORE_DIR_V2 if is_v2 else LESION_SCORE_DIR_V1
    EVAL_DIR = EVAL_DIR_V2 if is_v2 else EVAL_DIR_V1
    mask_type = "roi_0_0" if is_v2 else "model_roi"

    # CLI 오버라이드: --score-dir / --evaluation-dir
    if args.score_dir is not None:
        LESION_SCORE_DIR = REPO_ROOT / args.score_dir
    if args.evaluation_dir is not None:
        EVAL_DIR = REPO_ROOT / args.evaluation_dir

    # --- config 읽기 ---
    cfg_path = REPO_ROOT / "configs" / "paths.local.yaml"
    with open(cfg_path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f) or {}
    cfg_key = "nsclc_msd_usable_only_v2" if is_v2 else "nsclc_msd_usable_only"
    base = (cfg.get(cfg_key) or "").strip()
    if not base:
        print(f"[ERROR] configs/paths.local.yaml의 {cfg_key}가 비어 있습니다.")
        sys.exit(1)
    base_path = Path(base)
    if not base_path.exists():
        print(f"[ERROR] 병변 데이터 경로가 존재하지 않습니다: {base_path}")
        sys.exit(1)
    manifest_path = base_path / "manifests" / "patient_manifest.csv"
    if not manifest_path.exists():
        print(f"[ERROR] patient_manifest.csv 없음: {manifest_path}")
        sys.exit(1)

    # --- 모델 파일 확인 ---
    if not MODEL_NPZ.exists():
        print(f"[ERROR] PaDiM 분포 파일 없음: {MODEL_NPZ}")
        sys.exit(1)

    # --- 환자 목록 ---
    manifest_rows = load_lesion_manifest_rows(str(manifest_path))
    n_total = len(manifest_rows)
    target_rows = manifest_rows[: args.limit] if args.limit is not None else manifest_rows
    target_pids = [r["patient_id"] for r in target_rows]

    print(f"[evaluate_lesion_subset] dataset_profile = {args.dataset_profile}")
    print(f"[evaluate_lesion_subset] base_path = {base_path}")
    print(f"[evaluate_lesion_subset] mask_type = {mask_type}")
    print(f"[evaluate_lesion_subset] score_dir = {LESION_SCORE_DIR}")
    print(f"[evaluate_lesion_subset] eval_dir  = {EVAL_DIR}")
    print(f"[evaluate_lesion_subset] 전체 케이스 {n_total}명, 이번 실행 {len(target_pids)}명")
    print(f"[evaluate_lesion_subset] label_mode={args.label_mode}, threshold_mode={args.threshold_mode}(미적용), dry_run={args.dry_run}")
    print(f"[evaluate_lesion_subset] output_tag={args.output_tag}")
    print()

    # --- PathResolver / DataLoader (정상 base join 로직 그대로) ---
    anchor_key = "nsclc_msd_usable_only_v2_anchor" if is_v2 else "nsclc_msd_usable_only_anchor"
    anchor = (cfg.get(anchor_key) or "").strip() or None
    path_resolver = PathResolver(str(manifest_path), str(base_path), anchor=anchor)
    loader = DataLoader(str(manifest_path), path_resolver, str(ERROR_CSV), use_mmap=True)

    # --- dry-run: 로드/경로 확인만 ---
    if args.dry_run:
        print("[evaluate_lesion_subset] === DRY-RUN: score 없이 로드 확인만 ===")
        ok = 0
        for pid in target_pids:
            data = loader.load_patient_data(pid, mask_type=mask_type)
            if data is None:
                print(f"  [FAIL] {pid}: 로드 실패")
                continue
            ok += 1
            print(f"  [OK]   {pid}: ct_hu={data['ct_hu'].shape}, patch={len(data['patch_df'])}, "
                  f"lesion_mask={'있음' if data['lesion_mask'] is not None else '없음'}")
        elapsed = time.time() - start_time
        print(f"\n[evaluate_lesion_subset] dry-run 완료: {ok}/{len(target_pids)}명 로드 OK, {elapsed:.1f}s")
        record_runtime_rows([
            {"timestamp": datetime.now().isoformat(timespec="seconds"), "script": SCRIPT_NAME,
             "metric": "mode", "value": "dry_run"},
            {"timestamp": datetime.now().isoformat(timespec="seconds"), "script": SCRIPT_NAME,
             "metric": "n_loaded_ok", "value": ok},
            {"timestamp": datetime.now().isoformat(timespec="seconds"), "script": SCRIPT_NAME,
             "metric": "elapsed_seconds", "value": round(elapsed, 2)},
        ])
        return

    # --- 모델 / FeatureExtractor 로드 ---
    model = PaDiMModel(feature_dim=100, eps=1e-5)
    model.load(str(MODEL_NPZ))
    print(f"[evaluate_lesion_subset] 모델 로드 완료: position_bin {len(model.stats)}개")
    feature_extractor = FeatureExtractor()
    print(f"[evaluate_lesion_subset] FeatureExtractor device: {feature_extractor.device}")
    print()

    LESION_SCORE_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # --- 환자별 스코어링 + label 생성 ---
    n_scored = 0
    n_skipped = 0
    n_failed = 0
    scored_frames = []  # 지표 계산용 — METRIC_COLS만 누적해 전체 308명에서도 메모리 안전

    for row in target_rows:
        pid = row["patient_id"]
        score_path = LESION_SCORE_DIR / f"{pid}.csv"

        if score_path.exists():
            n_skipped += 1
            print(f"  [SKIP] {pid}: 이미 존재 (resume)")
            try:
                scored_frames.append(
                    pd.read_csv(score_path, encoding="utf-8-sig", usecols=METRIC_COLS)
                )
            except Exception:
                pass
            continue

        data = loader.load_patient_data(pid, mask_type=mask_type)
        if data is None:
            n_failed += 1
            print(f"  [FAIL] {pid}: 로드 실패 (error.csv 기록됨)")
            continue

        try:
            scored_df = model.score_patient(feature_extractor, data)
        except Exception as exc:
            n_failed += 1
            record_error(pid, "score_error", str(exc), "padim_model.score_patient")
            print(f"  [FAIL] {pid}: 스코어링 오류 — {exc}")
            continue

        # patch_label 생성 (patch CSV lesion 컬럼 기반)
        try:
            scored_df["patch_label"] = make_patch_label(scored_df, args.label_mode)
        except Exception as exc:
            n_failed += 1
            record_error(pid, "label_error", str(exc), "make_patch_label")
            print(f"  [FAIL] {pid}: label 생성 오류 — {exc}")
            continue

        try:
            scored_df.to_csv(score_path, index=False, encoding="utf-8-sig")
            n_scored += 1
            n_nan = int(scored_df["padim_score"].isna().sum())
            n_pos = int((scored_df["patch_label"] == 1).sum())
            print(f"  [OK]   {pid}: patch={len(scored_df)}, NaN={n_nan}, positive={n_pos}")
            scored_frames.append(scored_df[METRIC_COLS].copy())
        except Exception as exc:
            n_failed += 1
            record_error(pid, "save_error", str(exc), str(score_path))
            print(f"  [FAIL] {pid}: 저장 오류 — {exc}")
            continue

    # --- score-only 모드: 지표 계산 생략 ---
    if args.score_only:
        elapsed = time.time() - start_time
        summary_so: dict = {
            "output_tag": args.output_tag,
            "dataset_profile": args.dataset_profile,
            "score_only": True,
            "note": "score-only 실행: 지표 계산 생략. compute_lesion_metrics_fast.py로 별도 계산.",
            "n_patients_requested": len(target_pids),
            "n_patients_scored": n_scored,
            "n_patients_skipped_resume": n_skipped,
            "n_patients_failed": n_failed,
        }
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        summary_json = EVAL_DIR / f"{args.output_tag}_summary.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_so, f, ensure_ascii=False, indent=2)
        print(f"\n[evaluate_lesion_subset] score-only 완료: {n_scored} scored, {n_skipped} skip, {n_failed} fail, {elapsed:.1f}s")
        print(f"[evaluate_lesion_subset] summary: {summary_json}")
        ts = datetime.now().isoformat(timespec="seconds")
        record_runtime_rows([
            {"timestamp": ts, "script": SCRIPT_NAME, "metric": "mode", "value": f"score_only(tag={args.output_tag})"},
            {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_scored", "value": n_scored},
            {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_skipped", "value": n_skipped},
            {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_failed", "value": n_failed},
            {"timestamp": ts, "script": SCRIPT_NAME, "metric": "elapsed_seconds", "value": round(elapsed, 2)},
        ])
        print(f"[evaluate_lesion_subset] runtime_summary.csv 기록 완료")
        return

    # --- 지표 계산 ---
    evaluator = Evaluator()
    metrics_rows = []
    summary: dict = {
        "output_tag": args.output_tag,
        "dataset_profile": args.dataset_profile,
        "mask_type": mask_type,
        "label_mode": args.label_mode,
        "threshold_mode": args.threshold_mode,
        "threshold_applied": False,
        "n_patients_requested": len(target_pids),
        "n_patients_scored": n_scored,
        "n_patients_skipped_resume": n_skipped,
        "n_patients_failed": n_failed,
        "note": "smoke test: 계산 파이프라인 작동 확인용. 병변 성능 결론 아님.",
    }

    threshold = None
    thr_info: dict = {"threshold_mode": args.threshold_mode, "status": "not_applied"}
    if scored_frames:
        all_df = pd.concat(scored_frames, ignore_index=True)
        # patch 레벨 (NaN score 제외)
        patch_df = all_df[~all_df["padim_score"].isna()].copy()
        n_inf = int(np.isinf(patch_df["padim_score"].values).sum())
        summary["patch_total"] = int(len(all_df))
        summary["patch_score_nan"] = int(all_df["padim_score"].isna().sum())
        summary["patch_score_inf"] = n_inf
        summary["patch_positive"] = int((all_df["patch_label"] == 1).sum())
        summary["patch_negative"] = int((all_df["patch_label"] == 0).sum())

        # threshold 계산 (patch_dice/patch_iou용)
        lesion_scores = patch_df["padim_score"].values if len(patch_df) > 0 else np.array([])
        threshold, thr_info = compute_threshold(args.threshold_mode, lesion_scores, threshold_json=args.threshold_json)
        summary["threshold_value"] = threshold
        summary["threshold_applied"] = threshold is not None
        summary["threshold_info"] = thr_info

        def _safe_metrics(df, score_col="padim_score", label_col="patch_label"):
            yt = df[label_col].values
            ys = df[score_col].values
            return evaluator.compute_auroc(yt, ys), evaluator.compute_auprc(yt, ys)

        # patch 레벨 지표 (+ threshold 있으면 patch_dice/patch_iou)
        if n_inf == 0 and len(patch_df) > 0:
            au, ap = _safe_metrics(patch_df)
            patch_dice = None
            patch_iou = None
            if threshold is not None:
                pred = (patch_df["padim_score"].values >= threshold).astype(int)
                yt = patch_df["patch_label"].values
                patch_dice = evaluator.compute_dice(yt, pred)
                patch_iou = evaluator.compute_iou(yt, pred)
            metrics_rows.append({"level": "patch", "auroc": au, "auprc": ap,
                                 "n": len(patch_df),
                                 "n_pos": int((patch_df["patch_label"] == 1).sum()),
                                 "patch_dice": patch_dice, "patch_iou": patch_iou,
                                 "threshold": threshold, "note": ""})

        # slice 레벨: 환자별 (patient_id, local_z) max score + slice_label
        slice_records = []
        for pid, g in patch_df.groupby("patient_id"):
            slabels = evaluator.compute_slice_labels(g)  # local_z, slice_label, ...
            smax = g.groupby("local_z")["padim_score"].max().reset_index()
            merged = slabels.merge(smax, on="local_z", how="left")
            merged["patient_id"] = pid
            slice_records.append(merged)
        if slice_records:
            slice_df = pd.concat(slice_records, ignore_index=True)
            summary["slice_total"] = int(len(slice_df))
            summary["slice_positive"] = int((slice_df["slice_label"] == 1).sum())
            if len(slice_df) > 0:
                au = evaluator.compute_auroc(slice_df["slice_label"].values, slice_df["padim_score"].values)
                ap = evaluator.compute_auprc(slice_df["slice_label"].values, slice_df["padim_score"].values)
                metrics_rows.append({"level": "slice", "auroc": au, "auprc": ap,
                                     "n": len(slice_df),
                                     "n_pos": int((slice_df["slice_label"] == 1).sum()),
                                     "patch_dice": None, "patch_iou": None,
                                     "threshold": None, "note": ""})

        # patient 레벨: max score + patient_label
        plabels = evaluator.compute_patient_labels(patch_df)
        pmax = patch_df.groupby("patient_id")["padim_score"].max().reset_index()
        pmerged = plabels.merge(pmax, on="patient_id", how="left")
        n_ptot = int(len(pmerged))
        n_ppos = int((pmerged["patient_label"] == 1).sum())
        summary["patient_total"] = n_ptot
        summary["patient_positive"] = n_ppos
        if n_ptot > 0 and n_ppos == n_ptot:
            # 전부 양성(lesion-only) → patient AUROC 계산 불가
            summary["patient_auroc_status"] = "not_applicable_positive_only"
            metrics_rows.append({"level": "patient", "auroc": None, "auprc": None,
                                 "n": n_ptot, "n_pos": n_ppos,
                                 "patch_dice": None, "patch_iou": None,
                                 "threshold": None, "note": "not_applicable_positive_only"})
        elif n_ptot > 0:
            au = evaluator.compute_auroc(pmerged["patient_label"].values, pmerged["padim_score"].values)
            ap = evaluator.compute_auprc(pmerged["patient_label"].values, pmerged["padim_score"].values)
            summary["patient_auroc_status"] = "computed"
            metrics_rows.append({"level": "patient", "auroc": au, "auprc": ap,
                                 "n": n_ptot, "n_pos": n_ppos,
                                 "patch_dice": None, "patch_iou": None,
                                 "threshold": None, "note": ""})

    # patch_dice/patch_iou는 patch-level (threshold 적용 시). pixel-level은 미구현.
    summary["pixel_dice_iou"] = (
        "확인 필요 (patch score heatmap을 pixel grid로 투영 후 lesion_mask와 비교 — 다음 단계)"
    )

    # --- 결과 저장 ---
    metrics_df = pd.DataFrame(
        metrics_rows,
        columns=["level", "auroc", "auprc", "n", "n_pos",
                 "patch_dice", "patch_iou", "threshold", "note"],
    )
    metrics_csv = EVAL_DIR / f"{args.output_tag}_metrics.csv"
    summary_json = EVAL_DIR / f"{args.output_tag}_summary.json"
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time

    # --- 콘솔 요약 ---
    print()
    print(f"[evaluate_lesion_subset] 완료: {n_scored} scored, {n_skipped} skip, {n_failed} fail, {elapsed:.1f}s")
    print(f"[evaluate_lesion_subset] patch positive/negative: "
          f"{summary.get('patch_positive')}/{summary.get('patch_negative')}")
    print(f"[evaluate_lesion_subset] threshold_mode={args.threshold_mode}, "
          f"threshold_value={threshold}, status={thr_info.get('status')}")
    print(f"[evaluate_lesion_subset] 지표 (smoke = 계산 가능 여부 확인용):")
    for r in metrics_rows:
        extra = ""
        if r.get("patch_dice") is not None or r.get("patch_iou") is not None:
            extra = f", patch_dice={r.get('patch_dice')}, patch_iou={r.get('patch_iou')}"
        if r.get("note"):
            extra += f", note={r['note']}"
        print(f"   {r['level']:8s} AUROC={r['auroc']}, AUPRC={r['auprc']} "
              f"(n={r['n']}, pos={r['n_pos']}){extra}")
    print(f"[evaluate_lesion_subset] metrics: {metrics_csv}")
    print(f"[evaluate_lesion_subset] summary: {summary_json}")

    # --- runtime_summary 기록 ---
    ts = datetime.now().isoformat(timespec="seconds")
    record_runtime_rows([
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "mode", "value": f"score(tag={args.output_tag})"},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "label_mode", "value": args.label_mode},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "threshold_mode", "value": args.threshold_mode},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "threshold_value", "value": str(threshold)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_scored", "value": n_scored},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_skipped", "value": n_skipped},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_failed", "value": n_failed},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "patch_positive", "value": summary.get("patch_positive")},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "elapsed_seconds", "value": round(elapsed, 2)},
    ])
    print(f"[evaluate_lesion_subset] runtime_summary.csv 기록 완료")


if __name__ == "__main__":
    main()
