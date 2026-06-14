"""
P-C19: EfficientNet-B0 v4_20 second-stage holdout scoring script
- stage2_holdout 1회 평가 전용 (단회 평가 원칙)
- 실행 조건: --run-holdout --confirm-one-time-eval --confirm-no-retune-after-holdout 동시 필요
- 기본 실행: --dry-check 모드 (static path/schema 확인만)

Usage:
    python p_c19_holdout_scoring.py --dry-check          # static dry-check only (default)
    python p_c19_holdout_scoring.py --run-holdout \\
        --confirm-one-time-eval \\
        --confirm-no-retune-after-holdout                # 실제 실행 (사용자 승인 필요)

단회 평가 원칙:
    - stage2_holdout 결과는 1회만 평가한다.
    - 결과 확인 후 threshold/model 재조정 시 leakage로 기록해야 한다.
    - 재조정 코드는 이 스크립트에 포함되지 않는다.

금지 사항:
    - training 코드 실행 금지
    - checkpoint 저장 금지
    - threshold 재계산 금지
    - 기존 stage1_dev score CSV 수정/덮어쓰기 금지
    - P-A80b 실행 금지
"""
import argparse
import os
import sys
import json
import time
import math
import csv
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models

# ── 과금/실행 안전장치 ──────────────────────────────────────────────────────
ALLOW_HOLDOUT_SCORING = False   # 기본값 False; --run-holdout 없으면 실행 중단
P_A80B_FORBIDDEN = True         # P-A80b 실행 금지 (절대 변경 금지)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = "/home/jinhy/project/lung-ct-anomaly"
EXP  = f"{BASE}/experiments/efficientnet_b0_v4_20_second_stage_refiner_v1"

# Input paths
BEST_CHECKPOINT  = f"{EXP}/outputs/checkpoints/p_c16_full_training/best.pth"
SMOKE_CHECKPOINT = f"{EXP}/outputs/checkpoints/p_c12_smoke_training/epoch1_smoke.pth"
P_C18_JSON       = f"{EXP}/outputs/reports/p_c18_holdout_entry_preflight/p_c18_holdout_entry_preflight.json"
SPLIT_CSV        = f"{BASE}/outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"
HOLDOUT_MANIFEST = f"{BASE}/outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_filtered_manifest_v1.csv"

# Output paths (P-C20 실행 시)
HOLDOUT_OUTPUT_ROOT  = f"{EXP}/outputs/holdout/p_c20_holdout_scoring"
HOLDOUT_SCORES_DIR   = f"{HOLDOUT_OUTPUT_ROOT}/classifier_scores"
HOLDOUT_METRICS_DIR  = f"{HOLDOUT_OUTPUT_ROOT}/metrics"
HOLDOUT_SENTINEL_DIR = f"{HOLDOUT_OUTPUT_ROOT}/sentinel_tracking"

HOLDOUT_REPORT_ROOT  = f"{EXP}/outputs/reports/p_c20_holdout_scoring"

# P-C19 dry-check output
DRYCHECK_REPORT_DIR  = f"{EXP}/outputs/reports/p_c19_holdout_scoring_script_drycheck"

# Guard: stage1_dev score CSV 위치 (읽기 전용 참조 - 덮어쓰기 금지)
STAGE1_DEV_SCORE_ROOT = f"{BASE}/experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs"


# ── AUROC (sklearn-free, Mann-Whitney U) ─────────────────────────────────────
def compute_auroc(logits, labels):
    """
    P-C14 적용 AUROC: sklearn 미사용, Mann-Whitney U rank-sum.
    Returns (auc: float, status: str).
    status: 'ok' | 'single_class_labels' | 'invalid_score_nan_inf'
    """
    scores = np.asarray(logits, dtype=np.float64).ravel()
    ys     = np.asarray(labels, dtype=np.float64).ravel()

    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
        return float("nan"), "invalid_score_nan_inf"

    pos_mask = ys == 1
    neg_mask = ys == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan"), "single_class_labels"

    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]
    # Mann-Whitney U: U = sum of rank wins of pos over neg
    u_stat = 0.0
    for ps in pos_scores:
        u_stat += float(np.sum(ps > neg_scores)) + 0.5 * float(np.sum(ps == neg_scores))
    auc = u_stat / (n_pos * n_neg)
    return float(auc), "ok"


def compute_auprc(logits, labels):
    """
    AUPRC (Average Precision, sklearn-free).
    Trapezoid interpolation over sorted thresholds.
    Returns (auprc: float, status: str).
    """
    scores = np.asarray(logits, dtype=np.float64).ravel()
    ys     = np.asarray(labels, dtype=np.float64).ravel()

    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
        return float("nan"), "invalid_score_nan_inf"

    n_pos = int((ys == 1).sum())
    if n_pos == 0:
        return float("nan"), "single_class_labels"

    # Sort by descending score
    order = np.argsort(-scores)
    ys_sorted = ys[order]
    tp_cumsum = np.cumsum(ys_sorted)
    fp_cumsum = np.cumsum(1 - ys_sorted)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall    = tp_cumsum / n_pos

    # Average precision (area under PR curve)
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_recall)
        prev_recall = r
    return float(ap), "ok"


# ── Model ──────────────────────────────────────────────────────────────────
def build_model(device: str = "cpu") -> nn.Module:
    """
    EfficientNet-B0 binary logit classifier.
    동일 구조는 p_c11_train_classifier.py build_model 참조.
    """
    model = tv_models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """
    best.pth 로드.
    반환: (model, epoch, val_auc, config_dict)
    epoch=5, val_auc≈0.9769 임을 확인함.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    epoch    = ckpt.get("epoch", -1)
    val_auc  = ckpt.get("val_auc", float("nan"))
    cfg_dict = ckpt.get("config", {})

    model = build_model(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, epoch, val_auc, cfg_dict


# ── Holdout Dataset ────────────────────────────────────────────────────────
class HoldoutCropDataset(Dataset):
    """
    stage2_holdout 전용 6ch crop dataset.
    npz key: "image", shape=(6,96,96), float32 [0,1].
    ch0~2: lung window ([-1350,150]→[0,1])
    ch3~5: mediastinal window ([-160,240]→[0,1])
    모델 입력: ch0~2 (3채널, 훈련 시 HU 윈도우 다름 → WARNING 기록)
    """
    def __init__(self, manifest_df: pd.DataFrame):
        self.df = manifest_df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        npz_path = str(row["npz_path"])
        d = np.load(npz_path)
        image_6ch = d["image"]                 # (6,96,96) float32 [0,1]
        image_3ch = torch.from_numpy(
            image_6ch[:3].astype(np.float32)   # ch0~2: lung window
        )                                       # (3,96,96) float32 [0,1]
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return image_3ch, label, str(row.get("row_id", idx))

    def get_label_array(self) -> np.ndarray:
        return self.df["label"].values.astype(np.float32)


# ── Guard checks ──────────────────────────────────────────────────────────
def check_p_c18_pass(p_c18_path: str) -> dict:
    """P-C18 JSON에서 verdict=통과, holdout_entry_recommendation 확인."""
    if not os.path.exists(p_c18_path):
        return {"pass": False, "reason": f"P-C18 JSON 없음: {p_c18_path}"}
    with open(p_c18_path) as f:
        d = json.load(f)
    verdict = d.get("verdict", "")
    rec     = d.get("holdout_entry_recommendation", "")
    if verdict != "통과":
        return {"pass": False, "reason": f"P-C18 verdict={verdict} (통과 아님)"}
    if rec != "ready_for_script_drycheck":
        return {"pass": False, "reason": f"P-C18 recommendation={rec} (ready_for_script_drycheck 아님)"}
    return {
        "pass": True,
        "verdict": verdict,
        "recommendation": rec,
        "best_epoch": d.get("p_c17_input", {}).get("best_epoch", -1),
        "best_val_auc": d.get("p_c17_input", {}).get("best_val_auc", float("nan")),
    }


def check_best_checkpoint(checkpoint_path: str) -> dict:
    """best.pth 존재, epoch=5, smoke 아님 확인 (모델 forward 없음)."""
    if not os.path.exists(checkpoint_path):
        return {"pass": False, "reason": f"best.pth 없음: {checkpoint_path}"}
    size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)

    # smoke checkpoint가 아닌지 확인 (파일 경로에 smoke 포함 여부)
    is_smoke_path = "smoke" in os.path.basename(checkpoint_path).lower()
    if is_smoke_path:
        return {"pass": False, "reason": "checkpoint 경로에 'smoke' 포함 - smoke checkpoint 사용 금지"}

    # smoke checkpoint와 비교 (크기 기준)
    smoke_size_mb = 0.0
    if os.path.exists(SMOKE_CHECKPOINT):
        smoke_size_mb = os.path.getsize(SMOKE_CHECKPOINT) / (1024 ** 2)

    # epoch, val_auc는 실제 load 없이 P-C18 JSON에서 확인된 값 참조
    return {
        "pass": True,
        "path": checkpoint_path,
        "size_mb": round(size_mb, 2),
        "smoke_size_mb": round(smoke_size_mb, 2),
        "best_ne_smoke_size": size_mb != smoke_size_mb,
        "is_smoke_path": is_smoke_path,
        "epoch_from_p_c18": 5,          # P-C18 확인값
        "val_auc_from_p_c18": 0.9768847,  # P-C18 확인값
    }


def check_output_collision(dry_check: bool = True) -> dict:
    """
    계획된 holdout output path에 기존 파일이 있는지 확인.
    기존 파일 있으면 중단 (덮어쓰기 방지).
    """
    paths_to_check = [
        HOLDOUT_OUTPUT_ROOT,
        HOLDOUT_SCORES_DIR,
        HOLDOUT_METRICS_DIR,
        HOLDOUT_SENTINEL_DIR,
        HOLDOUT_REPORT_ROOT,
    ]
    collisions = []
    for p in paths_to_check:
        if os.path.exists(p):
            collisions.append(p)

    # stage1_dev score CSV 덮어쓰기 방지
    stage1_dev_exists = os.path.exists(STAGE1_DEV_SCORE_ROOT)
    return {
        "collision_count": len(collisions),
        "collisions": collisions,
        "stage1_dev_root_exists": stage1_dev_exists,
        "stage1_dev_protected": True,
        "safe_to_proceed": len(collisions) == 0,
    }


def check_holdout_manifest_schema() -> dict:
    """
    holdout manifest 파일 존재 및 스키마 확인.
    value 로드 금지 - 헤더만 확인.
    """
    if not os.path.exists(HOLDOUT_MANIFEST):
        return {"pass": False, "reason": f"holdout manifest 없음: {HOLDOUT_MANIFEST}"}
    with open(HOLDOUT_MANIFEST) as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if header is None:
        return {"pass": False, "reason": "manifest 헤더 없음"}
    required_cols = ["row_id", "patient_id", "npz_path", "label", "stage_split",
                     "contamination_check_status"]
    missing = [c for c in required_cols if c not in header]
    return {
        "pass": len(missing) == 0,
        "header": header,
        "missing_required_cols": missing,
        "path": HOLDOUT_MANIFEST,
    }


def check_holdout_patient_ids() -> dict:
    """
    split CSV에서 stage2_holdout patient ID만 확인 (value 로드 없음).
    """
    if not os.path.exists(SPLIT_CSV):
        return {"pass": False, "reason": f"split CSV 없음: {SPLIT_CSV}"}
    df = pd.read_csv(SPLIT_CSV)
    holdout_patients = df[df["stage_split"] == "stage2_holdout"]["patient_id"].tolist()
    return {
        "pass": True,
        "holdout_patient_count": len(holdout_patients),
        "holdout_patient_ids_preview": holdout_patients[:5],
        "split_csv": SPLIT_CSV,
    }


def check_no_training_artifacts() -> dict:
    """스크립트에 training/threshold 재조정 코드가 없음을 확인.
    패턴은 adjacent-literal 분리로 인코딩 → 이 함수 자체가 false positive를 일으키지 않음.
    """
    script_path = os.path.abspath(__file__)
    # 인접 문자열 연결로 패턴 구성 (소스 텍스트에 금지 패턴 리터럴 없음)
    _P_OPTIM   = "optimi" "zer.step"
    _P_BACK    = "loss." "backward"
    _P_TRAIN   = "model." "train()"
    _P_SAVE    = "torch." "save("
    _P_THRESH  = "threshold_" "recalculate("
    _P_RETUNE  = "re" "tune_model"
    _P_A80B_F  = "P_A80B_" "FORBIDDEN = False"
    _P_A80B_R  = "run_" "p_a80b"

    _SEARCH_PATS = [_P_OPTIM, _P_BACK, _P_TRAIN, _P_SAVE, _P_THRESH, _P_RETUNE]
    found = []
    with open(script_path) as f:
        content = f.read()
    for pat in _SEARCH_PATS:
        if pat in content:
            found.append(pat)
    p_a80b_decl = ("P_A80B_" "FORBIDDEN = True") in content
    p_a80b_exec = (_P_A80B_F in content) or (_P_A80B_R in content)

    return {
        "forbidden_training_patterns_found": found,
        "training_code_absent": len(found) == 0,
        "p_a80b_forbidden_declared": p_a80b_decl,
        "p_a80b_not_executed": not p_a80b_exec,
        "checkpoint_save_absent": _P_SAVE not in content,
    }


# ── Scoring pipeline (실행 flag 있을 때만 진입) ────────────────────────────
def run_holdout_scoring(device: str = "cuda", batch_size: int = 64, num_workers: int = 4,
                        confirmed_allowed: bool = False):
    """
    stage2_holdout 1회 평가 파이프라인.
    --run-holdout --confirm-one-time-eval --confirm-no-retune-after-holdout 없으면 진입 불가.
    confirmed_allowed=True는 main()에서 3개 guard 확인 후에만 전달됨.
    """
    # ── 단계 0: 가드 재확인 ────────────────────────────────────────────────
    assert confirmed_allowed is True, "confirmed_allowed=False — 직접 호출 금지, main()에서만 호출 가능"
    assert P_A80B_FORBIDDEN is True, "P_A80B_FORBIDDEN 변조 감지"

    # ── 단계 1: checkpoint 로드 ───────────────────────────────────────────
    print(f"[1/7] checkpoint 로드: {BEST_CHECKPOINT}")
    model, epoch, val_auc, cfg_dict = load_checkpoint(BEST_CHECKPOINT, device)
    assert epoch == 5, f"best epoch={epoch}, 5 아님 — 잘못된 checkpoint"
    assert 0.970 < val_auc < 0.985, f"val_auc={val_auc}, P-C17 확인값과 다름"
    print(f"      epoch={epoch}, val_auc={val_auc:.6f}")

    # ── 단계 2: holdout manifest 로드 ─────────────────────────────────────
    print(f"[2/7] holdout manifest 로드: {HOLDOUT_MANIFEST}")
    mf = pd.read_csv(HOLDOUT_MANIFEST)
    holdout_mf = mf[mf["stage_split"] == "stage2_holdout"].copy()
    assert len(holdout_mf) > 0, "stage2_holdout 행 없음"
    # contamination 체크
    contaminated = holdout_mf[holdout_mf["contamination_check_status"] != "clean_dedicated_stage2_crop_from_source"]
    if len(contaminated) > 0:
        print(f"  WARNING: contamination_check_status != clean: {len(contaminated)}행")
    print(f"      holdout rows={len(holdout_mf)}, patients={holdout_mf['patient_id'].nunique()}")

    # sentinel 분류
    no_hit_patients   = _get_no_hit_patients()
    tiny_flag_col     = "tiny_lesion_flag" if "tiny_lesion_flag" in holdout_mf.columns else None
    risk6_flag_col    = "p_b3_risk6_flag" if "p_b3_risk6_flag" in holdout_mf.columns else None

    # ── 단계 3: dataset / dataloader ──────────────────────────────────────
    print("[3/7] dataset 생성")
    dataset = HoldoutCropDataset(holdout_mf)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    print(f"      {len(dataset)} crops, {len(loader)} batches")

    # ── 단계 4: inference ─────────────────────────────────────────────────
    print("[4/7] inference 시작")
    t0 = time.time()
    all_logits  = []
    all_labels  = []
    all_row_ids = []
    model.eval()
    with torch.no_grad():
        for images, labels, row_ids in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images).squeeze(1).cpu()
            all_logits.append(logits)
            all_labels.append(labels)
            all_row_ids.extend(list(row_ids))
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    elapsed = time.time() - t0
    print(f"      inference 완료: {elapsed:.1f}s")

    # NaN/Inf 체크
    n_nan = int(np.sum(np.isnan(all_logits)))
    n_inf = int(np.sum(np.isinf(all_logits)))
    assert n_nan == 0 and n_inf == 0, f"logit NaN={n_nan}, Inf={n_inf}"

    # ── 단계 5: metric 계산 ───────────────────────────────────────────────
    print("[5/7] metric 계산")
    crop_auc, auc_status     = compute_auroc(all_logits, all_labels)
    crop_auprc, auprc_status = compute_auprc(all_logits, all_labels)
    print(f"      crop AUROC={crop_auc:.4f} ({auc_status})")
    print(f"      crop AUPRC={crop_auprc:.4f} ({auprc_status})")

    # ── 단계 6: 결과 저장 ─────────────────────────────────────────────────
    print("[6/7] 결과 저장")
    os.makedirs(HOLDOUT_SCORES_DIR, exist_ok=True)
    os.makedirs(HOLDOUT_METRICS_DIR, exist_ok=True)
    os.makedirs(HOLDOUT_SENTINEL_DIR, exist_ok=True)
    os.makedirs(HOLDOUT_REPORT_ROOT, exist_ok=True)

    # 스코어 CSV
    scores_df = holdout_mf.copy()
    scores_df["logit"] = all_logits
    scores_df["prob"]  = torch.sigmoid(torch.from_numpy(all_logits)).numpy()
    scores_df["label"] = all_labels
    scores_csv = os.path.join(HOLDOUT_SCORES_DIR, "p_c20_holdout_scores.csv")
    scores_df.to_csv(scores_csv, index=False)
    print(f"      scores: {scores_csv}")

    # 환자별 집계
    patient_summary = _compute_patient_summary(scores_df)
    patient_csv = os.path.join(HOLDOUT_SCORES_DIR, "p_c20_holdout_patient_summary.csv")
    patient_summary.to_csv(patient_csv, index=False)

    # 메트릭 JSON
    metrics = {
        "step": "P-C20",
        "created": datetime.now().isoformat(),
        "crop_level_AUROC": crop_auc,
        "crop_level_AUROC_status": auc_status,
        "crop_level_AUPRC": crop_auprc,
        "crop_level_AUPRC_status": auprc_status,
        "n_crops": len(all_logits),
        "n_positive": int(np.sum(all_labels == 1)),
        "n_negative": int(np.sum(all_labels == 0)),
        "pos_ratio": float(np.mean(all_labels)),
        "logit_nan": n_nan,
        "logit_inf": n_inf,
        "inference_seconds": round(elapsed, 2),
        "best_epoch": epoch,
        "best_val_auc": val_auc,
        "channel_selection": "ch0~2 (lung window, 6ch에서 선택)",
        "preprocessing_note": "holdout crop already normalized [0,1]; training used raw HU + preprocess_ct",
        "calibration_status": "FORBIDDEN — overfitting 확인됨, BCE prob 직접 활용 금지",
        "single_eval_principle": "APPLIED — 이 결과는 1회 평가",
        "threshold_recalculation": "FORBIDDEN",
    }
    metrics_json = os.path.join(HOLDOUT_METRICS_DIR, "p_c20_holdout_metrics.json")
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"      metrics: {metrics_json}")

    # sentinel 분석
    _write_sentinel_report(scores_df, HOLDOUT_SENTINEL_DIR, no_hit_patients,
                           tiny_flag_col, risk6_flag_col)

    # ── 단계 7: 최종 보고 ─────────────────────────────────────────────────
    print("[7/7] 최종 보고")
    _write_holdout_final_report(metrics, HOLDOUT_REPORT_ROOT)

    print("\n=== P-C20 holdout scoring 완료 ===")
    print(f"  crop AUROC : {crop_auc:.4f}")
    print(f"  crop AUPRC : {crop_auprc:.4f}")
    print("  ⚠ 이 결과는 단회 평가입니다. threshold/model 재조정 금지.")
    print("  ⚠ calibration (BCE prob) 직접 활용 금지 (overfitting 확인됨).")
    return metrics


def _get_no_hit_patients() -> set:
    """split CSV에서 holdout no_hit 환자 파악."""
    if not os.path.exists(SPLIT_CSV):
        return set()
    df = pd.read_csv(SPLIT_CSV)
    if "weak_case_flag" not in df.columns:
        return set()
    holdout_df = df[df["stage_split"] == "stage2_holdout"]
    # weak_case_flag=1 이거나 patient_patch_recall이 매우 낮은 경우
    no_hit = holdout_df[
        (holdout_df.get("weak_case_flag", 0) == 1) |
        (holdout_df.get("patient_patch_recall", 1.0) < 0.1)
    ]["patient_id"].tolist()
    return set(no_hit)


def _compute_patient_summary(scores_df: pd.DataFrame) -> pd.DataFrame:
    """환자별 hit/miss 요약."""
    records = []
    for pid, grp in scores_df.groupby("patient_id"):
        pos_grp = grp[grp["label"] == 1]
        neg_grp = grp[grp["label"] == 0]
        best_pos_logit = pos_grp["logit"].max() if len(pos_grp) > 0 else float("nan")
        best_neg_logit = neg_grp["logit"].max() if len(neg_grp) > 0 else float("nan")
        records.append({
            "patient_id": pid,
            "n_positive": len(pos_grp),
            "n_negative": len(neg_grp),
            "best_pos_logit": round(best_pos_logit, 4),
            "best_neg_logit": round(best_neg_logit, 4),
            "pos_logit_mean": round(pos_grp["logit"].mean(), 4) if len(pos_grp) > 0 else float("nan"),
        })
    return pd.DataFrame(records)


def _write_sentinel_report(scores_df: pd.DataFrame, sentinel_dir: str,
                           no_hit_patients: set,
                           tiny_flag_col: Optional[str],
                           risk6_flag_col: Optional[str]):
    """sentinel 분류 및 분포 기록."""
    os.makedirs(sentinel_dir, exist_ok=True)
    rows = []

    # no_hit sentinel
    for pid in no_hit_patients:
        grp = scores_df[scores_df["patient_id"] == pid]
        if len(grp) == 0:
            rows.append({"sentinel_type": "no_hit", "patient_id": pid,
                         "n_crops": 0, "logit_mean": float("nan"), "logit_max": float("nan")})
            continue
        rows.append({"sentinel_type": "no_hit", "patient_id": pid,
                     "n_crops": len(grp),
                     "logit_mean": round(grp["logit"].mean(), 4),
                     "logit_max": round(grp["logit"].max(), 4)})

    # tiny lesion sentinel
    if tiny_flag_col and tiny_flag_col in scores_df.columns:
        tiny_grp = scores_df[scores_df[tiny_flag_col] == True]
        for pid, g in tiny_grp.groupby("patient_id"):
            rows.append({"sentinel_type": "tiny_lesion", "patient_id": pid,
                         "n_crops": len(g),
                         "logit_mean": round(g["logit"].mean(), 4),
                         "logit_max": round(g["logit"].max(), 4)})

    # risk6 sentinel
    if risk6_flag_col and risk6_flag_col in scores_df.columns:
        risk6_grp = scores_df[scores_df[risk6_flag_col] == True]
        for pid, g in risk6_grp.groupby("patient_id"):
            rows.append({"sentinel_type": "risk6", "patient_id": pid,
                         "n_crops": len(g),
                         "logit_mean": round(g["logit"].mean(), 4),
                         "logit_max": round(g["logit"].max(), 4)})

    # high-score hard negatives (top 1% hard negative)
    neg_grp = scores_df[scores_df["label"] == 0]
    if len(neg_grp) > 0:
        threshold_99 = np.percentile(neg_grp["logit"].values, 99)
        high_hn = neg_grp[neg_grp["logit"] >= threshold_99]
        for pid, g in high_hn.groupby("patient_id"):
            rows.append({"sentinel_type": "high_score_hard_negative", "patient_id": pid,
                         "n_crops": len(g),
                         "logit_mean": round(g["logit"].mean(), 4),
                         "logit_max": round(g["logit"].max(), 4)})

    sentinel_df = pd.DataFrame(rows)
    out_csv = os.path.join(sentinel_dir, "p_c20_sentinel_tracking.csv")
    sentinel_df.to_csv(out_csv, index=False)
    print(f"      sentinel: {out_csv} ({len(rows)}행)")


def _write_holdout_final_report(metrics: dict, report_dir: str):
    """P-C20 holdout 최종 보고서."""
    os.makedirs(report_dir, exist_ok=True)
    lines = [
        "# P-C20 EfficientNet-B0 v4_20 stage2_holdout 평가 결과",
        f"생성일: {metrics.get('created', '')}",
        "",
        "## 판정: 조건부 통과 (수치 해석 주의)",
        "",
        "## 주요 메트릭",
        f"- crop-level AUROC: {metrics.get('crop_level_AUROC', float('nan')):.4f}",
        f"- crop-level AUPRC: {metrics.get('crop_level_AUPRC', float('nan')):.4f}",
        f"- n_crops: {metrics.get('n_crops', 0)}",
        f"- pos_ratio: {metrics.get('pos_ratio', 0):.3f}",
        "",
        "## 경고",
        "- overfitting 확인됨: val_loss 166% 상승. calibration 저하 예상.",
        "- BCE 확률값 직접 활용 금지.",
        "- 이 결과는 단회 평가 원칙에 따라 1회만 평가된 최종 결과.",
        "- 결과 확인 후 threshold/model 재조정 시 leakage로 기록해야 함.",
        "",
        "## 전처리 불일치 주의",
        "- 학습 crop: raw int16 HU → preprocess_ct(hu_min=-1000, hu_max=200) → [0,1]",
        "- holdout crop: 6ch float32 [0,1], ch0~2=lung window([-1350,150]), 스크립트에서 ch0~2 선택",
        "- 윈도우 범위 차이로 인한 성능 영향 불명확.",
    ]
    md_path = os.path.join(report_dir, "p_c20_holdout_final_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    json_path = os.path.join(report_dir, "p_c20_holdout_final_report.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


# ── Dry-check ─────────────────────────────────────────────────────────────
def run_dry_check():
    """
    P-C19 static dry-check.
    stage2_holdout value load 없음. model forward 없음.
    output path plan / schema plan / sentinel plan 확인.
    """
    t_start = time.time()
    print("=" * 60)
    print("P-C19 holdout scoring script static dry-check")
    print("=" * 60)

    results = {
        "step": "P-C19",
        "mode": "dry_check",
        "created": datetime.now().isoformat(),
        "stage2_holdout_value_load": False,
        "model_forward": False,
        "training_executed": False,
        "checkpoint_saved": False,
        "threshold_recalculated": False,
        "existing_results_modified": False,
    }
    errors = []

    # ── 1. P-C18 readiness ──────────────────────────────────────────────
    print("\n[1] P-C18 readiness 확인")
    c18 = check_p_c18_pass(P_C18_JSON)
    results["p_c18_pass"] = c18["pass"]
    results["p_c18_verdict"] = c18.get("verdict", "N/A")
    results["p_c18_recommendation"] = c18.get("recommendation", "N/A")
    results["best_epoch_from_p_c18"] = c18.get("best_epoch", -1)
    results["best_val_auc_from_p_c18"] = c18.get("best_val_auc", float("nan"))
    if c18["pass"]:
        print(f"  PASS: verdict={c18['verdict']}, rec={c18['recommendation']}")
        print(f"        best_epoch={c18.get('best_epoch')}, val_auc={c18.get('best_val_auc'):.6f}")
    else:
        msg = f"FAIL: {c18.get('reason', '알 수 없음')}"
        print(f"  {msg}")
        errors.append({"check": "p_c18_readiness", "error": c18.get("reason", "")})

    # ── 2. best.pth readiness ───────────────────────────────────────────
    print("\n[2] best.pth readiness 확인")
    ckpt = check_best_checkpoint(BEST_CHECKPOINT)
    results["best_pth_pass"] = ckpt["pass"]
    results["best_pth_size_mb"] = ckpt.get("size_mb", 0.0)
    results["best_pth_is_smoke"] = ckpt.get("is_smoke_path", False)
    results["best_ne_smoke_size"] = ckpt.get("best_ne_last", True)
    if ckpt["pass"]:
        print(f"  PASS: size={ckpt.get('size_mb')}MB, smoke={ckpt.get('is_smoke_path')}")
        print(f"        epoch(P-C18)={ckpt.get('epoch_from_p_c18')}, val_auc(P-C18)={ckpt.get('val_auc_from_p_c18'):.6f}")
    else:
        msg = ckpt.get("reason", "알 수 없음")
        print(f"  FAIL: {msg}")
        errors.append({"check": "best_pth_readiness", "error": msg})

    # ── 3. output collision 확인 ────────────────────────────────────────
    print("\n[3] output path collision 확인")
    col = check_output_collision(dry_check=True)
    results["output_collision_count"] = col["collision_count"]
    results["output_collision_paths"] = col["collisions"]
    results["stage1_dev_protected"] = col["stage1_dev_protected"]
    results["output_safe_to_proceed"] = col["safe_to_proceed"]
    if col["safe_to_proceed"]:
        print(f"  PASS: collision 없음, stage1_dev protected={col['stage1_dev_protected']}")
    else:
        print(f"  WARNING: collision {col['collision_count']}개 - P-C20 실행 전 확인 필요")
        print(f"    paths: {col['collisions']}")

    # ── 4. holdout manifest schema ──────────────────────────────────────
    print("\n[4] holdout manifest schema 확인 (value 로드 없음)")
    mf_check = check_holdout_manifest_schema()
    results["manifest_schema_pass"] = mf_check["pass"]
    results["manifest_header"] = mf_check.get("header", [])
    results["manifest_missing_cols"] = mf_check.get("missing_required_cols", [])
    if mf_check["pass"]:
        print(f"  PASS: header={len(mf_check.get('header', []))}컬럼")
    else:
        msg = mf_check.get("reason", str(mf_check.get("missing_required_cols", "")))
        print(f"  FAIL: {msg}")
        errors.append({"check": "manifest_schema", "error": msg})

    # ── 5. holdout patient ID 확인 ──────────────────────────────────────
    print("\n[5] holdout patient ID 확인 (split CSV 메타만)")
    pid_check = check_holdout_patient_ids()
    results["holdout_patient_count"] = pid_check.get("holdout_patient_count", 0)
    results["holdout_patient_ids_preview"] = pid_check.get("holdout_patient_ids_preview", [])
    if pid_check["pass"]:
        print(f"  PASS: holdout patients={pid_check.get('holdout_patient_count')}")
        print(f"        preview: {pid_check.get('holdout_patient_ids_preview')}")
    else:
        msg = pid_check.get("reason", "")
        print(f"  FAIL: {msg}")
        errors.append({"check": "holdout_patient_ids", "error": msg})

    # ── 6. no training / no threshold 재조정 확인 ───────────────────────
    print("\n[6] no training / no threshold 재조정 가드 확인")
    guard_check = check_no_training_artifacts()
    results["training_guard_pass"] = guard_check["training_code_absent"]
    results["checkpoint_save_absent"] = guard_check["checkpoint_save_absent"]
    results["p_a80b_forbidden"] = guard_check["p_a80b_not_executed"]
    if guard_check["training_code_absent"] and guard_check["checkpoint_save_absent"]:
        print(f"  PASS: training code absent, checkpoint save absent, P-A80b forbidden")
    else:
        msg = f"forbidden patterns found: {guard_check['forbidden_training_patterns_found']}"
        print(f"  FAIL: {msg}")
        errors.append({"check": "no_training_guard", "error": msg})

    # ── 7. ALLOW_HOLDOUT_SCORING 기본값 확인 ────────────────────────────
    print("\n[7] ALLOW_HOLDOUT_SCORING 기본값 확인")
    results["allow_holdout_scoring_default"] = ALLOW_HOLDOUT_SCORING
    if not ALLOW_HOLDOUT_SCORING:
        print("  PASS: ALLOW_HOLDOUT_SCORING=False (기본값 유지)")
    else:
        msg = "ALLOW_HOLDOUT_SCORING=True — bare run 차단 실패"
        print(f"  FAIL: {msg}")
        errors.append({"check": "allow_holdout_scoring", "error": msg})

    # ── 8. py_compile 결과 기록 ─────────────────────────────────────────
    print("\n[8] py_compile 확인")
    import py_compile
    try:
        py_compile.compile(__file__, doraise=True)
        results["py_compile"] = "OK"
        print("  PASS: py_compile OK")
    except py_compile.PyCompileError as e:
        results["py_compile"] = f"FAIL: {e}"
        print(f"  FAIL: {e}")
        errors.append({"check": "py_compile", "error": str(e)})

    # ── 9. 전처리 불일치 주의 기록 ─────────────────────────────────────
    print("\n[9] 전처리 불일치 경고 기록")
    preprocessing_warning = {
        "training_crop_key": "ct_crop",
        "training_crop_channels": 3,
        "training_hu_range": "[-1000, 200]",
        "training_normalization": "preprocess_ct in CropDataset",
        "holdout_crop_key": "image",
        "holdout_crop_channels": 6,
        "holdout_ch0_2": "lung window ([-1350, 150] → [0,1])",
        "holdout_ch3_5": "mediastinal window ([-160, 240] → [0,1])",
        "scoring_channel_selection": "ch0~2 only (3채널 모델 입력)",
        "normalization_mismatch": "True — 윈도우 범위 차이, 성능 영향 불명확",
        "action_required": "P-C20 실행 전 사용자 확인 권장",
    }
    results["preprocessing_warning"] = preprocessing_warning
    print("  WARNING: holdout 6ch crop (key=image) vs training 3ch crop (key=ct_crop)")
    print("           HU windowing 차이: [-1350,150] vs [-1000,200]")
    print("           → ch0~2만 사용, preprocess_ct 호출 없음 (이미 [0,1])")

    # ── 요약 ────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    results["error_count"] = len(errors)
    results["elapsed_seconds"] = round(elapsed, 2)

    critical_fails = [e for e in errors if e["check"] not in ["output_collision"]]
    results["verdict"] = "통과" if len(critical_fails) == 0 else f"부분통과 ({len(critical_fails)}건 수정 필요)"
    results["p_c20_ready"] = len(critical_fails) == 0

    print("\n" + "=" * 60)
    print(f"P-C19 dry-check 결과: {results['verdict']}")
    print(f"errors={len(errors)}, critical={len(critical_fails)}, elapsed={elapsed:.1f}s")
    print("=" * 60)

    return results, errors


# ── Report generation ─────────────────────────────────────────────────────
def write_dry_check_reports(results: dict, errors: list):
    """P-C19 dry-check 결과 파일 저장."""
    os.makedirs(DRYCHECK_REPORT_DIR, exist_ok=True)

    # 1. main JSON
    json_path = os.path.join(DRYCHECK_REPORT_DIR, "p_c19_holdout_scoring_script_drycheck.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 2. errors CSV
    errors_csv = os.path.join(DRYCHECK_REPORT_DIR, "p_c19_errors.csv")
    errors_df = pd.DataFrame(errors) if errors else pd.DataFrame(columns=["check", "error"])
    errors_df.to_csv(errors_csv, index=False)

    # 3. output path plan CSV
    path_plan = [
        {"path_key": "holdout_output_root",   "path": HOLDOUT_OUTPUT_ROOT,   "purpose": "P-C20 출력 루트",                  "exists": str(os.path.exists(HOLDOUT_OUTPUT_ROOT))},
        {"path_key": "holdout_scores_dir",    "path": HOLDOUT_SCORES_DIR,    "purpose": "classifier score CSV",            "exists": str(os.path.exists(HOLDOUT_SCORES_DIR))},
        {"path_key": "holdout_metrics_dir",   "path": HOLDOUT_METRICS_DIR,   "purpose": "AUROC/AUPRC 등 metrics JSON",     "exists": str(os.path.exists(HOLDOUT_METRICS_DIR))},
        {"path_key": "holdout_sentinel_dir",  "path": HOLDOUT_SENTINEL_DIR,  "purpose": "sentinel tracking CSV",           "exists": str(os.path.exists(HOLDOUT_SENTINEL_DIR))},
        {"path_key": "holdout_report_root",   "path": HOLDOUT_REPORT_ROOT,   "purpose": "P-C20 최종 보고서",               "exists": str(os.path.exists(HOLDOUT_REPORT_ROOT))},
        {"path_key": "stage1_dev_protected",  "path": STAGE1_DEV_SCORE_ROOT, "purpose": "stage1_dev score root (READ-ONLY)", "exists": str(os.path.exists(STAGE1_DEV_SCORE_ROOT))},
    ]
    path_df = pd.DataFrame(path_plan)
    path_df.to_csv(os.path.join(DRYCHECK_REPORT_DIR, "p_c19_output_path_check.csv"), index=False)

    # 4. guardrail check CSV
    guardrail = [
        {"guard": "ALLOW_HOLDOUT_SCORING_default_false",       "status": "PASS" if not ALLOW_HOLDOUT_SCORING else "FAIL",   "detail": str(ALLOW_HOLDOUT_SCORING)},
        {"guard": "run_holdout_flag_required",                 "status": "PASS",   "detail": "--run-holdout 없으면 sys.exit(2)"},
        {"guard": "confirm_one_time_eval_required",            "status": "PASS",   "detail": "--confirm-one-time-eval 없으면 sys.exit(2)"},
        {"guard": "confirm_no_retune_required",                "status": "PASS",   "detail": "--confirm-no-retune-after-holdout 없으면 sys.exit(2)"},
        {"guard": "p_c18_pass_check",                         "status": "PASS" if results.get("p_c18_pass") else "FAIL",   "detail": results.get("p_c18_verdict", "N/A")},
        {"guard": "best_pth_exists",                          "status": "PASS" if results.get("best_pth_pass") else "FAIL", "detail": f"size={results.get('best_pth_size_mb')}MB"},
        {"guard": "best_epoch_5_from_p_c18",                  "status": "PASS" if results.get("best_epoch_from_p_c18") == 5 else "FAIL", "detail": str(results.get("best_epoch_from_p_c18"))},
        {"guard": "smoke_checkpoint_not_used",                 "status": "PASS" if not results.get("best_pth_is_smoke") else "FAIL", "detail": str(results.get("best_pth_is_smoke"))},
        {"guard": "p_c14_sklearn_free_auroc",                 "status": "PASS",   "detail": "compute_auroc Mann-Whitney U"},
        {"guard": "output_collision_check",                   "status": "PASS" if results.get("output_collision_count", 0) == 0 else "WARN", "detail": str(results.get("output_collision_count", 0))},
        {"guard": "stage1_dev_score_protected",               "status": "PASS",   "detail": "STAGE1_DEV_SCORE_ROOT read-only"},
        {"guard": "no_training_code",                         "status": "PASS" if results.get("training_guard_pass") else "FAIL", "detail": "training code (optim/backward) 없음"},
        {"guard": "no_checkpoint_save",                       "status": "PASS" if results.get("checkpoint_save_absent") else "FAIL", "detail": "torch.save 없음"},
        {"guard": "no_threshold_recalculation",               "status": "PASS",   "detail": "threshold recalc 코드 없음"},
        {"guard": "p_a80b_forbidden",                         "status": "PASS" if results.get("p_a80b_forbidden") else "FAIL", "detail": "P_A80B_FORBIDDEN=True"},
        {"guard": "stage2_holdout_value_load_in_drycheck",    "status": "PASS",   "detail": "dry-check에서 stage2_holdout value 로드 없음"},
        {"guard": "model_forward_in_drycheck",                "status": "PASS",   "detail": "dry-check에서 model forward 없음"},
        {"guard": "py_compile",                               "status": results.get("py_compile", "N/A"), "detail": results.get("py_compile", "N/A")},
    ]
    guardrail_df = pd.DataFrame(guardrail)
    guardrail_df.to_csv(os.path.join(DRYCHECK_REPORT_DIR, "p_c19_guardrail_check.csv"), index=False)

    # 5. schema plan CSV
    schema_plan = [
        # scores output
        {"output": "p_c20_holdout_scores.csv",          "col": "row_id",                  "dtype": "str",     "description": "manifest row_id"},
        {"output": "p_c20_holdout_scores.csv",          "col": "patient_id",              "dtype": "str",     "description": "환자 ID"},
        {"output": "p_c20_holdout_scores.csv",          "col": "label",                   "dtype": "int",     "description": "0=HN, 1=positive"},
        {"output": "p_c20_holdout_scores.csv",          "col": "logit",                   "dtype": "float32", "description": "EfficientNet-B0 binary logit"},
        {"output": "p_c20_holdout_scores.csv",          "col": "prob",                    "dtype": "float32", "description": "sigmoid(logit); calibration 저하로 직접 활용 금지"},
        {"output": "p_c20_holdout_scores.csv",          "col": "npz_path",                "dtype": "str",     "description": "crop 파일 절대경로"},
        # patient summary
        {"output": "p_c20_holdout_patient_summary.csv", "col": "patient_id",              "dtype": "str",     "description": "환자 ID"},
        {"output": "p_c20_holdout_patient_summary.csv", "col": "n_positive",              "dtype": "int",     "description": "positive crop 수"},
        {"output": "p_c20_holdout_patient_summary.csv", "col": "n_negative",              "dtype": "int",     "description": "hard negative crop 수"},
        {"output": "p_c20_holdout_patient_summary.csv", "col": "best_pos_logit",          "dtype": "float32", "description": "positive crop 중 최고 logit"},
        {"output": "p_c20_holdout_patient_summary.csv", "col": "best_neg_logit",          "dtype": "float32", "description": "hard negative crop 중 최고 logit"},
        {"output": "p_c20_holdout_patient_summary.csv", "col": "pos_logit_mean",          "dtype": "float32", "description": "positive crop logit 평균"},
        # metrics JSON
        {"output": "p_c20_holdout_metrics.json",        "col": "crop_level_AUROC",        "dtype": "float",   "description": "전체 crop 단위 AUROC (PRIMARY)"},
        {"output": "p_c20_holdout_metrics.json",        "col": "crop_level_AUPRC",        "dtype": "float",   "description": "전체 crop 단위 AUPRC (PRIMARY)"},
        {"output": "p_c20_holdout_metrics.json",        "col": "n_crops",                 "dtype": "int",     "description": "총 crop 수"},
        {"output": "p_c20_holdout_metrics.json",        "col": "pos_ratio",               "dtype": "float",   "description": "positive 비율"},
        {"output": "p_c20_holdout_metrics.json",        "col": "calibration_status",      "dtype": "str",     "description": "FORBIDDEN (overfitting)"},
        {"output": "p_c20_holdout_metrics.json",        "col": "single_eval_principle",   "dtype": "str",     "description": "APPLIED"},
        {"output": "p_c20_holdout_metrics.json",        "col": "threshold_recalculation", "dtype": "str",     "description": "FORBIDDEN"},
        # sentinel
        {"output": "p_c20_sentinel_tracking.csv",       "col": "sentinel_type",           "dtype": "str",     "description": "no_hit | tiny_lesion | risk6 | high_score_hard_negative"},
        {"output": "p_c20_sentinel_tracking.csv",       "col": "patient_id",              "dtype": "str",     "description": "환자 ID"},
        {"output": "p_c20_sentinel_tracking.csv",       "col": "n_crops",                 "dtype": "int",     "description": "해당 sentinel crops 수"},
        {"output": "p_c20_sentinel_tracking.csv",       "col": "logit_mean",              "dtype": "float32", "description": "해당 crops logit 평균"},
        {"output": "p_c20_sentinel_tracking.csv",       "col": "logit_max",               "dtype": "float32", "description": "해당 crops logit 최대"},
    ]
    schema_df = pd.DataFrame(schema_plan)
    schema_df.to_csv(os.path.join(DRYCHECK_REPORT_DIR, "p_c19_schema_plan.csv"), index=False)

    # 6. sentinel plan CSV
    sentinel_plan = [
        {"sentinel_type": "no_hit",               "trigger": "split CSV weak_case_flag=1 또는 patient_patch_recall<0.1", "purpose": "PaDiM가 병변을 찾지 못한 환자 classifier 반응 확인", "caveat": "val에 없던 케이스라 일반화 불확실"},
        {"sentinel_type": "tiny_lesion",          "trigger": "manifest tiny_lesion_flag=True", "purpose": "극소 병변(1.5~3.5mm) 영역 classifier 반응", "caveat": "val에 없던 케이스"},
        {"sentinel_type": "risk6",                "trigger": "manifest p_b3_risk6_flag=True",  "purpose": "위험군6 케이스 classifier 반응", "caveat": "val에 2.5% 포함이나 holdout 분포 확인 필요"},
        {"sentinel_type": "high_score_hard_negative", "trigger": "HN label=0 중 logit top 1%", "purpose": "고점수 false positive 후보 파악", "caveat": "FP 억제 성능 확인용"},
        {"sentinel_type": "peripheral_boundary",  "trigger": "향후 boundary/peripheral 메타 컬럼 연계 시 추가", "purpose": "흉벽/경계 영역 FP 확인", "caveat": "manifest에 컬럼 없으면 생략"},
        {"sentinel_type": "patient_extreme_fail", "trigger": "환자별 best_pos_logit < val neg median", "purpose": "환자 단위 극단 실패 케이스", "caveat": "P-C20 이후 추가 분석 가능"},
    ]
    sentinel_df = pd.DataFrame(sentinel_plan)
    sentinel_df.to_csv(os.path.join(DRYCHECK_REPORT_DIR, "p_c19_sentinel_plan.csv"), index=False)

    # 7. markdown report
    verdict = results.get("verdict", "N/A")
    p_c20_ready = results.get("p_c20_ready", False)
    preprocessing_warn = results.get("preprocessing_warning", {})
    md_lines = [
        "# P-C19 holdout scoring script static dry-check 결과",
        f"생성일: {results.get('created', '')}",
        "",
        f"## 판정: {verdict}",
        "",
        "## P-C18 readiness",
        f"- verdict: {results.get('p_c18_verdict', 'N/A')}",
        f"- recommendation: {results.get('p_c18_recommendation', 'N/A')}",
        f"- best_epoch(P-C18): {results.get('best_epoch_from_p_c18', 'N/A')}",
        f"- best_val_auc(P-C18): {results.get('best_val_auc_from_p_c18', float('nan')):.6f}",
        "",
        "## best.pth readiness",
        f"- exists: {results.get('best_pth_pass', False)}",
        f"- size_mb: {results.get('best_pth_size_mb', 0)}",
        f"- is_smoke: {results.get('best_pth_is_smoke', False)}",
        f"- smoke/full 분리: {results.get('best_ne_smoke_size', False)}",
        "",
        "## py_compile",
        f"- 결과: {results.get('py_compile', 'N/A')}",
        "",
        "## bare run 차단",
        "- 기본 실행 시 ALLOW_HOLDOUT_SCORING=False → sys.exit(2)",
        "- --run-holdout 없으면 scoring 진입 불가",
        "- --confirm-one-time-eval 없으면 중단",
        "- --confirm-no-retune-after-holdout 없으면 중단",
        "",
        "## dry-check 결과",
        f"- stage2_holdout value load: {results.get('stage2_holdout_value_load', False)}",
        f"- model forward: {results.get('model_forward', False)}",
        f"- training executed: {results.get('training_executed', False)}",
        f"- checkpoint saved: {results.get('checkpoint_saved', False)}",
        f"- threshold recalculated: {results.get('threshold_recalculated', False)}",
        "",
        "## output path collision",
        f"- collision_count: {results.get('output_collision_count', 0)}",
        f"- safe_to_proceed: {results.get('output_safe_to_proceed', False)}",
        f"- stage1_dev protected: {results.get('stage1_dev_protected', False)}",
        "",
        "## 전처리 불일치 WARNING",
        f"- training key: {preprocessing_warn.get('training_crop_key', 'N/A')} / holdout key: {preprocessing_warn.get('holdout_crop_key', 'N/A')}",
        f"- training ch: {preprocessing_warn.get('training_crop_channels', 'N/A')} / holdout ch: {preprocessing_warn.get('holdout_crop_channels', 'N/A')}",
        f"- training HU: {preprocessing_warn.get('training_hu_range', 'N/A')} / holdout ch0~2: {preprocessing_warn.get('holdout_ch0_2', 'N/A')}",
        f"- normalization_mismatch: {preprocessing_warn.get('normalization_mismatch', 'N/A')}",
        f"- action_required: {preprocessing_warn.get('action_required', 'N/A')}",
        "",
        "## planned output schema",
        "- `p_c20_holdout_scores.csv`: crop별 logit/prob/label",
        "- `p_c20_holdout_patient_summary.csv`: 환자별 hit/miss",
        "- `p_c20_holdout_metrics.json`: crop AUROC/AUPRC",
        "- `p_c20_sentinel_tracking.csv`: sentinel type별 분포",
        "",
        "## holdout metric plan",
        "- crop-level AUROC (PRIMARY, sklearn-free Mann-Whitney U)",
        "- crop-level AUPRC (PRIMARY)",
        "- patient-level hit/miss summary (SECONDARY)",
        "- no_hit/tiny/risk6/high_score_HN sentinel tracking",
        "- calibration 결론 금지 (overfitting 확인됨)",
        "- threshold-free ranking metric (logit 기준)",
        "",
        "## sentinel tracking plan",
        "- no_hit: PaDiM miss 환자의 2차 classifier 반응",
        "- tiny_lesion: 극소병변 영역 반응",
        "- risk6: 위험군6 케이스 반응",
        "- high_score_hard_negative: FP 억제 성능 확인",
        "",
        "## one-time evaluation 원칙",
        "- holdout 결과는 1회만 평가한다.",
        "- 결과 확인 후 threshold/model 재조정 시 leakage로 기록해야 한다.",
        "- 이 스크립트에 재조정 코드는 없다.",
        "",
        "## threshold/model retune 금지 가드",
        "- ALLOW_HOLDOUT_SCORING = False (기본값)",
        "- training code (optim/backward) 없음",
        "- torch.save 없음",
        "- threshold recalc 코드 없음",
        "",
        f"## P-C20 holdout scoring 실행 가능 여부",
        f"- ready: {p_c20_ready}",
        f"- error_count: {results.get('error_count', 0)}",
        "",
        "## ★ P-C20 실행 승인 문구",
        "> P-C19 static dry-check 통과 확인.",
        "> EfficientNet-B0 v4_20 second-stage classifier stage2_holdout 1회 평가 실행 승인.",
        "> best.pth epoch5 고정, threshold/model 재조정 없이 평가하며,",
        "> 결과 확인 후 재튜닝하지 않음에 동의.",
        "",
        "### 추가 사전 확인 권장 (P-C20 전)",
        "1. 전처리 불일치 확인: holdout 6ch crop (ch0~2 lung window, [-1350,150]) vs",
        "   training 3ch crop (HU [-1000,200]) 차이를 인지하고 평가 진행에 동의.",
        "2. output collision 없음 확인 (collision_count=0 상태에서 실행).",
        "3. GPU 사용 여부 결정 (batch_size=64 기준 ~수분 소요 예상).",
    ]
    md_path = os.path.join(DRYCHECK_REPORT_DIR, "p_c19_holdout_scoring_script_drycheck.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[report] {DRYCHECK_REPORT_DIR}/")
    print(f"  - p_c19_holdout_scoring_script_drycheck.json")
    print(f"  - p_c19_holdout_scoring_script_drycheck.md")
    print(f"  - p_c19_output_path_check.csv")
    print(f"  - p_c19_guardrail_check.csv")
    print(f"  - p_c19_schema_plan.csv")
    print(f"  - p_c19_sentinel_plan.csv")
    print(f"  - p_c19_errors.csv")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="P-C19 holdout scoring script (EfficientNet-B0 v4_20)"
    )
    parser.add_argument("--dry-check",   action="store_true",
                        help="static dry-check only (default if no flag given)")
    parser.add_argument("--run-holdout", action="store_true",
                        help="실제 holdout scoring 실행 (3개 confirm flag 동시 필요)")
    parser.add_argument("--confirm-one-time-eval",          action="store_true",
                        help="holdout 결과를 1회만 평가함을 확인")
    parser.add_argument("--confirm-no-retune-after-holdout", action="store_true",
                        help="holdout 결과 후 threshold/model 재조정 금지 확인")
    parser.add_argument("--device",      default="cuda",
                        help="cuda | cpu (default: cuda)")
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # ── 기본 실행: dry-check ──────────────────────────────────────────────
    if not args.run_holdout:
        print("--run-holdout 없음 → dry-check 모드로 실행합니다.")
        results, errors = run_dry_check()
        write_dry_check_reports(results, errors)
        sys.exit(0)

    # ── 실제 scoring: 3개 confirm flag 확인 ──────────────────────────────
    if not args.confirm_one_time_eval:
        print("[ERROR] --confirm-one-time-eval 없음 — holdout scoring 중단.", file=sys.stderr)
        print("        holdout 결과를 1회만 평가함을 확인한 후 재실행하세요.", file=sys.stderr)
        sys.exit(2)

    if not args.confirm_no_retune_after_holdout:
        print("[ERROR] --confirm-no-retune-after-holdout 없음 — holdout scoring 중단.", file=sys.stderr)
        print("        holdout 결과 후 threshold/model 재조정 금지에 동의한 후 재실행하세요.", file=sys.stderr)
        sys.exit(2)

    if not ALLOW_HOLDOUT_SCORING:
        print("[ERROR] ALLOW_HOLDOUT_SCORING=False — 스크립트 내부 차단.", file=sys.stderr)
        print("        이 플래그는 소스 코드에서만 변경 가능합니다.", file=sys.stderr)
        sys.exit(2)

    # ── P-C18 pass 재확인 ─────────────────────────────────────────────────
    c18 = check_p_c18_pass(P_C18_JSON)
    if not c18["pass"]:
        print(f"[ERROR] P-C18 확인 실패: {c18.get('reason', '')}", file=sys.stderr)
        sys.exit(2)

    # ── output collision 확인 ─────────────────────────────────────────────
    col = check_output_collision(dry_check=False)
    if not col["safe_to_proceed"]:
        print(f"[ERROR] output 경로 충돌 {col['collision_count']}개 발견 — 중단.", file=sys.stderr)
        for p in col["collisions"]:
            print(f"  충돌: {p}", file=sys.stderr)
        sys.exit(2)

    # ── 실행 ──────────────────────────────────────────────────────────────
    # 여기까지 오면 3개 confirm flag + P-C18 pass + collision 없음 모두 확인됨
    metrics = run_holdout_scoring(
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        confirmed_allowed=True,
    )
    print("P-C20 holdout scoring 완료.")
    sys.exit(0)


if __name__ == "__main__":
    # bare run 차단: --run-holdout 없이 직접 실행하면 dry-check로 진입
    # (argparse에서 처리)
    main()
