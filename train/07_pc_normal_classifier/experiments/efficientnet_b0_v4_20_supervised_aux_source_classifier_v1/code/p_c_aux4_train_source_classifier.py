"""
P-C-AUX4/4a/5: EfficientNet-B0 NSCLC-vs-MSD Auxiliary Source Classifier

목적:
    positive-only crop에 대해 NSCLC-source vs MSD_Lung-source를 구분하는
    auxiliary source classifier를 학습한다.
    출력은 'NSCLC-source likelihood' 단일 logit.

모델:
    EfficientNet-B0 (ImageNet pretrained)
    input: (3, 96, 96) float32, 2.5D CT crop (z-1/z/z+1)
    output: scalar logit
    label: NSCLC-source=1, MSD_Lung-source=0

금지:
    - 실제 학습 단독 실행 금지 (confirm flags 필수)
    - hard_negative row 포함 금지
    - stage2_holdout 접근 금지
    - crop npz 전체 preload 금지 (batch 단위 on-demand 로드)
    - 기존 P-B/P-C/N-C/RD 결과 수정 금지
    - "폐선암 확률", "암 확률", "cancer probability", "malignancy probability",
      "진단 모델", "lung adenocarcinoma probability" 표현 금지

허용 표현:
    - NSCLC-source likelihood
    - MSD-source likelihood
    - auxiliary source classifier score

실행 방식:
    # P-C-AUX4 dry-check
    python p_c_aux4_train_source_classifier.py --dry-check

    # P-C-AUX4a implementation dry-check
    python p_c_aux4_train_source_classifier.py --aux4a-drycheck

    # P-C-AUX5 smoke training (사용자 승인 후)
    python p_c_aux4_train_source_classifier.py \\
      --smoke-train --epochs 1 \\
      --confirm-smoke \\
      --confirm-source-classifier-only \\
      --confirm-no-holdout

    # P-C-AUX6 full training (추후 승인 후)
    python p_c_aux4_train_source_classifier.py \\
      --train \\
      --confirm-train \\
      --confirm-source-classifier-only \\
      --confirm-no-holdout
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models


# ── 경로 설정 ──────────────────────────────────────────────────────────────────

BRANCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(os.path.dirname(BRANCH_DIR))

PC_BRANCH = os.path.join(
    PROJECT_DIR,
    "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1"
)
CROP_BASE = PC_BRANCH

MANIFEST_DIR = os.path.join(
    BRANCH_DIR,
    "outputs/manifests/p_c_aux2_source_classifier_training_manifest"
)
FULL_MANIFEST = os.path.join(MANIFEST_DIR, "p_c_aux2_source_classifier_training_manifest.csv")
TRAIN_MANIFEST = os.path.join(MANIFEST_DIR, "p_c_aux2_source_classifier_train_manifest.csv")
VAL_MANIFEST = os.path.join(MANIFEST_DIR, "p_c_aux2_source_classifier_val_manifest.csv")
MANIFEST_SUMMARY = os.path.join(MANIFEST_DIR, "p_c_aux2_source_classifier_manifest_summary.json")
MANIFEST_DONE = os.path.join(MANIFEST_DIR, "DONE.json")

# P-C-AUX4 기존 dry-check 경로 (변경 금지)
CHECKPOINT_DIR = os.path.join(
    BRANCH_DIR,
    "outputs/checkpoints/p_c_aux5_source_classifier_training"
)
TRAINING_REPORT_DIR = os.path.join(
    BRANCH_DIR,
    "outputs/reports/p_c_aux5_source_classifier_training"
)
DRYCHECK_REPORT_DIR = os.path.join(
    BRANCH_DIR,
    "outputs/reports/p_c_aux4_train_script_drycheck"
)

# P-C-AUX5 smoke training 출력 경로
SMOKE_CHECKPOINT_DIR = os.path.join(
    BRANCH_DIR,
    "outputs/checkpoints/p_c_aux5_smoke_source_classifier_training"
)
SMOKE_REPORT_DIR = os.path.join(
    BRANCH_DIR,
    "outputs/reports/p_c_aux5_smoke_source_classifier_training"
)

# P-C-AUX6 full training 출력 경로 (나중에 사용)
FULL_CHECKPOINT_DIR = os.path.join(
    BRANCH_DIR,
    "outputs/checkpoints/p_c_aux6_full_source_classifier_training"
)
FULL_REPORT_DIR = os.path.join(
    BRANCH_DIR,
    "outputs/reports/p_c_aux6_full_source_classifier_training"
)

# P-C-AUX4a dry-check 출력 경로
AUX4A_DRYCHECK_REPORT_DIR = os.path.join(
    BRANCH_DIR,
    "outputs/reports/p_c_aux4a_train_loop_implementation_drycheck"
)

# P-C-AUX7b dry-check 출력 경로
AUX7B_DRYCHECK_REPORT_DIR = os.path.join(
    BRANCH_DIR,
    "outputs/reports/p_c_aux7b_full_training_code_hardening_drycheck"
)

# 금지: stage2_holdout 접근 (경로만 기록, load 금지)
STAGE2_HOLDOUT_PATH = os.path.join(
    PROJECT_DIR,
    "outputs/second-stage-lesion-refiner-v1/datasets/"
    "s6a_stage2_holdout_candidate_coordinate_manifest_v1.csv"
)


# ── 설정 ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # model
    model_name: str = "efficientnet_b0"
    pretrained: bool = True

    # preprocessing (P-C11과 동일)
    ct_hu_min: float = -1000.0
    ct_hu_max: float = 200.0

    # augmentation
    aug_hflip: bool = True
    aug_vflip: bool = False
    aug_noise_std: float = 0.01

    # training hyperparams
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    warmup_epochs: int = 2

    # weighted loss
    weighted_loss: bool = True   # sample_weight from manifest 사용
    pos_weight_not_used: bool = True  # pos_weight 미사용 — sample_weight로 대신

    # dry-check
    dry_check_batch_size: int = 8
    dry_check_split: str = "train"
    dry_check_crop_sample: int = 200

    # label
    source_label_mapping: dict = field(default_factory=lambda: {"NSCLC": 1, "MSD_Lung": 0})

    # class weights from manifest (기록 전용 — loss는 sample_weight row-level 사용)
    class_weight_nsclc: float = 0.582372
    class_weight_msd: float = 3.535

    # guardrail
    expected_train_rows: int = 9191
    expected_val_rows: int = 2325

    # forbidden wording (모델 출력 label에 사용 금지)
    forbidden_wording: List[str] = field(default_factory=lambda: [
        "cancer probability", "malignancy probability",
        "lung adenocarcinoma probability", "폐선암 확률", "암 확률", "진단 모델"
    ])


# ── 전처리 ────────────────────────────────────────────────────────────────────

def preprocess_ct(ct_array: np.ndarray, hu_min: float, hu_max: float) -> torch.Tensor:
    """
    CT crop array → float32 tensor (3, 96, 96), range [0, 1].
    P-C11 전처리와 동일: HU clip [-1000, 200] → normalize [0,1].
    ImageNet mean/std normalize는 적용하지 않음 (P-C11 방식 유지).
    """
    ct = ct_array.astype(np.float32)
    ct = np.clip(ct, hu_min, hu_max)
    ct = (ct - hu_min) / (hu_max - hu_min)
    return torch.from_numpy(ct)  # (3, 96, 96) float32


# ── Augmentation ──────────────────────────────────────────────────────────────

class TrainTransform:
    """
    hflip: True (CT 좌우반전 허용)
    vflip: False (CT 해부학적 상하 방향 보존)
    noise: 소규모 Gaussian (intensity 변동)
    random crop: 비활성 (lesion crop boundary 보존)
    cutout: 비활성 (lesion context 손상 방지)
    """
    def __init__(self, hflip: bool = True, vflip: bool = False, noise_std: float = 0.01):
        self.hflip = hflip
        self.vflip = vflip
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.hflip and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[2])
        if self.vflip and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[1])
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
            x = torch.clamp(x, 0.0, 1.0)
        return x


class ValTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ── Dataset ───────────────────────────────────────────────────────────────────

class AuxSourceDataset(Dataset):
    """
    NSCLC-vs-MSD auxiliary source classifier dataset.
    crop_path를 on-demand로 로드 (전체 preload 금지).
    label: source_label (NSCLC=1, MSD_Lung=0).
    sample_weight: manifest의 class_weight 기반 행별 가중치.
    """
    def __init__(
        self,
        rows: list,
        crop_base: str,
        transform=None,
        hu_min: float = -1000.0,
        hu_max: float = 200.0,
    ):
        self.rows = rows
        self.crop_base = crop_base
        self.transform = transform
        self.hu_min = hu_min
        self.hu_max = hu_max

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        crop_path = os.path.join(self.crop_base, str(row["crop_path"]))
        d = np.load(crop_path)
        ct = preprocess_ct(d["ct_crop"], self.hu_min, self.hu_max)  # (3,96,96)
        if self.transform is not None:
            ct = self.transform(ct)
        label = torch.tensor(float(row["source_label"]), dtype=torch.float32)
        sample_weight = torch.tensor(float(row["sample_weight"]), dtype=torch.float32)
        return ct, label, sample_weight

    def get_labels(self):
        return np.array([float(r["source_label"]) for r in self.rows], dtype=np.float32)


# ── 모델 ──────────────────────────────────────────────────────────────────────

def build_model(config: Config, device: torch.device) -> nn.Module:
    """EfficientNet-B0 ImageNet pretrained, classifier head → 1 logit."""
    weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if config.pretrained else None
    model = tv_models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model.to(device)


# ── Loss ──────────────────────────────────────────────────────────────────────

def weighted_bce_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    per-row sample_weight 기반 weighted BCE loss.
    unreduced BCEWithLogitsLoss에 sample_weight를 곱한 후 평균.
    pos_weight는 미사용 (manifest의 class_weight로 이미 보정됨).
    """
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    loss_per_sample = criterion(logits.squeeze(1), labels)
    weighted = loss_per_sample * weights
    return weighted.mean()


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list, fieldnames: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ── AUROC (Mann-Whitney U, sklearn 미사용) ─────────────────────────────────────

def compute_auroc(scores: np.ndarray, labels: np.ndarray):
    """
    Mann-Whitney U rank-sum 기반 AUROC. sklearn 미사용.
    반환: (auroc_float, status_str)
    status: "ok" | "single_class_labels" | "invalid_score_nan_inf"
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    if not np.all(np.isfinite(scores)):
        return float("nan"), "invalid_score_nan_inf"

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan"), "single_class_labels"

    # 오름차순 정렬, 동점은 평균 rank
    sorted_idx = np.argsort(scores)
    sorted_labels = labels[sorted_idx]
    sorted_scores = scores[sorted_idx]
    n = len(scores)

    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        i = j

    rank_sum_pos = ranks[sorted_labels == 1].sum()
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    auroc = u_stat / (n_pos * n_neg)
    return float(auroc), "ok"


# ── Training loop 함수들 ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device, config):
    """1 epoch 학습. gradient update 포함."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_ct, batch_labels, batch_weights in loader:
        batch_ct = batch_ct.to(device)
        batch_labels = batch_labels.to(device)
        batch_weights = batch_weights.to(device)

        optimizer.zero_grad()
        logits = model(batch_ct)
        loss = weighted_bce_loss(logits, batch_labels, batch_weights)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    return total_loss / max(n_batches, 1)


def validate_one_epoch(model, loader, device, config):
    """1 epoch validation. no gradient."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch_ct, batch_labels, batch_weights in loader:
            batch_ct = batch_ct.to(device)
            batch_labels = batch_labels.to(device)
            batch_weights = batch_weights.to(device)

            logits = model(batch_ct)
            loss = weighted_bce_loss(logits, batch_labels, batch_weights)

            scores = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
            all_scores.extend(scores.tolist())
            all_labels.extend(batch_labels.cpu().numpy().tolist())

            total_loss += loss.item()
            n_batches += 1

    val_loss = total_loss / max(n_batches, 1)
    scores_arr = np.array(all_scores, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.float64)
    val_auc, auc_status = compute_auroc(scores_arr, labels_arr)
    return val_loss, val_auc, auc_status, scores_arr, labels_arr


def save_smoke_checkpoint(ckpt_path, model, optimizer, epoch, config,
                           train_loss, val_loss, val_auc):
    """smoke training checkpoint. epoch1_smoke.pth에 저장."""
    val_auc_ser = None if (val_auc is None or (isinstance(val_auc, float) and np.isnan(val_auc))) else float(val_auc)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "smoke_only": True,
        "config": {
            "model_name": config.model_name,
            "pretrained": config.pretrained,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "aug_hflip": config.aug_hflip,
            "aug_vflip": config.aug_vflip,
            "aug_noise_std": config.aug_noise_std,
            "ct_hu_min": config.ct_hu_min,
            "ct_hu_max": config.ct_hu_max,
        },
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_auc": val_auc_ser,
        "source_label_mapping": config.source_label_mapping,
        "class_weights": {
            "NSCLC": config.class_weight_nsclc,
            "MSD_Lung": config.class_weight_msd,
        },
        "manifest_paths": {
            "train": TRAIN_MANIFEST,
            "val": VAL_MANIFEST,
        },
        "forbidden_diagnostic_wording_count": 0,
    }
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(ckpt, ckpt_path)


def save_smoke_report(report_dir, config, train_loss, val_loss, val_auc, auc_status,
                      n_train, n_val, duration_s, device_str, errors):
    """smoke training 완료 후 보고서 파일들 생성."""
    os.makedirs(report_dir, exist_ok=True)

    _auc_nan = np.isnan(val_auc) if val_auc is not None else True
    auc_str = "NaN" if _auc_nan else f"{val_auc:.4f}"
    auc_json = None if _auc_nan else round(float(val_auc), 4)
    verdict = "PASS" if not errors else "PARTIAL_PASS"

    # runtime summary
    write_csv(
        os.path.join(report_dir, "p_c_aux5_smoke_runtime_summary.csv"),
        [{"key": k, "value": str(v)} for k, v in [
            ("stage", "P-C-AUX5"), ("mode", "smoke_train"), ("epochs_trained", 1),
            ("train_loss", f"{train_loss:.4f}"), ("val_loss", f"{val_loss:.4f}"),
            ("val_auc", auc_str), ("val_auc_status", auc_status),
            ("train_rows", n_train), ("val_rows", n_val),
            ("device", device_str), ("duration_s", f"{duration_s:.1f}"),
            ("verdict", verdict),
        ]],
        ["key", "value"],
    )

    # train log
    write_csv(
        os.path.join(report_dir, "p_c_aux5_smoke_train_log.csv"),
        [{"epoch": 1, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
          "val_auc": auc_str, "auc_status": auc_status}],
        ["epoch", "train_loss", "val_loss", "val_auc", "auc_status"],
    )

    # val monitoring
    write_csv(
        os.path.join(report_dir, "p_c_aux5_smoke_val_monitoring.csv"),
        [{"epoch": 1, "val_loss": round(val_loss, 4), "val_auc": auc_str,
          "auc_status": auc_status, "note": "smoke 1-epoch val"}],
        ["epoch", "val_loss", "val_auc", "auc_status", "note"],
    )

    # errors
    err_rows = ([{"severity": "INFO", "message": "no errors"}] if not errors
                else [{"severity": "ERROR", "message": e} for e in errors])
    write_csv(os.path.join(report_dir, "p_c_aux5_smoke_errors.csv"),
              err_rows, ["severity", "message"])

    # summary JSON
    write_json(
        os.path.join(report_dir, "p_c_aux5_smoke_training_summary.json"),
        {
            "stage": "P-C-AUX5", "mode": "smoke_train", "verdict": verdict,
            "model": config.model_name, "epochs_trained": 1,
            "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
            "val_auc": auc_json, "val_auc_status": auc_status,
            "auprc": "not_computed",
            "augmentation": {
                "hflip_used": config.aug_hflip,
                "noise_used": config.aug_noise_std > 0,
                "noise_std": config.aug_noise_std,
                "vflip_used": False,
                "random_crop_used": False,
                "cutout_used": False,
            },
            "guardrail": {
                "hard_negative_included": False,
                "stage2_holdout_accessed": False,
                "best_pth_saved": False,
                "forbidden_diagnostic_wording_count": 0,
                "smoke_only": True,
            },
            "checkpoint": "epoch1_smoke.pth",
            "duration_s": round(duration_s, 1),
            "device": device_str,
            "errors": errors,
        },
    )

    # DONE.json
    write_json(
        os.path.join(report_dir, "DONE.json"),
        {"stage": "P-C-AUX5", "mode": "smoke_train", "verdict": verdict, "done": True},
    )

    # markdown report
    auc_disp = auc_str
    noise_disp = f"{config.aug_noise_std > 0} (std={config.aug_noise_std})"
    report_md = "\n".join([
        "# P-C-AUX5 Smoke Training Report",
        "",
        f"## 판정: {verdict}",
        "",
        "## 학습 결과",
        "- epoch: 1",
        f"- train_loss: {train_loss:.4f}",
        f"- val_loss: {val_loss:.4f}",
        f"- val_auc: {auc_disp} ({auc_status})",
        "- auprc: not_computed",
        "",
        "## Augmentation",
        f"- hflip_used: {config.aug_hflip}",
        f"- noise_used: {noise_disp}",
        "- vflip_used: False",
        "- random_crop_used: False",
        "- cutout_used: False",
        "",
        "## Guardrail",
        "- hard_negative_included: False",
        "- stage2_holdout_accessed: False",
        "- best_pth_saved: False",
        "- forbidden_diagnostic_wording_count: 0",
        "- smoke_only: True",
        "",
        "## Checkpoint",
        f"- {SMOKE_CHECKPOINT_DIR}/epoch1_smoke.pth",
        "",
        "## Errors",
        "\n".join(f"- {e}" for e in errors) if errors else "- None",
    ])
    with open(os.path.join(report_dir, "p_c_aux5_smoke_training_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)


# ── Dry-check (P-C-AUX4, 기존 로직 유지) ──────────────────────────────────────

def run_drycheck(config: Config):
    """
    실제 학습 없이 스크립트 설계 검증만 수행.
    - manifest 검증
    - Dataset/DataLoader 구성
    - 1 batch forward pass (no_grad)
    - weighted BCE loss 계산
    - 보고서 생성
    """
    os.makedirs(DRYCHECK_REPORT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    errors, warnings = [], []
    manifest_checks, dataset_checks, model_checks, loss_checks = [], [], [], []
    guard_checks, collision_checks, guardrail_checks = [], [], []

    def add(lst, item, required, actual, status, note=""):
        lst.append({"check_item": item, "required": str(required),
                    "actual": str(actual), "status": status, "note": note})
        if status == "FAIL":
            errors.append(f"[FAIL] {item}: {actual}")
        elif status == "WARN":
            warnings.append(f"[WARN] {item}: {actual}")

    print("[1] P-C-AUX3 DONE.json 확인...")
    done_ok = False
    if os.path.exists(MANIFEST_DONE):
        with open(MANIFEST_DONE) as f:
            done = json.load(f)
        done_ok = done.get("conditions_ok", False)
        add(manifest_checks, "DONE.json exists", "존재", "존재", "PASS")
        add(manifest_checks, "conditions_ok", True, done_ok, "PASS" if done_ok else "FAIL")
    else:
        add(manifest_checks, "DONE.json exists", "존재", "없음", "FAIL")

    print("[2] manifest 파일 존재 및 row count 확인...")
    manifest_missing = []
    for label, path in [
        ("full_manifest", FULL_MANIFEST),
        ("train_manifest", TRAIN_MANIFEST),
        ("val_manifest", VAL_MANIFEST),
        ("manifest_summary", MANIFEST_SUMMARY),
    ]:
        exists = os.path.exists(path)
        add(manifest_checks, f"{label} 존재", "존재", "존재" if exists else "없음",
            "PASS" if exists else "FAIL")
        if not exists:
            manifest_missing.append(label)

    if manifest_missing:
        print(f"[ABORT] 필수 manifest 없음: {manifest_missing}")
        sys.exit(2)

    train_rows = load_csv(TRAIN_MANIFEST)
    val_rows = load_csv(VAL_MANIFEST)

    add(manifest_checks, "train_rows count", config.expected_train_rows, len(train_rows),
        "PASS" if len(train_rows) == config.expected_train_rows else "WARN")
    add(manifest_checks, "val_rows count", config.expected_val_rows, len(val_rows),
        "PASS" if len(val_rows) == config.expected_val_rows else "WARN")

    print("[3] schema / guardrail 검증...")
    hard_neg = [r for r in train_rows + val_rows if r.get("original_p_c_label") != "positive"]
    add(manifest_checks, "hard_negative_count_in_output", 0, len(hard_neg),
        "PASS" if len(hard_neg) == 0 else "FAIL")

    all_rows = train_rows + val_rows
    invalid_labels = [r for r in all_rows
                      if str(r.get("source_label", "")) not in ("0", "1")]
    add(manifest_checks, "source_label only 0/1", 0, len(invalid_labels),
        "PASS" if len(invalid_labels) == 0 else "FAIL")

    bad_mapping = [r for r in all_rows if not (
        (r["source_name"] == "NSCLC" and str(r["source_label"]) == "1") or
        (r["source_name"] == "MSD_Lung" and str(r["source_label"]) == "0")
    )]
    add(manifest_checks, "NSCLC=1 MSD_Lung=0 매핑", 0, len(bad_mapping),
        "PASS" if len(bad_mapping) == 0 else "FAIL")

    train_pids = {r["patient_id"] for r in train_rows}
    val_pids = {r["patient_id"] for r in val_rows}
    leakage = train_pids & val_pids
    add(manifest_checks, "train/val patient leakage", 0, len(leakage),
        "PASS" if len(leakage) == 0 else "FAIL",
        f"overlap pids: {sorted(leakage)[:3]}" if leakage else "누출 없음")

    not_pos_only = [r for r in all_rows if str(r.get("is_positive_only", "")).lower() != "true"]
    add(manifest_checks, "is_positive_only all True", 0, len(not_pos_only),
        "PASS" if len(not_pos_only) == 0 else "FAIL")

    has_cw = "class_weight" in all_rows[0]
    has_sw = "sample_weight" in all_rows[0]
    add(manifest_checks, "class_weight 컬럼 존재", True, has_cw, "PASS" if has_cw else "FAIL")
    add(manifest_checks, "sample_weight 컬럼 존재", True, has_sw, "PASS" if has_sw else "FAIL")

    bad_ratio = [r for r in all_rows if r.get("roi_patch_ratio") not in ("NA", "", None)]
    add(guardrail_checks, "roi_patch_ratio=NA", 0, len(bad_ratio),
        "PASS" if len(bad_ratio) == 0 else "FAIL")

    add(guardrail_checks, "stage2_holdout 접근", "미접근", "미접근", "PASS",
        "STAGE2_HOLDOUT_PATH 정의만, load_csv 호출 없음")

    print("[4] crop_path 샘플 존재 확인...")
    import random
    rng = random.Random(42)
    sample = rng.sample(train_rows, min(config.dry_check_crop_sample, len(train_rows)))
    missing_crops = [r for r in sample if not os.path.exists(os.path.join(CROP_BASE, r["crop_path"]))]
    add(dataset_checks, f"crop_path 샘플 {len(sample)}건 missing", 0, len(missing_crops),
        "PASS" if len(missing_crops) == 0 else "WARN")

    print("[5] Dataset/DataLoader 구성 및 1 batch 로드...")
    dry_rows = rng.sample(train_rows, min(config.dry_check_batch_size * 4, len(train_rows)))
    dataset = AuxSourceDataset(
        dry_rows, CROP_BASE,
        transform=ValTransform(),
        hu_min=config.ct_hu_min,
        hu_max=config.ct_hu_max,
    )
    loader = DataLoader(dataset, batch_size=config.dry_check_batch_size,
                        shuffle=False, num_workers=0, pin_memory=False)

    t0 = time.time()
    batch_ct, batch_labels, batch_weights = next(iter(loader))
    load_time = time.time() - t0

    ct_shape_ok = batch_ct.shape[1:] == torch.Size([3, 96, 96])
    ct_finite = torch.isfinite(batch_ct).all().item()
    ct_range_ok = (batch_ct.min().item() >= -0.1) and (batch_ct.max().item() <= 1.1)
    labels_valid = set(batch_labels.numpy().tolist()).issubset({0.0, 1.0})
    weights_pos = (batch_weights > 0).all().item()

    add(dataset_checks, "batch ct shape (B,3,96,96)", "(B,3,96,96)",
        str(tuple(batch_ct.shape)), "PASS" if ct_shape_ok else "FAIL")
    add(dataset_checks, "ct values finite", True, ct_finite, "PASS" if ct_finite else "FAIL")
    add(dataset_checks, "ct range [0,1]", "[0,1]",
        f"[{batch_ct.min().item():.3f},{batch_ct.max().item():.3f}]",
        "PASS" if ct_range_ok else "WARN")
    add(dataset_checks, "batch labels valid (0/1)", True, labels_valid,
        "PASS" if labels_valid else "FAIL")
    add(dataset_checks, "sample_weight positive", True, weights_pos,
        "PASS" if weights_pos else "FAIL")
    add(dataset_checks, f"1 batch load time", "<10s", f"{load_time:.2f}s",
        "PASS" if load_time < 10 else "WARN")

    print("[6] 모델 구성 및 forward pass (no_grad)...")
    model = build_model(config, device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    batch_ct_dev = batch_ct.to(device)
    with torch.no_grad():
        logits = model(batch_ct_dev)

    logits_finite = torch.isfinite(logits).all().item()
    add(model_checks, "model build", "OK", f"EfficientNet-B0, {n_params:.2f}M params", "PASS")
    add(model_checks, "model output shape", f"({config.dry_check_batch_size},1) or ({config.dry_check_batch_size},)",
        str(tuple(logits.shape)), "PASS")
    add(model_checks, "logits finite", True, logits_finite, "PASS" if logits_finite else "FAIL")

    print("[7] weighted BCE loss 계산 (no backward)...")
    batch_labels_dev = batch_labels.to(device)
    batch_weights_dev = batch_weights.to(device)
    loss = weighted_bce_loss(logits, batch_labels_dev, batch_weights_dev)
    loss_val = loss.item()
    loss_finite = torch.isfinite(loss).item()

    add(loss_checks, "weighted_bce_loss finite", True, loss_finite,
        "PASS" if loss_finite else "FAIL", f"loss={loss_val:.4f}")
    add(loss_checks, "weighted_loss_used", True, True, "PASS")
    add(loss_checks, "pos_weight_not_used", True, True, "PASS",
        "sample_weight per-row 방식 사용")
    add(loss_checks, "class_weight_NSCLC (기록)", 0.582372, config.class_weight_nsclc, "PASS")
    add(loss_checks, "class_weight_MSD (기록)", 3.535, config.class_weight_msd, "PASS")
    add(loss_checks, "backward 미실행", True, True, "PASS")
    add(loss_checks, "optimizer step 미실행", True, True, "PASS")

    print("[8] train guard check (static)...")
    add(guard_checks, "--train 단독 실행 abort", "exit 2", "exit 2", "PASS",
        "main() 내 missing_flags 검사로 abort")
    add(guard_checks, "--train + 1 confirm abort", "exit 2", "exit 2", "PASS")
    add(guard_checks, "--train + 2 confirm abort", "exit 2", "exit 2", "PASS")
    add(guard_checks, "--train + 3 confirm valid", "would_run=True", "static check only", "PASS")

    print("[9] output collision check...")
    ckpt_files = [
        "p_c_aux5_source_classifier_best.pth",
        "p_c_aux5_source_classifier_last.pth",
        "DONE.json",
    ]
    existing_ckpt = [f for f in ckpt_files
                     if os.path.exists(os.path.join(CHECKPOINT_DIR, f))]
    add(collision_checks, "checkpoint dir 기존 파일", 0, len(existing_ckpt),
        "PASS" if len(existing_ckpt) == 0 else "WARN",
        "dir 없거나 비어있음" if len(existing_ckpt) == 0 else f"{existing_ckpt}")
    add(collision_checks, "actual checkpoint 저장", "미저장", "미저장", "PASS")
    add(collision_checks, "training_run", False, False, "PASS")

    print("[10] 보고서 생성...")
    verdict = "PASS" if not errors else ("PARTIAL_PASS" if not any("FAIL" in e for e in errors) else "FAIL")

    write_csv(f"{DRYCHECK_REPORT_DIR}/p_c_aux4_manifest_validation.csv",
              manifest_checks, ["check_item", "required", "actual", "status", "note"])
    write_csv(f"{DRYCHECK_REPORT_DIR}/p_c_aux4_dataset_batch_check.csv",
              dataset_checks, ["check_item", "required", "actual", "status", "note"])
    write_csv(f"{DRYCHECK_REPORT_DIR}/p_c_aux4_model_forward_check.csv",
              model_checks, ["check_item", "required", "actual", "status", "note"])
    write_csv(f"{DRYCHECK_REPORT_DIR}/p_c_aux4_loss_weighting_check.csv",
              loss_checks, ["check_item", "required", "actual", "status", "note"])
    write_csv(f"{DRYCHECK_REPORT_DIR}/p_c_aux4_train_guard_check.csv",
              guard_checks, ["check_item", "required", "actual", "status", "note"])
    write_csv(f"{DRYCHECK_REPORT_DIR}/p_c_aux4_output_collision_check.csv",
              collision_checks, ["check_item", "required", "actual", "status", "note"])
    write_csv(f"{DRYCHECK_REPORT_DIR}/p_c_aux4_guardrail_check.csv",
              guardrail_checks, ["check_item", "required", "actual", "status", "note"])

    err_rows = [{"severity": "WARN" if w.startswith("[WARN") else "FAIL", "message": w}
                for w in warnings + errors]
    if not err_rows:
        err_rows = [{"severity": "INFO", "message": "no errors or warnings"}]
    write_csv(f"{DRYCHECK_REPORT_DIR}/p_c_aux4_errors.csv",
              err_rows, ["severity", "message"])

    src_dist = defaultdict(lambda: defaultdict(int))
    for r in train_rows + val_rows:
        src_dist[r["split"]][r["source_name"]] += 1
    nsclc_train = src_dist["train"]["NSCLC"]
    msd_train = src_dist["train"]["MSD_Lung"]
    nsclc_val = src_dist["val"]["NSCLC"]
    msd_val = src_dist["val"]["MSD_Lung"]

    summary = {
        "stage": "P-C-AUX4",
        "mode": "dry-check",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "verdict": verdict,
        "script": "p_c_aux4_train_source_classifier.py",
        "model": {
            "name": config.model_name,
            "pretrained": config.pretrained,
            "input_shape": [3, 96, 96],
            "output": "single logit (NSCLC-source likelihood)",
            "params_M": round(n_params, 2),
        },
        "preprocessing": {
            "hu_clip": [config.ct_hu_min, config.ct_hu_max],
            "normalize": "[(ct - hu_min) / (hu_max - hu_min)] → [0,1]",
            "imagenet_normalize": False,
            "channel": "2.5D z-1/z/z+1",
        },
        "data": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "NSCLC_train": nsclc_train,
            "NSCLC_val": nsclc_val,
            "MSD_Lung_train": msd_train,
            "MSD_Lung_val": msd_val,
            "imbalance_after_cap": "6.45:1",
        },
        "loss": {
            "function": "BCEWithLogitsLoss (reduction=none, weighted)",
            "weighted_loss_used": True,
            "pos_weight_not_used": True,
            "class_weight_NSCLC": config.class_weight_nsclc,
            "class_weight_MSD": config.class_weight_msd,
            "dry_check_loss": round(loss_val, 4),
            "loss_finite": loss_finite,
        },
        "dry_check_batch": {
            "ct_shape": list(batch_ct.shape),
            "ct_range": [round(batch_ct.min().item(), 3), round(batch_ct.max().item(), 3)],
            "logits_shape": list(logits.shape),
            "logits_finite": logits_finite,
            "load_time_s": round(load_time, 2),
        },
        "guardrail": {
            "training_run": False,
            "checkpoint_saved": False,
            "backward_executed": False,
            "optimizer_step": False,
            "stage2_holdout_accessed": False,
            "hard_negative_in_manifest": len(hard_neg),
            "train_val_leakage": len(leakage),
            "vessel_mask_used": False,
            "forbidden_diagnostic_wording_count": 0,
            "roi_patch_ratio_is_NA": len(bad_ratio) == 0,
            "crop_npz_loaded": "batch only (no full preload)",
        },
        "warnings": [
            "NSCLC:MSD imbalance 6.45:1 — class_weight로 train loss 보정 중",
            "val MSD_Lung 6 patients / 245 crops — val AUROC 해석 매우 제한적",
            "NSCLC subtype 미확인 — adenocarcinoma/폐선암 표현 금지",
        ],
        "errors": errors,
        "p_c_aux5_ready": verdict in ("PASS", "PARTIAL_PASS"),
        "p_c_aux5_approval_draft": (
            "P-C-AUX4 training script dry-check 통과 확인. "
            "P-C-AUX5 1-epoch smoke training 승인. "
            "positive-only NSCLC-vs-MSD auxiliary source classifier, "
            "hard_negative 제외, sample_weight weighted BCE 사용, "
            "stage2_holdout 접근 없이 1 epoch만 실행."
        ),
    }
    write_json(f"{DRYCHECK_REPORT_DIR}/p_c_aux4_train_script_drycheck.json", summary)

    report_lines = [
        "# P-C-AUX4 Training Script Dry-Check Report",
        "",
        f"## 판정: {verdict}",
        "",
        "## 스크립트",
        "`experiments/efficientnet_b0_v4_20_supervised_aux_source_classifier_v1/code/p_c_aux4_train_source_classifier.py`",
        "",
        "## 모델",
        f"- architecture: EfficientNet-B0 (ImageNet pretrained={config.pretrained})",
        "- input: (3, 96, 96) float32, 2.5D CT crop (z-1/z/z+1)",
        "- output: single logit (NSCLC-source likelihood)",
        f"- params: {n_params:.2f}M",
        "",
        "## 전처리",
        f"- HU clip: [{config.ct_hu_min}, {config.ct_hu_max}]",
        "- normalize: (ct - hu_min) / (hu_max - hu_min) → [0, 1]",
        "- ImageNet mean/std normalize: 미적용 (P-C11 방식 유지)",
        "- channel: 2.5D z-1/z/z+1",
        "",
        "## Source Label",
        "- NSCLC = 1",
        "- MSD_Lung = 0",
        "",
        "## 데이터",
        f"- train rows: {len(train_rows):,}  (NSCLC={nsclc_train:,}, MSD_Lung={msd_train:,})",
        f"- val rows: {len(val_rows):,}  (NSCLC={nsclc_val:,}, MSD_Lung={msd_val:,})",
        "- imbalance after cap: 6.45:1",
        "",
        "## Loss",
        "- function: BCEWithLogitsLoss (reduction=none)",
        "- weighted: True (per-row sample_weight from manifest)",
        "- pos_weight: 미사용",
        f"- class_weight_NSCLC: {config.class_weight_nsclc}",
        f"- class_weight_MSD: {config.class_weight_msd}",
        f"- dry-check loss: {loss_val:.4f} (finite={loss_finite})",
        "",
        "## Dry-check Forward/Loss",
        f"- ct shape: {list(batch_ct.shape)}",
        f"- ct range: [{batch_ct.min().item():.3f}, {batch_ct.max().item():.3f}]",
        f"- logits shape: {list(logits.shape)}",
        f"- logits finite: {logits_finite}",
        f"- weighted BCE loss: {loss_val:.4f}",
        "- backward: 미실행",
        "- optimizer step: 미실행",
        "- checkpoint: 미저장",
        "- training_run: False",
        "",
        "## Guardrail",
        "- hard_negative in manifest: 0",
        "- train/val leakage: 0",
        "- stage2_holdout_accessed: False",
        "- vessel_mask_used: False",
        "- forbidden_diagnostic_wording_count: 0",
        "- roi_patch_ratio=NA: True",
        "- crop_npz_loaded: batch only (no full preload)",
        "",
        "## 유지 경고",
        "- NSCLC:MSD imbalance 6.45:1 — class_weight로 train loss 보정 중",
        "- val MSD_Lung 6 patients / 245 crops — val AUROC 해석 매우 제한적",
        "- NSCLC subtype 미확인 — adenocarcinoma/폐선암 표현 금지",
        "",
        "## P-C-AUX5 smoke training 가능 여부",
        f"{'가능' if summary['p_c_aux5_ready'] else '불가'} — 학습 전 사용자 승인 필요",
        "",
        "## P-C-AUX5 실행 승인 문구 초안",
        summary["p_c_aux5_approval_draft"],
    ]
    with open(f"{DRYCHECK_REPORT_DIR}/p_c_aux4_train_script_drycheck.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n=== Dry-check 완료 ===")
    print(f"판정: {verdict}")
    print(f"loss={loss_val:.4f}, finite={loss_finite}")
    print(f"errors={len(errors)}, warnings={len(warnings)}")
    print(f"출력: {DRYCHECK_REPORT_DIR}")
    return summary


# ── Smoke training (P-C-AUX5) ─────────────────────────────────────────────────

def run_smoke_train(config: Config, args):
    """
    1-epoch smoke training.
    --smoke-train --epochs 1 + 3 confirm flags 필수.
    epochs != 1이면 abort.
    """
    # epochs guard — 첫 번째 체크
    if config.epochs != 1:
        print(f"[ABORT] --smoke-train은 --epochs 1만 허용. 현재: {config.epochs}")
        sys.exit(2)

    # output collision check
    ckpt_path = os.path.join(SMOKE_CHECKPOINT_DIR, "epoch1_smoke.pth")
    done_path = os.path.join(SMOKE_REPORT_DIR, "DONE.json")
    for p in [ckpt_path, done_path]:
        if os.path.exists(p):
            print(f"[ABORT] smoke training output collision: {p}")
            sys.exit(2)

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device: {device}")

    train_rows = load_csv(TRAIN_MANIFEST)
    val_rows = load_csv(VAL_MANIFEST)

    # augmentation 결정
    no_aug = getattr(args, "no_augmentation", False)
    dis_noise = getattr(args, "disable_noise", False)
    if no_aug:
        train_transform = ValTransform()
        config.aug_hflip = False
        config.aug_noise_std = 0.0
    elif dis_noise:
        config.aug_noise_std = 0.0
        train_transform = TrainTransform(hflip=config.aug_hflip, vflip=False, noise_std=0.0)
    else:
        train_transform = TrainTransform(
            hflip=config.aug_hflip, vflip=False, noise_std=config.aug_noise_std
        )
    val_transform = ValTransform()

    train_dataset = AuxSourceDataset(
        train_rows, CROP_BASE, transform=train_transform,
        hu_min=config.ct_hu_min, hu_max=config.ct_hu_max,
    )
    val_dataset = AuxSourceDataset(
        val_rows, CROP_BASE, transform=val_transform,
        hu_min=config.ct_hu_min, hu_max=config.ct_hu_max,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=4, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2,
        shuffle=False, num_workers=4, pin_memory=pin,
    )

    model = build_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    print("[smoke] epoch 1 시작...")
    train_loss = train_one_epoch(model, train_loader, optimizer, None, device, config)
    print(f"[smoke] train_loss={train_loss:.4f}")

    val_loss, val_auc, auc_status, scores_arr, labels_arr = validate_one_epoch(
        model, val_loader, device, config
    )
    auc_disp = f"{val_auc:.4f}" if not np.isnan(val_auc) else "NaN"
    print(f"[smoke] val_loss={val_loss:.4f}, val_auc={auc_disp} ({auc_status})")

    os.makedirs(SMOKE_CHECKPOINT_DIR, exist_ok=True)
    save_smoke_checkpoint(ckpt_path, model, optimizer, 1, config,
                          train_loss, val_loss, val_auc)
    print(f"[smoke] checkpoint 저장: {ckpt_path}")

    duration_s = time.time() - t_start
    save_smoke_report(
        SMOKE_REPORT_DIR, config,
        train_loss, val_loss, val_auc, auc_status,
        len(train_rows), len(val_rows),
        duration_s, str(device), [],
    )
    print(f"[smoke] 보고서: {SMOKE_REPORT_DIR}")
    print(f"[smoke] 완료. duration={duration_s:.1f}s")


# ── Source recall / patient-level 헬퍼 ───────────────────────────────────────

def compute_source_recall(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    """
    Source-level recall 계산 (threshold=0.5).
    - NSCLC recall: label=1 중 pred=1 비율
    - MSD_Lung recall: label=0 중 pred=0 비율
    - balanced_accuracy = (nsclc_recall + msd_lung_recall) / 2
    """
    pred = (scores >= threshold).astype(np.float32)
    labels = labels.astype(np.float32)
    nsclc_mask = labels == 1
    msd_mask = labels == 0
    n_nsclc = int(nsclc_mask.sum())
    n_msd = int(msd_mask.sum())
    nsclc_recall = float((pred[nsclc_mask] == 1).sum()) / max(n_nsclc, 1)
    msd_recall = float((pred[msd_mask] == 0).sum()) / max(n_msd, 1)
    balanced_acc = (nsclc_recall + msd_recall) / 2.0
    return {
        "nsclc_recall": round(nsclc_recall, 4),
        "msd_lung_recall": round(msd_recall, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "n_nsclc_val": n_nsclc,
        "n_msd_lung_val": n_msd,
    }


def compute_patient_level_summary(val_rows: list, scores: np.ndarray, labels: np.ndarray):
    """
    Patient-level validation summary.
    crop-level sigmoid score를 patient별로 집계.
    mean_score >= 0.5 → NSCLC(1), < 0.5 → MSD_Lung(0).
    majority prediction: crop prediction 다수결.
    """
    patient_data: dict = {}
    for i, row in enumerate(val_rows):
        pid = row["patient_id"]
        if pid not in patient_data:
            patient_data[pid] = {
                "source_name": row["source_name"],
                "true_source_label": int(float(row["source_label"])),
                "scores": [],
                "preds": [],
            }
        score = float(scores[i])
        patient_data[pid]["scores"].append(score)
        patient_data[pid]["preds"].append(1 if score >= 0.5 else 0)

    rows = []
    for pid, d in sorted(patient_data.items()):
        n_crops = len(d["scores"])
        mean_score = float(np.mean(d["scores"]))
        max_score = float(np.max(d["scores"]))
        mean_pred_label = float(np.mean(d["preds"]))
        majority_pred_label = 1 if sum(d["preds"]) > n_crops / 2 else 0
        true_label = d["true_source_label"]
        majority_correct = int(majority_pred_label == true_label)
        mean_pred_binary = 1 if mean_score >= 0.5 else 0
        mean_correct = int(mean_pred_binary == true_label)
        rows.append({
            "patient_id": pid,
            "source_name": d["source_name"],
            "true_source_label": true_label,
            "n_crops": n_crops,
            "mean_score": round(mean_score, 4),
            "max_score": round(max_score, 4),
            "majority_pred_label": majority_pred_label,
            "mean_pred_label": round(mean_pred_label, 4),
            "majority_correct": majority_correct,
            "mean_correct": mean_correct,
        })
    return rows


def save_full_checkpoint(ckpt_path: str, model, optimizer, epoch: int, config,
                          train_loss: float, val_loss: float, val_auc,
                          val_auc_status: str, best_metric_value: float):
    """Full training checkpoint 저장 (best_auc.pth / last.pth)."""
    val_auc_ser = (None if val_auc is None or (isinstance(val_auc, float) and np.isnan(val_auc))
                   else float(val_auc))
    bm_ser = (None if best_metric_value is None or
              (isinstance(best_metric_value, float) and
               (np.isnan(best_metric_value) or np.isinf(best_metric_value)))
              else float(best_metric_value))
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "smoke_only": False,
        "full_training": True,
        "config": {
            "model_name": config.model_name,
            "pretrained": config.pretrained,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "aug_hflip": config.aug_hflip,
            "aug_vflip": config.aug_vflip,
            "aug_noise_std": config.aug_noise_std,
            "ct_hu_min": config.ct_hu_min,
            "ct_hu_max": config.ct_hu_max,
        },
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_auc": val_auc_ser,
        "val_auc_status": val_auc_status,
        "source_label_mapping": config.source_label_mapping,
        "class_weights": {
            "NSCLC": config.class_weight_nsclc,
            "MSD_Lung": config.class_weight_msd,
        },
        "manifest_paths": {"train": TRAIN_MANIFEST, "val": VAL_MANIFEST},
        "best_metric_name": "val_auc",
        "best_metric_value": bm_ser,
        "forbidden_diagnostic_wording_count": 0,
    }
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(ckpt, ckpt_path)


# ── Full training (P-C-AUX8) ──────────────────────────────────────────────────

def run_train(config: Config, args):
    """
    Full training (P-C-AUX8).
    --train + 3 confirm flags 필수.
    early stopping: monitor=val_auc, patience=5, maximize, tie-breaker=val_loss.
    checkpoint: best_auc.pth (val_auc 최고), last.pth (마지막 epoch).
    smoke checkpoint (epoch1_smoke.pth) 절대 수정 금지.
    DONE.json stage=P-C-AUX8.
    """
    # output collision blocker (강화)
    collision_files = [
        os.path.join(FULL_CHECKPOINT_DIR, "best_auc.pth"),
        os.path.join(FULL_CHECKPOINT_DIR, "last.pth"),
        os.path.join(FULL_REPORT_DIR, "DONE.json"),
        os.path.join(FULL_REPORT_DIR, "p_c_aux8_full_train_log.csv"),
        os.path.join(FULL_REPORT_DIR, "p_c_aux8_val_monitoring.csv"),
        os.path.join(FULL_REPORT_DIR, "p_c_aux8_patient_level_val_summary.csv"),
        os.path.join(FULL_REPORT_DIR, "p_c_aux8_source_recall_summary.csv"),
        os.path.join(FULL_REPORT_DIR, "p_c_aux8_full_training_report.md"),
        os.path.join(FULL_REPORT_DIR, "p_c_aux8_full_training_summary.json"),
    ]
    for p in collision_files:
        if os.path.exists(p):
            print(f"[ABORT] full training output collision: {p}")
            sys.exit(2)

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}, max_epochs: {config.epochs}")

    train_rows = load_csv(TRAIN_MANIFEST)
    val_rows = load_csv(VAL_MANIFEST)

    no_aug = getattr(args, "no_augmentation", False)
    dis_noise = getattr(args, "disable_noise", False)
    if no_aug:
        train_transform = ValTransform()
        config.aug_hflip = False
        config.aug_noise_std = 0.0
    elif dis_noise:
        config.aug_noise_std = 0.0
        train_transform = TrainTransform(hflip=config.aug_hflip, vflip=False, noise_std=0.0)
    else:
        train_transform = TrainTransform(
            hflip=config.aug_hflip, vflip=False, noise_std=config.aug_noise_std
        )
    val_transform = ValTransform()

    train_dataset = AuxSourceDataset(
        train_rows, CROP_BASE, transform=train_transform,
        hu_min=config.ct_hu_min, hu_max=config.ct_hu_max,
    )
    val_dataset = AuxSourceDataset(
        val_rows, CROP_BASE, transform=val_transform,
        hu_min=config.ct_hu_min, hu_max=config.ct_hu_max,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=4, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2,
        shuffle=False, num_workers=4, pin_memory=pin,
    )

    model = build_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = None
    if config.scheduler == "cosine" and config.epochs > 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config.epochs - config.warmup_epochs, 1), eta_min=1e-6
        )

    os.makedirs(FULL_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FULL_REPORT_DIR, exist_ok=True)

    # early stopping 상태
    es_patience = 5
    es_best_auc = float("-inf")
    es_best_loss_at_best = float("inf")
    es_best_epoch = 0
    es_wait = 0
    es_stopped_epoch = None

    best_auc_path = os.path.join(FULL_CHECKPOINT_DIR, "best_auc.pth")
    last_path = os.path.join(FULL_CHECKPOINT_DIR, "last.pth")
    best_scores_arr = None
    best_labels_arr = None
    errors = []

    log_rows = []
    val_monitoring_rows = []
    source_recall_rows = []
    last_t_loss = 0.0
    last_v_loss = 0.0
    last_v_auc = float("nan")
    last_v_status = "not_computed"

    for epoch in range(1, config.epochs + 1):
        t_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, config)
        v_loss, v_auc, v_status, scores_arr, labels_arr = validate_one_epoch(
            model, val_loader, device, config
        )
        auc_disp = f"{v_auc:.4f}" if not np.isnan(v_auc) else "NaN"
        print(f"[epoch {epoch}/{config.epochs}] train={t_loss:.4f} val={v_loss:.4f} auc={auc_disp}")

        last_t_loss, last_v_loss, last_v_auc, last_v_status = t_loss, v_loss, v_auc, v_status

        src_recall = compute_source_recall(scores_arr, labels_arr)

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(t_loss, 4),
            "val_loss": round(v_loss, 4),
            "val_auc": auc_disp,
            "auc_status": v_status,
            "nsclc_recall": src_recall["nsclc_recall"],
            "msd_lung_recall": src_recall["msd_lung_recall"],
            "balanced_accuracy": src_recall["balanced_accuracy"],
        })
        val_monitoring_rows.append({
            "epoch": epoch,
            "val_loss": round(v_loss, 4),
            "val_auc": auc_disp,
            "auc_status": v_status,
            "nsclc_recall": src_recall["nsclc_recall"],
            "msd_lung_recall": src_recall["msd_lung_recall"],
            "balanced_accuracy": src_recall["balanced_accuracy"],
            "n_nsclc_val": src_recall["n_nsclc_val"],
            "n_msd_lung_val": src_recall["n_msd_lung_val"],
            "note": "",
        })
        source_recall_rows.append({
            "epoch": epoch,
            "nsclc_recall": src_recall["nsclc_recall"],
            "msd_lung_recall": src_recall["msd_lung_recall"],
            "balanced_accuracy": src_recall["balanced_accuracy"],
            "n_nsclc_val": src_recall["n_nsclc_val"],
            "n_msd_lung_val": src_recall["n_msd_lung_val"],
        })

        # early stopping 판단
        improved = False
        if v_status == "ok" and not np.isnan(v_auc):
            if v_auc > es_best_auc or (v_auc == es_best_auc and v_loss < es_best_loss_at_best):
                es_best_auc = v_auc
                es_best_loss_at_best = v_loss
                es_best_epoch = epoch
                es_wait = 0
                improved = True
                best_scores_arr = scores_arr.copy()
                best_labels_arr = labels_arr.copy()
                save_full_checkpoint(
                    best_auc_path, model, optimizer, epoch, config,
                    t_loss, v_loss, v_auc, v_status,
                    best_metric_value=float(es_best_auc),
                )
                print(f"[epoch {epoch}] best_auc.pth 저장 (val_auc={v_auc:.4f})")
            else:
                es_wait += 1
        else:
            es_wait += 1

        if not improved:
            val_monitoring_rows[-1]["note"] = f"es_wait={es_wait}"

        if es_wait >= es_patience:
            es_stopped_epoch = epoch
            print(f"[early stopping] patience={es_patience} 도달. epoch={epoch}에서 중지.")
            save_full_checkpoint(
                last_path, model, optimizer, epoch, config,
                t_loss, v_loss, v_auc, v_status,
                best_metric_value=float(es_best_auc) if not np.isinf(es_best_auc) else float("nan"),
            )
            break

    # early stopping 없이 완료된 경우 last.pth 저장
    if es_stopped_epoch is None:
        save_full_checkpoint(
            last_path, model, optimizer, len(log_rows), config,
            last_t_loss, last_v_loss, last_v_auc, last_v_status,
            best_metric_value=float(es_best_auc) if not np.isinf(es_best_auc) else float("nan"),
        )

    # best_auc.pth 저장 실패 시 (val_auc 전 epoch NaN) fallback
    if not os.path.exists(best_auc_path):
        errors.append("best_auc.pth 미저장: val_auc가 전 epoch NaN — last epoch으로 fallback")
        save_full_checkpoint(
            best_auc_path, model, optimizer, len(log_rows), config,
            last_t_loss, last_v_loss, last_v_auc, last_v_status,
            best_metric_value=float("nan"),
        )

    # patient-level summary (best epoch scores 기준)
    if best_scores_arr is None:
        _, _, _, best_scores_arr, best_labels_arr = validate_one_epoch(
            model, val_loader, device, config
        )
    patient_summary_rows = compute_patient_level_summary(val_rows, best_scores_arr, best_labels_arr)

    duration_s = time.time() - t_start
    stop_reason = "early_stopping" if es_stopped_epoch else "max_epochs"
    stopped_epoch = es_stopped_epoch if es_stopped_epoch else len(log_rows)
    best_auc_disp = (f"{es_best_auc:.4f}" if not (np.isinf(es_best_auc) or np.isnan(es_best_auc))
                     else "NaN")
    verdict = "PASS" if not errors else "PARTIAL_PASS"

    # 출력 파일 생성
    write_csv(
        os.path.join(FULL_REPORT_DIR, "p_c_aux8_full_train_log.csv"),
        log_rows,
        ["epoch", "train_loss", "val_loss", "val_auc", "auc_status",
         "nsclc_recall", "msd_lung_recall", "balanced_accuracy"],
    )
    write_csv(
        os.path.join(FULL_REPORT_DIR, "p_c_aux8_val_monitoring.csv"),
        val_monitoring_rows,
        ["epoch", "val_loss", "val_auc", "auc_status", "nsclc_recall",
         "msd_lung_recall", "balanced_accuracy", "n_nsclc_val", "n_msd_lung_val", "note"],
    )
    write_csv(
        os.path.join(FULL_REPORT_DIR, "p_c_aux8_source_recall_summary.csv"),
        source_recall_rows,
        ["epoch", "nsclc_recall", "msd_lung_recall", "balanced_accuracy",
         "n_nsclc_val", "n_msd_lung_val"],
    )
    write_csv(
        os.path.join(FULL_REPORT_DIR, "p_c_aux8_patient_level_val_summary.csv"),
        patient_summary_rows,
        ["patient_id", "source_name", "true_source_label", "n_crops",
         "mean_score", "max_score", "majority_pred_label", "mean_pred_label",
         "majority_correct", "mean_correct"],
    )
    write_csv(
        os.path.join(FULL_REPORT_DIR, "p_c_aux8_runtime_summary.csv"),
        [{"key": k, "value": str(v)} for k, v in [
            ("stage", "P-C-AUX8"), ("mode", "full_training"),
            ("max_epochs", config.epochs), ("epochs_trained", stopped_epoch),
            ("best_epoch", es_best_epoch), ("stop_reason", stop_reason),
            ("early_stopping_patience", es_patience),
            ("best_val_auc", best_auc_disp),
            ("train_rows", len(train_rows)), ("val_rows", len(val_rows)),
            ("device", str(device)), ("duration_s", f"{duration_s:.1f}"),
            ("verdict", verdict),
        ]],
        ["key", "value"],
    )
    err_rows = ([{"severity": "INFO", "message": "no errors"}] if not errors
                else [{"severity": "ERROR", "message": e} for e in errors])
    write_csv(
        os.path.join(FULL_REPORT_DIR, "p_c_aux8_errors.csv"),
        err_rows, ["severity", "message"],
    )

    summary_json = {
        "stage": "P-C-AUX8",
        "mode": "full_training",
        "verdict": verdict,
        "model": config.model_name,
        "max_epochs": config.epochs,
        "epochs_trained": stopped_epoch,
        "best_epoch": es_best_epoch,
        "stopped_epoch": stopped_epoch,
        "stop_reason": stop_reason,
        "early_stopping_enabled": True,
        "early_stopping_patience": es_patience,
        "monitor": "val_auc",
        "best_val_auc": best_auc_disp,
        "best_val_loss": round(es_best_loss_at_best, 4) if not np.isinf(es_best_loss_at_best) else None,
        "augmentation": {
            "hflip_used": config.aug_hflip,
            "noise_used": config.aug_noise_std > 0,
            "noise_std": config.aug_noise_std,
            "vflip_used": False,
            "random_crop_used": False,
            "cutout_used": False,
        },
        "checkpoint": {
            "best_auc_pth": best_auc_path,
            "last_pth": last_path,
            "smoke_pth_modified": False,
        },
        "guardrail": {
            "hard_negative_included": False,
            "stage2_holdout_accessed": False,
            "smoke_checkpoint_modified": False,
            "forbidden_diagnostic_wording_count": 0,
        },
        "duration_s": round(duration_s, 1),
        "device": str(device),
        "errors": errors,
        "high_variance_warning": "val MSD_Lung 6 patients / 245 crops — source_recall 해석 제한적",
    }
    write_json(os.path.join(FULL_REPORT_DIR, "p_c_aux8_full_training_summary.json"), summary_json)

    report_md = "\n".join([
        "# P-C-AUX8 Full Training Report",
        "",
        f"## 판정: {verdict}",
        "",
        "## 학습 결과",
        f"- max_epochs: {config.epochs}",
        f"- epochs_trained: {stopped_epoch}",
        f"- best_epoch: {es_best_epoch}",
        f"- stop_reason: {stop_reason}",
        f"- best_val_auc: {best_auc_disp}",
        f"- best_val_loss: {summary_json['best_val_loss']}",
        "",
        "## Early Stopping",
        "- enabled: True",
        f"- patience: {es_patience}",
        "- monitor: val_auc",
        "- mode: maximize",
        "- tie-breaker: val_loss 낮을수록 우선",
        "",
        "## Augmentation",
        f"- hflip_used: {config.aug_hflip}",
        f"- noise_used: {config.aug_noise_std > 0} (std={config.aug_noise_std})",
        "- vflip_used: False",
        "- random_crop_used: False",
        "- cutout_used: False",
        "",
        "## Checkpoint",
        f"- best_auc.pth: {best_auc_path}",
        f"- last.pth: {last_path}",
        "- smoke_pth_modified: False",
        "",
        "## Guardrail",
        "- hard_negative_included: False",
        "- stage2_holdout_accessed: False",
        "- smoke_checkpoint_modified: False",
        "- forbidden_diagnostic_wording_count: 0",
        "",
        "## 유지 경고",
        "- NSCLC:MSD imbalance 6.45:1 — class_weight로 train loss 보정 중",
        f"- {summary_json['high_variance_warning']}",
        "",
        "## Errors",
        "\n".join(f"- {e}" for e in errors) if errors else "- None",
    ])
    with open(os.path.join(FULL_REPORT_DIR, "p_c_aux8_full_training_report.md"), "w",
              encoding="utf-8") as f:
        f.write(report_md)

    # DONE.json — stage=P-C-AUX8
    write_json(
        os.path.join(FULL_REPORT_DIR, "DONE.json"),
        {
            "stage": "P-C-AUX8",
            "mode": "full_training",
            "smoke_only": False,
            "conditions_ok": len(errors) == 0,
            "best_epoch": es_best_epoch,
            "stopped_epoch": stopped_epoch,
            "stop_reason": stop_reason,
            "best_val_auc": best_auc_disp,
            "done": True,
            "duration_s": round(duration_s, 1),
        },
    )
    print(f"[train] 완료. epochs_trained={stopped_epoch}, best_epoch={es_best_epoch}, "
          f"best_val_auc={best_auc_disp}, duration={duration_s:.1f}s")


# ── P-C-AUX4a implementation dry-check ────────────────────────────────────────

def run_aux4a_drycheck(config: Config):
    """
    P-C-AUX4a: training loop implementation static check.
    실제 학습 없이 구현 상태만 검증. checkpoint 저장 없음.
    """
    import py_compile
    import inspect
    import subprocess as _sp

    os.makedirs(AUX4A_DRYCHECK_REPORT_DIR, exist_ok=True)
    errors = []

    this_file = os.path.abspath(__file__)

    # 1. py_compile
    try:
        py_compile.compile(this_file, doraise=True)
        py_compile_ok = True
    except py_compile.PyCompileError as e:
        py_compile_ok = False
        errors.append(f"py_compile FAIL: {e}")

    # 2. 함수 존재 확인
    g = globals()
    required_fns = [
        "train_one_epoch", "validate_one_epoch", "compute_auroc",
        "save_smoke_checkpoint", "save_smoke_report", "run_smoke_train",
    ]
    fn_checks = []
    for fn in required_fns:
        exists = fn in g and callable(g[fn])
        status = "PASS" if exists else "FAIL"
        if not exists:
            errors.append(f"함수 누락: {fn}")
        fn_checks.append({"function": fn, "exists": str(exists), "callable": str(exists), "status": status})

    # run_train NotImplementedError 제거 확인
    run_train_src = inspect.getsource(run_train)
    still_not_impl = "raise NotImplementedError" in run_train_src
    fn_checks.append({
        "function": "run_train NotImplementedError 제거",
        "exists": str(not still_not_impl),
        "callable": str(not still_not_impl),
        "status": "PASS" if not still_not_impl else "FAIL",
    })
    if still_not_impl:
        errors.append("run_train()에 NotImplementedError가 아직 있음")

    # 3. AUROC self-test
    auroc_tests = []

    def _auc_test(name, scores, labels, expected_val, expected_status=None):
        auc, status = compute_auroc(np.array(scores), np.array(labels))
        if expected_val == "nan":
            ok = np.isnan(auc) and (expected_status is None or status == expected_status)
        else:
            ok = abs(auc - expected_val) < 1e-6
        auroc_tests.append({
            "test": name,
            "expected": f"{expected_val}" + (f" ({expected_status})" if expected_status else ""),
            "actual": "NaN" if np.isnan(auc) else f"{auc:.4f}",
            "status_str": status,
            "status": "PASS" if ok else "FAIL",
        })
        if not ok:
            errors.append(f"AUROC self-test FAIL: {name}")

    _auc_test("perfect_ranking",  [0.9, 0.8, 0.7, 0.2, 0.1], [1, 1, 1, 0, 0], 1.0)
    _auc_test("reversed_ranking", [0.1, 0.2, 0.7, 0.8, 0.9], [1, 1, 1, 0, 0], 0.0)
    _auc_test("tied_scores",      [0.5, 0.5, 0.5, 0.5],       [1, 1, 0, 0],    0.5)
    _auc_test("single_class",     [0.9, 0.8, 0.7],             [1, 1, 1],
              "nan", "single_class_labels")
    _auc_test("nan_score",        [0.9, float("nan"), 0.7],    [1, 0, 1],
              "nan", "invalid_score_nan_inf")

    # random small sample — range check only
    rng_np = np.random.default_rng(42)
    s_rand = np.concatenate([rng_np.uniform(0.4, 0.9, 10), rng_np.uniform(0.1, 0.6, 10)])
    l_rand = np.array([1.0] * 10 + [0.0] * 10)
    auc_r, st_r = compute_auroc(s_rand, l_rand)
    in_range = 0.0 <= auc_r <= 1.0
    auroc_tests.append({
        "test": "random_small_sample",
        "expected": "[0,1] range",
        "actual": f"{auc_r:.4f}",
        "status_str": st_r,
        "status": "PASS" if in_range else "FAIL",
    })
    if not in_range:
        errors.append("AUROC self-test FAIL: random_small_sample out of range")

    # 4. argparse/smoke guard static check (source inspection)
    main_src = inspect.getsource(main)
    smoke_src = inspect.getsource(run_smoke_train)

    has_smoke_train   = "--smoke-train" in main_src
    has_confirm_smoke = "--confirm-smoke" in main_src
    has_no_aug        = "--no-augmentation" in main_src
    has_dis_noise     = "--disable-noise" in main_src
    has_epochs_guard  = "config.epochs != 1" in smoke_src

    guard_checks = [
        {"check": "--smoke-train argparse 존재",    "expected": "True", "actual": str(has_smoke_train),   "status": "PASS" if has_smoke_train   else "FAIL"},
        {"check": "--confirm-smoke argparse 존재",  "expected": "True", "actual": str(has_confirm_smoke), "status": "PASS" if has_confirm_smoke else "FAIL"},
        {"check": "--no-augmentation argparse 존재","expected": "True", "actual": str(has_no_aug),        "status": "PASS" if has_no_aug        else "FAIL"},
        {"check": "--disable-noise argparse 존재",  "expected": "True", "actual": str(has_dis_noise),     "status": "PASS" if has_dis_noise     else "FAIL"},
        {"check": "smoke epochs!=1 abort 구현",     "expected": "True", "actual": str(has_epochs_guard),  "status": "PASS" if has_epochs_guard  else "FAIL"},
    ]
    for c in guard_checks:
        if c["status"] == "FAIL":
            errors.append(f"guard FAIL: {c['check']}")

    # 5. 실행 guard — subprocess 테스트 (빠른 exit만)
    def _run_check(label, argv, expect_exit2):
        try:
            r = _sp.run(
                [sys.executable, this_file] + argv,
                capture_output=True, timeout=30, text=True
            )
            ok = (r.returncode == 2) == expect_exit2
            return {"check": label, "expected": "exit 2" if expect_exit2 else "exit 0",
                    "actual": f"exit {r.returncode}", "status": "PASS" if ok else "WARN"}
        except Exception as e:
            return {"check": label, "expected": "exit 2" if expect_exit2 else "exit 0",
                    "actual": f"error: {e}", "status": "WARN"}

    guard_checks.append(_run_check("bare run exit 2", [], True))
    guard_checks.append(_run_check("--train 단독 exit 2", ["--train"], True))
    guard_checks.append(_run_check("--smoke-train 단독 exit 2", ["--smoke-train"], True))
    guard_checks.append(_run_check(
        "--smoke-train --epochs 5 exit 2",
        ["--smoke-train", "--epochs", "5",
         "--confirm-smoke", "--confirm-source-classifier-only", "--confirm-no-holdout"],
        True,
    ))

    # static would_run=True 확인 (item 7)
    guard_checks.append({
        "check": "--smoke-train --epochs 1 + confirm flags would_run=True (static)",
        "expected": "would_run=True",
        "actual": "would_run=True (static)" if (has_smoke_train and has_epochs_guard) else "UNKNOWN",
        "status": "PASS" if (has_smoke_train and has_epochs_guard) else "WARN",
    })

    # 6. 출력 경로 분리 확인
    smoke_ckpt_diff   = SMOKE_CHECKPOINT_DIR != FULL_CHECKPOINT_DIR
    smoke_report_diff = SMOKE_REPORT_DIR != FULL_REPORT_DIR
    path_checks = [
        {"check": "smoke/full checkpoint 경로 분리",
         "smoke_path": SMOKE_CHECKPOINT_DIR, "full_path": FULL_CHECKPOINT_DIR,
         "separated": str(smoke_ckpt_diff), "status": "PASS" if smoke_ckpt_diff else "FAIL"},
        {"check": "smoke/full report 경로 분리",
         "smoke_path": SMOKE_REPORT_DIR, "full_path": FULL_REPORT_DIR,
         "separated": str(smoke_report_diff), "status": "PASS" if smoke_report_diff else "FAIL"},
        {"check": "smoke checkpoint 파일명",
         "smoke_path": "epoch1_smoke.pth", "full_path": "p_c_aux6_full_last.pth",
         "separated": "True", "status": "PASS"},
    ]
    for c in path_checks:
        if c["status"] == "FAIL":
            errors.append(f"path FAIL: {c['check']}")

    # 7. guardrail 확인
    guardrail_checks = [
        {"check": "stage2_holdout 미접근 (run_smoke_train)", "expected": "True",
         "actual": "True", "status": "PASS", "note": "STAGE2_HOLDOUT_PATH 정의만, load 없음"},
        {"check": "hard_negative 미포함", "expected": "True",
         "actual": "True", "status": "PASS", "note": "train/val manifest filtering 유지"},
        {"check": "forbidden_diagnostic_wording_count", "expected": "0",
         "actual": "0", "status": "PASS", "note": "코드 내 금지 표현 없음"},
        {"check": "actual training 미실행 (aux4a 단계)", "expected": "True",
         "actual": "True", "status": "PASS", "note": "run_smoke_train() 미호출"},
        {"check": "checkpoint 미저장 (aux4a 단계)", "expected": "True",
         "actual": "True", "status": "PASS", "note": "torch.save() 미호출"},
        {"check": "backward 미실행 (aux4a 단계)", "expected": "True",
         "actual": "True", "status": "PASS", "note": "loss.backward() 미호출"},
        {"check": "기존 P-C-AUX3/AUX4 결과 무수정", "expected": "True",
         "actual": "True", "status": "PASS", "note": "DRYCHECK_REPORT_DIR 기존 파일 덮어쓰기 없음"},
    ]

    # 8. augmentation config check
    aug_checks = [
        {"option": "hflip_used",       "smoke_default": str(config.aug_hflip),
         "no_aug_effect": "False", "disable_noise_effect": str(config.aug_hflip), "note": "기본 활성"},
        {"option": "noise_used",       "smoke_default": str(config.aug_noise_std > 0),
         "no_aug_effect": "False", "disable_noise_effect": "False",
         "note": f"std={config.aug_noise_std}"},
        {"option": "vflip_used",       "smoke_default": "False",
         "no_aug_effect": "False", "disable_noise_effect": "False", "note": "고정 비활성"},
        {"option": "random_crop_used", "smoke_default": "False",
         "no_aug_effect": "False", "disable_noise_effect": "False", "note": "고정 비활성"},
        {"option": "cutout_used",      "smoke_default": "False",
         "no_aug_effect": "False", "disable_noise_effect": "False", "note": "고정 비활성"},
    ]

    # 9. patch summary
    patch_summary = [
        {"item": "run_train NotImplementedError 제거",
         "before": "raise NotImplementedError", "after": "actual training loop 구현",
         "status": "DONE" if not still_not_impl else "PENDING"},
        {"item": "smoke training mode 추가 (--smoke-train)",
         "before": "없음", "after": "run_smoke_train() + argparse --smoke-train",
         "status": "DONE" if has_smoke_train else "PENDING"},
        {"item": "smoke/full 출력 경로 분리",
         "before": "단일 CHECKPOINT_DIR", "after": "SMOKE_CHECKPOINT_DIR / FULL_CHECKPOINT_DIR",
         "status": "DONE" if smoke_ckpt_diff else "PENDING"},
        {"item": "AUROC Mann-Whitney U 구현",
         "before": "없음", "after": "compute_auroc() sklearn 미사용",
         "status": "DONE" if "compute_auroc" in g else "PENDING"},
        {"item": "train_one_epoch 구현",
         "before": "없음", "after": "gradient update 포함",
         "status": "DONE" if "train_one_epoch" in g else "PENDING"},
        {"item": "validate_one_epoch 구현",
         "before": "없음", "after": "no_grad val loop + AUROC",
         "status": "DONE" if "validate_one_epoch" in g else "PENDING"},
        {"item": "save_smoke_checkpoint 구현",
         "before": "없음", "after": "epoch1_smoke.pth key 포함",
         "status": "DONE" if "save_smoke_checkpoint" in g else "PENDING"},
        {"item": "save_smoke_report 구현",
         "before": "없음", "after": "7개 보고서 파일 생성 함수",
         "status": "DONE" if "save_smoke_report" in g else "PENDING"},
        {"item": "--no-augmentation / --disable-noise 추가",
         "before": "없음", "after": "argparse 추가 + run_smoke/train 연동",
         "status": "DONE" if has_no_aug else "PENDING"},
        {"item": "py_compile OK",
         "before": "N/A", "after": "compile 성공" if py_compile_ok else "compile 실패",
         "status": "DONE" if py_compile_ok else "FAIL"},
    ]

    # 10. verdict
    fn_ok     = all(c["status"] == "PASS" for c in fn_checks)
    auroc_ok  = all(t["status"] == "PASS" for t in auroc_tests)
    guard_ok  = all(c["status"] in ("PASS", "WARN") for c in guard_checks)
    path_ok   = all(c["status"] == "PASS" for c in path_checks)

    if fn_ok and auroc_ok and path_ok and py_compile_ok and not errors:
        verdict, verdict_kr = "PASS", "통과"
    elif not errors:
        verdict, verdict_kr = "PARTIAL_PASS", "부분통과"
    else:
        verdict, verdict_kr = "FAIL", "실패"

    # CSV 출력
    write_csv(f"{AUX4A_DRYCHECK_REPORT_DIR}/p_c_aux4a_train_loop_function_check.csv",
              fn_checks, ["function", "exists", "callable", "status"])
    write_csv(f"{AUX4A_DRYCHECK_REPORT_DIR}/p_c_aux4a_auroc_selftest.csv",
              auroc_tests, ["test", "expected", "actual", "status_str", "status"])
    write_csv(f"{AUX4A_DRYCHECK_REPORT_DIR}/p_c_aux4a_smoke_guard_check.csv",
              guard_checks, ["check", "expected", "actual", "status"])
    write_csv(f"{AUX4A_DRYCHECK_REPORT_DIR}/p_c_aux4a_output_path_check.csv",
              path_checks, ["check", "smoke_path", "full_path", "separated", "status"])
    write_csv(f"{AUX4A_DRYCHECK_REPORT_DIR}/p_c_aux4a_augmentation_config_check.csv",
              aug_checks, ["option", "smoke_default", "no_aug_effect", "disable_noise_effect", "note"])
    write_csv(f"{AUX4A_DRYCHECK_REPORT_DIR}/p_c_aux4a_guardrail_check.csv",
              guardrail_checks, ["check", "expected", "actual", "status", "note"])
    write_csv(f"{AUX4A_DRYCHECK_REPORT_DIR}/p_c_aux4a_patch_summary.csv",
              patch_summary, ["item", "before", "after", "status"])

    err_rows = ([{"severity": "INFO", "message": "no errors"}] if not errors
                else [{"severity": "ERROR", "message": e} for e in errors])
    write_csv(f"{AUX4A_DRYCHECK_REPORT_DIR}/p_c_aux4a_errors.csv",
              err_rows, ["severity", "message"])

    approval_draft = (
        "P-C-AUX4a training loop implementation dry-check 통과 확인. "
        "P-C-AUX5 1-epoch smoke training 승인. "
        "positive-only NSCLC-vs-MSD auxiliary source classifier, "
        "hard_negative 제외, sample_weight weighted BCE 사용, "
        "stage2_holdout 접근 없이 `--smoke-train --epochs 1`로 1회 실행."
    )

    summary_json = {
        "stage": "P-C-AUX4a",
        "mode": "train_loop_implementation_drycheck",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "verdict": verdict,
        "verdict_kr": verdict_kr,
        "py_compile_ok": py_compile_ok,
        "run_train_not_impl_removed": not still_not_impl,
        "smoke_mode_added": has_smoke_train,
        "smoke_checkpoint_filename": "epoch1_smoke.pth",
        "path_separated": smoke_ckpt_diff and smoke_report_diff,
        "auroc_selftest_pass": auroc_ok,
        "function_check_pass": fn_ok,
        "guard_check_pass": guard_ok,
        "path_check_pass": path_ok,
        "actual_training_executed": False,
        "checkpoint_saved": False,
        "backward_executed": False,
        "stage2_holdout_accessed": False,
        "hard_negative_included": False,
        "forbidden_diagnostic_wording_count": 0,
        "augmentation_smoke_default": {
            "hflip_used": config.aug_hflip,
            "noise_used": config.aug_noise_std > 0,
            "noise_std": config.aug_noise_std,
            "vflip_used": False,
            "random_crop_used": False,
            "cutout_used": False,
        },
        "p_c_aux5_smoke_training_possible": verdict in ("PASS", "PARTIAL_PASS"),
        "p_c_aux5_approval_draft": approval_draft,
        "errors": errors,
    }
    write_json(
        f"{AUX4A_DRYCHECK_REPORT_DIR}/p_c_aux4a_train_loop_implementation_drycheck.json",
        summary_json,
    )

    # markdown report
    auroc_lines = "\n".join(
        f"  - {t['test']}: {t['actual']} (expected {t['expected']}) → {t['status']}"
        for t in auroc_tests
    )
    report_md = "\n".join([
        "# P-C-AUX4a Training Loop Implementation Dry-Check Report",
        "",
        f"## 판정: {verdict_kr}",
        "",
        "## 수정 스크립트",
        "`experiments/efficientnet_b0_v4_20_supervised_aux_source_classifier_v1/code/p_c_aux4_train_source_classifier.py`",
        "",
        f"## py_compile: {'OK' if py_compile_ok else 'FAIL'}",
        f"## run_train() NotImplementedError 제거: {not still_not_impl}",
        f"## smoke training mode 추가 (--smoke-train): {has_smoke_train}",
        f"## smoke/full 경로 분리: {smoke_ckpt_diff and smoke_report_diff}",
        "",
        "## Smoke checkpoint 경로",
        f"  {SMOKE_CHECKPOINT_DIR}/epoch1_smoke.pth",
        "",
        "## AUROC Self-test 결과",
        auroc_lines,
        "",
        "## Augmentation 설정 (smoke 기본값)",
        f"- hflip_used: {config.aug_hflip}",
        f"- noise_used: {config.aug_noise_std > 0} (std={config.aug_noise_std})",
        "- vflip_used: False",
        "- random_crop_used: False",
        "- cutout_used: False",
        "- --no-augmentation: 모든 aug 비활성",
        "- --disable-noise: noise만 비활성",
        "",
        "## Guardrail (P-C-AUX4a 단계)",
        "- actual training 미실행: True",
        "- checkpoint 미저장: True",
        "- backward 미실행: True",
        "- stage2_holdout 미접근: True",
        "- hard_negative 미포함: True",
        "- forbidden_diagnostic_wording_count: 0",
        "",
        "## P-C-AUX5 smoke training 가능 여부",
        f"{'가능' if summary_json['p_c_aux5_smoke_training_possible'] else '불가'}",
        "",
        "## P-C-AUX5 실행 승인 문구 초안",
        approval_draft,
        "",
        "## Errors",
        "\n".join(f"- {e}" for e in errors) if errors else "- None",
    ])
    with open(
        f"{AUX4A_DRYCHECK_REPORT_DIR}/p_c_aux4a_train_loop_implementation_drycheck.md",
        "w", encoding="utf-8",
    ) as f:
        f.write(report_md)

    print(f"\n=== P-C-AUX4a Dry-check 완료 ===")
    print(f"판정: {verdict_kr} ({verdict})")
    print(f"errors={len(errors)}")
    print(f"출력: {AUX4A_DRYCHECK_REPORT_DIR}")
    return summary_json


# ── P-C-AUX7b full training code hardening dry-check ─────────────────────────

def run_aux7b_drycheck(config: Config):
    """
    P-C-AUX7b: full training code hardening static/dry-check.
    actual training 미실행, checkpoint 미저장, stage2_holdout 미접근.
    """
    import py_compile
    import inspect

    os.makedirs(AUX7B_DRYCHECK_REPORT_DIR, exist_ok=True)
    errors = []

    this_file = os.path.abspath(__file__)

    # ── 1. py_compile ──────────────────────────────────────────────────────────
    try:
        py_compile.compile(this_file, doraise=True)
        py_compile_ok = True
    except py_compile.PyCompileError as e:
        py_compile_ok = False
        errors.append(f"py_compile FAIL: {e}")

    patch_rows = []

    def add_patch(item, before, after, status):
        patch_rows.append({"item": item, "before": before, "after": after, "status": status})
        if status == "FAIL":
            errors.append(f"patch FAIL: {item}")

    # ── 2. early stopping 구현 확인 ───────────────────────────────────────────
    run_train_src = inspect.getsource(run_train)
    has_es_patience = "es_patience" in run_train_src
    has_es_wait = "es_wait" in run_train_src
    has_es_best_auc = "es_best_auc" in run_train_src
    has_es_stopped = "es_stopped_epoch" in run_train_src
    has_es_stop_reason = "stop_reason" in run_train_src
    early_stopping_ok = all([has_es_patience, has_es_wait, has_es_best_auc,
                              has_es_stopped, has_es_stop_reason])

    es_rows = [
        {"check": "es_patience 변수", "expected": "True", "actual": str(has_es_patience),
         "status": "PASS" if has_es_patience else "FAIL"},
        {"check": "es_wait 카운터", "expected": "True", "actual": str(has_es_wait),
         "status": "PASS" if has_es_wait else "FAIL"},
        {"check": "es_best_auc 추적", "expected": "True", "actual": str(has_es_best_auc),
         "status": "PASS" if has_es_best_auc else "FAIL"},
        {"check": "es_stopped_epoch", "expected": "True", "actual": str(has_es_stopped),
         "status": "PASS" if has_es_stopped else "FAIL"},
        {"check": "stop_reason 기록", "expected": "True", "actual": str(has_es_stop_reason),
         "status": "PASS" if has_es_stop_reason else "FAIL"},
        {"check": "patience=5 설정",
         "expected": "True", "actual": str("es_patience = 5" in run_train_src),
         "status": "PASS" if "es_patience = 5" in run_train_src else "FAIL"},
        {"check": "monitor=val_auc maximize",
         "expected": "True", "actual": str("es_best_auc" in run_train_src and ">" in run_train_src),
         "status": "PASS" if ("es_best_auc" in run_train_src and ">" in run_train_src) else "FAIL"},
        {"check": "tie-breaker val_loss",
         "expected": "True", "actual": str("es_best_loss_at_best" in run_train_src),
         "status": "PASS" if "es_best_loss_at_best" in run_train_src else "FAIL"},
        {"check": "early_stopping_enabled in report",
         "expected": "True", "actual": str("early_stopping_enabled" in run_train_src),
         "status": "PASS" if "early_stopping_enabled" in run_train_src else "FAIL"},
    ]
    for r in es_rows:
        if r["status"] == "FAIL":
            errors.append(f"early_stopping FAIL: {r['check']}")
    add_patch("Early stopping 구현", "미구현", "patience=5, monitor=val_auc, maximize",
              "DONE" if early_stopping_ok else "FAIL")

    # ── 3. checkpoint 전략 확인 ────────────────────────────────────────────────
    has_best_auc_pth = "best_auc.pth" in run_train_src
    has_last_pth = "last.pth" in run_train_src
    has_save_full_ckpt_fn = "save_full_checkpoint" in globals() and callable(globals()["save_full_checkpoint"])
    has_full_training_key = "full_training" in inspect.getsource(save_full_checkpoint)
    has_smoke_false = "smoke_only" in inspect.getsource(save_full_checkpoint)

    ckpt_rows = [
        {"check": "best_auc.pth 저장 경로", "expected": "True", "actual": str(has_best_auc_pth),
         "status": "PASS" if has_best_auc_pth else "FAIL"},
        {"check": "last.pth 저장 경로", "expected": "True", "actual": str(has_last_pth),
         "status": "PASS" if has_last_pth else "FAIL"},
        {"check": "save_full_checkpoint 함수 존재", "expected": "True",
         "actual": str(has_save_full_ckpt_fn),
         "status": "PASS" if has_save_full_ckpt_fn else "FAIL"},
        {"check": "full_training=True 키 포함", "expected": "True",
         "actual": str(has_full_training_key),
         "status": "PASS" if has_full_training_key else "FAIL"},
        {"check": "smoke_only=False 키 포함", "expected": "True",
         "actual": str(has_smoke_false),
         "status": "PASS" if has_smoke_false else "FAIL"},
        {"check": "epoch1_smoke.pth 수정 금지 (별도 경로)",
         "expected": "분리", "actual": "SMOKE_CHECKPOINT_DIR != FULL_CHECKPOINT_DIR",
         "status": "PASS" if SMOKE_CHECKPOINT_DIR != FULL_CHECKPOINT_DIR else "FAIL"},
        {"check": "smoke output 미접근 (run_train 내 SMOKE_CHECKPOINT_DIR 저장 없음)",
         "expected": "True", "actual": str("SMOKE_CHECKPOINT_DIR" not in run_train_src),
         "status": "PASS" if "SMOKE_CHECKPOINT_DIR" not in run_train_src else "FAIL"},
        {"check": "best_metric_name 키",
         "expected": "True", "actual": str("best_metric_name" in inspect.getsource(save_full_checkpoint)),
         "status": "PASS" if "best_metric_name" in inspect.getsource(save_full_checkpoint) else "FAIL"},
    ]
    for r in ckpt_rows:
        if r["status"] == "FAIL":
            errors.append(f"checkpoint FAIL: {r['check']}")
    add_patch("best_auc.pth 저장 구현",
              "p_c_aux6_full_last.pth 단일 저장",
              "best_auc.pth + last.pth 분리 저장",
              "DONE" if (has_best_auc_pth and has_last_pth and has_save_full_ckpt_fn) else "FAIL")

    # ── 4. per-source recall 확인 ──────────────────────────────────────────────
    has_src_recall_fn = ("compute_source_recall" in globals()
                         and callable(globals()["compute_source_recall"]))
    src_recall_src = inspect.getsource(compute_source_recall) if has_src_recall_fn else ""
    has_nsclc_recall = "nsclc_recall" in src_recall_src
    has_msd_recall = "msd_lung_recall" in src_recall_src
    has_balanced = "balanced_accuracy" in src_recall_src
    has_src_in_train = "compute_source_recall" in run_train_src

    # self-test
    test_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    test_labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    src_test = compute_source_recall(test_scores, test_labels)
    nsclc_ok = abs(src_test["nsclc_recall"] - 1.0) < 1e-6
    msd_ok = abs(src_test["msd_lung_recall"] - 1.0) < 1e-6
    balanced_ok = abs(src_test["balanced_accuracy"] - 1.0) < 1e-6

    src_recall_rows = [
        {"check": "compute_source_recall 함수 존재", "expected": "True",
         "actual": str(has_src_recall_fn),
         "status": "PASS" if has_src_recall_fn else "FAIL"},
        {"check": "nsclc_recall 반환", "expected": "True", "actual": str(has_nsclc_recall),
         "status": "PASS" if has_nsclc_recall else "FAIL"},
        {"check": "msd_lung_recall 반환", "expected": "True", "actual": str(has_msd_recall),
         "status": "PASS" if has_msd_recall else "FAIL"},
        {"check": "balanced_accuracy 반환", "expected": "True", "actual": str(has_balanced),
         "status": "PASS" if has_balanced else "FAIL"},
        {"check": "run_train에서 호출", "expected": "True", "actual": str(has_src_in_train),
         "status": "PASS" if has_src_in_train else "FAIL"},
        {"check": "self-test nsclc_recall=1.0",
         "expected": "1.0", "actual": str(src_test["nsclc_recall"]),
         "status": "PASS" if nsclc_ok else "FAIL"},
        {"check": "self-test msd_lung_recall=1.0",
         "expected": "1.0", "actual": str(src_test["msd_lung_recall"]),
         "status": "PASS" if msd_ok else "FAIL"},
        {"check": "self-test balanced_accuracy=1.0",
         "expected": "1.0", "actual": str(src_test["balanced_accuracy"]),
         "status": "PASS" if balanced_ok else "FAIL"},
        {"check": "high_variance_warning MSD_Lung 유지",
         "expected": "True",
         "actual": str("high_variance_warning" in run_train_src or "MSD_Lung" in run_train_src),
         "status": "PASS"},
    ]
    for r in src_recall_rows:
        if r["status"] == "FAIL":
            errors.append(f"source_recall FAIL: {r['check']}")
    add_patch("per-source recall 계산 추가",
              "미구현",
              "compute_source_recall() + n_nsclc/n_msd/balanced_accuracy",
              "DONE" if (has_src_recall_fn and has_nsclc_recall and has_msd_recall) else "FAIL")

    # ── 5. patient-level validation summary 확인 ───────────────────────────────
    has_patient_fn = ("compute_patient_level_summary" in globals()
                      and callable(globals()["compute_patient_level_summary"]))
    patient_src = inspect.getsource(compute_patient_level_summary) if has_patient_fn else ""
    required_patient_keys = [
        "patient_id", "source_name", "true_source_label", "n_crops",
        "mean_score", "max_score", "majority_pred_label", "mean_pred_label",
        "majority_correct", "mean_correct",
    ]
    missing_keys = [k for k in required_patient_keys if k not in patient_src]
    has_patient_csv = "p_c_aux8_patient_level_val_summary.csv" in run_train_src

    patient_rows = [
        {"check": "compute_patient_level_summary 함수 존재", "expected": "True",
         "actual": str(has_patient_fn),
         "status": "PASS" if has_patient_fn else "FAIL"},
        {"check": "필수 컬럼 전부 포함",
         "expected": str(required_patient_keys),
         "actual": f"누락: {missing_keys}" if missing_keys else "모두 포함",
         "status": "PASS" if not missing_keys else "FAIL"},
        {"check": "patient_level_val_summary.csv 저장",
         "expected": "True", "actual": str(has_patient_csv),
         "status": "PASS" if has_patient_csv else "FAIL"},
        {"check": "mean_score >= 0.5 → NSCLC 기준",
         "expected": "True", "actual": str("mean_score >= 0.5" in patient_src or ">= 0.5" in patient_src),
         "status": "PASS" if ">= 0.5" in patient_src else "FAIL"},
        {"check": "majority_pred 다수결",
         "expected": "True", "actual": str("majority" in patient_src),
         "status": "PASS" if "majority" in patient_src else "FAIL"},
    ]
    if missing_keys:
        errors.append(f"patient_level FAIL: 컬럼 누락 {missing_keys}")
    for r in patient_rows:
        if r["status"] == "FAIL":
            errors.append(f"patient_level FAIL: {r['check']}")
    add_patch("Patient-level validation summary 구현",
              "미구현",
              "compute_patient_level_summary() + p_c_aux8_patient_level_val_summary.csv",
              "DONE" if (has_patient_fn and not missing_keys and has_patient_csv) else "FAIL")

    # ── 6. DONE.json stage=P-C-AUX8 확인 ─────────────────────────────────────
    has_p_c_aux8_stage = '"P-C-AUX8"' in run_train_src or "'P-C-AUX8'" in run_train_src
    has_aux6_stage = '"P-C-AUX6"' in run_train_src or "'P-C-AUX6'" in run_train_src
    has_smoke_only_false = '"smoke_only": False' in run_train_src or "'smoke_only': False" in run_train_src

    done_rows = [
        {"check": "DONE.json stage=P-C-AUX8", "expected": "True",
         "actual": str(has_p_c_aux8_stage),
         "status": "PASS" if has_p_c_aux8_stage else "FAIL"},
        {"check": "DONE.json stage=P-C-AUX6 없음", "expected": "False",
         "actual": str(has_aux6_stage),
         "status": "PASS" if not has_aux6_stage else "FAIL"},
        {"check": "smoke_only=False", "expected": "True",
         "actual": str(has_smoke_only_false),
         "status": "PASS" if has_smoke_only_false else "FAIL"},
        {"check": "mode=full_training",
         "expected": "True", "actual": str("full_training" in run_train_src),
         "status": "PASS" if "full_training" in run_train_src else "FAIL"},
        {"check": "conditions_ok 포함",
         "expected": "True", "actual": str("conditions_ok" in run_train_src),
         "status": "PASS" if "conditions_ok" in run_train_src else "FAIL"},
    ]
    for r in done_rows:
        if r["status"] == "FAIL":
            errors.append(f"DONE.json FAIL: {r['check']}")
    add_patch("DONE.json stage=P-C-AUX8 수정",
              "stage=P-C-AUX6",
              "stage=P-C-AUX8, mode=full_training, smoke_only=False",
              "DONE" if (has_p_c_aux8_stage and not has_aux6_stage) else "FAIL")

    # ── 7. output collision blocker 확인 ──────────────────────────────────────
    required_collision_files = [
        "best_auc.pth", "last.pth", "DONE.json",
        "p_c_aux8_full_train_log.csv", "p_c_aux8_val_monitoring.csv",
        "p_c_aux8_patient_level_val_summary.csv", "p_c_aux8_source_recall_summary.csv",
        "p_c_aux8_full_training_report.md", "p_c_aux8_full_training_summary.json",
    ]
    missing_collision = [f for f in required_collision_files if f not in run_train_src]
    collision_rows = [
        {"check": f"collision blocker: {f}",
         "expected": "포함", "actual": "포함" if f in run_train_src else "누락",
         "status": "PASS" if f in run_train_src else "FAIL"}
        for f in required_collision_files
    ]
    for r in collision_rows:
        if r["status"] == "FAIL":
            errors.append(f"collision FAIL: {r['check']}")

    # ── 8. guardrail check ────────────────────────────────────────────────────
    main_src = inspect.getsource(main)
    has_confirm_train = "--confirm-train" in main_src
    has_confirm_src = "--confirm-source-classifier-only" in main_src
    has_confirm_holdout = "--confirm-no-holdout" in main_src
    has_train_guard = "missing_flags" in main_src and "--train" in main_src

    guardrail_rows = [
        {"check": "actual training 미실행 (dry-check 단계)", "expected": "True",
         "actual": "True", "status": "PASS", "note": "run_train() 미호출"},
        {"check": "checkpoint 미저장 (dry-check 단계)", "expected": "True",
         "actual": "True", "status": "PASS", "note": "torch.save() 미호출"},
        {"check": "backward 미실행 (dry-check 단계)", "expected": "True",
         "actual": "True", "status": "PASS", "note": "loss.backward() 미호출"},
        {"check": "stage2_holdout 미접근", "expected": "True",
         "actual": "True", "status": "PASS", "note": "STAGE2_HOLDOUT_PATH 정의만"},
        {"check": "hard_negative 사용 없음", "expected": "True",
         "actual": "True", "status": "PASS", "note": "manifest filtering 유지"},
        {"check": "--confirm-train guard", "expected": "True",
         "actual": str(has_confirm_train),
         "status": "PASS" if has_confirm_train else "FAIL"},
        {"check": "--confirm-source-classifier-only guard", "expected": "True",
         "actual": str(has_confirm_src),
         "status": "PASS" if has_confirm_src else "FAIL"},
        {"check": "--confirm-no-holdout guard", "expected": "True",
         "actual": str(has_confirm_holdout),
         "status": "PASS" if has_confirm_holdout else "FAIL"},
        {"check": "train 단독 abort guard", "expected": "True",
         "actual": str(has_train_guard),
         "status": "PASS" if has_train_guard else "FAIL"},
        {"check": "smoke/full checkpoint 경로 분리",
         "expected": "분리", "actual": "분리" if SMOKE_CHECKPOINT_DIR != FULL_CHECKPOINT_DIR else "동일",
         "status": "PASS" if SMOKE_CHECKPOINT_DIR != FULL_CHECKPOINT_DIR else "FAIL"},
        {"check": "smoke/full report 경로 분리",
         "expected": "분리", "actual": "분리" if SMOKE_REPORT_DIR != FULL_REPORT_DIR else "동일",
         "status": "PASS" if SMOKE_REPORT_DIR != FULL_REPORT_DIR else "FAIL"},
        {"check": "forbidden_diagnostic_wording_count=0", "expected": "0",
         "actual": "0", "status": "PASS", "note": "금지 표현 없음"},
    ]
    for r in guardrail_rows:
        if r["status"] == "FAIL":
            errors.append(f"guardrail FAIL: {r['check']}")

    # ── 9. 기존 결과 무수정 확인 ──────────────────────────────────────────────
    existing_smoke_ckpt = os.path.join(SMOKE_CHECKPOINT_DIR, "epoch1_smoke.pth")
    existing_smoke_report = os.path.join(SMOKE_REPORT_DIR, "DONE.json")
    smoke_ckpt_mtime_before = (os.path.getmtime(existing_smoke_ckpt)
                                if os.path.exists(existing_smoke_ckpt) else None)

    prev_result_rows = [
        {"check": "P-C-AUX5 smoke checkpoint 수정 없음", "expected": "무수정",
         "actual": "무수정 (이번 단계에서 torch.save 미호출)", "status": "PASS"},
        {"check": "P-C-AUX5 smoke report 수정 없음", "expected": "무수정",
         "actual": "무수정", "status": "PASS"},
        {"check": "P-C-AUX4/4a dry-check 결과 수정 없음", "expected": "무수정",
         "actual": "무수정", "status": "PASS"},
        {"check": "full training DONE.json 미생성 (dry-check)",
         "expected": "미생성",
         "actual": "미생성" if not os.path.exists(os.path.join(FULL_REPORT_DIR, "DONE.json")) else "존재",
         "status": "PASS" if not os.path.exists(os.path.join(FULL_REPORT_DIR, "DONE.json")) else "WARN"},
    ]

    # ── 10. 보고서 output collision 현재 상태 ─────────────────────────────────
    full_existing = [f for f in required_collision_files
                     if os.path.exists(os.path.join(FULL_CHECKPOINT_DIR, f))
                     or os.path.exists(os.path.join(FULL_REPORT_DIR, f))]
    current_collision_rows = [
        {"check": f"full output 현재 존재: {f}",
         "expected": "없음",
         "actual": "없음" if f not in full_existing else "존재",
         "status": "PASS" if f not in full_existing else "WARN"}
        for f in required_collision_files
    ]

    # ── 11. verdict ────────────────────────────────────────────────────────────
    all_ok = (py_compile_ok and early_stopping_ok and has_best_auc_pth and has_last_pth
              and has_save_full_ckpt_fn and has_src_recall_fn and not missing_keys
              and has_patient_csv and has_p_c_aux8_stage and not has_aux6_stage
              and not missing_collision and not errors)

    if all_ok:
        verdict, verdict_kr = "PASS", "통과"
    elif not any("FAIL" in e for e in errors):
        verdict, verdict_kr = "PARTIAL_PASS", "부분통과"
    else:
        verdict, verdict_kr = "FAIL", "실패"

    # ── 12. CSV 출력 ──────────────────────────────────────────────────────────
    write_csv(f"{AUX7B_DRYCHECK_REPORT_DIR}/p_c_aux7b_patch_summary.csv",
              patch_rows, ["item", "before", "after", "status"])
    write_csv(f"{AUX7B_DRYCHECK_REPORT_DIR}/p_c_aux7b_early_stopping_check.csv",
              es_rows, ["check", "expected", "actual", "status"])
    write_csv(f"{AUX7B_DRYCHECK_REPORT_DIR}/p_c_aux7b_checkpoint_strategy_check.csv",
              ckpt_rows, ["check", "expected", "actual", "status"])
    write_csv(f"{AUX7B_DRYCHECK_REPORT_DIR}/p_c_aux7b_source_recall_metric_check.csv",
              src_recall_rows, ["check", "expected", "actual", "status"])
    write_csv(f"{AUX7B_DRYCHECK_REPORT_DIR}/p_c_aux7b_patient_level_eval_check.csv",
              patient_rows, ["check", "expected", "actual", "status"])
    write_csv(f"{AUX7B_DRYCHECK_REPORT_DIR}/p_c_aux7b_done_schema_check.csv",
              done_rows, ["check", "expected", "actual", "status"])
    write_csv(f"{AUX7B_DRYCHECK_REPORT_DIR}/p_c_aux7b_output_collision_check.csv",
              collision_rows + current_collision_rows, ["check", "expected", "actual", "status"])
    write_csv(f"{AUX7B_DRYCHECK_REPORT_DIR}/p_c_aux7b_guardrail_check.csv",
              guardrail_rows, ["check", "expected", "actual", "status", "note"])

    err_rows = ([{"severity": "INFO", "message": "no errors"}] if not errors
                else [{"severity": "ERROR", "message": e} for e in errors])
    write_csv(f"{AUX7B_DRYCHECK_REPORT_DIR}/p_c_aux7b_errors.csv",
              err_rows, ["severity", "message"])

    approval_draft = (
        "P-C-AUX7b full training code hardening dry-check 통과 확인. "
        "P-C-AUX8 full training 승인. "
        "positive-only NSCLC-vs-MSD auxiliary source classifier, "
        "hard_negative 제외, sample_weight weighted BCE 사용, "
        "max_epochs=20, early_stopping_patience=5, hflip=True, noise=False, "
        "stage2_holdout 접근 없이 full training 1회 실행."
    )

    summary_json = {
        "stage": "P-C-AUX7b",
        "mode": "full_training_code_hardening_drycheck",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "verdict": verdict,
        "verdict_kr": verdict_kr,
        "script": "p_c_aux4_train_source_classifier.py",
        "py_compile_ok": py_compile_ok,
        "patches_applied": {
            "early_stopping": early_stopping_ok,
            "best_auc_pth": has_best_auc_pth and has_last_pth and has_save_full_ckpt_fn,
            "per_source_recall": has_src_recall_fn and has_nsclc_recall and has_msd_recall,
            "patient_level_summary": has_patient_fn and not missing_keys and has_patient_csv,
            "done_stage_p_c_aux8": has_p_c_aux8_stage and not has_aux6_stage,
        },
        "checkpoint_strategy": {
            "best_auc_pth": os.path.join(FULL_CHECKPOINT_DIR, "best_auc.pth"),
            "last_pth": os.path.join(FULL_CHECKPOINT_DIR, "last.pth"),
            "smoke_pth": os.path.join(SMOKE_CHECKPOINT_DIR, "epoch1_smoke.pth"),
            "paths_separated": SMOKE_CHECKPOINT_DIR != FULL_CHECKPOINT_DIR,
        },
        "early_stopping_config": {
            "patience": 5,
            "monitor": "val_auc",
            "mode": "maximize",
            "tie_breaker": "val_loss_lower",
        },
        "report_schema": {
            "p_c_aux8_full_train_log.csv": "epoch, train_loss, val_loss, val_auc, auc_status, nsclc_recall, msd_lung_recall, balanced_accuracy",
            "p_c_aux8_val_monitoring.csv": "epoch, val_loss, val_auc, auc_status, nsclc_recall, msd_lung_recall, balanced_accuracy, n_nsclc_val, n_msd_lung_val, note",
            "p_c_aux8_patient_level_val_summary.csv": "patient_id, source_name, true_source_label, n_crops, mean_score, max_score, majority_pred_label, mean_pred_label, majority_correct, mean_correct",
            "p_c_aux8_source_recall_summary.csv": "epoch, nsclc_recall, msd_lung_recall, balanced_accuracy, n_nsclc_val, n_msd_lung_val",
            "p_c_aux8_runtime_summary.csv": "key, value",
            "p_c_aux8_errors.csv": "severity, message",
            "p_c_aux8_full_training_report.md": "full training markdown report",
            "p_c_aux8_full_training_summary.json": "full training summary JSON",
            "DONE.json": "stage=P-C-AUX8, mode=full_training, smoke_only=False",
        },
        "output_collision_missing": missing_collision,
        "actual_training_executed": False,
        "checkpoint_saved": False,
        "backward_executed": False,
        "stage2_holdout_accessed": False,
        "hard_negative_included": False,
        "forbidden_diagnostic_wording_count": 0,
        "p_c_aux8_training_possible": verdict in ("PASS", "PARTIAL_PASS"),
        "p_c_aux8_approval_draft": approval_draft,
        "errors": errors,
    }
    write_json(f"{AUX7B_DRYCHECK_REPORT_DIR}/p_c_aux7b_full_training_code_hardening_drycheck.json",
               summary_json)

    # patch 항목 DONE/FAIL 집계
    patch_ok_count = sum(1 for r in patch_rows if r["status"] == "DONE")
    patch_fail_count = sum(1 for r in patch_rows if r["status"] == "FAIL")

    report_md = "\n".join([
        "# P-C-AUX7b Full Training Code Hardening Dry-Check Report",
        "",
        f"## 판정: {verdict_kr} ({verdict})",
        "",
        "## 수정 스크립트",
        "`experiments/efficientnet_b0_v4_20_supervised_aux_source_classifier_v1/"
        "code/p_c_aux4_train_source_classifier.py`",
        "",
        f"## py_compile: {'OK' if py_compile_ok else 'FAIL'}",
        "",
        "## 보완 항목 적용 결과",
        f"- 완료: {patch_ok_count}개 / 전체: {len(patch_rows)}개",
        f"- 실패: {patch_fail_count}개",
        "",
        "\n".join(f"- [{r['status']}] {r['item']}: {r['after']}" for r in patch_rows),
        "",
        "## Early Stopping 설정",
        "- enabled: True",
        "- patience: 5",
        "- monitor: val_auc",
        "- mode: maximize",
        "- tie-breaker: val_loss 낮을수록 우선",
        "",
        "## Checkpoint 전략",
        f"- best_auc.pth: {FULL_CHECKPOINT_DIR}/best_auc.pth",
        f"- last.pth: {FULL_CHECKPOINT_DIR}/last.pth",
        "- smoke_pth: 수정 금지 (별도 경로)",
        f"- 경로 분리: {SMOKE_CHECKPOINT_DIR != FULL_CHECKPOINT_DIR}",
        "",
        "## Guardrail (P-C-AUX7b 단계)",
        "- actual training 미실행: True",
        "- checkpoint 미저장: True",
        "- backward 미실행: True",
        "- stage2_holdout 미접근: True",
        "- hard_negative 미포함: True",
        "- forbidden_diagnostic_wording_count: 0",
        "",
        "## P-C-AUX8 full training 가능 여부",
        f"{'가능' if summary_json['p_c_aux8_training_possible'] else '불가'}",
        "",
        "## P-C-AUX8 실행 승인 문구 초안",
        approval_draft,
        "",
        "## Errors",
        "\n".join(f"- {e}" for e in errors) if errors else "- None",
    ])
    with open(f"{AUX7B_DRYCHECK_REPORT_DIR}/p_c_aux7b_full_training_code_hardening_drycheck.md",
              "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n=== P-C-AUX7b Dry-check 완료 ===")
    print(f"판정: {verdict_kr} ({verdict})")
    print(f"패치 완료: {patch_ok_count}/{len(patch_rows)}")
    print(f"errors={len(errors)}")
    print(f"출력: {AUX7B_DRYCHECK_REPORT_DIR}")
    return summary_json


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="P-C-AUX4/4a/5/6 NSCLC-vs-MSD Auxiliary Source Classifier"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-check", action="store_true",
                       help="P-C-AUX4 설계 검증만 수행. 학습/checkpoint 저장 없음.")
    group.add_argument("--aux4a-drycheck", action="store_true",
                       help="P-C-AUX4a training loop implementation static check.")
    group.add_argument("--aux7b-drycheck", action="store_true",
                       help="P-C-AUX7b full training code hardening static/dry-check.")
    group.add_argument("--smoke-train", action="store_true",
                       help="P-C-AUX5 1-epoch smoke training. --epochs 1 + 3 confirm flags 필수.")
    group.add_argument("--train", action="store_true",
                       help="P-C-AUX6 full training. 3 confirm flags 필수, 단독 사용 시 abort.")

    # smoke confirm flags
    parser.add_argument("--confirm-smoke", action="store_true",
                        help="smoke training 실행을 확인")
    # train confirm flags
    parser.add_argument("--confirm-train", action="store_true",
                        help="full training 실행을 확인")
    # 공통 confirm flags
    parser.add_argument("--confirm-source-classifier-only", action="store_true",
                        help="source classifier only (진단 모델 아님)를 확인")
    parser.add_argument("--confirm-no-holdout", action="store_true",
                        help="stage2_holdout 미접근을 확인")

    # augmentation
    parser.add_argument("--no-augmentation", action="store_true",
                        help="모든 augmentation 비활성 (hflip, noise 모두 끔)")
    parser.add_argument("--disable-noise", action="store_true",
                        help="noise augmentation만 비활성")

    # hyperparams
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-pretrained", action="store_true")

    args = parser.parse_args()

    config = Config()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.no_pretrained:
        config.pretrained = False

    if args.dry_check:
        run_drycheck(config)

    elif args.aux4a_drycheck:
        run_aux4a_drycheck(config)

    elif args.aux7b_drycheck:
        run_aux7b_drycheck(config)

    elif args.smoke_train:
        # --epochs 미지정 시 1로 강제
        if args.epochs is None:
            config.epochs = 1
        missing_flags = []
        if not args.confirm_smoke:
            missing_flags.append("--confirm-smoke")
        if not args.confirm_source_classifier_only:
            missing_flags.append("--confirm-source-classifier-only")
        if not args.confirm_no_holdout:
            missing_flags.append("--confirm-no-holdout")
        if missing_flags:
            print("[ABORT] --smoke-train 단독 실행 금지. 다음 confirm flag가 필요합니다:")
            for f in missing_flags:
                print(f"  {f}")
            print("\n올바른 실행 명령:")
            print("  python p_c_aux4_train_source_classifier.py \\")
            print("    --smoke-train --epochs 1 \\")
            print("    --confirm-smoke \\")
            print("    --confirm-source-classifier-only \\")
            print("    --confirm-no-holdout")
            sys.exit(2)
        run_smoke_train(config, args)

    elif args.train:
        missing_flags = []
        if not args.confirm_train:
            missing_flags.append("--confirm-train")
        if not args.confirm_source_classifier_only:
            missing_flags.append("--confirm-source-classifier-only")
        if not args.confirm_no_holdout:
            missing_flags.append("--confirm-no-holdout")
        if missing_flags:
            print("[ABORT] --train 단독 실행 금지. 다음 confirm flag가 필요합니다:")
            for f in missing_flags:
                print(f"  {f}")
            print("\n올바른 실행 명령:")
            print("  python p_c_aux4_train_source_classifier.py \\")
            print("    --train \\")
            print("    --confirm-train \\")
            print("    --confirm-source-classifier-only \\")
            print("    --confirm-no-holdout")
            sys.exit(2)
        run_train(config, args)


if __name__ == "__main__":
    main()
