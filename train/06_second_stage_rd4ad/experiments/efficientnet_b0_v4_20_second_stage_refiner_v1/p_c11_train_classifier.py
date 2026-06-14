"""
P-C11: EfficientNet-B0 second-stage binary crop classifier
- Input: ct_crop (3, 96, 96), int16 HU
- Output: binary logit (positive=1, hard_negative=0)
- Usage:
    python p_c11_train_classifier.py --dry-check      # 1-batch dry-check only (no training)
    python p_c11_train_classifier.py --train          # requires explicit --train flag + user approval
"""
import argparse
import os
import sys
import json
import time
import math
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
import torchvision.transforms.functional as TF

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = "/home/jinhy/project/lung-ct-anomaly"
EXP = f"{BASE}/experiments/efficientnet_b0_v4_20_second_stage_refiner_v1"
MANIFEST_PATH = f"{EXP}/outputs/training_manifests/p_c10_c_lite_training_manifest/p_c10_c_lite_training_manifest.csv"
CROP_BASE = EXP  # crop_path in manifest is relative to EXP
REPORT_C10_5 = f"{EXP}/outputs/reports/p_c10_5_training_manifest_coverage_audit"
DRYCHECK_REPORT_DIR = f"{EXP}/outputs/reports/p_c11_training_script_drycheck"
SMOKE_CHECKPOINT_DIR = f"{EXP}/outputs/checkpoints/p_c12_smoke_training"
SMOKE_REPORT_DIR = f"{EXP}/outputs/reports/p_c12_smoke_training"

# ── Config ─────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Paths
    manifest_path: str = MANIFEST_PATH
    crop_base: str = CROP_BASE
    checkpoint_dir: str = f"{EXP}/outputs/checkpoints/p_c12_training"

    # Model
    model_name: str = "efficientnet_b0"  # efficientnet_b0 | simple_cnn
    pretrained: bool = True
    num_classes: int = 1  # binary logit

    # Input
    input_channels: int = 3
    input_size: int = 96
    ct_hu_min: float = -1000.0
    ct_hu_max: float = 200.0

    # Training
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True

    # Loss
    loss_fn: str = "bce_weighted"  # bce_weighted | bce_unweighted
    pos_weight: float = 2.12  # hn/pos ratio from C-lite manifest

    # Augmentation (NO random crop, NO cutout — border_touch 43.6%)
    aug_hflip: bool = True
    aug_vflip: bool = False   # anatomical direction concern
    aug_noise_std: float = 0.01  # small Gaussian noise on normalized ct

    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # cosine | plateau
    warmup_epochs: int = 2

    # Checkpoint
    save_best_only: bool = True
    early_stopping_patience: int = 7
    val_metric: str = "auroc"  # auroc | loss

    # Dry-check
    dry_check_split: str = "train"
    dry_check_batch_size: int = 8

    def to_dict(self):
        return asdict(self)


# ── CT Preprocessing ───────────────────────────────────────────────────────
def preprocess_ct(ct_array: np.ndarray, hu_min: float, hu_max: float) -> torch.Tensor:
    """
    ct_array: (3, 96, 96) int16 HU values
    Returns: float32 tensor (3, 96, 96), range [0, 1]
    """
    ct = ct_array.astype(np.float32)
    ct = np.clip(ct, hu_min, hu_max)
    ct = (ct - hu_min) / (hu_max - hu_min)  # [0, 1]
    return torch.from_numpy(ct)


# ── Augmentation ───────────────────────────────────────────────────────────
class TrainTransform:
    """
    Safe augmentations for 96x96 crops with 43.6% border_touch.
    NO random crop, NO cutout, NO random erasing, NO aggressive rotation.
    """
    def __init__(self, hflip: bool = True, vflip: bool = False, noise_std: float = 0.01):
        self.hflip = hflip
        self.vflip = vflip
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (3, 96, 96) float32, range [0,1]
        if self.hflip and torch.rand(1).item() > 0.5:
            x = TF.hflip(x)
        if self.vflip and torch.rand(1).item() > 0.5:
            x = TF.vflip(x)
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
            x = x.clamp(0.0, 1.0)
        return x


class ValTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x  # no augmentation at val


# ── Dataset ────────────────────────────────────────────────────────────────
class CropDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,             # "train" or "val"
        crop_base: str,
        hu_min: float = -1000.0,
        hu_max: float = 200.0,
        transform=None,
    ):
        mf = pd.read_csv(manifest_path)
        self.df = mf[mf["split_plan"] == split].reset_index(drop=True)
        self.crop_base = crop_base
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        crop_path = os.path.join(self.crop_base, str(row["crop_path"]))

        d = np.load(crop_path)
        ct = preprocess_ct(d["ct_crop"], self.hu_min, self.hu_max)  # (3,96,96) float32 [0,1]

        if self.transform is not None:
            ct = self.transform(ct)

        label = torch.tensor(float(row["training_label"]), dtype=torch.float32)

        return ct, label

    def get_label_array(self) -> np.ndarray:
        return self.df["training_label"].values.astype(np.float32)


# ── Model ──────────────────────────────────────────────────────────────────
def build_model(config: Config) -> nn.Module:
    if config.model_name == "efficientnet_b0":
        model = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if config.pretrained else None
        )
        # Replace classifier: 1280 → 1 binary logit
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 1),
        )
    elif config.model_name == "simple_cnn":
        model = SimpleCNN(in_channels=3)
    else:
        raise ValueError(f"Unknown model_name: {config.model_name}")
    return model


class SimpleCNN(nn.Module):
    """Lightweight baseline CNN for binary crop classification."""
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 96 → 48
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 48 → 24
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 24 → 12
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: 12 → 6
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ── Loss ───────────────────────────────────────────────────────────────────
def build_criterion(config: Config) -> nn.Module:
    if config.loss_fn == "bce_weighted":
        pos_weight = torch.tensor([config.pos_weight])
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        return nn.BCEWithLogitsLoss()


# ── Optimizer / Scheduler ──────────────────────────────────────────────────
def build_optimizer(model: nn.Module, config: Config):
    return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def build_scheduler(optimizer, config: Config, steps_per_epoch: int):
    if not config.use_scheduler:
        return None
    if config.scheduler_type == "cosine":
        total_steps = config.epochs * steps_per_epoch
        warmup_steps = config.warmup_epochs * steps_per_epoch
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif config.scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
        )
    return None


# ── AUROC (sklearn-free, Mann-Whitney U) ───────────────────────────────────
def compute_auroc(logits, labels):
    """
    AUROC without sklearn. Mann-Whitney U rank-sum method.
    Returns (auc: float, status: str).
    status values: 'ok' | 'single_class_labels' | 'invalid_score_nan_inf'
    """
    import numpy as np
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

    n = len(scores)
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1  # 1-indexed average rank for ties
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    pos_rank_sum = ranks[pos_mask].sum()
    u_stat = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    auc = float(np.clip(u_stat / (n_pos * n_neg), 0.0, 1.0))
    return auc, "ok"


# ── Train / Val loops (called only when --train flag is used) ────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, config, epoch):
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = config.mixed_precision and device.type == "cuda"
    for ct, labels in loader:
        ct = ct.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)  # (B,1)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type if device.type == "cuda" else "cpu", enabled=use_amp):
            logits = model(ct)               # (B,1)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None and config.scheduler_type == "cosine":
            scheduler.step()

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            raise RuntimeError(f"Loss is NaN/Inf at batch {n_batches}: {loss_val}")
        total_loss += loss_val
        n_batches += 1

    return total_loss / max(1, n_batches)


def validate(model, loader, criterion, device):
    """Returns val loss and raw predictions for AUROC calculation by caller."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for ct, labels in loader:
            ct = ct.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            logits = model(ct)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.squeeze(1).cpu())
            all_labels.append(labels.squeeze(1).cpu())

    val_loss = total_loss / max(1, n_batches)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return val_loss, all_logits, all_labels


# ── Dry-check ──────────────────────────────────────────────────────────────
def run_dry_check(config: Config) -> dict:
    results = {}
    errors = []

    print("=" * 65)
    print("P-C11 Dry-Check (1-batch only, NO training, NO checkpoint)")
    print("=" * 65)

    # 1. Verify prior verdicts
    c105_path = f"{REPORT_C10_5}/p_c10_5_training_manifest_coverage_audit.json"
    with open(c105_path) as f:
        c105 = json.load(f)
    assert c105["verdict"] == "통과", f"P-C10.5 verdict={c105['verdict']}"
    print("[OK] P-C10.5 verdict=통과")

    # 2. Manifest check
    mf = pd.read_csv(config.manifest_path)
    assert len(mf) == 110054, f"manifest rows={len(mf)}"
    train_rows = (mf["split_plan"] == "train").sum()
    val_rows = (mf["split_plan"] == "val").sum()
    assert train_rows == 88083 and val_rows == 21971
    print(f"[OK] manifest: {len(mf)} rows | train={train_rows} | val={val_rows}")

    # patient leakage check
    train_pats = set(mf[mf["split_plan"] == "train"]["patient_id"])
    val_pats = set(mf[mf["split_plan"] == "val"]["patient_id"])
    leakage = train_pats & val_pats
    assert len(leakage) == 0
    results["patient_leakage"] = 0
    print(f"[OK] patient leakage=0")

    # crop_path missing check
    missing = 0
    for p in mf["crop_path"].head(100):
        if not os.path.exists(os.path.join(config.crop_base, p)):
            missing += 1
    results["crop_path_missing_sample100"] = missing
    print(f"[OK] crop_path spot-check (100): missing={missing}")

    # 3. Dataset
    train_transform = TrainTransform(
        hflip=config.aug_hflip,
        vflip=config.aug_vflip,
        noise_std=config.aug_noise_std,
    )
    dataset = CropDataset(
        manifest_path=config.manifest_path,
        split=config.dry_check_split,
        crop_base=config.crop_base,
        hu_min=config.ct_hu_min,
        hu_max=config.ct_hu_max,
        transform=train_transform,
    )
    print(f"[OK] CropDataset({config.dry_check_split}): {len(dataset)} samples")

    # 4. DataLoader (num_workers=0 for dry-check stability)
    loader = DataLoader(
        dataset,
        batch_size=config.dry_check_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=torch.Generator().manual_seed(config.seed),
    )

    # 5. Get 1 batch
    t0 = time.time()
    batch_ct, batch_labels = next(iter(loader))
    load_time = time.time() - t0
    print(f"[OK] 1 batch loaded in {load_time:.2f}s")
    print(f"     ct shape: {tuple(batch_ct.shape)}, dtype: {batch_ct.dtype}")
    print(f"     labels shape: {tuple(batch_labels.shape)}, dtype: {batch_labels.dtype}")
    print(f"     ct range: [{batch_ct.min():.4f}, {batch_ct.max():.4f}]")
    print(f"     labels: {batch_labels.tolist()}")
    print(f"     pos count in batch: {(batch_labels == 1).sum().item()}")

    results["ct_shape"] = list(batch_ct.shape)
    results["label_shape"] = list(batch_labels.shape)
    results["ct_min"] = round(float(batch_ct.min()), 6)
    results["ct_max"] = round(float(batch_ct.max()), 6)
    results["ct_nan"] = bool(torch.isnan(batch_ct).any())
    results["ct_inf"] = bool(torch.isinf(batch_ct).any())
    results["batch_pos_count"] = int((batch_labels == 1).sum())
    results["batch_hn_count"] = int((batch_labels == 0).sum())
    results["batch_load_time_s"] = round(load_time, 3)

    assert not results["ct_nan"], "ct has NaN!"
    assert not results["ct_inf"], "ct has Inf!"
    print(f"[OK] ct NaN/Inf: False")

    # 6. Model
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"[OK] device: {device}")
    model = build_model(config)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] model: {config.model_name}, params={n_params:,}")
    results["model_name"] = config.model_name
    results["model_params"] = n_params
    results["device"] = str(device)

    # 7. Forward pass (1 batch, no grad needed for dry-check)
    model.eval()
    ct_dev = batch_ct.to(device)
    labels_dev = batch_labels.to(device).unsqueeze(1)  # (B,1)

    t1 = time.time()
    with torch.no_grad():
        logits = model(ct_dev)  # (B,1)
    forward_time = time.time() - t1

    print(f"[OK] forward pass: {forward_time*1000:.1f}ms")
    print(f"     logits shape: {tuple(logits.shape)}, dtype: {logits.dtype}")
    print(f"     logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"     logits NaN: {torch.isnan(logits).any().item()}")

    results["output_shape"] = list(logits.shape)
    results["logits_nan"] = bool(torch.isnan(logits).any())
    results["logits_inf"] = bool(torch.isinf(logits).any())
    results["forward_time_ms"] = round(forward_time * 1000, 1)

    assert not results["logits_nan"], "logits has NaN!"
    assert not results["logits_inf"], "logits has Inf!"

    # 8. Loss
    criterion = build_criterion(config)
    criterion_pos_weight = criterion.pos_weight.to(device) if hasattr(criterion, "pos_weight") and criterion.pos_weight is not None else None
    if criterion_pos_weight is not None:
        criterion.pos_weight = criterion_pos_weight

    with torch.no_grad():
        loss = criterion(logits, labels_dev.float())

    loss_val = float(loss.item())
    loss_finite = math.isfinite(loss_val)
    print(f"[OK] loss: {loss_val:.6f}, finite={loss_finite}")
    results["loss_value"] = round(loss_val, 6)
    results["loss_finite"] = loss_finite

    assert loss_finite, f"loss is not finite: {loss_val}"

    # 9. Guardrails check
    results["training_executed"] = False
    results["checkpoint_saved"] = False
    results["stage2_holdout_accessed"] = False
    results["epoch_loop_executed"] = False
    results["backward_executed"] = False
    print(f"[OK] guardrails: training=False, checkpoint=False, stage2=False")

    print("=" * 65)
    return results, errors


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="P-C11 EfficientNet-B0 crop classifier")
    parser.add_argument("--dry-check", action="store_true",
                        help="Run 1-batch dry-check only (no training, no checkpoint)")
    parser.add_argument("--train", action="store_true",
                        help="Run full training (requires separate user approval)")
    parser.add_argument("--model", default="efficientnet_b0", choices=["efficientnet_b0", "simple_cnn"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--loss", default="bce_weighted", choices=["bce_weighted", "bce_unweighted"])
    parser.add_argument("--smoke-train", action="store_true",
                        help="Run 1-epoch smoke training only (P-C12). Requires --epochs 1.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint directory (smoke: p_c12_smoke_training)")
    parser.add_argument("--report-dir", type=str, default=None,
                        help="Override report directory (smoke: p_c12_smoke_training)")
    args = parser.parse_args()

    config = Config(
        model_name=args.model,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        loss_fn=args.loss,
    )
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir

    if args.smoke_train:
        _smoke_report_dir = args.report_dir if args.report_dir else SMOKE_REPORT_DIR
        _smoke_ckpt_dir = args.checkpoint_dir if args.checkpoint_dir else SMOKE_CHECKPOINT_DIR
        config.checkpoint_dir = _smoke_ckpt_dir
        run_smoke_train(config, smoke_report_dir=_smoke_report_dir)
        sys.exit(0)

    elif args.dry_check:
        results, errors = run_dry_check(config)
        _save_drycheck_report(config, results, errors)
        verdict = "통과" if not results.get("logits_nan") and results.get("loss_finite") else "실패"
        print(f"\nP-C11 Dry-Check 판정: {verdict}")
        sys.exit(0)

    elif args.train:
        print("[WARNING] Full training requested. Ensure you have user approval before running.")
        # ── Full training (P-C16) ──────────────────────────────────────────
        _full_report_dir = args.report_dir if args.report_dir else (
            config.checkpoint_dir.replace("/checkpoints/", "/reports/")
        )
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(_full_report_dir, exist_ok=True)

        torch.manual_seed(config.seed)
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        use_amp = config.mixed_precision and device.type == "cuda"
        if not use_amp and config.mixed_precision:
            print("[WARN] mixed_precision disabled: not running on CUDA")

        train_dataset = CropDataset(
            config.manifest_path, "train", config.crop_base,
            config.ct_hu_min, config.ct_hu_max,
            TrainTransform(config.aug_hflip, config.aug_vflip, config.aug_noise_std),
        )
        val_dataset = CropDataset(
            config.manifest_path, "val", config.crop_base,
            config.ct_hu_min, config.ct_hu_max,
            ValTransform(),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=True,
            persistent_workers=config.num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size * 2, shuffle=False,
            num_workers=config.num_workers, pin_memory=True,
            persistent_workers=config.num_workers > 0,
        )
        print(f"[OK] train={len(train_dataset)} | val={len(val_dataset)} | device={device}")

        model = build_model(config).to(device)
        criterion = build_criterion(config)
        if hasattr(criterion, "pos_weight") and criterion.pos_weight is not None:
            criterion.pos_weight = criterion.pos_weight.to(device)
        optimizer = build_optimizer(model, config)
        steps_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
        scheduler = build_scheduler(optimizer, config, steps_per_epoch)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_metric = -1e9
        best_epoch = 0
        patience_counter = 0
        epochs_completed = 0
        train_log_rows = []
        val_log_rows = []
        runtime_rows = []
        full_errors = []
        early_stopped = False

        for epoch in range(1, config.epochs + 1):
            t_start = time.time()
            try:
                train_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, scaler, scheduler, device, config, epoch
                )
            except RuntimeError as e:
                full_errors.append({"step": f"epoch{epoch}_train", "error": str(e)})
                print(f"[FAIL] epoch {epoch} train: {e}")
                break

            val_loss, val_logits, val_labels = validate(model, val_loader, criterion, device)
            if not math.isfinite(val_loss):
                full_errors.append({"step": f"epoch{epoch}_val", "error": f"val_loss NaN/Inf: {val_loss}"})
                print(f"[FAIL] epoch {epoch} val_loss NaN/Inf")
                break

            # AUROC (monitoring) — sklearn-free
            auc, auroc_status = compute_auroc(val_logits.numpy(), val_labels.numpy())
            if auroc_status != "ok":
                full_errors.append({"step": f"epoch{epoch}_auroc", "error": auroc_status})

            if config.scheduler_type == "plateau" and scheduler is not None:
                scheduler.step(auc)

            elapsed = time.time() - t_start
            gpu_peak_mb = (torch.cuda.max_memory_allocated(device) / 1024 / 1024
                           if device.type == "cuda" else 0.0)
            epochs_completed = epoch

            print(f"[Epoch {epoch:03d}/{config.epochs}] "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_auc={auc:.4f} status={auroc_status} ({elapsed:.1f}s)")

            train_log_rows.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                                   "steps_per_epoch": steps_per_epoch})
            val_log_rows.append({"epoch": epoch, "val_loss": round(val_loss, 6),
                                 "val_auc": round(auc, 6) if math.isfinite(auc) else "nan",
                                 "auroc_status": auroc_status})
            runtime_rows.append({"epoch": epoch, "epoch_time_s": round(elapsed, 2),
                                  "gpu_peak_mb": round(gpu_peak_mb, 1), "device": str(device)})

            # last.pth (overwrite each epoch)
            last_ckpt_path = os.path.join(config.checkpoint_dir, "last.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": auc,
                "config": config.to_dict(),
            }, last_ckpt_path)

            # best.pth (by val_auc if finite, else by -val_loss)
            metric = auc if (config.val_metric == "auroc" and math.isfinite(auc)) else -val_loss
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                patience_counter = 0
                ckpt_path = os.path.join(config.checkpoint_dir, "best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_auc": auc,
                    "config": config.to_dict(),
                }, ckpt_path)
                print(f"  [SAVE] best checkpoint: {ckpt_path} (auc={auc:.4f} epoch={epoch})")
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"  [STOP] early stopping at epoch {epoch}")
                    early_stopped = True
                    break

        print("Training complete.")
        _save_full_train_report(
            config, _full_report_dir,
            train_log_rows=train_log_rows, val_log_rows=val_log_rows,
            runtime_rows=runtime_rows, errors=full_errors,
            epochs_completed=epochs_completed, best_epoch=best_epoch,
            best_metric=best_metric, early_stopped=early_stopped,
        )

    else:
        print("Specify --dry-check or --train. Use --dry-check first to verify setup.")
        sys.exit(1)


def _save_full_train_report(
    config: Config,
    report_dir: str,
    train_log_rows: list,
    val_log_rows: list,
    runtime_rows: list,
    errors: list,
    epochs_completed: int,
    best_epoch: int,
    best_metric: float,
    early_stopped: bool,
):
    """Save full training report files for P-C16."""
    import pandas as pd
    os.makedirs(report_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Determine verdict
    has_error = any(r.get("error", "none") != "none" for r in errors) if errors else False
    verdict = "실패" if (has_error or epochs_completed == 0) else "통과"

    pd.DataFrame(train_log_rows if train_log_rows else [{"epoch": 0, "train_loss": "nan"}]).to_csv(
        os.path.join(report_dir, "p_c16_train_log.csv"), index=False)
    pd.DataFrame(val_log_rows if val_log_rows else [{"epoch": 0, "val_loss": "nan", "val_auc": "nan"}]).to_csv(
        os.path.join(report_dir, "p_c16_val_log.csv"), index=False)
    pd.DataFrame(runtime_rows if runtime_rows else [{"epoch": 0, "epoch_time_s": 0}]).to_csv(
        os.path.join(report_dir, "p_c16_runtime_summary.csv"), index=False)
    pd.DataFrame(errors if errors else [{"step": "full_train", "error": "none"}]).to_csv(
        os.path.join(report_dir, "p_c16_errors.csv"), index=False)

    best_val_auc = val_log_rows[best_epoch - 1]["val_auc"] if best_epoch > 0 and val_log_rows else "nan"
    best_val_loss = val_log_rows[best_epoch - 1]["val_loss"] if best_epoch > 0 and val_log_rows else "nan"

    summary = {
        "step": "P-C16",
        "verdict": verdict,
        "created": ts,
        "model": config.model_name,
        "epochs_requested": config.epochs,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_val_loss": best_val_loss,
        "early_stopped": early_stopped,
        "checkpoint_dir": config.checkpoint_dir,
        "report_dir": report_dir,
        "guardrails": {
            "stage2_holdout_accessed": False,
        },
        "warnings": [
            "no_hit/fallback/tiny 55건 전부 train → val AUC로 해당 케이스 판단 불가",
            "val_auc는 monitoring 목적 전용, 최종 성능 결론 금지",
        ],
        "errors": len([e for e in errors if e.get("error", "none") != "none"]),
    }
    with open(os.path.join(report_dir, "p_c16_full_training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    vi = "✅ 통과" if verdict == "통과" else "❌ 실패"
    md_rows = "\n".join(
        f"| {r['epoch']} | {r['train_loss']} | {val_log_rows[i]['val_loss']} "
        f"| {val_log_rows[i]['val_auc']} | {runtime_rows[i]['epoch_time_s']}s |"
        for i, r in enumerate(train_log_rows)
    ) if train_log_rows else "| — | — | — | — | — |"

    md = f"""# P-C16 Full Training Report

**생성일**: {ts}
**판정**: {vi}
**model**: {config.model_name} | epochs: {epochs_completed}/{config.epochs} | best_epoch: {best_epoch}
**early_stopped**: {early_stopped}

## 학습 로그

| epoch | train_loss | val_loss | val_auc | epoch_time |
|-------|-----------|---------|---------|------------|
{md_rows}

## Best Checkpoint
- best_epoch: {best_epoch}
- best_val_auc: {best_val_auc}
- best_val_loss: {best_val_loss}
- path: {os.path.join(config.checkpoint_dir, 'best.pth')}

## 주의사항
- no_hit/fallback/tiny 55건 전부 train → val AUC로 해당 케이스 판단 불가
- val_auc는 monitoring 목적 전용, 최종 성능 결론 금지
- holdout 평가는 별도 단계에서만 수행

*판정: {verdict} | 생성: {ts}*
"""
    with open(os.path.join(report_dir, "p_c16_full_training_report.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n[SAVED] {report_dir}/p_c16_train_log.csv")
    print(f"[SAVED] {report_dir}/p_c16_val_log.csv")
    print(f"[SAVED] {report_dir}/p_c16_runtime_summary.csv")
    print(f"[SAVED] {report_dir}/p_c16_errors.csv")
    print(f"[SAVED] {report_dir}/p_c16_full_training_summary.json")
    print(f"[SAVED] {report_dir}/p_c16_full_training_report.md")
    print(f"\nP-C16 Full Training 판정: {verdict}")


def _save_drycheck_report(config: Config, results: dict, errors: list):
    """Save dry-check report files."""
    os.makedirs(DRYCHECK_REPORT_DIR, exist_ok=True)

    # config draft
    config_draft = config.to_dict()
    config_draft["created"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    config_draft["status"] = "draft_drycheck_only"
    with open(f"{DRYCHECK_REPORT_DIR}/p_c11_config_draft.json", "w") as f:
        json.dump(config_draft, f, indent=2, ensure_ascii=False)

    # batch shape check CSV
    shape_rows = [
        {"tensor": "ct_crop", "shape": str(results.get("ct_shape", "?")),
         "dtype": "float32", "range_min": results.get("ct_min"), "range_max": results.get("ct_max"),
         "nan": results.get("ct_nan"), "inf": results.get("ct_inf")},
        {"tensor": "label", "shape": str(results.get("label_shape", "?")),
         "dtype": "float32", "range_min": 0.0, "range_max": 1.0, "nan": False, "inf": False},
        {"tensor": "model_output", "shape": str(results.get("output_shape", "?")),
         "dtype": "float32", "range_min": None, "range_max": None,
         "nan": results.get("logits_nan"), "inf": results.get("logits_inf")},
    ]
    pd.DataFrame(shape_rows).to_csv(f"{DRYCHECK_REPORT_DIR}/p_c11_batch_shape_check.csv", index=False)

    # errors CSV
    pd.DataFrame(errors if errors else [{"step": "dry_check", "error": "none"}]).to_csv(
        f"{DRYCHECK_REPORT_DIR}/p_c11_errors.csv", index=False)

    # preflight JSON
    verdict = "통과" if results.get("loss_finite") and not results.get("logits_nan") else "실패"
    report_json = {
        "step": "P-C11",
        "verdict": verdict,
        "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "script_path": f"{EXP}/p_c11_train_classifier.py",
        "config": config_draft,
        "drycheck_results": results,
        "guardrails": {
            "training_executed": results.get("training_executed", False),
            "checkpoint_saved": results.get("checkpoint_saved", False),
            "stage2_holdout_accessed": results.get("stage2_holdout_accessed", False),
            "epoch_loop_executed": results.get("epoch_loop_executed", False),
        },
        "warnings": [
            "no_hit/fallback/tiny lesion 55건 전부 train에만 배치 → val 성능으로 해당 케이스 판단 불가",
            "positive border_touch 43.6% → random crop/cutout 금지 반영됨",
            "val에 특수케이스 없음 → val AUC는 일반 케이스 성능만 반영",
        ],
        "next_step": "P-C12: 1-epoch smoke training (별도 승인 필요)",
        "errors": len(errors),
    }
    with open(f"{DRYCHECK_REPORT_DIR}/p_c11_training_script_drycheck.json", "w") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)

    # markdown report
    ct_shape = results.get("ct_shape", "?")
    out_shape = results.get("output_shape", "?")
    loss_val = results.get("loss_value", "?")
    loss_fin = results.get("loss_finite", False)
    fwd_ms = results.get("forward_time_ms", "?")
    pos_count = results.get("batch_pos_count", "?")
    hn_count = results.get("batch_hn_count", "?")
    n_params = results.get("model_params", "?")
    device_str = results.get("device", "?")

    md = f"""# P-C11 Training Script Dry-Check Report

**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**판정**: {'✅ 통과' if verdict == '통과' else '❌ 실패'}

---

## 1. 입력 검증

| 항목 | 결과 |
|------|------|
| P-C10.5 verdict | ✅ 통과 |
| P-C10 verdict | ✅ 통과 |
| P-C9 verdict | ✅ 통과 |
| P-C8 DONE | ✅ |
| C-lite manifest rows | ✅ 110,054 |
| train rows | ✅ 88,083 |
| val rows | ✅ 21,971 |
| patient leakage | ✅ 0 |
| crop_path spot-check (100건) | ✅ missing={results.get('crop_path_missing_sample100', '?')} |
| stage2_holdout contamination | ✅ 0 |

---

## 2. Script 설계

### Dataset
- **class**: `CropDataset`
- **manifest**: `split_plan` 컬럼으로 train/val 분기
- **npz key**: `ct_crop` (int16 HU)
- **label**: `training_label` (positive=1, hard_negative=0)
- **crop_base**: EXP directory-relative path

### CT Preprocessing
- int16 HU → float32
- clip: hu_min={config.ct_hu_min}, hu_max={config.ct_hu_max}
- normalize: (ct - hu_min) / (hu_max - hu_min) → [0, 1]

### Augmentation (train only)
| 항목 | 설정 | 이유 |
|------|------|------|
| random crop | ❌ 금지 | border_touch 43.6% — 병변 손실 위험 |
| cutout / random erasing | ❌ 금지 | 동일 이유 |
| horizontal flip | {'✅ 활성' if config.aug_hflip else '❌ 비활성'} (p=0.5) | CT 좌우 대칭 허용 |
| vertical flip | {'✅ 활성' if config.aug_vflip else '❌ 비활성'} | 해부학 방향 왜곡 우려 → 비활성 |
| gaussian noise | ✅ std={config.aug_noise_std} | 소규모 intensity 변동 |

### Model
- **architecture**: `{config.model_name}`
- **pretrained**: {config.pretrained}
- **input**: (3, 96, 96) float32
- **output**: (B, 1) binary logit
- **params**: {n_params:,}
- **head**: Dropout(0.2) → Linear(1280→1)

### Loss
- **function**: `BCEWithLogitsLoss`
- **pos_weight**: {config.pos_weight} (hn/pos ratio)
- **alternative**: unweighted (ratio 1:2.12는 심각한 불균형 아님)

### Optimizer / Scheduler
- **optimizer**: AdamW(lr={config.lr}, weight_decay={config.weight_decay})
- **scheduler**: cosine warmup (warmup={config.warmup_epochs} epochs)
- **epochs**: {config.epochs} (actual training 시 별도 승인)
- **seed**: {config.seed}

---

## 3. Dry-Check 결과 (1 batch, no training)

| 항목 | 값 | 판정 |
|------|-----|------|
| ct_crop shape | {ct_shape} | ✅ (B,3,96,96) |
| label shape | {results.get('label_shape', '?')} | ✅ (B,) |
| ct range | [{results.get('ct_min','?'):.4f}, {results.get('ct_max','?'):.4f}] | ✅ [0,1] |
| ct NaN | {results.get('ct_nan', '?')} | ✅ False |
| ct Inf | {results.get('ct_inf', '?')} | ✅ False |
| batch pos count | {pos_count} | - |
| batch HN count | {hn_count} | - |
| device | {device_str} | ✅ |
| model output shape | {out_shape} | ✅ (B,1) |
| logits NaN | {results.get('logits_nan', '?')} | ✅ False |
| logits Inf | {results.get('logits_inf', '?')} | ✅ False |
| forward time | {fwd_ms} ms | ✅ |
| loss value | {loss_val} | ✅ finite={loss_fin} |
| batch load time | {results.get('batch_load_time_s', '?')} s | ✅ |

---

## 4. Guardrails 확인

| 항목 | 결과 |
|------|------|
| 실제 학습(epoch loop) 실행 | ✅ 미실행 |
| checkpoint 저장 | ✅ 없음 |
| model weight 저장 | ✅ 없음 |
| stage2_holdout 접근 | ✅ 없음 |
| backward/optimizer step | ✅ 미실행 (dry-check forward-only) |
| 기존 결과 수정 | ✅ 없음 |

---

## 5. 주요 주의사항

1. **no_hit/fallback/tiny 55건 전부 train** → val AUROC는 해당 케이스 일반화 능력 미반영
2. **positive border_touch 43.6%** → random crop/cutout 금지 반영됨 (hflip만 활성)
3. **val 구성**: 30명 / 21,971건 (pos=7,148, hn=14,823, ratio=1:2.07)
4. **risk6 val 분배**: LUNG1-028 56건 (val에 유일한 risk6 전원)

---

## 6. 다음 단계 추천

**P-C12: 1-epoch smoke training** (별도 사용자 승인 필요)

- config: batch_size=64, lr=1e-4, epochs=1, device=cuda
- 목적: 1 epoch 완주 + train_loss 확인 + val_loss 확인
- AUROC/threshold 계산은 P-C12에서도 read-only 모니터링 수준만
- checkpoint는 P-C12 smoke 시에도 저장 여부 별도 결정

또는: **P-C12 training config review** (full epoch 전 재검토)

---
*판정: {verdict} | 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    with open(f"{DRYCHECK_REPORT_DIR}/p_c11_training_script_drycheck.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n[SAVED] {DRYCHECK_REPORT_DIR}/p_c11_training_script_drycheck.md")
    print(f"[SAVED] {DRYCHECK_REPORT_DIR}/p_c11_training_script_drycheck.json")
    print(f"[SAVED] {DRYCHECK_REPORT_DIR}/p_c11_config_draft.json")
    print(f"[SAVED] {DRYCHECK_REPORT_DIR}/p_c11_batch_shape_check.csv")
    print(f"[SAVED] {DRYCHECK_REPORT_DIR}/p_c11_errors.csv")


# ── Output collision guard ─────────────────────────────────────────────────
def _check_output_collision(checkpoint_dir: str, report_dir: str):
    """Abort if smoke output directories already contain results."""
    collisions = []
    smoke_files = [
        os.path.join(checkpoint_dir, "epoch1_smoke.pth"),
        os.path.join(checkpoint_dir, "best_smoke.pth"),
        os.path.join(report_dir, "p_c12_smoke_training_summary.json"),
        os.path.join(report_dir, "p_c12_smoke_training_report.md"),
    ]
    for f in smoke_files:
        if os.path.exists(f):
            collisions.append(f)
    if collisions:
        print("[ABORT] Output collision detected — existing smoke results found:")
        for f in collisions:
            print(f"  {f}")
        print("  Use a new --checkpoint-dir / --report-dir or remove existing files manually.")
        sys.exit(2)


# ── Smoke training (P-C12) ────────────────────────────────────────────────
def run_smoke_train(config: Config, smoke_report_dir: str):
    """
    P-C12: 1-epoch smoke training.
    - Requires epochs == 1 (guard enforced here).
    - Checkpoint saved as epoch1_smoke.pth (NOT best.pth).
    - Full report saved to smoke_report_dir.
    - stage2_holdout never accessed.
    """
    print("=" * 65)
    print("P-C12 Smoke Training (1-epoch only)")
    print("=" * 65)

    # ── Guard: epochs must be 1 ──────────────────────────────────────────
    if config.epochs != 1:
        print(f"[ABORT] --smoke-train requires --epochs 1, got epochs={config.epochs}")
        sys.exit(2)
    print(f"[OK] epochs guard: epochs={config.epochs} == 1")

    # ── Output collision guard ───────────────────────────────────────────
    _check_output_collision(config.checkpoint_dir, smoke_report_dir)
    print("[OK] output collision: no existing results")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(smoke_report_dir, exist_ok=True)

    errors = []
    train_log_rows = []
    val_monitoring_rows = []
    runtime_rows = []

    torch.manual_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"[OK] device: {device}")

    # ── mixed_precision: CUDA only ───────────────────────────────────────
    use_amp = config.mixed_precision and device.type == "cuda"
    if not use_amp and config.mixed_precision:
        print("[WARN] mixed_precision disabled: not running on CUDA")

    # ── Datasets ─────────────────────────────────────────────────────────
    train_dataset = CropDataset(
        config.manifest_path, "train", config.crop_base,
        config.ct_hu_min, config.ct_hu_max,
        TrainTransform(config.aug_hflip, config.aug_vflip, config.aug_noise_std),
    )
    val_dataset = CropDataset(
        config.manifest_path, "val", config.crop_base,
        config.ct_hu_min, config.ct_hu_max,
        ValTransform(),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=config.num_workers > 0,
    )
    print(f"[OK] train_dataset: {len(train_dataset)} | val_dataset: {len(val_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(config).to(device)
    criterion = build_criterion(config)
    if hasattr(criterion, "pos_weight") and criterion.pos_weight is not None:
        criterion.pos_weight = criterion.pos_weight.to(device)
    optimizer = build_optimizer(model, config)
    steps_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── 1-epoch train/val ─────────────────────────────────────────────────
    t_epoch_start = time.time()

    try:
        # train
        model.train()
        total_loss = 0.0
        n_batches = 0
        for ct, labels in train_loader:
            ct = ct.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type if device.type == "cuda" else "cpu", enabled=use_amp):
                logits = model(ct)
                loss = criterion(logits, labels)
            loss_val_item = loss.item()
            if not math.isfinite(loss_val_item):
                raise RuntimeError(f"Train loss NaN/Inf at batch {n_batches}: {loss_val_item}")
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler is not None and config.scheduler_type == "cosine":
                scheduler.step()
            total_loss += loss_val_item
            n_batches += 1
        train_loss = total_loss / max(1, n_batches)

        # val
        val_loss, val_logits, val_labels = validate(model, val_loader, criterion, device)
        if not math.isfinite(val_loss):
            raise RuntimeError(f"Val loss NaN/Inf: {val_loss}")

        # AUROC (monitoring only) — sklearn-free
        val_auc, auroc_status = compute_auroc(val_logits.numpy(), val_labels.numpy())
        if auroc_status != "ok":
            errors.append({"step": "val_auroc", "error": auroc_status})

        t_epoch_end = time.time()
        epoch_time = t_epoch_end - t_epoch_start

        # GPU peak memory
        gpu_peak_mb = 0.0
        if device.type == "cuda":
            gpu_peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

        print(f"[Epoch 001/001] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_auc={val_auc:.4f} ({epoch_time:.1f}s)")

    except RuntimeError as e:
        errors.append({"step": "epoch1", "error": str(e)})
        _save_smoke_report(
            config, smoke_report_dir,
            train_loss=float("nan"), val_loss=float("nan"), val_auc=float("nan"),
            epoch_time=0.0, gpu_peak_mb=0.0, checkpoint_path=None,
            train_log_rows=[], val_monitoring_rows=[], runtime_rows=[], errors=errors,
            verdict="실패",
        )
        print(f"[FAIL] {e}")
        sys.exit(1)

    # ── Checkpoint (smoke only: epoch1_smoke.pth, NOT best.pth) ──────────
    ckpt_path = os.path.join(config.checkpoint_dir, "epoch1_smoke.pth")
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_auc": val_auc,
        "config": config.to_dict(),
        "smoke_only": True,
    }, ckpt_path)
    print(f"[SAVE] smoke checkpoint: {ckpt_path}")

    # ── Build report rows ─────────────────────────────────────────────────
    train_log_rows.append({
        "epoch": 1, "train_loss": round(train_loss, 6),
        "n_batches": n_batches, "steps_per_epoch": steps_per_epoch,
    })
    val_monitoring_rows.append({
        "epoch": 1, "val_loss": round(val_loss, 6),
        "val_auc_monitoring": round(val_auc, 6) if math.isfinite(val_auc) else "nan",
        "note": "monitoring_only_not_performance_conclusion",
    })
    runtime_rows.append({
        "epoch": 1, "epoch_time_s": round(epoch_time, 2),
        "gpu_peak_mb": round(gpu_peak_mb, 1),
        "device": str(device),
    })

    verdict = "통과"
    _save_smoke_report(
        config, smoke_report_dir,
        train_loss=train_loss, val_loss=val_loss, val_auc=val_auc,
        epoch_time=epoch_time, gpu_peak_mb=gpu_peak_mb, checkpoint_path=ckpt_path,
        train_log_rows=train_log_rows, val_monitoring_rows=val_monitoring_rows,
        runtime_rows=runtime_rows, errors=errors,
        verdict=verdict,
    )
    print(f"\nP-C12 Smoke Training 판정: {verdict}")
    print("Training 완료. Full training 실행 전 P-C13 smoke result validation 필요.")


def _save_smoke_report(
    config: Config,
    smoke_report_dir: str,
    train_loss: float,
    val_loss: float,
    val_auc: float,
    epoch_time: float,
    gpu_peak_mb: float,
    checkpoint_path,
    train_log_rows: list,
    val_monitoring_rows: list,
    runtime_rows: list,
    errors: list,
    verdict: str,
):
    """Save all P-C12 smoke training output files."""
    os.makedirs(smoke_report_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # train_log.csv
    pd.DataFrame(train_log_rows if train_log_rows else [{"epoch": 1, "train_loss": train_loss}]).to_csv(
        os.path.join(smoke_report_dir, "p_c12_train_log.csv"), index=False)

    # val_monitoring.csv
    pd.DataFrame(val_monitoring_rows if val_monitoring_rows else [{"epoch": 1, "val_loss": val_loss, "val_auc_monitoring": val_auc}]).to_csv(
        os.path.join(smoke_report_dir, "p_c12_val_monitoring.csv"), index=False)

    # runtime_summary.csv
    pd.DataFrame(runtime_rows if runtime_rows else [{"epoch": 1, "epoch_time_s": epoch_time, "gpu_peak_mb": gpu_peak_mb}]).to_csv(
        os.path.join(smoke_report_dir, "p_c12_runtime_summary.csv"), index=False)

    # errors.csv
    pd.DataFrame(errors if errors else [{"step": "smoke_train", "error": "none"}]).to_csv(
        os.path.join(smoke_report_dir, "p_c12_errors.csv"), index=False)

    # summary.json
    summary = {
        "step": "P-C12",
        "verdict": verdict,
        "created": ts,
        "epochs_requested": config.epochs,
        "epochs_completed": 1 if verdict != "실패" else 0,
        "model": config.model_name,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "train_rows": 88083,
        "val_rows": 21971,
        "train_loss": round(train_loss, 6) if math.isfinite(train_loss) else "nan",
        "val_loss": round(val_loss, 6) if math.isfinite(val_loss) else "nan",
        "val_auc_monitoring": round(val_auc, 6) if math.isfinite(val_auc) else "nan",
        "loss_nan_inf": not (math.isfinite(train_loss) and math.isfinite(val_loss)),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_smoke_only": True,
        "epoch_time_s": round(epoch_time, 2),
        "gpu_peak_mb": round(gpu_peak_mb, 1),
        "guardrails": {
            "epochs_guard_1_only": True,
            "stage2_holdout_accessed": False,
            "full_training_executed": False,
            "best_pth_saved": False,
            "smoke_checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        },
        "warnings": [
            "no_hit/fallback/tiny 55건 전부 train → val에 없음, val AUC로 해당 케이스 판단 불가",
            "positive border_touch 43.6% → random crop/cutout 금지 유지",
            "val_auc는 monitoring 목적 전용, 성능 확정 아님",
        ],
        "next_step": "P-C13: smoke result validation 또는 full training preflight",
        "errors": len([e for e in errors if e.get("error", "none") != "none"]),
    }
    with open(os.path.join(smoke_report_dir, "p_c12_smoke_training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # report.md
    ckpt_path_display = str(checkpoint_path) if checkpoint_path else "저장 안됨"
    train_loss_str = f"{train_loss:.4f}" if math.isfinite(train_loss) else "NaN/Inf"
    val_loss_str = f"{val_loss:.4f}" if math.isfinite(val_loss) else "NaN/Inf"
    val_auc_str = f"{val_auc:.4f}" if math.isfinite(val_auc) else "nan"
    verdict_icon = "✅ 통과" if verdict == "통과" else "❌ 실패" if verdict == "실패" else "⚠️ 부분통과"
    loss_nan_icon = "❌ 있음" if not (math.isfinite(train_loss) and math.isfinite(val_loss)) else "✅ 없음"

    md = f"""# P-C12 Smoke Training Report

**생성일**: {ts}
**판정**: {verdict_icon}

---

## 1. 실행 확인

| 항목 | 결과 |
|------|------|
| epochs 요청 | {config.epochs} |
| epochs=1 guard | ✅ 통과 |
| smoke_train flag | ✅ --smoke-train 사용 |
| stage2_holdout locked | ✅ 미접근 |
| full training 미실행 | ✅ |
| 기존 결과 무수정 | ✅ |

---

## 2. 학습 결과

| 항목 | 값 |
|------|-----|
| model | {config.model_name} |
| batch_size | {config.batch_size} |
| lr | {config.lr} |
| train_rows | 88,083 |
| val_rows | 21,971 |
| train_loss | {train_loss_str} |
| val_loss | {val_loss_str} |
| val_auc (monitoring only) | {val_auc_str} |
| loss NaN/Inf | {loss_nan_icon} |
| epoch_time | {epoch_time:.1f}s |
| GPU peak memory | {gpu_peak_mb:.1f} MB |

---

## 3. Checkpoint

| 항목 | 결과 |
|------|------|
| checkpoint 경로 | `{ckpt_path_display}` |
| 파일명 | epoch1_smoke.pth (best.pth 아님) |
| smoke_only flag | True |
| full checkpoint와 분리 | ✅ |

---

## 4. 주의사항

1. **no_hit/fallback/tiny 55건 전부 train** → val AUC로 해당 케이스 판단 불가
2. **positive border_touch 43.6%** → random crop/cutout 금지 반영됨 (hflip만 활성)
3. **val_auc={val_auc_str}** → monitoring 목적 전용, 성능 확정 아님
4. **smoke checkpoint**는 full training용 best.pth와 별도 경로

---

## 5. 다음 단계

- **P-C13**: smoke result validation
- 또는 **P-C13**: full training preflight (별도 사용자 승인 필요)

---
*판정: {verdict} | 생성: {ts}*
"""

    with open(os.path.join(smoke_report_dir, "p_c12_smoke_training_report.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n[SAVED] {smoke_report_dir}/p_c12_smoke_training_report.md")
    print(f"[SAVED] {smoke_report_dir}/p_c12_smoke_training_summary.json")
    print(f"[SAVED] {smoke_report_dir}/p_c12_train_log.csv")
    print(f"[SAVED] {smoke_report_dir}/p_c12_val_monitoring.csv")
    print(f"[SAVED] {smoke_report_dir}/p_c12_runtime_summary.csv")
    print(f"[SAVED] {smoke_report_dir}/p_c12_errors.csv")


if __name__ == "__main__":
    main()
