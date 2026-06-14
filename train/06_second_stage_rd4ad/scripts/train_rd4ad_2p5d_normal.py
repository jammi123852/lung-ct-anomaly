"""
RD4AD-style 2.5D reconstruction verifier v1
Minimal reconstruction baseline (not full RD4AD teacher-student)

Phase 5.26 - train script for normal 2.5D crop dataset
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
FORBIDDEN_PATH_PATTERNS = ["stage2_holdout", "v2"]
HARD_NEGATIVE_DIR_MARKER = "rd4ad_train_2p5d_mw_fixed96_thr001"
DEFAULT_MODEL_TAG = "rd4ad_2p5d_normal_mw_fixed96_v1"

DEFAULT_CONFIG = (
    "outputs/second-stage-lesion-refiner-v1/models/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/configs/"
    "train_config_rd4ad_2p5d_normal_mw_fixed96_v1.yaml"
)


# ──────────────────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="RD4AD-style 2.5D reconstruction verifier v1 - minimal reconstruction baseline"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "cuda_if_available"],
        default="cuda_if_available",
    )
    parser.add_argument("--no-runtime-append", action="store_true")
    parser.add_argument(
        "--allow-draft-config",
        action="store_true",
        default=False,
        help="[DEBUG ONLY] Allow execution with draft config. DO NOT use for real training.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# Config Loader
# ──────────────────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("[ERROR] PyYAML is not installed. Cannot load config.")
        print("        Do NOT run pip install. Please check your Python environment.")
        sys.exit(1)

    p = Path(config_path)
    if not p.exists():
        print(f"[ERROR] Config file not found: {p}")
        sys.exit(1)

    with open(p, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"[INFO] Config loaded: {p}")
    return cfg


# ──────────────────────────────────────────────────────────
# Config Validation
# ──────────────────────────────────────────────────────────
def validate_config(cfg: dict, allow_draft_config: bool = False, allow_draft_for_preflight: bool = False) -> bool:
    errors = []

    status = cfg.get("experiment", {}).get("implementation_status", "")
    if status == "draft_config_not_executable_yet":
        if allow_draft_for_preflight:
            print("[INFO] Draft config allowed for read-only preflight only. Training/dry-run still blocked.")
        elif allow_draft_config:
            print("[WARNING] --allow-draft-config active. DRAFT config. DO NOT use for real training.")
        else:
            errors.append(
                "implementation_status='draft_config_not_executable_yet'. "
                "Training and dry-run are blocked. "
                "Update to 'executable_config' after user review."
            )

    data = cfg.get("data", {})

    manifest_path = data.get("crop_manifest", "")
    if not Path(manifest_path).exists():
        errors.append(f"crop_manifest not found: {manifest_path}")

    summary_path = data.get("crop_summary", "")
    if not Path(summary_path).exists():
        errors.append(f"crop_summary not found: {summary_path}")

    count_checks = [
        ("expected_total_crops", 18100),
        ("expected_train_crops", 14500),
        ("expected_val_crops", 1800),
        ("expected_test_crops", 1800),
    ]
    for key, expected in count_checks:
        val = data.get(key)
        if val != expected:
            errors.append(f"{key}: expected {expected}, got {val}")

    if data.get("expected_input_shape") != [6, 96, 96]:
        errors.append(f"expected_input_shape must be [6,96,96], got {data.get('expected_input_shape')}")

    if data.get("expected_dtype") != "float32":
        errors.append(f"expected_dtype must be float32, got {data.get('expected_dtype')}")

    for key in ("use_test_for_training", "use_test_for_early_stopping", "use_test_for_threshold_tuning"):
        if data.get(key) is not False:
            errors.append(f"data.{key} must be false")

    excluded = cfg.get("excluded_data", {})
    for key in ("use_hard_negative_for_train", "use_stage2_holdout", "use_v2_data"):
        if excluded.get(key) is not False:
            errors.append(f"excluded_data.{key} must be false")

    # forbidden section note
    forbidden = cfg.get("forbidden", {})
    if forbidden:
        note = forbidden.get("_note", "")
        active = [k for k, v in forbidden.items() if k != "_note" and v is True]
        print(f"[INFO] forbidden._note: {note}")
        print(f"[INFO] forbidden flags (true=금지): {active}")

    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        return False

    print("[INFO] Config validation passed.")
    return True


# ──────────────────────────────────────────────────────────
# Safety Guards
# ──────────────────────────────────────────────────────────
def check_path_safety(path: str, label: str):
    """v2 / stage2_holdout 경로 차단"""
    for pattern in FORBIDDEN_PATH_PATTERNS:
        if pattern in str(path):
            print(f"[ERROR] {label} contains forbidden pattern '{pattern}': {path}")
            sys.exit(1)


def check_hard_negative_not_in_manifest(manifest_path: str):
    if HARD_NEGATIVE_DIR_MARKER in str(manifest_path):
        print(f"[ERROR] Hard negative crop dir detected in manifest path: {manifest_path}")
        print("[ERROR] Hard negative 7,700 crops must NOT be used for training.")
        sys.exit(1)


def check_split_not_test(split: str, context: str):
    if split == "test":
        print(f"[ERROR] test split must not be used in {context}.")
        sys.exit(1)


# ──────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────
class NormalCropDataset(Dataset):
    """
    Minimal reconstruction baseline dataset.
    Only train or val split is allowed. test split is blocked.
    reconstruction target == input (self-supervised).
    """

    def __init__(self, manifest_df: pd.DataFrame, split: str, image_key: str = "image"):
        check_split_not_test(split, "NormalCropDataset")
        self.split = split
        self.image_key = image_key

        df = manifest_df[manifest_df["normal_split"] == split].reset_index(drop=True)
        if len(df) == 0:
            print(f"[ERROR] No crops found for split='{split}'")
            sys.exit(1)

        for i, row in df.iterrows():
            cp = str(row.get("crop_path", ""))
            check_path_safety(cp, f"crop_path row={i}")
            check_hard_negative_not_in_manifest(cp)

        self.df = df
        print(f"[INFO] NormalCropDataset split={split}: {len(self.df)} crops")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        crop_path = str(row["crop_path"])

        if not Path(crop_path).exists():
            raise FileNotFoundError(f"crop_path not found: {crop_path}")

        data = np.load(crop_path)
        if self.image_key not in data:
            raise KeyError(f"image key '{self.image_key}' not found in {crop_path}")

        image = data[self.image_key].astype(np.float32)

        if image.shape != (6, 96, 96):
            raise ValueError(f"Unexpected shape {image.shape} at {crop_path}, expected (6,96,96)")

        if not np.isfinite(image).all():
            raise ValueError(f"NaN or Inf detected in {crop_path}")

        if image.min() < -1e-4 or image.max() > 1.0 + 1e-4:
            raise ValueError(
                f"Intensity out of [0,1] in {crop_path}: "
                f"min={image.min():.4f}, max={image.max():.4f}"
            )

        tensor = torch.tensor(image, dtype=torch.float32)
        # reconstruction target == input
        return tensor, tensor


# ──────────────────────────────────────────────────────────
# DataLoaders
# ──────────────────────────────────────────────────────────
def build_dataloaders(manifest_df: pd.DataFrame, cfg: dict, device: torch.device):
    dl_cfg = cfg.get("dataloader", {})
    batch_size = dl_cfg.get("batch_size", 32)
    num_workers = dl_cfg.get("num_workers", 2)
    pin_memory = dl_cfg.get("pin_memory", True) and (device.type == "cuda")
    image_key = cfg.get("data", {}).get("image_key", "image")

    # patient overlap guard
    train_ids = set(manifest_df[manifest_df["normal_split"] == "train"]["patient_id"].unique())
    val_ids = set(manifest_df[manifest_df["normal_split"] == "val"]["patient_id"].unique())
    test_ids = set(manifest_df[manifest_df["normal_split"] == "test"]["patient_id"].unique())

    tv_overlap = train_ids & val_ids
    if tv_overlap:
        print(f"[ERROR] train/val patient_id overlap detected: {tv_overlap}")
        sys.exit(1)
    tt_overlap = train_ids & test_ids
    if tt_overlap:
        print(f"[ERROR] train/test patient_id overlap detected: {tt_overlap}")
        sys.exit(1)
    print("[INFO] Patient overlap check: OK (train/val/test disjoint)")

    train_ds = NormalCropDataset(manifest_df, split="train", image_key=image_key)
    val_ds = NormalCropDataset(manifest_df, split="val", image_key=image_key)
    # test_loader는 생성하지 않음

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(f"[INFO] train_loader: {len(train_ds)} crops / {len(train_loader)} batches")
    print(f"[INFO] val_loader:   {len(val_ds)} crops / {len(val_loader)} batches")
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────
class ConvAutoencoder2p5D(nn.Module):
    """
    Minimal reconstruction baseline for 2.5D normal crop verifier.
    NOT a full RD4AD teacher-student implementation.

    input_channels=6  (lung z-1/z/z+1, mediastinal z-1/z/z+1)
    output_channels=6 (reconstruction target == input)
    Encoder: 96→48→24→12 (3× MaxPool2d)
    Decoder: 12→24→48→96 (3× ConvTranspose2d) + Sigmoid
    """

    def __init__(self, input_channels: int = 6, base_channels: int = 32):
        super().__init__()

        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 48×48

            nn.Conv2d(c, c * 2, 3, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 24×24

            nn.Conv2d(c * 2, c * 4, 3, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 12×12

            # bottleneck
            nn.Conv2d(c * 4, c * 8, 3, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2),   # 24×24
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2),   # 48×48
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(c * 2, c, 2, stride=2),       # 96×96
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),

            nn.Conv2d(c, input_channels, 1),
            nn.Sigmoid(),                                      # output in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ──────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────
def get_reconstruction_loss(cfg: dict) -> nn.Module:
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("reconstruction_loss", "l1").lower()

    if loss_type == "l1":
        criterion = nn.L1Loss()
    elif loss_type == "mse":
        criterion = nn.MSELoss()
    else:
        print(f"[WARNING] Unknown reconstruction_loss='{loss_type}', falling back to L1.")
        criterion = nn.L1Loss()

    if loss_cfg.get("feature_loss_enabled", False):
        print(
            "[WARNING] feature_loss_enabled=true in config, but v1 train script implements "
            "pixel reconstruction loss only. Feature loss is NOT applied in this version. "
            "Set feature_loss_enabled=false in the executable config before real training."
        )

    print(f"[INFO] Reconstruction loss: {loss_type}")
    return criterion


# ──────────────────────────────────────────────────────────
# Seed & Device
# ──────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda_if_available":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            print("[ERROR] --device cuda requested but CUDA is not available.")
            sys.exit(1)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Device: {device}")
    return device


# ──────────────────────────────────────────────────────────
# Training & Validation
# ──────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_x, batch_target in loader:
        batch_x = batch_x.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_target)

        if not torch.isfinite(loss):
            print("[ERROR] NaN/Inf loss during training. Stopping.")
            sys.exit(1)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_x, batch_target in loader:
            batch_x = batch_x.to(device)
            batch_target = batch_target.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_target)

            if not torch.isfinite(loss):
                print("[ERROR] NaN/Inf loss during validation. Stopping.")
                sys.exit(1)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ──────────────────────────────────────────────────────────
# Preflight
# ──────────────────────────────────────────────────────────
def run_preflight(cfg: dict, manifest_df: pd.DataFrame, output_root: Path, args) -> bool:
    print("\n[PREFLIGHT] Starting preflight checks...")
    ok = True
    data = cfg.get("data", {})
    image_key = data.get("image_key", "image")

    # total crop count
    total_count = len(manifest_df)
    expected_total = data.get("expected_total_crops", 18100)
    if total_count != expected_total:
        print(f"[PREFLIGHT ERROR] total crops: expected {expected_total}, got {total_count}")
        ok = False
    else:
        print(f"[PREFLIGHT OK] total crops: {total_count}")

    # split counts
    for split, key in [
        ("train", "expected_train_crops"),
        ("val", "expected_val_crops"),
        ("test", "expected_test_crops"),
    ]:
        cnt = (manifest_df["normal_split"] == split).sum()
        expected = data.get(key, 0)
        if cnt != expected:
            print(f"[PREFLIGHT ERROR] {split} crops: expected {expected}, got {cnt}")
            ok = False
        else:
            print(f"[PREFLIGHT OK] {split} crops: {cnt}")

    # patient overlap
    train_ids = set(manifest_df[manifest_df["normal_split"] == "train"]["patient_id"].unique())
    val_ids = set(manifest_df[manifest_df["normal_split"] == "val"]["patient_id"].unique())
    test_ids = set(manifest_df[manifest_df["normal_split"] == "test"]["patient_id"].unique())
    for a_name, b_name, a_ids, b_ids in [
        ("train", "val", train_ids, val_ids),
        ("train", "test", train_ids, test_ids),
        ("val", "test", val_ids, test_ids),
    ]:
        overlap = a_ids & b_ids
        if overlap:
            print(f"[PREFLIGHT ERROR] {a_name}/{b_name} patient overlap: {overlap}")
            ok = False
        else:
            print(f"[PREFLIGHT OK] {a_name}/{b_name} patient overlap: 0")

    # sample crop_path existence (first 5)
    sample_df = manifest_df.head(5)
    for i, row in sample_df.iterrows():
        cp = Path(str(row.get("crop_path", "")))
        if not cp.exists():
            print(f"[PREFLIGHT ERROR] crop_path not found: {cp}")
            ok = False

    # sample npz shape/dtype/range/NaN (3 from train)
    train_df = manifest_df[manifest_df["normal_split"] == "train"]
    sample_train = train_df.sample(min(3, len(train_df)), random_state=42)
    for i, row in sample_train.iterrows():
        cp = Path(str(row.get("crop_path", "")))
        if not cp.exists():
            print(f"[PREFLIGHT ERROR] sample npz not found: {cp}")
            ok = False
            continue
        try:
            arr = np.load(cp)[image_key].astype(np.float32)
            if arr.shape != (6, 96, 96):
                print(f"[PREFLIGHT ERROR] shape {arr.shape} != (6,96,96): {cp}")
                ok = False
            elif not np.isfinite(arr).all():
                print(f"[PREFLIGHT ERROR] NaN/Inf in: {cp}")
                ok = False
            elif arr.min() < -1e-4 or arr.max() > 1.0 + 1e-4:
                print(f"[PREFLIGHT ERROR] intensity out of [0,1]: min={arr.min():.4f}, max={arr.max():.4f}: {cp}")
                ok = False
            else:
                print(f"[PREFLIGHT OK] sample npz: {cp.name}")
        except Exception as e:
            print(f"[PREFLIGHT ERROR] failed to load {cp}: {e}")
            ok = False

    # model instantiation
    try:
        _m = ConvAutoencoder2p5D(input_channels=6)
        print("[PREFLIGHT OK] model instantiation: OK")
        del _m
    except Exception as e:
        print(f"[PREFLIGHT ERROR] model instantiation failed: {e}")
        ok = False

    # output dir 충돌
    ckpt_dir = output_root / "checkpoints"
    if ckpt_dir.exists() and any(ckpt_dir.iterdir()) and not args.force:
        print(f"[PREFLIGHT WARNING] checkpoint dir has existing files: {ckpt_dir}")
        print("  Use --force to allow overwrite.")
    else:
        print(f"[PREFLIGHT OK] output dir: {output_root}")

    # safety checks
    print("[PREFLIGHT OK] hard_negative: excluded from train")
    print("[PREFLIGHT OK] stage2_holdout: not used")
    print("[PREFLIGHT OK] v2: not accessed")

    if ok:
        print("\n[PREFLIGHT] All checks passed.")
    else:
        print("\n[PREFLIGHT] Some checks FAILED.")

    return ok


# ──────────────────────────────────────────────────────────
# Dry Run
# ──────────────────────────────────────────────────────────
def run_dry_run(cfg: dict, manifest_df: pd.DataFrame, output_root: Path, device: torch.device, args):
    print("\n[DRY-RUN] Starting...")

    ok = run_preflight(cfg, manifest_df, output_root, args)
    if not ok:
        print("[DRY-RUN] Preflight failed. Aborting.")
        sys.exit(1)

    train_loader, val_loader = build_dataloaders(manifest_df, cfg, device)
    model = ConvAutoencoder2p5D(input_channels=6).to(device)
    criterion = get_reconstruction_loss(cfg)

    # train 1 batch
    train_x, train_t = next(iter(train_loader))
    train_x, train_t = train_x.to(device), train_t.to(device)
    train_out = model(train_x)
    train_loss = criterion(train_out, train_t)
    print(
        f"[DRY-RUN] train batch: input={tuple(train_x.shape)}, "
        f"output={tuple(train_out.shape)}, loss={train_loss.item():.6f}"
    )

    # val 1 batch
    val_x, val_t = next(iter(val_loader))
    val_x, val_t = val_x.to(device), val_t.to(device)
    with torch.no_grad():
        val_out = model(val_x)
        val_loss = criterion(val_out, val_t)
    print(
        f"[DRY-RUN] val batch:   input={tuple(val_x.shape)}, "
        f"output={tuple(val_out.shape)}, loss={val_loss.item():.6f}"
    )

    print("[DRY-RUN] backward/optimizer step: SKIPPED")
    print("[DRY-RUN] checkpoint/log: NOT generated")
    print("[DRY-RUN] Done.")


# ──────────────────────────────────────────────────────────
# Main Training
# ──────────────────────────────────────────────────────────
def run_training(cfg: dict, manifest_df: pd.DataFrame, output_root: Path, device: torch.device, args, model_tag: str = DEFAULT_MODEL_TAG):
    print("\n[TRAIN] Starting training...")

    checkpoint_dir = output_root / "checkpoints"
    log_dir = output_root / "logs"
    report_dir = output_root / "reports"
    config_save_dir = output_root / "configs"
    for d in [checkpoint_dir, log_dir, report_dir, config_save_dir]:
        d.mkdir(parents=True, exist_ok=True)

    best_ckpt = checkpoint_dir / "best_val_loss.pt"
    last_ckpt = checkpoint_dir / "last.pt"
    if best_ckpt.exists() and not args.force and not args.resume:
        print(f"[ERROR] {best_ckpt} already exists. Use --force or --resume.")
        sys.exit(1)

    # resolved config 저장
    import yaml
    resolved_config_path = config_save_dir / f"resolved_train_config_{model_tag}.yaml"
    with open(resolved_config_path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    print(f"[TRAIN] Resolved config saved: {resolved_config_path}")

    train_loader, val_loader = build_dataloaders(manifest_df, cfg, device)
    model = ConvAutoencoder2p5D(input_channels=6).to(device)
    criterion = get_reconstruction_loss(cfg)

    opt_cfg = cfg.get("optimizer", {})
    train_cfg = cfg.get("training", {})

    lr = opt_cfg.get("learning_rate", 1e-4)
    wd = opt_cfg.get("weight_decay", 1e-5)
    max_epochs = train_cfg.get("max_epochs", 100)
    patience = train_cfg.get("early_stopping_patience", 12)
    early_stopping_enabled = train_cfg.get("early_stopping_enabled", True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler_name = opt_cfg.get("scheduler", "none")
    scheduler = None
    if scheduler_name == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    elif scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    start_epoch = 0
    best_val_loss = float("inf")
    no_improve_count = 0

    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        no_improve_count = ckpt.get("no_improve_count", 0)
        print(f"[TRAIN] Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    log_path = log_dir / f"train_log_{model_tag}.csv"
    log_exists = log_path.exists() and args.resume
    log_file = open(log_path, "a" if log_exists else "w", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(["epoch", "train_loss", "val_loss", "lr", "timestamp"])

    final_epoch = start_epoch
    try:
        for epoch in range(start_epoch, max_epochs):
            final_epoch = epoch
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = validate_one_epoch(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]
            ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

            print(
                f"[TRAIN] Epoch {epoch + 1}/{max_epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.2e}"
            )
            log_writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.2e}", ts])
            log_file.flush()

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "no_improve_count": no_improve_count,
                        "config": cfg,
                    },
                    best_ckpt,
                )
                print(f"[TRAIN] Best checkpoint saved (epoch={epoch + 1}, val_loss={best_val_loss:.6f})")
            else:
                no_improve_count += 1

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "no_improve_count": no_improve_count,
                    "config": cfg,
                },
                last_ckpt,
            )

            if early_stopping_enabled and no_improve_count >= patience:
                print(f"[TRAIN] Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs).")
                break
    finally:
        log_file.close()

    summary = {
        "script": "train_rd4ad_2p5d_normal.py",
        "model": "ConvAutoencoder2p5D",
        "note": "minimal reconstruction baseline, not full RD4AD teacher-student",
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "best_val_loss": best_val_loss,
        "total_epochs_run": final_epoch + 1,
        "output_root": str(output_root),
    }
    summary_path = report_dir / f"train_summary_{model_tag}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[TRAIN] Summary saved: {summary_path}")
    print("[TRAIN] Done.")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(42)

    cfg = load_config(args.config)

    status = cfg.get("experiment", {}).get("implementation_status", "")
    is_draft = status == "draft_config_not_executable_yet"

    # implementation_status guard
    if is_draft and not args.allow_draft_config:
        if args.preflight_only:
            print(f"[INFO] Config status: {status}")
            print("[INFO] --preflight-only: draft config detected. read-only preflight validation only. Training is NOT allowed.")
        else:
            print(f"[ERROR] Config implementation_status='{status}'.")
            print("[ERROR] Training and dry-run are blocked for draft configs.")
            print("[ERROR] Change status to 'executable_config' after user review.")
            sys.exit(1)
    elif is_draft and args.allow_draft_config:
        print("[WARNING] --allow-draft-config active. DRAFT config. DO NOT use for real training.")

    if not validate_config(cfg, allow_draft_config=args.allow_draft_config, allow_draft_for_preflight=args.preflight_only):
        print("[ERROR] Config validation failed.")
        sys.exit(1)

    manifest_path = cfg.get("data", {}).get("crop_manifest", "")
    check_path_safety(manifest_path, "crop_manifest")
    check_hard_negative_not_in_manifest(manifest_path)

    manifest_df = pd.read_csv(manifest_path)

    output_root = Path(
        cfg.get("output", {}).get(
            "output_root",
            "outputs/second-stage-lesion-refiner-v1/models/rd4ad_2p5d_normal_mw_fixed96_v1/",
        )
    )
    model_tag = (
        cfg.get("experiment", {}).get("name")
        or cfg.get("run_name")
        or DEFAULT_MODEL_TAG
    )
    device = resolve_device(args.device)

    if args.preflight_only:
        run_preflight(cfg, manifest_df, output_root, args)
        sys.exit(0)

    if is_draft and not args.allow_draft_config:
        print("[ERROR] Cannot run dry-run or training with draft config.")
        sys.exit(1)

    if args.dry_run:
        run_dry_run(cfg, manifest_df, output_root, device, args)
        sys.exit(0)

    # 실제 학습 실행: executable_config 상태에서만 허용
    if status != "executable_config":
        print(f"[ERROR] Real training requires implementation_status='executable_config', got '{status}'.")
        print("[ERROR] Update the config after user review and approval.")
        sys.exit(1)

    run_training(cfg, manifest_df, output_root, device, args, model_tag=model_tag)


if __name__ == "__main__":
    main()
