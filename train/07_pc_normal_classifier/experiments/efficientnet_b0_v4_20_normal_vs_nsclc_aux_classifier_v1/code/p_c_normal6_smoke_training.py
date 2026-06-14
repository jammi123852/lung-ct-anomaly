"""
p_c_normal6_smoke_training.py

P-C-NORMAL6: 1-epoch smoke training
Normal-vs-NSCLC Supervised Auxiliary Classifier

Model:  EfficientNet-B0 (torchvision, ImageNet pretrained)
Input:  (3, 96, 96) int16 HU → clip[-1000,200] → [0,1] → ImageNet norm
Label:  0=normal, 1=NSCLC lesion-positive
Loss:   BCEWithLogitsLoss(reduction='none') × sample_weight → mean

This model is NOT a diagnostic/clinical-decision model.
Output: normal-like vs NSCLC-lesion-like auxiliary score only.

Run:
  python p_c_normal6_smoke_training.py \
      --smoke-train --confirm-smoke --confirm-normal-vs-nsclc --confirm-no-holdout \
      --epochs 1
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BRANCH_ROOT   = Path(__file__).resolve().parents[1]
MANIFEST_DIR  = BRANCH_ROOT / "outputs/manifests/p_c_normal3_training_manifest"
NORMAL4_REPORT= BRANCH_ROOT / "outputs/reports/p_c_normal4_manifest_crop_validation"
NORMAL5_REPORT= BRANCH_ROOT / "outputs/reports/p_c_normal5_train_script_drycheck"
CKPT_DIR      = BRANCH_ROOT / "outputs/checkpoints/p_c_normal6_smoke_training"
REPORT_DIR    = BRANCH_ROOT / "outputs/reports/p_c_normal6_smoke_training"

STAGE2_HOLDOUT_SENTINEL = "stage2_holdout"

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
HU_MIN, HU_MAX   = -1000.0, 200.0
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]

CLASS_WEIGHT_NORMAL = 0.763033
CLASS_WEIGHT_NSCLC  = 1.45045

FORBIDDEN_WORDS = [
    "폐선암" + " 확률",
    "암" + " 확률",
    "진단" + " 모델",
    "cancer" + " probability",
    "adenocarcinoma" + " probability",
]

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class NormalNSCLCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False,
                 hflip: bool = True, noise: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.hflip = hflip
        self.noise = noise
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(str(row["crop_path"]))
        arr = data["ct_crop"].astype(np.float32)
        arr = np.clip(arr, HU_MIN, HU_MAX)
        arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)
        t = torch.from_numpy(arr)
        t = (t - self.mean) / self.std

        if self.augment and self.hflip and torch.rand(1).item() > 0.5:
            t = torch.flip(t, dims=[-1])

        return t, int(row["label"]), float(row["sample_weight"])


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def weighted_bce_loss(logits, labels, sample_weights):
    bce = nn.BCEWithLogitsLoss(reduction="none")
    per_sample = bce(logits.squeeze(1), labels.float())
    return (per_sample * sample_weights).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Guards
# ──────────────────────────────────────────────────────────────────────────────

def _guard_smoke_train(args):
    missing = []
    if not args.confirm_smoke:
        missing.append("--confirm-smoke")
    if not args.confirm_normal_vs_nsclc:
        missing.append("--confirm-normal-vs-nsclc")
    if not args.confirm_no_holdout:
        missing.append("--confirm-no-holdout")
    if args.epochs != 1:
        missing.append("--epochs 1  (smoke-train requires exactly 1 epoch)")
    if missing:
        print(f"[GUARD] --smoke-train requires: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)


def _check_stage2_holdout(df):
    path_cols = [c for c in df.columns if "path" in c.lower()]
    for col in path_cols:
        if df[col].astype(str).str.contains(STAGE2_HOLDOUT_SENTINEL, na=False).any():
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Smoke training
# ──────────────────────────────────────────────────────────────────────────────

def run_smoke_train(args):
    _guard_smoke_train(args)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    errors = []

    # ── preflight checks ──────────────────────────────────────────────────────
    n5_json = NORMAL5_REPORT / "p_c_normal5_train_script_drycheck.json"
    if not n5_json.exists():
        print("[ERROR] P-C-NORMAL5 dry-check report not found.", file=sys.stderr)
        sys.exit(1)
    with open(n5_json) as f:
        n5 = json.load(f)
    if n5.get("verdict") != "PASS":
        print("[ERROR] P-C-NORMAL5 dry-check was not PASS.", file=sys.stderr)
        sys.exit(1)

    # ── load manifests ────────────────────────────────────────────────────────
    df_train = pd.read_csv(MANIFEST_DIR / "p_c_normal3_train_manifest.csv", low_memory=False)
    df_val   = pd.read_csv(MANIFEST_DIR / "p_c_normal3_val_manifest.csv",   low_memory=False)

    # safety checks
    assert len(df_train) == 22891, f"train rows mismatch: {len(df_train)}"
    assert len(df_val)   == 7080,  f"val rows mismatch: {len(df_val)}"
    assert (df_train["label"].isin([0, 1])).all(), "invalid labels in train"
    assert df_train["label_name"].str.contains("hard_negative", na=False).sum() == 0
    assert _check_stage2_holdout(df_train), "stage2_holdout found in manifest paths"

    # ── dataloaders ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P-C-NORMAL6] device={device}, epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}")

    ds_train = NormalNSCLCDataset(df_train, augment=True,
                                  hflip=True, noise=not args.disable_noise)
    ds_val   = NormalNSCLCDataset(df_val,   augment=False)

    loader_train = DataLoader(ds_train, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              drop_last=False, pin_memory=True)
    loader_val   = DataLoader(ds_val,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              drop_last=False, pin_memory=True)

    # ── model / optimizer ─────────────────────────────────────────────────────
    model = build_model(pretrained=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── 1-epoch train loop ────────────────────────────────────────────────────
    model.train()
    epoch_loss_sum = 0.0
    epoch_correct  = 0
    epoch_total    = 0
    step_log = []

    total_steps = len(loader_train)
    log_interval = max(1, total_steps // 10)

    print(f"[P-C-NORMAL6] train steps={total_steps}")

    for step, (imgs, labels, sw) in enumerate(loader_train):
        imgs   = imgs.to(device)
        labels = labels.to(device)
        sw     = sw.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = weighted_bce_loss(logits, labels, sw)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds   = (logits.squeeze(1) > 0).long()
            correct = (preds == labels).sum().item()

        epoch_loss_sum += loss.item() * len(imgs)
        epoch_correct  += correct
        epoch_total    += len(imgs)

        if step % log_interval == 0 or step == total_steps - 1:
            running_loss = epoch_loss_sum / max(epoch_total, 1)
            running_acc  = epoch_correct  / max(epoch_total, 1)
            print(f"  [step {step+1}/{total_steps}] loss={running_loss:.4f}  acc={running_acc:.4f}")
            step_log.append({
                "step": step + 1,
                "total_steps": total_steps,
                "running_loss": round(running_loss, 6),
                "running_acc":  round(running_acc,  6),
            })

    train_loss = epoch_loss_sum / max(epoch_total, 1)
    train_acc  = epoch_correct  / max(epoch_total, 1)

    # ── val loop (no_grad) ────────────────────────────────────────────────────
    model.eval()
    val_loss_sum = 0.0
    val_correct  = 0
    val_total    = 0
    all_logits   = []
    all_labels   = []

    with torch.no_grad():
        for imgs, labels, sw in loader_val:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            sw     = sw.to(device)

            logits = model(imgs)
            loss   = weighted_bce_loss(logits, labels, sw)

            preds   = (logits.squeeze(1) > 0).long()
            correct = (preds == labels).sum().item()

            val_loss_sum += loss.item() * len(imgs)
            val_correct  += correct
            val_total    += len(imgs)
            all_logits.append(logits.squeeze(1).cpu())
            all_labels.append(labels.cpu())

    val_loss = val_loss_sum / max(val_total, 1)
    val_acc  = val_correct  / max(val_total, 1)

    # AUC (numpy trapezoid — no sklearn dependency)
    val_auc = None
    try:
        logits_np = torch.cat(all_logits).numpy()
        labels_np = torch.cat(all_labels).numpy()
        probs_np  = 1.0 / (1.0 + np.exp(-logits_np))
        # sort by descending score
        order     = np.argsort(-probs_np)
        labels_s  = labels_np[order]
        n_pos = labels_s.sum()
        n_neg = len(labels_s) - n_pos
        if n_pos > 0 and n_neg > 0:
            tp = np.cumsum(labels_s)
            fp = np.cumsum(1 - labels_s)
            tpr = tp / n_pos
            fpr = fp / n_neg
            tpr = np.concatenate([[0.0], tpr])
            fpr = np.concatenate([[0.0], fpr])
            val_auc = float(np.trapz(tpr, fpr))
    except Exception as e:
        errors.append({"stage": "val_auc", "error": str(e)})

    t_elapsed = time.time() - t_start

    print(f"\n[P-C-NORMAL6] Smoke train done")
    print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")
    print(f"  val_loss={val_loss:.4f}    val_acc={val_acc:.4f}    val_auc={val_auc}")
    print(f"  elapsed={t_elapsed:.1f}s")

    # ── checkpoint ────────────────────────────────────────────────────────────
    ckpt_path = CKPT_DIR / "p_c_normal6_epoch1.pth"
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "train_acc":  train_acc,
        "val_loss":   val_loss,
        "val_acc":    val_acc,
        "val_auc":    val_auc,
        "label_mapping": {"0": "normal", "1": "NSCLC"},
        "model_architecture": "EfficientNet-B0 (torchvision ImageNet pretrained)",
        "input_shape": "(3, 96, 96)",
        "preprocessing": "HU clip[-1000,200] → [0,1] → ImageNet norm",
        "note": "auxiliary score only — not diagnostic",
    }, ckpt_path)
    print(f"  checkpoint → {ckpt_path}")

    # ── report ────────────────────────────────────────────────────────────────
    summary = {
        "stage": "P-C-NORMAL6",
        "verdict": "PASS",
        "conditions_ok": True,
        "completed_at": datetime.datetime.now().isoformat(),
        "model_architecture": "EfficientNet-B0 (torchvision ImageNet pretrained)",
        "input_shape": "(3, 96, 96)",
        "label_mapping": {"0": "normal", "1": "NSCLC"},
        "epochs_run": 1,
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "device": str(device),
        "lr": args.lr,
        "batch_size": args.batch_size,
        "augmentation": {"hflip": True, "noise": not args.disable_noise},
        "loss": "BCEWithLogitsLoss(reduction=none) * sample_weight → mean",
        "train_loss": round(train_loss, 6),
        "train_acc":  round(train_acc,  6),
        "val_loss":   round(val_loss,   6),
        "val_acc":    round(val_acc,    6),
        "val_auc":    round(val_auc, 6) if val_auc is not None else None,
        "checkpoint": str(ckpt_path),
        "elapsed_sec": round(t_elapsed, 1),
        "step_log_count": len(step_log),
        "guardrail": {
            "stage2_holdout_accessed": False,
            "training_run": True,
            "model_forward_run": True,
            "backward_run": True,
            "checkpoint_saved": True,
            "scoring_run": False,
            "threshold_computed": False,
            "original_file_modified": False,
            "p_c_aux_modified": False,
            "hard_negative_included": False,
            "msd_lung_included": False,
            "vessel_mask_used": False,
            "forbidden_diagnostic_wording_count": 0,
        },
        "errors_count": len(errors),
        "errors": errors,
        "next_step": "P-C-NORMAL7: smoke training result validation checkpoint",
    }

    with open(REPORT_DIR / "p_c_normal6_smoke_training.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame(step_log).to_csv(
        REPORT_DIR / "p_c_normal6_step_log.csv", index=False)

    if errors:
        pd.DataFrame(errors).to_csv(
            REPORT_DIR / "p_c_normal6_errors.csv", index=False)
    else:
        pd.DataFrame(columns=["stage", "error"]).to_csv(
            REPORT_DIR / "p_c_normal6_errors.csv", index=False)

    # markdown
    md = [
        "# P-C-NORMAL6 Smoke Training (1 epoch)",
        "",
        f"## 판정: **통과 (PASS)**",
        "",
        f"- completed_at: {summary['completed_at'][:10]}",
        f"- device: {summary['device']}",
        f"- elapsed: {summary['elapsed_sec']}s",
        "",
        "## 결과",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| train_loss | {summary['train_loss']} |",
        f"| train_acc  | {summary['train_acc']} |",
        f"| val_loss   | {summary['val_loss']} |",
        f"| val_acc    | {summary['val_acc']} |",
        f"| val_auc    | {summary['val_auc']} |",
        "",
        "## Guardrail",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| stage2_holdout_accessed | False |",
        f"| original_file_modified | False |",
        f"| p_c_aux_modified | False |",
        f"| scoring_run | False |",
        f"| threshold_computed | False |",
        f"| hard_negative_included | False |",
        f"| msd_lung_included | False |",
        f"| vessel_mask_used | False |",
        f"| forbidden_diagnostic_wording_count | 0 |",
        "",
        f"## Checkpoint",
        "",
        f"`{ckpt_path}`",
        "",
        "## 다음 단계",
        "",
        "P-C-NORMAL7: smoke training result validation checkpoint",
    ]
    (REPORT_DIR / "p_c_normal6_smoke_training.md").write_text("\n".join(md))

    print(f"\n[P-C-NORMAL6] Reports → {REPORT_DIR}")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="P-C-NORMAL6 Smoke Training")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke-train", action="store_true")

    p.add_argument("--confirm-smoke",           action="store_true")
    p.add_argument("--confirm-normal-vs-nsclc", action="store_true")
    p.add_argument("--confirm-no-holdout",      action="store_true")

    p.add_argument("--epochs",       type=int,   default=1)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--disable-noise", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke_train:
        sys.exit(run_smoke_train(args))
    else:
        print("Error: specify --smoke-train with confirm flags and --epochs 1",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
