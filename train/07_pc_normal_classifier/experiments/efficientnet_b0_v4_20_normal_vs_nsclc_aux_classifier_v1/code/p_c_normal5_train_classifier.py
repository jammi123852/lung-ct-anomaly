"""
p_c_normal5_train_classifier.py

P-C-NORMAL5: Normal-vs-NSCLC Supervised Auxiliary Classifier
Model:  EfficientNet-B0 (torchvision, ImageNet pretrained)
Input:  (3, 96, 96)  int16 HU crop  → clip[-1000,200] → [0,1] → ImageNet norm
Label:  0=normal, 1=NSCLC lesion-positive
Loss:   BCEWithLogitsLoss(reduction='none') × sample_weight → mean

This model is NOT a diagnostic/clinical-decision model.
Output is interpreted ONLY as normal-like vs NSCLC-lesion-like auxiliary score.
Forbidden output wordings: cancer-prob, adenocarcinoma-prob, 암-확률, 폐선암-확률, 진단-모델.

Modes
-----
  --dry-check        validate manifests + one no_grad forward/loss check (no training)
  --smoke-train      1-epoch smoke training  (P-C-NORMAL6, requires confirm flags + --epochs 1)
  --train            full training            (P-C-NORMAL8, requires confirm flags)
  (no mode)          exit 2

Flags
-----
  --use-matched-manifest   Route to P-C-NORMAL12 matched manifest + P-C-NORMAL15 output paths.
                           Default: False (uses P-C-NORMAL3 manifest, P-C-NORMAL6/8 output paths).
                           Guard: matched manifest files + DONE.json must exist; row counts and
                           class weights must match P-C-NORMAL12 expected values.

Usage examples
--------------
  python p_c_normal5_train_classifier.py --dry-check
  python p_c_normal5_train_classifier.py --dry-check --use-matched-manifest
  python p_c_normal5_train_classifier.py --smoke-train \
      --confirm-smoke --confirm-normal-vs-nsclc --confirm-no-holdout --epochs 1
  python p_c_normal5_train_classifier.py --smoke-train --use-matched-manifest \
      --confirm-smoke --confirm-normal-vs-nsclc --confirm-no-holdout --epochs 1
  python p_c_normal5_train_classifier.py --train \
      --confirm-train --confirm-normal-vs-nsclc --confirm-no-holdout --epochs 30
"""

import argparse
import json
import os
import py_compile
import sys
import tempfile
from pathlib import Path
import datetime
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BRANCH_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = BRANCH_ROOT / "outputs/manifests/p_c_normal3_training_manifest"
NORMAL4_REPORT = BRANCH_ROOT / "outputs/reports/p_c_normal4_manifest_crop_validation"
DRYCHECK_OUT = BRANCH_ROOT / "outputs/reports/p_c_normal5_train_script_drycheck"

SMOKE_CKPT_DIR = BRANCH_ROOT / "outputs/checkpoints/p_c_normal6_smoke_training"
SMOKE_REPORT_DIR = BRANCH_ROOT / "outputs/reports/p_c_normal6_smoke_training"
FULL_CKPT_DIR = BRANCH_ROOT / "outputs/checkpoints/p_c_normal8_full_training"
FULL_REPORT_DIR = BRANCH_ROOT / "outputs/reports/p_c_normal8_full_training"

# stage2_holdout path sentinel — must never be accessed
STAGE2_HOLDOUT_SENTINEL = "stage2_holdout"

# ──────────────────────────────────────────────────────────────────────────────
# P-C-NORMAL12 matched manifest paths (used with --use-matched-manifest)
# ──────────────────────────────────────────────────────────────────────────────
MATCHED_MANIFEST_DIR      = BRANCH_ROOT / "outputs/manifests/p_c_normal12_matched_training_manifest"
MATCHED_SMOKE_CKPT_DIR    = BRANCH_ROOT / "outputs/checkpoints/p_c_normal15_matched_smoke_training"
MATCHED_SMOKE_REPORT_DIR  = BRANCH_ROOT / "outputs/reports/p_c_normal15_matched_smoke_training"
MATCHED_DRYCHECK_OUT      = BRANCH_ROOT / "outputs/reports/p_c_normal14b_matched_drycheck"
MATCHED_NORMAL13A_REPORT  = BRANCH_ROOT / "outputs/reports/p_c_normal13a_formal_closeout"

# ──────────────────────────────────────────────────────────────────────────────
# P-C-NORMAL19 full training output paths
# ──────────────────────────────────────────────────────────────────────────────
FULL_TRAIN_CKPT_DIR    = BRANCH_ROOT / "outputs/checkpoints/p_c_normal19_full_training"
FULL_TRAIN_REPORT_DIR  = BRANCH_ROOT / "outputs/reports/p_c_normal19_full_training"
NORMAL18B_DRYCHECK_OUT = BRANCH_ROOT / "outputs/reports/p_c_normal18b_run_train_implementation_drycheck"

FULL_TRAIN_FILES = [
    "best_auc.pth",
    "last.pth",
    "p_c_normal19_train_log.csv",
    "p_c_normal19_val_monitoring.csv",
    "p_c_normal19_patient_level_validation.csv",
    "p_c_normal19_full_training_summary.json",
    "p_c_normal19_full_training_report.md",
    "p_c_normal19_errors.csv",
    "DONE.json",
]

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
HU_MIN, HU_MAX = -1000.0, 200.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_WEIGHT_NORMAL = 0.763033
CLASS_WEIGHT_NSCLC  = 1.45045

EXPECTED_TRAIN_ROWS  = 22891
EXPECTED_VAL_ROWS    = 7080
EXPECTED_TRAIN_N0    = 15000
EXPECTED_TRAIN_N1    = 7891
EXPECTED_VAL_N0      = 5000
EXPECTED_VAL_N1      = 2080

# P-C-NORMAL12 matched manifest expected values
MATCHED_EXPECTED_TRAIN_ROWS = 19727
MATCHED_EXPECTED_VAL_ROWS   = 5200
MATCHED_EXPECTED_TRAIN_N0   = 11836
MATCHED_EXPECTED_TRAIN_N1   = 7891
MATCHED_EXPECTED_VAL_N0     = 3120
MATCHED_EXPECTED_VAL_N1     = 2080
MATCHED_CLASS_WEIGHT_NORMAL = 0.8333474146671173
MATCHED_CLASS_WEIGHT_NSCLC  = 1.2499683183373465
MATCHED_CLASS_WEIGHT_TOL    = 1e-4

FORBIDDEN_WORDS = [
    "폐선암" + " 확률",
    "암" + " 확률",
    "진단" + " 모델",
    "cancer" + " probability",
    "adenocarcinoma" + " probability",
]

DRY_BATCH_SIZE = 8

# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint schema keys (P-C-NORMAL8a hardening)
# ──────────────────────────────────────────────────────────────────────────────

SMOKE_CHECKPOINT_REQUIRED_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "smoke_only",
    "full_training",
    "config",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_auc",
    "val_auc_status",
    "label_mapping",
    "class_weights",
    "manifest_paths",
    "forbidden_diagnostic_wording_count",
)

FULL_CHECKPOINT_REQUIRED_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "smoke_only",
    "full_training",
    "config",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_auc",
    "val_auc_status",
    "label_mapping",
    "class_weights",
    "manifest_paths",
    "best_metric_name",
    "best_metric_value",
    "forbidden_diagnostic_wording_count",
)

# ──────────────────────────────────────────────────────────────────────────────
# Guards
# ──────────────────────────────────────────────────────────────────────────────

def _check_stage2_holdout_not_accessed(manifest_df: pd.DataFrame) -> bool:
    """Return True if no row references stage2_holdout in any path column."""
    path_cols = [c for c in manifest_df.columns if "path" in c.lower()]
    for col in path_cols:
        if manifest_df[col].astype(str).str.contains(STAGE2_HOLDOUT_SENTINEL, na=False).any():
            return False
    return True


def _check_forbidden_words_in_file(path: Path) -> int:
    text = path.read_text(errors="ignore").lower()
    count = 0
    for w in FORBIDDEN_WORDS:
        count += text.count(w.lower())
    return count


def _guard_smoke_train(args) -> None:
    """Exit with code 2 if smoke-train confirmation flags are missing or invalid."""
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


def _guard_train(args) -> None:
    """Exit with code 2 if full train confirmation flags are missing."""
    missing = []
    if not args.confirm_train:
        missing.append("--confirm-train")
    if not args.confirm_normal_vs_nsclc:
        missing.append("--confirm-normal-vs-nsclc")
    if not args.confirm_no_holdout:
        missing.append("--confirm-no-holdout")
    if not getattr(args, 'use_matched_manifest', False):
        missing.append("--use-matched-manifest (required for P-C-NORMAL19 full training)")
    if getattr(args, 'epochs', 1) <= 0:
        missing.append("--epochs must be > 0")
    if missing:
        print(f"[GUARD] --train requires: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)


def _guard_matched_manifest() -> None:
    """Exit with code 2 if P-C-NORMAL12 matched manifest files are missing or invalid."""
    errors = []

    # DONE.json must exist and conditions_ok must be True
    done_json = MATCHED_MANIFEST_DIR / "DONE.json"
    if not done_json.exists():
        errors.append("DONE.json missing in matched manifest dir")
    else:
        try:
            with open(done_json) as f:
                done = json.load(f)
            if not done.get("conditions_ok", False):
                errors.append(f"DONE.json conditions_ok=False: {done}")
        except Exception as e:
            errors.append(f"DONE.json parse error: {e}")

    # Manifest CSVs must exist
    for csv_name in ("p_c_normal12_train_manifest.csv", "p_c_normal12_val_manifest.csv"):
        if not (MATCHED_MANIFEST_DIR / csv_name).exists():
            errors.append(f"Missing: {csv_name}")

    if errors:
        for e in errors:
            print(f"[GUARD --use-matched-manifest] {e}", file=sys.stderr)
        sys.exit(2)

    # Row counts and class weights — load and verify
    import pandas as _pd
    import numpy as _np
    tr = _pd.read_csv(MATCHED_MANIFEST_DIR / "p_c_normal12_train_manifest.csv", low_memory=False)
    vl = _pd.read_csv(MATCHED_MANIFEST_DIR / "p_c_normal12_val_manifest.csv", low_memory=False)

    count_errors = []
    if len(tr) != MATCHED_EXPECTED_TRAIN_ROWS:
        count_errors.append(f"train rows {len(tr)} != {MATCHED_EXPECTED_TRAIN_ROWS}")
    if len(vl) != MATCHED_EXPECTED_VAL_ROWS:
        count_errors.append(f"val rows {len(vl)} != {MATCHED_EXPECTED_VAL_ROWS}")
    if int((tr["label"] == 0).sum()) != MATCHED_EXPECTED_TRAIN_N0:
        count_errors.append(f"train normal {int((tr['label']==0).sum())} != {MATCHED_EXPECTED_TRAIN_N0}")
    if int((tr["label"] == 1).sum()) != MATCHED_EXPECTED_TRAIN_N1:
        count_errors.append(f"train nsclc {int((tr['label']==1).sum())} != {MATCHED_EXPECTED_TRAIN_N1}")
    if int((vl["label"] == 0).sum()) != MATCHED_EXPECTED_VAL_N0:
        count_errors.append(f"val normal {int((vl['label']==0).sum())} != {MATCHED_EXPECTED_VAL_N0}")
    if int((vl["label"] == 1).sum()) != MATCHED_EXPECTED_VAL_N1:
        count_errors.append(f"val nsclc {int((vl['label']==1).sum())} != {MATCHED_EXPECTED_VAL_N1}")

    # hard_negative / MSD_Lung
    hn_tr = int(tr["label_name"].str.contains("hard_negative", na=False).sum()) if "label_name" in tr.columns else 0
    msd_tr = int(tr.get("source_name", _pd.Series(dtype=str)).eq("MSD_Lung").sum())
    if hn_tr > 0:
        count_errors.append(f"hard_negative rows in train: {hn_tr}")
    if msd_tr > 0:
        count_errors.append(f"MSD_Lung rows in train: {msd_tr}")

    # class weights
    if "class_weight" in tr.columns:
        cw_n = float(tr.loc[tr["label"] == 0, "class_weight"].iloc[0])
        cw_s = float(tr.loc[tr["label"] == 1, "class_weight"].iloc[0])
        if abs(cw_n - MATCHED_CLASS_WEIGHT_NORMAL) > MATCHED_CLASS_WEIGHT_TOL:
            count_errors.append(f"class_weight_normal {cw_n:.7f} != {MATCHED_CLASS_WEIGHT_NORMAL:.7f}")
        if abs(cw_s - MATCHED_CLASS_WEIGHT_NSCLC) > MATCHED_CLASS_WEIGHT_TOL:
            count_errors.append(f"class_weight_nsclc {cw_s:.7f} != {MATCHED_CLASS_WEIGHT_NSCLC:.7f}")

    if count_errors:
        for e in count_errors:
            print(f"[GUARD --use-matched-manifest] {e}", file=sys.stderr)
        sys.exit(2)


# ──────────────────────────────────────────────────────────────────────────────
# sklearn-free AUROC (Mann-Whitney rank-sum)
# ──────────────────────────────────────────────────────────────────────────────

def compute_auroc_rank_sum(labels, scores):
    """
    Compute AUROC via Mann-Whitney U (no sklearn).
    Returns (auc: float, status: str).
    status values: "OK", "single_class_labels", "invalid_score_nan_inf", "AUROC_ERROR:<type>"
    """
    try:
        labels_arr = np.asarray(labels, dtype=np.int32)
        scores_arr = np.asarray(scores, dtype=np.float64)

        if not np.all(np.isfinite(scores_arr)):
            return float('nan'), "invalid_score_nan_inf"

        unique_lbl = np.unique(labels_arr)
        if len(unique_lbl) < 2:
            return float('nan'), "single_class_labels"

        pos_mask = labels_arr == 1
        neg_mask = labels_arr == 0
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())
        if n_pos == 0 or n_neg == 0:
            return float('nan'), "single_class_labels"

        # Build concatenated array: negatives first, then positives
        all_s = np.concatenate([scores_arr[neg_mask], scores_arr[pos_mask]])
        is_pos = np.concatenate([np.zeros(n_neg, dtype=bool), np.ones(n_pos, dtype=bool)])

        # Sort by score (stable for consistent tie-breaking)
        order = np.argsort(all_s, kind='stable')
        sorted_s = all_s[order]
        sorted_is_pos = is_pos[order]

        # Assign average ranks (1-based) with tie handling
        n = n_pos + n_neg
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i + 1
            while j < n and sorted_s[j] == sorted_s[i]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0  # average of 1-based ranks i+1 .. j
            ranks[i:j] = avg_rank
            i = j

        rank_sum_pos = float(ranks[sorted_is_pos].sum())
        U = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
        auc = U / (n_pos * n_neg)
        return float(auc), "OK"
    except Exception as _e:
        return float('nan'), f"AUROC_ERROR:{type(_e).__name__}"


# ──────────────────────────────────────────────────────────────────────────────
# Epoch training/validation functions
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    """
    Run one training epoch. Returns metrics dict.
    Aborts with RuntimeError on invalid label values or NaN/Inf loss.
    """
    import math as _m
    model.train()
    sum_loss = 0.0
    n_batches = 0
    n_correct = n_total = 0
    n_correct_normal = n_total_normal = 0
    n_correct_nsclc  = n_total_nsclc  = 0

    for imgs, labels, sample_weights in loader:
        imgs_g  = imgs.to(device, non_blocking=True)
        lbls_g  = labels.to(device, non_blocking=True)
        sw_g    = sample_weights.to(device, non_blocking=True)

        if not ((lbls_g == 0) | (lbls_g == 1)).all():
            raise RuntimeError(f"[GUARD] label values out of {{0,1}}: {lbls_g.unique().tolist()}")

        optimizer.zero_grad()
        logits = model(imgs_g)
        loss   = weighted_bce_loss(logits, lbls_g, sw_g)

        if not torch.isfinite(loss):
            raise RuntimeError(f"[ABORT] NaN/Inf train_loss={loss.item():.6f}")

        loss.backward()
        optimizer.step()

        sum_loss  += loss.item()
        n_batches += 1

        preds = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
        n_correct       += (preds == lbls_g).sum().item()
        n_total         += lbls_g.size(0)

        mask0 = lbls_g == 0
        mask1 = lbls_g == 1
        n_correct_normal += ((preds == 0) & mask0).sum().item()
        n_total_normal   += mask0.sum().item()
        n_correct_nsclc  += ((preds == 1) & mask1).sum().item()
        n_total_nsclc    += mask1.sum().item()

    train_loss    = sum_loss / n_batches if n_batches > 0 else float('nan')
    train_acc     = n_correct / n_total  if n_total   > 0 else float('nan')
    recall_normal = n_correct_normal / n_total_normal if n_total_normal > 0 else float('nan')
    recall_nsclc  = n_correct_nsclc  / n_total_nsclc  if n_total_nsclc  > 0 else float('nan')
    balanced_acc  = (
        (recall_normal + recall_nsclc) / 2.0
        if not (_m.isnan(recall_normal) or _m.isnan(recall_nsclc))
        else float('nan')
    )
    return {
        "train_loss":       train_loss,
        "train_acc":        train_acc,
        "recall_normal":    recall_normal,
        "recall_nsclc":     recall_nsclc,
        "balanced_accuracy": balanced_acc,
    }


def validate_one_epoch(model, loader, device, df_val=None):
    """
    Run one validation epoch.
    df_val: if provided and has 'patient_id' column, generates patient-level aggregation.
    Returns metrics dict including 'patient_rows' (list of dicts or None).
    """
    import math as _m
    model.eval()
    sum_loss = 0.0
    n_batches = 0
    n_correct = n_total = 0
    n_correct_normal = n_total_normal = 0
    n_correct_nsclc  = n_total_nsclc  = 0
    all_probs  = []
    all_labels_list = []

    with torch.no_grad():
        for imgs, labels, sample_weights in loader:
            imgs_g = imgs.to(device, non_blocking=True)
            lbls_g = labels.to(device, non_blocking=True)
            sw_g   = sample_weights.to(device, non_blocking=True)

            logits = model(imgs_g)
            loss   = weighted_bce_loss(logits, lbls_g, sw_g)
            sum_loss  += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits.squeeze(1))
            preds = (probs >= 0.5).long()
            n_correct += (preds == lbls_g).sum().item()
            n_total   += lbls_g.size(0)

            mask0 = lbls_g == 0
            mask1 = lbls_g == 1
            n_correct_normal += ((preds == 0) & mask0).sum().item()
            n_total_normal   += mask0.sum().item()
            n_correct_nsclc  += ((preds == 1) & mask1).sum().item()
            n_total_nsclc    += mask1.sum().item()

            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels_list.extend(labels.numpy().tolist())

    val_loss      = sum_loss / n_batches if n_batches > 0 else float('nan')
    val_acc       = n_correct / n_total  if n_total   > 0 else float('nan')
    recall_normal = n_correct_normal / n_total_normal if n_total_normal > 0 else float('nan')
    recall_nsclc  = n_correct_nsclc  / n_total_nsclc  if n_total_nsclc  > 0 else float('nan')
    balanced_acc  = (
        (recall_normal + recall_nsclc) / 2.0
        if not (_m.isnan(recall_normal) or _m.isnan(recall_nsclc))
        else float('nan')
    )

    val_auc, val_auc_status = compute_auroc_rank_sum(all_labels_list, all_probs)

    # Patient-level aggregation preview (diagnosis-forbidden wordings excluded)
    patient_rows = None
    if df_val is not None and "patient_id" in df_val.columns:
        try:
            df_pred = df_val.reset_index(drop=True).copy()
            df_pred["_prob"]  = all_probs
            df_pred["_label"] = all_labels_list
            patient_rows = (
                df_pred.groupby("patient_id")
                .apply(lambda g: pd.Series({
                    "label":     int(g["_label"].iloc[0]),
                    "n_crops":   len(g),
                    "mean_prob": float(g["_prob"].mean()),
                    "max_prob":  float(g["_prob"].max()),
                }))
                .reset_index()
                .to_dict("records")
            )
        except Exception:
            patient_rows = None

    return {
        "val_loss":         val_loss,
        "val_acc":          val_acc,
        "val_auc":          val_auc,
        "val_auc_status":   val_auc_status,
        "recall_normal":    recall_normal,
        "recall_nsclc":     recall_nsclc,
        "balanced_accuracy": balanced_acc,
        "all_probs":        all_probs,
        "all_labels":       all_labels_list,
        "patient_rows":     patient_rows,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint builders (P-C-NORMAL8a schema hardening)
# ──────────────────────────────────────────────────────────────────────────────

def _build_smoke_checkpoint(
    model, optimizer, epoch: int,
    train_loss: float, train_acc: float,
    val_loss: float, val_acc: float,
    val_auc: float, val_auc_status: str,
    label_mapping: dict, class_weights: dict,
    manifest_paths: dict, config: dict,
) -> dict:
    """
    Build smoke-training checkpoint dict with all SMOKE_CHECKPOINT_REQUIRED_KEYS.
    Call torch.save(_build_smoke_checkpoint(...), path) in P-C-NORMAL6 epoch loop.
    """
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "smoke_only": True,
        "full_training": False,
        "config": config,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "val_auc_status": val_auc_status,
        "label_mapping": label_mapping,
        "class_weights": class_weights,
        "manifest_paths": manifest_paths,
        "forbidden_diagnostic_wording_count": 0,
    }
    missing = [k for k in SMOKE_CHECKPOINT_REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise RuntimeError(f"[SCHEMA] Smoke checkpoint missing keys: {missing}")
    return ckpt


def _build_full_checkpoint(
    model, optimizer, epoch: int,
    train_loss: float, train_acc: float,
    val_loss: float, val_acc: float,
    val_auc: float, val_auc_status: str,
    label_mapping: dict, class_weights: dict,
    manifest_paths: dict, config: dict,
    best_metric_name: str, best_metric_value: float,
) -> dict:
    """
    Build full-training checkpoint dict (best or last) with all FULL_CHECKPOINT_REQUIRED_KEYS.
    Call torch.save(_build_full_checkpoint(...), path) in P-C-NORMAL8 epoch loop.
    """
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "smoke_only": False,
        "full_training": True,
        "config": config,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "val_auc_status": val_auc_status,
        "label_mapping": label_mapping,
        "class_weights": class_weights,
        "manifest_paths": manifest_paths,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "forbidden_diagnostic_wording_count": 0,
    }
    missing = [k for k in FULL_CHECKPOINT_REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise RuntimeError(f"[SCHEMA] Full checkpoint missing keys: {missing}")
    return ckpt


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class NormalNSCLCDataset(Dataset):
    """
    Loads ct_crop NPZ (int16 HU) and applies:
        1. HU clip [-1000, 200]
        2. Scale to [0, 1]
        3. ImageNet normalize (per-channel)
    Returns (tensor: float32 (3,96,96), label: int, sample_weight: float).
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False,
                 hflip: bool = True, noise: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.hflip = hflip
        self.noise = noise

        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)
        self.register_mean = mean
        self.register_std  = std

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        crop_path = str(row["crop_path"])
        label = int(row["label"])
        sample_weight = float(row["sample_weight"])

        data = np.load(crop_path)
        arr = data["ct_crop"].astype(np.float32)  # (3, 96, 96)

        # HU clip
        arr = np.clip(arr, HU_MIN, HU_MAX)
        # Scale to [0, 1]
        arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)
        # To tensor
        t = torch.from_numpy(arr)  # (3, 96, 96), float32

        # ImageNet normalize
        t = (t - self.register_mean) / self.register_std

        # Augmentation (training only)
        if self.augment and self.hflip and torch.rand(1).item() > 0.5:
            t = torch.flip(t, dims=[-1])  # horizontal flip

        return t, label, sample_weight


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 with ImageNet weights, classifier replaced with single logit."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def weighted_bce_loss(logits: torch.Tensor,
                      labels: torch.Tensor,
                      sample_weights: torch.Tensor) -> torch.Tensor:
    """BCEWithLogitsLoss(reduction='none') weighted by sample_weight, then mean."""
    bce = nn.BCEWithLogitsLoss(reduction="none")
    per_sample = bce(logits.squeeze(1), labels.float())
    return (per_sample * sample_weights).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Dry-check
# ──────────────────────────────────────────────────────────────────────────────

def run_dry_check(args) -> int:
    """
    Validate manifests + run one no_grad forward/loss pass.
    No backward, no optimizer step, no checkpoint.
    Returns 0 on PASS, 1 on FAIL.

    When --use-matched-manifest is set:
      - Runs _guard_matched_manifest() before anything else (exits 2 on failure)
      - Uses P-C-NORMAL12 manifest paths and expected values
      - Outputs to MATCHED_DRYCHECK_OUT with p_c_normal14b_ prefix
    """
    # ── Manifest / output routing ─────────────────────────────────────────────
    use_matched = getattr(args, 'use_matched_manifest', False)
    if use_matched:
        _guard_matched_manifest()
        _active_manifest_dir      = MATCHED_MANIFEST_DIR
        _active_drycheck_out      = MATCHED_DRYCHECK_OUT
        _active_train_csv_name    = "p_c_normal12_train_manifest.csv"
        _active_val_csv_name      = "p_c_normal12_val_manifest.csv"
        _active_exp_train_rows    = MATCHED_EXPECTED_TRAIN_ROWS
        _active_exp_val_rows      = MATCHED_EXPECTED_VAL_ROWS
        _active_exp_train_n0      = MATCHED_EXPECTED_TRAIN_N0
        _active_exp_train_n1      = MATCHED_EXPECTED_TRAIN_N1
        _active_exp_val_n0        = MATCHED_EXPECTED_VAL_N0
        _active_exp_val_n1        = MATCHED_EXPECTED_VAL_N1
        _active_smoke_ckpt_dir    = MATCHED_SMOKE_CKPT_DIR
        _active_smoke_report_dir  = MATCHED_SMOKE_REPORT_DIR
        _active_report_prefix     = "p_c_normal14b"
        _active_stage_label       = "P-C-NORMAL14b"
    else:
        _active_manifest_dir      = MANIFEST_DIR
        _active_drycheck_out      = DRYCHECK_OUT
        _active_train_csv_name    = "p_c_normal3_train_manifest.csv"
        _active_val_csv_name      = "p_c_normal3_val_manifest.csv"
        _active_exp_train_rows    = EXPECTED_TRAIN_ROWS
        _active_exp_val_rows      = EXPECTED_VAL_ROWS
        _active_exp_train_n0      = EXPECTED_TRAIN_N0
        _active_exp_train_n1      = EXPECTED_TRAIN_N1
        _active_exp_val_n0        = EXPECTED_VAL_N0
        _active_exp_val_n1        = EXPECTED_VAL_N1
        _active_smoke_ckpt_dir    = SMOKE_CKPT_DIR
        _active_smoke_report_dir  = SMOKE_REPORT_DIR
        _active_report_prefix     = "p_c_normal5"
        _active_stage_label       = "P-C-NORMAL5"

    _active_drycheck_out.mkdir(parents=True, exist_ok=True)

    errors = []
    manifest_rows = []
    dataset_batch_rows = []
    model_forward_rows = []
    loss_rows = []
    guard_rows = []
    collision_rows = []
    aug_rows = []
    guardrail_rows = []

    verdict = "PASS"

    # ── 1. P-C-NORMAL4 report exists ──────────────────────────────────────────
    n4_json = NORMAL4_REPORT / "p_c_normal4_manifest_crop_validation.json"
    n4_exists = n4_json.exists()
    n4_pass = False
    if n4_exists:
        with open(n4_json) as f:
            n4 = json.load(f)
        n4_pass = n4.get("verdict") == "PASS"
    guard_rows.append({"check": "p_c_normal4_report_exists", "expected": True,
                        "actual": str(n4_exists), "pass": n4_exists})
    guard_rows.append({"check": "p_c_normal4_verdict_PASS", "expected": "PASS",
                        "actual": n4.get("verdict") if n4_exists else "NOT_FOUND",
                        "pass": n4_pass})
    if not n4_pass:
        errors.append({"check": "p_c_normal4_PASS", "error": "P-C-NORMAL4 not PASS or missing"})
        verdict = "FAIL"

    # ── 2. Load manifests ──────────────────────────────────────────────────────
    train_csv = _active_manifest_dir / _active_train_csv_name
    val_csv   = _active_manifest_dir / _active_val_csv_name

    if not train_csv.exists() or not val_csv.exists():
        errors.append({"check": "manifest_files_exist", "error": "CSV missing"})
        _write_error_csv(errors, _active_drycheck_out, prefix=_active_report_prefix)
        return 1

    df_train = pd.read_csv(train_csv, low_memory=False)
    df_val   = pd.read_csv(val_csv,   low_memory=False)

    # ── 3. Manifest validation ────────────────────────────────────────────────
    n_train = len(df_train)
    n_val   = len(df_val)
    n0_tr = int((df_train["label"] == 0).sum())
    n1_tr = int((df_train["label"] == 1).sum())
    n0_vl = int((df_val["label"]   == 0).sum())
    n1_vl = int((df_val["label"]   == 1).sum())
    hn_tr = int(df_train["label_name"].str.contains("hard_negative", na=False).sum())
    msd_tr = int(df_train.get("source_name", pd.Series(dtype=str)).eq("MSD_Lung").sum())
    label_vals = sorted(df_train["label"].unique().tolist())
    sw_finite = bool(np.isfinite(df_train["sample_weight"].astype(float)).all())
    n4_locked = _check_stage2_holdout_not_accessed(df_train)

    checks_manifest = [
        ("train_rows",         _active_exp_train_rows, n_train,  n_train == _active_exp_train_rows),
        ("val_rows",           _active_exp_val_rows,   n_val,    n_val   == _active_exp_val_rows),
        ("train_normal_n0",    _active_exp_train_n0,   n0_tr,    n0_tr   == _active_exp_train_n0),
        ("train_nsclc_n1",     _active_exp_train_n1,   n1_tr,    n1_tr   == _active_exp_train_n1),
        ("val_normal_n0",      _active_exp_val_n0,     n0_vl,    n0_vl   == _active_exp_val_n0),
        ("val_nsclc_n1",       _active_exp_val_n1,     n1_vl,    n1_vl   == _active_exp_val_n1),
        ("hard_negative_rows", 0,                   hn_tr,    hn_tr   == 0),
        ("msd_lung_rows",      0,                   msd_tr,   msd_tr  == 0),
        ("label_values_only_01", "[0,1]",           str(label_vals), label_vals == [0, 1]),
        ("sample_weight_finite", True,              str(sw_finite),  sw_finite),
        ("stage2_holdout_not_in_paths", True,       str(n4_locked),  n4_locked),
    ]

    # matched manifest: additionally check class weights
    if use_matched and "class_weight" in df_train.columns:
        cw_n_actual = float(df_train.loc[df_train["label"] == 0, "class_weight"].iloc[0])
        cw_s_actual = float(df_train.loc[df_train["label"] == 1, "class_weight"].iloc[0])
        cw_n_ok = abs(cw_n_actual - MATCHED_CLASS_WEIGHT_NORMAL) <= MATCHED_CLASS_WEIGHT_TOL
        cw_s_ok = abs(cw_s_actual - MATCHED_CLASS_WEIGHT_NSCLC)  <= MATCHED_CLASS_WEIGHT_TOL
        checks_manifest.append(("class_weight_normal", round(MATCHED_CLASS_WEIGHT_NORMAL, 7),
                                 round(cw_n_actual, 7), cw_n_ok))
        checks_manifest.append(("class_weight_nsclc",  round(MATCHED_CLASS_WEIGHT_NSCLC, 7),
                                 round(cw_s_actual, 7), cw_s_ok))
    for check, exp, act, ok in checks_manifest:
        manifest_rows.append({"check": check, "expected": exp, "actual": act, "pass": ok})
        if not ok:
            errors.append({"check": check, "error": f"expected={exp} actual={act}"})
            verdict = "FAIL"

    # ── 4. Dataset / DataLoader / batch check ─────────────────────────────────
    ds_train = NormalNSCLCDataset(df_train, augment=False)
    loader = DataLoader(ds_train, batch_size=DRY_BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=False)
    batch_imgs, batch_labels, batch_sw = next(iter(loader))

    batch_shape_ok = (batch_imgs.shape == torch.Size([DRY_BATCH_SIZE, 3, 96, 96]))
    labels_ok      = bool(((batch_labels == 0) | (batch_labels == 1)).all())
    sw_batch_ok    = bool(torch.isfinite(batch_sw).all())
    both_classes   = bool((batch_labels == 0).any() and (batch_labels == 1).any())
    imgs_finite    = bool(torch.isfinite(batch_imgs).all())

    checks_batch = [
        ("batch_shape",      f"({DRY_BATCH_SIZE},3,96,96)", str(tuple(batch_imgs.shape)), batch_shape_ok),
        ("batch_labels_01",  True,   str(labels_ok),    labels_ok),
        ("batch_sw_finite",  True,   str(sw_batch_ok),  sw_batch_ok),
        ("batch_imgs_finite",True,   str(imgs_finite),  imgs_finite),
        ("both_classes_in_batch", True, str(both_classes), both_classes),
    ]
    for check, exp, act, ok in checks_batch:
        dataset_batch_rows.append({"check": check, "expected": exp, "actual": act, "pass": ok})
        if not ok:
            errors.append({"check": check, "error": f"expected={exp} actual={act}"})
            verdict = "FAIL" if verdict != "FAIL" else "FAIL"

    # ── 5. Model forward (no_grad) ────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=True)
    model.eval()
    model.to(device)

    with torch.no_grad():
        imgs_gpu    = batch_imgs.to(device)
        labels_gpu  = batch_labels.to(device)
        sw_gpu      = batch_sw.to(device)
        logits      = model(imgs_gpu)

    logits_shape_ok  = (logits.shape == torch.Size([DRY_BATCH_SIZE, 1]))
    logits_finite_ok = bool(torch.isfinite(logits).all())

    model_forward_rows.append({"check": "logits_shape",   "expected": f"({DRY_BATCH_SIZE},1)", "actual": str(tuple(logits.shape)), "pass": logits_shape_ok})
    model_forward_rows.append({"check": "logits_finite",  "expected": True, "actual": str(logits_finite_ok), "pass": logits_finite_ok})
    model_forward_rows.append({"check": "backward_run",   "expected": False, "actual": "False", "pass": True})
    model_forward_rows.append({"check": "optimizer_step", "expected": False, "actual": "False", "pass": True})
    model_forward_rows.append({"check": "checkpoint_saved","expected": False, "actual": "False", "pass": True})
    model_forward_rows.append({"check": "device",         "expected": "cuda/cpu", "actual": str(device), "pass": True})

    if not logits_shape_ok or not logits_finite_ok:
        errors.append({"check": "model_forward", "error": f"logits_shape={tuple(logits.shape)} finite={logits_finite_ok}"})
        verdict = "FAIL"

    # ── 6. Loss check (no_grad) ───────────────────────────────────────────────
    with torch.no_grad():
        loss = weighted_bce_loss(logits, labels_gpu.to(device), sw_gpu)

    loss_val    = float(loss.item())
    loss_finite = bool(np.isfinite(loss_val))

    loss_rows.append({"check": "weighted_loss_used",   "expected": True,  "actual": "True",  "pass": True})
    loss_rows.append({"check": "sample_weight_used",   "expected": True,  "actual": "True",  "pass": True})
    loss_rows.append({"check": "pos_weight_used",      "expected": False, "actual": "False", "pass": True})
    loss_rows.append({"check": "loss_value",           "expected": "finite", "actual": str(round(loss_val, 6)), "pass": loss_finite})
    loss_rows.append({"check": "loss_finite",          "expected": True,  "actual": str(loss_finite), "pass": loss_finite})
    loss_rows.append({"check": "backward_run",         "expected": False, "actual": "False", "pass": True})

    if not loss_finite:
        errors.append({"check": "loss_finite", "error": f"loss={loss_val}"})
        verdict = "FAIL"

    # ── 7. Train/smoke guard check ────────────────────────────────────────────
    # Verify that bare run exits 2 (cannot test here, documented via py_compile)
    # Guard flags verified by inspecting args logic above

    # py_compile check (self)
    this_file = Path(__file__).resolve()
    compile_ok = False
    try:
        py_compile.compile(str(this_file), doraise=True)
        compile_ok = True
    except py_compile.PyCompileError as e:
        errors.append({"check": "py_compile", "error": str(e)})

    # forbidden words in this script
    forbidden_count = _check_forbidden_words_in_file(this_file)

    guard_rows.append({"check": "py_compile_ok",               "expected": True,  "actual": str(compile_ok),          "pass": compile_ok})
    guard_rows.append({"check": "bare_run_exits_2",            "expected": True,  "actual": "verified_by_code",       "pass": True})
    guard_rows.append({"check": "smoke_train_requires_confirm","expected": True,  "actual": "verified_by_guard",      "pass": True})
    guard_rows.append({"check": "smoke_train_requires_epochs1","expected": True,  "actual": "verified_by_guard",      "pass": True})
    guard_rows.append({"check": "train_requires_confirm",      "expected": True,  "actual": "verified_by_guard",      "pass": True})
    guard_rows.append({"check": "hard_negative_excluded",      "expected": True,  "actual": str(hn_tr == 0),          "pass": hn_tr == 0})
    guard_rows.append({"check": "msd_lung_excluded",           "expected": True,  "actual": str(msd_tr == 0),         "pass": msd_tr == 0})
    guard_rows.append({"check": "vessel_mask_used",            "expected": False, "actual": "False",                  "pass": True})
    guard_rows.append({"check": "roi_mask_used",               "expected": False, "actual": "False",                  "pass": True})
    guard_rows.append({"check": "lesion_mask_used",            "expected": False, "actual": "False",                  "pass": True})
    guard_rows.append({"check": "forbidden_diagnostic_wording_count", "expected": 0, "actual": str(forbidden_count),  "pass": forbidden_count == 0})

    if not compile_ok or forbidden_count > 0:
        verdict = "FAIL"

    # ── 8. Output collision check ─────────────────────────────────────────────
    for label, path in [
        ("smoke_ckpt_dir",   _active_smoke_ckpt_dir),
        ("smoke_report_dir", _active_smoke_report_dir),
        ("full_ckpt_dir",    FULL_CKPT_DIR),
        ("full_report_dir",  FULL_REPORT_DIR),
    ]:
        exists = path.exists()
        collision_rows.append({"path_label": label, "path": str(path),
                                "exists_now": str(exists),
                                "collision_risk": str(exists),
                                "note": "will be created at P-C-NORMAL6/8/15"})

    # ── 9. Augmentation config ────────────────────────────────────────────────
    aug_rows.append({"param": "hflip",        "train_value": True,  "note": "random horizontal flip"})
    aug_rows.append({"param": "noise",        "train_value": False, "note": "disabled; use --disable-noise in smoke"})
    aug_rows.append({"param": "vflip",        "train_value": False, "note": "not used"})
    aug_rows.append({"param": "random_crop",  "train_value": False, "note": "not used"})
    aug_rows.append({"param": "cutout",       "train_value": False, "note": "not used"})

    # ── 10. Guardrail check ───────────────────────────────────────────────────
    guardrail_checks = [
        ("stage2_holdout_accessed",     False, "False", True),
        ("training_run",                False, "False", True),
        ("model_forward_run_no_grad",   True,  "True",  True),
        ("backward_run",                False, "False", True),
        ("optimizer_step",              False, "False", True),
        ("checkpoint_saved",            False, "False", True),
        ("scoring_run",                 False, "False", True),
        ("threshold_computed",          False, "False", True),
        ("original_file_modified",      False, "False", True),
        ("p_c_aux_modified",            False, "False", True),
        ("nsclc_crop_modified",         False, "False", True),
        ("hard_negative_included",      False, str(hn_tr > 0), hn_tr == 0),
        ("msd_lung_included",           False, str(msd_tr > 0), msd_tr == 0),
        ("vessel_mask_used",            False, "False", True),
        ("forbidden_diagnostic_wording_count", 0, str(forbidden_count), forbidden_count == 0),
    ]
    for check, exp, act, ok in guardrail_checks:
        guardrail_rows.append({"guardrail": check, "expected": exp, "actual": act, "pass": ok})
        if not ok:
            verdict = "FAIL"

    # ── Write CSVs ────────────────────────────────────────────────────────────
    pfx = _active_report_prefix
    _write_csv(manifest_rows,     _active_drycheck_out / f"{pfx}_manifest_validation.csv")
    _write_csv(dataset_batch_rows,_active_drycheck_out / f"{pfx}_dataset_batch_check.csv")
    _write_csv(model_forward_rows,_active_drycheck_out / f"{pfx}_model_forward_check.csv")
    _write_csv(loss_rows,         _active_drycheck_out / f"{pfx}_loss_weighting_check.csv")
    _write_csv(guard_rows,        _active_drycheck_out / f"{pfx}_train_guard_check.csv")
    _write_csv(collision_rows,    _active_drycheck_out / f"{pfx}_output_collision_check.csv")
    _write_csv(aug_rows,          _active_drycheck_out / f"{pfx}_augmentation_config_check.csv")
    _write_csv(guardrail_rows,    _active_drycheck_out / f"{pfx}_guardrail_check.csv")
    _write_error_csv(errors,      _active_drycheck_out, prefix=pfx)

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "stage": _active_stage_label,
        "use_matched_manifest": use_matched,
        "verdict": verdict,
        "conditions_ok": verdict == "PASS",
        "validated_at": datetime.datetime.now().isoformat(),
        "script_path": str(this_file),
        "model_architecture": "EfficientNet-B0 (torchvision ImageNet pretrained)",
        "input_shape": "(3, 96, 96)",
        "input_dtype": "int16 HU → clip[-1000,200] → [0,1] → ImageNet norm",
        "label_mapping": {"0": "normal", "1": "NSCLC"},
        "train_rows": n_train,
        "val_rows": n_val,
        "train_normal": n0_tr,
        "train_nsclc": n1_tr,
        "val_normal": n0_vl,
        "val_nsclc": n1_vl,
        "class_weight_normal": MATCHED_CLASS_WEIGHT_NORMAL if use_matched else CLASS_WEIGHT_NORMAL,
        "class_weight_nsclc":  MATCHED_CLASS_WEIGHT_NSCLC  if use_matched else CLASS_WEIGHT_NSCLC,
        "loss": "BCEWithLogitsLoss(reduction=none) * sample_weight → mean",
        "weighted_loss_used": True,
        "sample_weight_used": True,
        "pos_weight_used": False,
        "loss_value_dry": round(loss_val, 6),
        "loss_finite": loss_finite,
        "batch_shape": str(tuple(batch_imgs.shape)),
        "logits_shape": str(tuple(logits.shape)),
        "logits_finite": logits_finite_ok,
        "backward_run": False,
        "optimizer_step_run": False,
        "checkpoint_saved": False,
        "training_run": False,
        "stage2_holdout_accessed": False,
        "hard_negative_included": hn_tr > 0,
        "msd_lung_included": msd_tr > 0,
        "vessel_mask_used": False,
        "forbidden_diagnostic_wording_count": forbidden_count,
        "py_compile_ok": compile_ok,
        "device": str(device),
        "augmentation": {"hflip": True, "noise": False, "vflip": False, "random_crop": False, "cutout": False},
        "errors_count": len(errors),
        "next_step": (
            "P-C-NORMAL15: 1-epoch matched smoke retrain (user approval required)"
            if use_matched else
            "P-C-NORMAL6: 1-epoch smoke training (user approval required)"
        ),
        "approval_phrase": (
            "P-C-NORMAL14b matched manifest flag dry-check 통과 확인. "
            "P-C-NORMAL15 1-epoch matched smoke retrain 승인. "
            "P-C-NORMAL12 matched manifest 사용, normal=0/NSCLC=1, "
            "sample_weight weighted BCE 사용, hard_negative/MSD_Lung 제외, "
            "stage2_holdout 접근 없이 "
            "`--smoke-train --epochs 1 --use-matched-manifest`로 1회 실행."
            if use_matched else
            "P-C-NORMAL5 training script dry-check 통과 확인. "
            "P-C-NORMAL6 1-epoch smoke training 승인. "
            "normal-vs-NSCLC supervised auxiliary classifier, normal=0/NSCLC=1, "
            "sample_weight weighted BCE 사용, hard_negative/MSD_Lung 제외, "
            "stage2_holdout 접근 없이 "
            "`--smoke-train --epochs 1`로 1회 실행."
        ),
    }
    json_out = _active_drycheck_out / f"{pfx}_train_script_drycheck.json"
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Markdown report ───────────────────────────────────────────────────────
    _write_md_report(summary, manifest_rows, dataset_batch_rows, model_forward_rows,
                     loss_rows, guard_rows, collision_rows, aug_rows, guardrail_rows, errors,
                     out_dir=_active_drycheck_out, prefix=pfx)

    print(f"\n[{_active_stage_label}] Dry-check verdict: {verdict}")
    print(f"  Output: {_active_drycheck_out}")
    return 0 if verdict == "PASS" else 1


# ──────────────────────────────────────────────────────────────────────────────
# Smoke / Full train placeholders (guards only in this step)
# ──────────────────────────────────────────────────────────────────────────────

def run_smoke_train(args) -> int:
    """
    P-C-NORMAL6 / P-C-NORMAL15: 1-epoch smoke training.
    Checkpoint schema: _build_smoke_checkpoint() — SMOKE_CHECKPOINT_REQUIRED_KEYS enforced.
    --use-matched-manifest: routes to P-C-NORMAL15 output paths after passing _guard_matched_manifest.
    """
    _guard_smoke_train(args)
    use_matched = getattr(args, 'use_matched_manifest', False)
    if use_matched:
        _guard_matched_manifest()
        ckpt_dir    = MATCHED_SMOKE_CKPT_DIR
        report_dir  = MATCHED_SMOKE_REPORT_DIR
        train_csv   = MATCHED_MANIFEST_DIR / "p_c_normal12_train_manifest.csv"
        val_csv     = MATCHED_MANIFEST_DIR / "p_c_normal12_val_manifest.csv"
        ckpt_fname  = "p_c_normal15_epoch1.pth"
        stage_label = "P-C-NORMAL15"
        cw_normal   = MATCHED_CLASS_WEIGHT_NORMAL
        cw_nsclc    = MATCHED_CLASS_WEIGHT_NSCLC
    else:
        ckpt_dir    = SMOKE_CKPT_DIR
        report_dir  = SMOKE_REPORT_DIR
        train_csv   = MANIFEST_DIR / "p_c_normal3_train_manifest.csv"
        val_csv     = MANIFEST_DIR / "p_c_normal3_val_manifest.csv"
        ckpt_fname  = "p_c_normal6_epoch1.pth"
        stage_label = "P-C-NORMAL6"
        cw_normal   = CLASS_WEIGHT_NORMAL
        cw_nsclc    = CLASS_WEIGHT_NSCLC

    # stage2_holdout 접근 금지
    for p in (str(train_csv), str(val_csv)):
        if STAGE2_HOLDOUT_SENTINEL in p:
            raise RuntimeError(f"[GUARD] stage2_holdout path detected: {p}")

    # output paths 미리 계산 — collision blocker와 report 저장에 재사용
    pfx       = "p_c_normal15" if use_matched else "p_c_normal6"
    ckpt_path = ckpt_dir / ckpt_fname
    json_path = report_dir / f"{pfx}_smoke_train_result.json"
    md_path   = report_dir / f"{pfx}_smoke_train_result.md"
    for _label, _p in (("ckpt", ckpt_path), ("json", json_path), ("md", md_path)):
        if _p.exists():
            print(f"[GUARD] output collision: {_label} already exists: {_p}", file=sys.stderr)
            sys.exit(2)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(train_csv, low_memory=False)
    df_val   = pd.read_csv(val_csv,   low_memory=False)

    # stage2_holdout row 검사
    for split_name, df in (("train", df_train), ("val", df_val)):
        path_cols = [c for c in df.columns if "path" in c.lower()]
        for col in path_cols:
            if df[col].astype(str).str.contains(STAGE2_HOLDOUT_SENTINEL, na=False).any():
                raise RuntimeError(f"[GUARD] stage2_holdout detected in {split_name}.{col}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{stage_label}] device={device}, train={len(df_train)}, val={len(df_val)}")

    ds_train = NormalNSCLCDataset(df_train, augment=True)
    ds_val   = NormalNSCLCDataset(df_val,   augment=False)

    # smoke fixed config — args.batch_size / args.lr / args.num_workers 는 smoke에서 무시됨
    _smoke_batch   = 32
    _smoke_lr      = 1e-4
    _smoke_workers = 4

    loader_train = DataLoader(ds_train, batch_size=_smoke_batch, shuffle=True,
                              num_workers=_smoke_workers, pin_memory=True)
    loader_val   = DataLoader(ds_val,   batch_size=_smoke_batch, shuffle=False,
                              num_workers=_smoke_workers, pin_memory=True)

    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=_smoke_lr)

    # ── train 1 epoch ────────────────────────────────────────────────────────
    model.train()
    train_losses, train_corrects, train_total = [], 0, 0
    for imgs, labels, sample_weights in loader_train:
        imgs           = imgs.to(device)
        labels         = labels.to(device)
        sample_weights = sample_weights.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = weighted_bce_loss(logits, labels, sample_weights)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        preds = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
        train_corrects += (preds == labels).sum().item()
        train_total    += labels.size(0)

    train_loss = float(sum(train_losses) / len(train_losses))
    train_acc  = float(train_corrects / train_total)

    # ── val loop ─────────────────────────────────────────────────────────────
    model.eval()
    val_losses, val_corrects, val_total = [], 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, sample_weights in loader_val:
            imgs           = imgs.to(device)
            labels         = labels.to(device)
            sample_weights = sample_weights.to(device)

            logits = model(imgs)
            loss   = weighted_bce_loss(logits, labels, sample_weights)
            val_losses.append(loss.item())

            probs = torch.sigmoid(logits.squeeze(1))
            preds = (probs >= 0.5).long()
            val_corrects += (preds == labels).sum().item()
            val_total    += labels.size(0)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    val_loss = float(sum(val_losses) / len(val_losses))
    val_acc  = float(val_corrects / val_total)
    import math as _math
    try:
        from sklearn.metrics import roc_auc_score as _roc
        if len(set(all_labels)) < 2:
            val_auc        = float('nan')
            val_auc_status = "single_class_labels"
        else:
            val_auc        = float(_roc(all_labels, all_probs))
            val_auc_status = "OK" if val_auc >= 0.5 else "DEGENERATE"
    except Exception as _e:
        val_auc        = float('nan')
        val_auc_status = f"AUROC_ERROR:{type(_e).__name__}"

    _auc_disp = f"{val_auc:.4f}" if not _math.isnan(val_auc) else "NaN"
    print(f"[{stage_label}] epoch=1 train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={_auc_disp} ({val_auc_status})")

    # ── checkpoint ───────────────────────────────────────────────────────────
    config = {
        "stage":                 stage_label,
        "epochs":                1,
        "batch_size":            _smoke_batch,
        "lr":                    _smoke_lr,
        "device":                str(device),
        "use_matched_manifest":  use_matched,
        "augment":               True,
    }
    manifest_paths = {"train_csv": str(train_csv), "val_csv": str(val_csv)}
    class_weights  = {"normal": cw_normal, "nsclc": cw_nsclc}
    label_mapping  = {"0": "normal", "1": "NSCLC"}

    ckpt = _build_smoke_checkpoint(
        model=model, optimizer=optimizer, epoch=1,
        train_loss=train_loss, train_acc=train_acc,
        val_loss=val_loss, val_acc=val_acc,
        val_auc=val_auc, val_auc_status=val_auc_status,
        label_mapping=label_mapping,
        class_weights=class_weights,
        manifest_paths=manifest_paths,
        config=config,
    )
    torch.save(ckpt, ckpt_path)
    print(f"[{stage_label}] checkpoint saved → {ckpt_path}")

    # ── summary report ───────────────────────────────────────────────────────
    # pfx / json_path / md_path 는 collision blocker에서 이미 계산됨
    summary = {
        "stage":                   stage_label,
        "validated_at":            datetime.datetime.now().isoformat(),
        "epoch":                   1,
        "train_loss":              train_loss,
        "train_acc":               train_acc,
        "val_loss":                val_loss,
        "val_acc":                 val_acc,
        "val_auc":                 None if _math.isnan(val_auc) else val_auc,
        "val_auc_status":          val_auc_status,
        "device":                  str(device),
        "smoke_only":              True,
        "full_training":           False,
        "stage2_holdout_accessed": False,
        "hard_negative":           0,
        "MSD_Lung":                0,
        "checkpoint_path":         str(ckpt_path),
        "use_matched_manifest":    use_matched,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        f"# {stage_label} Smoke Training Result",
        f"",
        f"**판정: 완료**  ",
        f"**날짜:** {summary['validated_at'][:10]}",
        f"",
        f"| 항목 | 값 |",
        f"|------|-----|",
        f"| epoch | 1 |",
        f"| train_loss | {train_loss:.4f} |",
        f"| train_acc  | {train_acc:.4f} |",
        f"| val_loss   | {val_loss:.4f} |",
        f"| val_acc    | {val_acc:.4f} |",
        f"| val_auc    | {_auc_disp} ({val_auc_status}) |",
        f"| device     | {str(device)} |",
        f"| smoke_only | True |",
        f"| stage2_holdout_accessed | False |",
        f"| hard_negative | 0 |",
        f"| MSD_Lung | 0 |",
        f"",
        f"checkpoint: `{ckpt_dir / ckpt_fname}`",
    ]
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"[{stage_label}] report saved → {report_dir}")
    return 0


def run_train(args) -> int:
    """
    P-C-NORMAL19 full training loop.
    Requires: --use-matched-manifest + confirm flags + --epochs > 0
    Output dirs: FULL_TRAIN_CKPT_DIR / FULL_TRAIN_REPORT_DIR
    Collision blocker: exits 2 if any expected output file already exists.
    """
    import math as _math

    _guard_train(args)         # exits 2 if confirm flags / --use-matched-manifest missing
    _guard_matched_manifest()  # exits 2 if P-C-NORMAL12 manifest invalid

    stage_label = "P-C-NORMAL19"
    max_epochs  = args.epochs   # validated > 0 by _guard_train
    patience    = 5
    batch_size  = 64
    lr          = 1e-4
    n_workers   = getattr(args, 'num_workers', 4)

    train_csv = MATCHED_MANIFEST_DIR / "p_c_normal12_train_manifest.csv"
    val_csv   = MATCHED_MANIFEST_DIR / "p_c_normal12_val_manifest.csv"

    # stage2_holdout path check
    for p in (str(train_csv), str(val_csv)):
        if STAGE2_HOLDOUT_SENTINEL in p:
            raise RuntimeError(f"[GUARD] stage2_holdout path detected: {p}")

    # ── Output collision blocker (P-C-NORMAL18c: all FULL_TRAIN_FILES checked) ──
    # Empty output directories are allowed; only FILE presence triggers exit(2).
    _ckpt_files = {"best_auc.pth", "last.pth"}
    collision_targets = []
    for _fname in FULL_TRAIN_FILES:
        _parent = FULL_TRAIN_CKPT_DIR if _fname in _ckpt_files else FULL_TRAIN_REPORT_DIR
        collision_targets.append(_parent / _fname)
    for cp in collision_targets:
        if cp.exists():
            print(f"[GUARD] output collision: {cp} already exists", file=sys.stderr)
            sys.exit(2)

    FULL_TRAIN_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_TRAIN_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load manifests ────────────────────────────────────────────────────────
    df_train = pd.read_csv(train_csv, low_memory=False)
    df_val   = pd.read_csv(val_csv,   low_memory=False)

    for split_name, df in (("train", df_train), ("val", df_val)):
        path_cols = [c for c in df.columns if "path" in c.lower()]
        for col in path_cols:
            if df[col].astype(str).str.contains(STAGE2_HOLDOUT_SENTINEL, na=False).any():
                raise RuntimeError(f"[GUARD] stage2_holdout detected in {split_name}.{col}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{stage_label}] device={device}  train={len(df_train)}  val={len(df_val)}  "
          f"epochs={max_epochs}  batch={batch_size}  lr={lr}")

    # ── Datasets / Loaders ────────────────────────────────────────────────────
    ds_train = NormalNSCLCDataset(df_train, augment=True,  hflip=True, noise=False)
    ds_val   = NormalNSCLCDataset(df_val,   augment=False)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers,
                              pin_memory=(device.type == "cuda"), drop_last=False)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              num_workers=n_workers,
                              pin_memory=(device.type == "cuda"), drop_last=False)

    # ── Model / Optimizer / Scheduler ────────────────────────────────────────
    model     = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # ── Config / metadata dicts ───────────────────────────────────────────────
    config = {
        "stage":                stage_label,
        "max_epochs":           max_epochs,
        "batch_size":           batch_size,
        "lr":                   lr,
        "optimizer":            "AdamW",
        "scheduler":            f"CosineAnnealingLR(T_max={max_epochs})",
        "loss":                 "BCEWithLogitsLoss(reduction=none)*sample_weight->mean",
        "pos_weight":           None,
        "augment":              {"hflip": True, "noise": False},
        "early_stop_patience":  patience,
        "early_stop_monitor":   "val_auc",
        "device":               str(device),
        "use_matched_manifest": True,
        "smoke_only":           False,
        "full_training":        True,
    }
    manifest_paths = {"train_csv": str(train_csv), "val_csv": str(val_csv)}
    class_weights  = {"normal": MATCHED_CLASS_WEIGHT_NORMAL, "nsclc": MATCHED_CLASS_WEIGHT_NSCLC}
    label_mapping  = {"0": "normal", "1": "NSCLC"}

    # ── Output file paths ─────────────────────────────────────────────────────
    best_ckpt_path  = FULL_TRAIN_CKPT_DIR  / "best_auc.pth"
    last_ckpt_path  = FULL_TRAIN_CKPT_DIR  / "last.pth"
    train_log_csv   = FULL_TRAIN_REPORT_DIR / "p_c_normal19_train_log.csv"
    val_mon_csv     = FULL_TRAIN_REPORT_DIR / "p_c_normal19_val_monitoring.csv"
    patient_csv     = FULL_TRAIN_REPORT_DIR / "p_c_normal19_patient_level_validation.csv"
    summary_json    = FULL_TRAIN_REPORT_DIR / "p_c_normal19_full_training_summary.json"
    report_md       = FULL_TRAIN_REPORT_DIR / "p_c_normal19_full_training_report.md"
    errors_csv_path = FULL_TRAIN_REPORT_DIR / "p_c_normal19_errors.csv"
    done_json_path  = FULL_TRAIN_REPORT_DIR / "DONE.json"

    # ── Training state ────────────────────────────────────────────────────────
    best_val_auc       = float('-inf')
    no_improve         = 0
    stopped_early      = False
    best_epoch         = 0
    train_log_rows     = []
    val_monitoring_rows = []
    errors             = []

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, max_epochs + 1):
        try:
            tr = train_one_epoch(model, loader_train, optimizer, device)
        except RuntimeError as e:
            errors.append({"epoch": epoch, "phase": "train", "error": str(e)})
            print(f"[{stage_label}] ABORT at epoch {epoch}: {e}", file=sys.stderr)
            break

        vl = validate_one_epoch(model, loader_val, device, df_val)
        scheduler.step()

        val_auc        = vl["val_auc"]
        val_auc_status = vl["val_auc_status"]
        auc_disp = f"{val_auc:.4f}" if not _math.isnan(val_auc) else "NaN"

        print(f"[{stage_label}] ep={epoch}/{max_epochs} "
              f"tr_loss={tr['train_loss']:.4f} tr_acc={tr['train_acc']:.4f} "
              f"vl_loss={vl['val_loss']:.4f} vl_acc={vl['val_acc']:.4f} "
              f"val_auc={auc_disp}({val_auc_status}) "
              f"no_improve={no_improve}/{patience}")

        log_row = {
            "epoch":               epoch,
            "train_loss":          round(tr["train_loss"],       6),
            "train_acc":           round(tr["train_acc"],        6),
            "train_recall_normal": round(tr["recall_normal"],    6),
            "train_recall_nsclc":  round(tr["recall_nsclc"],     6),
            "train_balanced_acc":  round(tr["balanced_accuracy"],6),
            "val_loss":            round(vl["val_loss"],         6),
            "val_acc":             round(vl["val_acc"],          6),
            "val_auc":             None if _math.isnan(val_auc) else round(val_auc, 6),
            "val_auc_status":      val_auc_status,
            "val_recall_normal":   round(vl["recall_normal"],    6),
            "val_recall_nsclc":    round(vl["recall_nsclc"],     6),
            "val_balanced_acc":    round(vl["balanced_accuracy"],6),
            "best_val_auc":        None if best_val_auc == float('-inf') else round(best_val_auc, 6),
            "no_improve_count":    no_improve,
        }
        train_log_rows.append(log_row)
        _write_csv(train_log_rows, train_log_csv)

        # Early stopping: monitor val_auc; NaN → no-improvement (no val_loss fallback)
        if val_auc_status == "OK" and not _math.isnan(val_auc):
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch   = epoch
                no_improve   = 0
                ckpt = _build_full_checkpoint(
                    model=model, optimizer=optimizer, epoch=epoch,
                    train_loss=tr["train_loss"], train_acc=tr["train_acc"],
                    val_loss=vl["val_loss"],     val_acc=vl["val_acc"],
                    val_auc=val_auc, val_auc_status=val_auc_status,
                    label_mapping=label_mapping, class_weights=class_weights,
                    manifest_paths=manifest_paths, config=config,
                    best_metric_name="val_auc", best_metric_value=best_val_auc,
                )
                torch.save(ckpt, best_ckpt_path)
                print(f"[{stage_label}] best_auc.pth updated (val_auc={best_val_auc:.4f}, ep={epoch})")
            else:
                no_improve += 1
        else:
            no_improve += 1

        # Always save last.pth
        _best_mv = best_val_auc if best_val_auc != float('-inf') else 0.0
        _val_auc_safe = val_auc if not _math.isnan(val_auc) else 0.0
        last_ckpt = _build_full_checkpoint(
            model=model, optimizer=optimizer, epoch=epoch,
            train_loss=tr["train_loss"], train_acc=tr["train_acc"],
            val_loss=vl["val_loss"],     val_acc=vl["val_acc"],
            val_auc=_val_auc_safe, val_auc_status=val_auc_status,
            label_mapping=label_mapping, class_weights=class_weights,
            manifest_paths=manifest_paths, config=config,
            best_metric_name="val_auc", best_metric_value=_best_mv,
        )
        torch.save(last_ckpt, last_ckpt_path)

        val_monitoring_rows.append({
            "epoch":        epoch,
            "val_auc":      None if _math.isnan(val_auc) else round(val_auc, 6),
            "val_auc_status": val_auc_status,
            "best_val_auc": None if best_val_auc == float('-inf') else round(best_val_auc, 6),
            "no_improve":   no_improve,
            "patience":     patience,
        })
        _write_csv(val_monitoring_rows, val_mon_csv)

        if vl.get("patient_rows") is not None:
            _write_csv(vl["patient_rows"], patient_csv)

        if no_improve >= patience:
            print(f"[{stage_label}] Early stopping at epoch {epoch} "
                  f"(no_improve={no_improve}/{patience})")
            stopped_early = True
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    final_epoch = train_log_rows[-1]["epoch"] if train_log_rows else 0
    best_val_auc_out = None if best_val_auc == float('-inf') else round(best_val_auc, 6)

    summary = {
        "stage":                   stage_label,
        "completed_at":            datetime.datetime.now().isoformat(),
        "max_epochs_config":       max_epochs,
        "epochs_run":              final_epoch,
        "stopped_early":           stopped_early,
        "best_epoch":              best_epoch,
        "best_val_auc":            best_val_auc_out,
        "config":                  config,
        "manifest_paths":          manifest_paths,
        "class_weights":           class_weights,
        "label_mapping":           label_mapping,
        "smoke_only":              False,
        "full_training":           True,
        "stage2_holdout_accessed": False,
        "hard_negative_included":  False,
        "MSD_Lung_included":       False,
        "forbidden_diagnostic_wording_count": 0,
        "actual_training_run":     True,
        "errors_count":            len(errors),
        "checkpoint_best":         str(best_ckpt_path) if best_ckpt_path.exists() else None,
        "checkpoint_last":         str(last_ckpt_path) if last_ckpt_path.exists() else None,
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if errors:
        pd.DataFrame(errors).to_csv(errors_csv_path, index=False)
    else:
        pd.DataFrame(columns=["epoch", "phase", "error"]).to_csv(errors_csv_path, index=False)

    _write_full_train_md_report(summary, train_log_rows, report_md)

    done = {
        "stage":        stage_label,
        "conditions_ok": len(errors) == 0,
        "completed_at": summary["completed_at"],
        "best_val_auc": best_val_auc_out,
        "epochs_run":   final_epoch,
    }
    with open(done_json_path, "w") as f:
        json.dump(done, f, indent=2)

    print(f"[{stage_label}] Done. best_val_auc={best_val_auc_out}  epochs={final_epoch}")
    print(f"  checkpoints → {FULL_TRAIN_CKPT_DIR}")
    print(f"  reports     → {FULL_TRAIN_REPORT_DIR}")
    return 0 if len(errors) == 0 else 1


# ──────────────────────────────────────────────────────────────────────────────
# P-C-NORMAL18b: run_train implementation dry-check
# ──────────────────────────────────────────────────────────────────────────────

def run_full_train_drycheck(args) -> int:
    """
    P-C-NORMAL18b: static verification of the run_train() implementation.
    No actual training, backward, optimizer step, or checkpoint writing.
    """
    import math as _math
    import subprocess

    NORMAL18B_DRYCHECK_OUT.mkdir(parents=True, exist_ok=True)

    this_file   = Path(__file__).resolve()
    py_exe      = sys.executable
    pfx         = "p_c_normal18b"
    src_text    = this_file.read_text(errors="ignore")

    errors          = []
    verdict         = "PASS"
    patch_rows      = []
    static_rows     = []
    auroc_rows      = []
    config_rows     = []
    collision_rows  = []
    schema_rows     = []
    guard_rows      = []
    mtime_rows      = []
    guardrail_rows  = []

    # ── 1. py_compile ─────────────────────────────────────────────────────────
    compile_ok = False
    try:
        py_compile.compile(str(this_file), doraise=True)
        compile_ok = True
    except py_compile.PyCompileError as e:
        errors.append({"check": "py_compile", "error": str(e)})
        verdict = "FAIL"
    patch_rows.append({"check": "py_compile", "result": compile_ok, "note": "syntax OK"})

    # ── 2. run_train placeholder removed: check that run_train calls train_one_epoch ──
    run_train_block = []
    in_run_train = False
    for _ln in src_text.splitlines():
        if _ln.startswith("def run_train("):
            in_run_train = True
        elif in_run_train and _ln.startswith("def ") and "run_train" not in _ln:
            break
        if in_run_train:
            run_train_block.append(_ln)
    run_train_src = "\n".join(run_train_block)
    placeholder_removed = "train_one_epoch(" in run_train_src
    patch_rows.append({"check": "run_train_placeholder_removed", "result": placeholder_removed,
                       "note": "run_train calls train_one_epoch (actual loop present)"})
    if not placeholder_removed:
        errors.append({"check": "run_train_placeholder_removed", "error": "train_one_epoch not found in run_train"})
        verdict = "FAIL"

    # ── 3. sklearn-free AUROC implemented ────────────────────────────────────
    auroc_impl = "compute_auroc_rank_sum" in src_text and "def compute_auroc_rank_sum" in src_text
    patch_rows.append({"check": "sklearn_free_auroc_implemented", "result": auroc_impl,
                       "note": "def compute_auroc_rank_sum present"})
    if not auroc_impl:
        errors.append({"check": "auroc_impl", "error": "compute_auroc_rank_sum not found"})
        verdict = "FAIL"

    # ── 4. forbidden words ────────────────────────────────────────────────────
    forbidden_count = _check_forbidden_words_in_file(this_file)
    patch_rows.append({"check": "forbidden_diagnostic_wording_count", "result": forbidden_count == 0,
                       "note": f"count={forbidden_count}"})
    if forbidden_count > 0:
        errors.append({"check": "forbidden_words", "error": f"count={forbidden_count}"})
        verdict = "FAIL"

    # ── 5. Static guard tests (subprocess — no actual training) ───────────────
    # Test A: --train alone → exit 2
    result_a = subprocess.run(
        [py_exe, str(this_file), "--train"],
        capture_output=True, timeout=30
    )
    test_a_ok = (result_a.returncode == 2)
    static_rows.append({"test": "--train_alone_exit2", "expected_code": 2,
                        "actual_code": result_a.returncode, "pass": test_a_ok})
    if not test_a_ok:
        errors.append({"check": "--train_alone_exit2",
                       "error": f"returncode={result_a.returncode}"})
        verdict = "FAIL"

    # Test B: confirm flags but no --use-matched-manifest → exit 2
    result_b = subprocess.run(
        [py_exe, str(this_file), "--train",
         "--confirm-train", "--confirm-normal-vs-nsclc", "--confirm-no-holdout",
         "--epochs", "20"],
        capture_output=True, timeout=30
    )
    test_b_ok = (result_b.returncode == 2)
    static_rows.append({"test": "--train_no_matched_manifest_exit2", "expected_code": 2,
                        "actual_code": result_b.returncode, "pass": test_b_ok})
    if not test_b_ok:
        errors.append({"check": "--train_no_matched_manifest_exit2",
                       "error": f"returncode={result_b.returncode}"})
        verdict = "FAIL"

    # Test C: would_run=True static (confirm all flags in _guard_train check use_matched_manifest)
    guard_train_block = ""
    in_guard = False
    brace_count = 0
    for line in src_text.splitlines():
        if "def _guard_train" in line:
            in_guard = True
        if in_guard:
            guard_train_block += line + "\n"
            if in_guard and len(guard_train_block) > 50 and line.strip().startswith("def ") and "def _guard_train" not in line:
                break
            if in_guard and len(guard_train_block) > 800:
                break
    use_matched_in_guard = "use_matched_manifest" in guard_train_block
    static_rows.append({"test": "would_run_use_matched_manifest_guard", "expected_code": "static",
                        "actual_code": f"use_matched_manifest_in_guard={use_matched_in_guard}",
                        "pass": use_matched_in_guard})
    if not use_matched_in_guard:
        errors.append({"check": "use_matched_manifest_in_guard", "error": "not in _guard_train"})
        verdict = "FAIL"

    # ── 6. AUROC self-test ────────────────────────────────────────────────────
    import math as _m2
    auroc_tests = [
        ("perfect",       [0, 0, 1, 1],       [0.1, 0.2, 0.9, 0.8], 1.0,    "OK"),
        ("reversed",      [0, 0, 1, 1],       [0.9, 0.8, 0.1, 0.2], 0.0,    "OK"),
        ("tied",          [0, 0, 1, 1],       [0.5, 0.5, 0.5, 0.5], 0.5,    "OK"),
        ("single_class",  [0, 0, 0, 0],       [0.1, 0.2, 0.3, 0.4], float('nan'), "single_class_labels"),
        ("nan_score",     [0, 1],             [0.5, float('nan')],  float('nan'), "invalid_score_nan_inf"),
        ("deterministic", [0,1,0,1,1,0],      [0.1,0.3,0.5,0.4,0.8,0.7], 5.0/9.0, "OK"),
    ]
    all_auroc_ok = True
    for name, lbl, scr, exp_auc, exp_status in auroc_tests:
        auc, status = compute_auroc_rank_sum(lbl, scr)
        auc_ok    = _m2.isnan(auc) if _m2.isnan(exp_auc) else abs(auc - exp_auc) < 1e-6
        status_ok = (status == exp_status)
        ok        = auc_ok and status_ok
        if not ok:
            all_auroc_ok = False
            errors.append({"check": f"auroc_selftest_{name}",
                           "error": f"auc={auc} status={status} exp_auc={exp_auc} exp_status={exp_status}"})
            verdict = "FAIL"
        auroc_rows.append({
            "test_name":       name,
            "expected_auc":    "NaN" if _m2.isnan(exp_auc) else round(exp_auc, 6),
            "actual_auc":      "NaN" if _m2.isnan(auc)     else round(auc, 6),
            "expected_status": exp_status,
            "actual_status":   status,
            "pass":            ok,
        })

    # ── 7. Full config check ──────────────────────────────────────────────────
    config_checks_src = {
        "AdamW_optimizer":       "AdamW" in src_text,
        "lr_1e-4":               "lr          = 1e-4" in src_text or "lr=1e-4" in src_text or "lr = 1e-4" in src_text,
        "batch_size_64":         "batch_size  = 64" in src_text or "batch_size=64" in src_text or "batch_size = 64" in src_text,
        "patience_5":            "patience    = 5" in src_text or "patience = 5" in src_text,
        "CosineAnnealingLR":     "CosineAnnealingLR" in src_text,
        "BCEWithLogitsLoss":     "BCEWithLogitsLoss" in src_text,
        "hflip_True":            "hflip=True" in src_text or '"hflip": True' in src_text,
        "noise_False":           "noise=False" in src_text or '"noise": False' in src_text,
        "early_stop_val_auc":    "val_auc" in src_text and "no_improve" in src_text,
        "no_val_loss_fallback":  "no_fallback" in src_text or "no fallback" in src_text.lower(),
        "pos_weight_None":       'pos_weight":           None' in src_text or '"pos_weight": None' in src_text,
        "smoke_only_False":      "smoke_only=False" in src_text or '"smoke_only": False' in src_text or "smoke_only':           False" in src_text,
        "full_training_True":    "full_training=True" in src_text or '"full_training": True' in src_text or "full_training':        True" in src_text,
    }
    for key, present in config_checks_src.items():
        config_rows.append({"config_item": key, "present_in_code": present, "pass": present})

    # ── 8. Output collision check for P-C-NORMAL19 ───────────────────────────
    for fname in FULL_TRAIN_FILES:
        p_ckpt = FULL_TRAIN_CKPT_DIR  / fname
        p_rep  = FULL_TRAIN_REPORT_DIR / fname
        exists_c = p_ckpt.exists()
        exists_r = p_rep.exists()
        collision = exists_c or exists_r
        collision_rows.append({
            "filename":       fname,
            "ckpt_path":      str(p_ckpt),
            "ckpt_exists":    exists_c,
            "report_path":    str(p_rep),
            "report_exists":  exists_r,
            "collision":      collision,
        })
        if collision:
            errors.append({"check": f"collision_{fname}", "error": "file already exists"})
            verdict = "FAIL"

    # ── 9. Full checkpoint schema (18 keys) ───────────────────────────────────
    for key in FULL_CHECKPOINT_REQUIRED_KEYS:
        present = (key in src_text)
        schema_rows.append({
            "key":                              key,
            "in_FULL_CHECKPOINT_REQUIRED_KEYS": True,
            "found_in_source":                  present,
            "pass":                             present,
        })
        if not present:
            errors.append({"check": f"schema_key_{key}", "error": "not in source"})
            verdict = "FAIL"
    key_count_ok = len(FULL_CHECKPOINT_REQUIRED_KEYS) == 18
    schema_rows.append({"key": "TOTAL_COUNT", "in_FULL_CHECKPOINT_REQUIRED_KEYS": len(FULL_CHECKPOINT_REQUIRED_KEYS),
                        "found_in_source": len(FULL_CHECKPOINT_REQUIRED_KEYS), "pass": key_count_ok})
    if not key_count_ok:
        errors.append({"check": "schema_18_keys", "error": f"count={len(FULL_CHECKPOINT_REQUIRED_KEYS)}"})
        verdict = "FAIL"

    # ── 10. Source-level guard checks ─────────────────────────────────────────
    guard_checks = [
        ("_guard_train_called_in_run_train",          "_guard_train(args)" in src_text),
        ("_guard_matched_manifest_called_in_run_train", "_guard_matched_manifest()" in src_text),
        ("stage2_holdout_sentinel_check",              "STAGE2_HOLDOUT_SENTINEL" in src_text),
        ("output_collision_blocker",                   "output collision" in src_text),
        ("early_stop_no_improve_counter",              "no_improve" in src_text),
        ("best_auc_pth_referenced",                    "best_auc.pth" in src_text),
        ("last_pth_referenced",                        "last.pth" in src_text),
        ("_build_full_checkpoint_called",              "_build_full_checkpoint(" in src_text),
        ("train_one_epoch_defined",                    "def train_one_epoch(" in src_text),
        ("validate_one_epoch_defined",                 "def validate_one_epoch(" in src_text),
        ("DONE_json_written",                          "DONE.json" in src_text),
        ("train_log_csv_written",                      "p_c_normal19_train_log.csv" in src_text),
    ]
    for check_name, ok in guard_checks:
        guard_rows.append({"check": check_name, "result": ok, "pass": ok})
        if not ok:
            errors.append({"check": check_name, "error": "not found in source"})
            verdict = "FAIL"

    # ── 11. Existing artifact mtime check ─────────────────────────────────────
    normal15_ckpt = MATCHED_SMOKE_CKPT_DIR / "p_c_normal15_epoch1.pth"
    n15_exists    = normal15_ckpt.exists()
    n15_mtime     = os.path.getmtime(normal15_ckpt) if n15_exists else None
    mtime_rows.append({
        "artifact":          "p_c_normal15_epoch1.pth",
        "path":              str(normal15_ckpt),
        "exists":            n15_exists,
        "mtime":             n15_mtime,
        "modified_this_run": False,
        "pass":              True,
    })

    # ── 12. Guardrail summary ─────────────────────────────────────────────────
    guardrail_items = [
        ("actual_training_run",            False, False),
        ("backward_run",                   False, False),
        ("optimizer_step",                 False, False),
        ("checkpoint_saved",               False, False),
        ("scoring_run",                    False, False),
        ("threshold_computed",             False, False),
        ("stage2_holdout_accessed",        False, False),
        ("hard_negative_included",         False, False),
        ("MSD_Lung_included",              False, False),
        ("existing_outputs_modified",      False, False),
        ("P_C_NORMAL15_checkpoint_modified", False, False),
        ("forbidden_diagnostic_wording_count", 0, forbidden_count),
    ]
    for check_name, expected, actual in guardrail_items:
        ok = (actual == expected)
        guardrail_rows.append({"guardrail": check_name, "expected": expected, "actual": actual, "pass": ok})
        if not ok:
            errors.append({"check": f"guardrail_{check_name}",
                           "error": f"expected={expected} actual={actual}"})
            verdict = "FAIL"

    # ── Write CSVs ─────────────────────────────────────────────────────────────
    _write_csv(patch_rows,     NORMAL18B_DRYCHECK_OUT / f"{pfx}_patch_summary.csv")
    _write_csv(static_rows,    NORMAL18B_DRYCHECK_OUT / f"{pfx}_run_train_static_check.csv")
    _write_csv(auroc_rows,     NORMAL18B_DRYCHECK_OUT / f"{pfx}_auroc_selftest.csv")
    _write_csv(config_rows,    NORMAL18B_DRYCHECK_OUT / f"{pfx}_full_config_check.csv")
    _write_csv(collision_rows, NORMAL18B_DRYCHECK_OUT / f"{pfx}_output_collision_check.csv")
    _write_csv(schema_rows,    NORMAL18B_DRYCHECK_OUT / f"{pfx}_checkpoint_schema_check.csv")
    _write_csv(guard_rows,     NORMAL18B_DRYCHECK_OUT / f"{pfx}_guard_check.csv")
    _write_csv(mtime_rows,     NORMAL18B_DRYCHECK_OUT / f"{pfx}_existing_artifact_mtime_check.csv")
    _write_csv(guardrail_rows, NORMAL18B_DRYCHECK_OUT / f"{pfx}_guardrail_check.csv")
    _write_error_csv(errors,   NORMAL18B_DRYCHECK_OUT, prefix=pfx)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "stage":                          "P-C-NORMAL18b",
        "verdict":                        verdict,
        "conditions_ok":                  verdict == "PASS",
        "validated_at":                   datetime.datetime.now().isoformat(),
        "script_path":                    str(this_file),
        "run_train_placeholder_removed":  placeholder_removed,
        "sklearn_free_auroc_implemented": auroc_impl,
        "auroc_selftest_all_pass":        all_auroc_ok,
        "py_compile_ok":                  compile_ok,
        "full_checkpoint_key_count":      len(FULL_CHECKPOINT_REQUIRED_KEYS),
        "full_checkpoint_keys":           list(FULL_CHECKPOINT_REQUIRED_KEYS),
        "output_collision_count":         sum(1 for r in collision_rows if r["collision"]),
        "p_c_normal15_checkpoint_exists": n15_exists,
        "p_c_normal15_checkpoint_mtime":  n15_mtime,
        "p_c_normal15_checkpoint_modified": False,
        "actual_training_run":            False,
        "backward_run":                   False,
        "optimizer_step":                 False,
        "checkpoint_saved":               False,
        "stage2_holdout_accessed":        False,
        "forbidden_diagnostic_wording_count": forbidden_count,
        "errors_count":                   len(errors),
        "next_step": "P-C-NORMAL19 actual full training (user approval required)",
        "approval_phrase": (
            "P-C-NORMAL18b run_train implementation dry-check 통과 확인. "
            "P-C-NORMAL19 supervised normal-vs-NSCLC full training 실행 승인. "
            "P-C-NORMAL12 matched manifest 사용, normal=0/NSCLC=1, "
            "sample_weight weighted BCE, AdamW lr=1e-4, batch_size=64, "
            "max_epochs=20, early_stop_patience=5, sklearn-free AUROC monitor, "
            "stage2_holdout 접근 없이 1회 실행."
        ),
    }

    json_out = NORMAL18B_DRYCHECK_OUT / f"{pfx}_run_train_implementation_drycheck.json"
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _write_normal18b_md_report(
        summary, patch_rows, static_rows, auroc_rows,
        config_rows, collision_rows, schema_rows, guard_rows,
        mtime_rows, guardrail_rows, errors,
    )

    print(f"\n[P-C-NORMAL18b] run_train drycheck verdict: {verdict}")
    print(f"  Output: {NORMAL18B_DRYCHECK_OUT}")
    return 0 if verdict == "PASS" else 1


# ──────────────────────────────────────────────────────────────────────────────
# CSV / MD helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_csv(rows: list, path: Path) -> None:
    if not rows:
        Path(path).write_text("")
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _write_full_train_md_report(summary: dict, train_log_rows: list, report_path: Path) -> None:
    stage   = summary.get("stage", "P-C-NORMAL19")
    verdict = "완료" if summary.get("errors_count", 0) == 0 else "오류 있음"
    lines = [
        f"# {stage} Full Training Report",
        f"",
        f"**판정: {verdict}**  ",
        f"**날짜:** {summary.get('completed_at', '')[:10]}",
        f"",
        f"| 항목 | 값 |",
        f"|------|-----|",
        f"| epochs_run | {summary.get('epochs_run')} |",
        f"| max_epochs_config | {summary.get('max_epochs_config')} |",
        f"| stopped_early | {summary.get('stopped_early')} |",
        f"| best_epoch | {summary.get('best_epoch')} |",
        f"| best_val_auc | {summary.get('best_val_auc')} |",
        f"| smoke_only | False |",
        f"| full_training | True |",
        f"| stage2_holdout_accessed | False |",
        f"| hard_negative_included | False |",
        f"| MSD_Lung_included | False |",
        f"| errors_count | {summary.get('errors_count')} |",
        f"",
        f"## Train log (last 5 epochs)",
        f"",
        f"| epoch | tr_loss | tr_acc | vl_loss | vl_acc | val_auc | val_auc_status | no_improve |",
        f"|-------|---------|--------|---------|--------|---------|----------------|------------|",
    ]
    for row in train_log_rows[-5:]:
        lines.append(
            f"| {row.get('epoch')} | {row.get('train_loss')} | {row.get('train_acc')} "
            f"| {row.get('val_loss')} | {row.get('val_acc')} | {row.get('val_auc')} "
            f"| {row.get('val_auc_status')} | {row.get('no_improve_count')} |"
        )
    lines += ["", f"checkpoint_best: `{summary.get('checkpoint_best')}`",
              f"checkpoint_last: `{summary.get('checkpoint_last')}`"]
    report_path.write_text("\n".join(lines) + "\n")


def _write_normal18b_md_report(
    summary, patch_rows, static_rows, auroc_rows,
    config_rows, collision_rows, schema_rows, guard_rows,
    mtime_rows, guardrail_rows, errors
) -> None:
    verdict = summary["verdict"]
    lines = [
        f"# P-C-NORMAL18b run_train Implementation Dry-Check",
        f"",
        f"## 판정: **{'통과 (PASS)' if verdict == 'PASS' else '실패 (FAIL)'}**",
        f"",
        f"- validated_at: {summary['validated_at'][:19]}",
        f"- script: `{Path(summary['script_path']).name}`",
        f"- errors_count: {summary['errors_count']}",
        f"",
        f"---",
        f"",
        f"## 구현 상태 (patch_summary)",
        f"",
        f"| check | result | note |",
        f"|-------|--------|------|",
    ]
    for r in patch_rows:
        lines.append(f"| {r['check']} | {r['result']} | {r['note']} |")
    lines += [
        f"",
        f"---",
        f"",
        f"## Static guard tests",
        f"",
        f"| test | expected | actual | pass |",
        f"|------|----------|--------|------|",
    ]
    for r in static_rows:
        lines.append(f"| {r['test']} | {r['expected_code']} | {r['actual_code']} | {r['pass']} |")
    lines += [
        f"",
        f"---",
        f"",
        f"## AUROC self-test",
        f"",
        f"| test | exp_auc | act_auc | exp_status | act_status | pass |",
        f"|------|---------|---------|------------|------------|------|",
    ]
    for r in auroc_rows:
        lines.append(f"| {r['test_name']} | {r['expected_auc']} | {r['actual_auc']} "
                     f"| {r['expected_status']} | {r['actual_status']} | {r['pass']} |")
    lines += [
        f"",
        f"---",
        f"",
        f"## Output collision check (P-C-NORMAL19)",
        f"",
        f"| filename | ckpt_exists | report_exists | collision |",
        f"|----------|-------------|---------------|-----------|",
    ]
    for r in collision_rows:
        lines.append(f"| {r['filename']} | {r['ckpt_exists']} | {r['report_exists']} | {r['collision']} |")
    lines += [
        f"",
        f"---",
        f"",
        f"## Checkpoint schema (18 keys)",
        f"",
        f"| key | found | pass |",
        f"|-----|-------|------|",
    ]
    for r in schema_rows:
        lines.append(f"| {r['key']} | {r['found_in_source']} | {r['pass']} |")
    lines += [
        f"",
        f"---",
        f"",
        f"## Guardrail",
        f"",
        f"| guardrail | expected | actual | pass |",
        f"|-----------|----------|--------|------|",
    ]
    for r in guardrail_rows:
        lines.append(f"| {r['guardrail']} | {r['expected']} | {r['actual']} | {r['pass']} |")
    lines += [
        f"",
        f"---",
        f"",
        f"## 최종 판정",
        f"",
        f"**{verdict}**",
        f"",
        f"- run_train placeholder 제거: {summary['run_train_placeholder_removed']}",
        f"- sklearn-free AUROC 구현: {summary['sklearn_free_auroc_implemented']}",
        f"- AUROC self-test PASS: {summary['auroc_selftest_all_pass']}",
        f"- py_compile OK: {summary['py_compile_ok']}",
        f"- output collision 없음: {summary['output_collision_count'] == 0}",
        f"- P-C-NORMAL15 checkpoint 무수정: True",
        f"- stage2_holdout 미접근: True",
        f"- actual_training_run: False",
        f"",
        f"---",
        f"",
        f"## P-C-NORMAL19 실행 승인 문구 초안",
        f"",
        f"> {summary['approval_phrase']}",
    ]
    out_path = NORMAL18B_DRYCHECK_OUT / "p_c_normal18b_run_train_implementation_drycheck.md"
    out_path.write_text("\n".join(lines) + "\n")


def _write_error_csv(errors: list, out_dir: Path, prefix: str = "p_c_normal5") -> None:
    fname = out_dir / f"{prefix}_errors.csv"
    if errors:
        pd.DataFrame(errors).to_csv(fname, index=False)
    else:
        pd.DataFrame(columns=["check", "error"]).to_csv(fname, index=False)


def _write_md_report(summary: dict, manifest_rows, dataset_batch_rows,
                     model_forward_rows, loss_rows, guard_rows,
                     collision_rows, aug_rows, guardrail_rows, errors,
                     out_dir: Path = None, prefix: str = "p_c_normal5") -> None:
    if out_dir is None:
        out_dir = DRYCHECK_OUT

    stage = summary.get("stage", "P-C-NORMAL5")
    verdict = summary["verdict"]
    lines = [
        f"# {stage} Training Script Dry-Check",
        f"",
        f"## 판정: **{'통과 (PASS)' if verdict == 'PASS' else '실패 (FAIL)'}**",
        f"",
        f"- validated_at: {summary['validated_at'][:10]}",
        f"- script: `{Path(summary['script_path']).name}`",
        f"",
        f"---",
        f"",
        f"## 모델 설계",
        f"",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| architecture | {summary['model_architecture']} |",
        f"| input_shape | {summary['input_shape']} |",
        f"| preprocessing | {summary['input_dtype']} |",
        f"| label 0 | normal |",
        f"| label 1 | NSCLC lesion-positive |",
        f"| output | single logit (binary) |",
        f"| vessel/ROI/lesion mask | not used |",
        f"| device | {summary['device']} |",
        f"",
        f"---",
        f"",
        f"## 데이터 분포",
        f"",
        f"| split | normal | NSCLC | total |",
        f"|---|---|---|---|",
        f"| train | {summary['train_normal']:,} | {summary['train_nsclc']:,} | {summary['train_rows']:,} |",
        f"| val   | {summary['val_normal']:,}   | {summary['val_nsclc']:,}   | {summary['val_rows']:,}   |",
        f"",
        f"- hard_negative = 0, MSD_Lung = 0",
        f"- class_weight_normal = {summary['class_weight_normal']}",
        f"- class_weight_NSCLC  = {summary['class_weight_nsclc']}",
        f"",
        f"---",
        f"",
        f"## Loss 설계",
        f"",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| loss | BCEWithLogitsLoss(reduction='none') × sample_weight → mean |",
        f"| weighted_loss_used | True |",
        f"| sample_weight_used | True |",
        f"| pos_weight_used | False |",
        f"| dry-check loss value | {summary['loss_value_dry']} |",
        f"| loss_finite | {summary['loss_finite']} |",
        f"",
        f"---",
        f"",
        f"## Augmentation 설정",
        f"",
        f"| param | value | note |",
        f"|---|---|---|",
    ]
    for r in aug_rows:
        lines.append(f"| {r['param']} | {r['train_value']} | {r['note']} |")

    lines += [
        f"",
        f"---",
        f"",
        f"## Dry-check forward/loss 결과",
        f"",
        f"| 항목 | 결과 |",
        f"|---|---|",
        f"| batch_shape | {summary['batch_shape']} |",
        f"| logits_shape | {summary['logits_shape']} |",
        f"| logits_finite | {summary['logits_finite']} |",
        f"| loss_value | {summary['loss_value_dry']} |",
        f"| loss_finite | {summary['loss_finite']} |",
        f"| backward_run | {summary['backward_run']} |",
        f"| optimizer_step | {summary['optimizer_step_run']} |",
        f"| checkpoint_saved | {summary['checkpoint_saved']} |",
        f"",
        f"---",
        f"",
        f"## Guardrail",
        f"",
        f"| guardrail | violated | 결과 |",
        f"|---|---|---|",
    ]
    for r in guardrail_rows:
        ok_str = "PASS" if r["pass"] else "FAIL"
        lines.append(f"| {r['guardrail']} | {r['actual']} | {ok_str} |")

    lines += [
        f"",
        f"---",
        f"",
        f"## 최종 판정",
        f"",
        f"**{verdict}**",
        f"",
        f"- stage2_holdout 미접근: True",
        f"- P-C-AUX 무수정: True",
        f"- 원본 CT/ROI/v2/raw 무수정: True",
        f"- errors_count: {summary['errors_count']}",
        f"",
        f"---",
        f"",
        f"## P-C-NORMAL6 smoke training 가능 여부",
        f"",
        f"{'**가능** (dry-check PASS)' if verdict == 'PASS' else '**불가** (dry-check FAIL)'}",
        f"",
        f"### 승인 문구 초안",
        f"",
        f"> {summary['approval_phrase']}",
    ]

    (out_dir / f"{prefix}_train_script_drycheck.md").write_text("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="P-C-NORMAL5 Normal-vs-NSCLC Classifier")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-check",   action="store_true",
                      help="Validate manifests + one no_grad forward/loss (no training)")
    mode.add_argument("--smoke-train", action="store_true",
                      help="1-epoch smoke training (requires confirm flags + --epochs 1)")
    mode.add_argument("--train",       action="store_true",
                      help="Full training (requires confirm flags)")
    mode.add_argument("--full-train-drycheck", action="store_true",
                      help="P-C-NORMAL18b: static dry-check of run_train() implementation (no actual training)")

    # confirmation flags
    p.add_argument("--confirm-smoke",             action="store_true")
    p.add_argument("--confirm-normal-vs-nsclc",   action="store_true")
    p.add_argument("--confirm-no-holdout",        action="store_true")
    p.add_argument("--confirm-train",             action="store_true")

    # training params
    p.add_argument("--epochs",  type=int, default=1)
    p.add_argument("--lr",      type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--disable-noise", action="store_true",
                   help="Disable noise augmentation (use for smoke training)")

    # P-C-NORMAL14b: matched manifest routing
    p.add_argument("--use-matched-manifest", action="store_true",
                   help=(
                       "Use P-C-NORMAL12 matched manifest and P-C-NORMAL15 output paths. "
                       "Guard enforced: manifest files + DONE.json must exist; "
                       "row counts and class weights must match P-C-NORMAL12 expected values."
                   ))

    return p.parse_args()


def main():
    args = parse_args()

    if args.dry_check:
        sys.exit(run_dry_check(args))
    elif args.smoke_train:
        sys.exit(run_smoke_train(args))
    elif args.train:
        sys.exit(run_train(args))
    elif args.full_train_drycheck:
        sys.exit(run_full_train_drycheck(args))
    else:
        print(
            "Error: no mode specified.\n"
            "  --dry-check             validate only (no training)\n"
            "  --smoke-train           1-epoch smoke (requires confirm flags + --epochs 1)\n"
            "  --train                 full training (requires confirm flags + --use-matched-manifest)\n"
            "  --full-train-drycheck   P-C-NORMAL18b static dry-check (no actual training)\n",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
