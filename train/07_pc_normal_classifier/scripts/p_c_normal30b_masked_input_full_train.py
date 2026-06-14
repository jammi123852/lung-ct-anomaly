"""
p_c_normal30b_masked_input_full_train.py

P-C-NORMAL30b: masked-input full training (30 epochs)
기준: P-C-NORMAL30a smoke PASS → full training 진행

불변 조건 (24j-fix-balanced-w1 대비 masking만 추가):
  - train/val manifest: P-C-NORMAL24g-fix-balanced-w1
  - scalar features: lung_z_percentile, crop_lung_roi_ratio (24h-fix stats)
  - balanced-w1 setting (sample_weight=1.0)
  - BCEWithLogitsLoss, lr=1e-4, batch=64, epochs=30, patience=7
  - vessel feature 금지, ROI-masked loss 금지
  - final_test 접근 금지, threshold 최적화 금지

masking 방식:
  - masked_ct = ct_crop * mask_3ch (이미지에만 적용)
  - scalar feature, label, sample_weight 수정 없음

실행:
  python p_c_normal30b_masked_input_full_train.py \\
      --train \\
      --confirm-train \\
      --confirm-normal-vs-nsclc \\
      --confirm-no-final-test \\
      --epochs 30
"""

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_MANIFEST    = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_balanced_w1_feature_manifest/p_c_normal24g_fix_balanced_w1_train_manifest.csv"
VAL_MANIFEST      = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_balanced_w1_feature_manifest/p_c_normal24g_fix_balanced_w1_val_manifest.csv"
SCALAR_STATS_PATH = PROJECT_ROOT / "outputs/reports/p_c_normal24h_fix_zroi_scalar_fusion_drycheck/p_c_normal24h_fix_scalar_normalization_stats.json"
MASK_MANIFEST     = PROJECT_ROOT / "outputs/reports/p_c_normal29b_crop_level_mask_generation/p_c_normal29b_mask_manifest.csv"

OUTPUT_ROOT  = PROJECT_ROOT / "outputs/p_c_normal30b_masked_input_full_train"
CKPT_DIR     = OUTPUT_ROOT / "checkpoints"
REPORT_ROOT  = PROJECT_ROOT / "outputs/reports/p_c_normal30b_masked_input_full_train"

# ── Constants ─────────────────────────────────────────────────────────────────
HU_MIN, HU_MAX  = -1000.0, 200.0
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
SCALAR_FEATURES = ["lung_z_percentile", "crop_lung_roi_ratio"]

EXPECTED_TRAIN_ROWS = 15782
EXPECTED_VAL_ROWS   = 4160
SEED                = 42
STAGE_LABEL         = "P-C-NORMAL30b"

FORBIDDEN_VESSEL_COLS = {
    "vessel_candidate_ratio", "vessel_softmask_max", "vessel_center_ratio",
    "vessel_high_risk_ratio", "vessel_low_risk_ratio",
}

# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Helper writers ────────────────────────────────────────────────────────────

def _write_csv(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"note": "empty"}]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ── Scalar normalization ──────────────────────────────────────────────────────

def apply_scalar_norm(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    for col, s in stats.items():
        df[col] = (df[col].astype(float) - s["mean"]) / s["std"]
    return df


# ── AUROC (sklearn-free) ──────────────────────────────────────────────────────

def compute_auroc(labels, scores):
    try:
        labels_arr = np.asarray(labels, dtype=np.int32)
        scores_arr = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(scores_arr)):
            return float("nan"), "invalid_score_nan_inf"
        if len(np.unique(labels_arr)) < 2:
            return float("nan"), "single_class_labels"
        pos_mask = labels_arr == 1
        neg_mask = labels_arr == 0
        n_pos, n_neg = int(pos_mask.sum()), int(neg_mask.sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan"), "single_class_labels"
        all_s  = np.concatenate([scores_arr[neg_mask], scores_arr[pos_mask]])
        is_pos = np.concatenate([np.zeros(n_neg, bool), np.ones(n_pos, bool)])
        order  = np.argsort(all_s, kind="stable")
        sorted_s, sorted_is_pos = all_s[order], is_pos[order]
        n = n_pos + n_neg
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i + 1
            while j < n and sorted_s[j] == sorted_s[i]:
                j += 1
            ranks[i:j] = (i + 1 + j) / 2.0
            i = j
        U   = float(ranks[sorted_is_pos].sum()) - n_pos * (n_pos + 1) / 2.0
        auc = U / (n_pos * n_neg)
        return float(auc), "OK"
    except Exception as e:
        return float("nan"), f"AUROC_ERROR:{type(e).__name__}"


# ── AUPRC (sklearn-free) ──────────────────────────────────────────────────────

def compute_auprc(labels, scores):
    try:
        labels_arr = np.asarray(labels, dtype=np.int32)
        scores_arr = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(scores_arr)):
            return float("nan"), "invalid_score_nan_inf"
        if len(np.unique(labels_arr)) < 2:
            return float("nan"), "single_class_labels"
        n_pos = int((labels_arr == 1).sum())
        if n_pos == 0:
            return float("nan"), "no_positive"
        order  = np.argsort(scores_arr, kind="stable")[::-1]
        sorted_labels = labels_arr[order]
        tp = np.cumsum(sorted_labels)
        fp = np.cumsum(1 - sorted_labels)
        precision = tp / (tp + fp)
        recall    = tp / n_pos
        precision = np.concatenate([[1.0], precision])
        recall    = np.concatenate([[0.0], recall])
        auprc = float(np.trapezoid(precision, recall))
        return auprc, "OK"
    except Exception as e:
        return float("nan"), f"AUPRC_ERROR:{type(e).__name__}"


# ── Mask lookup ───────────────────────────────────────────────────────────────

def build_mask_lookup(mask_manifest_path: Path) -> dict:
    df_mask = pd.read_csv(mask_manifest_path, low_memory=False)
    df_pass = df_mask[df_mask["status"] == "PASS"]
    return dict(zip(df_pass["crop_path"].astype(str), df_pass["mask_path"].astype(str)))


def join_mask_to_manifest(df: pd.DataFrame, mask_lookup: dict) -> tuple:
    df = df.copy()
    df["mask_path"] = df["crop_path"].astype(str).map(mask_lookup)
    missing = df[df["mask_path"].isna()]
    return df, missing


# ── Dataset ───────────────────────────────────────────────────────────────────

class MaskedInputDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)

        vessel_cols = [c for c in df.columns if c in FORBIDDEN_VESSEL_COLS]
        if vessel_cols:
            raise RuntimeError(f"[GUARD] forbidden vessel columns: {vessel_cols}")
        if "mask_path" not in df.columns or df["mask_path"].isna().any():
            n = int(df["mask_path"].isna().sum()) if "mask_path" in df.columns else -1
            raise RuntimeError(f"[GUARD] {n} rows missing mask_path")
        for col in SCALAR_FEATURES:
            if df[col].isna().any():
                raise RuntimeError(f"[GUARD] NaN in scalar '{col}'")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        data = np.load(str(row["crop_path"]))
        arr  = data["ct_crop"].astype(np.float32)
        arr  = np.clip(arr, HU_MIN, HU_MAX)
        arr  = (arr - HU_MIN) / (HU_MAX - HU_MIN)
        img  = torch.from_numpy(arr)

        mask_data = np.load(str(row["mask_path"]))
        mask_t    = torch.from_numpy(mask_data["mask_3ch"].astype(np.float32))

        img = img * mask_t                        # 이미지에만 masking 적용
        img = (img - self.mean) / self.std

        if self.augment and torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[-1])

        scalar = torch.tensor(
            [float(row["lung_z_percentile"]), float(row["crop_lung_roi_ratio"])],
            dtype=torch.float32,
        )
        return img, scalar, int(row["label"]), float(row["sample_weight"])


# ── Model ─────────────────────────────────────────────────────────────────────

class ScalarFusionModel(nn.Module):
    def __init__(self, scalar_hidden: int = 32, scalar_out: int = 16, dropout: float = 0.2):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.img_features = backbone.features
        self.img_avgpool  = backbone.avgpool
        self.scalar_branch = nn.Sequential(
            nn.Linear(2, scalar_hidden),
            nn.BatchNorm1d(scalar_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(scalar_hidden, scalar_out),
            nn.ReLU(inplace=True),
        )
        fusion_in = 1280 + scalar_out
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fusion_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, img: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        x = self.img_features(img)
        x = self.img_avgpool(x)
        x = torch.flatten(x, 1)
        s = self.scalar_branch(scalar)
        return self.fusion_head(torch.cat([x, s], dim=1))


# ── Loss ──────────────────────────────────────────────────────────────────────

def weighted_bce_loss(logits, labels, sample_weights):
    bce = nn.BCEWithLogitsLoss(reduction="none")
    per_sample = bce(logits.squeeze(1), labels.float())
    return (per_sample * sample_weights).mean()


# ── Guards ────────────────────────────────────────────────────────────────────

def _guard_args(args) -> None:
    missing = []
    if not getattr(args, "confirm_train", False):
        missing.append("--confirm-train")
    if not getattr(args, "confirm_normal_vs_nsclc", False):
        missing.append("--confirm-normal-vs-nsclc")
    if not getattr(args, "confirm_no_final_test", False):
        missing.append("--confirm-no-final-test")
    if getattr(args, "epochs", 0) <= 0:
        missing.append("--epochs N (N > 0)")
    if missing:
        print(f"[GUARD] --train requires: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)


# ── Full training ─────────────────────────────────────────────────────────────

def run_full_train(args) -> int:
    _guard_args(args)
    set_seed(SEED)

    max_epochs = args.epochs
    batch_size = 64
    lr         = 1e-4
    patience   = 7
    n_workers  = 4
    ts         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 출력 충돌 방지 ────────────────────────────────────────────────────────
    for out_dir in [OUTPUT_ROOT, REPORT_ROOT]:
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[ABORT] output dir already exists and is not empty: {out_dir}", file=sys.stderr)
            print("[GUARD] Existing result overwrite is forbidden.", file=sys.stderr)
            sys.exit(2)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    best_ckpt = CKPT_DIR / "p_c_normal30b_best_val_auc_checkpoint.pt"
    last_ckpt = CKPT_DIR / "p_c_normal30b_last_checkpoint.pt"
    for cp in [best_ckpt, last_ckpt]:
        if cp.exists():
            print(f"[GUARD] output collision: {cp}", file=sys.stderr)
            sys.exit(2)

    errors = []

    # ── 1. 입력 파일 확인 ─────────────────────────────────────────────────────
    for p in [TRAIN_MANIFEST, VAL_MANIFEST, SCALAR_STATS_PATH, MASK_MANIFEST]:
        if not p.exists():
            errors.append({"check": "input_file", "error": f"MISSING: {p}"})
    if errors:
        _write_csv(errors, REPORT_ROOT / "p_c_normal30b_errors.csv")
        sys.exit(1)

    # ── 2. scalar stats ────────────────────────────────────────────────────────
    with open(SCALAR_STATS_PATH) as f:
        norm_payload = json.load(f)
    scalar_stats = norm_payload["features"]

    # ── 3. manifest 로드 ──────────────────────────────────────────────────────
    df_train_raw = pd.read_csv(TRAIN_MANIFEST, low_memory=False)
    df_val_raw   = pd.read_csv(VAL_MANIFEST,   low_memory=False)
    n_train = len(df_train_raw)
    n_val   = len(df_val_raw)
    print(f"[{STAGE_LABEL}] train={n_train} val={n_val}")

    if n_train != EXPECTED_TRAIN_ROWS:
        errors.append({"check": "train_row_count", "error": f"expected {EXPECTED_TRAIN_ROWS}, got {n_train}"})
    if n_val != EXPECTED_VAL_ROWS:
        errors.append({"check": "val_row_count",   "error": f"expected {EXPECTED_VAL_ROWS}, got {n_val}"})

    # sample_weight=1.0 확인
    for split_name, df in [("train", df_train_raw), ("val", df_val_raw)]:
        if "sample_weight" in df.columns:
            sw_vals = df["sample_weight"].astype(float)
            if not (sw_vals == 1.0).all():
                n_bad = int((sw_vals != 1.0).sum())
                errors.append({"check": f"{split_name}_sw", "error": f"{n_bad} rows != 1.0"})
        else:
            df["sample_weight"] = 1.0

    if errors:
        _write_csv(errors, REPORT_ROOT / "p_c_normal30b_errors.csv")
        sys.exit(1)

    # ── 4. mask join ──────────────────────────────────────────────────────────
    mask_lookup = build_mask_lookup(MASK_MANIFEST)
    df_train_masked, missing_tr = join_mask_to_manifest(df_train_raw, mask_lookup)
    df_val_masked,   missing_vl = join_mask_to_manifest(df_val_raw,   mask_lookup)

    if len(missing_tr) > 0 or len(missing_vl) > 0:
        errors.append({"check": "mask_join", "error": f"train={len(missing_tr)} val={len(missing_vl)} missing"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal30b_errors.csv")
        sys.exit(1)
    print(f"[{STAGE_LABEL}] mask join OK (train={n_train} val={n_val})")

    # ── 5. scalar 정규화 ──────────────────────────────────────────────────────
    df_train_norm = apply_scalar_norm(df_train_masked, scalar_stats)
    df_val_norm   = apply_scalar_norm(df_val_masked,   scalar_stats)

    for col in SCALAR_FEATURES:
        mu  = float(df_train_norm[col].mean())
        std = float(df_train_norm[col].std())
        print(f"[{STAGE_LABEL}] scalar {col}: mean={mu:.4f} std={std:.4f}")

    # ── 6. dataset / dataloader ───────────────────────────────────────────────
    ds_train = MaskedInputDataset(df_train_norm, augment=True)
    ds_val   = MaskedInputDataset(df_val_norm,   augment=False)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True, drop_last=False)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              num_workers=n_workers, pin_memory=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{STAGE_LABEL}] device={device} epochs={max_epochs} lr={lr} batch={batch_size} patience={patience}")

    # ── 7. 모델 + optimizer ───────────────────────────────────────────────────
    model     = ScalarFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── 8. training loop ──────────────────────────────────────────────────────
    train_log_rows    = []
    epoch_metric_rows = []

    best_val_auc    = float("-inf")
    best_epoch      = -1
    no_improve      = 0
    all_loss_finite = True
    all_grad_finite = True

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_losses, tr_correct, tr_total = [], 0, 0

        for step, (imgs, scalars, labels, sw) in enumerate(loader_train):
            imgs    = imgs.to(device)
            scalars = scalars.to(device)
            labels  = labels.to(device)
            sw      = sw.to(device)

            optimizer.zero_grad()
            logits = model(imgs, scalars)
            loss   = weighted_bce_loss(logits, labels, sw)
            loss_v = float(loss.item())

            if not math.isfinite(loss_v):
                all_loss_finite = False
                errors.append({"check": f"ep{epoch}_step{step}_loss", "error": f"NaN/Inf={loss_v}"})
                print(f"[{STAGE_LABEL}] [ERROR] NaN/Inf loss ep={epoch} step={step}", file=sys.stderr)
                break

            loss.backward()

            if epoch == 1 and step == 0:
                grad_ok = all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters() if p.requires_grad
                )
                if not grad_ok:
                    all_grad_finite = False
                    errors.append({"check": "ep1_step0_grad", "error": "NaN/Inf gradient"})

            optimizer.step()
            preds = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
            tr_correct += (preds == labels).sum().item()
            tr_total   += labels.size(0)
            tr_losses.append(loss_v)
            train_log_rows.append({"epoch": epoch, "step": step, "loss": round(loss_v, 6)})

        if not all_loss_finite:
            break

        train_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        train_acc  = float(tr_correct / tr_total) if tr_total > 0 else float("nan")

        # ── val ──
        model.eval()
        vl_losses, all_probs, all_labels_ep = [], [], []

        with torch.no_grad():
            for imgs, scalars, labels, sw in loader_val:
                imgs    = imgs.to(device)
                scalars = scalars.to(device)
                labels  = labels.to(device)
                sw      = sw.to(device)
                logits  = model(imgs, scalars)
                loss    = weighted_bce_loss(logits, labels, sw)
                loss_v  = float(loss.item())
                if not math.isfinite(loss_v):
                    all_loss_finite = False
                    errors.append({"check": f"ep{epoch}_val_loss", "error": f"NaN/Inf={loss_v}"})
                vl_losses.append(loss_v)
                probs = torch.sigmoid(logits.squeeze(1))
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels_ep.extend(labels.cpu().numpy().tolist())

        val_loss  = float(np.mean(vl_losses)) if vl_losses else float("nan")
        val_auc,  val_auc_status  = compute_auroc(all_labels_ep, all_probs)
        val_auprc, val_auprc_status = compute_auprc(all_labels_ep, all_probs)

        auc_disp   = f"{val_auc:.4f}"   if not math.isnan(val_auc)   else "NaN"
        auprc_disp = f"{val_auprc:.4f}" if not math.isnan(val_auprc) else "NaN"

        print(f"[{STAGE_LABEL}] ep={epoch}/{max_epochs} "
              f"tr_loss={train_loss:.4f} tr_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_auc={auc_disp} val_auprc={auprc_disp}")

        epoch_metric_rows.append({
            "epoch":        epoch,
            "train_loss":   round(train_loss, 6),
            "train_acc":    round(train_acc, 4),
            "val_loss":     round(val_loss, 6),
            "val_auc":      auc_disp,
            "val_auprc":    auprc_disp,
            "val_auc_status":   val_auc_status,
            "val_auprc_status": val_auprc_status,
        })

        # ── checkpoint ──
        ckpt_base = {
            "model_state_dict":        model.state_dict(),
            "optimizer_state_dict":    optimizer.state_dict(),
            "epoch":                   epoch,
            "smoke_only":              False,
            "is_final_model_candidate": True,
            "full_training":           True,
            "stage":                   STAGE_LABEL,
            "train_loss":              train_loss,
            "train_acc":               train_acc,
            "val_loss":                val_loss,
            "val_auc":                 val_auc,
            "val_auprc":               val_auprc,
            "val_auc_status":          val_auc_status,
            "scalar_features":         SCALAR_FEATURES,
            "scalar_norm_source":      str(SCALAR_STATS_PATH),
            "mask_manifest":           str(MASK_MANIFEST),
            "mask_applied_to_image_only": True,
            "train_manifest":          str(TRAIN_MANIFEST),
            "val_manifest":            str(VAL_MANIFEST),
            "balanced_w1":             True,
            "sample_weight_reset_to_1": True,
            "vessel_feature_used":     False,
            "roi_masked_loss_used":    False,
            "final_test_used":         False,
            "threshold_optimized":     False,
            "label_mapping":           {"0": "normal", "1": "NSCLC"},
        }
        torch.save({**ckpt_base, "checkpoint_type": "last"}, last_ckpt)

        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            no_improve   = 0
            torch.save({**ckpt_base,
                        "checkpoint_type":    "best_val_auc",
                        "best_metric_name":   "val_auc",
                        "best_metric_value":  best_val_auc}, best_ckpt)
            print(f"[{STAGE_LABEL}] best_val_auc={best_val_auc:.4f} (ep={epoch})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{STAGE_LABEL}] early stopping at ep={epoch}")
                break

    # ── 9. 결과 파일 ──────────────────────────────────────────────────────────
    _write_csv(train_log_rows,    REPORT_ROOT / "p_c_normal30b_train_log.csv")
    _write_csv(epoch_metric_rows, REPORT_ROOT / "p_c_normal30b_epoch_metrics.csv")

    verdict = "PASS"
    fail_reasons = []
    if not all_loss_finite:
        verdict = "FAIL"; fail_reasons.append("loss_nan_inf")
    if not all_grad_finite:
        verdict = "FAIL"; fail_reasons.append("gradient_nan_inf")
    if errors:
        verdict = "PARTIAL_PASS" if verdict != "FAIL" else "FAIL"
        fail_reasons.extend([e["error"] for e in errors])

    best_ep_row = next((r for r in epoch_metric_rows if r["epoch"] == best_epoch), {})

    summary = {
        "stage":           STAGE_LABEL,
        "timestamp":       ts,
        "verdict":         verdict,
        "fail_reasons":    fail_reasons,
        "n_train":         n_train,
        "n_val":           n_val,
        "epochs_trained":  len(epoch_metric_rows),
        "best_epoch":      best_epoch,
        "best_val_auc":    f"{best_val_auc:.4f}" if not math.isinf(best_val_auc) else "N/A",
        "best_val_auprc":  best_ep_row.get("val_auprc", "N/A"),
        "all_loss_finite": all_loss_finite,
        "all_grad_finite": all_grad_finite,
        "n_errors":        len(errors),
        "smoke_only":      False,
        "full_training_run": True,
        "final_test_accessed": False,
        "mask_applied_to_image_only": True,
        "best_ckpt":       str(best_ckpt),
        "last_ckpt":       str(last_ckpt),
        "note":            "P-C-NORMAL30b masked-input full training complete",
    }
    _write_json(summary, REPORT_ROOT / "p_c_normal30b_summary.json")

    guardrail_rows = [
        {"key": "full_training_run",          "value": True,  "expected": True,  "status": "OK"},
        {"key": "final_test_accessed",         "value": False, "expected": False, "status": "OK"},
        {"key": "threshold_optimization",      "value": False, "expected": False, "status": "OK"},
        {"key": "vessel_feature_used",         "value": False, "expected": False, "status": "OK"},
        {"key": "roi_masked_loss_used",        "value": False, "expected": False, "status": "OK"},
        {"key": "mask_applied_to_image_only",  "value": True,  "expected": True,  "status": "OK"},
        {"key": "balanced_w1_setting",         "value": True,  "expected": True,  "status": "OK"},
        {"key": "sample_weight_reset_to_1",    "value": True,  "expected": True,  "status": "OK"},
        {"key": "scalar_features_unchanged",   "value": True,  "expected": True,  "status": "OK"},
        {"key": "existing_result_overwrite",   "value": False, "expected": False, "status": "OK"},
    ]
    _write_csv(guardrail_rows, REPORT_ROOT / "p_c_normal30b_guardrail_check.csv")

    if verdict in ("PASS", "PARTIAL_PASS"):
        _write_json({
            "stage":     STAGE_LABEL,
            "verdict":   verdict,
            "timestamp": ts,
            "best_epoch":    best_epoch,
            "best_val_auc":  f"{best_val_auc:.4f}" if not math.isinf(best_val_auc) else "N/A",
        }, REPORT_ROOT / "DONE.json")

    print(f"[{STAGE_LABEL}] {'='*60}")
    print(f"[{STAGE_LABEL}] VERDICT: {verdict}")
    print(f"[{STAGE_LABEL}] best_epoch={best_epoch} best_val_auc={best_val_auc:.4f}" if not math.isinf(best_val_auc) else f"[{STAGE_LABEL}] best_val_auc=N/A")
    for r in fail_reasons:
        print(f"[{STAGE_LABEL}]   FAIL: {r}")
    print(f"[{STAGE_LABEL}] {'='*60}")

    return 0 if verdict in ("PASS", "PARTIAL_PASS") else 1


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P-C-NORMAL30b masked-input full training")
    parser.add_argument("--train",                    action="store_true")
    parser.add_argument("--confirm-train",            action="store_true", dest="confirm_train")
    parser.add_argument("--confirm-normal-vs-nsclc",  action="store_true", dest="confirm_normal_vs_nsclc")
    parser.add_argument("--confirm-no-final-test",    action="store_true", dest="confirm_no_final_test")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    if not args.train:
        print("[GUARD] --train flag required.", file=sys.stderr)
        sys.exit(2)

    return run_full_train(args)


if __name__ == "__main__":
    sys.exit(main())
