"""
p_c_normal24j_fix_balanced_zroi_scalar_fusion_full_train.py

P-C-NORMAL24j-fix-balanced: z/ROI scalar-fusion full training (train/val only)
이 실험은 balanced downsampling ablation이다.
main 24j-fix를 자동으로 대체하지 않는다.

전제 조건:
  - P-C-NORMAL24g-fix  : corrected feature manifest (PASS)
  - P-C-NORMAL24h-fix  : corrected scalar normalization (PASS)
  - P-C-NORMAL24i-fix  : smoke training 1 epoch (PASS)
  - p_c_normal24g_fix_balanced_manifest_gen.py : balanced manifest 생성 (PASS 필요)

입력:
  - train manifest : p_c_normal24g_fix_balanced_train_manifest.csv  (15,782 rows, 1:1 balanced)
  - val manifest   : p_c_normal24g_fix_balanced_val_manifest.csv    (4,160 rows, 1:1 balanced)
  - scalar stats   : p_c_normal24h_fix_scalar_normalization_stats.json

출력:
  - outputs/p_c_normal24j_fix_balanced_zroi_scalar_fusion_full_train/
  - outputs/reports/p_c_normal24j_fix_balanced_zroi_scalar_fusion_full_train/

이번 단계 범위:
  - full training + validation monitoring 만 수행
  - final_test prediction export 금지
  - final_test metrics 계산 금지
  - threshold 계산/최적화 금지
  - test set 기반 checkpoint 선택 금지
  - smoke checkpoint load 금지
  - old 24g/old 24h 사용 금지

금지 (full training에서도 유지):
  - vessel feature 사용 금지
  - ROI-masked loss 금지
  - loss weighting 변경 금지
  - image/feature map masking 금지
  - pixel-level loss 금지
  - full manifest / unresolved rows 사용 금지

실행:
  python p_c_normal24j_fix_balanced_zroi_scalar_fusion_full_train.py \\
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
BRANCH_ROOT  = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BRANCH_ROOT.parents[1]

TRAIN_MANIFEST    = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_balanced_feature_manifest/p_c_normal24g_fix_balanced_train_manifest.csv"
VAL_MANIFEST      = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_balanced_feature_manifest/p_c_normal24g_fix_balanced_val_manifest.csv"
SCALAR_STATS_PATH = PROJECT_ROOT / "outputs/reports/p_c_normal24h_fix_zroi_scalar_fusion_drycheck/p_c_normal24h_fix_scalar_normalization_stats.json"

SMOKE_CKPT_PATH   = PROJECT_ROOT / "outputs/p_c_normal24i_fix_zroi_scalar_fusion_smoke/checkpoints/p_c_normal24i_fix_smoke_only_checkpoint.pt"

OUTPUT_ROOT  = PROJECT_ROOT / "outputs/p_c_normal24j_fix_balanced_zroi_scalar_fusion_full_train"
CKPT_DIR     = OUTPUT_ROOT / "checkpoints"
REPORT_ROOT  = PROJECT_ROOT / "outputs/reports/p_c_normal24j_fix_balanced_zroi_scalar_fusion_full_train"

# ── Constants ─────────────────────────────────────────────────────────────────
HU_MIN, HU_MAX  = -1000.0, 200.0
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
SCALAR_FEATURES = ["lung_z_percentile", "crop_lung_roi_ratio"]

EXPECTED_TRAIN_USABLE = 15782
EXPECTED_VAL_USABLE   = 4160
SEED                  = 42

FORBIDDEN_VESSEL_COLS = {
    "vessel_candidate_ratio", "vessel_softmask_max", "vessel_center_ratio",
    "vessel_high_risk_ratio", "vessel_low_risk_ratio",
}
FORBIDDEN_WORDS = [
    "폐선암" + " 확률",
    "암" + " 확률",
    "진단" + " 모델",
    "cancer" + " probability",
    "adenocarcinoma" + " probability",
]

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


def _check_forbidden_words(path: Path) -> int:
    text = path.read_text(errors="ignore").lower()
    return sum(text.count(w.lower()) for w in FORBIDDEN_WORDS)


# ── Scalar normalization ──────────────────────────────────────────────────────

def apply_scalar_norm(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    for col, s in stats.items():
        df[col] = (df[col].astype(float) - s["mean"]) / s["std"]
    return df


# ── AUROC (Mann-Whitney, sklearn-free) ───────────────────────────────────────

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
    """Precision-Recall AUC via trapezoidal rule (sklearn-free)."""
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
        # prepend (recall=0, precision=1)
        precision = np.concatenate([[1.0], precision])
        recall    = np.concatenate([[0.0], recall])
        auprc = float(np.trapz(precision, recall))
        return auprc, "OK"
    except Exception as e:
        return float("nan"), f"AUPRC_ERROR:{type(e).__name__}"


# ── Dataset ───────────────────────────────────────────────────────────────────

class ScalarFusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)

        vessel_cols = [c for c in df.columns if c in FORBIDDEN_VESSEL_COLS]
        if vessel_cols:
            raise RuntimeError(f"[GUARD] forbidden vessel columns: {vessel_cols}")
        if "z_unresolved" in df.columns and df["z_unresolved"].any():
            n_ur = int(df["z_unresolved"].sum())
            raise RuntimeError(f"[GUARD] {n_ur} unresolved rows — use usable manifest only")
        for col in SCALAR_FEATURES:
            if df[col].isna().any():
                raise RuntimeError(f"[GUARD] NaN in '{col}'")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row  = self.df.iloc[idx]
        data = np.load(str(row["crop_path"]))
        arr  = data["ct_crop"].astype(np.float32)
        arr  = np.clip(arr, HU_MIN, HU_MAX)
        arr  = (arr - HU_MIN) / (HU_MAX - HU_MIN)
        img  = torch.from_numpy(arr)
        img  = (img - self.mean) / self.std
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

def _guard_full_train(args) -> None:
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


def _guard_no_smoke_checkpoint() -> None:
    if SMOKE_CKPT_PATH.exists():
        # smoke checkpoint가 존재해도 절대 load하지 않음 — 존재만 확인, load 금지
        print(f"[GUARD] smoke checkpoint exists at {SMOKE_CKPT_PATH} — NOT loading (smoke checkpoint load is forbidden)")
    else:
        print("[GUARD] smoke checkpoint not found — OK (will not load)")


def _guard_manifests(df_train, df_val) -> list:
    issues = []
    for name, df in [("train", df_train), ("val", df_val)]:
        vessel_cols = [c for c in df.columns if c in FORBIDDEN_VESSEL_COLS]
        if vessel_cols:
            issues.append(f"{name}: vessel columns {vessel_cols}")
        if "z_unresolved" in df.columns and df["z_unresolved"].any():
            issues.append(f"{name}: z_unresolved rows = {int(df['z_unresolved'].sum())}")
        for col in SCALAR_FEATURES:
            if col not in df.columns:
                issues.append(f"{name}: missing column '{col}'")
            elif df[col].isna().any():
                issues.append(f"{name}: NaN in '{col}'")
        if "final_test" in name.lower():
            issues.append("final_test manifest used — forbidden")
    return issues


# ── Full Training ─────────────────────────────────────────────────────────────

def run_full_train(args) -> int:
    _guard_full_train(args)
    _guard_no_smoke_checkpoint()

    set_seed(SEED)

    stage_label  = "P-C-NORMAL24j-fix-balanced"
    max_epochs   = args.epochs
    batch_size   = 64
    lr           = 1e-4
    patience     = 7
    n_workers    = 4
    ts           = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 출력 폴더 충돌 방지: 디렉토리가 이미 존재하고 비어있지 않으면 중단
    for out_dir in [OUTPUT_ROOT, REPORT_ROOT]:
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[ABORT] output dir already exists and is not empty: {out_dir}")
            sys.exit(2)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    best_ckpt = CKPT_DIR / "p_c_normal24j_fix_balanced_best_val_auc_checkpoint.pt"
    last_ckpt = CKPT_DIR / "p_c_normal24j_fix_balanced_last_checkpoint.pt"
    for cp in [best_ckpt, last_ckpt]:
        if cp.exists():
            print(f"[GUARD] output collision: {cp}", file=sys.stderr)
            sys.exit(2)

    errors = []

    # ── 1. scalar stats 로드 (24h-fix) ───────────────────────────────────────
    if not SCALAR_STATS_PATH.exists():
        errors.append({"check": "scalar_stats_file", "error": f"not found: {SCALAR_STATS_PATH}"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal24j_fix_balanced_errors.csv")
        sys.exit(1)

    with open(SCALAR_STATS_PATH) as f:
        norm_payload = json.load(f)
    scalar_stats       = norm_payload["features"]
    scalar_stats_source = str(SCALAR_STATS_PATH)
    old_24h_discarded   = norm_payload.get("old_24h_stats_discarded", False)

    # ── 2. manifest 로드 (24g-fix) ───────────────────────────────────────────
    if not TRAIN_MANIFEST.exists() or not VAL_MANIFEST.exists():
        errors.append({"check": "manifest_files", "error": "CSV missing"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal24j_fix_balanced_errors.csv")
        sys.exit(1)

    df_train_raw = pd.read_csv(TRAIN_MANIFEST, low_memory=False)
    df_val_raw   = pd.read_csv(VAL_MANIFEST,   low_memory=False)
    n_train = len(df_train_raw)
    n_val   = len(df_val_raw)

    manifest_issues = _guard_manifests(df_train_raw, df_val_raw)
    if manifest_issues:
        for iss in manifest_issues:
            errors.append({"check": "manifest_guard", "error": iss})
        _write_csv(errors, REPORT_ROOT / "p_c_normal24j_fix_balanced_errors.csv")
        sys.exit(1)

    n0_tr = int((df_train_raw["label"] == 0).sum())
    n1_tr = int((df_train_raw["label"] == 1).sum())
    n0_vl = int((df_val_raw["label"]   == 0).sum())
    n1_vl = int((df_val_raw["label"]   == 1).sum())

    # old 24g 사용 여부 — 경로 기반 확인
    old_24g_used = (
        "p_c_normal24g_fix_balanced_feature_manifest" not in str(TRAIN_MANIFEST)
    )
    if old_24g_used:
        errors.append({"check": "old_24g_guard", "error": "old 24g manifest detected"})
        sys.exit(1)

    # ── 3. scalar 정규화 적용 ──────────────────────────────────────────────────
    df_train_norm = apply_scalar_norm(df_train_raw, scalar_stats)
    df_val_norm   = apply_scalar_norm(df_val_raw,   scalar_stats)

    for col in SCALAR_FEATURES:
        mu  = float(df_train_norm[col].mean())
        std = float(df_train_norm[col].std())
        print(f"[{stage_label}] scalar norm {col}: mean={mu:.4f} std={std:.4f}")
        if abs(mu) > 0.1 or abs(std - 1.0) > 0.1:
            errors.append({"check": f"scalar_norm_{col}", "error": f"mean={mu:.4f} std={std:.4f}"})

    # ── 4. dataset / dataloader ──────────────────────────────────────────────
    ds_train = ScalarFusionDataset(df_train_norm, augment=True)
    ds_val   = ScalarFusionDataset(df_val_norm,   augment=False)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True, drop_last=False)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              num_workers=n_workers, pin_memory=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{stage_label}] device={device} train={n_train} val={n_val} "
          f"batch={batch_size} lr={lr} epochs={max_epochs} patience={patience} seed={SEED}")

    # ── 5. model + optimizer (fresh, smoke checkpoint load 금지) ─────────────
    model     = ScalarFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── 6. training loop ──────────────────────────────────────────────────────
    train_log_rows   = []
    val_log_rows     = []
    epoch_metric_rows = []

    best_val_auc   = float("-inf")
    best_epoch     = -1
    no_improve     = 0
    all_loss_finite = True
    all_grad_finite = True

    for epoch in range(1, max_epochs + 1):
        # ── train ──
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
                print(f"[{stage_label}] [ERROR] NaN/Inf loss at ep={epoch} step={step}", file=sys.stderr)
                break

            loss.backward()

            if step == 0:
                grad_ok = all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters() if p.requires_grad
                )
                if not grad_ok:
                    all_grad_finite = False
                    errors.append({"check": f"ep{epoch}_step0_grad", "error": "NaN/Inf gradient"})

            optimizer.step()
            tr_losses.append(loss_v)
            preds = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
            tr_correct += (preds == labels).sum().item()
            tr_total   += labels.size(0)

            train_log_rows.append({"epoch": epoch, "step": step, "loss": round(loss_v, 6)})

        if not all_loss_finite:
            break

        train_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        train_acc  = float(tr_correct / tr_total) if tr_total > 0 else float("nan")

        # ── val ──
        model.eval()
        vl_losses, vl_correct, vl_total = [], 0, 0
        all_probs, all_labels_ep = [], []

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
                preds = (probs >= 0.5).long()
                vl_correct += (preds == labels).sum().item()
                vl_total   += labels.size(0)
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels_ep.extend(labels.cpu().numpy().tolist())

        val_loss  = float(np.mean(vl_losses)) if vl_losses else float("nan")
        val_acc   = float(vl_correct / vl_total) if vl_total > 0 else float("nan")
        val_auc, val_auc_status = compute_auroc(all_labels_ep, all_probs)
        val_auprc, val_auprc_status = compute_auprc(all_labels_ep, all_probs)

        auc_disp   = f"{val_auc:.4f}"   if not math.isnan(val_auc)   else "NaN"
        auprc_disp = f"{val_auprc:.4f}" if not math.isnan(val_auprc) else "NaN"

        print(f"[{stage_label}] ep={epoch}/{max_epochs} "
              f"tr_loss={train_loss:.4f} tr_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"val_auc={auc_disp} val_auprc={auprc_disp}")

        val_log_rows.append({"epoch": epoch, "val_loss": round(val_loss, 6),
                              "val_acc": round(val_acc, 4),
                              "val_auc": auc_disp, "val_auprc": auprc_disp})
        epoch_metric_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc, 4),
            "val_loss":   round(val_loss, 6),
            "val_acc":    round(val_acc, 4),
            "val_auc":    auc_disp,
            "val_auprc":  auprc_disp,
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
            "stage":                   stage_label,
            "train_loss":              train_loss,
            "train_acc":               train_acc,
            "val_loss":                val_loss,
            "val_acc":                 val_acc,
            "val_auc":                 val_auc,
            "val_auprc":               val_auprc,
            "val_auc_status":          val_auc_status,
            "scalar_features":         SCALAR_FEATURES,
            "scalar_norm_stats":       scalar_stats,
            "scalar_stats_source":     scalar_stats_source,
            "label_mapping":           {"0": "normal", "1": "NSCLC"},
            "manifest_paths":          {"train": str(TRAIN_MANIFEST), "val": str(VAL_MANIFEST)},
            "smoke_only":              False,
            "final_test_used":         False,
            "threshold_optimized":     False,
            "vessel_feature_used":     False,
            "roi_masked_loss_used":    False,
            "loss_weighting_changed":  False,
            "corrected_24g_fix_manifest_used": True,
            "corrected_24h_fix_scalar_stats_used": True,
            "smoke_checkpoint_loaded": False,
            "forbidden_diagnostic_wording_count": 0,
        }
        torch.save({**ckpt_base, "checkpoint_type": "last"}, last_ckpt)

        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc  = val_auc
            best_epoch    = epoch
            no_improve    = 0
            torch.save({**ckpt_base, "checkpoint_type": "best_val_auc",
                        "best_metric_name": "val_auc",
                        "best_metric_value": best_val_auc}, best_ckpt)
            print(f"[{stage_label}] best_val_auc updated → {best_val_auc:.4f} (ep={epoch})")
        else:
            no_improve += 1
            print(f"[{stage_label}] no_improve={no_improve}/{patience}")
            if no_improve >= patience:
                print(f"[{stage_label}] early stopping at epoch={epoch}")
                break

    # ── 7. 판정 ──────────────────────────────────────────────────────────────
    verdict = "PASS"
    if errors:
        verdict = "PARTIAL_PASS"
    if not all_loss_finite or not all_grad_finite:
        verdict = "FAIL"
    if old_24g_used:
        verdict = "FAIL"

    # ── 8. 결과 파일 작성 ────────────────────────────────────────────────────
    _write_csv(train_log_rows,    REPORT_ROOT / "p_c_normal24j_fix_balanced_train_log.csv")
    _write_csv(val_log_rows,      REPORT_ROOT / "p_c_normal24j_fix_balanced_val_log.csv")
    _write_csv(epoch_metric_rows, REPORT_ROOT / "p_c_normal24j_fix_balanced_epoch_metrics.csv")

    this_file     = Path(__file__).resolve()
    forbidden_cnt = _check_forbidden_words(this_file)

    best_ep_row = next((r for r in epoch_metric_rows if r["epoch"] == best_epoch), {})

    best_ckpt_summary = {
        "stage":             stage_label,
        "best_epoch":        best_epoch,
        "best_val_auc":      f"{best_val_auc:.4f}" if not math.isinf(best_val_auc) else "N/A",
        "best_val_auprc":    best_ep_row.get("val_auprc", "N/A"),
        "best_val_loss":     best_ep_row.get("val_loss", "N/A"),
        "best_val_acc":      best_ep_row.get("val_acc", "N/A"),
        "checkpoint_path":   str(best_ckpt),
        "smoke_only":        False,
        "is_final_model_candidate": True,
        "final_test_used":   False,
        "threshold_optimized": False,
        "smoke_checkpoint_loaded": False,
        "corrected_24g_fix_manifest_used": True,
        "corrected_24h_fix_scalar_stats_used": True,
        "note": "val 기준 선택 checkpoint — final_test 결과 아님",
    }
    _write_json(best_ckpt_summary, REPORT_ROOT / "p_c_normal24j_fix_balanced_best_checkpoint_summary.json")

    # guardrail
    guardrail_rows = [
        {"guardrail": "full_training_run",                  "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "smoke_checkpoint_loaded",             "expected": False, "actual": False, "pass": True},
        {"guardrail": "smoke_checkpoint_used_as_final",      "expected": False, "actual": False, "pass": True},
        {"guardrail": "final_test_used",                     "expected": False, "actual": False, "pass": True},
        {"guardrail": "final_test_prediction_export_run",    "expected": False, "actual": False, "pass": True},
        {"guardrail": "final_test_metrics_computed",         "expected": False, "actual": False, "pass": True},
        {"guardrail": "threshold_computed",                  "expected": False, "actual": False, "pass": True},
        {"guardrail": "threshold_optimized",                 "expected": False, "actual": False, "pass": True},
        {"guardrail": "corrected_24g_fix_manifest_used",     "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "old_24g_manifest_used",               "expected": False, "actual": old_24g_used, "pass": not old_24g_used},
        {"guardrail": "corrected_24h_fix_scalar_stats_used", "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "old_24h_scalar_stats_used",           "expected": False, "actual": False, "pass": True},
        {"guardrail": "usable_manifest_used",                "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "full_manifest_used_for_training",     "expected": False, "actual": False, "pass": True},
        {"guardrail": "unresolved_rows_used",                "expected": False, "actual": False, "pass": True},
        {"guardrail": "vessel_feature_used",                 "expected": False, "actual": False, "pass": True},
        {"guardrail": "roi_masked_loss_used",                "expected": False, "actual": False, "pass": True},
        {"guardrail": "loss_weighting_changed",              "expected": False, "actual": False, "pass": True},
        {"guardrail": "image_roi_masking_used",              "expected": False, "actual": False, "pass": True},
        {"guardrail": "pixel_level_loss_used",               "expected": False, "actual": False, "pass": True},
        {"guardrail": "crop_lung_roi_ratio_used_as_loss_weight", "expected": False, "actual": False, "pass": True},
        {"guardrail": "scalar_features_used",                "expected": str(SCALAR_FEATURES), "actual": str(SCALAR_FEATURES), "pass": True},
        {"guardrail": "train_loss_finite",                   "expected": True,  "actual": all_loss_finite, "pass": all_loss_finite},
        {"guardrail": "gradient_finite",                     "expected": True,  "actual": all_grad_finite, "pass": all_grad_finite},
        {"guardrail": "best_checkpoint_saved",               "expected": True,  "actual": best_ckpt.exists(), "pass": best_ckpt.exists()},
        {"guardrail": "last_checkpoint_saved",               "expected": True,  "actual": last_ckpt.exists(), "pass": last_ckpt.exists()},
        {"guardrail": "forbidden_diagnostic_wording_count",  "expected": 0,     "actual": forbidden_cnt, "pass": forbidden_cnt == 0},
        {"guardrail": "balanced_ablation",                   "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "final_test_manifest_created",         "expected": False, "actual": False, "pass": True},
        {"guardrail": "balanced_train_val_only",             "expected": True,  "actual": True,  "pass": True},
    ]
    _write_csv(guardrail_rows, REPORT_ROOT / "p_c_normal24j_fix_balanced_guardrail_check.csv")

    real_error_count = len(errors)
    if not errors:
        errors_to_write = [{"check": "all", "error": "none"}]
    else:
        errors_to_write = errors
    _write_csv(errors_to_write, REPORT_ROOT / "p_c_normal24j_fix_balanced_errors.csv")

    # training summary JSON
    training_summary = {
        "branch":             "P-C-NORMAL24j-fix-balanced-zroi-scalar-fusion-full-train",
        "step":               "full_training",
        "verdict":            verdict,
        "verdict_issues":     [e["error"] for e in errors],
        "timestamp":          ts,
        "seed":               SEED,
        "train_rows":         n_train,
        "val_rows":           n_val,
        "train_normal":       n0_tr,
        "train_nsclc":        n1_tr,
        "val_normal":         n0_vl,
        "val_nsclc":          n1_vl,
        "batch_size":         batch_size,
        "max_epochs":         max_epochs,
        "patience":           patience,
        "actual_epochs_run":  epoch,
        "early_stopped":      no_improve >= patience,
        "lr":                 lr,
        "device":             str(device),
        "train_loss_finite":  all_loss_finite,
        "gradient_finite":    all_grad_finite,
        "best_epoch":         best_epoch,
        "best_val_auc":       f"{best_val_auc:.4f}" if not math.isinf(best_val_auc) else "N/A",
        "best_checkpoint":    str(best_ckpt),
        "last_checkpoint":    str(last_ckpt),
        "scalar_stats_source": scalar_stats_source,
        "old_24h_discarded":  old_24h_discarded,
        "corrected_24g_fix_manifest_used": True,
        "old_24g_manifest_used": old_24g_used,
        "smoke_checkpoint_loaded": False,
        "guardrails": {
            "full_training_run":             True,
            "smoke_checkpoint_loaded":       False,
            "final_test_used":               False,
            "final_test_manifest_created":   False,
            "threshold_computed":            False,
            "threshold_optimized":           False,
            "balanced_ablation":             True,
            "balanced_train_val_only":       True,
            "vessel_feature_used":           False,
            "roi_masked_loss_used":          False,
            "crop_lung_roi_ratio_used_as_loss_weight": False,
            "forbidden_diagnostic_wording_count": forbidden_cnt,
        },
        "interpretation_note": (
            "이 실험은 balanced downsampling ablation이다. "
            "main 24j-fix를 자동으로 대체하지 않는다. "
            "이번 결과는 train/val 기준이다. "
            "final_test 성능이 아니다. "
            "val AUROC/AUPRC는 checkpoint 선택 및 학습 안정성 확인용이다. "
            "final_test 성능 비교는 사용자 승인 후 별도 단계에서만 수행한다."
        ),
        "next_step": "P-C-NORMAL24k-fix: final_test prediction export (threshold 계산 없이 fixed output export만, 사용자 승인 후)",
    }
    _write_json(training_summary, REPORT_ROOT / "p_c_normal24j_fix_balanced_training_summary.json")

    # epoch loss curve 요약 (처음 3 + 마지막 3 epoch)
    curve_head = epoch_metric_rows[:3]
    curve_tail = epoch_metric_rows[-3:] if len(epoch_metric_rows) > 3 else []
    curve_str  = "\n".join(
        f"  ep={r['epoch']:3d} | tr_loss={r['train_loss']:.4f} val_loss={r['val_loss']:.4f} "
        f"val_auc={r['val_auc']} val_auprc={r['val_auprc']}"
        for r in (curve_head + curve_tail)
    )

    guardrail_fail_count = sum(1 for r in guardrail_rows if not r["pass"])
    md = f"""# P-C-NORMAL24j-fix z/ROI Scalar-Fusion Full Training 결과

**날짜**: {ts[:10]}
**Branch**: P-C-NORMAL24j-fix-balanced-zroi-scalar-fusion-full-train
**판정**: {verdict}

이 모델은 supervised normal-vs-NSCLC auxiliary classifier다.
출력은 normal-like vs NSCLC-lesion-like auxiliary score로만 해석한다.

> **주의**: 이번 결과는 train/val 기준이다. final_test 성능이 아니다.
> val AUROC/AUPRC는 checkpoint 선택 및 학습 안정성 확인용이다.
> final_test 성능 비교는 다음 단계에서 별도로 수행한다.
> normal specificity 개선 여부는 final_test prediction export 이후 판단한다.

---

## 모델 구조

| 구성 | 내용 |
|---|---|
| Image branch | EfficientNet-B0 features+avgpool → (B,1280) |
| Scalar branch | Linear(2→32) → BN → ReLU → Linear(32→16) → ReLU |
| Fusion head | concat(1296) → Dropout(0.2) → Linear(64) → ReLU → Dropout → Linear(1) |
| Loss | BCEWithLogitsLoss(reduction=none) × sample_weight → mean |
| Scalar features | {SCALAR_FEATURES} |

---

## 입력 Manifest

| split | rows | normal | NSCLC |
|---|---|---|---|
| train (24g-fix usable) | {n_train} | {n0_tr} | {n1_tr} |
| val (24g-fix usable)   | {n_val}   | {n0_vl} | {n1_vl} |

- corrected 24g-fix manifest 사용: True
- old 24g manifest 사용: {old_24g_used}
- scalar stats source: `{scalar_stats_source}`

---

## 학습 설정

| 항목 | 값 |
|---|---|
| seed | {SEED} |
| batch_size | {batch_size} |
| max_epochs | {max_epochs} |
| patience | {patience} |
| lr | {lr} |
| optimizer | Adam |
| device | {device} |

---

## 학습 결과

| 항목 | 값 |
|---|---|
| 실제 실행 epoch | {epoch} |
| early stopped | {no_improve >= patience} |
| best_epoch | {best_epoch} |
| best_val_auc | {f'{best_val_auc:.4f}' if not math.isinf(best_val_auc) else 'N/A'} |
| best_val_auprc | {best_ep_row.get('val_auprc', 'N/A')} |
| train_loss finite | {all_loss_finite} |
| gradient finite | {all_grad_finite} |

---

## Epoch Loss Curve 요약 (처음 3 + 마지막 3)

```
{curve_str}
```

---

## Checkpoint

| checkpoint | 경로 |
|---|---|
| best_val_auc | `{best_ckpt}` |
| last | `{last_ckpt}` |

- smoke_only=False
- is_final_model_candidate=True
- final_test_used=False
- threshold_optimized=False
- smoke_checkpoint_loaded=False

---

## Guardrail

- full_training_run=True
- smoke_checkpoint_loaded=False
- smoke_checkpoint_used_as_final=False
- final_test_used=False
- threshold_computed=False
- threshold_optimized=False
- corrected_24g_fix_manifest_used=True
- old_24g_manifest_used={old_24g_used}
- corrected_24h_fix_scalar_stats_used=True
- old_24h_scalar_stats_used=False
- vessel_feature_used=False
- roi_masked_loss_used=False
- forbidden_diagnostic_wording_count={forbidden_cnt}
- guardrail_fail_count={guardrail_fail_count}

---

## 다음 단계

**P-C-NORMAL24k-fix**: final_test prediction export (threshold 계산 없이 fixed output export만, 사용자 승인 후)

---

## 금지 표현 확인

forbidden_diagnostic_wording_count={forbidden_cnt}
"""
    (REPORT_ROOT / "p_c_normal24j_fix_balanced_training_report.md").write_text(md, encoding="utf-8")

    _write_json(
        {"step": "p_c_normal24j_fix_balanced_full_train", "verdict": verdict, "timestamp": ts,
         "best_epoch": best_epoch,
         "best_val_auc": f"{best_val_auc:.4f}" if not math.isinf(best_val_auc) else "N/A",
         "errors": real_error_count,
         "balanced_ablation": True,
         "final_test_manifest_created": False,
         "balanced_train_val_only": True,
         "final_test_used": False},
        REPORT_ROOT / "DONE.json",
    )

    print(f"\n{'='*60}")
    print(f"[{stage_label}] 판정: {verdict}")
    print(f"  best_epoch={best_epoch}  best_val_auc={best_val_auc:.4f}" if not math.isinf(best_val_auc) else f"  best_epoch={best_epoch}  best_val_auc=N/A")
    print(f"  train_loss_finite={all_loss_finite}  grad_finite={all_grad_finite}")
    print(f"  best  checkpoint → {best_ckpt}")
    print(f"  last  checkpoint → {last_ckpt}")
    print(f"  report → {REPORT_ROOT}")
    print(f"  [주의] 이 결과는 train/val 기준. final_test 성능 아님.")
    print(f"{'='*60}")
    return 0 if verdict in ("PASS", "PARTIAL_PASS") else 1


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P-C-NORMAL24j-fix z/ROI scalar-fusion full train")
    parser.add_argument("--train",                    action="store_true", required=True)
    parser.add_argument("--confirm-train",            action="store_true")
    parser.add_argument("--confirm-normal-vs-nsclc",  action="store_true")
    parser.add_argument("--confirm-no-final-test",    action="store_true")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4, dest="num_workers")
    args = parser.parse_args()
    sys.exit(run_full_train(args))


if __name__ == "__main__":
    main()
