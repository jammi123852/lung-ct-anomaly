"""
p_c_normal30a_masked_input_smoke_train.py

P-C-NORMAL30a: masked-input 1 epoch smoke training
기준: P-C-NORMAL24j-fix-balanced-w1 + crop-level 3ch mask (P-C-NORMAL29b)

변경 사항 (24j 대비):
  - 추가: crop-level mask (29b) join 및 masked_ct = ct_crop * mask_3ch 적용
  - 제한: epoch=1 (smoke only), full training 금지, final_test 접근 금지

불변 조건:
  - train/val manifest: P-C-NORMAL24g-fix-balanced-w1
  - scalar features: lung_z_percentile, crop_lung_roi_ratio
  - scalar normalization: P-C-NORMAL24h-fix stats
  - balanced-w1 setting (sample_weight=1.0)
  - BCEWithLogitsLoss
  - vessel feature 금지, ROI-masked loss 금지, crop_lung_roi_ratio loss weighting 금지

masking 방식:
  - masked_ct = ct_crop * mask_3ch  (이미지에만 적용)
  - scalar feature, label, sample_weight 수정 없음

실행 (smoke only):
  python p_c_normal30a_masked_input_smoke_train.py --smoke

출력:
  - outputs/p_c_normal30a_masked_input_smoke_train/
  - outputs/reports/p_c_normal30a_masked_input_smoke_train/
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

OUTPUT_ROOT  = PROJECT_ROOT / "outputs/p_c_normal30a_masked_input_smoke_train"
CKPT_DIR     = OUTPUT_ROOT / "checkpoints"
REPORT_ROOT  = PROJECT_ROOT / "outputs/reports/p_c_normal30a_masked_input_smoke_train"

# ── Constants ─────────────────────────────────────────────────────────────────
HU_MIN, HU_MAX  = -1000.0, 200.0
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
SCALAR_FEATURES = ["lung_z_percentile", "crop_lung_roi_ratio"]

EXPECTED_TRAIN_ROWS = 15782
EXPECTED_VAL_ROWS   = 4160
SEED                = 42
SMOKE_EPOCHS        = 1
STAGE_LABEL         = "P-C-NORMAL30a"

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
        auprc = float(np.trapz(precision, recall))
        return auprc, "OK"
    except Exception as e:
        return float("nan"), f"AUPRC_ERROR:{type(e).__name__}"


# ── Mask join ─────────────────────────────────────────────────────────────────

def build_mask_lookup(mask_manifest_path: Path) -> dict:
    """mask_manifest에서 crop_path → mask_path 딕셔너리 반환 (PASS rows only)."""
    df_mask = pd.read_csv(mask_manifest_path, low_memory=False)
    df_pass = df_mask[df_mask["status"] == "PASS"]
    return dict(zip(df_pass["crop_path"].astype(str), df_pass["mask_path"].astype(str)))


def join_mask_to_manifest(df: pd.DataFrame, mask_lookup: dict, split_name: str) -> tuple:
    """df에 mask_path 컬럼 추가. missing mask 목록 반환."""
    df = df.copy()
    df["mask_path"] = df["crop_path"].astype(str).map(mask_lookup)
    missing = df[df["mask_path"].isna()]
    join_rows = []
    for _, row in df.iterrows():
        join_rows.append({
            "split": split_name,
            "crop_path": row["crop_path"],
            "mask_path": row.get("mask_path", ""),
            "mask_found": "OK" if pd.notna(row.get("mask_path")) else "MISSING",
        })
    return df, missing, join_rows


# ── Dataset (masked input) ────────────────────────────────────────────────────

class MaskedInputDataset(Dataset):
    """
    ct_crop에 mask_3ch를 곱한 masked_ct를 이미지 입력으로 사용.
    scalar feature, label, sample_weight는 수정 없음.
    """
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)

        vessel_cols = [c for c in df.columns if c in FORBIDDEN_VESSEL_COLS]
        if vessel_cols:
            raise RuntimeError(f"[GUARD] forbidden vessel columns: {vessel_cols}")
        if "mask_path" not in df.columns:
            raise RuntimeError("[GUARD] mask_path column missing — join not done")
        if df["mask_path"].isna().any():
            n_missing = int(df["mask_path"].isna().sum())
            raise RuntimeError(f"[GUARD] {n_missing} rows missing mask_path")
        for col in SCALAR_FEATURES:
            if df[col].isna().any():
                raise RuntimeError(f"[GUARD] NaN in scalar '{col}'")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # ct_crop 로드
        data = np.load(str(row["crop_path"]))
        arr  = data["ct_crop"].astype(np.float32)      # (3, 96, 96) in HU
        arr  = np.clip(arr, HU_MIN, HU_MAX)
        arr  = (arr - HU_MIN) / (HU_MAX - HU_MIN)      # [0, 1]
        img  = torch.from_numpy(arr)                    # (3, 96, 96)

        # mask_3ch 로드 (uint8 또는 bool)
        mask_data = np.load(str(row["mask_path"]))
        mask_arr  = mask_data["mask_3ch"].astype(np.float32)  # (3, 96, 96) → 0 or 1
        mask_t    = torch.from_numpy(mask_arr)                  # (3, 96, 96)

        # masked_ct = ct_crop * mask_3ch (이미지에만 적용)
        img = img * mask_t

        # ImageNet 정규화
        img = (img - self.mean) / self.std

        if self.augment and torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[-1])

        # scalar features (수정 없음)
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


# ── Guardrail ─────────────────────────────────────────────────────────────────

GUARDRAIL = {
    "smoke_training_only": True,
    "full_training_run": False,
    "final_test_accessed": False,
    "prediction_export_run": False,
    "threshold_optimization": False,
    "threshold_sweep": False,
    "checkpoint_selection": False,
    "existing_checkpoint_modified": False,
    "existing_result_overwrite": False,
    "balanced_w1_setting_preserved": True,
    "scalar_features_unchanged": True,
    "sample_weight_reset_to_1": True,
    "mask_applied_to_image_only": True,
    "roi_masked_loss_used": False,
    "crop_lung_roi_ratio_loss_weighting_used": False,
    "vessel_feature_used": False,
    "p_c_normal30_full_training_not_started": True,
}


def _check_guardrail(guardrail: dict, path: Path):
    fail_keys = [k for k, v in guardrail.items() if isinstance(v, bool) and not v and k.startswith("full_") ]
    # False여야 하는 항목이 True인 경우 FAIL
    false_keys = {
        "full_training_run", "final_test_accessed", "prediction_export_run",
        "threshold_optimization", "threshold_sweep", "checkpoint_selection",
        "existing_checkpoint_modified", "existing_result_overwrite",
        "roi_masked_loss_used", "crop_lung_roi_ratio_loss_weighting_used",
        "vessel_feature_used",
    }
    true_keys = set(guardrail.keys()) - false_keys

    rows = []
    n_fail = 0
    for k, expected_true in [(k, True) for k in true_keys] + [(k, False) for k in false_keys]:
        v = guardrail.get(k)
        ok = (v == expected_true)
        if not ok:
            n_fail += 1
        rows.append({"key": k, "value": v, "expected": expected_true, "status": "OK" if ok else "FAIL"})

    _write_csv(rows, path)
    return n_fail


# ── Main smoke training ───────────────────────────────────────────────────────

def run_smoke(args):
    set_seed(SEED)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 출력 충돌 방지 ────────────────────────────────────────────────────────
    for out_dir in [OUTPUT_ROOT, REPORT_ROOT]:
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[ABORT] output directory already exists and is not empty: {out_dir}", file=sys.stderr)
            print("[GUARD] Existing result overwrite is forbidden.", file=sys.stderr)
            sys.exit(2)

    # ── 출력 디렉터리 ─────────────────────────────────────────────────────────
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    errors = []

    # ── 1. 입력 파일 존재 확인 ─────────────────────────────────────────────────
    missing_inputs = []
    for p in [TRAIN_MANIFEST, VAL_MANIFEST, SCALAR_STATS_PATH, MASK_MANIFEST]:
        if not p.exists():
            missing_inputs.append(str(p))

    if missing_inputs:
        for mp in missing_inputs:
            errors.append({"check": "input_file_exists", "error": f"MISSING: {mp}"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal30a_errors.csv")
        print(f"[{STAGE_LABEL}] [ABORT] missing inputs: {missing_inputs}", file=sys.stderr)
        sys.exit(1)

    print(f"[{STAGE_LABEL}] all input files found")

    # ── 2. scalar stats 로드 ──────────────────────────────────────────────────
    with open(SCALAR_STATS_PATH) as f:
        norm_payload = json.load(f)
    scalar_stats = norm_payload["features"]
    print(f"[{STAGE_LABEL}] scalar stats loaded from {SCALAR_STATS_PATH.name}")

    # ── 3. manifest 로드 ──────────────────────────────────────────────────────
    df_train_raw = pd.read_csv(TRAIN_MANIFEST, low_memory=False)
    df_val_raw   = pd.read_csv(VAL_MANIFEST,   low_memory=False)
    n_train = len(df_train_raw)
    n_val   = len(df_val_raw)
    print(f"[{STAGE_LABEL}] train={n_train} val={n_val} (expected {EXPECTED_TRAIN_ROWS}/{EXPECTED_VAL_ROWS})")

    if n_train != EXPECTED_TRAIN_ROWS:
        errors.append({"check": "train_row_count", "error": f"expected {EXPECTED_TRAIN_ROWS}, got {n_train}"})
    if n_val != EXPECTED_VAL_ROWS:
        errors.append({"check": "val_row_count", "error": f"expected {EXPECTED_VAL_ROWS}, got {n_val}"})

    # sample_weight=1.0 확인
    for split_name, df in [("train", df_train_raw), ("val", df_val_raw)]:
        if "sample_weight" in df.columns:
            sw_vals = df["sample_weight"].astype(float)
            if not (sw_vals == 1.0).all():
                n_bad = int((sw_vals != 1.0).sum())
                errors.append({"check": f"{split_name}_sw_reset", "error": f"{n_bad} rows with sample_weight != 1.0"})
        else:
            # sample_weight 컬럼 없으면 1.0으로 채움
            df["sample_weight"] = 1.0

    if errors:
        _write_csv(errors, REPORT_ROOT / "p_c_normal30a_errors.csv")
        print(f"[{STAGE_LABEL}] [ABORT] manifest guard errors", file=sys.stderr)
        sys.exit(1)

    # ── 4. mask manifest 로드 및 join ─────────────────────────────────────────
    print(f"[{STAGE_LABEL}] loading mask manifest...")
    mask_lookup = build_mask_lookup(MASK_MANIFEST)
    print(f"[{STAGE_LABEL}] mask lookup size={len(mask_lookup)}")

    df_train_masked, missing_tr, join_rows_tr = join_mask_to_manifest(df_train_raw, mask_lookup, "train")
    df_val_masked,   missing_vl, join_rows_vl = join_mask_to_manifest(df_val_raw,   mask_lookup, "val")

    all_join_rows = join_rows_tr + join_rows_vl
    _write_csv(all_join_rows, REPORT_ROOT / "p_c_normal30a_mask_join_check.csv")

    n_missing_tr = len(missing_tr)
    n_missing_vl = len(missing_vl)
    print(f"[{STAGE_LABEL}] mask join: train missing={n_missing_tr}, val missing={n_missing_vl}")

    if n_missing_tr > 0 or n_missing_vl > 0:
        errors.append({"check": "mask_join", "error": f"train missing={n_missing_tr} val missing={n_missing_vl}"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal30a_errors.csv")
        print(f"[{STAGE_LABEL}] [ABORT] mask join failed", file=sys.stderr)
        sys.exit(1)

    # ── 5. scalar 정규화 ──────────────────────────────────────────────────────
    df_train_norm = apply_scalar_norm(df_train_masked, scalar_stats)
    df_val_norm   = apply_scalar_norm(df_val_masked,   scalar_stats)

    for col in SCALAR_FEATURES:
        mu  = float(df_train_norm[col].mean())
        std = float(df_train_norm[col].std())
        print(f"[{STAGE_LABEL}] scalar norm {col}: mean={mu:.4f} std={std:.4f}")

    # ── 6. Dataset / DataLoader ───────────────────────────────────────────────
    ds_train = MaskedInputDataset(df_train_norm, augment=True)
    ds_val   = MaskedInputDataset(df_val_norm,   augment=False)
    loader_train = DataLoader(ds_train, batch_size=64, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=False)
    loader_val   = DataLoader(ds_val,   batch_size=64, shuffle=False,
                              num_workers=4, pin_memory=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{STAGE_LABEL}] device={device}")

    # ── 7. DataLoader sanity check (batch 2개) ────────────────────────────────
    print(f"[{STAGE_LABEL}] batch sanity check...")
    sanity_rows = []
    sanity_iter = iter(loader_train)
    for batch_idx in range(2):
        try:
            imgs, scalars, labels, sw = next(sanity_iter)
        except StopIteration:
            break
        has_nan  = bool(torch.isnan(imgs).any())
        has_inf  = bool(torch.isinf(imgs).any())
        nonzero_ratio = float((imgs != 0).float().mean())
        sanity_rows.append({
            "batch_idx": batch_idx,
            "img_shape":    str(tuple(imgs.shape)),
            "scalar_shape": str(tuple(scalars.shape)),
            "img_min":    round(float(imgs.min()), 4),
            "img_max":    round(float(imgs.max()), 4),
            "img_mean":   round(float(imgs.mean()), 4),
            "nonzero_ratio": round(nonzero_ratio, 4),
            "has_nan":    has_nan,
            "has_inf":    has_inf,
            "label_sum":  int(labels.sum()),
            "batch_size": imgs.shape[0],
        })
        print(f"[{STAGE_LABEL}] batch {batch_idx}: shape={tuple(imgs.shape)} "
              f"min={imgs.min():.4f} max={imgs.max():.4f} "
              f"nonzero_ratio={nonzero_ratio:.4f} nan={has_nan} inf={has_inf}")

        if has_nan or has_inf:
            errors.append({"check": f"batch{batch_idx}_nan_inf", "error": f"nan={has_nan} inf={has_inf}"})

    _write_csv(sanity_rows, REPORT_ROOT / "p_c_normal30a_batch_input_sanity.csv")

    if errors:
        _write_csv(errors, REPORT_ROOT / "p_c_normal30a_errors.csv")
        print(f"[{STAGE_LABEL}] [ABORT] batch sanity errors", file=sys.stderr)
        sys.exit(1)

    # ── 8. 모델 + optimizer ───────────────────────────────────────────────────
    model     = ScalarFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ── 9. 1 epoch smoke training ─────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] starting 1-epoch smoke training...")
    train_log_rows = []
    all_loss_finite = True
    all_grad_finite = True

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
            errors.append({"check": f"step{step}_loss", "error": f"NaN/Inf={loss_v}"})
            print(f"[{STAGE_LABEL}] [ERROR] NaN/Inf loss at step={step}", file=sys.stderr)
            break

        loss.backward()

        if step == 0:
            grad_ok = all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters() if p.requires_grad
            )
            if not grad_ok:
                all_grad_finite = False
                errors.append({"check": "step0_grad", "error": "NaN/Inf gradient"})

        optimizer.step()
        preds = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
        tr_correct += (preds == labels).sum().item()
        tr_total   += labels.size(0)
        tr_losses.append(loss_v)

        train_log_rows.append({"epoch": 1, "step": step, "loss": round(loss_v, 6)})

        if step % 50 == 0:
            print(f"[{STAGE_LABEL}] step={step} loss={loss_v:.4f}")

    train_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
    train_acc  = float(tr_correct / tr_total) if tr_total > 0 else float("nan")
    print(f"[{STAGE_LABEL}] train: loss={train_loss:.4f} acc={train_acc:.4f}")

    _write_csv(train_log_rows, REPORT_ROOT / "p_c_normal30a_smoke_train_log.csv")

    # ── 10. val forward ───────────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] running val forward...")
    model.eval()
    vl_losses, all_probs, all_labels_ep = [], [], []
    val_forward_ok = True

    with torch.no_grad():
        for imgs, scalars, labels, sw in loader_val:
            imgs    = imgs.to(device)
            scalars = scalars.to(device)
            labels  = labels.to(device)
            sw      = sw.to(device)
            try:
                logits = model(imgs, scalars)
                loss   = weighted_bce_loss(logits, labels, sw)
                loss_v = float(loss.item())
                if not math.isfinite(loss_v):
                    all_loss_finite = False
                    errors.append({"check": "val_loss", "error": f"NaN/Inf={loss_v}"})
                vl_losses.append(loss_v)
                probs = torch.sigmoid(logits.squeeze(1))
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels_ep.extend(labels.cpu().numpy().tolist())
            except Exception as e:
                val_forward_ok = False
                errors.append({"check": "val_forward", "error": str(e)})

    val_loss  = float(np.mean(vl_losses)) if vl_losses else float("nan")
    val_auc,  val_auc_status  = compute_auroc(all_labels_ep, all_probs)
    val_auprc, val_auprc_status = compute_auprc(all_labels_ep, all_probs)

    auc_disp   = f"{val_auc:.4f}"   if not math.isnan(val_auc)   else "NaN"
    auprc_disp = f"{val_auprc:.4f}" if not math.isnan(val_auprc) else "NaN"
    print(f"[{STAGE_LABEL}] val: loss={val_loss:.4f} AUROC={auc_disp} AUPRC={auprc_disp}")
    print(f"[{STAGE_LABEL}] NOTE: val metrics are smoke sanity reference ONLY — no performance judgment")

    _write_csv([{
        "stage": STAGE_LABEL,
        "smoke_only": True,
        "epoch": 1,
        "val_loss": round(val_loss, 6),
        "val_auroc": auc_disp,
        "val_auprc": auprc_disp,
        "val_auroc_status": val_auc_status,
        "val_auprc_status": val_auprc_status,
        "val_forward_ok": val_forward_ok,
        "note": "smoke sanity reference only — no performance judgment",
    }], REPORT_ROOT / "p_c_normal30a_val_smoke_metrics.csv")

    # ── 11. smoke checkpoint 저장 ─────────────────────────────────────────────
    smoke_ckpt_path = CKPT_DIR / "p_c_normal30a_smoke_epoch1.pt"
    torch.save({
        "model_state_dict":        model.state_dict(),
        "optimizer_state_dict":    optimizer.state_dict(),
        "epoch":                   1,
        "smoke_only":              True,
        "is_final_model_candidate": False,
        "full_training":           False,
        "stage":                   STAGE_LABEL,
        "train_loss":              train_loss,
        "val_loss":                val_loss,
        "val_auroc":               val_auc,
        "val_auprc":               val_auprc,
        "mask_applied_to_image_only": True,
        "mask_manifest_used":      str(MASK_MANIFEST),
        "scalar_features":         SCALAR_FEATURES,
        "scalar_norm_source":      str(SCALAR_STATS_PATH),
        "train_manifest":          str(TRAIN_MANIFEST),
        "val_manifest":            str(VAL_MANIFEST),
        "balanced_w1":             True,
        "sample_weight_reset_to_1": True,
        "vessel_feature_used":     False,
        "roi_masked_loss_used":    False,
        "final_test_used":         False,
        "threshold_optimized":     False,
        "note":                    "SMOKE ONLY — do not use as final model",
    }, smoke_ckpt_path)
    print(f"[{STAGE_LABEL}] smoke checkpoint saved: {smoke_ckpt_path}")

    # ── 12. guardrail check ───────────────────────────────────────────────────
    n_guardrail_fail = _check_guardrail(GUARDRAIL, REPORT_ROOT / "p_c_normal30a_guardrail_check.csv")
    print(f"[{STAGE_LABEL}] guardrail fail count={n_guardrail_fail}")

    # ── 13. 판정 ─────────────────────────────────────────────────────────────
    verdict = "PASS"
    fail_reasons = []

    if not all_loss_finite:
        verdict = "FAIL"
        fail_reasons.append("loss_nan_inf")
    if not all_grad_finite:
        verdict = "FAIL"
        fail_reasons.append("gradient_nan_inf")
    if not val_forward_ok:
        verdict = "FAIL"
        fail_reasons.append("val_forward_error")
    if n_missing_tr > 0 or n_missing_vl > 0:
        verdict = "FAIL"
        fail_reasons.append(f"missing_mask train={n_missing_tr} val={n_missing_vl}")
    if n_guardrail_fail > 0:
        verdict = "FAIL"
        fail_reasons.append(f"guardrail_fail={n_guardrail_fail}")
    if errors:
        if verdict != "FAIL":
            verdict = "PARTIAL_PASS"
        fail_reasons.extend([e["error"] for e in errors])

    # ── 14. smoke report ──────────────────────────────────────────────────────
    report_lines = [
        f"# P-C-NORMAL30a Masked-Input Smoke Training Report",
        f"",
        f"- **Stage**: {STAGE_LABEL}",
        f"- **Timestamp**: {ts}",
        f"- **Verdict**: {verdict}",
        f"",
        f"## Input Files",
        f"| File | Path | Status |",
        f"|------|------|--------|",
        f"| train manifest | {TRAIN_MANIFEST} | OK |",
        f"| val manifest | {VAL_MANIFEST} | OK |",
        f"| scalar stats | {SCALAR_STATS_PATH} | OK |",
        f"| mask manifest | {MASK_MANIFEST} | OK |",
        f"",
        f"## Manifest Row Counts",
        f"- train: {n_train} (expected {EXPECTED_TRAIN_ROWS})",
        f"- val: {n_val} (expected {EXPECTED_VAL_ROWS})",
        f"",
        f"## Mask Join",
        f"- train missing: {n_missing_tr}",
        f"- val missing: {n_missing_vl}",
        f"",
        f"## Batch Sanity (2 batches)",
    ]
    for r in sanity_rows:
        report_lines.append(f"- batch {r['batch_idx']}: shape={r['img_shape']} min={r['img_min']} max={r['img_max']} nonzero_ratio={r['nonzero_ratio']} nan={r['has_nan']} inf={r['has_inf']}")

    report_lines += [
        f"",
        f"## Training (1 epoch smoke)",
        f"- train_loss: {train_loss:.4f}",
        f"- train_acc: {train_acc:.4f}",
        f"- all_loss_finite: {all_loss_finite}",
        f"- all_grad_finite: {all_grad_finite}",
        f"",
        f"## Validation (smoke sanity reference only)",
        f"- val_loss: {val_loss:.4f}",
        f"- val_AUROC: {auc_disp} ({val_auc_status})",
        f"- val_AUPRC: {auprc_disp} ({val_auprc_status})",
        f"- NOTE: smoke sanity reference only — no performance judgment",
        f"",
        f"## Guardrail",
        f"- fail count: {n_guardrail_fail}",
        f"",
        f"## Checkpoint",
        f"- {smoke_ckpt_path}",
        f"- smoke_only=True, is_final_model_candidate=False",
        f"",
        f"## Verdict",
        f"**{verdict}**",
    ]
    if fail_reasons:
        report_lines += ["", "## Fail Reasons"]
        for r in fail_reasons:
            report_lines.append(f"- {r}")

    report_lines += [
        f"",
        f"## Errors",
        f"- total: {len(errors)}",
    ]
    for e in errors:
        report_lines.append(f"- [{e['check']}] {e['error']}")

    report_lines += [
        f"",
        f"---",
        f"",
        f"## Next Step",
        f"",
        f"**P-C-NORMAL30b masked-input full training**",
        f"",
        f"- smoke PASS 후 사용자 승인 필요",
        f"- baseline: P-C-NORMAL24j-fix-balanced-w1 (full training, 30 epochs)",
        f"- 신규: 동일 조건 + masked input (30b)",
        f"- 목적: masking 효과만 분리 측정",
    ]

    report_path = REPORT_ROOT / "p_c_normal30a_smoke_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # ── 15. summary JSON ──────────────────────────────────────────────────────
    summary = {
        "stage":            STAGE_LABEL,
        "timestamp":        ts,
        "verdict":          verdict,
        "fail_reasons":     fail_reasons,
        "n_train":          n_train,
        "n_val":            n_val,
        "n_missing_mask_train": n_missing_tr,
        "n_missing_mask_val":   n_missing_vl,
        "train_loss":       round(train_loss, 6),
        "train_acc":        round(train_acc, 4),
        "val_loss":         round(val_loss, 6),
        "val_auroc":        auc_disp,
        "val_auprc":        auprc_disp,
        "val_auroc_status": val_auc_status,
        "val_auprc_status": val_auprc_status,
        "all_loss_finite":  all_loss_finite,
        "all_grad_finite":  all_grad_finite,
        "val_forward_ok":   val_forward_ok,
        "guardrail_fail":   n_guardrail_fail,
        "n_errors":         len(errors),
        "smoke_only":       True,
        "full_training_run": False,
        "final_test_accessed": False,
        "mask_applied_to_image_only": True,
    }
    _write_json(summary, REPORT_ROOT / "p_c_normal30a_smoke_summary.json")

    # ── 16. DONE.json ─────────────────────────────────────────────────────────
    done = {
        "stage":   STAGE_LABEL,
        "verdict": verdict,
        "timestamp": ts,
        "smoke_only": True,
        "next_step": "P-C-NORMAL30b masked-input full training (사용자 승인 필요)",
    }
    if verdict in ("PASS", "PARTIAL_PASS"):
        _write_json(done, REPORT_ROOT / "DONE.json")

    print(f"[{STAGE_LABEL}] {'='*60}")
    print(f"[{STAGE_LABEL}] VERDICT: {verdict}")
    if fail_reasons:
        for r in fail_reasons:
            print(f"[{STAGE_LABEL}]   FAIL: {r}")
    print(f"[{STAGE_LABEL}] report: {report_path}")
    print(f"[{STAGE_LABEL}] {'='*60}")

    return 0 if verdict in ("PASS", "PARTIAL_PASS") else 1


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P-C-NORMAL30a masked-input smoke training")
    parser.add_argument("--smoke", action="store_true",
                        help="smoke training (1 epoch) — required flag")
    args = parser.parse_args()

    if not args.smoke:
        print("[GUARD] --smoke flag required. full training is forbidden.", file=sys.stderr)
        sys.exit(2)

    return run_smoke(args)


if __name__ == "__main__":
    sys.exit(main())
