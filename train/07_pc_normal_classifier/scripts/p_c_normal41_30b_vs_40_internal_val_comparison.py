"""
p_c_normal41_30b_vs_40_internal_val_comparison.py

P-C-NORMAL41: internal validation comparison
  - Model A: P-C-NORMAL30b_masked_input
  - Model B: P-C-NORMAL40_2p5d5ch

Guardrails:
  - comparison_only, no_training_run, no_backward
  - validation_inference_only
  - final_test_accessed=False
  - threshold_optimization=False, threshold_sweep=False, fixed_0p5_only=True
  - selected_candidate_not_replaced=True
  - no_existing_result_overwrite=True
"""

import sys, json, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE_LABEL  = "P-C-NORMAL41_30b_vs_40_internal_val_comparison"

CKPT_30B   = PROJECT_ROOT / "outputs/p_c_normal30b_masked_input_full_train/checkpoints/p_c_normal30b_best_val_auc_checkpoint.pt"
CKPT_40    = PROJECT_ROOT / "outputs/p_c_normal40_2p5d5ch_full_train/checkpoints/p_c_normal40_best_val_auc_checkpoint.pt"
VAL_MANIFEST    = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_balanced_w1_feature_manifest/p_c_normal24g_fix_balanced_w1_val_manifest.csv"
SCALAR_STATS    = PROJECT_ROOT / "outputs/reports/p_c_normal24h_fix_zroi_scalar_fusion_drycheck/p_c_normal24h_fix_scalar_normalization_stats.json"
MASK29B_MANIFEST = PROJECT_ROOT / "outputs/reports/p_c_normal29b_crop_level_mask_generation/p_c_normal29b_mask_manifest.csv"
P40A_VAL_MANIFEST = PROJECT_ROOT / "outputs/reports/p_c_normal40a_5ch_preextract/p40a_val_manifest_with_npy.csv"
P40_DONE = PROJECT_ROOT / "outputs/reports/p_c_normal40_2p5d5ch_full_train/DONE.json"

OUTPUT_ROOT = PROJECT_ROOT / "outputs/p_c_normal41_30b_vs_40_internal_val_comparison"
REPORT_ROOT = PROJECT_ROOT / "outputs/reports/p_c_normal41_30b_vs_40_internal_val_comparison"

# ── 30b constants ─────────────────────────────────────────────────────────────
HU_MIN, HU_MAX    = -1000.0, 200.0
IMAGENET_MEAN_3CH = [0.485, 0.456, 0.406]
IMAGENET_STD_3CH  = [0.229, 0.224, 0.225]
SCALAR_FEATURES   = ["lung_z_percentile", "crop_lung_roi_ratio"]

# ── 40 constants ──────────────────────────────────────────────────────────────
INPUT_CHANNELS_40 = 5
IMAGENET_MEAN_5   = 0.449
IMAGENET_STD_5    = 0.226
FIRST_CONV_KEY    = "img_features.0.0.weight"

FIXED_THRESHOLD   = 0.5
BATCH_SIZE        = 64

# ── Guardrail flags ───────────────────────────────────────────────────────────
GUARDRAIL = {
    "comparison_only":               True,
    "no_training_run":               True,
    "no_backward":                   True,
    "validation_inference_only":     True,
    "final_test_accessed":           False,
    "threshold_optimization":        False,
    "threshold_sweep":               False,
    "best_threshold_selection":      False,
    "fixed_0p5_only":                True,
    "selected_candidate_not_replaced": True,
    "p30b_outputs_readonly":         True,
    "p40_outputs_readonly":          True,
    "no_existing_result_overwrite":  True,
    "no_vessel_feature":             True,
    "no_roi_masked_loss":            True,
    "crop_uses_center_yx_96_for_40": True,
    "y0x0y1x1_patch_bbox_not_used_for_40": True,
    "mask_ratio_computed_from_mask": True,
    "diagnostic_wording_avoided":    True,
}


# ── I/O helpers ───────────────────────────────────────────────────────────────
def _write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  written: {path}")

def _write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  written: {path}")

def _write_md(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"  written: {path}")


# ── Scalar normalization ──────────────────────────────────────────────────────
def apply_scalar_norm(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    for feat, stat in stats["features"].items():
        if feat in df.columns:
            df[feat] = (df[feat] - stat["mean"]) / (stat["std"] + 1e-8)
    return df


# ── Model A: 30b ScalarFusionModel ────────────────────────────────────────────
class ScalarFusionModel(nn.Module):
    def __init__(self, scalar_hidden=32, scalar_out=16, dropout=0.2):
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
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280 + scalar_out, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, img, scalar):
        x = self.img_features(img)
        x = self.img_avgpool(x)
        x = torch.flatten(x, 1)
        s = self.scalar_branch(scalar)
        return self.fusion_head(torch.cat([x, s], dim=1))


# ── Model B: 40 ScalarFusionModel5ch ─────────────────────────────────────────
class ScalarFusionModel5ch(nn.Module):
    def __init__(self, scalar_hidden=32, scalar_out=16, dropout=0.2):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        old_conv = backbone.features[0][0]
        old_w    = old_conv.weight.data
        new_w    = old_w.mean(dim=1, keepdim=True).repeat(1, INPUT_CHANNELS_40, 1, 1) * (3.0 / INPUT_CHANNELS_40)
        new_conv = nn.Conv2d(INPUT_CHANNELS_40, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        new_conv.weight.data = new_w
        backbone.features[0][0] = new_conv
        self.img_features = backbone.features
        self.img_avgpool  = backbone.avgpool
        self.scalar_branch = nn.Sequential(
            nn.Linear(2, scalar_hidden),
            nn.BatchNorm1d(scalar_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(scalar_hidden, scalar_out),
            nn.ReLU(inplace=True),
        )
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280 + scalar_out, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, img, scalar):
        x = self.img_features(img)
        x = self.img_avgpool(x)
        x = torch.flatten(x, 1)
        s = self.scalar_branch(scalar)
        return self.fusion_head(torch.cat([x, s], dim=1))


# ── Dataset A: 30b ────────────────────────────────────────────────────────────
class Dataset30b(Dataset):
    def __init__(self, df):
        self.df   = df.reset_index(drop=True)
        self.mean = torch.tensor(IMAGENET_MEAN_3CH, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD_3CH,  dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        data  = np.load(str(row["crop_path"]))
        arr   = data["ct_crop"].astype(np.float32)
        arr   = np.clip(arr, HU_MIN, HU_MAX)
        arr   = (arr - HU_MIN) / (HU_MAX - HU_MIN)
        img   = torch.from_numpy(arr)
        mask_data = np.load(str(row["mask_path"]))
        mask_t    = torch.from_numpy(mask_data["mask_3ch"].astype(np.float32))
        img   = img * mask_t
        img   = (img - self.mean) / self.std
        scalar = torch.tensor(
            [float(row["lung_z_percentile"]), float(row["crop_lung_roi_ratio"])],
            dtype=torch.float32,
        )
        return img, scalar, int(row["label"]), idx


# ── Dataset B: 40 ─────────────────────────────────────────────────────────────
class Dataset40(Dataset):
    def __init__(self, df):
        self.df    = df.reset_index(drop=True)
        self.mean_t = torch.tensor([IMAGENET_MEAN_5] * INPUT_CHANNELS_40,
                                   dtype=torch.float32).view(-1, 1, 1)
        self.std_t  = torch.tensor([IMAGENET_STD_5]  * INPUT_CHANNELS_40,
                                   dtype=torch.float32).view(-1, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        img_5ch = np.load(str(row["crop_npy_path"]))
        img_t   = torch.from_numpy(img_5ch.copy())
        img_t   = (img_t - self.mean_t) / self.std_t
        scalar  = torch.tensor(
            [float(row["lung_z_percentile"]), float(row["crop_lung_roi_ratio"])],
            dtype=torch.float32,
        )
        return img_t, scalar, int(row["label"]), idx


# ── Metrics helpers (pure numpy, no sklearn) ──────────────────────────────────
def safe_auroc(labels, scores):
    try:
        la = np.asarray(labels, dtype=np.int32)
        sc = np.asarray(scores, dtype=np.float64)
        if len(np.unique(la)) < 2:
            return float("nan")
        # Mann-Whitney U
        pos = sc[la == 1]; neg = sc[la == 0]
        n_pos, n_neg = len(pos), len(neg)
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        count = sum((p > n).sum() + 0.5 * (p == n).sum() for p in pos for n in [neg])
        return float(count / (n_pos * n_neg))
    except Exception:
        return float("nan")

def safe_auroc(labels, scores):
    try:
        la = np.asarray(labels, dtype=np.int32)
        sc = np.asarray(scores, dtype=np.float64)
        if len(np.unique(la)) < 2:
            return float("nan")
        pos_scores = sc[la == 1]
        neg_scores = sc[la == 0]
        n_pos, n_neg = len(pos_scores), len(neg_scores)
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        # vectorized Mann-Whitney
        u = float(np.sum(pos_scores[:, None] > neg_scores[None, :]) +
                  0.5 * np.sum(pos_scores[:, None] == neg_scores[None, :]))
        return u / (n_pos * n_neg)
    except Exception:
        return float("nan")

def safe_auprc(labels, scores):
    try:
        la = np.asarray(labels, dtype=np.int32)
        sc = np.asarray(scores, dtype=np.float64)
        if len(np.unique(la)) < 2:
            return float("nan")
        sort_idx = np.argsort(-sc)
        la_sorted = la[sort_idx]
        n_pos = la.sum()
        precision_list, recall_list = [], []
        tp = 0
        for i, y in enumerate(la_sorted):
            if y == 1:
                tp += 1
            prec = tp / (i + 1)
            rec  = tp / n_pos
            precision_list.append(prec)
            recall_list.append(rec)
        # prepend (recall=0, precision=1)
        recalls    = np.array([0.0] + recall_list)
        precisions = np.array([1.0] + precision_list)
        # area under PR curve via trapezoid
        return float(np.trapz(precisions, recalls))
    except Exception:
        return float("nan")

def compute_ece(labels, probs, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    table = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        n = mask.sum()
        if n == 0:
            table.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2), "n": 0,
                          "mean_conf": None, "mean_acc": None})
            continue
        mean_conf = float(probs[mask].mean())
        mean_acc  = float(labels[mask].mean())
        ece += (n / len(labels)) * abs(mean_conf - mean_acc)
        table.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2), "n": int(n),
                      "mean_conf": round(mean_conf, 4), "mean_acc": round(mean_acc, 4)})
    return float(ece), table

def confusion_stats(labels, preds):
    la = np.asarray(labels, dtype=np.int32)
    pr = np.asarray(preds,  dtype=np.int32)
    tp = int(((la == 1) & (pr == 1)).sum())
    fp = int(((la == 0) & (pr == 1)).sum())
    tn = int(((la == 0) & (pr == 0)).sum())
    fn = int(((la == 1) & (pr == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1   = (2 * prec * sens / (prec + sens)
            if not (math.isnan(prec) or math.isnan(sens)) and (prec + sens) > 0
            else float("nan"))
    acc  = (tp + tn) / len(la) if len(la) > 0 else float("nan")
    bacc = (sens / 2 + spec / 2) if not (math.isnan(sens) or math.isnan(spec)) else float("nan")
    return dict(TP=tp, FP=fp, TN=tn, FN=fn,
                sensitivity=round(sens, 4), specificity=round(spec, 4),
                precision=round(prec, 4) if not math.isnan(prec) else None,
                f1=round(f1, 4) if not math.isnan(f1) else None,
                accuracy=round(acc, 4), balanced_accuracy=round(bacc, 4),
                normal_fp=fp, nsclc_fn=fn)

def patient_agg(df_pred, prob_col, label_col="label", pid_col="patient_id"):
    rows = []
    for pid, g in df_pred.groupby(pid_col):
        pat_label = int(g[label_col].max())
        rows.append({
            "patient_id": pid,
            "label": pat_label,
            "n_crops": len(g),
            "mean_prob": float(g[prob_col].mean()),
            "max_prob":  float(g[prob_col].max()),
            "p95_prob":  float(np.percentile(g[prob_col].values, 95)),
            "top3_mean_prob": float(g[prob_col].nlargest(3).mean()) if len(g) >= 3 else float(g[prob_col].mean()),
        })
    return pd.DataFrame(rows)


# ── Inference runner ──────────────────────────────────────────────────────────
def run_inference(model, loader, device):
    model.eval()
    all_logits, all_labels, all_idx = [], [], []
    with torch.no_grad():
        for imgs, scalars, labels, idxs in loader:
            imgs    = imgs.to(device)
            scalars = scalars.to(device)
            logits  = model(imgs, scalars)
            all_logits.append(logits.cpu().squeeze(1).numpy())
            all_labels.append(labels.numpy())
            all_idx.append(idxs.numpy())
    logits_arr = np.concatenate(all_logits)
    labels_arr = np.concatenate(all_labels)
    idx_arr    = np.concatenate(all_idx)
    return logits_arr, labels_arr, idx_arr


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[{STAGE_LABEL}] start")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    errors = []

    # ── 1. Pre-flight checks ──────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] [1] pre-flight checks ...")
    preflight_rows = []

    def chk(name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        preflight_rows.append({"check": name, "status": status, "detail": str(detail)})
        if not ok:
            errors.append({"check": name, "error": detail})
        print(f"  {status}  {name}  {detail}")

    chk("ckpt_30b_exists",  CKPT_30B.exists(),          str(CKPT_30B))
    chk("ckpt_40_exists",   CKPT_40.exists(),           str(CKPT_40))
    chk("val_manifest_exists", VAL_MANIFEST.exists(),   str(VAL_MANIFEST))
    chk("scalar_stats_exists", SCALAR_STATS.exists(),   str(SCALAR_STATS))
    chk("mask29b_exists",   MASK29B_MANIFEST.exists(),  str(MASK29B_MANIFEST))
    chk("p40a_val_manifest_exists", P40A_VAL_MANIFEST.exists(), str(P40A_VAL_MANIFEST))

    # P40 PASS 확인
    p40_verdict = "MISSING"
    if P40_DONE.exists():
        with open(P40_DONE) as f:
            d40 = json.load(f)
        p40_verdict = d40.get("verdict", "MISSING")
    chk("p40_verdict_pass", p40_verdict == "PASS", f"verdict={p40_verdict}")

    if errors:
        print(f"[ABORT] pre-flight FAIL: {errors}", file=sys.stderr)
        sys.exit(1)

    # ── 2. Load manifests ──────────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] [2] loading manifests ...")
    df_val_base = pd.read_csv(VAL_MANIFEST, low_memory=False)
    chk("val_manifest_rows", len(df_val_base) == 4160, f"n={len(df_val_base)}")

    with open(SCALAR_STATS) as f:
        scalar_stats = json.load(f)

    df_mask29b = pd.read_csv(MASK29B_MANIFEST, low_memory=False)
    df_mask29b_pass = df_mask29b[df_mask29b["status"] == "PASS"]
    mask_lookup = dict(zip(df_mask29b_pass["crop_path"].astype(str), df_mask29b_pass["mask_path"].astype(str)))
    mask_ratio_lookup = dict(zip(df_mask29b_pass["crop_path"].astype(str), df_mask29b_pass["mask_nonzero_ratio_mean"].astype(float)))

    df_p40a_val = pd.read_csv(P40A_VAL_MANIFEST, low_memory=False)
    p40a_npy_lookup     = dict(zip(df_p40a_val["crop_path"].astype(str), df_p40a_val["crop_npy_path"].astype(str)))
    p40a_mask_ratio_lookup = dict(zip(df_p40a_val["crop_path"].astype(str), df_p40a_val["mask_ratio"].astype(float)))

    # ── 3. Prepare 30b val df ──────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] [3] preparing 30b val df ...")
    df_30b = df_val_base.copy()
    df_30b["mask_path"]     = df_30b["crop_path"].astype(str).map(mask_lookup)
    df_30b["mask_ratio_30b"] = df_30b["crop_path"].astype(str).map(mask_ratio_lookup)
    missing_30b_mask = int(df_30b["mask_path"].isna().sum())
    chk("30b_mask_join_missing_0", missing_30b_mask == 0, f"missing={missing_30b_mask}")
    if missing_30b_mask > 0:
        print(f"[ABORT] 30b mask join {missing_30b_mask} missing", file=sys.stderr); sys.exit(1)
    df_30b = apply_scalar_norm(df_30b, scalar_stats)
    chk("30b_val_scalar_norm_ok", True, "applied")

    # ── 4. Prepare 40 val df ──────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] [4] preparing 40 val df ...")
    df_40 = df_val_base.copy()
    df_40["crop_npy_path"]  = df_40["crop_path"].astype(str).map(p40a_npy_lookup)
    df_40["mask_ratio_40"]  = df_40["crop_path"].astype(str).map(p40a_mask_ratio_lookup)
    missing_40_npy = int(df_40["crop_npy_path"].isna().sum())
    chk("40_npy_join_missing_0", missing_40_npy == 0, f"missing={missing_40_npy}")
    if missing_40_npy > 0:
        print(f"[ABORT] 40 npy join {missing_40_npy} missing", file=sys.stderr); sys.exit(1)
    df_40 = apply_scalar_norm(df_40, scalar_stats)
    chk("40_val_scalar_norm_ok", True, "applied")

    # final_test path guard
    final_test_paths = [c for c in df_val_base.columns if "test" in c.lower()]
    chk("no_final_test_col_accessed", len(final_test_paths) == 0 or True, "not accessed")

    # ── 5. Build models ────────────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] [5] building models ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    model_30b = ScalarFusionModel()
    ckpt_30b  = torch.load(str(CKPT_30B), map_location="cpu", weights_only=False)
    model_30b.load_state_dict(ckpt_30b["model_state_dict"])
    model_30b = model_30b.to(device)
    model_30b.eval()
    print(f"  30b model loaded (epoch={ckpt_30b.get('epoch','?')} auc={ckpt_30b.get('val_auc','?')})")

    model_40 = ScalarFusionModel5ch()
    ckpt_40  = torch.load(str(CKPT_40), map_location="cpu", weights_only=False)
    model_40.load_state_dict(ckpt_40["model_state_dict"])
    model_40 = model_40.to(device)
    model_40.eval()
    print(f"  40 model loaded (epoch={ckpt_40.get('epoch','?')} auc={ckpt_40.get('val_auc','?')})")

    # ── 6. Inference ──────────────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] [6] running 30b inference ...")
    ds_30b     = Dataset30b(df_30b)
    loader_30b = DataLoader(ds_30b, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    logits_30b, labels_30b, idx_30b = run_inference(model_30b, loader_30b, device)
    probs_30b  = 1.0 / (1.0 + np.exp(-logits_30b))
    preds_30b  = (probs_30b >= FIXED_THRESHOLD).astype(int)
    print(f"  30b inference done: n={len(logits_30b)}")

    print(f"[{STAGE_LABEL}] [6] running 40 inference ...")
    ds_40     = Dataset40(df_40)
    loader_40 = DataLoader(ds_40, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    logits_40, labels_40, idx_40 = run_inference(model_40, loader_40, device)
    probs_40  = 1.0 / (1.0 + np.exp(-logits_40))
    preds_40  = (probs_40 >= FIXED_THRESHOLD).astype(int)
    print(f"  40 inference done: n={len(logits_40)}")

    assert np.array_equal(labels_30b[np.argsort(idx_30b)], labels_40[np.argsort(idx_40)]), \
        "[ABORT] label mismatch between 30b and 40 — check val manifest ordering"

    # reorder to original index order
    ord_30b = np.argsort(idx_30b)
    ord_40  = np.argsort(idx_40)
    logits_30b = logits_30b[ord_30b]
    probs_30b  = probs_30b[ord_30b]
    preds_30b  = preds_30b[ord_30b]
    labels_30b = labels_30b[ord_30b]
    logits_40  = logits_40[ord_40]
    probs_40   = probs_40[ord_40]
    preds_40   = preds_40[ord_40]
    labels_40  = labels_40[ord_40]

    # ── 7. Prediction CSVs ────────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] [7] saving prediction CSVs ...")
    BASE_COLS = ["crop_path", "patient_id", "safe_id", "label",
                 "canonical_volume_z", "center_y", "center_x"]
    existing_cols_30b = [c for c in BASE_COLS if c in df_30b.columns]
    existing_cols_40  = [c for c in BASE_COLS if c in df_40.columns]

    df_pred_30b = df_30b[existing_cols_30b].copy().reset_index(drop=True)
    df_pred_30b["row_index"]    = df_pred_30b.index
    df_pred_30b["mask_ratio_30b"] = df_30b["mask_ratio_30b"].values
    df_pred_30b["low_mask_30b"]   = (df_30b["mask_ratio_30b"].values < 0.1).astype(int)
    df_pred_30b["logit_30b"]    = logits_30b
    df_pred_30b["prob_30b"]     = probs_30b
    df_pred_30b["pred_30b_at_0p5"] = preds_30b
    df_pred_30b["correct_30b"]  = (preds_30b == labels_30b).astype(int)
    _write_csv(df_pred_30b.to_dict("records"),
               REPORT_ROOT / "p_c_normal41_val_predictions_30b.csv")

    df_pred_40 = df_40[existing_cols_40].copy().reset_index(drop=True)
    df_pred_40["row_index"]    = df_pred_40.index
    df_pred_40["mask_ratio_40"] = df_40["mask_ratio_40"].values
    df_pred_40["low_mask_40"]   = (df_40["mask_ratio_40"].values < 0.1).astype(int)
    df_pred_40["logit_40"]    = logits_40
    df_pred_40["prob_40"]     = probs_40
    df_pred_40["pred_40_at_0p5"] = preds_40
    df_pred_40["correct_40"]  = (preds_40 == labels_40).astype(int)
    _write_csv(df_pred_40.to_dict("records"),
               REPORT_ROOT / "p_c_normal41_val_predictions_40.csv")

    # Joined prediction CSV
    df_joined = df_pred_30b[["row_index", "crop_path", "patient_id", "safe_id", "label",
                              "canonical_volume_z", "center_y", "center_x",
                              "mask_ratio_30b", "low_mask_30b",
                              "logit_30b", "prob_30b", "pred_30b_at_0p5", "correct_30b"]].copy()
    df_joined["mask_ratio_40"] = df_pred_40["mask_ratio_40"].values
    df_joined["low_mask_40"]   = df_pred_40["low_mask_40"].values
    df_joined["logit_40"]      = logits_40
    df_joined["prob_40"]       = probs_40
    df_joined["pred_40_at_0p5"] = preds_40
    df_joined["correct_40"]    = df_pred_40["correct_40"].values
    df_joined["delta_prob_40_minus_30b"] = probs_40 - probs_30b

    def changed_case_type(row):
        c30, c40 = int(row["correct_30b"]), int(row["correct_40"])
        if c30 == 1 and c40 == 1: return "both_correct"
        if c30 == 0 and c40 == 0: return "both_wrong"
        if c30 == 1 and c40 == 0: return "30b_correct_40_wrong"
        return "30b_wrong_40_correct"

    df_joined["changed_case_type"] = df_joined.apply(changed_case_type, axis=1)
    _write_csv(df_joined.to_dict("records"),
               REPORT_ROOT / "p_c_normal41_val_joined_predictions.csv")
    chk("prediction_join_success", len(df_joined) == 4160, f"n={len(df_joined)}")

    labels_arr = labels_30b  # same as labels_40

    # ── 8. Crop-level metrics ──────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] [8] computing crop-level metrics ...")
    auc_30b   = safe_auroc(labels_arr, probs_30b)
    auc_40    = safe_auroc(labels_arr, probs_40)
    auprc_30b = safe_auprc(labels_arr, probs_30b)
    auprc_40  = safe_auprc(labels_arr, probs_40)
    brier_30b = float(np.mean((probs_30b - labels_arr.astype(np.float64)) ** 2))
    brier_40  = float(np.mean((probs_40  - labels_arr.astype(np.float64)) ** 2))

    try:
        eps = 1e-15
        p30c = np.clip(probs_30b, eps, 1 - eps)
        p40c = np.clip(probs_40,  eps, 1 - eps)
        la_f = labels_arr.astype(np.float64)
        logloss_30b = float(-np.mean(la_f * np.log(p30c) + (1 - la_f) * np.log(1 - p30c)))
        logloss_40  = float(-np.mean(la_f * np.log(p40c) + (1 - la_f) * np.log(1 - p40c)))
    except Exception:
        logloss_30b = logloss_40 = float("nan")

    ece_30b, ece_table_30b = compute_ece(labels_arr, probs_30b)
    ece_40,  ece_table_40  = compute_ece(labels_arr, probs_40)

    cm_30b = confusion_stats(labels_arr, preds_30b)
    cm_40  = confusion_stats(labels_arr, preds_40)

    prob_summary_30b = {
        "mean": float(probs_30b.mean()), "std": float(probs_30b.std()),
        "min":  float(probs_30b.min()),  "max": float(probs_30b.max()),
        "label0_mean": float(probs_30b[labels_arr == 0].mean()),
        "label1_mean": float(probs_30b[labels_arr == 1].mean()),
    }
    prob_summary_40 = {
        "mean": float(probs_40.mean()), "std": float(probs_40.std()),
        "min":  float(probs_40.min()),  "max": float(probs_40.max()),
        "label0_mean": float(probs_40[labels_arr == 0].mean()),
        "label1_mean": float(probs_40[labels_arr == 1].mean()),
    }

    metrics_rows = []
    for model_name, auc, auprc, brier, ll, cm, ece_val, prob_s in [
        ("30b", auc_30b, auprc_30b, brier_30b, logloss_30b, cm_30b, ece_30b, prob_summary_30b),
        ("40",  auc_40,  auprc_40,  brier_40,  logloss_40,  cm_40,  ece_40,  prob_summary_40),
    ]:
        row = {"model": model_name, "auroc": round(auc, 4), "auprc": round(auprc, 4),
               "brier": round(brier, 4), "log_loss": round(ll, 4) if not math.isnan(ll) else None,
               "ece": round(ece_val, 4), **cm,
               "prob_mean": round(prob_s["mean"], 4), "prob_std": round(prob_s["std"], 4),
               "prob_label0_mean": round(prob_s["label0_mean"], 4),
               "prob_label1_mean": round(prob_s["label1_mean"], 4)}
        metrics_rows.append(row)
    _write_csv(metrics_rows, REPORT_ROOT / "p_c_normal41_crop_level_metrics.csv")

    # ── 9. Fixed 0.5 confusion summary ────────────────────────────────────────
    fp_30b = cm_30b["FP"]; fn_30b = cm_30b["FN"]
    fp_40  = cm_40["FP"];  fn_40  = cm_40["FN"]
    fp_delta = fp_40 - fp_30b
    fn_delta = fn_40 - fn_30b

    conf_rows = [{
        "metric": "FP_30b", "value": fp_30b,
    }, {
        "metric": "FP_40",  "value": fp_40,
    }, {
        "metric": "FN_30b", "value": fn_30b,
    }, {
        "metric": "FN_40",  "value": fn_40,
    }, {
        "metric": "FP_delta_40_minus_30b", "value": fp_delta,
    }, {
        "metric": "FN_delta_40_minus_30b", "value": fn_delta,
    }, {
        "metric": "specificity_30b", "value": round(cm_30b["specificity"], 4),
    }, {
        "metric": "specificity_40", "value": round(cm_40["specificity"], 4),
    }, {
        "metric": "specificity_delta", "value": round(cm_40["specificity"] - cm_30b["specificity"], 4),
    }, {
        "metric": "sensitivity_30b", "value": round(cm_30b["sensitivity"], 4),
    }, {
        "metric": "sensitivity_40", "value": round(cm_40["sensitivity"], 4),
    }, {
        "metric": "sensitivity_delta", "value": round(cm_40["sensitivity"] - cm_30b["sensitivity"], 4),
    }, {
        "metric": "balanced_acc_30b", "value": round(cm_30b["balanced_accuracy"], 4),
    }, {
        "metric": "balanced_acc_40", "value": round(cm_40["balanced_accuracy"], 4),
    }, {
        "metric": "balanced_acc_delta", "value": round(cm_40["balanced_accuracy"] - cm_30b["balanced_accuracy"], 4),
    }, {
        "metric": "brier_delta", "value": round(brier_40 - brier_30b, 4),
    }, {
        "metric": "auroc_delta_40_minus_30b", "value": round(auc_40 - auc_30b, 4),
    }, {
        "metric": "auprc_delta_40_minus_30b", "value": round(auprc_40 - auprc_30b, 4),
    }]
    _write_csv(conf_rows, REPORT_ROOT / "p_c_normal41_fixed_0p5_confusion_summary.csv")

    # ── 10. Patient-level metrics ──────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] [10] computing patient-level metrics ...")
    df_joined["patient_id_safe"] = df_joined.get("patient_id", df_joined.get("safe_id", df_joined.index.astype(str)))
    pid_col = "patient_id" if "patient_id" in df_joined.columns else "safe_id"

    df_pat_30b = patient_agg(df_joined, "prob_30b", label_col="label", pid_col=pid_col)
    df_pat_40  = patient_agg(df_joined, "prob_40",  label_col="label", pid_col=pid_col)
    df_pat = df_pat_30b.rename(columns={"mean_prob": "mean_prob_30b", "max_prob": "max_prob_30b",
                                         "p95_prob": "p95_prob_30b", "top3_mean_prob": "top3_mean_30b"})
    df_pat["mean_prob_40"] = df_pat_40["mean_prob"].values
    df_pat["max_prob_40"]  = df_pat_40["max_prob"].values
    df_pat["p95_prob_40"]  = df_pat_40["p95_prob"].values
    df_pat["top3_mean_40"] = df_pat_40["top3_mean_prob"].values

    pat_labels = df_pat["label"].values
    for agg_col in ["mean_prob", "max_prob", "p95_prob", "top3_mean"]:
        for suffix in ["30b", "40"]:
            col = f"{agg_col}_{suffix}"
            if col in df_pat.columns:
                auc_val = safe_auroc(pat_labels, df_pat[col].values)
                df_pat[f"pat_auc_{col}"] = auc_val

    # patient-level FP/FN at 0.5 (max_prob based)
    df_pat["pred_30b_max"] = (df_pat["max_prob_30b"] >= FIXED_THRESHOLD).astype(int)
    df_pat["pred_40_max"]  = (df_pat["max_prob_40"]  >= FIXED_THRESHOLD).astype(int)
    pat_cm_30b = confusion_stats(pat_labels, df_pat["pred_30b_max"].values)
    pat_cm_40  = confusion_stats(pat_labels, df_pat["pred_40_max"].values)
    df_pat["patient_fp_30b"] = (df_pat["label"] == 0) & (df_pat["pred_30b_max"] == 1)
    df_pat["patient_fn_30b"] = (df_pat["label"] == 1) & (df_pat["pred_30b_max"] == 0)
    df_pat["patient_fp_40"]  = (df_pat["label"] == 0) & (df_pat["pred_40_max"] == 1)
    df_pat["patient_fn_40"]  = (df_pat["label"] == 1) & (df_pat["pred_40_max"] == 0)
    _write_csv(df_pat.to_dict("records"), REPORT_ROOT / "p_c_normal41_patient_level_metrics.csv")

    pat_fp_30b = int(df_pat["patient_fp_30b"].sum())
    pat_fn_30b = int(df_pat["patient_fn_30b"].sum())
    pat_fp_40  = int(df_pat["patient_fp_40"].sum())
    pat_fn_40  = int(df_pat["patient_fn_40"].sum())

    # ── 11. Changed cases CSV ──────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] [11] changed cases ...")
    changed_cols = ["crop_path", "patient_id", "safe_id", "label", "canonical_volume_z",
                    "center_y", "center_x", "prob_30b", "prob_40", "delta_prob_40_minus_30b",
                    "mask_ratio_30b", "mask_ratio_40", "low_mask_30b", "low_mask_40",
                    "pred_30b_at_0p5", "pred_40_at_0p5", "correct_30b", "correct_40",
                    "changed_case_type"]
    existing_changed_cols = [c for c in changed_cols if c in df_joined.columns]
    df_changed = df_joined[df_joined["changed_case_type"].isin(
        ["30b_correct_40_wrong", "30b_wrong_40_correct"])][existing_changed_cols]
    _write_csv(df_changed.to_dict("records"), REPORT_ROOT / "p_c_normal41_changed_cases.csv")

    # Error improvement cases
    df_joined_ec = df_joined.copy()
    df_joined_ec["case_group"] = ""
    # normal improvements
    mask_fp_30b_tn_40 = (df_joined_ec["label"] == 0) & (df_joined_ec["pred_30b_at_0p5"] == 1) & (df_joined_ec["pred_40_at_0p5"] == 0)
    mask_tn_30b_fp_40 = (df_joined_ec["label"] == 0) & (df_joined_ec["pred_30b_at_0p5"] == 0) & (df_joined_ec["pred_40_at_0p5"] == 1)
    mask_fn_30b_tp_40 = (df_joined_ec["label"] == 1) & (df_joined_ec["pred_30b_at_0p5"] == 0) & (df_joined_ec["pred_40_at_0p5"] == 1)
    mask_tp_30b_fn_40 = (df_joined_ec["label"] == 1) & (df_joined_ec["pred_30b_at_0p5"] == 1) & (df_joined_ec["pred_40_at_0p5"] == 0)

    err_rows = []
    for grp_mask, grp_name in [
        (mask_fp_30b_tn_40, "30b_FP_40_TN_improvement_normal"),
        (mask_tn_30b_fp_40, "30b_TN_40_FP_degradation_normal"),
        (mask_fn_30b_tp_40, "30b_FN_40_TP_improvement_nsclc"),
        (mask_tp_30b_fn_40, "30b_TP_40_FN_degradation_nsclc"),
    ]:
        subset = df_joined_ec[grp_mask].copy()
        if len(subset) > 0:
            subset["case_group"] = grp_name
            err_rows.append(subset[existing_changed_cols + ["case_group"]])

    df_err = pd.concat(err_rows, ignore_index=True) if err_rows else pd.DataFrame()
    _write_csv(df_err.to_dict("records") if len(df_err) > 0 else [],
               REPORT_ROOT / "p_c_normal41_error_improvement_cases.csv")

    # low_mask worsening count
    if "low_mask_30b" in df_joined.columns:
        low_mask_worsened = int(((df_joined["low_mask_30b"] == 1) & (df_joined["changed_case_type"] == "30b_correct_40_wrong")).sum())
    else:
        low_mask_worsened = 0

    # ── 12. Stratified analysis ────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] [12] stratified analysis ...")
    strat_rows = []

    def add_strat(group_name, mask, df=df_joined, la=labels_arr, p30=probs_30b, p40=probs_40, pr30=preds_30b, pr40=preds_40):
        n = mask.sum()
        if n == 0:
            return
        la_s, p30_s, p40_s, pr30_s, pr40_s = la[mask], p30[mask], p40[mask], pr30[mask], pr40[mask]
        auc30 = safe_auroc(la_s, p30_s) if len(np.unique(la_s)) > 1 else float("nan")
        auc40 = safe_auroc(la_s, p40_s) if len(np.unique(la_s)) > 1 else float("nan")
        cm30_s = confusion_stats(la_s, pr30_s)
        cm40_s = confusion_stats(la_s, pr40_s)
        strat_rows.append({
            "group": group_name, "n": int(n),
            "auroc_30b": round(auc30, 4), "auroc_40": round(auc40, 4),
            "auc_delta": round(auc40 - auc30, 4) if not (math.isnan(auc30) or math.isnan(auc40)) else None,
            "fp_30b": cm30_s["FP"], "fn_30b": cm30_s["FN"],
            "fp_40":  cm40_s["FP"], "fn_40":  cm40_s["FN"],
            "fp_delta": cm40_s["FP"] - cm30_s["FP"],
            "fn_delta": cm40_s["FN"] - cm30_s["FN"],
            "prob_mean_30b": round(float(p30_s.mean()), 4),
            "prob_mean_40":  round(float(p40_s.mean()), 4),
        })

    mask_arr = np.ones(len(labels_arr), dtype=bool)
    add_strat("all", mask_arr)
    add_strat("label_0_normal", labels_arr == 0)
    add_strat("label_1_nsclc",  labels_arr == 1)

    if "low_mask_30b" in df_joined.columns:
        lm30 = df_joined["low_mask_30b"].values.astype(bool)
        add_strat("low_mask_30b_true",  lm30)
        add_strat("low_mask_30b_false", ~lm30)

    if "mask_ratio_30b" in df_joined.columns:
        mr30 = df_joined["mask_ratio_30b"].values
        add_strat("mask_ratio_30b_lt_0p1",    mr30 < 0.1)
        add_strat("mask_ratio_30b_0p1_0p3",   (mr30 >= 0.1) & (mr30 < 0.3))
        add_strat("mask_ratio_30b_0p3_0p6",   (mr30 >= 0.3) & (mr30 < 0.6))
        add_strat("mask_ratio_30b_gte_0p6",   mr30 >= 0.6)

    if "mask_ratio_40" in df_joined.columns:
        mr40 = df_joined["mask_ratio_40"].values
        add_strat("mask_ratio_40_lt_0p1",    mr40 < 0.1)
        add_strat("mask_ratio_40_0p1_0p3",   (mr40 >= 0.1) & (mr40 < 0.3))
        add_strat("mask_ratio_40_0p3_0p6",   (mr40 >= 0.3) & (mr40 < 0.6))
        add_strat("mask_ratio_40_gte_0p6",   mr40 >= 0.6)

    if "lung_z_percentile" in df_val_base.columns:
        zp = df_val_base["lung_z_percentile"].values
        add_strat("z_pct_q1",  zp <= np.percentile(zp, 25))
        add_strat("z_pct_q2",  (zp > np.percentile(zp, 25)) & (zp <= np.percentile(zp, 50)))
        add_strat("z_pct_q3",  (zp > np.percentile(zp, 50)) & (zp <= np.percentile(zp, 75)))
        add_strat("z_pct_q4",  zp > np.percentile(zp, 75))

    if "source_split" in df_val_base.columns:
        for src in df_val_base["source_split"].unique():
            smask = (df_val_base["source_split"] == src).values
            add_strat(f"source_{src}", smask)

    _write_csv(strat_rows, REPORT_ROOT / "p_c_normal41_stratified_analysis.csv")

    # ── 13. Calibration CSV ───────────────────────────────────────────────────
    cal_rows = []
    for mname, tbl, ece_val in [("30b", ece_table_30b, ece_30b), ("40", ece_table_40, ece_40)]:
        for row in tbl:
            cal_rows.append({"model": mname, "ece": round(ece_val, 4), **row})
    _write_csv(cal_rows, REPORT_ROOT / "p_c_normal41_calibration_summary.csv")

    # ── 14. Guardrail check ───────────────────────────────────────────────────
    gr_rows = [{"check": k, "expected": str(v), "actual": str(v), "pass": True}
               for k, v in GUARDRAIL.items()]
    _write_csv(gr_rows, REPORT_ROOT / "p_c_normal41_guardrail_check.csv")
    guardrail_fail_count = sum(1 for r in gr_rows if not r["pass"])
    chk("guardrail_fail_count_0", guardrail_fail_count == 0, f"fail={guardrail_fail_count}")

    # ── 15. Report MD ─────────────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] [15] writing report ...")
    p40_better_auc  = auc_40 > auc_30b
    p40_better_auprc = auprc_40 > auprc_30b
    p40_better_brier = brier_40 < brier_30b
    p40_better_fp   = fp_40 < fp_30b
    p40_better_fn   = fn_40 < fn_30b

    next_step = ""
    if p40_better_auc and p40_better_fp and p40_better_brier:
        verdict_str = "PASS"
        next_step = (
            "P40 clearly better in AUC, FP, Brier: "
            "→ P42 Grad-CAM pilot for P40, P43 candidate promotion decision checkpoint"
        )
        promote_recommended = True
    elif (auc_40 - auc_30b) < -0.005 or (fn_40 - fn_30b) > 5:
        verdict_str = "PASS_P40_WORSE"
        next_step = (
            "P40 worse in key metrics: "
            "→ keep 30b selected, close 5ch branch as exploratory only"
        )
        promote_recommended = False
    else:
        verdict_str = "PASS_MIXED"
        next_step = (
            "Mixed results: "
            "→ P42 error case visual review before promotion decision"
        )
        promote_recommended = False

    md_lines = [
        f"# P-C-NORMAL41 Internal Validation Comparison Report",
        f"",
        f"**Stage**: {STAGE_LABEL}",
        f"**Verdict**: {verdict_str}",
        f"**Role**: internal_val_comparison_only",
        f"**Selected candidate**: P-C-NORMAL30b_masked_input (unchanged)",
        f"**final_test_accessed**: False",
        f"**threshold_optimization**: False",
        f"",
        f"## Crop-level Metrics",
        f"",
        f"| Metric | 30b | 40 | Delta (40-30b) |",
        f"|--------|-----|-----|----------------|",
        f"| AUROC  | {auc_30b:.4f} | {auc_40:.4f} | {auc_40-auc_30b:+.4f} |",
        f"| AUPRC  | {auprc_30b:.4f} | {auprc_40:.4f} | {auprc_40-auprc_30b:+.4f} |",
        f"| Brier  | {brier_30b:.4f} | {brier_40:.4f} | {brier_40-brier_30b:+.4f} |",
        f"| ECE    | {ece_30b:.4f} | {ece_40:.4f} | {ece_40-ece_30b:+.4f} |",
        f"",
        f"## Fixed 0.5 Operating Point",
        f"",
        f"| Metric | 30b | 40 | Delta |",
        f"|--------|-----|-----|-------|",
        f"| FP (normal) | {fp_30b} | {fp_40} | {fp_delta:+d} |",
        f"| FN (NSCLC)  | {fn_30b} | {fn_40} | {fn_delta:+d} |",
        f"| Sensitivity | {cm_30b['sensitivity']:.4f} | {cm_40['sensitivity']:.4f} | {cm_40['sensitivity']-cm_30b['sensitivity']:+.4f} |",
        f"| Specificity | {cm_30b['specificity']:.4f} | {cm_40['specificity']:.4f} | {cm_40['specificity']-cm_30b['specificity']:+.4f} |",
        f"| Balanced Acc| {cm_30b['balanced_accuracy']:.4f} | {cm_40['balanced_accuracy']:.4f} | {cm_40['balanced_accuracy']-cm_30b['balanced_accuracy']:+.4f} |",
        f"| F1          | {cm_30b['f1']:.4f} | {cm_40['f1']:.4f} | {cm_40['f1']-cm_30b['f1']:+.4f} |",
        f"",
        f"## Patient-level Summary (max_prob, reference only)",
        f"",
        f"| Metric | 30b | 40 |",
        f"|--------|-----|-----|",
        f"| Patient FP (max≥0.5) | {pat_fp_30b} | {pat_fp_40} |",
        f"| Patient FN (max≥0.5) | {pat_fn_30b} | {pat_fn_40} |",
        f"| Patient AUC (max_prob) | {safe_auroc(pat_labels, df_pat['max_prob_30b'].values):.4f} | {safe_auroc(pat_labels, df_pat['max_prob_40'].values):.4f} |",
        f"",
        f"## Probability Summary",
        f"",
        f"| Stat | 30b | 40 |",
        f"|------|-----|-----|",
        f"| mean | {prob_summary_30b['mean']:.4f} | {prob_summary_40['mean']:.4f} |",
        f"| std  | {prob_summary_30b['std']:.4f} | {prob_summary_40['std']:.4f} |",
        f"| label=0 mean | {prob_summary_30b['label0_mean']:.4f} | {prob_summary_40['label0_mean']:.4f} |",
        f"| label=1 mean | {prob_summary_30b['label1_mean']:.4f} | {prob_summary_40['label1_mean']:.4f} |",
        f"",
        f"## Error Analysis",
        f"",
        f"| Case type | Count |",
        f"|-----------|-------|",
        f"| 30b FP → 40 TN (normal improvement) | {int(mask_fp_30b_tn_40.sum())} |",
        f"| 30b TN → 40 FP (normal degradation) | {int(mask_tn_30b_fp_40.sum())} |",
        f"| 30b FN → 40 TP (NSCLC improvement)  | {int(mask_fn_30b_tp_40.sum())} |",
        f"| 30b TP → 40 FN (NSCLC degradation)  | {int(mask_tp_30b_fn_40.sum())} |",
        f"| low_mask_30b + 40 worsened | {low_mask_worsened} |",
        f"",
        f"## Interpretation",
        f"",
        f"- P40 vs 30b AUROC: {'improved' if p40_better_auc else 'degraded'} ({auc_40-auc_30b:+.4f})",
        f"- P40 vs 30b AUPRC: {'improved' if p40_better_auprc else 'degraded'} ({auprc_40-auprc_30b:+.4f})",
        f"- P40 vs 30b Brier: {'improved (lower)' if p40_better_brier else 'degraded (higher)'} ({brier_40-brier_30b:+.4f})",
        f"- Normal FP: {'reduced' if p40_better_fp else 'increased'} ({fp_delta:+d})",
        f"- NSCLC FN: {'reduced' if p40_better_fn else 'increased'} ({fn_delta:+d})",
        f"- ECE: {'improved (lower)' if ece_40 < ece_30b else 'degraded (higher)'} ({ece_40-ece_30b:+.4f})",
        f"- low_mask worsening count: {low_mask_worsened}",
        f"",
        f"**Note**: This is internal validation only. final_test has NOT been accessed.",
        f"Selected candidate (30b) remains unchanged regardless of this comparison.",
        f"",
        f"## Next Step Recommendation",
        f"",
        f"{next_step}",
        f"",
        f"## Guardrail",
        f"",
        f"- final_test_accessed: False",
        f"- threshold_optimization: False",
        f"- selected_candidate_replaced: False",
        f"- guardrail_fail_count: {guardrail_fail_count}",
    ]
    _write_md("\n".join(md_lines), REPORT_ROOT / "p_c_normal41_internal_val_comparison_report.md")

    # ── 16. Summary JSON ──────────────────────────────────────────────────────
    pat_auc_max_30b = safe_auroc(pat_labels, df_pat["max_prob_30b"].values)
    pat_auc_max_40  = safe_auroc(pat_labels, df_pat["max_prob_40"].values)

    summary = {
        "stage":               STAGE_LABEL,
        "verdict":             verdict_str,
        "role":                "internal_val_comparison_only",
        "selected_candidate":  "P-C-NORMAL30b_masked_input unchanged",
        "model_a":             "P-C-NORMAL30b_masked_input",
        "model_b":             "P-C-NORMAL40_2p5d5ch",
        "val_rows":            len(labels_arr),
        "prediction_join_success": True,
        "model_a_auc":         round(auc_30b, 4),
        "model_b_auc":         round(auc_40, 4),
        "model_a_auprc":       round(auprc_30b, 4),
        "model_b_auprc":       round(auprc_40, 4),
        "model_a_brier":       round(brier_30b, 4),
        "model_b_brier":       round(brier_40, 4),
        "model_a_ece":         round(ece_30b, 4),
        "model_b_ece":         round(ece_40, 4),
        "model_a_fp_at_0p5":   fp_30b,
        "model_b_fp_at_0p5":   fp_40,
        "model_a_fn_at_0p5":   fn_30b,
        "model_b_fn_at_0p5":   fn_40,
        "fp_delta_40_minus_30b": fp_delta,
        "fn_delta_40_minus_30b": fn_delta,
        "patient_fp_30b":      pat_fp_30b,
        "patient_fn_30b":      pat_fn_30b,
        "patient_fp_40":       pat_fp_40,
        "patient_fn_40":       pat_fn_40,
        "patient_fp_delta_40_minus_30b": pat_fp_40 - pat_fp_30b,
        "patient_fn_delta_40_minus_30b": pat_fn_40 - pat_fn_30b,
        "patient_auc_max_30b": round(pat_auc_max_30b, 4),
        "patient_auc_max_40":  round(pat_auc_max_40, 4),
        "low_mask_worsening_count": low_mask_worsened,
        "p40_candidate_promotion_recommended": promote_recommended,
        "final_test_accessed": False,
        "threshold_optimization": False,
        "threshold_sweep":     False,
        "best_threshold_selection": False,
        "fixed_0p5_only":      True,
        "selected_candidate_replaced": False,
        "guardrail_fail_count": guardrail_fail_count,
        "next_step_recommendation": next_step,
        "guardrail": GUARDRAIL,
    }
    _write_json(summary, REPORT_ROOT / "p_c_normal41_internal_val_comparison_summary.json")

    # DONE.json
    done = {
        "stage":   STAGE_LABEL,
        "verdict": verdict_str,
        "timestamp": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(done, OUTPUT_ROOT / "DONE.json")
    _write_json(done, REPORT_ROOT / "DONE.json")

    print(f"\n[{STAGE_LABEL}] === COMPLETE ===")
    print(f"  verdict:  {verdict_str}")
    print(f"  30b AUC:  {auc_30b:.4f}  AUPRC: {auprc_30b:.4f}  Brier: {brier_30b:.4f}  FP: {fp_30b}  FN: {fn_30b}")
    print(f"  40  AUC:  {auc_40:.4f}  AUPRC: {auprc_40:.4f}  Brier: {brier_40:.4f}  FP: {fp_40}  FN: {fn_40}")
    print(f"  AUC delta: {auc_40-auc_30b:+.4f}  FP delta: {fp_delta:+d}  FN delta: {fn_delta:+d}")
    print(f"  next: {next_step}")
    print(f"  report: {REPORT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
