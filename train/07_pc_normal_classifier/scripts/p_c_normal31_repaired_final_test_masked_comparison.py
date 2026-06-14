"""
p_c_normal31_repaired_final_test_masked_comparison.py

P-C-NORMAL31: repaired final_test masked-input comparison
  Phase A: final_test crop-level 3ch mask 생성 및 검증
  Phase B: reference(24j balanced_w1) vs candidate(30b masked) prediction export
  Phase C: crop-level / patient-level metrics 계산
  Phase D: comparison report

비교:
  reference: P-C-NORMAL24j-fix-balanced-w1 (raw CT + scalar, mask 없음)
  candidate: P-C-NORMAL30b masked-input (CT × mask + scalar)
  공통 조건: repaired final_test manifest, fixed threshold 0.5

금지:
  재학습, threshold 최적화/sweep, checkpoint 수정, 기존 결과 덮어쓰기

실행:
  python p_c_normal31_repaired_final_test_masked_comparison.py --run
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_TEST_MANIFEST = PROJECT_ROOT / "outputs/manifests/p_c_normal27_scalar_repair_final_test_manifest/p_c_normal27_final_test_feature_manifest_repaired_usable.csv"
SCALAR_STATS_PATH   = PROJECT_ROOT / "outputs/reports/p_c_normal24h_fix_zroi_scalar_fusion_drycheck/p_c_normal24h_fix_scalar_normalization_stats.json"
ROI_ROOT            = PROJECT_ROOT / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"
REF_CKPT_PATH       = PROJECT_ROOT / "outputs/p_c_normal24j_fix_balanced_w1_zroi_scalar_fusion_full_train/checkpoints/p_c_normal24j_fix_balanced_w1_best_val_auc_checkpoint.pt"
CAND_CKPT_PATH      = PROJECT_ROOT / "outputs/p_c_normal30b_masked_input_full_train/checkpoints/p_c_normal30b_best_val_auc_checkpoint.pt"
PREV_REPORT_28      = PROJECT_ROOT / "outputs/reports/p_c_normal28_repaired_prediction_export/p_c_normal28_prediction_export_summary.json"

OUTPUT_ROOT  = PROJECT_ROOT / "outputs/p_c_normal31_repaired_final_test_masked_comparison"
MASK_DIR     = OUTPUT_ROOT / "masks/final_test"
REPORT_ROOT  = PROJECT_ROOT / "outputs/reports/p_c_normal31_repaired_final_test_masked_comparison"

# ── Constants ─────────────────────────────────────────────────────────────────
EXPECTED_TOTAL  = 66283
EXPECTED_NORMAL = 21560
EXPECTED_NSCLC  = 44723
CROP_HALF       = 48       # center ± 48 → 96 px
HU_MIN, HU_MAX  = -1000.0, 200.0
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
SCALAR_FEATURES = ["lung_z_percentile", "crop_lung_roi_ratio"]
FIXED_THRESH    = 0.5
STAGE_LABEL     = "P-C-NORMAL31"
SEED            = 42

# P-C-NORMAL28 balanced_w1 기준값 (재현성 체크용)
REF28_AUROC       = 0.951743
REF28_AUPRC       = 0.973271
REF28_SPECIFICITY = 0.49564
REF28_SENSITIVITY = 0.992912
REF28_FP          = 10874
REF28_FN          = 317
REPRO_AUROC_TOL   = 0.005
REPRO_SPEC_TOL    = 0.02
REPRO_FP_TOL_REL  = 0.05

# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _apply_scalar_norm(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    for col, s in stats.items():
        df[col] = (df[col].astype(float) - s["mean"]) / s["std"]
    return df


# ── ROI mask extraction ───────────────────────────────────────────────────────

def get_roi_path(safe_id: str, label: int) -> Path:
    subdir = "normal" if label == 0 else "lesion"
    return ROI_ROOT / subdir / safe_id / "refined_roi.npy"


def extract_mask_96(roi_volume: np.ndarray, z: int, cy: int, cx: int):
    Z, H, W = roi_volume.shape
    nearest_repeat_used = False
    z_eff = int(np.clip(z, 0, Z - 1))
    if z_eff != int(z):
        nearest_repeat_used = True

    y0_vol = cy - CROP_HALF
    y1_vol = cy + CROP_HALF
    x0_vol = cx - CROP_HALF
    x1_vol = cx + CROP_HALF

    pad_y0 = max(0, -y0_vol)
    pad_y1 = max(0, y1_vol - H)
    pad_x0 = max(0, -x0_vol)
    pad_x1 = max(0, x1_vol - W)

    y0c = max(0, y0_vol)
    y1c = min(H, y1_vol)
    x0c = max(0, x0_vol)
    x1c = min(W, x1_vol)

    patch = roi_volume[z_eff][y0c:y1c, x0c:x1c]
    if pad_y0 > 0 or pad_y1 > 0 or pad_x0 > 0 or pad_x1 > 0:
        patch = np.pad(patch, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant")

    return patch.astype(np.uint8), (pad_y0, pad_y1, pad_x0, pad_x1), z_eff, nearest_repeat_used


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


def load_model(ckpt_path: Path, device: torch.device) -> tuple:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = ScalarFusionModel().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", -1)
    val_auc = ckpt.get("val_auc", float("nan"))
    return model, epoch, val_auc


# ── Dataset ───────────────────────────────────────────────────────────────────

class RawCTDataset(Dataset):
    """reference 모델용: raw CT crop + scalar, mask 없음."""
    def __init__(self, df: pd.DataFrame):
        self.df   = df.reset_index(drop=True)
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)

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
        scalar = torch.tensor(
            [float(row["lung_z_percentile"]), float(row["crop_lung_roi_ratio"])],
            dtype=torch.float32,
        )
        return img, scalar, int(row["label"]), idx


class MaskedCTDataset(Dataset):
    """candidate 모델용: CT × mask_3ch + scalar."""
    def __init__(self, df: pd.DataFrame):
        self.df   = df.reset_index(drop=True)
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row  = self.df.iloc[idx]
        data = np.load(str(row["crop_path"]))
        arr  = data["ct_crop"].astype(np.float32)
        arr  = np.clip(arr, HU_MIN, HU_MAX)
        arr  = (arr - HU_MIN) / (HU_MAX - HU_MIN)
        img  = torch.from_numpy(arr)

        mask_data = np.load(str(row["mask_path"]))
        mask_t    = torch.from_numpy(mask_data["mask_3ch"].astype(np.float32))
        img = img * mask_t

        img = (img - self.mean) / self.std
        scalar = torch.tensor(
            [float(row["lung_z_percentile"]), float(row["crop_lung_roi_ratio"])],
            dtype=torch.float32,
        )
        return img, scalar, int(row["label"]), idx


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_auroc(labels, scores):
    try:
        la = np.asarray(labels, dtype=np.int32)
        sc = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(sc)):
            return float("nan"), "nan_inf"
        if len(np.unique(la)) < 2:
            return float("nan"), "single_class"
        n_pos, n_neg = int((la == 1).sum()), int((la == 0).sum())
        all_s  = np.concatenate([sc[la == 0], sc[la == 1]])
        is_pos = np.concatenate([np.zeros(n_neg, bool), np.ones(n_pos, bool)])
        order  = np.argsort(all_s, kind="stable")
        ss, sp = all_s[order], is_pos[order]
        n = n_pos + n_neg
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i + 1
            while j < n and ss[j] == ss[i]:
                j += 1
            ranks[i:j] = (i + 1 + j) / 2.0
            i = j
        U = float(ranks[sp].sum()) - n_pos * (n_pos + 1) / 2.0
        return float(U / (n_pos * n_neg)), "OK"
    except Exception as e:
        return float("nan"), str(e)


def compute_auprc(labels, scores):
    try:
        la = np.asarray(labels, dtype=np.int32)
        sc = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(sc)):
            return float("nan"), "nan_inf"
        if len(np.unique(la)) < 2:
            return float("nan"), "single_class"
        n_pos = int((la == 1).sum())
        if n_pos == 0:
            return float("nan"), "no_positive"
        order = np.argsort(sc, kind="stable")[::-1]
        sl    = la[order]
        tp    = np.cumsum(sl)
        fp    = np.cumsum(1 - sl)
        prec  = tp / (tp + fp)
        rec   = tp / n_pos
        prec  = np.concatenate([[1.0], prec])
        rec   = np.concatenate([[0.0], rec])
        return float(np.trapezoid(prec, rec)), "OK"
    except Exception as e:
        return float("nan"), str(e)


def compute_crop_metrics(labels, probs, threshold=FIXED_THRESH):
    la = np.asarray(labels, dtype=np.int32)
    pr = np.asarray(probs, dtype=np.float64)
    pred = (pr >= threshold).astype(np.int32)

    TP = int(((pred == 1) & (la == 1)).sum())
    TN = int(((pred == 0) & (la == 0)).sum())
    FP = int(((pred == 1) & (la == 0)).sum())
    FN = int(((pred == 0) & (la == 1)).sum())

    n_total  = len(la)
    n_normal = int((la == 0).sum())
    n_nsclc  = int((la == 1).sum())

    acc      = (TP + TN) / n_total if n_total > 0 else float("nan")
    sens     = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    spec     = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
    prec_v   = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    bal_acc  = (sens + spec) / 2 if not (math.isnan(sens) or math.isnan(spec)) else float("nan")
    f1       = 2 * prec_v * sens / (prec_v + sens) if (prec_v + sens) > 0 else float("nan")
    brier    = float(np.mean((pr - la.astype(np.float64)) ** 2))

    auroc, _ = compute_auroc(la, pr)
    auprc, _ = compute_auprc(la, pr)

    # class-weighted
    w_normal = n_total / (2 * n_normal) if n_normal > 0 else 1.0
    w_nsclc  = n_total / (2 * n_nsclc)  if n_nsclc  > 0 else 1.0
    weights  = np.where(la == 0, w_normal, w_nsclc)
    w_acc    = float(np.average((pred == la).astype(float), weights=weights))
    w_brier  = float(np.average((pr - la.astype(float)) ** 2, weights=weights))
    w_sens   = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    w_spec   = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
    w_prec   = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    w_f1     = 2 * w_prec * w_sens / (w_prec + w_sens) if (w_prec + w_sens) > 0 else float("nan")

    return {
        "auroc": round(auroc, 6), "auprc": round(auprc, 6), "brier": round(brier, 6),
        "accuracy": round(acc, 6), "balanced_accuracy": round(bal_acc, 6),
        "sensitivity": round(sens, 6), "specificity": round(spec, 6),
        "precision": round(prec_v, 6), "f1": round(f1, 6),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "normal_FP_count": FP, "nsclc_FN_count": FN,
        # weighted supplementary
        "w_accuracy": round(w_acc, 6), "w_brier": round(w_brier, 6),
        "w_sensitivity": round(w_sens, 6), "w_specificity": round(w_spec, 6),
        "w_precision": round(w_prec, 6), "w_f1": round(w_f1, 6),
        "w_normal": round(w_normal, 4), "w_nsclc": round(w_nsclc, 4),
    }


def compute_patient_metrics(df_pred: pd.DataFrame, agg: str):
    """agg: mean_prob / p95_prob / max_prob"""
    grp = df_pred.groupby("patient_id")
    pat_rows = []
    for pid, g in grp:
        label_pat = int(g["label"].max())
        if agg == "mean_prob":
            score = float(g["prob"].mean())
        elif agg == "p95_prob":
            score = float(np.percentile(g["prob"], 95))
        else:
            score = float(g["prob"].max())
        pat_rows.append({"patient_id": pid, "label": label_pat, "score": score})
    df_pat = pd.DataFrame(pat_rows)
    labels = df_pat["label"].values
    scores = df_pat["score"].values
    preds  = (scores >= FIXED_THRESH).astype(int)

    auroc, _ = compute_auroc(labels, scores)
    auprc, _ = compute_auprc(labels, scores)
    TP = int(((preds == 1) & (labels == 1)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())
    n_normal_fp_pat = FP
    n_nsclc_fn_pat  = FN
    return {
        "agg": agg, "n_patients": len(df_pat),
        "auroc": round(auroc, 6), "auprc": round(auprc, 6),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "normal_FP_patients": n_normal_fp_pat,
        "nsclc_FN_patients":  n_nsclc_fn_pat,
    }


# ── Phase A: mask generation ──────────────────────────────────────────────────

def phase_a(df: pd.DataFrame) -> tuple:
    """final_test crop-level 3ch mask 생성. (df_with_mask_path, audit_rows, verdict)"""
    print(f"[{STAGE_LABEL}] Phase A: mask generation ({len(df)} crops)...")
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    # ROI 사전 점검
    checked = set()
    missing_roi = []
    for _, row in df.iterrows():
        key = (str(row["safe_id"]), int(row["label"]))
        if key not in checked:
            checked.add(key)
            rp = get_roi_path(str(row["safe_id"]), int(row["label"]))
            if not rp.exists():
                missing_roi.append(str(rp))

    if missing_roi:
        print(f"[{STAGE_LABEL}] Phase A ABORT: {len(missing_roi)} ROI missing", file=sys.stderr)
        return df, [], "FAIL"

    # 환자별 정렬 처리
    df_sorted = df.sort_values(["safe_id", "canonical_volume_z"]).reset_index(drop=True)
    manifest_rows, audit_rows, zero_low_rows = [], [], []
    n_done = n_error = n_nearest = n_spatial_pad = 0
    current_safe_id = None
    roi_volume = roi_vol_z = None
    mask_paths = {}

    for idx, row in df_sorted.iterrows():
        safe_id   = str(row["safe_id"])
        label     = int(row["label"])
        z_raw     = row["canonical_volume_z"]
        cy        = int(row["center_y"])
        cx        = int(row["center_x"])
        crop_path = str(row["crop_path"])

        if idx % 5000 == 0:
            print(f"[{STAGE_LABEL}] Phase A {idx}/{len(df_sorted)} done={n_done} err={n_error}")

        if safe_id != current_safe_id:
            roi_path = get_roi_path(safe_id, label)
            if not roi_path.exists():
                n_error += 1
                current_safe_id = safe_id
                roi_volume = None
                continue
            roi_volume      = np.load(str(roi_path))
            roi_vol_z       = roi_volume.shape[0]
            current_safe_id = safe_id

        if roi_volume is None:
            n_error += 1
            audit_rows.append({"crop_path": crop_path, "safe_id": safe_id, "label": label,
                                "status": "FAIL_ROI_NONE", "error": "roi_volume is None"})
            continue

        try:
            z_int = int(float(z_raw))
        except Exception as e:
            n_error += 1
            audit_rows.append({"crop_path": crop_path, "safe_id": safe_id, "label": label,
                                "status": "FAIL_Z_PARSE", "error": str(e)})
            continue

        try:
            ch0, pads0, zm1_eff, nr0 = extract_mask_96(roi_volume, z_int - 1, cy, cx)
            ch1, pads1, z0_eff,  nr1 = extract_mask_96(roi_volume, z_int,     cy, cx)
            ch2, pads2, zp1_eff, nr2 = extract_mask_96(roi_volume, z_int + 1, cy, cx)
        except Exception as e:
            n_error += 1
            audit_rows.append({"crop_path": crop_path, "safe_id": safe_id, "label": label,
                                "status": "FAIL_EXTRACT", "error": str(e)})
            continue

        mask_3ch = np.stack([ch0, ch1, ch2], axis=0)
        if mask_3ch.shape != (3, 96, 96):
            n_error += 1
            audit_rows.append({"crop_path": crop_path, "safe_id": safe_id, "label": label,
                                "status": "FAIL_SHAPE", "error": str(mask_3ch.shape)})
            continue

        nr_any  = any([nr0, nr1, nr2])
        sp_any  = any(any(p > 0 for p in pads) for pads in [pads0, pads1, pads2])
        if nr_any:
            n_nearest += 1
        if sp_any:
            n_spatial_pad += 1

        nzr_ch0 = float(ch0.astype(bool).mean())
        nzr_ch1 = float(ch1.astype(bool).mean())
        nzr_ch2 = float(ch2.astype(bool).mean())
        nzr_mean = (nzr_ch0 + nzr_ch1 + nzr_ch2) / 3

        crop_stem = Path(crop_path).stem
        mask_fname = f"{crop_stem}_mask.npz"
        mask_out   = MASK_DIR / mask_fname
        np.savez_compressed(str(mask_out), mask_3ch=mask_3ch)

        mask_paths[crop_path] = str(mask_out)
        n_done += 1

        is_zero = int(mask_3ch.sum() == 0)
        is_low  = int(nzr_mean < 0.05 and not is_zero)

        manifest_rows.append({
            "crop_path": crop_path,
            "mask_path": str(mask_out),
            "safe_id":   safe_id,
            "label":     label,
            "canonical_volume_z": z_raw,
            "center_y":  cy,
            "center_x":  cx,
            "mask_shape":    "3x96x96",
            "mask_dtype":    "uint8",
            "mask_nonzero_ratio_ch0":  round(nzr_ch0, 6),
            "mask_nonzero_ratio_ch1":  round(nzr_ch1, 6),
            "mask_nonzero_ratio_ch2":  round(nzr_ch2, 6),
            "mask_nonzero_ratio_mean": round(nzr_mean, 6),
            "nearest_repeat_used": nr_any,
            "spatial_pad_used":    sp_any,
            "zero_mask":           bool(is_zero),
            "low_mask":            bool(is_low),
            "status":              "PASS",
        })

        if is_zero or is_low:
            zero_low_rows.append({"crop_path": crop_path, "safe_id": safe_id,
                                   "label": label, "nzr_mean": round(nzr_mean, 6),
                                   "zero_mask": bool(is_zero), "low_mask": bool(is_low)})

        audit_rows.append({
            "crop_path": crop_path, "safe_id": safe_id, "label": label,
            "status": "PASS", "nzr_mean": round(nzr_mean, 6),
            "nearest_repeat": nr_any, "spatial_pad": sp_any, "error": "",
        })

    print(f"[{STAGE_LABEL}] Phase A done={n_done} error={n_error} nearest={n_nearest} spatial_pad={n_spatial_pad}")

    # manifest 저장
    _write_csv(manifest_rows, REPORT_ROOT / "p_c_normal31_final_test_mask_manifest.csv")
    _write_csv(audit_rows,    REPORT_ROOT / "p_c_normal31_final_test_mask_generation_audit.csv")
    _write_csv(zero_low_rows if zero_low_rows else [{"note": "no zero or low mask"}],
               REPORT_ROOT / "p_c_normal31_final_test_mask_low_or_zero_cases.csv")

    # distribution summary
    if manifest_rows:
        nzr_vals = [r["mask_nonzero_ratio_mean"] for r in manifest_rows]
        norm_nzr = [r["mask_nonzero_ratio_mean"] for r in manifest_rows if r["label"] == 0]
        nsclc_nzr = [r["mask_nonzero_ratio_mean"] for r in manifest_rows if r["label"] == 1]
        dist_rows = [
            {"group": "all",   "n": len(nzr_vals),  "mean": round(np.mean(nzr_vals), 4),
             "p10": round(np.percentile(nzr_vals, 10), 4), "p50": round(np.percentile(nzr_vals, 50), 4),
             "p90": round(np.percentile(nzr_vals, 90), 4)},
            {"group": "normal", "n": len(norm_nzr), "mean": round(np.mean(norm_nzr), 4) if norm_nzr else 0,
             "p10": round(np.percentile(norm_nzr, 10), 4) if norm_nzr else 0,
             "p50": round(np.percentile(norm_nzr, 50), 4) if norm_nzr else 0,
             "p90": round(np.percentile(norm_nzr, 90), 4) if norm_nzr else 0},
            {"group": "nsclc", "n": len(nsclc_nzr), "mean": round(np.mean(nsclc_nzr), 4) if nsclc_nzr else 0,
             "p10": round(np.percentile(nsclc_nzr, 10), 4) if nsclc_nzr else 0,
             "p50": round(np.percentile(nsclc_nzr, 50), 4) if nsclc_nzr else 0,
             "p90": round(np.percentile(nsclc_nzr, 90), 4) if nsclc_nzr else 0},
        ]
        _write_csv(dist_rows, REPORT_ROOT / "p_c_normal31_final_test_mask_distribution_summary.csv")

    # df에 mask_path 추가
    df_out = df.copy()
    df_out["mask_path"] = df_out["crop_path"].astype(str).map(mask_paths)
    n_missing = int(df_out["mask_path"].isna().sum())
    n_zero    = len(zero_low_rows)

    phase_a_verdict = "PASS"
    if n_error > 0 or n_missing > 0:
        phase_a_verdict = "FAIL"
    elif n_zero > 0:
        phase_a_verdict = "PARTIAL_PASS"

    print(f"[{STAGE_LABEL}] Phase A verdict={phase_a_verdict} missing={n_missing} zero={n_zero}")
    return df_out, audit_rows, phase_a_verdict


# ── Phase B: prediction export ────────────────────────────────────────────────

def run_inference(model, dataset, device, batch_size=64, n_workers=4) -> tuple:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=n_workers, pin_memory=True, drop_last=False)
    all_logits, all_probs, all_labels, all_idx = [], [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, scalars, labels, idxs in loader:
            imgs    = imgs.to(device)
            scalars = scalars.to(device)
            logits  = model(imgs, scalars)
            probs   = torch.sigmoid(logits.squeeze(1))
            all_logits.extend(logits.squeeze(1).cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_idx.extend(idxs.numpy().tolist())
    return all_logits, all_probs, all_labels, all_idx


def phase_b(df_norm: pd.DataFrame, df_masked: pd.DataFrame, device: torch.device) -> tuple:
    """두 모델 prediction export."""
    print(f"[{STAGE_LABEL}] Phase B: loading models...")

    ref_model,  ref_epoch,  ref_auc  = load_model(REF_CKPT_PATH,  device)
    cand_model, cand_epoch, cand_auc = load_model(CAND_CKPT_PATH, device)

    ckpt_meta = [
        {"model": "reference_balanced_w1", "ckpt_path": str(REF_CKPT_PATH),
         "epoch": ref_epoch, "val_auc": ref_auc,
         "masked_input": False, "smoke_only": False},
        {"model": "masked_30b", "ckpt_path": str(CAND_CKPT_PATH),
         "epoch": cand_epoch, "val_auc": cand_auc,
         "masked_input": True, "smoke_only": False},
    ]
    _write_csv(ckpt_meta, REPORT_ROOT / "p_c_normal31_checkpoint_metadata_check.csv")

    # reference inference (raw CT)
    print(f"[{STAGE_LABEL}] Phase B: reference inference (raw CT, epoch={ref_epoch})...")
    ds_ref = RawCTDataset(df_norm)
    logits_r, probs_r, labels_r, idx_r = run_inference(ref_model, ds_ref, device)

    # candidate inference (masked CT)
    print(f"[{STAGE_LABEL}] Phase B: candidate inference (masked CT, epoch={cand_epoch})...")
    ds_cand = MaskedCTDataset(df_masked)
    logits_c, probs_c, labels_c, idx_c = run_inference(cand_model, ds_cand, device)

    # prediction CSV 작성
    meta_cols = ["patient_id", "safe_id", "crop_path", "label", "source_split",
                 "canonical_volume_z", "local_z", "lung_z_percentile", "crop_lung_roi_ratio",
                 "position_bin", "center_y", "center_x"]

    def make_pred_rows(df_src, logits, probs, labels_out, idxs, model_name, masked, mask_col=None):
        rows = []
        for i, (li, pr, lb, ix) in enumerate(zip(logits, probs, labels_out, idxs)):
            row = df_src.iloc[ix]
            d = {c: row.get(c, "") for c in meta_cols}
            d["mask_path"]        = str(row.get("mask_path", "")) if mask_col else ""
            d["logit"]            = round(li, 6)
            d["prob"]             = round(pr, 6)
            d["pred_at_0p5"]      = int(pr >= FIXED_THRESH)
            d["model_name"]       = model_name
            d["checkpoint_path"]  = str(REF_CKPT_PATH if not masked else CAND_CKPT_PATH)
            d["masked_input_used"] = masked
            rows.append(d)
        return rows

    ref_rows  = make_pred_rows(df_norm,   logits_r, probs_r, labels_r, idx_r,
                               "reference_balanced_w1", False)
    cand_rows = make_pred_rows(df_masked, logits_c, probs_c, labels_c, idx_c,
                               "masked_30b", True, mask_col="mask_path")

    _write_csv(ref_rows,  REPORT_ROOT / "p_c_normal31_reference_balanced_w1_predictions.csv")
    _write_csv(cand_rows, REPORT_ROOT / "p_c_normal31_masked_30b_predictions.csv")

    print(f"[{STAGE_LABEL}] Phase B done: ref={len(ref_rows)} cand={len(cand_rows)}")
    return (probs_r, labels_r, df_norm["patient_id"].tolist(),
            probs_c, labels_c, df_masked["patient_id"].tolist())


# ── Phase C: metrics ──────────────────────────────────────────────────────────

def phase_c(probs_r, labels_r, pids_r, probs_c, labels_c, pids_c):
    la_r = np.array(labels_r)
    la_c = np.array(labels_c)
    pr_r = np.array(probs_r)
    pr_c = np.array(probs_c)

    m_r = compute_crop_metrics(la_r, pr_r)
    m_c = compute_crop_metrics(la_c, pr_c)

    raw_cols   = ["auroc", "auprc", "brier", "accuracy", "balanced_accuracy",
                  "sensitivity", "specificity", "precision", "f1",
                  "TP", "TN", "FP", "FN", "normal_FP_count", "nsclc_FN_count"]
    w_cols     = ["w_accuracy", "w_brier", "w_sensitivity", "w_specificity",
                  "w_precision", "w_f1", "w_normal", "w_nsclc"]

    raw_rows = []
    for model_name, m in [("reference_balanced_w1", m_r), ("masked_30b", m_c)]:
        d = {"model": model_name, "threshold": FIXED_THRESH}
        for k in raw_cols:
            d[k] = m[k]
        raw_rows.append(d)
    _write_csv(raw_rows, REPORT_ROOT / "p_c_normal31_crop_raw_metrics_comparison.csv")

    cm_rows = []
    for model_name, m in [("reference_balanced_w1", m_r), ("masked_30b", m_c)]:
        cm_rows.append({"model": model_name, "TP": m["TP"], "TN": m["TN"],
                         "FP": m["FP"], "FN": m["FN"],
                         "normal_FP": m["normal_FP_count"], "nsclc_FN": m["nsclc_FN_count"]})
    _write_csv(cm_rows, REPORT_ROOT / "p_c_normal31_crop_confusion_matrix_comparison.csv")

    w_rows = []
    for model_name, m in [("reference_balanced_w1", m_r), ("masked_30b", m_c)]:
        d = {"model": model_name, "supplementary_only": True}
        for k in w_cols:
            d[k] = m[k]
        w_rows.append(d)
    _write_csv(w_rows, REPORT_ROOT / "p_c_normal31_crop_weighted_metrics_comparison.csv")

    rv_rows = []
    for model_name, m in [("reference_balanced_w1", m_r), ("masked_30b", m_c)]:
        d = {"model": model_name}
        for k in raw_cols:
            d[f"raw_{k}"] = m[k]
        for k in w_cols:
            d[f"w_{k}"] = m[k]
        rv_rows.append(d)
    _write_csv(rv_rows, REPORT_ROOT / "p_c_normal31_crop_raw_vs_weighted_metrics_comparison.csv")

    # patient-level
    df_r = pd.DataFrame({"patient_id": pids_r, "prob": pr_r, "label": la_r})
    df_c = pd.DataFrame({"patient_id": pids_c, "prob": pr_c, "label": la_c})

    pat_rows = []
    for agg in ["mean_prob", "p95_prob", "max_prob"]:
        mr = compute_patient_metrics(df_r, agg)
        mc = compute_patient_metrics(df_c, agg)
        pat_rows.append({"model": "reference_balanced_w1", **mr})
        pat_rows.append({"model": "masked_30b", **mc})
    _write_csv(pat_rows, REPORT_ROOT / "p_c_normal31_patient_metrics_comparison.csv")

    pat_cm_rows = []
    for agg in ["mean_prob", "p95_prob", "max_prob"]:
        mr = compute_patient_metrics(df_r, agg)
        mc = compute_patient_metrics(df_c, agg)
        pat_cm_rows.append({"model": "reference_balanced_w1", "agg": agg,
                             "TP": mr["TP"], "TN": mr["TN"], "FP": mr["FP"], "FN": mr["FN"],
                             "normal_FP_patients": mr["normal_FP_patients"],
                             "nsclc_FN_patients":  mr["nsclc_FN_patients"]})
        pat_cm_rows.append({"model": "masked_30b", "agg": agg,
                             "TP": mc["TP"], "TN": mc["TN"], "FP": mc["FP"], "FN": mc["FN"],
                             "normal_FP_patients": mc["normal_FP_patients"],
                             "nsclc_FN_patients":  mc["nsclc_FN_patients"]})
    _write_csv(pat_cm_rows, REPORT_ROOT / "p_c_normal31_patient_confusion_matrix_comparison.csv")

    return m_r, m_c


# ── Phase D: report ───────────────────────────────────────────────────────────

def phase_d(m_r, m_c, n_total, n_normal, n_nsclc, phase_a_verdict, ts):
    # 재현성 체크 (reference vs P-C-NORMAL28 balanced_w1)
    auroc_diff = abs(m_r["auroc"] - REF28_AUROC)
    spec_diff  = abs(m_r["specificity"] - REF28_SPECIFICITY)
    fp_rel     = abs(m_r["FP"] - REF28_FP) / max(REF28_FP, 1)

    repro_ok = (auroc_diff <= REPRO_AUROC_TOL and
                spec_diff  <= REPRO_SPEC_TOL  and
                fp_rel     <= REPRO_FP_TOL_REL)
    repro_status = "PASS" if repro_ok else "WARN"

    repro_rows = [
        {"check": "auroc",       "ref31": m_r["auroc"],       "ref28": REF28_AUROC,
         "diff": round(auroc_diff, 6), "tol": REPRO_AUROC_TOL, "status": "OK" if auroc_diff <= REPRO_AUROC_TOL else "WARN"},
        {"check": "specificity", "ref31": m_r["specificity"], "ref28": REF28_SPECIFICITY,
         "diff": round(spec_diff, 6), "tol": REPRO_SPEC_TOL, "status": "OK" if spec_diff <= REPRO_SPEC_TOL else "WARN"},
        {"check": "FP_count",    "ref31": m_r["FP"],          "ref28": REF28_FP,
         "diff": abs(m_r["FP"] - REF28_FP), "tol_rel": REPRO_FP_TOL_REL,
         "status": "OK" if fp_rel <= REPRO_FP_TOL_REL else "WARN"},
    ]
    _write_csv(repro_rows, REPORT_ROOT / "p_c_normal31_reference_reproducibility_check.csv")

    # masking 효과 해석
    auroc_delta = m_c["auroc"] - m_r["auroc"]
    spec_delta  = m_c["specificity"] - m_r["specificity"]
    sens_delta  = m_c["sensitivity"] - m_r["sensitivity"]
    fp_delta    = m_c["FP"] - m_r["FP"]
    fn_delta    = m_c["FN"] - m_r["FN"]

    if auroc_delta > 0.002:
        masking_effect = "masking이 fixed 0.5 operating point에서 개선 가능성을 보였다"
    elif auroc_delta < -0.002:
        masking_effect = "masking으로 정보 손실 또는 distribution shift가 발생했을 수 있음"
    else:
        masking_effect = "masking 적용 전후 성능 차이가 미미함 (AUROC delta < 0.002)"

    # report.md
    lines = [
        f"# P-C-NORMAL31 Repaired Final_Test Masked Comparison Report",
        f"",
        f"- **Stage**: {STAGE_LABEL}",
        f"- **Timestamp**: {ts}",
        f"- **Phase A (mask generation)**: {phase_a_verdict}",
        f"- **Reference reproducibility**: {repro_status}",
        f"",
        f"## Dataset",
        f"- total: {n_total} (expected {EXPECTED_TOTAL})",
        f"- normal: {n_normal} (expected {EXPECTED_NORMAL})",
        f"- NSCLC: {n_nsclc} (expected {EXPECTED_NSCLC})",
        f"- fixed threshold: {FIXED_THRESH}",
        f"",
        f"## Crop-level Raw Metrics (PRIMARY)",
        f"",
        f"| Metric | Reference (24j bw1) | Masked 30b | Delta |",
        f"|--------|---------------------|------------|-------|",
        f"| AUROC | {m_r['auroc']:.6f} | {m_c['auroc']:.6f} | {auroc_delta:+.6f} |",
        f"| AUPRC | {m_r['auprc']:.6f} | {m_c['auprc']:.6f} | {m_c['auprc']-m_r['auprc']:+.6f} |",
        f"| Brier | {m_r['brier']:.6f} | {m_c['brier']:.6f} | {m_c['brier']-m_r['brier']:+.6f} |",
        f"| accuracy | {m_r['accuracy']:.6f} | {m_c['accuracy']:.6f} | {m_c['accuracy']-m_r['accuracy']:+.6f} |",
        f"| balanced_acc | {m_r['balanced_accuracy']:.6f} | {m_c['balanced_accuracy']:.6f} | {m_c['balanced_accuracy']-m_r['balanced_accuracy']:+.6f} |",
        f"| sensitivity | {m_r['sensitivity']:.6f} | {m_c['sensitivity']:.6f} | {sens_delta:+.6f} |",
        f"| specificity | {m_r['specificity']:.6f} | {m_c['specificity']:.6f} | {spec_delta:+.6f} |",
        f"| precision | {m_r['precision']:.6f} | {m_c['precision']:.6f} | {m_c['precision']-m_r['precision']:+.6f} |",
        f"| F1 | {m_r['f1']:.6f} | {m_c['f1']:.6f} | {m_c['f1']-m_r['f1']:+.6f} |",
        f"| FP (normal) | {m_r['FP']} | {m_c['FP']} | {fp_delta:+d} |",
        f"| FN (NSCLC) | {m_r['FN']} | {m_c['FN']} | {fn_delta:+d} |",
        f"",
        f"## Reference Reproducibility vs P-C-NORMAL28 balanced_w1",
        f"",
        f"| Check | ref31 | ref28 | diff | status |",
        f"|-------|-------|-------|------|--------|",
    ]
    for r in repro_rows:
        lines.append(f"| {r['check']} | {r['ref31']} | {r['ref28']} | {r['diff']} | {r['status']} |")

    lines += [
        f"",
        f"## Weighted Metrics (SUPPLEMENTARY ONLY)",
        f"",
        f"| Metric | Reference | Masked 30b |",
        f"|--------|-----------|------------|",
        f"| w_accuracy | {m_r['w_accuracy']:.6f} | {m_c['w_accuracy']:.6f} |",
        f"| w_brier | {m_r['w_brier']:.6f} | {m_c['w_brier']:.6f} |",
        f"| w_sensitivity | {m_r['w_sensitivity']:.6f} | {m_c['w_sensitivity']:.6f} |",
        f"| w_specificity | {m_r['w_specificity']:.6f} | {m_c['w_specificity']:.6f} |",
        f"| w_precision | {m_r['w_precision']:.6f} | {m_c['w_precision']:.6f} |",
        f"| w_f1 | {m_r['w_f1']:.6f} | {m_c['w_f1']:.6f} |",
        f"",
        f"## Masking Effect Interpretation",
        f"",
        f"- AUROC delta (masked - reference): {auroc_delta:+.6f}",
        f"- Specificity delta: {spec_delta:+.6f}",
        f"- Sensitivity delta: {sens_delta:+.6f}",
        f"- FP delta: {fp_delta:+d}",
        f"- FN delta: {fn_delta:+d}",
        f"",
        f"**해석**: {masking_effect}",
        f"",
        f"※ 어떤 경우에도 진단 성능/암 확률 표현 금지. 이 결과는 research comparison 목적이다.",
        f"",
        f"---",
        f"",
        f"## Next Steps",
        f"",
        f"**P-C-NORMAL32 decision checkpoint**",
        f"",
        f"- P-C-NORMAL31 결과 기준으로 masked 30b 채택 여부 결정",
        f"- 또는 masked 30b error review",
        f"- 또는 masking branch 종료",
        f"",
        f"사용자 승인 후 진행.",
    ]

    (REPORT_ROOT / "p_c_normal31_repaired_final_test_masked_comparison_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return repro_status, masking_effect, auroc_delta, spec_delta, fp_delta, fn_delta


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 출력 충돌 방지
    for out_dir in [OUTPUT_ROOT, REPORT_ROOT]:
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[ABORT] output dir exists and is not empty: {out_dir}", file=sys.stderr)
            sys.exit(2)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    errors = []

    # 입력 파일 확인
    for p in [FINAL_TEST_MANIFEST, SCALAR_STATS_PATH, REF_CKPT_PATH, CAND_CKPT_PATH]:
        if not p.exists():
            errors.append({"check": "input_file", "error": f"MISSING: {p}"})
    if errors:
        _write_csv(errors, REPORT_ROOT / "p_c_normal31_errors.csv")
        sys.exit(1)

    # manifest 로드
    df_raw = pd.read_csv(FINAL_TEST_MANIFEST, low_memory=False)
    n_total  = len(df_raw)
    n_normal = int((df_raw["label"] == 0).sum())
    n_nsclc  = int((df_raw["label"] == 1).sum())

    print(f"[{STAGE_LABEL}] manifest: total={n_total} normal={n_normal} NSCLC={n_nsclc}")

    sanity_rows = [
        {"check": "total",  "expected": EXPECTED_TOTAL,  "actual": n_total,  "status": "OK" if n_total  == EXPECTED_TOTAL  else "FAIL"},
        {"check": "normal", "expected": EXPECTED_NORMAL, "actual": n_normal, "status": "OK" if n_normal == EXPECTED_NORMAL else "FAIL"},
        {"check": "nsclc",  "expected": EXPECTED_NSCLC,  "actual": n_nsclc,  "status": "OK" if n_nsclc  == EXPECTED_NSCLC  else "FAIL"},
    ]
    _write_csv(sanity_rows, REPORT_ROOT / "p_c_normal31_manifest_sanity_check.csv")

    if n_total != EXPECTED_TOTAL or n_normal != EXPECTED_NORMAL or n_nsclc != EXPECTED_NSCLC:
        errors.append({"check": "manifest_count", "error": f"total={n_total} normal={n_normal} nsclc={n_nsclc}"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal31_errors.csv")
        sys.exit(1)

    # scalar stats
    with open(SCALAR_STATS_PATH) as f:
        norm_payload = json.load(f)
    scalar_stats = norm_payload["features"]

    # ── Phase A ───────────────────────────────────────────────────────────────
    df_masked_raw, audit_rows, phase_a_verdict = phase_a(df_raw)

    if phase_a_verdict == "FAIL":
        errors.append({"check": "phase_a", "error": "mask generation FAIL"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal31_errors.csv")
        # guardrail (partial)
        _write_csv([{"key": "final_test_mask_audit_passed", "value": False, "expected": True, "status": "FAIL"}],
                   REPORT_ROOT / "p_c_normal31_guardrail_check.csv")
        sys.exit(1)

    # ── Phase B ───────────────────────────────────────────────────────────────
    # scalar 정규화 (raw → normalized: label/mask_path 수정 없음)
    df_norm   = _apply_scalar_norm(df_raw,        scalar_stats)
    df_masked = _apply_scalar_norm(df_masked_raw, scalar_stats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{STAGE_LABEL}] device={device}")

    probs_r, labels_r, pids_r, probs_c, labels_c, pids_c = phase_b(df_norm, df_masked, device)

    # row count check
    if len(probs_r) != EXPECTED_TOTAL or len(probs_c) != EXPECTED_TOTAL:
        errors.append({"check": "pred_row_count",
                       "error": f"ref={len(probs_r)} cand={len(probs_c)} expected={EXPECTED_TOTAL}"})

    # ── Phase C ───────────────────────────────────────────────────────────────
    m_r, m_c = phase_c(probs_r, labels_r, pids_r, probs_c, labels_c, pids_c)

    # ── Phase D ───────────────────────────────────────────────────────────────
    repro_status, masking_effect, auroc_delta, spec_delta, fp_delta, fn_delta = \
        phase_d(m_r, m_c, n_total, n_normal, n_nsclc, phase_a_verdict, ts)

    # guardrail
    guardrail_items = {
        "final_test_reporting_only": True,
        "no_training_run": True,
        "no_threshold_optimization": True,
        "no_threshold_sweep": True,
        "no_best_threshold_selection": True,
        "fixed_threshold_0p5_reporting_only": True,
        "no_checkpoint_modification": True,
        "no_existing_result_overwrite": True,
        "repaired_manifest_used": True,
        "corrupted_24g_manifest_used_for_prediction": False,
        "final_test_mask_generated": True,
        "final_test_mask_audit_passed": phase_a_verdict in ("PASS", "PARTIAL_PASS"),
        "mask_applied_to_30b_image_only": True,
        "mask_not_applied_to_reference_24j": True,
        "scalar_features_unchanged": True,
        "sample_weight_not_modified": True,
        "raw_metrics_primary": True,
        "weighted_metrics_supplementary_only": True,
        "test_set_downsampling_used": False,
        "normal_patch_duplication_used": False,
        "vessel_feature_used": False,
        "roi_masked_loss_used": False,
        "diagnostic_wording_avoided": True,
    }
    false_keys = {"corrupted_24g_manifest_used_for_prediction", "no_training_run", "no_threshold_optimization",
                  "no_threshold_sweep", "no_best_threshold_selection", "no_checkpoint_modification",
                  "no_existing_result_overwrite", "test_set_downsampling_used",
                  "normal_patch_duplication_used", "vessel_feature_used", "roi_masked_loss_used"}
    g_rows = []
    n_gfail = 0
    for k, v in guardrail_items.items():
        expected = False if k in false_keys else True
        ok = (v == expected)
        if not ok:
            n_gfail += 1
        g_rows.append({"key": k, "value": v, "expected": expected, "status": "OK" if ok else "FAIL"})
    _write_csv(g_rows, REPORT_ROOT / "p_c_normal31_guardrail_check.csv")

    # verdict
    verdict = "PASS"
    fail_reasons = []
    if phase_a_verdict == "FAIL":
        verdict = "FAIL"; fail_reasons.append("phase_a_mask_fail")
    if errors:
        verdict = "PARTIAL_PASS" if verdict != "FAIL" else "FAIL"
        fail_reasons.extend([e["error"] for e in errors])
    if n_gfail > 0:
        verdict = "FAIL"; fail_reasons.append(f"guardrail_fail={n_gfail}")
    if repro_status == "WARN" and verdict == "PASS":
        verdict = "PARTIAL_PASS"; fail_reasons.append("reference_reproducibility_warn")

    # summary
    summary = {
        "stage":           STAGE_LABEL,
        "timestamp":       ts,
        "verdict":         verdict,
        "fail_reasons":    fail_reasons,
        "phase_a_verdict": phase_a_verdict,
        "repro_status":    repro_status,
        "n_total":   n_total, "n_normal": n_normal, "n_nsclc": n_nsclc,
        "reference_balanced_w1": {
            "auroc": m_r["auroc"], "auprc": m_r["auprc"], "brier": m_r["brier"],
            "specificity": m_r["specificity"], "sensitivity": m_r["sensitivity"],
            "FP": m_r["FP"], "FN": m_r["FN"],
        },
        "masked_30b": {
            "auroc": m_c["auroc"], "auprc": m_c["auprc"], "brier": m_c["brier"],
            "specificity": m_c["specificity"], "sensitivity": m_c["sensitivity"],
            "FP": m_c["FP"], "FN": m_c["FN"],
        },
        "delta_masked_minus_ref": {
            "auroc": round(auroc_delta, 6), "specificity": round(spec_delta, 6),
            "FP": fp_delta, "FN": fn_delta,
        },
        "masking_effect_note": masking_effect,
        "guardrail_fail":  n_gfail,
        "n_errors":        len(errors),
    }
    _write_json(summary, REPORT_ROOT / "p_c_normal31_repaired_final_test_masked_comparison_summary.json")

    if errors:
        _write_csv(errors, REPORT_ROOT / "p_c_normal31_errors.csv")

    if verdict in ("PASS", "PARTIAL_PASS"):
        _write_json({"stage": STAGE_LABEL, "verdict": verdict, "timestamp": ts,
                     "next_step": "P-C-NORMAL32 decision checkpoint (사용자 승인 필요)"},
                    REPORT_ROOT / "DONE.json")

    print(f"[{STAGE_LABEL}] {'='*60}")
    print(f"[{STAGE_LABEL}] VERDICT: {verdict}")
    for r in fail_reasons:
        print(f"[{STAGE_LABEL}]   FAIL: {r}")
    print(f"[{STAGE_LABEL}] ref AUROC={m_r['auroc']:.4f} masked AUROC={m_c['auroc']:.4f} delta={auroc_delta:+.4f}")
    print(f"[{STAGE_LABEL}] ref FP={m_r['FP']} masked FP={m_c['FP']} delta={fp_delta:+d}")
    print(f"[{STAGE_LABEL}] {'='*60}")

    return 0 if verdict in ("PASS", "PARTIAL_PASS") else 1


def main():
    parser = argparse.ArgumentParser(description="P-C-NORMAL31 repaired final_test masked comparison")
    parser.add_argument("--run", action="store_true", help="실행 확인 flag (required)")
    args = parser.parse_args()
    if not args.run:
        print("[GUARD] --run flag required.", file=sys.stderr)
        sys.exit(2)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
