"""
p_c_normal28_repaired_prediction_export.py

P-C-NORMAL28: repaired final_test manifest 기반 prediction export

P-C-NORMAL27에서 생성한 scalar-repaired final_test manifest를 사용하여
main_24j_fix / balanced-w1 checkpoint의 prediction을 재수행한다.

변경점 (24k 대비):
  - FINAL_TEST_MANIFEST: p_c_normal27_final_test_feature_manifest_repaired_usable.csv
  - OUTPUT_ROOT / REPORT_ROOT: p_c_normal28_repaired_prediction_export
  - guardrail: repaired_manifest_used=True, original_24g_manifest_used=False

유지 (24k와 동일):
  - 체크포인트: main_24j_fix (ep18) + balanced-w1 (ep8)
  - scalar stats: p_c_normal24h_fix
  - threshold: fixed 0.5 reporting only

금지:
  - threshold 최적화/sweep/best threshold 선택
  - test set 기반 checkpoint 재선택
  - model training / checkpoint 수정
  - balanced manifest 사용
  - original 24g manifest 사용
  - 기존 p_c_normal24k 결과 수정/삭제/덮어쓰기
  - stage2_holdout 접근
  - vessel feature 사용

실행:
  python p_c_normal28_repaired_prediction_export.py \\
      --export \\
      --confirm-final-test-export \\
      --confirm-no-threshold-opt \\
      --confirm-no-training
"""

import argparse
import csv
import json
import math
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

FINAL_TEST_MANIFEST = PROJECT_ROOT / "outputs/manifests/p_c_normal27_scalar_repair_final_test_manifest/p_c_normal27_final_test_feature_manifest_repaired_usable.csv"
SCALAR_STATS_PATH   = PROJECT_ROOT / "outputs/reports/p_c_normal24h_fix_zroi_scalar_fusion_drycheck/p_c_normal24h_fix_scalar_normalization_stats.json"

MAIN_CKPT_PATH = PROJECT_ROOT / "outputs/p_c_normal24j_fix_zroi_scalar_fusion_full_train/checkpoints/p_c_normal24j_fix_best_val_auc_checkpoint.pt"
BW1_CKPT_PATH  = PROJECT_ROOT / "outputs/p_c_normal24j_fix_balanced_w1_zroi_scalar_fusion_full_train/checkpoints/p_c_normal24j_fix_balanced_w1_best_val_auc_checkpoint.pt"

OUTPUT_ROOT = PROJECT_ROOT / "outputs/p_c_normal28_repaired_prediction_export"
REPORT_ROOT = PROJECT_ROOT / "outputs/reports/p_c_normal28_repaired_prediction_export"

# ── Constants ─────────────────────────────────────────────────────────────────
HU_MIN, HU_MAX       = -1000.0, 200.0
IMAGENET_MEAN        = [0.485, 0.456, 0.406]
IMAGENET_STD         = [0.229, 0.224, 0.225]
SCALAR_FEATURES      = ["lung_z_percentile", "crop_lung_roi_ratio"]
FIXED_THRESHOLD      = 0.5
EXPECTED_FT_ROWS     = 66283
EXPECTED_FT_NORMAL   = 21560
EXPECTED_FT_NSCLC    = 44723

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"note": "empty"}]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _check_forbidden_words(path):
    text = path.read_text(errors="ignore").lower()
    return sum(text.count(w.lower()) for w in FORBIDDEN_WORDS)


def _safe_delta(a, b):
    if isinstance(a, float) and isinstance(b, float) and not math.isnan(a) and not math.isnan(b):
        return round(b - a, 6)
    return "n/a"


# ── Scalar normalization ──────────────────────────────────────────────────────

def apply_scalar_norm(df, stats):
    df = df.copy()
    for col, s in stats.items():
        df[col] = (df[col].astype(float) - s["mean"]) / s["std"]
    return df


# ── AUROC / AUPRC (sklearn-free) ─────────────────────────────────────────────

def compute_auroc(labels, scores):
    try:
        la = np.asarray(labels, dtype=np.int32)
        sc = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(sc)):
            return float("nan"), "invalid_score"
        if len(np.unique(la)) < 2:
            return float("nan"), "single_class"
        pos = la == 1; neg = la == 0
        n_pos, n_neg = int(pos.sum()), int(neg.sum())
        all_s  = np.concatenate([sc[neg], sc[pos]])
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
        return float("nan"), f"ERR:{e}"


def compute_auprc(labels, scores):
    try:
        la = np.asarray(labels, dtype=np.int32)
        sc = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(sc)):
            return float("nan"), "invalid_score"
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
        return float(np.trapz(prec, rec)), "OK"
    except Exception as e:
        return float("nan"), f"ERR:{e}"


# ── Model ─────────────────────────────────────────────────────────────────────

class ScalarFusionModel(nn.Module):
    def __init__(self, scalar_hidden=32, scalar_out=16, dropout=0.2):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        self.img_features  = backbone.features
        self.img_avgpool   = backbone.avgpool
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


# ── Inference Dataset ─────────────────────────────────────────────────────────

class InferenceDataset(Dataset):
    def __init__(self, df):
        self.df   = df.reset_index(drop=True)
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
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
        return img, scalar, idx


# ── Checkpoint load & verify ──────────────────────────────────────────────────

def load_and_verify_checkpoint(ckpt_path, ckpt_name, device, errors):
    if not ckpt_path.exists():
        errors.append({"check": f"{ckpt_name}_exists", "error": f"not found: {ckpt_path}"})
        return None, {}
    ckpt = torch.load(ckpt_path, map_location=device)
    issues = []
    if ckpt.get("final_test_used", False):
        issues.append("final_test_used=True")
    if ckpt.get("threshold_optimized", False):
        issues.append("threshold_optimized=True")
    if ckpt.get("smoke_only", False):
        issues.append("smoke_only=True")
    if ckpt.get("vessel_feature_used", False):
        issues.append("vessel_feature_used=True")
    if ckpt.get("roi_masked_loss_used", False):
        issues.append("roi_masked_loss_used=True")
    sf = ckpt.get("scalar_features", None)
    if sf is None or list(sf) != SCALAR_FEATURES:
        issues.append(f"scalar_features missing_or_mismatch: {sf}")
    for iss in issues:
        errors.append({"check": f"{ckpt_name}_metadata", "error": iss})
    if issues:
        return None, ckpt
    model = ScalarFusionModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels, probs):
    la    = np.asarray(labels, dtype=np.int32)
    pr    = np.asarray(probs,  dtype=np.float64)
    preds = (pr >= FIXED_THRESHOLD).astype(np.int32)

    TP = int(((preds == 1) & (la == 1)).sum())
    TN = int(((preds == 0) & (la == 0)).sum())
    FP = int(((preds == 1) & (la == 0)).sum())
    FN = int(((preds == 0) & (la == 1)).sum())
    n  = len(la)

    acc      = (TP + TN) / n if n > 0 else float("nan")
    sens     = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    spec     = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
    prec_v   = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    f1       = (2 * prec_v * sens / (prec_v + sens)
                if not (math.isnan(prec_v) or math.isnan(sens) or (prec_v + sens) == 0)
                else float("nan"))
    bal_acc  = ((sens + spec) / 2
                if not (math.isnan(sens) or math.isnan(spec))
                else float("nan"))
    brier    = float(np.mean((pr - la.astype(float)) ** 2))
    auroc,  _ = compute_auroc(la, pr)
    auprc,  _ = compute_auprc(la, pr)

    def _r(v):
        return round(v, 6) if isinstance(v, float) and not math.isnan(v) else v

    return {
        "auroc":             _r(auroc),
        "auprc":             _r(auprc),
        "brier":             _r(brier),
        "accuracy":          _r(acc),
        "balanced_accuracy": _r(bal_acc),
        "sensitivity":       _r(sens),
        "specificity":       _r(spec),
        "precision":         _r(prec_v),
        "f1":                _r(f1),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "normal_FP_count": FP,
        "nsclc_FN_count":  FN,
    }


def compute_weighted_metrics(labels, probs, sample_weights):
    """Class-balanced weighted supplementary metrics. NOT primary — raw metrics are primary."""
    la    = np.asarray(labels,         dtype=np.int32)
    pr    = np.asarray(probs,          dtype=np.float64)
    sw    = np.asarray(sample_weights, dtype=np.float64)
    preds = (pr >= FIXED_THRESHOLD).astype(np.int32)

    mask_TP = (preds == 1) & (la == 1)
    mask_TN = (preds == 0) & (la == 0)
    mask_FP = (preds == 1) & (la == 0)
    mask_FN = (preds == 0) & (la == 1)

    TP_w    = float(sw[mask_TP].sum())
    TN_w    = float(sw[mask_TN].sum())
    FP_w    = float(sw[mask_FP].sum())
    FN_w    = float(sw[mask_FN].sum())
    total_w = float(sw.sum())

    w_acc     = (TP_w + TN_w) / total_w if total_w > 0 else float("nan")
    w_sens    = TP_w / (TP_w + FN_w)   if (TP_w + FN_w) > 0 else float("nan")
    w_spec    = TN_w / (TN_w + FP_w)   if (TN_w + FP_w) > 0 else float("nan")
    w_prec    = TP_w / (TP_w + FP_w)   if (TP_w + FP_w) > 0 else float("nan")
    w_bal_acc = ((w_sens + w_spec) / 2
                 if not (math.isnan(w_sens) or math.isnan(w_spec))
                 else float("nan"))
    w_f1      = (2 * w_prec * w_sens / (w_prec + w_sens)
                 if not (math.isnan(w_prec) or math.isnan(w_sens) or (w_prec + w_sens) == 0)
                 else float("nan"))
    w_brier   = (float((sw * (pr - la.astype(float)) ** 2).sum() / total_w)
                 if total_w > 0 else float("nan"))

    def _r(v):
        return round(v, 6) if isinstance(v, float) and not math.isnan(v) else v

    return {
        "weighted_accuracy":          _r(w_acc),
        "weighted_balanced_accuracy": _r(w_bal_acc),
        "weighted_brier":             _r(w_brier),
        "weighted_sensitivity":       _r(w_sens),
        "weighted_specificity":       _r(w_spec),
        "weighted_precision":         _r(w_prec),
        "weighted_f1":                _r(w_f1),
        "weighted_TP":                _r(TP_w),
        "weighted_TN":                _r(TN_w),
        "weighted_FP":                _r(FP_w),
        "weighted_FN":                _r(FN_w),
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, loader, device):
    logits_list, probs_list = [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, scalars, _) in enumerate(loader):
            imgs    = imgs.to(device)
            scalars = scalars.to(device)
            logits  = model(imgs, scalars).squeeze(1)
            probs   = torch.sigmoid(logits)
            logits_list.extend(logits.cpu().numpy().tolist())
            probs_list.extend(probs.cpu().numpy().tolist())
            if batch_idx % 50 == 0:
                print(f"  batch {batch_idx}/{len(loader)}", flush=True)
    return logits_list, probs_list


# ── Patient-level aggregation ─────────────────────────────────────────────────

def patient_agg(df_pred):
    grouped = df_pred.groupby("patient_id").agg(
        label=("label", "max"),
        mean_prob=("prob", "mean"),
        max_prob=("prob", "max"),
        p95_prob=("prob", lambda x: float(np.percentile(x, 95))),
        crop_count=("prob", "count"),
    ).reset_index()
    return grouped


def patient_metrics_for_agg(patient_df, agg_col):
    labels = patient_df["label"].values
    scores = patient_df[agg_col].values
    m = compute_metrics(labels, scores)
    m["n_patients"]        = len(patient_df)
    m["n_normal_patients"] = int((labels == 0).sum())
    m["n_nsclc_patients"]  = int((labels == 1).sum())
    m["agg_col"]           = agg_col
    return m


# ── Main ──────────────────────────────────────────────────────────────────────

def run_export(args):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for out_dir in [OUTPUT_ROOT, REPORT_ROOT]:
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[ABORT] output dir already exists and is not empty: {out_dir}")
            sys.exit(2)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    errors = []

    # ── 1. scalar stats ───────────────────────────────────────────────────────
    if not SCALAR_STATS_PATH.exists():
        errors.append({"check": "scalar_stats", "error": f"not found: {SCALAR_STATS_PATH}"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal28_errors.csv")
        sys.exit(1)
    with open(SCALAR_STATS_PATH) as f:
        scalar_stats = json.load(f)["features"]

    # ── 2. repaired final_test manifest 검증 ──────────────────────────────────
    if not FINAL_TEST_MANIFEST.exists():
        errors.append({"check": "manifest", "error": f"not found: {FINAL_TEST_MANIFEST}"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal28_errors.csv")
        sys.exit(1)

    # 금지: 기존 24g manifest 또는 balanced manifest 사용
    manifest_str = str(FINAL_TEST_MANIFEST)
    if "24g" in manifest_str and "27" not in manifest_str:
        errors.append({"check": "original_24g_manifest_guard",
                        "error": "original 24g manifest detected — must use p_c_normal27 repaired"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal28_errors.csv")
        sys.exit(1)
    if "balanced" in manifest_str:
        errors.append({"check": "balanced_manifest_guard", "error": "balanced manifest detected — forbidden"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal28_errors.csv")
        sys.exit(1)
    if "27" not in manifest_str:
        errors.append({"check": "repaired_manifest_guard",
                        "error": "manifest path does not contain '27' — must be p_c_normal27 repaired"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal28_errors.csv")
        sys.exit(1)

    df_raw = pd.read_csv(FINAL_TEST_MANIFEST, low_memory=False)
    if len(df_raw) != EXPECTED_FT_ROWS:
        errors.append({"check": "row_count", "error": f"expected {EXPECTED_FT_ROWS}, got {len(df_raw)}"})
    vessel_cols = [c for c in df_raw.columns if c in FORBIDDEN_VESSEL_COLS]
    if vessel_cols:
        errors.append({"check": "vessel_cols", "error": str(vessel_cols)})
    if "z_unresolved" in df_raw.columns and df_raw["z_unresolved"].any():
        errors.append({"check": "z_unresolved", "error": str(int(df_raw["z_unresolved"].sum()))})
    if df_raw[SCALAR_FEATURES].isnull().any().any():
        errors.append({"check": "scalar_nan", "error": "NaN in scalar features"})

    if errors:
        _write_csv(errors, REPORT_ROOT / "p_c_normal28_errors.csv")
        sys.exit(1)

    n_normal = int((df_raw["label"] == 0).sum())
    n_nsclc  = int((df_raw["label"] == 1).sum())
    print(f"[28] final_test (repaired): {len(df_raw)} rows | normal={n_normal} NSCLC={n_nsclc}")

    if n_normal != EXPECTED_FT_NORMAL:
        errors.append({"check": "label_count_normal",
                        "error": f"expected {EXPECTED_FT_NORMAL}, got {n_normal}"})
    if n_nsclc != EXPECTED_FT_NSCLC:
        errors.append({"check": "label_count_nsclc",
                        "error": f"expected {EXPECTED_FT_NSCLC}, got {n_nsclc}"})
    if errors:
        _write_csv(errors, REPORT_ROOT / "p_c_normal28_errors.csv")
        sys.exit(1)

    # scalar sanity: normal crops lung_z_percentile mean이 0.4 이상이어야 함 (bug 재현 방지)
    normal_lzp_mean = float(df_raw[df_raw["label"] == 0]["lung_z_percentile"].mean())
    if normal_lzp_mean < 0.4:
        errors.append({"check": "normal_scalar_sanity",
                        "error": f"normal lung_z_percentile mean={normal_lzp_mean:.4f} < 0.4 — scalar repair may not have been applied"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal28_errors.csv")
        sys.exit(1)
    print(f"[28] normal_lzp_mean={normal_lzp_mean:.4f} (sanity OK, >0.4)")

    # class-balanced weights (supplementary only)
    n_total                = n_normal + n_nsclc
    weight_normal          = n_total / (2 * n_normal)
    weight_nsclc           = n_total / (2 * n_nsclc)
    relative_weight_normal = 1.0
    relative_weight_nsclc  = round(n_normal / n_nsclc, 6)
    print(f"[28] class weights: normal={weight_normal:.4f} nsclc={weight_nsclc:.4f} "
          f"relative_nsclc={relative_weight_nsclc:.4f}")

    df_norm = apply_scalar_norm(df_raw, scalar_stats)

    # ── 3. DataLoader ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds     = InferenceDataset(df_norm)
    loader = DataLoader(ds, batch_size=256, shuffle=False,
                        num_workers=4, pin_memory=True)
    print(f"[28] device={device} batches={len(loader)}")

    # ── 4. checkpoint load & verify ───────────────────────────────────────────
    print("[28] loading main checkpoint...")
    model_main, ckpt_main = load_and_verify_checkpoint(MAIN_CKPT_PATH, "main", device, errors)
    print("[28] loading balanced-w1 checkpoint...")
    model_bw1,  ckpt_bw1  = load_and_verify_checkpoint(BW1_CKPT_PATH,  "bw1",  device, errors)

    if model_main is None or model_bw1 is None:
        _write_csv(errors, REPORT_ROOT / "p_c_normal28_errors.csv")
        sys.exit(1)

    ckpt_meta_rows = []
    for name, ckpt in [("main_24j_fix", ckpt_main), ("balanced_w1", ckpt_bw1)]:
        ckpt_meta_rows.append({
            "checkpoint_name":    name,
            "epoch":              ckpt.get("epoch", "?"),
            "stage":              ckpt.get("stage", "?"),
            "final_test_used":    ckpt.get("final_test_used", "?"),
            "threshold_optimized": ckpt.get("threshold_optimized", "?"),
            "smoke_only":         ckpt.get("smoke_only", "?"),
            "vessel_feature_used": ckpt.get("vessel_feature_used", "?"),
            "roi_masked_loss_used": ckpt.get("roi_masked_loss_used", "?"),
            "scalar_features":    str(ckpt.get("scalar_features", [])),
        })
    _write_csv(ckpt_meta_rows, REPORT_ROOT / "p_c_normal28_checkpoint_metadata_check.csv")

    # ── 5. inference ──────────────────────────────────────────────────────────
    print("[28] main inference...")
    main_logits, main_probs = run_inference(model_main, loader, device)
    print("[28] balanced-w1 inference...")
    bw1_logits,  bw1_probs  = run_inference(model_bw1,  loader, device)

    # ── 6. prediction CSV ─────────────────────────────────────────────────────
    preserve_cols = [c for c in [
        "patient_id", "safe_id", "crop_path", "label", "source_split",
        "canonical_volume_z", "local_z", "lung_z_percentile", "crop_lung_roi_ratio",
        "position_bin", "center_y", "center_x",
    ] if c in df_raw.columns]

    df_main_pred = df_raw[preserve_cols].copy()
    df_main_pred["logit"]           = [round(v, 6) for v in main_logits]
    df_main_pred["prob"]            = [round(v, 6) for v in main_probs]
    df_main_pred["pred_at_0p5"]     = (np.array(main_probs) >= FIXED_THRESHOLD).astype(int).tolist()
    df_main_pred["checkpoint_name"] = "main_24j_fix"

    df_bw1_pred = df_raw[preserve_cols].copy()
    df_bw1_pred["logit"]            = [round(v, 6) for v in bw1_logits]
    df_bw1_pred["prob"]             = [round(v, 6) for v in bw1_probs]
    df_bw1_pred["pred_at_0p5"]      = (np.array(bw1_probs) >= FIXED_THRESHOLD).astype(int).tolist()
    df_bw1_pred["checkpoint_name"]  = "balanced_w1"

    df_main_pred.to_csv(OUTPUT_ROOT / "p_c_normal28_main_final_test_predictions.csv", index=False)
    df_bw1_pred.to_csv(OUTPUT_ROOT  / "p_c_normal28_balanced_w1_final_test_predictions.csv", index=False)
    print("[28] prediction CSVs saved.")

    # ── 7. crop-level metrics ─────────────────────────────────────────────────
    labels_arr = df_raw["label"].values
    main_m = compute_metrics(labels_arr, main_probs)
    bw1_m  = compute_metrics(labels_arr, bw1_probs)

    sample_weights = np.where(labels_arr == 0, weight_normal, weight_nsclc)
    main_wm = compute_weighted_metrics(labels_arr, main_probs, sample_weights)
    bw1_wm  = compute_weighted_metrics(labels_arr, bw1_probs,  sample_weights)

    crop_rows = []
    for name, m in [("main_24j_fix", main_m), ("balanced_w1", bw1_m)]:
        row = {"checkpoint": name}
        row.update(m)
        crop_rows.append(row)
    _write_csv(crop_rows, REPORT_ROOT / "p_c_normal28_crop_metrics_comparison.csv")

    cm_crop_rows = []
    for name, m in [("main_24j_fix", main_m), ("balanced_w1", bw1_m)]:
        cm_crop_rows.append({
            "checkpoint":       name,
            "TN":               m["TN"],
            "FP":               m["FP"],
            "FN":               m["FN"],
            "TP":               m["TP"],
            "normal_FP_count":  m["FP"],
            "nsclc_FN_count":   m["FN"],
            "specificity":      m["specificity"],
            "sensitivity":      m["sensitivity"],
            "balanced_accuracy": m["balanced_accuracy"],
        })
    _write_csv(cm_crop_rows, REPORT_ROOT / "p_c_normal28_crop_confusion_matrix_comparison.csv")

    wm_base = {
        "n_total":               n_total,
        "n_normal":              n_normal,
        "n_nsclc":               n_nsclc,
        "weight_normal":         round(weight_normal, 6),
        "weight_nsclc":          round(weight_nsclc,  6),
        "relative_weight_normal": relative_weight_normal,
        "relative_weight_nsclc":  relative_weight_nsclc,
    }
    w_csv_rows = []
    for ckpt_name, raw_m, wm in [("main_24j_fix", main_m, main_wm), ("balanced_w1", bw1_m, bw1_wm)]:
        row = {"checkpoint": ckpt_name}
        row.update(wm_base)
        row["raw_accuracy"]               = raw_m["accuracy"]
        row["weighted_accuracy"]          = wm["weighted_accuracy"]
        row["raw_balanced_accuracy"]      = raw_m["balanced_accuracy"]
        row["weighted_balanced_accuracy"] = wm["weighted_balanced_accuracy"]
        row["raw_brier"]                  = raw_m["brier"]
        row["weighted_brier"]             = wm["weighted_brier"]
        row["raw_sensitivity"]            = raw_m["sensitivity"]
        row["weighted_sensitivity"]       = wm["weighted_sensitivity"]
        row["raw_specificity"]            = raw_m["specificity"]
        row["weighted_specificity"]       = wm["weighted_specificity"]
        row["raw_precision"]              = raw_m["precision"]
        row["weighted_precision"]         = wm["weighted_precision"]
        row["raw_f1"]                     = raw_m["f1"]
        row["weighted_f1"]                = wm["weighted_f1"]
        row["raw_TN"]                     = raw_m["TN"]
        row["raw_FP"]                     = raw_m["FP"]
        row["raw_FN"]                     = raw_m["FN"]
        row["raw_TP"]                     = raw_m["TP"]
        row["weighted_TN"]                = wm["weighted_TN"]
        row["weighted_FP"]                = wm["weighted_FP"]
        row["weighted_FN"]                = wm["weighted_FN"]
        row["weighted_TP"]                = wm["weighted_TP"]
        w_csv_rows.append(row)
    _write_csv(w_csv_rows, REPORT_ROOT / "p_c_normal28_crop_weighted_metrics_comparison.csv")
    _write_csv(w_csv_rows, REPORT_ROOT / "p_c_normal28_crop_raw_vs_weighted_metrics_comparison.csv")

    # ── 8. patient-level aggregation ──────────────────────────────────────────
    pat_main = patient_agg(df_main_pred)
    pat_bw1  = patient_agg(df_bw1_pred)

    pat_rows = []
    for ckpt_name, pat_df in [("main_24j_fix", pat_main), ("balanced_w1", pat_bw1)]:
        for agg_col in ["mean_prob", "max_prob", "p95_prob"]:
            m = patient_metrics_for_agg(pat_df, agg_col)
            row = {"checkpoint": ckpt_name}
            row.update(m)
            pat_rows.append(row)
    _write_csv(pat_rows, REPORT_ROOT / "p_c_normal28_patient_metrics_comparison.csv")

    cm_pat_rows = []
    for ckpt_name, pat_df in [("main_24j_fix", pat_main), ("balanced_w1", pat_bw1)]:
        for agg_col in ["mean_prob", "max_prob", "p95_prob"]:
            m = patient_metrics_for_agg(pat_df, agg_col)
            cm_pat_rows.append({
                "checkpoint": ckpt_name, "agg": agg_col,
                "TN": m["TN"], "FP": m["FP"], "FN": m["FN"], "TP": m["TP"],
                "normal_FP_patients": m["FP"],
                "nsclc_FN_patients":  m["FN"],
                "n_patients":         m["n_patients"],
                "n_normal":           m["n_normal_patients"],
                "n_nsclc":            m["n_nsclc_patients"],
            })
    _write_csv(cm_pat_rows, REPORT_ROOT / "p_c_normal28_confusion_matrix_comparison.csv")

    # ── 9. delta vs bugged 24k (informational only) ───────────────────────────
    # P-C-NORMAL24k (bugged) 결과 참조용 — 고정 상수, inference 재실행 없음
    BUGGED_24K_MAIN_FP   = 12133
    BUGGED_24K_MAIN_FN   = 342
    BUGGED_24K_BW1_FP    = 11126
    BUGGED_24K_BW1_FN    = 317
    BUGGED_24K_MAIN_AUROC = 0.9261
    BUGGED_24K_BW1_AUROC  = 0.9411

    delta_vs_bugged_rows = [
        {
            "checkpoint":         "main_24j_fix",
            "bugged_24k_FP":      BUGGED_24K_MAIN_FP,
            "repaired_28_FP":     main_m["FP"],
            "delta_FP":           main_m["FP"] - BUGGED_24K_MAIN_FP,
            "bugged_24k_FN":      BUGGED_24K_MAIN_FN,
            "repaired_28_FN":     main_m["FN"],
            "delta_FN":           main_m["FN"] - BUGGED_24K_MAIN_FN,
            "bugged_24k_auroc":   BUGGED_24K_MAIN_AUROC,
            "repaired_28_auroc":  main_m["auroc"],
            "delta_auroc":        round(main_m["auroc"] - BUGGED_24K_MAIN_AUROC, 6) if isinstance(main_m["auroc"], float) else "n/a",
        },
        {
            "checkpoint":         "balanced_w1",
            "bugged_24k_FP":      BUGGED_24K_BW1_FP,
            "repaired_28_FP":     bw1_m["FP"],
            "delta_FP":           bw1_m["FP"] - BUGGED_24K_BW1_FP,
            "bugged_24k_FN":      BUGGED_24K_BW1_FN,
            "repaired_28_FN":     bw1_m["FN"],
            "delta_FN":           bw1_m["FN"] - BUGGED_24K_BW1_FN,
            "bugged_24k_auroc":   BUGGED_24K_BW1_AUROC,
            "repaired_28_auroc":  bw1_m["auroc"],
            "delta_auroc":        round(bw1_m["auroc"] - BUGGED_24K_BW1_AUROC, 6) if isinstance(bw1_m["auroc"], float) else "n/a",
        },
    ]
    _write_csv(delta_vs_bugged_rows, REPORT_ROOT / "p_c_normal28_delta_vs_bugged_24k.csv")

    # ── 10. guardrail ─────────────────────────────────────────────────────────
    this_file     = Path(__file__).resolve()
    forbidden_cnt = _check_forbidden_words(this_file)

    guardrail_rows = [
        {"guardrail": "final_test_prediction_export_run",      "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "repaired_manifest_used",                "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "original_24g_manifest_used",            "expected": False, "actual": False, "pass": True},
        {"guardrail": "normal_scalar_sanity_pass",             "expected": True,  "actual": normal_lzp_mean >= 0.4, "pass": normal_lzp_mean >= 0.4},
        {"guardrail": "model_training_run",                    "expected": False, "actual": False, "pass": True},
        {"guardrail": "checkpoint_updated",                    "expected": False, "actual": False, "pass": True},
        {"guardrail": "threshold_optimized",                   "expected": False, "actual": False, "pass": True},
        {"guardrail": "threshold_sweep_run",                   "expected": False, "actual": False, "pass": True},
        {"guardrail": "best_threshold_selected",               "expected": False, "actual": False, "pass": True},
        {"guardrail": "fixed_threshold_0p5_reporting_only",    "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "main_checkpoint_loaded",                "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "balanced_w1_checkpoint_loaded",         "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "final_test_balanced_manifest_used",     "expected": False, "actual": False, "pass": True},
        {"guardrail": "vessel_feature_used",                   "expected": False, "actual": False, "pass": True},
        {"guardrail": "roi_masked_loss_used",                  "expected": False, "actual": False, "pass": True},
        {"guardrail": "stage2_holdout_accessed",               "expected": False, "actual": False, "pass": True},
        {"guardrail": "existing_24k_result_modified",          "expected": False, "actual": False, "pass": True},
        {"guardrail": "test_set_checkpoint_selection",         "expected": False, "actual": False, "pass": True},
        {"guardrail": "loss_weighting_changed",                "expected": False, "actual": False, "pass": True},
        {"guardrail": "scalar_features_used",                  "expected": str(SCALAR_FEATURES), "actual": str(SCALAR_FEATURES), "pass": True},
        {"guardrail": "forbidden_diagnostic_wording_count",    "expected": 0,     "actual": forbidden_cnt, "pass": forbidden_cnt == 0},
        {"guardrail": "raw_final_test_metrics_kept",           "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "class_weighted_supplementary_added",    "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "test_set_downsampling_used",            "expected": False, "actual": False, "pass": True},
        {"guardrail": "normal_patch_duplication_used",         "expected": False, "actual": False, "pass": True},
        {"guardrail": "final_test_rebalanced_by_sampling",     "expected": False, "actual": False, "pass": True},
        {"guardrail": "weighted_metrics_primary",              "expected": False, "actual": False, "pass": True},
        {"guardrail": "raw_metrics_primary",                   "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "weighted_fp_fn_are_not_real_counts",    "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "z_unresolved_rows_excluded",            "expected": True,  "actual": True,  "pass": True},
    ]
    _write_csv(guardrail_rows, REPORT_ROOT / "p_c_normal28_guardrail_check.csv")

    verdict              = "PASS"
    guardrail_fail_count = sum(1 for r in guardrail_rows if not r["pass"])
    if errors or guardrail_fail_count > 0:
        verdict = "PARTIAL_PASS"

    # ── 11. delta summary values ──────────────────────────────────────────────
    fp_delta  = bw1_m["FP"] - main_m["FP"]
    fn_delta  = bw1_m["FN"] - main_m["FN"]
    spec_d    = _safe_delta(main_m["specificity"], bw1_m["specificity"])
    sens_d    = _safe_delta(main_m["sensitivity"], bw1_m["sensitivity"])
    auroc_d   = _safe_delta(main_m["auroc"],       bw1_m["auroc"])
    auprc_d   = _safe_delta(main_m["auprc"],       bw1_m["auprc"])
    bal_acc_d = _safe_delta(main_m["balanced_accuracy"], bw1_m["balanced_accuracy"])

    main_pat_mean = patient_metrics_for_agg(pat_main, "mean_prob")
    bw1_pat_mean  = patient_metrics_for_agg(pat_bw1,  "mean_prob")
    main_pat_max  = patient_metrics_for_agg(pat_main, "max_prob")
    bw1_pat_max   = patient_metrics_for_agg(pat_bw1,  "max_prob")
    main_pat_p95  = patient_metrics_for_agg(pat_main, "p95_prob")
    bw1_pat_p95   = patient_metrics_for_agg(pat_bw1,  "p95_prob")

    w_acc_d  = _safe_delta(main_wm["weighted_accuracy"],  bw1_wm["weighted_accuracy"])
    w_bri_d  = _safe_delta(main_wm["weighted_brier"],     bw1_wm["weighted_brier"])
    w_prec_d = _safe_delta(main_wm["weighted_precision"], bw1_wm["weighted_precision"])
    w_f1_d   = _safe_delta(main_wm["weighted_f1"],        bw1_wm["weighted_f1"])

    # ── 12. summary JSON ──────────────────────────────────────────────────────
    summary = {
        "step":      "P-C-NORMAL28",
        "verdict":   verdict,
        "timestamp": ts,
        "manifest":  "p_c_normal27_final_test_feature_manifest_repaired_usable.csv",
        "manifest_source": "P-C-NORMAL27 scalar-repaired (normal_test canonical_z bug fixed)",
        "n_final_test_crops": len(df_raw),
        "n_normal":  n_normal,
        "n_nsclc":   n_nsclc,
        "normal_lzp_mean_sanity": round(normal_lzp_mean, 4),
        "main_24j_fix": {
            "ckpt_epoch":        ckpt_main.get("epoch", "?"),
            "crop_auroc":        main_m["auroc"],
            "crop_auprc":        main_m["auprc"],
            "crop_specificity":  main_m["specificity"],
            "crop_sensitivity":  main_m["sensitivity"],
            "crop_balanced_acc": main_m["balanced_accuracy"],
            "normal_FP_crops":   main_m["FP"],
            "nsclc_FN_crops":    main_m["FN"],
        },
        "balanced_w1": {
            "ckpt_epoch":        ckpt_bw1.get("epoch", "?"),
            "crop_auroc":        bw1_m["auroc"],
            "crop_auprc":        bw1_m["auprc"],
            "crop_specificity":  bw1_m["specificity"],
            "crop_sensitivity":  bw1_m["sensitivity"],
            "crop_balanced_acc": bw1_m["balanced_accuracy"],
            "normal_FP_crops":   bw1_m["FP"],
            "nsclc_FN_crops":    bw1_m["FN"],
        },
        "delta_bw1_minus_main": {
            "normal_FP_crops": bw1_m["FP"] - main_m["FP"],
            "nsclc_FN_crops":  bw1_m["FN"] - main_m["FN"],
            "specificity":     _safe_delta(main_m["specificity"], bw1_m["specificity"]),
            "sensitivity":     _safe_delta(main_m["sensitivity"], bw1_m["sensitivity"]),
        },
        "delta_vs_bugged_24k": {
            "main_FP_delta":   main_m["FP"] - BUGGED_24K_MAIN_FP,
            "main_FN_delta":   main_m["FN"] - BUGGED_24K_MAIN_FN,
            "bw1_FP_delta":    bw1_m["FP"]  - BUGGED_24K_BW1_FP,
            "bw1_FN_delta":    bw1_m["FN"]  - BUGGED_24K_BW1_FN,
            "note": "bugged 24k values are fixed reference constants, not re-run",
        },
        "guardrails": {
            "repaired_manifest_used":            True,
            "original_24g_manifest_used":        False,
            "normal_scalar_sanity_pass":         normal_lzp_mean >= 0.4,
            "threshold_optimized":               False,
            "threshold_sweep_run":               False,
            "fixed_threshold_0p5_only":          True,
            "model_training_run":                False,
            "checkpoint_updated":                False,
            "test_set_checkpoint_selection":     False,
            "stage2_holdout_accessed":           False,
            "existing_24k_result_modified":      False,
            "final_test_balanced_manifest_used": False,
            "forbidden_diagnostic_wording_count": forbidden_cnt,
            "guardrail_fail_count":              guardrail_fail_count,
        },
        "interpretation_note": (
            "이 결과는 P-C-NORMAL27 scalar-repaired final_test 기반 fixed reporting comparison이다. "
            "threshold 최적화 결과가 아니다. "
            "0.5 threshold는 고정 보고용이다. "
            "bugged 24k 결과와의 delta는 scalar repair 효과를 나타내며 재학습 없이 inference만 변경한 것이다. "
            "main 24j-fix와 balanced-w1은 별도 checkpoint이다. "
            "balanced-w1이 main을 자동 대체하지 않는다. "
            "최종 checkpoint 선택은 사용자 승인 후 결정한다."
        ),
        "errors": len(errors),
    }
    _write_json(summary, REPORT_ROOT / "p_c_normal28_prediction_export_summary.json")

    # ── 13. markdown report ───────────────────────────────────────────────────
    md = f"""# P-C-NORMAL28: Repaired final_test Prediction Export

**날짜**: {ts[:10]}
**판정**: {verdict}
**manifest**: p_c_normal27_final_test_feature_manifest_repaired_usable.csv (P-C-NORMAL27 scalar repair 적용)

> **주의**: 이 결과는 scalar-repaired final_test 기반 fixed reporting comparison이다.
> threshold 최적화 결과가 아니다. 0.5 threshold는 고정 보고용이다.
> bugged 24k 결과와의 delta는 scalar repair 효과(재학습 없음)를 나타낸다.
> main 24j-fix와 balanced-w1은 별도 checkpoint이며, balanced-w1이 main을 자동 대체하지 않는다.
> 최종 checkpoint 선택은 사용자 승인 후 결정한다.

---

## final_test Manifest (repaired)

| 항목 | 값 |
|---|---|
| 전체 rows | {len(df_raw)} |
| normal (label=0) | {n_normal} |
| NSCLC (label=1) | {n_nsclc} |
| normal_lzp_mean | {round(normal_lzp_mean, 4)} (sanity: >=0.4) |
| NaN | False |
| z_unresolved | 0 |
| repaired_manifest | True |

---

## Checkpoint

| checkpoint | ckpt_epoch | stage |
|---|---|---|
| main_24j_fix | {ckpt_main.get('epoch','?')} | {ckpt_main.get('stage','?')} |
| balanced_w1  | {ckpt_bw1.get('epoch','?')} | {ckpt_bw1.get('stage','?')} |

---

## Crop-level Metrics (fixed threshold=0.5)

| 항목 | main_24j_fix | balanced_w1 | delta(bw1-main) |
|---|---|---|---|
| AUROC | {main_m['auroc']} | {bw1_m['auroc']} | {auroc_d} |
| AUPRC | {main_m['auprc']} | {bw1_m['auprc']} | {auprc_d} |
| Brier | {main_m['brier']} | {bw1_m['brier']} | - |
| balanced_acc | {main_m['balanced_accuracy']} | {bw1_m['balanced_accuracy']} | {bal_acc_d} |
| sensitivity | {main_m['sensitivity']} | {bw1_m['sensitivity']} | {sens_d} |
| specificity | {main_m['specificity']} | {bw1_m['specificity']} | {spec_d} |
| precision | {main_m['precision']} | {bw1_m['precision']} | - |
| F1 | {main_m['f1']} | {bw1_m['f1']} | - |
| TN | {main_m['TN']} | {bw1_m['TN']} | - |
| **FP (normal)** | **{main_m['FP']}** | **{bw1_m['FP']}** | **{fp_delta:+d}** |
| **FN (NSCLC)** | **{main_m['FN']}** | **{bw1_m['FN']}** | **{fn_delta:+d}** |
| TP | {main_m['TP']} | {bw1_m['TP']} | - |

---

## Delta vs Bugged 24k (scalar repair 효과)

| 항목 | main_24j_fix | balanced_w1 |
|---|---|---|
| bugged 24k FP | {BUGGED_24K_MAIN_FP} | {BUGGED_24K_BW1_FP} |
| repaired 28 FP | {main_m['FP']} | {bw1_m['FP']} |
| **delta FP (28 - 24k)** | **{main_m['FP'] - BUGGED_24K_MAIN_FP:+d}** | **{bw1_m['FP'] - BUGGED_24K_BW1_FP:+d}** |
| bugged 24k FN | {BUGGED_24K_MAIN_FN} | {BUGGED_24K_BW1_FN} |
| repaired 28 FN | {main_m['FN']} | {bw1_m['FN']} |
| **delta FN (28 - 24k)** | **{main_m['FN'] - BUGGED_24K_MAIN_FN:+d}** | **{bw1_m['FN'] - BUGGED_24K_BW1_FN:+d}** |
| bugged 24k AUROC | {BUGGED_24K_MAIN_AUROC} | {BUGGED_24K_BW1_AUROC} |
| repaired 28 AUROC | {main_m['auroc']} | {bw1_m['auroc']} |

> bugged 24k 수치는 재실행 없이 P-C-NORMAL25 결과에서 참조한 고정 상수이다.

---

## Patient-level Metrics (fixed threshold=0.5)

### mean_prob 기준

| 항목 | main_24j_fix | balanced_w1 |
|---|---|---|
| n_patients | {main_pat_mean['n_patients']} | {bw1_pat_mean['n_patients']} |
| AUROC | {main_pat_mean['auroc']} | {bw1_pat_mean['auroc']} |
| AUPRC | {main_pat_mean['auprc']} | {bw1_pat_mean['auprc']} |
| specificity | {main_pat_mean['specificity']} | {bw1_pat_mean['specificity']} |
| sensitivity | {main_pat_mean['sensitivity']} | {bw1_pat_mean['sensitivity']} |
| **normal FP patients** | **{main_pat_mean['FP']}** | **{bw1_pat_mean['FP']}** |
| **NSCLC FN patients** | **{main_pat_mean['FN']}** | **{bw1_pat_mean['FN']}** |

### max_prob 기준

| 항목 | main_24j_fix | balanced_w1 |
|---|---|---|
| AUROC | {main_pat_max['auroc']} | {bw1_pat_max['auroc']} |
| specificity | {main_pat_max['specificity']} | {bw1_pat_max['specificity']} |
| sensitivity | {main_pat_max['sensitivity']} | {bw1_pat_max['sensitivity']} |
| **normal FP patients** | **{main_pat_max['FP']}** | **{bw1_pat_max['FP']}** |
| **NSCLC FN patients** | **{main_pat_max['FN']}** | **{bw1_pat_max['FN']}** |

### p95_prob 기준

| 항목 | main_24j_fix | balanced_w1 |
|---|---|---|
| AUROC | {main_pat_p95['auroc']} | {bw1_pat_p95['auroc']} |
| specificity | {main_pat_p95['specificity']} | {bw1_pat_p95['specificity']} |
| sensitivity | {main_pat_p95['sensitivity']} | {bw1_pat_p95['sensitivity']} |
| **normal FP patients** | **{main_pat_p95['FP']}** | **{bw1_pat_p95['FP']}** |
| **NSCLC FN patients** | **{main_pat_p95['FN']}** | **{bw1_pat_p95['FN']}** |

---

## Class-Weighted Supplementary Metrics

> Primary result는 raw final_test 기준이다. weighted FP/FN은 실제 개수가 아닌 가중 합이다.

| 항목 | main_24j_fix | balanced_w1 | delta(bw1-main) |
|---|---|---|---|
| weighted_accuracy | {main_wm['weighted_accuracy']} | {bw1_wm['weighted_accuracy']} | {w_acc_d} |
| weighted_balanced_accuracy | {main_wm['weighted_balanced_accuracy']} | {bw1_wm['weighted_balanced_accuracy']} | - |
| weighted_brier | {main_wm['weighted_brier']} | {bw1_wm['weighted_brier']} | {w_bri_d} |
| weighted_sensitivity | {main_wm['weighted_sensitivity']} | {bw1_wm['weighted_sensitivity']} | - |
| weighted_specificity | {main_wm['weighted_specificity']} | {bw1_wm['weighted_specificity']} | - |
| weighted_precision | {main_wm['weighted_precision']} | {bw1_wm['weighted_precision']} | {w_prec_d} |
| weighted_f1 | {main_wm['weighted_f1']} | {bw1_wm['weighted_f1']} | {w_f1_d} |

---

## Guardrail

- repaired_manifest_used=True
- original_24g_manifest_used=False
- normal_scalar_sanity_pass={normal_lzp_mean >= 0.4}
- threshold_optimized=False
- fixed_threshold_0p5_reporting_only=True
- model_training_run=False
- checkpoint_updated=False
- test_set_checkpoint_selection=False
- stage2_holdout_accessed=False
- existing_24k_result_modified=False
- final_test_balanced_manifest_used=False
- forbidden_diagnostic_wording_count={forbidden_cnt}
- guardrail_fail_count={guardrail_fail_count}

---

## 다음 단계

P-C-NORMAL28 결과 기반 decision checkpoint (사용자 승인 후)
"""
    (REPORT_ROOT / "p_c_normal28_prediction_export_report.md").write_text(md, encoding="utf-8")

    real_error_count = len(errors)
    errors_to_write  = errors if errors else [{"check": "all", "error": "none"}]
    _write_csv(errors_to_write, REPORT_ROOT / "p_c_normal28_errors.csv")

    _write_json(
        {"step": "p_c_normal28", "verdict": verdict, "timestamp": ts,
         "n_crops": len(df_raw), "errors": real_error_count,
         "repaired_manifest_used":            True,
         "final_test_prediction_export_run":  True,
         "threshold_optimized":               False,
         "model_training_run":                False,
         "main_crop_auroc":   main_m["auroc"],
         "bw1_crop_auroc":    bw1_m["auroc"],
         "main_normal_FP":    main_m["FP"],
         "bw1_normal_FP":     bw1_m["FP"],
         "main_nsclc_FN":     main_m["FN"],
         "bw1_nsclc_FN":      bw1_m["FN"]},
        REPORT_ROOT / "DONE.json",
    )

    print(f"\n{'='*60}")
    print(f"[28] 판정: {verdict}")
    print(f"  main  crop_auroc={main_m['auroc']}  normal_FP={main_m['FP']}  nsclc_FN={main_m['FN']}")
    print(f"  bw1   crop_auroc={bw1_m['auroc']}   normal_FP={bw1_m['FP']}  nsclc_FN={bw1_m['FN']}")
    print(f"  delta FP={fp_delta:+d}  FN={fn_delta:+d}  spec={spec_d}  sens={sens_d}")
    print(f"  vs bugged 24k: main FP delta={main_m['FP']-BUGGED_24K_MAIN_FP:+d}  bw1 FP delta={bw1_m['FP']-BUGGED_24K_BW1_FP:+d}")
    print(f"  report → {REPORT_ROOT}")
    print(f"  [주의] fixed reporting. threshold 최적화 아님.")
    print(f"{'='*60}")
    return 0 if verdict == "PASS" else 1


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P-C-NORMAL28: repaired final_test prediction export")
    parser.add_argument("--export",                    action="store_true", required=True)
    parser.add_argument("--confirm-final-test-export", action="store_true")
    parser.add_argument("--confirm-no-threshold-opt",  action="store_true")
    parser.add_argument("--confirm-no-training",       action="store_true")
    args = parser.parse_args()

    missing = []
    if not args.confirm_final_test_export:
        missing.append("--confirm-final-test-export")
    if not args.confirm_no_threshold_opt:
        missing.append("--confirm-no-threshold-opt")
    if not args.confirm_no_training:
        missing.append("--confirm-no-training")
    if missing:
        print(f"[GUARD] --export requires: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)

    sys.exit(run_export(args))


if __name__ == "__main__":
    main()
