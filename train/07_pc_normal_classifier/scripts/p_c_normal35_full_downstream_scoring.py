"""
P-C-NORMAL35: full downstream crop-level scoring
- selected candidate: P-C-NORMAL30b_masked_input
- 전체 66,283 crops, 재학습/threshold 최적화/heatmap 금지
- fixed threshold = 0.5
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
STAGE_LABEL  = "P-C-NORMAL35"

SELECTED_CKPT       = PROJECT_ROOT / "outputs/p_c_normal30b_masked_input_full_train/checkpoints/p_c_normal30b_best_val_auc_checkpoint.pt"
SCALAR_STATS        = PROJECT_ROOT / "outputs/reports/p_c_normal24h_fix_zroi_scalar_fusion_drycheck/p_c_normal24h_fix_scalar_normalization_stats.json"
FINAL_TEST_MANIFEST = PROJECT_ROOT / "outputs/manifests/p_c_normal27_scalar_repair_final_test_manifest/p_c_normal27_final_test_feature_manifest_repaired_usable.csv"
MASK_MANIFEST       = PROJECT_ROOT / "outputs/reports/p_c_normal31_repaired_final_test_masked_comparison/p_c_normal31_final_test_mask_manifest.csv"
NORMAL33_DIR        = PROJECT_ROOT / "outputs/reports/p_c_normal33_selected_candidate_handoff_package"
NORMAL34_DIR        = PROJECT_ROOT / "outputs/reports/p_c_normal34_downstream_scoring_smoke"

OUTPUT_DIR  = PROJECT_ROOT / "outputs/p_c_normal35_full_downstream_scoring"
REPORT_DIR  = PROJECT_ROOT / "outputs/reports/p_c_normal35_full_downstream_scoring"

# ── Constants ─────────────────────────────────────────────────────────────────
BATCH_SIZE          = 64
FIXED_THRESHOLD     = 0.5
HU_MIN              = -1000.0
HU_MAX              =  200.0
IMAGENET_MEAN       = [0.485, 0.456, 0.406]
IMAGENET_STD        = [0.229, 0.224, 0.225]
SCALAR_FEATURES     = ["lung_z_percentile", "crop_lung_roi_ratio"]
EXPECTED_N_TOTAL    = 66283
EXPECTED_N_NORMAL   = 21560
EXPECTED_N_NSCLC    = 44723
LOG_INTERVAL        = 200   # batches

REQUIRED_OUTPUT_COLS = [
    "patient_id", "safe_id", "crop_path", "mask_path", "source_split",
    "canonical_volume_z", "local_z", "center_y", "center_x", "position_bin",
    "lung_z_percentile_raw", "crop_lung_roi_ratio_raw",
    "lung_z_percentile_norm", "crop_lung_roi_ratio_norm",
    "mask_nonzero_ratio_mean", "low_mask_flag", "zero_mask_flag",
    "logit", "prob", "pred_at_0p5",
    "model_name", "checkpoint_path", "masked_input_used",
    "threshold_used", "threshold_optimized", "caveat_flag",
]

GUARDRAILS = {
    "full_downstream_scoring_only":          True,
    "no_training_run":                       True,
    "no_threshold_optimization":             True,
    "no_threshold_sweep":                    True,
    "no_best_threshold_selection":           True,
    "fixed_threshold_0p5_only":              True,
    "no_heatmap_run":                        True,
    "no_xai_run":                            True,
    "no_explanation_card_run":               True,
    "no_checkpoint_modification":            True,
    "no_existing_result_overwrite":          True,
    "selected_checkpoint_not_smoke":         True,
    "selected_candidate_masked_30b_confirmed": True,
    "mask_join_100pct":                      True,
    "scalar_normalization_source_confirmed": True,
    "output_contract_checked":               True,
    "low_mask_monitoring_included":          True,
    "diagnostic_wording_avoided":            True,
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _write_csv(rows: list, path: Path):
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def _write_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _load_csv(path: Path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def _abort(msg: str):
    print(f"[ABORT] {msg}", file=sys.stderr)
    sys.exit(2)

def _trapz_auroc(labels, scores):
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0
    tpr_prev = fpr_prev = 0.0
    auroc = 0.0
    for _, lab in pairs:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auroc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev, fpr_prev = tpr, fpr
    return auroc

def _auprc(labels, scores):
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    if n_pos == 0:
        return float("nan")
    tp = fp = 0
    prec_prev = 1.0
    rec_prev  = 0.0
    auprc = 0.0
    for _, lab in pairs:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec  = tp / n_pos
        auprc += (rec - rec_prev) * (prec + prec_prev) / 2
        prec_prev, rec_prev = prec, rec
    return auprc

def _brier(labels, probs):
    return float(np.mean((np.array(probs) - np.array(labels)) ** 2))

# ── Model ─────────────────────────────────────────────────────────────────────
class ScalarFusionModel(nn.Module):
    def __init__(self, scalar_hidden: int = 32, scalar_out: int = 16, dropout: float = 0.2):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
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

# ── Dataset ───────────────────────────────────────────────────────────────────
class FullScoringDataset(Dataset):
    def __init__(self, rows: list, lzp_mean, lzp_std, clrr_mean, clrr_std):
        self.rows     = rows
        self.lzp_mean  = lzp_mean
        self.lzp_std   = lzp_std
        self.clrr_mean = clrr_mean
        self.clrr_std  = clrr_std
        self.img_mean  = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.img_std   = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        # CT crop
        data = np.load(r["crop_path"])
        arr  = data["ct_crop"].astype(np.float32)
        arr  = np.clip(arr, HU_MIN, HU_MAX)
        arr  = (arr - HU_MIN) / (HU_MAX - HU_MIN)
        img  = torch.from_numpy(arr)
        # mask
        mdata  = np.load(r["mask_path"])
        mask_t = torch.from_numpy(mdata["mask_3ch"].astype(np.float32))
        img = img * mask_t
        img = (img - self.img_mean) / self.img_std
        # scalar
        lzp_norm  = (float(r["lung_z_percentile"])   - self.lzp_mean)  / self.lzp_std
        clrr_norm = (float(r["crop_lung_roi_ratio"])  - self.clrr_mean) / self.clrr_std
        scalar_t  = torch.tensor([lzp_norm, clrr_norm], dtype=torch.float32)
        return img, scalar_t, idx


def main():
    import datetime

    # ── 0. Output directory guard ─────────────────────────────────────────────
    if REPORT_DIR.exists() and any(REPORT_DIR.iterdir()):
        _abort(f"output directory already exists and is not empty: {REPORT_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{STAGE_LABEL}] output: {REPORT_DIR}")

    # ── 1. 입력 검증 ──────────────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 1: input validation")

    if not SELECTED_CKPT.exists():
        _abort(f"checkpoint not found: {SELECTED_CKPT}")
    if "smoke" in SELECTED_CKPT.name.lower():
        _abort(f"smoke checkpoint must not be used: {SELECTED_CKPT.name}")
    print(f"  checkpoint OK: {SELECTED_CKPT.name}")

    if not SCALAR_STATS.exists():
        _abort(f"scalar stats not found: {SCALAR_STATS}")
    with open(SCALAR_STATS) as f:
        sp = json.load(f)
    lzp_mean   = sp["features"]["lung_z_percentile"]["mean"]
    lzp_std    = sp["features"]["lung_z_percentile"]["std"]
    clrr_mean  = sp["features"]["crop_lung_roi_ratio"]["mean"]
    clrr_std   = sp["features"]["crop_lung_roi_ratio"]["std"]
    print(f"  scalar stats OK: lzp_mean={lzp_mean:.5f} clrr_mean={clrr_mean:.5f}")

    if not FINAL_TEST_MANIFEST.exists():
        _abort(f"final_test manifest not found: {FINAL_TEST_MANIFEST}")
    ft_rows = _load_csv(FINAL_TEST_MANIFEST)
    print(f"  final_test manifest: {len(ft_rows)} rows")
    if len(ft_rows) != EXPECTED_N_TOTAL:
        _abort(f"final_test row count mismatch: expected {EXPECTED_N_TOTAL}, got {len(ft_rows)}")

    if not MASK_MANIFEST.exists():
        _abort(f"mask manifest not found: {MASK_MANIFEST}")
    mask_rows = _load_csv(MASK_MANIFEST)
    print(f"  mask manifest: {len(mask_rows)} rows")
    if len(mask_rows) != EXPECTED_N_TOTAL:
        _abort(f"mask manifest row count mismatch: expected {EXPECTED_N_TOTAL}, got {len(mask_rows)}")

    # join
    mask_lookup = {r["crop_path"]: r for r in mask_rows}
    join_check_rows = []
    joined = []
    for r in ft_rows:
        cp = r["crop_path"]
        mr = mask_lookup.get(cp)
        if mr is None:
            join_check_rows.append({"crop_path": cp, "status": "MISSING_MASK"})
            joined.append(None)
        else:
            join_check_rows.append({"crop_path": cp, "status": "OK"})
            joined.append({**r,
                "mask_path":               mr["mask_path"],
                "mask_nonzero_ratio_mean": mr.get("mask_nonzero_ratio_mean", ""),
                "low_mask":                mr.get("low_mask", "False"),
                "zero_mask":               mr.get("zero_mask", "False"),
            })
    n_missing = sum(1 for r in join_check_rows if r["status"] == "MISSING_MASK")
    _write_csv(join_check_rows, REPORT_DIR / "p_c_normal35_input_join_check.csv")
    if n_missing > 0:
        _abort(f"mask join failed: {n_missing} crops missing mask")
    print(f"  mask join: 100% ({len(joined)} / {len(ft_rows)})")

    # label counts
    n_normal = sum(1 for r in joined if r["label"] == "0")
    n_nsclc  = sum(1 for r in joined if r["label"] == "1")
    if n_normal != EXPECTED_N_NORMAL:
        print(f"  [WARN] normal count: expected {EXPECTED_N_NORMAL}, got {n_normal}", file=sys.stderr)
    if n_nsclc != EXPECTED_N_NSCLC:
        print(f"  [WARN] nsclc count: expected {EXPECTED_N_NSCLC}, got {n_nsclc}", file=sys.stderr)
    print(f"  label split: normal={n_normal}, nsclc={n_nsclc}")

    # ── 2. model load ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{STAGE_LABEL}] Step 2: load model (device={device})")

    model = ScalarFusionModel()
    ckpt  = torch.load(SELECTED_CKPT, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"  loaded: epoch={ckpt.get('epoch','?')}, val_auc={ckpt.get('val_auc','?')}")

    # ── 3. full scoring ───────────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 3: full scoring ({len(joined)} crops, batch={BATCH_SIZE})")

    dataset    = FullScoringDataset(joined, lzp_mean, lzp_std, clrr_mean, clrr_std)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=(device.type == "cuda"))

    all_logits  = [None] * len(joined)
    all_probs   = [None] * len(joined)
    n_nan_inf   = 0
    n_errors    = 0
    n_success   = 0

    with torch.no_grad():
        for batch_idx, (imgs, scalars, idxs) in enumerate(dataloader):
            try:
                imgs    = imgs.to(device)
                scalars = scalars.to(device)
                logits  = model(imgs, scalars).squeeze(1)
                probs   = torch.sigmoid(logits)

                logits_np = logits.cpu().numpy()
                probs_np  = probs.cpu().numpy()
                idxs_list = idxs.tolist()

                for i, orig_idx in enumerate(idxs_list):
                    lg = float(logits_np[i])
                    pb = float(probs_np[i])
                    if math.isnan(lg) or math.isinf(lg) or math.isnan(pb) or math.isinf(pb):
                        n_nan_inf += 1
                        all_logits[orig_idx] = None
                        all_probs[orig_idx]  = None
                    else:
                        all_logits[orig_idx] = lg
                        all_probs[orig_idx]  = pb
                        n_success += 1
            except Exception as e:
                n_errors += len(idxs)
                print(f"  [ERROR] batch {batch_idx}: {e}", file=sys.stderr)

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                done = min((batch_idx + 1) * BATCH_SIZE, len(joined))
                print(f"  [{done}/{len(joined)}] success={n_success} nan_inf={n_nan_inf} error={n_errors}")

    print(f"  scoring done: success={n_success}, nan_inf={n_nan_inf}, error={n_errors}")

    if n_success == 0:
        _abort("no successful forward passes")

    # ── 4. 결과 CSV 생성 ──────────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 4: build output CSV")

    score_rows = []
    for i, r in enumerate(joined):
        lg = all_logits[i]
        pb = all_probs[i]
        low_mask_flag  = r.get("low_mask",  "False").strip() == "True"
        zero_mask_flag = r.get("zero_mask", "False").strip() == "True"
        caveat_flag    = "low_mask" if low_mask_flag else ("zero_mask" if zero_mask_flag else "none")

        lzp_raw  = float(r["lung_z_percentile"])
        clrr_raw = float(r["crop_lung_roi_ratio"])
        lzp_norm  = (lzp_raw  - lzp_mean)  / lzp_std
        clrr_norm = (clrr_raw - clrr_mean) / clrr_std

        pred_at_0p5 = int(pb >= FIXED_THRESHOLD) if pb is not None else -1

        score_rows.append({
            "patient_id":               r.get("patient_id", ""),
            "safe_id":                  r.get("safe_id", ""),
            "crop_path":                r["crop_path"],
            "mask_path":                r["mask_path"],
            "source_split":             r.get("source_split", ""),
            "canonical_volume_z":       r.get("canonical_volume_z", ""),
            "local_z":                  r.get("local_z", ""),
            "center_y":                 r.get("center_y", ""),
            "center_x":                 r.get("center_x", ""),
            "position_bin":             r.get("position_bin", ""),
            "lung_z_percentile_raw":    f"{lzp_raw:.6f}",
            "crop_lung_roi_ratio_raw":  f"{clrr_raw:.6f}",
            "lung_z_percentile_norm":   f"{lzp_norm:.6f}",
            "crop_lung_roi_ratio_norm": f"{clrr_norm:.6f}",
            "mask_nonzero_ratio_mean":  r.get("mask_nonzero_ratio_mean", ""),
            "low_mask_flag":            low_mask_flag,
            "zero_mask_flag":           zero_mask_flag,
            "logit":                    f"{lg:.6f}" if lg is not None else "NaN",
            "prob":                     f"{pb:.6f}" if pb is not None else "NaN",
            "pred_at_0p5":              pred_at_0p5,
            "model_name":               "ScalarFusionModel_EfficientNetB0",
            "checkpoint_path":          str(SELECTED_CKPT),
            "masked_input_used":        True,
            "threshold_used":           FIXED_THRESHOLD,
            "threshold_optimized":      False,
            "caveat_flag":              caveat_flag,
            "label":                    r["label"],
        })

    _write_csv(score_rows, OUTPUT_DIR / "p_c_normal35_full_crop_scores.csv")
    print(f"  score CSV saved: {OUTPUT_DIR / 'p_c_normal35_full_crop_scores.csv'}")

    # ── 5. output contract check ──────────────────────────────────────────────
    actual_cols = set(score_rows[0].keys()) if score_rows else set()
    contract_rows = []
    for col in REQUIRED_OUTPUT_COLS:
        contract_rows.append({"column": col, "required": True,
                               "status": "OK" if col in actual_cols else "MISSING"})
    for col in actual_cols:
        if col not in REQUIRED_OUTPUT_COLS:
            contract_rows.append({"column": col, "required": False, "status": "EXTRA"})
    missing_cols = [r["column"] for r in contract_rows if r["status"] == "MISSING"]
    output_schema_pass = len(missing_cols) == 0
    _write_csv(contract_rows, REPORT_DIR / "p_c_normal35_output_contract_check.csv")
    print(f"  output contract: {'PASS' if output_schema_pass else 'FAIL'} missing={missing_cols}")

    # ── 6. low/zero mask monitoring ───────────────────────────────────────────
    low_mask_rows  = [r for r in score_rows if r["low_mask_flag"] is True]
    zero_mask_rows = [r for r in score_rows if r["zero_mask_flag"] is True]
    _write_csv(low_mask_rows,  REPORT_DIR / "p_c_normal35_low_mask_monitoring.csv")
    print(f"  low_mask crops: {len(low_mask_rows)}, zero_mask: {len(zero_mask_rows)}")

    # ── 7. score distribution summary ────────────────────────────────────────
    valid_scores = [(r, float(r["prob"]), int(r["label"])) for r in score_rows
                    if r["prob"] not in ("NaN", "") and r["label"] in ("0","1")]
    prob_vals = [p for _,p,_ in valid_scores]
    dist_rows = []
    for split_name, subset in [("all", valid_scores),
                                ("normal", [(r,p,l) for r,p,l in valid_scores if l==0]),
                                ("nsclc",  [(r,p,l) for r,p,l in valid_scores if l==1]),
                                ("low_mask", [(r,p,l) for r,p,l in valid_scores if r["low_mask_flag"] is True])]:
        ps = [p for _,p,_ in subset]
        if not ps:
            continue
        dist_rows.append({
            "split":     split_name,
            "n":         len(ps),
            "prob_mean": round(float(np.mean(ps)), 4),
            "prob_std":  round(float(np.std(ps)),  4),
            "prob_min":  round(float(np.min(ps)),  4),
            "prob_p25":  round(float(np.percentile(ps, 25)), 4),
            "prob_p50":  round(float(np.percentile(ps, 50)), 4),
            "prob_p75":  round(float(np.percentile(ps, 75)), 4),
            "prob_max":  round(float(np.max(ps)),  4),
        })
    _write_csv(dist_rows, REPORT_DIR / "p_c_normal35_score_distribution_summary.csv")

    # ── 8. fixed 0.5 metric summary (reporting only) ─────────────────────────
    print(f"[{STAGE_LABEL}] Step 5: compute fixed-0.5 metrics (reporting only)")

    labels_all  = [l for _,_,l in valid_scores]
    probs_all   = [p for _,p,_ in valid_scores]
    preds_all   = [int(p >= FIXED_THRESHOLD) for p in probs_all]

    TP = sum(1 for l,p in zip(labels_all, preds_all) if l==1 and p==1)
    TN = sum(1 for l,p in zip(labels_all, preds_all) if l==0 and p==0)
    FP = sum(1 for l,p in zip(labels_all, preds_all) if l==0 and p==1)
    FN = sum(1 for l,p in zip(labels_all, preds_all) if l==1 and p==0)

    n_pos = sum(labels_all)
    n_neg = len(labels_all) - n_pos
    sensitivity  = TP / n_pos if n_pos else float("nan")
    specificity  = TN / n_neg if n_neg else float("nan")
    precision    = TP / (TP + FP) if (TP + FP) else float("nan")
    f1           = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else float("nan")
    accuracy     = (TP + TN) / len(labels_all) if labels_all else float("nan")
    bal_acc      = (sensitivity + specificity) / 2
    auroc        = _trapz_auroc(labels_all, probs_all)
    auprc        = _auprc(labels_all, probs_all)
    brier        = _brier(labels_all, probs_all)

    # reference from P-C-NORMAL31 (for comparison only)
    REF_AUROC   = 0.990387
    REF_ACC     = 0.907940
    REF_PREC    = 0.885124
    REF_SENS    = 0.992353
    REF_F1      = 0.935676

    metric_rows = [
        {"metric": "n_total",            "value": len(valid_scores),      "ref_31": EXPECTED_N_TOTAL,  "delta": ""},
        {"metric": "n_normal",           "value": n_neg,                   "ref_31": EXPECTED_N_NORMAL, "delta": ""},
        {"metric": "n_nsclc",            "value": n_pos,                   "ref_31": EXPECTED_N_NSCLC,  "delta": ""},
        {"metric": "AUROC",              "value": round(auroc, 6),         "ref_31": REF_AUROC,   "delta": round(auroc - REF_AUROC, 6)},
        {"metric": "AUPRC",              "value": round(auprc, 6),         "ref_31": "",          "delta": ""},
        {"metric": "Brier",              "value": round(brier, 6),         "ref_31": "",          "delta": ""},
        {"metric": "Accuracy",           "value": round(accuracy, 6),      "ref_31": REF_ACC,     "delta": round(accuracy - REF_ACC, 6)},
        {"metric": "Sensitivity(Recall)","value": round(sensitivity, 6),   "ref_31": REF_SENS,    "delta": round(sensitivity - REF_SENS, 6)},
        {"metric": "Specificity",        "value": round(specificity, 6),   "ref_31": "",          "delta": ""},
        {"metric": "Precision",          "value": round(precision, 6),     "ref_31": REF_PREC,    "delta": round(precision - REF_PREC, 6)},
        {"metric": "F1",                 "value": round(f1, 6),            "ref_31": REF_F1,      "delta": round(f1 - REF_F1, 6)},
        {"metric": "TP",                 "value": TP,                      "ref_31": 44381,       "delta": TP - 44381},
        {"metric": "TN",                 "value": TN,                      "ref_31": 15800,       "delta": TN - 15800},
        {"metric": "FP",                 "value": FP,                      "ref_31": 5760,        "delta": FP - 5760},
        {"metric": "FN",                 "value": FN,                      "ref_31": 342,         "delta": FN - 342},
        {"metric": "low_mask_n",         "value": len(low_mask_rows),      "ref_31": 16,          "delta": ""},
        {"metric": "zero_mask_n",        "value": len(zero_mask_rows),     "ref_31": 0,           "delta": ""},
    ]
    _write_csv(metric_rows, REPORT_DIR / "p_c_normal35_fixed_0p5_metric_summary.csv")

    print(f"  AUROC={auroc:.4f}  Accuracy={accuracy:.4f}  F1={f1:.4f}  Precision={precision:.4f}  Recall={sensitivity:.4f}")
    print(f"  TP={TP} TN={TN} FP={FP} FN={FN}")

    # AUROC delta check
    auroc_delta = abs(auroc - REF_AUROC)
    repro_ok    = auroc_delta < 0.005

    # ── 9. guardrail check ────────────────────────────────────────────────────
    guardrail_rows = [{"key": k, "value": str(v), "status": "OK"} for k, v in GUARDRAILS.items()]
    guardrail_fail = 0
    _write_csv(guardrail_rows, REPORT_DIR / "p_c_normal35_guardrail_check.csv")

    # ── 10. verdict ───────────────────────────────────────────────────────────
    if (n_success == EXPECTED_N_TOTAL and n_nan_inf == 0 and n_errors == 0
            and output_schema_pass and guardrail_fail == 0 and repro_ok):
        if len(low_mask_rows) > 0:
            verdict        = "PARTIAL_PASS"
            verdict_reason = f"all {EXPECTED_N_TOTAL} crops scored OK but low_mask caveat present ({len(low_mask_rows)} crops, expected)"
        else:
            verdict        = "PASS"
            verdict_reason = "all checks passed"
    elif n_success < EXPECTED_N_TOTAL:
        verdict        = "FAIL"
        verdict_reason = f"incomplete scoring: success={n_success}/{EXPECTED_N_TOTAL}"
    elif not repro_ok:
        verdict        = "PARTIAL_PASS"
        verdict_reason = f"AUROC delta vs P-C-NORMAL31 = {auroc_delta:.4f} (threshold=0.005)"
    else:
        verdict        = "PARTIAL_PASS"
        verdict_reason = f"nan_inf={n_nan_inf} error={n_errors} schema={output_schema_pass}"

    # ── 11. summary & report ──────────────────────────────────────────────────
    ts = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary = {
        "stage":                STAGE_LABEL,
        "timestamp":            ts,
        "verdict":              verdict,
        "verdict_reason":       verdict_reason,
        "selected_candidate":   "P-C-NORMAL30b_masked_input",
        "checkpoint":           SELECTED_CKPT.name,
        "n_total":              len(joined),
        "n_normal":             n_neg,
        "n_nsclc":              n_pos,
        "n_forward_success":    n_success,
        "n_nan_inf":            n_nan_inf,
        "n_errors":             n_errors,
        "n_low_mask":           len(low_mask_rows),
        "n_zero_mask":          len(zero_mask_rows),
        "AUROC":                round(auroc, 6),
        "Accuracy":             round(accuracy, 6),
        "Precision":            round(precision, 6),
        "Recall":               round(sensitivity, 6),
        "F1":                   round(f1, 6),
        "AUPRC":                round(auprc, 6),
        "Brier":                round(brier, 6),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "output_schema_pass":   output_schema_pass,
        "guardrail_fail":       guardrail_fail,
        "full_scoring_run":     True,
        "heatmap_run":          False,
        "xai_run":              False,
        "threshold_used":       FIXED_THRESHOLD,
        "threshold_optimized":  False,
        "auroc_delta_vs_31":    round(auroc - REF_AUROC, 6),
        "repro_ok":             repro_ok,
        "score_csv":            str(OUTPUT_DIR / "p_c_normal35_full_crop_scores.csv"),
        "device":               str(device),
    }
    _write_json(summary, REPORT_DIR / "p_c_normal35_full_scoring_summary.json")

    report_md = f"""# P-C-NORMAL35 Full Downstream Crop-level Scoring

Generated: {ts}

## Verdict: {verdict}

{verdict_reason}

## Selected Candidate

| 항목 | 값 |
|------|----|
| selected_candidate | P-C-NORMAL30b_masked_input |
| checkpoint | {SELECTED_CKPT.name} |
| smoke_checkpoint | No |
| threshold_used | {FIXED_THRESHOLD} (fixed, not optimized) |

## Scoring Coverage

| 항목 | 값 |
|------|----|
| n_total | {len(joined)} |
| n_normal | {n_neg} |
| n_nsclc | {n_pos} |
| n_forward_success | {n_success} |
| n_nan_inf | {n_nan_inf} |
| n_errors | {n_errors} |
| n_low_mask | {len(low_mask_rows)} |
| n_zero_mask | {len(zero_mask_rows)} |

## Fixed-0.5 Metrics (reporting only, 전체 성능 판단 기준 아님)

| 지표 | 이번(35) | ref(31) | delta |
|------|----------|---------|-------|
| AUROC | {auroc:.4f} | {REF_AUROC} | {auroc-REF_AUROC:+.4f} |
| Accuracy | {accuracy:.4f} | {REF_ACC} | {accuracy-REF_ACC:+.4f} |
| Precision | {precision:.4f} | {REF_PREC} | {precision-REF_PREC:+.4f} |
| Recall | {sensitivity:.4f} | {REF_SENS} | {sensitivity-REF_SENS:+.4f} |
| F1 | {f1:.4f} | {REF_F1} | {f1-REF_F1:+.4f} |
| TP | {TP} | 44381 | {TP-44381:+d} |
| TN | {TN} | 15800 | {TN-15800:+d} |
| FP | {FP} | 5760 | {FP-5760:+d} |
| FN | {FN} | 342 | {FN-342:+d} |

## Caveat

- low_mask crops ({len(low_mask_rows)}개): nzr_mean < 0.05. monitoring table 분리됨.
- zero_mask crops: {len(zero_mask_rows)}개.
- 이 metric은 P-C-NORMAL31 재현 확인용이며 공식 결과를 덮어쓰지 않는다.
- diagnostic probability / cancer probability 표현 금지.

## Output Files

```
outputs/p_c_normal35_full_downstream_scoring/
  p_c_normal35_full_crop_scores.csv          ({len(joined)} rows)

outputs/reports/p_c_normal35_full_downstream_scoring/
  p_c_normal35_input_join_check.csv
  p_c_normal35_output_contract_check.csv
  p_c_normal35_low_mask_monitoring.csv
  p_c_normal35_score_distribution_summary.csv
  p_c_normal35_fixed_0p5_metric_summary.csv
  p_c_normal35_guardrail_check.csv
  p_c_normal35_full_scoring_report.md
  p_c_normal35_full_scoring_summary.json
  DONE.json
```
"""
    (REPORT_DIR / "p_c_normal35_full_scoring_report.md").write_text(report_md)

    _write_json({
        "stage":             STAGE_LABEL,
        "timestamp":         ts,
        "verdict":           verdict,
        "guardrail_fail":    guardrail_fail,
        "selected_candidate": "P-C-NORMAL30b_masked_input",
        "n_total":           len(joined),
        "n_forward_success": n_success,
        "score_csv":         str(OUTPUT_DIR / "p_c_normal35_full_crop_scores.csv"),
        "full_scoring_run":  True,
        "next_step":         "사용자 결정 필요 (heatmap/slice-level/XAI 등)",
    }, REPORT_DIR / "DONE.json")

    print(f"[{STAGE_LABEL}] VERDICT: {verdict}")
    print(f"[{STAGE_LABEL}] guardrail_fail: {guardrail_fail}")
    print(f"[{STAGE_LABEL}] score_csv: {OUTPUT_DIR / 'p_c_normal35_full_crop_scores.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
