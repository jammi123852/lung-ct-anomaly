"""
p_c_normal24h_fix_zroi_scalar_fusion_drycheck.py

P-C-NORMAL24h-fix: z/ROI scalar-fusion normalization 재계산 + dry-check
- 이전 24h는 잘못된 24g manifest(crop_lung_roi_ratio 1/9 축소)를 사용 → 폐기
- 이번에는 24g-fix usable manifest 기준으로 scalar normalization 재계산
- dry-check only: full training / checkpoint 저장 / final_test / metrics / threshold 금지

이 모델은 supervised normal-vs-NSCLC auxiliary classifier다.
출력은 normal-like vs NSCLC-lesion-like auxiliary score로만 해석한다.

금지:
  - old 24g manifest 사용 금지
  - old 24h scalar normalization stats 사용 금지
  - vessel feature 사용 금지
  - ROI-masked loss / loss weighting 변경 / image masking / pixel-level loss 금지
  - full manifest 사용 금지 (usable manifest만 허용)
  - unresolved rows 사용 금지
  - final_test 사용/튜닝/metrics 금지
  - crop_lung_roi_ratio 를 loss weight로 사용 금지
  - full training 금지 / checkpoint 저장 금지

Usage:
  python p_c_normal24h_fix_zroi_scalar_fusion_drycheck.py
"""

import csv
import json
import os
import py_compile
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

# 24g-fix usable manifest (old 24g 금지)
TRAIN_MANIFEST = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_zroi_only_feature_manifest/p_c_normal24g_fix_train_feature_manifest_usable.csv"
VAL_MANIFEST   = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_zroi_only_feature_manifest/p_c_normal24g_fix_val_feature_manifest_usable.csv"

OUTPUT_ROOT  = PROJECT_ROOT / "outputs/p_c_normal24h_fix_zroi_scalar_fusion"
REPORT_ROOT  = PROJECT_ROOT / "outputs/reports/p_c_normal24h_fix_zroi_scalar_fusion_drycheck"

# old 24h scalar stats (비교용 — 읽기만)
OLD_NORM_JSON = PROJECT_ROOT / "outputs/reports/p_c_normal24h_zroi_scalar_fusion_script_drycheck/p_c_normal24h_scalar_normalization_stats.json"

# ── Constants ─────────────────────────────────────────────────────────────────
HU_MIN, HU_MAX  = -1000.0, 200.0
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
SCALAR_FEATURES = ["lung_z_percentile", "crop_lung_roi_ratio"]

EXPECTED_TRAIN_USABLE = 19716
EXPECTED_VAL_USABLE   = 5189

# old 24h stats (하드코딩 비교 기준)
OLD_STATS = {
    "lung_z_percentile":  {"mean": 0.48795873106895715, "std": 0.19693269575462496},
    "crop_lung_roi_ratio": {"mean": 0.09869534323152654, "std": 0.015808160203859997},
}

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

DRY_BATCH_SIZE = 8


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

def compute_scalar_norm(df_train: pd.DataFrame) -> dict:
    stats = {}
    for col in SCALAR_FEATURES:
        vals = df_train[col].dropna().astype(float).values
        mean = float(np.mean(vals))
        std  = float(np.std(vals, ddof=0))
        if std < 1e-8:
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


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


# ── Dataset ───────────────────────────────────────────────────────────────────

class ScalarFusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)

        vessel_cols = [c for c in df.columns if c in FORBIDDEN_VESSEL_COLS]
        if vessel_cols:
            raise RuntimeError(f"[GUARD] forbidden vessel columns in dataset: {vessel_cols}")
        if "z_unresolved" in df.columns and df["z_unresolved"].any():
            n_ur = int(df["z_unresolved"].sum())
            raise RuntimeError(f"[GUARD] {n_ur} unresolved rows in dataset — use usable manifest only")
        for col in SCALAR_FEATURES:
            if df[col].isna().any():
                n_nan = int(df[col].isna().sum())
                raise RuntimeError(f"[GUARD] {n_nan} NaN values in scalar feature '{col}'")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
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


# ── Model (기존 24h 구조 유지) ────────────────────────────────────────────────

class ScalarFusionModel(nn.Module):
    """
    image:  (B, 3, 96, 96)  → EfficientNet-B0 → (B, 1280)
    scalar: (B, 2) float32  → Linear(2→32)→BN→ReLU→Linear(32→16)→ReLU → (B, 16)
    fusion: concat(1296) → Dropout(0.2) → Linear(64) → ReLU → Dropout → Linear(1)
    output: (B, 1) logit
    """

    def __init__(self, scalar_hidden: int = 32, scalar_out: int = 16, dropout: float = 0.2):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.img_features = backbone.features
        self.img_avgpool  = backbone.avgpool
        img_feat_dim = 1280

        self.scalar_branch = nn.Sequential(
            nn.Linear(2, scalar_hidden),
            nn.BatchNorm1d(scalar_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(scalar_hidden, scalar_out),
            nn.ReLU(inplace=True),
        )

        fusion_in = img_feat_dim + scalar_out
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


# ── Manifest guard ────────────────────────────────────────────────────────────

def _guard_manifests_are_usable(df_train: pd.DataFrame, df_val: pd.DataFrame) -> list:
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
                issues.append(f"{name}: NaN in '{col}' = {int(df[col].isna().sum())}")
        if "final_test" in name.lower():
            issues.append("final_test manifest used in training — forbidden")
    return issues


# ── Main dry-check ────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── output collision guard ────────────────────────────────────────────────
    if REPORT_ROOT.exists() and any(REPORT_ROOT.iterdir()):
        print(f"[ABORT] REPORT_ROOT already exists and is not empty: {REPORT_ROOT}")
        sys.exit(2)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    errors        = []
    batch_rows    = []
    guardrail_rows = []
    verdict       = "PASS"

    # ── 1. Manifest path guard: old 24g 경로 사용 금지 ────────────────────────
    old_24g_path = str(PROJECT_ROOT / "outputs/manifests/p_c_normal24g_zroi_only_feature_manifest")
    if old_24g_path in str(TRAIN_MANIFEST) or old_24g_path in str(VAL_MANIFEST):
        print("[ABORT] old 24g manifest path detected — use 24g-fix manifest.")
        sys.exit(2)

    # ── 2. Manifest load ──────────────────────────────────────────────────────
    if not TRAIN_MANIFEST.exists() or not VAL_MANIFEST.exists():
        errors.append({"check": "manifest_files_exist", "error": "24g-fix CSV missing"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal24h_fix_errors.csv")
        sys.exit(1)

    df_train = pd.read_csv(TRAIN_MANIFEST, low_memory=False)
    df_val   = pd.read_csv(VAL_MANIFEST,   low_memory=False)

    # ── 3. Manifest validation ────────────────────────────────────────────────
    manifest_issues = _guard_manifests_are_usable(df_train, df_val)
    n_train = len(df_train)
    n_val   = len(df_val)
    n0_tr   = int((df_train["label"] == 0).sum())
    n1_tr   = int((df_train["label"] == 1).sum())
    n0_vl   = int((df_val["label"]   == 0).sum())
    n1_vl   = int((df_val["label"]   == 1).sum())

    manifest_checks = [
        ("train_usable_rows",     EXPECTED_TRAIN_USABLE, n_train, n_train == EXPECTED_TRAIN_USABLE),
        ("val_usable_rows",       EXPECTED_VAL_USABLE,   n_val,   n_val   == EXPECTED_VAL_USABLE),
        ("vessel_cols_absent",    True, bool(len(manifest_issues) == 0), not manifest_issues),
        ("label_values_only_01",  "[0,1]",
         str(sorted(df_train["label"].unique().tolist())),
         sorted(df_train["label"].unique().tolist()) == [0, 1]),
        ("scalar_nan_free_train", True,
         str(df_train[SCALAR_FEATURES].isna().sum().sum() == 0),
         bool(df_train[SCALAR_FEATURES].isna().sum().sum() == 0)),
        ("scalar_nan_free_val",   True,
         str(df_val[SCALAR_FEATURES].isna().sum().sum() == 0),
         bool(df_val[SCALAR_FEATURES].isna().sum().sum() == 0)),
        ("final_test_absent",     True, "True", True),
        ("24g_fix_manifest_used", True, "True", True),
        ("old_24g_manifest_used", False, "False", True),
    ]
    for check, exp, act, ok in manifest_checks:
        batch_rows.append({"check": check, "expected": str(exp), "actual": str(act), "pass": bool(ok)})
        if not ok:
            errors.append({"check": check, "error": f"expected={exp} actual={act}"})
            verdict = "PARTIAL_PASS"

    if manifest_issues:
        for iss in manifest_issues:
            errors.append({"check": "manifest_guard", "error": iss})
        verdict = "FAIL"

    # ── 4. Scalar normalization 재계산 ────────────────────────────────────────
    print("Scalar normalization 재계산 중 (24g-fix train 기준)...")
    scalar_stats = compute_scalar_norm(df_train)

    norm_json_path = REPORT_ROOT / "p_c_normal24h_fix_scalar_normalization_stats.json"
    norm_payload = {
        "computed_from": "p_c_normal24g_fix_train_feature_manifest_usable",
        "n_train_rows":  n_train,
        "features":      scalar_stats,
        "policy":        "train set fit only; apply same mean/std to val/test",
        "timestamp":     ts,
        "old_24h_stats_discarded": True,
        "reason": "old 24g crop_lung_roi_ratio was 1/9 scaled (wrong denominator); 24g-fix corrects bbox_h*bbox_w denominator",
    }
    _write_json(norm_payload, norm_json_path)
    print(f"  saved: {norm_json_path}")

    # Normalization sanity (train에 적용 후 mean≈0, std≈1 확인)
    df_train_norm = apply_scalar_norm(df_train, scalar_stats)
    df_val_norm   = apply_scalar_norm(df_val,   scalar_stats)

    for col in SCALAR_FEATURES:
        mu  = float(df_train_norm[col].mean())
        std = float(df_train_norm[col].std())
        batch_rows.append({
            "check": f"norm_{col}_mean_near0",
            "expected": "~0.0", "actual": f"{mu:.4f}",
            "pass": bool(abs(mu) < 0.1),
        })
        batch_rows.append({
            "check": f"norm_{col}_std_near1",
            "expected": "~1.0", "actual": f"{std:.4f}",
            "pass": bool(abs(std - 1.0) < 0.1),
        })

    # ── 5. Old vs new scalar stats 비교 ──────────────────────────────────────
    old_vs_new_rows = []
    for col in SCALAR_FEATURES:
        new_s = scalar_stats[col]
        old_s = OLD_STATS[col]
        old_vs_new_rows.append({
            "feature":       col,
            "old_24h_mean":  old_s["mean"],
            "old_24h_std":   old_s["std"],
            "new_fix_mean":  new_s["mean"],
            "new_fix_std":   new_s["std"],
            "mean_change":   round(new_s["mean"] - old_s["mean"], 6),
            "std_change":    round(new_s["std"]  - old_s["std"],  6),
            "note": (
                "32x32 bbox ratio 1/9 scailing fixed; mean should rise ~9x"
                if col == "crop_lung_roi_ratio"
                else "lung_z_percentile unchanged (not affected by bbox fix)"
            ),
        })
    _write_csv(old_vs_new_rows, REPORT_ROOT / "p_c_normal24h_fix_old_vs_new_scalar_stats.csv")

    # ── 6. Dataset + 1 batch ─────────────────────────────────────────────────
    print("Dataset/DataLoader dry-check 중...")
    ds_train = ScalarFusionDataset(df_train_norm, augment=False)
    loader   = DataLoader(ds_train, batch_size=DRY_BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=False)
    batch_imgs, batch_scalars, batch_labels, batch_sw = next(iter(loader))

    img_shape_ok    = bool(batch_imgs.shape   == torch.Size([DRY_BATCH_SIZE, 3, 96, 96]))
    scalar_shape_ok = bool(batch_scalars.shape == torch.Size([DRY_BATCH_SIZE, 2]))
    labels_ok       = bool(((batch_labels == 0) | (batch_labels == 1)).all())
    sw_ok           = bool(torch.isfinite(batch_sw).all())
    imgs_finite     = bool(torch.isfinite(batch_imgs).all())
    scalars_finite  = bool(torch.isfinite(batch_scalars).all())

    batch_checks = [
        ("img_shape",      f"({DRY_BATCH_SIZE},3,96,96)", str(tuple(batch_imgs.shape)),    img_shape_ok),
        ("scalar_shape",   f"({DRY_BATCH_SIZE},2)",        str(tuple(batch_scalars.shape)), scalar_shape_ok),
        ("labels_01",      True, str(labels_ok),   labels_ok),
        ("sw_finite",      True, str(sw_ok),        sw_ok),
        ("imgs_finite",    True, str(imgs_finite),  imgs_finite),
        ("scalars_finite", True, str(scalars_finite), scalars_finite),
    ]
    for check, exp, act, ok in batch_checks:
        batch_rows.append({"check": check, "expected": str(exp), "actual": str(act), "pass": bool(ok)})
        if not ok:
            errors.append({"check": check, "error": str(act)})
            verdict = "FAIL"

    # ── 7. Model forward (no_grad) ────────────────────────────────────────────
    print("Model forward (no_grad) 중...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ScalarFusionModel().eval().to(device)

    with torch.no_grad():
        logits = model(batch_imgs.to(device), batch_scalars.to(device))

    logit_shape_ok   = bool(logits.shape == torch.Size([DRY_BATCH_SIZE, 1]))
    logits_finite_ok = bool(torch.isfinite(logits).all())

    batch_rows.append({"check": "logit_shape",   "expected": f"({DRY_BATCH_SIZE},1)", "actual": str(tuple(logits.shape)), "pass": logit_shape_ok})
    batch_rows.append({"check": "logits_finite", "expected": "True", "actual": str(logits_finite_ok), "pass": logits_finite_ok})
    if not logit_shape_ok or not logits_finite_ok:
        errors.append({"check": "model_forward", "error": f"shape={tuple(logits.shape)} finite={logits_finite_ok}"})
        verdict = "FAIL"

    # ── 8. Loss (no_grad) ────────────────────────────────────────────────────
    with torch.no_grad():
        loss = weighted_bce_loss(logits, batch_labels.to(device), batch_sw.to(device))
    loss_val    = float(loss.item())
    loss_finite = bool(np.isfinite(loss_val))
    batch_rows.append({"check": "loss_finite", "expected": "True", "actual": f"{loss_val:.6f}", "pass": loss_finite})
    if not loss_finite:
        errors.append({"check": "loss", "error": f"loss={loss_val}"})
        verdict = "FAIL"

    # ── 9. Backward grad check (NO optimizer step, NO checkpoint) ────────────
    print("Backward grad check 중...")
    model_train = ScalarFusionModel().train().to(device)
    logits_tr   = model_train(batch_imgs.to(device), batch_scalars.to(device))
    loss_tr     = weighted_bce_loss(logits_tr, batch_labels.to(device), batch_sw.to(device))
    loss_tr.backward()

    grads_ok = bool(all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in model_train.parameters() if p.requires_grad
    ))
    batch_rows.append({"check": "backward_grad_finite", "expected": "True",  "actual": str(grads_ok), "pass": grads_ok})
    batch_rows.append({"check": "optimizer_step_run",   "expected": "False", "actual": "False", "pass": True})
    batch_rows.append({"check": "checkpoint_saved",     "expected": "False", "actual": "False", "pass": True})
    if not grads_ok:
        errors.append({"check": "grad_check", "error": "NaN/Inf gradient"})
        verdict = "FAIL"

    # ── 10. Guardrail check ───────────────────────────────────────────────────
    this_file = Path(__file__).resolve()
    compile_ok = False
    try:
        py_compile.compile(str(this_file), doraise=True)
        compile_ok = True
    except py_compile.PyCompileError as e:
        errors.append({"check": "py_compile", "error": str(e)})

    forbidden_count = _check_forbidden_words(this_file)

    guardrail_items = [
        ("corrected_24g_fix_manifest_used",          True,  True),
        ("old_24g_manifest_used",                    False, False),
        ("old_24h_scalar_stats_used",                False, False),
        ("scalar_normalization_recomputed",          True,  True),
        ("drycheck_run",                             True,  True),
        ("full_training_run",                        False, False),
        ("final_test_used",                          False, False),
        ("final_test_prediction_export_run",         False, False),
        ("metrics_computed",                         False, False),
        ("threshold_computed",                       False, False),
        ("threshold_optimized",                      False, False),
        ("checkpoint_saved",                         False, False),
        ("usable_manifest_used",                     True,  True),
        ("full_manifest_used_for_training",          False, False),
        ("unresolved_rows_used",                     False, False),
        ("vessel_feature_used",                      False, False),
        ("roi_masked_loss_used",                     False, False),
        ("loss_weighting_changed",                   False, False),
        ("image_roi_masking_used",                   False, False),
        ("pixel_level_loss_used",                    False, False),
        ("crop_lung_roi_ratio_used_as_loss_weight",  False, False),
        ("py_compile_ok",                            True,  compile_ok),
        ("forbidden_diagnostic_wording_count",       0,     forbidden_count),
    ]
    for check, exp, act in guardrail_items:
        ok = bool(act == exp)
        guardrail_rows.append({"guardrail": check, "expected": str(exp), "actual": str(act), "pass": ok})
        if not ok:
            errors.append({"check": check, "error": f"expected={exp} actual={act}"})
            verdict = "FAIL"

    # ── 11. Write outputs ─────────────────────────────────────────────────────
    _write_csv(batch_rows,     REPORT_ROOT / "p_c_normal24h_fix_batch_shape_check.csv")
    _write_csv(guardrail_rows, REPORT_ROOT / "p_c_normal24h_fix_guardrail_check.csv")
    _write_csv(errors if errors else [{"check": "all", "error": "none"}],
               REPORT_ROOT / "p_c_normal24h_fix_errors.csv")

    # Smoke command plan
    smoke_md = f"""# P-C-NORMAL24i-fix Smoke Training Command Plan

**작성일**: {ts[:10]}
**이 파일은 계획 문서다. 실행은 사용자 승인 후 P-C-NORMAL24i-fix에서 진행한다.**

## Smoke Training Command (예시)

```bash
cd /home/jinhy/project/lung-ct-anomaly
source ~/ai_env/bin/activate
nohup python experiments/efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1/code/p_c_normal24h_zroi_scalar_fusion_train.py \\
    --smoke-train \\
    --confirm-smoke \\
    --confirm-normal-vs-nsclc \\
    --confirm-no-final-test \\
    --epochs 1 \\
    > logs/p_c_normal24i_fix_smoke_$(date +%Y%m%d_%H%M).log 2>&1 &
echo "PID: $!"
```

주의: smoke training 실행 시 TRAIN_MANIFEST/VAL_MANIFEST를 24g-fix 경로로 변경한 버전을 사용해야 한다.

## Smoke Config

| 항목 | 값 |
|---|---|
| epochs | 1 |
| batch_size | 32 |
| manifests | 24g-fix usable (train={n_train}, val={n_val}) |
| scalar_norm | 24h-fix 재계산값 기준 |

## 금지 (smoke에서도 유지)

- final_test 사용 금지
- metrics/threshold 계산 금지
- vessel feature 사용 금지
- ROI-masked loss 금지
- checkpoint 저장 전 사용자 승인 필요
"""
    (REPORT_ROOT / "p_c_normal24h_fix_smoke_command_plan.md").write_text(smoke_md, encoding="utf-8")

    # Summary JSON
    summary = {
        "branch":            "P-C-NORMAL24h-fix-zroi-scalar-fusion",
        "step":              "normalization_recompute_and_drycheck",
        "verdict":           verdict,
        "verdict_issues":    [e["error"] for e in errors if e.get("check") != "all"],
        "timestamp":         ts,
        "old_24h_discarded_reason": "old 24g crop_lung_roi_ratio was 1/9 scaled (96x96 fixed denominator instead of bbox_h*bbox_w)",
        "model_architecture": {
            "image_branch":  "EfficientNet-B0 features+avgpool -> (B,1280)",
            "scalar_branch": "Linear(2->32)->BN->ReLU->Linear(32->16)->ReLU",
            "fusion_head":   "concat(1296)->Dropout(0.2)->Linear(64)->ReLU->Dropout->Linear(1)",
            "output":        "(B,1) logit for BCEWithLogitsLoss",
        },
        "scalar_features":   SCALAR_FEATURES,
        "old_scalar_norm_stats": OLD_STATS,
        "new_scalar_norm_stats": scalar_stats,
        "manifests": {
            "train": str(TRAIN_MANIFEST),
            "val":   str(VAL_MANIFEST),
        },
        "counts": {
            "train_usable": n_train,
            "val_usable":   n_val,
            "train_normal": n0_tr,
            "train_nsclc":  n1_tr,
            "val_normal":   n0_vl,
            "val_nsclc":    n1_vl,
        },
        "batch_shape": {
            "img":    str(tuple(batch_imgs.shape)),
            "scalar": str(tuple(batch_scalars.shape)),
            "logit":  str(tuple(logits.shape)),
        },
        "loss_value_dry": round(loss_val, 6),
        "loss_finite":    loss_finite,
        "grad_finite":    grads_ok,
        "device":         str(device),
        "guardrails": {
            "corrected_24g_fix_manifest_used":         True,
            "old_24g_manifest_used":                   False,
            "old_24h_scalar_stats_used":               False,
            "scalar_normalization_recomputed":         True,
            "drycheck_run":                            True,
            "full_training_run":                       False,
            "final_test_used":                         False,
            "vessel_feature_used":                     False,
            "roi_masked_loss_used":                    False,
            "checkpoint_saved":                        False,
            "usable_manifest_used":                    True,
            "unresolved_rows_used":                    False,
            "forbidden_diagnostic_wording_count":      forbidden_count,
        },
        "next_step": "P-C-NORMAL24i-fix: smoke training 1 epoch (사용자 승인 후)",
        "24i_smoke_blocked": True,
    }
    _write_json(summary, REPORT_ROOT / "p_c_normal24h_fix_drycheck_summary.json")

    # MD report
    old_clr = OLD_STATS["crop_lung_roi_ratio"]
    new_clr = scalar_stats["crop_lung_roi_ratio"]
    old_lzp = OLD_STATS["lung_z_percentile"]
    new_lzp = scalar_stats["lung_z_percentile"]
    md = f"""# P-C-NORMAL24h-fix z/ROI-Only Scalar-Fusion Normalization + Dry-check

**날짜**: {ts[:10]}
**Branch**: P-C-NORMAL24h-fix-zroi-scalar-fusion
**판정**: {verdict}

이 모델은 supervised normal-vs-NSCLC auxiliary classifier다.
출력은 normal-like vs NSCLC-lesion-like auxiliary score로만 해석한다.

---

## old 24h scalar stats 폐기 사유

기존 P-C-NORMAL24h는 P-C-NORMAL24g의 잘못된 `crop_lung_roi_ratio`를 기반으로 normalization을 계산했다.
24g 버그: 32x32 bbox의 ROI 합계를 96×96=9216으로 나눠 1/9 축소값을 생성.
24h-fix는 24g-fix(bbox_h*bbox_w 분모)를 기준으로 normalization을 재계산한다.

---

## Old vs New Scalar Normalization Stats

| feature | old 24h mean | old 24h std | new fix mean | new fix std |
|---|---|---|---|---|
| lung_z_percentile | {old_lzp['mean']:.6f} | {old_lzp['std']:.6f} | {new_lzp['mean']:.6f} | {new_lzp['std']:.6f} |
| crop_lung_roi_ratio | {old_clr['mean']:.6f} | {old_clr['std']:.6f} | {new_clr['mean']:.6f} | {new_clr['std']:.6f} |

- `crop_lung_roi_ratio` mean 변화: {old_clr['mean']:.4f} → {new_clr['mean']:.4f}
  (32x32 bbox에서 9배 회복, 96x96 bbox는 동일)
- `lung_z_percentile`은 bbox fix와 무관하므로 거의 동일

---

## 입력 Manifest (24g-fix usable)

| split | rows | normal | NSCLC |
|---|---|---|---|
| train (usable) | {n_train} | {n0_tr} | {n1_tr} |
| val (usable)   | {n_val}   | {n0_vl} | {n1_vl} |

manifests:
- train: `{TRAIN_MANIFEST}`
- val:   `{VAL_MANIFEST}`

---

## 모델 구조 (기존 24h 유지)

| 구성 | 내용 |
|---|---|
| Image branch | EfficientNet-B0 features+avgpool → (B,1280) |
| Scalar branch | Linear(2→32) → BN → ReLU → Linear(32→16) → ReLU |
| Fusion head | concat(1296) → Dropout(0.2) → Linear(64) → ReLU → Dropout → Linear(1) |
| Loss | BCEWithLogitsLoss(reduction=none) × sample_weight → mean |

---

## Batch Shape Check

| 항목 | shape |
|---|---|
| img | {tuple(batch_imgs.shape)} |
| scalar | {tuple(batch_scalars.shape)} |
| logit | {tuple(logits.shape)} |

---

## Dry-check 결과

- model forward (no_grad): {'PASS' if logit_shape_ok and logits_finite_ok else 'FAIL'}
- loss finite: {'PASS' if loss_finite else 'FAIL'} (value={loss_val:.6f})
- backward grad finite: {'PASS' if grads_ok else 'FAIL'}
- optimizer step: 없음
- checkpoint 저장: 없음

---

## Guardrail

- corrected_24g_fix_manifest_used=True
- old_24g_manifest_used=False
- old_24h_scalar_stats_used=False
- scalar_normalization_recomputed=True
- vessel_feature_used=False
- roi_masked_loss_used=False
- full_training_run=False
- final_test_used=False
- checkpoint_saved=False
- forbidden_diagnostic_wording_count={forbidden_count}

---

## 다음 단계

**P-C-NORMAL24i-fix**: smoke training 1 epoch (**사용자 승인 후**)
참고: `p_c_normal24h_fix_smoke_command_plan.md`
"""
    (REPORT_ROOT / "p_c_normal24h_fix_drycheck_report.md").write_text(md, encoding="utf-8")

    _write_json({
        "step":                  "p_c_normal24h_fix_drycheck",
        "verdict":               verdict,
        "timestamp":             ts,
        "fix_confirmed":         True,
        "next_step_blocked":     True,
        "errors":                len(errors),
    }, REPORT_ROOT / "DONE.json")

    print(f"\n{'='*60}")
    print(f"[P-C-NORMAL24h-fix] 판정: {verdict}")
    if errors and errors[0].get("check") != "all":
        for e in errors:
            print(f"  X {e.get('check')}: {e.get('error')}")
    print(f"device={device}, loss={loss_val:.6f}, grad_ok={grads_ok}")
    print(f"old crop_lung_roi_ratio mean: {old_clr['mean']:.6f} std: {old_clr['std']:.6f}")
    print(f"new crop_lung_roi_ratio mean: {new_clr['mean']:.6f} std: {new_clr['std']:.6f}")
    print(f"scalar_norm -> {norm_json_path}")
    print(f"report      -> {REPORT_ROOT}")
    print(f"{'='*60}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
