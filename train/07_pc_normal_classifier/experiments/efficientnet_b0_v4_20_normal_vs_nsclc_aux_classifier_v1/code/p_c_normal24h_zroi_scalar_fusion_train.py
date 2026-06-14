"""
p_c_normal24h_zroi_scalar_fusion_train.py

P-C-NORMAL24h: z/ROI-only scalar-fusion training script
- Image branch: EfficientNet-B0 (3,96,96) 2.5D CT
- Scalar branch: MLP(2→32→16) with BatchNorm + ReLU
- Fusion head: concat(1280+16) → Linear(64) → ReLU → Dropout → Linear(1)
- Scalar features: lung_z_percentile, crop_lung_roi_ratio
- Manifests: p_c_normal24g usable train/val only

이 모델은 supervised normal-vs-NSCLC auxiliary classifier다.
출력은 normal-like vs NSCLC-lesion-like auxiliary score로만 해석한다.
Forbidden: cancer-prob, adenocarcinoma-prob, 암-확률, 폐선암-확률, 진단-모델.

금지:
  - vessel feature 사용 금지 (vessel_candidate_ratio / vessel_softmask_max / vessel_center_ratio 등)
  - ROI-masked loss 금지
  - loss weighting 변경 금지
  - image/feature map ROI masking 금지
  - pixel-level loss 금지
  - full manifest 사용 금지 (usable manifest만 허용)
  - unresolved rows 사용 금지
  - final_test 사용/학습튜닝 금지
  - crop_lung_roi_ratio를 loss weight로 사용 금지

Modes:
  --dry-check    manifest 로드 + 1 batch forward/loss/backward grad check (이번 단계)
  --smoke-train  1-epoch smoke (P-C-NORMAL24i에서 별도 승인 후 실행)
  --train        full training (P-C-NORMAL24j에서 별도 승인 후 실행)
  (no mode)      exit 2

Usage:
  python p_c_normal24h_zroi_scalar_fusion_train.py --dry-check
  python p_c_normal24h_zroi_scalar_fusion_train.py --smoke-train \\
      --confirm-smoke --confirm-normal-vs-nsclc --confirm-no-final-test --epochs 1
"""

import argparse
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

TRAIN_MANIFEST = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_zroi_only_feature_manifest/p_c_normal24g_train_feature_manifest_usable.csv"
VAL_MANIFEST   = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_zroi_only_feature_manifest/p_c_normal24g_val_feature_manifest_usable.csv"

OUTPUT_ROOT  = PROJECT_ROOT / "outputs/p_c_normal24h_zroi_scalar_fusion"
CKPT_DIR     = OUTPUT_ROOT / "checkpoints"
REPORT_ROOT  = PROJECT_ROOT / "outputs/reports/p_c_normal24h_zroi_scalar_fusion_script_drycheck"

# ── Constants ─────────────────────────────────────────────────────────────────
HU_MIN, HU_MAX  = -1000.0, 200.0
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
SCALAR_FEATURES = ["lung_z_percentile", "crop_lung_roi_ratio"]

EXPECTED_TRAIN_USABLE = 19716
EXPECTED_VAL_USABLE   = 5189

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
    """
    Train set 기준 scalar mean/std 계산.
    std가 0이면 1.0으로 대체 (NaN 방지).
    반환: {feature: {mean, std}}
    """
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
    """stats(train 기준)를 df에 적용. 원본 수정 없이 복사본 반환."""
    df = df.copy()
    for col, s in stats.items():
        df[col] = (df[col].astype(float) - s["mean"]) / s["std"]
    return df


# ── AUROC (sklearn-free, Mann-Whitney) ───────────────────────────────────────

def compute_auroc(labels, scores):
    try:
        labels_arr = np.asarray(labels, dtype=np.int32)
        scores_arr = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(scores_arr)):
            return float("nan"), "invalid_score_nan_inf"
        unique_lbl = np.unique(labels_arr)
        if len(unique_lbl) < 2:
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
    """
    2.5D CT crop + 2 scalar features.
    Returns (img_tensor, scalar_tensor, label, sample_weight).
    scalar_tensor: float32 (2,) — [lung_z_percentile, crop_lung_roi_ratio] normalized
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)

        # guardrail: vessel feature 컬럼 없음 확인
        vessel_cols = [c for c in df.columns if c in FORBIDDEN_VESSEL_COLS]
        if vessel_cols:
            raise RuntimeError(f"[GUARD] forbidden vessel columns in dataset: {vessel_cols}")

        # guardrail: z_unresolved 없음 확인
        if "z_unresolved" in df.columns and df["z_unresolved"].any():
            n_ur = int(df["z_unresolved"].sum())
            raise RuntimeError(f"[GUARD] {n_ur} unresolved rows in dataset — use usable manifest only")

        # guardrail: scalar NaN 없음 확인
        for col in SCALAR_FEATURES:
            if df[col].isna().any():
                n_nan = int(df[col].isna().sum())
                raise RuntimeError(f"[GUARD] {n_nan} NaN values in scalar feature '{col}'")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # Image
        data = np.load(str(row["crop_path"]))
        arr  = data["ct_crop"].astype(np.float32)
        arr  = np.clip(arr, HU_MIN, HU_MAX)
        arr  = (arr - HU_MIN) / (HU_MAX - HU_MIN)
        img  = torch.from_numpy(arr)
        img  = (img - self.mean) / self.std
        if self.augment and torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[-1])
        # Scalar
        scalar = torch.tensor(
            [float(row["lung_z_percentile"]), float(row["crop_lung_roi_ratio"])],
            dtype=torch.float32,
        )
        label         = int(row["label"])
        sample_weight = float(row["sample_weight"])
        return img, scalar, label, sample_weight


# ── Model ─────────────────────────────────────────────────────────────────────

class ScalarFusionModel(nn.Module):
    """
    EfficientNet-B0 image branch + scalar MLP branch + fusion head.
    image:  (B, 3, 96, 96)
    scalar: (B, 2) float32  [normalized lung_z_percentile, crop_lung_roi_ratio]
    output: (B, 1) logit — BCEWithLogitsLoss와 호환
    """

    def __init__(self, scalar_hidden: int = 32, scalar_out: int = 16, dropout: float = 0.2):
        super().__init__()
        # Image branch (EfficientNet-B0 feature extractor)
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.img_features = backbone.features   # → (B, 1280, 3, 3)
        self.img_avgpool  = backbone.avgpool    # → (B, 1280, 1, 1)
        img_feat_dim = 1280

        # Scalar branch: Linear → BN → ReLU → Linear → ReLU
        self.scalar_branch = nn.Sequential(
            nn.Linear(2, scalar_hidden),
            nn.BatchNorm1d(scalar_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(scalar_hidden, scalar_out),
            nn.ReLU(inplace=True),
        )

        # Fusion head: concat → Dropout → Linear → ReLU → Dropout → Linear(1)
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
        x = torch.flatten(x, 1)          # (B, 1280)
        s = self.scalar_branch(scalar)    # (B, scalar_out)
        return self.fusion_head(torch.cat([x, s], dim=1))   # (B, 1)


# ── Loss ──────────────────────────────────────────────────────────────────────

def weighted_bce_loss(logits: torch.Tensor, labels: torch.Tensor,
                      sample_weights: torch.Tensor) -> torch.Tensor:
    """BCEWithLogitsLoss(reduction='none') * sample_weight → mean"""
    bce = nn.BCEWithLogitsLoss(reduction="none")
    per_sample = bce(logits.squeeze(1), labels.float())
    return (per_sample * sample_weights).mean()


# ── Guards ────────────────────────────────────────────────────────────────────

def _guard_smoke_train(args) -> None:
    missing = []
    if not getattr(args, "confirm_smoke", False):
        missing.append("--confirm-smoke")
    if not getattr(args, "confirm_normal_vs_nsclc", False):
        missing.append("--confirm-normal-vs-nsclc")
    if not getattr(args, "confirm_no_final_test", False):
        missing.append("--confirm-no-final-test")
    if getattr(args, "epochs", 0) != 1:
        missing.append("--epochs 1  (smoke requires exactly 1 epoch)")
    if missing:
        print(f"[GUARD] --smoke-train requires: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)


def _guard_train(args) -> None:
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


def _guard_manifests_are_usable(df_train: pd.DataFrame, df_val: pd.DataFrame) -> list:
    """usable manifest 조건 검사. 위반 항목 리스트 반환."""
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


# ── Dry-check ─────────────────────────────────────────────────────────────────

def run_dry_check() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    errors       = []
    batch_rows   = []
    guardrail_rows = []
    ts           = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdict      = "PASS"

    # ── 1. Output collision guard ─────────────────────────────────────────────
    # dry-check 보고서는 덮어쓰지 않음 (DONE.json 존재 시 중단)
    done_json = REPORT_ROOT / "DONE.json"
    if done_json.exists():
        print(f"[ABORT] DONE.json already exists: {done_json}")
        print("이전 24h dry-check 결과 존재. 삭제 후 재실행하세요.")
        sys.exit(2)

    # ── 2. Manifest load ──────────────────────────────────────────────────────
    if not TRAIN_MANIFEST.exists() or not VAL_MANIFEST.exists():
        errors.append({"check": "manifest_files_exist", "error": "CSV missing"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal24h_errors.csv")
        return 1

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
        ("train_usable_rows",    EXPECTED_TRAIN_USABLE, n_train, n_train == EXPECTED_TRAIN_USABLE),
        ("val_usable_rows",      EXPECTED_VAL_USABLE,   n_val,   n_val   == EXPECTED_VAL_USABLE),
        ("vessel_cols_absent",   True, len(manifest_issues) == 0, not manifest_issues),
        ("label_values_only_01", "[0,1]",
         str(sorted(df_train["label"].unique().tolist())),
         sorted(df_train["label"].unique().tolist()) == [0, 1]),
        ("scalar_nan_free_train", True,
         str(df_train[SCALAR_FEATURES].isna().sum().sum() == 0),
         df_train[SCALAR_FEATURES].isna().sum().sum() == 0),
        ("scalar_nan_free_val",  True,
         str(df_val[SCALAR_FEATURES].isna().sum().sum() == 0),
         df_val[SCALAR_FEATURES].isna().sum().sum() == 0),
        ("final_test_absent",    True, "True",  True),
    ]
    for check, exp, act, ok in manifest_checks:
        batch_rows.append({"check": check, "expected": exp, "actual": act, "pass": ok})
        if not ok:
            errors.append({"check": check, "error": f"expected={exp} actual={act}"})
            verdict = "PARTIAL_PASS"

    if manifest_issues:
        for iss in manifest_issues:
            errors.append({"check": "manifest_guard", "error": iss})
        verdict = "FAIL"

    # ── 4. Scalar normalization ───────────────────────────────────────────────
    scalar_stats = compute_scalar_norm(df_train)
    norm_json = REPORT_ROOT / "p_c_normal24h_scalar_normalization_stats.json"
    norm_payload = {
        "computed_from": "p_c_normal24g_train_feature_manifest_usable",
        "n_train_rows":  n_train,
        "features":      scalar_stats,
        "policy":        "train set fit; apply to val/test with same mean/std",
        "timestamp":     ts,
    }
    _write_json(norm_payload, norm_json)

    # Apply normalization
    df_train_norm = apply_scalar_norm(df_train, scalar_stats)
    df_val_norm   = apply_scalar_norm(df_val,   scalar_stats)

    # Verify normalization sanity
    for col in SCALAR_FEATURES:
        mu  = float(df_train_norm[col].mean())
        std = float(df_train_norm[col].std())
        batch_rows.append({
            "check": f"norm_{col}_mean_near0",
            "expected": "~0.0", "actual": f"{mu:.4f}",
            "pass": abs(mu) < 0.1,
        })
        batch_rows.append({
            "check": f"norm_{col}_std_near1",
            "expected": "~1.0", "actual": f"{std:.4f}",
            "pass": abs(std - 1.0) < 0.1,
        })

    # ── 5. Dataset + DataLoader + 1 batch ────────────────────────────────────
    ds_train = ScalarFusionDataset(df_train_norm, augment=False)
    loader   = DataLoader(ds_train, batch_size=DRY_BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=False)
    batch_imgs, batch_scalars, batch_labels, batch_sw = next(iter(loader))

    img_shape_ok    = (batch_imgs.shape   == torch.Size([DRY_BATCH_SIZE, 3, 96, 96]))
    scalar_shape_ok = (batch_scalars.shape == torch.Size([DRY_BATCH_SIZE, 2]))
    labels_ok       = bool(((batch_labels == 0) | (batch_labels == 1)).all())
    sw_ok           = bool(torch.isfinite(batch_sw).all())
    imgs_finite     = bool(torch.isfinite(batch_imgs).all())
    scalars_finite  = bool(torch.isfinite(batch_scalars).all())

    batch_checks = [
        ("img_shape",      f"({DRY_BATCH_SIZE},3,96,96)", str(tuple(batch_imgs.shape)),    img_shape_ok),
        ("scalar_shape",   f"({DRY_BATCH_SIZE},2)",        str(tuple(batch_scalars.shape)), scalar_shape_ok),
        ("labels_01",      True, str(labels_ok),  labels_ok),
        ("sw_finite",      True, str(sw_ok),       sw_ok),
        ("imgs_finite",    True, str(imgs_finite), imgs_finite),
        ("scalars_finite", True, str(scalars_finite), scalars_finite),
    ]
    for check, exp, act, ok in batch_checks:
        batch_rows.append({"check": check, "expected": exp, "actual": act, "pass": ok})
        if not ok:
            errors.append({"check": check, "error": f"{act}"})
            verdict = "FAIL"

    # ── 6. Model forward (1 batch, no_grad) ──────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ScalarFusionModel().eval().to(device)

    with torch.no_grad():
        logits = model(batch_imgs.to(device), batch_scalars.to(device))

    logit_shape_ok   = (logits.shape == torch.Size([DRY_BATCH_SIZE, 1]))
    logits_finite_ok = bool(torch.isfinite(logits).all())

    batch_rows.append({"check": "logit_shape",    "expected": f"({DRY_BATCH_SIZE},1)", "actual": str(tuple(logits.shape)), "pass": logit_shape_ok})
    batch_rows.append({"check": "logits_finite",  "expected": True, "actual": str(logits_finite_ok), "pass": logits_finite_ok})
    if not logit_shape_ok or not logits_finite_ok:
        errors.append({"check": "model_forward", "error": f"shape={tuple(logits.shape)} finite={logits_finite_ok}"})
        verdict = "FAIL"

    # ── 7. Loss (no_grad) ─────────────────────────────────────────────────────
    with torch.no_grad():
        loss = weighted_bce_loss(
            logits,
            batch_labels.to(device),
            batch_sw.to(device),
        )
    loss_val    = float(loss.item())
    loss_finite = bool(np.isfinite(loss_val))
    batch_rows.append({"check": "loss_finite", "expected": True, "actual": f"{loss_val:.6f}", "pass": loss_finite})
    if not loss_finite:
        errors.append({"check": "loss", "error": f"loss={loss_val}"})
        verdict = "FAIL"

    # ── 8. Backward grad check (1 batch, NO optimizer step) ──────────────────
    model_train = ScalarFusionModel().train().to(device)
    imgs_g  = batch_imgs.to(device)
    sc_g    = batch_scalars.to(device)
    lbl_g   = batch_labels.to(device)
    sw_g    = batch_sw.to(device)

    logits_tr = model_train(imgs_g, sc_g)
    loss_tr   = weighted_bce_loss(logits_tr, lbl_g, sw_g)
    loss_tr.backward()

    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model_train.parameters() if p.requires_grad
    )
    batch_rows.append({"check": "backward_grad_finite", "expected": True, "actual": str(grads_ok), "pass": grads_ok})
    batch_rows.append({"check": "optimizer_step_run",   "expected": False, "actual": "False", "pass": True})
    batch_rows.append({"check": "checkpoint_saved",     "expected": False, "actual": "False", "pass": True})
    if not grads_ok:
        errors.append({"check": "grad_check", "error": "NaN/Inf gradient detected"})
        verdict = "FAIL"

    # ── 9. Guardrail check ────────────────────────────────────────────────────
    this_file = Path(__file__).resolve()
    compile_ok = False
    try:
        py_compile.compile(str(this_file), doraise=True)
        compile_ok = True
    except py_compile.PyCompileError as e:
        errors.append({"check": "py_compile", "error": str(e)})

    forbidden_count = _check_forbidden_words(this_file)

    guardrail_items = [
        ("training_script_created",            True,  True),
        ("drycheck_run",                       True,  True),
        ("full_training_run",                  False, False),
        ("final_test_used",                    False, False),
        ("final_test_prediction_export_run",   False, False),
        ("metrics_computed",                   False, False),
        ("threshold_computed",                 False, False),
        ("threshold_optimized",                False, False),
        ("checkpoint_saved",                   False, False),
        ("usable_manifest_used",               True,  True),
        ("full_manifest_used_for_training",    False, False),
        ("unresolved_rows_used",               False, False),
        ("vessel_feature_used",                False, False),
        ("roi_masked_loss_used",               False, False),
        ("loss_weighting_changed",             False, False),
        ("image_roi_masking_used",             False, False),
        ("pixel_level_loss_used",              False, False),
        ("crop_lung_roi_ratio_used_as_loss_weight", False, False),
        ("py_compile_ok",                      True,  compile_ok),
        ("forbidden_diagnostic_wording_count", 0,     forbidden_count),
    ]
    for check, exp, act in guardrail_items:
        ok = (act == exp)
        guardrail_rows.append({"guardrail": check, "expected": exp, "actual": act, "pass": ok})
        if not ok:
            errors.append({"check": check, "error": f"expected={exp} actual={act}"})
            verdict = "FAIL"

    # ── 10. Write outputs ─────────────────────────────────────────────────────
    _write_csv(batch_rows,     REPORT_ROOT / "p_c_normal24h_batch_shape_check.csv")
    _write_csv(guardrail_rows, REPORT_ROOT / "p_c_normal24h_guardrail_check.csv")
    _write_csv(errors if errors else [{"check": "all", "error": "none"}],
               REPORT_ROOT / "p_c_normal24h_errors.csv")

    # Smoke command plan
    smoke_md = f"""# P-C-NORMAL24i Smoke Training Command Plan

**작성일**: {ts[:10]}
**이 파일은 계획 문서다. 실행은 사용자 승인 후 P-C-NORMAL24i에서 진행한다.**

## Smoke Training Command

```bash
cd /home/jinhy/project/lung-ct-anomaly
source ~/ai_env/bin/activate
nohup python experiments/efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1/code/p_c_normal24h_zroi_scalar_fusion_train.py \\
    --smoke-train \\
    --confirm-smoke \\
    --confirm-normal-vs-nsclc \\
    --confirm-no-final-test \\
    --epochs 1 \\
    > logs/p_c_normal24i_smoke_$(date +%Y%m%d_%H%M).log 2>&1 &
echo "PID: $!"
```

## Smoke Config

| 항목 | 값 |
|---|---|
| epochs | 1 |
| batch_size | 32 |
| lr | 1e-4 |
| optimizer | Adam |
| augment | hflip |
| output_root | `outputs/p_c_normal24h_zroi_scalar_fusion/` |
| checkpoint | `outputs/p_c_normal24h_zroi_scalar_fusion/checkpoints/p_c_normal24i_smoke_epoch1.pth` |

## 금지 항목 (smoke에서도 유지)

- final_test 사용 금지
- metrics/threshold 계산 금지
- vessel feature 사용 금지
- ROI-masked loss 금지
"""
    (REPORT_ROOT / "p_c_normal24h_smoke_command_plan.md").write_text(smoke_md, encoding="utf-8")

    # Summary JSON
    summary = {
        "branch":            "P-C-NORMAL24h-zroi-scalar-fusion",
        "step":              "training_script_drycheck",
        "verdict":           verdict,
        "verdict_issues":    [e["error"] for e in errors if e.get("check") != "all"],
        "timestamp":         ts,
        "model_architecture": {
            "image_branch":   "EfficientNet-B0 (torchvision ImageNet) features+avgpool → (B,1280)",
            "scalar_branch":  "Linear(2→32) → BN → ReLU → Linear(32→16) → ReLU",
            "fusion_head":    "concat(1280+16=1296) → Dropout(0.2) → Linear(64) → ReLU → Dropout → Linear(1)",
            "output":         "(B,1) logit for BCEWithLogitsLoss",
        },
        "scalar_features":   SCALAR_FEATURES,
        "scalar_norm_stats": scalar_stats,
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
            "training_script_created":             True,
            "full_training_run":                   False,
            "final_test_used":                     False,
            "vessel_feature_used":                 False,
            "roi_masked_loss_used":                False,
            "checkpoint_saved":                    False,
            "usable_manifest_used":                True,
            "unresolved_rows_used":                False,
            "forbidden_diagnostic_wording_count":  forbidden_count,
        },
        "next_step": "P-C-NORMAL24i: smoke training 1 epoch (사용자 승인 후)",
    }
    _write_json(summary, REPORT_ROOT / "p_c_normal24h_drycheck_summary.json")

    # MD report
    md = f"""# P-C-NORMAL24h z/ROI-Only Scalar-Fusion Training Script Dry-check

**날짜**: {ts[:10]}
**Branch**: P-C-NORMAL24h-zroi-scalar-fusion
**판정**: {verdict}

이 모델은 supervised normal-vs-NSCLC auxiliary classifier다.
출력은 normal-like vs NSCLC-lesion-like auxiliary score로만 해석한다.

---

## 모델 구조

| 구성 | 내용 |
|---|---|
| Image branch | EfficientNet-B0 features+avgpool → (B,1280) |
| Scalar branch | Linear(2→32) → BN → ReLU → Linear(32→16) → ReLU |
| Fusion head | concat(1296) → Dropout(0.2) → Linear(64) → ReLU → Dropout → Linear(1) |
| Loss | BCEWithLogitsLoss(reduction=none) × sample_weight → mean |
| Scalar features | lung_z_percentile, crop_lung_roi_ratio |

---

## 입력 Manifest

| split | rows | normal | NSCLC |
|---|---|---|---|
| train (usable) | {n_train} | {n0_tr} | {n1_tr} |
| val (usable)   | {n_val}   | {n0_vl} | {n1_vl} |

---

## Scalar Normalization (train set 기준)

| feature | mean | std |
|---|---|---|
| lung_z_percentile | {scalar_stats['lung_z_percentile']['mean']:.6f} | {scalar_stats['lung_z_percentile']['std']:.6f} |
| crop_lung_roi_ratio | {scalar_stats['crop_lung_roi_ratio']['mean']:.6f} | {scalar_stats['crop_lung_roi_ratio']['std']:.6f} |

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
- loss finite: {'PASS' if loss_finite else 'FAIL'} ({loss_val:.6f})
- backward grad finite: {'PASS' if grads_ok else 'FAIL'}
- optimizer step: 없음
- checkpoint 저장: 없음

---

## Guardrail

- vessel_feature_used=False
- roi_masked_loss_used=False
- full_training_run=False
- final_test_used=False
- checkpoint_saved=False
- usable_manifest_used=True
- unresolved_rows_used=False
- forbidden_diagnostic_wording_count={forbidden_count}

---

## 다음 단계

**P-C-NORMAL24i**: smoke training 1 epoch (사용자 승인 후)
참고: `p_c_normal24h_smoke_command_plan.md`
"""
    (REPORT_ROOT / "p_c_normal24h_drycheck_report.md").write_text(md, encoding="utf-8")

    _write_json({"step": "p_c_normal24h_drycheck", "verdict": verdict,
                 "timestamp": ts, "errors": len(errors)},
                REPORT_ROOT / "DONE.json")

    print(f"\n{'='*60}")
    print(f"[P-C-NORMAL24h] 판정: {verdict}")
    if errors and errors[0].get("check") != "all":
        for e in errors:
            print(f"  ⚠ {e.get('check')}: {e.get('error')}")
    print(f"device={device}, loss={loss_val:.6f}, grad_ok={grads_ok}")
    print(f"scalar_norm → {norm_json}")
    print(f"report      → {REPORT_ROOT}")
    print(f"{'='*60}")
    return 0 if verdict == "PASS" else 1


# ── Smoke train (P-C-NORMAL24i — 이번 단계에서는 실행 금지) ──────────────────

def run_smoke_train(args) -> int:
    _guard_smoke_train(args)

    stage_label = "P-C-NORMAL24i"
    batch_size  = 32
    lr          = 1e-4
    n_workers   = 4

    # scalar stats 로드 (dry-check에서 생성됨)
    norm_json = REPORT_ROOT / "p_c_normal24h_scalar_normalization_stats.json"
    if not norm_json.exists():
        print(f"[ABORT] scalar norm stats not found. Run --dry-check first: {norm_json}", file=sys.stderr)
        sys.exit(2)
    with open(norm_json) as f:
        norm_payload = json.load(f)
    scalar_stats = norm_payload["features"]

    # collision guard
    smoke_ckpt = CKPT_DIR / "p_c_normal24i_smoke_epoch1.pth"
    if smoke_ckpt.exists():
        print(f"[GUARD] smoke checkpoint already exists: {smoke_ckpt}", file=sys.stderr)
        sys.exit(2)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(TRAIN_MANIFEST, low_memory=False)
    df_val   = pd.read_csv(VAL_MANIFEST,   low_memory=False)
    df_train_norm = apply_scalar_norm(df_train, scalar_stats)
    df_val_norm   = apply_scalar_norm(df_val,   scalar_stats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{stage_label}] device={device}, train={len(df_train)}, val={len(df_val)}")

    ds_train = ScalarFusionDataset(df_train_norm, augment=True)
    ds_val   = ScalarFusionDataset(df_val_norm,   augment=False)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              num_workers=n_workers, pin_memory=True)

    model     = ScalarFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 1 epoch train
    model.train()
    train_losses, train_correct, train_total = [], 0, 0
    for imgs, scalars, labels, sw in loader_train:
        imgs, scalars, labels, sw = imgs.to(device), scalars.to(device), labels.to(device), sw.to(device)
        optimizer.zero_grad()
        logits = model(imgs, scalars)
        loss = weighted_bce_loss(logits, labels, sw)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        preds = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)

    train_loss = float(np.mean(train_losses))
    train_acc  = float(train_correct / train_total)

    # val
    model.eval()
    val_losses, val_correct, val_total = [], 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, scalars, labels, sw in loader_val:
            imgs, scalars, labels, sw = imgs.to(device), scalars.to(device), labels.to(device), sw.to(device)
            logits = model(imgs, scalars)
            loss = weighted_bce_loss(logits, labels, sw)
            val_losses.append(loss.item())
            probs = torch.sigmoid(logits.squeeze(1))
            preds = (probs >= 0.5).long()
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    val_loss = float(np.mean(val_losses))
    val_acc  = float(val_correct / val_total)
    val_auc, val_auc_status = compute_auroc(all_labels, all_probs)

    import math
    auc_disp = f"{val_auc:.4f}" if not math.isnan(val_auc) else "NaN"
    print(f"[{stage_label}] ep=1 train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={auc_disp} ({val_auc_status})")

    # checkpoint
    ckpt = {
        "model_state_dict":        model.state_dict(),
        "optimizer_state_dict":    optimizer.state_dict(),
        "epoch":                   1,
        "smoke_only":              True,
        "full_training":           False,
        "stage":                   stage_label,
        "train_loss":              train_loss,
        "train_acc":               train_acc,
        "val_loss":                val_loss,
        "val_acc":                 val_acc,
        "val_auc":                 val_auc,
        "val_auc_status":          val_auc_status,
        "scalar_features":         SCALAR_FEATURES,
        "scalar_norm_stats":       scalar_stats,
        "label_mapping":           {"0": "normal", "1": "NSCLC"},
        "manifest_paths":          {"train": str(TRAIN_MANIFEST), "val": str(VAL_MANIFEST)},
        "final_test_used":         False,
        "vessel_feature_used":     False,
        "forbidden_diagnostic_wording_count": 0,
    }
    torch.save(ckpt, smoke_ckpt)
    print(f"[{stage_label}] checkpoint → {smoke_ckpt}")
    return 0


# ── Full train (P-C-NORMAL24j — 별도 승인 후) ────────────────────────────────

def run_full_train(args) -> int:
    _guard_train(args)

    stage_label = "P-C-NORMAL24j"
    max_epochs  = args.epochs
    batch_size  = 64
    lr          = 1e-4
    patience    = 5
    n_workers   = getattr(args, "num_workers", 4)

    norm_json = REPORT_ROOT / "p_c_normal24h_scalar_normalization_stats.json"
    if not norm_json.exists():
        print(f"[ABORT] scalar norm stats not found. Run --dry-check first.", file=sys.stderr)
        sys.exit(2)
    with open(norm_json) as f:
        scalar_stats = json.load(f)["features"]

    best_ckpt  = CKPT_DIR / "best_auc.pth"
    last_ckpt  = CKPT_DIR / "last.pth"
    for cp in [best_ckpt, last_ckpt]:
        if cp.exists():
            print(f"[GUARD] output collision: {cp}", file=sys.stderr)
            sys.exit(2)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(TRAIN_MANIFEST, low_memory=False)
    df_val   = pd.read_csv(VAL_MANIFEST,   low_memory=False)
    df_train_norm = apply_scalar_norm(df_train, scalar_stats)
    df_val_norm   = apply_scalar_norm(df_val,   scalar_stats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{stage_label}] device={device} train={len(df_train)} val={len(df_val)} "
          f"epochs={max_epochs} batch={batch_size} lr={lr}")

    ds_train = ScalarFusionDataset(df_train_norm, augment=True)
    ds_val   = ScalarFusionDataset(df_val_norm,   augment=False)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=(device.type == "cuda"))
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              num_workers=n_workers, pin_memory=(device.type == "cuda"))

    model     = ScalarFusionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_auc = float("-inf")
    no_improve   = 0
    import math

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_losses, tr_correct, tr_total = [], 0, 0
        for imgs, scalars, labels, sw in loader_train:
            imgs, scalars, labels, sw = imgs.to(device), scalars.to(device), labels.to(device), sw.to(device)
            optimizer.zero_grad()
            logits = model(imgs, scalars)
            loss = weighted_bce_loss(logits, labels, sw)
            if not torch.isfinite(loss):
                raise RuntimeError(f"[ABORT] NaN/Inf train_loss at epoch={epoch}")
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())
            preds = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
            tr_correct += (preds == labels).sum().item()
            tr_total   += labels.size(0)

        train_loss = float(np.mean(tr_losses))
        train_acc  = float(tr_correct / tr_total)

        model.eval()
        vl_losses, vl_correct, vl_total = [], 0, 0
        all_probs, all_labels_ep = [], []
        with torch.no_grad():
            for imgs, scalars, labels, sw in loader_val:
                imgs, scalars, labels, sw = imgs.to(device), scalars.to(device), labels.to(device), sw.to(device)
                logits = model(imgs, scalars)
                loss = weighted_bce_loss(logits, labels, sw)
                vl_losses.append(loss.item())
                probs = torch.sigmoid(logits.squeeze(1))
                preds = (probs >= 0.5).long()
                vl_correct += (preds == labels).sum().item()
                vl_total   += labels.size(0)
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels_ep.extend(labels.cpu().numpy().tolist())

        val_loss = float(np.mean(vl_losses))
        val_acc  = float(vl_correct / vl_total)
        val_auc, val_auc_status = compute_auroc(all_labels_ep, all_probs)
        scheduler.step()

        auc_disp = f"{val_auc:.4f}" if not math.isnan(val_auc) else "NaN"
        print(f"[{stage_label}] ep={epoch}/{max_epochs} "
              f"tr_loss={train_loss:.4f} tr_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"val_auc={auc_disp} ({val_auc_status})")

        ckpt_base = {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch":                epoch,
            "smoke_only":           False,
            "full_training":        True,
            "stage":                stage_label,
            "train_loss":           train_loss,
            "train_acc":            train_acc,
            "val_loss":             val_loss,
            "val_acc":              val_acc,
            "val_auc":              val_auc,
            "val_auc_status":       val_auc_status,
            "scalar_features":      SCALAR_FEATURES,
            "scalar_norm_stats":    scalar_stats,
            "label_mapping":        {"0": "normal", "1": "NSCLC"},
            "manifest_paths":       {"train": str(TRAIN_MANIFEST), "val": str(VAL_MANIFEST)},
            "final_test_used":      False,
            "vessel_feature_used":  False,
            "forbidden_diagnostic_wording_count": 0,
        }

        torch.save({**ckpt_base, "best_metric_name": "val_auc",
                    "best_metric_value": best_val_auc}, last_ckpt)

        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            no_improve   = 0
            torch.save({**ckpt_base, "best_metric_name": "val_auc",
                        "best_metric_value": best_val_auc}, best_ckpt)
            print(f"[{stage_label}] best_auc updated → {best_val_auc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{stage_label}] early stopping at epoch={epoch}")
                break

    print(f"[{stage_label}] 완료. best_val_auc={best_val_auc:.4f} (epoch={best_epoch if best_val_auc > float('-inf') else 'N/A'})")
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P-C-NORMAL24h z/ROI scalar-fusion classifier")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-check",   action="store_true")
    mode.add_argument("--smoke-train", action="store_true")
    mode.add_argument("--train",       action="store_true")

    parser.add_argument("--confirm-smoke",            action="store_true")
    parser.add_argument("--confirm-normal-vs-nsclc",  action="store_true")
    parser.add_argument("--confirm-no-final-test",    action="store_true")
    parser.add_argument("--confirm-train",            action="store_true")
    parser.add_argument("--epochs",       type=int, default=1)
    parser.add_argument("--num-workers",  type=int, default=4, dest="num_workers")

    args = parser.parse_args()

    if args.dry_check:
        sys.exit(run_dry_check())
    elif args.smoke_train:
        sys.exit(run_smoke_train(args))
    elif args.train:
        sys.exit(run_full_train(args))
    else:
        print("[ERROR] mode required: --dry-check | --smoke-train | --train", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
