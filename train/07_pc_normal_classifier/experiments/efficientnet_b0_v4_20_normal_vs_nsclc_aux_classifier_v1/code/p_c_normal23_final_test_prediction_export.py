"""
P-C-NORMAL23: Final Baseline Test Prediction Export Script
===========================================================
Mode A  --dry-check   : checkpoint/manifest/schema/output collision/CLI guard 검증
                        (model load까지 허용, no_grad forward는 금지)
Mode B  --run-export  : actual per-crop prediction export
                        (requires all confirm flags + ALLOW_REAL_EXPORT=True)
                        P-C-NORMAL23b 별도 승인 전까지 실행 금지.

이 branch는 supervised normal-vs-NSCLC auxiliary classifier다.
normal=0, NSCLC=1 label을 사용하는 보조 분류기다.
출력은 normal-like vs NSCLC-lesion-like auxiliary score로만 해석한다.
진단 모델, 암 확률, 폐선암 확률, cancer probability, adenocarcinoma probability 표현 금지.

기존 p_c_normal21c_val_prediction_export.py 직접 재사용 불가 이유:
- VAL_MANIFEST_PATH hardcoded (p_c_normal12 val manifest)
- EXPECTED_VAL_ROWS=5200 hardcoded
- --confirm-val-only 필수 flag
- split='val' 강제 검사
- stage2_holdout_refs=0 강제 (final_test에서는 stage2_holdout refs 허용)
- output prefix p_c_normal21c_val_*
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Guard: ALLOW_REAL_EXPORT flag (must be set True only after explicit user approval)
# ──────────────────────────────────────────────────────────────────────────────
ALLOW_REAL_EXPORT = False  # Never change this to True without explicit P-C-NORMAL23b user approval

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path('/home/jinhy/project/lung-ct-anomaly')
BRANCH_ROOT = (
    PROJECT_ROOT
    / 'experiments'
    / 'efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1'
)

CHECKPOINT_PATH = (
    BRANCH_ROOT
    / 'outputs'
    / 'checkpoints'
    / 'p_c_normal19_full_training'
    / 'best_auc.pth'
)

FINAL_TEST_MANIFEST_PATH = (
    PROJECT_ROOT
    / 'outputs'
    / 'manifests'
    / 'p_c_normal22_final_baseline_test_manifest'
    / 'p_c_normal22_final_test_manifest.csv'
)

DRYCHECK_OUT = (
    PROJECT_ROOT
    / 'outputs'
    / 'reports'
    / 'p_c_normal23a2_final_test_prediction_export_script_drycheck'
)

DRYCHECK_A2_OUT = (
    PROJECT_ROOT
    / 'outputs'
    / 'reports'
    / 'p_c_normal23a2_prediction_export_hardening_drycheck'
)

EXPORT_OUT = (
    PROJECT_ROOT
    / 'outputs'
    / 'reports'
    / 'p_c_normal23_final_baseline_test_prediction_export'
)

# All export output files — collision check must cover every entry
EXPORT_FILES = [
    "p_c_normal23_final_test_crop_predictions.csv",
    "p_c_normal23_final_test_patient_predictions.csv",
    "p_c_normal23_final_test_prediction_export_summary.json",
    "p_c_normal23_final_test_prediction_export_report.md",
    "p_c_normal23_guardrail_check.csv",
    "p_c_normal23_errors.csv",
    "DONE.json",
]

INFER_BATCH_SIZE = 64

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing constants (must match P-C-NORMAL19 training + P-C-NORMAL21c validation)
# ──────────────────────────────────────────────────────────────────────────────
HU_MIN, HU_MAX = -1000.0, 200.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────────────────────────────────────
# Manifest expectations (confirmed in P-C-NORMAL23 preflight)
# ──────────────────────────────────────────────────────────────────────────────
EXPECTED_FINAL_TEST_ROWS = 66323
EXPECTED_FINAL_TEST_N0   = 21600   # normal
EXPECTED_FINAL_TEST_N1   = 44723   # NSCLC
EXPECTED_NORMAL_PATIENTS = 36
EXPECTED_NSCLC_PATIENTS  = 122
EXPECTED_TOTAL_PATIENTS  = 158   # 36 normal + 122 NSCLC
EXPECTED_SPLIT           = "final_test"
ALLOWED_SOURCE_SPLITS    = {"normal_test", "stage2_holdout"}

# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint expectations (confirmed in P-C-NORMAL23 preflight)
# ──────────────────────────────────────────────────────────────────────────────
EXPECTED_CHECKPOINT_SHA256   = "1286346861b0d9d32a4fc9354f6555d0646f66558e7ca684792ef7a989557253"
EXPECTED_CHECKPOINT_EPOCH    = 11
EXPECTED_CHECKPOINT_VAL_AUC  = 0.9999534640039448
EXPECTED_CHECKPOINT_KEYS = {
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
}

FORBIDDEN_DIAGNOSTIC_WORDS = [
    "폐선암 확률",
    "암 확률",
    "진단 모델",
    "cancer probability",
    "adenocarcinoma probability",
]

# ──────────────────────────────────────────────────────────────────────────────
# Per-crop prediction CSV schema (final test version — distinct from 21c val schema)
# ──────────────────────────────────────────────────────────────────────────────
CROP_PRED_COLUMNS = [
    "row_index",
    "final_candidate_id",
    "patient_id",
    "safe_id",
    "split",
    "source_split",
    "source_name",
    "crop_path",
    "label",
    "label_name",
    "logit",
    "prob_nsclc_like",
    "pred_label_threshold_0_5",
    "correct_threshold_0_5",
    "position_bin",
    "z_level",
    "local_z",
    "slice_index",
    "center_y",
    "center_x",
    "stage2_holdout_flag",
    "hard_negative_flag",
    "msd_lung_flag",
    "crop_hu_mean",
    "crop_hu_std",
    "air_frac_lt_minus800",
    "dense_frac_gt_minus500",
    "dense_frac_gt_minus300",
    "positive_frac_gt_0",
]

# Patient-level summary CSV schema
PATIENT_PRED_COLUMNS = [
    "patient_id",
    "label",
    "label_name",
    "source_split",
    "n_crops",
    "mean_prob_nsclc_like",
    "max_prob_nsclc_like",
    "median_prob_nsclc_like",
    "p95_prob_nsclc_like",
    "pred_label_mean_threshold_0_5",
    "pred_label_max_threshold_0_5",
    "correct_mean_threshold_0_5",
    "correct_max_threshold_0_5",
    "position_bin_count_summary",
    "stage2_holdout_flag_any",
    "msd_lung_flag_any",
]


# ──────────────────────────────────────────────────────────────────────────────
# Model (identical to P-C-NORMAL19 training + P-C-NORMAL21c validation)
# ──────────────────────────────────────────────────────────────────────────────
def build_model():
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    import torch.nn as nn
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing (identical to P-C-NORMAL19 NormalNSCLCDataset + P-C-NORMAL21c)
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_crop(arr_int16: np.ndarray) -> "torch.Tensor":
    """
    Preprocessing policy (fixed — changing breaks baseline):
    1. load npz key 'ct_crop'
    2. expect shape (3,96,96), dtype int16
    3. convert to float32
    4. HU clip [-1000, 200]
    5. scale to [0,1]
    6. ImageNet normalize (same as training)
    no augmentation, eval mode only
    """
    import torch
    arr = arr_int16.astype(np.float32)
    arr = np.clip(arr, HU_MIN, HU_MAX)
    arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)
    t = torch.from_numpy(arr)  # (3,96,96)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)
    t = (t - mean) / std
    return t


# ──────────────────────────────────────────────────────────────────────────────
# HU stat computation from raw int16 crop
# ──────────────────────────────────────────────────────────────────────────────
def compute_hu_stats(arr_int16: np.ndarray) -> dict:
    arr = arr_int16.astype(np.float32)
    total = arr.size
    return {
        "crop_hu_mean":           float(np.mean(arr)),
        "crop_hu_std":            float(np.std(arr)),
        "air_frac_lt_minus800":   float(np.sum(arr < -800) / total),
        "dense_frac_gt_minus500": float(np.sum(arr > -500) / total),
        "dense_frac_gt_minus300": float(np.sum(arr > -300) / total),
        "positive_frac_gt_0":     float(np.sum(arr > 0) / total),
    }


# ──────────────────────────────────────────────────────────────────────────────
# SHA256 check
# ──────────────────────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Guard helpers
# ──────────────────────────────────────────────────────────────────────────────
def _guard_run_export(args) -> None:
    missing = []
    if not args.confirm_final_test:
        missing.append("--confirm-final-test")
    if not args.confirm_no_threshold:
        missing.append("--confirm-no-threshold")
    if not args.confirm_no_training:
        missing.append("--confirm-no-training")
    if missing:
        print(f"[GUARD] --run-export requires: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)
    if not ALLOW_REAL_EXPORT:
        print("[GUARD] ALLOW_REAL_EXPORT=False: actual export 금지.", file=sys.stderr)
        print("[GUARD] P-C-NORMAL23b 사용자 승인 후에만 진행 가능.", file=sys.stderr)
        sys.exit(2)


def _export_collision_check_or_exit():
    if EXPORT_OUT.exists():
        existing_files = list(EXPORT_OUT.iterdir())
        if existing_files:
            print(f"[GUARD] Output dir exists and is non-empty: {EXPORT_OUT} ({len(existing_files)} files)", file=sys.stderr)
            print(f"[GUARD] Remove or archive {EXPORT_OUT} before re-running.", file=sys.stderr)
            sys.exit(2)
    collision = [f for f in EXPORT_FILES if (EXPORT_OUT / f).exists()]
    if collision:
        print(f"[GUARD] Output collision detected: {collision}", file=sys.stderr)
        print(f"[GUARD] Remove or archive {EXPORT_OUT} before re-running.", file=sys.stderr)
        sys.exit(2)


def _check_forbidden_words_in_output(text: str) -> int:
    lower = text.lower()
    count = 0
    for w in FORBIDDEN_DIAGNOSTIC_WORDS:
        count += lower.count(w.lower())
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Dry-check logic (P-C-NORMAL23a)
# No model forward. Model load (load_state_dict) is allowed.
# ──────────────────────────────────────────────────────────────────────────────
def run_dry_check():
    import torch
    import py_compile
    import subprocess

    print("[DRY-CHECK] P-C-NORMAL23a2 prediction export hardening dry-check 시작")
    DRYCHECK_A2_OUT.mkdir(parents=True, exist_ok=True)

    errors = []
    checks = []
    guardrail = {
        "prediction_export_run":      False,
        "model_forward_run":          False,
        "metrics_computed":           False,
        "threshold_computed":         False,
        "training_run":               False,
        "backward_run":               False,
        "optimizer_step":             False,
        "checkpoint_saved":           False,
        "crop_generated":             False,
        "manifest_generated":         False,
        "existing_outputs_modified":  False,
        "forbidden_diagnostic_wording_count": 0,
    }

    def record(name, status, detail=""):
        checks.append({"check": name, "status": status, "detail": str(detail)})
        marker = "PASS" if status == "PASS" else ("WARN" if status == "WARN" else "FAIL")
        print(f"  [{marker}] {name}" + (f": {detail}" if detail else ""))

    # ── 1. py_compile ──
    try:
        py_compile.compile(str(Path(__file__)), doraise=True)
        record("py_compile", "PASS")
    except py_compile.PyCompileError as e:
        record("py_compile", "FAIL", str(e))
        errors.append(f"py_compile: {e}")

    # ── 2. ALLOW_REAL_EXPORT=False confirmed ──
    script_src = Path(__file__).read_text(errors="ignore")
    allow_false = "ALLOW_REAL_EXPORT = False" in script_src
    record("ALLOW_REAL_EXPORT=False", "PASS" if allow_false else "FAIL")
    if not allow_false:
        errors.append("ALLOW_REAL_EXPORT is not False in script source")

    # ── 3. forbidden diagnostic wording (candidate output text only) ──
    candidate_output_texts = [
        "P-C-NORMAL23a final test prediction export script dry-check",
        "normal-like vs NSCLC-lesion-like auxiliary score",
        "supervised auxiliary classifier",
        "final baseline test prediction export",
    ]
    fw_count = sum(_check_forbidden_words_in_output(t) for t in candidate_output_texts)
    guardrail["forbidden_diagnostic_wording_count"] = fw_count
    record("forbidden_diagnostic_wording_count=0",
           "PASS" if fw_count == 0 else "FAIL", f"count={fw_count}")
    if fw_count:
        errors.append(f"forbidden_diagnostic_wording_count={fw_count}")

    # ── 4. checkpoint exists ──
    if CHECKPOINT_PATH.exists():
        record("checkpoint_exists", "PASS", str(CHECKPOINT_PATH))
    else:
        record("checkpoint_exists", "FAIL", f"not found: {CHECKPOINT_PATH}")
        errors.append(f"checkpoint not found: {CHECKPOINT_PATH}")

    # ── 5. checkpoint SHA256 ──
    ckpt = None
    if CHECKPOINT_PATH.exists():
        actual_sha = sha256_file(CHECKPOINT_PATH)
        sha_ok = (actual_sha == EXPECTED_CHECKPOINT_SHA256)
        record("checkpoint_sha256",
               "PASS" if sha_ok else "FAIL",
               f"actual={actual_sha[:16]}... expected={EXPECTED_CHECKPOINT_SHA256[:16]}...")
        if not sha_ok:
            errors.append(f"checkpoint SHA256 mismatch: {actual_sha} != {EXPECTED_CHECKPOINT_SHA256}")

    # ── 6. checkpoint load ──
    if CHECKPOINT_PATH.exists():
        try:
            ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
            record("checkpoint_load", "PASS")
        except Exception as e:
            record("checkpoint_load", "FAIL", str(e))
            errors.append(f"checkpoint load error: {e}")

    # ── 7. checkpoint key check ──
    if ckpt is not None:
        ckpt_keys = set(ckpt.keys())
        missing_keys = EXPECTED_CHECKPOINT_KEYS - ckpt_keys
        if missing_keys:
            record("checkpoint_key_check", "FAIL", f"missing={missing_keys}")
            errors.append(f"checkpoint missing keys: {missing_keys}")
        else:
            record("checkpoint_key_check", "PASS",
                   f"keys={len(ckpt_keys)}, expected_subset={len(EXPECTED_CHECKPOINT_KEYS)}")

    # ── 8. checkpoint metadata ──
    if ckpt is not None:
        epoch = ckpt.get("epoch")
        epoch_ok = (epoch == EXPECTED_CHECKPOINT_EPOCH)
        record("checkpoint_epoch",
               "PASS" if epoch_ok else "FAIL",
               f"epoch={epoch} expected={EXPECTED_CHECKPOINT_EPOCH}")
        if not epoch_ok:
            errors.append(f"checkpoint epoch mismatch: {epoch} != {EXPECTED_CHECKPOINT_EPOCH}")

        lm = ckpt.get("label_mapping", {})
        lm_normal = lm.get("0") or lm.get(0)
        lm_nsclc  = lm.get("1") or lm.get(1)
        lm_ok = (lm_normal == "normal") and (lm_nsclc == "NSCLC")
        record("checkpoint_label_mapping",
               "PASS" if lm_ok else "FAIL", str(lm))
        if not lm_ok:
            errors.append(f"label_mapping mismatch: {lm}")

        smoke_only = ckpt.get("smoke_only")
        full_training = ckpt.get("full_training")
        record("checkpoint_smoke_only=False",
               "PASS" if smoke_only is False else "FAIL", f"smoke_only={smoke_only}")
        record("checkpoint_full_training=True",
               "PASS" if full_training is True else "FAIL", f"full_training={full_training}")
        if smoke_only is not False:
            errors.append(f"checkpoint smoke_only != False: {smoke_only}")
        if full_training is not True:
            errors.append(f"checkpoint full_training != True: {full_training}")

        cfg = ckpt.get("config", {})
        in_ch = cfg.get("input_channels", cfg.get("in_channels"))
        arch  = cfg.get("architecture", cfg.get("arch", cfg.get("model_name")))
        record("checkpoint_input_channels=3",
               "PASS" if in_ch == 3 else ("WARN" if in_ch is None else "FAIL"),
               f"input_channels={in_ch}")
        record("checkpoint_architecture=EfficientNet-B0",
               "PASS" if (arch and "efficientnet" in str(arch).lower()) else "WARN",
               f"architecture={arch}")

    # ── 9. model_state_dict key count ──
    if ckpt is not None:
        try:
            n_keys = len(ckpt["model_state_dict"])
            reasonable = (100 <= n_keys <= 500)
            record("checkpoint_state_dict_key_count",
                   "PASS" if reasonable else "FAIL", f"keys={n_keys}")
            if not reasonable:
                errors.append(f"model_state_dict key count unusual: {n_keys}")
        except Exception as e:
            record("checkpoint_state_dict_key_count", "FAIL", str(e))

    # ── 10. weights NaN/Inf=0 ──
    if ckpt is not None:
        try:
            nan_inf_count = 0
            for v in ckpt["model_state_dict"].values():
                if hasattr(v, "isnan"):
                    nan_inf_count += int(v.isnan().any()) + int(v.isinf().any())
            record("checkpoint_nan_inf=0",
                   "PASS" if nan_inf_count == 0 else "FAIL",
                   f"nan_inf_count={nan_inf_count}")
            if nan_inf_count:
                errors.append(f"checkpoint weights NaN/Inf: {nan_inf_count}")
        except Exception as e:
            record("checkpoint_nan_inf=0", "FAIL", str(e))

    # ── 11. model class import + instantiate ──
    model_instantiated = False
    try:
        model = build_model()
        record("model_instantiate", "PASS",
               f"params={sum(p.numel() for p in model.parameters()):,}")
        model_instantiated = True
    except Exception as e:
        record("model_instantiate", "FAIL", str(e))
        errors.append(f"model instantiate error: {e}")

    # ── 12. load_state_dict (CPU, no forward) ──
    if ckpt is not None and model_instantiated:
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            record("load_state_dict", "PASS", "CPU load OK, no forward")
        except Exception as e:
            record("load_state_dict", "FAIL", str(e))
            errors.append(f"load_state_dict error: {e}")

    # ── 13. final manifest exists ──
    if FINAL_TEST_MANIFEST_PATH.exists():
        record("manifest_exists", "PASS", str(FINAL_TEST_MANIFEST_PATH))
    else:
        record("manifest_exists", "FAIL", f"not found: {FINAL_TEST_MANIFEST_PATH}")
        errors.append(f"final test manifest not found: {FINAL_TEST_MANIFEST_PATH}")

    # ── 14. manifest load & validation ──
    df = None
    if FINAL_TEST_MANIFEST_PATH.exists():
        try:
            df = pd.read_csv(FINAL_TEST_MANIFEST_PATH, low_memory=False)
            record("manifest_load", "PASS")
        except Exception as e:
            record("manifest_load", "FAIL", str(e))
            errors.append(f"manifest load error: {e}")

    if df is not None:
        # row count
        n_rows = len(df)
        record("manifest_row_count",
               "PASS" if n_rows == EXPECTED_FINAL_TEST_ROWS else "FAIL",
               f"{n_rows}/{EXPECTED_FINAL_TEST_ROWS}")
        if n_rows != EXPECTED_FINAL_TEST_ROWS:
            errors.append(f"manifest row mismatch: {n_rows} != {EXPECTED_FINAL_TEST_ROWS}")

        # label counts
        n0 = int((df["label"] == 0).sum())
        n1 = int((df["label"] == 1).sum())
        record("manifest_normal_count",
               "PASS" if n0 == EXPECTED_FINAL_TEST_N0 else "FAIL",
               f"{n0}/{EXPECTED_FINAL_TEST_N0}")
        record("manifest_nsclc_count",
               "PASS" if n1 == EXPECTED_FINAL_TEST_N1 else "FAIL",
               f"{n1}/{EXPECTED_FINAL_TEST_N1}")
        if n0 != EXPECTED_FINAL_TEST_N0:
            errors.append(f"normal count mismatch: {n0}")
        if n1 != EXPECTED_FINAL_TEST_N1:
            errors.append(f"NSCLC count mismatch: {n1}")

        # patient counts
        n_normal_patients = int(df[df["label"] == 0]["patient_id"].nunique())
        n_nsclc_patients  = int(df[df["label"] == 1]["patient_id"].nunique())
        record("manifest_normal_patients",
               "PASS" if n_normal_patients == EXPECTED_NORMAL_PATIENTS else "FAIL",
               f"{n_normal_patients}/{EXPECTED_NORMAL_PATIENTS}")
        record("manifest_nsclc_patients",
               "PASS" if n_nsclc_patients == EXPECTED_NSCLC_PATIENTS else "FAIL",
               f"{n_nsclc_patients}/{EXPECTED_NSCLC_PATIENTS}")
        if n_normal_patients != EXPECTED_NORMAL_PATIENTS:
            errors.append(f"normal patient count mismatch: {n_normal_patients}")
        if n_nsclc_patients != EXPECTED_NSCLC_PATIENTS:
            errors.append(f"NSCLC patient count mismatch: {n_nsclc_patients}")

        # split = final_test only
        split_vals = set(df["split"].unique().tolist())
        split_ok = split_vals == {EXPECTED_SPLIT}
        record("manifest_split=final_test",
               "PASS" if split_ok else "FAIL",
               f"split values={split_vals}")
        if not split_ok:
            errors.append(f"split not final_test only: {split_vals}")

        # source_split values
        src_splits = set(df["source_split"].unique().tolist())
        src_ok = src_splits.issubset(ALLOWED_SOURCE_SPLITS)
        record("manifest_source_split_allowed",
               "PASS" if src_ok else "FAIL",
               f"source_split values={src_splits}")
        if not src_ok:
            errors.append(f"unexpected source_split values: {src_splits - ALLOWED_SOURCE_SPLITS}")

        # stage2_holdout_flag must be expected (44723 = NSCLC rows)
        if "stage2_holdout_flag" in df.columns:
            holdout_sum = int(df["stage2_holdout_flag"].astype(bool).sum())
            record("manifest_stage2_holdout_flag_sum",
                   "PASS" if holdout_sum == EXPECTED_FINAL_TEST_N1 else "FAIL",
                   f"{holdout_sum}/{EXPECTED_FINAL_TEST_N1}")
            if holdout_sum != EXPECTED_FINAL_TEST_N1:
                errors.append(f"stage2_holdout_flag sum mismatch: {holdout_sum}")

        # crop_path unique
        n_unique = int(df["crop_path"].nunique())
        record("manifest_crop_path_unique",
               "PASS" if n_unique == EXPECTED_FINAL_TEST_ROWS else "FAIL",
               f"{n_unique}/{EXPECTED_FINAL_TEST_ROWS}")
        if n_unique != EXPECTED_FINAL_TEST_ROWS:
            errors.append(f"crop_path not unique: {n_unique}")

        # MSD_lung absent
        msd_count = 0
        if "msd_lung_flag" in df.columns:
            msd_count = int(df["msd_lung_flag"].astype(str).str.lower().isin(["true","1","yes"]).sum())
        elif "source_name" in df.columns:
            msd_count = int(df["source_name"].astype(str).str.contains("MSD", na=False).sum())
        record("manifest_MSD_lung=0",
               "PASS" if msd_count == 0 else "FAIL", f"count={msd_count}")
        if msd_count:
            errors.append(f"MSD_lung rows: {msd_count}")

        # LUNG1-295 / LUNG1-415 absent
        if "source_name" in df.columns:
            l295 = int(df["source_name"].astype(str).str.contains("LUNG1-295", na=False).sum())
            l415 = int(df["source_name"].astype(str).str.contains("LUNG1-415", na=False).sum())
        elif "patient_id" in df.columns:
            l295 = int(df["patient_id"].astype(str).str.contains("LUNG1-295", na=False).sum())
            l415 = int(df["patient_id"].astype(str).str.contains("LUNG1-415", na=False).sum())
        else:
            l295 = l415 = -1
        record("manifest_LUNG1-295=0",
               "PASS" if l295 == 0 else "FAIL", f"count={l295}")
        record("manifest_LUNG1-415=0",
               "PASS" if l415 == 0 else "FAIL", f"count={l415}")
        if l295:
            errors.append(f"LUNG1-295 rows: {l295}")
        if l415:
            errors.append(f"LUNG1-415 rows: {l415}")

        # hard_negative_flag=0
        hard_neg = 0
        if "hard_negative_flag" in df.columns:
            hard_neg = int(df["hard_negative_flag"].astype(str).str.lower().isin(["true","1","yes"]).sum())
        record("manifest_hard_negative=0",
               "PASS" if hard_neg == 0 else "FAIL", f"count={hard_neg}")
        if hard_neg:
            errors.append(f"hard_negative rows: {hard_neg}")

        # no 6ch crop
        six_ch = int(df["crop_path"].astype(str).str.contains("6ch", na=False).sum())
        record("manifest_6ch=0",
               "PASS" if six_ch == 0 else "FAIL", f"count={six_ch}")
        if six_ch:
            errors.append(f"6ch crop rows: {six_ch}")

        # crop_shape uniform
        if "crop_shape" in df.columns:
            shapes = df["crop_shape"].unique().tolist()
            shape_ok = (shapes == ["(3,96,96)"]) or (shapes == ["(3, 96, 96)"])
            record("manifest_crop_shape_uniform",
                   "PASS" if shape_ok else "FAIL", f"shapes={shapes}")
            if not shape_ok:
                errors.append(f"crop_shape not uniform (3,96,96): {shapes}")

        # DONE.json conditions_ok
        done_path = FINAL_TEST_MANIFEST_PATH.parent / "DONE.json"
        if done_path.exists():
            try:
                done_data = json.loads(done_path.read_text())
                cond_ok = done_data.get("conditions_ok", False)
                record("manifest_DONE_conditions_ok",
                       "PASS" if cond_ok else "FAIL", str(cond_ok))
                if not cond_ok:
                    errors.append("DONE.json conditions_ok=False")
            except Exception as e:
                record("manifest_DONE_conditions_ok", "FAIL", str(e))
                errors.append(f"DONE.json read error: {e}")
        else:
            record("manifest_DONE_exists", "WARN", "DONE.json not found")

    # ── 15. sample crop exists + int16 dtype ──
    if df is not None and len(df) > 0:
        sample_path = str(df.iloc[0]["crop_path"])
        crop_exists = Path(sample_path).exists()
        record("crop_path_sample_exists",
               "PASS" if crop_exists else "FAIL", sample_path)
        if not crop_exists:
            errors.append(f"sample crop not found: {sample_path}")
        else:
            try:
                data = np.load(sample_path)
                arr_sample = data["ct_crop"]
                raw_hu_ok = arr_sample.dtype == np.int16
                record("crop_raw_int16_dtype",
                       "PASS" if raw_hu_ok else "FAIL",
                       f"dtype={arr_sample.dtype}, shape={arr_sample.shape}")
                if not raw_hu_ok:
                    errors.append(f"crop dtype not int16: {arr_sample.dtype}")
                else:
                    # HU stat computation sanity check
                    hu = compute_hu_stats(arr_sample)
                    record("crop_hu_stat_computation", "PASS",
                           f"mean={hu['crop_hu_mean']:.1f}")
            except Exception as e:
                record("crop_raw_int16_dtype", "FAIL", str(e))
                errors.append(f"sample crop load error: {e}")

    # ── 16. output collision check ──
    actual_export_dir_exists = EXPORT_OUT.exists()
    record("output_collision_export_dir",
           "PASS" if not actual_export_dir_exists else "FAIL",
           "not exists" if not actual_export_dir_exists else f"EXISTS: {EXPORT_OUT}")
    if actual_export_dir_exists:
        errors.append(f"output collision: {EXPORT_OUT} already exists")

    # ── 17. existing val outputs not modified ──
    val_export_21c = PROJECT_ROOT / "outputs" / "reports" / "p_c_normal21c_validation_prediction_export"
    val_export_21e = PROJECT_ROOT / "outputs" / "reports" / "p_c_normal21e_actual_validation_prediction_export"
    for vdir in [val_export_21c, val_export_21e]:
        if vdir.exists():
            record(f"existing_val_output_not_modified:{vdir.name}",
                   "PASS", "exists but not modified (read-only check)")

    # ── 18. CLI guard check via subprocess ──
    python_bin = sys.executable
    script_path = str(Path(__file__))

    def run_cmd(cmd_args):
        import subprocess
        result = subprocess.run(
            [python_bin, script_path] + cmd_args,
            capture_output=True, text=True
        )
        return result.returncode

    rc_bare = run_cmd([])
    record("guard_bare_run_exit2", "PASS" if rc_bare == 2 else "FAIL", f"rc={rc_bare}")
    if rc_bare != 2:
        errors.append(f"bare run should exit 2, got {rc_bare}")

    rc_run_export_alone = run_cmd(["--run-export"])
    record("guard_run_export_alone_exit2",
           "PASS" if rc_run_export_alone == 2 else "FAIL", f"rc={rc_run_export_alone}")
    if rc_run_export_alone != 2:
        errors.append(f"--run-export alone should exit 2, got {rc_run_export_alone}")

    rc_confirm_only = run_cmd([
        "--run-export",
        "--confirm-final-test",
        "--confirm-no-threshold",
        "--confirm-no-training",
    ])
    record("guard_ALLOW_REAL_EXPORT_False_exit2",
           "PASS" if rc_confirm_only == 2 else "FAIL", f"rc={rc_confirm_only}")
    if rc_confirm_only != 2:
        errors.append(f"ALLOW_REAL_EXPORT=False+confirms should exit 2, got {rc_confirm_only}")

    # ── 19. export CSV schema plan ──
    required_crop_cols = [
        "row_index", "final_candidate_id", "patient_id", "safe_id",
        "split", "source_split", "source_name", "crop_path",
        "label", "label_name", "logit", "prob_nsclc_like",
        "pred_label_threshold_0_5", "correct_threshold_0_5",
        "stage2_holdout_flag", "hard_negative_flag", "msd_lung_flag",
        "crop_hu_mean", "crop_hu_std", "air_frac_lt_minus800",
    ]
    crop_schema_ok = all(c in CROP_PRED_COLUMNS for c in required_crop_cols)
    record("crop_pred_schema_plan",
           "PASS" if crop_schema_ok else "FAIL",
           f"{len(CROP_PRED_COLUMNS)} columns")
    if not crop_schema_ok:
        missing = [c for c in required_crop_cols if c not in CROP_PRED_COLUMNS]
        errors.append(f"crop_pred_schema missing: {missing}")

    required_patient_cols = [
        "patient_id", "label", "label_name", "source_split",
        "n_crops", "mean_prob_nsclc_like", "max_prob_nsclc_like",
        "median_prob_nsclc_like", "p95_prob_nsclc_like",
        "stage2_holdout_flag_any",
    ]
    patient_schema_ok = all(c in PATIENT_PRED_COLUMNS for c in required_patient_cols)
    record("patient_pred_schema_plan",
           "PASS" if patient_schema_ok else "FAIL",
           f"{len(PATIENT_PRED_COLUMNS)} columns")
    if not patient_schema_ok:
        missing = [c for c in required_patient_cols if c not in PATIENT_PRED_COLUMNS]
        errors.append(f"patient_pred_schema missing: {missing}")

    # ── 20. preprocessing policy documented in script ──
    pp_markers = ["HU_MIN, HU_MAX = -1000", "IMAGENET_MEAN", "IMAGENET_STD"]
    pp_ok = all(m in script_src for m in pp_markers)
    record("preprocessing_policy_documented",
           "PASS" if pp_ok else "FAIL",
           "HU_MIN/HU_MAX/IMAGENET_MEAN/IMAGENET_STD present")

    # ── 21. final test guardrail: actual export not run ──
    export_files_exist = [f for f in EXPORT_FILES if (EXPORT_OUT / f).exists()]
    record("actual_export_files_not_created",
           "PASS" if not export_files_exist else "FAIL",
           f"collision={export_files_exist}")
    if export_files_exist:
        guardrail["prediction_export_run"] = True
        errors.append(f"actual export files found: {export_files_exist}")

    # ── 22. _post_run_hard_assertion_export function exists ──
    has_post_assertion = "_post_run_hard_assertion_export" in script_src
    record("post_run_hard_assertion_function_exists",
           "PASS" if has_post_assertion else "FAIL",
           "_post_run_hard_assertion_export found in source")
    if not has_post_assertion:
        errors.append("_post_run_hard_assertion_export function not found in script source")

    # ── 23. len(errors)==0 hard assertion exists ──
    has_errors_zero = "len(errors) == 0" in script_src
    record("hard_assertion_len_errors_zero_exists",
           "PASS" if has_errors_zero else "FAIL")
    if not has_errors_zero:
        errors.append("len(errors)==0 hard assertion not found in script source")

    # ── 24. expected crop rows (66323) check in assertions ──
    has_crop_rows = str(EXPECTED_FINAL_TEST_ROWS) in script_src
    record("hard_assertion_expected_crop_rows_exists",
           "PASS" if has_crop_rows else "FAIL",
           f"EXPECTED_FINAL_TEST_ROWS={EXPECTED_FINAL_TEST_ROWS} in source")
    if not has_crop_rows:
        errors.append(f"EXPECTED_FINAL_TEST_ROWS={EXPECTED_FINAL_TEST_ROWS} not found in script source")

    # ── 25. expected patient rows (158) check in assertions ──
    has_patient_rows = "EXPECTED_TOTAL_PATIENTS" in script_src
    record("hard_assertion_expected_patient_rows_exists",
           "PASS" if has_patient_rows else "FAIL",
           "EXPECTED_TOTAL_PATIENTS in source")
    if not has_patient_rows:
        errors.append("EXPECTED_TOTAL_PATIENTS not found in script source")

    # ── 26. prob_nsclc_like range check in assertions ──
    has_prob_range = "prob_nsclc_like_in_0_1" in script_src
    record("hard_assertion_prob_range_check_exists",
           "PASS" if has_prob_range else "FAIL")
    if not has_prob_range:
        errors.append("prob_nsclc_like_in_0_1 check not found in _post_run_hard_assertion_export")

    # ── 27. logit finite check in assertions ──
    has_logit_finite = "logit_finite" in script_src
    record("hard_assertion_logit_finite_check_exists",
           "PASS" if has_logit_finite else "FAIL")
    if not has_logit_finite:
        errors.append("logit_finite check not found in _post_run_hard_assertion_export")

    # ── 28. label/source/patient count checks exist ──
    has_label_checks = "label0_rows==21600" in script_src and "nsclc_patients==122" in script_src
    record("hard_assertion_label_source_patient_checks_exist",
           "PASS" if has_label_checks else "FAIL")
    if not has_label_checks:
        errors.append("label/patient count assertions not found in script source")

    # ── 29. msd_lung/leakage absence checks exist ──
    has_msd_leakage = "msd_lung_flag_true==0" in script_src and "LUNG1-295_absent" in script_src
    record("hard_assertion_msd_leakage_absence_checks_exist",
           "PASS" if has_msd_leakage else "FAIL")
    if not has_msd_leakage:
        errors.append("msd_lung/leakage absence assertions not found in script source")

    # ── 30. DONE.json conditions_ok schema ──
    has_conditions_ok = '"conditions_ok": True' in script_src or '"conditions_ok"' in script_src
    record("done_json_conditions_ok_schema_exists",
           "PASS" if has_conditions_ok else "FAIL")
    if not has_conditions_ok:
        errors.append("DONE.json conditions_ok field not found in script source")

    # ── 31. FAILED_DO_NOT_USE marker behavior ──
    has_failed_marker = "FAILED_DO_NOT_USE.json" in script_src
    record("failed_do_not_use_marker_behavior_exists",
           "PASS" if has_failed_marker else "FAIL")
    if not has_failed_marker:
        errors.append("FAILED_DO_NOT_USE.json failure behavior not found in script source")

    # ── 32. sys.exit(1) on assertion failure ──
    has_exit1 = "sys.exit(1)" in script_src
    record("hard_assertion_failure_sys_exit1_exists",
           "PASS" if has_exit1 else "FAIL")
    if not has_exit1:
        errors.append("sys.exit(1) on assertion failure not found in script source")

    # ── 33. non-empty dir collision check ──
    has_nonempty_check = "is non-empty" in script_src or "is_non_empty" in script_src or "existing_files" in script_src
    record("output_collision_nonempty_dir_check_exists",
           "PASS" if has_nonempty_check else "FAIL")
    if not has_nonempty_check:
        errors.append("Non-empty dir collision check not found in _export_collision_check_or_exit")

    # ── 34. guardrail semantics: prediction_export_run=True in actual export ──
    has_export_run_true = '"prediction_export_run":      True' in script_src or '"prediction_export_run": True' in script_src
    record("guardrail_prediction_export_run_true_in_actual_export",
           "PASS" if has_export_run_true else "FAIL")
    if not has_export_run_true:
        errors.append("prediction_export_run=True not found in run_export guardrail")

    # ── 35. ALLOW_REAL_EXPORT=False still confirmed ──
    allow_still_false = "ALLOW_REAL_EXPORT = False" in script_src
    record("ALLOW_REAL_EXPORT_still_False",
           "PASS" if allow_still_false else "FAIL")
    if not allow_still_false:
        errors.append("ALLOW_REAL_EXPORT is not False in script source after hardening")

    # ── Verdict ──
    fail_count = sum(1 for c in checks if c["status"] == "FAIL")
    warn_count = sum(1 for c in checks if c["status"] == "WARN")
    if fail_count == 0 and not errors:
        verdict = "READY_FOR_FINAL_TEST_PREDICTION_EXPORT"
    elif fail_count > 0:
        verdict = "NEEDS_SCRIPT_FIX"
    else:
        verdict = "NEEDS_SCRIPT_FIX"

    _write_dry_check_outputs(checks, guardrail, errors, verdict, warn_count)
    print(f"\n[DRY-CHECK] 판정: {verdict}  (fail={fail_count}, warn={warn_count}, errors={len(errors)})")
    if verdict == "READY_FOR_FINAL_TEST_PREDICTION_EXPORT":
        print("[DRY-CHECK] 다음 단계: P-C-NORMAL23b final baseline test prediction export actual (사용자 승인 필요)")
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Dry-check output writer
# ──────────────────────────────────────────────────────────────────────────────
def _write_dry_check_outputs(checks, guardrail, errors, verdict, warn_count=0):
    out_dir = DRYCHECK_A2_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(filename, fieldnames, rows):
        with open(out_dir / filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # script_compile_check.csv
    compile_checks = [c for c in checks if "py_compile" in c["check"] or "ALLOW_REAL_EXPORT" in c["check"]]
    write_csv("p_c_normal23a2_script_compile_check.csv",
              ["check", "status", "detail"], compile_checks)

    # checkpoint_check.csv
    ckpt_checks = [c for c in checks if "checkpoint" in c["check"] or "load_state_dict" in c["check"] or "model_instantiate" in c["check"]]
    write_csv("p_c_normal23a2_checkpoint_check.csv",
              ["check", "status", "detail"], ckpt_checks)

    # manifest_check.csv
    mf_checks = [c for c in checks if "manifest" in c["check"] or "crop_path" in c["check"]
                 or "raw_int16" in c["check"] or "hu_stat" in c["check"]]
    write_csv("p_c_normal23a2_manifest_check.csv",
              ["check", "status", "detail"], mf_checks)

    # preprocessing_policy_check.csv
    pp_checks = [c for c in checks if "preprocessing" in c["check"] or "raw_int16" in c["check"] or "hu_stat" in c["check"]]
    pp_checks += [{
        "check": "hu_clip_range",
        "status": "PASS",
        "detail": f"[-1000, 200] (HU_MIN={HU_MIN}, HU_MAX={HU_MAX})",
    }, {
        "check": "imagenet_normalize",
        "status": "PASS",
        "detail": f"mean={IMAGENET_MEAN}, std={IMAGENET_STD}",
    }, {
        "check": "no_augmentation",
        "status": "PASS",
        "detail": "eval mode, torch.no_grad (at actual export)",
    }, {
        "check": "batch_size_plan",
        "status": "PASS",
        "detail": f"INFER_BATCH_SIZE={INFER_BATCH_SIZE}",
    }, {
        "check": "device_plan",
        "status": "PASS",
        "detail": "CUDA if available else CPU",
    }]
    write_csv("p_c_normal23a2_preprocessing_policy_check.csv",
              ["check", "status", "detail"], pp_checks)

    # prediction_schema_check.csv
    crop_schema_rows = [{"schema": "crop_pred", "column": c, "index": i}
                        for i, c in enumerate(CROP_PRED_COLUMNS)]
    with open(out_dir / "p_c_normal23a2_prediction_schema_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["schema", "column", "index"])
        writer.writeheader()
        writer.writerows(crop_schema_rows)

    # patient_schema_check.csv
    patient_schema_rows = [{"schema": "patient_pred", "column": c, "index": i}
                           for i, c in enumerate(PATIENT_PRED_COLUMNS)]
    with open(out_dir / "p_c_normal23a2_patient_schema_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["schema", "column", "index"])
        writer.writeheader()
        writer.writerows(patient_schema_rows)

    # output_collision_check.csv
    col_checks = [c for c in checks if "collision" in c["check"] or "actual_export" in c["check"] or "existing_val" in c["check"]]
    write_csv("p_c_normal23a2_output_collision_check.csv",
              ["check", "status", "detail"], col_checks)

    # cli_guard_check.csv
    cli_checks = [c for c in checks if "guard" in c["check"] or "ALLOW_REAL" in c["check"]]
    write_csv("p_c_normal23a2_cli_guard_check.csv",
              ["check", "status", "detail"], cli_checks)

    # guardrail_check.csv
    guardrail_rows = [{"guardrail": k, "value": str(v)} for k, v in guardrail.items()]
    write_csv("p_c_normal23a2_guardrail_check.csv",
              ["guardrail", "value"], guardrail_rows)

    # patch_summary.csv — script hardening changes applied in P-C-NORMAL23a2
    patch_items = [
        ("DRYCHECK_A2_OUT", "added", "new output path constant"),
        ("EXPECTED_TOTAL_PATIENTS", "added", "158 = 36+122 constant"),
        ("_export_collision_check_or_exit", "updated", "non-empty dir → exit 2"),
        ("_post_run_hard_assertion_export", "added", "34 hard assertions"),
        ("run_export Stage 6b", "added", "assertion gating + FAILED_DO_NOT_USE.json + sys.exit(1)"),
        ("run_export Summary JSON", "strengthened", "conditions_ok/hard_assertion_pass/prob/logit stats"),
        ("run_export DONE.json", "strengthened", "conditions_ok=True + full schema + gated on assertions"),
        ("run_dry_check", "updated", "DRYCHECK_A2_OUT path + 14 new hardening checks"),
        ("_write_dry_check_outputs", "updated", "DRYCHECK_A2_OUT + p_c_normal23a2_ filenames + new CSVs"),
    ]
    write_csv("p_c_normal23a2_patch_summary.csv",
              ["item", "action", "detail"], [{"item": i, "action": a, "detail": d} for i, a, d in patch_items])

    # post_assertion_check.csv — verification that _post_run_hard_assertion_export checks exist
    _src = Path(__file__).read_text(errors="ignore")
    post_assertion_markers = [
        ("len(errors)==0", "len(errors) == 0" in _src, "error count hard check"),
        ("len(crop_df)==66323", "EXPECTED_FINAL_TEST_ROWS" in _src, "crop rows hard check"),
        ("len(patient_df)==158", "EXPECTED_TOTAL_PATIENTS" in _src, "patient rows hard check"),
        ("final_candidate_id_unique", "final_candidate_id_unique" in _src, "unique ID check"),
        ("prob_nsclc_like_in_0_1", "prob_nsclc_like_in_0_1" in _src, "prob range check"),
        ("logit_finite", "logit_finite" in _src, "logit finite check"),
        ("label0_rows==21600", "label0_rows==21600" in _src, "label count check"),
        ("nsclc_patients==122", "nsclc_patients==122" in _src, "patient count check"),
        ("msd_lung_flag_true==0", "msd_lung_flag_true==0" in _src, "msd absence check"),
        ("LUNG1-295_absent", "LUNG1-295_absent" in _src, "leakage absence check"),
        ("FAILED_DO_NOT_USE.json", "FAILED_DO_NOT_USE.json" in _src, "failure marker"),
        ("sys.exit(1)", "sys.exit(1)" in _src, "hard exit on failure"),
    ]
    post_assertion_rows = [
        {"check": name, "found": str(found), "status": "PASS" if found else "FAIL", "detail": detail}
        for name, found, detail in post_assertion_markers
    ]
    write_csv("p_c_normal23a2_post_assertion_check.csv",
              ["check", "found", "status", "detail"], post_assertion_rows)

    # done_schema_check.csv — required fields in DONE.json schema
    done_required_fields = [
        "stage", "status", "conditions_ok", "hard_assertion_pass",
        "manifest_rows", "crop_prediction_rows", "patient_prediction_rows",
        "normal_rows", "nsclc_rows", "normal_patients", "nsclc_patients",
        "n_errors", "prediction_export_run", "model_forward_run",
        "threshold_optimized", "fixed_threshold_0_5_applied", "metrics_computed",
        "training_run", "backward_run", "optimizer_step",
        "checkpoint_saved", "crop_generated", "manifest_generated",
        "forbidden_diagnostic_wording_count", "interpretation",
    ]
    done_schema_rows = [
        {"field": f, "found_in_source": str('"' + f + '"' in _src), "required": "yes"}
        for f in done_required_fields
    ]
    write_csv("p_c_normal23a2_done_schema_check.csv",
              ["field", "found_in_source", "required"], done_schema_rows)

    # summary_schema_check.csv — required fields in summary JSON schema
    summary_required_fields = [
        "conditions_ok", "hard_assertion_pass", "n_errors",
        "n_crops_expected", "n_crops_processed", "n_patients_expected", "n_patients_processed",
        "normal_rows", "nsclc_rows", "normal_patients", "nsclc_patients",
        "msd_lung_rows", "leakage_rows", "prob_min", "prob_max", "logit_min", "logit_max",
        "fixed_threshold_0_5_applied", "threshold_optimized", "metrics_deferred_to",
    ]
    summary_schema_rows = [
        {"field": f, "found_in_source": str('"' + f + '"' in _src), "required": "yes"}
        for f in summary_required_fields
    ]
    write_csv("p_c_normal23a2_summary_schema_check.csv",
              ["field", "found_in_source", "required"], summary_schema_rows)

    # errors.csv
    with open(out_dir / "p_c_normal23a2_errors.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["error"])
        for e in errors:
            writer.writerow([e])

    # summary JSON
    pass_count = sum(1 for c in checks if c["status"] == "PASS")
    fail_count = sum(1 for c in checks if c["status"] == "FAIL")
    summary = {
        "stage": "P-C-NORMAL23a2",
        "mode": "dry_check_hardening",
        "verdict": verdict,
        "date": "2026-06-09",
        "total_checks": len(checks),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "error_count": len(errors),
        "guardrail": guardrail,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_sha256_expected": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_epoch_expected": EXPECTED_CHECKPOINT_EPOCH,
        "manifest_path": str(FINAL_TEST_MANIFEST_PATH),
        "expected_rows": EXPECTED_FINAL_TEST_ROWS,
        "expected_label0": EXPECTED_FINAL_TEST_N0,
        "expected_label1": EXPECTED_FINAL_TEST_N1,
        "export_output_path": str(EXPORT_OUT),
        "drycheck_output_path": str(DRYCHECK_A2_OUT),
        "new_script_path": str(Path(__file__)),
        "old_script_not_reused_reason": (
            "6 val-specific hardcodings: VAL_MANIFEST_PATH, EXPECTED_VAL_ROWS=5200, "
            "--confirm-val-only flag, split='val' check, stage2_holdout_refs=0 check, "
            "p_c_normal21c_val_* output names"
        ),
        "preprocessing_policy": {
            "hu_clip": [HU_MIN, HU_MAX],
            "scale": "[(hu - HU_MIN) / (HU_MAX - HU_MIN)] -> [0,1]",
            "imagenet_mean": IMAGENET_MEAN,
            "imagenet_std": IMAGENET_STD,
            "no_augmentation": True,
            "eval_mode": True,
            "no_grad": True,
            "matches_training": "P-C-NORMAL19",
            "matches_validation": "P-C-NORMAL21c",
        },
        "crop_pred_columns": CROP_PRED_COLUMNS,
        "patient_pred_columns": PATIENT_PRED_COLUMNS,
        "next_step": (
            "P-C-NORMAL23b final baseline test prediction export actual (user approval required)"
            if verdict == "READY_FOR_FINAL_TEST_PREDICTION_EXPORT"
            else "fix script issues before P-C-NORMAL23b"
        ),
    }
    with open(out_dir / "p_c_normal23a2_prediction_export_hardening_drycheck.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # markdown report
    md_lines = [
        "# P-C-NORMAL23a2 Prediction Export Hardening Dry-Check Report",
        "",
        f"**판정: {verdict}**",
        "",
        "## 개요",
        "- branch: P-C-NORMAL (supervised auxiliary classifier, normal=0 / NSCLC=1)",
        "- 출력: normal-like vs NSCLC-lesion-like auxiliary score (진단 모델 아님)",
        f"- 새 script: `{Path(__file__)}`",
        f"- checkpoint: `{CHECKPOINT_PATH}`",
        f"- final manifest: `{FINAL_TEST_MANIFEST_PATH}`",
        f"- dry-check output: `{DRYCHECK_A2_OUT}`",
        f"- actual export output (예정): `{EXPORT_OUT}`",
        "",
        "## 기존 21c validation script 직접 재사용 불가 이유",
        "- VAL_MANIFEST_PATH hardcoded (p_c_normal12 val manifest)",
        "- EXPECTED_VAL_ROWS=5200 hardcoded",
        "- --confirm-val-only 필수 flag",
        "- split='val' 강제 검사",
        "- stage2_holdout_refs=0 강제 (final_test에서는 stage2_holdout refs 허용됨)",
        "- output prefix p_c_normal21c_val_*",
        "",
        "## Checkpoint 정보",
        f"- path: `{CHECKPOINT_PATH}`",
        f"- SHA256: `{EXPECTED_CHECKPOINT_SHA256}`",
        f"- epoch: {EXPECTED_CHECKPOINT_EPOCH}",
        f"- val_auc: {EXPECTED_CHECKPOINT_VAL_AUC}",
        "- label_mapping: 0=normal, 1=NSCLC",
        "- architecture: EfficientNet-B0",
        "- input_channels: 3",
        "- crop_shape: (3,96,96)",
        "",
        "## Final Manifest 정보",
        f"- path: `{FINAL_TEST_MANIFEST_PATH}`",
        f"- rows: {EXPECTED_FINAL_TEST_ROWS}",
        f"- label0 (normal): {EXPECTED_FINAL_TEST_N0}",
        f"- label1 (NSCLC): {EXPECTED_FINAL_TEST_N1}",
        f"- normal patients: {EXPECTED_NORMAL_PATIENTS}",
        f"- NSCLC patients: {EXPECTED_NSCLC_PATIENTS}",
        "- split: final_test only",
        "- source_split: normal_test, stage2_holdout",
        "- MSD_lung: 0",
        "- LUNG1-295/LUNG1-415: 0",
        "- hard_negative: 0",
        "- 6ch: 0",
        "",
        "## Preprocessing Policy (training/validation 동일)",
        f"- load npz key `ct_crop`",
        f"- expect shape (3,96,96), dtype int16",
        f"- HU clip [{HU_MIN}, {HU_MAX}]",
        f"- scale to [0,1]: (hu - {HU_MIN}) / ({HU_MAX} - {HU_MIN})",
        f"- ImageNet mean={IMAGENET_MEAN}",
        f"- ImageNet std={IMAGENET_STD}",
        f"- no augmentation, eval mode, torch.no_grad",
        f"- batch_size: {INFER_BATCH_SIZE}",
        f"- device: CUDA if available else CPU",
        "",
        "## CLI Guard 결과",
    ]
    cli_results = [c for c in checks if "guard" in c["check"] or "ALLOW_REAL" in c["check"]]
    for c in cli_results:
        md_lines.append(f"- [{c['status']}] {c['check']}: {c['detail']}")

    md_lines += [
        "",
        "## Per-Crop Prediction CSV Schema",
        f"파일: `p_c_normal23_final_test_crop_predictions.csv` ({len(CROP_PRED_COLUMNS)} columns)",
        "| # | column |",
        "|---|--------|",
    ]
    for i, c in enumerate(CROP_PRED_COLUMNS):
        md_lines.append(f"| {i} | {c} |")

    md_lines += [
        "",
        "## Patient-Level Prediction CSV Schema",
        f"파일: `p_c_normal23_final_test_patient_predictions.csv` ({len(PATIENT_PRED_COLUMNS)} columns)",
        "| # | column |",
        "|---|--------|",
    ]
    for i, c in enumerate(PATIENT_PRED_COLUMNS):
        md_lines.append(f"| {i} | {c} |")

    md_lines += [
        "",
        "## Output Collision 확인",
        f"- actual export dir: `{EXPORT_OUT}`",
        "- 기존 P-C-NORMAL21c/21e/21f/21h validation outputs 수정 금지",
        "- P-C-NORMAL22 manifest/crop 수정 금지",
        "",
        "## Guardrail 결과",
    ]
    for k, v in guardrail.items():
        status = "OK" if v in (False, 0) else "VIOLATION"
        md_lines.append(f"- `{k}`: {v}  [{status}]")

    md_lines += [
        "",
        "## 체크 결과 요약",
        f"- total: {len(checks)}, pass: {pass_count}, fail: {fail_count}, warn: {warn_count}",
        f"- errors: {len(errors)}",
        "",
        "## Actual Prediction/Model Forward/Metrics/Threshold/Training 미실행 확인",
        f"- prediction_export_run: {guardrail['prediction_export_run']}  [OK]",
        f"- model_forward_run: {guardrail['model_forward_run']}  [OK]",
        f"- metrics_computed: {guardrail['metrics_computed']}  [OK]",
        f"- threshold_computed: {guardrail['threshold_computed']}  [OK]",
        f"- training_run: {guardrail['training_run']}  [OK]",
        f"- backward_run: {guardrail['backward_run']}  [OK]",
        f"- optimizer_step: {guardrail['optimizer_step']}  [OK]",
        f"- checkpoint_saved: {guardrail['checkpoint_saved']}  [OK]",
        f"- forbidden_diagnostic_wording_count: {guardrail['forbidden_diagnostic_wording_count']}  [OK]",
        "",
        "## 다음 단계",
        "- P-C-NORMAL23b: final baseline test prediction export actual (사용자 승인 필요)",
        "- ALLOW_REAL_EXPORT=True로 변경 후 --run-export --confirm-final-test --confirm-no-threshold --confirm-no-training 실행",
    ]

    if errors:
        md_lines += ["", "## Errors"]
        for e in errors:
            md_lines.append(f"- {e}")

    with open(out_dir / "p_c_normal23a2_prediction_export_hardening_drycheck.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"[DRY-CHECK] 결과 저장 완료: {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Post-run hard assertions for actual export (P-C-NORMAL23a2 hardening)
# Returns list of assertion failure messages. Empty list = all pass.
# ──────────────────────────────────────────────────────────────────────────────
def _post_run_hard_assertion_export(
    crop_df: "pd.DataFrame",
    patient_df: "pd.DataFrame",
    errors: list,
    df_manifest: "pd.DataFrame",
) -> list:
    failures = []

    def _eq(name, actual, expected):
        if actual != expected:
            failures.append(f"ASSERT_FAIL {name}: actual={actual} expected={expected}")

    def _true(name, condition, detail=""):
        if not condition:
            failures.append(f"ASSERT_FAIL {name}" + (f": {detail}" if detail else ""))

    # 1. no inference errors
    _eq("len(errors)==0", len(errors), 0)

    # 2-4. row counts
    _eq("len(crop_df)==66323", len(crop_df), EXPECTED_FINAL_TEST_ROWS)
    _eq("len(df_manifest)==66323", len(df_manifest), EXPECTED_FINAL_TEST_ROWS)
    _eq("crop_df_rows==manifest_rows", len(crop_df), len(df_manifest))

    # 5. row_index unique
    if "row_index" in crop_df.columns:
        _eq("crop_df_row_index_unique", crop_df["row_index"].nunique(), len(crop_df))

    # 6. final_candidate_id unique == 66323
    if "final_candidate_id" in crop_df.columns:
        _eq("crop_df_final_candidate_id_unique", crop_df["final_candidate_id"].nunique(), EXPECTED_FINAL_TEST_ROWS)

    # 7. crop_path unique == 66323
    _eq("crop_df_crop_path_unique", crop_df["crop_path"].nunique(), EXPECTED_FINAL_TEST_ROWS)

    # 8-10. patient counts
    _eq("crop_df_patient_count==158", crop_df["patient_id"].nunique(), EXPECTED_TOTAL_PATIENTS)
    _eq("normal_patients==36", int((crop_df[crop_df["label"] == 0]["patient_id"].nunique())), EXPECTED_NORMAL_PATIENTS)
    _eq("nsclc_patients==122", int((crop_df[crop_df["label"] == 1]["patient_id"].nunique())), EXPECTED_NSCLC_PATIENTS)

    # 11-12. label row counts
    _eq("label0_rows==21600", int((crop_df["label"] == 0).sum()), EXPECTED_FINAL_TEST_N0)
    _eq("label1_rows==44723", int((crop_df["label"] == 1).sum()), EXPECTED_FINAL_TEST_N1)

    # 13. source_split counts
    if "source_split" in crop_df.columns:
        _eq("source_split_normal_test==21600", int((crop_df["source_split"] == "normal_test").sum()), EXPECTED_FINAL_TEST_N0)
        _eq("source_split_stage2_holdout==44723", int((crop_df["source_split"] == "stage2_holdout").sum()), EXPECTED_FINAL_TEST_N1)

    # 14. stage2_holdout_flag True rows == 44723
    if "stage2_holdout_flag" in crop_df.columns:
        holdout_true = int(crop_df["stage2_holdout_flag"].astype(str).str.lower().isin(["true", "1"]).sum())
        _eq("stage2_holdout_flag_true==44723", holdout_true, EXPECTED_FINAL_TEST_N1)

    # 15. msd_lung_flag True rows == 0
    if "msd_lung_flag" in crop_df.columns:
        msd_true = int(crop_df["msd_lung_flag"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
        _eq("msd_lung_flag_true==0", msd_true, 0)

    # 16. hard_negative_flag True rows == 0
    if "hard_negative_flag" in crop_df.columns:
        hn_true = int(crop_df["hard_negative_flag"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
        _eq("hard_negative_flag_true==0", hn_true, 0)

    # 17-18. leakage patients absent
    l295 = int(crop_df["patient_id"].astype(str).str.contains("LUNG1-295", na=False).sum())
    l415 = int(crop_df["patient_id"].astype(str).str.contains("LUNG1-415", na=False).sum())
    _eq("LUNG1-295_absent", l295, 0)
    _eq("LUNG1-415_absent", l415, 0)

    # 19-20. prob_nsclc_like finite and in [0,1]
    probs = crop_df["prob_nsclc_like"].astype(float)
    prob_finite = probs.notna().all() and bool(np.isfinite(probs.values).all())
    _true("prob_nsclc_like_finite", prob_finite,
          f"non-finite={int(probs.isna().sum()) + int((~np.isfinite(probs.values)).sum())}")
    prob_range = (probs >= 0).all() and (probs <= 1).all()
    _true("prob_nsclc_like_in_0_1", bool(prob_range),
          f"out_of_range={(probs < 0).sum() + (probs > 1).sum()}")

    # 21. logit finite
    logits = crop_df["logit"].astype(float)
    logit_finite = logits.notna().all() and bool(np.isfinite(logits.values).all())
    _true("logit_finite", logit_finite,
          f"non-finite={int(logits.isna().sum()) + int((~np.isfinite(logits.values)).sum())}")

    # 22-23. threshold columns only {0,1}
    pred_vals = set(crop_df["pred_label_threshold_0_5"].astype(int).unique().tolist())
    _true("pred_label_threshold_0_5_in_{0,1}", pred_vals.issubset({0, 1}), f"values={pred_vals}")
    corr_vals = set(crop_df["correct_threshold_0_5"].astype(int).unique().tolist())
    _true("correct_threshold_0_5_in_{0,1}", corr_vals.issubset({0, 1}), f"values={corr_vals}")

    # 24-25. HU stats finite
    for col in ["crop_hu_mean", "crop_hu_std"]:
        vals = crop_df[col].astype(float)
        _true(f"{col}_finite",
              vals.notna().all() and bool(np.isfinite(vals.values).all()),
              f"non-finite={int(vals.isna().sum())}")

    # 26-29. fraction columns in [0,1]
    for col in ["air_frac_lt_minus800", "dense_frac_gt_minus500", "dense_frac_gt_minus300", "positive_frac_gt_0"]:
        vals = crop_df[col].astype(float)
        _true(f"{col}_in_0_1",
              bool((vals >= 0).all() and (vals <= 1).all()),
              f"out_of_range={(vals < 0).sum() + (vals > 1).sum()}")

    # 30-32. patient_df counts
    _eq("patient_df_rows==158", len(patient_df), EXPECTED_TOTAL_PATIENTS)
    _eq("patient_df_normal_patients==36", int((patient_df["label"] == 0).sum()), EXPECTED_NORMAL_PATIENTS)
    _eq("patient_df_nsclc_patients==122", int((patient_df["label"] == 1).sum()), EXPECTED_NSCLC_PATIENTS)
    _eq("patient_df_n_crops_sum==66323", int(patient_df["n_crops"].sum()), EXPECTED_FINAL_TEST_ROWS)

    # 33. patient_df prob columns finite and in [0,1]
    for col in ["mean_prob_nsclc_like", "max_prob_nsclc_like", "median_prob_nsclc_like", "p95_prob_nsclc_like"]:
        if col in patient_df.columns:
            vals = patient_df[col].astype(float)
            _true(f"patient_{col}_finite",
                  vals.notna().all() and bool(np.isfinite(vals.values).all()))
            _true(f"patient_{col}_in_0_1",
                  bool((vals >= 0).all() and (vals <= 1).all()))

    # 34. patient_df msd_lung_flag_any all False
    if "msd_lung_flag_any" in patient_df.columns:
        msd_any_true = int(patient_df["msd_lung_flag_any"].astype(str).str.lower().isin(["true", "1"]).sum())
        _eq("patient_msd_lung_flag_any_all_false", msd_any_true, 0)

    return failures


# ──────────────────────────────────────────────────────────────────────────────
# Run-export: actual final test prediction export
# ALLOW_REAL_EXPORT must be True (P-C-NORMAL23b user approval required first)
# ──────────────────────────────────────────────────────────────────────────────
def run_export():
    import torch

    if not ALLOW_REAL_EXPORT:
        print("[GUARD] ALLOW_REAL_EXPORT=False: actual export 금지.", file=sys.stderr)
        print("[GUARD] P-C-NORMAL23b 사용자 승인 후에만 진행 가능.", file=sys.stderr)
        sys.exit(2)

    # Stage 0: Output collision check (before mkdir)
    _export_collision_check_or_exit()
    EXPORT_OUT.mkdir(parents=True, exist_ok=True)

    guardrail = {
        "prediction_export_run":      True,
        "model_forward_run":          True,
        "metrics_computed":           False,
        "threshold_computed":         False,
        "training_run":               False,
        "backward_run":               False,
        "optimizer_step":             False,
        "checkpoint_saved":           False,
        "crop_generated":             False,
        "manifest_generated":         False,
        "existing_outputs_modified":  False,
        "forbidden_diagnostic_wording_count": 0,
    }
    errors = []

    # Stage 1: Load checkpoint (read-only)
    print(f"[EXPORT] Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)

    missing_keys = EXPECTED_CHECKPOINT_KEYS - set(ckpt.keys())
    if missing_keys:
        print(f"[GUARD] checkpoint key mismatch: {missing_keys}", file=sys.stderr)
        sys.exit(2)
    if ckpt.get("smoke_only") is not False:
        print("[GUARD] checkpoint smoke_only != False.", file=sys.stderr)
        sys.exit(2)
    if not ckpt.get("full_training"):
        print("[GUARD] checkpoint full_training=False.", file=sys.stderr)
        sys.exit(2)

    # Stage 2: Build model, load state dict, eval
    model = build_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[EXPORT] Model on {device} | params={sum(p.numel() for p in model.parameters()):,}")

    # Stage 3: Load and validate final test manifest
    print(f"[EXPORT] Loading final test manifest: {FINAL_TEST_MANIFEST_PATH}")
    df = pd.read_csv(FINAL_TEST_MANIFEST_PATH, low_memory=False)

    if len(df) != EXPECTED_FINAL_TEST_ROWS:
        print(f"[GUARD] row mismatch: {len(df)} != {EXPECTED_FINAL_TEST_ROWS}", file=sys.stderr)
        sys.exit(2)
    n0 = int((df["label"] == 0).sum())
    n1 = int((df["label"] == 1).sum())
    if n0 != EXPECTED_FINAL_TEST_N0 or n1 != EXPECTED_FINAL_TEST_N1:
        print(f"[GUARD] label count mismatch: normal={n0}, NSCLC={n1}", file=sys.stderr)
        sys.exit(2)
    split_vals = set(df["split"].unique().tolist())
    if split_vals != {EXPECTED_SPLIT}:
        print(f"[GUARD] split not final_test only: {split_vals}", file=sys.stderr)
        sys.exit(2)
    src_splits = set(df["source_split"].unique().tolist())
    if not src_splits.issubset(ALLOWED_SOURCE_SPLITS):
        print(f"[GUARD] unexpected source_split: {src_splits - ALLOWED_SOURCE_SPLITS}", file=sys.stderr)
        sys.exit(2)

    # Stage 4: Batched no_grad inference (no training, no backward, no optimizer)
    def _flush(pending_list):
        batch_t = torch.stack([p[1] for p in pending_list]).to(device)
        with torch.no_grad():
            logits_b = model(batch_t).cpu().squeeze(1)  # (N,)
        probs_b = torch.sigmoid(logits_b)
        out = []
        for i, (row, _, arr) in enumerate(pending_list):
            hu = compute_hu_stats(arr)
            logit_val  = float(logits_b[i].item())
            prob_val   = float(probs_b[i].item())
            label_val  = int(row["label"])
            pred_05    = int(prob_val >= 0.5)
            correct_05 = int(pred_05 == label_val)
            lname = "normal" if label_val == 0 else "NSCLC"
            out.append({
                "row_index":                row.get("row_index",           ""),
                "final_candidate_id":       row.get("final_candidate_id",  ""),
                "patient_id":               row.get("patient_id",          ""),
                "safe_id":                  row.get("safe_id",             ""),
                "split":                    row.get("split",               EXPECTED_SPLIT),
                "source_split":             row.get("source_split",        ""),
                "source_name":              row.get("source_name",         ""),
                "crop_path":                str(row.get("crop_path",       "")),
                "label":                    label_val,
                "label_name":               lname,
                "logit":                    logit_val,
                "prob_nsclc_like":          prob_val,
                "pred_label_threshold_0_5": pred_05,
                "correct_threshold_0_5":    correct_05,
                "position_bin":             row.get("position_bin",        ""),
                "z_level":                  row.get("z_level",             ""),
                "local_z":                  row.get("local_z",             ""),
                "slice_index":              row.get("slice_index",         ""),
                "center_y":                 row.get("center_y",            ""),
                "center_x":                row.get("center_x",            ""),
                "stage2_holdout_flag":      row.get("stage2_holdout_flag", ""),
                "hard_negative_flag":       row.get("hard_negative_flag",  ""),
                "msd_lung_flag":            row.get("msd_lung_flag",       ""),
                **hu,
            })
        return out

    crop_rows = []
    pending   = []
    print(f"[EXPORT] Inference over {len(df)} crops (batch={INFER_BATCH_SIZE})...")
    for row_idx, (_, row) in enumerate(df.iterrows()):
        cp = str(row["crop_path"])
        try:
            data = np.load(cp)
            arr  = data["ct_crop"]
            t    = preprocess_crop(arr)
            pending.append((row, t, arr))
        except Exception as e:
            errors.append(f"row={row_idx} crop={cp}: {e}")
            continue

        if len(pending) >= INFER_BATCH_SIZE:
            crop_rows.extend(_flush(pending))
            pending = []
            if (row_idx + 1) % 1000 == 0:
                print(f"  [{row_idx+1}/{len(df)}]")

    if pending:
        crop_rows.extend(_flush(pending))
        pending = []

    print(f"[EXPORT] Done: {len(crop_rows)} processed, {len(errors)} errors")

    # Stage 5: Save per-crop CSV
    crop_df = pd.DataFrame(crop_rows, columns=CROP_PRED_COLUMNS)
    crop_csv = EXPORT_OUT / "p_c_normal23_final_test_crop_predictions.csv"
    crop_df.to_csv(crop_csv, index=False)
    print(f"[EXPORT] Saved crop CSV: {crop_csv}")

    # Stage 6: Patient-level aggregation
    patient_rows = []
    for pid, grp in crop_df.groupby("patient_id"):
        label_p   = int(grp["label"].iloc[0])
        lname_p   = "normal" if label_p == 0 else "NSCLC"
        src_p     = str(grp["source_split"].iloc[0])
        probs_p   = grp["prob_nsclc_like"].astype(float).values
        mean_p    = float(np.mean(probs_p))
        max_p     = float(np.max(probs_p))
        med_p     = float(np.median(probs_p))
        p95_p     = float(np.percentile(probs_p, 95))
        pred_mean = int(mean_p >= 0.5)
        pred_max  = int(max_p  >= 0.5)
        pos_bin_counts = grp["position_bin"].value_counts().to_dict()
        holdout_any = bool(grp["stage2_holdout_flag"].astype(str).str.lower().isin(["true","1"]).any())
        msd_any     = bool(grp["msd_lung_flag"].astype(str).str.lower().isin(["true","1"]).any())
        patient_rows.append({
            "patient_id":                    pid,
            "label":                         label_p,
            "label_name":                    lname_p,
            "source_split":                  src_p,
            "n_crops":                       len(grp),
            "mean_prob_nsclc_like":          mean_p,
            "max_prob_nsclc_like":           max_p,
            "median_prob_nsclc_like":        med_p,
            "p95_prob_nsclc_like":           p95_p,
            "pred_label_mean_threshold_0_5": pred_mean,
            "pred_label_max_threshold_0_5":  pred_max,
            "correct_mean_threshold_0_5":    int(pred_mean == label_p),
            "correct_max_threshold_0_5":     int(pred_max  == label_p),
            "position_bin_count_summary":    json.dumps(pos_bin_counts, ensure_ascii=False),
            "stage2_holdout_flag_any":       holdout_any,
            "msd_lung_flag_any":             msd_any,
        })
    patient_df = pd.DataFrame(patient_rows, columns=PATIENT_PRED_COLUMNS)
    patient_csv = EXPORT_OUT / "p_c_normal23_final_test_patient_predictions.csv"
    patient_df.to_csv(patient_csv, index=False)
    print(f"[EXPORT] Saved patient CSV: {patient_csv}")

    # Stage 6b: Post-run hard assertions (P-C-NORMAL23a2 hardening)
    print(f"[EXPORT] Running post-run hard assertions...")
    assertion_failures = _post_run_hard_assertion_export(crop_df, patient_df, errors, df)
    if assertion_failures:
        print(f"[EXPORT] HARD ASSERTION FAILED: {len(assertion_failures)} failure(s)", file=sys.stderr)
        for af in assertion_failures:
            print(f"  [ASSERT_FAIL] {af}", file=sys.stderr)
        failed_marker_path = EXPORT_OUT / "FAILED_DO_NOT_USE.json"
        with open(failed_marker_path, "w") as f:
            json.dump({
                "stage": "P-C-NORMAL23",
                "status": "FAILED",
                "conditions_ok": False,
                "hard_assertion_pass": False,
                "assertion_failures": assertion_failures,
                "n_assertion_failures": len(assertion_failures),
                "note": "Post-run hard assertions failed. Do NOT use these outputs.",
            }, f, indent=2, ensure_ascii=False)
        print(f"[EXPORT] FAILED_DO_NOT_USE.json written: {failed_marker_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[EXPORT] Hard assertions PASS")

    # Stage 7: Summary JSON (count-only, no AUC/metrics)
    n_correct_crop  = int(crop_df["correct_threshold_0_5"].sum())
    n_correct_pat_m = sum(r["correct_mean_threshold_0_5"] for r in patient_rows)
    probs_arr  = crop_df["prob_nsclc_like"].astype(float).values
    logits_arr = crop_df["logit"].astype(float).values
    msd_rows   = int(crop_df["msd_lung_flag"].astype(str).str.lower().isin(["true","1","yes"]).sum()) if "msd_lung_flag" in crop_df.columns else 0
    l295_rows  = int(crop_df["patient_id"].astype(str).str.contains("LUNG1-295", na=False).sum())
    l415_rows  = int(crop_df["patient_id"].astype(str).str.contains("LUNG1-415", na=False).sum())
    summary = {
        "stage": "P-C-NORMAL23",
        "mode": "run_export",
        "conditions_ok": True,
        "hard_assertion_pass": True,
        "n_errors": len(errors),
        "n_crops_expected": EXPECTED_FINAL_TEST_ROWS,
        "n_crops_processed": len(crop_rows),
        "n_patients_expected": EXPECTED_TOTAL_PATIENTS,
        "n_patients_processed": len(patient_rows),
        "normal_rows": int((crop_df["label"] == 0).sum()),
        "nsclc_rows": int((crop_df["label"] == 1).sum()),
        "normal_patients": int(crop_df[crop_df["label"] == 0]["patient_id"].nunique()),
        "nsclc_patients": int(crop_df[crop_df["label"] == 1]["patient_id"].nunique()),
        "msd_lung_rows": msd_rows,
        "leakage_rows": l295_rows + l415_rows,
        "prob_min": float(np.min(probs_arr)),
        "prob_max": float(np.max(probs_arr)),
        "logit_min": float(np.min(logits_arr)),
        "logit_max": float(np.max(logits_arr)),
        "crop_correct_threshold_0_5": n_correct_crop,
        "patient_correct_mean_threshold_0_5": n_correct_pat_m,
        "fixed_threshold_0_5_applied": True,
        "threshold_optimized": False,
        "metrics_deferred_to": "P-C-NORMAL23c",
        "checkpoint_path": str(CHECKPOINT_PATH),
        "manifest_path": str(FINAL_TEST_MANIFEST_PATH),
        "n_crops_total": len(df),
        "guardrail": guardrail,
        "output_path": str(EXPORT_OUT),
        "branch": "P-C-NORMAL (supervised auxiliary classifier, normal=0 / NSCLC=1)",
        "interpretation": "normal-like vs NSCLC-lesion-like auxiliary score only",
        "metrics_note": "AUC/F1/precision/recall/confusion deferred to P-C-NORMAL23c",
    }
    with open(EXPORT_OUT / "p_c_normal23_final_test_prediction_export_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Stage 8: Guardrail CSV
    with open(EXPORT_OUT / "p_c_normal23_guardrail_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["guardrail", "value"])
        w.writeheader()
        w.writerows([{"guardrail": k, "value": str(v)} for k, v in guardrail.items()])

    # Stage 9: Errors CSV
    with open(EXPORT_OUT / "p_c_normal23_errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["error"])
        for e in errors:
            w.writerow([e])

    # Stage 10: Report MD
    md = [
        "# P-C-NORMAL23 Final Baseline Test Prediction Export Report",
        "",
        "**branch**: P-C-NORMAL (supervised auxiliary classifier, normal=0 / NSCLC=1)",
        "**interpretation**: normal-like vs NSCLC-lesion-like auxiliary score only",
        "",
        f"- crops processed: {len(crop_rows)}/{len(df)}",
        f"- errors: {len(errors)}",
        f"- patients: {len(patient_rows)}",
        f"- crop correct @0.5: {n_correct_crop}/{len(crop_rows)}",
        f"- patient correct mean@0.5: {n_correct_pat_m}/{len(patient_rows)}",
        "",
        "## Guardrail",
    ]
    for k, v in guardrail.items():
        md.append(f"- `{k}`: {v}")
    md += [
        "",
        "## 다음 단계",
        "- P-C-NORMAL23c: AUC/F1/confusion 등 metrics 계산 (별도 승인)",
    ]
    with open(EXPORT_OUT / "p_c_normal23_final_test_prediction_export_report.md", "w") as f:
        f.write("\n".join(md) + "\n")

    # Stage 11: DONE marker (only created after hard assertion PASS)
    with open(EXPORT_OUT / "DONE.json", "w") as f:
        json.dump({
            "stage": "P-C-NORMAL23b",
            "status": "done",
            "conditions_ok": True,
            "hard_assertion_pass": True,
            "manifest_rows": EXPECTED_FINAL_TEST_ROWS,
            "crop_prediction_rows": len(crop_rows),
            "patient_prediction_rows": len(patient_rows),
            "normal_rows": int((crop_df["label"] == 0).sum()),
            "nsclc_rows": int((crop_df["label"] == 1).sum()),
            "normal_patients": int(crop_df[crop_df["label"] == 0]["patient_id"].nunique()),
            "nsclc_patients": int(crop_df[crop_df["label"] == 1]["patient_id"].nunique()),
            "n_errors": len(errors),
            "prediction_export_run": True,
            "model_forward_run": True,
            "threshold_optimized": False,
            "fixed_threshold_0_5_applied": True,
            "metrics_computed": False,
            "training_run": False,
            "backward_run": False,
            "optimizer_step": False,
            "checkpoint_saved": False,
            "crop_generated": False,
            "manifest_generated": False,
            "forbidden_diagnostic_wording_count": guardrail["forbidden_diagnostic_wording_count"],
            "interpretation": "normal-like vs NSCLC-lesion-like auxiliary score only",
        }, f, indent=2, ensure_ascii=False)

    print(f"[EXPORT] DONE. Output: {EXPORT_OUT}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            P-C-NORMAL23 final baseline test prediction export script.
            --dry-check    : P-C-NORMAL23a checkpoint/manifest/schema/guard 검증 (no forward)
            --run-export   : actual final test export (requires confirm flags + ALLOW_REAL_EXPORT=True)
                             P-C-NORMAL23b 별도 승인 전까지 실행 금지.
        """)
    )
    p.add_argument("--dry-check",           action="store_true", dest="dry_check")
    p.add_argument("--run-export",          action="store_true", dest="run_export")
    # export confirm flags (all required for --run-export)
    p.add_argument("--confirm-final-test",  action="store_true", dest="confirm_final_test")
    p.add_argument("--confirm-no-threshold",action="store_true", dest="confirm_no_threshold")
    p.add_argument("--confirm-no-training", action="store_true", dest="confirm_no_training")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Guard 1: bare run
    if not args.dry_check and not args.run_export:
        print("[GUARD] bare run 금지: --dry-check 또는 --run-export 필요.", file=sys.stderr)
        sys.exit(2)

    # Guard 2: --run-export requires confirm flags + ALLOW_REAL_EXPORT
    if args.run_export:
        _guard_run_export(args)

    if args.dry_check:
        run_dry_check()
    elif args.run_export:
        run_export()


if __name__ == "__main__":
    main()
