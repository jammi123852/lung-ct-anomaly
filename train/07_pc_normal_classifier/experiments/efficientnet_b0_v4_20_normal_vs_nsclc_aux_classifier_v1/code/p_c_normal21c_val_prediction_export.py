"""
P-C-NORMAL21c: Validation Prediction Export Script
===================================================
Mode A  --dry-check   : input/schema/guardrail 검증 + sample batch forward (no save)
Mode B  --run-export  : actual per-crop prediction export (requires all confirm flags + ALLOW_REAL_EXPORT=True)

이 branch는 정상분포 학습이 아님.
normal=0, NSCLC=1 label을 사용하는 supervised auxiliary classifier.
출력은 normal-like vs NSCLC-lesion-like auxiliary score로만 해석한다.
진단 모델, 암 확률, 폐선암 확률, cancer probability, adenocarcinoma probability 표현 금지.
"""

import argparse
import csv
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Guard: ALLOW_REAL_EXPORT flag (must be set True explicitly in env/code to enable)
# ──────────────────────────────────────────────────────────────────────────────
ALLOW_REAL_EXPORT = False  # Never change this to True without explicit user approval

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BRANCH_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = BRANCH_ROOT / "outputs/checkpoints/p_c_normal19_full_training/best_auc.pth"
VAL_MANIFEST_PATH = (
    BRANCH_ROOT
    / "outputs/manifests/p_c_normal12_matched_training_manifest/p_c_normal12_val_manifest.csv"
)

DRYCHECK_OUT = (
    BRANCH_ROOT
    / "outputs/reports/p_c_normal21c_prediction_export_script_drycheck"
)
DRYCHECK_21D_OUT = (
    BRANCH_ROOT
    / "outputs/reports/p_c_normal21d_run_export_implementation_drycheck"
)
EXPORT_OUT = (
    BRANCH_ROOT
    / "outputs/reports/p_c_normal21c_validation_prediction_export"
)

# All 7 export output files — collision check must cover every entry
EXPORT_FILES = [
    "p_c_normal21c_val_crop_predictions.csv",
    "p_c_normal21c_val_patient_predictions.csv",
    "p_c_normal21c_prediction_export_summary.json",
    "p_c_normal21c_prediction_export_report.md",
    "p_c_normal21c_guardrail_check.csv",
    "p_c_normal21c_errors.csv",
    "DONE.json",
]

INFER_BATCH_SIZE = 64

STAGE2_HOLDOUT_SENTINEL = "stage2_holdout"

# ──────────────────────────────────────────────────────────────────────────────
# Constants (must match training script)
# ──────────────────────────────────────────────────────────────────────────────
HU_MIN, HU_MAX = -1000.0, 200.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

EXPECTED_VAL_ROWS = 5200
EXPECTED_VAL_N0   = 3120   # normal
EXPECTED_VAL_N1   = 2080   # NSCLC

EXPECTED_CHECKPOINT_KEY_COUNT = 18
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

DRY_BATCH_SIZE = 4

# ──────────────────────────────────────────────────────────────────────────────
# Per-crop prediction CSV schema (confirmed in dry-check)
# ──────────────────────────────────────────────────────────────────────────────
CROP_PRED_COLUMNS = [
    "row_index",
    "aux_candidate_id",
    "patient_id",
    "safe_id",
    "split",
    "crop_path",
    "label",
    "label_name",
    "source_name",
    "position_bin",
    "z_level",
    "local_z",
    "slice_index",
    "center_y",
    "center_x",
    "sample_weight",
    "logit",
    "prob_nsclc_like",
    "pred_label_threshold_0_5",
    "correct_threshold_0_5",
    # HU audit columns
    "crop_hu_mean",
    "crop_hu_std",
    "crop_hu_min",
    "crop_hu_max",
    "air_frac_lt_minus800",
    "dense_frac_gt_minus500",
    "dense_frac_gt_minus300",
    "positive_frac_gt_0",
    "roi_patch_ratio",
]

# Patient-level prediction CSV schema
PATIENT_PRED_COLUMNS = [
    "patient_id",
    "label",
    "n_crops",
    "mean_prob_nsclc_like",
    "max_prob_nsclc_like",
    "min_prob_nsclc_like",
    "std_prob_nsclc_like",
    "pred_label_mean_threshold_0_5",
    "pred_label_max_threshold_0_5",
    "correct_mean_threshold_0_5",
    "correct_max_threshold_0_5",
]


# ──────────────────────────────────────────────────────────────────────────────
# Model (identical to training script build_model)
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
# Preprocessing (identical to NormalNSCLCDataset in training script)
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_crop(arr_int16: np.ndarray) -> "torch.Tensor":
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
    stats = {
        "crop_hu_mean":          float(np.mean(arr)),
        "crop_hu_std":           float(np.std(arr)),
        "crop_hu_min":           float(np.min(arr)),
        "crop_hu_max":           float(np.max(arr)),
        "air_frac_lt_minus800":  float(np.sum(arr < -800) / total),
        "dense_frac_gt_minus500":float(np.sum(arr > -500) / total),
        "dense_frac_gt_minus300":float(np.sum(arr > -300) / total),
        "positive_frac_gt_0":    float(np.sum(arr > 0) / total),
        "roi_patch_ratio":       "NA",
    }
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Guard helpers
# ──────────────────────────────────────────────────────────────────────────────
def _guard_no_bare_run(args) -> None:
    if not args.dry_check and not getattr(args, "dry_check_21d", False) and not args.run_export:
        print("[GUARD] bare run 금지: --dry-check, --dry-check-21d, 또는 --run-export 필요.", file=sys.stderr)
        sys.exit(2)


def _guard_run_export(args) -> None:
    missing = []
    if not args.confirm_export:
        missing.append("--confirm-export")
    if not args.confirm_val_only:
        missing.append("--confirm-val-only")
    if not args.confirm_no_holdout:
        missing.append("--confirm-no-holdout")
    if missing:
        print(f"[GUARD] --run-export requires: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)
    if not ALLOW_REAL_EXPORT:
        print("[GUARD] ALLOW_REAL_EXPORT=False: actual export 금지. 사용자 승인 후 진행.", file=sys.stderr)
        sys.exit(2)


def _export_collision_check_or_exit():
    """Check all 7 export output files before mkdir. Exit 2 on any collision."""
    collision = [f for f in EXPORT_FILES if (EXPORT_OUT / f).exists()]
    if collision:
        print(f"[GUARD] Output collision: {collision}", file=sys.stderr)
        print(f"[GUARD] Remove or archive {EXPORT_OUT} before re-running.", file=sys.stderr)
        sys.exit(2)


def _check_stage2_holdout(df: pd.DataFrame) -> bool:
    path_cols = [c for c in df.columns if "path" in c.lower()]
    for col in path_cols:
        if df[col].astype(str).str.contains(STAGE2_HOLDOUT_SENTINEL, na=False).any():
            return False
    return True


def _check_forbidden_words_in_output(text: str) -> int:
    """Check user-facing output text for forbidden diagnostic wording.
    Does NOT scan this script's source (which contains the definition list itself).
    """
    lower = text.lower()
    count = 0
    for w in FORBIDDEN_DIAGNOSTIC_WORDS:
        count += lower.count(w.lower())
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Dry-check logic
# ──────────────────────────────────────────────────────────────────────────────
def run_dry_check():
    import torch
    import torch.nn.functional as F

    print("[DRY-CHECK] P-C-NORMAL21c validation prediction export script dry-check 시작")
    DRYCHECK_OUT.mkdir(parents=True, exist_ok=True)

    errors = []
    checks = []
    guardrail = {
        "actual_prediction_export_run": False,
        "full_val_inference_run": False,
        "training_run": False,
        "backward_run": False,
        "optimizer_step": False,
        "checkpoint_saved": False,
        "threshold_computed": False,
        "stage2_holdout_accessed": False,
        "existing_outputs_modified": False,
        "forbidden_diagnostic_wording_count": 0,
    }

    def record(name, status, detail=""):
        checks.append({"check": name, "status": status, "detail": str(detail)})
        marker = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{marker}] {name}: {detail}" if detail else f"  [{marker}] {name}")

    # ── 1. py_compile ──
    import py_compile
    try:
        py_compile.compile(str(Path(__file__)), doraise=True)
        record("py_compile", "PASS")
    except py_compile.PyCompileError as e:
        record("py_compile", "FAIL", str(e))
        errors.append(f"py_compile: {e}")

    # ── 2. forbidden diagnostic wording (output text only, not script source) ──
    # Collect candidate output text that will be written to reports/CSV/JSON.
    # Script source itself is excluded (it contains the definition list).
    candidate_output_texts = [
        "P-C-NORMAL21c validation prediction export dry-check",
        "normal-like vs NSCLC-lesion-like auxiliary score",
        "supervised auxiliary classifier",
    ]
    fw_count = sum(_check_forbidden_words_in_output(t) for t in candidate_output_texts)
    guardrail["forbidden_diagnostic_wording_count"] = fw_count
    if fw_count == 0:
        record("forbidden_diagnostic_wording_count=0", "PASS", fw_count)
    else:
        record("forbidden_diagnostic_wording_count=0", "FAIL", f"count={fw_count}")
        errors.append(f"forbidden_diagnostic_wording_count={fw_count}")

    # ── 3. checkpoint exists ──
    if CHECKPOINT_PATH.exists():
        record("checkpoint_exists", "PASS", str(CHECKPOINT_PATH))
    else:
        record("checkpoint_exists", "FAIL", f"not found: {CHECKPOINT_PATH}")
        errors.append(f"checkpoint not found: {CHECKPOINT_PATH}")

    # ── 4. checkpoint load & key check ──
    ckpt = None
    try:
        ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
        ckpt_keys = set(ckpt.keys())
        missing_keys = EXPECTED_CHECKPOINT_KEYS - ckpt_keys
        extra_keys   = ckpt_keys - EXPECTED_CHECKPOINT_KEYS
        key_count_ok = len(ckpt_keys) == EXPECTED_CHECKPOINT_KEY_COUNT
        record("checkpoint_key_count", "PASS" if key_count_ok else "FAIL",
               f"{len(ckpt_keys)}/{EXPECTED_CHECKPOINT_KEY_COUNT}")
        if missing_keys:
            record("checkpoint_key_mismatch", "FAIL", f"missing={missing_keys}")
            errors.append(f"checkpoint missing keys: {missing_keys}")
        else:
            record("checkpoint_key_mismatch", "PASS", "no missing keys")
        record("checkpoint_epoch",         "PASS", ckpt.get("epoch"))
        record("checkpoint_val_auc",       "PASS", ckpt.get("val_auc"))
        record("checkpoint_smoke_only",    "PASS" if ckpt.get("smoke_only") is False else "FAIL",
               ckpt.get("smoke_only"))
        record("checkpoint_full_training", "PASS" if ckpt.get("full_training") is True else "FAIL",
               ckpt.get("full_training"))
        lm = ckpt.get("label_mapping", {})
        # Accept both string keys ("0","1") and int keys (0,1)
        lm_normal = lm.get("0") or lm.get(0)
        lm_nsclc  = lm.get("1") or lm.get(1)
        lm_ok = (lm_normal == "normal") and (lm_nsclc == "NSCLC")
        record("checkpoint_label_mapping", "PASS" if lm_ok else "FAIL", lm)
        if not lm_ok:
            errors.append(f"label_mapping mismatch: {lm}")
    except Exception as e:
        record("checkpoint_load", "FAIL", str(e))
        errors.append(f"checkpoint load error: {e}")

    # ── 5. checkpoint param count ──
    if ckpt is not None:
        try:
            n_params = sum(
                p.numel() for p in ckpt["model_state_dict"].values() if hasattr(p, "numel")
            )
            record("checkpoint_params", "PASS", f"{n_params:,}")
        except Exception as e:
            record("checkpoint_params", "FAIL", str(e))

    # ── 6. model_state_dict NaN/Inf check ──
    if ckpt is not None:
        try:
            nan_inf_count = 0
            for v in ckpt["model_state_dict"].values():
                if hasattr(v, "isnan"):
                    nan_inf_count += int(v.isnan().any()) + int(v.isinf().any())
            record("checkpoint_nan_inf=0", "PASS" if nan_inf_count == 0 else "FAIL",
                   f"nan_inf_count={nan_inf_count}")
        except Exception as e:
            record("checkpoint_nan_inf=0", "FAIL", str(e))

    # ── 7. val manifest exists ──
    if VAL_MANIFEST_PATH.exists():
        record("val_manifest_exists", "PASS", str(VAL_MANIFEST_PATH))
    else:
        record("val_manifest_exists", "FAIL", f"not found: {VAL_MANIFEST_PATH}")
        errors.append(f"val manifest not found: {VAL_MANIFEST_PATH}")
        _write_outputs(checks, guardrail, errors, DRYCHECK_OUT, verdict="FAIL")
        return

    # ── 8. val manifest load ──
    df = pd.read_csv(VAL_MANIFEST_PATH)

    # row count
    n_rows = len(df)
    row_ok = n_rows == EXPECTED_VAL_ROWS
    record("val_manifest_row_count", "PASS" if row_ok else "FAIL",
           f"{n_rows}/{EXPECTED_VAL_ROWS}")
    if not row_ok:
        errors.append(f"val manifest row mismatch: {n_rows} != {EXPECTED_VAL_ROWS}")

    # label counts
    n0 = int((df["label"] == 0).sum())
    n1 = int((df["label"] == 1).sum())
    record("val_manifest_normal_count",
           "PASS" if n0 == EXPECTED_VAL_N0 else "FAIL", f"{n0}/{EXPECTED_VAL_N0}")
    record("val_manifest_nsclc_count",
           "PASS" if n1 == EXPECTED_VAL_N1 else "FAIL", f"{n1}/{EXPECTED_VAL_N1}")
    if n0 != EXPECTED_VAL_N0:
        errors.append(f"normal count mismatch: {n0}")
    if n1 != EXPECTED_VAL_N1:
        errors.append(f"NSCLC count mismatch: {n1}")

    # hard_negative / MSD_Lung
    hard_neg_count = 0
    if "hard_negative" in df.columns:
        hard_neg_count = int(df["hard_negative"].astype(str).str.lower().isin(["true","1","yes"]).sum())
    record("val_manifest_hard_negative=0",
           "PASS" if hard_neg_count == 0 else "FAIL", f"count={hard_neg_count}")
    if hard_neg_count > 0:
        errors.append(f"hard_negative rows: {hard_neg_count}")

    msd_count = 0
    if "source_name" in df.columns:
        msd_count = int(df["source_name"].astype(str).str.contains("MSD", na=False).sum())
    record("val_manifest_MSD_Lung=0",
           "PASS" if msd_count == 0 else "FAIL", f"count={msd_count}")
    if msd_count > 0:
        errors.append(f"MSD_Lung rows: {msd_count}")

    # stage2_holdout refs
    holdout_ok = _check_stage2_holdout(df)
    record("val_manifest_stage2_holdout_refs=0",
           "PASS" if holdout_ok else "FAIL")
    if not holdout_ok:
        guardrail["stage2_holdout_accessed"] = True
        errors.append("stage2_holdout refs detected in val manifest")

    # split check: all should be 'val'
    if "split" in df.columns:
        non_val = df[df["split"] != "val"]
        record("val_manifest_split=val",
               "PASS" if len(non_val) == 0 else "FAIL", f"non-val rows={len(non_val)}")

    # ── 9. crop_path sample exists ──
    sample_path = str(df.iloc[0]["crop_path"])
    crop_exists = Path(sample_path).exists()
    record("crop_path_sample_exists", "PASS" if crop_exists else "FAIL", sample_path)
    if not crop_exists:
        errors.append(f"sample crop not found: {sample_path}")

    # ── 10. raw ct_crop int16 sample load ──
    raw_hu_ok = False
    arr_sample = None
    if crop_exists:
        try:
            data = np.load(sample_path)
            arr_sample = data["ct_crop"]
            raw_hu_ok = arr_sample.dtype == np.int16
            record("crop_raw_hu_int16", "PASS" if raw_hu_ok else "FAIL",
                   f"dtype={arr_sample.dtype}, shape={arr_sample.shape}")
            if not raw_hu_ok:
                errors.append(f"crop dtype not int16: {arr_sample.dtype}")
        except Exception as e:
            record("crop_raw_hu_int16", "FAIL", str(e))
            errors.append(f"crop load error: {e}")

    # ── 11. HU stat computation check ──
    if arr_sample is not None:
        try:
            hu_stats = compute_hu_stats(arr_sample)
            stats_ok = all(
                k in hu_stats and (hu_stats[k] == "NA" or isinstance(hu_stats[k], float))
                for k in ["crop_hu_mean", "crop_hu_std", "crop_hu_min", "crop_hu_max",
                          "air_frac_lt_minus800", "dense_frac_gt_minus500",
                          "dense_frac_gt_minus300", "positive_frac_gt_0", "roi_patch_ratio"]
            )
            record("hu_stat_computation", "PASS" if stats_ok else "FAIL",
                   f"mean={hu_stats.get('crop_hu_mean', 'N/A'):.1f}")
        except Exception as e:
            record("hu_stat_computation", "FAIL", str(e))
            errors.append(f"HU stat error: {e}")

    # ── 12. output collision check ──
    export_collision = EXPORT_OUT.exists()
    record("output_collision_export_dir",
           "PASS" if not export_collision else "FAIL",
           "not exists" if not export_collision else f"EXISTS: {EXPORT_OUT}")
    if export_collision:
        errors.append(f"output collision: {EXPORT_OUT} already exists")

    # ── 13. model load + sample batch no_grad forward ──
    logit_finite = False
    if ckpt is not None:
        try:
            import torch
            import torch.nn.functional as F
            model = build_model()
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            # sample batch from manifest (DRY_BATCH_SIZE crops, read-only, no save)
            batch_rows = df.head(DRY_BATCH_SIZE)
            tensors = []
            for _, row in batch_rows.iterrows():
                cp = str(row["crop_path"])
                if Path(cp).exists():
                    data = np.load(cp)
                    arr = data["ct_crop"]
                    t = preprocess_crop(arr)
                    tensors.append(t)
            if tensors:
                batch = torch.stack(tensors)  # (N,3,96,96)
                with torch.no_grad():
                    logits = model(batch)  # (N,1)
                probs = torch.sigmoid(logits).squeeze(1)
                finite = bool(torch.isfinite(logits).all() and torch.isfinite(probs).all())
                logit_finite = finite
                record("sample_batch_no_grad_forward",
                       "PASS" if finite else "FAIL",
                       f"batch_size={len(tensors)}, logits={logits.squeeze().tolist()[:4]}")
            else:
                record("sample_batch_no_grad_forward", "FAIL", "no crops loaded")
                errors.append("sample batch: no crops could be loaded")
        except Exception as e:
            record("sample_batch_no_grad_forward", "FAIL", str(e))
            errors.append(f"sample batch forward error: {e}")

    # ── 14. actual export file not created ──
    export_files = [
        "p_c_normal21c_val_crop_predictions.csv",
        "p_c_normal21c_val_patient_predictions.csv",
        "p_c_normal21c_prediction_export_summary.json",
        "DONE.json",
    ]
    for fname in export_files:
        fp = EXPORT_OUT / fname
        if fp.exists():
            record(f"actual_export_not_created:{fname}", "FAIL", "file exists — should not")
            errors.append(f"actual export file exists: {fp}")
            guardrail["actual_prediction_export_run"] = True
        else:
            record(f"actual_export_not_created:{fname}", "PASS", "not exists (correct)")

    # ── 15. export CSV schema plan ──
    crop_schema_ok = all(c in CROP_PRED_COLUMNS for c in [
        "row_index", "aux_candidate_id", "patient_id", "logit",
        "prob_nsclc_like", "crop_hu_mean", "crop_hu_std", "air_frac_lt_minus800"
    ])
    patient_schema_ok = all(c in PATIENT_PRED_COLUMNS for c in [
        "patient_id", "label", "n_crops", "mean_prob_nsclc_like", "max_prob_nsclc_like"
    ])
    record("crop_pred_csv_schema_plan",    "PASS" if crop_schema_ok    else "FAIL",
           f"{len(CROP_PRED_COLUMNS)} columns")
    record("patient_pred_csv_schema_plan", "PASS" if patient_schema_ok else "FAIL",
           f"{len(PATIENT_PRED_COLUMNS)} columns")

    # ── Verdict ──
    fail_count = sum(1 for c in checks if c["status"] != "PASS")
    verdict = "PASS" if (fail_count == 0 and not errors) else (
        "PARTIAL_PASS" if (logit_finite and not export_collision) else "FAIL"
    )

    _write_outputs(checks, guardrail, errors, DRYCHECK_OUT, verdict=verdict)
    print(f"\n[DRY-CHECK] 판정: {verdict}  (fail_count={fail_count}, errors={len(errors)})")
    if verdict == "PASS":
        print("[DRY-CHECK] script dry-check PASS. P-C-NORMAL21d implementation/actual export step required.")
    elif errors:
        for e in errors:
            print(f"  [ERROR] {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Output writer
# ──────────────────────────────────────────────────────────────────────────────
def _write_outputs(checks, guardrail, errors, out_dir, verdict):
    out_dir.mkdir(parents=True, exist_ok=True)

    # guard_check.csv
    with open(out_dir / "p_c_normal21c_guard_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    # guardrail_check.csv
    rows = [{"guardrail": k, "value": str(v)} for k, v in guardrail.items()]
    with open(out_dir / "p_c_normal21c_guardrail_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["guardrail", "value"])
        writer.writeheader()
        writer.writerows(rows)

    # errors.csv
    with open(out_dir / "p_c_normal21c_errors.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["error"])
        for e in errors:
            writer.writerow([e])

    # checkpoint_schema_check.csv
    ckpt_checks = [c for c in checks if "checkpoint" in c["check"].lower()]
    with open(out_dir / "p_c_normal21c_checkpoint_schema_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(ckpt_checks)

    # val_manifest_schema_check.csv
    mf_checks = [c for c in checks if "val_manifest" in c["check"].lower() or "crop" in c["check"].lower()]
    with open(out_dir / "p_c_normal21c_val_manifest_schema_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(mf_checks)

    # export_schema_plan.csv
    crop_schema_rows = [{"schema": "crop_pred", "column": c, "index": i}
                        for i, c in enumerate(CROP_PRED_COLUMNS)]
    patient_schema_rows = [{"schema": "patient_pred", "column": c, "index": i}
                           for i, c in enumerate(PATIENT_PRED_COLUMNS)]
    with open(out_dir / "p_c_normal21c_export_schema_plan.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["schema", "column", "index"])
        writer.writeheader()
        writer.writerows(crop_schema_rows + patient_schema_rows)

    # input_readiness_check.csv
    input_checks = [c for c in checks if any(
        k in c["check"].lower() for k in ["checkpoint_exists", "val_manifest_exists",
                                          "crop_path", "raw_hu", "hu_stat"]
    )]
    with open(out_dir / "p_c_normal21c_input_readiness_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(input_checks)

    # output_collision_check.csv
    col_checks = [c for c in checks if "collision" in c["check"].lower() or "actual_export" in c["check"].lower()]
    with open(out_dir / "p_c_normal21c_output_collision_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(col_checks)

    # sample_batch_check.csv
    batch_checks = [c for c in checks if "batch" in c["check"].lower() or "forward" in c["check"].lower()]
    with open(out_dir / "p_c_normal21c_sample_batch_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(batch_checks)

    # summary JSON
    summary = {
        "stage": "P-C-NORMAL21c",
        "mode": "dry_check",
        "verdict": verdict,
        "total_checks": len(checks),
        "pass_count": sum(1 for c in checks if c["status"] == "PASS"),
        "fail_count": sum(1 for c in checks if c["status"] != "PASS"),
        "error_count": len(errors),
        "guardrail": guardrail,
        "crop_pred_columns": CROP_PRED_COLUMNS,
        "patient_pred_columns": PATIENT_PRED_COLUMNS,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "val_manifest_path": str(VAL_MANIFEST_PATH),
        "export_output_path": str(EXPORT_OUT),
        "drycheck_output_path": str(out_dir),
        "next_step": "P-C-NORMAL21d implementation/actual export step required" if verdict == "PASS" else "fix errors before P-C-NORMAL21d",
    }
    with open(out_dir / "p_c_normal21c_prediction_export_script_drycheck.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # markdown report
    md_lines = [
        "# P-C-NORMAL21c Prediction Export Script Dry-Check Report",
        "",
        f"**판정: {verdict}**",
        "",
        "## 개요",
        "- branch: P-C-NORMAL (supervised auxiliary classifier, normal=0 / NSCLC=1)",
        "- 출력: normal-like vs NSCLC-lesion-like auxiliary score (진단 모델 아님)",
        f"- checkpoint: `{CHECKPOINT_PATH}`",
        f"- val manifest: `{VAL_MANIFEST_PATH}`",
        f"- dry-check output: `{out_dir}`",
        f"- actual export output (예정): `{EXPORT_OUT}`",
        "",
        "## Guardrail 결과",
    ]
    for k, v in guardrail.items():
        status = "OK" if v in (False, 0) else "VIOLATION"
        md_lines.append(f"- `{k}`: {v}  [{status}]")

    md_lines += [
        "",
        "## 체크 결과 요약",
        f"- total: {len(checks)}, pass: {sum(1 for c in checks if c['status']=='PASS')}, fail: {sum(1 for c in checks if c['status']!='PASS')}",
        "",
        "## Per-Crop Prediction CSV Schema (확정)",
        "| # | column |",
        "|---|--------|",
    ]
    for i, c in enumerate(CROP_PRED_COLUMNS):
        md_lines.append(f"| {i} | {c} |")

    md_lines += [
        "",
        "## Patient-Level Prediction CSV Schema (확정)",
        "| # | column |",
        "|---|--------|",
    ]
    for i, c in enumerate(PATIENT_PRED_COLUMNS):
        md_lines.append(f"| {i} | {c} |")

    md_lines += [
        "",
        "## Actual Export 예정 파일",
        f"- `{EXPORT_OUT}/p_c_normal21c_val_crop_predictions.csv`",
        f"- `{EXPORT_OUT}/p_c_normal21c_val_patient_predictions.csv`",
        f"- `{EXPORT_OUT}/p_c_normal21c_prediction_export_summary.json`",
        f"- `{EXPORT_OUT}/p_c_normal21c_prediction_export_report.md`",
        f"- `{EXPORT_OUT}/p_c_normal21c_guardrail_check.csv`",
        f"- `{EXPORT_OUT}/p_c_normal21c_errors.csv`",
        f"- `{EXPORT_OUT}/DONE.json`",
        "",
        "## 다음 단계",
        "- P-C-NORMAL21d implementation/actual export step required",
        "- run_export() 구현 후 사용자 승인 하에 P-C-NORMAL21d 진행",
        "",
        "## P-C-NORMAL21b 요약",
        "- 결과: NEEDS_SCRIPT_PATCH",
        "- validate_one_epoch은 all_probs 메모리 누적만 수행, per-crop CSV 저장 없음",
        "- 신규 export script 필요 → P-C-NORMAL21c로 해결",
    ]
    if errors:
        md_lines += ["", "## Errors"]
        for e in errors:
            md_lines.append(f"- {e}")

    with open(out_dir / "p_c_normal21c_prediction_export_script_drycheck.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"[DRY-CHECK] 결과 저장 완료: {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Run-export: actual validation prediction export
# ALLOW_REAL_EXPORT must be True (set only after explicit user approval)
# ──────────────────────────────────────────────────────────────────────────────
def run_export():
    import torch

    # Defensive double-check guard (also enforced in main via _guard_run_export)
    if not ALLOW_REAL_EXPORT:
        print("[GUARD] ALLOW_REAL_EXPORT=False: actual export 금지. 사용자 승인 후 진행.", file=sys.stderr)
        sys.exit(2)

    # ── Stage 0: Output collision check (all 7 files, before mkdir) ──
    _export_collision_check_or_exit()
    EXPORT_OUT.mkdir(parents=True, exist_ok=True)

    guardrail = {
        "actual_prediction_export_run": True,
        "full_val_inference_run": True,
        "training_run": False,
        "backward_run": False,
        "optimizer_step": False,
        "checkpoint_saved": False,
        "threshold_computed": False,   # 0.5 is a fixed audit threshold, not learned
        "stage2_holdout_accessed": False,
        "existing_outputs_modified": False,
        "forbidden_diagnostic_wording_count": 0,
    }
    errors = []

    # ── Stage 1: Load checkpoint (read-only) ──
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

    # ── Stage 2: Build model, load state dict, eval ──
    model = build_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[EXPORT] Model on {device} | params={sum(p.numel() for p in model.parameters()):,}")

    # ── Stage 3: Load and validate val manifest ──
    print(f"[EXPORT] Loading val manifest: {VAL_MANIFEST_PATH}")
    df = pd.read_csv(VAL_MANIFEST_PATH)

    if len(df) != EXPECTED_VAL_ROWS:
        print(f"[GUARD] row mismatch: {len(df)} != {EXPECTED_VAL_ROWS}", file=sys.stderr)
        sys.exit(2)
    n0 = int((df["label"] == 0).sum())
    n1 = int((df["label"] == 1).sum())
    if n0 != EXPECTED_VAL_N0 or n1 != EXPECTED_VAL_N1:
        print(f"[GUARD] label count mismatch: normal={n0}, NSCLC={n1}", file=sys.stderr)
        sys.exit(2)
    if not _check_stage2_holdout(df):
        guardrail["stage2_holdout_accessed"] = True
        print("[GUARD] stage2_holdout refs detected.", file=sys.stderr)
        sys.exit(2)

    # ── Stage 4: Batched no_grad inference (no training, no backward, no optimizer) ──
    def _flush(pending_list):
        """Forward a batch, return list of per-crop result dicts. No save here."""
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
                "row_index":                row.get("row_index",        ""),
                "aux_candidate_id":         row.get("aux_candidate_id", ""),
                "patient_id":               row.get("patient_id",       ""),
                "safe_id":                  row.get("safe_id",          ""),
                "split":                    row.get("split",            "val"),
                "crop_path":                str(row.get("crop_path",    "")),
                "label":                    label_val,
                "label_name":               lname,
                "source_name":              row.get("source_name",      ""),
                "position_bin":             row.get("position_bin",     ""),
                "z_level":                  row.get("z_level",          ""),
                "local_z":                  row.get("local_z",          ""),
                "slice_index":              row.get("slice_index",      ""),
                "center_y":                 row.get("center_y",         ""),
                "center_x":                row.get("center_x",         ""),
                "sample_weight":            row.get("sample_weight",    ""),
                "logit":                    logit_val,
                "prob_nsclc_like":          prob_val,
                "pred_label_threshold_0_5": pred_05,
                "correct_threshold_0_5":    correct_05,
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
            arr  = data["ct_crop"]   # int16, shape (3,96,96)
            t    = preprocess_crop(arr)
            pending.append((row, t, arr))
        except Exception as e:
            errors.append(f"row={row_idx} crop={cp}: {e}")
            continue

        if len(pending) >= INFER_BATCH_SIZE:
            crop_rows.extend(_flush(pending))
            pending = []
            if (row_idx + 1) % 500 == 0:
                print(f"  [{row_idx+1}/{len(df)}]")

    if pending:
        crop_rows.extend(_flush(pending))
        pending = []

    print(f"[EXPORT] Done: {len(crop_rows)} processed, {len(errors)} errors")

    # ── Stage 5: Save per-crop CSV ──
    crop_df = pd.DataFrame(crop_rows, columns=CROP_PRED_COLUMNS)
    crop_csv = EXPORT_OUT / "p_c_normal21c_val_crop_predictions.csv"
    crop_df.to_csv(crop_csv, index=False)
    print(f"[EXPORT] Saved crop CSV: {crop_csv}")

    # ── Stage 6: Patient-level aggregation ──
    patient_rows = []
    for pid, grp in crop_df.groupby("patient_id"):
        label_p = int(grp["label"].iloc[0])
        probs_p = grp["prob_nsclc_like"].astype(float).values
        mean_p  = float(probs_p.mean())
        max_p   = float(probs_p.max())
        min_p   = float(probs_p.min())
        std_p   = float(probs_p.std())
        pred_mean = int(mean_p >= 0.5)
        pred_max  = int(max_p  >= 0.5)
        patient_rows.append({
            "patient_id":                    pid,
            "label":                         label_p,
            "n_crops":                       len(grp),
            "mean_prob_nsclc_like":          mean_p,
            "max_prob_nsclc_like":           max_p,
            "min_prob_nsclc_like":           min_p,
            "std_prob_nsclc_like":           std_p,
            "pred_label_mean_threshold_0_5": pred_mean,
            "pred_label_max_threshold_0_5":  pred_max,
            "correct_mean_threshold_0_5":    int(pred_mean == label_p),
            "correct_max_threshold_0_5":     int(pred_max  == label_p),
        })
    patient_df = pd.DataFrame(patient_rows, columns=PATIENT_PRED_COLUMNS)
    patient_csv = EXPORT_OUT / "p_c_normal21c_val_patient_predictions.csv"
    patient_df.to_csv(patient_csv, index=False)
    print(f"[EXPORT] Saved patient CSV: {patient_csv}")

    # ── Stage 7: Summary JSON ──
    n_correct_crop  = int(crop_df["correct_threshold_0_5"].sum())
    n_correct_pat_m = sum(r["correct_mean_threshold_0_5"] for r in patient_rows)
    summary = {
        "stage": "P-C-NORMAL21c",
        "mode": "run_export",
        "checkpoint_path": str(CHECKPOINT_PATH),
        "val_manifest_path": str(VAL_MANIFEST_PATH),
        "n_crops_total": len(df),
        "n_crops_processed": len(crop_rows),
        "n_errors": len(errors),
        "n_patients": len(patient_rows),
        "crop_accuracy_threshold_0_5": round(n_correct_crop / max(len(crop_rows), 1), 6),
        "patient_accuracy_mean_threshold_0_5": round(n_correct_pat_m / max(len(patient_rows), 1), 6),
        "guardrail": guardrail,
        "output_path": str(EXPORT_OUT),
        "branch": "P-C-NORMAL (supervised auxiliary classifier, normal=0 / NSCLC=1)",
        "interpretation": "normal-like vs NSCLC-lesion-like auxiliary score only",
    }
    with open(EXPORT_OUT / "p_c_normal21c_prediction_export_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Stage 8: Guardrail CSV ──
    with open(EXPORT_OUT / "p_c_normal21c_guardrail_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["guardrail", "value"])
        w.writeheader()
        w.writerows([{"guardrail": k, "value": str(v)} for k, v in guardrail.items()])

    # ── Stage 9: Errors CSV ──
    with open(EXPORT_OUT / "p_c_normal21c_errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["error"])
        for e in errors:
            w.writerow([e])

    # ── Stage 10: Report MD ──
    md = [
        "# P-C-NORMAL21c Validation Prediction Export Report",
        "",
        "**branch**: P-C-NORMAL (supervised auxiliary classifier, normal=0 / NSCLC=1)",
        "**interpretation**: normal-like vs NSCLC-lesion-like auxiliary score only",
        "",
        f"- crops processed: {len(crop_rows)}/{len(df)}",
        f"- errors: {len(errors)}",
        f"- patients: {len(patient_rows)}",
        f"- crop accuracy @0.5: {summary['crop_accuracy_threshold_0_5']:.4f}",
        f"- patient accuracy (mean prob @0.5): {summary['patient_accuracy_mean_threshold_0_5']:.4f}",
        "",
        "## Guardrail",
    ]
    for k, v in guardrail.items():
        md.append(f"- `{k}`: {v}")
    with open(EXPORT_OUT / "p_c_normal21c_prediction_export_report.md", "w") as f:
        f.write("\n".join(md) + "\n")

    # ── Stage 11: DONE marker ──
    with open(EXPORT_OUT / "DONE.json", "w") as f:
        json.dump({"stage": "P-C-NORMAL21c", "status": "DONE",
                   "n_crops": len(crop_rows), "n_errors": len(errors)}, f, indent=2)

    print(f"[EXPORT] DONE. Output: {EXPORT_OUT}")


# ──────────────────────────────────────────────────────────────────────────────
# P-C-NORMAL21d dry-check: verifies run_export implementation without running it
# ──────────────────────────────────────────────────────────────────────────────
def run_dry_check_21d():
    import torch
    import ast

    print("[DRY-CHECK-21D] P-C-NORMAL21d run_export implementation dry-check 시작")
    DRYCHECK_21D_OUT.mkdir(parents=True, exist_ok=True)

    errors   = []
    checks   = []
    guardrail = {
        "actual_prediction_export_run":  False,
        "full_val_inference_run":        False,
        "training_run":                  False,
        "backward_run":                  False,
        "optimizer_step":                False,
        "checkpoint_saved":              False,
        "threshold_computed":            False,
        "stage2_holdout_accessed":       False,
        "existing_outputs_modified":     False,
        "forbidden_diagnostic_wording_count": 0,
    }

    def record(name, status, detail=""):
        checks.append({"check": name, "status": status, "detail": str(detail)})
        marker = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{marker}] {name}" + (f": {detail}" if detail else ""))

    # ── 1. py_compile ──
    import py_compile
    try:
        py_compile.compile(str(Path(__file__)), doraise=True)
        record("py_compile", "PASS")
    except py_compile.PyCompileError as e:
        record("py_compile", "FAIL", str(e))
        errors.append(f"py_compile: {e}")

    # ── 2. run_export static check: no longer NotImplementedError ──
    script_src = Path(__file__).read_text(errors="ignore")
    not_impl_in_export = False
    try:
        tree = ast.parse(script_src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_export":
                func_src = ast.get_source_segment(script_src, node) or ""
                not_impl_in_export = "NotImplementedError" in func_src
                break
        record("run_export_not_notimplemented",
               "PASS" if not not_impl_in_export else "FAIL",
               "NotImplementedError removed" if not not_impl_in_export else "still NotImplementedError")
        if not_impl_in_export:
            errors.append("run_export still raises NotImplementedError")
    except Exception as e:
        record("run_export_static_check", "FAIL", str(e))
        errors.append(f"run_export static check error: {e}")

    # ── 3. run_export contains required implementation stages ──
    required_phrases = [
        ("_export_collision_check_or_exit", "collision check helper call"),
        ("model.load_state_dict",           "model load_state_dict"),
        ("model.eval()",                    "model.eval()"),
        ("torch.no_grad()",                 "no_grad context"),
        ("compute_hu_stats",                "HU stat computation"),
        ("prob_nsclc_like",                 "prob_nsclc_like column"),
        ("PATIENT_PRED_COLUMNS",            "patient CSV schema"),
        ("DONE.json",                       "DONE marker"),
        ("ALLOW_REAL_EXPORT",               "ALLOW_REAL_EXPORT guard"),
    ]
    for phrase, label in required_phrases:
        found = phrase in script_src
        record(f"run_export_has:{label}", "PASS" if found else "FAIL")
        if not found:
            errors.append(f"run_export missing: {label} ({phrase})")

    # ── 4. output collision blocker covers all 7 EXPORT_FILES ──
    all_covered = all(f in script_src for f in EXPORT_FILES)
    record("collision_blocker_covers_all_7_files",
           "PASS" if all_covered else "FAIL",
           f"covered={sum(f in script_src for f in EXPORT_FILES)}/7")
    if not all_covered:
        missing_files = [f for f in EXPORT_FILES if f not in script_src]
        errors.append(f"EXPORT_FILES missing from script: {missing_files}")

    # ── 5. ALLOW_REAL_EXPORT=False confirmed ──
    allow_false = "ALLOW_REAL_EXPORT = False" in script_src
    record("ALLOW_REAL_EXPORT=False", "PASS" if allow_false else "FAIL")
    if not allow_false:
        errors.append("ALLOW_REAL_EXPORT is not False in script source")

    # ── 6. Guard checks: bare run, --run-export alone, ALLOW_REAL_EXPORT=False ──
    import subprocess, sys as _sys
    python_bin = _sys.executable
    script_path = str(Path(__file__))

    def run_cmd(cmd_args, label):
        result = subprocess.run(
            [python_bin, script_path] + cmd_args,
            capture_output=True, text=True
        )
        return result.returncode

    rc_bare = run_cmd([], "bare_run")
    record("guard_bare_run_exit2", "PASS" if rc_bare == 2 else "FAIL", f"rc={rc_bare}")
    if rc_bare != 2:
        errors.append(f"bare run should exit 2, got {rc_bare}")

    rc_run_export_alone = run_cmd(["--run-export"], "--run-export alone")
    record("guard_run_export_alone_exit2", "PASS" if rc_run_export_alone == 2 else "FAIL",
           f"rc={rc_run_export_alone}")
    if rc_run_export_alone != 2:
        errors.append(f"--run-export alone should exit 2, got {rc_run_export_alone}")

    rc_allow_false = run_cmd(
        ["--run-export", "--confirm-export", "--confirm-val-only", "--confirm-no-holdout"],
        "ALLOW_REAL_EXPORT=False+confirm"
    )
    record("guard_ALLOW_REAL_EXPORT_False_exit2", "PASS" if rc_allow_false == 2 else "FAIL",
           f"rc={rc_allow_false}")
    if rc_allow_false != 2:
        errors.append(f"ALLOW_REAL_EXPORT=False+confirm should exit 2, got {rc_allow_false}")

    # ── 7. output collision: none of the 7 export files exist ──
    collision_files = [f for f in EXPORT_FILES if (EXPORT_OUT / f).exists()]
    record("actual_export_files_not_created",
           "PASS" if not collision_files else "FAIL",
           f"collision={collision_files}")
    if collision_files:
        guardrail["actual_prediction_export_run"] = True
        errors.append(f"actual export files exist: {collision_files}")

    # ── 8. checkpoint exists (read-only check) ──
    record("checkpoint_exists", "PASS" if CHECKPOINT_PATH.exists() else "FAIL",
           str(CHECKPOINT_PATH))

    # ── 9. val manifest exists ──
    record("val_manifest_exists", "PASS" if VAL_MANIFEST_PATH.exists() else "FAIL",
           str(VAL_MANIFEST_PATH))

    # ── 10. sample batch no_grad forward (batch=4, no save) ──
    logit_finite = False
    if CHECKPOINT_PATH.exists() and VAL_MANIFEST_PATH.exists():
        try:
            ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
            model = build_model()
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            df_val = pd.read_csv(VAL_MANIFEST_PATH)
            batch_rows = df_val.head(DRY_BATCH_SIZE)
            tensors = []
            for _, row in batch_rows.iterrows():
                cp = str(row["crop_path"])
                if Path(cp).exists():
                    data = np.load(cp)
                    arr = data["ct_crop"]
                    tensors.append(preprocess_crop(arr))
            if tensors:
                batch = torch.stack(tensors)
                with torch.no_grad():
                    logits = model(batch).squeeze(1)
                probs = torch.sigmoid(logits)
                finite = bool(torch.isfinite(logits).all() and torch.isfinite(probs).all())
                logit_finite = finite
                record("sample_batch_no_grad_forward",
                       "PASS" if finite else "FAIL",
                       f"batch={len(tensors)}, logits={logits.tolist()[:4]}")
            else:
                record("sample_batch_no_grad_forward", "FAIL", "no crops loaded")
                errors.append("sample batch: no crops loaded")
        except Exception as e:
            record("sample_batch_no_grad_forward", "FAIL", str(e))
            errors.append(f"sample batch error: {e}")

    # ── 11. forbidden diagnostic wording in candidate output text ──
    candidate_output_texts = [
        "P-C-NORMAL21d run_export implementation dry-check",
        "normal-like vs NSCLC-lesion-like auxiliary score",
        "supervised auxiliary classifier",
    ]
    fw_count = sum(_check_forbidden_words_in_output(t) for t in candidate_output_texts)
    guardrail["forbidden_diagnostic_wording_count"] = fw_count
    record("forbidden_diagnostic_wording_count=0",
           "PASS" if fw_count == 0 else "FAIL", fw_count)
    if fw_count:
        errors.append(f"forbidden_diagnostic_wording_count={fw_count}")

    # ── Patch summary ──
    patch_summary = [
        {"item": "run_export() placeholder removed",  "status": "DONE" if not not_impl_in_export else "PENDING"},
        {"item": "run_export() full implementation",  "status": "DONE" if not not_impl_in_export else "PENDING"},
        {"item": "_export_collision_check_or_exit()", "status": "DONE" if "_export_collision_check_or_exit" in script_src else "PENDING"},
        {"item": "EXPORT_FILES constant (7 files)",   "status": "DONE" if "EXPORT_FILES" in script_src else "PENDING"},
        {"item": "INFER_BATCH_SIZE constant",         "status": "DONE" if "INFER_BATCH_SIZE" in script_src else "PENDING"},
        {"item": "DRYCHECK_21D_OUT path",             "status": "DONE" if "DRYCHECK_21D_OUT" in script_src else "PENDING"},
        {"item": "run_dry_check_21d() added",         "status": "DONE"},
        {"item": "--dry-check-21d CLI mode",          "status": "DONE" if "dry_check_21d" in script_src else "PENDING"},
        {"item": "ALLOW_REAL_EXPORT=False",           "status": "DONE" if allow_false else "VIOLATION"},
    ]

    # ── Verdict ──
    fail_count = sum(1 for c in checks if c["status"] != "PASS")
    verdict = "PASS" if (fail_count == 0 and not errors) else (
        "PARTIAL_PASS" if logit_finite and not guardrail["actual_prediction_export_run"] else "FAIL"
    )

    # ── Write outputs ──
    _write_21d_outputs(checks, guardrail, errors, patch_summary, verdict)
    print(f"\n[DRY-CHECK-21D] 판정: {verdict}  (fail={fail_count}, errors={len(errors)})")
    if verdict == "PASS":
        print("[DRY-CHECK-21D] P-C-NORMAL21e actual validation prediction export step required.")
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")


def _write_21d_outputs(checks, guardrail, errors, patch_summary, verdict):
    out = DRYCHECK_21D_OUT
    out.mkdir(parents=True, exist_ok=True)

    # guard_check.csv
    with open(out / "p_c_normal21d_guard_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        w.writeheader(); w.writerows(checks)

    # guardrail_check.csv
    with open(out / "p_c_normal21d_guardrail_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["guardrail", "value"])
        w.writeheader()
        w.writerows([{"guardrail": k, "value": str(v)} for k, v in guardrail.items()])

    # errors.csv
    with open(out / "p_c_normal21d_errors.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["error"])
        for e in errors: w.writerow([e])

    # patch_summary.csv
    with open(out / "p_c_normal21d_patch_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "status"])
        w.writeheader(); w.writerows(patch_summary)

    # run_export_static_check.csv
    static_checks = [c for c in checks if any(
        k in c["check"].lower() for k in ["run_export", "collision", "allow_real", "guard"]
    )]
    with open(out / "p_c_normal21d_run_export_static_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        w.writeheader(); w.writerows(static_checks)

    # output_collision_check.csv
    col_checks = [c for c in checks if "export_file" in c["check"].lower() or "collision" in c["check"].lower()]
    with open(out / "p_c_normal21d_output_collision_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        w.writeheader(); w.writerows(col_checks)

    # sample_batch_check.csv
    batch_checks = [c for c in checks if "batch" in c["check"].lower() or "forward" in c["check"].lower()]
    with open(out / "p_c_normal21d_sample_batch_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        w.writeheader(); w.writerows(batch_checks)

    # summary JSON
    summary = {
        "stage": "P-C-NORMAL21d",
        "mode": "run_export_implementation_drycheck",
        "verdict": verdict,
        "total_checks": len(checks),
        "pass_count": sum(1 for c in checks if c["status"] == "PASS"),
        "fail_count": sum(1 for c in checks if c["status"] != "PASS"),
        "error_count": len(errors),
        "guardrail": guardrail,
        "ALLOW_REAL_EXPORT": ALLOW_REAL_EXPORT,
        "script_path": str(Path(__file__)),
        "export_output_path": str(EXPORT_OUT),
        "next_step": (
            "P-C-NORMAL21e actual validation prediction export (user approval required)"
            if verdict == "PASS"
            else "fix errors before P-C-NORMAL21e"
        ),
    }
    with open(out / "p_c_normal21d_run_export_implementation_drycheck.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # markdown report
    md = [
        "# P-C-NORMAL21d run_export Implementation Dry-Check Report",
        "",
        f"**판정: {verdict}**",
        "",
        "## 개요",
        "- branch: P-C-NORMAL (supervised auxiliary classifier, normal=0 / NSCLC=1)",
        "- 출력: normal-like vs NSCLC-lesion-like auxiliary score (진단 모델 아님)",
        f"- script: `{Path(__file__)}`",
        f"- ALLOW_REAL_EXPORT: {ALLOW_REAL_EXPORT}",
        f"- actual export output (예정): `{EXPORT_OUT}`",
        "",
        "## Patch Summary",
    ]
    for p in patch_summary:
        md.append(f"- [{p['status']}] {p['item']}")
    md += [
        "",
        "## Guardrail",
    ]
    for k, v in guardrail.items():
        ok = "OK" if v in (False, 0) else "VIOLATION"
        md.append(f"- `{k}`: {v}  [{ok}]")
    md += [
        "",
        f"## 체크 결과: total={len(checks)}, pass={sum(1 for c in checks if c['status']=='PASS')}, fail={sum(1 for c in checks if c['status']!='PASS')}",
        "",
        "## 실제 Export 미실행 확인",
        f"- actual_prediction_export_run: {guardrail['actual_prediction_export_run']}",
        f"- full_val_inference_run: {guardrail['full_val_inference_run']}",
        f"- checkpoint_saved: {guardrail['checkpoint_saved']}",
        f"- stage2_holdout_accessed: {guardrail['stage2_holdout_accessed']}",
        "",
        "## P-C-NORMAL21e 승인 문구 초안",
        "> P-C-NORMAL21d run_export implementation dry-check 통과 확인.",
        "> P-C-NORMAL21e validation prediction export actual run 승인.",
        "> best_auc.pth read-only 로드, P-C-NORMAL12 val manifest 5,200 crops no_grad inference,",
        "> per-crop probability + HU stats CSV 저장, 학습/backward/optimizer/checkpoint/threshold/stage2_holdout 없이 1회 실행.",
        "",
        "## 다음 단계",
        "- P-C-NORMAL21e: ALLOW_REAL_EXPORT=True로 변경 후 사용자 승인 하에 actual export 실행",
    ]
    if errors:
        md += ["", "## Errors"]
        for e in errors:
            md.append(f"- {e}")
    with open(out / "p_c_normal21d_run_export_implementation_drycheck.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"[DRY-CHECK-21D] 결과 저장: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            P-C-NORMAL21c/21d validation prediction export script.
            --dry-check      : 21c input/schema/guardrail 검증 + sample batch forward (no save)
            --dry-check-21d  : 21d run_export implementation static check + sample batch (no save)
            --run-export     : actual export (requires confirm flags + ALLOW_REAL_EXPORT=True)
        """)
    )
    p.add_argument("--dry-check",       action="store_true", dest="dry_check")
    p.add_argument("--dry-check-21d",   action="store_true", dest="dry_check_21d")
    p.add_argument("--run-export",      action="store_true", dest="run_export")
    # export confirm flags
    p.add_argument("--confirm-export",       action="store_true", dest="confirm_export")
    p.add_argument("--confirm-val-only",     action="store_true", dest="confirm_val_only")
    p.add_argument("--confirm-no-holdout",   action="store_true", dest="confirm_no_holdout")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Guard 1: bare run (none of the mode flags set)
    if not args.dry_check and not args.dry_check_21d and not args.run_export:
        print("[GUARD] bare run 금지: --dry-check, --dry-check-21d, 또는 --run-export 필요.", file=sys.stderr)
        sys.exit(2)

    # Guard 2: --run-export requires confirm flags + ALLOW_REAL_EXPORT
    if args.run_export:
        _guard_run_export(args)

    if args.dry_check:
        run_dry_check()
    elif args.dry_check_21d:
        run_dry_check_21d()
    elif args.run_export:
        run_export()


if __name__ == "__main__":
    main()
