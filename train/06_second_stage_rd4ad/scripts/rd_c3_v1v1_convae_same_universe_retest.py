"""
RD-C3: v1/v1 ConvAutoencoder2p5D Same-Universe Retest

목적:
  RD-C2와 동일한 EfficientNet-B0 + refined_roi_v4_20_modeB 기반
  stage1_dev 후보 113,447개에 v1/v1 ConvAE reconstruction score를 계산하고
  RD-C2 RD4AD 결과와 같은 지표로 비교한다.

안전 조건:
  - stage2_holdout 접근 금지
  - training / backward / optimizer 금지
  - checkpoint 저장 금지
  - threshold 재계산 금지 (analysis-only sweep만 허용)
  - 기존 파일 수정/삭제 금지
  - output root 이미 있으면 즉시 ABORT

실행 모드:
  bare run   → exit 2, 파일 생성 금지
  --dry-plan → inventory + preflight 보고 (scoring 없음)
  --run-score → scoring + 분석 + DONE 생성
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/models"
    / "rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/best_val_loss.pt"
)

RD_C2_MANIFEST = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c2_effb0_v420_candidate_rd4ad_retest_v1"
    / "rd_c2_effb0_v420_candidate_manifest.csv"
)

RD_C2_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c2_effb0_v420_candidate_rd4ad_retest_v1"
    / "rd_c2_effb0_v420_rd4ad_candidate_score.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c3_v1v1_convae_same_universe_retest_v1"
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_TAG       = "rd4ad_2p5d_normal_mw_fixed96_v1"
MODEL_CLASS     = "ConvAutoencoder2p5D"
CROP_SIZE       = 96
LUNG_WIN_MIN    = -1350.0
LUNG_WIN_MAX    =   150.0
MEDI_WIN_MIN    =  -160.0
MEDI_WIN_MAX    =   240.0
BATCH_SIZE      = 64
EXPECTED_ROWS   = 113447
EXPECTED_POS    = 35247
EXPECTED_HN     = 78200
STAGE2_HOLDOUT_PATIENTS = {"LUNG1-295", "LUNG1-415"}

SCORE_COLS = [
    "convAE_mediastinal_channels_l1_mean",
    "convAE_lung_channels_l1_mean",
    "convAE_crop_score_l1_mean",
    "convAE_crop_score_mse_mean",
]

# ──────────────────────────────────────────────────────────────────────────────
# Formatting helper (avoids f-string format-spec conditional bugs)
# ──────────────────────────────────────────────────────────────────────────────
def fmt_float(x, ndigits: int = 4) -> str:
    if x is None:
        return "N/A"
    try:
        if np.isnan(float(x)):
            return "N/A"
    except Exception:
        pass
    return f"{float(x):.{ndigits}f}"


def compute_auroc_mann_whitney(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(y_score)
    sorted_scores = y_score[order]

    ranks = np.empty(len(sorted_scores), dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        i = j

    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks

    rank_sum_pos = float(original_ranks[y_true == 1].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def compute_average_precision(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]

    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None

    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    score_sorted = y_score[order]

    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0

    while i < len(score_sorted):
        j = i + 1
        while j < len(score_sorted) and score_sorted[j] == score_sorted[i]:
            j += 1

        group = y_true_sorted[i:j]
        tp += int((group == 1).sum())
        fp += int((group == 0).sum())

        recall = tp / n_pos
        precision = tp / max(tp + fp, 1)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j

    return float(ap)


# ──────────────────────────────────────────────────────────────────────────────
# Model definition (identical to train_rd4ad_2p5d_normal.py / score scripts)
# ──────────────────────────────────────────────────────────────────────────────
def _build_model():
    import torch
    import torch.nn as nn

    class ConvAutoencoder2p5D(nn.Module):
        def __init__(self, input_channels: int = 6, base_channels: int = 32):
            super().__init__()
            c = base_channels
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, c, 3, padding=1),
                nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c, c * 2, 3, padding=1),
                nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c * 2, c * 4, 3, padding=1),
                nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c * 4, c * 8, 3, padding=1),
                nn.BatchNorm2d(c * 8), nn.ReLU(inplace=True),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2),
                nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2),
                nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c * 2, c, 2, stride=2),
                nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.Conv2d(c, input_channels, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return ConvAutoencoder2p5D

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────
def apply_lung_window(arr: np.ndarray) -> np.ndarray:
    c = np.clip(arr, LUNG_WIN_MIN, LUNG_WIN_MAX)
    return ((c - LUNG_WIN_MIN) / (LUNG_WIN_MAX - LUNG_WIN_MIN)).astype(np.float32)

def apply_medi_window(arr: np.ndarray) -> np.ndarray:
    c = np.clip(arr, MEDI_WIN_MIN, MEDI_WIN_MAX)
    return ((c - MEDI_WIN_MIN) / (MEDI_WIN_MAX - MEDI_WIN_MIN)).astype(np.float32)

def build_6ch_crop(ct: np.ndarray, z: int, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    """
    ct: [Z, H, W] float32 HU values
    Returns: [6, 96, 96] float32 normalized
    ch0-2: lung_window (z-1, z, z+1)
    ch3-5: medi_window (z-1, z, z+1)
    """
    Z = ct.shape[0]
    z_prev = max(0, z - 1)
    z_next = min(Z - 1, z + 1)

    raw_prev = ct[z_prev, y0:y1, x0:x1].astype(np.float32)
    raw_curr = ct[z,      y0:y1, x0:x1].astype(np.float32)
    raw_next = ct[z_next, y0:y1, x0:x1].astype(np.float32)

    image = np.stack([
        apply_lung_window(raw_prev),
        apply_lung_window(raw_curr),
        apply_lung_window(raw_next),
        apply_medi_window(raw_prev),
        apply_medi_window(raw_curr),
        apply_medi_window(raw_next),
    ], axis=0)
    return image  # (6, 96, 96)

# ──────────────────────────────────────────────────────────────────────────────
# Score computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_scores_from_batch(inputs_np: np.ndarray, recons_np: np.ndarray) -> list:
    """
    inputs_np, recons_np: [B, 6, H, W] float32
    Returns: list of dicts, one per sample
    """
    diff    = np.abs(recons_np - inputs_np)            # [B,6,H,W]
    diff_sq = (recons_np - inputs_np) ** 2             # [B,6,H,W]

    results = []
    for b in range(inputs_np.shape[0]):
        d    = diff[b]     # [6,H,W]
        d_sq = diff_sq[b]

        crop_score_l1_mean  = float(d.mean())
        crop_score_mse_mean = float(d_sq.mean())
        lung_channels_l1_mean        = float(d[0:3].mean())
        mediastinal_channels_l1_mean = float(d[3:6].mean())

        score_nan = int(np.isnan(crop_score_l1_mean) or
                        np.isnan(mediastinal_channels_l1_mean) or
                        np.isnan(lung_channels_l1_mean) or
                        np.isnan(crop_score_mse_mean))
        score_inf = int(np.isinf(crop_score_l1_mean) or
                        np.isinf(mediastinal_channels_l1_mean) or
                        np.isinf(lung_channels_l1_mean) or
                        np.isinf(crop_score_mse_mean))

        results.append({
            "convAE_mediastinal_channels_l1_mean": mediastinal_channels_l1_mean,
            "convAE_lung_channels_l1_mean":        lung_channels_l1_mean,
            "convAE_crop_score_l1_mean":           crop_score_l1_mean,
            "convAE_crop_score_mse_mean":          crop_score_mse_mean,
            "score_nan":                           score_nan,
            "score_inf":                           score_inf,
        })
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Crop bounds preflight
# ──────────────────────────────────────────────────────────────────────────────
def preflight_crop_bounds(df: pd.DataFrame) -> int:
    """
    CT shape을 patient(safe_id)별로 한 번만 읽어 전체 crop 좌표를 검증한다.
    out-of-bounds 건수를 반환. 0이면 OK.
    """
    error_count = 0
    bad_samples = []
    ct_shape_cache = {}
    for safe_id, group in df.groupby("safe_id", sort=False):
        ct_path = CT_ROOT / safe_id / "ct_hu.npy"
        if not ct_path.exists():
            # CT missing → crop bounds 확인 불가 → error로 간주
            error_count += len(group)
            bad_samples.append(f"  CT_MISSING: {safe_id} ({len(group)} candidates)")
            continue
        if safe_id not in ct_shape_cache:
            arr = np.load(str(ct_path), mmap_mode="r")
            ct_shape_cache[safe_id] = arr.shape  # (Z, H, W)
        Z, H, W = ct_shape_cache[safe_id]
        for row in group.itertuples(index=False):
            y0 = int(row.crop_y0); x0 = int(row.crop_x0)
            y1 = int(row.crop_y1); x1 = int(row.crop_x1)
            z  = int(row.local_z)
            oob = (y0 < 0 or x0 < 0 or y1 > H or x1 > W or
                   (y1 - y0) != CROP_SIZE or (x1 - x0) != CROP_SIZE or
                   not (0 <= z < Z))
            if oob:
                error_count += 1
                if len(bad_samples) < 10:
                    bad_samples.append(
                        f"  OOB: {safe_id} cid={row.candidate_id} "
                        f"z={z}/[0,{Z}) y=[{y0},{y1})/[0,{H}) x=[{x0},{x1})/[0,{W})"
                    )
    if bad_samples:
        for s in bad_samples:
            print(s)
    return error_count

# ──────────────────────────────────────────────────────────────────────────────
# Safety checks
# ──────────────────────────────────────────────────────────────────────────────
def assert_no_stage2_holdout(df: pd.DataFrame, context: str) -> None:
    if "patient_id" in df.columns:
        found = set(df["patient_id"].unique()) & STAGE2_HOLDOUT_PATIENTS
    elif "safe_id" in df.columns:
        found = set()
        for p in STAGE2_HOLDOUT_PATIENTS:
            if df["safe_id"].str.contains(p, na=False).any():
                found.add(p)
    else:
        found = set()

    if found:
        print(f"[ABORT] stage2_holdout patients found in {context}: {found}", file=sys.stderr)
        sys.exit(1)
    print(f"  [OK] stage2_holdout intersection = 0 ({context})")

# ──────────────────────────────────────────────────────────────────────────────
# Dry-plan
# ──────────────────────────────────────────────────────────────────────────────
def run_dry_plan() -> None:
    print()
    print("=" * 72)
    print("  RD-C3a: Artifact / Code Inventory")
    print("=" * 72)

    # Checkpoint
    ckpt_exists = CHECKPOINT_PATH.exists()
    ckpt_size   = CHECKPOINT_PATH.stat().st_size // 1024 if ckpt_exists else 0
    print(f"\n[CHECKPOINT]")
    print(f"  path   : {CHECKPOINT_PATH}")
    print(f"  exists : {ckpt_exists}")
    print(f"  size   : {ckpt_size} KB")

    if ckpt_exists:
        import torch
        ck = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
        ck_keys = list(ck.keys()) if isinstance(ck, dict) else ["<tensor>"]
        print(f"  keys   : {ck_keys[:10]}")
    else:
        print("  [WARN] checkpoint not found")

    # Scoring code source
    print(f"\n[SCORING CODE SOURCE]")
    scoring_scripts = [
        PROJECT_ROOT / "scripts/score_rd4ad_2p5d_hard_negative.py",
        PROJECT_ROOT / "scripts/phase8_4_stage2_full_scoring.py",
        PROJECT_ROOT / "scripts/train_rd4ad_2p5d_normal.py",
    ]
    for s in scoring_scripts:
        print(f"  {'EXISTS' if s.exists() else 'MISSING'} : {s.name}")

    # ConvAutoencoder2p5D definition
    print(f"\n[MODEL CLASS]")
    print(f"  ConvAutoencoder2p5D: defined inline in this script (matches train + score scripts)")
    print(f"  input_channels: 6 (lung ch0-2 + mediastinal ch3-5)")
    print(f"  crop_size: {CROP_SIZE}")
    print(f"  preprocessing:")
    print(f"    lung_window    : [{LUNG_WIN_MIN}, {LUNG_WIN_MAX}] → [0, 1]")
    print(f"    mediastinal    : [{MEDI_WIN_MIN}, {MEDI_WIN_MAX}] → [0, 1]")
    print(f"    channel layout : [lung_z-1, lung_z, lung_z+1, medi_z-1, medi_z, medi_z+1]")

    # phase8_9 AUROC reference
    phase89_json = (
        PROJECT_ROOT
        / "outputs/second-stage-lesion-refiner-v1/review_annotations"
        / "phase8_9_paper_report_summary_v1/phase8_9_paper_report_summary.json"
    )
    print(f"\n[PHASE8_9 OLD AUROC REFERENCE]")
    if phase89_json.exists():
        with open(phase89_json) as f:
            p89 = json.load(f)
        print(f"  best_auroc              : {p89.get('best_auroc')}")
        print(f"  best_score_column       : {p89.get('best_score_column')}")
        print(f"  dataset                 : {p89.get('evaluation_scope', {}).get('dataset')}")
        print(f"  positive_crops          : {p89.get('evaluation_scope', {}).get('positive_crops')}")
        print(f"  hard_negative_crops     : {p89.get('evaluation_scope', {}).get('hard_negative_crops')}")
        for col, v in p89.get("crop_level_metrics", {}).items():
            print(f"    {col}: auroc={v['auroc']}, auprc={v['auprc']}")
    else:
        print("  [WARN] phase8_9 result JSON not found")

    print()
    print("=" * 72)
    print("  RD-C3b: RD-C2 Candidate Universe Compatibility Preflight")
    print("=" * 72)

    # Manifest
    if not RD_C2_MANIFEST.exists():
        print(f"  [ABORT] manifest not found: {RD_C2_MANIFEST}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(RD_C2_MANIFEST)
    print(f"\n[MANIFEST]")
    print(f"  path           : {RD_C2_MANIFEST}")
    print(f"  rows           : {len(df)}")
    print(f"  expected_rows  : {EXPECTED_ROWS}")
    print(f"  row_match      : {len(df) == EXPECTED_ROWS}")
    print(f"  columns        : {list(df.columns)}")

    label_counts = df["label"].value_counts().to_dict()
    print(f"\n[LABEL COUNTS]")
    for k, v in label_counts.items():
        print(f"  {k}: {v}")

    pos_ok = label_counts.get("positive", 0) == EXPECTED_POS
    hn_ok  = label_counts.get("hard_negative", 0) == EXPECTED_HN
    print(f"  positive match ({EXPECTED_POS}): {pos_ok}")
    print(f"  hard_negative match ({EXPECTED_HN}): {hn_ok}")

    # stage_split
    splits = df["stage_split"].value_counts().to_dict()
    print(f"\n[STAGE_SPLIT]")
    for k, v in splits.items():
        print(f"  {k}: {v}")
    if set(splits.keys()) != {"stage1_dev"}:
        print("  [ABORT] non-stage1_dev rows found", file=sys.stderr)
        sys.exit(1)
    print("  [OK] all rows are stage1_dev")

    # holdout intersection
    assert_no_stage2_holdout(df, "RD-C2 manifest")

    # CT path check (sample 10)
    print(f"\n[CT PATH CHECK (sample 10 patients)]")
    ct_ok = 0; ct_missing = 0
    for sid in df["safe_id"].unique()[:10]:
        ct_p = CT_ROOT / sid / "ct_hu.npy"
        if ct_p.exists():
            ct_ok += 1
        else:
            print(f"  [WARN] missing: {ct_p}")
            ct_missing += 1
    print(f"  CT found: {ct_ok}/10, missing: {ct_missing}/10")

    # All patients
    n_missing_full = sum(
        0 if (CT_ROOT / sid / "ct_hu.npy").exists() else 1
        for sid in df["safe_id"].unique()
    )
    n_patients = df["safe_id"].nunique()
    print(f"\n  Full CT check: {n_patients - n_missing_full}/{n_patients} patients have ct_hu.npy")
    if n_missing_full > 0:
        print(f"  [WARN] {n_missing_full} patients missing CT - scoring will ABORT on these")

    # Crop bounds preflight (전수 검증 - CT shape vs crop 좌표)
    print(f"\n[CROP BOUNDS PREFLIGHT (전수 검증)]")
    print(f"  정책: strict bounds. y0≥0, x0≥0, y1≤H, x1≤W, crop={CROP_SIZE}×{CROP_SIZE}, 0≤z<Z")
    print(f"  CT shapes loading (mmap, {n_patients} patients)...")
    crop_oob_count = preflight_crop_bounds(df)
    if crop_oob_count > 0:
        print(f"  [ABORT] crop_bounds_error_count={crop_oob_count}", file=sys.stderr)
        sys.exit(1)
    print(f"  [OK] crop_bounds_error_count=0 (all {len(df):,} candidates within CT bounds)")

    # Input format note
    print(f"\n[INPUT FORMAT NOTE]")
    print(f"  v1/v1 ConvAE input: ct_hu.npy (HU values) → on-the-fly 6ch crop")
    print(f"  Candidate source: v4_20 (roi_source=refined_roi_v4_20_modeB)")
    print(f"  v1/v1 preprocessing: roi_0_0.npy NOT applied to input pixels")
    print(f"  (v4_20 ROI is used only for candidate selection, not for pixel masking)")
    print(f"  This difference (candidate source=v4_20, pixel input=raw ct_hu) is noted in report")

    # RD-C2 score CSV
    print(f"\n[RD-C2 SCORE CSV]")
    if RD_C2_SCORE_CSV.exists():
        df_score = pd.read_csv(RD_C2_SCORE_CSV)
        print(f"  path  : {RD_C2_SCORE_CSV}")
        print(f"  rows  : {len(df_score)}")
        print(f"  cols  : {list(df_score.columns)}")
    else:
        print(f"  [WARN] not found: {RD_C2_SCORE_CSV}")

    # Output root guard
    print(f"\n[OUTPUT ROOT GUARD]")
    if OUTPUT_ROOT.exists():
        print(f"  [ABORT] output root already exists: {OUTPUT_ROOT}", file=sys.stderr)
        sys.exit(1)
    print(f"  OK - output root does not exist: {OUTPUT_ROOT}")

    print()
    print("=" * 72)
    print("  DRY-PLAN COMPLETE — Ready for --run-score after user approval")
    print("=" * 72)
    print(f"  Candidates to score : {len(df):,}")
    print(f"  CT root accessible  : {CT_ROOT.exists()}")
    print(f"  Checkpoint exists   : {CHECKPOINT_PATH.exists()}")
    print(f"  Output root clear   : {not OUTPUT_ROOT.exists()}")
    print()

# ──────────────────────────────────────────────────────────────────────────────
# Run scoring
# ──────────────────────────────────────────────────────────────────────────────
def run_score() -> None:
    import torch
    import torch.nn as nn

    t_start = time.time()

    # ── output root guard ──────────────────────────────────────────────────────
    if OUTPUT_ROOT.exists():
        print(f"[ABORT] output root already exists: {OUTPUT_ROOT}", file=sys.stderr)
        sys.exit(1)
    OUTPUT_ROOT.mkdir(parents=True)

    error_rows = []

    # ── load manifest ──────────────────────────────────────────────────────────
    print("[1/9] Loading RD-C2 candidate manifest...")
    if not RD_C2_MANIFEST.exists():
        print(f"[ABORT] manifest not found: {RD_C2_MANIFEST}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(RD_C2_MANIFEST)
    assert len(df) == EXPECTED_ROWS, f"row count mismatch: {len(df)} vs {EXPECTED_ROWS}"
    assert_no_stage2_holdout(df, "manifest")

    label_counts = df["label"].value_counts().to_dict()
    n_pos = label_counts.get("positive", 0)
    n_hn  = label_counts.get("hard_negative", 0)
    n_amb = len(df) - n_pos - n_hn
    print(f"  rows={len(df)}, positive={n_pos}, hard_negative={n_hn}, ambiguous={n_amb}")

    # ── load checkpoint ────────────────────────────────────────────────────────
    print("[2/9] Loading ConvAutoencoder2p5D checkpoint...")
    if not CHECKPOINT_PATH.exists():
        print(f"[ABORT] checkpoint not found: {CHECKPOINT_PATH}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    ConvAutoencoder2p5D = _build_model()
    model = ConvAutoencoder2p5D(input_channels=6).to(device)
    ck = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=False)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state = ck["model_state_dict"]
    elif isinstance(ck, dict) and "state_dict" in ck:
        state = ck["state_dict"]
    else:
        state = ck
    model.load_state_dict(state)
    model.eval()
    print(f"  checkpoint loaded: {CHECKPOINT_PATH.name}")
    print(f"  [OK] model.training={model.training}")

    # Safety: ensure no training mode
    assert not model.training, "model must be in eval mode"

    # ── load RD-C2 score for rd4ad_b8f_score join ─────────────────────────────
    print("[3/9] Loading RD-C2 RD4AD score CSV for join...")
    rd4ad_score_map = {}
    if RD_C2_SCORE_CSV.exists():
        df_c2 = pd.read_csv(RD_C2_SCORE_CSV)
        # rd4ad_b8f_score = rd4ad_crop_score column
        if "rd4ad_crop_score" in df_c2.columns and "candidate_id" in df_c2.columns:
            rd4ad_score_map = dict(zip(df_c2["candidate_id"], df_c2["rd4ad_crop_score"]))
            print(f"  rd4ad_b8f_score loaded: {len(rd4ad_score_map)} entries")
        else:
            print("  [WARN] rd4ad_crop_score column not found in RD-C2 score CSV")
    else:
        print("  [WARN] RD-C2 score CSV not found, rd4ad_b8f_score will be NaN")

    # ── crop bounds preflight (전수 검증) ─────────────────────────────────────
    print("[4/9] Crop bounds preflight (전수 검증 - strict bounds policy)...")
    crop_oob_count = preflight_crop_bounds(df)
    if crop_oob_count > 0:
        print(f"[ABORT] crop_bounds_error_count={crop_oob_count}", file=sys.stderr)
        sys.exit(1)
    print(f"  [OK] crop_bounds_error_count=0")

    # ── group by safe_id for CT caching ───────────────────────────────────────
    print("[5/9] Grouping candidates by safe_id (patient)...")
    groups = df.groupby("safe_id", sort=False)
    n_patients = df["safe_id"].nunique()
    print(f"  patients: {n_patients}")

    # ── scoring loop ──────────────────────────────────────────────────────────
    print("[6/9] ConvAE scoring (eval only, no_grad)...")
    all_score_rows = []
    n_scored = 0
    n_failed = 0

    with torch.no_grad():
        for patient_idx, (safe_id, group_df) in enumerate(groups):
            ct_path = CT_ROOT / safe_id / "ct_hu.npy"
            if not ct_path.exists():
                msg = f"ct_hu.npy not found: {ct_path}"
                print(f"[ABORT] {msg}", file=sys.stderr)
                # errors csv 저장 후 ABORT
                df_err_early = pd.DataFrame(error_rows + [{
                    "candidate_id": "ALL", "safe_id": safe_id,
                    "error_type": "CT_MISSING", "error_msg": msg,
                }])
                df_err_early.to_csv(OUTPUT_ROOT / "rd_c3_errors.csv", index=False)
                sys.exit(1)

            # Load CT (mmap for memory efficiency)
            try:
                ct = np.load(str(ct_path), mmap_mode="r")
            except Exception as e:
                msg = f"CT load error: {e}"
                print(f"[ABORT] {safe_id}: {msg}", file=sys.stderr)
                df_err_early = pd.DataFrame(error_rows + [{
                    "candidate_id": "ALL", "safe_id": safe_id,
                    "error_type": "CT_LOAD_ERROR", "error_msg": msg,
                }])
                df_err_early.to_csv(OUTPUT_ROOT / "rd_c3_errors.csv", index=False)
                sys.exit(1)

            Z, H, W = ct.shape
            rows_list = group_df.to_dict("records")

            # Process in batches
            for batch_start in range(0, len(rows_list), BATCH_SIZE):
                batch_rows = rows_list[batch_start : batch_start + BATCH_SIZE]
                crops_np   = []
                meta_list  = []
                crop_errors = []

                for row in batch_rows:
                    cid  = str(row["candidate_id"])
                    z    = int(row["local_z"])
                    y0   = int(row["crop_y0"])
                    x0   = int(row["crop_x0"])
                    y1   = int(row["crop_y1"])
                    x1   = int(row["crop_x1"])

                    # Validate crop size
                    if (y1 - y0) != CROP_SIZE or (x1 - x0) != CROP_SIZE:
                        msg = f"crop size mismatch: h={y1-y0}, w={x1-x0}, expected {CROP_SIZE}"
                        crop_errors.append({
                            "candidate_id": cid,
                            "safe_id": safe_id,
                            "error_type": "CROP_SIZE_MISMATCH",
                            "error_msg": msg,
                        })
                        continue

                    # Validate z range
                    if not (0 <= z < Z):
                        msg = f"local_z={z} out of range [0,{Z})"
                        crop_errors.append({
                            "candidate_id": cid,
                            "safe_id": safe_id,
                            "error_type": "Z_OUT_OF_RANGE",
                            "error_msg": msg,
                        })
                        continue

                    # Validate crop bounds
                    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
                        msg = f"crop bounds [{y0}:{y1},{x0}:{x1}] out of [H={H},W={W}]"
                        crop_errors.append({
                            "candidate_id": cid,
                            "safe_id": safe_id,
                            "error_type": "CROP_BOUNDS_ERROR",
                            "error_msg": msg,
                        })
                        continue

                    try:
                        img = build_6ch_crop(ct, z, y0, x0, y1, x1)
                    except Exception as e:
                        crop_errors.append({
                            "candidate_id": cid,
                            "safe_id": safe_id,
                            "error_type": "CROP_BUILD_ERROR",
                            "error_msg": str(e),
                        })
                        continue

                    if img.shape != (6, CROP_SIZE, CROP_SIZE):
                        crop_errors.append({
                            "candidate_id": cid,
                            "safe_id": safe_id,
                            "error_type": "CROP_SHAPE_ERROR",
                            "error_msg": f"shape={img.shape}",
                        })
                        continue

                    if not np.isfinite(img).all():
                        crop_errors.append({
                            "candidate_id": cid,
                            "safe_id": safe_id,
                            "error_type": "CROP_NONFINITE",
                            "error_msg": f"NaN/Inf in crop",
                        })
                        continue

                    crops_np.append(img)
                    meta_list.append(row)

                if crop_errors:
                    # 전수 통과 정책: crop 에러 1개라도 나면 즉시 ABORT
                    error_rows.extend(crop_errors)
                    pd.DataFrame(error_rows).to_csv(OUTPUT_ROOT / "rd_c3_errors.csv", index=False)
                    print(f"[ABORT] crop errors in batch: {len(crop_errors)}", file=sys.stderr)
                    for ce in crop_errors[:5]:
                        print(f"  {ce}", file=sys.stderr)
                    sys.exit(1)

                if not crops_np:
                    continue

                # Forward pass
                batch_t = torch.from_numpy(np.stack(crops_np, axis=0)).to(device)  # [B,6,96,96]
                recon_t = model(batch_t)

                inputs_np = batch_t.cpu().numpy()
                recons_np = recon_t.cpu().numpy()

                scores = compute_scores_from_batch(inputs_np, recons_np)

                for row_meta, score_dict in zip(meta_list, scores):
                    cid = str(row_meta["candidate_id"])
                    out_row = {
                        "candidate_id":         cid,
                        "patient_id":           row_meta["patient_id"],
                        "safe_id":              row_meta["safe_id"],
                        "stage_split":          row_meta["stage_split"],
                        "local_z":              row_meta["local_z"],
                        "crop_y0":              row_meta["crop_y0"],
                        "crop_x0":              row_meta["crop_x0"],
                        "crop_y1":              row_meta["crop_y1"],
                        "crop_x1":              row_meta["crop_x1"],
                        "label":                row_meta["label"],
                        "first_stage_score":    row_meta["first_stage_score"],
                        "rd4ad_b8f_score":      rd4ad_score_map.get(cid, float("nan")),
                        **score_dict,
                    }
                    all_score_rows.append(out_row)
                    n_scored += 1

            if (patient_idx + 1) % 20 == 0 or (patient_idx + 1) == n_patients:
                elapsed = time.time() - t_start
                print(f"  [{patient_idx+1}/{n_patients}] scored={n_scored}, failed={n_failed}, "
                      f"elapsed={elapsed:.0f}s")

    # ── safety: no backward/optimizer/checkpoint ──────────────────────────────
    # (guaranteed by model.eval() + torch.no_grad() + no optimizer creation)

    print(f"\n[7/9] Scoring complete: scored={n_scored}, failed={n_failed}, errors={len(error_rows)}")

    if n_failed > 0:
        print(f"[ABORT] failed_candidates={n_failed} (must be 0)", file=sys.stderr)
        sys.exit(1)
    if n_scored == 0:
        print("[ABORT] No candidates scored", file=sys.stderr)
        sys.exit(1)
    if n_scored != EXPECTED_ROWS:
        print(f"[ABORT] scored={n_scored} != expected={EXPECTED_ROWS}", file=sys.stderr)
        sys.exit(1)

    df_score = pd.DataFrame(all_score_rows)

    # NaN/Inf check
    total_nan = int(df_score["score_nan"].sum())
    total_inf = int(df_score["score_inf"].sum())
    print(f"  score_nan={total_nan}, score_inf={total_inf}")
    if total_nan > 0 or total_inf > 0:
        print(f"[ABORT] NaN/Inf detected in scores: nan={total_nan}, inf={total_inf}", file=sys.stderr)
        sys.exit(1)

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("[8/9] Analysis: AUROC/AUPRC/distribution/suppression...")

    df_eval = df_score[df_score["label"].isin(["positive", "hard_negative"])].copy()
    y_true  = (df_eval["label"] == "positive").astype(int).values

    auroc_by_col  = {}
    auprc_by_col  = {}
    for col in SCORE_COLS:
        if col not in df_eval.columns:
            continue
        try:
            auroc_by_col[col] = compute_auroc_mann_whitney(y_true, df_eval[col].values)
            auprc_by_col[col] = compute_average_precision(y_true, df_eval[col].values)
        except Exception as e:
            auroc_by_col[col] = None
            auprc_by_col[col] = None
            print(f"  [WARN] AUROC/AUPRC failed for {col}: {e}")

    # rd4ad_b8f_score AUROC (from RD-C2)
    rd4ad_auroc = None
    if "rd4ad_b8f_score" in df_eval.columns:
        valid = df_eval["rd4ad_b8f_score"].notna()
        if valid.sum() > 0:
            try:
                rd4ad_auroc = compute_auroc_mann_whitney(
                    y_true[valid.values],
                    df_eval.loc[valid, "rd4ad_b8f_score"].values,
                )
            except Exception:
                pass

    best_col  = max(auroc_by_col, key=lambda c: auroc_by_col[c] or 0)
    best_auroc = auroc_by_col.get(best_col)
    best_auprc = auprc_by_col.get(best_col)

    print(f"\n  AUROC by score column:")
    for col, v in auroc_by_col.items():
        print(f"    {col}: {v:.4f}" if v is not None else f"    {col}: N/A")
    print(f"\n  AUPRC by score column:")
    for col, v in auprc_by_col.items():
        print(f"    {col}: {v:.4f}" if v is not None else f"    {col}: N/A")
    print(f"  RD-B8f (rd4ad_b8f_score) AUROC: {fmt_float(rd4ad_auroc, 4)}")
    print(f"  Best ConvAE column: {best_col} (AUROC={fmt_float(best_auroc, 4)})")

    # Score distribution
    dist_rows = []
    for col in SCORE_COLS:
        if col not in df_eval.columns:
            continue
        for lbl in ["positive", "hard_negative"]:
            sub = df_eval[df_eval["label"] == lbl][col]
            dist_rows.append({
                "score_col": col,
                "label": lbl,
                "count": len(sub),
                "mean": float(sub.mean()),
                "median": float(sub.median()),
                "p95": float(sub.quantile(0.95)),
                "p99": float(sub.quantile(0.99)),
            })
    df_dist = pd.DataFrame(dist_rows)

    # Pearson/Spearman correlation with first_stage_score
    from scipy.stats import pearsonr, spearmanr
    corr_rows = []
    for col in SCORE_COLS:
        if col not in df_eval.columns:
            continue
        try:
            pr = pearsonr(df_eval["first_stage_score"], df_eval[col])[0]
            sr = spearmanr(df_eval["first_stage_score"], df_eval[col])[0]
        except Exception:
            pr = sr = float("nan")
        corr_rows.append({"score_col": col, "pearson_r": pr, "spearman_r": sr})
    df_corr = pd.DataFrame(corr_rows)

    # ── Safety-constrained threshold sweep (analysis-only) ────────────────────
    print("[9/9] Safety-constrained threshold sweep (analysis-only)...")
    df_pos = df_eval[df_eval["label"] == "positive"]
    df_hn  = df_eval[df_eval["label"] == "hard_negative"]

    # Patient-level data
    pos_patients = df_pos["patient_id"].unique()
    n_pos_patients = len(pos_patients)

    sweep_rows = []
    safety_best = {}  # col -> {le1pct, le3pct, le5pct}
    thresh_by_col = {}

    for col in SCORE_COLS:
        if col not in df_eval.columns:
            continue
        # Percentile sweep
        all_scores = df_eval[col].values
        pct_range = np.linspace(0, 100, 201)  # 0.5% steps
        thresholds = np.percentile(all_scores, pct_range)

        best_le1 = {"threshold": None, "hn_suppressed_rate": 0.0, "lesion_patient_all_suppressed": n_pos_patients}
        best_le3 = {"threshold": None, "hn_suppressed_rate": 0.0, "lesion_patient_all_suppressed": n_pos_patients}
        best_le5 = {"threshold": None, "hn_suppressed_rate": 0.0, "lesion_patient_all_suppressed": n_pos_patients}

        for thr in thresholds:
            # suppress if score <= thr (lower score = normal-like)
            # For ConvAE, higher score = more anomalous, so:
            # suppress = score <= thr (normal-like)
            pos_suppressed = (df_pos[col] <= thr).sum()
            hn_suppressed  = (df_hn[col]  <= thr).sum()

            les_sup_rate = pos_suppressed / max(len(df_pos), 1)
            hn_sup_rate  = hn_suppressed  / max(len(df_hn),  1)

            # Patient-level: all candidates of a lesion patient suppressed
            pat_all_sup = 0
            for pid in pos_patients:
                pat_df = df_pos[df_pos["patient_id"] == pid]
                if (pat_df[col] <= thr).all():
                    pat_all_sup += 1

            row = {
                "score_col": col,
                "threshold": float(thr),
                "lesion_suppressed_rate": float(les_sup_rate),
                "hn_suppressed_rate": float(hn_sup_rate),
                "lesion_patient_all_suppressed": pat_all_sup,
            }
            sweep_rows.append(row)

            # Find best for each constraint
            if les_sup_rate <= 0.01 and hn_sup_rate > best_le1["hn_suppressed_rate"]:
                best_le1 = {"threshold": float(thr), "hn_suppressed_rate": float(hn_sup_rate),
                            "lesion_patient_all_suppressed": pat_all_sup}
            if les_sup_rate <= 0.03 and hn_sup_rate > best_le3["hn_suppressed_rate"]:
                best_le3 = {"threshold": float(thr), "hn_suppressed_rate": float(hn_sup_rate),
                            "lesion_patient_all_suppressed": pat_all_sup}
            if les_sup_rate <= 0.05 and hn_sup_rate > best_le5["hn_suppressed_rate"]:
                best_le5 = {"threshold": float(thr), "hn_suppressed_rate": float(hn_sup_rate),
                            "lesion_patient_all_suppressed": pat_all_sup}

        safety_best[col] = {
            "le1pct": best_le1,
            "le3pct": best_le3,
            "le5pct": best_le5,
        }
        thresh_by_col[col] = {
            "le1pct": best_le1["threshold"],
            "le3pct": best_le3["threshold"],
            "le5pct": best_le5["threshold"],
        }

        thr1 = fmt_float(best_le1["threshold"], 6)
        thr3 = fmt_float(best_le3["threshold"], 6)
        thr5 = fmt_float(best_le5["threshold"], 6)
        print(f"  {col}:")
        print(f"    @le1%: thr={thr1}, "
              f"hn_sup={best_le1['hn_suppressed_rate']:.2%}, pat_all_sup={best_le1['lesion_patient_all_suppressed']}")
        print(f"    @le3%: thr={thr3}, "
              f"hn_sup={best_le3['hn_suppressed_rate']:.2%}, pat_all_sup={best_le3['lesion_patient_all_suppressed']}")
        print(f"    @le5%: thr={thr5}, "
              f"hn_sup={best_le5['hn_suppressed_rate']:.2%}, pat_all_sup={best_le5['lesion_patient_all_suppressed']}")

    df_sweep = pd.DataFrame(sweep_rows)

    # Patient-level safety summary
    pat_rows = []
    for col in SCORE_COLS:
        if col not in df_eval.columns:
            continue
        for pid in pos_patients:
            pat_df = df_eval[df_eval["patient_id"] == pid]
            pat_pos = pat_df[pat_df["label"] == "positive"]
            pat_row = {
                "score_col": col,
                "patient_id": pid,
                "n_candidates": len(pat_pos),
                "score_min": float(pat_pos[col].min()) if len(pat_pos) else float("nan"),
                "score_max": float(pat_pos[col].max()) if len(pat_pos) else float("nan"),
                "score_mean": float(pat_pos[col].mean()) if len(pat_pos) else float("nan"),
            }
            pat_rows.append(pat_row)
    df_pat = pd.DataFrame(pat_rows)

    # Comparison table
    comp_rows = [
        {
            "source":         "phase8_9 v1/v1 stage2_holdout",
            "dataset":        "stage2_holdout",
            "n_positive":     51335,
            "n_hard_negative": 92400,
            "mediastinal_channels_l1_mean_auroc": 0.7071,
            "crop_score_l1_mean_auroc":          0.6822,
            "crop_score_mse_mean_auroc":         0.6581,
            "lung_channels_l1_mean_auroc":       0.5365,
        },
        {
            "source":         "RD-B8f/RD-C2 true RD4AD same-universe",
            "dataset":        "stage1_dev",
            "n_positive":     EXPECTED_POS,
            "n_hard_negative": EXPECTED_HN,
            "rd4ad_crop_score_auroc": rd4ad_auroc if rd4ad_auroc else "N/A",
        },
        {
            "source":         "RD-C3 v1/v1 ConvAE same-universe",
            "dataset":        "stage1_dev",
            "n_positive":     n_pos,
            "n_hard_negative": n_hn,
            "mediastinal_channels_l1_mean_auroc": auroc_by_col.get("convAE_mediastinal_channels_l1_mean"),
            "crop_score_l1_mean_auroc":           auroc_by_col.get("convAE_crop_score_l1_mean"),
            "crop_score_mse_mean_auroc":          auroc_by_col.get("convAE_crop_score_mse_mean"),
            "lung_channels_l1_mean_auroc":        auroc_by_col.get("convAE_lung_channels_l1_mean"),
        },
    ]
    df_comp = pd.DataFrame(comp_rows)

    # ── Final decision ────────────────────────────────────────────────────────
    rdb8f_auroc_ref = 0.5021  # known from RD-C2 summary
    best_convae_auroc = best_auroc or 0.0
    best_le1_min_pat_sup = min(
        (safety_best[c]["le1pct"]["lesion_patient_all_suppressed"]
         for c in SCORE_COLS if c in safety_best),
        default=n_pos_patients,
    )
    best_le1_hn_sup = max(
        (safety_best[c]["le1pct"]["hn_suppressed_rate"]
         for c in SCORE_COLS if c in safety_best),
        default=0.0,
    )

    if best_convae_auroc > rdb8f_auroc_ref + 0.05:
        if best_le1_min_pat_sup == 0 and best_le1_hn_sup > 0.01:
            recommended_decision = "CONVAE_USEFUL_FOR_RANKING"
        else:
            recommended_decision = "CONVAE_ANALYSIS_ONLY"
    elif best_convae_auroc > rdb8f_auroc_ref:
        recommended_decision = "CONVAE_ANALYSIS_ONLY"
    else:
        recommended_decision = "CONVAE_NOT_USEFUL"

    print(f"\n  Recommended decision: {recommended_decision}")

    # ── AUC/AUPRC comparison CSV ──────────────────────────────────────────────
    auc_rows = []
    for col in SCORE_COLS:
        auc_rows.append({
            "score_col": col,
            "auroc": auroc_by_col.get(col),
            "auprc": auprc_by_col.get(col),
        })
    df_auc = pd.DataFrame(auc_rows)

    # Fixed threshold summary (using phase8_9 threshold - none recorded, so skip)
    df_fixed = pd.DataFrame([{"note": "phase8_9_old_threshold_not_available",
                               "threshold_source_missing": True}])

    # Suppression safety by fixed threshold
    df_safety_fixed = pd.DataFrame([{
        "threshold_source": "old_phase8_9",
        "available": False,
        "note": "phase8_9 did not record explicit suppression threshold"
    }])

    # Inventory CSV
    inv_rows = [
        {"item": "checkpoint",    "path": str(CHECKPOINT_PATH),    "exists": CHECKPOINT_PATH.exists(),
         "model_class": MODEL_CLASS, "input_channels": 6},
        {"item": "rd_c2_manifest", "path": str(RD_C2_MANIFEST),   "exists": RD_C2_MANIFEST.exists(),
         "rows": len(df), "note": "candidate universe for scoring"},
        {"item": "rd_c2_score",   "path": str(RD_C2_SCORE_CSV),   "exists": RD_C2_SCORE_CSV.exists(),
         "note": "rd4ad_crop_score = rd4ad_b8f_score"},
        {"item": "scoring_code",  "path": "inline in this script", "exists": True,
         "note": "ConvAutoencoder2p5D class identical to train/score scripts"},
    ]
    df_inv = pd.DataFrame(inv_rows)

    # Preflight CSV
    pf_rows = [
        {"check": "manifest_rows",        "expected": EXPECTED_ROWS,  "actual": len(df),
         "pass": len(df) == EXPECTED_ROWS},
        {"check": "positive_count",       "expected": EXPECTED_POS,   "actual": n_pos,
         "pass": n_pos == EXPECTED_POS},
        {"check": "hard_negative_count",  "expected": EXPECTED_HN,    "actual": n_hn,
         "pass": n_hn == EXPECTED_HN},
        {"check": "stage2_holdout_intersection", "expected": 0,       "actual": 0,  "pass": True},
        {"check": "score_nan",            "expected": 0,              "actual": total_nan, "pass": total_nan == 0},
        {"check": "score_inf",            "expected": 0,              "actual": total_inf, "pass": total_inf == 0},
        {"check": "scored_candidates",    "expected": EXPECTED_ROWS,  "actual": n_scored,
         "pass": n_scored == EXPECTED_ROWS},
    ]
    df_pf = pd.DataFrame(pf_rows)

    # Salvage strategy summary
    salv_rows = []
    for col in SCORE_COLS:
        if col not in safety_best:
            continue
        salv_rows.append({
            "score_col": col,
            "auroc": auroc_by_col.get(col),
            "auprc": auprc_by_col.get(col),
            "best_hn_sup_at_le1pct_lesion": safety_best[col]["le1pct"]["hn_suppressed_rate"],
            "pat_all_sup_at_le1pct": safety_best[col]["le1pct"]["lesion_patient_all_suppressed"],
            "best_hn_sup_at_le3pct_lesion": safety_best[col]["le3pct"]["hn_suppressed_rate"],
            "pat_all_sup_at_le3pct": safety_best[col]["le3pct"]["lesion_patient_all_suppressed"],
            "best_hn_sup_at_le5pct_lesion": safety_best[col]["le5pct"]["hn_suppressed_rate"],
            "pat_all_sup_at_le5pct": safety_best[col]["le5pct"]["lesion_patient_all_suppressed"],
        })
    df_salv = pd.DataFrame(salv_rows)

    # ── Save all outputs ──────────────────────────────────────────────────────
    print("Saving outputs...")

    def _save(df_out, fname):
        p = OUTPUT_ROOT / fname
        df_out.to_csv(p, index=False)
        print(f"  saved: {p.name}")

    _save(df_inv,       "rd_c3_v1v1_artifact_inventory.csv")
    _save(df_pf,        "rd_c3_input_compatibility_preflight.csv")
    _save(df_score,     "rd_c3_v1v1_convae_candidate_score.csv")
    _save(df_dist,      "rd_c3_score_distribution_summary.csv")
    _save(df_auc,       "rd_c3_auc_auprc_comparison.csv")
    _save(df_safety_fixed, "rd_c3_suppression_safety_by_fixed_threshold.csv")
    _save(df_sweep,     "rd_c3_safety_constrained_threshold_sweep.csv")
    _save(df_pat,       "rd_c3_patient_level_safety_summary.csv")
    _save(df_comp,      "rd_c3_old_vs_rdc2_vs_convae_comparison.csv")
    _save(df_salv,      "rd_c3_salvage_strategy_summary.csv")

    # Error CSV
    df_err = pd.DataFrame(error_rows) if error_rows else pd.DataFrame(
        columns=["candidate_id", "safe_id", "error_type", "error_msg"])
    _save(df_err, "rd_c3_errors.csv")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    t_elapsed = time.time() - t_start
    all_checks_passed = (
        total_nan == 0
        and total_inf == 0
        and n_failed == 0
        and n_scored == EXPECTED_ROWS
        and label_counts.get("positive", 0) == EXPECTED_POS
        and label_counts.get("hard_negative", 0) == EXPECTED_HN
    )

    summary = {
        "input_rows":          EXPECTED_ROWS,
        "scored_candidates":   n_scored,
        "failed_candidates":   n_failed,
        "positive_count":      n_pos,
        "hard_negative_count": n_hn,
        "ambiguous_count":     n_amb,
        "checkpoint_loaded":   True,
        "convAE_model_class":  MODEL_CLASS,
        "input_channels":      6,
        "preprocessing_source": (
            "phase8_2f: lung_window[-1350,150], medi_window[-160,240], "
            "6ch=[lung_z-1,lung_z,lung_z+1,medi_z-1,medi_z,medi_z+1], crop=96x96"
        ),
        "score_nan_count":     total_nan,
        "score_inf_count":     total_inf,
        "stage2_holdout_intersection": 0,
        "auroc_by_score_column": auroc_by_col,
        "auprc_by_score_column": auprc_by_col,
        "best_score_column":   best_col,
        "rd_b8f_auroc_same_universe": rd4ad_auroc,
        "convae_best_auroc_same_universe": best_auroc,
        "safety_constrained_best_threshold_by_score": thresh_by_col,
        "best_hn_suppression_at_lesion_suppression_le_1pct": {
            col: safety_best[col]["le1pct"]["hn_suppressed_rate"]
            for col in SCORE_COLS if col in safety_best
        },
        "best_hn_suppression_at_lesion_suppression_le_3pct": {
            col: safety_best[col]["le3pct"]["hn_suppressed_rate"]
            for col in SCORE_COLS if col in safety_best
        },
        "best_hn_suppression_at_lesion_suppression_le_5pct": {
            col: safety_best[col]["le5pct"]["hn_suppressed_rate"]
            for col in SCORE_COLS if col in safety_best
        },
        "lesion_patient_all_suppressed_min": best_le1_min_pat_sup,
        "recommended_decision": recommended_decision,
        "training_started":                  False,
        "backward_called":                   False,
        "optimizer_created":                 False,
        "checkpoint_saved":                  False,
        "threshold_applied":                 False,
        "threshold_recalculated":            False,
        "analysis_only_threshold_sweep":     True,
        "first_stage_score_modified":        False,
        "stage2_holdout_access":             0,
        "all_checks_passed":                 all_checks_passed,
        "elapsed_seconds":                   round(t_elapsed, 1),
        "timestamp":                         datetime.now().isoformat(),
        "metric_backend":                    "sklearn_free_mann_whitney_auroc_and_step_average_precision",
        "sklearn_used":                      False,
    }

    summary_path = OUTPUT_ROOT / "rd_c3_v1v1_convae_same_universe_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  saved: {summary_path.name}")

    # ── Report MD ─────────────────────────────────────────────────────────────
    rd4ad_auroc_str = f"{rd4ad_auroc:.4f}" if rd4ad_auroc is not None else "N/A"
    best_auroc_str  = f"{best_auroc:.4f}"  if best_auroc  is not None else "N/A"

    report_lines = [
        "# RD-C3: v1/v1 ConvAE Same-Universe Retest Report",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        f"**scored_candidates**: {n_scored:,}",
        f"**recommended_decision**: {recommended_decision}",
        "",
        "## 1. 왜 RD-C3가 필요한지",
        "",
        "RD-B8f true RD4AD teacher-student는 RD-C2 EfficientNet-B0 v4_20 후보 기준",
        "crop-level AUROC 0.5021로 실질적 판별력이 없었다 (final_decision=NOT_USEFUL).",
        "반면 이전 phase8_9에서 v1/v1 ConvAutoencoder2p5D는 stage2_holdout에서",
        "mediastinal_channels_l1_mean AUROC=0.7071을 보였다.",
        "RD-C3는 동일한 후보 universe(stage1_dev, EfficientNet-B0+v4_20 기반)에서",
        "ConvAE가 실제로 유의미한 판별력을 갖는지 확인하기 위해 설계되었다.",
        "",
        "## 2. v1/v1과 RD-B8f 구조 차이",
        "",
        "| 항목 | v1/v1 ConvAE | RD-B8f true RD4AD |",
        "|------|--------------|-------------------|",
        "| 구조 | Encoder-Decoder (ConvAutoencoder) | ResNet18 teacher-student |",
        "| 입력 | 6ch HU windowed [96,96] | 3ch feature map |",
        "| 학습 | normal crop reconstruction | feature distillation |",
        "| 이상 신호 | reconstruction error ↑ | feature mismatch ↑ |",
        "",
        "## 3. 동일 universe 비교 이유",
        "",
        "이전 phase8_9 결과는 stage2_holdout(154명 전원 lesion)에서 측정된 것으로",
        "candidate selection 방식(roi_0_0 기반)도 달랐다.",
        "RD-C3는 RD-C2와 동일한 stage1_dev 113,447개 후보에 적용해",
        "동일 universe에서 공정한 비교를 수행한다.",
        "",
        "## 4. checkpoint/input/preprocessing 확인 결과",
        "",
        f"- checkpoint: `{CHECKPOINT_PATH.name}` (6.5MB, ConvAutoencoder2p5D)",
        f"- input: ct_hu.npy → 6ch [lung_z-1/z/z+1, medi_z-1/z/z+1] @ 96x96",
        f"- lung_window: [{LUNG_WIN_MIN}, {LUNG_WIN_MAX}] → [0, 1]",
        f"- medi_window: [{MEDI_WIN_MIN}, {MEDI_WIN_MAX}] → [0, 1]",
        f"- **주의**: candidate source는 v4_20이지만, pixel preprocessing은 roi_0_0 학습 당시와 동일 (ROI masking 없음)",
        "",
        "## 5. ConvAE scoring 결과",
        "",
        f"- scored: {n_scored:,}",
        f"- failed: {n_failed}",
        f"- score_nan: {total_nan}",
        f"- score_inf: {total_inf}",
        "",
        "## 6. AUROC/AUPRC 비교",
        "",
        "| score column | AUROC | AUPRC |",
        "|-------------|-------|-------|",
    ]
    for col in SCORE_COLS:
        auroc_v = auroc_by_col.get(col)
        auprc_v = auprc_by_col.get(col)
        report_lines.append(
            f"| {col} | {fmt_float(auroc_v, 4)} | {fmt_float(auprc_v, 4)} |"
        )
    report_lines += [
        f"| rd4ad_b8f_score (RD-C2) | {rd4ad_auroc_str} | N/A |",
        "",
        "## 7. 병변 억제율 해석",
        "",
        "억제 방향: score ≤ threshold → suppress (normal-like로 판단)",
        "ConvAE 특성상 정상 crop은 reconstruction error가 낮고 병변 crop은 높음.",
        "",
        "## 8. safety-constrained threshold sweep 결과",
        "",
        f"- lesion_patient_all_suppressed 최소: {best_le1_min_pat_sup}",
        f"- best_hn_suppression @le1%: {best_le1_hn_sup:.2%}",
        "",
        "## 9. 활용 가능성",
        "",
        f"**best AUROC**: {best_auroc_str} ({best_col})",
        f"**RD-B8f AUROC**: {rd4ad_auroc_str}",
        f"**phase8_9 AUROC (구 universe)**: 0.7071",
        "",
        "## 10. 최종 결정",
        "",
        f"**recommended_decision**: `{recommended_decision}`",
        "",
        "## 11. 절대 하지 않은 것",
        "",
        "- 새 학습 없음 (training_started=false)",
        "- threshold 실제 적용 없음 (threshold_applied=false)",
        "- stage2_holdout 접근 없음 (stage2_holdout_access=0)",
        "- first-stage score 수정 없음 (first_stage_score_modified=false)",
        "- backward/optimizer 없음 (backward_called=false, optimizer_created=false)",
        "- checkpoint 저장 없음 (checkpoint_saved=false)",
    ]

    report_path = OUTPUT_ROOT / "rd_c3_v1v1_convae_same_universe_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"  saved: {report_path.name}")

    # ── DONE ──────────────────────────────────────────────────────────────────
    done = {
        "status":            "DONE",
        "recommended_decision": recommended_decision,
        "all_checks_passed": all_checks_passed,
        "scored_candidates": n_scored,
        "convae_best_auroc": best_auroc,
        "rd_b8f_auroc":      rd4ad_auroc,
        "timestamp":         datetime.now().isoformat(),
    }
    done_path = OUTPUT_ROOT / "DONE"
    with open(done_path, "w") as f:
        json.dump(done, f, indent=2, default=str)
    print(f"  saved: DONE")

    print()
    print("=" * 72)
    print("  RD-C3 COMPLETE")
    print(f"  scored       : {n_scored:,}")
    print(f"  best_auroc   : {best_auroc_str} ({best_col})")
    print(f"  rd_b8f_auroc : {rd4ad_auroc_str}")
    print(f"  decision     : {recommended_decision}")
    print(f"  elapsed      : {t_elapsed:.0f}s")
    print("=" * 72)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) == 1:
        print("[EXIT 2] bare run guard - no mode flag provided", file=sys.stderr)
        print("  usage: --dry-plan or --run-score", file=sys.stderr)
        sys.exit(2)

    parser = argparse.ArgumentParser(description="RD-C3 v1/v1 ConvAE Same-Universe Retest")
    parser.add_argument("--dry-plan",   action="store_true", help="Artifact inventory + preflight")
    parser.add_argument("--run-score",  action="store_true", help="ConvAE scoring + analysis")
    args = parser.parse_args()

    if args.dry_plan:
        run_dry_plan()
    elif args.run_score:
        run_score()
    else:
        print("[EXIT 2] no valid mode flag", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
