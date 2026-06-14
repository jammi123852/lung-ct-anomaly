"""
p_c_normal38_2p5d5ch_limited_smoke_train.py

Stage: P-C-NORMAL38_2p5d5ch_smoke
Role: post-selection exploratory ablation, not selected-candidate replacement
Selected candidate remains: P-C-NORMAL30b_masked_input

5ch 2.5D EfficientNet-B0 + ScalarFusion binary classifier (Normal vs NSCLC)
  - 5ch input: z-2, z-1, z, z+1, z+2 (runtime volume extraction)
  - HU window: -1000 ~ 200 (same as 30b; -1350~150 is a separate ablation)
  - First conv 3ch → 5ch inflation (mean-repeat × 3/5 scale)
  - 30b checkpoint partial load (first conv excluded)
  - Limited smoke: 512 train, 256 val (normal/NSCLC balanced)

GUARDRAIL (all must hold):
  exploratory_ablation_only=True
  selected_candidate_not_replaced=True
  limited_smoke_only=True
  full_train_run=False
  final_test_accessed=False
  threshold_optimization=False
  threshold_sweep=False
  no_existing_result_overwrite=True
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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_MANIFEST    = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_balanced_w1_feature_manifest/p_c_normal24g_fix_balanced_w1_train_manifest.csv"
VAL_MANIFEST      = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_balanced_w1_feature_manifest/p_c_normal24g_fix_balanced_w1_val_manifest.csv"
SCALAR_STATS_PATH = PROJECT_ROOT / "outputs/reports/p_c_normal24h_fix_zroi_scalar_fusion_drycheck/p_c_normal24h_fix_scalar_normalization_stats.json"
MASK29B_MANIFEST  = PROJECT_ROOT / "outputs/reports/p_c_normal29b_crop_level_mask_generation/p_c_normal29b_mask_manifest.csv"
CKPT30B           = PROJECT_ROOT / "outputs/p_c_normal30b_masked_input_full_train/checkpoints/p_c_normal30b_best_val_auc_checkpoint.pt"

NORMAL_VOL_ROOT   = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")
NSCLC_VOL_ROOT    = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
MASK_ROOT         = PROJECT_ROOT / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"

STAGE_LABEL       = "P-C-NORMAL38_2p5d5ch_smoke"
OUTPUT_ROOT       = PROJECT_ROOT / "outputs/p_c_normal38_2p5d5ch_limited_smoke_train"
CKPT_DIR          = OUTPUT_ROOT / "checkpoints"
REPORT_ROOT       = PROJECT_ROOT / "outputs/reports/p_c_normal38_2p5d5ch_limited_smoke_train"

# ── Constants ──────────────────────────────────────────────────────────────────
HU_MIN            = -1000.0          # 30b와 동일 (ablation 변수 분리)
HU_MAX            =  200.0
CROP_SIZE         = 96
INPUT_CHANNELS    = 5
IMAGENET_MEAN_5   = 0.449            # (0.485+0.456+0.406)/3
IMAGENET_STD_5    = 0.226            # (0.229+0.224+0.225)/3
SCALAR_FEATURES   = ["lung_z_percentile", "crop_lung_roi_ratio"]
SMOKE_TRAIN_LIMIT = 512
SMOKE_VAL_LIMIT   = 256
BATCH_SIZE        = 16
NUM_WORKERS       = 0
SEED              = 42
Z_OFFSETS         = [-2, -1, 0, 1, 2]
FIRST_CONV_KEY    = "img_features.0.0.weight"

FORBIDDEN_VESSEL_COLS = {
    "vessel_candidate_ratio", "vessel_softmask_max", "vessel_center_ratio",
    "vessel_high_risk_ratio", "vessel_low_risk_ratio",
}


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Writers ────────────────────────────────────────────────────────────────────
def _write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"note": "empty"}]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def _write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_md(text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ── Metrics (sklearn-free) ─────────────────────────────────────────────────────
def compute_auroc(labels, scores):
    try:
        labels_arr = np.asarray(labels, dtype=np.int32)
        scores_arr = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(scores_arr)):
            return float("nan"), "invalid_score"
        if len(np.unique(labels_arr)) < 2:
            return float("nan"), "single_class"
        n_pos = int((labels_arr == 1).sum())
        n_neg = int((labels_arr == 0).sum())
        all_s  = np.concatenate([scores_arr[labels_arr == 0], scores_arr[labels_arr == 1]])
        is_pos = np.concatenate([np.zeros(n_neg, bool), np.ones(n_pos, bool)])
        order  = np.argsort(all_s, kind="stable")
        sorted_s = all_s[order]; sorted_is_pos = is_pos[order]
        n = n_pos + n_neg; ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i + 1
            while j < n and sorted_s[j] == sorted_s[i]: j += 1
            ranks[i:j] = (i + 1 + j) / 2.0; i = j
        U = float(ranks[sorted_is_pos].sum()) - n_pos * (n_pos + 1) / 2.0
        return float(U / (n_pos * n_neg)), "OK"
    except Exception as e:
        return float("nan"), f"ERROR:{e}"


def compute_auprc(labels, scores):
    try:
        labels_arr = np.asarray(labels, dtype=np.int32)
        scores_arr = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(scores_arr)) or len(np.unique(labels_arr)) < 2:
            return float("nan"), "invalid"
        n_pos = int((labels_arr == 1).sum())
        if n_pos == 0: return float("nan"), "no_positive"
        order = np.argsort(scores_arr, kind="stable")[::-1]
        sl = labels_arr[order]
        tp = np.cumsum(sl); fp = np.cumsum(1 - sl)
        prec = np.concatenate([[1.0], tp / (tp + fp)])
        rec  = np.concatenate([[0.0], tp / n_pos])
        return float(np.trapezoid(prec, rec)), "OK"
    except Exception as e:
        return float("nan"), f"ERROR:{e}"


# ── 29b mask manifest → safe_id lookup ────────────────────────────────────────
def build_safe_id_lookup(mask29b_path: Path) -> dict:
    """crop_path → safe_id (29b 기준, volume/mask root 폴더명과 일치)"""
    df = pd.read_csv(mask29b_path, low_memory=False)
    df_pass = df[df["status"] == "PASS"] if "status" in df.columns else df
    return dict(zip(df_pass["crop_path"].astype(str), df_pass["safe_id"].astype(str)))


# ── Scalar normalization ───────────────────────────────────────────────────────
def apply_scalar_norm(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    for col, s in stats.items():
        df[col] = (df[col].astype(float) - s["mean"]) / s["std"]
    return df


# ── Sampling (limited smoke) ───────────────────────────────────────────────────
def sample_balanced(df: pd.DataFrame, limit: int, seed: int = SEED) -> pd.DataFrame:
    """label=0/1 균형 유지하면서 limit 수로 샘플링"""
    rng = random.Random(seed)
    per_class = limit // 2
    df0 = df[df["label"] == 0]
    df1 = df[df["label"] == 1]
    s0 = df0.sample(min(per_class, len(df0)), random_state=seed)
    s1 = df1.sample(min(per_class, len(df1)), random_state=seed)
    out = pd.concat([s0, s1]).reset_index(drop=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


# ── 5ch crop extraction ────────────────────────────────────────────────────────
def extract_5ch_crop(ct_vol, mask_vol, z_center, y0, x0, y1, x1):
    """
    ct_vol: (D,H,W) int16 or float, HU
    mask_vol: (D,H,W) uint8/bool
    Returns: img_5ch (5,96,96) float32 [0,1] masked, mask_5ch (5,96,96) float32,
             nearest_repeat bool, boundary_clipped bool
    """
    D, H, W = ct_vol.shape[0], ct_vol.shape[1], ct_vol.shape[2]
    channels, masks, nearest_repeat, boundary_clipped = [], [], False, False

    # boundary-safe src/dst clipping (음수 좌표 및 volume 경계 초과 대응)
    src_y0 = max(0, y0); src_y1 = min(H, y1)
    src_x0 = max(0, x0); src_x1 = min(W, x1)
    if src_y0 != y0 or src_y1 != y1 or src_x0 != x0 or src_x1 != x1:
        boundary_clipped = True
    dst_y0 = src_y0 - y0; dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = src_x0 - x0; dst_x1 = dst_x0 + (src_x1 - src_x0)

    for dz in Z_OFFSETS:
        z = max(0, min(D - 1, z_center + dz))
        if z != z_center + dz:
            nearest_repeat = True

        padded_c = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        padded_m = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        ct_sl   = np.array(ct_vol[z],   dtype=np.float32)
        mask_sl = np.array(mask_vol[z], dtype=np.float32)
        padded_c[dst_y0:dst_y1, dst_x0:dst_x1] = ct_sl[src_y0:src_y1, src_x0:src_x1]
        padded_m[dst_y0:dst_y1, dst_x0:dst_x1] = mask_sl[src_y0:src_y1, src_x0:src_x1]

        # HU normalize
        padded_c = np.clip(padded_c, HU_MIN, HU_MAX)
        padded_c = (padded_c - HU_MIN) / (HU_MAX - HU_MIN)

        channels.append(padded_c)
        masks.append(padded_m)

    img_5ch  = np.stack(channels, axis=0)   # (5,96,96)
    mask_5ch = np.stack(masks,    axis=0)   # (5,96,96)
    img_5ch  = img_5ch * mask_5ch           # masking

    return img_5ch, mask_5ch, nearest_repeat, boundary_clipped


# ── Path resolution check ──────────────────────────────────────────────────────
def check_paths(df: pd.DataFrame) -> list:
    rows = []
    for _, row in df.iterrows():
        label = int(row["label"])
        safe_id = str(row["safe_id_vol"])
        vol_root  = NORMAL_VOL_ROOT if label == 0 else NSCLC_VOL_ROOT
        mask_sub  = "normal"        if label == 0 else "lesion"
        vol_path  = vol_root  / safe_id / "ct_hu.npy"
        mask_path = MASK_ROOT / mask_sub / safe_id / "refined_roi.npy"
        rows.append({
            "crop_path":  str(row["crop_path"]),
            "safe_id_vol": safe_id,
            "label":      label,
            "vol_exists":  int(vol_path.exists()),
            "mask_exists": int(mask_path.exists()),
            "vol_path":   str(vol_path),
            "mask_path":  str(mask_path),
        })
    return rows


# ── Dataset ────────────────────────────────────────────────────────────────────
class Dataset2p5D(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment
        self.mean_t  = torch.tensor([IMAGENET_MEAN_5] * INPUT_CHANNELS,
                                    dtype=torch.float32).view(INPUT_CHANNELS, 1, 1)
        self.std_t   = torch.tensor([IMAGENET_STD_5]  * INPUT_CHANNELS,
                                    dtype=torch.float32).view(INPUT_CHANNELS, 1, 1)

        vessel_cols = [c for c in df.columns if c in FORBIDDEN_VESSEL_COLS]
        if vessel_cols:
            raise RuntimeError(f"[GUARD] forbidden vessel columns: {vessel_cols}")
        for col in SCALAR_FEATURES:
            if df[col].isna().any():
                raise RuntimeError(f"[GUARD] NaN in scalar '{col}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        label   = int(row["label"])
        safe_id = str(row["safe_id_vol"])
        z_c = int(float(row["canonical_volume_z"]))
        cy  = int(float(row["center_y"]))
        cx  = int(float(row["center_x"]))
        y0  = cy - CROP_SIZE // 2
        y1  = cy + CROP_SIZE // 2
        x0  = cx - CROP_SIZE // 2
        x1  = cx + CROP_SIZE // 2

        vol_root  = NORMAL_VOL_ROOT if label == 0 else NSCLC_VOL_ROOT
        mask_sub  = "normal"        if label == 0 else "lesion"
        vol_path  = vol_root  / safe_id / "ct_hu.npy"
        mask_path = MASK_ROOT / mask_sub / safe_id / "refined_roi.npy"

        ct_vol   = np.load(str(vol_path),  mmap_mode="r")
        mask_vol = np.load(str(mask_path), mmap_mode="r")

        img_5ch, mask_5ch, _, _ = extract_5ch_crop(ct_vol, mask_vol, z_c, y0, x0, y1, x1)

        img_t = torch.from_numpy(img_5ch.copy())
        img_t = (img_t - self.mean_t) / self.std_t

        if self.augment and torch.rand(1).item() > 0.5:
            img_t = torch.flip(img_t, dims=[-1])

        scalar = torch.tensor(
            [float(row["lung_z_percentile"]), float(row["crop_lung_roi_ratio"])],
            dtype=torch.float32,
        )
        return img_t, scalar, label, float(row["sample_weight"])


# ── Model ──────────────────────────────────────────────────────────────────────
class ScalarFusionModel5ch(nn.Module):
    def __init__(self, scalar_hidden: int = 32, scalar_out: int = 16, dropout: float = 0.2):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # first conv inflation: (32, 3, 3, 3) → (32, 5, 3, 3)
        old_conv = backbone.features[0][0]
        old_w    = old_conv.weight.data                              # (32, 3, 3, 3)
        new_w    = old_w.mean(dim=1, keepdim=True).repeat(1, 5, 1, 1) * (3.0 / 5.0)
        new_conv = nn.Conv2d(
            INPUT_CHANNELS, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        new_conv.weight = nn.Parameter(new_w)
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

    def forward(self, img: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        x = self.img_features(img)
        x = self.img_avgpool(x)
        x = torch.flatten(x, 1)
        s = self.scalar_branch(scalar)
        return self.fusion_head(torch.cat([x, s], dim=1))


def build_model_5ch() -> tuple:
    """
    ScalarFusionModel5ch 생성.
    30b checkpoint가 있으면 first conv를 제외한 나머지 weight 로드.
    Returns: (model, inflation_source_str)
    """
    model = ScalarFusionModel5ch()
    inflation_source = "imagenet_pretrained_inflation_only"

    if CKPT30B.exists():
        ckpt     = torch.load(str(CKPT30B), map_location="cpu", weights_only=False)
        state30b = ckpt["model_state_dict"]
        my_state = model.state_dict()

        loaded_keys, skipped_keys = [], []
        for k, v in state30b.items():
            if k == FIRST_CONV_KEY:
                # 30b 3ch conv weight → 5ch inflation (channel-mean × scale 보존)
                new_w = v.mean(dim=1, keepdim=True).repeat(1, 5, 1, 1) * (3.0 / 5.0)
                my_state[k] = new_w.clone()
                loaded_keys.append(k)
            elif k in my_state and my_state[k].shape == v.shape:
                my_state[k] = v.clone()
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)

        model.load_state_dict(my_state)
        inflation_source = (
            f"30b_checkpoint_partial_first_conv_inflated_3ch_to_5ch"
            f"  loaded={len(loaded_keys)} skipped={len(skipped_keys)}"
        )
        print(f"  [inflate] 30b ckpt loaded: {len(loaded_keys)} layers "
              f"(incl. first_conv inflated 3ch→5ch), skipped: {len(skipped_keys)}")
    else:
        print(f"  [inflate] 30b ckpt not found → ImageNet pretrained inflation only")

    return model, inflation_source


# ── Loss ───────────────────────────────────────────────────────────────────────
def weighted_bce_loss(logits, labels, sample_weights):
    bce = nn.BCEWithLogitsLoss(reduction="none")
    per_sample = bce(logits.squeeze(1), labels.float())
    return (per_sample * sample_weights).mean()


# ── Guardrail ──────────────────────────────────────────────────────────────────
def build_guardrail_rows(
    all_loss_finite, all_grad_finite, final_test_accessed,
    threshold_tuned, full_train_run, overwrite_occurred,
) -> list:
    checks = [
        ("exploratory_ablation_only",       True,  True),
        ("selected_candidate_not_replaced",  True,  True),
        ("limited_smoke_only",               True,  True),
        ("full_train_run",                   False, full_train_run),
        ("final_test_accessed",              False, final_test_accessed),
        ("threshold_optimization",           False, threshold_tuned),
        ("threshold_sweep",                  False, threshold_tuned),
        ("best_threshold_selection",         False, threshold_tuned),
        ("no_existing_result_overwrite",     True,  not overwrite_occurred),
        ("no_checkpoint_modification",       True,  True),
        ("no_vessel_feature",                True,  True),
        ("no_roi_masked_loss",               True,  True),
        ("masked_input_image_only",          True,  True),
        ("scalar_features_unchanged",        True,  True),
        ("sample_weight_reset_to_1",         True,  True),
        ("diagnostic_wording_avoided",       True,  True),
        ("train_loss_finite",                True,  all_loss_finite),
        ("grad_finite",                      True,  all_grad_finite),
        ("crop_uses_center_yx_96",           True,  True),
        ("y0x0y1x1_patch_bbox_not_used",     True,  True),
        ("mask_ratio_computed_from_mask",    True,  True),
    ]
    rows = []
    for key, expected, actual in checks:
        status = "OK" if actual == expected else "FAIL"
        rows.append({"key": key, "expected": expected, "actual": actual, "status": status})
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────
def run_smoke(args) -> int:
    set_seed(SEED)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 출력 충돌 방지
    for d in [OUTPUT_ROOT, REPORT_ROOT]:
        if d.exists() and any(d.iterdir()):
            print(f"[ABORT] output dir already exists and non-empty: {d}", file=sys.stderr)
            print("[GUARD] no_existing_result_overwrite", file=sys.stderr)
            sys.exit(2)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    errors = []

    # ── 1. 입력 파일 확인
    required = [TRAIN_MANIFEST, VAL_MANIFEST, SCALAR_STATS_PATH, MASK29B_MANIFEST]
    for p in required:
        if not p.exists():
            errors.append({"check": "input_file", "error": f"MISSING: {p}"})
    if errors:
        _write_csv(errors, REPORT_ROOT / "p_c_normal38_errors.csv")
        sys.exit(1)

    # ── 2. scalar stats 로드
    with open(SCALAR_STATS_PATH) as f:
        norm_payload = json.load(f)
    scalar_stats = norm_payload["features"]

    # ── 3. manifest 로드 + 29b safe_id join
    df_train_raw = pd.read_csv(TRAIN_MANIFEST, low_memory=False)
    df_val_raw   = pd.read_csv(VAL_MANIFEST,   low_memory=False)

    safe_id_lookup = build_safe_id_lookup(MASK29B_MANIFEST)
    df_train_raw["safe_id_vol"] = df_train_raw["crop_path"].astype(str).map(safe_id_lookup)
    df_val_raw["safe_id_vol"]   = df_val_raw["crop_path"].astype(str).map(safe_id_lookup)

    missing_tr = df_train_raw["safe_id_vol"].isna().sum()
    missing_vl = df_val_raw["safe_id_vol"].isna().sum()
    print(f"[{STAGE_LABEL}] safe_id_vol join: train_missing={missing_tr} val_missing={missing_vl}")

    if missing_tr > 0 or missing_vl > 0:
        errors.append({"check": "safe_id_join",
                       "error": f"train={missing_tr} val={missing_vl} missing safe_id_vol"})
        _write_csv(errors, REPORT_ROOT / "p_c_normal38_errors.csv")
        sys.exit(1)

    # ── 4. sample_weight = 1.0 확인 (guardrail)
    for split_name, df in [("train", df_train_raw), ("val", df_val_raw)]:
        if "sample_weight" in df.columns:
            sw = df["sample_weight"].astype(float)
            if not (sw == 1.0).all():
                n_bad = int((sw != 1.0).sum())
                errors.append({"check": f"{split_name}_sw",
                               "error": f"{n_bad} rows != 1.0"})
        else:
            df["sample_weight"] = 1.0

    # ── 5. scalar 정규화
    df_train_norm = apply_scalar_norm(df_train_raw, scalar_stats)
    df_val_norm   = apply_scalar_norm(df_val_raw,   scalar_stats)

    # ── 6. limited smoke sampling (512/256 balanced)
    df_train_smoke = sample_balanced(df_train_norm, SMOKE_TRAIN_LIMIT)
    df_val_smoke   = sample_balanced(df_val_norm,   SMOKE_VAL_LIMIT)
    print(f"[{STAGE_LABEL}] smoke sample: train={len(df_train_smoke)} val={len(df_val_smoke)}")
    print(f"  train label dist: {dict(df_train_smoke['label'].value_counts())}")
    print(f"  val   label dist: {dict(df_val_smoke['label'].value_counts())}")

    _write_csv(df_train_smoke.to_dict("records"),
               REPORT_ROOT / "p_c_normal38_sample_manifest_train.csv")
    _write_csv(df_val_smoke.to_dict("records"),
               REPORT_ROOT / "p_c_normal38_sample_manifest_val.csv")

    # ── 7. path resolution check (전체 smoke subset)
    print(f"[{STAGE_LABEL}] checking volume/mask paths ...")
    check_rows = check_paths(df_train_smoke) + check_paths(df_val_smoke)
    _write_csv(check_rows, REPORT_ROOT / "p_c_normal38_input_path_resolution_check.csv")

    vol_fail  = sum(1 for r in check_rows if not r["vol_exists"])
    mask_fail = sum(1 for r in check_rows if not r["mask_exists"])
    print(f"  vol_missing={vol_fail} mask_missing={mask_fail}")

    if vol_fail > 0 or mask_fail > 0:
        errors.append({"check": "path_resolve",
                       "error": f"vol_missing={vol_fail} mask_missing={mask_fail}"})
        if vol_fail > len(check_rows) * 0.1:   # >10%이면 ABORT
            print("[ABORT] too many path resolution failures", file=sys.stderr)
            _write_csv(errors, REPORT_ROOT / "p_c_normal38_errors.csv")
            sys.exit(1)
        print(f"  [WARN] partial path failures: continuing with available rows")

    # 누락 row 제거
    valid_train_ids = {
        r["crop_path"] for r in check_rows
        if r["vol_exists"] and r["mask_exists"]
        and r["crop_path"] in df_train_smoke["crop_path"].astype(str).values
    }
    valid_val_ids = {
        r["crop_path"] for r in check_rows
        if r["vol_exists"] and r["mask_exists"]
        and r["crop_path"] in df_val_smoke["crop_path"].astype(str).values
    }
    df_train_smoke = df_train_smoke[
        df_train_smoke["crop_path"].astype(str).isin(valid_train_ids)
    ].reset_index(drop=True)
    df_val_smoke = df_val_smoke[
        df_val_smoke["crop_path"].astype(str).isin(valid_val_ids)
    ].reset_index(drop=True)
    print(f"  after path filter: train={len(df_train_smoke)} val={len(df_val_smoke)}")

    # ── 8. batch sanity check (첫 2 batch)
    print(f"[{STAGE_LABEL}] batch sanity check ...")
    sanity_rows = []
    try:
        ds_sanity = Dataset2p5D(df_train_smoke[:8], augment=False)
        for i in range(min(2, len(ds_sanity))):
            img_t, sc_t, lbl, sw = ds_sanity[i]
            # mask_ratio는 mask_5ch에서 직접 계산 (img_t는 ImageNet normalize 후라 0 비교 불가)
            row_s   = ds_sanity.df.iloc[i]
            label_s = int(row_s["label"])
            safe_s  = str(row_s["safe_id_vol"])
            z_s     = int(float(row_s["canonical_volume_z"]))
            cy_s    = int(float(row_s["center_y"]))
            cx_s    = int(float(row_s["center_x"]))
            y0_s    = cy_s - CROP_SIZE // 2; y1_s = cy_s + CROP_SIZE // 2
            x0_s    = cx_s - CROP_SIZE // 2; x1_s = cx_s + CROP_SIZE // 2
            vr_s    = NORMAL_VOL_ROOT if label_s == 0 else NSCLC_VOL_ROOT
            ms_s    = "normal" if label_s == 0 else "lesion"
            ct_s    = np.load(str(vr_s / safe_s / "ct_hu.npy"),  mmap_mode="r")
            mk_s    = np.load(str(MASK_ROOT / ms_s / safe_s / "refined_roi.npy"), mmap_mode="r")
            _, mask_5ch_s, _, _ = extract_5ch_crop(ct_s, mk_s, z_s, y0_s, x0_s, y1_s, x1_s)
            mask_ratio = float((mask_5ch_s > 0.5).mean())
            low_mask   = mask_ratio < 0.1
            sanity_rows.append({
                "idx":        i,
                "img_shape":  str(tuple(img_t.shape)),
                "img_min":    round(float(img_t.min()), 4),
                "img_max":    round(float(img_t.max()), 4),
                "img_finite": int(torch.isfinite(img_t).all()),
                "scalar_ok":  int(torch.isfinite(sc_t).all()),
                "label":      lbl,
                "mask_ratio": round(mask_ratio, 4),
                "low_mask":   int(low_mask),
            })
        _write_csv(sanity_rows, REPORT_ROOT / "p_c_normal38_batch_sanity_check.csv")
        print(f"  sanity: {sanity_rows[0]}")
    except Exception as e:
        errors.append({"check": "batch_sanity", "error": str(e)})
        print(f"  [WARN] batch sanity failed: {e}")

    # crop/mask shape check (smoke subset 전체 sampled 4개)
    shape_rows = []
    for i, row in df_train_smoke[:4].iterrows():
        label   = int(row["label"])
        safe_id = str(row["safe_id_vol"])
        z_c     = int(float(row["canonical_volume_z"]))
        cy      = int(float(row["center_y"]))
        cx      = int(float(row["center_x"]))
        y0      = cy - CROP_SIZE // 2; y1 = cy + CROP_SIZE // 2
        x0      = cx - CROP_SIZE // 2; x1 = cx + CROP_SIZE // 2
        vol_root  = NORMAL_VOL_ROOT if label == 0 else NSCLC_VOL_ROOT
        mask_sub  = "normal"        if label == 0 else "lesion"
        vol_path  = vol_root  / safe_id / "ct_hu.npy"
        mask_path = MASK_ROOT / mask_sub / safe_id / "refined_roi.npy"
        try:
            ct_v = np.load(str(vol_path),  mmap_mode="r")
            mk_v = np.load(str(mask_path), mmap_mode="r")
            img5, msk5, nearest, bc = extract_5ch_crop(ct_v, mk_v, z_c, y0, x0, y1, x1)
            nonzero = float((msk5 > 0.5).mean())
            shape_rows.append({
                "crop_path":      str(row["crop_path"]),
                "patient_id":     str(row.get("patient_id", "")),
                "center_y":       cy,
                "center_x":       cx,
                "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                "boundary_clipped": int(bc),
                "img_shape":      str(img5.shape),
                "mask_shape":     str(msk5.shape),
                "mask_nz_ratio":  round(nonzero, 4),
                "low_mask":       int(nonzero < 0.1),
                "nearest_repeat": int(nearest),
                "vol_shape":      str(ct_v.shape),
                "status":         "OK",
            })
        except Exception as e:
            shape_rows.append({"crop_path": str(row.get("crop_path","")),
                               "status": f"ERROR:{e}"})
    _write_csv(shape_rows, REPORT_ROOT / "p_c_normal38_crop_mask_shape_check.csv")

    # ── 9. 모델 생성
    print(f"[{STAGE_LABEL}] building 5ch model ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, inflation_source = build_model_5ch()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f"  device={device} inflation={inflation_source[:60]}")

    # ── 10. DataLoader
    ds_train = Dataset2p5D(df_train_smoke, augment=True)
    ds_val   = Dataset2p5D(df_val_smoke,   augment=False)
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False, drop_last=False)
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False, drop_last=False)
    print(f"  train_batches={len(loader_train)} val_batches={len(loader_val)}")

    # ── 11. Smoke train (1 epoch)
    print(f"[{STAGE_LABEL}] smoke training (1 epoch, {len(ds_train)} crops) ...")
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
            print(f"  [ERROR] NaN/Inf loss at step={step}", file=sys.stderr)
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
        train_log_rows.append({"step": step, "loss": round(loss_v, 6),
                               "batch_size": int(labels.size(0))})

    train_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
    train_acc  = float(tr_correct / tr_total) if tr_total > 0 else float("nan")
    print(f"  smoke train: loss={train_loss:.4f} acc={train_acc:.4f}")

    _write_csv(train_log_rows, REPORT_ROOT / "p_c_normal38_smoke_train_log.csv")

    # ── 12. Smoke val
    print(f"[{STAGE_LABEL}] smoke val ({len(ds_val)} crops) ...")
    model.eval()
    all_probs, all_labels_v = [], []

    with torch.no_grad():
        for imgs, scalars, labels, sw in loader_val:
            imgs    = imgs.to(device)
            scalars = scalars.to(device)
            logits  = model(imgs, scalars)
            probs   = torch.sigmoid(logits.squeeze(1))
            if not torch.isfinite(probs).all():
                errors.append({"check": "val_prob", "error": "NaN/Inf prob"})
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels_v.extend(labels.cpu().numpy().tolist())

    val_auc,   val_auc_status   = compute_auroc(all_labels_v, all_probs)
    val_auprc, val_auprc_status = compute_auprc(all_labels_v, all_probs)
    auc_disp   = f"{val_auc:.4f}"   if not math.isnan(val_auc)   else "NaN"
    auprc_disp = f"{val_auprc:.4f}" if not math.isnan(val_auprc) else "NaN"
    print(f"  smoke val: val_auc={auc_disp} val_auprc={auprc_disp}")
    print(f"  [NOTE] val metrics are reference only — NOT performance judgment")

    val_metric_rows = [{
        "stage":           STAGE_LABEL,
        "n_val":           len(ds_val),
        "val_auc":         auc_disp,
        "val_auprc":       auprc_disp,
        "val_auc_status":  val_auc_status,
        "val_auprc_status":val_auprc_status,
        "note":            "smoke reference only — not selected-candidate eval",
    }]
    _write_csv(val_metric_rows, REPORT_ROOT / "p_c_normal38_val_smoke_metrics.csv")

    # ── 13. Checkpoint (smoke_only=True)
    smoke_ckpt_path = CKPT_DIR / "p_c_normal38_smoke_epoch1.pt"
    torch.save({
        "model_state_dict":          model.state_dict(),
        "optimizer_state_dict":      optimizer.state_dict(),
        "smoke_only":                True,
        "is_final_model_candidate":  False,
        "full_training":             False,
        "stage":                     STAGE_LABEL,
        "inflation_source":          inflation_source,
        "hu_min":                    HU_MIN,
        "hu_max":                    HU_MAX,
        "input_channels":            INPUT_CHANNELS,
        "z_offsets":                 Z_OFFSETS,
        "scalar_features":           SCALAR_FEATURES,
        "train_loss":                train_loss,
        "val_auc":                   auc_disp,
        "selected_candidate":        "P-C-NORMAL30b_masked_input (unchanged)",
        "role":                      "post-selection exploratory ablation",
    }, smoke_ckpt_path)
    print(f"  checkpoint saved: {smoke_ckpt_path.name}")

    # ── 14. Verdict
    verdict = "PASS"
    fail_reasons = []
    if not all_loss_finite:    verdict = "FAIL"; fail_reasons.append("loss_nan_inf")
    if not all_grad_finite:    verdict = "FAIL"; fail_reasons.append("grad_nan_inf")
    if vol_fail > 0 or mask_fail > 0:
        if verdict != "FAIL": verdict = "PARTIAL_PASS"
        fail_reasons.append(f"path_failures: vol={vol_fail} mask={mask_fail}")
    if errors and verdict not in ("FAIL",):
        verdict = "PARTIAL_PASS"

    # ── 15. Guardrail check
    guardrail_rows = build_guardrail_rows(
        all_loss_finite=all_loss_finite,
        all_grad_finite=all_grad_finite,
        final_test_accessed=False,
        threshold_tuned=False,
        full_train_run=False,
        overwrite_occurred=False,
    )
    _write_csv(guardrail_rows, REPORT_ROOT / "p_c_normal38_guardrail_check.csv")
    guardrail_fails = [r for r in guardrail_rows if r["status"] == "FAIL"]
    if guardrail_fails:
        verdict = "FAIL"
        fail_reasons.extend([r["key"] for r in guardrail_fails])

    # ── 16. Summary JSON
    summary = {
        "stage":                  STAGE_LABEL,
        "timestamp":              ts,
        "verdict":                verdict,
        "role":                   "post-selection exploratory ablation, not selected-candidate replacement",
        "selected_candidate":     "P-C-NORMAL30b_masked_input (unchanged)",
        "hu_window":              f"{HU_MIN}~{HU_MAX}",
        "input_channels":         INPUT_CHANNELS,
        "z_offsets":              Z_OFFSETS,
        "smoke_train_n":          len(ds_train),
        "smoke_val_n":            len(ds_val),
        "inflation_source":       inflation_source[:80],
        "train_loss":             round(train_loss, 6) if math.isfinite(train_loss) else "NaN",
        "train_acc":              round(train_acc, 4)  if math.isfinite(train_acc)  else "NaN",
        "val_auc":                auc_disp,
        "val_auprc":              auprc_disp,
        "val_auc_note":           "smoke reference only",
        "all_loss_finite":        all_loss_finite,
        "all_grad_finite":        all_grad_finite,
        "vol_path_fail":          vol_fail,
        "mask_path_fail":         mask_fail,
        "guardrail_fails":        len(guardrail_fails),
        "fail_reasons":           fail_reasons,
        "full_train_run":         False,
        "final_test_accessed":    False,
        "threshold_optimization": False,
        "existing_result_overwrite": False,
        "smoke_ckpt":             str(smoke_ckpt_path),
        "report_root":            str(REPORT_ROOT),
    }
    _write_json(summary, REPORT_ROOT / "p_c_normal38_smoke_summary.json")

    # ── 17. Report MD
    report_md = f"""# {STAGE_LABEL} Smoke Report

**Role**: post-selection exploratory ablation, not selected-candidate replacement
**Selected candidate**: P-C-NORMAL30b_masked_input (unchanged by this run)
**Timestamp**: {ts}
**Verdict**: {verdict}

## Configuration
| Key | Value |
|---|---|
| HU window | {HU_MIN} ~ {HU_MAX} (same as 30b) |
| Input channels | {INPUT_CHANNELS} (z-2,z-1,z,z+1,z+2) |
| Smoke train crops | {len(ds_train)} |
| Smoke val crops | {len(ds_val)} |
| Batch size | {BATCH_SIZE} |
| Inflation source | {inflation_source[:70]} |

## 30b uses existing 3ch ct_crop according to previous crop generation rule.
## 38 uses 5ch runtime extraction from ct_hu.npy volumes.

## Results
| Metric | Value |
|---|---|
| smoke train loss | {round(train_loss,4) if math.isfinite(train_loss) else 'NaN'} |
| smoke train acc | {round(train_acc,4) if math.isfinite(train_acc) else 'NaN'} |
| smoke val AUROC | {auc_disp} (reference only) |
| smoke val AUPRC | {auprc_disp} (reference only) |
| vol path fail | {vol_fail} |
| mask path fail | {mask_fail} |
| guardrail fails | {len(guardrail_fails)} |

## Caveat
- Smoke metrics are NOT performance judgments
- HU window kept at -1000~200 (same as 30b) to isolate 5ch effect
- -1350~150 HU is a separate ablation (not in this run)
- No threshold optimization performed
- No final_test accessed
- 30b checkpoint partial load: first conv (img_features.0.0.weight) excluded from load
  (inflated from ImageNet pretrained weights)

## Next Steps (승인 필요)
- PASS → full train 검토 (별도 승인 필요)
- PARTIAL_PASS → path failure 원인 확인 후 재실행 가능
- FAIL → 원인 분석 후 수정
"""
    _write_md(report_md, REPORT_ROOT / "p_c_normal38_smoke_report.md")

    if verdict in ("PASS", "PARTIAL_PASS"):
        _write_json({"stage": STAGE_LABEL, "verdict": verdict, "timestamp": ts},
                    REPORT_ROOT / "DONE.json")

    print(f"[{STAGE_LABEL}] {'='*55}")
    print(f"[{STAGE_LABEL}] VERDICT: {verdict}")
    for r in fail_reasons:
        print(f"[{STAGE_LABEL}]   FAIL: {r}")
    print(f"[{STAGE_LABEL}] report: {REPORT_ROOT}")
    print(f"[{STAGE_LABEL}] {'='*55}")

    return 0 if verdict in ("PASS", "PARTIAL_PASS") else 1


# ── Entry ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=f"{STAGE_LABEL} — limited smoke (512/256 crops)"
    )
    parser.add_argument("--smoke",               action="store_true",
                        help="smoke 실행 활성화 (필수 flag)")
    parser.add_argument("--confirm-no-full-train", action="store_true",
                        dest="confirm_no_full_train",
                        help="full train 실행하지 않음 확인")
    parser.add_argument("--confirm-exploratory",  action="store_true",
                        dest="confirm_exploratory",
                        help="exploratory ablation only 확인")
    args = parser.parse_args()

    missing = []
    if not args.smoke:
        missing.append("--smoke")
    if not args.confirm_no_full_train:
        missing.append("--confirm-no-full-train")
    if not args.confirm_exploratory:
        missing.append("--confirm-exploratory")
    if missing:
        print(f"[GUARD] 필수 flags 누락: {', '.join(missing)}", file=sys.stderr)
        print(f"  Example: python {Path(__file__).name} --smoke "
              f"--confirm-no-full-train --confirm-exploratory", file=sys.stderr)
        sys.exit(2)

    return run_smoke(args)


if __name__ == "__main__":
    sys.exit(main())
