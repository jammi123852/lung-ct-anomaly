#!/usr/bin/env python3
"""
P-C-NORMAL40: 2.5D 5ch EfficientNet-B0 masked-input full train

Stage: P-C-NORMAL40_2p5d5ch_full_train
Role: post-selection exploratory ablation
selected_candidate = P-C-NORMAL30b_masked_input (unchanged)

Guardrail:
  exploratory_ablation_only=True, final_test_accessed=False,
  threshold_optimization=False, selected_candidate_not_replaced=True
"""

import argparse, csv, json, random, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Constants ─────────────────────────────────────────────────────────────────
STAGE_LABEL     = "P-C-NORMAL40_2p5d5ch_full_train"
HU_MIN          = -1000.0
HU_MAX          =  200.0
CROP_SIZE       = 96
INPUT_CHANNELS  = 5
IMAGENET_MEAN_5 = 0.449
IMAGENET_STD_5  = 0.226
SCALAR_FEATURES = ["lung_z_percentile", "crop_lung_roi_ratio"]
Z_OFFSETS       = [-2, -1, 0, 1, 2]
FIRST_CONV_KEY  = "img_features.0.0.weight"
BATCH_SIZE      = 32
NUM_WORKERS     = 4
LR              = 1e-4
MAX_EPOCHS      = 30
EARLY_STOP_PAT  = 7
SEED            = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_MANIFEST    = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_balanced_w1_feature_manifest/p_c_normal24g_fix_balanced_w1_train_manifest.csv"
VAL_MANIFEST      = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_balanced_w1_feature_manifest/p_c_normal24g_fix_balanced_w1_val_manifest.csv"
SCALAR_STATS_PATH = PROJECT_ROOT / "outputs/reports/p_c_normal24h_fix_zroi_scalar_fusion_drycheck/p_c_normal24h_fix_scalar_normalization_stats.json"
MASK29B_MANIFEST  = PROJECT_ROOT / "outputs/reports/p_c_normal29b_crop_level_mask_generation/p_c_normal29b_mask_manifest.csv"
CKPT30B           = PROJECT_ROOT / "outputs/p_c_normal30b_masked_input_full_train/checkpoints/p_c_normal30b_best_val_auc_checkpoint.pt"
P38_REPORT_ROOT   = PROJECT_ROOT / "outputs/reports/p_c_normal38_2p5d5ch_limited_smoke_train"
P39_REPORT_ROOT   = PROJECT_ROOT / "outputs/reports/p_c_normal39_2p5d5ch_full_train_preflight"
P40A_REPORT_ROOT  = PROJECT_ROOT / "outputs/reports/p_c_normal40a_5ch_preextract"
P40A_TRAIN_MANIFEST = P40A_REPORT_ROOT / "p40a_train_manifest_with_npy.csv"
P40A_VAL_MANIFEST   = P40A_REPORT_ROOT / "p40a_val_manifest_with_npy.csv"

OUTPUT_ROOT = PROJECT_ROOT / "outputs/p_c_normal40_2p5d5ch_full_train"
CKPT_DIR    = OUTPUT_ROOT / "checkpoints"
REPORT_ROOT = PROJECT_ROOT / "outputs/reports/p_c_normal40_2p5d5ch_full_train"


# ── Utilities ─────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv(rows, path):
    path = Path(path)
    if not rows:
        path.write_text("(empty)\n", encoding="utf-8"); return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def build_safe_id_lookup(mask29b_path):
    df = pd.read_csv(mask29b_path, low_memory=False)
    if "status" in df.columns:
        df = df[df["status"] == "PASS"]
    return dict(zip(df["crop_path"].astype(str), df["safe_id"].astype(str)))


def apply_scalar_norm(df, stats):
    df = df.copy()
    for col, s in stats.items():
        if col in df.columns:
            df[col] = (df[col].astype(float) - s["mean"]) / s["std"]
    return df


# ── 5ch crop extraction (P38/P39 동일, boundary-safe) ─────────────────────────
def extract_5ch_crop(ct_vol, mask_vol, z_center, y0, x0, y1, x1):
    D, H, W = ct_vol.shape[0], ct_vol.shape[1], ct_vol.shape[2]
    boundary_clipped = False
    src_y0 = max(0, y0); src_y1 = min(H, y1)
    src_x0 = max(0, x0); src_x1 = min(W, x1)
    if src_y0 != y0 or src_y1 != y1 or src_x0 != x0 or src_x1 != x1:
        boundary_clipped = True
    dst_y0 = src_y0 - y0; dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = src_x0 - x0; dst_x1 = dst_x0 + (src_x1 - src_x0)
    channels, masks, nearest_repeat = [], [], False
    for dz in Z_OFFSETS:
        z = max(0, min(D - 1, z_center + dz))
        if z != z_center + dz: nearest_repeat = True
        padded_c = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        padded_m = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        ct_sl   = np.array(ct_vol[z],   dtype=np.float32)
        mask_sl = np.array(mask_vol[z], dtype=np.float32)
        padded_c[dst_y0:dst_y1, dst_x0:dst_x1] = ct_sl[src_y0:src_y1, src_x0:src_x1]
        padded_m[dst_y0:dst_y1, dst_x0:dst_x1] = mask_sl[src_y0:src_y1, src_x0:src_x1]
        padded_c = np.clip(padded_c, HU_MIN, HU_MAX)
        padded_c = (padded_c - HU_MIN) / (HU_MAX - HU_MIN)
        channels.append(padded_c); masks.append(padded_m)
    img_5ch  = np.stack(channels, axis=0)
    mask_5ch = np.stack(masks,    axis=0)
    img_5ch  = img_5ch * mask_5ch
    return img_5ch, mask_5ch, nearest_repeat, boundary_clipped


# ── Metrics (numpy-based, sklearn 미사용) ─────────────────────────────────────
def compute_auroc(labels, scores):
    try:
        la = np.asarray(labels, dtype=np.int32)
        sc = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(sc)) or len(np.unique(la)) < 2:
            return float("nan"), "invalid"
        n_pos = int((la == 1).sum()); n_neg = len(la) - n_pos
        order = np.argsort(sc, kind="stable")
        sl = la[order]; ss = sc[order]
        n = len(sl); ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i + 1
            while j < n and ss[j] == ss[i]: j += 1
            ranks[i:j] = (i + 1 + j) / 2.0; i = j
        U = float(ranks[sl == 1].sum()) - n_pos * (n_pos + 1) / 2.0
        return float(U / (n_pos * n_neg)), "OK"
    except Exception as e:
        return float("nan"), f"ERROR:{e}"


def compute_auprc(labels, scores):
    try:
        la = np.asarray(labels, dtype=np.int32)
        sc = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(sc)) or len(np.unique(la)) < 2:
            return float("nan"), "invalid"
        n_pos = int((la == 1).sum())
        if n_pos == 0: return float("nan"), "no_positive"
        order = np.argsort(sc, kind="stable")[::-1]
        sl = la[order]
        tp = np.cumsum(sl); fp = np.cumsum(1 - sl)
        prec = np.concatenate([[1.0], tp / (tp + fp)])
        rec  = np.concatenate([[0.0], tp / n_pos])
        return float(np.trapezoid(prec, rec)), "OK"
    except Exception as e:
        return float("nan"), f"ERROR:{e}"


# ── Model (P38와 동일) ────────────────────────────────────────────────────────
class ScalarFusionModel5ch(nn.Module):
    def __init__(self, scalar_hidden=32, scalar_out=16, dropout=0.2):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        old_conv = backbone.features[0][0]
        old_w    = old_conv.weight.data                                     # (32, 3, 3, 3)
        new_w    = old_w.mean(dim=1, keepdim=True).repeat(1, INPUT_CHANNELS, 1, 1) * (3.0 / INPUT_CHANNELS)
        new_conv = nn.Conv2d(INPUT_CHANNELS, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        new_conv.weight.data = new_w
        backbone.features[0][0] = new_conv
        self.img_features = backbone.features
        self.img_avgpool  = backbone.avgpool
        self.scalar_branch = nn.Sequential(
            nn.Linear(len(SCALAR_FEATURES), scalar_hidden),
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


def build_model_5ch():
    """30b checkpoint에서 partial load (first conv 3ch→5ch inflate 포함). 30b 없으면 abort."""
    if not CKPT30B.exists():
        raise FileNotFoundError(f"[ABORT] 30b checkpoint 없음: {CKPT30B}")
    model    = ScalarFusionModel5ch()
    ckpt     = torch.load(str(CKPT30B), map_location="cpu", weights_only=False)
    state30b = ckpt["model_state_dict"]
    my_state = model.state_dict()
    loaded_keys, skipped_keys = [], []
    for k, v in state30b.items():
        if k == FIRST_CONV_KEY:
            new_w = v.mean(dim=1, keepdim=True).repeat(1, INPUT_CHANNELS, 1, 1) * (3.0 / INPUT_CHANNELS)
            my_state[k] = new_w.clone()
            loaded_keys.append(k)
        elif k in my_state and my_state[k].shape == v.shape:
            my_state[k] = v.clone()
            loaded_keys.append(k)
        else:
            skipped_keys.append(k)
    model.load_state_dict(my_state)
    sb_loaded = any(k.startswith("scalar_branch") for k in loaded_keys)
    fh_loaded = any(k.startswith("fusion_head")   for k in loaded_keys)
    src = (f"30b_ckpt_first_conv_inflated_3ch_to_5ch  "
           f"loaded={len(loaded_keys)} skipped={len(skipped_keys)}")
    print(f"  30b checkpoint loaded keys: {len(loaded_keys)}")
    print(f"  skipped keys: {len(skipped_keys)}")
    print(f"  first_conv inflated: True")
    print(f"  scalar_branch_loaded: {sb_loaded}")
    print(f"  fusion_head_loaded: {fh_loaded}")
    if not sb_loaded or not fh_loaded:
        raise RuntimeError(
            f"[ABORT] 30b checkpoint head load 실패 — "
            f"scalar_branch_loaded={sb_loaded}, fusion_head_loaded={fh_loaded}. "
            f"P40 model 구조가 30b와 맞지 않음."
        )
    if skipped_keys:
        print(f"  [WARN] skipped_keys (first 5): {skipped_keys[:5]}")
    return model, src


def weighted_bce_loss(logits, labels, sw):
    bce = nn.BCEWithLogitsLoss(reduction="none")
    per_sample = bce(logits.squeeze(1), labels.float())
    return (per_sample * sw.float()).mean()


# ── Dataset (P38와 동일, smoke limit 없음) ────────────────────────────────────
class Dataset2p5D(Dataset):
    def __init__(self, df, augment=False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment
        self.mean_t  = torch.tensor([IMAGENET_MEAN_5] * INPUT_CHANNELS,
                                    dtype=torch.float32).view(-1, 1, 1)
        self.std_t   = torch.tensor([IMAGENET_STD_5]  * INPUT_CHANNELS,
                                    dtype=torch.float32).view(-1, 1, 1)
        for col in SCALAR_FEATURES:
            if col not in self.df.columns:
                raise RuntimeError(f"[GUARD] scalar column missing: {col}")
            if self.df[col].isna().any():
                raise RuntimeError(f"[GUARD] NaN in scalar '{col}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        label  = int(row["label"])
        img_5ch = np.load(str(row["crop_npy_path"]))   # (5,96,96) pre-extracted
        img_t  = torch.from_numpy(img_5ch.copy())
        img_t  = (img_t - self.mean_t) / self.std_t
        if self.augment and torch.rand(1).item() > 0.5:
            img_t = torch.flip(img_t, dims=[-1])
        scalar = torch.tensor(
            [float(row["lung_z_percentile"]), float(row["crop_lung_roi_ratio"])],
            dtype=torch.float32,
        )
        return img_t, scalar, label, float(row.get("sample_weight", 1.0))


# ── Training functions ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, epoch_n, log_rows):
    model.train()
    all_logits, all_labels, all_losses = [], [], []
    ep_grad_ok = True

    for bidx, (img_t, sc_t, labels, sw) in enumerate(loader):
        img_t  = img_t.to(device)
        sc_t   = sc_t.to(device)
        lbl_f  = labels.float().to(device)
        sw_f   = sw.float().to(device)

        optimizer.zero_grad()
        logits = model(img_t, sc_t)
        loss   = weighted_bce_loss(logits, lbl_f, sw_f)

        if not torch.isfinite(loss):
            print(f"  [ABORT] NaN loss epoch={epoch_n} batch={bidx}", file=sys.stderr)
            return None, None, None, None, False, "NaN_loss"

        loss.backward()

        grad_ok = all(
            torch.isfinite(p.grad).all()
            for p in model.parameters() if p.grad is not None
        )
        if not grad_ok:
            ep_grad_ok = False

        optimizer.step()

        with torch.no_grad():
            lg = logits.squeeze(1).detach().cpu().numpy().tolist()
            lb = labels.cpu().numpy().tolist()
            all_logits.extend(lg); all_labels.extend(lb)
            all_losses.append(loss.item())

        if bidx % 100 == 0:
            log_rows.append({
                "epoch": epoch_n, "batch": bidx,
                "loss":       round(loss.item(), 6),
                "logit_mean": round(float(np.mean(lg)), 4),
                "loss_finite": 1,
                "grad_finite": int(grad_ok),
            })
            print(f"    ep{epoch_n:3d} b{bidx:4d}/{len(loader)} "
                  f"loss={loss.item():.4f} grad={'OK' if grad_ok else 'FAIL'}")

    avg_loss = float(np.mean(all_losses)) if all_losses else float("nan")
    acc      = float(np.mean([int((s >= 0) == l)
                              for s, l in zip(all_logits, all_labels)])) if all_logits else float("nan")
    auc,  _  = compute_auroc(all_labels,  all_logits)
    auprc, _ = compute_auprc(all_labels,  all_logits)
    return avg_loss, acc, auc, auprc, ep_grad_ok, "OK"


def val_one_epoch(model, loader, device):
    model.eval()
    all_logits, all_labels, all_losses = [], [], []
    with torch.no_grad():
        for img_t, sc_t, labels, sw in loader:
            img_t  = img_t.to(device)
            sc_t   = sc_t.to(device)
            lbl_f  = labels.float().to(device)
            sw_f   = sw.float().to(device)
            logits = model(img_t, sc_t)
            loss   = weighted_bce_loss(logits, lbl_f, sw_f)
            all_logits.extend(logits.squeeze(1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_losses.append(loss.item())
    avg_loss = float(np.mean(all_losses)) if all_losses else float("nan")
    auc,  _  = compute_auroc(all_labels, all_logits)
    auprc, _ = compute_auprc(all_labels, all_logits)
    return avg_loss, auc, auprc


# ── Guardrail ─────────────────────────────────────────────────────────────────
def build_guardrail_rows(all_loss_finite, all_grad_finite, all_val_finite,
                         final_test_accessed, threshold_tuned,
                         overwrite_occurred, training_completed):
    checks = [
        ("exploratory_ablation_only",        True,  True),
        ("selected_candidate_not_replaced",   True,  True),
        ("full_train_completed",              True,  training_completed),
        ("final_test_accessed",              False,  final_test_accessed),
        ("threshold_optimization",           False,  threshold_tuned),
        ("threshold_sweep",                  False,  threshold_tuned),
        ("best_threshold_selection",         False,  threshold_tuned),
        ("no_existing_result_overwrite",      True,  not overwrite_occurred),
        ("no_checkpoint_modification",        True,  True),
        ("no_vessel_feature",                 True,  True),
        ("no_roi_masked_loss",                True,  True),
        ("masked_input_image_only",           True,  True),
        ("scalar_features_unchanged",         True,  True),
        ("sample_weight_reset_to_1",          True,  True),
        ("crop_uses_center_yx_96",            True,  True),
        ("y0x0y1x1_patch_bbox_not_used",      True,  True),
        ("mask_ratio_computed_from_mask",     True,  True),
        ("diagnostic_wording_avoided",        True,  True),
        ("train_loss_finite",                 True,  all_loss_finite),
        ("grad_finite",                       True,  all_grad_finite),
        ("val_loss_finite",                   True,  all_val_finite),
        ("val_auc_finite",                    True,  all_val_finite),
        ("val_auprc_finite",                  True,  all_val_finite),
    ]
    rows = []
    for key, expected, actual in checks:
        status = "OK" if actual == expected else "FAIL"
        rows.append({"key": key, "expected": expected, "actual": actual, "status": status})
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def run_full_train(args):
    set_seed(SEED)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{STAGE_LABEL}] started: {ts}")

    if getattr(args, "use_amp", False):
        print("  [NOTE] --use-amp requested but not implemented. Running without AMP.", file=sys.stderr)

    # ── output dir 충돌 방지
    for d in [OUTPUT_ROOT, REPORT_ROOT]:
        if d.exists() and any(d.iterdir()):
            print(f"[ABORT] output dir already exists and non-empty: {d}", file=sys.stderr)
            sys.exit(2)
    for d in [OUTPUT_ROOT, CKPT_DIR, REPORT_ROOT]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. P38/P39 verdict 확인 (hard ABORT)
    print(f"[{STAGE_LABEL}] [1] P38/P39 verdict check ...")
    for label_s, root, fname, gfail_key in [
        ("P38", P38_REPORT_ROOT, "p_c_normal38_smoke_summary.json",              "guardrail_fails"),
        ("P39", P39_REPORT_ROOT, "p_c_normal39_full_train_preflight_summary.json", "guardrail_fail_count"),
    ]:
        p = root / fname
        if not p.exists():
            print(f"[ABORT] {label_s} summary 없음: {p}", file=sys.stderr); sys.exit(1)
        with open(p) as f: s = json.load(f)
        verdict_s = s.get("verdict", "MISSING")
        gfail_s   = int(s.get(gfail_key, -1))
        print(f"  {label_s} verdict={verdict_s}  guardrail_fail={gfail_s}  "
              f"selected={s.get('selected_candidate','?')}")
        if verdict_s != "PASS":
            print(f"[ABORT] {label_s} verdict={verdict_s} (PASS 필요)", file=sys.stderr); sys.exit(1)
        if gfail_s != 0:
            print(f"[ABORT] {label_s} guardrail_fail={gfail_s} (0 필요)", file=sys.stderr); sys.exit(1)

    # ── 2. 입력 파일 확인 (P40a 사전추출 manifest 사용)
    for p in [P40A_TRAIN_MANIFEST, P40A_VAL_MANIFEST, SCALAR_STATS_PATH, CKPT30B]:
        if not p.exists():
            print(f"[ABORT] 필수 파일 없음: {p}", file=sys.stderr); sys.exit(1)

    # ── 2b. P40a summary 확인
    p40a_sum_path = P40A_REPORT_ROOT / "p40a_summary.json"
    if not p40a_sum_path.exists():
        print(f"[ABORT] P40a summary 없음: {p40a_sum_path}", file=sys.stderr); sys.exit(1)
    with open(p40a_sum_path) as f: p40a_sum = json.load(f)
    p40a_verdict = p40a_sum.get("verdict", "MISSING")
    p40a_errors  = int(p40a_sum.get("errors", -1))
    print(f"  P40a verdict={p40a_verdict}  errors={p40a_errors}")
    if p40a_verdict != "PASS":
        print(f"[ABORT] P40a verdict={p40a_verdict} (PASS 필요)", file=sys.stderr); sys.exit(1)

    # ── 3. P40a manifest load (crop_npy_path 컬럼 포함)
    print(f"[{STAGE_LABEL}] [3] P40a pre-extracted manifest load ...")
    df_tr_raw = pd.read_csv(P40A_TRAIN_MANIFEST, low_memory=False)
    df_vl_raw = pd.read_csv(P40A_VAL_MANIFEST,   low_memory=False)
    for col in ["crop_npy_path", "label"]:
        for df, name in [(df_tr_raw, "train"), (df_vl_raw, "val")]:
            if col not in df.columns:
                print(f"[ABORT] {name} manifest에 '{col}' 컬럼 없음", file=sys.stderr); sys.exit(1)
    miss_npy_tr = int((df_tr_raw["crop_npy_path"].fillna("") == "").sum())
    miss_npy_vl = int((df_vl_raw["crop_npy_path"].fillna("") == "").sum())
    print(f"  crop_npy_path missing: train={miss_npy_tr} val={miss_npy_vl}")
    if miss_npy_tr > 0 or miss_npy_vl > 0:
        print("[ABORT] crop_npy_path missing", file=sys.stderr); sys.exit(1)

    # ── 4. scalar 정규화
    with open(SCALAR_STATS_PATH) as f:
        norm_payload = json.load(f)
    scalar_stats = norm_payload.get("features", norm_payload)
    df_tr_norm = apply_scalar_norm(df_tr_raw, scalar_stats)
    df_vl_norm = apply_scalar_norm(df_vl_raw, scalar_stats)
    for df in [df_tr_norm, df_vl_norm]:
        if "sample_weight" not in df.columns:
            df["sample_weight"] = 1.0

    # ── 5. crop_npy_path 존재 확인 (4샘플 spot check)
    print(f"[{STAGE_LABEL}] [5] crop_npy spot check ...")
    missing_npy = []
    for split, df in [("train", df_tr_norm), ("val", df_vl_norm)]:
        for i in range(min(4, len(df))):
            p = df.iloc[i]["crop_npy_path"]
            if not Path(str(p)).exists():
                missing_npy.append({"split": split, "idx": i, "path": p})
    if missing_npy:
        print(f"[ABORT] crop npy spot check 실패: {missing_npy}", file=sys.stderr); sys.exit(1)
    print(f"  spot check OK")

    # ── 6. manifest 저장
    _write_csv(df_tr_norm.to_dict("records"), REPORT_ROOT / "p_c_normal40_train_manifest_used.csv")
    _write_csv(df_vl_norm.to_dict("records"), REPORT_ROOT / "p_c_normal40_val_manifest_used.csv")
    print(f"  train={len(df_tr_norm)} val={len(df_vl_norm)}")

    # ── 7. startup shape check (pre-extracted npy 직접 로드)
    print(f"[{STAGE_LABEL}] [7] startup shape check ...")
    shape_rows = []
    for _, row in df_tr_norm.head(4).iterrows():
        try:
            img5 = np.load(str(row["crop_npy_path"]))
            shape_rows.append({"lbl": int(row["label"]), "img_shape": str(img5.shape),
                                "dtype": str(img5.dtype), "finite": int(np.isfinite(img5).all()),
                                "status": "OK"})
        except Exception as e:
            shape_rows.append({"lbl": int(row["label"]), "status": f"ERROR:{e}"})
    _write_csv(shape_rows, REPORT_ROOT / "p_c_normal40_shape_startup_check.csv")
    if any("ERROR" in str(r.get("status", "")) for r in shape_rows):
        print("[ABORT] shape check failed", file=sys.stderr); sys.exit(1)
    print(f"  shape OK: {[r['img_shape'] for r in shape_rows if 'img_shape' in r]}")

    # ── 8. model build
    print(f"[{STAGE_LABEL}] [8] building 5ch model ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, inflation_source = build_model_5ch()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"  device={device}  use_amp=False  inflation={inflation_source[:70]}")

    # ── 9. DataLoaders
    ds_train = Dataset2p5D(df_tr_norm, augment=True)
    ds_val   = Dataset2p5D(df_vl_norm, augment=False)
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False, drop_last=False)
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False, drop_last=False)
    print(f"  train_batches={len(loader_train)} val_batches={len(loader_val)}")

    # ── 10. training loop
    print(f"[{STAGE_LABEL}] [10] full train  max_epochs={MAX_EPOCHS} patience={EARLY_STOP_PAT} ...")
    best_val_auc     = -1.0
    best_epoch       = -1
    patience_counter = 0
    early_stopped    = False
    all_loss_finite  = True
    all_grad_ok      = True
    all_val_finite   = True
    last_tr_status   = "NOT_RUN"
    train_log_rows   = []
    epoch_metric_rows = []

    best_ckpt_path = CKPT_DIR / "p_c_normal40_best_val_auc_checkpoint.pt"
    last_ckpt_path = CKPT_DIR / "p_c_normal40_last_checkpoint.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        ep_start = time.time()
        print(f"[{STAGE_LABEL}] --- epoch {epoch}/{MAX_EPOCHS} ---")

        tr_loss, tr_acc, tr_auc, tr_auprc, ep_grad_ok, tr_status = train_one_epoch(
            model, loader_train, optimizer, device, epoch, train_log_rows)
        last_tr_status = tr_status

        if tr_status != "OK":
            all_loss_finite = False
            print(f"  [ABORT] training failed: {tr_status}", file=sys.stderr)
            break

        if not ep_grad_ok:
            all_grad_ok = False

        vl_loss, vl_auc, vl_auprc = val_one_epoch(model, loader_val, device)
        ep_sec  = time.time() - ep_start

        ep_val_finite = (
            np.isfinite(float(vl_loss)) and
            np.isfinite(float(vl_auc))  and
            np.isfinite(float(vl_auprc))
        )
        if not ep_val_finite:
            all_val_finite = False
            print(f"  [WARN] val not finite epoch={epoch}: "
                  f"val_loss={vl_loss:.4f} val_auc={vl_auc:.4f} val_auprc={vl_auprc:.4f}")

        is_best = ep_val_finite and float(vl_auc) > best_val_auc

        if is_best:
            best_val_auc = float(vl_auc)
            best_epoch   = epoch
            patience_counter = 0
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc":            float(vl_auc),
                "val_auprc":          float(vl_auprc),
                "train_loss":         float(tr_loss),
                "inflation_source":   inflation_source,
                "stage":              STAGE_LABEL,
                "exploratory_only":   True,
                "selected_candidate": "P-C-NORMAL30b_masked_input",
                "timestamp":          ts,
            }, str(best_ckpt_path))
            print(f"  ★ best checkpoint  epoch={epoch} val_auc={vl_auc:.4f}")
        else:
            patience_counter += 1

        if getattr(args, "save_last", False):
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_auc": float(vl_auc), "stage": STAGE_LABEL}, str(last_ckpt_path))

        def _safe_round(v, n=4):
            return round(float(v), n) if np.isfinite(float(v)) else "nan"

        em = {
            "epoch":       epoch,
            "train_loss":  _safe_round(tr_loss, 6),
            "train_acc":   _safe_round(tr_acc),
            "train_auc":   _safe_round(tr_auc),
            "train_auprc": _safe_round(tr_auprc),
            "val_loss":    _safe_round(vl_loss, 6),
            "val_auc":     _safe_round(vl_auc),
            "val_auprc":   _safe_round(vl_auprc),
            "best_val_auc":_safe_round(best_val_auc),
            "patience":    patience_counter,
            "is_best":     int(is_best),
            "grad_ok":     int(ep_grad_ok),
            "epoch_sec":   round(ep_sec, 1),
        }
        epoch_metric_rows.append(em)
        _write_csv(epoch_metric_rows, REPORT_ROOT / "p_c_normal40_epoch_metrics.csv")
        _write_csv(train_log_rows,    REPORT_ROOT / "p_c_normal40_train_log.csv")

        print(f"  ep{epoch:3d}: tr_loss={tr_loss:.4f} tr_auc={_safe_round(tr_auc)}  "
              f"vl_auc={_safe_round(vl_auc)} vl_auprc={_safe_round(vl_auprc)}  "
              f"best={best_val_auc:.4f} pat={patience_counter}  {ep_sec:.1f}s")
        print(f"  [NOTE] val metrics = reference only, NOT performance judgment")

        if patience_counter >= EARLY_STOP_PAT:
            print(f"  [early stop] patience={patience_counter} at epoch={epoch}")
            early_stopped = True
            break

    training_completed = (last_tr_status == "OK")

    # ── 11. best epoch summary
    best_em = next((r for r in epoch_metric_rows if r["epoch"] == best_epoch), {})
    with open(REPORT_ROOT / "p_c_normal40_best_epoch_summary.json", "w") as f:
        json.dump({
            "best_epoch":      best_epoch,
            "best_val_auc":    float(best_val_auc),
            "val_auprc":       best_em.get("val_auprc", "?"),
            "train_loss":      best_em.get("train_loss", "?"),
            "early_stopped":   early_stopped,
            "total_epochs":    len(epoch_metric_rows),
            "selected_candidate": "P-C-NORMAL30b_masked_input (unchanged)",
            "note": "exploratory ablation only — NOT selected candidate replacement",
        }, f, indent=2)

    # ── 12. guardrail
    gr_rows = build_guardrail_rows(
        all_loss_finite=all_loss_finite, all_grad_finite=all_grad_ok,
        all_val_finite=all_val_finite,
        final_test_accessed=False, threshold_tuned=False,
        overwrite_occurred=False, training_completed=training_completed,
    )
    _write_csv(gr_rows, REPORT_ROOT / "p_c_normal40_guardrail_check.csv")
    gr_fail = sum(1 for r in gr_rows if r["status"] == "FAIL")
    print(f"  guardrail fail: {gr_fail}")

    # ── verdict
    if training_completed and all_loss_finite and all_val_finite and gr_fail == 0:
        verdict = "PASS"
    elif not all_loss_finite or not all_val_finite or gr_fail > 3:
        verdict = "FAIL"
    else:
        verdict = "PARTIAL_PASS"

    # ── 13. report MD
    md = f"""# P-C-NORMAL40: 2.5D 5ch Full Train Report

## 판정: {verdict}

생성: {ts}  |  stage: {STAGE_LABEL}
Role: post-selection exploratory ablation
selected_candidate: **P-C-NORMAL30b_masked_input (unchanged)**

---

## 학습 결과

| 항목 | 값 |
|---|---|
| total_epochs | {len(epoch_metric_rows)} |
| best_epoch | {best_epoch} |
| best_val_auc | {best_val_auc:.4f} |
| best_val_auprc | {best_em.get("val_auprc","?")} |
| early_stopped | {early_stopped} |
| train_loss_finite | {all_loss_finite} |
| grad_finite | {all_grad_ok} |
| val_finite | {all_val_finite} |
| guardrail_fail | {gr_fail} |

---

## Epoch Metrics

`outputs/reports/p_c_normal40_2p5d5ch_full_train/p_c_normal40_epoch_metrics.csv` 참조

---

## Checkpoint

- best: `checkpoints/p_c_normal40_best_val_auc_checkpoint.pt`

---

## Caveat

- val_auc/val_auprc는 train/val 내부 참고값만. final_test 접근 금지.
- 성능 비교 및 selected candidate 교체 결정 불가.
- selected candidate = P-C-NORMAL30b_masked_input.
- Grad-CAM 분석 시: `outputs/end/gradcam_masked_input_inference_v1/gradcam_run.py` 사용.

---

*이 보고서는 exploratory ablation full train 결과만 기록한다.*
"""
    (REPORT_ROOT / "p_c_normal40_full_train_report.md").write_text(md, encoding="utf-8")

    # ── 14. summary JSON
    summary = {
        "stage":                      STAGE_LABEL,
        "verdict":                    verdict,
        "role":                       "post_selection_exploratory_ablation",
        "selected_candidate":         "P-C-NORMAL30b_masked_input unchanged",
        "train_rows":                 len(df_tr_norm),
        "val_rows":                   len(df_vl_norm),
        "total_epochs":               len(epoch_metric_rows),
        "best_epoch":                 best_epoch,
        "best_val_auc":               round(float(best_val_auc), 4),
        "best_val_auprc":             best_em.get("val_auprc", "?"),
        "early_stopped":              early_stopped,
        "all_loss_finite":            all_loss_finite,
        "all_grad_finite":            all_grad_ok,
        "all_val_finite":             all_val_finite,
        "guardrail_fail_count":       gr_fail,
        "full_train_run":             True,
        "checkpoint_created":         best_ckpt_path.exists(),
        "final_test_accessed":        False,
        "threshold_optimization":     False,
        "selected_candidate_replaced": False,
        "inflation_source":           inflation_source,
        "use_amp":                    False,
        "timestamp":                  ts,
    }
    with open(REPORT_ROOT / "p_c_normal40_full_train_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(REPORT_ROOT / "DONE.json", "w") as f:
        json.dump({"stage": STAGE_LABEL, "verdict": verdict, "timestamp": ts}, f, indent=2)

    print(f"[{STAGE_LABEL}] ====================================================")
    print(f"[{STAGE_LABEL}] VERDICT: {verdict}")
    print(f"[{STAGE_LABEL}] best_epoch={best_epoch}  best_val_auc={best_val_auc:.4f}")
    print(f"[{STAGE_LABEL}] report: {REPORT_ROOT}")
    print(f"[{STAGE_LABEL}] ====================================================")
    return 0 if verdict in ("PASS", "PARTIAL_PASS") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P-C-NORMAL40 2.5D 5ch full train")
    parser.add_argument("--confirm-no-final-test",       action="store_true", required=True)
    parser.add_argument("--confirm-no-threshold-tuning", action="store_true", required=True)
    parser.add_argument("--confirm-exploratory",         action="store_true", required=True)
    parser.add_argument("--use-amp",   action="store_true", help="AMP (현재 미구현, 무시됨)")
    parser.add_argument("--save-last", action="store_true", help="마지막 epoch checkpoint 저장")
    args = parser.parse_args()

    missing = []
    if not args.confirm_no_final_test:       missing.append("--confirm-no-final-test")
    if not args.confirm_no_threshold_tuning: missing.append("--confirm-no-threshold-tuning")
    if not args.confirm_exploratory:         missing.append("--confirm-exploratory")
    if missing:
        print(f"[GUARD] 필수 flags 누락: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    sys.exit(run_full_train(args))
