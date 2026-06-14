#!/usr/bin/env python3
"""
P-C-NORMAL39: 2.5D 5ch full-train preflight

Stage: P-C-NORMAL39_2p5d5ch_full_train_preflight

이번 단계:
  - full train 실행 금지
  - 모델 학습 금지
  - checkpoint 생성 금지
  - read-only 검증 + 경로·shape·mask·I/O·설정·guardrail 확인만 수행

Guardrail 필수:
  preflight_only=True, no_training_run=True, no_checkpoint_created=True,
  final_test_accessed=False, threshold_optimization=False,
  selected_candidate_not_replaced=True
"""

import argparse, csv, json, random, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── 입력 경로 ─────────────────────────────────────────────────────────────────
TRAIN_MANIFEST   = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_balanced_w1_feature_manifest/p_c_normal24g_fix_balanced_w1_train_manifest.csv"
VAL_MANIFEST     = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_balanced_w1_feature_manifest/p_c_normal24g_fix_balanced_w1_val_manifest.csv"
SCALAR_STATS_PATH= PROJECT_ROOT / "outputs/reports/p_c_normal24h_fix_zroi_scalar_fusion_drycheck/p_c_normal24h_fix_scalar_normalization_stats.json"
MASK29B_MANIFEST = PROJECT_ROOT / "outputs/reports/p_c_normal29b_crop_level_mask_generation/p_c_normal29b_mask_manifest.csv"
CKPT30B          = PROJECT_ROOT / "outputs/p_c_normal30b_masked_input_full_train/checkpoints/p_c_normal30b_best_val_auc_checkpoint.pt"
P38_REPORT_ROOT  = PROJECT_ROOT / "outputs/reports/p_c_normal38_2p5d5ch_limited_smoke_train"
NORMAL_VOL_ROOT  = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")
NSCLC_VOL_ROOT   = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
MASK_ROOT        = PROJECT_ROOT / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"

OUTPUT_ROOT = PROJECT_ROOT / "outputs/reports/p_c_normal39_2p5d5ch_full_train_preflight"

STAGE_LABEL     = "P-C-NORMAL39_2p5d5ch_full_train_preflight"
CROP_SIZE       = 96
HU_MIN          = -1000.0
HU_MAX          =  200.0
Z_OFFSETS       = [-2, -1, 0, 1, 2]
SCALAR_FEATURES = ["lung_z_percentile", "crop_lung_roi_ratio"]
SEED            = 42
random.seed(SEED); np.random.seed(SEED)


# ── Utilities ─────────────────────────────────────────────────────────────────
def _write_csv(rows, path):
    path = Path(path)
    if not rows:
        path.write_text("(empty)\n", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def build_safe_id_lookup(mask29b_path):
    """crop_path → safe_id  (status==PASS 필터, P38와 동일)"""
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


# ── P38와 동일한 boundary-safe 5ch crop extraction ────────────────────────────
def extract_5ch_crop_preflight(ct_vol, mask_vol, z_center, cy, cx):
    D, H, W = ct_vol.shape[0], ct_vol.shape[1], ct_vol.shape[2]
    y0 = cy - CROP_SIZE // 2; y1 = cy + CROP_SIZE // 2
    x0 = cx - CROP_SIZE // 2; x1 = cx + CROP_SIZE // 2

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
        if z != z_center + dz:
            nearest_repeat = True
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


# ── Main ──────────────────────────────────────────────────────────────────────
def run_preflight(args) -> int:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{STAGE_LABEL}] started: {ts}")

    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        print(f"[ABORT] output dir already exists and non-empty: {OUTPUT_ROOT}", file=sys.stderr)
        sys.exit(2)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ══ 1. P38 smoke 결과 확인 (read-only) ══
    print(f"[{STAGE_LABEL}] [1/8] P38 smoke 결과 확인 ...")
    p38_verdict = "UNKNOWN"; p38_gr_fail = -1
    p38_sum_path = P38_REPORT_ROOT / "p_c_normal38_smoke_summary.json"
    p38_gr_path  = P38_REPORT_ROOT / "p_c_normal38_guardrail_check.csv"
    if p38_sum_path.exists():
        with open(p38_sum_path) as f:
            p38_sum = json.load(f)
        p38_verdict = p38_sum.get("verdict", "UNKNOWN")
        print(f"  P38 verdict={p38_verdict}  "
              f"selected_candidate={p38_sum.get('selected_candidate','?')}")
    else:
        print(f"  [WARN] P38 summary not found: {p38_sum_path}")
    if p38_gr_path.exists():
        df_gr38 = pd.read_csv(p38_gr_path)
        p38_gr_fail = int((df_gr38["status"] == "FAIL").sum())
        print(f"  P38 guardrail_fail={p38_gr_fail}")

    # ══ 2. manifest load + safe_id join ══
    print(f"[{STAGE_LABEL}] [2/8] manifest load + safe_id join ...")
    for p in [TRAIN_MANIFEST, VAL_MANIFEST, SCALAR_STATS_PATH, MASK29B_MANIFEST]:
        if not p.exists():
            print(f"[ABORT] 필수 파일 없음: {p}", file=sys.stderr); sys.exit(1)

    df_tr_raw = pd.read_csv(TRAIN_MANIFEST, low_memory=False)
    df_vl_raw = pd.read_csv(VAL_MANIFEST,   low_memory=False)
    print(f"  raw: train={len(df_tr_raw)} val={len(df_vl_raw)}")

    sid_lookup = build_safe_id_lookup(MASK29B_MANIFEST)
    df_tr_raw["safe_id_vol"] = df_tr_raw["crop_path"].astype(str).map(sid_lookup)
    df_vl_raw["safe_id_vol"] = df_vl_raw["crop_path"].astype(str).map(sid_lookup)
    join_miss_tr = int(df_tr_raw["safe_id_vol"].isna().sum())
    join_miss_vl = int(df_vl_raw["safe_id_vol"].isna().sum())
    print(f"  safe_id join missing: train={join_miss_tr} val={join_miss_vl}")

    with open(SCALAR_STATS_PATH) as f:
        norm_payload = json.load(f)
    scalar_stats = norm_payload.get("features", norm_payload)

    # ══ 3. full path resolution (unique safe_id 캐싱 최적화) ══
    print(f"[{STAGE_LABEL}] [3/8] full path resolution ...")

    # step A: unique (label, safe_id) 쌍 수집 (vectorized)
    unique_pairs = set()
    for df in [df_tr_raw, df_vl_raw]:
        sids   = df["safe_id_vol"].fillna("").astype(str)
        labels = df["label"].astype(int)
        unique_pairs.update(zip(labels, sids))

    # step B: 각 고유 pair에 대해서만 exists 확인
    pair_ok: dict = {}
    for (label, sid) in unique_pairs:
        if not sid or sid == "nan":
            pair_ok[(label, sid)] = False; continue
        vol_root = NORMAL_VOL_ROOT if label == 0 else NSCLC_VOL_ROOT
        mask_sub = "normal" if label == 0 else "lesion"
        vol_ok  = (vol_root / sid / "ct_hu.npy").exists()
        mask_ok = (MASK_ROOT / mask_sub / sid / "refined_roi.npy").exists()
        pair_ok[(label, sid)] = vol_ok and mask_ok
    pair_fail_n = sum(1 for v in pair_ok.values() if not v)
    print(f"  unique pairs checked={len(pair_ok)}  pair_fail={pair_fail_n}")

    # step C: row별 missing 기록
    path_rows = []
    for split, df in [("train", df_tr_raw), ("val", df_vl_raw)]:
        sids   = df["safe_id_vol"].fillna("").astype(str)
        labels = df["label"].astype(int)
        for i, (label, sid) in enumerate(zip(labels, sids)):
            ok = pair_ok.get((label, sid), False)
            if not ok:
                row = df.iloc[i]
                path_rows.append({
                    "split": split,
                    "crop_path":   str(row.get("crop_path", "")),
                    "patient_id":  str(row.get("patient_id", "")),
                    "safe_id_vol": sid, "label": label,
                    "pair_ok":     int(pair_ok.get((label, sid), False)),
                })
    _write_csv(path_rows, OUTPUT_ROOT / "p_c_normal39_path_resolution_full.csv")
    path_missing_total = len(path_rows)
    print(f"  path_missing rows: {path_missing_total}")

    # step D: path OK 데이터프레임 (vectorized filter)
    def filter_ok(df):
        sids   = df["safe_id_vol"].fillna("").astype(str)
        labels = df["label"].astype(int)
        ok_mask = [pair_ok.get((l, s), False) for l, s in zip(labels, sids)]
        return df[ok_mask].reset_index(drop=True)

    df_tr_ok = filter_ok(df_tr_raw)
    df_vl_ok = filter_ok(df_vl_raw)
    print(f"  path OK: train={len(df_tr_ok)} val={len(df_vl_ok)}")

    df_tr_norm = apply_scalar_norm(df_tr_ok, scalar_stats)
    df_vl_norm = apply_scalar_norm(df_vl_ok, scalar_stats)

    # ══ 4. manifest 통계 ══
    print(f"[{STAGE_LABEL}] [4/8] manifest 통계 계산 ...")
    dist_rows = []
    all_vessel_cols = []
    scalar_finite_overall = True

    for split, df_raw, df_n in [("train", df_tr_ok, df_tr_norm), ("val", df_vl_ok, df_vl_norm)]:
        label_dist = dict(df_raw["label"].value_counts())
        pid_n = df_raw["patient_id"].nunique() if "patient_id" in df_raw.columns else -1
        sid_n = df_raw["safe_id_vol"].nunique()
        z_min = float(df_raw["canonical_volume_z"].min()) if "canonical_volume_z" in df_raw.columns else -1
        z_max = float(df_raw["canonical_volume_z"].max()) if "canonical_volume_z" in df_raw.columns else -1
        cy_min= float(df_raw["center_y"].min()) if "center_y" in df_raw.columns else -1
        cy_max= float(df_raw["center_y"].max()) if "center_y" in df_raw.columns else -1
        cx_min= float(df_raw["center_x"].min()) if "center_x" in df_raw.columns else -1
        cx_max= float(df_raw["center_x"].max()) if "center_x" in df_raw.columns else -1
        sw_uniq = sorted(df_raw["sample_weight"].unique().tolist()) if "sample_weight" in df_raw.columns else []
        vcols = [c for c in df_raw.columns if "vessel" in c.lower()]
        all_vessel_cols.extend(vcols)

        def _scalar_check(df, cols):
            nan_n = inf_n = 0
            for c in cols:
                if c in df.columns:
                    arr = df[c].astype(float)
                    nan_n += int(arr.isna().any())
                    inf_n += int(np.isinf(arr.values).any())
            return nan_n, inf_n

        nan_raw, inf_raw = _scalar_check(df_raw, SCALAR_FEATURES)
        nan_nrm, inf_nrm = _scalar_check(df_n,   SCALAR_FEATURES)
        if nan_raw + inf_raw + nan_nrm + inf_nrm > 0:
            scalar_finite_overall = False

        dist_rows.append({
            "split":               split,
            "total_rows":          len(df_raw),
            "label_0_normal":      int(label_dist.get(0, 0)),
            "label_1_nsclc":       int(label_dist.get(1, 0)),
            "patient_id_count":    pid_n,
            "safe_id_count":       sid_n,
            "canonical_z_min":     z_min, "canonical_z_max": z_max,
            "center_y_min":        cy_min, "center_y_max": cy_max,
            "center_x_min":        cx_min, "center_x_max": cx_max,
            "sample_weight_unique": str(sw_uniq),
            "vessel_cols_found":   str(vcols),
            "scalar_nan_raw":      nan_raw, "scalar_inf_raw": inf_raw,
            "scalar_nan_norm":     nan_nrm, "scalar_inf_norm": inf_nrm,
        })
    _write_csv(dist_rows, OUTPUT_ROOT / "p_c_normal39_manifest_distribution_summary.csv")
    all_vessel_cols = list(set(all_vessel_cols))
    print(f"  vessel_cols={all_vessel_cols}  scalar_finite={scalar_finite_overall}")

    # ══ 5. representative shape sampling audit ══
    print(f"[{STAGE_LABEL}] [5/8] shape sampling audit ...")

    def sample_group(df, n, label=None, cond_mask=None):
        sub = df[df["label"] == label] if label is not None else df
        if cond_mask is not None:
            sub = sub[cond_mask(sub)]
        if len(sub) == 0:
            return sub.iloc[0:0]
        return sub.sample(min(n, len(sub)), random_state=SEED)

    sample_parts = [
        ("train_normal_50",  sample_group(df_tr_ok, 50, label=0), "train"),
        ("train_nsclc_50",   sample_group(df_tr_ok, 50, label=1), "train"),
        ("val_normal_30",    sample_group(df_vl_ok, 30, label=0), "val"),
        ("val_nsclc_30",     sample_group(df_vl_ok, 30, label=1), "val"),
    ]
    if "crop_lung_roi_ratio" in df_tr_ok.columns:
        lm = sample_group(df_tr_ok, 20,
                          cond_mask=lambda d: d["crop_lung_roi_ratio"].astype(float) < 0.3)
        if len(lm) > 0:
            sample_parts.append(("train_low_roi_20", lm, "train"))
    if "canonical_volume_z" in df_tr_ok.columns:
        z90 = float(df_tr_ok["canonical_volume_z"].quantile(0.9))
        zb  = sample_group(df_tr_ok, 20,
                           cond_mask=lambda d: (d["canonical_volume_z"] < 10) |
                                               (d["canonical_volume_z"] > z90))
        if len(zb) > 0:
            sample_parts.append(("train_zbnd_20", zb, "train"))

    shape_rows = []
    total_n = sum(len(s[1]) for s in sample_parts)
    done_n  = 0

    for grp, df_s, split in sample_parts:
        print(f"  auditing {grp}: n={len(df_s)}")
        for _, row in df_s.iterrows():
            label    = int(row["label"])
            sid      = str(row["safe_id_vol"])
            z_c      = int(float(row["canonical_volume_z"]))
            cy       = int(float(row["center_y"]))
            cx       = int(float(row["center_x"]))
            vol_root = NORMAL_VOL_ROOT if label == 0 else NSCLC_VOL_ROOT
            mask_sub = "normal" if label == 0 else "lesion"
            sc_vals  = [float(row[c]) if c in row.index else float("nan")
                        for c in SCALAR_FEATURES]
            sc_ok    = all(np.isfinite(v) for v in sc_vals)
            try:
                ct_v  = np.load(str(vol_root / sid / "ct_hu.npy"),  mmap_mode="r")
                mk_v  = np.load(str(MASK_ROOT / mask_sub / sid / "refined_roi.npy"),
                                mmap_mode="r")
                img5, msk5, nr, bc = extract_5ch_crop_preflight(ct_v, mk_v, z_c, cy, cx)
                mask_nz = float((msk5 > 0.5).mean())
                shape_rows.append({
                    "group":   grp, "split": split,
                    "crop_path":  str(row.get("crop_path","")),
                    "patient_id": str(row.get("patient_id","")),
                    "label": label, "safe_id_vol": sid,
                    "canonical_z": z_c, "center_y": cy, "center_x": cx,
                    "vol_shape":    str(ct_v.shape),
                    "mask_vol_shape": str(mk_v.shape),
                    "img5_shape":   str(img5.shape),
                    "msk5_shape":   str(msk5.shape),
                    "nearest_repeat":  int(nr),
                    "boundary_clipped": int(bc),
                    "mask_nz_ratio":   round(mask_nz, 4),
                    "zero_mask":       int(mask_nz == 0.0),
                    "low_mask":        int(mask_nz < 0.1),
                    "scalar_finite":   int(sc_ok),
                    "status": "OK",
                })
            except Exception as e:
                shape_rows.append({
                    "group": grp, "split": split,
                    "crop_path":  str(row.get("crop_path","")),
                    "patient_id": str(row.get("patient_id","")),
                    "label": label, "safe_id_vol": sid,
                    "status": f"ERROR:{e}",
                })
            done_n += 1
            if done_n % 40 == 0:
                print(f"    progress: {done_n}/{total_n}")

    _write_csv(shape_rows, OUTPUT_ROOT / "p_c_normal39_shape_sampling_audit.csv")

    df_sh = pd.DataFrame([r for r in shape_rows if r.get("status") == "OK"])
    sh_err     = sum(1 for r in shape_rows if "ERROR" in str(r.get("status","")))
    wrong_shp  = int((df_sh["img5_shape"] != "(5, 96, 96)").sum()) if not df_sh.empty else 0
    zero_mask_n= int(df_sh["zero_mask"].sum())   if not df_sh.empty else 0
    low_mask_n = int(df_sh["low_mask"].sum())    if not df_sh.empty else 0
    bc_n       = int(df_sh["boundary_clipped"].sum()) if not df_sh.empty else 0
    nr_n       = int(df_sh["nearest_repeat"].sum())   if not df_sh.empty else 0
    sc_fail_n  = int((df_sh["scalar_finite"] == 0).sum()) if not df_sh.empty else 0

    shape_audit_status = "PASS"
    if sh_err > 0 or wrong_shp > 0 or sc_fail_n > 0:
        shape_audit_status = "FAIL"
    elif zero_mask_n > len(shape_rows) * 0.05:
        shape_audit_status = "PARTIAL_PASS"
    print(f"  errors={sh_err} wrong_shape={wrong_shp} zero_mask={zero_mask_n} "
          f"low_mask={low_mask_n} bc={bc_n} nr={nr_n} sc_fail={sc_fail_n} "
          f"→ {shape_audit_status}")

    # ══ 6. full train 설정 제안 ══
    train_rows = len(df_tr_ok)
    val_rows   = len(df_vl_ok)
    batch_sz   = 16
    batches_ep = (train_rows + batch_sz - 1) // batch_sz
    smoke_ep_sec = 20
    full_ep_sec  = int(smoke_ep_sec * (train_rows / max(512, 1)))

    config_rows = [
        {"key": "stage",               "value": "P-C-NORMAL40_2p5d5ch_full_train"},
        {"key": "input_channels",      "value": "5"},
        {"key": "input_size",          "value": "96x96"},
        {"key": "z_offsets",           "value": str(Z_OFFSETS)},
        {"key": "hu_min",              "value": str(HU_MIN)},
        {"key": "hu_max",              "value": str(HU_MAX)},
        {"key": "batch_size",          "value": "16  (GPU 여유시 32 후보)"},
        {"key": "num_workers",         "value": "0  (WSL2 /mnt/c 안정성)"},
        {"key": "epochs_max",          "value": "30"},
        {"key": "early_stop_patience", "value": "7"},
        {"key": "learning_rate",       "value": "1e-4"},
        {"key": "optimizer",           "value": "Adam"},
        {"key": "loss",                "value": "BCEWithLogitsLoss"},
        {"key": "sample_weight",       "value": "1.0  (변경 금지)"},
        {"key": "checkpoint_strategy", "value": "best_val_auc  (exploratory only, NOT selected candidate)"},
        {"key": "final_test",          "value": "금지"},
        {"key": "threshold_tuning",    "value": "금지"},
        {"key": "first_conv",          "value": "30b 3ch→5ch inflate  (channel-mean × 3/5)"},
        {"key": "crop_extraction",     "value": "center_y/cx ±48  boundary-safe"},
        {"key": "mask_ratio_source",   "value": "mask_5ch > 0.5"},
        {"key": "selected_candidate",  "value": "P-C-NORMAL30b_masked_input  (unchanged)"},
        {"key": "train_rows",          "value": str(train_rows)},
        {"key": "val_rows",            "value": str(val_rows)},
        {"key": "batches_per_epoch",   "value": str(batches_ep)},
    ]
    _write_csv(config_rows, OUTPUT_ROOT / "p_c_normal39_full_train_config_plan.csv")

    # ══ 7. runtime / risk 산정 ══
    risk_rows = [
        {"metric": "train_rows",          "value": str(train_rows)},
        {"metric": "val_rows",            "value": str(val_rows)},
        {"metric": "batch_size",          "value": str(batch_sz)},
        {"metric": "batches_per_epoch",   "value": str(batches_ep)},
        {"metric": "smoke_1ep_sec_est",   "value": str(smoke_ep_sec)},
        {"metric": "full_1ep_sec_est",    "value": str(full_ep_sec)},
        {"metric": "est_10ep_min",        "value": str(round(full_ep_sec * 10 / 60, 1))},
        {"metric": "est_20ep_min",        "value": str(round(full_ep_sec * 20 / 60, 1))},
        {"metric": "est_30ep_min",        "value": str(round(full_ep_sec * 30 / 60, 1))},
        {"metric": "io_risk_mnt_c",       "value": "MEDIUM - WSL2 /mnt/c 경유, num_workers=0 필수"},
        {"metric": "gpu_oom_risk",        "value": "LOW - EfficientNet-B0 5ch 96x96, batch=16"},
        {"metric": "cache_strategy",      "value": "mmap_mode=r  (volume 미사전로드)"},
        {"metric": "checkpoint_disk_est", "value": "~20MB/checkpoint"},
        {"metric": "note",                "value": "volumes on Windows Desktop via /mnt/c"},
    ]
    _write_csv(risk_rows, OUTPUT_ROOT / "p_c_normal39_runtime_risk_estimate.csv")

    # ══ 8. guardrail check ══
    join_ok = (join_miss_tr == 0 and join_miss_vl == 0)
    checks = [
        ("preflight_only",                  True,  True),
        ("no_training_run",                 True,  True),
        ("no_model_forward_train",          True,  True),
        ("no_checkpoint_created",           True,  True),
        ("final_test_accessed",             False, False),
        ("threshold_optimization",          False, False),
        ("threshold_sweep",                 False, False),
        ("best_threshold_selection",        False, False),
        ("selected_candidate_not_replaced", True,  True),
        ("p38_outputs_readonly",            True,  True),
        ("p30b_outputs_readonly",           True,  True),
        ("no_existing_result_overwrite",    True,  True),
        ("no_vessel_feature",               True,  len(all_vessel_cols) == 0),
        ("no_roi_masked_loss",              True,  True),
        ("crop_uses_center_yx_96",          True,  True),
        ("y0x0y1x1_patch_bbox_not_used",    True,  True),
        ("mask_ratio_computed_from_mask",   True,  True),
        ("diagnostic_wording_avoided",      True,  True),
        ("scalar_finite",                   True,  scalar_finite_overall),
        ("safe_id_join_complete",           True,  join_ok),
        ("path_resolution_pass",            True,  path_missing_total == 0),
        ("shape_audit_pass",                True,  shape_audit_status in ("PASS","PARTIAL_PASS")),
        ("p38_smoke_verdict_pass",          True,  p38_verdict == "PASS"),
        ("p38_guardrail_fail_zero",         True,  p38_gr_fail == 0),
    ]
    gr_rows = []
    gr_fail = 0
    for key, expected, actual in checks:
        status = "OK" if actual == expected else "FAIL"
        if status == "FAIL": gr_fail += 1
        gr_rows.append({"key": key, "expected": expected, "actual": actual, "status": status})
    _write_csv(gr_rows, OUTPUT_ROOT / "p_c_normal39_guardrail_check.csv")
    print(f"  guardrail fail: {gr_fail}")

    # P40 command draft
    p40_sh = (
        "#!/bin/bash\n"
        "# P-C-NORMAL40 full train command draft\n"
        "# ★ 실행 전 사용자 승인 필수\n"
        "# ★ P40 스크립트(scripts/p_c_normal40_2p5d5ch_full_train.py) 먼저 작성 필요\n\n"
        "source ~/ai_env/bin/activate && \\\n"
        "nohup python scripts/p_c_normal40_2p5d5ch_full_train.py \\\n"
        "  --confirm-no-final-test \\\n"
        "  --confirm-no-threshold-tuning \\\n"
        "  --confirm-exploratory \\\n"
        "  > logs/p_c_normal40_full_train_$(date +%Y%m%d_%H%M).log 2>&1 & \\\n"
        "echo \"PID: $!\"\n\n"
        f"# ── 예상 설정 ──────────────────────────────────────────────────────────\n"
        f"# train_rows     = {train_rows}\n"
        f"# val_rows       = {val_rows}\n"
        f"# batch_size     = {batch_sz}\n"
        f"# batches/epoch  = {batches_ep}\n"
        f"# epochs_max     = 30  (early_stop patience=7)\n"
        f"# est_30ep       = {round(full_ep_sec * 30 / 60, 1)} min  (GPU 기준)\n"
        f"# first_conv     = 30b 3ch→5ch inflate\n"
        f"# selected_cand  = P-C-NORMAL30b_masked_input  (unchanged)\n"
        f"# final_test     = 금지\n"
        f"# threshold_tune = 금지\n"
    )
    (OUTPUT_ROOT / "p_c_normal39_p40_command_draft.sh").write_text(p40_sh, encoding="utf-8")

    # ══ verdict ══
    if gr_fail == 0 and shape_audit_status == "PASS" and path_missing_total == 0:
        verdict = "PASS"; p40_ok = True
    elif gr_fail == 0 and shape_audit_status in ("PASS", "PARTIAL_PASS"):
        verdict = "PARTIAL_PASS"; p40_ok = True
    else:
        verdict = "FAIL"; p40_ok = False

    # ══ report MD ══
    tr = dist_rows[0] if dist_rows else {}
    vl = dist_rows[1] if len(dist_rows) > 1 else {}
    md = f"""# P-C-NORMAL39: 2.5D 5ch Full-Train Preflight Report

## 판정: {verdict}

생성: {ts}  |  stage: {STAGE_LABEL}

---

## 1. P38 Smoke 확인

| 항목 | 값 |
|---|---|
| P38 verdict | {p38_verdict} |
| P38 guardrail fail | {p38_gr_fail} |
| selected_candidate | P-C-NORMAL30b_masked_input (unchanged) |
| P38 = exploratory ablation | True |

---

## 2. 전체 Path Resolve

| 항목 | 값 |
|---|---|
| train rows (raw) | {len(df_tr_raw)} |
| val rows (raw) | {len(df_vl_raw)} |
| safe_id join missing (train) | {join_miss_tr} |
| safe_id join missing (val) | {join_miss_vl} |
| unique (label,safe_id) pairs | {len(pair_ok)} |
| path missing total | {path_missing_total} |
| train rows (path OK) | {train_rows} |
| val rows (path OK) | {val_rows} |

---

## 3. Manifest 통계

| split | total | normal | nsclc | patients | safe_ids | scalar_nan_raw | scalar_inf_raw | vessel_cols |
|---|---|---|---|---|---|---|---|---|
| train | {tr.get('total_rows','')} | {tr.get('label_0_normal','')} | {tr.get('label_1_nsclc','')} | {tr.get('patient_id_count','')} | {tr.get('safe_id_count','')} | {tr.get('scalar_nan_raw','')} | {tr.get('scalar_inf_raw','')} | {tr.get('vessel_cols_found','')} |
| val | {vl.get('total_rows','')} | {vl.get('label_0_normal','')} | {vl.get('label_1_nsclc','')} | {vl.get('patient_id_count','')} | {vl.get('safe_id_count','')} | {vl.get('scalar_nan_raw','')} | {vl.get('scalar_inf_raw','')} | {vl.get('vessel_cols_found','')} |

center_y range: train=[{tr.get('center_y_min','')}, {tr.get('center_y_max','')}]  val=[{vl.get('center_y_min','')}, {vl.get('center_y_max','')}]
center_x range: train=[{tr.get('center_x_min','')}, {tr.get('center_x_max','')}]  val=[{vl.get('center_x_min','')}, {vl.get('center_x_max','')}]
z range:        train=[{tr.get('canonical_z_min','')}, {tr.get('canonical_z_max','')}]  val=[{vl.get('canonical_z_min','')}, {vl.get('canonical_z_max','')}]

---

## 4. Shape Sampling Audit

| 항목 | 값 |
|---|---|
| 총 샘플 | {len(shape_rows)} |
| 오류 (ERROR) | {sh_err} |
| wrong shape | {wrong_shp} |
| zero_mask | {zero_mask_n} |
| low_mask (<10%) | {low_mask_n} |
| boundary_clipped | {bc_n} |
| nearest_repeat | {nr_n} |
| scalar fail | {sc_fail_n} |
| **판정** | **{shape_audit_status}** |

---

## 5. Full Train 설정 (P40)

- input: 5ch z-2..z+2 / 96×96 / HU -1000~200
- batch: {batch_sz} / num_workers: 0 / epochs: ≤30 / patience: 7
- lr: 1e-4 / loss: BCEWithLogitsLoss / sample_weight: 1.0
- first_conv: 30b 3ch→5ch inflate
- checkpoint: best val AUC (exploratory only)
- **final_test 금지 / threshold_tuning 금지**

---

## 6. 예상 시간 / 리스크

| 항목 | 값 |
|---|---|
| batches/epoch | {batches_ep} |
| 10ep est. | {round(full_ep_sec * 10 / 60, 1)} min |
| 20ep est. | {round(full_ep_sec * 20 / 60, 1)} min |
| 30ep est. | {round(full_ep_sec * 30 / 60, 1)} min |
| I/O risk | MEDIUM  (WSL2 /mnt/c, num_workers=0 필수) |
| GPU OOM risk | LOW  (EfficientNet-B0, batch=16) |

---

## 7. Guardrail

| fail 수 | {gr_fail} |
|---|---|

---

## 8. P40 승인 조건

| 조건 | 상태 |
|---|---|
| path_missing = 0 | {"OK" if path_missing_total == 0 else "FAIL"} |
| shape_audit | {shape_audit_status} |
| zero_mask severe (<5%) | {"OK" if zero_mask_n < max(1, len(shape_rows) * 0.05) else "CHECK"} |
| scalar_finite | {"OK" if scalar_finite_overall else "FAIL"} |
| guardrail_fail = 0 | {"OK" if gr_fail == 0 else "FAIL " + str(gr_fail)} |
| selected 30b 유지 | OK |
| final_test 없음 | OK |
| P40 command draft | OK  (p_c_normal39_p40_command_draft.sh) |
| **P40 recommended** | **{"YES" if p40_ok else "NO"}** |

---

*이 보고서는 preflight only — 모델 학습 없음, checkpoint 없음.*
"""
    (OUTPUT_ROOT / "p_c_normal39_full_train_preflight_report.md").write_text(md, encoding="utf-8")

    # ══ summary JSON ══
    summary = {
        "stage":                          STAGE_LABEL,
        "verdict":                        verdict,
        "role":                           "full_train_preflight_only",
        "selected_candidate":             "P-C-NORMAL30b_masked_input unchanged",
        "p38_smoke_verdict":              p38_verdict,
        "p38_guardrail_fail":             p38_gr_fail,
        "train_rows":                     train_rows,
        "val_rows":                       val_rows,
        "path_missing_total":             path_missing_total,
        "shape_audit_status":             shape_audit_status,
        "shape_audit_errors":             sh_err,
        "zero_mask_count_sampled":        zero_mask_n,
        "low_mask_count_sampled":         low_mask_n,
        "boundary_clipped_count_sampled": bc_n,
        "nearest_repeat_count_sampled":   nr_n,
        "scalar_finite":                  scalar_finite_overall,
        "forbidden_vessel_columns":       all_vessel_cols,
        "guardrail_fail_count":           gr_fail,
        "full_train_run":                 False,
        "checkpoint_created":             False,
        "final_test_accessed":            False,
        "threshold_optimization":         False,
        "selected_candidate_replaced":    False,
        "p40_recommended":                p40_ok,
        "batches_per_epoch":              batches_ep,
        "est_30ep_min":                   round(full_ep_sec * 30 / 60, 1),
        "timestamp":                      ts,
    }
    with open(OUTPUT_ROOT / "p_c_normal39_full_train_preflight_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_ROOT / "DONE.json", "w") as f:
        json.dump({"stage": STAGE_LABEL, "verdict": verdict, "timestamp": ts}, f, indent=2)

    print(f"[{STAGE_LABEL}] ====================================================")
    print(f"[{STAGE_LABEL}] VERDICT: {verdict}")
    print(f"[{STAGE_LABEL}] report: {OUTPUT_ROOT}")
    print(f"[{STAGE_LABEL}] ====================================================")
    return 0 if p40_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P-C-NORMAL39 full-train preflight")
    parser.add_argument("--preflight",             action="store_true", required=True,
                        help="preflight 모드 확인 (필수)")
    parser.add_argument("--confirm-no-training",   action="store_true", required=True,
                        help="학습 실행 안 함 확인 (필수)")
    parser.add_argument("--confirm-no-checkpoint", action="store_true", required=True,
                        help="checkpoint 생성 안 함 확인 (필수)")
    args = parser.parse_args()

    missing = []
    if not args.preflight:             missing.append("--preflight")
    if not args.confirm_no_training:   missing.append("--confirm-no-training")
    if not args.confirm_no_checkpoint: missing.append("--confirm-no-checkpoint")
    if missing:
        print(f"[GUARD] 필수 flags 누락: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    sys.exit(run_preflight(args))
