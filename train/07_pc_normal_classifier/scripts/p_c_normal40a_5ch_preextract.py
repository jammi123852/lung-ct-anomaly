#!/usr/bin/env python3
"""
P-C-NORMAL40a: 5ch crop 사전 추출 → Linux 파일시스템 저장

/mnt/c/ 볼륨에서 5ch crop을 한 번만 읽어 Linux fs에 저장.
같은 volume은 1번만 load (그룹 단위 처리).
결과: outputs/p_c_normal40_5ch_preextract/{train,val}/<idx>.npy (5,96,96)
      + 업데이트된 manifest CSV (crop_npy_path, mask_ratio 컬럼 추가)
"""

import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Constants (P40과 동일) ────────────────────────────────────────────────────
HU_MIN       = -1000.0
HU_MAX       =  200.0
CROP_SIZE    = 96
Z_OFFSETS    = [-2, -1, 0, 1, 2]

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_MANIFEST   = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_balanced_w1_feature_manifest/p_c_normal24g_fix_balanced_w1_train_manifest.csv"
VAL_MANIFEST     = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_balanced_w1_feature_manifest/p_c_normal24g_fix_balanced_w1_val_manifest.csv"
MASK29B_MANIFEST = PROJECT_ROOT / "outputs/reports/p_c_normal29b_crop_level_mask_generation/p_c_normal29b_mask_manifest.csv"
NORMAL_VOL_ROOT  = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")
NSCLC_VOL_ROOT   = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
MASK_ROOT        = PROJECT_ROOT / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"

OUTPUT_ROOT  = PROJECT_ROOT / "outputs/p_c_normal40_5ch_preextract"
REPORT_ROOT  = PROJECT_ROOT / "outputs/reports/p_c_normal40a_5ch_preextract"


# ── extract_5ch_crop (P40과 동일) ─────────────────────────────────────────────
def extract_5ch_crop(ct_vol, mask_vol, z_center, y0, x0, y1, x1):
    D, H, W = ct_vol.shape[0], ct_vol.shape[1], ct_vol.shape[2]
    src_y0 = max(0, y0); src_y1 = min(H, y1)
    src_x0 = max(0, x0); src_x1 = min(W, x1)
    dst_y0 = src_y0 - y0; dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = src_x0 - x0; dst_x1 = dst_x0 + (src_x1 - src_x0)
    channels, masks = [], []
    for dz in Z_OFFSETS:
        z = max(0, min(D - 1, z_center + dz))
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
    mask_ratio = float((mask_5ch > 0.5).mean())
    return img_5ch, mask_ratio


def build_safe_id_lookup(mask29b_path):
    df = pd.read_csv(mask29b_path, low_memory=False)
    if "status" in df.columns:
        df = df[df["status"] == "PASS"]
    return dict(zip(df["crop_path"].astype(str), df["safe_id"].astype(str)))


def process_split(df_raw, split_name, sid_lookup, out_dir):
    """한 split 전체 추출. 볼륨 그룹 단위로 로드."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df_raw.copy().reset_index(drop=True)
    df["safe_id_vol"] = df["crop_path"].astype(str).map(sid_lookup)

    crop_npy_paths = [""] * len(df)
    mask_ratios    = [float("nan")] * len(df)
    errors         = []

    # 같은 (label, safe_id) 그룹으로 묶어서 볼륨 1번만 load
    df["_idx"] = df.index
    groups = df.groupby(["label", "safe_id_vol"], sort=False)
    n_groups = len(groups)

    t0 = time.time()
    done_vols = 0
    for (lbl, sid), grp in groups:
        done_vols += 1
        if done_vols % 20 == 0 or done_vols == 1:
            elapsed = time.time() - t0
            rate = done_vols / max(elapsed, 1)
            eta  = (n_groups - done_vols) / max(rate, 1e-9)
            print(f"  [{split_name}] {done_vols}/{n_groups} vols  "
                  f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m", flush=True)

        vol_root = NORMAL_VOL_ROOT if lbl == 0 else NSCLC_VOL_ROOT
        mask_sub = "normal" if lbl == 0 else "lesion"
        ct_path  = vol_root  / sid / "ct_hu.npy"
        mk_path  = MASK_ROOT / mask_sub / sid / "refined_roi.npy"

        if not ct_path.exists() or not mk_path.exists():
            for row_idx in grp["_idx"]:
                errors.append({"split": split_name, "row": row_idx,
                               "reason": "vol_or_mask_missing", "sid": sid})
            continue

        try:
            ct_vol   = np.load(str(ct_path),  mmap_mode="r")
            mask_vol = np.load(str(mk_path),  mmap_mode="r")
        except Exception as e:
            for row_idx in grp["_idx"]:
                errors.append({"split": split_name, "row": row_idx,
                               "reason": f"load_error:{e}", "sid": sid})
            continue

        for _, row in grp.iterrows():
            row_idx = int(row["_idx"])
            npy_path = out_dir / f"{row_idx:06d}.npy"

            # resume: 이미 추출됐으면 skip
            if npy_path.exists():
                crop_npy_paths[row_idx] = str(npy_path)
                # mask_ratio는 nan 유지 (나중에 재계산 불필요)
                continue

            try:
                z_c = int(float(row["canonical_volume_z"]))
                cy  = int(float(row["center_y"]))
                cx  = int(float(row["center_x"]))
                y0  = cy - CROP_SIZE // 2; y1 = cy + CROP_SIZE // 2
                x0  = cx - CROP_SIZE // 2; x1 = cx + CROP_SIZE // 2
                img_5ch, mrat = extract_5ch_crop(ct_vol, mask_vol, z_c, y0, x0, y1, x1)
                np.save(str(npy_path), img_5ch)
                crop_npy_paths[row_idx] = str(npy_path)
                mask_ratios[row_idx]    = mrat
            except Exception as e:
                errors.append({"split": split_name, "row": row_idx,
                               "reason": f"extract_error:{e}", "sid": sid})

    df["crop_npy_path"] = crop_npy_paths
    df["mask_ratio"]    = mask_ratios
    df.drop(columns=["_idx"], inplace=True)
    return df, errors


def main():
    print("[P-C-NORMAL40a] 5ch crop 사전 추출 시작", flush=True)
    t_total = time.time()

    for p in [TRAIN_MANIFEST, VAL_MANIFEST, MASK29B_MANIFEST]:
        if not p.exists():
            print(f"[ABORT] 파일 없음: {p}", file=sys.stderr); sys.exit(1)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    sid_lookup = build_safe_id_lookup(MASK29B_MANIFEST)
    df_tr = pd.read_csv(TRAIN_MANIFEST, low_memory=False)
    df_vl = pd.read_csv(VAL_MANIFEST,   low_memory=False)
    print(f"  train={len(df_tr)} val={len(df_vl)}", flush=True)

    all_errors = []

    # ── train
    print(f"\n[train] 시작", flush=True)
    df_tr_out, err_tr = process_split(
        df_tr, "train", sid_lookup, OUTPUT_ROOT / "train")
    all_errors.extend(err_tr)

    # ── val
    print(f"\n[val] 시작", flush=True)
    df_vl_out, err_vl = process_split(
        df_vl, "val", sid_lookup, OUTPUT_ROOT / "val")
    all_errors.extend(err_vl)

    # ── 결과 저장
    train_ok = int((df_tr_out["crop_npy_path"] != "").sum())
    val_ok   = int((df_vl_out["crop_npy_path"] != "").sum())

    df_tr_out.to_csv(REPORT_ROOT / "p40a_train_manifest_with_npy.csv", index=False)
    df_vl_out.to_csv(REPORT_ROOT / "p40a_val_manifest_with_npy.csv",   index=False)

    if all_errors:
        pd.DataFrame(all_errors).to_csv(
            REPORT_ROOT / "p40a_extract_errors.csv", index=False)

    elapsed = time.time() - t_total
    summary = {
        "stage":       "P-C-NORMAL40a_5ch_preextract",
        "train_total": len(df_tr_out),
        "train_ok":    train_ok,
        "val_total":   len(df_vl_out),
        "val_ok":      val_ok,
        "errors":      len(all_errors),
        "elapsed_min": round(elapsed / 60, 1),
        "verdict":     "PASS" if len(all_errors) == 0 else "PARTIAL",
        "output_root": str(OUTPUT_ROOT),
        "train_manifest_with_npy": str(REPORT_ROOT / "p40a_train_manifest_with_npy.csv"),
        "val_manifest_with_npy":   str(REPORT_ROOT / "p40a_val_manifest_with_npy.csv"),
    }
    with open(REPORT_ROOT / "p40a_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[P-C-NORMAL40a] 완료  train_ok={train_ok}/{len(df_tr_out)}"
          f"  val_ok={val_ok}/{len(df_vl_out)}"
          f"  errors={len(all_errors)}  elapsed={elapsed/60:.1f}m", flush=True)
    print(f"  verdict: {summary['verdict']}", flush=True)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
