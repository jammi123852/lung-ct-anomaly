# -*- coding: utf-8 -*-
"""
임의 NSCLC 환자 1명에 대해 manifest 생성 + rd4ad inference 실행.
usage: python rd4ad_infer_single_patient_v1.py --patient LUNG1-001
"""
import os, sys, csv, argparse
import numpy as np

CTBASE  = "/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy"
ROOT    = "/home/jinhy/project/lung-ct-anomaly"
INFER_DIR = os.path.join(ROOT, "outputs/end/rd4ad_lung_mip3ch_infer_v1")
CKPT    = os.path.join(INFER_DIR, "best_train_loss.pth")
INFER_PY = os.path.join(INFER_DIR, "rd4ad_infer.py")

CROP_SIZE   = 96
CROP_STRIDE = 16

PBIN_MAP = {
    (0, 0): "upper_peripheral", (0, 1): "upper_central",   (0, 2): "upper_peripheral",
    (1, 0): "middle_peripheral",(1, 1): "middle_central",  (1, 2): "middle_peripheral",
    (2, 0): "lower_peripheral", (2, 1): "lower_central",   (2, 2): "lower_peripheral",
}


def find_safe_id(patient_id):
    prefix = f"NSCLC_{patient_id}__"
    for d in os.listdir(CTBASE):
        if d.startswith(prefix):
            return d
    return None


def assign_pbin(cyc, cxc, roi_slice):
    ys, xs = np.where(roi_slice > 0)
    if ys.size == 0:
        return "unknown"
    by0, by1 = int(ys.min()), int(ys.max())
    bx0, bx1 = int(xs.min()), int(xs.max())
    ry = (cyc - by0) / max(by1 - by0, 1)
    rx = (cxc - bx0) / max(bx1 - bx0, 1)
    ri = min(int(ry * 3), 2)
    ci_left  = rx < 0.5
    ci_right = rx >= 0.5
    # 좌폐/우폐 구분으로 peripheral/central
    mid_x = (bx0 + bx1) / 2
    if cxc < mid_x:
        ci = 0 if (cxc - bx0) / max(mid_x - bx0, 1) < 0.33 else 1
    else:
        ci = 2 if (cxc - mid_x) / max(bx1 - mid_x, 1) > 0.67 else 1
    return PBIN_MAP.get((ri, ci), "unknown")


def make_manifest(patient_id, safe_id, out_csv):
    cdir = os.path.join(CTBASE, safe_id)
    ct   = np.load(os.path.join(cdir, "ct_hu.npy"), mmap_mode="r")
    roi  = np.load(os.path.join(cdir, "roi_0_0.npy"), mmap_mode="r")
    lmask_path = os.path.join(cdir, "lesion_mask_roi_0_0.npy")
    has_lmask = os.path.exists(lmask_path)
    if has_lmask:
        lmask = np.load(lmask_path, mmap_mode="r")

    Z, H, W = ct.shape
    rows = []
    for z in range(Z):
        roi_sl = np.asarray(roi[z])
        if (roi_sl > 0).sum() < 100:
            continue
        for y0 in range(0, H - CROP_SIZE + 1, CROP_STRIDE):
            for x0 in range(0, W - CROP_SIZE + 1, CROP_STRIDE):
                y1, x1 = y0 + CROP_SIZE, x0 + CROP_SIZE
                cyc, cxc = (y0 + y1) / 2, (x0 + x1) / 2
                # ROI overlap 확인 (crop 중심이 폐 안에 있어야)
                if roi_sl[int(cyc), int(cxc)] == 0:
                    continue
                pbin = assign_pbin(cyc, cxc, roi_sl)
                label = 0
                if has_lmask:
                    lsl = np.asarray(lmask[z, y0:y1, x0:x1])
                    if lsl.sum() > 0:
                        label = 1
                rows.append({
                    "patient_id": patient_id,
                    "safe_id": safe_id,
                    "local_z": z,
                    "crop_y0": y0, "crop_x0": x0,
                    "crop_y1": y1, "crop_x1": x1,
                    "position_bin": pbin,
                    "label": label,
                })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"manifest: {len(rows)} rows → {out_csv}")
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True, help="e.g. LUNG1-001")
    ap.add_argument("--device",  default="cuda")
    a  = ap.parse_args()

    patient = a.patient
    safe_id = find_safe_id(patient)
    if safe_id is None:
        print(f"ERROR: {patient} not found in CTBASE", file=sys.stderr); sys.exit(1)
    print(f"safe_id: {safe_id}")

    manifest_csv = os.path.join(INFER_DIR, f"test_{patient}_manifest.csv")
    scores_csv   = os.path.join(INFER_DIR, f"test_{patient}_scores.csv")
    ct_npy       = os.path.join(CTBASE, safe_id, "ct_hu.npy")

    make_manifest(patient, safe_id, manifest_csv)

    import subprocess, sys as _sys
    cmd = [
        sys.executable, INFER_PY,
        "--ct_npy",     ct_npy,
        "--manifest",   manifest_csv,
        "--checkpoint", CKPT,
        "--output",     scores_csv,
        "--device",     a.device,
        "--batch_size", "64",
    ]
    print("실행:", " ".join(cmd))
    ret = subprocess.run(cmd)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
