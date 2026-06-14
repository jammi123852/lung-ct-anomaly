# -*- coding: utf-8 -*-
"""
rd4ad_e2_infer_single_patient_v1.py

E2 RD4AD (EfficientNet-B0 teacher) 로 단일 환자 scoring.
usage: python rd4ad_e2_infer_single_patient_v1.py --patient LUNG1-308 --device cuda
"""
import os, sys, csv, json, argparse
from pathlib import Path
import numpy as np

ROOT     = "/home/jinhy/project/lung-ct-anomaly"
CTBASE   = "/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy"
OUT_DIR  = os.path.join(ROOT, "outputs/end/rd4ad_e2_infer_v1")
CKPT     = os.path.join(ROOT, "outputs/end/rd4ad_e2_best_model_package_v1/checkpoints/best_train_loss.pth")
EFFB0_W  = "/home/jinhy/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth"

HU_MIN, HU_MAX = -1000.0, 600.0
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
    mid_x = (bx0 + bx1) / 2
    ri = min(int(ry * 3), 2)
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
    has_lmask  = os.path.exists(lmask_path)
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
                if roi_sl[int(cyc), int(cxc)] == 0:
                    continue
                label = 0
                if has_lmask:
                    if np.asarray(lmask[z, y0:y1, x0:x1]).sum() > 0:
                        label = 1
                rows.append({"patient_id": patient_id, "safe_id": safe_id,
                              "local_z": z, "crop_y0": y0, "crop_x0": x0,
                              "crop_y1": y1, "crop_x1": x1,
                              "position_bin": assign_pbin(cyc, cxc, roi_sl),
                              "label": label})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"manifest: {len(rows)} rows → {out_csv}")
    return rows


def build_lung3ch_crop(ct_arr, z, y0, x0, y1, x1):
    Z, H, W = ct_arr.shape
    zm, zp  = max(z - 1, 0), min(z + 1, Z - 1)
    pad_top = max(0, -y0); pad_bot = max(0, y1 - H)
    pad_lft = max(0, -x0); pad_rgt = max(0, x1 - W)
    cy0, cy1 = max(0, y0), min(H, y1)
    cx0, cx1 = max(0, x0), min(W, x1)
    needs_pad = pad_top > 0 or pad_bot > 0 or pad_lft > 0 or pad_rgt > 0
    def _sl(zi):
        s = np.asarray(ct_arr[zi, cy0:cy1, cx0:cx1], dtype="float32")
        s = (s.clip(HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
        if needs_pad:
            mode = "reflect" if (cy1-cy0 > 1 and cx1-cx0 > 1) else "edge"
            s = np.pad(s, ((pad_top, pad_bot), (pad_lft, pad_rgt)), mode=mode)
        return s
    return np.stack([_sl(zm), _sl(z), _sl(zp)], axis=0).astype("float32")


def build_e2_model(device):
    import torch
    import torch.nn as nn
    import torchvision.models as tvm

    effnet = tvm.efficientnet_b0(weights=None)
    effnet.load_state_dict(torch.load(EFFB0_W, map_location="cpu", weights_only=True))
    effnet.eval(); effnet.requires_grad_(False)
    teacher = effnet.to(device)

    class StudentDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.de_late  = nn.Sequential(nn.Conv2d(80,80,3,1,1), nn.BatchNorm2d(80), nn.ReLU(inplace=True))
            self.de_mid   = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                           nn.Conv2d(80,40,3,1,1), nn.BatchNorm2d(40), nn.ReLU(inplace=True))
            self.de_early = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                           nn.Conv2d(40,24,3,1,1), nn.BatchNorm2d(24), nn.ReLU(inplace=True))
        def forward(self, late_feat):
            x = self.de_late(late_feat);  de_l = x
            x = self.de_mid(x);           de_m = x
            x = self.de_early(x);         de_e = x
            return de_l, de_m, de_e

    student = StudentDecoder().to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    if "student_state_dict" in ckpt:
        student.load_state_dict(ckpt["student_state_dict"])
    elif "model_state_dict" in ckpt:
        student.load_state_dict(ckpt["model_state_dict"])
    else:
        student.load_state_dict(ckpt)
    student.eval()
    print(f"E2 model loaded (epoch={ckpt.get('epoch','?')})")

    tf_feats = {}
    for name, module in [("early", teacher.features[2]),
                          ("mid",   teacher.features[3]),
                          ("late",  teacher.features[4])]:
        def _hook(m, i, o, _n=name): tf_feats[_n] = o
        module.register_forward_hook(_hook)

    return teacher, student, tf_feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--device",  default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    patient = a.patient
    safe_id = find_safe_id(patient)
    if not safe_id:
        print(f"ERROR: {patient} not found", file=sys.stderr); sys.exit(1)
    print(f"safe_id: {safe_id}")

    manifest_csv = os.path.join(OUT_DIR, f"test_{patient}_manifest.csv")
    scores_csv   = os.path.join(OUT_DIR, f"test_{patient}_scores.csv")

    if not os.path.exists(manifest_csv):
        rows = make_manifest(patient, safe_id, manifest_csv)
    else:
        rows = list(csv.DictReader(open(manifest_csv, encoding="utf-8-sig")))
        print(f"manifest 재사용: {len(rows)} rows")

    import torch, torch.nn.functional as F
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ct_arr = np.load(os.path.join(CTBASE, safe_id, "ct_hu.npy"), mmap_mode="r")
    teacher, student, tf_feats = build_e2_model(device)

    out_rows = []
    for i in range(0, len(rows), a.batch_size):
        batch_rows = rows[i:i + a.batch_size]
        crops = []
        for r in batch_rows:
            c = build_lung3ch_crop(ct_arr, int(r["local_z"]),
                                   int(r["crop_y0"]), int(r["crop_x0"]),
                                   int(r["crop_y1"]), int(r["crop_x1"]))
            crops.append(c)
        batch_t = torch.from_numpy(np.stack(crops, axis=0)).to(device)
        with torch.no_grad():
            teacher(batch_t)
            tf_late  = tf_feats["late"]
            tf_mid   = tf_feats["mid"]
            tf_early = tf_feats["early"]
            de_l, de_m, de_e = student(tf_late)
            s_l = (1 - F.cosine_similarity(de_l,  tf_late,  dim=1)).mean(dim=(1,2))
            s_m = (1 - F.cosine_similarity(de_m,  tf_mid,   dim=1)).mean(dim=(1,2))
            s_e = (1 - F.cosine_similarity(de_e,  tf_early, dim=1)).mean(dim=(1,2))
            scores = ((s_l + s_m + s_e) / 3).cpu().numpy()

        for j, r in enumerate(batch_rows):
            out_rows.append({**r,
                             "score_late":  round(float(s_l[j].item()), 6),
                             "score_mid":   round(float(s_m[j].item()), 6),
                             "score_early": round(float(s_e[j].item()), 6),
                             "rd4ad_score": round(float(scores[j]), 6)})
        if (i // a.batch_size) % 20 == 0:
            print(f"  {i+len(batch_rows)}/{len(rows)} done")

    with open(scores_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"DONE: {len(out_rows)} rows → {scores_csv}")


if __name__ == "__main__":
    main()
