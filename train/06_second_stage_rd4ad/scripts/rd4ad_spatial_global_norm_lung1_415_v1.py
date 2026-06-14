# -*- coding: utf-8 -*-
"""
전체 602개 crop에 대해 spatial map 생성 → global normalization → 병변 overlay 분석.
model forward: 전 crop (heatmap 목적, score CSV overwrite 없음).
"""
import os, csv, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = "/home/jinhy/project/lung-ct-anomaly"
CTBASE  = "/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy"
SAFE_ID = "NSCLC_LUNG1-415__75e5b68d83"
SCORE_CSV = os.path.join(ROOT, "outputs/end/rd4ad_lung_mip3ch_infer_v1/test_LUNG1-415_scores.csv")
CKPT      = os.path.join(ROOT, "outputs/end/rd4ad_lung_mip3ch_infer_v1/best_train_loss.pth")
OUT_ROOT  = os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/rd4ad_spatial_map_preflight_lung1_415_z99_v1")
FONT_KO   = "/mnt/c/Windows/Fonts/malgun.ttf"

HU_MIN, HU_MAX = -1000.0, 600.0
CROP_SIZE = 96
BATCH = 64

# ── 헬퍼 ──────────────────────────────────────────────────────────────────────
def build_mip3ch_crop(ct_arr, local_z, y0, x0, y1, x1):
    Z, H, W = ct_arr.shape
    z = int(local_z); y0,x0,y1,x1 = int(y0),int(x0),int(y1),int(x1)
    pt = max(0,-y0); pb = max(0,y1-H); pl = max(0,-x0); pr = max(0,x1-W)
    needs = any([pt,pb,pl,pr])
    cy0=max(0,y0); cy1=min(H,y1); cx0=max(0,x0); cx1=min(W,x1)
    def _w(sl): return np.clip((sl.astype("float32")-HU_MIN)/(HU_MAX-HU_MIN),0,1)
    def _ch(zi):
        zi=int(np.clip(zi,0,Z-1)); s=_w(ct_arr[zi,cy0:cy1,cx0:cx1])
        if needs:
            pm="reflect" if (cy1-cy0>1)and(cx1-cx0>1) else "edge"
            s=np.pad(s,((pt,pb),(pl,pr)),mode=pm)
        return s
    def _mip(zs): return np.stack([_ch(zi) for zi in zs]).max(0)
    return np.stack([_mip([z-3,z-2,z-1]),_mip([z-1,z,z+1]),_mip([z+1,z+2,z+3])]).astype("float32")

def build_student_decoder():
    import torch.nn as nn
    class SD(nn.Module):
        def __init__(self):
            super().__init__()
            self.de_layer3 = nn.Sequential(nn.Conv2d(256,256,3,1,1),nn.BatchNorm2d(256),nn.ReLU(True))
            self.de_layer2 = nn.Sequential(nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),nn.Conv2d(256,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(True))
            self.de_layer1 = nn.Sequential(nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),nn.Conv2d(128,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(True))
        def forward(self,x):
            x=self.de_layer3(x); de3=x; x=self.de_layer2(x); de2=x; x=self.de_layer1(x); de1=x
            return de3,de2,de1
    return SD()

def load_model(ckpt, device):
    import torchvision.models as M
    teacher = M.resnet18(weights=None).to(device); teacher.eval(); teacher.requires_grad_(False)
    student = build_student_decoder().to(device)
    sd = torch.load(str(ckpt), map_location=device, weights_only=True)
    student.load_state_dict(sd["student_state_dict"]); student.eval()
    tf={}
    for nm,mod in [("layer1",teacher.layer1),("layer2",teacher.layer2),("layer3",teacher.layer3)]:
        def _h(m,i,o,_n=nm): tf[_n]=o
        mod.register_forward_hook(_h)
    return teacher,student,tf

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    rows = list(csv.DictReader(open(SCORE_CSV, encoding="utf-8-sig")))
    print(f"total crops: {len(rows)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    teacher, student, tf_feats = load_model(CKPT, device)

    ct = np.load(os.path.join(CTBASE, SAFE_ID, "ct_hu.npy"), mmap_mode="r")

    # 전체 배치 forward → layer1 spatial map (24×24) 수집
    all_sm_l1 = []  # per-crop (24,24) raw cosine distance
    all_sm_l2 = []

    for i in range(0, len(rows), BATCH):
        batch_rows = rows[i:i+BATCH]
        crops = []
        for r in batch_rows:
            crops.append(build_mip3ch_crop(ct, int(r["local_z"]),
                                           int(r["crop_y0"]), int(r["crop_x0"]),
                                           int(r["crop_y1"]), int(r["crop_x1"])))
        bt = torch.from_numpy(np.stack(crops)).to(device)
        with torch.no_grad():
            teacher(bt)
            tf3=tf_feats["layer3"]; tf2=tf_feats["layer2"]; tf1=tf_feats["layer1"]
            de3,de2,de1 = student(tf3)
            # layer1: de1 vs tf1 → (B,24,24)
            sm1 = (1-F.cosine_similarity(de1, tf1, dim=1)).cpu().numpy()
            sm2 = (1-F.cosine_similarity(de2, tf2, dim=1)).cpu().numpy()
        all_sm_l1.append(sm1); all_sm_l2.append(sm2)
        if (i//BATCH+1) % 3 == 0 or i+BATCH >= len(rows):
            print(f"  {min(i+BATCH,len(rows))}/{len(rows)}")

    all_sm_l1 = np.concatenate(all_sm_l1, axis=0)  # (602,24,24)
    all_sm_l2 = np.concatenate(all_sm_l2, axis=0)  # (602,12,12)
    print(f"all_sm_l1: {all_sm_l1.shape}  min={all_sm_l1.min():.4f} max={all_sm_l1.max():.4f}")
    print(f"all_sm_l2: {all_sm_l2.shape}  min={all_sm_l2.min():.4f} max={all_sm_l2.max():.4f}")

    # global min/max
    g1_min, g1_max = all_sm_l1.min(), all_sm_l1.max()
    g2_min, g2_max = all_sm_l2.min(), all_sm_l2.max()

    def global_norm(m, gmin, gmax):
        return (m - gmin) / max(gmax - gmin, 1e-8)

    def upsample96(m):
        t = torch.from_numpy(m[np.newaxis,np.newaxis].astype("float32"))
        return F.interpolate(t,(CROP_SIZE,CROP_SIZE),mode="bilinear",align_corners=False)[0,0].numpy()

    # z=99 crop index 찾기 (y0=176, x0=128)
    idx99 = None
    for ii, r in enumerate(rows):
        if int(r["local_z"])==99 and int(r["crop_y0"])==176 and int(r["crop_x0"])==128:
            idx99 = ii; break
    assert idx99 is not None, "z=99 row not found"
    print(f"z=99 row index: {idx99}  label={rows[idx99]['label']}  rd4ad_score={rows[idx99]['rd4ad_score']}")

    sm1_z99_raw = all_sm_l1[idx99]  # (24,24) raw
    sm2_z99_raw = all_sm_l2[idx99]  # (12,12) raw

    sm1_gn = global_norm(sm1_z99_raw, g1_min, g1_max)
    sm2_gn = global_norm(sm2_z99_raw, g2_min, g2_max)
    sm_comb_gn = (upsample96(sm1_gn) + upsample96(sm2_gn)) / 2.0

    sm1_up = upsample96(sm1_gn)
    sm2_up = upsample96(sm2_gn)

    # 전체 분포에서 이 crop의 percentile
    flat_all = all_sm_l1.flatten()
    pct_raw = float(np.mean(flat_all < sm1_z99_raw.mean()))
    print(f"z=99 crop: raw_mean={sm1_z99_raw.mean():.4f}  percentile in all crops={pct_raw*100:.1f}%")

    # 병변 마스크 crop
    lmask = np.load(os.path.join(CTBASE, SAFE_ID, "lesion_mask_roi_0_0.npy"), mmap_mode="r")
    Y0,X0,Y1,X1 = 176,128,272,224
    mask_crop = np.asarray(lmask[99,Y0:Y1,X0:X1]).astype("float32")
    contour = ndimage.binary_dilation(mask_crop>0,iterations=2).astype("float32") - (mask_crop>0).astype("float32")

    # background (MIP ch1)
    def _w(sl): return np.clip((sl.astype("float32")-HU_MIN)/(HU_MAX-HU_MIN),0,1)
    def _mip_bg(zs): return np.stack([_w(ct[max(0,min(ct.shape[0]-1,z)),Y0:Y1,X0:X1]) for z in zs]).max(0)
    bg = _mip_bg([98,99,100])

    import matplotlib.cm as cm
    cmap_fn = cm.get_cmap("jet")
    def make_ov(bg_01, map_01, contour_bin, alpha=0.5):
        colored = cmap_fn(map_01)[:,:,:3]
        ov = alpha*colored + (1-alpha)*np.stack([bg_01]*3,-1)
        ov[contour_bin>0] = [0.05,1.0,0.05]
        return np.clip(ov*255,0,255).astype("uint8")

    # ── 그림 1: global norm overlay ───────────────────────────────────────────
    fig, axes = plt.subplots(1,4,figsize=(14,3.5),facecolor="#111")
    axes[0].imshow(bg,cmap="gray",vmin=0,vmax=1)
    axes[0].contour(mask_crop,levels=[0.5],colors=["#00ff00"],linewidths=1.5)
    axes[0].set_title("Patch\n+ lesion(green)",color="white",fontsize=8); axes[0].axis("off")

    im1=axes[1].imshow(sm1_up,cmap="jet",vmin=0,vmax=1)
    axes[1].contour(mask_crop,levels=[0.5],colors=["#00ff00"],linewidths=1.5)
    axes[1].set_title(f"Layer1 GLOBAL norm\n(z99 pct={pct_raw*100:.0f}%)",color="white",fontsize=8); axes[1].axis("off")
    fig.colorbar(im1,ax=axes[1],fraction=0.046,pad=0.04)

    axes[2].imshow(make_ov(bg,sm1_up,contour))
    axes[2].set_title("Layer1 overlay (global)\n+ lesion contour",color="white",fontsize=8); axes[2].axis("off")

    axes[3].imshow(make_ov(bg,sm_comb_gn,contour))
    axes[3].set_title("L1+L2 combined (global)\n+ lesion contour",color="white",fontsize=8); axes[3].axis("off")

    fig.suptitle(f"RD4AD Global-Normalized Spatial Anomaly  LUNG1-415 z=99 label=1  [global min={g1_min:.4f} max={g1_max:.4f}]",
                 color="#aaaaff",fontsize=8)
    fig.tight_layout()
    out1 = os.path.join(OUT_ROOT,"spatial_overlay_layer1_global_norm_lesion.png")
    fig.savefig(out1,dpi=120,bbox_inches="tight",facecolor="#111"); plt.close(fig)
    print(f"saved: {out1}")

    # ── 그림 2: lesion vs non-lesion (global norm) ────────────────────────────
    lesion_px = mask_crop > 0; non_px = ~lesion_px
    fig2, axes2 = plt.subplots(1,3,figsize=(10,3.5),facecolor="#111")
    for ax, smap, name in zip(axes2,[sm1_up,sm2_up,sm_comb_gn],["Layer1","Layer2","L1+L2"]):
        ls=smap[lesion_px]; ns=smap[non_px]
        ax.hist(ns.flatten(),bins=20,alpha=0.6,color="#5599ff",label=f"non-lesion n={non_px.sum()}")
        ax.hist(ls.flatten(),bins=10,alpha=0.8,color="#ff5555",label=f"lesion n={lesion_px.sum()}")
        ax.set_title(f"{name} [global norm]\nlesion={ls.mean():.3f}  non={ns.mean():.3f}  diff={ls.mean()-ns.mean():+.3f}",
                    color="white",fontsize=8)
        ax.legend(fontsize=7); ax.tick_params(colors="gray",labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#555")
        ax.set_facecolor("#1a1a1a")
        print(f"{name} global: lesion={ls.mean():.4f}  non={ns.mean():.4f}  diff={ls.mean()-ns.mean():+.4f}")

    fig2.suptitle("Score dist (global norm): lesion vs non-lesion  [LUNG1-415 z=99]",color="#aaaaff",fontsize=9)
    fig2.tight_layout()
    out2 = os.path.join(OUT_ROOT,"lesion_score_distribution_global_norm.png")
    fig2.savefig(out2,dpi=100,bbox_inches="tight",facecolor="#111"); plt.close(fig2)
    print(f"saved: {out2}")

    # ── 그림 3: 전체 602개 crop score 분포 (l1 mean per crop) ─────────────────
    crop_means = all_sm_l1.mean(axis=(1,2))  # (602,)
    fig3, ax3 = plt.subplots(figsize=(8,3.5),facecolor="#111")
    ax3.hist(crop_means, bins=40, color="#5599ff", alpha=0.8, label="all crops")
    ax3.axvline(sm1_z99_raw.mean(), color="#ff5555", linewidth=2,
                label=f"z=99 label=1 (mean={sm1_z99_raw.mean():.4f})")
    # label=1 crop들 표시
    label1_means = [crop_means[ii] for ii,r in enumerate(rows) if r.get("label")=="1"]
    for lm in label1_means:
        ax3.axvline(lm, color="#ffaa00", linewidth=1.5, alpha=0.8)
    ax3.set_title("Layer1 spatial map mean per crop (all 602 crops)\nred=z99 label=1, orange=all label=1",
                  color="white",fontsize=9)
    ax3.legend(fontsize=8); ax3.tick_params(colors="gray")
    for sp in ax3.spines.values(): sp.set_color("#555")
    ax3.set_facecolor("#1a1a1a")
    fig3.tight_layout()
    out3 = os.path.join(OUT_ROOT,"all_crop_layer1_mean_distribution.png")
    fig3.savefig(out3,dpi=100,bbox_inches="tight",facecolor="#111"); plt.close(fig3)
    print(f"saved: {out3}")

    # 요약
    print(f"\n=== 전체 분포 요약 ===")
    print(f"전체 602개 crop layer1 mean: {crop_means.mean():.4f} ± {crop_means.std():.4f}")
    print(f"z=99 label=1: {sm1_z99_raw.mean():.4f}  (전체 중 {pct_raw*100:.1f}% percentile)")
    label0_means = [crop_means[ii] for ii,r in enumerate(rows) if r.get("label")=="0"]
    label1_means_arr = np.array(label1_means)
    label0_means_arr = np.array(label0_means)
    print(f"label=1 crops mean: {label1_means_arr.mean():.4f}")
    print(f"label=0 crops mean: {label0_means_arr.mean():.4f}")
    print(f"label1 vs label0 diff: {label1_means_arr.mean()-label0_means_arr.mean():+.4f}")

if __name__ == "__main__":
    main()
