"""
단일 환자 성능 평가: PaDiM / RD4AD(P5) / NSCLC
- 입력: ct_hu.npy, roi_0_0.npy, lesion_mask_roi_0_0.npy
- 출력: 콘솔 AUROC + hit rate (파일 저장 없음)
"""
import sys, os, argparse, numpy as np
from collections import defaultdict

# 스크립트 위치 기준 경로 자동 설정
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch
import torch.nn as nn
import torchvision.models as tvm
from pathlib import Path


def roc_auc_score(labels, scores):
    labels = np.array(labels); scores = np.array(scores)
    order  = np.argsort(scores)[::-1]
    labels = labels[order]
    npos = labels.sum(); nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    tp = np.cumsum(labels); fp = np.cumsum(1 - labels)
    tpr = np.concatenate([[0], tp / npos])
    fpr = np.concatenate([[0], fp / nneg])
    return float(np.trapz(tpr, fpr))


# ── 모델 임포트 ──────────────────────────────────────────────────────────────
from position_aware_padim.padim_model import PaDiMModel
from position_aware_padim.feature_extractor_effnet_b0_scaffold import FeatureExtractorEffNetB0
from position_aware_padim.preprocessing import preprocess_ct_slice
from models.nsclc_classifier import run_nsclc, build_nsclc_hu_crop
import json

WEIGHTS   = os.path.join(_HERE, "weights")
DATA_ROOT = ""  # 실행 시 --ct / --roi 인수로 지정

PADIM_THRESHOLD = 12.20
PATCH_SIZE, STRIDE = 32, 16
RD4AD_CROP = 96
MIN_RUN = 2
HU_MIN_RD, HU_MAX_RD = -1000, 600
HU_MIN_NS, HU_MAX_NS = -1000, 200


def assign_position_bin(cy, cx, H, W) -> str:
    zone   = "upper" if cy < H/3 else ("middle" if cy < 2*H/3 else "lower")
    region = "central" if abs(cx - W/2) < W * 0.2 else "peripheral"
    return f"{zone}_{region}"


class _StudentDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_late  = nn.Sequential(nn.Conv2d(80,80,3,1,1), nn.BatchNorm2d(80), nn.ReLU(True))
        self.de_mid   = nn.Sequential(nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
                                      nn.Conv2d(80,40,3,1,1), nn.BatchNorm2d(40), nn.ReLU(True))
        self.de_early = nn.Sequential(nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
                                      nn.Conv2d(40,24,3,1,1), nn.BatchNorm2d(24), nn.ReLU(True))
    def forward(self, late):
        x=self.de_late(late); dl=x
        x=self.de_mid(x);    dm=x
        x=self.de_early(x);  de=x
        return dl, dm, de


def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    padim = PaDiMModel(
        selected_feature_indices_path=f"{WEIGHTS}/selected_feature_indices.npy",
        feature_dim=100, eps=1e-5)
    padim.load(f"{WEIGHTS}/position_bin_stats.npz")
    extractor = FeatureExtractorEffNetB0()

    # RD4AD teacher
    effb0_local = f"{WEIGHTS}/efficientnet_b0_rwightman-7f5810bc.pth"
    effb0_cache = Path.home() / ".cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth"
    effb0_path  = effb0_local if Path(effb0_local).exists() else str(effb0_cache)
    teacher = tvm.efficientnet_b0(weights=None)
    teacher.load_state_dict(torch.load(effb0_path, map_location="cpu", weights_only=True))
    teacher.eval().requires_grad_(False).to(device)

    # RD4AD student
    student = _StudentDecoder()
    ckpt  = torch.load(f"{WEIGHTS}/best_train_loss.pth", map_location=device, weights_only=False)
    state = ckpt.get("student_state_dict", ckpt.get("model_state_dict", ckpt))
    student.load_state_dict(state)
    student.eval().to(device)

    rd_feats = {}
    def _hook(name):
        def h(m,i,o): rd_feats[name] = o
        return h
    teacher.features[2].register_forward_hook(_hook("early"))
    teacher.features[3].register_forward_hook(_hook("mid"))
    teacher.features[4].register_forward_hook(_hook("late"))

    print(f"[INFO] 모델 로드 완료 (device={device})")
    return padim, extractor, teacher, student, rd_feats, device


def score_slice_padim(z, hu_vol, roi_vol, padim, extractor):
    """z 슬라이스의 모든 패치 → [(y0,x0,y1,x1,cy,cx,score)] """
    Z, H, W = hu_vol.shape
    hu_sl  = hu_vol[z]
    roi_sl = roi_vol[z]
    preprocessed = preprocess_ct_slice(hu_sl)

    coords = []
    for y0 in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x0 in range(0, W - PATCH_SIZE + 1, STRIDE):
            cy = y0 + PATCH_SIZE // 2; cx = x0 + PATCH_SIZE // 2
            if not roi_sl[cy, cx]: continue
            if roi_sl[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE].sum() < PATCH_SIZE*PATCH_SIZE*0.5:
                continue
            coords.append((y0, x0, y0+PATCH_SIZE, x0+PATCH_SIZE))

    if not coords:
        return []

    feats_448 = extractor.extract_patch_features(preprocessed, coords)
    feats_100 = feats_448[:, padim.selected_feature_indices]

    out = []
    for i, (y0, x0, y1, x1) in enumerate(coords):
        cy = (y0 + y1) // 2; cx = (x0 + x1) // 2
        pb = assign_position_bin(cy, cx, H, W)
        try:
            s = float(padim.score_patch(feats_100[i], pb))
        except Exception:
            s = 0.0
        out.append((y0, x0, y1, x1, cy, cx, s))
    return out


def rd4ad_cosine_dist(teacher, student, rd_feats, crop_t):
    with torch.no_grad():
        teacher(crop_t)
        tf_late = rd_feats["late"]
        dl, dm, de = student(tf_late)
    def cd(a, b):
        a=a.flatten(); b=b.flatten()
        return float(1 - torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))
    return (cd(rd_feats["late"], dl) + cd(rd_feats["mid"], dm) + cd(rd_feats["early"], de)) / 3.0


def score_slice_rd4ad(z, hu_vol, roi_vol, rd_teacher, rd_student, rd_feats, device):
    """z 슬라이스의 z-track 후보 수집용 raw 패치 반환 [(y0,x0,y1,x1,cosine,roi_ratio,P1)]"""
    Z, H, W = hu_vol.shape
    roi_sl = roi_vol[z]

    coords = []
    for y0 in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x0 in range(0, W - PATCH_SIZE + 1, STRIDE):
            cy = y0 + PATCH_SIZE // 2; cx = x0 + PATCH_SIZE // 2
            if not roi_sl[cy, cx]: continue
            patch_roi = roi_sl[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
            if patch_roi.sum() < PATCH_SIZE*PATCH_SIZE*0.5: continue
            coords.append((y0, x0, y0+PATCH_SIZE, x0+PATCH_SIZE,
                           float(patch_roi.sum()) / (PATCH_SIZE*PATCH_SIZE)))
    if not coords:
        return []

    out = []
    for y0, x0, y1, x1, roi_ratio in coords:
        ny0 = max(0, (y0+y1)//2 - RD4AD_CROP//2)
        nx0 = max(0, (x0+x1)//2 - RD4AD_CROP//2)
        ny0 = min(ny0, H - RD4AD_CROP); nx0 = min(nx0, W - RD4AD_CROP)

        def get_ch(dz):
            zz = max(0, min(Z-1, z+dz))
            sl = hu_vol[zz, ny0:ny0+RD4AD_CROP, nx0:nx0+RD4AD_CROP].astype(np.float32)
            return np.clip((sl - HU_MIN_RD) / (HU_MAX_RD - HU_MIN_RD), 0, 1)

        crop = torch.tensor(np.stack([get_ch(-1), get_ch(0), get_ch(1)], axis=0),
                            dtype=torch.float32).unsqueeze(0).to(device)

        cosine = rd4ad_cosine_dist(rd_teacher, rd_student, rd_feats, crop)
        P1 = cosine * roi_ratio
        out.append({"z": z, "y0": y0, "x0": x0, "y1": y1, "x1": x1,
                    "cosine": cosine, "roi_ratio": roi_ratio, "P1": P1})
    return out


def compute_tracks(all_raw):
    groups = defaultdict(list)
    for r in all_raw:
        groups[(r["y0"], r["x0"], r["y1"], r["x1"])].append(r)
    tracks = []
    for key, grp in groups.items():
        grp.sort(key=lambda x: x["z"])
        runs = []
        cur = [grp[0]]
        for i in range(1, len(grp)):
            if grp[i]["z"] - grp[i-1]["z"] == 1:
                cur.append(grp[i])
            else:
                runs.append(cur); cur = [grp[i]]
        runs.append(cur)
        for run in runs:
            if len(run) < MIN_RUN: continue
            tlen = len(run)
            p1_vals = sorted([r["P1"] for r in run], reverse=True)
            top3_p1 = sum(p1_vals[:3]) / min(3, tlen)
            p5 = top3_p1 * (tlen / 3.0)
            best = max(run, key=lambda x: x["P1"])
            tracks.append({
                "y0": key[0], "x0": key[1], "y1": key[2], "x1": key[3],
                "z_start": run[0]["z"], "z_end": run[-1]["z"],
                "z": best["z"], "track_len": tlen,
                "p5_score": p5, "cosine": best["cosine"],
                "all_z": [r["z"] for r in run],
            })
    tracks.sort(key=lambda x: x["p5_score"], reverse=True)
    return tracks


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct",       required=True,  help="ct_hu.npy 경로")
    ap.add_argument("--roi",      required=True,  help="roi_0_0.npy 경로")
    ap.add_argument("--lesion",   default=None,   help="lesion_mask_roi_0_0.npy (성능평가용, 없으면 생략)")
    ap.add_argument("--patient",  required=True,  help="환자 ID (예: LUNG1-001)")
    ap.add_argument("--safe_id",  required=True,  help="safe_id (예: NSCLC_LUNG1-001__5d369af301)")
    ap.add_argument("--out",      default="./out", help="출력 폴더")
    ap.add_argument("--top_k",    type=int, default=5)
    args = ap.parse_args()

    hu_vol  = np.load(args.ct).astype(np.float32)
    roi_vol = np.load(args.roi).astype(bool)
    les_vol = np.load(args.lesion).astype(bool) if args.lesion else None
    Z, H, W = hu_vol.shape

    if les_vol is not None:
        lesion_z_set = set(int(z) for z in np.where(les_vol.any(axis=(1,2)))[0])
        print(f"[INFO] volume={hu_vol.shape}  병변 z 범위: {min(lesion_z_set)}~{max(lesion_z_set)} ({len(lesion_z_set)}슬)")
    else:
        lesion_z_set = set()
        print(f"[INFO] volume={hu_vol.shape}  병변 마스크 없음 (성능평가 생략)")

    padim, extractor, rd_teacher, rd_student, rd_feats, device = load_models()
    print("[INFO] 전체 슬라이스 스코어링 시작...")

    # ── 1. PaDiM 전체 패치 수집 + z-track 상태 관리 ────────────────────────
    padim_scores, padim_labels = [], []
    rd_raw_all = []
    z_track_state: dict = {}   # (y0,x0) → 현재 연속 run 길이 (최대 50 리셋)

    for z in range(Z):
        if z % 30 == 0:
            print(f"  z={z}/{Z}  padim_patches={len(padim_scores)}", flush=True)

        # PaDiM
        patches = score_slice_padim(z, hu_vol, roi_vol, padim, extractor)
        padim_hits = {}  # (y0,x0) → patch info (P90 초과만)
        for y0, x0, y1, x1, cy, cx, score in patches:
            padim_scores.append(score)
            padim_labels.append(int(les_vol[z, cy, cx]) if les_vol is not None else 0)
            if score > PADIM_THRESHOLD:
                padim_hits[(y0, x0)] = {"y1": y1, "x1": x1, "score": score}

        # z-track 상태 업데이트 (main.py 동일: run > 50 리셋)
        new_state: dict = {}
        for key, hit in padim_hits.items():
            prev_run = z_track_state.get(key, 0)
            new_run = prev_run + 1
            if new_run > 50:
                new_run = 0
            new_state[key] = new_run
        z_track_state.clear()
        z_track_state.update(new_state)

        # run_len >= 2 인 패치만 RD4AD 수집
        for (y0, x0), run_len in z_track_state.items():
            if run_len < 2:
                continue
            hit = padim_hits[(y0, x0)]
            y1, x1 = hit["y1"], hit["x1"]
            roi_sl = roi_vol[z]
            pr = roi_sl[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
            roi_ratio = float(pr.sum()) / (PATCH_SIZE * PATCH_SIZE)
            ny0 = max(0, min((y0+y1)//2 - RD4AD_CROP//2, H - RD4AD_CROP))
            nx0 = max(0, min((x0+x1)//2 - RD4AD_CROP//2, W - RD4AD_CROP))
            def get_ch(dz, _ny0=ny0, _nx0=nx0, _z=z):
                zz = max(0, min(Z-1, _z+dz))
                sl = hu_vol[zz, _ny0:_ny0+RD4AD_CROP, _nx0:_nx0+RD4AD_CROP].astype(np.float32)
                return np.clip((sl - HU_MIN_RD)/(HU_MAX_RD - HU_MIN_RD), 0, 1)
            crop = torch.tensor(np.stack([get_ch(-1), get_ch(0), get_ch(1)], 0),
                                dtype=torch.float32).unsqueeze(0).to(device)
            cosine = rd4ad_cosine_dist(rd_teacher, rd_student, rd_feats, crop)
            P1 = cosine * roi_ratio
            # track_len = run_len (현재 슬라이스까지의 연속 길이)
            rd_raw_all.append({"z": z, "y0": y0, "x0": x0, "y1": y1, "x1": x1,
                               "cosine": cosine, "roi_ratio": roi_ratio, "P1": P1,
                               "run_len": run_len})

    print(f"\n[INFO] PaDiM 패치 수: {len(padim_scores)}  양성: {sum(padim_labels)}")

    # ── 2. PaDiM AUROC ──────────────────────────────────────────────────────
    if sum(padim_labels) > 0:
        padim_auroc = roc_auc_score(padim_labels, padim_scores)
        print(f"\n[PaDiM] patch AUROC = {padim_auroc:.4f}")
        # 임계값 12.2 기준 hit
        above = [(s > PADIM_THRESHOLD and l == 1) for s, l in zip(padim_scores, padim_labels)]
        below = [(s <= PADIM_THRESHOLD and l == 0) for s, l in zip(padim_scores, padim_labels)]
        tp = sum(1 for s,l in zip(padim_scores,padim_labels) if s>PADIM_THRESHOLD and l==1)
        fn = sum(1 for s,l in zip(padim_scores,padim_labels) if s<=PADIM_THRESHOLD and l==1)
        fp = sum(1 for s,l in zip(padim_scores,padim_labels) if s>PADIM_THRESHOLD and l==0)
        print(f"[PaDiM] threshold={PADIM_THRESHOLD}: TP={tp} FN={fn} FP={fp}  recall={tp/(tp+fn+1e-9):.3f}")
    else:
        print("[PaDiM] 병변 패치 없음 (lesion mask가 패치 중심에 해당 안 됨)")

    # ── 3. RD4AD 트랙 집계 → AUROC ─────────────────────────────────────────
    tracks = compute_tracks(rd_raw_all)
    print(f"\n[INFO] RD4AD 트랙 수: {len(tracks)}")

    # 트랙 라벨: 해당 트랙의 z + (y0,x0,y1,x1) 안에 lesion 픽셀이 하나라도 있으면 positive
    rd_p5_scores, rd_track_labels = [], []
    for t in tracks:
        has_lesion = False
        for z in t["all_z"]:
            if les_vol[z, t["y0"]:t["y1"], t["x0"]:t["x1"]].any():
                has_lesion = True; break
        rd_p5_scores.append(t["p5_score"])
        rd_track_labels.append(int(has_lesion))

    pos_tracks = sum(rd_track_labels)
    print(f"[INFO] 병변 트랙: {pos_tracks} / {len(tracks)}")
    if pos_tracks > 0:
        rd_auroc = roc_auc_score(rd_track_labels, rd_p5_scores)
        print(f"[RD4AD P5] track AUROC = {rd_auroc:.4f}")
        top10_hit = sum(rd_track_labels[:10])
        top5_hit  = sum(rd_track_labels[:5])
        top1_hit  = rd_track_labels[0] if rd_track_labels else 0
        print(f"[RD4AD P5] top-1 hit={top1_hit}  top-5 hit={top5_hit}/5  top-10 hit={top10_hit}/10")
        print(f"[RD4AD P5] top-5 트랙 상세:")
        for i, t in enumerate(tracks[:5]):
            print(f"  #{i+1} z={t['z']} z_range={t['z_start']}~{t['z_end']} p5={t['p5_score']:.4f} lesion={rd_track_labels[i]}")
    else:
        print("[RD4AD] 병변 트랙 없음")

    # ── 4. NSCLC top-K hit ─────────────────────────────────────────────────
    lung_zs = np.where(roi_vol.any(axis=(1,2)))[0]
    lung_zmin, lung_zmax = int(lung_zs.min()), int(lung_zs.max())

    print(f"\n[NSCLC] top-10 트랙 추론:")
    nsclc_probs, nsclc_labels = [], []
    for i, t in enumerate(tracks[:10]):
        tz = t["z"]
        ny0 = max(0, min((t["y0"]+t["y1"])//2 - RD4AD_CROP//2, H - RD4AD_CROP))
        nx0 = max(0, min((t["x0"]+t["x1"])//2 - RD4AD_CROP//2, W - RD4AD_CROP))
        lz_pct  = (tz - lung_zmin) / max(lung_zmax - lung_zmin, 1)
        roi_sl  = roi_vol[tz, ny0:ny0+RD4AD_CROP, nx0:nx0+RD4AD_CROP]
        crop_roi = float(roi_sl.sum()) / (RD4AD_CROP*RD4AD_CROP)
        res = run_nsclc(hu_vol, tz, ny0, nx0, lz_pct, crop_roi)
        prob = res["prob"]
        has_lesion = int(rd_track_labels[i])
        nsclc_probs.append(prob); nsclc_labels.append(has_lesion)
        print(f"  #{i+1} z={tz} p5={t['p5_score']:.4f} nsclc_prob={prob:.3f} lesion={has_lesion}")

    if sum(nsclc_labels) > 0 and len(set(nsclc_labels)) > 1:
        ns_auroc = roc_auc_score(nsclc_labels, nsclc_probs)
        print(f"[NSCLC] top-10 AUROC = {ns_auroc:.4f}")
    top5_nsclc_hit = sum(1 for p, l in zip(nsclc_probs[:5], nsclc_labels[:5]) if p >= 0.5 and l == 1)
    print(f"[NSCLC] top-5 중 prob≥0.5 && 병변 = {top5_nsclc_hit}개")

    print("\n" + "="*50)
    print("요약")
    print("="*50)
    if sum(padim_labels) > 0:
        print(f"PaDiM  patch AUROC : {padim_auroc:.4f}")
    if pos_tracks > 0:
        print(f"RD4AD  track AUROC : {rd_auroc:.4f}")
        print(f"RD4AD  top-1 hit   : {top1_hit}")
        print(f"RD4AD  top-5 hit   : {top5_hit}/5")
    if sum(nsclc_labels) > 0 and len(set(nsclc_labels)) > 1:
        print(f"NSCLC  top-10 AUROC: {ns_auroc:.4f}")
    print(f"NSCLC  top-5 hit(≥0.5): {top5_nsclc_hit}/5")

    # ── 5. score CSV 저장 → 카드 빌드 ──────────────────────────────────────
    import csv as csvmod, subprocess
    PATIENT   = args.patient
    SAFE_ID   = args.safe_id
    OUT_DIR   = args.out
    OUT_CSV   = os.path.join(OUT_DIR, f"score_{PATIENT}.csv")
    CARD_OUT  = os.path.join(OUT_DIR, "xai_card")
    CARD_PY   = os.path.join(_HERE, "build_dynamic_ref_card_any_candidate_rd4ad_v1_panel4_clinical_compact.py")

    os.makedirs(OUT_DIR, exist_ok=True)

    # rd_raw_all 에서 각 z-track 후보 행 생성 (label=1 설정해서 카드 스크립트가 선택)
    print(f"\n[CSV] score CSV 저장 중...")
    cols = ["patient_id","safe_id","local_z","crop_y0","crop_x0","crop_y1","crop_x1",
            "position_bin","label","score_late","score_mid","score_early","rd4ad_score"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csvmod.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rd_raw_all:
            cy = (r["y0"] + r["y1"]) // 2; cx = (r["x0"] + r["x1"]) // 2
            pb = assign_position_bin(cy, cx, H, W)
            # 카드 스크립트는 96px RD4AD 크롭 좌표를 기대함
            ny0 = max(0, min(cy - RD4AD_CROP//2, H - RD4AD_CROP))
            nx0 = max(0, min(cx - RD4AD_CROP//2, W - RD4AD_CROP))
            w.writerow({
                "patient_id":  PATIENT,
                "safe_id":     SAFE_ID,
                "local_z":     r["z"],
                "crop_y0":     ny0,
                "crop_x0":     nx0,
                "crop_y1":     ny0 + RD4AD_CROP,
                "crop_x1":     nx0 + RD4AD_CROP,
                "position_bin": pb,
                "label":       1,
                "score_late":  round(r["cosine"], 6),
                "score_mid":   round(r["cosine"], 6),
                "score_early": round(r["cosine"], 6),
                "rd4ad_score": round(r["cosine"], 6),
            })
    print(f"[CSV] 저장 완료: {OUT_CSV}  ({len(rd_raw_all)}행)")

    # 카드 빌드
    print(f"\n[CARD] 카드 빌드 시작...")
    env = os.environ.copy()
    env.update({
        "ALLOW_CARD_RENDER": "1",
        "ALLOW_CT_LOAD": "1",
        "ALLOW_SOURCE_IMAGE_READ": "1",
        "ALLOW_PNG_WRITE": "1",
    })
    cmd = [
        sys.executable, CARD_PY,
        "--patient", PATIENT,
        "--score_csv", OUT_CSV,
        "--out_root", CARD_OUT,
        "--render", "--confirm",
    ]
    ret = subprocess.run(cmd, env=env, capture_output=False)
    if ret.returncode == 0:
        print(f"[CARD] 완료: {CARD_OUT}")
    else:
        print(f"[CARD] 오류 (exit={ret.returncode})")


if __name__ == "__main__":
    main()
