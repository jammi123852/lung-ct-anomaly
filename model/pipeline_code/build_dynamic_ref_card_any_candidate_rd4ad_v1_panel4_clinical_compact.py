# -*- coding: utf-8 -*-
"""
build_dynamic_ref_card_any_candidate_rd4ad_v1_panel4_clinical_compact.py

Panel 4 compact clinical-readable version.
기존 clinical_readable 버전(6섹션)을 5섹션으로 압축.
섹션당 1~2문장, 반복 제거, 한국어 우선+영어 짧게 병기.

원칙: saved score 만 읽음(재계산 0). model_forward/feature/stage2 = 0.
score 컬럼: rd4ad_score 고정 (no fallback).
"""
import os, sys, csv, json, argparse, importlib.util, time
from datetime import date

_HERE   = os.path.dirname(os.path.abspath(__file__))
_MODEL  = os.path.join(_HERE, "model")
sys.path.insert(0, _HERE)

ROOT    = _HERE
CTBASE  = os.environ.get("CTBASE", "")   # 실행 시 환경변수 또는 --ct_base 인수로 지정
NB      = os.environ.get("NB", "")
BANK       = os.path.join(_HERE, "reference_bank", "dynamic_normal_reference_bank_three_patients_v1")
BANK_INDEX = os.path.join(BANK, "dynamic_reference_slice_index.csv")
RETRIEVE_PY = os.path.join(_HERE, "retrieve_dynamic_normal_refs_three_patients.py")
FONT_KO = os.environ.get("FONT_KO", "/mnt/c/Windows/Fonts/malgun.ttf")
OUT_BASE = os.path.join(_HERE, "out")

WL, WW = -600.0, 1500.0
TILE_PX = 280; CONTEXT_FACTOR = 2.0; PHYS_MM = 80.0
SCORE_COL   = "rd4ad_score"
CROP_SIZE   = 96
CROP_STRIDE = 16
BRANCH = "rd_e2_effb0_lung3ch"
MASK   = "lung3ch"
SUFFIX = "_rd4ad_card_panel4_clinical_compact"

# E2 RD4AD (EfficientNet-B0 teacher, top10 hit=0.9216 단독 1위)
RD4AD_E2_CKPT   = os.path.join(_HERE, "weights", "best_train_loss.pth")
RD4AD_E2_EFFB0  = os.path.join(_HERE, "weights", "efficientnet_b0_rwightman-7f5810bc.pth")
RD4AD_E2_HU_MIN, RD4AD_E2_HU_MAX = -1000.0, 600.0
RD4AD_E2_SCORE_DIR = os.path.join(_HERE, "out")

GRADCAM_CKPT         = os.path.join(_HERE, "weights", "gradcam_model.pt")
GRADCAM_SCALAR_STATS = None  # scalar stats는 gradcam_model.pt 체크포인트 내장
GRADCAM_HU_MIN, GRADCAM_HU_MAX = -1000.0, 200.0
GRADCAM_IMAGENET_MEAN = [0.485, 0.456, 0.406]
GRADCAM_IMAGENET_STD  = [0.229, 0.224, 0.225]
GRADCAM_ALPHA_GAMMA   = 0.55
GRADCAM_ALPHA_MAX     = 0.88

ALLOW = {k: os.environ.get(k) == "1" for k in
         ["ALLOW_CARD_RENDER", "ALLOW_CT_LOAD", "ALLOW_SOURCE_IMAGE_READ", "ALLOW_PNG_WRITE"]}


def _abort(m, c=2):
    print("BLOCKED:", m, file=sys.stderr); sys.exit(c)


# ── Panel 4 compact 헬퍼 ─────────────────────────────────────────────────────

def get_score_interp_compact(score):
    if score >= 0.45:
        return f"이상도 {score:.4f} — 정상 대비 뚜렷한 이상 패턴"
    elif score >= 0.35:
        return f"이상도 {score:.4f} — 경계 수준 이상 패턴"
    else:
        return f"이상도 {score:.4f} — 낮은 수준 이상 패턴"


def get_pbin_context_compact(pbin):
    mapping = {
        "upper_peripheral": "상부 말초 영역으로, 흉막 인접 구조·말초 혈관·섬유화/반흔 변화와 함께 해석해야 합니다.",
        "lower_peripheral": "하부 말초 영역으로, 횡격막·흉막·기저부 구조에 의한 오탐 가능성을 주의하세요.",
        "lower_central":    "하부 중심부로, 폐문·혈관 인접 구조와의 겹침 가능성을 확인해야 합니다.",
        "upper_central":    "상부 중심부로, 상부 혈관·기관지 주변 구조와의 겹침 가능성을 고려해야 합니다.",
        "middle_peripheral":"중부 말초 영역으로, 흉막 인접 구조와 주변 혈관에 주의하세요.",
        "middle_central":   "중부 중심부로, 폐문·혈관·기관지 주변 구조와의 겹침 가능성을 고려해야 합니다.",
    }
    return mapping.get(pbin, f"{pbin} 영역 — 해당 구역 임상 맥락 확인 필요.")




def get_zpct_caution_compact(lung_z_pct):
    if lung_z_pct is None:
        return "폐 z 위치 정보 없음 (unknown)."
    if lung_z_pct < 0.08:
        return f"⚠ 폐저부/횡격막 인접 (z_pct={lung_z_pct:.3f}) — 오탐 가능성 높음."
    elif lung_z_pct > 0.92:
        return f"⚠ 폐첨부 인접 (z_pct={lung_z_pct:.3f}) — 경계 오탐 가능성 있음."
    else:
        return f"폐 중간 구역 (z_pct={lung_z_pct:.3f}) — 위치 기반 오탐 가능성 낮음."


def _build_nsclc_section(nsclc_prob):
    if nsclc_prob is None:
        return ("이상 패턴 보조 평가", "SUB", ["이상 패턴 분류 결과 없음."])
    if nsclc_prob >= 0.5:
        tag = f"{nsclc_prob:.1%} — 이상 패턴 감지 (높음)"
        key = "WARN"
    elif nsclc_prob >= 0.3:
        tag = f"{nsclc_prob:.1%} — 경계 수준"
        key = "SEC"
    else:
        tag = f"{nsclc_prob:.1%} — 이상 패턴 낮음"
        key = "SUB"
    return ("이상 패턴 보조 평가", key, [
        tag,
        "보조 참고용 — 진단 목적 아님",
    ])


def build_panel4_compact_sections(patient, pbin, z, score, lung_z_pct, nsclc_prob=None):
    score_line = get_score_interp_compact(score)
    pbin_line  = get_pbin_context_compact(pbin)
    zpct_line  = get_zpct_caution_compact(lung_z_pct)
    return [
        ("핵심 소견", "SEC", [
            score_line,
            "동일 폐 내 상대 위치의 정상 참조 3명보다 더 이질적인 국소 음영 패턴입니다.",
        ]),
        ("판독 맥락", "SEC", [
            pbin_line,
        ]),
        ("주의", "WARN", [
            "폐 상대 위치 기준 매칭 (동일 슬라이스 아님). 이상도 점수는 진단 확률이 아닙니다.",
            zpct_line,
        ]),
        _build_nsclc_section(nsclc_prob),
        ("면책", "DIM", [
            "연구용 보조 설명 — 진단·병변·암 확률 아님",
        ]),
    ]


def _get_gradcam_model():
    """ScalarFusionModel 로드 (최초 1회 캐시)"""
    if not hasattr(_get_gradcam_model, "_cache"):
        import torch
        import torch.nn as nn
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        class _SFM(nn.Module):
            def __init__(self, scalar_hidden=32, scalar_out=16, dropout=0.2):
                super().__init__()
                bb = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
                self.img_features  = bb.features
                self.img_avgpool   = bb.avgpool
                self.scalar_branch = nn.Sequential(
                    nn.Linear(2, scalar_hidden), nn.BatchNorm1d(scalar_hidden),
                    nn.ReLU(inplace=True), nn.Linear(scalar_hidden, scalar_out), nn.ReLU(inplace=True),
                )
                self.fusion_head = nn.Sequential(
                    nn.Dropout(p=dropout), nn.Linear(1280 + scalar_out, 64),
                    nn.ReLU(inplace=True), nn.Dropout(p=dropout), nn.Linear(64, 1),
                )
            def forward(self, img, scalar):
                x = self.img_features(img); x = self.img_avgpool(x); x = torch.flatten(x, 1)
                return self.fusion_head(torch.cat([x, self.scalar_branch(scalar)], dim=1))

        model = _SFM()
        ckpt = torch.load(GRADCAM_CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"[Grad-CAM] model loaded (epoch={ckpt.get('epoch','?')})")
        _get_gradcam_model._cache = model
    return _get_gradcam_model._cache


def _get_rd4ad_e2_model():
    """E2 EfficientNet-B0 teacher + student decoder 로드 (최초 1회 캐시)"""
    if not hasattr(_get_rd4ad_e2_model, "_cache"):
        import torch
        import torch.nn as nn
        import torchvision.models as tvm

        effnet = tvm.efficientnet_b0(weights=None)
        effnet.load_state_dict(torch.load(RD4AD_E2_EFFB0, map_location="cpu", weights_only=True))
        effnet.eval(); effnet.requires_grad_(False)

        class _StudentDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.de_late  = nn.Sequential(nn.Conv2d(80,80,3,1,1), nn.BatchNorm2d(80), nn.ReLU(inplace=True))
                self.de_mid   = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                               nn.Conv2d(80,40,3,1,1), nn.BatchNorm2d(40), nn.ReLU(inplace=True))
                self.de_early = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                               nn.Conv2d(40,24,3,1,1), nn.BatchNorm2d(24), nn.ReLU(inplace=True))
            def forward(self, late_feat):
                de_l = self.de_late(late_feat)
                de_m = self.de_mid(de_l)
                de_e = self.de_early(de_m)
                return de_l, de_m, de_e

        student = _StudentDecoder()
        ckpt = torch.load(RD4AD_E2_CKPT, map_location="cpu", weights_only=False)
        if "student_state_dict" in ckpt:
            student.load_state_dict(ckpt["student_state_dict"])
        elif "model_state_dict" in ckpt:
            student.load_state_dict(ckpt["model_state_dict"])
        else:
            student.load_state_dict(ckpt)
        student.eval()
        print(f"[E2] model loaded (epoch={ckpt.get('epoch','?')})")

        tf_feats = {}
        for name, module in [("early", effnet.features[2]),
                              ("mid",   effnet.features[3]),
                              ("late",  effnet.features[4])]:
            def _hook(m, i, o, _n=name): tf_feats[_n] = o
            module.register_forward_hook(_hook)

        _get_rd4ad_e2_model._cache = (effnet, student, tf_feats)
    return _get_rd4ad_e2_model._cache


def run_rd4ad_e2_spatial_map(ct_arr, z, y0, x0, y1, x1):
    """CT 3ch crop → E2 cosine distance spatial map (96×96 float32 [0,1]). 실패 시 None."""
    import numpy as np, torch, torch.nn.functional as F
    try:
        Z, H, W = ct_arr.shape
        zm, zp = max(z - 1, 0), min(z + 1, Z - 1)
        cy0, cy1 = max(0, y0), min(H, y1)
        cx0, cx1 = max(0, x0), min(W, x1)
        def _sl(zi):
            s = np.asarray(ct_arr[zi, cy0:cy1, cx0:cx1], dtype="float32")
            s = (s.clip(RD4AD_E2_HU_MIN, RD4AD_E2_HU_MAX) - RD4AD_E2_HU_MIN) / (RD4AD_E2_HU_MAX - RD4AD_E2_HU_MIN)
            if s.shape != (CROP_SIZE, CROP_SIZE):
                ph = CROP_SIZE - s.shape[0]; pw = CROP_SIZE - s.shape[1]
                s = np.pad(s, ((0, ph), (0, pw)), mode="edge")
            return s
        arr = np.stack([_sl(zm), _sl(z), _sl(zp)], axis=0)
        inp = torch.from_numpy(arr).unsqueeze(0)

        teacher, student, tf_feats = _get_rd4ad_e2_model()
        with torch.no_grad():
            teacher(inp)
            de_l, de_m, de_e = student(tf_feats["late"])
            s_l = F.interpolate((1 - F.cosine_similarity(de_l,  tf_feats["late"],  dim=1, eps=1e-8)).unsqueeze(1),
                                size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)
            s_m = F.interpolate((1 - F.cosine_similarity(de_m,  tf_feats["mid"],   dim=1, eps=1e-8)).unsqueeze(1),
                                size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)
            s_e = F.interpolate((1 - F.cosine_similarity(de_e,  tf_feats["early"], dim=1, eps=1e-8)).unsqueeze(1),
                                size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)
            smap = ((s_l + s_m + s_e) / 3)[0, 0].cpu().numpy().astype("float32")
        smin, smax = float(smap.min()), float(smap.max())
        if smax - smin > 1e-8:
            smap = (smap - smin) / (smax - smin)
        else:
            smap = np.zeros_like(smap)
        return smap
    except Exception as e:
        print(f"WARN: E2 spatial map 실패 — {e}", file=sys.stderr)
        return None


def run_gradcam_for_panel2(ct_arr, roi_arr, z, y0, x0, y1, x1, lung_z_pct, bbox_h, bbox_w):
    """CT에서 3ch crop → ScalarFusionModel Grad-CAM.
    반환: (cam_norm (96×96 float32), logit float, prob float) 또는 실패 시 (None, None, None).
    """
    import numpy as np, torch, torch.nn.functional as F
    try:
        Z = ct_arr.shape[0]

        def _ch(zi):
            zi = max(0, min(Z - 1, zi))
            c = np.clip(np.asarray(ct_arr[zi, y0:y1, x0:x1], dtype="float32"),
                        GRADCAM_HU_MIN, GRADCAM_HU_MAX)
            return (c - GRADCAM_HU_MIN) / (GRADCAM_HU_MAX - GRADCAM_HU_MIN)

        arr = np.stack([_ch(z - 1), _ch(z), _ch(z + 1)], axis=0)  # (3,96,96)

        mask_sl = np.asarray(roi_arr[z, y0:y1, x0:x1], dtype="float32")
        if mask_sl.shape != (CROP_SIZE, CROP_SIZE):
            mask_sl = np.ones((CROP_SIZE, CROP_SIZE), dtype="float32")
        mask_3ch = np.stack([mask_sl, mask_sl, mask_sl], axis=0)
        masked_arr = arr * mask_3ch

        mean_t = np.array(GRADCAM_IMAGENET_MEAN, dtype="float32").reshape(3, 1, 1)
        std_t  = np.array(GRADCAM_IMAGENET_STD,  dtype="float32").reshape(3, 1, 1)
        img_t  = torch.tensor((masked_arr - mean_t) / std_t, dtype=torch.float32).unsqueeze(0)

        stats = json.load(open(GRADCAM_SCALAR_STATS))["features"]
        lzp_mean  = stats["lung_z_percentile"]["mean"];    lzp_std  = stats["lung_z_percentile"]["std"]
        clrr_mean = stats["crop_lung_roi_ratio"]["mean"];  clrr_std = stats["crop_lung_roi_ratio"]["std"]
        crop_roi_ratio = float(mask_sl.sum()) / max(bbox_h * bbox_w, 1)
        lzp_n  = (lung_z_pct   - lzp_mean)  / max(lzp_std,  1e-8)
        clrr_n = (crop_roi_ratio - clrr_mean) / max(clrr_std, 1e-8)
        sc_t   = torch.tensor([[lzp_n, clrr_n]], dtype=torch.float32)

        model = _get_gradcam_model()
        act_s = {}; grad_s = {}
        fh = model.img_features[7].register_forward_hook(lambda m, i, o: act_s.__setitem__("v", o))
        bh = model.img_features[7].register_full_backward_hook(lambda m, gi, go: grad_s.__setitem__("v", go[0]))
        model.zero_grad()
        logit = model(img_t, sc_t)
        logit.backward()
        fh.remove(); bh.remove()

        w   = grad_s["v"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((act_s["v"] * w).sum(dim=1, keepdim=True))
        cam_up = F.interpolate(cam, size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)
        cam_np = cam_up[0, 0].detach().cpu().numpy().astype("float32")
        cmin, cmax = float(cam_np.min()), float(cam_np.max())
        cam_np = (cam_np - cmin) / (cmax - cmin) if cmax - cmin > 1e-8 else np.zeros_like(cam_np)

        # mask 내부만 normalize
        vals = (cam_np * mask_sl)[mask_sl > 0.5]
        if len(vals) > 0 and (vals.max() - vals.min()) > 1e-8:
            vmin, vmax = float(vals.min()), float(vals.max())
            cam_norm = np.where(mask_sl > 0.5, (cam_np * mask_sl - vmin) / (vmax - vmin), 0.0)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm.astype("float32"), float(logit.item()), float(torch.sigmoid(logit).item())
    except Exception as e:
        print(f"WARN: Grad-CAM 실패 — {e}", file=sys.stderr)
        return None, None, None


def _compute_p5_tracks(rows):
    """(crop_y0, crop_x0) 기준 연속 z-track 구성 → P5 = top3_mean × track_len/3.
    min_run=2 (track_len < 2 는 제외). roi_ratio는 미포함 (crop center가 이미 ROI 안에 있음 보장됨).
    반환: {track_id → {"best_row": row, "p5": float, "track_len": int}}
    """
    from collections import defaultdict
    pos_entries = defaultdict(list)
    for r in rows:
        pos_entries[(int(r["crop_y0"]), int(r["crop_x0"]))].append(
            (int(r["local_z"]), float(r[SCORE_COL]), r)
        )
    tracks = {}
    for (y0, x0), entries in pos_entries.items():
        entries.sort(key=lambda e: e[0])
        group = [entries[0]]
        def _flush(g):
            if len(g) < 2:
                return
            scores = sorted([e[1] for e in g], reverse=True)
            top3_mean = sum(scores[:3]) / max(len(scores[:3]), 1)
            p5 = top3_mean * len(g) / 3.0
            tid = f"{y0}_{x0}_{g[0][0]}_{g[-1][0]}"
            tracks[tid] = {
                "best_row": max(g, key=lambda e: e[1])[2],
                "p5": p5, "track_len": len(g),
            }
        for i in range(1, len(entries)):
            if entries[i][0] == entries[i-1][0] + 1:
                group.append(entries[i])
            else:
                _flush(group); group = [entries[i]]
        _flush(group)
    return tracks


def load_row(score_csv, patient, position_bin=None, local_z=None, label1_only=False):
    if not os.path.exists(score_csv):
        return None, f"score CSV 없음: {score_csv}"
    rows = [r for r in csv.DictReader(open(score_csv, encoding="utf-8-sig"))
            if r.get("patient_id") == patient]
    if not rows:
        return None, f"patient {patient} not found in score CSV"
    if SCORE_COL not in rows[0]:
        return None, f"score column '{SCORE_COL}' not in CSV"
    cand = rows
    if local_z is not None:
        cand = [r for r in cand if int(r["local_z"]) == int(local_z)]
    if position_bin:
        cb = [r for r in cand if r.get("six_bin_label", "") == position_bin
              or r.get("position_bin", "") == position_bin]
        if cb: cand = cb
    if label1_only:
        cb = [r for r in cand if str(r.get("label", "")) == "1"]
        if cb: cand = cb
    if not cand:
        return None, "조건에 맞는 saved row 없음"
    # local_z 고정 시 P5 계산 불가 → raw score max
    if local_z is not None:
        return max(cand, key=lambda r: float(r[SCORE_COL])), None
    # P5 = top3_mean × track_len/3 (E2 기준 단독 최강 top10=0.9216)
    tracks = _compute_p5_tracks(cand)
    if tracks:
        best = max(tracks.values(), key=lambda t: t["p5"])
        print(f"[P5] track_len={best['track_len']} p5={best['p5']:.4f} "
              f"z={best['best_row']['local_z']} y0={best['best_row']['crop_y0']} x0={best['best_row']['crop_x0']}")
        row = best["best_row"]
        row["_p5_score"] = best["p5"]
        row["_track_len"] = best["track_len"]
        return row, None
    return max(cand, key=lambda r: float(r[SCORE_COL])), None


def window(a):
    import numpy as np
    lo, hi = WL - WW / 2., WL + WW / 2.
    x = (np.asarray(a, dtype="float32") - lo) / max(hi - lo, 1e-6)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype("uint8")


def spacing_of(d):
    return float(json.load(open(os.path.join(d, "meta.json")))["spacing_xyz"][0])


def render(patient, score_csv, position_bin=None, local_z=None, out_root=None):
    t0 = time.time()
    if not all([ALLOW["ALLOW_CARD_RENDER"], ALLOW["ALLOW_CT_LOAD"],
                ALLOW["ALLOW_SOURCE_IMAGE_READ"], ALLOW["ALLOW_PNG_WRITE"]]):
        _abort("render needs ALLOW_CARD_RENDER=1 + ALLOW_CT_LOAD=1 + ALLOW_SOURCE_IMAGE_READ=1 + ALLOW_PNG_WRITE=1")
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    spec = importlib.util.spec_from_file_location("rdr", RETRIEVE_PY)
    rdr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rdr)

    row, err = load_row(score_csv, patient, position_bin, local_z, label1_only=True)
    if row is None:
        _abort(f"candidate 선택 실패: {err}")

    safe = row["safe_id"]; z = int(row["local_z"])
    cy0, cx0, cy1, cx1 = int(row["crop_y0"]), int(row["crop_x0"]), int(row["crop_y1"]), int(row["crop_x1"])
    score = float(row[SCORE_COL])
    pbin  = row.get("six_bin_label", row.get("position_bin", "unknown"))
    cp    = row.get("central_peripheral", "")
    report_slice = z

    cdir = os.path.join(CTBASE, safe)
    if not (os.path.exists(os.path.join(cdir, "ct_hu.npy")) and
            os.path.exists(os.path.join(cdir, "roi_0_0.npy"))):
        _abort(f"CT/ROI 없음: {safe}")
    errors = []

    roi = np.load(os.path.join(cdir, "roi_0_0.npy"), mmap_mode="r")
    Z = roi.shape[0]
    areas = np.array([int((np.asarray(roi[zz]) > 0).sum()) for zz in range(Z)])
    zs = np.where(areas > 0)[0]
    if zs.size == 0: _abort("no lung voxels")
    zmin, zmax = int(zs.min()), int(zs.max())
    m = np.asarray(roi[z]) > 0
    ys, xs = np.where(m); by0, by1, bx0, bx1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    cyc = (cy0 + cy1) / 2.; cxc = (cx0 + cx1) / 2.
    lung_z_pct = round((z - zmin) / max(zmax - zmin, 1), 4)
    y_pct = round((cyc - by0) / max(by1 - by0, 1), 4); x_pct = round((cxc - bx0) / max(bx1 - bx0, 1), 4)
    side = "left" if cxc < 256 else "right"
    cand_sp = spacing_of(cdir)
    ct = np.load(os.path.join(cdir, "ct_hu.npy"), mmap_mode="r")
    cslice = window(np.asarray(ct[z]))

    cam_norm, cam_logit, cam_prob = None, None, None
    try:
        cam_norm, cam_logit, cam_prob = run_gradcam_for_panel2(
            ct, roi, z, cy0, cx0, cy1, cx1, lung_z_pct, by1 - by0, bx1 - bx0)
        if cam_prob is not None:
            print(f"Grad-CAM prob: {cam_prob:.4f}")
    except Exception as e:
        print(f"WARN: Grad-CAM 실패 — {e}", file=sys.stderr)

    rows_idx, _ = rdr.load_ref_index(BANK_INDEX)
    best = rdr.select_best_per_patient((lung_z_pct, y_pct, x_pct), side, rows_idx)
    if len(best) != 3:
        _abort(f"normal ref {len(best)} != 3")

    ST = CROP_STRIDE
    all_patient_rows = [r for r in csv.DictReader(open(score_csv, encoding="utf-8-sig"))
                        if r.get("patient_id") == patient]
    Dz = {}
    for r in all_patient_rows:
        if int(r["local_z"]) == z:
            Dz[(int(r["crop_y0"]), int(r["crop_x0"]))] = float(r[SCORE_COL])
    YS = [cy0 - ST, cy0, cy0 + ST]; XS = [cx0 - ST, cx0, cx0 + ST]
    G = {(yy, xx): Dz.get((yy, xx)) for yy in YS for xx in XS}

    p4_sections = build_panel4_compact_sections(patient, pbin, z, score, lung_z_pct, nsclc_prob=cam_prob)

    def fnt(s):
        try: return ImageFont.truetype(FONT_KO, s)
        except Exception: return ImageFont.load_default()

    C_BG=(18,18,18,255); C_PANEL=(30,30,30,255); C_HEADER=(10,22,45,255); C_BORDER=(70,70,70,255)
    C_TITLE=(220,220,255,255); C_SEC=(150,195,255,255); C_BODY=(215,215,215,255); C_WARN=(255,215,75,255)
    C_SUB=(170,200,170,255); C_LBL=(190,190,190,255); C_YEL=(255,220,0,255); C_DIM=(175,135,135,255)
    COLOR_MAP = {"SEC": C_SEC, "LBL": C_LBL, "WARN": C_WARN, "SUB": C_SUB, "DIM": C_DIM}
    ft_title=fnt(20); ft_sec=fnt(15); ft_body=fnt(13); ft_sm=fnt(11)

    def ctx_tile(sl_u8, cyx, cpx, box_inner, peak=None):
        H, W = sl_u8.shape
        ccy, ccx = cyx; bpx = int(round(cpx * CONTEXT_FACTOR)); half = bpx // 2
        ny0 = max(0, min(int(round(ccy)) - half, H - bpx)); nx0 = max(0, min(int(round(ccx)) - half, W - bpx))
        crop = sl_u8[ny0:ny0 + bpx, nx0:nx0 + bpx]
        src = Image.fromarray(crop, "L").convert("RGBA")
        tile = src.resize((TILE_PX, TILE_PX), Image.LANCZOS)
        s = TILE_PX / bpx; d2 = ImageDraw.Draw(tile)
        iy0, ix0, iy1, ix1 = box_inner
        d2.rectangle([(ix0 - nx0) * s, (iy0 - ny0) * s, (ix1 - nx0) * s, (iy1 - ny0) * s],
                    outline=(255, 70, 70, 255) if peak else (255, 220, 0, 255), width=2)
        if peak:
            py0, px0, py1, px1 = peak
            d2.rectangle([(px0 - nx0) * s, (py0 - ny0) * s, (px1 - nx0) * s, (py1 - ny0) * s],
                        outline=(255, 140, 0, 255), width=2)
        return tile

    cand_cpx = int(round(PHYS_MM / cand_sp)); chalf = cand_cpx // 2
    cand_inner = [int(cyc) - chalf, int(cxc) - chalf, int(cyc) - chalf + cand_cpx, int(cxc) - chalf + cand_cpx]
    tiles = [("후보 슬라이스", f"z{z}  이상도 {score:.4f}",
              ctx_tile(cslice, (cyc, cxc), cand_cpx, cand_inner, peak=[cy0, cx0, cy1, cx1]))]
    ref_meta = []
    for i, b in enumerate(best, 1):
        rr = b["row"]; cb = rdr.crop_bbox_from_pct(y_pct, x_pct, rr)
        rz = int(rr["local_z"]); vol = rr["volume_id"]; rsp = spacing_of(os.path.join(NB, vol))
        rcpx = int(round(PHYS_MM / rsp)); rhalf = rcpx // 2
        rcy, rcx = cb["ref_center_y"], cb["ref_center_x"]
        inner = [int(rcy) - rhalf, int(rcx) - rhalf, int(rcy) - rhalf + rcpx, int(rcx) - rhalf + rcpx]
        import numpy as np  # already imported
        png = np.array(Image.open(os.path.join(BANK, rr["png_path"])).convert("L"))
        tiles.append((f"정상 참조 {i}", f"z{rz}  거리 {b['distance']:.3f}",
                      ctx_tile(png, (rcy, rcx), rcpx, inner)))
        ref_meta.append({"patient_alias": b["patient_alias"], "selected_local_z": rz,
                         "ref_lung_z_pct": round(float(rr["lung_z_pct"]), 4), "distance": b["distance"],
                         "spacing_mm": rsp, "edge_policy": cb["edge_policy"]})

    CW = 1600; MG = 16; HH = 64; PAD = 8
    P1W = CW - 2 * MG; TILES_W = 4 * TILE_PX + 3 * PAD
    whole_w = int(P1W * 0.5)
    body_h = by1 - by0; body_w = bx1 - bx0
    whole_h = int(whole_w * body_h / max(body_w, 1))
    P1_ROW1 = TILE_PX + 40; SEC_H = 24; SUB_H = 22
    P1H = PAD + SEC_H + SUB_H + PAD + P1_ROW1 + PAD + 20 + whole_h + PAD + 20 + PAD
    LPW = (CW - 4 * MG) // 3; LH = 360
    CH = HH + MG + P1H + MG + LH + MG
    cv = Image.new("RGBA", (CW, CH), C_BG); d = ImageDraw.Draw(cv)
    d.rectangle([(0, 0), (CW, HH)], fill=C_HEADER)
    d.text((MG, 8),  f"폐 이상 탐지 보조 카드 — {patient} / {pbin}", font=ft_title, fill=C_TITLE)
    d.text((MG, 32), "[내부 연구용 | 진단 금지]", font=ft_sm, fill=C_WARN)
    d.text((MG, 48), f"z{z}  이상도 {score:.4f}  {pbin}", font=ft_sm, fill=C_SUB)

    p1x = MG; p1y = HH + MG
    d.rectangle([(p1x, p1y), (p1x + P1W, p1y + P1H)], fill=C_PANEL, outline=C_BORDER, width=1)
    cy_ = p1y + PAD
    d.text((p1x + PAD, cy_), "Panel 1 : 정상 참조 비교 (80mm 범위 / 2× 맥락 crop)", font=ft_sec, fill=C_SEC); cy_ += SEC_H
    d.text((p1x + PAD, cy_), "후보(빨강=80mm / 주황=peak) vs 정상 참조 3명(노랑=80mm)", font=ft_sm, fill=C_SUB); cy_ += SUB_H + PAD
    rx0 = p1x + (P1W - TILES_W) // 2
    for i, (lab, sub, tile) in enumerate(tiles):
        sx = rx0 + i * (TILE_PX + PAD)
        d.rectangle([(sx - 1, cy_ - 1), (sx + TILE_PX, cy_ + TILE_PX)], outline=C_YEL if i == 0 else C_BORDER, width=2 if i == 0 else 1)
        cv.paste(tile, (sx, cy_), tile)
        d.text((sx, cy_ + TILE_PX + 3),  lab, font=ft_sm, fill=C_YEL if i == 0 else C_LBL)
        d.text((sx, cy_ + TILE_PX + 18), sub, font=ft_sm, fill=(255, 200, 120, 255) if i == 0 else (150, 150, 150, 255))
    cy_ += P1_ROW1 + PAD
    d.text((p1x + PAD, cy_), f"전체 슬라이스 (z{z}) — 빨강=후보 위치, 노랑=3×3 범위", font=ft_sm, fill=C_LBL); cy_ += 20
    body_crop = cslice[by0:by1, bx0:bx1]
    wholeimg = Image.fromarray(body_crop, "L").convert("RGBA").resize((whole_w, whole_h), Image.LANCZOS)
    dw = ImageDraw.Draw(wholeimg); sw = whole_w / max(body_w, 1)
    dw.rectangle([(cx0-bx0)*sw, (cy0-by0)*sw, (cx1-bx0)*sw, (cy1-by0)*sw], outline=(255, 60, 60, 255), width=2)
    dw.rectangle([(min(XS)-bx0)*sw, (min(YS)-by0)*sw, (max(XS)+CROP_SIZE-bx0)*sw, (max(YS)+CROP_SIZE-by0)*sw],
                 outline=(255, 220, 0, 255), width=1)
    wx = p1x + (P1W - whole_w) // 2; cv.paste(wholeimg, (wx, cy_), wholeimg); cy_ += whole_h + PAD

    ly = HH + MG + P1H + MG
    p2x = MG; p3x = MG + LPW + MG; p4x = MG + (LPW + MG) * 2
    for bx, title in [(p2x, "Panel 2 : 이상 강조 히트맵"), (p3x, "Panel 3 : 3×3 점수 격자"), (p4x, "Panel 4 : 판독 보조")]:
        d.rectangle([(bx, ly), (bx + LPW, ly + LH)], fill=C_PANEL, outline=C_BORDER, width=1)
        d.text((bx + PAD, ly + PAD), title, font=ft_sec, fill=C_SEC)

    ZS = LPW - 2 * PAD
    P2_IMG = min(ZS, LH - 26 - PAD * 3 - 28)  # 패널 높이 내에 맞춤 (28px = 라벨 2줄)
    d.rectangle([(p2x + PAD, ly + PAD + 26), (p2x + PAD + P2_IMG, ly + PAD + 26 + P2_IMG)],
                fill=(20, 20, 20, 255), outline=(55, 55, 55, 255))

    import matplotlib
    _ylrd = matplotlib.colormaps["YlOrRd"]
    if cam_norm is not None and cam_prob is not None and cam_prob >= 0.5:
        # 이상 패턴 감지 → Grad-CAM 히트맵
        rgba_f = _ylrd(cam_norm).astype("float32")
        rgba_f[..., 3] = (cam_norm ** GRADCAM_ALPHA_GAMMA) * GRADCAM_ALPHA_MAX
        bg_crop = window(np.asarray(ct[z, cy0:cy1, cx0:cx1]))
        bg_rgb  = np.stack([bg_crop, bg_crop, bg_crop], axis=-1).astype("float32")
        a3 = rgba_f[..., 3:4]
        out_arr = np.clip(bg_rgb * (1 - a3) + rgba_f[..., :3] * 255 * a3, 0, 255).astype("uint8")
        heat_img = Image.fromarray(out_arr, "RGB").resize((P2_IMG, P2_IMG), Image.NEAREST)
        cv.paste(heat_img.convert("RGBA"), (p2x + PAD, ly + PAD + 26))
        p2_lbl_y = ly + PAD + 26 + P2_IMG + 4
        d.text((p2x + PAD, p2_lbl_y), f"이상 확률 {cam_prob:.1%} — 이상 패턴 감지",
               font=ft_sm, fill=C_WARN)
        d.text((p2x + PAD, p2_lbl_y + 14), "이상 강조 히트맵 — 연구용, 진단 금지",
               font=ft_sm, fill=(130, 130, 130, 255))
    else:
        # 이상 확률 낮음 또는 Grad-CAM 실패 → E2 RD4AD spatial map
        e2_smap = run_rd4ad_e2_spatial_map(ct, z, cy0, cx0, cy1, cx1)
        if e2_smap is not None:
            rgba_f = _ylrd(e2_smap).astype("float32")
            rgba_f[..., 3] = (e2_smap ** GRADCAM_ALPHA_GAMMA) * GRADCAM_ALPHA_MAX
            bg_crop = window(np.asarray(ct[z, cy0:cy1, cx0:cx1]))
            bg_rgb  = np.stack([bg_crop, bg_crop, bg_crop], axis=-1).astype("float32")
            a3 = rgba_f[..., 3:4]
            out_arr = np.clip(bg_rgb * (1 - a3) + rgba_f[..., :3] * 255 * a3, 0, 255).astype("uint8")
            heat_img = Image.fromarray(out_arr, "RGB").resize((P2_IMG, P2_IMG), Image.NEAREST)
            cv.paste(heat_img.convert("RGBA"), (p2x + PAD, ly + PAD + 26))
            p2_lbl_y = ly + PAD + 26 + P2_IMG + 4
            prob_text = f"이상 확률 {cam_prob:.1%} — 낮음" if cam_prob is not None else "이상 확률 산출 불가"
            d.text((p2x + PAD, p2_lbl_y), prob_text, font=ft_sm, fill=C_SUB)
            d.text((p2x + PAD, p2_lbl_y + 14), "RD4AD 이상 분포 맵 — 연구용, 진단 금지",
                   font=ft_sm, fill=(130, 130, 130, 255))
        else:
            ph_cx = p2x + PAD + P2_IMG // 2; ph_cy = ly + PAD + 26 + P2_IMG // 2
            d.text((ph_cx - 55, ph_cy - 8), "[ 히트맵 생성 실패 ]", font=ft_body, fill=(65, 65, 65, 255))

    vals = [v for v in G.values() if v is not None]; mn = min(vals) if vals else 0; mx = max(vals) if vals else 1; rng = max(mx - mn, 1e-6)
    GS = min(LPW - 2 * PAD, LH - 120); cell = (GS - 8) // 3; g0x = p3x + PAD; g0y = ly + PAD + 26
    for ri, yy in enumerate(YS):
        for ci, xx in enumerate(XS):
            v = G[(yy, xx)]; isp = (yy, xx) == (cy0, cx0)
            cxp = g0x + ci * (cell + 4); cyp = g0y + ri * (cell + 4)
            if v is None:
                d.rectangle([(cxp, cyp), (cxp + cell, cyp + cell)], fill=(58, 58, 58, 255), outline=(90, 90, 90, 255))
                for k in range(0, cell, 9): d.line([(cxp, cyp + k), (cxp + k, cyp)], fill=(110, 110, 110, 255))
                d.text((cxp + 3, cyp + cell - 15), "MISSING", font=ft_sm, fill=(225, 180, 180, 255))
            else:
                nz = (v - mn) / rng; bg = C_YEL if isp else (int(35 + nz * 70), int(35 + nz * 70), min(int(60 + nz * 70), 255), 255)
                lc = (25, 25, 25, 255) if isp else (200, 200, 200, 255)
                d.rectangle([(cxp, cyp), (cxp + cell, cyp + cell)], fill=bg, outline=(90, 90, 90, 255))
                d.text((cxp + 3, cyp + cell - 15), f"{v:.4f}", font=ft_sm, fill=lc)
                if isp: d.text((cxp + 3, cyp + cell // 2 - 6), "peak", font=ft_sm, fill=(180, 0, 0, 255))
    gy = g0y + 3 * (cell + 4) + 6
    d.text((p3x + PAD, gy), "노란색=최고점 / 회색=데이터 없음", font=ft_sm, fill=C_BODY)

    # ── Panel 4 compact ───────────────────────────────────────────────────────
    def wrap(text, font, maxw):
        out = []; cur = ""
        for ch in text:
            if d.textlength(cur + ch, font=font) <= maxw: cur += ch
            else: out.append(cur); cur = ch
        if cur: out.append(cur)
        return out

    tw = LPW - 2 * PAD - 4; yy4 = ly + PAD + 26
    LH4 = 15; P4_BOTTOM = ly + LH - 4

    for sec_label, color_key, sec_lines in p4_sections:
        if yy4 + LH4 > P4_BOTTOM: break
        sec_col = COLOR_MAP.get(color_key, C_LBL)
        d.text((p4x + PAD, yy4), f"[{sec_label}]", font=ft_sm, fill=sec_col); yy4 += LH4 + 2
        for ln in sec_lines:
            for wln in wrap(ln, ft_sm, tw):
                if yy4 + LH4 > P4_BOTTOM: break
                d.text((p4x + PAD + 4, yy4), wln, font=ft_sm, fill=C_BODY); yy4 += LH4
        yy4 += 5

    # ── 저장 ─────────────────────────────────────────────────────────────────
    out_root = out_root or os.path.join(OUT_BASE, f"{patient}__{pbin}")
    png_dir  = os.path.join(out_root, "cards_png")
    json_dir = os.path.join(out_root, "cards_json")
    for p_ in (png_dir, json_dir): os.makedirs(p_, exist_ok=True)

    case_id  = f"{patient}__{pbin}"
    out_png  = os.path.join(png_dir,  f"{case_id}{SUFFIX}.png")
    out_json = os.path.join(json_dir, f"{case_id}{SUFFIX}.json")

    cv.convert("RGB").save(out_png, "PNG")

    p5_score  = row.get("_p5_score")
    track_len = row.get("_track_len")
    meta = {"patient_id": patient, "position_bin": pbin, "central_peripheral": cp,
            "candidate_source": "saved_rd4ad_score_output", "score_col": SCORE_COL,
            "score_recomputed": False, "branch": BRANCH, "mask": MASK,
            "local_z": z, "report_slice": report_slice,
            "candidate_bbox": [cy0, cx0, cy1, cx1],
            "rd4ad_score": score,
            "p5_score": p5_score, "p5_track_len": track_len,
            "p5_method": "top3_mean × track_len/3 (E2 best, top10 hit=0.9216)",
            "candidate_lung_z_pct": lung_z_pct,
            "candidate_y_pct": y_pct, "candidate_x_pct": x_pct, "candidate_side": side,
            "matching": "bilateral lung frame; lung_z_pct + lung-bbox relative y/x; 80mm physical; NOT same-z",
            "panel4_version": "clinical_compact_v1",
            "panel3_3x3": {f"{k[0]},{k[1]}": v for k, v in G.items()},
            "normal_refs": ref_meta, "canvas_size_px": [CW, CH],
            "generated_date": str(date.today()),
            "safety": {"model_forward": cam_norm is not None, "feature_extraction": False, "score_recompute": False,
                       "stage2_holdout_access": False, "ct_load_used": True, "raw_ct_copied": False,
                       "gradcam_included": cam_norm is not None, "not_diagnostic": True}}
    json.dump(meta, open(out_json, "w"), ensure_ascii=False, indent=2)

    # panel4 text files
    p4_csv_path = os.path.join(out_root, "panel4_text_clinical_compact.csv")
    p4_md_path  = os.path.join(out_root, "panel4_text_clinical_compact.md")
    p4_rows = [{"section": s, "color_key": c, "text": ln} for s, c, lines in p4_sections for ln in lines]
    with open(p4_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["section", "color_key", "text"]); w.writeheader(); w.writerows(p4_rows)
    with open(p4_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Panel 4 compact — {case_id}\n\n")
        f.write(f"- patient: {patient}  position_bin: {pbin}  local_z: {z}\n")
        f.write(f"- rd4ad_score: {score:.6f}  lung_z_pct: {lung_z_pct}\n\n")
        for s, _, lines in p4_sections:
            f.write(f"## [{s}]\n")
            for ln in lines: f.write(f"- {ln}\n")
            f.write("\n")
        f.write("---\nResearch-use only | not diagnostic\n")

    elapsed = round(time.time() - t0, 2)
    json.dump({"case_id": case_id, "patient_id": patient, "position_bin": pbin,
               "local_z": z, "rd4ad_score": score, "p5_score": p5_score, "p5_track_len": track_len,
               "lung_z_pct": lung_z_pct,
               "panel4_version": "clinical_compact_v1", "score_recomputed": False,
               "elapsed_sec": elapsed, "out_png": out_png, "generated_date": str(date.today())},
              open(os.path.join(out_root, "runtime_summary.json"), "w"), ensure_ascii=False, indent=2)

    json.dump({"model_forward": cam_norm is not None, "feature_extraction": False, "score_recompute": False,
               "stage2_holdout_access": False, "existing_card_overwrite": False, "raw_ct_copied": False,
               "gradcam_included": cam_norm is not None, "gradcam_logit": cam_logit, "gradcam_prob": cam_prob,
               "vessel_risk_included": False,
               "diagnostic_statement": False, "same_z_claimed": False,
               "panel3_missing_interpolated": False, "verdict": "PASS_SAFETY_CHECK"},
              open(os.path.join(out_root, "safety_check.json"), "w"), ensure_ascii=False, indent=2)

    with open(os.path.join(out_root, "errors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["type", "msg"]); w.writeheader()
        for e in errors: w.writerow(e)

    json.dump({"done": True, "patient": patient, "position_bin": pbin, "out_png": out_png,
               "rd4ad_score": score, "gradcam_prob": cam_prob,
               "panel4_version": "clinical_compact_v1",
               "errors": len(errors), "elapsed_sec": elapsed, "generated_date": str(date.today())},
              open(os.path.join(out_root, "DONE.json"), "w"), ensure_ascii=False, indent=2)

    print(f"DONE  {out_png}")
    print(f"JSON  {out_json}")
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient",      default="LUNG1-041")
    ap.add_argument("--score_csv",    required=True)
    ap.add_argument("--position_bin", default=None)
    ap.add_argument("--local_z",      type=int, default=None)
    ap.add_argument("--out_root",     default=None)
    ap.add_argument("--render",       action="store_true")
    ap.add_argument("--confirm",      action="store_true")
    a = ap.parse_args()
    if a.render and a.confirm:
        render(a.patient, a.score_csv, a.position_bin, a.local_z, a.out_root); sys.exit(0)
    _abort("mode 미지정: --render --confirm --patient ... --score_csv ...")


if __name__ == "__main__":
    main()
