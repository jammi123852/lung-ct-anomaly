"""
rd_e1_abc_stage2_scoring.py

목적:
  A/B/C/C2/A2/E1/E2 실험 각각에 대해 stage2_holdout 128,827개 후보 RD4AD scoring.
  D1s stage2 scoring script 기반으로 --exp-id만 바꿔 동일 조건으로 평가.

실험별 차이:
  A   : lung3ch, [-1000,600], ResNet18
  B   : medi_mip3ch, [-160,240], ResNet18
  C   : lung_mip3ch, [-1000,600], ResNet18
  C2  : lung_mip3ch+ROI픽셀마스크, [-1000,600], ResNet18
  A2  : lung3ch+per-ch ROI픽셀마스크, [-1000,600], ResNet18
  E1  : lung_mip3ch, [-1000,600], EfficientNet-B0
  E2  : lung3ch, [-1000,600], EfficientNet-B0

실행:
  dry-run :  python rd_e1_abc_stage2_scoring.py --exp-id A --dry-run
  smoke   :  python rd_e1_abc_stage2_scoring.py --exp-id A --run-shard --shard-id 0 --smoke-test
             --confirm-model-forward --confirm-stage2-holdout-eval-only
  full    :  python rd_e1_abc_stage2_scoring.py --exp-id A --run-shard --shard-id {0..7}
             --confirm-model-forward --confirm-stage2-holdout-eval-only

출력 위치: experiments/rd_e1_abc_stage2_eval_v1/{EXP_ID}/shards/shard_{N}/
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

# =============================================================================
# 실험별 설정
# =============================================================================

PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

EXP_CONFIGS = {
    "A":  {
        "hu_min": -1000.0, "hu_max": 600.0,
        "input_type": "lung3ch", "teacher_type": "resnet18",
        "ckpt_rel": "outputs/models/rd_e1a_true_rd4ad_resnet18_lung3ch_shard_v1/checkpoints/best_train_loss.pth",
    },
    "B":  {
        "hu_min": -160.0, "hu_max": 240.0,
        "input_type": "medi_mip3ch", "teacher_type": "resnet18",
        "ckpt_rel": "outputs/models/rd_e1b_true_rd4ad_resnet18_medi_mip3ch_shard_v1/checkpoints/best_train_loss.pth",
    },
    "C":  {
        "hu_min": -1000.0, "hu_max": 600.0,
        "input_type": "lung_mip3ch", "teacher_type": "resnet18",
        "ckpt_rel": "outputs/models/rd_e1c_true_rd4ad_resnet18_lung_mip3ch_shard_v1/checkpoints/best_train_loss.pth",
    },
    "C2": {
        "hu_min": -1000.0, "hu_max": 600.0,
        "input_type": "lung_mip3ch_roipx", "teacher_type": "resnet18",
        "ckpt_rel": "outputs/models/rd_e1c2_true_rd4ad_resnet18_lung_mip3ch_roipx_shard_v1/checkpoints/best_train_loss.pth",
    },
    "A2": {
        "hu_min": -1000.0, "hu_max": 600.0,
        "input_type": "lung3ch_roipx", "teacher_type": "resnet18",
        "ckpt_rel": "outputs/models/rd_e1a2_true_rd4ad_resnet18_lung3ch_roipx_shard_v1/checkpoints/best_train_loss.pth",
    },
    "E1": {
        "hu_min": -1000.0, "hu_max": 600.0,
        "input_type": "lung_mip3ch", "teacher_type": "effb0",
        "ckpt_rel": "outputs/models/rd_e1e1_effb0_lung_mip3ch_shard_v1/checkpoints/best_train_loss.pth",
    },
    "E2": {
        "hu_min": -1000.0, "hu_max": 600.0,
        "input_type": "lung3ch", "teacher_type": "effb0",
        "ckpt_rel": "outputs/models/rd_e1e2_effb0_lung3ch_shard_v1/checkpoints/best_train_loss.pth",
    },
    "E2z": {
        "hu_min": -1000.0, "hu_max": 600.0,
        "input_type": "lung3ch", "teacher_type": "effb0z",
        "ckpt_rel": "outputs/models/rd_e1e2z_effb0_lung3ch_zpct_v1/checkpoints/best_train_loss.pth",
    },
    "D1s": {
        "hu_min": -160.0, "hu_max": 240.0,
        "input_type": "lung3ch", "teacher_type": "resnet18",
        "ckpt_rel": "outputs/models/rd_d1s_true_rd4ad_resnet18_medi3ch_shard_v1/checkpoints/best_train_loss.pth",
    },
}

# =============================================================================
# 공통 경로 상수
# =============================================================================

EXPERIMENT_BASE = PROJECT_ROOT / "experiments/rd_e1_abc_stage2_eval_v1"
D1S_MANIFEST_ROOT = (
    PROJECT_ROOT
    / "experiments/stage2_strict_ztrack_rd4ad_scoring_preflight_v1"
)
CANDIDATE_MANIFEST_CSV = (
    D1S_MANIFEST_ROOT / "manifests/stage2_rd4ad_scoring_manifest_minrun2.csv"
)
SHARD_PLAN_CSV = (
    D1S_MANIFEST_ROOT / "manifests/stage2_rd4ad_scoring_shard_plan.csv"
)
CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)
ROI_MASK_ROOT = (
    PROJECT_ROOT
    / "outputs/mip-postprocess-research-v1/masks"
    / "refined_roi_v4_20_modeB_all_v1"
)
LOCAL_RESNET_WEIGHT = Path(
    "/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)
LOCAL_EFFB0_WEIGHT = Path(
    "/home/jinhy/.cache/torch/hub/checkpoints"
    "/efficientnet_b0_rwightman-7f5810bc.pth"
)

SHARD_COUNT               = 8
CROP_SIZE                 = 96
EXPECTED_TOTAL_CANDIDATES = 128_827
SMOKE_N                   = 50

# =============================================================================
# Early parse: --exp-id로 실험별 상수 설정
# =============================================================================

_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument("--exp-id", choices=list(EXP_CONFIGS.keys()))
_early_args, _ = _early_parser.parse_known_args()

EXP_ID = _early_args.exp_id
if EXP_ID is not None:
    _cfg        = EXP_CONFIGS[EXP_ID]
    HU_MIN       = _cfg["hu_min"]
    HU_MAX       = _cfg["hu_max"]
    INPUT_TYPE   = _cfg["input_type"]
    TEACHER_TYPE = _cfg["teacher_type"]
    CKPT_PATH    = PROJECT_ROOT / _cfg["ckpt_rel"]
    SHARDS_DIR   = EXPERIMENT_BASE / EXP_ID / "shards"
else:
    HU_MIN = HU_MAX = INPUT_TYPE = TEACHER_TYPE = CKPT_PATH = SHARDS_DIR = None

# =============================================================================
# guardrail
# =============================================================================

GUARDRAILS = {
    "exp_id":                                None,
    "stage2_holdout_used_for_method_tuning": False,
    "stage2_holdout_eval_only":              True,
    "checkpoint_loaded":                     False,
    "model_forward_executed":                False,
    "training_executed":                     False,
    "backward_executed":                     False,
    "optimizer_created":                     False,
    "checkpoint_saved":                      False,
    "crop_generation_executed":              False,
    "scoring_executed":                      False,
    "existing_artifact_modified":            False,
    "output_overwrite":                      False,
    "label_used_for_evaluation_only":        True,
    "label_used_as_selector":                False,
    "roi_hard_filter_applied":               False,
    "vessel_mask_applied":                   False,
    "all_survived_track_candidates_scored":  False,
    "primary_candidate_score":               "rd4ad_ztrack_score_raw",
    "primary_track_score":                   "raw_track_top3_mean",
}

# =============================================================================
# 안전 경로 검사
# =============================================================================

_PROTECTED_INPUTS = [
    CANDIDATE_MANIFEST_CSV,
    SHARD_PLAN_CSV,
]
if CKPT_PATH:
    _PROTECTED_INPUTS.append(CKPT_PATH)


def ensure_output_path_safe(p: Path) -> None:
    rp = Path(p).resolve()
    for pi in _PROTECTED_INPUTS:
        try:
            if rp == pi.resolve():
                GUARDRAILS["existing_artifact_modified"] = True
                raise RuntimeError(f"[ABORT] 입력 파일 덮어쓰기 차단: {p}")
        except RuntimeError:
            raise
        except Exception:
            pass
    if SHARDS_DIR is None:
        return
    shards_root = str(SHARDS_DIR.resolve().parent)  # EXP_ID/ 하위
    exp_root = str((EXPERIMENT_BASE / (EXP_ID or "UNKNOWN")).resolve())
    if not str(rp).startswith(exp_root):
        GUARDRAILS["existing_artifact_modified"] = True
        raise RuntimeError(f"[ABORT] 실험 폴더 외부 쓰기 차단: {p}")


# =============================================================================
# CSV 유틸
# =============================================================================

def read_csv(path: Path) -> list:
    rows = []
    with open(str(path), encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def write_csv(path: Path, fieldnames: list, rows: list) -> None:
    ensure_output_path_safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  saved: {path} ({len(rows)} rows)")


def make_error_logger(error_csv: Path):
    def _append(msg: str, exc: Exception = None) -> None:
        ensure_output_path_safe(error_csv)
        error_csv.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if error_csv.exists() else "w"
        with open(str(error_csv), mode, encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if mode == "w":
                w.writerow(["timestamp", "message", "traceback"])
            tb = traceback.format_exc() if exc else ""
            w.writerow([
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                msg,
                tb.replace("\n", " | "),
            ])
    return _append


# =============================================================================
# shard 할당 (D1s와 동일)
# =============================================================================

def _patient_shard(patient_id: str, n_shards: int = SHARD_COUNT) -> int:
    return int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % n_shards


# =============================================================================
# ROI mask cache
# =============================================================================

class RoiMaskCache:
    def __init__(self, max_size: int = 12):
        self._cache: OrderedDict = OrderedDict()
        self._max = max_size

    def get(self, safe_id: str):
        import numpy as np
        if safe_id in self._cache:
            self._cache.move_to_end(safe_id)
            return self._cache[safe_id]
        mask_arr = None
        for subset in ("lesion", "normal"):
            p = ROI_MASK_ROOT / subset / safe_id / "refined_roi.npy"
            if p.exists():
                mask_arr = np.load(str(p), mmap_mode="r")
                break
        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        self._cache[safe_id] = mask_arr
        return mask_arr


# =============================================================================
# CT mmap 캐시
# =============================================================================

class CTMmapCache:
    def __init__(self, max_size: int = 12):
        self._cache: OrderedDict = OrderedDict()
        self._max = max_size

    def get(self, safe_id: str):
        import numpy as np
        if safe_id in self._cache:
            self._cache.move_to_end(safe_id)
            return self._cache[safe_id]
        ct_path = CT_ROOT / safe_id / "ct_hu.npy"
        if not ct_path.exists():
            raise FileNotFoundError(f"CT 없음: {ct_path}")
        arr = np.load(str(ct_path), mmap_mode="r")
        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        self._cache[safe_id] = arr
        return arr


# =============================================================================
# 모델 빌드 (ResNet18)
# =============================================================================

def build_teacher_resnet18():
    import torch
    import torchvision.models as models
    resnet = models.resnet18(weights=None)
    state_dict = torch.load(
        str(LOCAL_RESNET_WEIGHT), map_location="cpu", weights_only=True
    )
    resnet.load_state_dict(state_dict)
    resnet.eval()
    resnet.requires_grad_(False)
    return resnet


def build_student_decoder_resnet18():
    import torch.nn as nn

    class StudentDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.de_layer3 = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.de_layer2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.de_layer1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        def forward(self, layer3_feat):
            x   = self.de_layer3(layer3_feat)
            de3 = x
            x   = self.de_layer2(x)
            de2 = x
            x   = self.de_layer1(x)
            de1 = x
            return de3, de2, de1

    return StudentDecoder()


# =============================================================================
# 모델 빌드 (EfficientNet-B0)
# =============================================================================

def build_student_decoder_effb0z():
    """E2z용 decoder: de_late 입력 81ch (teacher 80ch + z_pct 1ch)"""
    import torch.nn as nn

    class StudentDecoderZ(nn.Module):
        def __init__(self):
            super().__init__()
            self.de_late = nn.Sequential(
                nn.Conv2d(81, 80, 3, 1, 1),
                nn.BatchNorm2d(80),
                nn.ReLU(inplace=True),
            )
            self.de_mid = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(80, 40, 3, 1, 1),
                nn.BatchNorm2d(40),
                nn.ReLU(inplace=True),
            )
            self.de_early = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(40, 24, 3, 1, 1),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
            )

        def forward(self, late_feat_with_z):
            x    = self.de_late(late_feat_with_z)
            de_l = x
            x    = self.de_mid(x)
            de_m = x
            x    = self.de_early(x)
            de_e = x
            return de_l, de_m, de_e

    return StudentDecoderZ()


def build_teacher_effb0():
    import torch
    import torchvision.models as models
    effnet = models.efficientnet_b0(weights=None)
    state_dict = torch.load(
        str(LOCAL_EFFB0_WEIGHT), map_location="cpu", weights_only=True
    )
    effnet.load_state_dict(state_dict)
    effnet.eval()
    effnet.requires_grad_(False)
    return effnet


def build_student_decoder_effb0():
    import torch.nn as nn

    class StudentDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.de_late = nn.Sequential(
                nn.Conv2d(80, 80, 3, 1, 1),
                nn.BatchNorm2d(80),
                nn.ReLU(inplace=True),
            )
            self.de_mid = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(80, 40, 3, 1, 1),
                nn.BatchNorm2d(40),
                nn.ReLU(inplace=True),
            )
            self.de_early = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(40, 24, 3, 1, 1),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
            )

        def forward(self, late_feat):
            x     = self.de_late(late_feat)
            de_l  = x
            x     = self.de_mid(x)
            de_m  = x
            x     = self.de_early(x)
            de_e  = x
            return de_l, de_m, de_e

    return StudentDecoder()


# =============================================================================
# checkpoint 로드 (teacher_type 분기)
# =============================================================================

def load_model_from_checkpoint(device):
    import torch
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"checkpoint 없음: {CKPT_PATH}")

    if TEACHER_TYPE == "resnet18":
        teacher = build_teacher_resnet18().to(device)
        student = build_student_decoder_resnet18().to(device)
    elif TEACHER_TYPE == "effb0z":
        teacher = build_teacher_effb0().to(device)
        student = build_student_decoder_effb0z().to(device)
    else:  # effb0
        teacher = build_teacher_effb0().to(device)
        student = build_student_decoder_effb0().to(device)

    teacher.eval()
    student.eval()
    teacher.requires_grad_(False)
    for p in student.parameters():
        p.requires_grad_(False)

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    if "student_state_dict" in ckpt:
        student.load_state_dict(ckpt["student_state_dict"])
    elif "model_state_dict" in ckpt:
        student.load_state_dict(ckpt["model_state_dict"])
    else:
        student.load_state_dict(ckpt)

    GUARDRAILS["checkpoint_loaded"] = True
    return teacher, student


def setup_teacher_hooks(teacher, teacher_features: dict):
    """teacher_type에 따라 forward hook 등록. teacher_features dict를 채운다."""
    if TEACHER_TYPE == "resnet18":
        for layer_name, module in [
            ("layer1", teacher.layer1),
            ("layer2", teacher.layer2),
            ("layer3", teacher.layer3),
        ]:
            def _hook(mod, inp, output, _n=layer_name):
                teacher_features[_n] = output
            module.register_forward_hook(_hook)
    else:  # effb0, effb0z
        for feat_name, module in [
            ("early", teacher.features[2]),
            ("mid",   teacher.features[3]),
            ("late",  teacher.features[4]),
        ]:
            def _hook(mod, inp, output, _n=feat_name):
                teacher_features[_n] = output
            module.register_forward_hook(_hook)


def _forbidden_train(*args, **kwargs):
    GUARDRAILS["training_executed"] = True
    raise RuntimeError("[ABORT] training 호출 금지됨")


# =============================================================================
# crop 생성
# =============================================================================

def _clip_pad_slice(ct_arr, z_idx: int, cy0: int, cy1: int, cx0: int, cx1: int,
                    pad_top: int, pad_bottom: int, pad_left: int, pad_right: int,
                    needs_pad: bool, can_reflect: bool):
    import numpy as np
    Z, H, W = ct_arr.shape
    zi      = int(max(0, min(z_idx, Z - 1)))
    normed  = (ct_arr[zi, cy0:cy1, cx0:cx1].astype(np.float32).clip(HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
    if needs_pad:
        pad_mode = "reflect" if can_reflect else "edge"
        normed   = np.pad(normed, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=pad_mode)
    return normed


def build_lung3ch_crop(ct_arr, local_z: int, y0: int, x0: int, y1: int, x1: int):
    import numpy as np
    Z, H, W    = ct_arr.shape
    z, zm, zp  = int(local_z), max(int(local_z) - 1, 0), min(int(local_z) + 1, Z - 1)
    y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)
    pad_top    = max(0, -y0);  pad_bottom = max(0, y1 - H)
    pad_left   = max(0, -x0); pad_right  = max(0, x1 - W)
    needs_pad  = pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0
    cy0, cy1   = max(0, y0), min(H, y1)
    cx0, cx1   = max(0, x0), min(W, x1)
    can_reflect = (cy1 - cy0 > 1) and (cx1 - cx0 > 1)
    ch0 = _clip_pad_slice(ct_arr, zm, cy0, cy1, cx0, cx1, pad_top, pad_bottom, pad_left, pad_right, needs_pad, can_reflect)
    ch1 = _clip_pad_slice(ct_arr, z,  cy0, cy1, cx0, cx1, pad_top, pad_bottom, pad_left, pad_right, needs_pad, can_reflect)
    ch2 = _clip_pad_slice(ct_arr, zp, cy0, cy1, cx0, cx1, pad_top, pad_bottom, pad_left, pad_right, needs_pad, can_reflect)
    crop = np.stack([ch0, ch1, ch2], axis=0)
    if crop.shape != (3, CROP_SIZE, CROP_SIZE):
        raise ValueError(f"crop shape {crop.shape} != (3,{CROP_SIZE},{CROP_SIZE})")
    if not np.isfinite(crop).all():
        raise ValueError("crop contains NaN/Inf")
    return crop.astype(np.float32)


def build_mip3ch_crop(ct_arr, local_z: int, y0: int, x0: int, y1: int, x1: int):
    import numpy as np
    Z, H, W = ct_arr.shape
    z       = int(local_z)
    y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)
    pad_top    = max(0, -y0);  pad_bottom = max(0, y1 - H)
    pad_left   = max(0, -x0); pad_right  = max(0, x1 - W)
    needs_pad  = pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0
    cy0, cy1   = max(0, y0), min(H, y1)
    cx0, cx1   = max(0, x0), min(W, x1)
    can_reflect = (cy1 - cy0 > 1) and (cx1 - cx0 > 1)

    def _sl(zi):
        return _clip_pad_slice(ct_arr, zi, cy0, cy1, cx0, cx1,
                               pad_top, pad_bottom, pad_left, pad_right,
                               needs_pad, can_reflect)

    def _mip(z_list):
        return np.stack([_sl(zi) for zi in z_list], axis=0).max(axis=0)

    ch0  = _mip([z - 3, z - 2, z - 1])
    ch1  = _mip([z - 1, z,     z + 1])
    ch2  = _mip([z + 1, z + 2, z + 3])
    crop = np.stack([ch0, ch1, ch2], axis=0)
    if crop.shape != (3, CROP_SIZE, CROP_SIZE):
        raise ValueError(f"crop shape {crop.shape} != (3,{CROP_SIZE},{CROP_SIZE})")
    if not np.isfinite(crop).all():
        raise ValueError("crop contains NaN/Inf")
    return crop.astype(np.float32)


def apply_roi_mask_single_z(crop, mask_arr, z: int, y0: int, x0: int, y1: int, x1: int):
    """모든 채널에 center slice(z) 마스크 적용 (C2 방식)."""
    if mask_arr is None:
        return crop
    import numpy as np
    Z, H, W   = mask_arr.shape
    z_cl      = int(max(0, min(z, Z - 1)))
    cy0, cy1  = max(0, y0), min(H, y1)
    cx0, cx1  = max(0, x0), min(W, x1)
    roi       = mask_arr[z_cl, cy0:cy1, cx0:cx1].astype(bool)
    pad_top   = max(0, -y0);  pad_bottom = max(0, y1 - H)
    pad_left  = max(0, -x0); pad_right  = max(0, x1 - W)
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        roi = np.pad(roi, ((pad_top, pad_bottom), (pad_left, pad_right)),
                     mode="constant", constant_values=False)
    crop      = crop.copy()
    crop[:, ~roi] = 0.0
    return crop.astype(np.float32)


def apply_roi_mask_per_ch(crop, mask_arr, z: int, y0: int, x0: int, y1: int, x1: int):
    """채널별(z-1/z/z+1) 마스크 적용 (A2 방식)."""
    if mask_arr is None:
        return crop
    import numpy as np
    Z, H, W  = mask_arr.shape
    zm       = max(z - 1, 0)
    zp       = min(z + 1, Z - 1)
    cy0, cy1 = max(0, y0), min(H, y1)
    cx0, cx1 = max(0, x0), min(W, x1)
    pad_top  = max(0, -y0);  pad_bottom = max(0, y1 - H)
    pad_left = max(0, -x0); pad_right  = max(0, x1 - W)
    needs_pad = pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0

    def _roi(zi):
        r = mask_arr[int(max(0, min(zi, Z - 1))), cy0:cy1, cx0:cx1].astype(bool)
        if needs_pad:
            r = np.pad(r, ((pad_top, pad_bottom), (pad_left, pad_right)),
                       mode="constant", constant_values=False)
        return r

    crop = crop.copy()
    crop[0][~_roi(zm)] = 0.0
    crop[1][~_roi(z)]  = 0.0
    crop[2][~_roi(zp)] = 0.0
    return crop.astype(np.float32)


def build_crop(ct_arr, local_z: int, y0: int, x0: int, y1: int, x1: int, mask_arr):
    """INPUT_TYPE에 따라 crop 생성 + ROI 마스킹 적용."""
    if INPUT_TYPE == "lung3ch":
        crop = build_lung3ch_crop(ct_arr, local_z, y0, x0, y1, x1)
    elif INPUT_TYPE == "medi_mip3ch":
        crop = build_mip3ch_crop(ct_arr, local_z, y0, x0, y1, x1)
    elif INPUT_TYPE == "lung_mip3ch":
        crop = build_mip3ch_crop(ct_arr, local_z, y0, x0, y1, x1)
    elif INPUT_TYPE == "lung_mip3ch_roipx":
        crop = build_mip3ch_crop(ct_arr, local_z, y0, x0, y1, x1)
        crop = apply_roi_mask_single_z(crop, mask_arr, local_z, y0, x0, y1, x1)
    elif INPUT_TYPE == "lung3ch_roipx":
        crop = build_lung3ch_crop(ct_arr, local_z, y0, x0, y1, x1)
        crop = apply_roi_mask_per_ch(crop, mask_arr, local_z, y0, x0, y1, x1)
    else:
        raise ValueError(f"알 수 없는 INPUT_TYPE: {INPUT_TYPE}")
    return crop


# =============================================================================
# RD4AD score 계산
# =============================================================================

def compute_rd4ad_score(teacher, student, crop_tensor, teacher_features, device, z_pct=None):
    import torch
    import torch.nn.functional as F

    teacher_features.clear()
    with torch.no_grad():
        teacher(crop_tensor)

    if TEACHER_TYPE == "resnet18":
        tf3 = teacher_features["layer3"]
        tf2 = teacher_features["layer2"]
        tf1 = teacher_features["layer1"]
        with torch.no_grad():
            de3, de2, de1 = student(tf3)
        pairs = [(tf3, de3), (tf2, de2), (tf1, de1)]
    elif TEACHER_TYPE == "effb0z":
        tf3 = teacher_features["late"]
        tf2 = teacher_features["mid"]
        tf1 = teacher_features["early"]
        # z_pct broadcast concat: (1,80,H,W) + (1,1,H,W) → (1,81,H,W)
        B, _, H, W = tf3.shape
        z_val = float(z_pct) if z_pct is not None else 0.5
        z_map = torch.full((B, 1, H, W), z_val, dtype=tf3.dtype, device=tf3.device)
        tf3_with_z = torch.cat([tf3, z_map], dim=1)
        with torch.no_grad():
            de_l, de_m, de_e = student(tf3_with_z)
        pairs = [(tf3, de_l), (tf2, de_m), (tf1, de_e)]
    else:  # effb0
        tf3 = teacher_features["late"]
        tf2 = teacher_features["mid"]
        tf1 = teacher_features["early"]
        with torch.no_grad():
            de_l, de_m, de_e = student(tf3)
        pairs = [(tf3, de_l), (tf2, de_m), (tf1, de_e)]

    scores = []
    for tf, sf in pairs:
        cos_sim = F.cosine_similarity(tf, sf, dim=1, eps=1e-8)
        scores.append(float((1.0 - cos_sim).mean().item()))

    scalar = float(sum(scores) / len(scores))
    # (scalar, score_layer1, score_layer2, score_layer3)
    return scalar, scores[2], scores[1], scores[0]


# =============================================================================
# HU feature 계산 (D1s와 동일 구조, per-exp HU_MIN/HU_MAX 사용)
# =============================================================================

def compute_hu_features(ct_arr, local_z: int,
                         crop_y0: int, crop_x0: int, crop_y1: int, crop_x1: int,
                         mask_arr):
    import numpy as np
    Z, H, W = ct_arr.shape
    cy0c = max(crop_y0, 0);  cy1c = min(crop_y1, H)
    cx0c = max(crop_x0, 0);  cx1c = min(crop_x1, W)

    _nan = {"crop_hu_mean": float("nan"), "crop_hu_std": float("nan"),
            "crop_hu_p10": float("nan"),  "crop_hu_p50": float("nan"),
            "crop_hu_p90": float("nan"),  "roi_0_0_patch_ratio": 0.0}

    if cy1c <= cy0c or cx1c <= cx0c:
        return _nan

    raw_patch  = ct_arr[local_z, cy0c:cy1c, cx0c:cx1c].astype(np.float32)
    norm_patch = (raw_patch.clip(HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)

    if mask_arr is not None:
        mZ, mH, mW = mask_arr.shape
        z_idx = min(local_z, mZ - 1)
        mcy0c = max(crop_y0, 0);  mcy1c = min(crop_y1, mH)
        mcx0c = max(crop_x0, 0);  mcx1c = min(crop_x1, mW)
        if mcy1c > mcy0c and mcx1c > mcx0c:
            mask_crop = mask_arr[z_idx, mcy0c:mcy1c, mcx0c:mcx1c]
            mask_crop = (mask_crop > 0).astype(bool)
            h_min     = min(cy1c - cy0c, mcy1c - mcy0c)
            w_min     = min(cx1c - cx0c, mcx1c - mcx0c)
            norm_v    = norm_patch[:h_min, :w_min]
            mask_v    = mask_crop[:h_min, :w_min]
            inside    = norm_v[mask_v]
            roi_ratio = int(mask_v.sum()) / (h_min * w_min) if h_min * w_min > 0 else 0.0
        else:
            inside    = norm_patch.ravel()
            roi_ratio = 0.0
    else:
        inside    = norm_patch.ravel()
        roi_ratio = 0.0

    if len(inside) == 0:
        inside = norm_patch.ravel()

    return {
        "crop_hu_mean":        float(np.mean(inside)),
        "crop_hu_std":         float(np.std(inside)),
        "crop_hu_p10":         float(np.percentile(inside, 10)),
        "crop_hu_p50":         float(np.percentile(inside, 50)),
        "crop_hu_p90":         float(np.percentile(inside, 90)),
        "roi_0_0_patch_ratio": float(roi_ratio),
    }


# =============================================================================
# adjusted score
# =============================================================================

def compute_adjusted(rd4ad_raw: float, roi_ratio):
    if roi_ratio is None or not math.isfinite(roi_ratio):
        return None, None
    p1 = rd4ad_raw * roi_ratio
    p2 = rd4ad_raw * math.sqrt(max(roi_ratio, 0.0))
    return p1, p2


# =============================================================================
# shard CSV 필드 (D1s와 동일)
# =============================================================================

SHARD_CSV_FIELDS = [
    "shard_id",
    "candidate_id",
    "patient_id",
    "safe_id",
    "track_id",
    "track_len",
    "local_z",
    "pos_y0", "pos_x0", "pos_y1", "pos_x1",
    "crop_y0", "crop_x0", "crop_y1", "crop_x1",
    "label",
    "ztrack_min_run_len",
    "score_original",
    "rd4ad_ztrack_score_raw",
    "score_layer1",
    "score_layer2",
    "score_layer3",
    "crop_hu_mean",
    "crop_hu_std",
    "crop_hu_p10",
    "crop_hu_p50",
    "crop_hu_p90",
    "roi_0_0_patch_ratio",
    "P1_times_roi",
    "P2_times_sqrt_roi",
]


# =============================================================================
# shard summary / DONE
# =============================================================================

def _write_shard_summary(path: Path, shard_id: int, expected: int,
                          scored: int, failed: int, errors: int,
                          nan: int, inf_: int, runtime: float,
                          is_smoke: bool) -> tuple:
    ensure_output_path_safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hard_fail = (
        GUARDRAILS["stage2_holdout_used_for_method_tuning"]
        or GUARDRAILS["training_executed"]
        or GUARDRAILS["backward_executed"]
        or GUARDRAILS["optimizer_created"]
        or GUARDRAILS["checkpoint_saved"]
        or GUARDRAILS["existing_artifact_modified"]
    )
    count_ok = (expected == scored + failed)

    if is_smoke:
        verdict = "SMOKE_PASS" if failed == 0 and errors == 0 and not hard_fail else "SMOKE_FAIL"
    elif hard_fail:
        verdict = "FAIL"
    elif failed == 0 and errors == 0 and nan == 0 and inf_ == 0 and count_ok:
        verdict = "PASS"
    elif count_ok and not hard_fail:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "FAIL"

    summary = {
        "exp_id":                                EXP_ID,
        "input_type":                            INPUT_TYPE,
        "teacher_type":                          TEACHER_TYPE,
        "shard_id":                              shard_id,
        "is_smoke":                              is_smoke,
        "expected_candidate_count":              expected,
        "actual_scored_candidate_count":         scored,
        "failed_candidate_count":                failed,
        "error_count":                           errors,
        "score_nan_count":                       nan,
        "score_inf_count":                       inf_,
        "runtime_sec":                           round(runtime, 1),
        "stage2_holdout_used_for_method_tuning": GUARDRAILS["stage2_holdout_used_for_method_tuning"],
        "checkpoint_loaded":                     GUARDRAILS["checkpoint_loaded"],
        "model_forward_executed":                GUARDRAILS["model_forward_executed"],
        "training_executed":                     GUARDRAILS["training_executed"],
        "backward_executed":                     GUARDRAILS["backward_executed"],
        "optimizer_created":                     GUARDRAILS["optimizer_created"],
        "checkpoint_saved":                      GUARDRAILS["checkpoint_saved"],
        "existing_artifact_modified":            GUARDRAILS["existing_artifact_modified"],
        "roi_hard_filter_applied":               GUARDRAILS["roi_hard_filter_applied"],
        "vessel_mask_applied":                   GUARDRAILS["vessel_mask_applied"],
        "all_survived_track_candidates_scored":  (failed == 0),
        "primary_candidate_score":               GUARDRAILS["primary_candidate_score"],
        "primary_track_score":                   GUARDRAILS["primary_track_score"],
        "verdict":                               verdict,
    }
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  saved: {path}")
    return verdict, summary


def _write_done(path: Path, verdict: str, shard_id: int, summary: dict = None) -> None:
    ensure_output_path_safe(path)
    d = {
        "verdict":   verdict,
        "exp_id":    EXP_ID,
        "shard_id":  shard_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if summary:
        for k in ["expected_candidate_count", "actual_scored_candidate_count",
                  "failed_candidate_count", "error_count",
                  "stage2_holdout_used_for_method_tuning"]:
            if k in summary:
                d[k] = summary[k]
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    print(f"  saved: {path}")


# =============================================================================
# dry-run
# =============================================================================

def run_dry() -> None:
    print("=" * 70)
    print(f"[DRY-RUN] rd_e1_abc_stage2_scoring  exp_id={EXP_ID}")
    print(f"  input_type={INPUT_TYPE}  teacher_type={TEACHER_TYPE}")
    print(f"  HU window=[{HU_MIN}, {HU_MAX}]")
    print("=" * 70)
    issues = []

    print("\n[1] 입력 파일 존재 확인")
    weight_path = LOCAL_EFFB0_WEIGHT if TEACHER_TYPE == "effb0" else LOCAL_RESNET_WEIGHT
    checks = {
        "candidate manifest": CANDIDATE_MANIFEST_CSV,
        "shard plan":         SHARD_PLAN_CSV,
        "checkpoint":         CKPT_PATH,
        "teacher weight":     weight_path,
        "CT root":            CT_ROOT,
        "ROI mask root":      ROI_MASK_ROOT,
    }
    for name, p in checks.items():
        ok = p.exists()
        print(f"  [{'OK' if ok else 'MISSING'}] {name}: {p}")
        if not ok:
            issues.append(f"missing: {name}")

    print("\n[2] shard plan 후보 수 확인")
    try:
        shard_plan_data: dict = {}
        for row in read_csv(SHARD_PLAN_CSV):
            sid = int(row["shard_id"])
            shard_plan_data[sid] = int(row["candidate_count"])
        total_expected = sum(shard_plan_data.values())
        for sid in range(SHARD_COUNT):
            cnt = shard_plan_data.get(sid, "?")
            print(f"  shard {sid}: {cnt} candidates")
        print(f"  합계: {total_expected:,}  (기대 {EXPECTED_TOTAL_CANDIDATES:,})")
        if total_expected != EXPECTED_TOTAL_CANDIDATES:
            issues.append(f"shard plan 합계 {total_expected} != {EXPECTED_TOTAL_CANDIDATES}")
    except Exception as e:
        issues.append(f"shard plan 읽기 실패: {e}")

    print("\n[3] 출력 overwrite 위험 확인")
    for sid in range(SHARD_COUNT):
        out_csv  = SHARDS_DIR / f"shard_{sid}" / f"stage2_rd4ad_scores_shard_{sid}.csv"
        done_p   = SHARDS_DIR / f"shard_{sid}" / "DONE.json"
        if out_csv.exists() or done_p.exists():
            print(f"  [WARN] shard {sid}: 이미 존재 → 재실행 시 overwrite")
        else:
            print(f"  [OK]   shard {sid}: 출력 없음")

    print("\n[4] guardrail 상태")
    GUARDRAILS["exp_id"] = EXP_ID
    for k, v in GUARDRAILS.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    if issues:
        print("[DRY-RUN] 이슈:")
        for it in issues:
            print(f"  - {it}")
        print("판정: NEEDS_FIX")
        sys.exit(1)
    else:
        print(f"[DRY-RUN] 모든 입력/계획/경로 OK. (exp_id={EXP_ID})")
        print("판정: READY  →  --run-shard --shard-id 0 --smoke-test 로 smoke 먼저")
    print("=" * 70)


# =============================================================================
# run-shard (smoke + full)
# =============================================================================

def run_shard(shard_id: int, is_smoke: bool) -> None:
    mode_str = "SMOKE" if is_smoke else "FULL"
    print("=" * 70)
    print(f"[RUN-SHARD-{mode_str}] exp_id={EXP_ID}  shard={shard_id}")
    print(f"  input_type={INPUT_TYPE}  teacher_type={TEACHER_TYPE}")
    print(f"  HU window=[{HU_MIN}, {HU_MAX}]")
    print("=" * 70)
    t0 = time.perf_counter()

    shard_dir = SHARDS_DIR / f"shard_{shard_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    suffix       = "_smoke" if is_smoke else ""
    out_csv      = shard_dir / f"stage2_rd4ad_scores_shard_{shard_id}{suffix}.csv"
    summary_json = shard_dir / f"shard_{shard_id}{suffix}_summary.json"
    error_csv    = shard_dir / f"errors{suffix}.csv"
    done_json    = shard_dir / f"DONE{suffix}.json"

    GUARDRAILS["output_overwrite"] = bool(out_csv.exists() or done_json.exists())
    if GUARDRAILS["output_overwrite"]:
        print(f"  [WARN] 출력 이미 존재 → overwrite (output_overwrite=True)")

    append_error = make_error_logger(error_csv)
    GUARDRAILS["exp_id"] = EXP_ID

    # ── [1] shard plan ──────────────────────────────────────────────────────
    print("\n[1] shard plan 로드")
    shard_plan_data: dict = {}
    for row in read_csv(SHARD_PLAN_CSV):
        sid = int(row["shard_id"])
        shard_plan_data[sid] = int(row["candidate_count"])
    expected_candidate_count = shard_plan_data.get(shard_id, -1)
    print(f"  shard {shard_id} expected: {expected_candidate_count:,}")

    # ── [2] manifest 로드 ────────────────────────────────────────────────────
    print("\n[2] manifest 로드 및 shard 필터링")
    all_rows   = read_csv(CANDIDATE_MANIFEST_CSV)
    shard_rows = [r for r in all_rows if _patient_shard(r["patient_id"]) == shard_id]
    print(f"  전체: {len(all_rows):,}  shard {shard_id}: {len(shard_rows):,}")

    if len(shard_rows) != expected_candidate_count:
        msg = (f"shard {shard_id} 후보수 불일치: "
               f"manifest={len(shard_rows)} plan={expected_candidate_count}")
        append_error(msg)
        print(f"  [WARN] {msg}")

    if not shard_rows:
        print(f"  [ABORT] shard {shard_id} candidate 없음")
        sys.exit(2)

    if is_smoke:
        shard_rows = shard_rows[:SMOKE_N]
        print(f"  [SMOKE] 처음 {len(shard_rows)}개만 처리")

    # ── [3] 모델 로드 ────────────────────────────────────────────────────────
    print("\n[3] 모델 로드 (checkpoint read-only)")
    import torch
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    teacher, student = load_model_from_checkpoint(device)
    teacher.eval()
    student.eval()

    teacher_features: dict = {}
    setup_teacher_hooks(teacher, teacher_features)

    teacher.train = _forbidden_train
    student.train = _forbidden_train

    ct_cache   = CTMmapCache(max_size=12)
    mask_cache = RoiMaskCache(max_size=12)

    # ── E2z용 z_pct lookup table 빌드 ────────────────────────────────────────
    z_pct_lookup = {}  # (safe_id, local_z) → float
    if TEACHER_TYPE == "effb0z":
        safe_id_z_min = {}
        safe_id_z_max = {}
        for r in shard_rows:
            sid_v = r["safe_id"]
            lz_v  = int(r["local_z"])
            if sid_v not in safe_id_z_min or lz_v < safe_id_z_min[sid_v]:
                safe_id_z_min[sid_v] = lz_v
            if sid_v not in safe_id_z_max or lz_v > safe_id_z_max[sid_v]:
                safe_id_z_max[sid_v] = lz_v
        for r in shard_rows:
            sid_v = r["safe_id"]
            lz    = int(r["local_z"])
            z_min = safe_id_z_min[sid_v]
            z_max = safe_id_z_max[sid_v]
            z_pct_lookup[(sid_v, lz)] = (
                0.5 if z_max == z_min
                else float(lz - z_min) / float(z_max - z_min)
            )
        print(f"  [E2z] z_pct lookup built: {len(z_pct_lookup):,} entries")

    # ── [4] forward scoring ──────────────────────────────────────────────────
    print(f"\n[4] forward scoring — {len(shard_rows):,} candidates (shard {shard_id})")
    GUARDRAILS["model_forward_executed"]   = True
    GUARDRAILS["crop_generation_executed"] = True
    GUARDRAILS["scoring_executed"]         = True

    score_rows:      list = []
    error_count:     int  = 0
    failed_count:    int  = 0
    score_nan_count: int  = 0
    score_inf_count: int  = 0
    log_interval = max(1, len(shard_rows) // 20)

    for idx, row in enumerate(shard_rows):
        if idx % log_interval == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{idx:6d}/{len(shard_rows)}] elapsed={elapsed:.0f}s  failed={failed_count}")

        cid     = row["candidate_id"]
        safe_id = row["safe_id"]

        try:
            local_z = int(row["local_z"])
            pos_y0  = int(row["pos_y0"]);  pos_x0 = int(row["pos_x0"])
            pos_y1  = int(row["pos_y1"]);  pos_x1 = int(row["pos_x1"])
            crop_y0 = int(row["crop_y0"]); crop_x0 = int(row["crop_x0"])
            crop_y1 = int(row["crop_y1"]); crop_x1 = int(row["crop_x1"])
        except Exception as e:
            append_error(f"coord parse fail: {cid}: {e}", e)
            error_count += 1; failed_count += 1; continue

        try:
            ct_arr = ct_cache.get(safe_id)
        except Exception as e:
            append_error(f"CT load fail: {safe_id} (cid={cid}): {e}", e)
            error_count += 1; failed_count += 1; continue

        try:
            mask_arr = mask_cache.get(safe_id)
        except Exception:
            mask_arr = None

        try:
            crop = build_crop(ct_arr, local_z, crop_y0, crop_x0, crop_y1, crop_x1, mask_arr)
        except Exception as e:
            append_error(f"crop build fail: {cid}: {e}", e)
            error_count += 1; failed_count += 1; continue

        try:
            hu_feat = compute_hu_features(
                ct_arr, local_z, crop_y0, crop_x0, crop_y1, crop_x1, mask_arr)
        except Exception as e:
            append_error(f"HU feature fail: {cid}: {e}", e)
            hu_feat = {"crop_hu_mean": float("nan"), "crop_hu_std": float("nan"),
                       "crop_hu_p10": float("nan"), "crop_hu_p50": float("nan"),
                       "crop_hu_p90": float("nan"), "roi_0_0_patch_ratio": 0.0}

        crop_t = torch.from_numpy(crop[np.newaxis]).to(device)
        try:
            z_pct_val = z_pct_lookup.get((safe_id, local_z), 0.5) if TEACHER_TYPE == "effb0z" else None
            with torch.no_grad():
                rd4ad_raw, l1, l2, l3 = compute_rd4ad_score(
                    teacher, student, crop_t, teacher_features, device, z_pct=z_pct_val)
        except Exception as e:
            append_error(f"forward fail: {cid}: {e}", e)
            error_count += 1; failed_count += 1; continue

        if math.isnan(rd4ad_raw):  score_nan_count += 1
        if math.isinf(rd4ad_raw):  score_inf_count += 1

        roi_ratio = hu_feat["roi_0_0_patch_ratio"]
        p1, p2    = compute_adjusted(rd4ad_raw, roi_ratio)

        def _hf(k):
            v = hu_feat[k]
            return "" if math.isnan(v) else v

        score_rows.append({
            "shard_id":               shard_id,
            "candidate_id":           cid,
            "patient_id":             row["patient_id"],
            "safe_id":                safe_id,
            "track_id":               row.get("track_id", ""),
            "track_len":              row.get("track_len", ""),
            "local_z":                local_z,
            "pos_y0":  pos_y0,  "pos_x0":  pos_x0,  "pos_y1":  pos_y1,  "pos_x1":  pos_x1,
            "crop_y0": crop_y0, "crop_x0": crop_x0, "crop_y1": crop_y1, "crop_x1": crop_x1,
            "label":                  row.get("label", ""),
            "ztrack_min_run_len":     row.get("ztrack_min_run_len", "2"),
            "score_original":         row.get("score_original", ""),
            "rd4ad_ztrack_score_raw": rd4ad_raw,
            "score_layer1":           l1,
            "score_layer2":           l2,
            "score_layer3":           l3,
            "crop_hu_mean":           _hf("crop_hu_mean"),
            "crop_hu_std":            _hf("crop_hu_std"),
            "crop_hu_p10":            _hf("crop_hu_p10"),
            "crop_hu_p50":            _hf("crop_hu_p50"),
            "crop_hu_p90":            _hf("crop_hu_p90"),
            "roi_0_0_patch_ratio":    roi_ratio,
            "P1_times_roi":           "" if p1 is None else p1,
            "P2_times_sqrt_roi":      "" if p2 is None else p2,
        })

    GUARDRAILS["all_survived_track_candidates_scored"] = (failed_count == 0)
    runtime       = time.perf_counter() - t0
    actual_scored = len(score_rows)

    print(f"\n  scored={actual_scored:,}  failed={failed_count}  "
          f"errors={error_count}  NaN={score_nan_count}  Inf={score_inf_count}  "
          f"runtime={runtime:.0f}s  "
          f"({actual_scored / max(1, runtime):.0f} rows/s)")

    # ── [5] CSV 저장 ─────────────────────────────────────────────────────────
    print("\n[5] shard CSV 저장")
    if score_rows:
        write_csv(out_csv, SHARD_CSV_FIELDS, score_rows)

    # ── [6] summary / DONE ───────────────────────────────────────────────────
    print("\n[6] summary / DONE 저장")
    expected  = SMOKE_N if is_smoke else expected_candidate_count
    verdict, summ = _write_shard_summary(
        summary_json, shard_id,
        expected=expected, scored=actual_scored,
        failed=failed_count, errors=error_count,
        nan=score_nan_count, inf_=score_inf_count,
        runtime=runtime, is_smoke=is_smoke,
    )
    _write_done(done_json, verdict, shard_id, summ)

    print("\n" + "=" * 70)
    print(f"[RUN-SHARD-{mode_str} {shard_id}] exp={EXP_ID}  완료 ({runtime:.1f}s)  판정: {verdict}")
    print("=" * 70)

    if verdict in ("FAIL", "SMOKE_FAIL"):
        sys.exit(1)


# =============================================================================
# main
# =============================================================================

def main() -> None:
    if len(sys.argv) < 2:
        print("[ABORT] bare run 차단.", file=sys.stderr)
        print("  dry-run:  --exp-id {A|B|C|C2|A2|E1|E2} --dry-run", file=sys.stderr)
        print("  smoke:    --exp-id A --run-shard --shard-id 0 --smoke-test "
              "--confirm-model-forward --confirm-stage2-holdout-eval-only", file=sys.stderr)
        print("  full:     --exp-id A --run-shard --shard-id {0..7} "
              "--confirm-model-forward --confirm-stage2-holdout-eval-only", file=sys.stderr)
        sys.exit(2)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--exp-id",   choices=list(EXP_CONFIGS.keys()), required=True)
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--run-shard", action="store_true")
    parser.add_argument("--shard-id", type=int, choices=list(range(SHARD_COUNT)))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--confirm-model-forward",            action="store_true")
    parser.add_argument("--confirm-stage2-holdout-eval-only", action="store_true")
    args = parser.parse_args()

    if EXP_ID is None:
        print("[ABORT] --exp-id 필수.", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        run_dry()
        return

    if args.run_shard:
        if not (args.confirm_model_forward and args.confirm_stage2_holdout_eval_only):
            print("[ABORT] --run-shard 실행 시 --confirm-model-forward 와 "
                  "--confirm-stage2-holdout-eval-only 필요.", file=sys.stderr)
            sys.exit(2)
        if args.shard_id is None:
            print("[ABORT] --shard-id 필요.", file=sys.stderr)
            sys.exit(2)
        run_shard(args.shard_id, is_smoke=args.smoke_test)
        return

    print("[ABORT] --dry-run 또는 --run-shard 사용.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
