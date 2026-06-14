"""
stage2_strict_ztrack_rd4ad_scoring_run.py

목적:
  stage2_holdout survived 128,827개 후보에 대해 RD4AD scoring 수행.
  scoring loop에서 CT를 1회만 읽어 HU feature와 ROI ratio를 같이 계산한다.

고정 조건 (stage1_dev 확정, 변경 금지):
  - input = stage2_rd4ad_scoring_manifest_minrun2.csv
  - checkpoint = best_train_loss.pth
  - model.eval(), torch.no_grad()
  - no backward, no optimizer, no checkpoint save
  - shard별 실행 (patient_id stable hash, 8 shards)
  - primary_candidate_score = P1_times_roi
  - primary_track_score = P1_track_top3_mean
  - vessel mask 미적용, ROI hard filter 금지
  - score_original threshold 사용 금지

추가 출력 feature (CT 재로딩 병목 방지):
  crop_hu_mean, crop_hu_std, crop_hu_p10, crop_hu_p50, crop_hu_p90
  roi_0_0_patch_ratio (v4_20 마스크 기반)
  P1_times_roi, P2_times_sqrt_roi
  track_id, track_len

실행:
  bare run            → sys.exit(2) 차단
  dry-run             → --dry-run
  smoke test (50개)   → --run-shard --shard-id 0 --smoke-test
                        --confirm-model-forward --confirm-stage2-holdout-eval-only
  full shard          → --run-shard --shard-id {0..7}
                        --confirm-model-forward --confirm-stage2-holdout-eval-only
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
# 경로 상수
# =============================================================================

PROJECT_ROOT    = Path("/home/jinhy/project/lung-ct-anomaly")
EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "experiments/stage2_strict_ztrack_rd4ad_scoring_preflight_v1"
)

# 입력 (read-only)
CANDIDATE_MANIFEST_CSV = (
    EXPERIMENT_ROOT / "manifests/stage2_rd4ad_scoring_manifest_minrun2.csv"
)
SHARD_PLAN_CSV = (
    EXPERIMENT_ROOT / "manifests/stage2_rd4ad_scoring_shard_plan.csv"
)
CKPT_PATH = (
    PROJECT_ROOT
    / "outputs/models/rd_d1s_true_rd4ad_resnet18_medi3ch_shard_v1"
    / "checkpoints/best_train_loss.pth"
)
LOCAL_RESNET_WEIGHT = Path(
    "/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
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

# 출력 (shards/ 하위에만)
SHARDS_DIR = EXPERIMENT_ROOT / "shards"

# =============================================================================
# 상수
# =============================================================================

SHARD_COUNT               = 8
CROP_SIZE                 = 96
HU_MIN, HU_MAX            = -160.0, 240.0
EXPECTED_TOTAL_CANDIDATES = 128_827
COMPLETE_MISS_KNOWN       = {"LUNG1-415"}
SMOKE_N                   = 50

# =============================================================================
# guardrail
# =============================================================================

GUARDRAILS = {
    "stage2_holdout_used_for_method_tuning":  False,
    "stage2_holdout_eval_only":               True,
    "checkpoint_loaded":                      False,
    "model_forward_executed":                 False,
    "training_executed":                      False,
    "backward_executed":                      False,
    "optimizer_created":                      False,
    "checkpoint_saved":                       False,
    "crop_generation_executed":               False,
    "scoring_executed":                       False,
    "existing_artifact_modified":             False,
    "output_overwrite":                       False,
    "label_used_for_evaluation_only":         True,
    "label_used_as_selector":                 False,
    "min_run_exception_added":                False,
    "score_original_rescue_used":             False,
    "roi_hard_filter_applied":                False,
    "vessel_mask_applied":                    False,
    "all_survived_track_candidates_scored":   False,
    "primary_candidate_score":                "P1_times_roi",
    "primary_track_score":                    "P1_track_top3_mean",
    "auxiliary_track_score":                  "P1_track_top2_mean",
}

# =============================================================================
# 안전 경로 검사
# =============================================================================

_PROTECTED_INPUTS = [
    CANDIDATE_MANIFEST_CSV,
    SHARD_PLAN_CSV,
    CKPT_PATH,
    LOCAL_RESNET_WEIGHT,
]


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
    shards_root = str(SHARDS_DIR.resolve())
    if not str(rp).startswith(shards_root):
        GUARDRAILS["existing_artifact_modified"] = True
        raise RuntimeError(f"[ABORT] shards/ 외부 쓰기 차단: {p}")


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
# shard 할당
# =============================================================================

def _patient_shard(patient_id: str, n_shards: int = SHARD_COUNT) -> int:
    return int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % n_shards


# =============================================================================
# ROI mask cache (v4_20 modeB)
# =============================================================================

class RoiMaskCache:
    """safe_id별 refined_roi.npy mmap 캐시."""

    def __init__(self, max_size: int = 12):
        self._cache: OrderedDict = OrderedDict()
        self._max = max_size

    def get(self, safe_id: str):
        import numpy as np

        if safe_id in self._cache:
            self._cache.move_to_end(safe_id)
            return self._cache[safe_id]

        # lesion / normal 순서로 탐색
        mask_arr = None
        for subset in ("lesion", "normal"):
            p = ROI_MASK_ROOT / subset / safe_id / "refined_roi.npy"
            if p.exists():
                mask_arr = np.load(str(p), mmap_mode="r")
                break

        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        self._cache[safe_id] = mask_arr  # None이면 마스크 없음
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
# 모델 빌드 (stage1과 동일 구조)
# =============================================================================

def build_teacher():
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


def build_student_decoder():
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


def load_model_from_checkpoint(device):
    import torch

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"checkpoint 없음: {CKPT_PATH}")

    teacher = build_teacher().to(device)
    student = build_student_decoder().to(device)
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


def _forbidden_train(*args, **kwargs):
    GUARDRAILS["training_executed"] = True
    raise RuntimeError("[ABORT] training 호출 금지됨")


# =============================================================================
# crop 생성 (stage1과 동일)
# =============================================================================

def build_medi3ch_crop(ct_arr, local_z: int, y0: int, x0: int, y1: int, x1: int):
    import numpy as np

    Z, H, W = ct_arr.shape

    def _win(patch):
        c = patch.astype(np.float32)
        c = (c.clip(HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
        return c

    def _clip_and_pad(z_idx: int, cy0: int, cx0: int, cy1: int, cx1: int):
        cy0c = max(cy0, 0);  cy1c = min(cy1, H)
        cx0c = max(cx0, 0);  cx1c = min(cx1, W)
        if cy1c <= cy0c or cx1c <= cx0c:
            return np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        patch = ct_arr[z_idx, cy0c:cy1c, cx0c:cx1c]
        patch = _win(patch)
        out   = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        dy0   = cy0c - cy0
        dx0   = cx0c - cx0
        out[dy0:dy0 + (cy1c - cy0c), dx0:dx0 + (cx1c - cx0c)] = patch
        return out

    zs       = [max(0, local_z - 1), local_z, min(Z - 1, local_z + 1)]
    channels = [_clip_and_pad(z, y0, x0, y1, x1) for z in zs]
    crop     = __import__("numpy").stack(channels, axis=0)
    if crop.shape != (3, CROP_SIZE, CROP_SIZE):
        raise ValueError(f"crop shape {crop.shape} != (3,{CROP_SIZE},{CROP_SIZE})")
    if not __import__("numpy").isfinite(crop).all():
        raise ValueError("crop contains NaN/Inf")
    return crop.astype(__import__("numpy").float32)


# =============================================================================
# RD4AD score (stage1과 동일)
# =============================================================================

def compute_rd4ad_score(teacher, student, crop_tensor, teacher_features, device):
    import torch
    import torch.nn.functional as F

    teacher_features.clear()
    with torch.no_grad():
        teacher(crop_tensor)

    tf3 = teacher_features["layer3"]
    tf2 = teacher_features["layer2"]
    tf1 = teacher_features["layer1"]

    with torch.no_grad():
        de3, de2, de1 = student(tf3)

    scores = []
    for tf, sf in [(tf3, de3), (tf2, de2), (tf1, de1)]:
        cos_sim = F.cosine_similarity(tf, sf, dim=1, eps=1e-8)
        scores.append(float((1.0 - cos_sim).mean().item()))

    scalar = float(sum(scores) / len(scores))
    # layer1, layer2, layer3 순서
    return scalar, scores[2], scores[1], scores[0]


# =============================================================================
# HU feature 계산 (CT에서 직접 계산, 정규화 후 마스크 내부만)
# =============================================================================

def compute_hu_features(
    ct_arr,
    local_z: int,
    crop_y0: int, crop_x0: int, crop_y1: int, crop_x1: int,
    mask_arr,
):
    """
    CT crop 영역에서 raw HU 값을 읽고, v4_20 ROI mask 내부만 정규화 후 통계 계산.
    정규화: (clip(HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)

    Returns dict with crop_hu_mean, crop_hu_std, crop_hu_p10, crop_hu_p50, crop_hu_p90,
    roi_0_0_patch_ratio.
    """
    import numpy as np

    Z, H, W = ct_arr.shape
    cy0c = max(crop_y0, 0);  cy1c = min(crop_y1, H)
    cx0c = max(crop_x0, 0);  cx1c = min(crop_x1, W)

    # CT raw HU 추출 (중앙 슬라이스만)
    if cy1c <= cy0c or cx1c <= cx0c:
        return {
            "crop_hu_mean": float("nan"),
            "crop_hu_std":  float("nan"),
            "crop_hu_p10":  float("nan"),
            "crop_hu_p50":  float("nan"),
            "crop_hu_p90":  float("nan"),
            "roi_0_0_patch_ratio": 0.0,
        }

    raw_patch = ct_arr[local_z, cy0c:cy1c, cx0c:cx1c].astype(np.float32)
    norm_patch = (raw_patch.clip(HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)

    # crop 영역에 대응하는 마스크 추출
    if mask_arr is not None:
        mZ, mH, mW = mask_arr.shape if mask_arr.ndim == 3 else (1, *mask_arr.shape)
        if mask_arr.ndim == 3:
            # z 인덱스 사용 (mask z가 ct z와 동일하다고 가정)
            z_idx = min(local_z, mZ - 1)
            mask_slice = mask_arr[z_idx]
        else:
            mask_slice = mask_arr

        mH2, mW2 = mask_slice.shape
        # crop 좌표가 mask 범위 내인지 확인
        mcy0c = max(crop_y0, 0);  mcy1c = min(crop_y1, mH2)
        mcx0c = max(crop_x0, 0);  mcx1c = min(crop_x1, mW2)

        if mcy1c > mcy0c and mcx1c > mcx0c:
            mask_crop = mask_arr[z_idx, mcy0c:mcy1c, mcx0c:mcx1c] if mask_arr.ndim == 3 \
                        else mask_slice[mcy0c:mcy1c, mcx0c:mcx1c]
            mask_crop = (mask_crop > 0).astype(bool)

            # norm_patch와 mask_crop 크기 맞추기 (경계 padding으로 크기 다를 수 있음)
            h_crop = cy1c - cy0c
            w_crop = cx1c - cx0c
            h_mask = mcy1c - mcy0c
            w_mask = mcx1c - mcx0c
            h_min  = min(h_crop, h_mask)
            w_min  = min(w_crop, w_mask)

            norm_valid = norm_patch[:h_min, :w_min]
            mask_valid = mask_crop[:h_min, :w_min]

            inside = norm_valid[mask_valid]
            total_pixels   = h_min * w_min
            inside_pixels  = int(mask_valid.sum())
            roi_ratio      = inside_pixels / total_pixels if total_pixels > 0 else 0.0
        else:
            inside    = norm_patch.ravel()
            roi_ratio = 0.0
    else:
        inside    = norm_patch.ravel()
        roi_ratio = 0.0

    if len(inside) == 0:
        inside = norm_patch.ravel()

    return {
        "crop_hu_mean":       float(np.mean(inside)),
        "crop_hu_std":        float(np.std(inside)),
        "crop_hu_p10":        float(np.percentile(inside, 10)),
        "crop_hu_p50":        float(np.percentile(inside, 50)),
        "crop_hu_p90":        float(np.percentile(inside, 90)),
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
# shard CSV 필드
# =============================================================================

SHARD_CSV_FIELDS = [
    "shard_id",
    "candidate_id",
    "patient_id",
    "safe_id",
    "track_id",
    "track_len",
    "local_z",
    "pos_y0",  "pos_x0",  "pos_y1",  "pos_x1",
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

def _write_shard_summary(
    path: Path,
    shard_id: int,
    expected: int,
    scored: int,
    failed: int,
    errors: int,
    nan: int,
    inf_: int,
    runtime: float,
    is_smoke: bool,
) -> tuple:
    ensure_output_path_safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count_ok  = (expected == scored + failed)
    hard_fail = (
        GUARDRAILS["stage2_holdout_used_for_method_tuning"]
        or GUARDRAILS["training_executed"]
        or GUARDRAILS["backward_executed"]
        or GUARDRAILS["optimizer_created"]
        or GUARDRAILS["checkpoint_saved"]
        or GUARDRAILS["existing_artifact_modified"]
    )

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
        "min_run_exception_added":               GUARDRAILS["min_run_exception_added"],
        "score_original_rescue_used":            GUARDRAILS["score_original_rescue_used"],
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
        "shard_id":  shard_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if summary:
        for k in [
            "expected_candidate_count",
            "actual_scored_candidate_count",
            "failed_candidate_count",
            "error_count",
            "stage2_holdout_used_for_method_tuning",
        ]:
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
    print("[DRY-RUN] stage2 strict z-track RD4AD scoring RUN")
    print("  파일 생성 없음 / model forward 없음 / checkpoint load 없음")
    print("=" * 70)
    issues = []

    print("\n[1] 입력 파일 존재 확인")
    checks = {
        "candidate manifest": CANDIDATE_MANIFEST_CSV,
        "shard plan":         SHARD_PLAN_CSV,
        "checkpoint":         CKPT_PATH,
        "ResNet18 weight":    LOCAL_RESNET_WEIGHT,
        "CT root":            CT_ROOT,
        "ROI mask root":      ROI_MASK_ROOT,
    }
    for name, p in checks.items():
        ok = p.exists()
        print(f"  [{'OK' if ok else 'MISSING'}] {name}: {p}")
        if not ok:
            issues.append(f"missing: {name}")

    print("\n[2] shard plan 후보 수 확인")
    shard_plan_data: dict = {}
    try:
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
        out_csv   = SHARDS_DIR / f"shard_{sid}" / f"stage2_rd4ad_scores_shard_{sid}.csv"
        done_p    = SHARDS_DIR / f"shard_{sid}" / "DONE.json"
        smoke_csv = SHARDS_DIR / f"shard_{sid}" / f"stage2_rd4ad_scores_shard_{sid}_smoke.csv"
        if out_csv.exists() or done_p.exists():
            print(f"  [WARN] shard {sid}: 이미 존재 (재실행 시 overwrite)")
        elif smoke_csv.exists():
            print(f"  [INFO] shard {sid}: smoke 결과 있음, full run 없음")
        else:
            print(f"  [OK]   shard {sid}: 출력 없음")

    print("\n[4] guardrail 상태")
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
        print("[DRY-RUN] 모든 입력/계획/경로 OK.")
        print("판정: READY  →  --run-shard --shard-id 0 --smoke-test 로 smoke 먼저")
    print("=" * 70)


# =============================================================================
# run-shard (smoke + full)
# =============================================================================

def run_shard(shard_id: int, is_smoke: bool) -> None:
    mode_str = "SMOKE" if is_smoke else "FULL"
    print("=" * 70)
    print(f"[RUN-SHARD-{mode_str}] stage2 RD4AD scoring — shard {shard_id}")
    print("=" * 70)
    t0 = time.perf_counter()

    shard_dir = SHARDS_DIR / f"shard_{shard_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    suffix        = "_smoke" if is_smoke else ""
    out_csv       = shard_dir / f"stage2_rd4ad_scores_shard_{shard_id}{suffix}.csv"
    summary_json  = shard_dir / f"shard_{shard_id}{suffix}_summary.json"
    error_csv     = shard_dir / f"errors{suffix}.csv"
    done_json     = shard_dir / f"DONE{suffix}.json"

    output_existed = out_csv.exists() or done_json.exists()
    GUARDRAILS["output_overwrite"] = bool(output_existed)
    if output_existed:
        print(f"  [WARN] 출력 이미 존재 → overwrite (output_overwrite=True)")

    append_error = make_error_logger(error_csv)

    # ── [1] shard plan 로드 ──────────────────────────────────────────────────
    print("\n[1] shard plan 로드")
    shard_plan_data: dict = {}
    for row in read_csv(SHARD_PLAN_CSV):
        sid = int(row["shard_id"])
        shard_plan_data[sid] = int(row["candidate_count"])
    expected_candidate_count = shard_plan_data.get(shard_id, -1)
    print(f"  shard {shard_id} expected: {expected_candidate_count:,}")

    # ── [2] manifest 로드 및 shard 필터 ─────────────────────────────────────
    print("\n[2] manifest 로드 및 shard 필터링")
    all_rows   = read_csv(CANDIDATE_MANIFEST_CSV)
    shard_rows = [
        r for r in all_rows
        if _patient_shard(r["patient_id"]) == shard_id
    ]
    print(f"  전체 rows: {len(all_rows):,}  shard {shard_id}: {len(shard_rows):,}")

    if len(shard_rows) != expected_candidate_count:
        msg = (
            f"shard {shard_id} 후보수 불일치: "
            f"manifest={len(shard_rows)} plan={expected_candidate_count}"
        )
        append_error(msg)
        print(f"  [WARN] {msg}")

    if not shard_rows:
        print(f"  [ABORT] shard {shard_id} 에 candidate 없음")
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
    for layer_name, module in [
        ("layer1", teacher.layer1),
        ("layer2", teacher.layer2),
        ("layer3", teacher.layer3),
    ]:
        def _hook(mod, inp, output, _n=layer_name):
            teacher_features[_n] = output
        module.register_forward_hook(_hook)

    teacher.train = _forbidden_train
    student.train = _forbidden_train

    ct_cache   = CTMmapCache(max_size=12)
    mask_cache = RoiMaskCache(max_size=12)

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
            elapsed_so_far = time.perf_counter() - t0
            print(
                f"  [{idx:6d}/{len(shard_rows)}] "
                f"elapsed={elapsed_so_far:.0f}s  failed={failed_count}"
            )

        cid     = row["candidate_id"]
        safe_id = row["safe_id"]

        # 좌표 파싱
        try:
            local_z  = int(row["local_z"])
            pos_y0   = int(row["pos_y0"])
            pos_x0   = int(row["pos_x0"])
            pos_y1   = int(row["pos_y1"])
            pos_x1   = int(row["pos_x1"])
            crop_y0  = int(row["crop_y0"])
            crop_x0  = int(row["crop_x0"])
            crop_y1  = int(row["crop_y1"])
            crop_x1  = int(row["crop_x1"])
        except Exception as e:
            append_error(f"coord parse fail: {cid}: {e}", e)
            error_count  += 1
            failed_count += 1
            continue

        # CT load
        try:
            ct_arr = ct_cache.get(safe_id)
        except Exception as e:
            append_error(f"CT load fail: {safe_id} (cid={cid}): {e}", e)
            error_count  += 1
            failed_count += 1
            continue

        # ROI mask load (None 가능 → fallback)
        try:
            mask_arr = mask_cache.get(safe_id)
        except Exception:
            mask_arr = None

        # crop 생성 (96×96, center±48 이미 계산된 crop 좌표 사용)
        try:
            crop = build_medi3ch_crop(
                ct_arr, local_z, crop_y0, crop_x0, crop_y1, crop_x1
            )
        except Exception as e:
            append_error(f"crop build fail: {cid}: {e}", e)
            error_count  += 1
            failed_count += 1
            continue

        # HU feature + ROI ratio (CT 재로딩 없이 같이 계산)
        try:
            hu_feat = compute_hu_features(
                ct_arr, local_z,
                crop_y0, crop_x0, crop_y1, crop_x1,
                mask_arr,
            )
        except Exception as e:
            append_error(f"HU feature fail: {cid}: {e}", e)
            hu_feat = {
                "crop_hu_mean":       float("nan"),
                "crop_hu_std":        float("nan"),
                "crop_hu_p10":        float("nan"),
                "crop_hu_p50":        float("nan"),
                "crop_hu_p90":        float("nan"),
                "roi_0_0_patch_ratio": 0.0,
            }

        # RD4AD forward
        crop_t = torch.from_numpy(crop[np.newaxis]).to(device)
        try:
            with torch.no_grad():
                rd4ad_raw, l1, l2, l3 = compute_rd4ad_score(
                    teacher, student, crop_t, teacher_features, device
                )
        except Exception as e:
            append_error(f"forward fail: {cid}: {e}", e)
            error_count  += 1
            failed_count += 1
            continue

        if math.isnan(rd4ad_raw):
            score_nan_count += 1
        if math.isinf(rd4ad_raw):
            score_inf_count += 1

        # P1, P2 계산
        roi_ratio = hu_feat["roi_0_0_patch_ratio"]
        p1, p2    = compute_adjusted(rd4ad_raw, roi_ratio)

        score_rows.append({
            "shard_id":              shard_id,
            "candidate_id":          cid,
            "patient_id":            row["patient_id"],
            "safe_id":               safe_id,
            "track_id":              row.get("track_id", ""),
            "track_len":             row.get("track_len", ""),
            "local_z":               local_z,
            "pos_y0":                pos_y0,
            "pos_x0":                pos_x0,
            "pos_y1":                pos_y1,
            "pos_x1":                pos_x1,
            "crop_y0":               crop_y0,
            "crop_x0":               crop_x0,
            "crop_y1":               crop_y1,
            "crop_x1":               crop_x1,
            "label":                 row.get("label", ""),
            "ztrack_min_run_len":    row.get("ztrack_min_run_len", "2"),
            "score_original":        row.get("score_original", ""),
            "rd4ad_ztrack_score_raw": rd4ad_raw,
            "score_layer1":          l1,
            "score_layer2":          l2,
            "score_layer3":          l3,
            "crop_hu_mean":          "" if math.isnan(hu_feat["crop_hu_mean"]) else hu_feat["crop_hu_mean"],
            "crop_hu_std":           "" if math.isnan(hu_feat["crop_hu_std"])  else hu_feat["crop_hu_std"],
            "crop_hu_p10":           "" if math.isnan(hu_feat["crop_hu_p10"])  else hu_feat["crop_hu_p10"],
            "crop_hu_p50":           "" if math.isnan(hu_feat["crop_hu_p50"])  else hu_feat["crop_hu_p50"],
            "crop_hu_p90":           "" if math.isnan(hu_feat["crop_hu_p90"])  else hu_feat["crop_hu_p90"],
            "roi_0_0_patch_ratio":   roi_ratio,
            "P1_times_roi":          "" if p1 is None else p1,
            "P2_times_sqrt_roi":     "" if p2 is None else p2,
        })

    GUARDRAILS["all_survived_track_candidates_scored"] = (failed_count == 0)
    runtime       = time.perf_counter() - t0
    actual_scored = len(score_rows)

    print(
        f"\n  scored={actual_scored:,}  failed={failed_count}  "
        f"errors={error_count}  NaN={score_nan_count}  Inf={score_inf_count}  "
        f"runtime={runtime:.0f}s"
    )

    # ── [5] CSV 저장 ─────────────────────────────────────────────────────────
    print(f"\n[5] shard CSV 저장")
    if score_rows:
        write_csv(out_csv, SHARD_CSV_FIELDS, score_rows)

    # ── [6] summary / DONE 저장 ──────────────────────────────────────────────
    print(f"\n[6] summary / DONE 저장")
    expected = SMOKE_N if is_smoke else expected_candidate_count
    verdict, summ = _write_shard_summary(
        summary_json,
        shard_id,
        expected=expected,
        scored=actual_scored,
        failed=failed_count,
        errors=error_count,
        nan=score_nan_count,
        inf_=score_inf_count,
        runtime=runtime,
        is_smoke=is_smoke,
    )
    _write_done(done_json, verdict, shard_id, summ)

    print("\n" + "=" * 70)
    print(f"[RUN-SHARD-{mode_str} {shard_id}] 완료 ({runtime:.1f}s)  판정: {verdict}")
    print("=" * 70)

    if verdict in ("FAIL", "SMOKE_FAIL"):
        sys.exit(1)


# =============================================================================
# main
# =============================================================================

def main() -> None:
    if len(sys.argv) < 2:
        print("[ABORT] bare run 차단.", file=sys.stderr)
        print("  dry-run:   --dry-run", file=sys.stderr)
        print("  smoke:     --run-shard --shard-id 0 --smoke-test "
              "--confirm-model-forward --confirm-stage2-holdout-eval-only", file=sys.stderr)
        print("  full shard: --run-shard --shard-id {0..7} "
              "--confirm-model-forward --confirm-stage2-holdout-eval-only", file=sys.stderr)
        sys.exit(2)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dry-run",                              action="store_true")
    parser.add_argument("--run-shard",                            action="store_true")
    parser.add_argument("--shard-id",     type=int,
                        choices=list(range(SHARD_COUNT)))
    parser.add_argument("--smoke-test",                           action="store_true")
    parser.add_argument("--confirm-model-forward",                action="store_true")
    parser.add_argument("--confirm-stage2-holdout-eval-only",     action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        run_dry()
        return

    if args.run_shard:
        if not (args.confirm_model_forward and args.confirm_stage2_holdout_eval_only):
            print(
                "[ABORT] --run-shard 실행 시 --confirm-model-forward 와 "
                "--confirm-stage2-holdout-eval-only 필요.",
                file=sys.stderr,
            )
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
