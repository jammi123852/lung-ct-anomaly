"""
RD4AD strict same-position z-track actual scoring RUN v1

T1_minrun2 survived=True candidate 92,342개 전체를 RD-D1s RD4AD 로 scoring 한다.
shard 단위로 처리한다 (8 shards, patient_id stable hash).

설계 원칙:
  - ALL survived candidates scored  (대표 1개만 금지)
  - track_id 는 manifest 에서 직접 읽음 (group rebuild / xy_radius grouping 없음)
  - first_stage_score 기준 candidate 삭제 금지
  - label 은 evaluation 에만 사용 (scoring selection 금지)
  - stage2_holdout 접근 금지
  - T2_minrun3 은 primary 사용 금지

실행 방식:
  bare run (인자 없음) : exit 2 차단
  dry-run   : python <script> --dry-run
  run-shard : python <script> --run-shard --shard-id {0..7}
                  --confirm-model-forward --confirm-stage1dev-only

참조 로직:
  모델 빌드 / crop / score / CTMmapCache / RoiRatioLookup 은
  experiments/rd4ad_z_continuity_group_full_scoring_v1/scripts/
      rd4ad_z_continuity_group_full_scoring_run.py  (scalar_repro_ok 재현 보장)
  와 동일 구조를 사용한다.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from collections import OrderedDict, defaultdict
from pathlib import Path

# =============================================================================
# 경로 상수
# =============================================================================

PROJECT_ROOT    = Path("/home/jinhy/project/lung-ct-anomaly")
EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "experiments/rd4ad_strict_same_position_ztrack_actual_scoring_v1"
)

# 입력 (read-only)
CANDIDATE_MANIFEST_CSV = (
    EXPERIMENT_ROOT / "manifests/strict_ztrack_scoring_candidate_manifest.csv"
)
SHARD_PLAN_CSV = (
    EXPERIMENT_ROOT / "manifests/strict_ztrack_scoring_shard_plan.csv"
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
RD_D1S_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1s_medi3ch_true_rd4ad_shard_run_v1"
    / "rd_d1s_stage1dev_candidate_score.csv"
)
EFFB0_SCORE_BASE = (
    PROJECT_ROOT
    / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs/scores"
)

# 출력 (shards/ 하위에만 허용)
SHARDS_DIR = EXPERIMENT_ROOT / "shards"

# 상수
SHARD_COUNT                = 8
CROP_SIZE                  = 96
HU_MIN, HU_MAX             = -160.0, 240.0
SCALAR_REPRO_OK_THRESHOLD  = 1e-4
EXPECTED_TOTAL_CANDIDATES  = 92342

# =============================================================================
# guardrail
# =============================================================================

GUARDRAILS = {
    "stage2_holdout_accessed":                       False,
    "checkpoint_loaded":                             False,
    "model_forward_executed":                        False,
    "training_executed":                             False,
    "backward_executed":                             False,
    "optimizer_created":                             False,
    "checkpoint_saved":                              False,
    "crop_generation_executed":                      False,
    "full_scoring_executed":                         False,
    "threshold_recalculated":                        False,
    "existing_artifact_modified":                    False,
    "existing_script_modified":                      False,
    "output_overwrite":                              False,
    "first_stage_score_used_for_candidate_deletion": False,
    "label_used_for_evaluation_only":                True,
    "label_used_for_scoring_selection":              False,
    "xy_radius_grouping_used":                       False,
    "representative_only_scoring_used":              False,
    "all_survived_track_candidates_scored":          False,
    "raw_rd4ad_primary_score":                       True,
    "adjusted_score_preview_only":                   True,
}

# =============================================================================
# 안전 경로 검사
# =============================================================================

_PROTECTED_INPUTS = [
    CANDIDATE_MANIFEST_CSV,
    SHARD_PLAN_CSV,
    RD_D1S_SCORE_CSV,
    CKPT_PATH,
    LOCAL_RESNET_WEIGHT,
]


def assert_path_safe(p: Path) -> None:
    s = str(p).lower()
    if "stage2_holdout" in s or ("stage2" in s and "holdout" in s):
        GUARDRAILS["stage2_holdout_accessed"] = True
        raise RuntimeError(f"[ABORT] stage2_holdout 경로 접근 차단: {p}")


def ensure_output_path_safe(p: Path) -> None:
    """쓰기 경로는 반드시 shards/ 하위여야 하며, 입력 파일 덮어쓰기 금지."""
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
        raise RuntimeError(f"[ABORT] run-shard 는 shards/ 외부 쓰기 금지: {p}")


# =============================================================================
# CSV 유틸
# =============================================================================

def read_csv(path: Path) -> list:
    assert_path_safe(path)
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
    """shard 별 errors.csv 에 append 하는 logger 반환."""
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
# shard 할당 (preflight 와 동일 hash)
# =============================================================================

def _patient_shard(patient_id: str, n_shards: int = SHARD_COUNT) -> int:
    return int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % n_shards


# =============================================================================
# roi_0_0_patch_ratio 조회 (group scoring run 과 동일 로직)
# =============================================================================

class RoiRatioLookup:
    def __init__(self):
        self._cache: dict = {}

    def _load_patient(self, patient_id: str, source_csv_rel: str) -> None:
        if patient_id in self._cache:
            return
        p = EFFB0_SCORE_BASE / source_csv_rel
        assert_path_safe(p)
        if not p.exists():
            self._cache[patient_id] = []
            return
        rows = []
        try:
            with open(str(p), encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    rows.append({
                        "local_z": int(row["local_z"]),
                        "y0":      int(row["y0"]),
                        "x0":      int(row["x0"]),
                        "y1":      int(row["y1"]),
                        "x1":      int(row["x1"]),
                        "roi_0_0_patch_ratio": float(
                            row.get("roi_0_0_patch_ratio", 0.0) or 0.0
                        ),
                    })
        except Exception:
            rows = []
        self._cache[patient_id] = rows

    def lookup(
        self,
        patient_id: str,
        source_csv_rel: str,
        local_z: int,
        crop_y0: int,
        crop_x0: int,
        crop_y1: int,
        crop_x1: int,
    ):
        """None 반환 시 lookup 실패."""
        self._load_patient(patient_id, source_csv_rel)
        patches = self._cache.get(patient_id, [])
        y_center = (crop_y0 + crop_y1) / 2.0
        x_center = (crop_x0 + crop_x1) / 2.0

        matched = []
        for p in patches:
            if p["local_z"] != local_z:
                continue
            py_c = (p["y0"] + p["y1"]) / 2.0
            px_c = (p["x0"] + p["x1"]) / 2.0
            if crop_y0 <= py_c < crop_y1 and crop_x0 <= px_c < crop_x1:
                matched.append(p["roi_0_0_patch_ratio"])

        if not matched:
            for p in patches:
                if p["local_z"] != local_z:
                    continue
                if p["y0"] <= y_center < p["y1"] and p["x0"] <= x_center < p["x1"]:
                    matched.append(p["roi_0_0_patch_ratio"])

        if not matched:
            return None
        return sum(matched) / len(matched)


# =============================================================================
# 모델 빌드 (group scoring run 과 동일)
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
    import torch
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

    assert_path_safe(CKPT_PATH)
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
# crop / score (group scoring run 과 동일)
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
    # return: scalar, layer1, layer2, layer3
    return scalar, scores[2], scores[1], scores[0]


# =============================================================================
# CT mmap 캐시 (group scoring run 과 동일)
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
        assert_path_safe(ct_path)
        if not ct_path.exists():
            raise FileNotFoundError(f"CT 없음: {ct_path}")
        arr = np.load(str(ct_path), mmap_mode="r")
        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        self._cache[safe_id] = arr
        return arr


# =============================================================================
# adjusted score (P1 / P2 preview only, per-candidate)
# =============================================================================

def compute_candidate_adjusted(rd4ad_raw: float, roi_ratio):
    """Returns (P1, P2) or (None, None) if roi_ratio unavailable."""
    if roi_ratio is None or not math.isfinite(roi_ratio):
        return None, None
    p1 = rd4ad_raw * roi_ratio
    p2 = rd4ad_raw * math.sqrt(max(roi_ratio, 0.0))
    return p1, p2


# =============================================================================
# shard CSV 필수 컬럼
# =============================================================================

SHARD_CSV_FIELDS = [
    "shard_id",
    "candidate_id",
    "patient_id",
    "safe_id",
    "track_id",
    "local_z",
    "crop_y0", "crop_x0", "crop_y1", "crop_x1",
    "pos_y0", "pos_x0", "pos_y1", "pos_x1",
    "label",
    "stage_split",
    "first_stage_score",
    "ztrack_min_run_len",
    "survived",
    "rd4ad_ztrack_score_raw",
    "score_layer1",
    "score_layer2",
    "score_layer3",
    "scalar_repro_diff",
    "roi_0_0_patch_ratio",
    "boundary_like_ratio",
    "P1_times_roi",
    "P2_times_sqrt_roi",
]


# =============================================================================
# shard 출력 helpers
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
    scalar_repro_diffs: list,
) -> tuple:
    ensure_output_path_safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if scalar_repro_diffs:
        repro_mean = sum(scalar_repro_diffs) / len(scalar_repro_diffs)
        repro_max  = max(scalar_repro_diffs)
        repro_ok   = bool(repro_mean < SCALAR_REPRO_OK_THRESHOLD)
    else:
        repro_mean = None
        repro_max  = None
        repro_ok   = True  # no reference pairs → skip check

    count_ok = (expected == scored + failed)
    hard_fail = (
        GUARDRAILS["stage2_holdout_accessed"]
        or GUARDRAILS["training_executed"]
        or GUARDRAILS["backward_executed"]
        or GUARDRAILS["optimizer_created"]
        or GUARDRAILS["checkpoint_saved"]
        or GUARDRAILS["existing_artifact_modified"]
        or not repro_ok
    )
    all_pass = (
        count_ok
        and failed == 0
        and errors == 0
        and nan == 0
        and inf_ == 0
        and repro_ok
        and not hard_fail
    )
    if all_pass:
        verdict = "PASS"
    elif count_ok and not hard_fail:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "FAIL"

    summary = {
        "shard_id":                             shard_id,
        "expected_candidate_count":             expected,
        "actual_scored_candidate_count":        scored,
        "failed_candidate_count":               failed,
        "error_count":                          errors,
        "score_nan_count":                      nan,
        "score_inf_count":                      inf_,
        "scalar_repro_ok":                      repro_ok,
        "scalar_repro_mean_abs_diff":           repro_mean,
        "scalar_repro_max_abs_diff":            repro_max,
        "runtime_sec":                          round(runtime, 1),
        "stage2_holdout_accessed":              GUARDRAILS["stage2_holdout_accessed"],
        "checkpoint_loaded":                    GUARDRAILS["checkpoint_loaded"],
        "model_forward_executed":               GUARDRAILS["model_forward_executed"],
        "training_executed":                    GUARDRAILS["training_executed"],
        "backward_executed":                    GUARDRAILS["backward_executed"],
        "optimizer_created":                    GUARDRAILS["optimizer_created"],
        "checkpoint_saved":                     GUARDRAILS["checkpoint_saved"],
        "crop_generation_executed":             GUARDRAILS["crop_generation_executed"],
        "existing_artifact_modified":           GUARDRAILS["existing_artifact_modified"],
        "output_overwrite":                     GUARDRAILS["output_overwrite"],
        "representative_only_scoring_used":     False,
        "all_survived_track_candidates_scored": (failed == 0),
        "raw_rd4ad_primary_score":              True,
        "adjusted_score_preview_only":          True,
        "verdict":                              verdict,
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
            "score_nan_count",
            "score_inf_count",
            "scalar_repro_ok",
            "stage2_holdout_accessed",
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
    print("[DRY-RUN] RD4AD strict z-track actual scoring RUN v1")
    print("  파일 생성 없음 / model forward 없음 / checkpoint load 없음")
    print("=" * 70)
    issues = []

    print("\n[1] 입력 파일 존재 확인 (read-only)")
    checks = {
        "candidate manifest": CANDIDATE_MANIFEST_CSV,
        "shard plan":         SHARD_PLAN_CSV,
        "checkpoint":         CKPT_PATH,
        "ResNet18 weight":    LOCAL_RESNET_WEIGHT,
        "CT root":            CT_ROOT,
        "RD-D1s score CSV":  RD_D1S_SCORE_CSV,
    }
    for name, p in checks.items():
        ok = p.exists()
        mark = "OK" if ok else "MISSING"
        print(f"  [{mark}] {name}: {p}")
        if not ok and name != "RD-D1s score CSV":
            issues.append(f"missing: {name}")
    if not RD_D1S_SCORE_CSV.exists():
        print("  [INFO] RD-D1s score CSV 없음 → scalar_repro_diff 생략됨")

    print("\n[2] shard plan expected candidate count")
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
            issues.append(
                f"shard plan 합계 {total_expected} != {EXPECTED_TOTAL_CANDIDATES}"
            )
    except Exception as e:
        issues.append(f"shard plan 읽기 실패: {e}")

    print("\n[3] output overwrite 위험 확인")
    for sid in range(SHARD_COUNT):
        out_csv = SHARDS_DIR / f"shard_{sid}" / f"strict_ztrack_scores_shard_{sid}.csv"
        done_p  = SHARDS_DIR / f"shard_{sid}" / "DONE.json"
        if out_csv.exists() or done_p.exists():
            print(f"  [WARN] shard {sid}: 출력 이미 존재 (재실행 시 overwrite)")
        else:
            print(f"  [OK]   shard {sid}: 출력 없음")

    print("\n[4] stage2_holdout 접근 없음 확인")
    print(f"  stage2_holdout_accessed: {GUARDRAILS['stage2_holdout_accessed']}")

    print("\n[5] guardrail snapshot (dry-run)")
    for k in [
        "checkpoint_loaded", "model_forward_executed", "crop_generation_executed",
        "training_executed", "backward_executed", "optimizer_created",
        "checkpoint_saved", "existing_artifact_modified", "output_overwrite",
        "representative_only_scoring_used", "label_used_for_scoring_selection",
        "xy_radius_grouping_used", "first_stage_score_used_for_candidate_deletion",
    ]:
        print(f"  {k}: {GUARDRAILS[k]}")

    print(f"\n  expected shard 0 candidate count: "
          f"{shard_plan_data.get(0, 'N/A')}")

    print("\n" + "=" * 70)
    if issues:
        print("[DRY-RUN 결과] 이슈:")
        for it in issues:
            print(f"  - {it}")
        print("판정: NEEDS_FIX")
        sys.exit(1)
    else:
        print("[DRY-RUN 결과] 모든 입력/계획/경로 OK.")
        print("판정: READY_TO_RUN_SHARD0")
    print("=" * 70)


# =============================================================================
# run-shard
# =============================================================================

def run_shard(shard_id: int) -> None:
    print("=" * 70)
    print(f"[RUN-SHARD] RD4AD strict z-track actual scoring — shard {shard_id}")
    print("=" * 70)
    t0 = time.perf_counter()

    shard_dir    = SHARDS_DIR / f"shard_{shard_id}"
    out_csv      = shard_dir / f"strict_ztrack_scores_shard_{shard_id}.csv"
    summary_json = shard_dir / f"shard_{shard_id}_summary.json"
    error_csv    = shard_dir / "errors.csv"
    done_json    = shard_dir / "DONE.json"

    output_existed = out_csv.exists() or done_json.exists()
    GUARDRAILS["output_overwrite"] = bool(output_existed)
    if output_existed:
        print(
            f"  [WARN] shard {shard_id} 출력 이미 존재 → overwrite "
            f"(output_overwrite=True)"
        )

    shard_dir.mkdir(parents=True, exist_ok=True)
    append_error = make_error_logger(error_csv)

    # ── 1. shard plan 로드 ───────────────────────────────────────────────────
    print("\n[1] shard plan 로드")
    shard_plan_data: dict = {}
    for row in read_csv(SHARD_PLAN_CSV):
        sid = int(row["shard_id"])
        shard_plan_data[sid] = int(row["candidate_count"])
    expected_candidate_count = shard_plan_data.get(shard_id, -1)
    print(f"  shard {shard_id} expected candidates: {expected_candidate_count:,}")

    # ── 2. candidate manifest 로드 및 shard 필터 ────────────────────────────
    print("\n[2] candidate manifest 로드 및 shard 필터링")
    all_rows  = read_csv(CANDIDATE_MANIFEST_CSV)
    shard_rows = [
        r for r in all_rows
        if _patient_shard(r["patient_id"]) == shard_id
        and r.get("survived", "").strip().lower() in ("true", "1")
    ]
    print(
        f"  전체 rows: {len(all_rows):,}  "
        f"shard {shard_id} survived: {len(shard_rows):,}"
    )
    if len(shard_rows) != expected_candidate_count:
        msg = (
            f"shard {shard_id} 후보수 불일치: "
            f"manifest_filtered={len(shard_rows)} "
            f"plan_expected={expected_candidate_count}"
        )
        append_error(msg)
        print(f"  [WARN] {msg}")

    if not shard_rows:
        print(f"  [ABORT] shard {shard_id} 에 candidate 없음")
        sys.exit(2)

    # ── 3. RD-D1s scalar score 로드 (scalar_repro_diff 용) ──────────────────
    print("\n[3] RD-D1s scalar score CSV 로드")
    scalar_score_map: dict = {}
    if RD_D1S_SCORE_CSV.exists():
        for r in read_csv(RD_D1S_SCORE_CSV):
            if r.get("stage_split", "") == "stage1_dev":
                try:
                    scalar_score_map[r["candidate_id"]] = float(
                        r["rd_d1s_medi3ch_rd4ad_score"]
                    )
                except Exception:
                    pass
        print(f"  scalar score map: {len(scalar_score_map):,} entries")
    else:
        print("  [WARN] scalar score CSV 없음 → scalar_repro_diff 생략")

    # ── 4. roi lookup 초기화 ─────────────────────────────────────────────────
    roi_lookup = RoiRatioLookup()

    # ── 5. 모델 로드 ──────────────────────────────────────────────────────────
    print("\n[5] 모델 로드 (checkpoint read-only)")
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

    ct_cache = CTMmapCache(max_size=12)

    # ── 6. forward scoring ───────────────────────────────────────────────────
    print(
        f"\n[6] forward scoring — {len(shard_rows):,} candidates (shard {shard_id})"
    )
    GUARDRAILS["model_forward_executed"]   = True
    GUARDRAILS["crop_generation_executed"] = True
    GUARDRAILS["full_scoring_executed"]    = True

    score_rows:         list  = []
    scalar_repro_diffs: list  = []
    error_count:        int   = 0
    failed_count:       int   = 0
    score_nan_count:    int   = 0
    score_inf_count:    int   = 0

    log_interval = max(1, len(shard_rows) // 20)

    for idx, row in enumerate(shard_rows):
        if idx % log_interval == 0:
            elapsed_so_far = time.perf_counter() - t0
            print(
                f"  [{idx:6d}/{len(shard_rows)}] "
                f"elapsed={elapsed_so_far:.0f}s  "
                f"failed={failed_count}"
            )

        cid     = row["candidate_id"]
        safe_id = row["safe_id"]
        try:
            local_z = int(row["local_z"])
            crop_y0 = int(row["crop_y0"])
            crop_x0 = int(row["crop_x0"])
            crop_y1 = int(row["crop_y1"])
            crop_x1 = int(row["crop_x1"])
        except Exception as e:
            append_error(f"coord parse fail: {cid}: {e}", e)
            error_count  += 1
            failed_count += 1
            continue

        # roi lookup (optional)
        roi_ratio = None
        src_csv   = row.get("source_score_csv", "")
        if src_csv:
            try:
                roi_ratio = roi_lookup.lookup(
                    row["patient_id"], src_csv,
                    local_z, crop_y0, crop_x0, crop_y1, crop_x1,
                )
            except Exception:
                roi_ratio = None

        boundary_like = (1.0 - roi_ratio) if roi_ratio is not None else None

        # CT load
        try:
            ct_arr = ct_cache.get(safe_id)
        except Exception as e:
            append_error(f"CT load fail: {safe_id} (cid={cid}): {e}", e)
            error_count  += 1
            failed_count += 1
            continue

        # crop
        try:
            crop = build_medi3ch_crop(
                ct_arr, local_z, crop_y0, crop_x0, crop_y1, crop_x1
            )
        except Exception as e:
            append_error(f"crop build fail: {cid}: {e}", e)
            error_count  += 1
            failed_count += 1
            continue

        # forward
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

        # scalar_repro_diff
        scalar_diff = None
        if cid in scalar_score_map:
            scalar_diff = abs(rd4ad_raw - scalar_score_map[cid])
            scalar_repro_diffs.append(scalar_diff)

        # adjusted preview
        p1, p2 = compute_candidate_adjusted(rd4ad_raw, roi_ratio)

        score_rows.append({
            "shard_id":              shard_id,
            "candidate_id":          cid,
            "patient_id":            row["patient_id"],
            "safe_id":               safe_id,
            "track_id":              row.get("track_id", ""),
            "local_z":               row["local_z"],
            "crop_y0":               row["crop_y0"],
            "crop_x0":               row["crop_x0"],
            "crop_y1":               row["crop_y1"],
            "crop_x1":               row["crop_x1"],
            "pos_y0":                row.get("pos_y0", ""),
            "pos_x0":                row.get("pos_x0", ""),
            "pos_y1":                row.get("pos_y1", ""),
            "pos_x1":                row.get("pos_x1", ""),
            "label":                 row.get("label", ""),
            "stage_split":           row.get("stage_split", ""),
            "first_stage_score":     row.get("first_stage_score", ""),
            "ztrack_min_run_len":    row.get("ztrack_min_run_len", ""),
            "survived":              row.get("survived", ""),
            "rd4ad_ztrack_score_raw": rd4ad_raw,
            "score_layer1":          l1,
            "score_layer2":          l2,
            "score_layer3":          l3,
            "scalar_repro_diff":     "" if scalar_diff is None else scalar_diff,
            "roi_0_0_patch_ratio":   "" if roi_ratio is None else roi_ratio,
            "boundary_like_ratio":   "" if boundary_like is None else boundary_like,
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

    # ── 7. CSV 저장 ───────────────────────────────────────────────────────────
    print(f"\n[7] shard CSV 저장")
    if score_rows:
        write_csv(out_csv, SHARD_CSV_FIELDS, score_rows)

    # ── 8. summary / DONE 저장 ───────────────────────────────────────────────
    print(f"\n[8] summary / DONE 저장")
    verdict, summ = _write_shard_summary(
        summary_json,
        shard_id,
        expected=expected_candidate_count,
        scored=actual_scored,
        failed=failed_count,
        errors=error_count,
        nan=score_nan_count,
        inf_=score_inf_count,
        runtime=runtime,
        scalar_repro_diffs=scalar_repro_diffs,
    )
    _write_done(done_json, verdict, shard_id, summ)

    print("\n" + "=" * 70)
    print(f"[RUN-SHARD {shard_id}] 완료 ({runtime:.1f}s)  판정: {verdict}")
    print("=" * 70)
    if verdict == "FAIL":
        sys.exit(1)


# =============================================================================
# main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RD4AD strict z-track actual scoring RUN v1"
    )
    parser.add_argument("--dry-run",                action="store_true")
    parser.add_argument("--run-shard",              action="store_true")
    parser.add_argument(
        "--shard-id", type=int,
        choices=list(range(SHARD_COUNT)),
    )
    parser.add_argument("--confirm-model-forward",  action="store_true")
    parser.add_argument("--confirm-stage1dev-only", action="store_true")
    args = parser.parse_args()

    if not any([args.dry_run, args.run_shard]):
        print(
            "[ABORT] bare run 차단. --dry-run 또는 --run-shard 를 사용하세요.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.dry_run:
        run_dry()
        return

    if args.run_shard:
        if args.shard_id is None:
            print("[ABORT] --shard-id 필요", file=sys.stderr)
            sys.exit(2)
        if not args.confirm_model_forward:
            print("[ABORT] --confirm-model-forward 필요", file=sys.stderr)
            sys.exit(2)
        if not args.confirm_stage1dev_only:
            print("[ABORT] --confirm-stage1dev-only 필요", file=sys.stderr)
            sys.exit(2)
        run_shard(args.shard_id)
        return


if __name__ == "__main__":
    main()
