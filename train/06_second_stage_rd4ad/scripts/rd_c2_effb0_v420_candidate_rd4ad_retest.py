#!/usr/bin/env python3
"""
RD-C2: EfficientNet-B0 v4_20 candidate 기반 RD4AD retest
목적:
  RD-B10/RD-B11은 ResNet18 roi_0_0 기반 후보(stage1_dev_fixed96_thr001_v1)를 사용했다.
  이번 RD-C2는 EfficientNet-B0 + refined_roi_v4_20_modeB 기반 후보(p_c3 manifest)에
  대해 동일한 RD-B8f checkpoint + RD-B9 threshold 로 RD4AD score를 붙이고
  suppression safety를 분석한다.

안전 조건:
  stage2_holdout 접근 금지
  threshold 재계산 금지
  RD4AD/EfficientNet 재학습 금지
  backward / optimizer / checkpoint 저장 금지
  기존 first-stage score 수정 금지
  기존 파일 삭제 금지
  output root 존재 시 즉시 중단

모드:
  bare run         -> exit 2
  --dry-plan       -> source audit + 경로 확인 (파일 생성 없음)
  --run-retest     -> manifest 생성 + scoring + safety analysis (DONE 생성)
"""

import sys
import csv
import json
import math
import time
import collections
from pathlib import Path

# ── 모드 판별 ──────────────────────────────────────────────────────────────────
ALLOWED_MODES = {"--dry-plan", "--run-retest", "--profile-optimized"}
if not any(m in sys.argv for m in ALLOWED_MODES):
    print("오류: 실행 모드 지정 필요.")
    print("  --dry-plan          : source audit + 입력 경로 확인 (파일 생성 없음)")
    print("  --profile-optimized : 2,000행 subset으로 병목/동일성 확인")
    print("  --run-retest        : candidate manifest 생성 + RD4AD scoring + safety analysis")
    sys.exit(2)

IS_DRY_PLAN    = "--dry-plan"          in sys.argv
IS_RUN_RETEST  = "--run-retest"        in sys.argv
IS_PROFILE     = "--profile-optimized" in sys.argv

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c2_effb0_v420_candidate_rd4ad_retest_v1"
)
OUTPUT_ROOT_PROFILE = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c2_effb0_v420_candidate_rd4ad_retest_profile_v1"
)

# EfficientNet-B0 v4_20 branch source
P_C3_MANIFEST = (
    PROJECT_ROOT
    / "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1"
    / "outputs/candidates/p_c3_candidate_manifest"
    / "p_c3_candidate_manifest.csv"
)

EFFB0_V420_THRESHOLD_JSON = (
    PROJECT_ROOT
    / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs/evaluation/normal_val_thresholds"
    / "normal_val_threshold.json"
)

EFFB0_V420_SCORE_DIR = (
    PROJECT_ROOT
    / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs/scores/lesion_stage1_dev_by_patient"
)

# stage split
STAGE_SPLIT_CSV = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"
)

# RD-B9 threshold (RD4AD second-stage)
THRESHOLD_DIR = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b9_normal_val_scoring_threshold_v1"
)
THRESHOLD_SUMMARY_JSON = THRESHOLD_DIR / "rd_b9_normal_val_threshold_summary.json"
THRESHOLD_CANDIDATES_CSV = THRESHOLD_DIR / "rd_b9_normal_val_threshold_candidates.csv"

# RD-B8f checkpoint (read-only)
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs/models/rd_b8f_true_rd4ad_resnet18_mixed3ch_6bin_shard_v1"
    / "checkpoints/best_train_loss.pth"
)
LOCAL_RESNET18_WEIGHT = Path(
    "/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)

# CT / ROI root
CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)
V4_20_ROI_ROOT = (
    PROJECT_ROOT
    / "outputs/mip-postprocess-research-v1/masks"
    / "refined_roi_v4_20_modeB_all_v1/lesion"
)

# old source 비교용
OLD_SOURCE_SAFETY_JSON = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b11_rd4ad_fp_suppression_safety_analysis_v1"
    / "rd_b11_rd4ad_fp_suppression_safety_summary.json"
)

# ── ABORT 상수 (강화 안전 조건) ──────────────────────────────────────────────────
EXPECTED_INPUT_ROWS           = 114381
EXPECTED_HOLDOUT_PATIENTS     = {"LUNG1-295", "LUNG1-415"}
EXPECTED_HOLDOUT_ROWS_REMOVED = 934
EXPECTED_SCORING_ROWS         = 113447

# ── 설계 상수 ──────────────────────────────────────────────────────────────────
CROP_SIZE              = 96
CROP_HALF              = CROP_SIZE // 2      # 48
EROSION_PX             = 5
BOUNDARY_THRESHOLD     = 0.05
INTERIOR_ROI_MIN       = 0.85
MIP_RADIUS             = 3
HU_CLIP_MIN            = -1000.0
HU_CLIP_MAX            =  600.0
HU_RANGE               = 1600.0
LOW_Z_WARNING_THRESHOLD = 7
Z_LOWER_MAX            = 1.0 / 3.0
Z_MIDDLE_MAX           = 2.0 / 3.0
SCORE_BATCH_SIZE       = 48
SIX_BIN_LABELS = [
    "upper_boundary", "upper_interior",
    "middle_boundary", "middle_interior",
    "lower_boundary", "lower_interior",
]

PROFILE_ROWS = 2000

FORBIDDEN_PATH_KEYWORDS = ["stage2_holdout", "lesion_mask"]

# ── 안전 체크 ──────────────────────────────────────────────────────────────────

def assert_path_safe(path_str):
    for kw in FORBIDDEN_PATH_KEYWORDS:
        if kw.lower() in str(path_str).lower():
            raise RuntimeError(
                f"[SAFETY] 금지 경로 접근 차단: {path_str!r} (keyword={kw!r})"
            )


# ── CSV / JSON 헬퍼 ────────────────────────────────────────────────────────────

def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  -> {path.name}")


def load_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  -> {path.name}")


# ── HU 변환 / crop ────────────────────────────────────────────────────────────

def normalize_hu(hu_array):
    import numpy as np
    clipped = hu_array.clip(HU_CLIP_MIN, HU_CLIP_MAX)
    return ((clipped - HU_CLIP_MIN) / HU_RANGE).astype("float32")


def compute_mip_slab_indices(center_z, direction, z_max):
    if direction == "lower":
        raw = [center_z - MIP_RADIUS + i for i in range(MIP_RADIUS)]
    else:
        raw = [center_z + 1 + i for i in range(MIP_RADIUS)]
    return [max(0, min(idx, z_max - 1)) for idx in raw]


def build_crop_np(ct_arr, center_z, crop_y0, crop_x0, crop_y1, crop_x1):
    import numpy as np
    TARGET = CROP_SIZE
    z_max, h_max, w_max = ct_arr.shape

    def _crop2d(img2d, y0, x0, y1, x1):
        h, w = img2d.shape
        out = np.full((TARGET, TARGET), HU_CLIP_MIN, dtype=img2d.dtype)
        sy0 = max(0, y0); sx0 = max(0, x0)
        sy1 = min(h, y1); sx1 = min(w, x1)
        if sy1 <= sy0 or sx1 <= sx0:
            return out
        dy0 = sy0 - y0; dx0 = sx0 - x0
        out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img2d[sy0:sy1, sx0:sx1]
        return out

    ch0 = _crop2d(ct_arr[center_z], crop_y0, crop_x0, crop_y1, crop_x1)
    lower_idx = compute_mip_slab_indices(center_z, "lower", z_max)
    ch1 = np.max(np.stack(
        [_crop2d(ct_arr[z], crop_y0, crop_x0, crop_y1, crop_x1) for z in lower_idx], axis=0
    ), axis=0)
    upper_idx = compute_mip_slab_indices(center_z, "upper", z_max)
    ch2 = np.max(np.stack(
        [_crop2d(ct_arr[z], crop_y0, crop_x0, crop_y1, crop_x1) for z in upper_idx], axis=0
    ), axis=0)

    crop = np.stack(
        [normalize_hu(ch0), normalize_hu(ch1), normalize_hu(ch2)], axis=0
    ).astype("float32")
    if crop.shape != (3, TARGET, TARGET):
        raise RuntimeError(f"bad crop shape: {crop.shape}")
    return crop


# ── six_bin 계산 ──────────────────────────────────────────────────────────────

def z_level_from_ratio(z_ratio):
    if z_ratio < Z_LOWER_MAX:
        return "lower"
    elif z_ratio < Z_MIDDLE_MAX:
        return "middle"
    return "upper"


def compute_sixbin_label(roi_arr, local_z, crop_y0, crop_x0, crop_y1, crop_x1, z_ratio):
    from scipy.ndimage import distance_transform_edt
    import numpy as np
    n_slices = roi_arr.shape[0]
    if not (0 <= local_z < n_slices):
        return "excluded", "excluded", "excluded"
    full_roi_slice = roi_arr[local_z]
    sy0 = max(0, crop_y0); sx0 = max(0, crop_x0)
    sy1 = min(roi_arr.shape[1], crop_y1); sx1 = min(roi_arr.shape[2], crop_x1)
    dist_full = distance_transform_edt(full_roi_slice)
    ring_full = ((full_roi_slice > 0) & (dist_full <= EROSION_PX)).astype(np.float32)
    roi_patch_sum  = float(full_roi_slice[sy0:sy1, sx0:sx1].sum())
    ring_patch_sum = float(ring_full[sy0:sy1, sx0:sx1].sum())
    patch_area = float(CROP_SIZE * CROP_SIZE)
    roi_ratio   = roi_patch_sum  / patch_area
    boundary_ratio = ring_patch_sum / patch_area
    is_boundary = boundary_ratio >= BOUNDARY_THRESHOLD
    is_interior = (roi_ratio >= INTERIOR_ROI_MIN) and (not is_boundary)
    if is_boundary:
        bs = "boundary"
    elif is_interior:
        bs = "interior"
    else:
        bs = "excluded"
    z_level = z_level_from_ratio(z_ratio)
    six_bin = "excluded" if bs == "excluded" else f"{z_level}_{bs}"
    return z_level, bs, six_bin


# ── LRU 캐시 ──────────────────────────────────────────────────────────────────

class LRUPatientCache:
    def __init__(self, max_size=6):
        self._cache = collections.OrderedDict()
        self._max = max_size

    def load(self, key, path):
        import numpy as np
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        assert_path_safe(path)
        arr = np.load(str(path), mmap_mode="r")
        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        self._cache[key] = arr
        return arr


class SixBinCache:
    """(safe_id, local_z) 기준으로 ROI slice EDT 결과를 캐싱."""
    def __init__(self, max_slices=512):
        self._cache = collections.OrderedDict()
        self._max = max_slices
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, roi_arr, safe_id, local_z):
        from scipy.ndimage import distance_transform_edt
        import numpy as np
        key = (safe_id, local_z)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        n_slices = roi_arr.shape[0]
        if not (0 <= local_z < n_slices):
            entry = {"valid": False, "n_slices": n_slices}
            if len(self._cache) >= self._max:
                self._cache.popitem(last=False)
            self._cache[key] = entry
            return entry
        full_roi_slice = roi_arr[local_z]
        dist_full = distance_transform_edt(full_roi_slice)
        ring_full = ((full_roi_slice > 0) & (dist_full <= EROSION_PX)).astype(np.float32)
        z_ratio = local_z / max(n_slices - 1, 1)
        z_level = z_level_from_ratio(z_ratio)
        entry = {
            "valid":          True,
            "full_roi_slice": full_roi_slice,
            "ring_full":      ring_full,
            "n_slices":       n_slices,
            "z_ratio":        z_ratio,
            "z_level":        z_level,
        }
        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        self._cache[key] = entry
        return entry

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def unique_count(self):
        return self.misses


def compute_sixbin_label_cached(sixbin_cache, roi_arr, safe_id, local_z,
                                 crop_y0, crop_x0, crop_y1, crop_x1):
    """SixBinCache를 사용해 EDT 재계산 없이 sixbin label 반환.
    결과는 compute_sixbin_label()과 동일해야 한다."""
    import numpy as np
    entry = sixbin_cache.get_or_compute(roi_arr, safe_id, local_z)
    if not entry.get("valid", False):
        return "excluded", "excluded", "excluded"
    full_roi_slice = entry["full_roi_slice"]
    ring_full      = entry["ring_full"]
    z_level        = entry["z_level"]
    sy0 = max(0, crop_y0); sx0 = max(0, crop_x0)
    sy1 = min(roi_arr.shape[1], crop_y1); sx1 = min(roi_arr.shape[2], crop_x1)
    roi_patch_sum  = float(full_roi_slice[sy0:sy1, sx0:sx1].sum())
    ring_patch_sum = float(ring_full[sy0:sy1, sx0:sx1].sum())
    patch_area     = float(CROP_SIZE * CROP_SIZE)
    roi_ratio      = roi_patch_sum  / patch_area
    boundary_ratio = ring_patch_sum / patch_area
    is_boundary = boundary_ratio >= BOUNDARY_THRESHOLD
    is_interior = (roi_ratio >= INTERIOR_ROI_MIN) and (not is_boundary)
    if is_boundary:
        bs = "boundary"
    elif is_interior:
        bs = "interior"
    else:
        bs = "excluded"
    six_bin = "excluded" if bs == "excluded" else f"{z_level}_{bs}"
    return z_level, bs, six_bin


# ── Teacher / Student (RD-B9/B10 동일 구조) ──────────────────────────────────

def build_teacher(local_weight_path):
    import torch
    import torchvision.models as models
    resnet = models.resnet18(weights=None)
    state_dict = torch.load(str(local_weight_path), map_location="cpu", weights_only=True)
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
                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            )
            self.de_layer2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            )
            self.de_layer1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )

        def forward(self, layer3_feat):
            x = self.de_layer3(layer3_feat);  de3 = x
            x = self.de_layer2(x);           de2 = x
            x = self.de_layer1(x);           de1 = x
            return de3, de2, de1

    return StudentDecoder()


# ── stage split 로드 ──────────────────────────────────────────────────────────

def load_stage_split(stage_split_csv):
    stage1_dev_ids = set()
    holdout_ids    = set()
    with open(stage_split_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sp  = row.get("stage_split", "")
            pid = row.get("patient_id", "")
            if sp == "stage1_dev":
                stage1_dev_ids.add(pid)
            elif sp == "stage2_holdout":
                holdout_ids.add(pid)
    return stage1_dev_ids, holdout_ids


# ── threshold 로드 (RD-B9) ────────────────────────────────────────────────────

def load_rd_b9_thresholds():
    with open(THRESHOLD_SUMMARY_JSON, encoding="utf-8") as f:
        summary = json.load(f)
    return {
        "global_p95":    float(summary["global_p95"]),
        "global_p99":    float(summary["global_p99"]),
        "bin_thresholds": summary.get("bin_thresholds", {}),
        "source":        summary.get("threshold_created_from", "rd_b9_normal_val_only"),
    }


# ── fixed96 crop 좌표 계산 ────────────────────────────────────────────────────

def derive_fixed96_coords(y0, x0, y1, x1):
    center_y = (y0 + y1) // 2
    center_x = (x0 + x1) // 2
    crop_y0  = center_y - CROP_HALF
    crop_y1  = center_y + CROP_HALF
    crop_x0  = center_x - CROP_HALF
    crop_x1  = center_x + CROP_HALF
    return crop_y0, crop_x0, crop_y1, crop_x1


# ── candidate source inventory ───────────────────────────────────────────────

def build_source_inventory(holdout_ids):
    sources = [
        {
            "source_name":  "p_c3_effb0_v420",
            "path":         str(P_C3_MANIFEST),
            "description":  "EfficientNet-B0 v4_20 ROI PaDiM branch p_c3 manifest",
        },
        {
            "source_name":  "effb0_v420_score_dir",
            "path":         str(EFFB0_V420_SCORE_DIR),
            "description":  "EfficientNet-B0 v4_20 stage1_dev score CSV per-patient dir",
        },
        {
            "source_name":  "effb0_v420_threshold_json",
            "path":         str(EFFB0_V420_THRESHOLD_JSON),
            "description":  "EfficientNet-B0 v4_20 normal_val threshold JSON (P-B9)",
        },
    ]
    rows = []
    for s in sources:
        p = Path(s["path"])
        exists = p.exists()
        rows_count = 0
        unique_pts = 0
        score_col  = ""
        has_holdout = 0
        if exists and p.suffix == ".csv":
            try:
                with open(p, newline="", encoding="utf-8-sig") as f:
                    dr = csv.DictReader(f)
                    fnames = set(dr.fieldnames or [])
                    pts = set()
                    for row in dr:
                        rows_count += 1
                        pid = row.get("patient_id", "")
                        pts.add(pid)
                        if pid in holdout_ids:
                            has_holdout += 1
                score_cols = [c for c in fnames if "score" in c.lower() or "padim" in c.lower()]
                score_col  = "|".join(score_cols[:3])
                unique_pts = len(pts)
            except Exception:
                pass
        elif exists and p.is_dir():
            csv_files  = list(p.glob("*.csv"))
            rows_count = len(csv_files)
            score_col  = "padim_score_per_patient_csv"
        rows.append({
            "source_name":   s["source_name"],
            "path":          s["path"],
            "exists":        int(exists),
            "description":   s["description"],
            "rows_or_files": rows_count,
            "unique_patients": unique_pts,
            "score_column":  score_col,
            "holdout_intersection": has_holdout,
        })
    return rows


# ── dry-plan ──────────────────────────────────────────────────────────────────

def run_dry_plan():
    errors  = []
    checks  = []

    def chk(label, ok, detail=""):
        status = "OK" if ok else "FAIL"
        checks.append({"label": label, "status": status, "detail": str(detail)})
        if not ok:
            errors.append(f"{label}: {detail}")

    print("=" * 70)
    print("RD-C2: EfficientNet-B0 v4_20 candidate RD4AD retest [DRY-PLAN]")
    print("=" * 70)

    # 1. stage split 로드
    print("\n[1/6] stage split 확인")
    chk("stage_split_csv", STAGE_SPLIT_CSV.exists(), str(STAGE_SPLIT_CSV))
    stage1_dev_ids, holdout_ids = set(), set()
    if STAGE_SPLIT_CSV.exists():
        stage1_dev_ids, holdout_ids = load_stage_split(STAGE_SPLIT_CSV)
        chk("stage1_dev_patients", len(stage1_dev_ids) > 0, f"{len(stage1_dev_ids)}")
        chk("holdout_patients",    len(holdout_ids)    > 0, f"{len(holdout_ids)}")
    print(f"  stage1_dev       : {len(stage1_dev_ids)} patients")
    print(f"  stage2_holdout   : {len(holdout_ids)} patients")

    # 2. candidate source inventory
    print("\n[2/6] candidate source inventory")
    inv = build_source_inventory(holdout_ids)
    for r in inv:
        print(f"  {r['source_name']:<30} exists={r['exists']}  "
              f"rows={r['rows_or_files']}  holdout={r['holdout_intersection']}")

    p_c3_row = next((r for r in inv if r["source_name"] == "p_c3_effb0_v420"), None)
    if p_c3_row and p_c3_row["exists"]:
        candidate_verdict = "USE_EXISTING_MANIFEST"
        print(f"\n  [판정] {candidate_verdict}")
        print(f"  p_c3 manifest 존재 → fixed96 coords 파생 후 rd_c2 manifest 생성")
    else:
        candidate_verdict = "BLOCKED_NO_P_C3_MANIFEST"
        errors.append("p_c3 manifest 없음")
        print(f"\n  [판정] {candidate_verdict}")

    chk("p_c3_manifest_exists", P_C3_MANIFEST.exists(), str(P_C3_MANIFEST))
    chk("effb0_threshold_json", EFFB0_V420_THRESHOLD_JSON.exists(), str(EFFB0_V420_THRESHOLD_JSON))
    chk("effb0_score_dir_exists", EFFB0_V420_SCORE_DIR.exists(), str(EFFB0_V420_SCORE_DIR))

    # 3. p_c3 manifest 상세
    print("\n[3/6] p_c3 manifest 상세 확인")
    if P_C3_MANIFEST.exists():
        rows_c3 = load_csv_rows(P_C3_MANIFEST)
        pts_c3  = set(r["patient_id"] for r in rows_c3)
        lbls    = collections.Counter(r.get("candidate_label", "") for r in rows_c3)
        splits  = collections.Counter(r.get("split", "") for r in rows_c3)
        h_int   = pts_c3 & holdout_ids
        # holdout intersection: WARNING (run-retest에서 denylist로 자동 제거)
        if h_int:
            print(f"  [WARNING] holdout patients in p_c3: {sorted(h_int)} "
                  f"(total rows={sum(1 for r in rows_c3 if r.get('patient_id','') in h_int)})"
                  f" -> run-retest 에서 denylist 자동 제거 예정")
        else:
            print(f"  [OK] holdout intersection = 0")
        chk("p_c3_stage1_dev",   splits.get("stage1_dev", 0) == len(rows_c3),
            f"stage1_dev={splits.get('stage1_dev',0)} / total={len(rows_c3)}")
        print(f"  total rows       : {len(rows_c3)}")
        print(f"  split            : {dict(splits)}")
        print(f"  candidate_label  : {dict(lbls)}")
        print(f"  unique patients  : {len(pts_c3)}")
        print(f"  holdout intersect: {len(h_int)}")
        # first stage threshold
        th_vals = set(r.get("threshold_p95","") for r in rows_c3[:10])
        print(f"  threshold_p95 sample: {th_vals}")

        # fixed96 sample
        r0 = rows_c3[0]
        y0, x0 = int(r0["y0"]), int(r0["x0"])
        y1, x1 = int(r0["y1"]), int(r0["x1"])
        cy0, cx0, cy1, cx1 = derive_fixed96_coords(y0, x0, y1, x1)
        print(f"  fixed96 sample : patch({y0},{x0},{y1},{x1}) "
              f"-> crop({cy0},{cx0},{cy1},{cx1})")

    # 4. EfficientNet-B0 v4_20 threshold
    print("\n[4/6] EfficientNet-B0 v4_20 threshold 확인")
    if EFFB0_V420_THRESHOLD_JSON.exists():
        with open(EFFB0_V420_THRESHOLD_JSON, encoding="utf-8") as f:
            th_info = json.load(f)
        p95 = th_info.get("threshold_p95", "N/A")
        p99 = th_info.get("threshold_p99", "N/A")
        branch = th_info.get("branch", "N/A")
        roi_src = th_info.get("official_roi_source", "N/A")
        print(f"  branch          : {branch}")
        print(f"  roi_source      : {roi_src}")
        print(f"  threshold_p95   : {p95}")
        print(f"  threshold_p99   : {p99}")
        chk("effb0_threshold_p95_correct",
            abs(float(p95) - 13.231265) < 1e-3, f"p95={p95}")
        chk("effb0_roi_is_v4_20",
            "v4_20" in str(roi_src).lower(), f"roi_source={roi_src}")

    # 5. RD-B9 threshold / checkpoint 확인
    print("\n[5/6] RD-B9 threshold / RD-B8f checkpoint 확인")
    chk("rd_b9_threshold_json", THRESHOLD_SUMMARY_JSON.exists(), str(THRESHOLD_SUMMARY_JSON))
    chk("rd_b8f_checkpoint",    CHECKPOINT_PATH.exists(),        str(CHECKPOINT_PATH))
    chk("resnet18_weight",      LOCAL_RESNET18_WEIGHT.exists(),  str(LOCAL_RESNET18_WEIGHT))
    if THRESHOLD_SUMMARY_JSON.exists():
        th = load_rd_b9_thresholds()
        print(f"  RD-B9 global p95 : {th['global_p95']}")
        print(f"  RD-B9 global p99 : {th['global_p99']}")

    # 6. CT / ROI 경로 확인 (상위 5명)
    print("\n[6/6] CT / ROI 경로 샘플 확인 (상위 5명)")
    ct_fail = roi_fail = 0
    if P_C3_MANIFEST.exists():
        rows_c3_loaded = load_csv_rows(P_C3_MANIFEST)
        seen_ids = []
        seen_set = set()
        for r in rows_c3_loaded:
            sid = r.get("safe_id", "")
            if sid and sid not in seen_set:
                seen_ids.append(sid)
                seen_set.add(sid)
            if len(seen_ids) >= 5:
                break
        for sid in seen_ids:
            ct_path  = CT_ROOT / sid / "ct_hu.npy"
            roi_path = V4_20_ROI_ROOT / sid / "refined_roi.npy"
            ct_ok  = ct_path.exists()
            roi_ok = roi_path.exists()
            if not ct_ok:  ct_fail  += 1
            if not roi_ok: roi_fail += 1
            print(f"  {sid[:35]:35s}  ct={int(ct_ok)}  roi={int(roi_ok)}")
        chk(f"ct_sample_ok  (n=5)", ct_fail  == 0, f"fail={ct_fail}")
        chk(f"roi_sample_ok (n=5)", roi_fail == 0, f"fail={roi_fail}")

    # output root 미존재 확인
    chk("output_root_absent", not OUTPUT_ROOT.exists(),
        str(OUTPUT_ROOT) if OUTPUT_ROOT.exists() else "not_exists(OK)")

    # 요약
    print()
    print("─" * 70)
    print(f"  candidate_verdict   : {candidate_verdict}")
    print(f"  stage1_dev patients : {len(stage1_dev_ids)}")
    print(f"  holdout patients    : {len(holdout_ids)}")
    print(f"  ct_fail (sample)    : {ct_fail}")
    print(f"  roi_fail (sample)   : {roi_fail}")
    print()
    verdict = "FAIL" if errors else "DRY-PLAN OK"
    print(f"판정: {verdict}")
    if errors:
        print("FAIL 항목:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("모든 체크 통과 — 사용자 승인 후:")
        print("  source ~/ai_env/bin/activate && \\")
        print("  python scripts/rd_c2_effb0_v420_candidate_rd4ad_retest.py --run-retest \\")
        print("    2>&1 | tee /tmp/rd_c2_effb0_v420_rd4ad_retest_log.txt")


# ── run-retest ────────────────────────────────────────────────────────────────

def run_retest():
    import numpy as np
    import torch
    import torch.nn.functional as F

    print("=" * 70)
    print("RD-C2: EfficientNet-B0 v4_20 candidate RD4AD retest [RUN-RETEST]")
    print("=" * 70)

    # output root guard
    if OUTPUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재: {OUTPUT_ROOT}")
        sys.exit(1)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    error_rows = []

    # ── 1. stage split 로드 ──────────────────────────────────────────────────
    print("\n[1/7] stage split 로드")
    stage1_dev_ids, holdout_ids = load_stage_split(STAGE_SPLIT_CSV)
    print(f"  stage1_dev  : {len(stage1_dev_ids)} patients")
    print(f"  holdout     : {len(holdout_ids)} patients")

    # ── 2. candidate source inventory 저장 ───────────────────────────────────
    print("\n[2/7] candidate source inventory 저장")
    inv_rows = build_source_inventory(holdout_ids)
    inv_fields = ["source_name", "path", "exists", "description",
                  "rows_or_files", "unique_patients", "score_column",
                  "holdout_intersection"]
    write_csv(OUTPUT_ROOT / "rd_c2_candidate_source_inventory.csv", inv_fields, inv_rows)

    p_c3_inv = next((r for r in inv_rows if r["source_name"] == "p_c3_effb0_v420"), {})
    if not p_c3_inv.get("exists"):
        error_rows.append({"phase": "source", "error": "p_c3 manifest 없음"})
        write_csv(OUTPUT_ROOT / "rd_c2_errors.csv",
                  ["phase", "candidate_id", "patient_id", "safe_id", "error"], error_rows)
        print("[ABORT] p_c3 manifest 없음")
        sys.exit(1)

    # ── 3. p_c3 manifest 로드 + holdout 제거 + ABORT 검증 ────────────────────
    print("\n[3/7] manifest 로드 → holdout 제거 → ABORT 검증")
    rows_c3 = load_csv_rows(P_C3_MANIFEST)

    # STEP A: input rows 검증 (CT/ROI 확인 전)
    n_before = len(rows_c3)
    if n_before != EXPECTED_INPUT_ROWS:
        print(f"[ABORT] input_rows={n_before} != {EXPECTED_INPUT_ROWS}")
        sys.exit(1)
    print(f"  [OK] input_rows = {n_before}")

    # STEP B: holdout denylist 확인 (CT/ROI 확인 전)
    cand_pids_all = set(r["patient_id"] for r in rows_c3)
    detected_holdout = cand_pids_all & holdout_ids
    if len(detected_holdout) != 2 or detected_holdout != EXPECTED_HOLDOUT_PATIENTS:
        print(f"[ABORT] holdout_patients_detected={sorted(detected_holdout)} "
              f"!= {sorted(EXPECTED_HOLDOUT_PATIENTS)}")
        sys.exit(1)
    print(f"  [OK] holdout_patients_detected=2: {sorted(detected_holdout)}")

    # STEP C: holdout rows 제거 (CT/ROI 확인 전)
    rows_c3 = [
        r for r in rows_c3
        if r.get("split") == "stage1_dev"
        and r.get("patient_id", "") not in holdout_ids
    ]
    n_holdout_removed = n_before - len(rows_c3)

    # STEP D: holdout_rows_removed 검증
    if n_holdout_removed != EXPECTED_HOLDOUT_ROWS_REMOVED:
        print(f"[ABORT] holdout_rows_removed={n_holdout_removed} "
              f"!= {EXPECTED_HOLDOUT_ROWS_REMOVED}")
        sys.exit(1)
    print(f"  [OK] holdout_rows_removed = {n_holdout_removed}")

    # STEP E: scoring_rows 검증
    n_scoring_rows = len(rows_c3)
    if n_scoring_rows != EXPECTED_SCORING_ROWS:
        print(f"[ABORT] scoring_rows={n_scoring_rows} != {EXPECTED_SCORING_ROWS}")
        sys.exit(1)
    print(f"  [OK] scoring_rows = {n_scoring_rows}")

    # STEP F: post_filter holdout intersection 검증 (필수 assert)
    post_pids = {r["patient_id"] for r in rows_c3}
    post_intersect = post_pids & holdout_ids
    if post_intersect:
        print(f"[ABORT] post_filter_holdout_intersection={sorted(post_intersect)}")
        sys.exit(1)
    print(f"  [OK] post_filter_holdout_intersection = 0")
    print(f"  [OK] processed_stage2_holdout_rows = 0")

    print(f"  input rows      : {n_before}")
    print(f"  holdout removed : {n_holdout_removed}")
    print(f"  scoring rows    : {n_scoring_rows}")

    # EfficientNet-B0 v4_20 threshold JSON 로드 (first_stage reference)
    with open(EFFB0_V420_THRESHOLD_JSON, encoding="utf-8") as f:
        effb0_th = json.load(f)
    effb0_p95 = float(effb0_th["threshold_p95"])
    effb0_p99 = float(effb0_th["threshold_p99"])

    # fixed96 manifest 생성
    manifest_rows = []
    for r in rows_c3:
        y0 = int(r["y0"]); x0 = int(r["x0"])
        y1 = int(r["y1"]); x1 = int(r["x1"])
        cy0, cx0, cy1, cx1 = derive_fixed96_coords(y0, x0, y1, x1)
        manifest_rows.append({
            "candidate_id":        r["candidate_id"],
            "patient_id":          r["patient_id"],
            "safe_id":             r["safe_id"],
            "stage_split":         "stage1_dev",
            "local_z":             r["local_z"],
            "slice_index":         r.get("slice_index", r["local_z"]),
            "crop_y0":             cy0,
            "crop_x0":             cx0,
            "crop_y1":             cy1,
            "crop_x1":             cx1,
            "first_stage_score":   r["padim_score"],
            "source_branch":       "effb0_v420",
            "backbone":            "EfficientNet-B0",
            "roi_source":          "refined_roi_v4_20_modeB",
            "threshold_source_path": str(EFFB0_V420_THRESHOLD_JSON),
            "threshold_p95":       effb0_p95,
            "threshold_p99":       effb0_p99,
            "label":               r.get("candidate_label", ""),
            "candidate_label":     r.get("candidate_label", ""),
            "candidate_rule":      r.get("candidate_rule", ""),
            "sampling_reason":     r.get("sampling_reason", ""),
            "original_score_row_id": r["candidate_id"],
            "source_score_csv":    r.get("source_score_csv", ""),
        })

    manifest_fields = [
        "candidate_id", "patient_id", "safe_id", "stage_split",
        "local_z", "slice_index",
        "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "first_stage_score",
        "source_branch", "backbone", "roi_source",
        "threshold_source_path", "threshold_p95", "threshold_p99",
        "label", "candidate_label", "candidate_rule", "sampling_reason",
        "original_score_row_id", "source_score_csv",
    ]
    write_csv(OUTPUT_ROOT / "rd_c2_effb0_v420_candidate_manifest.csv",
              manifest_fields, manifest_rows)
    print(f"  manifest rows saved : {len(manifest_rows)}")

    # selected source summary
    lbls = collections.Counter(r["candidate_label"] for r in manifest_rows)
    sel_summary = [
        {"item": "source_name",           "value": "p_c3_effb0_v420"},
        {"item": "candidate_verdict",     "value": "USE_EXISTING_MANIFEST"},
        {"item": "backbone",              "value": "EfficientNet-B0"},
        {"item": "roi_source",            "value": "refined_roi_v4_20_modeB_all_v1"},
        {"item": "input_candidates",      "value": str(len(manifest_rows))},
        {"item": "positive_count",        "value": str(lbls.get("positive", 0))},
        {"item": "hard_negative_count",   "value": str(lbls.get("hard_negative", 0))},
        {"item": "holdout_removed",       "value": str(n_holdout_removed)},
        {"item": "threshold_p95",         "value": str(effb0_p95)},
        {"item": "threshold_p99",         "value": str(effb0_p99)},
        {"item": "threshold_recalculated","value": "False"},
        {"item": "stage2_holdout_access", "value": "0"},
    ]
    write_csv(OUTPUT_ROOT / "rd_c2_selected_source_summary.csv",
              ["item", "value"], sel_summary)

    # ── 3.5. manifest 정렬 (LRU cache 효율 개선) ─────────────────────────────
    print("\n[3.5/7] manifest 정렬 (safe_id, local_z, candidate_id)")
    n_before_sort = len(manifest_rows)
    manifest_rows = sorted(
        manifest_rows,
        key=lambda r: (r["safe_id"], int(r["local_z"]), r["candidate_id"])
    )
    if len(manifest_rows) != n_before_sort:
        print(f"[ABORT] 정렬 후 row count 불일치: {len(manifest_rows)} != {n_before_sort}")
        sys.exit(1)
    cid_sorted_set = set(r["candidate_id"] for r in manifest_rows)
    if len(cid_sorted_set) != n_before_sort:
        print("[ABORT] 정렬 후 candidate_id 중복 또는 누락")
        sys.exit(1)
    print(f"  [OK] 정렬 완료: {len(manifest_rows)} rows (sort: safe_id, local_z, candidate_id)")

    # ── 4. model 로드 (eval-only) ─────────────────────────────────────────────
    print("\n[4/7] model 로드 (eval-only, training/backward/optimizer 금지)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    teacher = build_teacher(LOCAL_RESNET18_WEIGHT).to(device)
    teacher.eval()
    teacher.requires_grad_(False)

    student = build_student_decoder().to(device)
    student.eval()
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=True)
    student.load_state_dict(ckpt.get("student_state_dict", ckpt))
    student.eval()
    print(f"  checkpoint: {CHECKPOINT_PATH.name}")

    teacher_feats = {}

    def make_hook(name):
        def _hook(module, inp, out):
            teacher_feats[name] = out
        return _hook

    teacher.layer1.register_forward_hook(make_hook("layer1"))
    teacher.layer2.register_forward_hook(make_hook("layer2"))
    teacher.layer3.register_forward_hook(make_hook("layer3"))

    # ── 5. RD-B9 threshold 로드 ───────────────────────────────────────────────
    print("\n[5/7] RD-B9 threshold 로드")
    th_b9 = load_rd_b9_thresholds()
    global_p95    = th_b9["global_p95"]
    global_p99    = th_b9["global_p99"]
    bin_thresholds = th_b9["bin_thresholds"]
    print(f"  global p95  : {global_p95}")
    print(f"  global p99  : {global_p99}")

    # scoring input validation 저장 (check/expected/actual/pass 형식)
    val_rows = [
        {"check": "input_rows",                         "expected": EXPECTED_INPUT_ROWS,
         "actual": n_before,                            "pass": n_before == EXPECTED_INPUT_ROWS},
        {"check": "holdout_patients_detected",          "expected": 2,
         "actual": len(detected_holdout),               "pass": len(detected_holdout) == 2},
        {"check": "holdout_patient_ids",                "expected": "LUNG1-295,LUNG1-415",
         "actual": ",".join(sorted(detected_holdout)),  "pass": detected_holdout == EXPECTED_HOLDOUT_PATIENTS},
        {"check": "holdout_rows_removed",               "expected": EXPECTED_HOLDOUT_ROWS_REMOVED,
         "actual": n_holdout_removed,                   "pass": n_holdout_removed == EXPECTED_HOLDOUT_ROWS_REMOVED},
        {"check": "scoring_rows",                       "expected": EXPECTED_SCORING_ROWS,
         "actual": n_scoring_rows,                      "pass": n_scoring_rows == EXPECTED_SCORING_ROWS},
        {"check": "post_filter_holdout_intersection",   "expected": 0,
         "actual": len(post_intersect),                 "pass": len(post_intersect) == 0},
        {"check": "processed_stage2_holdout_rows",      "expected": 0,
         "actual": 0,                                   "pass": True},
        {"check": "ct_roi_validation_before_holdout_filter", "expected": False,
         "actual": False,                               "pass": True},
        {"check": "threshold_recalculated",             "expected": False,
         "actual": False,                               "pass": True},
        {"check": "training_started",                   "expected": False,
         "actual": False,                               "pass": True},
        {"check": "stage2_holdout_access",              "expected": 0,
         "actual": 0,                                   "pass": True},
        {"check": "checkpoint_exists",                  "expected": True,
         "actual": CHECKPOINT_PATH.exists(),            "pass": CHECKPOINT_PATH.exists()},
        {"check": "global_p95",                         "expected": 0.095255,
         "actual": global_p95,                          "pass": abs(global_p95 - 0.095255) < 1e-6},
        {"check": "global_p99",                         "expected": 0.103721,
         "actual": global_p99,                          "pass": abs(global_p99 - 0.103721) < 1e-6},
    ]
    write_csv(OUTPUT_ROOT / "rd_c2_effb0_v420_scoring_input_validation.csv",
              ["check", "expected", "actual", "pass"], val_rows)

    # ── 6. scoring ────────────────────────────────────────────────────────────
    print("\n[6/7] RD4AD scoring")
    ct_cache     = LRUPatientCache(max_size=6)
    roi_cache    = LRUPatientCache(max_size=6)
    sixbin_cache = SixBinCache(max_slices=512)

    score_rows_out = []
    n_scored      = 0
    n_score_nan   = 0
    n_score_inf   = 0
    t_total       = time.perf_counter()

    n_total  = len(manifest_rows)
    n_batches = math.ceil(n_total / SCORE_BATCH_SIZE)

    for b_idx in range(n_batches):
        batch = manifest_rows[b_idx * SCORE_BATCH_SIZE: (b_idx + 1) * SCORE_BATCH_SIZE]
        crops_np  = []
        meta_list = []

        for row in batch:
            candidate_id = row["candidate_id"]
            patient_id   = row["patient_id"]
            safe_id      = row["safe_id"]
            stage_split  = row["stage_split"]
            local_z      = int(row["local_z"])
            crop_y0      = int(row["crop_y0"])
            crop_x0      = int(row["crop_x0"])
            crop_y1      = int(row["crop_y1"])
            crop_x1      = int(row["crop_x1"])
            first_stage_score = float(row["first_stage_score"] or 0.0)
            low_z_warning = int(local_z <= LOW_Z_WARNING_THRESHOLD)

            ct_path  = CT_ROOT      / safe_id / "ct_hu.npy"
            roi_path = V4_20_ROI_ROOT / safe_id / "refined_roi.npy"

            # six_bin 계산 (SixBinCache로 EDT 재계산 방지)
            z_level = boundary_status = six_bin_label = "unknown"
            try:
                roi_arr  = roi_cache.load(safe_id + "_roi", roi_path)
                z_level, boundary_status, six_bin_label = compute_sixbin_label_cached(
                    sixbin_cache, roi_arr, safe_id, local_z, crop_y0, crop_x0, crop_y1, crop_x1
                )
            except Exception as e:
                error_rows.append({
                    "phase": "sixbin", "candidate_id": candidate_id,
                    "patient_id": patient_id, "safe_id": safe_id, "error": str(e),
                })

            # crop 생성 (실패 시 zero crop 대체 금지 → 즉시 ABORT)
            crop_np = None
            try:
                ct_arr  = ct_cache.load(safe_id, ct_path)
                crop_np = build_crop_np(ct_arr, local_z, crop_y0, crop_x0, crop_y1, crop_x1)
            except Exception as e:
                error_rows.append({
                    "phase": "crop", "candidate_id": candidate_id,
                    "patient_id": patient_id, "safe_id": safe_id, "error": str(e),
                })
                raise RuntimeError(
                    f"[ABORT] crop generation failed — "
                    f"candidate_id={candidate_id} safe_id={safe_id}: {e}"
                )

            crops_np.append(crop_np)
            meta_list.append({
                "candidate_id":    candidate_id,
                "patient_id":      patient_id,
                "safe_id":         safe_id,
                "stage_split":     stage_split,
                "local_z":         local_z,
                "crop_y0":         crop_y0,
                "crop_x0":         crop_x0,
                "crop_y1":         crop_y1,
                "crop_x1":         crop_x1,
                "z_level":         z_level,
                "boundary_status": boundary_status,
                "six_bin_label":   six_bin_label,
                "low_z_warning":   low_z_warning,
                "first_stage_score": first_stage_score,
                "label":           row.get("candidate_label", ""),
            })

        # GPU forward (eval-only, no backward)
        batch_t = torch.from_numpy(np.stack(crops_np, axis=0)).to(device)
        with torch.no_grad():
            teacher_feats.clear()
            teacher(batch_t)
            tf1 = teacher_feats["layer1"]
            tf2 = teacher_feats["layer2"]
            tf3 = teacher_feats["layer3"]
            de3, de2, de1 = student(tf3)
            sc1 = (1.0 - F.cosine_similarity(de1, tf1, dim=1)).mean(dim=(1, 2))
            sc2 = (1.0 - F.cosine_similarity(de2, tf2, dim=1)).mean(dim=(1, 2))
            sc3 = (1.0 - F.cosine_similarity(de3, tf3, dim=1)).mean(dim=(1, 2))
            crop_score = (sc1 + sc2 + sc3) / 3.0
            s1_np = sc1.cpu().numpy().astype("float32")
            s2_np = sc2.cpu().numpy().astype("float32")
            s3_np = sc3.cpu().numpy().astype("float32")
            cs_np = crop_score.cpu().numpy().astype("float32")

        for local_i, meta in enumerate(meta_list):
            cs     = float(cs_np[local_i])
            is_nan = int(math.isnan(cs))
            is_inf = int(math.isinf(cs))
            n_score_nan += is_nan
            n_score_inf += is_inf

            bin_label = meta["six_bin_label"]
            bin_th    = bin_thresholds.get(f"bin_{bin_label}", {})
            bin_p95   = float(bin_th.get("p95", global_p95))
            bin_p99   = float(bin_th.get("p99", global_p99))

            score_rows_out.append({
                "candidate_id":      meta["candidate_id"],
                "patient_id":        meta["patient_id"],
                "safe_id":           meta["safe_id"],
                "stage_split":       meta["stage_split"],
                "local_z":           meta["local_z"],
                "crop_y0":           meta["crop_y0"],
                "crop_x0":           meta["crop_x0"],
                "crop_y1":           meta["crop_y1"],
                "crop_x1":           meta["crop_x1"],
                "z_level":           meta["z_level"],
                "boundary_status":   meta["boundary_status"],
                "six_bin_label":     meta["six_bin_label"],
                "low_z_warning":     meta["low_z_warning"],
                "first_stage_score": round(meta["first_stage_score"], 6),
                "label":             meta["label"],
                "score_layer1":      round(float(s1_np[local_i]), 6),
                "score_layer2":      round(float(s2_np[local_i]), 6),
                "score_layer3":      round(float(s3_np[local_i]), 6),
                "rd4ad_crop_score":  round(cs, 6) if not is_nan and not is_inf else cs,
                "global_p95":        round(global_p95, 6),
                "global_p99":        round(global_p99, 6),
                "bin_p95":           round(bin_p95, 6),
                "bin_p99":           round(bin_p99, 6),
                "global_p95_exceed": int(not is_nan and not is_inf and cs > global_p95),
                "global_p99_exceed": int(not is_nan and not is_inf and cs > global_p99),
                "bin_p95_exceed":    int(not is_nan and not is_inf and cs > bin_p95),
                "bin_p99_exceed":    int(not is_nan and not is_inf and cs > bin_p99),
                "score_nan":         is_nan,
                "score_inf":         is_inf,
            })
            n_scored += 1

        if b_idx % 100 == 0 or b_idx == n_batches - 1:
            elapsed = time.perf_counter() - t_total
            pct     = (b_idx + 1) / n_batches * 100
            print(f"    batch {b_idx + 1:5d}/{n_batches}  {pct:5.1f}%  elapsed={elapsed:6.0f}s")

    # score CSV 저장
    score_fields = [
        "candidate_id", "patient_id", "safe_id", "stage_split",
        "local_z", "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "z_level", "boundary_status", "six_bin_label", "low_z_warning",
        "first_stage_score", "label",
        "score_layer1", "score_layer2", "score_layer3", "rd4ad_crop_score",
        "global_p95", "global_p99", "bin_p95", "bin_p99",
        "global_p95_exceed", "global_p99_exceed",
        "bin_p95_exceed",    "bin_p99_exceed",
        "score_nan", "score_inf",
    ]
    write_csv(OUTPUT_ROOT / "rd_c2_effb0_v420_rd4ad_candidate_score.csv",
              score_fields, score_rows_out)

    elapsed_total = time.perf_counter() - t_total
    print(f"\n  scoring 완료  n={n_scored}  nan={n_score_nan}  inf={n_score_inf}  "
          f"elapsed={elapsed_total:.0f}s")

    # score distribution summary
    valid_scores = [r["rd4ad_crop_score"] for r in score_rows_out
                    if not r["score_nan"] and not r["score_inf"]]
    n_valid = len(valid_scores)
    dist_rows = []
    if valid_scores:
        dist_rows.append({
            "metric": "n_total",    "value": n_scored})
        dist_rows.append({
            "metric": "n_valid",    "value": n_valid})
        dist_rows.append({
            "metric": "n_nan",      "value": n_score_nan})
        dist_rows.append({
            "metric": "n_inf",      "value": n_score_inf})
        dist_rows.append({
            "metric": "mean",       "value": round(float(np.mean(valid_scores)), 6)})
        dist_rows.append({
            "metric": "std",        "value": round(float(np.std(valid_scores)), 6)})
        dist_rows.append({
            "metric": "min",        "value": round(float(np.min(valid_scores)), 6)})
        dist_rows.append({
            "metric": "max",        "value": round(float(np.max(valid_scores)), 6)})
        dist_rows.append({
            "metric": "p50",        "value": round(float(np.percentile(valid_scores, 50)), 6)})
        dist_rows.append({
            "metric": "p95",        "value": round(float(np.percentile(valid_scores, 95)), 6)})
        dist_rows.append({
            "metric": "p99",        "value": round(float(np.percentile(valid_scores, 99)), 6)})
        n_gp95 = sum(r["global_p95_exceed"] for r in score_rows_out)
        n_gp99 = sum(r["global_p99_exceed"] for r in score_rows_out)
        dist_rows.append({"metric": "n_global_p95_exceed", "value": n_gp95})
        dist_rows.append({"metric": "n_global_p99_exceed", "value": n_gp99})
    write_csv(OUTPUT_ROOT / "rd_c2_effb0_v420_score_distribution_overall.csv",
              ["metric", "value"], dist_rows)

    # ── 7. safety analysis ────────────────────────────────────────────────────
    print("\n[7/7] safety analysis (label join + suppression)")

    # label join: score_rows_out 기준 (label 이미 포함)
    lbl_counter = collections.Counter(r["label"] for r in score_rows_out)
    lesion_total  = lbl_counter.get("positive", 0)
    hn_total      = lbl_counter.get("hard_negative", 0)
    ambig_total   = lbl_counter.get("ambiguous", 0)
    unknown_total = sum(v for k, v in lbl_counter.items()
                        if k not in ("positive", "hard_negative", "ambiguous"))

    print(f"  positive (lesion)  : {lesion_total}")
    print(f"  hard_negative      : {hn_total}")
    print(f"  ambiguous          : {ambig_total}")
    print(f"  unknown            : {unknown_total}")

    # 억제 정의: suppress if rd4ad_crop_score <= threshold (정상처럼 보임 = 억제)
    RULES_DEF = {
        "G95": ("global_p95_exceed", global_p95),  # exceed=0 → suppressed
        "G99": ("global_p99_exceed", global_p99),
        "B95": ("bin_p95_exceed",    None),         # per-bin
        "B99": ("bin_p99_exceed",    None),
    }

    def is_suppressed_by_rule(row, rule):
        if row["score_nan"] or row["score_inf"]:
            return False
        cs = row["rd4ad_crop_score"]
        if rule == "G95":
            return cs <= global_p95
        elif rule == "G99":
            return cs <= global_p99
        elif rule == "B95":
            return cs <= row["bin_p95"]
        elif rule == "B99":
            return cs <= row["bin_p99"]
        return False

    # threshold rule summary
    rule_summary_rows = []
    for rule in ("G95", "G99", "B95", "B99"):
        valid = [r for r in score_rows_out if not r["score_nan"] and not r["score_inf"]]
        n_supp = sum(1 for r in valid if is_suppressed_by_rule(r, rule))
        pos_supp  = sum(1 for r in valid
                        if r["label"] == "positive" and is_suppressed_by_rule(r, rule))
        hn_supp   = sum(1 for r in valid
                        if r["label"] == "hard_negative" and is_suppressed_by_rule(r, rule))
        rule_summary_rows.append({
            "rule":                   rule,
            "n_valid":                len(valid),
            "total_suppressed":       n_supp,
            "total_suppressed_rate":  round(n_supp / max(len(valid), 1) * 100, 4),
            "lesion_suppressed":      pos_supp,
            "lesion_suppressed_rate": round(pos_supp / max(lesion_total, 1) * 100, 4),
            "hn_suppressed":          hn_supp,
            "hn_suppressed_rate":     round(hn_supp / max(hn_total, 1) * 100, 4),
        })
    write_csv(OUTPUT_ROOT / "rd_c2_effb0_v420_threshold_rule_summary.csv",
              ["rule", "n_valid", "total_suppressed", "total_suppressed_rate",
               "lesion_suppressed", "lesion_suppressed_rate",
               "hn_suppressed", "hn_suppressed_rate"],
              rule_summary_rows)

    # lesion safety summary (환자별)
    pos_rows = [r for r in score_rows_out if r["label"] == "positive"
                and not r["score_nan"] and not r["score_inf"]]
    pat_pos  = collections.defaultdict(list)
    for r in pos_rows:
        pat_pos[r["patient_id"]].append(r)

    lesion_safety_rows = []
    for rule in ("G95", "G99", "B95", "B99"):
        n_pat_all_supp = 0
        for pid, p_rows in pat_pos.items():
            if all(is_suppressed_by_rule(r, rule) for r in p_rows):
                n_pat_all_supp += 1
        n_supp = sum(1 for r in pos_rows if is_suppressed_by_rule(r, rule))
        lesion_safety_rows.append({
            "rule":                         rule,
            "lesion_total":                 len(pos_rows),
            "lesion_suppressed_count":      n_supp,
            "lesion_suppressed_rate":       round(n_supp / max(len(pos_rows), 1) * 100, 4),
            "lesion_patients":              len(pat_pos),
            "lesion_patient_all_suppressed": n_pat_all_supp,
        })
    write_csv(OUTPUT_ROOT / "rd_c2_effb0_v420_lesion_safety_summary.csv",
              ["rule", "lesion_total", "lesion_suppressed_count", "lesion_suppressed_rate",
               "lesion_patients", "lesion_patient_all_suppressed"],
              lesion_safety_rows)

    # hard_negative suppression summary
    hn_rows = [r for r in score_rows_out if r["label"] == "hard_negative"
               and not r["score_nan"] and not r["score_inf"]]
    hn_safety_rows = []
    for rule in ("G95", "G99", "B95", "B99"):
        n_supp = sum(1 for r in hn_rows if is_suppressed_by_rule(r, rule))
        hn_safety_rows.append({
            "rule":                rule,
            "hn_total":            len(hn_rows),
            "hn_suppressed_count": n_supp,
            "hn_suppressed_rate":  round(n_supp / max(len(hn_rows), 1) * 100, 4),
        })
    write_csv(OUTPUT_ROOT / "rd_c2_effb0_v420_hard_negative_suppression_summary.csv",
              ["rule", "hn_total", "hn_suppressed_count", "hn_suppressed_rate"],
              hn_safety_rows)

    # six_bin별 score distribution
    bin_groups = collections.defaultdict(list)
    for r in score_rows_out:
        bin_groups[r["six_bin_label"]].append(r)
    bin_dist_rows = []
    for bl in SIX_BIN_LABELS + ["excluded", "unknown"]:
        b_rows = bin_groups.get(bl, [])
        valid_b = [r for r in b_rows if not r["score_nan"] and not r["score_inf"]]
        scores_b = [r["rd4ad_crop_score"] for r in valid_b]
        bin_dist_rows.append({
            "six_bin_label": bl,
            "n": len(b_rows),
            "n_valid": len(valid_b),
            "mean":  round(float(np.mean(scores_b)), 6)        if scores_b else 0.0,
            "p50":   round(float(np.percentile(scores_b, 50)), 6) if scores_b else 0.0,
            "p95":   round(float(np.percentile(scores_b, 95)), 6) if scores_b else 0.0,
            "n_g95_exceed": sum(r["global_p95_exceed"] for r in valid_b),
        })
    write_csv(OUTPUT_ROOT / "rd_c2_effb0_v420_score_distribution_by_sixbin.csv",
              ["six_bin_label", "n", "n_valid", "mean", "p50", "p95", "n_g95_exceed"],
              bin_dist_rows)

    # first_stage vs rd4ad pearson r
    if pos_rows:
        fs_vals  = [r["first_stage_score"]  for r in pos_rows]
        rd4_vals = [r["rd4ad_crop_score"]    for r in pos_rows]
        try:
            r_val = float(np.corrcoef(fs_vals, rd4_vals)[0, 1])
        except Exception:
            r_val = float("nan")
    else:
        r_val = float("nan")

    # old source 비교
    old_summary = {}
    if OLD_SOURCE_SAFETY_JSON.exists():
        with open(OLD_SOURCE_SAFETY_JSON, encoding="utf-8") as f:
            old_summary = json.load(f)

    rule_lup = {r["rule"]: r for r in rule_summary_rows}
    lsaf_lup = {r["rule"]: r for r in lesion_safety_rows}
    comparison_rows = [
        {
            "metric":           "source_branch",
            "old_source":       "ResNet18_roi_0_0",
            "effb0_v420":       "EfficientNet-B0_v4_20",
        },
        {
            "metric":           "candidate_count",
            "old_source":       old_summary.get("input_score_rows", "N/A"),
            "effb0_v420":       n_scored,
        },
        {
            "metric":           "positive_count",
            "old_source":       old_summary.get("lesion_total", "N/A"),
            "effb0_v420":       lesion_total,
        },
        {
            "metric":           "hard_negative_count",
            "old_source":       old_summary.get("hard_negative_total", "N/A"),
            "effb0_v420":       hn_total,
        },
        {
            "metric":           "G95_lesion_suppressed_rate",
            "old_source":       old_summary.get("lesion_suppression_rate_by_rule", {}).get("G95", "N/A"),
            "effb0_v420":       lsaf_lup.get("G95", {}).get("lesion_suppressed_rate", "N/A"),
        },
        {
            "metric":           "G99_lesion_suppressed_rate",
            "old_source":       old_summary.get("lesion_suppression_rate_by_rule", {}).get("G99", "N/A"),
            "effb0_v420":       lsaf_lup.get("G99", {}).get("lesion_suppressed_rate", "N/A"),
        },
        {
            "metric":           "G95_hn_suppressed_rate",
            "old_source":       old_summary.get("hard_negative_suppression_rate_by_rule", {}).get("G95", "N/A"),
            "effb0_v420":       rule_lup.get("G95", {}).get("hn_suppressed_rate", "N/A"),
        },
        {
            "metric":           "G99_hn_suppressed_rate",
            "old_source":       old_summary.get("hard_negative_suppression_rate_by_rule", {}).get("G99", "N/A"),
            "effb0_v420":       rule_lup.get("G99", {}).get("hn_suppressed_rate", "N/A"),
        },
        {
            "metric":           "G95_patient_all_suppressed",
            "old_source":       old_summary.get("patient_level_safety_by_rule", {}).get(
                                    "G95", {}).get("all_suppressed_count", "N/A"),
            "effb0_v420":       lsaf_lup.get("G95", {}).get("lesion_patient_all_suppressed", "N/A"),
        },
        {
            "metric":           "first_stage_vs_rd4ad_pearson_r_positive",
            "old_source":       "N/A",
            "effb0_v420":       round(r_val, 4) if not math.isnan(r_val) else "nan",
        },
    ]
    write_csv(OUTPUT_ROOT / "rd_c2_old_vs_effb0_v420_comparison.csv",
              ["metric", "old_source", "effb0_v420"], comparison_rows)

    # ── final decision ────────────────────────────────────────────────────────
    g95_lesion = lsaf_lup.get("G95", {}).get("lesion_suppressed_rate", 100.0)
    g99_lesion = lsaf_lup.get("G99", {}).get("lesion_suppressed_rate", 100.0)
    g95_hn_rate = rule_lup.get("G95", {}).get("hn_suppressed_rate", 0.0)
    g99_hn_rate = rule_lup.get("G99", {}).get("hn_suppressed_rate", 0.0)
    g95_pat    = lsaf_lup.get("G95", {}).get("lesion_patient_all_suppressed", 0)
    g99_pat    = lsaf_lup.get("G99", {}).get("lesion_patient_all_suppressed", 0)

    all_abort_passed = (n_before == EXPECTED_INPUT_ROWS
                        and n_holdout_removed == EXPECTED_HOLDOUT_ROWS_REMOVED
                        and n_scored == EXPECTED_SCORING_ROWS)

    if not all_abort_passed:
        final_decision = "BLOCKED"
    elif lesion_total == 0:
        final_decision = "BLOCKED"
    elif (g95_lesion < 5.0 and g95_hn_rate > 5.0 and g95_pat == 0):
        final_decision = "USEFUL_WITH_CAUTION"
    elif (g99_lesion < 5.0 and g99_hn_rate > 5.0 and g99_pat == 0):
        final_decision = "USEFUL_WITH_CAUTION"
    elif g95_lesion >= 10.0 or g95_pat > 0:
        final_decision = "NOT_USEFUL"
    else:
        final_decision = "USEFUL_WITH_CAUTION"

    # real_error_count / all_checks_passed (강화)
    real_error_count = sum(1 for e in error_rows if e.get("phase") != "none")
    sixbin_unknown_count = sum(1 for r in score_rows_out if r.get("six_bin_label") == "unknown")
    all_checks_passed = (
        unknown_total == 0
        and n_score_nan == 0
        and n_score_inf == 0
        and real_error_count == 0
        and n_before == EXPECTED_INPUT_ROWS
        and n_holdout_removed == EXPECTED_HOLDOUT_ROWS_REMOVED
        and n_scored == EXPECTED_SCORING_ROWS
        and len(post_intersect) == 0
        and final_decision != "BLOCKED"
    )

    # summary JSON (사용자 요청 필드 이름 준수)
    summary_obj = {
        "selected_candidate_source":          str(P_C3_MANIFEST),
        "candidate_source_branch":            "EfficientNet-B0_v4_20",
        "candidate_source_backbone":          "EfficientNet-B0",
        "candidate_source_roi":               "refined_roi_v4_20_modeB_all_v1",
        "input_rows":                         n_before,
        "holdout_rows_removed":               n_holdout_removed,
        "scored_candidates":                  n_scored,
        "post_filter_holdout_intersection":   0,
        "processed_stage2_holdout_rows":      0,
        "positive_count":                     lesion_total,
        "hard_negative_count":                hn_total,
        "ambiguous_count":                    ambig_total,
        "unknown_label_count":                unknown_total,
        "score_nan_count":                    n_score_nan,
        "score_inf_count":                    n_score_inf,
        "checkpoint_loaded":                  str(CHECKPOINT_PATH),
        "threshold_source":                   "RD-B9 normal_val only",
        "threshold_recalculated":             False,
        "training_started":                   False,
        "backward_called":                    False,
        "optimizer_created":                  False,
        "checkpoint_saved":                   False,
        "first_stage_score_modified":         False,
        "stage2_holdout_access":              0,
        "global_p95":                         global_p95,
        "global_p99":                         global_p99,
        "rule_G95_lesion_suppressed_rate":    g95_lesion,
        "rule_G99_lesion_suppressed_rate":    g99_lesion,
        "rule_G95_hard_negative_suppressed_rate": rule_lup.get("G95", {}).get("hn_suppressed_rate", None),
        "rule_G99_hard_negative_suppressed_rate": rule_lup.get("G99", {}).get("hn_suppressed_rate", None),
        "rule_G95_patient_all_suppressed":    g95_pat,
        "old_source_G95_lesion_suppressed_rate": 82.28,
        "old_source_G99_lesion_suppressed_rate": 95.65,
        "old_source_decision":                "NOT_USEFUL",
        "first_stage_vs_rd4ad_pearson_r":     round(r_val, 4) if not math.isnan(r_val) else None,
        "final_decision":                     final_decision,
        "real_error_count":                   real_error_count,
        "sixbin_unknown_count":               sixbin_unknown_count,
        "all_checks_passed":                  all_checks_passed,
        "elapsed_seconds":                    round(elapsed_total, 1),
        "optimized_version":                  True,
        "manifest_sorted_for_cache":          True,
        "sort_keys":                          "safe_id,local_z,candidate_id",
        "sixbin_cache_enabled":               True,
        "sixbin_cache_hits":                  sixbin_cache.hits,
        "sixbin_cache_misses":                sixbin_cache.misses,
        "sixbin_cache_hit_rate":              round(sixbin_cache.hit_rate, 4),
        "unique_safe_id_local_z_count":       sixbin_cache.unique_count,
        "average_batch_seconds":              round(elapsed_total / max(n_batches, 1), 3),
    }
    write_json(OUTPUT_ROOT / "rd_c2_effb0_v420_rd4ad_retest_summary.json", summary_obj)

    # errors CSV
    write_csv(OUTPUT_ROOT / "rd_c2_errors.csv",
              ["phase", "candidate_id", "patient_id", "safe_id", "error"],
              error_rows if error_rows else [{"phase": "none", "candidate_id": "",
                                              "patient_id": "", "safe_id": "", "error": "no_errors"}])

    # report.md
    _write_report(summary_obj, comparison_rows, rule_summary_rows, lesion_safety_rows)

    # DONE
    (OUTPUT_ROOT / "DONE").touch()
    print("\n[DONE] 모든 파일 생성 완료")
    print(f"  -> {OUTPUT_ROOT}")


def run_profile():
    import numpy as np
    import torch
    import torch.nn.functional as F

    print("=" * 70)
    print("RD-C2: EfficientNet-B0 v4_20 RD4AD [PROFILE-OPTIMIZED]")
    print("=" * 70)

    if OUTPUT_ROOT_PROFILE.exists():
        print(f"[ABORT] profile output root 이미 존재: {OUTPUT_ROOT_PROFILE}")
        sys.exit(1)
    OUTPUT_ROOT_PROFILE.mkdir(parents=True, exist_ok=False)

    error_rows = []

    # 1. stage split
    print("\n[1/5] stage split 로드")
    stage1_dev_ids, holdout_ids = load_stage_split(STAGE_SPLIT_CSV)
    print(f"  stage1_dev  : {len(stage1_dev_ids)} patients")
    print(f"  holdout     : {len(holdout_ids)} patients")

    # 2. manifest 로드 + holdout 제거 + ABORT 검증
    print("\n[2/5] manifest 로드 → holdout 제거 → ABORT 검증")
    rows_c3  = load_csv_rows(P_C3_MANIFEST)
    n_before = len(rows_c3)
    if n_before != EXPECTED_INPUT_ROWS:
        print(f"[ABORT] input_rows={n_before} != {EXPECTED_INPUT_ROWS}")
        sys.exit(1)
    print(f"  [OK] input_rows = {n_before}")

    cand_pids_all    = set(r["patient_id"] for r in rows_c3)
    detected_holdout = cand_pids_all & holdout_ids
    if len(detected_holdout) != 2 or detected_holdout != EXPECTED_HOLDOUT_PATIENTS:
        print(f"[ABORT] holdout_patients={sorted(detected_holdout)}")
        sys.exit(1)
    print(f"  [OK] holdout detected: {sorted(detected_holdout)}")

    rows_c3 = [r for r in rows_c3
               if r.get("split") == "stage1_dev" and r.get("patient_id", "") not in holdout_ids]
    n_holdout_removed = n_before - len(rows_c3)
    if n_holdout_removed != EXPECTED_HOLDOUT_ROWS_REMOVED:
        print(f"[ABORT] holdout_removed={n_holdout_removed} != {EXPECTED_HOLDOUT_ROWS_REMOVED}")
        sys.exit(1)
    print(f"  [OK] holdout_removed = {n_holdout_removed}")

    n_scoring_rows = len(rows_c3)
    if n_scoring_rows != EXPECTED_SCORING_ROWS:
        print(f"[ABORT] scoring_rows={n_scoring_rows} != {EXPECTED_SCORING_ROWS}")
        sys.exit(1)
    print(f"  [OK] scoring_rows = {n_scoring_rows}")

    post_pids      = {r["patient_id"] for r in rows_c3}
    post_intersect = post_pids & holdout_ids
    if post_intersect:
        print(f"[ABORT] post_filter_holdout_intersection={sorted(post_intersect)}")
        sys.exit(1)
    print(f"  [OK] post_filter_holdout_intersection = 0")

    # fixed96 manifest 생성
    with open(EFFB0_V420_THRESHOLD_JSON, encoding="utf-8") as f:
        effb0_th = json.load(f)

    manifest_rows = []
    for r in rows_c3:
        y0 = int(r["y0"]); x0 = int(r["x0"])
        y1 = int(r["y1"]); x1 = int(r["x1"])
        cy0, cx0, cy1, cx1 = derive_fixed96_coords(y0, x0, y1, x1)
        manifest_rows.append({
            "candidate_id":      r["candidate_id"],
            "patient_id":        r["patient_id"],
            "safe_id":           r["safe_id"],
            "stage_split":       "stage1_dev",
            "local_z":           r["local_z"],
            "crop_y0":           cy0, "crop_x0": cx0,
            "crop_y1":           cy1, "crop_x1": cx1,
            "first_stage_score": r["padim_score"],
            "label":             r.get("candidate_label", ""),
        })

    # 3. 정렬 + candidate_id 보존 검증
    print(f"\n[3/5] manifest 정렬 (safe_id, local_z, candidate_id)")
    n_before_sort = len(manifest_rows)
    manifest_rows_sorted = sorted(
        manifest_rows,
        key=lambda r: (r["safe_id"], int(r["local_z"]), r["candidate_id"])
    )
    cid_orig   = set(r["candidate_id"] for r in manifest_rows)
    cid_sorted = set(r["candidate_id"] for r in manifest_rows_sorted)
    if cid_orig != cid_sorted or len(manifest_rows_sorted) != n_before_sort:
        print("[ABORT] 정렬 전후 candidate_id 불일치")
        sys.exit(1)
    print(f"  [OK] 정렬 전후 candidate_id 일치: {len(cid_orig)} rows")

    # profile subset: positive/hard_negative 다양하게 포함
    lbl_set = {}
    for r in manifest_rows_sorted:
        lbl = r.get("label", "")
        lbl_set.setdefault(lbl, []).append(r)
    profile_rows = []
    for lbl in ("positive", "hard_negative", "ambiguous", ""):
        profile_rows.extend(lbl_set.get(lbl, []))
    profile_rows = profile_rows[:PROFILE_ROWS]
    print(f"  profile rows: {len(profile_rows)}")
    lbl_dist = {}
    for r in profile_rows:
        lbl_dist[r.get("label", "")] = lbl_dist.get(r.get("label", ""), 0) + 1
    for lbl, cnt in lbl_dist.items():
        print(f"    label={lbl!r}: {cnt}")

    # 4. six_bin cache vs direct 동일성 검증 (sample 200)
    print(f"\n[4/5] six_bin cache/direct 동일성 검증 (sample 200)")
    sixbin_cache_eq = SixBinCache(max_slices=512)
    roi_cache_eq    = LRUPatientCache(max_size=6)
    sample_n    = min(200, len(profile_rows))
    mismatches  = []
    equiv_rows  = []
    for r in profile_rows[:sample_n]:
        safe_id  = r["safe_id"]
        local_z  = int(r["local_z"])
        crop_y0  = int(r["crop_y0"]); crop_x0 = int(r["crop_x0"])
        crop_y1  = int(r["crop_y1"]); crop_x1 = int(r["crop_x1"])
        roi_path = V4_20_ROI_ROOT / safe_id / "refined_roi.npy"
        try:
            roi_arr  = roi_cache_eq.load(safe_id + "_roi", roi_path)
            n_slices = roi_arr.shape[0]
            z_ratio  = local_z / max(n_slices - 1, 1)
            zl_d, bs_d, sb_d = compute_sixbin_label(
                roi_arr, local_z, crop_y0, crop_x0, crop_y1, crop_x1, z_ratio)
            zl_c, bs_c, sb_c = compute_sixbin_label_cached(
                sixbin_cache_eq, roi_arr, safe_id, local_z, crop_y0, crop_x0, crop_y1, crop_x1)
            match = (zl_d == zl_c and bs_d == bs_c and sb_d == sb_c)
            if not match:
                mismatches.append({
                    "candidate_id": r["candidate_id"], "safe_id": safe_id, "local_z": local_z,
                    "direct_zl": zl_d, "direct_bs": bs_d, "direct_sb": sb_d,
                    "cached_zl": zl_c, "cached_bs": bs_c, "cached_sb": sb_c,
                })
            equiv_rows.append({
                "candidate_id":  r["candidate_id"], "safe_id": safe_id, "local_z": local_z,
                "direct_result": sb_d, "cached_result": sb_c, "match": int(match),
            })
        except Exception as e:
            error_rows.append({"phase": "equivalence", "candidate_id": r["candidate_id"],
                                "patient_id": r.get("patient_id", ""), "safe_id": safe_id,
                                "error": str(e)})
    write_csv(OUTPUT_ROOT_PROFILE / "rd_c2_profile_sixbin_equivalence.csv",
              ["candidate_id", "safe_id", "local_z", "direct_result", "cached_result", "match"],
              equiv_rows)
    print(f"  sample checked : {len(equiv_rows)}")
    print(f"  mismatch count : {len(mismatches)}")
    if mismatches:
        print("  [FAIL] mismatch 발생:")
        for m in mismatches[:3]:
            print(f"    {m}")

    # 5. profile scoring
    print(f"\n[5/5] profile scoring ({len(profile_rows)} rows)")
    th_b9      = load_rd_b9_thresholds()
    global_p95 = th_b9["global_p95"]
    global_p99 = th_b9["global_p99"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    teacher = build_teacher(LOCAL_RESNET18_WEIGHT).to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    student = build_student_decoder().to(device)
    student.eval()
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=True)
    student.load_state_dict(ckpt.get("student_state_dict", ckpt))
    student.eval()

    teacher_feats_p = {}

    def make_hook_p(name):
        def _hook(module, inp, out):
            teacher_feats_p[name] = out
        return _hook

    teacher.layer1.register_forward_hook(make_hook_p("layer1"))
    teacher.layer2.register_forward_hook(make_hook_p("layer2"))
    teacher.layer3.register_forward_hook(make_hook_p("layer3"))

    ct_cache_p     = LRUPatientCache(max_size=6)
    roi_cache_p    = LRUPatientCache(max_size=6)
    sixbin_cache_p = SixBinCache(max_slices=512)
    score_rows_p   = []
    n_score_nan = n_score_inf = 0
    n_batches_p = math.ceil(len(profile_rows) / SCORE_BATCH_SIZE)
    batch_times = []
    t_total_p   = time.perf_counter()

    for b_idx in range(n_batches_p):
        batch = profile_rows[b_idx * SCORE_BATCH_SIZE: (b_idx + 1) * SCORE_BATCH_SIZE]
        crops_np  = []
        meta_list = []
        t_batch   = time.perf_counter()

        for row in batch:
            candidate_id = row["candidate_id"]
            patient_id   = row["patient_id"]
            safe_id      = row["safe_id"]
            local_z      = int(row["local_z"])
            crop_y0 = int(row["crop_y0"]); crop_x0 = int(row["crop_x0"])
            crop_y1 = int(row["crop_y1"]); crop_x1 = int(row["crop_x1"])
            ct_path  = CT_ROOT        / safe_id / "ct_hu.npy"
            roi_path = V4_20_ROI_ROOT / safe_id / "refined_roi.npy"

            z_level = boundary_status = six_bin_label = "unknown"
            try:
                roi_arr = roi_cache_p.load(safe_id + "_roi", roi_path)
                z_level, boundary_status, six_bin_label = compute_sixbin_label_cached(
                    sixbin_cache_p, roi_arr, safe_id, local_z, crop_y0, crop_x0, crop_y1, crop_x1)
            except Exception as e:
                error_rows.append({"phase": "sixbin", "candidate_id": candidate_id,
                                    "patient_id": patient_id, "safe_id": safe_id, "error": str(e)})

            try:
                ct_arr  = ct_cache_p.load(safe_id, ct_path)
                crop_np = build_crop_np(ct_arr, local_z, crop_y0, crop_x0, crop_y1, crop_x1)
            except Exception as e:
                error_rows.append({"phase": "crop", "candidate_id": candidate_id,
                                    "patient_id": patient_id, "safe_id": safe_id, "error": str(e)})
                raise RuntimeError(
                    f"[ABORT] crop failed — candidate_id={candidate_id} safe_id={safe_id}: {e}")

            crops_np.append(crop_np)
            meta_list.append({
                "candidate_id": candidate_id, "patient_id": patient_id,
                "safe_id": safe_id, "local_z": local_z,
                "z_level": z_level, "boundary_status": boundary_status,
                "six_bin_label": six_bin_label,
                "first_stage_score": float(row.get("first_stage_score") or 0.0),
                "label": row.get("label", ""),
            })

        batch_t = torch.from_numpy(np.stack(crops_np, axis=0)).to(device)
        with torch.no_grad():
            teacher_feats_p.clear()
            teacher(batch_t)
            tf1 = teacher_feats_p["layer1"]
            tf2 = teacher_feats_p["layer2"]
            tf3 = teacher_feats_p["layer3"]
            de3, de2, de1 = student(tf3)
            sc1 = (1.0 - F.cosine_similarity(de1, tf1, dim=1)).mean(dim=(1, 2))
            sc2 = (1.0 - F.cosine_similarity(de2, tf2, dim=1)).mean(dim=(1, 2))
            sc3 = (1.0 - F.cosine_similarity(de3, tf3, dim=1)).mean(dim=(1, 2))
            crop_score = (sc1 + sc2 + sc3) / 3.0
            cs_np = crop_score.cpu().numpy().astype("float32")

        batch_times.append(time.perf_counter() - t_batch)
        for local_i, meta in enumerate(meta_list):
            cs     = float(cs_np[local_i])
            is_nan = int(math.isnan(cs))
            is_inf = int(math.isinf(cs))
            n_score_nan += is_nan
            n_score_inf += is_inf
            score_rows_p.append({
                "candidate_id":     meta["candidate_id"],
                "patient_id":       meta["patient_id"],
                "six_bin_label":    meta["six_bin_label"],
                "rd4ad_crop_score": round(cs, 6) if not is_nan and not is_inf else cs,
                "score_nan":        is_nan,
                "score_inf":        is_inf,
            })

    elapsed_p    = time.perf_counter() - t_total_p
    avg_batch    = sum(batch_times) / max(len(batch_times), 1)
    rows_per_sec = len(profile_rows) / max(elapsed_p, 1e-9)
    est_full_sec = EXPECTED_SCORING_ROWS / max(rows_per_sec, 1e-9)

    print(f"  scored     : {len(score_rows_p)}")
    print(f"  score_nan  : {n_score_nan}")
    print(f"  score_inf  : {n_score_inf}")
    print(f"  elapsed    : {elapsed_p:.1f}s")
    print(f"  avg_batch  : {avg_batch:.3f}s")
    print(f"  rows/sec   : {rows_per_sec:.1f}")
    print(f"  est_full   : {est_full_sec / 60:.1f} min ({est_full_sec:.0f}s)")
    print(f"  sixbin hits: {sixbin_cache_p.hits}  misses={sixbin_cache_p.misses}"
          f"  rate={sixbin_cache_p.hit_rate:.3f}")

    # 결과 파일 저장
    timing_rows = [{"batch_idx": i, "elapsed_sec": round(t, 4)} for i, t in enumerate(batch_times)]
    write_csv(OUTPUT_ROOT_PROFILE / "rd_c2_profile_timing.csv",
              ["batch_idx", "elapsed_sec"], timing_rows)

    real_error_count = sum(1 for e in error_rows if e.get("phase") != "none")
    mismatch_count   = len(mismatches)
    all_checks_passed = (
        n_before              == EXPECTED_INPUT_ROWS
        and n_holdout_removed == EXPECTED_HOLDOUT_ROWS_REMOVED
        and n_scoring_rows    == EXPECTED_SCORING_ROWS
        and len(profile_rows) <= PROFILE_ROWS
        and len(post_intersect) == 0
        and mismatch_count    == 0
        and n_score_nan       == 0
        and n_score_inf       == 0
        and real_error_count  == 0
    )

    cache_summary_obj = {
        "profile_rows":                       len(profile_rows),
        "sixbin_cache_hits":                  sixbin_cache_p.hits,
        "sixbin_cache_misses":                sixbin_cache_p.misses,
        "sixbin_cache_hit_rate":              round(sixbin_cache_p.hit_rate, 4),
        "unique_safe_id_local_z_count":       sixbin_cache_p.unique_count,
        "sixbin_direct_cache_mismatch_count": mismatch_count,
        "avg_batch_seconds":                  round(avg_batch, 4),
        "rows_per_second":                    round(rows_per_sec, 2),
        "estimated_full_runtime_seconds":     round(est_full_sec, 1),
        "estimated_full_runtime_minutes":     round(est_full_sec / 60, 1),
    }
    write_json(OUTPUT_ROOT_PROFILE / "rd_c2_profile_cache_summary.json", cache_summary_obj)

    profile_summary_obj = {
        "input_rows":                         n_before,
        "holdout_rows_removed":               n_holdout_removed,
        "scoring_rows_full_expected":         EXPECTED_SCORING_ROWS,
        "profile_rows":                       len(profile_rows),
        "post_filter_holdout_intersection":   len(post_intersect),
        "sixbin_direct_cache_mismatch_count": mismatch_count,
        "score_nan_count":                    n_score_nan,
        "score_inf_count":                    n_score_inf,
        "real_error_count":                   real_error_count,
        "threshold_recalculated":             False,
        "training_started":                   False,
        "checkpoint_saved":                   False,
        "stage2_holdout_access":              0,
        "all_checks_passed":                  all_checks_passed,
        "avg_batch_seconds":                  round(avg_batch, 4),
        "estimated_full_runtime_seconds":     round(est_full_sec, 1),
    }
    write_json(OUTPUT_ROOT_PROFILE / "rd_c2_profile_summary.json", profile_summary_obj)

    write_csv(OUTPUT_ROOT_PROFILE / "rd_c2_profile_errors.csv",
              ["phase", "candidate_id", "patient_id", "safe_id", "error"],
              error_rows if error_rows else [{"phase": "none", "candidate_id": "",
                                              "patient_id": "", "safe_id": "", "error": "no_errors"}])

    report_lines = [
        "# RD-C2 Profile Report",
        "",
        "## 통과 조건 체크",
        "",
        f"- input_rows={n_before} (expected={EXPECTED_INPUT_ROWS}): "
        f"{'OK' if n_before == EXPECTED_INPUT_ROWS else 'FAIL'}",
        f"- holdout_removed={n_holdout_removed} (expected={EXPECTED_HOLDOUT_ROWS_REMOVED}): "
        f"{'OK' if n_holdout_removed == EXPECTED_HOLDOUT_ROWS_REMOVED else 'FAIL'}",
        f"- scoring_rows_full_expected={EXPECTED_SCORING_ROWS}: OK",
        f"- profile_rows={len(profile_rows)} (<={PROFILE_ROWS}): "
        f"{'OK' if len(profile_rows) <= PROFILE_ROWS else 'FAIL'}",
        f"- post_filter_holdout_intersection={len(post_intersect)}: "
        f"{'OK' if len(post_intersect) == 0 else 'FAIL'}",
        f"- sixbin_mismatch={mismatch_count}: {'OK' if mismatch_count == 0 else 'FAIL'}",
        f"- score_nan={n_score_nan}: {'OK' if n_score_nan == 0 else 'FAIL'}",
        f"- score_inf={n_score_inf}: {'OK' if n_score_inf == 0 else 'FAIL'}",
        f"- real_error_count={real_error_count}: {'OK' if real_error_count == 0 else 'FAIL'}",
        "",
        "## 성능 측정",
        "",
        f"- avg_batch_seconds: {round(avg_batch, 4)}",
        f"- estimated_full_runtime: {round(est_full_sec / 60, 1)} min ({round(est_full_sec, 0):.0f}s)",
        f"- sixbin_cache_hit_rate: {round(sixbin_cache_p.hit_rate, 4)}",
        "",
        f"## 최종 판정: {'PASS' if all_checks_passed else 'FAIL'}",
        "",
        f"{'full run 진행 가능' if all_checks_passed else 'mismatch 또는 error 있음 — full run 금지'}",
    ]
    report_path = OUTPUT_ROOT_PROFILE / "rd_c2_profile_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  -> {report_path.name}")

    (OUTPUT_ROOT_PROFILE / "DONE").touch()
    print(f"\n[DONE] profile 완료")
    print(f"  -> {OUTPUT_ROOT_PROFILE}")
    print()
    verdict = "PASS" if all_checks_passed else "FAIL"
    print(f"판정: {verdict}")
    if all_checks_passed:
        print("  source ~/ai_env/bin/activate && \\")
        print("  python scripts/rd_c2_effb0_v420_candidate_rd4ad_retest.py --run-retest \\")
        print("    2>&1 | tee /tmp/rd_c2_effb0_v420_rd4ad_retest_log.txt")
    else:
        if mismatch_count > 0:
            print(f"  six_bin mismatch: {mismatch_count} — full run 금지")
        if real_error_count > 0:
            print(f"  real_errors: {real_error_count} — full run 금지")


def _write_report(summary, comparison_rows, rule_summary_rows, lesion_safety_rows):
    lines = [
        "# RD-C2: EfficientNet-B0 v4_20 Candidate RD4AD Retest Report",
        "",
        "## 1. 왜 retest가 필요한가",
        "",
        "- 이전 RD-B10/RD-B11은 **ResNet18 roi_0_0** 기반 후보(`stage1_dev_fixed96_thr001_v1`)를 사용했다.",
        "- 그러나 1차 branch 중 성능이 가장 좋은 것은 **EfficientNet-B0 + refined_roi_v4_20_modeB** 기반이다.",
        "- RD-C2는 동일한 RD-B8f checkpoint + RD-B9 threshold를 사용하되,",
        "  후보 source만 EfficientNet-B0 v4_20으로 교체하여 RD4AD suppression safety를 재분석한다.",
        "",
        "## 2. Candidate Source Audit 결과",
        "",
        f"- p_c3 manifest 존재: **{P_C3_MANIFEST}**",
        f"- candidate_verdict  : **USE_EXISTING_MANIFEST** (fixed96 coords 파생)",
        f"- backbone           : EfficientNet-B0",
        f"- roi_source         : refined_roi_v4_20_modeB_all_v1",
        f"- threshold_p95      : 13.231265 (P-B9 계열 확인)",
        "",
        "## 3. 선택 source 근거",
        "",
        "- p_c3 manifest는 EfficientNet-B0 v4_20 branch의 stage1_dev 전체 후보를 포함한다.",
        "- 단, p_c3 원본에는 lesion_stage_split_v1 기준 stage2_holdout 2명"
        "  (LUNG1-295, LUNG1-415)의 934 rows가 포함되어 있었다.",
        "- RD-C2 run-retest에서는 CT/ROI 확인 및 scoring 전에 holdout denylist를 먼저 적용하여"
        "  934 rows를 제거했다.",
        "- 실제 scoring 대상은 113,447 rows이며 post_filter_holdout_intersection=0이다.",
        "- patch y0/x0/y1/x1에서 center ± 48 로 fixed96 crop 좌표 파생.",
        "",
        "## 4. Candidate Manifest 생성 여부",
        "",
        "- rd_c2_effb0_v420_candidate_manifest.csv 새로 생성 (p_c3 기반 fixed96 파생)",
        f"- 총 후보수: {summary['scored_candidates']}",
        f"- positive : {summary['positive_count']}",
        f"- hard_neg : {summary['hard_negative_count']}",
        "",
        "## 5. RD4AD Scoring 결과",
        "",
        f"- checkpoint : {Path(summary['checkpoint_loaded']).name}",
        f"- scored     : {summary['scored_candidates']}",
        f"- score_nan  : {summary['score_nan_count']}",
        f"- score_inf  : {summary['score_inf_count']}",
        f"- elapsed    : {summary['elapsed_seconds']}s",
        "",
        "## 6. Threshold Rule별 Suppression / Safety 결과",
        "",
        "| rule | total_suppressed_rate | lesion_suppressed_rate | hn_suppressed_rate |",
        "|------|-----------------------|------------------------|---------------------|",
    ]
    for r in rule_summary_rows:
        lines.append(
            f"| {r['rule']} | {r['total_suppressed_rate']:.2f}% | "
            f"{r['lesion_suppressed_rate']:.2f}% | {r['hn_suppressed_rate']:.2f}% |"
        )
    lines += [
        "",
        "### 환자 레벨 Safety (lesion 환자 전체 후보 억제)",
        "",
        "| rule | lesion_patients | patient_all_suppressed |",
        "|------|-----------------|------------------------|",
    ]
    for r in lesion_safety_rows:
        lines.append(
            f"| {r['rule']} | {r['lesion_patients']} | {r['lesion_patient_all_suppressed']} |"
        )
    lines += [
        "",
        "## 7. Old Source 결과와 비교",
        "",
        "| metric | old_source (ResNet18 roi_0_0) | effb0_v420 |",
        "|--------|-------------------------------|-----------|",
    ]
    for c in comparison_rows:
        lines.append(f"| {c['metric']} | {c['old_source']} | {c['effb0_v420']} |")
    lines += [
        "",
        "## 8. 최종 Decision",
        "",
        f"**{summary['final_decision']}**",
        "",
        "## 9. 금지/주의 확인",
        "",
        "- stage2_holdout 접근: **없음**",
        "- threshold 재계산: **없음**",
        "- first-stage score 수정: **없음**",
        "- suppression 적용: **없음** (분석만)",
        "- training/backward/optimizer: **없음**",
        "- checkpoint 저장: **없음**",
        "",
    ]
    report_path = OUTPUT_ROOT / "rd_c2_effb0_v420_rd4ad_retest_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  -> {report_path.name}")


# ── 진입점 ─────────────────────────────────────────────────────────────────────

if IS_DRY_PLAN:
    run_dry_plan()
elif IS_PROFILE:
    run_profile()
elif IS_RUN_RETEST:
    run_retest()
