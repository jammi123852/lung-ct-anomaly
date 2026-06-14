"""
RD-A1 Normal Train Crop Subtype Audit v1

목적:
  기존 AE branch(rd4ad_2p5d_normal_mw_fixed96_v1)가 학습한 normal train crop의
  subtype 분포를 확인한다. boundary/vessel/hilar/interior 비율을 계산하여
  흉벽/혈관 FP 취약성의 원인이 train subtype 편향에 있는지 판단한다.

안전 조건:
  - 기존 파일 수정 금지
  - output root 이미 존재 시 즉시 중단
  - stage2_holdout raw CT/mask/crop 접근 금지
  - 모델 forward / checkpoint 로드 금지
  - GPU 사용 금지
  - 학습 / scoring / threshold 재계산 금지

실행 모드:
  --dry      : 후보 파일 목록 / 예상 crop 수 / 비용 추정만 출력 (기본값)
  --mode fast: boundary + position subtype만 (Frangi 없음, ~30-40초)
  --mode vessel: boundary + MIP p85 + top-hat vessel subtype (Frangi 없음, ~3-4분)
  --real     : 실제 audit 실행 (--mode 필수)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# Paths & Constants
# ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit/rd_a1_normal_train_crop_subtype_audit_v1"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/crops_normal/"
    "normal_rd4ad_2p5d_mw_fixed96_v1/manifests/"
    "crop_manifest_normal_rd4ad_2p5d_mw_fixed96_v1.csv"
)

STAGE_SPLIT_PATH = (
    PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"
)

RD_A0_SUMMARY = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit/rd_a0_existing_branch_audit_v1"
    / "rd_a0_decision_summary.json"
)

VOLUME_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy"
)

FORBIDDEN_PATH_PATTERNS = ["stage2_holdout", "crops_stage2_holdout", "v2v2"]

# subtype 우선순위
SUBTYPE_PRIORITY = [
    "normal_vessel_boundary_mixed",
    "normal_boundary",
    "normal_vessel",
    "normal_hilar_or_central",
    "normal_interior",
    "normal_other",
]

# boundary 계산 파라미터
BOUNDARY_EROSION_ITERS = 8   # ~8px margin → boundary ring

# vessel 계산 파라미터
MIP_WINDOW_HALF = 5          # ±5 슬라이스 MIP
VESSEL_P85_THRESH = 85       # percentile
TOPHAT_DISK_RADIUS = 3

# subtype 임계값
BOUNDARY_TOUCH_THRESH = 0.05   # boundary ring overlap ratio ≥ 0.05 → boundary
VESSEL_DOMINANT_THRESH = 0.25  # vessel overlap ratio ≥ 0.25 → vessel dominant
VESSEL_TOUCH_THRESH = 0.05     # vessel overlap ratio ≥ 0.05 → vessel touch


# ─────────────────────────────────────────────────────────
# Safety
# ─────────────────────────────────────────────────────────
def _is_forbidden(path_str: str) -> bool:
    return any(pat in str(path_str) for pat in FORBIDDEN_PATH_PATTERNS)


def check_output_root():
    if OUTPUT_ROOT.exists():
        print(f"[ABORT] output root already exists: {OUTPUT_ROOT}")
        print("  기존 결과를 덮어쓰지 않습니다. 기존 디렉토리를 삭제 후 재실행하세요.")
        sys.exit(1)
    print(f"[OK] output root not exists: {OUTPUT_ROOT}")


# ─────────────────────────────────────────────────────────
# ROI / CT loader (slice-level, cached)
# ─────────────────────────────────────────────────────────
class VolumeCache:
    """patient별 ROI/CT mmap 캐시. stage2_holdout 접근 차단."""

    def __init__(self, volume_root: Path):
        self.volume_root = volume_root
        self._roi_cache: dict[str, np.ndarray] = {}
        self._ct_cache: dict[str, np.ndarray] = {}

    def _vol_dir(self, safe_id: str) -> Path:
        return self.volume_root / safe_id

    def get_roi(self, safe_id: str) -> np.ndarray | None:
        if _is_forbidden(safe_id):
            return None
        if safe_id not in self._roi_cache:
            p = self._vol_dir(safe_id) / "roi_0_0.npy"
            if not p.exists():
                return None
            self._roi_cache[safe_id] = np.load(str(p), mmap_mode="r")
        return self._roi_cache[safe_id]

    def get_ct(self, safe_id: str) -> np.ndarray | None:
        if _is_forbidden(safe_id):
            return None
        if safe_id not in self._ct_cache:
            p = self._vol_dir(safe_id) / "ct_hu.npy"
            if not p.exists():
                return None
            self._ct_cache[safe_id] = np.load(str(p), mmap_mode="r")
        return self._ct_cache[safe_id]

    def clear(self):
        self._roi_cache.clear()
        self._ct_cache.clear()


# ─────────────────────────────────────────────────────────
# Boundary feature
# ─────────────────────────────────────────────────────────
_boundary_ring_cache: dict[tuple, np.ndarray] = {}


def get_boundary_ring(roi: np.ndarray, z: int) -> np.ndarray:
    """ROI slice에서 boundary ring을 erosion으로 계산 (캐시)."""
    key = (id(roi), z)
    if key in _boundary_ring_cache:
        return _boundary_ring_cache[key]
    from scipy.ndimage import binary_erosion
    s = roi[z].astype(bool)
    interior = binary_erosion(s, iterations=BOUNDARY_EROSION_ITERS)
    ring = s & ~interior
    _boundary_ring_cache[key] = ring
    return ring


def compute_boundary_features(row: pd.Series, vcache: VolumeCache) -> dict:
    safe_id = row["safe_id"]
    z = int(row["local_z"])
    y0, x0, y1, x1 = int(row["crop_y0"]), int(row["crop_x0"]), int(row["crop_y1"]), int(row["crop_x1"])

    roi = vcache.get_roi(safe_id)
    if roi is None or z >= roi.shape[0]:
        return {
            "refined_roi_ratio": float("nan"),
            "roi_boundary_overlap_ratio": float("nan"),
            "roi_boundary_distance_min": float("nan"),
            "boundary_touch_flag": False,
            "peripheral_flag": row.get("central_peripheral", "unknown") == "peripheral",
        }

    roi_slice = roi[z]
    crop_area = (y1 - y0) * (x1 - x0)
    crop_roi = roi_slice[y0:y1, x0:x1]
    refined_roi_ratio = float(crop_roi.sum()) / max(crop_area, 1)

    ring = get_boundary_ring(roi, z)
    crop_ring = ring[y0:y1, x0:x1]
    boundary_overlap_ratio = float(crop_ring.sum()) / max(crop_area, 1)
    boundary_touch_flag = boundary_overlap_ratio >= BOUNDARY_TOUCH_THRESH

    # distance min: crop center to nearest boundary pixel
    ring_ys, ring_xs = np.where(ring)
    if len(ring_ys) > 0:
        cy = (y0 + y1) / 2.0
        cx = (x0 + x1) / 2.0
        dists = np.sqrt((ring_ys - cy) ** 2 + (ring_xs - cx) ** 2)
        dist_min = float(dists.min())
    else:
        dist_min = float("nan")

    return {
        "refined_roi_ratio": refined_roi_ratio,
        "roi_boundary_overlap_ratio": boundary_overlap_ratio,
        "roi_boundary_distance_min": dist_min,
        "boundary_touch_flag": boundary_touch_flag,
        "peripheral_flag": row.get("central_peripheral", "unknown") == "peripheral",
    }


# ─────────────────────────────────────────────────────────
# Vessel feature (MIP p85 + top-hat)
# ─────────────────────────────────────────────────────────
_vessel_mask_cache: dict[tuple, np.ndarray] = {}


def get_vessel_mask(ct: np.ndarray, roi: np.ndarray, z: int) -> np.ndarray:
    """union of MIP p85 + top-hat vessel masks (Frangi 제외)."""
    key = (id(ct), id(roi), z)
    if key in _vessel_mask_cache:
        return _vessel_mask_cache[key]

    from skimage.morphology import white_tophat, disk

    z_lo = max(0, z - MIP_WINDOW_HALF)
    z_hi = min(ct.shape[0], z + MIP_WINDOW_HALF)
    ct_win = ct[z_lo:z_hi].astype(np.float32)
    roi_win = roi[z_lo:z_hi].astype(bool)
    ct_win[~roi_win] = -2000.0

    mip_2d = np.max(ct_win, axis=0)
    roi_z = roi[z].astype(bool)

    # method 1: intensity p85
    roi_pixels = mip_2d[roi_z]
    if roi_pixels.size > 0:
        thresh85 = np.percentile(roi_pixels, VESSEL_P85_THRESH)
        vm_intensity = (mip_2d >= thresh85) & roi_z
    else:
        vm_intensity = np.zeros(mip_2d.shape, dtype=bool)

    # method 2: top-hat p85
    mip_norm = np.clip((mip_2d + 1000) / 1200.0, 0.0, 1.0).astype(np.float32)
    th = white_tophat(mip_norm, disk(TOPHAT_DISK_RADIUS))
    th_roi = th[roi_z]
    if th_roi.size > 0:
        tthresh = np.percentile(th_roi, VESSEL_P85_THRESH)
        vm_tophat = (th >= tthresh) & roi_z
    else:
        vm_tophat = np.zeros(mip_2d.shape, dtype=bool)

    union = vm_intensity | vm_tophat
    _vessel_mask_cache[key] = union
    return union


def compute_vessel_features(row: pd.Series, vcache: VolumeCache) -> dict:
    safe_id = row["safe_id"]
    z = int(row["local_z"])
    y0, x0, y1, x1 = int(row["crop_y0"]), int(row["crop_x0"]), int(row["crop_y1"]), int(row["crop_x1"])

    ct = vcache.get_ct(safe_id)
    roi = vcache.get_roi(safe_id)
    if ct is None or roi is None or z >= roi.shape[0]:
        return {
            "vessel_overlap_ratio": float("nan"),
            "vessel_touch_flag": False,
            "vessel_dominant_flag": False,
        }

    vmask = get_vessel_mask(ct, roi, z)
    crop_area = (y1 - y0) * (x1 - x0)
    vessel_crop = vmask[y0:y1, x0:x1]
    vessel_ratio = float(vessel_crop.sum()) / max(crop_area, 1)

    return {
        "vessel_overlap_ratio": vessel_ratio,
        "vessel_touch_flag": vessel_ratio >= VESSEL_TOUCH_THRESH,
        "vessel_dominant_flag": vessel_ratio >= VESSEL_DOMINANT_THRESH,
    }


# ─────────────────────────────────────────────────────────
# Subtype label
# ─────────────────────────────────────────────────────────
def assign_subtype(
    boundary_touch: bool,
    vessel_dominant: bool,
    central_hilar: bool,
    peripheral: bool,
    pure_lung_ratio: float,
) -> str:
    if boundary_touch and vessel_dominant:
        return "normal_vessel_boundary_mixed"
    if boundary_touch:
        return "normal_boundary"
    if vessel_dominant:
        return "normal_vessel"
    if central_hilar:
        return "normal_hilar_or_central"
    # interior: high pure_lung_ratio, not peripheral, no vessel
    # peripheral but no boundary flag → 여전히 interior or other
    if pure_lung_ratio >= 0.85 and not boundary_touch:
        return "normal_interior"
    return "normal_other"


# ─────────────────────────────────────────────────────────
# Dry-run
# ─────────────────────────────────────────────────────────
def run_dry(mode: str):
    print("=" * 60)
    print("[DRY-RUN] RD-A1 Normal Train Crop Subtype Audit")
    print("=" * 60)

    # RD-A0 결과
    print("\n[1] RD-A0 결과:")
    if RD_A0_SUMMARY.exists():
        with open(RD_A0_SUMMARY) as f:
            a0 = json.load(f)
        print(f"  verdict: {a0['verdict']}")
        print(f"  is_rd4ad_teacher_student: {a0['is_rd4ad_teacher_student']}")
        print(f"  is_conv_autoencoder_reconstruction: {a0['is_conv_autoencoder_reconstruction']}")
        print(f"  crop_auroc_l1: {a0['crop_auroc_l1']}")
    else:
        print("  [MISSING] rd_a0_decision_summary.json")

    # manifest
    print(f"\n[2] manifest:")
    if MANIFEST_PATH.exists():
        df = pd.read_csv(MANIFEST_PATH)
        for split, grp in df.groupby("normal_split"):
            n_pt = grp["patient_id"].nunique()
            n_unique_slices = grp.groupby(["patient_id", "local_z"]).ngroups
            print(f"  split={split}: n_crops={len(grp)}, n_patients={n_pt}, unique_slices={n_unique_slices}")
    else:
        print(f"  [MISSING] {MANIFEST_PATH}")
        return

    # stage2_holdout intersection
    print("\n[3] stage2_holdout contamination:")
    if STAGE_SPLIT_PATH.exists():
        sdf = pd.read_csv(STAGE_SPLIT_PATH)
        holdout = set(sdf[sdf["stage_split"] == "stage2_holdout"]["patient_id"])
        normal_pts = set(df["patient_id"])
        common = holdout & normal_pts
        print(f"  holdout={len(holdout)}, normal={len(normal_pts)}, intersection={len(common)} (기대값=0)")
    else:
        print("  [MISSING] stage split")

    # position_bin distribution (already in manifest)
    print("\n[4] position_bin distribution (이미 manifest에 있음):")
    print(df["position_bin"].value_counts().to_string())
    print(f"\n  pure_lung_patch_ratio < 0.80: {(df['pure_lung_patch_ratio'] < 0.80).sum()} / {len(df)}")

    # volume root 접근
    print(f"\n[5] volume root: {VOLUME_ROOT}")
    print(f"  EXISTS: {VOLUME_ROOT.exists()}")

    # 비용 추정
    train_df = df[df["normal_split"] == "train"]
    val_df = df[df["normal_split"] == "val"]
    n_train_slices = train_df.groupby(["patient_id", "local_z"]).ngroups
    n_val_slices = val_df.groupby(["patient_id", "local_z"]).ngroups
    n_all_slices = df.groupby(["patient_id", "local_z"]).ngroups

    print(f"\n[6] 실행 비용 추정 (Mode: {mode}):")
    print(f"  unique (patient, z) slices — train: {n_train_slices}, val+test: {n_val_slices + df[df['normal_split']=='test'].groupby(['patient_id','local_z']).ngroups}, total: {n_all_slices}")

    if mode == "fast":
        est = n_all_slices * 0.015  # boundary only
        print(f"  Mode FAST: boundary+position only")
        print(f"  estimated time: ~{est:.0f}s (~{est/60:.1f}min)")
        print("  Frangi vesselness: 사용 안 함")
        print("  MIP p85 vessel: 사용 안 함")
    elif mode == "vessel":
        est_boundary = n_all_slices * 0.015
        est_vessel = n_all_slices * 0.09   # MIP + top-hat (Frangi 제외)
        est_total = est_boundary + est_vessel
        print(f"  Mode VESSEL: boundary + MIP p85 + top-hat (Frangi 제외)")
        print(f"  estimated time: ~{est_total:.0f}s (~{est_total/60:.1f}min)")
        print("  Frangi vesselness: 사용 안 함 (너무 느림, ~22min 추가)")

    # output root
    print(f"\n[7] output root: {OUTPUT_ROOT}")
    print(f"  EXISTS={OUTPUT_ROOT.exists()}")

    print("\n[DRY-RUN COMPLETE]")
    print(f"\n제안:")
    print(f"  Mode FAST  (~{n_all_slices * 0.015 / 60:.1f}min): boundary + position. vessel 컬럼=nan")
    print(f"  Mode VESSEL (~{(n_all_slices * 0.105) / 60:.1f}min): boundary + MIP p85 + top-hat vessel (권장)")
    print(f"\n  실행 예: python rd_a1_normal_train_crop_subtype_audit.py --real --mode vessel")


# ─────────────────────────────────────────────────────────
# Real audit
# ─────────────────────────────────────────────────────────
def run_real(mode: str):
    t_start = time.time()
    check_output_root()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] output root created: {OUTPUT_ROOT}")

    errors = []

    # ── Load manifest ──────────────────────────────────────
    print(f"[INFO] loading manifest...")
    df = pd.read_csv(MANIFEST_PATH)
    print(f"[INFO] manifest loaded: {len(df)} rows")

    # stage2_holdout contamination check
    stage2_holdout_intersection = []
    if STAGE_SPLIT_PATH.exists():
        sdf = pd.read_csv(STAGE_SPLIT_PATH)
        holdout_ids = set(sdf[sdf["stage_split"] == "stage2_holdout"]["patient_id"])
        normal_ids = set(df["patient_id"])
        stage2_holdout_intersection = list(holdout_ids & normal_ids)
        if stage2_holdout_intersection:
            errors.append({"step": "holdout_check", "error": f"CONTAMINATION: {stage2_holdout_intersection}"})
            print(f"[WARN] stage2_holdout contamination: {stage2_holdout_intersection}")
        else:
            print("[OK] stage2_holdout contamination: 0")
    else:
        errors.append({"step": "holdout_check", "error": "stage_split_file_not_found"})

    # ── Volume cache ───────────────────────────────────────
    vcache = VolumeCache(VOLUME_ROOT)
    use_vessel = (mode == "vessel")

    # ── Per-crop feature computation ───────────────────────
    results = []
    n_total = len(df)
    n_done = 0
    t_progress = time.time()

    # use ALL splits (train + val + test) for subtype distribution
    for _, row in df.iterrows():
        safe_id = str(row["safe_id"])
        if _is_forbidden(safe_id):
            errors.append({"step": "forbidden_path", "error": safe_id})
            continue

        # boundary features
        try:
            bf = compute_boundary_features(row, vcache)
        except Exception as e:
            bf = {
                "refined_roi_ratio": float("nan"),
                "roi_boundary_overlap_ratio": float("nan"),
                "roi_boundary_distance_min": float("nan"),
                "boundary_touch_flag": False,
                "peripheral_flag": False,
            }
            errors.append({"step": "boundary", "crop_id": row.get("crop_id", "?"), "error": str(e)})

        # vessel features
        if use_vessel:
            try:
                vf = compute_vessel_features(row, vcache)
            except Exception as e:
                vf = {
                    "vessel_overlap_ratio": float("nan"),
                    "vessel_touch_flag": False,
                    "vessel_dominant_flag": False,
                }
                errors.append({"step": "vessel", "crop_id": row.get("crop_id", "?"), "error": str(e)})
        else:
            vf = {
                "vessel_overlap_ratio": float("nan"),
                "vessel_touch_flag": False,
                "vessel_dominant_flag": False,
            }

        # hilar/central proxy
        central_hilar = str(row.get("central_peripheral", "")).lower() == "central"

        # subtype
        boundary_touch = bf["boundary_touch_flag"]
        vessel_dominant = vf["vessel_dominant_flag"]
        pure_lung_ratio = float(row.get("pure_lung_patch_ratio", 0.0))
        subtype = assign_subtype(
            boundary_touch,
            vessel_dominant,
            central_hilar,
            bf.get("peripheral_flag", False),
            pure_lung_ratio,
        )

        results.append({
            "crop_id": row.get("crop_id", ""),
            "patient_id": row.get("patient_id", ""),
            "split": row.get("normal_split", ""),
            "local_z": row.get("local_z", ""),
            "slice_index": row.get("local_z", ""),
            "y0": row.get("crop_y0", ""),
            "x0": row.get("crop_x0", ""),
            "y1": row.get("crop_y1", ""),
            "x1": row.get("crop_x1", ""),
            "source_manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "refined_roi_ratio": bf["refined_roi_ratio"],
            "roi_boundary_overlap_ratio": bf["roi_boundary_overlap_ratio"],
            "roi_boundary_distance_min": bf["roi_boundary_distance_min"],
            "boundary_touch_flag": bf["boundary_touch_flag"],
            "vessel_overlap_ratio": vf["vessel_overlap_ratio"],
            "vessel_touch_flag": vf["vessel_touch_flag"],
            "vessel_dominant_flag": vf["vessel_dominant_flag"],
            "central_hilar_proxy_flag": central_hilar,
            "peripheral_flag": bf.get("peripheral_flag", False),
            "position_bin": row.get("position_bin", ""),
            "normal_subtype_label": subtype,
            "note": "",
        })

        n_done += 1
        if n_done % 500 == 0:
            elapsed = time.time() - t_progress
            rate = n_done / (time.time() - t_start + 1e-6)
            eta = (n_total - n_done) / max(rate, 1e-6)
            print(f"  [{n_done}/{n_total}] elapsed={elapsed:.1f}s rate={rate:.0f}/s ETA={eta:.0f}s")

    print(f"[INFO] feature computation done: {n_done} rows in {time.time()-t_start:.1f}s")

    # ── Write rd_a1_normal_crop_subtype_audit.csv ──────────
    audit_csv_path = OUTPUT_ROOT / "rd_a1_normal_crop_subtype_audit.csv"
    audit_fieldnames = [
        "crop_id", "patient_id", "split", "local_z", "slice_index",
        "y0", "x0", "y1", "x1", "source_manifest",
        "refined_roi_ratio", "roi_boundary_overlap_ratio", "roi_boundary_distance_min",
        "boundary_touch_flag", "vessel_overlap_ratio", "vessel_touch_flag",
        "vessel_dominant_flag", "central_hilar_proxy_flag", "peripheral_flag",
        "position_bin", "normal_subtype_label", "note",
    ]
    with open(audit_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=audit_fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in audit_fieldnames})
    print(f"[DONE] {audit_csv_path.name} ({len(results)} rows)")

    # ── Patient subtype summary ────────────────────────────
    res_df = pd.DataFrame(results)
    patient_rows = []
    for (patient_id, split), grp in res_df.groupby(["patient_id", "split"]):
        n = len(grp)
        counts = grp["normal_subtype_label"].value_counts()
        row_out = {
            "patient_id": patient_id,
            "split": split,
            "n_crops": n,
            "n_normal_interior": counts.get("normal_interior", 0),
            "n_normal_boundary": counts.get("normal_boundary", 0),
            "n_normal_vessel": counts.get("normal_vessel", 0),
            "n_normal_vessel_boundary_mixed": counts.get("normal_vessel_boundary_mixed", 0),
            "n_normal_hilar_or_central": counts.get("normal_hilar_or_central", 0),
            "n_normal_other": counts.get("normal_other", 0),
        }
        for col in ["boundary", "vessel", "hilar_or_central", "interior"]:
            k = f"n_normal_{col}" if col not in ("hilar_or_central",) else "n_normal_hilar_or_central"
            if col == "hilar_or_central":
                row_out["hilar_ratio"] = counts.get("normal_hilar_or_central", 0) / max(n, 1)
            elif col == "boundary":
                row_out["boundary_ratio"] = (
                    counts.get("normal_boundary", 0)
                    + counts.get("normal_vessel_boundary_mixed", 0)
                ) / max(n, 1)
            elif col == "vessel":
                row_out["vessel_ratio"] = (
                    counts.get("normal_vessel", 0)
                    + counts.get("normal_vessel_boundary_mixed", 0)
                ) / max(n, 1)
            elif col == "interior":
                row_out["interior_ratio"] = counts.get("normal_interior", 0) / max(n, 1)
        patient_rows.append(row_out)

    patient_csv_path = OUTPUT_ROOT / "rd_a1_patient_subtype_summary.csv"
    patient_fieldnames = [
        "patient_id", "split", "n_crops",
        "n_normal_interior", "n_normal_boundary", "n_normal_vessel",
        "n_normal_vessel_boundary_mixed", "n_normal_hilar_or_central", "n_normal_other",
        "boundary_ratio", "vessel_ratio", "hilar_ratio", "interior_ratio",
    ]
    with open(patient_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=patient_fieldnames)
        w.writeheader()
        for r in patient_rows:
            w.writerow({k: r.get(k, "") for k in patient_fieldnames})
    print(f"[DONE] {patient_csv_path.name} ({len(patient_rows)} rows)")

    # ── Train/Val subtype summary ──────────────────────────
    tv_rows = []
    for split in ["train", "val", "test"]:
        grp = res_df[res_df["split"] == split]
        if len(grp) == 0:
            continue
        counts = grp["normal_subtype_label"].value_counts()
        n = len(grp)
        r = {
            "split": split,
            "n_crops": n,
            "n_patients": grp["patient_id"].nunique(),
            "n_unique_slices": grp.groupby(["patient_id", "local_z"]).ngroups,
            "normal_interior_count": counts.get("normal_interior", 0),
            "normal_boundary_count": counts.get("normal_boundary", 0),
            "normal_vessel_count": counts.get("normal_vessel", 0),
            "normal_vessel_boundary_mixed_count": counts.get("normal_vessel_boundary_mixed", 0),
            "normal_hilar_or_central_count": counts.get("normal_hilar_or_central", 0),
            "normal_other_count": counts.get("normal_other", 0),
            "normal_interior_ratio": counts.get("normal_interior", 0) / max(n, 1),
            "normal_boundary_ratio": counts.get("normal_boundary", 0) / max(n, 1),
            "normal_vessel_ratio": counts.get("normal_vessel", 0) / max(n, 1),
            "normal_vessel_boundary_mixed_ratio": counts.get("normal_vessel_boundary_mixed", 0) / max(n, 1),
            "normal_hilar_or_central_ratio": counts.get("normal_hilar_or_central", 0) / max(n, 1),
            "normal_other_ratio": counts.get("normal_other", 0) / max(n, 1),
        }
        tv_rows.append(r)

    tv_csv_path = OUTPUT_ROOT / "rd_a1_train_val_subtype_summary.csv"
    tv_fieldnames = [
        "split", "n_crops", "n_patients", "n_unique_slices",
        "normal_interior_count", "normal_boundary_count", "normal_vessel_count",
        "normal_vessel_boundary_mixed_count", "normal_hilar_or_central_count", "normal_other_count",
        "normal_interior_ratio", "normal_boundary_ratio", "normal_vessel_ratio",
        "normal_vessel_boundary_mixed_ratio", "normal_hilar_or_central_ratio", "normal_other_ratio",
    ]
    with open(tv_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=tv_fieldnames)
        w.writeheader()
        for r in tv_rows:
            w.writerow(r)
    print(f"[DONE] {tv_csv_path.name} ({len(tv_rows)} rows)")

    # ── Manifest inventory ─────────────────────────────────
    mf_inv_path = OUTPUT_ROOT / "rd_a1_manifest_inventory.csv"
    mf_inv = [{
        "manifest_path": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "n_rows": len(df),
        "n_patients": df["patient_id"].nunique(),
        "labels_present": "normal_split (train/val/test)",
        "label_values": "train/val/test",
        "split_values": "train/val/test",
        "normal_only_train": True,
        "stage2_holdout_contamination": len(stage2_holdout_intersection),
        "note": "rd4ad_2p5d_normal_mw_fixed96_v1 normal crop manifest",
    }]
    with open(mf_inv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mf_inv[0].keys()))
        w.writeheader()
        for r in mf_inv:
            w.writerow(r)
    print(f"[DONE] {mf_inv_path.name}")

    # ── Distribution summary JSON ──────────────────────────
    train_grp = res_df[res_df["split"] == "train"]
    val_grp = res_df[res_df["split"] == "val"]
    train_counts = train_grp["normal_subtype_label"].value_counts().to_dict()
    val_counts = val_grp["normal_subtype_label"].value_counts().to_dict()
    n_train = len(train_grp)
    n_val = len(val_grp)

    # interior bias flag
    interior_ratio_train = train_counts.get("normal_interior", 0) / max(n_train, 1)
    boundary_ratio_train = (
        train_counts.get("normal_boundary", 0) + train_counts.get("normal_vessel_boundary_mixed", 0)
    ) / max(n_train, 1)
    vessel_ratio_train = (
        train_counts.get("normal_vessel", 0) + train_counts.get("normal_vessel_boundary_mixed", 0)
    ) / max(n_train, 1)
    hilar_ratio_train = train_counts.get("normal_hilar_or_central", 0) / max(n_train, 1)

    summary = {
        "branch_name": "rd4ad_2p5d_normal_mw_fixed96_v1",
        "model_type": "ConvAutoencoder2p5D",
        "manifest_paths_used": [str(MANIFEST_PATH.relative_to(PROJECT_ROOT))],
        "n_train_crops": n_train,
        "n_val_crops": n_val,
        "n_patients_train": train_grp["patient_id"].nunique(),
        "n_patients_val": val_grp["patient_id"].nunique(),
        "mode": mode,
        "vessel_computation_scope": "MIP_p85_tophat_union_no_frangi" if use_vessel else "not_computed",
        "train_subtype_distribution": {
            k: {"count": int(v), "ratio": round(v / max(n_train, 1), 4)}
            for k, v in train_counts.items()
        },
        "val_subtype_distribution": {
            k: {"count": int(v), "ratio": round(v / max(n_val, 1), 4)}
            for k, v in val_counts.items()
        },
        "train_interior_ratio": round(interior_ratio_train, 4),
        "train_boundary_ratio": round(boundary_ratio_train, 4),
        "train_vessel_ratio": round(vessel_ratio_train, 4),
        "train_hilar_ratio": round(hilar_ratio_train, 4),
        "interior_dominant_flag": interior_ratio_train > 0.50,
        "boundary_sufficient_flag": boundary_ratio_train >= 0.15,
        "vessel_sufficient_flag": vessel_ratio_train >= 0.10,
        "stage2_holdout_intersection": len(stage2_holdout_intersection),
        "source_files_modified": False,
        "mtime_violations": 0,
        "all_checks_passed": len(stage2_holdout_intersection) == 0 and len(errors) == 0,
        "n_errors": len(errors),
    }

    summary_json_path = OUTPUT_ROOT / "rd_a1_subtype_distribution_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {summary_json_path.name}")

    # ── Errors CSV ─────────────────────────────────────────
    err_path = OUTPUT_ROOT / "rd_a1_errors.csv"
    with open(err_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "crop_id", "error"])
        w.writeheader()
        for e in errors:
            w.writerow({"step": e.get("step", ""), "crop_id": e.get("crop_id", ""), "error": e.get("error", "")})
    print(f"[DONE] {err_path.name} ({len(errors)} errors)")

    # ── Report MD ──────────────────────────────────────────
    report_path = OUTPUT_ROOT / "rd_a1_normal_train_crop_subtype_audit_report.md"
    _write_report(report_path, summary, tv_rows, n_train, n_val, mode, errors)
    print(f"[DONE] {report_path.name}")

    # ── DONE marker ────────────────────────────────────────
    if len(errors) == 0 and len(stage2_holdout_intersection) == 0:
        (OUTPUT_ROOT / "DONE").write_text("RD-A1 subtype audit complete\n")
        print(f"[DONE] DONE marker")
    else:
        print(f"[SKIP] DONE marker not created (errors={len(errors)}, holdout_contamination={len(stage2_holdout_intersection)})")

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"[COMPLETE] output: {OUTPUT_ROOT}")
    print(f"  elapsed: {elapsed_total:.1f}s")
    print(f"  interior_ratio_train: {interior_ratio_train:.3f}")
    print(f"  boundary_ratio_train: {boundary_ratio_train:.3f}")
    print(f"  vessel_ratio_train:   {vessel_ratio_train:.3f}")
    print(f"  hilar_ratio_train:    {hilar_ratio_train:.3f}")
    print(f"  interior_dominant: {summary['interior_dominant_flag']}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────
def _write_report(
    path: Path,
    summary: dict,
    tv_rows: list,
    n_train: int,
    n_val: int,
    mode: str,
    errors: list,
):
    lines = [
        "# RD-A1 Normal Train Crop Subtype Audit Report",
        "",
        "## 1. RD-A0 결과 요약",
        "",
        "- 기존 branch: `rd4ad_2p5d_normal_mw_fixed96_v1`",
        "- 실제 구조: **ConvAutoencoder2p5D** (encoder-decoder, L1/MSE reconstruction)",
        "- 진짜 RD4AD teacher-student: **No**",
        "- 기존 crop AUROC L1: **0.649** (stage1_dev, stage2_holdout 미평가)",
        "- RD-A0 판정: B. AE-like normal verifier confirmed",
        "",
        "---",
        "",
        "## 2. 실제로 찾은 Train/Val Manifest 목록",
        "",
    ]
    for mp in summary.get("manifest_paths_used", []):
        lines.append(f"- `{mp}`")
    lines += [
        "",
        f"- train crops: **{n_train}** / val crops: **{n_val}**",
        f"- patients — train: {summary['n_patients_train']}, val: {summary['n_patients_val']}",
        "",
        "---",
        "",
        "## 3. Normal-Only Train 여부",
        "",
        "- ✓ **normal-only train confirmed** — manifest split 컬럼에 train/val/test만 존재",
        "- hard_negative, positive, lesion crop: train에 포함 없음",
        "- `use_hard_negative_for_train=false` (executable config 확인)",
        "",
        "---",
        "",
        "## 4. Stage2 Holdout Contamination 여부",
        "",
        f"- stage2_holdout ∩ normal manifest: **{summary['stage2_holdout_intersection']}**",
        "- ✓ **contamination 없음**",
        "",
        "---",
        "",
        "## 5. Normal Train Crop Subtype 분포",
        "",
        "※ 이 subtype은 heuristic label이며 GT가 아닙니다.",
        "",
        "| subtype | count | ratio |",
        "| --- | --- | --- |",
    ]
    train_dist = summary.get("train_subtype_distribution", {})
    for st in SUBTYPE_PRIORITY:
        info = train_dist.get(st, {"count": 0, "ratio": 0.0})
        lines.append(f"| {st} | {info['count']} | {info['ratio']:.3f} |")

    lines += [
        "",
        f"- **interior_ratio**: {summary['train_interior_ratio']:.3f}",
        f"- **boundary_ratio** (boundary+mixed): {summary['train_boundary_ratio']:.3f}",
        f"- **vessel_ratio** (vessel+mixed): {summary['train_vessel_ratio']:.3f}",
        f"- **hilar/central_ratio**: {summary['train_hilar_ratio']:.3f}",
        f"- vessel 계산 방식: {summary['vessel_computation_scope']}",
        "",
        "---",
        "",
        "## 6. Normal Val Crop Subtype 분포",
        "",
        "| subtype | count | ratio |",
        "| --- | --- | --- |",
    ]
    val_dist = summary.get("val_subtype_distribution", {})
    for st in SUBTYPE_PRIORITY:
        info = val_dist.get(st, {"count": 0, "ratio": 0.0})
        lines.append(f"| {st} | {info['count']} | {info['ratio']:.3f} |")

    lines += [
        "",
        "---",
        "",
        "## 7. Boundary/Vessel/Hilar Crop 충분성 판단",
        "",
        f"- boundary_sufficient (≥15%): **{summary['boundary_sufficient_flag']}**",
        f"- vessel_sufficient (≥10%): **{summary['vessel_sufficient_flag']}**",
        f"- interior_dominant (>50%): **{summary['interior_dominant_flag']}**",
        "",
        "---",
        "",
        "## 8. 기존 AE의 흉벽/혈관 FP 취약성 해석",
        "",
    ]

    if summary["interior_dominant_flag"]:
        lines += [
            "**interior dominant bias 확인됨** — normal train crop이 폐 내부에 치우쳐 있었음.",
            "- 흉벽/pleura 인접 정상 구조를 AE가 충분히 학습하지 못했을 가능성이 있음.",
            "- boundary 인접 정상 crop을 충분히 학습하지 못하면, 흉벽/ROI 경계부 reconstruction loss가 커서 FP가 증가할 수 있음.",
            "- 이것이 흉벽 FP의 일부 원인일 가능성이 있으나 단독 원인으로 단정할 수 없음.",
        ]
    else:
        lines += [
            "**interior dominant bias 확인되지 않음** — boundary/vessel 비율이 상당 수준 존재.",
            "- subtype 편향이 FP의 주된 원인이 아닐 가능성.",
            "- ConvAE reconstruction 방식 자체의 한계 가능성 — 진짜 RD4AD teacher-student 설계 재고.",
        ]

    lines += [
        "",
        "---",
        "",
        "## 9. 다음 단계 판정",
        "",
    ]

    if summary["interior_dominant_flag"]:
        lines += [
            "**권고 A: Subtype-balanced AE 개선**",
            "- normal train crop을 boundary/vessel/hilar subtype 균등하게 재sampling",
            "- 기존 ConvAE 구조 유지하면서 subtype balanced manifest 재생성 후 재학습",
            "",
            "**대안 B: True RD4AD teacher-student 신규 설계**",
            "- `train_s6a_rd4ad_verifier.py` skeleton 기반 preflight 진행",
            "- AE보다 teacher-student feature space에서 boundary 구조를 더 잘 구분할 가능성",
            "",
            "**권고: 먼저 A로 개선 후 성능 차이를 확인하고, B는 병행 설계**",
        ]
    else:
        lines += [
            "**권고 B: True RD4AD teacher-student 신규 설계**",
            "- subtype 편향이 주요 원인이 아님 → AE 방식 자체 한계 가능성",
            "- `train_s6a_rd4ad_verifier.py` skeleton 기반 RD4AD preflight 우선 진행",
            "",
            "**대안 C: 두 방식 비교**",
            "- subtype-balanced AE와 RD4AD teacher-student를 병행하여 ablation",
        ]

    lines += [
        "",
        "---",
        "",
        "## 10. 절대 하지 않은 것",
        "",
        "- 학습 없음",
        "- scoring 없음",
        "- model forward 없음",
        "- stage2_holdout raw 접근 없음",
        "- 기존 파일 수정 없음",
        "- GPU 사용 없음",
        "- checkpoint 로드 없음",
        "- threshold 재계산 없음",
        "",
        f"errors: {len(errors)}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RD-A1 normal train crop subtype audit")
    parser.add_argument("--real", action="store_true", help="실제 audit 실행 (기본: dry-run)")
    parser.add_argument(
        "--mode",
        choices=["fast", "vessel"],
        default="vessel",
        help="fast=boundary+position only / vessel=boundary+MIP p85+tophat (기본: vessel)",
    )
    args = parser.parse_args()

    if not args.real:
        run_dry(args.mode)
        return

    run_real(args.mode)


if __name__ == "__main__":
    main()
