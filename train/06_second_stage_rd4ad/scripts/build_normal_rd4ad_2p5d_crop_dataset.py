"""
build_normal_rd4ad_2p5d_crop_dataset.py

Normal 환자 기반 2.5D multi-window crop dataset을 생성한다.
- 입력: normal_sampling_manifest (18,100행)
- 출력: [6, 96, 96] float32 npz crop + crop manifest CSV + summary JSON
- 기본 채널: lung window (z-1, z, z+1) + mediastinal window (z-1, z, z+1)
- y0/x0/y1/x1는 32×32 patch bbox — 중심 기준 96×96 crop으로 재계산
- patch_y0/x0/y1/x1, patch_center_y/x, crop_y0/x0/y1/x1을 manifest에서 분리 기록
- v2 경로 접근 금지 (sampling manifest / source_ct_path)
- dry-run 모드에서 파일 생성 없음
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# ── 상수 ──────────────────────────────────────────────────────────────────────
EXPECTED_ROWS = 18100
EXPECTED_PATIENTS = 362
EXPECTED_TRAIN_ROWS = 14500
EXPECTED_VAL_ROWS = 1800
EXPECTED_TEST_ROWS = 1800
EXPECTED_TRAIN_PATIENTS = 290
EXPECTED_VAL_PATIENTS = 36
EXPECTED_TEST_PATIENTS = 36
PATCH_SIZE = 32
CROP_SIZE = 96

DEFAULT_VOLUME_ROOT = (
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1"
)
DEFAULT_METADATA_ROOT = (
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_v2_tslungguard_nochest"
)

REQUIRED_SAMPLING_COLS: list[str] = [
    "patient_id", "safe_id", "normal_split", "crop_id",
    "z_center", "y0", "x0", "y1", "x1", "crop_size",
    "source_ct_path", "source_patch_csv",
]

CHANNEL_NAMES = [
    "lung_z-1", "lung_z", "lung_z+1",
    "med_z-1", "med_z", "med_z+1",
]


# ── window clip / normalize ───────────────────────────────────────────────────

def apply_window(volume: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """HU clip 후 [0, 1] normalize. float32 반환."""
    clipped = np.clip(volume, hu_min, hu_max)
    if hu_max == hu_min:
        return np.zeros_like(clipped, dtype=np.float32)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)


# ── z 인덱스 계산 (nearest repeat padding) ───────────────────────────────────

def get_z_indices(z_center: int, z_range: int, z_max: int) -> tuple[list[int], bool]:
    """
    z_center ± z_range 인덱스를 반환한다.
    경계 초과 시 nearest slice repeat.
    반환: (z_indices_list, padding_applied)
    """
    z_list = list(range(z_center - z_range, z_center + z_range + 1))
    padded = False
    result = []
    for z in z_list:
        clamped = max(0, min(z, z_max - 1))
        if clamped != z:
            padded = True
        result.append(clamped)
    return result, padded


# ── patch bbox → crop 좌표 계산 ───────────────────────────────────────────────

def compute_crop_coords(
    patch_y0: int,
    patch_x0: int,
    patch_y1: int,
    patch_x1: int,
    crop_size: int,
    volume_H: int,
    volume_W: int,
) -> tuple[int, int, int, int]:
    """
    32×32 patch bbox 중심에서 crop_size×crop_size crop 좌표를 반환한다.
    volume bounds 초과 시 ValueError.
    반환: (crop_y0, crop_x0, crop_y1, crop_x1)
    """
    patch_cy = (patch_y0 + patch_y1) // 2
    patch_cx = (patch_x0 + patch_x1) // 2
    half = crop_size // 2
    crop_y0 = patch_cy - half
    crop_x0 = patch_cx - half
    crop_y1 = crop_y0 + crop_size
    crop_x1 = crop_x0 + crop_size
    if crop_y0 < 0 or crop_x0 < 0 or crop_y1 > volume_H or crop_x1 > volume_W:
        raise ValueError(
            f"crop 좌표가 volume 범위를 벗어납니다: "
            f"crop=({crop_y0},{crop_x0},{crop_y1},{crop_x1}), "
            f"volume H={volume_H}, W={volume_W}"
        )
    return crop_y0, crop_x0, crop_y1, crop_x1


# ── single crop 추출 ──────────────────────────────────────────────────────────

def extract_crop(
    ct_volume: np.ndarray,
    z_center: int,
    crop_y0: int,
    crop_x0: int,
    crop_y1: int,
    crop_x1: int,
    lung_window: tuple[float, float],
    med_window: tuple[float, float],
    z_range: int,
    crop_size: int,
) -> tuple[np.ndarray, dict]:
    """
    [6, crop_size, crop_size] float32 crop을 반환한다.
    전체 volume windowing 금지 — 2D crop 추출 후 windowing 적용.
    반환: (image_array, crop_meta_dict)
    """
    D, H, W = ct_volume.shape
    z_indices, z_padded = get_z_indices(int(z_center), z_range, D)

    channels = []
    for window_hu in [lung_window, med_window]:
        for zi in z_indices:
            raw_crop_2d = ct_volume[zi, crop_y0:crop_y1, crop_x0:crop_x1]
            if raw_crop_2d.shape != (crop_size, crop_size):
                raise ValueError(
                    f"crop shape 불일치: expected ({crop_size},{crop_size}), "
                    f"got {raw_crop_2d.shape} at z={zi}, "
                    f"crop_y0={crop_y0}, crop_x0={crop_x0}, "
                    f"crop_y1={crop_y1}, crop_x1={crop_x1}"
                )
            slice_2d = apply_window(raw_crop_2d, window_hu[0], window_hu[1])
            channels.append(slice_2d)

    image = np.stack(channels, axis=0)  # [6, H, W]

    if np.isnan(image).any():
        raise ValueError("crop에 NaN이 포함되어 있습니다.")
    if np.isinf(image).any():
        raise ValueError("crop에 Inf가 포함되어 있습니다.")

    crop_meta = {
        "z_indices_used": z_indices,
        "z_padding_mode": "nearest_repeat",
        "z_padding_applied": z_padded,
        "intensity_min": float(image.min()),
        "intensity_max": float(image.max()),
        "has_nan": bool(np.isnan(image).any()),
        "has_inf": bool(np.isinf(image).any()),
        "crop_mean": float(image.mean()),
        "crop_std": float(image.std()),
    }
    return image, crop_meta


# ── CT volume 로드 ────────────────────────────────────────────────────────────

def load_ct_volume(ct_path: Path) -> tuple[np.ndarray, Path]:
    """ct_hu.npy를 mmap_mode='r'로 로드한다."""
    if not ct_path.exists():
        raise FileNotFoundError(f"ct_hu.npy 없음: {ct_path}")
    volume = np.load(str(ct_path), mmap_mode="r")
    return volume, ct_path


# ── preflight ─────────────────────────────────────────────────────────────────

def run_preflight(
    args: argparse.Namespace,
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """실행 전 안전장치 검증. 문제 발견 시 sys.exit(1)."""

    volume_root = Path(args.volume_root)
    metadata_root = Path(args.metadata_root)
    volume_root_str = str(volume_root)
    metadata_root_str = str(metadata_root)

    # v2 경로 차단 (sampling_manifest, source_ct_path용)
    if "v2" in str(args.sampling_manifest).lower():
        print(f"[ABORT] sampling-manifest 경로에 v2가 포함됩니다: {args.sampling_manifest}")
        sys.exit(1)

    # 필수 컬럼 확인
    missing = [c for c in REQUIRED_SAMPLING_COLS if c not in df.columns]
    if missing:
        print(f"[ABORT] 필수 컬럼 누락: {missing}")
        sys.exit(1)

    # ── 보완 1: 전체 row 수 / patient 수 ──────────────────────────────────────
    n_rows = len(df)
    n_patients = df["patient_id"].nunique()
    print(f"[preflight] sampling manifest row 수: {n_rows}")
    print(f"[preflight] unique patient 수: {n_patients}")
    if n_rows != EXPECTED_ROWS:
        print(f"[ABORT] sampling manifest row 수 불일치: {n_rows} (기준 {EXPECTED_ROWS})")
        sys.exit(1)
    if n_patients != EXPECTED_PATIENTS:
        print(f"[ABORT] unique patient 수 불일치: {n_patients} (기준 {EXPECTED_PATIENTS})")
        sys.exit(1)

    # ── 보완 1: split별 row 수 / patient 수 guard ─────────────────────────────
    split_row_expected = {
        "train": EXPECTED_TRAIN_ROWS,
        "val": EXPECTED_VAL_ROWS,
        "test": EXPECTED_TEST_ROWS,
    }
    split_patient_expected = {
        "train": EXPECTED_TRAIN_PATIENTS,
        "val": EXPECTED_VAL_PATIENTS,
        "test": EXPECTED_TEST_PATIENTS,
    }
    for split, expected_rows in split_row_expected.items():
        actual_rows = int((df["normal_split"] == split).sum())
        print(f"[preflight] {split}: row {actual_rows} (기준 {expected_rows})")
        if actual_rows != expected_rows:
            print(f"[ABORT] {split} row 수 불일치: {actual_rows} ≠ {expected_rows}")
            sys.exit(1)
    for split, expected_pts in split_patient_expected.items():
        actual_pts = int(df[df["normal_split"] == split]["patient_id"].nunique())
        print(f"[preflight] {split}: patient {actual_pts} (기준 {expected_pts})")
        if actual_pts != expected_pts:
            print(f"[ABORT] {split} patient 수 불일치: {actual_pts} ≠ {expected_pts}")
            sys.exit(1)

    # normal_split 값 확인
    valid_splits = {"train", "val", "test"}
    actual_splits = set(df["normal_split"].astype(str).str.strip().unique())
    if not actual_splits.issubset(valid_splits):
        unexpected = sorted(actual_splits - valid_splits)
        print(f"[ABORT] normal_split에 예상 외 값: {unexpected}")
        sys.exit(1)

    # patch bbox guard: y1-y0 == 32, x1-x0 == 32
    h_check = (df["y1"] - df["y0"]) == PATCH_SIZE
    w_check = (df["x1"] - df["x0"]) == PATCH_SIZE
    if not h_check.all():
        bad = (~h_check).sum()
        print(f"[ABORT] patch bbox 높이가 {PATCH_SIZE}이 아닌 행 {bad}건")
        sys.exit(1)
    if not w_check.all():
        bad = (~w_check).sum()
        print(f"[ABORT] patch bbox 너비가 {PATCH_SIZE}이 아닌 행 {bad}건")
        sys.exit(1)

    # crop_size guard: == 96
    crop_size_check = df["crop_size"] == args.crop_size
    if not crop_size_check.all():
        bad = (~crop_size_check).sum()
        print(f"[ABORT] crop_size가 {args.crop_size}이 아닌 행 {bad}건")
        sys.exit(1)

    # crop_id 중복 확인
    if df["crop_id"].duplicated().any():
        dup_count = df["crop_id"].duplicated().sum()
        print(f"[ABORT] crop_id 중복 {dup_count}건")
        sys.exit(1)

    # ── 보완 3: sampling summary JSON read-only 확인 ──────────────────────────
    sampling_summary_path = Path(args.sampling_summary)
    if not sampling_summary_path.exists():
        print(f"[ABORT] sampling summary JSON 없음: {sampling_summary_path}")
        sys.exit(1)
    with open(sampling_summary_path, "r", encoding="utf-8") as f:
        ss = json.load(f)

    ss_n_total = ss.get("n_crops_total")
    ss_n_train = ss.get("n_crops_train")
    ss_n_val = ss.get("n_crops_val")
    ss_n_test = ss.get("n_crops_test")
    ss_crop_size = ss.get("crop_size")

    if ss_n_total != EXPECTED_ROWS:
        print(f"[ABORT] summary n_crops_total 불일치: {ss_n_total} (기준 {EXPECTED_ROWS})")
        sys.exit(1)
    if ss_n_train != EXPECTED_TRAIN_ROWS:
        print(f"[ABORT] summary n_crops_train 불일치: {ss_n_train} (기준 {EXPECTED_TRAIN_ROWS})")
        sys.exit(1)
    if ss_n_val != EXPECTED_VAL_ROWS:
        print(f"[ABORT] summary n_crops_val 불일치: {ss_n_val} (기준 {EXPECTED_VAL_ROWS})")
        sys.exit(1)
    if ss_n_test != EXPECTED_TEST_ROWS:
        print(f"[ABORT] summary n_crops_test 불일치: {ss_n_test} (기준 {EXPECTED_TEST_ROWS})")
        sys.exit(1)
    if ss_crop_size != args.crop_size:
        print(f"[ABORT] summary crop_size 불일치: {ss_crop_size} (기준 {args.crop_size})")
        sys.exit(1)

    # CSV ↔ summary 수치 일치 확인
    csv_n_train = int((df["normal_split"] == "train").sum())
    csv_n_val = int((df["normal_split"] == "val").sum())
    csv_n_test = int((df["normal_split"] == "test").sum())
    if csv_n_train != ss_n_train:
        print(f"[ABORT] CSV train {csv_n_train} ≠ summary n_crops_train {ss_n_train}")
        sys.exit(1)
    if csv_n_val != ss_n_val:
        print(f"[ABORT] CSV val {csv_n_val} ≠ summary n_crops_val {ss_n_val}")
        sys.exit(1)
    if csv_n_test != ss_n_test:
        print(f"[ABORT] CSV test {csv_n_test} ≠ summary n_crops_test {ss_n_test}")
        sys.exit(1)
    print("[preflight] sampling summary JSON 수치 일치 확인 완료")

    # ── 보완 2: source_ct_path volume root guard ───────────────────────────────
    print(f"[preflight] source_ct_path guard 시작 (전수 {len(df)}건)...")
    forbidden_subdirs = {"manifests", "patch_index_by_patient", "configs", "reports"}
    ct_path_errors: list[str] = []

    for i, (_, row) in enumerate(df.iterrows()):
        ct_str = str(row["source_ct_path"])
        ct_path_obj = Path(ct_str)

        # v2 경로 차단 (source_ct_path에 v2가 있으면 안 됨)
        if "v2" in ct_str.lower():
            ct_path_errors.append(f"[row {i}] source_ct_path에 v2 포함: {ct_str}")
            continue

        # volume root 하위 확인
        if not ct_str.startswith(volume_root_str):
            ct_path_errors.append(
                f"[row {i}] source_ct_path가 volume root 하위가 아님: {ct_str}"
            )
            continue

        # 상대경로 구조: volumes_npy/{safe_id}/ct_hu.npy
        try:
            rel = ct_path_obj.relative_to(volume_root)
            parts = rel.parts
        except ValueError:
            ct_path_errors.append(f"[row {i}] relative_to 실패: {ct_str}")
            continue

        if len(parts) != 3 or parts[0] != "volumes_npy" or parts[2] != "ct_hu.npy":
            ct_path_errors.append(
                f"[row {i}] 구조 불일치 (기준: volumes_npy/{{safe_id}}/ct_hu.npy): {rel}"
            )
            continue

        # 금지 subdir 접근 차단
        if parts[0] in forbidden_subdirs:
            ct_path_errors.append(f"[row {i}] 금지 subdir 접근: {parts[0]}")
            continue

        # 파일 존재 확인
        if not ct_path_obj.exists():
            ct_path_errors.append(f"[row {i}] ct_hu.npy 없음: {ct_str}")

    if ct_path_errors:
        print(f"[ABORT] source_ct_path guard 실패 {len(ct_path_errors)}건:")
        for e in ct_path_errors[:5]:
            print(f"  {e}")
        sys.exit(1)
    print(f"[preflight] source_ct_path guard 통과 ({len(df)}건)")

    # ── 보완 2: source_patch_csv metadata root guard ───────────────────────────
    print(f"[preflight] source_patch_csv guard 시작 (전수 {len(df)}건)...")
    patch_csv_errors: list[str] = []

    for i, (_, row) in enumerate(df.iterrows()):
        csv_str = str(row["source_patch_csv"])
        csv_path_obj = Path(csv_str)

        # volume root 하위면 즉시 중단
        if csv_str.startswith(volume_root_str):
            patch_csv_errors.append(
                f"[row {i}] source_patch_csv가 volume root 하위임 (금지): {csv_str}"
            )
            continue

        # metadata root 하위 확인
        if not csv_str.startswith(metadata_root_str):
            patch_csv_errors.append(
                f"[row {i}] source_patch_csv가 metadata root 하위가 아님: {csv_str}"
            )
            continue

        # 상대경로 구조: patch_index_by_patient/{safe_id}.csv
        try:
            rel = csv_path_obj.relative_to(metadata_root)
            parts = rel.parts
        except ValueError:
            patch_csv_errors.append(f"[row {i}] relative_to 실패: {csv_str}")
            continue

        if (
            len(parts) != 2
            or parts[0] != "patch_index_by_patient"
            or not parts[1].endswith(".csv")
        ):
            patch_csv_errors.append(
                f"[row {i}] 구조 불일치 (기준: patch_index_by_patient/{{safe_id}}.csv): {rel}"
            )
            continue

        # 파일 존재 확인
        if not csv_path_obj.exists():
            patch_csv_errors.append(f"[row {i}] source_patch_csv 없음: {csv_str}")

    if patch_csv_errors:
        print(f"[ABORT] source_patch_csv guard 실패 {len(patch_csv_errors)}건:")
        for e in patch_csv_errors[:5]:
            print(f"  {e}")
        sys.exit(1)
    print(f"[preflight] source_patch_csv guard 통과 ({len(df)}건)")

    # output dir overwrite guard
    if not args.dry_run and output_dir.exists() and not args.force:
        print(f"[ABORT] output dir 이미 존재합니다: {output_dir}")
        print("  --force 옵션을 사용하면 덮어쓸 수 있습니다.")
        sys.exit(1)

    print("[preflight] 전체 통과")


# ── main 처리 ─────────────────────────────────────────────────────────────────

def run_crop_pipeline(
    df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    환자 순서로 crop을 생성하고 결과를 반환한다.
    dry-run 모드에서는 파일 생성 없이 예상 결과만 집계한다.
    """
    lung_window = (float(args.lung_window[0]), float(args.lung_window[1]))
    med_window = (float(args.mediastinal_window[0]), float(args.mediastinal_window[1]))

    window_config = {
        "lung": {"hu_min": lung_window[0], "hu_max": lung_window[1]},
        "mediastinal": {"hu_min": med_window[0], "hu_max": med_window[1]},
    }
    window_config_str = json.dumps(window_config)
    crop_shape_str = f"[6, {args.crop_size}, {args.crop_size}]"
    input_channels_str = ",".join(CHANNEL_NAMES)
    dataset_tag = args.output_tag
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    manifest_rows: list[dict] = []
    failed_list: list[dict] = []
    boundary_failed_list: list[dict] = []  # 보완 5: boundary 실패 별도 추적
    z_padding_count = 0
    intensity_mins: list[float] = []
    intensity_maxs: list[float] = []
    means: list[float] = []
    stds: list[float] = []

    target_df = df.copy()
    # 1. --target-splits: split 필터
    if args.target_splits is not None:
        if not args.dry_run:
            print(f"[WARNING] --target-splits가 실제 실행에서 사용됩니다: {args.target_splits}")
        target_df = target_df[target_df["normal_split"].isin(args.target_splits)].copy()
        print(f"[info] --target-splits {args.target_splits} 적용 → {len(target_df)}행")
    # 2. --samples-per-split: split별 N개 선택 (deterministic, manifest 순서 유지)
    if args.samples_per_split is not None:
        parts = []
        for split in ["train", "val", "test"]:
            split_df = target_df[target_df["normal_split"] == split]
            if len(split_df) > 0:
                parts.append(split_df.head(args.samples_per_split))
        target_df = pd.concat(parts, ignore_index=False) if parts else pd.DataFrame(columns=df.columns)
        print(f"[info] --samples-per-split {args.samples_per_split} 적용 → {len(target_df)}행")
    # 3. --limit: 마지막에 전체 N개 제한
    if args.limit is not None:
        target_df = target_df.iloc[:args.limit].copy()
        print(f"[info] --limit {args.limit} 적용")

    n_expected = len(target_df)
    n_created = 0

    if not args.dry_run:
        for split in ("train", "val", "test"):
            (output_dir / "crops" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
        (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    # 환자별로 CT 로드 (같은 CT 반복 로드 방지)
    current_ct_str: Optional[str] = None
    current_volume: Optional[np.ndarray] = None
    current_ct_path: Optional[Path] = None

    for idx, row in target_df.iterrows():
        patient_id = str(row["patient_id"])
        safe_id = str(row["safe_id"])
        crop_id = str(row["crop_id"])
        normal_split = str(row["normal_split"])
        z_center = int(row["z_center"])
        patch_y0 = int(row["y0"])
        patch_x0 = int(row["x0"])
        patch_y1 = int(row["y1"])
        patch_x1 = int(row["x1"])

        # CT path: manifest의 source_ct_path 사용 (preflight에서 guard 통과)
        ct_path = Path(str(row["source_ct_path"]))

        # CT 로드 (경로 바뀔 때만)
        ct_path_str = str(ct_path)
        try:
            if ct_path_str != current_ct_str:
                current_volume, current_ct_path = load_ct_volume(ct_path)
                current_ct_str = ct_path_str
        except Exception as e:
            msg = f"CT 로드 실패 [{patient_id}/{crop_id}]: {e}"
            print(f"[FAIL] {msg}")
            failed_list.append({"crop_id": crop_id, "patient_id": patient_id, "reason": msg})
            continue

        # ── 보완 4: patch center 계산 ─────────────────────────────────────────
        patch_center_y = (patch_y0 + patch_y1) // 2
        patch_center_x = (patch_x0 + patch_x1) // 2

        # crop 좌표 계산 (32×32 patch bbox → 96×96 crop)
        try:
            D, H, W = current_volume.shape
            crop_y0, crop_x0, crop_y1, crop_x1 = compute_crop_coords(
                patch_y0, patch_x0, patch_y1, patch_x1,
                crop_size=args.crop_size,
                volume_H=H,
                volume_W=W,
            )
        except Exception as e:
            msg = f"crop 좌표 계산 실패 [{patient_id}/{crop_id}]: {e}"
            print(f"[FAIL] {msg}")
            failed_entry = {"crop_id": crop_id, "patient_id": patient_id, "reason": msg}
            failed_list.append(failed_entry)
            # 보완 5: boundary 실패 별도 기록
            boundary_failed_list.append(failed_entry)
            continue

        # crop 추출
        try:
            image, crop_meta = extract_crop(
                ct_volume=current_volume,
                z_center=z_center,
                crop_y0=crop_y0,
                crop_x0=crop_x0,
                crop_y1=crop_y1,
                crop_x1=crop_x1,
                lung_window=lung_window,
                med_window=med_window,
                z_range=args.z_range,
                crop_size=args.crop_size,
            )
        except Exception as e:
            msg = f"crop 추출 실패 [{patient_id}/{crop_id}]: {e}"
            print(f"[FAIL] {msg}")
            failed_list.append({"crop_id": crop_id, "patient_id": patient_id, "reason": msg})
            continue

        if crop_meta["z_padding_applied"]:
            z_padding_count += 1

        intensity_mins.append(crop_meta["intensity_min"])
        intensity_maxs.append(crop_meta["intensity_max"])
        means.append(crop_meta["crop_mean"])
        stds.append(crop_meta["crop_std"])

        # npz 저장 경로: crops/{normal_split}/{safe_id}/{crop_id}.npz
        npz_dir = output_dir / "crops" / normal_split / safe_id
        npz_path = npz_dir / f"{crop_id}.npz"

        if not args.dry_run:
            npz_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(npz_path), image=image)
        n_created += 1

        # manifest 행 구성
        manifest_row = row.to_dict()
        # patch bbox (원본 32×32) 분리 기록
        manifest_row["patch_y0"] = patch_y0
        manifest_row["patch_x0"] = patch_x0
        manifest_row["patch_y1"] = patch_y1
        manifest_row["patch_x1"] = patch_x1
        # 보완 4: patch center 기록
        manifest_row["patch_center_y"] = patch_center_y
        manifest_row["patch_center_x"] = patch_center_x
        # crop bbox (96×96) 분리 기록
        manifest_row["crop_y0"] = crop_y0
        manifest_row["crop_x0"] = crop_x0
        manifest_row["crop_y1"] = crop_y1
        manifest_row["crop_x1"] = crop_x1
        # 추가 메타
        manifest_row["crop_path"] = str(npz_path) if not args.dry_run else "[dry-run]"
        manifest_row["crop_dataset_tag"] = dataset_tag
        manifest_row["input_channels"] = input_channels_str
        manifest_row["crop_shape"] = crop_shape_str
        manifest_row["window_config"] = window_config_str
        manifest_row["z_indices_used"] = json.dumps(crop_meta["z_indices_used"])
        manifest_row["z_padding_mode"] = crop_meta["z_padding_mode"]
        manifest_row["intensity_min"] = crop_meta["intensity_min"]
        manifest_row["intensity_max"] = crop_meta["intensity_max"]
        manifest_row["has_nan"] = crop_meta["has_nan"]
        manifest_row["has_inf"] = crop_meta["has_inf"]
        manifest_row["crop_mean"] = crop_meta["crop_mean"]
        manifest_row["crop_std"] = crop_meta["crop_std"]
        manifest_row["source_ct_path"] = str(current_ct_path)
        manifest_row["created_at"] = timestamp
        manifest_rows.append(manifest_row)

    result_df = pd.DataFrame(manifest_rows) if manifest_rows else pd.DataFrame()

    split_created: dict[str, int] = {}
    if len(result_df) > 0 and "normal_split" in result_df.columns:
        split_created = result_df["normal_split"].value_counts().to_dict()

    summary = {
        "script": "build_normal_rd4ad_2p5d_crop_dataset.py",
        "timestamp": timestamp,
        "sampling_manifest_path": str(Path(args.sampling_manifest).resolve()),
        "output_tag": dataset_tag,
        "n_source_rows": len(df),
        "n_patients": df["patient_id"].nunique(),
        "normal_split_source_counts": df["normal_split"].value_counts().to_dict(),
        "n_crops_expected": n_expected,
        "n_crops_created": n_created,
        "normal_split_created_counts": split_created,
        "crop_shape": crop_shape_str,
        "dtype": "float32",
        "window_config": window_config,
        "z_padding_mode": "nearest_repeat",
        "z_padding_count": z_padding_count,
        "intensity_min_global": float(min(intensity_mins)) if intensity_mins else None,
        "intensity_max_global": float(max(intensity_maxs)) if intensity_maxs else None,
        "crop_mean_global": float(np.mean(means)) if means else None,
        "crop_std_global": float(np.mean(stds)) if stds else None,
        "failed_count": len(failed_list),
        "failed_examples": failed_list[:10],
        # 보완 5: boundary 실패 요약
        "boundary_failed_count": len(boundary_failed_list),
        "boundary_failed_examples": boundary_failed_list[:10],
        "dry_run": args.dry_run,
        "note": [
            "Normal 환자 기반 crop dataset",
            "y0/x0/y1/x1는 32×32 patch bbox — 중심 기준 96×96 crop으로 재계산",
            "patch_y0/x0/y1/x1 및 crop_y0/x0/y1/x1을 manifest에서 분리 기록",
            "patch_center_y/x는 patch bbox 중심 좌표",
            "lesion mask는 train input으로 사용하지 않음",
        ],
    }

    return result_df, summary


# ── disk 용량 추정 ────────────────────────────────────────────────────────────

def estimate_disk_usage(n_crops: int, mb_per_crop: float = 0.25) -> tuple[float, float]:
    """예상 용량(MB)과 현재 여유 공간(MB)을 반환한다."""
    estimated_mb = n_crops * mb_per_crop
    free_bytes = shutil.disk_usage(REPO).free
    free_mb = free_bytes / (1024 * 1024)
    return estimated_mb, free_mb


# ── argparse ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normal 환자 기반 2.5D multi-window crop dataset 생성"
    )
    parser.add_argument(
        "--sampling-manifest",
        default=str(
            REPO
            / "outputs/second-stage-lesion-refiner-v1/normal_sampling"
            / "normal_patch_index_position_balanced_fixed96_v1"
            / "normal_sampling_manifest_normal_patch_index_position_balanced_fixed96_v1.csv"
        ),
        help="입력 normal sampling manifest 경로",
    )
    parser.add_argument(
        "--sampling-summary",
        default=str(
            REPO
            / "outputs/second-stage-lesion-refiner-v1/normal_sampling"
            / "normal_patch_index_position_balanced_fixed96_v1"
            / "normal_sampling_manifest_normal_patch_index_position_balanced_fixed96_v1_summary.json"
        ),
        help="sampling summary JSON 경로 (read-only 확인용)",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO / "outputs/second-stage-lesion-refiner-v1/crops_normal"),
        help="출력 root 디렉토리",
    )
    parser.add_argument(
        "--output-tag",
        default="normal_rd4ad_2p5d_mw_fixed96_v1",
        help="출력 태그 (하위 폴더명 및 파일명에 사용)",
    )
    parser.add_argument(
        "--volume-root",
        default=DEFAULT_VOLUME_ROOT,
        help="CT volume root 경로 (source_ct_path guard 기준)",
    )
    parser.add_argument(
        "--metadata-root",
        default=DEFAULT_METADATA_ROOT,
        help="patch CSV metadata root 경로 (source_patch_csv guard 기준)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=CROP_SIZE,
        help=f"crop 크기 (default: {CROP_SIZE})",
    )
    parser.add_argument(
        "--z-range",
        type=int,
        default=1,
        help="z_center ± z_range (default: 1 → z-1, z, z+1)",
    )
    parser.add_argument(
        "--lung-window",
        type=float,
        nargs=2,
        default=[-1350.0, 150.0],
        metavar=("HU_MIN", "HU_MAX"),
        help="lung window HU 범위 (default: -1350 150)",
    )
    parser.add_argument(
        "--mediastinal-window",
        type=float,
        nargs=2,
        default=[-160.0, 240.0],
        metavar=("HU_MIN", "HU_MAX"),
        help="mediastinal window HU 범위 (default: -160 240)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="dry-run 모드: 파일 생성 없이 예상 결과만 출력",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="debug용: 처음 N개 후보만 처리 (target-splits/samples-per-split 적용 후 마지막에 전체 제한)",
    )
    parser.add_argument(
        "--target-splits",
        nargs="*",
        choices=["train", "val", "test"],
        default=None,
        help="지정된 split만 처리 (dry-run/실제 실행 모두 사용 가능, 실제 실행 시 경고 출력)",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=None,
        help="각 split에서 앞 N개만 선택, deterministic (dry-run에서 주로 사용)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 output dir이 있어도 덮어쓰기 허용",
    )
    parser.add_argument(
        "--no-runtime-append",
        action="store_true",
        help="runtime_summary.csv 기록 생략",
    )
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    manifest_path = Path(args.sampling_manifest)
    output_root = Path(args.output_root)
    output_dir = output_root / args.output_tag
    out_csv_name = f"crop_manifest_{args.output_tag}.csv"
    out_json_name = f"crop_summary_{args.output_tag}.json"
    out_csv = output_dir / "manifests" / out_csv_name
    out_json = output_dir / "reports" / out_json_name

    if not manifest_path.exists():
        print(f"[ABORT] sampling manifest 없음: {manifest_path}")
        sys.exit(1)

    print(f"[load] {manifest_path}")
    df = pd.read_csv(manifest_path)

    run_preflight(args, df, output_dir)

    # n_target 계산 (disk 추정용 — run_crop_pipeline 내 target_df 구성과 동일 순서 적용)
    _tmp = df.copy()
    if args.target_splits is not None:
        _tmp = _tmp[_tmp["normal_split"].isin(args.target_splits)]
    if args.samples_per_split is not None:
        _parts = []
        for _split in ["train", "val", "test"]:
            _s = _tmp[_tmp["normal_split"] == _split]
            if len(_s) > 0:
                _parts.append(_s.head(args.samples_per_split))
        _tmp = pd.concat(_parts, ignore_index=False) if _parts else pd.DataFrame()
    n_target = len(_tmp) if args.limit is None else min(args.limit, len(_tmp))
    est_mb, free_mb = estimate_disk_usage(n_target)
    print(f"[disk] 예상 용량: {est_mb:.1f} MB (0.25 MB × {n_target}개) / 여유 공간: {free_mb:.1f} MB")
    if not args.dry_run and free_mb < est_mb * 2:
        print(f"[ABORT] 디스크 여유 공간 부족: 여유 {free_mb:.1f} MB < 예상 {est_mb:.1f} MB × 2")
        sys.exit(1)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = output_root / f".tmp_{args.output_tag}_{run_timestamp}"
    write_dir = temp_dir if not args.dry_run else output_dir

    print(f"[crop] 2.5D crop {'(dry-run)' if args.dry_run else '실제 생성 (임시 dir 사용)'} 시작...")
    result_df, summary = run_crop_pipeline(df, args, write_dir)

    n_created = summary["n_crops_created"]
    n_failed = summary["failed_count"]
    n_expected = summary["n_crops_expected"]
    print(f"[crop] 생성 완료: {n_created}개 / 실패: {n_failed}개")
    print(f"[crop] z padding 발생: {summary['z_padding_count']}건")
    print(f"[crop] boundary 실패: {summary['boundary_failed_count']}건")

    if n_failed > 0:
        print(f"[WARNING] 실패 crop {n_failed}건. 예시: {summary['failed_examples'][:3]}")

    if args.dry_run:
        print("\n[dry-run] 파일 생성 없음. 위 예상 결과만 출력합니다.")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    # ── 실제 실행 후 partial output 방지 검증 ────────────────────────────────

    if n_failed > 0:
        print(f"[ABORT] 실패 crop {n_failed}건 → 최종 output dir 생성 금지.")
        print(f"  실패 목록: {json.dumps(summary['failed_examples'], ensure_ascii=False)}")
        print(f"  임시 dir (자동 삭제 안 함): {temp_dir}")
        sys.exit(1)

    n_npz = len(list((temp_dir / "crops").rglob("*.npz")))
    if n_npz != n_expected:
        print(f"[ABORT] npz 파일 수 불일치: 실제 {n_npz}건 ≠ 예상 {n_expected}건 → 최종 output dir 생성 금지.")
        print(f"  임시 dir (자동 삭제 안 함): {temp_dir}")
        sys.exit(1)

    if len(result_df) != n_expected:
        print(f"[ABORT] manifest row 수 불일치: {len(result_df)} ≠ {n_expected} → 최종 output dir 생성 금지.")
        print(f"  임시 dir (자동 삭제 안 함): {temp_dir}")
        sys.exit(1)

    if n_created != n_expected:
        print(f"[ABORT] n_crops_created 불일치: {n_created} ≠ {n_expected} → 최종 output dir 생성 금지.")
        print(f"  임시 dir (자동 삭제 안 함): {temp_dir}")
        sys.exit(1)

    summary["output_dir"] = str(output_dir)
    summary["temp_output_dir"] = str(temp_dir)
    summary["partial_output_policy"] = "temp_dir_then_rename_if_success"
    summary["n_npz_files_written"] = n_npz
    summary["finalized_successfully"] = True
    summary["finalized_output_dir"] = str(output_dir)

    temp_csv = temp_dir / "manifests" / out_csv_name
    temp_json_path = temp_dir / "reports" / out_json_name

    print(f"\n[save] crop manifest CSV → {temp_csv} (임시)")
    result_df.to_csv(temp_csv, index=False)

    print(f"[save] crop summary JSON → {temp_json_path} (임시)")
    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # temp dir → 최종 output dir rename
    if output_dir.exists():
        print(f"[ABORT] rename 시도 시 output_dir 이미 존재합니다: {output_dir}")
        print(f"  output_dir를 수동으로 정리 후 {temp_dir}를 rename 하세요.")
        sys.exit(1)
    print(f"\n[finalize] {temp_dir.name} → {output_dir.name}")
    temp_dir.rename(output_dir)

    # crop_path 경로 교정 (temp_dir → output_dir)
    manifest_path_final = output_dir / "manifests" / out_csv_name
    if manifest_path_final.exists():
        mdf = pd.read_csv(manifest_path_final)
        if "crop_path" in mdf.columns:
            mdf["crop_path"] = mdf["crop_path"].str.replace(
                str(temp_dir), str(output_dir), regex=False
            )
            mdf.to_csv(manifest_path_final, index=False)
            print("[finalize] crop_path 경로 교정 완료")

    print(f"\n[done] crop dataset 생성 완료.")
    print(f"  crops dir : {output_dir / 'crops'}")
    print(f"  manifest  : {out_csv}")
    print(f"  summary   : {out_json}")


if __name__ == "__main__":
    main()
