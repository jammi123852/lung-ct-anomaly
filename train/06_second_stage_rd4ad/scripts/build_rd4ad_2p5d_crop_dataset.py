"""
build_rd4ad_2p5d_crop_dataset.py

RD4AD 학습용 2.5D multi-window crop dataset을 생성한다.
- 입력: rd4ad_train sampling manifest (normal_like 7,700개)
- 출력: [6, 96, 96] float32 npz crop + crop manifest CSV + summary JSON
- 기본 채널: lung window (z-1, z, z+1) + mediastinal window (z-1, z, z+1)
- lesion mask는 train input으로 사용하지 않음
- stage2_holdout / 봉인 환자 차단
- v2 경로 접근 금지
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
import yaml

REPO = Path(__file__).resolve().parents[1]

# ── 상수 ──────────────────────────────────────────────────────────────────────
SEALED_PATIENTS: set[str] = {"LUNG1-089", "LUNG1-231", "LUNG1-372"}

REQUIRED_SAMPLING_COLS: list[str] = [
    "patient_id", "group", "stage_split", "safe_id", "candidate_id",
    "rd4ad_label", "binary_label", "selected_for_rd4ad_train",
    "max_padim_score", "bbox_too_large",
    "z_center",
    "y0_fixed_crop", "x0_fixed_crop", "y1_fixed_crop", "x1_fixed_crop",
]

CROP_MANIFEST_EXTRA_COLS: list[str] = [
    "crop_path", "crop_dataset_tag", "input_channels", "crop_shape",
    "window_config", "z_indices_used", "z_padding_mode",
    "intensity_min", "intensity_max", "has_nan", "has_inf",
    "crop_mean", "crop_std", "source_ct_path", "created_at",
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


# ── single crop 추출 ──────────────────────────────────────────────────────────

def extract_crop(
    ct_volume: np.ndarray,
    z_center: int,
    y0: int, x0: int, y1: int, x1: int,
    lung_window: tuple[float, float],
    med_window: tuple[float, float],
    z_range: int,
    crop_size: int,
) -> tuple[np.ndarray, dict]:
    """
    [6, crop_size, crop_size] float32 crop을 반환한다.
    반환: (image_array, crop_meta_dict)
    성능: windowing을 전체 volume이 아닌 2D crop 단위로 적용하여 I/O/CPU 비용 감소.
    """
    D, H, W = ct_volume.shape
    z_indices, z_padded = get_z_indices(int(z_center), z_range, D)

    channels = []
    for window_hu, win_name in [
        (lung_window, "lung"),
        (med_window, "med"),
    ]:
        for zi in z_indices:
            # 전체 volume windowing 금지 — 2D crop 추출 후 windowing 적용
            raw_crop_2d = ct_volume[zi, y0:y1, x0:x1]
            # shape guard: windowing 전에 확인
            if raw_crop_2d.shape != (crop_size, crop_size):
                raise ValueError(
                    f"crop shape 불일치: expected ({crop_size},{crop_size}), "
                    f"got {raw_crop_2d.shape} at z={zi}, y0={y0}, x0={x0}, y1={y1}, x1={x1}"
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


# ── dataset root 로드 ─────────────────────────────────────────────────────────

def load_dataset_root(args: argparse.Namespace) -> Path:
    """configs/paths.local.yaml에서 nsclc_msd_usable_only_v1 경로를 읽는다."""
    if args.dataset_root:
        path = Path(args.dataset_root)
        if "v2" in str(path).lower():
            print(f"[ABORT] dataset_root에 v2가 포함되어 있습니다: {path}")
            sys.exit(1)
        return path

    yaml_path = REPO / "configs" / "paths.local.yaml"
    if not yaml_path.exists():
        print(f"[ABORT] configs/paths.local.yaml 없음: {yaml_path}")
        sys.exit(1)

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    root_str = cfg.get("nsclc_msd_usable_only_v1", "")
    if not root_str:
        print("[ABORT] paths.local.yaml에 nsclc_msd_usable_only_v1 경로가 없습니다.")
        sys.exit(1)

    if "v2" in root_str.lower():
        print(f"[ABORT] nsclc_msd_usable_only_v1 경로에 v2가 포함되어 있습니다: {root_str}")
        sys.exit(1)

    return Path(root_str)


# ── preflight ─────────────────────────────────────────────────────────────────

def run_preflight(
    args: argparse.Namespace,
    df: pd.DataFrame,
    dataset_root: Path,
    output_dir: Path,
) -> None:
    """실행 전 안전장치 검증. 문제 발견 시 sys.exit(1)."""

    # v2 경로 차단
    if "v2" in str(args.sampling_manifest).lower():
        print(f"[ABORT] sampling-manifest 경로에 v2가 포함됩니다: {args.sampling_manifest}")
        sys.exit(1)

    # 필수 컬럼 확인
    missing = [c for c in REQUIRED_SAMPLING_COLS if c not in df.columns]
    if missing:
        print(f"[ABORT] 필수 컬럼 누락: {missing}")
        sys.exit(1)

    # row / patient 수
    n_rows = len(df)
    n_patients = df["patient_id"].nunique()
    print(f"[preflight] sampling manifest row 수: {n_rows}")
    print(f"[preflight] unique patient 수: {n_patients}")
    if n_rows != 7700:
        print(f"[ABORT] sampling manifest row 수 불일치: {n_rows} (기준 7700)")
        sys.exit(1)
    if n_patients != 154:
        print(f"[ABORT] unique patient 수 불일치: {n_patients} (기준 154)")
        sys.exit(1)

    # rd4ad_label / binary_label / selected_for_rd4ad_train
    if set(df["rd4ad_label"].unique()) != {"normal_like"}:
        print(f"[ABORT] rd4ad_label에 normal_like 외 값이 있습니다: {df['rd4ad_label'].unique().tolist()}")
        sys.exit(1)
    if "binary_label" in df.columns and set(df["binary_label"].unique()) != {"hard_negative"}:
        print(f"[ABORT] binary_label에 hard_negative 외 값이 있습니다: {df['binary_label'].unique().tolist()}")
        sys.exit(1)
    if "selected_for_rd4ad_train" in df.columns and not df["selected_for_rd4ad_train"].all():
        print("[ABORT] selected_for_rd4ad_train이 False인 행이 있습니다.")
        sys.exit(1)

    # stage_split guard
    if "stage_split" not in df.columns:
        print("[ABORT] stage_split 컬럼이 없습니다.")
        sys.exit(1)
    if df["stage_split"].isna().any():
        nan_pts = df[df["stage_split"].isna()]["patient_id"].unique().tolist()[:5]
        print(f"[ABORT] stage_split에 NaN이 있습니다. 환자 예시: {nan_pts}")
        sys.exit(1)
    unique_splits = set(df["stage_split"].astype(str).str.strip().unique())
    if unique_splits != {"stage1_dev"}:
        unexpected = sorted(unique_splits - {"stage1_dev"})
        pts = df[df["stage_split"].astype(str).str.strip().isin(unexpected)]["patient_id"].unique().tolist()[:5]
        print(f"[ABORT] stage_split에 stage1_dev 외 값: {sorted(unique_splits)}, 환자 예시: {pts}")
        sys.exit(1)

    # stage2_holdout 차단
    if (df["stage_split"] == "stage2_holdout").any():
        pts = df[df["stage_split"] == "stage2_holdout"]["patient_id"].unique().tolist()
        print(f"[ABORT] stage2_holdout 환자 포함: {pts}")
        sys.exit(1)

    # 봉인 환자 차단
    sealed_found = set(df["patient_id"].unique()) & SEALED_PATIENTS
    if sealed_found:
        print(f"[ABORT] 봉인 환자 포함: {sorted(sealed_found)}")
        sys.exit(1)

    # candidate_id 중복 확인
    if df["candidate_id"].duplicated().any():
        dup_count = df["candidate_id"].duplicated().sum()
        print(f"[ABORT] candidate_id 중복 {dup_count}건")
        sys.exit(1)

    # crop 좌표 96×96 확인
    coord_cols = ["y0_fixed_crop", "x0_fixed_crop", "y1_fixed_crop", "x1_fixed_crop"]
    if all(c in df.columns for c in coord_cols):
        h_check = (df["y1_fixed_crop"] - df["y0_fixed_crop"]) == args.crop_size
        w_check = (df["x1_fixed_crop"] - df["x0_fixed_crop"]) == args.crop_size
        if not h_check.all():
            bad = (~h_check).sum()
            print(f"[ABORT] fixed crop 높이가 {args.crop_size}이 아닌 행 {bad}건")
            sys.exit(1)
        if not w_check.all():
            bad = (~w_check).sum()
            print(f"[ABORT] fixed crop 너비가 {args.crop_size}이 아닌 행 {bad}건")
            sys.exit(1)

    # dataset root 존재 확인
    volumes_root = dataset_root / "volumes_npy"
    if not volumes_root.exists():
        print(f"[ABORT] volumes_npy 디렉토리 없음: {volumes_root}")
        sys.exit(1)
    print(f"[preflight] dataset root: {dataset_root}")

    # output dir overwrite guard
    if not args.dry_run and output_dir.exists() and not args.force:
        print(f"[ABORT] output dir 이미 존재합니다: {output_dir}")
        print("  --force 옵션을 사용하면 덮어쓸 수 있습니다.")
        sys.exit(1)

    print("[preflight] 전체 통과")


# ── CT volume 로드 ────────────────────────────────────────────────────────────

def load_ct_volume(
    safe_id: str, dataset_root: Path
) -> tuple[np.ndarray, Path]:
    """ct_hu.npy를 mmap_mode='r'로 로드한다."""
    ct_path = dataset_root / "volumes_npy" / safe_id / "ct_hu.npy"
    if not ct_path.exists():
        raise FileNotFoundError(f"ct_hu.npy 없음: {ct_path}")
    volume = np.load(str(ct_path), mmap_mode="r")
    return volume, ct_path


# ── main 처리 ─────────────────────────────────────────────────────────────────

def run_crop_pipeline(
    df: pd.DataFrame,
    args: argparse.Namespace,
    dataset_root: Path,
    output_dir: Path,
) -> dict:
    """
    환자 순서로 crop을 생성하고 결과를 반환한다.
    dry-run 모드에서는 파일 생성 없이 예상 결과만 집계한다.
    """
    lung_hu_min, lung_hu_max = args.lung_window
    med_hu_min, med_hu_max = args.mediastinal_window
    lung_window = (float(lung_hu_min), float(lung_hu_max))
    med_window = (float(med_hu_min), float(med_hu_max))

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
    z_padding_count = 0
    intensity_mins: list[float] = []
    intensity_maxs: list[float] = []
    means: list[float] = []
    stds: list[float] = []

    # limit 적용
    target_df = df.copy()
    if args.limit is not None:
        target_df = target_df.iloc[: args.limit].copy()
        print(f"[info] --limit {args.limit} 적용")

    n_expected = len(target_df)
    n_created = 0

    if not args.dry_run:
        (output_dir / "crops").mkdir(parents=True, exist_ok=True)
        (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
        (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    # 환자별로 CT 로드 (같은 환자 반복 로드 방지)
    current_safe_id: Optional[str] = None
    current_volume: Optional[np.ndarray] = None
    current_ct_path: Optional[Path] = None

    for idx, row in target_df.iterrows():
        patient_id = str(row["patient_id"])
        safe_id = str(row["safe_id"])
        candidate_id = str(row["candidate_id"])
        z_center = int(row["z_center"])
        y0 = int(row["y0_fixed_crop"])
        x0 = int(row["x0_fixed_crop"])
        y1 = int(row["y1_fixed_crop"])
        x1 = int(row["x1_fixed_crop"])

        # CT volume 로드 (환자 바뀔 때만)
        try:
            if safe_id != current_safe_id:
                current_volume, current_ct_path = load_ct_volume(safe_id, dataset_root)
                current_safe_id = safe_id
        except Exception as e:
            msg = f"CT 로드 실패 [{patient_id}/{candidate_id}]: {e}"
            print(f"[FAIL] {msg}")
            failed_list.append({"candidate_id": candidate_id, "patient_id": patient_id, "reason": msg})
            continue

        # crop 추출
        try:
            image, crop_meta = extract_crop(
                ct_volume=current_volume,
                z_center=z_center,
                y0=y0, x0=x0, y1=y1, x1=x1,
                lung_window=lung_window,
                med_window=med_window,
                z_range=args.z_range,
                crop_size=args.crop_size,
            )
        except Exception as e:
            msg = f"crop 추출 실패 [{patient_id}/{candidate_id}]: {e}"
            print(f"[FAIL] {msg}")
            failed_list.append({"candidate_id": candidate_id, "patient_id": patient_id, "reason": msg})
            continue

        if crop_meta["z_padding_applied"]:
            z_padding_count += 1

        intensity_mins.append(crop_meta["intensity_min"])
        intensity_maxs.append(crop_meta["intensity_max"])
        means.append(crop_meta["crop_mean"])
        stds.append(crop_meta["crop_std"])

        # npz 저장 경로 결정
        patient_crop_dir = output_dir / "crops" / patient_id
        npz_path = patient_crop_dir / f"{candidate_id}.npz"

        if not args.dry_run:
            patient_crop_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(npz_path), image=image)
            n_created += 1
        else:
            n_created += 1  # dry-run에서 생성 예정 수 카운트

        # manifest 행 구성
        manifest_row = row.to_dict()
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

    # summary 집계
    weak_case_patients = {"MSD_lung_043", "MSD_lung_079", "MSD_lung_096"}
    result_df = pd.DataFrame(manifest_rows) if manifest_rows else pd.DataFrame()

    weak_case_counts: dict[str, int] = {}
    for wc in sorted(weak_case_patients):
        if len(result_df) > 0 and "patient_id" in result_df.columns:
            weak_case_counts[wc] = int((result_df["patient_id"] == wc).sum())
        else:
            weak_case_counts[wc] = 0

    large_bbox_count = (
        int(result_df["bbox_too_large"].astype(bool).sum())
        if len(result_df) > 0 and "bbox_too_large" in result_df.columns
        else 0
    )

    summary = {
        "script": "build_rd4ad_2p5d_crop_dataset.py",
        "timestamp": timestamp,
        "sampling_manifest_path": str(Path(args.sampling_manifest).resolve()),
        "output_tag": dataset_tag,
        "dataset_root": str(dataset_root),
        "n_source_rows": len(df),
        "n_patients": df["patient_id"].nunique(),
        "n_crops_expected": n_expected,
        "n_crops_created": n_created,
        "crop_shape": crop_shape_str,
        "dtype": "float32",
        "window_config": window_config,
        "z_padding_mode": "nearest_repeat",
        "z_padding_count": z_padding_count,
        "weak_case_crop_count": weak_case_counts,
        "large_bbox_crop_count": large_bbox_count,
        "intensity_min_global": float(min(intensity_mins)) if intensity_mins else None,
        "intensity_max_global": float(max(intensity_maxs)) if intensity_maxs else None,
        "crop_mean_global": float(np.mean(means)) if means else None,
        "crop_std_global": float(np.mean(stds)) if stds else None,
        "failed_count": len(failed_list),
        "failed_examples": failed_list[:10],
        "dry_run": args.dry_run,
        "note": [
            "RD4AD train crop dataset이며 성능 우위 확정 아님",
            "lesion mask는 train input으로 사용하지 않음",
            "MIP/MPR/score map/ROI mask는 기본 v1 입력이 아님",
        ],
    }

    return result_df, summary


# ── targeted dry-run 필터 ─────────────────────────────────────────────────────

def select_target_rows_for_patient(
    patient_df: pd.DataFrame,
    mode: str,
    max_per_patient: Optional[int],
) -> pd.DataFrame:
    """
    단일 환자의 sampling rows에서 target 검증 대상을 선택한다.
    deterministic 재현을 위해 candidate_id 기준 base sort 후 선택.
    """
    df = patient_df.copy().sort_values("candidate_id").reset_index(drop=True)

    if mode == "score_top":
        df = df.sort_values("max_padim_score", ascending=False).reset_index(drop=True)

    elif mode == "large_bbox_first":
        df = df.sort_values(
            ["bbox_too_large", "max_padim_score"], ascending=[False, False]
        ).reset_index(drop=True)

    elif mode == "z_spread":
        df = df.sort_values("z_center").reset_index(drop=True)
        if max_per_patient is not None and len(df) > max_per_patient:
            n = max_per_patient
            total = len(df)
            indices = (
                [int(round(i * (total - 1) / max(n - 1, 1))) for i in range(n)]
                if n > 1
                else [0]
            )
            indices = sorted(set(indices))[:n]
            return df.iloc[indices].reset_index(drop=True)

    elif mode == "mixed":
        base = df.sort_values("candidate_id")
        selected_ids: list[str] = []

        # 1. large_bbox 최대 2개
        large = (
            base[base["bbox_too_large"].astype(bool)]
            .sort_values("max_padim_score", ascending=False)
            .head(2)
        )
        selected_ids.extend(large["candidate_id"].tolist())

        # 2. score_top 최대 2개 (중복 제외)
        score_top = (
            base[~base["candidate_id"].isin(selected_ids)]
            .sort_values("max_padim_score", ascending=False)
            .head(2)
        )
        selected_ids.extend(score_top["candidate_id"].tolist())

        # 3. z_center 낮은 1개 (중복 제외)
        z_low = (
            base[~base["candidate_id"].isin(selected_ids)]
            .sort_values("z_center", ascending=True)
            .head(1)
        )
        selected_ids.extend(z_low["candidate_id"].tolist())

        # 4. z_center 높은 1개 (중복 제외)
        z_high = (
            base[~base["candidate_id"].isin(selected_ids)]
            .sort_values("z_center", ascending=False)
            .head(1)
        )
        selected_ids.extend(z_high["candidate_id"].tolist())

        selected = base[base["candidate_id"].isin(selected_ids)].sort_values(
            "max_padim_score", ascending=False
        )
        remaining = base[~base["candidate_id"].isin(selected_ids)].sort_values(
            "max_padim_score", ascending=False
        )
        df = pd.concat([selected, remaining]).reset_index(drop=True)

    if max_per_patient is not None:
        df = df.head(max_per_patient)

    return df


def apply_target_patient_filter(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """
    --target-patients 지정 시 해당 환자 rows만 선택해 반환한다.
    지정 없으면 df 전체 반환.
    """
    if not args.target_patients:
        return df

    target_list = args.target_patients
    print(f"\n[target] 대상 환자 필터 적용: {target_list}")
    print(f"[target] 선택 모드: {args.target_selection_mode}"
          + (f"  / 환자당 최대: {args.target_max_per_patient}개" if args.target_max_per_patient else ""))

    # 봉인 환자 차단
    sealed_found = set(target_list) & SEALED_PATIENTS
    if sealed_found:
        print(f"[ABORT] --target-patients에 봉인 환자 포함: {sorted(sealed_found)}")
        sys.exit(1)

    # manifest에 없는 환자 차단
    manifest_patients = set(df["patient_id"].unique())
    missing = [p for p in target_list if p not in manifest_patients]
    if missing:
        print(f"[ABORT] --target-patients 중 sampling manifest에 없는 환자: {missing}")
        sys.exit(1)

    n_before = len(df)
    parts: list[pd.DataFrame] = []
    for patient_id in target_list:
        patient_df = df[df["patient_id"] == patient_id]
        selected = select_target_rows_for_patient(
            patient_df,
            mode=args.target_selection_mode,
            max_per_patient=args.target_max_per_patient,
        )
        print(f"  {patient_id}: {len(selected)}개 선택 (전체 {len(patient_df)}개)")
        parts.append(selected)

    result = pd.concat(parts).reset_index(drop=True)
    n_after = len(result)
    print(f"[target] 필터 전: {n_before}행 → 필터 후: {n_after}행")

    if n_after == 0:
        print("[ABORT] target 필터 후 처리할 row가 없습니다.")
        sys.exit(1)

    return result


# ── disk 용량 추정 ───────────────────────────────────────────────────────────

def estimate_disk_usage(n_crops: int, mb_per_crop: float = 0.25) -> tuple[float, float]:
    """예상 용량(MB)과 현재 여유 공간(MB)을 반환한다."""
    estimated_mb = n_crops * mb_per_crop
    free_bytes = shutil.disk_usage(REPO).free
    free_mb = free_bytes / (1024 * 1024)
    return estimated_mb, free_mb


# ── argparse ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RD4AD 학습용 2.5D multi-window crop dataset 생성"
    )
    parser.add_argument(
        "--sampling-manifest",
        default=str(
            REPO
            / "outputs/second-stage-lesion-refiner-v1/sampling"
            / "rd4ad_train_normal_like_fixed96_thr001_v1"
            / "rd4ad_train_sampling_manifest_rd4ad_train_normal_like_fixed96_thr001_v1.csv"
        ),
        help="입력 sampling manifest 경로",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO / "outputs/second-stage-lesion-refiner-v1/crops"),
        help="출력 root 디렉토리",
    )
    parser.add_argument(
        "--output-tag",
        default="rd4ad_train_2p5d_mw_fixed96_thr001_v1",
        help="출력 태그 (하위 폴더명 및 파일명에 사용)",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="CT volume root 디렉토리 (미지정 시 configs/paths.local.yaml에서 읽음)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=96,
        help="crop 크기 (default: 96)",
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
        help="mediastinal/soft tissue window HU 범위 (default: -160 240)",
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
        help="debug용: 처음 N개 후보만 처리",
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
    parser.add_argument(
        "--target-patients",
        nargs="*",
        default=None,
        help="[dry-run 전용] 지정 patient_id만 검증. 예: --target-patients MSD_lung_043 LUNG1-108",
    )
    parser.add_argument(
        "--target-max-per-patient",
        type=int,
        default=None,
        help="[dry-run 전용] target patient별 최대 검증 crop 수",
    )
    parser.add_argument(
        "--target-selection-mode",
        choices=["score_top", "large_bbox_first", "z_spread", "mixed"],
        default="mixed",
        help="target crop 선택 방식 (score_top/large_bbox_first/z_spread/mixed, default: mixed)",
    )
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # target 옵션 dry-run 전용 guard
    if args.target_patients is not None and not args.dry_run:
        print("[ABORT] --target-patients는 --dry-run 모드에서만 사용 가능합니다.")
        sys.exit(1)
    if args.target_max_per_patient is not None and not args.dry_run:
        print("[ABORT] --target-max-per-patient는 --dry-run 모드에서만 사용 가능합니다.")
        sys.exit(1)

    manifest_path = Path(args.sampling_manifest)
    output_root = Path(args.output_root)
    output_dir = output_root / args.output_tag
    out_csv_name = f"crop_manifest_{args.output_tag}.csv"
    out_json_name = f"crop_summary_{args.output_tag}.json"
    out_csv = output_dir / "manifests" / out_csv_name
    out_json = output_dir / "reports" / out_json_name

    # manifest 로드
    if not manifest_path.exists():
        print(f"[ABORT] sampling manifest 없음: {manifest_path}")
        sys.exit(1)

    print(f"[load] {manifest_path}")
    df = pd.read_csv(manifest_path)

    # dataset root 결정
    dataset_root = load_dataset_root(args)

    # preflight (전체 manifest 기준으로 검증)
    run_preflight(args, df, dataset_root, output_dir)

    # target patient 필터 (dry-run 전용, --target-patients 없으면 전체 사용)
    df_run = apply_target_patient_filter(df, args)

    # 임시 output dir 경로 (실제 실행 시 사용, dry-run에서는 생성하지 않음)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = output_root / f".tmp_{args.output_tag}_{run_timestamp}"

    # 디스크 용량 추정 출력
    n_target = len(df_run)
    est_mb, free_mb = estimate_disk_usage(n_target)
    print(f"[disk] 예상 용량: {est_mb:.1f} MB (0.25 MB × {n_target}개) / 여유 공간: {free_mb:.1f} MB")
    if not args.dry_run and free_mb < est_mb * 2:
        print(f"[ABORT] 디스크 여유 공간 부족: 여유 {free_mb:.1f} MB < 예상 {est_mb:.1f} MB × 2")
        sys.exit(1)

    # 실제 실행: npz/CSV/JSON을 temp dir에 저장 / dry-run: 무쓰기
    write_dir = temp_dir if not args.dry_run else output_dir

    # crop pipeline 실행
    print(f"[crop] 2.5D crop {'(dry-run)' if args.dry_run else '실제 생성 (임시 dir 사용)'} 시작...")
    result_df, summary = run_crop_pipeline(df_run, args, dataset_root, write_dir)

    n_created = summary["n_crops_created"]
    n_failed = summary["failed_count"]
    n_expected = summary["n_crops_expected"]
    print(f"[crop] 생성 완료: {n_created}개 / 실패: {n_failed}개")
    print(f"[crop] z padding 발생: {summary['z_padding_count']}건")
    print(f"[crop] weak_case: {summary['weak_case_crop_count']}")
    print(f"[crop] large_bbox crop: {summary['large_bbox_crop_count']}")

    if n_failed > 0:
        print(f"[WARNING] 실패 crop {n_failed}건 발생. 실패 예시: {summary['failed_examples'][:3]}")

    if args.dry_run:
        print("\n[dry-run] 파일 생성 없음. 위 예상 결과만 출력합니다.")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    # ── 실제 실행 후 partial output 방지 검증 ────────────────────────────────

    # 실패 건수 확인
    if n_failed > 0:
        print(f"[ABORT] 실패 crop {n_failed}건 → 최종 output dir 생성 금지.")
        print(f"  실패 목록: {json.dumps(summary['failed_examples'], ensure_ascii=False)}")
        print(f"  임시 dir (자동 삭제 안 함): {temp_dir}")
        sys.exit(1)

    # npz 파일 수 검증
    n_npz = len(list((temp_dir / "crops").rglob("*.npz")))
    if n_npz != n_expected:
        print(f"[ABORT] npz 파일 수 불일치: 실제 {n_npz}건 ≠ 예상 {n_expected}건 → 최종 output dir 생성 금지.")
        print(f"  임시 dir (자동 삭제 안 함): {temp_dir}")
        sys.exit(1)

    # crop manifest row 수 검증
    if len(result_df) != n_expected:
        print(f"[ABORT] manifest row 수 불일치: {len(result_df)} ≠ {n_expected} → 최종 output dir 생성 금지.")
        print(f"  임시 dir (자동 삭제 안 함): {temp_dir}")
        sys.exit(1)

    # n_crops_created == n_expected 검증
    if n_created != n_expected:
        print(f"[ABORT] n_crops_created 불일치: {n_created} ≠ {n_expected} → 최종 output dir 생성 금지.")
        print(f"  임시 dir (자동 삭제 안 함): {temp_dir}")
        sys.exit(1)

    # summary partial output 필드 추가
    summary["output_dir"] = str(output_dir)
    summary["temp_output_dir"] = str(temp_dir)
    summary["partial_output_policy"] = "temp_dir_then_rename_if_success"
    summary["n_npz_files_written"] = n_npz
    summary["finalized_successfully"] = True
    summary["finalized_output_dir"] = str(output_dir)

    # manifest CSV / summary JSON → temp dir에 먼저 저장
    temp_csv = temp_dir / "manifests" / out_csv_name
    temp_json_path = temp_dir / "reports" / out_json_name

    print(f"\n[save] crop manifest CSV → {temp_csv} (임시)")
    result_df.to_csv(temp_csv, index=False)

    print(f"[save] crop summary JSON → {temp_json_path} (임시)")
    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # temp dir → 최종 output dir rename (output_dir가 없어야 성공)
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
