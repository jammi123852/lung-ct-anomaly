#!/usr/bin/env python3
"""
Hard Negative Top-Score Visual QA용 Context Overlay PNG Pack 생성 스크립트

이 스크립트는 QA manifest의 crop에 대해 원본 volume(ct_hu, roi, lesion_mask)을
로드하여 context 크기로 잘라낸 뒤, 2행 3열 overlay 패널 PNG를 생성합니다.

주의:
- threshold 확정 아님
- 병변 성능 결론 금지
- stage2_holdout 미사용
- v2 미사용
- score 재계산 없음
- crop 복사 없음
"""

import argparse
import gc
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QA_MANIFEST = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/evaluation"
    / "rd4ad_2p5d_normal_mw_fixed96_v1/hard_negative_top_score_qa_v1"
    / "hard_negative_top_score_qa_manifest_v1.csv"
)

CROP_MANIFEST = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/crops"
    / "rd4ad_train_2p5d_mw_fixed96_thr001_v1/manifests"
    / "crop_manifest_rd4ad_train_2p5d_mw_fixed96_thr001_v1.csv"
)

PNG_INDEX = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/visualizations"
    / "hard_negative_top_score_qa_v1/manifest_png_index_v1.csv"
)

LABEL_SHEET = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "hard_negative_top_score_qa_v1"
    / "hard_negative_top_score_manual_label_sheet_v1.csv"
)

VOLUME_SOURCE_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/visualizations"
    / "hard_negative_top_score_context_overlay_qa_v1"
)

# ---------------------------------------------------------------------------
# 윈도우 프리셋
# ---------------------------------------------------------------------------

LUNG_HU_MIN = -1350
LUNG_HU_MAX = 150
MED_HU_MIN = -160
MED_HU_MAX = 240

# ---------------------------------------------------------------------------
# 금지 키워드
# ---------------------------------------------------------------------------

BLOCKED_PATH_KEYWORDS_OUTPUT = ["holdout", "stage2_holdout", "v2"]
BLOCKED_PATH_KEYWORDS_VOLUME = ["v2"]
BLOCKED_PATH_KEYWORDS_PATIENT = ["stage2_holdout"]

# ---------------------------------------------------------------------------
# manifest_png_index CSV 컬럼
# ---------------------------------------------------------------------------

PNG_INDEX_COLUMNS = [
    "patient_id",
    "crop_id",
    "crop_path",
    "png_path",
    "contact_sheet_path",
    "qa_group",
    "qa_priority",
    "crop_score_l1_mean",
    "padim_score_mean",
    "padim_score_max",
    "threshold_exceed_val_p90",
    "threshold_exceed_val_p95",
    "threshold_exceed_val_p99",
    "context_size_used",
    "z_center",
    "y_context_start",
    "y_context_end",
    "x_context_start",
    "x_context_end",
    "was_clipped",
    "pad_top",
    "pad_bottom",
    "pad_left",
    "pad_right",
    "png_generated",
    "generation_error",
]

# ---------------------------------------------------------------------------
# guard 함수
# ---------------------------------------------------------------------------


def _guard_path_keywords(path_str: str, keywords: list, label: str):
    """경로에 금지 키워드가 있으면 즉시 종료."""
    s = str(path_str).lower()
    for kw in keywords:
        if kw.lower() in s:
            print(
                f"[GUARD] {label} 경로에 금지 키워드 '{kw}' 포함 — 즉시 중단: {path_str}",
                file=sys.stderr,
            )
            sys.exit(1)


def run_path_guards(output_root: Path, volume_source_root: Path):
    """출력/볼륨 경로 guard 체크."""
    output_str = str(output_root)
    for kw in BLOCKED_PATH_KEYWORDS_OUTPUT:
        if kw.lower() in output_str.lower():
            print(
                f"[GUARD] OUTPUT_ROOT에 금지 키워드 '{kw}' 포함 — 즉시 중단: {output_root}",
                file=sys.stderr,
            )
            sys.exit(1)

    volume_str = str(volume_source_root)
    for kw in BLOCKED_PATH_KEYWORDS_VOLUME:
        if kw.lower() in volume_str.lower():
            print(
                f"[GUARD] VOLUME_SOURCE_ROOT에 금지 키워드 '{kw}' 포함 — 즉시 중단: {volume_source_root}",
                file=sys.stderr,
            )
            sys.exit(1)

    print("[GUARD] 경로 guard 통과")


def guard_patient_ids(df: pd.DataFrame):
    """patient_id에 금지 키워드가 있으면 즉시 종료."""
    for pid in df["patient_id"].dropna().astype(str):
        for kw in BLOCKED_PATH_KEYWORDS_PATIENT:
            if kw.lower() in pid.lower():
                print(
                    f"[GUARD] patient_id에 금지 키워드 '{kw}' 포함 — 즉시 중단: {pid}",
                    file=sys.stderr,
                )
                sys.exit(1)
    print("[GUARD] patient_id guard 통과")


def guard_crop_paths(df: pd.DataFrame):
    """crop_path가 .npz 확장자가 아니면 즉시 종료."""
    errors = []
    for cp in df["crop_path"].dropna().astype(str):
        if not cp.endswith(".npz"):
            errors.append(cp)
    if errors:
        for e in errors[:5]:
            print(f"[GUARD] crop_path가 .npz 아님: {e}", file=sys.stderr)
        if len(errors) > 5:
            print(f"[GUARD] ... 이하 {len(errors) - 5}건 더", file=sys.stderr)
        sys.exit(1)
    print("[GUARD] crop_path 확장자 guard 통과")


def guard_crop_id_unique(df: pd.DataFrame):
    """crop_id 중복이 있으면 즉시 종료."""
    dup = df[df.duplicated(subset=["crop_id"], keep=False)]
    if len(dup) > 0:
        dup_ids = dup["crop_id"].unique().tolist()[:5]
        print(
            f"[GUARD] crop_id 중복 발견 — 즉시 중단: {dup_ids}",
            file=sys.stderr,
        )
        sys.exit(1)
    print("[GUARD] crop_id 중복 guard 통과")


# ---------------------------------------------------------------------------
# patient 폴더 매칭
# ---------------------------------------------------------------------------


def find_patient_folder(patient_id: str, volumes_npy_root: Path) -> tuple:
    """
    volumes_npy_root 아래에서 patient_id에 해당하는 폴더를 검색.
    여러 패턴을 순서대로 시도하고 첫 번째로 1개 매칭되는 패턴을 사용.
    반환: (matched_folder, matched_pattern_label)
    정확히 1개가 아니면 sys.exit(1)
    """
    all_dirs = [p for p in volumes_npy_root.iterdir() if p.is_dir()]

    patterns = [
        (f"MSD_Lung_{patient_id}__prefix", lambda p: p.name.startswith(f"MSD_Lung_{patient_id}__")),
        (f"NSCLC_{patient_id}__prefix", lambda p: p.name.startswith(f"NSCLC_{patient_id}__")),
        (f"{patient_id}__prefix", lambda p: p.name.startswith(f"{patient_id}__")),
        (f"{patient_id}_in_name", lambda p: patient_id in p.name),
        (f"{patient_id}_exact", lambda p: p.name == patient_id),
    ]

    for pattern_label, match_fn in patterns:
        matches = [p for p in all_dirs if match_fn(p)]
        if len(matches) == 1:
            return (matches[0], pattern_label)
        if len(matches) > 1:
            names = [m.name for m in matches]
            print(
                f"[GUARD] patient_id={patient_id} 패턴={pattern_label} 매칭 폴더 2개 이상 — 즉시 중단: {names}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(
        f"[GUARD] patient_id={patient_id} 모든 패턴에서 매칭 폴더 없음 — 즉시 중단",
        file=sys.stderr,
    )
    sys.exit(1)


def build_patient_folder_map(patient_ids: list, volume_source_root: Path) -> dict:
    """
    환자별 volume 폴더 매핑 딕셔너리 반환.
    반환: {patient_id -> (folder_path, pattern_label)}
    """
    volumes_npy_root = volume_source_root / "volumes_npy"
    if not volumes_npy_root.is_dir():
        print(
            f"[ERROR] volumes_npy 폴더 없음: {volumes_npy_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    mapping = {}
    for pid in patient_ids:
        folder, pattern_label = find_patient_folder(str(pid), volumes_npy_root)
        mapping[pid] = (folder, pattern_label)
        print(f"[INFO] patient_id={pid} → {folder.name}  [pattern={pattern_label}]")
    return mapping


# ---------------------------------------------------------------------------
# 파일 로드
# ---------------------------------------------------------------------------


def load_qa_manifest() -> pd.DataFrame:
    path = QA_MANIFEST
    if not path.is_file():
        print(f"[ERROR] QA manifest 없음: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"[INFO] QA manifest 로드 완료 — rows={len(df)}")
    return df


def load_crop_manifest() -> pd.DataFrame:
    path = CROP_MANIFEST
    if not path.is_file():
        print(f"[ERROR] Crop manifest 없음: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"[INFO] Crop manifest 로드 완료 — rows={len(df)}")
    return df


def load_png_index() -> pd.DataFrame:
    path = PNG_INDEX
    if not path.is_file():
        print(f"[WARNING] PNG index 없음 (선택 입력): {path}")
        return None
    df = pd.read_csv(path)
    print(f"[INFO] PNG index 로드 완료 — rows={len(df)}")
    return df


def load_label_sheet() -> pd.DataFrame:
    path = LABEL_SHEET
    if not path.is_file():
        print(f"[WARNING] Label sheet 없음 (선택 입력): {path}")
        return None
    df = pd.read_csv(path)
    print(f"[INFO] Label sheet 로드 완료 — rows={len(df)}")
    return df


# ---------------------------------------------------------------------------
# join 로직
# ---------------------------------------------------------------------------

CROP_MANIFEST_REQUIRED_COLS = [
    "crop_path",
    "z_center",
    "y0_fixed_crop",
    "x0_fixed_crop",
    "y1_fixed_crop",
    "x1_fixed_crop",
]


def join_manifests(qa_df: pd.DataFrame, crop_df: pd.DataFrame) -> pd.DataFrame:
    """qa_manifest와 crop_manifest를 crop_path 기준 left join."""
    missing = [c for c in CROP_MANIFEST_REQUIRED_COLS if c not in crop_df.columns]
    if missing:
        print(f"[ERROR] crop manifest 필수 컬럼 누락: {missing}", file=sys.stderr)
        sys.exit(1)

    crop_subset = crop_df[CROP_MANIFEST_REQUIRED_COLS].copy()

    # crop_path 중복 확인 (crop_manifest 쪽)
    dup_crop = crop_subset[crop_subset.duplicated(subset=["crop_path"], keep=False)]
    if len(dup_crop) > 0:
        dup_paths = dup_crop["crop_path"].unique().tolist()[:3]
        print(
            f"[ERROR] crop_manifest에 crop_path 중복: {dup_paths}",
            file=sys.stderr,
        )
        sys.exit(1)

    merged = qa_df.merge(crop_subset, on="crop_path", how="left")

    # join 후 필수 컬럼 결측 확인
    missing_after = merged[CROP_MANIFEST_REQUIRED_COLS[1:]].isnull().any()
    if missing_after.any():
        bad_cols = missing_after[missing_after].index.tolist()
        n_bad = merged[bad_cols[0]].isnull().sum()
        print(
            f"[ERROR] join 후 필수 컬럼에 NaN 존재 ({n_bad}행). 컬럼: {bad_cols}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] join 완료 — merged rows={len(merged)}")
    return merged


# ---------------------------------------------------------------------------
# 좌표 계산
# ---------------------------------------------------------------------------


def compute_context_coords(
    row: pd.Series, context_size: int, vol_shape: tuple
) -> dict:
    """
    fixed crop 중심 기준으로 context crop 좌표를 계산한다.
    vol_shape: (Z, Y, X)
    반환: {y_start_raw, y_end_raw, x_start_raw, x_end_raw,
           y_start, y_end, x_start, x_end,
           pad_top, pad_bottom, pad_left, pad_right,
           was_clipped}
    """
    cy = int((row["y0_fixed_crop"] + row["y1_fixed_crop"]) // 2)
    cx = int((row["x0_fixed_crop"] + row["x1_fixed_crop"]) // 2)

    half = context_size // 2
    y_start_raw = cy - half
    y_end_raw = cy + half
    x_start_raw = cx - half
    x_end_raw = cx + half

    _, vol_y, vol_x = vol_shape

    y_start = int(np.clip(y_start_raw, 0, vol_y))
    y_end = int(np.clip(y_end_raw, 0, vol_y))
    x_start = int(np.clip(x_start_raw, 0, vol_x))
    x_end = int(np.clip(x_end_raw, 0, vol_x))

    was_clipped = (
        y_start != y_start_raw
        or y_end != y_end_raw
        or x_start != x_start_raw
        or x_end != x_end_raw
    )

    pad_top = max(0, -y_start_raw)
    pad_bottom = max(0, y_end_raw - vol_y)
    pad_left = max(0, -x_start_raw)
    pad_right = max(0, x_end_raw - vol_x)

    return {
        "y_start_raw": y_start_raw,
        "y_end_raw": y_end_raw,
        "x_start_raw": x_start_raw,
        "x_end_raw": x_end_raw,
        "y_start": y_start,
        "y_end": y_end,
        "x_start": x_start,
        "x_end": x_end,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "was_clipped": was_clipped,
    }


def extract_context_with_padding(
    volume_2d: np.ndarray,
    coords: dict,
    context_size: int,
    pad_value,
) -> np.ndarray:
    """
    clipped 좌표로 슬라이스를 추출하고 padding으로 context_size×context_size를 보장한다.
    volume_2d: shape (Y, X) — z slice된 2D array
    coords: compute_context_coords 반환값
    pad_value: padding에 채울 값
    반환: shape (context_size, context_size)
    """
    y_s = coords["y_start"]
    y_e = coords["y_end"]
    x_s = coords["x_start"]
    x_e = coords["x_end"]
    pad_top = coords["pad_top"]
    pad_bottom = coords["pad_bottom"]
    pad_left = coords["pad_left"]
    pad_right = coords["pad_right"]

    crop = volume_2d[y_s:y_e, x_s:x_e]
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop = np.pad(
            crop,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_value,
        )
    if crop.shape != (context_size, context_size):
        raise ValueError(
            f"extract_context_with_padding: shape={crop.shape}, "
            f"expected=({context_size},{context_size})"
        )
    return crop


# ---------------------------------------------------------------------------
# volume 로드
# ---------------------------------------------------------------------------


def load_volume_arrays(patient_folder: Path) -> tuple:
    """
    ct_hu.npy, roi_0_0.npy, lesion_mask_roi_0_0.npy 로드.
    shape mismatch이면 즉시 종료.
    반환: (ct_hu, roi, lesion_mask) — numpy arrays
    """
    ct_path = patient_folder / "ct_hu.npy"
    roi_path = patient_folder / "roi_0_0.npy"
    lesion_path = patient_folder / "lesion_mask_roi_0_0.npy"

    for p in [ct_path, roi_path, lesion_path]:
        if not p.is_file():
            print(f"[ERROR] 볼륨 파일 없음: {p}", file=sys.stderr)
            sys.exit(1)

    ct_hu = np.load(str(ct_path), mmap_mode="r")
    roi = np.load(str(roi_path), mmap_mode="r")
    lesion = np.load(str(lesion_path), mmap_mode="r")

    if ct_hu.shape != roi.shape or ct_hu.shape != lesion.shape:
        print(
            f"[GUARD] shape mismatch — ct={ct_hu.shape}, roi={roi.shape}, lesion={lesion.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    return ct_hu, roi, lesion


# ---------------------------------------------------------------------------
# HU windowing
# ---------------------------------------------------------------------------


def apply_window(slice_hu: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """HU 값을 [0, 1] float으로 클리핑 후 정규화."""
    clipped = np.clip(slice_hu.astype(np.float32), hu_min, hu_max)
    normed = (clipped - hu_min) / (hu_max - hu_min + 1e-8)
    return normed


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------


def draw_contour_overlay(ax, mask_2d: np.ndarray, color: str, linewidth: float = 0.8):
    """mask_2d (2D uint8)의 contour를 ax에 그린다."""
    if mask_2d.max() == 0:
        return
    ax.contour(mask_2d, levels=[0.5], colors=[color], linewidths=[linewidth])


def make_context_overlay_figure(
    ct_slice: np.ndarray,
    roi_slice: np.ndarray,
    lesion_slice: np.ndarray,
    patient_id: str,
    crop_id,
    z_center: int,
    row_meta: dict,
) -> plt.Figure:
    """
    2행 3열 overlay 패널 figure 생성.
    행0: lung window (raw, +roi, +roi+lesion)
    행1: mediastinal window (raw, +roi, +roi+lesion)
    """
    lung = apply_window(ct_slice, LUNG_HU_MIN, LUNG_HU_MAX)
    med = apply_window(ct_slice, MED_HU_MIN, MED_HU_MAX)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    panel_data = [
        # (row, col, image, draw_roi, draw_lesion, title)
        (0, 0, lung, False, False, "Lung"),
        (0, 1, lung, True, False, "Lung+ROI"),
        (0, 2, lung, True, True, "Lung+ROI+Lesion"),
        (1, 0, med, False, False, "Mediastinal"),
        (1, 1, med, True, False, "Mediastinal+ROI"),
        (1, 2, med, True, True, "Mediastinal+ROI+Lesion"),
    ]

    for r, c, img, draw_roi, draw_lesion, title in panel_data:
        ax = axes[r][c]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="equal")
        if draw_roi:
            draw_contour_overlay(ax, roi_slice.astype(np.uint8), color="lime", linewidth=0.8)
        if draw_lesion:
            draw_contour_overlay(ax, lesion_slice.astype(np.uint8), color="red", linewidth=0.8)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    # suptitle
    qa_group = row_meta.get("qa_group", "")
    qa_priority = row_meta.get("qa_priority", "")
    l1_mean = row_meta.get("crop_score_l1_mean", float("nan"))
    padim_mean = row_meta.get("padim_score_mean", float("nan"))
    padim_max = row_meta.get("padim_score_max", float("nan"))
    p90 = row_meta.get("threshold_exceed_val_p90", "")
    p95 = row_meta.get("threshold_exceed_val_p95", "")
    p99 = row_meta.get("threshold_exceed_val_p99", "")

    try:
        l1_mean_str = f"{float(l1_mean):.4f}"
    except (ValueError, TypeError):
        l1_mean_str = str(l1_mean)
    try:
        padim_mean_str = f"{float(padim_mean):.2f}"
    except (ValueError, TypeError):
        padim_mean_str = str(padim_mean)
    try:
        padim_max_str = f"{float(padim_max):.2f}"
    except (ValueError, TypeError):
        padim_max_str = str(padim_max)

    suptitle = (
        f"patient={patient_id} | crop_id={crop_id} | z={z_center}\n"
        f"qa_group={qa_group} | priority={qa_priority}\n"
        f"l1_mean={l1_mean_str} | padim_mean={padim_mean_str} | padim_max={padim_max_str}\n"
        f"p90={p90} p95={p95} p99={p99}"
    )
    fig.suptitle(suptitle, fontsize=8, y=1.01)
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# contact sheet 생성
# ---------------------------------------------------------------------------


def build_contact_sheet(
    png_paths: list,
    output_path: Path,
    max_per_sheet: int,
    sheet_label: str,
):
    """
    png_paths 목록을 max_per_sheet씩 나눠 contact sheet PNG로 저장.
    각 sheet는 n_cols=5 그리드로 배치.
    """
    n_cols = 5
    sheets_created = []

    for sheet_idx, start in enumerate(range(0, len(png_paths), max_per_sheet)):
        batch = png_paths[start : start + max_per_sheet]
        n_rows = math.ceil(len(batch) / n_cols)

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5)
        )
        # axes를 항상 2D로 처리
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes[np.newaxis, :]
        elif n_cols == 1:
            axes = axes[:, np.newaxis]

        for i, png_path in enumerate(batch):
            r = i // n_cols
            c = i % n_cols
            ax = axes[r][c]
            if Path(png_path).is_file():
                img = mpimg.imread(str(png_path))
                ax.imshow(img, aspect="auto")
            else:
                ax.text(0.5, 0.5, "MISSING", ha="center", va="center", fontsize=8)
            ax.axis("off")

        # 남은 빈 셀 끄기
        total_cells = n_rows * n_cols
        for i in range(len(batch), total_cells):
            r = i // n_cols
            c = i % n_cols
            axes[r][c].axis("off")

        sheet_path = output_path / f"contact_sheet_{sheet_label}_{sheet_idx:03d}.png"
        fig.suptitle(
            f"Contact Sheet: {sheet_label} [{sheet_idx}]  ({len(batch)} crops)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(str(sheet_path), dpi=100, bbox_inches="tight")
        plt.close(fig)
        sheets_created.append(sheet_path)
        print(f"[INFO] contact sheet 저장: {sheet_path.name}")

    return sheets_created


# ---------------------------------------------------------------------------
# volume 파일 검증
# ---------------------------------------------------------------------------


def validate_volume_files_for_patients(patient_folder_map: dict) -> dict:
    """
    모든 patient에 대해 ct_hu.npy, roi_0_0.npy, lesion_mask_roi_0_0.npy 검증.
    patient_folder_map: {patient_id -> (folder_path, pattern_label)}
    반환: 검증 결과 dict
    """
    result = {
        "total_patients": len(patient_folder_map),
        "ct_exists_count": 0,
        "roi_exists_count": 0,
        "lesion_exists_count": 0,
        "all_files_ok_count": 0,
        "shape_mismatch_count": 0,
        "has_nan_count": 0,
        "has_inf_count": 0,
        "dtype_warning_count": 0,
        "per_patient": {},
    }

    for pid, (folder, _pattern_label) in patient_folder_map.items():
        ct_path = folder / "ct_hu.npy"
        roi_path = folder / "roi_0_0.npy"
        lesion_path = folder / "lesion_mask_roi_0_0.npy"

        ct_exists = ct_path.is_file()
        roi_exists = roi_path.is_file()
        lesion_exists = lesion_path.is_file()

        if ct_exists:
            result["ct_exists_count"] += 1
        if roi_exists:
            result["roi_exists_count"] += 1
        if lesion_exists:
            result["lesion_exists_count"] += 1

        per = {
            "ct_exists": ct_exists,
            "roi_exists": roi_exists,
            "lesion_exists": lesion_exists,
            "ct_shape": None,
            "roi_shape": None,
            "lesion_shape": None,
            "shape_ok": False,
            "ct_dtype": None,
            "has_nan": False,
            "has_inf": False,
            "roi_min": None,
            "roi_max": None,
            "lesion_min": None,
            "lesion_max": None,
        }

        if ct_exists and roi_exists and lesion_exists:
            try:
                ct_mmap = np.load(str(ct_path), mmap_mode="r")
                roi_mmap = np.load(str(roi_path), mmap_mode="r")
                lesion_mmap = np.load(str(lesion_path), mmap_mode="r")

                per["ct_shape"] = tuple(ct_mmap.shape)
                per["roi_shape"] = tuple(roi_mmap.shape)
                per["lesion_shape"] = tuple(lesion_mmap.shape)
                per["ct_dtype"] = str(ct_mmap.dtype)

                # shape 비교
                if ct_mmap.shape == roi_mmap.shape == lesion_mmap.shape:
                    per["shape_ok"] = True
                    result["all_files_ok_count"] += 1
                else:
                    result["shape_mismatch_count"] += 1
                    print(
                        f"[WARNING] patient_id={pid} shape mismatch:"
                        f" ct={ct_mmap.shape}, roi={roi_mmap.shape}, lesion={lesion_mmap.shape}"
                    )

                # dtype 권장 확인 (warning, exit 아님)
                if ct_mmap.dtype != np.int16:
                    result["dtype_warning_count"] += 1
                    print(
                        f"[WARNING] patient_id={pid} ct_hu.npy dtype={ct_mmap.dtype} (권장: int16)"
                    )
                if roi_mmap.dtype != np.uint8:
                    result["dtype_warning_count"] += 1
                    print(
                        f"[WARNING] patient_id={pid} roi_0_0.npy dtype={roi_mmap.dtype} (권장: uint8)"
                    )
                if lesion_mmap.dtype != np.uint8:
                    result["dtype_warning_count"] += 1
                    print(
                        f"[WARNING] patient_id={pid} lesion_mask_roi_0_0.npy dtype={lesion_mmap.dtype} (권장: uint8)"
                    )

                # nan/inf 확인 (ct_hu만)
                # int dtype은 NaN/Inf가 불가능하므로 전체 float32 변환 없이 생략
                if np.issubdtype(ct_mmap.dtype, np.integer):
                    has_nan = False
                    has_inf = False
                else:
                    # float dtype: 전체 astype(float32) 금지, z-slice 단위 검사
                    has_nan = False
                    has_inf = False
                    for _z in range(ct_mmap.shape[0]):
                        _sl = ct_mmap[_z]
                        if not has_nan and bool(np.isnan(_sl).any()):
                            has_nan = True
                        if not has_inf and bool(np.isinf(_sl).any()):
                            has_inf = True
                        if has_nan and has_inf:
                            break
                per["has_nan"] = has_nan
                per["has_inf"] = has_inf
                if has_nan:
                    result["has_nan_count"] += 1
                    print(f"[WARNING] patient_id={pid} ct_hu.npy에 NaN 존재")
                if has_inf:
                    result["has_inf_count"] += 1
                    print(f"[WARNING] patient_id={pid} ct_hu.npy에 Inf 존재")

                # roi, lesion min/max
                per["roi_min"] = float(roi_mmap.min())
                per["roi_max"] = float(roi_mmap.max())
                per["lesion_min"] = float(lesion_mmap.min())
                per["lesion_max"] = float(lesion_mmap.max())

            except Exception as e:
                print(f"[WARNING] patient_id={pid} 파일 검증 중 오류: {e}")

        result["per_patient"][pid] = per

    return result


# ---------------------------------------------------------------------------
# context 좌표 검증
# ---------------------------------------------------------------------------


def validate_context_coords(merged_df: pd.DataFrame, patient_folder_map: dict, context_size: int) -> dict:
    """
    전체 merged_df rows에 대해 z_center 범위, context coord 계산 및 clip 여부 검증.
    patient_folder_map: {patient_id -> (folder_path, pattern_label)}
    반환: 검증 결과 dict
    """
    result = {
        "total_rows": len(merged_df),
        "z_out_of_range_count": 0,
        "clipped_count": 0,
        "padding_required_count": 0,
        "ok_count": 0,
        "min_context_h": None,
        "min_context_w": None,
        "clip_note": "padding 방식: volume 경계 초과분은 pad로 채워 context_size×context_size 보장",
    }

    # 환자별 shape 캐시
    shape_cache = {}

    min_h = None
    min_w = None

    for _, row in merged_df.iterrows():
        pid = str(row["patient_id"])

        if pid not in patient_folder_map:
            continue

        # shape 캐싱
        if pid not in shape_cache:
            folder = patient_folder_map[pid][0]
            ct_path = folder / "ct_hu.npy"
            if not ct_path.is_file():
                continue
            try:
                ct_mmap = np.load(str(ct_path), mmap_mode="r")
                shape_cache[pid] = tuple(ct_mmap.shape)
            except Exception:
                continue

        vol_shape = shape_cache[pid]  # (Z, Y, X)
        z_val = row.get("z_center")
        if z_val is None or (isinstance(z_val, float) and np.isnan(z_val)):
            continue

        z_center = int(z_val)
        Z = vol_shape[0]

        if not (0 <= z_center < Z):
            result["z_out_of_range_count"] += 1
            continue

        coords = compute_context_coords(row, context_size, vol_shape)
        was_clipped = coords["was_clipped"]

        # padding 후 항상 context_size×context_size 보장
        h = context_size
        w = context_size

        if min_h is None or h < min_h:
            min_h = h
        if min_w is None or w < min_w:
            min_w = w

        if was_clipped:
            result["clipped_count"] += 1
            result["padding_required_count"] += 1
        else:
            result["ok_count"] += 1

    result["min_context_h"] = min_h if min_h is not None else 0
    result["min_context_w"] = min_w if min_w is not None else 0

    return result


# ---------------------------------------------------------------------------
# preflight 보고
# ---------------------------------------------------------------------------


def run_preflight(
    qa_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    patient_folder_map: dict,
    volume_source_root: Path,
    output_root: Path,
    args,
    context_size: int = 192,
) -> dict:
    """preflight 조사 결과를 출력하고 dict를 반환."""
    report = {}

    print("\n" + "=" * 60)
    print("PREFLIGHT REPORT")
    print("=" * 60)

    # 입력 파일 존재 확인
    input_files = {
        "QA_MANIFEST": QA_MANIFEST,
        "CROP_MANIFEST": CROP_MANIFEST,
        "PNG_INDEX": PNG_INDEX,
        "LABEL_SHEET": LABEL_SHEET,
    }
    for label, path in input_files.items():
        exists = Path(path).is_file()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label}: {path}")
        report[label] = status

    # patient 매칭 확인 (패턴 포함)
    print(f"\n  Patient 매칭 결과 ({len(patient_folder_map)}명):")
    pattern_count = {}
    for pid, (folder, pattern_label) in patient_folder_map.items():
        print(f"    {pid} → {folder.name}  [pattern={pattern_label}]")
        pattern_count[pattern_label] = pattern_count.get(pattern_label, 0) + 1

    print(f"\n  Patient matching pattern별 count: {pattern_count}")
    report["pattern_count"] = pattern_count

    # join 확인
    n_null_z = merged_df["z_center"].isnull().sum()
    print(f"\n  join 후 z_center 결측: {n_null_z}행")
    report["join_z_center_null"] = int(n_null_z)

    # qa_group 분포
    group_counts = merged_df["qa_group"].value_counts().to_dict()
    print(f"\n  qa_group 분포: {group_counts}")
    report["qa_group_counts"] = group_counts

    # volume root 존재
    vols_npy = volume_source_root / "volumes_npy"
    vols_ok = vols_npy.is_dir()
    print(f"\n  volumes_npy 존재: {'OK' if vols_ok else 'MISSING'}")
    report["volumes_npy_exists"] = vols_ok

    # 출력 root 존재 여부
    out_exists = output_root.exists()
    print(f"  OUTPUT_ROOT 존재: {'YES' if out_exists else 'NO'} — {output_root}")
    report["output_root_exists"] = out_exists

    # volume 파일 검증
    print("\n  [VOLUME FILE VALIDATION]")
    vol_val = validate_volume_files_for_patients(patient_folder_map)
    print(f"    total_patients       : {vol_val['total_patients']}")
    print(f"    ct_exists_count      : {vol_val['ct_exists_count']}")
    print(f"    roi_exists_count     : {vol_val['roi_exists_count']}")
    print(f"    lesion_exists_count  : {vol_val['lesion_exists_count']}")
    print(f"    all_files_ok_count   : {vol_val['all_files_ok_count']}")
    print(f"    shape_mismatch_count : {vol_val['shape_mismatch_count']}")
    print(f"    has_nan_count        : {vol_val['has_nan_count']}")
    print(f"    has_inf_count        : {vol_val['has_inf_count']}")
    print(f"    dtype_warning_count  : {vol_val['dtype_warning_count']}")
    report["volume_validation"] = {
        k: v for k, v in vol_val.items() if k != "per_patient"
    }

    # context 좌표 검증
    print("\n  [CONTEXT COORD VALIDATION]")
    coord_val = validate_context_coords(merged_df, patient_folder_map, context_size)
    print(f"    total_rows           : {coord_val['total_rows']}")
    print(f"    z_out_of_range_count : {coord_val['z_out_of_range_count']}")
    print(f"    clipped_count        : {coord_val['clipped_count']}")
    print(f"    padding_required_count: {coord_val['padding_required_count']}")
    print(f"    ok_count             : {coord_val['ok_count']}")
    print(f"    min_context_h        : {coord_val['min_context_h']}")
    print(f"    min_context_w        : {coord_val['min_context_w']}")
    print(f"    clip_note            : {coord_val['clip_note']}")
    report["coord_validation"] = coord_val

    # volume source note
    print("\n  [NOTE] volume source에서는 CSV/manifest/score 파일을 읽지 않았음")

    print("=" * 60)
    print("PREFLIGHT 완료. PNG 생성 없음.\n")
    return report


# ---------------------------------------------------------------------------
# dry-run 보고
# ---------------------------------------------------------------------------


def run_dry_run(
    merged_df: pd.DataFrame,
    patient_folder_map: dict,
    context_size: int,
    max_per_sheet: int = 25,
    max_total: int = None,
):
    """dry-run: 파일 존재만 확인, 실제 array 로드 없음."""
    print("\n" + "=" * 60)
    print("DRY-RUN REPORT")
    print("=" * 60)

    ok_count = 0
    error_count = 0

    # max_total 적용
    rows_to_process = merged_df
    if max_total is not None:
        rows_to_process = merged_df.head(int(max_total))

    processing_rows = len(rows_to_process)
    print(f"\n  processing rows (max_total 적용 후): {processing_rows}")
    print(f"  expected individual PNG count       : {processing_rows}")

    # expected contact sheet count 계산 (qa_group별)
    qa_group_row_counts = rows_to_process["qa_group"].value_counts().to_dict()
    total_contact_sheets = sum(
        math.ceil(cnt / max_per_sheet) for cnt in qa_group_row_counts.values()
    )
    print(f"  expected contact sheet count        : {total_contact_sheets}")
    print(f"  context_size                        : {context_size}")
    print(f"  matched patient count               : {len(patient_folder_map)}")

    # volume shape sample (처음 환자 1명)
    if patient_folder_map:
        first_pid = next(iter(patient_folder_map))
        first_folder = patient_folder_map[first_pid][0]
        ct_sample_path = first_folder / "ct_hu.npy"
        if ct_sample_path.is_file():
            try:
                ct_sample = np.load(str(ct_sample_path), mmap_mode="r")
                print(f"  volume shape sample ({first_pid}): {ct_sample.shape}")
            except Exception as e:
                print(f"  volume shape sample 로드 실패: {e}")
        else:
            print(f"  volume shape sample: ct_hu.npy 없음 ({first_pid})")

    # context coord clipped count
    coord_val = validate_context_coords(rows_to_process, patient_folder_map, context_size)
    print(f"  context coord clipped_count         : {coord_val['clipped_count']}")
    print(f"  context coord padding_required_count: {coord_val['padding_required_count']}")
    print(f"  context coord z_out_of_range_count  : {coord_val['z_out_of_range_count']}")
    print(f"  context min_h x min_w               : {coord_val['min_context_h']} x {coord_val['min_context_w']}")

    for _, row in rows_to_process.iterrows():
        pid = str(row["patient_id"])
        crop_id = row["crop_id"]

        if pid not in patient_folder_map:
            print(f"  [SKIP] patient_id={pid} 폴더 매핑 없음")
            error_count += 1
            continue

        folder = patient_folder_map[pid][0]
        ct_path = folder / "ct_hu.npy"
        roi_path = folder / "roi_0_0.npy"
        lesion_path = folder / "lesion_mask_roi_0_0.npy"

        missing = [p for p in [ct_path, roi_path, lesion_path] if not p.is_file()]
        if missing:
            print(f"  [MISSING] crop_id={crop_id}, 누락 파일: {[m.name for m in missing]}")
            error_count += 1
        else:
            ok_count += 1

    print(f"\n  dry-run 결과: OK={ok_count}, ERROR={error_count}")
    print("  output files 생성 없음 확인")
    print("  PNG 생성 없음 확인")
    print("=" * 60)
    print("DRY-RUN 완료. PNG 생성 없음.\n")


# ---------------------------------------------------------------------------
# 메인 처리 루프
# ---------------------------------------------------------------------------


def process_crops(
    merged_df: pd.DataFrame,
    patient_folder_map: dict,
    output_root: Path,
    context_size: int,
    max_per_sheet: int,
    output_tag: str,
    force: bool,
    max_total: int = None,
    dry_run: bool = False,
) -> dict:
    """
    각 crop에 대해 context overlay PNG를 생성하고 manifest_png_index CSV,
    contact sheet, summary JSON을 저장한다.
    """
    individual_root = output_root / "individual_png"
    contact_root = output_root / "contact_sheets"
    individual_root.mkdir(parents=True, exist_ok=True)
    contact_root.mkdir(parents=True, exist_ok=True)

    index_rows = []
    generated_ok = 0
    generated_error = 0
    qa_group_counts = {}

    # 환자별 volume 캐시 (최대 1명 — 교체 시 명시적 해제 및 gc)
    _vol_cache = {}

    rows_to_process = merged_df
    if max_total is not None:
        rows_to_process = merged_df.head(int(max_total))

    for _, row in rows_to_process.iterrows():
        pid = str(row["patient_id"])
        crop_id = row["crop_id"]
        crop_path_str = str(row["crop_path"])
        qa_group = str(row.get("qa_group", "unknown"))
        qa_priority = row.get("qa_priority", "")
        z_center = int(row["z_center"])

        # qa_group 개수 집계
        qa_group_counts[qa_group] = qa_group_counts.get(qa_group, 0) + 1

        # 출력 PNG 경로
        group_dir = individual_root / qa_group
        group_dir.mkdir(parents=True, exist_ok=True)
        png_filename = (
            f"{output_tag}__{pid}__crop{crop_id}__z{z_center}.png"
        )
        png_path = group_dir / png_filename

        # skip 체크 (force 아닐 때)
        if png_path.exists() and not force:
            print(f"[SKIP] 이미 존재: {png_path.name}")
            index_rows.append(
                _make_index_row(
                    row, png_path, None, context_size, 0, 0, vol_shape=None,
                    was_clipped=False, generated=True, error=""
                )
            )
            generated_ok += 1
            continue

        # volume 로드 (최대 1명 캐시 — 교체 시 명시적 해제)
        if pid not in _vol_cache:
            # 기존 캐시 해제
            if _vol_cache:
                old_pid = next(iter(_vol_cache))
                old_vols = _vol_cache.pop(old_pid)
                del old_vols
                gc.collect()
                print(f"[INFO] released volume arrays for patient_id={old_pid}")

            if pid not in patient_folder_map:
                msg = f"patient_id={pid} 폴더 매핑 없음"
                print(f"[ERROR] {msg}")
                index_rows.append(
                    _make_index_row(
                        row, png_path, None, context_size, 0, 0, vol_shape=None,
                        was_clipped=False, generated=False, error=msg
                    )
                )
                generated_error += 1
                continue

            print(f"[INFO] processing patient_id={pid}")
            try:
                ct_hu, roi, lesion = load_volume_arrays(patient_folder_map[pid][0])
                _vol_cache[pid] = (ct_hu, roi, lesion)
            except SystemExit:
                raise
            except Exception as e:
                msg = f"volume 로드 실패: {e}"
                print(f"[ERROR] crop_id={crop_id}: {msg}")
                index_rows.append(
                    _make_index_row(
                        row, png_path, None, context_size, 0, 0, vol_shape=None,
                        was_clipped=False, generated=False, error=msg
                    )
                )
                generated_error += 1
                continue

        ct_hu, roi, lesion = _vol_cache[pid]
        vol_shape = ct_hu.shape  # (Z, Y, X)

        # z_center 범위 확인
        if z_center < 0 or z_center >= vol_shape[0]:
            msg = f"z_center={z_center} 범위 초과 (Z={vol_shape[0]})"
            print(f"[ERROR] crop_id={crop_id}: {msg}")
            index_rows.append(
                _make_index_row(
                    row, png_path, None, context_size, 0, 0, vol_shape=vol_shape,
                    was_clipped=False, generated=False, error=msg
                )
            )
            generated_error += 1
            continue

        # 좌표 계산
        coords = compute_context_coords(row, context_size, vol_shape)
        y_s, y_e = coords["y_start"], coords["y_end"]
        x_s, x_e = coords["x_start"], coords["x_end"]
        was_clipped = coords["was_clipped"]
        pad_top = coords["pad_top"]
        pad_bottom = coords["pad_bottom"]
        pad_left = coords["pad_left"]
        pad_right = coords["pad_right"]

        # slice 추출 (padding 방식으로 context_size×context_size 보장)
        ct_pad_value = int(LUNG_HU_MIN)
        ct_slice = extract_context_with_padding(ct_hu[z_center], coords, context_size, ct_pad_value)
        roi_slice = extract_context_with_padding(roi[z_center], coords, context_size, 0)
        lesion_slice = extract_context_with_padding(lesion[z_center], coords, context_size, 0)

        # shape 검증
        if ct_slice.shape != (context_size, context_size):
            msg = f"ct_slice shape mismatch: {ct_slice.shape}"
            print(f"[ERROR] crop_id={crop_id}: {msg}")
            index_rows.append(
                _make_index_row(
                    row, png_path, None, context_size, y_s, x_s, vol_shape=vol_shape,
                    was_clipped=was_clipped, pad_top=pad_top, pad_bottom=pad_bottom,
                    pad_left=pad_left, pad_right=pad_right,
                    generated=False, error=msg, y_e=y_e, x_e=x_e,
                )
            )
            generated_error += 1
            continue

        # PNG 생성
        try:
            row_meta = {
                "qa_group": qa_group,
                "qa_priority": qa_priority,
                "crop_score_l1_mean": row.get("crop_score_l1_mean"),
                "padim_score_mean": row.get("padim_score_mean"),
                "padim_score_max": row.get("padim_score_max"),
                "threshold_exceed_val_p90": row.get("threshold_exceed_val_p90"),
                "threshold_exceed_val_p95": row.get("threshold_exceed_val_p95"),
                "threshold_exceed_val_p99": row.get("threshold_exceed_val_p99"),
            }

            fig = make_context_overlay_figure(
                ct_slice, roi_slice, lesion_slice,
                pid, crop_id, z_center, row_meta
            )
            fig.savefig(str(png_path), dpi=100, bbox_inches="tight")
            plt.close(fig)

            print(f"[OK] {png_path.name}")
            index_rows.append(
                _make_index_row(
                    row, png_path, None, context_size,
                    y_s, x_s, vol_shape=vol_shape,
                    was_clipped=was_clipped, generated=True, error="",
                    y_e=y_e, x_e=x_e,
                    pad_top=pad_top, pad_bottom=pad_bottom,
                    pad_left=pad_left, pad_right=pad_right,
                )
            )
            generated_ok += 1

        except Exception as e:
            msg = f"PNG 생성 실패: {e}"
            print(f"[ERROR] crop_id={crop_id}: {msg}")
            index_rows.append(
                _make_index_row(
                    row, png_path, None, context_size,
                    y_s, x_s, vol_shape=vol_shape,
                    was_clipped=was_clipped, generated=False, error=msg,
                    y_e=y_e, x_e=x_e,
                    pad_top=pad_top, pad_bottom=pad_bottom,
                    pad_left=pad_left, pad_right=pad_right,
                )
            )
            generated_error += 1

    # contact sheet 생성
    index_df = pd.DataFrame(index_rows, columns=PNG_INDEX_COLUMNS)
    contact_sheet_count = 0
    contact_sheet_map = {}

    for group in sorted(qa_group_counts.keys()):
        group_rows = index_df[
            (index_df["qa_group"] == group) & (index_df["png_generated"] == True)
        ]
        group_pngs = group_rows["png_path"].dropna().tolist()
        if not group_pngs:
            continue

        sheets = build_contact_sheet(
            png_paths=group_pngs,
            output_path=contact_root,
            max_per_sheet=max_per_sheet,
            sheet_label=f"{group}_{output_tag}",
        )
        contact_sheet_count += len(sheets)
        # index_df에 contact_sheet_path 반영
        for i, sheet_path in enumerate(sheets):
            start_i = i * max_per_sheet
            end_i = min((i + 1) * max_per_sheet, len(group_pngs))
            batch_pngs = group_pngs[start_i:end_i]
            mask = index_df["png_path"].isin(batch_pngs)
            index_df.loc[mask, "contact_sheet_path"] = str(sheet_path)

    return {
        "index_df": index_df,
        "generated_ok": generated_ok,
        "generated_error": generated_error,
        "qa_group_counts": qa_group_counts,
        "contact_sheet_count": contact_sheet_count,
        "total_patients": len(patient_folder_map),
    }


def _make_index_row(
    row: pd.Series,
    png_path: Path,
    contact_sheet_path,
    context_size: int,
    y_s: int,
    x_s: int,
    vol_shape,
    was_clipped: bool,
    generated: bool,
    error: str,
    y_e: int = 0,
    x_e: int = 0,
    pad_top: int = 0,
    pad_bottom: int = 0,
    pad_left: int = 0,
    pad_right: int = 0,
) -> dict:
    """manifest_png_index 한 행 dict 생성."""
    return {
        "patient_id": row.get("patient_id", ""),
        "crop_id": row.get("crop_id", ""),
        "crop_path": row.get("crop_path", ""),
        "png_path": str(png_path) if png_path else "",
        "contact_sheet_path": str(contact_sheet_path) if contact_sheet_path else "",
        "qa_group": row.get("qa_group", ""),
        "qa_priority": row.get("qa_priority", ""),
        "crop_score_l1_mean": row.get("crop_score_l1_mean", ""),
        "padim_score_mean": row.get("padim_score_mean", ""),
        "padim_score_max": row.get("padim_score_max", ""),
        "threshold_exceed_val_p90": row.get("threshold_exceed_val_p90", ""),
        "threshold_exceed_val_p95": row.get("threshold_exceed_val_p95", ""),
        "threshold_exceed_val_p99": row.get("threshold_exceed_val_p99", ""),
        "context_size_used": context_size,
        "z_center": row.get("z_center", ""),
        "y_context_start": y_s,
        "y_context_end": y_e,
        "x_context_start": x_s,
        "x_context_end": x_e,
        "was_clipped": was_clipped,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "png_generated": generated,
        "generation_error": error,
    }


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hard Negative Top-Score Context Overlay PNG Pack 생성"
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="preflight 조사만 수행, PNG 생성 없음",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일 생성 없이 로직 확인",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 output 덮어쓰기 허용",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="처리할 최대 crop 수 (기본값 없음 = 전체)",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=192,
        help="context 크기 (기본값 192)",
    )
    parser.add_argument(
        "--max-per-sheet",
        type=int,
        default=25,
        help="contact sheet당 최대 개수 (기본값 25)",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="context_overlay_v1",
        help="output 폴더/파일명 태그 (기본값 context_overlay_v1)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    output_root = OUTPUT_ROOT
    output_tag = args.output_tag
    context_size = args.context_size
    max_per_sheet = args.max_per_sheet

    # ------------------------------------------------------------------
    # guard 체크 (최우선)
    # ------------------------------------------------------------------
    run_path_guards(output_root, VOLUME_SOURCE_ROOT)

    # ------------------------------------------------------------------
    # 입력 파일 로드
    # ------------------------------------------------------------------
    qa_df = load_qa_manifest()
    crop_df = load_crop_manifest()
    _png_index_df = load_png_index()   # 선택 입력, 현재 참고용
    _label_sheet_df = load_label_sheet()  # 선택 입력, 현재 참고용

    # ------------------------------------------------------------------
    # guard: patient_id 키워드 체크
    # ------------------------------------------------------------------
    guard_patient_ids(qa_df)

    # ------------------------------------------------------------------
    # guard: crop_path 확장자 체크
    # ------------------------------------------------------------------
    guard_crop_paths(qa_df)

    # ------------------------------------------------------------------
    # guard: crop_id 중복 체크
    # ------------------------------------------------------------------
    guard_crop_id_unique(qa_df)

    # ------------------------------------------------------------------
    # join
    # ------------------------------------------------------------------
    merged_df = join_manifests(qa_df, crop_df)

    # join 후 crop_id 중복 재확인
    guard_crop_id_unique(merged_df)

    # ------------------------------------------------------------------
    # patient 폴더 매핑
    # ------------------------------------------------------------------
    patient_ids = merged_df["patient_id"].dropna().unique().tolist()
    patient_folder_map = build_patient_folder_map(patient_ids, VOLUME_SOURCE_ROOT)

    # ------------------------------------------------------------------
    # preflight 모드
    # ------------------------------------------------------------------
    if args.preflight_only:
        run_preflight(
            qa_df, merged_df, patient_folder_map,
            VOLUME_SOURCE_ROOT, output_root, args,
            context_size=context_size,
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # dry-run 모드
    # ------------------------------------------------------------------
    if args.dry_run:
        run_preflight(
            qa_df, merged_df, patient_folder_map,
            VOLUME_SOURCE_ROOT, output_root, args,
            context_size=context_size,
        )
        run_dry_run(
            merged_df, patient_folder_map, context_size,
            max_per_sheet=max_per_sheet,
            max_total=args.max_total,
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # full-run: 기존 output root 존재 시 확인 (--force 없으면 중단)
    # ------------------------------------------------------------------
    if output_root.exists() and not args.force:
        print(
            f"[ERROR] OUTPUT_ROOT 이미 존재합니다. --force 옵션을 사용하거나 경로를 확인하세요: {output_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 처리 루프
    # ------------------------------------------------------------------
    result = process_crops(
        merged_df=merged_df,
        patient_folder_map=patient_folder_map,
        output_root=output_root,
        context_size=context_size,
        max_per_sheet=max_per_sheet,
        output_tag=output_tag,
        force=args.force,
        max_total=args.max_total,
    )

    index_df = result["index_df"]
    generated_ok = result["generated_ok"]
    generated_error = result["generated_error"]
    qa_group_counts = result["qa_group_counts"]
    contact_sheet_count = result["contact_sheet_count"]
    total_patients = result["total_patients"]

    # ------------------------------------------------------------------
    # manifest_png_index CSV 저장
    # ------------------------------------------------------------------
    index_csv_path = output_root / f"manifest_png_index_{output_tag}.csv"
    index_df.to_csv(str(index_csv_path), index=False)
    print(f"[INFO] manifest_png_index 저장: {index_csv_path}")

    # ------------------------------------------------------------------
    # summary JSON 저장
    # ------------------------------------------------------------------
    summary = {
        "output_tag": output_tag,
        "context_size": context_size,
        "total_crops": len(merged_df) if args.max_total is None else min(args.max_total, len(merged_df)),
        "total_patients": total_patients,
        "generated_ok": generated_ok,
        "generated_error": generated_error,
        "qa_group_counts": qa_group_counts,
        "contact_sheet_count": contact_sheet_count,
        "memory_strategy": "max_cache_size_1",
        "volume_load_mode": "mmap_mode_r",
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_path = output_root / f"summary_{output_tag}.json"
    with open(str(summary_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] summary JSON 저장: {summary_path}")

    # ------------------------------------------------------------------
    # 최종 보고
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FULL-RUN 완료")
    print(f"  output_tag      : {output_tag}")
    print(f"  context_size    : {context_size}")
    print(f"  total_crops     : {summary['total_crops']}")
    print(f"  total_patients  : {total_patients}")
    print(f"  generated_ok    : {generated_ok}")
    print(f"  generated_error : {generated_error}")
    print(f"  contact_sheets  : {contact_sheet_count}")
    print(f"  output_root     : {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
