"""
build_normal_padim_ready_roi0_0_from_final_all.py

Position-Aware PaDiM v2/v2 실험을 위한 정상 training-ready 데이터셋 생성 스크립트.

입력:
  - final_all CT (ct_1mm_lung_range/Normal/*.nii.gz)
  - final_all 폐엽 mask 5개 (organ_masks_1mm/Normal/{patient_id}/*.nii.gz)
  - v1 reference split CSV (manifests/train_val_test_split.csv)

출력:
  - volumes_npy/{safe_id}/ct_hu.npy         (int16)
  - volumes_npy/{safe_id}/roi_0_0.npy       (uint8, 0/1)
  - volumes_npy/{safe_id}/meta.json
  - volumes_npy/{safe_id}/.done
  - patch_index_by_patient/{safe_id}.csv
  - logs/error.csv
  - logs/runtime_summary.csv
  - manifests/patient_manifest.csv
  - manifests/train_val_test_split.csv
  - manifests/patch_count_by_patient.csv

저장 금지: pure_lung.npy, model_roi.npy, organ_exclusion.npy, lesion 관련 파일, nii.gz 복사본
dilate: 없음

실행 환경:
  Windows ct_core: /mnt/c/Users/jinhy/anaconda3/envs/ct_core/python.exe <script>
  (nibabel 이 ai_env 에 없으므로 ct_core python.exe 를 사용)
  경로 인자는 /mnt/c/... 형식(WSL 스타일)으로 지정하면 내부에서 자동 변환됨.
"""

import argparse
import csv
import json
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path, PureWindowsPath

import nibabel as nib
import numpy as np


# ============================================================
# OS 환경 감지 및 경로 변환 헬퍼
# ============================================================

_IS_WINDOWS = platform.system() == "Windows"


def _wsl_to_win(s: str) -> str:
    """
    /mnt/c/foo/bar  →  C:\\foo\\bar
    그 외 문자열은 그대로 반환.
    """
    if s.startswith("/mnt/") and len(s) > 5:
        drive = s[5].upper()           # 'c' → 'C'
        rest = s[6:].replace("/", "\\")  # '/foo/bar' → '\\foo\\bar'
        return drive + ":" + rest
    return s


def to_str(path) -> str:
    """
    open(), np.save(), nibabel.load() 등 OS API에 직접 넘길 경로 문자열.
    Windows 에서 /mnt/c/... 는 C:\\... 로 변환.

    주의: Windows Python 에서 Path('/mnt/c/...') 를 str() 하면 '\\mnt\\c\\...'
    가 되므로, as_posix() 를 통해 항상 forward-slash 문자열로 먼저 추출한다.
    """
    # Path 객체이면 as_posix() 로 POSIX 형식 유지, 그 외 str() 사용
    if hasattr(path, "as_posix"):
        s = path.as_posix()
    else:
        s = str(path)
    if _IS_WINDOWS:
        return _wsl_to_win(s)
    return s


def osp(path) -> Path:
    """
    Path 객체의 .exists(), .mkdir(), .glob(), .rename(), .write_text() 등을
    사용하기 전에 호출해야 하는 OS-native Path 반환.
    Windows: /mnt/c/... → Path('C:\\...')
    Linux:   그대로 Path(path) 반환.
    """
    return Path(to_str(path))


def join_path(*parts) -> Path:
    """
    경로 조각들을 연결하여 논리적 Path 를 반환.
    (내부 표현은 /mnt/c/... 형식 유지 가능 – osp() 를 통해서만 OS 에 전달)
    """
    result = Path(parts[0])
    for p in parts[1:]:
        result = result / p
    return result


# ============================================================
# 경로 헬퍼
# ============================================================

def get_ct_path(final_all_root: Path, patient_id: str) -> Path:
    return final_all_root / "ct_1mm_lung_range" / "Normal" / f"{patient_id}_1mm_lung_range_lps.nii.gz"


def get_lobe_mask_paths(final_all_root: Path, patient_id: str) -> dict:
    base = final_all_root / "organ_masks_1mm" / "Normal" / patient_id
    names = [
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
    ]
    return {name: base / f"{name}.nii.gz" for name in names}


# ============================================================
# 저장공간 확인
# ============================================================

def get_free_gb(path: Path) -> float:
    """path 가 포함된 파티션의 여유 공간 (GB).
    존재하지 않는 경로면 부모를 따라 올라간다.
    """
    check = osp(path)
    # 부모를 따라 올라가면서 존재하는 경로 찾기
    while not check.exists():
        check = check.parent
    usage = shutil.disk_usage(str(check))
    return usage.free / (1024 ** 3)


def check_free_space(output_root: Path, min_free_gb: float, label: str = ""):
    free = get_free_gb(output_root)
    tag = f"[{label}] " if label else ""
    if free < min_free_gb:
        print(f"\n[ABORT] {tag}저장공간 부족: {free:.1f} GB < {min_free_gb:.1f} GB (min-free-gb)")
        sys.exit(1)
    return free


# ============================================================
# v1 split CSV 로드
# ============================================================

def load_v1_split(v1_reference_root: Path) -> dict:
    """patient_id -> {'split': ..., 'safe_id': ...}"""
    csv_path = v1_reference_root / "manifests" / "train_val_test_split.csv"
    if not osp(csv_path).exists():
        print(f"[ERROR] v1 split CSV 없음: {csv_path}")
        sys.exit(1)
    mapping = {}
    with open(to_str(csv_path), encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"].strip()
            mapping[pid] = {
                "split": row["split"].strip(),
                "safe_id": row["safe_id"].strip(),
            }
    print(f"[INFO] v1 split CSV 로드 완료: {len(mapping)}명")
    return mapping


# ============================================================
# roi_0_0 생성 (폐엽 5개 union, dilate 없음)
# ============================================================

def build_roi_0_0(
    final_all_root: Path,
    patient_id: str,
    ct_shape_zyx: tuple,
    ct_img: "nib.Nifti1Image",
) -> np.ndarray:
    """
    폐엽 mask 5개를 logical OR 후 uint8 반환.
    CT lung_range와 lobe mask의 Z 범위가 다를 경우 affine/origin 기반으로 crop.
    X/Y mismatch, crop 범위 초과, roi voxel 0 시 ValueError 발생.
    """
    CT_Z_OFFSET_TOLERANCE = 0.1  # voxel 단위 허용 오차

    ct_aff = ct_img.affine
    ct_zooms = ct_img.header.get_zooms()[:3]  # (sx, sy, sz)

    lobe_paths = get_lobe_mask_paths(final_all_root, patient_id)
    roi = np.zeros(ct_shape_zyx, dtype=bool)

    for name, mpath in lobe_paths.items():
        # lobe mask 하나라도 없으면 error (조용히 skip 금지)
        if not osp(mpath).exists():
            raise ValueError(
                f"lobe mask 없음 (처리 불가): {name} ({mpath.name})"
            )
        mask_img = nib.load(to_str(mpath))
        arr = mask_img.get_fdata(dtype=np.float32) > 0.5
        # nibabel: (X, Y, Z) → ZYX
        arr_zyx = arr.transpose(2, 1, 0)

        # shape 일치 여부와 무관하게 모든 mask에 대해 spacing/direction 먼저 확인
        mask_aff = mask_img.affine
        mask_zooms = mask_img.header.get_zooms()[:3]

        # spacing 일치 확인
        if not np.allclose(ct_zooms[:3], mask_zooms[:3], atol=0.01):
            raise ValueError(
                f"spacing 불일치: {name} CT={ct_zooms} mask={mask_zooms}"
            )

        # direction/orientation 일치 확인
        # ct_aff[:3,:3] / ct_zooms → column j를 ct_zooms[j]로 나눔 → normalized direction matrix
        ct_dir = ct_aff[:3, :3] / ct_zooms
        mask_dir = mask_aff[:3, :3] / mask_zooms
        if not np.allclose(ct_dir, mask_dir, atol=1e-3):
            raise ValueError(
                f"direction/orientation 불일치 (보정 불가): {name} "
                f"ct_dir={ct_dir.tolist()} mask_dir={mask_dir.tolist()}"
            )

        if arr_zyx.shape == ct_shape_zyx:
            roi |= arr_zyx
            continue

        # X/Y mismatch → 임의 보정 금지
        if arr_zyx.shape[1] != ct_shape_zyx[1] or arr_zyx.shape[2] != ct_shape_zyx[2]:
            raise ValueError(
                f"lobe mask X/Y 불일치 (보정 불가): {name} "
                f"mask_ZYX={arr_zyx.shape} vs CT_ZYX={ct_shape_zyx}"
            )

        # Z 크기만 다른 경우 → affine 기반 crop (mask_aff, mask_zooms는 위에서 추출)

        # Z 방향 origin 기반 offset 계산
        ct_z_origin = ct_aff[2, 3]
        mask_z_origin = mask_aff[2, 3]
        z_spacing = float(ct_zooms[2])
        raw_offset = (ct_z_origin - mask_z_origin) / z_spacing
        z_offset = int(round(raw_offset))

        if abs(raw_offset - z_offset) > CT_Z_OFFSET_TOLERANCE:
            raise ValueError(
                f"Z offset이 정수에서 {abs(raw_offset - z_offset):.3f} voxel 벗어남: "
                f"{name}, raw_offset={raw_offset:.4f}"
            )

        z_start = z_offset
        z_end = z_offset + ct_shape_zyx[0]

        if z_start < 0 or z_end > arr_zyx.shape[0]:
            raise ValueError(
                f"crop 범위가 mask 배열 밖: {name} "
                f"z_start={z_start}, z_end={z_end}, mask_Z={arr_zyx.shape[0]}"
            )

        cropped = arr_zyx[z_start:z_end, :, :]
        if cropped.shape != ct_shape_zyx:
            raise ValueError(
                f"crop 후 shape 불일치: {name} {cropped.shape} vs {ct_shape_zyx}"
            )
        roi |= cropped

    if roi.sum() == 0:
        raise ValueError(
            f"roi_0_0 voxel 수가 0: 폐엽 mask 5개 모두 비어있거나 crop 후 내용 없음 (patient_id={patient_id})"
        )

    return roi.astype(np.uint8)


# ============================================================
# patch CSV 생성
# ============================================================

PATCH_SIZE = 32
PATCH_STRIDE = 16
MIN_ROI_RATIO = 0.50


def _position_bin(cy: float, cx: float, H: float, W: float) -> str:
    """y,x 중심 기준 3x3 grid → 'top_left' 등 9개 bin."""
    ry = cy / H
    rx = cx / W
    if ry < 1 / 3:
        row = "top"
    elif ry < 2 / 3:
        row = "middle"
    else:
        row = "bottom"
    if rx < 1 / 3:
        col = "left"
    elif rx < 2 / 3:
        col = "center"
    else:
        col = "right"
    return f"{row}_{col}"


def _z_level(z_ratio: float) -> str:
    if z_ratio < 0.33:
        return "upper"
    elif z_ratio < 0.66:
        return "middle"
    return "lower"


def generate_patch_rows(
    ct_shape_zyx: tuple,
    roi_0_0: np.ndarray,
    patient_id: str,
    safe_id: str,
    split: str,
) -> list:
    """
    roi_0_0 기준 patch 생성.
    반환: list of dict (patch CSV 한 행씩)
    """
    Z, H, W = ct_shape_zyx
    rows = []

    for local_z in range(Z):
        slice_roi = roi_0_0[local_z]  # (H, W) uint8
        slice_roi_pixels = int(slice_roi.sum())
        slice_roi_0_0_ratio = slice_roi_pixels / (H * W)

        # slice 에 roi 픽셀이 전혀 없으면 skip
        if slice_roi_pixels == 0:
            continue

        # centroid of roi on this slice
        ys, xs = np.where(slice_roi > 0)
        cy_roi = float(ys.mean())
        cx_roi = float(xs.mean())
        # bounding box 단변 절반 (central_peripheral 판정 반경)
        bbox_h = int(ys.max() - ys.min()) + 1
        bbox_w = int(xs.max() - xs.min()) + 1
        radius = min(bbox_h, bbox_w) / 2.0

        z_ratio = local_z / max(Z - 1, 1)
        level = _z_level(z_ratio)

        # 이미지 대각선 절반 (central_distance_ratio 분모)
        diag_half = np.sqrt(H ** 2 + W ** 2) / 2.0

        for y0 in range(0, H - PATCH_SIZE + 1, PATCH_STRIDE):
            for x0 in range(0, W - PATCH_SIZE + 1, PATCH_STRIDE):
                y1 = y0 + PATCH_SIZE
                x1 = x0 + PATCH_SIZE

                patch_roi = slice_roi[y0:y1, x0:x1]
                roi_pixels = int(patch_roi.sum())
                roi_ratio = roi_pixels / (PATCH_SIZE * PATCH_SIZE)

                if roi_ratio < MIN_ROI_RATIO:
                    continue

                # patch 중심
                cy_patch = y0 + PATCH_SIZE / 2.0
                cx_patch = x0 + PATCH_SIZE / 2.0

                grid_pos_bin = _position_bin(cy_patch, cx_patch, H, W)

                # central_peripheral
                dist_center = np.sqrt((cy_patch - cy_roi) ** 2 + (cx_patch - cx_roi) ** 2)
                central_peripheral = "central" if dist_center <= radius else "peripheral"

                # PaDiMModel 기대 형식: {z_level}_{central_peripheral}
                pos_bin = f"{level}_{central_peripheral}"

                # central_distance_ratio_mean
                py, px = np.mgrid[y0:y1, x0:x1]
                pixel_dists = np.sqrt((py - cy_roi) ** 2 + (px - cx_roi) ** 2)
                central_distance_ratio_mean = float(pixel_dists.mean() / diag_half) if diag_half > 0 else 0.0

                left_right = "left" if cx_patch < W / 2.0 else "right"

                rows.append({
                    "patient_id": patient_id,
                    "safe_id": safe_id,
                    "group": split,
                    "local_z": local_z,
                    "slice_index": local_z,
                    "y0": y0,
                    "x0": x0,
                    "y1": y1,
                    "x1": x1,
                    "patch_size": PATCH_SIZE,
                    "patch_stride": PATCH_STRIDE,
                    "roi_0_0_pixels": roi_pixels,
                    "roi_0_0_patch_ratio": round(roi_ratio, 6),
                    "slice_roi_0_0_ratio": round(slice_roi_0_0_ratio, 6),
                    "position_bin": pos_bin,
                    "grid_position_bin": grid_pos_bin,
                    "z_level": level,
                    "z_ratio": round(z_ratio, 6),
                    "central_peripheral": central_peripheral,
                    "central_distance_ratio_mean": round(central_distance_ratio_mean, 6),
                    "left_right_metadata": left_right,
                })
    return rows


PATCH_CSV_COLUMNS = [
    "patient_id", "safe_id", "group", "local_z", "slice_index",
    "y0", "x0", "y1", "x1", "patch_size", "patch_stride",
    "roi_0_0_pixels", "roi_0_0_patch_ratio", "slice_roi_0_0_ratio",
    "position_bin", "grid_position_bin", "z_level", "z_ratio",
    "central_peripheral", "central_distance_ratio_mean", "left_right_metadata",
]


# ============================================================
# atomic write 헬퍼
# ============================================================

def atomic_npy_save(arr: np.ndarray, dest: Path):
    """임시 파일에 저장 후 replace (atomic write, 기존 파일 덮어쓰기 안전)."""
    # np.save()는 확장자가 .npy로 끝나지 않으면 자동으로 .npy를 붙이므로
    # tmp 파일도 .npy로 끝내야 실제 생성 파일명과 일치한다.
    tmp_path = Path(str(dest) + ".tmp.npy")
    np.save(to_str(tmp_path), arr)
    osp(tmp_path).replace(osp(dest))


def atomic_json_save(obj: dict, dest: Path):
    tmp_path = Path(str(dest) + ".tmp")
    with open(to_str(tmp_path), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    osp(tmp_path).replace(osp(dest))


def atomic_csv_save(rows: list, dest: Path, columns: list):
    tmp_path = Path(str(dest) + ".tmp")
    with open(to_str(tmp_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    osp(tmp_path).replace(osp(dest))


# ============================================================
# error.csv / runtime_summary.csv append
# ============================================================

ERROR_CSV_COLUMNS = ["patient_id", "safe_id", "error_type", "error_message", "timestamp"]
RUNTIME_CSV_COLUMNS = [
    "patient_id", "safe_id", "split", "status",
    "ct_shape_zyx", "roi_0_0_voxels", "patch_count",
    "elapsed_seconds", "timestamp",
]


def append_error(error_csv: Path, patient_id: str, safe_id: str, error_type: str, error_message: str):
    write_header = not osp(error_csv).exists()
    with open(to_str(error_csv), "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "patient_id": patient_id,
            "safe_id": safe_id,
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
        })


def append_runtime(runtime_csv: Path, row: dict):
    write_header = not osp(runtime_csv).exists()
    with open(to_str(runtime_csv), "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# dry-run: 단일 환자 예상 용량 계산
# ============================================================

def estimate_patient_size_mb(ct_path: Path) -> float:
    """nibabel 로 shape 읽어서 ct_hu(int16) + roi_0_0(uint8) 예상 용량 MB 반환."""
    try:
        img = nib.load(to_str(ct_path))
        shape = img.shape  # (X, Y, Z) nibabel convention
        total_voxels = int(np.prod(shape))
        # int16 = 2 bytes, uint8 = 1 byte
        mb = (total_voxels * 2 + total_voxels * 1) / (1024 ** 2)
        return mb
    except Exception:
        return 150.0  # 추정 불가 시 기본값


# ============================================================
# 단일 환자 처리 (실제 실행)
# ============================================================

def process_patient(
    patient_id: str,
    safe_id: str,
    split: str,
    final_all_root: Path,
    output_root: Path,
    error_csv: Path,
    runtime_csv: Path,
    resume: bool,
    dry_run: bool,
) -> dict:
    """
    반환: {'status': 'done'|'dry_run_ok'|'error',
           'patch_count': int, 'ct_shape_zyx': str, 'roi_0_0_voxels': int}
    resume skip 환자도 기존 파일에서 실제 값을 복원하여 status='done'으로 반환.
    """
    t0 = time.time()
    vol_dir = output_root / "volumes_npy" / safe_id
    done_marker = vol_dir / ".done"
    patch_dir = output_root / "patch_index_by_patient"
    patch_csv_path = patch_dir / f"{safe_id}.csv"

    result = {
        "status": "error",
        "patch_count": 0,
        "ct_shape_zyx": "",
        "roi_0_0_voxels": 0,
    }

    # resume: .done 있으면 기존 파일에서 실제 값 복원
    if resume and osp(done_marker).exists():
        elapsed = time.time() - t0
        patch_csv_existing = patch_dir / f"{safe_id}.csv"
        meta_path = vol_dir / "meta.json"
        roi_path = vol_dir / "roi_0_0.npy"

        # patch CSV 존재 및 행 수 확인
        if not osp(patch_csv_existing).exists():
            msg = f"resume skip이지만 patch CSV 없음: {patch_csv_existing}"
            print(f"  [ERROR] {msg}")
            if not dry_run:
                append_error(error_csv, patient_id, safe_id, "MissingPatchCSV", msg)
                append_runtime(runtime_csv, {
                    "patient_id": patient_id, "safe_id": safe_id, "split": split,
                    "status": "error", "ct_shape_zyx": "", "roi_0_0_voxels": 0,
                    "patch_count": 0, "elapsed_seconds": round(elapsed, 2),
                    "timestamp": datetime.now().isoformat(),
                })
            return result

        with open(to_str(patch_csv_existing), encoding="utf-8") as _f:
            patch_count_existing = sum(1 for _ in _f) - 1  # 헤더 제외

        if patch_count_existing == 0:
            msg = f"resume skip이지만 patch_count=0: {patch_csv_existing}"
            print(f"  [ERROR] {msg}")
            if not dry_run:
                append_error(error_csv, patient_id, safe_id, "ZeroPatchCount", msg)
                append_runtime(runtime_csv, {
                    "patient_id": patient_id, "safe_id": safe_id, "split": split,
                    "status": "error", "ct_shape_zyx": "", "roi_0_0_voxels": 0,
                    "patch_count": 0, "elapsed_seconds": round(elapsed, 2),
                    "timestamp": datetime.now().isoformat(),
                })
            return result

        # meta.json에서 shape 복원
        ct_shape_existing = ""
        if osp(meta_path).exists():
            with open(to_str(meta_path), encoding="utf-8") as _f:
                meta_data = json.load(_f)
            shape = meta_data.get("shape_zyx", [])
            if shape:
                ct_shape_existing = f"{shape[0]}x{shape[1]}x{shape[2]}"

        # roi_0_0.npy voxel 수 복원 (dry_run 제외)
        roi_voxels_existing = 0
        if not dry_run and osp(roi_path).exists():
            roi_arr = np.load(to_str(roi_path))
            roi_voxels_existing = int(roi_arr.sum())

        result["status"] = "done"
        result["patch_count"] = patch_count_existing
        result["ct_shape_zyx"] = ct_shape_existing
        result["roi_0_0_voxels"] = roi_voxels_existing

        if not dry_run:
            append_runtime(runtime_csv, {
                "patient_id": patient_id, "safe_id": safe_id, "split": split,
                "status": "done", "ct_shape_zyx": ct_shape_existing,
                "roi_0_0_voxels": roi_voxels_existing,
                "patch_count": patch_count_existing,
                "elapsed_seconds": round(elapsed, 2),
                "timestamp": datetime.now().isoformat(),
            })
        return result

    ct_path = get_ct_path(final_all_root, patient_id)

    # CT 파일 존재 확인
    if not osp(ct_path).exists():
        msg = f"CT 파일 없음: {ct_path}"
        print(f"  [ERROR] {msg}")
        if not dry_run:
            append_error(error_csv, patient_id, safe_id, "FileNotFound", msg)
            elapsed = time.time() - t0
            append_runtime(runtime_csv, {
                "patient_id": patient_id, "safe_id": safe_id, "split": split,
                "status": "error", "ct_shape_zyx": "", "roi_0_0_voxels": 0,
                "patch_count": 0, "elapsed_seconds": round(elapsed, 2),
                "timestamp": datetime.now().isoformat(),
            })
        return result

    try:
        # CT 로드 (nibabel: X,Y,Z → transpose to Z,Y,X)
        ct_img = nib.load(to_str(ct_path))
        ct_arr = ct_img.get_fdata(dtype=np.float32)
        ct_zyx = ct_arr.transpose(2, 1, 0).astype(np.int16)
        ct_shape_zyx = ct_zyx.shape  # (Z, H, W)

        spacing_xyz = ct_img.header.get_zooms()[:3]
        origin = list(ct_img.affine[:3, 3].tolist())
        direction = ct_img.affine[:3, :3].tolist()

        shape_str = f"{ct_shape_zyx[0]}x{ct_shape_zyx[1]}x{ct_shape_zyx[2]}"
        result["ct_shape_zyx"] = shape_str

        # roi_0_0 생성 (폐엽 union, dilate 없음, affine 기반 Z crop 포함)
        roi_0_0 = build_roi_0_0(final_all_root, patient_id, ct_shape_zyx, ct_img)
        roi_voxels = int(roi_0_0.sum())
        result["roi_0_0_voxels"] = roi_voxels

        if dry_run:
            result["status"] = "dry_run_ok"
            return result

        # 출력 디렉토리 생성
        osp(vol_dir).mkdir(parents=True, exist_ok=True)
        osp(patch_dir).mkdir(parents=True, exist_ok=True)

        # atomic write: ct_hu.npy
        atomic_npy_save(ct_zyx, vol_dir / "ct_hu.npy")

        # atomic write: roi_0_0.npy
        atomic_npy_save(roi_0_0, vol_dir / "roi_0_0.npy")

        # meta.json
        meta = {
            "patient_id": patient_id,
            "safe_id": safe_id,
            "split": split,
            "label": "normal",
            "shape_zyx": list(ct_shape_zyx),
            "spacing_xyz": [float(s) for s in spacing_xyz],
            "origin": origin,
            "direction": direction,
            "source_ct_path": ct_path.as_posix(),
            "roi_rule": "lobe_union_5_no_dilate",
            "patch_rule": f"size={PATCH_SIZE}_stride={PATCH_STRIDE}_min_roi={MIN_ROI_RATIO}",
        }
        atomic_json_save(meta, vol_dir / "meta.json")

        # patch CSV 생성
        patch_rows = generate_patch_rows(ct_shape_zyx, roi_0_0, patient_id, safe_id, split)
        patch_count = len(patch_rows)
        result["patch_count"] = patch_count

        if patch_count == 0:
            raise ValueError(
                f"patch_count=0: roi_0_0 기준 유효 patch 없음 (patient_id={patient_id})"
            )

        atomic_csv_save(patch_rows, patch_csv_path, PATCH_CSV_COLUMNS)

        # .done marker
        osp(done_marker).write_text(datetime.now().isoformat(), encoding="utf-8")

        result["status"] = "done"

    except ValueError as e:
        msg = str(e)
        print(f"  [ERROR] {msg}")
        if not dry_run:
            append_error(error_csv, patient_id, safe_id, "ShapeMismatch", msg)
        result["status"] = "error"
    except Exception as e:
        print(f"  [ERROR] {patient_id}: {e}")
        if not dry_run:
            append_error(error_csv, patient_id, safe_id, type(e).__name__, str(e))
        result["status"] = "error"

    if not dry_run:
        elapsed = time.time() - t0
        append_runtime(runtime_csv, {
            "patient_id": patient_id, "safe_id": safe_id, "split": split,
            "status": result["status"],
            "ct_shape_zyx": result["ct_shape_zyx"],
            "roi_0_0_voxels": result["roi_0_0_voxels"],
            "patch_count": result["patch_count"],
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": datetime.now().isoformat(),
        })

    return result


# ============================================================
# manifests 생성
# ============================================================

def write_manifests(output_root: Path, summary_rows: list):
    manifest_dir = output_root / "manifests"
    osp(manifest_dir).mkdir(parents=True, exist_ok=True)

    # manifest에는 status=="done"이고 patch_count>0인 환자만 포함
    manifest_rows = [r for r in summary_rows if r["status"] == "done" and r["patch_count"] > 0]

    # patient_manifest.csv
    pm_cols = [
        "patient_id", "safe_id", "split", "status", "patch_count", "volume_dir",
        "ct_hu_npy", "roi_0_0_npy", "meta_json", "patch_csv", "pure_lung_npy",
    ]
    pm_rows = []
    for r in manifest_rows:
        vol_dir = output_root / "volumes_npy" / r["safe_id"]
        pm_rows.append({
            "patient_id": r["patient_id"],
            "safe_id": r["safe_id"],
            "split": r["split"],
            "status": r["status"],
            "patch_count": r["patch_count"],
            "volume_dir": vol_dir.as_posix(),
            "ct_hu_npy": (vol_dir / "ct_hu.npy").as_posix(),
            "roi_0_0_npy": (vol_dir / "roi_0_0.npy").as_posix(),
            "meta_json": (vol_dir / "meta.json").as_posix(),
            "patch_csv": (output_root / "patch_index_by_patient" / f"{r['safe_id']}.csv").as_posix(),
            "pure_lung_npy": "",
        })
    atomic_csv_save(pm_rows, manifest_dir / "patient_manifest.csv", pm_cols)

    # train_val_test_split.csv
    tvt_cols = ["patient_id", "split", "safe_id", "patch_count"]
    tvt_rows = [
        {"patient_id": r["patient_id"], "split": r["split"],
         "safe_id": r["safe_id"], "patch_count": r["patch_count"]}
        for r in manifest_rows
    ]
    atomic_csv_save(tvt_rows, manifest_dir / "train_val_test_split.csv", tvt_cols)

    # patch_count_by_patient.csv
    pc_cols = ["patient_id", "safe_id", "split", "patch_count"]
    pc_rows = [
        {"patient_id": r["patient_id"], "safe_id": r["safe_id"],
         "split": r["split"], "patch_count": r["patch_count"]}
        for r in manifest_rows
    ]
    atomic_csv_save(pc_rows, manifest_dir / "patch_count_by_patient.csv", pc_cols)

    excluded = len(summary_rows) - len(manifest_rows)
    if excluded > 0:
        print(f"[WARN] manifest에서 제외된 환자: {excluded}명 (error 또는 patch_count=0)")

    print(f"[INFO] manifests 저장 완료: {manifest_dir}")


# ============================================================
# main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Position-Aware PaDiM v2/v2 정상 training-ready 데이터셋 생성"
    )
    parser.add_argument(
        "--final-all-root",
        default="/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_final_all_nonfast_v2_tslungguard_nochest",
        help="final_all 루트 경로 (/mnt/c/... 형식 가능)",
    )
    parser.add_argument(
        "--v1-reference-root",
        default="/mnt/c/Users/jinhy/Desktop/v1 paicient",
        help="v1 reference 루트 경로 (manifests/train_val_test_split.csv 포함)",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1",
        help="출력 루트 경로",
    )
    parser.add_argument("--limit", type=int, default=None, help="처리할 최대 환자 수")
    parser.add_argument("--dry-run", action="store_true", help="실제 저장 없이 경로 확인 및 용량 추정")
    parser.add_argument("--min-free-gb", type=float, default=30.0, help="최소 필요 여유 공간 (GB)")
    parser.add_argument("--resume", action="store_true", help=".done 있으면 skip")
    parser.add_argument(
        "--sample-patients", nargs="+", default=None,
        help="처리할 특정 patient_id 목록 (공백 구분)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    final_all_root = Path(args.final_all_root)
    v1_reference_root = Path(args.v1_reference_root)
    output_root = Path(args.output_root)

    # ── 기존 output 폴더 확인 ──────────────────────────────────
    if osp(output_root).exists() and not args.resume and not args.dry_run:
        print(f"[WARN] 출력 폴더가 이미 존재합니다: {output_root}")
        print("  덮어쓰기를 방지하기 위해 --resume 플래그를 사용하거나 폴더를 확인하세요.")
        print("  계속하려면 --resume 을 추가하세요. 중단합니다.")
        sys.exit(1)

    # ── v1 split 로드 ──────────────────────────────────────────
    split_map = load_v1_split(v1_reference_root)

    # ── final_all 환자 목록 ────────────────────────────────────
    ct_dir = final_all_root / "ct_1mm_lung_range" / "Normal"
    if not osp(ct_dir).exists():
        print(f"[ERROR] CT 디렉토리 없음: {ct_dir}")
        sys.exit(1)

    all_ct_files = sorted(osp(ct_dir).glob("*_1mm_lung_range_lps.nii.gz"))
    all_patients = []
    for f in all_ct_files:
        pid = f.name.replace("_1mm_lung_range_lps.nii.gz", "")
        all_patients.append(pid)

    print(f"[INFO] final_all 환자 수: {len(all_patients)}")

    # sample-patients 필터
    if args.sample_patients:
        requested = set(args.sample_patients)
        all_patients = [p for p in all_patients if p in requested]
        print(f"[INFO] --sample-patients 적용 후: {len(all_patients)}명")

    # split_map 매칭 확인
    unmatched = [p for p in all_patients if p not in split_map]
    if unmatched:
        print(
            f"[WARN] v1 split CSV 에 없는 환자 {len(unmatched)}명 (skip됨): "
            f"{unmatched[:5]}{'...' if len(unmatched) > 5 else ''}"
        )
    all_patients = [p for p in all_patients if p in split_map]

    # limit 적용
    total_available = len(all_patients)
    if args.limit is not None:
        all_patients = all_patients[: args.limit]

    print(f"[INFO] 처리 대상: {len(all_patients)}명 (전체 매칭: {total_available}명)")

    # ── 저장공간 확인 (dry-run 포함) ───────────────────────────
    free_gb = check_free_space(output_root, args.min_free_gb, label="시작")
    print(f"[INFO] free space: {free_gb:.1f} GB  (min: {args.min_free_gb:.1f} GB)")

    # ── dry-run 모드 ────────────────────────────────────────────
    if args.dry_run:
        print()
        print("=== DRY-RUN MODE ===")
        print(f"free space: {free_gb:.1f} GB")
        print(f"min_free_gb: {args.min_free_gb:.1f} GB")
        print(f"available: {free_gb - args.min_free_gb:.1f} GB")
        print("---")

        total_est_mb = 0.0
        for idx, patient_id in enumerate(all_patients):
            info = split_map[patient_id]
            safe_id = info["safe_id"]
            split = info["split"]

            ct_path = get_ct_path(final_all_root, patient_id)
            lobe_paths = get_lobe_mask_paths(final_all_root, patient_id)

            ct_ok = osp(ct_path).exists()
            lobe_found = sum(1 for p in lobe_paths.values() if osp(p).exists())
            lobe_total = len(lobe_paths)

            est_mb = estimate_patient_size_mb(ct_path) if ct_ok else 0.0
            total_est_mb += est_mb

            ct_status = "OK" if ct_ok else "MISSING"
            lobe_status = "OK" if lobe_found == lobe_total else f"PARTIAL ({lobe_found}/{lobe_total})"

            print(f"[{idx+1}/{len(all_patients)}] {patient_id} ({split}) -> {safe_id}")
            print(f"  CT: {ct_path} [{ct_status}]")
            print(f"  lobe masks: {lobe_found}/{lobe_total} [{lobe_status}]")
            print(f"  estimated size: ~{est_mb:.0f} MB")

            # dry-run 에서도 shape/roi 로드 테스트
            if ct_ok:
                try:
                    result = process_patient(
                        patient_id=patient_id,
                        safe_id=safe_id,
                        split=split,
                        final_all_root=final_all_root,
                        output_root=output_root,
                        error_csv=Path("NUL" if _IS_WINDOWS else "/dev/null"),
                        runtime_csv=Path("NUL" if _IS_WINDOWS else "/dev/null"),
                        resume=False,
                        dry_run=True,
                    )
                    if result["status"] == "dry_run_ok":
                        print(f"  shape_zyx: {result['ct_shape_zyx']}, roi_voxels: {result['roi_0_0_voxels']}")
                    else:
                        print(f"  [WARN] dry-run 처리 실패: {result['status']}")
                except Exception as e:
                    print(f"  [ERROR] dry-run 예외: {e}")

        print()
        print("=== SUMMARY ===")
        print(f"total patients (matched): {total_available}")
        if args.limit is not None:
            print(f"to process: {len(all_patients)} (--limit applied)")
        else:
            print(f"to process: {len(all_patients)}")
        print(f"estimated total size: ~{total_est_mb:.0f} MB (~{total_est_mb/1024:.1f} GB)")
        sufficient = (free_gb - args.min_free_gb) * 1024 >= total_est_mb
        print(f"free space sufficient: {'YES' if sufficient else 'NO'}")
        return

    # ── 실제 실행 ───────────────────────────────────────────────
    logs_dir = output_root / "logs"
    osp(logs_dir).mkdir(parents=True, exist_ok=True)
    error_csv = logs_dir / "error.csv"
    runtime_csv = logs_dir / "runtime_summary.csv"

    summary_rows = []
    done_count = 0
    skip_count = 0
    error_count = 0

    for idx, patient_id in enumerate(all_patients):
        info = split_map[patient_id]
        safe_id = info["safe_id"]
        split_val = info["split"]

        print(f"[{idx+1}/{len(all_patients)}] {patient_id} ({split_val}) -> {safe_id}")

        # 환자 처리 전 저장공간 재확인
        free_after = get_free_gb(output_root)
        if free_after < args.min_free_gb:
            print(f"\n[ABORT] 저장공간 부족: {free_after:.1f} GB < {args.min_free_gb:.1f} GB. 안전 중단.")
            sys.exit(1)

        result = process_patient(
            patient_id=patient_id,
            safe_id=safe_id,
            split=split_val,
            final_all_root=final_all_root,
            output_root=output_root,
            error_csv=error_csv,
            runtime_csv=runtime_csv,
            resume=args.resume,
            dry_run=False,
        )

        status = result["status"]
        if status == "done":
            done_count += 1
            print(f"  -> done (patches: {result['patch_count']}, roi_voxels: {result['roi_0_0_voxels']})")
        elif status == "skipped":
            skip_count += 1
            print(f"  -> skipped (.done exists)")
        else:
            error_count += 1
            print(f"  -> ERROR")

        summary_rows.append({
            "patient_id": patient_id,
            "safe_id": safe_id,
            "split": split_val,
            "status": status,
            "patch_count": result["patch_count"],
        })

    # manifests 생성
    if summary_rows:
        write_manifests(output_root, summary_rows)

    print()
    print("=== DONE ===")
    print(f"처리 완료: {done_count}명  |  skip: {skip_count}명  |  error: {error_count}명")
    if error_count > 0:
        print(f"[WARN] error 환자 {error_count}명 — logs/error.csv 확인 필요. manifest에서 제외됨.")
    print(f"출력 경로: {output_root}")


if __name__ == "__main__":
    main()
