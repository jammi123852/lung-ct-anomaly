"""
export_raw_slices.py: 전처리 안 건드린 원본 ct_hu.npy를 PNG 슬라이스 시퀀스로 저장.

- patient_manifest.csv에서 patient_id → safe_id 매핑.
- volumes_npy/{safe_id}/ct_hu.npy 로드 (HU 값 int16, shape=(z, 512, 512)).
- HU window(WL/WW) 적용해서 0~255 uint8로 변환.
- 각 z 슬라이스를 PNG로 저장: outputs/raw_slice_view/{patient_id}/z{NNN}.png

금지:
- cv2 / matplotlib 사용 금지 (PIL만 사용)
- 학습 / 스코어링 / 병변 데이터 접근 없음
- pip install 없음
- 기존 결과 파일 수정 / 삭제 없음
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "reports"
ERROR_CSV = REPORTS_DIR / "error.csv"
RUNTIME_CSV = REPORTS_DIR / "runtime_summary.csv"

ERROR_COLUMNS = ["patient_id", "error_type", "error_msg", "file_logical"]
RUNTIME_COLUMNS = ["timestamp", "script", "metric", "value"]

SCRIPT_NAME = "export_raw_slices.py"


def record_error(patient_id, error_type, error_msg, file_logical):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not ERROR_CSV.exists() or ERROR_CSV.stat().st_size == 0
    with open(ERROR_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "patient_id": patient_id, "error_type": error_type,
            "error_msg": error_msg, "file_logical": file_logical,
        })


def record_runtime_rows(rows):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_CSV.exists() or RUNTIME_CSV.stat().st_size == 0
    with open(RUNTIME_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def apply_hu_window(volume: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """HU volume → uint8 [0, 255]. wl=window level, ww=window width."""
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    v = volume.astype(np.float32)
    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo) * 255.0
    return v.astype(np.uint8)


def apply_pure_lung_overlay(
    slice_u8: np.ndarray,
    mask: np.ndarray,
    color: tuple,
    alpha: float,
) -> np.ndarray:
    """grayscale 슬라이스에 pure_lung 마스크 영역을 색상으로 alpha blend.

    Parameters
    ----------
    slice_u8 : np.ndarray
        (H, W) uint8 grayscale (HU windowed)
    mask : np.ndarray
        (H, W) — mask>0 픽셀만 blend
    color : tuple of int
        RGB 0-255
    alpha : float
        0.0~1.0 (마스크 영역의 색상 강도)

    Returns
    -------
    np.ndarray
        (H, W, 3) uint8 RGB
    """
    h, w = slice_u8.shape
    rgb = np.stack([slice_u8, slice_u8, slice_u8], axis=-1).astype(np.float32)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    m = (mask > 0)
    if m.any():
        rgb[m] = rgb[m] * (1.0 - alpha) + color_arr * alpha
    return np.clip(rgb, 0, 255).astype(np.uint8)


# plane → (slice 축 index, 파일명 prefix)
# ct_hu shape = (z, y, x): axial=z축(0), coronal=y축(1), sagittal=x축(2)
_PLANE_AXIS = {"axial": 0, "coronal": 1, "sagittal": 2}
_PLANE_PREFIX = {"axial": "z", "coronal": "y", "sagittal": "x"}


def get_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """mmap volume에서 지정 축의 idx번째 2D slice를 in-memory copy로 반환."""
    if axis == 0:
        return np.array(vol[idx])
    elif axis == 1:
        return np.array(vol[:, idx, :])
    else:
        return np.array(vol[:, :, idx])


def parse_color(s: str) -> tuple:
    """'R,G,B' 문자열 → (R, G, B) int tuple."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--overlay-color는 'R,G,B' 형식이어야 합니다: {s!r}")
    out = []
    for p in parts:
        v = int(p)
        if v < 0 or v > 255:
            raise ValueError(f"색상 값은 0~255 범위: {v}")
        out.append(v)
    return tuple(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="원본 ct_hu.npy를 PNG 슬라이스 시퀀스로 export")
    parser.add_argument("--patient-id", required=True, dest="patient_id",
                        help="처리할 환자 ID (예: normal001)")
    parser.add_argument("--plane", default="axial", dest="plane",
                        choices=["axial", "sagittal", "coronal"],
                        help="단면 방향 (기본: axial). "
                             "ct_hu shape=(z,y,x) 기준 axial=z축, coronal=y축, sagittal=x축")
    parser.add_argument("--hu-window-center", type=float, default=-600.0,
                        dest="hu_window_center",
                        help="HU window level (기본: -600, lung window)")
    parser.add_argument("--hu-window-width", type=float, default=1500.0,
                        dest="hu_window_width",
                        help="HU window width (기본: 1500, lung window)")
    parser.add_argument("--z-step", type=int, default=1, dest="z_step",
                        help="z 슬라이스 간격 (기본: 1, 전부 저장)")
    parser.add_argument("--z-start", type=int, default=None, dest="z_start",
                        help="시작 z (기본: 0)")
    parser.add_argument("--z-end", type=int, default=None, dest="z_end",
                        help="종료 z (exclusive, 기본: 전체)")
    parser.add_argument("--output-subdir-suffix", default="",
                        dest="output_subdir_suffix",
                        help="출력 폴더 접미사 (예: _mediastinum). 동일 환자 다른 window 구분용")
    parser.add_argument("--overlay-pure-lung", action="store_true",
                        default=False, dest="overlay_pure_lung",
                        help="pure_lung.npy 마스크를 색상 overlay로 덮어 RGB PNG로 저장")
    parser.add_argument("--overlay-color", default="0,255,0",
                        dest="overlay_color",
                        help="overlay RGB 색상 'R,G,B' (기본: 0,255,0 green)")
    parser.add_argument("--overlay-alpha", type=float, default=0.35,
                        dest="overlay_alpha",
                        help="overlay alpha 0.0~1.0 (기본: 0.35)")
    args = parser.parse_args()

    pid = args.patient_id
    wl = args.hu_window_center
    ww = args.hu_window_width
    z_step = args.z_step
    suffix = args.output_subdir_suffix
    plane = args.plane
    plane_axis = _PLANE_AXIS[plane]
    plane_prefix = _PLANE_PREFIX[plane]
    overlay_pure_lung = args.overlay_pure_lung
    overlay_alpha = args.overlay_alpha
    try:
        overlay_color = parse_color(args.overlay_color)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    if overlay_pure_lung:
        if overlay_alpha < 0.0 or overlay_alpha > 1.0:
            print(f"[ERROR] --overlay-alpha는 0.0~1.0 범위: {overlay_alpha}")
            sys.exit(1)
    # 사용자가 suffix를 명시하지 않았으면 plane/overlay를 반영해 자동 부여
    # (axial + plain → suffix 없음, 기존 동작 유지)
    if suffix == "":
        auto_parts = []
        if plane != "axial":
            auto_parts.append(plane)
        if overlay_pure_lung:
            auto_parts.append("pure_lung_overlay")
        if auto_parts:
            suffix = "_" + "_".join(auto_parts)

    if z_step <= 0:
        print(f"[ERROR] --z-step은 양의 정수: {z_step}")
        sys.exit(1)
    if ww <= 0:
        print(f"[ERROR] --hu-window-width는 양수: {ww}")
        sys.exit(1)
    if "/" in suffix or "\\" in suffix or suffix.startswith("."):
        print(f"[ERROR] --output-subdir-suffix에 경로 구분자/상대경로 금지: {suffix!r}")
        sys.exit(1)

    # configs 로드
    cfg_path = REPO_ROOT / "configs" / "paths.local.yaml"
    with open(cfg_path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    base = cfg.get("normal_training_ready", "")
    if not base:
        print(f"[ERROR] configs/paths.local.yaml에 normal_training_ready 경로 없음")
        sys.exit(1)
    base_path = Path(base)
    manifest_path = base_path / "manifests" / "patient_manifest.csv"
    if not manifest_path.exists():
        print(f"[ERROR] manifest 없음: {manifest_path}")
        sys.exit(1)

    # safe_id 찾기
    safe_id = None
    with open(manifest_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("patient_id") == pid:
                safe_id = row.get("safe_id") or row.get("patient_safe_id")
                break
    if safe_id is None:
        print(f"[ERROR] patient_id={pid}를 manifest에서 찾지 못함")
        sys.exit(1)

    ct_path = base_path / "volumes_npy" / safe_id / "ct_hu.npy"
    if not ct_path.exists():
        msg = f"ct_hu.npy 없음: {ct_path}"
        record_error(pid, "ct_npy_not_found", msg, str(ct_path))
        print(f"[ERROR] {msg}")
        sys.exit(1)

    # overlay 모드: pure_lung.npy 확인
    mask_path = None
    if overlay_pure_lung:
        mask_path = base_path / "volumes_npy" / safe_id / "pure_lung.npy"
        if not mask_path.exists():
            msg = f"pure_lung.npy 없음: {mask_path}"
            record_error(pid, "pure_lung_npy_not_found", msg, str(mask_path))
            print(f"[ERROR] {msg}")
            sys.exit(1)

    # 출력 폴더
    out_dir_name = pid if not suffix else f"{pid}{suffix}"
    out_dir = REPO_ROOT / "outputs" / "raw_slice_view" / out_dir_name

    # 안전 가드: 기존 출력 폴더 존재 시 중단
    if out_dir.exists() and any(out_dir.iterdir()):
        print(
            f"[ERROR] 기존 출력 폴더가 이미 존재합니다: {out_dir}\n"
            "기존 폴더를 archive로 이동하거나 다른 --output-subdir-suffix를 지정하세요."
        )
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # CT 로드 (mmap)
    ct = np.load(str(ct_path), mmap_mode="r")
    z_total = ct.shape[plane_axis]

    # overlay 모드: pure_lung 로드 (mmap)
    mask_vol = None
    if overlay_pure_lung:
        mask_vol = np.load(str(mask_path), mmap_mode="r")
        if mask_vol.shape != ct.shape:
            msg = f"shape 불일치: ct={ct.shape}, pure_lung={mask_vol.shape}"
            record_error(pid, "shape_mismatch", msg, str(mask_path))
            print(f"[ERROR] {msg}")
            sys.exit(1)
    z_start = args.z_start if args.z_start is not None else 0
    z_end = args.z_end if args.z_end is not None else z_total
    z_start = max(0, z_start)
    z_end = min(z_total, z_end)
    if z_start >= z_end:
        print(f"[ERROR] z 범위가 비어 있음: [{z_start}, {z_end})")
        sys.exit(1)

    print(f"[export_raw_slices] patient_id={pid}, safe_id={safe_id}")
    print(f"[export_raw_slices] ct_hu.npy shape={ct.shape}, dtype={ct.dtype}")
    print(f"[export_raw_slices] plane={plane} (축 index={plane_axis}, prefix='{plane_prefix}'), 총 slice={z_total}")
    print(f"[export_raw_slices] HU window: WL={wl}, WW={ww}  → lo={wl-ww/2}, hi={wl+ww/2}")
    print(f"[export_raw_slices] slice 범위: [{z_start}, {z_end}), step={z_step}")
    print(f"[export_raw_slices] 출력 폴더: {out_dir}")
    if overlay_pure_lung:
        print(
            f"[export_raw_slices] pure_lung overlay: color={overlay_color}, "
            f"alpha={overlay_alpha}"
        )
    print()

    # z width (zero pad)
    z_width = max(3, len(str(z_total - 1)))

    n_saved = 0
    for z in range(z_start, z_end, z_step):
        slice_hu = get_slice(ct, plane_axis, z)  # mmap → in-memory copy
        slice_u8 = apply_hu_window(slice_hu, wl=wl, ww=ww)
        out_path = out_dir / f"{plane_prefix}{z:0{z_width}d}.png"
        if overlay_pure_lung:
            mask_slice = get_slice(mask_vol, plane_axis, z)
            rgb = apply_pure_lung_overlay(
                slice_u8, mask_slice, color=overlay_color, alpha=overlay_alpha,
            )
            Image.fromarray(rgb, mode="RGB").save(str(out_path))
        else:
            Image.fromarray(slice_u8, mode="L").save(str(out_path))
        n_saved += 1
        if n_saved % 50 == 0 or n_saved == 1:
            print(f"  z={z:>4} → {out_path.name}")

    elapsed = time.time() - start_time
    print()
    print(f"[export_raw_slices] 완료: {n_saved}장 저장 ({elapsed:.1f}s)")
    print(f"[export_raw_slices] 출력 폴더: {out_dir}")

    # runtime_summary 기록
    ts = datetime.now().isoformat(timespec="seconds")
    record_runtime_rows([
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "patient_id", "value": pid},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "safe_id", "value": safe_id},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "ct_shape", "value": str(ct.shape)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "plane", "value": plane},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "hu_window_center", "value": wl},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "hu_window_width", "value": ww},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "z_step", "value": z_step},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_slices_saved", "value": n_saved},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "elapsed_seconds", "value": round(elapsed, 2)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "output_dir", "value": str(out_dir)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "overlay_pure_lung", "value": str(overlay_pure_lung)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "overlay_color", "value": str(overlay_color) if overlay_pure_lung else ""},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "overlay_alpha", "value": overlay_alpha if overlay_pure_lung else ""},
    ])
    print(f"[export_raw_slices] runtime_summary.csv 기록 완료")


if __name__ == "__main__":
    main()
