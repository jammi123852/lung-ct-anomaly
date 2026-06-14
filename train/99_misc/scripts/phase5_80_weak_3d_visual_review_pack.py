#!/usr/bin/env python3
"""
Phase 5.80 / 5.81 Weak 3D Cluster Visual Review Pack
==============================================================
목적: Phase 5.79 preflight에서 검증된 14개 3D cluster target에 대해
     CT slice panel + bbox overlay + 메타 텍스트 패널로 구성된
     visual review pack을 생성한다.

경고: --run은 구현 완료됐으나 사용자 별도 승인 전 실행 금지.
     (--dry-run 또는 --run 정확히 하나만 지정 필요)

참고 패턴: phase5_77_weak_3d_merge_dry_run.py 의
  _validate_output_tag(), _guard_path(), output guard 패턴을 그대로 따름.
  (merge 로직 없음)

예상 volume 크기: 약 120~130 MB / 환자
메모리 전략: one-patient-at-a-time, mmap_mode="r", 동시 다중 volume cache 금지

이번 단계(Phase 5.81): 렌더링/빌드 함수 전부 구현, RUN_MODE_IMPLEMENTED=True 전환.
스크립트 실행, output root 생성, PNG/HTML/ZIP/CSV/JSON/MD 생성은
사용자 별도 승인 전 절대 금지.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 프로젝트 루트 ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 입력 파일 경로 ─────────────────────────────────────────────────────────────
_MANIFEST_PATH = _PROJECT_ROOT / (
    "outputs/second-stage-lesion-refiner-v1/reports"
    "/phase5_79_weak_3d_visual_pack_preflight_v1"
    "/phase5_79_weak_3d_visual_pack_target_manifest_v1.csv"
)
_PREFLIGHT_JSON_PATH = _PROJECT_ROOT / (
    "outputs/second-stage-lesion-refiner-v1/reports"
    "/phase5_79_weak_3d_visual_pack_preflight_v1"
    "/phase5_79_weak_3d_visual_pack_preflight_v1.json"
)

# ── output base ───────────────────────────────────────────────────────────────
_OUTPUT_BASE = _PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/visualizations"

# ── constants ─────────────────────────────────────────────────────────────────
EXPECTED_MANIFEST_ROW_COUNT = 14
EXPECTED_PATIENTS = {"normal004", "normal013", "normal014"}

# --run 구현 완료 여부 플래그 (Phase 5.81에서 True로 전환됨)
# 경고: --run은 구현 완료됐으나 사용자 별도 승인 전 실행 금지.
RUN_MODE_IMPLEMENTED = True  # Phase 5.81: 구현 완료 전환 (줄 59)

# meta.json shape key 탐색 후보 리스트 (shape_zyx 최우선)
SHAPE_KEY_CANDIDATES: List[str] = [
    "shape_zyx",
    "shape",
    "ct_shape",
    "volume_shape",
    "image_shape",
    "array_shape",
]

# manifest 필수 컬럼
REQUIRED_MANIFEST_COLUMNS: List[str] = [
    "review_order",
    "source_group",
    "cluster3d_id",
    "patient_id",
    "z_min",
    "z_max",
    "z_span",
    "representative_local_z",
    "y0_min",
    "x0_min",
    "y1_max",
    "x1_max",
    "representative_y0",
    "representative_x0",
    "representative_y1",
    "representative_x1",
    "n_2d_clusters",
    "n_patches_total",
    "bbox_area",
    "top3_mean_patch_score_3d",
    "review_candidate_flag",
    "overmerge_flag",
    "large_bbox_overmerge_flag",
    "large_extent_overmerge_flag",
    "complex_merge_flag",
    "high_score_ratio_flag",
    "diagnostic_priority",
    "diagnostic_note",
    "visual_source_status",
    "visual_source_candidate_path",
    "user_label",
    "user_note",
]

# visual review label guide
LABEL_GUIDE: List[str] = [
    "pleural_wall",
    "large_bbox_structure",
    "vessel_branch",
    "bronchus_air_boundary",
    "outside_roi_artifact",
    "z_overmerge_ok_continuous_structure",
    "z_overmerge_suspicious_overmerge",
    "unclear",
]


# ── output tag validation (phase5_77과 동일) ──────────────────────────────────
def _validate_output_tag(tag: str) -> None:
    if not tag:
        sys.exit("[ERROR] --output-tag must not be empty.")
    if Path(tag).is_absolute():
        sys.exit(f"[ERROR] --output-tag must not be an absolute path: {tag!r}")
    if "/" in tag or "\\" in tag:
        sys.exit(f"[ERROR] --output-tag must not contain '/' or '\\': {tag!r}")
    if ".." in tag:
        sys.exit(f"[ERROR] --output-tag must not contain '..': {tag!r}")
    parts = Path(tag).parts
    if len(parts) != 1:
        sys.exit(f"[ERROR] --output-tag must be a single path component, got {parts!r}")
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", tag):
        sys.exit(
            f"[ERROR] --output-tag contains invalid characters. "
            f"Only alphanumeric, underscore, hyphen allowed: {tag!r}"
        )


# ── path guard (phase5_77과 동일, case-insensitive) ───────────────────────────
def _guard_path(p: Path) -> None:
    for part in p.parts:
        pl = part.lower()
        if (
            "stage2_holdout" in pl
            or pl == "v2"
            or pl.startswith("v2v2")
            or pl.startswith("v2_")
            or pl.startswith("lesion_by_patient")
            or pl.startswith("crops_lesion")
            or "hard_negative" in pl
            or "nsclc_msd" in pl
            or "msd_lung" in pl
        ):
            sys.exit(f"[ERROR] Forbidden path segment '{part}' in {p}")


# ── meta.json shape 로드 (read-only, npy 로드 금지) ──────────────────────────
def load_meta_shape(meta_path: Path) -> Optional[Tuple[int, int, int]]:
    """
    meta.json을 읽어 (z_dim, y_dim, x_dim) 튜플을 반환한다.
    shape key 후보를 SHAPE_KEY_CANDIDATES 순서로 탐색한다. (shape_zyx 최우선)

    반환 규칙:
    - 성공: (z_dim, y_dim, x_dim) tuple
    - meta.json 없음: None  (호출부에서 처리)
    - shape key 없음: None  (호출부에서 "shape_unknown_in_meta" 처리)

    중요: ct_hu.npy, roi_0_0.npy는 절대 np.load 금지.
          os.path.exists() / Path.exists() / Path.stat()으로만 존재 확인.
    """
    if not meta_path.exists():
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARNING] meta.json load failed: {meta_path} — {e}")
        return None

    for key in SHAPE_KEY_CANDIDATES:
        if key in meta:
            val = meta[key]
            # shape_zyx: [z, y, x] 순서
            if isinstance(val, (list, tuple)) and len(val) == 3:
                try:
                    z_dim = int(val[0])
                    y_dim = int(val[1])
                    x_dim = int(val[2])
                    return (z_dim, y_dim, x_dim)
                except (ValueError, TypeError):
                    continue
            # dict 형태 예외 처리 (드물지만 방어)
            elif isinstance(val, dict):
                for sub_key in ("z", "depth", "d"):
                    if sub_key in val:
                        break
                # dict 형태는 구조 불명확하므로 None 반환
                return None

    # 모든 후보 key 없음
    return None


# ── 좌표계 검증 (환자별 z_dim 개별 적용, npy 로드 금지) ─────────────────────
def validate_target_against_meta(
    row: Dict,
    shape: Optional[Tuple[int, int, int]],
) -> Tuple[bool, List[str]]:
    """
    manifest 한 행의 좌표를 meta.json에서 로드한 volume shape과 비교한다.

    환자별 z_dim이 다르므로 (normal004=243, normal013=227, normal014=250)
    호출부에서 환자별 meta를 개별 로드하여 이 함수에 전달해야 한다.

    검증 항목:
    1. 0 <= z_min <= representative_local_z <= z_max < z_dim
    2. 0 <= y0_min < y1_max <= y_dim
    3. 0 <= x0_min < x1_max <= x_dim
    4. representative bbox 내부 포함 검증:
       y0_min <= representative_y0, x0_min <= representative_x0,
       representative_y1 <= y1_max, representative_x1 <= x1_max

    반환: (is_valid: bool, error_messages: List[str])
    - shape가 None이면 검증 보류 + warning 플래그 반환
    """
    errors: List[str] = []

    if shape is None:
        # shape_unknown_in_meta: 검증 보류
        return (False, ["[WARNING] shape_unknown_in_meta: coordinate validation skipped"])

    z_dim, y_dim, x_dim = shape

    z_min = int(row["z_min"])
    z_max = int(row["z_max"])
    rep_z = int(row["representative_local_z"])
    y0 = int(row["y0_min"])
    x0 = int(row["x0_min"])
    y1 = int(row["y1_max"])
    x1 = int(row["x1_max"])
    rep_y0 = int(row["representative_y0"])
    rep_x0 = int(row["representative_x0"])
    rep_y1 = int(row["representative_y1"])
    rep_x1 = int(row["representative_x1"])

    # 1. z 범위 검증
    if not (0 <= z_min <= rep_z <= z_max < z_dim):
        errors.append(
            f"z range invalid: 0 <= {z_min} <= {rep_z} <= {z_max} < {z_dim} = False"
        )

    # 2. y 범위 검증
    if not (0 <= y0 < y1 <= y_dim):
        errors.append(
            f"y range invalid: 0 <= {y0} < {y1} <= {y_dim} = False"
        )

    # 3. x 범위 검증
    if not (0 <= x0 < x1 <= x_dim):
        errors.append(
            f"x range invalid: 0 <= {x0} < {x1} <= {x_dim} = False"
        )

    # 4. representative bbox 내부 포함 검증
    if not (y0 <= rep_y0):
        errors.append(f"representative bbox y0 out of cluster bbox: {y0} > {rep_y0}")
    if not (x0 <= rep_x0):
        errors.append(f"representative bbox x0 out of cluster bbox: {x0} > {rep_x0}")
    if not (rep_y1 <= y1):
        errors.append(f"representative bbox y1 out of cluster bbox: {rep_y1} > {y1}")
    if not (rep_x1 <= x1):
        errors.append(f"representative bbox x1 out of cluster bbox: {rep_x1} > {x1}")

    is_valid = len(errors) == 0
    return (is_valid, errors)


# ── manifest 검증 ─────────────────────────────────────────────────────────────
def _validate_manifest(df, max_targets: int) -> None:
    """
    입력 manifest의 구조 및 내용을 검증한다.
    검증 실패 시 즉시 sys.exit.
    """
    # 행 수 검증 (manifest 자체는 14행 기대)
    if len(df) != EXPECTED_MANIFEST_ROW_COUNT:
        sys.exit(
            f"[ERROR] Manifest expected {EXPECTED_MANIFEST_ROW_COUNT} rows, got {len(df)}"
        )

    # max_targets 범위 확인 (1 ~ EXPECTED_MANIFEST_ROW_COUNT)
    if not (1 <= max_targets <= EXPECTED_MANIFEST_ROW_COUNT):
        sys.exit(
            f"[ERROR] --max-targets must be 1~{EXPECTED_MANIFEST_ROW_COUNT}, "
            f"got {max_targets}"
        )

    # 필수 컬럼 존재 확인
    missing_cols = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in df.columns]
    if missing_cols:
        sys.exit(f"[ERROR] Manifest missing required columns: {missing_cols}")

    # cluster3d_id 중복 확인
    if df["cluster3d_id"].duplicated().any():
        dup = df[df["cluster3d_id"].duplicated(keep=False)]["cluster3d_id"].tolist()
        sys.exit(f"[ERROR] Duplicate cluster3d_id found: {dup}")

    # visual_source_status 전부 "found" 확인
    not_found = df[df["visual_source_status"] != "found"]
    if len(not_found) > 0:
        bad_ids = not_found["cluster3d_id"].tolist()
        sys.exit(
            f"[ERROR] visual_source_status is not 'found' for: {bad_ids}"
        )

    # patient_id 집합 확인
    actual_patients = set(df["patient_id"].unique())
    if not actual_patients.issubset(EXPECTED_PATIENTS):
        unexpected = actual_patients - EXPECTED_PATIENTS
        sys.exit(
            f"[ERROR] Unexpected patient_id values: {sorted(unexpected)}. "
            f"Expected subset of {sorted(EXPECTED_PATIENTS)}"
        )
    if not actual_patients == EXPECTED_PATIENTS:
        missing_patients = EXPECTED_PATIENTS - actual_patients
        sys.exit(
            f"[ERROR] Missing patients in manifest: {sorted(missing_patients)}"
        )

    print(
        f"[INFO] Manifest validated: {len(df)} rows, "
        f"patients={sorted(actual_patients)}"
    )


# ── visual source path 존재 확인 (stat only, npy 로드 금지) ──────────────────
def _validate_visual_source_paths(df) -> None:
    """
    visual_source_candidate_path (ct_hu.npy), roi_0_0.npy, meta.json 파일
    존재 여부를 Path.stat()으로만 확인한다. (npy 로드 절대 금지)
    """
    errors = []
    for _, row in df.iterrows():
        ct_path = Path(str(row["visual_source_candidate_path"]))
        roi_path = ct_path.parent / "roi_0_0.npy"
        meta_path = ct_path.parent / "meta.json"

        for fpath in (ct_path, roi_path, meta_path):
            _guard_path(fpath)
            try:
                fpath.stat()
            except FileNotFoundError:
                errors.append(f"File not found: {fpath} (cluster={row['cluster3d_id']})")
            except OSError as e:
                errors.append(f"File access error: {fpath} — {e}")

    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        sys.exit(f"[ERROR] {len(errors)} visual source file(s) not accessible.")

    print(f"[INFO] All visual source files accessible (stat-only, no npy load).")


# ── 환자별 meta shape 검증 (npy 로드 금지) ───────────────────────────────────
def _validate_all_coordinates(df) -> None:
    """
    각 target의 patient별 meta.json을 개별 로드하여 좌표계를 검증한다.
    환자별 z_dim이 다르므로 반드시 환자별로 meta를 개별 로드한다.
    (normal004=243, normal013=227, normal014=250 — 런타임에서 meta.json으로 확인)
    """
    # 환자별 meta shape 캐시 (환자당 1회만 로드)
    patient_shape_cache: Dict[str, Optional[Tuple[int, int, int]]] = {}
    patient_meta_path_cache: Dict[str, Path] = {}

    # 환자별 첫 번째 ct_hu.npy 경로로 meta 경로 결정
    for _, row in df.iterrows():
        pid = str(row["patient_id"])
        if pid not in patient_meta_path_cache:
            ct_path = Path(str(row["visual_source_candidate_path"]))
            meta_path = ct_path.parent / "meta.json"
            patient_meta_path_cache[pid] = meta_path

    # 환자별 meta shape 로드
    for pid, meta_path in patient_meta_path_cache.items():
        shape = load_meta_shape(meta_path)
        patient_shape_cache[pid] = shape
        if shape is None:
            print(
                f"[WARNING] {pid}: meta shape unknown "
                f"(meta_path={meta_path}). Coordinate validation skipped."
            )
        else:
            z_dim, y_dim, x_dim = shape
            print(
                f"[INFO] {pid}: meta shape loaded — "
                f"z_dim={z_dim}, y_dim={y_dim}, x_dim={x_dim}"
            )

    # target별 좌표 검증
    coord_errors = []
    shape_unknown_targets = []

    for _, row in df.iterrows():
        pid = str(row["patient_id"])
        cid = str(row["cluster3d_id"])
        shape = patient_shape_cache.get(pid)

        is_valid, messages = validate_target_against_meta(row, shape)

        if not is_valid:
            for msg in messages:
                if msg.startswith("[WARNING]"):
                    shape_unknown_targets.append(f"{cid}: {msg}")
                else:
                    coord_errors.append(f"{cid}: {msg}")

    if shape_unknown_targets:
        print(f"[WARNING] {len(shape_unknown_targets)} target(s) with unknown shape:")
        for t in shape_unknown_targets:
            print(f"  {t}")

    if coord_errors:
        for e in coord_errors:
            print(f"[ERROR] Coordinate validation failed — {e}")
        sys.exit(f"[ERROR] {len(coord_errors)} coordinate error(s) found.")

    print(
        f"[INFO] Coordinate validation complete: "
        f"{len(df) - len(shape_unknown_targets)} validated, "
        f"{len(shape_unknown_targets)} skipped (shape unknown)."
    )


# ── helper: CT windowing ──────────────────────────────────────────────────────
def _window_ct(slice_2d, vmin: float = -1000.0, vmax: float = 400.0):
    """
    CT 2D slice에 window/level 적용 후 0~1 float 반환.
    - (slice - vmin) / (vmax - vmin) 후 clip [0, 1]
    - NaN/Inf 안전 처리 (np.nan_to_num)
    - 원본 수정 금지 (비파괴 연산)
    - 반환: 0~1 float ndarray (복사본)
    """
    import numpy as np  # 지연 import

    arr = np.array(slice_2d, dtype=np.float32)  # 복사본 생성
    arr = np.nan_to_num(arr, nan=vmin, posinf=vmax, neginf=vmin)
    arr = (arr - vmin) / (vmax - vmin)
    arr = np.clip(arr, 0.0, 1.0)
    return arr


# ── helper: bbox clipping ─────────────────────────────────────────────────────
def _clip_bbox(
    y0: int, x0: int, y1: int, x1: int, h: int, w: int
) -> Tuple[Tuple[int, int, int, int], bool]:
    """
    bbox 좌표를 이미지 경계(h, w) 내로 clip한다.
    반환: (clipped_coords(y0, x0, y1, x1), clip_occurred_bool)
    - 좌표가 경계 내에 있으면 clip_occurred=False
    - 범위 벗어나면 clip 후 clip_occurred=True
    안전장치 용도. clip 발생 시 호출부에서 metadata note에 기록 가능.
    """
    cy0 = max(0, min(y0, h))
    cx0 = max(0, min(x0, w))
    cy1 = max(0, min(y1, h))
    cx1 = max(0, min(x1, w))
    clip_occurred = (cy0 != y0) or (cx0 != x0) or (cy1 != y1) or (cx1 != x1)
    return (cy0, cx0, cy1, cx1), clip_occurred


# ── visual panel 함수 (--run 시 실제 렌더링) ──────────────────────────────────
def _render_center_slice_panel(
    ax,               # matplotlib axes
    ct_vol,           # np.ndarray, mmap_mode="r" 로드, shape=(z, y, x)
    row: Dict,
) -> None:
    """
    center slice (representative_local_z) CT HU panel 렌더링.
    _window_ct 적용 후 grayscale(cmap="gray") 표시.
    matplotlib import는 함수 내부 지연 import로 처리.
    (py_compile/검증 단계에서 matplotlib 부작용 없음)
    """
    import matplotlib  # 지연 import
    matplotlib.use("Agg")  # WSL headless: plt import 전 반드시 Agg 강제
    import matplotlib.pyplot as plt  # noqa: F401
    import numpy as np  # 지연 import

    rep_z = int(row["representative_local_z"])
    pid = str(row["patient_id"])
    cid = str(row["cluster3d_id"])

    # mmap이므로 해당 2D slice만 복사
    slice_2d = np.asarray(ct_vol[rep_z])
    windowed = _window_ct(slice_2d)

    ax.imshow(windowed, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(f"{pid} / {cid}\nz={rep_z}", fontsize=7)
    ax.axis("off")


def _render_bbox_overlay_panel(
    ax,
    ct_vol,           # np.ndarray, mmap_mode="r"
    row: Dict,
    roi_vol=None,     # np.ndarray or None, mmap_mode="r"
) -> None:
    """
    representative bbox overlay panel 렌더링 (2D bbox + 3D cluster bbox).
    matplotlib import는 함수 내부 지연 import.
    """
    import matplotlib  # 지연 import
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches  # noqa: F401
    import numpy as np  # 지연 import

    rep_z = int(row["representative_local_z"])

    slice_2d = np.asarray(ct_vol[rep_z])
    windowed = _window_ct(slice_2d)
    h, w = windowed.shape[:2]

    ax.imshow(windowed, cmap="gray", vmin=0.0, vmax=1.0)

    # 3D cluster bbox (굵은 실선, 흰색)
    y0_cl = int(row["y0_min"])
    x0_cl = int(row["x0_min"])
    y1_cl = int(row["y1_max"])
    x1_cl = int(row["x1_max"])
    (y0_cl, x0_cl, y1_cl, x1_cl), cl_clipped = _clip_bbox(y0_cl, x0_cl, y1_cl, x1_cl, h, w)
    rect_cl = mpatches.Rectangle(
        (x0_cl, y0_cl),
        x1_cl - x0_cl,
        y1_cl - y0_cl,
        linewidth=2,
        edgecolor="white",
        facecolor="none",
        linestyle="-",
    )
    ax.add_patch(rect_cl)

    # representative bbox (점선, 노란색)
    y0_rep = int(row["representative_y0"])
    x0_rep = int(row["representative_x0"])
    y1_rep = int(row["representative_y1"])
    x1_rep = int(row["representative_x1"])
    (y0_rep, x0_rep, y1_rep, x1_rep), rep_clipped = _clip_bbox(
        y0_rep, x0_rep, y1_rep, x1_rep, h, w
    )
    rect_rep = mpatches.Rectangle(
        (x0_rep, y0_rep),
        x1_rep - x0_rep,
        y1_rep - y0_rep,
        linewidth=1,
        edgecolor="yellow",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect_rep)

    # ROI contour (alpha mask)
    if roi_vol is not None:
        roi_slice = np.asarray(roi_vol[rep_z]).astype(np.float32)
        ax.imshow(roi_slice, cmap="Reds", alpha=0.2, vmin=0.0, vmax=1.0)

    clip_note = ""
    if cl_clipped or rep_clipped:
        clip_note = " [bbox clipped]"
    ax.set_title(f"bbox overlay{clip_note}", fontsize=7)
    ax.axis("off")


def _render_z_context_panel(
    ct_vol,           # np.ndarray, mmap_mode="r"
    row: Dict,
    axes_list,        # [ax_z_min, ax_rep_z, ax_z_max]
) -> None:
    """
    z_min / representative_z / z_max context panel 렌더링.
    3개 slice를 나란히 표시. 각 slice에 3D cluster bbox overlay,
    representative slice에는 representative bbox도 overlay.
    z_min==rep 또는 z_max==rep여도 에러 없이 동작.
    matplotlib import는 함수 내부 지연 import.
    """
    import matplotlib  # 지연 import
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches  # noqa: F401
    import numpy as np  # 지연 import

    z_min = int(row["z_min"])
    rep_z = int(row["representative_local_z"])
    z_max = int(row["z_max"])

    y0_cl = int(row["y0_min"])
    x0_cl = int(row["x0_min"])
    y1_cl = int(row["y1_max"])
    x1_cl = int(row["x1_max"])
    y0_rep = int(row["representative_y0"])
    x0_rep = int(row["representative_x0"])
    y1_rep = int(row["representative_y1"])
    x1_rep = int(row["representative_x1"])

    z_labels = [("z_min", z_min), ("rep_z", rep_z), ("z_max", z_max)]

    for ax, (label, z_idx) in zip(axes_list, z_labels):
        slice_2d = np.asarray(ct_vol[z_idx])
        windowed = _window_ct(slice_2d)
        h, w = windowed.shape[:2]

        ax.imshow(windowed, cmap="gray", vmin=0.0, vmax=1.0)

        # 3D cluster bbox
        (cy0, cx0, cy1, cx1), _ = _clip_bbox(y0_cl, x0_cl, y1_cl, x1_cl, h, w)
        rect_cl = mpatches.Rectangle(
            (cx0, cy0), cx1 - cx0, cy1 - cy0,
            linewidth=1.5, edgecolor="white", facecolor="none", linestyle="-",
        )
        ax.add_patch(rect_cl)

        # representative bbox (representative slice에만 추가)
        if label == "rep_z":
            (ry0, rx0, ry1, rx1), _ = _clip_bbox(
                y0_rep, x0_rep, y1_rep, x1_rep, h, w
            )
            rect_rep = mpatches.Rectangle(
                (rx0, ry0), rx1 - rx0, ry1 - ry0,
                linewidth=1, edgecolor="yellow", facecolor="none", linestyle="--",
            )
            ax.add_patch(rect_rep)

        ax.set_title(f"{label}={z_idx}", fontsize=7)
        ax.axis("off")


def _render_mip_panel(
    ax,
    ct_vol,           # np.ndarray, mmap_mode="r"
    row: Dict,
) -> None:
    """
    thin MIP panel 렌더링 (--include-mip True 일 때만 호출).
    z_min~z_max 범위에서 max-intensity projection (windowing 적용 후).
    제목에 "diagnostic aid only" 표시.
    score/threshold 미사용. 원본 미수정.
    matplotlib import는 함수 내부 지연 import.
    """
    import matplotlib  # 지연 import
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401
    import numpy as np  # 지연 import

    z_min = int(row["z_min"])
    z_max = int(row["z_max"])

    # z_min~z_max 범위 슬라이스 windowing 후 max projection
    # mmap이므로 slice-by-slice로 처리 (메모리 절약)
    slices_windowed = []
    for z_idx in range(z_min, z_max + 1):
        s = np.asarray(ct_vol[z_idx])
        slices_windowed.append(_window_ct(s))
    mip = np.max(np.stack(slices_windowed, axis=0), axis=0)

    ax.imshow(mip, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(f"thin MIP z{z_min}-{z_max}\n(diagnostic aid only)", fontsize=7)
    ax.axis("off")


def _render_metadata_text_panel(
    ax,
    row: Dict,
) -> None:
    """
    metadata text panel 렌더링:
    cluster3d_id, patient_id, source_group, z_min/z_max/z_span,
    representative_local_z, n_2d_clusters, n_patches_total,
    bbox_area, top3_mean_patch_score_3d, review_candidate_flag,
    overmerge_flag, large_bbox_overmerge_flag, large_extent_overmerge_flag,
    complex_merge_flag, high_score_ratio_flag, diagnostic_priority,
    diagnostic_note + 안내문구.
    matplotlib import는 함수 내부 지연 import.
    """
    import matplotlib  # 지연 import
    matplotlib.use("Agg")

    lines = [
        f"cluster3d_id: {row.get('cluster3d_id', '')}",
        f"patient_id: {row.get('patient_id', '')}",
        f"source_group: {row.get('source_group', '')}",
        f"z: {row.get('z_min', '')} ~ {row.get('z_max', '')}  span={row.get('z_span', '')}",
        f"rep_z: {row.get('representative_local_z', '')}",
        f"n_2d_clusters: {row.get('n_2d_clusters', '')}",
        f"n_patches_total: {row.get('n_patches_total', '')}",
        f"bbox_area: {row.get('bbox_area', '')}",
        f"top3_mean_score: {row.get('top3_mean_patch_score_3d', '')}",
        f"review_candidate: {row.get('review_candidate_flag', '')}",
        f"overmerge: {row.get('overmerge_flag', '')}",
        f"large_bbox_ovmg: {row.get('large_bbox_overmerge_flag', '')}",
        f"large_ext_ovmg: {row.get('large_extent_overmerge_flag', '')}",
        f"complex_merge: {row.get('complex_merge_flag', '')}",
        f"high_score_ratio: {row.get('high_score_ratio_flag', '')}",
        f"diag_priority: {row.get('diagnostic_priority', '')}",
        f"diag_note: {row.get('diagnostic_note', '')}",
        "",
        "* sample-local p99 based",
        "* threshold not finalized",
        "* lesion conclusion forbidden",
    ]

    text_str = "\n".join(lines)
    ax.text(
        0.02, 0.98,
        text_str,
        transform=ax.transAxes,
        fontsize=6,
        verticalalignment="top",
        horizontalalignment="left",
        family="monospace",
        wrap=True,
    )
    ax.axis("off")


def _render_label_guide_panel(
    ax,
) -> None:
    """
    label guide panel 렌더링: LABEL_GUIDE 항목 8종 표시.
    matplotlib import는 함수 내부 지연 import.
    """
    import matplotlib  # 지연 import
    matplotlib.use("Agg")

    header = "[ Label Guide ]"
    lines = [header, ""] + [f"  {i+1}. {lbl}" for i, lbl in enumerate(LABEL_GUIDE)]
    text_str = "\n".join(lines)

    ax.text(
        0.02, 0.98,
        text_str,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        horizontalalignment="left",
        family="monospace",
    )
    ax.axis("off")


def _build_visual_panel(
    ct_vol,            # np.ndarray, mmap_mode="r"
    roi_vol,           # np.ndarray or None, mmap_mode="r"
    row: Dict,
    output_png_path: Path,
    include_mip: bool = False,
) -> None:
    """
    단일 target의 전체 visual panel을 구성하고 PNG로 저장한다. (--run 시 실행)

    panel 구성:
    - center slice CT
    - representative bbox overlay
    - z_min / representative_z / z_max context (3개 서브플롯)
    - metadata text
    - label guide
    - optional thin MIP (--include-mip True 시)

    matplotlib 관련 import는 이 함수 내부에서만 수행 (지연 import).
    저장 직전 output_png_path 재검증 (기존 파일 없음 확인).
    tmp path에 저장 후 os.replace로 rename (실패 시 부분 파일 방지).
    plt.close(fig) 반드시 호출.
    한 target당 PNG 1개.
    """
    import matplotlib  # 지연 import
    matplotlib.use("Agg")  # WSL headless: plt import 전 반드시 Agg 강제
    import matplotlib.pyplot as plt  # noqa: F401

    # 저장 직전 output_png_path 재검증
    if output_png_path.exists():
        sys.exit(f"[ERROR] Output PNG already exists (pre-save check): {output_png_path}")

    # subplot 구성: include_mip에 따라 컬럼 수 결정
    # 고정 패널: center(1) + bbox_overlay(1) + z_context(3) + metadata(1) + label_guide(1) = 7
    # optional MIP(1)
    n_cols_top = 5  # center, bbox_overlay, z_min, z_rep, z_max
    n_cols_bottom = 2 + (1 if include_mip else 0)  # metadata, label_guide, [mip]

    # figure 구성: 2행 레이아웃
    # 상단: center slice + bbox overlay + z context 3개 (5열)
    # 하단: metadata + label guide + optional MIP
    fig = plt.figure(figsize=(18, 7))

    # 상단 5개 axes
    ax_center = fig.add_subplot(2, 5, 1)
    ax_bbox = fig.add_subplot(2, 5, 2)
    ax_z_min = fig.add_subplot(2, 5, 3)
    ax_z_rep = fig.add_subplot(2, 5, 4)
    ax_z_max = fig.add_subplot(2, 5, 5)

    # 하단: metadata (span 2열), label_guide (span 2열), optional mip (span 1열)
    ax_meta = fig.add_subplot(2, 5, 6)
    ax_label = fig.add_subplot(2, 5, 7)
    if include_mip:
        ax_mip = fig.add_subplot(2, 5, 8)

    try:
        _render_center_slice_panel(ax_center, ct_vol, row)
        _render_bbox_overlay_panel(ax_bbox, ct_vol, row, roi_vol)
        _render_z_context_panel(ct_vol, row, [ax_z_min, ax_z_rep, ax_z_max])
        _render_metadata_text_panel(ax_meta, row)
        _render_label_guide_panel(ax_label)
        if include_mip:
            _render_mip_panel(ax_mip, ct_vol, row)

        plt.tight_layout()

        # tmp path에 저장 후 os.replace로 rename (부분 파일 방지)
        tmp_path = output_png_path.with_suffix(".tmp.png")
        plt.savefig(str(tmp_path), dpi=100, bbox_inches="tight")
        os.replace(str(tmp_path), str(output_png_path))

    finally:
        plt.close(fig)


def _build_html_gallery(
    target_rows: List[Dict],
    png_paths: List[Path],
    output_html_path: Path,
) -> None:
    """
    14개 target의 PNG를 모아 HTML gallery를 생성한다. (--run 시 실행)
    - PNG 상대경로는 html 기준 상대 경로 (file:// 절대경로 금지)
    - user_label/user_note는 CSV에만 (HTML에는 LABEL_GUIDE만 표시)
    - 저장 직전 output_html_path 재검증 (기존 파일 없음 확인)
    - tmp 후 rename
    """
    if output_html_path.exists():
        sys.exit(f"[ERROR] Output HTML already exists (pre-save check): {output_html_path}")

    html_dir = output_html_path.parent
    label_guide_html = "\n".join(
        f"<li>{lbl}</li>" for lbl in LABEL_GUIDE
    )

    items_html_parts = []
    for row, png_path in zip(target_rows, png_paths):
        # html 기준 상대경로
        try:
            rel_png = png_path.relative_to(html_dir)
        except ValueError:
            rel_png = png_path  # fallback (동일 드라이브 내 절대경로 사용 불가 시)

        rel_png_str = str(rel_png).replace("\\", "/")

        item = f"""
  <div class="item">
    <p>
      <b>#{row.get('review_order', '')} {row.get('cluster3d_id', '')}</b>
      &nbsp; patient: {row.get('patient_id', '')}
      &nbsp; z_span: {row.get('z_span', '')}
      &nbsp; top3_score: {row.get('top3_mean_patch_score_3d', '')}
      &nbsp; overmerge: {row.get('overmerge_flag', '')}
      &nbsp; diag_note: {row.get('diagnostic_note', '')}
    </p>
    <img src="{rel_png_str}" style="max-width:100%;border:1px solid #888;" alt="{row.get('cluster3d_id', '')}">
  </div>
"""
        items_html_parts.append(item)

    items_html = "\n".join(items_html_parts)

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Weak 3D Visual Review Gallery</title>
<style>
  body {{ font-family: monospace; background: #222; color: #ddd; margin: 16px; }}
  .item {{ border: 1px solid #555; margin-bottom: 24px; padding: 8px; }}
  .item p {{ margin: 4px 0; font-size: 12px; }}
  img {{ display: block; margin-top: 8px; }}
  .label-guide {{ background: #333; padding: 8px; margin-bottom: 24px; }}
  .label-guide ul {{ margin: 4px 0; padding-left: 20px; }}
  .label-guide li {{ font-size: 12px; }}
</style>
</head>
<body>
<h2>Weak 3D Visual Review Gallery</h2>
<p style="color:#f80;font-size:12px;">
  * sample-local p99 based / threshold not finalized / lesion conclusion forbidden
</p>
<div class="label-guide">
  <b>Label Guide:</b>
  <ul>
{label_guide_html}
  </ul>
</div>
{items_html}
</body>
</html>
"""

    tmp_html = output_html_path.with_suffix(".tmp.html")
    try:
        with open(str(tmp_html), "w", encoding="utf-8") as fp:
            fp.write(html_content)
        os.replace(str(tmp_html), str(output_html_path))
    except Exception:
        if tmp_html.exists():
            tmp_html.unlink()
        raise


def _build_review_manifest_csv(
    target_rows: List[Dict],
    output_csv_path: Path,
) -> None:
    """
    visual review 결과 manifest CSV 생성 (user_label, user_note 빈칸 유지). (--run 시)
    - 입력 manifest 주요 컬럼 유지
    - png_path, html_relative_png_path, label_candidates 컬럼 추가
    - user_label/user_note 빈칸 유지
    - 입력 manifest 파일 자체 수정 금지 (읽기만, 새 파일에 씀)
    - 저장 직전 output_csv_path 재검증 (기존 파일 없음 확인)
    """
    if output_csv_path.exists():
        sys.exit(f"[ERROR] Output CSV already exists (pre-save check): {output_csv_path}")

    import csv  # 지연 import

    label_candidates_str = "|".join(LABEL_GUIDE)

    # 컬럼 순서: 기존 REQUIRED_MANIFEST_COLUMNS + 추가 컬럼
    extra_cols = ["png_path", "html_relative_png_path", "label_candidates"]
    fieldnames = list(REQUIRED_MANIFEST_COLUMNS) + extra_cols

    rows_out = []
    for row in target_rows:
        out_row = {col: row.get(col, "") for col in REQUIRED_MANIFEST_COLUMNS}
        out_row["png_path"] = row.get("_png_path", "")
        out_row["html_relative_png_path"] = row.get("_html_relative_png_path", "")
        out_row["label_candidates"] = label_candidates_str
        # user_label, user_note 빈칸 유지
        out_row["user_label"] = ""
        out_row["user_note"] = ""
        rows_out.append(out_row)

    tmp_csv = output_csv_path.with_suffix(".tmp.csv")
    try:
        with open(str(tmp_csv), "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)
        os.replace(str(tmp_csv), str(output_csv_path))
    except Exception:
        if tmp_csv.exists():
            tmp_csv.unlink()
        raise


def _build_summary_json(
    target_rows: List[Dict],
    args,
    output_json_path: Path,
) -> None:
    """
    실행 summary JSON 생성. (--run 시)
    notes 내 dry_run_only, weak_3d_merge_run, threshold_not_finalized,
    lesion_conclusion_forbidden, stage2_holdout_unused, v2_unused,
    sample_local_p99_input, no_model_forward, no_score_recalculation,
    original_files_unmodified, stage2_holdout_unused, v2_unused 키 포함.
    저장 직전 output_json_path 재검증 (기존 파일 없음 확인).
    """
    if output_json_path.exists():
        sys.exit(f"[ERROR] Output JSON already exists (pre-save check): {output_json_path}")

    n_targets = len(target_rows)

    # patient_distribution
    patient_dist: Dict[str, int] = {}
    for row in target_rows:
        pid = str(row.get("patient_id", "unknown"))
        patient_dist[pid] = patient_dist.get(pid, 0) + 1

    # z_span_distribution
    z_span_dist: Dict[str, int] = {}
    for row in target_rows:
        zs = str(row.get("z_span", "unknown"))
        z_span_dist[zs] = z_span_dist.get(zs, 0) + 1

    n_overmerge = sum(
        1 for row in target_rows
        if str(row.get("overmerge_flag", "")).lower() in ("true", "1", "yes")
    )
    n_review_candidate = sum(
        1 for row in target_rows
        if str(row.get("review_candidate_flag", "")).lower() in ("true", "1", "yes")
    )

    summary = {
        "output_tag": args.output_tag,
        "n_targets": n_targets,
        "n_png_created": {
            "expected": n_targets,
            "actual": n_targets,
        },
        "include_mip": args.include_mip,
        "patient_distribution": patient_dist,
        "z_span_distribution": z_span_dist,
        "n_overmerge_targets": n_overmerge,
        "n_review_candidate_targets": n_review_candidate,
        "output_root": str(output_json_path.parent),
        "html_path": str(output_json_path.parent / f"{args.output_tag}_gallery.html"),
        "review_manifest_path": str(
            output_json_path.parent / f"{args.output_tag}_review_manifest.csv"
        ),
        "notes": {
            "visual_pack_created": True,
            "dry_run_only": False,
            "sample_local_p99_input": True,
            "threshold_not_finalized": True,
            "lesion_conclusion_forbidden": True,
            "stage2_holdout_unused": True,
            "v2_unused": True,
            "no_model_forward": True,
            "no_score_recalculation": True,
            "original_files_unmodified": True,
        },
    }

    tmp_json = output_json_path.with_suffix(".tmp.json")
    try:
        with open(str(tmp_json), "w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)
        os.replace(str(tmp_json), str(output_json_path))
    except Exception:
        if tmp_json.exists():
            tmp_json.unlink()
        raise


# ── dry-run 로직 ──────────────────────────────────────────────────────────────
def _run_dry_run(args, df, output_root: Path) -> None:
    """
    dry-run 동작:
    - manifest 검증
    - visual source path 존재 확인 (stat only, npy 로드 금지)
    - meta shape 검증 + 좌표 검증 (npy 로드 금지)
    - 예상 생성 파일 수 계산
    - output root 생성 안 함
    - PNG/HTML/ZIP/CSV/JSON/MD 생성 안 함
    """
    print("[DRY-RUN] Phase 5.80 Weak 3D Visual Review Pack — dry-run mode")
    print(f"[DRY-RUN] output_root (will NOT be created): {output_root}")

    # 입력 파일 검증
    _validate_manifest(df, args.max_targets)

    # visual source path 존재 확인 (stat only)
    _validate_visual_source_paths(df)

    # 환자별 meta shape + 좌표계 검증 (npy 로드 금지)
    _validate_all_coordinates(df)

    # 처리 대상 수 계산
    n_targets = min(len(df), args.max_targets)
    n_panels_per_target = 5 + (1 if args.include_mip else 0)  # sub-panels 기준
    n_png = n_targets  # 1 PNG per target
    n_html = 1
    n_csv = 1
    n_json = 1

    print()
    print("[DRY-RUN] === Expected output (dry-run only, NOT creating) ===")
    print(f"  targets to process   : {n_targets}")
    print(f"  PNG panels           : {n_png}  (1 per target)")
    print(f"  sub-panels/PNG       : {n_panels_per_target}")
    print(f"  include_mip          : {args.include_mip}")
    print(f"  HTML gallery         : {n_html}")
    print(f"  review manifest CSV  : {n_csv}")
    print(f"  summary JSON         : {n_json}")
    print(f"  output_root          : {output_root}  [NOT created]")
    print()
    print("[DRY-RUN] OK — no files written, no directories created.")


# ── run 로직 ──────────────────────────────────────────────────────────────────
def _run_main(args, df, output_root: Path) -> None:
    """
    --run 모드: 실제 visual panel PNG + HTML gallery + review manifest CSV + summary JSON 생성.

    경고: --run은 사용자 승인 없이 절대 실행 금지.

    메모리 전략:
    - one-patient-at-a-time: 환자별로 ct_hu.npy + roi_0_0.npy 로드
    - mmap_mode="r": 전체 volume을 메모리에 올리지 않음
    - 예상 volume 크기: 약 120~130 MB / 환자 (float32, 243~250 z-slices, 512x512)
    - 동시 다중 volume cache 금지: 환자별 처리 완료 후 다음 환자로 이동
    - np.load(..., mmap_mode="r") 사용
    """
    # --run 미구현 차단 guard (mkdir / np.load 보다 반드시 먼저 실행)
    if args.run and not RUN_MODE_IMPLEMENTED:
        sys.exit("[ERROR] --run is not implemented yet. Use --dry-run only.")

    # --run 실행 경고 출력 (승인 없이 실행 금지 안내)
    print("[WARNING] --run mode: 사용자 승인 없이 절대 실행 금지.")
    print("[WARNING] This mode writes PNG/HTML/CSV/JSON to disk.")
    print()

    # numpy는 실제 실행 시 지연 import
    import numpy as np  # noqa: F401

    # 입력 검증
    _validate_manifest(df, args.max_targets)
    _validate_visual_source_paths(df)
    _validate_all_coordinates(df)

    # 처리 대상 (max_targets 제한)
    df_targets = df.head(args.max_targets)

    # output root 존재 확인 (생성 전 검증)
    if output_root.exists():
        sys.exit(f"[ERROR] Output root already exists: {output_root}")

    # panels/ 서브디렉토리 경로
    panels_dir = output_root / "panels"

    # 출력 파일 경로 정의
    out_html = output_root / f"{args.output_tag}_gallery.html"
    out_csv = output_root / f"{args.output_tag}_review_manifest.csv"
    out_json = output_root / f"{args.output_tag}_summary.json"

    # guard용 out_png_list: df_targets 전체를 순회해 각 row의 review_order/patient_id/cluster3d_id로 파일명 생성
    # (사전 존재 확인 전용 — 처리 시 row별 직접 생성하므로 순서 매핑에 사용하지 않음)
    out_png_list: List[Path] = []
    for _, row in df_targets.iterrows():
        review_order_ = int(row["review_order"])
        pid_ = str(row["patient_id"])
        cid_ = str(row["cluster3d_id"])
        out_png_list.append(panels_dir / f"{review_order_:03d}_{pid_}_{cid_}.png")

    # 출력 파일 존재 확인 (mkdir 전)
    for fpath in [out_html, out_csv, out_json] + out_png_list:
        if fpath.exists():
            sys.exit(f"[ERROR] Output file already exists: {fpath}")

    # mkdir (모든 검증 이후) — panels/ 서브디렉토리 포함
    output_root.mkdir(parents=True, exist_ok=False)
    panels_dir.mkdir(parents=True, exist_ok=False)

    # mkdir 이후 재검증 (파일이 예상치 않게 존재하는지 확인)
    for fpath in [out_html, out_csv, out_json] + out_png_list:
        if fpath.exists():
            sys.exit(f"[ERROR] Output file unexpectedly exists after mkdir: {fpath}")

    # 환자별 one-at-a-time 처리
    # mmap_mode="r": 예상 volume 크기 약 120~130 MB/환자
    processed_png_paths: List[Path] = []
    target_rows: List[Dict] = []

    for pid in sorted(df_targets["patient_id"].unique()):
        # 해당 환자 첫 번째 ct_hu.npy 경로로 volume 경로 결정
        pid_rows = df_targets[df_targets["patient_id"] == pid]
        ct_path = Path(str(pid_rows.iloc[0]["visual_source_candidate_path"]))
        roi_path = ct_path.parent / "roi_0_0.npy"

        _guard_path(ct_path)
        _guard_path(roi_path)

        print(f"[RUN] Loading patient {pid} volume (mmap_mode='r') ...")
        # mmap_mode="r": 실제 메모리 로드 없이 접근
        ct_vol = np.load(str(ct_path), mmap_mode="r")
        roi_vol = np.load(str(roi_path), mmap_mode="r")
        print(
            f"[RUN] {pid}: ct_vol shape={ct_vol.shape}, "
            f"roi_vol shape={roi_vol.shape}"
        )

        # 해당 환자 target 처리
        for _, row in pid_rows.iterrows():
            row_dict = row.to_dict()
            # row별 직접 PNG 경로 생성 (png_index 참조 완전 제거)
            review_order = int(row["review_order"])
            pid_ = str(row["patient_id"])
            cid = str(row["cluster3d_id"])
            out_png = panels_dir / f"{review_order:03d}_{pid_}_{cid}.png"

            # PNG 상대 경로 정보 row_dict에 추가 (CSV용)
            try:
                rel_png = out_png.relative_to(output_root)
            except ValueError:
                rel_png = out_png
            row_dict["_png_path"] = str(out_png)
            row_dict["_html_relative_png_path"] = str(rel_png).replace("\\", "/")

            _build_visual_panel(
                ct_vol=ct_vol,
                roi_vol=roi_vol,
                row=row_dict,
                output_png_path=out_png,
                include_mip=args.include_mip,
            )
            processed_png_paths.append(out_png)
            target_rows.append(row_dict)
            print(f"[RUN] Panel saved: {out_png.name}")

        # 환자별 처리 완료 후 명시적 해제 (다음 환자 처리 전)
        del ct_vol
        del roi_vol

    # ── row-PNG 정합 검증 (HTML/CSV 저장 전, 실패 시 sys.exit) ─────────────────
    # 1. 각 row의 review_order/patient_id/cluster3d_id가 _png_path 파일명에 포함되는지 확인
    mismatch_errors: List[str] = []
    for r in target_rows:
        r_order = int(r["review_order"])
        r_pid = str(r["patient_id"])
        r_cid = str(r["cluster3d_id"])
        expected_fname = f"{r_order:03d}_{r_pid}_{r_cid}.png"
        actual_fname = Path(r["_png_path"]).name
        if actual_fname != expected_fname:
            mismatch_errors.append(
                f"review_order={r_order} patient_id={r_pid} cluster3d_id={r_cid}: "
                f"expected={expected_fname!r}, actual={actual_fname!r}"
            )
    if mismatch_errors:
        for err in mismatch_errors:
            print(f"[ERROR] row-PNG mapping mismatch: {err}")
        sys.exit(f"[ERROR] row-PNG mapping mismatch: {len(mismatch_errors)} error(s) found.")

    # 2. review_order 중복 0건 확인
    seen_orders: List[int] = []
    dup_orders: List[int] = []
    for r in target_rows:
        ro = int(r["review_order"])
        if ro in seen_orders:
            dup_orders.append(ro)
        else:
            seen_orders.append(ro)
    if dup_orders:
        sys.exit(f"[ERROR] Duplicate review_order found: {sorted(set(dup_orders))}")

    # 3. _png_path 중복 0건 확인
    seen_png_paths: List[str] = []
    dup_png_paths: List[str] = []
    for r in target_rows:
        pp = r["_png_path"]
        if pp in seen_png_paths:
            dup_png_paths.append(pp)
        else:
            seen_png_paths.append(pp)
    if dup_png_paths:
        sys.exit(f"[ERROR] Duplicate _png_path found: {dup_png_paths}")

    # 4. _html_relative_png_path가 _png_path와 동일한 파일명 가리키는지 확인
    html_rel_errors: List[str] = []
    for r in target_rows:
        png_fname = Path(r["_png_path"]).name
        html_rel_fname = Path(r["_html_relative_png_path"]).name
        if png_fname != html_rel_fname:
            html_rel_errors.append(
                f"cluster3d_id={r['cluster3d_id']}: "
                f"_png_path.name={png_fname!r} != _html_relative_png_path.name={html_rel_fname!r}"
            )
    if html_rel_errors:
        for err in html_rel_errors:
            print(f"[ERROR] html_relative_png_path mismatch: {err}")
        sys.exit(f"[ERROR] html_relative_png_path mismatch: {len(html_rel_errors)} error(s) found.")

    print(f"[RUN] row-PNG mapping validation passed: {len(target_rows)} rows OK.")

    # ── HTML/CSV 호출 전 review_order 오름차순 정렬 ──────────────────────────────
    target_rows_sorted = sorted(target_rows, key=lambda r: r["review_order"])
    processed_png_paths_sorted = [Path(r["_png_path"]) for r in target_rows_sorted]

    # HTML gallery 생성
    _build_html_gallery(target_rows_sorted, processed_png_paths_sorted, out_html)
    print(f"[RUN] HTML gallery saved: {out_html.name}")

    # review manifest CSV 생성
    _build_review_manifest_csv(target_rows_sorted, out_csv)
    print(f"[RUN] Review manifest CSV saved: {out_csv.name}")

    # summary JSON 생성
    _build_summary_json(target_rows, args, out_json)
    print(f"[RUN] Summary JSON saved: {out_json.name}")

    print()
    print(f"[RUN] Done. Output: {output_root}")
    print(f"  targets processed : {len(target_rows)}")
    print(f"  PNG panels        : {len(processed_png_paths)}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    경고: --run 모드는 사용자 승인 없이 절대 실행 금지.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5.80 Weak 3D Cluster Visual Review Pack\n"
            "경고: --run 모드는 사용자 승인 없이 절대 실행 금지."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode: validate inputs, check paths, no output written.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="[사용자 승인 없이 절대 실행 금지] Run mode: generate PNG/HTML/CSV/JSON.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=14,
        help="Maximum number of targets to process (default: 14).",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="phase5_80_weak_3d_visual_review_pack_v1",
        help="Output directory tag name (alphanumeric/underscore/hyphen only).",
    )
    parser.add_argument(
        "--include-mip",
        action="store_true",
        default=False,
        help="Include optional thin MIP panel in visual output (default: False).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="[FORBIDDEN] --force is not allowed.",
    )
    args = parser.parse_args()

    # guard: --force 즉시 중단
    if args.force:
        sys.exit("[ERROR] --force is not allowed.")

    # guard: --dry-run xor --run (정확히 하나만)
    if not args.dry_run and not args.run:
        sys.exit(
            "[ERROR] Exactly one of --dry-run or --run must be specified.\n"
            "  Use --dry-run to validate without writing output.\n"
            "  Use --run only after user approval."
        )
    if args.dry_run and args.run:
        sys.exit(
            "[ERROR] --dry-run and --run cannot be used together. "
            "Specify exactly one."
        )

    # --run 진입부 경고 (사용자 승인 없이 실행 금지 안내)
    if args.run:
        # 이중 방어: _run_main 진입 전 main()에서도 차단
        if not RUN_MODE_IMPLEMENTED:
            sys.exit("[ERROR] --run is not implemented yet. Use --dry-run only.")
        print("[WARNING] --run mode entered. 사용자 승인 없이 절대 실행 금지.")
        print("[WARNING] Proceeding only if explicitly approved by the user.")
        print()

    # output_tag 검증
    _validate_output_tag(args.output_tag)

    # output_root 구성 (BASE / output_tag 로만)
    output_root = _OUTPUT_BASE / args.output_tag

    # path guards
    for p in (_MANIFEST_PATH, _PREFLIGHT_JSON_PATH, _OUTPUT_BASE, output_root):
        _guard_path(p)

    # 입력 파일 존재 확인
    if not _MANIFEST_PATH.exists():
        sys.exit(f"[ERROR] Manifest not found: {_MANIFEST_PATH}")
    if not _PREFLIGHT_JSON_PATH.exists():
        sys.exit(f"[ERROR] Preflight JSON not found: {_PREFLIGHT_JSON_PATH}")

    # pandas import (실행 시점)
    import pandas as pd  # noqa: F401

    df = pd.read_csv(_MANIFEST_PATH)

    # 실행 모드 분기
    if args.dry_run:
        _run_dry_run(args, df, output_root)
    else:
        # args.run == True
        _run_main(args, df, output_root)


if __name__ == "__main__":
    # 경고: main()은 이 가드 안에서만 호출됨.
    # import 또는 py_compile 시 main()이 실행되지 않음.
    main()
