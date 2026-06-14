#!/usr/bin/env python3
"""
Phase 5.95 Expanded Candidate Visual Pack
============================================================
목적: Phase 5.94에서 확정한 expanded candidate visual target 15개에 대해
     CT slice panel + candidate bbox overlay + metadata panel로 구성된
     visual review pack을 생성한다.

경고:
  - --preflight-only: 경로 확인만, output 생성 없음
  - --smoke: 최대 2개 target 처리 (사용자 별도 승인 필요)
  - --run: 15개 target 전체 처리 (사용자 별도 승인 필요)
  - 세 모드 중 정확히 하나만 지정 필요.
  - 이번 단계에서는 어떤 모드도 실행 금지.

메모리 전략: one-patient-at-a-time, mmap_mode="r", 동시 다중 volume cache 금지

safety notes:
  preflight_or_visual_review_only | no_model_forward | no_score_recalculation |
  threshold_not_finalized | lesion_conclusion_forbidden | hard_negative_not_finalized |
  not_training_manifest | no_training_dataset_modification | no_crop_manifest_modification |
  stage2_holdout_unused | v2_unused | sample_local_not_global |
  roi_0_0_patch_ratio_not_confirmed_as_pure_lung |
  candidate_class_hint_unknown_requires_visual_review
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 프로젝트 루트 ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 입력 파일 경로 ─────────────────────────────────────────────────────────────
_TARGET_CSV_PATH = _PROJECT_ROOT / (
    "outputs/second-stage-lesion-refiner-v1/review_annotations"
    "/phase5_94_expanded_candidate_visual_pack_preflight_v1"
    "/phase5_94_expanded_candidate_visual_pack_preflight_targets_v1.csv"
)
_TARGET_JSON_PATH = _PROJECT_ROOT / (
    "outputs/second-stage-lesion-refiner-v1/review_annotations"
    "/phase5_94_expanded_candidate_visual_pack_preflight_v1"
    "/phase5_94_expanded_candidate_visual_pack_preflight_summary_v1.json"
)
_SPLIT_CSV_PATH = _PROJECT_ROOT / (
    "data/normal_training_ready/manifests/train_val_test_split.csv"
)

# ── output base ───────────────────────────────────────────────────────────────
_OUTPUT_BASE = _PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/visualizations"

# ── constants ─────────────────────────────────────────────────────────────────
EXPECTED_ROW_COUNT = 15
EXPECTED_PATIENTS = {"normal023", "normal024", "normal036"}
EXPECTED_PER_PATIENT = 5
MAX_SMOKE_TARGETS = 2
MAX_RUN_TARGETS = 15

SAFETY_NOTES = [
    "preflight_or_visual_review_only",
    "no_model_forward",
    "no_score_recalculation",
    "threshold_not_finalized",
    "lesion_conclusion_forbidden",
    "hard_negative_not_finalized",
    "not_training_manifest",
    "no_training_dataset_modification",
    "no_crop_manifest_modification",
    "stage2_holdout_unused",
    "v2_unused",
    "sample_local_not_global",
    "roi_0_0_patch_ratio_not_confirmed_as_pure_lung",
    "candidate_class_hint_unknown_requires_visual_review",
]

# ASCII-safe metadata panel text (glyph warning 방지)
METADATA_PANEL_TEXT: List[str] = [
    "sample-local p99 based",
    "threshold not finalized",
    "not hard negative final",
    "lesion conclusion forbidden",
    "requires visual review",
    "candidate_class_hint: unknown",
    "roi_0_0_patch_ratio not confirmed as pure_lung",
]

LABEL_GUIDE: List[str] = [
    "requires_visual_review",
    "unknown_lesion_candidate",
    "normal_structure",
    "vessel",
    "bronchus",
    "pleural",
    "artifact",
    "unclear",
]

SHAPE_KEY_CANDIDATES: List[str] = [
    "shape_zyx",
    "shape",
    "ct_shape",
    "volume_shape",
    "image_shape",
    "array_shape",
]

REQUIRED_MANIFEST_COLUMNS: List[str] = [
    "target_order",
    "patient_id",
    "split",
    "local_z",
    "y0",
    "x0",
    "y1",
    "x1",
    "patch_size",
    "patch_stride",
    "padim_score",
    "roi_0_0_patch_ratio",
    "candidate_score_percentile_mode",
    "candidate_class_hint",
    "visual_review_status",
    "expected_png_filename",
]

REVIEW_MANIFEST_COLUMNS: List[str] = [
    "target_order",
    "expected_png_filename",
    "actual_png_filename",
    "patient_id",
    "split",
    "local_z",
    "y0",
    "x0",
    "y1",
    "x1",
    "patch_size",
    "patch_stride",
    "padim_score",
    "roi_0_0_patch_ratio",
    "candidate_class_hint",
    "visual_review_status",
    "png_path",
    "html_relative_png_path",
    "user_label",
    "user_note",
    "error_reason",
    "limitation",
]


# ── output tag validation ──────────────────────────────────────────────────────
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


# ── path guard (segment 단위, second-stage-lesion-refiner-v1 오탐 방지) ────────
def _guard_path(p: Path) -> None:
    """
    경로의 각 segment를 검사한다.
    second-stage-lesion-refiner-v1 내부 'lesion' 문자열은 오탐 차단하지 않음.
    segment 단위로 forbidden 목록과 비교한다.
    """
    for part in p.parts:
        pl = part.lower()
        if (
            "stage2_holdout" in pl
            or pl == "v2"
            or pl.startswith("v2v2")
            or pl.startswith("v2_")
            or "lesion_by_patient" in pl
            or "crops_lesion" in pl
            or "hard_negative" in pl
            or pl == "nsclc"
            or "nsclc_msd" in pl
            or pl == "msd"
            or "msd_lung" in pl
            or pl == "lung1"
            or pl == "src02"
        ):
            sys.exit(f"[ERROR] Forbidden path segment '{part}' in {p}")


# ── meta.json shape 로드 (read-only, npy 로드 금지) ──────────────────────────
def load_meta_shape(meta_path: Path) -> Optional[Tuple[int, int, int]]:
    """
    meta.json을 읽어 (z_dim, y_dim, x_dim) 튜플을 반환한다.
    npy 로드 절대 금지. JSON 읽기만 허용.
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
            if isinstance(val, (list, tuple)) and len(val) == 3:
                try:
                    return (int(val[0]), int(val[1]), int(val[2]))
                except (ValueError, TypeError):
                    continue
    return None


# ── safe_id map 로드 ──────────────────────────────────────────────────────────
def _load_safe_id_map(split_csv_path: Path) -> Dict[str, str]:
    """
    train_val_test_split.csv에서 patient_id → safe_id 매핑을 로드한다.
    Windows 경로가 포함된 다른 컬럼은 읽지 않는다.
    """
    if not split_csv_path.exists():
        sys.exit(f"[ERROR] train_val_test_split.csv not found: {split_csv_path}")
    safe_id_map: Dict[str, str] = {}
    try:
        with open(split_csv_path, "r", encoding="utf-8-sig") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                pid = str(row["patient_id"]).strip()
                sid = str(row["safe_id"]).strip()
                safe_id_map[pid] = sid
    except (KeyError, OSError) as e:
        sys.exit(f"[ERROR] Failed to load safe_id map from {split_csv_path}: {e}")
    return safe_id_map


# ── volume path resolution (safe_id 기반, Windows 경로 사용 안 함) ────────────
def _resolve_volume_paths(
    patient_id: str,
    safe_id: str,
    volume_root: Path,
    ct_filename: str,
    roi_filename: str,
    meta_filename: str,
) -> Tuple[Path, Path, Path]:
    """
    safe_id 기반으로 CT/ROI/meta 경로를 구성한다.
    npy 로드는 하지 않는다.
    """
    patient_dir = volume_root / safe_id
    _guard_path(patient_dir)
    ct_path = patient_dir / ct_filename
    roi_path = patient_dir / roi_filename
    meta_path = patient_dir / meta_filename
    return ct_path, roi_path, meta_path


# ── PNG 파일명 생성 (row 기반, png_index 기반 매핑 금지) ─────────────────────
def _make_png_filename(
    target_order: str,
    patient_id: str,
    local_z: int,
    y0: int,
    x0: int,
) -> str:
    """
    target_order/patient_id/local_z/y0/x0에서 PNG 파일명을 직접 생성한다.
    png_index 기반 매핑 금지.
    """
    return f"{target_order}_{patient_id}_z{local_z}_y{y0}_x{x0}.png"


# ── target manifest validation ────────────────────────────────────────────────
def _validate_target_manifest(df) -> None:
    """
    Phase 5.94 target CSV 구조 및 내용을 검증한다.
    검증 실패 시 즉시 sys.exit.
    """
    if len(df) != EXPECTED_ROW_COUNT:
        sys.exit(
            f"[ERROR] Target manifest expected {EXPECTED_ROW_COUNT} rows, got {len(df)}"
        )

    missing_cols = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in df.columns]
    if missing_cols:
        sys.exit(f"[ERROR] Target manifest missing required columns: {missing_cols}")

    actual_patients = set(df["patient_id"].unique())
    if actual_patients != EXPECTED_PATIENTS:
        sys.exit(
            f"[ERROR] patient_id mismatch. Got: {sorted(actual_patients)}, "
            f"Expected: {sorted(EXPECTED_PATIENTS)}"
        )

    for pid in EXPECTED_PATIENTS:
        n = len(df[df["patient_id"] == pid])
        if n != EXPECTED_PER_PATIENT:
            sys.exit(
                f"[ERROR] {pid}: expected {EXPECTED_PER_PATIENT} targets, got {n}"
            )

    orders = sorted(df["target_order"].astype(str).str.zfill(3).tolist())
    expected_orders = [f"{i:03d}" for i in range(1, 16)]
    if orders != expected_orders:
        sys.exit(
            f"[ERROR] target_order mismatch. Got: {orders}, Expected: {expected_orders}"
        )

    if df["target_order"].astype(str).duplicated().any():
        dup = df[df["target_order"].astype(str).duplicated(keep=False)]["target_order"].tolist()
        sys.exit(f"[ERROR] Duplicate target_order found: {dup}")

    if df["expected_png_filename"].duplicated().any():
        dup = df[df["expected_png_filename"].duplicated(keep=False)]["expected_png_filename"].tolist()
        sys.exit(f"[ERROR] Duplicate expected_png_filename found: {dup}")

    bad_hint = df[df["candidate_class_hint"] != "unknown_requires_visual_review"]
    if len(bad_hint) > 0:
        sys.exit(
            f"[ERROR] candidate_class_hint is not 'unknown_requires_visual_review' "
            f"for {len(bad_hint)} rows."
        )

    for idx, row in df.iterrows():
        vrs = str(row["visual_review_status"]).lower()
        ok = any(kw in vrs for kw in ["pending", "requires", "not_hard", "not_training"])
        if not ok:
            sys.exit(
                f"[ERROR] visual_review_status unexpected at row {idx}: "
                f"{row['visual_review_status']!r}"
            )

    bad_split = df[df["split"] != "val"]
    if len(bad_split) > 0:
        sys.exit(f"[ERROR] split is not 'val' for {len(bad_split)} rows.")

    bad_mode = df[df["candidate_score_percentile_mode"] != "sample_local_not_global"]
    if len(bad_mode) > 0:
        sys.exit(
            f"[ERROR] candidate_score_percentile_mode is not 'sample_local_not_global' "
            f"for {len(bad_mode)} rows."
        )

    if "candidate_status" in df.columns:
        for idx, row in df.iterrows():
            status = str(row["candidate_status"])
            if "hard_negative_final" in status and "not_hard_negative_final" not in status:
                sys.exit(f"[ERROR] hard_negative_final detected at row {idx}: {status!r}")

    if "notes" in df.columns:
        for idx, row in df.iterrows():
            notes = str(row["notes"])
            if "threshold_finalized" in notes and "threshold_not_finalized" not in notes:
                sys.exit(f"[ERROR] threshold_finalized detected at row {idx}: {notes!r}")

    # row-PNG 정합 검증 (png_index 기반 매핑 금지)
    computed_fnames = []
    for _, row in df.iterrows():
        computed = _make_png_filename(
            str(row["target_order"]).zfill(3),
            str(row["patient_id"]),
            int(row["local_z"]),
            int(row["y0"]),
            int(row["x0"]),
        )
        expected = str(row["expected_png_filename"])
        if computed != expected:
            sys.exit(
                f"[ERROR] PNG filename mismatch for target_order={row['target_order']}: "
                f"computed={computed!r}, expected={expected!r}"
            )
        computed_fnames.append(computed)

    if len(computed_fnames) != len(set(computed_fnames)):
        sys.exit("[ERROR] Computed PNG filenames are not unique.")

    print(
        f"[INFO] Target manifest validated: {len(df)} rows, "
        f"patients={sorted(actual_patients)}"
    )


# ── source path validation (stat only, npy 로드 금지) ─────────────────────────
def _validate_source_paths(
    df,
    safe_id_map: Dict[str, str],
    volume_root: Path,
    ct_filename: str,
    roi_filename: str,
    meta_filename: str,
) -> Dict[str, Tuple[Path, Path, Path]]:
    """
    각 target 환자의 CT/ROI/meta 경로를 stat-only로 확인한다.
    npy 로드 절대 금지.
    반환: patient_id → (ct_path, roi_path, meta_path)
    """
    errors = []
    patient_paths: Dict[str, Tuple[Path, Path, Path]] = {}
    seen_patients: set = set()

    for _, row in df.iterrows():
        pid = str(row["patient_id"])
        if pid in seen_patients:
            continue
        seen_patients.add(pid)

        if pid not in safe_id_map:
            errors.append(
                f"patient_id={pid!r} not found in train_val_test_split.csv"
            )
            continue

        safe_id = safe_id_map[pid]
        ct_path, roi_path, meta_path = _resolve_volume_paths(
            pid, safe_id, volume_root, ct_filename, roi_filename, meta_filename
        )
        _guard_path(ct_path)
        _guard_path(roi_path)

        for fpath, label in [(ct_path, "ct"), (roi_path, "roi"), (meta_path, "meta")]:
            try:
                fpath.stat()
            except FileNotFoundError:
                errors.append(f"File not found [{label}]: {fpath} (patient={pid})")
            except OSError as exc:
                errors.append(f"File access error [{label}]: {fpath} — {exc}")

        patient_paths[pid] = (ct_path, roi_path, meta_path)

    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        sys.exit(f"[ERROR] {len(errors)} source file(s) not accessible.")

    print(
        f"[INFO] All source files accessible (stat-only, no npy load). "
        f"patients={sorted(seen_patients)}"
    )
    return patient_paths


# ── coordinate validation (meta.json JSON만, npy 로드 금지) ───────────────────
def _validate_all_coordinates(
    df,
    patient_paths: Dict[str, Tuple[Path, Path, Path]],
) -> None:
    """
    각 target의 local_z/y0/x0/y1/x1을 meta.json shape과 비교 검증한다.
    환자별 meta를 개별 로드한다. npy 로드 금지.
    """
    patient_shape_cache: Dict[str, Optional[Tuple[int, int, int]]] = {}
    for pid, (ct_path, roi_path, meta_path) in patient_paths.items():
        shape = load_meta_shape(meta_path)
        patient_shape_cache[pid] = shape
        if shape is None:
            print(
                f"[WARNING] {pid}: meta shape unknown. Coordinate validation skipped."
            )
        else:
            print(f"[INFO] {pid}: meta shape (z,y,x)={shape}")

    coord_errors = []
    for _, row in df.iterrows():
        pid = str(row["patient_id"])
        order = str(row["target_order"])
        shape = patient_shape_cache.get(pid)
        if shape is None:
            continue
        z_dim, y_dim, x_dim = shape
        local_z = int(row["local_z"])
        y0 = int(row["y0"])
        x0 = int(row["x0"])
        y1 = int(row["y1"])
        x1 = int(row["x1"])
        if not (0 <= local_z < z_dim):
            coord_errors.append(
                f"target_order={order}: local_z={local_z} out of z_dim={z_dim}"
            )
        if not (0 <= y0 < y1 <= y_dim):
            coord_errors.append(
                f"target_order={order}: y range invalid y0={y0} y1={y1} y_dim={y_dim}"
            )
        if not (0 <= x0 < x1 <= x_dim):
            coord_errors.append(
                f"target_order={order}: x range invalid x0={x0} x1={x1} x_dim={x_dim}"
            )

    if coord_errors:
        for e in coord_errors:
            print(f"[ERROR] Coordinate validation failed — {e}")
        sys.exit(f"[ERROR] {len(coord_errors)} coordinate error(s) found.")

    n_checked = sum(1 for s in patient_shape_cache.values() if s is not None)
    n_skipped = sum(1 for s in patient_shape_cache.values() if s is None)
    print(
        f"[INFO] Coordinate validation complete: "
        f"{n_checked} patients checked, {n_skipped} skipped (shape unknown)."
    )


# ── CT windowing helper ────────────────────────────────────────────────────────
def _window_ct(slice_2d, vmin: float = -1000.0, vmax: float = 400.0):
    """CT 2D slice에 window/level 적용 후 0~1 float 반환. 원본 수정 금지."""
    import numpy as np  # 지연 import
    arr = np.array(slice_2d, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=vmin, posinf=vmax, neginf=vmin)
    arr = (arr - vmin) / (vmax - vmin)
    return np.clip(arr, 0.0, 1.0)


# ── bbox clip helper ──────────────────────────────────────────────────────────
def _clip_bbox(
    y0: int, x0: int, y1: int, x1: int, h: int, w: int
) -> Tuple[Tuple[int, int, int, int], bool]:
    cy0 = max(0, min(y0, h))
    cx0 = max(0, min(x0, w))
    cy1 = max(0, min(y1, h))
    cx1 = max(0, min(x1, w))
    clipped = (cy0, cx0, cy1, cx1) != (y0, x0, y1, x1)
    return (cy0, cx0, cy1, cx1), clipped


# ── 단일 target PNG 렌더링 ────────────────────────────────────────────────────
def _render_target_panel(
    row: Dict,
    ct_vol,   # np.ndarray, mmap_mode="r", shape=(z, y, x)
    roi_vol,  # np.ndarray or None, mmap_mode="r"
    output_root: Path,
    output_tag: str,
) -> Tuple[Optional[Path], str]:
    """
    단일 target에 대해 5개 패널 PNG를 렌더링한다.
    패널: CT+ROI overlay / bbox overlay / local context / metadata / label guide
    반환: (png_path, error_reason). 실패 시 (None, reason).
    """
    import matplotlib  # 지연 import
    matplotlib.use("Agg")  # WSL headless: plt import 전 반드시 Agg 강제
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np  # 지연 import

    panels_dir = output_root / "panels"
    order = str(row["target_order"]).zfill(3)
    pid = str(row["patient_id"])
    local_z = int(row["local_z"])
    y0 = int(row["y0"])
    x0 = int(row["x0"])
    y1 = int(row["y1"])
    x1 = int(row["x1"])
    patch_size = int(row["patch_size"])
    padim_score = float(row["padim_score"])
    roi_ratio = float(row["roi_0_0_patch_ratio"])
    expected_fname = str(row["expected_png_filename"])

    # row 기반 PNG 파일명 생성 (png_index 기반 매핑 금지)
    computed_fname = _make_png_filename(order, pid, local_z, y0, x0)
    if computed_fname != expected_fname:
        return (
            None,
            f"PNG filename mismatch: computed={computed_fname!r} expected={expected_fname!r}",
        )

    png_path = panels_dir / computed_fname

    # output overwrite guard
    if png_path.exists():
        return (None, f"Output file already exists: {png_path}")

    z_dim, y_dim, x_dim = ct_vol.shape
    if not (0 <= local_z < z_dim):
        return (None, f"local_z={local_z} out of z_dim={z_dim}")

    (cy0, cx0, cy1, cx1), clipped = _clip_bbox(y0, x0, y1, x1, y_dim, x_dim)
    ct_slice = np.asarray(ct_vol[local_z])
    windowed = _window_ct(ct_slice)
    roi_slice = np.asarray(roi_vol[local_z]).astype(np.float32) if roi_vol is not None else None

    half = patch_size // 2 + 16
    ctx_y0 = max(0, cy0 - half)
    ctx_x0 = max(0, cx0 - half)
    ctx_y1 = min(y_dim, cy1 + half)
    ctx_x1 = min(x_dim, cx1 + half)
    local_ctx = windowed[ctx_y0:ctx_y1, ctx_x0:ctx_x1]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(
        f"Phase5.95 | {pid} | target={order} | z={local_z} | score={padim_score:.4f}",
        fontsize=8,
    )

    # Panel 1: CT slice with ROI overlay
    ax = axes[0]
    ax.imshow(windowed, cmap="gray", vmin=0.0, vmax=1.0)
    if roi_slice is not None:
        ax.imshow(roi_slice > 0.5, cmap="Reds", alpha=0.25, vmin=0.0, vmax=1.0)
    ax.set_title(f"CT slice z={local_z}\nROI overlay", fontsize=7)
    ax.axis("off")

    # Panel 2: candidate bbox overlay
    ax = axes[1]
    ax.imshow(windowed, cmap="gray", vmin=0.0, vmax=1.0)
    rect = mpatches.Rectangle(
        (cx0, cy0), cx1 - cx0, cy1 - cy0,
        linewidth=1.5, edgecolor="yellow", facecolor="none",
    )
    ax.add_patch(rect)
    bbox_note = " [clipped]" if clipped else ""
    ax.set_title(
        f"Candidate bbox{bbox_note}\ny0={cy0} x0={cx0} y1={cy1} x1={cx1}", fontsize=7
    )
    ax.axis("off")

    # Panel 3: local context around candidate patch
    ax = axes[2]
    ax.imshow(local_ctx, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(
        f"Local context\n[{ctx_y0}:{ctx_y1}, {ctx_x0}:{ctx_x1}]", fontsize=7
    )
    ax.axis("off")

    # Panel 4: metadata panel (ASCII-safe only, glyph warning 방지)
    ax = axes[3]
    ax.axis("off")
    meta_lines = (
        [
            f"patient: {pid}",
            f"target_order: {order}",
            f"local_z: {local_z}",
            f"score: {padim_score:.4f}",
            f"roi_0_0_patch_ratio: {roi_ratio:.4f}",
            "---",
        ]
        + METADATA_PANEL_TEXT
    )
    ax.text(
        0.02, 0.98, "\n".join(meta_lines),
        transform=ax.transAxes,
        fontsize=6, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    ax.set_title("Metadata (ASCII-safe)", fontsize=7)

    # Panel 5: label guide panel
    ax = axes[4]
    ax.axis("off")
    guide_lines = ["Label Guide:"] + [f"  {lbl}" for lbl in LABEL_GUIDE]
    ax.text(
        0.02, 0.98, "\n".join(guide_lines),
        transform=ax.transAxes,
        fontsize=6, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8),
    )
    ax.set_title("Label Guide", fontsize=7)

    # 저장 직전 재확인
    if png_path.exists():
        plt.close(fig)
        return (None, f"Output file appeared before save: {png_path}")

    plt.tight_layout()
    plt.savefig(str(png_path), dpi=100, bbox_inches="tight")
    plt.close(fig)

    if not png_path.exists():
        return (None, f"PNG save failed: {png_path}")

    return (png_path, "")


# ── HTML gallery builder ──────────────────────────────────────────────────────
def _build_html_gallery(
    target_rows: List[Dict],
    output_root: Path,
    output_tag: str,
) -> Path:
    """
    HTML gallery를 생성한다.
    각 PNG는 expected_png_filename 기반 상대경로 사용 (png_index 기반 매핑 금지).
    HTML relative path와 expected_png_filename 일치 검증.
    """
    out_html = output_root / f"{output_tag}_gallery.html"
    if out_html.exists():
        sys.exit(f"[ERROR] HTML gallery already exists: {out_html}")

    rows_html = []
    for row in target_rows:
        expected_fname = str(row["expected_png_filename"])
        actual_fname = str(row.get("actual_png_filename", expected_fname))
        rel_path = f"panels/{actual_fname}"

        # HTML relative path와 expected_png_filename 일치 검증
        stored_rel = row.get("html_relative_png_path", rel_path)
        if stored_rel and stored_rel != rel_path:
            sys.exit(
                f"[ERROR] html_relative_png_path mismatch for "
                f"target_order={row['target_order']}: "
                f"{stored_rel!r} != {rel_path!r}"
            )

        err = str(row.get("error_reason", ""))
        err_note = f"<p style='color:red'>ERROR: {err}</p>" if err else ""
        rows_html.append(
            f'<div class="target">'
            f"<p>order={row['target_order']} | {row['patient_id']} | z={row['local_z']}</p>"
            f'<img src="{rel_path}" alt="{actual_fname}" style="max-width:100%">'
            f"<p>score={row['padim_score']} | roi_ratio={row['roi_0_0_patch_ratio']}</p>"
            f"{err_note}"
            f"</div>"
        )

    html_content = (
        "<!DOCTYPE html><html><head>"
        f"<title>Phase 5.95 Visual Pack - {output_tag}</title>"
        "<style>body{font-family:monospace;}"
        " .target{margin:20px;border:1px solid #ccc;padding:10px;}</style>"
        "</head><body>"
        "<h1>Phase 5.95 Expanded Candidate Visual Pack</h1>"
        f"<p>output_tag: {output_tag}</p>"
        f"<p>n_targets: {len(target_rows)}</p>"
        "<hr>"
        + "\n".join(rows_html)
        + "</body></html>"
    )
    with open(out_html, "w", encoding="utf-8") as fp:
        fp.write(html_content)
    return out_html


# ── review manifest CSV builder ───────────────────────────────────────────────
def _build_review_manifest_csv(
    target_rows: List[Dict],
    output_root: Path,
    output_tag: str,
) -> Path:
    """
    review manifest CSV를 생성한다.
    png_path 파일명과 expected_png_filename 일치 검증.
    target_order 중복 0건 확인. png filename 중복 0건 확인.
    """
    out_csv = output_root / f"{output_tag}_review_manifest.csv"
    if out_csv.exists():
        sys.exit(f"[ERROR] Review manifest CSV already exists: {out_csv}")

    seen_orders: set = set()
    seen_fnames: set = set()
    for row in target_rows:
        order = str(row["target_order"])
        if order in seen_orders:
            sys.exit(f"[ERROR] Duplicate target_order in review manifest: {order!r}")
        seen_orders.add(order)

        png_path_val = str(row.get("png_path", ""))
        if png_path_val:
            png_fname = Path(png_path_val).name
            expected_fname = str(row["expected_png_filename"])
            if png_fname != expected_fname:
                sys.exit(
                    f"[ERROR] review manifest png_path filename mismatch: "
                    f"{png_fname!r} != {expected_fname!r}"
                )
            if png_fname in seen_fnames:
                sys.exit(f"[ERROR] Duplicate PNG filename in review manifest: {png_fname!r}")
            seen_fnames.add(png_fname)

    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=REVIEW_MANIFEST_COLUMNS)
        writer.writeheader()
        for row in target_rows:
            writer.writerow({k: row.get(k, "") for k in REVIEW_MANIFEST_COLUMNS})

    return out_csv


# ── summary JSON builder ──────────────────────────────────────────────────────
def _build_summary_json(
    target_rows: List[Dict],
    args,
    output_root: Path,
    output_tag: str,
    n_errors: int,
) -> Path:
    """summary JSON을 생성한다. safety notes 포함."""
    out_json = output_root / f"{output_tag}_summary.json"
    if out_json.exists():
        sys.exit(f"[ERROR] Summary JSON already exists: {out_json}")

    mode = "smoke" if args.smoke else "run"
    summary = {
        "output_tag": output_tag,
        "mode": mode,
        "n_targets_requested": args.max_targets,
        "n_targets_processed": len(target_rows) - n_errors,
        "n_errors": n_errors,
        "safety_notes": SAFETY_NOTES,
        "limitations": [
            "sample_local_not_global: padim_score is local percentile within 3 patients only",
            "threshold_not_finalized: using p99 sample-local threshold",
            "candidate_class_hint=unknown: visual review required before any labeling",
            "roi_0_0_patch_ratio_not_confirmed_as_pure_lung",
            "not_hard_negative_final: visual review status pending",
        ],
        "targets": [
            {
                "target_order": str(r["target_order"]),
                "patient_id": str(r["patient_id"]),
                "local_z": r["local_z"],
                "expected_png_filename": str(r["expected_png_filename"]),
                "actual_png_filename": str(r.get("actual_png_filename", "")),
                "error_reason": str(r.get("error_reason", "")),
            }
            for r in target_rows
        ],
    }
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=True)
    return out_json


# ── smoke/run 공통 처리 루프 ──────────────────────────────────────────────────
def _process_targets(
    args,
    df,
    safe_id_map: Dict[str, str],
    output_root: Path,
    max_t: int,
    mode_label: str,
) -> None:
    """
    smoke/run 공통: output root 생성 → one-patient-at-a-time mmap 로드 → PNG 렌더링.
    """
    import numpy as np  # 지연 import

    _validate_target_manifest(df)

    if output_root.exists():
        sys.exit(f"[ERROR] Output root already exists: {output_root}")

    volume_root = Path(args.volume_root)
    _guard_path(volume_root)
    patient_paths = _validate_source_paths(
        df, safe_id_map, volume_root,
        args.ct_filename, args.roi_filename, args.meta_filename,
    )
    _validate_all_coordinates(df, patient_paths)

    panels_dir = output_root / "panels"
    panels_dir.mkdir(parents=True, exist_ok=False)

    target_rows: List[Dict] = []
    n_processed = 0
    n_errors = 0

    # one-patient-at-a-time (sorted for reproducibility)
    for pid in sorted(EXPECTED_PATIENTS):
        if n_processed >= max_t:
            break
        pid_df = df[df["patient_id"] == pid]
        ct_path, roi_path, meta_path = patient_paths[pid]
        _guard_path(ct_path)
        _guard_path(roi_path)

        print(f"[{mode_label}] Loading patient {pid} volume (mmap_mode='r') ...")
        ct_vol = np.load(str(ct_path), mmap_mode="r")
        roi_vol = np.load(str(roi_path), mmap_mode="r") if roi_path.exists() else None

        for _, row in pid_df.iterrows():
            if n_processed >= max_t:
                break

            row_dict = row.to_dict()
            png_path, error_reason = _render_target_panel(
                row_dict, ct_vol, roi_vol, output_root, args.output_tag,
            )

            computed_fname = _make_png_filename(
                str(row["target_order"]).zfill(3),
                str(row["patient_id"]),
                int(row["local_z"]),
                int(row["y0"]),
                int(row["x0"]),
            )
            actual_fname = computed_fname if png_path else ""
            rel_path = f"panels/{actual_fname}" if actual_fname else ""

            row_dict.update({
                "actual_png_filename": actual_fname,
                "png_path": str(png_path) if png_path else "",
                "html_relative_png_path": rel_path,
                "user_label": "",
                "user_note": "",
                "error_reason": error_reason,
                "limitation": str(row.get("limitation", "")),
            })
            target_rows.append(row_dict)

            if error_reason:
                n_errors += 1
                print(f"[{mode_label}] ERROR target_order={row['target_order']}: {error_reason}")
            else:
                n_processed += 1
                print(f"[{mode_label}] OK target_order={row['target_order']}: {png_path.name}")

    _build_html_gallery(target_rows, output_root, args.output_tag)
    _build_review_manifest_csv(target_rows, output_root, args.output_tag)
    _build_summary_json(target_rows, args, output_root, args.output_tag, n_errors)

    print(f"[{mode_label}] Done. processed={n_processed}, errors={n_errors}")
    print(f"[{mode_label}] Output: {output_root}")


# ── preflight-only 모드 ────────────────────────────────────────────────────────
def _run_preflight_only(args, df, safe_id_map: Dict[str, str]) -> None:
    """
    --preflight-only:
    target CSV row 수/컬럼 확인, expected PNG 15개 산출, CT/ROI/meta stat-only 확인,
    meta.json JSON 읽기(shape 확인). npy 로드 금지. output root 생성 금지.
    """
    print("[PREFLIGHT] Starting preflight-only validation ...")
    _validate_target_manifest(df)

    fnames = [
        _make_png_filename(
            str(row["target_order"]).zfill(3),
            str(row["patient_id"]),
            int(row["local_z"]),
            int(row["y0"]),
            int(row["x0"]),
        )
        for _, row in df.iterrows()
    ]
    assert len(fnames) == 15, f"[ERROR] Expected 15 filenames, got {len(fnames)}"
    assert len(set(fnames)) == 15, "[ERROR] Duplicate computed PNG filenames."
    print(f"[PREFLIGHT] 15 expected PNG filenames computed (unique: {len(set(fnames))})")

    volume_root = Path(args.volume_root)
    _guard_path(volume_root)
    patient_paths = _validate_source_paths(
        df, safe_id_map, volume_root,
        args.ct_filename, args.roi_filename, args.meta_filename,
    )
    _validate_all_coordinates(df, patient_paths)

    print("[PREFLIGHT] Done. No output written. No npy loaded. No PNG generated.")
    print("[PREFLIGHT] Output root NOT created.")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    경고: 이번 단계에서는 어떤 모드도 실행 금지.
    스크립트 작성과 py_compile만 허용됨.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5.95 Expanded Candidate Visual Pack\n"
            "WARNING: 세 모드 모두 사용자 별도 승인 전 실행 금지."
        )
    )

    # 세 모드 중 정확히 하나만 허용 (argparse mutually_exclusive_group required=True)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--preflight-only",
        dest="preflight_only",
        action="store_true",
        help="Preflight mode: validate inputs and paths only. No output written.",
    )
    mode_group.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"[사용자 승인 없이 절대 실행 금지] Smoke mode: max {MAX_SMOKE_TARGETS} targets."
        ),
    )
    mode_group.add_argument(
        "--run",
        action="store_true",
        help=(
            f"[사용자 승인 없이 절대 실행 금지] Run mode: max {MAX_RUN_TARGETS} targets."
        ),
    )

    parser.add_argument(
        "--output-tag",
        type=str,
        default="phase5_95_expanded_candidate_visual_pack_smoke_v1",
        help=(
            "Output directory tag (alphanumeric/underscore/hyphen only). "
            "For --run use: phase5_95_expanded_candidate_visual_pack_v1"
        ),
    )
    parser.add_argument(
        "--volume-root",
        type=str,
        default=None,
        help=(
            "WSL path to volume root directory containing patient subdirectories "
            "(safe_id based). Required for all modes."
        ),
    )
    parser.add_argument(
        "--ct-filename",
        type=str,
        default="ct_hu.npy",
        help="CT volume filename within patient subdirectory (default: ct_hu.npy).",
    )
    parser.add_argument(
        "--roi-filename",
        type=str,
        default="roi_0_0.npy",
        help="ROI mask filename within patient subdirectory (default: roi_0_0.npy).",
    )
    parser.add_argument(
        "--meta-filename",
        type=str,
        default="meta.json",
        help="Meta JSON filename within patient subdirectory (default: meta.json).",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help=(
            f"Maximum targets to process. "
            f"smoke default={MAX_SMOKE_TARGETS}, run default={MAX_RUN_TARGETS}."
        ),
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

    # smoke: --max-targets > MAX_SMOKE_TARGETS 금지
    if args.smoke:
        if args.max_targets is None:
            args.max_targets = MAX_SMOKE_TARGETS
        if args.max_targets > MAX_SMOKE_TARGETS:
            sys.exit(
                f"[ERROR] --smoke --max-targets must be <= {MAX_SMOKE_TARGETS}, "
                f"got {args.max_targets}"
            )

    # run: --max-targets > MAX_RUN_TARGETS 금지
    if args.run:
        if args.max_targets is None:
            args.max_targets = MAX_RUN_TARGETS
        if args.max_targets > MAX_RUN_TARGETS:
            sys.exit(
                f"[ERROR] --run --max-targets must be <= {MAX_RUN_TARGETS}, "
                f"got {args.max_targets}"
            )

    # preflight-only에서는 max_targets 미사용
    if args.preflight_only:
        args.max_targets = 0

    # --volume-root 필수 (모든 모드)
    if args.volume_root is None:
        sys.exit(
            "[ERROR] --volume-root is required. "
            "Provide WSL path to the volume root directory (safe_id subdirectories)."
        )

    # output_tag 검증
    _validate_output_tag(args.output_tag)

    # output_root 구성
    output_root = _OUTPUT_BASE / args.output_tag

    # path guards
    for guard_path in (
        _TARGET_CSV_PATH,
        _TARGET_JSON_PATH,
        _OUTPUT_BASE,
        output_root,
        Path(args.volume_root),
    ):
        _guard_path(guard_path)

    # 입력 파일 존재 확인
    if not _TARGET_CSV_PATH.exists():
        sys.exit(f"[ERROR] Target CSV not found: {_TARGET_CSV_PATH}")
    if not _TARGET_JSON_PATH.exists():
        sys.exit(f"[ERROR] Target JSON not found: {_TARGET_JSON_PATH}")
    if not _SPLIT_CSV_PATH.exists():
        sys.exit(f"[ERROR] Split CSV not found: {_SPLIT_CSV_PATH}")

    import pandas as pd  # 지연 import

    df = pd.read_csv(_TARGET_CSV_PATH, dtype={"target_order": str})
    safe_id_map = _load_safe_id_map(_SPLIT_CSV_PATH)

    # 실행 모드 분기
    if args.preflight_only:
        _run_preflight_only(args, df, safe_id_map)
    elif args.smoke:
        print("[SMOKE] WARNING: 사용자 승인 없이 절대 실행 금지.")
        _process_targets(args, df, safe_id_map, output_root, args.max_targets, "SMOKE")
    else:
        # args.run == True
        print("[RUN] WARNING: 사용자 승인 없이 절대 실행 금지.")
        _process_targets(args, df, safe_id_map, output_root, args.max_targets, "RUN")


if __name__ == "__main__":
    # 경고: main()은 이 가드 안에서만 호출됨.
    # import 또는 py_compile 시 main()이 실행되지 않음.
    main()
