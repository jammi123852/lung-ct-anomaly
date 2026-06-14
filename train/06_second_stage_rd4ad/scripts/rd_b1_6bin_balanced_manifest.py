#!/usr/bin/env python3
"""
RD-B1: 6-bin balanced normal coordinate manifest generation.
- crop 저장 없음. 좌표 manifest만 생성.
- stage2_holdout 접근 없음.
- GPU 사용 없음. 학습/scoring/모델 forward 없음.
"""
# ── 안전 guard ────────────────────────────────────────────────────────────────
ALLOW_REAL_PROCESSING = False   # --real 플래그로만 True로 바뀜

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
NORMAL_MANIFEST = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/normal_sampling"
    / "normal_patch_index_position_balanced_fixed96_v1"
    / "normal_sampling_manifest_normal_patch_index_position_balanced_fixed96_v1.csv"
)
STAGE2_HOLDOUT_CSV = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/datasets"
    / "s6a_stage2_holdout_filtered_manifest_v1.csv"
)
V4_20_ROI_ROOT = (
    PROJECT_ROOT
    / "outputs/mip-postprocess-research-v1/masks"
    / "refined_roi_v4_20_modeB_all_v1/normal"
)
PATCH_INDEX_LOCAL = (
    PROJECT_ROOT / "data/normal_training_ready/patch_index_by_patient"
)
PATCH_INDEX_REMOTE = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/Normal_LUNA16_padim_training_ready_v2_tslungguard_nochest"
    "/patch_index_by_patient"
)
META_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1"
    "/volumes_npy"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b1_6bin_balanced_manifest_preflight_v1"
)

# ── 파라미터 ──────────────────────────────────────────────────────────────────
EROSION_PX = 5
BOUNDARY_THRESHOLD = 0.05
INTERIOR_ROI_MIN = 0.85
CAP_PER_BIN = 50
SAMPLING_SEED = 42
CROP_SIZE = 96
Z_LOWER_MAX = 1.0 / 3.0
Z_MIDDLE_MAX = 2.0 / 3.0
SIX_BINS = [
    "upper_boundary", "upper_interior",
    "middle_boundary", "middle_interior",
    "lower_boundary", "lower_interior",
]
# crop 저장 관련 메타데이터 (manifest에 기록용)
CANDIDATE_INPUT_A = "baseline_2p5d_3ch: z-1, z, z+1"
CANDIDATE_INPUT_B = "mip_context_3ch: lower_3mm_MIP, center_3mm_MIP, upper_3mm_MIP"
CANDIDATE_INPUT_C = "mixed_3ch: CT_center, lower_MIP, upper_MIP"
NORMALIZATION_CANDIDATE = "HU clip [-1000,600] → (x+1000)/1600 → [0,1]"
OLD_AE_NORMALIZATION = "lung[-1350,150]+mediastinal[-160,240] → [0,1]"


# ── 안전 체크 ─────────────────────────────────────────────────────────────────
def assert_not_holdout(patient_id: str, holdout_set: set) -> None:
    if patient_id in holdout_set:
        raise RuntimeError(f"[SAFETY] stage2_holdout 환자 접근 차단: {patient_id}")


def assert_output_root_not_exists() -> None:
    if OUTPUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재: {OUTPUT_ROOT}")
        print("[ABORT] 기존 파일 덮어쓰기 금지. 중단합니다.")
        sys.exit(1)


def assert_no_forbidden_path(path: Path) -> None:
    pstr = str(path).lower()
    for forbidden in ["stage2_holdout", "/lesion/"]:
        if forbidden in pstr:
            raise RuntimeError(f"[SAFETY] 금지 경로 접근 차단: {path}")


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────
def z_level_from_ratio(z_ratio: float) -> str:
    if z_ratio < Z_LOWER_MAX:
        return "lower"
    elif z_ratio < Z_MIDDLE_MAX:
        return "middle"
    return "upper"


def get_patch_index_path(safe_id: str) -> Path:
    local = PATCH_INDEX_LOCAL / f"{safe_id}.csv"
    if local.exists():
        return local
    remote = PATCH_INDEX_REMOTE / f"{safe_id}.csv"
    if remote.exists():
        return remote
    return None


def load_patches(safe_id: str) -> pd.DataFrame:
    path = get_patch_index_path(safe_id)
    if path is None:
        raise FileNotFoundError(f"patch_index not found for {safe_id}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["source_patch_index"] = df.index
    return df


def load_roi(safe_id: str) -> np.ndarray:
    roi_path = V4_20_ROI_ROOT / safe_id / "refined_roi.npy"
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI not found: {roi_path}")
    assert_no_forbidden_path(roi_path)
    return np.load(roi_path)


def _make_integral(arr: np.ndarray) -> np.ndarray:
    return arr.cumsum(axis=0).cumsum(axis=1)


def _batch_rect_sum(
    integral: np.ndarray,
    y0: np.ndarray,
    x0: np.ndarray,
    size: int,
    H: int,
    W: int,
) -> np.ndarray:
    y1 = np.clip(y0 + size, 0, H)
    x1 = np.clip(x0 + size, 0, W)
    y0c = np.clip(y0, 0, H)
    x0c = np.clip(x0, 0, W)

    def safe_get(iy, ix):
        iy = np.clip(iy - 1, 0, H - 1)
        ix = np.clip(ix - 1, 0, W - 1)
        return integral[iy, ix]

    A = safe_get(y0c, x0c)
    B = safe_get(y0c, x1)
    C = safe_get(y1, x0c)
    D = integral[np.clip(y1 - 1, 0, H - 1), np.clip(x1 - 1, 0, W - 1)]
    return np.maximum(D - B - C + A, 0).astype(np.float32)


def stratified_z_round_robin(
    candidates: pd.DataFrame, cap: int, rng: np.random.Generator
) -> pd.DataFrame:
    """z 기준 round-robin sampling. 각 unique z에서 고르게 선발."""
    if len(candidates) <= cap:
        return candidates.copy()

    # z별 shuffle
    shuffled = candidates.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    z_groups = {z: grp.index.tolist() for z, grp in shuffled.groupby("local_z", sort=True)}
    z_vals = sorted(z_groups.keys())

    selected_idx = []
    pointers = {z: 0 for z in z_vals}
    z_active = list(z_vals)

    i = 0
    while len(selected_idx) < cap and z_active:
        z = z_active[i % len(z_active)]
        ptr = pointers[z]
        if ptr < len(z_groups[z]):
            selected_idx.append(z_groups[z][ptr])
            pointers[z] += 1
            i += 1
        else:
            z_active.remove(z)
            if z_active:
                i = i % len(z_active)

    return shuffled.loc[selected_idx].copy()


# ── 환자 처리 ─────────────────────────────────────────────────────────────────
def process_patient(
    patient_id: str,
    safe_id: str,
    holdout_set: set,
    rng: np.random.Generator,
    dry_run: bool = True,
) -> dict:
    """
    한 환자에 대해 6-bin 좌표 sampling을 수행한다.
    Returns: {
        'rows': list of dict (manifest rows),
        'patient_summary': dict,
        'shortage_rows': list of dict,
        'errors': list of str,
    }
    """
    assert_not_holdout(patient_id, holdout_set)

    result = {"rows": [], "patient_summary": {}, "shortage_rows": [], "errors": []}

    # ROI 로드
    try:
        roi = load_roi(safe_id)
    except FileNotFoundError as e:
        result["errors"].append(f"{patient_id}: ROI not found ({e})")
        return result

    n_slices, H, W = roi.shape
    patch_area = CROP_SIZE * CROP_SIZE

    # patch_index 로드
    try:
        patches = load_patches(safe_id)
    except FileNotFoundError as e:
        result["errors"].append(f"{patient_id}: patch_index not found ({e})")
        return result

    # z별 계산
    z_arr = patches["local_z"].values.astype(np.int32)
    z_ratio_arr = patches["z_ratio"].values
    y0_arr = patches["y0"].values.astype(np.int32)
    x0_arr = patches["x0"].values.astype(np.int32)

    # 각 patch에 대한 결과 저장
    boundary_overlap_ratios = np.zeros(len(patches), dtype=np.float32)
    roi_ratios = np.zeros(len(patches), dtype=np.float32)
    roi_boundary_dist_min = np.full(len(patches), np.nan, dtype=np.float32)

    for z_idx in np.unique(z_arr):
        if z_idx < 0 or z_idx >= n_slices:
            continue
        p_mask = z_arr == z_idx
        y0 = y0_arr[p_mask]
        x0 = x0_arr[p_mask]

        roi_slice = roi[z_idx]
        dist = distance_transform_edt(roi_slice)

        # ROI ratio
        roi_integral = _make_integral(roi_slice.astype(np.float32))
        roi_sums = _batch_rect_sum(roi_integral, y0, x0, CROP_SIZE, H, W)
        roi_ratios[p_mask] = roi_sums / patch_area

        # boundary ring ratio
        ring = ((roi_slice > 0) & (dist <= EROSION_PX)).astype(np.float32)
        ring_integral = _make_integral(ring)
        ring_sums = _batch_rect_sum(ring_integral, y0, x0, CROP_SIZE, H, W)
        boundary_overlap_ratios[p_mask] = ring_sums / patch_area

        # roi_boundary_distance_min: patch 내 dist 값의 최소값 (ROI 내 픽셀만)
        dist_integral = _make_integral(dist.astype(np.float32))
        # dist min은 integral로 계산 불가 → 근사: ring_sum > 0이면 boundary이므로 dist_min 근사
        # 대신 dist mean을 활용: 패치 내 ROI 픽셀 dist 평균으로 근사
        # 정확한 min은 루프가 필요해 성능에 영향, 여기서는 mean dist로 대체
        # 단 경계 여부 판정은 ring_ratio로만 함
        dist_sum_in_roi = _batch_rect_sum(
            _make_integral((dist * roi_slice).astype(np.float32)),
            y0, x0, CROP_SIZE, H, W
        )
        roi_pix = np.maximum(roi_sums, 1e-6)
        # mean dist in roi pixels: dist_sum_in_roi / roi_pix_count
        roi_pix_count = roi_sums.copy()
        roi_pix_count[roi_pix_count < 1] = 1
        mean_dist = dist_sum_in_roi / roi_pix_count
        roi_boundary_dist_min[p_mask] = mean_dist.astype(np.float32)

    # 6-bin label 할당
    is_boundary = boundary_overlap_ratios >= BOUNDARY_THRESHOLD
    is_interior = (roi_ratios >= INTERIOR_ROI_MIN) & (~is_boundary)

    z_level_arr = np.where(
        z_ratio_arr < Z_LOWER_MAX, "lower",
        np.where(z_ratio_arr < Z_MIDDLE_MAX, "middle", "upper")
    )
    six_bin_labels = np.where(
        is_boundary,
        np.char.add(z_level_arr, "_boundary"),
        np.where(
            is_interior,
            np.char.add(z_level_arr, "_interior"),
            "excluded"
        )
    )

    # patches에 계산값 추가
    patches = patches.copy()
    patches["six_bin_label"] = six_bin_labels
    patches["boundary_overlap_ratio"] = boundary_overlap_ratios
    patches["refined_roi_ratio"] = roi_ratios
    patches["roi_boundary_distance_min"] = roi_boundary_dist_min
    patches["boundary_status"] = np.where(is_boundary, "boundary", np.where(is_interior, "interior", "excluded"))

    # z_level 추가 (문자열)
    patches["z_level_calc"] = z_level_arr

    # 6-bin별 sampling
    bin_selected = {}
    selected_rows = []
    shortage_rows = []

    patient_rng = np.random.default_rng(SAMPLING_SEED + hash(safe_id) % (2**20))

    per_bin_counts = {}
    zero_bin_labels = []

    for bin_label in SIX_BINS:
        bin_candidates = patches[patches["six_bin_label"] == bin_label].copy()
        n_avail = len(bin_candidates)
        per_bin_counts[bin_label] = {"available": n_avail}

        if n_avail == 0:
            per_bin_counts[bin_label]["selected"] = 0
            zero_bin_labels.append(bin_label)
            continue

        selected_df = stratified_z_round_robin(bin_candidates, CAP_PER_BIN, patient_rng)
        n_sel = len(selected_df)
        per_bin_counts[bin_label]["selected"] = n_sel

        if n_avail < CAP_PER_BIN:
            shortage_rows.append({
                "patient_id": patient_id,
                "missing_or_short_bin": bin_label,
                "available_count": n_avail,
                "selected_count": n_sel,
                "shortage_from_cap50": CAP_PER_BIN - n_sel,
                "note": "partial" if n_avail > 0 else "zero_bin",
            })

        for _, row in selected_df.iterrows():
            # z_level에서 manifest row 생성
            zl = row.get("z_level", row.get("z_level_calc", "unknown"))
            selected_rows.append({
                "patient_id": patient_id,
                "safe_id": safe_id,
                "split": "train",
                "local_z": int(row["local_z"]),
                "slice_index": int(row["local_z"]),
                "z_ratio": float(row["z_ratio"]),
                "z_level": row.get("z_level_calc", zl),
                "boundary_status": row["boundary_status"],
                "six_bin_label": bin_label,
                "crop_y0": int(row["y0"]),
                "crop_x0": int(row["x0"]),
                "crop_y1": int(row["y0"]) + CROP_SIZE,
                "crop_x1": int(row["x0"]) + CROP_SIZE,
                "crop_size": CROP_SIZE,
                "source_patch_index": int(row["source_patch_index"]),
                "refined_roi_ratio": float(row["refined_roi_ratio"]),
                "boundary_overlap_ratio": float(row["boundary_overlap_ratio"]),
                "roi_boundary_distance_min": float(row["roi_boundary_distance_min"]),
                "pure_lung_patch_ratio": float(row["pure_lung_patch_ratio"]) if "pure_lung_patch_ratio" in row else None,
                "position_bin_old": row.get("position_bin", None),
                "selected_by": "z_stratified_round_robin",
                "sampling_seed": SAMPLING_SEED,
                "note": "",
            })

    result["rows"] = selected_rows
    result["shortage_rows"] = shortage_rows

    # patient summary
    n_selected_total = sum(per_bin_counts[b].get("selected", 0) for b in SIX_BINS)
    bin_sels = {b: per_bin_counts[b].get("selected", 0) for b in SIX_BINS}
    full_cap = all(per_bin_counts[b].get("selected", 0) >= CAP_PER_BIN for b in SIX_BINS)
    partial = (not full_cap) and (n_selected_total > 0)
    zero_count = len(zero_bin_labels)

    result["patient_summary"] = {
        "patient_id": patient_id,
        "safe_id": safe_id,
        "n_selected_total": n_selected_total,
        "upper_boundary_selected": bin_sels["upper_boundary"],
        "upper_interior_selected": bin_sels["upper_interior"],
        "middle_boundary_selected": bin_sels["middle_boundary"],
        "middle_interior_selected": bin_sels["middle_interior"],
        "lower_boundary_selected": bin_sels["lower_boundary"],
        "lower_interior_selected": bin_sels["lower_interior"],
        "full_cap_all_bins": full_cap,
        "partial_flag": partial,
        "zero_bin_count": zero_count,
        "missing_bin_labels": ";".join(zero_bin_labels) if zero_bin_labels else "",
        "min_selected_bin": min(bin_sels.values()),
        "max_selected_bin": max(bin_sels.values()),
        "note": "",
    }
    return result


# ── val/test count summary ────────────────────────────────────────────────────
def compute_split_count_summary(manifest_df: pd.DataFrame, split: str) -> dict:
    split_df = manifest_df[manifest_df["normal_split"] == split].drop_duplicates("patient_id")
    return {
        "split": split,
        "n_patients": len(split_df),
        "patient_ids": list(split_df["patient_id"].values),
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    global ALLOW_REAL_PROCESSING

    parser = argparse.ArgumentParser(description="RD-B1 6-bin balanced normal coordinate manifest")
    parser.add_argument("--real", action="store_true", help="실제 실행 (기본: dry-run)")
    parser.add_argument("--n-dry", type=int, default=5, help="dry-run 환자 수 (기본: 5)")
    parser.add_argument("--plan-only", action="store_true", help="계획만 출력, 아무 파일도 생성 안 함")
    args = parser.parse_args()

    if args.real:
        ALLOW_REAL_PROCESSING = True

    is_dry_run = not args.real
    is_plan_only = args.plan_only

    print("=" * 60)
    print("RD-B1 6-bin balanced normal coordinate manifest")
    print(f"  mode: {'plan-only' if is_plan_only else 'dry-run' if is_dry_run else 'REAL'}")
    print(f"  ALLOW_REAL_PROCESSING: {ALLOW_REAL_PROCESSING}")
    print("=" * 60)

    # ── output root 존재 여부 확인 (real에서만 생성) ───────────────────────────
    if not is_plan_only:
        assert_output_root_not_exists()

    # ── stage2_holdout 로드 ────────────────────────────────────────────────────
    holdout_df = pd.read_csv(STAGE2_HOLDOUT_CSV)
    holdout_set = set(holdout_df["patient_id"].dropna().unique())
    print(f"[INFO] stage2_holdout 고유 환자 수: {len(holdout_set)}")

    # ── normal manifest 로드 ──────────────────────────────────────────────────
    manifest = pd.read_csv(NORMAL_MANIFEST)
    train_df = manifest[manifest["normal_split"] == "train"].drop_duplicates("patient_id")
    all_train_patients = list(zip(train_df["patient_id"].values, train_df["safe_id"].values))
    print(f"[INFO] normal_train 환자 수: {len(all_train_patients)}")

    # ── 교집합 체크 ────────────────────────────────────────────────────────────
    cross = set(p[0] for p in all_train_patients) & holdout_set
    if cross:
        print(f"[SAFETY-FAIL] normal_train ∩ holdout 교집합 발견: {cross}", file=sys.stderr)
        sys.exit(1)
    print("[SAFETY] normal_train ∩ stage2_holdout = 0 ✓")

    # ── val/test count summary ────────────────────────────────────────────────
    val_summary = compute_split_count_summary(manifest, "val")
    test_summary = compute_split_count_summary(manifest, "test")
    print(f"[INFO] val 환자 수: {val_summary['n_patients']} (manifest 미생성)")
    print(f"[INFO] test 환자 수: {test_summary['n_patients']} (manifest 미생성)")

    # ── patch_index 290명 전부 존재 확인 ─────────────────────────────────────
    missing_patch = []
    for pid, sid in all_train_patients:
        if get_patch_index_path(sid) is None:
            missing_patch.append(sid)
    if missing_patch:
        print(f"[WARN] patch_index 없는 환자 {len(missing_patch)}명: {missing_patch[:5]}")
    else:
        print(f"[SAFETY] patch_index 290명 전부 존재 ✓")

    # ── ROI 290명 존재 확인 ────────────────────────────────────────────────────
    missing_roi = []
    for pid, sid in all_train_patients:
        if not (V4_20_ROI_ROOT / sid / "refined_roi.npy").exists():
            missing_roi.append(sid)
    if missing_roi:
        print(f"[WARN] ROI 없는 환자 {len(missing_roi)}명: {missing_roi[:5]}")
    else:
        print(f"[SAFETY] ROI 290명 전부 존재 ✓")

    # ── plan-only 종료 ─────────────────────────────────────────────────────────
    if is_plan_only:
        print()
        print("[PLAN-ONLY] 파일 생성 없이 종료.")
        print(f"  target_cap_per_bin = {CAP_PER_BIN}")
        print(f"  target_total_if_full = {len(all_train_patients)} × 6 × {CAP_PER_BIN} = {len(all_train_patients)*6*CAP_PER_BIN}")
        print(f"  sampling_seed = {SAMPLING_SEED}")
        print(f"  output_root = {OUTPUT_ROOT}")
        print(f"  erosion_px = {EROSION_PX}, boundary_threshold = {BOUNDARY_THRESHOLD}")
        return

    # ── dry-run guard ──────────────────────────────────────────────────────────
    if is_dry_run:
        patients_to_process = all_train_patients[: args.n_dry]
        print(f"[DRY-RUN] {len(patients_to_process)}명만 처리 (전체 {len(all_train_patients)}명)")
        print("[DRY-RUN] 파일 생성 없음.")
    else:
        if not ALLOW_REAL_PROCESSING:
            print("[ABORT] ALLOW_REAL_PROCESSING=False. --real 플래그를 확인하세요.", file=sys.stderr)
            sys.exit(1)
        patients_to_process = all_train_patients
        print(f"[REAL] {len(patients_to_process)}명 전체 처리 시작.")
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    # ── 환자별 처리 ────────────────────────────────────────────────────────────
    all_manifest_rows = []
    all_patient_summaries = []
    all_shortage_rows = []
    all_errors = []

    rng = np.random.default_rng(SAMPLING_SEED)
    t_start = time.time()

    for i, (patient_id, safe_id) in enumerate(patients_to_process):
        t0 = time.time()
        try:
            res = process_patient(patient_id, safe_id, holdout_set, rng, dry_run=is_dry_run)
            all_manifest_rows.extend(res["rows"])
            all_patient_summaries.append(res["patient_summary"])
            all_shortage_rows.extend(res["shortage_rows"])
            all_errors.extend(res["errors"])
        except Exception as e:
            all_errors.append(f"{patient_id}: {e}")
            all_patient_summaries.append({
                "patient_id": patient_id,
                "safe_id": safe_id,
                "n_selected_total": 0,
                "upper_boundary_selected": 0, "upper_interior_selected": 0,
                "middle_boundary_selected": 0, "middle_interior_selected": 0,
                "lower_boundary_selected": 0, "lower_interior_selected": 0,
                "full_cap_all_bins": False, "partial_flag": False, "zero_bin_count": 6,
                "missing_bin_labels": ";".join(SIX_BINS), "min_selected_bin": 0, "max_selected_bin": 0,
                "note": f"ERROR: {e}",
            })
        elapsed = time.time() - t0
        if (i + 1) % 10 == 0 or (i + 1) == len(patients_to_process):
            total_sel = sum(s.get("n_selected_total", 0) for s in all_patient_summaries)
            print(f"  [{i+1}/{len(patients_to_process)}] {patient_id}: "
                  f"sel={all_patient_summaries[-1].get('n_selected_total',0)}, "
                  f"total_so_far={total_sel}, elapsed={elapsed:.1f}s")

    total_elapsed = time.time() - t_start

    # ── 결과 집계 ─────────────────────────────────────────────────────────────
    manifest_df = pd.DataFrame(all_manifest_rows)
    if len(manifest_df) > 0:
        manifest_df.insert(0, "manifest_id", [f"rd_b1_{i:07d}" for i in range(len(manifest_df))])

    patient_sum_df = pd.DataFrame(all_patient_summaries)
    shortage_df = pd.DataFrame(all_shortage_rows) if all_shortage_rows else pd.DataFrame(columns=[
        "patient_id", "missing_or_short_bin", "available_count",
        "selected_count", "shortage_from_cap50", "note"
    ])
    errors_df = pd.DataFrame({"error": all_errors}) if all_errors else pd.DataFrame(columns=["error"])

    # bin summary
    bin_summary_rows = []
    for bin_label in SIX_BINS:
        bin_patients = [s for s in all_patient_summaries if s.get(f"{bin_label.split('_')[0]}_{bin_label.split('_')[1]}_selected", None) is not None]
        sel_key = f"{'_'.join(bin_label.split('_'))}_selected"
        per_patient_sel = [s.get(sel_key, 0) for s in all_patient_summaries]
        avail_total = sum(
            row.get("available_count", 0)
            for row in all_shortage_rows
            if row["missing_or_short_bin"] == bin_label
        )
        sel_total = sum(per_patient_sel)
        n_with = sum(1 for v in per_patient_sel if v > 0)
        n_full = sum(1 for v in per_patient_sel if v >= CAP_PER_BIN)
        n_partial = sum(1 for v in per_patient_sel if 0 < v < CAP_PER_BIN)
        n_zero = sum(1 for v in per_patient_sel if v == 0)
        bin_summary_rows.append({
            "six_bin_label": bin_label,
            "total_available": "n/a (see RD-B0)",
            "total_selected": sel_total,
            "n_patients_with_bin": n_with,
            "n_patients_full_cap": n_full,
            "n_patients_partial": n_partial,
            "n_patients_zero": n_zero,
            "mean_selected_per_patient": round(float(np.mean(per_patient_sel)), 2),
            "median_selected_per_patient": round(float(np.median(per_patient_sel)), 2),
            "min_selected_per_patient": int(np.min(per_patient_sel)),
            "max_selected_per_patient": int(np.max(per_patient_sel)),
        })
    bin_sum_df = pd.DataFrame(bin_summary_rows)

    # ── 집계 통계 ─────────────────────────────────────────────────────────────
    actual_selected_total = len(manifest_df)
    full_cap_count = sum(1 for s in all_patient_summaries if s.get("full_cap_all_bins", False))
    partial_count = sum(1 for s in all_patient_summaries if s.get("partial_flag", False))
    zero_bin_count = sum(1 for s in all_patient_summaries if s.get("zero_bin_count", 0) >= 6)

    print()
    print("[RESULT] ──────────────────────────────────────")
    print(f"  처리 환자 수: {len(patients_to_process)}")
    print(f"  actual_selected_total: {actual_selected_total}")
    print(f"  full_cap_patient_count: {full_cap_count}")
    print(f"  partial_patient_count: {partial_count}")
    print(f"  zero_bin_patient_count: {zero_bin_count}")
    print(f"  errors: {len(all_errors)}")
    print(f"  elapsed: {total_elapsed:.1f}s")
    if is_dry_run:
        projected = int(actual_selected_total / len(patients_to_process) * len(all_train_patients))
        print(f"  [PROJECTED total for 290명]: ~{projected}")

    # ── dry-run은 파일 저장 없이 종료 ─────────────────────────────────────────
    if is_dry_run:
        print()
        print("[DRY-RUN] 파일 저장 없이 종료.")
        if len(patient_sum_df) > 0:
            print("[DRY-RUN] patient summary preview:")
            print(patient_sum_df[["patient_id", "n_selected_total",
                                   "full_cap_all_bins", "zero_bin_count",
                                   "missing_bin_labels"]].to_string(index=False))
        return

    # ── real: 파일 저장 ────────────────────────────────────────────────────────
    print()
    print("[REAL] 파일 저장 중...")

    manifest_df.to_csv(
        OUTPUT_ROOT / "rd_b1_6bin_balanced_normal_train_coordinate_manifest.csv",
        index=False
    )
    patient_sum_df.to_csv(OUTPUT_ROOT / "rd_b1_patient_sampling_summary.csv", index=False)
    bin_sum_df.to_csv(OUTPUT_ROOT / "rd_b1_bin_sampling_summary.csv", index=False)
    shortage_df.to_csv(OUTPUT_ROOT / "rd_b1_shortage_summary.csv", index=False)
    errors_df.to_csv(OUTPUT_ROOT / "rd_b1_errors.csv", index=False)

    # summary JSON
    summary = {
        "version": "rd_b1_v1",
        "is_dry_run": False,
        "n_normal_train_patients": len(all_train_patients),
        "n_processed": len(patients_to_process),
        "target_cap_per_bin": CAP_PER_BIN,
        "target_total_if_full": len(all_train_patients) * 6 * CAP_PER_BIN,
        "actual_selected_total": actual_selected_total,
        "full_cap_patient_count": full_cap_count,
        "partial_patient_count": partial_count,
        "zero_bin_patient_count": zero_bin_count,
        "stage2_holdout_intersection": 0,
        "sampling_seed": SAMPLING_SEED,
        "erosion_px": EROSION_PX,
        "boundary_threshold": BOUNDARY_THRESHOLD,
        "interior_roi_min": INTERIOR_ROI_MIN,
        "source_files_modified": [],
        "crop_npz_generated": False,
        "training_started": False,
        "scoring_started": False,
        "all_checks_passed": len(all_errors) == 0,
        "n_errors": len(all_errors),
        "elapsed_seconds": round(total_elapsed, 1),
        "candidate_input_A": CANDIDATE_INPUT_A,
        "candidate_input_B": CANDIDATE_INPUT_B,
        "candidate_input_C": CANDIDATE_INPUT_C,
        "normalization_candidate": NORMALIZATION_CANDIDATE,
        "old_ae_normalization": OLD_AE_NORMALIZATION,
        "val_patient_count": val_summary["n_patients"],
        "test_patient_count": test_summary["n_patients"],
        "absolute_not_done": [
            "crop 생성 없음", "학습 없음", "scoring 없음",
            "stage2_holdout 접근 없음", "기존 파일 수정 없음",
            "모델 forward 없음", "GPU 사용 없음",
            "threshold 재계산 없음", "score 재계산 없음",
        ],
    }
    (OUTPUT_ROOT / "rd_b1_manifest_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    # report.md
    _write_report(OUTPUT_ROOT, summary, manifest_df, patient_sum_df, bin_sum_df, shortage_df, all_errors, total_elapsed)

    # DONE marker
    (OUTPUT_ROOT / "DONE").write_text("rd_b1 complete\n")

    print("[DONE] 모든 파일 저장 완료.")
    print(f"  출력 경로: {OUTPUT_ROOT}")


def _write_report(out_dir, summary, manifest_df, patient_sum_df, bin_sum_df, shortage_df, errors, elapsed):
    lines = []
    a = lines.append

    a("# RD-B1 6-bin Balanced Normal Coordinate Manifest – Generation Report")
    a("")
    a("## 1. RD-B0 결과 요약")
    a("- normal_train: 290명 확인 완료")
    a("- 6-bin 구조: upper/middle/lower × boundary/interior")
    a(f"- 추천 기준: erosion={summary['erosion_px']}px, boundary_threshold={summary['boundary_threshold']}")
    a(f"- cap=50 기준 충분한 환자 수: 272명 (RD-B0 확인)")
    a(f"- zero_bin 환자: 7명 (RD-B0 확인)")
    a("")
    a("## 2. RD-B1 목적")
    a("- RD-B0에서 확정한 6-bin 구조를 기반으로 실제 patch 좌표 manifest를 생성한다.")
    a("- 이번 단계에서 crop 저장, 학습, scoring은 하지 않는다.")
    a("- 좌표 manifest만 생성하여 다음 단계(RD-B2 crop visual smoke)를 준비한다.")
    a("")
    a("## 3. 6-bin Sampling 정의")
    a("- z_level: lower (z_ratio<1/3), middle (1/3≤z_ratio<2/3), upper (z_ratio≥2/3)")
    a(f"- boundary: ROI 내 distance_transform_edt ≤ {summary['erosion_px']}px 픽셀이 patch의 ≥{summary['boundary_threshold']:.0%} 차지")
    a(f"- interior: refined_roi_ratio ≥ {summary['interior_roi_min']:.0%} AND boundary_overlap < {summary['boundary_threshold']:.0%}")
    a("- excluded: 위 두 조건 모두 미해당 (manifest에 포함 안 됨)")
    a("")
    a("## 4. Cap=50 선택 이유")
    a("- RD-B0 feasibility: 290명 중 272명이 cap=50 all-bin 달성 가능")
    a("- cap=100 시 266명만 달성 가능하여 차이 크지 않으나, 학습 데이터 양은 2배")
    a("- cap=50 × 6 × 290 = 87,000 좌표가 RD4AD 초기 학습에 충분한 양")
    a("")
    a("## 5. zero-bin / partial 환자 처리 방식")
    a("- zero_bin 환자(특정 bin에 후보 없음): 제외하지 않음, 가능한 bin은 그대로 사용")
    a("- empty bin은 sampling_summary에 0으로 기록")
    a("- duplicate oversampling 금지 (available < cap 이면 available 전량 사용)")
    a("")
    a("## 6. 실제 선택 결과")
    a(f"- target_total_if_full: {summary['target_total_if_full']:,}")
    a(f"- **actual_selected_total: {summary['actual_selected_total']:,}**")
    a(f"- full_cap_patient_count: {summary['full_cap_patient_count']}")
    a(f"- partial_patient_count: {summary['partial_patient_count']}")
    a(f"- zero_bin_patient_count: {summary['zero_bin_patient_count']}")
    a(f"- n_errors: {summary['n_errors']}")
    a(f"- elapsed: {elapsed:.1f}s")
    a("")
    a("## 7. bin별 균형성")
    a("")
    a("| bin | total_selected | n_full_cap | n_partial | n_zero | mean/patient |")
    a("|-----|---------------|-----------|-----------|--------|-------------|")
    for _, row in bin_sum_df.iterrows():
        a(f"| {row['six_bin_label']} | {row['total_selected']} | "
          f"{row['n_patients_full_cap']} | {row['n_patients_partial']} | "
          f"{row['n_patients_zero']} | {row['mean_selected_per_patient']} |")
    a("")
    a("## 8. Manifest 성격 명시")
    a("- 이 manifest는 **좌표 manifest**이다.")
    a("- crop 이미지/npz 파일은 저장하지 않았다.")
    a("- 각 row는 (patient_id, local_z, y0, x0, six_bin_label) 조합이다.")
    a("- 다음 단계(RD-B2)에서 CT를 읽고 실제 crop을 생성하여 시각 검증한다.")
    a("")
    a("## 9. MIP/RD4AD 입력 설계 메타데이터")
    a(f"- candidate_input_A: {summary['candidate_input_A']}")
    a(f"- candidate_input_B: {summary['candidate_input_B']}")
    a(f"- candidate_input_C: {summary['candidate_input_C']}")
    a(f"- normalization_candidate: {summary['normalization_candidate']}")
    a(f"- old_ae_normalization: {summary['old_ae_normalization']}")
    a("- (이번 단계에서는 normalization 적용 없음. RD-B2 visual smoke에서 확인.)")
    a("")
    a("## 10. 다음 단계")
    a("- **RD-B2**: crop visual smoke – CT를 읽어 소수 patch를 실제 crop하여 boundary/interior 시각 검증")
    a("- **RD-B3**: RD4AD teacher-student architecture preflight")
    a("")
    a("## 11. 절대 하지 않은 것")
    for item in summary["absolute_not_done"]:
        a(f"- {item}")
    a("")
    a(f"---")
    a(f"*Generated by rd_b1_6bin_balanced_manifest.py | sampling_seed={summary['sampling_seed']}*")

    (out_dir / "rd_b1_6bin_balanced_manifest_preflight_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
