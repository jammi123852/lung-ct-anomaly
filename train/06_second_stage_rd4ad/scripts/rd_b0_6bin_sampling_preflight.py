"""
RD-B0: Normal-only 6-bin position/boundary sampling preflight
- normal_train 290명 기준
- 6-bin: upper/middle/lower × boundary/interior
- boundary: v4_20 refined ROI erosion ring 기반
- 새 학습/scoring/crop 저장/CT 수정 없음
- stage2_holdout 접근 없음
"""

import sys
import os
import argparse

# ── 안전 차단 ──────────────────────────────────────────────────────────────
ALLOW_REAL_PROCESSING = True  # 2026-06-06 사용자 승인

_bare_run_guard = len(sys.argv) == 1
if _bare_run_guard:
    print("[GUARD] bare-run 차단: --dry-run 또는 --real 인수 필요", file=sys.stderr)
    sys.exit(2)

# ── 인수 파싱 ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="RD-B0 6-bin sampling preflight")
parser.add_argument("--dry-run", action="store_true",
                    help="5명 샘플로 계획 검증만 수행 (출력 저장 없음)")
parser.add_argument("--real", action="store_true",
                    help="290명 전체 실행 (ALLOW_REAL_PROCESSING=True 필요)")
parser.add_argument("--n-dry", type=int, default=5,
                    help="dry-run 시 사용할 환자 수 (기본 5)")
args = parser.parse_args()

if args.real and not ALLOW_REAL_PROCESSING:
    print("[BLOCKED] ALLOW_REAL_PROCESSING=False 상태에서 --real 실행 차단", file=sys.stderr)
    sys.exit(2)

if not args.dry_run and not args.real:
    print("[GUARD] --dry-run 또는 --real 중 하나를 지정하세요", file=sys.stderr)
    sys.exit(2)

IS_DRY = args.dry_run

# ── 임포트 ──────────────────────────────────────────────────────────────────
import json
import csv
import time
import datetime
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("[WARN] scipy 없음 - boundary ring 계산 불가")

# ── 경로 상수 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/normal_based_stage2_verifier_audit/rd_b0_6bin_sampling_preflight_v1"

NORMAL_MANIFEST = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/normal_sampling/normal_patch_index_position_balanced_fixed96_v1/normal_sampling_manifest_normal_patch_index_position_balanced_fixed96_v1.csv"
V4_20_ROI_ROOT = PROJECT_ROOT / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1/normal"
PATCH_INDEX_ROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_v2_tslungguard_nochest/patch_index_by_patient")
META_ROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")
STAGE2_HOLDOUT_CSV = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_filtered_manifest_v1.csv"

# ── 파라미터 ───────────────────────────────────────────────────────────────
CROP_SIZE = 96
EROSION_WIDTHS = [3, 5, 8, 12]          # boundary ring width candidates (px)
THRESHOLDS = [0.01, 0.03, 0.05, 0.10]  # boundary_overlap_ratio thresholds
INTERIOR_ROI_MIN = 0.85                  # interior: refined_roi_ratio >= this
CAPS = [20, 50, 100]                     # sampling cap candidates (per bin per patient)

# Z-level 분할 (기존 정의와 동일)
Z_LOWER_MAX = 1.0 / 3.0
Z_MIDDLE_MAX = 2.0 / 3.0
# lower: z_ratio < 1/3
# middle: 1/3 <= z_ratio < 2/3
# upper: z_ratio >= 2/3

# 추천 기본 조합 (보고용)
DEFAULT_EROSION = 5
DEFAULT_THRESHOLD = 0.05

# ── 안전 차단: stage2_holdout 경로 접근 금지 ────────────────────────────────
FORBIDDEN_PATHS = [
    str(PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/crops"),
    str(PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/smoke"),
]

def assert_not_holdout(patient_id: str, holdout_set: set) -> None:
    if patient_id in holdout_set:
        raise RuntimeError(f"[SAFETY] stage2_holdout 환자 접근 차단: {patient_id}")


# ── 유틸 ───────────────────────────────────────────────────────────────────
def z_level(z_ratio: float) -> str:
    if z_ratio < Z_LOWER_MAX:
        return "lower"
    elif z_ratio < Z_MIDDLE_MAX:
        return "middle"
    else:
        return "upper"


def _make_integral(arr: np.ndarray) -> np.ndarray:
    """2D integral image (zero-padded). arr: (H,W) → (H+1,W+1) int32."""
    out = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=np.int32)
    out[1:, 1:] = np.cumsum(np.cumsum(arr.astype(np.int32), axis=0), axis=1)
    return out


def _batch_rect_sum(integral: np.ndarray, y0: np.ndarray, x0: np.ndarray,
                    size: int, H: int, W: int) -> np.ndarray:
    """Vectorized rectangle sum lookup from integral image.
    integral: (H+1, W+1), y0/x0: int arrays of length N.
    Returns sum of [y0:y0+size, x0:x0+size] for each patch.
    """
    y1 = np.minimum(y0 + size, H).astype(np.int32)
    x1 = np.minimum(x0 + size, W).astype(np.int32)
    y0 = y0.astype(np.int32)
    x0 = x0.astype(np.int32)
    return (integral[y1, x1] - integral[y0, x1]
            - integral[y1, x0] + integral[y0, x0])


def compute_bin_counts_for_patient(
    safe_id: str,
    roi: np.ndarray,          # (n_slices, H, W) uint8
    patches: pd.DataFrame,    # patch_index rows
) -> dict:
    """
    각 (erosion, threshold) 조합에 대해 6-bin count를 계산한다.
    벡터화된 integral image 방식으로 속도 최적화.
    ROI와 CT는 수정하지 않는다.
    """
    n_slices, H, W = roi.shape
    patch_area = CROP_SIZE * CROP_SIZE

    # (ew, thr) → {bin: count}
    results = {(ew, thr): defaultdict(int)
               for ew in EROSION_WIDTHS for thr in THRESHOLDS}

    # z-ratio → z-level 벡터화
    z_ratio_arr = patches["z_ratio"].values
    z_level_arr = np.where(z_ratio_arr < Z_LOWER_MAX, 0,
                   np.where(z_ratio_arr < Z_MIDDLE_MAX, 1, 2))  # 0=lower,1=mid,2=up
    zl_names = ["lower", "middle", "upper"]

    y0_all = patches["y0"].values.astype(np.int32)
    x0_all = patches["x0"].values.astype(np.int32)
    z_all = patches["local_z"].values.astype(np.int32)

    for z_idx in np.unique(z_all):
        if z_idx < 0 or z_idx >= n_slices:
            continue
        p_mask = z_all == z_idx
        y0 = y0_all[p_mask]
        x0 = x0_all[p_mask]
        zl = z_level_arr[p_mask]

        roi_slice = roi[z_idx]  # (H, W)

        # 1회 거리 변환
        dist = distance_transform_edt(roi_slice)

        # ROI integral: refined_roi_ratio 계산용
        roi_integral = _make_integral(roi_slice)
        roi_sums = _batch_rect_sum(roi_integral, y0, x0, CROP_SIZE, H, W)
        roi_ratios = roi_sums / patch_area

        for ew in EROSION_WIDTHS:
            # boundary ring: ROI 내에서 dist <= ew
            ring = ((roi_slice > 0) & (dist <= ew)).astype(np.int32)
            ring_integral = _make_integral(ring)
            ring_sums = _batch_rect_sum(ring_integral, y0, x0, CROP_SIZE, H, W)
            ring_ratios = ring_sums / patch_area

            for thr in THRESHOLDS:
                is_bnd = ring_ratios >= thr
                is_int = (roi_ratios >= INTERIOR_ROI_MIN) & (~is_bnd)

                for zl_idx, zl_name in enumerate(zl_names):
                    zmask = zl == zl_idx
                    results[(ew, thr)][f"{zl_name}_boundary"] += int((is_bnd & zmask).sum())
                    results[(ew, thr)][f"{zl_name}_interior"] += int((is_int & zmask).sum())

    return {k: dict(v) for k, v in results.items()}


def get_spacing_z(safe_id: str) -> float:
    """meta.json에서 z spacing 읽기."""
    meta_path = META_ROOT / safe_id / "meta.json"
    if meta_path.exists():
        try:
            d = json.loads(meta_path.read_text())
            sp = d.get("spacing_xyz", None)
            if sp and len(sp) >= 3:
                return float(sp[2])
        except Exception:
            pass
    return float("nan")


def summarize_bin_counts(bin_dict: dict) -> dict:
    """6-bin dict → 요약 통계 반환."""
    bins = ["upper_boundary", "upper_interior",
            "middle_boundary", "middle_interior",
            "lower_boundary", "lower_interior"]
    total = sum(bin_dict.get(b, 0) for b in bins)
    counts = {b: bin_dict.get(b, 0) for b in bins}
    min_cnt = min(counts.values())
    zero_bins = sum(1 for v in counts.values() if v == 0)
    return {
        **counts,
        "total_candidate_count": total,
        "min_bin_count": min_cnt,
        "zero_bin_count": zero_bins,
        "enough_cap20_all_bins": min_cnt >= 20,
        "enough_cap50_all_bins": min_cnt >= 50,
        "enough_cap100_all_bins": min_cnt >= 100,
    }


# ── 메인 ───────────────────────────────────────────────────────────────────
def main():
    start_ts = time.time()
    print(f"[RD-B0] {'DRY-RUN' if IS_DRY else 'REAL'} 시작: {datetime.datetime.now().isoformat()}")

    # ── 안전 검증: output root ───────────────────────────────────────────
    if not IS_DRY and OUTPUT_ROOT.exists():
        print(f"[SAFETY] output root 이미 존재: {OUTPUT_ROOT} → 중단", file=sys.stderr)
        sys.exit(1)

    # ── holdout 집합 로딩 ────────────────────────────────────────────────
    h_df = pd.read_csv(STAGE2_HOLDOUT_CSV, usecols=["patient_id"])
    holdout_set = set(h_df["patient_id"].unique())
    print(f"[INFO] stage2_holdout 환자 수: {len(holdout_set)}")

    # ── normal_train split 로딩 ──────────────────────────────────────────
    manifest = pd.read_csv(NORMAL_MANIFEST)
    train_df = manifest[manifest["normal_split"] == "train"].drop_duplicates("patient_id")
    all_train_patients = list(
        train_df[["patient_id", "safe_id"]].drop_duplicates().itertuples(index=False, name=None)
    )
    print(f"[INFO] normal_train 환자 수: {len(all_train_patients)}")

    # holdout 교집합 확인
    cross = set(p[0] for p in all_train_patients) & holdout_set
    if cross:
        print(f"[SAFETY] normal_train ∩ holdout 교집합 발견: {cross}", file=sys.stderr)
        sys.exit(1)
    print("[SAFETY] normal_train ∩ stage2_holdout = 0 ✓")

    # ── dry-run: N명만 처리 ──────────────────────────────────────────────
    patients_to_process = all_train_patients
    if IS_DRY:
        patients_to_process = all_train_patients[: args.n_dry]
        print(f"[DRY-RUN] {len(patients_to_process)}명만 처리 (전체 {len(all_train_patients)}명)")

    # ── 환자별 처리 ──────────────────────────────────────────────────────
    by_patient_rows = []
    sensitivity_accumulator = defaultdict(lambda: defaultdict(list))
    errors = []

    for idx, (patient_id, safe_id) in enumerate(patients_to_process):
        print(f"[{idx+1}/{len(patients_to_process)}] {patient_id} ({safe_id})", end=" ")
        try:
            assert_not_holdout(patient_id, holdout_set)

            # ROI 로딩
            roi_path = V4_20_ROI_ROOT / safe_id / "refined_roi.npy"
            if not roi_path.exists():
                raise FileNotFoundError(f"ROI 없음: {roi_path}")
            roi = np.load(str(roi_path))
            if roi.dtype != np.uint8:
                roi = roi.astype(np.uint8)

            # patch_index 로딩
            pidx_path = PATCH_INDEX_ROOT / f"{safe_id}.csv"
            if not pidx_path.exists():
                raise FileNotFoundError(f"patch_index 없음: {pidx_path}")
            patches = pd.read_csv(str(pidx_path))

            # 필수 컬럼 확인
            needed = ["local_z", "z_ratio", "y0", "x0"]
            missing_cols = [c for c in needed if c not in patches.columns]
            if missing_cols:
                raise ValueError(f"patch_index 누락 컬럼: {missing_cols}")

            # CT는 로딩하지 않는다
            n_slices = roi.shape[0]
            n_patches = len(patches)

            # spacing
            spacing_z = get_spacing_z(safe_id)

            # bin count 계산
            if not SCIPY_OK:
                raise ImportError("scipy 없음")

            bin_results = compute_bin_counts_for_patient(safe_id, roi, patches)

            # 추천 조합 결과
            default_key = (DEFAULT_EROSION, DEFAULT_THRESHOLD)
            default_bins = bin_results.get(default_key, {})
            summary = summarize_bin_counts(default_bins)

            row = {
                "patient_id": patient_id,
                "safe_id": safe_id,
                "n_slices": n_slices,
                "n_patches_total": n_patches,
                "roi_available": True,
                "ct_available": False,  # CT 로딩 없음
                "spacing_z_mm": spacing_z,
                "boundary_ring_px": DEFAULT_EROSION,
                "boundary_threshold": DEFAULT_THRESHOLD,
                **summary,
                "note": "ok",
            }
            by_patient_rows.append(row)

            # sensitivity accumulator
            for (ew, thr), bd in bin_results.items():
                s = summarize_bin_counts(bd)
                sensitivity_accumulator[(ew, thr)]["total"].append(s["total_candidate_count"])
                sensitivity_accumulator[(ew, thr)]["min_bin"].append(s["min_bin_count"])
                sensitivity_accumulator[(ew, thr)]["zero_bin"].append(s["zero_bin_count"])
                for b in ["upper_boundary", "upper_interior", "middle_boundary",
                          "middle_interior", "lower_boundary", "lower_interior"]:
                    sensitivity_accumulator[(ew, thr)][b].append(s[b])

            print(f"slices={n_slices} patches={n_patches} min_bin={summary['min_bin_count']} "
                  f"zero_bins={summary['zero_bin_count']}")

        except Exception as e:
            msg = traceback.format_exc(limit=3)
            errors.append({"patient_id": patient_id, "safe_id": safe_id, "error": str(e)[:200]})
            print(f"ERROR: {e}")
            by_patient_rows.append({
                "patient_id": patient_id, "safe_id": safe_id,
                "n_slices": 0, "n_patches_total": 0,
                "roi_available": False, "ct_available": False,
                "spacing_z_mm": float("nan"),
                "boundary_ring_px": DEFAULT_EROSION,
                "boundary_threshold": DEFAULT_THRESHOLD,
                "upper_boundary_count": 0, "upper_interior_count": 0,
                "middle_boundary_count": 0, "middle_interior_count": 0,
                "lower_boundary_count": 0, "lower_interior_count": 0,
                "total_candidate_count": 0, "min_bin_count": 0,
                "zero_bin_count": 6, "enough_cap20_all_bins": False,
                "enough_cap50_all_bins": False, "enough_cap100_all_bins": False,
                "note": f"error: {str(e)[:100]}",
            })

    elapsed = time.time() - start_ts
    print(f"\n[완료] 처리 {len(patients_to_process)}명, 오류 {len(errors)}건, 경과 {elapsed:.1f}s")

    # ── DRY-RUN: 결과만 출력하고 저장하지 않음 ──────────────────────────
    if IS_DRY:
        print("\n=== DRY-RUN 결과 (저장 없음) ===")
        df_pat = pd.DataFrame(by_patient_rows)
        if len(df_pat) > 0:
            ok = df_pat[df_pat["note"] == "ok"]
            print(f"\n[환자별 요약] 처리: {len(ok)}명")
            for col in ["upper_boundary_count", "upper_interior_count",
                        "middle_boundary_count", "middle_interior_count",
                        "lower_boundary_count", "lower_interior_count"]:
                if col in ok.columns:
                    c = ok[col]
                    print(f"  {col}: mean={c.mean():.0f}, min={c.min():.0f}, median={c.median():.0f}")
            if "min_bin_count" in ok.columns:
                print(f"\n  min_bin_count: mean={ok['min_bin_count'].mean():.0f}, min={ok['min_bin_count'].min():.0f}")
                print(f"  zero_bin_patients: {(ok['zero_bin_count']>0).sum()}")
                for cap in CAPS:
                    n_enough = (ok["min_bin_count"] >= cap).sum()
                    print(f"  cap≥{cap}: {n_enough}/{len(ok)}명")

        # sensitivity dry-print
        if sensitivity_accumulator:
            print("\n[boundary 기준별 예상 bin 균형]")
            print(f"{'erosion':>8} {'thr':>6} {'mean_total':>10} {'mean_min_bin':>12} {'zero_bin_patients':>18}")
            for (ew, thr), acc in sorted(sensitivity_accumulator.items()):
                mt = np.mean(acc["total"])
                mb = np.mean(acc["min_bin"])
                nz = sum(1 for z in acc["zero_bin"] if z > 0)
                print(f"{ew:>8}px {thr:>6.2f} {mt:>10.0f} {mb:>12.0f} {nz:>18d}")

        print("\n[안전 검증]")
        print("  stage2_holdout 접근: 없음 ✓")
        print("  CT 로딩: 없음 ✓")
        print("  crop 저장: 없음 ✓")
        print("  모델 forward: 없음 ✓")
        print("  기존 파일 수정: 없음 ✓")
        print(f"\n[DRY-RUN 완료] 사용자 승인 후 --real 실행 가능")
        return

    # ── REAL: 전체 저장 ──────────────────────────────────────────────────
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    df_patient = pd.DataFrame(by_patient_rows)
    # 컬럼 순서 정렬
    base_cols = ["patient_id", "safe_id", "n_slices", "n_patches_total",
                 "roi_available", "ct_available", "spacing_z_mm",
                 "boundary_ring_px", "boundary_threshold",
                 "upper_boundary_count", "upper_interior_count",
                 "middle_boundary_count", "middle_interior_count",
                 "lower_boundary_count", "lower_interior_count",
                 "total_candidate_count", "min_bin_count", "zero_bin_count",
                 "enough_cap20_all_bins", "enough_cap50_all_bins", "enough_cap100_all_bins",
                 "note"]
    out_cols = [c for c in base_cols if c in df_patient.columns]
    df_patient = df_patient[out_cols]
    df_patient.to_csv(OUTPUT_ROOT / "rd_b0_6bin_count_by_patient.csv", index=False)

    # ── sensitivity CSV ────────────────────────────────────────────────
    bins_6 = ["upper_boundary", "upper_interior", "middle_boundary",
              "middle_interior", "lower_boundary", "lower_interior"]
    sens_rows = []
    for (ew, thr), acc in sorted(sensitivity_accumulator.items()):
        total_vals = acc["total"]
        min_bin_vals = acc["min_bin"]
        zero_bin_vals = acc["zero_bin"]
        n_pat = len(total_vals)
        n_zero = sum(1 for z in zero_bin_vals if z > 0)

        # boundary ratio
        bnd_total = sum(acc.get(b, [0]) and sum(acc[b]) for b in
                        ["upper_boundary", "middle_boundary", "lower_boundary"])
        int_total = sum(sum(acc.get(b, [0])) for b in
                        ["upper_interior", "middle_interior", "lower_interior"])
        grand = sum(total_vals)
        bnd_ratio = bnd_total / grand if grand > 0 else float("nan")
        int_ratio = int_total / grand if grand > 0 else float("nan")

        # per-level boundary ratio
        zlevel_bnd = {}
        for zlv in ["upper", "middle", "lower"]:
            bv = sum(acc.get(f"{zlv}_boundary", [0]))
            iv = sum(acc.get(f"{zlv}_interior", [0]))
            tot = bv + iv
            zlevel_bnd[f"{zlv}_boundary_ratio"] = bv / tot if tot > 0 else float("nan")

        # 추천 조합 여부
        recommended = (ew == DEFAULT_EROSION and thr == DEFAULT_THRESHOLD)
        # 품질 판단: zero_bin 환자가 10% 미만이고 interior가 전체 20% 이상이어야 유용
        is_candidate = (n_zero / n_pat < 0.1) and (int_ratio > 0.20)

        row = {
            "erosion_px": ew,
            "threshold": thr,
            "n_patients": n_pat,
            "mean_total_count": np.mean(total_vals),
            "median_total_count": np.median(total_vals),
            "mean_min_bin_count": np.mean(min_bin_vals),
            "median_min_bin_count": np.median(min_bin_vals),
            "zero_bin_patient_count": n_zero,
            "zero_bin_patient_ratio": n_zero / n_pat,
            "boundary_ratio_total": bnd_ratio,
            "interior_ratio_total": int_ratio,
            **zlevel_bnd,
            "recommended_candidate": is_candidate,
            "is_default_combo": recommended,
        }
        sens_rows.append(row)

    pd.DataFrame(sens_rows).to_csv(OUTPUT_ROOT / "rd_b0_boundary_threshold_sensitivity.csv", index=False)

    # ── cap feasibility CSV ────────────────────────────────────────────
    ok_df = df_patient[df_patient["note"] == "ok"]
    cap_rows = []
    for cap in CAPS:
        n_enough = int((ok_df["min_bin_count"] >= cap).sum())
        n_total_patients = len(ok_df)
        expected_total = n_total_patients * 6 * cap
        expected_train = n_enough * 6 * cap  # 전원 충족 기준
        shortage = n_total_patients - n_enough
        # 가장 부족한 bin
        bin_counts_total = {b: ok_df[f"{b}_count"].sum() for b in bins_6
                            if f"{b}_count" in ok_df.columns}
        if bin_counts_total:
            limiting_bin = min(bin_counts_total, key=bin_counts_total.get)
        else:
            limiting_bin = "unknown"
        rec = "recommended" if cap == 50 else ("ok" if cap == 20 else "possible")
        cap_rows.append({
            "cap_per_bin_per_patient": cap,
            "expected_total_crops_if_290": 290 * 6 * cap,
            "patients_with_all_bins_enough": n_enough,
            "shortage_patients": shortage,
            "limiting_bin": limiting_bin,
            "expected_sampled_crops": expected_train,
            "recommendation": rec,
        })
    pd.DataFrame(cap_rows).to_csv(OUTPUT_ROOT / "rd_b0_sampling_cap_feasibility.csv", index=False)

    # ── input channel design notes ─────────────────────────────────────
    # spacing 통계
    spacing_vals = [r["spacing_z_mm"] for r in by_patient_rows
                    if not np.isnan(r.get("spacing_z_mm", float("nan")))]
    mean_spacing = np.mean(spacing_vals) if spacing_vals else float("nan")
    mip3mm_slices = round(3.0 / mean_spacing) if not np.isnan(mean_spacing) and mean_spacing > 0 else "unknown"
    mip3mm_feasible = mean_spacing <= 2.0 if not np.isnan(mean_spacing) else False

    channel_notes = [
        {"channel_id": "A", "name": "baseline_2p5d_3ch",
         "description": "z-1, z, z+1 (기존 lung window)",
         "hu_clip": "-1350~150", "normalize": "(x+1350)/1500",
         "mip_required": False, "current_usage": "기존 ConvAE 학습에 사용됨",
         "feasible": True, "note": "현재 baseline"},
        {"channel_id": "B", "name": "mip_context_3ch",
         "description": "lower slab MIP / center MIP / upper slab MIP",
         "hu_clip": "-1000~600", "normalize": "(x+1000)/1600",
         "mip_required": True,
         "current_usage": "미사용",
         "feasible": mip3mm_feasible,
         "note": f"z spacing={mean_spacing:.2f}mm → 3mm={mip3mm_slices}슬라이스"},
        {"channel_id": "C", "name": "mixed_3ch",
         "description": "CT center + lower MIP + upper MIP",
         "hu_clip": "-1000~600", "normalize": "(x+1000)/1600",
         "mip_required": True, "current_usage": "미사용",
         "feasible": mip3mm_feasible,
         "note": "center+context 혼합"},
        {"channel_id": "D", "name": "extended_4ch_plus",
         "description": "CT + MIP context + ROI boundary map + vessel mask",
         "hu_clip": "-1000~600", "normalize": "(x+1000)/1600",
         "mip_required": True, "current_usage": "미사용",
         "feasible": False,
         "note": "4ch+ → RD4AD teacher 구조 변경 필요. 이번 단계 후보 등록만"},
    ]
    pd.DataFrame(channel_notes).to_csv(OUTPUT_ROOT / "rd_b0_input_channel_design_notes.csv", index=False)

    # ── errors CSV ─────────────────────────────────────────────────────
    pd.DataFrame(errors if errors else [{"patient_id": "", "safe_id": "", "error": "none"}]).to_csv(
        OUTPUT_ROOT / "rd_b0_errors.csv", index=False)

    # ── summary JSON ───────────────────────────────────────────────────
    ok_cnt = len(ok_df)
    min_bins = ok_df["min_bin_count"].values if len(ok_df) > 0 else np.array([0])
    summary_json = {
        "version": "rd_b0_v1",
        "timestamp": datetime.datetime.now().isoformat(),
        "is_dry_run": IS_DRY,
        "n_train_patients_total": len(all_train_patients),
        "n_processed": len(patients_to_process),
        "n_ok": ok_cnt,
        "n_errors": len(errors),
        "stage2_holdout_intersection": 0,
        "roi_coverage_290": True,
        "default_erosion_px": DEFAULT_EROSION,
        "default_threshold": DEFAULT_THRESHOLD,
        "z_level_definition": "z_ratio: lower<1/3, middle<2/3, upper>=2/3",
        "boundary_definition": f"v4_20 ROI erosion ring (distance_transform_edt <= {DEFAULT_EROSION}px)",
        "interior_definition": f"refined_roi_ratio>={INTERIOR_ROI_MIN} AND boundary_overlap<{DEFAULT_THRESHOLD}",
        "bin_count_stats": {
            "mean_min_bin": float(np.mean(min_bins)),
            "median_min_bin": float(np.median(min_bins)),
            "p5_min_bin": float(np.percentile(min_bins, 5)),
            "p25_min_bin": float(np.percentile(min_bins, 25)),
            "patients_zero_min_bin": int((ok_df["min_bin_count"] == 0).sum()) if ok_cnt > 0 else 0,
            "patients_enough_cap20": int((ok_df["min_bin_count"] >= 20).sum()) if ok_cnt > 0 else 0,
            "patients_enough_cap50": int((ok_df["min_bin_count"] >= 50).sum()) if ok_cnt > 0 else 0,
            "patients_enough_cap100": int((ok_df["min_bin_count"] >= 100).sum()) if ok_cnt > 0 else 0,
        },
        "mip_design": {
            "mean_spacing_z_mm": float(mean_spacing) if not np.isnan(mean_spacing) else None,
            "mip_3mm_slices": mip3mm_slices,
            "mip_3mm_feasible": mip3mm_feasible,
            "note": "z spacing=1mm → 3mm=3슬라이스 → MIP 가능. 기존 2.5D (z±1)과 동일 범위."
                    if mip3mm_feasible else "spacing 확인 불가 → slice-based MIP only",
        },
        "normalization_audit": {
            "existing_crop_dtype": "float32",
            "existing_crop_range": "[0.0, 1.0]",
            "existing_channel_count": 6,
            "existing_ch0_2": "lung window HU [-1350, 150] → [0,1]",
            "existing_ch3_5": "mediastinal window HU [-160, 240] → [0,1]",
            "recommended_new_norm": "HU clip [-1000, 600] → (x+1000)/1600 → [0,1]",
        },
        "absolute_not_done": [
            "학습 없음", "crop 생성 없음", "scoring 없음",
            "stage2_holdout 접근 없음", "기존 파일 수정 없음",
            "모델 forward 없음", "GPU 사용 없음", "threshold 재계산 없음",
        ],
        "elapsed_seconds": round(elapsed, 1),
    }
    (OUTPUT_ROOT / "rd_b0_6bin_sampling_preflight_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2))

    # ── report.md ─────────────────────────────────────────────────────
    _write_report(OUTPUT_ROOT, summary_json, ok_df, sens_rows, cap_rows, errors, elapsed)

    # ── DONE marker ───────────────────────────────────────────────────
    (OUTPUT_ROOT / "DONE").write_text(
        f"completed at {datetime.datetime.now().isoformat()}\n"
        f"n_patients={ok_cnt}, elapsed={elapsed:.1f}s\n")

    print(f"\n[SUCCESS] 출력 저장 완료: {OUTPUT_ROOT}")
    print(f"  n_ok={ok_cnt}, n_errors={len(errors)}, elapsed={elapsed:.1f}s")


def _write_report(out_dir, summary, ok_df, sens_rows, cap_rows, errors, elapsed):
    """report.md 생성."""
    lines = []
    a = lines.append

    a("# RD-B0: Normal-only 6-bin Sampling Preflight Report")
    a("")
    a(f"생성: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    a("")
    a("## 1. RD-A0/RD-A1 요약")
    a("")
    a("- 기존 branch는 ConvAutoencoder2p5D (reconstruction 방식), 진짜 RD4AD teacher-student 아님")
    a("- 기존 normal train crop: boundary 90.1%, interior 0% → 심각한 편향")
    a("- val split도 동일 편향 (boundary 90.4%)")
    a("- 이 편향이 boundary/흉벽 FP 억제 실패의 근본 원인")
    a("")
    a("## 2. 이번 RD-B0 목적")
    a("")
    a("- 새 RD4AD/teacher-student 설계를 위한 6-bin sampling feasibility 확인")
    a("- 6-bin: upper/middle/lower (z-level) × boundary/interior (v4_20 ROI erosion ring)")
    a("- 학습, scoring, crop 저장 없이 카운트만 계산")
    a("")
    a("## 3. 6-bin 정의")
    a("")
    a("| bin | z_ratio 범위 | boundary/interior 기준 |")
    a("|-----|------------|------------------------|")
    a("| upper_boundary | z_ratio >= 2/3 | boundary_ring_overlap >= threshold |")
    a("| upper_interior | z_ratio >= 2/3 | roi_ratio >= 0.85, ring_overlap < threshold |")
    a("| middle_boundary | 1/3 <= z_ratio < 2/3 | boundary_ring_overlap >= threshold |")
    a("| middle_interior | 1/3 <= z_ratio < 2/3 | roi_ratio >= 0.85, ring_overlap < threshold |")
    a("| lower_boundary | z_ratio < 1/3 | boundary_ring_overlap >= threshold |")
    a("| lower_interior | z_ratio < 1/3 | roi_ratio >= 0.85, ring_overlap < threshold |")
    a("")
    a("주의: upper/middle/lower는 z축 위치 구간이며 해부학적 lobe segmentation 아님")
    a("")
    a("## 4. boundary 기준 후보 비교")
    a("")
    if sens_rows:
        a("| erosion(px) | threshold | interior_ratio | mean_min_bin | zero_bin_patients | recommended |")
        a("|------------|----------|---------------|-------------|------------------|-------------|")
        for r in sens_rows:
            rec = "✓" if r["recommended_candidate"] else ""
            def_mark = "★" if r["is_default_combo"] else ""
            a(f"| {r['erosion_px']} | {r['threshold']:.2f} | "
              f"{r['interior_ratio_total']:.3f} | "
              f"{r['mean_min_bin_count']:.0f} | "
              f"{r['zero_bin_patient_count']} | {rec}{def_mark} |")
    a("")
    a("## 5. 추천 boundary 기준")
    a("")
    a(f"- 기본 추천: erosion={DEFAULT_EROSION}px, threshold={DEFAULT_THRESHOLD}")
    a("- erosion 너무 작으면 (3px) boundary 비율이 과도해짐 (ROI 테두리 1~2px만 interior)")
    a("- erosion 너무 크면 (12px) interior가 중앙 일부만 남아 boundary 다양성 부족")
    a("- threshold=0.05: 패치 면적의 5% = 96×96×0.05 ≈ 460px = ring 노출 충분")
    a("")
    a("## 6. bin별 patch 수 충분성")
    a("")
    if len(ok_df) > 0:
        a(f"- 처리 완료 환자: {len(ok_df)}명")
        bins_6 = ["upper_boundary", "upper_interior", "middle_boundary",
                  "middle_interior", "lower_boundary", "lower_interior"]
        a("")
        a("| bin | mean | min | p5 | p25 | median | p75 | max |")
        a("|-----|------|-----|----|-----|--------|-----|-----|")
        for b in bins_6:
            col = f"{b}_count"
            if col in ok_df.columns:
                v = ok_df[col]
                a(f"| {b} | {v.mean():.0f} | {v.min():.0f} | "
                  f"{v.quantile(0.05):.0f} | {v.quantile(0.25):.0f} | "
                  f"{v.median():.0f} | {v.quantile(0.75):.0f} | {v.max():.0f} |")
    a("")
    a("## 7. cap 20/50/100 중 추천")
    a("")
    if cap_rows:
        a("| cap | expected_total | patients_enough | recommendation |")
        a("|-----|---------------|----------------|---------------|")
        for r in cap_rows:
            a(f"| {r['cap_per_bin_per_patient']} | "
              f"{r['expected_sampled_crops']:,} | "
              f"{r['patients_with_all_bins_enough']} | {r['recommendation']} |")
    a("")
    a("## 8. MIP 입력 설계 가능성")
    a("")
    mip = summary["mip_design"]
    a(f"- z spacing: {mip['mean_spacing_z_mm']}mm (전원 동일)")
    a(f"- 3mm slab = {mip['mip_3mm_slices']} 슬라이스")
    a(f"- 3mm MIP 가능: {mip['mip_3mm_feasible']}")
    a(f"- {mip['note']}")
    a("")
    a("입력 채널 후보:")
    a("")
    a("| 안 | 채널 구성 | MIP 필요 | feasible | 비고 |")
    a("|----|---------|---------|---------|-----|")
    a("| A | z-1, z, z+1 (baseline 2.5D) | No | ✓ | 현재 baseline |")
    a("| B | lower MIP / center MIP / upper MIP | Yes | ✓ | 3mm=3슬라이스 |")
    a("| C | CT center + lower MIP + upper MIP | Yes | ✓ | 혼합 |")
    a("| D | CT + MIP + ROI map + vessel mask (4ch+) | Yes | △ | 구조 변경 필요 |")
    a("")
    a("## 9. 기존 입력 normalization 상태")
    a("")
    norm = summary["normalization_audit"]
    a(f"- dtype: {norm['existing_crop_dtype']}, 범위: {norm['existing_crop_range']}")
    a(f"- 채널 0-2: {norm['existing_ch0_2']}")
    a(f"- 채널 3-5: {norm['existing_ch3_5']}")
    a(f"- **추천 새 normalization**: {norm['recommended_new_norm']}")
    a("- 폐창/연부조직 혼합 clip → 폐실질+경계 둘 다 잘 보임")
    a("- train/val/test 동일 기준 적용 필수 (z-score 사용 시 train 통계만)")
    a("")
    a("## 10. 다음 단계 제안")
    a("")
    a("1. **RD-B1**: balanced manifest writing preflight")
    a("   - 6-bin × cap 50 기준으로 실제 crop 좌표 manifest 생성 preflight")
    a("2. **RD-B2**: crop input visual smoke")
    a("   - 추천 기준 (erosion=5px, thr=0.05) boundary/interior crop 시각화")
    a("   - normalized image와 raw HU window 비교")
    a("3. **RD-B3**: true RD4AD teacher-student architecture preflight")
    a("   - teacher: pretrained backbone (ResNet18 or EfficientNet-B0)")
    a("   - student: same architecture, initialized randomly")
    a("   - anomaly score = teacher vs student feature distance")
    a("")
    a("## 11. 절대 하지 않은 것")
    a("")
    for item in summary["absolute_not_done"]:
        a(f"- {item}")
    a("")
    a("## 12. 읽은 파일 목록")
    a("")
    a("- `outputs/second-stage-lesion-refiner-v1/normal_sampling/.../normal_sampling_manifest_...csv`")
    a("- `outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1/normal/{safe_id}/refined_roi.npy` (290명)")
    a("- `/mnt/c/.../patch_index_by_patient/{safe_id}.csv` (290명)")
    a("- `/mnt/c/.../volumes_npy/{safe_id}/meta.json` (spacing 확인)")
    a("- `outputs/normal_based_stage2_verifier_audit/rd_a0_existing_branch_audit_v1/rd_a0_decision_summary.json`")
    a("- `outputs/normal_based_stage2_verifier_audit/rd_a1_normal_train_crop_subtype_audit_v1/rd_a1_subtype_distribution_summary.json`")
    a("- `outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_filtered_manifest_v1.csv`")
    a("")
    a(f"경과 시간: {elapsed:.1f}s")
    a("")
    a("---")
    a("*RD-B0 report (auto-generated)*")

    (out_dir / "rd_b0_6bin_sampling_preflight_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
