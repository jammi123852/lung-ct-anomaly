#!/usr/bin/env python3
"""
B1-F1 EfficientNet v4_20 FP Source Distribution / Stage2 Handoff Component Audit

목표:
  EfficientNet-B0 v4_20 ROI 1차 실험 결과에서 고점 component 분석.
  FP 원인 (vessel / chestwall-boundary / hilar-mediastinal / other) 분류.
  2차 학습 handoff 후보의 실제 병변/FP 비율 정리.

안전 조건:
  - stage2_holdout 접근 금지
  - score/model/threshold/ROI/CT/mask 수정 금지
  - 기존 파일 덮어쓰기 금지
  - GPU 사용 금지, 재학습/재추론/threshold 재계산 금지

실행:
  --dry-run / --plan   계획 출력만 (파일 생성 없음)
  --real               실제 분석 및 파일 생성
"""
import sys

if __name__ == "__main__" and len(sys.argv) < 2:
    print(
        "[ERROR] bare-run guard: 인수 없이 실행 금지. "
        "--dry-run 또는 --real 을 사용하세요.",
        file=sys.stderr,
    )
    sys.exit(2)

import os
import csv
import json
import traceback
from pathlib import Path

import numpy as np
from scipy.ndimage import label as nd_label, binary_dilation, binary_erosion
from skimage.filters import frangi
from skimage.morphology import white_tophat, disk, remove_small_objects

ALLOW_REAL = "--real" in sys.argv
DRY_RUN = not ALLOW_REAL

# ─── 경로 ─────────────────────────────────────────────────────────────────────
PROJECT = Path("/home/jinhy/project/lung-ct-anomaly")

SCORE_ROOT_LESION = (
    PROJECT
    / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs/scores/lesion_stage1_dev_by_patient"
)
THRESHOLD_JSON = (
    PROJECT
    / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs/reports/normal_val/p_b9_normal_val_threshold.json"
)
STAGE_SPLIT_CSV = (
    PROJECT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"
)
STAGE2_MANIFEST_CSV = (
    PROJECT
    / "outputs/second-stage-lesion-refiner-v1/candidates"
    / "rule_s6a_gs2_selected_candidate_manifest_dryrun.csv"
)
ROI_BASE_LESION = (
    PROJECT
    / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1/lesion"
)
NROOT_LESION = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy"
)

OUT_ROOT = (
    PROJECT
    / "outputs/position-aware-padim-v1"
    / "efficientnet_v4_20_fp_source_audit"
    / "b1f1_fp_source_distribution_v1"
)

# ─── 파라미터 ──────────────────────────────────────────────────────────────────
THRESHOLD_P95     = None  # JSON에서 로드
MIP_SLAB_HALF     = 5     # axial 10-slice MIP
VESSEL_SIGMAS     = (0.5, 1.0, 1.5, 2.0)
TOPHAT_RADIUS     = 10
VESSEL_PERCENTILE = 85.0
MIN_AREA          = 10
LESION_DILATE_PX  = 4     # near-lesion 판정 dilate

VESSEL_DOMINANT_RATIO   = 0.25
VESSEL_TOUCH_RATIO      = 0.0
BOUNDARY_TOUCH_RATIO    = 0.10
BOUNDARY_ERODE_PX       = 3    # ROI 내부 erode 픽셀 (= boundary 두께)
BOUNDARY_BBOX_PX        = 3    # bbox가 ROI boundary에 3px 이내

PATCH_STRIDE      = 16    # score CSV 기준

# ─── 참조 파일 목록 ────────────────────────────────────────────────────────────
REFERENCE_REPORTS = [
    "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/reports/normal_val/p_b9_normal_val_threshold.json",
    "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/reports/lesion_stage1_dev/p_b13_stage1_dev_metrics_report.json",
    "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/reports/p_b15_v4_20_roi_decision_checkpoint.json",
    "outputs/second-stage-lesion-refiner-v1/candidates/rule_s6a_gs2_selected_candidate_manifest_dryrun.csv",
    "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv",
]


# ══════════════════════════════════════════════════════════════════════════════
# 유틸 함수
# ══════════════════════════════════════════════════════════════════════════════

def load_threshold():
    with open(THRESHOLD_JSON, encoding="utf-8") as f:
        data = json.load(f)
    val = data["threshold_p95"]
    assert data.get("threshold_recalculated", True) is False or data.get("v4_20_branch_specific_threshold")
    return float(val)


def load_stage2_holdout():
    holdout = set()
    with open(STAGE_SPLIT_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["stage_split"] == "stage2_holdout":
                holdout.add(r["patient_id"])
    return holdout


def load_stage1_dev_meta():
    """(patient_id, safe_id, group) 목록 반환"""
    patients = []
    with open(STAGE_SPLIT_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["stage_split"] == "stage1_dev":
                patients.append({
                    "patient_id": r["patient_id"],
                    "safe_id":    r["safe_id"],
                    "group":      r["group"],
                })
    return patients


def load_stage2_manifest():
    """stage2 handoff manifest: {(patient_id, local_z, y0, x0) -> row}"""
    data = {}
    if not STAGE2_MANIFEST_CSV.exists():
        return data
    with open(STAGE2_MANIFEST_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("stage_split", "") == "stage2_holdout":
                continue  # holdout 접근 금지
            key = (r["patient_id"], int(r["local_z"]), int(r["y0"]), int(r["x0"]))
            data[key] = r
    return data


def load_score_csv(patient_id):
    path = SCORE_ROOT_LESION / f"{patient_id}.csv"
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def build_components(rows, threshold):
    """
    threshold 이상인 patch를 (local_z, y_bin, x_bin) 격자로 표현하여
    3D 26-connected component 라벨링.
    반환: list of component dict
    """
    hot = [r for r in rows if float(r["padim_score"]) >= threshold]
    if not hot:
        return [], hot

    # 격자 좌표로 변환
    coords = np.array([
        [int(r["local_z"]), int(r["y0"]) // PATCH_STRIDE, int(r["x0"]) // PATCH_STRIDE]
        for r in hot
    ], dtype=np.int64)

    # bounding box로 sparse grid 만들기
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    shape = (z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1)
    grid = np.zeros(shape, dtype=np.uint8)
    shifted = coords - np.array([z_min, y_min, x_min])
    for sz, sy, sx in shifted:
        grid[sz, sy, sx] = 1

    struct = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connectivity
    labeled, n_comp = nd_label(grid, structure=struct)

    # 각 component에 row 배정
    comp_map = {}  # comp_id -> [row_index]
    for idx, (sz, sy, sx) in enumerate(shifted):
        cid = labeled[sz, sy, sx]
        comp_map.setdefault(int(cid), []).append(idx)

    components = []
    for cid, idxs in comp_map.items():
        comp_rows = [hot[i] for i in idxs]
        scores = [float(r["padim_score"]) for r in comp_rows]
        zvals  = [int(r["local_z"]) for r in comp_rows]
        y0s    = [int(r["y0"]) for r in comp_rows]
        x0s    = [int(r["x0"]) for r in comp_rows]
        y1s    = [int(r["y1"]) for r in comp_rows]
        x1s    = [int(r["x1"]) for r in comp_rows]
        lp     = [float(r["lesion_patch_ratio"]) for r in comp_rows]
        lpx    = [int(r["lesion_pixels"]) for r in comp_rows]
        pbins  = [r["position_bin"] for r in comp_rows]
        top_i  = int(np.argmax(scores))

        pbin_cnt = {}
        for p in pbins:
            pbin_cnt[p] = pbin_cnt.get(p, 0) + 1
        pbin_majority = max(pbin_cnt, key=pbin_cnt.get)
        cp = comp_rows[0].get("central_peripheral", "unknown")

        components.append({
            "component_id":           cid,
            "rows":                   comp_rows,
            "max_score":              float(np.max(scores)),
            "mean_score":             float(np.mean(scores)),
            "patch_count":            len(comp_rows),
            "z_span":                 int(np.max(zvals) - np.min(zvals) + 1),
            "local_z_min":            int(np.min(zvals)),
            "local_z_max":            int(np.max(zvals)),
            "bbox_y0":                int(np.min(y0s)),
            "bbox_x0":                int(np.min(x0s)),
            "bbox_y1":                int(np.max(y1s)),
            "bbox_x1":                int(np.max(x1s)),
            "top_patch_local_z":      int(zvals[top_i]),
            "top_patch_y0":           int(y0s[top_i]),
            "top_patch_x0":           int(x0s[top_i]),
            "position_bin_majority":  pbin_majority,
            "central_peripheral":     cp,
            # lesion: score CSV 기반 (patch 수준)
            "lesion_pixels_sum":      int(np.sum(lpx)),
            "total_patch_pixels":     len(comp_rows) * (32 * 32),
            # 임시 (실제 overlap은 아래에서 업데이트)
            "lesion_overlap_ratio":   float(np.mean(lp)),
            "lesion_hit_flag":        any(l > 0 for l in lp),
            "lesion_near_flag":       False,  # mask-level 계산 후 업데이트
            "vessel_overlap_ratio":   0.0,
            "vessel_dominant_flag":   False,
            "vessel_touch_flag":      False,
            "boundary_touch_ratio":   0.0,
            "chestwall_boundary_flag": False,
            "peripheral_flag":        "peripheral" in cp,
            "hilar_proxy_flag":       False,
            "component_source_label": "other_fp",
            "stage2_handoff_label":   "not_in_manifest",
            "note":                   "",
        })

    return components, hot


# ─── CT / mask 로드 (mmap) ────────────────────────────────────────────────────

def safe_load_npy(path):
    path = Path(path)
    if not path.exists():
        return None
    return np.load(path, mmap_mode="r")


def apply_lung_window(arr, level=-600, width=1500):
    lo = level - width / 2
    hi = level + width / 2
    img = np.clip(arr.astype(np.float32), lo, hi)
    return (img - lo) / (hi - lo)


def hu_normalize(arr):
    return np.clip((arr.astype(np.float32) + 1000.0) / 1600.0, 0.0, 1.0)


def compute_vessel_mask_for_slices(ct_vol, roi_vol, z_set):
    """
    z_set에 포함된 slice에 대해 axial 10-slice MIP 3-method union vessel mask 계산.
    반환: {local_z: 2D bool mask}
    """
    n = ct_vol.shape[0]
    result = {}
    for local_z in sorted(z_set):
        z_lo = max(0, local_z - MIP_SLAB_HALF)
        z_hi = min(n, local_z + MIP_SLAB_HALF)
        slab = ct_vol[z_lo:z_hi].astype(np.float32)
        mip  = np.max(slab, axis=0)

        roi_slab = roi_vol[z_lo:z_hi]
        roi_proj = np.any(roi_slab > 0, axis=0)
        valid = roi_proj

        if not valid.any():
            result[local_z] = np.zeros(mip.shape, dtype=bool)
            continue

        # intensity p85
        lw  = apply_lung_window(mip)
        t   = float(np.percentile(lw[valid], VESSEL_PERCENTILE))
        m1  = (lw > t) & valid

        # top-hat p85
        hn  = hu_normalize(mip)
        th  = white_tophat(hn, disk(TOPHAT_RADIUS))
        t2  = float(np.percentile(th[valid], VESSEL_PERCENTILE))
        m2  = (th > t2) & valid

        # vesselness p85
        fr  = frangi(hn, sigmas=VESSEL_SIGMAS, black_ridges=False)
        t3  = float(np.percentile(fr[valid], VESSEL_PERCENTILE))
        m3  = (fr > t3) & valid

        union = m1 | m2 | m3
        if MIN_AREA > 0 and union.any():
            union = remove_small_objects(union.copy(), min_size=MIN_AREA)

        result[local_z] = union
    return result


def compute_lesion_near(lesion_vol, z_min, z_max, bbox_y0, bbox_x0, bbox_y1, bbox_x1):
    """component bbox dilate 후 lesion과 근접 여부 확인."""
    if lesion_vol is None:
        return False
    n = lesion_vol.shape[0]
    z_lo = max(0, z_min - 1)
    z_hi = min(n, z_max + 2)
    slab = np.array(lesion_vol[z_lo:z_hi])  # (dz, H, W)
    if not slab.any():
        return False
    h, w = slab.shape[1], slab.shape[2]
    # component bbox를 LESION_DILATE_PX만큼 확장
    ey0 = max(0, bbox_y0 - LESION_DILATE_PX)
    ex0 = max(0, bbox_x0 - LESION_DILATE_PX)
    ey1 = min(h, bbox_y1 + LESION_DILATE_PX)
    ex1 = min(w, bbox_x1 + LESION_DILATE_PX)
    roi_slab = slab[:, ey0:ey1, ex0:ex1]
    return bool(roi_slab.any())


def compute_boundary_stats(roi_vol, z_min, z_max, bbox_y0, bbox_x0, bbox_y1, bbox_x1):
    """
    ROI boundary (ROI - erode(ROI)) 와 component bbox overlap 계산.
    반환: (boundary_touch_ratio, roi_boundary_distance_min)
    """
    n = roi_vol.shape[0]
    z_lo = max(0, z_min)
    z_hi = min(n, z_max + 1)
    slab = np.array(roi_vol[z_lo:z_hi]).astype(bool)
    if not slab.any():
        return 0.0, 999

    # 2D slice별 boundary 계산하여 합산
    boundary_proj = np.zeros(slab.shape[1:], dtype=bool)
    for zi in range(slab.shape[0]):
        sl = slab[zi]
        eroded = binary_erosion(sl, iterations=BOUNDARY_ERODE_PX)
        boundary_proj |= (sl & ~eroded)

    h, w = boundary_proj.shape
    by0 = max(0, bbox_y0)
    bx0 = max(0, bbox_x0)
    by1 = min(h, bbox_y1)
    bx1 = min(w, bbox_x1)

    if by1 <= by0 or bx1 <= bx0:
        return 0.0, 999

    comp_mask = np.zeros((h, w), dtype=bool)
    comp_mask[by0:by1, bx0:bx1] = True

    comp_pixels = comp_mask.sum()
    touch_pixels = (boundary_proj & comp_mask).sum()
    touch_ratio = float(touch_pixels / comp_pixels) if comp_pixels > 0 else 0.0

    # boundary 최소 거리 (단순 bbox 기준)
    roi_proj = np.any(slab, axis=0)
    if roi_proj.any():
        iy, ix = np.where(roi_proj)
        ry_min, ry_max = iy.min(), iy.max()
        rx_min, rx_max = ix.min(), ix.max()
        dist = min(
            abs(bbox_y0 - ry_min), abs(bbox_y1 - ry_max),
            abs(bbox_x0 - rx_min), abs(bbox_x1 - rx_max),
        )
    else:
        dist = 999

    return touch_ratio, int(dist)


def compute_vessel_comp_overlap(vessel_masks, z_set, bbox_y0, bbox_x0, bbox_y1, bbox_x1, patch_count):
    """vessel mask와 component patch bbox overlap 계산."""
    total_patch_pixels = 0
    vessel_pixels = 0
    for lz in z_set:
        if lz not in vessel_masks:
            continue
        vm = vessel_masks[lz]
        h, w = vm.shape
        vy0 = max(0, bbox_y0)
        vx0 = max(0, bbox_x0)
        vy1 = min(h, bbox_y1)
        vx1 = min(w, bbox_x1)
        if vy1 <= vy0 or vx1 <= vx0:
            continue
        crop = vm[vy0:vy1, vx0:vx1]
        total_patch_pixels += crop.size
        vessel_pixels += int(crop.sum())

    if total_patch_pixels == 0:
        return 0.0
    return float(vessel_pixels / total_patch_pixels)


def assign_source_label(comp):
    """우선순위 기반 component_source_label 결정."""
    if comp["lesion_overlap_ratio"] > 0 or comp["lesion_hit_flag"]:
        return "lesion_hit"
    if comp["lesion_near_flag"]:
        return "lesion_near"
    if comp["vessel_overlap_ratio"] >= VESSEL_DOMINANT_RATIO:
        return "vessel_dominant"
    if comp["chestwall_boundary_flag"]:
        return "boundary_chestwall"
    if comp["vessel_touch_flag"] and comp["chestwall_boundary_flag"]:
        return "vessel_boundary_mixed"
    if comp["hilar_proxy_flag"]:
        return "hilar_mediastinal_proxy"
    return "other_fp"


# ══════════════════════════════════════════════════════════════════════════════
# 메인 처리
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global THRESHOLD_P95

    # ── 출력 루트 존재 확인 ──────────────────────────────────────────────────────
    if ALLOW_REAL and OUT_ROOT.exists():
        print(f"[ERROR] 출력 루트가 이미 존재합니다. 중단합니다: {OUT_ROOT}", file=sys.stderr)
        sys.exit(1)

    # ── threshold 로드 ──────────────────────────────────────────────────────────
    THRESHOLD_P95 = load_threshold()
    print(f"[INFO] threshold_p95 = {THRESHOLD_P95:.6f}  (출처: p_b9_normal_val_threshold.json)")

    # ── stage2 holdout 확인 ─────────────────────────────────────────────────────
    holdout = load_stage2_holdout()
    print(f"[INFO] stage2_holdout 환자 수: {len(holdout)}")

    # ── stage1_dev 환자 목록 ────────────────────────────────────────────────────
    patients = load_stage1_dev_meta()
    print(f"[INFO] stage1_dev 환자 수: {len(patients)}")

    # ── stage2 manifest 로드 ────────────────────────────────────────────────────
    s2_manifest = load_stage2_manifest()
    print(f"[INFO] stage2 manifest rows (stage1_dev only): {len(s2_manifest)}")

    # ── 참조 파일 존재 확인 ─────────────────────────────────────────────────────
    print("\n[참조 파일 목록]")
    for rp in REFERENCE_REPORTS:
        full = PROJECT / rp
        status = "OK" if full.exists() else "MISSING"
        print(f"  [{status}] {rp}")

    # ── dry-run 계획 출력 ───────────────────────────────────────────────────────
    print(f"\n[PLAN]")
    print(f"  분석 대상: stage1_dev lesion {len(patients)}명")
    print(f"  threshold_p95: {THRESHOLD_P95:.6f}")
    print(f"  stage2 manifest 행 수: {len(s2_manifest)}")
    print(f"  vessel mask: axial {MIP_SLAB_HALF*2}-slice MIP / intensity+tophat+vesselness p{VESSEL_PERCENTILE:.0f} union")
    print(f"  lesion near dilate: {LESION_DILATE_PX}px")
    print(f"  boundary erode: {BOUNDARY_ERODE_PX}px")
    print(f"  출력 루트: {OUT_ROOT}")
    print(f"  GPU 사용: False")
    print(f"  score/model/threshold 수정: False")
    print(f"  stage2_holdout 접근: False")

    if DRY_RUN:
        # dry-run에서도 샘플 1명으로 component 개수 미리 파악
        sample = patients[0]
        rows = load_score_csv(sample["patient_id"])
        hot  = [r for r in rows if float(r["padim_score"]) >= THRESHOLD_P95]
        comps, _ = build_components(rows, THRESHOLD_P95)
        print(f"\n[DRY-RUN SAMPLE] {sample['patient_id']}: total_patches={len(rows)}, "
              f"hot_patches(>=p95)={len(hot)}, components={len(comps)}")
        print("\n[READY] 사용자 승인 후 --real 로 실행하세요.")
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # REAL 처리
    # ═══════════════════════════════════════════════════════════════════════════
    OUT_ROOT.mkdir(parents=True, exist_ok=False)

    all_comp_rows   = []
    patient_summaries = []
    error_rows      = []
    s2_holdout_access_count = 0

    for pat in patients:
        pid  = pat["patient_id"]
        sid  = pat["safe_id"]
        grp  = pat["group"]

        # holdout 접근 금지 재확인
        if pid in holdout:
            error_rows.append({"patient_id": pid, "error": "stage2_holdout 접근 시도 차단"})
            s2_holdout_access_count += 1
            continue

        try:
            rows = load_score_csv(pid)
        except Exception as e:
            error_rows.append({"patient_id": pid, "error": f"score CSV 로드 실패: {e}"})
            continue

        # component 생성
        components, hot = build_components(rows, THRESHOLD_P95)
        if not components:
            patient_summaries.append({
                "patient_id": pid, "role": "lesion", "group": grp,
                "n_components": 0, "n_lesion_hit": 0, "n_lesion_near": 0,
                "n_vessel_dominant": 0, "n_boundary_chestwall": 0,
                "n_vessel_boundary_mixed": 0, "n_hilar_proxy": 0, "n_other_fp": 0,
                "max_score_component_label": "none", "top3_component_labels": "",
            })
            continue

        # CT / ROI / lesion mask 로드
        ct_path      = NROOT_LESION / sid / "ct_hu.npy"
        roi_path     = ROI_BASE_LESION / sid / "refined_roi.npy"
        lesion_path  = NROOT_LESION / sid / "lesion_mask_roi_0_0.npy"

        ct_vol      = safe_load_npy(ct_path)
        roi_vol     = safe_load_npy(roi_path)
        lesion_vol  = safe_load_npy(lesion_path)

        ct_ok  = ct_vol is not None
        roi_ok = roi_vol is not None

        # vessel mask: component가 있는 unique local_z만
        all_z = set()
        for c in components:
            for lz in range(c["local_z_min"], c["local_z_max"] + 1):
                all_z.add(lz)

        vessel_masks = {}
        if ct_ok and roi_ok:
            try:
                vessel_masks = compute_vessel_mask_for_slices(ct_vol, roi_vol, all_z)
            except Exception as e:
                error_rows.append({"patient_id": pid, "error": f"vessel mask 계산 실패: {e}"})

        # 각 component 상세 계산
        for c in components:
            z_set = set(range(c["local_z_min"], c["local_z_max"] + 1))

            # lesion near flag (mask-level)
            if lesion_vol is not None:
                c["lesion_near_flag"] = compute_lesion_near(
                    lesion_vol,
                    c["local_z_min"], c["local_z_max"],
                    c["bbox_y0"], c["bbox_x0"], c["bbox_y1"], c["bbox_x1"],
                )

            # vessel overlap
            if vessel_masks:
                vr = compute_vessel_comp_overlap(
                    vessel_masks, z_set,
                    c["bbox_y0"], c["bbox_x0"], c["bbox_y1"], c["bbox_x1"],
                    c["patch_count"],
                )
                c["vessel_overlap_ratio"]  = vr
                c["vessel_dominant_flag"]  = vr >= VESSEL_DOMINANT_RATIO
                c["vessel_touch_flag"]     = vr > VESSEL_TOUCH_RATIO

            # boundary/chestwall
            if roi_ok:
                touch_r, dist = compute_boundary_stats(
                    roi_vol,
                    c["local_z_min"], c["local_z_max"],
                    c["bbox_y0"], c["bbox_x0"], c["bbox_y1"], c["bbox_x1"],
                )
                c["boundary_touch_ratio"]    = touch_r
                c["roi_boundary_distance_min"] = dist
                c["chestwall_boundary_flag"]  = (
                    touch_r >= BOUNDARY_TOUCH_RATIO or dist <= BOUNDARY_BBOX_PX
                )

            # hilar proxy: position_bin에 central 포함
            pbin = c["position_bin_majority"]
            c["hilar_proxy_flag"] = "central" in pbin and c["vessel_overlap_ratio"] < VESSEL_DOMINANT_RATIO

            # source label
            c["component_source_label"] = assign_source_label(c)

            # stage2 handoff label (manifest 매칭: top patch 기준)
            top_key = (pid, c["top_patch_local_z"], c["top_patch_y0"], c["top_patch_x0"])
            if top_key in s2_manifest:
                mr = s2_manifest[top_key]
                c["stage2_handoff_label"] = mr.get("sampling_label", "unknown")
            else:
                # 같은 component 내 row 전체 시도
                found = False
                for row in c["rows"]:
                    k = (pid, int(row["local_z"]), int(row["y0"]), int(row["x0"]))
                    if k in s2_manifest:
                        c["stage2_handoff_label"] = s2_manifest[k].get("sampling_label", "unknown")
                        found = True
                        break
                if not found:
                    c["stage2_handoff_label"] = "not_in_manifest"

            # note
            notes = []
            if not ct_ok:
                notes.append("ct_missing")
            if not roi_ok:
                notes.append("roi_missing")
            if lesion_vol is None:
                notes.append("lesion_mask_missing")
            c["note"] = ";".join(notes) if notes else ""

            # 출력용 row 추가
            all_comp_rows.append({
                "patient_id":               pid,
                "role":                     "lesion",
                "group":                    grp,
                "component_id":             c["component_id"],
                "max_score":                round(c["max_score"], 4),
                "mean_score":               round(c["mean_score"], 4),
                "patch_count":              c["patch_count"],
                "z_span":                   c["z_span"],
                "position_bin_majority":    c["position_bin_majority"],
                "bbox_y0":                  c["bbox_y0"],
                "bbox_x0":                  c["bbox_x0"],
                "bbox_y1":                  c["bbox_y1"],
                "bbox_x1":                  c["bbox_x1"],
                "local_z_min":              c["local_z_min"],
                "local_z_max":              c["local_z_max"],
                "top_patch_local_z":        c["top_patch_local_z"],
                "lesion_overlap_ratio":     round(c["lesion_overlap_ratio"], 4),
                "lesion_near_flag":         int(c["lesion_near_flag"]),
                "vessel_overlap_ratio":     round(c["vessel_overlap_ratio"], 4),
                "vessel_dominant_flag":     int(c["vessel_dominant_flag"]),
                "boundary_touch_ratio":     round(c["boundary_touch_ratio"], 4),
                "chestwall_boundary_flag":  int(c["chestwall_boundary_flag"]),
                "peripheral_flag":          int(c["peripheral_flag"]),
                "hilar_proxy_flag":         int(c["hilar_proxy_flag"]),
                "component_source_label":   c["component_source_label"],
                "stage2_handoff_label":     c["stage2_handoff_label"],
                "note":                     c["note"],
            })

        # 환자 요약
        label_counts = {}
        for c in components:
            lb = c["component_source_label"]
            label_counts[lb] = label_counts.get(lb, 0) + 1

        sorted_comps = sorted(components, key=lambda x: -x["max_score"])
        top3 = [c["component_source_label"] for c in sorted_comps[:3]]

        patient_summaries.append({
            "patient_id":                pid,
            "role":                      "lesion",
            "group":                     grp,
            "n_components":              len(components),
            "n_lesion_hit":              label_counts.get("lesion_hit", 0),
            "n_lesion_near":             label_counts.get("lesion_near", 0),
            "n_vessel_dominant":         label_counts.get("vessel_dominant", 0),
            "n_boundary_chestwall":      label_counts.get("boundary_chestwall", 0),
            "n_vessel_boundary_mixed":   label_counts.get("vessel_boundary_mixed", 0),
            "n_hilar_proxy":             label_counts.get("hilar_mediastinal_proxy", 0),
            "n_other_fp":                label_counts.get("other_fp", 0),
            "max_score_component_label": sorted_comps[0]["component_source_label"] if sorted_comps else "none",
            "top3_component_labels":     "|".join(top3),
        })

    # ── holdout 접근 0 검증 ─────────────────────────────────────────────────────
    assert s2_holdout_access_count == 0, f"stage2_holdout 접근 감지: {s2_holdout_access_count}"

    # ── stage2 handoff label summary ────────────────────────────────────────────
    handoff_summary = {}
    for row in all_comp_rows:
        hl = row["stage2_handoff_label"]
        sl = row["component_source_label"]
        if hl not in handoff_summary:
            handoff_summary[hl] = {
                "label": hl, "n_candidates": 0,
                "lesion_hit_count": 0, "vessel_dominant_count": 0,
                "boundary_chestwall_count": 0, "vessel_boundary_mixed_count": 0,
                "other_fp_count": 0, "score_sum": 0.0, "patch_sum": 0, "zspan_sum": 0,
            }
        d = handoff_summary[hl]
        d["n_candidates"] += 1
        if sl == "lesion_hit":     d["lesion_hit_count"] += 1
        if sl == "vessel_dominant": d["vessel_dominant_count"] += 1
        if sl == "boundary_chestwall": d["boundary_chestwall_count"] += 1
        if sl == "vessel_boundary_mixed": d["vessel_boundary_mixed_count"] += 1
        if sl == "other_fp":       d["other_fp_count"] += 1
        d["score_sum"]  += row["max_score"]
        d["patch_sum"]  += row["patch_count"]
        d["zspan_sum"]  += row["z_span"]

    handoff_rows = []
    for d in handoff_summary.values():
        n = d["n_candidates"]
        handoff_rows.append({
            "label":                     d["label"],
            "n_candidates":              n,
            "lesion_hit_ratio":          round(d["lesion_hit_count"] / n, 4) if n > 0 else 0,
            "vessel_dominant_ratio":     round(d["vessel_dominant_count"] / n, 4) if n > 0 else 0,
            "boundary_chestwall_ratio":  round(d["boundary_chestwall_count"] / n, 4) if n > 0 else 0,
            "vessel_boundary_mixed_ratio": round(d["vessel_boundary_mixed_count"] / n, 4) if n > 0 else 0,
            "other_fp_ratio":            round(d["other_fp_count"] / n, 4) if n > 0 else 0,
            "mean_score":                round(d["score_sum"] / n, 4) if n > 0 else 0,
            "mean_patch_count":          round(d["patch_sum"] / n, 2) if n > 0 else 0,
            "mean_z_span":               round(d["zspan_sum"] / n, 2) if n > 0 else 0,
        })

    # ── source distribution summary ─────────────────────────────────────────────
    total_comps = len(all_comp_rows)
    label_dist = {}
    for row in all_comp_rows:
        lb = row["component_source_label"]
        label_dist[lb] = label_dist.get(lb, 0) + 1

    source_summary = {
        "step": "B1-F1",
        "threshold_p95": THRESHOLD_P95,
        "threshold_source": "p_b9_normal_val_threshold.json",
        "total_components": total_comps,
        "stage2_holdout_access": 0,
        "n_errors": len(error_rows),
        "label_distribution": {k: v for k, v in sorted(label_dist.items(), key=lambda x: -x[1])},
        "label_pct": {
            k: round(v / total_comps * 100, 2) if total_comps > 0 else 0
            for k, v in label_dist.items()
        },
        "note": (
            "component_source_label은 진단명이 아닌 FP 원인 분석용 heuristic proxy label임. "
            "GT 기반 확정 레이블 아님."
        ),
        "guardrails": {
            "score_modified": False,
            "model_rerun": False,
            "threshold_recalculated": False,
            "stage2_holdout_accessed": False,
            "gpu_used": False,
        },
    }

    # ── 파일 저장 ───────────────────────────────────────────────────────────────
    def write_csv(path, rows, fieldnames):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    comp_fields = [
        "patient_id", "role", "group", "component_id", "max_score", "mean_score",
        "patch_count", "z_span", "position_bin_majority",
        "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1",
        "local_z_min", "local_z_max", "top_patch_local_z",
        "lesion_overlap_ratio", "lesion_near_flag",
        "vessel_overlap_ratio", "vessel_dominant_flag",
        "boundary_touch_ratio", "chestwall_boundary_flag",
        "peripheral_flag", "hilar_proxy_flag",
        "component_source_label", "stage2_handoff_label", "note",
    ]
    pat_fields = [
        "patient_id", "role", "group", "n_components",
        "n_lesion_hit", "n_lesion_near", "n_vessel_dominant",
        "n_boundary_chestwall", "n_vessel_boundary_mixed",
        "n_hilar_proxy", "n_other_fp",
        "max_score_component_label", "top3_component_labels",
    ]
    s2_fields = [
        "label", "n_candidates",
        "lesion_hit_ratio", "vessel_dominant_ratio",
        "boundary_chestwall_ratio", "vessel_boundary_mixed_ratio", "other_fp_ratio",
        "mean_score", "mean_patch_count", "mean_z_span",
    ]
    err_fields = ["patient_id", "error"]

    write_csv(OUT_ROOT / "b1f1_component_source_audit.csv", all_comp_rows, comp_fields)
    write_csv(OUT_ROOT / "b1f1_patient_source_summary.csv", patient_summaries, pat_fields)
    write_csv(OUT_ROOT / "b1f1_stage2_handoff_label_summary.csv", handoff_rows, s2_fields)
    write_csv(OUT_ROOT / "b1f1_errors.csv", error_rows, err_fields)

    with open(OUT_ROOT / "b1f1_source_distribution_summary.json", "w", encoding="utf-8") as f:
        json.dump(source_summary, f, ensure_ascii=False, indent=2)

    # ── 보고서 ──────────────────────────────────────────────────────────────────
    fp_labels = {k: v for k, v in label_dist.items() if k not in ("lesion_hit", "lesion_near")}
    fp_total  = sum(fp_labels.values())

    report_lines = [
        "# B1-F1 FP Source Distribution Report",
        "",
        "## 1. 읽은 EfficientNet v4_20 기존 보고서",
        "",
    ]
    for rp in REFERENCE_REPORTS:
        full = PROJECT / rp
        status = "읽음" if full.exists() else "없음"
        report_lines.append(f"- [{status}] `{rp}`")

    report_lines += [
        "",
        "## 2. Fixed Threshold",
        "",
        f"- **threshold_p95 = {THRESHOLD_P95:.6f}**",
        f"- 출처: `p_b9_normal_val_threshold.json` (v4_20 branch 전용, 재계산 금지)",
        "",
        "## 3. 안전 조건 확인",
        "",
        f"- stage2_holdout 접근: **0회** (확인됨)",
        f"- score/model/threshold 수정: **없음**",
        f"- GPU 사용: **없음**",
        "",
        "## 4. 전체 Component Source 분포",
        "",
        f"- 전체 component 수: **{total_comps}**",
    ]
    for lb, cnt in sorted(label_dist.items(), key=lambda x: -x[1]):
        pct = cnt / total_comps * 100 if total_comps > 0 else 0
        report_lines.append(f"  - {lb}: {cnt} ({pct:.1f}%)")

    report_lines += [
        "",
        "> **중요**: `component_source_label`은 진단명이 아닌 FP 원인 분석용 heuristic proxy label임.",
        "> GT 기반 확정 레이블이 아님.",
        "",
        "## 5. Positive 후보 중 실제 Lesion 포함 비율",
        "",
    ]
    hit = label_dist.get("lesion_hit", 0)
    near = label_dist.get("lesion_near", 0)
    report_lines.append(f"- lesion_hit (직접 overlap): {hit} / {total_comps} = {hit/total_comps*100:.1f}%" if total_comps > 0 else "- 데이터 없음")
    report_lines.append(f"- lesion_near (dilate {LESION_DILATE_PX}px 이내): {near} / {total_comps} = {near/total_comps*100:.1f}%" if total_comps > 0 else "")

    report_lines += [
        "",
        "## 6. Hard Negative 후보 FP 원인 분포",
        "",
    ]
    for lb, cnt in sorted(fp_labels.items(), key=lambda x: -x[1]):
        pct = cnt / fp_total * 100 if fp_total > 0 else 0
        report_lines.append(f"- {lb}: {cnt} ({pct:.1f}%)")

    report_lines += [
        "",
        "## 7. Top-score Component 원인 분포",
        "(환자별 max_score component의 source label 분포)",
        "",
    ]
    top_dist = {}
    for ps in patient_summaries:
        lb = ps.get("max_score_component_label", "none")
        top_dist[lb] = top_dist.get(lb, 0) + 1
    for lb, cnt in sorted(top_dist.items(), key=lambda x: -x[1]):
        report_lines.append(f"- {lb}: {cnt}명")

    report_lines += [
        "",
        "## 8. Position_bin별 FP 원인 분포",
        "",
    ]
    pbin_fp = {}
    for row in all_comp_rows:
        if row["component_source_label"] in ("lesion_hit", "lesion_near"):
            continue
        pbin = row["position_bin_majority"]
        sl   = row["component_source_label"]
        pbin_fp.setdefault(pbin, {})
        pbin_fp[pbin][sl] = pbin_fp[pbin].get(sl, 0) + 1
    for pbin in sorted(pbin_fp.keys()):
        report_lines.append(f"**{pbin}**:")
        for sl, cnt in sorted(pbin_fp[pbin].items(), key=lambda x: -x[1]):
            report_lines.append(f"  - {sl}: {cnt}")

    # 결론
    vessel_total    = label_dist.get("vessel_dominant", 0)
    boundary_total  = label_dist.get("boundary_chestwall", 0)
    hilar_total     = label_dist.get("hilar_mediastinal_proxy", 0)
    mixed_total     = label_dist.get("vessel_boundary_mixed", 0)
    fp_rank = sorted(
        [("vessel", vessel_total), ("chestwall/boundary", boundary_total),
         ("hilar/mediastinal", hilar_total), ("mixed", mixed_total)],
        key=lambda x: -x[1],
    )

    report_lines += [
        "",
        "## 9. 결론",
        "",
        f"**FP 주원인 우선순위** (component 수 기준):",
    ]
    for rank, (lb, cnt) in enumerate(fp_rank, 1):
        pct = cnt / fp_total * 100 if fp_total > 0 else 0
        report_lines.append(f"  {rank}. {lb}: {cnt} ({pct:.1f}%)")

    report_lines += [
        "",
        "**2차 refiner가 배워야 할 것**: 위 FP 원인 분포에 따라 결정 필요.",
        "",
        "## 10. 다음 권장",
        "",
        "- 1차 vessel suppression: FP 주원인 분석 결과에 따라 유지/제외 결정",
        "- 2차 crop vessel mask aux channel 추가: vessel_dominant가 주요 FP라면 고려",
        "- chest-wall/boundary rule: boundary_chestwall 비율에 따라 유지/강화",
    ]

    with open(OUT_ROOT / "b1f1_fp_source_distribution_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # ── DONE marker ─────────────────────────────────────────────────────────────
    (OUT_ROOT / "DONE").write_text("B1-F1 complete\n")

    print(f"\n[DONE] 출력 루트: {OUT_ROOT}")
    print(f"  - component_source_audit.csv: {len(all_comp_rows)} rows")
    print(f"  - patient_source_summary.csv: {len(patient_summaries)} rows")
    print(f"  - stage2_handoff_label_summary.csv: {len(handoff_rows)} rows")
    print(f"  - errors.csv: {len(error_rows)} rows")
    print(f"  - source_distribution_summary.json")
    print(f"  - fp_source_distribution_report.md")
    print(f"  - DONE")
    print(f"\n[안전 확인]")
    print(f"  stage2_holdout_access: 0")
    print(f"  score/model/threshold 수정: 없음")


if __name__ == "__main__":
    main()
