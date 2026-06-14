#!/usr/bin/env python3
"""
B1-F1b: peripheral_fp component에 B1-E13 vessel mask 적용 → 혈관 비중 계산

입력:  b1f1a_component_source_audit.csv (peripheral_fp만)
방식:  axial 10-slice MIP, intensity p85 + tophat p85 + vesselness(frangi) p85 union
범위:  component top-z 슬라이스 1개 기준으로 vessel mask 계산
대상:  stage1_dev 전체 (holdout 접근 금지)

실행:
  --dry-run   계획 출력 (파일 생성 없음)
  --real      실제 계산 및 저장
"""
import sys

if __name__ == "__main__" and len(sys.argv) < 2:
    print("[ERROR] bare-run guard: --dry-run 또는 --real 필요", file=sys.stderr)
    sys.exit(2)

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from skimage.filters import frangi
from skimage.morphology import white_tophat, disk, remove_small_objects

ALLOW_REAL = "--real" in sys.argv

# ─── 경로 ─────────────────────────────────────────────────────────────────────
PROJECT = Path("/home/jinhy/project/lung-ct-anomaly")

B1F1A_CSV = (
    PROJECT
    / "outputs/position-aware-padim-v1/efficientnet_v4_20_fp_source_audit"
    / "b1f1a_fast_fp_source_distribution_v1/b1f1a_component_source_audit.csv"
)
STAGE_SPLIT_CSV = (
    PROJECT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"
)
NROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy"
)
ROI_BASE = (
    PROJECT
    / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1/lesion"
)
OUT_ROOT = (
    PROJECT
    / "outputs/position-aware-padim-v1/efficientnet_v4_20_fp_source_audit"
    / "b1f1b_peripheral_fp_vessel_ratio_v1"
)

# ─── 파라미터 (B1-E13 동일) ──────────────────────────────────────────────────
MIP_HALF       = 5          # axial 10-slice MIP (±5)
SIGMAS         = (0.5, 1.0, 1.5, 2.0)
TOPHAT_RADIUS  = 10
PERCENTILE     = 85.0
MIN_AREA       = 10
VESSEL_DOMINANT = 0.25


# ══════════════════════════════════════════════════════════════════════════════
# 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_holdout():
    s = set()
    with open(STAGE_SPLIT_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["stage_split"] == "stage2_holdout":
                s.add(r["patient_id"])
    return s


def load_safe_id_map():
    """patient_id → safe_id"""
    m = {}
    with open(STAGE_SPLIT_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            m[r["patient_id"]] = r["safe_id"]
    return m


def load_peripheral_comps():
    rows = []
    with open(B1F1A_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["component_source_label"] == "peripheral_fp":
                rows.append(r)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Vessel mask (B1-E13 동일)
# ══════════════════════════════════════════════════════════════════════════════

def lung_window(arr, level=-600, width=1500):
    lo, hi = level - width / 2, level + width / 2
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def hu_normalize(arr):
    return np.clip((arr.astype(np.float32) + 1000.0) / 1600.0, 0.0, 1.0)


def compute_vessel_mask(ct_vol, roi_vol, local_z):
    """
    axial 10-slice MIP → intensity p85 | tophat p85 | vesselness p85 union
    반환: 2D bool mask (slice 해상도)
    """
    n = ct_vol.shape[0]
    z_lo = max(0, local_z - MIP_HALF)
    z_hi = min(n, local_z + MIP_HALF)

    slab     = ct_vol[z_lo:z_hi].astype(np.float32)
    mip      = np.max(slab, axis=0)

    roi_slab = roi_vol[z_lo:z_hi]
    valid    = np.any(roi_slab > 0, axis=0)

    if not valid.any():
        return np.zeros(mip.shape, dtype=bool)

    # intensity p85
    lw   = lung_window(mip)
    t1   = float(np.percentile(lw[valid], PERCENTILE))
    m1   = (lw > t1) & valid

    # tophat p85
    hn   = hu_normalize(mip)
    th   = white_tophat(hn, disk(TOPHAT_RADIUS))
    t2   = float(np.percentile(th[valid], PERCENTILE))
    m2   = (th > t2) & valid

    # vesselness (frangi) p85
    fr   = frangi(hn, sigmas=SIGMAS, black_ridges=False)
    t3   = float(np.percentile(fr[valid], PERCENTILE))
    m3   = (fr > t3) & valid

    union = m1 | m2 | m3
    if MIN_AREA > 0 and union.any():
        union = remove_small_objects(union.copy(), min_size=MIN_AREA)
    return union


def vessel_overlap_in_bbox(vessel_mask, y0, x0, y1, x1):
    h, w = vessel_mask.shape
    by0, bx0 = max(0, y0), max(0, x0)
    by1, bx1 = min(h, y1), min(w, x1)
    if by1 <= by0 or bx1 <= bx0:
        return 0.0, 0, 0
    crop = vessel_mask[by0:by1, bx0:bx1]
    total = crop.size
    vessel = int(crop.sum())
    return float(vessel / total) if total > 0 else 0.0, vessel, total


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    holdout    = load_holdout()
    sid_map    = load_safe_id_map()
    peri_comps = load_peripheral_comps()

    # 환자별로 묶기
    by_patient = defaultdict(list)
    for c in peri_comps:
        pid = c["patient_id"]
        if pid not in holdout:
            by_patient[pid].append(c)

    # unique (patient, top_z) 수
    unique_pz = set()
    for pid, comps in by_patient.items():
        for c in comps:
            unique_pz.add((pid, int(c["top_patch_local_z"])))

    print(f"[INFO] peripheral_fp components: {len(peri_comps)}")
    print(f"[INFO] 관련 환자: {len(by_patient)}명")
    print(f"[INFO] unique (patient, top_z): {len(unique_pz)}")
    print(f"[INFO] 예상 frangi 계산: {len(unique_pz)}회 × ~0.3초 ≈ {len(unique_pz)*0.3/60:.0f}분")
    print(f"\n[PLAN]")
    print(f"  방식: B1-E13 axial 10-slice MIP, intensity+tophat+vesselness p{PERCENTILE:.0f} union")
    print(f"  대상: peripheral_fp top-z 1개/component")
    print(f"  stage2_holdout 접근: False")
    print(f"  출력: {OUT_ROOT}")

    if not ALLOW_REAL:
        print("\n[READY] --real 로 실행하세요.")
        return

    # ── real ──────────────────────────────────────────────────────────────────
    if OUT_ROOT.exists():
        print(f"[ERROR] 출력 루트 이미 존재: {OUT_ROOT}", file=sys.stderr)
        sys.exit(1)
    OUT_ROOT.mkdir(parents=True)

    result_rows = []
    error_rows  = []
    n_done      = 0

    for pid, comps in by_patient.items():
        sid = sid_map.get(pid)
        if sid is None:
            error_rows.append({"patient_id": pid, "error": "safe_id 없음"})
            continue

        ct_path  = NROOT / sid / "ct_hu.npy"
        roi_path = ROI_BASE / sid / "refined_roi.npy"

        if not ct_path.exists():
            error_rows.append({"patient_id": pid, "error": "ct_hu.npy 없음"})
            continue
        if not roi_path.exists():
            error_rows.append({"patient_id": pid, "error": "refined_roi.npy 없음"})
            continue

        try:
            ct_vol  = np.load(ct_path,  mmap_mode="r")
            roi_vol = np.load(roi_path, mmap_mode="r")
        except Exception as e:
            error_rows.append({"patient_id": pid, "error": f"로드 실패: {e}"})
            continue

        # 환자 내 unique top_z별 vessel mask 캐시
        vessel_cache = {}

        for c in comps:
            top_z = int(c["top_patch_local_z"])
            y0    = int(c["bbox_y0"])
            x0    = int(c["bbox_x0"])
            y1    = int(c["bbox_y1"])
            x1    = int(c["bbox_x1"])

            # vessel mask: top_z 기준 (캐시 활용)
            if top_z not in vessel_cache:
                try:
                    vessel_cache[top_z] = compute_vessel_mask(ct_vol, roi_vol, top_z)
                except Exception as e:
                    vessel_cache[top_z] = None
                    error_rows.append({"patient_id": pid, "error": f"z={top_z} vessel mask 실패: {e}"})

            vm = vessel_cache[top_z]
            if vm is None:
                vr, vp, tp = 0.0, 0, 0
            else:
                vr, vp, tp = vessel_overlap_in_bbox(vm, y0, x0, y1, x1)

            result_rows.append({
                "patient_id":            pid,
                "component_id":          c["component_id"],
                "top_patch_local_z":     top_z,
                "bbox_y0":               y0,
                "bbox_x0":               x0,
                "bbox_y1":               y1,
                "bbox_x1":               x1,
                "max_score":             c["max_score"],
                "patch_count":           c["patch_count"],
                "position_bin_majority": c["position_bin_majority"],
                "stage2_handoff_label":  c["stage2_handoff_label"],
                "vessel_overlap_ratio":  round(vr, 4),
                "vessel_pixels":         vp,
                "bbox_pixels":           tp,
                "vessel_dominant_flag":  int(vr >= VESSEL_DOMINANT),
            })

        n_done += 1
        if n_done % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_done}/{len(by_patient)}] elapsed={elapsed:.0f}s")

    # ── 요약 ──────────────────────────────────────────────────────────────────
    total = len(result_rows)
    vessel_dominant_n = sum(1 for r in result_rows if r["vessel_dominant_flag"])
    ratios = [r["vessel_overlap_ratio"] for r in result_rows]

    summary = {
        "step": "B1-F1b",
        "source": "b1f1a peripheral_fp",
        "n_components": total,
        "vessel_method": "B1-E13 3-method union (intensity+tophat+vesselness p85)",
        "mip_slabs": f"top_z ±{MIP_HALF} = 10 slices",
        "vessel_dominant_threshold": VESSEL_DOMINANT,
        "vessel_dominant_n": vessel_dominant_n,
        "vessel_dominant_pct": round(vessel_dominant_n / total * 100, 2) if total else 0,
        "vessel_ratio_mean": round(float(np.mean(ratios)), 4) if ratios else 0,
        "vessel_ratio_median": round(float(np.median(ratios)), 4) if ratios else 0,
        "vessel_ratio_p25": round(float(np.percentile(ratios, 25)), 4) if ratios else 0,
        "vessel_ratio_p75": round(float(np.percentile(ratios, 75)), 4) if ratios else 0,
        "n_errors": len(error_rows),
        "elapsed_seconds": round(time.time() - t0, 1),
        "stage2_holdout_access": 0,
        "guardrails": {
            "score_modified": False,
            "model_rerun": False,
            "existing_results_modified": False,
        },
    }

    # ── 파일 저장 ──────────────────────────────────────────────────────────────
    def write_csv(path, rows, fields):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    res_fields = [
        "patient_id", "component_id", "top_patch_local_z",
        "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1",
        "max_score", "patch_count", "position_bin_majority",
        "stage2_handoff_label",
        "vessel_overlap_ratio", "vessel_pixels", "bbox_pixels",
        "vessel_dominant_flag",
    ]
    write_csv(OUT_ROOT / "b1f1b_peripheral_fp_vessel_ratio.csv", result_rows, res_fields)
    write_csv(OUT_ROOT / "b1f1b_errors.csv", error_rows, ["patient_id", "error"])

    with open(OUT_ROOT / "b1f1b_vessel_ratio_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── 보고서 ────────────────────────────────────────────────────────────────
    # position_bin별 vessel_dominant 비율
    pbin_stats = defaultdict(lambda: {"n": 0, "dominant": 0, "ratio_sum": 0.0})
    for r in result_rows:
        pb = r["position_bin_majority"]
        pbin_stats[pb]["n"] += 1
        pbin_stats[pb]["dominant"] += r["vessel_dominant_flag"]
        pbin_stats[pb]["ratio_sum"] += r["vessel_overlap_ratio"]

    lines = [
        "# B1-F1b Peripheral FP Vessel Ratio Report",
        "",
        f"- 분석 대상: peripheral_fp {total}개 components",
        f"- 방식: B1-E13 3-method union (intensity+tophat+vesselness p{PERCENTILE:.0f})",
        f"- top-z ±{MIP_HALF} axial MIP 기준",
        f"- vessel_dominant 기준: overlap_ratio ≥ {VESSEL_DOMINANT}",
        "",
        "## 전체 Vessel 비중 분포",
        "",
        f"- vessel_dominant (≥{VESSEL_DOMINANT}): **{vessel_dominant_n} / {total} ({summary['vessel_dominant_pct']:.1f}%)**",
        f"- mean vessel_overlap_ratio: {summary['vessel_ratio_mean']:.4f}",
        f"- median: {summary['vessel_ratio_median']:.4f}",
        f"- p25 / p75: {summary['vessel_ratio_p25']:.4f} / {summary['vessel_ratio_p75']:.4f}",
        "",
        "## Position_bin별 vessel_dominant 비율",
        "",
    ]
    for pb in sorted(pbin_stats):
        d = pbin_stats[pb]
        n, dom = d["n"], d["dominant"]
        mean_r = d["ratio_sum"] / n if n else 0
        lines.append(f"- **{pb}**: {dom}/{n} dominant ({dom/n*100:.1f}%), mean_ratio={mean_r:.3f}")

    lines += [
        "",
        "## Stage2 Handoff Label별 vessel_dominant 비율",
        "",
    ]
    hl_stats = defaultdict(lambda: {"n": 0, "dominant": 0})
    for r in result_rows:
        hl = r["stage2_handoff_label"]
        hl_stats[hl]["n"] += 1
        hl_stats[hl]["dominant"] += r["vessel_dominant_flag"]
    for hl, d in sorted(hl_stats.items()):
        n = d["n"]
        dom = d["dominant"]
        lines.append(f"- {hl}: {dom}/{n} vessel_dominant ({dom/n*100:.1f}% )" if n else f"- {hl}: 0")

    lines += [
        "",
        f"- 오류: {len(error_rows)}명",
        f"- 실행시간: {summary['elapsed_seconds']:.1f}초",
    ]

    with open(OUT_ROOT / "b1f1b_vessel_ratio_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    (OUT_ROOT / "DONE").write_text("B1-F1b complete\n")

    elapsed = time.time() - t0
    print(f"\n[DONE] elapsed={elapsed:.1f}s")
    print(f"  peripheral_fp_vessel_ratio.csv: {total} rows")
    print(f"  vessel_dominant (≥{VESSEL_DOMINANT}): {vessel_dominant_n} ({summary['vessel_dominant_pct']:.1f}%)")
    print(f"  mean vessel_overlap_ratio: {summary['vessel_ratio_mean']:.4f}")
    print(f"  errors: {len(error_rows)}")
    print(f"  stage2_holdout_access: 0  ✓")


if __name__ == "__main__":
    main()
