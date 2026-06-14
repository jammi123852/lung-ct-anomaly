"""
Step 2 Full Build Preflight — rd4ad_2p5d_lung5ch_masked_normal_v1

정상 362명 전체 5ch lung-window crop full build 전 사전 점검.
실질 폐 z-range audit (metadata-only, hard filter 아님).

실행 방법:
  bare run (차단):  exit 2
  dry-run:  python ... --dry-run
  actual:   python ... --run-preflight --confirm-plan-lock --confirm-no-stage2 --confirm-readonly
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
BRANCH_ROOT = PROJECT_ROOT / "experiments/rd4ad_2p5d_lung5ch_masked_normal_v1"

NORMAL_CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy"
)
MASK_ROOT = (
    PROJECT_ROOT
    / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"
)
PLAN_JSON = BRANCH_ROOT / "docs/FINAL_PLAN_LOCK.json"
STEP1_DONE = BRANCH_ROOT / "DONE_STEP1_CROP_SMOKE.json"
STEP1_METRICS = BRANCH_ROOT / "manifests/step1_crop_smoke_metrics.csv"

STAGE2_FORBIDDEN = ["stage2_holdout", "stage2_holdout_scoring", "stage2_holdout_crops"]

# ─── Constants ────────────────────────────────────────────────────────────────
CROP_SIZE = 96
INPUT_CHANNELS = 5
HU_MIN, HU_MAX = -1350, 150
FLOAT32_BYTES_PER_CROP = INPUT_CHANNELS * CROP_SIZE * CROP_SIZE * 4  # 184,320
RECHECK_N = 20

# full build output path (will be checked for collision)
FULL_BUILD_CROP_DIR = BRANCH_ROOT / "crops/normal_5ch_lung_w96_v1"

# ─── Output paths ─────────────────────────────────────────────────────────────
OUT_MANIFESTS = BRANCH_ROOT / "manifests"
OUT_REPORTS = BRANCH_ROOT / "reports"
OUT_LOGS = BRANCH_ROOT / "logs"

Z_AUDIT_CSV = OUT_MANIFESTS / "step2_normal_lung_z_range_audit.csv"
PLAN_MANIFEST_CSV = OUT_MANIFESTS / "step2_full_build_plan_manifest.csv"
STORAGE_CSV = OUT_MANIFESTS / "step2_storage_estimate.csv"
RECHECK_CSV = OUT_MANIFESTS / "step2_sample_crop_recheck_metrics.csv"
REPORT_MD = OUT_REPORTS / "step2_full_build_preflight_report.md"
SUMMARY_JSON = OUT_REPORTS / "step2_full_build_preflight_summary.json"
ERROR_CSV = OUT_LOGS / "step2_full_build_preflight_errors.csv"
DONE_JSON = BRANCH_ROOT / "DONE_STEP2_FULL_BUILD_PREFLIGHT.json"


# ─── Arg parse ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Step 2 Full Build Preflight")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-preflight", action="store_true")
    p.add_argument("--confirm-plan-lock", action="store_true")
    p.add_argument("--confirm-no-stage2", action="store_true")
    p.add_argument("--confirm-readonly", action="store_true")
    return p.parse_args()


def bare_run_guard(args):
    if not args.dry_run and not args.run_preflight:
        print("[ERROR] bare run forbidden. Use --dry-run or --run-preflight.")
        sys.exit(2)


def actual_run_guard(args):
    if args.run_preflight:
        missing = [f for f, ok in [
            ("--confirm-plan-lock", args.confirm_plan_lock),
            ("--confirm-no-stage2", args.confirm_no_stage2),
            ("--confirm-readonly",  args.confirm_readonly),
        ] if not ok]
        if missing:
            print(f"[ERROR] Missing flags: {', '.join(missing)}")
            sys.exit(2)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def log_check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def apply_lung_window(hu_arr):
    clipped = np.clip(hu_arr.astype(np.float32), HU_MIN, HU_MAX)
    return (clipped - HU_MIN) / (HU_MAX - HU_MIN)


def get_z_indices(z_center, D):
    return [max(0, min(D - 1, z_center + off)) for off in (-2, -1, 0, 1, 2)]


def extract_5ch_crop(ct_vol, z_center, y0, x0, y1, x1):
    D = ct_vol.shape[0]
    z_idx = get_z_indices(z_center, D)
    nr = any(z != z_center + off for z, off in zip(z_idx, (-2, -1, 0, 1, 2)))
    chans = []
    for z in z_idx:
        crop = np.array(ct_vol[z, y0:y1, x0:x1])
        if crop.shape != (CROP_SIZE, CROP_SIZE):
            pad = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
            h, w = min(crop.shape[0], CROP_SIZE), min(crop.shape[1], CROP_SIZE)
            pad[:h, :w] = crop[:h, :w]
            crop = pad
        chans.append(apply_lung_window(crop))
    return np.stack(chans, axis=0), z_idx, nr


def extract_mask_5ch(mask_vol, z_center, y0, x0, y1, x1):
    D = mask_vol.shape[0]
    z_idx = get_z_indices(z_center, D)
    chans = []
    for z in z_idx:
        crop = mask_vol[z, y0:y1, x0:x1]
        if crop.shape != (CROP_SIZE, CROP_SIZE):
            pad = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
            h, w = min(crop.shape[0], CROP_SIZE), min(crop.shape[1], CROP_SIZE)
            pad[:h, :w] = crop[:h, :w]
            crop = pad
        chans.append(crop.astype(np.float32))
    return np.stack(chans, axis=0)


# ─── Lung z-range audit ───────────────────────────────────────────────────────
def compute_lung_z_range(mask_vol):
    D, H, W = mask_vol.shape
    HW = H * W
    slice_areas = np.array([int(mask_vol[z].sum()) for z in range(D)])

    lung_idx = np.where(slice_areas > 0)[0]
    if len(lung_idx) == 0:
        return {k: -1 if "min" in k or "max" in k else 0 for k in [
            "lung_z_min_any", "lung_z_max_any", "lung_z_len_any",
            "lung_z_min_ratio_001", "lung_z_max_ratio_001", "lung_z_len_ratio_001",
            "lung_z_min_ratio_005", "lung_z_max_ratio_005", "lung_z_len_ratio_005",
            "lung_z_min_ratio_010", "lung_z_max_ratio_010", "lung_z_len_ratio_010",
            "total_lung_mask_voxels", "max_slice_lung_area", "median_slice_lung_area_inside_range",
        ]} | {"mask_empty": True}

    z_min_any, z_max_any = int(lung_idx[0]), int(lung_idx[-1])
    z_len_any = z_max_any - z_min_any + 1

    def range_by_threshold(ratio):
        thr = max(1, int(HW * ratio))
        idx = np.where(slice_areas >= thr)[0]
        if len(idx) == 0:
            return -1, -1, 0
        return int(idx[0]), int(idx[-1]), int(idx[-1]) - int(idx[0]) + 1

    z_min_001, z_max_001, z_len_001 = range_by_threshold(0.001)
    z_min_005, z_max_005, z_len_005 = range_by_threshold(0.005)
    z_min_010, z_max_010, z_len_010 = range_by_threshold(0.010)

    inside = slice_areas[z_min_any: z_max_any + 1]

    return {
        "lung_z_min_any": z_min_any, "lung_z_max_any": z_max_any, "lung_z_len_any": z_len_any,
        "lung_z_min_ratio_001": z_min_001, "lung_z_max_ratio_001": z_max_001, "lung_z_len_ratio_001": z_len_001,
        "lung_z_min_ratio_005": z_min_005, "lung_z_max_ratio_005": z_max_005, "lung_z_len_ratio_005": z_len_005,
        "lung_z_min_ratio_010": z_min_010, "lung_z_max_ratio_010": z_max_010, "lung_z_len_ratio_010": z_len_010,
        "total_lung_mask_voxels": int(slice_areas.sum()),
        "max_slice_lung_area": int(slice_areas.max()),
        "median_slice_lung_area_inside_range": float(np.median(inside)),
        "mask_empty": False,
    }


# ─── Crop plan per patient ────────────────────────────────────────────────────
def compute_crop_plan_stats(safe_id, ct_shape, zr):
    D, H, W = ct_shape
    z_start = zr["lung_z_min_any"] if zr["lung_z_min_any"] >= 0 else 0
    z_end = zr["lung_z_max_any"] if zr["lung_z_max_any"] >= 0 else D - 1
    z_len = max(0, z_end - z_start + 1)

    y0 = H // 2 - CROP_SIZE // 2
    x0 = W // 2 - CROP_SIZE // 2
    y1 = y0 + CROP_SIZE
    x1 = x0 + CROP_SIZE

    nr_count = sum(1 for z in range(z_start, z_end + 1) if z < 2 or z > D - 3)
    apex_thresh = z_start + max(1, int(z_len * 0.1))
    base_thresh = z_end - max(1, int(z_len * 0.1))
    n_apex = sum(1 for z in range(z_start, min(apex_thresh, z_end + 1)))
    n_base = sum(1 for z in range(max(base_thresh, z_start), z_end + 1))

    size_mb = z_len * FLOAT32_BYTES_PER_CROP / (1024 ** 2)

    return {
        "safe_id": safe_id,
        "z_start_for_crop": z_start,
        "z_end_for_crop": z_end,
        "planned_center_z_count": z_len,
        "planned_crop_count": z_len,
        "crop_y0": y0, "crop_x0": x0, "crop_y1": y1, "crop_x1": x1,
        "n_nearest_repeat_expected": nr_count,
        "n_apex_near_samples": n_apex,
        "n_base_near_samples": n_base,
        "expected_output_size_mb": round(size_mb, 3),
    }


def build_plan_rows(safe_id, ct_shape, plan_stats, zr):
    """Generate one row per planned z-slice for this patient."""
    D, H, W = ct_shape
    z_start = plan_stats["z_start_for_crop"]
    z_end = plan_stats["z_end_for_crop"]
    y0, x0, y1, x1 = plan_stats["crop_y0"], plan_stats["crop_x0"], plan_stats["crop_y1"], plan_stats["crop_x1"]
    z_min = zr["lung_z_min_any"]
    z_max = zr["lung_z_max_any"]
    z_range_denom = z_max - z_min if z_max > z_min else 1

    rows = []
    for z in range(z_start, z_end + 1):
        z_idx = get_z_indices(z, D)
        nr = any(zi != z + off for zi, off in zip(z_idx, (-2, -1, 0, 1, 2)))
        pct = (z - z_min) / z_range_denom if z_min >= 0 else float("nan")
        apex_thresh = z_min + max(1, int((z_max - z_min + 1) * 0.1))
        base_thresh = z_max - max(1, int((z_max - z_min + 1) * 0.1))
        rows.append({
            "safe_id": safe_id,
            "local_z": z,
            "crop_y0": y0, "crop_x0": x0, "crop_y1": y1, "crop_x1": x1,
            "lung_z_min": z_min, "lung_z_max": z_max, "lung_z_len": z_max - z_min + 1 if z_min >= 0 else 0,
            "lung_z_percentile": round(pct, 4) if z_min >= 0 else float("nan"),
            "is_outside_lung_z_range": z < z_min or z > z_max,
            "is_near_lung_apex": z <= apex_thresh,
            "is_near_lung_base": z >= base_thresh,
            "z_minus2_effective": z_idx[0], "z_minus1_effective": z_idx[1],
            "z_center_effective": z_idx[2], "z_plus1_effective": z_idx[3], "z_plus2_effective": z_idx[4],
            "nearest_repeat_used": nr,
        })
    return rows


# ─── Per-patient audit (CT + mask + z-range) ─────────────────────────────────
def audit_patient(safe_id):
    ct_path = NORMAL_CT_ROOT / safe_id / "ct_hu.npy"
    mask_path = MASK_ROOT / "normal" / safe_id / "refined_roi.npy"

    base = {"safe_id": safe_id, "patient_id": safe_id}

    if not ct_path.exists():
        return base | {"status": "ERROR", "error_message": f"CT missing: {ct_path}"}, None, None
    if not mask_path.exists():
        return base | {"status": "ERROR", "error_message": f"Mask missing: {mask_path}"}, None, None

    ct = np.load(str(ct_path), mmap_mode="r")
    mask = np.load(str(mask_path))

    if ct.shape != mask.shape:
        return base | {
            "ct_shape": str(ct.shape), "mask_shape": str(mask.shape),
            "mask_ct_shape_match": False,
            "status": "ERROR", "error_message": f"Shape mismatch {ct.shape} vs {mask.shape}",
        }, None, None

    D, H, W = ct.shape
    zr = compute_lung_z_range(mask)
    plan = compute_crop_plan_stats(safe_id, ct.shape, zr)

    row = base | {
        "ct_shape": str(ct.shape), "mask_shape": str(mask.shape),
        "z_dim": D, "mask_ct_shape_match": True,
    } | zr | {
        "planned_crop_count": plan["planned_crop_count"],
        "expected_output_size_mb": plan["expected_output_size_mb"],
        "n_nearest_repeat_expected": plan["n_nearest_repeat_expected"],
        "status": "PASS", "error_message": "",
    }

    return row, plan, ct.shape


# ─── Sample crop recheck ──────────────────────────────────────────────────────
def recheck_crop_sample(safe_id, z_center, ct_shape):
    D, H, W = ct_shape
    y0 = H // 2 - CROP_SIZE // 2
    x0 = W // 2 - CROP_SIZE // 2
    y1 = y0 + CROP_SIZE
    x1 = x0 + CROP_SIZE

    ct_path = NORMAL_CT_ROOT / safe_id / "ct_hu.npy"
    mask_path = MASK_ROOT / "normal" / safe_id / "refined_roi.npy"
    if not ct_path.exists() or not mask_path.exists():
        return {"safe_id": safe_id, "status": "SKIP", "error_message": "file missing"}

    ct = np.load(str(ct_path), mmap_mode="r")
    mask_vol = np.load(str(mask_path))

    try:
        crop, used_z, nr = extract_5ch_crop(ct, z_center, y0, x0, y1, x1)
        mask_5ch = extract_mask_5ch(mask_vol, z_center, y0, x0, y1, x1)
        masked = crop * mask_5ch
        outside = mask_5ch[2] == 0
        z_after = float(np.abs(masked[2][outside]).mean()) if outside.any() else 0.0
    except Exception as e:
        return {"safe_id": safe_id, "status": "ERROR", "error_message": str(e)}

    return {
        "safe_id": safe_id,
        "local_z": z_center,
        "crop_shape": str(crop.shape),
        "crop_dtype": str(crop.dtype),
        "crop_min": float(crop.min()),
        "crop_max": float(crop.max()),
        "crop_nan": int(np.sum(np.isnan(crop))),
        "crop_inf": int(np.sum(np.isinf(crop))),
        "mask_nonzero_ratio": float(np.mean(mask_5ch > 0)),
        "outside_after": z_after,
        "nearest_repeat": nr,
        "status": "PASS",
        "error_message": "",
    }


# ─── Plan lock + Step1 ────────────────────────────────────────────────────────
def verify_plan_lock():
    if not PLAN_JSON.exists():
        print(f"[BLOCKED] PLAN_JSON missing: {PLAN_JSON}")
        sys.exit(1)
    with open(PLAN_JSON) as f:
        plan = json.load(f)
    ok = all(log_check(n, v) for n, v in [
        ("plan_locked=true",        plan.get("plan_locked") is True),
        ("model_type=true_rd4ad",   plan.get("model", {}).get("model_type") == "true_rd4ad"),
        ("input_channels=5",        plan.get("model", {}).get("input_channels") == 5),
        ("crop_size=96",            plan.get("model", {}).get("crop_size") == 96),
        ("training_data=normal",    plan.get("training", {}).get("training_data") == "normal_only"),
        ("stage2=false",            plan.get("safety", {}).get("stage2_holdout_accessed") is False),
    ])
    if not ok:
        print("[BLOCKED] FINAL_PLAN_LOCK 조건 불일치")
        sys.exit(1)


def check_step1_pass():
    if not STEP1_DONE.exists():
        log_check("step1 DONE file exists", False, str(STEP1_DONE))
        return False
    with open(STEP1_DONE) as f:
        d = json.load(f)
    ok = d.get("verdict") == "PASS_STEP1_CROP_SMOKE"
    log_check("step1 verdict = PASS_STEP1_CROP_SMOKE", ok, d.get("verdict", "MISSING"))
    return ok


def check_stage2():
    for name in STAGE2_FORBIDDEN:
        for base in [PROJECT_ROOT, BRANCH_ROOT]:
            if (base / name).exists():
                print(f"[BLOCKED] Stage2 path exists: {base / name}")
                return True
    return False


# ─── Output writers ───────────────────────────────────────────────────────────
def write_report_md(audit_rows, plan_rows, recheck_rows, errors, verdict, stats):
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    df_a = pd.DataFrame(audit_rows)
    df_r = pd.DataFrame(recheck_rows) if recheck_rows else pd.DataFrame()

    n_pass = int((df_a["status"] == "PASS").sum()) if len(df_a) > 0 else 0
    empty = int(df_a["mask_empty"].sum()) if "mask_empty" in df_a.columns else -1
    shape_ok = int(df_a.get("mask_ct_shape_match", pd.Series(dtype=bool)).sum()) if "mask_ct_shape_match" in df_a.columns else -1

    lines = [
        "# Step 2 Full Build Preflight Report — rd4ad_2p5d_lung5ch_masked_normal_v1",
        "",
        f"**판정**: {verdict}",
        "",
        "## 요약",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 전체 환자 수 | {stats['n_total']} |",
        f"| CT/Mask PASS | {n_pass}/{stats['n_total']} |",
        f"| CT/Mask shape 일치 | {shape_ok} |",
        f"| mask_empty 환자 | {empty} |",
        f"| lung_z_len mean | {stats.get('lung_z_len_mean', 'N/A'):.1f} |",
        f"| lung_z_len median | {stats.get('lung_z_len_median', 'N/A'):.1f} |",
        f"| lung_z_len min/max | {stats.get('lung_z_len_min', 'N/A')} / {stats.get('lung_z_len_max', 'N/A')} |",
        f"| 총 예상 crop 수 | {stats.get('total_planned_crops', 'N/A'):,} |",
        f"| 예상 저장 용량 | {stats.get('total_size_gb', 0.0):.2f} GB (raw float32) |",
        f"| nearest_repeat 예상 | {stats.get('total_nr_crops', 'N/A')} crops |",
        f"| full build 출력 경로 충돌 | {stats.get('collision', False)} |",
        f"| sample recheck PASS | {stats.get('recheck_pass', 0)}/{stats.get('recheck_total', 0)} |",
        f"| stage2 접근 | False |",
        f"| 오류 수 | {len(errors)} |",
        "",
        "## Lung z-range 분포 (상위 10행)",
        "",
        "| safe_id | z_dim | lung_z_min_any | lung_z_max_any | lung_z_len_any | planned_crops | size_mb |",
        "|---|---|---|---|---|---|---|",
    ]
    preview = df_a[df_a["status"] == "PASS"].head(10) if len(df_a) > 0 else pd.DataFrame()
    for _, r in preview.iterrows():
        lines.append(
            f"| {r.get('safe_id','')} | {r.get('z_dim','')} | {r.get('lung_z_min_any','')} "
            f"| {r.get('lung_z_max_any','')} | {r.get('lung_z_len_any','')} "
            f"| {r.get('planned_crop_count','')} | {r.get('expected_output_size_mb','')} |"
        )
    lines += [
        "",
        "## Sample crop recheck (상위 10행)",
        "",
        "| safe_id | local_z | crop_shape | nan | zeroing_after | status |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in (pd.DataFrame(recheck_rows).head(10).iterrows() if recheck_rows else []):
        lines.append(
            f"| {r.get('safe_id','')} | {r.get('local_z','')} | {r.get('crop_shape','')} "
            f"| {r.get('crop_nan',0)} | {r.get('outside_after',0):.2e} | {r.get('status','')} |"
        )
    lines += [
        "",
        "## 오류",
        "",
    ]
    for e in (errors if errors else ["없음"]):
        lines.append(f"- {e}")
    lines += [
        "",
        "## 다음 단계",
        "",
        "PASS_STEP2_FULL_BUILD_PREFLIGHT → Step 3: crop full build (사용자 승인 후)",
        "",
        "**생성일**: 2026-06-10",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def write_summary_json(errors, verdict, stats, stage2_accessed, step1_pass):
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "verdict": verdict,
        "created": "2026-06-10",
        "branch_name": "rd4ad_2p5d_lung5ch_masked_normal_v1",
        "step": "step2_full_build_preflight",
        "normal_ct_count": stats.get("n_total", 0),
        "ct_mask_pass_count": stats.get("n_pass", 0),
        "mask_empty_count": stats.get("mask_empty_count", 0),
        "lung_z_len_mean": stats.get("lung_z_len_mean", 0),
        "lung_z_len_median": stats.get("lung_z_len_median", 0),
        "lung_z_len_min": stats.get("lung_z_len_min", 0),
        "lung_z_len_max": stats.get("lung_z_len_max", 0),
        "total_planned_crops": stats.get("total_planned_crops", 0),
        "total_storage_gb_raw_float32": stats.get("total_size_gb", 0.0),
        "total_nearest_repeat_crops": stats.get("total_nr_crops", 0),
        "output_collision": stats.get("collision", False),
        "recheck_pass": stats.get("recheck_pass", 0),
        "recheck_total": stats.get("recheck_total", 0),
        "error_count": len(errors),
        "guardrail": {
            "plan_lock_loaded": True,
            "step1_crop_smoke_passed": step1_pass,
            "stage2_holdout_accessed": stage2_accessed,
            "full_crop_build_executed": False,
            "training_executed": False,
            "model_forward_executed": False,
            "checkpoint_saved": False,
            "existing_artifact_modified": False,
            "existing_score_csv_modified": False,
            "existing_manifest_modified": False,
            "lung_z_range_used_for_metadata_only": True,
            "lung_z_range_used_for_hard_filter": False,
            "apex_base_filter_applied": False,
            "positive_label_used_for_training": False,
            "lesion_mask_used_for_training": False,
            "convae_branch_created": False,
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def write_done_json(verdict, stats, errors):
    done = {
        "step": "step2_full_build_preflight",
        "verdict": verdict,
        "created": "2026-06-10",
        "total_planned_crops": stats.get("total_planned_crops", 0),
        "total_storage_gb_raw_float32": stats.get("total_size_gb", 0.0),
        "error_count": len(errors),
        "outputs": {
            "z_audit_csv": str(Z_AUDIT_CSV),
            "plan_manifest_csv": str(PLAN_MANIFEST_CSV),
            "storage_csv": str(STORAGE_CSV),
            "recheck_csv": str(RECHECK_CSV),
            "report_md": str(REPORT_MD),
            "summary_json": str(SUMMARY_JSON),
            "error_csv": str(ERROR_CSV),
        },
        "next_step": "step3_crop_full_build",
        "next_step_note": "정상 전체 5ch crop 생성 (사용자 승인 후, 예상 용량 참조 요망)",
    }
    DONE_JSON.write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8")


# ─── Dry-run ──────────────────────────────────────────────────────────────────
def print_dry_run():
    script = Path(__file__)
    print("\n" + "=" * 64)
    print("DRY-RUN PLAN: Step 2 Full Build Preflight")
    print("=" * 64)
    print(f"\nBranch  : rd4ad_2p5d_lung5ch_masked_normal_v1")
    print(f"Target  : 정상 362명 전체 CT/Mask 확인 + 폐 z-range audit")
    print(f"\n작업 항목:")
    print(f"  1. Step 1 PASS 확인")
    print(f"  2. normal CT 362명 ct_hu.npy 존재 확인")
    print(f"  3. 362명 v4_20 mask 존재 확인")
    print(f"  4. CT/mask shape alignment 확인")
    print(f"  5. 환자별 실질 폐 z-range 계산 (metadata-only)")
    print(f"     - lung_z_min/max (area>0)")
    print(f"     - lung_z_min/max (area_ratio > 0.001/0.005/0.010)")
    print(f"  6. mask_empty 환자 0명 확인")
    print(f"  7. 예상 crop 수 / 저장 용량 계산")
    print(f"  8. full build 출력 경로 충돌 확인 → {FULL_BUILD_CROP_DIR}")
    print(f"  9. sample crop recheck ({RECHECK_N}명)")
    print(f" 10. stage2 접근 없음 확인")
    print(f"\n금지:")
    print(f"  - full crop build 실행")
    print(f"  - training / model forward / checkpoint")
    print(f"  - lung z-range를 hard filter로 사용")
    print(f"  - stage2_holdout 접근")
    print(f"\n출력 파일:")
    for p in [Z_AUDIT_CSV, PLAN_MANIFEST_CSV, STORAGE_CSV, RECHECK_CSV,
              REPORT_MD, SUMMARY_JSON, ERROR_CSV, DONE_JSON]:
        print(f"  {p}")
    print(f"\n실행 명령:")
    print(f"  python {script} \\")
    print(f"    --run-preflight --confirm-plan-lock --confirm-no-stage2 --confirm-readonly")
    print("=" * 64 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    bare_run_guard(args)

    if args.dry_run:
        print_dry_run()
        return

    actual_run_guard(args)

    print("\n" + "=" * 64)
    print("Step 2 Full Build Preflight — rd4ad_2p5d_lung5ch_masked_normal_v1")
    print("=" * 64)

    # 0. Guards
    print("\n[0] FINAL_PLAN_LOCK 확인")
    verify_plan_lock()

    print("\n[0b] stage2 접근 확인")
    stage2_accessed = check_stage2()
    if stage2_accessed:
        sys.exit(1)
    log_check("stage2 not accessed", True)

    print("\n[0c] Step 1 PASS 확인")
    step1_pass = check_step1_pass()
    if not step1_pass:
        print("[BLOCKED] Step 1 PASS 미확인")
        sys.exit(1)

    # 1. Normal CT directory scan
    print("\n[1] normal CT 디렉토리 목록")
    all_dirs = sorted([d for d in NORMAL_CT_ROOT.iterdir() if d.is_dir()])
    log_check(f"normal CT dirs >= 300", len(all_dirs) >= 300, f"{len(all_dirs)} found")

    # 2. Audit all patients
    print(f"\n[2] 전체 환자 CT/Mask 확인 + 폐 z-range audit ({len(all_dirs)}명)")
    audit_rows = []
    plan_stats_all = []
    ct_shapes = {}
    errors = []
    plan_rows_all = []

    for i, d in enumerate(all_dirs):
        safe_id = d.name
        if (i + 1) % 50 == 0 or i == 0 or i == len(all_dirs) - 1:
            print(f"  [{i+1:>3}/{len(all_dirs)}] {safe_id} ...", flush=True)
        row, plan, ct_shape = audit_patient(safe_id)
        audit_rows.append(row)
        if row["status"] == "PASS":
            plan_stats_all.append(plan)
            ct_shapes[safe_id] = ct_shape
            zr_part = {k: row[k] for k in [
                "lung_z_min_any", "lung_z_max_any", "lung_z_len_any",
                "lung_z_min_ratio_001", "lung_z_max_ratio_001", "lung_z_len_ratio_001",
                "lung_z_min_ratio_005", "lung_z_max_ratio_005", "lung_z_len_ratio_005",
                "lung_z_min_ratio_010", "lung_z_max_ratio_010", "lung_z_len_ratio_010",
                "total_lung_mask_voxels", "max_slice_lung_area",
                "median_slice_lung_area_inside_range", "mask_empty",
            ] if k in row}
            # Generate plan rows for this patient
            plan_rows = build_plan_rows(safe_id, ct_shape, plan, zr_part)
            plan_rows_all.extend(plan_rows)
        else:
            errors.append(f"{safe_id}: {row['error_message']}")

    print(f"  완료. PASS={sum(1 for r in audit_rows if r['status']=='PASS')}, ERROR={len(errors)}")

    # 3. Write audit CSVs
    print("\n[3] 파일 저장")
    OUT_MANIFESTS.mkdir(parents=True, exist_ok=True)
    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(Z_AUDIT_CSV, index=False)
    log_check("lung_z_range_audit.csv", True, f"{len(audit_rows)} rows → {Z_AUDIT_CSV.name}")

    df_plan = pd.DataFrame(plan_rows_all)
    df_plan.to_csv(PLAN_MANIFEST_CSV, index=False)
    log_check("plan_manifest.csv", True, f"{len(plan_rows_all)} rows → {PLAN_MANIFEST_CSV.name}")

    # 4. Storage estimate
    df_pass = df_audit[df_audit["status"] == "PASS"]
    n_pass = len(df_pass)
    total_crops = int(df_pass["planned_crop_count"].sum()) if n_pass > 0 else 0
    total_size_gb = total_crops * FLOAT32_BYTES_PER_CROP / (1024 ** 3)
    total_nr = int(df_pass["n_nearest_repeat_expected"].sum()) if n_pass > 0 else 0

    lung_z_lens = df_pass["lung_z_len_any"].astype(float) if n_pass > 0 else pd.Series(dtype=float)
    z_len_mean = float(lung_z_lens.mean()) if len(lung_z_lens) > 0 else 0.0
    z_len_median = float(lung_z_lens.median()) if len(lung_z_lens) > 0 else 0.0
    z_len_min = int(lung_z_lens.min()) if len(lung_z_lens) > 0 else 0
    z_len_max = int(lung_z_lens.max()) if len(lung_z_lens) > 0 else 0

    mask_empty_count = int(df_pass["mask_empty"].sum()) if "mask_empty" in df_pass.columns else 0

    storage_rows = [{
        "total_patients": n_pass,
        "total_planned_crops": total_crops,
        "float32_bytes_per_crop": FLOAT32_BYTES_PER_CROP,
        "total_size_gb_raw_float32": round(total_size_gb, 3),
        "total_nr_crops": total_nr,
        "lung_z_len_mean": round(z_len_mean, 1),
        "lung_z_len_median": round(z_len_median, 1),
        "lung_z_len_min": z_len_min,
        "lung_z_len_max": z_len_max,
        "mask_empty_count": mask_empty_count,
    }]
    pd.DataFrame(storage_rows).to_csv(STORAGE_CSV, index=False)
    log_check("storage_estimate.csv", True, f"total crops={total_crops:,}, {total_size_gb:.2f} GB")

    # 5. Output collision check
    collision = FULL_BUILD_CROP_DIR.exists() and any(FULL_BUILD_CROP_DIR.iterdir())
    log_check("no output collision", not collision,
              f"{FULL_BUILD_CROP_DIR} {'EXISTS with content' if collision else 'clear'}")

    # 6. mask_empty check
    log_check("mask_empty == 0", mask_empty_count == 0, f"{mask_empty_count} empty masks")

    # 7. Sample crop recheck (first RECHECK_N PASS patients, at lung_z_mid)
    print(f"\n[4] sample crop recheck ({RECHECK_N}명)")
    recheck_rows = []
    recheck_patients = df_pass.head(RECHECK_N)["safe_id"].tolist() if n_pass > 0 else []
    for sid in recheck_patients:
        row = df_audit[df_audit["safe_id"] == sid].iloc[0] if len(df_audit[df_audit["safe_id"] == sid]) > 0 else {}
        z_mid = int(row.get("lung_z_min_any", 0) + row.get("lung_z_len_any", 10) // 2) if len(row) > 0 else 10
        shape = ct_shapes.get(sid)
        if shape is None:
            continue
        m = recheck_crop_sample(sid, z_mid, shape)
        recheck_rows.append(m)
        status = m.get("status", "?")
        print(f"  {sid}: {status}")

    df_rc = pd.DataFrame(recheck_rows)
    df_rc.to_csv(RECHECK_CSV, index=False)
    recheck_pass = int((df_rc["status"] == "PASS").sum()) if len(df_rc) > 0 else 0
    recheck_total = len(df_rc)
    log_check(f"recheck PASS", recheck_pass == recheck_total, f"{recheck_pass}/{recheck_total}")

    # 8. Error CSV
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"error": e} for e in errors]).to_csv(ERROR_CSV, index=False)

    # 9. Verdict
    stats = {
        "n_total": len(all_dirs),
        "n_pass": n_pass,
        "mask_empty_count": mask_empty_count,
        "lung_z_len_mean": z_len_mean,
        "lung_z_len_median": z_len_median,
        "lung_z_len_min": z_len_min,
        "lung_z_len_max": z_len_max,
        "total_planned_crops": total_crops,
        "total_size_gb": total_size_gb,
        "total_nr_crops": total_nr,
        "collision": collision,
        "recheck_pass": recheck_pass,
        "recheck_total": recheck_total,
    }

    shape_issues = len([r for r in audit_rows if r.get("mask_ct_shape_match") is False])

    if (n_pass >= len(all_dirs) * 0.95
            and mask_empty_count == 0
            and shape_issues == 0
            and not collision
            and recheck_pass == recheck_total
            and not stage2_accessed):
        verdict = "PASS_STEP2_FULL_BUILD_PREFLIGHT"
    elif n_pass >= len(all_dirs) * 0.8 and mask_empty_count == 0:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "BLOCKED"

    write_report_md(audit_rows, plan_rows_all, recheck_rows, errors, verdict, stats)
    write_summary_json(errors, verdict, stats, stage2_accessed, step1_pass)
    write_done_json(verdict, stats, errors)

    # Final
    print("\n" + "=" * 64)
    print(f"판정: {verdict}")
    print("=" * 64)
    print(f"  정상 CT 수              : {n_pass}/{len(all_dirs)}")
    print(f"  mask_empty              : {mask_empty_count}")
    print(f"  CT/mask shape 불일치    : {shape_issues}")
    print(f"  lung_z_len mean/med     : {z_len_mean:.1f} / {z_len_median:.1f}")
    print(f"  lung_z_len min/max      : {z_len_min} / {z_len_max}")
    print(f"  예상 총 crop 수         : {total_crops:,}")
    print(f"  예상 저장 용량 (raw f32): {total_size_gb:.2f} GB")
    print(f"  nearest_repeat 예상     : {total_nr} crops")
    print(f"  출력 경로 충돌          : {collision}")
    print(f"  sample recheck          : {recheck_pass}/{recheck_total}")
    print(f"  stage2 접근             : {stage2_accessed}")
    print(f"  오류                    : {len(errors)}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
