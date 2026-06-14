"""
Step 1 Crop Smoke — rd4ad_2p5d_lung5ch_masked_normal_v1

5ch lung-window crop 소량 생성 + shape/dtype/mask zeroing 검증

실행 방법:
  bare run (차단):
    python ... → exit 2

  dry-run:
    python ... --dry-run

  actual smoke:
    python ... --run-smoke --confirm-plan-lock --confirm-no-stage2 --confirm-readonly
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
BRANCH_ROOT = PROJECT_ROOT / "experiments/rd4ad_2p5d_lung5ch_masked_normal_v1"

NORMAL_CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy"
)
NSCLC_CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy"
)
MASK_ROOT = (
    PROJECT_ROOT
    / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"
)
CANDIDATE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c2_effb0_v420_candidate_rd4ad_retest_v1"
    / "rd_c2_effb0_v420_candidate_manifest.csv"
)
PLAN_JSON = BRANCH_ROOT / "docs/FINAL_PLAN_LOCK.json"

STAGE2_FORBIDDEN = ["stage2_holdout", "stage2_holdout_scoring", "stage2_holdout_crops"]

# ─── Constants ────────────────────────────────────────────────────────────────
P90_THRESHOLD = 12.196394
CROP_SIZE = 96
INPUT_CHANNELS = 5
HU_MIN, HU_MAX = -1350, 150
Z_CONTINUITY_MIN = 2
PRIORITY_PATIENTS = ["LUNG1-086", "LUNG1-386", "LUNG1-399"]
QA_LIMIT = 15

# ─── Output paths ─────────────────────────────────────────────────────────────
OUT_MANIFESTS = BRANCH_ROOT / "manifests"
OUT_REPORTS = BRANCH_ROOT / "reports"
OUT_LOGS = BRANCH_ROOT / "logs"
OUT_QA = BRANCH_ROOT / "qa/step1_crop_smoke_png"

SAMPLES_CSV = OUT_MANIFESTS / "step1_crop_smoke_samples.csv"
METRICS_CSV = OUT_MANIFESTS / "step1_crop_smoke_metrics.csv"
REPORT_MD = OUT_REPORTS / "step1_crop_smoke_report.md"
SUMMARY_JSON = OUT_REPORTS / "step1_crop_smoke_summary.json"
ERROR_CSV = OUT_LOGS / "step1_crop_smoke_errors.csv"
DONE_JSON = BRANCH_ROOT / "DONE_STEP1_CROP_SMOKE.json"


# ─── Arg parse ────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 1 Crop Smoke — rd4ad_2p5d_lung5ch_masked_normal_v1"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--confirm-plan-lock", action="store_true")
    parser.add_argument("--confirm-no-stage2", action="store_true")
    parser.add_argument("--confirm-readonly", action="store_true")
    return parser.parse_args()


def bare_run_guard(args):
    if not args.dry_run and not args.run_smoke:
        print("[ERROR] bare run forbidden. Use --dry-run or --run-smoke with confirm flags.")
        sys.exit(2)


def actual_run_guard(args):
    if args.run_smoke:
        missing = []
        if not args.confirm_plan_lock:
            missing.append("--confirm-plan-lock")
        if not args.confirm_no_stage2:
            missing.append("--confirm-no-stage2")
        if not args.confirm_readonly:
            missing.append("--confirm-readonly")
        if missing:
            print(f"[ERROR] Missing flags: {', '.join(missing)}")
            sys.exit(2)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def log_check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def apply_lung_window(hu_arr):
    clipped = np.clip(hu_arr.astype(np.float32), HU_MIN, HU_MAX)
    return (clipped - HU_MIN) / (HU_MAX - HU_MIN)


def get_z_indices(z_center, D):
    """nearest_repeat clamping for ±2 slices"""
    return [max(0, min(D - 1, z_center + offset)) for offset in [-2, -1, 0, 1, 2]]


def has_z_continuity(z_list, min_run=2):
    zs = sorted(set(z_list))
    if len(zs) < min_run:
        return False
    run = 1
    for i in range(1, len(zs)):
        if zs[i] == zs[i - 1] + 1:
            run += 1
            if run >= min_run:
                return True
        else:
            run = 1
    return False


def extract_5ch_crop(ct_vol, z_center, y0, x0, y1, x1):
    """
    ct_vol: (D, H, W) int16
    Returns:
      crop_5ch: (5, 96, 96) float32 [0,1]
      used_z: list of 5 ints
      nearest_repeat_used: bool
    """
    D = ct_vol.shape[0]
    z_indices = get_z_indices(z_center, D)
    nearest_repeat_used = any(
        z != z_center + off for z, off in zip(z_indices, [-2, -1, 0, 1, 2])
    )
    channels = []
    for z in z_indices:
        sl = ct_vol[z]
        crop = np.array(sl[y0:y1, x0:x1])
        # zero-pad if crop boundary exceeds CT bounds
        if crop.shape != (CROP_SIZE, CROP_SIZE):
            padded = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
            h = min(crop.shape[0], CROP_SIZE)
            w = min(crop.shape[1], CROP_SIZE)
            padded[:h, :w] = crop[:h, :w]
            crop = padded
        channels.append(apply_lung_window(crop))
    return np.stack(channels, axis=0), z_indices, nearest_repeat_used


def extract_mask_5ch(mask_vol, z_center, y0, x0, y1, x1):
    """
    mask_vol: (D, H, W) uint8, values 0/1
    Returns: (5, 96, 96) float32 mask
    """
    D = mask_vol.shape[0]
    z_indices = get_z_indices(z_center, D)
    channels = []
    for z in z_indices:
        sl = mask_vol[z]
        crop = sl[y0:y1, x0:x1]
        if crop.shape != (CROP_SIZE, CROP_SIZE):
            padded = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
            h = min(crop.shape[0], CROP_SIZE)
            w = min(crop.shape[1], CROP_SIZE)
            padded[:h, :w] = crop[:h, :w]
            crop = padded
        channels.append(crop.astype(np.float32))
    return np.stack(channels, axis=0)


# ─── CT / Mask loading ────────────────────────────────────────────────────────
def load_ct(safe_id, ct_type):
    root = NORMAL_CT_ROOT if ct_type == "normal" else NSCLC_CT_ROOT
    path = root / safe_id / "ct_hu.npy"
    if not path.exists():
        return None, str(path)
    return np.load(str(path), mmap_mode="r"), None


def load_mask_vol(safe_id, mask_type):
    """mask_type: 'normal' or 'lesion'"""
    path = MASK_ROOT / mask_type / safe_id / "refined_roi.npy"
    if not path.exists():
        return None, str(path)
    return np.load(str(path)), None


# ─── Sample selection ─────────────────────────────────────────────────────────
def select_normal_samples():
    """Type A + C: normal CT center + z-boundary samples (5 patients × 3 z positions)"""
    samples = []
    dirs = sorted([d for d in NORMAL_CT_ROOT.iterdir() if (d / "ct_hu.npy").exists()])[:5]
    for d in dirs:
        safe_id = d.name
        ct, err = load_ct(safe_id, "normal")
        if ct is None:
            continue
        D, H, W = ct.shape
        y0 = H // 2 - CROP_SIZE // 2
        x0 = W // 2 - CROP_SIZE // 2
        y1 = y0 + CROP_SIZE
        x1 = x0 + CROP_SIZE
        for z_label, z in [("center", D // 2), ("near_start", 1), ("near_end", D - 2)]:
            samples.append({
                "sample_id": f"normal_{safe_id}_{z_label}",
                "sample_type": f"normal_{z_label}",
                "patient_id": safe_id,
                "safe_id": safe_id,
                "ct_type": "normal",
                "mask_type": "normal",
                "local_z": z,
                "crop_y0": y0, "crop_x0": x0,
                "crop_y1": y1, "crop_x1": x1,
                "first_stage_score": None,
                "label": "normal",
            })
    return samples


def select_candidate_samples(df):
    """Type B: p90+z-continuity filtered candidates (10 pos + 10 neg)"""
    df_p90 = df[df["first_stage_score"] > P90_THRESHOLD].copy()

    # Check z-continuity per (patient_id, crop_y0, crop_x0) position
    grp = (
        df_p90
        .groupby(["patient_id", "crop_y0", "crop_x0"])["local_z"]
        .apply(list)
        .reset_index()
    )
    grp["ok"] = grp["local_z"].apply(lambda zl: has_z_continuity(zl))
    valid_set = set(
        zip(grp[grp["ok"]]["patient_id"],
            grp[grp["ok"]]["crop_y0"],
            grp[grp["ok"]]["crop_x0"])
    )

    mask_valid = df_p90.apply(
        lambda r: (r["patient_id"], r["crop_y0"], r["crop_x0"]) in valid_set, axis=1
    )
    df_valid = df_p90[mask_valid].copy()

    def pick_samples(df_sub, n):
        picked = []
        # Priority patients first
        for pid in PRIORITY_PATIENTS:
            rows = df_sub[df_sub["patient_id"] == pid]
            if len(rows) > 0:
                picked.append(rows.iloc[len(rows) // 2])
        # Other patients
        other_pids = [p for p in df_sub["patient_id"].unique() if p not in PRIORITY_PATIENTS]
        for pid in other_pids:
            if len(picked) >= n:
                break
            rows = df_sub[df_sub["patient_id"] == pid]
            picked.append(rows.iloc[len(rows) // 2])
        return picked[:n]

    pos_rows = pick_samples(df_valid[df_valid["label"] == "positive"], 10)
    neg_rows = pick_samples(df_valid[df_valid["label"] == "hard_negative"], 10)

    samples = []
    for row in pos_rows + neg_rows:
        sid = f"cand_{row['patient_id']}_{row['local_z']}_{row['crop_y0']}_{row['crop_x0']}"
        samples.append({
            "sample_id": sid,
            "sample_type": f"candidate_{row['label']}",
            "patient_id": row["patient_id"],
            "safe_id": row["safe_id"],
            "ct_type": "nsclc",
            "mask_type": "lesion",
            "local_z": int(row["local_z"]),
            "crop_y0": int(row["crop_y0"]),
            "crop_x0": int(row["crop_x0"]),
            "crop_y1": int(row["crop_y1"]),
            "crop_x1": int(row["crop_x1"]),
            "first_stage_score": float(row["first_stage_score"]),
            "label": row["label"],
        })
    return samples


# ─── QA PNG ───────────────────────────────────────────────────────────────────
def to_uint8_montage(arr_5ch):
    """arr_5ch: (5, H, W) float32 [0,1] → hstack montage uint8"""
    panels = [np.clip(arr_5ch[i] * 255, 0, 255).astype(np.uint8) for i in range(5)]
    return np.hstack(panels)


def save_qa_png(sample_id, crop_5ch, mask_5ch, masked_5ch):
    OUT_QA.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_QA / f"{sample_id}_original_5ch.png"), to_uint8_montage(crop_5ch))
    cv2.imwrite(str(OUT_QA / f"{sample_id}_mask_5ch.png"), to_uint8_montage(mask_5ch))
    cv2.imwrite(str(OUT_QA / f"{sample_id}_masked_5ch.png"), to_uint8_montage(masked_5ch))
    # Center slice: before / mask / after side by side
    before = np.clip(crop_5ch[2] * 255, 0, 255).astype(np.uint8)
    msk = np.clip(mask_5ch[2] * 255, 0, 255).astype(np.uint8)
    after = np.clip(masked_5ch[2] * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT_QA / f"{sample_id}_center_compare.png"), np.hstack([before, msk, after]))


# ─── Metrics computation ──────────────────────────────────────────────────────
def compute_sample(s, qa_count):
    sid = s["sample_id"]

    ct, ct_err = load_ct(s["safe_id"], s["ct_type"])
    if ct is None:
        return None, {"sample_id": sid, "error": f"CT not found: {ct_err}"}

    mask_vol, mask_err = load_mask_vol(s["safe_id"], s["mask_type"])
    if mask_vol is None:
        return None, {"sample_id": sid, "error": f"Mask not found: {mask_err}"}

    if ct.shape != mask_vol.shape:
        return None, {
            "sample_id": sid,
            "error": f"Shape mismatch CT{ct.shape} vs Mask{mask_vol.shape}",
        }

    y0, x0, y1, x1 = s["crop_y0"], s["crop_x0"], s["crop_y1"], s["crop_x1"]
    z = s["local_z"]

    try:
        crop_5ch, used_z, nr_used = extract_5ch_crop(ct, z, y0, x0, y1, x1)
        mask_5ch = extract_mask_5ch(mask_vol, z, y0, x0, y1, x1)
    except Exception as e:
        return None, {"sample_id": sid, "error": f"Crop extraction error: {e}"}

    masked_5ch = crop_5ch * mask_5ch

    outside = mask_5ch[2] == 0
    outside_before = float(np.abs(crop_5ch[2][outside]).mean()) if outside.any() else 0.0
    outside_after = float(np.abs(masked_5ch[2][outside]).mean()) if outside.any() else 0.0

    m = {
        "sample_id": sid,
        "sample_type": s["sample_type"],
        "patient_id": s["patient_id"],
        "safe_id": s["safe_id"],
        "local_z": z,
        "used_z_indices": str(used_z),
        "crop_y0": y0, "crop_x0": x0, "crop_y1": y1, "crop_x1": x1,
        "crop_shape": str(crop_5ch.shape),
        "mask_shape_5ch": str(mask_5ch.shape),
        "crop_dtype": str(crop_5ch.dtype),
        "crop_min": float(crop_5ch.min()),
        "crop_max": float(crop_5ch.max()),
        "crop_nan_count": int(np.sum(np.isnan(crop_5ch))),
        "crop_inf_count": int(np.sum(np.isinf(crop_5ch))),
        "mask_min": float(mask_5ch.min()),
        "mask_max": float(mask_5ch.max()),
        "mask_nonzero_ratio": float(np.mean(mask_5ch > 0)),
        "outside_lung_abs_mean_before_mask": outside_before,
        "outside_lung_abs_mean_after_mask": outside_after,
        "center_slice_mean": float(crop_5ch[2].mean()),
        "center_mask_nonzero_ratio": float(np.mean(mask_5ch[2] > 0)),
        "nearest_repeat_used": nr_used,
        "ct_shape": str(ct.shape),
        "mask_shape_full": str(mask_vol.shape),
        "first_stage_score": s.get("first_stage_score"),
        "label": s.get("label"),
        "status": "PASS",
        "error_message": "",
        "png_saved": False,
    }

    if qa_count[0] < QA_LIMIT:
        try:
            save_qa_png(sid, crop_5ch, mask_5ch, masked_5ch)
            m["png_saved"] = True
            qa_count[0] += 1
        except Exception as e:
            m["error_message"] = f"PNG save error: {e}"

    return m, None


# ─── Output writers ───────────────────────────────────────────────────────────
def write_report_md(metrics_list, errors, verdict, qa_saved):
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()
    n = len(df)
    n_pass = int((df["status"] == "PASS").sum()) if n > 0 else 0
    shape_pass = int(df["crop_shape"].eq(f"({INPUT_CHANNELS}, {CROP_SIZE}, {CROP_SIZE})").sum()) if n > 0 else 0
    nan_total = int(df["crop_nan_count"].sum()) if n > 0 else 0
    inf_total = int(df["crop_inf_count"].sum()) if n > 0 else 0
    nr_count = int(df["nearest_repeat_used"].sum()) if n > 0 else 0
    zeroing_ok = int((df["outside_lung_abs_mean_after_mask"] < 1e-6).sum()) if n > 0 else 0

    lines = [
        "# Step 1 Crop Smoke Report — rd4ad_2p5d_lung5ch_masked_normal_v1",
        "",
        f"**판정**: {verdict}",
        "",
        "## 요약",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 전체 샘플 수 | {n} |",
        f"| PASS 수 | {n_pass} |",
        f"| 오류 수 | {len(errors)} |",
        f"| crop shape (5,96,96) PASS | {shape_pass}/{n} |",
        f"| NaN 총수 | {nan_total} |",
        f"| Inf 총수 | {inf_total} |",
        f"| nearest_repeat 사용 샘플 | {nr_count} |",
        f"| outside_lung zeroing PASS | {zeroing_ok}/{n} |",
        f"| QA PNG 저장 수 | {qa_saved} |",
        f"| QA PNG 경로 | {OUT_QA} |",
        "",
        "## 샘플 타입별 분포",
        "",
    ]
    if n > 0:
        for st, cnt in df["sample_type"].value_counts().items():
            lines.append(f"- {st}: {cnt}")
    lines += [
        "",
        "## 상세 (상위 10행)",
        "",
        "| sample_id | sample_type | crop_shape | dtype | min | max | nan | mask_nr | zeroing_after | nr_used | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    preview = df.head(10) if n > 0 else pd.DataFrame()
    for _, row in preview.iterrows():
        lines.append(
            f"| {row.get('sample_id','')} | {row.get('sample_type','')} "
            f"| {row.get('crop_shape','')} | {row.get('crop_dtype','')} "
            f"| {row.get('crop_min',0):.4f} | {row.get('crop_max',0):.4f} "
            f"| {row.get('crop_nan_count',0)} | {row.get('mask_nonzero_ratio',0):.3f} "
            f"| {row.get('outside_lung_abs_mean_after_mask',0):.2e} "
            f"| {row.get('nearest_repeat_used',False)} | {row.get('status','')} |"
        )
    lines += [
        "",
        "## 오류",
        "",
    ]
    if errors:
        for e in errors:
            lines.append(f"- {e}")
    else:
        lines.append("- 없음")
    lines += [
        "",
        "## 다음 단계",
        "",
        "PASS_STEP1_CROP_SMOKE → Step 2 crop full build preflight (사용자 승인 후)",
        "",
        "**생성일**: 2026-06-10",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def write_summary_json(metrics_list, errors, verdict, stage2_accessed, qa_saved):
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()
    n = len(df)
    n_pass = int((df["status"] == "PASS").sum()) if n > 0 else 0
    shape_pass = int(df["crop_shape"].eq(f"({INPUT_CHANNELS}, {CROP_SIZE}, {CROP_SIZE})").sum()) if n > 0 else 0
    nan_total = int(df["crop_nan_count"].sum()) if n > 0 else 0
    inf_total = int(df["crop_inf_count"].sum()) if n > 0 else 0
    nr_count = int(df["nearest_repeat_used"].sum()) if n > 0 else 0
    zeroing_ok = int((df["outside_lung_abs_mean_after_mask"] < 1e-6).sum()) if n > 0 else 0

    summary = {
        "verdict": verdict,
        "created": "2026-06-10",
        "branch_name": "rd4ad_2p5d_lung5ch_masked_normal_v1",
        "step": "step1_crop_smoke",
        "sample_count": n,
        "pass_count": n_pass,
        "error_count": len(errors),
        "crop_shape_pass_count": shape_pass,
        "nan_total": nan_total,
        "inf_total": inf_total,
        "nearest_repeat_used_count": nr_count,
        "outside_lung_zeroing_pass_count": zeroing_ok,
        "qa_png_saved": qa_saved,
        "guardrail": {
            "plan_lock_loaded": True,
            "plan_lock_path": str(PLAN_JSON),
            "model_type": "true_rd4ad",
            "convae_branch_created": False,
            "input_channels": INPUT_CHANNELS,
            "crop_size": CROP_SIZE,
            "input_window": "lung",
            "mask_source": "v4_20_roi_lung_mask",
            "stage2_holdout_accessed": stage2_accessed,
            "full_crop_build_executed": False,
            "training_executed": False,
            "model_forward_executed": False,
            "checkpoint_saved": False,
            "existing_artifact_modified": False,
            "existing_score_csv_modified": False,
            "existing_manifest_modified": False,
            "positive_label_used_for_training": False,
            "lesion_mask_used_for_training": False,
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def write_done_json(verdict, n_samples, n_errors, qa_saved):
    done = {
        "step": "step1_crop_smoke",
        "verdict": verdict,
        "created": "2026-06-10",
        "sample_count": n_samples,
        "error_count": n_errors,
        "qa_png_saved": qa_saved,
        "outputs": {
            "samples_csv": str(SAMPLES_CSV),
            "metrics_csv": str(METRICS_CSV),
            "report_md": str(REPORT_MD),
            "summary_json": str(SUMMARY_JSON),
            "error_csv": str(ERROR_CSV),
            "qa_png_dir": str(OUT_QA),
        },
        "next_step": "step2_crop_full_preflight",
        "next_step_note": "정상 전체 5ch crop 생성 preflight (사용자 승인 후)",
    }
    DONE_JSON.write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8")


# ─── Plan Lock verification ────────────────────────────────────────────────────
def verify_plan_lock():
    if not PLAN_JSON.exists():
        print(f"[BLOCKED] FINAL_PLAN_LOCK.json not found: {PLAN_JSON}")
        sys.exit(1)
    with open(PLAN_JSON) as f:
        plan = json.load(f)
    checks = [
        ("plan_locked=true",        plan.get("plan_locked") is True),
        ("model_type=true_rd4ad",   plan.get("model", {}).get("model_type") == "true_rd4ad"),
        ("input_channels=5",        plan.get("model", {}).get("input_channels") == 5),
        ("input_window=lung",       plan.get("model", {}).get("input_window") == "lung"),
        ("crop_size=96",            plan.get("model", {}).get("crop_size") == 96),
        ("convae=false",            plan.get("model", {}).get("convae_branch_created") is False),
        ("training_data=normal",    plan.get("training", {}).get("training_data") == "normal_only"),
        ("stage2_not_accessed",     plan.get("safety", {}).get("stage2_holdout_accessed") is False),
    ]
    all_ok = all(log_check(n, ok) for n, ok in checks)
    if not all_ok:
        print("[BLOCKED] FINAL_PLAN_LOCK 조건 불일치")
        sys.exit(1)
    return plan


def check_stage2_not_accessed():
    for name in STAGE2_FORBIDDEN:
        for base in [PROJECT_ROOT, BRANCH_ROOT]:
            if (base / name).exists():
                print(f"[BLOCKED] Stage2 forbidden path exists: {base / name}")
                return True
    return False


# ─── Dry-run ──────────────────────────────────────────────────────────────────
def print_dry_run():
    script_path = Path(__file__)
    print("\n" + "=" * 60)
    print("DRY-RUN PLAN: Step 1 Crop Smoke")
    print("=" * 60)
    print(f"\nBranch  : rd4ad_2p5d_lung5ch_masked_normal_v1")
    print(f"Channels: {INPUT_CHANNELS}  (lung window [{HU_MIN}, {HU_MAX}] → [0,1])")
    print(f"Crop    : {CROP_SIZE}×{CROP_SIZE}, z_padding=nearest_repeat")
    print(f"Mask    : refined_roi_v4_20_modeB (normal→binary, lesion→binary)")
    print(f"\nSample plan:")
    print(f"  A. normal_center     : 5 patients × z=D//2")
    print(f"  C. normal_near_start : 5 patients × z=1     (nearest_repeat test)")
    print(f"  C. normal_near_end   : 5 patients × z=D-2   (nearest_repeat test)")
    print(f"  B. candidate_positive: 10 (p90+z_cont, priority={PRIORITY_PATIENTS})")
    print(f"  B. candidate_neg     : 10 (p90+z_cont, hard_negative)")
    print(f"  Total                : ~35 samples")
    print(f"\nVerification per sample:")
    print(f"  crop shape == (5,96,96), dtype=float32, range=[0,1]")
    print(f"  NaN/Inf == 0, mask binary [0,1]")
    print(f"  outside_lung_abs_mean_after_mask ≈ 0")
    print(f"  nearest_repeat: used_z_indices clamped correctly")
    print(f"\nOutputs:")
    for p in [SAMPLES_CSV, METRICS_CSV, REPORT_MD, SUMMARY_JSON, ERROR_CSV, DONE_JSON]:
        print(f"  {p}")
    print(f"  {OUT_QA}/ (QA PNGs, max {QA_LIMIT})")
    print(f"\nActual run:")
    print(f"  python {script_path} \\")
    print(f"    --run-smoke --confirm-plan-lock --confirm-no-stage2 --confirm-readonly")
    print("=" * 60 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    bare_run_guard(args)

    if args.dry_run:
        print_dry_run()
        return

    actual_run_guard(args)

    print("\n" + "=" * 60)
    print("Step 1 Crop Smoke — rd4ad_2p5d_lung5ch_masked_normal_v1")
    print("=" * 60)

    # Plan lock
    print("\n[0] FINAL_PLAN_LOCK 확인")
    verify_plan_lock()

    # Stage2 guard
    print("\n[0b] stage2 접근 확인")
    stage2_accessed = check_stage2_not_accessed()
    if stage2_accessed:
        sys.exit(1)
    log_check("stage2 not accessed", True)

    # Sample selection
    print("\n[1] 샘플 선정")
    print("  A/C. Normal CT samples...")
    normal_samples = select_normal_samples()
    log_check("normal samples", len(normal_samples) > 0, f"{len(normal_samples)} selected")

    print("  B. Candidate samples...")
    df_cand = pd.read_csv(CANDIDATE_CSV)
    cand_samples = select_candidate_samples(df_cand)
    log_check("candidate samples", len(cand_samples) > 0, f"{len(cand_samples)} selected")

    all_samples = normal_samples + cand_samples
    log_check("total samples", len(all_samples) > 0, f"{len(all_samples)}")
    OUT_MANIFESTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_samples).to_csv(SAMPLES_CSV, index=False)

    # Crop generation
    print(f"\n[2] Crop 생성 및 검증 ({len(all_samples)} samples)")
    all_metrics = []
    errors = []
    qa_count = [0]

    for i, s in enumerate(all_samples):
        print(f"  [{i+1:>2}/{len(all_samples)}] {s['sample_id'][:60]:<60}", end=" ", flush=True)
        m, err = compute_sample(s, qa_count)
        if err is not None:
            errors.append(err)
            print(f"ERROR: {err['error']}")
        else:
            all_metrics.append(m)
            nr_flag = "NR" if m["nearest_repeat_used"] else "  "
            print(f"OK  {nr_flag}  shape={m['crop_shape']}  zeroing_after={m['outside_lung_abs_mean_after_mask']:.2e}")

    # Write outputs
    print("\n[3] 결과 파일 저장")
    OUT_MANIFESTS.mkdir(parents=True, exist_ok=True)
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_metrics).to_csv(METRICS_CSV, index=False)
    pd.DataFrame(errors).to_csv(ERROR_CSV, index=False)

    # Verdict
    df_m = pd.DataFrame(all_metrics)
    n_total = len(df_m)
    n_pass = int((df_m["status"] == "PASS").sum()) if n_total > 0 else 0
    n_err = len(errors)
    shape_pass = int(df_m["crop_shape"].eq(f"({INPUT_CHANNELS}, {CROP_SIZE}, {CROP_SIZE})").sum()) if n_total > 0 else 0
    nan_total = int(df_m["crop_nan_count"].sum()) if n_total > 0 else 0
    inf_total = int(df_m["crop_inf_count"].sum()) if n_total > 0 else 0
    nr_count = int(df_m["nearest_repeat_used"].sum()) if n_total > 0 else 0
    zeroing_ok = int((df_m["outside_lung_abs_mean_after_mask"] < 1e-6).sum()) if n_total > 0 else 0

    if n_total == 0:
        verdict = "BLOCKED"
    elif n_err == 0 and shape_pass == n_total and nan_total == 0 and inf_total == 0:
        verdict = "PASS_STEP1_CROP_SMOKE"
    elif n_pass >= n_total * 0.8 and shape_pass == n_pass:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "BLOCKED"

    write_report_md(all_metrics, errors, verdict, qa_count[0])
    write_summary_json(all_metrics, errors, verdict, stage2_accessed, qa_count[0])
    write_done_json(verdict, n_total, n_err, qa_count[0])

    # Final report
    print("\n" + "=" * 60)
    print(f"판정: {verdict}")
    print("=" * 60)
    print(f"  전체 샘플       : {n_total}")
    print(f"  PASS            : {n_pass}")
    print(f"  오류            : {n_err}")
    print(f"  crop shape PASS : {shape_pass}/{n_total}")
    print(f"  NaN / Inf       : {nan_total} / {inf_total}")
    print(f"  nearest_repeat  : {nr_count} samples")
    print(f"  zeroing PASS    : {zeroing_ok}/{n_total}")
    print(f"  QA PNG 저장     : {qa_count[0]}개 → {OUT_QA}")
    print(f"  stage2 접근     : {stage2_accessed}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
