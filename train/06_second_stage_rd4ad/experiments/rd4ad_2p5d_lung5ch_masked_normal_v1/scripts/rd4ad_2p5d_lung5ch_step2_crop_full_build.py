"""
Step 2 Crop Full Build — rd4ad_2p5d_lung5ch_masked_normal_v1

정상 362명 5ch lung-window crop full build.
저장: float16, 2z 간격 샘플링 (~46k crops, ~4 GB)

  bare run → exit 2
  dry-run  → --dry-run
  actual   → --run-build --confirm-plan-lock --confirm-no-stage2 --confirm-float16-2z
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
STEP2_PREFLIGHT_DONE = BRANCH_ROOT / "DONE_STEP2_FULL_BUILD_PREFLIGHT.json"
Z_AUDIT_CSV = BRANCH_ROOT / "manifests/step2_normal_lung_z_range_audit.csv"

STAGE2_FORBIDDEN = ["stage2_holdout", "stage2_holdout_scoring", "stage2_holdout_crops"]

# ─── Constants ────────────────────────────────────────────────────────────────
CROP_SIZE = 96
INPUT_CHANNELS = 5
HU_MIN, HU_MAX = -1350, 150
Z_STEP = 2         # every 2nd z-slice
DTYPE_SAVE = np.float16

# ─── Output paths ─────────────────────────────────────────────────────────────
CROPS_DIR = BRANCH_ROOT / "crops/normal_5ch_lung_w96_v1"
OUT_MANIFESTS = BRANCH_ROOT / "manifests"
OUT_REPORTS = BRANCH_ROOT / "reports"
OUT_LOGS = BRANCH_ROOT / "logs"

MANIFEST_CSV = OUT_MANIFESTS / "step2_crop_build_manifest.csv"
SUMMARY_JSON = OUT_REPORTS / "step2_crop_build_summary.json"
ERROR_CSV = OUT_LOGS / "step2_crop_build_errors.csv"
DONE_JSON = BRANCH_ROOT / "DONE_STEP2_CROP_FULL_BUILD.json"


# ─── Arg parse ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Step 2 Crop Full Build")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-build", action="store_true")
    p.add_argument("--confirm-plan-lock", action="store_true")
    p.add_argument("--confirm-no-stage2", action="store_true")
    p.add_argument("--confirm-float16-2z", action="store_true")
    return p.parse_args()


def bare_run_guard(args):
    if not args.dry_run and not args.run_build:
        print("[ERROR] bare run forbidden. Use --dry-run or --run-build.")
        sys.exit(2)


def actual_run_guard(args):
    if args.run_build:
        missing = [f for f, ok in [
            ("--confirm-plan-lock",   args.confirm_plan_lock),
            ("--confirm-no-stage2",   args.confirm_no_stage2),
            ("--confirm-float16-2z",  args.confirm_float16_2z),
        ] if not ok]
        if missing:
            print(f"[ERROR] Missing flags: {', '.join(missing)}")
            sys.exit(2)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def log(msg):
    print(msg, flush=True)


def apply_lung_window(hu_arr):
    clipped = np.clip(hu_arr.astype(np.float32), HU_MIN, HU_MAX)
    return (clipped - HU_MIN) / (HU_MAX - HU_MIN)


def get_z_indices(z_center, D):
    return [max(0, min(D - 1, z_center + off)) for off in (-2, -1, 0, 1, 2)]


def extract_5ch_crop_f16(ct_vol, mask_vol, z_center, y0, x0, y1, x1):
    """Returns (5, 96, 96) float16, lung-window normalized, lung-masked."""
    D = ct_vol.shape[0]
    z_idx = get_z_indices(z_center, D)
    chans = []
    m_chans = []
    for z in z_idx:
        crop = np.array(ct_vol[z, y0:y1, x0:x1])
        mcrop = mask_vol[z, y0:y1, x0:x1]
        if crop.shape != (CROP_SIZE, CROP_SIZE):
            pad = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
            mpad = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
            h = min(crop.shape[0], CROP_SIZE)
            w = min(crop.shape[1], CROP_SIZE)
            pad[:h, :w] = crop[:h, :w]
            mpad[:h, :w] = mcrop[:h, :w]
            crop, mcrop = pad, mpad
        chans.append(apply_lung_window(crop))
        m_chans.append(mcrop.astype(np.float32))
    crop_5ch = np.stack(chans, axis=0)       # (5, 96, 96) float32
    mask_5ch = np.stack(m_chans, axis=0)
    masked = (crop_5ch * mask_5ch).astype(DTYPE_SAVE)  # → float16
    nr = any(z != z_center + off for z, off in zip(z_idx, (-2, -1, 0, 1, 2)))
    return masked, z_idx, nr


# ─── Guards ───────────────────────────────────────────────────────────────────
def verify_plan_lock():
    if not PLAN_JSON.exists():
        print(f"[BLOCKED] PLAN_JSON missing")
        sys.exit(1)
    with open(PLAN_JSON) as f:
        p = json.load(f)
    ok = (p.get("plan_locked") is True
          and p.get("model", {}).get("model_type") == "true_rd4ad"
          and p.get("model", {}).get("input_channels") == 5
          and p.get("training", {}).get("training_data") == "normal_only"
          and p.get("safety", {}).get("stage2_holdout_accessed") is False)
    if not ok:
        print("[BLOCKED] FINAL_PLAN_LOCK 조건 불일치")
        sys.exit(1)
    print("  [PASS] plan_lock verified")


def check_step2_preflight_pass():
    if not STEP2_PREFLIGHT_DONE.exists():
        print("[BLOCKED] Step 2 preflight DONE file missing")
        sys.exit(1)
    with open(STEP2_PREFLIGHT_DONE) as f:
        d = json.load(f)
    if d.get("verdict") != "PASS_STEP2_FULL_BUILD_PREFLIGHT":
        print(f"[BLOCKED] Step 2 preflight verdict = {d.get('verdict')}")
        sys.exit(1)
    print(f"  [PASS] step2 preflight = {d['verdict']}")


def check_stage2():
    for name in STAGE2_FORBIDDEN:
        for base in [PROJECT_ROOT, BRANCH_ROOT]:
            if (base / name).exists():
                print(f"[BLOCKED] Stage2 path exists: {base / name}")
                sys.exit(1)
    print("  [PASS] stage2 not accessed")


# ─── Dry-run ──────────────────────────────────────────────────────────────────
def print_dry_run(audit_df):
    n_pass = len(audit_df[audit_df["status"] == "PASS"]) if len(audit_df) > 0 else 0
    total_crops = 0
    if n_pass > 0:
        df_p = audit_df[audit_df["status"] == "PASS"]
        for _, r in df_p.iterrows():
            z_start = int(r["lung_z_min_any"])
            z_end = int(r["lung_z_max_any"])
            total_crops += len(range(z_start, z_end + 1, Z_STEP))
    size_gb = total_crops * INPUT_CHANNELS * CROP_SIZE * CROP_SIZE * 2 / (1024 ** 3)
    print("\n" + "=" * 64)
    print("DRY-RUN PLAN: Step 2 Crop Full Build")
    print("=" * 64)
    print(f"  환자 수          : {n_pass}")
    print(f"  z 샘플링         : 매 {Z_STEP}번째 슬라이스 (lung_z_min → lung_z_max)")
    print(f"  예상 crop 수     : {total_crops:,}")
    print(f"  저장 dtype       : float16")
    print(f"  예상 저장 용량   : {size_gb:.2f} GB")
    print(f"  폐 외부 zeroing  : mask 적용 (v4_20)")
    print(f"  resume           : 기존 .npy 파일 있으면 skip")
    print(f"  출력 디렉토리    : {CROPS_DIR}")
    print(f"  manifest         : {MANIFEST_CSV}")
    print(f"  done marker      : {DONE_JSON}")
    print(f"\n금지:")
    print(f"  training / model forward / checkpoint")
    print(f"  stage2_holdout 접근")
    print(f"  기존 artifact 수정")
    print(f"\n실행 명령:")
    script = Path(__file__)
    print(f"  python {script} \\")
    print(f"    --run-build --confirm-plan-lock --confirm-no-stage2 --confirm-float16-2z")
    print("=" * 64 + "\n")


# ─── Per-patient crop build ───────────────────────────────────────────────────
def build_patient_crops(safe_id, lung_z_min, lung_z_max, ct_shape):
    D, H, W = ct_shape
    y0 = H // 2 - CROP_SIZE // 2
    x0 = W // 2 - CROP_SIZE // 2
    y1 = y0 + CROP_SIZE
    x1 = x0 + CROP_SIZE

    out_path = CROPS_DIR / f"{safe_id}_crops_f16.npy"

    # Resume: skip if already done
    if out_path.exists():
        try:
            existing = np.load(str(out_path), mmap_mode="r")
            expected = len(range(lung_z_min, lung_z_max + 1, Z_STEP))
            if existing.shape[0] == expected:
                return None, out_path, []  # None = skipped (resume)
        except Exception:
            pass  # file corrupt → rebuild

    ct_path = NORMAL_CT_ROOT / safe_id / "ct_hu.npy"
    mask_path = MASK_ROOT / "normal" / safe_id / "refined_roi.npy"
    if not ct_path.exists():
        return "ERROR", None, [f"{safe_id}: CT missing"]
    if not mask_path.exists():
        return "ERROR", None, [f"{safe_id}: Mask missing"]

    ct = np.load(str(ct_path), mmap_mode="r")
    mask = np.load(str(mask_path))

    z_range = range(lung_z_min, lung_z_max + 1, Z_STEP)
    crops = []
    manifest_rows = []

    for z in z_range:
        try:
            crop_f16, used_z, nr = extract_5ch_crop_f16(ct, mask, z, y0, x0, y1, x1)
        except Exception as e:
            return "ERROR", None, [f"{safe_id} z={z}: {e}"]
        crops.append(crop_f16)
        z_denom = max(1, lung_z_max - lung_z_min)
        manifest_rows.append({
            "safe_id": safe_id,
            "local_z": z,
            "crop_y0": y0, "crop_x0": x0, "crop_y1": y1, "crop_x1": x1,
            "lung_z_min": lung_z_min, "lung_z_max": lung_z_max,
            "lung_z_percentile": round((z - lung_z_min) / z_denom, 4),
            "is_near_lung_apex": z <= lung_z_min + max(1, int((lung_z_max - lung_z_min) * 0.1)),
            "is_near_lung_base": z >= lung_z_max - max(1, int((lung_z_max - lung_z_min) * 0.1)),
            "nearest_repeat_used": nr,
            "file_path": str(out_path.relative_to(BRANCH_ROOT)),
            "crop_index_in_file": len(crops) - 1,
        })

    arr = np.stack(crops, axis=0)  # (N, 5, 96, 96) float16
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), arr)

    return "DONE", out_path, manifest_rows


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    bare_run_guard(args)

    # Load audit CSV (needed for dry-run too)
    if not Z_AUDIT_CSV.exists():
        print(f"[BLOCKED] Z audit CSV missing: {Z_AUDIT_CSV}")
        sys.exit(1)
    audit_df = pd.read_csv(Z_AUDIT_CSV)

    if args.dry_run:
        print_dry_run(audit_df)
        return

    actual_run_guard(args)

    log("\n" + "=" * 64)
    log("Step 2 Crop Full Build — rd4ad_2p5d_lung5ch_masked_normal_v1")
    log(f"  dtype=float16  z_step={Z_STEP}  crop_size={CROP_SIZE}")
    log("=" * 64)

    log("\n[0] Guards")
    verify_plan_lock()
    check_stage2()
    check_step2_preflight_pass()

    # Patient list
    df_pass = audit_df[audit_df["status"] == "PASS"].copy()
    n_patients = len(df_pass)
    log(f"\n[1] 환자 수: {n_patients}")

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MANIFESTS.mkdir(parents=True, exist_ok=True)
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)

    log(f"\n[2] Crop 생성 시작")
    all_manifest_rows = []
    errors = []
    n_done = 0
    n_skipped = 0
    n_error = 0
    total_crops = 0

    for i, (_, row) in enumerate(df_pass.iterrows()):
        safe_id = row["safe_id"]
        lung_z_min = int(row["lung_z_min_any"])
        lung_z_max = int(row["lung_z_max_any"])
        ct_shape_str = row["ct_shape"]  # "(D, H, W)"
        ct_shape = tuple(int(x) for x in ct_shape_str.strip("()").split(","))

        if (i + 1) % 20 == 0 or i == 0 or i == n_patients - 1:
            log(f"  [{i+1:>3}/{n_patients}] {safe_id} (z: {lung_z_min}~{lung_z_max}, step={Z_STEP})")

        result, out_path, rows_or_errors = build_patient_crops(
            safe_id, lung_z_min, lung_z_max, ct_shape
        )

        if result is None:  # skipped (resume)
            n_skipped += 1
            existing = np.load(str(out_path), mmap_mode="r")
            total_crops += existing.shape[0]
        elif result == "DONE":
            n_done += 1
            all_manifest_rows.extend(rows_or_errors)
            total_crops += len(rows_or_errors)
        else:  # ERROR
            n_error += 1
            errors.extend(rows_or_errors)
            log(f"    ERROR: {rows_or_errors}")

    log(f"\n  완료: done={n_done}, skipped={n_skipped}, error={n_error}")
    log(f"  총 crop 수: {total_crops:,}")

    # Write manifest (new crops only; skip가 있으면 기존 파일에서 재구성 필요 없음)
    if all_manifest_rows:
        df_manifest = pd.DataFrame(all_manifest_rows)
        if MANIFEST_CSV.exists():
            df_old = pd.read_csv(MANIFEST_CSV)
            df_manifest = pd.concat([df_old, df_manifest], ignore_index=True)
        df_manifest.to_csv(MANIFEST_CSV, index=False)
        log(f"  manifest: {len(df_manifest)} rows → {MANIFEST_CSV.name}")
    elif not MANIFEST_CSV.exists():
        pd.DataFrame().to_csv(MANIFEST_CSV, index=False)

    # Error CSV
    pd.DataFrame([{"error": e} for e in errors]).to_csv(ERROR_CSV, index=False)

    # Storage tally
    npy_files = list(CROPS_DIR.glob("*_crops_f16.npy"))
    actual_size_bytes = sum(f.stat().st_size for f in npy_files)
    actual_size_gb = actual_size_bytes / (1024 ** 3)

    # Summary JSON
    summary = {
        "verdict": "PASS_STEP2_CROP_FULL_BUILD" if n_error == 0 else "PARTIAL_PASS",
        "created": "2026-06-10",
        "branch_name": "rd4ad_2p5d_lung5ch_masked_normal_v1",
        "step": "step2_crop_full_build",
        "n_patients_processed": n_done,
        "n_patients_skipped_resume": n_skipped,
        "n_patients_error": n_error,
        "total_crops": total_crops,
        "npy_files": len(npy_files),
        "actual_size_gb": round(actual_size_gb, 3),
        "dtype": "float16",
        "z_step": Z_STEP,
        "crop_size": CROP_SIZE,
        "guardrail": {
            "stage2_holdout_accessed": False,
            "full_training_executed": False,
            "model_forward_executed": False,
            "checkpoint_saved": False,
            "existing_artifact_modified": False,
            "lung_z_range_used_for_hard_filter": False,
        },
    }
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    verdict = summary["verdict"]
    DONE_JSON.write_text(json.dumps({
        "step": "step2_crop_full_build",
        "verdict": verdict,
        "created": "2026-06-10",
        "total_crops": total_crops,
        "actual_size_gb": round(actual_size_gb, 3),
        "npy_files": len(npy_files),
        "crops_dir": str(CROPS_DIR),
        "manifest_csv": str(MANIFEST_CSV),
        "next_step": "step3_teacher_test",
        "next_step_note": "ResNet18 5ch forward 단일 배치 확인 (사용자 승인 후)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    log("\n" + "=" * 64)
    log(f"판정: {verdict}")
    log("=" * 64)
    log(f"  처리 완료  : {n_done}")
    log(f"  resume skip: {n_skipped}")
    log(f"  오류       : {n_error}")
    log(f"  총 crop    : {total_crops:,}")
    log(f"  실제 용량  : {actual_size_gb:.3f} GB")
    log(f"  파일 수    : {len(npy_files)}")
    log("=" * 64 + "\n")


if __name__ == "__main__":
    main()
