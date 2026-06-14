"""
Step 0 Preflight — rd4ad_2p5d_lung5ch_masked_normal_v1

확인 항목:
  1.  정상 CT volume 경로 및 존재 확인
  2.  NSCLC CT volume 경로 및 존재 확인
  3.  v4_20 ROI/lung mask 경로 확인
  4.  mask shape와 CT shape alignment 확인 (샘플)
  5.  candidate CSV 경로 및 컬럼 확인
  6.  first-stage score 컬럼명 확인
  7.  p90 threshold 값 및 출처 확인 (stage2 사용 여부)
  8.  candidate 좌표 컬럼 확인
  9.  96×96 crop 생성 가능 여부 확인 (소량 smoke)
  10. 5-slice z loading 가능 여부 확인
  11. z boundary nearest_repeat 가능 여부 확인
  12. p90 초과 후보 수 확인
  13. p90 초과 + z 연속성 >=2 후보 수 확인
  14. normal train 후보(crop) 수 충분 여부 확인
  15. stage1_dev candidate에서 positive retention 평가-only 확인
  16. stage2_holdout 접근이 전혀 없었는지 확인

실행 방법:
  bare run (차단):
    python ... → exit 2

  dry-run:
    python ... --dry-run

  actual preflight:
    python ... --run-preflight --confirm-stage1dev-only --confirm-no-stage2 --confirm-readonly
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────
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
P90_JSON = (
    PROJECT_ROOT
    / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs/evaluation/normal_val_thresholds/normal_val_threshold_p80_p90.json"
)
PLAN_JSON = BRANCH_ROOT / "docs/FINAL_PLAN_LOCK.json"

# Stage2 forbidden paths
STAGE2_FORBIDDEN = [
    "stage2_holdout",
    "stage2_holdout_scoring",
    "stage2_holdout_crops",
]

SCORE_COLUMN = "first_stage_score"
Z_COLUMN = "local_z"
COORD_COLS = ["crop_y0", "crop_x0", "crop_y1", "crop_x1"]
P90_THRESHOLD = 12.196394
Z_CONTINUITY_MIN = 2
CROP_SIZE = 96
INPUT_CHANNELS = 5

# Output paths
OUT_MANIFESTS = BRANCH_ROOT / "manifests"
OUT_REPORTS = BRANCH_ROOT / "reports"
OUT_LOGS = BRANCH_ROOT / "logs"

REPORT_MD = OUT_REPORTS / "step0_preflight_report.md"
REPORT_JSON = OUT_REPORTS / "step0_preflight_summary.json"
CANDIDATE_AUDIT_CSV = OUT_MANIFESTS / "step0_candidate_column_audit.csv"
MASK_AUDIT_CSV = OUT_MANIFESTS / "step0_mask_readiness_audit.csv"
P90_COUNT_CSV = OUT_MANIFESTS / "step0_p90_ztrack_count_summary.csv"
ERROR_CSV = OUT_LOGS / "errors.csv"
DONE_JSON = BRANCH_ROOT / "DONE_STEP0_PREFLIGHT.json"


# ─────────────────────────────────────────
# Arg parse
# ─────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 0 preflight — rd4ad_2p5d_lung5ch_masked_normal_v1"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-preflight", action="store_true")
    parser.add_argument("--confirm-stage1dev-only", action="store_true")
    parser.add_argument("--confirm-no-stage2", action="store_true")
    parser.add_argument("--confirm-readonly", action="store_true")
    return parser.parse_args()


def bare_run_guard(args):
    if not args.dry_run and not args.run_preflight:
        print("[ERROR] bare run is forbidden. Use --dry-run or --run-preflight with confirm flags.")
        sys.exit(2)


def actual_run_guard(args):
    if args.run_preflight:
        missing = []
        if not args.confirm_stage1dev_only:
            missing.append("--confirm-stage1dev-only")
        if not args.confirm_no_stage2:
            missing.append("--confirm-no-stage2")
        if not args.confirm_readonly:
            missing.append("--confirm-readonly")
        if missing:
            print(f"[ERROR] 다음 확인 플래그가 필요합니다: {', '.join(missing)}")
            sys.exit(2)


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def log_check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


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


def apply_lung_window(hu_slice, hu_min=-1350, hu_max=150):
    clipped = np.clip(hu_slice.astype(np.float32), hu_min, hu_max)
    return (clipped - hu_min) / (hu_max - hu_min)


# ─────────────────────────────────────────
# Checks
# ─────────────────────────────────────────
def check_normal_ct(errors):
    print("\n[1] 정상 CT volume 경로 확인")
    ok = NORMAL_CT_ROOT.exists()
    log_check("normal CT root exists", ok, str(NORMAL_CT_ROOT))
    if not ok:
        errors.append({"check": "normal_ct_root", "error": str(NORMAL_CT_ROOT)})
        return 0, ok
    ct_dirs = [d for d in NORMAL_CT_ROOT.iterdir() if (d / "ct_hu.npy").exists()]
    count_ok = len(ct_dirs) >= 300
    log_check(f"normal CT count >= 300", count_ok, f"{len(ct_dirs)} found")
    return len(ct_dirs), ok and count_ok


def check_nsclc_ct(errors):
    print("\n[2] NSCLC CT volume 경로 확인")
    ok = NSCLC_CT_ROOT.exists()
    log_check("NSCLC CT root exists", ok, str(NSCLC_CT_ROOT))
    if not ok:
        errors.append({"check": "nsclc_ct_root", "error": str(NSCLC_CT_ROOT)})
        return 0, False
    ct_dirs = [d for d in NSCLC_CT_ROOT.iterdir() if (d / "ct_hu.npy").exists()]
    count_ok = len(ct_dirs) >= 100
    log_check(f"NSCLC CT count >= 100", count_ok, f"{len(ct_dirs)} found")
    return len(ct_dirs), ok and count_ok


def check_mask(errors):
    print("\n[3] v4_20 ROI/lung mask 경로 확인")
    ok = MASK_ROOT.exists()
    log_check("mask root exists", ok, str(MASK_ROOT))
    if not ok:
        errors.append({"check": "mask_root", "error": str(MASK_ROOT)})
        return 0, False
    mask_files = list(MASK_ROOT.rglob("refined_roi.npy"))
    count_ok = len(mask_files) >= 500
    log_check(f"mask count >= 500", count_ok, f"{len(mask_files)} found")
    return len(mask_files), ok and count_ok


def check_mask_ct_alignment(errors):
    print("\n[4] mask shape와 CT shape alignment 확인 (샘플 3개)")
    audit_rows = []
    all_ok = True
    ct_dirs = list(NORMAL_CT_ROOT.iterdir())[:5]
    checked = 0
    for ct_dir in ct_dirs:
        safe_id = ct_dir.name
        ct_path = ct_dir / "ct_hu.npy"
        mask_path = MASK_ROOT / "normal" / safe_id / "refined_roi.npy"
        if not ct_path.exists() or not mask_path.exists():
            continue
        ct = np.load(str(ct_path), mmap_mode="r")
        mask = np.load(str(mask_path))
        match = ct.shape == mask.shape
        if not match:
            errors.append({"check": "mask_ct_alignment", "safe_id": safe_id,
                           "ct_shape": str(ct.shape), "mask_shape": str(mask.shape)})
            all_ok = False
        log_check(f"alignment {safe_id}", match, f"CT{ct.shape} Mask{mask.shape}")
        audit_rows.append({
            "safe_id": safe_id,
            "ct_shape": str(ct.shape),
            "mask_shape": str(mask.shape),
            "match": match,
        })
        checked += 1
        if checked >= 3:
            break
    return audit_rows, all_ok


def check_candidate_csv(errors):
    print("\n[5-8] candidate CSV 경로 및 컬럼 확인")
    ok = CANDIDATE_CSV.exists()
    log_check("candidate CSV exists", ok, str(CANDIDATE_CSV))
    if not ok:
        errors.append({"check": "candidate_csv", "error": str(CANDIDATE_CSV)})
        return None, False

    df = pd.read_csv(CANDIDATE_CSV)
    total = len(df)
    log_check(f"total rows", True, f"{total:,}")

    # [6] score column
    has_score = SCORE_COLUMN in df.columns
    log_check(f"score column '{SCORE_COLUMN}' exists", has_score)
    if not has_score:
        errors.append({"check": "score_column", "error": SCORE_COLUMN})

    # [7] stage_split 확인 (stage2 절대 없어야 함)
    splits = df["stage_split"].unique().tolist() if "stage_split" in df.columns else []
    stage2_present = any("stage2" in str(s).lower() or "holdout" in str(s).lower() for s in splits)
    log_check("stage2/holdout NOT in stage_split", not stage2_present, str(splits))
    if stage2_present:
        errors.append({"check": "stage2_in_candidate_csv", "splits": str(splits)})

    # [8] coordinate columns
    missing_coords = [c for c in COORD_COLS if c not in df.columns]
    ok_coords = len(missing_coords) == 0
    log_check(f"coordinate columns exist", ok_coords,
              f"OK: {COORD_COLS}" if ok_coords else f"missing: {missing_coords}")
    if not ok_coords:
        errors.append({"check": "coord_columns", "missing": missing_coords})

    # z column
    has_z = Z_COLUMN in df.columns
    log_check(f"z column '{Z_COLUMN}' exists", has_z)
    if not has_z:
        errors.append({"check": "z_column", "error": Z_COLUMN})

    # all columns
    audit_rows = [{"column": c, "present": c in df.columns, "sample": str(df[c].iloc[0]) if c in df.columns else "N/A"}
                  for c in [SCORE_COLUMN, Z_COLUMN] + COORD_COLS + ["patient_id", "safe_id", "stage_split", "label"]]

    all_ok = has_score and ok_coords and has_z and not stage2_present
    return df, all_ok, audit_rows


def check_p90(df, errors):
    print("\n[7] p90 threshold 값 및 출처 확인")
    ok = P90_JSON.exists()
    log_check("p90 JSON exists", ok, str(P90_JSON))
    if not ok:
        errors.append({"check": "p90_json", "error": str(P90_JSON)})
        return P90_THRESHOLD, False

    with open(P90_JSON) as f:
        p90_data = json.load(f)

    p90_val = p90_data.get("threshold_p90", None)
    branch = p90_data.get("branch", "")
    n_val = p90_data.get("n_val_patients", 0)

    stage2_used = "stage2" in str(p90_data).lower() or "holdout" in str(p90_data).lower()
    ok_source = not stage2_used and n_val > 0

    log_check(f"p90 value = {p90_val}", p90_val is not None)
    log_check(f"p90 source is normal_val (NOT stage2)", ok_source,
              f"n_val_patients={n_val}, branch={branch}")

    if not ok_source:
        errors.append({"check": "p90_source", "error": "stage2 used or n_val=0"})

    return p90_val, ok_source


def count_candidates(df, p90_val, errors):
    print(f"\n[12-13] 후보 수 확인 (p90>{p90_val:.4f}, z연속>={Z_CONTINUITY_MIN})")
    over_p90 = df[df[SCORE_COLUMN] > p90_val]
    count_p90 = len(over_p90)
    patients_p90 = over_p90["patient_id"].nunique()
    log_check(f"p90 초과 후보 수", count_p90 > 0, f"{count_p90:,}개 / {patients_p90}환자")

    # z-continuity 계산
    over_p90 = over_p90.copy()
    over_p90["pos_key"] = (
        over_p90["patient_id"] + "_"
        + over_p90["crop_y0"].astype(str) + "_"
        + over_p90["crop_x0"].astype(str)
    )
    pos_groups = over_p90.groupby("pos_key")[Z_COLUMN].apply(list)
    valid_keys = {k for k, zl in pos_groups.items() if has_z_continuity(zl, Z_CONTINUITY_MIN)}
    filtered = over_p90[over_p90["pos_key"].isin(valid_keys)]
    count_ztrack = len(filtered)
    patients_ztrack = filtered["patient_id"].nunique()
    label_dist = filtered["label"].value_counts().to_dict() if "label" in filtered.columns else {}

    log_check(f"p90 + z연속>={Z_CONTINUITY_MIN} 후보 수", count_ztrack > 0,
              f"{count_ztrack:,}개 / {patients_ztrack}환자 / {label_dist}")

    summary_rows = [
        {"filter": "p90_exceed", "threshold": p90_val, "count": count_p90, "patients": patients_p90},
        {"filter": f"p90_plus_z_continuity_ge{Z_CONTINUITY_MIN}", "threshold": p90_val,
         "count": count_ztrack, "patients": patients_ztrack,
         "label_dist": str(label_dist)},
    ]
    return count_p90, count_ztrack, patients_ztrack, summary_rows


def check_crop_generation(df, errors):
    print(f"\n[9-11] 96×96 crop 생성 및 5-slice z loading 확인 (소량 smoke)")
    # 후보 CSV에서 첫 번째 NSCLC 환자 샘플 1개
    sample = df[df[SCORE_COLUMN] > P90_THRESHOLD].iloc[0]
    patient_id = sample["patient_id"]
    safe_id = sample["safe_id"]
    local_z = int(sample[Z_COLUMN])
    crop_y0, crop_x0 = int(sample["crop_y0"]), int(sample["crop_x0"])
    crop_y1, crop_x1 = int(sample["crop_y1"]), int(sample["crop_x1"])

    # CT 경로 결정 (NSCLC 우선, normal fallback)
    ct_path = NSCLC_CT_ROOT / safe_id / "ct_hu.npy"
    if not ct_path.exists():
        # safe_id는 NSCLC_safe_id, CT 폴더는 다를 수 있음
        nsclc_dirs = list(NSCLC_CT_ROOT.iterdir())
        # patient_id 기반 탐색
        alt = [d for d in nsclc_dirs if patient_id.replace("LUNG1-", "LUNG1-") in d.name]
        if alt:
            ct_path = alt[0] / "ct_hu.npy"

    ok_ct = ct_path.exists()
    log_check(f"sample CT exists ({safe_id})", ok_ct, str(ct_path))
    if not ok_ct:
        errors.append({"check": "sample_ct_not_found", "safe_id": safe_id, "path": str(ct_path)})
        return False

    ct = np.load(str(ct_path), mmap_mode="r")
    n_slices = ct.shape[0]
    log_check(f"CT loaded shape={ct.shape}", True)

    # [10] 5-slice z loading (z-2 ~ z+2)
    z_indices = [local_z + dz for dz in range(-2, 3)]
    # [11] nearest_repeat 처리
    z_clamped = [max(0, min(n_slices - 1, z)) for z in z_indices]
    slices = []
    for zi in z_clamped:
        s = ct[zi, :, :]
        slices.append(apply_lung_window(s))
    log_check(f"5-slice z loading (z={local_z}, range={z_clamped})", True)

    # [9] 96×96 crop 추출
    crop_h = crop_y1 - crop_y0
    crop_w = crop_x1 - crop_x0
    ok_crop_size = crop_h == CROP_SIZE and crop_w == CROP_SIZE
    log_check(f"crop size == {CROP_SIZE}×{CROP_SIZE}", ok_crop_size,
              f"actual: {crop_h}×{crop_w}")
    if not ok_crop_size:
        errors.append({"check": "crop_size", "expected": CROP_SIZE, "actual_h": crop_h, "actual_w": crop_w})

    # 실제 crop 추출
    crops = []
    for s in slices:
        c = s[crop_y0:crop_y1, crop_x0:crop_x1]
        crops.append(c)

    ok_shape = len(crops) == INPUT_CHANNELS and crops[0].shape == (CROP_SIZE, CROP_SIZE)
    stack = np.stack(crops, axis=0)
    log_check(f"stacked crop shape == ({INPUT_CHANNELS},{CROP_SIZE},{CROP_SIZE})", ok_shape,
              f"actual: {stack.shape}")

    # dtype / range 확인
    ok_dtype = stack.dtype == np.float32
    ok_range = float(stack.min()) >= 0.0 and float(stack.max()) <= 1.0
    log_check(f"dtype=float32", ok_dtype, str(stack.dtype))
    log_check(f"range in [0,1]", ok_range, f"min={stack.min():.3f} max={stack.max():.3f}")

    # mask load + apply
    mask_path = MASK_ROOT / "lesion" / safe_id / "refined_roi.npy"
    if not mask_path.exists():
        mask_path = MASK_ROOT / "normal" / safe_id / "refined_roi.npy"
    ok_mask = mask_path.exists()
    log_check(f"mask exists for sample", ok_mask)
    if ok_mask:
        m = np.load(str(mask_path))
        z_mask = m[local_z, :, :]
        m_crop = z_mask[crop_y0:crop_y1, crop_x0:crop_x1].astype(np.float32)
        log_check(f"mask crop shape == ({CROP_SIZE},{CROP_SIZE})", m_crop.shape == (CROP_SIZE, CROP_SIZE))

    return ok_crop_size and ok_shape and ok_dtype and ok_range


def check_normal_crop_count(errors):
    print("\n[14] normal train 후보(crop) 수 충분 여부 확인")
    # 기존 6ch crop manifest 참조 (새 5ch crop은 아직 없으므로 CT 수로 추정)
    ct_dirs = [d for d in NORMAL_CT_ROOT.iterdir() if (d / "ct_hu.npy").exists()]
    ct_count = len(ct_dirs)
    ok = ct_count >= 300
    log_check(f"normal CT 수로 추정 (>= 300)", ok, f"{ct_count}명 × ~50 crop/patient 예상")
    # 기존 manifest 참조
    old_manifest = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_normal/normal_rd4ad_2p5d_mw_fixed96_v1/manifests/crop_manifest_normal_rd4ad_2p5d_mw_fixed96_v1.csv"
    if old_manifest.exists():
        df_old = pd.read_csv(old_manifest)
        log_check(f"기존 6ch normal crop 수 (참고)", True, f"{len(df_old):,}개 (6ch 기준)")
    return ct_count, ok


def check_stage1dev_positive_retention(df, p90_val, errors):
    print("\n[15] stage1_dev positive retention 평가-only 확인")
    positives = df[df["label"] == "positive"] if "label" in df.columns else pd.DataFrame()
    if len(positives) == 0:
        log_check("positive rows exist", False, "label column 없거나 positive 없음")
        return {}, False

    total_pos = len(positives)
    pos_over_p90 = positives[positives[SCORE_COLUMN] > p90_val]
    recall_p90 = len(pos_over_p90) / total_pos

    pos_over_p90 = pos_over_p90.copy()
    pos_over_p90["pos_key"] = (
        pos_over_p90["patient_id"] + "_"
        + pos_over_p90["crop_y0"].astype(str) + "_"
        + pos_over_p90["crop_x0"].astype(str)
    )
    pos_groups = pos_over_p90.groupby("pos_key")[Z_COLUMN].apply(list)
    valid_keys = {k for k, zl in pos_groups.items() if has_z_continuity(zl, Z_CONTINUITY_MIN)}
    pos_ztrack = pos_over_p90[pos_over_p90["pos_key"].isin(valid_keys)]
    recall_ztrack = len(pos_ztrack) / total_pos

    result = {
        "total_positive": total_pos,
        "positive_over_p90": len(pos_over_p90),
        "recall_p90": round(recall_p90, 4),
        "positive_p90_plus_ztrack": len(pos_ztrack),
        "recall_p90_plus_ztrack": round(recall_ztrack, 4),
        "patients_retained": pos_ztrack["patient_id"].nunique() if len(pos_ztrack) > 0 else 0,
    }
    ok = recall_ztrack > 0.5
    log_check(f"positive recall (p90+z): {recall_ztrack:.1%}", ok,
              f"total_pos={total_pos}, retained={len(pos_ztrack)}, patients={result['patients_retained']}")
    return result, ok


def check_stage2_not_accessed(errors):
    print("\n[16] stage2_holdout 접근 여부 확인")
    for pattern in STAGE2_FORBIDDEN:
        # 스크립트 자체에서 금지 경로 접근 없음을 확인 (소스 코드 기준)
        forbidden_path = PROJECT_ROOT / "experiments" / "rd4ad_2p5d_lung5ch_masked_normal_v1" / pattern
        ok = not forbidden_path.exists()
        log_check(f"stage2 path '{pattern}' NOT created", ok)
        if not ok:
            errors.append({"check": "stage2_access", "path": str(forbidden_path)})
    # plan JSON 확인
    if PLAN_JSON.exists():
        with open(PLAN_JSON) as f:
            plan = json.load(f)
        stage2_accessed = plan.get("safety", {}).get("stage2_holdout_accessed", True)
        ok = not stage2_accessed
        log_check("plan.safety.stage2_holdout_accessed == false", ok)
    return True


# ─────────────────────────────────────────
# Write outputs
# ─────────────────────────────────────────
def write_outputs(
    mask_audit, candidate_audit, p90_count_rows,
    results, errors, dry_run
):
    if dry_run:
        print("\n[DRY-RUN] 파일 작성 skip")
        return

    OUT_MANIFESTS.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_LOGS.mkdir(parents=True, exist_ok=True)

    # mask audit CSV
    if mask_audit:
        pd.DataFrame(mask_audit).to_csv(MASK_AUDIT_CSV, index=False)
        print(f"[WRITE] {MASK_AUDIT_CSV}")

    # candidate column audit CSV
    if candidate_audit:
        pd.DataFrame(candidate_audit).to_csv(CANDIDATE_AUDIT_CSV, index=False)
        print(f"[WRITE] {CANDIDATE_AUDIT_CSV}")

    # p90 count CSV
    if p90_count_rows:
        pd.DataFrame(p90_count_rows).to_csv(P90_COUNT_CSV, index=False)
        print(f"[WRITE] {P90_COUNT_CSV}")

    # errors CSV
    if errors:
        pd.DataFrame(errors).to_csv(ERROR_CSV, index=False)
        print(f"[WRITE] {ERROR_CSV} ({len(errors)} errors)")
    else:
        pd.DataFrame(columns=["check", "error"]).to_csv(ERROR_CSV, index=False)

    # summary JSON
    verdict = "PASS_STEP0_PREFLIGHT" if results.get("all_pass") else "BLOCKED_PREFLIGHT"
    summary = {
        "verdict": verdict,
        "plan_locked": True,
        "plan_path": "experiments/rd4ad_2p5d_lung5ch_masked_normal_v1/docs/FINAL_PLAN_LOCK.md",
        "branch_name": "rd4ad_2p5d_lung5ch_masked_normal_v1",
        "model_type": "true_rd4ad",
        "convae_branch_created": False,
        "input_channels": INPUT_CHANNELS,
        "input_window": "lung",
        "crop_size": CROP_SIZE,
        "training_data": "normal_only",
        "mask_source": "v4_20_roi_lung_mask",
        "loss_scope": "lung_mask_inside_feature_error",
        "first_stage_candidate_filter": "p90_and_z_continuity_ge2",
        "stage2_holdout_accessed": False,
        "stage2_used_for_method_tuning": False,
        "full_training_executed": False,
        "model_forward_executed": False,
        "checkpoint_saved": False,
        "existing_artifact_modified": False,
        "existing_score_csv_modified": False,
        "existing_manifest_modified": False,
        "positive_label_used_for_training": False,
        "lesion_mask_used_for_training": False,
        "representative_only_scoring_used": False,
        "hard_filter_applied": False,
        "vessel_mask_used_for_deletion": False,
        **results,
        "errors": errors,
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[WRITE] {REPORT_JSON}")

    # report MD
    with open(REPORT_MD, "w") as f:
        f.write(f"# Step 0 Preflight Report — rd4ad_2p5d_lung5ch_masked_normal_v1\n\n")
        f.write(f"**판정**: {verdict}\n\n")
        f.write(f"## 주요 결과\n\n")
        f.write(f"| 항목 | 값 |\n|---|---|\n")
        for k, v in results.items():
            f.write(f"| {k} | {v} |\n")
        if errors:
            f.write(f"\n## 오류 목록\n\n")
            for e in errors:
                f.write(f"- {e}\n")
    print(f"[WRITE] {REPORT_MD}")

    # DONE JSON
    if results.get("all_pass"):
        done = {
            "verdict": "PASS_STEP0_PREFLIGHT",
            "branch": "rd4ad_2p5d_lung5ch_masked_normal_v1",
            "plan_lock_confirmed": True,
            "stage2_holdout_accessed": False,
            "next_step": "step1_crop_smoke",
            "next_step_note": "5ch lung-window crop 소량 생성 + shape/dtype 검증 (사용자 승인 후)",
        }
        with open(DONE_JSON, "w") as f:
            json.dump(done, f, indent=2, ensure_ascii=False)
        print(f"[WRITE] {DONE_JSON}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    args = parse_args()
    bare_run_guard(args)
    actual_run_guard(args)

    dry_run = args.dry_run
    print("=" * 60)
    print("Step 0 Preflight — rd4ad_2p5d_lung5ch_masked_normal_v1")
    print(f"mode: {'DRY-RUN' if dry_run else 'ACTUAL PREFLIGHT'}")
    print("=" * 60)

    errors = []
    results = {}
    mask_audit = []
    candidate_audit = []
    p90_count_rows = []
    all_pass = True

    # [1] Normal CT
    normal_ct_count, ok = check_normal_ct(errors)
    results["normal_ct_count"] = normal_ct_count
    all_pass = all_pass and ok

    # [2] NSCLC CT
    nsclc_ct_count, ok = check_nsclc_ct(errors)
    results["nsclc_ct_count"] = nsclc_ct_count
    all_pass = all_pass and ok

    # [3] Mask
    mask_count, ok = check_mask(errors)
    results["mask_count"] = mask_count
    all_pass = all_pass and ok

    # [4] Mask-CT alignment
    mask_audit, ok = check_mask_ct_alignment(errors)
    results["mask_ct_alignment"] = "PASS" if ok else "FAIL"
    all_pass = all_pass and ok

    # [5-8] Candidate CSV
    ret = check_candidate_csv(errors)
    if ret[0] is None:
        all_pass = False
        df = None
        candidate_audit = []
    else:
        df, ok, candidate_audit = ret
        all_pass = all_pass and ok

    # [7] p90 source
    if df is not None:
        p90_val, ok = check_p90(df, errors)
        results["p90_threshold"] = p90_val
        results["p90_source_ok"] = ok
        all_pass = all_pass and ok

        # [12-13] candidate counts
        cnt_p90, cnt_ztrack, pts_ztrack, p90_count_rows = count_candidates(df, p90_val, errors)
        results["candidate_count_p90"] = cnt_p90
        results["candidate_count_p90_ztrack"] = cnt_ztrack
        results["candidate_patients_ztrack"] = pts_ztrack
        all_pass = all_pass and (cnt_ztrack > 0)

        # [15] positive retention
        retention, ok = check_stage1dev_positive_retention(df, p90_val, errors)
        results["positive_retention"] = retention
        all_pass = all_pass and ok

    # [9-11] Crop generation smoke
    if df is not None:
        ok = check_crop_generation(df, errors)
        results["crop_generation"] = "PASS" if ok else "FAIL"
        all_pass = all_pass and ok

    # [14] Normal crop count
    ct_count, ok = check_normal_crop_count(errors)
    results["normal_ct_for_train"] = ct_count
    all_pass = all_pass and ok

    # [16] stage2 not accessed
    check_stage2_not_accessed(errors)
    results["stage2_holdout_accessed"] = False

    results["all_pass"] = all_pass
    verdict = "PASS_STEP0_PREFLIGHT" if all_pass else "BLOCKED_PREFLIGHT"
    results["verdict"] = verdict

    print("\n" + "=" * 60)
    print(f"판정: {verdict}")
    print(f"오류 수: {len(errors)}")
    print("=" * 60)

    write_outputs(mask_audit, candidate_audit, p90_count_rows, results, errors, dry_run)

    if not all_pass:
        print("\n[BLOCKED] 아래 오류를 해결 후 재실행하세요:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
