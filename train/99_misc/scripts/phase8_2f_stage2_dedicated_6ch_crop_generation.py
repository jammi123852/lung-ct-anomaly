"""
Phase 8.2F Stage2 Dedicated 6ch Crop Generation

목적: Phase 8.2E에서 생성·검증된 stage2_holdout candidate coordinate manifest를
      입력으로 사용해 dedicated 6ch crop (npz)을 생성한다.

실행 방식:
  --run + --confirm-run 둘 다 있어야 실제 crop 생성
  둘 중 하나라도 없으면 dry-run 보고 후 종료

smoke 옵션:
  --max-patients N  : 최대 N명만 처리 (smoke 전용 경로 사용, full run 경로 미오염)
  --max-rows N      : 최대 N행만 처리 (smoke 전용 경로 사용, full run 경로 미오염)

금지: model forward, scoring, metric 계산, threshold, training 금지
      기존 mixed crop root 수정/삭제/이동 금지
"""

import sys
import os
import gc
import json
import pathlib
import datetime
import argparse

import numpy as np
import pandas as pd

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

SOURCE_VOL_ROOT = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
) / "volumes_npy"

MANIFEST_PATH = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/datasets"
    / "s6a_stage2_holdout_candidate_coordinate_manifest_v1.csv"
)

# ── full run 경로 ──────────────────────────────────────────────────────────────
CROP_TMP_ROOT   = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_stage2_holdout_6ch_dedicated_v1.tmp"
CROP_FINAL_ROOT = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_stage2_holdout_6ch_dedicated_v1"
MANIFEST_TMP    = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_filtered_manifest_v1.csv.tmp"
MANIFEST_FINAL  = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_filtered_manifest_v1.csv"

RUN_OUT_ROOT = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_2f_stage2_dedicated_6ch_crop_generation_v1"
)
OUT_REPORT_MD = RUN_OUT_ROOT / "phase8_2f_stage2_dedicated_6ch_crop_generation_report_v1.md"
OUT_SUMMARY   = RUN_OUT_ROOT / "phase8_2f_stage2_dedicated_6ch_crop_generation_summary_v1.json"
OUT_ERRORS    = RUN_OUT_ROOT / "phase8_2f_stage2_dedicated_6ch_crop_generation_errors_v1.csv"
OUT_RUNTIME   = RUN_OUT_ROOT / "phase8_2f_stage2_dedicated_6ch_crop_generation_runtime_summary_v1.csv"
OUT_DONE      = RUN_OUT_ROOT / "phase8_2f_stage2_dedicated_6ch_crop_generation_DONE.json"

# ── smoke 전용 경로 ────────────────────────────────────────────────────────────
_SMOKE_TAG = "phase8_2f_stage2_dedicated_6ch_crop_generation_smoke_v2"
SMOKE_BASE        = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/smoke" / _SMOKE_TAG
SMOKE_CROP_TMP_ROOT   = SMOKE_BASE / "crops_tmp"
SMOKE_CROP_FINAL_ROOT = SMOKE_BASE / "crops"
SMOKE_MANIFEST_TMP    = SMOKE_BASE / "s6a_stage2_holdout_filtered_manifest_smoke_v1.csv.tmp"
SMOKE_MANIFEST_FINAL  = SMOKE_BASE / "s6a_stage2_holdout_filtered_manifest_smoke_v1.csv"

SMOKE_RUN_OUT_ROOT = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / _SMOKE_TAG
)
SMOKE_OUT_REPORT_MD = SMOKE_RUN_OUT_ROOT / f"{_SMOKE_TAG}_report.md"
SMOKE_OUT_SUMMARY   = SMOKE_RUN_OUT_ROOT / f"{_SMOKE_TAG}_summary.json"
SMOKE_OUT_ERRORS    = SMOKE_RUN_OUT_ROOT / f"{_SMOKE_TAG}_errors.csv"
SMOKE_OUT_RUNTIME   = SMOKE_RUN_OUT_ROOT / f"{_SMOKE_TAG}_runtime_summary.csv"
SMOKE_OUT_DONE      = SMOKE_RUN_OUT_ROOT / f"{_SMOKE_TAG}_DONE.json"

# ── 수치 상수 (generate_s6a_crop_full_6ch.py와 동일) ───────────────────────────
LUNG_WIN_MIN = -1350.0
LUNG_WIN_MAX =   150.0
MEDI_WIN_MIN =  -160.0
MEDI_WIN_MAX =   240.0
CROP_SIZE = 96

EXPECTED_TOTAL    = 143735
EXPECTED_PATIENTS = 154
EXPECTED_POSITIVE = 51335
EXPECTED_HN       = 92400

CONTAMINATED_PATIENTS = {"LUNG1-295", "LUNG1-415"}

SOURCE_COORDINATE_MANIFEST_STR = str(MANIFEST_PATH)

# source 필수 파일 목록 (ct_hu.npy는 로드 가능 확인, 나머지 3개는 existence/size만)
REQUIRED_SOURCE_FILES = ["ct_hu.npy", "roi_0_0.npy", "lesion_mask_roi_0_0.npy", "meta.json"]


# ── windowing (reference: generate_s6a_crop_full_6ch.py와 동일) ───────────────
def apply_lung_window(arr: np.ndarray) -> np.ndarray:
    clipped = np.clip(arr, LUNG_WIN_MIN, LUNG_WIN_MAX)
    return ((clipped - LUNG_WIN_MIN) / (LUNG_WIN_MAX - LUNG_WIN_MIN)).astype(np.float32)


def apply_medi_window(arr: np.ndarray) -> np.ndarray:
    clipped = np.clip(arr, MEDI_WIN_MIN, MEDI_WIN_MAX)
    return ((clipped - MEDI_WIN_MIN) / (MEDI_WIN_MAX - MEDI_WIN_MIN)).astype(np.float32)


# ── crop 좌표 계산 (reference: generate_s6a_crop_full_6ch.py와 동일) ──────────
def compute_fixed_crop_coords(
    cy: float, cx: float, crop_size: int, img_h: int, img_w: int
) -> tuple:
    half = crop_size // 2
    y0 = max(0, int(cy) - half)
    x0 = max(0, int(cx) - half)
    y1 = min(img_h, y0 + crop_size)
    x1 = min(img_w, x0 + crop_size)
    if y1 - y0 < crop_size:
        y0 = max(0, y1 - crop_size)
    if x1 - x0 < crop_size:
        x0 = max(0, x1 - crop_size)
    return y0, x0, y1, x1


# ── 6ch crop 추출 (reference: generate_s6a_crop_full_6ch.py와 동일) ────────────
def extract_6ch_crop(
    ct: np.ndarray, z: int, y0: int, x0: int, y1: int, x1: int
) -> np.ndarray:
    """
    output shape: (6, 96, 96), dtype: float32
    ch0~2: lung window (z-1/z/z+1)
    ch3~5: mediastinal window (z-1/z/z+1)
    z boundary: edge-repeat
    """
    Z = ct.shape[0]
    z_prev = max(0, z - 1)
    z_next = min(Z - 1, z + 1)

    raw_prev = ct[z_prev, y0:y1, x0:x1].astype(np.float32)
    raw_curr = ct[z,      y0:y1, x0:x1].astype(np.float32)
    raw_next = ct[z_next, y0:y1, x0:x1].astype(np.float32)

    lung_prev = apply_lung_window(raw_prev)
    lung_curr = apply_lung_window(raw_curr)
    lung_next = apply_lung_window(raw_next)

    medi_prev = apply_medi_window(raw_prev)
    medi_curr = apply_medi_window(raw_curr)
    medi_next = apply_medi_window(raw_next)

    image = np.stack(
        [lung_prev, lung_curr, lung_next, medi_prev, medi_curr, medi_next],
        axis=0,
    )
    return image


# ── output guard ──────────────────────────────────────────────────────────────
def output_guard(paths: list) -> None:
    abort_msgs = []
    for p, label in paths:
        if pathlib.Path(p).exists():
            abort_msgs.append(f"  [ABORT] {label} 이미 존재: {p}")
    if abort_msgs:
        for m in abort_msgs:
            print(m, file=sys.stderr)
        sys.exit(1)


# ── dry-run 보고 ───────────────────────────────────────────────────────────────
def dry_run_report(df: pd.DataFrame, max_patients: int, max_rows: int) -> None:
    patients = sorted(df["patient_id"].unique().tolist())
    n_pos = int((df["label"] == 1).sum())
    n_hn  = int((df["label"] == 0).sum())
    per_patient = df.groupby("patient_id").size()

    # 디스크 추정: 143,735 * (6*96*96*4 bytes), npz압축 ~1/3 가정
    raw_bytes = EXPECTED_TOTAL * 6 * CROP_SIZE * CROP_SIZE * 4
    est_gb = raw_bytes / 3 / (1024 ** 3)

    is_smoke = max_patients > 0 or max_rows > 0

    print()
    print("=" * 70)
    print("  [DRY-RUN] Phase 8.2F Dedicated 6ch Crop Generation")
    print("=" * 70)
    print(f"  manifest rows   : {len(df):,}")
    print(f"  patient count   : {df['patient_id'].nunique()}")
    print(f"  positive        : {n_pos:,}")
    print(f"  hard_negative   : {n_hn:,}")
    print(f"  crop shape      : (6, {CROP_SIZE}, {CROP_SIZE}) float32")
    print(f"  npz key         : image")
    print(f"  추정 디스크 사용량: ~{est_gb:.1f} GB (압축 후, 143,735 crops 기준)")
    print()
    print(f"  환자별 row 수")
    print(f"    min    : {int(per_patient.min())}")
    print(f"    median : {int(per_patient.median())}")
    print(f"    max    : {int(per_patient.max())}")
    print()
    print(f"  --max-patients 제한 : {max_patients if max_patients else '없음 (전체 154명)'}")
    print(f"  --max-rows 제한     : {max_rows if max_rows else '없음 (전체 143,735행)'}")
    print()

    contaminated = [p for p in patients if p in CONTAMINATED_PATIENTS]
    print(f"  contamination 재생성 대상 : {contaminated}")
    print()
    print("  source root 존재 여부")
    for p in patients[:3]:
        rows = df[df["patient_id"] == p]
        sid = rows["safe_id"].iloc[0] if not rows.empty else "?"
        ct_p = SOURCE_VOL_ROOT / str(sid) / "ct_hu.npy"
        print(f"    {p} ({sid}): {'OK' if ct_p.exists() else 'NOT FOUND'}")
    print(f"    ... ({len(patients)-3}명 더)")
    print()

    if is_smoke:
        print("  [SMOKE] 출력 경로 (smoke 전용 — full run 경로 미오염)")
        for fp, label in [
            (SMOKE_CROP_FINAL_ROOT, "smoke crop root"),
            (SMOKE_CROP_TMP_ROOT,   "smoke crop tmp"),
            (SMOKE_MANIFEST_FINAL,  "smoke manifest"),
            (SMOKE_RUN_OUT_ROOT,    "smoke run output root"),
        ]:
            print(f"    {label}: {'충돌' if fp.exists() else '안전'}")
    else:
        print("  출력 경로 충돌 여부")
        for fp, label in [
            (CROP_FINAL_ROOT, "final crop root"),
            (CROP_TMP_ROOT,   "tmp crop root"),
            (MANIFEST_FINAL,  "final manifest"),
            (MANIFEST_TMP,    "tmp manifest"),
            (RUN_OUT_ROOT,    "run output root"),
        ]:
            print(f"    {label}: {'충돌' if fp.exists() else '안전'}")
    print()
    print("  실행 명령:")
    print("    source ~/ai_env/bin/activate && \\")
    print("    python scripts/phase8_2f_stage2_dedicated_6ch_crop_generation.py \\")
    print("      --run --confirm-run")
    print("=" * 70)
    print()


# ── 환자 처리 ─────────────────────────────────────────────────────────────────
def process_patient(
    patient_id: str,
    rows: pd.DataFrame,
    vol_dir: pathlib.Path,
    out_patient_dir_tmp: pathlib.Path,
    out_patient_dir_final: pathlib.Path,
    error_rows: list,
    manifest_rows: list,
    seen_npz_paths: set,
) -> dict:
    # ── source 4파일 stat 검증 (ct_hu.npy 포함) ──────────────────────────────
    for src_fname in REQUIRED_SOURCE_FILES:
        src_fp = vol_dir / src_fname
        if not src_fp.exists():
            msg = f"source file 없음: {src_fp}"
            print(f"  [ERROR] {msg}", file=sys.stderr)
            for _, row in rows.iterrows():
                error_rows.append({
                    "patient_id": patient_id,
                    "row_id": row.get("row_id", ""),
                    "local_z": row.get("local_z", ""),
                    "error_type": "SOURCE_FILE_MISSING",
                    "error_msg": msg,
                })
            return {"success": 0, "error": len(rows)}
        if src_fp.stat().st_size == 0:
            msg = f"source file 0-byte: {src_fp}"
            print(f"  [ERROR] {msg}", file=sys.stderr)
            for _, row in rows.iterrows():
                error_rows.append({
                    "patient_id": patient_id,
                    "row_id": row.get("row_id", ""),
                    "local_z": row.get("local_z", ""),
                    "error_type": "SOURCE_FILE_ZERO_BYTE",
                    "error_msg": msg,
                })
            return {"success": 0, "error": len(rows)}

    # ct_hu.npy 로드 가능 확인
    ct_path = vol_dir / "ct_hu.npy"
    ct = np.load(str(ct_path), mmap_mode="r")
    if ct.ndim != 3:
        msg = f"ct_hu.npy shape 오류: {ct.shape}"
        print(f"  [ERROR] {msg}", file=sys.stderr)
        del ct
        gc.collect()
        for _, row in rows.iterrows():
            error_rows.append({
                "patient_id": patient_id,
                "row_id": row.get("row_id", ""),
                "local_z": row.get("local_z", ""),
                "error_type": "CT_SHAPE_INVALID",
                "error_msg": msg,
            })
        return {"success": 0, "error": len(rows)}

    _, img_h, img_w = ct.shape
    n_success = 0
    n_error   = 0

    out_patient_dir_tmp.mkdir(parents=True, exist_ok=True)

    contamination_status = (
        "regenerated_crop_from_source_after_prior_contamination"
        if patient_id in CONTAMINATED_PATIENTS
        else "clean_dedicated_stage2_crop_from_source"
    )

    source_ct_path     = str(vol_dir / "ct_hu.npy")
    source_roi_path    = str(vol_dir / "roi_0_0.npy")
    source_lesion_path = str(vol_dir / "lesion_mask_roi_0_0.npy")

    for _, row in rows.iterrows():
        row_id   = str(row.get("row_id", ""))
        local_z  = row.get("local_z", float("nan"))
        safe_id  = str(row.get("safe_id", ""))
        label    = int(row["label"])
        sampling_label = str(row["sampling_label"])

        # fatal guard: stage_split
        if str(row.get("stage_split", "")) != "stage2_holdout":
            print(
                f"[ABORT] stage_split!=stage2_holdout 감지: "
                f"patient={patient_id} row_id={row_id}",
                file=sys.stderr,
            )
            del ct
            gc.collect()
            sys.exit(1)

        # fatal guard: model_type
        if str(row.get("model_type", "")) != "v2v2":
            print(
                f"[ABORT] model_type!=v2v2 감지: patient={patient_id} row_id={row_id}",
                file=sys.stderr,
            )
            del ct
            gc.collect()
            sys.exit(1)

        # local_z 유효성
        if pd.isna(local_z):
            error_rows.append({
                "patient_id": patient_id,
                "row_id": row_id,
                "local_z": local_z,
                "error_type": "LOCAL_Z_NAN",
                "error_msg": "local_z is NaN",
            })
            n_error += 1
            continue

        z = int(local_z)
        if z < 0 or z >= ct.shape[0]:
            error_rows.append({
                "patient_id": patient_id,
                "row_id": row_id,
                "local_z": z,
                "error_type": "LOCAL_Z_OUT_OF_RANGE",
                "error_msg": f"z={z} out of range (Z={ct.shape[0]})",
            })
            n_error += 1
            continue

        cy = (row["y0"] + row["y1"]) / 2.0
        cx = (row["x0"] + row["x1"]) / 2.0
        cy0, cx0, cy1, cx1 = compute_fixed_crop_coords(cy, cx, CROP_SIZE, img_h, img_w)

        image = extract_6ch_crop(ct, z, cy0, cx0, cy1, cx1)

        # shape 검증
        if image.shape != (6, CROP_SIZE, CROP_SIZE):
            error_rows.append({
                "patient_id": patient_id,
                "row_id": row_id,
                "local_z": z,
                "error_type": "CROP_SHAPE_INVALID",
                "error_msg": f"shape={image.shape}",
            })
            n_error += 1
            continue

        # NaN/Inf 검증
        if not np.isfinite(image).all():
            error_rows.append({
                "patient_id": patient_id,
                "row_id": row_id,
                "local_z": z,
                "error_type": "CROP_NAN_INF",
                "error_msg": "NaN or Inf in crop",
            })
            n_error += 1
            continue

        # value range 검증
        img_min = float(image.min())
        img_max = float(image.max())
        if img_min < -1e-6 or img_max > 1.0 + 1e-6:
            error_rows.append({
                "patient_id": patient_id,
                "row_id": row_id,
                "local_z": z,
                "error_type": "CROP_RANGE_INVALID",
                "error_msg": f"min={img_min:.6f}, max={img_max:.6f}",
            })
            n_error += 1
            continue

        # npz filename (unique: row_id + z + y0 + x0)
        fname = f"{row_id}_z{z}_y{int(row['y0'])}_x{int(row['x0'])}.npz"

        # final 경로 기준으로 manifest 기록 (rename 후 실제 위치)
        final_npz = out_patient_dir_final / fname
        final_npz_str = str(final_npz)

        # duplicate 검사 (fatal)
        if final_npz_str in seen_npz_paths:
            print(
                f"[ABORT] duplicate npz_path 감지: {final_npz_str}",
                file=sys.stderr,
            )
            del ct
            gc.collect()
            sys.exit(1)
        seen_npz_paths.add(final_npz_str)

        # tmp에 저장
        tmp_npz = out_patient_dir_tmp / fname
        np.savez_compressed(
            tmp_npz,
            image=image,
            label=np.array(label, dtype=np.int64),
            sampling_label=np.array(sampling_label),
            patient_id=np.array(patient_id),
            local_z=np.array(z),
            row_id=np.array(row_id),
        )

        # tmp npz 저장 직후 존재 확인
        if not tmp_npz.exists():
            error_rows.append({
                "patient_id": patient_id,
                "row_id": row_id,
                "local_z": z,
                "error_type": "NPZ_SAVE_FAILED",
                "error_msg": f"tmp npz 저장 후 파일 없음: {tmp_npz}",
            })
            n_error += 1
            continue

        manifest_rows.append({
            "row_id":                       row_id,
            "patient_id":                   patient_id,
            "safe_id":                      safe_id,
            "npz_path":                     final_npz_str,
            "label":                        label,
            "sampling_label":               sampling_label,
            "stage_split":                  "stage2_holdout",
            "source_coordinate_manifest":   SOURCE_COORDINATE_MANIFEST_STR,
            "source_crop_root":             str(out_patient_dir_final.parent),
            "source_ct_path":               source_ct_path,
            "source_roi_path":              source_roi_path,
            "source_lesion_mask_path":      source_lesion_path,
            "asset_scope":                  "dedicated_stage2_holdout_6ch_crop",
            "contamination_check_status":   contamination_status,
            "approval_required_before_scoring": True,
            "manifest_status":              "created_after_phase8_2f_run",
            "crop_shape":                   "(6,96,96)",
            "input_channels":               6,
            "crop_size":                    CROP_SIZE,
            "generation_status":            "generated",
            "issue":                        "",
            "note":                         "",
        })
        n_success += 1

    del ct
    gc.collect()
    return {"success": n_success, "error": n_error}


# ── 후처리 검증 ────────────────────────────────────────────────────────────────
def post_validate(manifest_rows: list, error_rows: list, is_partial: bool) -> dict:
    df_out = pd.DataFrame(manifest_rows)
    issues = []

    actual_total    = len(df_out)
    actual_patients = df_out["patient_id"].nunique() if len(df_out) else 0
    actual_pos      = int((df_out["label"] == 1).sum()) if len(df_out) else 0
    actual_hn       = int((df_out["label"] == 0).sum()) if len(df_out) else 0
    error_count     = len(error_rows)

    contamination_check = "SKIPPED_PARTIAL_RUN"

    if not is_partial:
        if actual_total != EXPECTED_TOTAL:
            issues.append(f"total_rows: expected={EXPECTED_TOTAL}, got={actual_total}")
        if actual_patients != EXPECTED_PATIENTS:
            issues.append(f"patient_count: expected={EXPECTED_PATIENTS}, got={actual_patients}")
        if actual_pos != EXPECTED_POSITIVE:
            issues.append(f"positive_count: expected={EXPECTED_POSITIVE}, got={actual_pos}")
        if actual_hn != EXPECTED_HN:
            issues.append(f"hard_negative_count: expected={EXPECTED_HN}, got={actual_hn}")

        # LUNG1-295, LUNG1-415 포함 확인 (full run에서만)
        contamination_issues = []
        for pid in CONTAMINATED_PATIENTS:
            if pid not in df_out["patient_id"].values:
                contamination_issues.append(f"contaminated patient missing in manifest: {pid}")
            else:
                status = df_out.loc[df_out["patient_id"] == pid, "contamination_check_status"].iloc[0]
                expected = "regenerated_crop_from_source_after_prior_contamination"
                if status != expected:
                    contamination_issues.append(f"{pid} contamination_check_status mismatch: {status}")
        if contamination_issues:
            issues.extend(contamination_issues)
            contamination_check = "FAILED"
        else:
            contamination_check = "PASSED"

    npz_dup_count   = 0
    npz_empty_count = 0
    if len(df_out) > 0:
        # duplicate npz_path
        npz_dup_count = int(df_out["npz_path"].duplicated().sum())
        if npz_dup_count > 0:
            issues.append(f"duplicate npz_path: {npz_dup_count}건")

        # empty/null npz_path
        npz_null = int(df_out["npz_path"].isna().sum())
        npz_blank = int((df_out["npz_path"].astype(str).str.strip() == "").sum())
        npz_empty_count = npz_null + npz_blank
        if npz_empty_count > 0:
            issues.append(f"empty/null npz_path: {npz_empty_count}건")

        # approval_required_before_scoring 전 행 True
        ap_norm = df_out["approval_required_before_scoring"].astype(str).str.strip().str.lower()
        ap_invalid = int((~ap_norm.isin(["true", "1"])).sum())
        if ap_invalid > 0:
            issues.append(f"approval_required_before_scoring invalid: {ap_invalid}건")

    return {
        "actual_total":        actual_total,
        "actual_patients":     actual_patients,
        "actual_positive":     actual_pos,
        "actual_hn":           actual_hn,
        "error_count":         error_count,
        "npz_dup_count":       npz_dup_count,
        "npz_empty_count":     npz_empty_count,
        "contamination_check": contamination_check,
        "issues":              issues,
        "passed":              len(issues) == 0,
    }


# ── DONE marker ────────────────────────────────────────────────────────────────
def write_done(summary: dict, done_path: pathlib.Path) -> None:
    with open(done_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "DONE",
                "created_at": datetime.datetime.now().isoformat(),
                "total_crops": summary.get("total_success"),
                "error_count": summary.get("total_error"),
                "patients": summary.get("patient_count"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8.2F stage2 dedicated 6ch crop generation"
    )
    parser.add_argument("--run",          action="store_true", help="실행 모드 (--confirm-run과 함께)")
    parser.add_argument("--confirm-run",  action="store_true", help="이중 승인 플래그")
    parser.add_argument("--max-patients", type=int, default=0,  help="최대 처리 환자 수 (smoke용)")
    parser.add_argument("--max-rows",     type=int, default=0,  help="최대 처리 행 수 (smoke용)")
    args = parser.parse_args()

    is_full_run  = args.run and args.confirm_run
    max_patients = args.max_patients
    max_rows     = args.max_rows
    is_smoke     = max_patients > 0 or max_rows > 0

    if not is_full_run:
        # dry-run
        if not MANIFEST_PATH.exists():
            print(f"[ABORT] manifest 없음: {MANIFEST_PATH}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(MANIFEST_PATH)
        dry_run_report(df, max_patients, max_rows)
        print("[DRY-RUN] 파일 생성 없음. 실행하려면 --run --confirm-run 을 사용하세요.")
        return

    # ── smoke/full 경로 선택 ──────────────────────────────────────────────────
    if is_smoke:
        crop_tmp_root  = SMOKE_CROP_TMP_ROOT
        crop_final_root = SMOKE_CROP_FINAL_ROOT
        manifest_tmp   = SMOKE_MANIFEST_TMP
        manifest_final = SMOKE_MANIFEST_FINAL
        run_out_root   = SMOKE_RUN_OUT_ROOT
        out_report_md  = SMOKE_OUT_REPORT_MD
        out_summary    = SMOKE_OUT_SUMMARY
        out_errors     = SMOKE_OUT_ERRORS
        out_runtime    = SMOKE_OUT_RUNTIME
        out_done       = SMOKE_OUT_DONE
    else:
        crop_tmp_root  = CROP_TMP_ROOT
        crop_final_root = CROP_FINAL_ROOT
        manifest_tmp   = MANIFEST_TMP
        manifest_final = MANIFEST_FINAL
        run_out_root   = RUN_OUT_ROOT
        out_report_md  = OUT_REPORT_MD
        out_summary    = OUT_SUMMARY
        out_errors     = OUT_ERRORS
        out_runtime    = OUT_RUNTIME
        out_done       = OUT_DONE

    # ── 실행 모드 ────────────────────────────────────────────────────────────
    run_start = datetime.datetime.now()
    mode_str = "SMOKE" if is_smoke else "FULL"
    print("=" * 70)
    print("  Phase 8.2F stage2 dedicated 6ch crop generation")
    print(f"  모드: {mode_str}")
    print(f"  시작: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if is_smoke:
        print(f"[INFO] SMOKE 모드 — smoke 전용 경로 사용 (full run 경로 미오염)")
        print(f"  smoke crop root: {crop_final_root}")

    # output guard (smoke/full 각자 경로만 검사)
    output_guard([
        (crop_final_root,  "final crop root"),
        (crop_tmp_root,    "tmp crop root"),
        (manifest_final,   "final crop manifest"),
        (manifest_tmp,     "tmp crop manifest"),
        (run_out_root,     "run output root"),
        (out_report_md,    "report MD"),
        (out_summary,      "summary JSON"),
        (out_errors,       "errors CSV"),
        (out_runtime,      "runtime summary CSV"),
        (out_done,         "DONE marker"),
    ])

    # RUN_OUT_ROOT 생성
    run_out_root.mkdir(parents=True, exist_ok=False)
    print(f"[OK] run output root 생성: {run_out_root}")

    # manifest 로드
    if not MANIFEST_PATH.exists():
        print(f"[ABORT] manifest 없음: {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(MANIFEST_PATH)
    print(f"[OK] manifest 로드: {len(df):,}행")

    # 사전 검증
    pre_issues = []
    if len(df) != EXPECTED_TOTAL:
        pre_issues.append(f"manifest row count mismatch: {len(df)} != {EXPECTED_TOTAL}")
    if df["patient_id"].nunique() != EXPECTED_PATIENTS:
        pre_issues.append(f"patient count mismatch: {df['patient_id'].nunique()} != {EXPECTED_PATIENTS}")
    v1v2_count = int((df["model_type"] != "v2v2").sum()) if "model_type" in df.columns else -1
    if v1v2_count != 0:
        pre_issues.append(f"non-v2v2 rows: {v1v2_count}")
    stage1_count = int((df["stage_split"] != "stage2_holdout").sum()) if "stage_split" in df.columns else -1
    if stage1_count != 0:
        pre_issues.append(f"non-stage2_holdout rows: {stage1_count}")
    if pre_issues:
        for msg in pre_issues:
            print(f"[ABORT] 사전 검증 실패: {msg}", file=sys.stderr)
        sys.exit(1)
    print("[OK] 사전 검증 통과")

    # source root 존재 확인
    if not SOURCE_VOL_ROOT.exists():
        print(f"[ABORT] source vol root 없음: {SOURCE_VOL_ROOT}", file=sys.stderr)
        sys.exit(1)

    # CROP_TMP_ROOT 생성
    crop_tmp_root.mkdir(parents=True, exist_ok=False)
    print(f"[OK] tmp crop root 생성: {crop_tmp_root}")

    # 환자 목록
    patients = sorted(df["patient_id"].unique().tolist())
    if max_patients > 0:
        patients = patients[:max_patients]
        print(f"[INFO] --max-patients={max_patients}: {len(patients)}명만 처리")

    is_partial = (max_patients > 0 and max_patients < EXPECTED_PATIENTS) or \
                 (max_rows > 0 and max_rows < EXPECTED_TOTAL)

    manifest_rows = []
    error_rows    = []
    runtime_rows  = []
    seen_npz_paths: set = set()
    total_rows_processed = 0

    for i, patient_id in enumerate(patients):
        patient_rows = df[df["patient_id"] == patient_id].copy()

        if max_rows > 0 and total_rows_processed >= max_rows:
            print(f"[INFO] --max-rows={max_rows} 도달. 이후 환자 건너뜀.")
            is_partial = True
            break

        if max_rows > 0:
            remaining = max_rows - total_rows_processed
            if len(patient_rows) > remaining:
                patient_rows = patient_rows.head(remaining)
                is_partial = True

        safe_id = str(patient_rows["safe_id"].iloc[0])
        vol_dir = SOURCE_VOL_ROOT / safe_id
        out_patient_tmp   = crop_tmp_root   / patient_id
        out_patient_final = crop_final_root / patient_id

        n_pos = int((patient_rows["label"] == 1).sum())
        n_hn  = int((patient_rows["label"] == 0).sum())
        print(
            f"[{i+1:3d}/{len(patients)}] {patient_id} "
            f"rows={len(patient_rows)} (pos={n_pos}, hn={n_hn}) safe_id={safe_id}"
        )

        p_start = datetime.datetime.now()
        result = process_patient(
            patient_id=patient_id,
            rows=patient_rows,
            vol_dir=vol_dir,
            out_patient_dir_tmp=out_patient_tmp,
            out_patient_dir_final=out_patient_final,
            error_rows=error_rows,
            manifest_rows=manifest_rows,
            seen_npz_paths=seen_npz_paths,
        )
        p_end = datetime.datetime.now()
        elapsed_p = (p_end - p_start).total_seconds()

        total_rows_processed += len(patient_rows)
        print(
            f"  → success={result['success']}, error={result['error']}, "
            f"elapsed={elapsed_p:.1f}s"
        )

        runtime_rows.append({
            "patient_id":       patient_id,
            "safe_id":          safe_id,
            "rows_attempted":   len(patient_rows),
            "rows_success":     result["success"],
            "rows_error":       result["error"],
            "elapsed_seconds":  round(elapsed_p, 2),
            "status":           "ok" if result["error"] == 0 else "partial_error",
        })

    run_end = datetime.datetime.now()
    total_elapsed = (run_end - run_start).total_seconds()
    total_success = len(manifest_rows)
    total_error   = len(error_rows)

    print()
    print(f"[INFO] 처리 완료: success={total_success:,}, error={total_error}")

    # 후처리 검증
    validation = post_validate(manifest_rows, error_rows, is_partial)
    print(f"[INFO] 후처리 검증: passed={validation['passed']}, issues={validation['issues']}")

    # error CSV 저장 (항상)
    import csv as csv_module
    error_fieldnames = ["patient_id", "row_id", "local_z", "error_type", "error_msg"]
    with open(out_errors, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=error_fieldnames)
        writer.writeheader()
        writer.writerows(error_rows)
    print(f"[OK] error CSV 저장: {out_errors}")

    # runtime summary CSV 저장
    rt_fieldnames = ["patient_id", "safe_id", "rows_attempted", "rows_success", "rows_error", "elapsed_seconds", "status"]
    with open(out_runtime, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=rt_fieldnames)
        writer.writeheader()
        writer.writerows(runtime_rows)
    print(f"[OK] runtime summary 저장: {out_runtime}")

    # ── tmp manifest → tmp file ────────────────────────────────────────────
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(manifest_tmp, index=False, encoding="utf-8")
    print(f"[OK] tmp manifest 저장: {manifest_tmp}")

    # ── rename: tmp crop root → final ─────────────────────────────────────
    npz_exists_count = 0
    npz_missing_count = 0

    if not validation["passed"]:
        print(
            f"[WARN] 후처리 검증 실패. tmp crop root/manifest를 final로 rename하지 않습니다.",
            file=sys.stderr,
        )
        print(f"  issues: {validation['issues']}", file=sys.stderr)
    else:
        crop_tmp_root.rename(crop_final_root)
        print(f"[OK] crop root rename: {crop_tmp_root.name} → {crop_final_root.name}")

        manifest_tmp.rename(manifest_final)
        print(f"[OK] manifest rename: {manifest_tmp.name} → {manifest_final.name}")

        # final npz_path 존재 검증 (rename 후)
        npz_missing = []
        for r in manifest_rows:
            fp = pathlib.Path(r["npz_path"])
            if fp.exists():
                npz_exists_count += 1
            else:
                npz_missing.append(r["npz_path"])
        npz_missing_count = len(npz_missing)

        if npz_missing_count > 0:
            print(f"[ERROR] final npz_path missing {npz_missing_count}건", file=sys.stderr)
            for p in npz_missing[:5]:
                print(f"  {p}", file=sys.stderr)
            validation["passed"] = False
            validation["issues"].append(f"final_npz_path_missing: {npz_missing_count}건")
        else:
            print(f"[OK] final npz_path 존재 확인: {npz_exists_count}건 전부 OK")

    validation["npz_exists_count"]  = npz_exists_count
    validation["npz_missing_count"] = npz_missing_count

    # ── summary JSON ──────────────────────────────────────────────────────
    summary = {
        "script":               "phase8_2f_stage2_dedicated_6ch_crop_generation.py",
        "mode":                 "smoke" if is_smoke else "full",
        "created_at":           run_end.isoformat(),
        "run_start":            run_start.isoformat(),
        "run_end":              run_end.isoformat(),
        "elapsed_seconds":      round(total_elapsed, 2),
        "is_partial_run":       is_partial,
        "coordinate_manifest":  str(MANIFEST_PATH),
        "crop_final_root":      str(crop_final_root),
        "manifest_final":       str(manifest_final),
        "crop_size":            CROP_SIZE,
        "input_channels":       6,
        "image_key":            "image",
        "channels_desc":        "ch0-2: lung_window [-1350,150]->[0,1], ch3-5: medi_window [-160,240]->[0,1]",
        "z_boundary":           "edge-repeat",
        "total_success":        total_success,
        "total_error":          total_error,
        "patient_count":        len(set(r["patient_id"] for r in manifest_rows)),
        "positive_count":       int(df_manifest["label"].eq(1).sum()) if len(df_manifest) else 0,
        "hard_negative_count":  int(df_manifest["label"].eq(0).sum()) if len(df_manifest) else 0,
        "npz_path_exists_count":    validation.get("npz_exists_count", 0),
        "npz_path_missing_count":   validation.get("npz_missing_count", 0),
        "npz_path_duplicate_count": validation.get("npz_dup_count", 0),
        "npz_path_empty_null_count": validation.get("npz_empty_count", 0),
        "post_validation":      validation,
        "notes": {
            "no_model_forward":         True,
            "no_scoring":               True,
            "no_metric_calculation":    True,
            "no_threshold":             True,
            "no_training":              True,
            "no_mixed_crop_modification": True,
        },
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] summary JSON 저장: {out_summary}")

    # ── MD report ─────────────────────────────────────────────────────────
    status_str = "DONE" if (total_error == 0 and validation["passed"] and not is_partial) else "PARTIAL / ERROR"
    md_content = f"""# Phase 8.2F Stage2 Dedicated 6ch Crop Generation Report

생성일: {run_end.isoformat()}
모드: {mode_str}

## 실행 요약

| 항목 | 값 |
|------|-----|
| 총 성공 crop | {total_success:,} |
| 총 오류 row | {total_error} |
| patient count | {summary['patient_count']} |
| positive count | {summary['positive_count']:,} |
| hard_negative count | {summary['hard_negative_count']:,} |
| 소요 시간 | {total_elapsed:.1f}s |
| partial run | {is_partial} |
| 후처리 검증 통과 | {validation['passed']} |
| status | {status_str} |

## npz_path 검증

| 항목 | 값 |
|------|-----|
| npz_path exists count | {validation.get('npz_exists_count', 0)} |
| npz_path missing count | {validation.get('npz_missing_count', 0)} |
| npz_path duplicate count | {validation.get('npz_dup_count', 0)} |
| npz_path empty/null count | {validation.get('npz_empty_count', 0)} |

## 후처리 검증 이슈

{chr(10).join('- ' + iss for iss in validation['issues']) if validation['issues'] else '없음'}

## 금지 사항 준수

- no model forward: True
- no scoring: True
- no metric calculation: True
- no threshold: True
- no training: True
- no mixed crop modification: True
"""
    with open(out_report_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] report MD 저장: {out_report_md}")

    # ── DONE marker ───────────────────────────────────────────────────────
    if total_error == 0 and validation["passed"] and not is_partial:
        write_done(summary, out_done)
        print(f"[OK] DONE marker 생성: {out_done}")
    else:
        reasons = []
        if total_error > 0:
            reasons.append(f"error_count={total_error}")
        if not validation["passed"]:
            reasons.append(f"validation_issues={validation['issues']}")
        if is_partial:
            reasons.append("partial_run")
        print(f"[INFO] DONE marker 생성 안 함: {reasons}")

    print()
    print("=" * 70)
    print(f"  최종 status: {status_str}")
    print(f"  crop root  : {crop_final_root}")
    print(f"  manifest   : {manifest_final}")
    print("=" * 70)


if __name__ == "__main__":
    main()
