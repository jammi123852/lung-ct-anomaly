#!/usr/bin/env python3
"""
generate_s4_crop_full.py

Second-Stage Lesion Refiner v1 — S4 Full Crop 생성 스크립트

목적:
- rule_c4_training_sampling_manifest_dryrun.csv에서 S4_patient_balanced만 사용
- stage1_dev 154명 전체에 대해 2.5D crop (3, 96, 96) npz 생성
- 총 48,411개 crop → crops_s4_full/ 전용 경로에만 저장

IMPORTANT:
- stage1_dev 환자만 처리 (stage2_holdout 포함 시 즉시 중단)
- crops_s4_smoke/ 경로와 절대 섞이면 안 됨
- 학습/평가/score 재계산 금지
- 기존 smoke/candidate/score/evaluation 결과 수정 금지
- ChatGPT 검토 + 사용자 승인 후에만 실행 가능

Guard 조건:
- crops_s4_full/ 안에 기존 npz 있으면 즉시 중단
- S4 대상 환자 수 != 154명이면 중단
- positive + hard_negative 합계 != 48,411이면 중단
- stage2_holdout 1명이라도 포함 시 즉시 중단
- 필수 컬럼 누락 시 중단
- free space < 10GB 시 중단
- volume 파일 누락 시 중단 (해당 환자 skip + error.csv 기록)
- crop shape != (3, 96, 96)이면 즉시 중단

2.5D crop 방식:
- 중심 slice (slice_index) 기준 z-1, z, z+1
- z boundary: edge-repeat (z<1 → z=0, z>=Z → z=Z-1)
- crop 좌표: y0,x0,y1,x1 중심 기반 96×96 고정 crop

출력 경로:
- crops:    outputs/second-stage-lesion-refiner-v1/crops_s4_full/{patient_id}/{idx:06d}.npz
- summary:  outputs/second-stage-lesion-refiner-v1/reports/crop_s4_full_summary.csv
            outputs/second-stage-lesion-refiner-v1/reports/crop_s4_full_summary.json
- error:    outputs/second-stage-lesion-refiner-v1/reports/crop_s4_full_error.csv

syntax check (실행 아님):
  python -m py_compile scripts/generate_s4_crop_full.py

실행 (ChatGPT 검토 + 사용자 승인 후):
  source ~/ai_env/bin/activate && \\
  python scripts/generate_s4_crop_full.py
"""

import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ============================================================
# 경로 상수
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/candidates"
    / "rule_c4_training_sampling_manifest_dryrun.csv"
)
PATHS_CONFIG = REPO_ROOT / "configs/paths.local.yaml"
OUTPUT_BASE = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1"
CROPS_FULL_DIR = OUTPUT_BASE / "crops_s4_full"
CROPS_SMOKE_DIR = OUTPUT_BASE / "crops_s4_smoke"  # 접근 금지 (참조 전용)
REPORTS_DIR = OUTPUT_BASE / "reports"

# v1 outputs 쓰기 절대 금지
V1_OUTPUT_DIR = REPO_ROOT / "outputs/position-aware-padim-v1"

SCRIPT_NAME = "generate_s4_crop_full.py"
SAMPLING_RULE_TARGET = "S4_patient_balanced"
EXPECTED_PATIENT_COUNT = 154
EXPECTED_TOTAL_CROPS = 48_411
CROP_SIZE = 96
MIN_FREE_SPACE_GB = 10.0
EXPECTED_CROP_SHAPE = (3, CROP_SIZE, CROP_SIZE)

REQUIRED_COLUMNS = [
    "patient_id", "stage_split", "sampling_rule", "sampling_label",
    "slice_index", "local_z", "y0", "x0", "y1", "x1",
    "padim_score", "patch_label", "lesion_overlap",
    "no_positive_patient", "position_bin", "z_level",
    "derived_grid_position_bin",
]

VALID_LABELS = {"positive", "hard_negative"}

# ============================================================
# 안전 유틸
# ============================================================

def abort(msg: str) -> None:
    print(f"[ABORT] {msg}", file=sys.stderr)
    sys.exit(1)


def check_not_v1_output(path: Path) -> None:
    try:
        path.relative_to(V1_OUTPUT_DIR)
        abort(f"v1 output 경로에 쓰기 시도 감지: {path}")
    except ValueError:
        pass


def check_not_smoke_dir(path: Path) -> None:
    try:
        path.relative_to(CROPS_SMOKE_DIR)
        abort(f"smoke 경로에 쓰기 시도 감지: {path}")
    except ValueError:
        pass


def load_paths_config() -> Dict:
    if not PATHS_CONFIG.exists():
        abort(f"paths.local.yaml 없음: {PATHS_CONFIG}")
    with open(PATHS_CONFIG, "r") as f:
        return yaml.safe_load(f) or {}


def get_v2_volume_root(paths_cfg: Dict) -> Path:
    v2_path = paths_cfg.get("nsclc_msd_usable_only_v2", "")
    if not v2_path:
        abort("paths.local.yaml에 nsclc_msd_usable_only_v2 경로가 없습니다.")
    p = Path(v2_path)
    if not p.exists():
        abort(f"v2 volume 경로가 존재하지 않습니다: {p}")
    return p


def build_patient_to_safeid(v2_root: Path) -> Dict[str, str]:
    manifest_csv = v2_root / "manifests" / "patient_manifest.csv"
    if not manifest_csv.exists():
        abort(f"patient_manifest.csv 없음: {manifest_csv}")
    df = pd.read_csv(manifest_csv, usecols=["patient_id", "safe_id"])
    return dict(zip(df["patient_id"].astype(str), df["safe_id"].astype(str)))


def check_free_space(path: Path, min_gb: float) -> None:
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    free_gb = usage.free / (1024 ** 3)
    print(f"[INFO] free space: {free_gb:.1f} GB (최소 요구: {min_gb} GB)")
    if free_gb < min_gb:
        abort(f"여유 공간 부족: {free_gb:.1f} GB < {min_gb} GB — 중단")


def load_split_csv() -> pd.DataFrame:
    split_csv = (
        REPO_ROOT
        / "outputs/second-stage-lesion-refiner-v1/splits"
        / "lesion_stage_split_v1_balanced.csv"
    )
    if not split_csv.exists():
        abort(f"split CSV 없음: {split_csv}")
    return pd.read_csv(split_csv, usecols=["patient_id", "stage_split"])


# ============================================================
# Guard 검사
# ============================================================

def guard_no_existing_npz() -> None:
    if CROPS_FULL_DIR.exists():
        existing = list(CROPS_FULL_DIR.rglob("*.npz"))
        if existing:
            abort(
                f"crops_s4_full/ 안에 기존 npz {len(existing)}개 존재 — 중단.\n"
                f"  첫 번째: {existing[0]}"
            )


def guard_manifest(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        abort(f"manifest 필수 컬럼 누락: {missing}")

    s4 = df[df["sampling_rule"] == SAMPLING_RULE_TARGET].copy()
    if s4.empty:
        abort(f"manifest에서 {SAMPLING_RULE_TARGET} 행 없음")

    # stage2_holdout 포함 여부
    holdout_in_s4 = s4[s4["stage_split"] == "stage2_holdout"]
    if not holdout_in_s4.empty:
        abort(
            f"stage2_holdout 환자 {holdout_in_s4['patient_id'].nunique()}명 포함 감지 — 즉시 중단"
        )

    # stage1_dev만 추출
    s4_dev = s4[s4["stage_split"] == "stage1_dev"].copy()

    patient_count = s4_dev["patient_id"].nunique()
    if patient_count != EXPECTED_PATIENT_COUNT:
        abort(
            f"S4 stage1_dev 환자 수 불일치: 예상 {EXPECTED_PATIENT_COUNT}명, 실제 {patient_count}명"
        )

    total_crops = len(s4_dev)
    if total_crops != EXPECTED_TOTAL_CROPS:
        abort(
            f"S4 총 crop 수 불일치: 예상 {EXPECTED_TOTAL_CROPS:,}개, 실제 {total_crops:,}개"
        )

    # label 유효성 검사
    invalid_labels = set(s4_dev["sampling_label"].unique()) - VALID_LABELS
    if invalid_labels:
        abort(f"sampling_label에 예상 외 값 존재: {invalid_labels}")

    pos = (s4_dev["sampling_label"] == "positive").sum()
    neg = (s4_dev["sampling_label"] == "hard_negative").sum()
    print(f"[INFO] S4 stage1_dev: 환자 {patient_count}명, 총 {total_crops:,}개 (pos={pos:,}, neg={neg:,})")

    return s4_dev


# ============================================================
# crop 좌표 계산
# ============================================================

def compute_fixed_crop_coords(
    cy: float, cx: float, crop_size: int, img_h: int, img_w: int
) -> Tuple[int, int, int, int]:
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


def load_ct_volume(volume_dir: Path) -> np.ndarray:
    ct_path = volume_dir / "ct_hu.npy"
    if not ct_path.exists():
        raise FileNotFoundError(f"ct_hu.npy 없음: {ct_path}")
    ct = np.load(ct_path, mmap_mode="r")
    if ct.ndim != 3:
        raise ValueError(f"ct_hu.npy shape 오류 (Z,H,W) 기대, 실제: {ct.shape}")
    return ct


def extract_2p5d_crop(
    ct: np.ndarray, z: int, y0: int, x0: int, y1: int, x1: int
) -> np.ndarray:
    Z = ct.shape[0]
    z_prev = max(0, z - 1)
    z_next = min(Z - 1, z + 1)
    crop = np.stack([
        ct[z_prev, y0:y1, x0:x1].astype(np.float32),
        ct[z, y0:y1, x0:x1].astype(np.float32),
        ct[z_next, y0:y1, x0:x1].astype(np.float32),
    ], axis=0)
    if crop.shape != EXPECTED_CROP_SHAPE:
        raise ValueError(
            f"crop shape 불일치: 기대 {EXPECTED_CROP_SHAPE}, 실제 {crop.shape} "
            f"(z={z}, y0={y0}, x0={x0}, y1={y1}, x1={x1})"
        )
    return crop


# ============================================================
# 환자 처리
# ============================================================

def process_patient(
    patient_id: str,
    rows: pd.DataFrame,
    volume_dir: Path,
    out_dir: Path,
    error_rows: List[Dict],
) -> Tuple[int, int]:
    """(saved_count, error_count) 반환"""
    try:
        ct = load_ct_volume(volume_dir)
    except FileNotFoundError as e:
        error_rows.append({"patient_id": patient_id, "error": str(e), "stage": "volume_load"})
        return 0, 1

    _, img_h, img_w = ct.shape
    saved = 0
    errors = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    check_not_v1_output(out_dir)
    check_not_smoke_dir(out_dir)

    for idx, row in rows.reset_index(drop=True).iterrows():
        try:
            bbox_cy = (row["y0"] + row["y1"]) / 2.0
            bbox_cx = (row["x0"] + row["x1"]) / 2.0
            cy0, cx0, cy1, cx1 = compute_fixed_crop_coords(
                bbox_cy, bbox_cx, CROP_SIZE, img_h, img_w
            )
                # local_z guard
            if "local_z" not in row.index:
                raise ValueError("local_z 컬럼 없음 — 중단")
            if pd.isna(row["local_z"]):
                raise ValueError(f"local_z NaN (slice_index={row['slice_index']}) — 중단")
            z = int(row["local_z"])
            if z < 0:
                raise ValueError(f"local_z < 0: {z} — 중단")
            if z >= ct.shape[0]:
                raise ValueError(f"local_z={z} >= ct.shape[0]={ct.shape[0]} — 중단")
            crop_arr = extract_2p5d_crop(ct, z, cy0, cx0, cy1, cx1)

            npz_path = out_dir / f"{idx:06d}.npz"
            check_not_v1_output(npz_path)
            check_not_smoke_dir(npz_path)

            np.savez_compressed(
                npz_path,
                crop=crop_arr,
                label=np.array(str(row["sampling_label"])),
                patient_id=np.array(patient_id),
                local_z=np.array(z),
                slice_index=np.array(int(row["slice_index"])),
                z_source=np.array("local_z"),
                crop_coords=np.array([cy0, cx0, cy1, cx1]),
                orig_bbox=np.array([int(row["y0"]), int(row["x0"]), int(row["y1"]), int(row["x1"])]),
                padim_score=np.array(float(row["padim_score"])),
                sampling_label=np.array(str(row["sampling_label"])),
                sampling_rule=np.array(str(row["sampling_rule"])),
                patch_label=np.array(str(row["patch_label"])),
                lesion_overlap=np.array(float(row["lesion_overlap"])),
                no_positive_patient=np.array(bool(row["no_positive_patient"])),
                position_bin=np.array(str(row["position_bin"])),
                z_level=np.array(str(row["z_level"])),
                derived_grid_position_bin=np.array(str(row["derived_grid_position_bin"])),
            )
            saved += 1
        except Exception as e:
            errors += 1
            error_rows.append({
                "patient_id": patient_id,
                "row_idx": idx,
                "slice_index": int(row.get("slice_index", -1)),
                "error": str(e),
                "stage": "crop_save",
            })

    return saved, errors


# ============================================================
# 메인
# ============================================================

def main() -> None:
    print(f"[{SCRIPT_NAME}] 시작: {datetime.now().isoformat()}")
    run_start = datetime.now()

    # ── Guard: 기존 npz 존재 여부 ────────────────────────
    guard_no_existing_npz()

    # ── 경로 설정 ────────────────────────────────────────
    paths_cfg = load_paths_config()
    v2_root = get_v2_volume_root(paths_cfg)
    patient_to_safeid = build_patient_to_safeid(v2_root)

    # ── Guard: free space ────────────────────────────────
    check_free_space(REPO_ROOT, MIN_FREE_SPACE_GB)

    # ── manifest 로딩 및 Guard ───────────────────────────
    if not MANIFEST_PATH.exists():
        abort(f"manifest 없음: {MANIFEST_PATH}")
    df = pd.read_csv(MANIFEST_PATH)
    s4_dev = guard_manifest(df)

    # ── split CSV에서 holdout 목록 확인 (이중 검사) ───────
    split_df = load_split_csv()
    holdout_set = set(split_df.loc[split_df["stage_split"] == "stage2_holdout", "patient_id"])
    s4_patients = set(s4_dev["patient_id"].unique())
    overlap = s4_patients & holdout_set
    if overlap:
        abort(f"stage2_holdout 환자가 S4 대상에 포함됨: {overlap}")

    # ── 환자별 처리 ──────────────────────────────────────
    error_rows: List[Dict] = []
    total_saved = 0
    total_errors = 0
    patients = sorted(s4_dev["patient_id"].unique())

    for i, patient_id in enumerate(patients, 1):
        safe_id = patient_to_safeid.get(patient_id)
        if safe_id is None:
            error_rows.append({
                "patient_id": patient_id, "error": "safe_id 매핑 없음", "stage": "mapping"
            })
            total_errors += 1
            continue

        volume_dir = v2_root / "volumes_npy" / safe_id
        if not volume_dir.exists():
            error_rows.append({
                "patient_id": patient_id, "error": f"volume_dir 없음: {volume_dir}", "stage": "volume_dir"
            })
            total_errors += 1
            continue

        patient_rows = s4_dev[s4_dev["patient_id"] == patient_id]
        out_dir = CROPS_FULL_DIR / patient_id

        saved, errors = process_patient(
            patient_id=patient_id,
            rows=patient_rows,
            volume_dir=volume_dir,
            out_dir=out_dir,
            error_rows=error_rows,
        )
        total_saved += saved
        total_errors += errors

        pos_cnt = (patient_rows["sampling_label"] == "positive").sum()
        neg_cnt = (patient_rows["sampling_label"] == "hard_negative").sum()
        npp = bool(patient_rows["no_positive_patient"].iloc[0])
        print(
            f"  [{i:3d}/{len(patients)}] {patient_id}: "
            f"rows={len(patient_rows)}, saved={saved}, errors={errors}, "
            f"pos={pos_cnt}, neg={neg_cnt}, npp={npp}"
        )

    # ── summary 저장 ─────────────────────────────────────
    run_end = datetime.now()
    elapsed = (run_end - run_start).total_seconds()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if error_rows:
        error_csv = REPORTS_DIR / "crop_s4_full_error.csv"
        check_not_v1_output(error_csv)
        pd.DataFrame(error_rows).to_csv(error_csv, index=False)
        print(f"[WARN] error.csv 저장: {error_csv} ({len(error_rows)}건)")

    summary = {
        "script": SCRIPT_NAME,
        "sampling_rule": SAMPLING_RULE_TARGET,
        "crop_size": CROP_SIZE,
        "z_boundary_method": "edge-repeat",
        "crop_shape": list(EXPECTED_CROP_SHAPE),
        "total_patients": len(patients),
        "total_crops_expected": EXPECTED_TOTAL_CROPS,
        "total_saved": total_saved,
        "total_errors": total_errors,
        "run_start": run_start.isoformat(),
        "run_end": run_end.isoformat(),
        "elapsed_seconds": round(elapsed, 2),
    }

    summary_csv_path = REPORTS_DIR / "crop_s4_full_summary.csv"
    summary_json_path = REPORTS_DIR / "crop_s4_full_summary.json"
    check_not_v1_output(summary_csv_path)
    check_not_v1_output(summary_json_path)

    s4_dev_copy = s4_dev.copy()
    s4_dev_copy["saved"] = True
    s4_dev_copy.to_csv(summary_csv_path, index=False)

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] 총 저장: {total_saved:,}개 / 오류: {total_errors}건")
    print(f"[DONE] 경과 시간: {elapsed:.1f}초")
    print(f"[DONE] summary JSON: {summary_json_path}")
    if total_errors > 0:
        print(f"[WARN] error.csv 확인 요망: {REPORTS_DIR / 'crop_s4_full_error.csv'}")


if __name__ == "__main__":
    main()
