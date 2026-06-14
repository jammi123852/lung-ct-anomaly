#!/usr/bin/env python3
"""
validate_s4_full_crops.py

crops_s4_full 폴더에 생성된 48,411개 npz 무결성 검증 스크립트.
모델 학습 전 데이터 무결성 확인이 목적이다.

검증 항목:
  1. 파일 개수 검증
  2. stage split 검증
  3. npz key 검증 (전수검사)
  4. crop shape / dtype / NaN/Inf 검증
  5. label 검증
  6. local_z 검증
  7. 좌표 검증 (전수검사)
  8. 환자별 분포 검증
  9. 그룹별 분포 검증
 10. 샘플 로드 검증

출력:
  outputs/second-stage-lesion-refiner-v1/reports/crops_s4_full_validation_summary.csv
  outputs/second-stage-lesion-refiner-v1/reports/crops_s4_full_validation_summary.json
  outputs/second-stage-lesion-refiner-v1/reports/crops_s4_full_validation_summary.md

syntax check (실행 아님):
  python -m py_compile scripts/validate_s4_full_crops.py

실행:
  source ~/ai_env/bin/activate && \\
  python scripts/validate_s4_full_crops.py
"""

import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ============================================================
# 경로 상수
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
CROPS_FULL_DIR = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_s4_full"
MANIFEST_PATH = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/candidates/rule_c4_training_sampling_manifest_dryrun.csv"
SPLIT_CSV_PATH = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
PATHS_CONFIG = REPO_ROOT / "configs/paths.local.yaml"
REPORTS_DIR = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports"
OUTPUT_CSV = REPORTS_DIR / "crops_s4_full_validation_summary.csv"
OUTPUT_JSON = REPORTS_DIR / "crops_s4_full_validation_summary.json"
OUTPUT_MD = REPORTS_DIR / "crops_s4_full_validation_summary.md"

# ============================================================
# 상수
# ============================================================
SAMPLING_RULE_TARGET = "S4_patient_balanced"
EXPECTED_PATIENT_COUNT = 154
EXPECTED_TOTAL_CROPS = 48_411
EXPECTED_CROP_SHAPE = (3, 96, 96)
EXPECTED_POS = 18_269
EXPECTED_NEG = 30_142
EXPECTED_KEYS = [
    "crop", "crop_coords", "derived_grid_position_bin", "label",
    "lesion_overlap", "local_z", "no_positive_patient", "orig_bbox",
    "padim_score", "patch_label", "patient_id", "position_bin",
    "sampling_label", "sampling_rule", "slice_index", "z_level", "z_source"
]
NPP_PATIENTS = {"LUNG1-156", "LUNG1-415", "MSD_lung_071", "MSD_lung_096"}

SAMPLE_COUNT = 500  # NaN/Inf 랜덤 샘플 수


# ============================================================
# 유틸
# ============================================================

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_group(patient_id: str) -> str:
    if patient_id.startswith("LUNG1-"):
        return "NSCLC"
    elif patient_id.startswith("MSD_lung_"):
        return "MSD_Lung"
    else:
        return "UNKNOWN"


def collect_all_npz() -> Dict[str, List[Path]]:
    """환자별 npz 파일 목록 수집. {patient_id: [Path, ...]}"""
    patient_dirs = sorted(p for p in CROPS_FULL_DIR.iterdir() if p.is_dir())
    result: Dict[str, List[Path]] = {}
    for pd_dir in patient_dirs:
        files = sorted(pd_dir.glob("*.npz"))
        if files:
            result[pd_dir.name] = files
    return result


# ============================================================
# 검증 1: 파일 개수 검증
# ============================================================

def check_1_file_count(patient_npz_map: Dict[str, List[Path]], manifest_df: pd.DataFrame) -> dict:
    total_npz = sum(len(v) for v in patient_npz_map.values())
    patient_count = len(patient_npz_map)

    # manifest에서 S4_patient_balanced 행 수
    manifest_s4 = manifest_df[manifest_df["sampling_rule"] == SAMPLING_RULE_TARGET]
    manifest_count = len(manifest_s4)

    result = {
        "total_npz": total_npz,
        "expected_total_npz": EXPECTED_TOTAL_CROPS,
        "total_npz_pass": total_npz == EXPECTED_TOTAL_CROPS,
        "patient_count": patient_count,
        "expected_patient_count": EXPECTED_PATIENT_COUNT,
        "patient_count_pass": patient_count == EXPECTED_PATIENT_COUNT,
        "manifest_s4_count": manifest_count,
        "manifest_match": manifest_count == EXPECTED_TOTAL_CROPS,
    }
    log(f"[1] 파일 개수: npz={total_npz}, 환자={patient_count}, manifest_s4={manifest_count}")
    return result


# ============================================================
# 검증 2: stage split 검증
# ============================================================

def check_2_stage_split(patient_npz_map: Dict[str, List[Path]], split_df: pd.DataFrame) -> dict:
    crops_patients = set(patient_npz_map.keys())

    # split CSV에서 각 환자의 stage_split 확인
    split_lookup = dict(zip(split_df["patient_id"].astype(str), split_df["stage_split"].astype(str)))

    stage2_holdout_patients: List[str] = []
    not_in_split: List[str] = []

    for pid in crops_patients:
        stage = split_lookup.get(pid)
        if stage is None:
            not_in_split.append(pid)
        elif stage == "stage2_holdout":
            stage2_holdout_patients.append(pid)

    pass_flag = len(stage2_holdout_patients) == 0

    result = {
        "stage2_holdout_patients": sorted(stage2_holdout_patients),
        "not_in_split": sorted(not_in_split),
        "pass": pass_flag,
    }
    log(f"[2] stage split: stage2_holdout={len(stage2_holdout_patients)}, not_in_split={len(not_in_split)}, pass={pass_flag}")
    return result


# ============================================================
# 검증 3: npz key 검증 (전수검사)
# ============================================================

def check_3_key_check(patient_npz_map: Dict[str, List[Path]]) -> Tuple[dict, Dict[str, int]]:
    expected_set = set(EXPECTED_KEYS)
    total_missing = 0
    patient_missing: Dict[str, int] = {}

    total = sum(len(v) for v in patient_npz_map.values())
    checked = 0

    for pid, files in patient_npz_map.items():
        pid_missing = 0
        for fpath in files:
            d = np.load(fpath, allow_pickle=True)
            actual_keys = set(d.files)
            missing = expected_set - actual_keys
            if missing:
                pid_missing += 1
                total_missing += 1
            d.close()
            checked += 1
            if checked % 5000 == 0:
                log(f"  key check 진행: {checked}/{total}")

        if pid_missing > 0:
            patient_missing[pid] = pid_missing

    patient_count_with_missing = len(patient_missing)
    result = {
        "total_missing": total_missing,
        "patient_count_with_missing": patient_count_with_missing,
        "pass": total_missing == 0,
    }
    log(f"[3] key 검증: 누락 npz={total_missing}, 누락 환자={patient_count_with_missing}")
    return result, patient_missing


# ============================================================
# 검증 4: crop shape / dtype / NaN/Inf 검증
# ============================================================

def check_4_shape_dtype_nan(
    patient_npz_map: Dict[str, List[Path]],
    patient_stats: Dict[str, dict],
) -> dict:
    """
    전수검사: shape, dtype
    랜덤 500개: NaN/Inf
    patient_stats는 환자별 집계를 누적하는 dict (side-effect).
    """
    shape_pass = 0
    shape_fail = 0
    dtype_pass = 0
    dtype_fail = 0

    # 랜덤 샘플 500개 선택
    all_paths: List[Path] = []
    for files in patient_npz_map.values():
        all_paths.extend(files)
    random.seed(42)
    sample_paths = random.sample(all_paths, min(SAMPLE_COUNT, len(all_paths)))
    sample_set = set(id(p) for p in sample_paths)
    # id() 대신 path 문자열로 비교
    sample_path_strs = set(str(p) for p in sample_paths)

    nan_count = 0
    inf_count = 0
    crop_min_vals: List[float] = []
    crop_max_vals: List[float] = []

    total = sum(len(v) for v in patient_npz_map.values())
    checked = 0

    for pid, files in patient_npz_map.items():
        pid_shape_err = 0
        pid_nan = 0
        for fpath in files:
            d = np.load(fpath, allow_pickle=True)
            crop = d["crop"]

            # shape 검증
            if tuple(crop.shape) == EXPECTED_CROP_SHAPE:
                shape_pass += 1
            else:
                shape_fail += 1
                pid_shape_err += 1

            # dtype 검증
            if crop.dtype == np.float32:
                dtype_pass += 1
            else:
                dtype_fail += 1

            # NaN/Inf (샘플만)
            if str(fpath) in sample_path_strs:
                has_nan = bool(np.isnan(crop).any())
                has_inf = bool(np.isinf(crop).any())
                if has_nan:
                    nan_count += 1
                    pid_nan += 1
                if has_inf:
                    inf_count += 1
                crop_min_vals.append(float(crop.min()))
                crop_max_vals.append(float(crop.max()))

            d.close()
            checked += 1
            if checked % 5000 == 0:
                log(f"  shape/dtype check 진행: {checked}/{total}")

        # patient_stats에 shape_error_count 누적
        patient_stats.setdefault(pid, {})
        patient_stats[pid]["shape_error_count"] = pid_shape_err
        patient_stats[pid]["nan_count"] = pid_nan

    crop_val_summary = {}
    if crop_min_vals:
        crop_val_summary = {
            "sample_count": len(crop_min_vals),
            "global_min": float(min(crop_min_vals)),
            "global_max": float(max(crop_max_vals)),
            "mean_min": float(np.mean(crop_min_vals)),
            "mean_max": float(np.mean(crop_max_vals)),
        }

    result = {
        "shape_check": {
            "pass_count": shape_pass,
            "fail_count": shape_fail,
            "pass": shape_fail == 0,
        },
        "dtype_check": {
            "pass_count": dtype_pass,
            "fail_count": dtype_fail,
            "pass": dtype_fail == 0,
        },
        "nan_check": {
            "nan_count": nan_count,
            "inf_count": inf_count,
            "sample_size": len(sample_path_strs),
            "pass": (nan_count == 0 and inf_count == 0),
        },
        "crop_value_summary": crop_val_summary,
    }
    log(f"[4] shape: fail={shape_fail}, dtype: fail={dtype_fail}, nan={nan_count}, inf={inf_count} (샘플 {len(sample_path_strs)}개 기준)")
    return result


# ============================================================
# 검증 5: label 검증
# ============================================================

def check_5_label(
    patient_npz_map: Dict[str, List[Path]],
    patient_stats: Dict[str, dict],
) -> dict:
    total_pos = 0
    total_neg = 0
    total_mismatch = 0
    invalid_sampling_label = 0

    valid_sampling_labels = {"positive", "hard_negative"}

    total = sum(len(v) for v in patient_npz_map.values())
    checked = 0

    for pid, files in patient_npz_map.items():
        pid_pos = 0
        pid_neg = 0
        pid_mismatch = 0

        for fpath in files:
            d = np.load(fpath, allow_pickle=True)
            label = str(d["label"])
            sampling_label = str(d["sampling_label"])

            if sampling_label not in valid_sampling_labels:
                invalid_sampling_label += 1

            if label != sampling_label:
                pid_mismatch += 1
                total_mismatch += 1

            if sampling_label == "positive":
                pid_pos += 1
                total_pos += 1
            elif sampling_label == "hard_negative":
                pid_neg += 1
                total_neg += 1

            d.close()
            checked += 1
            if checked % 5000 == 0:
                log(f"  label check 진행: {checked}/{total}")

        patient_stats.setdefault(pid, {})
        patient_stats[pid]["pos_count"] = pid_pos
        patient_stats[pid]["neg_count"] = pid_neg
        patient_stats[pid]["label_mismatch_count"] = pid_mismatch

    pos_pass = total_pos == EXPECTED_POS
    neg_pass = total_neg == EXPECTED_NEG

    result = {
        "pos_count": total_pos,
        "neg_count": total_neg,
        "mismatch_count": total_mismatch,
        "invalid_sampling_label_count": invalid_sampling_label,
        "expected_pos": EXPECTED_POS,
        "expected_neg": EXPECTED_NEG,
        "pos_pass": pos_pass,
        "neg_pass": neg_pass,
        "mismatch_pass": total_mismatch == 0,
        "pass": pos_pass and neg_pass and total_mismatch == 0,
    }
    log(f"[5] label: pos={total_pos} (기대={EXPECTED_POS}, pass={pos_pass}), neg={total_neg} (기대={EXPECTED_NEG}, pass={neg_pass}), mismatch={total_mismatch}")
    return result


# ============================================================
# 검증 6: local_z 검증
# ============================================================

def check_6_local_z(
    patient_npz_map: Dict[str, List[Path]],
    patient_stats: Dict[str, dict],
) -> dict:
    z_source_fail = 0
    negative_count = 0
    ne_slice_index_count = 0
    total_count = 0

    total = sum(len(v) for v in patient_npz_map.values())
    checked = 0

    for pid, files in patient_npz_map.items():
        pid_neg_z = 0
        pid_ne_slice = 0

        for fpath in files:
            d = np.load(fpath, allow_pickle=True)
            z_source = str(d["z_source"])
            local_z = int(d["local_z"])
            slice_index = int(d["slice_index"])
            total_count += 1

            # z_source 확인
            if z_source != "local_z":
                z_source_fail += 1

            # local_z >= 0 확인
            if local_z < 0:
                negative_count += 1
                pid_neg_z += 1

            # local_z != slice_index 확인
            if local_z != slice_index:
                ne_slice_index_count += 1
                pid_ne_slice += 1

            d.close()
            checked += 1
            if checked % 5000 == 0:
                log(f"  local_z check 진행: {checked}/{total}")

        patient_stats.setdefault(pid, {})
        patient_stats[pid]["local_z_negative_count"] = pid_neg_z
        patient_stats[pid]["local_z_ne_slice_index_count"] = pid_ne_slice

    ne_ratio = ne_slice_index_count / total_count if total_count > 0 else 0.0

    result = {
        "z_source_fail_count": z_source_fail,
        "negative_count": negative_count,
        "ne_slice_index_count": ne_slice_index_count,
        "ne_slice_index_ratio": round(ne_ratio, 6),
        "total_checked": total_count,
        "z_source_pass": z_source_fail == 0,
        "negative_pass": negative_count == 0,
    }
    log(f"[6] local_z: z_source_fail={z_source_fail}, negative={negative_count}, ne_slice_index={ne_slice_index_count} ({ne_ratio:.2%})")
    return result


# ============================================================
# 검증 7: 좌표 검증 (전수검사)
# ============================================================

def check_7_coord(
    patient_npz_map: Dict[str, List[Path]],
    patient_stats: Dict[str, dict],
) -> dict:
    total_fail = 0

    total = sum(len(v) for v in patient_npz_map.values())
    checked = 0

    for pid, files in patient_npz_map.items():
        pid_coord_err = 0

        for fpath in files:
            d = np.load(fpath, allow_pickle=True)
            crop_coords = d["crop_coords"]  # [y0, x0, y1, x1]
            orig_bbox = d["orig_bbox"]

            fail = False

            # crop_coords 검증
            if len(crop_coords) >= 4:
                y0, x0, y1, x1 = int(crop_coords[0]), int(crop_coords[1]), int(crop_coords[2]), int(crop_coords[3])
                if not (y0 >= 0 and x0 >= 0 and y1 >= 0 and x1 >= 0):
                    fail = True
                if not (y0 < y1):
                    fail = True
                if not (x0 < x1):
                    fail = True
                if (y1 - y0) != 96:
                    fail = True
                if (x1 - x0) != 96:
                    fail = True
            else:
                fail = True

            # orig_bbox 검증 (y0 < y1, x0 < x1, 모두 >= 0)
            if len(orig_bbox) >= 4:
                oy0, ox0, oy1, ox1 = int(orig_bbox[0]), int(orig_bbox[1]), int(orig_bbox[2]), int(orig_bbox[3])
                if not (oy0 >= 0 and ox0 >= 0 and oy1 >= 0 and ox1 >= 0):
                    fail = True
                if not (oy0 < oy1):
                    fail = True
                if not (ox0 < ox1):
                    fail = True
            else:
                fail = True

            if fail:
                total_fail += 1
                pid_coord_err += 1

            d.close()
            checked += 1
            if checked % 5000 == 0:
                log(f"  coord check 진행: {checked}/{total}")

        patient_stats.setdefault(pid, {})
        patient_stats[pid]["coord_error_count"] = pid_coord_err

    result = {
        "fail_count": total_fail,
        "pass": total_fail == 0,
    }
    log(f"[7] 좌표 검증: fail={total_fail}")
    return result


# ============================================================
# 검증 8: 환자별 분포 검증
# ============================================================

def check_8_patient_dist(
    patient_npz_map: Dict[str, List[Path]],
    patient_stats: Dict[str, dict],
) -> dict:
    npz_counts = [len(v) for v in patient_npz_map.values()]

    # NPP_PATIENTS 확인
    npp_check = {}
    for npp_pid in NPP_PATIENTS:
        ps = patient_stats.get(npp_pid, {})
        pos = ps.get("pos_count", -1)
        neg = ps.get("neg_count", -1)
        expected_pass = (pos == 0 and neg == 100)
        npp_check[npp_pid] = {
            "pos": pos,
            "neg": neg,
            "pass": expected_pass,
        }
        log(f"  NPP {npp_pid}: pos={pos}, neg={neg}, pass={expected_pass}")

    result = {
        "npz_count_min": int(min(npz_counts)) if npz_counts else 0,
        "npz_count_median": float(np.median(npz_counts)) if npz_counts else 0.0,
        "npz_count_mean": float(np.mean(npz_counts)) if npz_counts else 0.0,
        "npz_count_max": int(max(npz_counts)) if npz_counts else 0,
        "npp_check": npp_check,
    }
    log(f"[8] 환자별 npz: min={result['npz_count_min']}, median={result['npz_count_median']:.1f}, mean={result['npz_count_mean']:.1f}, max={result['npz_count_max']}")
    return result


# ============================================================
# 검증 9: 그룹별 분포 검증
# ============================================================

def check_9_group_dist(
    patient_npz_map: Dict[str, List[Path]],
    patient_stats: Dict[str, dict],
) -> dict:
    group_stats: Dict[str, dict] = {}

    for pid, files in patient_npz_map.items():
        group = get_group(pid)
        ps = patient_stats.get(pid, {})
        if group not in group_stats:
            group_stats[group] = {
                "patient_count": 0,
                "crop_count": 0,
                "pos_count": 0,
                "neg_count": 0,
            }
        group_stats[group]["patient_count"] += 1
        group_stats[group]["crop_count"] += len(files)
        group_stats[group]["pos_count"] += ps.get("pos_count", 0)
        group_stats[group]["neg_count"] += ps.get("neg_count", 0)

    for grp, gs in group_stats.items():
        log(f"[9] {grp}: 환자={gs['patient_count']}, crop={gs['crop_count']}, pos={gs['pos_count']}, neg={gs['neg_count']}")

    return group_stats


# ============================================================
# 검증 10: 샘플 로드 검증
# ============================================================

def check_10_sample_load(patient_npz_map: Dict[str, List[Path]]) -> dict:
    """positive 5개, hard_negative 5개, npp 환자 5개 샘플 로드 및 출력."""
    pos_samples: List[dict] = []
    neg_samples: List[dict] = []
    npp_samples: List[dict] = []

    all_paths: List[Tuple[str, Path]] = []
    for pid, files in patient_npz_map.items():
        for fpath in files:
            all_paths.append((pid, fpath))

    random.seed(42)
    random.shuffle(all_paths)

    for pid, fpath in all_paths:
        if len(pos_samples) >= 5 and len(neg_samples) >= 5 and len(npp_samples) >= 5:
            break

        d = np.load(fpath, allow_pickle=True)
        sampling_label = str(d["sampling_label"])
        local_z = int(d["local_z"])
        z_source = str(d["z_source"])
        crop_shape = tuple(d["crop"].shape)
        keys_sorted = sorted(d.files)
        label = str(d["label"])
        d.close()

        record = {
            "patient_id": pid,
            "file": str(fpath.name),
            "crop_shape": list(crop_shape),
            "keys": keys_sorted,
            "label": label,
            "sampling_label": sampling_label,
            "local_z": local_z,
            "z_source": z_source,
        }

        if pid in NPP_PATIENTS and len(npp_samples) < 5:
            npp_samples.append(record)
        elif sampling_label == "positive" and len(pos_samples) < 5:
            pos_samples.append(record)
        elif sampling_label == "hard_negative" and len(neg_samples) < 5:
            neg_samples.append(record)

    log(f"[10] 샘플 로드: positive={len(pos_samples)}, hard_negative={len(neg_samples)}, npp={len(npp_samples)}")

    print("\n[샘플 로드 결과]")
    print("=== positive samples ===")
    for s in pos_samples:
        print(f"  patient={s['patient_id']}, file={s['file']}, shape={s['crop_shape']}, label={s['label']}, local_z={s['local_z']}, z_source={s['z_source']}")
        print(f"  keys={s['keys']}")

    print("=== hard_negative samples ===")
    for s in neg_samples:
        print(f"  patient={s['patient_id']}, file={s['file']}, shape={s['crop_shape']}, label={s['label']}, local_z={s['local_z']}, z_source={s['z_source']}")
        print(f"  keys={s['keys']}")

    print("=== npp patient samples ===")
    for s in npp_samples:
        print(f"  patient={s['patient_id']}, file={s['file']}, shape={s['crop_shape']}, label={s['label']}, local_z={s['local_z']}, z_source={s['z_source']}")
        print(f"  keys={s['keys']}")

    return {
        "pos_samples": pos_samples,
        "neg_samples": neg_samples,
        "npp_samples": npp_samples,
    }


# ============================================================
# 출력 파일 생성
# ============================================================

def write_csv(
    patient_npz_map: Dict[str, List[Path]],
    patient_stats: Dict[str, dict],
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id", "group", "npz_count", "pos_count", "neg_count",
        "no_positive_patient", "key_missing_count", "shape_error_count",
        "nan_count", "coord_error_count", "label_mismatch_count",
        "local_z_negative_count", "local_z_ne_slice_index_count",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pid in sorted(patient_npz_map.keys()):
            ps = patient_stats.get(pid, {})
            npp = pid in NPP_PATIENTS
            writer.writerow({
                "patient_id": pid,
                "group": get_group(pid),
                "npz_count": len(patient_npz_map[pid]),
                "pos_count": ps.get("pos_count", 0),
                "neg_count": ps.get("neg_count", 0),
                "no_positive_patient": npp,
                "key_missing_count": ps.get("key_missing_count", 0),
                "shape_error_count": ps.get("shape_error_count", 0),
                "nan_count": ps.get("nan_count", 0),
                "coord_error_count": ps.get("coord_error_count", 0),
                "label_mismatch_count": ps.get("label_mismatch_count", 0),
                "local_z_negative_count": ps.get("local_z_negative_count", 0),
                "local_z_ne_slice_index_count": ps.get("local_z_ne_slice_index_count", 0),
            })
    log(f"CSV 저장: {OUTPUT_CSV}")


def write_json(summary: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"JSON 저장: {OUTPUT_JSON}")


def write_md(summary: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def pf(flag: bool) -> str:
        return "PASS" if flag else "FAIL"

    c1 = summary.get("check_1", {})
    c2 = summary.get("check_2", {})
    c3 = summary.get("check_3", {})
    c4 = summary.get("check_4", {})
    c5 = summary.get("check_5", {})
    c6 = summary.get("check_6", {})
    c7 = summary.get("check_7", {})
    c8 = summary.get("check_8", {})
    c9 = summary.get("check_9", {})
    overall_pass = summary.get("overall_pass", False)
    val_time = summary.get("validation_time_seconds", 0)

    lines = [
        "# crops_s4_full 검증 보고서",
        "",
        f"- 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 검증 소요 시간: {val_time:.1f}초",
        f"- 전체 판정: **{'PASS' if overall_pass else 'FAIL'}**",
        "",
        "## 항목별 결과",
        "",
        "| # | 항목 | 결과 | 상세 |",
        "|---|------|------|------|",
        f"| 1 | 파일 개수 | {pf(c1.get('total_npz_pass', False) and c1.get('patient_count_pass', False) and c1.get('manifest_match', False))} | npz={c1.get('total_npz', '?')}, 환자={c1.get('patient_count', '?')}, manifest={c1.get('manifest_s4_count', '?')} |",
        f"| 2 | stage split | {pf(c2.get('pass', False))} | stage2_holdout={len(c2.get('stage2_holdout_patients', []))} |",
        f"| 3 | npz key | {pf(c3.get('pass', False))} | 누락 npz={c3.get('total_missing', '?')}, 누락 환자={c3.get('patient_count_with_missing', '?')} |",
        f"| 4a | crop shape | {pf(c4.get('shape_check', {}).get('pass', False))} | fail={c4.get('shape_check', {}).get('fail_count', '?')} |",
        f"| 4b | crop dtype | {pf(c4.get('dtype_check', {}).get('pass', False))} | fail={c4.get('dtype_check', {}).get('fail_count', '?')} |",
        f"| 4c | NaN/Inf | {pf(c4.get('nan_check', {}).get('pass', False))} | nan={c4.get('nan_check', {}).get('nan_count', '?')}, inf={c4.get('nan_check', {}).get('inf_count', '?')} (샘플 {c4.get('nan_check', {}).get('sample_size', '?')}개) |",
        f"| 5 | label | {pf(c5.get('pass', False))} | pos={c5.get('pos_count', '?')} (기대={EXPECTED_POS}), neg={c5.get('neg_count', '?')} (기대={EXPECTED_NEG}), mismatch={c5.get('mismatch_count', '?')} |",
        f"| 6 | local_z | {pf(c6.get('z_source_pass', False) and c6.get('negative_pass', False))} | z_source_fail={c6.get('z_source_fail_count', '?')}, negative={c6.get('negative_count', '?')}, ne_slice_index={c6.get('ne_slice_index_count', '?')} ({c6.get('ne_slice_index_ratio', 0):.2%}) |",
        f"| 7 | 좌표 | {pf(c7.get('pass', False))} | fail={c7.get('fail_count', '?')} |",
        "",
        "## 환자별 분포 (8)",
        "",
        f"- npz 수: min={c8.get('npz_count_min', '?')}, median={c8.get('npz_count_median', '?'):.1f}, mean={c8.get('npz_count_mean', '?'):.1f}, max={c8.get('npz_count_max', '?')}",
        "",
        "### NPP 환자 확인",
        "",
        "| patient_id | pos | neg | pass |",
        "|------------|-----|-----|------|",
    ]

    npp_check = c8.get("npp_check", {})
    for npp_pid in sorted(NPP_PATIENTS):
        nc = npp_check.get(npp_pid, {})
        lines.append(f"| {npp_pid} | {nc.get('pos', '?')} | {nc.get('neg', '?')} | {pf(nc.get('pass', False))} |")

    lines += [
        "",
        "## 그룹별 분포 (9)",
        "",
        "| group | 환자 수 | crop 수 | pos | neg |",
        "|-------|---------|---------|-----|-----|",
    ]
    for grp, gs in sorted(c9.items()):
        lines.append(f"| {grp} | {gs.get('patient_count', '?')} | {gs.get('crop_count', '?')} | {gs.get('pos_count', '?')} | {gs.get('neg_count', '?')} |")

    lines += [""]

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"MD 저장: {OUTPUT_MD}")


# ============================================================
# 메인
# ============================================================

def main() -> None:
    start_time = time.time()
    log("=== crops_s4_full 검증 시작 ===")

    # 경로 확인
    if not CROPS_FULL_DIR.exists():
        print(f"[ABORT] crops_s4_full 경로가 존재하지 않습니다: {CROPS_FULL_DIR}", file=sys.stderr)
        sys.exit(1)
    if not MANIFEST_PATH.exists():
        print(f"[ABORT] manifest 파일이 없습니다: {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)
    if not SPLIT_CSV_PATH.exists():
        print(f"[ABORT] split CSV가 없습니다: {SPLIT_CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    # CSV 로드
    log("manifest 로드 중...")
    manifest_df = pd.read_csv(MANIFEST_PATH)
    log(f"manifest 행 수: {len(manifest_df)}")

    log("split CSV 로드 중...")
    split_df = pd.read_csv(SPLIT_CSV_PATH)
    log(f"split CSV 행 수: {len(split_df)}")

    # npz 파일 목록 수집
    log("npz 파일 목록 수집 중...")
    patient_npz_map = collect_all_npz()
    log(f"환자 수: {len(patient_npz_map)}, 총 npz: {sum(len(v) for v in patient_npz_map.values())}")

    # 환자별 집계 dict (각 check에서 side-effect로 채움)
    patient_stats: Dict[str, dict] = {}

    # 검증 실행 (3, 4, 5, 6, 7은 전수검사 → 한 번의 루프로 통합하지 않고 개별 실행)
    log("\n--- 검증 1: 파일 개수 ---")
    c1 = check_1_file_count(patient_npz_map, manifest_df)

    log("\n--- 검증 2: stage split ---")
    c2 = check_2_stage_split(patient_npz_map, split_df)

    log("\n--- 검증 3: npz key (전수검사) ---")
    c3, patient_key_missing = check_3_key_check(patient_npz_map)
    # patient_key_missing을 patient_stats에 반영
    for pid, cnt in patient_key_missing.items():
        patient_stats.setdefault(pid, {})
        patient_stats[pid]["key_missing_count"] = cnt

    log("\n--- 검증 4: crop shape / dtype / NaN/Inf ---")
    c4 = check_4_shape_dtype_nan(patient_npz_map, patient_stats)

    log("\n--- 검증 5: label ---")
    c5 = check_5_label(patient_npz_map, patient_stats)

    log("\n--- 검증 6: local_z ---")
    c6 = check_6_local_z(patient_npz_map, patient_stats)

    log("\n--- 검증 7: 좌표 (전수검사) ---")
    c7 = check_7_coord(patient_npz_map, patient_stats)

    log("\n--- 검증 8: 환자별 분포 ---")
    c8 = check_8_patient_dist(patient_npz_map, patient_stats)

    log("\n--- 검증 9: 그룹별 분포 ---")
    c9 = check_9_group_dist(patient_npz_map, patient_stats)

    log("\n--- 검증 10: 샘플 로드 ---")
    c10 = check_10_sample_load(patient_npz_map)

    # overall_pass 판정
    overall_pass = all([
        c1.get("total_npz_pass", False),
        c1.get("patient_count_pass", False),
        c1.get("manifest_match", False),
        c2.get("pass", False),
        c3.get("pass", False),
        c4.get("shape_check", {}).get("pass", False),
        c4.get("dtype_check", {}).get("pass", False),
        c4.get("nan_check", {}).get("pass", False),
        c5.get("pass", False),
        c6.get("z_source_pass", False),
        c6.get("negative_pass", False),
        c7.get("pass", False),
    ])

    elapsed = time.time() - start_time

    summary = {
        "total_npz": c1.get("total_npz"),
        "patient_count": c1.get("patient_count"),
        "manifest_match": c1.get("manifest_match"),
        "stage2_holdout_patients": c2.get("stage2_holdout_patients", []),
        "key_check": {
            "total_missing": c3.get("total_missing"),
            "patient_count_with_missing": c3.get("patient_count_with_missing"),
        },
        "shape_check": c4.get("shape_check"),
        "dtype_check": c4.get("dtype_check"),
        "nan_check": c4.get("nan_check"),
        "crop_value_summary": c4.get("crop_value_summary"),
        "label_check": {
            "pos_count": c5.get("pos_count"),
            "neg_count": c5.get("neg_count"),
            "mismatch_count": c5.get("mismatch_count"),
            "expected_pos": EXPECTED_POS,
            "expected_neg": EXPECTED_NEG,
        },
        "local_z_check": {
            "z_source_fail_count": c6.get("z_source_fail_count"),
            "negative_count": c6.get("negative_count"),
            "ne_slice_index_count": c6.get("ne_slice_index_count"),
            "ne_slice_index_ratio": c6.get("ne_slice_index_ratio"),
        },
        "coord_check": {
            "fail_count": c7.get("fail_count"),
        },
        "npp_check": c8.get("npp_check"),
        "group_stats": c9,
        "overall_pass": overall_pass,
        "validation_time_seconds": round(elapsed, 2),
        # 내부 참조용 (JSON에만)
        "check_1": c1,
        "check_2": c2,
        "check_3": c3,
        "check_4": c4,
        "check_5": c5,
        "check_6": c6,
        "check_7": c7,
        "check_8": c8,
        "check_9": c9,
    }

    # 출력 파일 생성
    log("\n--- 결과 파일 저장 ---")
    write_csv(patient_npz_map, patient_stats)
    write_json(summary)
    write_md(summary)

    # 최종 판정 출력
    log(f"\n=== 최종 판정: {'PASS' if overall_pass else 'FAIL'} (소요 시간: {elapsed:.1f}초) ===")
    if not overall_pass:
        log("FAIL 항목을 확인하세요:")
        if not c1.get("total_npz_pass"):
            log(f"  - 파일 개수 불일치: {c1.get('total_npz')} (기대: {EXPECTED_TOTAL_CROPS})")
        if not c1.get("patient_count_pass"):
            log(f"  - 환자 수 불일치: {c1.get('patient_count')} (기대: {EXPECTED_PATIENT_COUNT})")
        if not c1.get("manifest_match"):
            log(f"  - manifest 불일치: {c1.get('manifest_s4_count')} (기대: {EXPECTED_TOTAL_CROPS})")
        if not c2.get("pass"):
            log(f"  - stage2_holdout 환자 포함: {c2.get('stage2_holdout_patients')}")
        if not c3.get("pass"):
            log(f"  - key 누락 npz: {c3.get('total_missing')}")
        if not c4.get("shape_check", {}).get("pass"):
            log(f"  - shape 오류: {c4.get('shape_check', {}).get('fail_count')}")
        if not c4.get("dtype_check", {}).get("pass"):
            log(f"  - dtype 오류: {c4.get('dtype_check', {}).get('fail_count')}")
        if not c4.get("nan_check", {}).get("pass"):
            log(f"  - NaN/Inf 있음: nan={c4.get('nan_check', {}).get('nan_count')}, inf={c4.get('nan_check', {}).get('inf_count')}")
        if not c5.get("pass"):
            log(f"  - label 오류: pos={c5.get('pos_count')}, neg={c5.get('neg_count')}, mismatch={c5.get('mismatch_count')}")
        if not c6.get("z_source_pass"):
            log(f"  - z_source 오류: {c6.get('z_source_fail_count')}")
        if not c6.get("negative_pass"):
            log(f"  - local_z 음수: {c6.get('negative_count')}")
        if not c7.get("pass"):
            log(f"  - 좌표 오류: {c7.get('fail_count')}")


if __name__ == "__main__":
    main()
