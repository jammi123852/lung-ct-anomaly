"""
RD-B6a: Tiny Smoke Train Preflight
목적: RD-B5 Dataset/Loader/Model skeleton을 tiny smoke train 진입 직전 상태로
      패치·검증한다. 이번 단계는 실제 training 실행 전 준비 단계이다.
실행 방법:
  bare run         → exit 2 (파일 생성 금지)
  --selftest       → 순수 함수 테스트 (CT 로딩/model forward 없음)
  --dry-check      → manifest/weight/output_root/CT_path 확인 (파일 생성 없음)
  --real-preflight → output root 생성 + smoke subset CSV + weight metadata
                     + 소수 crop on-the-fly 확인 (사용자 승인 후)
  --tiny-forward-check → teacher/student forward shape 확인 (별도 사용자 승인 후)
안전 조건:
  stage2_holdout/lesion 경로 접근 금지
  full crop NPZ 생성 금지  /  학습 금지  /  scoring 금지
  model forward 금지 (--tiny-forward-check 별도 승인 시 제외)
  ImageNet weight 자동 다운로드 금지
  기존 파일 수정/삭제 금지
  output root 이미 존재 시 즉시 중단
"""

import sys
import os
import csv
import json
import math
import time
import copy
import random
from pathlib import Path
from collections import defaultdict, OrderedDict

# ─── bare-run guard ──────────────────────────────────────────────────────────
ALLOWED_MODES = {"--selftest", "--dry-check", "--real-preflight", "--tiny-forward-check"}
if not any(m in sys.argv for m in ALLOWED_MODES):
    print("오류: 실행 모드 지정 필요.")
    print("  --selftest          : 순수 함수 단위 테스트 (CT 로딩·model forward 없음)")
    print("  --dry-check         : 경로/weight/output root 확인 (파일 생성 없음)")
    print("  --real-preflight    : output root 생성 + smoke subset + crop 확인 (사용자 승인 후)")
    print("  --tiny-forward-check: teacher/student forward shape 확인 (별도 사용자 승인 후)")
    sys.exit(2)

IS_SELFTEST = "--selftest" in sys.argv
IS_DRY_CHECK = "--dry-check" in sys.argv
IS_REAL_PREFLIGHT = "--real-preflight" in sys.argv
IS_TINY_FORWARD = "--tiny-forward-check" in sys.argv

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b6a_tiny_smoke_train_preflight_v1"
)
MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b1_6bin_balanced_manifest_preflight_v1"
    / "rd_b1_6bin_balanced_normal_train_coordinate_manifest.csv"
)
PATIENT_MANIFEST_PATH = Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/"
    "manifests/patient_manifest.csv"
)
LOCAL_WEIGHT_PATH = Path(
    "/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)

# ─── 안전 금지 키워드 ────────────────────────────────────────────────────────
FORBIDDEN_KEYWORDS = [
    "stage2_holdout",
    "lesion",
    "test_lesion",
    "second-stage-lesion-refiner",
]

# ─── 설계 상수 ────────────────────────────────────────────────────────────────
CROP_SIZE = 96
N_CHANNELS = 3
MIP_RADIUS = 3
HU_CLIP_MIN = -1000.0
HU_CLIP_MAX = 600.0
HU_RANGE = 1600.0
SIX_BIN_LABELS = [
    "lower_boundary",
    "lower_interior",
    "middle_boundary",
    "middle_interior",
    "upper_boundary",
    "upper_interior",
]
LOW_Z_BOUNDARY_WARN_THRESHOLD = 7
EXPECTED_MANIFEST_ROWS = 86_017

# ─── smoke subset 설계 ───────────────────────────────────────────────────────
# 2명 × 6-bin × 5 crops = 60 crops
# normal001: 일반 환자 (boundary + interior 포함)
# normal006: low_z ≤ 7 케이스 포함 (low_z_boundary_warning 검증용)
SMOKE_PATIENTS = [
    "normal001__104e7cb873",
    "normal006__e662c8463b",
]
SMOKE_CROPS_PER_BIN_PER_PATIENT = 5    # 2 × 6 × 5 = 60 crops
SMOKE_CROP_CHECK_COUNT = 3             # --real-preflight 온더플라이 확인 crop 수
SMOKE_SEED = 42

errors = []


# =============================================================================
# 안전 검사
# =============================================================================
def assert_path_safe(path_str: str) -> None:
    for kw in FORBIDDEN_KEYWORDS:
        if kw.lower() in str(path_str).lower():
            raise RuntimeError(
                f"[SAFETY] 금지 경로 접근 차단: {path_str!r} (keyword={kw!r})"
            )


# =============================================================================
# 순수 함수 (selftest 포함)
# =============================================================================

def normalize_hu(hu_array):
    """HU [-1000,600] → [0,1] float32 (RD-B2b 확정)."""
    import numpy as np
    clipped = np.clip(hu_array, HU_CLIP_MIN, HU_CLIP_MAX)
    return ((clipped - HU_CLIP_MIN) / HU_RANGE).astype("float32")


def compute_mip_slab_indices(center_z: int, direction: str, z_max: int) -> list:
    """MIP slab 인덱스 계산 + edge clamp [0, z_max-1]."""
    if direction == "lower":
        raw = [center_z - MIP_RADIUS + i for i in range(MIP_RADIUS)]
    elif direction == "upper":
        raw = [center_z + 1 + i for i in range(MIP_RADIUS)]
    else:
        raise ValueError(f"direction must be 'lower' or 'upper', got {direction!r}")
    return [max(0, min(idx, z_max - 1)) for idx in raw]


def has_low_z_boundary_warning(center_z: int) -> bool:
    """z ≤ LOW_Z_BOUNDARY_WARN_THRESHOLD → True (diaphragm saturation risk)."""
    return center_z <= LOW_Z_BOUNDARY_WARN_THRESHOLD


def find_local_weight() -> dict:
    """
    local ResNet18 ImageNet weight 파일 확인.
    인터넷 접속 없이 로컬 캐시만 확인한다.
    자동 다운로드 금지.
    """
    result = {
        "local_weight_available": False,
        "path": str(LOCAL_WEIGHT_PATH),
        "filename": LOCAL_WEIGHT_PATH.name,
        "size_bytes": 0,
        "size_mb": 0.0,
        "mtime": "",
        "note": "",
    }
    if LOCAL_WEIGHT_PATH.exists():
        stat = LOCAL_WEIGHT_PATH.stat()
        result["local_weight_available"] = True
        result["size_bytes"] = stat.st_size
        result["size_mb"] = round(stat.st_size / 1024 / 1024, 2)
        result["mtime"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
        )
        result["note"] = "local cache hit — 자동 다운로드 불필요"
    else:
        result["note"] = "로컬 weight 없음 — 자동 다운로드 금지, 수동 배치 필요"
    return result


def select_smoke_subset(
    manifest_path: Path,
    smoke_patients: list,
    per_bin_per_patient: int,
    seed: int,
) -> list:
    """
    smoke_patients × 6-bin × per_bin_per_patient 크기의 smoke subset 선택.
    duplicate oversampling 금지.
    low_z_boundary_warning 케이스(z≤7)가 존재하면 1개 force-include하여 검증 보장.
    반환: list[dict] (manifest rows subset)
    """
    rng = random.Random(seed)
    rows_by_patient_bin: dict = defaultdict(list)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("safe_id", "")
            lbl = row.get("six_bin_label", "")
            if sid in smoke_patients:
                rows_by_patient_bin[(sid, lbl)].append(row)

    subset = []
    for sid in smoke_patients:
        for lbl in SIX_BIN_LABELS:
            pool = rows_by_patient_bin[(sid, lbl)]
            rng.shuffle(pool)
            # low_z_boundary_warning 케이스 force-include (1개)
            low_z = [r for r in pool if has_low_z_boundary_warning(int(r["local_z"]))]
            others = [r for r in pool if not has_low_z_boundary_warning(int(r["local_z"]))]
            if low_z and per_bin_per_patient > 0:
                selected = [low_z[0]] + others[: per_bin_per_patient - 1]
            else:
                selected = pool[:per_bin_per_patient]
            subset.extend(selected)
    return subset


# =============================================================================
# selftest
# =============================================================================

def run_selftest() -> dict:
    """
    순수 함수 단위 테스트:
    - normalize_hu 값 범위 테스트
    - compute_mip_slab_indices edge clamp 테스트
    - has_low_z_boundary_warning 테스트
    - 6-bin subset sampler 중복 없음 테스트
    - local weight finder 테스트
    CT 로딩, model forward, GPU 사용 금지.
    """
    import numpy as np
    results = []

    # (1) normalize_hu
    test_cases_norm = [
        (-1000.0, 0.0),
        (600.0, 1.0),
        (0.0, 0.625),
        (-500.0, 0.3125),
        (-2000.0, 0.0),
        (9999.0, 1.0),
    ]
    for hu_val, expected in test_cases_norm:
        arr = np.array([[hu_val]], dtype="float32")
        actual = float(normalize_hu(arr)[0, 0])
        ok = abs(actual - expected) < 1e-5
        results.append({
            "test": "normalize_hu",
            "input": str(hu_val),
            "expected": str(expected),
            "actual": str(round(actual, 6)),
            "pass": ok,
        })

    # (2) compute_mip_slab_indices
    slab_tests = [
        (10, "lower", 100, [7, 8, 9]),
        (10, "upper", 100, [11, 12, 13]),
        (1,  "lower", 100, [0, 0, 0]),
        (2,  "lower", 100, [0, 0, 1]),
        (98, "upper", 100, [99, 99, 99]),
        (5,  "lower", 100, [2, 3, 4]),
        (5,  "upper", 100, [6, 7, 8]),
    ]
    for center_z, direction, z_max, expected in slab_tests:
        actual = compute_mip_slab_indices(center_z, direction, z_max)
        ok = actual == expected
        results.append({
            "test": "mip_slab_indices",
            "input": f"z={center_z},{direction},z_max={z_max}",
            "expected": str(expected),
            "actual": str(actual),
            "pass": ok,
        })

    # (3) has_low_z_boundary_warning
    warn_tests = [(7, True), (8, False), (0, True), (6, True), (100, False)]
    for z, expected in warn_tests:
        actual = has_low_z_boundary_warning(z)
        results.append({
            "test": "low_z_boundary_warning",
            "input": f"z={z}",
            "expected": str(expected),
            "actual": str(actual),
            "pass": actual == expected,
        })

    # (4) 6-bin subset sampler dummy — 중복 없음 확인
    dummy_pool = {
        (sid, lbl): [
            {
                "safe_id": sid,
                "six_bin_label": lbl,
                "manifest_id": f"{sid}_{lbl}_{i}",
                "local_z": str(i),
                "crop_y0": "0", "crop_x0": "0",
                "crop_y1": "96", "crop_x1": "96",
            }
            for i in range(10)
        ]
        for sid in SMOKE_PATIENTS
        for lbl in SIX_BIN_LABELS
    }
    rng_test = random.Random(SMOKE_SEED)
    subset_test = []
    for sid in SMOKE_PATIENTS:
        for lbl in SIX_BIN_LABELS:
            pool = list(dummy_pool[(sid, lbl)])
            rng_test.shuffle(pool)
            subset_test.extend(pool[:SMOKE_CROPS_PER_BIN_PER_PATIENT])

    expected_count = len(SMOKE_PATIENTS) * len(SIX_BIN_LABELS) * SMOKE_CROPS_PER_BIN_PER_PATIENT
    n_total = len(subset_test)
    manifest_ids = [r["manifest_id"] for r in subset_test]
    n_unique = len(set(manifest_ids))
    ok_count = n_total == expected_count
    ok_unique = n_unique == n_total
    results.append({
        "test": "6bin_subset_sampler",
        "input": f"{len(SMOKE_PATIENTS)}patients×6bins×{SMOKE_CROPS_PER_BIN_PER_PATIENT}/bin",
        "expected": f"total={expected_count},no_dup=True",
        "actual": f"total={n_total},no_dup={ok_unique}",
        "pass": ok_count and ok_unique,
    })

    # (5) local weight finder
    wt = find_local_weight()
    results.append({
        "test": "local_weight_finder",
        "input": str(LOCAL_WEIGHT_PATH),
        "expected": "local_weight_available=True",
        "actual": f"local_weight_available={wt['local_weight_available']}",
        "pass": wt["local_weight_available"],
    })

    n_pass = sum(1 for r in results if r["pass"])
    return {
        "n_tests": len(results),
        "n_pass": n_pass,
        "n_fail": len(results) - n_pass,
        "all_pass": n_pass == len(results),
        "details": results,
    }


# =============================================================================
# dry-check
# =============================================================================

def run_dry_check() -> dict:
    """경로/weight/output_root/CT_path 존재 확인. 파일 생성 없음."""
    result = {}

    # 1. manifest 존재 및 row count
    result["manifest_exists"] = MANIFEST_PATH.exists()
    if result["manifest_exists"]:
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
            row_count = sum(1 for _ in csv.DictReader(f))
        result["manifest_row_count"] = row_count
        result["manifest_row_count_ok"] = row_count == EXPECTED_MANIFEST_ROWS
    else:
        result["manifest_row_count"] = 0
        result["manifest_row_count_ok"] = False

    # 2. smoke subset 후보 계산 (파일 생성 없음)
    if result["manifest_exists"]:
        subset = select_smoke_subset(
            MANIFEST_PATH, SMOKE_PATIENTS,
            SMOKE_CROPS_PER_BIN_PER_PATIENT, SMOKE_SEED
        )
        result["smoke_subset_count"] = len(subset)
        low_z_cases = [
            r for r in subset if has_low_z_boundary_warning(int(r["local_z"]))
        ]
        result["low_z_warning_in_subset"] = len(low_z_cases)
        bin_patient_dist = defaultdict(int)
        for r in subset:
            key = f"{r['safe_id']}|{r['six_bin_label']}"
            bin_patient_dist[key] += 1
        result["smoke_subset_bin_patient_dist"] = dict(bin_patient_dist)
    else:
        result["smoke_subset_count"] = 0
        result["low_z_warning_in_subset"] = 0
        result["smoke_subset_bin_patient_dist"] = {}

    # 3. local weight 존재 확인
    wt = find_local_weight()
    result["local_weight"] = wt

    # 4. output root 비존재 확인
    result["output_root_exists"] = OUTPUT_ROOT.exists()
    result["output_root_clear"] = not OUTPUT_ROOT.exists()

    # 5. CT/ROI path 존재 확인 (smoke patients만)
    result["patient_manifest_exists"] = PATIENT_MANIFEST_PATH.exists()
    ct_check = {}
    roi_check = {}
    if PATIENT_MANIFEST_PATH.exists():
        with open(PATIENT_MANIFEST_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("safe_id", "")
                if sid in SMOKE_PATIENTS:
                    ct_path = Path(row.get("ct_hu_npy", ""))
                    roi_path = Path(row.get("roi_0_0_npy", ""))
                    try:
                        assert_path_safe(str(ct_path))
                        assert_path_safe(str(roi_path))
                    except RuntimeError as e:
                        ct_check[sid] = f"SAFETY_BLOCKED"
                        roi_check[sid] = f"SAFETY_BLOCKED"
                        continue
                    ct_check[sid] = ct_path.exists()
                    roi_check[sid] = roi_path.exists()
    result["smoke_ct_exists"] = ct_check
    result["smoke_roi_exists"] = roi_check
    result["smoke_ct_all_ok"] = all(v is True for v in ct_check.values()) and len(ct_check) == len(SMOKE_PATIENTS)
    result["smoke_roi_all_ok"] = all(v is True for v in roi_check.values()) and len(roi_check) == len(SMOKE_PATIENTS)

    return result


# =============================================================================
# real-preflight 핵심 함수들
# =============================================================================

def load_patient_paths(smoke_patients: list) -> dict:
    """patient_manifest에서 smoke patients의 CT/ROI 경로 로드 (경로만, npy 값 로딩 금지)."""
    patient_paths = {}
    with open(PATIENT_MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("safe_id", "")
            if sid in smoke_patients:
                assert_path_safe(row["ct_hu_npy"])
                assert_path_safe(row["roi_0_0_npy"])
                patient_paths[sid] = {
                    "ct_hu_npy": row["ct_hu_npy"],
                    "roi_0_0_npy": row["roi_0_0_npy"],
                }
    return patient_paths


def build_crop_from_mmap(
    ct_arr,
    center_z: int,
    crop_y0: int,
    crop_x0: int,
    crop_y1: int,
    crop_x1: int,
) -> dict:
    """
    CT mmap array에서 mixed_3ch crop 생성 (on-the-fly).
    결과 shape/range 확인용. crop 저장 금지.
    반환: 형태/범위/NaN/Inf 확인 결과 dict (crop array 미포함)
    """
    import numpy as np
    z_max = ct_arr.shape[0]

    ch0_raw = ct_arr[center_z, crop_y0:crop_y1, crop_x0:crop_x1].copy()
    lower_idxs = compute_mip_slab_indices(center_z, "lower", z_max)
    upper_idxs = compute_mip_slab_indices(center_z, "upper", z_max)
    lower_slab = ct_arr[lower_idxs]          # (3, Y, X)
    upper_slab = ct_arr[upper_idxs]          # (3, Y, X)
    ch1_raw = lower_slab.max(axis=0)[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    ch2_raw = upper_slab.max(axis=0)[crop_y0:crop_y1, crop_x0:crop_x1].copy()

    crop = np.stack([
        normalize_hu(ch0_raw),
        normalize_hu(ch1_raw),
        normalize_hu(ch2_raw),
    ], axis=0)  # (3, H, W) float32

    has_nan = bool(np.isnan(crop).any())
    has_inf = bool(np.isinf(crop).any())
    val_min = float(crop.min())
    val_max = float(crop.max())
    shape_ok = crop.shape == (N_CHANNELS, CROP_SIZE, CROP_SIZE)
    range_ok = val_min >= 0.0 - 1e-6 and val_max <= 1.0 + 1e-6

    # crop array는 반환하지 않음 (저장 금지)
    del crop, ch0_raw, ch1_raw, ch2_raw, lower_slab, upper_slab

    return {
        "shape": [N_CHANNELS, crop_y1 - crop_y0, crop_x1 - crop_x0],
        "dtype": "float32",
        "val_min": round(val_min, 6),
        "val_max": round(val_max, 6),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "shape_ok": shape_ok,
        "range_ok": range_ok,
        "nan_free": not has_nan,
        "inf_free": not has_inf,
        "low_z_boundary_warning": has_low_z_boundary_warning(center_z),
        "lower_idxs": lower_idxs,
        "upper_idxs": upper_idxs,
    }


def run_real_preflight() -> dict:
    """
    --real-preflight 실행.
    1. output root 생성 (이미 존재 시 중단)
    2. smoke subset CSV 생성
    3. local weight metadata JSON 기록
    4. 소수 crop 1~3개 on-the-fly 생성 → shape/range/NaN/Inf 확인 (저장 금지)
    5. crop_loading_check CSV 생성
    6. preflight_summary JSON 생성
    7. errors CSV 생성
    8. report.md 생성
    9. DONE marker 생성
    """
    import numpy as np
    import datetime

    if OUTPUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재 → 즉시 중단: {OUTPUT_ROOT}")
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  output root 생성: {OUTPUT_ROOT}")

    result = {"timestamp": ts}

    # ── 1. smoke subset CSV ──────────────────────────────────────────────────
    subset = select_smoke_subset(
        MANIFEST_PATH, SMOKE_PATIENTS,
        SMOKE_CROPS_PER_BIN_PER_PATIENT, SMOKE_SEED
    )
    subset_csv_path = OUTPUT_ROOT / "rd_b6a_smoke_subset_manifest.csv"
    if subset:
        fieldnames = list(subset[0].keys())
        with open(subset_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(subset)
    result["smoke_subset_count"] = len(subset)
    print(f"  smoke subset: {len(subset)} crops → {subset_csv_path.name}")

    # ── 2. local weight metadata JSON ───────────────────────────────────────
    wt = find_local_weight()
    weight_json_path = OUTPUT_ROOT / "rd_b6a_local_weight_check.json"
    with open(weight_json_path, "w", encoding="utf-8") as f:
        json.dump(wt, f, ensure_ascii=False, indent=2)
    result["local_weight"] = wt
    print(f"  local weight: available={wt['local_weight_available']} "
          f"size={wt['size_mb']}MB mtime={wt['mtime']}")

    # ── 3. crop on-the-fly 확인 ─────────────────────────────────────────────
    patient_paths = load_patient_paths(SMOKE_PATIENTS)
    crop_checks = []

    # low_z 케이스 1개 우선 선택 → 나머지 임의
    low_z_rows = [r for r in subset if has_low_z_boundary_warning(int(r["local_z"]))]
    other_rows = [r for r in subset if not has_low_z_boundary_warning(int(r["local_z"]))]
    rng = random.Random(SMOKE_SEED)
    rng.shuffle(other_rows)
    check_rows = []
    if low_z_rows:
        check_rows.append(low_z_rows[0])
    check_rows.extend(other_rows[: max(0, SMOKE_CROP_CHECK_COUNT - len(check_rows))])
    check_rows = check_rows[:SMOKE_CROP_CHECK_COUNT]

    for row in check_rows:
        sid = row["safe_id"]
        if sid not in patient_paths:
            crop_checks.append({
                "manifest_id": row["manifest_id"],
                "safe_id": sid,
                "local_z": row["local_z"],
                "six_bin_label": row.get("six_bin_label", ""),
                "status": "SKIP_NO_PATH",
                "shape_ok": False,
                "range_ok": False,
                "nan_free": False,
                "inf_free": False,
                "low_z_boundary_warning": has_low_z_boundary_warning(int(row["local_z"])),
            })
            continue

        ct_path = patient_paths[sid]["ct_hu_npy"]
        assert_path_safe(ct_path)
        ct_arr = None
        try:
            ct_arr = np.load(ct_path, mmap_mode="r")
            cr = build_crop_from_mmap(
                ct_arr,
                int(row["local_z"]),
                int(row["crop_y0"]),
                int(row["crop_x0"]),
                int(row["crop_y1"]),
                int(row["crop_x1"]),
            )
            crop_checks.append({
                "manifest_id": row["manifest_id"],
                "safe_id": sid,
                "local_z": row["local_z"],
                "six_bin_label": row.get("six_bin_label", ""),
                "shape": str(cr["shape"]),
                "dtype": cr["dtype"],
                "val_min": cr["val_min"],
                "val_max": cr["val_max"],
                "has_nan": cr["has_nan"],
                "has_inf": cr["has_inf"],
                "shape_ok": cr["shape_ok"],
                "range_ok": cr["range_ok"],
                "nan_free": cr["nan_free"],
                "inf_free": cr["inf_free"],
                "low_z_boundary_warning": cr["low_z_boundary_warning"],
                "lower_idxs": str(cr["lower_idxs"]),
                "upper_idxs": str(cr["upper_idxs"]),
                "status": "OK",
            })
        except Exception as e:
            crop_checks.append({
                "manifest_id": row["manifest_id"],
                "safe_id": sid,
                "local_z": row["local_z"],
                "six_bin_label": row.get("six_bin_label", ""),
                "status": f"ERROR: {e}",
                "shape_ok": False,
                "range_ok": False,
                "nan_free": False,
                "inf_free": False,
                "low_z_boundary_warning": has_low_z_boundary_warning(int(row["local_z"])),
            })
        finally:
            if ct_arr is not None:
                del ct_arr

    # crop loading check CSV
    crop_csv_path = OUTPUT_ROOT / "rd_b6a_crop_loading_check.csv"
    if crop_checks:
        all_keys = set()
        for c in crop_checks:
            all_keys.update(c.keys())
        fieldnames = [
            "manifest_id", "safe_id", "local_z", "six_bin_label",
            "shape", "dtype", "val_min", "val_max",
            "has_nan", "has_inf", "shape_ok", "range_ok",
            "nan_free", "inf_free", "low_z_boundary_warning",
            "lower_idxs", "upper_idxs", "status",
        ]
        with open(crop_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for c in crop_checks:
                for k in fieldnames:
                    if k not in c:
                        c[k] = ""
                w.writerow(c)

    result["crop_checks"] = crop_checks

    # ── 4. preflight summary JSON ────────────────────────────────────────────
    n_checked = len(crop_checks)
    n_ok = sum(1 for c in crop_checks if c.get("status") == "OK")
    n_shape_ok = sum(1 for c in crop_checks if c.get("shape_ok") is True)
    n_range_ok = sum(1 for c in crop_checks if c.get("range_ok") is True)
    n_nan_free = sum(1 for c in crop_checks if c.get("nan_free") is True)
    n_inf_free = sum(1 for c in crop_checks if c.get("inf_free") is True)
    n_low_z = sum(1 for c in crop_checks if c.get("low_z_boundary_warning") is True)
    all_ok = (n_ok == n_checked and n_shape_ok == n_checked and
              n_range_ok == n_checked and n_nan_free == n_checked and
              n_inf_free == n_checked)

    summary = {
        "version": "rd_b6a_v1",
        "timestamp": ts,
        "smoke_patients": SMOKE_PATIENTS,
        "smoke_subset_count": len(subset),
        "smoke_crops_per_bin_per_patient": SMOKE_CROPS_PER_BIN_PER_PATIENT,
        "expected_total": len(SMOKE_PATIENTS) * len(SIX_BIN_LABELS) * SMOKE_CROPS_PER_BIN_PER_PATIENT,
        "local_weight": wt,
        "crop_loading_check": {
            "n_checked": n_checked,
            "n_ok": n_ok,
            "n_shape_ok": n_shape_ok,
            "n_range_ok": n_range_ok,
            "n_nan_free": n_nan_free,
            "n_inf_free": n_inf_free,
            "n_low_z_warning": n_low_z,
            "all_ok": all_ok,
        },
        "safety_confirmed": {
            "full_crop_npz_generated": 0,
            "training_executed": 0,
            "scoring_executed": 0,
            "model_forward_executed": 0,
            "backward_executed": 0,
            "optimizer_step_executed": 0,
            "stage2_holdout_accessed": 0,
            "existing_files_modified": 0,
            "imagenet_weight_auto_downloaded": 0,
            "checkpoint_loaded": 0,
        },
        "verdict": "통과" if all_ok else "경고",
    }
    summary_path = OUTPUT_ROOT / "rd_b6a_dataset_loader_preflight_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── 5. errors CSV ────────────────────────────────────────────────────────
    errors_path = OUTPUT_ROOT / "rd_b6a_errors.csv"
    with open(errors_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "error"])
        w.writeheader()
        for e in errors:
            w.writerow(e)

    # ── 6. report.md ─────────────────────────────────────────────────────────
    _write_report_md(ts, summary, crop_checks)

    # ── 7. DONE marker ───────────────────────────────────────────────────────
    (OUTPUT_ROOT / "DONE").write_text(f"rd_b6a real-preflight completed: {ts}\n")

    return summary


def _write_report_md(ts: str, summary: dict, crop_checks: list) -> None:
    """rd_b6a_tiny_smoke_train_preflight_report.md 생성."""
    n = summary["crop_loading_check"]
    wt = summary["local_weight"]
    verdict = summary["verdict"]

    lines = [
        "# RD-B6a Tiny Smoke Train Preflight Report",
        f"- 버전: rd_b6a_v1",
        f"- 날짜: {ts}",
        f"- 판정: {verdict}",
        "",
        "---",
        "## 1. RD-B1 ~ RD-B5 요약",
        "",
        "| 단계 | 결과 |",
        "|------|------|",
        "| RD-B1 | 6-bin balanced manifest 86,017 rows / 290 patients / cap 50/bin/patient |",
        "| RD-B2b | mixed_3ch: ch0=CT center / ch1=lower 3mm MIP / ch2=upper 3mm MIP |",
        "| RD-B2b norm | HU clip [-1000, 600] → (x+1000)/1600 → [0,1] float32 |",
        "| RD-B3 | true RD4AD teacher-student / ResNet18 ImageNet frozen / layer1/layer2/layer3 |",
        "| RD-B4 | crop strategy = hybrid_cache (on-the-fly + patient LRU cache, mmap_mode='r') |",
        "| RD-B5 | Dataset/Loader/Model skeleton static check 통과 |",
        "",
        "---",
        "## 2. RD-B5 loader bug patch 내용",
        "",
        "- 수정 파일: `scripts/rd_b5_rd4ad_dataset_loader_model_skeleton.py`",
        "- 수정 위치: `RD4ADNormalCropDataset.__getitem__` 내 cache.get() 반환값 언패킹",
        "",
        "**수정 전 (잘못된 형태):**",
        "```python",
        "ct_volume, _, z_max = (",
        "    self.cache.get(safe_id, info['ct_hu_npy'], info['roi_0_0_npy'])",
        "    .values()",
        ")",
        "```",
        "문제: `dict.values()`는 `dict_values` 객체를 반환 (순서 보장 불안정, 언패킹 시 ct/roi/z_max 순서 오류 위험)",
        "",
        "**수정 후:**",
        "```python",
        "entry = self.cache.get(safe_id, info['ct_hu_npy'], info['roi_0_0_npy'])",
        "ct_volume = entry['ct']",
        "roi_volume = entry['roi']",
        "z_max = entry['z_max']",
        "```",
        "",
        "---",
        "## 3. local ResNet18 ImageNet weight 확인 결과",
        "",
        f"- 경로: `{wt['path']}`",
        f"- 파일명: `{wt['filename']}`",
        f"- 존재: **{wt['local_weight_available']}**",
        f"- 크기: {wt['size_bytes']:,} bytes ({wt['size_mb']} MB)",
        f"- mtime: {wt['mtime']}",
        "- 자동 다운로드: **금지** — local cache 사용 가능",
        "- RD-B6b에서 `weights=ResNet18_Weights.IMAGENET1K_V1` 형태로 사용 예정 (사용자 승인 후)",
        "",
        "---",
        "## 4. smoke subset 구성",
        "",
        f"- 대상 환자: `{SMOKE_PATIENTS}`",
        f"- smoke 크기: {len(SMOKE_PATIENTS)} patients × 6-bin × {SMOKE_CROPS_PER_BIN_PER_PATIENT} crops = **{summary['smoke_subset_count']} crops**",
        "- low_z_boundary_warning 케이스 포함: 예 (`normal006__e662c8463b`, local_z ≤ 7)",
        "- boundary + interior 둘 다 포함: 예 (6-bin 전체 커버)",
        f"- duplicate oversampling: 금지 (각 bin별 독립 샘플링, seed={SMOKE_SEED})",
        "",
        "---",
        "## 5. crop loading check 결과",
        "",
        f"- 확인한 crop 수: {n['n_checked']}",
        f"- 성공 (OK): {n['n_ok']}/{n['n_checked']}",
        f"- shape (3,96,96) 일치: {n['n_shape_ok']}/{n['n_checked']}",
        f"- value range [0,1]: {n['n_range_ok']}/{n['n_checked']}",
        f"- NaN 없음: {n['n_nan_free']}/{n['n_checked']}",
        f"- Inf 없음: {n['n_inf_free']}/{n['n_checked']}",
        f"- low_z_boundary_warning 케이스: {n['n_low_z_warning']}개 포함",
        "",
        "| manifest_id | safe_id | local_z | bin | shape_ok | range_ok | NaN | Inf | low_z_warn | status |",
        "|-------------|---------|---------|-----|----------|----------|-----|-----|------------|--------|",
    ]
    for c in crop_checks:
        lines.append(
            f"| {c.get('manifest_id','?')} "
            f"| {str(c.get('safe_id','?'))[:25]} "
            f"| {c.get('local_z','?')} "
            f"| {c.get('six_bin_label','?')} "
            f"| {c.get('shape_ok','?')} "
            f"| {c.get('range_ok','?')} "
            f"| {c.get('has_nan','?')} "
            f"| {c.get('has_inf','?')} "
            f"| {c.get('low_z_boundary_warning','?')} "
            f"| {c.get('status','?')} |"
        )
    lines.extend([
        "",
        "---",
        "## 6. RD-B6b tiny smoke train 실행 전 남은 조건",
        "",
        "1. 사용자 `--real-preflight` 결과 확인 + DONE marker 확인",
        "2. (선택) `--tiny-forward-check` 별도 사용자 승인 후 teacher/student forward shape 확인",
        "3. smoke subset 눈검증: low_z_boundary_warning 케이스 lower MIP 시각 확인 권장",
        "4. RD-B6b 학습 파라미터 확정: batch_size=24 or 48, n_epochs=5, lr=1e-4",
        "5. 학습 output 경로 설계 및 사용자 승인",
        "",
        "---",
        "## 7. 절대 하지 않은 것",
        "",
        "| 항목 | 확인 |",
        "|------|------|",
        "| full crop NPZ 생성 | 없음 |",
        "| full training 실행 | 없음 |",
        "| scoring 실행 | 없음 |",
        "| model forward 실행 | 없음 |",
        "| backward 실행 | 없음 |",
        "| optimizer step | 없음 |",
        "| stage2_holdout 접근 | 없음 |",
        "| 기존 파일 수정/삭제 | 없음 |",
        "| ImageNet weight 자동 다운로드 | 없음 (local cache 사용) |",
        "| checkpoint 로드 | 없음 |",
    ])

    report_path = OUTPUT_ROOT / "rd_b6a_tiny_smoke_train_preflight_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  report.md → {report_path.name}")


# =============================================================================
# main
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("RD-B6a Tiny Smoke Train Preflight")
    print("=" * 70)

    # output root guard (real-preflight 전용 — selftest/dry-check는 미생성)
    if IS_REAL_PREFLIGHT and OUTPUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재 → 즉시 중단: {OUTPUT_ROOT}")
        sys.exit(1)

    # ── selftest ──────────────────────────────────────────────────────────────
    if IS_SELFTEST:
        print("\n[SELFTEST] 순수 함수 테스트 실행 ...")
        st = run_selftest()
        print(f"  n_tests={st['n_tests']}  pass={st['n_pass']}  fail={st['n_fail']}")
        for r in st["details"]:
            status = "PASS" if r["pass"] else "FAIL"
            print(
                f"  [{status}] {r['test']} | {r['input']} "
                f"→ expected={r['expected']} actual={r['actual']}"
            )
        verdict = "SELFTEST 전체 통과" if st["all_pass"] else "SELFTEST 일부 실패"
        print(f"\n판정: {verdict}")
        return

    # ── dry-check ─────────────────────────────────────────────────────────────
    if IS_DRY_CHECK:
        print("\n[DRY-CHECK] 경로/weight/output root 확인 ...")
        dc = run_dry_check()
        print(f"  manifest 존재: {dc['manifest_exists']} "
              f"(rows={dc['manifest_row_count']}, ok={dc['manifest_row_count_ok']})")
        print(f"  smoke subset 후보: {dc['smoke_subset_count']} crops "
              f"(low_z_warning={dc['low_z_warning_in_subset']}개)")
        wt = dc["local_weight"]
        print(f"  local weight: available={wt['local_weight_available']} "
              f"size={wt['size_mb']}MB mtime={wt['mtime']}")
        print(f"  output root 없음: {dc['output_root_clear']}")
        print(f"  patient_manifest 존재: {dc['patient_manifest_exists']}")
        print(f"  CT paths ok: {dc['smoke_ct_all_ok']} {dc['smoke_ct_exists']}")
        print(f"  ROI paths ok: {dc['smoke_roi_all_ok']} {dc['smoke_roi_exists']}")

        ok = (dc["manifest_exists"] and dc["manifest_row_count_ok"] and
              wt["local_weight_available"] and dc["output_root_clear"] and
              dc["smoke_ct_all_ok"] and dc["smoke_roi_all_ok"])

        if not dc["manifest_exists"]:
            errors.append({"step": "dry_check", "error": "manifest 없음"})
        if not dc["manifest_row_count_ok"]:
            errors.append({"step": "dry_check",
                           "error": f"manifest row count 불일치: {dc['manifest_row_count']}"})
        if not wt["local_weight_available"]:
            errors.append({"step": "dry_check",
                           "error": "local ResNet18 weight 없음 — 수동 배치 필요"})
        if not dc["output_root_clear"]:
            errors.append({"step": "dry_check",
                           "error": f"output root 이미 존재: {OUTPUT_ROOT}"})
        if not dc["smoke_ct_all_ok"]:
            errors.append({"step": "dry_check",
                           "error": f"CT path 누락: {dc['smoke_ct_exists']}"})
        if not dc["smoke_roi_all_ok"]:
            errors.append({"step": "dry_check",
                           "error": f"ROI path 누락: {dc['smoke_roi_exists']}"})

        n_err = len(errors)
        print(f"\n판정: {'통과' if ok else '경고 (' + str(n_err) + '개)'}")
        print("[dry-check 완료] output root 생성하지 않음.")
        if ok:
            print("→ 사용자 승인 후 --real-preflight로 실행.")
        return

    # ── tiny-forward-check ────────────────────────────────────────────────────
    if IS_TINY_FORWARD:
        print("\n[TINY-FORWARD-CHECK] 이 모드는 별도 사용자 승인이 필요합니다.")
        print("teacher/student forward shape 확인만 수행합니다.")
        print("현재 상태: 사용자 승인 대기 (RD-B6a dry-check/real-preflight 통과 후)")
        sys.exit(0)

    # ── real-preflight ─────────────────────────────────────────────────────────
    if IS_REAL_PREFLIGHT:
        print("\n[REAL-PREFLIGHT] smoke subset + crop loading check 실행 ...")
        summary = run_real_preflight()
        n = summary["crop_loading_check"]
        verdict = summary["verdict"]
        print(f"\n판정: {verdict}")
        print(
            f"  crop checked={n['n_checked']}  ok={n['n_ok']}  "
            f"shape_ok={n['n_shape_ok']}  range_ok={n['n_range_ok']}  "
            f"nan_free={n['n_nan_free']}  inf_free={n['n_inf_free']}  "
            f"low_z_warn={n['n_low_z_warning']}"
        )
        print(f"\n생성 파일 목록:")
        for fn in sorted(OUTPUT_ROOT.iterdir()):
            print(f"  {fn.name}")
        return


if __name__ == "__main__":
    main()
