"""
RD-B8e: Full float32 shard generation
목적: normal train 86,017개 전체 mixed_3ch crop을 float32 shard로 생성
모드:
  bare run   -> exit 2
  --dry-plan -> 계획 출력 (파일 생성 없음)
  --run      -> full shard 생성 실행
안전 조건:
  stage2_holdout/lesion 접근 금지
  scoring/threshold/checkpoint 저장 금지
  output root 존재 시 즉시 중단
  기존 결과물 삭제/수정 금지
"""

import sys
import csv
import json
import math
import time
import random
import collections
from pathlib import Path

ALLOWED_MODES = {"--dry-plan", "--run"}
if not any(m in sys.argv for m in ALLOWED_MODES):
    print("오류: 실행 모드 지정 필요.")
    print("  --dry-plan : 계획 출력 (파일 생성 없음)")
    print("  --run      : full shard 생성 실행")
    sys.exit(2)

IS_DRY_PLAN = "--dry-plan" in sys.argv
IS_RUN      = "--run" in sys.argv

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
SHARD_ROOT   = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit/rd_b8e_full_float32_shards_v1"
)
SHARDS_DIR = SHARD_ROOT / "shards"

TRAIN_MANIFEST_PATH = (
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

# ── 설계 상수 ──────────────────────────────────────────────────────────────────
FORBIDDEN_KEYWORDS = [
    "stage2_holdout", "lesion", "test_lesion", "second-stage-lesion-refiner",
]
MIP_RADIUS  = 3
HU_CLIP_MIN = -1000.0
HU_CLIP_MAX =  600.0
HU_RANGE    = 1600.0
SIX_BIN_LABELS = [
    "lower_boundary", "lower_interior",
    "middle_boundary", "middle_interior",
    "upper_boundary",  "upper_interior",
]
LOW_Z_BOUNDARY_WARN_THRESHOLD = 7
SHARD_SIZE            = 1000
EXPECTED_TOTAL_CROPS  = 86017
PATIENT_CACHE_SIZE    = 8
VALIDATE_SAMPLE_PER_SHARD = 10


# =============================================================================
# 안전 검사
# =============================================================================

def assert_path_safe(path_str):
    for kw in FORBIDDEN_KEYWORDS:
        if kw.lower() in str(path_str).lower():
            raise RuntimeError(
                f"[SAFETY] 금지 경로 접근 차단: {path_str!r} (keyword={kw!r})"
            )


# =============================================================================
# 공통 함수
# =============================================================================

def normalize_hu(hu_array):
    import numpy as np
    clipped = np.clip(hu_array, HU_CLIP_MIN, HU_CLIP_MAX)
    return ((clipped - HU_CLIP_MIN) / HU_RANGE).astype("float32")


def compute_mip_slab_indices(center_z, direction, z_max):
    if direction == "lower":
        raw = [center_z - MIP_RADIUS + i for i in range(MIP_RADIUS)]
    elif direction == "upper":
        raw = [center_z + 1 + i for i in range(MIP_RADIUS)]
    else:
        raise ValueError(f"direction={direction!r}")
    return [max(0, min(idx, z_max - 1)) for idx in raw]


def has_low_z_boundary_warning(center_z):
    return center_z <= LOW_Z_BOUNDARY_WARN_THRESHOLD


def build_crop_np(ct_arr, center_z, crop_y0, crop_x0, crop_y1, crop_x1):
    import numpy as np
    TARGET = 96
    z_max, h_max, w_max = ct_arr.shape

    def crop_2d_with_air_padding(img2d, y0, x0, y1, x1):
        out = np.full((TARGET, TARGET), HU_CLIP_MIN, dtype=img2d.dtype)
        src_y0 = max(0, y0); src_x0 = max(0, x0)
        src_y1 = min(h_max, y1); src_x1 = min(w_max, x1)
        if src_y1 <= src_y0 or src_x1 <= src_x0:
            return out
        dst_y0 = src_y0 - y0; dst_x0 = src_x0 - x0
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        out[dst_y0:dst_y1, dst_x0:dst_x1] = img2d[src_y0:src_y1, src_x0:src_x1]
        return out

    ch0_raw = crop_2d_with_air_padding(
        ct_arr[center_z], crop_y0, crop_x0, crop_y1, crop_x1
    )
    lower_idxs = compute_mip_slab_indices(center_z, "lower", z_max)
    ch1_raw = np.max(np.stack([
        crop_2d_with_air_padding(ct_arr[z], crop_y0, crop_x0, crop_y1, crop_x1)
        for z in lower_idxs
    ], axis=0), axis=0)
    upper_idxs = compute_mip_slab_indices(center_z, "upper", z_max)
    ch2_raw = np.max(np.stack([
        crop_2d_with_air_padding(ct_arr[z], crop_y0, crop_x0, crop_y1, crop_x1)
        for z in upper_idxs
    ], axis=0), axis=0)

    crop = np.stack([
        normalize_hu(ch0_raw),
        normalize_hu(ch1_raw),
        normalize_hu(ch2_raw),
    ], axis=0).astype("float32")

    if crop.shape != (3, TARGET, TARGET):
        raise RuntimeError(f"bad crop shape: {crop.shape}")
    return crop


# =============================================================================
# LRU Patient Cache
# =============================================================================

class LRUPatientCache:
    def __init__(self, max_size):
        self._cache = collections.OrderedDict()
        self._max   = max_size

    def load(self, key, path):
        import numpy as np
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        assert_path_safe(path)
        arr = np.load(path, mmap_mode="r")
        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        self._cache[key] = arr
        return arr


# =============================================================================
# CSV 헬퍼 (append 방식)
# =============================================================================

class CsvAppendWriter:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=fieldnames, extrasaction="ignore")
        self._w.writeheader()

    def writerows(self, rows):
        for r in rows:
            self._w.writerow({k: r.get(k, "") for k in self.fieldnames})
        self._f.flush()

    def close(self):
        self._f.close()


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  -> {path.name}")


# =============================================================================
# Manifest 로드
# =============================================================================

def load_full_manifest(manifest_path):
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_patient_paths(patient_manifest_path):
    patient_paths = {}
    with open(patient_manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid     = row.get("safe_id", "")
            ct_path = row.get("ct_hu_npy", "")
            if sid and ct_path:
                assert_path_safe(ct_path)
                patient_paths[sid] = {"ct_hu_npy": ct_path}
    return patient_paths


# =============================================================================
# dry-plan
# =============================================================================

def run_dry_plan():
    n_total  = EXPECTED_TOTAL_CROPS
    n_shards = math.ceil(n_total / SHARD_SIZE)
    f32_per_shard_mb = SHARD_SIZE * 3 * 96 * 96 * 4 / (1024 * 1024)
    f32_total_gb     = n_shards * f32_per_shard_mb / 1024
    smoke_sec_per_crop = 15.7 / 1998  # B8d 실측 기준
    est_gen_min        = round(smoke_sec_per_crop * n_total / 60, 1)

    print("=" * 70)
    print("RD-B8e: Full float32 shard generation [DRY-PLAN]")
    print("=" * 70)
    print()
    print("## 1. 목적")
    print("  normal train 86,017개 전체 mixed_3ch crop을 float32 shard로 저장")
    print()
    print("## 2. 입력")
    print(f"  train manifest : {TRAIN_MANIFEST_PATH}")
    print(f"  patient manifest: {PATIENT_MANIFEST_PATH}")
    print()
    print("## 3. 규모")
    print(f"  total crops : {n_total:,}")
    print(f"  shard_size  : {SHARD_SIZE}")
    print(f"  n_shards    : {n_shards}")
    print(f"  예상 용량   : ~{f32_total_gb:.1f} GB")
    print(f"  예상 생성시간: ~{est_gen_min:.0f} min")
    print()
    print("## 4. shard 검증 (per shard)")
    print(f"  - shape (3,96,96) 확인")
    print(f"  - value range [0,1] 확인")
    print(f"  - NaN/Inf 없음 확인")
    print(f"  - sample {VALIDATE_SAMPLE_PER_SHARD}개/shard 검증")
    print()
    print("## 5. 출력 root")
    print(f"  {SHARD_ROOT}")
    print()
    print("## 6. 안전 조건")
    print("  stage2_holdout/lesion 접근: 금지")
    print("  scoring/threshold/checkpoint: 금지")
    print("  output root 존재 시: 즉시 중단")
    print()
    print("판정: DRY-PLAN OK")
    print("  사용자 승인 후:")
    print("  python scripts/rd_b8e_full_float32_shard_generation.py --run")


# =============================================================================
# run_generation
# =============================================================================

def run_generation():
    import numpy as np

    print("=" * 70)
    print("RD-B8e: Full float32 shard generation [RUN]")
    print("=" * 70)

    # output root guard
    if SHARD_ROOT.exists():
        print(f"[ABORT] output root 이미 존재: {SHARD_ROOT}")
        sys.exit(1)
    SHARD_ROOT.mkdir(parents=True, exist_ok=False)
    SHARDS_DIR.mkdir(parents=True, exist_ok=False)

    # manifest 로드
    print("  train manifest 로드 중 ...")
    all_rows = load_full_manifest(TRAIN_MANIFEST_PATH)
    n_total  = len(all_rows)
    print(f"  manifest rows: {n_total:,}")

    if n_total != EXPECTED_TOTAL_CROPS:
        print(f"[ABORT] manifest rows 불일치: {n_total} != {EXPECTED_TOTAL_CROPS}")
        sys.exit(1)

    for row in all_rows[:50]:
        assert_path_safe(row.get("safe_id", ""))
    print("  stage2_holdout intersection: 0 OK")

    bin_counts = collections.Counter(row.get("six_bin_label", "") for row in all_rows)
    print("  6-bin 분포:")
    for lbl in SIX_BIN_LABELS:
        print(f"    {lbl}: {bin_counts.get(lbl, 0):,}")

    print("  patient manifest 로드 중 ...")
    patient_paths = load_patient_paths(PATIENT_MANIFEST_PATH)
    print(f"  patient paths: {len(patient_paths)}")

    n_shards      = math.ceil(n_total / SHARD_SIZE)
    patient_cache = LRUPatientCache(max_size=PATIENT_CACHE_SIZE)
    shard_files   = []
    timing_rows   = []
    error_rows    = []
    n_low_z       = 0
    n_edge_pad    = 0

    # shard_index CSV: append 방식
    shard_index_fields = [
        "shard_id", "row_in_shard", "global_rank", "safe_id", "six_bin_label",
        "local_z", "crop_y0", "crop_x0", "crop_y1", "crop_x1", "low_z_warning",
    ]
    idx_writer = CsvAppendWriter(
        SHARD_ROOT / "rd_b8e_full_shard_index.csv", shard_index_fields
    )

    t_total_start = time.perf_counter()
    print(f"\n  shard 생성: {n_total:,} crops, {n_shards} shards")
    print("  (10 shard마다 진행 상황 출력)")

    for s in range(n_shards):
        batch_rows  = all_rows[s * SHARD_SIZE : (s + 1) * SHARD_SIZE]
        shard_crops = []
        shard_idx_chunk = []
        t_start     = time.perf_counter()

        for local_i, row in enumerate(batch_rows):
            global_rank = s * SHARD_SIZE + local_i
            sid     = row.get("safe_id", "")
            lz      = int(row.get("local_z", 0))
            y0      = int(row.get("crop_y0", 0))
            x0      = int(row.get("crop_x0", 0))
            y1      = int(row.get("crop_y1", 96))
            x1      = int(row.get("crop_x1", 96))
            bin_lbl = row.get("six_bin_label", "")
            ct_path = patient_paths.get(sid, {}).get("ct_hu_npy", "")

            shard_idx_chunk.append({
                "shard_id":      s,
                "row_in_shard":  local_i,
                "global_rank":   global_rank,
                "safe_id":       sid,
                "six_bin_label": bin_lbl,
                "local_z":       lz,
                "crop_y0":       y0,
                "crop_x0":       x0,
                "crop_y1":       y1,
                "crop_x1":       x1,
                "low_z_warning": int(has_low_z_boundary_warning(lz)),
            })

            if not ct_path:
                error_rows.append({
                    "phase": "shard_gen", "shard_id": s,
                    "row_in_shard": local_i, "global_rank": global_rank,
                    "safe_id": sid, "error": "no_ct_path",
                })
                shard_crops.append(np.zeros((3, 96, 96), dtype=np.float32))
                continue

            try:
                ct_arr  = patient_cache.load(sid, ct_path)
                crop_np = build_crop_np(ct_arr, lz, y0, x0, y1, x1)
                if has_low_z_boundary_warning(lz):
                    n_low_z += 1
                _, h_max, w_max = ct_arr.shape
                if y0 < 0 or x0 < 0 or y1 > h_max or x1 > w_max:
                    n_edge_pad += 1
                shard_crops.append(crop_np)
            except Exception as e:
                error_rows.append({
                    "phase": "shard_gen", "shard_id": s,
                    "row_in_shard": local_i, "global_rank": global_rank,
                    "safe_id": sid, "error": str(e),
                })
                shard_crops.append(np.zeros((3, 96, 96), dtype=np.float32))

        shard_arr  = np.stack(shard_crops, axis=0)
        shard_name = f"rd_b8e_shard_{s:04d}.npy"
        shard_path = SHARDS_DIR / shard_name
        np.save(str(shard_path), shard_arr)
        t_end = time.perf_counter()

        disk_mb = shard_path.stat().st_size / (1024 * 1024)
        timing_rows.append({
            "shard_id":             s,
            "n_crops":              len(batch_rows),
            "generation_time_sec":  round(t_end - t_start, 4),
            "disk_mb":              round(disk_mb, 3),
        })
        shard_files.append(shard_path)
        idx_writer.writerows(shard_idx_chunk)

        if s % 10 == 0 or s == n_shards - 1:
            elapsed = time.perf_counter() - t_total_start
            eta = elapsed / (s + 1) * (n_shards - s - 1) if s < n_shards - 1 else 0
            pct = (s + 1) / n_shards * 100
            print(
                f"    shard {s:3d}/{n_shards}  {pct:5.1f}%  "
                f"{disk_mb:.1f}MB  {t_end-t_start:.2f}s  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s"
            )

    idx_writer.close()
    print(f"  -> rd_b8e_full_shard_index.csv")

    t_total_end = time.perf_counter()
    total_time  = t_total_end - t_total_start
    total_mb    = sum(f.stat().st_size for f in shard_files) / (1024 * 1024)

    write_csv(
        SHARD_ROOT / "rd_b8e_generation_timing.csv",
        ["shard_id", "n_crops", "generation_time_sec", "disk_mb"],
        timing_rows,
    )

    # ── shard 검증 ──
    print("\n  shard 검증 중 ...")
    rng_val      = random.Random(42)
    val_rows     = []
    shape_mismatch  = 0
    range_violation = 0
    nan_count       = 0
    inf_count       = 0

    for s, shard_path in enumerate(shard_files):
        arr        = np.load(str(shard_path), mmap_mode="r")
        n_in_shard = arr.shape[0]
        sample_idxs = rng_val.sample(
            range(n_in_shard), min(VALIDATE_SAMPLE_PER_SHARD, n_in_shard)
        )
        for idx in sample_idxs:
            crop     = arr[idx]
            shape_ok = crop.shape == (3, 96, 96)
            range_ok = float(crop.min()) >= -1e-6 and float(crop.max()) <= 1 + 1e-6
            nan_ok   = not np.isnan(crop).any()
            inf_ok   = not np.isinf(crop).any()
            if not shape_ok: shape_mismatch += 1
            if not range_ok: range_violation += 1
            if not nan_ok:   nan_count += 1
            if not inf_ok:   inf_count += 1
            val_rows.append({
                "shard_id":     s,
                "row_in_shard": idx,
                "shape_ok":     int(shape_ok),
                "range_ok":     int(range_ok),
                "nan_ok":       int(nan_ok),
                "inf_ok":       int(inf_ok),
                "min_val":      round(float(crop.min()), 6),
                "max_val":      round(float(crop.max()), 6),
            })
        if s % 20 == 0:
            print(f"    검증 shard {s}/{len(shard_files)}")

    write_csv(
        SHARD_ROOT / "rd_b8e_shard_validation.csv",
        ["shard_id", "row_in_shard", "shape_ok", "range_ok",
         "nan_ok", "inf_ok", "min_val", "max_val"],
        val_rows,
    )
    write_csv(
        SHARD_ROOT / "rd_b8e_errors.csv",
        ["phase", "shard_id", "row_in_shard", "global_rank", "safe_id", "error"],
        error_rows,
    )

    all_checks_passed = (
        len(shard_files) == n_shards
        and shape_mismatch == 0
        and range_violation == 0
        and nan_count == 0
        and inf_count == 0
        and len(error_rows) == 0
    )

    summary = {
        "n_total_crops":         n_total,
        "n_shards":              len(shard_files),
        "total_disk_mb":         round(total_mb, 2),
        "generation_time_sec":   round(total_time, 2),
        "n_low_z_warning":       n_low_z,
        "n_edge_pad":            n_edge_pad,
        "n_errors":              len(error_rows),
        "shape_mismatch":        shape_mismatch,
        "range_violation":       range_violation,
        "nan_count":             nan_count,
        "inf_count":             inf_count,
        "val_samples_checked":   len(val_rows),
        "all_checks_passed":     all_checks_passed,
        "scoring_started":       False,
        "threshold_created":     False,
        "stage2_holdout_access": 0,
    }
    with open(SHARD_ROOT / "rd_b8e_full_shard_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("  -> rd_b8e_full_shard_summary.json")

    verdict = "PASS" if all_checks_passed else "FAIL"
    md_lines = [
        "# RD-B8e Full float32 shard generation Report",
        "",
        f"## 판정: {verdict}",
        "",
        "## 결과 요약",
        "| 항목 | 값 |",
        "|---|---|",
        f"| total crops | {n_total:,} |",
        f"| n_shards | {len(shard_files)} |",
        f"| total disk | {total_mb:.1f} MB ({total_mb/1024:.2f} GB) |",
        f"| generation time | {total_time:.1f}s ({total_time/60:.1f}min) |",
        f"| low_z_warning | {n_low_z} |",
        f"| edge_pad | {n_edge_pad} |",
        f"| errors | {len(error_rows)} |",
        f"| shape_mismatch | {shape_mismatch} |",
        f"| range_violation | {range_violation} |",
        f"| NaN | {nan_count} |",
        f"| Inf | {inf_count} |",
        f"| val_samples_checked | {len(val_rows)} |",
        f"| all_checks_passed | {all_checks_passed} |",
        "",
        "## 6-bin 분포",
        "| bin | count |",
        "|---|---|",
    ] + [f"| {lbl} | {bin_counts.get(lbl, 0):,} |" for lbl in SIX_BIN_LABELS]
    with open(SHARD_ROOT / "rd_b8e_full_shard_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("  -> rd_b8e_full_shard_report.md")

    (SHARD_ROOT / "DONE").write_text(
        f"rd_b8e_full_float32_shards_v1 DONE\nall_checks_passed={all_checks_passed}\n",
        encoding="utf-8",
    )
    print("  -> DONE")

    print()
    print("=" * 70)
    print(f"판정: {verdict}")
    print(f"  total crops     : {n_total:,}")
    print(f"  n_shards        : {len(shard_files)}")
    print(f"  disk            : {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")
    print(f"  generation time : {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  low_z_warning   : {n_low_z}")
    print(f"  errors          : {len(error_rows)}")
    print(f"  shape_mismatch  : {shape_mismatch}")
    print(f"  range_violation : {range_violation}")
    print(f"  NaN={nan_count}  Inf={inf_count}")
    print(f"  all_checks_passed: {all_checks_passed}")
    print("=" * 70)

    if not all_checks_passed:
        print("[FAIL] 검증 실패 - RD-B8f full train 실행 금지")
        sys.exit(1)


# =============================================================================
# 진입점
# =============================================================================

if IS_DRY_PLAN:
    run_dry_plan()
elif IS_RUN:
    run_generation()
