"""
RD-B4: Crop Generation Strategy Preflight
목적: on-the-fly vs pre-generated NPZ vs hybrid cache 방식 비교 및 최종 추천
안전 조건: stage2_holdout 접근 금지, 학습/scoring/model forward/GPU 금지, 기존 파일 수정 금지
실행 방법: python rd_b4_crop_generation_strategy_preflight.py --dry-run  (계획 확인)
           python rd_b4_crop_generation_strategy_preflight.py --real     (실제 실행)
"""

import sys
import os
import json
import csv
import math
import time
from pathlib import Path
from collections import defaultdict

# ─── bare-run guard ────────────────────────────────────────────────────────────
if "--dry-run" not in sys.argv and "--real" not in sys.argv:
    print("오류: --dry-run 또는 --real 인자가 필요합니다.")
    print("  dry-run: python rd_b4_crop_generation_strategy_preflight.py --dry-run")
    print("  real:    python rd_b4_crop_generation_strategy_preflight.py --real")
    sys.exit(1)

IS_DRY_RUN = "--dry-run" in sys.argv

# ─── 경로 설정 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/normal_based_stage2_verifier_audit/rd_b4_crop_generation_strategy_preflight_v1"

MANIFEST_PATH = PROJECT_ROOT / "outputs/normal_based_stage2_verifier_audit/rd_b1_6bin_balanced_manifest_preflight_v1/rd_b1_6bin_balanced_normal_train_coordinate_manifest.csv"
PATIENT_MANIFEST_PATH = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/manifests/patient_manifest.csv")
VOLUMES_NPY_ROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")

# ─── 안전 검사 ─────────────────────────────────────────────────────────────────
STAGE2_HOLDOUT_FORBIDDEN = [
    "stage2_holdout", "lesion", "test_lesion", "second-stage-lesion-refiner-v1/datasets"
]

def check_path_safety(path_str):
    for keyword in STAGE2_HOLDOUT_FORBIDDEN:
        if keyword.lower() in str(path_str).lower():
            raise RuntimeError(f"[SAFETY] stage2_holdout/lesion 접근 금지: {path_str}")

# ─── crop 설정 ─────────────────────────────────────────────────────────────────
CROP_SIZE = 96
N_CHANNELS = 3
MIP_RADIUS = 3      # z±3 → 3mm MIP (z_spacing=1.0mm 기준)
HU_CLIP_MIN = -1000
HU_CLIP_MAX = 600
DTYPE_TRAIN = "float32"
DTYPE_CACHE = "float16"

# ─── 디스크 추정 상수 ───────────────────────────────────────────────────────────
CROP_ELEMENTS = N_CHANNELS * CROP_SIZE * CROP_SIZE       # 3 * 96 * 96 = 27,648
FLOAT32_BYTES_PER_CROP = CROP_ELEMENTS * 4               # 110,592 bytes ≈ 108 KB
FLOAT16_BYTES_PER_CROP = CROP_ELEMENTS * 2               # 55,296 bytes ≈ 54 KB
NPZ_COMPRESS_RATIO = 0.45                                # 압축 npz 예상 비율 (CT 특성 반영)
NPZ_OVERHEAD_PER_FILE = 256                              # per-file npz 메타 오버헤드 bytes

# ─── 실행 추정 상수 ─────────────────────────────────────────────────────────────
EST_CT_LOAD_SEC = 0.08           # mmap_mode='r' CT 로드 추정 (초)
EST_CROP_GEN_SEC = 0.0002        # crop 1개 생성 추정 (초, numpy slicing+MIP)
EST_CT_SIZE_MB = 123             # normal001 기준 CT 파일 크기 (MB)
EST_ROI_SIZE_MB = 62             # normal001 기준 ROI 파일 크기 (MB)

# ─── 학습 DataLoader 설정 ──────────────────────────────────────────────────────
BATCH_SIZE_CANDIDATE_A = 24      # bin당 4개
BATCH_SIZE_CANDIDATE_B = 48      # bin당 8개
N_BINS = 6
N_WORKERS_CANDIDATE = 4

errors = []

# =============================================================================
# STEP 1: manifest 읽기
# =============================================================================
print("[STEP 1] RD-B1 manifest 읽기 ...")
check_path_safety(MANIFEST_PATH)

manifest_rows = []
with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        manifest_rows.append(row)

total_rows = len(manifest_rows)
print(f"  manifest 총 rows: {total_rows}")

# ─── patient/bin 통계 ─────────────────────────────────────────────────────────
patients = set()
safe_ids = set()
bin_counts = defaultdict(int)
z_vals = []
patient_z_map = defaultdict(list)   # safe_id → [local_z, ...]

for row in manifest_rows:
    patients.add(row["patient_id"])
    safe_ids.add(row["safe_id"])
    bin_counts[row["six_bin_label"]] += 1
    z = int(row["local_z"])
    z_vals.append(z)
    patient_z_map[row["safe_id"]].append(z)

n_patients = len(patients)
n_safe_ids = len(safe_ids)
print(f"  unique patient_id: {n_patients}")
print(f"  unique safe_id:    {n_safe_ids}")
print(f"  local_z 범위: {min(z_vals)} ~ {max(z_vals)}")
print(f"  bin별 row 수:")
for k, v in sorted(bin_counts.items()):
    print(f"    {k}: {v}")

# =============================================================================
# STEP 2: patient_manifest에서 CT/ROI 경로 290명 존재 확인
# =============================================================================
print("\n[STEP 2] CT/ROI 경로 290명 존재 확인 ...")
check_path_safety(PATIENT_MANIFEST_PATH)

patient_info = {}
with open(PATIENT_MANIFEST_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["split"] == "train":
            patient_info[row["safe_id"]] = {
                "ct_hu_npy": row["ct_hu_npy"],
                "roi_0_0_npy": row["roi_0_0_npy"],
                "meta_json": row["meta_json"],
            }

ct_exist = 0
ct_missing = []
roi_exist = 0
roi_missing = []

for sid in safe_ids:
    if sid not in patient_info:
        ct_missing.append(sid)
        roi_missing.append(sid)
        errors.append({"safe_id": sid, "error": "patient_manifest에 없음", "step": "step2"})
        continue
    ct_path = Path(patient_info[sid]["ct_hu_npy"])
    roi_path = Path(patient_info[sid]["roi_0_0_npy"])
    if ct_path.exists():
        ct_exist += 1
    else:
        ct_missing.append(sid)
        errors.append({"safe_id": sid, "error": f"CT 파일 없음: {ct_path}", "step": "step2"})
    if roi_path.exists():
        roi_exist += 1
    else:
        roi_missing.append(sid)
        errors.append({"safe_id": sid, "error": f"ROI 파일 없음: {roi_path}", "step": "step2"})

print(f"  CT 존재: {ct_exist}/{n_safe_ids}, 누락: {len(ct_missing)}")
print(f"  ROI 존재: {roi_exist}/{n_safe_ids}, 누락: {len(roi_missing)}")

# =============================================================================
# STEP 3: z_max 수집 (meta.json에서 shape_zyx[0])
# =============================================================================
print("\n[STEP 3] z_max 수집 (meta.json) ...")
patient_z_max = {}
meta_read_errors = []

for sid in safe_ids:
    if sid not in patient_info:
        continue
    meta_path = Path(patient_info[sid]["meta_json"])
    if not meta_path.exists():
        meta_read_errors.append(sid)
        errors.append({"safe_id": sid, "error": f"meta.json 없음: {meta_path}", "step": "step3"})
        continue
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    z_size = meta["shape_zyx"][0]
    patient_z_max[sid] = z_size

print(f"  z_max 수집 완료: {len(patient_z_max)}/{n_safe_ids}")
if meta_read_errors:
    print(f"  meta 읽기 오류: {len(meta_read_errors)}")

# =============================================================================
# STEP 4: mixed_3ch z±3 접근 가능 여부 확인 → edge clamp 계산
# =============================================================================
print("\n[STEP 4] edge clamp 분석 ...")

n_lower_clamp = 0   # z < 3 → lower MIP (z-3~z-1) 일부 없음
n_upper_clamp = 0   # z > z_max-4 → upper MIP (z+1~z+3) 일부 없음
n_both_clamp = 0
n_low_z_boundary_warning = 0  # z <= 7 (diaphragm saturation risk)

clamp_details = []

for row in manifest_rows:
    sid = row["safe_id"]
    z = int(row["local_z"])
    z_max = patient_z_max.get(sid, None)
    if z_max is None:
        continue

    lower_clamp = z < MIP_RADIUS          # z < 3: z-3~z-1에서 일부 clamp
    upper_clamp = z > z_max - 1 - MIP_RADIUS  # z+3 >= z_max

    if lower_clamp:
        n_lower_clamp += 1
    if upper_clamp:
        n_upper_clamp += 1
    if lower_clamp and upper_clamp:
        n_both_clamp += 1

    if z <= 7:
        n_low_z_boundary_warning += 1

    if lower_clamp or upper_clamp:
        clamp_details.append({
            "safe_id": sid,
            "local_z": z,
            "z_max": z_max,
            "lower_clamp": lower_clamp,
            "upper_clamp": upper_clamp,
            "six_bin_label": row["six_bin_label"],
        })

n_total_clamp = n_lower_clamp + n_upper_clamp - n_both_clamp
pct_clamp = n_total_clamp / total_rows * 100 if total_rows > 0 else 0

print(f"  lower clamp (z < 3): {n_lower_clamp}")
print(f"  upper clamp (z > z_max-4): {n_upper_clamp}")
print(f"  both clamp: {n_both_clamp}")
print(f"  총 clamp 대상: {n_total_clamp} ({pct_clamp:.2f}%)")
print(f"  low_z_boundary_warning (z<=7): {n_low_z_boundary_warning}")

# =============================================================================
# STEP 5: 디스크 사용량 추정
# =============================================================================
print("\n[STEP 5] 디스크 추정 ...")

n_crops = total_rows

# float32 per-crop
f32_total_bytes = n_crops * FLOAT32_BYTES_PER_CROP
f32_total_gb = f32_total_bytes / (1024 ** 3)

# float16 per-crop
f16_total_bytes = n_crops * FLOAT16_BYTES_PER_CROP
f16_total_gb = f16_total_bytes / (1024 ** 3)

# compressed npz (float32 기준)
compressed_gb_low = f32_total_gb * 0.30   # 최적 압축
compressed_gb_high = f32_total_gb * 0.55  # 일반적 압축

# per-crop vs shard 파일 수
per_crop_files = n_crops
shard_size = 1000
n_shards = math.ceil(n_crops / shard_size)

print(f"  crop 1개 크기 (float32): {FLOAT32_BYTES_PER_CROP/1024:.1f} KB")
print(f"  전체 {n_crops}개 float32: {f32_total_gb:.2f} GB")
print(f"  전체 {n_crops}개 float16: {f16_total_gb:.2f} GB")
print(f"  compressed npz 예상: {compressed_gb_low:.2f} ~ {compressed_gb_high:.2f} GB")
print(f"  per-crop 파일 수: {per_crop_files:,}")
print(f"  shard 방식 ({shard_size}/shard): {n_shards}개")

# =============================================================================
# STEP 6: on-the-fly loader 설계 추정
# =============================================================================
print("\n[STEP 6] on-the-fly loader 설계 추정 ...")

# 290명 CT volume 메모리
avg_ct_mb = EST_CT_SIZE_MB
total_ct_gb = 290 * avg_ct_mb / 1024
avg_roi_mb = EST_ROI_SIZE_MB
total_roi_gb = 290 * avg_roi_mb / 1024

# 1 epoch에 필요한 CT 로드 수 (patient당 1회 mmap)
n_unique_patients_per_epoch = n_patients
est_io_per_epoch_sec = n_unique_patients_per_epoch * EST_CT_LOAD_SEC
est_crop_gen_per_epoch_sec = n_crops * EST_CROP_GEN_SEC
est_total_epoch_sec = est_io_per_epoch_sec + est_crop_gen_per_epoch_sec

# batch sampler
steps_per_epoch_bs24 = n_crops // BATCH_SIZE_CANDIDATE_A
steps_per_epoch_bs48 = n_crops // BATCH_SIZE_CANDIDATE_B

# LRU cache 효과 (290명 중 작업 batch에 중복 확률)
# worker=4, 290명 CT volume을 cache하면 약 290 * (123+62) MB = ~53 GB → 전체 cache 불가
# 현실적: worker당 LRU(8~16 patients) → cache hit 향상
lru_cache_per_worker = 8
total_cache_mb = lru_cache_per_worker * (EST_CT_SIZE_MB + EST_ROI_SIZE_MB)

print(f"  290명 CT 전체 mmap: {total_ct_gb:.1f} GB (disk IO)")
print(f"  290명 ROI 전체 mmap: {total_roi_gb:.1f} GB (disk IO)")
print(f"  1 epoch I/O 추정 (290명 CT 로드): {est_io_per_epoch_sec:.1f}s")
print(f"  1 epoch crop 생성 추정: {est_crop_gen_per_epoch_sec:.1f}s")
print(f"  1 epoch 순수 CPU 추정: {est_total_epoch_sec:.1f}s (GPU transfer 별도)")
print(f"  steps/epoch (bs=24): {steps_per_epoch_bs24}")
print(f"  steps/epoch (bs=48): {steps_per_epoch_bs48}")
print(f"  LRU cache per worker ({lru_cache_per_worker} patients): {total_cache_mb:.0f} MB")

# =============================================================================
# STEP 7: 방식 비교 및 최종 추천
# =============================================================================
print("\n[STEP 7] 방식 비교 및 최종 추천 ...")

strategy_comparison = [
    {
        "strategy": "on_the_fly_loader",
        "disk_gb_estimate": round(f32_total_gb * 0.0, 2),   # crop 저장 없음
        "setup_effort": "low",
        "io_risk": "medium",
        "flexibility": "high",
        "reproducibility": "medium",
        "implementation_complexity": "medium",
        "pros": "디스크 절약|normalization/MIP 변경 용이|실험 유연성 최대",
        "cons": "학습 중 I/O+crop 계산 비용|worker caching 설계 필요",
        "recommended": False,
    },
    {
        "strategy": "pregenerated_npz",
        "disk_gb_estimate": round(f32_total_gb, 2),
        "setup_effort": "high",
        "io_risk": "low",
        "flexibility": "low",
        "reproducibility": "high",
        "implementation_complexity": "low",
        "pros": "학습 속도 빠름|재현성 최고|DataLoader 단순",
        "cons": f"디스크 {f32_total_gb:.1f}GB|normalization 변경 시 전체 재생성|per-crop 파일 {per_crop_files:,}개 관리 부담",
        "recommended": False,
    },
    {
        "strategy": "hybrid_cache",
        "disk_gb_estimate": round(f32_total_gb * 0.0, 2),   # 기본은 on-the-fly
        "setup_effort": "medium",
        "io_risk": "low",
        "flexibility": "high",
        "reproducibility": "high",
        "implementation_complexity": "medium",
        "pros": "실험 유연성+속도 균형|patient-level LRU cache|smoke NPZ 소규모 생성만|normalization 변경 시 cache만 갱신",
        "cons": "구현 복잡도 중간|cache 일관성 관리 필요",
        "recommended": True,
    },
]

print("  최종 추천: hybrid_cache")
print("  이유:")
print("    - 현재 단계는 RD4AD 첫 smoke train → normalization/MIP 변경 가능성 높음")
print("    - 전체 NPZ 생성은 9.3GB+파일 86,017개 → 정착 전 과다 비용")
print("    - patient LRU cache로 I/O 위험 완화 가능")
print("    - smoke subset NPZ(1~2 환자)만 먼저 생성해 DataLoader 검증")

# =============================================================================
# STEP 8: loader 설계 상세
# =============================================================================
loader_design = {
    "dataset_class": "RD4ADCropDataset",
    "manifest_path": str(MANIFEST_PATH),
    "patient_manifest_path": str(PATIENT_MANIFEST_PATH),
    "crop_size": CROP_SIZE,
    "n_channels": N_CHANNELS,
    "mip_radius": MIP_RADIUS,
    "hu_clip": [HU_CLIP_MIN, HU_CLIP_MAX],
    "normalization": "(hu_clipped + 1000) / 1600",
    "output_dtype": DTYPE_TRAIN,
    "ct_load_mode": "mmap_mode='r'",
    "lru_cache_per_worker": lru_cache_per_worker,
    "batch_sampler": "SixBinBalancedBatchSampler",
    "batch_size_candidates": [BATCH_SIZE_CANDIDATE_A, BATCH_SIZE_CANDIDATE_B],
    "n_workers_candidate": N_WORKERS_CANDIDATE,
    "edge_clamp_strategy": "numpy.clip(z_slab_indices, 0, z_max-1)",
    "duplicate_oversampling": False,
    "patient_leakage_prevention": "safe_id를 split=train에서만 사용",
    "imagenet_normalization": "RD-B5/RD-B6 ablation으로 보류",
}

# =============================================================================
# dry-run이면 여기서 결과를 출력하고 종료
# =============================================================================
if IS_DRY_RUN:
    print("\n" + "="*60)
    print("[DRY-RUN 결과 요약]")
    print("="*60)
    print(f"  manifest rows: {total_rows}")
    print(f"  patients: {n_patients}, safe_ids: {n_safe_ids}")
    print(f"  CT 존재: {ct_exist}/{n_safe_ids}, 누락: {len(ct_missing)}")
    print(f"  ROI 존재: {roi_exist}/{n_safe_ids}, 누락: {len(roi_missing)}")
    print(f"  edge clamp 대상: {n_total_clamp} ({pct_clamp:.2f}%)")
    print(f"    lower clamp (z<3): {n_lower_clamp}")
    print(f"    upper clamp: {n_upper_clamp}")
    print(f"    low_z_boundary_warning (z<=7): {n_low_z_boundary_warning}")
    print(f"  float32 전체 디스크: {f32_total_gb:.2f} GB")
    print(f"  float16 전체 디스크: {f16_total_gb:.2f} GB")
    print(f"  compressed npz: {compressed_gb_low:.2f} ~ {compressed_gb_high:.2f} GB")
    print(f"  오류 수: {len(errors)}")
    print("\n  최종 추천: hybrid_cache")
    print("  출력 root (real 실행 시 생성):")
    print(f"    {OUTPUT_ROOT}")
    print("\n  [DRY-RUN 완료 — 파일 생성 없음]")
    sys.exit(0)

# =============================================================================
# real 실행: output root 생성 및 결과 파일 저장
# =============================================================================
if OUTPUT_ROOT.exists():
    print(f"\n[ABORT] output root 이미 존재: {OUTPUT_ROOT}")
    print("  기존 결과를 덮어쓰지 않습니다. 폴더를 삭제 후 재실행하세요.")
    sys.exit(1)

print(f"\n[REAL] output root 생성: {OUTPUT_ROOT}")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

t_start = time.time()

# ─── rd_b4_manifest_io_audit.csv ─────────────────────────────────────────────
print("  rd_b4_manifest_io_audit.csv 생성 ...")
audit_path = OUTPUT_ROOT / "rd_b4_manifest_io_audit.csv"
with open(audit_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["safe_id", "n_rows", "z_min", "z_max_vol", "z_max_manifest",
                     "ct_exists", "roi_exists", "n_lower_clamp", "n_upper_clamp",
                     "n_low_z_warning"])
    for sid in sorted(safe_ids):
        rows_for_sid = [r for r in manifest_rows if r["safe_id"] == sid]
        z_list = [int(r["local_z"]) for r in rows_for_sid]
        z_max_vol = patient_z_max.get(sid, -1)
        ct_ok = sid in patient_info and Path(patient_info[sid]["ct_hu_npy"]).exists()
        roi_ok = sid in patient_info and Path(patient_info[sid]["roi_0_0_npy"]).exists()
        n_lc = sum(1 for z in z_list if z < MIP_RADIUS)
        n_uc = sum(1 for z in z_list if z_max_vol > 0 and z > z_max_vol - 1 - MIP_RADIUS)
        n_lzw = sum(1 for z in z_list if z <= 7)
        writer.writerow([sid, len(rows_for_sid), min(z_list), z_max_vol, max(z_list),
                         ct_ok, roi_ok, n_lc, n_uc, n_lzw])

# ─── rd_b4_crop_strategy_comparison.csv ─────────────────────────────────────
print("  rd_b4_crop_strategy_comparison.csv 생성 ...")
comp_path = OUTPUT_ROOT / "rd_b4_crop_strategy_comparison.csv"
with open(comp_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(strategy_comparison[0].keys()))
    writer.writeheader()
    writer.writerows(strategy_comparison)

# ─── rd_b4_disk_memory_estimate.csv ─────────────────────────────────────────
print("  rd_b4_disk_memory_estimate.csv 생성 ...")
disk_path = OUTPUT_ROOT / "rd_b4_disk_memory_estimate.csv"
disk_rows = [
    {"item": "crop_elements", "value": CROP_ELEMENTS, "unit": "elements"},
    {"item": "float32_per_crop_bytes", "value": FLOAT32_BYTES_PER_CROP, "unit": "bytes"},
    {"item": "float32_per_crop_kb", "value": round(FLOAT32_BYTES_PER_CROP / 1024, 2), "unit": "KB"},
    {"item": "float16_per_crop_bytes", "value": FLOAT16_BYTES_PER_CROP, "unit": "bytes"},
    {"item": "n_crops_total", "value": n_crops, "unit": "crops"},
    {"item": "float32_total_gb", "value": round(f32_total_gb, 3), "unit": "GB"},
    {"item": "float16_total_gb", "value": round(f16_total_gb, 3), "unit": "GB"},
    {"item": "compressed_npz_low_gb", "value": round(compressed_gb_low, 3), "unit": "GB"},
    {"item": "compressed_npz_high_gb", "value": round(compressed_gb_high, 3), "unit": "GB"},
    {"item": "per_crop_file_count", "value": per_crop_files, "unit": "files"},
    {"item": "shard_file_count_1000_per_shard", "value": n_shards, "unit": "files"},
    {"item": "ct_total_disk_gb", "value": round(total_ct_gb, 2), "unit": "GB"},
    {"item": "roi_total_disk_gb", "value": round(total_roi_gb, 2), "unit": "GB"},
    {"item": "lru_cache_per_worker_mb", "value": total_cache_mb, "unit": "MB"},
    {"item": "est_epoch_io_sec", "value": round(est_io_per_epoch_sec, 1), "unit": "seconds"},
    {"item": "est_epoch_crop_gen_sec", "value": round(est_crop_gen_per_epoch_sec, 1), "unit": "seconds"},
    {"item": "steps_per_epoch_bs24", "value": steps_per_epoch_bs24, "unit": "steps"},
    {"item": "steps_per_epoch_bs48", "value": steps_per_epoch_bs48, "unit": "steps"},
]
with open(disk_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["item", "value", "unit"])
    writer.writeheader()
    writer.writerows(disk_rows)

# ─── rd_b4_on_the_fly_loader_design.csv ─────────────────────────────────────
print("  rd_b4_on_the_fly_loader_design.csv 생성 ...")
loader_path = OUTPUT_ROOT / "rd_b4_on_the_fly_loader_design.csv"
with open(loader_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["key", "value"])
    for k, v in loader_design.items():
        writer.writerow([k, v])

# ─── rd_b4_edge_case_summary.csv ─────────────────────────────────────────────
print("  rd_b4_edge_case_summary.csv 생성 ...")
edge_path = OUTPUT_ROOT / "rd_b4_edge_case_summary.csv"
with open(edge_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["category", "count", "pct_of_total", "risk_level", "mitigation"])
    writer.writerow(["lower_clamp_z_lt_3", n_lower_clamp,
                     round(n_lower_clamp/total_rows*100, 3), "low",
                     "numpy.clip(z_slab, 0, z_max-1) → 반복 slice 사용"])
    writer.writerow(["upper_clamp_z_gt_zmax_minus_4", n_upper_clamp,
                     round(n_upper_clamp/total_rows*100, 3), "low",
                     "numpy.clip(z_slab, 0, z_max-1) → 반복 slice 사용"])
    writer.writerow(["both_clamp", n_both_clamp,
                     round(n_both_clamp/total_rows*100, 3), "low",
                     "numpy.clip 동시 적용"])
    writer.writerow(["low_z_boundary_warning_z_le_7", n_low_z_boundary_warning,
                     round(n_low_z_boundary_warning/total_rows*100, 3), "medium",
                     "diaphragm saturation: lower_boundary bin에서 MIP 균일도 모니터링 필요"])
    writer.writerow(["ct_missing", len(ct_missing),
                     round(len(ct_missing)/n_safe_ids*100, 3), "critical" if ct_missing else "none",
                     "patient_manifest 재확인 필요" if ct_missing else "없음"])
    writer.writerow(["roi_missing", len(roi_missing),
                     round(len(roi_missing)/n_safe_ids*100, 3), "critical" if roi_missing else "none",
                     "patient_manifest 재확인 필요" if roi_missing else "없음"])

# ─── rd_b4_errors.csv ────────────────────────────────────────────────────────
print("  rd_b4_errors.csv 생성 ...")
err_path = OUTPUT_ROOT / "rd_b4_errors.csv"
with open(err_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["safe_id", "error", "step"])
    for e in errors:
        writer.writerow([e.get("safe_id", ""), e.get("error", ""), e.get("step", "")])

# ─── rd_b4_crop_generation_strategy_preflight_summary.json ───────────────────
print("  rd_b4_crop_generation_strategy_preflight_summary.json 생성 ...")
elapsed = round(time.time() - t_start, 2)
summary = {
    "version": "rd_b4_v1",
    "is_dry_run": False,
    "rd_b1_manifest_rows": total_rows,
    "n_patients": n_patients,
    "n_safe_ids": n_safe_ids,
    "bin_counts": dict(bin_counts),
    "local_z_min": int(min(z_vals)),
    "local_z_max": int(max(z_vals)),
    "ct_exist": ct_exist,
    "ct_missing": len(ct_missing),
    "roi_exist": roi_exist,
    "roi_missing": len(roi_missing),
    "edge_clamp": {
        "lower_clamp_z_lt_3": n_lower_clamp,
        "upper_clamp": n_upper_clamp,
        "both_clamp": n_both_clamp,
        "total_clamp": n_total_clamp,
        "pct_total_clamp": round(pct_clamp, 3),
        "low_z_boundary_warning_z_le_7": n_low_z_boundary_warning,
    },
    "disk_estimate": {
        "crop_float32_kb_each": round(FLOAT32_BYTES_PER_CROP / 1024, 2),
        "float32_total_gb": round(f32_total_gb, 3),
        "float16_total_gb": round(f16_total_gb, 3),
        "compressed_npz_gb_low": round(compressed_gb_low, 3),
        "compressed_npz_gb_high": round(compressed_gb_high, 3),
        "per_crop_files": per_crop_files,
        "shard_files_1000_per_shard": n_shards,
    },
    "loader_estimate": {
        "est_epoch_io_sec": round(est_io_per_epoch_sec, 1),
        "est_epoch_crop_gen_sec": round(est_crop_gen_per_epoch_sec, 1),
        "steps_per_epoch_bs24": steps_per_epoch_bs24,
        "steps_per_epoch_bs48": steps_per_epoch_bs48,
        "lru_cache_per_worker_mb": total_cache_mb,
    },
    "recommended_strategy": "hybrid_cache",
    "n_errors": len(errors),
    "elapsed_seconds": elapsed,
    "absolute_not_done": [
        "crop NPZ 대량 생성 없음",
        "학습 없음",
        "scoring 없음",
        "model forward 없음",
        "stage2_holdout 접근 없음",
        "기존 파일 수정 없음",
        "GPU 사용 없음",
        "checkpoint 로드 없음",
        "threshold 재계산 없음",
    ],
    "next_steps": [
        "RD-B5: Dataset/loader/model skeleton static check",
        "RD-B6: tiny smoke train (2 patients, 5 epochs, no eval)",
    ],
}
with open(OUTPUT_ROOT / "rd_b4_crop_generation_strategy_preflight_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ─── rd_b4_crop_generation_strategy_preflight_report.md ─────────────────────
print("  rd_b4_crop_generation_strategy_preflight_report.md 생성 ...")
report_lines = [
    "# RD-B4 Crop Generation Strategy Preflight Report",
    "",
    "## 1. 이전 단계 요약",
    "",
    "| 단계 | 결과 |",
    "|------|------|",
    f"| RD-B1 | 6-bin balanced manifest 86,017 rows / 290 patients / cap 50/bin/patient |",
    f"| RD-B2b | mixed_3ch ADOPT: ch1=CT center, ch2=lower 3mm MIP, ch3=upper 3mm MIP |",
    f"| RD-B2b norm | HU clip [-1000, 600] → (x+1000)/1600 → [0,1] |",
    f"| RD-B3 | true RD4AD teacher-student / ResNet18 ImageNet frozen / layer1/layer2/layer3 |",
    "",
    "## 2. Manifest IO Audit",
    "",
    f"- 총 rows: {total_rows:,}",
    f"- unique patient_id: {n_patients}",
    f"- unique safe_id: {n_safe_ids}",
    f"- local_z 범위: {min(z_vals)} ~ {max(z_vals)}",
    f"- CT 파일 존재: {ct_exist}/{n_safe_ids} (누락: {len(ct_missing)})",
    f"- ROI 파일 존재: {roi_exist}/{n_safe_ids} (누락: {len(roi_missing)})",
    "",
    "### bin별 row 수",
    "",
    "| bin | rows |",
    "|-----|------|",
]
for k, v in sorted(bin_counts.items()):
    report_lines.append(f"| {k} | {v:,} |")

report_lines += [
    "",
    "## 3. mixed_3ch Crop 정의",
    "",
    "```",
    "crop_size = 96×96",
    "ch1 = CT center at local_z",
    "ch2 = lower 3mm MIP: z-3 ~ z-1 (z_spacing=1.0mm)",
    "ch3 = upper 3mm MIP: z+1 ~ z+3 (z_spacing=1.0mm)",
    "경계 z: numpy.clip(z_slab_indices, 0, z_max-1) → 반복 slice 사용",
    "```",
    "",
    "## 4. Normalization 정의",
    "",
    "```",
    "HU clip [-1000, 600]",
    "normalized = (hu_clipped + 1000) / 1600  → [0, 1] float32",
    "```",
    "",
    "## 5. Edge Clamp / Low-Z Boundary Risk",
    "",
    f"| 항목 | 수 | 비율 | 위험 |",
    f"|------|-----|------|------|",
    f"| lower clamp (z<3) | {n_lower_clamp} | {n_lower_clamp/total_rows*100:.3f}% | low |",
    f"| upper clamp (z>z_max-4) | {n_upper_clamp} | {n_upper_clamp/total_rows*100:.3f}% | low |",
    f"| both clamp | {n_both_clamp} | {n_both_clamp/total_rows*100:.3f}% | low |",
    f"| 총 clamp | {n_total_clamp} | {pct_clamp:.3f}% | low |",
    f"| low_z_boundary_warning (z≤7) | {n_low_z_boundary_warning} | {n_low_z_boundary_warning/total_rows*100:.3f}% | medium |",
    "",
    "- clamp 전략: `numpy.clip(slab_indices, 0, z_max-1)` — 경계 slice 반복 사용",
    "- low_z_boundary_warning: diaphragm saturation risk → lower_boundary bin에서 MIP 균일도 모니터링 필요 (RD-B6 smoke에서 확인)",
    "",
    "## 6. 디스크 사용량 추정",
    "",
    f"| 방식 | 예상 크기 |",
    f"|------|-----------|",
    f"| float32 전체 ({n_crops:,}개) | {f32_total_gb:.2f} GB |",
    f"| float16 전체 | {f16_total_gb:.2f} GB |",
    f"| compressed npz | {compressed_gb_low:.2f} ~ {compressed_gb_high:.2f} GB |",
    f"| per-crop 파일 수 | {per_crop_files:,}개 (파일 관리 부담) |",
    f"| shard 방식 (1000/shard) | {n_shards}개 (권장) |",
    "",
    "## 7. 학습 DataLoader 설계 (on-the-fly 기준)",
    "",
    f"- Dataset class: `RD4ADCropDataset`",
    f"- manifest CSV → (safe_id, local_z, crop_y0, crop_x0, six_bin_label) 읽기",
    f"- CT: `np.load(ct_path, mmap_mode='r')` → patient-level LRU cache (per-worker, LRU={lru_cache_per_worker})",
    f"- MIP slab: `ct[np.clip(z_slab, 0, z_max-1)]` → max projection",
    f"- batch_sampler: `SixBinBalancedBatchSampler` (6-bin 균형 보장)",
    f"- batch_size 후보: {BATCH_SIZE_CANDIDATE_A} (bin당 4) / {BATCH_SIZE_CANDIDATE_B} (bin당 8)",
    f"- n_workers 후보: {N_WORKERS_CANDIDATE}",
    f"- duplicate oversampling: 금지",
    f"- patient leakage: split=train 환자만, safe_id 기준 분리",
    f"- ImageNet normalization: RD-B5/RD-B6 ablation으로 보류",
    "",
    "### 1 epoch 추정 (pure CPU, GPU transfer 별도)",
    "",
    f"| 항목 | 추정 |",
    f"|------|------|",
    f"| I/O (290명 CT mmap) | {est_io_per_epoch_sec:.1f}s |",
    f"| crop 생성 (86,017개) | {est_crop_gen_per_epoch_sec:.1f}s |",
    f"| steps/epoch (bs=24) | {steps_per_epoch_bs24} |",
    f"| steps/epoch (bs=48) | {steps_per_epoch_bs48} |",
    "",
    "## 8. 방식 비교",
    "",
    "| 방식 | 디스크 | IO 위험 | 유연성 | 재현성 | 복잡도 | 추천 |",
    "|------|--------|---------|--------|--------|--------|------|",
    f"| on_the_fly | 0 GB | medium | high | medium | medium | - |",
    f"| pregenerated_npz | {f32_total_gb:.1f} GB | low | low | high | low | - |",
    f"| hybrid_cache | ~0 GB | low | high | high | medium | ★ 추천 |",
    "",
    "## 9. 최종 추천: hybrid_cache",
    "",
    "**추천 근거:**",
    "- 현재 단계는 RD4AD 첫 smoke train → normalization/MIP slab 변경 가능성 있음",
    "- 전체 NPZ 생성은 9.3 GB + 86,017개 파일 → 정착 전 과다 비용",
    "- patient-level LRU cache로 I/O 위험 완화 가능",
    "- smoke subset NPZ(1~2 환자, ~200개 crop)만 먼저 생성해 DataLoader 검증 후",
    "  전체 학습 때 on-the-fly로 전환 또는 정착 후 전체 shard NPZ 생성 선택",
    "",
    "**hybrid 단계:**",
    "1. 기본 DataLoader: on-the-fly + LRU cache",
    "2. smoke test: patient 2명 subset NPZ 생성 → DataLoader 정합성 확인",
    "3. RD-B6 smoke train 이후: normalization/MIP 확정되면 전체 shard NPZ 선택 가능",
    "",
    "## 10. 다음 단계",
    "",
    "| 단계 | 내용 |",
    "|------|------|",
    f"| RD-B5 | Dataset/loader/model skeleton static check (GPU 금지, forward 금지) |",
    f"| RD-B6 | tiny smoke train (2 patients, 5 epochs, no eval, CPU) |",
    "",
    "## 11. 이번 단계에서 절대 하지 않은 것",
    "",
    "- crop NPZ 대량 생성 없음",
    "- 학습 없음",
    "- scoring 없음",
    "- model forward 없음",
    "- stage2_holdout 접근 없음",
    "- 기존 파일 수정 없음",
    "- GPU 사용 없음",
    "- checkpoint 로드 없음",
    "- threshold 재계산 없음",
    "",
    f"*생성 시각: RD-B4 preflight elapsed {elapsed}s*",
]
with open(OUTPUT_ROOT / "rd_b4_crop_generation_strategy_preflight_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# ─── DONE ────────────────────────────────────────────────────────────────────
(OUTPUT_ROOT / "DONE").write_text("rd_b4_crop_generation_strategy_preflight_v1 완료\n")

elapsed = round(time.time() - t_start, 2)
print(f"\n[완료] 경과 시간: {elapsed}s")
print(f"  출력 위치: {OUTPUT_ROOT}")
print(f"  오류 수: {len(errors)}")
print(f"  최종 추천: hybrid_cache")
print(f"  예상 NPZ 디스크: {f32_total_gb:.2f} GB (float32) / {f16_total_gb:.2f} GB (float16)")
print(f"  edge clamp 대상: {n_total_clamp} ({pct_clamp:.2f}%)")
print(f"  low_z_boundary_warning: {n_low_z_boundary_warning}")
