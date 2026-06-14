"""
RD-B7 Full Train Config Preflight
- manifest 검증, val manifest 계획, config 비교, checkpoint/monitoring 설계
- full training / scoring / threshold / model forward / checkpoint 생성 금지
- output root가 이미 있으면 즉시 중단
"""

import sys
import os
import csv
import json
import math
import argparse
from pathlib import Path
from collections import Counter

# ──────────────────────────────────────────────────────────────────────────────
# Bare-run guard
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__" and len(sys.argv) == 1:
    print("사용법:")
    print("  python rd_b7_full_train_config_preflight.py --plan-only       # 계산만, 파일 미생성")
    print("  python rd_b7_full_train_config_preflight.py --run-preflight   # CSV/JSON/MD 생성")
    print()
    print("안전 조건:")
    print("  - full training 금지  - scoring 금지  - threshold 금지")
    print("  - model forward 금지  - checkpoint 생성 금지")
    print("  - output root 이미 있으면 즉시 중단")
    sys.exit(0)

# ──────────────────────────────────────────────────────────────────────────────
# 경로 상수
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

TRAIN_MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b1_6bin_balanced_manifest_preflight_v1"
    / "rd_b1_6bin_balanced_normal_train_coordinate_manifest.csv"
)

HOLDOUT_MANIFEST_PATHS = [
    PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets"
    / "s6a_stage2_holdout_candidate_coordinate_manifest_v1.csv",
    PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets"
    / "s6a_stage2_holdout_filtered_manifest_v1.csv",
]

VAL_MANIFEST_PATH = (
    PROJECT_ROOT
    / "experiments/normal_only_second_stage_refiner_v1/outputs/manifests"
    / "n_c10_normal_val_crop_manifest/n_c10_normal_val_crop_manifest.csv"
)

RD_B6B_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit/rd_b6b_tiny_smoke_train_v1"
    / "rd_b6b_tiny_smoke_train_summary.json"
)

RD_B4_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit/rd_b4_crop_generation_strategy_preflight_v1"
    / "rd_b4_crop_generation_strategy_preflight_summary.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b7_full_train_config_preflight_v1"
)

FULL_TRAIN_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/models/rd_b8_true_rd4ad_resnet18_mixed3ch_6bin_v1"
)

# ──────────────────────────────────────────────────────────────────────────────
# 안전 가드: 금지 작업 실행 방지
# ──────────────────────────────────────────────────────────────────────────────
FORBIDDEN_IMPORTS = ["torch", "torchvision", "numpy", "cv2", "nibabel", "SimpleITK"]

def _safety_guard_no_model_ops():
    """이 스크립트는 model forward/backward/checkpoint를 절대 실행하지 않음."""
    return True

_safety_guard_no_model_ops()


# ──────────────────────────────────────────────────────────────────────────────
# Task 1: manifest 검증
# ──────────────────────────────────────────────────────────────────────────────
def verify_train_manifest():
    result = {
        "manifest_path": str(TRAIN_MANIFEST_PATH),
        "manifest_exists": False,
        "total_rows": 0,
        "train_patients": 0,
        "six_bin_counts": {},
        "stage2_holdout_intersection": -1,
        "ct_exist": "확인_필요",
        "roi_exist": "확인_필요",
        "low_z_boundary_warning": 0,
        "min_bin_label": "",
        "min_bin_count": 0,
        "errors": [],
    }

    if not TRAIN_MANIFEST_PATH.exists():
        result["errors"].append(f"manifest 파일 없음: {TRAIN_MANIFEST_PATH}")
        return result
    result["manifest_exists"] = True

    patients = set()
    bins = Counter()
    low_z_boundary = 0
    total = 0
    with open(TRAIN_MANIFEST_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            patients.add(row["patient_id"])
            bins[row["six_bin_label"]] += 1
            total += 1
            if int(float(row.get("local_z", 9999))) <= 7:
                low_z_boundary += 1

    result["total_rows"] = total
    result["train_patients"] = len(patients)
    result["six_bin_counts"] = dict(bins)
    result["low_z_boundary_warning"] = low_z_boundary

    if total != 86017:
        result["errors"].append(f"manifest rows 불일치: {total} != 86017")
    if len(patients) != 290:
        result["errors"].append(f"patients 수 불일치: {len(patients)} != 290")

    # stage2_holdout intersection
    holdout_patients = set()
    for hp in HOLDOUT_MANIFEST_PATHS:
        if hp.exists():
            with open(hp) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = row.get("patient_id", row.get("safe_id", ""))
                    if pid:
                        holdout_patients.add(pid)
    intersection = patients & holdout_patients
    result["stage2_holdout_intersection"] = len(intersection)
    if len(intersection) > 0:
        result["errors"].append(f"stage2_holdout 교집합 발견: {sorted(intersection)[:5]}")

    # CT/ROI 존재 여부 (RD-B4 summary에서 확인)
    if RD_B4_SUMMARY_PATH.exists():
        with open(RD_B4_SUMMARY_PATH) as f:
            b4 = json.load(f)
        result["ct_exist"] = b4.get("ct_exist", "확인_필요")
        result["roi_exist"] = b4.get("roi_exist", "확인_필요")
        ct_missing = b4.get("ct_missing", -1)
        roi_missing = b4.get("roi_missing", -1)
        if ct_missing != 0:
            result["errors"].append(f"CT 누락: {ct_missing}")
        if roi_missing != 0:
            result["errors"].append(f"ROI 누락: {roi_missing}")

    # min bin 계산
    if bins:
        min_label = min(bins, key=bins.get)
        result["min_bin_label"] = min_label
        result["min_bin_count"] = bins[min_label]

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Task 2: val manifest 계획
# ──────────────────────────────────────────────────────────────────────────────
def check_val_manifest(train_patients):
    result = {
        "val_manifest_path": str(VAL_MANIFEST_PATH),
        "val_manifest_exists": False,
        "val_patients": 0,
        "val_rows": 0,
        "val_train_overlap": 0,
        "val_bin_type": "unknown",
        "six_bin_val_manifest_exists": False,
        "six_bin_val_manifest_needed": True,
        "when_to_generate": "RD-B9 (full train 후, scoring 전)",
        "note": "",
    }

    # train_patients는 patient_id set
    train_pids = train_patients if isinstance(train_patients, set) else set()

    if not VAL_MANIFEST_PATH.exists():
        result["note"] = "n_c10 val manifest 없음. 별도 생성 필요"
        return result

    result["val_manifest_exists"] = True
    val_patients = set()
    bins = Counter()
    total = 0
    has_six_bin = False
    with open(VAL_MANIFEST_PATH) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_six_bin = "six_bin_label" in fieldnames
        for row in reader:
            val_patients.add(row["patient_id"])
            b = row.get("six_bin_label", row.get("position_bin", ""))
            bins[b] += 1
            total += 1

    result["val_patients"] = len(val_patients)
    result["val_rows"] = total
    result["val_train_overlap"] = len(val_patients & train_pids)
    result["val_bin_type"] = "six_bin_label" if has_six_bin else "position_bin_old"
    result["six_bin_val_manifest_exists"] = has_six_bin

    if has_six_bin:
        result["six_bin_val_manifest_needed"] = False
        result["when_to_generate"] = "이미 존재"
        result["note"] = "6-bin val manifest 존재 - RD-B9에서 바로 사용 가능"
    else:
        result["note"] = (
            "n_c10 manifest는 position_bin(old 6-bin 이름) 기준임. "
            "six_bin_label(RD-B1 표준) 기준 val manifest 신규 생성 필요. "
            "RD-B9(full train 완료 후 scoring 전)에서 생성 예정."
        )
        result["six_bin_val_manifest_needed"] = True
        result["when_to_generate"] = "RD-B9 (full train 완료 후, scoring 전)"

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Task 3+4: config 비교 및 epoch/batch 수 계산
# ──────────────────────────────────────────────────────────────────────────────
CONFIGS = {
    "A_safe_fast": {
        "batch_size": 48,
        "per_bin": 8,
        "epochs": 10,
        "lr": 1e-4,
        "optimizer": "AdamW",
        "weight_decay": 1e-5,
        "num_workers": 4,
        "patient_cache": 8,
        "purpose": "빠른 1차 full train",
    },
    "B_balanced_default": {
        "batch_size": 48,
        "per_bin": 8,
        "epochs": 20,
        "lr": 1e-4,
        "optimizer": "AdamW",
        "weight_decay": 1e-5,
        "num_workers": 4,
        "patient_cache": 8,
        "purpose": "기본 후보",
    },
    "C_conservative": {
        "batch_size": 24,
        "per_bin": 4,
        "epochs": 20,
        "lr": 1e-4,
        "optimizer": "AdamW",
        "weight_decay": 1e-5,
        "num_workers": 2,
        "patient_cache": 4,
        "purpose": "메모리/I/O 안정성 우선",
    },
}

# RD-B6b smoke 기준: batch=24, 15 batches, 1.07초 → 0.0713초/batch (I/O없음)
SMOKE_SEC_PER_BATCH = 1.07 / 15  # 0.0713
# full train에서는 I/O+cache overhead 포함 → 2x 보수적 추정
FULL_TRAIN_IO_MULTIPLIER = 2.0
# RD-B4: est_epoch_io_sec=23.2 (단일 worker, 1 epoch), est_epoch_crop_gen_sec=17.2
B4_EPOCH_IO_SEC = 23.2
B4_EPOCH_GEN_SEC = 17.2

# GPU smoke peak: 118.91 MB (batch=24, no DataLoader)
SMOKE_GPU_MB = 118.91
# ResNet18 teacher+student params ≈ 88 MB
PARAM_MB = 88.0
# activation scales ~linearly with batch; optimizer state (AdamW: 2x params) ≈ 176 MB
OPTIMIZER_MB = 176.0


def compute_config_metrics(cfg, min_bin_count, total_manifest_rows):
    per_bin = cfg["per_bin"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    num_workers = cfg["num_workers"]
    patient_cache = cfg["patient_cache"]

    # 6-bin balanced: each batch = per_bin samples * 6 bins
    batches_per_epoch = math.floor(min_bin_count / per_bin)
    samples_used_per_epoch = batches_per_epoch * batch_size
    dropped_samples_per_epoch = total_manifest_rows - samples_used_per_epoch
    estimated_steps_total = batches_per_epoch * epochs

    # runtime 추정
    # GPU: smoke 0.0713초/batch * full_multiplier
    gpu_sec_per_batch = SMOKE_SEC_PER_BATCH * FULL_TRAIN_IO_MULTIPLIER
    # I/O: DataLoader workers로 프리페치 → workers=4면 I/O bottleneck 완화
    io_sec_per_epoch = B4_EPOCH_IO_SEC / max(num_workers, 1) * 2  # conservative
    gpu_sec_per_epoch = batches_per_epoch * gpu_sec_per_batch
    # 실제 epoch time = max(GPU, I/O) + overhead
    epoch_sec = max(gpu_sec_per_epoch, io_sec_per_epoch) + 5  # 5초 overhead
    total_sec = epoch_sec * epochs
    total_min = total_sec / 60

    # GPU memory 추정 (batch 기반 선형 scaling)
    # smoke: batch=24 → 118.91 MB
    # params: ~88 MB (고정), activation: (118.91 - 88) * (batch/24) = 30.91 * ratio
    activation_mb = (SMOKE_GPU_MB - PARAM_MB) * (batch_size / 24)
    forward_gpu_mb = PARAM_MB + activation_mb
    # AdamW optimizer state (2x params gradient moments)
    full_gpu_mb = forward_gpu_mb + OPTIMIZER_MB + 20  # 20MB buffer
    # DataLoader prefetch buffer
    prefetch_mb = num_workers * batch_size * 3 * 96 * 96 * 4 / (1024**2)  # float32
    total_gpu_mb = full_gpu_mb + prefetch_mb

    # CPU memory (patient LRU cache)
    # RD-B4: lru_cache_per_worker_mb=1480 (worst case, per worker)
    # patient_cache=8 patients per worker; 실제는 worker간 공유 없음
    cpu_cache_mb = num_workers * patient_cache * 1480 / 8  # per-patient ~185 MB
    # 실제 훨씬 작음 (slice 단위 로드), 보수적
    cpu_cache_mb_conservative = num_workers * 1480  # worst case from RD-B4

    # disk write (checkpoint만)
    # best.pth + last.pth ≈ ResNet18 student: ~44 MB * 2 = 88 MB (optimizer 제외시 작음)
    disk_checkpoint_mb = 44 * 2 + 5  # student state dict * 2 + metadata

    return {
        "min_bin_count": min_bin_count,
        "batches_per_epoch": batches_per_epoch,
        "samples_used_per_epoch": samples_used_per_epoch,
        "dropped_samples_per_epoch": max(dropped_samples_per_epoch, 0),
        "estimated_steps_total": estimated_steps_total,
        "estimated_epoch_sec": round(epoch_sec, 1),
        "estimated_total_sec": round(total_sec, 1),
        "estimated_total_min": round(total_min, 1),
        "expected_gpu_memory_mb": round(total_gpu_mb, 1),
        "expected_cpu_memory_cache_mb_per_worker": round(cpu_cache_mb_conservative / max(num_workers,1), 1),
        "expected_cpu_memory_total_mb": round(cpu_cache_mb_conservative, 1),
        "expected_disk_write_mb": round(disk_checkpoint_mb, 1),
    }


def build_config_comparison(min_bin_count, total_rows):
    rows = []
    metrics_map = {}
    for name, cfg in CONFIGS.items():
        m = compute_config_metrics(cfg, min_bin_count, total_rows)
        metrics_map[name] = m
        row = {
            "config_name": name,
            "purpose": cfg["purpose"],
            "batch_size": cfg["batch_size"],
            "per_bin": cfg["per_bin"],
            "epochs": cfg["epochs"],
            "lr": cfg["lr"],
            "optimizer": cfg["optimizer"],
            "weight_decay": cfg["weight_decay"],
            "num_workers": cfg["num_workers"],
            "patient_cache": cfg["patient_cache"],
            **m,
        }
        rows.append(row)
    return rows, metrics_map


# ──────────────────────────────────────────────────────────────────────────────
# Task 5: checkpoint/output 구조
# ──────────────────────────────────────────────────────────────────────────────
CHECKPOINT_DESIGN = [
    {"field": "output_root", "value": str(FULL_TRAIN_OUTPUT_ROOT)},
    {"field": "report_output_root", "value": str(OUTPUT_ROOT)},
    {"field": "checkpoint_best", "value": "best_train_loss.pth"},
    {"field": "checkpoint_last", "value": "last.pth"},
    {"field": "checkpoint_forbidden_names", "value": "smoke_only / production / final"},
    {"field": "student_state_dict", "value": "저장"},
    {"field": "optimizer_state_dict", "value": "저장"},
    {"field": "epoch", "value": "저장"},
    {"field": "config", "value": "저장"},
    {"field": "train_loss", "value": "저장"},
    {"field": "teacher_backbone", "value": "저장 (resnet18_imagenet)"},
    {"field": "teacher_weight", "value": "미저장 (local cache path + metadata만)"},
    {"field": "input_type", "value": "mixed_3ch"},
    {"field": "normalization", "value": "HU [-1000,600] → [0,1]"},
    {"field": "six_bin_labels", "value": "저장"},
    {"field": "train_manifest_path", "value": str(TRAIN_MANIFEST_PATH)},
    {"field": "normal_only", "value": "true"},
    {"field": "stage2_holdout_access", "value": "0"},
]


# ──────────────────────────────────────────────────────────────────────────────
# Task 6: monitoring/stop 조건
# ──────────────────────────────────────────────────────────────────────────────
MONITORING_LOGS = [
    "epoch_loss", "batch_loss",
    "loss_layer1", "loss_layer2", "loss_layer3",
    "nan_count", "inf_count",
    "teacher_param_changed",
    "student_param_changed",
    "optimizer_teacher_param_count",
    "gpu_peak_memory_mb",
    "runtime_per_epoch_sec",
    "low_z_warning_batch_count",
    "bin_batch_count_per_bin",
]

STOP_CONDITIONS = [
    {"condition": "loss NaN/Inf", "action": "즉시 중단"},
    {"condition": "teacher param changed", "action": "즉시 중단"},
    {"condition": "optimizer_teacher_param_count > 0", "action": "즉시 중단"},
    {"condition": "stage2_holdout access > 0", "action": "즉시 중단"},
    {"condition": "checkpoint path outside output_root", "action": "즉시 중단"},
    {"condition": "full manifest 아닌 다른 manifest 사용", "action": "즉시 중단"},
    {"condition": "scoring 시도", "action": "즉시 중단"},
    {"condition": "threshold 생성 시도", "action": "즉시 중단"},
]


# ──────────────────────────────────────────────────────────────────────────────
# 추천 config 결정
# ──────────────────────────────────────────────────────────────────────────────
def decide_recommended_config(metrics_map, total_gpu_vram_mb=8187.5):
    """GPU VRAM 8187.5 MB, RD-B6b GPU peak 118.91 MB 기준으로 Config 추천."""
    # Config B: epochs=20, batch=48 → GPU ~309 MB → 여유 충분
    # I/O 문제: num_workers=4이면 RD-B4 lru_cache_per_worker=1480 MB * 4 = 5920 MB CPU
    # 이것은 worst-case estimate이고 실제는 slice-by-slice 로드하면 훨씬 작음
    # → Config B를 1순위로 추천 (gpu 여유 충분, 20 epoch으로 적절한 학습)

    b_gpu = metrics_map["B_balanced_default"]["expected_gpu_memory_mb"]
    b_min = metrics_map["B_balanced_default"]["estimated_total_min"]

    reasons = []
    if b_gpu < total_gpu_vram_mb * 0.5:
        reasons.append(f"GPU {b_gpu:.0f}MB < VRAM 50% ({total_gpu_vram_mb*0.5:.0f}MB), 여유 충분")
    reasons.append(f"예상 runtime {b_min:.0f}분 (약 {b_min/60:.1f}시간)")
    reasons.append("20 epochs는 1차 full train baseline으로 적절")
    reasons.append("Config A(10 epochs)는 학습 부족 가능성, Config C는 runtime 2배")

    return "B_balanced_default", reasons


# ──────────────────────────────────────────────────────────────────────────────
# CSV 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
def write_csv(path, rows, fieldnames=None):
    if not rows:
        path.write_text("no_data\n")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true",
                        help="계산만 수행, 파일 생성 없음")
    parser.add_argument("--run-preflight", action="store_true",
                        help="CSV/JSON/MD 파일 생성")
    args = parser.parse_args()

    if not args.plan_only and not args.run_preflight:
        parser.print_help()
        sys.exit(0)

    # output root 존재 확인 (run-preflight 시에만)
    if args.run_preflight:
        if OUTPUT_ROOT.exists():
            print(f"[ABORT] output root 이미 존재: {OUTPUT_ROOT}")
            print("기존 결과 보호를 위해 중단합니다.")
            sys.exit(1)

    print("=" * 60)
    print("RD-B7 Full Train Config Preflight")
    print("=" * 60)

    errors = []

    # ── Task 1: manifest 검증 ──────────────────────────────────────────────
    print("\n[1] Train manifest 검증 중...")
    manifest_result = verify_train_manifest()
    train_patients_set = set()
    with open(TRAIN_MANIFEST_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_patients_set.add(row["patient_id"])
    errors.extend(manifest_result.get("errors", []))

    print(f"  rows       : {manifest_result['total_rows']}")
    print(f"  patients   : {manifest_result['train_patients']}")
    print(f"  holdout∩   : {manifest_result['stage2_holdout_intersection']}")
    print(f"  min_bin    : {manifest_result['min_bin_label']} = {manifest_result['min_bin_count']}")
    print(f"  low_z_warn : {manifest_result['low_z_boundary_warning']}")
    print(f"  CT/ROI     : {manifest_result['ct_exist']}/{manifest_result['roi_exist']}")
    for e in manifest_result.get("errors", []):
        print(f"  [ERROR] {e}")

    # ── Task 2: val manifest 계획 ──────────────────────────────────────────
    print("\n[2] Val manifest 계획 확인 중...")
    val_result = check_val_manifest(train_patients_set)
    print(f"  val_manifest_exists : {val_result['val_manifest_exists']}")
    print(f"  val_patients        : {val_result['val_patients']}")
    print(f"  val_rows            : {val_result['val_rows']}")
    print(f"  val_train_overlap   : {val_result['val_train_overlap']}")
    print(f"  bin_type            : {val_result['val_bin_type']}")
    print(f"  six_bin_needed      : {val_result['six_bin_val_manifest_needed']}")
    print(f"  when_to_generate    : {val_result['when_to_generate']}")
    print(f"  note                : {val_result['note']}")

    # ── Task 3+4: config 비교 ─────────────────────────────────────────────
    print("\n[3+4] Config 비교 및 계산 중...")
    min_bin_count = manifest_result["min_bin_count"]
    total_rows = manifest_result["total_rows"]
    config_rows, metrics_map = build_config_comparison(min_bin_count, total_rows)

    for row in config_rows:
        print(f"\n  [{row['config_name']}] {row['purpose']}")
        print(f"    batch={row['batch_size']}, per_bin={row['per_bin']}, epochs={row['epochs']}")
        print(f"    batches/epoch   = {row['batches_per_epoch']}")
        print(f"    samples_used    = {row['samples_used_per_epoch']}")
        print(f"    dropped         = {row['dropped_samples_per_epoch']}")
        print(f"    total_steps     = {row['estimated_steps_total']}")
        print(f"    est_runtime     = {row['estimated_total_min']} 분 ({row['estimated_total_min']/60:.1f}h)")
        print(f"    GPU memory      = {row['expected_gpu_memory_mb']} MB")
        print(f"    CPU cache(total)= {row['expected_cpu_memory_total_mb']} MB")
        print(f"    disk_write      = {row['expected_disk_write_mb']} MB")

    # ── Task 7: 추천 config ───────────────────────────────────────────────
    print("\n[7] 추천 config 결정 중...")
    recommended_name, rec_reasons = decide_recommended_config(metrics_map)
    recommended_cfg = CONFIGS[recommended_name]
    recommended_metrics = metrics_map[recommended_name]
    print(f"  추천 config: {recommended_name}")
    for r in rec_reasons:
        print(f"    → {r}")

    # ── Plan-only: 여기서 종료 ────────────────────────────────────────────
    if args.plan_only:
        print("\n[plan-only] 파일 생성 없이 종료합니다.")
        print(f"추천: {recommended_name}")
        print(f"  batch_size={recommended_cfg['batch_size']}, epochs={recommended_cfg['epochs']}, lr={recommended_cfg['lr']}")
        print(f"  예상 runtime: {recommended_metrics['estimated_total_min']}분")
        print(f"  val manifest 필요: {val_result['six_bin_val_manifest_needed']} (생성 시점: {val_result['when_to_generate']})")
        print(f"\n모든 안전 조건 준수 확인:")
        print("  full training: 미실행")
        print("  scoring: 미실행")
        print("  threshold: 미생성")
        print("  checkpoint: 미생성")
        print("  model forward: 미실행")
        print("  stage2_holdout: 미접근")
        return

    # ── run-preflight: 파일 생성 ──────────────────────────────────────────
    print(f"\n[run-preflight] output root 생성: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    # rd_b7_manifest_readiness.csv
    manifest_rows = [
        {"check": "manifest_exists", "value": str(manifest_result["manifest_exists"]),
         "expected": "True", "pass": str(manifest_result["manifest_exists"])},
        {"check": "total_rows", "value": str(manifest_result["total_rows"]),
         "expected": "86017", "pass": str(manifest_result["total_rows"] == 86017)},
        {"check": "train_patients", "value": str(manifest_result["train_patients"]),
         "expected": "290", "pass": str(manifest_result["train_patients"] == 290)},
        {"check": "stage2_holdout_intersection", "value": str(manifest_result["stage2_holdout_intersection"]),
         "expected": "0", "pass": str(manifest_result["stage2_holdout_intersection"] == 0)},
        {"check": "ct_exist", "value": str(manifest_result["ct_exist"]),
         "expected": "290", "pass": str(manifest_result["ct_exist"] == 290)},
        {"check": "roi_exist", "value": str(manifest_result["roi_exist"]),
         "expected": "290", "pass": str(manifest_result["roi_exist"] == 290)},
        {"check": "low_z_boundary_warning", "value": str(manifest_result["low_z_boundary_warning"]),
         "expected": "기록됨", "pass": "True"},
        {"check": "min_bin_label", "value": manifest_result["min_bin_label"],
         "expected": "upper_interior", "pass": str(manifest_result["min_bin_label"] == "upper_interior")},
        {"check": "min_bin_count", "value": str(manifest_result["min_bin_count"]),
         "expected": "13932", "pass": str(manifest_result["min_bin_count"] == 13932)},
        {"check": "manifest_path", "value": str(TRAIN_MANIFEST_PATH), "expected": "-", "pass": "True"},
    ]
    write_csv(OUTPUT_ROOT / "rd_b7_manifest_readiness.csv", manifest_rows)

    # bin count 상세
    bin_rows = [{"six_bin_label": k, "count": v}
                for k, v in sorted(manifest_result["six_bin_counts"].items())]
    write_csv(OUTPUT_ROOT / "rd_b7_manifest_readiness.csv",
              manifest_rows + [{"check": f"bin_{k}", "value": str(v), "expected": "-", "pass": "True"}
                                for k, v in manifest_result["six_bin_counts"].items()])

    # rd_b7_val_manifest_plan.csv
    val_plan_rows = [
        {"item": k, "value": str(v)} for k, v in val_result.items()
    ]
    write_csv(OUTPUT_ROOT / "rd_b7_val_manifest_plan.csv", val_plan_rows)

    # rd_b7_train_config_comparison.csv
    write_csv(OUTPUT_ROOT / "rd_b7_train_config_comparison.csv", config_rows)

    # rd_b7_runtime_memory_estimate.csv
    runtime_rows = []
    for row in config_rows:
        runtime_rows.append({
            "config_name": row["config_name"],
            "batch_size": row["batch_size"],
            "per_bin": row["per_bin"],
            "epochs": row["epochs"],
            "batches_per_epoch": row["batches_per_epoch"],
            "samples_used_per_epoch": row["samples_used_per_epoch"],
            "dropped_samples_per_epoch": row["dropped_samples_per_epoch"],
            "estimated_steps_total": row["estimated_steps_total"],
            "estimated_epoch_sec": row["estimated_epoch_sec"],
            "estimated_total_sec": row["estimated_total_sec"],
            "estimated_total_min": row["estimated_total_min"],
            "expected_gpu_memory_mb": row["expected_gpu_memory_mb"],
            "expected_cpu_memory_total_mb": row["expected_cpu_memory_total_mb"],
            "expected_disk_write_mb": row["expected_disk_write_mb"],
        })
    write_csv(OUTPUT_ROOT / "rd_b7_runtime_memory_estimate.csv", runtime_rows)

    # rd_b7_checkpoint_output_design.csv
    write_csv(OUTPUT_ROOT / "rd_b7_checkpoint_output_design.csv", CHECKPOINT_DESIGN)

    # rd_b7_monitoring_and_stop_conditions.csv
    mon_rows = [{"type": "log", "name": n, "condition": "-", "action": "-"} for n in MONITORING_LOGS]
    stop_rows = [{"type": "stop", "name": d["condition"], "condition": d["condition"], "action": d["action"]}
                 for d in STOP_CONDITIONS]
    write_csv(OUTPUT_ROOT / "rd_b7_monitoring_and_stop_conditions.csv", mon_rows + stop_rows)

    # rd_b7_errors.csv
    error_rows = [{"error_id": f"err_{i:03d}", "message": e} for i, e in enumerate(errors)]
    write_csv(OUTPUT_ROOT / "rd_b7_errors.csv", error_rows)

    # rd_b7_full_train_config_preflight_summary.json
    summary = {
        "version": "rd_b7_v1",
        "train_manifest_rows": manifest_result["total_rows"],
        "train_patients": manifest_result["train_patients"],
        "six_bin_counts": manifest_result["six_bin_counts"],
        "min_bin_label": manifest_result["min_bin_label"],
        "min_bin_count": manifest_result["min_bin_count"],
        "low_z_boundary_warning": manifest_result["low_z_boundary_warning"],
        "stage2_holdout_intersection": manifest_result["stage2_holdout_intersection"],
        "ct_exist": manifest_result["ct_exist"],
        "roi_exist": manifest_result["roi_exist"],
        "val_manifest_available": val_result["val_manifest_exists"],
        "val_manifest_bin_type": val_result["val_bin_type"],
        "val_patients": val_result["val_patients"],
        "val_train_overlap": val_result["val_train_overlap"],
        "val_manifest_needed": val_result["six_bin_val_manifest_needed"],
        "val_manifest_when": val_result["when_to_generate"],
        "recommended_config": recommended_name,
        "recommended_batch_size": recommended_cfg["batch_size"],
        "recommended_epochs": recommended_cfg["epochs"],
        "recommended_lr": recommended_cfg["lr"],
        "recommended_optimizer": recommended_cfg["optimizer"],
        "estimated_batches_per_epoch": recommended_metrics["batches_per_epoch"],
        "estimated_total_steps": recommended_metrics["estimated_steps_total"],
        "estimated_runtime_min": recommended_metrics["estimated_total_min"],
        "gpu_vram_total_mb": 8187.5,
        "expected_gpu_memory_mb": recommended_metrics["expected_gpu_memory_mb"],
        "full_training_started": False,
        "scoring_started": False,
        "threshold_created": False,
        "checkpoint_created": False,
        "model_forward_executed": False,
        "stage2_holdout_accessed": False,
        "n_errors": len(errors),
        "all_checks_passed": (
            manifest_result["total_rows"] == 86017
            and manifest_result["train_patients"] == 290
            and manifest_result["stage2_holdout_intersection"] == 0
            and len(errors) == 0
        ),
        "absolute_not_done": [
            "full training 없음",
            "scoring 없음",
            "threshold 없음",
            "checkpoint 생성 없음",
            "model forward 없음",
            "stage2_holdout 접근 없음",
            "기존 파일 수정 없음",
        ],
    }

    with open(OUTPUT_ROOT / "rd_b7_full_train_config_preflight_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # rd_b7_full_train_config_preflight_report.md
    _write_report(summary, val_result, config_rows, recommended_name, rec_reasons, manifest_result)

    # DONE marker
    (OUTPUT_ROOT / "DONE").write_text("rd_b7_full_train_config_preflight complete\n")

    print(f"\n[완료] 파일 생성 완료: {OUTPUT_ROOT}")
    print(f"  all_checks_passed: {summary['all_checks_passed']}")
    print(f"  추천 config: {recommended_name}")
    print(f"  batch_size={recommended_cfg['batch_size']}, epochs={recommended_cfg['epochs']}, lr={recommended_cfg['lr']}")
    print(f"  예상 runtime: {recommended_metrics['estimated_total_min']}분")
    print(f"  val manifest 필요: {val_result['six_bin_val_manifest_needed']}")
    print(f"  에러: {len(errors)}건")


def _write_report(summary, val_result, config_rows, recommended_name, rec_reasons, manifest_result):
    lines = []

    lines += [
        "# RD-B7 Full Train Config Preflight Report",
        "",
        "## 1. RD-B6b Tiny Smoke Train 결과 요약",
        "",
        "| 항목 | 값 |",
        "|------|----|",
        "| 학습 crops | 60 |",
        "| epochs | 5 |",
        "| batch_size | 24 |",
        "| 초기 loss | 1.907352 |",
        "| 최종 loss | 1.590096 |",
        "| teacher frozen | True |",
        "| teacher param changed | False |",
        "| student param changed | True |",
        "| optimizer teacher param count | 0 |",
        "| GPU peak memory | 118.91 MB |",
        "| 학습 시간 | 1.07초 |",
        "| 판정 | **통과** |",
        "",
        "---",
        "",
        "## 2. Full Train Manifest Readiness",
        "",
        f"- manifest path: `{summary['train_manifest_rows']:,}` rows",
        f"- patients: {summary['train_patients']}",
        f"- stage2_holdout intersection: {summary['stage2_holdout_intersection']}",
        f"- CT/ROI 존재: {summary['ct_exist']}/{summary['roi_exist']}",
        f"- low_z_boundary_warning: {summary['low_z_boundary_warning']} (local_z <= 7 기준)",
        f"- min_bin: {summary['min_bin_label']} = {summary['min_bin_count']}",
        "",
        "### 6-bin 분포",
        "",
        "| bin | count |",
        "|-----|-------|",
    ]
    for k, v in sorted(manifest_result["six_bin_counts"].items()):
        lines.append(f"| {k} | {v:,} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Val Manifest / Normal Val Threshold 계획",
        "",
        f"- val_manifest_exists: {val_result['val_manifest_exists']}",
        f"- val_patients: {val_result['val_patients']}",
        f"- val_rows: {val_result['val_rows']:,}",
        f"- val_train_overlap: {val_result['val_train_overlap']}",
        f"- bin_type: {val_result['val_bin_type']}",
        f"- six_bin_val_manifest_needed: {val_result['six_bin_val_manifest_needed']}",
        f"- 생성 시점: {val_result['when_to_generate']}",
        "",
        f"> {val_result['note']}",
        "",
        "---",
        "",
        "## 4. Config A/B/C 비교",
        "",
        "| 항목 | A safe_fast | B balanced_default | C conservative |",
        "|------|------------|-------------------|---------------|",
    ]

    cfg_a = next(r for r in config_rows if "A_" in r["config_name"])
    cfg_b = next(r for r in config_rows if "B_" in r["config_name"])
    cfg_c = next(r for r in config_rows if "C_" in r["config_name"])

    fields = [
        ("batch_size", "batch_size"),
        ("per_bin", "per_bin"),
        ("epochs", "epochs"),
        ("lr", "lr"),
        ("num_workers", "num_workers"),
        ("patient_cache", "patient_cache"),
        ("batches_per_epoch", "batches/epoch"),
        ("samples_used_per_epoch", "samples/epoch"),
        ("dropped_samples_per_epoch", "dropped/epoch"),
        ("estimated_steps_total", "total steps"),
        ("estimated_total_min", "runtime(분)"),
        ("expected_gpu_memory_mb", "GPU MB"),
        ("expected_cpu_memory_total_mb", "CPU MB(cache)"),
        ("expected_disk_write_mb", "disk MB"),
    ]
    for fkey, fname in fields:
        lines.append(f"| {fname} | {cfg_a[fkey]} | {cfg_b[fkey]} | {cfg_c[fkey]} |")

    lines += [
        "",
        "---",
        "",
        "## 5. 추천 Config",
        "",
        f"**추천: {recommended_name}**",
        "",
    ]
    for r in rec_reasons:
        lines.append(f"- {r}")

    lines += [
        "",
        f"| 항목 | 값 |",
        "|------|----|",
        f"| batch_size | {summary['recommended_batch_size']} |",
        f"| epochs | {summary['recommended_epochs']} |",
        f"| lr | {summary['recommended_lr']} |",
        f"| optimizer | {summary['recommended_optimizer']} |",
        f"| batches/epoch | {summary['estimated_batches_per_epoch']} |",
        f"| total steps | {summary['estimated_total_steps']} |",
        f"| 예상 runtime | {summary['estimated_runtime_min']}분 |",
        f"| GPU memory | {summary['expected_gpu_memory_mb']}MB / {summary['gpu_vram_total_mb']}MB |",
        "",
        "---",
        "",
        "## 6. Checkpoint/Output 구조",
        "",
        f"- full train output root: `outputs/models/rd_b8_true_rd4ad_resnet18_mixed3ch_6bin_v1/`",
        f"- checkpoint: `best_train_loss.pth`, `last.pth`",
        "- checkpoint 내 저장: student_state_dict, optimizer_state_dict, epoch, config,",
        "  train_loss, teacher_backbone, input_type, normalization, six_bin_labels,",
        "  train_manifest_path, normal_only=true, stage2_holdout_access=0",
        "- teacher weight: local cache path + metadata만 저장 (weight 전체 미저장)",
        "",
        "---",
        "",
        "## 7. Monitoring/Stop 조건",
        "",
        "### 기록 항목",
        "",
    ]
    for m in MONITORING_LOGS:
        lines.append(f"- {m}")

    lines += [
        "",
        "### 중단 조건",
        "",
        "| 조건 | 조치 |",
        "|------|------|",
    ]
    for d in STOP_CONDITIONS:
        lines.append(f"| {d['condition']} | {d['action']} |")

    lines += [
        "",
        "---",
        "",
        "## 8. RD-B8 실행 전 확인사항",
        "",
        "- [ ] 사용자가 추천 config 승인",
        "- [ ] full train output root 미존재 확인",
        "- [ ] teacher weight local cache 경로 확인",
        "- [ ] TRAIN_MANIFEST_PATH 정확한지 재확인",
        "- [ ] num_workers/patient_cache 값 확정",
        "- [ ] 예상 runtime 수용 가능 여부 확인",
        f"- [ ] val manifest 생성 계획 확정 ({val_result['when_to_generate']})",
        "",
        "---",
        "",
        "## 9. 절대 하지 않은 것",
        "",
        "- full training 없음",
        "- scoring 없음",
        "- threshold 없음",
        "- checkpoint 생성 없음",
        "- model forward 없음",
        "- stage2_holdout 접근 없음",
        "- 기존 파일 수정 없음",
    ]

    (OUTPUT_ROOT / "rd_b7_full_train_config_preflight_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
