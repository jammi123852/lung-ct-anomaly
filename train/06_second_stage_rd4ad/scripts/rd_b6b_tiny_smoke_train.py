"""
RD-B6b: Tiny Smoke Train
목적: 60 crop smoke subset으로 teacher-student (RD4AD) 학습이 가능한지 확인
     full training / scoring / threshold / stage2_holdout 접근 금지
모드:
  bare run     -> exit 2 (파일 생성 금지)
  --dry-plan   -> 계획 출력만 (파일 생성 없음)
  --run-smoke  -> 실제 학습 실행 (사용자 승인 후)
안전 조건:
  stage2_holdout/lesion 경로 접근 금지
  full training/scoring/threshold 금지
  output root 이미 존재 시 즉시 중단
  checkpoint는 smoke-only 경로에만 저장
  best.pth / final.pth 이름 금지
"""

import sys
import csv
import json
import math
import time
import random
from pathlib import Path

# ── bare-run guard ────────────────────────────────────────────────────────────
ALLOWED_MODES = {"--dry-plan", "--run-smoke"}
if not any(m in sys.argv for m in ALLOWED_MODES):
    print("오류: 실행 모드 지정 필요.")
    print("  --dry-plan  : 학습 계획 출력 (파일 생성 없음)")
    print("  --run-smoke : tiny smoke train 실행 (사용자 승인 후)")
    sys.exit(2)

IS_DRY_PLAN  = "--dry-plan"   in sys.argv
IS_RUN_SMOKE = "--run-smoke"  in sys.argv

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path("/home/jinhy/project/lung-ct-anomaly")
OUTPUT_ROOT   = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b6b_tiny_smoke_train_v1"
)
CHECKPOINT_DIR  = OUTPUT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "rd_b6b_tiny_smoke_epoch5_smoke_only.pth"

SMOKE_MANIFEST = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b6a_tiny_smoke_train_preflight_v1"
    / "rd_b6a_smoke_subset_manifest.csv"
)
PATIENT_MANIFEST_PATH = Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/"
    "manifests/patient_manifest.csv"
)
LOCAL_WEIGHT_PATH = Path(
    "/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)

# ── 금지 키워드 ────────────────────────────────────────────────────────────────
FORBIDDEN_KEYWORDS = [
    "stage2_holdout",
    "lesion",
    "test_lesion",
    "second-stage-lesion-refiner",
]
FORBIDDEN_CHECKPOINT_NAMES = {"best.pth", "final.pth"}

# ── 설계 상수 ──────────────────────────────────────────────────────────────────
CROP_SIZE  = 96
N_CHANNELS = 3
MIP_RADIUS = 3
HU_CLIP_MIN = -1000.0
HU_CLIP_MAX  =  600.0
HU_RANGE     = 1600.0
SIX_BIN_LABELS = [
    "lower_boundary", "lower_interior",
    "middle_boundary", "middle_interior",
    "upper_boundary",  "upper_interior",
]
LOW_Z_BOUNDARY_WARN_THRESHOLD = 7

# ── 학습 하이퍼파라미터 ───────────────────────────────────────────────────────
BATCH_SIZE       = 24
N_EPOCHS         = 5
LR               = 1e-4
SEED             = 42
N_TRAIN_CROPS    = 60
TEACHER_BACKBONE = "resnet18"
INPUT_TYPE       = "mixed_3ch"
NORMALIZATION    = "HU[-1000,600]->[0,1]"


# =============================================================================
# 안전 검사
# =============================================================================

def assert_path_safe(path_str: str) -> None:
    for kw in FORBIDDEN_KEYWORDS:
        if kw.lower() in str(path_str).lower():
            raise RuntimeError(
                f"[SAFETY] 금지 경로 접근 차단: {path_str!r} (keyword={kw!r})"
            )


def assert_checkpoint_name_safe(path: Path) -> None:
    if path.name in FORBIDDEN_CHECKPOINT_NAMES:
        raise RuntimeError(
            f"[SAFETY] 금지 checkpoint 이름: {path.name!r}"
        )


# =============================================================================
# 공통 함수
# =============================================================================

def normalize_hu(hu_array):
    import numpy as np
    clipped = np.clip(hu_array, HU_CLIP_MIN, HU_CLIP_MAX)
    return ((clipped - HU_CLIP_MIN) / HU_RANGE).astype("float32")


def compute_mip_slab_indices(center_z: int, direction: str, z_max: int) -> list:
    if direction == "lower":
        raw = [center_z - MIP_RADIUS + i for i in range(MIP_RADIUS)]
    elif direction == "upper":
        raw = [center_z + 1 + i for i in range(MIP_RADIUS)]
    else:
        raise ValueError(f"direction={direction!r}")
    return [max(0, min(idx, z_max - 1)) for idx in raw]


def has_low_z_boundary_warning(center_z: int) -> bool:
    return center_z <= LOW_Z_BOUNDARY_WARN_THRESHOLD


def load_smoke_manifest(manifest_path: Path) -> list:
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_patient_paths(patient_manifest_path: Path, target_safe_ids: set) -> dict:
    patient_paths = {}
    with open(patient_manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("safe_id", "")
            if sid in target_safe_ids:
                ct_path = row.get("ct_hu_npy", "")
                assert_path_safe(ct_path)
                patient_paths[sid] = {"ct_hu_npy": ct_path}
    return patient_paths


def build_crop_np(ct_arr, center_z: int,
                  crop_y0: int, crop_x0: int,
                  crop_y1: int, crop_x1: int):
    import numpy as np
    z_max = ct_arr.shape[0]
    ch0_raw = ct_arr[center_z, crop_y0:crop_y1, crop_x0:crop_x1].copy()
    lower_idxs = compute_mip_slab_indices(center_z, "lower", z_max)
    upper_idxs = compute_mip_slab_indices(center_z, "upper", z_max)
    ch1_raw = ct_arr[lower_idxs].max(axis=0)[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    ch2_raw = ct_arr[upper_idxs].max(axis=0)[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    return np.stack([
        normalize_hu(ch0_raw),
        normalize_hu(ch1_raw),
        normalize_hu(ch2_raw),
    ], axis=0)  # (3, 96, 96) float32


# =============================================================================
# Teacher / Student 빌드
# =============================================================================

def build_teacher(local_weight_path: Path):
    import torch
    import torchvision.models as models
    resnet = models.resnet18(weights=None)
    state_dict = torch.load(
        str(local_weight_path), map_location="cpu", weights_only=True
    )
    resnet.load_state_dict(state_dict)
    resnet.eval()
    resnet.requires_grad_(False)
    return resnet


def build_student_decoder():
    import torch.nn as nn

    class StudentDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            # de_layer3: 256x6x6 -> 256x6x6
            self.de_layer3 = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            # de_layer2: 256x6x6 -> 128x12x12
            self.de_layer2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            # de_layer1: 128x12x12 -> 64x24x24
            self.de_layer1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        def forward(self, layer3_feat):
            x   = self.de_layer3(layer3_feat)  # (B,256,6,6)
            de3 = x
            x   = self.de_layer2(x)            # (B,128,12,12)
            de2 = x
            x   = self.de_layer1(x)            # (B,64,24,24)
            de1 = x
            return de3, de2, de1

    return StudentDecoder()


# =============================================================================
# 6-bin balanced sampler
# =============================================================================

class SixBinBalancedSampler:
    def __init__(self, crop_rows: list, rng: random.Random):
        self.bins = {lbl: [] for lbl in SIX_BIN_LABELS}
        for i, row in enumerate(crop_rows):
            lbl = row.get("six_bin_label", "")
            if lbl in self.bins:
                self.bins[lbl].append(i)
        self.rng = rng

    def epoch_indices(self) -> list:
        """6-bin round-robin balanced 순서로 전체 indices 반환."""
        shuffled = {}
        for lbl, idxs in self.bins.items():
            idxs_copy = list(idxs)
            self.rng.shuffle(idxs_copy)
            shuffled[lbl] = idxs_copy
        max_len = max((len(v) for v in shuffled.values()), default=0)
        result = []
        for i in range(max_len):
            for lbl in SIX_BIN_LABELS:
                if i < len(shuffled.get(lbl, [])):
                    result.append(shuffled[lbl][i])
        return result


# =============================================================================
# CSV / report 헬퍼
# =============================================================================

def write_csv(path: Path, fieldnames: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  -> {path.name}")


# =============================================================================
# dry-plan
# =============================================================================

def run_dry_plan() -> None:
    print("\n[DRY-PLAN] RD-B6b tiny smoke train 계획:")
    print(f"  smoke manifest : {SMOKE_MANIFEST}")
    print(f"  output root    : {OUTPUT_ROOT}")
    print(f"  checkpoint     : {CHECKPOINT_PATH}")
    print(f"  local weight   : {LOCAL_WEIGHT_PATH} (exists={LOCAL_WEIGHT_PATH.exists()})")
    print()

    if not SMOKE_MANIFEST.exists():
        print(f"  [ERROR] smoke manifest 없음: {SMOKE_MANIFEST}")
        return

    rows = load_smoke_manifest(SMOKE_MANIFEST)
    bins = {lbl: [] for lbl in SIX_BIN_LABELS}
    for row in rows:
        lbl = row.get("six_bin_label", "")
        if lbl in bins:
            bins[lbl].append(row)

    print(f"  smoke subset rows: {len(rows)} (예상: {N_TRAIN_CROPS})")
    print()
    print(f"  {'six_bin_label':>22} | count")
    print("  " + "-" * 32)
    for lbl in SIX_BIN_LABELS:
        print(f"  {lbl:>22} | {len(bins[lbl])}")

    total_batches_per_epoch = math.ceil(len(rows) / BATCH_SIZE)
    total_steps = total_batches_per_epoch * N_EPOCHS
    print()
    print(f"  학습 설정:")
    print(f"    batch_size        = {BATCH_SIZE}")
    print(f"    n_epochs          = {N_EPOCHS}")
    print(f"    lr                = {LR}")
    print(f"    batches/epoch     = {total_batches_per_epoch}")
    print(f"    total steps       = {total_steps}")
    print(f"    sampler           = 6-bin balanced round-robin")

    print()
    print(f"  checkpoint 이름 안전 검사:")
    for fn in sorted(FORBIDDEN_CHECKPOINT_NAMES):
        match = CHECKPOINT_PATH.name == fn
        print(f"    {fn:20s} -> {'DANGER ❌' if match else 'OK ✓'}")
    print(f"    actual name: {CHECKPOINT_PATH.name} -> OK ✓")

    print()
    print(f"  output root 존재: {OUTPUT_ROOT.exists()}")
    if OUTPUT_ROOT.exists():
        print(f"  [ABORT 조건] output root가 이미 존재합니다 -> --run-smoke 실행 불가.")
    else:
        print(f"  [OK] output root 없음 -> --run-smoke 실행 가능 (사용자 승인 필요)")

    print("\n[DRY-PLAN 완료] 파일 생성 없음.")


# =============================================================================
# smoke train
# =============================================================================

def run_smoke_train() -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import datetime

    # output root guard
    if OUTPUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재 -> 즉시 중단: {OUTPUT_ROOT}")
        sys.exit(1)

    # checkpoint name safety
    assert_checkpoint_name_safe(CHECKPOINT_PATH)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  output root 생성: {OUTPUT_ROOT}")

    # seed 고정
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # smoke manifest 로드 (60 rows only)
    smoke_rows = load_smoke_manifest(SMOKE_MANIFEST)
    if len(smoke_rows) != N_TRAIN_CROPS:
        print(f"[FAIL] 예상 {N_TRAIN_CROPS} crops != 실제 {len(smoke_rows)}")
        sys.exit(1)
    print(f"  smoke manifest rows: {len(smoke_rows)}")

    # patient CT 경로 로드
    target_sids = set(r["safe_id"] for r in smoke_rows)
    patient_paths = load_patient_paths(PATIENT_MANIFEST_PATH, target_sids)
    print(f"  patient paths loaded: {len(patient_paths)}")

    # 60 crops 미리 numpy로 생성 (~6.6MB)
    print("  60 crops on-the-fly 생성 중 ...")
    ct_cache: dict = {}
    all_crops: list = []
    for row in smoke_rows:
        sid = row["safe_id"]
        lz  = int(row["local_z"])
        y0, x0 = int(row["crop_y0"]), int(row["crop_x0"])
        y1, x1 = int(row["crop_y1"]), int(row["crop_x1"])

        if sid not in ct_cache:
            ct_path = patient_paths[sid]["ct_hu_npy"]
            assert_path_safe(ct_path)
            ct_cache[sid] = np.load(ct_path, mmap_mode="r")

        crop_np = build_crop_np(ct_cache[sid], lz, y0, x0, y1, x1)
        all_crops.append(crop_np)

    ct_cache.clear()
    print(f"  crops 생성 완료: {len(all_crops)}개")

    # teacher 빌드 (frozen + eval)
    print("  teacher 빌드 (local weight) ...")
    teacher = build_teacher(LOCAL_WEIGHT_PATH).to(device)
    teacher_features: dict = {}

    def make_hook(name: str):
        def _hook(module, inp, output):
            teacher_features[name] = output
        return _hook

    teacher.layer1.register_forward_hook(make_hook("layer1"))
    teacher.layer2.register_forward_hook(make_hook("layer2"))
    teacher.layer3.register_forward_hook(make_hook("layer3"))
    print("  teacher: eval+frozen, hook 등록 완료")

    # student 빌드 (random init + train)
    print("  student decoder 빌드 (random init) ...")
    student = build_student_decoder().to(device)
    student.train()

    # teacher parameter 초기 snapshot
    teacher_param_sum_before = sum(p.sum().item() for p in teacher.parameters())

    # student parameter 초기 snapshot
    student_param_sum_before = sum(p.sum().item() for p in student.parameters())

    # optimizer (student parameters only)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR)

    # optimizer가 teacher parameter를 포함하지 않는지 검증
    teacher_param_ids = set(id(p) for p in teacher.parameters())
    optimizer_teacher_count = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) in teacher_param_ids:
                optimizer_teacher_count += 1

    if optimizer_teacher_count > 0:
        print(f"[FAIL] optimizer에 teacher parameter {optimizer_teacher_count}개 포함 -> 학습 중단")
        sys.exit(1)
    print(f"  optimizer_teacher_param_count: {optimizer_teacher_count} (OK)")

    # 6-bin balanced sampler
    rng = random.Random(SEED)
    sampler = SixBinBalancedSampler(smoke_rows, rng)

    # 학습 루프
    epoch_log_rows:      list = []
    batch_loss_log_rows: list = []
    errors_list:         list = []
    global_step   = 0
    loss_nan_count = 0
    loss_inf_count = 0

    print(f"\n  학습 시작: {N_EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR}")
    train_start = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        epoch_start  = time.time()
        epoch_idxs   = sampler.epoch_indices()
        epoch_losses = []

        for batch_start in range(0, len(epoch_idxs), BATCH_SIZE):
            batch_idxs   = epoch_idxs[batch_start:batch_start + BATCH_SIZE]
            batch_actual = len(batch_idxs)

            # batch tensor
            batch_np = np.stack([all_crops[i] for i in batch_idxs], axis=0)
            batch_t  = torch.from_numpy(batch_np).to(device)

            # teacher forward (no_grad)
            teacher_features.clear()
            with torch.no_grad():
                _ = teacher(batch_t)

            t_l1 = teacher_features["layer1"].detach()
            t_l2 = teacher_features["layer2"].detach()
            t_l3 = teacher_features["layer3"].detach()

            # student forward
            de3, de2, de1 = student(t_l3)

            # multi-scale cosine loss
            loss_l1 = 1.0 - F.cosine_similarity(t_l1, de1, dim=1).mean()
            loss_l2 = 1.0 - F.cosine_similarity(t_l2, de2, dim=1).mean()
            loss_l3 = 1.0 - F.cosine_similarity(t_l3, de3, dim=1).mean()
            total_loss = loss_l1 + loss_l2 + loss_l3

            loss_val = float(total_loss)
            is_nan   = math.isnan(loss_val)
            is_inf   = math.isinf(loss_val)

            if is_nan:
                loss_nan_count += 1
                errors_list.append({
                    "epoch": epoch, "step": global_step,
                    "error": f"NaN loss (batch_start={batch_start})"
                })
            if is_inf:
                loss_inf_count += 1
                errors_list.append({
                    "epoch": epoch, "step": global_step,
                    "error": f"Inf loss (batch_start={batch_start})"
                })

            # backward + step (finite loss일 때만)
            if not is_nan and not is_inf:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_losses.append(loss_val)

            batch_loss_log_rows.append({
                "epoch":      epoch,
                "step":       global_step,
                "batch_size": batch_actual,
                "loss_l1":    round(float(loss_l1), 6),
                "loss_l2":    round(float(loss_l2), 6),
                "loss_l3":    round(float(loss_l3), 6),
                "total_loss": round(loss_val, 6),
                "is_nan":     is_nan,
                "is_inf":     is_inf,
            })
            global_step += 1

        epoch_elapsed = time.time() - epoch_start
        epoch_mean = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        epoch_log_rows.append({
            "epoch":          epoch,
            "n_batches":      math.ceil(len(epoch_idxs) / BATCH_SIZE),
            "n_crops":        len(epoch_idxs),
            "mean_loss":      round(epoch_mean, 6),
            "epoch_time_sec": round(epoch_elapsed, 2),
        })
        print(f"  Epoch {epoch}/{N_EPOCHS}  loss={epoch_mean:.6f}  t={epoch_elapsed:.1f}s")

    train_elapsed = time.time() - train_start

    # teacher parameter 변경 여부 확인
    teacher_param_sum_after = sum(p.sum().item() for p in teacher.parameters())
    teacher_param_changed = abs(teacher_param_sum_after - teacher_param_sum_before) > 1e-9

    # student parameter 변경 여부 확인
    student_param_sum_after = sum(p.sum().item() for p in student.parameters())
    student_param_changed = abs(student_param_sum_after - student_param_sum_before) > 1e-9

    print(f"\n  teacher_param_changed: {teacher_param_changed}  (OK if False)")
    print(f"  student_param_changed: {student_param_changed}  (OK if True)")
    print(f"  optimizer_teacher_param_count: {optimizer_teacher_count}  (OK if 0)")

    # GPU peak memory
    if device.type == "cuda":
        gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        gpu_peak_mb = 0.0

    # checkpoint 저장 (smoke-only 경로에만)
    assert_checkpoint_name_safe(CHECKPOINT_PATH)
    torch.save({
        "student_state_dict":  student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":               N_EPOCHS,
        "smoke_only":          True,
        "n_train_crops":       N_TRAIN_CROPS,
        "teacher_backbone":    TEACHER_BACKBONE,
        "input_type":          INPUT_TYPE,
        "normalization":       NORMALIZATION,
    }, str(CHECKPOINT_PATH))
    print(f"  checkpoint 저장: {CHECKPOINT_PATH}")

    # epoch/batch/param CSV
    first_loss = epoch_log_rows[0]["mean_loss"]  if epoch_log_rows else float("nan")
    last_loss  = epoch_log_rows[-1]["mean_loss"] if epoch_log_rows else float("nan")
    loss_decreased = (
        (last_loss < first_loss)
        if epoch_log_rows and not math.isnan(first_loss) and not math.isnan(last_loss)
        else False
    )

    write_csv(
        OUTPUT_ROOT / "rd_b6b_epoch_log.csv",
        ["epoch", "n_batches", "n_crops", "mean_loss", "epoch_time_sec"],
        epoch_log_rows,
    )
    write_csv(
        OUTPUT_ROOT / "rd_b6b_batch_loss_log.csv",
        ["epoch", "step", "batch_size",
         "loss_l1", "loss_l2", "loss_l3", "total_loss", "is_nan", "is_inf"],
        batch_loss_log_rows,
    )
    write_csv(
        OUTPUT_ROOT / "rd_b6b_parameter_update_check.csv",
        ["item", "value"],
        [
            {"item": "teacher_param_sum_before",      "value": round(teacher_param_sum_before, 6)},
            {"item": "teacher_param_sum_after",       "value": round(teacher_param_sum_after, 6)},
            {"item": "teacher_param_changed",         "value": teacher_param_changed},
            {"item": "student_param_sum_before",      "value": round(student_param_sum_before, 6)},
            {"item": "student_param_sum_after",       "value": round(student_param_sum_after, 6)},
            {"item": "student_param_changed",         "value": student_param_changed},
            {"item": "optimizer_teacher_param_count", "value": optimizer_teacher_count},
        ],
    )
    write_csv(
        OUTPUT_ROOT / "rd_b6b_errors.csv",
        ["epoch", "step", "error"],
        errors_list,
    )

    # GPU/runtime summary
    gpu_runtime = {
        "device":             str(device),
        "gpu_peak_memory_mb": round(gpu_peak_mb, 2),
        "train_elapsed_sec":  round(train_elapsed, 2),
        "n_epochs":           N_EPOCHS,
        "n_batches_total":    global_step,
        "batch_size":         BATCH_SIZE,
        "n_train_crops":      N_TRAIN_CROPS,
    }
    with open(OUTPUT_ROOT / "rd_b6b_gpu_runtime_summary.json", "w", encoding="utf-8") as f:
        json.dump(gpu_runtime, f, ensure_ascii=False, indent=2)
    print(f"  -> rd_b6b_gpu_runtime_summary.json")

    # 실패 조건 검사
    failure_flags = [
        teacher_param_changed,
        optimizer_teacher_count > 0,
        loss_nan_count > 0,
        loss_inf_count > 0,
        not CHECKPOINT_PATH.exists(),
    ]
    all_checks_passed = not any(failure_flags)

    # main summary JSON
    summary = {
        "version":                    "rd_b6b_v1",
        "timestamp":                  ts,
        "n_train_crops":              N_TRAIN_CROPS,
        "n_epochs":                   N_EPOCHS,
        "batch_size":                 BATCH_SIZE,
        "lr":                         LR,
        "teacher_frozen":             True,
        "teacher_param_changed":      teacher_param_changed,
        "student_param_changed":      student_param_changed,
        "optimizer_teacher_param_count": optimizer_teacher_count,
        "loss_nan_count":             loss_nan_count,
        "loss_inf_count":             loss_inf_count,
        "first_epoch_loss":           round(first_loss, 6),
        "last_epoch_loss":            round(last_loss, 6),
        "loss_decreased":             loss_decreased,
        "smoke_only_checkpoint_saved": CHECKPOINT_PATH.exists(),
        "checkpoint_path":            str(CHECKPOINT_PATH),
        "training_scope":             "tiny_smoke_only",
        "full_training_started":      False,
        "scoring_started":            False,
        "threshold_created":          False,
        "stage2_holdout_access":      0,
        "gpu_peak_memory_mb":         round(gpu_peak_mb, 2),
        "train_elapsed_sec":          round(train_elapsed, 2),
        "all_checks_passed":          all_checks_passed,
        "verdict":                    "통과" if all_checks_passed else "경고",
    }
    with open(OUTPUT_ROOT / "rd_b6b_tiny_smoke_train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  -> rd_b6b_tiny_smoke_train_summary.json")

    # report.md
    _write_report_md(OUTPUT_ROOT, ts, summary, epoch_log_rows)

    if all_checks_passed:
        (OUTPUT_ROOT / "DONE").write_text(f"rd_b6b tiny-smoke-train completed: {ts}\n")

    print(f"\n판정: {summary['verdict']}")
    print(f"  loss  first={first_loss:.6f}  last={last_loss:.6f}  decreased={loss_decreased}")
    print(f"  NaN={loss_nan_count}  Inf={loss_inf_count}")
    print(f"  teacher_param_changed={teacher_param_changed}")
    print(f"  student_param_changed={student_param_changed}")
    print(f"  checkpoint={CHECKPOINT_PATH}")
    print(f"  all_checks_passed={all_checks_passed}")


# =============================================================================
# report.md
# =============================================================================

def _write_report_md(out_dir: Path, ts: str, summary: dict, epoch_log_rows: list) -> None:
    v = summary["verdict"]
    lines = [
        "# RD-B6b Tiny Smoke Train Report",
        f"- 버전: rd_b6b_v1",
        f"- 날짜: {ts}",
        f"- 판정: **{v}**",
        "",
        "---",
        "## 1. RD-B6a / RD-B6a-2 결과 요약",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        "| smoke subset | 60 crops = 2 patients x 6-bin x 5 crops |",
        "| crop loading | OK |",
        "| crop shape | (3,96,96) OK |",
        "| value range | [0,1] OK |",
        "| NaN/Inf | 없음 OK |",
        "| teacher layer1/2/3 shape | (B,64,24,24)/(B,128,12,12)/(B,256,6,6) OK |",
        "| student de_layer3/2/1 shape | (B,256,6,6)/(B,128,12,12)/(B,64,24,24) OK |",
        "| loss finite 6/6 | OK |",
        "| backward/optimizer/checkpoint | 없음 (RD-B6a-2) OK |",
        "",
        "---",
        "## 2. smoke subset 범위",
        "",
        f"- manifest: rd_b6a_smoke_subset_manifest.csv",
        f"- 총 {summary['n_train_crops']} crops = 2 patients x 6-bin x 5 crops",
        "- normal-only / label 사용 없음 / positive/hard_negative 사용 없음",
        "- stage2_holdout 접근 없음",
        "",
        "---",
        "## 3. 모델 구조",
        "",
        "**Teacher (frozen)**",
        "- ResNet18 ImageNet local weight / eval mode / requires_grad=False",
        "- layer1=(B,64,24,24) / layer2=(B,128,12,12) / layer3=(B,256,6,6)",
        "",
        "**Student (reverse decoder, random init, train mode)**",
        "- de_layer3: 256x6x6 -> 256x6x6  (Conv2d+BN+ReLU)",
        "- de_layer2: 256x6x6 -> 128x12x12 (Upsample*2+Conv2d+BN+ReLU)",
        "- de_layer1: 128x12x12 -> 64x24x24 (Upsample*2+Conv2d+BN+ReLU)",
        "",
        "---",
        "## 4. loss / optimizer 설정",
        "",
        "- loss = (1-cosine_sim(t_l1,de1)) + (1-cosine_sim(t_l2,de2)) + (1-cosine_sim(t_l3,de3))",
        "- optimizer: AdamW, lr=1e-4, student parameters only",
        f"- optimizer_teacher_param_count: **{summary['optimizer_teacher_param_count']}** (0=OK)",
        f"- batch_size={summary['batch_size']} / n_epochs={summary['n_epochs']}",
        "- sampler: 6-bin balanced round-robin",
        "",
        "---",
        "## 5. epoch별 loss 변화",
        "",
        "| epoch | mean_loss | epoch_time_sec |",
        "|-------|-----------|----------------|",
    ]
    for r in epoch_log_rows:
        lines.append(f"| {r['epoch']} | {r['mean_loss']} | {r['epoch_time_sec']} |")
    lines.extend([
        "",
        f"- first_epoch_loss: **{summary['first_epoch_loss']}**",
        f"- last_epoch_loss:  **{summary['last_epoch_loss']}**",
        f"- loss_decreased:   **{summary['loss_decreased']}**",
        "",
        "---",
        "## 6. teacher frozen 검증",
        "",
        f"- teacher_param_changed: **{summary['teacher_param_changed']}** (False=OK)",
        "",
        "---",
        "## 7. student update 검증",
        "",
        f"- student_param_changed: **{summary['student_param_changed']}** (True=OK)",
        "",
        "---",
        "## 8. checkpoint smoke-only 검증",
        "",
        f"- 저장 경로: `{summary['checkpoint_path']}`",
        f"- smoke_only=True 포함",
        f"- smoke_only_checkpoint_saved: **{summary['smoke_only_checkpoint_saved']}**",
        "- best.pth / final.pth 이름 사용 안함 OK",
        "",
        "---",
        "## 9. 실패 조건 통과 여부",
        "",
        "| 실패 조건 | 결과 |",
        "|-----------|------|",
        f"| loss NaN | {summary['loss_nan_count']}건 {'OK' if summary['loss_nan_count']==0 else 'FAIL'} |",
        f"| loss Inf | {summary['loss_inf_count']}건 {'OK' if summary['loss_inf_count']==0 else 'FAIL'} |",
        f"| teacher param 변경 | {summary['teacher_param_changed']} {'OK' if not summary['teacher_param_changed'] else 'FAIL'} |",
        f"| optimizer teacher param 포함 | {summary['optimizer_teacher_param_count']}개 {'OK' if summary['optimizer_teacher_param_count']==0 else 'FAIL'} |",
        f"| checkpoint smoke 경로 밖 저장 | {'없음 OK' if summary['smoke_only_checkpoint_saved'] else 'FAIL'} |",
        f"| stage2_holdout 접근 | {summary['stage2_holdout_access']}건 OK |",
        f"| full training 시작 | {summary['full_training_started']} OK |",
        f"| scoring 시작 | {summary['scoring_started']} OK |",
        f"| threshold 생성 | {summary['threshold_created']} OK |",
        "",
        "---",
        "## 10. 다음 단계",
        "",
    ])
    if summary["all_checks_passed"]:
        lines.extend([
            "- **RD-B7**: full train config preflight",
            "  - full manifest 86,017 crops 기준 train config 설계",
            "  - batch_size / lr / epoch 선택, GPU memory 추정",
            "  - checkpoint 구조 설계",
        ])
    else:
        lines.extend([
            "- **RD-B6c**: debug",
            "  - 실패 원인 분석 후 수정",
        ])
    lines.extend([
        "",
        "---",
        "## 11. 절대 하지 않은 것",
        "",
        "| 항목 | 확인 |",
        "|------|------|",
        f"| full training | {summary['full_training_started']} -> 없음 OK |",
        f"| scoring | {summary['scoring_started']} -> 없음 OK |",
        f"| threshold 생성 | {summary['threshold_created']} -> 없음 OK |",
        f"| stage2_holdout 접근 | {summary['stage2_holdout_access']}건 -> 없음 OK |",
        "| production checkpoint (best.pth/final.pth) | 없음 OK |",
        "| 기존 파일 수정/삭제 | 없음 OK |",
    ])
    with open(out_dir / "rd_b6b_tiny_smoke_train_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> rd_b6b_tiny_smoke_train_report.md")


# =============================================================================
# main
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("RD-B6b Tiny Smoke Train")
    print("=" * 70)

    if IS_DRY_PLAN:
        run_dry_plan()
        return

    if IS_RUN_SMOKE:
        print("\n[RUN-SMOKE] tiny smoke train 실행 ...")
        run_smoke_train()
        return


if __name__ == "__main__":
    main()
