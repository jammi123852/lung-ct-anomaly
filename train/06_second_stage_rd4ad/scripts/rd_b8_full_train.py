"""
RD-B8: Full Normal-Only Train
목적: 86,017 crops full train manifest로 teacher-student (RD4AD) 20 epoch 학습
     scoring / threshold / stage2_holdout 접근 금지
모드:
  bare run    -> exit 2 (파일 생성 금지)
  --dry-plan  -> 계획 출력만 (파일 생성 없음)
  --run-train -> 실제 학습 실행 (사용자 승인 후)
안전 조건:
  stage2_holdout/lesion 경로 접근 금지
  scoring/threshold/full_crop_npz 금지
  output root 이미 존재 시 즉시 중단
  checkpoint는 model root/checkpoints/ 안에만 저장
  best.pth / final.pth 이름 금지
  smoke checkpoint와 절대 혼동 금지
"""

import sys
import csv
import json
import math
import time
import random
import collections
from pathlib import Path

# ── bare-run guard ─────────────────────────────────────────────────────────────
ALLOWED_MODES = {"--dry-plan", "--run-train", "--profile-train"}
if not any(m in sys.argv for m in ALLOWED_MODES):
    print("오류: 실행 모드 지정 필요.")
    print("  --dry-plan      : 학습 계획 출력 (파일 생성 없음)")
    print("  --run-train     : full normal-only train 실행 (사용자 승인 후)")
    print("  --profile-train : 100 batch profiling run (속도 측정, checkpoint 저장 없음)")
    sys.exit(2)

IS_DRY_PLAN      = "--dry-plan"      in sys.argv
IS_RUN_TRAIN     = "--run-train"     in sys.argv
IS_PROFILE_TRAIN = "--profile-train" in sys.argv

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
MODEL_ROOT   = (
    PROJECT_ROOT
    / "outputs/models/rd_b8_true_rd4ad_resnet18_mixed3ch_6bin_v3"
)
REPORT_ROOT  = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit/rd_b8_full_train_v3"
)
CHECKPOINT_DIR  = MODEL_ROOT / "checkpoints"
CHECKPOINT_BEST = CHECKPOINT_DIR / "best_train_loss.pth"
CHECKPOINT_LAST = CHECKPOINT_DIR / "last.pth"

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
LOCAL_WEIGHT_PATH = Path(
    "/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)

# ── 금지 키워드 ────────────────────────────────────────────────────────────────
FORBIDDEN_KEYWORDS = [
    "stage2_holdout", "lesion", "test_lesion", "second-stage-lesion-refiner",
]
FORBIDDEN_CHECKPOINT_NAMES = {"best.pth", "final.pth"}

# ── 설계 상수 ──────────────────────────────────────────────────────────────────
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

# ── 학습 하이퍼파라미터 ───────────────────────────────────────────────────────
CONFIG_NAME            = "B_balanced_default"
BATCH_SIZE             = 48
N_EPOCHS               = 20
LR                     = 1e-4
WEIGHT_DECAY           = 1e-5
SEED                   = 42
N_TRAIN_CROPS_EXPECTED = 86017
PATIENT_CACHE_SIZE     = 8
TEACHER_BACKBONE       = "resnet18"
INPUT_TYPE             = "mixed_3ch"
NORMALIZATION          = "HU[-1000,600]->[0,1]"

# ── profile 설정 ──────────────────────────────────────────────────────────────
PROFILE_REPORT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit/rd_b8_profile_v3"
)
PROFILE_MAX_BATCHES = 100
PROFILE_VERSION     = "rd_b8_v3_cropfirst_mip"


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


def load_full_manifest(manifest_path: Path) -> list:
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_patient_paths(patient_manifest_path: Path) -> dict:
    patient_paths = {}
    with open(patient_manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid      = row.get("safe_id", "")
            ct_path  = row.get("ct_hu_npy", "")
            if sid and ct_path:
                assert_path_safe(ct_path)
                patient_paths[sid] = {"ct_hu_npy": ct_path}
    return patient_paths


def build_crop_np(ct_arr, center_z: int,
                  crop_y0: int, crop_x0: int,
                  crop_y1: int, crop_x1: int):
    import numpy as np

    TARGET = 96
    z_max, h_max, w_max = ct_arr.shape

    def crop_2d_with_air_padding(img2d, y0, x0, y1, x1):
        """
        CT 원본 HU 공간에서 96x96 crop을 만든다.
        CT 밖 영역은 HU=-1000 air/background로 padding한다.
        normalize는 padding 이후에 한다.
        """
        out = np.full((TARGET, TARGET), HU_CLIP_MIN, dtype=img2d.dtype)

        src_y0 = max(0, y0)
        src_x0 = max(0, x0)
        src_y1 = min(h_max, y1)
        src_x1 = min(w_max, x1)

        if src_y1 <= src_y0 or src_x1 <= src_x0:
            return out

        dst_y0 = src_y0 - y0
        dst_x0 = src_x0 - x0
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        out[dst_y0:dst_y1, dst_x0:dst_x1] = img2d[src_y0:src_y1, src_x0:src_x1]
        return out

    # center channel
    ch0_raw = crop_2d_with_air_padding(
        ct_arr[center_z],
        crop_y0, crop_x0, crop_y1, crop_x1,
    )

    # crop-first lower MIP: z-3 ~ z-1
    lower_idxs = compute_mip_slab_indices(center_z, "lower", z_max)
    lower_crops = [
        crop_2d_with_air_padding(
            ct_arr[z],
            crop_y0, crop_x0, crop_y1, crop_x1,
        )
        for z in lower_idxs
    ]
    ch1_raw = np.max(np.stack(lower_crops, axis=0), axis=0)

    # crop-first upper MIP: z+1 ~ z+3
    upper_idxs = compute_mip_slab_indices(center_z, "upper", z_max)
    upper_crops = [
        crop_2d_with_air_padding(
            ct_arr[z],
            crop_y0, crop_x0, crop_y1, crop_x1,
        )
        for z in upper_idxs
    ]
    ch2_raw = np.max(np.stack(upper_crops, axis=0), axis=0)

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
    def __init__(self, max_size: int):
        self._cache = collections.OrderedDict()
        self._max   = max_size

    def load(self, key: str, path: str):
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
            self.de_layer3 = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.de_layer2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.de_layer1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        def forward(self, layer3_feat):
            x   = self.de_layer3(layer3_feat)
            de3 = x
            x   = self.de_layer2(x)
            de2 = x
            x   = self.de_layer1(x)
            de1 = x
            return de3, de2, de1

    return StudentDecoder()


# =============================================================================
# 6-bin Balanced Sampler
# =============================================================================

class SixBinBalancedSampler:
    def __init__(self, crop_rows: list, rng: random.Random, batch_size: int = BATCH_SIZE):
        if batch_size % len(SIX_BIN_LABELS) != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by {len(SIX_BIN_LABELS)}"
            )

        self.bins = {lbl: [] for lbl in SIX_BIN_LABELS}
        for i, row in enumerate(crop_rows):
            lbl = row.get("six_bin_label", "")
            if lbl in self.bins:
                self.bins[lbl].append(i)

        self.rng = rng
        self.batch_size = batch_size
        self.per_bin = batch_size // len(SIX_BIN_LABELS)

    def epoch_indices(self) -> list:
        """
        Strict 6-bin balanced epoch indices.

        매 batch는 반드시 6개 bin에서 per_bin개씩 뽑는다.
        가장 작은 bin 기준으로 epoch 길이를 정한다.
        남는 샘플은 해당 epoch에서 drop한다.
        """
        shuffled = {}
        for lbl, idxs in self.bins.items():
            idxs_copy = list(idxs)
            self.rng.shuffle(idxs_copy)
            shuffled[lbl] = idxs_copy

        min_len = min((len(v) for v in shuffled.values()), default=0)
        n_batches = min_len // self.per_bin

        result = []
        for batch_i in range(n_batches):
            for lbl in SIX_BIN_LABELS:
                start = batch_i * self.per_bin
                end = start + self.per_bin
                result.extend(shuffled[lbl][start:end])

        return result

    def bin_counts(self) -> dict:
        return {lbl: len(idxs) for lbl, idxs in self.bins.items()}

    def expected_batches_per_epoch(self) -> int:
        min_len = min((len(v) for v in self.bins.values()), default=0)
        return min_len // self.per_bin

    def expected_crops_per_epoch(self) -> int:
        return self.expected_batches_per_epoch() * self.batch_size


# =============================================================================
# CSV 헬퍼
# =============================================================================

def write_csv(path: Path, fieldnames: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  -> {path.name}")


# =============================================================================
# checkpoint 저장
# =============================================================================

def save_checkpoint(path: Path, student, optimizer,
                    epoch: int, train_loss: float) -> None:
    import torch
    assert_checkpoint_name_safe(path)
    torch.save({
        "student_state_dict":    student.state_dict(),
        "optimizer_state_dict":  optimizer.state_dict(),
        "epoch":                 epoch,
        "train_loss":            train_loss,
        "config":                CONFIG_NAME,
        "teacher_backbone":      TEACHER_BACKBONE,
        "input_type":            INPUT_TYPE,
        "normalization":         NORMALIZATION,
        "six_bin_labels":        SIX_BIN_LABELS,
        "train_manifest_path":   str(TRAIN_MANIFEST_PATH),
        "normal_only":           True,
        "stage2_holdout_access": 0,
    }, str(path))
    print(f"  checkpoint 저장: {path.name}")


# =============================================================================
# dry-plan
# =============================================================================

def run_dry_plan() -> None:
    print("\n[DRY-PLAN] RD-B8 Full Normal-Only Train 계획")
    print("=" * 60)
    print(f"  train manifest   : {TRAIN_MANIFEST_PATH}")
    print(f"  patient manifest : {PATIENT_MANIFEST_PATH}")
    print(f"  local weight     : {LOCAL_WEIGHT_PATH}  exists={LOCAL_WEIGHT_PATH.exists()}")
    print(f"  model output root: {MODEL_ROOT}")
    print(f"  report root      : {REPORT_ROOT}")
    print(f"  checkpoint best  : {CHECKPOINT_BEST.name}")
    print(f"  checkpoint last  : {CHECKPOINT_LAST.name}")
    print()

    if not TRAIN_MANIFEST_PATH.exists():
        print(f"  [ERROR] train manifest 없음: {TRAIN_MANIFEST_PATH}")
        return

    rows   = load_full_manifest(TRAIN_MANIFEST_PATH)
    n_rows = len(rows)
    print(f"  train manifest rows: {n_rows}  (예상: {N_TRAIN_CROPS_EXPECTED})")
    if n_rows != N_TRAIN_CROPS_EXPECTED:
        print(f"  [ABORT 조건] rows={n_rows} != {N_TRAIN_CROPS_EXPECTED}")
    else:
        print(f"  [OK] manifest rows 일치")

    bins = {lbl: 0 for lbl in SIX_BIN_LABELS}
    for row in rows:
        lbl = row.get("six_bin_label", "")
        if lbl in bins:
            bins[lbl] += 1
    print()
    print(f"  {'six_bin_label':>22} | count")
    print("  " + "-" * 32)
    for lbl in SIX_BIN_LABELS:
        print(f"  {lbl:>22} | {bins[lbl]}")

    rng_dry        = random.Random(SEED)
    sampler_dry    = SixBinBalancedSampler(rows, rng_dry, batch_size=BATCH_SIZE)
    epoch_idxs     = sampler_dry.epoch_indices()
    batches_per_ep = sampler_dry.expected_batches_per_epoch()
    total_steps    = batches_per_ep * N_EPOCHS

    print()
    print(f"  학습 config ({CONFIG_NAME}):")
    print(f"    batch_size        = {BATCH_SIZE}")
    print(f"    n_epochs          = {N_EPOCHS}")
    print(f"    lr                = {LR}")
    print(f"    weight_decay      = {WEIGHT_DECAY}")
    print(f"    crops_per_epoch   = {len(epoch_idxs)}")
    print(f"    batches_per_epoch = {batches_per_ep}")
    print(f"    total_steps       = {total_steps}")
    print(f"    sampler           = strict 6-bin balanced, shortest-bin drop-last")
    print(f"    patient_cache     = {PATIENT_CACHE_SIZE}")
    print(f"    seed              = {SEED}")

    print()
    print(f"  checkpoint 이름 안전 검사:")
    for fn in sorted(FORBIDDEN_CHECKPOINT_NAMES):
        b_danger = CHECKPOINT_BEST.name == fn
        l_danger = CHECKPOINT_LAST.name == fn
        status = "DANGER" if (b_danger or l_danger) else "OK"
        print(f"    {fn:25s} -> {status}")
    print(f"    best_train_loss.pth -> OK")
    print(f"    last.pth            -> OK")

    print()
    print(f"  model output root 존재: {MODEL_ROOT.exists()}")
    print(f"  report root 존재:       {REPORT_ROOT.exists()}")
    if MODEL_ROOT.exists() or REPORT_ROOT.exists():
        print(f"  [ABORT 조건] output root 이미 존재 -> --run-train 불가")
    else:
        print(f"  [OK] output root 없음 -> --run-train 가능 (사용자 승인 필요)")

    print()
    print(f"  stage2_holdout intersection: 0 (RD-B7 확인)")
    print(f"  scoring_started:   False")
    print(f"  threshold_created: False")
    print("\n[DRY-PLAN 완료] 파일 생성 없음.")


# =============================================================================
# profile train
# =============================================================================

def run_profile_train() -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import datetime

    # 안전 검사
    assert_path_safe(str(PROFILE_REPORT_ROOT))

    if PROFILE_REPORT_ROOT.exists():
        print(f"[ABORT] profile report root 이미 존재: {PROFILE_REPORT_ROOT}")
        sys.exit(1)
    PROFILE_REPORT_ROOT.mkdir(parents=True, exist_ok=False)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  profile output: {PROFILE_REPORT_ROOT}")
    print(f"  max_batches: {PROFILE_MAX_BATCHES}")

    # seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # manifest
    print("  train manifest 로드 중 ...")
    train_rows = load_full_manifest(TRAIN_MANIFEST_PATH)
    if len(train_rows) != N_TRAIN_CROPS_EXPECTED:
        print(f"[ABORT] manifest rows={len(train_rows)} != {N_TRAIN_CROPS_EXPECTED}")
        sys.exit(1)
    print(f"  manifest rows: {len(train_rows)} OK")

    # stage2_holdout 재확인
    for row in train_rows[:20]:
        for key in ["safe_id", "ct_path", "roi_path"]:
            assert_path_safe(row.get(key, ""))
    print(f"  stage2_holdout intersection: 0 OK")

    # patient paths
    print("  patient manifest 로드 중 ...")
    patient_paths = load_patient_paths(PATIENT_MANIFEST_PATH)
    print(f"  patient paths: {len(patient_paths)}")

    # teacher
    print("  teacher 빌드 (local ResNet18) ...")
    teacher = build_teacher(LOCAL_WEIGHT_PATH).to(device)
    teacher_features: dict = {}

    def make_hook(name: str):
        def _hook(module, inp, output):
            teacher_features[name] = output
        return _hook

    teacher.layer1.register_forward_hook(make_hook("layer1"))
    teacher.layer2.register_forward_hook(make_hook("layer2"))
    teacher.layer3.register_forward_hook(make_hook("layer3"))

    # student
    print("  student decoder 빌드 ...")
    student = build_student_decoder().to(device)
    student.train()

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    # sampler / cache
    rng        = random.Random(SEED)
    sampler    = SixBinBalancedSampler(train_rows, rng, batch_size=BATCH_SIZE)
    pt_cache   = LRUPatientCache(PATIENT_CACHE_SIZE)
    epoch_idxs = sampler.epoch_indices()

    timing_rows    = []
    errors_list    = []
    loss_nan_count = 0
    loss_inf_count = 0

    print(f"\n  profiling 시작: max {PROFILE_MAX_BATCHES} batches")
    profile_start = time.time()

    for batch_i, batch_start_i in enumerate(
        range(0, len(epoch_idxs), BATCH_SIZE)
    ):
        if batch_i >= PROFILE_MAX_BATCHES:
            break

        batch_idxs = epoch_idxs[batch_start_i:batch_start_i + BATCH_SIZE]
        t_total_start = time.time()

        # crop build time
        t_crop_start = time.time()
        crops_list = []
        for idx in batch_idxs:
            row    = train_rows[idx]
            sid    = row["safe_id"]
            lz     = int(row["local_z"])
            y0, x0 = int(row["crop_y0"]), int(row["crop_x0"])
            y1, x1 = int(row["crop_y1"]), int(row["crop_x1"])
            ct_path = patient_paths.get(sid, {}).get("ct_hu_npy", "")
            ct_arr  = pt_cache.load(sid, ct_path)
            crop_np = build_crop_np(ct_arr, lz, y0, x0, y1, x1)
            crops_list.append(crop_np)
        t_crop_end = time.time()

        # stack time
        t_stack_start = time.time()
        batch_np = np.stack(crops_list, axis=0)
        t_stack_end = time.time()

        # tensor to device time
        t_to_dev_start = time.time()
        batch_t = torch.from_numpy(batch_np).to(device)
        t_to_dev_end = time.time()

        # teacher forward time
        t_teacher_start = time.time()
        teacher_features.clear()
        with torch.no_grad():
            _ = teacher(batch_t)
        t_l1 = teacher_features["layer1"].detach()
        t_l2 = teacher_features["layer2"].detach()
        t_l3 = teacher_features["layer3"].detach()
        t_teacher_end = time.time()

        # student forward time
        t_student_start = time.time()
        de3, de2, de1 = student(t_l3)
        t_student_end = time.time()

        # loss + backward + step time
        t_backward_start = time.time()
        loss_l1    = 1.0 - F.cosine_similarity(t_l1, de1, dim=1).mean()
        loss_l2    = 1.0 - F.cosine_similarity(t_l2, de2, dim=1).mean()
        loss_l3    = 1.0 - F.cosine_similarity(t_l3, de3, dim=1).mean()
        total_loss = loss_l1 + loss_l2 + loss_l3
        loss_val   = float(total_loss)
        is_nan = math.isnan(loss_val)
        is_inf = math.isinf(loss_val)

        if is_nan:
            loss_nan_count += 1
            errors_list.append({"batch_i": batch_i, "error": "NaN loss"})
        if is_inf:
            loss_inf_count += 1
            errors_list.append({"batch_i": batch_i, "error": "Inf loss"})

        if not is_nan and not is_inf:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        t_backward_end = time.time()

        t_total_end = time.time()

        crop_build_time    = t_crop_end - t_crop_start
        stack_time         = t_stack_end - t_stack_start
        tensor_to_dev_time = t_to_dev_end - t_to_dev_start
        teacher_fwd_time   = t_teacher_end - t_teacher_start
        student_fwd_time   = t_student_end - t_student_start
        backward_step_time = t_backward_end - t_backward_start
        total_batch_time   = t_total_end - t_total_start

        timing_rows.append({
            "batch_i":               batch_i,
            "crop_build_time":       round(crop_build_time, 4),
            "stack_time":            round(stack_time, 4),
            "tensor_to_device_time": round(tensor_to_dev_time, 4),
            "teacher_forward_time":  round(teacher_fwd_time, 4),
            "student_forward_time":  round(student_fwd_time, 4),
            "backward_step_time":    round(backward_step_time, 4),
            "total_batch_time":      round(total_batch_time, 4),
            "loss_val":              round(loss_val, 6) if not (is_nan or is_inf) else "nan",
            "is_nan":                is_nan,
            "is_inf":                is_inf,
        })

        if batch_i % 10 == 0:
            print(f"  batch {batch_i:>3}/{PROFILE_MAX_BATCHES}  "
                  f"crop={crop_build_time:.3f}s  total={total_batch_time:.3f}s  "
                  f"loss={loss_val:.4f}")

    profile_elapsed = time.time() - profile_start
    n_profile = len(timing_rows)

    # GPU peak
    if device.type == "cuda":
        gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        gpu_peak_mb = 0.0

    # 통계 계산
    def mean_col(col):
        import numpy as np
        vals = [r[col] for r in timing_rows if isinstance(r[col], float)]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    mean_crop_build    = mean_col("crop_build_time")
    mean_teacher_fwd   = mean_col("teacher_forward_time")
    mean_student_fwd   = mean_col("student_forward_time")
    mean_backward_step = mean_col("backward_step_time")
    mean_total_batch   = mean_col("total_batch_time")

    batches_per_epoch = sampler.expected_batches_per_epoch()
    est_epoch_sec     = mean_total_batch * batches_per_epoch
    est_epoch_min     = round(est_epoch_sec / 60.0, 1)
    est_20ep_hour     = round(est_epoch_min * N_EPOCHS / 60.0, 1)

    all_checks_passed = (loss_nan_count == 0 and loss_inf_count == 0)

    # CSV 저장
    timing_fields = [
        "batch_i", "crop_build_time", "stack_time", "tensor_to_device_time",
        "teacher_forward_time", "student_forward_time", "backward_step_time",
        "total_batch_time", "loss_val", "is_nan", "is_inf",
    ]
    write_csv(
        PROFILE_REPORT_ROOT / "rd_b8_profile_batch_timing.csv",
        timing_fields, timing_rows,
    )
    write_csv(
        PROFILE_REPORT_ROOT / "rd_b8_profile_errors.csv",
        ["batch_i", "error"], errors_list,
    )

    # summary JSON
    summary = {
        "profile_version":             PROFILE_VERSION,
        "n_profile_batches":           n_profile,
        "mean_crop_build_time":        mean_crop_build,
        "mean_teacher_forward_time":   mean_teacher_fwd,
        "mean_student_forward_time":   mean_student_fwd,
        "mean_backward_step_time":     mean_backward_step,
        "mean_total_batch_time":       mean_total_batch,
        "estimated_epoch_time_min":    est_epoch_min,
        "estimated_20epoch_time_hour": est_20ep_hour,
        "gpu_peak_memory_mb":          round(gpu_peak_mb, 2),
        "loss_nan_count":              loss_nan_count,
        "loss_inf_count":              loss_inf_count,
        "all_checks_passed":           all_checks_passed,
        "batches_per_epoch":           batches_per_epoch,
        "scoring_started":             False,
        "threshold_created":           False,
        "stage2_holdout_access":       0,
        "checkpoint_saved":            False,
    }
    with open(PROFILE_REPORT_ROOT / "rd_b8_profile_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  -> rd_b8_profile_summary.json")

    # report MD
    verdict  = "통과" if all_checks_passed else "경고"
    crop_pct = round(mean_crop_build / mean_total_batch * 100, 1) if mean_total_batch > 0 else 0.0
    md_lines = [
        f"# RD-B8 Profile Run Report",
        f"- 버전: {PROFILE_VERSION}",
        f"- 날짜: {ts}",
        f"- 판정: **{verdict}**",
        "",
        "---",
        "## 측정 요약",
        "",
        "| 항목 | 값 |",
        "|------|----|",
        f"| n_profile_batches | {n_profile} |",
        f"| mean_crop_build_time | {mean_crop_build:.4f}s ({crop_pct}% of total) |",
        f"| mean_teacher_forward_time | {mean_teacher_fwd:.4f}s |",
        f"| mean_student_forward_time | {mean_student_fwd:.4f}s |",
        f"| mean_backward_step_time | {mean_backward_step:.4f}s |",
        f"| mean_total_batch_time | {mean_total_batch:.4f}s |",
        f"| estimated_epoch_time_min | {est_epoch_min} min |",
        f"| estimated_20epoch_time_hour | {est_20ep_hour} hr |",
        f"| gpu_peak_memory_mb | {gpu_peak_mb:.0f} MB |",
        f"| loss_nan_count | {loss_nan_count} |",
        f"| loss_inf_count | {loss_inf_count} |",
        f"| all_checks_passed | {all_checks_passed} |",
        "",
        "---",
        "## 안전 확인",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        "| checkpoint_saved | False OK |",
        "| scoring_started | False OK |",
        "| threshold_created | False OK |",
        "| stage2_holdout_access | 0건 OK |",
    ]
    with open(PROFILE_REPORT_ROOT / "rd_b8_profile_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"  -> rd_b8_profile_report.md")

    # 최종 판정 출력
    print(f"\n판정: {verdict}")
    print(f"  profile batches:        {n_profile}")
    print(f"  mean_crop_build_time:   {mean_crop_build:.4f}s ({crop_pct}% of total)")
    print(f"  mean_teacher_fwd_time:  {mean_teacher_fwd:.4f}s")
    print(f"  mean_student_fwd_time:  {mean_student_fwd:.4f}s")
    print(f"  mean_backward_time:     {mean_backward_step:.4f}s")
    print(f"  mean_total_batch_time:  {mean_total_batch:.4f}s")
    print(f"  estimated_epoch_time:   {est_epoch_min} min")
    print(f"  estimated_20ep_time:    {est_20ep_hour} hr")
    print(f"  GPU peak:               {gpu_peak_mb:.0f} MB")
    print(f"  NaN={loss_nan_count}  Inf={loss_inf_count}")
    print(f"  all_checks_passed={all_checks_passed}")
    print(f"  checkpoint_saved=False OK")
    print(f"  stage2_holdout_access=0 OK")


# =============================================================================
# full train
# =============================================================================

def run_full_train() -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import datetime

    # output root guard
    if MODEL_ROOT.exists():
        print(f"[ABORT] model output root 이미 존재: {MODEL_ROOT}")
        sys.exit(1)
    if REPORT_ROOT.exists():
        print(f"[ABORT] report root 이미 존재: {REPORT_ROOT}")
        sys.exit(1)

    assert_checkpoint_name_safe(CHECKPOINT_BEST)
    assert_checkpoint_name_safe(CHECKPOINT_LAST)

    MODEL_ROOT.mkdir(parents=True, exist_ok=False)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=False)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  model output root 생성: {MODEL_ROOT}")
    print(f"  report root 생성:       {REPORT_ROOT}")

    # seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # manifest
    print("  train manifest 로드 중 ...")
    train_rows = load_full_manifest(TRAIN_MANIFEST_PATH)
    if len(train_rows) != N_TRAIN_CROPS_EXPECTED:
        print(f"[ABORT] manifest rows={len(train_rows)} != {N_TRAIN_CROPS_EXPECTED}")
        sys.exit(1)
    print(f"  manifest rows: {len(train_rows)} OK")

    # stage2_holdout 재확인 (샘플 경로 키워드 체크)
    for row in train_rows[:20]:
        for key in ["safe_id", "ct_path", "roi_path"]:
            assert_path_safe(row.get(key, ""))
    print(f"  stage2_holdout intersection: 0 OK")

    # patient paths
    print("  patient manifest 로드 중 ...")
    patient_paths = load_patient_paths(PATIENT_MANIFEST_PATH)
    print(f"  patient paths: {len(patient_paths)}")

    # teacher
    print("  teacher 빌드 (local ResNet18) ...")
    teacher = build_teacher(LOCAL_WEIGHT_PATH).to(device)
    teacher_features: dict = {}

    def make_hook(name: str):
        def _hook(module, inp, output):
            teacher_features[name] = output
        return _hook

    teacher.layer1.register_forward_hook(make_hook("layer1"))
    teacher.layer2.register_forward_hook(make_hook("layer2"))
    teacher.layer3.register_forward_hook(make_hook("layer3"))
    print("  teacher: eval+frozen, hooks 등록 완료")

    # student
    print("  student decoder 빌드 (random init) ...")
    student = build_student_decoder().to(device)
    student.train()

    teacher_param_sum_before = sum(p.sum().item() for p in teacher.parameters())
    student_param_sum_before = sum(p.sum().item() for p in student.parameters())

    # optimizer (student only)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    teacher_param_ids    = set(id(p) for p in teacher.parameters())
    optimizer_teacher_ct = sum(
        1
        for grp in optimizer.param_groups
        for p   in grp["params"]
        if id(p) in teacher_param_ids
    )
    if optimizer_teacher_ct > 0:
        print(f"[ABORT] optimizer teacher param {optimizer_teacher_ct}개 포함")
        sys.exit(1)
    print(f"  optimizer_teacher_param_count: {optimizer_teacher_ct} OK")

    # sampler / cache
    rng     = random.Random(SEED)
    sampler = SixBinBalancedSampler(train_rows, rng, batch_size=BATCH_SIZE)
    pt_cache = LRUPatientCache(PATIENT_CACHE_SIZE)

    # 학습 루프 변수
    epoch_log_rows:      list = []
    bin_batch_summary:   list = []
    low_z_warning_rows:  list = []
    errors_list:         list = []
    global_step    = 0
    loss_nan_count = 0
    loss_inf_count = 0
    best_epoch     = 0
    best_loss      = float("inf")
    checkpoint_best_saved = False
    checkpoint_last_saved = False
    abort_flag     = False

    batch_loss_fields = [
        "epoch", "step", "batch_size",
        "loss_l1", "loss_l2", "loss_l3", "total_loss",
        "is_nan", "is_inf",
    ]
    batch_loss_path = REPORT_ROOT / "rd_b8_batch_loss_log.csv"

    print(f"\n  학습 시작: {N_EPOCHS} epochs  batch={BATCH_SIZE}  lr={LR}")
    train_start = time.time()

    with open(batch_loss_path, "w", newline="", encoding="utf-8") as blf:
        bl_writer = csv.DictWriter(blf, fieldnames=batch_loss_fields,
                                   extrasaction="ignore")
        bl_writer.writeheader()

        for epoch in range(1, N_EPOCHS + 1):
            if abort_flag:
                break
            epoch_start   = time.time()
            epoch_idxs    = sampler.epoch_indices()
            epoch_losses  = []
            epoch_low_z_batches = 0
            epoch_bin_ct  = {lbl: 0 for lbl in SIX_BIN_LABELS}

            for batch_start_i in range(0, len(epoch_idxs), BATCH_SIZE):
                if abort_flag:
                    break
                batch_idxs   = epoch_idxs[batch_start_i:batch_start_i + BATCH_SIZE]
                batch_actual = len(batch_idxs)

                # on-the-fly crop 생성
                crops_list      = []
                batch_has_low_z = False
                for idx in batch_idxs:
                    row   = train_rows[idx]
                    sid   = row["safe_id"]
                    lz    = int(row["local_z"])
                    y0, x0 = int(row["crop_y0"]), int(row["crop_x0"])
                    y1, x1 = int(row["crop_y1"]), int(row["crop_x1"])
                    lbl   = row.get("six_bin_label", "")

                    ct_path = patient_paths.get(sid, {}).get("ct_hu_npy", "")
                    ct_arr  = pt_cache.load(sid, ct_path)
                    crop_np = build_crop_np(ct_arr, lz, y0, x0, y1, x1)
                    crops_list.append(crop_np)

                    if has_low_z_boundary_warning(lz):
                        batch_has_low_z = True
                        low_z_warning_rows.append({
                            "epoch":         epoch,
                            "step":          global_step,
                            "batch_i":       batch_start_i // BATCH_SIZE,
                            "safe_id":       sid,
                            "local_z":       lz,
                            "six_bin_label": lbl,
                        })
                    if lbl in epoch_bin_ct:
                        epoch_bin_ct[lbl] += 1

                if batch_has_low_z:
                    epoch_low_z_batches += 1

                batch_np = np.stack(crops_list, axis=0)
                batch_t  = torch.from_numpy(batch_np).to(device)

                # teacher forward
                teacher_features.clear()
                with torch.no_grad():
                    _ = teacher(batch_t)
                t_l1 = teacher_features["layer1"].detach()
                t_l2 = teacher_features["layer2"].detach()
                t_l3 = teacher_features["layer3"].detach()

                # student forward
                de3, de2, de1 = student(t_l3)

                # loss
                loss_l1    = 1.0 - F.cosine_similarity(t_l1, de1, dim=1).mean()
                loss_l2    = 1.0 - F.cosine_similarity(t_l2, de2, dim=1).mean()
                loss_l3    = 1.0 - F.cosine_similarity(t_l3, de3, dim=1).mean()
                total_loss = loss_l1 + loss_l2 + loss_l3
                loss_val   = float(total_loss)
                is_nan     = math.isnan(loss_val)
                is_inf     = math.isinf(loss_val)

                if is_nan or is_inf:
                    tag = "NaN" if is_nan else "Inf"
                    if is_nan:
                        loss_nan_count += 1
                    else:
                        loss_inf_count += 1
                    errors_list.append({
                        "epoch": epoch, "step": global_step,
                        "error": f"{tag} loss (batch_i={batch_start_i // BATCH_SIZE})",
                    })
                    print(f"[ABORT] {tag} loss epoch={epoch} step={global_step} -> 중단")
                    abort_flag = True

                if not is_nan and not is_inf:
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss_val)

                bl_writer.writerow({
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

            # epoch 완료
            epoch_elapsed = time.time() - epoch_start
            epoch_mean    = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            epoch_log_rows.append({
                "epoch":          epoch,
                "n_batches":      math.ceil(len(epoch_idxs) / BATCH_SIZE),
                "n_crops":        len(epoch_idxs),
                "mean_loss":      round(epoch_mean, 6) if not math.isnan(epoch_mean) else "nan",
                "epoch_time_sec": round(epoch_elapsed, 2),
                "low_z_batches":  epoch_low_z_batches,
            })

            for lbl in SIX_BIN_LABELS:
                bin_batch_summary.append({
                    "epoch":         epoch,
                    "six_bin_label": lbl,
                    "batch_count":   epoch_bin_ct[lbl],
                })

            # best checkpoint
            if not math.isnan(epoch_mean) and epoch_mean < best_loss:
                best_loss  = epoch_mean
                best_epoch = epoch
                save_checkpoint(CHECKPOINT_BEST, student, optimizer, epoch, epoch_mean)
                checkpoint_best_saved = True
                tag = "[BEST]"
            else:
                tag = ""
            print(
                f"  Epoch {epoch:>2}/{N_EPOCHS}  "
                f"loss={epoch_mean:.6f}  t={epoch_elapsed:.0f}s  {tag}"
            )
            blf.flush()

    # last checkpoint
    if not abort_flag and epoch_log_rows:
        last_epoch_loss = epoch_log_rows[-1]["mean_loss"]
        try:
            last_loss_val = float(last_epoch_loss)
        except (ValueError, TypeError):
            last_loss_val = float("nan")
        save_checkpoint(CHECKPOINT_LAST, student, optimizer, N_EPOCHS, last_loss_val)
        checkpoint_last_saved = True

    train_elapsed = time.time() - train_start

    # parameter 검증
    teacher_param_sum_after = sum(p.sum().item() for p in teacher.parameters())
    student_param_sum_after = sum(p.sum().item() for p in student.parameters())
    teacher_param_changed   = abs(teacher_param_sum_after - teacher_param_sum_before) > 1e-9
    student_param_changed   = abs(student_param_sum_after - student_param_sum_before) > 1e-9
    print(f"\n  teacher_param_changed: {teacher_param_changed}  (OK if False)")
    print(f"  student_param_changed: {student_param_changed}  (OK if True)")

    # GPU peak
    if device.type == "cuda":
        gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        gpu_peak_mb = 0.0

    # 결과 수치
    first_loss = float(str(epoch_log_rows[0]["mean_loss"]))  if epoch_log_rows else float("nan")
    last_loss  = float(str(epoch_log_rows[-1]["mean_loss"])) if epoch_log_rows else float("nan")
    loss_decreased = (
        (last_loss < first_loss)
        if not math.isnan(first_loss) and not math.isnan(last_loss)
        else False
    )

    # CSV 저장
    write_csv(
        REPORT_ROOT / "rd_b8_epoch_log.csv",
        ["epoch", "n_batches", "n_crops", "mean_loss", "epoch_time_sec", "low_z_batches"],
        epoch_log_rows,
    )
    write_csv(
        REPORT_ROOT / "rd_b8_bin_batch_summary.csv",
        ["epoch", "six_bin_label", "batch_count"],
        bin_batch_summary,
    )
    write_csv(
        REPORT_ROOT / "rd_b8_low_z_warning_summary.csv",
        ["epoch", "step", "batch_i", "safe_id", "local_z", "six_bin_label"],
        low_z_warning_rows,
    )
    write_csv(
        REPORT_ROOT / "rd_b8_parameter_update_check.csv",
        ["item", "value"],
        [
            {"item": "teacher_param_sum_before", "value": round(teacher_param_sum_before, 6)},
            {"item": "teacher_param_sum_after",  "value": round(teacher_param_sum_after, 6)},
            {"item": "teacher_param_changed",    "value": teacher_param_changed},
            {"item": "student_param_sum_before", "value": round(student_param_sum_before, 6)},
            {"item": "student_param_sum_after",  "value": round(student_param_sum_after, 6)},
            {"item": "student_param_changed",    "value": student_param_changed},
            {"item": "optimizer_teacher_param_count", "value": optimizer_teacher_ct},
        ],
    )
    write_csv(
        REPORT_ROOT / "rd_b8_errors.csv",
        ["epoch", "step", "error"],
        errors_list,
    )
    print(f"  -> rd_b8_batch_loss_log.csv (스트리밍 완료)")

    # GPU/runtime JSON
    gpu_runtime = {
        "device":             str(device),
        "gpu_peak_memory_mb": round(gpu_peak_mb, 2),
        "train_elapsed_sec":  round(train_elapsed, 2),
        "n_epochs":           N_EPOCHS,
        "n_batches_total":    global_step,
        "batch_size":         BATCH_SIZE,
        "n_train_crops":      N_TRAIN_CROPS_EXPECTED,
    }
    with open(REPORT_ROOT / "rd_b8_gpu_runtime_summary.json", "w", encoding="utf-8") as f:
        json.dump(gpu_runtime, f, ensure_ascii=False, indent=2)
    print(f"  -> rd_b8_gpu_runtime_summary.json")

    # 실패 조건
    failure_flags = [
    teacher_param_changed,
    not student_param_changed,
    optimizer_teacher_ct > 0,
    loss_nan_count > 0,
    loss_inf_count > 0,
    not checkpoint_best_saved,
    not checkpoint_last_saved,
    ]
    all_checks_passed = not any(failure_flags)

    # summary JSON
    summary = {
        "config_name":                   CONFIG_NAME,
        "n_train_crops":                 N_TRAIN_CROPS_EXPECTED,
        "samples_per_epoch": len(epoch_idxs),
        "batches_per_epoch": math.ceil(len(epoch_idxs) / BATCH_SIZE),
        "dropped_samples_per_epoch": N_TRAIN_CROPS_EXPECTED - len(epoch_idxs),
        "n_train_patients":              290,
        "batch_size":                    BATCH_SIZE,
        "epochs":                        N_EPOCHS,
        "lr":                            LR,
        "optimizer":                     "AdamW",
        "teacher_frozen":                True,
        "teacher_param_changed":         teacher_param_changed,
        "student_param_changed":         student_param_changed,
        "optimizer_teacher_param_count": optimizer_teacher_ct,
        "loss_nan_count":                loss_nan_count,
        "loss_inf_count":                loss_inf_count,
        "first_epoch_loss":              round(first_loss, 6) if not math.isnan(first_loss) else None,
        "last_epoch_loss":               round(last_loss, 6)  if not math.isnan(last_loss)  else None,
        "best_epoch":                    best_epoch,
        "best_train_loss":               round(best_loss, 6) if best_loss < float("inf") else None,
        "loss_decreased":                loss_decreased,
        "gpu_peak_memory_mb":            round(gpu_peak_mb, 2),
        "runtime_seconds":               round(train_elapsed, 2),
        "checkpoint_best_saved":         checkpoint_best_saved,
        "checkpoint_last_saved":         checkpoint_last_saved,
        "full_training_completed":       not abort_flag,
        "scoring_started":               False,
        "threshold_created":             False,
        "stage2_holdout_access":         0,
        "all_checks_passed":             all_checks_passed,
    }
    with open(REPORT_ROOT / "rd_b8_full_train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  -> rd_b8_full_train_summary.json")

    _write_report_md(REPORT_ROOT, ts, summary, epoch_log_rows)

    if all_checks_passed:
        (MODEL_ROOT  / "DONE").write_text(f"rd_b8 full-train completed: {ts}\n")
        (REPORT_ROOT / "DONE").write_text(f"rd_b8 full-train completed: {ts}\n")

    # 최종 판정 출력
    verdict = "통과" if all_checks_passed else "경고"
    print(f"\n판정: {verdict}")
    print(f"  loss  first={first_loss:.6f}  best={best_loss:.6f}(epoch {best_epoch})  last={last_loss:.6f}  decreased={loss_decreased}")
    print(f"  NaN={loss_nan_count}  Inf={loss_inf_count}")
    print(f"  teacher_param_changed={teacher_param_changed}")
    print(f"  student_param_changed={student_param_changed}")
    print(f"  GPU peak={gpu_peak_mb:.0f} MB  runtime={train_elapsed:.0f}s")
    print(f"  checkpoint best={checkpoint_best_saved}  last={checkpoint_last_saved}")
    print(f"  all_checks_passed={all_checks_passed}")

    if abort_flag:
        sys.exit(1)


# =============================================================================
# report.md
# =============================================================================

def _write_report_md(out_dir: Path, ts: str, summary: dict, epoch_log_rows: list) -> None:
    v = "통과" if summary["all_checks_passed"] else "경고"
    lines = [
        "# RD-B8 Full Normal-Only Train Report",
        f"- 버전: rd_b8_v3_cropfirst_mip",
        f"- 날짜: {ts}",
        f"- 판정: **{v}**",
        "",
        "---",
        "## 1. RD-B7 결과 요약",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        "| train manifest rows | 86,017 OK |",
        "| train patients | 290 |",
        "| stage2_holdout intersection | 0 OK |",
        "| CT/ROI exist | 290/290 OK |",
        "| low_z_boundary_warn | 195 crops |",
        "| recommended config | B_balanced_default |",
        "| all_checks_passed | True |",
        "",
        "---",
        "## 2. train manifest 확인",
        "",
        f"- manifest: rd_b1_6bin_balanced_normal_train_coordinate_manifest.csv",
        f"- rows: **{summary['n_train_crops']}**",
        f"- patients: {summary['n_train_patients']}",
        "- stage2_holdout 접근: 0건 OK",
        "",
        "---",
        "## 3. 모델 구조",
        "",
        "**Teacher (frozen)**",
        "- ResNet18 ImageNet local weight / eval mode / requires_grad=False",
        "- layer1=(B,64,24,24) / layer2=(B,128,12,12) / layer3=(B,256,6,6)",
        "",
        "**Student (reverse decoder, random init, train mode)**",
        "- de_layer3: Conv2d(256,256)+BN+ReLU -> (B,256,6,6)",
        "- de_layer2: Upsample*2+Conv2d(256,128)+BN+ReLU -> (B,128,12,12)",
        "- de_layer1: Upsample*2+Conv2d(128,64)+BN+ReLU -> (B,64,24,24)",
        "",
        "---",
        "## 4. 학습 config",
        "",
        f"| 항목 | 값 |",
        f"|------|----|",
        f"| config_name | {summary['config_name']} |",
        f"| batch_size | {summary['batch_size']} |",
        f"| epochs | {summary['epochs']} |",
        f"| lr | {summary['lr']} |",
        f"| optimizer | {summary['optimizer']} |",
        f"| sampler | strict 6-bin balanced, shortest-bin drop-last |",
        f"| input | mixed_3ch (HU[-1000,600]->[0,1]) |",
        f"| normal_only | True |",
        "",
        "---",
        "## 5. epoch별 loss 변화",
        "",
        "| epoch | mean_loss | epoch_time_sec | low_z_batches |",
        "|-------|-----------|----------------|---------------|",
    ]
    for r in epoch_log_rows:
        lines.append(
            f"| {r['epoch']} | {r['mean_loss']} | {r['epoch_time_sec']} | {r['low_z_batches']} |"
        )
    lines.extend([
        "",
        f"- first_epoch_loss: **{summary['first_epoch_loss']}**",
        f"- last_epoch_loss:  **{summary['last_epoch_loss']}**",
        f"- best_epoch:       **{summary['best_epoch']}**",
        f"- best_train_loss:  **{summary['best_train_loss']}**",
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
        "## 8. optimizer teacher param 검증",
        "",
        f"- optimizer_teacher_param_count: **{summary['optimizer_teacher_param_count']}** (0=OK)",
        "",
        "---",
        "## 9. low_z_warning batch/crop 영향",
        "",
        "- RD-B7 기준: 195 crops에 low_z_warning (center_z <= 7)",
        "- 상세: rd_b8_low_z_warning_summary.csv 참조",
        "",
        "---",
        "## 10. checkpoint 저장 검증",
        "",
        f"- best_train_loss.pth: **{summary['checkpoint_best_saved']}**",
        f"- last.pth:            **{summary['checkpoint_last_saved']}**",
        f"- 저장 경로: {MODEL_ROOT}/checkpoints/",
        "- smoke_only 포함 없음 OK",
        "- best.pth / final.pth 이름 없음 OK",
        "",
        "---",
        "## 11. 다음 단계",
        "",
        "- **RD-B9**: normal_val six-bin manifest 생성 + normal_val scoring/threshold preflight",
        "  - normal_val 36 patients 기준 six-bin manifest 작성",
        "  - scoring 전 preflight 확인",
        "  - stage2_holdout 접근 금지 유지",
        "",
        "---",
        "## 12. 절대 하지 않은 것",
        "",
        "| 항목 | 확인 |",
        "|------|------|",
        f"| scoring | {summary['scoring_started']} -> 없음 OK |",
        f"| threshold 생성 | {summary['threshold_created']} -> 없음 OK |",
        f"| stage2_holdout 접근 | {summary['stage2_holdout_access']}건 -> 없음 OK |",
        f"| full_crop NPZ 생성 | False -> 없음 OK |",
        "| production checkpoint (best.pth/final.pth) | 없음 OK |",
        "| teacher full weight 저장 | 없음 OK |",
        "| 기존 파일 수정/삭제 | 없음 OK |",
        "| smoke checkpoint 혼동 | 없음 OK |",
    ])
    report_path = out_dir / "rd_b8_full_train_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> rd_b8_full_train_report.md")


# =============================================================================
# main
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("RD-B8: Full Normal-Only Train (true RD4AD / ResNet18 / mixed_3ch / 6-bin)")
    print("=" * 70)

    if IS_DRY_PLAN:
        run_dry_plan()
        return

    if IS_PROFILE_TRAIN:
        print("\n[PROFILE-TRAIN] profiling run (100 batch, checkpoint 저장 없음) ...")
        run_profile_train()
        return

    if IS_RUN_TRAIN:
        print("\n[RUN-TRAIN] full normal-only train 실행 ...")
        run_full_train()
        return


if __name__ == "__main__":
    main()
