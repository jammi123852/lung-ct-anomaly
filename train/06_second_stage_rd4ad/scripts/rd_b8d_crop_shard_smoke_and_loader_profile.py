"""
RD-B8d: Crop Shard Smoke + Shard Loader Speed Test
목적: on-the-fly crop vs pre-built shard 로딩 속도 비교
     smoke 최대 2,000 crops 한정 (full 86,017 shard 생성 금지)
모드:
  bare run    -> exit 2 (파일 생성 금지)
  --dry-plan  -> 계획 출력만 (파일 생성 없음)
  --run-smoke -> smoke shard 생성 + loader profile 실행 (사용자 승인 후)
안전 조건:
  stage2_holdout/lesion 경로 접근 금지
  scoring/threshold/checkpoint 저장 금지
  output root 이미 존재 시 즉시 중단
  smoke 최대 2,000 crops
  full 86,017 shard 생성 금지
  full train 금지
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
ALLOWED_MODES = {"--dry-plan", "--run-smoke"}
if not any(m in sys.argv for m in ALLOWED_MODES):
    print("오류: 실행 모드 지정 필요.")
    print("  --dry-plan  : 계획 출력 (파일 생성 없음)")
    print("  --run-smoke : smoke shard 생성 + loader profile 실행 (사용자 승인 후)")
    sys.exit(2)

IS_DRY_PLAN  = "--dry-plan"  in sys.argv
IS_RUN_SMOKE = "--run-smoke" in sys.argv

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path("/home/jinhy/project/lung-ct-anomaly")
SMOKE_ROOT     = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit/rd_b8d_crop_shard_smoke_v1"
)
SHARD_ROOT_F16 = SMOKE_ROOT / "shards_float16"
SHARD_ROOT_F32 = SMOKE_ROOT / "shards_float32"

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

BATCH_SIZE          = 48
SMOKE_MAX_CROPS     = 2000
SHARD_SIZE          = 1000
SMOKE_PER_BIN       = 333   # 6 × 333 = 1998 ≤ 2000
PROFILE_MAX_BATCHES = 100
PATIENT_CACHE_SIZE  = 8
FULL_MANIFEST_ROWS  = 86017

SEED = 42


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
# 공통 함수 (rd_b8_full_train.py 로직 재사용)
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


def build_crop_np(ct_arr, center_z: int,
                  crop_y0: int, crop_x0: int,
                  crop_y1: int, crop_x1: int):
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
# LRU Shard Cache
# =============================================================================

class LRUShardCache:
    def __init__(self, max_size: int = 4):
        self._cache = collections.OrderedDict()
        self._max   = max_size

    def load(self, shard_path: Path):
        import numpy as np
        key = str(shard_path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        arr = np.load(str(shard_path), mmap_mode="r")
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
# RD Loss
# =============================================================================

def rd_loss_fn(teacher_feats, student_feats):
    import torch.nn.functional as F
    import torch
    loss = torch.tensor(0.0, device=teacher_feats[0].device)
    for tf, sf in zip(teacher_feats, student_feats):
        cos_sim = F.cosine_similarity(sf, tf, dim=1)
        loss = loss + (1 - cos_sim).mean()
    return loss / len(teacher_feats)


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
# manifest 로드
# =============================================================================

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
            sid     = row.get("safe_id", "")
            ct_path = row.get("ct_hu_npy", "")
            if sid and ct_path:
                assert_path_safe(ct_path)
                patient_paths[sid] = {"ct_hu_npy": ct_path}
    return patient_paths


# =============================================================================
# Phase 1: smoke subset 구성
# =============================================================================

def build_smoke_subset(all_rows: list, rng: random.Random) -> list:
    bins = {lbl: [] for lbl in SIX_BIN_LABELS}
    for i, row in enumerate(all_rows):
        lbl = row.get("six_bin_label", "")
        if lbl in bins:
            bins[lbl].append(i)

    selected = []
    for lbl in SIX_BIN_LABELS:
        idxs = list(bins[lbl])
        rng.shuffle(idxs)
        selected.extend(idxs[:SMOKE_PER_BIN])

    selected.sort()
    return selected


# =============================================================================
# Phase 2: shard 생성
# =============================================================================

def generate_shards(
    smoke_indices: list,
    all_rows: list,
    patient_paths: dict,
    patient_cache: LRUPatientCache,
    shard_root: Path,
    dtype_str: str,
) -> tuple:
    import numpy as np

    dtype = np.float16 if dtype_str == "float16" else np.float32
    n_total  = len(smoke_indices)
    n_shards = math.ceil(n_total / SHARD_SIZE)
    shard_files  = []
    timing_rows  = []
    error_rows   = []
    n_low_z      = 0
    n_edge_pad   = 0

    print(f"  [{dtype_str}] shard 생성: {n_total} crops, {n_shards} shards")

    for s in range(n_shards):
        batch_idxs  = smoke_indices[s * SHARD_SIZE : (s + 1) * SHARD_SIZE]
        shard_crops = []
        t_start     = time.perf_counter()

        for idx in batch_idxs:
            row     = all_rows[idx]
            sid     = row.get("safe_id", "")
            lz      = int(row.get("local_z", 0))
            y0      = int(row.get("crop_y0", 0))
            x0      = int(row.get("crop_x0", 0))
            y1      = int(row.get("crop_y1", 96))
            x1      = int(row.get("crop_x1", 96))
            ct_path = patient_paths.get(sid, {}).get("ct_hu_npy", "")
            if not ct_path:
                error_rows.append({
                    "phase": "shard_gen", "dtype": dtype_str,
                    "idx": idx, "safe_id": sid, "error": "no_ct_path",
                })
                shard_crops.append(np.zeros((3, 96, 96), dtype=dtype))
                continue
            try:
                ct_arr  = patient_cache.load(sid, ct_path)
                crop_np = build_crop_np(ct_arr, lz, y0, x0, y1, x1)
                if has_low_z_boundary_warning(lz):
                    n_low_z += 1
                _, h_max, w_max = ct_arr.shape
                if y0 < 0 or x0 < 0 or y1 > h_max or x1 > w_max:
                    n_edge_pad += 1
                shard_crops.append(crop_np.astype(dtype))
            except Exception as e:
                error_rows.append({
                    "phase": "shard_gen", "dtype": dtype_str,
                    "idx": idx, "safe_id": sid, "error": str(e),
                })
                shard_crops.append(np.zeros((3, 96, 96), dtype=dtype))

        shard_arr  = np.stack(shard_crops, axis=0)
        shard_name = f"rd_b8d_smoke_shard_{s:03d}.npy"
        shard_path = shard_root / shard_name
        np.save(str(shard_path), shard_arr)
        t_end = time.perf_counter()

        disk_mb = shard_path.stat().st_size / (1024 * 1024)
        timing_rows.append({
            "dtype": dtype_str, "shard_id": s,
            "n_crops": len(batch_idxs),
            "generation_time_sec": round(t_end - t_start, 4),
            "disk_mb": round(disk_mb, 3),
        })
        shard_files.append(shard_path)
        print(f"    shard {s}: {len(batch_idxs)} crops, {disk_mb:.1f} MB, {t_end - t_start:.2f}s")

    return shard_files, timing_rows, error_rows, n_low_z, n_edge_pad


# =============================================================================
# Phase 3: 모델 setup / forward-backward
# =============================================================================

def setup_model(device):
    import torch

    teacher = build_teacher(LOCAL_WEIGHT_PATH).to(device)
    student = build_student_decoder().to(device)
    student.train()

    teacher_features = {}
    for name, module in [
        ("layer1", teacher.layer1),
        ("layer2", teacher.layer2),
        ("layer3", teacher.layer3),
    ]:
        def _hook(module, inp, output, _name=name):
            teacher_features[_name] = output
        module.register_forward_hook(_hook)

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=1e-4, weight_decay=1e-5
    )
    return teacher, student, optimizer, teacher_features


def forward_backward_step(batch_np, teacher, student, optimizer, teacher_features, device):
    import torch

    t0      = time.perf_counter()
    batch_t = torch.from_numpy(batch_np).to(device)
    t1      = time.perf_counter()

    with torch.no_grad():
        teacher(batch_t)
    t2 = time.perf_counter()

    tf3 = teacher_features["layer3"]
    tf2 = teacher_features["layer2"]
    tf1 = teacher_features["layer1"]

    de3, de2, de1 = student(tf3)
    t3 = time.perf_counter()

    loss = rd_loss_fn([tf3, tf2, tf1], [de3, de2, de1])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    t4 = time.perf_counter()

    return {
        "tensor_to_device_time": round(t1 - t0, 4),
        "teacher_fwd_time":      round(t2 - t1, 4),
        "student_fwd_time":      round(t3 - t2, 4),
        "backward_step_time":    round(t4 - t3, 4),
        "fwd_bwd_time":          round(t4 - t0, 4),
        "loss":                  round(float(loss.item()), 6),
        "loss_nan":              int(loss.isnan().item()),
        "loss_inf":              int(loss.isinf().item()),
    }


# =============================================================================
# Phase 3A: on-the-fly baseline
# =============================================================================

def profile_on_the_fly(smoke_indices, all_rows, patient_paths,
                       device, teacher, student, optimizer, teacher_features):
    import numpy as np

    n_batches = min(PROFILE_MAX_BATCHES, len(smoke_indices) // BATCH_SIZE)
    print(f"  [on_the_fly] profile batches: {n_batches}")
    patient_cache = LRUPatientCache(max_size=PATIENT_CACHE_SIZE)
    timing_rows   = []

    for batch_i in range(n_batches):
        t_load_s  = time.perf_counter()
        batch_crops = []
        sl = smoke_indices[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE]
        for idx in sl:
            row     = all_rows[idx]
            sid     = row.get("safe_id", "")
            lz      = int(row.get("local_z", 0))
            y0      = int(row.get("crop_y0", 0))
            x0      = int(row.get("crop_x0", 0))
            y1      = int(row.get("crop_y1", 96))
            x1      = int(row.get("crop_x1", 96))
            ct_path = patient_paths.get(sid, {}).get("ct_hu_npy", "")
            ct_arr  = patient_cache.load(sid, ct_path)
            crop_np = build_crop_np(ct_arr, lz, y0, x0, y1, x1)
            batch_crops.append(crop_np)
        batch_np  = np.stack(batch_crops, axis=0)
        t_load_e  = time.perf_counter()

        r = forward_backward_step(batch_np, teacher, student, optimizer, teacher_features, device)
        r["loader_type"]     = "on_the_fly"
        r["batch_i"]         = batch_i
        r["load_batch_time"] = round(t_load_e - t_load_s, 4)
        r["total_batch_time"] = round((t_load_e - t_load_s) + r["fwd_bwd_time"], 4)
        timing_rows.append(r)
        if batch_i % 10 == 0:
            print(f"    batch {batch_i:3d}/{n_batches}  load={r['load_batch_time']:.3f}s  total={r['total_batch_time']:.3f}s  loss={r['loss']:.4f}")

    return timing_rows


# =============================================================================
# Phase 3B/C: shard loader profile
# =============================================================================

def profile_shard_loader(smoke_indices, shard_root, dtype_str,
                         device, teacher, student, optimizer, teacher_features):
    import numpy as np

    # smoke-local index → (shard_id, row_in_shard)
    shard_index = [
        (i // SHARD_SIZE, i % SHARD_SIZE)
        for i in range(len(smoke_indices))
    ]
    shard_cache = LRUShardCache(max_size=4)
    n_batches   = min(PROFILE_MAX_BATCHES, len(smoke_indices) // BATCH_SIZE)
    print(f"  [{dtype_str}_shard] profile batches: {n_batches}")
    timing_rows = []

    for batch_i in range(n_batches):
        t_load_s    = time.perf_counter()
        batch_crops = []
        for local_i in range(batch_i * BATCH_SIZE, (batch_i + 1) * BATCH_SIZE):
            shard_id, row_in_shard = shard_index[local_i]
            shard_path = shard_root / f"rd_b8d_smoke_shard_{shard_id:03d}.npy"
            shard_arr  = shard_cache.load(shard_path)
            crop_np    = shard_arr[row_in_shard].astype("float32")
            batch_crops.append(crop_np)
        batch_np  = np.stack(batch_crops, axis=0)
        t_load_e  = time.perf_counter()

        r = forward_backward_step(batch_np, teacher, student, optimizer, teacher_features, device)
        r["loader_type"]     = f"{dtype_str}_shard"
        r["batch_i"]         = batch_i
        r["load_batch_time"] = round(t_load_e - t_load_s, 4)
        r["total_batch_time"] = round((t_load_e - t_load_s) + r["fwd_bwd_time"], 4)
        timing_rows.append(r)
        if batch_i % 10 == 0:
            print(f"    batch {batch_i:3d}/{n_batches}  load={r['load_batch_time']:.3f}s  total={r['total_batch_time']:.3f}s  loss={r['loss']:.4f}")

    return timing_rows


# =============================================================================
# dry-plan
# =============================================================================

def run_dry_plan():
    n_smoke = 6 * SMOKE_PER_BIN
    n_shards = math.ceil(n_smoke / SHARD_SIZE)
    f16_per_shard_mb = SHARD_SIZE * 3 * 96 * 96 * 2 / (1024 * 1024)
    f32_per_shard_mb = SHARD_SIZE * 3 * 96 * 96 * 4 / (1024 * 1024)
    max_batches = min(PROFILE_MAX_BATCHES, n_smoke // BATCH_SIZE)

    print("=" * 70)
    print("RD-B8d: Crop Shard Smoke + Shard Loader Speed Test [DRY-PLAN]")
    print("=" * 70)
    print()
    print("## 1. 목적")
    print("  RD-B8 v3 profile: crop build = 90.9% of batch time (0.561s / 0.618s)")
    print("  normalized mixed_3ch crop을 shard로 저장하면 학습이 빠른지 확인")
    print()
    print("## 2. smoke subset")
    print(f"  per_bin={SMOKE_PER_BIN}, 6bin × {SMOKE_PER_BIN} = {n_smoke} crops (≤ {SMOKE_MAX_CROPS})")
    print(f"  shard_size={SHARD_SIZE}, n_shards={n_shards}")
    print()
    print("## 3. shard 저장 방식")
    print(f"  float16 npy: shape=(N,3,96,96), 예상 {f16_per_shard_mb:.1f} MB/shard")
    print(f"  float32 npy: shape=(N,3,96,96), 예상 {f32_per_shard_mb:.1f} MB/shard")
    print()
    print("## 4. loader profile")
    print(f"  batch_size={BATCH_SIZE}, max_batches={max_batches}")
    print("  A. on_the_fly (baseline)")
    print("  B. float16 shard loader")
    print("  C. float32 shard loader")
    print()
    print("## 5. 출력 root")
    print(f"  {SMOKE_ROOT}")
    print()
    print("## 6. 안전 조건")
    print("  stage2_holdout/lesion 접근: 금지")
    print("  checkpoint 저장: 금지")
    print("  full train: 금지")
    print(f"  full shard 생성({FULL_MANIFEST_ROWS}): 금지 (smoke {n_smoke}만)")
    print()
    print("판정: DRY-PLAN OK")
    print("  사용자 승인 후 실행:")
    print("  python scripts/rd_b8d_crop_shard_smoke_and_loader_profile.py --run-smoke")


# =============================================================================
# run_smoke
# =============================================================================

def run_smoke():
    import numpy as np
    import torch

    print("=" * 70)
    print("RD-B8d: Crop Shard Smoke + Shard Loader Speed Test [RUN-SMOKE]")
    print("=" * 70)

    # output root guard
    if SMOKE_ROOT.exists():
        print(f"[ABORT] output root 이미 존재: {SMOKE_ROOT}")
        sys.exit(1)
    SMOKE_ROOT.mkdir(parents=True, exist_ok=False)
    SHARD_ROOT_F16.mkdir(parents=True, exist_ok=False)
    SHARD_ROOT_F32.mkdir(parents=True, exist_ok=False)

    rng = random.Random(SEED)

    # manifest 로드
    print("  train manifest 로드 중 ...")
    all_rows = load_full_manifest(TRAIN_MANIFEST_PATH)
    print(f"  manifest rows: {len(all_rows)}")

    # stage2_holdout 접근 0 확인
    for row in all_rows[:20]:
        assert_path_safe(row.get("safe_id", ""))
    print("  stage2_holdout intersection: 0 OK")

    print("  patient manifest 로드 중 ...")
    patient_paths = load_patient_paths(PATIENT_MANIFEST_PATH)
    print(f"  patient paths: {len(patient_paths)}")

    # Phase 1: smoke subset
    print()
    print("── Phase 1: smoke subset 구성 ──")
    smoke_indices = build_smoke_subset(all_rows, rng)
    n_smoke = len(smoke_indices)
    print(f"  smoke crops: {n_smoke}")

    n_low_z_check = sum(
        1 for i in smoke_indices
        if has_low_z_boundary_warning(int(all_rows[i].get("local_z", 0)))
    )
    print(f"  low_z_warning crops: {n_low_z_check}")

    # smoke subset manifest
    subset_rows = []
    for rank, idx in enumerate(smoke_indices):
        r = dict(all_rows[idx])
        r["smoke_rank"]              = rank
        r["shard_id"]                = rank // SHARD_SIZE
        r["row_in_shard"]            = rank % SHARD_SIZE
        r["original_manifest_index"] = idx
        r["low_z_warning"] = int(has_low_z_boundary_warning(int(r.get("local_z", 0))))
        subset_rows.append(r)

    if subset_rows:
        write_csv(
            SMOKE_ROOT / "rd_b8d_smoke_subset_manifest.csv",
            list(subset_rows[0].keys()),
            subset_rows,
        )

    # shard index CSV
    shard_index_rows = [
        {
            "shard_id":                rank // SHARD_SIZE,
            "row_in_shard":            rank % SHARD_SIZE,
            "smoke_rank":              rank,
            "original_manifest_index": idx,
            "safe_id":                 all_rows[idx].get("safe_id", ""),
            "six_bin_label":           all_rows[idx].get("six_bin_label", ""),
            "local_z":                 all_rows[idx].get("local_z", ""),
            "crop_y0":                 all_rows[idx].get("crop_y0", ""),
            "crop_x0":                 all_rows[idx].get("crop_x0", ""),
            "crop_y1":                 all_rows[idx].get("crop_y1", ""),
            "crop_x1":                 all_rows[idx].get("crop_x1", ""),
            "low_z_warning": int(has_low_z_boundary_warning(int(all_rows[idx].get("local_z", 0)))),
        }
        for rank, idx in enumerate(smoke_indices)
    ]
    write_csv(
        SMOKE_ROOT / "rd_b8d_smoke_shard_index.csv",
        ["shard_id", "row_in_shard", "smoke_rank", "original_manifest_index",
         "safe_id", "six_bin_label", "local_z",
         "crop_y0", "crop_x0", "crop_y1", "crop_x1", "low_z_warning"],
        shard_index_rows,
    )

    # Phase 2: shard 생성
    print()
    print("── Phase 2: shard 생성 ──")
    all_error_rows  = []
    gen_timing_rows = []

    t_f16_s = time.perf_counter()
    shard_files_f16, timing_f16, errors_f16, n_low_z_f16, n_edge_f16 = generate_shards(
        smoke_indices, all_rows, patient_paths,
        LRUPatientCache(max_size=PATIENT_CACHE_SIZE),
        SHARD_ROOT_F16, "float16",
    )
    t_f16_e = time.perf_counter()
    f16_gen_time = t_f16_e - t_f16_s
    gen_timing_rows.extend(timing_f16)
    all_error_rows.extend(errors_f16)

    t_f32_s = time.perf_counter()
    shard_files_f32, timing_f32, errors_f32, n_low_z_f32, n_edge_f32 = generate_shards(
        smoke_indices, all_rows, patient_paths,
        LRUPatientCache(max_size=PATIENT_CACHE_SIZE),
        SHARD_ROOT_F32, "float32",
    )
    t_f32_e = time.perf_counter()
    f32_gen_time = t_f32_e - t_f32_s
    gen_timing_rows.extend(timing_f32)
    all_error_rows.extend(errors_f32)

    write_csv(
        SMOKE_ROOT / "rd_b8d_shard_generation_timing.csv",
        ["dtype", "shard_id", "n_crops", "generation_time_sec", "disk_mb"],
        gen_timing_rows,
    )

    f16_total_mb = sum(f.stat().st_size for f in shard_files_f16) / (1024 * 1024)
    f32_total_mb = sum(f.stat().st_size for f in shard_files_f32) / (1024 * 1024)
    print(f"  float16 total: {f16_total_mb:.1f} MB, 생성시간: {f16_gen_time:.1f}s")
    print(f"  float32 total: {f32_total_mb:.1f} MB, 생성시간: {f32_gen_time:.1f}s")

    # Phase 3: loader profile
    print()
    print("── Phase 3: loader profile ──")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    teacher_otf, student_otf, opt_otf, tf_otf = setup_model(device)
    print("  [A] on-the-fly baseline ...")
    timing_otf = profile_on_the_fly(
        smoke_indices, all_rows, patient_paths,
        device, teacher_otf, student_otf, opt_otf, tf_otf,
    )

    teacher_f16, student_f16, opt_f16, tf_f16 = setup_model(device)
    print("  [B] float16 shard loader ...")
    timing_f16_ldr = profile_shard_loader(
        smoke_indices, SHARD_ROOT_F16, "float16",
        device, teacher_f16, student_f16, opt_f16, tf_f16,
    )

    teacher_f32, student_f32, opt_f32, tf_f32 = setup_model(device)
    print("  [C] float32 shard loader ...")
    timing_f32_ldr = profile_shard_loader(
        smoke_indices, SHARD_ROOT_F32, "float32",
        device, teacher_f32, student_f32, opt_f32, tf_f32,
    )

    gpu_peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda" else 0.0
    )

    # loader profile CSV
    all_ldr_rows = timing_otf + timing_f16_ldr + timing_f32_ldr
    write_csv(
        SMOKE_ROOT / "rd_b8d_shard_loader_profile.csv",
        ["loader_type", "batch_i", "load_batch_time",
         "tensor_to_device_time", "teacher_fwd_time", "student_fwd_time",
         "backward_step_time", "total_batch_time", "loss", "loss_nan", "loss_inf"],
        all_ldr_rows,
    )

    # error CSV
    write_csv(
        SMOKE_ROOT / "rd_b8d_errors.csv",
        ["phase", "dtype", "idx", "safe_id", "error"],
        all_error_rows,
    )

    # ── 통계 계산 ──
    def mean_field(rows, key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    otf_mean_load   = mean_field(timing_otf,     "load_batch_time")
    f16_mean_load   = mean_field(timing_f16_ldr, "load_batch_time")
    f32_mean_load   = mean_field(timing_f32_ldr, "load_batch_time")
    otf_mean_batch  = mean_field(timing_otf,     "total_batch_time")
    f16_mean_batch  = mean_field(timing_f16_ldr, "total_batch_time")
    f32_mean_batch  = mean_field(timing_f32_ldr, "total_batch_time")

    nan_total = sum(r.get("loss_nan", 0) for r in all_ldr_rows)
    inf_total = sum(r.get("loss_inf", 0) for r in all_ldr_rows)

    # 예상 시간 계산
    f16_sec_per_crop           = f16_gen_time / n_smoke if n_smoke > 0 else 0.0
    est_full_shard_gen_min     = round(f16_sec_per_crop * FULL_MANIFEST_ROWS / 60, 1)
    full_batches_per_epoch     = FULL_MANIFEST_ROWS // BATCH_SIZE
    est_20ep_otf_hr            = round(otf_mean_batch * full_batches_per_epoch * 20 / 3600, 2)
    est_20ep_f16_hr            = round(f16_mean_batch * full_batches_per_epoch * 20 / 3600, 2)
    est_20ep_f32_hr            = round(f32_mean_batch * full_batches_per_epoch * 20 / 3600, 2)
    f16_speedup                = round(otf_mean_batch / f16_mean_batch, 2) if f16_mean_batch > 0 else 0.0
    f32_speedup                = round(otf_mean_batch / f32_mean_batch, 2) if f32_mean_batch > 0 else 0.0
    est_total_wall_f16_hr      = round(est_full_shard_gen_min / 60 + est_20ep_f16_hr, 2)
    est_total_wall_f32_hr      = round(est_full_shard_gen_min / 60 + est_20ep_f32_hr, 2)

    # 판정
    if est_total_wall_f16_hr <= 3.0:
        verdict          = "GO"
        recommended_next = "full_shard_generation_float16"
    elif est_total_wall_f16_hr <= 4.0:
        verdict          = "조건부_GO"
        recommended_next = "full_shard_generation_float16_조건부"
    else:
        verdict          = "추가_분석_필요"
        recommended_next = "on_the_fly_6hr_train_or_further_analysis"

    recommended_dtype = (
        "float16"
        if (f16_total_mb < f32_total_mb and f16_speedup >= f32_speedup * 0.95)
        else "float32"
    )
    all_checks_passed = (nan_total == 0 and inf_total == 0 and len(all_error_rows) == 0)

    # summary JSON
    summary = {
        "n_smoke_crops":                        n_smoke,
        "n_shards_float16":                     len(shard_files_f16),
        "n_shards_float32":                     len(shard_files_f32),
        "float16_disk_mb":                      round(f16_total_mb, 2),
        "float32_disk_mb":                      round(f32_total_mb, 2),
        "float16_generation_time_sec":          round(f16_gen_time, 2),
        "float32_generation_time_sec":          round(f32_gen_time, 2),
        "on_the_fly_mean_batch_time":           otf_mean_batch,
        "float16_shard_mean_batch_time":        f16_mean_batch,
        "float32_shard_mean_batch_time":        f32_mean_batch,
        "float16_speedup_vs_on_the_fly":        f16_speedup,
        "float32_speedup_vs_on_the_fly":        f32_speedup,
        "estimated_full_shard_generation_min":  est_full_shard_gen_min,
        "estimated_20epoch_train_hour_float16": est_20ep_f16_hr,
        "estimated_20epoch_train_hour_float32": est_20ep_f32_hr,
        "estimated_20epoch_train_hour_on_the_fly": est_20ep_otf_hr,
        "estimated_total_wall_hour_float16":    est_total_wall_f16_hr,
        "estimated_total_wall_hour_float32":    est_total_wall_f32_hr,
        "recommended_storage_dtype":            recommended_dtype,
        "recommended_next_step":                recommended_next,
        "verdict":                              verdict,
        "loss_nan_count":                       nan_total,
        "loss_inf_count":                       inf_total,
        "gpu_peak_memory_mb":                   round(gpu_peak_mb, 1),
        "checkpoint_saved":                     False,
        "scoring_started":                      False,
        "threshold_created":                    False,
        "stage2_holdout_access":                0,
        "full_train_started":                   False,
        "full_shard_generated":                 False,
        "all_checks_passed":                    all_checks_passed,
    }
    with open(SMOKE_ROOT / "rd_b8d_shard_smoke_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("  -> rd_b8d_shard_smoke_summary.json")

    # report MD
    md_lines = [
        "# RD-B8d Crop Shard Smoke + Shard Loader Speed Test Report",
        "",
        "## 1. RD-B8 v3 profile 결과 요약",
        "| 항목 | 값 |",
        "|---|---|",
        "| mean batch time | 0.618s |",
        "| mean crop build time | 0.561s |",
        "| crop build 비중 | 90.9% |",
        "| estimated 20 epoch | 6.0 hr |",
        "| NaN/Inf | 0 |",
        "",
        "## 2. 왜 shard smoke를 하는가",
        "on-the-fly crop build가 batch time의 90.9%를 차지한다.",
        "미리 shard로 저장하면 학습 속도가 크게 향상될 수 있다.",
        f"전체 {FULL_MANIFEST_ROWS} shard 생성 전에 smoke {n_smoke} crops로 실제 속도를 먼저 확인한다.",
        "",
        "## 3. smoke subset 구성",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 총 smoke crops | {n_smoke} |",
        f"| per bin | {SMOKE_PER_BIN} |",
        f"| low_z_warning crops | {n_low_z_check} |",
        f"| shard size | {SHARD_SIZE} |",
        f"| n_shards (float16) | {len(shard_files_f16)} |",
        f"| n_shards (float32) | {len(shard_files_f32)} |",
        "",
        "## 4. float16/float32 shard 생성 시간과 용량",
        "| dtype | 총 용량 (MB) | 생성 시간 (s) |",
        "|---|---|---|",
        f"| float16 | {f16_total_mb:.1f} | {f16_gen_time:.1f} |",
        f"| float32 | {f32_total_mb:.1f} | {f32_gen_time:.1f} |",
        "",
        "## 5. shard loader profile 결과",
        "| loader | mean load time | mean total time |",
        "|---|---|---|",
        f"| on_the_fly | {otf_mean_load:.3f}s | {otf_mean_batch:.3f}s |",
        f"| float16_shard | {f16_mean_load:.3f}s | {f16_mean_batch:.3f}s |",
        f"| float32_shard | {f32_mean_load:.3f}s | {f32_mean_batch:.3f}s |",
        "",
        "## 6. on-the-fly 대비 speedup",
        "| dtype | speedup |",
        "|---|---|",
        f"| float16 shard | {f16_speedup}× |",
        f"| float32 shard | {f32_speedup}× |",
        "",
        "## 7. 전체 86,017 crop shard 생성 예상 시간",
        "| 항목 | 값 |",
        "|---|---|",
        f"| float16 per crop | {f16_sec_per_crop:.4f}s |",
        f"| 전체 예상 시간 | {est_full_shard_gen_min:.1f} min |",
        "",
        "## 8. shard 기반 20 epoch train 예상 시간",
        "| 방식 | 예상 시간 |",
        "|---|---|",
        f"| float16 shard | {est_20ep_f16_hr:.2f} hr |",
        f"| float32 shard | {est_20ep_f32_hr:.2f} hr |",
        f"| on-the-fly (기준) | {est_20ep_otf_hr:.2f} hr |",
        "",
        "## 9. 최종 추천",
        f"- **판정: {verdict}**",
        f"- 추천 dtype: **{recommended_dtype}**",
        f"- 추천 다음 단계: {recommended_next}",
        f"- 예상 total wall time (float16 shard):  {est_total_wall_f16_hr:.2f} hr",
        "  (= full shard 생성 + 20 epoch shard 학습)",
        f"- 예상 total wall time (float32 shard):  {est_total_wall_f32_hr:.2f} hr",
        "",
        "## 10. 절대 하지 않은 것",
        "| 항목 | 상태 |",
        "|---|---|",
        "| full train | False |",
        f"| full shard 생성 ({FULL_MANIFEST_ROWS}) | False |",
        "| checkpoint 저장 | False |",
        "| scoring | False |",
        "| threshold | False |",
        "| stage2_holdout 접근 | 0 |",
        "",
        "## 11. 안전 검사",
        "| 항목 | 결과 |",
        "|---|---|",
        f"| all_checks_passed | {all_checks_passed} |",
        f"| NaN | {nan_total} |",
        f"| Inf | {inf_total} |",
        f"| error rows | {len(all_error_rows)} |",
        f"| GPU peak | {gpu_peak_mb:.0f} MB |",
    ]
    with open(SMOKE_ROOT / "rd_b8d_shard_smoke_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("  -> rd_b8d_shard_smoke_report.md")

    # DONE marker
    (SMOKE_ROOT / "DONE").write_text("rd_b8d_crop_shard_smoke_v1 DONE\n", encoding="utf-8")
    print("  -> DONE")

    # ── 최종 출력 ──
    print()
    print("=" * 70)
    print(f"판정: {verdict}")
    print(f"  smoke crops:                    {n_smoke}")
    print(f"  float16 shard 생성 시간:        {f16_gen_time:.1f}s")
    print(f"  float32 shard 생성 시간:        {f32_gen_time:.1f}s")
    print(f"  float16 용량:                   {f16_total_mb:.1f} MB")
    print(f"  float32 용량:                   {f32_total_mb:.1f} MB")
    print(f"  on-the-fly mean batch time:     {otf_mean_batch:.3f}s")
    print(f"  float16 shard mean batch time:  {f16_mean_batch:.3f}s")
    print(f"  float32 shard mean batch time:  {f32_mean_batch:.3f}s")
    print(f"  float16 speedup:                {f16_speedup}×")
    print(f"  float32 speedup:                {f32_speedup}×")
    print(f"  예상 full shard gen:            {est_full_shard_gen_min:.1f} min")
    print(f"  예상 20ep shard train (f16):    {est_20ep_f16_hr:.2f} hr")
    print(f"  예상 total wall time (f16):     {est_total_wall_f16_hr:.2f} hr")
    print(f"  추천 dtype:                     {recommended_dtype}")
    print(f"  NaN={nan_total}  Inf={inf_total}  errors={len(all_error_rows)}")
    print(f"  GPU peak: {gpu_peak_mb:.0f} MB")
    print(f"  all_checks_passed={all_checks_passed}")
    print(f"  checkpoint_saved=False  scoring_started=False")
    print(f"  stage2_holdout_access=0  full_shard_generated=False")
    print("=" * 70)


# =============================================================================
# 진입점
# =============================================================================

if IS_DRY_PLAN:
    run_dry_plan()
elif IS_RUN_SMOKE:
    run_smoke()
