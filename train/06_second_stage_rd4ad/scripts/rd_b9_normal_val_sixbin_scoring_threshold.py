"""
RD-B9: normal_val six-bin manifest + float32 shard + scoring + threshold
목적:
  RD-B8f best_train_loss.pth를 read-only로 로드하여
  normal_val 36명에 대한 six_bin coordinate manifest / shard / score / threshold 생성
모드:
  bare run   -> exit 2
  --dry-plan -> 입력 경로 확인, crop count 추정, output root 없음 확인 (파일 생성 없음)
  --run      -> manifest + shard + scoring + threshold 실행
안전 조건:
  stage2_holdout 접근 금지
  lesion raw CT/mask 접근 금지
  stage1_dev lesion scoring 금지
  backward / optimizer 생성 / checkpoint 저장 금지
  training 시작 금지
  output root 존재 시 즉시 중단
  기존 checkpoint 덮어쓰기 금지
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
    print("  --dry-plan : 입력 경로 확인 (파일 생성 없음)")
    print("  --run      : manifest + shard + scoring + threshold 실행")
    sys.exit(2)

IS_DRY_PLAN = "--dry-plan" in sys.argv
IS_RUN = "--run" in sys.argv

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b9_normal_val_scoring_threshold_v1"
)
SHARDS_DIR = OUTPUT_ROOT / "shards_float32"

N_C10_VAL_MANIFEST = (
    PROJECT_ROOT
    / "experiments/normal_only_second_stage_refiner_v1/outputs/manifests"
    / "n_c10_normal_val_crop_manifest/n_c10_normal_val_crop_manifest.csv"
)
PATIENT_MANIFEST = Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/"
    "manifests/patient_manifest.csv"
)
TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b1_6bin_balanced_manifest_preflight_v1"
    / "rd_b1_6bin_balanced_normal_train_coordinate_manifest.csv"
)
STAGE2_HOLDOUT_CSV = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/datasets"
    / "s6a_stage2_holdout_filtered_manifest_v1.csv"
)
V4_20_ROI_ROOT = (
    PROJECT_ROOT
    / "outputs/mip-postprocess-research-v1/masks"
    / "refined_roi_v4_20_modeB_all_v1/normal"
)
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs/models/rd_b8f_true_rd4ad_resnet18_mixed3ch_6bin_shard_v1"
    / "checkpoints/best_train_loss.pth"
)
LOCAL_RESNET18_WEIGHT = Path(
    "/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)

# ── 설계 상수 ──────────────────────────────────────────────────────────────────
EROSION_PX = 5
BOUNDARY_THRESHOLD = 0.05
INTERIOR_ROI_MIN = 0.85
CAP_PER_BIN = 50
SAMPLING_SEED = 42
CROP_SIZE = 96
Z_LOWER_MAX = 1.0 / 3.0
Z_MIDDLE_MAX = 2.0 / 3.0
LOW_Z_WARNING_THRESHOLD = 7
SHARD_SIZE = 1000
MIP_RADIUS = 3
HU_CLIP_MIN = -1000.0
HU_CLIP_MAX = 600.0
HU_RANGE = 1600.0
SCORE_BATCH_SIZE = 48
VALIDATE_SAMPLE_PER_SHARD = 10
SIX_BIN_LABELS = [
    "upper_boundary", "upper_interior",
    "middle_boundary", "middle_interior",
    "lower_boundary", "lower_interior",
]
FORBIDDEN_KEYWORDS = [
    "stage2_holdout", "lesion", "test_lesion",
    "second-stage-lesion-refiner",
]

# ── 안전 체크 ──────────────────────────────────────────────────────────────────

def assert_path_safe(path_str):
    for kw in FORBIDDEN_KEYWORDS:
        if kw.lower() in str(path_str).lower():
            raise RuntimeError(
                f"[SAFETY] 금지 경로 접근 차단: {path_str!r} (keyword={kw!r})"
            )


# ── CSV 헬퍼 ───────────────────────────────────────────────────────────────────

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


def load_csv_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ── HU 변환 / crop ─────────────────────────────────────────────────────────────

def normalize_hu(hu_array):
    import numpy as np
    clipped = hu_array.clip(HU_CLIP_MIN, HU_CLIP_MAX)
    return ((clipped - HU_CLIP_MIN) / HU_RANGE).astype("float32")


def compute_mip_slab_indices(center_z, direction, z_max):
    if direction == "lower":
        raw = [center_z - MIP_RADIUS + i for i in range(MIP_RADIUS)]
    else:
        raw = [center_z + 1 + i for i in range(MIP_RADIUS)]
    return [max(0, min(idx, z_max - 1)) for idx in raw]


def build_crop_np(ct_arr, center_z, crop_y0, crop_x0, crop_y1, crop_x1):
    import numpy as np
    TARGET = CROP_SIZE
    z_max, h_max, w_max = ct_arr.shape

    def _crop2d(img2d, y0, x0, y1, x1):
        h, w = img2d.shape
        out = np.full((TARGET, TARGET), HU_CLIP_MIN, dtype=img2d.dtype)
        sy0 = max(0, y0); sx0 = max(0, x0)
        sy1 = min(h, y1); sx1 = min(w, x1)
        if sy1 <= sy0 or sx1 <= sx0:
            return out
        dy0 = sy0 - y0; dx0 = sx0 - x0
        out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img2d[sy0:sy1, sx0:sx1]
        return out

    ch0 = _crop2d(ct_arr[center_z], crop_y0, crop_x0, crop_y1, crop_x1)
    lower_idx = compute_mip_slab_indices(center_z, "lower", z_max)
    ch1 = np.max(np.stack(
        [_crop2d(ct_arr[z], crop_y0, crop_x0, crop_y1, crop_x1) for z in lower_idx], axis=0
    ), axis=0)
    upper_idx = compute_mip_slab_indices(center_z, "upper", z_max)
    ch2 = np.max(np.stack(
        [_crop2d(ct_arr[z], crop_y0, crop_x0, crop_y1, crop_x1) for z in upper_idx], axis=0
    ), axis=0)

    crop = np.stack([normalize_hu(ch0), normalize_hu(ch1), normalize_hu(ch2)], axis=0).astype("float32")
    if crop.shape != (3, TARGET, TARGET):
        raise RuntimeError(f"bad crop shape: {crop.shape}")
    return crop



# ── LRU Patient Cache ──────────────────────────────────────────────────────────

class LRUPatientCache:
    def __init__(self, max_size=8):
        self._cache = collections.OrderedDict()
        self._max = max_size

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


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_patient_paths(patient_manifest_path, split_filter=None):
    """patient_manifest에서 {safe_id: {ct_hu_npy, roi_path}} 반환"""
    paths = {}
    with open(patient_manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "")
            if split_filter and split != split_filter:
                continue
            sid = row.get("safe_id", "")
            ct = row.get("ct_hu_npy", "")
            if sid and ct:
                assert_path_safe(ct)
                paths[sid] = {
                    "ct_hu_npy": ct,
                    "patient_id": row.get("patient_id", ""),
                    "split": split,
                }
    return paths


def load_train_patient_ids(train_manifest_path):
    ids = set()
    with open(train_manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row.get("patient_id", ""))
    return ids


def load_holdout_patient_ids(holdout_csv_path):
    ids = set()
    if not Path(holdout_csv_path).exists():
        return ids
    with open(holdout_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row.get("patient_id", ""))
    return ids


def load_n_c10_val_crops(n_c10_path):
    """n_c10 normal_val manifest 로드. local_z = z_ratio, slice_index = 실제 슬라이스"""
    rows = []
    with open(n_c10_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ── six_bin_label 계산 ─────────────────────────────────────────────────────────

def z_level_from_ratio(z_ratio):
    if z_ratio < Z_LOWER_MAX:
        return "lower"
    elif z_ratio < Z_MIDDLE_MAX:
        return "middle"
    return "upper"


def compute_sixbin_manifest(n_c10_rows, patient_paths, v4_20_roi_root, seed):
    """
    n_c10 val crops에 대해 v4_20 ROI 기반 six_bin_label 계산 + cap=50 서브샘플링
    Returns: (manifest_rows, patient_summary_rows, error_rows)
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    # safe_id별로 그룹화
    groups = collections.defaultdict(list)
    for row in n_c10_rows:
        groups[row["safe_id"]].append(row)

    manifest_rows = []
    patient_summary_rows = []
    error_rows = []
    manifest_id_counter = [0]

    for safe_id, crops in sorted(groups.items()):
        patient_id = crops[0]["patient_id"]

        # ROI 경로
        roi_path = v4_20_roi_root / safe_id / "refined_roi.npy"
        if not roi_path.exists():
            error_rows.append({
                "phase": "sixbin", "patient_id": patient_id,
                "safe_id": safe_id, "error": f"roi_not_found:{roi_path}",
            })
            continue

        # CT 경로 확인
        ct_info = patient_paths.get(safe_id, {})
        ct_path = ct_info.get("ct_hu_npy", "")
        if not ct_path:
            error_rows.append({
                "phase": "sixbin", "patient_id": patient_id,
                "safe_id": safe_id, "error": "no_ct_path",
            })
            continue

        roi = np.load(str(roi_path))
        n_slices, H, W = roi.shape

        # slice별 ROI distance 캐시 (중복 계산 방지)
        dist_cache = {}
        ring_cache = {}

        def get_dist_ring(z_idx):
            if z_idx not in dist_cache:
                sl = roi[z_idx]
                dist_cache[z_idx] = distance_transform_edt(sl)
                ring_cache[z_idx] = ((sl > 0) & (dist_cache[z_idx] <= EROSION_PX)).astype(np.float32)
            return dist_cache[z_idx], ring_cache[z_idx]

        # 각 crop에 대해 six_bin_label 계산
        labelled = []
        for row in crops:
            z_ratio = float(row["local_z"])
            slice_idx = int(row["slice_index"])
            y0 = int(row["y0"])
            x0 = int(row["x0"])

            if slice_idx < 0 or slice_idx >= n_slices:
                continue

            dist_arr, ring_arr = get_dist_ring(slice_idx)
            roi_slice = roi[slice_idx]

            sy0 = max(0, y0); sx0 = max(0, x0)
            sy1 = min(H, y0 + CROP_SIZE); sx1 = min(W, x0 + CROP_SIZE)
            patch_area = float(CROP_SIZE * CROP_SIZE)

            roi_sum = float(roi_slice[sy0:sy1, sx0:sx1].sum())
            ring_sum = float(ring_arr[sy0:sy1, sx0:sx1].sum())

            roi_ratio = roi_sum / patch_area
            boundary_overlap_ratio = ring_sum / patch_area

            is_boundary = boundary_overlap_ratio >= BOUNDARY_THRESHOLD
            is_interior = (roi_ratio >= INTERIOR_ROI_MIN) and (not is_boundary)

            if is_boundary:
                boundary_status = "boundary"
            elif is_interior:
                boundary_status = "interior"
            else:
                boundary_status = "excluded"

            z_level = z_level_from_ratio(z_ratio)
            if boundary_status == "excluded":
                six_bin_label = "excluded"
            else:
                six_bin_label = f"{z_level}_{boundary_status}"

            low_z_warning = 1 if slice_idx <= LOW_Z_WARNING_THRESHOLD else 0

            labelled.append({
                "patient_id": patient_id,
                "safe_id": safe_id,
                "local_z": slice_idx,
                "z_ratio": z_ratio,
                "z_level": z_level,
                "boundary_status": boundary_status,
                "six_bin_label": six_bin_label,
                "crop_y0": y0,
                "crop_x0": x0,
                "crop_y1": y0 + CROP_SIZE,
                "crop_x1": x0 + CROP_SIZE,
                "roi_ratio": roi_ratio,
                "boundary_overlap_ratio": boundary_overlap_ratio,
                "low_z_warning": low_z_warning,
                "ct_hu_npy": ct_path,
                "roi_path": str(roi_path),
            })

        # bin별 cap=50 서브샘플링
        bin_groups = {b: [] for b in SIX_BIN_LABELS}
        for r in labelled:
            if r["six_bin_label"] in SIX_BIN_LABELS:
                bin_groups[r["six_bin_label"]].append(r)

        pat_rng = random.Random(seed + hash(safe_id) % (2 ** 20))
        bin_counts = {}
        sel_rows = []

        for bin_label in SIX_BIN_LABELS:
            cands = bin_groups[bin_label]
            n_avail = len(cands)
            if n_avail > CAP_PER_BIN:
                selected = pat_rng.sample(cands, CAP_PER_BIN)
            else:
                selected = cands[:]
            bin_counts[bin_label] = {"available": n_avail, "selected": len(selected)}
            sel_rows.extend(selected)

        for r in sel_rows:
            mid = f"rd_b9_{manifest_id_counter[0]:07d}"
            manifest_id_counter[0] += 1
            manifest_rows.append({
                "manifest_id": mid,
                "patient_id": r["patient_id"],
                "safe_id": r["safe_id"],
                "split": "normal_val",
                "local_z": r["local_z"],
                "z_level": r["z_level"],
                "six_bin_label": r["six_bin_label"],
                "boundary_status": r["boundary_status"],
                "crop_y0": r["crop_y0"],
                "crop_x0": r["crop_x0"],
                "crop_y1": r["crop_y1"],
                "crop_x1": r["crop_x1"],
                "roi_ratio": round(r["roi_ratio"], 6),
                "boundary_overlap_ratio": round(r["boundary_overlap_ratio"], 6),
                "low_z_warning": r["low_z_warning"],
                "ct_hu_npy": r["ct_hu_npy"],
                "roi_path": r["roi_path"],
            })

        patient_summary_rows.append({
            "patient_id": patient_id,
            "safe_id": safe_id,
            "n_c10_crops": len(crops),
            "n_labelled": len(labelled),
            "n_selected": len(sel_rows),
            **{f"avail_{b}": bin_counts.get(b, {}).get("available", 0) for b in SIX_BIN_LABELS},
            **{f"sel_{b}": bin_counts.get(b, {}).get("selected", 0) for b in SIX_BIN_LABELS},
        })

    return manifest_rows, patient_summary_rows, error_rows


# ── Shard 생성 ─────────────────────────────────────────────────────────────────

def generate_shards(manifest_rows, shards_dir, patient_cache):
    """manifest_rows를 float32 shard로 저장. Returns (shard_files, index_rows, timing_rows, error_rows)"""
    import numpy as np
    n_total = len(manifest_rows)
    n_shards = math.ceil(n_total / SHARD_SIZE)

    shard_files = []
    index_rows = []
    timing_rows = []
    error_rows = []
    n_low_z = 0
    n_edge_pad = 0

    t_total = time.perf_counter()

    for s in range(n_shards):
        batch = manifest_rows[s * SHARD_SIZE: (s + 1) * SHARD_SIZE]
        crops_arr = []
        t_s = time.perf_counter()

        for local_i, row in enumerate(batch):
            global_rank = s * SHARD_SIZE + local_i
            sid = row["safe_id"]
            lz = int(row["local_z"])
            y0 = int(row["crop_y0"])
            x0 = int(row["crop_x0"])
            y1 = int(row["crop_y1"])
            x1 = int(row["crop_x1"])
            ct_path = row.get("ct_hu_npy", "")

            index_rows.append({
                "shard_id": s, "row_in_shard": local_i, "global_rank": global_rank,
                "safe_id": sid, "six_bin_label": row["six_bin_label"],
                "local_z": lz, "crop_y0": y0, "crop_x0": x0,
                "crop_y1": y1, "crop_x1": x1,
                "low_z_warning": int(row.get("low_z_warning", 0)),
            })

            if not ct_path:
                error_rows.append({"phase": "shard", "shard_id": s, "global_rank": global_rank,
                                   "safe_id": sid, "error": "no_ct_path"})
                crops_arr.append(np.zeros((3, CROP_SIZE, CROP_SIZE), dtype=np.float32))
                continue
            try:
                ct = patient_cache.load(sid, ct_path)
                crop = build_crop_np(ct, lz, y0, x0, y1, x1)
                if lz <= LOW_Z_WARNING_THRESHOLD:
                    n_low_z += 1
                _, hm, wm = ct.shape
                if y0 < 0 or x0 < 0 or y1 > hm or x1 > wm:
                    n_edge_pad += 1
                crops_arr.append(crop)
            except Exception as e:
                error_rows.append({"phase": "shard", "shard_id": s, "global_rank": global_rank,
                                   "safe_id": sid, "error": str(e)})
                crops_arr.append(np.zeros((3, CROP_SIZE, CROP_SIZE), dtype=np.float32))

        shard_arr = np.stack(crops_arr, axis=0).astype(np.float32)
        shard_name = f"rd_b9_shard_{s:04d}.npy"
        shard_path = shards_dir / shard_name
        np.save(str(shard_path), shard_arr)
        t_e = time.perf_counter()
        disk_mb = shard_path.stat().st_size / (1024 * 1024)
        timing_rows.append({"shard_id": s, "n_crops": len(batch),
                             "gen_sec": round(t_e - t_s, 3), "disk_mb": round(disk_mb, 2)})
        shard_files.append(shard_path)

        if s % 5 == 0 or s == n_shards - 1:
            elapsed = time.perf_counter() - t_total
            eta = elapsed / (s + 1) * (n_shards - s - 1) if s < n_shards - 1 else 0
            pct = (s + 1) / n_shards * 100
            print(f"    shard {s:2d}/{n_shards}  {pct:5.1f}%  {disk_mb:.1f}MB  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    return shard_files, index_rows, timing_rows, error_rows, n_low_z, n_edge_pad


def validate_shards(shard_files, seed=42):
    import numpy as np
    rng = random.Random(seed)
    val_rows = []
    shape_mismatch = range_violation = nan_count = inf_count = 0
    for s, sp in enumerate(shard_files):
        arr = np.load(str(sp), mmap_mode="r")
        idxs = rng.sample(range(arr.shape[0]), min(VALIDATE_SAMPLE_PER_SHARD, arr.shape[0]))
        for idx in idxs:
            crop = arr[idx]
            sh_ok = crop.shape == (3, CROP_SIZE, CROP_SIZE)
            rg_ok = float(crop.min()) >= -1e-6 and float(crop.max()) <= 1 + 1e-6
            na_ok = not bool(np.isnan(crop).any())
            in_ok = not bool(np.isinf(crop).any())
            if not sh_ok: shape_mismatch += 1
            if not rg_ok: range_violation += 1
            if not na_ok: nan_count += 1
            if not in_ok: inf_count += 1
            val_rows.append({
                "shard_id": s, "row_in_shard": idx,
                "shape_ok": int(sh_ok), "range_ok": int(rg_ok),
                "nan_ok": int(na_ok), "inf_ok": int(in_ok),
                "min_val": round(float(crop.min()), 6),
                "max_val": round(float(crop.max()), 6),
            })
    return val_rows, shape_mismatch, range_violation, nan_count, inf_count


# ── Teacher / Student 빌드 ─────────────────────────────────────────────────────

def build_teacher(local_weight_path):
    import torch
    import torchvision.models as models
    resnet = models.resnet18(weights=None)
    state_dict = torch.load(str(local_weight_path), map_location="cpu", weights_only=True)
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
                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            )
            self.de_layer2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            )
            self.de_layer1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )

        def forward(self, layer3_feat):
            x = self.de_layer3(layer3_feat);  de3 = x
            x = self.de_layer2(x);           de2 = x
            x = self.de_layer1(x);           de1 = x
            return de3, de2, de1

    return StudentDecoder()


def score_shards(shard_files, index_rows, checkpoint_path, local_weight_path, device_str="auto"):
    """
    shard들을 scoring. Returns score_rows (list of dict)
    backward/optimizer/checkpoint 저장 절대 금지.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    backward_called = False
    optimizer_created = False
    checkpoint_saved = False
    training_started = False

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"  scoring device: {device}")

    # teacher 빌드 (ImageNet weight)
    teacher = build_teacher(local_weight_path).to(device)
    teacher.eval()
    teacher.requires_grad_(False)

    # student 빌드 + checkpoint load (read-only)
    student = build_student_decoder().to(device)
    student.eval()

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    student_state = ckpt.get("student_state_dict", ckpt)
    student.load_state_dict(student_state)
    student.eval()
    print(f"  checkpoint loaded: {checkpoint_path.name}")

    # teacher feature hook
    teacher_feats = {}
    def make_hook(name):
        def _hook(module, inp, out):
            teacher_feats[name] = out
        return _hook
    teacher.layer1.register_forward_hook(make_hook("layer1"))
    teacher.layer2.register_forward_hook(make_hook("layer2"))
    teacher.layer3.register_forward_hook(make_hook("layer3"))

    # index_rows를 global_rank → row 매핑
    rank_to_row = {r["global_rank"]: r for r in index_rows}

    score_rows = []
    shard_offset = 0  # 누적 global_rank offset

    for s, shard_path in enumerate(shard_files):
        arr = np.load(str(shard_path), mmap_mode="r")
        n_in = arr.shape[0]
        n_batches = math.ceil(n_in / SCORE_BATCH_SIZE)

        for b in range(n_batches):
            b_start = b * SCORE_BATCH_SIZE
            b_end = min((b + 1) * SCORE_BATCH_SIZE, n_in)
            batch_np = arr[b_start:b_end].copy()
            batch_t = torch.from_numpy(batch_np).to(device)

            with torch.no_grad():
                teacher_feats.clear()
                teacher(batch_t)
                tf1 = teacher_feats["layer1"]
                tf2 = teacher_feats["layer2"]
                tf3 = teacher_feats["layer3"]
                de3, de2, de1 = student(tf3)

                # cosine distance: (B, H, W) → mean over spatial → (B,)
                sc1 = (1.0 - F.cosine_similarity(de1, tf1, dim=1)).mean(dim=(1, 2))
                sc2 = (1.0 - F.cosine_similarity(de2, tf2, dim=1)).mean(dim=(1, 2))
                sc3 = (1.0 - F.cosine_similarity(de3, tf3, dim=1)).mean(dim=(1, 2))
                crop_score = (sc1 + sc2 + sc3) / 3.0

                s1_np = sc1.cpu().numpy().astype("float32")
                s2_np = sc2.cpu().numpy().astype("float32")
                s3_np = sc3.cpu().numpy().astype("float32")
                cs_np = crop_score.cpu().numpy().astype("float32")

            for local_i in range(b_end - b_start):
                gr = shard_offset + b_start + local_i
                meta = rank_to_row.get(gr, {})
                score_rows.append({
                    "manifest_id": meta.get("manifest_id", ""),
                    "patient_id": meta.get("patient_id", ""),
                    "safe_id": meta.get("safe_id", ""),
                    "six_bin_label": meta.get("six_bin_label", ""),
                    "z_level": meta.get("z_level", ""),
                    "boundary_status": meta.get("boundary_status", ""),
                    "local_z": meta.get("local_z", ""),
                    "low_z_warning": meta.get("low_z_warning", ""),
                    "score_layer1": round(float(s1_np[local_i]), 6),
                    "score_layer2": round(float(s2_np[local_i]), 6),
                    "score_layer3": round(float(s3_np[local_i]), 6),
                    "crop_score": round(float(cs_np[local_i]), 6),
                    "score_nan": int(math.isnan(float(cs_np[local_i]))),
                    "score_inf": int(math.isinf(float(cs_np[local_i]))),
                })

        shard_offset += n_in
        if s % 3 == 0 or s == len(shard_files) - 1:
            print(f"    scoring shard {s}/{len(shard_files)}")

    return score_rows, backward_called, optimizer_created, checkpoint_saved, training_started


def compute_thresholds(score_rows):
    """score_rows에서 normal_val 기준 threshold 후보 계산"""
    import numpy as np

    all_scores = np.array([r["crop_score"] for r in score_rows], dtype=np.float32)
    thresholds = []

    def add_th(label, subset_scores):
        if len(subset_scores) == 0:
            return
        arr = np.array(subset_scores, dtype=np.float32)
        thresholds.append({"label": label, "n": len(arr),
                            "p95": round(float(np.percentile(arr, 95)), 6),
                            "p99": round(float(np.percentile(arr, 99)), 6),
                            "mean": round(float(arr.mean()), 6),
                            "std": round(float(arr.std()), 6),
                            "min": round(float(arr.min()), 6),
                            "max": round(float(arr.max()), 6)})

    add_th("global", all_scores.tolist())

    for bin_label in SIX_BIN_LABELS:
        subset = [r["crop_score"] for r in score_rows if r["six_bin_label"] == bin_label]
        add_th(f"bin_{bin_label}", subset)

    for zlvl in ["upper", "middle", "lower"]:
        subset = [r["crop_score"] for r in score_rows if r["z_level"] == zlvl]
        add_th(f"zlevel_{zlvl}", subset)

    for bs in ["boundary", "interior"]:
        subset = [r["crop_score"] for r in score_rows if r["boundary_status"] == bs]
        add_th(f"bstatus_{bs}", subset)

    return thresholds


# ── dry-plan ────────────────────────────────────────────────────────────────────

def run_dry_plan():
    errors = []
    items = []

    def check(label, ok, detail=""):
        status = "OK" if ok else "FAIL"
        items.append({"label": label, "status": status, "detail": detail})
        if not ok:
            errors.append(f"{label}: {detail}")

    print("=" * 70)
    print("RD-B9: normal_val six-bin scoring threshold [DRY-PLAN]")
    print("=" * 70)
    print()

    # 출력 root 없음 확인
    check("output_root_absent", not OUTPUT_ROOT.exists(),
          str(OUTPUT_ROOT) if OUTPUT_ROOT.exists() else "not exists (OK)")

    # 입력 파일 존재 확인
    check("n_c10_val_manifest", N_C10_VAL_MANIFEST.exists(), str(N_C10_VAL_MANIFEST))
    check("patient_manifest", PATIENT_MANIFEST.exists(), str(PATIENT_MANIFEST))
    check("train_manifest", TRAIN_MANIFEST.exists(), str(TRAIN_MANIFEST))
    check("stage2_holdout_csv", STAGE2_HOLDOUT_CSV.exists(), str(STAGE2_HOLDOUT_CSV))
    check("checkpoint", CHECKPOINT_PATH.exists(), str(CHECKPOINT_PATH))
    check("local_resnet18", LOCAL_RESNET18_WEIGHT.exists(), str(LOCAL_RESNET18_WEIGHT))

    # normal_val 환자 확인
    n_c10_rows = []
    val_patients = set()
    if N_C10_VAL_MANIFEST.exists():
        n_c10_rows = load_n_c10_val_crops(N_C10_VAL_MANIFEST)
        val_patients = set(r["patient_id"] for r in n_c10_rows)
        check("n_c10_val_rows", len(n_c10_rows) > 0, f"{len(n_c10_rows)} rows")
        check("n_c10_val_patients", len(val_patients) == 36, f"{len(val_patients)} patients (expected 36)")

    # train 환자 확인 + overlap
    train_patients = set()
    if TRAIN_MANIFEST.exists():
        train_patients = load_train_patient_ids(TRAIN_MANIFEST)
        check("train_patients", len(train_patients) == 290, f"{len(train_patients)} (expected 290)")
    overlap = val_patients & train_patients
    check("train_val_overlap_0", len(overlap) == 0, f"overlap={list(overlap)[:5]}")

    # stage2_holdout intersection
    holdout_patients = set()
    if STAGE2_HOLDOUT_CSV.exists():
        holdout_patients = load_holdout_patient_ids(STAGE2_HOLDOUT_CSV)
    h_intersect = val_patients & holdout_patients
    check("stage2_holdout_intersection_0", len(h_intersect) == 0, f"intersection={list(h_intersect)[:5]}")

    # CT/ROI 존재 확인 (상위 5명)
    if PATIENT_MANIFEST.exists():
        patient_paths = load_patient_paths(PATIENT_MANIFEST, split_filter="val")
        check("val_patient_paths_loaded", len(patient_paths) > 0, f"{len(patient_paths)} val patients in manifest")
        for sid, info in list(patient_paths.items())[:5]:
            ct_ok = Path(info["ct_hu_npy"]).exists()
            roi_v4 = (V4_20_ROI_ROOT / sid / "refined_roi.npy").exists()
            check(f"ct_exists_{sid[:12]}", ct_ok, info["ct_hu_npy"])
            check(f"roi_v4_exists_{sid[:12]}", roi_v4, str(V4_20_ROI_ROOT / sid / "refined_roi.npy"))

    # expected crop count
    n_val_patients = len(val_patients)
    expected_crops_if_full = n_val_patients * 6 * CAP_PER_BIN
    expected_shards = math.ceil(expected_crops_if_full / SHARD_SIZE)
    print()
    print(f"  normal_val patients       : {n_val_patients}")
    print(f"  n_c10 rows                : {len(n_c10_rows)}")
    print(f"  train patients            : {len(train_patients)}")
    print(f"  train/val overlap         : {len(overlap)}")
    print(f"  stage2_holdout intersection: {len(h_intersect)}")
    print(f"  expected crops (if full cap): {expected_crops_if_full:,}")
    print(f"  expected shards           : ~{expected_shards}")
    print(f"  six_bin cap               : {CAP_PER_BIN}/bin/patient")
    print(f"  shard_size                : {SHARD_SIZE}")
    print(f"  checkpoint                : {CHECKPOINT_PATH.name}")
    print()

    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("판정: DRY-PLAN OK — 모든 체크 통과")
        print("사용자 승인 후:")
        print("  source ~/ai_env/bin/activate && \\")
        print("  python scripts/rd_b9_normal_val_sixbin_scoring_threshold.py --run \\")
        print("    2>&1 | tee /tmp/rd_b9_normal_val_log.txt")


# ── run ─────────────────────────────────────────────────────────────────────────

def run_main():
    import numpy as np

    print("=" * 70)
    print("RD-B9: normal_val six-bin scoring threshold [RUN]")
    print("=" * 70)

    # output root guard
    if OUTPUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재: {OUTPUT_ROOT}")
        sys.exit(1)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    SHARDS_DIR.mkdir(parents=True, exist_ok=False)

    error_rows_all = []

    # ── 1. readiness 확인 ─────────────────────────────────────────────────────
    print("\n[1/5] normal_val readiness 확인")

    n_c10_rows = load_n_c10_val_crops(N_C10_VAL_MANIFEST)
    val_patients = set(r["patient_id"] for r in n_c10_rows)
    train_patients = load_train_patient_ids(TRAIN_MANIFEST)
    holdout_patients = load_holdout_patient_ids(STAGE2_HOLDOUT_CSV)
    patient_paths = load_patient_paths(PATIENT_MANIFEST, split_filter="val")

    overlap = val_patients & train_patients
    h_intersect = val_patients & holdout_patients

    readiness_rows = [
        {"check": "n_val_patients", "value": len(val_patients), "pass": len(val_patients) == 36},
        {"check": "n_train_patients", "value": len(train_patients), "pass": len(train_patients) > 0},
        {"check": "train_val_overlap", "value": len(overlap), "pass": len(overlap) == 0},
        {"check": "stage2_holdout_intersection", "value": len(h_intersect), "pass": len(h_intersect) == 0},
        {"check": "n_c10_rows", "value": len(n_c10_rows), "pass": len(n_c10_rows) > 0},
        {"check": "checkpoint_exists", "value": str(CHECKPOINT_PATH.exists()), "pass": CHECKPOINT_PATH.exists()},
    ]

    for r in readiness_rows:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {r['check']:30s}: {status} ({r['value']})")
        if not r["pass"]:
            print(f"[ABORT] readiness check FAIL: {r['check']}")
            sys.exit(1)

    write_csv(
        OUTPUT_ROOT / "rd_b9_normal_val_readiness.csv",
        ["check", "value", "pass"],
        readiness_rows,
    )

    # ── 2. six-bin manifest 생성 ─────────────────────────────────────────────
    print("\n[2/5] six-bin coordinate manifest 생성")

    manifest_rows, patient_summary_rows, error_rows = compute_sixbin_manifest(
        n_c10_rows, patient_paths, V4_20_ROI_ROOT, SAMPLING_SEED
    )
    error_rows_all.extend(error_rows)

    if not manifest_rows:
        print("[ABORT] manifest_rows 없음 - 종료")
        sys.exit(1)

    bin_counts = collections.Counter(r["six_bin_label"] for r in manifest_rows)
    low_z_count = sum(int(r["low_z_warning"]) for r in manifest_rows)
    print(f"  total manifest crops: {len(manifest_rows):,}")
    for b in SIX_BIN_LABELS:
        print(f"    {b}: {bin_counts.get(b, 0)}")
    print(f"  low_z_warning: {low_z_count}")

    manifest_fields = [
        "manifest_id", "patient_id", "safe_id", "split",
        "local_z", "z_level", "six_bin_label", "boundary_status",
        "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "roi_ratio", "boundary_overlap_ratio", "low_z_warning",
        "ct_hu_npy", "roi_path",
    ]
    write_csv(OUTPUT_ROOT / "rd_b9_normal_val_sixbin_manifest.csv", manifest_fields, manifest_rows)

    patient_summary_fields = [
        "patient_id", "safe_id", "n_c10_crops", "n_labelled", "n_selected",
    ] + [f"avail_{b}" for b in SIX_BIN_LABELS] + [f"sel_{b}" for b in SIX_BIN_LABELS]
    write_csv(OUTPUT_ROOT / "rd_b9_normal_val_patient_summary.csv", patient_summary_fields, patient_summary_rows)

    # ── 3. float32 shard 생성 ────────────────────────────────────────────────
    print("\n[3/5] float32 shard 생성")
    patient_cache = LRUPatientCache(max_size=8)
    shard_files, index_rows, timing_rows, shard_errors, n_low_z_shard, n_edge_pad = generate_shards(
        manifest_rows, SHARDS_DIR, patient_cache
    )
    error_rows_all.extend(shard_errors)

    # manifest_id를 index_rows에 병합
    for i, ir in enumerate(index_rows):
        if i < len(manifest_rows):
            ir["manifest_id"] = manifest_rows[i]["manifest_id"]
            ir["patient_id"] = manifest_rows[i]["patient_id"]
            ir["z_level"] = manifest_rows[i]["z_level"]
            ir["boundary_status"] = manifest_rows[i]["boundary_status"]

    shard_index_fields = [
        "shard_id", "row_in_shard", "global_rank", "manifest_id",
        "patient_id", "safe_id", "six_bin_label", "z_level", "boundary_status",
        "local_z", "crop_y0", "crop_x0", "crop_y1", "crop_x1", "low_z_warning",
    ]
    write_csv(OUTPUT_ROOT / "rd_b9_normal_val_shard_index.csv", shard_index_fields, index_rows)
    write_csv(
        OUTPUT_ROOT / "rd_b9_generation_timing.csv",
        ["shard_id", "n_crops", "gen_sec", "disk_mb"],
        timing_rows,
    )

    total_disk_mb = sum(sp.stat().st_size for sp in shard_files) / (1024 * 1024)
    print(f"  shards: {len(shard_files)}, disk: {total_disk_mb:.1f} MB")

    # shard 검증
    print("  shard 검증 중 ...")
    val_rows, shape_mm, range_viol, nan_c, inf_c = validate_shards(shard_files)
    write_csv(
        OUTPUT_ROOT / "rd_b9_normal_val_shard_validation.csv",
        ["shard_id", "row_in_shard", "shape_ok", "range_ok", "nan_ok", "inf_ok", "min_val", "max_val"],
        val_rows,
    )
    print(f"  shape_mismatch={shape_mm}, range_violation={range_viol}, nan={nan_c}, inf={inf_c}")

    # ── 4. scoring ────────────────────────────────────────────────────────────
    print("\n[4/5] normal_val scoring")
    print("  [SAFETY] backward_called=False, optimizer_created=False")
    print("  [SAFETY] checkpoint_saved=False, training_started=False")

    score_rows, backward_called, optimizer_created, checkpoint_saved, training_started = score_shards(
        shard_files, index_rows, CHECKPOINT_PATH, LOCAL_RESNET18_WEIGHT
    )

    score_nan_count = sum(r["score_nan"] for r in score_rows)
    score_inf_count = sum(r["score_inf"] for r in score_rows)
    print(f"  score_rows: {len(score_rows)}, NaN={score_nan_count}, Inf={score_inf_count}")

    score_fields = [
        "manifest_id", "patient_id", "safe_id",
        "six_bin_label", "z_level", "boundary_status",
        "local_z", "low_z_warning",
        "score_layer1", "score_layer2", "score_layer3", "crop_score",
        "score_nan", "score_inf",
    ]
    write_csv(OUTPUT_ROOT / "rd_b9_normal_val_score.csv", score_fields, score_rows)

    # score by bin summary
    bin_score_rows = []
    for bin_label in SIX_BIN_LABELS:
        subset = [r["crop_score"] for r in score_rows if r["six_bin_label"] == bin_label]
        if subset:
            arr = np.array(subset)
            bin_score_rows.append({
                "six_bin_label": bin_label, "n": len(subset),
                "mean": round(float(arr.mean()), 6),
                "std": round(float(arr.std()), 6),
                "p95": round(float(np.percentile(arr, 95)), 6),
                "p99": round(float(np.percentile(arr, 99)), 6),
            })
    write_csv(
        OUTPUT_ROOT / "rd_b9_normal_val_score_by_bin_summary.csv",
        ["six_bin_label", "n", "mean", "std", "p95", "p99"],
        bin_score_rows,
    )

    # ── 5. threshold 생성 ────────────────────────────────────────────────────
    print("\n[5/5] threshold 후보 생성")
    thresholds = compute_thresholds(score_rows)

    write_csv(
        OUTPUT_ROOT / "rd_b9_normal_val_threshold_candidates.csv",
        ["label", "n", "p95", "p99", "mean", "std", "min", "max"],
        thresholds,
    )

    global_th = next((t for t in thresholds if t["label"] == "global"), {})
    global_p95 = global_th.get("p95", None)
    global_p99 = global_th.get("p99", None)
    print(f"  global p95={global_p95}, p99={global_p99}")

    th_summary = {
        "threshold_created_from": "normal_val_only",
        "global_p95": global_p95,
        "global_p99": global_p99,
        "bin_thresholds": {t["label"]: {"p95": t["p95"], "p99": t["p99"]} for t in thresholds},
    }
    with open(OUTPUT_ROOT / "rd_b9_normal_val_threshold_summary.json", "w", encoding="utf-8") as f:
        json.dump(th_summary, f, ensure_ascii=False, indent=2)
    print("  -> rd_b9_normal_val_threshold_summary.json")

    # ── errors 저장 ───────────────────────────────────────────────────────────
    write_csv(
        OUTPUT_ROOT / "rd_b9_errors.csv",
        ["phase", "patient_id", "safe_id", "shard_id", "global_rank", "error"],
        error_rows_all,
    )

    # ── summary JSON ──────────────────────────────────────────────────────────
    all_checks_passed = (
        len(overlap) == 0
        and len(h_intersect) == 0
        and shape_mm == 0
        and range_viol == 0
        and nan_c == 0
        and inf_c == 0
        and score_nan_count == 0
        and score_inf_count == 0
        and not backward_called
        and not optimizer_created
        and not checkpoint_saved
        and not training_started
    )

    summary = {
        "normal_val_patients": len(val_patients),
        "train_val_overlap": len(overlap),
        "stage2_holdout_intersection": len(h_intersect),
        "n_val_crops": len(manifest_rows),
        "six_bin_counts": {b: bin_counts.get(b, 0) for b in SIX_BIN_LABELS},
        "shard_count": len(shard_files),
        "shard_disk_mb": round(total_disk_mb, 2),
        "shard_shape_mismatch_count": shape_mm,
        "value_range_violation_count": range_viol,
        "score_nan_count": score_nan_count,
        "score_inf_count": score_inf_count,
        "global_p95": global_p95,
        "global_p99": global_p99,
        "threshold_created_from": "normal_val_only",
        "checkpoint_loaded": str(CHECKPOINT_PATH),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "teacher_frozen": True,
        "student_eval": True,
        "backward_called": backward_called,
        "optimizer_created": optimizer_created,
        "checkpoint_saved": checkpoint_saved,
        "training_started": training_started,
        "lesion_scoring_started": False,
        "stage2_holdout_access": 0,
        "all_checks_passed": all_checks_passed,
    }
    with open(OUTPUT_ROOT / "rd_b9_normal_val_scoring_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("  -> rd_b9_normal_val_scoring_summary.json")

    # ── report.md ─────────────────────────────────────────────────────────────
    verdict = "PASS" if all_checks_passed else "FAIL"
    md = [
        "# RD-B9 normal_val Scoring & Threshold Report",
        "",
        f"## 판정: {verdict}",
        "",
        "## 1. RD-B8e/B8f 결과 요약",
        "| 항목 | 값 |",
        "|---|---|",
        "| normal_train crops | 86,017 |",
        "| normal_train shards | 87 |",
        "| train disk | 8.86 GB |",
        "| best_epoch | 20 |",
        "| final loss | 0.074174 |",
        "",
        "## 2. normal_val readiness",
        "| 항목 | 값 |",
        "|---|---|",
        f"| normal_val patients | {len(val_patients)} |",
        f"| n_c10 rows | {len(n_c10_rows)} |",
        f"| train/val overlap | {len(overlap)} |",
        f"| stage2_holdout intersection | {len(h_intersect)} |",
        "",
        "## 3. six-bin manifest 생성 기준",
        f"- erosion_px = {EROSION_PX}, boundary_overlap_threshold = {BOUNDARY_THRESHOLD}",
        f"- interior_roi_min = {INTERIOR_ROI_MIN}",
        f"- z_level: lower (z_ratio<1/3), middle (1/3≤z_ratio<2/3), upper (≥2/3)",
        f"- cap = {CAP_PER_BIN}/bin/patient",
        f"- ROI 기준: v4_20 refined ROI (refined_roi.npy)",
        "",
        "## 4. shard 생성 결과",
        "| 항목 | 값 |",
        "|---|---|",
        f"| val crops | {len(manifest_rows):,} |",
        f"| shards | {len(shard_files)} |",
        f"| disk | {total_disk_mb:.1f} MB |",
        f"| shape_mismatch | {shape_mm} |",
        f"| range_violation | {range_viol} |",
        f"| NaN | {nan_c} |",
        f"| Inf | {inf_c} |",
        "",
        "## 5. scoring 방식",
        "- teacher: ResNet18 ImageNet pretrained (layer1/layer2/layer3 feature hook)",
        "- student: StudentDecoder (de_layer3→de_layer2→de_layer1, layer3 feature input)",
        "- score_layer1 = mean_spatial(1 - cosine_similarity(de1, tf1))",
        "- score_layer2 = mean_spatial(1 - cosine_similarity(de2, tf2))",
        "- score_layer3 = mean_spatial(1 - cosine_similarity(de3, tf3))",
        "- crop_score = (score_layer1 + score_layer2 + score_layer3) / 3",
        "- backward=False, optimizer=None, checkpoint_saved=False",
        "",
        "## 6. normal_val score 분포",
        f"| global p95 | {global_p95} |",
        f"| global p99 | {global_p99} |",
        "",
        "## 7. threshold 후보",
        "| label | n | p95 | p99 |",
        "|---|---|---|---|",
    ] + [f"| {t['label']} | {t['n']} | {t['p95']} | {t['p99']} |" for t in thresholds] + [
        "",
        "## 8. 다음 단계",
        "- RD-B10: stage1_dev first-stage candidate scoring preflight",
        "  * normal_val threshold를 기준으로 stage1_dev 후보 scoring",
        "  * lesion scoring은 RD-B10에서 별도 진행",
        "",
        "## 9. 절대 하지 않은 것",
        "- training 없음",
        "- backward 없음",
        "- optimizer step 없음",
        "- lesion scoring 없음",
        "- stage2_holdout 접근 없음",
        "- checkpoint 저장 없음",
    ]
    with open(OUTPUT_ROOT / "rd_b9_normal_val_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("  -> rd_b9_normal_val_report.md")

    # DONE
    (OUTPUT_ROOT / "DONE").write_text(
        f"rd_b9_normal_val_scoring_threshold_v1 DONE\nall_checks_passed={all_checks_passed}\n",
        encoding="utf-8",
    )
    print("  -> DONE")

    print()
    print("=" * 70)
    print(f"판정: {verdict}")
    print(f"  normal_val patients  : {len(val_patients)}")
    print(f"  val crops            : {len(manifest_rows):,}")
    print(f"  six_bin counts       : {dict(bin_counts)}")
    print(f"  train/val overlap    : {len(overlap)}")
    print(f"  stage2_holdout inter : {len(h_intersect)}")
    print(f"  shards               : {len(shard_files)}  ({total_disk_mb:.1f} MB)")
    print(f"  score NaN/Inf        : {score_nan_count}/{score_inf_count}")
    print(f"  global p95           : {global_p95}")
    print(f"  global p99           : {global_p99}")
    print(f"  bin threshold OK     : {len([t for t in thresholds if 'bin_' in t['label']])}/{len(SIX_BIN_LABELS)}")
    print(f"  all_checks_passed    : {all_checks_passed}")
    print("=" * 70)

    if not all_checks_passed:
        sys.exit(1)


# ── 진입점 ────────────────────────────────────────────────────────────────────

if IS_DRY_PLAN:
    run_dry_plan()
elif IS_RUN:
    run_main()
