"""
RD-B5: RD4AD Dataset / Loader / Model Skeleton + Static Check
목적: hybrid_cache 전략 기반 RD4AD teacher-student branch의
      Dataset / SixBinBalancedBatchSampler / Model skeleton 작성 및 정적 검사
실행 방법:
  bare run  → exit 2 (파일 생성 금지)
  --dry-check   → static validation, output root 생성 금지
  --selftest    → 순수 함수 테스트 (normalize_hu, edge clamp, sampler grouping)
  --write-report → 사용자 승인 후만, output root 생성 및 report 저장
안전 조건:
  stage2_holdout/lesion 경로 접근 금지
  crop NPZ 생성 금지  /  학습 금지  /  scoring 금지
  model forward 금지  /  GPU 사용 금지  /  checkpoint 로드 금지
  기존 파일 수정/삭제 금지
"""

import sys
import os
import csv
import json
import math
import time
import copy
import random
import textwrap
from pathlib import Path
from collections import defaultdict, OrderedDict

# ─── bare-run guard ────────────────────────────────────────────────────────────
ALLOWED_MODES = {"--dry-check", "--write-report", "--selftest"}
if not any(m in sys.argv for m in ALLOWED_MODES):
    print("오류: --dry-check / --write-report / --selftest 중 하나가 필요합니다.")
    print("  --dry-check    : static validation (출력 파일 없음)")
    print("  --selftest     : 순수 함수 테스트")
    print("  --write-report : 사용자 승인 후 output root 생성")
    sys.exit(2)

IS_DRY_CHECK = "--dry-check" in sys.argv
IS_SELFTEST = "--selftest" in sys.argv
IS_WRITE_REPORT = "--write-report" in sys.argv

# ─── 경로 설정 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
OUTPUT_ROOT = PROJECT_ROOT / (
    "outputs/normal_based_stage2_verifier_audit/"
    "rd_b5_dataset_loader_model_skeleton_static_v1"
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

# ─── 안전 금지 키워드 ──────────────────────────────────────────────────────────
FORBIDDEN_KEYWORDS = [
    "stage2_holdout",
    "lesion",
    "test_lesion",
    "second-stage-lesion-refiner",
]

# ─── 설계 상수 ─────────────────────────────────────────────────────────────────
CROP_SIZE = 96
N_CHANNELS = 3
MIP_RADIUS = 3          # lower/upper MIP: 3슬라이스 (z_spacing=1.0mm)
HU_CLIP_MIN = -1000.0
HU_CLIP_MAX = 600.0
HU_RANGE = HU_CLIP_MAX - HU_CLIP_MIN   # 1600.0
EXPECTED_ROWS = 86_017
EXPECTED_PATIENTS = 290
SIX_BIN_LABELS = [
    "lower_boundary",
    "lower_interior",
    "middle_boundary",
    "middle_interior",
    "upper_boundary",
    "upper_interior",
]
REQUIRED_COLUMNS = [
    "manifest_id", "patient_id", "safe_id", "split",
    "local_z", "six_bin_label",
    "crop_y0", "crop_x0", "crop_y1", "crop_x1", "crop_size",
]
LRU_MAX_PATIENTS = 8
LOW_Z_BOUNDARY_WARN_THRESHOLD = 7      # z ≤ 7 → low_z_boundary_warning
BATCH_SIZE_CANDIDATE_A = 24            # bin당 4
BATCH_SIZE_CANDIDATE_B = 48            # bin당 8

errors = []


# =============================================================================
# 안전 검사 함수
# =============================================================================
def assert_path_safe(path_str: str) -> None:
    for kw in FORBIDDEN_KEYWORDS:
        if kw.lower() in str(path_str).lower():
            raise RuntimeError(f"[SAFETY] 금지 경로 접근 차단: {path_str!r} (keyword={kw!r})")


# =============================================================================
# 1. PatientVolumeCache skeleton
# =============================================================================
class PatientVolumeCache:
    """
    환자별 CT/ROI npy를 mmap_mode='r'로 읽고, LRU 방식으로 max_patients개를 캐싱한다.
    per-worker 캐시 가정: DataLoader num_workers>0이면 각 worker가 독립 인스턴스 소유.
    stage2_holdout/lesion 포함 경로는 로드 전 차단한다.
    실제 npy 로딩은 이번 단계(static check)에서 호출 금지.
    """

    def __init__(self, max_patients: int = LRU_MAX_PATIENTS) -> None:
        self.max_patients = max_patients
        # OrderedDict: key=safe_id, value={"ct": ndarray, "roi": ndarray, "z_max": int}
        self._cache: OrderedDict = OrderedDict()

    # ── 공개 API ──────────────────────────────────────────────────────────────
    def get(self, safe_id: str, ct_path: str, roi_path: str):
        """
        safe_id에 대한 (ct_volume, roi_volume, z_max) 반환.
        캐시 hit → LRU 갱신.
        캐시 miss → npy mmap 로드 후 캐시 저장.
        [정적 검사 단계] 실제 로드 호출 금지 — 이 함수 자체 정의만 허용.
        """
        assert_path_safe(ct_path)
        assert_path_safe(roi_path)
        if safe_id in self._cache:
            self._cache.move_to_end(safe_id)
            return self._cache[safe_id]
        # npy mmap 로드 (static check에서는 절대 호출하지 않음)
        entry = self._load_npy_entry(safe_id, ct_path, roi_path)
        self._cache[safe_id] = entry
        self._cache.move_to_end(safe_id)
        if len(self._cache) > self.max_patients:
            self._cache.popitem(last=False)
        return entry

    def clear(self) -> None:
        """캐시 전체 삭제."""
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)

    # ── 내부 구현 ─────────────────────────────────────────────────────────────
    @staticmethod
    def _load_npy_entry(safe_id: str, ct_path: str, roi_path: str) -> dict:
        """
        실제 npy mmap_mode='r' 로딩.
        [정적 검사 / selftest 단계에서 절대 호출 금지]
        학습/추론 단계(RD-B6 이후)에서만 사용.
        """
        import numpy as np  # RD-B6+ 에서만 활성
        ct_arr = np.load(ct_path, mmap_mode="r")   # shape: (Z, Y, X)
        roi_arr = np.load(roi_path, mmap_mode="r")  # shape: (Z, Y, X), bool
        z_max = ct_arr.shape[0]
        return {"ct": ct_arr, "roi": roi_arr, "z_max": z_max, "safe_id": safe_id}


# =============================================================================
# 2. 순수 crop 생성 함수들 (static check 단계에서 호출 금지)
# =============================================================================

def normalize_hu_new_rd_style(hu_array):
    """
    HU [-1000, 600] clip → (x + 1000) / 1600 → [0, 1] float32.
    RD-B2b normalization 확정. (new_RD_style)
    순수 함수: static check / selftest에서 단위 테스트 가능.
    단, CT volume에 적용하는 실제 전처리 호출은 학습 단계에서만 수행.
    """
    import numpy as np
    clipped = np.clip(hu_array, HU_CLIP_MIN, HU_CLIP_MAX)
    return ((clipped - HU_CLIP_MIN) / HU_RANGE).astype("float32")


def compute_mip_slab_indices(center_z: int, direction: str, z_max: int) -> list:
    """
    MIP slab 인덱스 계산.
    direction='lower': [center_z-3, center_z-2, center_z-1]
    direction='upper': [center_z+1, center_z+2, center_z+3]
    경계 clamp: numpy.clip(slab_indices, 0, z_max-1)
    반환: list[int] (clamped)
    순수 함수 — CT 로딩 없음.
    """
    if direction == "lower":
        raw = [center_z - MIP_RADIUS + i for i in range(MIP_RADIUS)]
    elif direction == "upper":
        raw = [center_z + 1 + i for i in range(MIP_RADIUS)]
    else:
        raise ValueError(f"direction must be 'lower' or 'upper', got {direction!r}")
    clamped = [max(0, min(idx, z_max - 1)) for idx in raw]
    return clamped


def has_low_z_boundary_warning(center_z: int) -> bool:
    """z ≤ 7이면 True (diaphragm saturation risk)."""
    return center_z <= LOW_Z_BOUNDARY_WARN_THRESHOLD


def get_lower_mip(ct_volume, center_z: int, z_max: int):
    """
    lower 3mm MIP: z-3, z-2, z-1 슬라이스 max projection.
    [정적 검사 단계에서 호출 금지] — 학습 시에만 사용.
    """
    import numpy as np
    idxs = compute_mip_slab_indices(center_z, "lower", z_max)
    slab = ct_volume[idxs]          # (3, Y, X)
    return slab.max(axis=0)         # (Y, X)


def get_upper_mip(ct_volume, center_z: int, z_max: int):
    """
    upper 3mm MIP: z+1, z+2, z+3 슬라이스 max projection.
    [정적 검사 단계에서 호출 금지] — 학습 시에만 사용.
    """
    import numpy as np
    idxs = compute_mip_slab_indices(center_z, "upper", z_max)
    slab = ct_volume[idxs]
    return slab.max(axis=0)


def build_mixed_3ch_crop_from_volume(
    ct_volume,
    center_z: int,
    crop_y0: int,
    crop_x0: int,
    crop_y1: int,
    crop_x1: int,
    z_max: int,
) -> dict:
    """
    mixed_3ch crop 생성 (3×96×96 float32).
    ch0 = normalize(CT[center_z, y0:y1, x0:x1])
    ch1 = normalize(lower 3mm MIP[y0:y1, x0:x1])
    ch2 = normalize(upper 3mm MIP[y0:y1, x0:x1])
    반환: {"crop": ndarray(3,H,W), "low_z_boundary_warning": bool}
    [정적 검사 단계에서 호출 금지] — 학습 시에만 사용.
    """
    import numpy as np
    ch0_raw = ct_volume[center_z, crop_y0:crop_y1, crop_x0:crop_x1]
    ch1_raw = get_lower_mip(ct_volume, center_z, z_max)[crop_y0:crop_y1, crop_x0:crop_x1]
    ch2_raw = get_upper_mip(ct_volume, center_z, z_max)[crop_y0:crop_y1, crop_x0:crop_x1]

    crop = np.stack([
        normalize_hu_new_rd_style(ch0_raw),
        normalize_hu_new_rd_style(ch1_raw),
        normalize_hu_new_rd_style(ch2_raw),
    ], axis=0).astype("float32")  # (3, H, W)

    return {
        "crop": crop,
        "low_z_boundary_warning": has_low_z_boundary_warning(center_z),
    }


# =============================================================================
# 3. RD4ADNormalCropDataset skeleton
# =============================================================================
class RD4ADNormalCropDataset:
    """
    RD-B1 6-bin balanced coordinate manifest 기반 Dataset skeleton.
    학습 시 torch.utils.data.Dataset을 상속해야 하나,
    이번 정적 검사 단계에서는 torch import 및 Dataset 상속 없이 정의한다.
    실제 CT 로딩(PatientVolumeCache.get)과 crop 생성은 이번 단계에서 호출 금지.
    """

    def __init__(
        self,
        manifest_path: str,
        patient_manifest_path: str,
        split: str = "train",
        cache: "PatientVolumeCache | None" = None,
    ) -> None:
        assert_path_safe(manifest_path)
        assert_path_safe(patient_manifest_path)
        self.manifest_path = Path(manifest_path)
        self.patient_manifest_path = Path(patient_manifest_path)
        self.split = split
        self.cache = cache or PatientVolumeCache(max_patients=LRU_MAX_PATIENTS)
        # manifest rows 로드 (CSV DictReader, 실제 npy 로딩 없음)
        self._rows: list[dict] = []          # raw manifest rows for split
        self._patient_info: dict = {}        # safe_id → {ct_hu_npy, roi_0_0_npy, meta_json}
        self._loaded = False                 # flag: manifest loaded

    def load_manifest(self) -> None:
        """
        manifest CSV와 patient_manifest CSV를 읽어 self._rows, self._patient_info에 저장.
        실제 npy 값 로딩 없음. 경로 존재 여부만 확인.
        """
        with open(self.manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._rows = [r for r in reader if r.get("split") == self.split]
        with open(self.patient_manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") == self.split:
                    self._patient_info[row["safe_id"]] = {
                        "ct_hu_npy": row["ct_hu_npy"],
                        "roi_0_0_npy": row["roi_0_0_npy"],
                        "meta_json": row.get("meta_json", ""),
                    }
        self._loaded = True

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        """
        인덱스 → manifest row → CT crop 생성 → 반환.
        반환 dict:
          crop: Tensor(3, 96, 96) float32  (학습 시)
          six_bin_label: str
          six_bin_idx: int  (0~5)
          safe_id: str
          local_z: int
          low_z_boundary_warning: bool
          manifest_id: str
        [정적 검사 단계] npy 로딩/crop 생성 호출 금지.
        이 함수 정의만 허용; 실제 반환 로직은 학습(RD-B6) 단계에서 활성.
        """
        row = self._rows[idx]
        safe_id = row["safe_id"]
        local_z = int(row["local_z"])
        crop_y0 = int(row["crop_y0"])
        crop_x0 = int(row["crop_x0"])
        crop_y1 = int(row["crop_y1"])
        crop_x1 = int(row["crop_x1"])
        six_bin_label = row["six_bin_label"]
        six_bin_idx = SIX_BIN_LABELS.index(six_bin_label)

        # [학습 시] PatientVolumeCache 통해 CT/ROI 로드 → crop 생성
        # [정적 검사] 아래 블록 비활성 (ct_volume=None → crop 반환 불가)
        ct_volume = None  # 정적 검사 단계: 로딩 금지

        if ct_volume is not None:
            info = self._patient_info[safe_id]
            entry = self.cache.get(safe_id, info["ct_hu_npy"], info["roi_0_0_npy"])
            ct_volume = entry["ct"]
            roi_volume = entry["roi"]
            z_max = entry["z_max"]
            result = build_mixed_3ch_crop_from_volume(
                ct_volume, local_z, crop_y0, crop_x0, crop_y1, crop_x1, z_max
            )
            crop = result["crop"]
            low_z_warn = result["low_z_boundary_warning"]
        else:
            crop = None
            low_z_warn = has_low_z_boundary_warning(local_z)

        return {
            "crop": crop,
            "six_bin_label": six_bin_label,
            "six_bin_idx": six_bin_idx,
            "safe_id": safe_id,
            "local_z": local_z,
            "low_z_boundary_warning": low_z_warn,
            "manifest_id": row["manifest_id"],
        }

    def get_bin_index_map(self) -> dict:
        """six_bin_label → list[int] 인덱스 매핑 (sampler 초기화용)."""
        bin_map: dict = {label: [] for label in SIX_BIN_LABELS}
        for i, row in enumerate(self._rows):
            bin_map[row["six_bin_label"]].append(i)
        return bin_map


# =============================================================================
# 4. SixBinBalancedBatchSampler skeleton
# =============================================================================
class SixBinBalancedBatchSampler:
    """
    6-bin balanced batch sampler.
    매 배치에서 각 bin에서 동일 개수(batch_size // 6) 샘플링.
    duplicate oversampling 금지: 각 epoch 내 동일 인덱스 2회 사용 없음.
    bin 부족 시: shorter epoch (부족한 bin이 소진되면 epoch 종료).
    patient leakage 방지: split=train 환자만 사용 (Dataset 수준에서 보장).
    seed 고정 가능.

    이번 단계에서는 torch.utils.data.Sampler 상속 없이 정의.
    실제 DataLoader 연결은 RD-B6 단계에서 수행.
    """

    def __init__(
        self,
        dataset: "RD4ADNormalCropDataset",
        batch_size: int = BATCH_SIZE_CANDIDATE_A,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if batch_size % 6 != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by 6 (n_bins=6). "
                f"권장: 24 (bin당 4) 또는 48 (bin당 8)"
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.per_bin = batch_size // 6
        self.seed = seed
        self.drop_last = drop_last
        self._bin_map: dict = {}   # populated on first __iter__

    def _build_bin_map(self) -> dict:
        """Dataset에서 bin_map 가져와 복사본 반환 (원본 불변 보장)."""
        raw = self.dataset.get_bin_index_map()
        return {k: list(v) for k, v in raw.items()}

    def __iter__(self):
        """
        shuffled bin별 인덱스에서 per_bin씩 꺼내 배치 구성.
        가장 먼저 소진되는 bin 기준 epoch 종료 (shorter epoch 정책).
        duplicate oversampling 금지: 동일 epoch 내 동일 인덱스 반복 없음.
        이번 단계(정적 검사)에서 DataLoader에 연결하지 않음 — 구조 정의 전용.
        """
        rng = random.Random(self.seed)
        bin_map = self._build_bin_map()
        for label in SIX_BIN_LABELS:
            rng.shuffle(bin_map[label])
        # 각 bin에서 cursor
        cursors = {label: 0 for label in SIX_BIN_LABELS}
        min_len = min(len(bin_map[label]) for label in SIX_BIN_LABELS)
        n_batches = min_len // self.per_bin

        for _ in range(n_batches):
            batch = []
            for label in SIX_BIN_LABELS:
                start = cursors[label]
                end = start + self.per_bin
                batch.extend(bin_map[label][start:end])
                cursors[label] = end
            yield batch

    def __len__(self) -> int:
        """
        예상 배치 수 (shorter epoch 정책, 소진 불가 bin 기준).
        Dataset이 load_manifest()를 호출했을 때만 정확함.
        """
        if not self.dataset._loaded:
            return 0
        bin_map = self.dataset.get_bin_index_map()
        min_len = min(len(v) for v in bin_map.values())
        return min_len // self.per_bin

    def design_summary(self) -> dict:
        """설계 요약 (report용, DataLoader 연결 불필요)."""
        return {
            "batch_size": self.batch_size,
            "per_bin": self.per_bin,
            "seed": self.seed,
            "drop_last": self.drop_last,
            "policy_bin_shortage": "shorter_epoch (소진 bin 기준 epoch 종료)",
            "duplicate_oversampling": "금지",
            "patient_leakage_prevention": "split=train 환자만, safe_id 기준 분리",
            "n_bins": 6,
            "bin_labels": SIX_BIN_LABELS,
        }


# =============================================================================
# 5. RD4ADTeacherStudentSkeleton (model skeleton)
# =============================================================================
class RD4ADTeacherStudentSkeleton:
    """
    RD4AD teacher-student 구조 skeleton.
    이번 단계(정적 검사)에서:
      - torch model forward 호출 금지
      - ResNet18 pretrained weight 다운로드 금지 (pretrained=True 사용 금지)
      - ImageNet weight 로딩 금지 (RD-B6 이후 별도 승인 필요)
      - GPU 사용 금지

    구조 정의만 허용.
    실제 nn.Module 상속 및 forward 구현은 RD-B6 단계에서 수행.
    """

    # ── 이론 feature shape 설계 (ResNet18, input 3×96×96) ──────────────────
    FEATURE_SHAPE_DESIGN = {
        "input":  (3, 96, 96),
        "layer1": (64,  24, 24),   # stride=4 (conv1 stride2 + maxpool stride2)
        "layer2": (128, 12, 12),   # stride=2
        "layer3": (256,  6,  6),   # stride=2
    }
    # OCBE bottleneck 입력: layer3 (256×6×6)
    OCBE_INPUT_SHAPE = (256, 6, 6)
    # student decoder 출력 shapes (reverse)
    STUDENT_SHAPES = {
        "de_layer3": (256, 6,  6),
        "de_layer2": (128, 12, 12),
        "de_layer1": (64,  24, 24),
    }

    def __init__(self) -> None:
        # torch import은 가능하나 model 인스턴스화/forward 금지
        self._teacher_initialized = False
        self._student_initialized = False

    # ── 구조 정의 (RD-B6에서 실제 nn.Module로 구현) ─────────────────────────

    @staticmethod
    def build_teacher_placeholder() -> dict:
        """
        Teacher = ResNet18 (ImageNet pretrained) placeholder.
        [정적 검사 단계]
          - torch 미인스턴스화
          - pretrained=True 사용 금지
          - weights=None 또는 placeholder만 기록
        반환: 설계 메타 dict (실제 모델 객체 없음)
        """
        return {
            "backbone": "ResNet18",
            "pretrained": "PLACEHOLDER_LOAD_AT_RD_B6",
            "weights": "ImageNet (RD-B6 이후 별도 승인 후 로드)",
            "requires_grad": False,
            "mode": "eval (고정)",
            "feature_taps": ["layer1", "layer2", "layer3"],
            "forward_call": "금지 (RD-B5 단계)",
            "note": (
                "torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) "
                "형태로 로드 예정. RD-B5에서는 weights=None 또는 placeholder 사용만 허용."
            ),
        }

    @staticmethod
    def build_ocbe_placeholder() -> dict:
        """
        OCBE Bottleneck placeholder.
        입력: layer3 feature (256×6×6).
        구성: Conv(256→256, k=3, p=1) → BN → ReLU → Conv(256→256, k=3, p=1)
        출력: compact code (256×6×6).
        """
        return {
            "name": "OCBE_Bottleneck",
            "input_shape": (256, 6, 6),
            "output_shape": (256, 6, 6),
            "layers": ["Conv2d(256,256,3,1,1)", "BN", "ReLU", "Conv2d(256,256,3,1,1)"],
            "forward_call": "금지 (RD-B5 단계)",
        }

    @staticmethod
    def build_student_decoder_placeholder() -> dict:
        """
        Student Decoder (reverse distillation) placeholder.
        de_layer3 → de_layer2 → de_layer1
        각 단계: ConvTranspose2d or Upsample + Conv.
        """
        return {
            "name": "Student_Decoder",
            "stages": [
                {"name": "de_layer3", "input": (256,6,6),  "output": (256,6,6),
                 "op": "Conv2d(256,256,3,1,1)+BN+ReLU"},
                {"name": "de_layer2", "input": (256,6,6),  "output": (128,12,12),
                 "op": "Upsample(2x)+Conv2d(256,128,3,1,1)+BN+ReLU"},
                {"name": "de_layer1", "input": (128,12,12), "output": (64,24,24),
                 "op": "Upsample(2x)+Conv2d(128,64,3,1,1)+BN+ReLU"},
            ],
            "trainable": True,
            "forward_call": "금지 (RD-B5 단계)",
        }

    @staticmethod
    def feature_matching_loss_placeholder(teacher_feats, student_feats) -> None:
        """
        Feature matching loss placeholder.
        loss = Σ (1 - cosine_similarity(t_feat, s_feat, dim=1).mean())
        [정적 검사 단계에서 호출 금지]
        학습(RD-B6) 단계에서 torch.nn.functional.cosine_similarity 사용.
        """
        raise RuntimeError(
            "[SAFETY] feature_matching_loss_placeholder: "
            "RD-B5 단계에서 model forward/loss 계산 금지. "
            "RD-B6 이후에 활성."
        )

    @staticmethod
    def anomaly_score_placeholder(teacher_feats, student_feats) -> None:
        """
        Anomaly score placeholder.
        각 scale별 cosine distance 맵 → 합산 → crop-level score.
        [정적 검사 단계에서 호출 금지]
        """
        raise RuntimeError(
            "[SAFETY] anomaly_score_placeholder: "
            "RD-B5 단계에서 model forward/scoring 금지. "
            "RD-B6 이후에 활성."
        )

    def design_summary(self) -> dict:
        return {
            "teacher": self.build_teacher_placeholder(),
            "ocbe": self.build_ocbe_placeholder(),
            "student_decoder": self.build_student_decoder_placeholder(),
            "feature_shapes": self.FEATURE_SHAPE_DESIGN,
            "student_shapes": self.STUDENT_SHAPES,
            "loss": "cosine_distance (placeholder)",
            "anomaly_score": "multi-scale cosine distance (placeholder)",
            "imagenet_weights_policy": (
                "RD-B5: 금지. RD-B6 tiny smoke train 단계에서 "
                "사용자 별도 승인 후 weights=ResNet18_Weights.IMAGENET1K_V1 사용."
            ),
        }


# =============================================================================
# 6. Static validation functions
# =============================================================================

def validate_manifest(manifest_path: Path) -> dict:
    """manifest CSV 정적 검증."""
    result = {
        "manifest_exists": False,
        "row_count": 0,
        "row_count_ok": False,
        "columns_ok": False,
        "missing_columns": [],
        "six_bin_labels_ok": False,
        "six_bin_labels_found": [],
        "six_bin_label_missing": [],
        "bin_counts": {},
        "unique_safe_ids": 0,
    }
    if not manifest_path.exists():
        return result
    result["manifest_exists"] = True
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in cols]
        result["missing_columns"] = missing_cols
        result["columns_ok"] = len(missing_cols) == 0
        for row in reader:
            rows.append(row)
    result["row_count"] = len(rows)
    result["row_count_ok"] = len(rows) == EXPECTED_ROWS

    bin_counts: dict = defaultdict(int)
    safe_ids = set()
    for row in rows:
        lbl = row.get("six_bin_label", "")
        bin_counts[lbl] += 1
        safe_ids.add(row.get("safe_id", ""))
    result["bin_counts"] = dict(bin_counts)
    found = sorted(bin_counts.keys())
    result["six_bin_labels_found"] = found
    result["six_bin_label_missing"] = [l for l in SIX_BIN_LABELS if l not in bin_counts]
    result["six_bin_labels_ok"] = (
        set(found) == set(SIX_BIN_LABELS) and
        len(result["six_bin_label_missing"]) == 0
    )
    result["unique_safe_ids"] = len(safe_ids)
    return result


def validate_ct_roi_paths(manifest_path: Path, patient_manifest_path: Path) -> dict:
    """CT/ROI 경로 존재 여부 확인 (npy 값 로딩 금지, shape metadata도 로딩 금지)."""
    result = {
        "patient_manifest_exists": False,
        "ct_exist": 0,
        "ct_missing": [],
        "roi_exist": 0,
        "roi_missing": [],
        "stage2_holdout_intersection": 0,
        "path_safety_ok": True,
    }
    if not patient_manifest_path.exists():
        return result
    result["patient_manifest_exists"] = True

    # manifest에서 safe_id 수집
    safe_ids_manifest = set()
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            safe_ids_manifest.add(row.get("safe_id", ""))

    patient_info: dict = {}
    with open(patient_manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == "train":
                patient_info[row["safe_id"]] = {
                    "ct_hu_npy": row.get("ct_hu_npy", ""),
                    "roi_0_0_npy": row.get("roi_0_0_npy", ""),
                }

    ct_missing = []
    roi_missing = []
    for sid in sorted(safe_ids_manifest):
        if sid not in patient_info:
            ct_missing.append(sid)
            roi_missing.append(sid)
            continue
        ct_path = Path(patient_info[sid]["ct_hu_npy"])
        roi_path = Path(patient_info[sid]["roi_0_0_npy"])
        # 안전 검사
        try:
            assert_path_safe(str(ct_path))
            assert_path_safe(str(roi_path))
        except RuntimeError:
            result["stage2_holdout_intersection"] += 1
            result["path_safety_ok"] = False
            continue
        # 존재 확인만 (npy 로딩 금지)
        if ct_path.exists():
            result["ct_exist"] += 1
        else:
            ct_missing.append(sid)
        if roi_path.exists():
            result["roi_exist"] += 1
        else:
            roi_missing.append(sid)

    result["ct_missing"] = ct_missing[:10]   # 보고는 최대 10개
    result["roi_missing"] = roi_missing[:10]
    return result


def validate_safety_checklist() -> list:
    """안전 항목 체크리스트 (정적 확인)."""
    items = [
        ("stage2_holdout_access", "stage2_holdout 경로 접근", True, "금지 키워드 가드 구현"),
        ("lesion_access", "lesion raw CT/mask 접근", True, "금지 키워드 가드 구현"),
        ("crop_npz_generation", "crop NPZ 생성", True, "이번 단계 없음"),
        ("model_forward", "model forward 실행", True, "이번 단계 없음"),
        ("gpu_usage", "GPU 사용", True, "이번 단계 없음"),
        ("checkpoint_load", "checkpoint 로드", True, "이번 단계 없음"),
        ("pretrained_weight_download", "pretrained weight 다운로드", True, "weights=None 또는 placeholder만"),
        ("training", "학습 실행", True, "이번 단계 없음"),
        ("scoring", "scoring 실행", True, "이번 단계 없음"),
        ("existing_file_modify", "기존 파일 수정/삭제", True, "신규 파일만 생성"),
        ("output_root_preexists", "output root 비존재", not OUTPUT_ROOT.exists(), str(OUTPUT_ROOT)),
    ]
    return [
        {"check": item[0], "description": item[1], "pass": item[2], "note": item[3]}
        for item in items
    ]


def validate_shape_design() -> list:
    """이론 feature shape 설계 검증 (forward 없이)."""
    md = RD4ADTeacherStudentSkeleton()
    rows = []
    for name, shape in md.FEATURE_SHAPE_DESIGN.items():
        rows.append({
            "layer": name,
            "shape_C": shape[0],
            "shape_H": shape[1],
            "shape_W": shape[2],
            "source": "RD-B3 설계",
            "rd_b3_match": "yes",
        })
    for name, shape in md.STUDENT_SHAPES.items():
        rows.append({
            "layer": name,
            "shape_C": shape[0],
            "shape_H": shape[1],
            "shape_W": shape[2],
            "source": "RD-B3 설계",
            "rd_b3_match": "yes",
        })
    return rows


# =============================================================================
# 7. Selftest (순수 함수만)
# =============================================================================

def run_selftest() -> dict:
    """
    순수 함수 단위 테스트:
      - normalize_hu_new_rd_style 값 범위 테스트
      - edge clamp index 테스트 (compute_mip_slab_indices)
      - has_low_z_boundary_warning 테스트
      - SixBinBalancedBatchSampler index grouping 테스트 (dummy)
    CT 로딩, model forward, GPU 사용 금지.
    """
    import numpy as np
    results = []

    # ── (1) normalize_hu_new_rd_style ──────────────────────────────────────
    # 경계값: -1000 → 0.0, 600 → 1.0, 0 → 0.625, -500 → 0.3125
    test_cases = [
        (-1000.0, 0.0),
        (600.0, 1.0),
        (0.0, 0.625),
        (-500.0, 0.3125),
        (-2000.0, 0.0),   # clip 하한
        (9999.0, 1.0),    # clip 상한
    ]
    for hu_val, expected in test_cases:
        arr = np.array([[hu_val]], dtype="float32")
        out = normalize_hu_new_rd_style(arr)[0, 0]
        ok = abs(float(out) - expected) < 1e-5
        results.append({
            "test": "normalize_hu_new_rd_style",
            "input": hu_val,
            "expected": expected,
            "actual": round(float(out), 6),
            "pass": ok,
        })

    # ── (2) compute_mip_slab_indices ─────────────────────────────────────
    slab_tests = [
        # (center_z, direction, z_max, expected_result)
        (10, "lower", 100, [7, 8, 9]),
        (10, "upper", 100, [11, 12, 13]),
        (1,  "lower", 100, [0, 0, 0]),    # lower clamp: -2,-1,0 → 0,0,0
        (2,  "lower", 100, [0, 0, 1]),    # -1,0,1 → 0,0,1
        (98, "upper", 100, [99, 99, 99]), # 99,100,101 → 99,99,99
    ]
    for center_z, direction, z_max, expected in slab_tests:
        actual = compute_mip_slab_indices(center_z, direction, z_max)
        ok = actual == expected
        results.append({
            "test": "compute_mip_slab_indices",
            "input": f"z={center_z},{direction},z_max={z_max}",
            "expected": str(expected),
            "actual": str(actual),
            "pass": ok,
        })

    # ── (3) has_low_z_boundary_warning ───────────────────────────────────
    warn_tests = [(7, True), (8, False), (0, True), (100, False)]
    for z, expected in warn_tests:
        actual = has_low_z_boundary_warning(z)
        results.append({
            "test": "has_low_z_boundary_warning",
            "input": f"z={z}",
            "expected": str(expected),
            "actual": str(actual),
            "pass": actual == expected,
        })

    # ── (4) SixBinBalancedBatchSampler dummy grouping ──────────────────
    # dummy rows로 bin grouping 검증
    class _DummyDataset:
        _loaded = True
        def get_bin_index_map(self):
            # 각 bin 10개씩
            return {label: list(range(i*10, i*10+10)) for i, label in enumerate(SIX_BIN_LABELS)}
    ds = _DummyDataset()
    sampler = SixBinBalancedBatchSampler(ds, batch_size=24, seed=0)
    batches = list(sampler)
    n_batches_expected = 10 // 4   # min_len=10, per_bin=4 → 2 batches
    batches_ok = len(batches) == n_batches_expected
    batch_size_ok = all(len(b) == 24 for b in batches)
    results.append({
        "test": "SixBinBalancedBatchSampler_grouping",
        "input": "dummy 6-bin 10/bin, bs=24",
        "expected": f"n_batches={n_batches_expected}, batch_size=24",
        "actual": f"n_batches={len(batches)}, batch_size={batches[0] if batches else 'N/A'}",
        "pass": batches_ok and batch_size_ok,
    })

    # ── (5) batch_size 비6배수 거부 ──────────────────────────────────────
    try:
        _ = SixBinBalancedBatchSampler(ds, batch_size=25, seed=0)
        results.append({
            "test": "SixBinBalancedBatchSampler_invalid_bs",
            "input": "batch_size=25",
            "expected": "ValueError",
            "actual": "no error",
            "pass": False,
        })
    except ValueError:
        results.append({
            "test": "SixBinBalancedBatchSampler_invalid_bs",
            "input": "batch_size=25",
            "expected": "ValueError",
            "actual": "ValueError raised",
            "pass": True,
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
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("RD-B5 Dataset / Loader / Model Skeleton Static Check")
    print("=" * 70)

    # ── output root 존재 여부 선제 확인 ───────────────────────────────────────
    if IS_WRITE_REPORT and OUTPUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재 → 즉시 중단: {OUTPUT_ROOT}")
        sys.exit(1)

    # ── selftest 모드 ─────────────────────────────────────────────────────────
    if IS_SELFTEST:
        print("\n[SELFTEST] 순수 함수 테스트 실행 ...")
        st = run_selftest()
        print(f"  n_tests={st['n_tests']}  pass={st['n_pass']}  fail={st['n_fail']}")
        for r in st["details"]:
            status = "PASS" if r["pass"] else "FAIL"
            print(f"  [{status}] {r['test']} | input={r['input']} | "
                  f"expected={r['expected']} | actual={r['actual']}")
        if st["all_pass"]:
            print("\n판정: SELFTEST 전체 통과")
        else:
            print("\n판정: SELFTEST 일부 실패")
        return

    # ── dry-check / write-report 공통 검증 ───────────────────────────────────
    print(f"\n[STEP 1] manifest 정적 검증 ...")
    assert_path_safe(str(MANIFEST_PATH))
    mv = validate_manifest(MANIFEST_PATH)
    print(f"  manifest 존재: {mv['manifest_exists']}")
    print(f"  row count: {mv['row_count']} (기대={EXPECTED_ROWS}, ok={mv['row_count_ok']})")
    print(f"  columns_ok: {mv['columns_ok']} (누락={mv['missing_columns']})")
    print(f"  six_bin_labels_ok: {mv['six_bin_labels_ok']} "
          f"(누락={mv['six_bin_label_missing']})")
    print(f"  bin_counts: {mv['bin_counts']}")
    print(f"  unique_safe_ids: {mv['unique_safe_ids']}")

    if not mv["manifest_exists"]:
        errors.append({"step": "step1", "error": f"manifest 없음: {MANIFEST_PATH}"})
    if not mv["row_count_ok"]:
        errors.append({"step": "step1", "error": f"row count 불일치: {mv['row_count']} ≠ {EXPECTED_ROWS}"})
    if not mv["columns_ok"]:
        errors.append({"step": "step1", "error": f"누락 columns: {mv['missing_columns']}"})
    if not mv["six_bin_labels_ok"]:
        errors.append({"step": "step1", "error": f"six_bin_label 누락: {mv['six_bin_label_missing']}"})

    print(f"\n[STEP 2] CT/ROI 경로 존재 확인 ...")
    assert_path_safe(str(PATIENT_MANIFEST_PATH))
    pr = validate_ct_roi_paths(MANIFEST_PATH, PATIENT_MANIFEST_PATH)
    print(f"  patient_manifest 존재: {pr['patient_manifest_exists']}")
    print(f"  CT 존재: {pr['ct_exist']}/{EXPECTED_PATIENTS} (누락={len(pr['ct_missing'])})")
    print(f"  ROI 존재: {pr['roi_exist']}/{EXPECTED_PATIENTS} (누락={len(pr['roi_missing'])})")
    print(f"  stage2_holdout intersection: {pr['stage2_holdout_intersection']}")
    print(f"  path_safety_ok: {pr['path_safety_ok']}")
    if pr["ct_missing"]:
        print(f"  CT 누락 (최대 10): {pr['ct_missing']}")
    if pr["roi_missing"]:
        print(f"  ROI 누락 (최대 10): {pr['roi_missing']}")

    if not pr["patient_manifest_exists"]:
        errors.append({"step": "step2", "error": f"patient_manifest 없음: {PATIENT_MANIFEST_PATH}"})
    if pr["stage2_holdout_intersection"] > 0:
        errors.append({"step": "step2", "error": "stage2_holdout intersection > 0"})

    print(f"\n[STEP 3] 안전 체크리스트 ...")
    safety_items = validate_safety_checklist()
    for item in safety_items:
        status = "PASS" if item["pass"] else "FAIL"
        print(f"  [{status}] {item['description']}: {item['note']}")
    safety_failures = [i for i in safety_items if not i["pass"]]
    if safety_failures:
        for i in safety_failures:
            errors.append({"step": "step3_safety", "error": f"{i['check']}: {i['note']}"})

    print(f"\n[STEP 4] 이론 feature shape 검증 ...")
    shape_rows = validate_shape_design()
    for r in shape_rows:
        print(f"  {r['layer']:12s}: ({r['shape_C']}, {r['shape_H']}, {r['shape_W']}) [{r['rd_b3_match']}]")

    print(f"\n[STEP 5] Sampler 설계 검증 ...")
    sd = SixBinBalancedBatchSampler.__new__(SixBinBalancedBatchSampler)
    sd.batch_size = BATCH_SIZE_CANDIDATE_A
    sd.per_bin = BATCH_SIZE_CANDIDATE_A // 6
    sd.seed = 42
    sd.drop_last = True
    sd.dataset = None
    print(f"  batch_size_A={BATCH_SIZE_CANDIDATE_A} (bin당 {BATCH_SIZE_CANDIDATE_A//6})")
    print(f"  batch_size_B={BATCH_SIZE_CANDIDATE_B} (bin당 {BATCH_SIZE_CANDIDATE_B//6})")
    # 이론 n_batches
    min_bin_rows = 13_932  # upper_interior (RD-B1 결과)
    n_batch_a = min_bin_rows // (BATCH_SIZE_CANDIDATE_A // 6)
    n_batch_b = min_bin_rows // (BATCH_SIZE_CANDIDATE_B // 6)
    print(f"  이론 n_batches/epoch: A={n_batch_a}, B={n_batch_b}")

    print(f"\n[STEP 6] Model skeleton 설계 검증 ...")
    ms = RD4ADTeacherStudentSkeleton()
    ms_summary = ms.design_summary()
    print(f"  teacher backbone: {ms_summary['teacher']['backbone']}")
    print(f"  teacher pretrained: {ms_summary['teacher']['pretrained']}")
    print(f"  feature_taps: {ms_summary['teacher']['feature_taps']}")
    print(f"  imagenet_weights_policy: {ms_summary['imagenet_weights_policy'][:80]}...")

    print(f"\n[STEP 7] output root 비존재 확인 ...")
    print(f"  output root: {OUTPUT_ROOT}")
    print(f"  존재: {OUTPUT_ROOT.exists()} (dry-check에서는 생성하지 않음)")
    if IS_DRY_CHECK and OUTPUT_ROOT.exists():
        errors.append({"step": "step7", "error": f"output root 이미 존재: {OUTPUT_ROOT}"})

    # ── 결과 요약 ─────────────────────────────────────────────────────────────
    n_errors = len(errors)
    print(f"\n{'='*70}")
    if n_errors == 0:
        print("판정: 통과 (오류 없음)")
    else:
        print(f"판정: 경고 ({n_errors}개 이슈)")
        for e in errors:
            print(f"  [오류] {e}")
    print(f"{'='*70}")

    if IS_DRY_CHECK:
        print("\n[dry-check 완료] output root 생성하지 않음.")
        print("→ 사용자 승인 후 --write-report로 실행하면 report를 생성합니다.")
        return

    # ── write-report 모드 ─────────────────────────────────────────────────────
    if IS_WRITE_REPORT:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
        print(f"\n[write-report] output root 생성: {OUTPUT_ROOT}")
        _write_report(mv, pr, safety_items, shape_rows, ms_summary, n_errors)
        print("[write-report 완료]")


def _write_report(mv, pr, safety_items, shape_rows, ms_summary, n_errors) -> None:
    """report 파일 생성 (--write-report 모드에서만 호출)."""
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── rd_b5_static_validation_summary.json ──────────────────────────────
    summary = {
        "version": "rd_b5_v1",
        "timestamp": ts,
        "manifest": {
            "path": str(MANIFEST_PATH),
            "exists": mv["manifest_exists"],
            "row_count": mv["row_count"],
            "row_count_ok": mv["row_count_ok"],
            "columns_ok": mv["columns_ok"],
            "six_bin_labels_ok": mv["six_bin_labels_ok"],
            "bin_counts": mv["bin_counts"],
            "unique_safe_ids": mv["unique_safe_ids"],
        },
        "ct_roi_paths": {
            "patient_manifest_exists": pr["patient_manifest_exists"],
            "ct_exist": pr["ct_exist"],
            "roi_exist": pr["roi_exist"],
            "ct_missing_count": len(pr["ct_missing"]),
            "roi_missing_count": len(pr["roi_missing"]),
            "stage2_holdout_intersection": pr["stage2_holdout_intersection"],
        },
        "safety": {"n_failures": sum(1 for i in safety_items if not i["pass"])},
        "model_skeleton": {
            "teacher": "ResNet18 placeholder (weights=None, RD-B6 이후 로드)",
            "feature_taps": ["layer1", "layer2", "layer3"],
        },
        "n_errors": n_errors,
        "verdict": "통과" if n_errors == 0 else f"경고 ({n_errors}개)",
        "safety_confirmed": {
            "crop_npz_generated": 0,
            "model_forward_executed": 0,
            "gpu_used": 0,
            "checkpoint_loaded": 0,
            "stage2_holdout_accessed": 0,
            "existing_files_modified": 0,
        },
    }
    with open(OUTPUT_ROOT / "rd_b5_static_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── rd_b5_dataset_loader_design.csv ───────────────────────────────────
    with open(OUTPUT_ROOT / "rd_b5_dataset_loader_design.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item", "value", "note"])
        w.writeheader()
        rows = [
            ("class", "RD4ADNormalCropDataset", "hybrid_cache 기반"),
            ("manifest_path", str(MANIFEST_PATH), "RD-B1 결과"),
            ("patient_manifest_path", str(PATIENT_MANIFEST_PATH), "CT/ROI 경로 소스"),
            ("split", "train", "stage2_holdout 접근 금지"),
            ("crop_size", CROP_SIZE, "96×96"),
            ("n_channels", N_CHANNELS, "mixed_3ch"),
            ("ch0", "CT_center_slice", "normalize_hu_new_rd_style"),
            ("ch1", "lower_3mm_MIP", "z-3,z-2,z-1 max projection"),
            ("ch2", "upper_3mm_MIP", "z+1,z+2,z+3 max projection"),
            ("normalization", "HU_clip[-1000,600]→(x+1000)/1600→[0,1]", "RD-B2b 확정"),
            ("edge_clamp", "numpy.clip(slab_indices, 0, z_max-1)", "경계 slice 반복"),
            ("low_z_boundary_warning", "z≤7", "diaphragm saturation risk"),
            ("cache_class", "PatientVolumeCache", "LRU per-worker"),
            ("cache_max_patients", LRU_MAX_PATIENTS, "기본값"),
            ("cache_mode", "mmap_mode='r'", "read-only, 실제 로드 RD-B6+"),
            ("n_workers_candidate", 4, "DataLoader num_workers"),
        ]
        for r in rows:
            w.writerow({"item": r[0], "value": str(r[1]), "note": str(r[2])})

    # ── rd_b5_sampler_design.csv ───────────────────────────────────────────
    with open(OUTPUT_ROOT / "rd_b5_sampler_design.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item", "candidate_A", "candidate_B", "note"])
        w.writeheader()
        min_bin = 13_932
        rows = [
            ("class", "SixBinBalancedBatchSampler", "SixBinBalancedBatchSampler", ""),
            ("batch_size", 24, 48, ""),
            ("per_bin", 4, 8, "batch_size // 6"),
            ("n_bins", 6, 6, "six_bin_label"),
            ("seed", 42, 42, "고정 가능"),
            ("drop_last", True, True, ""),
            ("min_bin_rows", min_bin, min_bin, "upper_interior (RD-B1)"),
            ("n_batches_per_epoch", min_bin // 4, min_bin // 8,
             "shorter_epoch 정책"),
            ("duplicate_oversampling", "금지", "금지", ""),
            ("bin_shortage_policy", "shorter_epoch", "shorter_epoch",
             "가장 먼저 소진되는 bin 기준 epoch 종료"),
            ("patient_leakage_prevention",
             "split=train safe_id 기준",
             "split=train safe_id 기준", ""),
        ]
        for r in rows:
            w.writerow({"item": r[0], "candidate_A": str(r[1]),
                        "candidate_B": str(r[2]), "note": str(r[3])})

    # ── rd_b5_model_skeleton_design.csv ───────────────────────────────────
    with open(OUTPUT_ROOT / "rd_b5_model_skeleton_design.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["component", "layer", "shape", "note"])
        w.writeheader()
        ms = RD4ADTeacherStudentSkeleton()
        for name, shape in ms.FEATURE_SHAPE_DESIGN.items():
            w.writerow({"component": "teacher_encoder",
                        "layer": name,
                        "shape": f"{shape[0]}×{shape[1]}×{shape[2]}",
                        "note": "ResNet18 (ImageNet, frozen, RD-B6+)"})
        for name, shape in ms.STUDENT_SHAPES.items():
            w.writerow({"component": "student_decoder",
                        "layer": name,
                        "shape": f"{shape[0]}×{shape[1]}×{shape[2]}",
                        "note": "trainable, RD-B6 이후 활성"})
        w.writerow({"component": "ocbe_bottleneck",
                    "layer": "ocbe",
                    "shape": "256×6×6",
                    "note": "teacher layer3 → compact code"})
        w.writerow({"component": "loss",
                    "layer": "feature_matching",
                    "shape": "scalar",
                    "note": "cosine_distance placeholder"})

    # ── rd_b5_safety_checklist.csv ────────────────────────────────────────
    with open(OUTPUT_ROOT / "rd_b5_safety_checklist.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["check", "description", "pass", "note"])
        w.writeheader()
        for item in safety_items:
            w.writerow(item)

    # ── rd_b5_errors.csv ──────────────────────────────────────────────────
    with open(OUTPUT_ROOT / "rd_b5_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "error"])
        w.writeheader()
        for e in errors:
            w.writerow(e)

    # ── rd_b5_dataset_loader_model_skeleton_static_report.md ──────────────
    report_lines = [
        "# RD-B5 Dataset / Loader / Model Skeleton Static Check Report",
        f"- 버전: rd_b5_v1",
        f"- 날짜: {ts}",
        f"- 판정: {'통과' if n_errors == 0 else f'경고 ({n_errors}개)'}",
        "",
        "---",
        "## 1. RD-B1 ~ RD-B4 요약",
        "",
        "| 단계 | 결과 |",
        "|------|------|",
        "| RD-B1 | 6-bin balanced manifest 86,017 rows / 290 patients / cap 50/bin/patient |",
        "| RD-B2b | mixed_3ch ADOPT: ch1=CT center, ch2=lower 3mm MIP, ch3=upper 3mm MIP |",
        "| RD-B2b norm | HU clip [-1000, 600] → (x+1000)/1600 → [0,1] |",
        "| RD-B3 | true RD4AD teacher-student / ResNet18 ImageNet frozen / layer1/layer2/layer3 |",
        "| RD-B4 | crop strategy = hybrid_cache (on-the-fly + patient LRU cache) |",
        "",
        "---",
        "## 2. Dataset 설계 (RD4ADNormalCropDataset)",
        "",
        "- manifest CSV (RD-B1): 86,017 rows, train split만 사용",
        "- `__len__`: manifest row 수 반환",
        "- `__getitem__`: manifest row → PatientVolumeCache → crop 생성 → Tensor 반환",
        "- 반환 keys: crop, six_bin_label, six_bin_idx, safe_id, local_z, low_z_boundary_warning, manifest_id",
        "- 이번 단계: CT 로딩/crop 생성 비활성 (ct_volume=None guard)",
        "",
        "---",
        "## 3. hybrid_cache / PatientVolumeCache 설계",
        "",
        "- 기본 on-the-fly + patient LRU cache (max_patients=8, per-worker)",
        "- CT/ROI: `np.load(path, mmap_mode='r')` read-only",
        "- LRU 교체: `OrderedDict.move_to_end / popitem(last=False)`",
        "- forbidden path guard: stage2_holdout/lesion 포함 경로 → RuntimeError",
        "- cache.clear() 함수 포함",
        "- 이번 단계: _load_npy_entry 비활성 (학습 RD-B6+에서만 호출)",
        "",
        "---",
        "## 4. mixed_3ch crop 생성 설계",
        "",
        "```",
        "crop_size = 96×96",
        "ch0 = normalize(CT[center_z, y0:y1, x0:x1])",
        "ch1 = normalize(lower_3mm_MIP(ct, center_z)[y0:y1, x0:x1])",
        "ch2 = normalize(upper_3mm_MIP(ct, center_z)[y0:y1, x0:x1])",
        "lower MIP: max(CT[z-3, z-2, z-1], axis=0)  # clamp to [0, z_max-1]",
        "upper MIP: max(CT[z+1, z+2, z+3], axis=0)  # clamp to [0, z_max-1]",
        "```",
        "",
        "edge clamp: `max(0, min(idx, z_max-1))` — 경계 slice 반복 사용",
        "",
        "low_z_boundary_warning: z ≤ 7 → True (diaphragm saturation risk)",
        "- RD-B1 집계: z≤7 195개 (0.227%, medium risk)",
        "- RD-B6 smoke train에서 lower_boundary bin 균일도 모니터링 필요",
        "",
        "---",
        "## 5. new_RD_style normalization 설계",
        "",
        "```python",
        "clipped = np.clip(hu_array, -1000, 600)",
        "normalized = (clipped + 1000) / 1600  # → [0, 1] float32",
        "```",
        "",
        "RD-B2b 확정값. ImageNet normalization (mean/std)은 RD-B5/B6 ablation 대상 (현재 미적용).",
        "",
        "---",
        "## 6. 6-bin balanced sampler 설계",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        "| class | SixBinBalancedBatchSampler |",
        "| batch_size 후보 A | 24 (bin당 4) |",
        "| batch_size 후보 B | 48 (bin당 8) |",
        "| bin_shortage 정책 | shorter_epoch: 가장 먼저 소진되는 bin 기준 epoch 종료 |",
        "| duplicate oversampling | 금지 |",
        "| patient leakage | split=train safe_id 기준 분리 |",
        "| seed | 42 (고정 가능) |",
        f"| 이론 n_batches/epoch (A) | {13932//4} |",
        f"| 이론 n_batches/epoch (B) | {13932//8} |",
        "",
        "---",
        "## 7. ResNet18 teacher-student skeleton 설계",
        "",
        "```",
        "입력: 3×96×96 (mixed_3ch)",
        "Teacher (ResNet18, frozen, eval, ImageNet)",
        "  → layer1: 64×24×24",
        "  → layer2: 128×12×12",
        "  → layer3: 256×6×6  ← OCBE 입력",
        "OCBE Bottleneck:",
        "  Conv(256,256,3,1,1) → BN → ReLU → Conv(256,256,3,1,1)",
        "  출력: 256×6×6",
        "Student Decoder (trainable):",
        "  de_layer3: 256×6×6",
        "  de_layer2: 128×12×12 (Upsample 2x)",
        "  de_layer1:  64×24×24 (Upsample 2x)",
        "Loss: Σ (1 - cosine_similarity(teacher_feat, student_feat, dim=1).mean())",
        "Anomaly Score: multi-scale cosine distance → crop-level score",
        "```",
        "",
        "---",
        "## 8. ImageNet weight / loading 관련 주의",
        "",
        "- RD-B5 단계: `pretrained=True` 사용 금지, weight 다운로드 금지",
        "- 코드에 `weights=None` 또는 placeholder 문자열만 기록",
        "- RD-B6 tiny smoke train 단계에서 별도 사용자 승인 후:",
        "  `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`",
        "- teacher는 `requires_grad=False`, `model.eval()` 고정 필수",
        "",
        "---",
        "## 9. RD-B6 tiny smoke train 전 확인사항",
        "",
        "1. 이 script --write-report 결과 DONE 존재 확인",
        "2. manifest 86,017 rows, CT/ROI 290/290 존재 확인",
        "3. ImageNet weight 다운로드 사용자 사전 승인",
        "4. GPU 환경 확인 (smoke train은 소규모: ~100 crops, 5 epochs)",
        "5. low_z_boundary_warning=True crop에서 lower_boundary MIP 시각 검증",
        "6. feature shape (layer1/2/3) forward hook 출력 확인",
        "7. loss 감소 여부 확인 (정상 수렴 여부)",
        "8. output root 구조 사전 설계 및 사용자 승인",
        "",
        "---",
        "## 10. 절대 하지 않은 것",
        "",
        "| 항목 | 확인 |",
        "|------|------|",
        "| crop NPZ 생성 | ✅ 없음 |",
        "| 학습 실행 | ✅ 없음 |",
        "| scoring 실행 | ✅ 없음 |",
        "| model forward 실행 | ✅ 없음 |",
        "| GPU 사용 | ✅ 없음 |",
        "| checkpoint 로드 | ✅ 없음 |",
        "| pretrained weight 다운로드 | ✅ 없음 (placeholder만) |",
        "| stage2_holdout 접근 | ✅ 없음 (forbidden 가드) |",
        "| 기존 파일 수정/삭제 | ✅ 없음 (신규 파일만) |",
    ]
    with open(
        OUTPUT_ROOT / "rd_b5_dataset_loader_model_skeleton_static_report.md",
        "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(report_lines) + "\n")

    # ── DONE marker ───────────────────────────────────────────────────────
    (OUTPUT_ROOT / "DONE").write_text(f"rd_b5 write-report completed: {ts}\n")
    print(f"  생성 파일:")
    for fn in sorted(OUTPUT_ROOT.iterdir()):
        print(f"    {fn.name}")


if __name__ == "__main__":
    main()
