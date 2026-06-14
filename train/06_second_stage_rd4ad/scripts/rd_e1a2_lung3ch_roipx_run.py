"""
RD-E1a: Lung window 3ch true RD4AD shard-based full run
목적: 실험 A — lung window [-1000,600] + z-1/z/z+1 단순 스택.
     이후 normal_val threshold, RD-C2 candidate 113,447개 scoring, AUROC/AUPRC 분석.

모드:
  bare run         -> exit 2
  --dry-plan       -> 입력/output root 확인 (파일 생성 없음)
  --run-all-shard  -> shard 학습 + threshold + scoring + analysis + DONE

안전 조건:
  train 단계에서 on-the-fly crop 생성 금지 (shard만 사용)
  normal_val / candidate scoring은 on-the-fly 허용
  stage2_holdout 접근 금지
  기존 RD-B/RD-C 결과 수정 금지
  suppression 적용 금지 (analysis-only)
  output root 이미 있으면 즉시 ABORT
  기존 RD-D1 on-the-fly output root 사용 금지
  score NaN/Inf 발생 시 ABORT
  sklearn 사용 금지 (Mann-Whitney AUROC 직접 구현)
"""

import sys
import csv
import json
import math
import time
import collections
from pathlib import Path

ALLOWED_MODES = {"--dry-plan", "--run-all-shard"}
if not any(m in sys.argv for m in ALLOWED_MODES):
    print("오류: 실행 모드 지정 필요.")
    print("  --dry-plan       : 입력 확인 (파일 생성 없음)")
    print("  --run-all-shard  : shard 기반 full train + scoring + analysis + DONE")
    sys.exit(2)

IS_DRY_PLAN      = "--dry-plan"      in sys.argv
IS_RUN_ALL_SHARD = "--run-all-shard" in sys.argv

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

# shard 입력
SHARD_ROOT        = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_e1a2_lung3ch_roipx_train_shards_v1"
)
SHARD_INDEX_CSV   = SHARD_ROOT / "rd_d1s_full_shard_index.csv"
SHARD_SUMMARY_JSON = SHARD_ROOT / "rd_d1s_full_shard_generation_summary.json"
SHARD_DONE_MARKER = SHARD_ROOT / "DONE_SHARD_BUILD"

# CT root (normal_val / candidate scoring용 on-the-fly)
NORMAL_CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1"
    "/volumes_npy"
)
CANDIDATE_CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)

# manifest
NORMAL_VAL_MANIFEST = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b9_normal_val_scoring_threshold_v1"
    / "rd_b9_normal_val_sixbin_manifest.csv"
)
RD_C2_MANIFEST = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c2_effb0_v420_candidate_rd4ad_retest_v1"
    / "rd_c2_effb0_v420_candidate_manifest.csv"
)

# 출력
MODEL_ROOT = (
    PROJECT_ROOT / "outputs/models/rd_e1a2_true_rd4ad_resnet18_lung3ch_roipx_shard_v1"
)
CKPT_DIR = MODEL_ROOT / "checkpoints"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_e1a2_lung3ch_roipx_true_rd4ad_shard_run_v1"
)

# 기존 RD-D1 on-the-fly output (사용 금지)
RD_D1_ONTHEFLY_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1_medi3ch_true_rd4ad_revival_v1"
)

LOCAL_WEIGHT_PATH = Path(
    "/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)

# ROI 마스크 root (A2 픽셀 마스킹)
MASK_ROOT_NORMAL = (
    PROJECT_ROOT
    / "outputs/mip-postprocess-research-v1/masks"
    / "refined_roi_v4_25_modeB_train_v1"
)
MASK_ROOT_CAND = (
    PROJECT_ROOT
    / "outputs/mip-postprocess-research-v1/masks"
    / "refined_roi_v4_20_modeB_all_v1"
)

# ── 설계 상수 ──────────────────────────────────────────────────────────────────
FORBIDDEN_KEYWORDS = [
    "stage2_holdout", "second-stage-lesion-refiner",
    "test_lesion", "lesion_refiner",
]
SIX_BIN_LABELS = [
    "lower_boundary", "lower_interior",
    "middle_boundary", "middle_interior",
    "upper_boundary",  "upper_interior",
]

CROP_SIZE    = 96
BATCH_SIZE   = 48
PER_BIN      = 8
EPOCHS       = 20
LR           = 1e-4
WEIGHT_DECAY = 1e-5
SEED         = 42
MEDI_HU_MIN  = -1000.0
MEDI_HU_MAX  =   600.0

SHARD_ROWS_EXPECTED  = 86017
SHARD_COUNT_EXPECTED = 44
STEPS_PER_EPOCH      = 1741   # min_bin(13932) // per_bin(8)
TOTAL_STEPS_EXPECTED = 34820  # 1741 × 20

RD_B8F_AUROC_REF = 0.5021
RD_C3_AUROC_REF  = 0.7262

# ── padding 카운터 ─────────────────────────────────────────────────────────────
_G_PAD_APPLIED_COUNT: int = 0
_G_PAD_REFLECT_COUNT: int = 0
_G_PAD_EDGE_COUNT:    int = 0


# =============================================================================
# 안전 검사
# =============================================================================

def assert_path_safe(p):
    for kw in FORBIDDEN_KEYWORDS:
        if kw.lower() in str(p).lower():
            raise RuntimeError(
                f"[SAFETY] 금지 경로 접근 차단: {p!r} (keyword={kw!r})"
            )


def fmt_float(x, ndigits=4):
    if x is None:
        return "N/A"
    try:
        if math.isnan(float(x)):
            return "N/A"
    except Exception:
        pass
    return f"{float(x):.{ndigits}f}"


# =============================================================================
# sklearn-free metrics
# =============================================================================

def compute_auroc_mann_whitney(y_true, y_score):
    import numpy as np
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    valid   = np.isfinite(y_score)
    y_true  = y_true[valid]
    y_score = y_score[valid]
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty(len(sorted_scores), dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        i = j
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum_pos = float(original_ranks[y_true == 1].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def compute_average_precision(y_true, y_score):
    import numpy as np
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    valid   = np.isfinite(y_score)
    y_true  = y_true[valid]
    y_score = y_score[valid]
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    score_sorted  = y_score[order]
    tp = 0; fp = 0; prev_recall = 0.0; ap = 0.0; i = 0
    while i < len(score_sorted):
        j = i + 1
        while j < len(score_sorted) and score_sorted[j] == score_sorted[i]:
            j += 1
        group = y_true_sorted[i:j]
        tp += int((group == 1).sum())
        fp += int((group == 0).sum())
        recall = tp / n_pos
        precision = tp / max(tp + fp, 1)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return float(ap)


# =============================================================================
# 모델 빌드
# =============================================================================

def build_teacher():
    import torch
    import torchvision.models as models
    resnet = models.resnet18(weights=None)
    state_dict = torch.load(
        str(LOCAL_WEIGHT_PATH), map_location="cpu", weights_only=True
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


def rd_loss_fn(teacher_feats, student_feats):
    import torch
    import torch.nn.functional as F
    loss = torch.tensor(0.0, device=teacher_feats[0].device)
    for tf, sf in zip(teacher_feats, student_feats):
        cos_sim = F.cosine_similarity(sf, tf, dim=1)
        loss = loss + (1 - cos_sim).mean()
    return loss / len(teacher_feats)


def snapshot_params(model):
    import torch
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
    }


def params_changed(snap_before, snap_after):
    import torch
    for name in snap_before:
        if not torch.equal(snap_before[name], snap_after[name]):
            return True
    return False


# =============================================================================
# Shard mmap cache
# =============================================================================

class ShardMmapCache:
    """pre-built shard npy mmap LRU 캐시"""
    def __init__(self, max_size=8):
        self._cache = collections.OrderedDict()
        self._max   = max_size

    def get(self, shard_path):
        import numpy as np
        key = str(shard_path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        arr = np.load(key, mmap_mode="r")
        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        self._cache[key] = arr
        return arr


# =============================================================================
# CT mmap cache (normal_val / candidate scoring용)
# =============================================================================

class CtMmapCache:
    def __init__(self, max_size=16):
        self._cache = collections.OrderedDict()
        self._max   = max_size

    def get(self, ct_path):
        import numpy as np
        key = str(ct_path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        arr = np.load(key, mmap_mode="r")
        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        self._cache[key] = arr
        return arr


# =============================================================================
# 6-bin Shard Sampler (shard index 기반)
# =============================================================================

class SixBinShardSampler:
    """
    strict 6-bin balanced, shortest-bin drop-last.
    shard index CSV의 six_bin_label, shard_id, offset 사용.
    각 step마다 6개 bin에서 per_bin=8개씩 → batch_size=48.
    """
    def __init__(self, shard_index_csv, per_bin, seed):
        import random
        self.per_bin = per_bin
        self._seed   = seed

        self.bin_items = {lbl: [] for lbl in SIX_BIN_LABELS}
        with open(shard_index_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lbl = row.get("six_bin_label", "")
                if lbl not in self.bin_items:
                    continue
                shard_id = int(row["shard_id"])
                offset   = int(row["offset"])
                self.bin_items[lbl].append((shard_id, offset))

        bin_sizes = {lbl: len(v) for lbl, v in self.bin_items.items()}
        self.min_bin_size    = min(bin_sizes.values())
        self.steps_per_epoch = self.min_bin_size // per_bin

        print("  6-bin 분포 (shard index):")
        for lbl in SIX_BIN_LABELS:
            print(f"    {lbl}: {bin_sizes[lbl]:,}")
        print(f"  min_bin_size={self.min_bin_size:,}  "
              f"per_bin={per_bin}  steps_per_epoch={self.steps_per_epoch:,}")

    def epoch_batches(self, epoch):
        """yield: (step, list[(shard_id, offset)])"""
        import random
        rng = random.Random(self._seed + epoch * 997)

        shuffled = {}
        for lbl in SIX_BIN_LABELS:
            items = list(self.bin_items[lbl])
            rng.shuffle(items)
            shuffled[lbl] = items[:self.steps_per_epoch * self.per_bin]

        for step in range(self.steps_per_epoch):
            batch = []
            for lbl in SIX_BIN_LABELS:
                s = step * self.per_bin
                batch.extend(shuffled[lbl][s:s + self.per_bin])
            yield step, batch


# =============================================================================
# mediastinal 3ch crop builder (normal_val / candidate scoring용)
# =============================================================================

def build_lung3ch_crop(ct_arr, local_z, y0, x0, y1, x1):
    """ct_arr: (Z,H,W) mmap → (3,96,96) float32 [0,1]
    OOB crop coords → reflect padding (edge fallback if valid region <= 1px).
    """
    global _G_PAD_APPLIED_COUNT, _G_PAD_REFLECT_COUNT, _G_PAD_EDGE_COUNT
    import numpy as np
    Z, H, W = ct_arr.shape
    z  = int(local_z)
    zm = max(z - 1, 0)
    zp = min(z + 1, Z - 1)
    y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)

    pad_top    = max(0, -y0)
    pad_bottom = max(0, y1 - H)
    pad_left   = max(0, -x0)
    pad_right  = max(0, x1 - W)
    needs_pad  = (pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0)

    cy0 = max(0, y0)
    cy1 = min(H, y1)
    cx0 = max(0, x0)
    cx1 = min(W, x1)
    valid_h = cy1 - cy0
    valid_w = cx1 - cx0

    if needs_pad:
        can_reflect = (valid_h > 1) and (valid_w > 1)
        pad_mode    = "reflect" if can_reflect else "edge"

    def _win(sl):
        c = np.clip(sl.astype(np.float32), MEDI_HU_MIN, MEDI_HU_MAX)
        return (c - MEDI_HU_MIN) / (MEDI_HU_MAX - MEDI_HU_MIN)

    def _build_ch(z_idx):
        normed = _win(ct_arr[z_idx, cy0:cy1, cx0:cx1])
        if needs_pad:
            normed = np.pad(normed,
                            ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode=pad_mode)
        return normed

    ch0 = _build_ch(zm)
    ch1 = _build_ch(z)
    ch2 = _build_ch(zp)
    crop = np.stack([ch0, ch1, ch2], axis=0)

    if crop.shape != (3, CROP_SIZE, CROP_SIZE):
        raise RuntimeError(
            f"[ABORT] crop shape {crop.shape} != (3,{CROP_SIZE},{CROP_SIZE})  "
            f"y0={y0} x0={x0} y1={y1} x1={x1} H={H} W={W} "
            f"pad=({pad_top},{pad_bottom},{pad_left},{pad_right})"
        )
    if not np.isfinite(crop).all():
        raise RuntimeError(
            f"[ABORT] crop contains NaN/Inf  "
            f"y0={y0} x0={x0} y1={y1} x1={x1}"
        )

    if needs_pad:
        _G_PAD_APPLIED_COUNT += 1
        if can_reflect:
            _G_PAD_REFLECT_COUNT += 1
        else:
            _G_PAD_EDGE_COUNT += 1

    return crop.astype(np.float32)


def apply_roi_mask_3ch(crop, mask_arr, z, y0, x0, y1, x1):
    """Per-channel ROI 마스킹: ch0=mask[z-1], ch1=mask[z], ch2=mask[z+1]"""
    if mask_arr is None:
        return crop
    import numpy as np
    Z, H, W = mask_arr.shape
    z = int(z)
    zm = max(z - 1, 0)
    zp = min(z + 1, Z - 1)
    cy0 = max(0, y0); cy1 = min(H, y1)
    cx0 = max(0, x0); cx1 = min(W, x1)
    pad_top    = max(0, -y0); pad_bottom = max(0, y1 - H)
    pad_left   = max(0, -x0); pad_right  = max(0, x1 - W)
    needs_pad  = pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0

    def _get_roi(zi):
        r = mask_arr[int(np.clip(zi, 0, Z - 1)), cy0:cy1, cx0:cx1].astype(bool)
        if needs_pad:
            r = np.pad(r, ((pad_top, pad_bottom), (pad_left, pad_right)),
                       mode="constant", constant_values=False)
        return r

    crop = crop.copy()
    crop[0][~_get_roi(zm)] = 0.0
    crop[1][~_get_roi(z)]  = 0.0
    crop[2][~_get_roi(zp)] = 0.0
    return crop.astype(np.float32)


def find_cand_mask(safe_id):
    """candidate 마스크 경로 반환 (lesion/ 우선, 없으면 normal/)"""
    for sub in ("lesion", "normal"):
        p = MASK_ROOT_CAND / sub / safe_id / "refined_roi.npy"
        if p.exists():
            return p
    return None


# =============================================================================
# CSV 헬퍼
# =============================================================================

class CsvAppendWriter:
    def __init__(self, path, fieldnames):
        self.fieldnames = fieldnames
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=fieldnames, extrasaction="ignore")
        self._w.writeheader()

    def writerow(self, row):
        self._w.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._f.flush()

    def close(self):
        self._f.close()


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  saved: {path.name}")


# =============================================================================
# dry-plan
# =============================================================================

def run_dry_plan():
    print("=" * 72)
    print("RD-D1s: Mediastinal 3ch true RD4AD shard-based full run [DRY-PLAN]")
    print("=" * 72)

    ok_all = True

    print("\n[1] shard 입력 확인")
    shard_checks = [
        ("SHARD_ROOT",          SHARD_ROOT),
        ("SHARD_INDEX_CSV",     SHARD_INDEX_CSV),
        ("SHARD_SUMMARY_JSON",  SHARD_SUMMARY_JSON),
        ("SHARD_DONE_MARKER",   SHARD_DONE_MARKER),
        ("LOCAL_WEIGHT_PATH",   LOCAL_WEIGHT_PATH),
        ("NORMAL_VAL_MANIFEST", NORMAL_VAL_MANIFEST),
        ("RD_C2_MANIFEST",      RD_C2_MANIFEST),
    ]
    for label, p in shard_checks:
        exists = p.exists()
        print(f"  {'OK' if exists else 'MISSING'}: {label} -> {p}")
        if not exists:
            ok_all = False

    print("\n[2] shard 파일 수 확인")
    shard_files = sorted(SHARD_ROOT.glob("shard_*.npy")) if SHARD_ROOT.exists() else []
    ok_shard_cnt = len(shard_files) == SHARD_COUNT_EXPECTED
    print(f"  shard files: {len(shard_files)}/{SHARD_COUNT_EXPECTED}  {'OK' if ok_shard_cnt else 'FAIL'}")
    if not ok_shard_cnt:
        ok_all = False

    print("\n[3] shard index row 수 확인")
    if SHARD_INDEX_CSV.exists():
        with open(SHARD_INDEX_CSV, newline="", encoding="utf-8") as f:
            idx_rows = sum(1 for _ in csv.DictReader(f))
        ok_rows = idx_rows == SHARD_ROWS_EXPECTED
        print(f"  shard index rows: {idx_rows:,}/{SHARD_ROWS_EXPECTED:,}  {'OK' if ok_rows else 'FAIL'}")
        if not ok_rows:
            ok_all = False

    print("\n[4] output root guard")
    for label, p in [
        ("MODEL_ROOT",            MODEL_ROOT),
        ("OUTPUT_ROOT",           OUTPUT_ROOT),
        ("RD_D1_ONTHEFLY (check)", RD_D1_ONTHEFLY_ROOT),
    ]:
        exists = p.exists()
        if label.startswith("RD_D1"):
            print(f"  {'WARN(exists)' if exists else 'OK(not exist)'}: {label} -> {p}")
        else:
            print(f"  {'CONFLICT' if exists else 'OK'}: {label} -> {p}")
            if exists:
                ok_all = False

    print("\n[5] stage2_holdout intersection (RD-C2 manifest)")
    if RD_C2_MANIFEST.exists():
        with open(RD_C2_MANIFEST, newline="", encoding="utf-8") as f:
            c2_rows = list(csv.DictReader(f))
        holdout_cnt = sum(1 for r in c2_rows if r.get("stage_split", "") == "stage2_holdout")
        pos_cnt     = sum(1 for r in c2_rows if r.get("label", "") == "positive")
        hn_cnt      = sum(1 for r in c2_rows if r.get("label", "") == "hard_negative")
        ok_holdout  = holdout_cnt == 0
        print(f"  rows={len(c2_rows):,}  positive={pos_cnt:,}  hard_negative={hn_cnt:,}")
        print(f"  stage2_holdout intersection: {holdout_cnt}  {'OK' if ok_holdout else 'FAIL'}")
        if not ok_holdout:
            ok_all = False

    print("\n[6] 학습 설정")
    print(f"  train_source    : prebuilt_float32_shards")
    print(f"  input_mode      : mediastinal_3ch_zminus1_z_zplus1")
    print(f"  window          : HU[{MEDI_HU_MIN},{MEDI_HU_MAX}] -> [0,1]")
    print(f"  model_type      : true_RD4AD_ResNet18_teacher_student")
    print(f"  crop_size       : {CROP_SIZE}")
    print(f"  batch_size      : {BATCH_SIZE}  per_bin: {PER_BIN}")
    print(f"  epochs          : {EPOCHS}  lr: {LR}  wd: {WEIGHT_DECAY}")
    print(f"  sampler         : 6-bin balanced, shortest-bin drop-last (shard index)")
    print(f"  shard_rows      : {SHARD_ROWS_EXPECTED:,}")
    print(f"  shard_count     : {SHARD_COUNT_EXPECTED}")
    print(f"  steps/epoch     : {STEPS_PER_EPOCH:,}")
    print(f"  total_steps     : {TOTAL_STEPS_EXPECTED:,}")
    est_train_sec = STEPS_PER_EPOCH * EPOCHS * 0.0242
    print(f"  est train time  : ~{est_train_sec/60:.1f}min (0.0242s/batch 기준)")

    print("\n[7] 안전 조건")
    print("  train on-the-fly crop   : 금지 (shard만 사용)")
    print("  val/score on-the-fly    : 허용")
    print("  stage2_holdout_access   : 0")
    print("  suppression_applied     : false")
    print("  sklearn_used            : false")
    print("  existing_results_modified: false")

    print()
    verdict = "DRY-PLAN OK" if ok_all else "DRY-PLAN FAIL"
    print(f"판정: {verdict}")
    if ok_all:
        print("  사용자 승인 후:")
        print("  source ~/ai_env/bin/activate && \\")
        print("  python scripts/rd_d1s_medi3ch_true_rd4ad_shard_run_all.py --run-all-shard \\")
        print("  2>&1 | tee /tmp/rd_d1s_medi3ch_true_rd4ad_shard_run_log.txt")
    return ok_all


# =============================================================================
# run_all_shard
# =============================================================================

def run_all_shard():
    import numpy as np
    import torch
    import torch.nn.functional as F

    print("=" * 72)
    print("RD-D1s: Mediastinal 3ch true RD4AD shard-based full run [RUN-ALL-SHARD]")
    print("=" * 72)
    t_start = time.perf_counter()

    # ── output root guard ────────────────────────────────────────────────────
    for label, root in [("MODEL_ROOT", MODEL_ROOT), ("OUTPUT_ROOT", OUTPUT_ROOT)]:
        if root.exists():
            print(f"[ABORT] {label} 이미 존재: {root}", file=sys.stderr)
            sys.exit(1)
    MODEL_ROOT.mkdir(parents=True)
    CKPT_DIR.mkdir(parents=True)
    OUTPUT_ROOT.mkdir(parents=True)

    error_rows = []

    # ── shard 전제 조건 확인 ──────────────────────────────────────────────────
    print("\n[1/7] shard 전제 조건 확인")
    if not SHARD_DONE_MARKER.exists():
        print(f"[ABORT] DONE_SHARD_BUILD marker 없음: {SHARD_DONE_MARKER}", file=sys.stderr)
        sys.exit(1)
    if not SHARD_INDEX_CSV.exists():
        print(f"[ABORT] shard index CSV 없음: {SHARD_INDEX_CSV}", file=sys.stderr)
        sys.exit(1)
    if not LOCAL_WEIGHT_PATH.exists():
        print(f"[ABORT] ResNet18 local weight 없음: {LOCAL_WEIGHT_PATH}", file=sys.stderr)
        sys.exit(1)

    shard_files = sorted(SHARD_ROOT.glob("shard_*.npy"))
    if len(shard_files) != SHARD_COUNT_EXPECTED:
        print(f"[ABORT] shard 파일 수 불일치: {len(shard_files)} != {SHARD_COUNT_EXPECTED}",
              file=sys.stderr)
        sys.exit(1)

    shard_size_gb = SHARD_SUMMARY_JSON
    shard_gb_val = 8.859
    if SHARD_SUMMARY_JSON.exists():
        with open(SHARD_SUMMARY_JSON, encoding="utf-8") as f:
            shard_summary = json.load(f)
        shard_gb_val = shard_summary.get("total_shard_size_gb", 8.859)

    # shard index row 수 확인
    with open(SHARD_INDEX_CSV, newline="", encoding="utf-8") as f:
        idx_row_count = sum(1 for _ in csv.DictReader(f))
    if idx_row_count != SHARD_ROWS_EXPECTED:
        print(f"[ABORT] shard index rows={idx_row_count} != {SHARD_ROWS_EXPECTED}",
              file=sys.stderr)
        sys.exit(1)
    print(f"  shard files={len(shard_files)}  index_rows={idx_row_count:,}  OK")

    # ── RD-C2 manifest 로드 ──────────────────────────────────────────────────
    print("\n[2/7] RD-C2 candidate manifest 로드")
    if not RD_C2_MANIFEST.exists():
        print(f"[ABORT] manifest 없음: {RD_C2_MANIFEST}", file=sys.stderr)
        sys.exit(1)
    with open(RD_C2_MANIFEST, newline="", encoding="utf-8") as f:
        c2_rows = list(csv.DictReader(f))
    holdout_cnt = sum(1 for r in c2_rows if r.get("stage_split", "") == "stage2_holdout")
    if holdout_cnt != 0:
        print(f"[ABORT] stage2_holdout intersection={holdout_cnt}", file=sys.stderr)
        sys.exit(1)
    pos_cnt = sum(1 for r in c2_rows if r.get("label", "") == "positive")
    hn_cnt  = sum(1 for r in c2_rows if r.get("label", "") == "hard_negative")
    print(f"  rows={len(c2_rows):,}  positive={pos_cnt:,}  hard_negative={hn_cnt:,}  holdout=0 OK")

    write_csv(
        OUTPUT_ROOT / "rd_d1s_input_validation.csv",
        ["check", "result", "pass"],
        [
            {"check": "stage2_holdout_intersection",  "result": holdout_cnt,   "pass": holdout_cnt == 0},
            {"check": "positive_count",               "result": pos_cnt,       "pass": pos_cnt == 35247},
            {"check": "hard_negative_count",          "result": hn_cnt,        "pass": hn_cnt == 78200},
            {"check": "shard_index_rows",             "result": idx_row_count, "pass": idx_row_count == SHARD_ROWS_EXPECTED},
            {"check": "shard_file_count",             "result": len(shard_files), "pass": len(shard_files) == SHARD_COUNT_EXPECTED},
            {"check": "normal_val_manifest_exists",   "result": NORMAL_VAL_MANIFEST.exists(), "pass": NORMAL_VAL_MANIFEST.exists()},
            {"check": "local_weights_exists",         "result": LOCAL_WEIGHT_PATH.exists(),   "pass": LOCAL_WEIGHT_PATH.exists()},
        ],
    )

    # ── shard sampler + 모델 ─────────────────────────────────────────────────
    print("\n[3/7] 6-bin shard sampler 로드 + 모델 준비")
    sampler     = SixBinShardSampler(SHARD_INDEX_CSV, per_bin=PER_BIN, seed=SEED)
    shard_cache = ShardMmapCache(max_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    teacher = build_teacher().to(device)
    student = build_student_decoder().to(device)
    student.train()

    teacher_features = {}
    for layer_name, module in [
        ("layer1", teacher.layer1),
        ("layer2", teacher.layer2),
        ("layer3", teacher.layer3),
    ]:
        def _hook(module, inp, output, _n=layer_name):
            teacher_features[_n] = output
        module.register_forward_hook(_hook)

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    teacher_snap_before = snapshot_params(teacher)

    teacher_param_set = set(id(p) for p in teacher.parameters())
    opt_teacher_count = sum(
        1 for pg in optimizer.param_groups
        for p in pg["params"]
        if id(p) in teacher_param_set
    )
    print(f"  optimizer teacher param count: {opt_teacher_count}")
    if opt_teacher_count != 0:
        print("[ABORT] optimizer가 teacher param 포함 - 설계 오류", file=sys.stderr)
        sys.exit(1)

    # ── 학습 루프 (shard 기반) ───────────────────────────────────────────────
    print(f"\n[4/7] shard 기반 학습 시작 ({EPOCHS} epochs, "
          f"steps/epoch={sampler.steps_per_epoch:,})")

    epoch_log_writer = CsvAppendWriter(
        OUTPUT_ROOT / "rd_d1s_epoch_log.csv",
        ["epoch", "steps", "mean_loss", "min_loss", "max_loss",
         "nan_count", "inf_count", "epoch_time_sec", "cumulative_time_sec"],
    )
    batch_log_writer = CsvAppendWriter(
        OUTPUT_ROOT / "rd_d1s_batch_runtime_summary.csv",
        ["epoch", "step", "loss", "loss_nan", "loss_inf",
         "load_time_sec", "fwd_bwd_time_sec"],
    )

    best_loss   = float("inf")
    best_epoch  = -1
    epoch_logs  = []
    total_nan   = 0
    total_inf   = 0
    t_train_start = time.perf_counter()
    student_snap_before = snapshot_params(student)

    for epoch in range(EPOCHS):
        student.train()
        epoch_losses = []
        epoch_nan    = 0
        epoch_inf    = 0
        t_epoch_start = time.perf_counter()

        print(f"\n  [Epoch {epoch+1}/{EPOCHS}]")

        for step, batch_items in sampler.epoch_batches(epoch):
            t_load_s = time.perf_counter()
            crops = []
            for shard_id, offset in batch_items:
                shard_path = SHARD_ROOT / f"shard_{shard_id:04d}.npy"
                shard_arr  = shard_cache.get(shard_path)
                crop_np    = shard_arr[offset].astype("float32")
                crops.append(crop_np)
            batch_np = np.stack(crops, axis=0)
            t_load_e = time.perf_counter()

            t_fwd_s = time.perf_counter()
            batch_t = torch.from_numpy(batch_np).to(device)

            with torch.no_grad():
                teacher(batch_t)

            tf3 = teacher_features["layer3"]
            tf2 = teacher_features["layer2"]
            tf1 = teacher_features["layer1"]

            de3, de2, de1 = student(tf3)
            loss = rd_loss_fn([tf3, tf2, tf1], [de3, de2, de1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_fwd_e = time.perf_counter()

            loss_val = float(loss.item())
            is_nan   = int(math.isnan(loss_val))
            is_inf   = int(math.isinf(loss_val))
            epoch_nan += is_nan
            epoch_inf += is_inf

            if is_nan:
                print(f"[ABORT] loss NaN at epoch={epoch+1} step={step}", file=sys.stderr)
                sys.exit(1)
            if is_inf:
                print(f"[ABORT] loss Inf at epoch={epoch+1} step={step}", file=sys.stderr)
                sys.exit(1)

            epoch_losses.append(loss_val)
            batch_log_writer.writerow({
                "epoch":          epoch + 1,
                "step":           step,
                "loss":           round(loss_val, 6),
                "loss_nan":       is_nan,
                "loss_inf":       is_inf,
                "load_time_sec":  round(t_load_e - t_load_s, 4),
                "fwd_bwd_time_sec": round(t_fwd_e - t_fwd_s, 4),
            })

            if step % 200 == 0 or step == sampler.steps_per_epoch - 1:
                elapsed = time.perf_counter() - t_epoch_start
                print(
                    f"    step {step:5d}/{sampler.steps_per_epoch}  "
                    f"loss={loss_val:.4f}  load={t_load_e-t_load_s:.3f}s  "
                    f"elapsed={elapsed:.0f}s"
                )

        t_epoch_end  = time.perf_counter()
        epoch_time   = t_epoch_end - t_epoch_start
        cumul_time   = t_epoch_end - t_train_start
        valid_losses = [v for v in epoch_losses if not (math.isnan(v) or math.isinf(v))]
        mean_loss    = sum(valid_losses) / len(valid_losses) if valid_losses else float("nan")
        min_loss     = min(valid_losses) if valid_losses else float("nan")
        max_loss     = max(valid_losses) if valid_losses else float("nan")
        total_nan   += epoch_nan
        total_inf   += epoch_inf

        epoch_log_writer.writerow({
            "epoch":               epoch + 1,
            "steps":               len(epoch_losses),
            "mean_loss":           round(mean_loss, 6),
            "min_loss":            round(min_loss, 6),
            "max_loss":            round(max_loss, 6),
            "nan_count":           epoch_nan,
            "inf_count":           epoch_inf,
            "epoch_time_sec":      round(epoch_time, 2),
            "cumulative_time_sec": round(cumul_time, 2),
        })
        epoch_logs.append({"epoch": epoch + 1, "mean_loss": mean_loss})

        # checkpoint: last.pth
        torch.save(
            {"epoch": epoch + 1, "student_state_dict": student.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(), "mean_loss": mean_loss},
            CKPT_DIR / "last.pth",
        )
        # checkpoint: best_train_loss.pth
        if not math.isnan(mean_loss) and mean_loss < best_loss:
            best_loss  = mean_loss
            best_epoch = epoch + 1
            torch.save(
                {"epoch": epoch + 1, "student_state_dict": student.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "mean_loss": mean_loss},
                CKPT_DIR / "best_train_loss.pth",
            )

        print(
            f"  Epoch {epoch+1}: mean_loss={mean_loss:.4f}  "
            f"best={best_loss:.4f}(ep{best_epoch})  "
            f"time={epoch_time:.1f}s  NaN={epoch_nan}  Inf={epoch_inf}"
        )

    epoch_log_writer.close()
    batch_log_writer.close()
    t_train_end = time.perf_counter()
    train_time  = t_train_end - t_train_start
    gpu_peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda" else 0.0
    )

    teacher_snap_after    = snapshot_params(teacher)
    student_snap_after    = snapshot_params(student)
    teacher_param_changed = params_changed(teacher_snap_before, teacher_snap_after)
    student_param_changed = params_changed(student_snap_before, student_snap_after)

    if teacher_param_changed:
        print("[ABORT] teacher param 변경됨", file=sys.stderr)
        sys.exit(1)
    assert student_param_changed, "[ABORT] student param 변경 없음"

    valid_epoch_losses = [r["mean_loss"] for r in epoch_logs if not math.isnan(r["mean_loss"])]
    train_loss_first   = valid_epoch_losses[0]  if valid_epoch_losses else float("nan")
    train_loss_last    = valid_epoch_losses[-1] if valid_epoch_losses else float("nan")
    loss_decreased     = (
        len(valid_epoch_losses) >= 2
        and valid_epoch_losses[-1] < valid_epoch_losses[0]
    )
    print(f"\n  train complete: {EPOCHS}ep  {train_time:.1f}s  "
          f"loss {train_loss_first:.4f}→{train_loss_last:.4f}  best_ep={best_epoch}  "
          f"teacher_changed={teacher_param_changed}  student_changed={student_param_changed}")

    # best checkpoint 로드
    ckpt = torch.load(
        str(CKPT_DIR / "best_train_loss.pth"), map_location=device, weights_only=False
    )
    student.load_state_dict(ckpt["student_state_dict"])
    student.eval()
    print(f"  best_epoch={best_epoch}  best_loss={best_loss:.6f}")

    # ── normal_val threshold ─────────────────────────────────────────────────
    print("\n[5/7] normal_val threshold 생성 (on-the-fly)")
    if not NORMAL_VAL_MANIFEST.exists():
        print(f"[ABORT] normal_val manifest 없음: {NORMAL_VAL_MANIFEST}", file=sys.stderr)
        sys.exit(1)
    with open(NORMAL_VAL_MANIFEST, newline="", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))
    val_safe_ids = set(r["safe_id"] for r in val_rows)
    print(f"  val rows={len(val_rows):,}  patients={len(val_safe_ids)}")

    val_scores  = []
    val_errors  = []
    val_by_bin  = collections.defaultdict(list)
    val_groups  = collections.defaultdict(list)
    for r in val_rows:
        val_groups[r["safe_id"]].append(r)

    val_ct_cache   = CtMmapCache(max_size=8)
    val_mask_cache = CtMmapCache(max_size=48)
    BATCH_VAL    = 64
    n_val_scored = 0

    for safe_id, v_rows in val_groups.items():
        ct_path_str = v_rows[0].get("ct_hu_npy", "")
        ct_path = Path(ct_path_str) if ct_path_str else (NORMAL_CT_ROOT / safe_id / "ct_hu.npy")
        assert_path_safe(ct_path)
        if not ct_path.exists():
            val_errors.append({"safe_id": safe_id, "error": f"ct_not_found:{ct_path}"})
            continue
        ct_arr = val_ct_cache.get(ct_path)
        mask_path = MASK_ROOT_NORMAL / safe_id / "refined_roi.npy"
        mask_arr = val_mask_cache.get(mask_path) if mask_path.exists() else None

        for i in range(0, len(v_rows), BATCH_VAL):
            batch_rows = v_rows[i:i + BATCH_VAL]
            crops = [
                apply_roi_mask_3ch(
                    build_lung3ch_crop(
                        ct_arr,
                        int(row["local_z"]),
                        int(row["crop_y0"]), int(row["crop_x0"]),
                        int(row["crop_y1"]), int(row["crop_x1"]),
                    ),
                    mask_arr,
                    int(row["local_z"]),
                    int(row["crop_y0"]), int(row["crop_x0"]),
                    int(row["crop_y1"]), int(row["crop_x1"]),
                )
                for row in batch_rows
            ]
            batch_t = torch.from_numpy(np.stack(crops, axis=0)).to(device)
            with torch.no_grad():
                teacher(batch_t)
                tf3 = teacher_features["layer3"]
                tf2 = teacher_features["layer2"]
                tf1 = teacher_features["layer1"]
                de3, de2, de1 = student(tf3)

            for j, row in enumerate(batch_rows):
                s1 = float((1 - F.cosine_similarity(de3[j:j+1], tf3[j:j+1], dim=1)).mean())
                s2 = float((1 - F.cosine_similarity(de2[j:j+1], tf2[j:j+1], dim=1)).mean())
                s3 = float((1 - F.cosine_similarity(de1[j:j+1], tf1[j:j+1], dim=1)).mean())
                score = (s1 + s2 + s3) / 3.0
                val_scores.append(score)
                val_by_bin[row.get("six_bin_label", "")].append(score)
                n_val_scored += 1

    print(f"  val scored={n_val_scored:,}  errors={len(val_errors)}")

    val_arr    = np.array(val_scores, dtype=float)
    global_p95 = float(np.percentile(val_arr, 95)) if len(val_arr) > 0 else float("nan")
    global_p99 = float(np.percentile(val_arr, 99)) if len(val_arr) > 0 else float("nan")

    bin_thresholds = {
        "global": {"p95": round(global_p95, 6), "p99": round(global_p99, 6)}
    }
    for lbl, scores in val_by_bin.items():
        arr = np.array(scores, dtype=float)
        bin_thresholds[f"bin_{lbl}"] = {
            "p95": round(float(np.percentile(arr, 95)), 6),
            "p99": round(float(np.percentile(arr, 99)), 6),
        }

    threshold_summary = {
        "threshold_created_from": "rd_d1s_normal_val_only",
        "model_tag":              "rd_d1s_true_rd4ad_resnet18_medi3ch_shard_v1",
        "global_p95":             round(global_p95, 6),
        "global_p99":             round(global_p99, 6),
        "val_scored":             n_val_scored,
        "val_patients":           len(val_safe_ids),
        "rd_b9_threshold_modified": False,
        "bin_thresholds":         bin_thresholds,
    }
    with open(OUTPUT_ROOT / "rd_d1s_normal_val_threshold_summary.json", "w", encoding="utf-8") as f:
        json.dump(threshold_summary, f, indent=2)
    print(f"  saved: rd_d1s_normal_val_threshold_summary.json")
    print(f"  global_p95={global_p95:.6f}  global_p99={global_p99:.6f}")

    # ── RD-C2 candidate scoring ───────────────────────────────────────────────
    print("\n[6/7] RD-C2 candidate scoring (113,447) - on-the-fly")
    student.eval()
    score_rows        = []
    score_error_rows  = []
    score_ct_cache    = CtMmapCache(max_size=12)
    score_mask_cache  = CtMmapCache(max_size=48)
    c2_groups         = collections.defaultdict(list)
    for r in c2_rows:
        c2_groups[r["safe_id"]].append(r)

    n_scored      = 0
    n_failed      = 0
    n_nan         = 0
    n_inf         = 0
    BATCH_SCORE   = 64
    t_score_start = time.perf_counter()
    patients_done = 0

    for safe_id, rows_for_pat in c2_groups.items():
        ct_path = CANDIDATE_CT_ROOT / safe_id / "ct_hu.npy"
        assert_path_safe(ct_path)
        if not ct_path.exists():
            msg = f"ct_not_found:{ct_path}"
            score_error_rows.append({
                "candidate_id": "",
                "safe_id": safe_id,
                "error": msg,
            })
            write_csv(
                OUTPUT_ROOT / "rd_d1s_errors.csv",
                ["safe_id", "candidate_id", "error"],
                error_rows + score_error_rows + val_errors,
            )
            print(f"[ABORT] candidate CT missing: {safe_id} {ct_path}", file=sys.stderr)
            sys.exit(1)

        ct_arr = score_ct_cache.get(ct_path)
        cand_mask_path = find_cand_mask(safe_id)
        cand_mask_arr  = score_mask_cache.get(cand_mask_path) if cand_mask_path else None

        for i in range(0, len(rows_for_pat), BATCH_SCORE):
            batch_rows = rows_for_pat[i:i + BATCH_SCORE]
            crops = []
            for row in batch_rows:
                try:
                    crop = build_lung3ch_crop(
                        ct_arr,
                        int(row["local_z"]),
                        int(row["crop_y0"]), int(row["crop_x0"]),
                        int(row["crop_y1"]), int(row["crop_x1"]),
                    )
                    crop = apply_roi_mask_3ch(
                        crop, cand_mask_arr,
                        int(row["local_z"]),
                        int(row["crop_y0"]), int(row["crop_x0"]),
                        int(row["crop_y1"]), int(row["crop_x1"]),
                    )
                    crops.append(crop)
                except Exception as e:
                    print(f"[ABORT] crop 실패: {safe_id} - {e}", file=sys.stderr)
                    sys.exit(1)

            batch_t = torch.from_numpy(np.stack(crops, axis=0)).to(device)
            with torch.no_grad():
                teacher(batch_t)
                tf3 = teacher_features["layer3"]
                tf2 = teacher_features["layer2"]
                tf1 = teacher_features["layer1"]
                de3, de2, de1 = student(tf3)

            for j, row in enumerate(batch_rows):
                s1 = float((1 - F.cosine_similarity(de3[j:j+1], tf3[j:j+1], dim=1)).mean())
                s2 = float((1 - F.cosine_similarity(de2[j:j+1], tf2[j:j+1], dim=1)).mean())
                s3 = float((1 - F.cosine_similarity(de1[j:j+1], tf1[j:j+1], dim=1)).mean())
                rd4ad_score = (s1 + s2 + s3) / 3.0

                if math.isnan(rd4ad_score):
                    n_nan += 1
                    print(f"[ABORT] score NaN: {safe_id}", file=sys.stderr)
                    sys.exit(1)
                if math.isinf(rd4ad_score):
                    n_inf += 1
                    print(f"[ABORT] score Inf: {safe_id}", file=sys.stderr)
                    sys.exit(1)

                bin_lbl = row.get("six_bin_label", "unknown")
                b_p95   = bin_thresholds.get(f"bin_{bin_lbl}", {}).get("p95", global_p95)
                b_p99   = bin_thresholds.get(f"bin_{bin_lbl}", {}).get("p99", global_p99)

                score_rows.append({
                    "candidate_id":               row.get("candidate_id", ""),
                    "patient_id":                 row.get("patient_id", ""),
                    "safe_id":                    safe_id,
                    "stage_split":                row.get("stage_split", ""),
                    "local_z":                    row.get("local_z", ""),
                    "crop_y0":                    row.get("crop_y0", ""),
                    "crop_x0":                    row.get("crop_x0", ""),
                    "crop_y1":                    row.get("crop_y1", ""),
                    "crop_x1":                    row.get("crop_x1", ""),
                    "z_level":                    row.get("z_level", ""),
                    "boundary_status":            row.get("boundary_status", ""),
                    "six_bin_label":              bin_lbl,
                    "label":                      row.get("label", ""),
                    "score_layer1":               round(s3, 6),
                    "score_layer2":               round(s2, 6),
                    "score_layer3":               round(s1, 6),
                    "rd_d1s_medi3ch_rd4ad_score": round(rd4ad_score, 6),
                    "global_p95_exceed":          int(rd4ad_score > global_p95),
                    "global_p99_exceed":          int(rd4ad_score > global_p99),
                    "bin_p95_exceed":             int(rd4ad_score > b_p95),
                    "bin_p99_exceed":             int(rd4ad_score > b_p99),
                    "score_nan":                  0,
                    "score_inf":                  0,
                })
                n_scored += 1

        patients_done += 1
        if patients_done % 20 == 0 or patients_done == len(c2_groups):
            elapsed = time.perf_counter() - t_score_start
            print(f"  [{patients_done}/{len(c2_groups)}] scored={n_scored:,} "
                  f"failed={n_failed} elapsed={elapsed:.0f}s")

    print(f"\n  scoring complete: scored={n_scored:,}  failed={n_failed}  "
          f"NaN={n_nan}  Inf={n_inf}")

    score_fieldnames = [
        "candidate_id", "patient_id", "safe_id", "stage_split",
        "local_z", "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "z_level", "boundary_status", "six_bin_label", "label",
        "score_layer1", "score_layer2", "score_layer3",
        "rd_d1s_medi3ch_rd4ad_score",
        "global_p95_exceed", "global_p99_exceed",
        "bin_p95_exceed", "bin_p99_exceed",
        "score_nan", "score_inf",
    ]
    write_csv(
        OUTPUT_ROOT / "rd_d1s_stage1dev_candidate_score.csv",
        score_fieldnames, score_rows,
    )

    # ── AUROC/AUPRC + safety analysis ────────────────────────────────────────
    print("\n[7/7] AUROC/AUPRC + safety analysis")

    y_true  = np.array(
        [1 if r["label"] == "positive" else 0 for r in score_rows], dtype=int
    )
    y_score = np.array(
        [float(r["rd_d1s_medi3ch_rd4ad_score"]) for r in score_rows], dtype=float
    )

    rd_d1s_auroc = compute_auroc_mann_whitney(y_true, y_score)
    rd_d1s_auprc = compute_average_precision(y_true, y_score)
    auroc_vs_b8f = (rd_d1s_auroc - RD_B8F_AUROC_REF) if rd_d1s_auroc is not None else None
    auroc_vs_c3  = (rd_d1s_auroc - RD_C3_AUROC_REF)  if rd_d1s_auroc is not None else None

    print(f"  RD-D1s AUROC : {fmt_float(rd_d1s_auroc, 4)}")
    print(f"  RD-D1s AUPRC : {fmt_float(rd_d1s_auprc, 4)}")
    print(f"  RD-B8f ref   : {RD_B8F_AUROC_REF:.4f}  (vs D1s: {fmt_float(auroc_vs_b8f, 4)})")
    print(f"  RD-C3 ref    : {RD_C3_AUROC_REF:.4f}  (vs D1s: {fmt_float(auroc_vs_c3, 4)})")

    write_csv(
        OUTPUT_ROOT / "rd_d1s_auc_auprc_summary.csv",
        ["model", "auroc", "auprc", "auroc_vs_rdb8f", "auroc_vs_rdc3"],
        [
            {"model": "RD-B8f (reference)",        "auroc": RD_B8F_AUROC_REF, "auprc": "N/A",
             "auroc_vs_rdb8f": 0.0, "auroc_vs_rdc3": round(RD_B8F_AUROC_REF - RD_C3_AUROC_REF, 4)},
            {"model": "RD-C3 ConvAE (reference)",  "auroc": RD_C3_AUROC_REF,  "auprc": "N/A",
             "auroc_vs_rdb8f": round(RD_C3_AUROC_REF - RD_B8F_AUROC_REF, 4), "auroc_vs_rdc3": 0.0},
            {"model": "RD-D1s medi3ch true RD4AD", "auroc": rd_d1s_auroc,     "auprc": rd_d1s_auprc,
             "auroc_vs_rdb8f": auroc_vs_b8f, "auroc_vs_rdc3": auroc_vs_c3},
        ],
    )

    n_pos = int((y_true == 1).sum())
    n_hn  = int((y_true == 0).sum())

    # fixed threshold safety
    safety_rows = []
    for rule_name, thr in [("global_p95", global_p95), ("global_p99", global_p99)]:
        pos_mask = (y_true == 1)
        hn_mask  = (y_true == 0)
        sup_pos  = int((y_score[pos_mask] <= thr).sum())
        sup_hn   = int((y_score[hn_mask]  <= thr).sum())
        les_rate = sup_pos / n_pos if n_pos > 0 else 0.0
        hn_rate  = sup_hn  / n_hn  if n_hn  > 0 else 0.0
        by_pat   = collections.defaultdict(list)
        for r in score_rows:
            if r["label"] == "positive":
                by_pat[r["patient_id"]].append(
                    float(r["rd_d1s_medi3ch_rd4ad_score"]) <= thr
                )
        pat_all_sup = sum(1 for v in by_pat.values() if all(v))
        safety_rows.append({
            "threshold_rule":               rule_name,
            "threshold_value":              round(thr, 6),
            "lesion_suppressed_count":      sup_pos,
            "lesion_suppressed_rate":       round(les_rate, 4),
            "hn_suppressed_count":          sup_hn,
            "hn_suppressed_rate":           round(hn_rate, 4),
            "lesion_patient_all_suppressed": pat_all_sup,
        })

    write_csv(
        OUTPUT_ROOT / "rd_d1s_threshold_rule_safety_summary.csv",
        ["threshold_rule", "threshold_value", "lesion_suppressed_count",
         "lesion_suppressed_rate", "hn_suppressed_count", "hn_suppressed_rate",
         "lesion_patient_all_suppressed"],
        safety_rows,
    )

    # safety-constrained sweep
    pos_mask_np = (y_true == 1)
    hn_mask_np  = (y_true == 0)
    sweep_rows  = []
    for target_rate in [0.01, 0.03, 0.05]:
        sorted_pos = np.sort(y_score[pos_mask_np])
        thr_idx    = min(int(np.floor(target_rate * n_pos)), len(sorted_pos) - 1)
        thr        = float(sorted_pos[thr_idx])
        sup_pos    = int((y_score[pos_mask_np] <= thr).sum())
        sup_hn     = int((y_score[hn_mask_np]  <= thr).sum())
        les_rate   = sup_pos / n_pos if n_pos > 0 else 0.0
        hn_rate    = sup_hn  / n_hn  if n_hn  > 0 else 0.0
        by_pat     = collections.defaultdict(list)
        for r in score_rows:
            if r["label"] == "positive":
                by_pat[r["patient_id"]].append(
                    float(r["rd_d1s_medi3ch_rd4ad_score"]) <= thr
                )
        pat_all_sup = sum(1 for v in by_pat.values() if all(v))
        sweep_rows.append({
            "target_lesion_rate":           target_rate,
            "threshold":                    round(thr, 6),
            "lesion_suppressed_rate":       round(les_rate, 4),
            "hn_suppressed_rate":           round(hn_rate, 4),
            "lesion_patient_all_suppressed": pat_all_sup,
        })
        print(f"  @le{int(target_rate*100)}%: thr={thr:.6f}  "
              f"hn_sup={hn_rate:.2%}  pat_all_sup={pat_all_sup}")

    write_csv(
        OUTPUT_ROOT / "rd_d1s_safety_constrained_threshold_sweep.csv",
        ["target_lesion_rate", "threshold", "lesion_suppressed_rate",
         "hn_suppressed_rate", "lesion_patient_all_suppressed"],
        sweep_rows,
    )

    # patient-level summary
    by_pat_all = collections.defaultdict(lambda: {"label": "", "scores": []})
    for r in score_rows:
        by_pat_all[r["patient_id"]]["label"] = r["label"]
        by_pat_all[r["patient_id"]]["scores"].append(
            float(r["rd_d1s_medi3ch_rd4ad_score"])
        )
    write_csv(
        OUTPUT_ROOT / "rd_d1s_patient_level_safety_summary.csv",
        ["patient_id", "label", "n_crops", "score_mean", "score_max",
         "g95_exceed_count", "g99_exceed_count"],
        [
            {
                "patient_id":       pid,
                "label":            v["label"],
                "n_crops":          len(v["scores"]),
                "score_mean":       round(float(np.mean(v["scores"])), 6),
                "score_max":        round(float(np.max(v["scores"])), 6),
                "g95_exceed_count": sum(1 for s in v["scores"] if s > global_p95),
                "g99_exceed_count": sum(1 for s in v["scores"] if s > global_p99),
            }
            for pid, v in by_pat_all.items()
        ],
    )

    # 3-way comparison
    write_csv(
        OUTPUT_ROOT / "rd_d1s_compare_rdb8f_rdc3_rdd1s.csv",
        ["experiment", "model_type", "input_mode", "window", "auroc", "auprc",
         "auroc_vs_rdb8f", "decision"],
        [
            {"experiment": "RD-B8f", "model_type": "true_RD4AD_ResNet18",
             "input_mode": "3ch_mixed_MIP", "window": "HU[-1000,600]",
             "auroc": RD_B8F_AUROC_REF, "auprc": "N/A",
             "auroc_vs_rdb8f": 0.0, "decision": "NOT_USEFUL"},
            {"experiment": "RD-C3", "model_type": "ConvAutoencoder2p5D",
             "input_mode": "6ch_mediastinal", "window": "HU[-160,240]",
             "auroc": RD_C3_AUROC_REF, "auprc": "N/A",
             "auroc_vs_rdb8f": round(RD_C3_AUROC_REF - RD_B8F_AUROC_REF, 4),
             "decision": "CONVAE_USEFUL_FOR_RANKING"},
            {"experiment": "RD-D1s", "model_type": "true_RD4AD_ResNet18_shard",
             "input_mode": "3ch_mediastinal_zminus1_z_zplus1", "window": "HU[-160,240]",
             "auroc": rd_d1s_auroc, "auprc": rd_d1s_auprc,
             "auroc_vs_rdb8f": auroc_vs_b8f, "decision": "TBD"},
        ],
    )

    # ── final decision ───────────────────────────────────────────────────────
    le1_pat_sup = sweep_rows[0]["lesion_patient_all_suppressed"] if sweep_rows else 999
    g95_row     = next((r for r in safety_rows if r["threshold_rule"] == "global_p95"), {})
    g99_row     = next((r for r in safety_rows if r["threshold_rule"] == "global_p99"), {})

    if rd_d1s_auroc is None:
        final_decision = "BLOCKED"
    elif rd_d1s_auroc >= 0.60 and auroc_vs_b8f >= 0.10 and le1_pat_sup == 0:
        final_decision = "RD4AD_REVIVED_FOR_RANKING"
    elif rd_d1s_auroc >= 0.55 and auroc_vs_b8f > 0:
        final_decision = "RD4AD_ANALYSIS_ONLY"
    else:
        final_decision = "RD4AD_NOT_USEFUL"

    print(f"\n  final_decision : {final_decision}")
    print(f"  AUROC vs RD-B8f: {fmt_float(auroc_vs_b8f, 4)}")
    print(f"  AUROC vs RD-C3 : {fmt_float(auroc_vs_c3, 4)}")

    # ── all_checks_passed ────────────────────────────────────────────────────
    best_ckpt_saved = (CKPT_DIR / "best_train_loss.pth").exists()
    last_ckpt_saved = (CKPT_DIR / "last.pth").exists()
    all_checks_passed = (
        len(epoch_logs) == EPOCHS
        and loss_decreased
        and total_nan == 0
        and total_inf == 0
        and not teacher_param_changed
        and student_param_changed
        and opt_teacher_count == 0
        and best_ckpt_saved
        and last_ckpt_saved
        and n_failed == 0
        and n_nan == 0
        and n_inf == 0
        and n_scored == len(c2_rows)
        and holdout_cnt == 0
        and n_val_scored == len(val_rows)
        and len(val_errors) == 0
        and sampler.steps_per_epoch == STEPS_PER_EPOCH
        and sampler.steps_per_epoch * EPOCHS == TOTAL_STEPS_EXPECTED
    )

    t_elapsed = time.perf_counter() - t_start

    # ── summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "input_mode":                          "mediastinal_3ch_zminus1_z_zplus1",
        "train_source":                        "prebuilt_float32_shards",
        "model_type":                          "true_RD4AD_ResNet18_teacher_student",
        "window":                              f"HU[{MEDI_HU_MIN},{MEDI_HU_MAX}]",
        "shard_rows":                          SHARD_ROWS_EXPECTED,
        "shard_count":                         SHARD_COUNT_EXPECTED,
        "shard_size_gb":                       shard_gb_val,
        "train_epochs":                        EPOCHS,
        "steps_per_epoch":                     sampler.steps_per_epoch,
        "total_steps":                         sampler.steps_per_epoch * EPOCHS,
        "train_runtime_seconds":               round(train_time, 1),
        "total_runtime_seconds":               round(t_elapsed, 1),
        "gpu_peak_memory_mb":                  round(gpu_peak_mb, 1),
        "train_loss_first":                    round(train_loss_first, 6),
        "train_loss_last":                     round(train_loss_last, 6),
        "loss_decreased":                      loss_decreased,
        "best_epoch":                          best_epoch,
        "teacher_param_changed":               teacher_param_changed,
        "student_param_changed":               student_param_changed,
        "optimizer_teacher_param_count":       opt_teacher_count,
        "normal_val_rows":                     len(val_rows),
        "scored_candidates":                   n_scored,
        "positive_count":                      pos_cnt,
        "hard_negative_count":                 hn_cnt,
        "score_nan_count":                     n_nan,
        "score_inf_count":                     n_inf,
        "rd_b8f_auroc_reference":              RD_B8F_AUROC_REF,
        "rd_c3_convae_auroc_reference":        RD_C3_AUROC_REF,
        "rd_d1s_auroc":                        rd_d1s_auroc,
        "rd_d1s_auprc":                        rd_d1s_auprc,
        "auroc_improvement_vs_rd_b8f":         auroc_vs_b8f,
        "auroc_delta_vs_rd_c3":                auroc_vs_c3,
        "g95_lesion_suppressed_rate":          g95_row.get("lesion_suppressed_rate"),
        "g95_hard_negative_suppressed_rate":   g95_row.get("hn_suppressed_rate"),
        "lesion_patient_all_suppressed_g95":   g95_row.get("lesion_patient_all_suppressed"),
        "g99_lesion_suppressed_rate":          g99_row.get("lesion_suppressed_rate"),
        "g99_hard_negative_suppressed_rate":   g99_row.get("hn_suppressed_rate"),
        "best_hn_suppression_at_lesion_le1pct": sweep_rows[0]["hn_suppressed_rate"] if len(sweep_rows) > 0 else None,
        "best_hn_suppression_at_lesion_le3pct": sweep_rows[1]["hn_suppressed_rate"] if len(sweep_rows) > 1 else None,
        "best_hn_suppression_at_lesion_le5pct": sweep_rows[2]["hn_suppressed_rate"] if len(sweep_rows) > 2 else None,
        "final_decision":                      final_decision,
        "suppression_applied":                 False,
        "first_stage_score_modified":          False,
        "stage2_holdout_access":               0,
        "sklearn_used":                        False,
        "metric_backend":                      "sklearn_free_mann_whitney_auroc_and_step_average_precision",
        "normal_val_scored":                   n_val_scored,
        "normal_val_expected":                 len(val_rows),
        "normal_val_error_count":              len(val_errors),
        "done_created":                        all_checks_passed,
        "failed_marker_created":               not all_checks_passed,
        "all_checks_passed":                   all_checks_passed,
    }
    with open(OUTPUT_ROOT / "rd_d1s_medi3ch_true_rd4ad_shard_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  saved: rd_d1s_medi3ch_true_rd4ad_shard_run_summary.json")

    # errors CSV
    write_csv(
        OUTPUT_ROOT / "rd_d1s_errors.csv",
        ["safe_id", "candidate_id", "error"],
        error_rows + score_error_rows + val_errors,
    )

    # report MD
    verdict_str = "PASS" if all_checks_passed else "FAIL"
    md_lines = [
        "# RD-D1s: Mediastinal 3ch true RD4AD Shard-based Full Run Report",
        "",
        f"## 판정: {verdict_str} / {final_decision}",
        "",
        "## 핵심 결과",
        "| 항목 | 값 |",
        "|---|---|",
        f"| input_mode | mediastinal_3ch_zminus1_z_zplus1 |",
        f"| train_source | prebuilt_float32_shards |",
        f"| shard_rows | {SHARD_ROWS_EXPECTED:,} |",
        f"| shard_count | {SHARD_COUNT_EXPECTED} |",
        f"| window | HU[{MEDI_HU_MIN},{MEDI_HU_MAX}] → [0,1] |",
        f"| train_epochs | {EPOCHS} |",
        f"| best_epoch | {best_epoch} |",
        f"| train_loss_first | {fmt_float(train_loss_first, 6)} |",
        f"| train_loss_last | {fmt_float(train_loss_last, 6)} |",
        f"| loss_decreased | {loss_decreased} |",
        f"| train_runtime_sec | {train_time:.1f} |",
        f"| total_runtime_sec | {t_elapsed:.1f} |",
        f"| gpu_peak_memory_mb | {gpu_peak_mb:.1f} |",
        f"| scored_candidates | {n_scored:,} |",
        f"| score_nan | {n_nan} |",
        f"| score_inf | {n_inf} |",
        "",
        "## AUROC/AUPRC 비교",
        "| 모델 | AUROC | AUPRC | vs RD-B8f |",
        "|---|---|---|---|",
        f"| RD-B8f (reference) | {RD_B8F_AUROC_REF:.4f} | N/A | 0.0000 |",
        f"| RD-C3 ConvAE (reference) | {RD_C3_AUROC_REF:.4f} | N/A | +{RD_C3_AUROC_REF-RD_B8F_AUROC_REF:.4f} |",
        f"| RD-D1s medi3ch true RD4AD | {fmt_float(rd_d1s_auroc,4)} | {fmt_float(rd_d1s_auprc,4)} | {fmt_float(auroc_vs_b8f,4)} |",
        "",
        "## Safety (G95 threshold)",
        "| 항목 | 값 |",
        "|---|---|",
        f"| threshold | {fmt_float(global_p95, 6)} |",
        f"| lesion_suppressed_rate | {g95_row.get('lesion_suppressed_rate', 'N/A')} |",
        f"| hn_suppressed_rate | {g95_row.get('hn_suppressed_rate', 'N/A')} |",
        f"| lesion_patient_all_suppressed | {g95_row.get('lesion_patient_all_suppressed', 'N/A')} |",
        "",
        "## Safety-constrained Sweep",
        "| target_lesion_rate | hn_sup_rate | pat_all_sup |",
        "|---|---|---|",
    ]
    for sw in sweep_rows:
        md_lines.append(
            f"| {sw['target_lesion_rate']:.0%} | {sw['hn_suppressed_rate']:.4f} "
            f"| {sw['lesion_patient_all_suppressed']} |"
        )
    md_lines += [
        "",
        f"## all_checks_passed: {all_checks_passed}",
    ]
    with open(OUTPUT_ROOT / "rd_d1s_medi3ch_true_rd4ad_shard_run_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("  saved: rd_d1s_medi3ch_true_rd4ad_shard_run_report.md")

    # DONE marker
    if all_checks_passed:
        (OUTPUT_ROOT / "DONE").write_text("DONE\n", encoding="utf-8")
        print("\n[DONE] all_checks_passed:", all_checks_passed)
    else:
        (OUTPUT_ROOT / "FAILED").write_text(
            f"FAILED\nall_checks_passed={all_checks_passed}\nfinal_decision={final_decision}\n",
            encoding="utf-8",
        )
        print("\n[FAILED] all_checks_passed:", all_checks_passed, file=sys.stderr)
        sys.exit(1)


# =============================================================================
# entry point
# =============================================================================

if IS_DRY_PLAN:
    run_dry_plan()
elif IS_RUN_ALL_SHARD:
    run_all_shard()
