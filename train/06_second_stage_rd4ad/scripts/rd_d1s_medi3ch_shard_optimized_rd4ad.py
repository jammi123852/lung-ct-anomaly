"""
RD-D1s: Mediastinal 3ch shard-optimized true RD4AD
목적: RD-D1 on-the-fly train (264.3min) → float32 shard 기반으로 가속.
     모델 구조, 입력 정의, threshold/scoring 방식은 RD-D1과 동일.

모드:
  bare run                -> exit 2
  --dry-plan-shard        -> 입력 확인, shard 크기 추정, 디스크 확인
  --smoke-shard-profile   -> 2000 rows smoke shard build + 100 batch loader profile
  --build-full-shards     -> 86,017 rows 전체 float32 shard 생성
  --profile-shard-train   -> shard 기반 200 timed batch train profile

안전 조건:
  stage2_holdout 접근 금지
  기존 RD-D1 profile v1/v2 삭제 금지
  기존 RD-B/RD-C 결과 수정 금지
  기존 normal_train manifest 수정 금지
  threshold/scoring rule 변경 금지 (RD-D1과 동일)
  모델 구조 변경 금지 (RD-D1과 동일)
  suppression 적용 금지
  SHARD_ROOT 존재 시 --build-full-shards ABORT
  REPORT_ROOT 존재 시 --dry-plan-shard ABORT
  crop 실패 시 zero 대체 금지, ABORT
  OOB → reflect padding (edge fallback if valid_h or valid_w <= 1)
  edge fallback > 10 시 build ABORT
"""

import sys
import csv
import json
import math
import time
import collections
from pathlib import Path

ALLOWED_MODES = {
    "--dry-plan-shard", "--smoke-shard-profile",
    "--build-full-shards", "--profile-shard-train",
}
if not any(m in sys.argv for m in ALLOWED_MODES):
    print("오류: 실행 모드 지정 필요.")
    print("  --dry-plan-shard      : 입력 확인, shard 크기 추정")
    print("  --smoke-shard-profile : 2000 rows smoke build + loader profile")
    print("  --build-full-shards   : 86,017 rows 전체 shard 생성")
    print("  --profile-shard-train : shard train 200 timed batches profile")
    sys.exit(2)

IS_DRY_PLAN_SHARD      = "--dry-plan-shard"      in sys.argv
IS_SMOKE_SHARD         = "--smoke-shard-profile"  in sys.argv
IS_BUILD_FULL_SHARDS   = "--build-full-shards"    in sys.argv
IS_PROFILE_SHARD_TRAIN = "--profile-shard-train"  in sys.argv

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

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

NORMAL_TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b1_6bin_balanced_manifest_preflight_v1"
    / "rd_b1_6bin_balanced_normal_train_coordinate_manifest.csv"
)
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

LOCAL_WEIGHT_PATH = Path(
    "/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)

SHARD_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1_medi3ch_train_shards_v1"
)
MODEL_ROOT_SHARD = (
    PROJECT_ROOT / "outputs/models/rd_d1_true_rd4ad_resnet18_medi3ch_shard_v1"
)
REPORT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1_medi3ch_true_rd4ad_revival_shard_v1"
)

# ── 설계 상수 (RD-D1과 동일) ──────────────────────────────────────────────────
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
MEDI_HU_MIN  = -160.0
MEDI_HU_MAX  =  240.0

# shard 설정
SHARD_SIZE              = 2000   # rows per shard file
SMOKE_ROWS              = 2000
EDGE_FALLBACK_ABORT_THR = 10     # 이 수를 초과하면 build ABORT

# 예상 전체 학습 스텝 (RD-D1과 동일: 1741 steps/epoch × 20 epoch)
TOTAL_TRAIN_STEPS = 34820

# on-the-fly baseline (RD-D1 profile-time v2 결과)
ONTHEFLY_MEAN_SEC  = 0.4554
ONTHEFLY_TRAIN_MIN = 264.3

# reflect-padding 전역 카운터 (single-thread 전용)
_G_PAD_APPLIED_COUNT: int = 0
_G_PAD_REFLECT_COUNT: int = 0
_G_PAD_EDGE_COUNT:    int = 0


# ── safety ────────────────────────────────────────────────────────────────────

def assert_path_safe(p):
    for kw in FORBIDDEN_KEYWORDS:
        if kw.lower() in str(p).lower():
            raise RuntimeError(f"[ABORT] 금지 경로 접근: {p}  (keyword={kw})")


# ── model (RD-D1과 동일) ──────────────────────────────────────────────────────

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


# ── CT mmap cache ─────────────────────────────────────────────────────────────

class CtMmapCache:
    def __init__(self, max_size=48):
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


# ── shard mmap cache ──────────────────────────────────────────────────────────

class ShardMmapCache:
    """pre-built shard npy 파일 mmap 캐시"""
    def __init__(self, max_size=16):
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


# ── reflect-padding crop builder (RD-D1 v2와 동일) ───────────────────────────

def build_medi3ch_crop(ct_arr, local_z, y0, x0, y1, x1):
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
            f"[ABORT] crop contains NaN/Inf  y0={y0} x0={x0} y1={y1} x1={x1}"
        )

    if needs_pad:
        _G_PAD_APPLIED_COUNT += 1
        if can_reflect:
            _G_PAD_REFLECT_COUNT += 1
        else:
            _G_PAD_EDGE_COUNT += 1

    return crop.astype(np.float32)


# ── shard 기반 6-bin sampler ──────────────────────────────────────────────────

class ShardedSixBinSampler:
    """shard index CSV 기반 6-bin 균형 sampler"""

    def __init__(self, index_csv, per_bin, seed):
        self.per_bin = per_bin
        self._seed   = seed
        self.bin_items = {lbl: [] for lbl in SIX_BIN_LABELS}

        with open(index_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lbl = row.get("six_bin_label", "")
                if lbl not in self.bin_items:
                    continue
                self.bin_items[lbl].append({
                    "shard_path": row["shard_path"],
                    "offset":     int(row["offset"]),
                })

        bin_sizes = {lbl: len(v) for lbl, v in self.bin_items.items()}
        self.total_rows      = sum(bin_sizes.values())
        self.min_bin_size    = min(bin_sizes.values())
        self.steps_per_epoch = self.min_bin_size // per_bin

        print("  6-bin 분포 (shard index):")
        for lbl in SIX_BIN_LABELS:
            print(f"    {lbl}: {bin_sizes[lbl]:,}")
        print(f"  min_bin={self.min_bin_size:,}  per_bin={per_bin}  "
              f"steps/epoch={self.steps_per_epoch:,}")

    def epoch_batches(self, epoch):
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


# ── utility ───────────────────────────────────────────────────────────────────

def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  saved: {path.name}")


# ── 1단계: --dry-plan-shard ───────────────────────────────────────────────────

def run_dry_plan_shard():
    import shutil

    print("=" * 72)
    print("RD-D1s: Mediastinal shard-optimized RD4AD [DRY-PLAN-SHARD]")
    print("=" * 72)

    ok_all = True

    # [1] 입력 파일 확인
    print("\n[1] 입력 파일 확인")
    checks = [
        ("normal_train manifest", NORMAL_TRAIN_MANIFEST),
        ("ResNet18 local weights", LOCAL_WEIGHT_PATH),
        ("RD-C2 candidate manifest", RD_C2_MANIFEST),
    ]
    for label, p in checks:
        exists = p.exists()
        print(f"  {'OK' if exists else 'MISSING'}: {label} -> {p.name}")
        if not exists:
            ok_all = False

    # [2] output root 충돌 확인
    print("\n[2] output root 충돌 확인")
    for label, p in [
        ("REPORT_ROOT",    REPORT_ROOT),
        ("SHARD_ROOT",     SHARD_ROOT),
        ("MODEL_ROOT",     MODEL_ROOT_SHARD),
    ]:
        exists = p.exists()
        print(f"  {'CONFLICT' if exists else 'OK'}: {label} -> {p}")
        if exists:
            ok_all = False

    # [3] stage2_holdout 확인
    print("\n[3] stage2_holdout intersection 확인")
    stage2_holdout_cnt = 0
    if RD_C2_MANIFEST.exists():
        with open(RD_C2_MANIFEST, newline="", encoding="utf-8") as f:
            c2_rows = list(csv.DictReader(f))
        stage2_holdout_cnt = sum(
            1 for r in c2_rows if r.get("stage_split", "") == "stage2_holdout"
        )
        status = "OK" if stage2_holdout_cnt == 0 else "FAIL"
        print(f"  stage2_holdout intersection: {stage2_holdout_cnt} ({status})")
        if stage2_holdout_cnt != 0:
            ok_all = False

    # [4] manifest 분석
    print("\n[4] normal_train manifest 분석")
    train_rows = []
    bin_dist = {lbl: 0 for lbl in SIX_BIN_LABELS}
    oob_y_x_count = 0
    if NORMAL_TRAIN_MANIFEST.exists():
        with open(NORMAL_TRAIN_MANIFEST, newline="", encoding="utf-8") as f:
            train_rows = list(csv.DictReader(f))
        for r in train_rows:
            lbl = r.get("six_bin_label", "")
            if lbl in bin_dist:
                bin_dist[lbl] += 1
            if int(r.get("crop_y0", 0)) < 0 or int(r.get("crop_x0", 0)) < 0:
                oob_y_x_count += 1

    total_rows = len(train_rows)
    z_warn = sum(int(r.get("low_z_warning", 0) or 0) for r in train_rows)
    print(f"  total rows: {total_rows:,}")
    print(f"  six_bin 분포:")
    for lbl in SIX_BIN_LABELS:
        print(f"    {lbl}: {bin_dist[lbl]:,}")
    print(f"  OOB (y0<0 or x0<0): {oob_y_x_count:,}")
    print(f"  z OOB (low_z_warning): {z_warn:,}")

    # [5] shard 크기 추정
    print("\n[5] shard 크기 추정")
    bytes_per_crop  = 3 * CROP_SIZE * CROP_SIZE * 4   # float32
    total_bytes     = total_rows * bytes_per_crop
    n_shards        = math.ceil(total_rows / SHARD_SIZE)
    bytes_per_shard = SHARD_SIZE * bytes_per_crop
    total_gb        = total_bytes / (1024 ** 3)
    print(f"  bytes_per_crop    : {bytes_per_crop:,} B")
    print(f"  bytes_per_shard   : {bytes_per_shard/1024/1024:.1f} MB  (SHARD_SIZE={SHARD_SIZE})")
    print(f"  n_shards          : {n_shards}")
    print(f"  total_shard_size  : {total_gb:.2f} GB")

    # [6] 디스크 여유 확인
    print("\n[6] 디스크 여유 공간 확인 (PROJECT_ROOT 파티션)")
    stat     = shutil.disk_usage(str(PROJECT_ROOT))
    free_gb  = stat.free  / (1024 ** 3)
    total_d  = stat.total / (1024 ** 3)
    margin   = free_gb - total_gb
    print(f"  disk total : {total_d:.1f} GB")
    print(f"  disk free  : {free_gb:.1f} GB")
    print(f"  shard need : {total_gb:.2f} GB")
    print(f"  margin     : {margin:.1f} GB  {'OK' if margin > 2 else 'WARN: 여유 부족'}")
    if margin < 1:
        print("  [FAIL] 디스크 여유 부족")
        ok_all = False

    verdict = "DRY-PLAN-SHARD OK" if ok_all else "DRY-PLAN-SHARD FAIL"

    # [7] REPORT_ROOT 생성 및 결과 저장
    print(f"\n[7] REPORT_ROOT 생성")
    if REPORT_ROOT.exists():
        print(f"[ABORT] REPORT_ROOT 이미 존재: {REPORT_ROOT}", file=sys.stderr)
        sys.exit(1)
    REPORT_ROOT.mkdir(parents=True)
    print(f"  created: {REPORT_ROOT}")

    summary = {
        "total_train_rows":      total_rows,
        "six_bin_distribution":  bin_dist,
        "oob_y_x_count":         oob_y_x_count,
        "z_oob_low_z_warning":   z_warn,
        "n_shards":              n_shards,
        "shard_size_rows":       SHARD_SIZE,
        "bytes_per_crop":        bytes_per_crop,
        "bytes_per_shard_mb":    round(bytes_per_shard / 1024 / 1024, 1),
        "total_shard_size_gb":   round(total_gb, 3),
        "disk_free_gb":          round(free_gb, 2),
        "disk_margin_gb":        round(margin, 2),
        "stage2_holdout_access": stage2_holdout_cnt,
        "ok_all":                ok_all,
        "verdict":               verdict,
    }
    with open(REPORT_ROOT / "rd_d1s_shard_dryplan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("  saved: rd_d1s_shard_dryplan_summary.json")

    print(f"\n판정: {verdict}")
    if not ok_all:
        sys.exit(1)


# ── 2단계: --smoke-shard-profile ─────────────────────────────────────────────

def run_smoke_shard_profile():
    import numpy as np
    import torch
    import statistics as stats_mod
    import random as rnd_mod

    print("=" * 72)
    print(f"RD-D1s: [SMOKE-SHARD-PROFILE]  rows={SMOKE_ROWS}")
    print("=" * 72)

    # 전제 확인
    if not REPORT_ROOT.exists():
        print(f"[ABORT] REPORT_ROOT 없음 (--dry-plan-shard 먼저 실행): {REPORT_ROOT}",
              file=sys.stderr)
        sys.exit(1)
    if not NORMAL_TRAIN_MANIFEST.exists():
        print(f"[ABORT] manifest 없음: {NORMAL_TRAIN_MANIFEST}", file=sys.stderr)
        sys.exit(1)

    SMOKE_SHARD_DIR = REPORT_ROOT / "smoke_shards"
    if SMOKE_SHARD_DIR.exists():
        print(f"[ABORT] smoke_shards 이미 존재: {SMOKE_SHARD_DIR}", file=sys.stderr)
        sys.exit(1)
    SMOKE_SHARD_DIR.mkdir(parents=True)

    # manifest에서 첫 SMOKE_ROWS 행 읽기
    with open(NORMAL_TRAIN_MANIFEST, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    smoke_manifest_rows = all_rows[:SMOKE_ROWS]
    print(f"  smoke rows: {len(smoke_manifest_rows)}/{len(all_rows)}")

    # ── [1] smoke shard 빌드 ────────────────────────────────────────────────
    print("\n[1] smoke shard 빌드")
    global _G_PAD_APPLIED_COUNT, _G_PAD_REFLECT_COUNT, _G_PAD_EDGE_COUNT

    t_build_start = time.perf_counter()
    ct_cache  = CtMmapCache(max_size=48)
    shard_arr = np.zeros((len(smoke_manifest_rows), 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    index_rows = []
    pad_applied = pad_reflect = pad_edge = 0

    for i, row in enumerate(smoke_manifest_rows):
        assert_path_safe(NORMAL_CT_ROOT / row["safe_id"])
        ct_path = NORMAL_CT_ROOT / row["safe_id"] / "ct_hu.npy"
        if not ct_path.exists():
            print(f"[ABORT] CT file 없음: {ct_path}", file=sys.stderr)
            sys.exit(1)
        ct_arr = ct_cache.get(ct_path)

        _ba = _G_PAD_APPLIED_COUNT
        _br = _G_PAD_REFLECT_COUNT
        _be = _G_PAD_EDGE_COUNT
        try:
            crop = build_medi3ch_crop(
                ct_arr,
                int(row["local_z"]),
                int(row["crop_y0"]), int(row["crop_x0"]),
                int(row["crop_y1"]), int(row["crop_x1"]),
            )
        except Exception as e:
            print(f"[ABORT] crop 실패: row={i}  {e}", file=sys.stderr)
            sys.exit(1)

        da = _G_PAD_APPLIED_COUNT - _ba
        dr = _G_PAD_REFLECT_COUNT - _br
        de = _G_PAD_EDGE_COUNT    - _be
        pad_applied += da
        pad_reflect += dr
        pad_edge    += de

        pad_mode_str = "none"
        if da:
            pad_mode_str = "reflect" if dr else "edge"

        shard_arr[i] = crop
        index_rows.append({
            "shard_id":        0,
            "row_index":       i,
            "safe_id":         row["safe_id"],
            "local_z":         row["local_z"],
            "crop_y0":         row["crop_y0"],
            "crop_x0":         row["crop_x0"],
            "crop_y1":         row["crop_y1"],
            "crop_x1":         row["crop_x1"],
            "six_bin_label":   row.get("six_bin_label", ""),
            "shard_path":      "smoke_shards/shard_0000.npy",
            "offset":          i,
            "padding_applied": da,
            "padding_mode":    pad_mode_str,
        })
        if (i + 1) % 500 == 0:
            print(f"    built {i+1}/{len(smoke_manifest_rows)}")

    t_build_end = time.perf_counter()
    build_sec = t_build_end - t_build_start

    if not np.isfinite(shard_arr).all():
        print("[ABORT] shard_arr에 NaN/Inf 포함", file=sys.stderr)
        sys.exit(1)

    shard_file = SMOKE_SHARD_DIR / "shard_0000.npy"
    np.save(str(shard_file), shard_arr)
    shard_size_mb = shard_file.stat().st_size / 1024 / 1024

    print(f"  build time       : {build_sec:.2f}s  ({build_sec/len(smoke_manifest_rows)*1000:.2f}ms/crop)")
    print(f"  shard saved      : {shard_file.name}  ({shard_size_mb:.1f} MB)")
    print(f"  NaN/Inf          : 0 (OK)")
    print(f"  pad_applied      : {pad_applied}")
    print(f"  pad_reflect      : {pad_reflect}")
    print(f"  pad_edge         : {pad_edge}")

    INDEX_FIELDS = [
        "shard_id", "row_index", "safe_id", "local_z",
        "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "six_bin_label", "shard_path", "offset",
        "padding_applied", "padding_mode",
    ]
    write_csv(REPORT_ROOT / "rd_d1s_smoke_shard_index.csv", INDEX_FIELDS, index_rows)

    # ── [2] on-the-fly vs shard 값 일치 확인 (100개 샘플) ──────────────────
    print("\n[2] on-the-fly vs shard 값 일치 확인 (100 샘플)")
    loaded_shard = np.load(str(shard_file), mmap_mode="r")
    rng_check    = rnd_mod.Random(7)
    check_indices = rng_check.sample(range(len(smoke_manifest_rows)),
                                     min(100, len(smoke_manifest_rows)))
    ct_cache2    = CtMmapCache(max_size=48)
    max_abs_diff = 0.0

    for idx in check_indices:
        row = smoke_manifest_rows[idx]
        ct_path = NORMAL_CT_ROOT / row["safe_id"] / "ct_hu.npy"
        ct_arr  = ct_cache2.get(ct_path)
        crop_live = build_medi3ch_crop(
            ct_arr,
            int(row["local_z"]),
            int(row["crop_y0"]), int(row["crop_x0"]),
            int(row["crop_y1"]), int(row["crop_x1"]),
        )
        crop_shard = np.array(loaded_shard[idx], dtype=np.float32)
        diff = float(np.abs(crop_live - crop_shard).max())
        if diff > max_abs_diff:
            max_abs_diff = diff

    match_ok = max_abs_diff < 1e-5
    print(f"  max_abs_diff     : {max_abs_diff:.2e}  ({'OK' if match_ok else 'WARN'})")
    if not match_ok:
        print("  [FAIL] shard와 on-the-fly 값 불일치", file=sys.stderr)
        sys.exit(1)

    # ── [3] shard loader 100 timed batches profile ──────────────────────────
    print("\n[3] shard loader 100 timed batches profile")
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

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    shard_cache  = ShardMmapCache(max_size=4)
    WARMUP = 5
    TIMED  = 100
    batch_times  = []
    crop_times   = []
    fb_times     = []
    loss_nan_cnt = 0
    loss_inf_cnt = 0
    batch_rows   = []
    rng_smoke    = rnd_mod.Random(42)

    for b_idx in range(WARMUP + TIMED):
        sample_items = rng_smoke.choices(index_rows, k=BATCH_SIZE)

        t_crop_s = time.perf_counter()
        crops = []
        for item in sample_items:
            sp  = REPORT_ROOT / item["shard_path"]
            off = int(item["offset"])
            arr = shard_cache.get(sp)
            crops.append(arr[off].copy())
        t_crop_e = time.perf_counter()
        crop_t = t_crop_e - t_crop_s

        batch_np = np.stack(crops, axis=0)
        batch_t  = torch.from_numpy(batch_np).to(device)

        t_fb_s = time.perf_counter()
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
        t_fb_e = time.perf_counter()
        fb_t  = t_fb_e - t_fb_s
        tot_t = crop_t + fb_t
        loss_val = float(loss.item())

        if b_idx >= WARMUP:
            batch_times.append(tot_t)
            crop_times.append(crop_t)
            fb_times.append(fb_t)
            if math.isnan(loss_val):
                loss_nan_cnt += 1
            if math.isinf(loss_val):
                loss_inf_cnt += 1
            batch_rows.append({
                "batch_idx": b_idx - WARMUP,
                "total_sec": round(tot_t, 5),
                "crop_sec":  round(crop_t, 5),
                "fb_sec":    round(fb_t, 5),
                "loss":      round(loss_val, 6),
            })

    gpu_peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda" else 0.0
    )
    del teacher, student, optimizer

    shard_mean   = float(stats_mod.mean(batch_times))   if batch_times else 0.0
    shard_median = float(stats_mod.median(batch_times)) if batch_times else 0.0
    crop_mean    = float(stats_mod.mean(crop_times))    if crop_times  else 0.0
    fb_mean      = float(stats_mod.mean(fb_times))      if fb_times    else 0.0
    speedup      = ONTHEFLY_MEAN_SEC / shard_mean       if shard_mean > 0 else 0.0
    est_train_s  = shard_mean * TOTAL_TRAIN_STEPS
    est_train_m  = est_train_s / 60

    print(f"\n  [smoke loader profile 결과]")
    print(f"  timed_batches              : {len(batch_times)}")
    print(f"  shard_batch_mean           : {shard_mean:.4f}s")
    print(f"  shard_batch_median         : {shard_median:.4f}s")
    print(f"  shard_crop_load_mean       : {crop_mean:.4f}s")
    print(f"  forward_backward_mean      : {fb_mean:.4f}s")
    print(f"  loss_nan_count             : {loss_nan_cnt}")
    print(f"  loss_inf_count             : {loss_inf_cnt}")
    print(f"  on-the-fly baseline        : {ONTHEFLY_MEAN_SEC:.4f}s  ({ONTHEFLY_TRAIN_MIN:.1f}min)")
    print(f"  speedup (approx)           : {speedup:.1f}x")
    print(f"  est train total (smoke)    : {est_train_m:.1f}min")
    print(f"  GPU peak memory            : {gpu_peak_mb:.1f} MB")

    # ── 파일 저장 ───────────────────────────────────────────────────────────
    write_csv(
        REPORT_ROOT / "rd_d1s_shard_smoke_profile.csv",
        ["batch_idx", "total_sec", "crop_sec", "fb_sec", "loss"],
        batch_rows,
    )
    smoke_summary = {
        "smoke_rows":                    len(smoke_manifest_rows),
        "build_time_sec":                round(build_sec, 3),
        "build_ms_per_crop":             round(build_sec / len(smoke_manifest_rows) * 1000, 3),
        "shard_size_mb":                 round(shard_size_mb, 1),
        "pad_applied":                   pad_applied,
        "pad_reflect":                   pad_reflect,
        "pad_edge":                      pad_edge,
        "max_abs_diff_onthefly_vs_shard": round(max_abs_diff, 12),
        "timed_batches":                 len(batch_times),
        "shard_batch_mean_sec":          round(shard_mean,   5),
        "shard_batch_median_sec":        round(shard_median, 5),
        "shard_crop_load_mean_sec":      round(crop_mean,    5),
        "forward_backward_mean_sec":     round(fb_mean,      5),
        "loss_nan_count":                loss_nan_cnt,
        "loss_inf_count":                loss_inf_cnt,
        "onthefly_baseline_mean_sec":    ONTHEFLY_MEAN_SEC,
        "speedup_approx":                round(speedup, 2),
        "est_train_total_min_smoke":     round(est_train_m, 1),
        "gpu_peak_memory_mb":            round(gpu_peak_mb, 1),
    }
    with open(REPORT_ROOT / "rd_d1s_shard_smoke_profile.json", "w", encoding="utf-8") as f:
        json.dump(smoke_summary, f, indent=2)
    print("  saved: rd_d1s_shard_smoke_profile.json")

    md_lines = [
        "# RD-D1s Smoke Shard Profile",
        "",
        "## Shard Build",
        "| 항목 | 값 |",
        "|---|---|",
        f"| smoke_rows | {len(smoke_manifest_rows)} |",
        f"| build_time_sec | {build_sec:.3f}s |",
        f"| ms_per_crop | {build_sec/len(smoke_manifest_rows)*1000:.2f}ms |",
        f"| shard_size_mb | {shard_size_mb:.1f} MB |",
        f"| pad_applied | {pad_applied} |",
        f"| pad_reflect | {pad_reflect} |",
        f"| pad_edge | {pad_edge} |",
        f"| max_abs_diff | {max_abs_diff:.2e} |",
        "",
        "## Smoke Loader Profile (100 timed batches)",
        "| 항목 | 값 |",
        "|---|---|",
        f"| shard_batch_mean | {shard_mean:.4f}s |",
        f"| on-the-fly baseline | {ONTHEFLY_MEAN_SEC:.4f}s ({ONTHEFLY_TRAIN_MIN:.1f}min) |",
        f"| speedup | {speedup:.1f}x |",
        f"| est_train_total (smoke) | {est_train_m:.1f}min |",
        f"| loss_nan | {loss_nan_cnt} |",
        f"| loss_inf | {loss_inf_cnt} |",
        f"| gpu_peak_mb | {gpu_peak_mb:.1f} |",
    ]
    with open(REPORT_ROOT / "rd_d1s_shard_smoke_profile.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("  saved: rd_d1s_shard_smoke_profile.md")

    verdict = "PASS" if (loss_nan_cnt == 0 and loss_inf_cnt == 0 and match_ok) else "FAIL"
    print(f"\n판정: smoke {verdict}")
    if verdict == "FAIL":
        sys.exit(1)


# ── 3단계: --build-full-shards ────────────────────────────────────────────────

def run_build_full_shards():
    import numpy as np

    print("=" * 72)
    print("RD-D1s: [BUILD-FULL-SHARDS]  total=86,017 rows")
    print("=" * 72)
    t_total_start = time.perf_counter()

    # 안전 확인
    if SHARD_ROOT.exists():
        print(f"[ABORT] SHARD_ROOT 이미 존재: {SHARD_ROOT}", file=sys.stderr)
        sys.exit(1)
    assert_path_safe(SHARD_ROOT)
    SHARD_ROOT.mkdir(parents=True)

    if not NORMAL_TRAIN_MANIFEST.exists():
        print(f"[ABORT] manifest 없음: {NORMAL_TRAIN_MANIFEST}", file=sys.stderr)
        sys.exit(1)
    with open(NORMAL_TRAIN_MANIFEST, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    total_rows = len(all_rows)
    n_shards   = math.ceil(total_rows / SHARD_SIZE)
    print(f"  total rows: {total_rows:,}  n_shards: {n_shards}")

    INDEX_FIELDS = [
        "shard_id", "row_index", "safe_id", "local_z",
        "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "six_bin_label", "shard_path", "offset",
        "padding_applied", "padding_mode",
    ]
    index_rows  = []
    error_rows  = []
    ct_cache    = CtMmapCache(max_size=48)
    total_pad_applied = total_pad_reflect = total_pad_edge = 0

    global _G_PAD_APPLIED_COUNT, _G_PAD_REFLECT_COUNT, _G_PAD_EDGE_COUNT

    for shard_id in range(n_shards):
        row_start  = shard_id * SHARD_SIZE
        row_end    = min(row_start + SHARD_SIZE, total_rows)
        s_rows     = all_rows[row_start:row_end]
        n_in_shard = len(s_rows)

        shard_data   = np.zeros((n_in_shard, 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        s_pad_a = s_pad_r = s_pad_e = 0

        for local_off, row in enumerate(s_rows):
            assert_path_safe(NORMAL_CT_ROOT / row["safe_id"])
            ct_path = NORMAL_CT_ROOT / row["safe_id"] / "ct_hu.npy"
            if not ct_path.exists():
                print(f"[ABORT] CT 없음: {ct_path}", file=sys.stderr)
                sys.exit(1)
            ct_arr = ct_cache.get(ct_path)

            _ba = _G_PAD_APPLIED_COUNT
            _br = _G_PAD_REFLECT_COUNT
            _be = _G_PAD_EDGE_COUNT
            try:
                crop = build_medi3ch_crop(
                    ct_arr,
                    int(row["local_z"]),
                    int(row["crop_y0"]), int(row["crop_x0"]),
                    int(row["crop_y1"]), int(row["crop_x1"]),
                )
            except Exception as e:
                print(f"[ABORT] crop 실패: shard={shard_id} offset={local_off}  {e}",
                      file=sys.stderr)
                sys.exit(1)

            da = _G_PAD_APPLIED_COUNT - _ba
            dr = _G_PAD_REFLECT_COUNT - _br
            de = _G_PAD_EDGE_COUNT    - _be
            s_pad_a += da
            s_pad_r += dr
            s_pad_e += de

            pad_mode_str = "none"
            if da:
                pad_mode_str = "reflect" if dr else "edge"

            shard_data[local_off] = crop
            index_rows.append({
                "shard_id":        shard_id,
                "row_index":       row_start + local_off,
                "safe_id":         row["safe_id"],
                "local_z":         row["local_z"],
                "crop_y0":         row["crop_y0"],
                "crop_x0":         row["crop_x0"],
                "crop_y1":         row["crop_y1"],
                "crop_x1":         row["crop_x1"],
                "six_bin_label":   row.get("six_bin_label", ""),
                "shard_path":      f"shard_{shard_id:04d}.npy",
                "offset":          local_off,
                "padding_applied": da,
                "padding_mode":    pad_mode_str,
            })

        # NaN/Inf 검증
        if not np.isfinite(shard_data).all():
            print(f"[ABORT] shard {shard_id}에 NaN/Inf 포함", file=sys.stderr)
            sys.exit(1)

        # edge fallback 과다 검사
        if s_pad_e > EDGE_FALLBACK_ABORT_THR:
            print(f"[ABORT] shard {shard_id}: edge_fallback {s_pad_e} > "
                  f"threshold {EDGE_FALLBACK_ABORT_THR}", file=sys.stderr)
            sys.exit(1)

        total_pad_applied += s_pad_a
        total_pad_reflect += s_pad_r
        total_pad_edge    += s_pad_e

        shard_file = SHARD_ROOT / f"shard_{shard_id:04d}.npy"
        np.save(str(shard_file), shard_data)

        elapsed = time.perf_counter() - t_total_start
        print(f"  shard {shard_id:03d}/{n_shards-1}  rows={n_in_shard}  "
              f"pad={s_pad_a}(r={s_pad_r},e={s_pad_e})  "
              f"elapsed={elapsed:.0f}s  saved={shard_file.name}")

    t_total_end    = time.perf_counter()
    total_build_sec = t_total_end - t_total_start

    # ── 검증 ────────────────────────────────────────────────────────────────
    print("\n[검증]")
    idx_total = len(index_rows)
    ok_rows   = idx_total == total_rows
    print(f"  index_rows      : {idx_total:,}  "
          f"(expected {total_rows:,})  {'OK' if ok_rows else 'FAIL'}")
    bin_dist_out = {lbl: 0 for lbl in SIX_BIN_LABELS}
    for r in index_rows:
        lbl = r.get("six_bin_label", "")
        if lbl in bin_dist_out:
            bin_dist_out[lbl] += 1
    print(f"  six_bin 분포:")
    for lbl in SIX_BIN_LABELS:
        print(f"    {lbl}: {bin_dist_out[lbl]:,}")
    print(f"  total_pad_applied : {total_pad_applied}")
    print(f"  total_pad_reflect : {total_pad_reflect}")
    print(f"  total_pad_edge    : {total_pad_edge}")
    print(f"  build time        : {total_build_sec:.1f}s  ({total_build_sec/60:.1f}min)")

    # ── 파일 저장 ───────────────────────────────────────────────────────────
    idx_csv = SHARD_ROOT / "rd_d1s_full_shard_index.csv"
    write_csv(idx_csv, INDEX_FIELDS, index_rows)

    # 실제 shard 파일 크기 합산
    shard_total_bytes = sum(
        (SHARD_ROOT / f"shard_{i:04d}.npy").stat().st_size for i in range(n_shards)
    )
    shard_total_gb = shard_total_bytes / (1024 ** 3)

    gen_summary = {
        "total_rows":              total_rows,
        "n_shards":                n_shards,
        "shard_size_rows":         SHARD_SIZE,
        "total_shard_size_gb":     round(shard_total_gb, 3),
        "build_time_sec":          round(total_build_sec, 1),
        "build_time_min":          round(total_build_sec / 60, 2),
        "total_pad_applied":       total_pad_applied,
        "total_pad_reflect":       total_pad_reflect,
        "total_pad_edge":          total_pad_edge,
        "index_rows":              idx_total,
        "six_bin_distribution":    bin_dist_out,
        "ok":                      ok_rows and total_pad_edge == 0,
    }
    with open(SHARD_ROOT / "rd_d1s_full_shard_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(gen_summary, f, indent=2)
    print("  saved: rd_d1s_full_shard_generation_summary.json")

    # errors.csv (empty if no errors)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(
        REPORT_ROOT / "rd_d1s_errors.csv",
        ["phase", "shard_id", "offset", "safe_id", "error"],
        error_rows,
    )

    (SHARD_ROOT / "DONE_SHARD_BUILD").write_text(
        f"DONE\ntotal_rows={total_rows}\nn_shards={n_shards}\n"
        f"total_pad_applied={total_pad_applied}\n"
        f"total_pad_reflect={total_pad_reflect}\n"
        f"total_pad_edge={total_pad_edge}\n",
        encoding="utf-8",
    )
    print("  saved: DONE_SHARD_BUILD")

    verdict = "OK" if gen_summary["ok"] else "FAIL"
    print(f"\n  total_shard_size   : {shard_total_gb:.3f} GB")
    print(f"판정: full shard build {verdict}")
    if verdict == "FAIL":
        sys.exit(1)


# ── 4단계: --profile-shard-train ─────────────────────────────────────────────

def run_profile_shard_train():
    import numpy as np
    import torch
    import statistics as stats_mod

    print("=" * 72)
    print("RD-D1s: [PROFILE-SHARD-TRAIN]  warmup=20 timed=200")
    print("=" * 72)

    # 전제 확인
    done_marker = SHARD_ROOT / "DONE_SHARD_BUILD"
    if not done_marker.exists():
        print(f"[ABORT] DONE_SHARD_BUILD 없음 (--build-full-shards 먼저): {done_marker}",
              file=sys.stderr)
        sys.exit(1)
    idx_csv = SHARD_ROOT / "rd_d1s_full_shard_index.csv"
    if not idx_csv.exists():
        print(f"[ABORT] index CSV 없음: {idx_csv}", file=sys.stderr)
        sys.exit(1)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    profile_done = REPORT_ROOT / "DONE_SHARD_PROFILE"
    if profile_done.exists():
        print(f"[ABORT] DONE_SHARD_PROFILE 이미 존재: {profile_done}", file=sys.stderr)
        sys.exit(1)

    WARMUP_BATCHES = 20
    TIMED_BATCHES  = 200

    # [1] sampler 초기화
    print("\n[1] sampler 초기화")
    sampler = ShardedSixBinSampler(idx_csv, per_bin=PER_BIN, seed=SEED)

    # [2] 모델 준비
    print("\n[2] 모델 준비")
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

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # [3] train-like profile
    print(f"\n[3] train-like profile (warmup={WARMUP_BATCHES}, timed={TIMED_BATCHES})")
    shard_cache       = ShardMmapCache(max_size=16)
    timed_batch_times = []
    timed_crop_times  = []
    timed_fb_times    = []
    loss_nan_count    = 0
    loss_inf_count    = 0
    train_batch_rows  = []

    batch_idx = 0
    for step, batch_items in sampler.epoch_batches(0):
        t_crop_s = time.perf_counter()
        crops = []
        for item in batch_items:
            sp  = SHARD_ROOT / item["shard_path"]
            off = int(item["offset"])
            arr = shard_cache.get(sp)
            crops.append(arr[off].copy())
        t_crop_e = time.perf_counter()
        crop_t = t_crop_e - t_crop_s

        batch_np = np.stack(crops, axis=0)
        batch_t  = torch.from_numpy(batch_np).to(device)

        t_fb_s = time.perf_counter()
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
        t_fb_e = time.perf_counter()
        fb_t  = t_fb_e - t_fb_s
        tot_t = crop_t + fb_t
        loss_val = float(loss.item())

        is_warmup = int(batch_idx < WARMUP_BATCHES)
        train_batch_rows.append({
            "batch_idx":            batch_idx,
            "is_warmup":            is_warmup,
            "batch_total_sec":      round(tot_t, 5),
            "crop_load_sec":        round(crop_t, 5),
            "forward_backward_sec": round(fb_t, 5),
            "loss":                 round(loss_val, 6),
        })

        if not is_warmup:
            timed_batch_times.append(tot_t)
            timed_crop_times.append(crop_t)
            timed_fb_times.append(fb_t)
            if math.isnan(loss_val):
                loss_nan_count += 1
            if math.isinf(loss_val):
                loss_inf_count += 1
            timed_idx = batch_idx - WARMUP_BATCHES + 1
            if timed_idx % 50 == 1 or timed_idx == TIMED_BATCHES:
                print(f"    timed {timed_idx:3d}/{TIMED_BATCHES}: "
                      f"total={tot_t:.3f}s  crop={crop_t:.4f}s  fb={fb_t:.3f}s  "
                      f"loss={loss_val:.4f}")

        batch_idx += 1
        if batch_idx >= WARMUP_BATCHES + TIMED_BATCHES:
            break

    gpu_peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda" else 0.0
    )
    student.eval()
    del optimizer, teacher, student

    def _pct(data, pct):
        if not data:
            return 0.0
        s = sorted(data)
        return s[min(int(pct / 100.0 * len(s)), len(s) - 1)]

    shard_mean   = float(stats_mod.mean(timed_batch_times))   if timed_batch_times else 0.0
    shard_median = float(stats_mod.median(timed_batch_times)) if timed_batch_times else 0.0
    shard_p95    = _pct(timed_batch_times, 95)
    crop_mean    = float(stats_mod.mean(timed_crop_times))    if timed_crop_times  else 0.0
    fb_mean      = float(stats_mod.mean(timed_fb_times))      if timed_fb_times    else 0.0
    speedup      = ONTHEFLY_MEAN_SEC / shard_mean             if shard_mean > 0    else 0.0

    # 예상 시간 계산 (val/scoring은 on-the-fly 그대로 사용)
    est_train_sec  = shard_mean * TOTAL_TRAIN_STEPS
    est_train_min  = est_train_sec / 60
    EST_VAL_MIN    = 0.5    # profile v2 결과
    EST_CAND_MIN   = 3.1    # profile v2 결과
    est_total_raw  = est_train_min + EST_VAL_MIN + EST_CAND_MIN
    est_total_ovh  = est_total_raw * 1.15

    import datetime
    finish_dt  = datetime.datetime.now() + datetime.timedelta(seconds=est_total_ovh * 60)
    finish_str = finish_dt.strftime("%Y-%m-%d %H:%M")

    print(f"\n  [shard train profile 결과]")
    print(f"  timed_batches              : {len(timed_batch_times)}")
    print(f"  shard_batch_mean           : {shard_mean:.4f}s")
    print(f"  shard_batch_median         : {shard_median:.4f}s")
    print(f"  shard_batch_p95            : {shard_p95:.4f}s")
    print(f"  crop_load_mean             : {crop_mean:.4f}s")
    print(f"  forward_backward_mean      : {fb_mean:.4f}s")
    print(f"  loss_nan_count             : {loss_nan_count}")
    print(f"  loss_inf_count             : {loss_inf_count}")
    print(f"  on-the-fly baseline        : {ONTHEFLY_MEAN_SEC:.4f}s  ({ONTHEFLY_TRAIN_MIN:.1f}min)")
    print(f"  speedup                    : {speedup:.1f}x")
    print(f"  est_train_total            : {est_train_min:.1f}min  ({est_train_sec:.0f}s)")
    print(f"  est_val + scoring          : {EST_VAL_MIN:.1f} + {EST_CAND_MIN:.1f} min")
    print(f"  est_total_raw              : {est_total_raw:.1f}min")
    print(f"  est_total_with_overhead    : {est_total_ovh:.1f}min  (×1.15)")
    print(f"  expected_finish            : {finish_str}")
    print(f"  GPU peak memory            : {gpu_peak_mb:.1f} MB")

    all_checks_passed    = (
        len(timed_batch_times) >= TIMED_BATCHES
        and loss_nan_count == 0
        and loss_inf_count == 0
    )
    full_train_eligible  = est_train_min <= 60.0

    print(f"\n[통과 조건]")
    print(f"  timed_batches >= 200       : {len(timed_batch_times) >= TIMED_BATCHES}  ({len(timed_batch_times)})")
    print(f"  loss_nan_count == 0        : {loss_nan_count == 0}  ({loss_nan_count})")
    print(f"  loss_inf_count == 0        : {loss_inf_count == 0}  ({loss_inf_count})")
    print(f"  est_train_min <= 60min     : {full_train_eligible}  ({est_train_min:.1f}min)")

    verdict = "PASS" if all_checks_passed else "FAIL"
    print(f"\n판정: profile {verdict}")
    print(f"full_train_eligible (<=60min): {full_train_eligible}")

    # ── 파일 저장 ───────────────────────────────────────────────────────────
    write_csv(
        REPORT_ROOT / "rd_d1s_shard_train_profile.csv",
        ["batch_idx", "is_warmup", "batch_total_sec", "crop_load_sec",
         "forward_backward_sec", "loss"],
        train_batch_rows,
    )
    profile_summary = {
        "timed_batches":                  len(timed_batch_times),
        "shard_batch_mean_sec":           round(shard_mean,   5),
        "shard_batch_median_sec":         round(shard_median, 5),
        "shard_batch_p95_sec":            round(shard_p95,    5),
        "crop_load_mean_sec":             round(crop_mean,    5),
        "forward_backward_mean_sec":      round(fb_mean,      5),
        "loss_nan_count":                 loss_nan_count,
        "loss_inf_count":                 loss_inf_count,
        "onthefly_baseline_mean_sec":     ONTHEFLY_MEAN_SEC,
        "onthefly_baseline_train_min":    ONTHEFLY_TRAIN_MIN,
        "speedup":                        round(speedup, 2),
        "est_train_total_sec":            round(est_train_sec, 1),
        "est_train_total_min":            round(est_train_min, 1),
        "est_val_min":                    EST_VAL_MIN,
        "est_cand_scoring_min":           EST_CAND_MIN,
        "est_total_raw_min":              round(est_total_raw, 1),
        "est_total_with_overhead_min":    round(est_total_ovh, 1),
        "expected_finish_time":           finish_str,
        "gpu_peak_memory_mb":             round(gpu_peak_mb, 1),
        "all_checks_passed":              all_checks_passed,
        "full_train_eligible_le60min":    full_train_eligible,
    }
    with open(REPORT_ROOT / "rd_d1s_shard_train_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile_summary, f, indent=2)
    print("  saved: rd_d1s_shard_train_profile.json")

    md_lines = [
        "# RD-D1s Shard Train Profile",
        "",
        f"## 판정: {verdict}",
        "",
        "## Shard Train Profile",
        "| 항목 | 값 |",
        "|---|---|",
        f"| timed_batches | {len(timed_batch_times)} |",
        f"| shard_batch_mean | {shard_mean:.5f}s |",
        f"| shard_batch_median | {shard_median:.5f}s |",
        f"| shard_batch_p95 | {shard_p95:.5f}s |",
        f"| crop_load_mean | {crop_mean:.5f}s |",
        f"| forward_backward_mean | {fb_mean:.5f}s |",
        f"| loss_nan | {loss_nan_count} |",
        f"| loss_inf | {loss_inf_count} |",
        "",
        "## 속도 비교 (vs RD-D1 on-the-fly profile v2)",
        "| 항목 | 값 |",
        "|---|---|",
        f"| on-the-fly baseline | {ONTHEFLY_MEAN_SEC:.4f}s ({ONTHEFLY_TRAIN_MIN:.1f}min) |",
        f"| shard mean | {shard_mean:.4f}s |",
        f"| speedup | {speedup:.1f}x |",
        f"| est_train_total | {est_train_min:.1f}min |",
        f"| est_total_with_overhead | {est_total_ovh:.1f}min |",
        f"| expected_finish | {finish_str} |",
        f"| gpu_peak_mb | {gpu_peak_mb:.1f} |",
        f"| full_train_eligible | {full_train_eligible} |",
        f"| all_checks_passed | {all_checks_passed} |",
    ]
    with open(REPORT_ROOT / "rd_d1s_shard_train_profile.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("  saved: rd_d1s_shard_train_profile.md")

    profile_done.write_text(
        f"DONE_SHARD_PROFILE\nall_checks_passed={all_checks_passed}\n"
        f"full_train_eligible={full_train_eligible}\n",
        encoding="utf-8",
    )
    print("  saved: DONE_SHARD_PROFILE")

    print()
    print("=" * 72)
    print(f"  PROFILE-SHARD-TRAIN COMPLETE  판정: {verdict}")
    print(f"  shard batch mean     : {shard_mean:.4f}s")
    print(f"  speedup              : {speedup:.1f}x  (vs on-the-fly {ONTHEFLY_MEAN_SEC:.4f}s)")
    print(f"  est train total      : {est_train_min:.1f}min")
    print(f"  est total (×1.15)    : {est_total_ovh:.1f}min")
    print(f"  expected finish      : {finish_str}")
    print(f"  GPU peak memory      : {gpu_peak_mb:.1f}MB")
    print(f"  NaN/Inf loss         : {loss_nan_count} / {loss_inf_count}")
    print(f"  all_checks_passed    : {all_checks_passed}")
    print(f"  full_train_eligible  : {full_train_eligible}")
    print("=" * 72)

    if not all_checks_passed:
        sys.exit(1)


# ── 진입점 ────────────────────────────────────────────────────────────────────

if IS_DRY_PLAN_SHARD:
    run_dry_plan_shard()
elif IS_SMOKE_SHARD:
    run_smoke_shard_profile()
elif IS_BUILD_FULL_SHARDS:
    run_build_full_shards()
elif IS_PROFILE_SHARD_TRAIN:
    run_profile_shard_train()
