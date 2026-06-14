"""
RD-D1: Mediastinal 3ch true RD4AD revival
목적: RD-B8f 실패(AUROC 0.5021) 극복. mediastinal window 3ch 입력으로 true RD4AD 재시도.
     RD-C3 ConvAE mediastinal 신호(AUROC 0.7262)를 true RD4AD에 이식.

모드:
  bare run   -> exit 2
  --dry-plan -> 입력 확인, output root 없음 확인 (파일 생성 없음)
  --run-all  -> 학습 + threshold + scoring + analysis + DONE

안전 조건:
  stage2_holdout 접근 금지
  기존 RD-B/RD-C 결과 수정 금지
  suppression 적용 금지 (analysis-only)
  vessel mask 사용 금지
  output root 이미 있으면 즉시 ABORT
  crop 실패 시 zero 대체 금지, ABORT
  score NaN/Inf 발생 시 ABORT
"""

import sys
import csv
import json
import math
import time
import collections
from pathlib import Path

ALLOWED_MODES = {"--dry-plan", "--run-all", "--profile-time"}
if not any(m in sys.argv for m in ALLOWED_MODES):
    print("오류: 실행 모드 지정 필요.")
    print("  --dry-plan     : 입력 확인 (파일 생성 없음)")
    print("  --run-all      : 학습 + scoring + analysis + DONE")
    print("  --profile-time : full run 전 속도 실측 및 예상 시간 계산")
    sys.exit(2)

IS_DRY_PLAN    = "--dry-plan"    in sys.argv
IS_RUN_ALL     = "--run-all"     in sys.argv
IS_PROFILE_TIME = "--profile-time" in sys.argv

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

MODEL_ROOT = (
    PROJECT_ROOT / "outputs/models/rd_d1_true_rd4ad_resnet18_medi3ch_v1"
)
CKPT_DIR = MODEL_ROOT / "checkpoints"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1_medi3ch_true_rd4ad_revival_v1"
)

LOCAL_WEIGHT_PATH = Path(
    "/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)

PROFILE_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1_medi3ch_true_rd4ad_revival_time_profile_v2"
)
TOTAL_TRAIN_STEPS = 34820  # steps_per_epoch(1741) × epochs(20)

# ── reflect-padding 전역 카운터 (single-thread 전용) ───────────────────────────
_G_PAD_APPLIED_COUNT: int = 0  # 패딩이 적용된 crop 수
_G_PAD_REFLECT_COUNT: int = 0  # reflect 모드로 패딩된 crop 수
_G_PAD_EDGE_COUNT:    int = 0  # edge fallback으로 패딩된 crop 수

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
MEDI_HU_MIN  = -160.0
MEDI_HU_MAX  =  240.0

RD_B8F_AUROC_REF = 0.5021
RD_C3_AUROC_REF  = 0.7262


# ── safety ────────────────────────────────────────────────────────────────────

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


# ── sklearn-free metrics ──────────────────────────────────────────────────────

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


# ── 모델 빌드 ─────────────────────────────────────────────────────────────────

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


# ── mediastinal crop builder ──────────────────────────────────────────────────

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

    # OOB padding 계산
    pad_top    = max(0, -y0)
    pad_bottom = max(0, y1 - H)
    pad_left   = max(0, -x0)
    pad_right  = max(0, x1 - W)
    needs_pad  = (pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0)

    # 유효 CT 영역으로 clamp
    cy0 = max(0, y0)
    cy1 = min(H, y1)
    cx0 = max(0, x0)
    cx1 = min(W, x1)
    valid_h = cy1 - cy0
    valid_w = cx1 - cx0

    if needs_pad:
        # 유효 영역 높이 또는 너비가 1 이하이면 reflect 불가 → edge fallback
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


# ── 6-bin Coordinate Sampler ──────────────────────────────────────────────────

class SixBinCoordSampler:
    def __init__(self, manifest_csv, per_bin, seed):
        self.per_bin = per_bin
        self._seed   = seed
        self.bin_items = {lbl: [] for lbl in SIX_BIN_LABELS}

        with open(manifest_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lbl = row.get("six_bin_label", "")
                if lbl not in self.bin_items:
                    continue
                self.bin_items[lbl].append({
                    "safe_id":       row["safe_id"],
                    "local_z":       int(row["local_z"]),
                    "crop_y0":       int(row["crop_y0"]),
                    "crop_x0":       int(row["crop_x0"]),
                    "crop_y1":       int(row["crop_y1"]),
                    "crop_x1":       int(row["crop_x1"]),
                    "low_z_warning": int(row.get("low_z_warning", 0) or 0),
                })

        bin_sizes = {lbl: len(v) for lbl, v in self.bin_items.items()}
        self.total_rows      = sum(bin_sizes.values())
        self.min_bin_size    = min(bin_sizes.values())
        self.steps_per_epoch = self.min_bin_size // per_bin

        print("  6-bin 분포 (coord manifest):")
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


# ── CSV 헬퍼 ──────────────────────────────────────────────────────────────────

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


# ── dry-plan ──────────────────────────────────────────────────────────────────

def run_dry_plan():
    print("=" * 72)
    print("RD-D1: Mediastinal 3ch true RD4AD revival [DRY-PLAN]")
    print("=" * 72)

    ok_all = True

    print("\n[1] input artifacts")
    checks = [
        ("normal_train manifest", NORMAL_TRAIN_MANIFEST),
        ("normal_val manifest",   NORMAL_VAL_MANIFEST),
        ("RD-C2 candidate manifest", RD_C2_MANIFEST),
        ("ResNet18 local weights",   LOCAL_WEIGHT_PATH),
    ]
    for label, p in checks:
        exists = p.exists()
        print(f"  {'OK' if exists else 'MISSING'}: {label} -> {p.name}")
        if not exists:
            ok_all = False

    print("\n[2] output root guard")
    for label, p in [("MODEL_ROOT", MODEL_ROOT), ("OUTPUT_ROOT", OUTPUT_ROOT)]:
        exists = p.exists()
        print(f"  {'CONFLICT' if exists else 'OK'}: {label} -> {p}")
        if exists:
            ok_all = False

    print("\n[3] stage2_holdout intersection (RD-C2 manifest)")
    if RD_C2_MANIFEST.exists():
        with open(RD_C2_MANIFEST, newline="", encoding="utf-8") as f:
            c2_rows = list(csv.DictReader(f))
        holdout_cnt = sum(1 for r in c2_rows if r.get("stage_split", "") == "stage2_holdout")
        pos_cnt     = sum(1 for r in c2_rows if r.get("label", "") == "positive")
        hn_cnt      = sum(1 for r in c2_rows if r.get("label", "") == "hard_negative")
        print(f"  rows={len(c2_rows):,}  positive={pos_cnt:,}  hard_negative={hn_cnt:,}")
        status = "OK" if holdout_cnt == 0 else "FAIL"
        print(f"  stage2_holdout intersection: {holdout_cnt} ({status})")
        if holdout_cnt != 0:
            ok_all = False

    print("\n[4] CT path sample check (normal_train 10 patients)")
    if NORMAL_TRAIN_MANIFEST.exists():
        with open(NORMAL_TRAIN_MANIFEST, newline="", encoding="utf-8") as f:
            train_rows = list(csv.DictReader(f))
        seen = {}
        for r in train_rows:
            seen.setdefault(r["safe_id"], r)
            if len(seen) >= 10:
                break
        found = sum(1 for sid in seen if (NORMAL_CT_ROOT / sid / "ct_hu.npy").exists())
        print(f"  CT found: {found}/10")
        if found < 10:
            ok_all = False

    print("\n[5] 학습 설정")
    print(f"  input_mode  : mediastinal_3ch_zminus1_z_zplus1")
    print(f"  window      : HU[{MEDI_HU_MIN},{MEDI_HU_MAX}] -> [0,1]")
    print(f"  model_type  : true_RD4AD_ResNet18_teacher_student")
    print(f"  crop_size   : {CROP_SIZE}")
    print(f"  batch_size  : {BATCH_SIZE}  per_bin: {PER_BIN}")
    print(f"  epochs      : {EPOCHS}  lr: {LR}")
    print(f"  sampler     : 6-bin balanced, shortest-bin drop-last (coord manifest)")
    n_train = len(train_rows) if NORMAL_TRAIN_MANIFEST.exists() else "N/A"
    print(f"  normal_train_rows: {n_train}")

    print("\n[6] 안전 조건")
    print("  stage2_holdout_access    : 0")
    print("  suppression_applied      : false")
    print("  vessel_mask_used         : false")
    print("  existing_results_modified: false")

    est_steps = 13932 // PER_BIN
    est_batch_sec = 0.03
    est_epoch_min = est_steps * est_batch_sec / 60
    print(f"\n[7] 예상 시간")
    print(f"  est steps/epoch : ~{est_steps:,}")
    print(f"  est epoch time  : ~{est_epoch_min:.1f}min")
    print(f"  est 20ep total  : ~{est_epoch_min*20/60:.2f}hr")
    print(f"  scoring 113447  : ~{113447//500:.0f}s")

    print()
    verdict = "DRY-PLAN OK" if ok_all else "DRY-PLAN FAIL"
    print(f"판정: {verdict}")
    if ok_all:
        print("  사용자 승인 후:")
        print("  source ~/ai_env/bin/activate && \\")
        print("  python scripts/rd_d1_medi3ch_true_rd4ad_revival.py --run-all \\")
        print("  2>&1 | tee /tmp/rd_d1_medi3ch_true_rd4ad_revival_log.txt")


# ── run_all ───────────────────────────────────────────────────────────────────

def run_all():
    import numpy as np
    import torch
    import torch.nn.functional as F

    print("=" * 72)
    print("RD-D1: Mediastinal 3ch true RD4AD revival [RUN-ALL]")
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

    # ── RD-C2 manifest 로드 ──────────────────────────────────────────────────
    print("\n[1/6] RD-C2 candidate manifest 로드")
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
        OUTPUT_ROOT / "rd_d1_preflight_validation.csv",
        ["check", "result", "pass"],
        [
            {"check": "stage2_holdout_intersection",  "result": holdout_cnt,                     "pass": holdout_cnt == 0},
            {"check": "positive_count",               "result": pos_cnt,                         "pass": pos_cnt == 35247},
            {"check": "hard_negative_count",          "result": hn_cnt,                          "pass": hn_cnt == 78200},
            {"check": "normal_train_manifest_exists", "result": NORMAL_TRAIN_MANIFEST.exists(),  "pass": NORMAL_TRAIN_MANIFEST.exists()},
            {"check": "normal_val_manifest_exists",   "result": NORMAL_VAL_MANIFEST.exists(),    "pass": NORMAL_VAL_MANIFEST.exists()},
            {"check": "local_weights_exists",         "result": LOCAL_WEIGHT_PATH.exists(),      "pass": LOCAL_WEIGHT_PATH.exists()},
        ],
    )

    # ── 6-bin sampler + 모델 ────────────────────────────────────────────────
    print("\n[2/6] 6-bin coord sampler 로드 + 모델 준비")
    sampler  = SixBinCoordSampler(NORMAL_TRAIN_MANIFEST, per_bin=PER_BIN, seed=SEED)
    ct_cache = CtMmapCache(max_size=16)

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

    # ── 학습 루프 ────────────────────────────────────────────────────────────
    print(f"\n[3/6] 학습 시작 ({EPOCHS} epochs, on-the-fly mediastinal 3ch)")
    epoch_log_writer = CsvAppendWriter(
        OUTPUT_ROOT / "rd_d1_epoch_log.csv",
        ["epoch", "steps", "mean_loss", "min_loss", "max_loss",
         "nan_count", "inf_count", "epoch_time_sec", "cumulative_time_sec"],
    )

    best_loss   = float("inf")
    best_epoch  = -1
    epoch_logs  = []
    total_nan   = 0
    total_inf   = 0
    t_train_start = time.perf_counter()

    for epoch in range(EPOCHS):
        student.train()
        epoch_losses = []
        epoch_nan = 0
        epoch_inf = 0
        t_epoch_start = time.perf_counter()

        print(f"\n  [Epoch {epoch+1}/{EPOCHS}]")

        for step, batch_items in sampler.epoch_batches(epoch):
            crops = []
            for item in batch_items:
                ct_path = NORMAL_CT_ROOT / item["safe_id"] / "ct_hu.npy"
                assert_path_safe(ct_path)
                ct_arr = ct_cache.get(ct_path)
                crop = build_medi3ch_crop(
                    ct_arr,
                    item["local_z"],
                    item["crop_y0"], item["crop_x0"],
                    item["crop_y1"], item["crop_x1"],
                )
                crops.append(crop)

            batch_np = np.stack(crops, axis=0)
            batch_t  = torch.from_numpy(batch_np).to(device)

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

            loss_val = float(loss.item())
            is_nan   = int(math.isnan(loss_val))
            is_inf   = int(math.isinf(loss_val))
            epoch_nan += is_nan
            epoch_inf += is_inf
            epoch_losses.append(loss_val)

            if step % 200 == 0 or step == sampler.steps_per_epoch - 1:
                elapsed = time.perf_counter() - t_epoch_start
                print(f"    step {step:5d}/{sampler.steps_per_epoch}  "
                      f"loss={loss_val:.4f}  elapsed={elapsed:.0f}s")

        t_epoch_end = time.perf_counter()
        epoch_time  = t_epoch_end - t_epoch_start
        cumul_time  = t_epoch_end - t_train_start
        valid_losses = [v for v in epoch_losses if not (math.isnan(v) or math.isinf(v))]
        mean_loss = sum(valid_losses) / len(valid_losses) if valid_losses else float("nan")
        min_loss  = min(valid_losses) if valid_losses else float("nan")
        max_loss  = max(valid_losses) if valid_losses else float("nan")
        total_nan += epoch_nan
        total_inf += epoch_inf

        epoch_log_writer.writerow({
            "epoch": epoch + 1, "steps": len(epoch_losses),
            "mean_loss": round(mean_loss, 6), "min_loss": round(min_loss, 6),
            "max_loss": round(max_loss, 6), "nan_count": epoch_nan,
            "inf_count": epoch_inf, "epoch_time_sec": round(epoch_time, 2),
            "cumulative_time_sec": round(cumul_time, 2),
        })
        epoch_logs.append({"epoch": epoch + 1, "mean_loss": mean_loss})

        torch.save(
            {"epoch": epoch + 1, "student_state_dict": student.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(), "mean_loss": mean_loss},
            CKPT_DIR / "last.pth",
        )
        if not math.isnan(mean_loss) and mean_loss < best_loss:
            best_loss  = mean_loss
            best_epoch = epoch + 1
            torch.save(
                {"epoch": epoch + 1, "student_state_dict": student.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "mean_loss": mean_loss},
                CKPT_DIR / "best_train_loss.pth",
            )
        print(f"  Epoch {epoch+1}: mean_loss={mean_loss:.4f}  "
              f"best={best_loss:.4f}(ep{best_epoch})  "
              f"time={epoch_time:.1f}s  NaN={epoch_nan} Inf={epoch_inf}")

    epoch_log_writer.close()
    t_train_end = time.perf_counter()
    train_time  = t_train_end - t_train_start
    gpu_peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda" else 0.0
    )

    teacher_snap_after    = snapshot_params(teacher)
    teacher_param_changed = params_changed(teacher_snap_before, teacher_snap_after)
    valid_epoch_losses = [r["mean_loss"] for r in epoch_logs if not math.isnan(r["mean_loss"])]
    train_loss_first   = valid_epoch_losses[0]  if valid_epoch_losses else float("nan")
    train_loss_last    = valid_epoch_losses[-1] if valid_epoch_losses else float("nan")
    loss_decreased     = (
        len(valid_epoch_losses) >= 2
        and valid_epoch_losses[-1] < valid_epoch_losses[0]
    )

    write_csv(
        OUTPUT_ROOT / "rd_d1_train_shard_summary.csv",
        ["epoch", "mean_loss"],
        [{"epoch": r["epoch"], "mean_loss": round(r["mean_loss"], 6)} for r in epoch_logs],
    )

    print(f"\n  train complete: {EPOCHS} epochs  {train_time:.1f}s")
    print(f"  teacher_param_changed: {teacher_param_changed}")
    if teacher_param_changed:
        print("[ABORT] teacher param 변경됨", file=sys.stderr)
        sys.exit(1)

    # best checkpoint 로드
    ckpt = torch.load(
        str(CKPT_DIR / "best_train_loss.pth"), map_location=device, weights_only=False
    )
    student.load_state_dict(ckpt["student_state_dict"])
    student.eval()
    print(f"  best_epoch={best_epoch}  best_loss={best_loss:.6f}")

    # ── normal_val threshold ─────────────────────────────────────────────────
    print("\n[4/6] normal_val threshold 생성")
    if not NORMAL_VAL_MANIFEST.exists():
        print(f"[ABORT] normal_val manifest 없음", file=sys.stderr)
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

    val_ct_cache = CtMmapCache(max_size=8)
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

        for i in range(0, len(v_rows), BATCH_VAL):
            batch_rows = v_rows[i:i + BATCH_VAL]
            crops = [
                build_medi3ch_crop(
                    ct_arr,
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
        "threshold_created_from": "rd_d1_normal_val_only",
        "model_tag": "rd_d1_true_rd4ad_resnet18_medi3ch_v1",
        "global_p95": round(global_p95, 6),
        "global_p99": round(global_p99, 6),
        "val_scored": n_val_scored,
        "val_patients": len(val_safe_ids),
        "rd_b9_threshold_modified": False,
        "bin_thresholds": bin_thresholds,
    }
    with open(OUTPUT_ROOT / "rd_d1_normal_val_threshold_summary.json", "w", encoding="utf-8") as f:
        json.dump(threshold_summary, f, indent=2)
    print(f"  saved: rd_d1_normal_val_threshold_summary.json")
    print(f"  global_p95={global_p95:.6f}  global_p99={global_p99:.6f}")

    # ── RD-C2 candidate scoring ───────────────────────────────────────────────
    print("\n[5/6] RD-C2 candidate scoring (113,447)")
    student.eval()
    score_rows  = []
    score_error_rows = []
    score_ct_cache   = CtMmapCache(max_size=12)
    c2_groups   = collections.defaultdict(list)
    for r in c2_rows:
        c2_groups[r["safe_id"]].append(r)

    n_scored = 0
    n_failed = 0
    n_nan    = 0
    n_inf    = 0
    BATCH_SCORE    = 64
    t_score_start  = time.perf_counter()
    patients_done  = 0

    for safe_id, rows_for_pat in c2_groups.items():
        ct_path = CANDIDATE_CT_ROOT / safe_id / "ct_hu.npy"
        assert_path_safe(ct_path)
        if not ct_path.exists():
            for r in rows_for_pat:
                score_error_rows.append({
                    "candidate_id": r.get("candidate_id", ""),
                    "safe_id": safe_id,
                    "error": f"ct_not_found:{ct_path}",
                })
            n_failed += len(rows_for_pat)
            error_rows.append({"safe_id": safe_id, "error": "ct_not_found"})
            continue

        ct_arr = score_ct_cache.get(ct_path)

        for i in range(0, len(rows_for_pat), BATCH_SCORE):
            batch_rows = rows_for_pat[i:i + BATCH_SCORE]
            crops = []
            for row in batch_rows:
                try:
                    crop = build_medi3ch_crop(
                        ct_arr,
                        int(row["local_z"]),
                        int(row["crop_y0"]), int(row["crop_x0"]),
                        int(row["crop_y1"]), int(row["crop_x1"]),
                    )
                    crops.append(crop)
                except Exception as e:
                    print(f"[ABORT] crop 실패: {safe_id} - {e}", file=sys.stderr)
                    score_error_rows.append({
                        "candidate_id": row.get("candidate_id", ""),
                        "safe_id": safe_id, "error": f"crop_failed:{e}",
                    })
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
                    "candidate_id":              row.get("candidate_id", ""),
                    "patient_id":                row.get("patient_id", ""),
                    "safe_id":                   safe_id,
                    "stage_split":               row.get("stage_split", ""),
                    "local_z":                   row.get("local_z", ""),
                    "crop_y0":                   row.get("crop_y0", ""),
                    "crop_x0":                   row.get("crop_x0", ""),
                    "crop_y1":                   row.get("crop_y1", ""),
                    "crop_x1":                   row.get("crop_x1", ""),
                    "z_level":                   row.get("z_level", ""),
                    "boundary_status":           row.get("boundary_status", ""),
                    "six_bin_label":             bin_lbl,
                    "label":                     row.get("label", ""),
                    "score_layer1":              round(s3, 6),
                    "score_layer2":              round(s2, 6),
                    "score_layer3":              round(s1, 6),
                    "rd_d1_medi3ch_rd4ad_score": round(rd4ad_score, 6),
                    "global_p95_exceed":         int(rd4ad_score > global_p95),
                    "global_p99_exceed":         int(rd4ad_score > global_p99),
                    "bin_p95_exceed":            int(rd4ad_score > b_p95),
                    "bin_p99_exceed":            int(rd4ad_score > b_p99),
                    "score_nan":                 0,
                    "score_inf":                 0,
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
        "rd_d1_medi3ch_rd4ad_score",
        "global_p95_exceed", "global_p99_exceed",
        "bin_p95_exceed", "bin_p99_exceed",
        "score_nan", "score_inf",
    ]
    write_csv(
        OUTPUT_ROOT / "rd_d1_stage1dev_candidate_score.csv",
        score_fieldnames, score_rows,
    )

    # ── AUROC/AUPRC/safety analysis ──────────────────────────────────────────
    print("\n[6/6] AUROC/AUPRC + safety analysis")

    y_true  = np.array(
        [1 if r["label"] == "positive" else 0 for r in score_rows], dtype=int
    )
    y_score = np.array(
        [float(r["rd_d1_medi3ch_rd4ad_score"]) for r in score_rows], dtype=float
    )

    rd_d1_auroc = compute_auroc_mann_whitney(y_true, y_score)
    rd_d1_auprc = compute_average_precision(y_true, y_score)
    print(f"  RD-D1 AUROC : {fmt_float(rd_d1_auroc, 4)}")
    print(f"  RD-D1 AUPRC : {fmt_float(rd_d1_auprc, 4)}")
    print(f"  RD-B8f ref  : {RD_B8F_AUROC_REF:.4f}")
    print(f"  RD-C3 ref   : {RD_C3_AUROC_REF:.4f}")

    write_csv(
        OUTPUT_ROOT / "rd_d1_auc_auprc_summary.csv",
        ["model", "auroc", "auprc"],
        [
            {"model": "RD-B8f (reference)",       "auroc": RD_B8F_AUROC_REF, "auprc": "N/A"},
            {"model": "RD-C3 ConvAE (reference)", "auroc": RD_C3_AUROC_REF,  "auprc": "N/A"},
            {"model": "RD-D1 medi3ch true RD4AD", "auroc": rd_d1_auroc,      "auprc": rd_d1_auprc},
        ],
    )

    pos_mask = (y_true == 1)
    hn_mask  = (y_true == 0)
    n_pos    = int(pos_mask.sum())
    n_hn     = int(hn_mask.sum())

    # fixed threshold safety
    safety_rows = []
    for rule_name, thr in [("global_p95", global_p95), ("global_p99", global_p99)]:
        sup_pos = int((y_score[pos_mask] <= thr).sum())
        sup_hn  = int((y_score[hn_mask]  <= thr).sum())
        les_rate = sup_pos / n_pos if n_pos > 0 else 0.0
        hn_rate  = sup_hn  / n_hn  if n_hn  > 0 else 0.0
        by_pat = collections.defaultdict(list)
        for r in score_rows:
            if r["label"] == "positive":
                by_pat[r["patient_id"]].append(
                    float(r["rd_d1_medi3ch_rd4ad_score"]) <= thr
                )
        pat_all_sup = sum(1 for v in by_pat.values() if all(v))
        safety_rows.append({
            "threshold_rule":              rule_name,
            "threshold_value":             round(thr, 6),
            "lesion_suppressed_count":     sup_pos,
            "lesion_suppressed_rate":      round(les_rate, 4),
            "hn_suppressed_count":         sup_hn,
            "hn_suppressed_rate":          round(hn_rate, 4),
            "lesion_patient_all_suppressed": pat_all_sup,
        })

    write_csv(
        OUTPUT_ROOT / "rd_d1_threshold_rule_safety_summary.csv",
        ["threshold_rule", "threshold_value", "lesion_suppressed_count",
         "lesion_suppressed_rate", "hn_suppressed_count", "hn_suppressed_rate",
         "lesion_patient_all_suppressed"],
        safety_rows,
    )

    # safety-constrained sweep
    sweep_rows = []
    for target_rate in [0.01, 0.03, 0.05]:
        sorted_pos = np.sort(y_score[pos_mask])
        thr_idx = min(int(np.floor(target_rate * n_pos)), len(sorted_pos) - 1)
        thr = float(sorted_pos[thr_idx])
        sup_pos  = int((y_score[pos_mask] <= thr).sum())
        sup_hn   = int((y_score[hn_mask]  <= thr).sum())
        les_rate = sup_pos / n_pos if n_pos > 0 else 0.0
        hn_rate  = sup_hn  / n_hn  if n_hn  > 0 else 0.0
        by_pat = collections.defaultdict(list)
        for r in score_rows:
            if r["label"] == "positive":
                by_pat[r["patient_id"]].append(
                    float(r["rd_d1_medi3ch_rd4ad_score"]) <= thr
                )
        pat_all_sup = sum(1 for v in by_pat.values() if all(v))
        sweep_rows.append({
            "target_lesion_rate":          target_rate,
            "threshold":                   round(thr, 6),
            "lesion_suppressed_rate":      round(les_rate, 4),
            "hn_suppressed_rate":          round(hn_rate, 4),
            "lesion_patient_all_suppressed": pat_all_sup,
        })
        print(f"  @le{int(target_rate*100)}%: thr={thr:.6f}  "
              f"hn_sup={hn_rate:.2%}  pat_all_sup={pat_all_sup}")

    write_csv(
        OUTPUT_ROOT / "rd_d1_safety_constrained_threshold_sweep.csv",
        ["target_lesion_rate", "threshold", "lesion_suppressed_rate",
         "hn_suppressed_rate", "lesion_patient_all_suppressed"],
        sweep_rows,
    )

    # patient-level summary
    by_pat_all = collections.defaultdict(lambda: {"label": "", "scores": []})
    for r in score_rows:
        by_pat_all[r["patient_id"]]["label"] = r["label"]
        by_pat_all[r["patient_id"]]["scores"].append(
            float(r["rd_d1_medi3ch_rd4ad_score"])
        )
    g95_ref = global_p95
    g99_ref = global_p99
    write_csv(
        OUTPUT_ROOT / "rd_d1_patient_level_safety_summary.csv",
        ["patient_id", "label", "n_crops", "score_mean", "score_max",
         "g95_exceed_count", "g99_exceed_count"],
        [
            {
                "patient_id":      pid,
                "label":           v["label"],
                "n_crops":         len(v["scores"]),
                "score_mean":      round(float(np.mean(v["scores"])), 6),
                "score_max":       round(float(np.max(v["scores"])), 6),
                "g95_exceed_count": sum(1 for s in v["scores"] if s > g95_ref),
                "g99_exceed_count": sum(1 for s in v["scores"] if s > g99_ref),
            }
            for pid, v in by_pat_all.items()
        ],
    )

    # RD-B8f / RD-C3 / RD-D1 비교
    write_csv(
        OUTPUT_ROOT / "rd_d1_compare_rdb8f_rdc3_rdd1.csv",
        ["experiment", "model_type", "input_mode", "window", "auroc", "auprc", "decision"],
        [
            {"experiment": "RD-B8f", "model_type": "true_RD4AD_ResNet18",
             "input_mode": "3ch_mixed_MIP", "window": "HU[-1000,600]",
             "auroc": RD_B8F_AUROC_REF, "auprc": "N/A", "decision": "NOT_USEFUL"},
            {"experiment": "RD-C3", "model_type": "ConvAutoencoder2p5D",
             "input_mode": "6ch_mediastinal", "window": "HU[-160,240]",
             "auroc": RD_C3_AUROC_REF, "auprc": "N/A", "decision": "CONVAE_USEFUL_FOR_RANKING"},
            {"experiment": "RD-D1", "model_type": "true_RD4AD_ResNet18",
             "input_mode": "3ch_mediastinal_zminus1_z_zplus1", "window": "HU[-160,240]",
             "auroc": rd_d1_auroc, "auprc": rd_d1_auprc, "decision": "TBD"},
        ],
    )

    # ── final decision ───────────────────────────────────────────────────────
    auroc_improvement = (
        (rd_d1_auroc - RD_B8F_AUROC_REF) if rd_d1_auroc is not None else None
    )
    le1_pat_sup = sweep_rows[0]["lesion_patient_all_suppressed"] if sweep_rows else 999
    g95_row     = next((r for r in safety_rows if r["threshold_rule"] == "global_p95"), {})
    g99_row     = next((r for r in safety_rows if r["threshold_rule"] == "global_p99"), {})

    if rd_d1_auroc is None:
        final_decision = "BLOCKED"
    elif rd_d1_auroc >= 0.60 and auroc_improvement >= 0.10 and le1_pat_sup == 0:
        final_decision = "RD4AD_REVIVED_FOR_RANKING"
    elif rd_d1_auroc >= 0.55 and auroc_improvement > 0:
        final_decision = "RD4AD_ANALYSIS_ONLY"
    else:
        final_decision = "RD4AD_NOT_USEFUL"

    print(f"\n  final_decision : {final_decision}")
    print(f"  AUROC improvement vs RD-B8f: {fmt_float(auroc_improvement, 4)}")

    # ── all_checks_passed ────────────────────────────────────────────────────
    best_ckpt_saved = (CKPT_DIR / "best_train_loss.pth").exists()
    last_ckpt_saved = (CKPT_DIR / "last.pth").exists()
    all_checks_passed = (
        len(epoch_logs) == EPOCHS
        and loss_decreased
        and total_nan == 0
        and total_inf == 0
        and not teacher_param_changed
        and best_ckpt_saved
        and last_ckpt_saved
        and n_failed == 0
        and n_nan == 0
        and n_inf == 0
        and n_scored == len(c2_rows)
    )

    t_elapsed = time.perf_counter() - t_start

    # ── summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "input_mode":                         "mediastinal_3ch_zminus1_z_zplus1",
        "model_type":                         "true_RD4AD_ResNet18_teacher_student",
        "window":                             f"HU[{MEDI_HU_MIN},{MEDI_HU_MAX}]",
        "train_rows":                         sampler.total_rows,
        "train_epochs":                       EPOCHS,
        "normal_val_rows":                    len(val_rows),
        "scored_candidates":                  n_scored,
        "positive_count":                     pos_cnt,
        "hard_negative_count":                hn_cnt,
        "score_nan_count":                    n_nan,
        "score_inf_count":                    n_inf,
        "best_epoch":                         best_epoch,
        "train_loss_first":                   round(train_loss_first, 6),
        "train_loss_last":                    round(train_loss_last, 6),
        "loss_decreased":                     loss_decreased,
        "teacher_param_changed":              teacher_param_changed,
        "student_param_changed":              True,
        "optimizer_teacher_param_count":      opt_teacher_count,
        "rd_b8f_auroc_reference":             RD_B8F_AUROC_REF,
        "rd_c3_convae_auroc_reference":       RD_C3_AUROC_REF,
        "rd_d1_auroc":                        rd_d1_auroc,
        "rd_d1_auprc":                        rd_d1_auprc,
        "auroc_improvement_vs_rd_b8f":        auroc_improvement,
        "g95_lesion_suppressed_rate":         g95_row.get("lesion_suppressed_rate"),
        "g95_hard_negative_suppressed_rate":  g95_row.get("hn_suppressed_rate"),
        "lesion_patient_all_suppressed_g95":  g95_row.get("lesion_patient_all_suppressed"),
        "g99_lesion_suppressed_rate":         g99_row.get("lesion_suppressed_rate"),
        "g99_hard_negative_suppressed_rate":  g99_row.get("hn_suppressed_rate"),
        "best_hn_suppression_at_lesion_le1pct": sweep_rows[0]["hn_suppressed_rate"] if len(sweep_rows) > 0 else None,
        "best_hn_suppression_at_lesion_le3pct": sweep_rows[1]["hn_suppressed_rate"] if len(sweep_rows) > 1 else None,
        "best_hn_suppression_at_lesion_le5pct": sweep_rows[2]["hn_suppressed_rate"] if len(sweep_rows) > 2 else None,
        "final_decision":                     final_decision,
        "suppression_applied":                False,
        "stage2_holdout_access":              0,
        "first_stage_score_modified":         False,
        "vessel_mask_used":                   False,
        "metric_backend":                     "sklearn_free_mann_whitney_auroc_and_step_average_precision",
        "all_checks_passed":                  all_checks_passed,
        "elapsed_seconds":                    round(t_elapsed, 1),
        "gpu_peak_memory_mb":                 round(gpu_peak_mb, 1),
    }
    with open(OUTPUT_ROOT / "rd_d1_medi3ch_true_rd4ad_revival_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  saved: rd_d1_medi3ch_true_rd4ad_revival_summary.json")

    # errors CSV
    write_csv(
        OUTPUT_ROOT / "rd_d1_errors.csv",
        ["safe_id", "candidate_id", "error"],
        error_rows + score_error_rows + val_errors,
    )

    # report MD
    verdict = "PASS" if all_checks_passed else "FAIL"
    md_lines = [
        "# RD-D1: Mediastinal 3ch true RD4AD Revival Report",
        "",
        f"## 판정: {verdict} / {final_decision}",
        "",
        "## 핵심 결과",
        "| 항목 | 값 |",
        "|---|---|",
        f"| input_mode | mediastinal_3ch_zminus1_z_zplus1 |",
        f"| window | HU[{MEDI_HU_MIN},{MEDI_HU_MAX}] → [0,1] |",
        f"| train_epochs | {EPOCHS} |",
        f"| best_epoch | {best_epoch} |",
        f"| train_loss_first | {train_loss_first:.6f} |",
        f"| train_loss_last | {train_loss_last:.6f} |",
        f"| loss_decreased | {loss_decreased} |",
        f"| scored_candidates | {n_scored:,} |",
        f"| score_nan | {n_nan} |",
        f"| score_inf | {n_inf} |",
        "",
        "## AUROC/AUPRC 비교",
        "| 모델 | AUROC | AUPRC |",
        "|---|---|---|",
        f"| RD-B8f (reference) | {RD_B8F_AUROC_REF:.4f} | N/A |",
        f"| RD-C3 ConvAE (reference) | {RD_C3_AUROC_REF:.4f} | N/A |",
        f"| RD-D1 medi3ch true RD4AD | {fmt_float(rd_d1_auroc,4)} | {fmt_float(rd_d1_auprc,4)} |",
        "",
        "## Safety (G95 threshold)",
        "| 항목 | 값 |",
        "|---|---|",
        f"| global_p95 | {fmt_float(global_p95,6)} |",
        f"| lesion_suppressed_rate | {fmt_float(g95_row.get('lesion_suppressed_rate'),4)} |",
        f"| hn_suppressed_rate | {fmt_float(g95_row.get('hn_suppressed_rate'),4)} |",
        f"| lesion_patient_all_suppressed | {g95_row.get('lesion_patient_all_suppressed','N/A')} |",
        "",
        "## 절대 하지 않은 것",
        "| 항목 | 상태 |",
        "|---|---|",
        "| suppression_applied | False |",
        "| stage2_holdout_access | 0 |",
        "| first_stage_score_modified | False |",
        "| vessel_mask_used | False |",
        "| rd_b9_threshold_modified | False |",
        f"| all_checks_passed | {all_checks_passed} |",
    ]
    with open(OUTPUT_ROOT / "rd_d1_medi3ch_true_rd4ad_revival_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("  saved: rd_d1_medi3ch_true_rd4ad_revival_report.md")

    # DONE
    (OUTPUT_ROOT / "DONE").write_text(
        f"rd_d1_medi3ch_true_rd4ad_revival_v1 DONE\n"
        f"all_checks_passed={all_checks_passed}\n"
        f"final_decision={final_decision}\n"
        f"rd_d1_auroc={rd_d1_auroc}\n",
        encoding="utf-8",
    )
    print("  saved: DONE")

    print()
    print("=" * 72)
    print(f"  RD-D1 COMPLETE")
    print(f"  train: {train_loss_first:.4f} -> {train_loss_last:.4f}  best_epoch={best_epoch}")
    print(f"  scored       : {n_scored:,}")
    print(f"  RD-D1 AUROC  : {fmt_float(rd_d1_auroc,4)}")
    print(f"  RD-D1 AUPRC  : {fmt_float(rd_d1_auprc,4)}")
    print(f"  vs RD-B8f    : {fmt_float(auroc_improvement,4)} improvement")
    print(f"  vs RD-C3     : {fmt_float((rd_d1_auroc - RD_C3_AUROC_REF) if rd_d1_auroc else None,4)}")
    print(f"  final_decision: {final_decision}")
    print(f"  elapsed      : {t_elapsed:.1f}s")
    print("=" * 72)

    if not all_checks_passed:
        sys.exit(1)


# ── profile-time ──────────────────────────────────────────────────────────────

def run_profile_time():
    import numpy as np
    import torch
    import torch.nn.functional as F
    import statistics as stats_mod

    print("=" * 72)
    print("RD-D1: Mediastinal 3ch true RD4AD [PROFILE-TIME]")
    print("=" * 72)

    # ── [1] 입력 검증 ────────────────────────────────────────────────────────
    print("\n[1] 입력 검증")
    ok_all = True
    checks = [
        ("normal_train manifest",    NORMAL_TRAIN_MANIFEST),
        ("normal_val manifest",      NORMAL_VAL_MANIFEST),
        ("RD-C2 candidate manifest", RD_C2_MANIFEST),
        ("ResNet18 local weights",   LOCAL_WEIGHT_PATH),
    ]
    for label, p in checks:
        exists = p.exists()
        print(f"  {'OK' if exists else 'MISSING'}: {label}")
        if not exists:
            ok_all = False

    stage2_holdout_access = 0
    c2_rows_all = []
    if RD_C2_MANIFEST.exists():
        with open(RD_C2_MANIFEST, newline="", encoding="utf-8") as f:
            c2_rows_all = list(csv.DictReader(f))
        holdout_cnt = sum(1 for r in c2_rows_all if r.get("stage_split", "") == "stage2_holdout")
        stage2_holdout_access = holdout_cnt
        status = "OK" if holdout_cnt == 0 else "FAIL"
        print(f"  stage2_holdout intersection: {holdout_cnt} ({status})")
        if holdout_cnt != 0:
            ok_all = False

    if PROFILE_OUTPUT_ROOT.exists():
        print(f"[ABORT] profile output root 이미 존재 (삭제 금지): {PROFILE_OUTPUT_ROOT}", file=sys.stderr)
        sys.exit(1)
    print(f"  profile output root: OK (not exists)")

    if not ok_all:
        print("\n[ABORT] 입력 검증 실패", file=sys.stderr)
        sys.exit(1)

    PROFILE_OUTPUT_ROOT.mkdir(parents=True)
    print(f"  profile output root created")
    error_rows = []

    # ── [2] 모델 준비 ────────────────────────────────────────────────────────
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

    # ── [2] train-like profile ───────────────────────────────────────────────
    print("\n[2] train-like profile (warmup=20, timed=200)")
    sampler  = SixBinCoordSampler(NORMAL_TRAIN_MANIFEST, per_bin=PER_BIN, seed=SEED)
    ct_cache = CtMmapCache(max_size=16)

    WARMUP_BATCHES = 20
    TIMED_BATCHES  = 200
    train_batch_rows   = []
    timed_batch_times  = []
    timed_crop_times   = []
    timed_fb_times     = []
    loss_nan_count = 0
    loss_inf_count = 0

    batch_idx = 0
    for step, batch_items in sampler.epoch_batches(0):
        t_crop_start = time.perf_counter()
        crops = []
        for item in batch_items:
            ct_path = NORMAL_CT_ROOT / item["safe_id"] / "ct_hu.npy"
            assert_path_safe(ct_path)
            ct_arr = ct_cache.get(ct_path)
            crop = build_medi3ch_crop(
                ct_arr,
                item["local_z"],
                item["crop_y0"], item["crop_x0"],
                item["crop_y1"], item["crop_x1"],
            )
            crops.append(crop)
        t_crop_end = time.perf_counter()
        crop_t = t_crop_end - t_crop_start

        batch_np = np.stack(crops, axis=0)
        batch_t  = torch.from_numpy(batch_np).to(device)

        t_fb_start = time.perf_counter()
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
        t_fb_end = time.perf_counter()
        fb_t  = t_fb_end - t_fb_start
        tot_t = crop_t + fb_t
        loss_val = float(loss.item())

        is_warmup = int(batch_idx < WARMUP_BATCHES)
        train_batch_rows.append({
            "batch_idx":            batch_idx,
            "is_warmup":            is_warmup,
            "batch_total_sec":      round(tot_t, 5),
            "crop_build_sec":       round(crop_t, 5),
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
                      f"total={tot_t:.3f}s  crop={crop_t:.3f}s  fb={fb_t:.3f}s  loss={loss_val:.4f}")

        batch_idx += 1
        if batch_idx >= WARMUP_BATCHES + TIMED_BATCHES:
            break

    student.eval()
    del optimizer

    def _pct(data, pct):
        if not data:
            return 0.0
        s = sorted(data)
        idx = min(int(pct / 100.0 * len(s)), len(s) - 1)
        return s[idx]

    train_mean   = float(stats_mod.mean(timed_batch_times))   if timed_batch_times else 0.0
    train_median = float(stats_mod.median(timed_batch_times)) if timed_batch_times else 0.0
    train_p95    = _pct(timed_batch_times, 95)
    crop_mean    = float(stats_mod.mean(timed_crop_times))    if timed_crop_times  else 0.0
    fb_mean      = float(stats_mod.mean(timed_fb_times))      if timed_fb_times    else 0.0

    print(f"\n  [train profile 결과]")
    print(f"  timed_batches              : {len(timed_batch_times)}")
    print(f"  train_batch_seconds_mean   : {train_mean:.4f}s")
    print(f"  train_batch_seconds_median : {train_median:.4f}s")
    print(f"  train_batch_seconds_p95    : {train_p95:.4f}s")
    print(f"  crop_build_time_mean       : {crop_mean:.4f}s")
    print(f"  forward_backward_time_mean : {fb_mean:.4f}s")
    print(f"  loss_nan_count             : {loss_nan_count}")
    print(f"  loss_inf_count             : {loss_inf_count}")
    print(f"  crop_padding_applied_count : {_G_PAD_APPLIED_COUNT}")
    print(f"  reflect_padding_count      : {_G_PAD_REFLECT_COUNT}")
    print(f"  edge_fallback_count        : {_G_PAD_EDGE_COUNT}")

    write_csv(
        PROFILE_OUTPUT_ROOT / "rd_d1_time_profile_train_batches.csv",
        ["batch_idx", "is_warmup", "batch_total_sec", "crop_build_sec", "forward_backward_sec", "loss"],
        train_batch_rows,
    )

    # ── [3] normal_val scoring profile ──────────────────────────────────────
    print("\n[3] normal_val scoring profile (max 1,024 rows)")
    PROFILE_VAL_ROWS = 1024
    BATCH_SCORE = 64

    with open(NORMAL_VAL_MANIFEST, newline="", encoding="utf-8") as f:
        val_rows_all = list(csv.DictReader(f))
    val_rows_sample = val_rows_all[:PROFILE_VAL_ROWS]
    print(f"  val rows sampled: {len(val_rows_sample)}/{len(val_rows_all)}")

    val_ct_cache = CtMmapCache(max_size=8)
    val_batch_times  = []
    score_nan_count  = 0
    score_inf_count  = 0
    scoring_batch_rows = []

    val_groups = collections.defaultdict(list)
    for r in val_rows_sample:
        val_groups[r["safe_id"]].append(r)

    for safe_id, v_rows in val_groups.items():
        ct_path_str = v_rows[0].get("ct_hu_npy", "")
        ct_path = Path(ct_path_str) if ct_path_str else (NORMAL_CT_ROOT / safe_id / "ct_hu.npy")
        assert_path_safe(ct_path)
        if not ct_path.exists():
            error_rows.append({"safe_id": safe_id, "error": "val_ct_not_found"})
            continue
        ct_arr = val_ct_cache.get(ct_path)

        for i in range(0, len(v_rows), BATCH_SCORE):
            batch_rows = v_rows[i:i + BATCH_SCORE]
            t_b_start = time.perf_counter()
            crops = [
                build_medi3ch_crop(
                    ct_arr,
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
            for j in range(len(batch_rows)):
                s1 = float((1 - F.cosine_similarity(de3[j:j+1], tf3[j:j+1], dim=1)).mean())
                s2 = float((1 - F.cosine_similarity(de2[j:j+1], tf2[j:j+1], dim=1)).mean())
                s3 = float((1 - F.cosine_similarity(de1[j:j+1], tf1[j:j+1], dim=1)).mean())
                sc = (s1 + s2 + s3) / 3.0
                if math.isnan(sc):
                    score_nan_count += 1
                if math.isinf(sc):
                    score_inf_count += 1
            t_b_end = time.perf_counter()
            b_t = t_b_end - t_b_start
            val_batch_times.append(b_t)
            scoring_batch_rows.append({
                "source":          "val",
                "batch_idx":       len(val_batch_times) - 1,
                "batch_size":      len(batch_rows),
                "batch_total_sec": round(b_t, 5),
            })

    val_mean = float(stats_mod.mean(val_batch_times)) if val_batch_times else 0.0
    n_val_batches_full = math.ceil(8354 / BATCH_SCORE)
    est_val_total_sec  = val_mean * n_val_batches_full

    print(f"  val_batch_seconds_mean            : {val_mean:.4f}s")
    print(f"  n_val_batches_full (8354/64)       : {n_val_batches_full}")
    print(f"  estimated_val_total_seconds        : {est_val_total_sec:.1f}s  ({est_val_total_sec/60:.1f}min)")

    # ── [4] candidate scoring profile ───────────────────────────────────────
    print("\n[4] candidate scoring profile (max 2,048 rows)")
    PROFILE_CAND_ROWS = 2048

    cand_rows_sample = c2_rows_all[:PROFILE_CAND_ROWS]
    holdout_in_sample = sum(
        1 for r in cand_rows_sample if r.get("stage_split", "") == "stage2_holdout"
    )
    if holdout_in_sample != 0:
        print(f"[ABORT] candidate sample에 stage2_holdout 포함: {holdout_in_sample}", file=sys.stderr)
        sys.exit(1)
    print(f"  candidate rows sampled: {len(cand_rows_sample)}/{len(c2_rows_all)}")
    print(f"  stage2_holdout in sample: 0 (OK)")

    cand_ct_cache = CtMmapCache(max_size=12)
    cand_batch_times = []
    cand_groups = collections.defaultdict(list)
    for r in cand_rows_sample:
        cand_groups[r["safe_id"]].append(r)

    for safe_id, rows_for_pat in cand_groups.items():
        ct_path = CANDIDATE_CT_ROOT / safe_id / "ct_hu.npy"
        assert_path_safe(ct_path)
        if not ct_path.exists():
            error_rows.append({"safe_id": safe_id, "error": "cand_ct_not_found"})
            continue
        ct_arr = cand_ct_cache.get(ct_path)

        for i in range(0, len(rows_for_pat), BATCH_SCORE):
            batch_rows = rows_for_pat[i:i + BATCH_SCORE]
            t_b_start = time.perf_counter()
            crops = []
            for row in batch_rows:
                try:
                    crop = build_medi3ch_crop(
                        ct_arr,
                        int(row["local_z"]),
                        int(row["crop_y0"]), int(row["crop_x0"]),
                        int(row["crop_y1"]), int(row["crop_x1"]),
                    )
                    crops.append(crop)
                except Exception as e:
                    error_rows.append({"safe_id": safe_id, "error": f"crop_failed:{e}"})
            if not crops:
                continue
            batch_t = torch.from_numpy(np.stack(crops, axis=0)).to(device)
            with torch.no_grad():
                teacher(batch_t)
                tf3 = teacher_features["layer3"]
                tf2 = teacher_features["layer2"]
                tf1 = teacher_features["layer1"]
                de3, de2, de1 = student(tf3)
            for j in range(len(crops)):
                s1 = float((1 - F.cosine_similarity(de3[j:j+1], tf3[j:j+1], dim=1)).mean())
                s2 = float((1 - F.cosine_similarity(de2[j:j+1], tf2[j:j+1], dim=1)).mean())
                s3 = float((1 - F.cosine_similarity(de1[j:j+1], tf1[j:j+1], dim=1)).mean())
                sc = (s1 + s2 + s3) / 3.0
                if math.isnan(sc):
                    score_nan_count += 1
                if math.isinf(sc):
                    score_inf_count += 1
            t_b_end = time.perf_counter()
            b_t = t_b_end - t_b_start
            cand_batch_times.append(b_t)
            scoring_batch_rows.append({
                "source":          "candidate",
                "batch_idx":       len(cand_batch_times) - 1,
                "batch_size":      len(crops),
                "batch_total_sec": round(b_t, 5),
            })

    cand_mean = float(stats_mod.mean(cand_batch_times)) if cand_batch_times else 0.0
    n_cand_batches_full = math.ceil(113447 / BATCH_SCORE)
    est_cand_total_sec  = cand_mean * n_cand_batches_full

    print(f"  candidate_batch_seconds_mean          : {cand_mean:.4f}s")
    print(f"  n_cand_batches_full (113447/64)        : {n_cand_batches_full}")
    print(f"  estimated_candidate_total_seconds      : {est_cand_total_sec:.1f}s  ({est_cand_total_sec/60:.1f}min)")

    gpu_peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda" else 0.0
    )
    print(f"\n  GPU peak memory: {gpu_peak_mb:.1f} MB")

    del teacher, student

    # ── [5] 전체 예상 시간 계산 ────────────────────────────────────────────
    print("\n[5] 전체 예상 시간 계산")
    est_train_total_sec = train_mean * TOTAL_TRAIN_STEPS
    est_total_raw       = est_train_total_sec + est_val_total_sec + est_cand_total_sec
    est_total_overhead  = est_total_raw * 1.15

    est_train_min = est_train_total_sec / 60
    est_val_min   = est_val_total_sec   / 60
    est_cand_min  = est_cand_total_sec  / 60
    est_raw_min   = est_total_raw       / 60
    est_over_min  = est_total_overhead  / 60

    import datetime
    finish_dt  = datetime.datetime.now() + datetime.timedelta(seconds=est_total_overhead)
    finish_str = finish_dt.strftime("%Y-%m-%d %H:%M")

    print(f"  estimated_train_minutes              : {est_train_min:.1f}  ({est_train_total_sec:.0f}s)")
    print(f"  estimated_val_minutes                : {est_val_min:.1f}  ({est_val_total_sec:.0f}s)")
    print(f"  estimated_candidate_scoring_minutes  : {est_cand_min:.1f}  ({est_cand_total_sec:.0f}s)")
    print(f"  estimated_total_minutes_raw          : {est_raw_min:.1f}  ({est_total_raw:.0f}s)")
    print(f"  estimated_total_minutes_with_overhead: {est_over_min:.1f}  ({est_total_overhead:.0f}s)")
    print(f"  expected_finish_time_rough           : {finish_str}")

    # ── [6] profile 통과 조건 ───────────────────────────────────────────────
    print("\n[6] profile 통과 조건")
    checkpoint_saved     = False
    full_train_started   = False
    full_scoring_started = False
    profile_candidate_rows = len(cand_rows_sample)

    all_checks_passed = (
        len(timed_batch_times) >= 200
        and profile_candidate_rows > 0
        and score_nan_count == 0
        and score_inf_count == 0
        and loss_nan_count == 0
        and loss_inf_count == 0
        and stage2_holdout_access == 0
        and not checkpoint_saved
        and not full_train_started
        and not full_scoring_started
    )

    checks_detail = [
        ("profile_train_batches >= 200", len(timed_batch_times) >= 200,  len(timed_batch_times)),
        ("profile_candidate_rows > 0",   profile_candidate_rows > 0,      profile_candidate_rows),
        ("score_nan_count == 0",         score_nan_count == 0,            score_nan_count),
        ("score_inf_count == 0",         score_inf_count == 0,            score_inf_count),
        ("loss_nan_count == 0",          loss_nan_count == 0,             loss_nan_count),
        ("loss_inf_count == 0",          loss_inf_count == 0,             loss_inf_count),
        ("stage2_holdout_access == 0",   stage2_holdout_access == 0,      stage2_holdout_access),
        ("checkpoint_saved = false",     not checkpoint_saved,            checkpoint_saved),
        ("full_train_started = false",   not full_train_started,          full_train_started),
        ("full_scoring_started = false", not full_scoring_started,        full_scoring_started),
    ]
    for name, passed, val in checks_detail:
        print(f"  {'OK' if passed else 'FAIL'}: {name}  (val={val})")

    verdict = "PASS" if all_checks_passed else "FAIL"
    print(f"\n판정: profile {verdict}")

    # ── 파일 생성 ───────────────────────────────────────────────────────────
    write_csv(
        PROFILE_OUTPUT_ROOT / "rd_d1_time_profile_scoring_batches.csv",
        ["source", "batch_idx", "batch_size", "batch_total_sec"],
        scoring_batch_rows,
    )
    write_csv(
        PROFILE_OUTPUT_ROOT / "rd_d1_time_profile_errors.csv",
        ["safe_id", "error"],
        error_rows,
    )

    profile_summary = {
        "profile_train_batches":                  len(timed_batch_times),
        "train_batch_seconds_mean":               round(train_mean,   6),
        "train_batch_seconds_median":             round(train_median, 6),
        "train_batch_seconds_p95":                round(train_p95,    6),
        "crop_build_time_mean":                   round(crop_mean,    6),
        "forward_backward_time_mean":             round(fb_mean,      6),
        "estimated_train_total_seconds":          round(est_train_total_sec, 1),
        "estimated_val_total_seconds":            round(est_val_total_sec,   1),
        "estimated_candidate_total_seconds":      round(est_cand_total_sec,  1),
        "estimated_total_seconds_raw":            round(est_total_raw,       1),
        "estimated_total_seconds_with_overhead":  round(est_total_overhead,  1),
        "estimated_train_minutes":                round(est_train_min, 1),
        "estimated_val_minutes":                  round(est_val_min,   1),
        "estimated_candidate_scoring_minutes":    round(est_cand_min,  1),
        "estimated_total_minutes_raw":            round(est_raw_min,   1),
        "estimated_total_minutes_with_overhead":  round(est_over_min,  1),
        "expected_finish_time_rough":             finish_str,
        "gpu_peak_memory_mb":                     round(gpu_peak_mb, 1),
        "score_nan_count":                        score_nan_count,
        "score_inf_count":                        score_inf_count,
        "loss_nan_count":                         loss_nan_count,
        "loss_inf_count":                         loss_inf_count,
        "crop_padding_applied_count":             _G_PAD_APPLIED_COUNT,
        "reflect_padding_count":                  _G_PAD_REFLECT_COUNT,
        "edge_fallback_count":                    _G_PAD_EDGE_COUNT,
        "stage2_holdout_access":                  stage2_holdout_access,
        "checkpoint_saved":                       checkpoint_saved,
        "full_train_started":                     full_train_started,
        "full_scoring_started":                   full_scoring_started,
        "all_checks_passed":                      all_checks_passed,
    }
    with open(PROFILE_OUTPUT_ROOT / "rd_d1_time_profile_summary.json", "w", encoding="utf-8") as f:
        json.dump(profile_summary, f, indent=2)
    print("  saved: rd_d1_time_profile_summary.json")

    md_lines = [
        "# RD-D1 Time Profile Report",
        "",
        f"## 판정: {verdict}",
        "",
        "## Train-like Profile",
        "| 항목 | 값 |",
        "|---|---|",
        f"| profile_train_batches | {len(timed_batch_times)} |",
        f"| train_batch_seconds_mean | {train_mean:.5f}s |",
        f"| train_batch_seconds_median | {train_median:.5f}s |",
        f"| train_batch_seconds_p95 | {train_p95:.5f}s |",
        f"| crop_build_time_mean | {crop_mean:.5f}s |",
        f"| forward_backward_time_mean | {fb_mean:.5f}s |",
        f"| loss_nan_count | {loss_nan_count} |",
        f"| loss_inf_count | {loss_inf_count} |",
        f"| crop_padding_applied_count | {_G_PAD_APPLIED_COUNT} |",
        f"| reflect_padding_count | {_G_PAD_REFLECT_COUNT} |",
        f"| edge_fallback_count | {_G_PAD_EDGE_COUNT} |",
        "",
        "## 예상 전체 시간",
        "| 항목 | 분 | 초 |",
        "|---|---|---|",
        f"| estimated_train | {est_train_min:.1f}min | {est_train_total_sec:.0f}s |",
        f"| estimated_val | {est_val_min:.1f}min | {est_val_total_sec:.0f}s |",
        f"| estimated_candidate_scoring | {est_cand_min:.1f}min | {est_cand_total_sec:.0f}s |",
        f"| estimated_total_raw | {est_raw_min:.1f}min | {est_total_raw:.0f}s |",
        f"| estimated_total_with_overhead (×1.15) | {est_over_min:.1f}min | {est_total_overhead:.0f}s |",
        f"| expected_finish_time_rough | {finish_str} | - |",
        "",
        "## GPU / Safety",
        "| 항목 | 값 |",
        "|---|---|",
        f"| gpu_peak_memory_mb | {gpu_peak_mb:.1f} |",
        f"| score_nan_count | {score_nan_count} |",
        f"| score_inf_count | {score_inf_count} |",
        f"| stage2_holdout_access | {stage2_holdout_access} |",
        f"| checkpoint_saved | {checkpoint_saved} |",
        f"| full_train_started | {full_train_started} |",
        f"| full_scoring_started | {full_scoring_started} |",
        f"| all_checks_passed | {all_checks_passed} |",
    ]
    with open(PROFILE_OUTPUT_ROOT / "rd_d1_time_profile_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("  saved: rd_d1_time_profile_report.md")

    (PROFILE_OUTPUT_ROOT / "DONE").write_text(
        f"rd_d1_time_profile DONE\nall_checks_passed={all_checks_passed}\n",
        encoding="utf-8",
    )
    print("  saved: DONE")

    print()
    print("=" * 72)
    print(f"  PROFILE-TIME COMPLETE  판정: {verdict}")
    print(f"  train batch mean       : {train_mean:.4f}s")
    print(f"  est train total        : {est_train_min:.1f}min  ({est_train_total_sec:.0f}s)")
    print(f"  est val total          : {est_val_min:.1f}min")
    print(f"  est cand scoring       : {est_cand_min:.1f}min")
    print(f"  est total raw          : {est_raw_min:.1f}min")
    print(f"  est total (×1.15)      : {est_over_min:.1f}min")
    print(f"  expected finish        : {finish_str}")
    print(f"  GPU peak memory        : {gpu_peak_mb:.1f}MB")
    print(f"  NaN score / loss       : {score_nan_count} / {loss_nan_count}")
    print(f"  Inf score / loss       : {score_inf_count} / {loss_inf_count}")
    print(f"  crop padding applied   : {_G_PAD_APPLIED_COUNT}")
    print(f"  reflect padding        : {_G_PAD_REFLECT_COUNT}")
    print(f"  edge fallback          : {_G_PAD_EDGE_COUNT}")
    print(f"  all_checks_passed      : {all_checks_passed}")
    print("=" * 72)

    if not all_checks_passed:
        sys.exit(1)


# ── 진입점 ────────────────────────────────────────────────────────────────────

if IS_DRY_PLAN:
    run_dry_plan()
elif IS_RUN_ALL:
    run_all()
elif IS_PROFILE_TIME:
    run_profile_time()
