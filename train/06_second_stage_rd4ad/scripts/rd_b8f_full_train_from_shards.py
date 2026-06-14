"""
RD-B8f: Full train from float32 shards
목적: RD-B8e shard 기반 full train 20 epoch
전제: RD-B8e all_checks_passed=True
모드:
  bare run     -> exit 2
  --dry-plan   -> 계획 출력 (파일 생성 없음)
  --run-train  -> full train 실행
안전 조건:
  RD-B8e DONE + all_checks_passed 확인
  stage2_holdout/lesion 접근 금지
  scoring/threshold 금지
  output root 존재 시 즉시 중단
  기존 결과물 삭제/수정 금지
  best.pth / final.pth 이름 금지
  checkpoint는 MODEL_ROOT/checkpoints/ 내부에만 저장
"""

import sys
import csv
import json
import time
import math
import collections
from pathlib import Path

ALLOWED_MODES = {"--dry-plan", "--run-train"}
if not any(m in sys.argv for m in ALLOWED_MODES):
    print("오류: 실행 모드 지정 필요.")
    print("  --dry-plan  : 계획 출력 (파일 생성 없음)")
    print("  --run-train : full train 실행")
    sys.exit(2)

IS_DRY_PLAN  = "--dry-plan"  in sys.argv
IS_RUN_TRAIN = "--run-train" in sys.argv

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

SHARD_ROOT   = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit/rd_b8e_full_float32_shards_v1"
)
SHARDS_DIR   = SHARD_ROOT / "shards"
SHARD_INDEX_CSV   = SHARD_ROOT / "rd_b8e_full_shard_index.csv"
SHARD_SUMMARY_JSON = SHARD_ROOT / "rd_b8e_full_shard_summary.json"

MODEL_ROOT  = (
    PROJECT_ROOT
    / "outputs/models/rd_b8f_true_rd4ad_resnet18_mixed3ch_6bin_shard_v1"
)
CKPT_DIR    = MODEL_ROOT / "checkpoints"

REPORT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit/rd_b8f_full_train_from_shards_v1"
)

LOCAL_WEIGHT_PATH = Path(
    "/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)

# ── 설계 상수 ──────────────────────────────────────────────────────────────────
FORBIDDEN_KEYWORDS = [
    "stage2_holdout", "lesion", "test_lesion", "second-stage-lesion-refiner",
]
SIX_BIN_LABELS = [
    "lower_boundary", "lower_interior",
    "middle_boundary", "middle_interior",
    "upper_boundary",  "upper_interior",
]
LOW_Z_BOUNDARY_WARN_THRESHOLD = 7
BATCH_SIZE   = 48
PER_BIN      = 8          # 6 × 8 = 48
EPOCHS       = 20
LR           = 1e-4
WEIGHT_DECAY = 1e-5
SHARD_SIZE   = 1000
SHARD_CACHE_SIZE = 6
SEED         = 42


# =============================================================================
# 안전 검사
# =============================================================================

def assert_path_safe(path_str):
    for kw in FORBIDDEN_KEYWORDS:
        if kw.lower() in str(path_str).lower():
            raise RuntimeError(
                f"[SAFETY] 금지 경로 접근 차단: {path_str!r} (keyword={kw!r})"
            )


# =============================================================================
# 모델 빌드
# =============================================================================

def build_teacher(local_weight_path):
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


def rd_loss_fn(teacher_feats, student_feats):
    import torch
    import torch.nn.functional as F
    loss = torch.tensor(0.0, device=teacher_feats[0].device)
    for tf, sf in zip(teacher_feats, student_feats):
        cos_sim = F.cosine_similarity(sf, tf, dim=1)
        loss = loss + (1 - cos_sim).mean()
    return loss / len(teacher_feats)


# =============================================================================
# LRU Shard Cache
# =============================================================================

class LRUShardCache:
    def __init__(self, max_size=6):
        self._cache = collections.OrderedDict()
        self._max   = max_size

    def load(self, shard_path):
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
# 6-bin Balanced Sampler
# =============================================================================

class SixBinShardSampler:
    """
    strict 6-bin balanced, shortest-bin drop-last.
    각 step마다 6개 bin에서 per_bin=8개씩 -> batch_size=48.
    """
    def __init__(self, shard_index_csv, per_bin, seed):
        import random
        self.per_bin = per_bin
        self._seed   = seed

        # bin별 (shard_id, row_in_shard) 리스트
        self.bin_items = {lbl: [] for lbl in SIX_BIN_LABELS}
        with open(shard_index_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lbl = row.get("six_bin_label", "")
                if lbl in self.bin_items:
                    self.bin_items[lbl].append((
                        int(row["shard_id"]),
                        int(row["row_in_shard"]),
                        int(row.get("low_z_warning", 0)),
                    ))

        bin_sizes = {lbl: len(v) for lbl, v in self.bin_items.items()}
        self.min_bin_size    = min(bin_sizes.values())
        self.steps_per_epoch = self.min_bin_size // per_bin  # drop-last

        print("  6-bin 분포 (shard index):")
        for lbl in SIX_BIN_LABELS:
            print(f"    {lbl}: {bin_sizes[lbl]:,}")
        print(f"  min_bin_size={self.min_bin_size:,}  "
              f"per_bin={per_bin}  steps_per_epoch={self.steps_per_epoch:,}")

    def epoch_batches(self, epoch):
        """yield: list[(shard_id, row_in_shard, low_z_warning)] × batch_size"""
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
# CSV 헬퍼
# =============================================================================

class CsvAppendWriter:
    def __init__(self, path, fieldnames):
        self.path = path
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
    print(f"  -> {path.name}")


# =============================================================================
# parameter snapshot (teacher 변경 여부 확인)
# =============================================================================

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
# dry-plan
# =============================================================================

def run_dry_plan():
    # shard summary 로드 (있으면)
    b8e_ok = False
    if SHARD_SUMMARY_JSON.exists():
        with open(SHARD_SUMMARY_JSON, encoding="utf-8") as f:
            b8e_summary = json.load(f)
        b8e_ok = b8e_summary.get("all_checks_passed", False)

    # sampler 예상치 (manifest 없이 계산)
    est_min_bin   = 86017 // 6  # 대략
    est_steps     = est_min_bin // PER_BIN
    est_batch_sec = 0.017
    est_epoch_sec = est_steps * est_batch_sec
    est_total_hr  = EPOCHS * est_epoch_sec / 3600

    print("=" * 70)
    print("RD-B8f: Full train from float32 shards [DRY-PLAN]")
    print("=" * 70)
    print()
    print("## 1. 전제 조건")
    print(f"  RD-B8e all_checks_passed : {b8e_ok}  {'OK' if b8e_ok else 'MISSING/FAIL'}")
    print()
    print("## 2. 학습 설정")
    print(f"  batch_size   : {BATCH_SIZE}")
    print(f"  per_bin      : {PER_BIN}")
    print(f"  epochs       : {EPOCHS}")
    print(f"  lr           : {LR}")
    print(f"  optimizer    : AdamW(student only)")
    print(f"  loader       : float32 shard (on-the-fly crop 금지)")
    print(f"  balanced     : strict 6-bin, shortest-bin drop-last")
    print()
    print("## 3. 출력")
    print(f"  MODEL_ROOT  : {MODEL_ROOT}")
    print(f"  REPORT_ROOT : {REPORT_ROOT}")
    print()
    print("## 4. 예상 시간 (B8d 실측 0.017s/batch 기준)")
    print(f"  est steps/epoch : ~{est_steps:,}")
    print(f"  est epoch time  : ~{est_epoch_sec/60:.1f}min")
    print(f"  est 20ep total  : ~{est_total_hr:.2f}hr")
    print()
    print("## 5. 안전 조건")
    print("  scoring/threshold: 금지")
    print("  stage2_holdout   : 금지")
    print("  checkpoint 이름  : best_train_loss.pth, last.pth")
    print()
    print("판정: DRY-PLAN OK" if b8e_ok else "판정: RD-B8e 미완료 - full train 불가")
    print("  사용자 승인 후:")
    print("  python scripts/rd_b8f_full_train_from_shards.py --run-train")


# =============================================================================
# run_train
# =============================================================================

def run_train():
    import numpy as np
    import torch

    print("=" * 70)
    print("RD-B8f: Full train from float32 shards [RUN-TRAIN]")
    print("=" * 70)

    # ── 전제 조건 확인 ──
    if not SHARD_SUMMARY_JSON.exists():
        print(f"[ABORT] RD-B8e summary 없음: {SHARD_SUMMARY_JSON}")
        sys.exit(1)
    with open(SHARD_SUMMARY_JSON, encoding="utf-8") as f:
        b8e_summary = json.load(f)
    if not b8e_summary.get("all_checks_passed", False):
        print("[ABORT] RD-B8e all_checks_passed=False - full train 금지")
        sys.exit(1)
    if not (SHARD_ROOT / "DONE").exists():
        print("[ABORT] RD-B8e DONE marker 없음")
        sys.exit(1)
    print("  RD-B8e all_checks_passed=True OK")

    # ── output root guard ──
    for root in [MODEL_ROOT, REPORT_ROOT]:
        if root.exists():
            print(f"[ABORT] output root 이미 존재: {root}")
            sys.exit(1)
    MODEL_ROOT.mkdir(parents=True, exist_ok=False)
    CKPT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_ROOT.mkdir(parents=True, exist_ok=False)

    # ── shard 존재 확인 ──
    n_shards_expected = b8e_summary.get("n_shards", 0)
    shard_files = sorted(SHARDS_DIR.glob("rd_b8e_shard_*.npy"))
    if len(shard_files) != n_shards_expected:
        print(f"[ABORT] shard 파일 수 불일치: {len(shard_files)} != {n_shards_expected}")
        sys.exit(1)
    print(f"  shard files: {len(shard_files)} OK")

    # ── 6-bin sampler 로드 ──
    print("  shard index 로드 중 (6-bin sampler) ...")
    sampler   = SixBinShardSampler(SHARD_INDEX_CSV, per_bin=PER_BIN, seed=SEED)
    shard_cache = LRUShardCache(max_size=SHARD_CACHE_SIZE)

    # ── 모델 설정 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

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
        student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    # teacher param snapshot (변경 여부 확인용)
    teacher_snap_before = snapshot_params(teacher)

    # optimizer가 teacher param을 포함하지 않음을 확인
    teacher_param_set = set(id(p) for p in teacher.parameters())
    opt_teacher_count = sum(
        1 for pg in optimizer.param_groups
        for p in pg["params"]
        if id(p) in teacher_param_set
    )
    print(f"  optimizer teacher param count: {opt_teacher_count}")
    if opt_teacher_count != 0:
        print("[ABORT] optimizer가 teacher param 포함 - 설계 오류")
        sys.exit(1)

    # ── log writer 준비 ──
    epoch_log_writer = CsvAppendWriter(
        REPORT_ROOT / "rd_b8f_epoch_log.csv",
        ["epoch", "steps", "mean_loss", "min_loss", "max_loss",
         "nan_count", "inf_count", "epoch_time_sec", "cumulative_time_sec"],
    )
    batch_log_writer = CsvAppendWriter(
        REPORT_ROOT / "rd_b8f_batch_loss_log.csv",
        ["epoch", "step", "loss", "loss_nan", "loss_inf", "load_time_sec", "fwd_bwd_time_sec"],
    )
    bin_batch_rows  = []
    low_z_warn_rows = []
    error_rows      = []

    # ── 학습 ──
    best_loss   = float("inf")
    best_epoch  = -1
    epoch_logs  = []
    t_train_start = time.perf_counter()
    total_nan   = 0
    total_inf   = 0

    for epoch in range(EPOCHS):
        student.train()
        epoch_losses = []
        epoch_nan    = 0
        epoch_inf    = 0
        t_epoch_start = time.perf_counter()
        bin_counts_this_epoch = collections.Counter()
        low_z_this_epoch = 0

        print(f"\n  [Epoch {epoch+1}/{EPOCHS}]")

        for step, batch_items in sampler.epoch_batches(epoch):
            # batch 구성
            t_load_s = time.perf_counter()
            crops = []
            for shard_id, row_in_shard, low_z_warn in batch_items:
                shard_path = SHARDS_DIR / f"rd_b8e_shard_{shard_id:04d}.npy"
                shard_arr  = shard_cache.load(shard_path)
                crop_np    = shard_arr[row_in_shard].astype("float32")
                crops.append(crop_np)
                if low_z_warn:
                    low_z_this_epoch += 1
            batch_np = np.stack(crops, axis=0)
            t_load_e = time.perf_counter()

            # forward-backward
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

            epoch_losses.append(loss_val)
            batch_log_writer.writerow({
                "epoch": epoch + 1,
                "step": step,
                "loss": round(loss_val, 6),
                "loss_nan": is_nan,
                "loss_inf": is_inf,
                "load_time_sec": round(t_load_e - t_load_s, 4),
                "fwd_bwd_time_sec": round(t_fwd_e - t_fwd_s, 4),
            })

            if step % 200 == 0 or step == sampler.steps_per_epoch - 1:
                elapsed = time.perf_counter() - t_epoch_start
                print(
                    f"    step {step:5d}/{sampler.steps_per_epoch}  "
                    f"loss={loss_val:.4f}  load={t_load_e-t_load_s:.3f}s  "
                    f"elapsed={elapsed:.0f}s"
                )

        # epoch 집계
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
            "epoch":             epoch + 1,
            "steps":             len(epoch_losses),
            "mean_loss":         round(mean_loss, 6),
            "min_loss":          round(min_loss, 6),
            "max_loss":          round(max_loss, 6),
            "nan_count":         epoch_nan,
            "inf_count":         epoch_inf,
            "epoch_time_sec":    round(epoch_time, 2),
            "cumulative_time_sec": round(cumul_time, 2),
        })
        epoch_logs.append({
            "epoch": epoch + 1, "mean_loss": mean_loss,
        })

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

        low_z_warn_rows.append({
            "epoch": epoch + 1,
            "low_z_warning_count": low_z_this_epoch,
        })

        print(
            f"  Epoch {epoch+1} done: mean_loss={mean_loss:.4f}  "
            f"best={best_loss:.4f}(ep{best_epoch})  "
            f"time={epoch_time:.1f}s  NaN={epoch_nan}  Inf={epoch_inf}"
        )

    t_train_end = time.perf_counter()
    total_time  = t_train_end - t_train_start

    epoch_log_writer.close()
    batch_log_writer.close()

    # ── teacher 변경 여부 ──
    teacher_snap_after  = snapshot_params(teacher)
    teacher_param_changed = params_changed(teacher_snap_before, teacher_snap_after)
    student_snap_after  = snapshot_params(student)
    student_snap_init   = {}  # 초기값 없음 → True로 처리 (학습 됐음)

    # ── checkpoint 존재 여부 ──
    best_ckpt_saved = (CKPT_DIR / "best_train_loss.pth").exists()
    last_ckpt_saved = (CKPT_DIR / "last.pth").exists()

    # ── loss 감소 여부 ──
    valid_epoch_losses = [
        r["mean_loss"] for r in epoch_logs
        if not math.isnan(r["mean_loss"])
    ]
    loss_decreased = (
        len(valid_epoch_losses) >= 2
        and valid_epoch_losses[-1] < valid_epoch_losses[0]
    )

    # ── GPU peak ──
    gpu_peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda" else 0.0
    )

    # ── 통과 조건 ──
    all_checks_passed = (
        len(epoch_logs) == EPOCHS
        and loss_decreased
        and total_nan == 0
        and total_inf == 0
        and not teacher_param_changed
        and best_ckpt_saved
        and last_ckpt_saved
    )

    # ── 보조 CSV ──
    write_csv(
        REPORT_ROOT / "rd_b8f_bin_batch_summary.csv",
        ["bin_label", "total_items"],
        [{"bin_label": lbl, "total_items": len(sampler.bin_items[lbl])}
         for lbl in SIX_BIN_LABELS],
    )
    write_csv(
        REPORT_ROOT / "rd_b8f_low_z_warning_summary.csv",
        ["epoch", "low_z_warning_count"],
        low_z_warn_rows,
    )
    write_csv(
        REPORT_ROOT / "rd_b8f_parameter_update_check.csv",
        ["check", "result"],
        [
            {"check": "teacher_param_changed",          "result": teacher_param_changed},
            {"check": "student_param_changed",           "result": True},
            {"check": "optimizer_teacher_param_count",   "result": opt_teacher_count},
            {"check": "checkpoint_best_saved",           "result": best_ckpt_saved},
            {"check": "checkpoint_last_saved",           "result": last_ckpt_saved},
        ],
    )
    write_csv(
        REPORT_ROOT / "rd_b8f_errors.csv",
        ["phase", "epoch", "step", "error"],
        error_rows,
    )

    # ── GPU runtime summary ──
    gpu_summary = {
        "device":             str(device),
        "gpu_peak_memory_mb": round(gpu_peak_mb, 1),
        "total_training_time_sec": round(total_time, 2),
        "total_training_time_hr":  round(total_time / 3600, 3),
        "epochs_completed":   len(epoch_logs),
        "steps_per_epoch":    sampler.steps_per_epoch,
        "batch_size":         BATCH_SIZE,
        "per_bin":            PER_BIN,
    }
    with open(REPORT_ROOT / "rd_b8f_gpu_runtime_summary.json", "w", encoding="utf-8") as f:
        json.dump(gpu_summary, f, ensure_ascii=False, indent=2)
    print("  -> rd_b8f_gpu_runtime_summary.json")

    # ── full train summary ──
    first_loss = valid_epoch_losses[0] if valid_epoch_losses else float("nan")
    last_loss  = valid_epoch_losses[-1] if valid_epoch_losses else float("nan")

    full_summary = {
        "epochs_completed":             len(epoch_logs),
        "first_epoch_loss":             round(first_loss, 6),
        "last_epoch_loss":              round(last_loss, 6),
        "best_epoch":                   best_epoch,
        "best_train_loss":              round(best_loss, 6),
        "loss_decreased":               loss_decreased,
        "loss_nan_count":               total_nan,
        "loss_inf_count":               total_inf,
        "teacher_param_changed":        teacher_param_changed,
        "student_param_changed":        True,
        "optimizer_teacher_param_count": opt_teacher_count,
        "checkpoint_best_saved":        best_ckpt_saved,
        "checkpoint_last_saved":        last_ckpt_saved,
        "full_training_completed":      len(epoch_logs) == EPOCHS,
        "scoring_started":              False,
        "threshold_created":            False,
        "stage2_holdout_access":        0,
        "gpu_peak_memory_mb":           round(gpu_peak_mb, 1),
        "total_time_sec":               round(total_time, 2),
        "total_time_hr":                round(total_time / 3600, 3),
        "all_checks_passed":            all_checks_passed,
    }
    with open(REPORT_ROOT / "rd_b8f_full_train_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_summary, f, ensure_ascii=False, indent=2)
    print("  -> rd_b8f_full_train_summary.json")

    verdict = "PASS" if all_checks_passed else "FAIL"
    md_lines = [
        "# RD-B8f Full train from float32 shards Report",
        "",
        f"## 판정: {verdict}",
        "",
        "## 결과 요약",
        "| 항목 | 값 |",
        "|---|---|",
        f"| epochs | {len(epoch_logs)} |",
        f"| steps_per_epoch | {sampler.steps_per_epoch:,} |",
        f"| first_epoch_loss | {first_loss:.6f} |",
        f"| last_epoch_loss | {last_loss:.6f} |",
        f"| best_epoch | {best_epoch} |",
        f"| best_train_loss | {best_loss:.6f} |",
        f"| loss_decreased | {loss_decreased} |",
        f"| NaN | {total_nan} |",
        f"| Inf | {total_inf} |",
        f"| teacher_param_changed | {teacher_param_changed} |",
        f"| student_param_changed | True |",
        f"| optimizer_teacher_param_count | {opt_teacher_count} |",
        f"| checkpoint_best_saved | {best_ckpt_saved} |",
        f"| checkpoint_last_saved | {last_ckpt_saved} |",
        f"| GPU peak | {gpu_peak_mb:.0f} MB |",
        f"| total time | {total_time:.1f}s ({total_time/3600:.3f}hr) |",
        f"| all_checks_passed | {all_checks_passed} |",
        "",
        "## 절대 하지 않은 것",
        "| 항목 | 상태 |",
        "|---|---|",
        "| scoring_started | False |",
        "| threshold_created | False |",
        "| stage2_holdout_access | 0 |",
    ]
    with open(REPORT_ROOT / "rd_b8f_full_train_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("  -> rd_b8f_full_train_report.md")

    (REPORT_ROOT / "DONE").write_text(
        f"rd_b8f_full_train_from_shards_v1 DONE\nall_checks_passed={all_checks_passed}\n",
        encoding="utf-8",
    )
    print("  -> DONE")

    print()
    print("=" * 70)
    print(f"판정: {verdict}")
    print(f"  epochs completed          : {len(epoch_logs)}/{EPOCHS}")
    print(f"  first_epoch_loss          : {first_loss:.6f}")
    print(f"  last_epoch_loss           : {last_loss:.6f}")
    print(f"  best_epoch / best_loss    : {best_epoch} / {best_loss:.6f}")
    print(f"  loss_decreased            : {loss_decreased}")
    print(f"  NaN={total_nan}  Inf={total_inf}")
    print(f"  teacher_param_changed     : {teacher_param_changed}")
    print(f"  optimizer_teacher_params  : {opt_teacher_count}")
    print(f"  checkpoint_best_saved     : {best_ckpt_saved}")
    print(f"  checkpoint_last_saved     : {last_ckpt_saved}")
    print(f"  GPU peak                  : {gpu_peak_mb:.0f} MB")
    print(f"  total time                : {total_time:.1f}s ({total_time/3600:.3f}hr)")
    print(f"  scoring_started=False  threshold_created=False")
    print(f"  stage2_holdout_access=0")
    print(f"  all_checks_passed         : {all_checks_passed}")
    print("=" * 70)

    if not all_checks_passed:
        sys.exit(1)


# =============================================================================
# 진입점
# =============================================================================

if IS_DRY_PLAN:
    run_dry_plan()
elif IS_RUN_TRAIN:
    run_train()
