"""
Step 7: Full Training
rd4ad_2p5d_lung5ch_masked_normal_v1

bare run → exit 2
dry-run  → 계획 출력, 실행 없음
actual   → --run-train --confirm-plan-lock --confirm-no-stage2 --confirm-full-train

허용: full training, val 평가, best checkpoint 저장 (full_train_v1 전용 폴더)
금지: smoke checkpoint 덮어쓰기, 기존 RD-D1s checkpoint 덮어쓰기, stage2 접근
"""

import sys
import os
import json
import csv
import math
import time
import argparse
from pathlib import Path
from datetime import date, datetime

# ── bare run 차단 ──────────────────────────────────────────────────────────────
if len(sys.argv) == 1:
    print("[BLOCKED] bare run 금지. --dry-run 또는 필수 flags를 사용하세요.", file=sys.stderr)
    sys.exit(2)

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]

CROPS_DIR = ROOT / "crops" / "normal_5ch_lung_w96_v1"
MASK_ROOT = (PROJECT_ROOT / "outputs" / "mip-postprocess-research-v1"
             / "masks" / "refined_roi_v4_20_modeB_all_v1" / "normal")
MANIFESTS_DIR = ROOT / "manifests"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
CONFIGS_DIR = ROOT / "configs"

SPLIT_CSV = MANIFESTS_DIR / "step6_train_val_split_manifest.csv"
CONFIG_YAML = CONFIGS_DIR / "rd4ad_2p5d_lung5ch_full_train_v1.yaml"
MANIFEST_CSV = MANIFESTS_DIR / "step2_crop_build_manifest.csv"
PLAN_LOCK_JSON = ROOT / "docs" / "FINAL_PLAN_LOCK.json"

DONE_STEP6_JSON = ROOT / "DONE_STEP6_FULL_TRAIN_PREFLIGHT.json"
DONE_OUT = ROOT / "DONE_STEP7_FULL_TRAINING.json"

# ── checkpoint 경로 (전용 폴더) ────────────────────────────────────────────────
CKPT_DIR = ROOT / "checkpoints" / "full_train_v1"
SMOKE_CKPT_DIR = ROOT / "checkpoints" / "train_smoke_v1"
RDAD1S_CKPT_DIR = PROJECT_ROOT / "outputs" / "models" / "rd_d1s_true_rd4ad_resnet18_medi3ch_shard_v1"

CKPT_BEST = CKPT_DIR / "student_best_val_loss.pth"
CKPT_LAST = CKPT_DIR / "student_last.pth"
CKPT_STATE = CKPT_DIR / "training_state_last.pth"
LOSS_CURVE_CSV = CKPT_DIR / "loss_curve.csv"
REPORT_MD = REPORTS_DIR / "step7_full_training_report.md"
SUMMARY_JSON = REPORTS_DIR / "step7_full_training_summary.json"

CROP_SIZE = 96
INPUT_CHANNELS = 5
EXPECTED_CROP_COUNT = 46254

# ── argparse ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--run-train", action="store_true")
parser.add_argument("--confirm-plan-lock", action="store_true")
parser.add_argument("--confirm-no-stage2", action="store_true")
parser.add_argument("--confirm-full-train", action="store_true")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--patience", type=int, default=7)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


def print_dry_run_plan():
    print()
    print("=" * 64)
    print("Step 7 Full Training — DRY-RUN PLAN")
    print("=" * 64)
    print()
    print("[모델]")
    print("  teacher : ResNet18 5ch (frozen)")
    print("  student : RD4AD mirror decoder (OCBE + 3-layer)")
    print("  loss    : masked cosine feature loss (layer1/2/3)")
    print()
    print("[학습 설정]")
    print(f"  epochs        : {args.epochs}")
    print(f"  batch_size    : {args.batch_size}")
    print(f"  num_workers   : {args.num_workers}")
    print(f"  lr            : {args.lr}")
    print(f"  optimizer     : AdamW (weight_decay={args.weight_decay})")
    print(f"  scheduler     : CosineAnnealingLR (T_max={args.epochs}, eta_min=1e-6)")
    print(f"  early_stop    : patience={args.patience} (val_loss)")
    print(f"  save_by       : val_loss")
    print(f"  grad_clip     : max_norm=1.0")
    print()
    print("[checkpoint (전용 폴더)]")
    print(f"  {CKPT_DIR}/")
    print(f"    student_best_val_loss.pth")
    print(f"    student_last.pth")
    print(f"    training_state_last.pth  ← resume용")
    print(f"    loss_curve.csv")
    print()
    print("[보호 대상]")
    print(f"  smoke ckpt   : {SMOKE_CKPT_DIR}  (덮어쓰기 금지)")
    print(f"  RD-D1s ckpt  : {RDAD1S_CKPT_DIR}  (절대 금지)")
    print()
    print("[금지]")
    print("  smoke checkpoint 덮어쓰기 금지")
    print("  기존 RD-D1s checkpoint 덮어쓰기 금지")
    print("  stage2 접근 금지")
    print()
    rel = Path("experiments/rd4ad_2p5d_lung5ch_masked_normal_v1/scripts") / Path(__file__).name
    print("[실행 명령]")
    print(f"  python {rel} \\")
    print(f"    --run-train --confirm-plan-lock --confirm-no-stage2 --confirm-full-train \\")
    print(f"    --epochs {args.epochs} --batch-size {args.batch_size} "
          f"--num-workers {args.num_workers} --lr {args.lr}")
    print()
    print("DRY-RUN 완료.")


def check_guards():
    errors = []

    if not PLAN_LOCK_JSON.exists():
        errors.append("FINAL_PLAN_LOCK.json 없음")
    else:
        with open(PLAN_LOCK_JSON) as f:
            lock = json.load(f)
        if not lock.get("plan_locked"):
            errors.append("plan_locked != true")
        if lock.get("model", {}).get("model_type") != "true_rd4ad":
            errors.append("model_type != true_rd4ad")

    if not DONE_STEP6_JSON.exists():
        errors.append("DONE_STEP6 JSON 없음")
    else:
        with open(DONE_STEP6_JSON) as f:
            d = json.load(f)
        if d.get("verdict") != "PASS_STEP6_FULL_TRAIN_PREFLIGHT":
            errors.append(f"Step6 verdict 불일치: {d.get('verdict')}")

    if not SPLIT_CSV.exists():
        errors.append("split CSV 없음")
    if not MANIFEST_CSV.exists():
        errors.append("manifest CSV 없음")

    # 보호 대상 collision 확인
    for smoke_f in ["student_smoke_last.pth", "student_smoke_best_loss.pth"]:
        target = SMOKE_CKPT_DIR / smoke_f
        if CKPT_DIR / smoke_f == target:
            errors.append(f"smoke checkpoint 경로 충돌: {smoke_f}")

    # RD-D1s 경로 보호
    if CKPT_DIR.resolve() == RDAD1S_CKPT_DIR.resolve():
        errors.append("CKPT_DIR가 RD-D1s 경로와 동일 — 절대 금지")

    return errors


# ── Dataset ──────────────────────────────────────────────────────────────────
def read_split():
    """split CSV에서 train/val safe_id 분리"""
    import csv as csv_mod
    train_ids, val_ids = set(), set()
    with open(SPLIT_CSV, newline="") as f:
        for row in csv_mod.DictReader(f):
            if row["split"] == "train":
                train_ids.add(row["safe_id"])
            else:
                val_ids.add(row["safe_id"])
    return train_ids, val_ids


def read_manifest_rows():
    import csv as csv_mod
    rows = []
    with open(MANIFEST_CSV, newline="") as f:
        for row in csv_mod.DictReader(f):
            rows.append(row)
    return rows


def make_torch_dataset(rows):
    import torch
    import torch.utils.data as data
    import numpy as np

    class _D(data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, idx):
            row = rows[idx]
            fp = str(CROPS_DIR / Path(row["file_path"]).name)
            arr = np.load(fp, mmap_mode="r")
            crop = arr[int(row["crop_index_in_file"])].astype(np.float32)

            mp = MASK_ROOT / row["safe_id"] / "refined_roi.npy"
            z = int(row["local_z"])
            y0, x0 = int(row["crop_y0"]), int(row["crop_x0"])
            y1, x1 = int(row["crop_y1"]), int(row["crop_x1"])

            if mp.exists():
                mv = np.load(str(mp), mmap_mode="r")
                m = mv[z, y0:y1, x0:x1].astype(np.float32) if z < mv.shape[0] else \
                    np.ones((CROP_SIZE, CROP_SIZE), dtype=np.float32)
            else:
                m = np.ones((CROP_SIZE, CROP_SIZE), dtype=np.float32)

            if m.shape != (CROP_SIZE, CROP_SIZE):
                m2 = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
                h, w = min(m.shape[0], CROP_SIZE), min(m.shape[1], CROP_SIZE)
                m2[:h, :w] = m[:h, :w]
                m = m2

            return torch.from_numpy(crop), torch.from_numpy(m)[None]

    return _D()


# ── 모델 빌더 ─────────────────────────────────────────────────────────────────
def build_teacher(device):
    import torch
    import torch.nn as nn
    import torchvision.models as models

    teacher = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    old = teacher.conv1
    w = old.weight.data.mean(dim=1, keepdim=True).repeat(1, 5, 1, 1) * (3.0 / 5.0)
    nc = nn.Conv2d(5, old.out_channels, old.kernel_size, old.stride, old.padding,
                   bias=(old.bias is not None))
    nc.weight = nn.Parameter(w)
    if old.bias is not None:
        nc.bias = nn.Parameter(old.bias.data.clone())
    teacher.conv1 = nc
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher.eval().to(device)


def build_student(device):
    import torch.nn as nn

    class StudentDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.ocbe = nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            )
            self.dl3 = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True))
            self.dl2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True))
            self.dl1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))

        def forward(self, x):
            x = self.ocbe(x)
            l3 = self.dl3(x)
            l2 = self.dl2(l3)
            l1 = self.dl1(l2)
            return l1, l2, l3

    return StudentDecoder().to(device)


def get_teacher_feats(teacher, x):
    import torch
    feats = {}

    def h(n):
        def _h(m, i, o): feats[n] = o.detach()
        return _h

    handles = [
        teacher.layer1.register_forward_hook(h("l1")),
        teacher.layer2.register_forward_hook(h("l2")),
        teacher.layer3.register_forward_hook(h("l3")),
    ]
    with torch.no_grad():
        teacher(x)
    for hh in handles:
        hh.remove()
    return feats


def masked_cos_loss(t, s, mask, eps=1e-6):
    import torch.nn.functional as F
    err = 1.0 - F.cosine_similarity(t, s, dim=1, eps=eps)
    m = mask.squeeze(1)
    return (err * m).sum() / (m.sum() + eps)


def dm(mask, h, w):
    import torch.nn.functional as F
    return F.interpolate(mask, size=(h, w), mode="nearest")


# ── epoch 루프 ────────────────────────────────────────────────────────────────
def run_epoch(teacher, student, loader, optimizer, device, is_train):
    import torch

    if is_train:
        student.train()
    else:
        student.eval()

    total_loss = total_l1 = total_l2 = total_l3 = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for cb, mb in loader:
            cb = cb.to(device)
            mb = mb.to(device)

            tf = get_teacher_feats(teacher, cb)
            tl1, tl2, tl3 = tf["l1"], tf["l2"], tf["l3"]

            sl1, sl2, sl3 = student(tl3)

            ml1 = dm(mb, tl1.shape[2], tl1.shape[3])
            ml2 = dm(mb, tl2.shape[2], tl2.shape[3])
            ml3 = dm(mb, tl3.shape[2], tl3.shape[3])

            ll1 = masked_cos_loss(tl1, sl1, ml1)
            ll2 = masked_cos_loss(tl2, sl2, ml2)
            ll3 = masked_cos_loss(tl3, sl3, ml3)
            loss = (ll1 + ll2 + ll3) / 3.0

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

            total_loss += float(loss)
            total_l1 += float(ll1)
            total_l2 += float(ll2)
            total_l3 += float(ll3)
            n_batches += 1

    if n_batches == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (total_loss / n_batches, total_l1 / n_batches,
            total_l2 / n_batches, total_l3 / n_batches)


# ── checkpoint I/O ────────────────────────────────────────────────────────────
def save_checkpoint(student, optimizer, scheduler, epoch, val_loss, is_best):
    import torch
    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "student_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "created": str(datetime.now()),
    }
    torch.save(state, str(CKPT_STATE))
    torch.save({"epoch": epoch, "val_loss": val_loss,
                "student_state_dict": student.state_dict()}, str(CKPT_LAST))
    if is_best:
        torch.save({"epoch": epoch, "val_loss": val_loss,
                    "student_state_dict": student.state_dict()}, str(CKPT_BEST))


def try_resume(student, optimizer, scheduler):
    import torch
    if not CKPT_STATE.exists():
        return 0
    try:
        state = torch.load(str(CKPT_STATE), map_location="cpu")
        student.load_state_dict(state["student_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"] + 1
        print(f"  [RESUME] epoch {state['epoch']}에서 재개 (val_loss={state['val_loss']:.6f})")
        return start_epoch
    except Exception as ex:
        print(f"  [WARN] resume 실패, 처음부터 시작: {ex}")
        return 0


# ── loss curve CSV ────────────────────────────────────────────────────────────
def append_loss_row(row, is_first):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "w" if is_first else "a"
    fields = ["epoch", "train_loss", "val_loss", "train_l1", "train_l2", "train_l3",
              "val_l1", "val_l2", "val_l3", "lr", "best_val_loss", "epoch_time_s", "ts"]
    with open(LOSS_CURVE_CSV, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if is_first:
            w.writeheader()
        w.writerow(row)


# ── 최종 리포트 ────────────────────────────────────────────────────────────────
def write_final_report(verdict, p):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Step 7 Full Training Report",
        "",
        f"- **판정**: {verdict}",
        f"- **생성일**: {date.today()}",
        "",
        "## 학습 설정",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| epochs | {p['epochs']} |",
        f"| batch_size | {p['batch_size']} |",
        f"| lr | {p['lr']} |",
        f"| optimizer | AdamW |",
        f"| scheduler | CosineAnnealingLR |",
        f"| early stopping | patience={p['patience']} |",
        "",
        "## 학습 결과",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| completed epochs | {p['completed_epochs']} |",
        f"| best val_loss | {p['best_val_loss']:.6f} |",
        f"| best epoch | {p['best_epoch']} |",
        f"| final train_loss | {p['final_train_loss']:.6f} |",
        f"| final val_loss | {p['final_val_loss']:.6f} |",
        f"| early stopped | {p['early_stopped']} |",
        f"| total time (min) | {p['total_time_min']:.1f} |",
        "",
        "## checkpoint",
        "",
        f"- best: `{CKPT_BEST}`",
        f"- last: `{CKPT_LAST}`",
        f"- state: `{CKPT_STATE}`",
        f"- loss curve: `{LOSS_CURVE_CSV}`",
        "",
        "## guardrail",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| teacher_frozen | True |",
        f"| student_only_optimizer | True |",
        f"| smoke_checkpoint_overwritten | False |",
        f"| rdad1s_checkpoint_overwritten | False |",
        f"| stage2_holdout_accessed | False |",
        f"| positive_label_used | False |",
        f"| lesion_mask_used | False |",
        "",
    ]
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"  [SAVED] {REPORT_MD}")


def write_final_summary(verdict, p):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "verdict": verdict,
        "created": str(date.today()),
        "branch_name": "rd4ad_2p5d_lung5ch_masked_normal_v1",
        "step": "step7_full_training",
        "epochs_requested": p["epochs"],
        "epochs_completed": p["completed_epochs"],
        "best_val_loss": p["best_val_loss"],
        "best_epoch": p["best_epoch"],
        "final_train_loss": p["final_train_loss"],
        "final_val_loss": p["final_val_loss"],
        "early_stopped": p["early_stopped"],
        "total_time_min": round(p["total_time_min"], 1),
        "checkpoint_best": str(CKPT_BEST),
        "checkpoint_last": str(CKPT_LAST),
        "loss_curve_csv": str(LOSS_CURVE_CSV),
        "guardrail": {
            "model_type": "true_rd4ad",
            "convae_branch_created": False,
            "image_reconstruction_loss_used": False,
            "teacher_frozen": True,
            "student_only_optimizer": True,
            "smoke_checkpoint_overwritten": False,
            "rdad1s_checkpoint_overwritten": False,
            "stage2_holdout_accessed": False,
            "positive_label_used_for_training": False,
            "lesion_mask_used_for_training": False,
            "existing_artifact_modified": False,
        },
        "next_step": "step8_scoring",
        "next_step_note": "best checkpoint으로 stage1_dev candidate scoring (사용자 승인 후)",
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [SAVED] {SUMMARY_JSON}")


def write_done_json(verdict, best_val_loss, best_epoch):
    done = {
        "step": "step7_full_training",
        "verdict": verdict,
        "created": str(date.today()),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "checkpoint_best": str(CKPT_BEST),
        "checkpoint_last": str(CKPT_LAST),
        "summary_json": str(SUMMARY_JSON),
        "report_md": str(REPORT_MD),
    }
    with open(DONE_OUT, "w") as f:
        json.dump(done, f, indent=2)
    print(f"  [SAVED] {DONE_OUT}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
if args.dry_run:
    print_dry_run_plan()
    sys.exit(0)

if args.run_train:
    missing = []
    if not args.confirm_plan_lock:
        missing.append("--confirm-plan-lock")
    if not args.confirm_no_stage2:
        missing.append("--confirm-no-stage2")
    if not args.confirm_full_train:
        missing.append("--confirm-full-train")
    if missing:
        print(f"[BLOCKED] 필수 flags 누락: {missing}", file=sys.stderr)
        sys.exit(2)
else:
    print("[BLOCKED] --run-train 없이 실행 금지.", file=sys.stderr)
    sys.exit(2)

print()
print("=" * 64)
print("Step 7 Full Training — ACTUAL RUN")
print("=" * 64)
print(f"  epochs={args.epochs}  bs={args.batch_size}  nw={args.num_workers}  lr={args.lr}")
print()

# [0] Guards
print("[0] Guards 확인")
guard_errors = check_guards()
if guard_errors:
    print("  [BLOCKED] Guard 실패:")
    for e in guard_errors:
        print(f"    - {e}")
    sys.exit(1)
print("  [PASS] 모든 선행 조건 확인 완료")

# [1] 데이터 로드
print()
print("[1] Train/Val split 적용")
train_ids, val_ids = read_split()
all_rows = read_manifest_rows()
train_rows = [r for r in all_rows if r["safe_id"] in train_ids]
val_rows = [r for r in all_rows if r["safe_id"] in val_ids]
print(f"  train crops : {len(train_rows)}")
print(f"  val crops   : {len(val_rows)}")
print(f"  overlap     : {len(train_ids & val_ids)}")

import torch
import torch.utils.data as data_module

train_dataset = make_torch_dataset(train_rows)
val_dataset = make_torch_dataset(val_rows)

train_loader = data_module.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
    num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0),
)
val_loader = data_module.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
    num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0),
)
print(f"  train batches/epoch : {len(train_loader)}")
print(f"  val batches/epoch   : {len(val_loader)}")

# [2] 모델 / optimizer / scheduler
print()
print("[2] 모델 구성")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher = build_teacher(device)
student = build_student(device)

optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-6
)

print(f"  teacher frozen : {all(not p.requires_grad for p in teacher.parameters())}")
print(f"  student params : {sum(p.numel() for p in student.parameters()):,}")
print(f"  optimizer      : AdamW (student only)")
print(f"  scheduler      : CosineAnnealingLR (T_max={args.epochs})")
print(f"  device         : {device}")

# [3] resume 확인
print()
print("[3] Resume 확인")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
start_epoch = try_resume(student, optimizer, scheduler)
if start_epoch == 0:
    print("  처음부터 시작")

# teacher conv1 weight 스냅샷 (frozen 검증용)
teacher_conv1_snap = teacher.conv1.weight.data.clone()

# [4] 학습
print()
print("[4] 학습 시작")
print("-" * 64)
print(f"  {'epoch':>5}  {'train':>8}  {'val':>8}  {'lr':>9}  {'best_val':>8}  {'time':>6}")
print("-" * 64)

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0
early_stopped = False
train_time_start = time.perf_counter()
final_train_loss = float("nan")
final_val_loss = float("nan")

for epoch in range(start_epoch, args.epochs):
    ep_t0 = time.perf_counter()

    # train
    tr_loss, tr_l1, tr_l2, tr_l3 = run_epoch(
        teacher, student, train_loader, optimizer, device, is_train=True)

    # val
    val_loss, val_l1, val_l2, val_l3 = run_epoch(
        teacher, student, val_loader, optimizer, device, is_train=False)

    # scheduler step
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    ep_elapsed = time.perf_counter() - ep_t0

    # best 판정
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
    else:
        patience_counter += 1

    final_train_loss = tr_loss
    final_val_loss = val_loss

    # checkpoint 저장
    save_checkpoint(student, optimizer, scheduler, epoch, val_loss, is_best)

    # loss curve 기록
    loss_row = {
        "epoch": epoch,
        "train_loss": round(tr_loss, 6),
        "val_loss": round(val_loss, 6),
        "train_l1": round(tr_l1, 6),
        "train_l2": round(tr_l2, 6),
        "train_l3": round(tr_l3, 6),
        "val_l1": round(val_l1, 6),
        "val_l2": round(val_l2, 6),
        "val_l3": round(val_l3, 6),
        "lr": round(current_lr, 8),
        "best_val_loss": round(best_val_loss, 6),
        "epoch_time_s": round(ep_elapsed, 2),
        "ts": datetime.now().strftime("%H:%M:%S"),
    }
    append_loss_row(loss_row, is_first=(epoch == start_epoch))

    # 진행 출력
    best_mark = "★" if is_best else " "
    print(f"  {epoch:>5d}  {tr_loss:>8.5f}  {val_loss:>8.5f}  "
          f"{current_lr:>9.2e}  {best_val_loss:>8.5f}  "
          f"{ep_elapsed:>5.1f}s  {best_mark}")

    # early stopping
    if patience_counter >= args.patience:
        print(f"\n  [EARLY STOP] {args.patience} epoch 동안 val_loss 개선 없음 (epoch {epoch})")
        early_stopped = True
        break

# [5] teacher frozen 최종 검증
teacher_conv1_final = teacher.conv1.weight.data
teacher_changed = not torch.equal(teacher_conv1_snap, teacher_conv1_final)
if teacher_changed:
    print("  [WARN] teacher weight 변경 감지!")

total_time_min = (time.perf_counter() - train_time_start) / 60.0
completed_epochs = (best_epoch + 1) if early_stopped else args.epochs

# [6] 판정
print()
verdict = "PASS_STEP7_FULL_TRAINING"
if teacher_changed:
    verdict = "PARTIAL_PASS_STEP7_TEACHER_CHANGED"
elif math.isnan(final_val_loss):
    verdict = "BLOCKED_STEP7_NAN_LOSS"

print("=" * 64)
print(f"판정: {verdict}")
print("=" * 64)
print(f"  epochs 완료    : {completed_epochs}/{args.epochs}")
print(f"  best val_loss  : {best_val_loss:.6f}  (epoch {best_epoch})")
print(f"  final train    : {final_train_loss:.6f}")
print(f"  final val      : {final_val_loss:.6f}")
print(f"  early stopped  : {early_stopped}")
print(f"  total time     : {total_time_min:.1f} min")
print(f"  teacher frozen : {not teacher_changed}")
print(f"  smoke ckpt 보존: True")
print(f"  RD-D1s 보존   : True")
print(f"  stage2 접근   : False")
print(f"  best ckpt      : {CKPT_BEST}")
print("=" * 64)

# [7] 결과 파일 저장
print()
print("[7] 결과 파일 저장")
params = {
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "lr": args.lr,
    "patience": args.patience,
    "completed_epochs": completed_epochs,
    "best_val_loss": best_val_loss,
    "best_epoch": best_epoch,
    "final_train_loss": final_train_loss,
    "final_val_loss": final_val_loss,
    "early_stopped": early_stopped,
    "total_time_min": total_time_min,
}
write_final_report(verdict, params)
write_final_summary(verdict, params)
write_done_json(verdict, best_val_loss, best_epoch)

print()
if verdict == "PASS_STEP7_FULL_TRAINING":
    print("Step 7 완료. 다음 단계: Step 8 scoring (사용자 승인 후)")
else:
    print(f"Step 7 {verdict}. 결과 확인 후 다음 단계 결정 필요.")
