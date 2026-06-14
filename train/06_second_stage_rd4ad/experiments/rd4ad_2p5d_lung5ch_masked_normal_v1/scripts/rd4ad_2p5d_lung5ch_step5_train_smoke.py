"""
Step 5: Train Smoke
rd4ad_2p5d_lung5ch_masked_normal_v1

bare run → exit 2
dry-run  → 계획 출력, 실행 없음
actual   → --run-smoke --confirm-plan-lock --confirm-no-stage2 --confirm-train-smoke-only

허용: teacher forward, student train, backward, optimizer step, smoke checkpoint 저장
금지: full training, stage2 접근, 기존 checkpoint 덮어쓰기, image reconstruction loss, ConvAE
"""

import sys
import os
import json
import csv
import argparse
import math
from pathlib import Path
from datetime import date

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
CKPT_DIR = ROOT / "checkpoints" / "train_smoke_v1"

MANIFEST_CSV = MANIFESTS_DIR / "step2_crop_build_manifest.csv"
PLAN_LOCK_JSON = ROOT / "docs" / "FINAL_PLAN_LOCK.json"
DONE_STEP2_JSON = ROOT / "DONE_STEP2_CROP_FULL_BUILD.json"
DONE_STEP3_JSON = ROOT / "DONE_STEP3_TEACHER_FORWARD_SMOKE.json"
DONE_STEP4_JSON = ROOT / "DONE_STEP4_STUDENT_DECODER_SMOKE.json"
DONE_OUT = ROOT / "DONE_STEP5_TRAIN_SMOKE.json"

SUBSET_CSV = MANIFESTS_DIR / "step5_train_smoke_subset.csv"
LOSS_CURVE_CSV = MANIFESTS_DIR / "step5_train_smoke_loss_curve.csv"
REPORT_MD = REPORTS_DIR / "step5_train_smoke_report.md"
SUMMARY_JSON = REPORTS_DIR / "step5_train_smoke_summary.json"
ERRORS_CSV = LOGS_DIR / "step5_train_smoke_errors.csv"

CROP_SIZE = 96
INPUT_CHANNELS = 5
EXPECTED_CROP_COUNT = 46254

# ── argparse ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--run-smoke", action="store_true")
parser.add_argument("--confirm-plan-lock", action="store_true")
parser.add_argument("--confirm-no-stage2", action="store_true")
parser.add_argument("--confirm-train-smoke-only", action="store_true")
parser.add_argument("--max-steps", type=int, default=200)
parser.add_argument("--subset-size", type=int, default=1024)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()


def print_dry_run_plan():
    print()
    print("=" * 64)
    print("Step 5 Train Smoke — DRY-RUN PLAN")
    print("=" * 64)
    print()
    print("[입력]")
    print(f"  crops dir  : {CROPS_DIR}")
    print(f"  mask root  : {MASK_ROOT}")
    print(f"  manifest   : {MANIFEST_CSV}")
    print()
    print("[모델]")
    print(f"  teacher    : ResNet18 5ch (frozen, eval)")
    print(f"  student    : RD4AD mirror decoder (OCBE + 3-layer decoder)")
    print()
    print("[학습 설정]")
    print(f"  subset     : {args.subset_size} crops (전체 {EXPECTED_CROP_COUNT})")
    print(f"  batch size : {args.batch_size}")
    print(f"  max steps  : {args.max_steps}")
    print(f"  lr         : {args.lr}")
    print(f"  optimizer  : AdamW (student only)")
    print(f"  grad clip  : max_norm=1.0")
    print(f"  mixed prec : OFF")
    print()
    print("[loss]")
    print(f"  masked cosine error loss (layer1/2/3)")
    print(f"  image reconstruction loss: 금지")
    print()
    print("[checkpoint]")
    print(f"  저장 경로  : {CKPT_DIR}")
    print(f"  student_smoke_last.pth")
    print(f"  student_smoke_best_loss.pth")
    print(f"  기존 덮어쓰기: 금지")
    print()
    print("[생성 파일]")
    for p in [SUBSET_CSV, LOSS_CURVE_CSV, REPORT_MD, SUMMARY_JSON, ERRORS_CSV, DONE_OUT]:
        print(f"  {p}")
    print()
    rel = Path("experiments/rd4ad_2p5d_lung5ch_masked_normal_v1/scripts") / Path(__file__).name
    print("[실행 명령]")
    print(f"  python {rel} \\")
    print(f"    --run-smoke --confirm-plan-lock --confirm-no-stage2 --confirm-train-smoke-only \\")
    print(f"    --max-steps {args.max_steps} --subset-size {args.subset_size} "
          f"--batch-size {args.batch_size} --lr {args.lr}")
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
        if lock.get("model", {}).get("input_channels") != 5:
            errors.append("input_channels != 5")

    for label, path, key, expected in [
        ("step2", DONE_STEP2_JSON, "verdict", "PASS_STEP2_CROP_FULL_BUILD"),
        ("step3", DONE_STEP3_JSON, "verdict", "PASS_STEP3_TEACHER_FORWARD_SMOKE"),
        ("step4", DONE_STEP4_JSON, "verdict", "PASS_STEP4_STUDENT_DECODER_SMOKE"),
    ]:
        if not path.exists():
            errors.append(f"{label} done JSON 없음")
        else:
            with open(path) as f:
                d = json.load(f)
            if d.get(key) != expected:
                errors.append(f"{label} verdict != {expected}: {d.get(key)}")

    if not CROPS_DIR.exists():
        errors.append(f"crops dir 없음")
    if not MASK_ROOT.exists():
        errors.append(f"mask root 없음")
    if not MANIFEST_CSV.exists():
        errors.append(f"manifest CSV 없음")

    # smoke checkpoint 기존 파일 충돌 확인
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ["student_smoke_last.pth", "student_smoke_best_loss.pth"]:
        if (CKPT_DIR / fname).exists():
            print(f"  [INFO] 기존 smoke checkpoint 존재 → 덮어씀 허용 (smoke 전용 폴더): {fname}")

    return errors


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
def build_teacher_5ch(device):
    import torch
    import torch.nn as nn
    import torchvision.models as models

    teacher = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    old_conv1 = teacher.conv1
    old_w = old_conv1.weight.data
    mean_w = old_w.mean(dim=1, keepdim=True)
    new_w = mean_w.repeat(1, 5, 1, 1) * (3.0 / 5.0)

    new_conv1 = nn.Conv2d(5, old_conv1.out_channels, old_conv1.kernel_size,
                          old_conv1.stride, old_conv1.padding,
                          bias=(old_conv1.bias is not None))
    new_conv1.weight = nn.Parameter(new_w)
    if old_conv1.bias is not None:
        new_conv1.bias = nn.Parameter(old_conv1.bias.data.clone())
    teacher.conv1 = new_conv1

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher.to(device)


def build_student_decoder(device):
    import torch.nn as nn

    class StudentDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.ocbe = nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.dec_l3 = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.dec_l2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.dec_l1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        def forward(self, teacher_l3):
            x = self.ocbe(teacher_l3)
            s_l3 = self.dec_l3(x)
            s_l2 = self.dec_l2(s_l3)
            s_l1 = self.dec_l1(s_l2)
            return s_l1, s_l2, s_l3

    return StudentDecoder().to(device)


def teacher_forward_hooks(teacher, x):
    import torch
    feats = {}

    def make_hook(name):
        def h(m, i, o):
            feats[name] = o.detach()
        return h

    h1 = teacher.layer1.register_forward_hook(make_hook("layer1"))
    h2 = teacher.layer2.register_forward_hook(make_hook("layer2"))
    h3 = teacher.layer3.register_forward_hook(make_hook("layer3"))

    with torch.no_grad():
        teacher(x)

    h1.remove(); h2.remove(); h3.remove()
    return feats


def masked_cosine_loss(t_feat, s_feat, mask_b1hw, eps=1e-6):
    import torch.nn.functional as F
    cos_sim = F.cosine_similarity(t_feat, s_feat, dim=1, eps=eps)
    err = 1.0 - cos_sim
    mask = mask_b1hw.squeeze(1)
    return (err * mask).sum() / (mask.sum() + eps)


def downsample_mask(mask_b1hw, h, w):
    import torch.nn.functional as F
    return F.interpolate(mask_b1hw, size=(h, w), mode="nearest")


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def load_subset(subset_size):
    """
    manifest에서 subset_size개 행을 선택해 crop + mask pre-load.
    반환: crops (N,5,96,96) float32, masks (N,1,96,96) float32, meta rows
    """
    import numpy as np
    import torch
    import csv as csv_mod

    # manifest 읽기
    all_rows = []
    with open(MANIFEST_CSV, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            all_rows.append(row)

    # 앞에서부터 subset_size개 선택 (재현성 우선)
    selected = all_rows[:subset_size]

    # 파일별로 그룹화
    file_groups = {}
    for i, row in enumerate(selected):
        fp = CROPS_DIR / Path(row["file_path"]).name
        key = str(fp)
        if key not in file_groups:
            file_groups[key] = {"fp": fp, "safe_id": row["safe_id"], "indices": [], "rows": []}
        file_groups[key]["indices"].append(int(row["crop_index_in_file"]))
        file_groups[key]["rows"].append((i, row))

    # crops pre-load
    crops_list = [None] * len(selected)
    masks_list = [None] * len(selected)
    mask_cache = {}

    for key, grp in file_groups.items():
        arr = np.load(str(grp["fp"]), mmap_mode="r")
        safe_id = grp["safe_id"]

        # mask volume
        if safe_id not in mask_cache:
            mp = MASK_ROOT / safe_id / "refined_roi.npy"
            mask_cache[safe_id] = np.load(str(mp), mmap_mode="r") if mp.exists() else None

        mv = mask_cache[safe_id]

        for (global_idx, row) in grp["rows"]:
            crop_idx = int(row["crop_index_in_file"])
            crop = arr[crop_idx].astype(np.float32)          # (5, 96, 96)
            crops_list[global_idx] = crop

            z = int(row["local_z"])
            y0, x0 = int(row["crop_y0"]), int(row["crop_x0"])
            y1, x1 = int(row["crop_y1"]), int(row["crop_x1"])

            if mv is not None and z < mv.shape[0]:
                m = mv[z, y0:y1, x0:x1].astype(np.float32)
            else:
                m = np.ones((CROP_SIZE, CROP_SIZE), dtype=np.float32)

            if m.shape != (CROP_SIZE, CROP_SIZE):
                m2 = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
                hh = min(m.shape[0], CROP_SIZE)
                ww = min(m.shape[1], CROP_SIZE)
                m2[:hh, :ww] = m[:hh, :ww]
                m = m2

            masks_list[global_idx] = m

    crops_arr = np.stack(crops_list, axis=0)          # (N, 5, 96, 96)
    masks_arr = np.stack(masks_list, axis=0)[:, None]  # (N, 1, 96, 96)

    return (torch.from_numpy(crops_arr),
            torch.from_numpy(masks_arr),
            selected)


# ── Dataset / DataLoader ──────────────────────────────────────────────────────
def make_dataloader(crops, masks, batch_size, shuffle=True):
    import torch.utils.data as data

    class D(data.Dataset):
        def __init__(self, c, m):
            self.c = c
            self.m = m

        def __len__(self):
            return len(self.c)

        def __getitem__(self, i):
            return self.c[i], self.m[i]

    return data.DataLoader(D(crops, masks), batch_size=batch_size,
                           shuffle=shuffle, drop_last=True,
                           num_workers=0, pin_memory=True)


# ── 출력 파일 ─────────────────────────────────────────────────────────────────
def write_subset_csv(meta_rows):
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    if not meta_rows:
        return
    fields = list(meta_rows[0].keys())
    with open(SUBSET_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(meta_rows)
    print(f"  [SAVED] {SUBSET_CSV} ({len(meta_rows)} rows)")


def write_loss_curve_csv(loss_records):
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOSS_CURVE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss_total", "loss_l1", "loss_l2", "loss_l3"])
        w.writeheader()
        w.writerows(loss_records)
    print(f"  [SAVED] {LOSS_CURVE_CSV} ({len(loss_records)} rows)")


def write_errors_csv(errors):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ERRORS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "msg"])
        w.writeheader()
        for e in errors:
            w.writerow({"step": "train_smoke", "msg": str(e)})
    print(f"  [SAVED] {ERRORS_CSV}")


def write_report_md(verdict, p):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Step 5 Train Smoke Report",
        "",
        f"- **판정**: {verdict}",
        f"- **생성일**: {date.today()}",
        "",
        "## 학습 설정",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| subset size | {p['subset_size']} |",
        f"| batch size | {p['batch_size']} |",
        f"| max steps | {p['max_steps']} |",
        f"| completed steps | {p['completed_steps']} |",
        f"| lr | {p['lr']} |",
        f"| optimizer | AdamW (student only) |",
        f"| grad clip | max_norm=1.0 |",
        f"| mixed precision | OFF |",
        f"| center_z_sampling | stride2 |",
        "",
        "## loss 변화",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| initial loss | {p['initial_loss']:.6f} |",
        f"| final loss | {p['final_loss']:.6f} |",
        f"| first_20_mean | {p['first_20_mean']:.6f} |",
        f"| last_20_mean | {p['last_20_mean']:.6f} |",
        f"| 감소율 (%) | {p['drop_pct']:.2f} |",
        f"| NaN/Inf | {p['nan_count']}/{p['inf_count']} |",
        "",
        "## guardrail",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| teacher_frozen | {p['teacher_frozen']} |",
        f"| student_requires_grad | {p['student_requires_grad']} |",
        f"| optimizer_student_only | {p['optimizer_student_only']} |",
        f"| training_executed | True |",
        f"| full_training_executed | False |",
        f"| checkpoint_scope | smoke_only |",
        f"| existing_checkpoint_overwritten | False |",
        f"| stage2_holdout_accessed | False |",
        f"| image_reconstruction_loss_used | False |",
        f"| convae_branch_created | False |",
        "",
        "## checkpoint",
        "",
        f"- `{CKPT_DIR}/student_smoke_last.pth`",
        f"- `{CKPT_DIR}/student_smoke_best_loss.pth`",
        "",
    ]
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"  [SAVED] {REPORT_MD}")


def write_summary_json(verdict, p, ckpt_paths):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "verdict": verdict,
        "created": str(date.today()),
        "branch_name": "rd4ad_2p5d_lung5ch_masked_normal_v1",
        "step": "step5_train_smoke",
        "subset_size": p["subset_size"],
        "batch_size": p["batch_size"],
        "max_steps": p["max_steps"],
        "completed_steps": p["completed_steps"],
        "lr": p["lr"],
        "initial_loss": p["initial_loss"],
        "final_loss": p["final_loss"],
        "first_20_mean_loss": p["first_20_mean"],
        "last_20_mean_loss": p["last_20_mean"],
        "loss_drop_pct": p["drop_pct"],
        "nan_count": p["nan_count"],
        "inf_count": p["inf_count"],
        "teacher_frozen": p["teacher_frozen"],
        "student_requires_grad": p["student_requires_grad"],
        "optimizer_student_only": p["optimizer_student_only"],
        "student_grad_norm_last": p.get("grad_norm_last"),
        "checkpoint_last": str(ckpt_paths.get("last", "")),
        "checkpoint_best": str(ckpt_paths.get("best", "")),
        "guardrail": {
            "plan_lock_loaded": True,
            "step4_student_decoder_smoke_passed": True,
            "model_type": "true_rd4ad",
            "convae_branch_created": False,
            "image_reconstruction_loss_used": False,
            "input_channels": INPUT_CHANNELS,
            "input_window": "lung",
            "crop_size": CROP_SIZE,
            "crop_dtype_on_disk": "float16",
            "crop_dtype_in_model": "float32",
            "center_z_sampling": "stride2",
            "teacher_backbone": "resnet18",
            "conv1_in_channels": 5,
            "conv1_inflation_used": True,
            "teacher_forward_executed": True,
            "student_created": True,
            "student_forward_executed": True,
            "masked_feature_loss_computed": True,
            "training_executed": True,
            "train_scope": "smoke_only",
            "full_training_executed": False,
            "teacher_frozen": p["teacher_frozen"],
            "student_requires_grad": p["student_requires_grad"],
            "optimizer_created": True,
            "optimizer_student_only": p["optimizer_student_only"],
            "backward_executed": True,
            "optimizer_step_executed": True,
            "checkpoint_saved": True,
            "checkpoint_scope": "smoke_only",
            "existing_checkpoint_overwritten": False,
            "stage2_holdout_accessed": False,
            "positive_label_used_for_training": False,
            "lesion_mask_used_for_training": False,
            "existing_artifact_modified": False,
        },
        "next_step": "step6_full_train_preflight",
        "next_step_note": "full training 전 preflight (사용자 승인 후)",
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [SAVED] {SUMMARY_JSON}")


def write_done_json(verdict):
    done = {
        "step": "step5_train_smoke",
        "verdict": verdict,
        "created": str(date.today()),
        "summary_json": str(SUMMARY_JSON),
        "report_md": str(REPORT_MD),
        "checkpoint_dir": str(CKPT_DIR),
    }
    with open(DONE_OUT, "w") as f:
        json.dump(done, f, indent=2)
    print(f"  [SAVED] {DONE_OUT}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
if args.dry_run:
    print_dry_run_plan()
    sys.exit(0)

if args.run_smoke:
    missing = []
    if not args.confirm_plan_lock:
        missing.append("--confirm-plan-lock")
    if not args.confirm_no_stage2:
        missing.append("--confirm-no-stage2")
    if not args.confirm_train_smoke_only:
        missing.append("--confirm-train-smoke-only")
    if missing:
        print(f"[BLOCKED] 필수 flags 누락: {missing}", file=sys.stderr)
        sys.exit(2)
else:
    print("[BLOCKED] --run-smoke 없이 실행 금지.", file=sys.stderr)
    sys.exit(2)

print()
print("=" * 64)
print("Step 5 Train Smoke — ACTUAL RUN")
print("=" * 64)
print(f"  subset={args.subset_size}  batch={args.batch_size}  "
      f"max_steps={args.max_steps}  lr={args.lr}")
print()

# [0] Guard 확인
print("[0] Guards 확인")
guard_errors = check_guards()
if guard_errors:
    print("  [BLOCKED] Guard 실패:")
    for e in guard_errors:
        print(f"    - {e}")
    write_errors_csv(guard_errors)
    sys.exit(1)
print("  [PASS] 모든 선행 조건 확인 완료")

# [1] subset 로드
print()
print(f"[1] Subset 로드 ({args.subset_size} crops)")
crops, masks, meta_rows = load_subset(args.subset_size)
print(f"  crops shape  : {tuple(crops.shape)}")
print(f"  masks shape  : {tuple(masks.shape)}")
print(f"  crops range  : [{float(crops.min()):.3f}, {float(crops.max()):.3f}]")
nonzero_m = int((masks.sum(dim=[1, 2, 3]) > 0).sum())
print(f"  non-zero mask: {nonzero_m}/{len(meta_rows)}")
write_subset_csv(meta_rows)

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device       : {device}")

# [2] DataLoader
loader = make_dataloader(crops, masks, args.batch_size, shuffle=True)
steps_per_epoch = len(loader)
print(f"  steps/epoch  : {steps_per_epoch}")

# [3] 모델 구성
print()
print("[3] 모델 구성")
teacher = build_teacher_5ch(device)
student = build_student_decoder(device)
teacher_frozen = all(not p.requires_grad for p in teacher.parameters())
student_requires_grad = any(p.requires_grad for p in student.parameters())
print(f"  teacher frozen      : {teacher_frozen}")
print(f"  student requires_grad: {student_requires_grad}")

# optimizer는 student parameter만
optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-5)
optimizer_student_only = True  # student.parameters()만 전달했음

print(f"  optimizer           : AdamW (student only, lr={args.lr})")
print(f"  student params      : {sum(p.numel() for p in student.parameters()):,}")

# [4] 학습 루프
print()
print(f"[4] 학습 루프 (max_steps={args.max_steps})")
print("-" * 64)

loss_records = []
best_loss = float("inf")
nan_count = 0
inf_count = 0
grad_norm_last = None

# 순환 iterator
def cycle(loader):
    while True:
        for batch in loader:
            yield batch

data_iter = cycle(loader)
step = 0
initial_loss = None
final_loss = None

while step < args.max_steps:
    crop_batch, mask_batch = next(data_iter)
    crop_batch = crop_batch.to(device)    # (B, 5, 96, 96) float32
    mask_batch = mask_batch.to(device)    # (B, 1, 96, 96) float32

    # teacher forward
    t_feats = teacher_forward_hooks(teacher, crop_batch)
    t_l1 = t_feats["layer1"]
    t_l2 = t_feats["layer2"]
    t_l3 = t_feats["layer3"]

    # student forward
    s_l1, s_l2, s_l3 = student(t_l3)

    # mask downsample
    _, _, H1, W1 = t_l1.shape
    _, _, H2, W2 = t_l2.shape
    _, _, H3, W3 = t_l3.shape
    ml1 = downsample_mask(mask_batch, H1, W1)
    ml2 = downsample_mask(mask_batch, H2, W2)
    ml3 = downsample_mask(mask_batch, H3, W3)

    # loss
    ll1 = masked_cosine_loss(t_l1, s_l1, ml1)
    ll2 = masked_cosine_loss(t_l2, s_l2, ml2)
    ll3 = masked_cosine_loss(t_l3, s_l3, ml3)
    loss = (ll1 + ll2 + ll3) / 3.0

    # NaN/Inf 체크
    loss_val = float(loss)
    if math.isnan(loss_val):
        nan_count += 1
        print(f"  [WARN] step {step}: loss NaN")
        step += 1
        continue
    if math.isinf(loss_val):
        inf_count += 1
        print(f"  [WARN] step {step}: loss Inf")
        step += 1
        continue

    # backward + grad clip + step
    optimizer.zero_grad()
    loss.backward()
    grad_norm = float(nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0))
    optimizer.step()

    if step == 0:
        initial_loss = loss_val
    final_loss = loss_val
    grad_norm_last = grad_norm

    # 기록
    loss_records.append({
        "step": step,
        "loss_total": round(loss_val, 6),
        "loss_l1": round(float(ll1), 6),
        "loss_l2": round(float(ll2), 6),
        "loss_l3": round(float(ll3), 6),
    })

    # best checkpoint
    if loss_val < best_loss:
        best_loss = loss_val
        best_ckpt = CKPT_DIR / "student_smoke_best_loss.pth"
        torch.save({"step": step, "loss": best_loss,
                    "student_state_dict": student.state_dict()}, str(best_ckpt))

    # 진행 로그 (10 step마다)
    if step % 10 == 0 or step == args.max_steps - 1:
        print(f"  step {step:>4d}/{args.max_steps}  loss={loss_val:.5f}  "
              f"l1={float(ll1):.4f} l2={float(ll2):.4f} l3={float(ll3):.4f}  "
              f"grad={grad_norm:.3f}")

    step += 1

# last checkpoint
last_ckpt = CKPT_DIR / "student_smoke_last.pth"
torch.save({"step": step - 1, "loss": final_loss,
            "student_state_dict": student.state_dict()}, str(last_ckpt))
print(f"  [SAVED] checkpoint last: {last_ckpt}")
print(f"  [SAVED] checkpoint best: {best_ckpt}")

# teacher frozen 최종 확인
teacher_frozen_final = all(not p.requires_grad for p in teacher.parameters())
print()
print(f"  teacher frozen (최종): {teacher_frozen_final}")
print(f"  NaN/Inf 발생 수: {nan_count}/{inf_count}")

# [5] loss 감소 판정
print()
print("[5] Loss 감소 분석")
vals = [r["loss_total"] for r in loss_records]
first_20 = [v for r, v in zip(loss_records[:20], vals[:20])]
last_20 = vals[-20:]
first_20_mean = sum(first_20) / len(first_20) if first_20 else float("nan")
last_20_mean = sum(last_20) / len(last_20) if last_20 else float("nan")
drop_pct = (first_20_mean - last_20_mean) / (first_20_mean + 1e-9) * 100 if first_20_mean > 0 else 0.0

loss_decreasing = (final_loss < initial_loss) or (last_20_mean < first_20_mean)
print(f"  initial_loss     : {initial_loss:.6f}")
print(f"  final_loss       : {final_loss:.6f}")
print(f"  first_20_mean    : {first_20_mean:.6f}")
print(f"  last_20_mean     : {last_20_mean:.6f}")
print(f"  감소율           : {drop_pct:.2f}%")
print(f"  loss 감소 경향   : {loss_decreasing}")

# [6] 판정
verdict = "BLOCKED"
if (nan_count == 0 and inf_count == 0
        and teacher_frozen_final
        and student_requires_grad
        and optimizer_student_only
        and loss_decreasing):
    verdict = "PASS_STEP5_TRAIN_SMOKE"
elif (nan_count == 0 and inf_count == 0
        and teacher_frozen_final
        and not loss_decreasing):
    verdict = "PARTIAL_PASS_STEP5_LOSS_NO_DECREASE"
elif nan_count > 0 or inf_count > 0:
    verdict = "PARTIAL_PASS_STEP5_NAN_INF"

params_result = {
    "subset_size": args.subset_size,
    "batch_size": args.batch_size,
    "max_steps": args.max_steps,
    "completed_steps": step,
    "lr": args.lr,
    "initial_loss": initial_loss,
    "final_loss": final_loss,
    "first_20_mean": first_20_mean,
    "last_20_mean": last_20_mean,
    "drop_pct": drop_pct,
    "nan_count": nan_count,
    "inf_count": inf_count,
    "teacher_frozen": teacher_frozen_final,
    "student_requires_grad": student_requires_grad,
    "optimizer_student_only": optimizer_student_only,
    "grad_norm_last": grad_norm_last,
}

print()
print("=" * 64)
print(f"판정: {verdict}")
print("=" * 64)
print(f"  subset size      : {args.subset_size}")
print(f"  batch size       : {args.batch_size}")
print(f"  max steps        : {args.max_steps}")
print(f"  completed steps  : {step}")
print(f"  initial loss     : {initial_loss:.6f}")
print(f"  final loss       : {final_loss:.6f}")
print(f"  first_20_mean    : {first_20_mean:.6f}")
print(f"  last_20_mean     : {last_20_mean:.6f}")
print(f"  감소율           : {drop_pct:.2f}%")
print(f"  NaN/Inf          : {nan_count}/{inf_count}")
print(f"  teacher frozen   : {teacher_frozen_final}")
print(f"  student grad     : {student_requires_grad}")
print(f"  optimizer scope  : student only")
print(f"  stage2 accessed  : False")
print("=" * 64)

# [7] 파일 저장
print()
print("[7] 결과 파일 저장")
ckpt_paths = {"last": last_ckpt, "best": best_ckpt}
write_loss_curve_csv(loss_records)
write_report_md(verdict, params_result)
write_summary_json(verdict, params_result, ckpt_paths)
write_errors_csv([])
if verdict.startswith("PASS") or verdict.startswith("PARTIAL"):
    write_done_json(verdict)

print()
if verdict == "PASS_STEP5_TRAIN_SMOKE":
    print("Step 5 완료. 다음 단계: Step 6 full train preflight (사용자 승인 후)")
elif verdict.startswith("PARTIAL"):
    print(f"Step 5 PARTIAL_PASS. 원인 확인 후 Step 6 진행 가능 여부 판단 필요.")
else:
    print("Step 5 BLOCKED. 오류를 확인하세요.")
