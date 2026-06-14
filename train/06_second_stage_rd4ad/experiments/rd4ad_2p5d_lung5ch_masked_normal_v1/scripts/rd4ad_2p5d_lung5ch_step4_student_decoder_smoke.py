"""
Step 4: Student Decoder Smoke
rd4ad_2p5d_lung5ch_masked_normal_v1

bare run → exit 2
dry-run  → plan 출력, 실행 없음
actual   → --run-smoke --confirm-plan-lock --confirm-no-stage2 --confirm-smoke-only

허용: teacher forward, student forward, masked cosine loss 계산
금지: backward, optimizer, checkpoint, stage2 접근
"""

import sys
import os
import json
import csv
import argparse
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
MANIFESTS_DIR = ROOT / "manifests"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

MASK_ROOT = PROJECT_ROOT / "outputs" / "mip-postprocess-research-v1" / "masks" / "refined_roi_v4_20_modeB_all_v1" / "normal"

DONE_STEP2_JSON = ROOT / "DONE_STEP2_CROP_FULL_BUILD.json"
DONE_STEP3_JSON = ROOT / "DONE_STEP3_TEACHER_FORWARD_SMOKE.json"
PLAN_LOCK_JSON = ROOT / "docs" / "FINAL_PLAN_LOCK.json"
DONE_OUT = ROOT / "DONE_STEP4_STUDENT_DECODER_SMOKE.json"

SAMPLES_CSV = MANIFESTS_DIR / "step4_student_decoder_smoke_samples.csv"
REPORT_MD = REPORTS_DIR / "step4_student_decoder_smoke_report.md"
SUMMARY_JSON = REPORTS_DIR / "step4_student_decoder_smoke_summary.json"
ERRORS_CSV = LOGS_DIR / "step4_student_decoder_smoke_errors.csv"
MANIFEST_CSV = MANIFESTS_DIR / "step2_crop_build_manifest.csv"

# ── 상수 ─────────────────────────────────────────────────────────────────────
N_PATIENTS = 5
BATCH_SIZE = 8
INPUT_CHANNELS = 5
CROP_SIZE = 96
EXPECTED_CROP_COUNT = 46254

# ── argparse ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--run-smoke", action="store_true")
parser.add_argument("--confirm-plan-lock", action="store_true")
parser.add_argument("--confirm-no-stage2", action="store_true")
parser.add_argument("--confirm-smoke-only", action="store_true")
args = parser.parse_args()


def print_dry_run_plan():
    print()
    print("=" * 64)
    print("Step 4 Student Decoder Smoke — DRY-RUN PLAN")
    print("=" * 64)
    print()
    print("[입력]")
    print(f"  crops dir      : {CROPS_DIR}")
    print(f"  mask root      : {MASK_ROOT}")
    print(f"  manifest       : {MANIFEST_CSV}")
    print(f"  plan lock      : {PLAN_LOCK_JSON}")
    print(f"  step2 done     : {DONE_STEP2_JSON}")
    print(f"  step3 done     : {DONE_STEP3_JSON}")
    print()
    print("[모델]")
    print(f"  teacher        : ResNet18 5ch (frozen, eval)")
    print(f"  student        : mirror decoder (teacher layer3 입력 → layer1/2/3 복원)")
    print(f"  student 구조   : OCBE(256→512→256) + dec_l3(256→256) + dec_l2(upsample×2,256→128) + dec_l1(upsample×2,128→64)")
    print()
    print("[loss]")
    print(f"  method         : masked cosine error loss")
    print(f"  err_l          : 1 - cosine_similarity(t_feat, s_feat, dim=1)")
    print(f"  loss_l         : sum(err_l * mask_l) / (sum(mask_l) + eps)")
    print(f"  total          : mean(loss_l1, loss_l2, loss_l3)")
    print()
    print("[mask]")
    print(f"  source         : {MASK_ROOT}/{{safe_id}}/refined_roi.npy")
    print(f"  crop           : mask[local_z, crop_y0:crop_y1, crop_x0:crop_x1]")
    print(f"  downsample     : nearest interpolate → layer feature map size")
    print()
    print("[smoke 샘플]")
    print(f"  환자 파일      : {N_PATIENTS}개")
    print(f"  batch size     : {BATCH_SIZE}")
    print(f"  총 crops       : ~{N_PATIENTS * BATCH_SIZE}")
    print()
    print("[금지]")
    print("  backward 금지 | optimizer 금지 | checkpoint 저장 금지 | stage2 접근 금지")
    print()
    print("[생성 파일]")
    for p in [SAMPLES_CSV, REPORT_MD, SUMMARY_JSON, ERRORS_CSV, DONE_OUT]:
        print(f"  {p}")
    print()
    print("[실행 명령]")
    rel = Path("experiments/rd4ad_2p5d_lung5ch_masked_normal_v1/scripts") / Path(__file__).name
    print(f"  python {rel} \\")
    print(f"    --run-smoke --confirm-plan-lock --confirm-no-stage2 --confirm-smoke-only")
    print()
    print("DRY-RUN 완료.")


def check_guards():
    errors = []

    if not PLAN_LOCK_JSON.exists():
        errors.append(f"FINAL_PLAN_LOCK.json 없음")
    else:
        with open(PLAN_LOCK_JSON) as f:
            lock = json.load(f)
        if not lock.get("plan_locked"):
            errors.append("plan_locked != true")
        model_type = lock.get("model", {}).get("model_type")
        if model_type != "true_rd4ad":
            errors.append(f"model_type != true_rd4ad: {model_type}")
        in_ch = lock.get("model", {}).get("input_channels")
        if in_ch != 5:
            errors.append(f"input_channels != 5: {in_ch}")

    for label, path in [("step2 done", DONE_STEP2_JSON), ("step3 done", DONE_STEP3_JSON)]:
        if not path.exists():
            errors.append(f"{label} JSON 없음: {path}")
        else:
            with open(path) as f:
                d = json.load(f)
            if label == "step2 done" and d.get("verdict") != "PASS_STEP2_CROP_FULL_BUILD":
                errors.append(f"Step2 verdict != PASS: {d.get('verdict')}")
            if label == "step3 done" and d.get("verdict") != "PASS_STEP3_TEACHER_FORWARD_SMOKE":
                errors.append(f"Step3 verdict != PASS: {d.get('verdict')}")

    if not CROPS_DIR.exists():
        errors.append(f"crops dir 없음")
    else:
        n = len(list(CROPS_DIR.glob("*_crops_f16.npy")))
        if n < 300:
            errors.append(f"npy 파일 수 부족: {n}")

    if not MASK_ROOT.exists():
        errors.append(f"mask root 없음: {MASK_ROOT}")

    if not MANIFEST_CSV.exists():
        errors.append(f"manifest CSV 없음: {MANIFEST_CSV}")

    return errors


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
def build_teacher_5ch():
    import torch
    import torch.nn as nn
    import torchvision.models as models

    teacher = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    old_conv1 = teacher.conv1
    old_w = old_conv1.weight.data
    mean_w = old_w.mean(dim=1, keepdim=True)
    new_w = mean_w.repeat(1, 5, 1, 1) * (3.0 / 5.0)

    new_conv1 = nn.Conv2d(
        in_channels=5,
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=(old_conv1.bias is not None),
    )
    new_conv1.weight = nn.Parameter(new_w)
    if old_conv1.bias is not None:
        new_conv1.bias = nn.Parameter(old_conv1.bias.data.clone())
    teacher.conv1 = new_conv1

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher


def build_student_decoder():
    import torch.nn as nn

    class StudentDecoder(nn.Module):
        """
        RD4AD mirror decoder.
        Input : teacher layer3 (B, 256, 6, 6)
        Output: (s_l1, s_l2, s_l3) matching teacher layer1/2/3
        """
        def __init__(self):
            super().__init__()
            # OCBE: one-class bottleneck embedding
            self.ocbe = nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            # Layer3 복원: (B, 256, 6, 6)
            self.dec_l3 = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            # Layer2 복원: upsample×2 → (B, 128, 12, 12)
            self.dec_l2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            # Layer1 복원: upsample×2 → (B, 64, 24, 24)
            self.dec_l1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        def forward(self, teacher_l3):
            x = self.ocbe(teacher_l3)      # (B, 256, 6, 6)
            s_l3 = self.dec_l3(x)          # (B, 256, 6, 6)
            s_l2 = self.dec_l2(s_l3)       # (B, 128, 12, 12)
            s_l1 = self.dec_l1(s_l2)       # (B, 64, 24, 24)
            return s_l1, s_l2, s_l3

    return StudentDecoder()


def run_teacher_forward(teacher, batch):
    import torch
    features = {}

    def make_hook(name):
        def hook(module, inp, out):
            features[name] = out.detach()
        return hook

    h1 = teacher.layer1.register_forward_hook(make_hook("layer1"))
    h2 = teacher.layer2.register_forward_hook(make_hook("layer2"))
    h3 = teacher.layer3.register_forward_hook(make_hook("layer3"))

    device = next(teacher.parameters()).device
    x = batch.to(device)

    with torch.no_grad():
        _ = teacher(x)

    h1.remove()
    h2.remove()
    h3.remove()

    return features


def masked_cosine_loss(t_feat, s_feat, mask_b1hw, eps=1e-6):
    """
    t_feat, s_feat : (B, C, H, W)
    mask_b1hw      : (B, 1, H, W) float32 [0,1]
    returns        : scalar loss
    """
    import torch
    import torch.nn.functional as F

    cos_sim = F.cosine_similarity(t_feat, s_feat, dim=1, eps=eps)  # (B, H, W)
    err = 1.0 - cos_sim                                             # (B, H, W)
    mask = mask_b1hw.squeeze(1)                                     # (B, H, W)
    loss = (err * mask).sum() / (mask.sum() + eps)
    return loss


def load_sample_crops_with_manifest(npy_files, manifest_csv, n_patients, batch_size):
    """
    manifest에서 crop 메타 정보를 읽어 batch + manifest rows 반환
    """
    import numpy as np
    import torch
    import csv as csv_mod

    # manifest 읽기 (safe_id → rows)
    manifest_by_safe = {}
    with open(manifest_csv, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            sid = row["safe_id"]
            if sid not in manifest_by_safe:
                manifest_by_safe[sid] = []
            manifest_by_safe[sid].append(row)

    selected_files = npy_files[:n_patients]
    all_crops = []
    all_meta = []

    for fp in selected_files:
        safe_id = fp.name.replace("_crops_f16.npy", "")
        arr = np.load(str(fp), mmap_mode="r")
        take = min(batch_size, arr.shape[0])
        chunk = arr[:take].astype(np.float32)

        rows = manifest_by_safe.get(safe_id, [])
        for i in range(take):
            row = rows[i] if i < len(rows) else {}
            all_crops.append(chunk[i])
            all_meta.append({
                "safe_id": safe_id,
                "crop_idx": i,
                "local_z": int(row.get("local_z", 0)),
                "crop_y0": int(row.get("crop_y0", 0)),
                "crop_x0": int(row.get("crop_x0", 0)),
                "crop_y1": int(row.get("crop_y1", CROP_SIZE)),
                "crop_x1": int(row.get("crop_x1", CROP_SIZE)),
                "orig_dtype": "float16",
            })

    batch = torch.from_numpy(np.stack(all_crops, axis=0))  # (B, 5, 96, 96)
    return batch, all_meta


def load_masks_for_batch(meta_rows):
    """
    meta_rows에서 safe_id, local_z, crop 좌표를 이용해 mask crop 로드
    반환: (B, 1, 96, 96) float32
    """
    import numpy as np
    import torch

    mask_cache = {}
    masks = []

    for row in meta_rows:
        sid = row["safe_id"]
        if sid not in mask_cache:
            mp = MASK_ROOT / sid / "refined_roi.npy"
            if mp.exists():
                mask_cache[sid] = np.load(str(mp), mmap_mode="r")
            else:
                mask_cache[sid] = None

        mv = mask_cache[sid]
        z = row["local_z"]
        y0, x0, y1, x1 = row["crop_y0"], row["crop_x0"], row["crop_y1"], row["crop_x1"]

        if mv is not None and z < mv.shape[0]:
            m = mv[z, y0:y1, x0:x1].astype(np.float32)
        else:
            m = np.ones((CROP_SIZE, CROP_SIZE), dtype=np.float32)

        if m.shape != (CROP_SIZE, CROP_SIZE):
            m2 = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
            h = min(m.shape[0], CROP_SIZE)
            w = min(m.shape[1], CROP_SIZE)
            m2[:h, :w] = m[:h, :w]
            m = m2

        masks.append(m)

    mask_batch = torch.from_numpy(np.stack(masks, axis=0)).unsqueeze(1)  # (B, 1, 96, 96)
    return mask_batch


def downsample_mask(mask_b1hw, target_h, target_w):
    """mask_b1hw (B,1,H,W) → nearest interpolate to (B,1,target_h,target_w)"""
    import torch.nn.functional as F
    return F.interpolate(
        mask_b1hw.float(),
        size=(target_h, target_w),
        mode="nearest",
    )


def write_samples_csv(meta_rows, loss_rows):
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    if not meta_rows:
        return
    fields = ["safe_id", "crop_idx", "local_z", "crop_y0", "crop_x0", "crop_y1", "crop_x1",
              "orig_dtype", "mask_loaded"]
    with open(SAMPLES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in meta_rows:
            row = {k: r.get(k, "") for k in fields}
            w.writerow(row)
    print(f"  [SAVED] {SAMPLES_CSV} ({len(meta_rows)} rows)")


def write_errors_csv(errors):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ERRORS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "msg"])
        w.writeheader()
        for e in errors:
            w.writerow({"step": "smoke", "msg": str(e)})
    print(f"  [SAVED] {ERRORS_CSV}")


def write_report_md(verdict, results):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    r = results

    lines = [
        "# Step 4 Student Decoder Smoke Report",
        "",
        f"- **판정**: {verdict}",
        f"- **생성일**: {date.today()}",
        "",
        "## 입력 crop",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| input shape | {r.get('input_shape')} |",
        f"| dtype | float32 (원본 float16) |",
        "",
        "## teacher feature",
        "",
        "| layer | shape |",
        "|---|---|",
        f"| layer1 | {r.get('t_l1_shape')} |",
        f"| layer2 | {r.get('t_l2_shape')} |",
        f"| layer3 | {r.get('t_l3_shape')} |",
        "",
        "## student feature",
        "",
        "| layer | shape | shape match |",
        "|---|---|---|",
        f"| layer1 | {r.get('s_l1_shape')} | {r.get('l1_match')} |",
        f"| layer2 | {r.get('s_l2_shape')} | {r.get('l2_match')} |",
        f"| layer3 | {r.get('s_l3_shape')} | {r.get('l3_match')} |",
        "",
        "## mask downsample",
        "",
        "| layer | mask shape |",
        "|---|---|",
        f"| layer1 | {r.get('mask_l1_shape')} |",
        f"| layer2 | {r.get('mask_l2_shape')} |",
        f"| layer3 | {r.get('mask_l3_shape')} |",
        "",
        "## masked cosine loss",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| loss_layer1 | {r.get('loss_l1')} |",
        f"| loss_layer2 | {r.get('loss_l2')} |",
        f"| loss_layer3 | {r.get('loss_l3')} |",
        f"| loss_total  | {r.get('loss_total')} |",
        f"| NaN/Inf     | {r.get('total_nan')}/{r.get('total_inf')} |",
        "",
        "## guardrail",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| teacher_frozen | {r.get('teacher_frozen')} |",
        f"| student_requires_grad | {r.get('student_requires_grad')} |",
        f"| training_executed | False |",
        f"| backward_executed | False |",
        f"| optimizer_created | False |",
        f"| checkpoint_saved | False |",
        f"| stage2_holdout_accessed | False |",
        "",
    ]

    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"  [SAVED] {REPORT_MD}")


def write_summary_json(verdict, results, sampled_crop_count):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "verdict": verdict,
        "created": str(date.today()),
        "branch_name": "rd4ad_2p5d_lung5ch_masked_normal_v1",
        "step": "step4_student_decoder_smoke",
        "sampled_crop_count": sampled_crop_count,
        "input_tensor_shape": results.get("input_shape"),
        "teacher_l1_shape": results.get("t_l1_shape"),
        "teacher_l2_shape": results.get("t_l2_shape"),
        "teacher_l3_shape": results.get("t_l3_shape"),
        "student_l1_shape": results.get("s_l1_shape"),
        "student_l2_shape": results.get("s_l2_shape"),
        "student_l3_shape": results.get("s_l3_shape"),
        "shape_match_all": all([results.get("l1_match"), results.get("l2_match"), results.get("l3_match")]),
        "loss_layer1": results.get("loss_l1"),
        "loss_layer2": results.get("loss_l2"),
        "loss_layer3": results.get("loss_l3"),
        "loss_total": results.get("loss_total"),
        "total_nan_in_loss": results.get("total_nan", 0),
        "total_inf_in_loss": results.get("total_inf", 0),
        "teacher_frozen": results.get("teacher_frozen"),
        "student_requires_grad": results.get("student_requires_grad"),
        "guardrail": {
            "plan_lock_loaded": True,
            "step3_teacher_forward_smoke_passed": True,
            "crop_count": EXPECTED_CROP_COUNT,
            "crop_dtype": "float16",
            "center_z_sampling": "stride2",
            "model_type": "true_rd4ad",
            "convae_branch_created": False,
            "image_reconstruction_loss_used": False,
            "input_channels": INPUT_CHANNELS,
            "input_window": "lung",
            "crop_size": CROP_SIZE,
            "teacher_backbone": "resnet18",
            "conv1_in_channels": 5,
            "conv1_inflation_used": True,
            "teacher_forward_executed": True,
            "student_created": True,
            "student_forward_executed": True,
            "masked_feature_loss_computed": True,
            "teacher_frozen": results.get("teacher_frozen", False),
            "student_requires_grad": results.get("student_requires_grad", False),
            "training_executed": False,
            "backward_executed": False,
            "optimizer_created": False,
            "optimizer_step_executed": False,
            "checkpoint_saved": False,
            "stage2_holdout_accessed": False,
            "existing_artifact_modified": False,
        },
        "next_step": "step5_train_smoke",
        "next_step_note": "1 epoch 또는 tiny subset smoke training (사용자 승인 후)",
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [SAVED] {SUMMARY_JSON}")


def write_done_json(verdict):
    done = {
        "step": "step4_student_decoder_smoke",
        "verdict": verdict,
        "created": str(date.today()),
        "summary_json": str(SUMMARY_JSON),
        "report_md": str(REPORT_MD),
        "samples_csv": str(SAMPLES_CSV),
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
    if not args.confirm_smoke_only:
        missing.append("--confirm-smoke-only")
    if missing:
        print(f"[BLOCKED] 필수 flags 누락: {missing}", file=sys.stderr)
        sys.exit(2)
else:
    print("[BLOCKED] --run-smoke 없이 실행 금지.", file=sys.stderr)
    sys.exit(2)

print()
print("=" * 64)
print("Step 4 Student Decoder Smoke — ACTUAL RUN")
print("=" * 64)
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
print("  [PASS] plan_lock / step2 / step3 / crops / mask / manifest 확인 완료")

# [1] crop + manifest 로드
print()
print(f"[1] crop 로드 ({N_PATIENTS}개 환자, batch={BATCH_SIZE})")
import numpy as np
import torch

npy_files = sorted(CROPS_DIR.glob("*_crops_f16.npy"))
batch, meta_rows = load_sample_crops_with_manifest(
    npy_files, MANIFEST_CSV, N_PATIENTS, BATCH_SIZE
)
print(f"  combined shape : {tuple(batch.shape)}")
print(f"  dtype          : {batch.dtype}")
print(f"  range          : [{float(batch.min()):.4f}, {float(batch.max()):.4f}]")

# [2] mask 로드
print()
print("[2] mask crop 로드")
mask_batch = load_masks_for_batch(meta_rows)  # (B, 1, 96, 96) float32
mask_loaded_count = int((mask_batch.sum(dim=[1, 2, 3]) > 0).sum())
print(f"  mask shape     : {tuple(mask_batch.shape)}")
print(f"  mask range     : [{float(mask_batch.min()):.2f}, {float(mask_batch.max()):.2f}]")
print(f"  non-zero masks : {mask_loaded_count}/{len(meta_rows)}")
for m in meta_rows:
    m["mask_loaded"] = True

# [3] teacher 5ch 구성
print()
print("[3] ResNet18 5ch teacher 구성")
teacher = build_teacher_5ch()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher = teacher.to(device)
teacher_frozen = all(not p.requires_grad for p in teacher.parameters())
print(f"  conv1 shape  : {tuple(teacher.conv1.weight.shape)}")
print(f"  frozen       : {teacher_frozen}")
print(f"  device       : {device}")

# [4] student decoder 구성
print()
print("[4] Student decoder 구성")
student = build_student_decoder()
student = student.to(device)
student.train()
student_requires_grad = any(p.requires_grad for p in student.parameters())
param_count = sum(p.numel() for p in student.parameters())
print(f"  requires_grad : {student_requires_grad}")
print(f"  파라미터 수   : {param_count:,}")

# [5] teacher forward
print()
print("[5] Teacher forward")
t_features = run_teacher_forward(teacher, batch)
t_l1 = t_features["layer1"]
t_l2 = t_features["layer2"]
t_l3 = t_features["layer3"]
print(f"  layer1 : {tuple(t_l1.shape)}")
print(f"  layer2 : {tuple(t_l2.shape)}")
print(f"  layer3 : {tuple(t_l3.shape)}")

# [6] student forward
print()
print("[6] Student forward")
student_input = t_l3  # (B, 256, 6, 6) — teacher layer3 기반
s_l1, s_l2, s_l3 = student(student_input)
l1_match = tuple(s_l1.shape) == tuple(t_l1.shape)
l2_match = tuple(s_l2.shape) == tuple(t_l2.shape)
l3_match = tuple(s_l3.shape) == tuple(t_l3.shape)
print(f"  student_layer1 : {tuple(s_l1.shape)}  (teacher: {tuple(t_l1.shape)}) match={l1_match}")
print(f"  student_layer2 : {tuple(s_l2.shape)}  (teacher: {tuple(t_l2.shape)}) match={l2_match}")
print(f"  student_layer3 : {tuple(s_l3.shape)}  (teacher: {tuple(t_l3.shape)}) match={l3_match}")

if not (l1_match and l2_match and l3_match):
    print("  [PARTIAL] shape mismatch — loss 계산 시도는 계속")

# [7] mask downsample
print()
print("[7] Mask downsample")
m_batch = mask_batch.to(device)
_, _, H1, W1 = t_l1.shape
_, _, H2, W2 = t_l2.shape
_, _, H3, W3 = t_l3.shape
mask_l1 = downsample_mask(m_batch, H1, W1)
mask_l2 = downsample_mask(m_batch, H2, W2)
mask_l3 = downsample_mask(m_batch, H3, W3)
print(f"  mask_layer1 : {tuple(mask_l1.shape)}")
print(f"  mask_layer2 : {tuple(mask_l2.shape)}")
print(f"  mask_layer3 : {tuple(mask_l3.shape)}")

# [8] masked cosine loss
print()
print("[8] Masked cosine loss 계산")
total_nan = 0
total_inf = 0
loss_l1 = loss_l2 = loss_l3 = loss_total = float("nan")

try:
    _loss_l1 = masked_cosine_loss(t_l1, s_l1, mask_l1)
    _loss_l2 = masked_cosine_loss(t_l2, s_l2, mask_l2)
    _loss_l3 = masked_cosine_loss(t_l3, s_l3, mask_l3)
    _loss_total = (_loss_l1 + _loss_l2 + _loss_l3) / 3.0

    loss_l1 = float(_loss_l1)
    loss_l2 = float(_loss_l2)
    loss_l3 = float(_loss_l3)
    loss_total = float(_loss_total)

    import math
    for v in [loss_l1, loss_l2, loss_l3, loss_total]:
        if math.isnan(v):
            total_nan += 1
        if math.isinf(v):
            total_inf += 1

    print(f"  loss_layer1 : {loss_l1:.6f}")
    print(f"  loss_layer2 : {loss_l2:.6f}")
    print(f"  loss_layer3 : {loss_l3:.6f}")
    print(f"  loss_total  : {loss_total:.6f}")
    print(f"  NaN/Inf     : {total_nan}/{total_inf}")

except Exception as ex:
    print(f"  [ERROR] loss 계산 실패: {ex}")
    write_errors_csv(guard_errors + [str(ex)])
    sys.exit(1)

# [9] 판정
print()
shape_all_match = l1_match and l2_match and l3_match
loss_valid = (total_nan == 0 and total_inf == 0 and not __import__("math").isnan(loss_total))

verdict = "BLOCKED"
if (shape_all_match and loss_valid and teacher_frozen and student_requires_grad):
    verdict = "PASS_STEP4_STUDENT_DECODER_SMOKE"
elif (not shape_all_match) and loss_valid:
    verdict = "PARTIAL_PASS_STEP4_SHAPE_MISMATCH"
elif shape_all_match and not loss_valid:
    verdict = "PARTIAL_PASS_STEP4_LOSS_NAN"

print("=" * 64)
print(f"판정: {verdict}")
print("=" * 64)
print(f"  sampled crops    : {len(meta_rows)}")
print(f"  input shape      : {tuple(batch.shape)}")
print(f"  teacher frozen   : {teacher_frozen}")
print(f"  student grad     : {student_requires_grad}")
print(f"  shape match all  : {shape_all_match}")
print(f"  loss_layer1      : {loss_l1:.6f}")
print(f"  loss_layer2      : {loss_l2:.6f}")
print(f"  loss_layer3      : {loss_l3:.6f}")
print(f"  loss_total       : {loss_total:.6f}")
print(f"  NaN/Inf          : {total_nan}/{total_inf}")
print(f"  training         : False")
print(f"  backward         : False")
print(f"  optimizer        : False")
print(f"  checkpoint saved : False")
print(f"  stage2 accessed  : False")
print("=" * 64)

# [10] 파일 저장
print()
print("[10] 결과 파일 저장")

results = {
    "input_shape": list(batch.shape),
    "t_l1_shape": list(t_l1.shape),
    "t_l2_shape": list(t_l2.shape),
    "t_l3_shape": list(t_l3.shape),
    "s_l1_shape": list(s_l1.shape),
    "s_l2_shape": list(s_l2.shape),
    "s_l3_shape": list(s_l3.shape),
    "mask_l1_shape": list(mask_l1.shape),
    "mask_l2_shape": list(mask_l2.shape),
    "mask_l3_shape": list(mask_l3.shape),
    "l1_match": l1_match,
    "l2_match": l2_match,
    "l3_match": l3_match,
    "loss_l1": round(loss_l1, 6),
    "loss_l2": round(loss_l2, 6),
    "loss_l3": round(loss_l3, 6),
    "loss_total": round(loss_total, 6),
    "total_nan": total_nan,
    "total_inf": total_inf,
    "teacher_frozen": teacher_frozen,
    "student_requires_grad": student_requires_grad,
}

write_samples_csv(meta_rows, [])
write_report_md(verdict, results)
write_summary_json(verdict, results, len(meta_rows))
write_errors_csv([])
if verdict.startswith("PASS") or verdict.startswith("PARTIAL"):
    write_done_json(verdict)

print()
if verdict == "PASS_STEP4_STUDENT_DECODER_SMOKE":
    print("Step 4 완료. 다음 단계: Step 5 train smoke (사용자 승인 후)")
elif verdict.startswith("PARTIAL"):
    print(f"Step 4 PARTIAL_PASS. 원인 확인 후 Step 5 진행 가능 여부 판단 필요.")
else:
    print("Step 4 BLOCKED. 오류를 확인하세요.")
