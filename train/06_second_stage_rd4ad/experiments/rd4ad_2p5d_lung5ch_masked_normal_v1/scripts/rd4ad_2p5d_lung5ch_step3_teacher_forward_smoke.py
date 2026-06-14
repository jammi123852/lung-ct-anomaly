"""
Step 3: Teacher (ResNet18 5ch) Forward Smoke
rd4ad_2p5d_lung5ch_masked_normal_v1

bare run → exit 2
dry-run  → plan 출력, 실행 없음
actual   → --run-smoke --confirm-plan-lock --confirm-no-stage2 --confirm-forward-only
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
CROPS_DIR = ROOT / "crops" / "normal_5ch_lung_w96_v1"
MANIFESTS_DIR = ROOT / "manifests"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
DONE_STEP2_JSON = ROOT / "DONE_STEP2_CROP_FULL_BUILD.json"
PLAN_LOCK_JSON = ROOT / "docs" / "FINAL_PLAN_LOCK.json"
DONE_OUT = ROOT / "DONE_STEP3_TEACHER_FORWARD_SMOKE.json"

SAMPLES_CSV = MANIFESTS_DIR / "step3_teacher_forward_smoke_samples.csv"
REPORT_MD = REPORTS_DIR / "step3_teacher_forward_smoke_report.md"
SUMMARY_JSON = REPORTS_DIR / "step3_teacher_forward_smoke_summary.json"
ERRORS_CSV = LOGS_DIR / "step3_teacher_forward_smoke_errors.csv"

# ── 상수 ─────────────────────────────────────────────────────────────────────
N_PATIENTS = 5
BATCH_SIZE = 8
INPUT_CHANNELS = 5
CROP_SIZE = 96
EXPECTED_CROP_COUNT = 46254
EXPECTED_DTYPE = "float16"

# ── argparse ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--run-smoke", action="store_true")
parser.add_argument("--confirm-plan-lock", action="store_true")
parser.add_argument("--confirm-no-stage2", action="store_true")
parser.add_argument("--confirm-forward-only", action="store_true")
args = parser.parse_args()


def print_dry_run_plan():
    print()
    print("=" * 64)
    print("Step 3 Teacher Forward Smoke — DRY-RUN PLAN")
    print("=" * 64)
    print()
    print("[대상 파일]")
    print(f"  crops dir      : {CROPS_DIR}")
    print(f"  plan lock json : {PLAN_LOCK_JSON}")
    print(f"  step2 done     : {DONE_STEP2_JSON}")
    print()
    print("[모델]")
    print(f"  backbone       : ResNet18 (pretrained ImageNet)")
    print(f"  conv1 5ch      : (64,3,7,7) → mean RGB → repeat×5 → scale 3/5")
    print(f"  teacher mode   : eval + frozen + torch.no_grad()")
    print(f"  hook layers    : layer1, layer2, layer3")
    print()
    print("[smoke 샘플]")
    print(f"  환자 파일      : {N_PATIENTS}개")
    print(f"  batch size     : {BATCH_SIZE}")
    print(f"  총 crops       : ~{N_PATIENTS * BATCH_SIZE} crops")
    print()
    print("[생성 파일]")
    print(f"  {SAMPLES_CSV}")
    print(f"  {REPORT_MD}")
    print(f"  {SUMMARY_JSON}")
    print(f"  {ERRORS_CSV}")
    print(f"  {DONE_OUT}")
    print()
    print("[금지 항목]")
    print("  student 생성 금지  | training 금지  | backward 금지")
    print("  optimizer 금지     | checkpoint 저장 금지  | stage2 접근 금지")
    print()
    print("[실행 명령]")
    print("  python " + str(Path(__file__).relative_to(ROOT.parent.parent)) + " \\")
    print("    --run-smoke --confirm-plan-lock --confirm-no-stage2 --confirm-forward-only")
    print()
    print("DRY-RUN 완료. 위 내용 확인 후 actual 실행.")
    print()


def check_guards():
    errors = []

    # plan lock
    if not PLAN_LOCK_JSON.exists():
        errors.append(f"FINAL_PLAN_LOCK.json 없음: {PLAN_LOCK_JSON}")
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

    # step2 done
    if not DONE_STEP2_JSON.exists():
        errors.append(f"DONE_STEP2_CROP_FULL_BUILD.json 없음")
    else:
        with open(DONE_STEP2_JSON) as f:
            s2 = json.load(f)
        if s2.get("verdict") != "PASS_STEP2_CROP_FULL_BUILD":
            errors.append(f"Step2 verdict != PASS: {s2.get('verdict')}")
        if s2.get("total_crops") != EXPECTED_CROP_COUNT:
            errors.append(f"crop count 불일치: {s2.get('total_crops')} != {EXPECTED_CROP_COUNT}")

    # crops dir
    if not CROPS_DIR.exists():
        errors.append(f"crops dir 없음: {CROPS_DIR}")
    else:
        npy_files = sorted(CROPS_DIR.glob("*_crops_f16.npy"))
        if len(npy_files) < 300:
            errors.append(f"npy 파일 수 부족: {len(npy_files)}")

    return errors


def load_sample_crops(npy_files, n_patients, batch_size):
    """float16 npy에서 crops 로드 → float32 tensor 반환"""
    import numpy as np
    import torch

    selected = npy_files[:n_patients]
    batches = []
    meta_rows = []

    for fp in selected:
        arr = np.load(str(fp), mmap_mode="r")  # (N, 5, 96, 96) float16
        N = arr.shape[0]
        take = min(batch_size, N)
        chunk = arr[:take].astype(np.float32)
        t = torch.from_numpy(chunk)  # (take, 5, 96, 96) float32
        batches.append(t)

        for i in range(take):
            meta_rows.append({
                "patient_file": fp.name,
                "crop_idx": i,
                "orig_dtype": str(arr.dtype),
                "tensor_dtype": str(t.dtype),
                "shape": str(tuple(t[i].shape)),
                "vmin": float(t[i].min()),
                "vmax": float(t[i].max()),
                "has_nan": bool(torch.isnan(t[i]).any()),
                "has_inf": bool(torch.isinf(t[i]).any()),
            })

    combined = torch.cat(batches, dim=0)  # (total, 5, 96, 96) float32
    return combined, meta_rows


def build_teacher_5ch():
    """ResNet18 pretrained + conv1 3ch→5ch inflation"""
    import torch
    import torch.nn as nn
    import torchvision.models as models

    teacher = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # conv1 inflation: (64,3,7,7) → (64,5,7,7)
    old_conv1 = teacher.conv1
    old_w = old_conv1.weight.data  # (64, 3, 7, 7)
    mean_w = old_w.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
    new_w = mean_w.repeat(1, 5, 1, 1) * (3.0 / 5.0)  # (64, 5, 7, 7)

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

    # frozen
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    return teacher


def run_teacher_forward(teacher, batch):
    """hook으로 layer1/layer2/layer3 feature 수집"""
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


def check_features(features):
    rows = []
    total_nan = 0
    total_inf = 0
    import torch

    for name, feat in features.items():
        n = int(torch.isnan(feat).sum())
        i = int(torch.isinf(feat).sum())
        total_nan += n
        total_inf += i
        rows.append({
            "layer": name,
            "shape": str(tuple(feat.shape)),
            "dtype": str(feat.dtype),
            "vmin": float(feat.min()),
            "vmax": float(feat.max()),
            "nan_count": n,
            "inf_count": i,
        })

    return rows, total_nan, total_inf


def check_teacher_frozen(teacher):
    import torch
    grad_params = [(n, p.requires_grad) for n, p in teacher.named_parameters()]
    frozen = all(not rg for _, rg in grad_params)
    return frozen, grad_params[:3]


def write_samples_csv(meta_rows):
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    if not meta_rows:
        return
    fields = list(meta_rows[0].keys())
    with open(SAMPLES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(meta_rows)
    print(f"  [SAVED] {SAMPLES_CSV} ({len(meta_rows)} rows)")


def write_errors_csv(errors):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ERRORS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "msg"])
        w.writeheader()
        for e in errors:
            w.writerow({"step": "guard", "msg": e})
    print(f"  [SAVED] {ERRORS_CSV}")


def write_report_md(verdict, batch_shape, conv1_shape, feat_rows, total_nan, total_inf,
                    teacher_frozen, meta_rows, guard_errors):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Step 3 Teacher Forward Smoke Report",
        f"",
        f"- **판정**: {verdict}",
        f"- **생성일**: {date.today()}",
        f"",
        f"## 입력 crop",
        f"",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| 배치 shape | {batch_shape} |",
        f"| dtype (torch) | float32 |",
        f"| 원본 dtype | float16 |",
        f"| NaN/Inf (입력) | {sum(r['has_nan'] for r in meta_rows)} / {sum(r['has_inf'] for r in meta_rows)} |",
        f"",
        f"## conv1 inflation",
        f"",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| conv1 weight shape | {conv1_shape} |",
        f"| inflation 방식 | mean RGB → ×5 → scale 3/5 |",
        f"",
        f"## feature hook 결과",
        f"",
        f"| layer | shape | dtype | vmin | vmax | NaN | Inf |",
        f"|---|---|---|---|---|---|---|",
    ]
    for r in feat_rows:
        lines.append(
            f"| {r['layer']} | {r['shape']} | {r['dtype']} "
            f"| {r['vmin']:.4f} | {r['vmax']:.4f} "
            f"| {r['nan_count']} | {r['inf_count']} |"
        )
    lines += [
        f"",
        f"## guardrail",
        f"",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| teacher_frozen | {teacher_frozen} |",
        f"| total_nan | {total_nan} |",
        f"| total_inf | {total_inf} |",
        f"| training_executed | False |",
        f"| backward_executed | False |",
        f"| optimizer_created | False |",
        f"| checkpoint_saved | False |",
        f"| stage2_holdout_accessed | False |",
        f"",
    ]
    if guard_errors:
        lines += ["## guard 오류", ""]
        for e in guard_errors:
            lines.append(f"- {e}")
        lines.append("")

    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"  [SAVED] {REPORT_MD}")


def write_summary_json(verdict, batch_shape, conv1_shape, feat_rows, total_nan, total_inf,
                       teacher_frozen, sampled_crop_count):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "verdict": verdict,
        "created": str(date.today()),
        "branch_name": "rd4ad_2p5d_lung5ch_masked_normal_v1",
        "step": "step3_teacher_forward_smoke",
        "sampled_crop_count": sampled_crop_count,
        "input_tensor_shape": list(batch_shape) if hasattr(batch_shape, "__iter__") else batch_shape,
        "input_dtype_before": "float16",
        "input_dtype_after": "float32",
        "conv1_weight_shape": conv1_shape,
        "feature_layers": {r["layer"]: r["shape"] for r in feat_rows},
        "total_nan_in_features": total_nan,
        "total_inf_in_features": total_inf,
        "teacher_frozen": teacher_frozen,
        "guardrail": {
            "plan_lock_loaded": True,
            "step2_crop_full_build_passed": True,
            "crop_count": EXPECTED_CROP_COUNT,
            "crop_dtype": "float16",
            "center_z_sampling": "stride2",
            "input_channels": INPUT_CHANNELS,
            "input_window": "lung",
            "crop_size": CROP_SIZE,
            "teacher_backbone": "resnet18",
            "conv1_in_channels": 5,
            "conv1_inflation_used": True,
            "teacher_forward_executed": True,
            "teacher_frozen": teacher_frozen,
            "student_created": False,
            "training_executed": False,
            "backward_executed": False,
            "optimizer_created": False,
            "checkpoint_saved": False,
            "stage2_holdout_accessed": False,
            "existing_artifact_modified": False,
            "convae_branch_created": False,
        },
        "next_step": "step4_student_decoder_smoke",
        "next_step_note": "Student decoder smoke (사용자 승인 후)",
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [SAVED] {SUMMARY_JSON}")


def write_done_json(verdict):
    done = {
        "step": "step3_teacher_forward_smoke",
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

# actual smoke: 필수 flags 확인
if args.run_smoke:
    missing = []
    if not args.confirm_plan_lock:
        missing.append("--confirm-plan-lock")
    if not args.confirm_no_stage2:
        missing.append("--confirm-no-stage2")
    if not args.confirm_forward_only:
        missing.append("--confirm-forward-only")
    if missing:
        print(f"[BLOCKED] 필수 flags 누락: {missing}", file=sys.stderr)
        sys.exit(2)
else:
    print("[BLOCKED] --run-smoke 없이 실행 금지.", file=sys.stderr)
    sys.exit(2)

print()
print("=" * 64)
print("Step 3 Teacher Forward Smoke — ACTUAL RUN")
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
print("  [PASS] plan_lock / step2 / crops 확인 완료")

# [1] crop 로드
print()
print(f"[1] crop 로드 ({N_PATIENTS}개 환자, batch={BATCH_SIZE})")
import numpy as np

npy_files = sorted(CROPS_DIR.glob("*_crops_f16.npy"))
batch, meta_rows = load_sample_crops(npy_files, N_PATIENTS, BATCH_SIZE)
print(f"  combined shape : {tuple(batch.shape)}")
print(f"  dtype (torch)  : {batch.dtype}")
print(f"  range          : [{float(batch.min()):.4f}, {float(batch.max()):.4f}]")
print(f"  NaN            : {int((batch != batch).sum())}")
print(f"  Inf            : {int(batch.isinf().sum())}")
write_samples_csv(meta_rows)

# [2] teacher 5ch 구성
print()
print("[2] ResNet18 5ch teacher 구성")
teacher = build_teacher_5ch()
conv1_w = teacher.conv1.weight.data
conv1_shape = str(tuple(conv1_w.shape))
print(f"  conv1 weight shape : {conv1_shape}")
teacher_frozen, sample_params = check_teacher_frozen(teacher)
print(f"  teacher frozen     : {teacher_frozen}")
for n, rg in sample_params:
    print(f"    {n}: requires_grad={rg}")

# cuda 이동 (가능하면)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device : {device}")
teacher = teacher.to(device)

# [3] forward
print()
print("[3] Teacher forward (torch.no_grad)")
feat_rows, total_nan, total_inf = [], 0, 0
forward_ok = False
try:
    features = run_teacher_forward(teacher, batch)
    feat_rows, total_nan, total_inf = check_features(features)
    forward_ok = True
    for r in feat_rows:
        print(f"  {r['layer']:8s}  shape={r['shape']:22s}  NaN={r['nan_count']}  Inf={r['inf_count']}")
except Exception as ex:
    print(f"  [ERROR] teacher forward 실패: {ex}")
    write_errors_csv(guard_errors + [str(ex)])
    sys.exit(1)

# [4] 판정
print()
verdict = "BLOCKED"
if (
    forward_ok
    and total_nan == 0
    and total_inf == 0
    and teacher_frozen
    and len(feat_rows) == 3
):
    verdict = "PASS_STEP3_TEACHER_FORWARD_SMOKE"

print("=" * 64)
print(f"판정: {verdict}")
print("=" * 64)
print(f"  sampled crops      : {len(meta_rows)}")
print(f"  input shape        : {tuple(batch.shape)}")
print(f"  conv1 weight shape : {conv1_shape}")
for r in feat_rows:
    print(f"  {r['layer']:8s} shape : {r['shape']}")
print(f"  NaN/Inf (features) : {total_nan} / {total_inf}")
print(f"  teacher frozen     : {teacher_frozen}")
print(f"  training           : False")
print(f"  backward           : False")
print(f"  optimizer          : False")
print(f"  checkpoint saved   : False")
print(f"  stage2 accessed    : False")
print("=" * 64)

# [5] 파일 저장
print()
print("[5] 결과 파일 저장")
write_report_md(verdict, tuple(batch.shape), conv1_shape, feat_rows, total_nan, total_inf,
                teacher_frozen, meta_rows, guard_errors)
write_summary_json(verdict, list(batch.shape), conv1_shape, feat_rows, total_nan, total_inf,
                   teacher_frozen, len(meta_rows))
write_errors_csv([])
if verdict.startswith("PASS"):
    write_done_json(verdict)

print()
if verdict.startswith("PASS"):
    print("Step 3 완료. 다음 단계: Step 4 student decoder smoke (사용자 승인 후)")
else:
    print("Step 3 BLOCKED. 오류를 확인하세요.")
