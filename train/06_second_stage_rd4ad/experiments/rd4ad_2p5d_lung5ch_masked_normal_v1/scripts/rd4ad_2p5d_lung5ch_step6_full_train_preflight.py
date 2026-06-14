"""
Step 6: Full Training Preflight
rd4ad_2p5d_lung5ch_masked_normal_v1

bare run → exit 2
dry-run  → 계획 출력, 실행 없음
actual   → --run-preflight --confirm-plan-lock --confirm-no-stage2 --confirm-preflight-only

허용: Dataset/DataLoader 로드 테스트, GPU memory probe (3 micro steps), config 생성
금지: full training, checkpoint 저장, stage2 접근, 기존 artifact 수정
"""

import sys
import os
import json
import csv
import math
import time
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
MASK_ROOT = (PROJECT_ROOT / "outputs" / "mip-postprocess-research-v1"
             / "masks" / "refined_roi_v4_20_modeB_all_v1" / "normal")
MANIFESTS_DIR = ROOT / "manifests"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
CONFIGS_DIR = ROOT / "configs"

MANIFEST_CSV = MANIFESTS_DIR / "step2_crop_build_manifest.csv"
PLAN_LOCK_JSON = ROOT / "docs" / "FINAL_PLAN_LOCK.json"
DONE_STEP2_JSON = ROOT / "DONE_STEP2_CROP_FULL_BUILD.json"
DONE_STEP3_JSON = ROOT / "DONE_STEP3_TEACHER_FORWARD_SMOKE.json"
DONE_STEP4_JSON = ROOT / "DONE_STEP4_STUDENT_DECODER_SMOKE.json"
DONE_STEP5_JSON = ROOT / "DONE_STEP5_TRAIN_SMOKE.json"
DONE_OUT = ROOT / "DONE_STEP6_FULL_TRAIN_PREFLIGHT.json"

SPLIT_CSV = MANIFESTS_DIR / "step6_train_val_split_manifest.csv"
DL_BENCH_CSV = MANIFESTS_DIR / "step6_dataloader_benchmark.csv"
GPU_PROBE_CSV = MANIFESTS_DIR / "step6_gpu_memory_probe.csv"
CONFIG_YAML = CONFIGS_DIR / "rd4ad_2p5d_lung5ch_full_train_v1.yaml"
REPORT_MD = REPORTS_DIR / "step6_full_train_preflight_report.md"
SUMMARY_JSON = REPORTS_DIR / "step6_full_train_preflight_summary.json"
ERRORS_CSV = LOGS_DIR / "step6_full_train_preflight_errors.csv"

# full train checkpoint 경로 (생성 예정, 아직 저장 금지)
FULL_TRAIN_CKPT_DIR = ROOT / "checkpoints" / "full_train_v1"
SMOKE_CKPT_DIR = ROOT / "checkpoints" / "train_smoke_v1"

CROP_SIZE = 96
INPUT_CHANNELS = 5
EXPECTED_CROP_COUNT = 46254
EXPECTED_PATIENT_COUNT = 362

# ── argparse ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--run-preflight", action="store_true")
parser.add_argument("--confirm-plan-lock", action="store_true")
parser.add_argument("--confirm-no-stage2", action="store_true")
parser.add_argument("--confirm-preflight-only", action="store_true")
parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 24, 32])
parser.add_argument("--num-workers-list", type=int, nargs="+", default=[0, 2, 4])
parser.add_argument("--probe-steps", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


def print_dry_run_plan():
    print()
    print("=" * 64)
    print("Step 6 Full Train Preflight — DRY-RUN PLAN")
    print("=" * 64)
    print()
    print("[확인 항목]")
    print("  1. Plan lock (true_rd4ad, 5ch, normal-only, v4_20 mask)")
    print("  2. Full crop dataset (46,254 crops, 362 patients)")
    print("  3. Train/val patient-level split (90/10, seed=42)")
    print("  4. DataLoader benchmark")
    print("     batch_sizes  :", args.batch_sizes)
    print("     num_workers  :", args.num_workers_list)
    print("  5. GPU memory probe (micro steps=3)")
    print("     batch_sizes  :", args.batch_sizes)
    print("  6. Full train config 생성")
    print("  7. Checkpoint 경로 충돌 확인")
    print("  8. Runtime 추정")
    print()
    print("[금지]")
    print("  full training 금지 | checkpoint 저장 금지 | stage2 접근 금지")
    print()
    print("[생성 파일]")
    for p in [SPLIT_CSV, DL_BENCH_CSV, GPU_PROBE_CSV, CONFIG_YAML,
              REPORT_MD, SUMMARY_JSON, ERRORS_CSV, DONE_OUT]:
        print(f"  {p}")
    print()
    rel = Path("experiments/rd4ad_2p5d_lung5ch_masked_normal_v1/scripts") / Path(__file__).name
    print("[실행 명령]")
    print(f"  python {rel} \\")
    print(f"    --run-preflight --confirm-plan-lock --confirm-no-stage2 --confirm-preflight-only \\")
    bs = " ".join(map(str, args.batch_sizes))
    nw = " ".join(map(str, args.num_workers_list))
    print(f"    --batch-sizes {bs} --num-workers-list {nw} "
          f"--probe-steps {args.probe_steps} --seed {args.seed}")
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
        if lock.get("training", {}).get("training_data") not in (None, "normal_only"):
            errors.append("training_data not normal_only")

    step_checks = [
        ("step2", DONE_STEP2_JSON, "PASS_STEP2_CROP_FULL_BUILD"),
        ("step3", DONE_STEP3_JSON, "PASS_STEP3_TEACHER_FORWARD_SMOKE"),
        ("step4", DONE_STEP4_JSON, "PASS_STEP4_STUDENT_DECODER_SMOKE"),
        ("step5", DONE_STEP5_JSON, "PASS_STEP5_TRAIN_SMOKE"),
    ]
    for label, path, expected in step_checks:
        if not path.exists():
            errors.append(f"{label} done JSON 없음")
        else:
            with open(path) as f:
                d = json.load(f)
            if d.get("verdict") != expected:
                errors.append(f"{label} verdict 불일치: {d.get('verdict')}")

    if not CROPS_DIR.exists():
        errors.append("crops dir 없음")
    if not MASK_ROOT.exists():
        errors.append("mask root 없음")
    if not MANIFEST_CSV.exists():
        errors.append("manifest CSV 없음")

    # full train checkpoint 경로 collision
    if FULL_TRAIN_CKPT_DIR.exists():
        existing = list(FULL_TRAIN_CKPT_DIR.glob("*.pth"))
        if existing:
            errors.append(f"full_train_v1 checkpoint 폴더에 기존 .pth 파일 존재: {len(existing)}개")

    return errors


# ── Dataset ──────────────────────────────────────────────────────────────────
def read_manifest():
    import csv as csv_mod
    rows = []
    with open(MANIFEST_CSV, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def make_split(all_rows, seed):
    """Patient-level train/val split (90/10)"""
    import random
    # unique patients
    patient_info = {}
    for row in all_rows:
        sid = row["safe_id"]
        if sid not in patient_info:
            patient_info[sid] = {"n_crops": 0, "crop_file": Path(row["file_path"]).name}
        patient_info[sid]["n_crops"] += 1

    patients = sorted(patient_info.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_val = max(1, round(len(patients) * 0.10))
    val_set = set(patients[:n_val])
    train_set = set(patients[n_val:])

    split_rows = []
    for i, sid in enumerate(patients):
        split = "val" if sid in val_set else "train"
        split_rows.append({
            "safe_id": sid,
            "split": split,
            "n_crops": patient_info[sid]["n_crops"],
            "crop_file": patient_info[sid]["crop_file"],
            "patient_index": i,
            "seed": seed,
        })

    train_rows = [r for r in all_rows if r["safe_id"] in train_set]
    val_rows = [r for r in all_rows if r["safe_id"] in val_set]

    # overlap 검증
    overlap = train_set & val_set
    return split_rows, train_rows, val_rows, train_set, val_set, len(overlap)


class CropDataset:
    def __init__(self, rows):
        import torch
        self._rows = rows
        self._torch = torch

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        import numpy as np
        row = self._rows[idx]
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

        return self._torch.from_numpy(crop), self._torch.from_numpy(m)[None]


def make_torch_dataset(rows):
    import torch
    import torch.utils.data as data

    class _D(data.Dataset):
        def __init__(self, r):
            self.r = r

        def __len__(self):
            return len(self.r)

        def __getitem__(self, idx):
            import numpy as np
            row = self.r[idx]
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
                h2, w2 = min(m.shape[0], CROP_SIZE), min(m.shape[1], CROP_SIZE)
                m2[:h2, :w2] = m[:h2, :w2]
                m = m2

            return torch.from_numpy(crop), torch.from_numpy(m)[None]

    return _D(rows)


# ── 모델 빌더 ─────────────────────────────────────────────────────────────────
def build_temp_teacher(device):
    import torch
    import torch.nn as nn
    import torchvision.models as models

    teacher = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    old = teacher.conv1
    w = old.weight.data.mean(dim=1, keepdim=True).repeat(1, 5, 1, 1) * (3.0 / 5.0)
    new_c = nn.Conv2d(5, old.out_channels, old.kernel_size, old.stride, old.padding,
                      bias=(old.bias is not None))
    new_c.weight = nn.Parameter(w)
    if old.bias is not None:
        new_c.bias = nn.Parameter(old.bias.data.clone())
    teacher.conv1 = new_c
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher.eval().to(device)


def build_temp_student(device):
    import torch.nn as nn

    class _S(nn.Module):
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

    return _S().to(device)


def teacher_feats(teacher, x):
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


def cos_loss_masked(t, s, mask, eps=1e-6):
    import torch.nn.functional as F
    err = 1.0 - F.cosine_similarity(t, s, dim=1, eps=eps)
    m = mask.squeeze(1)
    return (err * m).sum() / (m.sum() + eps)


def downsample_m(m, h, w):
    import torch.nn.functional as F
    return F.interpolate(m, size=(h, w), mode="nearest")


# ── [3] train/val split ───────────────────────────────────────────────────────
def section_split(all_rows, seed):
    split_rows, train_rows, val_rows, train_set, val_set, overlap = make_split(all_rows, seed)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["safe_id", "split", "n_crops", "crop_file", "patient_index", "seed"]
    with open(SPLIT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(split_rows)
    print(f"  train patients : {len(train_set)}")
    print(f"  val patients   : {len(val_set)}")
    print(f"  train crops    : {len(train_rows)}")
    print(f"  val crops      : {len(val_rows)}")
    print(f"  overlap        : {overlap}")
    print(f"  [SAVED] {SPLIT_CSV}")
    return train_rows, val_rows, train_set, val_set, overlap


# ── [4] DataLoader benchmark ──────────────────────────────────────────────────
def section_dataloader_bench(train_rows, batch_sizes, num_workers_list, n_batches=10):
    import torch.utils.data as data
    import torch

    bench_rows = []
    # 벤치마크용 subset (처음 2048개 rows)
    bench_subset = train_rows[:2048]
    dataset = make_torch_dataset(bench_subset)

    for bs in batch_sizes:
        for nw in num_workers_list:
            try:
                loader = data.DataLoader(
                    dataset, batch_size=bs, shuffle=True, drop_last=True,
                    num_workers=nw, pin_memory=(nw > 0),
                    persistent_workers=(nw > 0),
                )
                t0 = time.perf_counter()
                actual = 0
                for cb, mb in loader:
                    actual += 1
                    # 기본 shape/dtype 확인
                    assert cb.shape == torch.Size([bs, 5, 96, 96]), f"crop shape mismatch: {cb.shape}"
                    assert cb.dtype == torch.float32
                    assert float(cb.min()) >= 0.0 and float(cb.max()) <= 1.0
                    assert mb.shape == torch.Size([bs, 1, 96, 96])
                    assert mb.dtype == torch.float32
                    if actual >= n_batches:
                        break
                elapsed = time.perf_counter() - t0
                time_per_batch = elapsed / actual
                throughput = bs * actual / elapsed
                status = "OK"
                msg = ""
            except Exception as ex:
                time_per_batch = float("nan")
                throughput = float("nan")
                status = "ERROR"
                msg = str(ex)[:100]

            row = {
                "batch_size": bs,
                "num_workers": nw,
                "n_batches": n_batches,
                "time_per_batch_s": round(time_per_batch, 4) if not math.isnan(time_per_batch) else "nan",
                "throughput_crops_s": round(throughput, 1) if not math.isnan(throughput) else "nan",
                "status": status,
                "error_msg": msg,
            }
            bench_rows.append(row)
            if status == "OK":
                print(f"  bs={bs:2d} nw={nw}  {time_per_batch*1000:.1f}ms/batch  "
                      f"{throughput:.0f} crops/s  [OK]")
            else:
                print(f"  bs={bs:2d} nw={nw}  ERROR: {msg[:60]}")

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["batch_size", "num_workers", "n_batches", "time_per_batch_s",
              "throughput_crops_s", "status", "error_msg"]
    with open(DL_BENCH_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(bench_rows)
    print(f"  [SAVED] {DL_BENCH_CSV}")
    return bench_rows


# ── [5] GPU memory probe ──────────────────────────────────────────────────────
def section_gpu_probe(train_rows, batch_sizes, probe_steps, device):
    import torch
    import torch.utils.data as data
    import numpy as np

    probe_rows = []

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        try:
            # 임시 teacher + student 생성
            t = build_temp_teacher(device)
            s = build_temp_student(device)
            s.train()
            opt = torch.optim.AdamW(s.parameters(), lr=1e-4, weight_decay=1e-5)

            # 샘플 배치 로드
            subset = train_rows[:bs * probe_steps]
            dataset = make_torch_dataset(subset)
            loader = data.DataLoader(dataset, batch_size=bs, shuffle=False,
                                     drop_last=False, num_workers=0)

            step_times = []
            for cb, mb in loader:
                if len(step_times) >= probe_steps:
                    break
                cb = cb.to(device)
                mb = mb.to(device)

                t0 = time.perf_counter()
                tf = teacher_feats(t, cb)
                tl1, tl2, tl3 = tf["l1"], tf["l2"], tf["l3"]

                sl1, sl2, sl3 = s(tl3)

                ml1 = downsample_m(mb, tl1.shape[2], tl1.shape[3])
                ml2 = downsample_m(mb, tl2.shape[2], tl2.shape[3])
                ml3 = downsample_m(mb, tl3.shape[2], tl3.shape[3])

                loss = (cos_loss_masked(tl1, sl1, ml1)
                        + cos_loss_masked(tl2, sl2, ml2)
                        + cos_loss_masked(tl3, sl3, ml3)) / 3.0

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0)
                opt.step()
                step_times.append(time.perf_counter() - t0)

            peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
            avg_step_ms = sum(step_times) / len(step_times) * 1000 if step_times else float("nan")
            loss_val = float(loss)
            status = "OOM" if peak_mb > 8000 else "OK"
            msg = ""

            # 임시 모델 삭제 (checkpoint 저장 없음)
            del t, s, opt
            torch.cuda.empty_cache()

        except RuntimeError as ex:
            if "out of memory" in str(ex).lower():
                status = "OOM"
                peak_mb = -1.0
                avg_step_ms = float("nan")
                loss_val = float("nan")
                msg = "CUDA OOM"
            else:
                raise
        except Exception as ex:
            status = "ERROR"
            peak_mb = -1.0
            avg_step_ms = float("nan")
            loss_val = float("nan")
            msg = str(ex)[:100]

        row = {
            "batch_size": bs,
            "probe_steps": probe_steps,
            "peak_gpu_mb": round(peak_mb, 1),
            "avg_step_ms": round(avg_step_ms, 1) if not math.isnan(avg_step_ms) else "nan",
            "loss_sample": round(loss_val, 5) if not math.isnan(loss_val) else "nan",
            "status": status,
            "error_msg": msg,
        }
        probe_rows.append(row)
        if status in ("OK", "OOM"):
            print(f"  bs={bs:2d}  peak={peak_mb:.0f}MB  {avg_step_ms:.1f}ms/step  "
                  f"loss={loss_val:.4f}  [{status}]")
        else:
            print(f"  bs={bs:2d}  ERROR: {msg[:60]}")

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["batch_size", "probe_steps", "peak_gpu_mb", "avg_step_ms",
              "loss_sample", "status", "error_msg"]
    with open(GPU_PROBE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(probe_rows)
    print(f"  [SAVED] {GPU_PROBE_CSV}")
    return probe_rows


# ── [6] Config YAML 생성 ──────────────────────────────────────────────────────
def section_write_config(rec_bs, rec_nw, train_crop_count, val_crop_count):
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    import yaml

    cfg = {
        "branch": "rd4ad_2p5d_lung5ch_masked_normal_v1",
        "created": str(date.today()),
        "model": {
            "type": "true_rd4ad",
            "teacher": "resnet18",
            "conv1_channels": 5,
            "conv1_inflation": "imagenet_mean_repeat_scale_3_5",
            "teacher_frozen": True,
            "student": "mirror_decoder_ocbe_3layer",
        },
        "data": {
            "crop_dir": str(CROPS_DIR),
            "mask_root": str(MASK_ROOT),
            "crop_size": CROP_SIZE,
            "input_channels": INPUT_CHANNELS,
            "input_window": "lung",
            "crop_dtype_on_disk": "float16",
            "crop_dtype_in_model": "float32",
            "center_z_sampling": "stride2",
            "total_crops": EXPECTED_CROP_COUNT,
            "train_crops": train_crop_count,
            "val_crops": val_crop_count,
            "split_csv": str(SPLIT_CSV),
            "training_data": "normal_only",
        },
        "training": {
            "epochs": 30,
            "batch_size": rec_bs,
            "num_workers": rec_nw,
            "lr": 1e-4,
            "optimizer": "AdamW",
            "weight_decay": 1e-5,
            "scheduler": "CosineAnnealingLR",
            "scheduler_params": {"T_max": 30, "eta_min": 1e-6},
            "early_stopping_patience": 7,
            "grad_clip_max_norm": 1.0,
            "mixed_precision": False,
            "save_best_by": "val_loss",
        },
        "loss": {
            "type": "masked_cosine_feature_loss",
            "layers": ["layer1", "layer2", "layer3"],
            "mask_downsample": "nearest",
            "image_reconstruction_loss": False,
            "convae_branch": False,
        },
        "checkpoint": {
            "dir": str(FULL_TRAIN_CKPT_DIR),
            "best": "student_best_val_loss.pth",
            "last": "student_last.pth",
            "state": "training_state_last.pth",
            "loss_curve": "loss_curve.csv",
            "smoke_checkpoint_dir": str(SMOKE_CKPT_DIR),
            "overwrite_smoke": False,
        },
        "guardrail": {
            "stage2_holdout_accessed": False,
            "positive_label_used_for_training": False,
            "lesion_mask_used_for_training": False,
            "convae_branch_created": False,
            "image_reconstruction_loss_used": False,
        },
    }

    with open(CONFIG_YAML, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  [SAVED] {CONFIG_YAML}")
    return cfg


# ── 출력 파일 ─────────────────────────────────────────────────────────────────
def write_errors_csv(errors):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ERRORS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "msg"])
        w.writeheader()
        for e in errors:
            w.writerow({"step": "preflight", "msg": str(e)})
    print(f"  [SAVED] {ERRORS_CSV}")


def write_report_md(verdict, p):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # best OK batch sizes from probe
    probe_ok = [r for r in p["probe_rows"] if r["status"] == "OK"]
    probe_oom = [r["batch_size"] for r in p["probe_rows"] if r["status"] == "OOM"]
    bench_ok = [r for r in p["bench_rows"] if r["status"] == "OK"]

    lines = [
        "# Step 6 Full Train Preflight Report",
        "",
        f"- **판정**: {verdict}",
        f"- **생성일**: {date.today()}",
        "",
        "## Dataset",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| total crops | {p['total_crop_count']} |",
        f"| patient count | {p['patient_count']} |",
        f"| train patients | {p['n_train_patients']} |",
        f"| val patients | {p['n_val_patients']} |",
        f"| train crops | {p['n_train_crops']} |",
        f"| val crops | {p['n_val_crops']} |",
        f"| overlap | {p['overlap']} |",
        f"| split seed | {args.seed} |",
        "",
        "## DataLoader Benchmark",
        "",
        "| batch_size | num_workers | ms/batch | crops/s | status |",
        "|---|---|---|---|---|",
    ]
    for r in p["bench_rows"]:
        lines.append(
            f"| {r['batch_size']} | {r['num_workers']} "
            f"| {r['time_per_batch_s'] if r['status']=='OK' else 'N/A'} "
            f"| {r['throughput_crops_s'] if r['status']=='OK' else 'N/A'} "
            f"| {r['status']} |"
        )

    lines += [
        "",
        "## GPU Memory Probe",
        "",
        "| batch_size | peak_MB | ms/step | loss | status |",
        "|---|---|---|---|---|",
    ]
    for r in p["probe_rows"]:
        lines.append(
            f"| {r['batch_size']} | {r['peak_gpu_mb']} "
            f"| {r['avg_step_ms']} | {r['loss_sample']} | {r['status']} |"
        )

    lines += [
        "",
        "## Runtime 추정",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| recommended batch_size | {p['rec_bs']} |",
        f"| recommended num_workers | {p['rec_nw']} |",
        f"| steps per epoch | {p['steps_per_epoch']} |",
        f"| ms/step (probe) | {p['ms_per_step']:.1f} |",
        f"| time per epoch (min) | {p['time_per_epoch_min']:.1f} |",
        f"| 30 epoch 예상 시간 (hr) | {p['time_30ep_hr']:.2f} |",
        f"| GPU VRAM peak (MB) | {p['rec_bs_peak_mb']} |",
        "",
        "## Checkpoint 경로",
        "",
        f"- 신규: `{FULL_TRAIN_CKPT_DIR}`",
        f"- smoke (기존): `{SMOKE_CKPT_DIR}`",
        f"- output collision: {p.get('ckpt_collision', False)}",
        "",
        "## guardrail",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        "| full_training_executed | False |",
        "| checkpoint_saved | False |",
        f"| train_val_patient_overlap | {p['overlap']} |",
        "| stage2_holdout_accessed | False |",
        "| image_reconstruction_loss | False |",
        "| convae_branch | False |",
        "",
    ]
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"  [SAVED] {REPORT_MD}")


def write_summary_json(verdict, p, cfg):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "verdict": verdict,
        "created": str(date.today()),
        "branch_name": "rd4ad_2p5d_lung5ch_masked_normal_v1",
        "step": "step6_full_train_preflight",
        "crop_count": p["total_crop_count"],
        "patient_count": p["patient_count"],
        "train_patients": p["n_train_patients"],
        "val_patients": p["n_val_patients"],
        "train_crops": p["n_train_crops"],
        "val_crops": p["n_val_crops"],
        "patient_overlap": p["overlap"],
        "split_seed": args.seed,
        "tested_batch_sizes": args.batch_sizes,
        "tested_num_workers": args.num_workers_list,
        "recommended_batch_size": p["rec_bs"],
        "recommended_num_workers": p["rec_nw"],
        "rec_bs_peak_gpu_mb": p["rec_bs_peak_mb"],
        "steps_per_epoch": p["steps_per_epoch"],
        "ms_per_step": round(p["ms_per_step"], 2),
        "time_per_epoch_min": round(p["time_per_epoch_min"], 2),
        "time_30epochs_hr": round(p["time_30ep_hr"], 2),
        "full_train_config_path": str(CONFIG_YAML),
        "full_train_checkpoint_dir": str(FULL_TRAIN_CKPT_DIR),
        "guardrail": {
            "plan_lock_loaded": True,
            "step5_train_smoke_passed": True,
            "full_training_executed": False,
            "preflight_micro_steps_executed": True,
            "model_type": "true_rd4ad",
            "convae_branch_created": False,
            "image_reconstruction_loss_used": False,
            "input_channels": INPUT_CHANNELS,
            "input_window": "lung",
            "crop_size": CROP_SIZE,
            "crop_count": EXPECTED_CROP_COUNT,
            "crop_dtype_on_disk": "float16",
            "crop_dtype_in_model": "float32",
            "center_z_sampling": "stride2",
            "training_data": "normal_only",
            "split_level": "patient_level",
            "train_val_patient_overlap": p["overlap"] > 0,
            "teacher_frozen": True,
            "optimizer_student_only": True,
            "checkpoint_saved": False,
            "existing_checkpoint_overwritten": False,
            "stage2_holdout_accessed": False,
            "positive_label_used_for_training": False,
            "lesion_mask_used_for_training": False,
            "existing_artifact_modified": False,
        },
        "next_step": "step7_full_training_launch",
        "next_step_note": "full training 실행 (사용자 승인 후)",
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [SAVED] {SUMMARY_JSON}")


def write_done_json(verdict):
    FULL_TRAIN_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    done = {
        "step": "step6_full_train_preflight",
        "verdict": verdict,
        "created": str(date.today()),
        "summary_json": str(SUMMARY_JSON),
        "report_md": str(REPORT_MD),
        "config_yaml": str(CONFIG_YAML),
        "split_csv": str(SPLIT_CSV),
        "full_train_checkpoint_dir": str(FULL_TRAIN_CKPT_DIR),
    }
    with open(DONE_OUT, "w") as f:
        json.dump(done, f, indent=2)
    print(f"  [SAVED] {DONE_OUT}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
if args.dry_run:
    print_dry_run_plan()
    sys.exit(0)

if args.run_preflight:
    missing = []
    if not args.confirm_plan_lock:
        missing.append("--confirm-plan-lock")
    if not args.confirm_no_stage2:
        missing.append("--confirm-no-stage2")
    if not args.confirm_preflight_only:
        missing.append("--confirm-preflight-only")
    if missing:
        print(f"[BLOCKED] 필수 flags 누락: {missing}", file=sys.stderr)
        sys.exit(2)
else:
    print("[BLOCKED] --run-preflight 없이 실행 금지.", file=sys.stderr)
    sys.exit(2)

print()
print("=" * 64)
print("Step 6 Full Train Preflight — ACTUAL RUN")
print("=" * 64)
print()

# [0] Guards
print("[0] Guards 확인")
guard_errors = check_guards()
if guard_errors:
    print("  [BLOCKED] Guard 실패:")
    for e in guard_errors:
        print(f"    - {e}")
    write_errors_csv(guard_errors)
    sys.exit(1)
print("  [PASS] 모든 선행 조건 확인 완료")

# [1] manifest 로드
print()
print("[1] Manifest 로드")
all_rows = read_manifest()
npy_files = sorted(CROPS_DIR.glob("*_crops_f16.npy"))
total_crop_count = len(all_rows)
patient_count = len(set(r["safe_id"] for r in all_rows))
print(f"  총 crops    : {total_crop_count}  (기대 {EXPECTED_CROP_COUNT})")
print(f"  환자 수     : {patient_count}  (기대 {EXPECTED_PATIENT_COUNT})")
print(f"  npy 파일    : {len(npy_files)}")
if total_crop_count != EXPECTED_CROP_COUNT:
    print(f"  [WARN] crop count 불일치")

# [2] dataset 샘플 확인
print()
print("[2] Dataset 샘플 확인")
import numpy as np

sample_indices = [0, 100, 1000, 5000, 10000]
for idx in sample_indices:
    row = all_rows[idx]
    fp = CROPS_DIR / Path(row["file_path"]).name
    arr = np.load(str(fp), mmap_mode="r")
    c = arr[int(row["crop_index_in_file"])].astype(np.float32)
    nan_c = int(np.isnan(c).sum())
    inf_c = int(np.isinf(c).sum())
    outside_zero = float((c == 0.0).mean())
    print(f"  idx={idx:5d}  shape={c.shape}  range=[{c.min():.3f},{c.max():.3f}]  "
          f"NaN={nan_c}  Inf={inf_c}  zero_pct={outside_zero:.1%}")

# [3] train/val split
print()
print("[3] Train/Val split (patient-level, seed=42)")
train_rows, val_rows, train_set, val_set, overlap = section_split(all_rows, args.seed)
if overlap > 0:
    print(f"  [BLOCKED] train/val overlap 발생: {overlap}")
    write_errors_csv([f"train/val overlap: {overlap}"])
    sys.exit(1)
print(f"  [PASS] overlap = 0")

# [4] DataLoader benchmark
print()
print("[4] DataLoader benchmark")
bench_rows = section_dataloader_bench(train_rows, args.batch_sizes, args.num_workers_list, n_batches=10)

# best num_workers 선택 (nw=0 안정, OK 중 fastest)
bench_ok = [r for r in bench_rows if r["status"] == "OK"]
best_bench = None
if bench_ok:
    # batch_size=16 기준, 가장 빠른 nw
    bs16 = [r for r in bench_ok if r["batch_size"] == 16]
    if bs16:
        best_bench = min(bs16, key=lambda r: float(r["time_per_batch_s"]) if r["time_per_batch_s"] != "nan" else 9999)
    else:
        best_bench = bench_ok[0]

# [5] GPU memory probe
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print()
print(f"[5] GPU memory probe (device={device})")
probe_rows = section_gpu_probe(train_rows, args.batch_sizes, args.probe_steps, device)

# recommended batch size
probe_ok = [r for r in probe_rows if r["status"] == "OK"]
if probe_ok:
    rec_bs_row = max(probe_ok, key=lambda r: r["batch_size"])
    rec_bs = rec_bs_row["batch_size"]
    rec_bs_peak_mb = rec_bs_row["peak_gpu_mb"]
    ms_per_step = float(rec_bs_row["avg_step_ms"]) if rec_bs_row["avg_step_ms"] != "nan" else 200.0
else:
    rec_bs = 16
    rec_bs_peak_mb = -1.0
    ms_per_step = 200.0

# recommended num_workers
rec_nw = 0
if best_bench and best_bench["status"] == "OK":
    t0_nw = next((r for r in bench_ok if r["batch_size"] == rec_bs and r["num_workers"] == 0), None)
    t2_nw = next((r for r in bench_ok if r["batch_size"] == rec_bs and r["num_workers"] == 2), None)
    if t2_nw and t0_nw:
        if float(str(t2_nw["time_per_batch_s"])) < float(str(t0_nw["time_per_batch_s"])):
            rec_nw = 2

# [6] Config 생성
print()
print("[6] Full train config 생성")
cfg = section_write_config(rec_bs, rec_nw, len(train_rows), len(val_rows))

# [7] Checkpoint 경로 확인
print()
print("[7] Checkpoint 경로 확인")
ckpt_collision = False
FULL_TRAIN_CKPT_DIR.mkdir(parents=True, exist_ok=True)
existing_pths = list(FULL_TRAIN_CKPT_DIR.glob("*.pth"))
if existing_pths:
    ckpt_collision = True
    print(f"  [WARN] 기존 .pth 파일 존재: {[f.name for f in existing_pths]}")
else:
    print(f"  [PASS] 충돌 없음: {FULL_TRAIN_CKPT_DIR}")
print(f"  smoke ckpt (보존): {SMOKE_CKPT_DIR}")

# [8] Runtime 추정
print()
print("[8] Runtime 추정")
steps_per_epoch = math.ceil(len(train_rows) / rec_bs)
time_per_epoch_s = steps_per_epoch * ms_per_step / 1000.0
time_per_epoch_min = time_per_epoch_s / 60.0
time_30ep_hr = time_per_epoch_min * 30 / 60.0
print(f"  train crops       : {len(train_rows)}")
print(f"  rec batch_size    : {rec_bs}")
print(f"  steps/epoch       : {steps_per_epoch}")
print(f"  ms/step (probe)   : {ms_per_step:.1f}")
print(f"  time/epoch        : {time_per_epoch_min:.1f} min")
print(f"  30 epoch 예상     : {time_30ep_hr:.2f} hr")
print(f"  GPU peak (MB)     : {rec_bs_peak_mb}")

# [9] 판정
print()
all_bench_ok = any(r["status"] == "OK" for r in bench_rows)
all_probe_ok = any(r["status"] == "OK" for r in probe_rows)

verdict = "BLOCKED"
if (total_crop_count == EXPECTED_CROP_COUNT
        and overlap == 0
        and all_bench_ok
        and all_probe_ok
        and not ckpt_collision):
    verdict = "PASS_STEP6_FULL_TRAIN_PREFLIGHT"
elif overlap == 0 and all_bench_ok and all_probe_ok and ckpt_collision:
    verdict = "PARTIAL_PASS_STEP6_CKPT_COLLISION"
elif overlap == 0 and all_bench_ok and not all_probe_ok:
    verdict = "PARTIAL_PASS_STEP6_GPU_PROBE_PARTIAL"
elif overlap > 0:
    verdict = "BLOCKED_STEP6_SPLIT_OVERLAP"

params = {
    "total_crop_count": total_crop_count,
    "patient_count": patient_count,
    "n_train_patients": len(train_set),
    "n_val_patients": len(val_set),
    "n_train_crops": len(train_rows),
    "n_val_crops": len(val_rows),
    "overlap": overlap,
    "bench_rows": bench_rows,
    "probe_rows": probe_rows,
    "rec_bs": rec_bs,
    "rec_nw": rec_nw,
    "rec_bs_peak_mb": rec_bs_peak_mb,
    "steps_per_epoch": steps_per_epoch,
    "ms_per_step": ms_per_step,
    "time_per_epoch_min": time_per_epoch_min,
    "time_30ep_hr": time_30ep_hr,
    "ckpt_collision": ckpt_collision,
}

print("=" * 64)
print(f"판정: {verdict}")
print("=" * 64)
print(f"  crop count         : {total_crop_count}")
print(f"  patient count      : {patient_count}")
print(f"  train patients     : {len(train_set)}")
print(f"  val patients       : {len(val_set)}")
print(f"  train crops        : {len(train_rows)}")
print(f"  val crops          : {len(val_rows)}")
print(f"  patient overlap    : {overlap}")
print(f"  rec batch_size     : {rec_bs}")
print(f"  rec num_workers    : {rec_nw}")
print(f"  GPU peak (MB)      : {rec_bs_peak_mb}")
print(f"  steps/epoch        : {steps_per_epoch}")
print(f"  time/epoch (min)   : {time_per_epoch_min:.1f}")
print(f"  30ep 예상 (hr)     : {time_30ep_hr:.2f}")
print(f"  ckpt collision     : {ckpt_collision}")
print(f"  stage2 accessed    : False")
print(f"  checkpoint saved   : False")
print("=" * 64)

# [10] 파일 저장
print()
print("[10] 결과 파일 저장")
write_errors_csv([])
write_report_md(verdict, params)
write_summary_json(verdict, params, cfg)
if verdict.startswith("PASS") or verdict.startswith("PARTIAL"):
    write_done_json(verdict)

print()
if verdict == "PASS_STEP6_FULL_TRAIN_PREFLIGHT":
    print("Step 6 완료. 다음 단계: Step 7 full training launch (사용자 승인 후)")
elif verdict.startswith("PARTIAL"):
    print(f"Step 6 PARTIAL_PASS. 상세 확인 후 Step 7 진행 가능 여부 판단.")
else:
    print(f"Step 6 BLOCKED: {verdict}")
