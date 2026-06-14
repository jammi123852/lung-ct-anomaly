"""
Step 8: stage1_dev Candidate Scoring
rd4ad_2p5d_lung5ch_masked_normal_v1

대상: stage1_dev p90 초과 + 동일 위치 z연속≥2 후보
checkpoint: checkpoints/full_train_v1/student_best_val_loss.pth

bare run → exit 2
dry-run  → 계획 출력, 실행 없음
actual   → --run-scoring --confirm-plan-lock --confirm-no-stage2 --confirm-score-only

금지: stage2 접근, 추가 학습, checkpoint 수정, threshold 새로 선택,
       후보 삭제, representative-only scoring
"""

import sys
import os
import json
import csv
import math
import time
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import date

# ── bare run 차단 ──────────────────────────────────────────────────────────────
if len(sys.argv) == 1:
    print("[BLOCKED] bare run 금지. --dry-run 또는 필수 flags를 사용하세요.", file=sys.stderr)
    sys.exit(2)

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]

PLAN_LOCK_JSON = ROOT / "docs" / "FINAL_PLAN_LOCK.json"
DONE_STEP7_JSON = ROOT / "DONE_STEP7_FULL_TRAINING.json"
CKPT_BEST = ROOT / "checkpoints" / "full_train_v1" / "student_best_val_loss.pth"

NSCLC_CT_ROOT = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
LESION_MASK_ROOT = PROJECT_ROOT / "outputs" / "mip-postprocess-research-v1" / "masks" / "refined_roi_v4_20_modeB_all_v1" / "lesion"

CANDIDATE_CSV = PROJECT_ROOT / "outputs" / "normal_based_stage2_verifier_audit" / "rd_c2_effb0_v420_candidate_rd4ad_retest_v1" / "rd_c2_effb0_v420_candidate_manifest.csv"

OUT_DIR = ROOT / "scoring" / "step8_stage1dev_v1"
SCORE_CSV = OUT_DIR / "rd4ad_lung5ch_stage1dev_scores_v1.csv"
SUMMARY_JSON = OUT_DIR / "step8_scoring_summary.json"
ERRORS_CSV = OUT_DIR / "step8_scoring_errors.csv"
DONE_OUT = ROOT / "DONE_STEP8_SCORING.json"

P90_THRESHOLD = 12.196394
Z_CONT_MIN = 2
CROP_SIZE = 96
HU_MIN, HU_MAX = -1350, 150
BATCH_SIZE = 32

# ── argparse ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--run-scoring", action="store_true")
parser.add_argument("--confirm-plan-lock", action="store_true")
parser.add_argument("--confirm-no-stage2", action="store_true")
parser.add_argument("--confirm-score-only", action="store_true")
args = parser.parse_args()


def print_dry_run():
    print()
    print("=" * 64)
    print("Step 8 Scoring — DRY-RUN PLAN")
    print("=" * 64)
    print()
    print("[대상]")
    print(f"  candidate CSV  : {CANDIDATE_CSV}")
    print(f"  필터           : first_stage_score > {P90_THRESHOLD}  (p90)")
    print(f"                   동일 위치 z연속 ≥ {Z_CONT_MIN}")
    print(f"  NSCLC CT root  : {NSCLC_CT_ROOT}")
    print(f"  lesion mask    : {LESION_MASK_ROOT}")
    print()
    print("[모델]")
    print(f"  checkpoint     : {CKPT_BEST}")
    print(f"  teacher        : ResNet18 5ch (frozen, eval)")
    print(f"  student        : RD4AD mirror decoder (eval)")
    print(f"  batch_size     : {BATCH_SIZE}")
    print()
    print("[출력 컬럼 (추가)]")
    for col in ["rd4ad_lung5ch_score_raw", "score_layer1", "score_layer2", "score_layer3",
                "roi_ratio", "P1_times_roi", "P2_times_sqrt_roi",
                "track_id", "track_len", "lung_z_percentile", "nearest_repeat_used"]:
        print(f"  {col}")
    print()
    print("[금지]")
    print("  stage2 접근 금지 | 추가 학습 금지 | checkpoint 수정 금지")
    print("  threshold 선택 금지 | 후보 삭제 금지 | representative-only 금지")
    print()
    print("[생성 파일]")
    for p in [SCORE_CSV, SUMMARY_JSON, ERRORS_CSV, DONE_OUT]:
        print(f"  {p}")
    print()
    rel = Path("experiments/rd4ad_2p5d_lung5ch_masked_normal_v1/scripts") / Path(__file__).name
    print("[실행 명령]")
    print(f"  python {rel} \\")
    print(f"    --run-scoring --confirm-plan-lock --confirm-no-stage2 --confirm-score-only")
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
        safety = lock.get("safety", {})
        if safety.get("stage2_holdout_accessed"):
            errors.append("stage2 already accessed!")

    if not DONE_STEP7_JSON.exists():
        errors.append("DONE_STEP7 JSON 없음")
    else:
        with open(DONE_STEP7_JSON) as f:
            d = json.load(f)
        if d.get("verdict") != "PASS_STEP7_FULL_TRAINING":
            errors.append(f"Step7 verdict 불일치: {d.get('verdict')}")

    if not CKPT_BEST.exists():
        errors.append(f"checkpoint 없음: {CKPT_BEST}")
    if not CANDIDATE_CSV.exists():
        errors.append(f"candidate CSV 없음: {CANDIDATE_CSV}")
    if not NSCLC_CT_ROOT.exists():
        errors.append(f"NSCLC CT root 없음: {NSCLC_CT_ROOT}")
    if not LESION_MASK_ROOT.exists():
        errors.append(f"lesion mask root 없음: {LESION_MASK_ROOT}")

    # output collision 확인: stage2 경로와 분리 여부
    if "stage2" in str(SCORE_CSV).lower():
        errors.append("output 경로에 stage2 포함 — 금지")

    return errors


# ── 후보 필터링 + track 할당 ───────────────────────────────────────────────────
def load_and_filter_candidates():
    """p90 초과 + z연속≥2 필터, track_id/track_len 할당"""
    rows = []
    with open(CANDIDATE_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    # p90 필터
    above = [r for r in rows if float(r["first_stage_score"]) > P90_THRESHOLD]

    # position별 z 수집
    pos_zs = defaultdict(list)
    for r in above:
        key = (r["patient_id"], r["crop_y0"], r["crop_x0"])
        pos_zs[key].append((int(r["local_z"]), r))

    # z연속 그룹화 (stride2 기준: z 차이 ≤ 2)
    def cont_groups(z_row_list):
        z_row_list = sorted(z_row_list, key=lambda x: x[0])
        groups, cur = [], [z_row_list[0]]
        for item in z_row_list[1:]:
            if item[0] - cur[-1][0] <= 2:
                cur.append(item)
            else:
                groups.append(cur)
                cur = [item]
        groups.append(cur)
        return [g for g in groups if len(g) >= Z_CONT_MIN]

    valid_rows = []
    track_id = 0
    for key, z_rows in pos_zs.items():
        for grp in cont_groups(z_rows):
            t_len = len(grp)
            for z, row in grp:
                r = dict(row)
                r["track_id"] = track_id
                r["track_len"] = t_len
                valid_rows.append(r)
            track_id += 1

    return valid_rows, len(above), track_id


# ── 5ch crop 추출 ──────────────────────────────────────────────────────────────
HU_RANGE = HU_MAX - HU_MIN

def apply_lung_window(hu):
    import numpy as np
    return (np.clip(hu.astype(np.float32), HU_MIN, HU_MAX) - HU_MIN) / HU_RANGE


def get_z_indices(z, D):
    return [max(0, min(D - 1, z + off)) for off in (-2, -1, 0, 1, 2)]


def extract_5ch(ct_vol, mask_vol, z, y0, x0, y1, x1):
    """(5, H, W) float32 crop + roi_ratio + nearest_repeat_used"""
    import numpy as np
    D = ct_vol.shape[0]
    z_idxs = get_z_indices(z, D)
    nearest_repeat = (z_idxs[0] == z_idxs[1]) or (z_idxs[3] == z_idxs[4])

    chans = []
    for zi in z_idxs:
        sl = ct_vol[zi, y0:y1, x0:x1]
        chans.append(apply_lung_window(sl))

    crop = np.stack(chans, axis=0)  # (5, H, W)

    # mask at center z
    if mask_vol is not None and z < mask_vol.shape[0]:
        m = mask_vol[z, y0:y1, x0:x1].astype(np.float32)
    else:
        m = np.ones((y1 - y0, x1 - x0), dtype=np.float32)

    # shape 보정
    h_actual, w_actual = crop.shape[1], crop.shape[2]
    if m.shape != (h_actual, w_actual):
        m2 = np.zeros((h_actual, w_actual), dtype=np.float32)
        hh = min(m.shape[0], h_actual)
        ww = min(m.shape[1], w_actual)
        m2[:hh, :ww] = m[:hh, :ww]
        m = m2

    # lung exterior zeroing
    crop = crop * m[None]

    roi_ratio = float(m.sum()) / max(1, h_actual * w_actual)
    return crop, m, roi_ratio, nearest_repeat


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

    handles = [teacher.layer1.register_forward_hook(h("l1")),
               teacher.layer2.register_forward_hook(h("l2")),
               teacher.layer3.register_forward_hook(h("l3"))]
    with torch.no_grad():
        teacher(x)
    for hh in handles:
        hh.remove()
    return feats


def batch_score(teacher, student, crops_t, masks_t, device):
    """
    crops_t : (B, 5, 96, 96) float32 tensor (already on device)
    masks_t : (B, 1, 96, 96) float32 tensor (already on device)
    returns : (B,) arrays for score_raw, l1, l2, l3
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        tf = get_teacher_feats(teacher, crops_t)
        tl1, tl2, tl3 = tf["l1"], tf["l2"], tf["l3"]
        sl1, sl2, sl3 = student(tl3)

    def dm(m, h, w):
        return F.interpolate(m, size=(h, w), mode="nearest")

    def cos_err(t, s, mask):
        cos = F.cosine_similarity(t, s, dim=1, eps=1e-6)   # (B, H, W)
        err = 1.0 - cos
        m = mask.squeeze(1)
        # per-sample loss
        n = err.shape[0]
        losses = []
        for i in range(n):
            denom = float(m[i].sum()) + 1e-6
            losses.append(float((err[i] * m[i]).sum()) / denom)
        return losses

    ml1 = dm(masks_t, tl1.shape[2], tl1.shape[3])
    ml2 = dm(masks_t, tl2.shape[2], tl2.shape[3])
    ml3 = dm(masks_t, tl3.shape[2], tl3.shape[3])

    ll1 = cos_err(tl1, sl1, ml1)
    ll2 = cos_err(tl2, sl2, ml2)
    ll3 = cos_err(tl3, sl3, ml3)

    raw = [(a + b + c) / 3.0 for a, b, c in zip(ll1, ll2, ll3)]
    return raw, ll1, ll2, ll3


# ── lung_z_percentile 계산 ────────────────────────────────────────────────────
def compute_lung_z_range(mask_vol):
    import numpy as np
    D = mask_vol.shape[0]
    areas = np.array([int(mask_vol[z].sum()) for z in range(D)])
    idx = np.where(areas > 0)[0]
    if len(idx) == 0:
        return 0, D - 1
    return int(idx[0]), int(idx[-1])


def lung_z_pct(z, lung_z_min, lung_z_max):
    span = max(1, lung_z_max - lung_z_min)
    return round((z - lung_z_min) / span, 4)


# ── 출력 파일 ─────────────────────────────────────────────────────────────────
SCORE_FIELDS_EXTRA = [
    "rd4ad_lung5ch_score_raw", "score_layer1", "score_layer2", "score_layer3",
    "roi_ratio", "P1_times_roi", "P2_times_sqrt_roi",
    "track_id", "track_len", "lung_z_percentile", "nearest_repeat_used",
]

def write_errors_csv(errors):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ERRORS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "safe_id", "msg"])
        w.writeheader()
        for e in errors:
            w.writerow(e)
    print(f"  [SAVED] {ERRORS_CSV}")


def write_summary_json(verdict, p):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "verdict": verdict,
        "created": str(date.today()),
        "branch_name": "rd4ad_2p5d_lung5ch_masked_normal_v1",
        "step": "step8_scoring_stage1dev",
        "p90_threshold": P90_THRESHOLD,
        "z_cont_min": Z_CONT_MIN,
        "candidate_total_before_filter": p["n_total"],
        "candidate_p90_above": p["n_above_p90"],
        "candidate_after_z_cont": p["n_valid"],
        "n_tracks": p["n_tracks"],
        "n_unique_patients": p["n_patients"],
        "scored_crops": p["scored"],
        "failed_crops": p["failed"],
        "nan_count": p["nan_count"],
        "inf_count": p["inf_count"],
        "score_csv": str(SCORE_CSV),
        "checkpoint_used": str(CKPT_BEST),
        "guardrail": {
            "stage2_holdout_accessed": False,
            "additional_training_executed": False,
            "checkpoint_modified": False,
            "threshold_newly_selected": False,
            "candidate_deleted": False,
            "representative_only_scoring": False,
            "model_type": "true_rd4ad",
            "scoring_scope": "stage1_dev_only",
        },
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [SAVED] {SUMMARY_JSON}")


def write_done_json(verdict, scored, failed):
    done = {
        "step": "step8_scoring_stage1dev",
        "verdict": verdict,
        "created": str(date.today()),
        "scored_crops": scored,
        "failed_crops": failed,
        "score_csv": str(SCORE_CSV),
        "summary_json": str(SUMMARY_JSON),
    }
    with open(DONE_OUT, "w") as f:
        json.dump(done, f, indent=2)
    print(f"  [SAVED] {DONE_OUT}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
if args.dry_run:
    print_dry_run()
    sys.exit(0)

if args.run_scoring:
    missing = []
    if not args.confirm_plan_lock:
        missing.append("--confirm-plan-lock")
    if not args.confirm_no_stage2:
        missing.append("--confirm-no-stage2")
    if not args.confirm_score_only:
        missing.append("--confirm-score-only")
    if missing:
        print(f"[BLOCKED] 필수 flags 누락: {missing}", file=sys.stderr)
        sys.exit(2)
else:
    print("[BLOCKED] --run-scoring 없이 실행 금지.", file=sys.stderr)
    sys.exit(2)

print()
print("=" * 64)
print("Step 8 Scoring — ACTUAL RUN")
print("=" * 64)
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

# [1] 후보 필터링
print()
print("[1] 후보 필터링 (p90 + z연속≥2)")

import csv as csv_mod
total_rows_in_csv = sum(1 for _ in open(CANDIDATE_CSV)) - 1
valid_rows, n_above_p90, n_tracks = load_and_filter_candidates()

n_unique_pts = len(set(r["patient_id"] for r in valid_rows))
print(f"  CSV 전체 행    : {total_rows_in_csv}")
print(f"  p90 초과       : {n_above_p90}")
print(f"  z연속≥2 후보   : {len(valid_rows)}")
print(f"  트랙 수        : {n_tracks}")
print(f"  unique patients: {n_unique_pts}")

# patient별 그룹화
patient_rows = defaultdict(list)
for r in valid_rows:
    patient_rows[r["safe_id"]].append(r)

# [2] 모델 로드
print()
print("[2] 모델 + checkpoint 로드")
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher = build_teacher(device)
student = build_student(device)

ckpt = torch.load(str(CKPT_BEST), map_location=device)
student.load_state_dict(ckpt["student_state_dict"])
student.eval()

print(f"  checkpoint epoch : {ckpt.get('epoch')}")
print(f"  checkpoint val_loss: {ckpt.get('val_loss', 'N/A'):.6f}")
print(f"  teacher frozen   : {all(not p.requires_grad for p in teacher.parameters())}")
print(f"  device           : {device}")

# [3] 스코어링
print()
print("[3] Scoring 시작")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 원본 컬럼 + extra 컬럼
orig_fields = list(valid_rows[0].keys()) if valid_rows else []
# track_id/track_len은 이미 포함됨
extra_fields = [c for c in SCORE_FIELDS_EXTRA if c not in orig_fields]
all_fields = orig_fields + extra_fields

cnt = {"scored": 0, "failed": 0, "nan": 0, "inf": 0}
error_rows = []
t_start = time.perf_counter()

with open(SCORE_CSV, "w", newline="") as out_f:
    writer = csv_mod.DictWriter(out_f, fieldnames=all_fields)
    writer.writeheader()

    n_patients = len(patient_rows)
    for pi, (safe_id, rows) in enumerate(sorted(patient_rows.items())):
        pt_id = rows[0]["patient_id"]

        # CT 로드
        ct_path = NSCLC_CT_ROOT / safe_id / "ct_hu.npy"
        if not ct_path.exists():
            msg = f"CT 없음: {ct_path}"
            error_rows.append({"patient_id": pt_id, "safe_id": safe_id, "msg": msg})
            cnt["failed"] += len(rows)
            print(f"  [{pi+1:3d}/{n_patients}] {safe_id[:40]}  ERROR: CT 없음")
            continue

        ct_vol = np.load(str(ct_path), mmap_mode="r")

        # mask 로드
        mask_path = LESION_MASK_ROOT / safe_id / "refined_roi.npy"
        mask_vol = np.load(str(mask_path), mmap_mode="r") if mask_path.exists() else None

        # lung_z_range (percentile용)
        if mask_vol is not None:
            lung_z_min, lung_z_max = compute_lung_z_range(mask_vol)
        else:
            lung_z_min, lung_z_max = 0, ct_vol.shape[0] - 1

        D = ct_vol.shape[0]

        # 배치 단위 처리
        batch_crops, batch_masks, batch_meta, batch_rois, batch_nr = [], [], [], [], []

        def flush_batch(writer, batch_crops, batch_masks, batch_meta,
                        batch_rois, batch_nr, lung_z_min, lung_z_max, cnt):
            if not batch_crops:
                return
            crops_t = torch.from_numpy(np.stack(batch_crops, axis=0)).to(device)
            masks_t = torch.from_numpy(
                np.stack([m[None] for m in batch_masks], axis=0)).to(device)

            raw_list, l1_list, l2_list, l3_list = batch_score(
                teacher, student, crops_t, masks_t, device)

            for i, meta in enumerate(batch_meta):
                raw = raw_list[i]
                l1 = l1_list[i]
                l2 = l2_list[i]
                l3 = l3_list[i]
                roi = batch_rois[i]
                nr = batch_nr[i]

                cnt["nan"] += int(math.isnan(raw))
                cnt["inf"] += int(math.isinf(raw))

                out_row = dict(meta)
                out_row["rd4ad_lung5ch_score_raw"] = round(raw, 8)
                out_row["score_layer1"] = round(l1, 8)
                out_row["score_layer2"] = round(l2, 8)
                out_row["score_layer3"] = round(l3, 8)
                out_row["roi_ratio"] = round(roi, 6)
                out_row["P1_times_roi"] = round(raw * roi, 8)
                out_row["P2_times_sqrt_roi"] = round(raw * math.sqrt(max(roi, 0)), 8)
                out_row["lung_z_percentile"] = lung_z_pct(
                    int(meta["local_z"]), lung_z_min, lung_z_max)
                out_row["nearest_repeat_used"] = nr
                writer.writerow(out_row)
                cnt["scored"] += 1

        for row in rows:
            z = int(row["local_z"])
            y0, x0 = int(row["crop_y0"]), int(row["crop_x0"])
            y1, x1 = int(row["crop_y1"]), int(row["crop_x1"])

            if z >= D:
                error_rows.append({"patient_id": pt_id, "safe_id": safe_id,
                                   "msg": f"z={z} >= D={D}"})
                cnt["failed"] += 1
                continue

            crop, mask_crop, roi_ratio, nearest_repeat = extract_5ch(
                ct_vol, mask_vol, z, y0, x0, y1, x1)

            if crop.shape != (5, CROP_SIZE, CROP_SIZE):
                error_rows.append({"patient_id": pt_id, "safe_id": safe_id,
                                   "msg": f"crop shape {crop.shape}"})
                cnt["failed"] += 1
                continue

            batch_crops.append(crop)
            batch_masks.append(mask_crop)
            batch_meta.append(row)
            batch_rois.append(roi_ratio)
            batch_nr.append(nearest_repeat)

            if len(batch_crops) >= BATCH_SIZE:
                flush_batch(writer, batch_crops, batch_masks, batch_meta,
                            batch_rois, batch_nr, lung_z_min, lung_z_max, cnt)
                batch_crops.clear(); batch_masks.clear(); batch_meta.clear()
                batch_rois.clear(); batch_nr.clear()

        flush_batch(writer, batch_crops, batch_masks, batch_meta,
                    batch_rois, batch_nr, lung_z_min, lung_z_max, cnt)

        elapsed = time.perf_counter() - t_start
        print(f"  [{pi+1:3d}/{n_patients}] {safe_id[:50]:50s}  "
              f"done={cnt['scored']}  err={cnt['failed']}  {elapsed:.1f}s")

scored  = cnt["scored"]
failed  = cnt["failed"]
nan_count = cnt["nan"]
inf_count = cnt["inf"]

# [4] 판정
elapsed_total = time.perf_counter() - t_start
verdict = "BLOCKED"
if failed == 0 and nan_count == 0 and inf_count == 0:
    verdict = "PASS_STEP8_SCORING"
elif scored > 0 and failed < scored * 0.01:
    verdict = "PARTIAL_PASS_STEP8_FEW_ERRORS"
elif scored > 0:
    verdict = f"PARTIAL_PASS_STEP8_ERRORS_{failed}"
else:
    verdict = "BLOCKED_STEP8_NO_SCORED"

print()
print("=" * 64)
print(f"판정: {verdict}")
print("=" * 64)
print(f"  candidate 총수  : {total_rows_in_csv}")
print(f"  p90 초과        : {n_above_p90}")
print(f"  z연속≥2 후보    : {len(valid_rows)}")
print(f"  트랙 수         : {n_tracks}")
print(f"  unique patients : {n_unique_pts}")
print(f"  scoring 완료    : {scored}")
print(f"  failed          : {failed}")
print(f"  NaN             : {nan_count}")
print(f"  Inf             : {inf_count}")
print(f"  총 소요 시간    : {elapsed_total:.1f}s")
print(f"  stage2 접근     : False")
print("=" * 64)

# [5] 파일 저장
print()
print("[5] 결과 파일 저장")
write_errors_csv(error_rows)
p_summary = {
    "n_total": total_rows_in_csv,
    "n_above_p90": n_above_p90,
    "n_valid": len(valid_rows),
    "n_tracks": n_tracks,
    "n_patients": n_unique_pts,
    "scored": scored,
    "failed": failed,
    "nan_count": nan_count,
    "inf_count": inf_count,
}
write_summary_json(verdict, p_summary)
write_done_json(verdict, scored, failed)
print(f"  [SAVED] {SCORE_CSV} ({scored} rows)")

print()
if verdict == "PASS_STEP8_SCORING":
    print("Step 8 완료. 다음 단계: Step 9 환자별 이상 후보 정렬 (사용자 승인 후)")
else:
    print(f"Step 8 {verdict}. 결과 확인 후 다음 단계 결정.")
