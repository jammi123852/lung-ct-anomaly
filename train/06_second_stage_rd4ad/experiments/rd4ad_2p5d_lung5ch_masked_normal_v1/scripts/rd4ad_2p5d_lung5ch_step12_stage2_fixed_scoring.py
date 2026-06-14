"""
Step 12: Stage2 Fixed Scoring
rd4ad_2p5d_lung5ch_masked_normal_v1

대상: step11 manifest (127,947 z-continuity>=2 stage2_holdout candidates)
      8 patient-stable shards
입력: step11_stage2_scoring_plan_manifest.csv
      NSCLC CT volumes, v4_20 lesion mask
checkpoint: checkpoints/full_train_v1/student_best_val_loss.pth

bare run → exit 2
dry-run  → 계획 출력, 실행 없음
--run-shard → shard-level scoring

금지: training / backward / optimizer / checkpoint 수정
      P1/P2 primary 사용 / score family 변경
      stage2 threshold 튜닝 / label 사용
      candidate 삭제 / representative-only scoring
"""

import sys
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
    print("[BLOCKED] bare run 금지. --dry-run 또는 --run-shard를 사용하세요.", file=sys.stderr)
    sys.exit(2)

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]

PLAN_LOCK_JSON   = ROOT / "docs" / "FINAL_PLAN_LOCK.json"
DONE_STEP10_JSON = ROOT / "DONE_STEP10_DECISION_CHECKPOINT.json"
DONE_STEP11_JSON = ROOT / "DONE_STEP11_STAGE2_FIXED_PREFLIGHT.json"
CKPT_BEST        = ROOT / "checkpoints" / "full_train_v1" / "student_best_val_loss.pth"
STEP12_MANIFEST  = ROOT / "manifests" / "step11_stage2_scoring_plan_manifest.csv"

STAGE2_MANIFEST_ORIG = PROJECT_ROOT / "outputs" / \
    "second-stage-lesion-refiner-v1" / "datasets" / \
    "s6a_stage2_holdout_candidate_coordinate_manifest_v1.csv"

NSCLC_CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)
LESION_MASK_ROOT = PROJECT_ROOT / "outputs" / "mip-postprocess-research-v1" / \
    "masks" / "refined_roi_v4_20_modeB_all_v1" / "lesion"

OUT_DIR       = ROOT / "scoring" / "step12_stage2_fixed_v1"
OUT_MANIFESTS = ROOT / "manifests"
OUT_REPORTS   = ROOT / "reports"
OUT_LOGS      = ROOT / "logs"
DONE_OUT      = ROOT / "DONE_STEP12_STAGE2_FIXED_SCORING.json"

SHARD_STATUS_CSV = OUT_MANIFESTS / "step12_stage2_scoring_shard_status.csv"
ERRORS_CSV       = OUT_LOGS / "step12_stage2_fixed_scoring_errors.csv"
REPORT_MD        = OUT_REPORTS / "step12_stage2_fixed_scoring_report.md"
SUMMARY_JSON     = OUT_REPORTS / "step12_stage2_fixed_scoring_summary.json"

# ── 고정 파라미터 ─────────────────────────────────────────────────────────────
P90_THRESHOLD           = 12.196394
PRIMARY_CANDIDATE_SCORE = "rd4ad_lung5ch_score_raw"
PRIMARY_TRACK_SCORE     = "raw_track_top3_mean"
CROP_SIZE               = 96
HU_MIN, HU_MAX          = -1350.0, 150.0
HU_RANGE                = HU_MAX - HU_MIN
N_SHARDS                = 8
BATCH_SIZE              = 32
EXPECTED_TOTAL_ROWS     = 127947


# ── 출력 컬럼 정의 ─────────────────────────────────────────────────────────────
OUT_FIELDS = [
    "row_id", "patient_id", "safe_id", "local_z",
    "score_original",
    "pos_y0", "pos_x0", "pos_y1", "pos_x1",
    "crop_y0", "crop_x0", "crop_y1", "crop_x1",
    "z_minus2_effective", "z_minus1_effective", "z_center_effective",
    "z_plus1_effective", "z_plus2_effective",
    "nearest_repeat_used",
    "track_id", "track_len", "track_z_start", "track_z_end",
    "lung_z_percentile",
    "mask_area_center", "mask_area_5ch_mean",
    "crop_min", "crop_max", "crop_mean", "crop_std",
    "roi_ratio",
    "rd4ad_lung5ch_score_raw", "score_layer1", "score_layer2", "score_layer3",
    "shard_id", "status", "error_message",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-shard", action="store_true")
    ap.add_argument("--shard-id", type=int, default=-1)
    ap.add_argument("--max-rows", type=int, default=None, help="smoke test 행 수 제한")
    ap.add_argument("--confirm-plan-lock", action="store_true")
    ap.add_argument("--confirm-fixed-score", action="store_true")
    ap.add_argument("--confirm-no-training", action="store_true")
    return ap.parse_args()


# ── dry-run 출력 ───────────────────────────────────────────────────────────────
def print_dry_run():
    print()
    print("=" * 64)
    print("Step 12 Stage2 Fixed Scoring — DRY-RUN PLAN")
    print("=" * 64)
    print()
    print("[선행 조건]")
    print(f"  DONE_STEP10  : {DONE_STEP10_JSON}")
    print(f"  DONE_STEP11  : {DONE_STEP11_JSON}")
    print(f"  PLAN_LOCK    : {PLAN_LOCK_JSON}")
    print()
    print("[입력]")
    print(f"  manifest     : {STEP12_MANIFEST}")
    print(f"  s2 manifest  : {STAGE2_MANIFEST_ORIG}  (score_original 조회용)")
    print(f"  CT root      : {NSCLC_CT_ROOT}")
    print(f"  mask root    : {LESION_MASK_ROOT}")
    print(f"  checkpoint   : {CKPT_BEST}")
    print()
    print("[파라미터]")
    print(f"  p90 threshold: {P90_THRESHOLD}  (고정, 재계산 금지)")
    print(f"  coordinate   : 32x32 position → center±48 → 96x96 crop")
    print(f"  crop size    : {CROP_SIZE}x{CROP_SIZE}")
    print(f"  HU window    : [{HU_MIN}, {HU_MAX}]")
    print(f"  batch size   : {BATCH_SIZE}")
    print(f"  n_shards     : {N_SHARDS}")
    print(f"  expected rows: {EXPECTED_TOTAL_ROWS:,}")
    print()
    print("[score 컬럼 (primary)]")
    print(f"  {PRIMARY_CANDIDATE_SCORE}")
    print(f"  score_layer1, score_layer2, score_layer3")
    print()
    print("[primary lock]")
    print(f"  candidate: {PRIMARY_CANDIDATE_SCORE}")
    print(f"  track    : {PRIMARY_TRACK_SCORE}")
    print(f"  P1 REJECT: P1_times_roi 사용 금지")
    print(f"  P2 REJECT: P2_times_sqrt_roi 사용 금지")
    print()
    print("[StudentDecoder keys]  dl3 / dl2 / dl1  (dec_l3/dec_l2/dec_l1 금지)")
    print()
    print("[금지]")
    print("  training / backward / optimizer / checkpoint 저장·수정")
    print("  stage2 label 사용 / threshold 재계산 / score family 변경")
    print("  candidate 삭제 / representative-only scoring")
    print()
    print("[생성 파일]")
    for i in range(N_SHARDS):
        print(f"  {OUT_DIR}/shard_{i:03d}_scores.csv")
        print(f"  {OUT_DIR}/shard_{i:03d}_summary.json")
    for fp in [ERRORS_CSV, SHARD_STATUS_CSV, REPORT_MD, SUMMARY_JSON, DONE_OUT]:
        print(f"  {fp}")
    print()
    rel = Path("experiments/rd4ad_2p5d_lung5ch_masked_normal_v1/scripts") / Path(__file__).name
    print("[실행 명령]")
    print(f"  dry-run:")
    print(f"    python {rel} --dry-run")
    print()
    print(f"  single shard smoke (shard 0, 256행):")
    print(f"    python {rel} \\")
    print(f"      --run-shard --shard-id 0 --max-rows 256 \\")
    print(f"      --confirm-plan-lock --confirm-fixed-score --confirm-no-training")
    print()
    print(f"  full shard run:")
    print(f"    for i in 0 1 2 3 4 5 6 7; do")
    print(f"      python {rel} \\")
    print(f"        --run-shard --shard-id $i \\")
    print(f"        --confirm-plan-lock --confirm-fixed-score --confirm-no-training")
    print(f"    done")
    print()
    print("DRY-RUN 완료.")


# ── guard checks ──────────────────────────────────────────────────────────────
def check_guards():
    errors = []
    for fp, expected_verdict in [
        (DONE_STEP10_JSON, "PASS_STEP10_DECISION_CHECKPOINT"),
        (DONE_STEP11_JSON, "PASS_STEP11_STAGE2_FIXED_PREFLIGHT"),
    ]:
        if not fp.exists():
            errors.append(f"DONE 없음: {fp.name}")
        else:
            with open(fp) as f:
                d = json.load(f)
            if d.get("verdict") != expected_verdict:
                errors.append(f"{fp.name} verdict={d.get('verdict')} (expected {expected_verdict})")

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
            errors.append("plan_lock: stage2_holdout_accessed = True (불일치)")

    for fp, label in [
        (STEP12_MANIFEST,    "step12 manifest"),
        (CKPT_BEST,          "checkpoint"),
        (NSCLC_CT_ROOT,      "CT root"),
        (LESION_MASK_ROOT,   "mask root"),
        (STAGE2_MANIFEST_ORIG, "stage2 manifest (score_original 조회용)"),
    ]:
        if not fp.exists():
            errors.append(f"{label} 없음: {fp}")
    return errors


# ── score_original 조회 테이블 ─────────────────────────────────────────────────
def load_score_original_lookup():
    """row_id → score_original dict.  BOM-safe (utf-8-sig)."""
    lookup = {}
    with open(STAGE2_MANIFEST_ORIG, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rid = row.get("row_id", "")
            sc  = row.get("score_original", "")
            if rid and sc:
                lookup[rid] = sc
    return lookup


# ── 5ch crop 추출 ──────────────────────────────────────────────────────────────
def apply_lung_window(hu_arr):
    import numpy as np
    return (hu_arr.clip(HU_MIN, HU_MAX).astype(np.float32) - HU_MIN) / HU_RANGE


def get_z_indices(z, D):
    return [max(0, min(D - 1, z + off)) for off in (-2, -1, 0, 1, 2)]


def extract_5ch_stage2(ct_vol, mask_vol, z, pos_y0, pos_x0, pos_y1, pos_x1):
    """
    32x32 position → center±48 → 96x96 crop
    returns dict or None+error_str
    """
    import numpy as np

    D, H, W = ct_vol.shape

    # coordinate conversion
    cy = (pos_y0 + pos_y1) // 2
    cx = (pos_x0 + pos_x1) // 2
    cy0, cy1 = cy - 48, cy + 48
    cx0, cx1 = cx - 48, cx + 48

    if cy0 < 0 or cy1 > H or cx0 < 0 or cx1 > W:
        return None, f"crop out of bounds: cy0={cy0} cy1={cy1} H={H} cx0={cx0} cx1={cx1} W={W}"

    z_effs = get_z_indices(z, D)
    nearest_repeat = (z_effs[0] == z_effs[1]) or (z_effs[3] == z_effs[4])

    # 5ch crop (lung window, before mask zeroing)
    chans = [apply_lung_window(ct_vol[zi, cy0:cy1, cx0:cx1]) for zi in z_effs]
    crop_raw = np.stack(chans, axis=0)  # (5, 96, 96)

    if crop_raw.shape != (5, CROP_SIZE, CROP_SIZE):
        return None, f"crop shape {crop_raw.shape}"

    # crop stats before zeroing
    crop_min  = float(crop_raw.min())
    crop_max  = float(crop_raw.max())
    crop_mean = float(crop_raw.mean())
    crop_std  = float(crop_raw.std())

    # mask at center z
    if mask_vol is not None and z < mask_vol.shape[0]:
        m_center = mask_vol[z, cy0:cy1, cx0:cx1].astype(np.float32)
    else:
        m_center = np.ones((CROP_SIZE, CROP_SIZE), dtype=np.float32)

    # shape guard
    if m_center.shape != (CROP_SIZE, CROP_SIZE):
        m2 = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        hh, ww = min(m_center.shape[0], CROP_SIZE), min(m_center.shape[1], CROP_SIZE)
        m2[:hh, :ww] = m_center[:hh, :ww]
        m_center = m2

    mask_area_center = float(m_center.sum())

    # mask area across 5 z-slices
    if mask_vol is not None:
        areas = []
        for zi in z_effs:
            if zi < mask_vol.shape[0]:
                mz = mask_vol[zi, cy0:cy1, cx0:cx1]
                if mz.shape == (CROP_SIZE, CROP_SIZE):
                    areas.append(float(mz.sum()))
                else:
                    areas.append(mask_area_center)
            else:
                areas.append(mask_area_center)
    else:
        areas = [float(CROP_SIZE * CROP_SIZE)] * 5
    mask_area_5ch_mean = float(sum(areas) / len(areas))

    # lung exterior zeroing
    crop = crop_raw * m_center[None]
    roi_ratio = mask_area_center / max(1, CROP_SIZE * CROP_SIZE)

    return {
        "crop": crop,                        # (5, 96, 96)
        "m_center": m_center,                # (96, 96)
        "roi_ratio": roi_ratio,
        "nearest_repeat": nearest_repeat,
        "z_effs": z_effs,
        "mask_area_center": mask_area_center,
        "mask_area_5ch_mean": mask_area_5ch_mean,
        "crop_min": crop_min,
        "crop_max": crop_max,
        "crop_mean": crop_mean,
        "crop_std": crop_std,
        "crop_y0": cy0, "crop_y1": cy1,
        "crop_x0": cx0, "crop_x1": cx1,
    }, None


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

    def hook(name):
        def _h(m, inp, out):
            feats[name] = out.detach()
        return _h

    handles = [
        teacher.layer1.register_forward_hook(hook("l1")),
        teacher.layer2.register_forward_hook(hook("l2")),
        teacher.layer3.register_forward_hook(hook("l3")),
    ]
    with torch.no_grad():
        teacher(x)
    for h in handles:
        h.remove()
    return feats


def batch_score(teacher, student, crops_t, masks_t):
    """
    crops_t : (B, 5, 96, 96) tensor already on device
    masks_t : (B, 1, 96, 96) tensor already on device
    returns : (raw_list, l1_list, l2_list, l3_list) each B floats
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        tf = get_teacher_feats(teacher, crops_t)
        tl1, tl2, tl3 = tf["l1"], tf["l2"], tf["l3"]
        sl1, sl2, sl3 = student(tl3)

    def resize_mask(m, h, w):
        return F.interpolate(m, size=(h, w), mode="nearest")

    def masked_cos_err(t, s, m):
        cos = F.cosine_similarity(t, s, dim=1, eps=1e-6)  # (B, H, W)
        err = 1.0 - cos
        mq = m.squeeze(1)
        losses = []
        for i in range(err.shape[0]):
            denom = float(mq[i].sum()) + 1e-6
            losses.append(float((err[i] * mq[i]).sum()) / denom)
        return losses

    ml1 = resize_mask(masks_t, tl1.shape[2], tl1.shape[3])
    ml2 = resize_mask(masks_t, tl2.shape[2], tl2.shape[3])
    ml3 = resize_mask(masks_t, tl3.shape[2], tl3.shape[3])

    ll1 = masked_cos_err(tl1, sl1, ml1)
    ll2 = masked_cos_err(tl2, sl2, ml2)
    ll3 = masked_cos_err(tl3, sl3, ml3)
    raw = [(a + b + c) / 3.0 for a, b, c in zip(ll1, ll2, ll3)]
    return raw, ll1, ll2, ll3


# ── lung z percentile ─────────────────────────────────────────────────────────
def compute_lung_z_range(mask_vol):
    import numpy as np
    D = mask_vol.shape[0]
    areas = [int(mask_vol[z].sum()) for z in range(D)]
    idx = [i for i, a in enumerate(areas) if a > 0]
    if not idx:
        return 0, D - 1
    return idx[0], idx[-1]


def lung_z_pct(z, z_min, z_max):
    span = max(1, z_max - z_min)
    return round((z - z_min) / span, 4)


# ── error row helper ───────────────────────────────────────────────────────────
def make_error_row(row, status, msg):
    pos_y0 = int(row["y0"]); pos_x0 = int(row["x0"])
    pos_y1 = int(row["y1"]); pos_x1 = int(row["x1"])
    cy = (pos_y0 + pos_y1) // 2
    cx = (pos_x0 + pos_x1) // 2
    out = {f: "" for f in OUT_FIELDS}
    out.update({
        "row_id"        : row.get("row_id", ""),
        "patient_id"    : row.get("patient_id", ""),
        "safe_id"       : row.get("safe_id", ""),
        "local_z"       : row.get("local_z", ""),
        "pos_y0": pos_y0, "pos_x0": pos_x0,
        "pos_y1": pos_y1, "pos_x1": pos_x1,
        "crop_y0": cy - 48, "crop_x0": cx - 48,
        "crop_y1": cy + 48, "crop_x1": cx + 48,
        "track_id"      : row.get("track_id", ""),
        "track_len"     : row.get("track_len", ""),
        "track_z_start" : row.get("track_z_start", ""),
        "track_z_end"   : row.get("track_z_end", ""),
        "shard_id"      : row.get("shard_id", ""),
        "status"        : status,
        "error_message" : str(msg)[:200],
    })
    return out


# ── finalize (모든 shard 완료 시) ─────────────────────────────────────────────
def check_and_finalize(is_smoke):
    if is_smoke:
        return

    shard_verdicts = {}
    for i in range(N_SHARDS):
        sf = OUT_DIR / f"shard_{i:03d}_summary.json"
        if sf.exists():
            with open(sf) as f:
                shard_verdicts[i] = json.load(f).get("verdict", "MISSING")
        else:
            shard_verdicts[i] = "MISSING"

    all_done = all(v != "MISSING" for v in shard_verdicts.values())
    if not all_done:
        missing = [i for i, v in shard_verdicts.items() if v == "MISSING"]
        print(f"\n  [INFO] 미완료 shard: {missing} — 전체 finalize 대기")
        return

    total_scored = total_failed = total_nan = total_inf = 0
    for i in range(N_SHARDS):
        with open(OUT_DIR / f"shard_{i:03d}_summary.json") as f:
            d = json.load(f)
        total_scored += d.get("scored_rows", 0)
        total_failed += d.get("failed_rows", 0)
        total_nan    += d.get("nan_count", 0)
        total_inf    += d.get("inf_count", 0)

    all_pass = all(v.startswith("PASS") for v in shard_verdicts.values())

    if (all_pass and total_failed == 0 and total_nan == 0
            and total_inf == 0 and total_scored == EXPECTED_TOTAL_ROWS):
        final_verdict = "PASS_STEP12_STAGE2_FIXED_SCORING"
    elif all_pass and total_scored > 0:
        final_verdict = f"PARTIAL_PASS_STEP12_STAGE2_FIXED_SCORING_ERRORS_{total_failed}"
    else:
        final_verdict = "BLOCKED_STEP12_STAGE2_FIXED_SCORING"

    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_MANIFESTS.mkdir(parents=True, exist_ok=True)

    guardrail = {
        "plan_lock_loaded": True,
        "step11_stage2_preflight_passed": True,
        "actual_stage2_scoring_executed": True,
        "fixed_eval_only": True,
        "primary_candidate_score_locked": PRIMARY_CANDIDATE_SCORE,
        "primary_track_score_locked": PRIMARY_TRACK_SCORE,
        "P1_rejected_for_lung5ch": True,
        "P2_rejected_for_lung5ch": True,
        "p90_threshold": P90_THRESHOLD,
        "p90_recomputed_on_stage2": False,
        "threshold_tuning_executed": False,
        "score_family_changed_on_stage2": False,
        "candidate_deletion_executed": False,
        "representative_only_scoring_used": False,
        "survived_candidates_all_scored": (total_scored == EXPECTED_TOTAL_ROWS),
        "stage2_label_used_for_tuning": False,
        "stage2_label_used_for_metric": False,
        "training_executed": False,
        "backward_executed": False,
        "optimizer_created": False,
        "checkpoint_saved": False,
        "checkpoint_modified": False,
        "convae_branch_created": False,
        "image_reconstruction_loss_used": False,
        "student_decoder_keys": "dl3/dl2/dl1",
    }

    summary = {
        "step": "step12_stage2_fixed_scoring",
        "verdict": final_verdict,
        "created": str(date.today()),
        "expected_total_rows": EXPECTED_TOTAL_ROWS,
        "scored_rows": total_scored,
        "failed_rows": total_failed,
        "nan_count": total_nan,
        "inf_count": total_inf,
        "n_shards": N_SHARDS,
        "shard_verdicts": shard_verdicts,
        "checkpoint_path": str(CKPT_BEST),
        "student_decoder_keys": "dl3/dl2/dl1",
        "primary_candidate_score_locked": PRIMARY_CANDIDATE_SCORE,
        "primary_track_score_locked": PRIMARY_TRACK_SCORE,
        "guardrail": guardrail,
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [SAVED] {SUMMARY_JSON}")

    done = {
        "step": "step12_stage2_fixed_scoring",
        "verdict": final_verdict,
        "created": str(date.today()),
        "scored_rows": total_scored,
        "failed_rows": total_failed,
        "n_shards": N_SHARDS,
        "shard_verdicts": shard_verdicts,
        "output_dir": str(OUT_DIR),
        "summary_json": str(SUMMARY_JSON),
    }
    with open(DONE_OUT, "w") as f:
        json.dump(done, f, indent=2)
    print(f"  [SAVED] {DONE_OUT}")

    # shard status CSV
    with open(SHARD_STATUS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["shard_id", "verdict", "scored", "failed", "nan", "inf"])
        w.writeheader()
        for i in range(N_SHARDS):
            with open(OUT_DIR / f"shard_{i:03d}_summary.json") as sf:
                d = json.load(sf)
            w.writerow({
                "shard_id": i,
                "verdict" : d.get("verdict"),
                "scored"  : d.get("scored_rows", 0),
                "failed"  : d.get("failed_rows", 0),
                "nan"     : d.get("nan_count", 0),
                "inf"     : d.get("inf_count", 0),
            })
    print(f"  [SAVED] {SHARD_STATUS_CSV}")

    # report.md
    lines = [
        f"# Step 12 Stage2 Fixed Scoring Report",
        f"",
        f"**verdict**: {final_verdict}",
        f"",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| expected_total_rows | {EXPECTED_TOTAL_ROWS:,} |",
        f"| scored_rows | {total_scored:,} |",
        f"| failed_rows | {total_failed} |",
        f"| nan_count | {total_nan} |",
        f"| inf_count | {total_inf} |",
        f"| checkpoint | student_best_val_loss.pth |",
        f"| student_decoder_keys | dl3/dl2/dl1 |",
        f"| primary_candidate_score | {PRIMARY_CANDIDATE_SCORE} |",
        f"| primary_track_score | {PRIMARY_TRACK_SCORE} |",
        f"",
        f"## Shard Status",
        f"",
    ]
    for i, v in shard_verdicts.items():
        lines.append(f"- shard {i:03d}: {v}")
    lines += [
        f"",
        f"## Guardrail",
        f"",
        f"- training_executed: False",
        f"- backward_executed: False",
        f"- optimizer_created: False",
        f"- checkpoint_saved: False",
        f"- stage2_label_used_for_tuning: False",
        f"- p90_recomputed_on_stage2: False",
        f"- score_family_changed: False",
        f"- P1_rejected: True",
        f"- P2_rejected: True",
        f"",
        f"## Next Step",
        f"",
        f"Step 13: stage2 fixed scoring merge + integrity check",
    ]
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"  [SAVED] {REPORT_MD}")

    print()
    print("=" * 64)
    print(f"최종 판정: {final_verdict}")
    print(f"  expected : {EXPECTED_TOTAL_ROWS:,}")
    print(f"  scored   : {total_scored:,}")
    print(f"  failed   : {total_failed}")
    print(f"  NaN      : {total_nan}")
    print(f"  Inf      : {total_inf}")
    print("=" * 64)
    if final_verdict == "PASS_STEP12_STAGE2_FIXED_SCORING":
        print("\nStep 12 완료. 다음 단계: Step 13 stage2 fixed scoring merge + integrity check (사용자 승인 후)")


# ── shard scoring 메인 ─────────────────────────────────────────────────────────
def run_shard(args):
    import numpy as np
    import torch

    shard_id = args.shard_id
    max_rows = args.max_rows
    is_smoke = (max_rows is not None)

    print()
    print("=" * 64)
    print(f"Step 12 Stage2 Fixed Scoring — SHARD {shard_id:03d}")
    if is_smoke:
        print(f"  *** SMOKE TEST (max_rows={max_rows}) ***")
    print("=" * 64)

    # [0] guards
    print("\n[0] Guards 확인")
    guard_errors = check_guards()
    if guard_errors:
        print("  [BLOCKED] Guard 실패:")
        for e in guard_errors:
            print(f"    - {e}")
        sys.exit(1)
    if shard_id < 0 or shard_id >= N_SHARDS:
        print(f"  [BLOCKED] shard_id {shard_id} out of range", file=sys.stderr)
        sys.exit(1)
    print("  [PASS]")

    # resume 확인
    shard_csv     = OUT_DIR / f"shard_{shard_id:03d}_scores.csv"
    shard_summary = OUT_DIR / f"shard_{shard_id:03d}_summary.json"
    if not is_smoke and shard_summary.exists():
        with open(shard_summary) as f:
            prev = json.load(f)
        if str(prev.get("verdict", "")).startswith("PASS") and not prev.get("smoke_test"):
            print(f"  [SKIP] shard {shard_id} already PASS: {prev['verdict']}")
            check_and_finalize(is_smoke)
            return
        else:
            reason = "smoke_test result" if prev.get("smoke_test") else f"verdict={prev.get('verdict')}"
            print(f"  [RESUME] {reason}, re-running shard {shard_id}")

    # [1] manifest + score_original 로드
    print("\n[1] Manifest 로드 (shard 필터)")
    rows = []
    with open(STEP12_MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["shard_id"]) == shard_id:
                rows.append(r)
    if max_rows is not None:
        rows = rows[:max_rows]
    expected_rows = len(rows)
    print(f"  shard {shard_id}: {expected_rows:,}행")
    if expected_rows == 0:
        print(f"  [BLOCKED] 0행 — shard_id 확인 필요", file=sys.stderr)
        sys.exit(1)

    print("  score_original 조회 테이블 로드 중...")
    score_orig_lookup = load_score_original_lookup()
    print(f"  lookup 크기: {len(score_orig_lookup):,}건")

    # [2] 모델 로드
    print("\n[2] 모델 + checkpoint 로드")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = build_teacher(device)
    student = build_student(device)
    ckpt = torch.load(str(CKPT_BEST), map_location=device)
    student.load_state_dict(ckpt["student_state_dict"])
    student.eval()
    print(f"  checkpoint epoch   : {ckpt.get('epoch')}")
    print(f"  checkpoint val_loss: {ckpt.get('val_loss', float('nan')):.6f}")
    print(f"  teacher frozen     : {all(not p.requires_grad for p in teacher.parameters())}")
    print(f"  device             : {device}")

    # StudentDecoder key 확인
    sd_keys = list(student.state_dict().keys())
    prefixes = sorted(set(k.split(".")[0] for k in sd_keys))
    print(f"  student key prefixes: {prefixes}")
    expected_pfx = {"ocbe", "dl3", "dl2", "dl1"}
    if set(prefixes) != expected_pfx:
        print(f"  [BLOCKED] decoder key prefix 불일치: {prefixes} != {sorted(expected_pfx)}", file=sys.stderr)
        sys.exit(1)
    print(f"  decoder keys: {prefixes}  [OK]")

    # [3] scoring
    print(f"\n[3] Scoring 시작 (shard {shard_id})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LOGS.mkdir(parents=True, exist_ok=True)

    # patient별 그룹화
    patient_rows = defaultdict(list)
    for r in rows:
        patient_rows[r["safe_id"]].append(r)

    cnt = {"scored": 0, "failed": 0, "nan": 0, "inf": 0}
    error_rows_local = []
    t_start = time.perf_counter()

    with open(shard_csv, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=OUT_FIELDS)
        writer.writeheader()

        n_pts = len(patient_rows)

        # ── flush_batch: shard 내 클로저 ─────────────────────────────────────
        def flush_batch(b_crops, b_masks, b_meta, b_extras,
                        lz_min, lz_max):
            if not b_crops:
                return
            crops_t = torch.from_numpy(np.stack(b_crops, axis=0)).to(device)
            masks_t = torch.from_numpy(
                np.stack([m[None] for m in b_masks], axis=0)).to(device)

            raw_list, l1_list, l2_list, l3_list = batch_score(
                teacher, student, crops_t, masks_t)

            for i, meta in enumerate(b_meta):
                raw = raw_list[i]
                cnt["nan"] += int(math.isnan(raw))
                cnt["inf"] += int(math.isinf(raw))

                ex = b_extras[i]
                pos_y0 = int(meta["y0"]); pos_x0 = int(meta["x0"])
                pos_y1 = int(meta["y1"]); pos_x1 = int(meta["x1"])

                out_row = {
                    "row_id"             : meta["row_id"],
                    "patient_id"         : meta["patient_id"],
                    "safe_id"            : meta["safe_id"],
                    "local_z"            : meta["local_z"],
                    "score_original"     : score_orig_lookup.get(meta["row_id"], ""),
                    "pos_y0": pos_y0, "pos_x0": pos_x0,
                    "pos_y1": pos_y1, "pos_x1": pos_x1,
                    "crop_y0"            : ex["crop_y0"],
                    "crop_x0"            : ex["crop_x0"],
                    "crop_y1"            : ex["crop_y1"],
                    "crop_x1"            : ex["crop_x1"],
                    "z_minus2_effective" : ex["z_effs"][0],
                    "z_minus1_effective" : ex["z_effs"][1],
                    "z_center_effective" : ex["z_effs"][2],
                    "z_plus1_effective"  : ex["z_effs"][3],
                    "z_plus2_effective"  : ex["z_effs"][4],
                    "nearest_repeat_used": ex["nearest_repeat"],
                    "track_id"           : meta["track_id"],
                    "track_len"          : meta["track_len"],
                    "track_z_start"      : meta["track_z_start"],
                    "track_z_end"        : meta["track_z_end"],
                    "lung_z_percentile"  : lung_z_pct(int(meta["local_z"]), lz_min, lz_max),
                    "mask_area_center"   : round(ex["mask_area_center"], 2),
                    "mask_area_5ch_mean" : round(ex["mask_area_5ch_mean"], 2),
                    "crop_min"           : round(ex["crop_min"], 6),
                    "crop_max"           : round(ex["crop_max"], 6),
                    "crop_mean"          : round(ex["crop_mean"], 6),
                    "crop_std"           : round(ex["crop_std"], 6),
                    "roi_ratio"          : round(ex["roi_ratio"], 6),
                    "rd4ad_lung5ch_score_raw": round(raw, 8),
                    "score_layer1"       : round(l1_list[i], 8),
                    "score_layer2"       : round(l2_list[i], 8),
                    "score_layer3"       : round(l3_list[i], 8),
                    "shard_id"           : meta["shard_id"],
                    "status"             : "SCORED",
                    "error_message"      : "",
                }
                writer.writerow(out_row)
                cnt["scored"] += 1

        # ── patient loop ──────────────────────────────────────────────────────
        for pi, (safe_id, pt_rows) in enumerate(sorted(patient_rows.items())):
            pt_id = pt_rows[0]["patient_id"]

            ct_path = NSCLC_CT_ROOT / safe_id / "ct_hu.npy"
            if not ct_path.exists():
                msg = f"CT 없음: {ct_path}"
                error_rows_local.append({"patient_id": pt_id, "safe_id": safe_id, "msg": msg})
                cnt["failed"] += len(pt_rows)
                for r in pt_rows:
                    writer.writerow(make_error_row(r, "CT_MISSING", msg))
                print(f"  [{pi+1:3d}/{n_pts}] {safe_id[:40]}  ERROR: CT 없음")
                continue

            ct_vol = np.load(str(ct_path), mmap_mode="r")
            mask_path = LESION_MASK_ROOT / safe_id / "refined_roi.npy"
            mask_vol = np.load(str(mask_path), mmap_mode="r") if mask_path.exists() else None

            if mask_vol is not None:
                lz_min, lz_max = compute_lung_z_range(mask_vol)
            else:
                lz_min, lz_max = 0, ct_vol.shape[0] - 1

            b_crops, b_masks, b_meta, b_extras = [], [], [], []

            for row in pt_rows:
                z = int(row["local_z"])
                if z >= ct_vol.shape[0]:
                    msg = f"z={z} >= D={ct_vol.shape[0]}"
                    error_rows_local.append({"patient_id": pt_id, "safe_id": safe_id, "msg": msg})
                    cnt["failed"] += 1
                    writer.writerow(make_error_row(row, "Z_OUT_OF_RANGE", msg))
                    continue

                crop_info, err = extract_5ch_stage2(
                    ct_vol, mask_vol, z,
                    int(row["y0"]), int(row["x0"]),
                    int(row["y1"]), int(row["x1"]),
                )
                if crop_info is None:
                    error_rows_local.append({"patient_id": pt_id, "safe_id": safe_id, "msg": err})
                    cnt["failed"] += 1
                    writer.writerow(make_error_row(row, "CROP_ERROR", err))
                    continue

                b_crops.append(crop_info["crop"])
                b_masks.append(crop_info["m_center"])
                b_meta.append(row)
                b_extras.append(crop_info)

                if len(b_crops) >= BATCH_SIZE:
                    flush_batch(b_crops, b_masks, b_meta, b_extras, lz_min, lz_max)
                    b_crops.clear(); b_masks.clear(); b_meta.clear(); b_extras.clear()

            flush_batch(b_crops, b_masks, b_meta, b_extras, lz_min, lz_max)

            elapsed = time.perf_counter() - t_start
            print(f"  [{pi+1:3d}/{n_pts}] {safe_id[:50]:50s}  "
                  f"done={cnt['scored']:,} err={cnt['failed']}  {elapsed:.1f}s")

    elapsed_total = time.perf_counter() - t_start
    scored  = cnt["scored"]
    failed  = cnt["failed"]

    # shard verdict
    if (failed == 0 and cnt["nan"] == 0 and cnt["inf"] == 0
            and scored == expected_rows):
        verdict = f"PASS_STEP12_SHARD_{shard_id:03d}"
    elif scored > 0 and failed < expected_rows * 0.01:
        verdict = f"PARTIAL_PASS_STEP12_SHARD_{shard_id:03d}_FEW_ERRORS"
    elif scored > 0:
        verdict = f"PARTIAL_PASS_STEP12_SHARD_{shard_id:03d}_ERRORS_{failed}"
    else:
        verdict = f"BLOCKED_STEP12_SHARD_{shard_id:03d}_NO_SCORED"

    print()
    print("=" * 64)
    print(f"판정: {verdict}")
    print(f"  expected  : {expected_rows:,}")
    print(f"  scored    : {scored:,}")
    print(f"  failed    : {failed}")
    print(f"  NaN       : {cnt['nan']}")
    print(f"  Inf       : {cnt['inf']}")
    print(f"  runtime   : {elapsed_total:.1f}s")
    print("=" * 64)

    shard_summ = {
        "step"             : "step12_stage2_fixed_scoring",
        "shard_id"         : shard_id,
        "verdict"          : verdict,
        "created"          : str(date.today()),
        "smoke_test"       : is_smoke,
        "max_rows"         : max_rows,
        "expected_rows"    : expected_rows,
        "scored_rows"      : scored,
        "failed_rows"      : failed,
        "nan_count"        : cnt["nan"],
        "inf_count"        : cnt["inf"],
        "runtime_s"        : round(elapsed_total, 2),
        "checkpoint_path"  : str(CKPT_BEST),
        "student_decoder_keys": "dl3/dl2/dl1",
        "primary_candidate_score": PRIMARY_CANDIDATE_SCORE,
        "primary_track_score"    : PRIMARY_TRACK_SCORE,
        "guardrail": {
            "training_executed"             : False,
            "backward_executed"             : False,
            "optimizer_created"             : False,
            "checkpoint_saved"              : False,
            "checkpoint_modified"           : False,
            "stage2_label_used_for_tuning"  : False,
            "stage2_label_used_for_metric"  : False,
            "score_family_changed"          : False,
            "P1_used_as_primary"            : False,
            "P2_used_as_primary"            : False,
            "p90_recomputed_on_stage2"      : False,
            "threshold_tuning_executed"     : False,
            "candidate_deletion_executed"   : False,
            "representative_only_scoring_used": False,
        },
    }
    with open(shard_summary, "w") as f:
        json.dump(shard_summ, f, indent=2)
    print(f"  [SAVED] {shard_summary}")
    print(f"  [SAVED] {shard_csv} ({scored:,} rows)")

    # errors append
    if error_rows_local:
        err_exists = ERRORS_CSV.exists()
        with open(ERRORS_CSV, "a", newline="") as ef:
            ew = csv.DictWriter(ef, fieldnames=["shard_id", "patient_id", "safe_id", "msg"])
            if not err_exists:
                ew.writeheader()
            for e in error_rows_local:
                ew.writerow({"shard_id": shard_id, **e})

    check_and_finalize(is_smoke)


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.dry_run:
        print_dry_run()
        sys.exit(0)

    if args.run_shard:
        missing = []
        if not args.confirm_plan_lock:
            missing.append("--confirm-plan-lock")
        if not args.confirm_fixed_score:
            missing.append("--confirm-fixed-score")
        if not args.confirm_no_training:
            missing.append("--confirm-no-training")
        if missing:
            print(f"[BLOCKED] 필수 flags 누락: {missing}", file=sys.stderr)
            sys.exit(2)
        if args.shard_id < 0:
            print("[BLOCKED] --shard-id 필요", file=sys.stderr)
            sys.exit(2)
        run_shard(args)
        return

    print("[BLOCKED] --dry-run 또는 --run-shard 필요", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
