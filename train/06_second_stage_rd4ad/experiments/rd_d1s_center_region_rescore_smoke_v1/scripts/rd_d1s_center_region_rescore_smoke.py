#!/usr/bin/env python3
"""
RD-D1s center-region rescoring smoke test v1

목적: 기존 RD-D1s scalar score(전체 공간 평균) 대신 96×96 crop 중심 영역만
      재집계한 score가 ranking/safety를 개선하는지 확인하는 smoke test.

실행:
  dry-run:
    python rd_d1s_center_region_rescore_smoke.py --dry-run
  실제 smoke:
    python rd_d1s_center_region_rescore_smoke.py --run-smoke --confirm-model-forward --confirm-stage1dev-only

금지 사항:
  - training / fine-tuning / backward
  - full scoring (1,000개 초과 forward 금지)
  - stage2_holdout 접근
  - 기존 artifact 수정
  - checkpoint 저장
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

# =============================================================================
# 경로 상수 (read-only)
# =============================================================================

PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/rd_d1s_center_region_rescore_smoke_v1"

# 기존 RD-D1s 결과 (read-only)
RD_D1S_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1s_medi3ch_true_rd4ad_shard_run_v1"
    / "rd_d1s_stage1dev_candidate_score.csv"
)

# 기존 candidate manifest (read-only)
CANDIDATE_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c2_effb0_v420_candidate_rd4ad_retest_v1"
    / "rd_c2_effb0_v420_candidate_manifest.csv"
)

# D2 problem patient file (read-only)
D2_PROBLEM_PATIENTS_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d2_rdd1s_ranking_guard_strategy_v1"
    / "rd_d2_patient_all_suppressed_risk_cases.csv"
)
D2_TOPK_RETENTION_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d2_rdd1s_ranking_guard_strategy_v1"
    / "rd_d2_patient_level_topk_retention_summary.csv"
)

# checkpoint (read-only)
CKPT_DIR = PROJECT_ROOT / "outputs/models/rd_d1s_true_rd4ad_resnet18_medi3ch_shard_v1/checkpoints"
CKPT_PATH = CKPT_DIR / "best_train_loss.pth"
LOCAL_RESNET_WEIGHT = Path("/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")

# CT root (read-only)
CANDIDATE_CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)

# 출력 (새 폴더에만)
OUTPUT_ROOT = EXPERIMENT_ROOT
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
REPORT_DIR   = OUTPUT_ROOT / "reports"
LOG_DIR      = OUTPUT_ROOT / "logs"

SMOKE_MANIFEST_CSV  = MANIFEST_DIR / "rd_d1s_center_region_rescore_smoke_manifest.csv"
SMOKE_SCORES_CSV    = MANIFEST_DIR / "rd_d1s_center_region_rescore_smoke_scores.csv"
PATIENT_SUMMARY_CSV = MANIFEST_DIR / "rd_d1s_center_region_rescore_patient_summary.csv"
ERROR_CSV           = LOG_DIR / "errors.csv"
REPORT_MD           = REPORT_DIR / "rd_d1s_center_region_rescore_smoke_report.md"
SUMMARY_JSON        = REPORT_DIR / "rd_d1s_center_region_rescore_smoke_summary.json"
DONE_JSON           = OUTPUT_ROOT / "DONE.json"

# smoke 제한
MAX_PATIENTS   = 20
MAX_PER_PATIENT = 50
MAX_TOTAL      = 1000
CROP_SIZE      = 96
HU_MIN, HU_MAX = -160.0, 240.0

# guardrail 기록
GUARDRAILS = {
    "stage2_holdout_accessed":       False,
    "checkpoint_loaded":             False,
    "model_forward_executed":        False,
    "training_executed":             False,
    "backward_executed":             False,
    "optimizer_created":             False,
    "checkpoint_saved":              False,
    "full_scoring_executed":         False,
    "threshold_recalculated":        False,
    "existing_artifact_modified":    False,
    "existing_script_modified":      False,
    "label_used_for_smoke_sampling": True,
    "label_used_as_deployment_selector": False,
    "output_overwrite":              False,
    "max_forward_candidates":        MAX_TOTAL,
    "actual_forward_candidates":     0,
}


# =============================================================================
# 안전 경로 검사
# =============================================================================

def assert_path_safe(p: Path):
    """stage2_holdout 경로 접근 차단."""
    s = str(p).lower()
    if "stage2_holdout" in s or "holdout" in s:
        GUARDRAILS["stage2_holdout_accessed"] = True
        raise RuntimeError(f"[ABORT] stage2_holdout 경로 접근 시도 차단: {p}")
    if "stage2" in s and "holdout" in s:
        GUARDRAILS["stage2_holdout_accessed"] = True
        raise RuntimeError(f"[ABORT] stage2 holdout 경로 접근 시도 차단: {p}")


def check_no_existing_modified(paths_written):
    """기존 artifact 수정 여부 확인."""
    for p in paths_written:
        p = Path(p)
        if not str(p).startswith(str(OUTPUT_ROOT)):
            GUARDRAILS["existing_artifact_modified"] = True
            raise RuntimeError(f"[ABORT] 기존 파일 수정 시도: {p}")


# =============================================================================
# CSV 유틸
# =============================================================================

def read_csv(path: Path):
    assert_path_safe(path)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def write_csv(path: Path, rows, fieldnames=None):
    check_no_existing_modified([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_error(msg: str, context: str = ""):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    existed = ERROR_CSV.exists()
    with open(ERROR_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "context", "message"])
        if not existed:
            writer.writeheader()
        writer.writerow({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "context": context,
            "message": msg,
        })


# =============================================================================
# sklearn-free AUROC / AUPRC
# =============================================================================

def compute_auroc_mann_whitney(y_true, y_score):
    import numpy as np
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    valid   = np.isfinite(y_score)
    y_true  = y_true[valid]; y_score = y_score[valid]
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score)
    sorted_s = y_score[order]
    ranks = np.empty(len(sorted_s), dtype=float)
    i = 0
    while i < len(sorted_s):
        j = i + 1
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        i = j
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    u = float(original_ranks[y_true == 1].sum()) - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def compute_average_precision(y_true, y_score):
    import numpy as np
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    valid   = np.isfinite(y_score)
    y_true  = y_true[valid]; y_score = y_score[valid]
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-y_score)
    y_sorted = y_true[order]; s_sorted = y_score[order]
    tp = 0; fp = 0; prev_recall = 0.0; ap = 0.0; i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        group = y_sorted[i:j]
        tp += int((group == 1).sum())
        fp += int((group == 0).sum())
        recall = tp / n_pos
        precision = tp / max(tp + fp, 1)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return float(ap)


def spearman_corr(x, y):
    import numpy as np
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]; y = y[valid]
    n = len(x)
    if n < 2:
        return None
    rx = np.argsort(np.argsort(x)).astype(float) + 1
    ry = np.argsort(np.argsort(y)).astype(float) + 1
    d2 = ((rx - ry) ** 2).sum()
    return float(1.0 - 6 * d2 / (n * (n * n - 1)))


# =============================================================================
# 모델 빌드 (read-only, 기존 코드와 동일 구조)
# =============================================================================

def build_teacher():
    import torch
    import torchvision.models as models
    resnet = models.resnet18(weights=None)
    state_dict = torch.load(str(LOCAL_RESNET_WEIGHT), map_location="cpu", weights_only=True)
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


# =============================================================================
# 학습 안전 체크 (optimizer / backward 금지)
# =============================================================================

def _forbidden_train(*args, **kwargs):
    GUARDRAILS["training_executed"] = True
    raise RuntimeError("[ABORT] training 호출 금지됨")


def _forbidden_backward(*args, **kwargs):
    GUARDRAILS["backward_executed"] = True
    raise RuntimeError("[ABORT] backward 호출 금지됨")


# =============================================================================
# HU 윈도잉
# =============================================================================

def hu_window(arr):
    import numpy as np
    out = (arr.astype(np.float32) - HU_MIN) / (HU_MAX - HU_MIN)
    return np.clip(out, 0.0, 1.0)


# =============================================================================
# 3ch crop 빌드 (reflect padding, 기존 방식과 동일)
# =============================================================================

def build_medi3ch_crop(ct_arr, local_z, y0, x0, y1, x1):
    """ct_arr: (Z,H,W) float32 or mmap → (3,96,96) float32 [0,1]"""
    import numpy as np
    Z, H, W = ct_arr.shape

    def _win(patch):
        return hu_window(patch)

    def _clip_and_pad(z_idx, cy0, cx0, cy1, cx1):
        cy0c = max(cy0, 0); cy1c = min(cy1, H)
        cx0c = max(cx0, 0); cx1c = min(cx1, W)
        if cy1c <= cy0c or cx1c <= cx0c:
            return np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        patch = ct_arr[z_idx, cy0c:cy1c, cx0c:cx1c].astype(np.float32)
        patch = _win(patch)
        out = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        dy0 = cy0c - cy0; dx0 = cx0c - cx0
        out[dy0:dy0 + (cy1c - cy0c), dx0:dx0 + (cx1c - cx0c)] = patch
        return out

    zs = [max(0, local_z - 1), local_z, min(Z - 1, local_z + 1)]
    channels = [_clip_and_pad(z, y0, x0, y1, x1) for z in zs]
    crop = np.stack(channels, axis=0)
    if crop.shape != (3, CROP_SIZE, CROP_SIZE):
        raise ValueError(f"crop shape {crop.shape} != (3,{CROP_SIZE},{CROP_SIZE})")
    if not np.isfinite(crop).all():
        raise ValueError("crop contains NaN/Inf")
    return crop.astype(np.float32)


# =============================================================================
# center-region score 계산
# =============================================================================

def center_region_indices(map_size, crop_ref=CROP_SIZE):
    """
    96×96 crop 기준 center32(32:64) / center16(40:56) 영역을
    feature map 해상도로 비례 변환한다.
    반환: (c32_y0, c32_y1, c32_x0, c32_x1, c16_y0, c16_y1, c16_x0, c16_x1)
    """
    H = W = map_size
    c32_y0 = round(H * 32 / crop_ref)
    c32_y1 = round(H * 64 / crop_ref)
    c32_x0 = round(W * 32 / crop_ref)
    c32_x1 = round(W * 64 / crop_ref)
    c16_y0 = round(H * 40 / crop_ref)
    c16_y1 = round(H * 56 / crop_ref)
    c16_x0 = round(W * 40 / crop_ref)
    c16_x1 = round(W * 56 / crop_ref)
    # 최소 1 픽셀 보장
    c32_y1 = max(c32_y1, c32_y0 + 1)
    c32_x1 = max(c32_x1, c32_x0 + 1)
    c16_y1 = max(c16_y1, c16_y0 + 1)
    c16_x1 = max(c16_x1, c16_x0 + 1)
    return c32_y0, c32_y1, c32_x0, c32_x1, c16_y0, c16_y1, c16_x0, c16_x1


def top_k_pct_mean(arr_flat, pct=5.0):
    """상위 pct% 평균. 최소 1개."""
    import numpy as np
    n = max(1, math.ceil(len(arr_flat) * pct / 100.0))
    return float(np.sort(arr_flat)[-n:].mean())


def compute_center_scores(error_maps):
    """
    error_maps: list of 2-D numpy arrays (H×W error map per level)
    각 레벨 error map의 center/full 통계를 평균 집계.

    반환 dict:
      full_map_mean, full_map_top5_mean
      center32_mean, center32_top5_mean, center32_max
      center16_mean, center16_top5_mean, center16_max
    """
    import numpy as np

    accum = {
        "full_map_mean": [], "full_map_top5_mean": [],
        "center32_mean": [], "center32_top5_mean": [], "center32_max": [],
        "center16_mean": [], "center16_top5_mean": [], "center16_max": [],
    }

    for emap in error_maps:
        H, W = emap.shape
        c32_y0, c32_y1, c32_x0, c32_x1, c16_y0, c16_y1, c16_x0, c16_x1 = center_region_indices(H)

        full_flat = emap.flatten()
        c32_flat  = emap[c32_y0:c32_y1, c32_x0:c32_x1].flatten()
        c16_flat  = emap[c16_y0:c16_y1, c16_x0:c16_x1].flatten()

        accum["full_map_mean"].append(float(full_flat.mean()))
        accum["full_map_top5_mean"].append(top_k_pct_mean(full_flat))
        accum["center32_mean"].append(float(c32_flat.mean()))
        accum["center32_top5_mean"].append(top_k_pct_mean(c32_flat))
        accum["center32_max"].append(float(c32_flat.max()))
        accum["center16_mean"].append(float(c16_flat.mean()))
        accum["center16_top5_mean"].append(top_k_pct_mean(c16_flat))
        accum["center16_max"].append(float(c16_flat.max()))

    return {k: float(np.mean(v)) for k, v in accum.items()}


# =============================================================================
# CT mmap 캐시
# =============================================================================

class CTMmapCache:
    def __init__(self, max_size=5):
        import collections
        self._cache = collections.OrderedDict()
        self._max = max_size

    def get(self, ct_path: Path):
        import numpy as np
        key = str(ct_path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        assert_path_safe(ct_path)
        arr = np.load(key, mmap_mode="r")
        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        self._cache[key] = arr
        return arr


# =============================================================================
# smoke 후보 선정
# =============================================================================

def select_smoke_candidates(score_rows, manifest_map, d2_problem_pids):
    """
    priority:
      A. D2 pat_all_sup 위험 환자 (문제 환자)
      B. D2 top1 retention이 낮은 환자 (topk_retention_csv에서)
      C. positive score 기준 rank가 낮은 환자 5명
      D. positive score 기준 rank가 높은 환자 5명
      E. hard_negative 고득점 환자 5명
      F. 무작위 대표 환자 5명

    label은 sampling용으로만 사용. deployment selector 금지.
    """
    import numpy as np

    # patient별 그룹
    pat_groups = defaultdict(list)
    for r in score_rows:
        if r.get("stage_split", "") != "stage1_dev":
            continue
        pat_groups[r["patient_id"]].append(r)

    # patient별 positive 후보 최고 score
    pat_pos_max   = {}
    pat_hn_max    = {}
    pat_pos_count = {}
    for pid, rows in pat_groups.items():
        pos_scores = [float(r["rd_d1s_medi3ch_rd4ad_score"])
                      for r in rows if r.get("label") == "positive"]
        hn_scores  = [float(r["rd_d1s_medi3ch_rd4ad_score"])
                      for r in rows if r.get("label") == "hard_negative"]
        pat_pos_count[pid] = len(pos_scores)
        pat_pos_max[pid]   = max(pos_scores) if pos_scores else -1.0
        pat_hn_max[pid]    = max(hn_scores) if hn_scores else -1.0

    selected_pids = []
    selection_reason = {}

    # A: D2 문제 환자
    for pid in d2_problem_pids:
        if pid in pat_groups and len(selected_pids) < MAX_PATIENTS:
            selected_pids.append(pid)
            selection_reason[pid] = "D2_pat_all_sup_risk"

    # positive가 있는 환자만
    pos_pids = sorted([p for p, c in pat_pos_count.items() if c > 0],
                      key=lambda p: pat_pos_max[p])

    # C: positive score 낮은 환자 5명 (positive가 밀린 환자)
    for pid in pos_pids[:5]:
        if pid not in selected_pids and len(selected_pids) < MAX_PATIENTS:
            selected_pids.append(pid)
            selection_reason[pid] = "low_pos_max_score"

    # D: positive score 높은 환자 5명
    for pid in pos_pids[-5:]:
        if pid not in selected_pids and len(selected_pids) < MAX_PATIENTS:
            selected_pids.append(pid)
            selection_reason[pid] = "high_pos_max_score"

    # E: hard_negative 고득점 환자 5명
    hn_sorted = sorted(pat_groups.keys(), key=lambda p: pat_hn_max.get(p, -1.0), reverse=True)
    for pid in hn_sorted[:8]:
        if pid not in selected_pids and len(selected_pids) < MAX_PATIENTS:
            selected_pids.append(pid)
            selection_reason[pid] = "high_hn_score"

    # F: 무작위 대표 환자 (seed 고정)
    all_pids = list(pat_groups.keys())
    np.random.seed(42)
    rand_pids = np.random.choice(all_pids, size=min(len(all_pids), 10), replace=False)
    for pid in rand_pids:
        if pid not in selected_pids and len(selected_pids) < MAX_PATIENTS:
            selected_pids.append(pid)
            selection_reason[pid] = "random_representative"

    # 후보 수집
    smoke_rows = []
    for pid in selected_pids:
        rows = pat_groups[pid]
        pos_rows = [r for r in rows if r.get("label") == "positive"]
        hn_rows  = [r for r in rows if r.get("label") == "hard_negative"]

        # patient당 최대 50개, positive 우선
        selected_rows = []
        # positive는 score 기준 상/하위 배분
        pos_sorted = sorted(pos_rows, key=lambda r: float(r["rd_d1s_medi3ch_rd4ad_score"]))
        pos_budget = min(len(pos_sorted), 25)
        selected_rows.extend(pos_sorted[:pos_budget // 2])
        selected_rows.extend(pos_sorted[-(pos_budget - pos_budget // 2):])
        # hard_negative score 높은 순
        hn_sorted_r = sorted(hn_rows, key=lambda r: float(r["rd_d1s_medi3ch_rd4ad_score"]), reverse=True)
        remaining = MAX_PER_PATIENT - len(selected_rows)
        selected_rows.extend(hn_sorted_r[:remaining])

        for r in selected_rows[:MAX_PER_PATIENT]:
            smoke_rows.append({
                "candidate_id":                r["candidate_id"],
                "patient_id":                  r["patient_id"],
                "safe_id":                     r.get("safe_id", ""),
                "local_z":                     r["local_z"],
                "crop_y0":                     r["crop_y0"],
                "crop_x0":                     r["crop_x0"],
                "crop_y1":                     r["crop_y1"],
                "crop_x1":                     r["crop_x1"],
                "label":                       r.get("label", ""),
                "first_stage_score":           manifest_map.get(r["candidate_id"], {}).get("first_stage_score", ""),
                "rd_d1s_medi3ch_rd4ad_score":  r["rd_d1s_medi3ch_rd4ad_score"],
                "sampling_reason":             selection_reason.get(pid, "unknown"),
                "smoke_selection_reason":      selection_reason.get(pid, "unknown"),
            })

    return smoke_rows


# =============================================================================
# dry-run
# =============================================================================

def run_dry():
    print("=" * 70)
    print("[DRY-RUN] RD-D1s center-region rescore smoke v1")
    print("=" * 70)

    issues = []

    # 1. 입력 파일 확인
    checks = {
        "RD-D1s score CSV":    RD_D1S_SCORE_CSV,
        "candidate manifest":  CANDIDATE_MANIFEST_CSV,
        "checkpoint":          CKPT_PATH,
        "ResNet18 weight":     LOCAL_RESNET_WEIGHT,
        "D2 problem patients": D2_PROBLEM_PATIENTS_CSV,
    }
    print("\n[1] 입력 파일 확인")
    for name, path in checks.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  {status:8s} {name}: {path}")
        if not exists:
            issues.append(f"MISSING: {name} ({path})")

    # 2. CANDIDATE_CT_ROOT 존재 확인
    print(f"\n[2] CT root 확인")
    ct_exists = CANDIDATE_CT_ROOT.exists()
    print(f"  {'OK' if ct_exists else 'MISSING':8s} {CANDIDATE_CT_ROOT}")
    if not ct_exists:
        issues.append(f"MISSING: CANDIDATE_CT_ROOT ({CANDIDATE_CT_ROOT})")

    # 3. stage2_holdout 없음 확인
    print(f"\n[3] stage2_holdout 접근 없음 확인")
    print(f"  OK  stage2_holdout 경로는 이 스크립트에서 참조하지 않음")

    # 4. output root 충돌 확인
    print(f"\n[4] output root 충돌 확인")
    for p in [SMOKE_SCORES_CSV, PATIENT_SUMMARY_CSV, SUMMARY_JSON, DONE_JSON]:
        if p.exists():
            print(f"  WARN 이미 존재 (overwrite 될 수 있음): {p}")
        else:
            print(f"  OK  없음: {p}")

    # 5. smoke 후보 plan (파일 있으면)
    print(f"\n[5] smoke 후보 선정 plan")
    if RD_D1S_SCORE_CSV.exists() and CANDIDATE_MANIFEST_CSV.exists():
        try:
            score_rows = read_csv(RD_D1S_SCORE_CSV)
            stage1_rows = [r for r in score_rows if r.get("stage_split") == "stage1_dev"]
            pos_count = sum(1 for r in stage1_rows if r.get("label") == "positive")
            hn_count  = sum(1 for r in stage1_rows if r.get("label") == "hard_negative")
            pids = set(r["patient_id"] for r in stage1_rows)
            print(f"  stage1_dev rows: {len(stage1_rows):,}")
            print(f"  positive: {pos_count:,} / hard_negative: {hn_count:,}")
            print(f"  unique patients: {len(pids):,}")

            d2_problem = set()
            if D2_PROBLEM_PATIENTS_CSV.exists():
                d2_rows = read_csv(D2_PROBLEM_PATIENTS_CSV)
                d2_problem = set(r["patient_id"] for r in d2_rows)
                print(f"  D2 problem patients: {sorted(d2_problem)}")
            else:
                print(f"  D2 problem CSV not found → fallback sampling 사용")

            manifest_rows = read_csv(CANDIDATE_MANIFEST_CSV)
            manifest_map = {r["candidate_id"]: r for r in manifest_rows}
            smoke_rows = select_smoke_candidates(stage1_rows, manifest_map, d2_problem)

            n_pat = len(set(r["patient_id"] for r in smoke_rows))
            n_pos = sum(1 for r in smoke_rows if r.get("label") == "positive")
            n_hn  = sum(1 for r in smoke_rows if r.get("label") == "hard_negative")
            print(f"  plan: {n_pat} 환자 / {len(smoke_rows)} 후보 (pos={n_pos}, hn={n_hn})")
            if len(smoke_rows) > MAX_TOTAL:
                issues.append(f"smoke 후보 수 {len(smoke_rows)} > MAX {MAX_TOTAL}")
        except Exception as e:
            issues.append(f"smoke plan 생성 실패: {e}")
            traceback.print_exc()
    else:
        print("  score CSV 또는 manifest 없음 → plan 생성 불가")

    print(f"\n{'='*70}")
    if issues:
        print(f"[DRY-RUN] FAIL: {len(issues)} 문제 발견")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("[DRY-RUN] PASS: 모든 입력 확인 완료. --run-smoke 로 실행 가능.")


# =============================================================================
# 실제 smoke
# =============================================================================

def run_smoke():
    import numpy as np
    import torch
    import torch.nn.functional as F

    print("=" * 70)
    print("[SMOKE] RD-D1s center-region rescore smoke v1")
    print("=" * 70)

    t_start = time.time()

    # BLOCKER 2: 기존 output overwrite 감지
    planned_outputs = [
        SMOKE_MANIFEST_CSV, SMOKE_SCORES_CSV, PATIENT_SUMMARY_CSV,
        ERROR_CSV, REPORT_MD, SUMMARY_JSON, DONE_JSON,
    ]
    existing_outputs = [p for p in planned_outputs if p.exists()]
    if existing_outputs:
        GUARDRAILS["output_overwrite"] = True
        print("[ABORT] output overwrite risk: 기존 결과 파일이 존재합니다.", file=sys.stderr)
        for p in existing_outputs:
            print(f"  {p}", file=sys.stderr)
        sys.exit(2)

    # 입력 파일 필수 확인
    for name, path in [
        ("score CSV", RD_D1S_SCORE_CSV),
        ("manifest", CANDIDATE_MANIFEST_CSV),
        ("checkpoint", CKPT_PATH),
        ("ResNet18 weight", LOCAL_RESNET_WEIGHT),
        ("CT root", CANDIDATE_CT_ROOT),
    ]:
        if not Path(path).exists():
            print(f"[ABORT] 필수 파일 없음: {name} = {path}", file=sys.stderr)
            sys.exit(2)

    # 출력 디렉토리 생성
    for d in [MANIFEST_DIR, REPORT_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    print("\n[1] score CSV / manifest 로드...")
    score_rows = read_csv(RD_D1S_SCORE_CSV)
    stage1_rows = [r for r in score_rows if r.get("stage_split") == "stage1_dev"]
    print(f"  stage1_dev rows: {len(stage1_rows):,}")

    manifest_rows = read_csv(CANDIDATE_MANIFEST_CSV)
    manifest_map = {r["candidate_id"]: r for r in manifest_rows}

    # D2 문제 환자
    d2_problem_pids = set()
    if D2_PROBLEM_PATIENTS_CSV.exists():
        d2_rows = read_csv(D2_PROBLEM_PATIENTS_CSV)
        d2_problem_pids = set(r["patient_id"] for r in d2_rows)
        print(f"  D2 problem patients: {sorted(d2_problem_pids)}")
    else:
        print("  D2 문제 환자 CSV 없음 → fallback sampling 사용")

    # smoke 후보 선정
    print("\n[2] smoke 후보 선정...")
    smoke_rows = select_smoke_candidates(stage1_rows, manifest_map, d2_problem_pids)

    n_smoke = len(smoke_rows)
    print(f"  선정: {len(set(r['patient_id'] for r in smoke_rows))} 환자 / {n_smoke} 후보")

    if n_smoke > MAX_TOTAL:
        msg = f"smoke 후보 수 {n_smoke} > MAX {MAX_TOTAL} → ABORT"
        append_error(msg, "candidate_count_check")
        print(f"[ABORT] {msg}", file=sys.stderr)
        GUARDRAILS["full_scoring_executed"] = True
        sys.exit(2)

    GUARDRAILS["actual_forward_candidates"] = n_smoke

    # manifest 저장
    write_csv(SMOKE_MANIFEST_CSV, smoke_rows)
    print(f"  smoke manifest 저장: {SMOKE_MANIFEST_CSV}")

    # 모델 로드
    print("\n[3] checkpoint 로드 (read-only)...")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"  device: {device}")

    teacher = build_teacher().to(device)
    student = build_student_decoder().to(device)

    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True)
    student.load_state_dict(ckpt["student_state_dict"])

    # teacher parameter 불변 확인
    teacher_param_snap = {n: p.detach().cpu().clone() for n, p in teacher.named_parameters()}

    teacher.eval()
    student.eval()
    # train / optimizer 호출 방지
    student.train = _forbidden_train
    teacher.train = _forbidden_train

    GUARDRAILS["checkpoint_loaded"] = True
    print("  checkpoint 로드 완료")

    # teacher feature hook
    teacher_features = {}
    for layer_name, module in [
        ("layer1", teacher.layer1),
        ("layer2", teacher.layer2),
        ("layer3", teacher.layer3),
    ]:
        def _hook(module, inp, output, _n=layer_name):
            teacher_features[_n] = output
        module.register_forward_hook(_hook)

    # safe_id → ct_path 매핑
    def get_ct_path(safe_id: str, patient_id: str) -> Path:
        # safe_id 형식: NSCLC_LUNG1-001__xxx 또는 MSD_Lung_MSD_lung_001__xxx
        safe_id_clean = safe_id.strip()
        p = CANDIDATE_CT_ROOT / safe_id_clean / "ct_hu.npy"
        if not p.exists():
            # patient_id로 fallback (LUNG1-001 → NSCLC_LUNG1-001__*)
            for d in CANDIDATE_CT_ROOT.iterdir():
                if patient_id.replace("-", "_") in d.name or patient_id in d.name:
                    cand = d / "ct_hu.npy"
                    if cand.exists():
                        return cand
        return p

    ct_cache = CTMmapCache(max_size=5)

    # scalar score 재현 확인용 (최초 10개)
    scalar_repro_check = []

    # scoring
    print("\n[4] center-region scoring 시작...")
    score_results = []
    error_rows = []

    GUARDRAILS["model_forward_executed"] = True

    for idx, row in enumerate(smoke_rows):
        try:
            cid    = row["candidate_id"]
            pid    = row["patient_id"]
            safe   = row["safe_id"]
            local_z = int(row["local_z"])
            y0 = int(row["crop_y0"])
            x0 = int(row["crop_x0"])
            y1 = int(row["crop_y1"])
            x1 = int(row["crop_x1"])
            orig_score = float(row["rd_d1s_medi3ch_rd4ad_score"])

            ct_path = get_ct_path(safe, pid)
            assert_path_safe(ct_path)
            if not ct_path.exists():
                raise FileNotFoundError(f"ct_hu.npy not found: {ct_path}")

            ct_arr = ct_cache.get(ct_path)
            crop   = build_medi3ch_crop(ct_arr, local_z, y0, x0, y1, x1)

            # forward
            with torch.no_grad():
                t_in = torch.from_numpy(crop).unsqueeze(0).to(device)
                teacher(t_in)
                tf3 = teacher_features["layer3"]
                tf2 = teacher_features["layer2"]
                tf1 = teacher_features["layer1"]
                de3, de2, de1 = student(tf3)

                # scalar score 재현
                s3 = float((1 - F.cosine_similarity(de3, tf3, dim=1)).mean())
                s2 = float((1 - F.cosine_similarity(de2, tf2, dim=1)).mean())
                s1 = float((1 - F.cosine_similarity(de1, tf1, dim=1)).mean())
                scalar_repro = (s3 + s2 + s1) / 3.0

                # spatial error maps (1 - cos_sim per spatial position)
                emap3 = (1 - F.cosine_similarity(de3, tf3, dim=1)).squeeze(0).cpu().numpy()  # 6×6
                emap2 = (1 - F.cosine_similarity(de2, tf2, dim=1)).squeeze(0).cpu().numpy()  # 12×12
                emap1 = (1 - F.cosine_similarity(de1, tf1, dim=1)).squeeze(0).cpu().numpy()  # 24×24

            center_scores = compute_center_scores([emap3, emap2, emap1])

            # scalar 재현 오차 기록 (최초 10개)
            if len(scalar_repro_check) < 10:
                scalar_repro_check.append(abs(scalar_repro - orig_score))

            result = {
                "candidate_id":               cid,
                "patient_id":                 pid,
                "label":                      row.get("label", ""),
                "sampling_reason":            row.get("sampling_reason", ""),
                "rd_d1s_medi3ch_rd4ad_score": orig_score,
                "scalar_repro":               round(scalar_repro, 6),
                "scalar_repro_abs_diff":      round(abs(scalar_repro - orig_score), 6),
            }
            result.update({k: round(v, 6) for k, v in center_scores.items()})
            score_results.append(result)

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            error_rows.append({"candidate_id": row.get("candidate_id", ""),
                                "patient_id": row.get("patient_id", ""),
                                "error": err_msg})
            append_error(err_msg, f"forward:{row.get('candidate_id','')}")

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{n_smoke} 완료  errors={len(error_rows)}")

    print(f"  forward 완료: {len(score_results)} 성공 / {len(error_rows)} 실패")

    # teacher parameter 불변 재확인
    for n, p_snap in teacher_param_snap.items():
        p_cur = dict(teacher.named_parameters())[n].detach().cpu()
        if not torch.equal(p_snap, p_cur):
            append_error(f"teacher param changed: {n}", "guardrail_check")

    # WARNING 2: 오류 0건이어도 errors.csv 생성 (헤더-only 형태로)
    if error_rows:
        write_csv(ERROR_CSV, error_rows, ["candidate_id", "patient_id", "error"])
    else:
        write_csv(ERROR_CSV, [{"candidate_id": "", "patient_id": "", "error": "none"}],
                  ["candidate_id", "patient_id", "error"])

    # scores CSV 저장
    if not score_results:
        print("[WARN] score 결과 없음 → FAIL 처리", file=sys.stderr)
        write_summary(GUARDRAILS, {}, "FAIL", "score 결과 없음", t_start)
        sys.exit(2)

    write_csv(SMOKE_SCORES_CSV, score_results)
    print(f"  scores CSV 저장: {SMOKE_SCORES_CSV}")

    # ==========================================================================
    # 분석
    # ==========================================================================
    print("\n[5] 분석...")
    import numpy as np

    y_true   = np.array([1 if r["label"] == "positive" else 0 for r in score_results])
    n_pos_t  = int(y_true.sum())
    n_hn_t   = int((y_true == 0).sum())

    def scores_of(key):
        return np.array([float(r[key]) for r in score_results])

    score_keys = [
        "rd_d1s_medi3ch_rd4ad_score",
        "full_map_mean",
        "full_map_top5_mean",
        "center32_mean",
        "center32_top5_mean",
        "center32_max",
        "center16_mean",
        "center16_top5_mean",
        "center16_max",
    ]

    metrics_per_key = {}
    for key in score_keys:
        sv = scores_of(key)
        auroc = compute_auroc_mann_whitney(y_true.tolist(), sv.tolist())
        auprc = compute_average_precision(y_true.tolist(), sv.tolist())
        metrics_per_key[key] = {"auroc": auroc, "auprc": auprc}

    # scalar 재현 오차
    repro_mean = float(np.mean(scalar_repro_check)) if scalar_repro_check else None
    repro_max  = float(np.max(scalar_repro_check)) if scalar_repro_check else None
    scalar_repro_ok = repro_mean is not None and repro_mean < 1e-4

    # Spearman
    spearman_c32t5 = spearman_corr(
        scores_of("rd_d1s_medi3ch_rd4ad_score").tolist(),
        scores_of("center32_top5_mean").tolist()
    )

    # patient-level 분석
    pat_data = defaultdict(lambda: {"pos_ranks": [], "scores_scalar": [], "scores_c32t5": [],
                                    "labels": [], "n_candidates": 0})
    # build rank within patient for each score
    pat_groups_res = defaultdict(list)
    for r in score_results:
        pat_groups_res[r["patient_id"]].append(r)

    pat_summary_rows = []
    scalar_all_sup_set    = set()
    c32t5_all_sup_set     = set()

    # threshold 없이 rank 기반 분석 (상대적 ranking)
    for pid, rows in pat_groups_res.items():
        n_total = len(rows)
        sorted_scalar = sorted(rows, key=lambda r: float(r["rd_d1s_medi3ch_rd4ad_score"]), reverse=True)
        sorted_c32t5  = sorted(rows, key=lambda r: float(r["center32_top5_mean"]), reverse=True)

        pos_mask = np.array([1 if r["label"] == "positive" else 0 for r in rows])
        n_pos_p  = int(pos_mask.sum())

        # top1 / top3 positive retention (positive가 top-k 안에 몇 개 있나)
        def topk_retention(sorted_rows, k):
            top_k_labels = [r["label"] for r in sorted_rows[:k]]
            return 1 if "positive" in top_k_labels else 0

        scalar_top1 = topk_retention(sorted_scalar, 1)
        scalar_top3 = topk_retention(sorted_scalar, 3)
        c32t5_top1  = topk_retention(sorted_c32t5, 1)
        c32t5_top3  = topk_retention(sorted_c32t5, 3)

        # best positive rank (1-indexed)
        def best_pos_rank(sorted_rows):
            for i, r in enumerate(sorted_rows):
                if r["label"] == "positive":
                    return i + 1
            return n_total + 1

        scalar_best_rank = best_pos_rank(sorted_scalar)
        c32t5_best_rank  = best_pos_rank(sorted_c32t5)

        # pat_all_sup: positive가 0개인 환자는 제외, n_pos > 0인데 top-k에 없으면 위험
        if n_pos_p > 0:
            # all_sup 개념: 모든 positive가 hard_negative보다 낮게 ranking됨
            pos_scores_sc   = sorted([float(r["rd_d1s_medi3ch_rd4ad_score"]) for r in rows if r["label"] == "positive"], reverse=True)
            hn_scores_sc    = sorted([float(r["rd_d1s_medi3ch_rd4ad_score"]) for r in rows if r["label"] == "hard_negative"], reverse=True)
            pos_scores_c32  = sorted([float(r["center32_top5_mean"]) for r in rows if r["label"] == "positive"], reverse=True)
            hn_scores_c32   = sorted([float(r["center32_top5_mean"]) for r in rows if r["label"] == "hard_negative"], reverse=True)

            if hn_scores_sc and pos_scores_sc and max(pos_scores_sc) < min(hn_scores_sc[:1]):
                scalar_all_sup_set.add(pid)
            if hn_scores_c32 and pos_scores_c32 and max(pos_scores_c32) < min(hn_scores_c32[:1]):
                c32t5_all_sup_set.add(pid)

        rank_improved = c32t5_best_rank < scalar_best_rank
        rank_worsened = c32t5_best_rank > scalar_best_rank

        pat_summary_rows.append({
            "patient_id":           pid,
            "n_candidates":         n_total,
            "n_positive":           n_pos_p,
            "scalar_best_rank":     scalar_best_rank,
            "c32t5_best_rank":      c32t5_best_rank,
            "rank_improved":        int(rank_improved),
            "rank_worsened":        int(rank_worsened),
            "scalar_top1_retention": scalar_top1,
            "scalar_top3_retention": scalar_top3,
            "c32t5_top1_retention": c32t5_top1,
            "c32t5_top3_retention": c32t5_top3,
        })

    write_csv(PATIENT_SUMMARY_CSV, pat_summary_rows)

    # 집계
    pos_patients = [r for r in pat_summary_rows if r["n_positive"] > 0]
    n_pos_pats   = len(pos_patients)
    scalar_top1_ret  = sum(r["scalar_top1_retention"] for r in pos_patients) / max(n_pos_pats, 1)
    scalar_top3_ret  = sum(r["scalar_top3_retention"] for r in pos_patients) / max(n_pos_pats, 1)
    c32t5_top1_ret   = sum(r["c32t5_top1_retention"] for r in pos_patients) / max(n_pos_pats, 1)
    c32t5_top3_ret   = sum(r["c32t5_top3_retention"] for r in pos_patients) / max(n_pos_pats, 1)
    rank_improved_n  = sum(r["rank_improved"] for r in pos_patients)
    rank_worsened_n  = sum(r["rank_worsened"] for r in pos_patients)

    # D2 문제 환자에서 개선 비율
    d2_pats_in_smoke = [r for r in pos_patients if r["patient_id"] in d2_problem_pids]
    d2_improved = sum(r["rank_improved"] for r in d2_pats_in_smoke)
    d2_total    = len(d2_pats_in_smoke)
    d2_improve_rate = d2_improved / d2_total if d2_total > 0 else None

    scalar_auroc = metrics_per_key["rd_d1s_medi3ch_rd4ad_score"]["auroc"]
    c32t5_auroc  = metrics_per_key["center32_top5_mean"]["auroc"]
    scalar_auprc = metrics_per_key["rd_d1s_medi3ch_rd4ad_score"]["auprc"]
    c32t5_auprc  = metrics_per_key["center32_top5_mean"]["auprc"]

    # 성공 판정
    passed_conditions = []
    failed_conditions = []

    def chk(name, passed_val):
        if passed_val:
            passed_conditions.append(name)
        else:
            failed_conditions.append(name)

    scalar_all_sup = len(scalar_all_sup_set)
    c32t5_all_sup  = len(c32t5_all_sup_set)

    # BLOCKER 1: scalar 재현 실패 시 PASS 불가
    chk("scalar_repro_ok",                   scalar_repro_ok)
    chk("pat_all_sup_count_decreased",       c32t5_all_sup < scalar_all_sup)
    chk("top1_retention_improved",           c32t5_top1_ret >= scalar_top1_ret)
    chk("top3_retention_improved",           c32t5_top3_ret >= scalar_top3_ret)
    chk("d2_problem_patients_50pct_improved", d2_improve_rate is None or d2_improve_rate >= 0.5)
    chk("guardrail_stage2_not_accessed",     not GUARDRAILS["stage2_holdout_accessed"])
    chk("guardrail_no_training",             not GUARDRAILS["training_executed"])
    chk("guardrail_no_full_scoring",         not GUARDRAILS["full_scoring_executed"])
    chk("guardrail_no_existing_modified",    not GUARDRAILS["existing_artifact_modified"])

    # scalar_repro_ok 실패 시 최대 PARTIAL_PASS_EXPLORATORY로 제한
    all_passed = len(failed_conditions) == 0
    partial = (not all_passed and scalar_repro_ok and
               (c32t5_top1_ret >= scalar_top1_ret or
                d2_improve_rate is not None and d2_improve_rate >= 0.3))

    if all_passed:
        verdict = "PASS_CANDIDATE"
    elif partial:
        verdict = "PARTIAL_PASS_EXPLORATORY"
    else:
        verdict = "FAIL"

    print(f"\n판정: {verdict}")
    print(f"  scalar AUROC={scalar_auroc:.4f}  center32_top5_mean AUROC={c32t5_auroc:.4f}")
    print(f"  scalar pat_all_sup={scalar_all_sup}  c32t5 pat_all_sup={c32t5_all_sup}")
    print(f"  scalar top1_ret={scalar_top1_ret:.3f}  c32t5 top1_ret={c32t5_top1_ret:.3f}")
    print(f"  rank_improved={rank_improved_n}  rank_worsened={rank_worsened_n}")

    t_end = time.time()
    analytics = {
        "selected_patients":           len(pat_groups_res),
        "selected_candidates":         n_smoke,
        "positive_candidates":         n_pos_t,
        "hard_negative_candidates":    n_hn_t,
        "scalar_repro_mean_abs_diff":  repro_mean,
        "scalar_repro_max_abs_diff":   repro_max,
        "scalar_repro_ok":             scalar_repro_ok,
        "scalar_auroc":                scalar_auroc,
        "scalar_auprc":                scalar_auprc,
        "center32_top5_auroc":         c32t5_auroc,
        "center32_top5_auprc":         c32t5_auprc,
        "spearman_scalar_vs_c32t5":    spearman_c32t5,
        "metrics_all_keys": {k: {"auroc": round(v["auroc"], 4) if v["auroc"] is not None else None,
                                 "auprc": round(v["auprc"], 4) if v["auprc"] is not None else None}
                             for k, v in metrics_per_key.items()},
        "scalar_pat_all_sup_count":    scalar_all_sup,
        "c32t5_pat_all_sup_count":     c32t5_all_sup,
        "scalar_top1_retention":       round(scalar_top1_ret, 4),
        "scalar_top3_retention":       round(scalar_top3_ret, 4),
        "c32t5_top1_retention":        round(c32t5_top1_ret, 4),
        "c32t5_top3_retention":        round(c32t5_top3_ret, 4),
        "rank_improved_patients":      rank_improved_n,
        "rank_worsened_patients":      rank_worsened_n,
        "d2_problem_patients_in_smoke": d2_total,
        "d2_problem_improve_rate":     round(d2_improve_rate, 4) if d2_improve_rate is not None else None,
        "passed_conditions":           passed_conditions,
        "failed_conditions":           failed_conditions,
        "verdict":                     verdict,
        "runtime_seconds":             round(t_end - t_start, 1),
    }

    write_summary(GUARDRAILS, analytics, verdict, "", t_start)
    write_report(analytics, GUARDRAILS, score_results, pat_summary_rows)

    DONE_JSON.write_text(json.dumps({
        "verdict": verdict, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }, indent=2))

    print(f"\n생성 파일:")
    for p in [SMOKE_MANIFEST_CSV, SMOKE_SCORES_CSV, PATIENT_SUMMARY_CSV,
              ERROR_CSV, REPORT_MD, SUMMARY_JSON, DONE_JSON]:
        if p.exists():
            print(f"  {p}")


def write_summary(guardrails, analytics, verdict, note, t_start):
    summary = {**guardrails, **analytics}
    summary["verdict"] = verdict
    if note:
        summary["note"] = note
    check_no_existing_modified([SUMMARY_JSON])
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def write_report(analytics, guardrails, score_results, pat_summary_rows):
    check_no_existing_modified([REPORT_MD])
    import numpy as np

    lines = [
        "# RD-D1s center-region rescore smoke v1 report",
        "",
        f"**판정: {analytics['verdict']}**",
        "",
        "## 핵심 수치",
        "",
        f"- selected patients: {analytics['selected_patients']}",
        f"- selected candidates: {analytics['selected_candidates']}",
        f"- positive: {analytics['positive_candidates']}",
        f"- hard_negative: {analytics['hard_negative_candidates']}",
        f"- scalar repro mean abs diff: {analytics['scalar_repro_mean_abs_diff']}",
        f"- scalar repro ok (< 1e-4): {analytics['scalar_repro_ok']}",
        "",
        "## AUROC / AUPRC",
        "",
    ]
    for key, m in analytics.get("metrics_all_keys", {}).items():
        lines.append(f"- {key}: AUROC={m['auroc']}  AUPRC={m['auprc']}")
    lines += [
        "",
        "## Safety 지표",
        "",
        "**Note:** pat_all_sup_count in this smoke is rank-based exploratory proxy "
        "(pos_max < top-1 hn_score), not identical to RD-D2 guard-simulation pat_all_sup.",
        "",
        f"- scalar pat_all_sup_count: {analytics['scalar_pat_all_sup_count']}",
        f"- center32_top5 pat_all_sup_count: {analytics['c32t5_pat_all_sup_count']}",
        f"- scalar top1_retention: {analytics['scalar_top1_retention']}",
        f"- scalar top3_retention: {analytics['scalar_top3_retention']}",
        f"- c32t5 top1_retention: {analytics['c32t5_top1_retention']}",
        f"- c32t5 top3_retention: {analytics['c32t5_top3_retention']}",
        f"- rank_improved_patients: {analytics['rank_improved_patients']}",
        f"- rank_worsened_patients: {analytics['rank_worsened_patients']}",
        f"- D2 problem patients in smoke: {analytics['d2_problem_patients_in_smoke']}",
        f"- D2 problem improve rate: {analytics['d2_problem_improve_rate']}",
        f"- Spearman scalar vs c32t5: {analytics.get('spearman_scalar_vs_c32t5')}",
        "",
        "## 통과 / 실패 조건",
        "",
    ]
    for c in analytics.get("passed_conditions", []):
        lines.append(f"- [PASS] {c}")
    for c in analytics.get("failed_conditions", []):
        lines.append(f"- [FAIL] {c}")
    lines += [
        "",
        "## Guardrail 기록",
        "",
    ]
    for k, v in guardrails.items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## 다음 단계",
        "",
    ]
    verdict = analytics["verdict"]
    if verdict == "PASS_CANDIDATE":
        lines.append("PASS_CANDIDATE → center-region score full preflight 진행")
    elif verdict == "PARTIAL_PASS_EXPLORATORY":
        lines.append("PARTIAL_PASS_EXPLORATORY → D2 문제 환자 중심 smoke scope 확장 검토")
    else:
        lines.append("FAIL → RD-D branch를 analysis-only로 닫음")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RD-D1s center-region rescore smoke")
    parser.add_argument("--dry-run",                 action="store_true")
    parser.add_argument("--run-smoke",               action="store_true")
    parser.add_argument("--confirm-model-forward",   action="store_true")
    parser.add_argument("--confirm-stage1dev-only",  action="store_true")
    args = parser.parse_args()

    # bare run 차단
    if not args.dry_run and not args.run_smoke:
        print("[EXIT 2] bare run은 허용되지 않음.", file=sys.stderr)
        print("  사용: --dry-run  또는  --run-smoke --confirm-model-forward --confirm-stage1dev-only",
              file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        run_dry()
        return

    if args.run_smoke:
        if not args.confirm_model_forward:
            print("[EXIT 2] --confirm-model-forward 없이 실행 불가", file=sys.stderr)
            sys.exit(2)
        if not args.confirm_stage1dev_only:
            print("[EXIT 2] --confirm-stage1dev-only 없이 실행 불가", file=sys.stderr)
            sys.exit(2)
        run_smoke()


if __name__ == "__main__":
    main()
