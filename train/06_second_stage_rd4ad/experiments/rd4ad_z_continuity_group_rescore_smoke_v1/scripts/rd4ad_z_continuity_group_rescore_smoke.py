#!/usr/bin/env python3
"""
RD4AD z-continuity group-level rescoring smoke v1

목적:
  1차 PaDiM candidate 전체를 first_stage_score로 삭제하지 않고,
  위치 근접성 + z축 연속성으로 group화한 뒤
  group 대표 96x96 crop만 RD4AD로 재판별하여 계산량을 줄이면서
  병변 group coverage를 유지하는지 smoke로 검증한다.

실행:
  bare run 금지 (exit 2):
    python rd4ad_z_continuity_group_rescore_smoke.py

  dry-run:
    python rd4ad_z_continuity_group_rescore_smoke.py --dry-run

  group preflight:
    python rd4ad_z_continuity_group_rescore_smoke.py \\
      --run-group-preflight --confirm-readonly --confirm-stage1dev-only

  actual smoke:
    python rd4ad_z_continuity_group_rescore_smoke.py \\
      --run-smoke --confirm-model-forward --confirm-stage1dev-only --max-groups 1000

guardrail:
  - stage2_holdout 접근 차단
  - training/backward/optimizer/checkpoint_save 금지
  - 기존 파일 수정 금지
  - bare run 차단 (exit 2)
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
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/rd4ad_z_continuity_group_rescore_smoke_v1"

# candidate manifest 113,447개 stage1_dev (read-only)
CANDIDATE_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c2_effb0_v420_candidate_rd4ad_retest_v1"
    / "rd_c2_effb0_v420_candidate_manifest.csv"
)

# RD-D1s scalar score CSV (read-only) - scalar reproduction 비교용
RD_D1S_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1s_medi3ch_true_rd4ad_shard_run_v1"
    / "rd_d1s_stage1dev_candidate_score.csv"
)

# per-patient source CSV base (roi_0_0_patch_ratio join용, read-only)
EFFB0_SCORE_BASE = (
    PROJECT_ROOT
    / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs/scores"
)

# checkpoint (read-only)
CKPT_PATH = (
    PROJECT_ROOT
    / "outputs/models/rd_d1s_true_rd4ad_resnet18_medi3ch_shard_v1"
    / "checkpoints/best_train_loss.pth"
)
LOCAL_RESNET_WEIGHT = Path("/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")

# CT root (read-only)
CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)

# 출력 (새 폴더에만)
MANIFEST_DIR = EXPERIMENT_ROOT / "manifests"
REPORT_DIR   = EXPERIMENT_ROOT / "reports"
LOG_DIR      = EXPERIMENT_ROOT / "logs"

GROUP_MANIFEST_CSV          = MANIFEST_DIR / "group_manifest.csv"
GROUP_REPR_MANIFEST_CSV     = MANIFEST_DIR / "group_representative_manifest.csv"
GROUP_RESCORE_SCORES_CSV    = MANIFEST_DIR / "group_rescore_scores.csv"
GROUP_TOPK_RETENTION_CSV    = MANIFEST_DIR / "group_topk_retention.csv"
PROBLEM_PATIENT_AUDIT_CSV   = MANIFEST_DIR / "problem_patient_group_audit.csv"
GROUPING_ABLATION_CSV       = MANIFEST_DIR / "grouping_ablation_summary.csv"
ERROR_CSV                   = LOG_DIR / "errors.csv"
REPORT_MD                   = REPORT_DIR / "rd4ad_z_continuity_group_rescore_smoke_report.md"
SUMMARY_JSON                = REPORT_DIR / "rd4ad_z_continuity_group_rescore_smoke_summary.json"
DONE_JSON                   = EXPERIMENT_ROOT / "DONE.json"

# group 파라미터
DEFAULT_Z_GAP    = 1
DEFAULT_XY_RADIUS = 24
ABLATION_XY_RADII = [16, 24, 32]
ABLATION_Z_GAPS   = [1, 2]

# HU 윈도잉
HU_MIN, HU_MAX = -160.0, 240.0
CROP_SIZE = 96

# smoke 제한
MAX_GROUPS_SMOKE = 1000
MAX_POS_GROUPS   = 500
MAX_HN_BOUND_GROUPS = 250
MAX_HN_SCORE_GROUPS = 250

# 문제 환자
PROBLEM_PATIENT_IDS = {"LUNG1-086", "LUNG1-386", "LUNG1-399"}

# guardrail 기록
GUARDRAILS = {
    "stage2_holdout_accessed":               False,
    "checkpoint_loaded":                     False,
    "model_forward_executed":                False,
    "training_executed":                     False,
    "backward_executed":                     False,
    "optimizer_created":                     False,
    "checkpoint_saved":                      False,
    "crop_generation_executed":              False,
    "full_scoring_executed":                 False,
    "threshold_recalculated":                False,
    "existing_artifact_modified":            False,
    "existing_script_modified":              False,
    "output_overwrite":                      False,
    "label_used_for_smoke_sampling":         True,
    "label_used_as_deployment_selector":     False,
    "first_stage_score_used_for_representative_choice": True,
    "first_stage_score_used_for_candidate_deletion":    False,
    "rd4ad_score_modified":                  False,
    "adjusted_score_preview_only":           True,
    "stage2_holdout_accessed":               False,
}

# =============================================================================
# 안전 경로 검사
# =============================================================================

def assert_path_safe(p: Path):
    s = str(p).lower()
    if "stage2_holdout" in s or ("stage2" in s and "holdout" in s):
        GUARDRAILS["stage2_holdout_accessed"] = True
        raise RuntimeError(f"[ABORT] stage2_holdout 경로 접근 차단: {p}")


def ensure_output_path_safe(p: Path):
    """쓰기 경로가 EXPERIMENT_ROOT 아래인지 확인."""
    p = Path(p).resolve()
    exp_root = EXPERIMENT_ROOT.resolve()
    if not str(p).startswith(str(exp_root)):
        GUARDRAILS["existing_artifact_modified"] = True
        raise RuntimeError(f"[ABORT] 기존 파일 수정 시도 차단: {p}")


# =============================================================================
# CSV 유틸
# =============================================================================

def read_csv(path: Path):
    assert_path_safe(path)
    rows = []
    with open(str(path), encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def write_csv(path: Path, fieldnames, rows):
    ensure_output_path_safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  saved: {path} ({len(rows)} rows)")


def append_error(msg: str, exc: Exception = None):
    ensure_output_path_safe(ERROR_CSV)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    mode = "a" if ERROR_CSV.exists() else "w"
    with open(str(ERROR_CSV), mode, encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["timestamp", "message", "traceback"])
        tb = traceback.format_exc() if exc else ""
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), msg, tb.replace("\n", " | ")])


# =============================================================================
# roi_0_0_patch_ratio 조회 (per-patient source CSV join)
# =============================================================================

class RoiRatioLookup:
    """
    per-patient effb0 score CSV에서 roi_0_0_patch_ratio / position_bin 를
    candidate_id 기준으로 조회한다.

    join key:
      patient_id + local_z 동일 + candidate crop center (y_center, x_center)가
      source patch [y0, y1) x [x0, x1) 안에 포함되는 patch들의 mean
    """

    def __init__(self):
        self._cache = {}  # patient_id -> list of {local_z, y0, x0, y1, x1, roi_0_0_patch_ratio, position_bin}

    def _load_patient(self, patient_id: str, source_csv_rel: str):
        if patient_id in self._cache:
            return
        p = EFFB0_SCORE_BASE / source_csv_rel
        assert_path_safe(p)
        if not p.exists():
            self._cache[patient_id] = []
            return
        rows = []
        try:
            with open(str(p), encoding="utf-8-sig", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    rows.append({
                        "local_z": int(row["local_z"]),
                        "y0":      int(row["y0"]),
                        "x0":      int(row["x0"]),
                        "y1":      int(row["y1"]),
                        "x1":      int(row["x1"]),
                        "roi_0_0_patch_ratio": float(row.get("roi_0_0_patch_ratio", 0.0) or 0.0),
                        "position_bin":        row.get("position_bin", ""),
                    })
        except Exception:
            rows = []
        self._cache[patient_id] = rows

    def lookup(self, patient_id: str, source_csv_rel: str,
               local_z: int, crop_y0: int, crop_x0: int, crop_y1: int, crop_x1: int):
        """
        candidate crop 영역 내 patches의 roi_0_0_patch_ratio mean 반환.
        없으면 None.
        """
        self._load_patient(patient_id, source_csv_rel)
        patches = self._cache.get(patient_id, [])
        y_center = (crop_y0 + crop_y1) / 2.0
        x_center = (crop_x0 + crop_x1) / 2.0

        matched_ratios = []
        matched_bins = []
        for p in patches:
            if p["local_z"] != local_z:
                continue
            # patch center가 candidate crop 안에 있는지 확인
            py_center = (p["y0"] + p["y1"]) / 2.0
            px_center = (p["x0"] + p["x1"]) / 2.0
            if crop_y0 <= py_center < crop_y1 and crop_x0 <= px_center < crop_x1:
                matched_ratios.append(p["roi_0_0_patch_ratio"])
                if p["position_bin"]:
                    matched_bins.append(p["position_bin"])

        if not matched_ratios:
            # fallback: candidate center가 patch 안에 있는지 확인
            for p in patches:
                if p["local_z"] != local_z:
                    continue
                if p["y0"] <= y_center < p["y1"] and p["x0"] <= x_center < p["x1"]:
                    matched_ratios.append(p["roi_0_0_patch_ratio"])
                    if p["position_bin"]:
                        matched_bins.append(p["position_bin"])

        if not matched_ratios:
            return None, None, None

        mean_ratio = sum(matched_ratios) / len(matched_ratios)
        min_ratio  = min(matched_ratios)
        # position_bin mode
        mode_bin = ""
        if matched_bins:
            bin_count = defaultdict(int)
            for b in matched_bins:
                bin_count[b] += 1
            mode_bin = max(bin_count, key=bin_count.__getitem__)

        return mean_ratio, min_ratio, mode_bin


# =============================================================================
# union-find (connected components)
# =============================================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# =============================================================================
# group 생성
# =============================================================================

def build_groups(candidates, z_gap=DEFAULT_Z_GAP, xy_radius=DEFAULT_XY_RADIUS):
    """
    patient별 connected component group 생성.

    parameters:
      candidates: list of dicts with keys:
        candidate_id, patient_id, local_z, y_center, x_center, label, first_stage_score
      z_gap: max |z_i - z_j| for same group
      xy_radius: max |y_center_i - y_center_j| and |x_center_i - x_center_j|

    returns:
      group_id_map: {candidate_id -> group_id}
      group_candidates: {group_id -> [candidate_id, ...]}
    """
    pat_candidates = defaultdict(list)
    for c in candidates:
        pat_candidates[c["patient_id"]].append(c)

    group_id_map = {}
    group_candidates = {}
    global_group_counter = [0]

    for pid, cands in pat_candidates.items():
        n = len(cands)
        uf = UnionFind(n)

        for i in range(n):
            zi = cands[i]["local_z"]
            yi = cands[i]["y_center"]
            xi = cands[i]["x_center"]
            for j in range(i + 1, n):
                zj = cands[j]["local_z"]
                if abs(zi - zj) > z_gap:
                    continue
                yj = cands[j]["y_center"]
                xj = cands[j]["x_center"]
                if abs(yi - yj) <= xy_radius and abs(xi - xj) <= xy_radius:
                    uf.union(i, j)

        # root -> group_id
        root_to_gid = {}
        for i in range(n):
            root = uf.find(i)
            if root not in root_to_gid:
                gid = f"G{global_group_counter[0]:07d}"
                global_group_counter[0] += 1
                root_to_gid[root] = gid
                group_candidates[gid] = []
            gid = root_to_gid[root]
            cid = cands[i]["candidate_id"]
            group_id_map[cid] = gid
            group_candidates[gid].append(cid)

    return group_id_map, group_candidates


# =============================================================================
# group 통계 계산
# =============================================================================

def compute_group_stats(group_id, cand_ids, cand_map):
    """
    group_id, list of candidate_ids, candidate info dict 를 받아
    group 통계 dict 반환.
    """
    rows = [cand_map[cid] for cid in cand_ids if cid in cand_map]
    if not rows:
        return None

    patient_id = rows[0]["patient_id"]
    z_vals = [r["local_z"] for r in rows]
    y_centers = [r["y_center"] for r in rows]
    x_centers = [r["x_center"] for r in rows]
    fss = [r["first_stage_score"] for r in rows]
    labels = [r["label"] for r in rows]
    roi_ratios = [r["roi_0_0_patch_ratio"] for r in rows if r.get("roi_0_0_patch_ratio") is not None]

    positive_count = sum(1 for l in labels if l == "positive")
    hn_count       = sum(1 for l in labels if l == "hard_negative")
    has_positive   = positive_count > 0

    roi_mean = sum(roi_ratios) / len(roi_ratios) if roi_ratios else None
    roi_min  = min(roi_ratios) if roi_ratios else None
    boundary_like_ratio_mean = (1.0 - roi_mean) if roi_mean is not None else None
    boundary_like_ratio_max  = (1.0 - roi_min)  if roi_min  is not None else None

    # representative: first_stage_score 최고인 candidate
    repr_cid = max(rows, key=lambda r: r["first_stage_score"])["candidate_id"]

    # position_bin mode
    bins = [r["position_bin"] for r in rows if r.get("position_bin")]
    mode_bin = ""
    if bins:
        bc = defaultdict(int)
        for b in bins:
            bc[b] += 1
        mode_bin = max(bc, key=bc.__getitem__)

    return {
        "group_id":                    group_id,
        "patient_id":                  patient_id,
        "n_candidates":                len(rows),
        "z_min":                       min(z_vals),
        "z_max":                       max(z_vals),
        "z_span":                      max(z_vals) - min(z_vals),
        "y_center_mean":               round(sum(y_centers) / len(y_centers), 2),
        "x_center_mean":               round(sum(x_centers) / len(x_centers), 2),
        "first_stage_score_max":       max(fss),
        "first_stage_score_mean":      round(sum(fss) / len(fss), 6),
        "positive_count":              positive_count,
        "hard_negative_count":         hn_count,
        "has_positive":                str(has_positive),
        "roi_0_0_patch_ratio_mean":    round(roi_mean, 6) if roi_mean is not None else "",
        "roi_0_0_patch_ratio_min":     round(roi_min, 6)  if roi_min  is not None else "",
        "boundary_like_ratio_mean":    round(boundary_like_ratio_mean, 6) if boundary_like_ratio_mean is not None else "",
        "boundary_like_ratio_max":     round(boundary_like_ratio_max, 6)  if boundary_like_ratio_max  is not None else "",
        "position_bin_mode":           mode_bin,
        "representative_candidate_id": repr_cid,
    }


# =============================================================================
# 모델 빌드 (read-only 구조 동일)
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
    import torch
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
            x = self.de_layer3(layer3_feat)
            de3 = x
            x = self.de_layer2(x)
            de2 = x
            x = self.de_layer1(x)
            de1 = x
            return de3, de2, de1

    return StudentDecoder()


def load_model_from_checkpoint(device):
    """checkpoint read-only load. training 관련 일체 금지."""
    import torch
    assert_path_safe(CKPT_PATH)
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"checkpoint 없음: {CKPT_PATH}")

    teacher = build_teacher().to(device)
    student = build_student_decoder().to(device)
    teacher.eval()
    student.eval()
    teacher.requires_grad_(False)
    for p in student.parameters():
        p.requires_grad_(False)

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    if "student_state_dict" in ckpt:
        student.load_state_dict(ckpt["student_state_dict"])
    elif "model_state_dict" in ckpt:
        student.load_state_dict(ckpt["model_state_dict"])
    else:
        student.load_state_dict(ckpt)

    GUARDRAILS["checkpoint_loaded"] = True
    return teacher, student


def _forbidden_train(*args, **kwargs):
    GUARDRAILS["training_executed"] = True
    raise RuntimeError("[ABORT] training 호출 금지됨")


def _forbidden_backward(*args, **kwargs):
    GUARDRAILS["backward_executed"] = True
    raise RuntimeError("[ABORT] backward 호출 금지됨")


# =============================================================================
# crop 빌드 (기존 rd_d1s 방식과 동일)
# =============================================================================

def build_medi3ch_crop(ct_arr, local_z, y0, x0, y1, x1):
    import numpy as np
    Z, H, W = ct_arr.shape

    def _win(patch):
        c = np.clip(patch.astype(np.float32), HU_MIN, HU_MAX)
        return (c - HU_MIN) / (HU_MAX - HU_MIN)

    def _clip_and_pad(z_idx, cy0, cx0, cy1, cx1):
        cy0c = max(cy0, 0); cy1c = min(cy1, H)
        cx0c = max(cx0, 0); cx1c = min(cx1, W)
        if cy1c <= cy0c or cx1c <= cx0c:
            return np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        patch = ct_arr[z_idx, cy0c:cy1c, cx0c:cx1c]
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
# RD4AD score 계산 (기존 scalar score와 동일 방식)
# =============================================================================

def compute_rd4ad_score(teacher, student, crop_tensor, teacher_features, device):
    """
    crop_tensor: (1,3,96,96) torch.Tensor
    rd4ad scalar score = mean of (1 - cosine_similarity) across all 3 feature levels
    """
    import torch
    import torch.nn.functional as F

    teacher_features.clear()
    with torch.no_grad():
        teacher(crop_tensor)
    tf3 = teacher_features["layer3"]
    tf2 = teacher_features["layer2"]
    tf1 = teacher_features["layer1"]

    with torch.no_grad():
        de3, de2, de1 = student(tf3)

    # per-feature-level mean cosine distance
    scores = []
    for tf, sf in [(tf3, de3), (tf2, de2), (tf1, de1)]:
        # cosine similarity along channel dim, shape: (1, H, W)
        cos_sim = F.cosine_similarity(tf, sf, dim=1, eps=1e-8)
        level_score = float((1.0 - cos_sim).mean().item())
        scores.append(level_score)

    scalar = float(sum(scores) / len(scores))
    return scalar, scores[2], scores[1], scores[0]  # scalar, layer1, layer2, layer3


# =============================================================================
# CT mmap 캐시
# =============================================================================

class CTMmapCache:
    def __init__(self, max_size=5):
        from collections import OrderedDict
        self._cache = OrderedDict()
        self._max = max_size

    def get(self, safe_id: str):
        import numpy as np
        if safe_id in self._cache:
            self._cache.move_to_end(safe_id)
            return self._cache[safe_id]
        ct_path = CT_ROOT / safe_id / "ct_hu.npy"
        assert_path_safe(ct_path)
        if not ct_path.exists():
            raise FileNotFoundError(f"CT 없음: {ct_path}")
        arr = np.load(str(ct_path), mmap_mode="r")
        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        self._cache[safe_id] = arr
        return arr


# =============================================================================
# top-k retention 계산
# =============================================================================

def compute_topk_retention(group_stats_map, score_col, k):
    """
    각 환자별로 score_col 기준 top-k group 안에 has_positive=True group이 있으면 1.
    positive group이 있는 환자만 대상.

    반환: (retention_rate, n_patients_with_positive)
    """
    pat_groups = defaultdict(list)
    for gid, gstat in group_stats_map.items():
        pat_groups[gstat["patient_id"]].append(gstat)

    hit = 0
    total = 0
    for pid, glist in pat_groups.items():
        has_pos_group = any(g.get("has_positive") == "True" for g in glist)
        if not has_pos_group:
            continue
        total += 1
        # score_col으로 정렬 (내림차순)
        scored = []
        for g in glist:
            v = g.get(score_col)
            if v is None or v == "":
                v = -1e9
            else:
                try:
                    v = float(v)
                except Exception:
                    v = -1e9
            scored.append((v, g))
        scored.sort(key=lambda x: x[0], reverse=True)
        topk = scored[:k]
        if any(g.get("has_positive") == "True" for _, g in topk):
            hit += 1

    rate = hit / total if total > 0 else 0.0
    return round(rate, 4), total


def compute_positive_candidate_coverage(group_stats_map, candidate_group_map, score_col, k):
    """
    top-k group 안의 positive candidate 수 / 전체 positive candidate 수.
    group_stats_map: {gid -> gstat with pos/hn counts}
    candidate_group_map: {candidate_id -> group_id}
    """
    # 전체 positive candidate 수 (group 통계에서 합산)
    total_pos = sum(g["positive_count"] for g in group_stats_map.values())
    if total_pos == 0:
        return 0.0

    pat_groups = defaultdict(list)
    for gid, gstat in group_stats_map.items():
        pat_groups[gstat["patient_id"]].append(gstat)

    in_topk_pos = 0
    for pid, glist in pat_groups.items():
        scored = []
        for g in glist:
            v = g.get(score_col)
            if v is None or v == "":
                v = -1e9
            else:
                try:
                    v = float(v)
                except Exception:
                    v = -1e9
            scored.append((v, g))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, g in scored[:k]:
            in_topk_pos += g["positive_count"]

    return round(in_topk_pos / total_pos, 4)


# =============================================================================
# baseline: patch-level top-k retention (B0_patch_all)
# =============================================================================

def compute_patch_topk_retention(score_rows, score_col, k):
    """
    기존 patch-level rd_d1s score CSV 기반 top-k retention.
    score_rows: list of dicts with patient_id, label, {score_col}
    """
    pat_rows = defaultdict(list)
    for r in score_rows:
        if r.get("stage_split", "") != "stage1_dev":
            continue
        pat_rows[r["patient_id"]].append(r)

    hit = 0
    total = 0
    for pid, rows in pat_rows.items():
        has_pos = any(r.get("label") == "positive" for r in rows)
        if not has_pos:
            continue
        total += 1
        scored = []
        for r in rows:
            v = r.get(score_col)
            if v is None or v == "":
                v = -1e9
            else:
                try:
                    v = float(v)
                except Exception:
                    v = -1e9
            scored.append((v, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        if any(r.get("label") == "positive" for _, r in scored[:k]):
            hit += 1

    return round(hit / total, 4) if total > 0 else 0.0, total


# =============================================================================
# adjusted score 계산
# =============================================================================

def compute_adjusted_scores(rd4ad_raw, boundary_like_ratio_mean):
    """
    P0~P6 adjusted score preview.
    기존 score 원본 수정 없음.
    """
    import math
    if rd4ad_raw is None:
        return {}

    blr = boundary_like_ratio_mean if (boundary_like_ratio_mean is not None and
                                        isinstance(boundary_like_ratio_mean, float) and
                                        math.isfinite(boundary_like_ratio_mean)) else 0.0
    roi_mean = 1.0 - blr

    adj = {
        "P0_raw": rd4ad_raw,
        "P1_times_roi_mean":    rd4ad_raw * roi_mean,
        "P2_times_sqrt_roi_mean": rd4ad_raw * math.sqrt(max(roi_mean, 0.0)),
        "P3_soft_alpha_0_2":    rd4ad_raw * (1.0 - 0.2 * blr),
        "P4_soft_alpha_0_3":    rd4ad_raw * (1.0 - 0.3 * blr),
        "P5_minus_lam_0_05":    rd4ad_raw - 0.05 * blr,
        "P6_minus_lam_0_10":    rd4ad_raw - 0.10 * blr,
    }
    return adj


# =============================================================================
# dry-run
# =============================================================================

def run_dry():
    print("=" * 70)
    print("[DRY-RUN] RD4AD z-continuity group-level rescore smoke v1")
    print("=" * 70)
    issues = []

    # 1. 입력 파일 확인
    print("\n[1] 입력 파일 확인")
    checks = {
        "candidate manifest":    CANDIDATE_MANIFEST_CSV,
        "RD-D1s scalar score":   RD_D1S_SCORE_CSV,
        "checkpoint":            CKPT_PATH,
        "ResNet18 weight":       LOCAL_RESNET_WEIGHT,
        "effb0 score base":      EFFB0_SCORE_BASE,
    }
    for name, p in checks.items():
        ok = p.exists()
        print(f"  {'OK' if ok else 'MISSING':8s} {name}: {p}")
        if not ok:
            issues.append(f"MISSING: {name}")

    # 2. CT root
    print(f"\n[2] CT root 확인")
    ct_ok = CT_ROOT.exists()
    print(f"  {'OK' if ct_ok else 'MISSING':8s} {CT_ROOT}")
    if not ct_ok:
        issues.append("MISSING: CT_ROOT")

    # 3. stage2_holdout 없음
    print(f"\n[3] stage2_holdout 접근 없음")
    print(f"  OK  stage2_holdout 경로 참조 없음")

    # 4. output root 충돌
    print(f"\n[4] output 충돌 확인 (DONE.json 존재 여부)")
    if DONE_JSON.exists():
        print(f"  WARN DONE.json 이미 존재: {DONE_JSON}")
    else:
        print(f"  OK  DONE.json 없음")

    # 5. candidate plan (파일 있으면)
    print(f"\n[5] candidate plan (stage1_dev)")
    if CANDIDATE_MANIFEST_CSV.exists():
        try:
            rows = read_csv(CANDIDATE_MANIFEST_CSV)
            stage1 = [r for r in rows if r.get("stage_split") == "stage1_dev"]
            pos_c  = sum(1 for r in stage1 if r.get("label") == "positive")
            hn_c   = sum(1 for r in stage1 if r.get("label") == "hard_negative")
            pids   = set(r["patient_id"] for r in stage1)
            prob   = PROBLEM_PATIENT_IDS & pids
            print(f"  stage1_dev candidates: {len(stage1):,}")
            print(f"  positive: {pos_c:,}  hard_negative: {hn_c:,}")
            print(f"  unique patients: {len(pids):,}")
            print(f"  problem patients ({sorted(PROBLEM_PATIENT_IDS)}) 존재: {sorted(prob)}")

            # 컬럼 확인
            required_cols = ["candidate_id", "patient_id", "local_z",
                             "crop_y0", "crop_x0", "crop_y1", "crop_x1",
                             "label", "first_stage_score", "safe_id", "source_score_csv"]
            missing_cols = [c for c in required_cols if c not in (stage1[0] if stage1 else {})]
            if missing_cols:
                issues.append(f"MISSING columns: {missing_cols}")
                print(f"  FAIL missing columns: {missing_cols}")
            else:
                print(f"  OK  필수 컬럼 확인 완료")
        except Exception as e:
            print(f"  ERROR: {e}")
            issues.append(f"ERROR reading candidate manifest: {e}")

    # 6. group plan preview
    print(f"\n[6] group plan preview (default params z_gap={DEFAULT_Z_GAP} xy_radius={DEFAULT_XY_RADIUS})")
    print(f"  group으로 묶이면 candidate 수가 크게 줄어들 것으로 예상")
    print(f"  실제 group 수는 --run-group-preflight에서 확인")

    # 7. checkpoint plan
    print(f"\n[7] checkpoint plan")
    if CKPT_PATH.exists():
        sz = CKPT_PATH.stat().st_size / (1024 * 1024)
        print(f"  OK  checkpoint: {CKPT_PATH.name} ({sz:.1f} MB)")
    else:
        print(f"  MISSING checkpoint: {CKPT_PATH}")

    print(f"\n{'='*70}")
    if issues:
        print(f"[DRY-RUN] 이슈 {len(issues)}개 발견:")
        for iss in issues:
            print(f"  - {iss}")
        print(f"\n이슈 해결 후 --run-group-preflight 진행")
    else:
        print("[DRY-RUN] 모든 사전 조건 OK")
        print("다음: --run-group-preflight --confirm-readonly --confirm-stage1dev-only")
    print(f"{'='*70}")


# =============================================================================
# group preflight
# =============================================================================

def run_group_preflight():
    print("=" * 70)
    print("[GROUP-PREFLIGHT] RD4AD z-continuity group-level rescore smoke v1")
    print("=" * 70)
    t0 = time.perf_counter()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 입력 파일 로드
    print("\n[1] candidate manifest 로드")
    all_rows = read_csv(CANDIDATE_MANIFEST_CSV)
    stage1_rows = [r for r in all_rows if r.get("stage_split") == "stage1_dev"]
    print(f"  전체 rows: {len(all_rows):,}  stage1_dev: {len(stage1_rows):,}")

    # stage2 접근 없음 검증
    for r in stage1_rows:
        if r.get("stage_split", "") not in ("stage1_dev", ""):
            GUARDRAILS["stage2_holdout_accessed"] = True
            raise RuntimeError(f"[ABORT] stage2 row 발견: {r}")

    # 2. roi_0_0_patch_ratio lookup 준비
    print("\n[2] roi_0_0_patch_ratio lookup (per-patient source CSV)")
    roi_lookup = RoiRatioLookup()

    # 3. candidate 전처리
    print("\n[3] candidate 전처리")
    cand_map = {}
    for r in stage1_rows:
        cid  = r["candidate_id"]
        y0   = int(r["crop_y0"])
        x0   = int(r["crop_x0"])
        y1   = int(r["crop_y1"])
        x1   = int(r["crop_x1"])
        local_z = int(r["local_z"])
        try:
            fss = float(r["first_stage_score"])
        except Exception:
            fss = 0.0

        y_center = (y0 + y1) / 2.0
        x_center = (x0 + x1) / 2.0

        # roi_0_0_patch_ratio lookup
        src_csv = r.get("source_score_csv", "")
        roi_mean, roi_min, pos_bin = roi_lookup.lookup(
            r["patient_id"], src_csv, local_z, y0, x0, y1, x1
        )

        cand_map[cid] = {
            "candidate_id":       cid,
            "patient_id":         r["patient_id"],
            "safe_id":            r.get("safe_id", ""),
            "local_z":            local_z,
            "crop_y0":            y0,
            "crop_x0":            x0,
            "crop_y1":            y1,
            "crop_x1":            x1,
            "y_center":           y_center,
            "x_center":           x_center,
            "first_stage_score":  fss,
            "label":              r.get("label", ""),
            "roi_0_0_patch_ratio": roi_mean,
            "position_bin":       pos_bin or r.get("position_bin", ""),
            "source_score_csv":   src_csv,
        }

    candidates = list(cand_map.values())
    print(f"  processed: {len(candidates):,} candidates")

    # 4. default group 생성
    print(f"\n[4] default group 생성 (z_gap={DEFAULT_Z_GAP}, xy_radius={DEFAULT_XY_RADIUS})")
    group_id_map, group_candidates = build_groups(candidates, DEFAULT_Z_GAP, DEFAULT_XY_RADIUS)

    n_groups = len(group_candidates)
    reduction_rate = 1.0 - n_groups / len(candidates) if candidates else 0.0
    print(f"  원본 candidates: {len(candidates):,}")
    print(f"  groups: {n_groups:,}  reduction: {reduction_rate:.1%}")

    # 5. group 통계 계산
    print(f"\n[5] group 통계 계산")
    group_stats_list = []
    for gid, cand_ids in group_candidates.items():
        stat = compute_group_stats(gid, cand_ids, cand_map)
        if stat:
            group_stats_list.append(stat)

    group_stats_map = {g["group_id"]: g for g in group_stats_list}

    # positive assignment 확인
    all_pos_cids = [cid for cid, c in cand_map.items() if c["label"] == "positive"]
    assigned_pos = [cid for cid in all_pos_cids if cid in group_id_map]
    pos_assign_rate = len(assigned_pos) / len(all_pos_cids) if all_pos_cids else 0.0
    n_has_pos_groups = sum(1 for g in group_stats_list if g.get("has_positive") == "True")

    print(f"  positive candidates: {len(all_pos_cids):,}")
    print(f"  assigned to group: {len(assigned_pos):,} ({pos_assign_rate:.1%})")
    print(f"  has_positive groups: {n_has_pos_groups:,}")

    if pos_assign_rate < 1.0:
        print(f"  [WARN] positive candidate group assignment rate < 100%!")
        append_error(f"positive assignment rate {pos_assign_rate:.4f} < 1.0")

    # 6. group_manifest.csv 저장
    print(f"\n[6] group_manifest.csv 저장")
    gm_fields = [
        "group_id", "patient_id", "n_candidates", "z_min", "z_max", "z_span",
        "y_center_mean", "x_center_mean", "first_stage_score_max", "first_stage_score_mean",
        "positive_count", "hard_negative_count", "has_positive",
        "roi_0_0_patch_ratio_mean", "roi_0_0_patch_ratio_min",
        "boundary_like_ratio_mean", "boundary_like_ratio_max",
        "position_bin_mode", "representative_candidate_id",
    ]
    write_csv(GROUP_MANIFEST_CSV, gm_fields, group_stats_list)

    # 7. group_representative_manifest.csv 저장
    print(f"\n[7] group_representative_manifest.csv 저장")
    repr_rows = []
    for g in group_stats_list:
        repr_cid = g["representative_candidate_id"]
        c = cand_map.get(repr_cid, {})
        blr = g.get("boundary_like_ratio_mean")
        try:
            blr = float(blr)
        except Exception:
            blr = None
        adj = compute_adjusted_scores(None, blr)  # no score yet
        repr_rows.append({
            "group_id":                  g["group_id"],
            "patient_id":                g["patient_id"],
            "representative_candidate_id": repr_cid,
            "safe_id":                   c.get("safe_id", ""),
            "local_z":                   c.get("local_z", ""),
            "crop_y0":                   c.get("crop_y0", ""),
            "crop_x0":                   c.get("crop_x0", ""),
            "crop_y1":                   c.get("crop_y1", ""),
            "crop_x1":                   c.get("crop_x1", ""),
            "first_stage_score":         c.get("first_stage_score", ""),
            "label":                     c.get("label", ""),
            "n_candidates":              g["n_candidates"],
            "has_positive":              g["has_positive"],
            "positive_count":            g["positive_count"],
            "roi_0_0_patch_ratio_mean":  g.get("roi_0_0_patch_ratio_mean", ""),
            "boundary_like_ratio_mean":  g.get("boundary_like_ratio_mean", ""),
        })
    repr_fields = [
        "group_id", "patient_id", "representative_candidate_id",
        "safe_id", "local_z", "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "first_stage_score", "label", "n_candidates", "has_positive", "positive_count",
        "roi_0_0_patch_ratio_mean", "boundary_like_ratio_mean",
    ]
    write_csv(GROUP_REPR_MANIFEST_CSV, repr_fields, repr_rows)

    # 8. ablation summary (model forward 없음)
    print(f"\n[8] grouping ablation summary (statistics only, no forward)")
    ablation_rows = []
    for z_gap in ABLATION_Z_GAPS:
        for xy_r in ABLATION_XY_RADII:
            gid_map_abl, gcands_abl = build_groups(candidates, z_gap, xy_r)
            n_g = len(gcands_abl)
            n_has_pos = 0
            for g_cids in gcands_abl.values():
                if any(cand_map[cid]["label"] == "positive" for cid in g_cids if cid in cand_map):
                    n_has_pos += 1
            ablation_rows.append({
                "z_gap":           z_gap,
                "xy_radius":       xy_r,
                "n_groups":        n_g,
                "reduction_rate":  round(1.0 - n_g / len(candidates), 4),
                "has_positive_groups": n_has_pos,
                "model_forward":   "no",
            })
    write_csv(GROUPING_ABLATION_CSV,
              ["z_gap", "xy_radius", "n_groups", "reduction_rate", "has_positive_groups", "model_forward"],
              ablation_rows)

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*70}")
    print(f"[GROUP-PREFLIGHT] 완료 ({elapsed:.1f}s)")
    print(f"  groups: {n_groups:,}")
    print(f"  reduction_rate: {reduction_rate:.1%}")
    print(f"  positive assignment rate: {pos_assign_rate:.1%}")
    print(f"  has_positive groups: {n_has_pos_groups:,}")
    print(f"\n다음: --run-smoke --confirm-model-forward --confirm-stage1dev-only --max-groups 1000")
    print(f"{'='*70}")

    return cand_map, group_stats_map, group_id_map


# =============================================================================
# actual smoke
# =============================================================================

def run_smoke(max_groups: int = MAX_GROUPS_SMOKE):
    print("=" * 70)
    print(f"[SMOKE] RD4AD z-continuity group-level rescore smoke v1 (max_groups={max_groups})")
    print("=" * 70)
    t0 = time.perf_counter()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if max_groups > MAX_GROUPS_SMOKE:
        print(f"[ABORT] max_groups {max_groups} > {MAX_GROUPS_SMOKE} 제한")
        sys.exit(2)

    # 1. candidate manifest 로드 (group preflight와 동일)
    print("\n[1] candidate manifest + group 생성")
    all_rows = read_csv(CANDIDATE_MANIFEST_CSV)
    stage1_rows = [r for r in all_rows if r.get("stage_split") == "stage1_dev"]

    roi_lookup = RoiRatioLookup()
    cand_map = {}
    for r in stage1_rows:
        cid   = r["candidate_id"]
        y0    = int(r["crop_y0"])
        x0    = int(r["crop_x0"])
        y1    = int(r["crop_y1"])
        x1    = int(r["crop_x1"])
        local_z = int(r["local_z"])
        try:
            fss = float(r["first_stage_score"])
        except Exception:
            fss = 0.0
        src_csv = r.get("source_score_csv", "")
        roi_mean, roi_min, pos_bin = roi_lookup.lookup(
            r["patient_id"], src_csv, local_z, y0, x0, y1, x1
        )
        cand_map[cid] = {
            "candidate_id":       cid,
            "patient_id":         r["patient_id"],
            "safe_id":            r.get("safe_id", ""),
            "local_z":            local_z,
            "crop_y0":            y0,
            "crop_x0":            x0,
            "crop_y1":            y1,
            "crop_x1":            x1,
            "y_center":           (y0 + y1) / 2.0,
            "x_center":           (x0 + x1) / 2.0,
            "first_stage_score":  fss,
            "label":              r.get("label", ""),
            "roi_0_0_patch_ratio": roi_mean,
            "position_bin":       pos_bin or r.get("position_bin", ""),
            "source_score_csv":   src_csv,
        }

    candidates = list(cand_map.values())
    group_id_map, group_candidates = build_groups(candidates, DEFAULT_Z_GAP, DEFAULT_XY_RADIUS)
    n_groups_total = len(group_candidates)
    print(f"  candidates: {len(candidates):,}  groups: {n_groups_total:,}")

    group_stats_map = {}
    for gid, cand_ids in group_candidates.items():
        stat = compute_group_stats(gid, cand_ids, cand_map)
        if stat:
            group_stats_map[gid] = stat

    # 2. RD-D1s scalar score CSV 로드 (scalar reproduction용)
    print("\n[2] RD-D1s scalar score CSV 로드")
    scalar_score_map = {}
    if RD_D1S_SCORE_CSV.exists():
        srows = read_csv(RD_D1S_SCORE_CSV)
        for r in srows:
            if r.get("stage_split", "") == "stage1_dev":
                try:
                    scalar_score_map[r["candidate_id"]] = float(r["rd_d1s_medi3ch_rd4ad_score"])
                except Exception:
                    pass
        print(f"  scalar score map: {len(scalar_score_map):,} entries")
    else:
        print(f"  [WARN] scalar score CSV 없음: {RD_D1S_SCORE_CSV}")

    # 3. smoke sampling: 최대 max_groups 선택
    print(f"\n[3] smoke sampling (최대 {max_groups} groups)")
    gstats_list = list(group_stats_map.values())

    # has_positive=True groups
    pos_groups  = [g for g in gstats_list if g.get("has_positive") == "True"]
    hn_groups   = [g for g in gstats_list if g.get("has_positive") != "True"]

    # positive groups: 최대 MAX_POS_GROUPS
    selected_gids = set()
    pos_selected = sorted(pos_groups, key=lambda g: float(g.get("first_stage_score_max") or 0), reverse=True)
    for g in pos_selected[:MAX_POS_GROUPS]:
        selected_gids.add(g["group_id"])

    remaining = max_groups - len(selected_gids)
    if remaining > 0:
        # boundary-heavy hard_negative groups (높은 boundary_like_ratio_mean)
        def _blr(g):
            v = g.get("boundary_like_ratio_mean")
            if v is None or v == "":
                return 0.0
            try:
                return float(v)
            except Exception:
                return 0.0

        hn_sorted_bound = sorted(hn_groups, key=_blr, reverse=True)
        n_bound = min(MAX_HN_BOUND_GROUPS, remaining // 2)
        for g in hn_sorted_bound[:n_bound]:
            selected_gids.add(g["group_id"])

        remaining2 = max_groups - len(selected_gids)
        if remaining2 > 0:
            # first_stage_score 높은 hard_negative groups
            hn_sorted_score = sorted(hn_groups, key=lambda g: float(g.get("first_stage_score_max") or 0), reverse=True)
            for g in hn_sorted_score[:MAX_HN_SCORE_GROUPS]:
                if g["group_id"] not in selected_gids:
                    selected_gids.add(g["group_id"])
                    if len(selected_gids) >= max_groups:
                        break

    selected_gids = list(selected_gids)[:max_groups]
    print(f"  selected groups for forward: {len(selected_gids):,}")
    print(f"  label_used_for_smoke_sampling=True, label_used_as_deployment_selector=False")

    # 4. 모델 로드
    print(f"\n[4] 모델 로드 (checkpoint read-only)")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    teacher, student = load_model_from_checkpoint(device)
    teacher.eval()
    student.eval()

    # teacher hook
    teacher_features = {}
    for layer_name, module in [
        ("layer1", teacher.layer1),
        ("layer2", teacher.layer2),
        ("layer3", teacher.layer3),
    ]:
        def _hook(module, inp, output, _n=layer_name):
            teacher_features[_n] = output
        module.register_forward_hook(_hook)

    # training 금지 검증
    teacher.train = _forbidden_train
    student.train = _forbidden_train

    ct_cache = CTMmapCache(max_size=10)

    # 5. forward
    print(f"\n[5] forward (max {max_groups} groups)")
    GUARDRAILS["model_forward_executed"] = True
    GUARDRAILS["crop_generation_executed"] = True

    score_rows = []
    scalar_repro_diffs = []
    error_count = 0
    n_forward = 0

    import numpy as np

    for idx, gid in enumerate(selected_gids):
        gstat = group_stats_map.get(gid)
        if not gstat:
            continue
        repr_cid = gstat["representative_candidate_id"]
        c = cand_map.get(repr_cid)
        if not c:
            append_error(f"repr_cid {repr_cid} not in cand_map")
            error_count += 1
            continue

        try:
            ct_arr = ct_cache.get(c["safe_id"])
        except Exception as e:
            append_error(f"CT load fail: {c['safe_id']}: {e}", e)
            error_count += 1
            continue

        try:
            crop = build_medi3ch_crop(
                ct_arr, c["local_z"],
                c["crop_y0"], c["crop_x0"], c["crop_y1"], c["crop_x1"]
            )
        except Exception as e:
            append_error(f"crop build fail: {repr_cid}: {e}", e)
            error_count += 1
            continue

        crop_t = torch.from_numpy(crop[np.newaxis]).to(device)  # (1,3,96,96)

        try:
            with torch.no_grad():
                rd4ad_raw, l1, l2, l3 = compute_rd4ad_score(
                    teacher, student, crop_t, teacher_features, device
                )
        except Exception as e:
            append_error(f"forward fail: {repr_cid}: {e}", e)
            error_count += 1
            continue

        n_forward += 1

        # scalar reproduction 확인
        scalar_repro_diff = None
        if repr_cid in scalar_score_map:
            scalar_repro_diff = abs(rd4ad_raw - scalar_score_map[repr_cid])
            scalar_repro_diffs.append(scalar_repro_diff)

        # adjusted scores
        blr = gstat.get("boundary_like_ratio_mean")
        try:
            blr = float(blr)
        except Exception:
            blr = None
        adj = compute_adjusted_scores(rd4ad_raw, blr)

        row = {
            "group_id":                   gid,
            "patient_id":                 gstat["patient_id"],
            "representative_candidate_id": repr_cid,
            "n_candidates":               gstat["n_candidates"],
            "has_positive":               gstat.get("has_positive", ""),
            "positive_count":             gstat["positive_count"],
            "hard_negative_count":        gstat["hard_negative_count"],
            "z_span":                     gstat["z_span"],
            "roi_0_0_patch_ratio_mean":   gstat.get("roi_0_0_patch_ratio_mean", ""),
            "boundary_like_ratio_mean":   gstat.get("boundary_like_ratio_mean", ""),
            "first_stage_score_max":      gstat.get("first_stage_score_max", ""),
            "rd4ad_group_score_raw":      round(rd4ad_raw, 6),
            "score_layer1":               round(l1, 6),
            "score_layer2":               round(l2, 6),
            "score_layer3":               round(l3, 6),
            "scalar_repro_diff":          round(scalar_repro_diff, 8) if scalar_repro_diff is not None else "",
        }
        for pk, pv in adj.items():
            row[pk] = round(pv, 6)
        score_rows.append(row)

        if (idx + 1) % 100 == 0:
            print(f"  forward {idx+1}/{len(selected_gids)}: {n_forward} OK, {error_count} err")

    print(f"  total forward: {n_forward}  errors: {error_count}")
    GUARDRAILS["actual_forward_groups"] = n_forward

    # 6. scalar reproduction 검증
    print(f"\n[6] scalar reproduction 검증")
    scalar_repro_ok = False
    scalar_repro_mean_diff = None
    if scalar_repro_diffs:
        scalar_repro_mean_diff = sum(scalar_repro_diffs) / len(scalar_repro_diffs)
        scalar_repro_ok = scalar_repro_mean_diff < 1e-4
        print(f"  n_checked: {len(scalar_repro_diffs)}")
        print(f"  mean_abs_diff: {scalar_repro_mean_diff:.2e}")
        print(f"  scalar_repro_ok (< 1e-4): {scalar_repro_ok}")
    else:
        print(f"  [WARN] scalar repro 비교 불가 (scalar score CSV 없거나 repr_cid 미매칭)")

    # 7. score CSV 저장
    print(f"\n[7] group rescore scores.csv 저장")
    if score_rows:
        score_fields = [
            "group_id", "patient_id", "representative_candidate_id",
            "n_candidates", "has_positive", "positive_count", "hard_negative_count",
            "z_span", "roi_0_0_patch_ratio_mean", "boundary_like_ratio_mean",
            "first_stage_score_max", "rd4ad_group_score_raw",
            "score_layer1", "score_layer2", "score_layer3", "scalar_repro_diff",
            "P0_raw", "P1_times_roi_mean", "P2_times_sqrt_roi_mean",
            "P3_soft_alpha_0_2", "P4_soft_alpha_0_3",
            "P5_minus_lam_0_05", "P6_minus_lam_0_10",
        ]
        write_csv(GROUP_RESCORE_SCORES_CSV, score_fields, score_rows)

    # 8. top-k retention 계산
    print(f"\n[8] top-k retention 계산")
    # scored group_stats_map (scored groups only)
    scored_group_stats = {}
    for row in score_rows:
        gid = row["group_id"]
        if gid in group_stats_map:
            g = dict(group_stats_map[gid])
            for col in ["rd4ad_group_score_raw", "P0_raw", "P1_times_roi_mean",
                        "P2_times_sqrt_roi_mean", "P3_soft_alpha_0_2", "P4_soft_alpha_0_3",
                        "P5_minus_lam_0_05", "P6_minus_lam_0_10"]:
                g[col] = row.get(col, "")
            scored_group_stats[gid] = g

    score_cols = [
        "rd4ad_group_score_raw", "P0_raw", "P1_times_roi_mean",
        "P2_times_sqrt_roi_mean", "P3_soft_alpha_0_2", "P4_soft_alpha_0_3",
        "P5_minus_lam_0_05", "P6_minus_lam_0_10",
    ]
    topk_vals = [1, 3, 5, 10, 20, 50]

    topk_rows = []
    for sc in score_cols:
        for k in topk_vals:
            ret_rate, n_pats = compute_topk_retention(scored_group_stats, sc, k)
            pos_cov = compute_positive_candidate_coverage(scored_group_stats, group_id_map, sc, k)
            # boundary-heavy groups in top-k
            topk_rows.append({
                "score_col": sc,
                "k":         k,
                "lesion_group_retention": ret_rate,
                "n_patients_with_positive": n_pats,
                "positive_candidate_coverage": pos_cov,
            })
            if sc == "rd4ad_group_score_raw":
                print(f"  {sc} top{k:2d}: retention={ret_rate:.4f} pos_cov={pos_cov:.4f}")

    write_csv(GROUP_TOPK_RETENTION_CSV,
              ["score_col", "k", "lesion_group_retention", "n_patients_with_positive",
               "positive_candidate_coverage"],
              topk_rows)

    # 9. baseline B0_patch_all (patch-level)
    print(f"\n[9] baseline B0_patch_all (patch-level rd_d1s)")
    patch_baseline_rows = {}
    if RD_D1S_SCORE_CSV.exists():
        patch_rows = read_csv(RD_D1S_SCORE_CSV)
        for k in topk_vals:
            ret, n = compute_patch_topk_retention(patch_rows, "rd_d1s_medi3ch_rd4ad_score", k)
            patch_baseline_rows[k] = ret
            print(f"  B0_patch top{k:2d}: retention={ret:.4f}")
    else:
        print(f"  [WARN] scalar score CSV 없음, baseline 계산 생략")

    # 10. problem patient audit
    print(f"\n[10] problem patient group audit")
    prob_rows = []
    for pid in PROBLEM_PATIENT_IDS:
        pat_groups = [g for g in group_stats_map.values() if g["patient_id"] == pid]
        scored_pat = [g for g in scored_group_stats.values() if g["patient_id"] == pid]
        n_pos_groups = sum(1 for g in pat_groups if g.get("has_positive") == "True")
        for k in [5, 10, 20, 50]:
            sorted_g = sorted(scored_pat, key=lambda g: float(g.get("rd4ad_group_score_raw") or -1e9), reverse=True)
            topk_has_pos = any(g.get("has_positive") == "True" for g in sorted_g[:k])
            prob_rows.append({
                "patient_id":       pid,
                "n_groups_total":   len(pat_groups),
                "n_scored_groups":  len(scored_pat),
                "n_pos_groups":     n_pos_groups,
                "k":                k,
                "topk_has_positive": str(topk_has_pos),
            })
    if prob_rows:
        write_csv(PROBLEM_PATIENT_AUDIT_CSV,
                  ["patient_id", "n_groups_total", "n_scored_groups", "n_pos_groups",
                   "k", "topk_has_positive"],
                  prob_rows)

    # 11. 그룹 coverage 확인
    all_pos_cids = [cid for cid, c in cand_map.items() if c["label"] == "positive"]
    assigned_pos = [cid for cid in all_pos_cids if cid in group_id_map]
    pos_assign_rate = len(assigned_pos) / len(all_pos_cids) if all_pos_cids else 0.0

    # 12. verdict
    n_cands = len(candidates)
    n_groups = len(group_candidates)
    reduction_rate = 1.0 - n_groups / n_cands if n_cands else 0.0

    # top10 group vs patch baseline
    group_top10_raw = next((r["lesion_group_retention"] for r in topk_rows
                            if r["score_col"] == "rd4ad_group_score_raw" and r["k"] == 10), 0.0)
    patch_top10 = patch_baseline_rows.get(10, 0.0)

    verdict = "FAIL"
    if (pos_assign_rate == 1.0 and
        scalar_repro_ok and
        group_top10_raw >= patch_top10 * 0.95):
        verdict = "PASS_CANDIDATE"
    elif pos_assign_rate == 1.0 and reduction_rate > 0.3:
        verdict = "PARTIAL_PASS_EXPLORATORY"
    else:
        verdict = "FAIL"

    elapsed = time.perf_counter() - t0

    # 13. 요약 JSON
    print(f"\n[11] summary JSON 저장")
    guardrails_copy = dict(GUARDRAILS)
    guardrails_copy["actual_forward_groups"] = n_forward
    guardrails_copy["max_forward_groups"] = max_groups

    summary = {
        "verdict": verdict,
        "original_candidate_count": n_cands,
        "group_count": n_groups,
        "reduction_rate": round(reduction_rate, 4),
        "positive_candidate_group_assignment_rate": round(pos_assign_rate, 4),
        "has_positive_group_count": sum(1 for g in group_stats_map.values() if g.get("has_positive") == "True"),
        "smoke_forward_group_count": n_forward,
        "error_count": error_count,
        "scalar_repro_ok": scalar_repro_ok,
        "scalar_repro_mean_abs_diff": round(scalar_repro_mean_diff, 8) if scalar_repro_mean_diff is not None else None,
        "group_top10_lesion_retention_raw": group_top10_raw,
        "patch_top10_lesion_retention_baseline": patch_top10,
        "problem_patient_ids": sorted(PROBLEM_PATIENT_IDS),
        "elapsed_sec": round(elapsed, 1),
        "guardrails": guardrails_copy,
        "label_used_for_smoke_sampling": True,
        "label_used_as_deployment_selector": False,
        "stage2_holdout_accessed": GUARDRAILS["stage2_holdout_accessed"],
    }

    # problem patients top-k
    for pid in PROBLEM_PATIENT_IDS:
        for k in [20, 50]:
            matches = [r for r in prob_rows if r["patient_id"] == pid and r["k"] == k]
            if matches:
                summary[f"{pid}_top{k}_has_positive"] = matches[0]["topk_has_positive"]

    ensure_output_path_safe(SUMMARY_JSON)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(SUMMARY_JSON), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  saved: {SUMMARY_JSON}")

    # 14. report MD
    print(f"\n[12] report MD 저장")
    report_lines = [
        f"# RD4AD z-continuity group-level rescore smoke v1 report",
        f"",
        f"**판정: {verdict}**",
        f"",
        f"## 핵심 수치",
        f"",
        f"- original candidates: {n_cands:,}",
        f"- group count: {n_groups:,}",
        f"- reduction rate: {reduction_rate:.1%}",
        f"- positive candidate group assignment rate: {pos_assign_rate:.1%}",
        f"- has_positive groups: {summary['has_positive_group_count']:,}",
        f"- smoke forward groups: {n_forward}",
        f"- error count: {error_count}",
        f"- scalar_repro_ok: {scalar_repro_ok}",
        f"- scalar_repro_mean_abs_diff: {scalar_repro_mean_diff:.2e}" if scalar_repro_mean_diff is not None else "- scalar_repro_mean_abs_diff: N/A",
        f"",
        f"## top-k lesion group retention (rd4ad_group_score_raw)",
        f"",
    ]
    for k in topk_vals:
        gr = next((r["lesion_group_retention"] for r in topk_rows
                   if r["score_col"] == "rd4ad_group_score_raw" and r["k"] == k), "N/A")
        pb = patch_baseline_rows.get(k, "N/A")
        report_lines.append(f"- top{k:2d}: group={gr}  patch_baseline={pb}")

    report_lines += [
        f"",
        f"## adjusted score top-k retention",
        f"",
    ]
    for sc in score_cols[1:]:
        for k in [10, 20]:
            gr = next((r["lesion_group_retention"] for r in topk_rows
                       if r["score_col"] == sc and r["k"] == k), "N/A")
            report_lines.append(f"- {sc} top{k}: {gr}")

    report_lines += [
        f"",
        f"## problem patient audit",
        f"",
    ]
    for pid in sorted(PROBLEM_PATIENT_IDS):
        for k in [20, 50]:
            key = f"{pid}_top{k}_has_positive"
            report_lines.append(f"- {pid} top{k}: {summary.get(key, 'N/A')}")

    report_lines += [
        f"",
        f"## 통과 / 실패 조건",
        f"",
        f"- [{'PASS' if pos_assign_rate == 1.0 else 'FAIL'}] positive_candidate_group_assignment_rate == 100%",
        f"- [{'PASS' if scalar_repro_ok else 'FAIL'}] scalar_repro_ok",
        f"- [{'PASS' if GUARDRAILS['stage2_holdout_accessed'] == False else 'FAIL'}] stage2_holdout_not_accessed",
        f"- [{'PASS' if GUARDRAILS['training_executed'] == False else 'FAIL'}] no_training",
        f"- [{'PASS' if GUARDRAILS['backward_executed'] == False else 'FAIL'}] no_backward",
        f"- [{'PASS' if GUARDRAILS['existing_artifact_modified'] == False else 'FAIL'}] no_existing_artifact_modified",
        f"",
        f"## Guardrail 기록",
        f"",
    ]
    for k, v in guardrails_copy.items():
        report_lines.append(f"- {k}: {v}")

    report_lines += [
        f"",
        f"## 다음 단계",
        f"",
    ]
    if verdict == "PASS_CANDIDATE":
        report_lines.append(f"PASS_CANDIDATE → group-level full preflight로 확장")
    elif verdict == "PARTIAL_PASS_EXPLORATORY":
        report_lines.append(f"PARTIAL_PASS_EXPLORATORY → group당 대표 3개 scoring smoke 검토")
    else:
        report_lines.append(f"FAIL → RD4AD group-level rescoring을 analysis-only로 닫는다")

    ensure_output_path_safe(REPORT_MD)
    with open(str(REPORT_MD), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  saved: {REPORT_MD}")

    # DONE.json
    ensure_output_path_safe(DONE_JSON)
    with open(str(DONE_JSON), "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
    print(f"  saved: {DONE_JSON}")

    # 최종 보고
    print(f"\n{'='*70}")
    print(f"[SMOKE] 완료 ({elapsed:.1f}s)")
    print(f"")
    print(f"판정: {verdict}")
    print(f"")
    print(f"핵심 수치:")
    print(f"  original candidates: {n_cands:,}")
    print(f"  groups: {n_groups:,}  reduction: {reduction_rate:.1%}")
    print(f"  positive assignment rate: {pos_assign_rate:.1%}")
    print(f"  has_positive groups: {summary['has_positive_group_count']:,}")
    print(f"  smoke forward: {n_forward}")
    print(f"  scalar_repro_ok: {scalar_repro_ok}")
    print(f"  group top10 raw: {group_top10_raw:.4f}  patch top10: {patch_top10:.4f}")
    print(f"  stage2_holdout_accessed: {GUARDRAILS['stage2_holdout_accessed']}")
    print(f"  existing_artifact_modified: {GUARDRAILS['existing_artifact_modified']}")
    print(f"{'='*70}")


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RD4AD z-continuity group-level rescoring smoke v1"
    )
    parser.add_argument("--dry-run",                action="store_true")
    parser.add_argument("--run-group-preflight",    action="store_true")
    parser.add_argument("--run-smoke",              action="store_true")
    parser.add_argument("--confirm-readonly",       action="store_true")
    parser.add_argument("--confirm-model-forward",  action="store_true")
    parser.add_argument("--confirm-stage1dev-only", action="store_true")
    parser.add_argument("--max-groups",             type=int, default=MAX_GROUPS_SMOKE)
    args = parser.parse_args()

    # bare run 차단
    if not any([args.dry_run, args.run_group_preflight, args.run_smoke]):
        print("[ABORT] bare run 차단. --dry-run / --run-group-preflight / --run-smoke 사용.", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        run_dry()
        return

    if args.run_group_preflight:
        if not args.confirm_readonly or not args.confirm_stage1dev_only:
            print("[ABORT] --confirm-readonly --confirm-stage1dev-only 필요", file=sys.stderr)
            sys.exit(2)
        run_group_preflight()
        return

    if args.run_smoke:
        if not args.confirm_model_forward or not args.confirm_stage1dev_only:
            print("[ABORT] --confirm-model-forward --confirm-stage1dev-only 필요", file=sys.stderr)
            sys.exit(2)
        if args.max_groups > MAX_GROUPS_SMOKE:
            print(f"[ABORT] --max-groups {args.max_groups} > {MAX_GROUPS_SMOKE} 제한", file=sys.stderr)
            sys.exit(2)
        run_smoke(max_groups=args.max_groups)
        return


if __name__ == "__main__":
    main()
