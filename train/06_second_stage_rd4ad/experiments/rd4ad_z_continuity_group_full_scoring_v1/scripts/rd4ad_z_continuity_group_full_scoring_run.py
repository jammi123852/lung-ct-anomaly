"""
RD4AD z-continuity group-level full scoring RUN v1

stage1_dev 전체 20,216 groups 에 대해 RD-D1s RD4AD group-level scoring 을
4-shard 로 실행한다. 한 번에 한 shard 만 처리한다.

설계 근거(검증된 로직 재사용):
  experiments/rd4ad_z_continuity_group_rescore_smoke_v1/scripts/
    rd4ad_z_continuity_group_rescore_smoke.py  (smoke v1, PASS_CANDIDATE)
  experiments/rd4ad_z_continuity_group_full_scoring_v1/scripts/
    rd4ad_z_continuity_group_full_scoring_preflight.py  (preflight, PASS)

  build_groups / compute_group_stats / build_teacher / build_student_decoder /
  load_model_from_checkpoint / build_medi3ch_crop / compute_rd4ad_score /
  RoiRatioLookup / CTMmapCache / compute_adjusted_scores 는
  smoke v1 과 동일 로직을 그대로 사용한다(scalar_repro_ok 재현 보장 목적).

group_id 일관성:
  preflight 와 smoke 의 build_groups 는 동일하며, candidate manifest 를
  file 순서로 읽어 group_id(G{counter:07d}) 를 결정한다.
  따라서 본 run 의 rebuild 결과 group_id 집합은
  group_manifest_full.csv / full_scoring_shard_plan.csv 와 동일해야 한다.
  run-shard 시작 시 이를 hard-validate 하고, 불일치하면 abort 한다.

실행 방식:
  bare run (인자 없음): exit 2 로 막음
  dry-run:    python <script> --dry-run
  run-shard:  python <script> --run-shard --shard-id {0,1,2,3} \
                  --confirm-model-forward --confirm-stage1dev-only

금지:
  training / backward / optimizer / checkpoint save / stage2_holdout 접근 /
  기존 artifact(다른 실험/체크포인트/manifest/score CSV) 수정 / 입력 manifest 덮어쓰기.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict, OrderedDict
from pathlib import Path

# =============================================================================
# 경로 상수 (preflight 와 동일, read-only 입력)
# =============================================================================

PROJECT_ROOT    = Path("/home/jinhy/project/lung-ct-anomaly")
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/rd4ad_z_continuity_group_full_scoring_v1"

# 입력 (read-only)
CANDIDATE_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c2_effb0_v420_candidate_rd4ad_retest_v1"
    / "rd_c2_effb0_v420_candidate_manifest.csv"
)
RD_D1S_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1s_medi3ch_true_rd4ad_shard_run_v1"
    / "rd_d1s_stage1dev_candidate_score.csv"
)
EFFB0_SCORE_BASE = (
    PROJECT_ROOT
    / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs/scores"
)
CKPT_PATH = (
    PROJECT_ROOT
    / "outputs/models/rd_d1s_true_rd4ad_resnet18_medi3ch_shard_v1"
    / "checkpoints/best_train_loss.pth"
)
LOCAL_RESNET_WEIGHT = Path("/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)

# 입력 manifest (preflight 가 생성한 것, read-only)
MANIFEST_DIR            = EXPERIMENT_ROOT / "manifests"
GROUP_MANIFEST_CSV      = MANIFEST_DIR / "group_manifest_full.csv"
GROUP_REPR_MANIFEST_CSV = MANIFEST_DIR / "group_representative_manifest_full.csv"
SHARD_PLAN_CSV          = MANIFEST_DIR / "full_scoring_shard_plan.csv"

# 출력 (shards/ 하위에만)
SHARDS_DIR = EXPERIMENT_ROOT / "shards"

# group 파라미터 (smoke v1 / preflight 와 동일)
DEFAULT_Z_GAP     = 1
DEFAULT_XY_RADIUS = 24
SHARD_COUNT       = 4

# HU 윈도잉 / crop
HU_MIN, HU_MAX = -160.0, 240.0
CROP_SIZE = 96

# =============================================================================
# guardrail 상태
# =============================================================================

GUARDRAILS = {
    "stage2_holdout_accessed":     False,
    "checkpoint_loaded":           False,
    "model_forward_executed":      False,
    "training_executed":           False,
    "backward_executed":           False,
    "optimizer_created":           False,
    "checkpoint_saved":            False,
    "crop_generation_executed":    False,
    "full_scoring_executed":       False,   # shard 단위 → "shard_only" 로 기록
    "threshold_recalculated":      False,
    "existing_artifact_modified":  False,
    "existing_script_modified":    False,
    "output_overwrite":            False,
    "raw_rd4ad_primary_score":     True,
    "adjusted_score_preview_only": True,
}

# =============================================================================
# 안전 경로 검사
# =============================================================================

def assert_path_safe(p: Path):
    s = str(p).lower()
    if "stage2_holdout" in s or ("stage2" in s and "holdout" in s):
        GUARDRAILS["stage2_holdout_accessed"] = True
        raise RuntimeError(f"[ABORT] stage2_holdout 경로 접근 차단: {p}")


# 절대 덮어쓰면 안 되는 입력 manifest
_PROTECTED_INPUTS = {
    GROUP_MANIFEST_CSV.resolve(),
    GROUP_REPR_MANIFEST_CSV.resolve(),
    SHARD_PLAN_CSV.resolve(),
}


def ensure_output_path_safe(p: Path):
    """쓰기 경로는 반드시 shards/ 하위여야 하며, 입력 manifest 는 금지."""
    rp = Path(p).resolve()
    if rp in _PROTECTED_INPUTS:
        GUARDRAILS["existing_artifact_modified"] = True
        raise RuntimeError(f"[ABORT] 입력 manifest 덮어쓰기 차단: {p}")
    shards_root = SHARDS_DIR.resolve()
    if not str(rp).startswith(str(shards_root)):
        GUARDRAILS["existing_artifact_modified"] = True
        raise RuntimeError(f"[ABORT] run-shard 는 shards/ 외부 쓰기 금지: {p}")


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


def make_error_logger(error_csv: Path):
    """shard 별 errors.csv 에 append 하는 logger 반환."""
    def _append(msg: str, exc: Exception = None):
        ensure_output_path_safe(error_csv)
        error_csv.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if error_csv.exists() else "w"
        with open(str(error_csv), mode, encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if mode == "w":
                w.writerow(["timestamp", "message", "traceback"])
            tb = traceback.format_exc() if exc else ""
            w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), msg, tb.replace("\n", " | ")])
    return _append


# =============================================================================
# roi_0_0_patch_ratio 조회 (smoke v1 와 동일)
# =============================================================================

class RoiRatioLookup:
    def __init__(self):
        self._cache = {}

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
        self._load_patient(patient_id, source_csv_rel)
        patches = self._cache.get(patient_id, [])
        y_center = (crop_y0 + crop_y1) / 2.0
        x_center = (crop_x0 + crop_x1) / 2.0

        matched_ratios = []
        matched_bins = []
        for p in patches:
            if p["local_z"] != local_z:
                continue
            py_center = (p["y0"] + p["y1"]) / 2.0
            px_center = (p["x0"] + p["x1"]) / 2.0
            if crop_y0 <= py_center < crop_y1 and crop_x0 <= px_center < crop_x1:
                matched_ratios.append(p["roi_0_0_patch_ratio"])
                if p["position_bin"]:
                    matched_bins.append(p["position_bin"])

        if not matched_ratios:
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
        mode_bin = ""
        if matched_bins:
            bin_count = defaultdict(int)
            for b in matched_bins:
                bin_count[b] += 1
            mode_bin = max(bin_count, key=bin_count.__getitem__)
        return mean_ratio, min_ratio, mode_bin


# =============================================================================
# union-find / build_groups (smoke v1 / preflight 와 동일)
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


def build_groups(candidates, z_gap=DEFAULT_Z_GAP, xy_radius=DEFAULT_XY_RADIUS):
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
# group 통계 (smoke v1 와 동일)
# =============================================================================

def compute_group_stats(group_id, cand_ids, cand_map):
    rows = [cand_map[cid] for cid in cand_ids if cid in cand_map]
    if not rows:
        return None

    patient_id = rows[0]["patient_id"]
    z_vals    = [r["local_z"] for r in rows]
    fss       = [r["first_stage_score"] for r in rows]
    labels    = [r["label"] for r in rows]
    roi_ratios = [r["roi_0_0_patch_ratio"] for r in rows if r.get("roi_0_0_patch_ratio") is not None]

    positive_count = sum(1 for l in labels if l == "positive")
    hn_count       = sum(1 for l in labels if l == "hard_negative")
    has_positive   = positive_count > 0

    roi_mean = sum(roi_ratios) / len(roi_ratios) if roi_ratios else None
    boundary_like_ratio_mean = (1.0 - roi_mean) if roi_mean is not None else None

    repr_cid = max(rows, key=lambda r: r["first_stage_score"])["candidate_id"]

    return {
        "group_id":                    group_id,
        "patient_id":                  patient_id,
        "n_candidates":                len(rows),
        "z_min":                       min(z_vals),
        "z_max":                       max(z_vals),
        "z_span":                      max(z_vals) - min(z_vals),
        "first_stage_score_max":       max(fss),
        "first_stage_score_mean":      round(sum(fss) / len(fss), 6),
        "positive_count":              positive_count,
        "hard_negative_count":         hn_count,
        "has_positive":                str(has_positive),
        "roi_0_0_patch_ratio_mean":    round(roi_mean, 6) if roi_mean is not None else "",
        "boundary_like_ratio_mean":    round(boundary_like_ratio_mean, 6) if boundary_like_ratio_mean is not None else "",
        "representative_candidate_id": repr_cid,
    }


# =============================================================================
# 모델 빌드 (smoke v1 와 동일)
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
# crop / score (smoke v1 와 동일)
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


def compute_rd4ad_score(teacher, student, crop_tensor, teacher_features, device):
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

    scores = []
    for tf, sf in [(tf3, de3), (tf2, de2), (tf1, de1)]:
        cos_sim = F.cosine_similarity(tf, sf, dim=1, eps=1e-8)
        level_score = float((1.0 - cos_sim).mean().item())
        scores.append(level_score)

    scalar = float(sum(scores) / len(scores))
    return scalar, scores[2], scores[1], scores[0]  # scalar, layer1, layer2, layer3


# =============================================================================
# CT mmap 캐시 (smoke v1 와 동일)
# =============================================================================

class CTMmapCache:
    def __init__(self, max_size=10):
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
# adjusted score preview (P1~P4 만, smoke v1 와 동일 식)
# =============================================================================

def compute_adjusted_scores(rd4ad_raw, boundary_like_ratio_mean):
    if rd4ad_raw is None:
        return {}
    blr = boundary_like_ratio_mean if (boundary_like_ratio_mean is not None and
                                       isinstance(boundary_like_ratio_mean, float) and
                                       math.isfinite(boundary_like_ratio_mean)) else 0.0
    roi_mean = 1.0 - blr
    return {
        "P1_times_roi_mean":      rd4ad_raw * roi_mean,
        "P2_times_sqrt_roi_mean": rd4ad_raw * math.sqrt(max(roi_mean, 0.0)),
        "P3_soft_alpha_0_2":      rd4ad_raw * (1.0 - 0.2 * blr),
        "P4_soft_alpha_0_3":      rd4ad_raw * (1.0 - 0.3 * blr),
    }


# =============================================================================
# 공통: candidate manifest -> cand_map / group_stats_map rebuild
# =============================================================================

def build_cand_and_groups():
    """candidate manifest(stage1_dev) 로 cand_map / group_stats_map / group_id_map 재현."""
    all_rows = read_csv(CANDIDATE_MANIFEST_CSV)
    stage1_rows = [r for r in all_rows if r.get("stage_split") == "stage1_dev"]

    roi_lookup = RoiRatioLookup()
    cand_map = {}
    for r in stage1_rows:
        cid = r["candidate_id"]
        y0 = int(r["crop_y0"]); x0 = int(r["crop_x0"])
        y1 = int(r["crop_y1"]); x1 = int(r["crop_x1"])
        local_z = int(r["local_z"])
        try:
            fss = float(r["first_stage_score"])
        except Exception:
            fss = 0.0
        src_csv = r.get("source_score_csv", "")
        roi_mean, _roi_min, pos_bin = roi_lookup.lookup(
            r["patient_id"], src_csv, local_z, y0, x0, y1, x1
        )
        cand_map[cid] = {
            "candidate_id":        cid,
            "patient_id":          r["patient_id"],
            "safe_id":             r.get("safe_id", ""),
            "local_z":             local_z,
            "crop_y0":             y0,
            "crop_x0":             x0,
            "crop_y1":             y1,
            "crop_x1":             x1,
            "y_center":            (y0 + y1) / 2.0,
            "x_center":            (x0 + x1) / 2.0,
            "first_stage_score":   fss,
            "label":               r.get("label", ""),
            "roi_0_0_patch_ratio": roi_mean,
            "position_bin":        pos_bin or "",
            "source_score_csv":    src_csv,
        }

    candidates = list(cand_map.values())
    group_id_map, group_candidates = build_groups(candidates, DEFAULT_Z_GAP, DEFAULT_XY_RADIUS)

    group_stats_map = {}
    for gid, cand_ids in group_candidates.items():
        stat = compute_group_stats(gid, cand_ids, cand_map)
        if stat:
            group_stats_map[gid] = stat

    return cand_map, group_id_map, group_candidates, group_stats_map


def load_shard_plan():
    """full_scoring_shard_plan.csv -> {group_id: shard_id}, {shard_id: [group_id...]}"""
    rows = read_csv(SHARD_PLAN_CSV)
    gid_to_shard = {}
    shard_to_gids = defaultdict(list)
    for r in rows:
        gid = r["group_id"]
        sid = int(r["shard_id"])
        gid_to_shard[gid] = sid
        shard_to_gids[sid].append(gid)
    return gid_to_shard, shard_to_gids


def load_manifest_repr():
    """group_manifest_full.csv -> {group_id: representative_candidate_id}"""
    rows = read_csv(GROUP_MANIFEST_CSV)
    return {r["group_id"]: r["representative_candidate_id"] for r in rows}


# =============================================================================
# dry-run
# =============================================================================

def run_dry():
    print("=" * 70)
    print("[DRY-RUN] RD4AD z-continuity group-level full scoring RUN v1")
    print("=" * 70)
    issues = []

    print("\n[1] 입력 파일 존재 확인 (read-only)")
    checks = {
        "candidate manifest":     CANDIDATE_MANIFEST_CSV,
        "RD-D1s scalar score":    RD_D1S_SCORE_CSV,
        "checkpoint":             CKPT_PATH,
        "ResNet18 weight":        LOCAL_RESNET_WEIGHT,
        "group_manifest_full":    GROUP_MANIFEST_CSV,
        "group_repr_full":        GROUP_REPR_MANIFEST_CSV,
        "shard_plan":             SHARD_PLAN_CSV,
        "CT root":                CT_ROOT,
        "effb0 score base":       EFFB0_SCORE_BASE,
    }
    for name, p in checks.items():
        ok = p.exists()
        print(f"  [{'OK' if ok else 'MISSING'}] {name}: {p}")
        if not ok:
            issues.append(f"missing: {name}")

    print("\n[2] shard plan 별 expected group count")
    shard_summary = {}
    try:
        gid_to_shard, shard_to_gids = load_shard_plan()
        total = 0
        for sid in range(SHARD_COUNT):
            c = len(shard_to_gids.get(sid, []))
            shard_summary[sid] = c
            total += c
            print(f"  shard {sid}: {c:,} groups")
        print(f"  total: {total:,} groups")
        if total != len(gid_to_shard):
            issues.append("shard plan total mismatch")
    except Exception as e:
        issues.append(f"shard plan 읽기 실패: {e}")

    print("\n[3] output overwrite 위험 확인")
    for sid in range(SHARD_COUNT):
        out_csv = SHARDS_DIR / f"shard_{sid}" / f"group_scores_shard_{sid}.csv"
        done    = SHARDS_DIR / f"shard_{sid}" / "DONE.json"
        if out_csv.exists() or done.exists():
            print(f"  [WARN] shard {sid} 출력 이미 존재 (재실행 시 overwrite)")
        else:
            print(f"  [OK] shard {sid} 출력 없음")

    print("\n[4] stage2_holdout 경로 접근 없음 확인")
    print(f"  stage2_holdout_accessed: {GUARDRAILS['stage2_holdout_accessed']}")

    print("\n[5] guardrail (dry-run: model forward/checkpoint load/파일생성 없음)")
    for k in ["checkpoint_loaded", "model_forward_executed", "crop_generation_executed",
              "training_executed", "backward_executed", "optimizer_created",
              "checkpoint_saved", "existing_artifact_modified", "output_overwrite"]:
        print(f"  {k}: {GUARDRAILS[k]}")

    print("\n" + "=" * 70)
    if issues:
        print("[DRY-RUN 결과] 이슈 발견:")
        for it in issues:
            print(f"  - {it}")
        print("판정: NEEDS_FIX")
    else:
        print("[DRY-RUN 결과] 입력/계획/경로 OK. shard 실행 준비됨.")
        print("판정: READY_TO_RUN_SHARD")
    print("=" * 70)


# =============================================================================
# run-shard
# =============================================================================

SHARD_CSV_FIELDS = [
    "shard_id", "group_id", "patient_id", "representative_candidate_id", "safe_id",
    "local_z", "crop_y0", "crop_x0", "crop_y1", "crop_x1",
    "n_candidates", "has_positive", "positive_count", "hard_negative_count",
    "z_min", "z_max", "z_span",
    "first_stage_score_max", "first_stage_score_mean",
    "roi_0_0_patch_ratio_mean", "boundary_like_ratio_mean",
    "rd4ad_group_score_raw", "score_layer1", "score_layer2", "score_layer3",
    "scalar_repro_diff",
    "P1_times_roi_mean", "P2_times_sqrt_roi_mean", "P3_soft_alpha_0_2", "P4_soft_alpha_0_3",
]


def run_shard(shard_id: int):
    print("=" * 70)
    print(f"[RUN-SHARD] RD4AD z-continuity group-level full scoring — shard {shard_id}")
    print("=" * 70)
    t0 = time.perf_counter()

    shard_dir = SHARDS_DIR / f"shard_{shard_id}"
    out_csv     = shard_dir / f"group_scores_shard_{shard_id}.csv"
    summary_json = shard_dir / f"shard_{shard_id}_summary.json"
    error_csv   = shard_dir / "errors.csv"
    done_json   = shard_dir / "DONE.json"

    output_existed = out_csv.exists() or done_json.exists()
    GUARDRAILS["output_overwrite"] = bool(output_existed)
    if output_existed:
        print(f"  [WARN] shard {shard_id} 출력 이미 존재 → overwrite (output_overwrite=True)")

    shard_dir.mkdir(parents=True, exist_ok=True)
    append_error = make_error_logger(error_csv)

    # 1. candidate -> group rebuild
    print("\n[1] candidate manifest + group rebuild")
    cand_map, group_id_map, group_candidates, group_stats_map = build_cand_and_groups()
    print(f"  candidates: {len(cand_map):,}  groups(rebuild): {len(group_stats_map):,}")

    # 2. shard plan / manifest 일관성 검증
    print("\n[2] shard plan / manifest 일관성 검증")
    gid_to_shard, shard_to_gids = load_shard_plan()
    manifest_repr = load_manifest_repr()

    rebuilt_gids  = set(group_stats_map.keys())
    plan_gids     = set(gid_to_shard.keys())
    manifest_gids = set(manifest_repr.keys())

    consistency_ok = True
    if rebuilt_gids != plan_gids:
        consistency_ok = False
        only_rebuild = list(rebuilt_gids - plan_gids)[:5]
        only_plan    = list(plan_gids - rebuilt_gids)[:5]
        append_error(f"group_id set mismatch rebuild vs plan. rebuild_only={only_rebuild} plan_only={only_plan}")
        print(f"  [FAIL] rebuild vs shard_plan group_id 집합 불일치")
    if rebuilt_gids != manifest_gids:
        consistency_ok = False
        append_error("group_id set mismatch rebuild vs group_manifest_full")
        print(f"  [FAIL] rebuild vs group_manifest_full group_id 집합 불일치")

    if not consistency_ok:
        print("  [ABORT] group_id 일관성 실패 — scoring 중단")
        _write_shard_summary(summary_json, shard_id,
                             expected=len(shard_to_gids.get(shard_id, [])),
                             scored=0, failed=0, errors=1,
                             nan=0, inf=0, runtime=time.perf_counter() - t0,
                             verdict="FAIL", consistency_ok=False)
        _write_done(done_json, "FAIL", shard_id)
        print("판정: FAIL (group_id 일관성)")
        sys.exit(1)
    print("  [OK] rebuild == shard_plan == group_manifest_full (group_id 집합 일치)")

    # 3. 대표 candidate 일관성(샘플) 검증
    repr_mismatch = 0
    for gid in list(shard_to_gids.get(shard_id, []))[:200]:
        rebuilt_repr = group_stats_map[gid]["representative_candidate_id"]
        if manifest_repr.get(gid) != rebuilt_repr:
            repr_mismatch += 1
    print(f"  representative 일치(샘플 200): mismatch={repr_mismatch}")
    if repr_mismatch > 0:
        append_error(f"representative mismatch sample count={repr_mismatch}")

    # 4. RD-D1s scalar score CSV (scalar_repro_diff)
    print("\n[4] RD-D1s scalar score CSV 로드")
    scalar_score_map = {}
    if RD_D1S_SCORE_CSV.exists():
        for r in read_csv(RD_D1S_SCORE_CSV):
            if r.get("stage_split", "") == "stage1_dev":
                try:
                    scalar_score_map[r["candidate_id"]] = float(r["rd_d1s_medi3ch_rd4ad_score"])
                except Exception:
                    pass
        print(f"  scalar score map: {len(scalar_score_map):,} entries")
    else:
        print(f"  [WARN] scalar score CSV 없음")

    # 5. shard 대상 group
    shard_gids = shard_to_gids.get(shard_id, [])
    expected_group_count = len(shard_gids)
    print(f"\n[5] shard {shard_id} 대상 groups: {expected_group_count:,}")
    if expected_group_count == 0:
        print(f"  [ABORT] shard {shard_id} 에 group 없음")
        sys.exit(2)

    # 6. 모델 로드
    print("\n[6] 모델 로드 (checkpoint read-only)")
    import torch
    import numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    teacher, student = load_model_from_checkpoint(device)
    teacher.eval()
    student.eval()

    teacher_features = {}
    for layer_name, module in [("layer1", teacher.layer1),
                               ("layer2", teacher.layer2),
                               ("layer3", teacher.layer3)]:
        def _hook(module, inp, output, _n=layer_name):
            teacher_features[_n] = output
        module.register_forward_hook(_hook)

    teacher.train = _forbidden_train
    student.train = _forbidden_train

    ct_cache = CTMmapCache(max_size=10)

    # 7. forward scoring
    print(f"\n[7] forward scoring (shard {shard_id})")
    GUARDRAILS["model_forward_executed"] = True
    GUARDRAILS["crop_generation_executed"] = True
    GUARDRAILS["full_scoring_executed"] = True  # shard 단위

    score_rows = []
    scalar_repro_diffs = []
    error_count = 0
    failed_group_count = 0
    score_nan_count = 0
    score_inf_count = 0

    for idx, gid in enumerate(shard_gids):
        gstat = group_stats_map.get(gid)
        if not gstat:
            append_error(f"gid {gid} not in group_stats_map")
            error_count += 1
            failed_group_count += 1
            continue
        repr_cid = gstat["representative_candidate_id"]
        c = cand_map.get(repr_cid)
        if not c:
            append_error(f"repr_cid {repr_cid} not in cand_map (gid={gid})")
            error_count += 1
            failed_group_count += 1
            continue

        try:
            ct_arr = ct_cache.get(c["safe_id"])
        except Exception as e:
            append_error(f"CT load fail: {c['safe_id']} (gid={gid}): {e}", e)
            error_count += 1
            failed_group_count += 1
            continue

        try:
            crop = build_medi3ch_crop(ct_arr, c["local_z"],
                                      c["crop_y0"], c["crop_x0"], c["crop_y1"], c["crop_x1"])
        except Exception as e:
            append_error(f"crop build fail: {repr_cid} (gid={gid}): {e}", e)
            error_count += 1
            failed_group_count += 1
            continue

        crop_t = torch.from_numpy(crop[np.newaxis]).to(device)
        try:
            with torch.no_grad():
                rd4ad_raw, l1, l2, l3 = compute_rd4ad_score(
                    teacher, student, crop_t, teacher_features, device)
        except Exception as e:
            append_error(f"forward fail: {repr_cid} (gid={gid}): {e}", e)
            error_count += 1
            failed_group_count += 1
            continue

        if math.isnan(rd4ad_raw):
            score_nan_count += 1
        if math.isinf(rd4ad_raw):
            score_inf_count += 1

        scalar_repro_diff = None
        if repr_cid in scalar_score_map:
            scalar_repro_diff = abs(rd4ad_raw - scalar_score_map[repr_cid])
            scalar_repro_diffs.append(scalar_repro_diff)

        blr = gstat.get("boundary_like_ratio_mean")
        try:
            blr = float(blr)
        except Exception:
            blr = None
        adj = compute_adjusted_scores(rd4ad_raw, blr)

        row = {
            "shard_id":                    shard_id,
            "group_id":                    gid,
            "patient_id":                  gstat["patient_id"],
            "representative_candidate_id": repr_cid,
            "safe_id":                     c["safe_id"],
            "local_z":                     c["local_z"],
            "crop_y0":                     c["crop_y0"],
            "crop_x0":                     c["crop_x0"],
            "crop_y1":                     c["crop_y1"],
            "crop_x1":                     c["crop_x1"],
            "n_candidates":                gstat["n_candidates"],
            "has_positive":                gstat["has_positive"],
            "positive_count":              gstat["positive_count"],
            "hard_negative_count":         gstat["hard_negative_count"],
            "z_min":                       gstat["z_min"],
            "z_max":                       gstat["z_max"],
            "z_span":                      gstat["z_span"],
            "first_stage_score_max":       gstat["first_stage_score_max"],
            "first_stage_score_mean":      gstat["first_stage_score_mean"],
            "roi_0_0_patch_ratio_mean":    gstat.get("roi_0_0_patch_ratio_mean", ""),
            "boundary_like_ratio_mean":    gstat.get("boundary_like_ratio_mean", ""),
            "rd4ad_group_score_raw":       round(rd4ad_raw, 6),
            "score_layer1":                round(l1, 6),
            "score_layer2":                round(l2, 6),
            "score_layer3":                round(l3, 6),
            "scalar_repro_diff":           round(scalar_repro_diff, 8) if scalar_repro_diff is not None else "",
        }
        for pk, pv in adj.items():
            row[pk] = round(pv, 6)
        score_rows.append(row)

        if (idx + 1) % 500 == 0:
            print(f"  {idx+1}/{expected_group_count}: scored={len(score_rows)} failed={failed_group_count}")

    actual_scored_group_count = len(score_rows)
    print(f"  scored={actual_scored_group_count}  failed={failed_group_count}  errors={error_count}")

    # 8. scalar repro 요약
    scalar_repro_ok = None
    scalar_repro_mean_diff = None
    if scalar_repro_diffs:
        scalar_repro_mean_diff = sum(scalar_repro_diffs) / len(scalar_repro_diffs)
        scalar_repro_ok = scalar_repro_mean_diff < 1e-4
        print(f"  scalar_repro: n={len(scalar_repro_diffs)} mean_abs_diff={scalar_repro_mean_diff:.2e} ok={scalar_repro_ok}")

    # 9. CSV 저장
    print("\n[8] group_scores CSV 저장")
    write_csv(out_csv, SHARD_CSV_FIELDS, score_rows)

    # 10. summary / DONE
    verdict = "PASS" if failed_group_count == 0 else "PARTIAL"
    runtime = time.perf_counter() - t0
    _write_shard_summary(summary_json, shard_id,
                         expected=expected_group_count,
                         scored=actual_scored_group_count,
                         failed=failed_group_count,
                         errors=error_count,
                         nan=score_nan_count, inf=score_inf_count,
                         runtime=runtime, verdict=verdict, consistency_ok=True,
                         scalar_repro_ok=scalar_repro_ok,
                         scalar_repro_mean_diff=scalar_repro_mean_diff,
                         repr_mismatch=repr_mismatch)
    _write_done(done_json, verdict, shard_id)

    # 11. 최종 보고
    print("\n" + "=" * 70)
    print(f"[RUN-SHARD {shard_id}] 완료 ({runtime:.1f}s)")
    print(f"판정: {verdict}")
    print(f"  expected groups : {expected_group_count}")
    print(f"  scored groups   : {actual_scored_group_count}")
    print(f"  failed groups   : {failed_group_count}")
    print(f"  error count     : {error_count}")
    print(f"  score NaN/Inf   : {score_nan_count}/{score_inf_count}")
    print(f"  scalar_repro_ok : {scalar_repro_ok}")
    print(f"  stage2_holdout_accessed   : {GUARDRAILS['stage2_holdout_accessed']}")
    print(f"  existing_artifact_modified: {GUARDRAILS['existing_artifact_modified']}")
    print(f"  output_overwrite          : {GUARDRAILS['output_overwrite']}")
    print("=" * 70)


def _write_shard_summary(path, shard_id, expected, scored, failed, errors,
                         nan, inf, runtime, verdict, consistency_ok,
                         scalar_repro_ok=None, scalar_repro_mean_diff=None, repr_mismatch=None):
    ensure_output_path_safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "verdict":                    verdict,
        "shard_id":                   shard_id,
        "expected_group_count":       expected,
        "actual_scored_group_count":  scored,
        "failed_group_count":         failed,
        "error_count":                errors,
        "score_nan_count":            nan,
        "score_inf_count":            inf,
        "runtime_sec":                round(runtime, 1),
        "group_id_consistency_ok":    consistency_ok,
        "representative_mismatch_sample": repr_mismatch,
        "scalar_repro_ok":            scalar_repro_ok,
        "scalar_repro_mean_abs_diff": round(scalar_repro_mean_diff, 8) if scalar_repro_mean_diff is not None else None,
        # guardrails
        "stage2_holdout_accessed":    GUARDRAILS["stage2_holdout_accessed"],
        "checkpoint_loaded":          GUARDRAILS["checkpoint_loaded"],
        "model_forward_executed":     GUARDRAILS["model_forward_executed"],
        "training_executed":          GUARDRAILS["training_executed"],
        "backward_executed":          GUARDRAILS["backward_executed"],
        "optimizer_created":          GUARDRAILS["optimizer_created"],
        "checkpoint_saved":           GUARDRAILS["checkpoint_saved"],
        "crop_generation_executed":   "in_memory_only" if GUARDRAILS["crop_generation_executed"] else False,
        "full_scoring_executed":      "shard_only" if GUARDRAILS["full_scoring_executed"] else False,
        "threshold_recalculated":     GUARDRAILS["threshold_recalculated"],
        "existing_artifact_modified": GUARDRAILS["existing_artifact_modified"],
        "existing_script_modified":   GUARDRAILS["existing_script_modified"],
        "output_overwrite":           GUARDRAILS["output_overwrite"],
        "raw_rd4ad_primary_score":    GUARDRAILS["raw_rd4ad_primary_score"],
        "adjusted_score_preview_only": GUARDRAILS["adjusted_score_preview_only"],
    }
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  saved: {path}")


def _write_done(path, verdict, shard_id):
    ensure_output_path_safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, "shard_id": shard_id,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
    print(f"  saved: {path}")


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RD4AD z-continuity group-level full scoring RUN v1")
    parser.add_argument("--dry-run",                action="store_true")
    parser.add_argument("--run-shard",              action="store_true")
    parser.add_argument("--shard-id",               type=int, default=None)
    parser.add_argument("--confirm-model-forward",  action="store_true")
    parser.add_argument("--confirm-stage1dev-only", action="store_true")
    args = parser.parse_args()

    if not any([args.dry_run, args.run_shard]):
        print("[ABORT] bare run 차단. --dry-run / --run-shard 사용.", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        run_dry()
        return

    if args.run_shard:
        if args.shard_id is None or args.shard_id not in range(SHARD_COUNT):
            print(f"[ABORT] --shard-id {{0..{SHARD_COUNT-1}}} 필요", file=sys.stderr)
            sys.exit(2)
        if not args.confirm_model_forward or not args.confirm_stage1dev_only:
            print("[ABORT] --confirm-model-forward --confirm-stage1dev-only 필요", file=sys.stderr)
            sys.exit(2)
        run_shard(args.shard_id)
        return


if __name__ == "__main__":
    main()
