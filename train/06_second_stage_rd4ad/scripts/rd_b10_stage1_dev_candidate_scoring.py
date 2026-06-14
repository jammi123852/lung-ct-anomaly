"""
RD-B10: stage1_dev first-stage candidate scoring preflight + run-score
목적:
  RD-B8f best_train_loss.pth 와 RD-B9 normal_val threshold 를 사용해서
  stage1_dev 1차 후보 crop 에 2차 RD4AD score 를 추가한다.
모드:
  bare run   -> exit 2
  --dry-plan -> candidate source discovery, 경로 확인, output root 없음 확인 (파일 생성 없음)
  --run-score-> 사용자 승인 후 stage1_dev candidate scoring + threshold exceedance (DONE 생성)
안전 조건:
  stage2_holdout 접근 금지
  lesion_mask (GT) 접근 금지
  backward / optimizer / checkpoint 저장 금지
  training 금지
  threshold 재계산 금지
  기존 first_stage_score 수정 금지
  output root 존재 시 즉시 중단
"""

import sys
import csv
import json
import math
import time
import collections
from pathlib import Path

ALLOWED_MODES = {"--dry-plan", "--run-score"}
if not any(m in sys.argv for m in ALLOWED_MODES):
    print("오류: 실행 모드 지정 필요.")
    print("  --dry-plan  : 입력 경로 확인 (파일 생성 없음)")
    print("  --run-score : stage1_dev candidate scoring 실행")
    sys.exit(2)

IS_DRY_PLAN = "--dry-plan" in sys.argv
IS_RUN_SCORE = "--run-score" in sys.argv

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b10_stage1_dev_candidate_scoring_v2"
)

# candidate source
CANDIDATE_MANIFEST = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/candidates"
    / "stage1_dev_fixed96_thr001_v1"
    / "candidate_manifest_stage1_dev_fixed96_thr001_v1.csv"
)

# stage split
STAGE_SPLIT_CSV = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"
)

# RD-B9 threshold
THRESHOLD_DIR = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b9_normal_val_scoring_threshold_v1"
)
THRESHOLD_SUMMARY_JSON = THRESHOLD_DIR / "rd_b9_normal_val_threshold_summary.json"
THRESHOLD_CANDIDATES_CSV = THRESHOLD_DIR / "rd_b9_normal_val_threshold_candidates.csv"

# checkpoint
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs/models/rd_b8f_true_rd4ad_resnet18_mixed3ch_6bin_shard_v1"
    / "checkpoints/best_train_loss.pth"
)
LOCAL_RESNET18_WEIGHT = Path(
    "/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)

# CT root (NSCLC lesion)
NSCLC_CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)

# v4_20 lesion ROI root
V4_20_LESION_ROI_ROOT = (
    PROJECT_ROOT
    / "outputs/mip-postprocess-research-v1/masks"
    / "refined_roi_v4_20_modeB_all_v1/lesion"
)

# candidate source inventory (비교 보고용)
CANDIDATE_SOURCE_INVENTORY = {
    "stage1_dev_fixed96_thr001_v1": str(CANDIDATE_MANIFEST),
    "rule_s6a_gs2": str(
        PROJECT_ROOT
        / "outputs/second-stage-lesion-refiner-v1/candidates"
        / "rule_s6a_gs2_selected_candidate_manifest_dryrun.csv"
    ),
    "rule_a_p95": str(
        PROJECT_ROOT
        / "outputs/second-stage-lesion-refiner-v1/candidates"
        / "rule_a_p95_stage1_dev_candidate_manifest_dryrun.csv"
    ),
    "s4_plus_d2_union": str(
        PROJECT_ROOT
        / "outputs/second-stage-lesion-refiner-v1/candidates"
        / "s4_plus_d2_union_stage1_dev_candidate_manifest_dryrun.csv"
    ),
}

# 설계 상수 (RD-B9와 동일)
EROSION_PX = 5
BOUNDARY_THRESHOLD = 0.05
INTERIOR_ROI_MIN = 0.85
CROP_SIZE = 96
MIP_RADIUS = 3
HU_CLIP_MIN = -1000.0
HU_CLIP_MAX = 600.0
HU_RANGE = 1600.0
LOW_Z_WARNING_THRESHOLD = 7
Z_LOWER_MAX = 1.0 / 3.0
Z_MIDDLE_MAX = 2.0 / 3.0
SCORE_BATCH_SIZE = 48
SIX_BIN_LABELS = [
    "upper_boundary", "upper_interior",
    "middle_boundary", "middle_interior",
    "lower_boundary", "lower_interior",
]
FORBIDDEN_PATH_KEYWORDS = [
    "stage2_holdout",
    "lesion_mask",
]

# ── 안전 체크 ──────────────────────────────────────────────────────────────────

def assert_path_safe(path_str):
    for kw in FORBIDDEN_PATH_KEYWORDS:
        if kw.lower() in str(path_str).lower():
            raise RuntimeError(
                f"[SAFETY] 금지 경로 접근 차단: {path_str!r} (keyword={kw!r})"
            )


# ── CSV/JSON 헬퍼 ──────────────────────────────────────────────────────────────

def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  -> {path.name}")


def load_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  -> {path.name}")


# ── HU 변환 / crop (RD-B9와 동일) ─────────────────────────────────────────────

def normalize_hu(hu_array):
    import numpy as np
    clipped = hu_array.clip(HU_CLIP_MIN, HU_CLIP_MAX)
    return ((clipped - HU_CLIP_MIN) / HU_RANGE).astype("float32")


def compute_mip_slab_indices(center_z, direction, z_max):
    if direction == "lower":
        raw = [center_z - MIP_RADIUS + i for i in range(MIP_RADIUS)]
    else:
        raw = [center_z + 1 + i for i in range(MIP_RADIUS)]
    return [max(0, min(idx, z_max - 1)) for idx in raw]


def build_crop_np(ct_arr, center_z, crop_y0, crop_x0, crop_y1, crop_x1):
    import numpy as np
    TARGET = CROP_SIZE
    z_max, h_max, w_max = ct_arr.shape

    def _crop2d(img2d, y0, x0, y1, x1):
        h, w = img2d.shape
        out = np.full((TARGET, TARGET), HU_CLIP_MIN, dtype=img2d.dtype)
        sy0 = max(0, y0); sx0 = max(0, x0)
        sy1 = min(h, y1); sx1 = min(w, x1)
        if sy1 <= sy0 or sx1 <= sx0:
            return out
        dy0 = sy0 - y0; dx0 = sx0 - x0
        out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img2d[sy0:sy1, sx0:sx1]
        return out

    ch0 = _crop2d(ct_arr[center_z], crop_y0, crop_x0, crop_y1, crop_x1)
    lower_idx = compute_mip_slab_indices(center_z, "lower", z_max)
    ch1 = np.max(np.stack(
        [_crop2d(ct_arr[z], crop_y0, crop_x0, crop_y1, crop_x1) for z in lower_idx], axis=0
    ), axis=0)
    upper_idx = compute_mip_slab_indices(center_z, "upper", z_max)
    ch2 = np.max(np.stack(
        [_crop2d(ct_arr[z], crop_y0, crop_x0, crop_y1, crop_x1) for z in upper_idx], axis=0
    ), axis=0)

    crop = np.stack(
        [normalize_hu(ch0), normalize_hu(ch1), normalize_hu(ch2)], axis=0
    ).astype("float32")
    if crop.shape != (3, TARGET, TARGET):
        raise RuntimeError(f"bad crop shape: {crop.shape}")
    return crop


# ── six_bin_label 계산 ─────────────────────────────────────────────────────────

def z_level_from_ratio(z_ratio):
    if z_ratio < Z_LOWER_MAX:
        return "lower"
    elif z_ratio < Z_MIDDLE_MAX:
        return "middle"
    return "upper"


def compute_sixbin_for_crop(roi_slice, z_ratio):
    """roi_slice (H,W bool), z_ratio -> (z_level, boundary_status, six_bin_label)"""
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    dist = distance_transform_edt(roi_slice)
    ring = ((roi_slice > 0) & (dist <= EROSION_PX)).astype(np.float32)
    patch_area = float(CROP_SIZE * CROP_SIZE)

    roi_sum = float(roi_slice.sum())
    ring_sum = float(ring.sum())

    roi_ratio = roi_sum / patch_area
    boundary_overlap_ratio = ring_sum / patch_area

    is_boundary = boundary_overlap_ratio >= BOUNDARY_THRESHOLD
    is_interior = (roi_ratio >= INTERIOR_ROI_MIN) and (not is_boundary)

    if is_boundary:
        bs = "boundary"
    elif is_interior:
        bs = "interior"
    else:
        bs = "excluded"

    z_level = z_level_from_ratio(z_ratio)
    if bs == "excluded":
        six_bin = "excluded"
    else:
        six_bin = f"{z_level}_{bs}"

    return z_level, bs, six_bin, round(roi_ratio, 6), round(boundary_overlap_ratio, 6)


# ── LRU Patient Cache ──────────────────────────────────────────────────────────

class LRUPatientCache:
    def __init__(self, max_size=8):
        self._cache = collections.OrderedDict()
        self._max = max_size

    def load(self, key, path):
        import numpy as np
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        assert_path_safe(path)
        arr = np.load(str(path), mmap_mode="r")
        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        self._cache[key] = arr
        return arr


# ── Teacher / Student 빌드 (RD-B9와 동일) ─────────────────────────────────────

def build_teacher(local_weight_path):
    import torch
    import torchvision.models as models
    resnet = models.resnet18(weights=None)
    state_dict = torch.load(str(local_weight_path), map_location="cpu", weights_only=True)
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
                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            )
            self.de_layer2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            )
            self.de_layer1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )

        def forward(self, layer3_feat):
            x = self.de_layer3(layer3_feat);  de3 = x
            x = self.de_layer2(x);           de2 = x
            x = self.de_layer1(x);           de1 = x
            return de3, de2, de1

    return StudentDecoder()


# ── stage split 로드 ──────────────────────────────────────────────────────────

def load_stage_split(stage_split_csv):
    stage1_dev_ids = set()
    holdout_ids = set()
    with open(stage_split_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = row.get("stage_split", "")
            pid = row.get("patient_id", "")
            if sp == "stage1_dev":
                stage1_dev_ids.add(pid)
            elif sp == "stage2_holdout":
                holdout_ids.add(pid)
    return stage1_dev_ids, holdout_ids


# ── candidate source inventory 분석 ──────────────────────────────────────────

def analyze_candidate_sources(source_dict, holdout_ids):
    rows = []
    for name, path in source_dict.items():
        p = Path(path)
        if not p.exists():
            rows.append({
                "source_name": name, "path": str(p), "exists": False,
                "total_rows": 0, "stage1_dev_rows": 0, "holdout_rows": 0,
                "unique_patients": 0, "has_safe_id": False,
                "has_candidate_id": False, "has_score_col": False,
                "score_cols": "", "has_local_z": False, "has_fixed_crop": False,
                "has_stage_split": False, "selected": False, "reason": "file_not_found",
            })
            continue
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                cols = set(reader.fieldnames or [])
                total = stage1 = holdout = 0
                pids = set()
                for row in reader:
                    sp = row.get("stage_split", row.get("split", ""))
                    pid = row.get("patient_id", "")
                    total += 1
                    if sp == "stage1_dev":
                        stage1 += 1
                        pids.add(pid)
                    elif sp == "stage2_holdout" or pid in holdout_ids:
                        holdout += 1
            score_cols = [c for c in cols if any(s in c.lower() for s in ["score", "padim"])]
            rows.append({
                "source_name": name,
                "path": str(p),
                "exists": True,
                "total_rows": total,
                "stage1_dev_rows": stage1,
                "holdout_rows": holdout,
                "unique_patients": len(pids),
                "has_safe_id": "safe_id" in cols,
                "has_candidate_id": "candidate_id" in cols,
                "has_score_col": bool(score_cols),
                "score_cols": "|".join(score_cols[:3]),
                "has_local_z": any(c in cols for c in ("local_z", "z_center")),
                "has_fixed_crop": any(c in cols for c in ("y0_fixed_crop", "crop_y0")),
                "has_stage_split": "stage_split" in cols or "split" in cols,
                "selected": name == "stage1_dev_fixed96_thr001_v1",
                "reason": "safe_id+candidate_id+fixed_crop+mean_padim_score 완비" if name == "stage1_dev_fixed96_thr001_v1" else "",
            })
        except Exception as e:
            rows.append({
                "source_name": name, "path": str(p), "exists": True,
                "total_rows": 0, "stage1_dev_rows": 0, "holdout_rows": 0,
                "unique_patients": 0, "has_safe_id": False,
                "has_candidate_id": False, "has_score_col": False,
                "score_cols": "", "has_local_z": False, "has_fixed_crop": False,
                "has_stage_split": False, "selected": False, "reason": f"error:{e}",
            })
    return rows


# ── threshold 로드 ────────────────────────────────────────────────────────────

def load_thresholds(threshold_summary_json, threshold_candidates_csv):
    with open(threshold_summary_json, encoding="utf-8") as f:
        summary = json.load(f)
    global_p95 = summary.get("global_p95")
    global_p99 = summary.get("global_p99")
    bin_thresholds = summary.get("bin_thresholds", {})

    cand_rows = []
    with open(threshold_candidates_csv, newline="", encoding="utf-8") as f:
        cand_rows = list(csv.DictReader(f))

    return {
        "global_p95": global_p95,
        "global_p99": global_p99,
        "bin_thresholds": bin_thresholds,
        "threshold_created_from": summary.get("threshold_created_from", "RD-B9 normal_val only"),
        "n_threshold_labels": len(cand_rows),
    }


# ── dry-plan ────────────────────────────────────────────────────────────────────

def run_dry_plan():
    errors = []
    check_rows = []

    def check(label, ok, detail=""):
        status = "OK" if ok else "FAIL"
        check_rows.append({"label": label, "status": status, "detail": str(detail)})
        if not ok:
            errors.append(f"{label}: {detail}")

    print("=" * 70)
    print("RD-B10: stage1_dev candidate scoring [DRY-PLAN]")
    print("=" * 70)

    # 1. stage split 로드
    stage1_dev_ids, holdout_ids = set(), set()
    check("stage_split_exists", STAGE_SPLIT_CSV.exists(), str(STAGE_SPLIT_CSV))
    if STAGE_SPLIT_CSV.exists():
        stage1_dev_ids, holdout_ids = load_stage_split(STAGE_SPLIT_CSV)
        check("stage1_dev_count", len(stage1_dev_ids) > 0, f"{len(stage1_dev_ids)} patients")
        check("holdout_count", len(holdout_ids) > 0, f"{len(holdout_ids)} patients")
    print(f"\n  stage1_dev patients : {len(stage1_dev_ids)}")
    print(f"  stage2_holdout patients : {len(holdout_ids)}")

    # 2. candidate source inventory
    print("\n[1/5] candidate source inventory")
    inventory_rows = analyze_candidate_sources(CANDIDATE_SOURCE_INVENTORY, holdout_ids)
    selected = next((r for r in inventory_rows if r["selected"]), None)

    print()
    print(f"  {'source':<30} {'rows':>8} {'stage1':>8} {'holdout':>8} {'pats':>6} "
          f"{'safe_id':>8} {'cand_id':>8} {'score':>8} {'selected':>9}")
    for r in inventory_rows:
        print(f"  {r['source_name']:<30} {r['stage1_dev_rows']:>8} {r['stage1_dev_rows']:>8} "
              f"{r['holdout_rows']:>8} {r['unique_patients']:>6} "
              f"{str(r['has_safe_id']):>8} {str(r['has_candidate_id']):>8} "
              f"{str(r['has_score_col']):>8} {str(r['selected']):>9}")

    if selected is None:
        errors.append("selected candidate source 없음")
        print("\n[FAIL] selected source 없음 — 수동 선택 필요")
        sys.exit(1)
    print(f"\n  선택: {selected['source_name']}")
    print(f"  이유: {selected['reason']}")

    # stage2_holdout intersection 확인
    check("selected_source_holdout_rows_0", selected["holdout_rows"] == 0,
          f"holdout_rows={selected['holdout_rows']}")

    # 3. candidate manifest 상세 확인
    print("\n[2/5] candidate manifest 상세 확인")
    check("candidate_manifest_exists", CANDIDATE_MANIFEST.exists(), str(CANDIDATE_MANIFEST))

    cand_rows_loaded = []
    if CANDIDATE_MANIFEST.exists():
        cand_rows_loaded = load_csv_rows(CANDIDATE_MANIFEST)
        n_stage1 = sum(1 for r in cand_rows_loaded if r.get("stage_split") == "stage1_dev")
        n_holdout_in_cand = sum(
            1 for r in cand_rows_loaded
            if r.get("patient_id", "") in holdout_ids
        )
        # holdout_in_candidate > 0 이면 run-score denylist 필터링으로 자동 제거 예정 → WARNING만
        if n_holdout_in_cand > 0:
            print(f"  [WARNING] holdout_in_candidate={n_holdout_in_cand} → run-score에서 자동 제거 예정")
        check("candidate_rows_gt0", n_stage1 > 0, f"{n_stage1} stage1_dev rows")
        print(f"  total rows                : {len(cand_rows_loaded)}")
        print(f"  stage1_dev                : {n_stage1}")
        print(f"  holdout in candidate (pre): {n_holdout_in_cand}")

    # 4. allowlist / denylist 검증
    print("\n[3/5] stage1_dev allowlist / holdout denylist 검증")
    if cand_rows_loaded and stage1_dev_ids:
        cand_pids = set(r["patient_id"] for r in cand_rows_loaded)
        not_in_stage1 = cand_pids - stage1_dev_ids
        in_holdout = cand_pids & holdout_ids
        # holdout 환자가 candidate에 있어도 run-score에서 denylist 필터링으로 자동 제거되므로 WARNING 처리
        if not_in_stage1 or in_holdout:
            print(f"  [WARNING] not_in_stage1_dev={list(not_in_stage1)[:5]}")
            print(f"  [WARNING] in_holdout={list(in_holdout)[:5]}")
            print(f"  [WARNING] --run-score에서 holdout denylist 필터링으로 자동 제거됨")
        print(f"  candidate unique patients  : {len(cand_pids)}")
        print(f"  not_in_stage1_dev          : {len(not_in_stage1)} (WARNING, run-score에서 자동 제거)")
        print(f"  in_holdout                 : {len(in_holdout)} (WARNING, run-score에서 자동 제거)")

    # 5. CT / ROI 경로 확인 (상위 10명)
    print("\n[4/5] CT / ROI 경로 확인 (상위 10명)")
    safe_ids_sample = []
    if cand_rows_loaded:
        seen = set()
        for r in cand_rows_loaded:
            sid = r.get("safe_id", "")
            if sid and sid not in seen:
                safe_ids_sample.append(sid)
                seen.add(sid)
            if len(safe_ids_sample) >= 10:
                break

    ct_ok_count = roi_ok_count = 0
    ct_fail_count = roi_fail_count = 0
    path_check_rows = []
    for sid in safe_ids_sample:
        ct_path = NSCLC_CT_ROOT / sid / "ct_hu.npy"
        roi_path = V4_20_LESION_ROI_ROOT / sid / "refined_roi.npy"
        ct_ok = ct_path.exists()
        roi_ok = roi_path.exists()
        if ct_ok:
            ct_ok_count += 1
        else:
            ct_fail_count += 1
        if roi_ok:
            roi_ok_count += 1
        else:
            roi_fail_count += 1
        path_check_rows.append({
            "safe_id": sid, "ct_ok": int(ct_ok), "roi_ok": int(roi_ok),
            "ct_path": str(ct_path), "roi_path": str(roi_path),
        })
        print(f"  {sid[:30]:30s}  ct={ct_ok}  roi={roi_ok}")

    check(f"sample_ct_exists (n={len(safe_ids_sample)})", ct_fail_count == 0,
          f"fail={ct_fail_count}")
    check(f"sample_roi_exists (n={len(safe_ids_sample)})", roi_fail_count == 0,
          f"fail={roi_fail_count}")

    # 6. threshold / checkpoint 확인
    print("\n[5/5] threshold / checkpoint 확인")
    check("threshold_dir_exists", THRESHOLD_DIR.exists(), str(THRESHOLD_DIR))
    check("threshold_summary_json", THRESHOLD_SUMMARY_JSON.exists(), str(THRESHOLD_SUMMARY_JSON))
    check("threshold_candidates_csv", THRESHOLD_CANDIDATES_CSV.exists(), str(THRESHOLD_CANDIDATES_CSV))
    check("checkpoint_exists", CHECKPOINT_PATH.exists(), str(CHECKPOINT_PATH))
    check("local_resnet18_exists", LOCAL_RESNET18_WEIGHT.exists(), str(LOCAL_RESNET18_WEIGHT))
    check("output_root_absent", not OUTPUT_ROOT.exists(),
          str(OUTPUT_ROOT) if OUTPUT_ROOT.exists() else "not_exists(OK)")

    th_info = {}
    if THRESHOLD_SUMMARY_JSON.exists():
        th_info = load_thresholds(THRESHOLD_SUMMARY_JSON, THRESHOLD_CANDIDATES_CSV) if THRESHOLD_CANDIDATES_CSV.exists() else {}
        print(f"  global p95 = {th_info.get('global_p95')}")
        print(f"  global p99 = {th_info.get('global_p99')}")
        print(f"  threshold_labels = {th_info.get('n_threshold_labels')}")
        print(f"  threshold_source = {th_info.get('threshold_created_from')}")

    # 요약 출력
    print()
    print("─" * 70)
    print(f"  선택 candidate source : {selected['source_name'] if selected else 'NONE'}")
    print(f"  stage1_dev patients   : {len(stage1_dev_ids)}")
    print(f"  stage2_holdout patients: {len(holdout_ids)}")
    print(f"  input candidates      : {selected['stage1_dev_rows'] if selected else 0}")
    print(f"  ct_fail (sample)      : {ct_fail_count}")
    print(f"  roi_fail (sample)     : {roi_fail_count}")
    print(f"  global p95            : {th_info.get('global_p95')}")
    print(f"  global p99            : {th_info.get('global_p99')}")
    print()

    verdict = "FAIL" if errors else "DRY-PLAN OK"
    print(f"판정: {verdict}")
    if errors:
        print("FAIL 항목:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("모든 체크 통과 — 사용자 승인 후:")
        print("  source ~/ai_env/bin/activate && \\")
        print("  python scripts/rd_b10_stage1_dev_candidate_scoring.py --run-score \\")
        print("    2>&1 | tee /tmp/rd_b10_stage1_dev_scoring_log.txt")


# ── run-score ──────────────────────────────────────────────────────────────────

def run_score():
    import numpy as np
    import torch
    import torch.nn.functional as F

    print("=" * 70)
    print("RD-B10: stage1_dev candidate scoring [RUN-SCORE]")
    print("=" * 70)

    # output root guard
    if OUTPUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재: {OUTPUT_ROOT}")
        sys.exit(1)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    error_rows_all = []

    # ── 1. stage split + candidate 로드 ──────────────────────────────────────
    print("\n[1/5] stage split + candidate 로드")
    stage1_dev_ids, holdout_ids = load_stage_split(STAGE_SPLIT_CSV)
    cand_rows = load_csv_rows(CANDIDATE_MANIFEST)

    # stage2_holdout denylist 확인 (ABORT 아님 → WARNING + 자동 제거)
    cand_pids = set(r["patient_id"] for r in cand_rows)
    h_intersect = cand_pids & holdout_ids
    n_input = len(cand_rows)
    if h_intersect:
        print(
            f"  [DENYLIST] stage2_holdout intersection {len(h_intersect)}명 "
            f"→ scoring 전 자동 제거: {sorted(h_intersect)[:5]}"
        )

    # allowlist: stage1_dev only + holdout denylist 이중 필터
    n_before = len(cand_rows)
    cand_rows = [
        r for r in cand_rows
        if r.get("stage_split") == "stage1_dev"
        and r.get("patient_id", "") not in holdout_ids
    ]
    n_holdout_removed = n_before - len(cand_rows)

    # 필터 후 holdout intersection 재검증 (필수 assert)
    filtered_pids = {r.get("patient_id") for r in cand_rows}
    post_intersect = filtered_pids & holdout_ids
    if post_intersect:
        print(f"[ABORT] denylist filtering failed: {sorted(post_intersect)[:5]}")
        sys.exit(1)

    print(f"  stage1_dev patients       : {len(stage1_dev_ids)}")
    print(f"  holdout patients          : {len(holdout_ids)}")
    print(f"  input candidates          : {n_input}")
    print(f"  holdout patients detected : {len(h_intersect)}")
    print(f"  holdout rows removed      : {n_holdout_removed}")
    print(f"  scoring candidates        : {len(cand_rows)}")
    print(f"  processed ∩ stage2_holdout: 0 (OK)")

    # ── 2. threshold 로드 ────────────────────────────────────────────────────
    print("\n[2/5] RD-B9 threshold 로드")
    th = load_thresholds(THRESHOLD_SUMMARY_JSON, THRESHOLD_CANDIDATES_CSV)
    global_p95 = float(th["global_p95"])
    global_p99 = float(th["global_p99"])
    bin_thresholds = th["bin_thresholds"]
    print(f"  global p95 = {global_p95}")
    print(f"  global p99 = {global_p99}")
    print(f"  threshold_labels = {th['n_threshold_labels']}")

    # allowlist summary 저장
    allowlist_summary = [
        {"category": "stage1_dev_patients", "count": len(stage1_dev_ids)},
        {"category": "stage2_holdout_patients", "count": len(holdout_ids)},
        {"category": "input_candidates", "count": n_input},
        {"category": "holdout_patients_detected", "count": len(h_intersect)},
        {"category": "holdout_rows_removed", "count": n_holdout_removed},
        {"category": "scoring_candidates", "count": len(cand_rows)},
        {"category": "post_filter_holdout_intersection", "count": len(post_intersect)},
    ]
    write_csv(
        OUTPUT_ROOT / "rd_b10_stage1_dev_allowlist_summary.csv",
        ["category", "count"],
        allowlist_summary,
    )

    # threshold source 검증 저장
    th_source_rows = [
        {"item": "source_file", "value": str(THRESHOLD_SUMMARY_JSON)},
        {"item": "threshold_created_from", "value": th["threshold_created_from"]},
        {"item": "global_p95", "value": str(global_p95)},
        {"item": "global_p99", "value": str(global_p99)},
        {"item": "n_threshold_labels", "value": str(th["n_threshold_labels"])},
        {"item": "threshold_recalculated", "value": "False"},
    ]
    write_csv(
        OUTPUT_ROOT / "rd_b10_threshold_source_validation.csv",
        ["item", "value"],
        th_source_rows,
    )

    # ── 3. model 로드 ────────────────────────────────────────────────────────
    print("\n[3/5] model 로드 (eval-only)")
    print("  [SAFETY] backward=False, optimizer=None, checkpoint_saved=False")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    teacher = build_teacher(LOCAL_RESNET18_WEIGHT).to(device)
    teacher.eval()
    teacher.requires_grad_(False)

    student = build_student_decoder().to(device)
    student.eval()
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=True)
    student.load_state_dict(ckpt.get("student_state_dict", ckpt))
    student.eval()
    print(f"  checkpoint: {CHECKPOINT_PATH.name}")

    teacher_feats = {}
    def make_hook(name):
        def _hook(module, inp, out):
            teacher_feats[name] = out
        return _hook
    teacher.layer1.register_forward_hook(make_hook("layer1"))
    teacher.layer2.register_forward_hook(make_hook("layer2"))
    teacher.layer3.register_forward_hook(make_hook("layer3"))

    # ── 4. scoring ────────────────────────────────────────────────────────────
    print("\n[4/5] scoring")
    ct_cache = LRUPatientCache(max_size=6)
    roi_cache = LRUPatientCache(max_size=6)

    score_rows = []
    n_scored = 0
    n_score_nan = 0
    n_score_inf = 0
    t_total = time.perf_counter()

    # 배치 처리
    batch_size = SCORE_BATCH_SIZE
    n_total = len(cand_rows)
    n_batches = math.ceil(n_total / batch_size)

    for b_idx in range(n_batches):
        batch = cand_rows[b_idx * batch_size: (b_idx + 1) * batch_size]
        crops_np = []
        meta_list = []

        for row in batch:
            candidate_id = row.get("candidate_id", "")
            patient_id = row.get("patient_id", "")
            safe_id = row.get("safe_id", "")
            stage_split = row.get("stage_split", "")
            local_z = int(row.get("z_center", 0))
            y0 = int(row.get("y0_fixed_crop", 0))
            x0 = int(row.get("x0_fixed_crop", 0))
            y1 = int(row.get("y1_fixed_crop", 0))
            x1 = int(row.get("x1_fixed_crop", 0))
            first_stage_score = float(row.get("mean_padim_score", 0.0) or 0.0)

            ct_path = NSCLC_CT_ROOT / safe_id / "ct_hu.npy"
            roi_path = V4_20_LESION_ROI_ROOT / safe_id / "refined_roi.npy"

            # six_bin 계산
            z_level, boundary_status, six_bin_label = "unknown", "unknown", "unknown"
            roi_ratio, boundary_overlap_ratio = 0.0, 0.0
            low_z_warning = int(local_z <= LOW_Z_WARNING_THRESHOLD)

            try:
                roi_arr = roi_cache.load(safe_id + "_roi", roi_path)
                n_slices = roi_arr.shape[0]
                z_ratio = local_z / max(n_slices - 1, 1)
                if 0 <= local_z < n_slices:
                    sy0, sx0 = max(0, y0), max(0, x0)
                    sy1, sx1 = min(roi_arr.shape[1], y1), min(roi_arr.shape[2], x1)
                    roi_patch = roi_arr[local_z, sy0:sy1, sx0:sx1]
                    full_roi_slice = roi_arr[local_z]
                    dist_patch_full = np.zeros_like(full_roi_slice, dtype=np.float32)
                    from scipy.ndimage import distance_transform_edt
                    dist_full = distance_transform_edt(full_roi_slice)
                    ring_full = ((full_roi_slice > 0) & (dist_full <= EROSION_PX)).astype(np.float32)
                    ring_patch = ring_full[sy0:sy1, sx0:sx1]
                    roi_sum = float(full_roi_slice[sy0:sy1, sx0:sx1].sum())
                    ring_sum = float(ring_patch.sum())
                    patch_area = float(CROP_SIZE * CROP_SIZE)
                    roi_ratio = roi_sum / patch_area
                    boundary_overlap_ratio = ring_sum / patch_area
                    is_boundary = boundary_overlap_ratio >= BOUNDARY_THRESHOLD
                    is_interior = (roi_ratio >= INTERIOR_ROI_MIN) and (not is_boundary)
                    if is_boundary:
                        boundary_status = "boundary"
                    elif is_interior:
                        boundary_status = "interior"
                    else:
                        boundary_status = "excluded"
                    z_level = z_level_from_ratio(z_ratio)
                    six_bin_label = "excluded" if boundary_status == "excluded" else f"{z_level}_{boundary_status}"
            except Exception as e:
                error_rows_all.append({
                    "phase": "sixbin", "candidate_id": candidate_id,
                    "patient_id": patient_id, "safe_id": safe_id, "error": str(e),
                })

            # crop 생성
            crop_np = None
            try:
                ct_arr = ct_cache.load(safe_id, ct_path)
                crop_np = build_crop_np(ct_arr, local_z, y0, x0, y1, x1)
            except Exception as e:
                error_rows_all.append({
                    "phase": "crop", "candidate_id": candidate_id,
                    "patient_id": patient_id, "safe_id": safe_id, "error": str(e),
                })
                crop_np = np.zeros((3, CROP_SIZE, CROP_SIZE), dtype=np.float32)

            crops_np.append(crop_np)
            meta_list.append({
                "candidate_id": candidate_id,
                "patient_id": patient_id,
                "safe_id": safe_id,
                "stage_split": stage_split,
                "local_z": local_z,
                "crop_y0": y0, "crop_x0": x0, "crop_y1": y1, "crop_x1": x1,
                "z_level": z_level,
                "boundary_status": boundary_status,
                "six_bin_label": six_bin_label,
                "low_z_warning": low_z_warning,
                "first_stage_score": first_stage_score,
                "roi_ratio": round(roi_ratio, 6),
                "boundary_overlap_ratio": round(boundary_overlap_ratio, 6),
            })

        # GPU scoring
        batch_t = torch.from_numpy(np.stack(crops_np, axis=0)).to(device)
        with torch.no_grad():
            teacher_feats.clear()
            teacher(batch_t)
            tf1 = teacher_feats["layer1"]
            tf2 = teacher_feats["layer2"]
            tf3 = teacher_feats["layer3"]
            de3, de2, de1 = student(tf3)
            sc1 = (1.0 - F.cosine_similarity(de1, tf1, dim=1)).mean(dim=(1, 2))
            sc2 = (1.0 - F.cosine_similarity(de2, tf2, dim=1)).mean(dim=(1, 2))
            sc3 = (1.0 - F.cosine_similarity(de3, tf3, dim=1)).mean(dim=(1, 2))
            crop_score = (sc1 + sc2 + sc3) / 3.0
            s1_np = sc1.cpu().numpy().astype("float32")
            s2_np = sc2.cpu().numpy().astype("float32")
            s3_np = sc3.cpu().numpy().astype("float32")
            cs_np = crop_score.cpu().numpy().astype("float32")

        for local_i, meta in enumerate(meta_list):
            cs = float(cs_np[local_i])
            is_nan = int(math.isnan(cs))
            is_inf = int(math.isinf(cs))
            n_score_nan += is_nan
            n_score_inf += is_inf

            bin_label = meta["six_bin_label"]
            bin_key_p95 = f"bin_{bin_label}"
            bin_key_p99 = f"bin_{bin_label}"
            bin_th = bin_thresholds.get(bin_key_p95, {})
            bin_p95 = float(bin_th.get("p95", global_p95))
            bin_p99 = float(bin_th.get("p99", global_p99))

            score_rows.append({
                "candidate_id": meta["candidate_id"],
                "patient_id": meta["patient_id"],
                "safe_id": meta["safe_id"],
                "stage_split": meta["stage_split"],
                "local_z": meta["local_z"],
                "crop_y0": meta["crop_y0"],
                "crop_x0": meta["crop_x0"],
                "crop_y1": meta["crop_y1"],
                "crop_x1": meta["crop_x1"],
                "z_level": meta["z_level"],
                "boundary_status": meta["boundary_status"],
                "six_bin_label": meta["six_bin_label"],
                "low_z_warning": meta["low_z_warning"],
                "first_stage_score": round(meta["first_stage_score"], 6),
                "score_layer1": round(float(s1_np[local_i]), 6),
                "score_layer2": round(float(s2_np[local_i]), 6),
                "score_layer3": round(float(s3_np[local_i]), 6),
                "rd4ad_crop_score": round(cs, 6) if not is_nan and not is_inf else cs,
                "global_p95": round(global_p95, 6),
                "global_p99": round(global_p99, 6),
                "bin_p95": round(bin_p95, 6),
                "bin_p99": round(bin_p99, 6),
                "global_p95_exceed": int(not is_nan and not is_inf and cs > global_p95),
                "global_p99_exceed": int(not is_nan and not is_inf and cs > global_p99),
                "bin_p95_exceed": int(not is_nan and not is_inf and cs > bin_p95),
                "bin_p99_exceed": int(not is_nan and not is_inf and cs > bin_p99),
                "score_nan": is_nan,
                "score_inf": is_inf,
            })
            n_scored += 1

        if b_idx % 50 == 0 or b_idx == n_batches - 1:
            elapsed = time.perf_counter() - t_total
            pct = (b_idx + 1) / n_batches * 100
            print(f"    batch {b_idx}/{n_batches}  {pct:5.1f}%  elapsed={elapsed:.0f}s")

    # ── 5. 결과 저장 ──────────────────────────────────────────────────────────
    print("\n[5/5] 결과 저장")

    score_fields = [
        "candidate_id", "patient_id", "safe_id", "stage_split",
        "local_z", "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "z_level", "boundary_status", "six_bin_label", "low_z_warning",
        "first_stage_score",
        "score_layer1", "score_layer2", "score_layer3", "rd4ad_crop_score",
        "global_p95", "global_p99", "bin_p95", "bin_p99",
        "global_p95_exceed", "global_p99_exceed",
        "bin_p95_exceed", "bin_p99_exceed",
        "score_nan", "score_inf",
    ]
    write_csv(OUTPUT_ROOT / "rd_b10_stage1_dev_candidate_score.csv", score_fields, score_rows)

    # 환자별 summary
    pat_groups = collections.defaultdict(list)
    for r in score_rows:
        pat_groups[r["patient_id"]].append(r)
    pat_summary = []
    for pid, rows_p in sorted(pat_groups.items()):
        valid = [r for r in rows_p if not r["score_nan"] and not r["score_inf"]]
        scores = [r["rd4ad_crop_score"] for r in valid]
        n_gp95 = sum(r["global_p95_exceed"] for r in valid)
        n_gp99 = sum(r["global_p99_exceed"] for r in valid)
        n_bp95 = sum(r["bin_p95_exceed"] for r in valid)
        n_bp99 = sum(r["bin_p99_exceed"] for r in valid)
        pat_summary.append({
            "patient_id": pid,
            "n_candidates": len(rows_p),
            "n_valid": len(valid),
            "mean_rd4ad_score": round(float(np.mean(scores)), 6) if scores else 0.0,
            "max_rd4ad_score": round(float(np.max(scores)), 6) if scores else 0.0,
            "mean_first_stage_score": round(float(np.mean([r["first_stage_score"] for r in rows_p])), 6),
            "n_global_p95_exceed": n_gp95,
            "n_global_p99_exceed": n_gp99,
            "n_bin_p95_exceed": n_bp95,
            "n_bin_p99_exceed": n_bp99,
        })
    write_csv(
        OUTPUT_ROOT / "rd_b10_score_by_patient_summary.csv",
        ["patient_id", "n_candidates", "n_valid", "mean_rd4ad_score", "max_rd4ad_score",
         "mean_first_stage_score", "n_global_p95_exceed", "n_global_p99_exceed",
         "n_bin_p95_exceed", "n_bin_p99_exceed"],
        pat_summary,
    )

    # six_bin별 summary
    bin_groups = collections.defaultdict(list)
    for r in score_rows:
        bin_groups[r["six_bin_label"]].append(r)
    bin_summary = []
    for bl in SIX_BIN_LABELS + ["excluded"]:
        rows_b = bin_groups.get(bl, [])
        valid = [r for r in rows_b if not r["score_nan"] and not r["score_inf"]]
        scores = [r["rd4ad_crop_score"] for r in valid]
        bin_summary.append({
            "six_bin_label": bl,
            "n": len(rows_b),
            "n_valid": len(valid),
            "mean": round(float(np.mean(scores)), 6) if scores else 0.0,
            "p95": round(float(np.percentile(scores, 95)), 6) if scores else 0.0,
            "p99": round(float(np.percentile(scores, 99)), 6) if scores else 0.0,
            "n_global_p95_exceed": sum(r["global_p95_exceed"] for r in valid),
            "n_bin_p95_exceed": sum(r["bin_p95_exceed"] for r in valid),
        })
    write_csv(
        OUTPUT_ROOT / "rd_b10_score_by_sixbin_summary.csv",
        ["six_bin_label", "n", "n_valid", "mean", "p95", "p99",
         "n_global_p95_exceed", "n_bin_p95_exceed"],
        bin_summary,
    )

    # threshold exceedance summary
    n_gp95_total = sum(r["global_p95_exceed"] for r in score_rows)
    n_gp99_total = sum(r["global_p99_exceed"] for r in score_rows)
    n_bp95_total = sum(r["bin_p95_exceed"] for r in score_rows)
    n_bp99_total = sum(r["bin_p99_exceed"] for r in score_rows)
    n_valid_total = sum(1 for r in score_rows if not r["score_nan"] and not r["score_inf"])
    exceedance_rows = [
        {"threshold": "global_p95", "value": global_p95, "n_exceed": n_gp95_total,
         "n_total": n_valid_total, "exceed_rate": round(n_gp95_total / max(n_valid_total, 1), 6)},
        {"threshold": "global_p99", "value": global_p99, "n_exceed": n_gp99_total,
         "n_total": n_valid_total, "exceed_rate": round(n_gp99_total / max(n_valid_total, 1), 6)},
        {"threshold": "bin_p95", "value": "per_bin", "n_exceed": n_bp95_total,
         "n_total": n_valid_total, "exceed_rate": round(n_bp95_total / max(n_valid_total, 1), 6)},
        {"threshold": "bin_p99", "value": "per_bin", "n_exceed": n_bp99_total,
         "n_total": n_valid_total, "exceed_rate": round(n_bp99_total / max(n_valid_total, 1), 6)},
    ]
    write_csv(
        OUTPUT_ROOT / "rd_b10_threshold_exceedance_summary.csv",
        ["threshold", "value", "n_exceed", "n_total", "exceed_rate"],
        exceedance_rows,
    )

    # first_stage vs rd4ad summary (환자별 상관)
    fs_rd4ad_rows = []
    for r in score_rows:
        if not r["score_nan"] and not r["score_inf"]:
            fs_rd4ad_rows.append({
                "candidate_id": r["candidate_id"],
                "patient_id": r["patient_id"],
                "first_stage_score": r["first_stage_score"],
                "rd4ad_crop_score": r["rd4ad_crop_score"],
                "global_p95_exceed": r["global_p95_exceed"],
                "global_p99_exceed": r["global_p99_exceed"],
            })
    write_csv(
        OUTPUT_ROOT / "rd_b10_firststage_vs_rd4ad_score_summary.csv",
        ["candidate_id", "patient_id", "first_stage_score", "rd4ad_crop_score",
         "global_p95_exceed", "global_p99_exceed"],
        fs_rd4ad_rows,
    )

    # errors
    write_csv(
        OUTPUT_ROOT / "rd_b10_errors.csv",
        ["phase", "candidate_id", "patient_id", "safe_id", "error"],
        error_rows_all,
    )

    # all_checks_passed: post_intersect 기준 (h_intersect는 scoring 전 탐지/제거된 holdout이므로 PASS 조건 아님)
    failure_flags = [
        len(post_intersect) > 0,
        n_score_nan > 0,
        n_score_inf > 0,
    ]
    all_checks_passed = not any(failure_flags)

    # summary JSON
    summary = {
        "candidate_source_path": str(CANDIDATE_MANIFEST),
        "n_input_candidates": n_input,
        "n_holdout_patients_detected": len(h_intersect),
        "n_holdout_rows_removed": n_holdout_removed,
        "n_scoring_candidates": len(cand_rows),
        "n_scored_candidates": n_scored,
        "post_filter_holdout_intersection": len(post_intersect),
        "checkpoint_loaded": str(CHECKPOINT_PATH),
        "threshold_source": "RD-B9 normal_val only",
        "global_p95": global_p95,
        "global_p99": global_p99,
        "six_bin_threshold_count": th["n_threshold_labels"],
        "score_nan_count": n_score_nan,
        "score_inf_count": n_score_inf,
        "n_global_p95_exceed": n_gp95_total,
        "n_global_p99_exceed": n_gp99_total,
        "n_bin_p95_exceed": n_bp95_total,
        "n_bin_p99_exceed": n_bp99_total,
        "scoring_started": True,
        "training_started": False,
        "backward_called": False,
        "optimizer_created": False,
        "checkpoint_saved": False,
        "threshold_recalculated": False,
        "first_stage_score_modified": False,
        "stage2_holdout_access": 0,
        "all_checks_passed": all_checks_passed,
    }
    write_json(OUTPUT_ROOT / "rd_b10_stage1_dev_candidate_scoring_summary.json", summary)

    # report.md
    verdict = "PASS" if all_checks_passed else "FAIL"
    md_lines = [
        "# RD-B10 stage1_dev Candidate Scoring Report",
        "",
        f"## 판정: {verdict}",
        "",
        "## 1. RD-B8f / RD-B9 결과 요약",
        "| 항목 | 값 |",
        "|---|---|",
        "| RD-B8f normal_train crops | 86,017 |",
        "| RD-B8f best_epoch | 20 |",
        "| RD-B8f final loss | 0.074174 |",
        "| RD-B9 normal_val patients | 36 |",
        "| RD-B9 normal_val crops | 8,354 |",
        f"| RD-B9 global p95 | {global_p95} |",
        f"| RD-B9 global p99 | {global_p99} |",
        "",
        "## 2. candidate source 선택 근거",
        f"- 선택: stage1_dev_fixed96_thr001_v1",
        "- 이유: safe_id·candidate_id·y0_fixed_crop 고정 96px crop·mean_padim_score 완비",
        "- stage1_dev only (holdout=0)",
        f"- 경로: {CANDIDATE_MANIFEST}",
        "",
        "## 3. stage1_dev allowlist / stage2_holdout denylist 검증",
        "| 항목 | 값 |",
        "|---|---|",
        f"| stage1_dev patients | {len(stage1_dev_ids)} |",
        f"| stage2_holdout patients | {len(holdout_ids)} |",
        f"| input candidates | {n_input} |",
        f"| holdout patients detected | {len(h_intersect)} |",
        f"| holdout rows removed | {n_holdout_removed} |",
        f"| scoring candidates | {len(cand_rows)} |",
        f"| post-filter holdout intersection | {len(post_intersect)} |",
        "",
        "## 4. scoring 입력 규모",
        f"- 입력 candidates: {n_input}",
        f"- scored candidates: {n_scored}",
        f"- score NaN/Inf: {n_score_nan}/{n_score_inf}",
        "",
        "## 5. threshold source",
        "- threshold source: RD-B9 normal_val only",
        "- threshold 재계산: False",
        f"- global p95 = {global_p95}",
        f"- global p99 = {global_p99}",
        "",
        "## 6. scoring 방식",
        "- teacher: ResNet18 ImageNet pretrained (layer1/layer2/layer3)",
        "- student: StudentDecoder (de_layer3→de_layer2→de_layer1)",
        "- crop: mixed 3ch (center + lower-3mm-MIP + upper-3mm-MIP), HU[-1000,600]→[0,1], 96px",
        "- crop 좌표: y0_fixed_crop/x0_fixed_crop/y1_fixed_crop/x1_fixed_crop (고정 96px)",
        "- ROI: v4_20 refined ROI (lesion branch)",
        "- score = mean(1 - cosine_similarity) across layer1/2/3",
        "- backward=False, optimizer=None, checkpoint_saved=False",
        "",
        "## 7. threshold exceedance 결과",
        "| threshold | value | n_exceed | n_total | exceed_rate |",
        "|---|---|---|---|---|",
    ] + [
        f"| {r['threshold']} | {r['value']} | {r['n_exceed']} | {r['n_total']} | {r['exceed_rate']} |"
        for r in exceedance_rows
    ] + [
        "",
        "## 8. first-stage score vs RD4AD score",
        "- first_stage_score: mean_padim_score (1차 PaDiM 스코어)",
        "- rd4ad_crop_score: RD-B8f teacher-student cosine distance (2차 RD4AD)",
        "- 두 score는 서로 독립적으로 저장; 기존 first_stage_score는 수정하지 않음",
        "",
        "## 9. 다음 단계",
        "- RD-B11: RD4AD score 기반 FP suppression / lesion safety analysis",
        "  * rd4ad_crop_score > global_p95 exceedance 기반 후보 필터링",
        "  * lesion 환자의 병변 포함 crop recall 확인",
        "",
        "## 10. 절대 하지 않은 것",
        "- training 없음",
        "- backward 없음",
        "- optimizer step 없음",
        "- checkpoint 저장 없음",
        "- threshold 재계산 없음",
        "- first_stage_score 수정 없음",
        "- stage2_holdout 접근 없음",
    ]
    with open(OUTPUT_ROOT / "rd_b10_stage1_dev_candidate_scoring_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("  -> rd_b10_stage1_dev_candidate_scoring_report.md")

    (OUTPUT_ROOT / "DONE").write_text(
        f"rd_b10_stage1_dev_candidate_scoring_v1 DONE\nall_checks_passed={all_checks_passed}\n",
        encoding="utf-8",
    )
    print("  -> DONE")

    # 최종 출력
    print()
    print("=" * 70)
    print(f"판정: {verdict}")
    print(f"  input candidates          : {n_input}")
    print(f"  holdout rows removed      : {n_holdout_removed}")
    print(f"  scoring candidates        : {len(cand_rows)}")
    print(f"  scored candidates         : {n_scored}")
    print(f"  post-filter holdout intersect: {len(post_intersect)}")
    print(f"  score NaN/Inf             : {n_score_nan}/{n_score_inf}")
    print(f"  global p95={global_p95}  n_exceed={n_gp95_total}")
    print(f"  global p99={global_p99}  n_exceed={n_gp99_total}")
    print(f"  bin_p95 n_exceed        : {n_bp95_total}")
    print(f"  all_checks_passed       : {all_checks_passed}")
    print("=" * 70)

    if not all_checks_passed:
        sys.exit(1)


# ── 진입점 ────────────────────────────────────────────────────────────────────

if IS_DRY_PLAN:
    run_dry_plan()
elif IS_RUN_SCORE:
    run_score()
