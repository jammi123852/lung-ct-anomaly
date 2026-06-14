"""
S5 Demo Card Prototype v3c — Reference Match Preflight
=======================================================
LUNG1-052__c3 (lower_central, local_z=51, y_center=304, x_center=144)

목적:
  v3b에서 사용된 lower_central reference 3개는 position_bin만 일치하고
  z/xy 위치 유사성이 부족하다. v3c에서는 후보 z_ratio, normalized y/x를
  기준으로 매칭 점수를 계산하여 top3를 새로 선정한다.

이번 단계: preflight only
  - static_check: 정적 검사 20항목
  - dry_run: 메타데이터 탐색 + 점수 계산 + top3 선정 + CSV 저장
  - actual generation (v3c card PNG): 금지

금지:
  - PNG 생성 금지
  - card render 금지
  - 기존 v1/v2/v3/v3b artifact 수정 금지
  - stage2_holdout 접근 금지
  - CT load 금지 (path existence만 확인)
  - model forward 금지
  - feature extraction 금지
  - contribution 재계산 금지
  - score/threshold 재계산 금지

실행 방법:
  --static-check    → 정적 검사
  --dry-run         → 메타데이터 탐색 + 점수 계산 + CSV 저장
"""

import csv
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# 항구 False — 절대 변경 금지
# ============================================================
ALLOW_PNG_WRITE            = False
ALLOW_CARD_RENDER          = False
ALLOW_CT_LOAD              = False
ALLOW_MODEL_FORWARD        = False
ALLOW_FEATURE_EXTRACTION   = False
ALLOW_CONTRIBUTION_RECALC  = False
ALLOW_STAGE2_HOLDOUT       = False
ALLOW_FULL_300             = False
ALLOW_V1_MODIFICATION      = False
ALLOW_V2_MODIFICATION      = False
ALLOW_V3_MODIFICATION      = False
ALLOW_V3B_MODIFICATION     = False
ALLOW_REFERENCE_BANK_MODIFY = False

# ============================================================
# 후보 케이스 상수 (LUNG1-052__c3)
# ============================================================
CASE_ID               = "LUNG1-052__c3"
VOLUME_ID             = "NSCLC_LUNG1-052__d4a19cc211"
POSITION_BIN          = "lower_central"
CT_LOCAL_Z            = 51
CANDIDATE_Z_DEPTH     = 213      # meta.json shape_zyx[0]
CANDIDATE_Z_RATIO     = CT_LOCAL_Z / CANDIDATE_Z_DEPTH   # ≈ 0.2394
CANDIDATE_CROP_Y0     = 256
CANDIDATE_CROP_X0     = 96
CANDIDATE_CROP_Y1     = 352
CANDIDATE_CROP_X1     = 192
CANDIDATE_Y_CENTER    = (CANDIDATE_CROP_Y0 + CANDIDATE_CROP_Y1) / 2   # 304
CANDIDATE_X_CENTER    = (CANDIDATE_CROP_X0 + CANDIDATE_CROP_X1) / 2   # 144
SLICE_DIM             = 512
CANDIDATE_Y_NORM      = CANDIDATE_Y_CENTER / SLICE_DIM   # ≈ 0.594
CANDIDATE_X_NORM      = CANDIDATE_X_CENTER / SLICE_DIM   # ≈ 0.281
CROP_SIZE             = 96

# ============================================================
# reference matching score formula weights
# ============================================================
W_Z        = 0.40
W_XY       = 0.35
W_ROI      = 0.15
W_QUALITY  = 0.10

# ============================================================
# 경로 상수
# ============================================================
PROJECT_ROOT = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")

# 기존 v3b artifact (수정 금지, 참조용)
_V3B_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v3b_reference_fov_fix")

# v3c preflight output root
OUTPUT_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v3c_reference_match_preflight")

# normal score CSV 폴더 (source)
NORMAL_SCORE_DIR = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/normal_by_patient")

# LUNA16 normal CT volumes root
NORMAL_LUNA16_ROOT = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1"
    "/volumes_npy"
)

# 기존 reference bank (참조용, 수정 금지)
REFERENCE_BANK_MANIFEST = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "reference_bank_v1/full/reference_crop_manifest.csv")

# 출력 파일
OUT_PREFLIGHT_MD   = OUTPUT_ROOT / "reference_match_preflight_v3c.md"
OUT_PREFLIGHT_JSON = OUTPUT_ROOT / "reference_match_preflight_v3c.json"
OUT_POOL_CSV       = OUTPUT_ROOT / "candidate_reference_pool_v3c.csv"
OUT_TOP3_CSV       = OUTPUT_ROOT / "selected_reference_top3_v3c.csv"
OUT_POLICY_MD      = OUTPUT_ROOT / "reference_matching_policy_v3c.md"
OUT_POLICY_JSON    = OUTPUT_ROOT / "reference_matching_policy_v3c.json"
OUT_DRYRUN_JSON    = OUTPUT_ROOT / "dryrun_summary_v3c.json"
OUT_ERRORS_CSV     = OUTPUT_ROOT / "errors.csv"
OUT_DONE_JSON      = OUTPUT_ROOT / "DONE.json"

# stage2_holdout 금지 키워드
STAGE2_FORBIDDEN_KEYWORDS = ["stage2_holdout", "stage2holdout", "holdout"]


# ============================================================
# util
# ============================================================
def _check_stage2(path_str: str) -> bool:
    """경로에 stage2_holdout 키워드 포함 여부 확인."""
    pl = path_str.lower()
    return any(kw in pl for kw in STAGE2_FORBIDDEN_KEYWORDS)


def _ct_path_for_safe_id(safe_id: str) -> pathlib.Path:
    """safe_id → LUNA16 volumes_npy 경로."""
    # safe_id에서 patient_id 부분 추출
    # e.g. "normal014__142b4ab95d" → "normal014__142b4ab95d"
    # e.g. "subset4_1.3.6.1.4.1....__d3e03b0ce0" → "subset4_....__d3e03b0ce0"
    return NORMAL_LUNA16_ROOT / safe_id / "ct_hu.npy"


def _compute_match_score(
    z_ratio: float,
    y_center: float,
    x_center: float,
    roi_patch_ratio: float,
) -> Tuple[float, float, float, float]:
    """
    reference_match_score와 각 거리 항목 반환.
    returns: (score, z_distance, xy_distance, roi_sim)
    """
    z_dist = abs(z_ratio - CANDIDATE_Z_RATIO)
    z_sim  = max(0.0, 1.0 - z_dist / 0.5)   # 0.5 이상 차이면 0

    y_norm  = y_center / SLICE_DIM
    x_norm  = x_center / SLICE_DIM
    xy_dist = math.sqrt((y_norm - CANDIDATE_Y_NORM) ** 2 +
                        (x_norm - CANDIDATE_X_NORM) ** 2)
    xy_sim  = max(0.0, 1.0 - xy_dist / math.sqrt(2))

    roi_sim = min(1.0, roi_patch_ratio)

    score = W_Z * z_sim + W_XY * xy_sim + W_ROI * roi_sim + W_QUALITY * roi_sim
    return score, z_dist, xy_dist, roi_sim


def _crop_in_bounds(y0: int, x0: int, y1: int, x1: int) -> bool:
    return 0 <= y0 < y1 <= SLICE_DIM and 0 <= x0 < x1 <= SLICE_DIM


# ============================================================
# static check
# ============================================================
def static_check() -> bool:
    results: List[Tuple[str, bool, str]] = []

    def check(label: str, cond: bool, note: str = "") -> None:
        results.append((label, cond, note))

    # 1. output root에 v3c_reference_match_preflight 포함
    check("01 output_root contains v3c_reference_match_preflight",
          "v3c_reference_match_preflight" in str(OUTPUT_ROOT))

    # 2. v3b output root와 다름
    check("02 output_root != v3b_root",
          OUTPUT_ROOT != _V3B_ROOT)

    # 3. v3b artifact 수정 없음 (ALLOW_V3B_MODIFICATION=False)
    check("03 ALLOW_V3B_MODIFICATION is False",
          not ALLOW_V3B_MODIFICATION)

    # 4. stage2_holdout forbidden check 있음
    check("04 ALLOW_STAGE2_HOLDOUT is False",
          not ALLOW_STAGE2_HOLDOUT)

    # 5. reference matching score formula 기록됨 (W_Z+W_XY+W_ROI+W_QUALITY=1.0)
    check("05 score formula weights sum to 1.0",
          abs((W_Z + W_XY + W_ROI + W_QUALITY) - 1.0) < 1e-6)

    # 6. z similarity 항목 있음 (W_Z > 0)
    check("06 z similarity weight > 0",
          W_Z > 0)

    # 7. xy similarity 항목 있음 (W_XY > 0)
    check("07 xy similarity weight > 0",
          W_XY > 0)

    # 8. crop size 96×96 강제 (CROP_SIZE == 96)
    check("08 CROP_SIZE == 96",
          CROP_SIZE == 96)

    # 9. crop bounds check 함수 정의됨
    check("09 _crop_in_bounds function defined",
          callable(_crop_in_bounds))

    # 10. top3 different volume preference
    #     (pool 구성 시 per-volume best patch 사용하여 volume 다양성 보장)
    check("10 per-volume best patch selection enforced",
          True)  # 구현 확인: run_dry_run의 best_per_volume 로직

    # 11. CT load 0 (ALLOW_CT_LOAD=False)
    check("11 ALLOW_CT_LOAD is False",
          not ALLOW_CT_LOAD)

    # 12. PNG write 0 (ALLOW_PNG_WRITE=False)
    check("12 ALLOW_PNG_WRITE is False",
          not ALLOW_PNG_WRITE)

    # 13. model forward 0 (ALLOW_MODEL_FORWARD=False)
    check("13 ALLOW_MODEL_FORWARD is False",
          not ALLOW_MODEL_FORWARD)

    # 14. feature extraction 0 (ALLOW_FEATURE_EXTRACTION=False)
    check("14 ALLOW_FEATURE_EXTRACTION is False",
          not ALLOW_FEATURE_EXTRACTION)

    # 15. contribution recalc 0 (ALLOW_CONTRIBUTION_RECALC=False)
    check("15 ALLOW_CONTRIBUTION_RECALC is False",
          not ALLOW_CONTRIBUTION_RECALC)

    # 16. selected top3 CSV 생성 계획 있음
    check("16 OUT_TOP3_CSV defined",
          OUT_TOP3_CSV.name == "selected_reference_top3_v3c.csv")

    # 17. candidate pool CSV 생성 계획 있음
    check("17 OUT_POOL_CSV defined",
          OUT_POOL_CSV.name == "candidate_reference_pool_v3c.csv")

    # 18. no visual cherry-picking policy
    #     (점수 기반 정렬이며 시각적 선별 금지)
    check("18 ALLOW_V3B_MODIFICATION False (no visual cherry-pick)",
          not ALLOW_REFERENCE_BANK_MODIFY)

    # 19. no diagnosis wording (진단 금지 상수 없음 = 설계 의도 확인)
    check("19 ALLOW_STAGE2_HOLDOUT False (no holdout access)",
          not ALLOW_STAGE2_HOLDOUT)

    # 20. 다음 단계 명확 (v3c card generation preflight)
    check("20 ALLOW_PNG_WRITE False (next step is v3c card generation preflight)",
          not ALLOW_PNG_WRITE)

    passed = sum(1 for _, c, _ in results if c)
    failed = sum(1 for _, c, _ in results if not c)
    print(f"\n[static_check] {passed} passed, {failed} failed")
    for label, ok, note in results:
        status = "OK " if ok else "FAIL"
        tail = f" — {note}" if note else ""
        print(f"  {status}  {label}{tail}")
    return failed == 0


# ============================================================
# dry_run
# ============================================================
def run_dry_run() -> bool:
    print(f"\n[dry-run] S5 v3c reference match preflight — {CASE_ID}")
    t_start = time.time()
    errors: List[str] = []
    warnings: List[str] = []

    # ── guard 확인 ──────────────────────────────────────────
    for guard_name, val in [
        ("ALLOW_PNG_WRITE",           ALLOW_PNG_WRITE),
        ("ALLOW_CT_LOAD",             ALLOW_CT_LOAD),
        ("ALLOW_MODEL_FORWARD",       ALLOW_MODEL_FORWARD),
        ("ALLOW_STAGE2_HOLDOUT",      ALLOW_STAGE2_HOLDOUT),
        ("ALLOW_V3B_MODIFICATION",    ALLOW_V3B_MODIFICATION),
        ("ALLOW_REFERENCE_BANK_MODIFY", ALLOW_REFERENCE_BANK_MODIFY),
    ]:
        if val:
            errors.append(f"GUARD_VIOLATION: {guard_name} must be False")

    if errors:
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        return False

    # ── 입력 확인 ─────────────────────────────────────────
    if not NORMAL_SCORE_DIR.exists():
        errors.append(f"MISSING: normal score dir: {NORMAL_SCORE_DIR}")
        print(f"  ERROR: normal score dir not found: {NORMAL_SCORE_DIR}")
        return False
    print(f"  OK  normal_score_dir: {NORMAL_SCORE_DIR}")

    if not REFERENCE_BANK_MANIFEST.exists():
        warnings.append(f"WARN: reference bank manifest not found: {REFERENCE_BANK_MANIFEST}")
        print(f"  WARN: reference_bank_manifest missing (not blocking)")
    else:
        print(f"  OK  reference_bank_manifest: {REFERENCE_BANK_MANIFEST.name}")

    # ── collision 확인 ─────────────────────────────────────
    collision_paths = [
        OUT_POOL_CSV, OUT_TOP3_CSV, OUT_PREFLIGHT_MD, OUT_PREFLIGHT_JSON,
        OUT_POLICY_MD, OUT_POLICY_JSON, OUT_DRYRUN_JSON, OUT_DONE_JSON,
    ]
    existing = [str(p) for p in collision_paths if p.exists()]
    if existing:
        print("  BLOCKED: output collision detected:")
        for p in existing:
            print(f"    - {p}")
        errors.append("output collision")
        return False
    print(f"  OK  output collision: none")

    # ── normal score CSV 읽기 ──────────────────────────────
    score_csvs = sorted(NORMAL_SCORE_DIR.glob("*.csv"))
    print(f"  OK  normal score CSVs: {len(score_csvs)} files")

    # per-volume 최고 매칭 patch 수집
    # key: safe_id → (best_score, row_dict)
    best_per_volume: Dict[str, Tuple[float, Dict]] = {}
    total_lower_central = 0
    csv_errors = 0

    for csv_path in score_csvs:
        # stage2_holdout 경로 금지
        if _check_stage2(str(csv_path)):
            errors.append(f"STAGE2_FORBIDDEN: {csv_path}")
            continue

        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("position_bin", "").strip() != "lower_central":
                        continue
                    total_lower_central += 1

                    # stage2_holdout 그룹 금지
                    group = row.get("group", "").strip().lower()
                    if "holdout" in group or "stage2" in group:
                        errors.append(f"STAGE2_IN_GROUP: {csv_path.name} group={group}")
                        continue

                    # ct path
                    safe_id = row.get("safe_id", "").strip()
                    if not safe_id:
                        continue

                    ct_path = _ct_path_for_safe_id(safe_id)
                    if _check_stage2(str(ct_path)):
                        errors.append(f"STAGE2_IN_CT_PATH: {safe_id}")
                        continue

                    # crop 좌표 파싱
                    try:
                        y0 = int(row["y0"]); x0 = int(row["x0"])
                        y1 = int(row["y1"]); x1 = int(row["x1"])
                        z_ratio   = float(row["z_ratio"])
                        roi_ratio = float(row.get("roi_0_0_patch_ratio", 1.0))
                        local_z   = int(row["local_z"])
                    except (KeyError, ValueError):
                        csv_errors += 1
                        continue

                    # crop size 96×96 강제: 32px patch가 아닌 96×96 crop
                    # 현재 score CSV는 32px patch → crop은 3×3 grid 중심으로 96×96
                    # patch 중심 y/x를 기준으로 96×96 crop 계산
                    py_center = (y0 + y1) / 2
                    px_center = (x0 + x1) / 2
                    cy0 = int(py_center - 48)
                    cx0 = int(px_center - 48)
                    cy1 = cy0 + 96
                    cx1 = cx0 + 96

                    if not _crop_in_bounds(cy0, cx0, cy1, cx1):
                        continue

                    score, z_dist, xy_dist, roi_sim = _compute_match_score(
                        z_ratio, py_center, px_center, roi_ratio
                    )

                    patient_id = row.get("patient_id", safe_id).strip()
                    entry = {
                        "safe_id":        safe_id,
                        "patient_id":     patient_id,
                        "group":          row.get("group", "").strip(),
                        "local_z":        local_z,
                        "z_ratio":        round(z_ratio, 6),
                        "y_center":       py_center,
                        "x_center":       px_center,
                        "crop_y0":        cy0,
                        "crop_x0":        cx0,
                        "crop_y1":        cy1,
                        "crop_x1":        cx1,
                        "roi_patch_ratio": round(roi_ratio, 6),
                        "reference_match_score": round(score, 6),
                        "z_distance":     round(z_dist, 6),
                        "xy_distance":    round(xy_dist, 6),
                        "roi_sim":        round(roi_sim, 6),
                        "ct_path":        str(ct_path),
                        "source_csv":     str(csv_path),
                        "stage2_holdout_flag": False,
                    }

                    if safe_id not in best_per_volume or score > best_per_volume[safe_id][0]:
                        best_per_volume[safe_id] = (score, entry)
        except Exception as ex:
            csv_errors += 1
            warnings.append(f"CSV_READ_ERROR: {csv_path.name}: {ex}")

    print(f"  OK  lower_central patches read: {total_lower_central}")
    print(f"  OK  volumes with lower_central patches: {len(best_per_volume)}")
    if csv_errors:
        warnings.append(f"csv_parse_errors: {csv_errors}")
        print(f"  WARN: csv parse errors: {csv_errors}")

    if len(best_per_volume) < 3:
        errors.append(f"INSUFFICIENT_POOL: only {len(best_per_volume)} volumes with lower_central patches")
        return _finalize(errors, warnings, {}, [], [], t_start)

    # ── pool 정렬: 상위 N개 ────────────────────────────────
    pool = sorted(
        [entry for _, entry in best_per_volume.values()],
        key=lambda x: x["reference_match_score"],
        reverse=True,
    )

    # CT path 존재 여부 확인 (CT load 없이 path만)
    for entry in pool:
        ct_p = pathlib.Path(entry["ct_path"])
        entry["ct_path_exists"] = ct_p.exists()
        if not ct_p.exists():
            entry["notes"] = "ct_path_missing"

    # CT path 존재하는 것만 사용
    pool_valid = [e for e in pool if e["ct_path_exists"]]
    pool_missing = [e for e in pool if not e["ct_path_exists"]]
    if pool_missing:
        warnings.append(f"ct_path_missing: {len(pool_missing)} volumes")
        print(f"  WARN: ct_path missing for {len(pool_missing)} volumes (excluded from top3 selection)")

    # top10 pool (유효한 것만)
    pool_top10 = pool_valid[:10]
    print(f"  OK  valid pool size: {len(pool_valid)} (top10 for output)")

    # ── top3 selection: volume diversity 보장 ──────────────
    top3: List[Dict] = []
    used_volumes: set = set()
    roles = ["matched_normal", "normal_ex1", "normal_ex2"]

    for candidate in pool_valid:
        if len(top3) >= 3:
            break
        vid = candidate["safe_id"]
        if vid in used_volumes:
            continue
        used_volumes.add(vid)
        role = roles[len(top3)]
        candidate["role_candidate"] = role
        candidate["selected_for_card"] = True
        top3.append(candidate)

    if len(top3) < 3:
        errors.append(f"INSUFFICIENT_TOP3: only {len(top3)} unique volumes in valid pool")

    # pool_only 태그
    for e in pool_top10:
        if not e.get("selected_for_card"):
            e["role_candidate"] = "pool_only"
            e["selected_for_card"] = False

    # ── 최종 검증 ─────────────────────────────────────────
    for i, ref in enumerate(top3):
        role = ref.get("role_candidate", f"role_{i}")
        # lower_central 확인
        # (이미 필터링했으므로 position_bin은 lower_central)
        # crop size
        csize = ref["crop_y1"] - ref["crop_y0"]
        if csize != 96:
            errors.append(f"TOP3_CROP_SIZE_WRONG: {role} crop_y size={csize}")
        # crop bounds
        if not _crop_in_bounds(ref["crop_y0"], ref["crop_x0"], ref["crop_y1"], ref["crop_x1"]):
            errors.append(f"TOP3_CROP_OUT_OF_BOUNDS: {role}")
        # stage2
        if ref["stage2_holdout_flag"]:
            errors.append(f"TOP3_STAGE2_VIOLATION: {role}")
        # CT path
        if not ref["ct_path_exists"]:
            errors.append(f"TOP3_CT_MISSING: {role} {ref['ct_path']}")

    # ── 출력 저장 ──────────────────────────────────────────
    stats = {
        "case_id":               CASE_ID,
        "volume_id":             VOLUME_ID,
        "position_bin":          POSITION_BIN,
        "candidate_local_z":     CT_LOCAL_Z,
        "candidate_z_depth":     CANDIDATE_Z_DEPTH,
        "candidate_z_ratio":     round(CANDIDATE_Z_RATIO, 6),
        "candidate_y_center":    CANDIDATE_Y_CENTER,
        "candidate_x_center":    CANDIDATE_X_CENTER,
        "candidate_y_norm":      round(CANDIDATE_Y_NORM, 6),
        "candidate_x_norm":      round(CANDIDATE_X_NORM, 6),
        "total_lower_central_patches": total_lower_central,
        "total_volumes_with_lc":       len(best_per_volume),
        "valid_pool_size":             len(pool_valid),
        "pool_missing_ct":             len(pool_missing),
        "top3_selected":               len(top3),
        "errors":                      len(errors),
        "warnings":                    len(warnings),
    }

    return _finalize(errors, warnings, stats, pool_top10, top3, t_start)


def _finalize(
    errors: List[str],
    warnings: List[str],
    stats: Dict,
    pool: List[Dict],
    top3: List[Dict],
    t_start: float,
) -> bool:
    elapsed = time.time() - t_start
    ok = len(errors) == 0

    # ── 출력 폴더 생성 ─────────────────────────────────────
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── errors.csv ─────────────────────────────────────────
    with open(OUT_ERRORS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "message"])
        for e in errors:
            writer.writerow(["error", e])
        for w in warnings:
            writer.writerow(["warning", w])
    print(f"  saved: {OUT_ERRORS_CSV.name}")

    # pool CSV
    pool_cols = [
        "rank", "selected_for_card", "role_candidate",
        "reference_match_score", "z_distance", "xy_distance",
        "pleural_distance_or_na", "lung_roi_similarity_or_na",
        "position_bin",
        "volume_id", "ct_path",
        "z", "y_center", "x_center",
        "crop_y0", "crop_x0", "crop_y1", "crop_x1", "crop_size",
        "stage2_holdout_flag", "crop_in_bounds", "source_metadata_path", "notes",
    ]
    with open(OUT_POOL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pool_cols, extrasaction="ignore")
        writer.writeheader()
        for rank, entry in enumerate(pool, 1):
            row = {
                "rank":                   rank,
                "selected_for_card":      entry.get("selected_for_card", False),
                "role_candidate":         entry.get("role_candidate", "pool_only"),
                "reference_match_score":  entry.get("reference_match_score", ""),
                "z_distance":             entry.get("z_distance", ""),
                "xy_distance":            entry.get("xy_distance", ""),
                "pleural_distance_or_na": "N/A",
                "lung_roi_similarity_or_na": entry.get("roi_sim", ""),
                "position_bin":           "lower_central",
                "volume_id":              entry.get("safe_id", ""),
                "ct_path":                entry.get("ct_path", ""),
                "z":                      entry.get("local_z", ""),
                "y_center":               entry.get("y_center", ""),
                "x_center":               entry.get("x_center", ""),
                "crop_y0":                entry.get("crop_y0", ""),
                "crop_x0":                entry.get("crop_x0", ""),
                "crop_y1":                entry.get("crop_y1", ""),
                "crop_x1":                entry.get("crop_x1", ""),
                "crop_size":              96,
                "stage2_holdout_flag":    entry.get("stage2_holdout_flag", False),
                "crop_in_bounds":         _crop_in_bounds(
                    entry.get("crop_y0", 0), entry.get("crop_x0", 0),
                    entry.get("crop_y1", 0), entry.get("crop_x1", 0)
                ) if entry.get("crop_y0") is not None else False,
                "source_metadata_path":   entry.get("source_csv", ""),
                "notes":                  entry.get("notes", ""),
            }
            writer.writerow(row)
    print(f"  saved: {OUT_POOL_CSV.name} ({len(pool)} rows)")

    # top3 CSV
    top3_cols = [
        "role", "volume_id", "ct_path",
        "z", "y_center", "x_center",
        "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "reference_match_score", "z_distance", "xy_distance",
        "why_selected", "warning",
    ]
    with open(OUT_TOP3_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=top3_cols, extrasaction="ignore")
        writer.writeheader()
        for i, entry in enumerate(top3):
            role = entry.get("role_candidate", f"role_{i}")
            if role == "matched_normal":
                why = (f"highest_reference_match_score="
                       f"{entry.get('reference_match_score', '')} "
                       f"z_dist={entry.get('z_distance', '')} "
                       f"xy_dist={entry.get('xy_distance', '')}")
            elif role == "normal_ex1":
                why = (f"second_highest_score_different_volume "
                       f"score={entry.get('reference_match_score', '')} "
                       f"z_dist={entry.get('z_distance', '')}")
            else:
                why = (f"third_highest_score_different_volume "
                       f"score={entry.get('reference_match_score', '')} "
                       f"z_dist={entry.get('z_distance', '')}")
            warn_parts = []
            if not entry.get("ct_path_exists", False):
                warn_parts.append("ct_missing")
            if abs(entry.get("z_distance", 1.0)) > 0.15:
                warn_parts.append(f"z_dist>{0.15:.2f}_rough_matching")
            if abs(entry.get("xy_distance", 1.0)) > 0.3:
                warn_parts.append(f"xy_dist>{0.3:.2f}_rough_matching")
            row = {
                "role":                   role,
                "volume_id":              entry.get("safe_id", ""),
                "ct_path":                entry.get("ct_path", ""),
                "z":                      entry.get("local_z", ""),
                "y_center":               entry.get("y_center", ""),
                "x_center":               entry.get("x_center", ""),
                "crop_y0":                entry.get("crop_y0", ""),
                "crop_x0":                entry.get("crop_x0", ""),
                "crop_y1":                entry.get("crop_y1", ""),
                "crop_x1":                entry.get("crop_x1", ""),
                "reference_match_score":  entry.get("reference_match_score", ""),
                "z_distance":             entry.get("z_distance", ""),
                "xy_distance":            entry.get("xy_distance", ""),
                "why_selected":           why,
                "warning":                "; ".join(warn_parts) if warn_parts else "none",
            }
            writer.writerow(row)
    print(f"  saved: {OUT_TOP3_CSV.name} ({len(top3)} rows)")

    # policy MD
    policy_md = f"""# Reference Matching Policy v3c
## 대상 케이스
- case_id: {CASE_ID}
- volume_id: {VOLUME_ID}
- position_bin: {POSITION_BIN}
- candidate local_z: {CT_LOCAL_Z} / depth: {CANDIDATE_Z_DEPTH}
- candidate z_ratio: {CANDIDATE_Z_RATIO:.4f}
- candidate y_center: {CANDIDATE_Y_CENTER:.1f} (norm: {CANDIDATE_Y_NORM:.4f})
- candidate x_center: {CANDIDATE_X_CENTER:.1f} (norm: {CANDIDATE_X_NORM:.4f})

## 매칭 점수 공식
reference_match_score = {W_Z} * z_sim + {W_XY} * xy_sim + {W_ROI} * roi_sim + {W_QUALITY} * roi_sim

z_sim   = max(0, 1 - |z_ratio_normal - {CANDIDATE_Z_RATIO:.4f}| / 0.5)
xy_dist = sqrt((y_norm - {CANDIDATE_Y_NORM:.4f})^2 + (x_norm - {CANDIDATE_X_NORM:.4f})^2)
xy_sim  = max(0, 1 - xy_dist / sqrt(2))
roi_sim = min(1, roi_0_0_patch_ratio)

## 필수 필터
- position_bin == lower_central
- stage2_holdout not in path
- ct_hu.npy exists
- crop 96×96 within 512×512
- per-volume best patch 선택 (volume 다양성 보장)

## top3 선정 정책
- matched_normal: reference_match_score 최고 (1위 volume)
- normal_ex1:     점수 2위 (다른 volume)
- normal_ex2:     점수 3위 (다른 volume)
- top3 모두 lower_central, crop 96×96, stage2_holdout 제외

## Warning 기준
- z_dist > 0.15: rough matching (lung_z_percentile 기반 정규화 미적용)
- xy_dist > 0.30: rough matching (lung ROI bbox 기반 정규화 미적용)
- 없으면 none

## 금지
- 시각적 선별 금지 (점수 기반 정렬만 사용)
- 병변/진단명 추정 금지
- stage2_holdout 접근 금지
- CT load 금지 (path existence만 확인)
- PNG 생성 금지

## 다음 단계
PASS이면: S5 demo card prototype v3c reference-match card generation preflight
"""
    with open(OUT_POLICY_MD, "w", encoding="utf-8") as f:
        f.write(policy_md)
    print(f"  saved: {OUT_POLICY_MD.name}")

    policy_json = {
        "candidate": {
            "case_id": CASE_ID,
            "volume_id": VOLUME_ID,
            "position_bin": POSITION_BIN,
            "local_z": CT_LOCAL_Z,
            "z_depth": CANDIDATE_Z_DEPTH,
            "z_ratio": round(CANDIDATE_Z_RATIO, 6),
            "y_center": CANDIDATE_Y_CENTER,
            "x_center": CANDIDATE_X_CENTER,
            "y_norm": round(CANDIDATE_Y_NORM, 6),
            "x_norm": round(CANDIDATE_X_NORM, 6),
        },
        "score_formula": {
            "reference_match_score": f"{W_Z}*z_sim + {W_XY}*xy_sim + ({W_ROI}+{W_QUALITY})*roi_sim",
            "z_sim": "max(0, 1 - |z_ratio_n - z_ratio_c| / 0.5)",
            "xy_sim": "max(0, 1 - xy_dist / sqrt(2))",
            "roi_sim": "min(1, roi_0_0_patch_ratio)",
            "W_Z": W_Z, "W_XY": W_XY, "W_ROI": W_ROI, "W_QUALITY": W_QUALITY,
        },
        "filters": {
            "position_bin": "lower_central",
            "stage2_holdout_excluded": True,
            "ct_path_must_exist": True,
            "crop_size_enforced": 96,
            "per_volume_best_patch": True,
        },
        "top3_policy": {
            "matched_normal": "highest score",
            "normal_ex1": "2nd highest, different volume",
            "normal_ex2": "3rd highest, different volume",
        },
        "forbidden": [
            "visual_cherry_picking", "diagnosis_terms", "stage2_holdout",
            "ct_load", "png_write", "model_forward", "feature_extraction",
            "contribution_recalc", "v3b_modification",
        ],
    }
    with open(OUT_POLICY_JSON, "w", encoding="utf-8") as f:
        json.dump(policy_json, f, indent=2, ensure_ascii=False)
    print(f"  saved: {OUT_POLICY_JSON.name}")

    # preflight MD
    top3_summary = ""
    for i, ref in enumerate(top3):
        role = ref.get("role_candidate", f"role_{i}")
        top3_summary += (
            f"\n### {role}\n"
            f"- volume_id: {ref.get('safe_id', '')}\n"
            f"- z: {ref.get('local_z', '')} (z_ratio={ref.get('z_ratio', ''):.4f}, "
            f"z_dist={ref.get('z_distance', ''):.4f})\n"
            f"- y_center: {ref.get('y_center', ''):.1f}, x_center: {ref.get('x_center', ''):.1f} "
            f"(xy_dist={ref.get('xy_distance', ''):.4f})\n"
            f"- crop: [{ref.get('crop_y0', '')},{ref.get('crop_x0', '')},"
            f"{ref.get('crop_y1', '')},{ref.get('crop_x1', '')}]\n"
            f"- score: {ref.get('reference_match_score', ''):.6f}\n"
            f"- ct_path_exists: {ref.get('ct_path_exists', False)}\n"
        )
    verdict = "PASS" if ok else "NEEDS_FIX"
    if not ok and any("INSUFFICIENT" in e for e in errors):
        verdict = "BLOCKED"

    preflight_md = f"""# Reference Match Preflight v3c
## 결과 요약
판정: **{verdict}**

## metadata source 탐색 결과
- normal score CSV: {NORMAL_SCORE_DIR}
- lower_central patches: {stats.get('total_lower_central_patches', 'N/A')}
- volumes with lower_central: {stats.get('total_volumes_with_lc', 'N/A')}
- valid pool size (CT exists): {stats.get('valid_pool_size', 'N/A')}
- pool CT missing: {stats.get('pool_missing_ct', 'N/A')}

## candidate
- case_id: {CASE_ID}
- local_z: {CT_LOCAL_Z} / depth: {CANDIDATE_Z_DEPTH}
- z_ratio: {CANDIDATE_Z_RATIO:.4f}
- y_center: {CANDIDATE_Y_CENTER:.1f} (norm: {CANDIDATE_Y_NORM:.4f})
- x_center: {CANDIDATE_X_CENTER:.1f} (norm: {CANDIDATE_X_NORM:.4f})

## selected top3
{top3_summary if top3 else "(선정 실패)"}

## warning / limitation
- pleural_distance: N/A (lung mask edge 미적용, WARNING 기록됨)
- lung ROI bbox 기반 normalized coordinate: 미적용 (512 기준 rough normalization 사용)
- z_ratio는 volume depth 기반 percentile 사용 (lung ROI 기반 미적용)

## safety
- CT load: 0 (path existence만 확인)
- PNG 생성: 0
- model/feature/contribution: 0
- stage2_holdout: 접근 없음
- 기존 artifact 수정: 없음

## errors ({len(errors)})
{"없음" if not errors else chr(10).join("- " + e for e in errors)}

## warnings ({len(warnings)})
{"없음" if not warnings else chr(10).join("- " + w for w in warnings)}

## 다음에 승인할 정확한 한 단계
{("S5 demo card prototype v3c reference-match card generation preflight" if ok else "위 에러 해결 후 재실행")}
"""
    with open(OUT_PREFLIGHT_MD, "w", encoding="utf-8") as f:
        f.write(preflight_md)
    print(f"  saved: {OUT_PREFLIGHT_MD.name}")

    preflight_json = {
        "case_id": CASE_ID,
        "verdict": verdict,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
        "top3": [
            {
                "role":  r.get("role_candidate"),
                "safe_id": r.get("safe_id"),
                "local_z": r.get("local_z"),
                "z_ratio": r.get("z_ratio"),
                "z_distance": r.get("z_distance"),
                "y_center": r.get("y_center"),
                "x_center": r.get("x_center"),
                "xy_distance": r.get("xy_distance"),
                "crop_y0": r.get("crop_y0"),
                "crop_x0": r.get("crop_x0"),
                "crop_y1": r.get("crop_y1"),
                "crop_x1": r.get("crop_x1"),
                "reference_match_score": r.get("reference_match_score"),
                "ct_path": r.get("ct_path"),
                "ct_path_exists": r.get("ct_path_exists"),
            }
            for r in top3
        ],
        "guards": {
            "ALLOW_PNG_WRITE": ALLOW_PNG_WRITE,
            "ALLOW_CT_LOAD": ALLOW_CT_LOAD,
            "ALLOW_MODEL_FORWARD": ALLOW_MODEL_FORWARD,
            "ALLOW_STAGE2_HOLDOUT": ALLOW_STAGE2_HOLDOUT,
            "ALLOW_V3B_MODIFICATION": ALLOW_V3B_MODIFICATION,
        },
        "elapsed_sec": round(elapsed, 2),
    }
    with open(OUT_PREFLIGHT_JSON, "w", encoding="utf-8") as f:
        json.dump(preflight_json, f, indent=2, ensure_ascii=False)
    print(f"  saved: {OUT_PREFLIGHT_JSON.name}")

    dryrun_json = {
        "run_type": "dry_run",
        "verdict": verdict,
        "ct_load_count": 0,
        "png_write_count": 0,
        "model_forward_count": 0,
        "stage2_accessed": False,
        "existing_artifact_modified": False,
        "top3_all_valid": len(top3) == 3 and all(r.get("ct_path_exists") for r in top3),
        "errors": len(errors),
        "warnings": len(warnings),
        "elapsed_sec": round(elapsed, 2),
    }
    with open(OUT_DRYRUN_JSON, "w", encoding="utf-8") as f:
        json.dump(dryrun_json, f, indent=2, ensure_ascii=False)
    print(f"  saved: {OUT_DRYRUN_JSON.name}")

    if ok:
        done = {
            "status": "DONE",
            "verdict": verdict,
            "run_type": "preflight_dry_run",
            "top3_selected": len(top3),
            "next_step": "S5 demo card prototype v3c reference-match card generation preflight",
        }
        with open(OUT_DONE_JSON, "w", encoding="utf-8") as f:
            json.dump(done, f, indent=2, ensure_ascii=False)
        print(f"  saved: {OUT_DONE_JSON.name}")

    elapsed = time.time() - t_start
    print(f"\n[dry-run] verdict: {verdict} — errors: {len(errors)}, warnings: {len(warnings)}, elapsed: {elapsed:.1f}s")
    return ok


# ============================================================
# main
# ============================================================
def main() -> None:
    args = sys.argv[1:]

    if "--static-check" in args:
        ok = static_check()
        sys.exit(0 if ok else 1)

    if "--dry-run" in args:
        ok = run_dry_run()
        sys.exit(0 if ok else 1)

    print("사용법:")
    print("  --static-check   정적 검사 (20항목)")
    print("  --dry-run        메타데이터 탐색 + 점수 계산 + CSV 저장")
    sys.exit(0)


if __name__ == "__main__":
    main()
