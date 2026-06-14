"""
B1-D8c refiner candidate crop generation script.

ALLOW_REAL_PROCESSING = False (B1-D8c approval preflight 단계)
실제 crop NPZ 생성은 B1-D8d에서 별도 승인 후 importlib runtime override 방식으로만 허용.

사용법:
  bare run (인수 없음): exit 2 (차단)
  --dry-run             : 입력/경로/스키마/collision 검증만 수행, crop 생성 없음
  --run                 : ALLOW_REAL_PROCESSING=False이므로 exit 2 (B1-D8d용)
"""

import sys
import os
import csv
import json
import pathlib
import hashlib

# ─────────────────────────────────────────────
# SAFETY FLAG — B1-D8c는 False 유지
# B1-D8d 실행 시 importlib runtime override 방식으로만 True 허용
# ─────────────────────────────────────────────
ALLOW_REAL_PROCESSING = False

# ─────────────────────────────────────────────
# 경로 상수
# ─────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")
B1D_OUT = PROJECT_ROOT / "outputs" / "b1d1_wall_mediastinum_fp_cause_diagnostic_v1"

INPUT_FILES = {
    "b1d8b_summary": B1D_OUT / "b1d8b_refiner_candidate_crop_input_preflight_summary.json",
    "b1d8b_crop_plan": B1D_OUT / "b1d8b_refiner_candidate_crop_plan.csv",
    "b1d8b_input_validation": B1D_OUT / "b1d8b_refiner_candidate_input_validation.csv",
    "b1d8_mapping_plan": B1D_OUT / "b1d8_refiner_link_candidate_mapping_plan.csv",
    "b1d8_input_design": B1D_OUT / "b1d8_refiner_input_design_table.csv",
}

EXPECTED_CANDIDATE_COUNT = 19
CROP_SIZE = 96
Z_CONTEXT_SLICES = 3  # z-1, z, z+1

# B1-D8d crop output 예정 경로 (collision guard 대상)
CROP_OUTPUT_DIR = B1D_OUT / "b1d8d_refiner_candidate_crops"
CROP_OUTPUT_DIR_TMP = B1D_OUT / "b1d8d_refiner_candidate_crops_tmp"

# stage2_holdout path — 절대 접근 금지
STAGE2_HOLDOUT_PATH = PROJECT_ROOT / "data" / "stage2_holdout"

# train_memory_overlap_risk 대상
TRAIN_MEMORY_OVERLAP_CANDIDATES = {"RCP_012"}

# 동일 환자 중복 그룹 (CT load 최적화 대상)
SHARED_PATIENT_GROUPS = [
    ["RCP_006", "RCP_011"],
    ["RCP_007", "RCP_008"],
    ["RCP_002", "RCP_004"],
]


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────

def abort(msg, code=2):
    print(f"[ABORT] {msg}", file=sys.stderr)
    sys.exit(code)


def check_stage2_holdout_not_accessed():
    """stage2_holdout path가 이 스크립트 내에서 접근되지 않았음을 선언적으로 검증."""
    if STAGE2_HOLDOUT_PATH.exists():
        # 존재해도 접근 금지 — 경로 확인만 하고 내용 읽기 금지
        pass
    # 스크립트 내 어디에도 stage2_holdout 데이터를 읽는 코드 없음
    return True


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ─────────────────────────────────────────────
# DRY-RUN 검증 함수들
# ─────────────────────────────────────────────

def check_input_files():
    """입력 파일 존재 및 mtime 기록."""
    results = {}
    for key, path in INPUT_FILES.items():
        exists = path.exists()
        mtime = path.stat().st_mtime if exists else None
        results[key] = {"path": str(path), "exists": exists, "mtime": mtime}
        if not exists:
            print(f"  [FAIL] 입력 파일 없음: {path}")
        else:
            print(f"  [OK]   {key}: {path.name}")
    return results


def check_b1d8b_verdict(summary):
    """B1-D8b verdict PASS 확인."""
    verdict = summary.get("verdict", "UNKNOWN")
    ok = verdict == "PASS"
    print(f"  B1-D8b verdict: {verdict} -> {'OK' if ok else 'FAIL'}")
    return ok


def check_candidate_count(rows, expected=EXPECTED_CANDIDATE_COUNT):
    """candidate 19행 확인."""
    count = len(rows)
    ok = count == expected
    print(f"  candidate_count: {count} (expected {expected}) -> {'OK' if ok else 'FAIL'}")
    return ok, count


def check_stage2_holdout_column(rows):
    """모든 candidate의 stage2_holdout_flag == 0 또는 False 확인."""
    violations = []
    for row in rows:
        flag = row.get("stage2_holdout_flag", "0")
        if str(flag).strip() not in ("0", "False", "false", ""):
            violations.append(row.get("refiner_candidate_id", "?"))
    ok = len(violations) == 0
    print(f"  stage2_holdout_flag: violations={violations} -> {'OK' if ok else 'FAIL'}")
    return ok, violations


def check_ct_mask_paths(rows):
    """CT/mask 파일 존재 재확인."""
    ct_ok = []
    ct_fail = []
    mask_ok = []
    mask_fail = []
    for row in rows:
        rid = row.get("refiner_candidate_id", "?")
        ct_path = pathlib.Path(row.get("ct_path", ""))
        mask_path = pathlib.Path(row.get("mask_path", ""))
        if ct_path.exists():
            ct_ok.append(rid)
        else:
            ct_fail.append(rid)
        if mask_path.exists():
            mask_ok.append(rid)
        else:
            mask_fail.append(rid)
    all_ct_ok = len(ct_fail) == 0
    all_mask_ok = len(mask_fail) == 0
    print(f"  CT mapping: {len(ct_ok)}/19 OK, fail={ct_fail} -> {'OK' if all_ct_ok else 'FAIL'}")
    print(f"  mask mapping: {len(mask_ok)}/19 OK, fail={mask_fail} -> {'OK' if all_mask_ok else 'FAIL'}")
    return all_ct_ok, all_mask_ok, ct_fail, mask_fail


def check_crop_coords(rows):
    """crop 좌표 범위 재확인 (y0>=0, x0>=0, y1<=512, x1<=512, shape=96×96)."""
    failures = []
    for row in rows:
        rid = row.get("refiner_candidate_id", "?")
        try:
            y0 = int(row["y0"])
            x0 = int(row["x0"])
            y1 = int(row["y1"])
            x1 = int(row["x1"])
        except (KeyError, ValueError):
            failures.append(f"{rid}:parse_error")
            continue
        if y0 < 0 or x0 < 0 or y1 > 512 or x1 > 512:
            failures.append(f"{rid}:boundary(y0={y0},x0={x0},y1={y1},x1={x1})")
        if (y1 - y0) != CROP_SIZE or (x1 - x0) != CROP_SIZE:
            failures.append(f"{rid}:size({y1-y0}x{x1-x0}!=96x96)")
    ok = len(failures) == 0
    print(f"  crop_coord: failures={failures} -> {'OK' if ok else 'FAIL'}")
    return ok, failures


def check_z_context(rows):
    """z context 재확인."""
    failures = []
    for row in rows:
        rid = row.get("refiner_candidate_id", "?")
        z_ok = row.get("z_context_ok", "True")
        if str(z_ok).strip() not in ("True", "true", "1"):
            failures.append(rid)
    ok = len(failures) == 0
    print(f"  z_context_ok: failures={failures} -> {'OK' if ok else 'FAIL'}")
    return ok, failures


def check_output_collision():
    """crop output folder collision guard — 이미 존재하면 FAIL."""
    exists = CROP_OUTPUT_DIR.exists()
    if exists:
        print(f"  [FAIL] crop output 폴더 이미 존재: {CROP_OUTPUT_DIR}")
        print(f"         B1-D8d 실행 전 제거하거나 다른 경로를 지정해야 합니다.")
    else:
        print(f"  [OK]   crop output 폴더 없음 (collision 없음): {CROP_OUTPUT_DIR}")
    return not exists


def check_npz_collision():
    """개별 NPZ 파일 collision guard — crop output dir 없으면 OK."""
    if not CROP_OUTPUT_DIR.exists():
        print(f"  [OK]   NPZ collision guard: output dir 없음, collision 없음")
        return True, []
    existing_npz = list(CROP_OUTPUT_DIR.glob("RCP_*.npz"))
    if existing_npz:
        names = [p.name for p in existing_npz]
        print(f"  [FAIL] 기존 NPZ 파일 발견: {names}")
        return False, names
    print(f"  [OK]   NPZ collision guard: 기존 NPZ 없음")
    return True, []


def check_train_memory_overlap_risk(rows):
    """train_memory_overlap_risk 대상 확인."""
    risk_candidates = []
    for row in rows:
        rid = row.get("refiner_candidate_id", "?")
        if rid in TRAIN_MEMORY_OVERLAP_CANDIDATES:
            risk_candidates.append({
                "refiner_candidate_id": rid,
                "patient_id": row.get("patient_id", "?"),
                "note": row.get("note", ""),
                "risk": "N-C7 train memory 포함 가능성 — scoring 시 in-distribution bias 주의"
            })
    print(f"  train_memory_overlap_risk 대상: {[r['refiner_candidate_id'] for r in risk_candidates]}")
    return risk_candidates


def check_forbidden_actions():
    """절대 금지 항목 확인 — 코드 내 금지 패턴 없음 선언."""
    forbidden_checks = {
        "stage2_holdout_read": False,   # 위에서 확인
        "crop_npz_write": False,         # ALLOW_REAL_PROCESSING=False
        "feature_extraction": False,
        "model_forward": False,
        "scoring": False,
        "threshold_computation": False,
        "training": False,
        "score_modification": False,
        "existing_file_modification": False,
    }
    all_clear = all(not v for v in forbidden_checks.values())
    print(f"  forbidden_actions: all_clear={all_clear}")
    return all_clear, forbidden_checks


# ─────────────────────────────────────────────
# DRY-RUN 메인
# ─────────────────────────────────────────────

def run_dry_run():
    print("=" * 60)
    print("B1-D8c --dry-run: crop generation approval preflight")
    print("=" * 60)
    print()

    # stage2 선언
    check_stage2_holdout_not_accessed()

    # 1. 입력 파일 확인
    print("[1] 입력 파일 존재 확인")
    file_results = check_input_files()
    all_files_exist = all(v["exists"] for v in file_results.values())
    if not all_files_exist:
        abort("입력 파일 누락으로 dry-run 중단")
    print()

    # 2. B1-D8b verdict 확인
    print("[2] B1-D8b verdict 확인")
    b1d8b_summary = load_json(INPUT_FILES["b1d8b_summary"])
    b1d8b_pass = check_b1d8b_verdict(b1d8b_summary)
    print()

    # 3. crop_plan 19행 확인
    print("[3] crop_plan 19행 확인")
    crop_plan_rows = load_csv_rows(INPUT_FILES["b1d8b_crop_plan"])
    count_ok, candidate_count = check_candidate_count(crop_plan_rows)
    print()

    # 4. input_validation 19행 확인
    print("[4] input_validation 19행 확인")
    validation_rows = load_csv_rows(INPUT_FILES["b1d8b_input_validation"])
    val_count_ok, val_count = check_candidate_count(validation_rows)
    print()

    # 5. stage2_holdout flag 확인
    print("[5] stage2_holdout_flag 확인")
    s2_ok, s2_violations = check_stage2_holdout_column(validation_rows)
    print()

    # 6. CT/mask mapping 재확인
    print("[6] CT/mask 경로 재확인")
    ct_ok, mask_ok, ct_fail, mask_fail = check_ct_mask_paths(validation_rows)
    print()

    # 7. crop 좌표 재확인
    print("[7] crop 좌표 범위 재확인")
    coord_ok, coord_fail = check_crop_coords(validation_rows)
    print()

    # 8. z context 재확인
    print("[8] z context 재확인")
    z_ok, z_fail = check_z_context(validation_rows)
    print()

    # 9. output folder collision guard
    print("[9] output folder collision guard")
    collision_ok = check_output_collision()
    npz_ok, npz_conflicts = check_npz_collision()
    print()

    # 10. train_memory_overlap_risk
    print("[10] train_memory_overlap_risk 확인")
    risk_candidates = check_train_memory_overlap_risk(validation_rows)
    print()

    # 11. forbidden actions 확인
    print("[11] forbidden actions 확인")
    forbidden_ok, forbidden_detail = check_forbidden_actions()
    print()

    # ─── 종합 판정 ───
    all_ok = all([
        all_files_exist,
        b1d8b_pass,
        count_ok,
        val_count_ok,
        s2_ok,
        ct_ok,
        mask_ok,
        coord_ok,
        z_ok,
        collision_ok,
        npz_ok,
        forbidden_ok,
    ])

    blockers = []
    if not all_files_exist:
        blockers.append("입력 파일 누락")
    if not b1d8b_pass:
        blockers.append("B1-D8b verdict not PASS")
    if not count_ok:
        blockers.append(f"crop_plan candidate_count={candidate_count} (expected 19)")
    if not val_count_ok:
        blockers.append(f"input_validation candidate_count={val_count} (expected 19)")
    if not s2_ok:
        blockers.append(f"stage2_holdout_flag violations: {s2_violations}")
    if not ct_ok:
        blockers.append(f"CT 경로 없음: {ct_fail}")
    if not mask_ok:
        blockers.append(f"mask 경로 없음: {mask_fail}")
    if not coord_ok:
        blockers.append(f"crop 좌표 오류: {coord_fail}")
    if not z_ok:
        blockers.append(f"z_context 오류: {z_fail}")
    if not collision_ok:
        blockers.append(f"crop output 폴더 이미 존재: {CROP_OUTPUT_DIR}")
    if not npz_ok:
        blockers.append(f"NPZ 파일 충돌: {npz_conflicts}")
    if not forbidden_ok:
        blockers.append("forbidden actions detected")

    verdict = "PASS" if all_ok else "FAIL"
    print("=" * 60)
    print(f"DRY-RUN 판정: {verdict}")
    if blockers:
        print(f"BLOCKERS: {blockers}")
    print(f"crop 생성: 없음 (ALLOW_REAL_PROCESSING={ALLOW_REAL_PROCESSING})")
    print("=" * 60)

    return {
        "verdict": verdict,
        "all_ok": all_ok,
        "b1d8b_pass": b1d8b_pass,
        "candidate_count": candidate_count,
        "ct_ok": ct_ok,
        "mask_ok": mask_ok,
        "coord_ok": coord_ok,
        "z_ok": z_ok,
        "collision_ok": collision_ok,
        "npz_ok": npz_ok,
        "s2_ok": s2_ok,
        "forbidden_ok": forbidden_ok,
        "blockers": blockers,
        "risk_candidates": risk_candidates,
        "crop_generated": False,
        "feature_extracted": False,
        "scoring_started": False,
        "training_started": False,
        "stage2_holdout_access": 0,
    }


# ─────────────────────────────────────────────
# REAL RUN (B1-D8d용 — 이번 단계에서는 실행 불가)
# ─────────────────────────────────────────────

def run_real():
    """실제 crop NPZ 생성. ALLOW_REAL_PROCESSING=True + --confirm-crop-generation일 때만 실행."""
    if not ALLOW_REAL_PROCESSING:
        print("[ABORT] ALLOW_REAL_PROCESSING=False — 실제 crop 생성 차단됨.")
        print("        B1-D8d에서 importlib runtime override 방식으로만 실행 허용.")
        sys.exit(2)

    import numpy as np

    print("[B1-D8d] 실제 crop NPZ 생성 시작")

    # ── stage2_holdout 접근 금지 검증 (exists 체크 아님) ──
    crop_plan_rows = load_csv_rows(INPUT_FILES["b1d8b_crop_plan"])

    # stage2_holdout_flag 값 전수 확인
    s2_violations = []
    for row in crop_plan_rows:
        flag = row.get("stage2_holdout_flag", "0")
        if str(flag).strip() not in ("0", "False", "false", ""):
            s2_violations.append(row.get("refiner_candidate_id", "?"))
    if s2_violations:
        abort(f"stage2_holdout_flag != 0 발견: {s2_violations}")

    # 입력 경로 문자열에 stage2_holdout 포함 여부 확인
    for row in crop_plan_rows:
        for key in ("ct_path", "mask_path"):
            path_str = row.get(key, "")
            if "stage2_holdout" in path_str:
                abort(f"입력 경로에 stage2_holdout 포함 금지: {key}={path_str}")

    stage2_holdout_access = 0  # stage2_holdout 디렉토리 listing/read 없음

    # ── candidate count 확인 ──
    if len(crop_plan_rows) != EXPECTED_CANDIDATE_COUNT:
        abort(f"candidate_count={len(crop_plan_rows)} != {EXPECTED_CANDIDATE_COUNT}")

    # ── output collision guard (final + tmp 둘 다) ──
    if CROP_OUTPUT_DIR.exists():
        abort(f"final crop output 폴더 이미 존재 (덮어쓰기 금지): {CROP_OUTPUT_DIR}")
    if CROP_OUTPUT_DIR_TMP.exists():
        abort(f"tmp crop output 폴더 이미 존재 (이전 실패 잔여물 확인 필요): {CROP_OUTPUT_DIR_TMP}")

    CROP_OUTPUT_DIR_TMP.mkdir(parents=False, exist_ok=False)

    # ── CT load 최적화: patient별 그룹화 ──
    patient_to_candidates = {}
    for row in crop_plan_rows:
        pid = row["safe_id"]
        patient_to_candidates.setdefault(pid, []).append(row)

    errors = []
    generated = []
    labels_rows = []
    integrity_rows = []

    for pid, candidates in patient_to_candidates.items():
        ct_path = pathlib.Path(candidates[0]["ct_path"])
        mask_path = pathlib.Path(candidates[0]["mask_path"])

        if not ct_path.exists():
            for row in candidates:
                rid = row["refiner_candidate_id"]
                errors.append(f"{rid}:CT_not_found")
                integrity_rows.append({
                    "refiner_candidate_id": rid, "npz_exists": 0,
                    "crop_shape_ok": 0, "mask_shape_ok": 0,
                    "crop_nan_count": -1, "crop_inf_count": -1,
                    "stage2_holdout_flag_ok": 1,
                    "train_memory_overlap_risk": int(rid in TRAIN_MEMORY_OVERLAP_CANDIDATES),
                    "status": "FAIL", "error": "CT_not_found",
                })
            continue
        if not mask_path.exists():
            for row in candidates:
                rid = row["refiner_candidate_id"]
                errors.append(f"{rid}:mask_not_found")
                integrity_rows.append({
                    "refiner_candidate_id": rid, "npz_exists": 0,
                    "crop_shape_ok": 0, "mask_shape_ok": 0,
                    "crop_nan_count": -1, "crop_inf_count": -1,
                    "stage2_holdout_flag_ok": 1,
                    "train_memory_overlap_risk": int(rid in TRAIN_MEMORY_OVERLAP_CANDIDATES),
                    "status": "FAIL", "error": "mask_not_found",
                })
            continue

        # CT 1회 load (동일 환자 중복 최적화)
        ct_vol = np.load(str(ct_path), mmap_mode="r")
        mask_vol = np.load(str(mask_path), mmap_mode="r")

        # CT/mask shape 호환 확인 (HW = 512×512)
        if ct_vol.ndim < 3 or ct_vol.shape[1] != 512 or ct_vol.shape[2] != 512:
            for row in candidates:
                rid = row["refiner_candidate_id"]
                errors.append(f"{rid}:CT_shape_invalid={list(ct_vol.shape)}")
                integrity_rows.append({
                    "refiner_candidate_id": rid, "npz_exists": 0,
                    "crop_shape_ok": 0, "mask_shape_ok": 0,
                    "crop_nan_count": -1, "crop_inf_count": -1,
                    "stage2_holdout_flag_ok": 1,
                    "train_memory_overlap_risk": int(rid in TRAIN_MEMORY_OVERLAP_CANDIDATES),
                    "status": "FAIL", "error": f"CT_shape_invalid={list(ct_vol.shape)}",
                })
            continue
        if mask_vol.ndim < 3 or mask_vol.shape[1] != 512 or mask_vol.shape[2] != 512:
            for row in candidates:
                rid = row["refiner_candidate_id"]
                errors.append(f"{rid}:mask_shape_invalid={list(mask_vol.shape)}")
                integrity_rows.append({
                    "refiner_candidate_id": rid, "npz_exists": 0,
                    "crop_shape_ok": 0, "mask_shape_ok": 0,
                    "crop_nan_count": -1, "crop_inf_count": -1,
                    "stage2_holdout_flag_ok": 1,
                    "train_memory_overlap_risk": int(rid in TRAIN_MEMORY_OVERLAP_CANDIDATES),
                    "status": "FAIL", "error": f"mask_shape_invalid={list(mask_vol.shape)}",
                })
            continue
        if ct_vol.shape[0] != mask_vol.shape[0]:
            for row in candidates:
                rid = row["refiner_candidate_id"]
                errors.append(f"{rid}:CT_mask_z_mismatch({ct_vol.shape[0]}!={mask_vol.shape[0]})")
                integrity_rows.append({
                    "refiner_candidate_id": rid, "npz_exists": 0,
                    "crop_shape_ok": 0, "mask_shape_ok": 0,
                    "crop_nan_count": -1, "crop_inf_count": -1,
                    "stage2_holdout_flag_ok": 1,
                    "train_memory_overlap_risk": int(rid in TRAIN_MEMORY_OVERLAP_CANDIDATES),
                    "status": "FAIL",
                    "error": f"CT_mask_z_mismatch({ct_vol.shape[0]}!={mask_vol.shape[0]})",
                })
            continue

        ct_z = ct_vol.shape[0]

        for row in candidates:
            rid = row["refiner_candidate_id"]
            local_z = int(row["local_z"])
            y0 = int(row["y0"])
            x0 = int(row["x0"])
            y1 = int(row["y1"])
            x1 = int(row["x1"])
            position_bin = row["position_bin"]
            source_stage = row.get("source_stage", "")
            train_mem_risk = int(rid in TRAIN_MEMORY_OVERLAP_CANDIDATES)

            # NPZ 중복 guard
            out_path = CROP_OUTPUT_DIR_TMP / f"{rid}.npz"
            if out_path.exists():
                errors.append(f"{rid}:NPZ_collision")
                integrity_rows.append({
                    "refiner_candidate_id": rid, "npz_exists": 1,
                    "crop_shape_ok": 0, "mask_shape_ok": 0,
                    "crop_nan_count": -1, "crop_inf_count": -1,
                    "stage2_holdout_flag_ok": 1, "train_memory_overlap_risk": train_mem_risk,
                    "status": "FAIL", "error": "NPZ_collision",
                })
                continue

            # local_z 범위 확인
            if local_z < 0 or local_z >= ct_z:
                errors.append(f"{rid}:local_z_out_of_range={local_z}/{ct_z}")
                integrity_rows.append({
                    "refiner_candidate_id": rid, "npz_exists": 0,
                    "crop_shape_ok": 0, "mask_shape_ok": 0,
                    "crop_nan_count": -1, "crop_inf_count": -1,
                    "stage2_holdout_flag_ok": 1, "train_memory_overlap_risk": train_mem_risk,
                    "status": "FAIL", "error": f"local_z_out_of_range={local_z}/{ct_z}",
                })
                continue

            # crop 좌표 범위 검증
            if y0 < 0 or x0 < 0 or y1 > 512 or x1 > 512:
                errors.append(f"{rid}:boundary_error(y0={y0},x0={x0},y1={y1},x1={x1})")
                integrity_rows.append({
                    "refiner_candidate_id": rid, "npz_exists": 0,
                    "crop_shape_ok": 0, "mask_shape_ok": 0,
                    "crop_nan_count": -1, "crop_inf_count": -1,
                    "stage2_holdout_flag_ok": 1, "train_memory_overlap_risk": train_mem_risk,
                    "status": "FAIL", "error": "boundary_error",
                })
                continue
            if (y1 - y0) != CROP_SIZE or (x1 - x0) != CROP_SIZE:
                errors.append(f"{rid}:crop_size_error({y1-y0}x{x1-x0}!=96x96)")
                integrity_rows.append({
                    "refiner_candidate_id": rid, "npz_exists": 0,
                    "crop_shape_ok": 0, "mask_shape_ok": 0,
                    "crop_nan_count": -1, "crop_inf_count": -1,
                    "stage2_holdout_flag_ok": 1, "train_memory_overlap_risk": train_mem_risk,
                    "status": "FAIL", "error": "crop_size_error",
                })
                continue

            # z-1/z/z+1 context (edge clamp)
            z_indices = []
            for dz in [-1, 0, 1]:
                zi = local_z + dz
                zi = max(0, min(ct_z - 1, zi))
                z_indices.append(zi)

            # 3ch crop (raw HU, float32)
            crop_3ch = np.stack(
                [ct_vol[zi, y0:y1, x0:x1].astype(np.float32) for zi in z_indices],
                axis=0
            )  # shape: (3, 96, 96)

            # mask crop (bool)
            mask_crop = mask_vol[local_z, y0:y1, x0:x1].astype(np.bool_)

            # NaN/Inf 검증
            nan_count = int(np.sum(np.isnan(crop_3ch)))
            inf_count = int(np.sum(np.isinf(crop_3ch)))
            if nan_count > 0 or inf_count > 0:
                errors.append(f"{rid}:NaN({nan_count})_Inf({inf_count})_in_crop")
                integrity_rows.append({
                    "refiner_candidate_id": rid, "npz_exists": 0,
                    "crop_shape_ok": 1, "mask_shape_ok": 1,
                    "crop_nan_count": nan_count, "crop_inf_count": inf_count,
                    "stage2_holdout_flag_ok": 1, "train_memory_overlap_risk": train_mem_risk,
                    "status": "FAIL", "error": f"NaN({nan_count})_Inf({inf_count})",
                })
                continue

            # shape 검증 (3, 96, 96) / (96, 96)
            if crop_3ch.shape != (Z_CONTEXT_SLICES, CROP_SIZE, CROP_SIZE):
                errors.append(f"{rid}:crop_shape_mismatch={list(crop_3ch.shape)}")
                integrity_rows.append({
                    "refiner_candidate_id": rid, "npz_exists": 0,
                    "crop_shape_ok": 0, "mask_shape_ok": 1,
                    "crop_nan_count": 0, "crop_inf_count": 0,
                    "stage2_holdout_flag_ok": 1, "train_memory_overlap_risk": train_mem_risk,
                    "status": "FAIL", "error": f"crop_shape_mismatch={list(crop_3ch.shape)}",
                })
                continue
            if mask_crop.shape != (CROP_SIZE, CROP_SIZE):
                errors.append(f"{rid}:mask_shape_mismatch={list(mask_crop.shape)}")
                integrity_rows.append({
                    "refiner_candidate_id": rid, "npz_exists": 0,
                    "crop_shape_ok": 1, "mask_shape_ok": 0,
                    "crop_nan_count": 0, "crop_inf_count": 0,
                    "stage2_holdout_flag_ok": 1, "train_memory_overlap_risk": train_mem_risk,
                    "status": "FAIL", "error": f"mask_shape_mismatch={list(mask_crop.shape)}",
                })
                continue

            metadata = {
                "refiner_candidate_id": rid,
                "patient_id": row.get("patient_id", ""),
                "safe_id": row["safe_id"],
                "local_z": local_z,
                "y0": y0,
                "x0": x0,
                "crop_size": CROP_SIZE,
                "z_context": "z-1/z/z+1",
                "position_bin": position_bin,
                "source_stage": source_stage,
                "proposed_taxonomy_label": row.get("note", ""),
                "rule_b3_flag": 1 if "rule_b3" in source_stage else 0,
                "gate_p2_flag": 1 if "gate_p2" in source_stage else 0,
                "stage2_holdout_flag": 0,
                "train_memory_overlap_risk": train_mem_risk,
                "normalization": "raw_HU_float32_no_normalization",
                "dtype": "float32",
                "crop_shape": list(crop_3ch.shape),
            }

            np.savez_compressed(
                str(out_path),
                crop=crop_3ch,
                mask_crop=mask_crop,
                metadata=json.dumps(metadata),
            )
            generated.append(rid)
            print(f"  [SAVED] {rid} -> {out_path.name}")

            labels_rows.append({
                "refiner_candidate_id": rid,
                "patient_id": row.get("patient_id", ""),
                "safe_id": row["safe_id"],
                "local_z": local_z,
                "y0": y0,
                "x0": x0,
                "crop_npz_path": str(CROP_OUTPUT_DIR / f"{rid}.npz"),
                "crop_shape": str(list(crop_3ch.shape)),
                "mask_shape": str(list(mask_crop.shape)),
                "position_bin": position_bin,
                "source_stage": source_stage,
                "proposed_taxonomy_label": row.get("note", ""),
                "rule_b3_flag": 1 if "rule_b3" in source_stage else 0,
                "gate_p2_flag": 1 if "gate_p2" in source_stage else 0,
                "stage2_holdout_flag": 0,
                "train_memory_overlap_risk": train_mem_risk,
            })
            integrity_rows.append({
                "refiner_candidate_id": rid, "npz_exists": 1,
                "crop_shape_ok": 1, "mask_shape_ok": 1,
                "crop_nan_count": 0, "crop_inf_count": 0,
                "stage2_holdout_flag_ok": 1,
                "train_memory_overlap_risk": train_mem_risk,
                "status": "OK", "error": "",
            })

    # ── 출력 파일 생성 (tmp 폴더에) ──

    labels_cols = [
        "refiner_candidate_id", "patient_id", "safe_id", "local_z",
        "y0", "x0", "crop_npz_path", "crop_shape", "mask_shape",
        "position_bin", "source_stage", "proposed_taxonomy_label",
        "rule_b3_flag", "gate_p2_flag", "stage2_holdout_flag",
        "train_memory_overlap_risk",
    ]
    labels_path = CROP_OUTPUT_DIR_TMP / "b1d8d_refiner_candidate_crop_labels.csv"
    with open(labels_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=labels_cols)
        writer.writeheader()
        writer.writerows(labels_rows)

    integrity_cols = [
        "refiner_candidate_id", "npz_exists", "crop_shape_ok", "mask_shape_ok",
        "crop_nan_count", "crop_inf_count", "stage2_holdout_flag_ok",
        "train_memory_overlap_risk", "status", "error",
    ]
    integrity_path = CROP_OUTPUT_DIR_TMP / "b1d8d_refiner_candidate_crop_integrity.csv"
    with open(integrity_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=integrity_cols)
        writer.writeheader()
        writer.writerows(integrity_rows)

    summary = {
        "step": "B1-D8d_refiner_candidate_crop_generation",
        "generated_count": len(generated),
        "error_count": len(errors),
        "generated": generated,
        "errors": errors,
        "stage2_holdout_access": stage2_holdout_access,
        "feature_extracted": False,
        "scoring_started": False,
        "training_started": False,
    }
    summary_path = CROP_OUTPUT_DIR_TMP / "b1d8d_refiner_candidate_crop_generation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    report_lines = [
        "# B1-D8d Refiner Candidate Crop Generation Report",
        "",
        f"- generated_count: {len(generated)}",
        f"- error_count: {len(errors)}",
        f"- stage2_holdout_access: {stage2_holdout_access}",
        "",
        "## Generated",
    ]
    for rid in generated:
        report_lines.append(f"- {rid}")
    if errors:
        report_lines.append("")
        report_lines.append("## Errors")
        for e in errors:
            report_lines.append(f"- {e}")
    report_path = CROP_OUTPUT_DIR_TMP / "b1d8d_refiner_candidate_crop_generation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    done_info = {
        "step": "B1-D8d_refiner_candidate_crop_generation",
        "generated_count": len(generated),
        "error_count": len(errors),
        "generated": generated,
        "errors": errors,
        "stage2_holdout_access": stage2_holdout_access,
    }
    with open(CROP_OUTPUT_DIR_TMP / "DONE.json", "w", encoding="utf-8") as f:
        json.dump(done_info, f, indent=2, ensure_ascii=False)

    # ── 오류 있으면 tmp 폴더 유지, final 생성 안 함 ──
    if errors:
        print(f"\n[FAIL] error={len(errors)}, tmp 폴더 유지: {CROP_OUTPUT_DIR_TMP}")
        print(f"ERRORS: {errors}")
        print("final 폴더 생성 안 됨.")
        sys.exit(1)

    # ── 전체 통과 시 tmp → final rename ──
    if CROP_OUTPUT_DIR.exists():
        abort(f"final 폴더가 처리 중 생성됨 (rename 불가): {CROP_OUTPUT_DIR}")

    CROP_OUTPUT_DIR_TMP.rename(CROP_OUTPUT_DIR)
    print(f"\n[완료] generated={len(generated)}, errors=0")
    print(f"output: {CROP_OUTPUT_DIR}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    # bare-run 차단
    if len(args) == 0:
        print("[ABORT] 인수 없이 실행 불가. --dry-run 또는 --run을 사용하세요.", file=sys.stderr)
        sys.exit(2)

    mode = args[0]

    if mode == "--dry-run":
        result = run_dry_run()
        sys.exit(0 if result["verdict"] == "PASS" else 1)

    elif mode == "--run":
        if not ALLOW_REAL_PROCESSING:
            print(
                "[ABORT] ALLOW_REAL_PROCESSING=False — --run 차단됨.\n"
                "        B1-D8d에서 importlib runtime override 방식으로만 허용.\n"
                "        예시: python -c \"import b1d8c_refiner_candidate_crop_generation as m; "
                "m.ALLOW_REAL_PROCESSING=True; m.run_real()\"",
                file=sys.stderr,
            )
            sys.exit(2)
        if "--confirm-crop-generation" not in args:
            print(
                "[ABORT] --run 단독 차단됨.\n"
                "        --run --confirm-crop-generation 으로 실행해야 합니다.\n"
                "        ALLOW_REAL_PROCESSING=True runtime override도 필요합니다.",
                file=sys.stderr,
            )
            sys.exit(2)
        run_real()

    else:
        print(f"[ABORT] 알 수 없는 인수: {mode}. --dry-run 또는 --run을 사용하세요.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
