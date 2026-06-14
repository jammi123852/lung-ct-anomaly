"""
rule_s6a_manifest_validation.py
================================
S6-A selected candidate manifest의 crop 생성 전 유효성을 read-only로 검증하는 스크립트.

절대 금지 목록:
- crop / npz / PNG 생성 금지
- 모델 학습 / scoring 재실행 금지
- 기존 score / candidate / evaluation / crop 파일 수정 금지
- manifest 원본 수정 금지
- stage2_holdout 환자 사용 금지
- pip / conda install 금지
- npy 파일 전체 로드 금지 (shape 확인은 mmap_mode='r'로만)

실행 방식:
    python scripts/rule_s6a_manifest_validation.py            # preflight 보고만
    python scripts/rule_s6a_manifest_validation.py --run      # 실제 검증 + 출력 저장
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ─────────────────────────────────────────────
# 경로 상수 정의
# ─────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_CSV    = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/candidates/rule_s6a_gs2_selected_candidate_manifest_dryrun.csv"
SUMMARY_CSV     = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/rule_s6a_gs2_selected_candidate_manifest_summary.csv"
SUMMARY_JSON    = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/rule_s6a_gs2_selected_candidate_manifest_summary.json"
STAGE_SPLIT_CSV = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
PATHS_CONFIG    = REPO_ROOT / "configs/paths.local.yaml"

OUT_RPT_DIR = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports"
OUT_VAL_CSV  = OUT_RPT_DIR / "rule_s6a_manifest_validation_summary.csv"
OUT_VAL_JSON = OUT_RPT_DIR / "rule_s6a_manifest_validation_summary.json"
OUT_VAL_MD   = OUT_RPT_DIR / "rule_s6a_manifest_validation_summary.md"

EXPECTED_TOTAL_ROWS    = 130659
EXPECTED_N_PATIENTS    = 154
EXPECTED_N_POSITIVE    = 43553
EXPECTED_N_HARD_NEG    = 87106
EXPECTED_SAMPLING_RULE = "S6-A_positive_all_hn_ratio2"
EXPECTED_SAMPLING_LABELS = {"positive", "hard_negative"}
EXPLOSION_THRESHOLD    = 2000
CROP_SIZE              = 96
CROP_HALF              = CROP_SIZE // 2  # 48

REQUIRED_COLUMNS = [
    "patient_id", "local_z", "y0", "x0", "y1", "x1",
    "sampling_label", "sampling_rule",
]

SCORE_CANDIDATE_DIRS = [
    REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/candidates",
    REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops",
]


# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────
def load_volume_root() -> Path:
    """paths.local.yaml에서 nsclc_msd_usable_only_v2 경로를 읽는다."""
    with open(PATHS_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw = cfg.get("nsclc_msd_usable_only_v2", "")
    if not raw:
        raise ValueError("paths.local.yaml에 nsclc_msd_usable_only_v2 키가 없거나 비어 있습니다.")
    return Path(raw)


def build_safe_id_map(
    split_df: pd.DataFrame, vol_root
) -> "tuple[dict, str, int, int]":
    """safe_id 매핑 딕셔너리 생성.
    반환: (mapping_dict, method_name, n_success, n_fail)
    method: from_stage_split_safe_id | meta_json_reverse_lookup | patient_id_as_folder | failed
    """
    all_pids = set(split_df["patient_id"].dropna().tolist())

    # 방식 1: stage split에 safe_id 컬럼이 있고 비어있지 않으면 사용
    if "safe_id" in split_df.columns:
        raw_map = dict(zip(split_df["patient_id"], split_df["safe_id"]))
        mapping = {k: v for k, v in raw_map.items() if pd.notna(v) and v != ""}
        if mapping:
            n_success = len(set(mapping.keys()) & all_pids)
            n_fail = len(all_pids) - n_success
            return mapping, "from_stage_split_safe_id", n_success, n_fail

    # 방식 2: volumes_npy 각 폴더의 meta.json에서 patient_id → safe_id 역방향 조회
    if vol_root is not None:
        vol_npy_dir = Path(vol_root) / "volumes_npy"
        if vol_npy_dir.exists():
            mapping = {}
            for folder in vol_npy_dir.iterdir():
                if not folder.is_dir():
                    continue
                meta_path = folder / "meta.json"
                if not meta_path.exists():
                    continue
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    pid = d.get("patient_id")
                    sid = d.get("safe_id", folder.name)
                    if pid:
                        mapping[pid] = sid
                except Exception:
                    continue
            if mapping:
                n_success = len(set(mapping.keys()) & all_pids)
                n_fail = len(all_pids) - n_success
                return mapping, "meta_json_reverse_lookup", n_success, n_fail

    # 방식 3: patient_id를 그대로 폴더명으로 시도
    if vol_root is not None:
        vol_npy_dir = Path(vol_root) / "volumes_npy"
        if vol_npy_dir.exists():
            folder_names = {f.name for f in vol_npy_dir.iterdir() if f.is_dir()}
            mapping = {pid: pid for pid in all_pids if pid in folder_names}
            if mapping:
                n_success = len(mapping)
                n_fail = len(all_pids) - n_success
                return mapping, "patient_id_as_folder", n_success, n_fail

    # 실패
    return {}, "failed", 0, len(all_pids)


def make_result(check_id: int, name: str, status: str, detail: str) -> dict:
    """검증 항목 딕셔너리 생성. status: pass / fail / warn / info"""
    return {"check_id": check_id, "name": name, "status": status, "detail": detail}


def file_size_str(p: Path) -> str:
    if not p.exists():
        return "없음"
    sz = p.stat().st_size
    if sz >= 1_000_000:
        return f"{sz/1_000_000:.1f} MB"
    elif sz >= 1_000:
        return f"{sz/1_000:.1f} KB"
    return f"{sz} B"


# ─────────────────────────────────────────────
# guard_check
# ─────────────────────────────────────────────
def guard_check():
    """출력 파일 중복 및 입력 파일 존재 여부 확인."""
    errors = []

    # 출력 파일 중복 방지
    for out_path in [OUT_VAL_CSV, OUT_VAL_JSON, OUT_VAL_MD]:
        if out_path.exists():
            errors.append(f"[GUARD] 출력 파일이 이미 존재합니다: {out_path}")

    if errors:
        for e in errors:
            print(e)
        print("\n[중단] 기존 출력 파일 덮어쓰기 방지를 위해 종료합니다.")
        print("       기존 파일을 삭제하거나 이름을 변경한 뒤 재실행하세요.")
        sys.exit(1)

    # 입력 파일 존재 확인
    missing = []
    for p in [MANIFEST_CSV, SUMMARY_CSV, SUMMARY_JSON, STAGE_SPLIT_CSV]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("[GUARD] 아래 입력 파일이 없습니다:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    # PATHS_CONFIG 존재 확인
    if not PATHS_CONFIG.exists():
        print(f"[GUARD] paths.local.yaml 없음: {PATHS_CONFIG}")
        sys.exit(1)

    # VOLUME_ROOT 존재 확인
    try:
        vol_root = load_volume_root()
        if not vol_root.exists():
            print(f"[GUARD] VOLUME_ROOT 경로가 존재하지 않습니다: {vol_root}")
            sys.exit(1)
    except Exception as ex:
        print(f"[GUARD] VOLUME_ROOT 로드 실패: {ex}")
        sys.exit(1)

    print("[GUARD] 모든 guard 조건 통과.")


# ─────────────────────────────────────────────
# preflight_report
# ─────────────────────────────────────────────
def preflight_report():
    print("=" * 70)
    print("  rule_s6a_manifest_validation.py  —  PREFLIGHT 보고")
    print("=" * 70)
    print()

    print("[입력 파일]")
    for label, p in [
        ("MANIFEST_CSV   ", MANIFEST_CSV),
        ("SUMMARY_CSV    ", SUMMARY_CSV),
        ("SUMMARY_JSON   ", SUMMARY_JSON),
        ("STAGE_SPLIT_CSV", STAGE_SPLIT_CSV),
        ("PATHS_CONFIG   ", PATHS_CONFIG),
    ]:
        exists = "O" if p.exists() else "X"
        size   = file_size_str(p)
        print(f"  [{exists}] {label}  {size:>10}  {p}")
    print()

    try:
        vol_root = load_volume_root()
        vol_exists = "O" if vol_root.exists() else "X"
        print(f"  [{vol_exists}] VOLUME_ROOT  {vol_root}")
    except Exception as ex:
        print(f"  [X] VOLUME_ROOT  로드 실패: {ex}")
    print()

    print("[stage split 정보]")
    try:
        _split_df = pd.read_csv(STAGE_SPLIT_CSV)
        split_cols = list(_split_df.columns)
        has_safe_id = "safe_id" in split_cols
        print(f"  컬럼 목록: {split_cols}")
        print(f"  safe_id 컬럼 존재: {has_safe_id}")
        if has_safe_id:
            print("  safe_id 매핑 예상 방식: from_stage_split_safe_id")
        else:
            print("  safe_id 컬럼 없음 → fallback: meta_json_reverse_lookup / patient_id_as_folder 순으로 시도 예정")
    except Exception as _ex:
        print(f"  stage split 읽기 실패: {_ex}")
    print()

    print("[출력 파일]")
    for p in [OUT_VAL_CSV, OUT_VAL_JSON, OUT_VAL_MD]:
        exists = "이미 존재 (주의)" if p.exists() else "미생성"
        print(f"  {p}  [{exists}]")
    print()

    print("[예상 소요 시간]  5~15분  (volume shape mmap 154명)")
    print("[메모리 위험]     낮음  (npy mmap_mode='r', manifest 전체 로드 ~130K rows)")
    print()
    print("[실행 명령]")
    print("  source ~/ai_env/bin/activate && python scripts/rule_s6a_manifest_validation.py --run")
    print()
    print("--run 없이 실행 시 preflight 보고만 하고 종료합니다.")
    print("=" * 70)


# ─────────────────────────────────────────────
# 검증 함수들
# ─────────────────────────────────────────────

def check_01_manifest_exists(results: list):
    cid, name = 1, "manifest_exists"
    if MANIFEST_CSV.exists():
        results.append(make_result(cid, name, "pass", str(MANIFEST_CSV)))
    else:
        results.append(make_result(cid, name, "fail", f"파일 없음: {MANIFEST_CSV}"))


def check_02_manifest_row_count(results: list, df: pd.DataFrame):
    cid, name = 2, "manifest_row_count"
    n = len(df)
    if n == EXPECTED_TOTAL_ROWS:
        results.append(make_result(cid, name, "pass", f"row 수 = {n} (기대값 {EXPECTED_TOTAL_ROWS})"))
    else:
        results.append(make_result(cid, name, "fail",
            f"row 수 = {n}, 기대값 = {EXPECTED_TOTAL_ROWS}, 차이 = {n - EXPECTED_TOTAL_ROWS:+d}"))


def check_03_patient_count(results: list, df: pd.DataFrame):
    cid, name = 3, "patient_count"
    n = df["patient_id"].nunique()
    if n == EXPECTED_N_PATIENTS:
        results.append(make_result(cid, name, "pass", f"환자 수 = {n}"))
    else:
        results.append(make_result(cid, name, "fail",
            f"환자 수 = {n}, 기대값 = {EXPECTED_N_PATIENTS}"))


def check_04_stage_split_all_dev(results: list, df: pd.DataFrame):
    cid, name = 4, "stage_split_all_dev"
    if "stage_split" not in df.columns:
        results.append(make_result(cid, name, "fail", "stage_split 컬럼 없음"))
        return
    vals = df["stage_split"].unique().tolist()
    non_dev = [v for v in vals if v != "stage1_dev"]
    if not non_dev:
        results.append(make_result(cid, name, "pass", f"전부 stage1_dev. unique={vals}"))
    else:
        cnt = df[df["stage_split"] != "stage1_dev"].shape[0]
        results.append(make_result(cid, name, "fail",
            f"stage1_dev 아닌 값 존재: {non_dev}, 해당 row 수={cnt}"))


def check_05_no_stage2_holdout(results: list, df: pd.DataFrame, split_df: pd.DataFrame):
    cid, name = 5, "no_stage2_holdout_patients"
    holdout_ids = set(split_df[split_df["stage_split"] == "stage2_holdout"]["patient_id"].tolist())
    manifest_ids = set(df["patient_id"].unique().tolist())
    overlap = holdout_ids & manifest_ids
    if not overlap:
        results.append(make_result(cid, name, "pass",
            f"stage2_holdout 환자 0명. (holdout 전체 {len(holdout_ids)}명)"))
    else:
        results.append(make_result(cid, name, "fail",
            f"stage2_holdout 환자 {len(overlap)}명이 manifest에 포함됨: {sorted(overlap)[:10]}"))


def check_06_sampling_rule(results: list, df: pd.DataFrame):
    cid, name = 6, "sampling_rule_uniform"
    if "sampling_rule" not in df.columns:
        results.append(make_result(cid, name, "fail", "sampling_rule 컬럼 없음"))
        return
    vals = df["sampling_rule"].unique().tolist()
    if vals == [EXPECTED_SAMPLING_RULE]:
        results.append(make_result(cid, name, "pass", f"전부 '{EXPECTED_SAMPLING_RULE}'"))
    else:
        results.append(make_result(cid, name, "fail",
            f"예상 외 값 존재: {vals}"))


def check_07_sampling_label_values(results: list, df: pd.DataFrame):
    cid, name = 7, "sampling_label_values"
    if "sampling_label" not in df.columns:
        results.append(make_result(cid, name, "fail", "sampling_label 컬럼 없음"))
        return
    vals = set(df["sampling_label"].unique().tolist())
    unexpected = vals - EXPECTED_SAMPLING_LABELS
    if not unexpected:
        results.append(make_result(cid, name, "pass",
            f"값 집합 = {sorted(vals)} (기대: {sorted(EXPECTED_SAMPLING_LABELS)})"))
    else:
        results.append(make_result(cid, name, "fail",
            f"예상 외 값: {unexpected}. 전체 unique={sorted(vals)}"))


def check_08_positive_count(results: list, df: pd.DataFrame):
    cid, name = 8, "positive_count"
    n = (df["sampling_label"] == "positive").sum()
    if n == EXPECTED_N_POSITIVE:
        results.append(make_result(cid, name, "pass", f"positive = {n}"))
    else:
        results.append(make_result(cid, name, "fail",
            f"positive = {n}, 기대값 = {EXPECTED_N_POSITIVE}, 차이 = {n - EXPECTED_N_POSITIVE:+d}"))


def check_09_hard_negative_count(results: list, df: pd.DataFrame):
    cid, name = 9, "hard_negative_count"
    n = (df["sampling_label"] == "hard_negative").sum()
    if n == EXPECTED_N_HARD_NEG:
        results.append(make_result(cid, name, "pass", f"hard_negative = {n}"))
    else:
        results.append(make_result(cid, name, "fail",
            f"hard_negative = {n}, 기대값 = {EXPECTED_N_HARD_NEG}, 차이 = {n - EXPECTED_N_HARD_NEG:+d}"))


def check_10_duplicate_keys(results: list, df: pd.DataFrame):
    cid, name = 10, "no_duplicate_keys"
    key_cols = ["patient_id", "local_z", "y0", "x0", "y1", "x1"]
    missing_cols = [c for c in key_cols if c not in df.columns]
    if missing_cols:
        results.append(make_result(cid, name, "fail", f"key 컬럼 없음: {missing_cols}"))
        return
    dup_count = df.duplicated(subset=key_cols).sum()
    if dup_count == 0:
        results.append(make_result(cid, name, "pass", "중복 key 없음"))
    else:
        results.append(make_result(cid, name, "fail",
            f"중복 key {dup_count}개 존재"))


def check_11_local_z_valid(results: list, df: pd.DataFrame):
    cid, name = 11, "local_z_valid"
    if "local_z" not in df.columns:
        results.append(make_result(cid, name, "fail", "local_z 컬럼 없음"))
        return
    nan_count = df["local_z"].isna().sum()
    neg_count = (df["local_z"] < 0).sum()
    issues = []
    if nan_count > 0:
        issues.append(f"NaN={nan_count}")
    if neg_count > 0:
        issues.append(f"음수={neg_count}")
    if issues:
        results.append(make_result(cid, name, "fail", ", ".join(issues)))
    else:
        results.append(make_result(cid, name, "pass",
            f"local_z 유효 (min={df['local_z'].min()}, max={df['local_z'].max()})"))


def check_12_bbox_nan(results: list, df: pd.DataFrame):
    cid, name = 12, "bbox_no_nan"
    coord_cols = ["y0", "x0", "y1", "x1"]
    missing_cols = [c for c in coord_cols if c not in df.columns]
    if missing_cols:
        results.append(make_result(cid, name, "fail", f"컬럼 없음: {missing_cols}"))
        return
    nan_counts = {c: int(df[c].isna().sum()) for c in coord_cols}
    total_nan = sum(nan_counts.values())
    if total_nan == 0:
        results.append(make_result(cid, name, "pass", "y0/x0/y1/x1 NaN 없음"))
    else:
        results.append(make_result(cid, name, "fail", f"NaN 존재: {nan_counts}"))


def check_13_bbox_nonneg(results: list, df: pd.DataFrame):
    cid, name = 13, "bbox_nonneg"
    coord_cols = ["y0", "x0", "y1", "x1"]
    if any(c not in df.columns for c in coord_cols):
        results.append(make_result(cid, name, "fail", "bbox 컬럼 없음, check_12 참조"))
        return
    neg_counts = {c: int((df[c] < 0).sum()) for c in coord_cols}
    total_neg = sum(neg_counts.values())
    if total_neg == 0:
        results.append(make_result(cid, name, "pass", "모든 bbox 좌표 >= 0"))
    else:
        results.append(make_result(cid, name, "fail", f"음수 좌표: {neg_counts}"))


def check_14_bbox_order(results: list, df: pd.DataFrame):
    cid, name = 14, "bbox_order"
    if any(c not in df.columns for c in ["y0", "y1", "x0", "x1"]):
        results.append(make_result(cid, name, "fail", "bbox 컬럼 없음"))
        return
    bad_y = int((df["y1"] <= df["y0"]).sum())
    bad_x = int((df["x1"] <= df["x0"]).sum())
    if bad_y == 0 and bad_x == 0:
        results.append(make_result(cid, name, "pass", "y1>y0, x1>x0 모두 통과"))
    else:
        results.append(make_result(cid, name, "fail",
            f"y1<=y0 인 row={bad_y}, x1<=x0 인 row={bad_x}"))


def check_15_patch_size_distribution(results: list, df: pd.DataFrame):
    cid, name = 15, "patch_size_distribution"
    if any(c not in df.columns for c in ["y0", "y1", "x0", "x1"]):
        results.append(make_result(cid, name, "fail", "bbox 컬럼 없음"))
        return
    h_series = df["y1"] - df["y0"]
    w_series = df["x1"] - df["x0"]
    h_unique = sorted(h_series.unique().tolist())
    w_unique = sorted(w_series.unique().tolist())
    h_mode   = int(h_series.mode().iloc[0])
    w_mode   = int(w_series.mode().iloc[0])
    detail = (
        f"height unique={h_unique[:10]}{'...' if len(h_unique)>10 else ''} mode={h_mode} | "
        f"width unique={w_unique[:10]}{'...' if len(w_unique)>10 else ''} mode={w_mode}"
    )
    status = "pass" if len(h_unique) == 1 and len(w_unique) == 1 else "warn"
    results.append(make_result(cid, name, status, detail))


def check_16_boundary_cases(results: list, df: pd.DataFrame, safe_id_map: dict, vol_root: Path):
    """
    crop_size=96으로 확장 시 boundary 처리 필요한 후보 수 확인.
    center = ((y0+y1)//2, (x0+x1)//2).
    ct_hu.npy를 mmap_mode='r'로 shape만 읽어 H, W, Z를 얻는다.
    """
    cid, name = 16, "boundary_cases_crop96"
    if any(c not in df.columns for c in ["patient_id", "y0", "y1", "x0", "x1", "local_z"]):
        results.append(make_result(cid, name, "fail", "필요 컬럼 없음"))
        return

    boundary_count = 0
    z_oob_count    = 0
    patient_ids    = df["patient_id"].unique()
    missing_vols   = []

    for pid in patient_ids:
        safe_id = safe_id_map.get(pid)
        if safe_id is None:
            missing_vols.append(f"{pid}(safe_id없음)")
            continue
        ct_path = vol_root / "volumes_npy" / safe_id / "ct_hu.npy"
        if not ct_path.exists():
            missing_vols.append(f"{pid}(ct없음)")
            continue
        try:
            arr = np.load(str(ct_path), mmap_mode="r")
            Z, H, W = arr.shape
        except Exception:
            missing_vols.append(f"{pid}(load오류)")
            continue

        sub = df[df["patient_id"] == pid]
        cy = ((sub["y0"] + sub["y1"]) // 2).astype(int)
        cx = ((sub["x0"] + sub["x1"]) // 2).astype(int)
        lz = sub["local_z"].astype(int)

        # boundary: 96px 크롭 시 이미지 경계 초과
        bd_mask = (
            (cy - CROP_HALF < 0) | (cy + CROP_HALF > H) |
            (cx - CROP_HALF < 0) | (cx + CROP_HALF > W)
        )
        boundary_count += int(bd_mask.sum())

        # z out of bounds
        z_oob = ((lz < 0) | (lz >= Z))
        z_oob_count += int(z_oob.sum())

    if missing_vols:
        status = "fail"
    elif boundary_count == 0 and z_oob_count == 0:
        status = "pass"
    else:
        status = "warn"
    detail = (
        f"boundary 처리 필요 후보={boundary_count}, "
        f"z 범위 이탈={z_oob_count}"
    )
    if missing_vols:
        detail += f", volume 미확인 환자={len(missing_vols)}명: {missing_vols[:5]}"
    results.append(make_result(cid, name, status, detail))


def check_17_local_z_range(results: list, df: pd.DataFrame, safe_id_map: dict, vol_root: Path):
    """환자별 local_z가 실제 ct Z 범위 안에 있는지 확인. mmap_mode='r'로 shape만."""
    cid, name = 17, "local_z_in_range"
    if any(c not in df.columns for c in ["patient_id", "local_z"]):
        results.append(make_result(cid, name, "fail", "필요 컬럼 없음"))
        return

    fail_patients = []
    missing_vols  = []
    patient_ids   = df["patient_id"].unique()

    for pid in patient_ids:
        safe_id = safe_id_map.get(pid)
        if safe_id is None:
            missing_vols.append(f"{pid}(safe_id없음)")
            continue
        ct_path = vol_root / "volumes_npy" / safe_id / "ct_hu.npy"
        if not ct_path.exists():
            missing_vols.append(f"{pid}(ct없음)")
            continue
        try:
            arr = np.load(str(ct_path), mmap_mode="r")
            Z = arr.shape[0]
        except Exception:
            missing_vols.append(f"{pid}(load오류)")
            continue

        sub = df[df["patient_id"] == pid]
        lz  = sub["local_z"].astype(int)
        oob = ((lz < 0) | (lz >= Z)).sum()
        if oob > 0:
            fail_patients.append(f"{pid}(oob={oob}, Z={Z})")

    if missing_vols and not fail_patients:
        status = "warn"
    elif fail_patients:
        status = "fail"
    else:
        status = "pass"
    if fail_patients:
        detail = f"z 범위 이탈 환자={len(fail_patients)}: {fail_patients[:10]}"
    elif missing_vols:
        detail = f"모든 확인 환자 local_z 범위 통과 (일부 미확인 있음)"
    else:
        detail = "모든 환자 local_z 범위 통과"
    if missing_vols:
        detail += f" | volume 미확인={len(missing_vols)}명: {missing_vols[:5]}"
    results.append(make_result(cid, name, status, detail))


def check_18_slice_index_memo(results: list, df: pd.DataFrame):
    cid, name = 18, "slice_index_memo"
    if "slice_index" in df.columns:
        detail = (
            "slice_index 컬럼이 존재합니다. "
            "crop z 기준은 local_z이며 slice_index는 참고용(절대 slice 번호)입니다. "
            "crop 생성 시 반드시 local_z를 사용해야 합니다."
        )
    else:
        detail = "slice_index 컬럼 없음. crop z 기준은 local_z 사용."
    results.append(make_result(cid, name, "info", detail))


def check_19_required_columns(results: list, df: pd.DataFrame):
    cid, name = 19, "required_columns_for_npz"
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if not missing:
        results.append(make_result(cid, name, "pass",
            f"필요 컬럼 전부 존재: {REQUIRED_COLUMNS}"))
    else:
        results.append(make_result(cid, name, "fail",
            f"누락 컬럼: {missing}"))


def check_20_lung1_140(results: list, df: pd.DataFrame):
    cid, name = 20, "lung1_140_candidate_counts"
    sub = df[df["patient_id"] == "LUNG1-140"]
    total = len(sub)
    pos   = int((sub["sampling_label"] == "positive").sum())
    hn    = int((sub["sampling_label"] == "hard_negative").sum())
    detail = f"LUNG1-140: total={total}, positive={pos}, hard_negative={hn}"
    status = "pass" if total > 0 else "warn"
    results.append(make_result(cid, name, status, detail))


def check_21_explosion_patients(results: list, df: pd.DataFrame):
    cid, name = 21, "explosion_patients_over_2000"
    counts = df.groupby("patient_id").size()
    exploded = counts[counts > EXPLOSION_THRESHOLD]
    if exploded.empty:
        results.append(make_result(cid, name, "pass",
            f"EXPLOSION_THRESHOLD={EXPLOSION_THRESHOLD} 초과 환자 없음"))
    else:
        detail = ", ".join(f"{pid}({cnt})" for pid, cnt in exploded.items())
        results.append(make_result(cid, name, "warn",
            f"임계값 초과 환자 {len(exploded)}명: {detail}"))


def check_22_special_patients(results: list, df: pd.DataFrame):
    cid, name = 22, "special_patients_positive_exists"
    checks = {}
    for pid in ["LUNG1-415", "LUNG1-156"]:
        sub = df[(df["patient_id"] == pid) & (df["sampling_label"] == "positive")]
        checks[pid] = len(sub)
    detail = " | ".join(f"{pid}: positive={cnt}" for pid, cnt in checks.items())
    status = "pass" if all(v > 0 for v in checks.values()) else "warn"
    results.append(make_result(cid, name, status, detail))


def check_23_summary_consistency(results: list, df: pd.DataFrame):
    cid, name = 23, "summary_csv_json_consistency"
    issues = []

    # summary CSV 읽기
    try:
        sum_csv = pd.read_csv(SUMMARY_CSV)
        csv_pos = int(sum_csv["n_positive"].iloc[0])
        csv_hn  = int(sum_csv["n_hard_negative"].iloc[0])
        csv_pat = int(sum_csv["n_patients"].iloc[0])
    except Exception as ex:
        results.append(make_result(cid, name, "fail", f"summary CSV 로드 실패: {ex}"))
        return

    # manifest 재계산
    man_pos = int((df["sampling_label"] == "positive").sum())
    man_hn  = int((df["sampling_label"] == "hard_negative").sum())
    man_pat = int(df["patient_id"].nunique())

    if csv_pos != man_pos:
        issues.append(f"n_positive: summary={csv_pos} vs manifest={man_pos}")
    if csv_hn != man_hn:
        issues.append(f"n_hard_negative: summary={csv_hn} vs manifest={man_hn}")
    if csv_pat != man_pat:
        issues.append(f"n_patients: summary={csv_pat} vs manifest={man_pat}")

    # summary JSON의 patient_lesion_size_recall 존재 확인
    try:
        with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
            sjson = json.load(f)
        has_size_recall = "patient_lesion_size_recall" in sjson
    except Exception as ex:
        issues.append(f"summary JSON 로드 실패: {ex}")
        has_size_recall = False

    size_recall_msg = (
        f"patient_lesion_size_recall={sjson.get('patient_lesion_size_recall')}"
        if has_size_recall else "patient_lesion_size_recall 키 없음"
    )

    if not issues:
        results.append(make_result(cid, name, "pass",
            f"summary 일관성 통과. {size_recall_msg}"))
    else:
        results.append(make_result(cid, name, "fail",
            f"불일치: {'; '.join(issues)}. {size_recall_msg}"))


def check_24_mtime_snapshot(results: list):
    """score/candidate/evaluation/crop 파일 mtime 현재 상태 스냅샷."""
    cid, name = 24, "existing_files_mtime_snapshot"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshots = []

    for d in SCORE_CANDIDATE_DIRS:
        if d.exists():
            try:
                mt = datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                snapshots.append(f"{d.name}: mtime={mt}")
            except Exception as ex:
                snapshots.append(f"{d.name}: mtime 읽기 실패({ex})")
        else:
            snapshots.append(f"{d.name}: 없음")

    detail = f"스냅샷 시각={now} | " + " | ".join(snapshots)
    results.append(make_result(cid, name, "info", detail))


# ─────────────────────────────────────────────
# save_outputs
# ─────────────────────────────────────────────
def save_outputs(results: list):
    OUT_RPT_DIR.mkdir(parents=True, exist_ok=True)

    n_pass = sum(1 for r in results if r["status"] == "pass")
    n_fail = sum(1 for r in results if r["status"] == "fail")
    n_warn = sum(1 for r in results if r["status"] == "warn")
    n_info = sum(1 for r in results if r["status"] == "info")

    if n_fail == 0 and n_warn == 0:
        overall = "통과"
    elif n_fail == 0:
        overall = "부분 통과 (warn 있음)"
    else:
        overall = "미통과"

    # CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_VAL_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[저장] {OUT_VAL_CSV}")

    # JSON
    payload = {
        "generated_at": datetime.now().isoformat(),
        "overall_status": overall,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_warn": n_warn,
        "n_info": n_info,
        "results": results,
    }
    with open(OUT_VAL_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[저장] {OUT_VAL_JSON}")

    # Markdown
    lines = [
        "# rule_s6a_manifest_validation 결과\n",
        f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**전체 판정: {overall}**  "
        f"(pass={n_pass}, fail={n_fail}, warn={n_warn}, info={n_info})\n",
        "",
        "| check_id | name | status | detail |",
        "|----------|------|--------|--------|",
    ]
    for r in results:
        detail_escaped = r["detail"].replace("|", "\\|")
        lines.append(f"| {r['check_id']} | {r['name']} | **{r['status']}** | {detail_escaped} |")

    with open(OUT_VAL_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[저장] {OUT_VAL_MD}")

    return overall, n_pass, n_fail, n_warn, n_info


# ─────────────────────────────────────────────
# 메인 검증 실행
# ─────────────────────────────────────────────
def run_validation():
    print("\n[검증 시작]", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    results = []

    # manifest 로드
    print("  manifest CSV 로드 중...")
    try:
        df = pd.read_csv(MANIFEST_CSV)
    except Exception as ex:
        results.append(make_result(1, "manifest_exists", "fail", f"로드 실패: {ex}"))
        # 이후 검증 불가
        overall, n_pass, n_fail, n_warn, n_info = save_outputs(results)
        _print_summary(overall, n_pass, n_fail, n_warn, n_info)
        return

    # stage split 로드
    print("  stage split CSV 로드 중...")
    try:
        split_df = pd.read_csv(STAGE_SPLIT_CSV)
    except Exception as ex:
        print(f"  [ERROR] stage split 로드 실패: {ex}")
        split_df = pd.DataFrame(columns=["patient_id", "safe_id", "stage_split"])

    # stage split 컬럼 확인
    actual_split_cols = list(split_df.columns)
    required_split_cols = ["patient_id", "stage_split", "safe_id"]
    missing_split_cols = [c for c in required_split_cols if c not in actual_split_cols]
    print(f"  stage split 컬럼: {actual_split_cols}")
    if missing_split_cols:
        print(f"  [경고] 누락 컬럼: {missing_split_cols}")

    # VOLUME_ROOT
    try:
        vol_root = load_volume_root()
    except Exception as ex:
        print(f"  [ERROR] VOLUME_ROOT 로드 실패: {ex}")
        vol_root = None

    # safe_id 매핑 딕셔너리 생성 (fallback 포함)
    safe_id_map, safe_id_method, n_map_success, n_map_fail = build_safe_id_map(split_df, vol_root)
    print(f"  safe_id 매핑 방식: {safe_id_method}, 성공={n_map_success}명, 실패={n_map_fail}명")
    if safe_id_method == "failed":
        print("  [경고] safe_id 매핑 실패 — check_16/check_17은 volume 미확인으로 처리됩니다")

    # ── 검증 항목 실행 ──────────────────────────
    print("  [1/24] manifest 존재 여부")
    check_01_manifest_exists(results)

    print("  [2/24] manifest row 수")
    check_02_manifest_row_count(results, df)

    print("  [3/24] patient 수")
    check_03_patient_count(results, df)

    print("  [4/24] stage_split 전부 stage1_dev")
    check_04_stage_split_all_dev(results, df)

    print("  [5/24] stage2_holdout 환자 없음")
    check_05_no_stage2_holdout(results, df, split_df)

    print("  [6/24] sampling_rule 일관성")
    check_06_sampling_rule(results, df)

    print("  [7/24] sampling_label 값 종류")
    check_07_sampling_label_values(results, df)

    print("  [8/24] positive 수")
    check_08_positive_count(results, df)

    print("  [9/24] hard_negative 수")
    check_09_hard_negative_count(results, df)

    print("  [10/24] 중복 key 확인")
    check_10_duplicate_keys(results, df)

    print("  [11/24] local_z 유효성")
    check_11_local_z_valid(results, df)

    print("  [12/24] bbox NaN 확인")
    check_12_bbox_nan(results, df)

    print("  [13/24] bbox 좌표 0 이상")
    check_13_bbox_nonneg(results, df)

    print("  [14/24] bbox 순서 (y1>y0, x1>x0)")
    check_14_bbox_order(results, df)

    print("  [15/24] patch size 분포")
    check_15_patch_size_distribution(results, df)

    if vol_root is not None:
        print("  [16/24] boundary 처리 필요 후보 수 (volume mmap, 시간 걸릴 수 있음)")
        check_16_boundary_cases(results, df, safe_id_map, vol_root)

        print("  [17/24] local_z 범위 확인 (volume mmap)")
        check_17_local_z_range(results, df, safe_id_map, vol_root)
    else:
        results.append(make_result(16, "boundary_cases_crop96", "fail", "VOLUME_ROOT 없어서 건너뜀"))
        results.append(make_result(17, "local_z_in_range", "fail", "VOLUME_ROOT 없어서 건너뜀"))

    print("  [18/24] slice_index 메모")
    check_18_slice_index_memo(results, df)

    print("  [19/24] npz 생성 필요 컬럼 확인")
    check_19_required_columns(results, df)

    print("  [20/24] LUNG1-140 후보 수")
    check_20_lung1_140(results, df)

    print("  [21/24] 2000개 초과 환자 목록")
    check_21_explosion_patients(results, df)

    print("  [22/24] LUNG1-415, LUNG1-156 positive 존재")
    check_22_special_patients(results, df)

    print("  [23/24] summary CSV/JSON 일관성")
    check_23_summary_consistency(results, df)

    print("  [24/24] 기존 파일 mtime 스냅샷")
    check_24_mtime_snapshot(results)

    # 저장
    overall, n_pass, n_fail, n_warn, n_info = save_outputs(results)
    _print_summary(overall, n_pass, n_fail, n_warn, n_info)


def _print_summary(overall, n_pass, n_fail, n_warn, n_info):
    print()
    print("=" * 70)
    print(f"  전체 판정: {overall}")
    print(f"  pass={n_pass}  fail={n_fail}  warn={n_warn}  info={n_info}")
    print("=" * 70)
    if n_fail > 0:
        print("  [주의] fail 항목이 있습니다. OUT_VAL_JSON / OUT_VAL_MD를 확인하세요.")
    elif n_warn > 0:
        print("  [확인] warn 항목이 있습니다. 내용을 검토하세요.")
    else:
        print("  모든 검증 항목 통과.")
    print()


# ─────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="S6-A manifest read-only validation (crop 생성 전 점검)"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="실제 검증 실행 + 출력 파일 저장. 없으면 preflight 보고만.",
    )
    args = parser.parse_args()

    if not args.run:
        preflight_report()
        return

    guard_check()
    run_validation()


if __name__ == "__main__":
    main()
