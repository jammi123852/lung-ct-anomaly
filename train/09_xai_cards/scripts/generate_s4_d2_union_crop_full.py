#!/usr/bin/env python3
"""
generate_s4_d2_union_crop_full.py

Second-Stage Lesion Refiner v1 — S4+D2 Union Crop 생성 스크립트

목적:
  - s4_plus_d2_union_stage1_dev_candidate_manifest_dryrun.csv 기반
  - S4 원본(rule_c4_training_sampling_manifest_dryrun.csv)과
    D2 원본(rule_d_stage1_dev_candidate_manifest_dryrun.csv)에서
    join하여 가능한 컬럼 최대한 복원
  - D2에 없는 컬럼은 NaN 유지
  - slice_index는 D2 행에서 NaN (local_z로 대체 금지)
  - crop z 좌표는 항상 local_z 사용
  - npz에 z_source="local_z" 저장
  - union_source는 "S4" 또는 "D2"로 npz에 저장
  - 출력: crops_s4_plus_d2_union_full/

IMPORTANT:
  - generate_s4_crop_full.py 수정 금지
  - crops_s4_full/ 수정 금지
  - stage2_holdout 포함 시 즉시 중단
  - full run은 FULL_RUN_BLOCKED = True 동안 차단
    (ChatGPT 검토 + 사용자 승인 후 safe-code-edit으로 False 변경)

Guard 조건:
  - crops_s4_plus_d2_union_full/ 안에 기존 npz 있으면 즉시 중단
  - stage2_holdout 1명이라도 포함 시 즉시 중단
  - local_z NaN이면 해당 행 error.csv 기록 후 skip
  - crop shape != (3, 96, 96)이면 즉시 중단
  - free space < 10GB 시 중단 (--dry-run 제외)

2.5D crop 방식:
  - 중심 slice (local_z) 기준 z-1, z, z+1
  - z boundary: edge-repeat (z<1 → z=0, z>=Z → z=Z-1)
  - crop 좌표: y0,x0,y1,x1 중심 기반 96×96 고정 crop

출력 경로:
  - crops: outputs/second-stage-lesion-refiner-v1/crops_s4_plus_d2_union_full/{patient_id}/{idx:06d}.npz
  - summary: outputs/second-stage-lesion-refiner-v1/reports/crop_s4_d2_union_full_summary_{mode}.json
  - error:   outputs/second-stage-lesion-refiner-v1/reports/crop_s4_d2_union_full_error_{mode}.csv

실행:
  # dry-run: manifest join 검증만 (crop 생성 안 함)
  source ~/ai_env/bin/activate && \\
  python scripts/generate_s4_d2_union_crop_full.py --dry-run

  # smoke: SMOKE_PATIENTS만 처리 (ChatGPT 검토 + 사용자 승인 필요)
  source ~/ai_env/bin/activate && \\
  python scripts/generate_s4_d2_union_crop_full.py --smoke

  # full run: FULL_RUN_BLOCKED = False 변경 후 실행 (별도 승인 필요)
  source ~/ai_env/bin/activate && \\
  python scripts/generate_s4_d2_union_crop_full.py

syntax check:
  python -m py_compile scripts/generate_s4_d2_union_crop_full.py
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# ============================================================
# 전역 상수
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_DIR = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/candidates"

UNION_MANIFEST = CANDIDATES_DIR / "s4_plus_d2_union_stage1_dev_candidate_manifest_dryrun.csv"
S4_MANIFEST    = CANDIDATES_DIR / "rule_c4_training_sampling_manifest_dryrun.csv"
D2_MANIFEST    = CANDIDATES_DIR / "rule_d_stage1_dev_candidate_manifest_dryrun.csv"
SPLIT_CSV      = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
PATHS_CONFIG   = REPO_ROOT / "configs/paths.local.yaml"

OUTPUT_BASE         = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1"
CROPS_FULL_OUT_DIR  = OUTPUT_BASE / "crops_s4_plus_d2_union_full"
CROPS_SMOKE_OUT_DIR = OUTPUT_BASE / "crops_s4_plus_d2_union_smoke"
REPORTS_DIR         = OUTPUT_BASE / "reports"

# 쓰기 금지 경로
V1_OUTPUT_DIR  = REPO_ROOT / "outputs/position-aware-padim-v1"
CROPS_S4_FULL  = OUTPUT_BASE / "crops_s4_full"
CROPS_S4_SMOKE = OUTPUT_BASE / "crops_s4_smoke"

SCRIPT_NAME         = "generate_s4_d2_union_crop_full.py"
CROP_SIZE           = 96
MIN_FREE_GB         = 10.0
EXPECTED_CROP_SHAPE = (3, CROP_SIZE, CROP_SIZE)

S4_RULE = "S4_patient_balanced"
D2_RULE = "D2_grid4x4_all_suspicious_slices"

# full run 보호 플래그: ChatGPT 검토 + 사용자 승인 후 False로 변경
FULL_RUN_BLOCKED = True

# smoke 대상 환자 (stage1_dev 내, weak case)
SMOKE_PATIENTS = [
    "MSD_lung_043",
    "MSD_lung_079",
    "MSD_lung_096",
]

DEDUP_COLS = ["patient_id", "local_z", "y0", "x0", "y1", "x1"]

# ============================================================
# 안전 유틸
# ============================================================

def abort(msg: str) -> None:
    print(f"[ABORT] {msg}", file=sys.stderr)
    sys.exit(1)


def check_not_forbidden(path: Path) -> None:
    """v1 출력 경로, crops_s4_full, crops_s4_smoke 쓰기 시도 감지"""
    for forbidden in [V1_OUTPUT_DIR, CROPS_S4_FULL, CROPS_S4_SMOKE]:
        try:
            path.relative_to(forbidden)
            abort(f"금지된 경로에 쓰기 시도 감지: {path}  (금지: {forbidden})")
        except ValueError:
            pass


def load_paths_config() -> Dict:
    if not PATHS_CONFIG.exists():
        abort(f"paths.local.yaml 없음: {PATHS_CONFIG}")
    with open(PATHS_CONFIG, "r") as f:
        return yaml.safe_load(f) or {}


def get_v2_volume_root(paths_cfg: Dict) -> Path:
    key = "nsclc_msd_usable_only_v2"
    v2_path = paths_cfg.get(key, "")
    if not v2_path:
        abort(f"paths.local.yaml에 '{key}' 경로 없음")
    p = Path(v2_path)
    if not p.exists():
        abort(f"v2 volume 경로 없음: {p}")
    return p


def build_patient_to_safeid(v2_root: Path) -> Dict[str, str]:
    csv_path = v2_root / "manifests" / "patient_manifest.csv"
    if not csv_path.exists():
        abort(f"patient_manifest.csv 없음: {csv_path}")
    df = pd.read_csv(csv_path, usecols=["patient_id", "safe_id"])
    return dict(zip(df["patient_id"].astype(str), df["safe_id"].astype(str)))


def check_free_space(min_gb: float) -> None:
    p = REPO_ROOT if REPO_ROOT.exists() else REPO_ROOT.parent
    free_gb = shutil.disk_usage(p).free / (1024 ** 3)
    print(f"[INFO] free space: {free_gb:.1f} GB (최소 요구: {min_gb} GB)")
    if free_gb < min_gb:
        abort(f"여유 공간 부족: {free_gb:.1f} GB < {min_gb} GB")


# ============================================================
# Guard
# ============================================================

def guard_no_existing_npz(crops_out_dir: Path) -> None:
    if crops_out_dir.exists():
        existing = list(crops_out_dir.rglob("*.npz"))
        if existing:
            abort(
                f"{crops_out_dir.name}/ 안에 기존 npz {len(existing)}개 존재.\n"
                f"  첫 번째: {existing[0]}\n"
                "수동 삭제 후 재실행하세요."
            )


def guard_no_holdout(df: pd.DataFrame, holdout_set: set) -> None:
    overlap = set(df["patient_id"].unique()) & holdout_set
    if overlap:
        abort(f"stage2_holdout 환자 포함 감지 — 즉시 중단: {sorted(overlap)}")


# ============================================================
# derived_grid_position_bin 재계산 (512 기준 3×3)
# ============================================================

def calc_grid_position_bin(y0: float, x0: float, y1: float, x1: float,
                            img_size: int = 512, n: int = 3) -> str:
    """bbox 중심 기준 n×n 그리드 셀 위치 반환 (예: 'row1_col2')"""
    cy = (y0 + y1) / 2.0
    cx = (x0 + x1) / 2.0
    cell = img_size / n
    row = min(int(cy // cell), n - 1)
    col = min(int(cx // cell), n - 1)
    return f"row{row}_col{col}"


# ============================================================
# z_level 재계산 (total_z 기준)
# ============================================================

def calc_z_level(local_z: int, total_z: int) -> str:
    ratio = local_z / max(total_z - 1, 1)
    if ratio < 1 / 3:
        return "upper"
    elif ratio < 2 / 3:
        return "middle"
    return "lower"


# ============================================================
# Union manifest 로딩 및 컬럼 복원
# ============================================================

def load_and_join_manifests() -> pd.DataFrame:
    """
    union manifest → S4/D2 원본과 join, 컬럼 최대 복원.

    - S4 행: S4 원본에서 DEDUP_COLS 기준 join → S4 전용 컬럼 복원
    - D2 행: D2 원본에서 DEDUP_COLS 기준 join → D2 전용 컬럼 복원
    - slice_index: D2 행은 NaN 강제 (local_z로 대체 금지)
    - padim_score: D2 행은 NaN 강제
    - derived_grid_position_bin: NaN이면 y0/x0/y1/x1에서 재계산
    - union_source: source 컬럼에서 복사
    """
    for p in [UNION_MANIFEST, S4_MANIFEST, D2_MANIFEST]:
        if not p.exists():
            abort(f"필수 입력 파일 없음: {p}")

    df_union   = pd.read_csv(UNION_MANIFEST)
    df_s4_orig = pd.read_csv(S4_MANIFEST)
    df_d2_orig = pd.read_csv(D2_MANIFEST)

    df_s4_src = df_s4_orig[df_s4_orig["sampling_rule"] == S4_RULE].copy()
    df_d2_src = df_d2_orig[df_d2_orig["rule_d_variant"] == D2_RULE].copy()

    n_s4_union = (df_union["source"] == "S4").sum()
    n_d2_union = (df_union["source"] == "D2").sum()
    print(f"[INFO] union manifest: 총 {len(df_union)}행 (S4={n_s4_union}, D2={n_d2_union})")
    print(f"[INFO] S4 원본: {len(df_s4_src)}행 | D2 원본: {len(df_d2_src)}행")

    # ── S4 join ───────────────────────────────────────────
    df_s4_union = df_union[df_union["source"] == "S4"].copy()
    df_s4_merged = df_s4_union.merge(df_s4_src, on=DEDUP_COLS, how="left", suffixes=("", "_s4"))
    # row count guard: 중복 key로 행이 늘어나면 즉시 abort
    if len(df_s4_merged) != len(df_s4_union):
        abort(
            f"S4 join 후 행 수 불일치: union={len(df_s4_union)}, joined={len(df_s4_merged)}\n"
            "  중복 key로 행이 늘었을 가능성 있음. S4 원본 DEDUP_COLS 확인 필요."
        )
    # union manifest 값 우선: 충돌 컬럼(_s4 suffix) 제거
    drop_cols = [c for c in df_s4_merged.columns if c.endswith("_s4")]
    df_s4_merged.drop(columns=drop_cols, inplace=True)

    # ── D2 join ───────────────────────────────────────────
    df_d2_union = df_union[df_union["source"] == "D2"].copy()
    df_d2_merged = df_d2_union.merge(df_d2_src, on=DEDUP_COLS, how="left", suffixes=("", "_d2"))
    # row count guard: 중복 key로 행이 늘어나면 즉시 abort
    if len(df_d2_merged) != len(df_d2_union):
        abort(
            f"D2 join 후 행 수 불일치: union={len(df_d2_union)}, joined={len(df_d2_merged)}\n"
            "  중복 key로 행이 늘었을 가능성 있음. D2 원본 DEDUP_COLS 확인 필요."
        )
    drop_cols = [c for c in df_d2_merged.columns if c.endswith("_d2")]
    df_d2_merged.drop(columns=drop_cols, inplace=True)

    # ── concat ────────────────────────────────────────────
    df_all = pd.concat([df_s4_merged, df_d2_merged], ignore_index=True)
    # row count guard: concat 후 전체 행 수 == union manifest 행 수
    if len(df_all) != len(df_union):
        abort(
            f"concat 후 행 수 불일치: union={len(df_union)}, all={len(df_all)}\n"
            "  S4 또는 D2 join에서 행이 증감했습니다."
        )

    # ── union_source 컬럼 ─────────────────────────────────
    df_all["union_source"] = df_all["source"]

    # ── slice_index: D2 행 NaN 강제 ───────────────────────
    if "slice_index" not in df_all.columns:
        df_all["slice_index"] = np.nan
    df_all.loc[df_all["union_source"] == "D2", "slice_index"] = np.nan

    # ── padim_score: D2 행 NaN 강제 ───────────────────────
    if "padim_score" not in df_all.columns:
        df_all["padim_score"] = np.nan
    df_all.loc[df_all["union_source"] == "D2", "padim_score"] = np.nan

    # ── derived_grid_position_bin 재계산 (NaN이면) ─────────
    if "derived_grid_position_bin" not in df_all.columns:
        df_all["derived_grid_position_bin"] = np.nan
    mask_no_dpb = df_all["derived_grid_position_bin"].isna()
    if mask_no_dpb.any():
        df_all.loc[mask_no_dpb, "derived_grid_position_bin"] = df_all.loc[mask_no_dpb].apply(
            lambda r: calc_grid_position_bin(r["y0"], r["x0"], r["y1"], r["x1"]),
            axis=1,
        )
        n_recalc = mask_no_dpb.sum()
        print(f"[INFO] derived_grid_position_bin 재계산: {n_recalc}행 (주로 D2)")

    # ── z_level: volume 로드 시점에 재계산 (여기서는 초기화) ─
    if "z_level" not in df_all.columns:
        df_all["z_level"] = np.nan

    # ── position_bin: D2 행 없으면 NaN 유지 ──────────────
    if "position_bin" not in df_all.columns:
        df_all["position_bin"] = np.nan

    print(f"[INFO] join 완료: 총 {len(df_all)}행")
    d2_slice_nan = df_all.loc[df_all["union_source"] == "D2", "slice_index"].isna().sum()
    d2_padim_nan = df_all.loc[df_all["union_source"] == "D2", "padim_score"].isna().sum()
    print(f"[INFO] D2 행 slice_index NaN: {d2_slice_nan} / padim_score NaN: {d2_padim_nan}")

    return df_all


# ============================================================
# crop 좌표 계산
# ============================================================

def compute_fixed_crop_coords(
    cy: float, cx: float, crop_size: int, img_h: int, img_w: int
) -> Tuple[int, int, int, int]:
    half = crop_size // 2
    y0 = max(0, int(cy) - half)
    x0 = max(0, int(cx) - half)
    y1 = min(img_h, y0 + crop_size)
    x1 = min(img_w, x0 + crop_size)
    if y1 - y0 < crop_size:
        y0 = max(0, y1 - crop_size)
    if x1 - x0 < crop_size:
        x0 = max(0, x1 - crop_size)
    return y0, x0, y1, x1


def load_ct_volume(volume_dir: Path) -> np.ndarray:
    ct_path = volume_dir / "ct_hu.npy"
    if not ct_path.exists():
        raise FileNotFoundError(f"ct_hu.npy 없음: {ct_path}")
    ct = np.load(ct_path, mmap_mode="r")
    if ct.ndim != 3:
        raise ValueError(f"ct_hu.npy shape 오류 (Z,H,W) 기대, 실제: {ct.shape}")
    return ct


def extract_2p5d_crop(
    ct: np.ndarray, z: int, y0: int, x0: int, y1: int, x1: int
) -> np.ndarray:
    Z = ct.shape[0]
    z_prev = max(0, z - 1)
    z_next = min(Z - 1, z + 1)
    crop = np.stack([
        ct[z_prev, y0:y1, x0:x1].astype(np.float32),
        ct[z,      y0:y1, x0:x1].astype(np.float32),
        ct[z_next, y0:y1, x0:x1].astype(np.float32),
    ], axis=0)
    if crop.shape != EXPECTED_CROP_SHAPE:
        raise ValueError(
            f"crop shape 불일치: 기대 {EXPECTED_CROP_SHAPE}, 실제 {crop.shape} "
            f"(z={z}, y={y0}:{y1}, x={x0}:{x1})"
        )
    return crop


# ============================================================
# 환자 처리
# ============================================================

def process_patient(
    patient_id: str,
    rows: pd.DataFrame,
    volume_dir: Path,
    out_dir: Path,
    error_rows: List[Dict],
    dry_run: bool = False,
) -> Tuple[int, int]:
    """(saved_count, error_count) 반환"""
    try:
        ct = load_ct_volume(volume_dir)
    except FileNotFoundError as e:
        error_rows.append({"patient_id": patient_id, "error": str(e), "stage": "volume_load"})
        return 0, 1

    Z, img_h, img_w = ct.shape
    saved = 0
    errors = 0

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        check_not_forbidden(out_dir)

    for idx, row in rows.reset_index(drop=True).iterrows():
        try:
            # local_z 검사 (slice_index로 대체 금지)
            if pd.isna(row["local_z"]):
                raise ValueError("local_z NaN — crop 생성 불가 (slice_index로 대체 금지)")
            z = int(row["local_z"])
            if z < 0:
                raise ValueError(f"local_z < 0: {z}")
            if z >= Z:
                raise ValueError(f"local_z={z} >= 총 slice 수={Z}")

            bbox_cy = (row["y0"] + row["y1"]) / 2.0
            bbox_cx = (row["x0"] + row["x1"]) / 2.0
            cy0, cx0, cy1, cx1 = compute_fixed_crop_coords(bbox_cy, bbox_cx, CROP_SIZE, img_h, img_w)

            if dry_run:
                saved += 1
                continue

            crop_arr = extract_2p5d_crop(ct, z, cy0, cx0, cy1, cx1)

            # slice_index: D2는 NaN → -1로 저장, valid flag 별도 저장
            raw_si = row.get("slice_index", np.nan)
            si_valid = bool(pd.notna(raw_si))
            si_val   = int(raw_si) if si_valid else -1

            # padim_score: D2는 NaN 허용
            raw_ps = row.get("padim_score", np.nan)
            ps_val = float(raw_ps) if pd.notna(raw_ps) else float("nan")

            # z_level: NaN이면 volume 기준 재계산
            raw_zl = row.get("z_level", np.nan)
            if pd.isna(raw_zl) or str(raw_zl) == "nan":
                z_lv = calc_z_level(z, Z)
            else:
                z_lv = str(raw_zl)

            # derived_grid_position_bin: NaN이면 재계산 (join 시 이미 채웠지만 방어)
            raw_dpb = row.get("derived_grid_position_bin", np.nan)
            if pd.isna(raw_dpb) or str(raw_dpb) == "nan":
                dpb = calc_grid_position_bin(row["y0"], row["x0"], row["y1"], row["x1"])
            else:
                dpb = str(raw_dpb)

            # position_bin
            raw_pb = row.get("position_bin", np.nan)
            pb = str(raw_pb) if pd.notna(raw_pb) else "UNKNOWN"

            # union_source (S4 또는 D2)
            u_src = str(row.get("union_source", row.get("source", "UNKNOWN")))

            npz_path = out_dir / f"{idx:06d}.npz"
            check_not_forbidden(npz_path)

            np.savez_compressed(
                npz_path,
                crop=crop_arr,
                label=np.array(str(row["sampling_label"])),
                patient_id=np.array(patient_id),
                local_z=np.array(z),
                z_source=np.array("local_z"),
                slice_index=np.array(si_val),
                slice_index_valid=np.array(si_valid),
                crop_coords=np.array([cy0, cx0, cy1, cx1]),
                orig_bbox=np.array([int(row["y0"]), int(row["x0"]),
                                    int(row["y1"]), int(row["x1"])]),
                padim_score=np.array(ps_val),
                sampling_label=np.array(str(row["sampling_label"])),
                union_source=np.array(u_src),
                position_bin=np.array(pb),
                z_level=np.array(z_lv),
                derived_grid_position_bin=np.array(dpb),
            )
            saved += 1

        except Exception as e:
            errors += 1
            error_rows.append({
                "patient_id": patient_id,
                "row_idx": idx,
                "local_z": row.get("local_z", "N/A"),
                "union_source": row.get("union_source", row.get("source", "N/A")),
                "error": str(e),
                "stage": "crop_save",
            })

    return saved, errors


# ============================================================
# 메인
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=SCRIPT_NAME)
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"smoke 모드: {SMOKE_PATIENTS}만 처리",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="manifest join 검증만 (crop 생성 안 함)",
    )
    args = parser.parse_args()

    print(f"[{SCRIPT_NAME}] 시작: {datetime.now().isoformat()}")

    if args.dry_run:
        print("[MODE] DRY-RUN: manifest join 검증만 수행 (crop 생성 안 함)")
        mode = "dry_run"
    elif args.smoke:
        print(f"[MODE] SMOKE: {SMOKE_PATIENTS}만 처리")
        mode = "smoke"
    else:
        if FULL_RUN_BLOCKED:
            abort(
                "FULL RUN은 현재 차단 상태입니다.\n"
                "  ChatGPT 검토 + 사용자 승인 후 FULL_RUN_BLOCKED = False로 변경하세요.\n"
                "  smoke 테스트: --smoke\n"
                "  manifest join 검증: --dry-run"
            )
        mode = "full"
        print("[MODE] FULL: 모든 환자 처리")

    # ── crops 출력 경로: smoke와 full 분리 ───────────────
    if args.smoke:
        crops_out_dir = CROPS_SMOKE_OUT_DIR
        print(f"[INFO] smoke 출력 경로: {crops_out_dir}")
    elif args.dry_run:
        crops_out_dir = CROPS_FULL_OUT_DIR  # dry-run은 실제 쓰지 않으므로 경로만
    else:
        crops_out_dir = CROPS_FULL_OUT_DIR
        print(f"[INFO] full 출력 경로: {crops_out_dir}")

    run_start = datetime.now()

    # ── Guard: 기존 npz 존재 여부 (dry-run 제외) ──────────
    if not args.dry_run:
        guard_no_existing_npz(crops_out_dir)

    # ── manifest 로딩 및 join ─────────────────────────────
    df_all = load_and_join_manifests()

    # ── split CSV → holdout 봉인 ──────────────────────────
    if not SPLIT_CSV.exists():
        abort(f"split CSV 없음: {SPLIT_CSV}")
    df_split = pd.read_csv(SPLIT_CSV, usecols=["patient_id", "stage_split"])
    holdout_set = set(df_split.loc[df_split["stage_split"] == "stage2_holdout", "patient_id"])
    guard_no_holdout(df_all, holdout_set)

    # ── smoke 필터 ────────────────────────────────────────
    if args.smoke:
        df_all = df_all[df_all["patient_id"].isin(SMOKE_PATIENTS)].copy()
        if df_all.empty:
            abort(f"smoke 대상 환자({SMOKE_PATIENTS})가 union manifest에 없습니다.")
        print(f"[SMOKE] 대상 환자: {sorted(df_all['patient_id'].unique())}")

    # ── 경로 설정 ────────────────────────────────────────
    paths_cfg = load_paths_config()
    v2_root = get_v2_volume_root(paths_cfg)
    patient_to_safeid = build_patient_to_safeid(v2_root)

    # ── Guard: free space (dry-run 제외) ─────────────────
    if not args.dry_run:
        check_free_space(MIN_FREE_GB)

    # ── 환자별 처리 ──────────────────────────────────────
    error_rows: List[Dict] = []
    total_saved = 0
    total_errors = 0
    patients = sorted(df_all["patient_id"].unique())
    print(f"[INFO] 처리 대상 환자 수: {len(patients)}")

    for i, patient_id in enumerate(patients, 1):
        safe_id = patient_to_safeid.get(patient_id)
        if safe_id is None:
            error_rows.append({
                "patient_id": patient_id, "error": "safe_id 매핑 없음", "stage": "mapping",
            })
            total_errors += 1
            continue

        volume_dir = v2_root / "volumes_npy" / safe_id
        if not volume_dir.exists():
            error_rows.append({
                "patient_id": patient_id,
                "error": f"volume_dir 없음: {volume_dir}",
                "stage": "volume_dir",
            })
            total_errors += 1
            continue

        patient_rows = df_all[df_all["patient_id"] == patient_id]
        out_dir = crops_out_dir / patient_id

        saved, errors = process_patient(
            patient_id=patient_id,
            rows=patient_rows,
            volume_dir=volume_dir,
            out_dir=out_dir,
            error_rows=error_rows,
            dry_run=args.dry_run,
        )
        total_saved += saved
        total_errors += errors

        n_s4 = int((patient_rows["union_source"] == "S4").sum())
        n_d2 = int((patient_rows["union_source"] == "D2").sum())
        dr_tag = " [DRY-RUN]" if args.dry_run else ""
        print(
            f"  [{i:3d}/{len(patients)}] {patient_id}: "
            f"rows={len(patient_rows)} (S4={n_s4}, D2={n_d2}), "
            f"saved={saved}, errors={errors}{dr_tag}"
        )

    # ── smoke 보고 ────────────────────────────────────────
    if args.smoke:
        d2_rows     = df_all[df_all["union_source"] == "D2"]
        s4_count    = int((df_all["union_source"] == "S4").sum())
        d2_count    = int(len(d2_rows))
        d2_ps_nan   = int(d2_rows["padim_score"].isna().sum())
        d2_si_false = int(d2_rows["slice_index"].isna().sum())
        print(f"\n[SMOKE 보고]")
        print(f"  S4 행 수: {s4_count}")
        print(f"  D2 행 수: {d2_count}")
        print(f"  D2 padim_score NaN: {d2_ps_nan}/{d2_count}")
        print(f"  D2 slice_index_valid=False (slice_index NaN): {d2_si_false}/{d2_count}")
        print(f"  union_source: npz에 'union_source' 키로 저장")

    # ── summary/error 저장 ────────────────────────────────
    run_end = datetime.now()
    elapsed = (run_end - run_start).total_seconds()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if error_rows and not args.dry_run:
        error_csv = REPORTS_DIR / f"crop_s4_d2_union_full_error_{mode}.csv"
        if error_csv.exists():
            abort(
                f"error CSV 이미 존재 (덮어쓰기 금지): {error_csv}\n"
                "  기존 파일 보존. 파일명 변경 또는 이동 후 재실행하세요."
            )
        check_not_forbidden(error_csv)
        pd.DataFrame(error_rows).to_csv(error_csv, index=False)
        print(f"[WARN] error.csv 저장: {error_csv} ({len(error_rows)}건)")

    summary = {
        "script": SCRIPT_NAME,
        "mode": mode,
        "smoke_patients": SMOKE_PATIENTS if args.smoke else None,
        "total_patients": len(patients),
        "total_rows_in_manifest": len(df_all),
        "union_source_counts": {
            "S4": int((df_all["union_source"] == "S4").sum()),
            "D2": int((df_all["union_source"] == "D2").sum()),
        },
        "total_saved": total_saved,
        "total_errors": total_errors,
        "run_start": run_start.isoformat(),
        "run_end": run_end.isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "z_source": "local_z",
        "crop_size": CROP_SIZE,
        "note": (
            "slice_index는 D2 행에서 NaN(-1로 저장, slice_index_valid=False). "
            "crop z 좌표는 항상 local_z 사용. "
            "padim_score는 D2 행에서 NaN 허용."
        ),
    }

    sum_json = REPORTS_DIR / f"crop_s4_d2_union_full_summary_{mode}.json"
    if sum_json.exists():
        abort(
            f"summary 파일 이미 존재 (덮어쓰기 금지): {sum_json}\n"
            "  기존 파일 보존. 파일명 변경 또는 이동 후 재실행하세요."
        )
    check_not_forbidden(sum_json)
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    tag = "[DRY-RUN]" if args.dry_run else "[DONE]"
    print(f"\n{tag} 처리 완료: saved={total_saved:,} / errors={total_errors}")
    print(f"{tag} 경과 시간: {elapsed:.1f}초")
    print(f"{tag} summary: {sum_json}")
    if total_errors > 0:
        print(f"[WARN] error CSV 확인 요망")


if __name__ == "__main__":
    main()
