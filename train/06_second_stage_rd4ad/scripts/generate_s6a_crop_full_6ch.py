#!/usr/bin/env python3
"""
generate_s6a_crop_full_6ch.py
=============================
S6-A v2/v2 후보를 정상기반 2차 모델 입력 형식인 6ch image npz로 변환하는 스크립트.

목적:
- rule_s6a_gs2_selected_candidate_manifest_dryrun.csv 후보를 6ch crop npz로 생성
- ch0~2: lung window (z-1/z/z+1), ch3~5: mediastinal window (z-1/z/z+1)
- output key: image (shape: (6, 96, 96), dtype: float32, range: [0, 1])
- local_z 기준 crop, label은 숫자 0/1, sampling_label은 문자열로 별도 저장

실행 모드:
- 인자 없음   : 즉시 sys.exit(1)로 중단
- --dry-run   : 파일 생성 없이 예상 crop 수, 환자별 pos/hn 수 출력
- --smoke     : smoke 5명만 실제 생성
- --run-full                      : 단독 실행 금지 → sys.exit(1)
- --run-full --confirm-full-run  : 전체 154명 full crop 생성 (이중 승인 구조)

절대 금지:
- full 6ch crop 생성 금지 (--run-full 단독 실행 시 sys.exit(1))
- crops_s6a_full/ 경로 수정/삭제 금지 (상수 정의만, 접근 금지)
- manifest 수정 금지
- 모델 학습 금지 (epoch loop, optimizer, checkpoint 포함)
- scoring 재실행 금지
- supervised BCE 학습 구현 금지
- stage2_holdout 사용 금지
- pip/conda install 금지

실행 명령 (사용자 승인 후):
  [dry-run]:
    source ~/ai_env/bin/activate && \\
    python scripts/generate_s6a_crop_full_6ch.py --dry-run

  [smoke — 별도 승인 후]:
    source ~/ai_env/bin/activate && \\
    python scripts/generate_s6a_crop_full_6ch.py --smoke

  [full — 이중 승인 후]:
    source ~/ai_env/bin/activate && \\
    python scripts/generate_s6a_crop_full_6ch.py --run-full --confirm-full-run

syntax check (실행 아님):
  python -m py_compile scripts/generate_s6a_crop_full_6ch.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ─────────────────────────────────────────────
# 경로 상수
# ─────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_CSV     = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/candidates/rule_s6a_gs2_selected_candidate_manifest_dryrun.csv"
STAGE_SPLIT_CSV  = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
PATHS_CONFIG     = REPO_ROOT / "configs/paths.local.yaml"
OUT_RPT_DIR      = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports"

# smoke 출력 경로
OUT_CROPS_SMOKE  = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_s6a_6ch_smoke"
OUT_SUMMARY_SMOKE = OUT_RPT_DIR / "crop_s6a_6ch_smoke_summary.json"

# full 출력 경로 (상수 정의만 — 접근 금지, --run-full 시 sys.exit(1))
OUT_CROPS_FULL        = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_s6a_6ch_full"  # 접근 금지
OUT_SUMMARY_FULL_JSON = OUT_RPT_DIR / "crop_s6a_6ch_full_summary.json"                           # 접근 금지
OUT_SUMMARY_FULL_CSV  = OUT_RPT_DIR / "crop_s6a_6ch_full_summary.csv"                            # 접근 금지
OUT_SUMMARY_FULL_MD   = OUT_RPT_DIR / "crop_s6a_6ch_full_summary.md"                             # 접근 금지
OUT_SUMMARY_FULL      = OUT_SUMMARY_FULL_JSON  # 호환성 별칭

# 기존 crops_s6a_full/ — 절대 접근하지 않음 (수정/삭제 금지)
# LEGACY_CROPS_S6A_FULL = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_s6a_full"

SAMPLING_RULE_TARGET = "S6-A_positive_all_hn_ratio2"
CROP_SIZE    = 96
MAX_POSITIVE = 15
MAX_PER_PATIENT = 30

SMOKE_PATIENTS = [
    "LUNG1-001",
    "LUNG1-004",
    "LUNG1-008",
    "MSD_lung_001",
    "MSD_lung_003",
]

# ─────────────────────────────────────────────
# window 파라미터
# ─────────────────────────────────────────────
LUNG_WIN_MIN   = -1350.0
LUNG_WIN_MAX   =   150.0
MEDI_WIN_MIN   = -160.0
MEDI_WIN_MAX   =  240.0


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
def load_paths_config() -> dict:
    with open(PATHS_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_vol_root(paths_cfg: dict) -> Path:
    raw = paths_cfg.get("nsclc_msd_usable_only_v2", "")
    if not raw:
        raise RuntimeError("paths.local.yaml에 nsclc_msd_usable_only_v2 경로가 없습니다.")
    p = Path(raw)
    if not p.exists():
        raise RuntimeError(f"v2 volume 경로가 존재하지 않습니다: {p}")
    return p


def build_safe_id_map(split_df: pd.DataFrame) -> dict:
    if "safe_id" not in split_df.columns:
        raise RuntimeError("stage split CSV에 safe_id 컬럼 없음")
    mapping = dict(zip(split_df["patient_id"], split_df["safe_id"]))
    return {k: v for k, v in mapping.items() if pd.notna(v) and str(v).strip() != ""}


def get_holdout_set(split_df: pd.DataFrame) -> set:
    return set(split_df.loc[split_df["stage_split"] == "stage2_holdout", "patient_id"].tolist())


# ─────────────────────────────────────────────
# window 정규화
# ─────────────────────────────────────────────
def apply_lung_window(arr: np.ndarray) -> np.ndarray:
    """lung window: clip [-1350, 150], scale to [0, 1]"""
    clipped = np.clip(arr, LUNG_WIN_MIN, LUNG_WIN_MAX)
    return ((clipped - LUNG_WIN_MIN) / (LUNG_WIN_MAX - LUNG_WIN_MIN)).astype(np.float32)


def apply_medi_window(arr: np.ndarray) -> np.ndarray:
    """mediastinal window: clip [-160, 240], scale to [0, 1]"""
    clipped = np.clip(arr, MEDI_WIN_MIN, MEDI_WIN_MAX)
    return ((clipped - MEDI_WIN_MIN) / (MEDI_WIN_MAX - MEDI_WIN_MIN)).astype(np.float32)


# ─────────────────────────────────────────────
# crop 좌표 계산 (기존과 동일)
# ─────────────────────────────────────────────
def compute_fixed_crop_coords(
    cy: float, cx: float, crop_size: int, img_h: int, img_w: int
) -> tuple:
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


# ─────────────────────────────────────────────
# 6ch image 추출
# ─────────────────────────────────────────────
def extract_6ch_crop(
    ct: np.ndarray, z: int, y0: int, x0: int, y1: int, x1: int
) -> np.ndarray:
    """
    output shape: (6, 96, 96), dtype: float32
    ch0~2: lung window (z-1/z/z+1)
    ch3~5: mediastinal window (z-1/z/z+1)
    z boundary: edge-repeat
    """
    Z = ct.shape[0]
    z_prev = max(0, z - 1)
    z_next = min(Z - 1, z + 1)

    # HU 원본 패치 추출
    raw_prev = ct[z_prev, y0:y1, x0:x1].astype(np.float32)
    raw_curr = ct[z,      y0:y1, x0:x1].astype(np.float32)
    raw_next = ct[z_next, y0:y1, x0:x1].astype(np.float32)

    # lung window 채널 (ch0~2)
    lung_prev = apply_lung_window(raw_prev)
    lung_curr = apply_lung_window(raw_curr)
    lung_next = apply_lung_window(raw_next)

    # mediastinal window 채널 (ch3~5)
    medi_prev = apply_medi_window(raw_prev)
    medi_curr = apply_medi_window(raw_curr)
    medi_next = apply_medi_window(raw_next)

    # (6, H, W) 스택
    image = np.stack(
        [lung_prev, lung_curr, lung_next, medi_prev, medi_curr, medi_next],
        axis=0,
    )  # (6, 96, 96)
    return image


# ─────────────────────────────────────────────
# guard_check (6ch 버전)
# ─────────────────────────────────────────────
def guard_check(mode: str) -> None:
    """
    mode: 'dry-run' | 'smoke' | 'run-full'
    smoke/full 출력 폴더 및 summary 파일 overwrite 방지 포함.
    """
    errors = []

    # guard 1: smoke 출력 폴더 이미 있으면 중단 (smoke 모드)
    if mode == "smoke" and OUT_CROPS_SMOKE.exists():
        errors.append(f"[GUARD 1] crops_s6a_6ch_smoke 폴더가 이미 존재합니다: {OUT_CROPS_SMOKE}")

    # guard 2: summary 파일 이미 있으면 중단 (smoke 모드)
    if mode == "smoke" and OUT_SUMMARY_SMOKE.exists():
        errors.append(f"[GUARD 2] smoke summary 파일이 이미 존재합니다: {OUT_SUMMARY_SMOKE}")

    # guard 1-f: full 출력 폴더 이미 있으면 중단 (run-full 모드)
    if mode == "run-full" and OUT_CROPS_FULL.exists():
        errors.append(f"[GUARD 1-F] crops_s6a_6ch_full 폴더가 이미 존재합니다: {OUT_CROPS_FULL}")

    # guard 2-f: full summary 파일 (CSV/JSON/MD) 중 하나라도 있으면 중단 (run-full 모드)
    if mode == "run-full":
        for _p, _label in [
            (OUT_SUMMARY_FULL_JSON, "JSON"),
            (OUT_SUMMARY_FULL_CSV,  "CSV"),
            (OUT_SUMMARY_FULL_MD,   "MD"),
        ]:
            if _p.exists():
                errors.append(f"[GUARD 2-F] full summary {_label} 파일이 이미 존재합니다: {_p}")

    # guard 3: manifest에 stage1_dev 이외 환자 있으면 중단
    if MANIFEST_CSV.exists():
        df_check = pd.read_csv(MANIFEST_CSV, usecols=["patient_id", "stage_split"])
        non_stage1 = df_check[df_check["stage_split"] != "stage1_dev"]
        if len(non_stage1) > 0:
            unique_splits = non_stage1["stage_split"].unique().tolist()
            errors.append(
                f"[GUARD 3] manifest에 stage1_dev 이외 행 존재: {len(non_stage1)}건, "
                f"stage_split 값={unique_splits}"
            )

    # guard 4: stage2_holdout 0명 확인
    if STAGE_SPLIT_CSV.exists():
        split_check = pd.read_csv(STAGE_SPLIT_CSV, usecols=["patient_id", "stage_split"])
        holdout_in_manifest = set()
        if MANIFEST_CSV.exists():
            df_pid = pd.read_csv(MANIFEST_CSV, usecols=["patient_id"])
            manifest_pids = set(df_pid["patient_id"].unique())
            holdout_pids = set(
                split_check.loc[split_check["stage_split"] == "stage2_holdout", "patient_id"]
            )
            holdout_in_manifest = manifest_pids & holdout_pids
        if holdout_in_manifest:
            errors.append(
                f"[GUARD 4] stage2_holdout 환자가 manifest에 존재: {sorted(holdout_in_manifest)}"
            )

    # guard 5: smoke 대상 5명이 manifest에 존재하는지 확인 (smoke 모드만)
    if mode == "smoke" and MANIFEST_CSV.exists():
        df_pid = pd.read_csv(MANIFEST_CSV, usecols=["patient_id"])
        manifest_pids = set(df_pid["patient_id"].unique())
        missing_smoke = [p for p in SMOKE_PATIENTS if p not in manifest_pids]
        if missing_smoke:
            errors.append(
                f"[GUARD 5] smoke 대상 환자가 manifest에 없음: {missing_smoke}"
            )

    # guard 6: ct_hu.npy 누락 환자 있으면 중단 (stage1_dev 전체, manifest 존재 시)
    if MANIFEST_CSV.exists() and STAGE_SPLIT_CSV.exists() and PATHS_CONFIG.exists():
        try:
            paths_cfg = load_paths_config()
            vol_root = get_vol_root(paths_cfg)
            split_df = pd.read_csv(STAGE_SPLIT_CSV)
            safe_id_map = build_safe_id_map(split_df)
            df_pid2 = pd.read_csv(MANIFEST_CSV, usecols=["patient_id", "stage_split"])
            stage1_pids = df_pid2[df_pid2["stage_split"] == "stage1_dev"]["patient_id"].unique().tolist()
            missing_ct = []
            for pid in stage1_pids:
                safe_id = safe_id_map.get(pid)
                if safe_id is None:
                    missing_ct.append(f"{pid}: safe_id 매핑 없음")
                    continue
                ct_path = vol_root / "volumes_npy" / safe_id / "ct_hu.npy"
                if not ct_path.exists():
                    missing_ct.append(f"{pid}: ct_hu.npy 없음")
            if missing_ct:
                errors.append(
                    f"[GUARD 6] ct_hu.npy 누락 환자 {len(missing_ct)}명:\n"
                    + "\n".join(f"  {m}" for m in missing_ct[:10])
                    + (f"\n  ... ({len(missing_ct) - 10}명 더)" if len(missing_ct) > 10 else "")
                )
        except Exception as e:
            errors.append(f"[GUARD 6] volume 파일 확인 중 오류: {e}")

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print("\n[중단] guard 조건 미통과.", file=sys.stderr)
        sys.exit(1)

    print("[GUARD] 모든 guard 조건 통과.")


# ─────────────────────────────────────────────
# smoke 선택 규칙
# ─────────────────────────────────────────────
def select_smoke_rows(rows: pd.DataFrame, max_positive: int = 15, max_total: int = 30) -> pd.DataFrame:
    pos = rows[rows["sampling_label"] == "positive"].sort_values(
        "composite_rank_v2", ascending=False
    )
    neg = rows[rows["sampling_label"] == "hard_negative"].sort_values(
        "composite_rank_v2", ascending=False
    )
    pos_sel = pos.head(max_positive)
    neg_sel = neg.head(max_total - len(pos_sel))
    return pd.concat([pos_sel, neg_sel]).reset_index(drop=True)


# ─────────────────────────────────────────────
# dry-run 예상 보고 (smoke 모드 포함)
# ─────────────────────────────────────────────
def dry_run_report(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    safe_id_map: dict,
    vol_root: Path,
) -> None:
    holdout_set = get_holdout_set(split_df)
    stage1_pids = df[df["stage_split"] == "stage1_dev"]["patient_id"].unique().tolist()

    per_patient_counts = df.groupby("patient_id").size()
    total_expected = int(per_patient_counts.sum())
    n_pos = int((df["sampling_label"] == "positive").sum())
    n_hn  = int((df["sampling_label"] == "hard_negative").sum())

    print()
    print("=" * 70)
    print("  [DRY-RUN] S6-A 6ch Crop 예상 보고")
    print("=" * 70)
    print(f"  1. 총 생성 예정 crop 수  : {total_expected:,}")
    print(f"  2. patient 수            : {len(stage1_pids)}")
    print(f"  3. positive              : {n_pos:,}")
    print(f"     hard_negative         : {n_hn:,}")
    print(f"  4. 환자별 crop 수")
    print(f"       min    : {int(per_patient_counts.min())}")
    print(f"       median : {int(per_patient_counts.median())}")
    print(f"       mean   : {per_patient_counts.mean():.1f}")
    print(f"       max    : {int(per_patient_counts.max())}")
    print()
    print("  5. smoke 대상 5명 환자별 예상 crop 수:")
    smoke_total = 0
    for pid in SMOKE_PATIENTS:
        if pid in holdout_set:
            print(f"       {pid}: [SKIP] stage2_holdout")
            continue
        rows = df[df["patient_id"] == pid]
        if rows.empty:
            print(f"       {pid}: [WARN] manifest에 없음")
            continue
        sampled = select_smoke_rows(rows, max_positive=MAX_POSITIVE, max_total=MAX_PER_PATIENT)
        np_ = int((sampled["sampling_label"] == "positive").sum())
        nh  = int((sampled["sampling_label"] == "hard_negative").sum())
        print(f"       {pid}: total={len(sampled)} (pos={np_}, hn={nh})")
        smoke_total += len(sampled)
    print(f"       smoke 예상 합계: {smoke_total}")
    print()
    print(f"  6. 출력 경로 충돌 여부")
    smoke_exists = OUT_CROPS_SMOKE.exists()
    full_exists  = OUT_CROPS_FULL.exists()
    smoke_summ      = OUT_SUMMARY_SMOKE.exists()
    full_summ_json  = OUT_SUMMARY_FULL_JSON.exists()
    full_summ_csv   = OUT_SUMMARY_FULL_CSV.exists()
    full_summ_md    = OUT_SUMMARY_FULL_MD.exists()
    print(f"       crops_s6a_6ch_smoke    : {'이미 존재 (충돌)' if smoke_exists else '없음 (안전)'}")
    print(f"       crops_s6a_6ch_full     : {'이미 존재 (충돌)' if full_exists else '없음 (안전)'}")
    print(f"       smoke summary          : {'이미 존재 (충돌)' if smoke_summ else '없음 (안전)'}")
    print(f"       full summary (json)    : {'이미 존재 (충돌)' if full_summ_json else '없음 (안전)'}")
    print(f"       full summary (csv)     : {'이미 존재 (충돌)' if full_summ_csv else '없음 (안전)'}")
    print(f"       full summary (md)      : {'이미 존재 (충돌)' if full_summ_md else '없음 (안전)'}")
    print()
    print("  7. 6ch 구성")
    print("       ch0~2: lung window [-1350, 150] → [0, 1]")
    print("       ch3~5: mediastinal window [-160, 240] → [0, 1]")
    print("       z 기준: local_z (z boundary: edge-repeat)")
    print("       output shape: (6, 96, 96) float32")
    print()
    print("  8. 실행 명령 초안")
    print("       source ~/ai_env/bin/activate && \\")
    print("       python scripts/generate_s6a_crop_full_6ch.py --smoke")
    print("=" * 70)
    print()


# ─────────────────────────────────────────────
# 환자 처리 (6ch 생성)
# ─────────────────────────────────────────────
def process_patient(
    patient_id: str,
    rows: pd.DataFrame,
    volume_dir: Path,
    out_patient_dir: Path,
    dry_run: bool,
) -> list:
    ct_path = volume_dir / "ct_hu.npy"

    if not ct_path.exists():
        print(f"  [SKIP {patient_id}] ct_hu.npy 없음: {ct_path}", file=sys.stderr)
        return []

    ct = np.load(str(ct_path), mmap_mode="r")
    if ct.ndim != 3:
        print(f"  [SKIP {patient_id}] ct_hu.npy shape 오류: {ct.shape}", file=sys.stderr)
        return []

    _, img_h, img_w = ct.shape
    records = []

    for seq_idx, (_, row) in enumerate(rows.iterrows()):
        # stage2_holdout 이중 방어
        if str(row.get("stage_split", "")) == "stage2_holdout":
            print(f"[ABORT] stage2_holdout 감지: {patient_id}", file=sys.stderr)
            sys.exit(1)

        # guard: local_z 유효성
        if pd.isna(row["local_z"]):
            print(f"[ABORT] local_z NaN — patient={patient_id}", file=sys.stderr)
            sys.exit(1)
        z = int(row["local_z"])
        if z < 0 or z >= ct.shape[0]:
            print(
                f"[ABORT] local_z={z} 범위 이탈 (Z={ct.shape[0]}) — patient={patient_id}",
                file=sys.stderr,
            )
            sys.exit(1)

        label_int = 1 if row["sampling_label"] == "positive" else 0

        raw_si = row.get("slice_index", float("nan"))
        if pd.isna(raw_si):
            slice_idx = -1
            slice_index_valid = False
        else:
            slice_idx = int(raw_si)
            slice_index_valid = True

        cy = (row["y0"] + row["y1"]) / 2.0
        cx = (row["x0"] + row["x1"]) / 2.0
        cy0, cx0, cy1, cx1 = compute_fixed_crop_coords(cy, cx, CROP_SIZE, img_h, img_w)

        image = extract_6ch_crop(ct, z, cy0, cx0, cy1, cx1)

        # guard 7: image shape (6, 96, 96)
        if image.shape != (6, CROP_SIZE, CROP_SIZE):
            print(
                f"[ABORT] image shape 오류 {image.shape} (기대 (6,{CROP_SIZE},{CROP_SIZE}))"
                f" — patient={patient_id}",
                file=sys.stderr,
            )
            sys.exit(1)

        # guard 8: dtype float32
        if image.dtype != np.float32:
            print(
                f"[ABORT] image dtype 오류 {image.dtype} (기대 float32)"
                f" — patient={patient_id}",
                file=sys.stderr,
            )
            sys.exit(1)

        # guard 9: [0, 1] 범위 확인
        img_min = float(image.min())
        img_max = float(image.max())
        if img_min < -1e-6 or img_max > 1.0 + 1e-6:
            print(
                f"[ABORT] image 범위 이탈 [min={img_min:.6f}, max={img_max:.6f}]"
                f" — patient={patient_id}, z={z}",
                file=sys.stderr,
            )
            sys.exit(1)

        # guard 10: NaN/Inf 0개
        if not np.isfinite(image).all():
            print(f"[ABORT] image에 NaN/Inf 존재 — patient={patient_id}, z={z}", file=sys.stderr)
            sys.exit(1)

        rec = {
            "patient_id":               patient_id,
            "seq_idx":                  seq_idx,
            "label_int":                label_int,
            "sampling_label":           row["sampling_label"],
            "sampling_rule":            row["sampling_rule"],
            "local_z":                  z,
            "slice_index":              slice_idx,
            "slice_index_valid":        slice_index_valid,
            "z_source":                 "local_z",
            "orig_y0":                  int(row["y0"]),
            "orig_x0":                  int(row["x0"]),
            "orig_y1":                  int(row["y1"]),
            "orig_x1":                  int(row["x1"]),
            "crop_y0":                  cy0,
            "crop_x0":                  cx0,
            "crop_y1":                  cy1,
            "crop_x1":                  cx1,
            "image_shape":              list(image.shape),
            "image_min":                img_min,
            "image_max":                img_max,
            "score_original":           float(row["score_original"]),
            "score_valid950_weighted":  float(row["score_valid950_weighted"]),
            "score_valid950_soft":      float(row["score_valid950_soft"]),
            "composite_rank_v2":        float(row["composite_rank_v2"]),
            "position_bin":             str(row["position_bin"]),
            "z_level":                  str(row["z_level"]),
            "central_peripheral":       str(row["central_peripheral"]),
            "lesion_patch_ratio":       float(row["lesion_patch_ratio"]),
            "roi_inside_ratio":         float(row["roi_inside_ratio"]),
            "air_ratio_950":            float(row["air_ratio_950"]),
            "air_ratio_970":            float(row["air_ratio_970"]),
            "valid_ratio_roi_air950":   float(row["valid_ratio_roi_air950"]),
            "valid_ratio_roi_air970":   float(row["valid_ratio_roi_air970"]),
        }

        if not dry_run:
            out_patient_dir.mkdir(parents=True, exist_ok=True)
            npz_path = out_patient_dir / f"{seq_idx:06d}.npz"
            np.savez_compressed(
                npz_path,
                image=image,
                label=np.array(label_int),
                sampling_label=np.array(row["sampling_label"]),
                sampling_rule=np.array(row["sampling_rule"]),
                patient_id=np.array(patient_id),
                local_z=np.array(z),
                slice_index=np.array(slice_idx),
                slice_index_valid=np.array(slice_index_valid),
                z_source=np.array("local_z"),
                crop_coords=np.array([cy0, cx0, cy1, cx1]),
                orig_bbox=np.array([int(row["y0"]), int(row["x0"]), int(row["y1"]), int(row["x1"])]),
                score_original=np.array(float(row["score_original"])),
                score_valid950_weighted=np.array(float(row["score_valid950_weighted"])),
                score_valid950_soft=np.array(float(row["score_valid950_soft"])),
                composite_rank_v2=np.array(float(row["composite_rank_v2"])),
                position_bin=np.array(str(row["position_bin"])),
                z_level=np.array(str(row["z_level"])),
                central_peripheral=np.array(str(row["central_peripheral"])),
                lesion_patch_ratio=np.array(float(row["lesion_patch_ratio"])),
                roi_inside_ratio=np.array(float(row["roi_inside_ratio"])),
                air_ratio_950=np.array(float(row["air_ratio_950"])),
                air_ratio_970=np.array(float(row["air_ratio_970"])),
                valid_ratio_roi_air950=np.array(float(row["valid_ratio_roi_air950"])),
                valid_ratio_roi_air970=np.array(float(row["valid_ratio_roi_air970"])),
            )
            rec["saved_path"] = str(npz_path)
        else:
            rec["saved_path"] = "[dry-run: 저장 안 함]"

        records.append(rec)

    return records


# ─────────────────────────────────────────────
# summary 저장
# ─────────────────────────────────────────────
def save_summary(
    all_records: list,
    run_start: datetime,
    run_end: datetime,
    mode: str,
) -> None:
    elapsed = (run_end - run_start).total_seconds()
    df_out = pd.DataFrame(all_records)
    OUT_RPT_DIR.mkdir(parents=True, exist_ok=True)

    if mode == "smoke":
        json_path = OUT_SUMMARY_SMOKE
    else:
        json_path = OUT_SUMMARY_FULL_JSON

    # image 통계 계산 (run-full 모드에서 JSON/MD에 포함)
    img_min_global = float(df_out["image_min"].min()) if "image_min" in df_out.columns else None
    img_max_global = float(df_out["image_max"].max()) if "image_max" in df_out.columns else None
    image_range_valid = (
        img_min_global is not None
        and img_max_global is not None
        and img_min_global >= 0.0
        and img_max_global <= 1.0
    )

    summary_json = {
        "script":               "generate_s6a_crop_full_6ch.py",
        "mode":                 mode,
        "sampling_rule":        SAMPLING_RULE_TARGET,
        "crop_size":            CROP_SIZE,
        "image_key":            "image",
        "image_shape":          f"(6, {CROP_SIZE}, {CROP_SIZE})",
        "channels":             "ch0-2: lung_window, ch3-5: mediastinal_window",
        "lung_window":          f"[{LUNG_WIN_MIN}, {LUNG_WIN_MAX}] -> [0, 1]",
        "mediastinal_window":   f"[{MEDI_WIN_MIN}, {MEDI_WIN_MAX}] -> [0, 1]",
        "z_basis":              "local_z",
        "z_boundary_method":    "edge-repeat",
        "total_crops":          len(all_records),
        "positive_crops":       int((df_out["sampling_label"] == "positive").sum()),
        "hard_negative_crops":  int((df_out["sampling_label"] == "hard_negative").sum()),
        "n_patients":           int(df_out["patient_id"].nunique()),
        "run_start":            run_start.isoformat(),
        "run_end":              run_end.isoformat(),
        "elapsed_seconds":      round(elapsed, 2),
    }
    if mode == "smoke":
        summary_json["smoke_patients"] = SMOKE_PATIENTS
        summary_json["max_per_patient"] = MAX_PER_PATIENT
        summary_json["max_positive"] = MAX_POSITIVE

    if mode == "run-full":
        summary_json["image_min_global"]   = img_min_global
        summary_json["image_max_global"]   = img_max_global
        summary_json["image_range_valid"]  = image_range_valid
        summary_json["n_nan"]              = 0
        summary_json["n_inf"]              = 0

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {json_path}")

    # run-full 전용: CSV 및 MD 추가 저장
    if mode == "run-full":
        df_out.to_csv(OUT_SUMMARY_FULL_CSV, index=False, encoding="utf-8")
        print(f"[저장] {OUT_SUMMARY_FULL_CSV}")

        md_lines = [
            "# S6-A 6ch Crop Full Summary",
            "",
            "| 항목 | 값 |",
            "|---|---|",
            f"| mode | {summary_json['mode']} |",
            f"| total_crops | {summary_json['total_crops']:,} |",
            f"| n_patients | {summary_json['n_patients']} |",
            f"| positive_crops | {summary_json['positive_crops']:,} |",
            f"| hard_negative_crops | {summary_json['hard_negative_crops']:,} |",
            f"| image_key | {summary_json['image_key']} |",
            f"| image_shape | {summary_json['image_shape']} |",
            f"| channels | {summary_json['channels']} |",
            f"| lung_window | {summary_json['lung_window']} |",
            f"| mediastinal_window | {summary_json['mediastinal_window']} |",
            f"| z_basis | {summary_json['z_basis']} |",
            f"| z_boundary_method | {summary_json['z_boundary_method']} |",
            f"| image_min_global | {img_min_global} |",
            f"| image_max_global | {img_max_global} |",
            f"| image_range_valid | {image_range_valid} |",
            "| n_nan | 0 |",
            "| n_inf | 0 |",
            f"| run_start | {summary_json['run_start']} |",
            f"| run_end | {summary_json['run_end']} |",
            f"| elapsed_seconds | {summary_json['elapsed_seconds']} |",
        ]
        with open(OUT_SUMMARY_FULL_MD, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"[저장] {OUT_SUMMARY_FULL_MD}")

    print(f"\n[SUMMARY] total={len(all_records):,}, elapsed={elapsed:.1f}s")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="S6-A manifest 기반 6ch crop 생성 (smoke: 5명 한정)"
    )
    parser.add_argument("--dry-run",  action="store_true", help="파일 생성 없이 예상 수/분포 보고")
    parser.add_argument("--smoke",    action="store_true", help="smoke 5명만 실제 생성 (사용자 승인 후)")
    parser.add_argument("--run-full",          action="store_true", help="full 생성 모드 (--confirm-full-run과 함께 사용)")
    parser.add_argument("--confirm-full-run",  action="store_true", help="full 생성 이중 승인 플래그")
    args = parser.parse_args()

    # 인자 없음: 즉시 중단
    if not args.dry_run and not args.smoke and not args.run_full:
        print("[중단] --dry-run, --smoke, --run-full 중 하나가 필요합니다.", file=sys.stderr)
        print("  예시 (dry-run): python scripts/generate_s6a_crop_full_6ch.py --dry-run", file=sys.stderr)
        print("  예시 (smoke):   python scripts/generate_s6a_crop_full_6ch.py --smoke", file=sys.stderr)
        sys.exit(1)

    # --run-full 단독 실행 금지
    if args.run_full and not args.confirm_full_run:
        print("[중단] --run-full 단독 실행 금지.", file=sys.stderr)
        print("  full 생성은 --confirm-full-run 플래그를 함께 명시해야만 실행됩니다.", file=sys.stderr)
        print("  명령: python scripts/generate_s6a_crop_full_6ch.py --run-full --confirm-full-run", file=sys.stderr)
        sys.exit(1)

    if args.dry_run and args.smoke:
        print("[중단] --dry-run과 --smoke를 동시에 사용할 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        mode = "dry-run"
    elif args.smoke:
        mode = "smoke"
    else:
        mode = "run-full"

    print("=" * 70)
    print("  generate_s6a_crop_full_6ch.py")
    print(f"  모드: {mode}")
    print(f"  시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # guard_check
    guard_check(mode=mode)

    # 데이터 로드
    print("  manifest CSV 로드 중...")
    df = pd.read_csv(MANIFEST_CSV)
    print(f"  manifest rows: {len(df):,}")

    print("  stage split CSV 로드 중...")
    split_df    = pd.read_csv(STAGE_SPLIT_CSV)
    safe_id_map = build_safe_id_map(split_df)
    holdout_set = get_holdout_set(split_df)

    paths_cfg = load_paths_config()
    vol_root  = get_vol_root(paths_cfg)

    # dry-run: 예상 보고 후 종료
    if args.dry_run:
        dry_run_report(df, split_df, safe_id_map, vol_root)
        print("[DRY-RUN] 파일 저장 없음. 실제 실행은 --smoke 플래그와 사용자 승인 후 진행하세요.")
        return

    # full 생성 모드
    if mode == "run-full":
        stage1_pids = sorted(df[df["stage_split"] == "stage1_dev"]["patient_id"].unique().tolist())
        n_full = len(df)
        print()
        print("=" * 70)
        print("  [FULL RUN] S6-A 6ch Crop 전체 생성")
        print("=" * 70)
        print(f"  예상 생성 개수 : {n_full:,}")
        print(f"  대상 환자 수   : {len(stage1_pids)}명")
        print()
        print("  [주의] 이 명령은 ~130,659개의 npz 파일을 생성합니다.")
        print("         예상 디스크 사용량: 약 9 GB")
        print("         중단 시 기존 생성 파일을 삭제하지 않습니다.")
        print("         오류 발생 시 즉시 중단합니다.")
        print("=" * 70)
        print()

        all_records = []
        run_start = datetime.now()

        for patient_id in stage1_pids:
            safe_id = safe_id_map.get(patient_id)
            if safe_id is None:
                print(f"[WARN] safe_id 매핑 없음: {patient_id} — 건너뜀", file=sys.stderr)
                continue

            vol_dir = vol_root / "volumes_npy" / safe_id
            rows = df[df["patient_id"] == patient_id]
            if rows.empty:
                print(f"[WARN] manifest에 {patient_id} 없음 — 건너뜀", file=sys.stderr)
                continue

            stage = rows["stage_split"].iloc[0]
            n_pos = int((rows["sampling_label"] == "positive").sum())
            n_neg = int((rows["sampling_label"] == "hard_negative").sum())
            print(
                f"[{patient_id}] stage={stage}, rows={len(rows)} "
                f"(pos={n_pos}, hn={n_neg}), vol={vol_dir.name}"
            )

            out_patient_dir = OUT_CROPS_FULL / patient_id
            records = process_patient(
                patient_id=patient_id,
                rows=rows,
                volume_dir=vol_dir,
                out_patient_dir=out_patient_dir,
                dry_run=False,
            )
            all_records.extend(records)
            print(f"  → 저장 완료: {len(records)} crops")

        run_end = datetime.now()

        if all_records:
            save_summary(all_records, run_start, run_end, mode="run-full")
        else:
            print("\n[완료] 저장된 crop 없음.")
        return

    # smoke 실행
    all_records = []
    run_start   = datetime.now()

    for patient_id in SMOKE_PATIENTS:
        # stage2_holdout 방어
        if patient_id in holdout_set:
            print(f"[ABORT] stage2_holdout 환자: {patient_id}", file=sys.stderr)
            sys.exit(1)

        safe_id = safe_id_map.get(patient_id)
        if safe_id is None:
            print(f"[WARN] safe_id 매핑 없음: {patient_id} — 건너뜀", file=sys.stderr)
            continue

        vol_dir = vol_root / "volumes_npy" / safe_id
        rows = df[df["patient_id"] == patient_id]
        if rows.empty:
            print(f"[WARN] manifest에 {patient_id} 없음 — 건너뜀", file=sys.stderr)
            continue

        stage = rows["stage_split"].iloc[0]
        sampled = select_smoke_rows(rows, max_positive=MAX_POSITIVE, max_total=MAX_PER_PATIENT)
        n_pos = int((sampled["sampling_label"] == "positive").sum())
        n_neg = int((sampled["sampling_label"] == "hard_negative").sum())
        print(
            f"[{patient_id}] stage={stage}, 전체 rows={len(rows)}, "
            f"선택={len(sampled)} (pos={n_pos}, hn={n_neg}), vol={vol_dir.name}"
        )

        out_patient_dir = OUT_CROPS_SMOKE / patient_id
        records = process_patient(
            patient_id=patient_id,
            rows=sampled,
            volume_dir=vol_dir,
            out_patient_dir=out_patient_dir,
            dry_run=False,
        )
        all_records.extend(records)
        print(f"  → 저장 완료: {len(records)} crops")

    run_end = datetime.now()

    if all_records:
        save_summary(all_records, run_start, run_end, mode="smoke")
    else:
        print("\n[완료] 저장된 crop 없음.")


if __name__ == "__main__":
    main()
