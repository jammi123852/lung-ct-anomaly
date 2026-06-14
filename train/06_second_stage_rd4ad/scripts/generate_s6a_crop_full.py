#!/usr/bin/env python3
"""
generate_s6a_crop_full.py
=========================
S6-A selected candidate manifest 기반 full 2.5D crop 생성 스크립트.

목적:
- rule_s6a_gs2_selected_candidate_manifest_dryrun.csv 전체 130,659개 후보를 2.5D crop npz로 생성
- smoke와 동일한 npz 구조 유지
- local_z 기준 crop
- label은 숫자 0/1 저장, sampling_label은 문자열로 별도 저장

실행 모드:
- 인자 없음: 중단 (보호)
- --dry-run: 실제 npz 생성 없이 예상 개수/분포/디스크량 보고
- --run: 실제 full crop 생성 (사용자 승인 후)

절대 금지:
- full crop 생성 금지 (--run 없이)
- 모델 학습 금지
- scoring 재실행 금지
- 기존 score CSV 수정 금지
- 기존 candidate/evaluation/crop 파일 수정 금지
- S6-A manifest 원본 수정 금지
- smoke crop 파일 수정/삭제 금지
- stage2_holdout 사용 금지
- weak 환자 전용 예외 추가 금지
- lesion local_z 직접 후보 추가 금지
- pip/conda install 금지

실행 명령 (사용자 승인 후):
  [dry-run]:
    source ~/ai_env/bin/activate && \\
    python scripts/generate_s6a_crop_full.py --dry-run

  [실제 full crop — 별도 승인 후]:
    source ~/ai_env/bin/activate && \\
    python scripts/generate_s6a_crop_full.py --run

syntax check (실행 아님):
  python -m py_compile scripts/generate_s6a_crop_full.py
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

MANIFEST_CSV        = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/candidates/rule_s6a_gs2_selected_candidate_manifest_dryrun.csv"
VAL_SUMMARY_JSON    = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/rule_s6a_manifest_validation_summary.json"
SMOKE_VAL_JSON      = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/crop_s6a_smoke_validation_summary.json"
SMOKE_VAL_MD        = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/crop_s6a_smoke_validation_summary.md"
STAGE_SPLIT_CSV     = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
PATHS_CONFIG        = REPO_ROOT / "configs/paths.local.yaml"
OUT_CROPS_DIR       = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_s6a_full"
OUT_RPT_DIR         = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports"

EXPECTED_TOTAL_ROWS   = 130_659
EXPECTED_N_POSITIVE   = 43_553
EXPECTED_N_HN         = 87_106
EXPECTED_N_STAGE1_DEV = 154
EXPECTED_N_HOLDOUT    = 0

SAMPLING_RULE_TARGET = "S6-A_positive_all_hn_ratio2"
CROP_SIZE = 96

SMOKE_NPZ_AVG_BYTES = 51_000  # 약 51KB (smoke npz 실측 기준)


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
# guard_check
# ─────────────────────────────────────────────
def guard_check(mode: str, df: pd.DataFrame | None = None) -> None:
    """
    mode: 'dry-run' | 'run'
    df: manifest DataFrame (guard 8~11에서 사용). None이면 guard 8~11 스킵.
    """
    errors = []

    # guard 1: 출력 폴더 이미 있으면 중단 (--run 시)
    if mode == "run" and OUT_CROPS_DIR.exists():
        errors.append(f"[GUARD 1] crops_s6a_full 폴더가 이미 존재합니다: {OUT_CROPS_DIR}")

    # guard 2: full summary 파일 이미 있으면 중단 (--run 시)
    if mode == "run":
        for _sp in [
            OUT_RPT_DIR / "crop_s6a_full_summary.csv",
            OUT_RPT_DIR / "crop_s6a_full_summary.json",
            OUT_RPT_DIR / "crop_s6a_full_summary.md",
        ]:
            if _sp.exists():
                errors.append(f"[GUARD 2] full summary 파일이 이미 존재합니다: {_sp}")

    # guard 3: manifest 존재
    if not MANIFEST_CSV.exists():
        errors.append(f"[GUARD 3] manifest 없음: {MANIFEST_CSV}")

    # guard 4: manifest validation summary 존재
    if not VAL_SUMMARY_JSON.exists():
        errors.append(f"[GUARD 4] manifest validation summary JSON 없음: {VAL_SUMMARY_JSON}")

    # guard 5: manifest validation n_fail == 0
    if VAL_SUMMARY_JSON.exists():
        with open(VAL_SUMMARY_JSON, "r", encoding="utf-8") as f:
            val = json.load(f)
        n_fail = val.get("n_fail", -1)
        if n_fail != 0:
            errors.append(f"[GUARD 5] manifest validation n_fail={n_fail} (0이어야 함)")

    # guard 6: smoke validation summary 존재
    if not SMOKE_VAL_JSON.exists():
        errors.append(f"[GUARD 6] smoke validation summary JSON 없음: {SMOKE_VAL_JSON}")

    # guard 7: smoke validation overall_pass 확인
    if SMOKE_VAL_JSON.exists():
        with open(SMOKE_VAL_JSON, "r", encoding="utf-8") as f:
            sv = json.load(f)
        if not sv.get("overall_pass", False):
            errors.append(f"[GUARD 7] smoke validation overall_pass=False (전체 통과 필요)")

    # guard 8~11: manifest 수치 확인 (df 제공 시)
    if df is not None:
        total_rows = len(df)
        if total_rows != EXPECTED_TOTAL_ROWS:
            errors.append(
                f"[GUARD 8] manifest row 수 불일치: {total_rows:,} (기대 {EXPECTED_TOTAL_ROWS:,})"
            )

        n_pos = int((df["sampling_label"] == "positive").sum())
        n_hn  = int((df["sampling_label"] == "hard_negative").sum())
        if n_pos != EXPECTED_N_POSITIVE:
            errors.append(
                f"[GUARD 9a] positive 수 불일치: {n_pos:,} (기대 {EXPECTED_N_POSITIVE:,})"
            )
        if n_hn != EXPECTED_N_HN:
            errors.append(
                f"[GUARD 9b] hard_negative 수 불일치: {n_hn:,} (기대 {EXPECTED_N_HN:,})"
            )

        n_stage1 = int((df["stage_split"] == "stage1_dev").sum() > 0)
        stage1_pids = df[df["stage_split"] == "stage1_dev"]["patient_id"].nunique()
        if stage1_pids != EXPECTED_N_STAGE1_DEV:
            errors.append(
                f"[GUARD 10] stage1_dev 환자 수 불일치: {stage1_pids} (기대 {EXPECTED_N_STAGE1_DEV})"
            )

        n_holdout = int((df["stage_split"] == "stage2_holdout").sum())
        if n_holdout != EXPECTED_N_HOLDOUT:
            errors.append(
                f"[GUARD 11] stage2_holdout row 수: {n_holdout} (반드시 0이어야 함)"
            )

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print("\n[중단] guard 조건 미통과.", file=sys.stderr)
        sys.exit(1)

    print("[GUARD] 모든 guard 조건 통과.")


# ─────────────────────────────────────────────
# crop 좌표 계산
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
# 2.5D crop 추출
# ─────────────────────────────────────────────
def extract_2p5d_crop(
    ct: np.ndarray, z: int, y0: int, x0: int, y1: int, x1: int
) -> np.ndarray:
    Z = ct.shape[0]
    z_prev = max(0, z - 1)
    z_next = min(Z - 1, z + 1)
    slices = [
        ct[z_prev, y0:y1, x0:x1].astype(np.float32),
        ct[z,      y0:y1, x0:x1].astype(np.float32),
        ct[z_next, y0:y1, x0:x1].astype(np.float32),
    ]
    return np.stack(slices, axis=0)  # (3, 96, 96)


# ─────────────────────────────────────────────
# local_z / bbox 전수 검증 (--run 전)
# ─────────────────────────────────────────────
def validate_manifest_rows(df: pd.DataFrame) -> None:
    errors = []

    # guard 12: local_z NaN/음수
    lz = df["local_z"]
    nan_count = int(lz.isna().sum())
    neg_count = int((lz.dropna() < 0).sum())
    if nan_count > 0:
        errors.append(f"[GUARD 12a] local_z NaN {nan_count}건")
    if neg_count > 0:
        errors.append(f"[GUARD 12b] local_z 음수 {neg_count}건")

    # guard 13: bbox NaN/음수/순서
    for col in ["y0", "x0", "y1", "x1"]:
        nc = int(df[col].isna().sum())
        if nc > 0:
            errors.append(f"[GUARD 13a] bbox {col} NaN {nc}건")
        ng = int((df[col].dropna() < 0).sum())
        if ng > 0:
            errors.append(f"[GUARD 13b] bbox {col} 음수 {ng}건")

    bad_order_y = int((df["y1"] <= df["y0"]).sum())
    bad_order_x = int((df["x1"] <= df["x0"]).sum())
    if bad_order_y > 0:
        errors.append(f"[GUARD 13c] y1<=y0 {bad_order_y}건")
    if bad_order_x > 0:
        errors.append(f"[GUARD 13d] x1<=x0 {bad_order_x}건")

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print("\n[중단] manifest row 검증 미통과.", file=sys.stderr)
        sys.exit(1)

    print("[GUARD 12-13] local_z / bbox 검증 통과.")


# ─────────────────────────────────────────────
# 필수 volume 파일 누락 환자 확인 (--run 전)
# ─────────────────────────────────────────────
def validate_volume_files(
    patient_ids: list, safe_id_map: dict, vol_root: Path
) -> None:
    missing_patients = []
    for pid in patient_ids:
        safe_id = safe_id_map.get(pid)
        if safe_id is None:
            missing_patients.append(f"{pid}: safe_id 매핑 없음")
            continue
        vol_dir = vol_root / "volumes_npy" / safe_id
        for fname in ["ct_hu.npy", "roi_0_0.npy", "lesion_mask_roi_0_0.npy", "meta.json"]:
            if not (vol_dir / fname).exists():
                missing_patients.append(f"{pid}: {fname} 없음")
    if missing_patients:
        print("[GUARD 14] 필수 volume 파일 누락 환자:", file=sys.stderr)
        for m in missing_patients:
            print(f"  {m}", file=sys.stderr)
        print("\n[중단] volume 파일 누락 환자 존재.", file=sys.stderr)
        sys.exit(1)
    print("[GUARD 14] 모든 환자 volume 파일 존재 확인.")


# ─────────────────────────────────────────────
# 환자 처리
# ─────────────────────────────────────────────
def process_patient(
    patient_id: str,
    rows: pd.DataFrame,
    volume_dir: Path,
    out_patient_dir: Path,
    dry_run: bool,
) -> list:
    ct_path    = volume_dir / "ct_hu.npy"
    roi_path   = volume_dir / "roi_0_0.npy"
    lmask_path = volume_dir / "lesion_mask_roi_0_0.npy"
    meta_path  = volume_dir / "meta.json"

    missing = [p.name for p in [ct_path, roi_path, lmask_path, meta_path] if not p.exists()]
    if missing:
        print(f"  [SKIP {patient_id}] 필수 파일 없음: {missing}", file=sys.stderr)
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

        # guard 12: local_z 유효성
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

        crop_arr = extract_2p5d_crop(ct, z, cy0, cx0, cy1, cx1)

        # guard 15: shape
        if crop_arr.shape != (3, CROP_SIZE, CROP_SIZE):
            print(
                f"[ABORT] crop shape 오류 {crop_arr.shape} (기대 (3,{CROP_SIZE},{CROP_SIZE}))"
                f" — patient={patient_id}",
                file=sys.stderr,
            )
            sys.exit(1)

        # guard 16: NaN/Inf
        if not np.isfinite(crop_arr).all():
            print(f"[ABORT] crop에 NaN/Inf 존재 — patient={patient_id}, z={z}", file=sys.stderr)
            sys.exit(1)

        rec = {
            "patient_id":              patient_id,
            "seq_idx":                 seq_idx,
            "label_int":               label_int,
            "sampling_label":          row["sampling_label"],
            "sampling_rule":           row["sampling_rule"],
            "local_z":                 z,
            "slice_index":             slice_idx,
            "slice_index_valid":       slice_index_valid,
            "z_source":                "local_z",
            "orig_y0":                 int(row["y0"]),
            "orig_x0":                 int(row["x0"]),
            "orig_y1":                 int(row["y1"]),
            "orig_x1":                 int(row["x1"]),
            "crop_y0":                 cy0,
            "crop_x0":                 cx0,
            "crop_y1":                 cy1,
            "crop_x1":                 cx1,
            "crop_shape":              list(crop_arr.shape),
            "score_original":          float(row["score_original"]),
            "score_valid950_weighted": float(row["score_valid950_weighted"]),
            "score_valid950_soft":     float(row["score_valid950_soft"]),
            "composite_rank_v2":       float(row["composite_rank_v2"]),
            "position_bin":            str(row["position_bin"]),
            "z_level":                 str(row["z_level"]),
            "central_peripheral":      str(row["central_peripheral"]),
            "lesion_patch_ratio":      float(row["lesion_patch_ratio"]),
            "roi_inside_ratio":        float(row["roi_inside_ratio"]),
            "air_ratio_950":           float(row["air_ratio_950"]),
            "air_ratio_970":           float(row["air_ratio_970"]),
            "valid_ratio_roi_air950":  float(row["valid_ratio_roi_air950"]),
            "valid_ratio_roi_air970":  float(row["valid_ratio_roi_air970"]),
        }

        if not dry_run:
            out_patient_dir.mkdir(parents=True, exist_ok=True)
            npz_path = out_patient_dir / f"{seq_idx:06d}.npz"
            np.savez_compressed(
                npz_path,
                crop=crop_arr,
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
# dry-run 예상 보고
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

    over2000 = per_patient_counts[per_patient_counts > 2000].sort_values(ascending=False)
    lung1_140_count = int(per_patient_counts.get("LUNG1-140", 0))

    estimated_bytes = total_expected * SMOKE_NPZ_AVG_BYTES
    estimated_gb    = estimated_bytes / (1024 ** 3)

    # 실행 시간 추정 (smoke 150개 = 0.23s 기준 → 단순 비례, I/O 포함 보수적으로 10배)
    smoke_seconds_per_crop = 0.23 / 150
    estimated_seconds_optimistic = total_expected * smoke_seconds_per_crop
    estimated_minutes_conservative = (estimated_seconds_optimistic * 10) / 60

    print()
    print("=" * 70)
    print("  [DRY-RUN] S6-A Full Crop 예상 보고")
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
    print(f"  5. 2000개 초과 환자 목록 ({len(over2000)}명):")
    if len(over2000) > 0:
        for pid, cnt in over2000.items():
            print(f"       {pid}: {cnt:,}")
    else:
        print("       없음")
    print()
    print(f"  6. LUNG1-140 후보 수     : {lung1_140_count:,}")
    print()
    print(f"  7. 예상 디스크 사용량")
    print(f"       smoke npz 평균 크기 : {SMOKE_NPZ_AVG_BYTES/1024:.0f} KB")
    print(f"       추정               : {estimated_gb:.1f} GB (보수적 범위: 6~10 GB)")
    print()
    print(f"  8. 예상 실행 시간")
    print(f"       낙관적 (smoke 비례) : {estimated_seconds_optimistic:.0f}초")
    print(f"       보수적 (I/O 포함)  : {estimated_minutes_conservative:.0f}분")
    print()
    print(f"  9. OOM 위험")
    print(f"       mmap_mode='r' 사용, 환자별 ct 1개씩 로드 → OOM 위험 낮음")
    print(f"       단, max 환자({int(per_patient_counts.max())}개)는 연속 처리 중 주의")
    print()
    print(f"  10. 출력 경로 충돌 여부")
    crops_exists = OUT_CROPS_DIR.exists()
    summary_conflicts = [
        p for p in [
            OUT_RPT_DIR / "crop_s6a_full_summary.csv",
            OUT_RPT_DIR / "crop_s6a_full_summary.json",
            OUT_RPT_DIR / "crop_s6a_full_summary.md",
        ] if p.exists()
    ]
    print(f"       crops_s6a_full 폴더 : {'이미 존재 (충돌)' if crops_exists else '없음 (안전)'}")
    if summary_conflicts:
        for p in summary_conflicts:
            print(f"       {p.name}: 이미 존재 (충돌)")
    else:
        print(f"       summary 파일       : 없음 (안전)")
    print()
    print(f"  11. full 실행 명령 초안")
    print(f"       source ~/ai_env/bin/activate && \\")
    print(f"       python scripts/generate_s6a_crop_full.py --run")
    print("=" * 70)
    print()


# ─────────────────────────────────────────────
# summary 저장
# ─────────────────────────────────────────────
def save_summary(all_records: list, run_start: datetime, run_end: datetime) -> None:
    elapsed = (run_end - run_start).total_seconds()
    df_out = pd.DataFrame(all_records)
    OUT_RPT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path  = OUT_RPT_DIR / "crop_s6a_full_summary.csv"
    json_path = OUT_RPT_DIR / "crop_s6a_full_summary.json"
    md_path   = OUT_RPT_DIR / "crop_s6a_full_summary.md"

    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary_json = {
        "script":               "generate_s6a_crop_full.py",
        "dry_run":              False,
        "sampling_rule":        SAMPLING_RULE_TARGET,
        "crop_size":            CROP_SIZE,
        "total_crops":          len(all_records),
        "positive_crops":       int((df_out["sampling_label"] == "positive").sum()),
        "hard_negative_crops":  int((df_out["sampling_label"] == "hard_negative").sum()),
        "n_patients":           int(df_out["patient_id"].nunique()),
        "z_basis":              "local_z",
        "z_boundary_method":    "edge-repeat",
        "crop_shape":           f"(3, {CROP_SIZE}, {CROP_SIZE})",
        "run_start":            run_start.isoformat(),
        "run_end":              run_end.isoformat(),
        "elapsed_seconds":      round(elapsed, 2),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    n_pos = summary_json["positive_crops"]
    n_hn  = summary_json["hard_negative_crops"]
    md_lines = [
        "# crop_s6a_full_summary",
        "",
        f"| 항목 | 값 |",
        f"|------|-----|",
        f"| 총 crop 수 | {len(all_records):,} |",
        f"| positive | {n_pos:,} |",
        f"| hard_negative | {n_hn:,} |",
        f"| 환자 수 | {summary_json['n_patients']} |",
        f"| crop shape | (3, {CROP_SIZE}, {CROP_SIZE}) |",
        f"| z 기준 | local_z |",
        f"| 실행 시간 | {elapsed:.1f}초 |",
        "",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[저장] {csv_path}")
    print(f"[저장] {json_path}")
    print(f"[저장] {md_path}")
    print(f"\n[SUMMARY] total={len(all_records):,}, elapsed={elapsed:.1f}s")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="S6-A manifest 기반 full 2.5D crop 생성 (승인 후 실행)"
    )
    parser.add_argument("--dry-run", action="store_true", help="파일 생성 없이 예상 수/분포/디스크 보고")
    parser.add_argument("--run",     action="store_true", help="실제 full crop 생성 (사용자 승인 후)")
    args = parser.parse_args()

    if not args.dry_run and not args.run:
        print("[중단] --dry-run 또는 --run 플래그가 필요합니다.", file=sys.stderr)
        print("  예시 (dry-run): python scripts/generate_s6a_crop_full.py --dry-run", file=sys.stderr)
        print("  예시 (실행):    python scripts/generate_s6a_crop_full.py --run", file=sys.stderr)
        sys.exit(1)

    if args.dry_run and args.run:
        print("[중단] --dry-run과 --run을 동시에 사용할 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    mode = "dry-run" if args.dry_run else "run"

    print("=" * 70)
    print("  generate_s6a_crop_full.py")
    print(f"  모드: {mode}")
    print(f"  시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 기본 guard (df 없이)
    guard_check(mode=mode, df=None)

    # manifest 로드
    print("  manifest CSV 로드 중...")
    df = pd.read_csv(MANIFEST_CSV)
    print(f"  manifest rows: {len(df):,}")

    # manifest 수치 guard
    guard_check(mode=mode, df=df)

    # stage split 로드
    print("  stage split CSV 로드 중...")
    split_df   = pd.read_csv(STAGE_SPLIT_CSV)
    safe_id_map = build_safe_id_map(split_df)
    holdout_set = get_holdout_set(split_df)

    paths_cfg = load_paths_config()
    vol_root  = get_vol_root(paths_cfg)

    # dry-run
    if args.dry_run:
        dry_run_report(df, split_df, safe_id_map, vol_root)
        print("[DRY-RUN] 파일 저장 없음. 실제 실행은 --run 플래그와 사용자 승인 후 진행하세요.")
        return

    # --run: 추가 전처리 검증
    validate_manifest_rows(df)

    stage1_pids = df[df["stage_split"] == "stage1_dev"]["patient_id"].unique().tolist()
    validate_volume_files(stage1_pids, safe_id_map, vol_root)

    # 실제 full crop 생성
    all_records = []
    run_start   = datetime.now()
    total_pids  = len(stage1_pids)

    for i, patient_id in enumerate(stage1_pids, 1):
        if patient_id in holdout_set:
            print(f"[ABORT] stage2_holdout 환자: {patient_id}", file=sys.stderr)
            sys.exit(1)

        safe_id = safe_id_map.get(patient_id)
        if safe_id is None:
            print(f"[WARN] safe_id 매핑 없음: {patient_id} — 건너뜀", file=sys.stderr)
            continue

        vol_dir = vol_root / "volumes_npy" / safe_id
        rows = df[df["patient_id"] == patient_id].reset_index(drop=True)
        if rows.empty:
            print(f"[WARN] manifest에 {patient_id} 없음 — 건너뜀", file=sys.stderr)
            continue

        n_pos = int((rows["sampling_label"] == "positive").sum())
        n_hn  = int((rows["sampling_label"] == "hard_negative").sum())
        print(
            f"[{i}/{total_pids}] {patient_id}: total={len(rows)} (pos={n_pos}, hn={n_hn})"
            f"  vol={vol_dir.name}"
        )

        out_patient_dir = OUT_CROPS_DIR / patient_id
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
        save_summary(all_records, run_start, run_end)
    else:
        print("\n[완료] 저장된 crop 없음.")


if __name__ == "__main__":
    main()
