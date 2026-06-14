#!/usr/bin/env python3
"""
generate_s6a_crop_smoke.py
==========================
S6-A selected candidate manifest 기반 2.5D crop smoke test 스크립트.

목적:
- rule_s6a_gs2_selected_candidate_manifest_dryrun.csv에서 5명 한정으로 2.5D crop npz 생성
- 저장 구조, label, 2.5D shape, 좌표 crop, local_z 기준 정상 여부 확인
- 전체 crop 생성 금지 (환자당 최대 30개, positive 최대 15개)

절대 금지:
- full crop 생성 금지
- 모델 학습 금지
- scoring 재실행 금지
- 기존 score CSV 수정 금지
- 기존 candidate/evaluation/crop 파일 수정 금지
- S6-A manifest 원본 수정 금지
- stage2_holdout 사용 금지
- weak 환자 전용 예외 추가 금지
- lesion local_z 직접 후보 추가 금지
- pip/conda install 금지

실행 명령 초안 (ChatGPT 검토 + 사용자 승인 후):
  [dry-run]:
    source ~/ai_env/bin/activate && \\
    python scripts/generate_s6a_crop_smoke.py --dry-run

  [실제 smoke — 별도 승인 후]:
    source ~/ai_env/bin/activate && \\
    python scripts/generate_s6a_crop_smoke.py

syntax check (실행 아님):
  python -m py_compile scripts/generate_s6a_crop_smoke.py
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

MANIFEST_CSV    = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/candidates/rule_s6a_gs2_selected_candidate_manifest_dryrun.csv"
VAL_SUMMARY_JSON = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/rule_s6a_manifest_validation_summary.json"
STAGE_SPLIT_CSV = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
PATHS_CONFIG    = REPO_ROOT / "configs/paths.local.yaml"
OUT_CROPS_DIR   = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_s6a_smoke"
OUT_RPT_DIR     = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports"

SAMPLING_RULE_TARGET = "S6-A_positive_all_hn_ratio2"
CROP_SIZE    = 96
CROP_HALF    = CROP_SIZE // 2  # 48
MAX_POSITIVE = 15
MAX_PER_PATIENT = 30

SMOKE_PATIENTS = [
    "LUNG1-140",   # 후보 폭주 환자
    "LUNG1-415",   # 이전 no-hit 회수 확인 환자
    "LUNG1-156",   # tiny/weak 계열 확인 환자
    "MSD_lung_071",  # weak 환자
    "MSD_lung_096",  # weak 환자
]


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
def guard_check(dry_run: bool) -> None:
    errors = []

    # guard 1: 출력 폴더 이미 있으면 중단
    if not dry_run and OUT_CROPS_DIR.exists():
        errors.append(f"[GUARD 1] crops_s6a_smoke 폴더가 이미 존재합니다: {OUT_CROPS_DIR}")

    # guard 1b: summary 파일 overwrite 방지 (dry-run 아닐 때만)
    if not dry_run:
        for _sp in [
            OUT_RPT_DIR / "crop_s6a_smoke_summary.csv",
            OUT_RPT_DIR / "crop_s6a_smoke_summary.json",
        ]:
            if _sp.exists():
                errors.append(f"[GUARD 1b] summary 파일이 이미 존재합니다: {_sp}")

    # guard 2: manifest 존재
    if not MANIFEST_CSV.exists():
        errors.append(f"[GUARD 2] manifest 없음: {MANIFEST_CSV}")

    # guard 3: validation summary JSON 존재
    if not VAL_SUMMARY_JSON.exists():
        errors.append(f"[GUARD 3] validation summary JSON 없음: {VAL_SUMMARY_JSON}")

    # guard 4: validation fail 0 확인
    if VAL_SUMMARY_JSON.exists():
        with open(VAL_SUMMARY_JSON, "r", encoding="utf-8") as f:
            val = json.load(f)
        n_fail = val.get("n_fail", -1)
        if n_fail != 0:
            errors.append(f"[GUARD 4] validation n_fail={n_fail} (0이어야 crop 생성 가능)")

    # guard 5: stage split CSV 존재
    if not STAGE_SPLIT_CSV.exists():
        errors.append(f"[GUARD 5] stage split CSV 없음: {STAGE_SPLIT_CSV}")

    # guard 6: paths.local.yaml 존재
    if not PATHS_CONFIG.exists():
        errors.append(f"[GUARD 6] paths.local.yaml 없음: {PATHS_CONFIG}")

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print("\n[중단] guard 조건 미통과.", file=sys.stderr)
        sys.exit(1)

    # guard 7: smoke 대상 5명이 manifest에 있는지 확인 (warn만)
    df_check = pd.read_csv(MANIFEST_CSV, usecols=["patient_id"])
    manifest_pids = set(df_check["patient_id"].unique().tolist())
    for pid in SMOKE_PATIENTS:
        if pid not in manifest_pids:
            print(f"[GUARD 7 WARN] {pid} 가 manifest에 없음", file=sys.stderr)

    # guard 8: smoke 대상에 holdout 포함 여부
    split_df = pd.read_csv(STAGE_SPLIT_CSV, usecols=["patient_id", "stage_split"])
    holdout_set = get_holdout_set(split_df)
    bad = [p for p in SMOKE_PATIENTS if p in holdout_set]
    if bad:
        print(f"[GUARD 8] stage2_holdout 환자 포함 감지 — 즉시 중단: {bad}", file=sys.stderr)
        sys.exit(1)

    print("[GUARD] 모든 guard 조건 통과.")


# ─────────────────────────────────────────────
# smoke 선택 규칙
# ─────────────────────────────────────────────
def select_smoke_rows(rows: pd.DataFrame, max_positive: int = 15, max_total: int = 30) -> pd.DataFrame:
    pos = rows[rows["sampling_label"] == "positive"]
    neg = rows[rows["sampling_label"] == "hard_negative"].sort_values(
        "composite_rank_v2", ascending=False
    )
    pos_sel = pos.head(max_positive)
    neg_sel = neg.head(max_total - len(pos_sel))
    return pd.concat([pos_sel, neg_sel]).reset_index(drop=True)


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
# 환자 처리
# ─────────────────────────────────────────────
def process_patient(
    patient_id: str,
    sampled: pd.DataFrame,
    volume_dir: Path,
    out_patient_dir: Path,
    dry_run: bool,
) -> list:
    ct_path   = volume_dir / "ct_hu.npy"
    roi_path  = volume_dir / "roi_0_0.npy"
    lmask_path = volume_dir / "lesion_mask_roi_0_0.npy"
    meta_path = volume_dir / "meta.json"

    # per-patient guard: 필수 파일 존재 확인
    missing = []
    for p in [ct_path, roi_path, lmask_path, meta_path]:
        if not p.exists():
            missing.append(p.name)
    if missing:
        print(f"  [SKIP {patient_id}] 필수 파일 없음: {missing}", file=sys.stderr)
        return []

    ct = np.load(str(ct_path), mmap_mode="r")
    if ct.ndim != 3:
        print(f"  [SKIP {patient_id}] ct_hu.npy shape 오류: {ct.shape}", file=sys.stderr)
        return []

    _, img_h, img_w = ct.shape
    records = []

    for seq_idx, (_, row) in enumerate(sampled.iterrows()):
        # per-patient guard: stage2_holdout
        if str(row.get("stage_split", "")) == "stage2_holdout":
            print(f"[ABORT] stage2_holdout 감지: {patient_id}", file=sys.stderr)
            sys.exit(1)

        # per-patient guard: local_z 유효성
        if pd.isna(row["local_z"]):
            print(f"[ABORT] local_z NaN — patient={patient_id}", file=sys.stderr)
            sys.exit(1)
        z = int(row["local_z"])
        if z < 0 or z >= ct.shape[0]:
            print(f"[ABORT] local_z={z} 범위 이탈 (Z={ct.shape[0]}) — patient={patient_id}", file=sys.stderr)
            sys.exit(1)

        # label 숫자 변환 (positive=1, hard_negative=0)
        label_int = 1 if row["sampling_label"] == "positive" else 0

        # slice_index NaN 안전 처리
        raw_si = row.get("slice_index", float("nan"))
        if pd.isna(raw_si):
            slice_idx = -1
            slice_index_valid = False
        else:
            slice_idx = int(raw_si)
            slice_index_valid = True

        # crop 좌표 계산 (bbox center 기준)
        cy = (row["y0"] + row["y1"]) / 2.0
        cx = (row["x0"] + row["x1"]) / 2.0
        cy0, cx0, cy1, cx1 = compute_fixed_crop_coords(cy, cx, CROP_SIZE, img_h, img_w)

        crop_arr = extract_2p5d_crop(ct, z, cy0, cx0, cy1, cx1)

        # per-patient guard: shape
        if crop_arr.shape != (3, CROP_SIZE, CROP_SIZE):
            print(
                f"[ABORT] crop shape 오류 {crop_arr.shape} (기대 (3,{CROP_SIZE},{CROP_SIZE}))"
                f" — patient={patient_id}",
                file=sys.stderr,
            )
            sys.exit(1)

        # per-patient guard: NaN/Inf
        if not np.isfinite(crop_arr).all():
            print(f"[ABORT] crop에 NaN/Inf 존재 — patient={patient_id}, z={z}", file=sys.stderr)
            sys.exit(1)

        rec = {
            "patient_id":          patient_id,
            "seq_idx":             seq_idx,
            "label_int":           label_int,
            "sampling_label":      row["sampling_label"],
            "sampling_rule":       row["sampling_rule"],
            "local_z":             z,
            "slice_index":         slice_idx,
            "slice_index_valid":   slice_index_valid,
            "z_source":            "local_z",
            "orig_y0":             int(row["y0"]),
            "orig_x0":             int(row["x0"]),
            "orig_y1":             int(row["y1"]),
            "orig_x1":             int(row["x1"]),
            "crop_y0":             cy0,
            "crop_x0":             cx0,
            "crop_y1":             cy1,
            "crop_x1":             cx1,
            "crop_shape":          list(crop_arr.shape),
            "score_original":      float(row["score_original"]),
            "score_valid950_weighted": float(row["score_valid950_weighted"]),
            "score_valid950_soft": float(row["score_valid950_soft"]),
            "composite_rank_v2":   float(row["composite_rank_v2"]),
            "position_bin":        str(row["position_bin"]),
            "z_level":             str(row["z_level"]),
            "central_peripheral":  str(row["central_peripheral"]),
            "lesion_patch_ratio":  float(row["lesion_patch_ratio"]),
            "roi_inside_ratio":    float(row["roi_inside_ratio"]),
            "air_ratio_950":       float(row["air_ratio_950"]),
            "air_ratio_970":       float(row["air_ratio_970"]),
            "valid_ratio_roi_air950": float(row["valid_ratio_roi_air950"]),
            "valid_ratio_roi_air970": float(row["valid_ratio_roi_air970"]),
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
def dry_run_report(df: pd.DataFrame, split_df: pd.DataFrame, safe_id_map: dict, vol_root: Path) -> None:
    holdout_set = get_holdout_set(split_df)
    total_expected = 0
    print()
    print("=" * 65)
    print("  [DRY-RUN] 환자별 예상 crop 수 보고")
    print("=" * 65)
    for pid in SMOKE_PATIENTS:
        if pid in holdout_set:
            print(f"  {pid}: [SKIP] stage2_holdout")
            continue
        rows = df[df["patient_id"] == pid]
        if rows.empty:
            print(f"  {pid}: [WARN] manifest에 없음")
            continue
        safe_id = safe_id_map.get(pid)
        if safe_id is None:
            print(f"  {pid}: [WARN] safe_id 매핑 없음")
            continue
        vol_dir = vol_root / "volumes_npy" / safe_id
        missing_files = []
        for fname in ["ct_hu.npy", "roi_0_0.npy", "lesion_mask_roi_0_0.npy", "meta.json"]:
            if not (vol_dir / fname).exists():
                missing_files.append(fname)
        if missing_files:
            print(f"  {pid}: [WARN] 필수 파일 없음: {missing_files}")
            continue

        sampled = select_smoke_rows(rows, max_positive=MAX_POSITIVE, max_total=MAX_PER_PATIENT)
        n_pos = int((sampled["sampling_label"] == "positive").sum())
        n_neg = int((sampled["sampling_label"] == "hard_negative").sum())
        stage = rows["stage_split"].iloc[0]
        print(
            f"  {pid}: total={len(sampled)} (pos={n_pos}, hn={n_neg})"
            f"  stage={stage}  vol={vol_dir.name}"
        )
        total_expected += len(sampled)
    print("-" * 65)
    print(f"  예상 총 crop 수: {total_expected}")
    print("=" * 65)
    print()


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="S6-A manifest 기반 2.5D crop smoke test (승인 후 실행)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일 생성 없이 예상 crop 수만 보고",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  generate_s6a_crop_smoke.py")
    print(f"  모드: {'dry-run' if args.dry_run else '실제 실행'}")
    print(f"  시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    guard_check(dry_run=args.dry_run)

    # 데이터 로드
    print("  manifest CSV 로드 중...")
    df = pd.read_csv(MANIFEST_CSV)
    print(f"  manifest rows: {len(df):,}")

    print("  stage split CSV 로드 중...")
    split_df = pd.read_csv(STAGE_SPLIT_CSV)

    safe_id_map = build_safe_id_map(split_df)
    holdout_set = get_holdout_set(split_df)

    paths_cfg = load_paths_config()
    vol_root  = get_vol_root(paths_cfg)

    # dry-run: 예상 보고 후 종료
    if args.dry_run:
        dry_run_report(df, split_df, safe_id_map, vol_root)
        print("[DRY-RUN] 파일 저장 없음. 실제 실행은 사용자 승인 후 진행하세요.")
        return

    # 실제 실행
    all_records = []
    run_start = datetime.now()

    for patient_id in SMOKE_PATIENTS:
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

        out_patient_dir = OUT_CROPS_DIR / patient_id
        records = process_patient(
            patient_id=patient_id,
            sampled=sampled,
            volume_dir=vol_dir,
            out_patient_dir=out_patient_dir,
            dry_run=False,
        )
        all_records.extend(records)
        print(f"  → 저장 완료: {len(records)} crops")

    run_end = datetime.now()
    elapsed = (run_end - run_start).total_seconds()

    # summary 저장
    if all_records:
        df_out = pd.DataFrame(all_records)
        OUT_RPT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path  = OUT_RPT_DIR / "crop_s6a_smoke_summary.csv"
        json_path = OUT_RPT_DIR / "crop_s6a_smoke_summary.json"
        df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")

        summary_json = {
            "script":           "generate_s6a_crop_smoke.py",
            "dry_run":          False,
            "smoke_patients":   SMOKE_PATIENTS,
            "sampling_rule":    SAMPLING_RULE_TARGET,
            "crop_size":        CROP_SIZE,
            "max_per_patient":  MAX_PER_PATIENT,
            "max_positive":     MAX_POSITIVE,
            "total_crops":      len(all_records),
            "positive_crops":   int((df_out["sampling_label"] == "positive").sum()),
            "hard_negative_crops": int((df_out["sampling_label"] == "hard_negative").sum()),
            "z_basis":          "local_z",
            "z_boundary_method": "edge-repeat",
            "crop_shape":       f"(3, {CROP_SIZE}, {CROP_SIZE})",
            "run_start":        run_start.isoformat(),
            "run_end":          run_end.isoformat(),
            "elapsed_seconds":  round(elapsed, 2),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_json, f, ensure_ascii=False, indent=2)

        print(f"\n[저장] {csv_path}")
        print(f"[저장] {json_path}")
        print(f"\n[SUMMARY] total={len(all_records)}, elapsed={elapsed:.1f}s")
    else:
        print("\n[완료] 저장된 crop 없음.")


if __name__ == "__main__":
    main()
