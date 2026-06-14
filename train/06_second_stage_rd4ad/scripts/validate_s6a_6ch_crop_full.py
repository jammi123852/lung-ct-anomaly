#!/usr/bin/env python3
"""
validate_s6a_6ch_crop_full.py
=============================
S6-A 6ch full crop 130,659개 전수 validation 스크립트.

실행 모드:
- --dry-run : 전수 검증 수행, summary 저장 없음
- 인자 없음 : 전수 검증 + validation summary CSV/JSON/MD 저장 (사용자 승인 후)

절대 금지:
- npz 수정/삭제/재생성 금지
- 기존 crops_s6a_full / crops_s6a_6ch_smoke 접근 금지
- 학습/forward/scoring 금지
- stage2_holdout 접근 금지
- pip/conda install 금지

실행 명령:
  [dry-run]:
    source ~/ai_env/bin/activate && \\
    python scripts/validate_s6a_6ch_crop_full.py --dry-run

  [저장 — 사용자 승인 후]:
    source ~/ai_env/bin/activate && \\
    python scripts/validate_s6a_6ch_crop_full.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 경로 상수
# ─────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

FULL_CROP_DIR   = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_s6a_6ch_full"
SUMMARY_CSV     = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/crop_s6a_6ch_full_summary.csv"
SUMMARY_JSON    = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/crop_s6a_6ch_full_summary.json"
SUMMARY_MD      = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/crop_s6a_6ch_full_summary.md"
MANIFEST_CSV    = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/candidates/rule_s6a_gs2_selected_candidate_manifest_dryrun.csv"
STAGE_SPLIT_CSV = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"

OUT_RPT_DIR      = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports"
VAL_SUMMARY_CSV  = OUT_RPT_DIR / "crop_s6a_6ch_full_validation_summary.csv"
VAL_SUMMARY_JSON = OUT_RPT_DIR / "crop_s6a_6ch_full_validation_summary.json"
VAL_SUMMARY_MD   = OUT_RPT_DIR / "crop_s6a_6ch_full_validation_summary.md"

# ─────────────────────────────────────────────
# 기대값
# ─────────────────────────────────────────────
EXPECTED_TOTAL    = 130659
EXPECTED_PATIENTS = 154
EXPECTED_POS      = 43553
EXPECTED_HN       = 87106

REQUIRED_KEYS = [
    "image", "label", "sampling_label", "sampling_rule", "patient_id",
    "local_z", "slice_index", "slice_index_valid", "z_source",
    "crop_coords", "orig_bbox", "score_original", "score_valid950_weighted",
    "score_valid950_soft", "composite_rank_v2", "position_bin", "z_level",
    "central_peripheral", "lesion_patch_ratio", "roi_inside_ratio",
    "air_ratio_950", "air_ratio_970", "valid_ratio_roi_air950", "valid_ratio_roi_air970",
]
FORBIDDEN_KEYS = ["crop"]


# ─────────────────────────────────────────────
# guard_check
# ─────────────────────────────────────────────
def guard_check(dry_run: bool) -> None:
    errors = []

    if not FULL_CROP_DIR.exists():
        errors.append(f"[GUARD 1] full crop 폴더 없음: {FULL_CROP_DIR}")

    for p, label in [(SUMMARY_CSV, "CSV"), (SUMMARY_JSON, "JSON"), (SUMMARY_MD, "MD")]:
        if not p.exists():
            errors.append(f"[GUARD 2] full summary {label} 없음: {p}")

    if not MANIFEST_CSV.exists():
        errors.append(f"[GUARD 3] manifest 없음: {MANIFEST_CSV}")

    if not STAGE_SPLIT_CSV.exists():
        errors.append(f"[GUARD 4] stage split 없음: {STAGE_SPLIT_CSV}")

    if not dry_run:
        for p, label in [(VAL_SUMMARY_CSV, "CSV"), (VAL_SUMMARY_JSON, "JSON"), (VAL_SUMMARY_MD, "MD")]:
            if p.exists():
                errors.append(f"[GUARD 5] validation summary {label} 이미 존재: {p}")

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print("\n[중단] guard 조건 미통과.", file=sys.stderr)
        sys.exit(1)

    print("[GUARD] 모든 guard 조건 통과.")


# ─────────────────────────────────────────────
# 전수 검증
# ─────────────────────────────────────────────
def run_validation() -> dict:
    # stage split 로드 — holdout 확인
    split_df    = pd.read_csv(STAGE_SPLIT_CSV)
    holdout_pids = set(split_df.loc[split_df["stage_split"] == "stage2_holdout", "patient_id"])

    # manifest 로드
    manifest_df = pd.read_csv(MANIFEST_CSV)
    manifest_pids_set = set(manifest_df["patient_id"].unique())

    # stage2_holdout × manifest 교집합 확인
    holdout_in_manifest = manifest_pids_set & holdout_pids
    if holdout_in_manifest:
        print(f"[ABORT] stage2_holdout 환자가 manifest에 존재: {sorted(holdout_in_manifest)}", file=sys.stderr)
        sys.exit(1)

    # summary JSON 로드
    with open(SUMMARY_JSON, encoding="utf-8") as f:
        summary_json = json.load(f)

    # summary CSV 로드
    summary_csv_df = pd.read_csv(SUMMARY_CSV)

    # 폴더/파일 탐색
    patient_dirs  = sorted([d for d in FULL_CROP_DIR.iterdir() if d.is_dir()])
    all_npz_files = sorted(FULL_CROP_DIR.rglob("*.npz"))
    n_npz = len(all_npz_files)

    errors = []

    # ── 파일 수준 검증 ──────────────────────────

    # 1. npz 총 개수
    if n_npz != EXPECTED_TOTAL:
        errors.append(f"[FAIL 1] npz 총 개수 불일치: 실제={n_npz}, 예상={EXPECTED_TOTAL}")
    else:
        print(f"[OK 1] npz 총 개수: {n_npz:,}")

    # 2. 환자 폴더 수
    n_patients = len(patient_dirs)
    if n_patients != EXPECTED_PATIENTS:
        errors.append(f"[FAIL 2] 환자 폴더 수 불일치: 실제={n_patients}, 예상={EXPECTED_PATIENTS}")
    else:
        print(f"[OK 2] 환자 폴더 수: {n_patients}")

    # 3. summary CSV row 수
    n_csv_rows = len(summary_csv_df)
    if n_csv_rows != n_npz:
        errors.append(f"[FAIL 3] summary CSV row 수 불일치: CSV={n_csv_rows}, npz={n_npz}")
    else:
        print(f"[OK 3] summary CSV row 수: {n_csv_rows:,}")

    # 4. summary JSON total_crops
    json_total = summary_json.get("total_crops", -1)
    if json_total != n_npz:
        errors.append(f"[FAIL 4] summary JSON total_crops 불일치: JSON={json_total}, npz={n_npz}")
    else:
        print(f"[OK 4] summary JSON total_crops: {json_total:,}")

    # 5. manifest row 수
    n_manifest = len(manifest_df)
    if n_manifest != n_npz:
        errors.append(f"[FAIL 5] manifest row 수 불일치: manifest={n_manifest}, npz={n_npz}")
    else:
        print(f"[OK 5] manifest rows: {n_manifest:,}")

    # 6. positive / hard_negative 개수
    n_pos_csv = int((summary_csv_df["sampling_label"] == "positive").sum())
    n_hn_csv  = int((summary_csv_df["sampling_label"] == "hard_negative").sum())
    if n_pos_csv != EXPECTED_POS or n_hn_csv != EXPECTED_HN:
        errors.append(
            f"[FAIL 6] pos/hn 불일치: pos={n_pos_csv}(예상{EXPECTED_POS}), "
            f"hn={n_hn_csv}(예상{EXPECTED_HN})"
        )
    else:
        print(f"[OK 6] positive={n_pos_csv:,}, hard_negative={n_hn_csv:,}")

    # 7. 환자별 npz 수 일치
    per_patient_npz      = {d.name: len(list(d.glob("*.npz"))) for d in patient_dirs}
    per_patient_manifest = manifest_df.groupby("patient_id").size().to_dict()
    per_patient_summary  = (
        summary_csv_df.groupby("patient_id").size().to_dict()
        if "patient_id" in summary_csv_df.columns else {}
    )
    mismatch_patients = []
    for pid, cnt in per_patient_npz.items():
        m_cnt = per_patient_manifest.get(pid, 0)
        s_cnt = per_patient_summary.get(pid, 0)
        if cnt != m_cnt or cnt != s_cnt:
            mismatch_patients.append(f"{pid}: npz={cnt}, manifest={m_cnt}, summary_csv={s_cnt}")
    if mismatch_patients:
        errors.append(
            f"[FAIL 7] 환자별 npz 수 불일치 {len(mismatch_patients)}명:\n"
            + "\n".join(f"  {m}" for m in mismatch_patients[:5])
        )
    else:
        print(f"[OK 7] 환자별 npz 수 일치")

    # 8. stage2_holdout 0명 (이미 확인)
    print(f"[OK 8] stage2_holdout 0명 확인")

    # 9. stage1_dev만 포함
    manifest_stages = set(manifest_df["stage_split"].unique())
    if manifest_stages != {"stage1_dev"}:
        errors.append(f"[FAIL 9] stage1_dev 이외 stage 존재: {manifest_stages}")
    else:
        print(f"[OK 9] stage1_dev만 포함")

    # ── 전수 npz 검증 ──────────────────────────
    print(f"\n전수 npz 검증 시작 ({n_npz:,}개)...")

    n_key_missing    = 0
    n_forbidden_key  = 0
    n_shape_err      = 0
    n_dtype_err      = 0
    n_range_err      = 0
    n_nan_inf        = 0
    n_label_err      = 0
    n_label_mismatch = 0
    n_zsource_err    = 0
    n_localz_err     = 0
    n_cropcoords_err = 0
    n_origbbox_err   = 0
    n_patid_mismatch = 0

    img_min_vals = []
    img_max_vals = []

    for i, npz_path in enumerate(all_npz_files):
        patient_id_folder = npz_path.parent.name

        # stage2_holdout 이중 방어
        if patient_id_folder in holdout_pids:
            print(f"[ABORT] stage2_holdout 감지: {patient_id_folder}", file=sys.stderr)
            sys.exit(1)

        try:
            data = np.load(str(npz_path), allow_pickle=True)
        except Exception as e:
            errors.append(f"[FAIL] npz 로드 오류: {npz_path.name}: {e}")
            continue

        keys = set(data.files)

        # 10. 필수 key 존재
        missing_keys = [k for k in REQUIRED_KEYS if k not in keys]
        if missing_keys:
            n_key_missing += 1
            if n_key_missing <= 3:
                errors.append(f"[FAIL 10] 필수 key 누락: {npz_path.name}, 누락={missing_keys}")

        # 11. 금지 key 없음
        found_forbidden = [k for k in FORBIDDEN_KEYS if k in keys]
        if found_forbidden:
            n_forbidden_key += 1
            if n_forbidden_key <= 3:
                errors.append(f"[FAIL 11] 금지 key 존재: {npz_path.name}, 발견={found_forbidden}")

        image = data["image"]

        # 12. image shape
        if image.shape != (6, 96, 96):
            n_shape_err += 1
            if n_shape_err <= 3:
                errors.append(f"[FAIL 12] image shape 오류: {npz_path.name}, shape={image.shape}")

        # 13. dtype
        if image.dtype != np.float32:
            n_dtype_err += 1
            if n_dtype_err <= 3:
                errors.append(f"[FAIL 13] dtype 오류: {npz_path.name}, dtype={image.dtype}")

        # 14. image range [0, 1]
        img_min = float(image.min())
        img_max = float(image.max())
        img_min_vals.append(img_min)
        img_max_vals.append(img_max)
        if img_min < -1e-6 or img_max > 1.0 + 1e-6:
            n_range_err += 1
            if n_range_err <= 3:
                errors.append(
                    f"[FAIL 14] image range 이탈: {npz_path.name}, "
                    f"min={img_min:.6f}, max={img_max:.6f}"
                )

        # 15. NaN/Inf
        if not np.isfinite(image).all():
            n_nan_inf += 1
            if n_nan_inf <= 3:
                errors.append(f"[FAIL 15] NaN/Inf 존재: {npz_path.name}")

        # 16. label 값 0/1
        label_val = int(data["label"])
        if label_val not in (0, 1):
            n_label_err += 1
            if n_label_err <= 3:
                errors.append(f"[FAIL 16] label 값 오류: {npz_path.name}, label={label_val}")

        # 17. sampling_label 값 + 18. label/sampling_label 일치
        sampling_label_val = str(data["sampling_label"])
        if sampling_label_val not in ("positive", "hard_negative"):
            n_label_err += 1
            if n_label_err <= 3:
                errors.append(
                    f"[FAIL 17] sampling_label 값 오류: {npz_path.name}, val={sampling_label_val}"
                )
        else:
            expected_label = 1 if sampling_label_val == "positive" else 0
            if label_val != expected_label:
                n_label_mismatch += 1
                if n_label_mismatch <= 3:
                    errors.append(
                        f"[FAIL 18] label/sampling_label 불일치: {npz_path.name}, "
                        f"label={label_val}, sampling_label={sampling_label_val}"
                    )

        # 19. z_source == "local_z"
        z_source_val = str(data["z_source"])
        if z_source_val != "local_z":
            n_zsource_err += 1
            if n_zsource_err <= 3:
                errors.append(f"[FAIL 19] z_source 오류: {npz_path.name}, z_source={z_source_val}")

        # 20. local_z NaN/음수
        local_z_val = float(data["local_z"])
        if np.isnan(local_z_val) or local_z_val < 0:
            n_localz_err += 1
            if n_localz_err <= 3:
                errors.append(f"[FAIL 20] local_z 오류: {npz_path.name}, local_z={local_z_val}")

        # 21/22. crop_coords 길이 4 + 96×96 크기
        crop_coords = data["crop_coords"]
        if len(crop_coords) != 4:
            n_cropcoords_err += 1
            if n_cropcoords_err <= 3:
                errors.append(
                    f"[FAIL 21] crop_coords 길이 오류: {npz_path.name}, len={len(crop_coords)}"
                )
        else:
            y0, x0, y1, x1 = crop_coords
            h = int(y1 - y0)
            w = int(x1 - x0)
            if h != 96 or w != 96:
                n_cropcoords_err += 1
                if n_cropcoords_err <= 3:
                    errors.append(
                        f"[FAIL 22] crop_coords 크기 오류: {npz_path.name}, h={h}, w={w}"
                    )

        # 23. orig_bbox 길이 4
        orig_bbox = data["orig_bbox"]
        if len(orig_bbox) != 4:
            n_origbbox_err += 1
            if n_origbbox_err <= 3:
                errors.append(
                    f"[FAIL 23] orig_bbox 길이 오류: {npz_path.name}, len={len(orig_bbox)}"
                )

        # 24. patient_id 폴더명 일치
        npz_patient_id = str(data["patient_id"])
        if npz_patient_id != patient_id_folder:
            n_patid_mismatch += 1
            if n_patid_mismatch <= 3:
                errors.append(
                    f"[FAIL 24] patient_id 불일치: folder={patient_id_folder}, "
                    f"npz={npz_patient_id}"
                )

        if (i + 1) % 10000 == 0:
            print(f"  ... {i + 1:,} / {n_npz:,} 완료")

    # 25. image_min/max_global 검증
    actual_img_min    = float(np.min(img_min_vals)) if img_min_vals else None
    actual_img_max    = float(np.max(img_max_vals)) if img_max_vals else None
    expected_img_min  = summary_json.get("image_min_global")
    expected_img_max  = summary_json.get("image_max_global")

    if actual_img_min != expected_img_min or actual_img_max != expected_img_max:
        errors.append(
            f"[FAIL 25] image_min/max_global 불일치: "
            f"실제=({actual_img_min}, {actual_img_max}), "
            f"summary=({expected_img_min}, {expected_img_max})"
        )
    else:
        print(f"[OK 25] image_min_global={actual_img_min}, image_max_global={actual_img_max}")

    # 26~28: 비접촉 / 미실행 확인 (read-only 스크립트이므로 항상 OK)
    print(f"[OK 26] crops_s6a_full 비접촉 확인 (read-only 스크립트)")
    print(f"[OK 27] crops_s6a_6ch_smoke 비접촉 확인 (read-only 스크립트)")
    print(f"[OK 28] 학습/forward/scoring 미실행 확인 (read-only 스크립트)")

    # 집계 출력
    print()
    print("=" * 60)
    print("  전수 검증 집계")
    print("=" * 60)
    print(f"  key 누락:          {n_key_missing}")
    print(f"  금지 key 존재:     {n_forbidden_key}")
    print(f"  shape 오류:        {n_shape_err}")
    print(f"  dtype 오류:        {n_dtype_err}")
    print(f"  range 이탈:        {n_range_err}")
    print(f"  NaN/Inf:           {n_nan_inf}")
    print(f"  label 오류:        {n_label_err}")
    print(f"  label 불일치:      {n_label_mismatch}")
    print(f"  z_source 오류:     {n_zsource_err}")
    print(f"  local_z 오류:      {n_localz_err}")
    print(f"  crop_coords 오류:  {n_cropcoords_err}")
    print(f"  orig_bbox 오류:    {n_origbbox_err}")
    print(f"  patient_id 불일치: {n_patid_mismatch}")
    print("=" * 60)

    return {
        "n_npz":              n_npz,
        "n_patients":         n_patients,
        "n_pos":              n_pos_csv,
        "n_hn":               n_hn_csv,
        "n_key_missing":      n_key_missing,
        "n_forbidden_key":    n_forbidden_key,
        "n_shape_err":        n_shape_err,
        "n_dtype_err":        n_dtype_err,
        "n_range_err":        n_range_err,
        "n_nan_inf":          n_nan_inf,
        "n_label_err":        n_label_err,
        "n_label_mismatch":   n_label_mismatch,
        "n_zsource_err":      n_zsource_err,
        "n_localz_err":       n_localz_err,
        "n_cropcoords_err":   n_cropcoords_err,
        "n_origbbox_err":     n_origbbox_err,
        "n_patid_mismatch":   n_patid_mismatch,
        "image_min_global":   actual_img_min,
        "image_max_global":   actual_img_max,
        "errors":             errors,
    }


# ─────────────────────────────────────────────
# validation summary 저장
# ─────────────────────────────────────────────
def save_validation_summary(result: dict, run_start: datetime, run_end: datetime) -> None:
    elapsed  = (run_end - run_start).total_seconds()
    n_errors = len(result["errors"])
    passed   = n_errors == 0

    OUT_RPT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "script":              "validate_s6a_6ch_crop_full.py",
        "mode":                "validation",
        "n_npz":               result["n_npz"],
        "n_patients":          result["n_patients"],
        "n_pos":               result["n_pos"],
        "n_hn":                result["n_hn"],
        "n_key_missing":       result["n_key_missing"],
        "n_forbidden_key":     result["n_forbidden_key"],
        "n_shape_err":         result["n_shape_err"],
        "n_dtype_err":         result["n_dtype_err"],
        "n_range_err":         result["n_range_err"],
        "n_nan_inf":           result["n_nan_inf"],
        "n_label_err":         result["n_label_err"],
        "n_label_mismatch":    result["n_label_mismatch"],
        "n_zsource_err":       result["n_zsource_err"],
        "n_localz_err":        result["n_localz_err"],
        "n_cropcoords_err":    result["n_cropcoords_err"],
        "n_origbbox_err":      result["n_origbbox_err"],
        "n_patid_mismatch":    result["n_patid_mismatch"],
        "image_min_global":    result["image_min_global"],
        "image_max_global":    result["image_max_global"],
        "n_errors":            n_errors,
        "passed":              passed,
        "errors":              result["errors"][:20],
        "run_start":           run_start.isoformat(),
        "run_end":             run_end.isoformat(),
        "elapsed_seconds":     round(elapsed, 2),
    }

    with open(VAL_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {VAL_SUMMARY_JSON}")

    error_df = pd.DataFrame({"error": result["errors"]})
    error_df.to_csv(VAL_SUMMARY_CSV, index=False, encoding="utf-8")
    print(f"[저장] {VAL_SUMMARY_CSV}")

    status_str = "통과" if passed else f"미통과 (오류 {n_errors}건)"
    md_lines = [
        "# S6-A 6ch Crop Full Validation Summary",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 판정 | {status_str} |",
        f"| n_npz | {result['n_npz']:,} |",
        f"| n_patients | {result['n_patients']} |",
        f"| n_pos | {result['n_pos']:,} |",
        f"| n_hn | {result['n_hn']:,} |",
        f"| n_errors | {n_errors} |",
        f"| image_min_global | {result['image_min_global']} |",
        f"| image_max_global | {result['image_max_global']} |",
        f"| n_nan_inf | {result['n_nan_inf']} |",
        f"| n_shape_err | {result['n_shape_err']} |",
        f"| n_dtype_err | {result['n_dtype_err']} |",
        f"| n_range_err | {result['n_range_err']} |",
        f"| n_label_mismatch | {result['n_label_mismatch']} |",
        f"| elapsed_seconds | {round(elapsed, 2)} |",
    ]
    with open(VAL_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[저장] {VAL_SUMMARY_MD}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="S6-A 6ch full crop 전수 validation")
    parser.add_argument("--dry-run", action="store_true", help="validation 수행, summary 저장 없음")
    args = parser.parse_args()

    dry_run = args.dry_run

    print("=" * 70)
    print("  validate_s6a_6ch_crop_full.py")
    print(f"  모드: {'dry-run' if dry_run else 'validation'}")
    print(f"  시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    guard_check(dry_run=dry_run)

    run_start = datetime.now()
    result    = run_validation()
    run_end   = datetime.now()

    elapsed  = (run_end - run_start).total_seconds()
    n_errors = len(result["errors"])

    print()
    print("=" * 70)
    if n_errors == 0:
        print(f"  [VALIDATION] 전체 통과 — 오류 0건, elapsed={elapsed:.1f}s")
    else:
        print(f"  [VALIDATION] 미통과 — 오류 {n_errors}건, elapsed={elapsed:.1f}s")
        for e in result["errors"][:10]:
            print(f"    {e}")
    print("=" * 70)

    if dry_run:
        print("\n[DRY-RUN] validation summary 저장 안 함.")
        print("저장 실행 명령:")
        print("  source ~/ai_env/bin/activate && \\")
        print("  python scripts/validate_s6a_6ch_crop_full.py")
    else:
        save_validation_summary(result, run_start, run_end)


if __name__ == "__main__":
    main()
