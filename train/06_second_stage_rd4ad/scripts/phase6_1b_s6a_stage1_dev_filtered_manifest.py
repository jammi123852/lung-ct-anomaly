#!/usr/bin/env python
"""
Phase 6.1b: S6-A stage1_dev-only filtered shadow manifest 생성
- 원본 dataset index / split 파일 수정 없음
- stage2_holdout row만 제외한 shadow manifest를 새 output root에 생성
- 이 파일은 training manifest가 아님 (approval_required_before_training=True)
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATASET_INDEX = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_6ch_full_dataset_index.csv"
STAGE_SPLIT   = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"
OUTPUT_ROOT   = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase6_1b_s6a_stage1_dev_filtered_manifest_v1"

FILTERED_CSV  = OUTPUT_ROOT / "phase6_1b_s6a_stage1_dev_filtered_manifest_v1.csv"
SUMMARY_JSON  = OUTPUT_ROOT / "phase6_1b_s6a_stage1_dev_filtered_manifest_summary_v1.json"
REPORT_MD     = OUTPUT_ROOT / "phase6_1b_s6a_stage1_dev_filtered_manifest_report_v1.md"
EXCLUDED_CSV  = OUTPUT_ROOT / "phase6_1b_s6a_stage2_holdout_excluded_rows_v1.csv"

EXPECTED_S2_PATIENTS   = {"LUNG1-295", "LUNG1-415"}
EXPECTED_CONTAM_ROWS   = 1222
EXPECTED_FILTERED_ROWS = 129437
EXPECTED_FILTERED_PATS = 152

SPLIT_SOURCE = "splits/lesion_stage_split_v1.csv"


def check_paths():
    ok = True
    for p, name in [(DATASET_INDEX, "dataset index"), (STAGE_SPLIT, "stage split")]:
        if not p.exists():
            print(f"[ERROR] 입력 파일 없음: {name} — {p}")
            ok = False
    return ok


def check_v2_paths(df):
    return int(df["npz_path"].astype(str).str.contains("v2", na=False).sum())


def sample_npz_exists(df, n=16):
    sample = df["npz_path"].dropna().sample(n=min(n, len(df)), random_state=42)
    exists  = [Path(p).exists() for p in sample]
    return sum(exists), len(exists)


def run():
    blockers = []

    # ── 1. 입력 로드 (read-only) ─────────────────────────────────────────
    print(f"[1] dataset index 로드: {DATASET_INDEX}")
    idx = pd.read_csv(DATASET_INDEX)
    orig_rows    = len(idx)
    orig_pats    = idx["patient_id"].nunique()
    print(f"    rows={orig_rows}  unique patients={orig_pats}")

    print(f"[2] stage split 로드: {STAGE_SPLIT}")
    sp = pd.read_csv(STAGE_SPLIT)

    # ── 2. join ──────────────────────────────────────────────────────────
    print("[3] patient_id join")
    merged = idx.merge(sp[["patient_id", "stage_split"]], on="patient_id", how="left")
    unknown = int(merged["stage_split"].isna().sum())
    if unknown:
        print(f"    [WARN] stage 미확인 rows: {unknown}")

    # ── 3. stage2_holdout 식별 ───────────────────────────────────────────
    s2_mask    = merged["stage_split"] == "stage2_holdout"
    s2_rows    = int(s2_mask.sum())
    s2_pats    = set(merged.loc[s2_mask, "patient_id"].unique().tolist())
    print(f"[4] stage2_holdout rows: {s2_rows}  patients: {sorted(s2_pats)}")

    # 검증: 예상 환자와 일치하는지
    if s2_pats != EXPECTED_S2_PATIENTS:
        msg = f"stage2_holdout patients 불일치 expected={sorted(EXPECTED_S2_PATIENTS)} actual={sorted(s2_pats)}"
        blockers.append(msg)
        print(f"    [BLOCKER] {msg}")
    else:
        print(f"    환자 일치: {sorted(s2_pats)}")

    if s2_rows != EXPECTED_CONTAM_ROWS:
        msg = f"contaminated rows 불일치 expected={EXPECTED_CONTAM_ROWS} actual={s2_rows}"
        blockers.append(msg)
        print(f"    [BLOCKER] {msg}")
    else:
        print(f"    contaminated rows 일치: {s2_rows}")

    # ── 4. filtered manifest 생성 ────────────────────────────────────────
    print("[5] filtered manifest 생성 (stage1_dev only)")
    filtered = merged[~s2_mask].copy()
    filtered["split_source"]                  = SPLIT_SOURCE
    filtered["filtered_manifest_status"]      = "stage1_dev_only_filtered_shadow_manifest"
    filtered["training_manifest_status"]      = "not_training_manifest"
    filtered["approval_required_before_training"] = True

    filtered_rows = len(filtered)
    filtered_pats = filtered["patient_id"].nunique()
    print(f"    filtered rows={filtered_rows}  unique patients={filtered_pats}")

    if filtered_rows != EXPECTED_FILTERED_ROWS:
        msg = f"filtered rows 불일치 expected={EXPECTED_FILTERED_ROWS} actual={filtered_rows}"
        blockers.append(msg)
        print(f"    [BLOCKER] {msg}")
    else:
        print(f"    filtered rows 일치: {filtered_rows}")

    if filtered_pats != EXPECTED_FILTERED_PATS:
        msg = f"filtered unique patients 불일치 expected={EXPECTED_FILTERED_PATS} actual={filtered_pats}"
        blockers.append(msg)
        print(f"    [BLOCKER] {msg}")
    else:
        print(f"    filtered patients 일치: {filtered_pats}")

    # ── 5. filtered manifest 추가 검증 ───────────────────────────────────
    print("[6] filtered manifest 검증")
    filt_s2_count = int((filtered["stage_split"] == "stage2_holdout").sum())
    v2_count      = check_v2_paths(filtered)
    print(f"    filtered stage2_holdout rows: {filt_s2_count}")
    print(f"    v2/v2v2 경로: {v2_count}건")
    if filt_s2_count > 0:
        blockers.append(f"filtered manifest 내 stage2_holdout {filt_s2_count}건")
    if v2_count > 0:
        blockers.append(f"filtered manifest 내 v2 경로 {v2_count}건")

    # npz_path 샘플 존재 확인 (16개)
    exist_ok, exist_total = sample_npz_exists(filtered)
    print(f"    npz_path 샘플 존재 확인: {exist_ok}/{exist_total}")
    if exist_ok < exist_total:
        blockers.append(f"npz_path 샘플 {exist_total - exist_ok}건 파일 없음")

    # ── 6. excluded_rows CSV ─────────────────────────────────────────────
    excluded = merged[s2_mask].copy()

    # ── 7. 저장 ─────────────────────────────────────────────────────────
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    filtered.to_csv(FILTERED_CSV, index=False)
    print(f"[7] filtered CSV 저장: {FILTERED_CSV}")

    excluded.to_csv(EXCLUDED_CSV, index=False)
    print(f"    excluded CSV 저장: {EXCLUDED_CSV}")

    # ── 8. summary JSON ──────────────────────────────────────────────────
    summary = {
        "original_dataset_index_path":      str(DATASET_INDEX),
        "split_source_path":                str(STAGE_SPLIT),
        "original_row_count":               orig_rows,
        "original_unique_patient_count":    orig_pats,
        "stage2_holdout_patients_detected": sorted(s2_pats),
        "stage2_holdout_row_count":         s2_rows,
        "filtered_row_count":               filtered_rows,
        "filtered_unique_patient_count":    filtered_pats,
        "expected_filtered_row_count":      EXPECTED_FILTERED_ROWS,
        "filtered_stage2_holdout_row_count": filt_s2_count,
        "filtered_manifest_path":           str(FILTERED_CSV),
        "excluded_rows_path":               str(EXCLUDED_CSV),
        "original_dataset_index_unmodified": True,
        "original_split_unmodified":         True,
        "crop_files_unmodified":             True,
        "training_manifest_status":          "not_training_manifest",
        "next_step_recommendation":          "rerun Phase 6.1 loader smoke using filtered shadow manifest",
        "blockers": (
            blockers if blockers else
            ["original S6-A index contains stage2_holdout rows, do not use original index for model smoke/training"]
        ),
        "npz_sample_exist_check": f"{exist_ok}/{exist_total}",
        "v2_path_detected":       v2_count,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"    summary JSON 저장: {SUMMARY_JSON}")

    # ── 9. report MD ─────────────────────────────────────────────────────
    overall = "전체 통과" if not blockers else "미통과"
    md = [
        "# Phase 6.1b S6-A stage1_dev-only Filtered Shadow Manifest Report",
        "",
        f"**최종 판정: {overall}**",
        "",
        "## 1. contamination 발견 요약",
        "",
        f"- 원본 dataset index(`s6a_6ch_full_dataset_index.csv`)에 stage2_holdout 환자 {len(s2_pats)}명이 포함되어 있었음.",
        f"- 오염 환자: {sorted(s2_pats)}",
        f"- 오염 rows: {s2_rows}건 / 전체 {orig_rows}건",
        "",
        "## 2. 원본 split을 수정하면 안 되는 이유",
        "",
        "- `lesion_stage_split_v1.csv`는 전체 308명의 stage1_dev/stage2_holdout 분할 기준 파일임.",
        "- 이 파일을 수정하면 split 기준이 소급 적용되어 다른 분석 결과의 재현성이 깨짐.",
        "- LUNG1-295, LUNG1-415가 실제로 stage1_dev인지 stage2_holdout인지는 별도 확인이 필요하며,",
        "  split 파일을 임의 수정해서 해결하는 것은 설계 원칙 위반임.",
        "",
        "## 3. 원본 dataset index를 직접 수정하지 않는 이유",
        "",
        "- 원본 `s6a_6ch_full_dataset_index.csv`는 crop 생성 단계의 결과물로 변경 추적이 필요함.",
        "- 직접 수정 시 contamination 사실이 은폐되고, 재현이 어려워짐.",
        "- shadow manifest 방식으로 원본을 보존하면서 안전한 입력을 제공함.",
        "",
        "## 4. filtered shadow manifest 생성 결과",
        "",
        f"- 원본 rows: {orig_rows:,}",
        f"- stage2_holdout 제외: {s2_rows}건",
        f"- filtered rows: {filtered_rows:,}  (expected {EXPECTED_FILTERED_ROWS:,}: {'일치' if filtered_rows == EXPECTED_FILTERED_ROWS else '불일치'})",
        f"- filtered unique patients: {filtered_pats}  (expected {EXPECTED_FILTERED_PATS}: {'일치' if filtered_pats == EXPECTED_FILTERED_PATS else '불일치'})",
        f"- filtered manifest 저장 경로: `{FILTERED_CSV}`",
        "",
        "## 5. excluded patients/rows",
        "",
        f"- 제외 환자: {sorted(s2_pats)}",
        f"- 제외 rows: {s2_rows}건",
        f"- excluded rows 저장 경로: `{EXCLUDED_CSV}`",
        "",
        "## 6. filtered manifest 검증 결과",
        "",
        f"- filtered stage2_holdout rows: {filt_s2_count}  ({'PASS' if filt_s2_count == 0 else 'FAIL'})",
        f"- v2/v2v2 경로: {v2_count}건  ({'PASS' if v2_count == 0 else 'FAIL'})",
        f"- npz_path 샘플 존재 확인: {exist_ok}/{exist_total}  ({'PASS' if exist_ok == exist_total else 'WARN'})",
        f"- training_manifest_status: not_training_manifest",
        f"- approval_required_before_training: True",
        "",
        "## 7. 다음 단계",
        "",
        "- **Phase 6.1c**: filtered shadow manifest를 입력으로 loader smoke 재실행",
        "- **Phase 6.2**: Phase 6.1c 통과 후에만 model forward smoke preflight 진행 가능",
        "",
        "## 8. 금지 사항",
        "",
        "- `lesion_stage_split_v1.csv` 수정 금지",
        "- `s6a_6ch_full_dataset_index.csv` 수정 금지",
        "- crop 파일 삭제/이동/수정 금지",
        "- stage2_holdout crop 삭제 금지",
        "- model forward / training / checkpoint / threshold 금지",
        "- v2/v2v2 접근 금지",
        "- 기존 결과 삭제/이동/덮어쓰기 금지",
        "- filtered shadow manifest를 training manifest로 사용 금지 (별도 승인 필요)",
    ]
    REPORT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"    report MD 저장: {REPORT_MD}")

    # ── 10. 최종 요약 ────────────────────────────────────────────────────
    print(f"\n=== Phase 6.1b 결과 ===")
    print(f"판정:            {overall}")
    print(f"blockers:        {blockers}")
    print(f"filtered rows:   {filtered_rows}  (expected {EXPECTED_FILTERED_ROWS})")
    print(f"filtered pats:   {filtered_pats}  (expected {EXPECTED_FILTERED_PATS})")
    print(f"s2 holdout rows: {filt_s2_count}")

    return not blockers


def main():
    if not check_paths():
        sys.exit(1)
    ok = run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
