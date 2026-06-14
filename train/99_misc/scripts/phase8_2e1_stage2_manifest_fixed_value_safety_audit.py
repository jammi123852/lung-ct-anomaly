"""
Phase 8.2E.1 Stage2 Manifest Fixed-Value Safety Audit

목적: Phase 8.2E candidate coordinate manifest의 고정 메타값들이
      crop generation 입력으로 안전한지 추가 검증한다.

금지: crop 생성, manifest 수정, npy/npz 로드, model forward,
      scoring, metric 계산, threshold, training 금지
"""

import sys
import os
import json
import pathlib
import datetime

def main():
    if "--run" not in sys.argv:
        print("[DRY-RUN] --run 인자 없음. 실행하지 않습니다.")
        print("실행하려면: python scripts/phase8_2e1_stage2_manifest_fixed_value_safety_audit.py --run")
        sys.exit(0)

    import pandas as pd

    # ── 경로 설정 ──────────────────────────────────────────────────────────────
    PROJECT_ROOT = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")

    MANIFEST_PATH = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_candidate_coordinate_manifest_v1.csv"
    PREV_VALIDATION_DIR = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_2e_stage2_candidate_coordinate_manifest_validation_v1"
    PREV_SUMMARY_JSON = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation_v1/phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation_summary_v1.json"

    OUT_DIR = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_2e1_stage2_manifest_fixed_value_safety_audit_v1"
    OUT_CSV = OUT_DIR / "phase8_2e1_stage2_manifest_fixed_value_safety_audit_v1.csv"
    OUT_JSON = OUT_DIR / "phase8_2e1_stage2_manifest_fixed_value_safety_audit_v1.json"
    OUT_MD = OUT_DIR / "phase8_2e1_stage2_manifest_fixed_value_safety_audit_report_v1.md"

    # ── output guard ───────────────────────────────────────────────────────────
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"[ABORT] 출력 폴더가 이미 존재합니다: {OUT_DIR}")
        sys.exit(1)

    for fp in [OUT_CSV, OUT_JSON, OUT_MD]:
        if fp.exists():
            print(f"[ABORT] 출력 파일이 이미 존재합니다: {fp}")
            sys.exit(1)

    # ── 입력 파일 존재 확인 ────────────────────────────────────────────────────
    for fp in [MANIFEST_PATH, PREV_SUMMARY_JSON]:
        if not fp.exists():
            print(f"[ABORT] 입력 파일이 없습니다: {fp}")
            sys.exit(1)

    if not PREV_VALIDATION_DIR.exists():
        print(f"[ABORT] 이전 validation 폴더가 없습니다: {PREV_VALIDATION_DIR}")
        sys.exit(1)

    print(f"[INFO] manifest 로드 중: {MANIFEST_PATH}")
    df = pd.read_csv(MANIFEST_PATH)
    created_at = datetime.datetime.now().isoformat()

    blockers = []

    # ══════════════════════════════════════════════════════════════════════════
    # Section A: manifest count recheck
    # ══════════════════════════════════════════════════════════════════════════
    EXPECTED_TOTAL = 143735
    EXPECTED_PATIENTS = 154
    EXPECTED_POSITIVE = 51335
    EXPECTED_HN = 92400

    actual_total = len(df)
    actual_patients = df["patient_id"].nunique()
    actual_positive = int((df["label"] == 1).sum())
    actual_hn = int((df["label"] == 0).sum())

    sec_a_rows = []

    def _a_row(check_item, expected, observed):
        ok = str(expected) == str(observed)
        status = "PASS" if ok else "FAIL"
        if not ok:
            blockers.append(f"BLOCKED_ROW_COUNT_MISMATCH: {check_item} expected={expected} observed={observed}")
        return {
            "section": "A_manifest_count_recheck",
            "check_item": check_item,
            "expected": str(expected),
            "observed": str(observed),
            "status": status,
            "note": "" if ok else f"mismatch: {check_item}",
            # 나머지 섹션 전용 컬럼
            "column_name": "", "expected_value": "", "unique_values": "",
            "mismatch_count": "", "null_count": "", "nan_count": "",
            "item": "", "blocker": "", "next_required_action": "",
        }

    sec_a_rows.append(_a_row("total_rows", EXPECTED_TOTAL, actual_total))
    sec_a_rows.append(_a_row("patient_count", EXPECTED_PATIENTS, actual_patients))
    sec_a_rows.append(_a_row("positive_count", EXPECTED_POSITIVE, actual_positive))
    sec_a_rows.append(_a_row("hard_negative_count", EXPECTED_HN, actual_hn))

    manifest_count_recheck = {
        "total_rows": {"expected": EXPECTED_TOTAL, "observed": actual_total, "ok": actual_total == EXPECTED_TOTAL},
        "patient_count": {"expected": EXPECTED_PATIENTS, "observed": actual_patients, "ok": actual_patients == EXPECTED_PATIENTS},
        "positive_count": {"expected": EXPECTED_POSITIVE, "observed": actual_positive, "ok": actual_positive == EXPECTED_POSITIVE},
        "hard_negative_count": {"expected": EXPECTED_HN, "observed": actual_hn, "ok": actual_hn == EXPECTED_HN},
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Section B: approval flag audit
    # ══════════════════════════════════════════════════════════════════════════
    sec_b_rows = []

    def _b_row(check_item, expected, observed, ok, note=""):
        status = "PASS" if ok else "FAIL"
        if not ok:
            blockers.append(f"BLOCKED_APPROVAL_FLAG_INVALID: {check_item}")
        return {
            "section": "B_approval_flag_audit",
            "check_item": check_item,
            "expected": str(expected),
            "observed": str(observed),
            "status": status,
            "note": note,
            "column_name": "", "expected_value": "", "unique_values": "",
            "mismatch_count": "", "null_count": "", "nan_count": "",
            "item": "", "blocker": "", "next_required_action": "",
        }

    col_exists = "approval_required_before_crop_generation" in df.columns
    sec_b_rows.append(_b_row(
        "column_exists",
        "True", str(col_exists), col_exists,
        "" if col_exists else "컬럼 없음",
    ))

    if col_exists:
        ap_col = df["approval_required_before_crop_generation"]
        ap_norm = ap_col.astype(str).str.strip().str.lower()
        true_like = ap_norm.isin(["true", "1"])
        null_like = ap_col.isna() | (ap_norm == "") | (ap_norm == "nan")
        invalid_count = int((~true_like | null_like).sum())
        true_count = int(true_like.sum())
        b_ok = (true_count == len(df)) and (invalid_count == 0)

        sec_b_rows.append(_b_row(
            "all_rows_true",
            f"true_like_count={len(df)}, invalid=0",
            f"true_like_count={true_count}, invalid={invalid_count}",
            b_ok,
            "" if b_ok else f"비정상 {invalid_count}건",
        ))
    else:
        true_count = 0
        invalid_count = len(df)
        b_ok = False

    approval_flag_audit = {
        "column_exists": col_exists,
        "true_like_count": true_count,
        "invalid_count": invalid_count,
        "ok": col_exists and b_ok,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Section C: fixed metadata audit
    # ══════════════════════════════════════════════════════════════════════════
    FIXED_META = [
        ("stage_split", "stage2_holdout"),
        ("model_type", "v2v2"),
        ("asset_scope", "dedicated_stage2_holdout_candidate_coordinate_manifest"),
        ("coordinate_source", "ratio_adjusted_score_full_diagnostic_csv_existing_stage2_v2v2_rows"),
        ("coordinate_rule", "existing_diag_csv_coordinates_reused_without_change"),
        ("sampling_rule", "existing_S6A_GS2_positive_all_hn_ratio2_reused_without_change"),
        ("manifest_status", "created_after_phase8_2e_run"),
    ]

    sec_c_rows = []
    fixed_metadata_audit = []

    for col_name, expected_val in FIXED_META:
        if col_name not in df.columns:
            unique_vals = []
            mismatch = len(df)
            ok = False
            note = "컬럼 없음"
        else:
            unique_vals = [str(v) for v in df[col_name].unique().tolist()]
            mismatch = int((df[col_name] != expected_val).sum())
            ok = mismatch == 0
            note = "" if ok else f"mismatch {mismatch}건"

        status = "PASS" if ok else "FAIL"
        if not ok:
            blockers.append(f"BLOCKED_FIXED_METADATA_MISMATCH: {col_name}")

        sec_c_rows.append({
            "section": "C_fixed_metadata_audit",
            "check_item": "",
            "expected": "",
            "observed": "",
            "status": status,
            "note": note,
            "column_name": col_name,
            "expected_value": expected_val,
            "unique_values": "|".join(unique_vals),
            "mismatch_count": str(mismatch),
            "null_count": "", "nan_count": "",
            "item": "", "blocker": "", "next_required_action": "",
        })

        fixed_metadata_audit.append({
            "column_name": col_name,
            "expected_value": expected_val,
            "unique_values": unique_vals,
            "mismatch_count": mismatch,
            "ok": ok,
            "note": note,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # Section D: source_diag_csv audit
    # ══════════════════════════════════════════════════════════════════════════
    sec_d_rows = []

    def _d_row(check_item, expected, observed, ok, note=""):
        status = "PASS" if ok else "FAIL"
        if not ok:
            blockers.append(f"BLOCKED_SOURCE_DIAG_CSV_INVALID: {check_item}")
        return {
            "section": "D_source_diag_csv_audit",
            "check_item": check_item,
            "expected": str(expected),
            "observed": str(observed),
            "status": status,
            "note": note,
            "column_name": "", "expected_value": "", "unique_values": "",
            "mismatch_count": "", "null_count": "", "nan_count": "",
            "item": "", "blocker": "", "next_required_action": "",
        }

    d_col_exists = "source_diag_csv" in df.columns
    sec_d_rows.append(_d_row("column_exists", "True", str(d_col_exists), d_col_exists))

    if d_col_exists:
        src_col = df["source_diag_csv"]
        null_empty = int(src_col.isna().sum()) + int((src_col == "").sum())
        sec_d_rows.append(_d_row("null_empty_count", 0, null_empty, null_empty == 0,
                                 "" if null_empty == 0 else f"null/empty {null_empty}건"))

        unique_paths = src_col.dropna().unique().tolist()
        unique_count = len(unique_paths)
        sec_d_rows.append(_d_row("unique_path_count", 1, unique_count, unique_count == 1,
                                 "" if unique_count == 1 else f"경로 {unique_count}가지"))

        sample_path = unique_paths[0] if unique_paths else ""
        has_keyword = "ratio_adjusted_score_full_diagnostic.csv" in sample_path
        sec_d_rows.append(_d_row(
            "path_contains_ratio_adjusted_diag_csv",
            "ratio_adjusted_score_full_diagnostic.csv in path",
            sample_path,
            has_keyword,
            "" if has_keyword else "경로에 키워드 없음",
        ))

        # 파일 존재 확인 (read-only, 내용 로드 금지)
        file_exists = os.path.exists(sample_path) if sample_path else False
        sec_d_rows.append(_d_row(
            "source_diag_csv_file_exists",
            "True",
            str(file_exists),
            file_exists,
            "" if file_exists else "파일 없음",
        ))
    else:
        unique_paths = []
        sample_path = ""
        has_keyword = False
        file_exists = False
        null_empty = len(df)
        unique_count = 0

    source_diag_csv_audit = {
        "column_exists": d_col_exists,
        "null_empty_count": null_empty if d_col_exists else len(df),
        "unique_path_count": unique_count if d_col_exists else 0,
        "sample_path": sample_path,
        "path_contains_keyword": has_keyword,
        "file_exists": file_exists,
        "ok": d_col_exists and null_empty == 0 and unique_count == 1 and has_keyword and file_exists,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Section E: score/provenance null audit
    # ══════════════════════════════════════════════════════════════════════════
    SCORE_COLS = [
        "score_original",
        "score_valid950_weighted",
        "lesion_patch_ratio",
        "composite_rank_v2",
    ]

    sec_e_rows = []
    score_provenance_null_audit = []

    for col_name in SCORE_COLS:
        if col_name not in df.columns:
            null_cnt = len(df)
            nan_cnt = len(df)
            ok = False
            note = "컬럼 없음"
        else:
            null_cnt = int(df[col_name].isna().sum())
            # pandas isna()가 NaN, None, NaT 포함하므로 동일
            nan_cnt = null_cnt
            ok = null_cnt == 0
            note = "" if ok else f"null {null_cnt}건"

        status = "PASS" if ok else "FAIL"
        if not ok:
            blockers.append(f"BLOCKED_SCORE_PROVENANCE_NULL: {col_name} null={null_cnt}")

        sec_e_rows.append({
            "section": "E_score_provenance_null_audit",
            "check_item": "",
            "expected": "",
            "observed": "",
            "status": status,
            "note": note,
            "column_name": col_name,
            "expected_value": "",
            "unique_values": "",
            "mismatch_count": "",
            "null_count": str(null_cnt),
            "nan_count": str(nan_cnt),
            "item": "", "blocker": "", "next_required_action": "",
        })

        score_provenance_null_audit.append({
            "column_name": col_name,
            "null_count": null_cnt,
            "nan_count": nan_cnt,
            "ok": ok,
            "note": note,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # Section F: readiness decision
    # ══════════════════════════════════════════════════════════════════════════
    # 중복 blocker 제거 (키 기준)
    unique_blocker_keys = sorted(set(b.split(":")[0].strip() for b in blockers))

    if not unique_blocker_keys:
        readiness = "READY_FOR_PHASE8_2F_DEDICATED_6CH_CROP_GENERATION_SCRIPT"
        f_status = "READY"
        f_blocker = ""
        f_next = "Phase 8.2F dedicated 6ch crop generation script 작성/실행 전 검토 진행"
    else:
        readiness = " | ".join(unique_blocker_keys)
        f_status = "BLOCKED"
        f_blocker = " | ".join(unique_blocker_keys)
        f_next = "blocker 해소 후 Phase 8.2E.1 재실행"

    sec_f_rows = [{
        "section": "F_readiness_decision",
        "check_item": "",
        "expected": "",
        "observed": "",
        "status": f_status,
        "note": "",
        "column_name": "",
        "expected_value": "",
        "unique_values": "",
        "mismatch_count": "",
        "null_count": "",
        "nan_count": "",
        "item": "readiness_for_phase8_2f",
        "blocker": f_blocker,
        "next_required_action": f_next,
    }]

    # ══════════════════════════════════════════════════════════════════════════
    # CSV 저장
    # ══════════════════════════════════════════════════════════════════════════
    import csv

    all_rows = sec_a_rows + sec_b_rows + sec_c_rows + sec_d_rows + sec_e_rows + sec_f_rows
    fieldnames = [
        "section", "check_item", "expected", "observed", "status", "note",
        "column_name", "expected_value", "unique_values", "mismatch_count",
        "null_count", "nan_count", "item", "blocker", "next_required_action",
    ]

    # 저장 직전 output file 재검증
    if OUT_CSV.exists():
        print(f"[ABORT] CSV 파일이 이미 존재합니다: {OUT_CSV}")
        sys.exit(1)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[OK] CSV 저장: {OUT_CSV}")

    # ══════════════════════════════════════════════════════════════════════════
    # JSON 저장
    # ══════════════════════════════════════════════════════════════════════════
    result_json = {
        "created_at": created_at,
        "input_paths": {
            "manifest": str(MANIFEST_PATH),
            "prev_validation_dir": str(PREV_VALIDATION_DIR),
            "prev_summary_json": str(PREV_SUMMARY_JSON),
        },
        "manifest_count_recheck": manifest_count_recheck,
        "approval_flag_audit": approval_flag_audit,
        "fixed_metadata_audit": fixed_metadata_audit,
        "source_diag_csv_audit": source_diag_csv_audit,
        "score_provenance_null_audit": score_provenance_null_audit,
        "readiness_for_phase8_2f": readiness,
        "blockers": blockers,
        "notes": {
            "audit_only": True,
            "no_crop_generation": True,
            "no_manifest_modification": True,
            "no_npy_loading": True,
            "no_npz_loading": True,
            "no_model_forward": True,
            "no_scoring": True,
            "no_metric_calculation": True,
            "no_threshold": True,
            "no_training": True,
        },
    }

    if OUT_JSON.exists():
        print(f"[ABORT] JSON 파일이 이미 존재합니다: {OUT_JSON}")
        sys.exit(1)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    print(f"[OK] JSON 저장: {OUT_JSON}")

    # ══════════════════════════════════════════════════════════════════════════
    # MD report 저장
    # ══════════════════════════════════════════════════════════════════════════
    def _status_icon(s):
        return "✅" if s == "PASS" else "❌"

    sec_a_table = "\n".join(
        f"| {r['check_item']} | {r['expected']} | {r['observed']} | {_status_icon(r['status'])} {r['status']} |"
        for r in sec_a_rows
    )
    sec_b_table = "\n".join(
        f"| {r['check_item']} | {r['expected']} | {r['observed']} | {_status_icon(r['status'])} {r['status']} | {r['note']} |"
        for r in sec_b_rows
    )
    sec_c_table = "\n".join(
        f"| {r['column_name']} | {r['expected_value']} | {r['unique_values']} | {r['mismatch_count']} | {_status_icon(r['status'])} {r['status']} |"
        for r in sec_c_rows
    )
    sec_d_table = "\n".join(
        f"| {r['check_item']} | {r['expected']} | {r['observed']} | {_status_icon(r['status'])} {r['status']} | {r['note']} |"
        for r in sec_d_rows
    )
    sec_e_table = "\n".join(
        f"| {r['column_name']} | {r['null_count']} | {r['nan_count']} | {_status_icon(r['status'])} {r['status']} |"
        for r in sec_e_rows
    )

    if f_status == "READY":
        next_step_text = (
            "Phase 8.2F dedicated 6ch crop generation script 작성/실행 전 검토를 진행한다.\n"
            "- crop generation 전 스크립트 설계 및 사용자 승인 필수"
        )
    else:
        next_step_text = (
            f"아래 blocker를 해소한 후 Phase 8.2E.1을 재실행한다.\n\n"
            + "\n".join(f"- {b}" for b in blockers)
        )

    md_content = f"""# Phase 8.2E.1 Stage2 Manifest Fixed-Value Safety Audit

생성일: {created_at}

---

## 1. Phase 8.2E.1 목적

Phase 8.2E candidate coordinate manifest의 고정 메타값(approval flag, fixed metadata, source_diag_csv, score/provenance)이
crop generation 입력으로 안전한지 추가 검증한다.

이번 단계는 **manifest fixed-value audit only**다.

---

## 2. 왜 보강 검증이 필요한지

Phase 8.2E validation 결과는 READY로 나왔지만,
`approval_required_before_crop_generation=True` 전 행 검증이 readiness 조건에 명확히 포함되지 않았다.
crop generation 전에 이 값이 전 행 True인지 반드시 확인해야 하며,
asset_scope / coordinate_source / coordinate_rule / sampling_rule / manifest_status 등
고정값도 전 행 일관성을 확인해야 한다.

---

## 3. Manifest Count 재확인

| 항목 | 기대값 | 실측값 | 판정 |
|------|--------|--------|------|
{sec_a_table}

---

## 4. Approval Flag Audit

| 항목 | 기대값 | 실측값 | 판정 | 비고 |
|------|--------|--------|------|------|
{sec_b_table}

---

## 5. Fixed Metadata Audit

| 컬럼 | 기대값 | unique_values | mismatch_count | 판정 |
|------|--------|---------------|----------------|------|
{sec_c_table}

---

## 6. source_diag_csv Audit

| 항목 | 기대값 | 실측값 | 판정 | 비고 |
|------|--------|--------|------|------|
{sec_d_table}

---

## 7. Score/Provenance Null Audit

| 컬럼 | null_count | nan_count | 판정 |
|------|-----------|-----------|------|
{sec_e_table}

---

## 8. Readiness 판정

**{readiness}**

Blockers: {", ".join(blockers) if blockers else "없음"}

---

## 9. 다음 단계

{next_step_text}

---

## 10. 금지 사항

- crop 생성 금지
- manifest 수정 금지
- npy 로드 금지
- npz 로드 금지
- CT/ROI/mask 내용 확인 금지
- model forward 금지
- scoring 금지
- metric 계산 금지
- threshold / p95 / p99 / hit-rate 계산 금지
- training 금지
- checkpoint 생성 금지
- 기존 Phase 6/7/8 output 수정 금지
- final manifest 수정 금지
- DIAG_CSV 수정 금지
- v1v2/stage1_dev row 사용 금지
- v2/v2v2 재스코어링 금지
- suppression/mask/ROI 수정 금지
- pip/conda install 금지
- 외부 다운로드 금지
"""

    if OUT_MD.exists():
        print(f"[ABORT] MD 파일이 이미 존재합니다: {OUT_MD}")
        sys.exit(1)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[OK] MD 저장: {OUT_MD}")

    # ── 최종 보고 ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"readiness_for_phase8_2f: {readiness}")
    if blockers:
        print("blockers:")
        for b in blockers:
            print(f"  - {b}")
    else:
        print("blockers: 없음")
    print("=" * 60)


if __name__ == "__main__":
    main()
