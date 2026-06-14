"""
phase8_2e_stage2_candidate_coordinate_manifest_validation.py

Phase 8.2E에서 생성된 stage2_holdout candidate coordinate manifest가
crop 생성 입력으로 사용할 수 있는지 검증한다.

실행:
  --run 없이 : dry-run 보고만 출력 후 종료
  --run      : 실제 검증 결과 생성

절대 금지:
- crop 생성 금지
- manifest 수정 금지
- npy/npz 로드 금지
- CT/ROI/mask 내용 확인 금지
- model forward 금지
- scoring 금지
- metric 계산 금지
- threshold/p95/p99/hit-rate 계산 금지
- training 금지
- checkpoint 생성 금지
- 기존 Phase 6/7/8 output 수정 금지
- DIAG_CSV 수정 금지
- v1v2/stage1_dev row 사용 금지
- pip/conda install 금지
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

PHASE8_2E_ROOT = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation_v1"
MANIFEST_PATH  = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_candidate_coordinate_manifest_v1.csv"
SUMMARY_JSON   = PHASE8_2E_ROOT / "phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation_summary_v1.json"
ERRORS_CSV     = PHASE8_2E_ROOT / "phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation_errors_v1.csv"
RUNTIME_CSV    = PHASE8_2E_ROOT / "phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation_runtime_summary_v1.csv"
DONE_JSON      = PHASE8_2E_ROOT / "phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation_DONE.json"
REPORT_MD      = PHASE8_2E_ROOT / "phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation_report_v1.md"
SPLIT_CSV      = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"

OUT_ROOT     = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_2e_stage2_candidate_coordinate_manifest_validation_v1"
OUT_CSV      = OUT_ROOT / "phase8_2e_stage2_candidate_coordinate_manifest_validation_v1.csv"
OUT_JSON     = OUT_ROOT / "phase8_2e_stage2_candidate_coordinate_manifest_validation_v1.json"
OUT_MD       = OUT_ROOT / "phase8_2e_stage2_candidate_coordinate_manifest_validation_report_v1.md"

# ---------------------------------------------------------------------------
# 기대값 상수
# ---------------------------------------------------------------------------
EXPECTED_TOTAL_ROWS    = 143_735
EXPECTED_PATIENTS      = 154
EXPECTED_POSITIVE      = 51_335
EXPECTED_HARD_NEGATIVE = 92_400
EXPECTED_STAGE_SPLIT   = "stage2_holdout"
EXPECTED_MODEL_TYPE    = "v2v2"
EXPECTED_DONE_STATUS   = "DONE"
EXPECTED_RUNTIME_STATUS = "DONE"
PATIENT_HN_CAP         = 600
CONTAMINATION_PATIENTS = {"LUNG1-295", "LUNG1-415"}
EXPECTED_CONTAMINATION_STATUS = "coordinate_from_existing_stage2_diag_after_prior_crop_contamination"

REQUIRED_SCHEMA_COLS = [
    "row_id", "patient_id", "safe_id", "local_z",
    "y0", "x0", "y1", "x1",
    "label", "sampling_label", "stage_split", "model_type",
    "score_original", "score_valid950_weighted", "lesion_patch_ratio",
    "composite_rank_v2", "source_diag_csv", "asset_scope",
    "coordinate_source", "coordinate_rule", "sampling_rule",
    "contamination_check_status", "approval_required_before_crop_generation",
    "manifest_status", "issue", "note",
]


# ---------------------------------------------------------------------------
# output guard
# ---------------------------------------------------------------------------
def output_guard() -> None:
    for target in [OUT_ROOT, OUT_CSV, OUT_JSON, OUT_MD]:
        if target.exists():
            print(f"[중단] 출력 대상 이미 존재: {target}")
            print("  기존 파일/디렉토리를 삭제하거나 이름을 바꾼 후 재실행하세요.")
            sys.exit(1)
    print("[Guard] 출력 파일 미존재 확인 완료")


# ---------------------------------------------------------------------------
# dry-run 보고
# ---------------------------------------------------------------------------
def dry_run_report() -> None:
    print("\n=== Dry-run 보고 (실제 실행 아님) ===")
    print(f"입력 manifest   : {MANIFEST_PATH}")
    print(f"입력 summary    : {SUMMARY_JSON}")
    print(f"입력 errors CSV : {ERRORS_CSV}")
    print(f"입력 runtime    : {RUNTIME_CSV}")
    print(f"입력 DONE       : {DONE_JSON}")
    print(f"입력 report MD  : {REPORT_MD}")
    print(f"입력 split CSV  : {SPLIT_CSV}")
    print(f"\n출력 root : {OUT_ROOT}")
    print(f"출력 CSV  : {OUT_CSV}")
    print(f"출력 JSON : {OUT_JSON}")
    print(f"출력 MD   : {OUT_MD}")
    print(f"\n실행 명령:")
    print(f"  source ~/ai_env/bin/activate && python scripts/phase8_2e_stage2_candidate_coordinate_manifest_validation.py --run")
    print(f"\n[Dry-run 완료] 실제 실행은 --run 추가 후 진행하세요.")


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def ok(note: str = "") -> str:
    return "OK"

def fail(note: str = "") -> str:
    return "FAIL"

def warn(note: str = "") -> str:
    return "WARNING"

def _status(passed: bool, fatal: bool = True) -> str:
    if passed:
        return "OK"
    return "FAIL" if fatal else "WARNING"


# ---------------------------------------------------------------------------
# Section A: artifact existence
# ---------------------------------------------------------------------------
def check_artifact_existence() -> tuple[list[dict], dict]:
    print("\n[Section A] artifact existence 확인")
    artifacts = [
        ("final_manifest",  MANIFEST_PATH),
        ("summary_json",    SUMMARY_JSON),
        ("errors_csv",      ERRORS_CSV),
        ("runtime_summary", RUNTIME_CSV),
        ("done_marker",     DONE_JSON),
        ("report_md",       REPORT_MD),
    ]
    rows = []
    summary = {}
    for item, path in artifacts:
        exists = path.exists()
        status = "OK" if exists else "FAIL"
        note = "" if exists else "파일 없음"
        rows.append({"section": "A", "item": item, "path": str(path), "exists": exists, "status": status, "note": note})
        summary[item] = exists
        print(f"  [{status}] {item}: {exists}")
    return rows, summary


# ---------------------------------------------------------------------------
# Section B: completion consistency
# ---------------------------------------------------------------------------
def check_completion_consistency(manifest_rows: int) -> tuple[list[dict], dict]:
    print("\n[Section B] Phase 8.2E completion consistency 확인")
    rows = []
    summary = {}

    def add(item: str, expected, observed, passed: bool, note: str = "") -> None:
        status = _status(passed)
        rows.append({"section": "B", "item": item, "expected": str(expected), "observed": str(observed), "status": status, "note": note})
        summary[item] = {"expected": expected, "observed": observed, "passed": passed}
        print(f"  [{status}] {item}: expected={expected}, observed={observed}" + (f" ({note})" if note else ""))

    # DONE marker
    done_status = None
    if DONE_JSON.exists():
        try:
            done_data = json.loads(DONE_JSON.read_text(encoding="utf-8"))
            done_status = done_data.get("status")
            done_rows = done_data.get("n_output_rows")
        except Exception as e:
            done_status = f"parse_error: {e}"
            done_rows = None
    add("done_status", EXPECTED_DONE_STATUS, done_status, done_status == EXPECTED_DONE_STATUS)

    # summary JSON
    summary_data = {}
    if SUMMARY_JSON.exists():
        try:
            summary_data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        except Exception as e:
            summary_data = {}
            print(f"  [FAIL] summary JSON parse error: {e}")

    add("summary_n_total_rows",    EXPECTED_TOTAL_ROWS,    summary_data.get("n_total_rows"),    summary_data.get("n_total_rows") == EXPECTED_TOTAL_ROWS)
    add("summary_n_patients",      EXPECTED_PATIENTS,      summary_data.get("n_patients"),      summary_data.get("n_patients") == EXPECTED_PATIENTS)
    add("summary_n_positive",      EXPECTED_POSITIVE,      summary_data.get("n_positive"),      summary_data.get("n_positive") == EXPECTED_POSITIVE)
    add("summary_n_hard_negative", EXPECTED_HARD_NEGATIVE, summary_data.get("n_hard_negative"), summary_data.get("n_hard_negative") == EXPECTED_HARD_NEGATIVE)

    # error CSV row count
    error_row_count = None
    if ERRORS_CSV.exists():
        try:
            df_err = pd.read_csv(ERRORS_CSV, encoding="utf-8-sig")
            error_row_count = len(df_err)
        except Exception:
            error_row_count = None
    add("error_csv_row_count", 0, error_row_count, error_row_count == 0)

    # runtime status
    runtime_status = None
    if RUNTIME_CSV.exists():
        try:
            df_rt = pd.read_csv(RUNTIME_CSV, encoding="utf-8-sig")
            if len(df_rt) > 0 and "status" in df_rt.columns:
                runtime_status = df_rt["status"].iloc[0]
        except Exception:
            runtime_status = None
    add("runtime_status", EXPECTED_RUNTIME_STATUS, runtime_status, runtime_status == EXPECTED_RUNTIME_STATUS)

    # summary / DONE / manifest row count 일치
    summary_rows = summary_data.get("n_total_rows")
    done_rows_val = done_rows if DONE_JSON.exists() else None
    counts_match = (summary_rows == EXPECTED_TOTAL_ROWS == manifest_rows == done_rows_val)
    add("row_count_consistency",
        f"summary={EXPECTED_TOTAL_ROWS} DONE={EXPECTED_TOTAL_ROWS} manifest={EXPECTED_TOTAL_ROWS}",
        f"summary={summary_rows} DONE={done_rows_val} manifest={manifest_rows}",
        counts_match)

    return rows, summary


# ---------------------------------------------------------------------------
# Section C: manifest schema validation
# ---------------------------------------------------------------------------
def check_schema(df: pd.DataFrame) -> tuple[list[dict], dict]:
    print("\n[Section C] manifest schema validation")
    rows = []
    summary = {"all_required_present": True}
    for col in REQUIRED_SCHEMA_COLS:
        exists = col in df.columns
        status = "OK" if exists else "FAIL"
        note = "" if exists else "필수 컬럼 누락"
        rows.append({"section": "C", "column_name": col, "required": True, "exists": exists, "status": status, "note": note})
        if not exists:
            summary["all_required_present"] = False
        print(f"  [{status}] {col}")
    return rows, summary


# ---------------------------------------------------------------------------
# Section D: split/model/patient validation
# ---------------------------------------------------------------------------
def check_split_model_patient(df: pd.DataFrame) -> tuple[list[dict], dict]:
    print("\n[Section D] split/model/patient validation")
    rows = []
    summary = {}

    def add(item: str, expected, observed, passed: bool, note: str = "") -> None:
        status = _status(passed)
        rows.append({"section": "D", "check_item": item, "expected": str(expected), "observed": str(observed), "status": status, "note": note})
        summary[item] = {"expected": expected, "observed": observed, "passed": passed}
        print(f"  [{status}] {item}: expected={expected}, observed={observed}" + (f" ({note})" if note else ""))

    # stage_split unique
    ss_unique = set(df["stage_split"].unique().tolist()) if "stage_split" in df.columns else set()
    add("stage_split_unique", {EXPECTED_STAGE_SPLIT}, ss_unique, ss_unique == {EXPECTED_STAGE_SPLIT})

    # model_type unique
    mt_unique = set(df["model_type"].unique().tolist()) if "model_type" in df.columns else set()
    add("model_type_unique", {EXPECTED_MODEL_TYPE}, mt_unique, mt_unique == {EXPECTED_MODEL_TYPE})

    # patient count
    n_patients = df["patient_id"].nunique() if "patient_id" in df.columns else 0
    add("patient_count", EXPECTED_PATIENTS, n_patients, n_patients == EXPECTED_PATIENTS)

    # split CSV의 stage2_holdout patient set과 manifest patient set 일치
    split_holdout_patients: set[str] = set()
    split_safe_id_map: dict[str, str] = {}
    if SPLIT_CSV.exists():
        try:
            df_split = pd.read_csv(SPLIT_CSV, encoding="utf-8-sig")
            holdout_df = df_split[df_split["stage_split"] == EXPECTED_STAGE_SPLIT]
            split_holdout_patients = set(holdout_df["patient_id"].tolist())
            if "safe_id" in holdout_df.columns:
                for _, row in holdout_df.iterrows():
                    pid = row["patient_id"]
                    sid = row["safe_id"]
                    if pd.notna(sid) and str(sid).strip() != "":
                        split_safe_id_map[pid] = str(sid).strip()
        except Exception as e:
            print(f"  [FAIL] split CSV 로드 오류: {e}")

    manifest_patients = set(df["patient_id"].tolist()) if "patient_id" in df.columns else set()
    in_split_not_manifest = split_holdout_patients - manifest_patients
    in_manifest_not_split = manifest_patients - split_holdout_patients
    patient_set_match = (split_holdout_patients == manifest_patients)
    note = ""
    if in_split_not_manifest:
        note += f"split에만: {sorted(in_split_not_manifest)[:5]} "
    if in_manifest_not_split:
        note += f"manifest에만: {sorted(in_manifest_not_split)[:5]}"
    add("patient_set_match_with_split_csv", f"split={len(split_holdout_patients)}", f"manifest={len(manifest_patients)}", patient_set_match, note.strip())

    # stage1_dev patient 0명 (stage_split이 stage1_dev인 행)
    n_dev = int((df["stage_split"] == "stage1_dev").sum()) if "stage_split" in df.columns else 0
    add("stage1_dev_row_count", 0, n_dev, n_dev == 0)

    # v1v2 row 0행
    n_v1v2 = int((df["model_type"] == "v1v2").sum()) if "model_type" in df.columns else 0
    add("v1v2_row_count", 0, n_v1v2, n_v1v2 == 0)

    # safe_id empty/null 0행
    n_safe_empty = 0
    if "safe_id" in df.columns:
        n_safe_empty = int(df["safe_id"].isna().sum() + (df["safe_id"].fillna("").astype(str).str.strip() == "").sum())
    add("safe_id_empty_null_count", 0, n_safe_empty, n_safe_empty == 0)

    # patient_id-safe_id mapping이 split CSV와 일치
    mapping_mismatch_count = 0
    mapping_mismatch_examples = []
    if split_safe_id_map and "safe_id" in df.columns and "patient_id" in df.columns:
        per_patient = df.groupby("patient_id")["safe_id"].apply(
            lambda s: s.dropna().astype(str).str.strip().unique().tolist()
        )
        for pid, expected_sid in split_safe_id_map.items():
            if pid not in per_patient.index:
                continue
            observed_sids = per_patient[pid]
            observed_non_empty = [v for v in observed_sids if v != ""]
            if observed_non_empty and observed_non_empty != [expected_sid]:
                mapping_mismatch_count += 1
                mapping_mismatch_examples.append(f"{pid}:{observed_non_empty[0]}!={expected_sid}")
    mismatch_note = "; ".join(mapping_mismatch_examples[:3]) if mapping_mismatch_examples else ""
    add("safe_id_mapping_match_with_split_csv", 0, mapping_mismatch_count, mapping_mismatch_count == 0, mismatch_note)

    summary["split_holdout_patients"] = sorted(split_holdout_patients)
    return rows, summary


# ---------------------------------------------------------------------------
# Section E: label/sampling validation
# ---------------------------------------------------------------------------
def check_label_sampling(df: pd.DataFrame) -> tuple[list[dict], dict]:
    print("\n[Section E] label/sampling validation")
    rows = []
    summary = {}

    def add(item: str, expected, observed, passed: bool, fatal: bool = True, note: str = "") -> None:
        status = _status(passed, fatal)
        rows.append({"section": "E", "check_item": item, "expected": str(expected), "observed": str(observed), "status": status, "note": note})
        summary[item] = {"expected": expected, "observed": observed, "passed": passed}
        print(f"  [{status}] {item}: expected={expected}, observed={observed}" + (f" ({note})" if note else ""))

    label_unique = set(df["label"].unique().tolist()) if "label" in df.columns else set()
    add("label_unique_subset_{0,1}", "{0,1}", label_unique, label_unique.issubset({0, 1}))

    sl_unique = set(df["sampling_label"].unique().tolist()) if "sampling_label" in df.columns else set()
    add('sampling_label_unique_subset', '{"positive","hard_negative"}', sl_unique, sl_unique.issubset({"positive", "hard_negative"}))

    n_pos = int((df["sampling_label"] == "positive").sum()) if "sampling_label" in df.columns else 0
    add("positive_count", EXPECTED_POSITIVE, n_pos, n_pos == EXPECTED_POSITIVE)

    n_hn = int((df["sampling_label"] == "hard_negative").sum()) if "sampling_label" in df.columns else 0
    add("hard_negative_count", EXPECTED_HARD_NEGATIVE, n_hn, n_hn == EXPECTED_HARD_NEGATIVE)

    # sampling_label=positive → label=1
    n_pos_label_mismatch = 0
    if "sampling_label" in df.columns and "label" in df.columns:
        pos_rows = df[df["sampling_label"] == "positive"]
        n_pos_label_mismatch = int((pos_rows["label"] != 1).sum())
    add("positive_label_eq_1", 0, n_pos_label_mismatch, n_pos_label_mismatch == 0)

    # sampling_label=hard_negative → label=0
    n_hn_label_mismatch = 0
    if "sampling_label" in df.columns and "label" in df.columns:
        hn_rows = df[df["sampling_label"] == "hard_negative"]
        n_hn_label_mismatch = int((hn_rows["label"] != 0).sum())
    add("hard_negative_label_eq_0", 0, n_hn_label_mismatch, n_hn_label_mismatch == 0)

    # hard_negative count <= positive × 2
    hn_ratio_ok = (n_hn <= n_pos * 2) if n_pos > 0 else True
    add("hard_negative_le_positive_x2", f"<={n_pos*2}", n_hn, hn_ratio_ok, fatal=False)

    # patient별 hard_negative cap <= 600
    max_hn_per_patient = 0
    over_cap_patients = []
    if "sampling_label" in df.columns and "patient_id" in df.columns:
        hn_df = df[df["sampling_label"] == "hard_negative"]
        if len(hn_df) > 0:
            per_p = hn_df.groupby("patient_id").size()
            max_hn_per_patient = int(per_p.max())
            over_cap_patients = sorted(per_p[per_p > PATIENT_HN_CAP].index.tolist())
    add("patient_hn_cap_le_600", 0, len(over_cap_patients),
        len(over_cap_patients) == 0,
        note=f"max={max_hn_per_patient}" if over_cap_patients else f"max={max_hn_per_patient}")

    return rows, summary


# ---------------------------------------------------------------------------
# Section F: coordinate validation
# ---------------------------------------------------------------------------
def check_coordinates(df: pd.DataFrame) -> tuple[list[dict], dict]:
    print("\n[Section F] coordinate validation")
    rows = []
    summary = {}

    def add(item: str, expected, observed, passed: bool, fatal: bool = True, note: str = "") -> None:
        status = _status(passed, fatal)
        rows.append({"section": "F", "check_item": item, "expected": str(expected), "observed": str(observed), "status": status, "note": note})
        summary[item] = {"expected": expected, "observed": observed, "passed": passed}
        print(f"  [{status}] {item}: expected={expected}, observed={observed}" + (f" ({note})" if note else ""))

    for col in ["y0", "x0", "y1", "x1"]:
        n_null = int(df[col].isna().sum()) if col in df.columns else -1
        add(f"{col}_null_count", 0, n_null, n_null == 0)

    n_y_bad = int((df["y1"] <= df["y0"]).sum()) if ("y1" in df.columns and "y0" in df.columns) else -1
    add("y1_gt_y0_violations", 0, n_y_bad, n_y_bad == 0)

    n_x_bad = int((df["x1"] <= df["x0"]).sum()) if ("x1" in df.columns and "x0" in df.columns) else -1
    add("x1_gt_x0_violations", 0, n_x_bad, n_x_bad == 0)

    n_h_bad = int(((df["y1"] - df["y0"]) != 32).sum()) if ("y1" in df.columns and "y0" in df.columns) else -1
    add("patch_height_eq_32_violations", 0, n_h_bad, n_h_bad == 0)

    n_w_bad = int(((df["x1"] - df["x0"]) != 32).sum()) if ("x1" in df.columns and "x0" in df.columns) else -1
    add("patch_width_eq_32_violations", 0, n_w_bad, n_w_bad == 0)

    n_lz_null = int(df["local_z"].isna().sum()) if "local_z" in df.columns else -1
    add("local_z_null_count", 0, n_lz_null, n_lz_null == 0)

    n_dup_row_id = int(df["row_id"].duplicated().sum()) if "row_id" in df.columns else -1
    add("duplicate_row_id_count", 0, n_dup_row_id, n_dup_row_id == 0)

    dup_coord_count = 0
    if all(c in df.columns for c in ["patient_id", "local_z", "y0", "x0", "y1", "x1"]):
        dup_coord_count = int(df.duplicated(subset=["patient_id", "local_z", "y0", "x0", "y1", "x1"]).sum())
    add("duplicate_coordinate_count", 0, dup_coord_count, dup_coord_count == 0, fatal=False,
        note="WARNING: duplicate coordinates 존재" if dup_coord_count > 0 else "")

    return rows, summary


# ---------------------------------------------------------------------------
# Section G: contamination patient validation
# ---------------------------------------------------------------------------
def check_contamination(df: pd.DataFrame) -> tuple[list[dict], dict]:
    print("\n[Section G] contamination patient validation")
    rows = []
    summary = {}
    manifest_patients = set(df["patient_id"].tolist()) if "patient_id" in df.columns else set()

    for pid in sorted(CONTAMINATION_PATIENTS):
        sub = df[df["patient_id"] == pid] if "patient_id" in df.columns else pd.DataFrame()

        # 포함 여부
        exists = pid in manifest_patients
        status = "OK" if exists else "FAIL"
        rows.append({"section": "G", "patient_id": pid, "check_item": "포함 여부",
                     "expected": "True", "observed": str(exists), "status": status, "note": ""})
        summary.setdefault(pid, {})["exists"] = exists
        print(f"  [{status}] {pid} 포함: {exists}")

        # safe_id non-empty
        if exists:
            n_empty = int(sub["safe_id"].isna().sum() + (sub["safe_id"].fillna("").astype(str).str.strip() == "").sum()) if "safe_id" in sub.columns else len(sub)
            s_ok = n_empty == 0
            s_status = "OK" if s_ok else "FAIL"
            rows.append({"section": "G", "patient_id": pid, "check_item": "safe_id_non_empty",
                         "expected": "0", "observed": str(n_empty), "status": s_status, "note": ""})
            summary[pid]["safe_id_empty_count"] = n_empty
            print(f"  [{s_status}] {pid} safe_id empty: {n_empty}")

            # contamination_check_status
            obs_status_vals = sub["contamination_check_status"].unique().tolist() if "contamination_check_status" in sub.columns else []
            cs_ok = len(obs_status_vals) == 1 and obs_status_vals[0] == EXPECTED_CONTAMINATION_STATUS
            cs_status = "OK" if cs_ok else "FAIL"
            rows.append({"section": "G", "patient_id": pid, "check_item": "contamination_check_status",
                         "expected": EXPECTED_CONTAMINATION_STATUS, "observed": str(obs_status_vals), "status": cs_status, "note": ""})
            summary[pid]["contamination_check_status"] = obs_status_vals
            print(f"  [{cs_status}] {pid} contamination_check_status: {obs_status_vals}")
        else:
            for item in ["safe_id_non_empty", "contamination_check_status"]:
                rows.append({"section": "G", "patient_id": pid, "check_item": item,
                             "expected": "-", "observed": "환자 없음", "status": "FAIL", "note": ""})

    return rows, summary


# ---------------------------------------------------------------------------
# Section H: readiness decision
# ---------------------------------------------------------------------------
def check_readiness(
    sec_a: dict, sec_b_rows: list[dict], sec_c: dict, sec_d_rows: list[dict],
    sec_e_rows: list[dict], sec_f_rows: list[dict], sec_g_rows: list[dict],
    manifest_rows: int,
) -> tuple[list[dict], str, list[str]]:
    print("\n[Section H] readiness decision")

    blockers: list[str] = []
    h_rows: list[dict] = []

    def check(item: str, condition: bool, blocker_code: str, next_action: str) -> None:
        status = "PASS" if condition else "BLOCKED"
        blocker = "" if condition else blocker_code
        h_rows.append({"section": "H", "item": item, "status": status, "blocker": blocker, "next_required_action": next_action})
        if not condition:
            blockers.append(blocker_code)
        print(f"  [{status}] {item}" + (f" → {blocker_code}" if not condition else ""))

    # 1. final manifest exists
    check("final_manifest_exists",
          sec_a.get("final_manifest", False),
          "BLOCKED_PHASE8_2E_INCOMPLETE",
          "Phase 8.2E 재실행")

    # 2. DONE marker exists
    check("done_marker_exists",
          sec_a.get("done_marker", False),
          "BLOCKED_PHASE8_2E_INCOMPLETE",
          "Phase 8.2E DONE marker 확인")

    # 3. fatal validation issue 0 (B섹션 FAIL 없음)
    b_fails = [r for r in sec_b_rows if r["status"] == "FAIL"]
    check("completion_consistency_no_fail",
          len(b_fails) == 0,
          "BLOCKED_PHASE8_2E_INCOMPLETE",
          f"B섹션 FAIL 항목 {len(b_fails)}개 해소")

    # 4. required schema complete
    check("required_schema_complete",
          sec_c.get("all_required_present", False),
          "BLOCKED_MANIFEST_SCHEMA_MISMATCH",
          "누락 필수 컬럼 확인")

    # 5. 154 patients exactly
    d_patient_ok = any(
        r["check_item"] == "patient_count" and r["status"] == "OK"
        for r in sec_d_rows
    )
    check("patient_count_154",
          d_patient_ok,
          "BLOCKED_PATIENT_SET_MISMATCH",
          "stage2_holdout 환자 154명 확인")

    # 6. stage1_dev/v1v2 0
    d_dev_ok  = any(r["check_item"] == "stage1_dev_row_count" and r["status"] == "OK" for r in sec_d_rows)
    d_v1v2_ok = any(r["check_item"] == "v1v2_row_count" and r["status"] == "OK" for r in sec_d_rows)
    check("stage1_dev_v1v2_zero",
          d_dev_ok and d_v1v2_ok,
          "BLOCKED_PATIENT_SET_MISMATCH",
          "stage1_dev/v1v2 row 제거")

    # 7. safe_id complete
    d_safe_ok = any(r["check_item"] == "safe_id_empty_null_count" and r["status"] == "OK" for r in sec_d_rows)
    check("safe_id_complete",
          d_safe_ok,
          "BLOCKED_SAFE_ID_MISSING",
          "safe_id 공백/null 행 확인")

    # 8. coordinate valid (F섹션 FAIL 없음 — WARNING 제외)
    f_fatal_fails = [r for r in sec_f_rows if r["status"] == "FAIL"]
    check("coordinate_valid",
          len(f_fatal_fails) == 0,
          "BLOCKED_COORDINATE_VALIDATION_ERROR",
          f"F섹션 FAIL 항목 {len(f_fatal_fails)}개 해소")

    # 9. label/sampling_label consistent (E섹션 FAIL 없음)
    e_fails = [r for r in sec_e_rows if r["status"] == "FAIL"]
    check("label_sampling_consistent",
          len(e_fails) == 0,
          "BLOCKED_LABEL_SAMPLING_MISMATCH",
          f"E섹션 FAIL 항목 {len(e_fails)}개 해소")

    # 10. approval_required_before_crop_generation=True 전 row
    # row count mismatch
    check("row_count_match",
          manifest_rows == EXPECTED_TOTAL_ROWS,
          "BLOCKED_ROW_COUNT_MISMATCH",
          f"manifest row count 확인 (expected={EXPECTED_TOTAL_ROWS})")

    # 11. G섹션 FAIL 없음
    g_fails = [r for r in sec_g_rows if r["status"] == "FAIL"]
    check("contamination_patient_valid",
          len(g_fails) == 0,
          "BLOCKED_PATIENT_SET_MISMATCH",
          f"G섹션 FAIL 항목 {len(g_fails)}개 해소")

    # 최종 readiness
    unique_blockers = list(dict.fromkeys(blockers))
    if not unique_blockers:
        readiness = "READY_FOR_PHASE8_2F_DEDICATED_6CH_CROP_GENERATION_SCRIPT"
    else:
        readiness = unique_blockers[0]

    print(f"\n  최종 readiness: {readiness}")
    if unique_blockers:
        print(f"  blockers: {unique_blockers}")

    return h_rows, readiness, unique_blockers


# ---------------------------------------------------------------------------
# 저장 함수
# ---------------------------------------------------------------------------
def save_csv(all_rows: list[dict]) -> None:
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"  저장: {OUT_CSV} ({len(df_out)}행)")


def save_json(
    sec_a: dict, sec_b_rows: list[dict], sec_c: dict,
    sec_d_rows: list[dict], sec_e_rows: list[dict],
    sec_f_rows: list[dict], sec_g_rows: list[dict],
    readiness: str, blockers: list[str],
) -> None:
    def rows_to_summary(rows: list[dict]) -> dict:
        return {
            "total": len(rows),
            "fail_count": sum(1 for r in rows if r.get("status") == "FAIL"),
            "warning_count": sum(1 for r in rows if r.get("status") == "WARNING"),
            "ok_count": sum(1 for r in rows if r.get("status") == "OK"),
        }

    data = {
        "created_at": datetime.datetime.now().isoformat(),
        "input_paths": {
            "manifest": str(MANIFEST_PATH),
            "summary_json": str(SUMMARY_JSON),
            "errors_csv": str(ERRORS_CSV),
            "runtime_csv": str(RUNTIME_CSV),
            "done_json": str(DONE_JSON),
            "report_md": str(REPORT_MD),
            "split_csv": str(SPLIT_CSV),
        },
        "artifact_existence": sec_a,
        "completion_consistency": rows_to_summary(sec_b_rows),
        "manifest_schema_validation": sec_c,
        "split_model_patient_validation": rows_to_summary(sec_d_rows),
        "label_sampling_validation": rows_to_summary(sec_e_rows),
        "coordinate_validation": rows_to_summary(sec_f_rows),
        "contamination_patient_validation": rows_to_summary(sec_g_rows),
        "readiness_for_phase8_2f": readiness,
        "blockers": blockers,
        "notes": {
            "validation_only": True,
            "no_crop_generation": True,
            "no_npy_loading": True,
            "no_npz_loading": True,
            "no_model_forward": True,
            "no_scoring": True,
            "no_metric_calculation": True,
            "no_threshold": True,
            "no_training": True,
            "no_existing_file_modification": True,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  저장: {OUT_JSON}")


def save_md(
    manifest_rows: int, n_patients: int, n_pos: int, n_hn: int,
    sec_a_rows: list[dict], sec_b_rows: list[dict], sec_c_rows: list[dict],
    sec_d_rows: list[dict], sec_e_rows: list[dict], sec_f_rows: list[dict],
    sec_g_rows: list[dict], sec_h_rows: list[dict],
    readiness: str, blockers: list[str],
) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Phase 8.2E Output Validation 보고서\n",
        f"생성 일시: {now}\n",
        "## 1. 목적\n",
        "Phase 8.2E에서 생성된 stage2_holdout candidate coordinate manifest가",
        "crop 생성 입력으로 사용할 수 있는지 검증한다.\n",
        "## 2. Phase 8.2E 생성 결과 요약\n",
        f"| 항목 | 값 |",
        f"|------|-----|",
        f"| manifest row count | {manifest_rows:,} |",
        f"| patient count | {n_patients} |",
        f"| positive | {n_pos:,} |",
        f"| hard_negative | {n_hn:,} |",
        f"| stage_split | stage2_holdout |",
        f"| model_type | v2v2 |\n",
    ]

    def section_table(title: str, rows: list[dict], cols: list[str]) -> list[str]:
        out = [f"## {title}\n"]
        if not rows:
            out.append("(항목 없음)\n")
            return out
        header = " | ".join(cols)
        sep = " | ".join(["---"] * len(cols))
        out.append(f"| {header} |")
        out.append(f"| {sep} |")
        for r in rows:
            vals = " | ".join(str(r.get(c, "")) for c in cols)
            out.append(f"| {vals} |")
        out.append("")
        return out

    lines += section_table("3. artifact existence", sec_a_rows,
                           ["section", "item", "exists", "status", "note"])
    lines += section_table("4. completion consistency", sec_b_rows,
                           ["section", "item", "expected", "observed", "status", "note"])
    lines += section_table("5. manifest schema validation", sec_c_rows,
                           ["section", "column_name", "required", "exists", "status", "note"])
    lines += section_table("6. split/model/patient validation", sec_d_rows,
                           ["section", "check_item", "expected", "observed", "status", "note"])
    lines += section_table("7. label/sampling validation", sec_e_rows,
                           ["section", "check_item", "expected", "observed", "status", "note"])
    lines += section_table("8. coordinate validation", sec_f_rows,
                           ["section", "check_item", "expected", "observed", "status", "note"])
    lines += section_table("9. LUNG1-295/LUNG1-415 validation", sec_g_rows,
                           ["section", "patient_id", "check_item", "expected", "observed", "status", "note"])

    lines += [
        "## 10. readiness 판정\n",
        f"**{readiness}**\n",
    ]
    if blockers:
        lines += ["blockers:"]
        for b in blockers:
            lines.append(f"- {b}")
        lines.append("")

    lines += ["## 11. 다음 단계\n"]
    if readiness.startswith("READY"):
        lines += [
            "- Phase 8.2F dedicated 6ch crop generation script 작성/실행 전 검토",
            "- `approval_required_before_crop_generation=True` 확인 후 crop 생성 진행\n",
        ]
    else:
        lines += [f"- blocker 해소 후 재검증 필요: {blockers}\n"]

    lines += [
        "## 12. 금지 사항\n",
        "- crop 생성 금지",
        "- manifest 수정 금지",
        "- npy/npz 로드 금지",
        "- CT/ROI/mask 내용 확인 금지",
        "- model forward 금지",
        "- scoring 금지",
        "- metric 계산 금지",
        "- threshold/p95/p99/hit-rate 계산 금지",
        "- training 금지",
        "- checkpoint 생성 금지",
        "- 기존 Phase 6/7/8 output 수정 금지",
        "- DIAG_CSV 수정 금지",
        "- v1v2/stage1_dev row 사용 금지",
        "- pip/conda install 금지",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  저장: {OUT_MD}")


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 8.2E output validation"
    )
    parser.add_argument("--run", action="store_true", help="실제 검증 결과 생성")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # output guard
    output_guard()

    if not args.run:
        dry_run_report()
        return

    print("\n=== Phase 8.2E Output Validation 실행 시작 ===")

    # manifest 로드
    print(f"\n[manifest 로드] {MANIFEST_PATH.name}")
    df = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig", low_memory=False)
    manifest_rows = len(df)
    n_patients = df["patient_id"].nunique() if "patient_id" in df.columns else 0
    n_pos = int((df["sampling_label"] == "positive").sum()) if "sampling_label" in df.columns else 0
    n_hn = int((df["sampling_label"] == "hard_negative").sum()) if "sampling_label" in df.columns else 0
    print(f"  로드 완료: {manifest_rows:,}행, {n_patients}명, pos={n_pos:,}, hn={n_hn:,}")

    # 섹션별 검증
    sec_a_rows, sec_a_summary = check_artifact_existence()
    sec_b_rows, _             = check_completion_consistency(manifest_rows)
    sec_c_rows, sec_c_summary = check_schema(df)
    sec_d_rows, _             = check_split_model_patient(df)
    sec_e_rows, _             = check_label_sampling(df)
    sec_f_rows, _             = check_coordinates(df)
    sec_g_rows, _             = check_contamination(df)
    sec_h_rows, readiness, blockers = check_readiness(
        sec_a_summary, sec_b_rows, sec_c_summary,
        sec_d_rows, sec_e_rows, sec_f_rows, sec_g_rows,
        manifest_rows,
    )

    # 출력 디렉토리 생성 (exist_ok=False)
    OUT_ROOT.mkdir(parents=True, exist_ok=False)

    # 저장
    print(f"\n[저장] 결과 파일 저장 중...")
    all_rows = sec_a_rows + sec_b_rows + sec_c_rows + sec_d_rows + sec_e_rows + sec_f_rows + sec_g_rows + sec_h_rows
    save_csv(all_rows)
    save_json(
        sec_a_summary, sec_b_rows, sec_c_summary,
        sec_d_rows, sec_e_rows, sec_f_rows, sec_g_rows,
        readiness, blockers,
    )
    save_md(
        manifest_rows, n_patients, n_pos, n_hn,
        sec_a_rows, sec_b_rows, sec_c_rows,
        sec_d_rows, sec_e_rows, sec_f_rows,
        sec_g_rows, sec_h_rows,
        readiness, blockers,
    )

    print(f"\n=== Phase 8.2E Output Validation 완료 ===")
    print(f"  readiness: {readiness}")
    print(f"  출력 디렉토리: {OUT_ROOT}")


if __name__ == "__main__":
    main()
