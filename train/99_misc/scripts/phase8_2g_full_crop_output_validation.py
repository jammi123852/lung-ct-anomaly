"""
Phase 8.2G Full Crop Output Validation

목적: Phase 8.2F full run으로 생성된 stage2_holdout dedicated 6ch crop 산출물이
      scoring 입력으로 사용 가능한지 검증한다.

실행 방식:
  --run 없으면 dry-run 출력 후 종료
  --run 있으면 실제 검증 실행

금지: 추가 crop 생성, crop/manifest 수정, model forward, scoring, metric 계산,
      threshold, p95/p99/hit-rate 계산, training/checkpoint 생성,
      48개 초과 npz 로드, 기존 Phase 6/7/8 output 수정, DIAG_CSV 수정, pip install
"""

import sys
import gc
import json
import random
import pathlib
import datetime
import argparse
import csv as csv_module

import numpy as np
import pandas as pd

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

CROP_ROOT     = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_stage2_holdout_6ch_dedicated_v1"
MANIFEST_PATH = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_filtered_manifest_v1.csv"

PHASE8F_OUT_ROOT = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_2f_stage2_dedicated_6ch_crop_generation_v1"
)
SUMMARY_JSON = PHASE8F_OUT_ROOT / "phase8_2f_stage2_dedicated_6ch_crop_generation_summary_v1.json"
ERRORS_CSV   = PHASE8F_OUT_ROOT / "phase8_2f_stage2_dedicated_6ch_crop_generation_errors_v1.csv"
RUNTIME_CSV  = PHASE8F_OUT_ROOT / "phase8_2f_stage2_dedicated_6ch_crop_generation_runtime_summary_v1.csv"
DONE_JSON    = PHASE8F_OUT_ROOT / "phase8_2f_stage2_dedicated_6ch_crop_generation_DONE.json"
REPORT_MD    = PHASE8F_OUT_ROOT / "phase8_2f_stage2_dedicated_6ch_crop_generation_report_v1.md"

SPLIT_CSV = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"

OUT_ROOT   = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_2g_full_crop_output_validation_v1"
)
OUT_CSV    = OUT_ROOT / "phase8_2g_full_crop_output_validation_v1.csv"
OUT_JSON   = OUT_ROOT / "phase8_2g_full_crop_output_validation_v1.json"
OUT_REPORT = OUT_ROOT / "phase8_2g_full_crop_output_validation_report_v1.md"

# ── 기대값 상수 ────────────────────────────────────────────────────────────────
EXPECTED_TOTAL    = 143735
EXPECTED_PATIENTS = 154
EXPECTED_POSITIVE = 51335
EXPECTED_HN       = 92400
CONTAMINATED_PATIENTS = {"LUNG1-295", "LUNG1-415"}
REQUIRED_COLUMNS = [
    "row_id", "patient_id", "safe_id", "npz_path", "label", "sampling_label",
    "stage_split", "source_coordinate_manifest", "source_crop_root",
    "source_ct_path", "source_roi_path", "source_lesion_mask_path",
    "asset_scope", "contamination_check_status", "approval_required_before_scoring",
    "manifest_status", "crop_shape", "input_channels", "crop_size",
    "generation_status", "issue", "note",
]
SAMPLE_LIMIT = 48

ALL_COLUMNS = [
    "section", "item", "path", "exists", "check_item", "expected", "observed",
    "column_name", "required", "patient_id", "sample_id", "sampling_label",
    "npz_path", "blocker", "next_required_action", "status", "note",
]


def make_row(**kwargs):
    row = {c: "" for c in ALL_COLUMNS}
    row.update(kwargs)
    return row


def output_guard():
    if OUT_ROOT.exists():
        print(f"[ABORT] output root already exists: {OUT_ROOT}", file=sys.stderr)
        sys.exit(1)
    for fp in [OUT_CSV, OUT_JSON, OUT_REPORT]:
        if fp.exists():
            print(f"[ABORT] output file already exists: {fp}", file=sys.stderr)
            sys.exit(1)


def dry_run_report():
    print()
    print("=" * 70)
    print("  [DRY-RUN] Phase 8.2G Full Crop Output Validation")
    print("=" * 70)
    print(f"  crop root   : {CROP_ROOT}")
    print(f"  manifest    : {MANIFEST_PATH}")
    print(f"  split CSV   : {SPLIT_CSV}")
    print(f"  output root : {OUT_ROOT}")
    print()
    for fp, label in [
        (OUT_ROOT,   "output root"),
        (OUT_CSV,    "output CSV"),
        (OUT_JSON,   "output JSON"),
        (OUT_REPORT, "output MD"),
    ]:
        print(f"    {label}: {'충돌' if fp.exists() else '안전'}")
    print()
    print("  실행 명령:")
    print("    source ~/ai_env/bin/activate && \\")
    print("    python scripts/phase8_2g_full_crop_output_validation.py --run")
    print("=" * 70)


# ── Section A ─────────────────────────────────────────────────────────────────
def check_section_a():
    rows = []
    blockers = []
    artifacts = [
        ("crop_root",    CROP_ROOT,     "Phase 8.2F dedicated crop root"),
        ("manifest",     MANIFEST_PATH, "Phase 8.2F filtered manifest CSV"),
        ("summary_json", SUMMARY_JSON,  "Phase 8.2F summary JSON"),
        ("errors_csv",   ERRORS_CSV,    "Phase 8.2F errors CSV"),
        ("runtime_csv",  RUNTIME_CSV,   "Phase 8.2F runtime summary CSV"),
        ("done_json",    DONE_JSON,     "Phase 8.2F DONE marker JSON"),
        ("report_md",    REPORT_MD,     "Phase 8.2F report MD"),
    ]
    for item, path, note in artifacts:
        exists = path.exists()
        status = "PASS" if exists else "FAIL"
        if not exists:
            blockers.append("BLOCKED_FULL_CROP_ARTIFACT_MISSING")
        rows.append(make_row(
            section="A", item=item, path=str(path),
            exists=str(exists), status=status, note=note,
        ))
    return rows, list(set(blockers))


# ── Section B ─────────────────────────────────────────────────────────────────
def check_section_b(df_manifest):
    rows = []
    blockers = []

    done_data = {}
    if DONE_JSON.exists():
        with open(DONE_JSON, encoding="utf-8") as f:
            done_data = json.load(f)

    summary_data = {}
    if SUMMARY_JSON.exists():
        with open(SUMMARY_JSON, encoding="utf-8") as f:
            summary_data = json.load(f)

    error_count = 0
    if ERRORS_CSV.exists():
        df_err = pd.read_csv(ERRORS_CSV)
        error_count = len(df_err)

    runtime_patient_count = 0
    if RUNTIME_CSV.exists():
        df_rt = pd.read_csv(RUNTIME_CSV)
        if "patient_id" in df_rt.columns:
            runtime_patient_count = df_rt["patient_id"].nunique()

    manifest_row_count = len(df_manifest) if df_manifest is not None else -1

    checks = [
        ("done_status",               "DONE",                    done_data.get("status", ""),
         done_data.get("status", "") == "DONE"),
        ("mode",                      "full",                    summary_data.get("mode", ""),
         summary_data.get("mode", "") == "full"),
        ("is_partial_run",            "False",                   str(summary_data.get("is_partial_run", True)),
         str(summary_data.get("is_partial_run", True)) == "False"),
        ("total_success",             str(EXPECTED_TOTAL),       str(summary_data.get("total_success", -1)),
         str(summary_data.get("total_success", -1)) == str(EXPECTED_TOTAL)),
        ("total_error",               "0",                       str(summary_data.get("total_error", -1)),
         str(summary_data.get("total_error", -1)) == "0"),
        ("patient_count",             str(EXPECTED_PATIENTS),    str(summary_data.get("patient_count", -1)),
         str(summary_data.get("patient_count", -1)) == str(EXPECTED_PATIENTS)),
        ("positive_count",            str(EXPECTED_POSITIVE),    str(summary_data.get("positive_count", -1)),
         str(summary_data.get("positive_count", -1)) == str(EXPECTED_POSITIVE)),
        ("hard_negative_count",       str(EXPECTED_HN),          str(summary_data.get("hard_negative_count", -1)),
         str(summary_data.get("hard_negative_count", -1)) == str(EXPECTED_HN)),
        ("npz_path_exists_count",     str(EXPECTED_TOTAL),       str(summary_data.get("npz_path_exists_count", -1)),
         str(summary_data.get("npz_path_exists_count", -1)) == str(EXPECTED_TOTAL)),
        ("npz_path_missing_count",    "0",                       str(summary_data.get("npz_path_missing_count", -1)),
         str(summary_data.get("npz_path_missing_count", -1)) == "0"),
        ("npz_path_duplicate_count",  "0",                       str(summary_data.get("npz_path_duplicate_count", -1)),
         str(summary_data.get("npz_path_duplicate_count", -1)) == "0"),
        ("npz_path_empty_null_count", "0",                       str(summary_data.get("npz_path_empty_null_count", -1)),
         str(summary_data.get("npz_path_empty_null_count", -1)) == "0"),
        ("post_validation_passed",    "True",
         str(summary_data.get("post_validation", {}).get("passed", False)),
         str(summary_data.get("post_validation", {}).get("passed", False)) == "True"),
        ("summary_total_vs_manifest", str(EXPECTED_TOTAL),       str(manifest_row_count),
         str(manifest_row_count) == str(EXPECTED_TOTAL)),
        ("error_csv_row_count",       "0",                       str(error_count),
         str(error_count) == "0"),
        ("runtime_patient_count",     str(EXPECTED_PATIENTS),    str(runtime_patient_count),
         str(runtime_patient_count) == str(EXPECTED_PATIENTS)),
    ]

    for check_item, expected, observed, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            blockers.append("BLOCKED_FULL_SUMMARY_MISMATCH")
        rows.append(make_row(
            section="B", check_item=check_item,
            expected=expected, observed=observed,
            status=status, note="",
        ))
    return rows, list(set(blockers))


# ── Section C ─────────────────────────────────────────────────────────────────
def check_section_c(df_manifest):
    rows = []
    blockers = []
    actual_cols = set(df_manifest.columns) if df_manifest is not None else set()
    for col in REQUIRED_COLUMNS:
        exists = col in actual_cols
        status = "PASS" if exists else "FAIL"
        if not exists:
            blockers.append("BLOCKED_FULL_MANIFEST_SCHEMA_MISMATCH")
        rows.append(make_row(
            section="C", column_name=col, required="True",
            exists=str(exists), status=status, note="",
        ))
    return rows, list(set(blockers))


# ── Section D ─────────────────────────────────────────────────────────────────
def check_section_d(df_manifest):
    rows = []
    blockers = []

    if df_manifest is None:
        rows.append(make_row(section="D", check_item="manifest_load", status="FAIL",
                             note="manifest not loaded"))
        blockers.append("BLOCKED_FULL_MANIFEST_CONTENT_MISMATCH")
        return rows, blockers

    split_stage2_patients = set()
    if SPLIT_CSV.exists():
        df_split = pd.read_csv(SPLIT_CSV)
        if "stage_split" in df_split.columns and "patient_id" in df_split.columns:
            split_stage2_patients = set(
                df_split.loc[df_split["stage_split"] == "stage2_holdout", "patient_id"].unique()
            )

    manifest_patients = set(df_manifest["patient_id"].unique())
    patient_set_ok    = manifest_patients == split_stage2_patients
    row_count         = len(df_manifest)
    patient_count     = df_manifest["patient_id"].nunique()

    stage_split_set = set(df_manifest["stage_split"].unique()) if "stage_split" in df_manifest.columns else set()
    label_set       = set(df_manifest["label"].unique()) if "label" in df_manifest.columns else set()
    sl_set          = set(df_manifest["sampling_label"].unique()) if "sampling_label" in df_manifest.columns else set()

    pos_count = int((df_manifest["label"] == 1).sum()) if "label" in df_manifest.columns else -1
    hn_count  = int((df_manifest["label"] == 0).sum()) if "label" in df_manifest.columns else -1

    mismatch_pos = 0
    mismatch_hn  = 0
    if "sampling_label" in df_manifest.columns and "label" in df_manifest.columns:
        mismatch_pos = int(((df_manifest["sampling_label"] == "positive") & (df_manifest["label"] != 1)).sum())
        mismatch_hn  = int(((df_manifest["sampling_label"] == "hard_negative") & (df_manifest["label"] != 0)).sum())

    safe_id_null  = 0
    npz_path_null = 0
    npz_dup       = 0
    if "safe_id" in df_manifest.columns:
        safe_id_null = int(df_manifest["safe_id"].isna().sum()) + int(
            (df_manifest["safe_id"].astype(str).str.strip() == "").sum()
        )
    if "npz_path" in df_manifest.columns:
        npz_path_null = int(df_manifest["npz_path"].isna().sum()) + int(
            (df_manifest["npz_path"].astype(str).str.strip() == "").sum()
        )
        npz_dup = int(df_manifest["npz_path"].duplicated().sum())

    approval_invalid    = 0
    asset_scope_invalid = 0
    manifest_st_invalid = 0
    gen_status_invalid  = 0
    crop_shape_invalid  = 0
    in_ch_invalid       = 0
    crop_size_invalid   = 0

    if "approval_required_before_scoring" in df_manifest.columns:
        ap_norm = df_manifest["approval_required_before_scoring"].astype(str).str.strip().str.lower()
        approval_invalid = int((~ap_norm.isin(["true", "1"])).sum())
    if "asset_scope" in df_manifest.columns:
        asset_scope_invalid = int((df_manifest["asset_scope"] != "dedicated_stage2_holdout_6ch_crop").sum())
    if "manifest_status" in df_manifest.columns:
        manifest_st_invalid = int((df_manifest["manifest_status"] != "created_after_phase8_2f_run").sum())
    if "generation_status" in df_manifest.columns:
        gen_status_invalid = int((df_manifest["generation_status"] != "generated").sum())
    if "crop_shape" in df_manifest.columns:
        crop_shape_invalid = int((df_manifest["crop_shape"] != "(6,96,96)").sum())
    if "input_channels" in df_manifest.columns:
        in_ch_invalid = int((df_manifest["input_channels"].astype(str) != "6").sum())
    if "crop_size" in df_manifest.columns:
        crop_size_invalid = int((df_manifest["crop_size"].astype(str) != "96").sum())

    checks = [
        ("row_count",
         str(EXPECTED_TOTAL), str(row_count), str(row_count) == str(EXPECTED_TOTAL)),
        ("patient_count",
         str(EXPECTED_PATIENTS), str(patient_count), str(patient_count) == str(EXPECTED_PATIENTS)),
        ("stage2_patient_set_vs_split",
         "MATCH",
         "MATCH" if patient_set_ok else f"MISMATCH(m={len(manifest_patients)},s={len(split_stage2_patients)})",
         patient_set_ok),
        ("stage_split_unique",
         "{'stage2_holdout'}", str(stage_split_set), stage_split_set == {"stage2_holdout"}),
        ("label_unique_subset",
         "subset_of_{0,1}", str(sorted(label_set)), label_set <= {0, 1}),
        ("sampling_label_unique",
         "subset_of_{positive,hard_negative}", str(sorted(sl_set)),
         sl_set <= {"positive", "hard_negative"}),
        ("positive_count",
         str(EXPECTED_POSITIVE), str(pos_count), str(pos_count) == str(EXPECTED_POSITIVE)),
        ("hard_negative_count",
         str(EXPECTED_HN), str(hn_count), str(hn_count) == str(EXPECTED_HN)),
        ("pos_label_mismatch",       "0", str(mismatch_pos),       mismatch_pos == 0),
        ("hn_label_mismatch",        "0", str(mismatch_hn),        mismatch_hn == 0),
        ("safe_id_empty_null",       "0", str(safe_id_null),       safe_id_null == 0),
        ("npz_path_empty_null",      "0", str(npz_path_null),      npz_path_null == 0),
        ("npz_path_duplicate",       "0", str(npz_dup),            npz_dup == 0),
        ("approval_required_invalid","0", str(approval_invalid),   approval_invalid == 0),
        ("asset_scope_invalid",      "0", str(asset_scope_invalid),asset_scope_invalid == 0),
        ("manifest_status_invalid",  "0", str(manifest_st_invalid),manifest_st_invalid == 0),
        ("generation_status_invalid","0", str(gen_status_invalid), gen_status_invalid == 0),
        ("crop_shape_invalid",       "0", str(crop_shape_invalid), crop_shape_invalid == 0),
        ("input_channels_invalid",   "0", str(in_ch_invalid),      in_ch_invalid == 0),
        ("crop_size_invalid",        "0", str(crop_size_invalid),  crop_size_invalid == 0),
    ]

    for check_item, expected, observed, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            blockers.append("BLOCKED_FULL_MANIFEST_CONTENT_MISMATCH")
        rows.append(make_row(
            section="D", check_item=check_item,
            expected=expected, observed=observed,
            status=status, note="",
        ))
    return rows, list(set(blockers))


# ── Section E ─────────────────────────────────────────────────────────────────
def check_section_e(df_manifest):
    rows = []
    blockers = []

    for pid in sorted(CONTAMINATED_PATIENTS):
        sub = df_manifest[df_manifest["patient_id"] == pid] if df_manifest is not None else pd.DataFrame()
        contains = len(sub) > 0

        rows.append(make_row(
            section="E", patient_id=pid, check_item="patient_id_present",
            expected="True", observed=str(contains),
            status="PASS" if contains else "FAIL", note="",
        ))

        if contains and "safe_id" in sub.columns:
            safe_val = str(sub["safe_id"].iloc[0])
            safe_ok  = safe_val.strip() not in ("", "nan", "None")
        else:
            safe_val = ""
            safe_ok  = False
        rows.append(make_row(
            section="E", patient_id=pid, check_item="safe_id_non_empty",
            expected="True", observed=str(safe_ok),
            status="PASS" if safe_ok else "FAIL",
            note=safe_val[:80] if contains else "",
        ))

        expected_cs = "regenerated_crop_from_source_after_prior_contamination"
        if contains and "contamination_check_status" in sub.columns:
            obs_cs = str(sub["contamination_check_status"].iloc[0])
        else:
            obs_cs = ""
        cs_ok = obs_cs == expected_cs
        rows.append(make_row(
            section="E", patient_id=pid, check_item="contamination_check_status",
            expected=expected_cs, observed=obs_cs,
            status="PASS" if cs_ok else "FAIL", note="",
        ))

        if contains and "npz_path" in sub.columns:
            npz_ec = int(sub["npz_path"].apply(lambda p: pathlib.Path(str(p)).exists()).sum())
        else:
            npz_ec = 0
        rows.append(make_row(
            section="E", patient_id=pid, check_item="npz_path_exists_count_gt0",
            expected=">0", observed=str(npz_ec),
            status="PASS" if npz_ec > 0 else "FAIL", note="",
        ))

        if not (contains and safe_ok and cs_ok and npz_ec > 0):
            blockers.append("BLOCKED_CONTAMINATION_PATIENT_STATUS")

    return rows, list(set(blockers))


# ── Section F ─────────────────────────────────────────────────────────────────
def check_section_f(df_manifest):
    rows = []
    blockers = []

    exists_count  = 0
    missing_count = 0
    zero_byte     = 0
    bad_ext       = 0
    sample_issues = []

    if df_manifest is not None and "npz_path" in df_manifest.columns:
        total = len(df_manifest)
        print(f"  [F] stat 시작 (총 {total:,}개)...")
        for i, npz_str in enumerate(df_manifest["npz_path"]):
            p = pathlib.Path(str(npz_str))
            if not p.exists():
                missing_count += 1
                if len(sample_issues) < 5:
                    sample_issues.append(f"MISSING:{npz_str}")
                continue
            exists_count += 1
            try:
                sz = p.stat().st_size
            except OSError:
                sz = 0
            if sz == 0:
                zero_byte += 1
                if len(sample_issues) < 5:
                    sample_issues.append(f"ZERO_BYTE:{npz_str}")
            if p.suffix != ".npz":
                bad_ext += 1
                if len(sample_issues) < 5:
                    sample_issues.append(f"BAD_EXT({p.suffix}):{npz_str}")
            if (i + 1) % 10000 == 0:
                print(f"  [F] {i+1:,}/{total:,} 처리 중...")
        print(f"  [F] stat 완료: exists={exists_count:,}, missing={missing_count}, "
              f"zero_byte={zero_byte}, bad_ext={bad_ext}")

    issue_note = "; ".join(sample_issues[:5])

    checks = [
        ("npz_exists_count",  str(EXPECTED_TOTAL), str(exists_count),  exists_count == EXPECTED_TOTAL),
        ("npz_missing_count", "0",                 str(missing_count),  missing_count == 0),
        ("npz_zero_byte",     "0",                 str(zero_byte),      zero_byte == 0),
        ("npz_bad_extension", "0",                 str(bad_ext),        bad_ext == 0),
    ]
    for check_item, expected, observed, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            blockers.append("BLOCKED_NPZ_PATH_STAT_ERROR")
        rows.append(make_row(
            section="F", check_item=check_item,
            expected=expected, observed=observed,
            status=status,
            note=issue_note if not ok else "",
        ))
    return rows, list(set(blockers))


# ── Section G ─────────────────────────────────────────────────────────────────
def _sample_diverse(df_sub, n, seed):
    if len(df_sub) == 0:
        return pd.DataFrame()
    patients = sorted(df_sub["patient_id"].unique())
    rng = random.Random(seed)
    groups = {}
    for p in patients:
        idx = df_sub[df_sub["patient_id"] == p].index.tolist()
        rng.shuffle(idx)
        groups[p] = idx
    result_idx = []
    i = 0
    max_iter = n * (len(patients) + 1)
    while len(result_idx) < n and i < max_iter:
        p = patients[i % len(patients)]
        if groups[p]:
            result_idx.append(groups[p].pop(0))
        i += 1
        if all(len(v) == 0 for v in groups.values()):
            break
    return df_sub.loc[result_idx]


def check_section_g(df_manifest):
    rows = []
    blockers = []

    if df_manifest is None or "sampling_label" not in df_manifest.columns:
        rows.append(make_row(section="G", check_item="sample_npz_load",
                             status="SKIP", note="manifest not available"))
        return rows, blockers

    df_pos = df_manifest[df_manifest["sampling_label"] == "positive"].copy()
    df_hn  = df_manifest[df_manifest["sampling_label"] == "hard_negative"].copy()

    n_pos = SAMPLE_LIMIT // 2
    n_hn  = SAMPLE_LIMIT // 2

    df_pos_sample = _sample_diverse(df_pos, n_pos, seed=42)
    df_hn_sample  = _sample_diverse(df_hn,  n_hn,  seed=43)

    # CONTAMINATED_PATIENTS 포함 보장
    contaminated_present = [p for p in CONTAMINATED_PATIENTS
                            if p in set(df_manifest["patient_id"].unique())]
    if contaminated_present:
        already = set()
        if len(df_pos_sample) > 0:
            already.update(df_pos_sample["patient_id"].tolist())
        if len(df_hn_sample) > 0:
            already.update(df_hn_sample["patient_id"].tolist())
        for pid in contaminated_present:
            if pid not in already:
                sub = df_manifest[df_manifest["patient_id"] == pid]
                if len(sub) > 0:
                    add_r = sub.iloc[[0]]
                    if len(df_pos_sample) < n_pos:
                        df_pos_sample = pd.concat([df_pos_sample, add_r])
                    else:
                        df_hn_sample = pd.concat([df_hn_sample, add_r])
                    already.add(pid)

    df_samples = pd.concat([df_pos_sample, df_hn_sample]).head(SAMPLE_LIMIT).reset_index(drop=True)

    loaded_count = 0
    for idx, mrow in df_samples.iterrows():
        if loaded_count >= SAMPLE_LIMIT:
            break

        sid_str    = str(idx + 1)
        npz_str    = str(mrow.get("npz_path", ""))
        pid_str    = str(mrow.get("patient_id", ""))
        sl_str     = str(mrow.get("sampling_label", ""))
        m_label    = mrow.get("label", None)

        p = pathlib.Path(npz_str)
        if not p.exists():
            rows.append(make_row(
                section="G", sample_id=sid_str, patient_id=pid_str,
                sampling_label=sl_str, npz_path=npz_str,
                check_item="file_exists", expected="True", observed="False",
                status="FAIL", note="npz not found",
            ))
            blockers.append("BLOCKED_SAMPLE_NPZ_INVALID")
            continue

        try:
            data = np.load(str(p), allow_pickle=False)
        except Exception as exc:
            rows.append(make_row(
                section="G", sample_id=sid_str, patient_id=pid_str,
                sampling_label=sl_str, npz_path=npz_str,
                check_item="npz_load", expected="OK", observed="ERROR",
                status="FAIL", note=str(exc)[:120],
            ))
            blockers.append("BLOCKED_SAMPLE_NPZ_INVALID")
            continue

        loaded_count += 1
        npz_checks = []

        has_image = "image" in data.files
        npz_checks.append(("key_image_exists", "True", str(has_image), has_image))

        if has_image:
            img = data["image"]
            shape_ok  = img.shape == (6, 96, 96)
            dtype_ok  = np.issubdtype(img.dtype, np.floating)
            img_min   = float(img.min())
            img_max   = float(img.max())
            finite_ok = bool(np.isfinite(img).all())
            npz_checks.append(("image_shape",    "(6, 96, 96)", str(img.shape),        shape_ok))
            npz_checks.append(("dtype_float",    "floating",    str(img.dtype),         dtype_ok))
            npz_checks.append(("value_min_ge_0", "True",        str(round(img_min, 6)), img_min >= 0))
            npz_checks.append(("value_max_le_1", "True",        str(round(img_max, 6)), img_max <= 1.0))
            npz_checks.append(("no_nan_inf",      "True",       str(finite_ok),         finite_ok))
            del img

        if "label" in data.files and m_label is not None:
            npz_label = int(data["label"])
            npz_checks.append(("label_vs_manifest", str(int(m_label)), str(npz_label),
                                npz_label == int(m_label)))

        if "sampling_label" in data.files:
            npz_sl = str(data["sampling_label"])
            npz_checks.append(("sampling_label_vs_manifest", sl_str, npz_sl, npz_sl == sl_str))

        if "patient_id" in data.files:
            npz_pid = str(data["patient_id"])
            npz_checks.append(("patient_id_vs_manifest", pid_str, npz_pid, npz_pid == pid_str))

        if "local_z" in data.files:
            npz_checks.append(("local_z_recorded", "info", str(data["local_z"]), True))

        del data
        gc.collect()

        for check_item, expected, observed, ok in npz_checks:
            status = "PASS" if ok else "FAIL"
            if not ok:
                blockers.append("BLOCKED_SAMPLE_NPZ_INVALID")
            rows.append(make_row(
                section="G", sample_id=sid_str, patient_id=pid_str,
                sampling_label=sl_str, npz_path=npz_str,
                check_item=check_item, expected=expected, observed=observed,
                status=status, note="",
            ))

    print(f"  [G] 샘플 {loaded_count}개 로드 완료")
    return rows, list(set(blockers))


# ── Section H ─────────────────────────────────────────────────────────────────
def build_section_h(unique_blockers):
    rows = []
    if not unique_blockers:
        rows.append(make_row(
            section="H", item="readiness",
            status="READY_FOR_PHASE8_3_STAGE2_SCORING_SMOKE_PREFLIGHT",
            blocker="",
            next_required_action="Phase 8.3 stage2_holdout scoring smoke preflight 진행",
        ))
    else:
        for blocker in unique_blockers:
            rows.append(make_row(
                section="H", item="blocker",
                status="BLOCKED", blocker=blocker,
                next_required_action="blocker 해소 후 재검증",
            ))
    return rows


# ── 저장 ──────────────────────────────────────────────────────────────────────
def save_csv(all_rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)


def save_json(results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def save_report(results, path):
    phase8f   = results.get("phase8f_summary", {})
    blockers  = results.get("blockers", [])
    readiness = results.get("readiness_for_phase8_3", "UNKNOWN")

    def tbl(headers, data_rows):
        sep  = "| " + " | ".join(["---"] * len(headers)) + " |"
        hdr  = "| " + " | ".join(headers) + " |"
        lines = [hdr, sep]
        for dr in data_rows:
            lines.append("| " + " | ".join(str(v) for v in dr) + " |")
        return "\n".join(lines)

    a_rows = results["artifact_existence"]["rows"]
    b_rows = results["completion_consistency"]["rows"]
    c_rows = results["manifest_schema_validation"]["rows"]
    d_rows = results["manifest_content_validation"]["rows"]
    e_rows = results["contamination_patient_validation"]["rows"]
    f_rows = results["full_npz_path_stat_validation"]["rows"]
    g_rows = results["sample_npz_validation"]["rows"]

    g_sample_count = len(set(r["sample_id"] for r in g_rows if r["sample_id"]))
    blocker_str = ", ".join(blockers) if blockers else "없음"
    next_step = (
        "Phase 8.3 stage2_holdout scoring smoke preflight 진행"
        if not blockers else "blocker 해소 후 재검증"
    )

    md = f"""# Phase 8.2G Full Crop Output Validation Report

생성일: {results.get("run_end", "")}

## 1. Phase 8.2G 목적

Phase 8.2F full run으로 생성된 stage2_holdout dedicated 6ch crop 산출물이
scoring 입력으로 사용 가능한지 검증한다.
model forward, scoring, metric 계산, threshold 계산은 하지 않는다.

## 2. Phase 8.2F Full Run 결과 요약

| 항목 | 값 |
|------|-----|
| mode | {phase8f.get("mode", "")} |
| total_success | {phase8f.get("total_success", "")} |
| total_error | {phase8f.get("total_error", "")} |
| patient_count | {phase8f.get("patient_count", "")} |
| positive_count | {phase8f.get("positive_count", "")} |
| hard_negative_count | {phase8f.get("hard_negative_count", "")} |
| npz_path_exists_count | {phase8f.get("npz_path_exists_count", "")} |
| elapsed_seconds | {phase8f.get("elapsed_seconds", "")} |

## 3. Artifact Existence

{tbl(["item", "exists", "status", "note"],
     [[r["item"], r["exists"], r["status"], r["note"]] for r in a_rows])}

## 4. Completion Consistency

{tbl(["check_item", "expected", "observed", "status"],
     [[r["check_item"], r["expected"], r["observed"], r["status"]] for r in b_rows])}

## 5. Manifest Schema Validation

{tbl(["column_name", "exists", "status"],
     [[r["column_name"], r["exists"], r["status"]] for r in c_rows])}

## 6. Manifest Content Validation

{tbl(["check_item", "expected", "observed", "status"],
     [[r["check_item"], r["expected"], r["observed"], r["status"]] for r in d_rows])}

## 7. LUNG1-295 / LUNG1-415 Validation

{tbl(["patient_id", "check_item", "expected", "observed", "status"],
     [[r["patient_id"], r["check_item"], r["expected"], r["observed"], r["status"]] for r in e_rows])}

## 8. Full NPZ Path Stat Validation

{tbl(["check_item", "expected", "observed", "status", "note"],
     [[r["check_item"], r["expected"], r["observed"], r["status"], r["note"]] for r in f_rows])}

## 9. Sample NPZ Validation ({g_sample_count}개 샘플)

{tbl(["sample_id", "patient_id", "sampling_label", "check_item", "expected", "observed", "status"],
     [[r["sample_id"], r["patient_id"], r["sampling_label"],
       r["check_item"], r["expected"], r["observed"], r["status"]] for r in g_rows])}

## 10. Readiness 판정

**{readiness}**

blockers: {blocker_str}

## 11. 다음 단계

{next_step}

## 12. 금지 사항 준수

| 항목 | 준수 |
|------|------|
| no crop 생성 | True |
| no manifest 수정 | True |
| no model forward | True |
| no scoring | True |
| no metric 계산 | True |
| no threshold | True |
| no training | True |
| no checkpoint | True |
| npz 로드 {SAMPLE_LIMIT}개 이하 | True |
| 기존 Phase 6/7/8 output 미수정 | True |
| DIAG_CSV 미수정 | True |
| pip install 없음 | True |
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 8.2G Full Crop Output Validation")
    parser.add_argument("--run", action="store_true", help="실제 검증 실행 (없으면 dry-run)")
    args = parser.parse_args()

    if not args.run:
        dry_run_report()
        print("[DRY-RUN] 파일 생성 없음. 실행하려면 --run 을 사용하세요.")
        return

    output_guard()

    run_start = datetime.datetime.now()
    print("=" * 70)
    print("  Phase 8.2G Full Crop Output Validation")
    print(f"  시작: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    OUT_ROOT.mkdir(parents=True, exist_ok=False)
    print(f"[OK] output root 생성: {OUT_ROOT}")

    df_manifest = None
    if MANIFEST_PATH.exists():
        df_manifest = pd.read_csv(MANIFEST_PATH)
        print(f"[OK] manifest 로드: {len(df_manifest):,}행")

    phase8f_summary = {}
    if SUMMARY_JSON.exists():
        with open(SUMMARY_JSON, encoding="utf-8") as f:
            phase8f_summary = json.load(f)

    all_rows     = []
    all_blockers = []

    print("[INFO] Section A: artifact existence...")
    a_rows, a_bl = check_section_a()
    all_rows.extend(a_rows); all_blockers.extend(a_bl)
    print(f"  → {'PASS' if not a_bl else 'FAIL'} ({len(a_rows)} 항목)")

    print("[INFO] Section B: completion consistency...")
    b_rows, b_bl = check_section_b(df_manifest)
    all_rows.extend(b_rows); all_blockers.extend(b_bl)
    print(f"  → {'PASS' if not b_bl else 'FAIL'} ({len(b_rows)} 항목)")

    print("[INFO] Section C: manifest schema validation...")
    c_rows, c_bl = check_section_c(df_manifest)
    all_rows.extend(c_rows); all_blockers.extend(c_bl)
    print(f"  → {'PASS' if not c_bl else 'FAIL'} ({len(c_rows)} 항목)")

    print("[INFO] Section D: manifest content validation...")
    d_rows, d_bl = check_section_d(df_manifest)
    all_rows.extend(d_rows); all_blockers.extend(d_bl)
    print(f"  → {'PASS' if not d_bl else 'FAIL'} ({len(d_rows)} 항목)")

    print("[INFO] Section E: contamination patient validation...")
    e_rows, e_bl = check_section_e(df_manifest)
    all_rows.extend(e_rows); all_blockers.extend(e_bl)
    print(f"  → {'PASS' if not e_bl else 'FAIL'} ({len(e_rows)} 항목)")

    print("[INFO] Section F: full npz path stat validation...")
    f_rows, f_bl = check_section_f(df_manifest)
    all_rows.extend(f_rows); all_blockers.extend(f_bl)
    print(f"  → {'PASS' if not f_bl else 'FAIL'} ({len(f_rows)} 항목)")

    print(f"[INFO] Section G: sample npz validation ({SAMPLE_LIMIT}개)...")
    g_rows, g_bl = check_section_g(df_manifest)
    all_rows.extend(g_rows); all_blockers.extend(g_bl)
    print(f"  → {'PASS' if not g_bl else 'FAIL'} ({len(g_rows)} 항목)")

    unique_blockers = sorted(set(all_blockers))
    h_rows = build_section_h(unique_blockers)
    all_rows.extend(h_rows)

    readiness = (
        "READY_FOR_PHASE8_3_STAGE2_SCORING_SMOKE_PREFLIGHT"
        if not unique_blockers else "BLOCKED"
    )

    run_end = datetime.datetime.now()
    elapsed = (run_end - run_start).total_seconds()

    results = {
        "script":          "phase8_2g_full_crop_output_validation.py",
        "run_start":       run_start.isoformat(),
        "run_end":         run_end.isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "phase8f_summary": phase8f_summary,
        "input_paths": {
            "crop_root":    str(CROP_ROOT),
            "manifest":     str(MANIFEST_PATH),
            "summary_json": str(SUMMARY_JSON),
            "errors_csv":   str(ERRORS_CSV),
            "runtime_csv":  str(RUNTIME_CSV),
            "done_json":    str(DONE_JSON),
            "report_md":    str(REPORT_MD),
            "split_csv":    str(SPLIT_CSV),
        },
        "artifact_existence":               {"rows": a_rows, "pass": not a_bl},
        "completion_consistency":           {"rows": b_rows, "pass": not b_bl},
        "manifest_schema_validation":       {"rows": c_rows, "pass": not c_bl},
        "manifest_content_validation":      {"rows": d_rows, "pass": not d_bl},
        "contamination_patient_validation": {"rows": e_rows, "pass": not e_bl},
        "full_npz_path_stat_validation":    {"rows": f_rows, "pass": not f_bl},
        "sample_npz_validation":            {"rows": g_rows, "pass": not g_bl},
        "readiness_for_phase8_3":           readiness,
        "blockers":                         unique_blockers,
        "notes": {
            "validation_only":               True,
            "full_npz_stat_allowed":         True,
            "sample_npz_load_only":          True,
            "no_model_forward":              True,
            "no_scoring":                    True,
            "no_metric_calculation":         True,
            "no_threshold":                  True,
            "no_training":                   True,
            "no_existing_file_modification": True,
        },
    }

    for fp in [OUT_CSV, OUT_JSON, OUT_REPORT]:
        if fp.exists():
            print(f"[ABORT] 출력 파일 저장 직전 이미 존재: {fp}", file=sys.stderr)
            sys.exit(1)

    save_csv(all_rows, OUT_CSV)
    print(f"[OK] CSV 저장: {OUT_CSV}")

    save_json(results, OUT_JSON)
    print(f"[OK] JSON 저장: {OUT_JSON}")

    save_report(results, OUT_REPORT)
    print(f"[OK] MD report 저장: {OUT_REPORT}")

    print()
    print("=" * 70)
    print(f"  readiness : {readiness}")
    for b in unique_blockers:
        print(f"  blocker   : {b}")
    print(f"  소요 시간  : {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
