"""
Phase 8.2F smoke output validation

목적: smoke v2 산출물이 full run 전 검증 기준을 만족하는지 확인
     validation only — full run / crop 생성 / scoring / metric / model forward 금지
"""

import sys
import json
import pathlib
import datetime
import argparse
import csv as csv_module

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# ── 입력 경로 ─────────────────────────────────────────────────────────────────
_SMOKE_V2_BASE = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/smoke"
    / "phase8_2f_stage2_dedicated_6ch_crop_generation_smoke_v2"
)
SMOKE_CROP_ROOT = _SMOKE_V2_BASE / "crops"
SMOKE_MANIFEST  = _SMOKE_V2_BASE / "s6a_stage2_holdout_filtered_manifest_smoke_v1.csv"

_SMOKE_V2_ANN = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_2f_stage2_dedicated_6ch_crop_generation_smoke_v2"
)
SMOKE_SUMMARY   = _SMOKE_V2_ANN / "phase8_2f_stage2_dedicated_6ch_crop_generation_smoke_v2_summary.json"
SMOKE_ERRORS    = _SMOKE_V2_ANN / "phase8_2f_stage2_dedicated_6ch_crop_generation_smoke_v2_errors.csv"
SMOKE_RUNTIME   = _SMOKE_V2_ANN / "phase8_2f_stage2_dedicated_6ch_crop_generation_smoke_v2_runtime_summary.csv"
SMOKE_REPORT_MD = _SMOKE_V2_ANN / "phase8_2f_stage2_dedicated_6ch_crop_generation_smoke_v2_report.md"

# ── full run 경로 (존재하면 안 됨) ─────────────────────────────────────────────
FULL_CROP_ROOT    = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_stage2_holdout_6ch_dedicated_v1"
FULL_MANIFEST     = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_filtered_manifest_v1.csv"
FULL_RUN_OUT_ROOT = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_2f_stage2_dedicated_6ch_crop_generation_v1"
)

# ── 출력 경로 ─────────────────────────────────────────────────────────────────
OUT_DIR  = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_2f_smoke_output_validation_v1"
)
OUT_CSV  = OUT_DIR / "phase8_2f_smoke_output_validation_v1.csv"
OUT_JSON = OUT_DIR / "phase8_2f_smoke_output_validation_v1.json"
OUT_MD   = OUT_DIR / "phase8_2f_smoke_output_validation_report_v1.md"

# ── 기대값 ──────────────────────────────────────────────────────────────────
EXP_TOTAL_SUCCESS       = 3381
EXP_TOTAL_ERROR         = 0
EXP_PATIENT_COUNT       = 3
EXP_POSITIVE_COUNT      = 1581
EXP_HARD_NEGATIVE_COUNT = 1800
EXP_NPZ_EXISTS          = 3381
EXP_NPZ_MISSING         = 0
EXP_NPZ_DUPLICATE       = 0
EXP_NPZ_EMPTY_NULL      = 0

MAX_SAMPLE_NPZ = 12
SAMPLE_SEED    = 42

CSV_FIELDNAMES = [
    "section", "item", "path", "exists",
    "check_item", "expected", "observed", "status", "note",
    "sample_id", "patient_id", "sampling_label", "npz_path",
    "blocker", "next_required_action",
]


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def row(**kwargs) -> dict:
    r = {f: "" for f in CSV_FIELDNAMES}
    r.update(kwargs)
    return r


def ps(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


# ── Section A: artifact existence ─────────────────────────────────────────────
def check_artifact_existence() -> tuple:
    artifacts = [
        ("smoke_crop_root",   SMOKE_CROP_ROOT),
        ("smoke_manifest",    SMOKE_MANIFEST),
        ("smoke_summary_json", SMOKE_SUMMARY),
        ("smoke_errors_csv",  SMOKE_ERRORS),
        ("smoke_runtime_csv", SMOKE_RUNTIME),
        ("smoke_report_md",   SMOKE_REPORT_MD),
    ]
    rows = []
    result = {}
    all_pass = True
    for item_name, p in artifacts:
        ex = p.exists()
        st = ps(ex)
        if not ex:
            all_pass = False
        rows.append(row(
            section="A_artifact_existence",
            item=item_name,
            path=str(p),
            exists=str(ex),
            status=st,
        ))
        result[item_name] = {"path": str(p), "exists": ex, "status": st}
    return rows, result, all_pass


# ── Section B: summary consistency ───────────────────────────────────────────
def check_summary_consistency() -> tuple:
    rows = []
    result = {}
    all_pass = True

    if not SMOKE_SUMMARY.exists():
        rows.append(row(
            section="B_summary_consistency",
            check_item="smoke_summary_json_exists",
            expected="True", observed="False",
            status="FAIL", note="smoke summary JSON 없어서 B 섹션 스킵",
        ))
        return rows, {"skipped": True}, False

    with open(SMOKE_SUMMARY, encoding="utf-8") as f:
        s = json.load(f)

    checks = [
        ("mode",                          "smoke",                  s.get("mode")),
        ("is_partial_run",                True,                     s.get("is_partial_run")),
        ("total_success",                 EXP_TOTAL_SUCCESS,        s.get("total_success")),
        ("total_error",                   EXP_TOTAL_ERROR,          s.get("total_error")),
        ("patient_count",                 EXP_PATIENT_COUNT,        s.get("patient_count")),
        ("positive_count",                EXP_POSITIVE_COUNT,       s.get("positive_count")),
        ("hard_negative_count",           EXP_HARD_NEGATIVE_COUNT,  s.get("hard_negative_count")),
        ("npz_path_exists_count",         EXP_NPZ_EXISTS,           s.get("npz_path_exists_count")),
        ("npz_path_missing_count",        EXP_NPZ_MISSING,          s.get("npz_path_missing_count")),
        ("npz_path_duplicate_count",      EXP_NPZ_DUPLICATE,        s.get("npz_path_duplicate_count")),
        ("npz_path_empty_null_count",     EXP_NPZ_EMPTY_NULL,       s.get("npz_path_empty_null_count")),
        ("post_validation.passed",        True,                     (s.get("post_validation") or {}).get("passed")),
        ("post_validation.contamination_check", "SKIPPED_PARTIAL_RUN",
         (s.get("post_validation") or {}).get("contamination_check")),
    ]

    for check_item, expected, observed in checks:
        match = (observed == expected)
        st = ps(match)
        if not match:
            all_pass = False
        rows.append(row(
            section="B_summary_consistency",
            check_item=check_item,
            expected=str(expected),
            observed=str(observed),
            status=st,
        ))
        result[check_item] = {"expected": expected, "observed": observed, "status": st}

    return rows, result, all_pass


# ── Section C: smoke manifest validation ──────────────────────────────────────
def check_manifest_validation() -> tuple:
    rows = []
    result = {}
    all_pass = True

    if not SMOKE_MANIFEST.exists():
        rows.append(row(
            section="C_manifest_validation",
            check_item="manifest_exists",
            expected="True", observed="False",
            status="FAIL", note="manifest 없어서 C 섹션 스킵",
        ))
        return rows, {"skipped": True}, False

    df = pd.read_csv(SMOKE_MANIFEST)

    # row count
    def _chk(check_item, expected, observed):
        match = (observed == expected)
        st = ps(match)
        if not match:
            nonlocal all_pass
            all_pass = False
        rows.append(row(
            section="C_manifest_validation",
            check_item=check_item,
            expected=str(expected),
            observed=str(observed),
            status=st,
        ))
        result[check_item] = {"expected": expected, "observed": observed, "status": st}

    _chk("row_count",        EXP_TOTAL_SUCCESS, len(df))
    _chk("patient_count",    EXP_PATIENT_COUNT, df["patient_id"].nunique())

    # stage_split unique
    ss_unique = set(df["stage_split"].unique().tolist())
    ss_ok = ss_unique == {"stage2_holdout"}
    if not ss_ok:
        all_pass = False
    rows.append(row(
        section="C_manifest_validation",
        check_item="stage_split_unique",
        expected="{stage2_holdout}",
        observed=str(ss_unique),
        status=ps(ss_ok),
    ))
    result["stage_split_unique"] = {"expected": "{stage2_holdout}", "observed": str(ss_unique), "status": ps(ss_ok)}

    # sampling_label unique ⊆ {positive, hard_negative}
    sl_unique = set(df["sampling_label"].unique().tolist())
    sl_ok = sl_unique <= {"positive", "hard_negative"}
    if not sl_ok:
        all_pass = False
    rows.append(row(
        section="C_manifest_validation",
        check_item="sampling_label_unique_subset",
        expected="{positive,hard_negative}",
        observed=str(sl_unique),
        status=ps(sl_ok),
    ))
    result["sampling_label_unique_subset"] = {"expected": "{positive,hard_negative}", "observed": str(sl_unique), "status": ps(sl_ok)}

    # label unique ⊆ {0,1}
    lbl_unique = set(df["label"].unique().tolist())
    lbl_ok = lbl_unique <= {0, 1}
    if not lbl_ok:
        all_pass = False
    rows.append(row(
        section="C_manifest_validation",
        check_item="label_unique_subset",
        expected="{0,1}",
        observed=str(lbl_unique),
        status=ps(lbl_ok),
    ))
    result["label_unique_subset"] = {"expected": "{0,1}", "observed": str(lbl_unique), "status": ps(lbl_ok)}

    _chk("positive_count",      EXP_POSITIVE_COUNT,      int((df["label"] == 1).sum()))
    _chk("hard_negative_count", EXP_HARD_NEGATIVE_COUNT, int((df["label"] == 0).sum()))

    # npz_path null/empty
    npz_null  = int(df["npz_path"].isna().sum())
    npz_blank = int((df["npz_path"].astype(str).str.strip() == "").sum())
    _chk("npz_path_null_empty", 0, npz_null + npz_blank)

    # npz_path duplicate
    _chk("npz_path_duplicate", 0, int(df["npz_path"].duplicated().sum()))

    # npz_path exists (실제 파일 존재)
    npz_exists_count = int(df["npz_path"].apply(lambda p: pathlib.Path(str(p)).exists()).sum())
    _chk("npz_path_exists", EXP_NPZ_EXISTS, npz_exists_count)

    # approval_required_before_scoring = True 전 행
    ap_norm = df["approval_required_before_scoring"].astype(str).str.strip().str.lower()
    ap_all_true = int((ap_norm.isin(["true", "1"])).sum())
    _chk("approval_required_before_scoring_all_true", EXP_TOTAL_SUCCESS, ap_all_true)

    # asset_scope 전 행
    as_count = int((df["asset_scope"].astype(str).str.strip() == "dedicated_stage2_holdout_6ch_crop").sum())
    _chk("asset_scope_all_correct", EXP_TOTAL_SUCCESS, as_count)

    # manifest_status 전 행
    ms_count = int((df["manifest_status"].astype(str).str.strip() == "created_after_phase8_2f_run").sum())
    _chk("manifest_status_all_correct", EXP_TOTAL_SUCCESS, ms_count)

    # crop_shape 전 행
    cs_count = int((df["crop_shape"].astype(str).str.strip() == "(6,96,96)").sum())
    _chk("crop_shape_all_correct", EXP_TOTAL_SUCCESS, cs_count)

    # input_channels 전 행
    ic_count = int((df["input_channels"].astype(str).str.strip() == "6").sum())
    _chk("input_channels_all_correct", EXP_TOTAL_SUCCESS, ic_count)

    # crop_size 전 행
    cz_count = int((df["crop_size"].astype(str).str.strip() == "96").sum())
    _chk("crop_size_all_correct", EXP_TOTAL_SUCCESS, cz_count)

    return rows, result, all_pass


# ── Section D: sample npz validation ─────────────────────────────────────────
def check_sample_npz() -> tuple:
    rows = []
    result = {"samples_loaded": 0, "all_passed": False, "details": []}
    all_pass = True

    if not SMOKE_MANIFEST.exists():
        rows.append(row(
            section="D_sample_npz_validation",
            check_item="manifest_exists",
            expected="True", observed="False",
            status="FAIL", note="manifest 없어서 D 섹션 스킵",
        ))
        return rows, result, False

    df = pd.read_csv(SMOKE_MANIFEST)

    rng = np.random.default_rng(SAMPLE_SEED)

    # 환자별 균등 샘플
    patients = sorted(df["patient_id"].unique().tolist())
    pos_per_patient = 6 // len(patients)
    hn_per_patient  = 6 // len(patients)
    pos_rem = 6 % len(patients)
    hn_rem  = 6 % len(patients)

    pos_samples = []
    hn_samples  = []
    for i, pid in enumerate(patients):
        sub = df[df["patient_id"] == pid]
        pos_sub = sub[sub["sampling_label"] == "positive"]
        hn_sub  = sub[sub["sampling_label"] == "hard_negative"]

        n_pos = pos_per_patient + (1 if i < pos_rem else 0)
        n_hn  = hn_per_patient  + (1 if i < hn_rem  else 0)

        if len(pos_sub) > 0:
            idx = rng.choice(len(pos_sub), size=min(n_pos, len(pos_sub)), replace=False)
            pos_samples.append(pos_sub.iloc[idx])
        if len(hn_sub) > 0:
            idx = rng.choice(len(hn_sub), size=min(n_hn, len(hn_sub)), replace=False)
            hn_samples.append(hn_sub.iloc[idx])

    pos_df = pd.concat(pos_samples, ignore_index=True).head(6) if pos_samples else pd.DataFrame()
    hn_df  = pd.concat(hn_samples,  ignore_index=True).head(6) if hn_samples  else pd.DataFrame()
    samples = pd.concat([pos_df, hn_df], ignore_index=True).head(MAX_SAMPLE_NPZ)

    result["samples_loaded"] = len(samples)
    samples_all_pass = True

    for i, (_, mrow) in enumerate(samples.iterrows()):
        sample_id   = f"sample_{i+1:02d}"
        pid         = str(mrow["patient_id"])
        slabel      = str(mrow["sampling_label"])
        npz_p       = str(mrow["npz_path"])
        m_label     = int(mrow["label"])
        m_slabel    = str(mrow["sampling_label"])
        m_pid       = str(mrow["patient_id"])
        m_lz        = int(mrow["local_z"]) if "local_z" in mrow and not pd.isna(mrow["local_z"]) else None

        detail = {
            "sample_id": sample_id,
            "patient_id": pid,
            "sampling_label": slabel,
            "npz_path": npz_p,
            "checks": [],
        }

        def npz_row(check_item, expected, observed, passed, note=""):
            nonlocal samples_all_pass, all_pass
            st = ps(passed)
            if not passed:
                samples_all_pass = False
                all_pass = False
            rows.append(row(
                section="D_sample_npz_validation",
                sample_id=sample_id,
                patient_id=pid,
                sampling_label=slabel,
                npz_path=npz_p,
                check_item=check_item,
                expected=str(expected),
                observed=str(observed),
                status=st,
                note=note,
            ))
            detail["checks"].append({
                "check_item": check_item,
                "expected": str(expected),
                "observed": str(observed),
                "status": st,
            })

        # npz 파일 존재 여부
        if not pathlib.Path(npz_p).exists():
            npz_row("npz_exists", True, False, False, "파일 없음 — 이후 체크 스킵")
            result["details"].append(detail)
            continue

        try:
            npz = np.load(npz_p)
        except Exception as e:
            npz_row("npz_loadable", True, False, False, str(e))
            result["details"].append(detail)
            continue

        # key "image" 존재
        has_image = "image" in npz
        npz_row("key_image_exists", True, has_image, has_image)

        if has_image:
            img = npz["image"]
            npz_row("image_shape",  "(6, 96, 96)", str(img.shape),   img.shape == (6, 96, 96))
            npz_row("image_dtype",  "float32계열", str(img.dtype),    np.issubdtype(img.dtype, np.floating))
            npz_row("image_min_ge0",   ">=0",      f"{img.min():.6f}", float(img.min()) >= 0.0)
            npz_row("image_max_le1",   "<=1",      f"{img.max():.6f}", float(img.max()) <= 1.0)
            npz_row("image_no_nan_inf","finite",   str(np.isfinite(img).all()), bool(np.isfinite(img).all()))

        # 정합성 체크 (key 있을 때만)
        if "label" in npz:
            obs_lbl = int(npz["label"])
            npz_row("label_match", m_label, obs_lbl, obs_lbl == m_label)
        if "sampling_label" in npz:
            obs_sl = str(npz["sampling_label"])
            npz_row("sampling_label_match", m_slabel, obs_sl, obs_sl == m_slabel)
        if "patient_id" in npz:
            obs_pid = str(npz["patient_id"])
            npz_row("patient_id_match", m_pid, obs_pid, obs_pid == m_pid)
        if "local_z" in npz and m_lz is not None:
            obs_lz = int(npz["local_z"])
            npz_row("local_z_match", m_lz, obs_lz, obs_lz == m_lz)

        npz.close()
        result["details"].append(detail)

    result["all_passed"] = samples_all_pass
    return rows, result, all_pass


# ── Section E: full run path safety ──────────────────────────────────────────
def check_full_run_path_safety() -> tuple:
    paths = [
        ("full_run_crop_root",    FULL_CROP_ROOT,    False),
        ("full_run_manifest",     FULL_MANIFEST,     False),
        ("full_run_out_root",     FULL_RUN_OUT_ROOT, False),
    ]
    rows = []
    result = {}
    all_pass = True
    for item_name, p, expected_exists in paths:
        actual_exists = p.exists()
        passed = (actual_exists == expected_exists)
        st = ps(passed)
        if not passed:
            all_pass = False
        rows.append(row(
            section="E_full_run_path_safety",
            item=item_name,
            path=str(p),
            expected="not_exists",
            observed="exists" if actual_exists else "not_exists",
            status=st,
        ))
        result[item_name] = {
            "path": str(p),
            "expected_exists": expected_exists,
            "actual_exists": actual_exists,
            "status": st,
        }
    return rows, result, all_pass


# ── Section F: readiness decision ─────────────────────────────────────────────
def make_readiness_decision(a_pass, b_pass, c_pass, d_pass, e_pass) -> tuple:
    blockers = []
    if not a_pass:
        blockers.append("BLOCKED_SMOKE_ARTIFACT_MISSING")
    if not b_pass:
        blockers.append("BLOCKED_SMOKE_SUMMARY_MISMATCH")
    if not c_pass:
        blockers.append("BLOCKED_SMOKE_MANIFEST_MISMATCH")
    if not d_pass:
        blockers.append("BLOCKED_SMOKE_NPZ_SAMPLE_INVALID")
    if not e_pass:
        blockers.append("BLOCKED_FULL_RUN_PATH_CONTAMINATION")

    if not blockers:
        readiness = "READY_FOR_PHASE8_2F_FULL_RUN_APPROVAL"
        next_action = "Phase 8.2F full run 승인 요청"
    else:
        readiness = blockers[0]
        next_action = f"blocker 해소 후 재검증: {', '.join(blockers)}"

    rows = [row(
        section="F_readiness_decision",
        item="overall",
        status=readiness,
        blocker="; ".join(blockers),
        next_required_action=next_action,
    )]
    return rows, readiness, blockers


# ── MD report ─────────────────────────────────────────────────────────────────
def write_md(
    readiness, blockers,
    a_res, b_res, c_res, d_res, e_res,
    created_at,
) -> None:
    def _tbl(items):
        lines = ["| 항목 | 기대 | 실제 | 상태 |", "|------|------|------|------|"]
        for k, v in items.items():
            if isinstance(v, dict):
                lines.append(
                    f"| {k} | {v.get('expected', '')} | {v.get('observed', v.get('actual_exists', v.get('exists', '')))} | {v.get('status', '')} |"
                )
        return "\n".join(lines)

    smoke_summary_ok = "통과" if not blockers else f"FAIL: {', '.join(blockers)}"

    a_stat = "PASS" if all(v["status"] == "PASS" for v in a_res.values() if isinstance(v, dict) and "status" in v) else "FAIL"
    b_stat = "PASS" if all(v["status"] == "PASS" for v in b_res.values() if isinstance(v, dict) and "status" in v) else "FAIL"
    c_stat = "PASS" if all(v["status"] == "PASS" for v in c_res.values() if isinstance(v, dict) and "status" in v) else "FAIL"
    d_stat = "PASS" if d_res.get("all_passed", False) else "FAIL"
    e_stat = "PASS" if all(v["status"] == "PASS" for v in e_res.values() if isinstance(v, dict) and "status" in v) else "FAIL"

    md = f"""# Phase 8.2F smoke output validation report

생성일: {created_at}

## 1. 목적

Phase 8.2F smoke v2 산출물이 full run 전 검증 기준을 만족하는지 확인한다.
validation only — full run / crop 생성 / scoring / metric / model forward 금지.

## 2. smoke v2 결과 요약

| 항목 | 값 |
|------|-----|
| mode | SMOKE |
| 처리 patient | 3 |
| 처리 row | 3,381 |
| success crop | 3,381 |
| error row | 0 |
| positive | 1,581 |
| hard_negative | 1,800 |
| validation.passed | True |
| contamination_check | SKIPPED_PARTIAL_RUN |

## 3. artifact existence — {a_stat}

| 항목 | 경로 | 존재 | 상태 |
|------|------|------|------|
{"".join(f"| {k} | {v['path']} | {v['exists']} | {v['status']} |" + chr(10) for k, v in a_res.items() if isinstance(v, dict))}

## 4. summary consistency — {b_stat}

{"(manifest JSON 로드 실패로 스킵)" if b_res.get("skipped") else _tbl(b_res)}

## 5. smoke manifest validation — {c_stat}

{"(manifest 로드 실패로 스킵)" if c_res.get("skipped") else _tbl(c_res)}

## 6. sample npz validation — {d_stat}

샘플 로드 수: {d_res.get("samples_loaded", 0)} / {MAX_SAMPLE_NPZ}
전체 통과: {d_res.get("all_passed", False)}

## 7. full run path safety — {e_stat}

| 항목 | 경로 | 기대 | 실제 | 상태 |
|------|------|------|------|------|
{"".join(f"| {k} | {v['path']} | not_exists | {'exists' if v['actual_exists'] else 'not_exists'} | {v['status']} |" + chr(10) for k, v in e_res.items() if isinstance(v, dict))}

## 8. readiness 판정

**{readiness}**

blockers: {', '.join(blockers) if blockers else '없음'}

## 9. 다음 단계

{"- Phase 8.2F full run 승인 요청" if not blockers else "- blocker 해소 후 smoke output validation 재실행"}

## 10. 금지 사항

- full run 실행 금지
- 추가 crop 생성 금지
- smoke crop/manifest 수정 금지
- model forward / scoring / metric / threshold / training 금지
- p95/p99 / hit-rate 계산 금지
- 샘플 12개 초과 npz 로드 금지
- pip/conda install 금지
"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)


# ── dry-run 보고 ──────────────────────────────────────────────────────────────
def dry_run_report() -> None:
    print()
    print("=" * 60)
    print("  [DRY-RUN] Phase 8.2F smoke output validation")
    print("=" * 60)
    print(f"  smoke crop root : {SMOKE_CROP_ROOT}")
    print(f"  smoke manifest  : {SMOKE_MANIFEST}")
    print(f"  smoke summary   : {SMOKE_SUMMARY}")
    print(f"  out dir         : {OUT_DIR}")
    print(f"  out CSV         : {OUT_CSV}")
    print(f"  out JSON        : {OUT_JSON}")
    print(f"  out MD          : {OUT_MD}")
    print()
    print("  output guard 확인:")
    print(f"    OUT_DIR 존재 : {OUT_DIR.exists()}")
    print(f"    OUT_CSV 존재 : {OUT_CSV.exists()}")
    print(f"    OUT_JSON 존재: {OUT_JSON.exists()}")
    print(f"    OUT_MD 존재  : {OUT_MD.exists()}")
    print()
    print("  실행 명령:")
    print("    source ~/ai_env/bin/activate && \\")
    print("    python scripts/phase8_2f_smoke_output_validation.py --run")
    print("=" * 60)
    print()


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8.2F smoke output validation")
    parser.add_argument("--run", action="store_true", help="실행 모드")
    args = parser.parse_args()

    if not args.run:
        dry_run_report()
        print("[DRY-RUN] 파일 생성 없음. 실행하려면 --run 을 사용하세요.")
        return

    # output guard
    for p, label in [(OUT_DIR, "OUT_DIR"), (OUT_CSV, "OUT_CSV"), (OUT_JSON, "OUT_JSON"), (OUT_MD, "OUT_MD")]:
        if p.exists():
            print(f"[ABORT] {label} 이미 존재: {p}", file=sys.stderr)
            sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    print(f"[OK] output dir 생성: {OUT_DIR}")

    created_at = datetime.datetime.now().isoformat()

    print("[INFO] Section A: artifact existence ...")
    a_rows, a_res, a_pass = check_artifact_existence()

    print("[INFO] Section B: summary consistency ...")
    b_rows, b_res, b_pass = check_summary_consistency()

    print("[INFO] Section C: smoke manifest validation ...")
    c_rows, c_res, c_pass = check_manifest_validation()

    print("[INFO] Section D: sample npz validation ...")
    d_rows, d_res, d_pass = check_sample_npz()

    print("[INFO] Section E: full run path safety ...")
    e_rows, e_res, e_pass = check_full_run_path_safety()

    print("[INFO] Section F: readiness decision ...")
    f_rows, readiness, blockers = make_readiness_decision(a_pass, b_pass, c_pass, d_pass, e_pass)

    all_rows = a_rows + b_rows + c_rows + d_rows + e_rows + f_rows

    # CSV 저장
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[OK] CSV 저장: {OUT_CSV}")

    # JSON 저장
    summary_json = {
        "script": "phase8_2f_smoke_output_validation.py",
        "created_at": created_at,
        "input_paths": {
            "smoke_crop_root": str(SMOKE_CROP_ROOT),
            "smoke_manifest":  str(SMOKE_MANIFEST),
            "smoke_summary":   str(SMOKE_SUMMARY),
            "smoke_errors":    str(SMOKE_ERRORS),
            "smoke_runtime":   str(SMOKE_RUNTIME),
            "smoke_report_md": str(SMOKE_REPORT_MD),
        },
        "artifact_existence":       a_res,
        "summary_consistency":      b_res,
        "smoke_manifest_validation": c_res,
        "sample_npz_validation":    d_res,
        "full_run_path_safety":     e_res,
        "readiness_for_full_run":   readiness,
        "blockers":                 blockers,
        "notes": {
            "smoke_validation_only":        True,
            "sample_npz_load_only":         True,
            "no_full_run":                  True,
            "no_model_forward":             True,
            "no_scoring":                   True,
            "no_metric_calculation":        True,
            "no_threshold":                 True,
            "no_training":                  True,
            "no_existing_file_modification": True,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    print(f"[OK] JSON 저장: {OUT_JSON}")

    # MD 저장
    write_md(readiness, blockers, a_res, b_res, c_res, d_res, e_res, created_at)
    print(f"[OK] MD 저장: {OUT_MD}")

    # 콘솔 요약
    print()
    print("=" * 60)
    print(f"  A artifact existence  : {'PASS' if a_pass else 'FAIL'}")
    print(f"  B summary consistency : {'PASS' if b_pass else 'FAIL'}")
    print(f"  C manifest validation : {'PASS' if c_pass else 'FAIL'}")
    print(f"  D sample npz          : {'PASS' if d_pass else 'FAIL'} (로드={d_res.get('samples_loaded',0)}개)")
    print(f"  E full run path safety: {'PASS' if e_pass else 'FAIL'}")
    print(f"  readiness             : {readiness}")
    if blockers:
        print(f"  blockers              : {', '.join(blockers)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
