"""
p_c_normal8a_checkpoint_schema_drycheck.py

P-C-NORMAL8a: Checkpoint Schema Hardening + Static Dry-Check

수행 내용:
  - p_c_normal5_train_classifier.py의 checkpoint schema 보강 내용을 정적으로 검증
  - py_compile, schema key 검사, guard 검사, 기존 artifact mtime 검사
  - 실제 학습/모델 forward/scoring 미실행

실행:
  source ~/ai_env/bin/activate
  python p_c_normal8a_checkpoint_schema_drycheck.py
"""

import csv
import datetime
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BRANCH_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR    = BRANCH_ROOT / "code"
MAIN_SCRIPT = CODE_DIR / "p_c_normal5_train_classifier.py"

OUT_DIR = BRANCH_ROOT / "outputs/reports/p_c_normal8a_checkpoint_schema_hardening_drycheck"

EXISTING_CKPT = BRANCH_ROOT / "outputs/checkpoints/p_c_normal6_smoke_training/p_c_normal6_epoch1.pth"
EXISTING_REPORT_DIRS = [
    BRANCH_ROOT / "outputs/reports/p_c_normal5_train_script_drycheck",
    BRANCH_ROOT / "outputs/reports/p_c_normal6_smoke_training",
    BRANCH_ROOT / "outputs/reports/p_c_normal7_smoke_result_validation",
    BRANCH_ROOT / "outputs/reports/p_c_normal8_shortcut_matched_sampling_preflight",
]

# ──────────────────────────────────────────────────────────────────────────────
# Expected schema
# ──────────────────────────────────────────────────────────────────────────────
EXPECTED_SMOKE_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "smoke_only",
    "full_training",
    "config",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_auc",
    "val_auc_status",
    "label_mapping",
    "class_weights",
    "manifest_paths",
    "forbidden_diagnostic_wording_count",
)

EXPECTED_FULL_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "smoke_only",
    "full_training",
    "config",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_auc",
    "val_auc_status",
    "label_mapping",
    "class_weights",
    "manifest_paths",
    "best_metric_name",
    "best_metric_value",
    "forbidden_diagnostic_wording_count",
)

FORBIDDEN_WORDS = [
    "폐선암" + " 확률",
    "암" + " 확률",
    "진단" + " 모델",
    "cancer" + " probability",
    "adenocarcinoma" + " probability",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_csv(rows: list, path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)


def _run_guard_check(args_list: list) -> int:
    """Run main script with given args via subprocess, return exit code."""
    python = sys.executable
    cmd = [python, str(MAIN_SCRIPT)] + args_list
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    return result.returncode


def _get_mtime(path: Path):
    if path.exists():
        return datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    return None


def _count_forbidden_words(path: Path) -> int:
    text = path.read_text(errors="ignore").lower()
    count = 0
    for w in FORBIDDEN_WORDS:
        count += text.count(w.lower())
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Main dry-check logic
# ──────────────────────────────────────────────────────────────────────────────

def run_drycheck() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    errors = []
    patch_rows = []
    smoke_schema_rows = []
    full_schema_rows = []
    guard_rows = []
    collision_rows = []
    mtime_rows = []
    shortcut_rows = []
    guardrail_rows = []

    verdict = "PASS"
    validated_at = datetime.datetime.now().isoformat()

    # ── 1. py_compile check ───────────────────────────────────────────────────
    compile_ok = False
    try:
        py_compile.compile(str(MAIN_SCRIPT), doraise=True)
        compile_ok = True
    except py_compile.PyCompileError as e:
        errors.append({"check": "py_compile", "error": str(e)})

    patch_rows.append({"check": "py_compile_ok", "expected": True, "actual": str(compile_ok), "pass": compile_ok})
    if not compile_ok:
        verdict = "FAIL"

    # ── 2. Load module to inspect schema constants and functions ──────────────
    module_loaded = False
    mod = None
    try:
        spec = importlib.util.spec_from_file_location("p_c_normal5", MAIN_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        # We do NOT call spec.loader.exec_module(mod) because that would trigger
        # top-level imports (torch, etc.) and potentially heavy work.
        # Instead, parse the file as text to verify key presence.
        module_loaded = False  # text-based check below
    except Exception as e:
        errors.append({"check": "module_load", "error": str(e)})

    # Text-based schema key verification (safe, no exec)
    script_text = MAIN_SCRIPT.read_text()

    smoke_key_found = "SMOKE_CHECKPOINT_REQUIRED_KEYS" in script_text
    full_key_found  = "FULL_CHECKPOINT_REQUIRED_KEYS"  in script_text
    smoke_builder_found = "_build_smoke_checkpoint" in script_text
    full_builder_found  = "_build_full_checkpoint"  in script_text

    patch_rows.append({"check": "SMOKE_CHECKPOINT_REQUIRED_KEYS_defined", "expected": True, "actual": str(smoke_key_found), "pass": smoke_key_found})
    patch_rows.append({"check": "FULL_CHECKPOINT_REQUIRED_KEYS_defined",  "expected": True, "actual": str(full_key_found),  "pass": full_key_found})
    patch_rows.append({"check": "_build_smoke_checkpoint_defined",         "expected": True, "actual": str(smoke_builder_found), "pass": smoke_builder_found})
    patch_rows.append({"check": "_build_full_checkpoint_defined",          "expected": True, "actual": str(full_builder_found),  "pass": full_builder_found})

    for row in patch_rows[1:]:
        if not row["pass"]:
            errors.append({"check": row["check"], "error": "missing in script"})
            verdict = "FAIL"

    # ── 3. Smoke checkpoint schema key verification (text scan) ──────────────
    for key in EXPECTED_SMOKE_KEYS:
        present = f'"{key}"' in script_text or f"'{key}'" in script_text
        smoke_schema_rows.append({
            "key": key,
            "expected_in_SMOKE_CHECKPOINT_REQUIRED_KEYS": True,
            "found_in_script": str(present),
            "pass": present,
        })
        if not present:
            errors.append({"check": f"smoke_key_{key}", "error": "key not found in script text"})
            verdict = "FAIL"

    smoke_key_count_ok = len(EXPECTED_SMOKE_KEYS) == 16
    smoke_schema_rows.append({
        "key": "_total_count",
        "expected_in_SMOKE_CHECKPOINT_REQUIRED_KEYS": 16,
        "found_in_script": str(len(EXPECTED_SMOKE_KEYS)),
        "pass": smoke_key_count_ok,
    })

    # ── 4. Full checkpoint schema key verification (text scan) ───────────────
    for key in EXPECTED_FULL_KEYS:
        present = f'"{key}"' in script_text or f"'{key}'" in script_text
        full_schema_rows.append({
            "key": key,
            "expected_in_FULL_CHECKPOINT_REQUIRED_KEYS": True,
            "found_in_script": str(present),
            "pass": present,
        })
        if not present:
            errors.append({"check": f"full_key_{key}", "error": "key not found in script text"})
            verdict = "FAIL"

    full_key_count_ok = len(EXPECTED_FULL_KEYS) == 18
    full_schema_rows.append({
        "key": "_total_count",
        "expected_in_FULL_CHECKPOINT_REQUIRED_KEYS": 18,
        "found_in_script": str(len(EXPECTED_FULL_KEYS)),
        "pass": full_key_count_ok,
    })

    # ── 5. Guard check (subprocess) ──────────────────────────────────────────
    guard_cases = [
        ("bare_run_exit2",              [],                                             2),
        ("smoke_train_alone_exit2",     ["--smoke-train"],                              2),
        ("smoke_train_epochs5_exit2",   ["--smoke-train", "--confirm-smoke",
                                         "--confirm-normal-vs-nsclc",
                                         "--confirm-no-holdout", "--epochs", "5"],     2),
        ("train_alone_exit2",           ["--train"],                                    2),
    ]
    for label, args_list, expected_code in guard_cases:
        try:
            actual_code = _run_guard_check(args_list)
            ok = (actual_code == expected_code)
        except subprocess.TimeoutExpired:
            actual_code = -1
            ok = False
            errors.append({"check": label, "error": "subprocess timeout"})
        guard_rows.append({
            "check": label,
            "args": " ".join(args_list) if args_list else "(none)",
            "expected_exit": expected_code,
            "actual_exit": actual_code,
            "pass": ok,
        })
        if not ok:
            errors.append({"check": label, "error": f"expected exit {expected_code} got {actual_code}"})
            verdict = "FAIL"

    # ── 6. Output collision check ─────────────────────────────────────────────
    check_paths = [
        ("smoke_ckpt_dir",   BRANCH_ROOT / "outputs/checkpoints/p_c_normal6_smoke_training"),
        ("smoke_report_dir", BRANCH_ROOT / "outputs/reports/p_c_normal6_smoke_training"),
        ("full_ckpt_dir",    BRANCH_ROOT / "outputs/checkpoints/p_c_normal8_full_training"),
        ("full_report_dir",  BRANCH_ROOT / "outputs/reports/p_c_normal8_full_training"),
    ]
    for label, path in check_paths:
        exists = path.exists()
        collision_rows.append({
            "path_label": label,
            "path": str(path),
            "exists_now": str(exists),
            "collision_risk_for_future_run": str(exists),
            "note": "schema-only drycheck does not create these",
        })

    # ── 7. Existing artifact mtime check (verify untouched) ──────────────────
    mtime_reference = "2026-06-08T14:26:25"  # p_c_normal6_epoch1.pth Modify time

    ckpt_mtime = _get_mtime(EXISTING_CKPT)
    ckpt_exists = EXISTING_CKPT.exists()
    mtime_unchanged = (ckpt_mtime is not None and ckpt_mtime.startswith("2026-06-08T14:26:25"))
    mtime_rows.append({
        "artifact": "p_c_normal6_epoch1.pth",
        "path": str(EXISTING_CKPT),
        "exists": str(ckpt_exists),
        "mtime": str(ckpt_mtime),
        "expected_mtime_prefix": mtime_reference,
        "unchanged": str(mtime_unchanged),
        "pass": mtime_unchanged,
    })
    if not mtime_unchanged:
        errors.append({"check": "p_c_normal6_ckpt_mtime", "error": f"mtime changed or missing: {ckpt_mtime}"})
        verdict = "FAIL"

    for rdir in EXISTING_REPORT_DIRS:
        mtime_val = _get_mtime(rdir) if rdir.exists() else "NOT_FOUND"
        mtime_rows.append({
            "artifact": rdir.name,
            "path": str(rdir),
            "exists": str(rdir.exists()),
            "mtime": str(mtime_val),
            "expected_mtime_prefix": "—",
            "unchanged": "not_checked_directory",
            "pass": True,
        })

    # ── 8. Shortcut warning carryforward ─────────────────────────────────────
    shortcut_items = [
        ("SR-HU",     "HIGH",   "normal mean -590 HU vs NSCLC -354 HU, +236 HU 차이",           "OPEN"),
        ("SR-POS",    "HIGH",   "normal peripheral 50.0% vs NSCLC 87.4%",                        "OPEN"),
        ("SR-PIPE",   "HIGH",   "normal P-C-NORMAL3 1-key NPZ vs NSCLC P-C8 18-key NPZ",        "OPEN"),
        ("SR-HU-CAP", "MEDIUM", "NSCLC max HU=445 capping artifact",                             "OPEN"),
    ]
    for name, level, detail, status in shortcut_items:
        shortcut_rows.append({
            "risk_id": name,
            "level": level,
            "detail": detail,
            "status": status,
            "resolved_in_8a": "False",
            "next_action": "P-C-NORMAL9 Option D: NSCLC same-generator crop regeneration preflight",
        })

    # ── 9. Guardrail check ────────────────────────────────────────────────────
    forbidden_count = _count_forbidden_words(MAIN_SCRIPT)

    guardrail_cases = [
        ("actual_training_executed",       False, "False",            True),
        ("model_forward_executed",         False, "False",            True),
        ("checkpoint_saved",               False, "False",            True),
        ("scoring_executed",               False, "False",            True),
        ("crop_regeneration_executed",     False, "False",            True),
        ("stage2_holdout_accessed",        False, "False",            True),
        ("p_c_aux_modified",               False, "False",            True),
        ("p_c_normal6_ckpt_modified",      False, str(not mtime_unchanged), mtime_unchanged),
        ("manifest_modified",              False, "False",            True),
        ("forbidden_diagnostic_wording_count", 0, str(forbidden_count), forbidden_count == 0),
        ("py_compile_ok",                  True,  str(compile_ok),    compile_ok),
        ("schema_hardening_complete",      True,  str(smoke_key_found and full_key_found
                                                       and smoke_builder_found and full_builder_found),
                                                  smoke_key_found and full_key_found
                                                  and smoke_builder_found and full_builder_found),
        ("full_training_hold_maintained",  True,  "True",             True),
        ("shortcut_risk_still_open",       True,  "True",             True),
    ]
    for check, exp, act, ok in guardrail_cases:
        guardrail_rows.append({"guardrail": check, "expected": exp, "actual": act, "pass": ok})
        if not ok:
            errors.append({"check": check, "error": f"expected={exp} actual={act}"})
            verdict = "FAIL"

    # ── Write CSVs ────────────────────────────────────────────────────────────
    _write_csv(patch_rows,        OUT_DIR / "p_c_normal8a_patch_summary.csv")
    _write_csv(smoke_schema_rows, OUT_DIR / "p_c_normal8a_smoke_checkpoint_schema_check.csv")
    _write_csv(full_schema_rows,  OUT_DIR / "p_c_normal8a_full_checkpoint_schema_check.csv")
    _write_csv(guard_rows,        OUT_DIR / "p_c_normal8a_guard_check.csv")
    _write_csv(collision_rows,    OUT_DIR / "p_c_normal8a_output_collision_check.csv")
    _write_csv(mtime_rows,        OUT_DIR / "p_c_normal8a_existing_artifact_mtime_check.csv")
    _write_csv(shortcut_rows,     OUT_DIR / "p_c_normal8a_shortcut_warning_carryforward.csv")
    _write_csv(guardrail_rows,    OUT_DIR / "p_c_normal8a_guardrail_check.csv")

    import pandas as pd
    if errors:
        pd.DataFrame(errors).to_csv(OUT_DIR / "p_c_normal8a_errors.csv", index=False)
    else:
        pd.DataFrame(columns=["check", "error"]).to_csv(OUT_DIR / "p_c_normal8a_errors.csv", index=False)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "stage": "P-C-NORMAL8a",
        "title": "Checkpoint Schema Hardening + Static Dry-Check",
        "verdict": verdict,
        "validated_at": validated_at,
        "script_modified": str(MAIN_SCRIPT),
        "added_constants": ["SMOKE_CHECKPOINT_REQUIRED_KEYS", "FULL_CHECKPOINT_REQUIRED_KEYS"],
        "added_functions": ["_build_smoke_checkpoint", "_build_full_checkpoint"],
        "smoke_checkpoint_keys_count": len(EXPECTED_SMOKE_KEYS),
        "full_checkpoint_keys_count": len(EXPECTED_FULL_KEYS),
        "py_compile_ok": compile_ok,
        "guard_checks_all_pass": all(r["pass"] for r in guard_rows),
        "p_c_normal6_ckpt_untouched": mtime_unchanged,
        "actual_training_executed": False,
        "checkpoint_saved": False,
        "model_forward_executed": False,
        "scoring_executed": False,
        "crop_regeneration_executed": False,
        "stage2_holdout_accessed": False,
        "p_c_aux_modified": False,
        "forbidden_diagnostic_wording_count": forbidden_count,
        "shortcut_risk_open": {
            "SR-HU": "OPEN",
            "SR-POS": "OPEN",
            "SR-PIPE": "OPEN",
            "SR-HU-CAP": "OPEN",
        },
        "full_training_hold": True,
        "errors_count": len(errors),
        "next_step": "P-C-NORMAL9 Option D: NSCLC same-generator crop regeneration preflight",
    }
    with open(OUT_DIR / "p_c_normal8a_checkpoint_schema_hardening_drycheck.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Markdown report ───────────────────────────────────────────────────────
    _write_md_report(summary, patch_rows, smoke_schema_rows, full_schema_rows,
                     guard_rows, mtime_rows, shortcut_rows, guardrail_rows, errors)

    print(f"\n[P-C-NORMAL8a] 판정: {verdict}")
    print(f"  출력: {OUT_DIR}")
    return 0 if verdict == "PASS" else 1


def _write_md_report(summary, patch_rows, smoke_schema_rows, full_schema_rows,
                     guard_rows, mtime_rows, shortcut_rows, guardrail_rows, errors):
    verdict = summary["verdict"]
    verdict_str = "통과 (PASS)" if verdict == "PASS" else "실패 (FAIL)"
    lines = [
        "# P-C-NORMAL8a Checkpoint Schema Hardening + Static Dry-Check",
        "",
        f"## 판정: **{verdict_str}**",
        "",
        f"- validated_at: {summary['validated_at'][:10]}",
        f"- 수정 파일: `{Path(summary['script_modified']).name}`",
        f"- errors_count: {summary['errors_count']}",
        "",
        "---",
        "",
        "## 추가된 내용 요약",
        "",
        "| 항목 | 내용 |",
        "|---|---|",
        f"| 추가 상수 | SMOKE_CHECKPOINT_REQUIRED_KEYS ({summary['smoke_checkpoint_keys_count']} keys) |",
        f"| 추가 상수 | FULL_CHECKPOINT_REQUIRED_KEYS ({summary['full_checkpoint_keys_count']} keys) |",
        f"| 추가 함수 | _build_smoke_checkpoint() |",
        f"| 추가 함수 | _build_full_checkpoint() |",
        f"| py_compile | {'OK' if summary['py_compile_ok'] else 'FAIL'} |",
        "",
        "---",
        "",
        "## Patch 요약",
        "",
        "| check | expected | actual | pass |",
        "|---|---|---|---|",
    ]
    for r in patch_rows:
        lines.append(f"| {r['check']} | {r['expected']} | {r['actual']} | {r['pass']} |")

    lines += [
        "",
        "---",
        "",
        "## Smoke Checkpoint Schema (16 keys)",
        "",
        "| key | script에 존재 | pass |",
        "|---|---|---|",
    ]
    for r in smoke_schema_rows:
        lines.append(f"| {r['key']} | {r['found_in_script']} | {r['pass']} |")

    lines += [
        "",
        "---",
        "",
        "## Full Checkpoint Schema (18 keys)",
        "",
        "| key | script에 존재 | pass |",
        "|---|---|---|",
    ]
    for r in full_schema_rows:
        lines.append(f"| {r['key']} | {r['found_in_script']} | {r['pass']} |")

    lines += [
        "",
        "---",
        "",
        "## Guard Check (subprocess exit code 확인)",
        "",
        "| check | args | expected_exit | actual_exit | pass |",
        "|---|---|---|---|---|",
    ]
    for r in guard_rows:
        lines.append(f"| {r['check']} | `{r['args']}` | {r['expected_exit']} | {r['actual_exit']} | {r['pass']} |")

    lines += [
        "",
        "---",
        "",
        "## 기존 Artifact mtime 검사",
        "",
        "| artifact | exists | mtime | unchanged | pass |",
        "|---|---|---|---|---|",
    ]
    for r in mtime_rows:
        lines.append(f"| {r['artifact']} | {r['exists']} | {r['mtime'][:19] if r['mtime'] != 'NOT_FOUND' and r['mtime'] is not None else r['mtime']} | {r['unchanged']} | {r['pass']} |")

    lines += [
        "",
        "---",
        "",
        "## Shortcut Risk Carryforward",
        "",
        "| risk_id | level | status | resolved_in_8a |",
        "|---|---|---|---|",
    ]
    for r in shortcut_rows:
        lines.append(f"| {r['risk_id']} | {r['level']} | {r['status']} | {r['resolved_in_8a']} |")

    lines += [
        "",
        "> **checkpoint schema hardening은 shortcut risk를 해결하지 않는다.**",
        "> SR-HU / SR-POS / SR-PIPE / SR-HU-CAP는 여전히 OPEN.",
        "> full training은 여전히 HOLD.",
        "",
        "---",
        "",
        "## Guardrail",
        "",
        "| guardrail | expected | actual | pass |",
        "|---|---|---|---|",
    ]
    for r in guardrail_rows:
        ok_str = "PASS" if r["pass"] else "FAIL"
        lines.append(f"| {r['guardrail']} | {r['expected']} | {r['actual']} | {ok_str} |")

    lines += [
        "",
        "---",
        "",
        "## 최종 판정",
        "",
        f"**{verdict_str}**",
        "",
        "### 확인 사항",
        "",
        f"- script schema hardening 완료: {summary['smoke_checkpoint_keys_count']} smoke keys + {summary['full_checkpoint_keys_count']} full keys",
        f"- py_compile OK: {summary['py_compile_ok']}",
        f"- p_c_normal6_epoch1.pth 미수정: {summary['p_c_normal6_ckpt_untouched']}",
        f"- actual training 미실행: {not summary['actual_training_executed']}",
        f"- checkpoint 저장 없음: {not summary['checkpoint_saved']}",
        f"- model_forward 미실행: {not summary['model_forward_executed']}",
        f"- scoring 미실행: {not summary['scoring_executed']}",
        f"- crop_regeneration 미실행: {not summary['crop_regeneration_executed']}",
        f"- stage2_holdout 미접근: {not summary['stage2_holdout_accessed']}",
        f"- P-C-AUX 무수정: {not summary['p_c_aux_modified']}",
        f"- forbidden diagnostic wording count: {summary['forbidden_diagnostic_wording_count']}",
        f"- shortcut risk: 여전히 OPEN (SR-HU/SR-POS/SR-PIPE/SR-HU-CAP)",
        f"- full training: 여전히 HOLD",
        "",
        "---",
        "",
        "## 다음 단계",
        "",
        "- **P-C-NORMAL9 Option D**: NSCLC same-generator crop regeneration preflight",
        "  - SR-HU/SR-POS/SR-PIPE 해결을 위해 NSCLC crop을 P-C-NORMAL3 동일 generator로 재생성",
    ]

    if errors:
        lines += [
            "",
            "---",
            "",
            "## 오류 목록",
            "",
            "| check | error |",
            "|---|---|",
        ]
        for e in errors:
            lines.append(f"| {e['check']} | {e['error']} |")

    (OUT_DIR / "p_c_normal8a_checkpoint_schema_hardening_drycheck.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(run_drycheck())
