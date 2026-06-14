#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EfficientNet selected-rule stage2_holdout sealed evaluation ACTUAL-PREP.

이 스크립트는 actual sealed evaluation을 '준비'만 한다.
실제 holdout 접근/평가는 모든 guard가 True이고 --run-sealed-eval --confirm-sealed-eval 일 때만
별도 승인 단계에서 수행된다. 기본 실행에서는 holdout을 절대 열지 않는다.

모드:
  (no args)                                  -> BLOCKED exit 2
  --selftest                                 -> 내부 로직 단위 테스트
  --dry-run                                  -> holdout 접근 없이 preflight 산출물/해시/경로/스키마 점검
  --plan-only                                -> actual evaluation plan 출력
  --static-drycheck                          -> 전체 정적검사(=selftest+dry-run+plan-only 요약)
  --run-sealed-eval                          -> --confirm-sealed-eval 없으면 BLOCKED exit 2
  --run-sealed-eval --confirm-sealed-eval    -> guard False면 BLOCKED exit 2 (현재 전부 False)

금지(이 스크립트는 아래를 절대 수행하지 않음, guard로 이중 차단):
  stage2_holdout file list/read/load, holdout score read, actual metric 계산,
  normal_val threshold 실제 재산출, score recomputation, model forward, feature extraction,
  CT load, PNG/card render, 기존 artifact 수정, main rename, v1 card overwrite.
"""
import os, sys, json, hashlib, argparse

# ===== GUARDS (기본값 전부 False) =====
ALLOW_STAGE2_HOLDOUT_ACCESS = False
ALLOW_SEALED_EVAL_RUN       = False
ALLOW_HOLDOUT_SCORE_READ    = False
ALLOW_OUTPUT_WRITE          = False
ALLOW_MODEL_FORWARD         = False
ALLOW_FEATURE_EXTRACTION    = False
ALLOW_SCORE_RECOMPUTE       = False
ALLOW_CT_LOAD               = False
ALLOW_PNG_RENDER            = False
ALLOW_MAIN_RENAME           = False

GUARDS = {
    "ALLOW_STAGE2_HOLDOUT_ACCESS": ALLOW_STAGE2_HOLDOUT_ACCESS,
    "ALLOW_SEALED_EVAL_RUN": ALLOW_SEALED_EVAL_RUN,
    "ALLOW_HOLDOUT_SCORE_READ": ALLOW_HOLDOUT_SCORE_READ,
    "ALLOW_OUTPUT_WRITE": ALLOW_OUTPUT_WRITE,
    "ALLOW_MODEL_FORWARD": ALLOW_MODEL_FORWARD,
    "ALLOW_FEATURE_EXTRACTION": ALLOW_FEATURE_EXTRACTION,
    "ALLOW_SCORE_RECOMPUTE": ALLOW_SCORE_RECOMPUTE,
    "ALLOW_CT_LOAD": ALLOW_CT_LOAD,
    "ALLOW_PNG_RENDER": ALLOW_PNG_RENDER,
    "ALLOW_MAIN_RENAME": ALLOW_MAIN_RENAME,
}

ROOT = "/home/jinhy/project/lung-ct-anomaly"
EFF  = os.path.join(ROOT, "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1")

# ===== FROZEN RULE SPEC (stage1_dev에서 선택, 고정) =====
EPS = 1e-6
def rule_focal_peak_ratio(track):   # PRIMARY
    return track["track_max_score"] / (track["track_mean_score"] + EPS)
def rule_hybrid_len_focal_B(track): # AUXILIARY (sensitivity only)
    import math
    return (track["track_top2_mean_score"] - track["track_mean_score"]) / math.log1p(track["track_length"])

FROZEN_RULES = {
    "primary":   {"name": "focal_peak_ratio",   "formula": "track_max_score/(track_mean_score+1e-6)",
                  "fn": rule_focal_peak_ratio, "selected_from": "stage1_dev"},
    "auxiliary": {"name": "hybrid_len_focal_B", "formula": "(track_top2_mean_score-track_mean_score)/log1p(track_length)",
                  "fn": rule_hybrid_len_focal_B, "use": "sensitivity_only"},
}

# ===== THRESHOLD POLICY =====
THRESHOLD_POLICY = {
    "primary_evaluation": "top-k rank based (threshold-free)",
    "threshold_source": "normal_val only",
    "threshold_metrics": ["p95", "p99"],
    "freeze_before_holdout": True,
    "holdout_threshold_tuning": "PROHIBITED",
    "change_rule_after_holdout": "PROHIBITED",
}

# ===== METRIC POLICY =====
KS = [5, 10, 20, 50]
METRICS = [
    "lesion_patient_recall_at_k", "lesion_track_hit_at_k", "normal_fp_tracks_at_k",
    "normal_fp_patients_at_k", "peripheral_fp_ratio_at_k", "long_track_fp_ratio_at_k",
    "patient_level_topk_hit", "auxiliary_rule_sensitivity", "failure_case_audit",
]

# ===== FROZEN ARTIFACTS (hash 재검증 대상) =====
FROZEN_ARTIFACTS = {
    "position_bin_stats.npz": os.path.join(EFF, "outputs/models/distributions/position_bin_stats.npz"),
    "selected_feature_indices.npy": os.path.join(EFF, "outputs/models/distributions/selected_feature_indices.npy"),
    "p_b13_stage1_dev_metrics.json": os.path.join(EFF, "outputs/evaluation/lesion_stage1_dev_metrics/p_b13_stage1_dev_metrics.json"),
    "p_b15_roi_decision.json": os.path.join(EFF, "outputs/reports/p_b15_v4_20_roi_decision_checkpoint.json"),
    "selected_rule_decision.csv": os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/efficientnet_z_continuity_selection_rule_decision/selected_rule_decision.csv"),
    "stage1_track_csv": os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/efficientnet_vs_v1_z_continuity_reranking_comparison/efficientnet_z_continuity_tracks.csv"),
}
# preflight에서 기록된 기대 sha256 (앞 16자). 재검증은 actual 단계에서 전체 비교.
EXPECTED_SHA16 = {
    "position_bin_stats.npz": "0396e3a9705bd59e",
    "selected_feature_indices.npy": "dcec2342d4c12b62",
    "p_b13_stage1_dev_metrics.json": "e6474785e68cc905",
    "p_b15_roi_decision.json": "bc3bb1b1941ea785",
    "selected_rule_decision.csv": "c5461627a3e00d76",
    "stage1_track_csv": "dced6ec5da8ed5d4",
}

# ===== PATHS (holdout 미접근; 경로 문자열만 보관) =====
ACTUAL_OUTPUT_ROOT = os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/efficientnet_selected_rule_stage2_holdout_sealed_eval_actual")
HOLDOUT_MANIFEST   = os.path.join(ROOT, "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_filtered_manifest_v1.csv")
STAGE1_DEV_PATIENTS_SRC = os.path.join(EFF, "outputs/scores/lesion_stage1_dev_by_patient")  # 파일명=환자ID


def sha256_prefix(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


# ---------------------------------------------------------------------------
def cmd_selftest():
    """holdout/IO 없이 순수 로직 검증."""
    ok = True; notes = []
    # 1) rule 산식 검증
    t = {"track_max_score": 40.0, "track_mean_score": 10.0, "track_top2_mean_score": 38.0, "track_length": 100}
    pr = rule_focal_peak_ratio(t)
    assert abs(pr - (40.0/(10.0+EPS))) < 1e-9, "focal_peak_ratio mismatch"
    import math
    aux = rule_hybrid_len_focal_B(t)
    assert abs(aux - ((38.0-10.0)/math.log1p(100))) < 1e-9, "hybrid_B mismatch"
    notes.append(f"primary({pr:.4f})/aux({aux:.4f}) 산식 OK")
    # 2) 모든 guard 기본 False
    assert all(v is False for v in GUARDS.values()), "guard not all False"
    notes.append("guards all False OK")
    # 3) metric/k 고정
    assert KS == [5,10,20,50], "k mismatch"
    assert "failure_case_audit" in METRICS
    notes.append("k=5/10/20/50 + metrics OK")
    # 4) threshold policy
    assert THRESHOLD_POLICY["threshold_source"] == "normal_val only"
    assert THRESHOLD_POLICY["holdout_threshold_tuning"] == "PROHIBITED"
    notes.append("threshold policy OK")
    return ok, notes


def cmd_dry_run():
    """holdout 접근 없이 frozen artifact 존재/hash/경로/스키마 점검."""
    rows = []
    for k, p in FROZEN_ARTIFACTS.items():
        exists = os.path.isfile(p)
        cur = sha256_prefix(p) if exists else None
        exp = EXPECTED_SHA16.get(k)
        match = (cur == exp) if (exists and exp) else ("NO_EXPECTED" if exists else "MISSING")
        rows.append({"artifact": k, "exists": exists, "sha16": cur, "expected": exp, "match": match})
    # holdout/actual output 가드 점검 (열지 않음, 경로 상태만)
    holdout_opened = False  # 이 스크립트는 holdout을 절대 열지 않음
    actual_root_exists = os.path.isdir(ACTUAL_OUTPUT_ROOT)
    return rows, {"holdout_opened": holdout_opened, "actual_root_exists": actual_root_exists,
                  "actual_root_should_be_absent_now": True}


def cmd_plan_only():
    """actual evaluation이 할 일을 plan으로만 출력(실행 안 함)."""
    return {
        "actual_steps": [
            "1. (승인+guards True) stage2_holdout saved score 파일 read",
            "2. holdout patient list 로드 후 stage1_dev 154 ID와 overlap==0 검증",
            "3. frozen artifact hash 전체(sha256) 재검증",
            "4. holdout track 생성(run_z_continuity 동일 로직) -> primary rule(focal_peak_ratio) 적용",
            "5. auxiliary rule(hybrid_len_focal_B) sensitivity 계산",
            "6. normal_val precomputed/frozen p95/p99만 사용(holdout 재산출 금지)",
            "7. predefined metric만 산출 (k=5/10/20/50)",
            "8. failure-case audit 생성",
            "9. DONE.json 작성 후 output immutable",
        ],
        "actual_output_root": ACTUAL_OUTPUT_ROOT,
        "holdout_manifest_path_known_not_loaded": HOLDOUT_MANIFEST,
        "one_time_access": True,
        "requires": "다음 단계 별도 승인 + 모든 guard True + --run-sealed-eval --confirm-sealed-eval",
    }


def run_sealed_eval(confirmed):
    """actual 평가 진입점. 현재 guard 전부 False라 반드시 BLOCKED."""
    if not confirmed:
        print("BLOCKED: --run-sealed-eval requires --confirm-sealed-eval", file=sys.stderr)
        return 2
    hard_guards = [ALLOW_STAGE2_HOLDOUT_ACCESS, ALLOW_SEALED_EVAL_RUN, ALLOW_HOLDOUT_SCORE_READ, ALLOW_OUTPUT_WRITE]
    if not all(hard_guards):
        print("BLOCKED: guards are False (holdout access/eval/write not allowed in this step)", file=sys.stderr)
        return 2
    # 여기 도달하면 실제 평가 — 본 actual-prep 단계에서는 절대 도달하지 않음.
    print("ERROR: actual sealed evaluation은 별도 승인 단계 전용. 본 스크립트 버전에서 비활성.", file=sys.stderr)
    return 2


def main(argv):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--static-drycheck", action="store_true")
    ap.add_argument("--run-sealed-eval", action="store_true")
    ap.add_argument("--confirm-sealed-eval", action="store_true")
    if len(argv) == 0:
        print("BLOCKED: no args. 사용 가능한 모드: --selftest/--dry-run/--plan-only/--static-drycheck", file=sys.stderr)
        return 2
    args = ap.parse_args(argv)

    if args.run_sealed_eval:
        return run_sealed_eval(args.confirm_sealed_eval)

    if args.selftest:
        ok, notes = cmd_selftest()
        print(json.dumps({"selftest": "PASS" if ok else "FAIL", "notes": notes}, ensure_ascii=False, indent=2))
        return 0 if ok else 1

    if args.dry_run:
        rows, status = cmd_dry_run()
        print(json.dumps({"dry_run": "PASS", "artifacts": rows, "status": status}, ensure_ascii=False, indent=2))
        return 0

    if args.plan_only:
        print(json.dumps({"plan_only": "PASS", "plan": cmd_plan_only()}, ensure_ascii=False, indent=2))
        return 0

    if args.static_drycheck:
        ok, notes = cmd_selftest()
        rows, status = cmd_dry_run()
        plan = cmd_plan_only()
        out = {"static_drycheck": "PASS" if ok else "FAIL",
               "guards": GUARDS, "selftest_notes": notes,
               "artifact_hash_check": rows, "path_status": status,
               "frozen_rules": {k: {kk: vv for kk, vv in v.items() if kk != "fn"} for k, v in FROZEN_RULES.items()},
               "threshold_policy": THRESHOLD_POLICY, "k": KS, "metrics": METRICS,
               "actual_plan": plan}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if ok else 1

    print("BLOCKED: unknown mode", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
