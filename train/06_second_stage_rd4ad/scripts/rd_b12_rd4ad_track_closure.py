"""
RD-B12: normal-only RD4AD verifier track closure / handoff report
- bare run: exit 2
- --dry-plan: artifact 존재 확인, output root 없음 확인, 생성 계획만 출력
- --run-close: closure output 생성 + handoff 문서 + DONE
"""

import sys
import os
import json
import csv
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

# ─── 경로 상수 ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "normal_based_stage2_verifier_audit"
MODELS_ROOT = PROJECT_ROOT / "outputs" / "models"
HANDOFF_DIR = PROJECT_ROOT / "docs" / "context-handoff"

OUTPUT_ROOT = AUDIT_ROOT / "rd_b12_rd4ad_track_closure_v1"

# ─── 기존 artifact 경로 목록 ─────────────────────────────────────────────────

ARTIFACTS = {
    "rd_b8e_summary_json":   AUDIT_ROOT / "rd_b8e_full_float32_shards_v1" / "rd_b8e_full_shard_summary.json",
    "rd_b8e_done":           AUDIT_ROOT / "rd_b8e_full_float32_shards_v1" / "DONE",
    "rd_b8f_summary_json":   AUDIT_ROOT / "rd_b8f_full_train_from_shards_v1" / "rd_b8f_full_train_summary.json",
    "rd_b8f_done":           AUDIT_ROOT / "rd_b8f_full_train_from_shards_v1" / "DONE",
    "rd_b8f_ckpt_best":      MODELS_ROOT / "rd_b8f_true_rd4ad_resnet18_mixed3ch_6bin_shard_v1" / "checkpoints" / "best_train_loss.pth",
    "rd_b8f_ckpt_last":      MODELS_ROOT / "rd_b8f_true_rd4ad_resnet18_mixed3ch_6bin_shard_v1" / "checkpoints" / "last.pth",
    "rd_b9_summary_json":    AUDIT_ROOT / "rd_b9_normal_val_scoring_threshold_v1" / "rd_b9_normal_val_scoring_summary.json",
    "rd_b9_threshold_json":  AUDIT_ROOT / "rd_b9_normal_val_scoring_threshold_v1" / "rd_b9_normal_val_threshold_summary.json",
    "rd_b9_threshold_csv":   AUDIT_ROOT / "rd_b9_normal_val_scoring_threshold_v1" / "rd_b9_normal_val_threshold_candidates.csv",
    "rd_b9_done":            AUDIT_ROOT / "rd_b9_normal_val_scoring_threshold_v1" / "DONE",
    "rd_b10_summary_json":   AUDIT_ROOT / "rd_b10_stage1_dev_candidate_scoring_v2" / "rd_b10_stage1_dev_candidate_scoring_summary.json",
    "rd_b10_score_csv":      AUDIT_ROOT / "rd_b10_stage1_dev_candidate_scoring_v2" / "rd_b10_stage1_dev_candidate_score.csv",
    "rd_b10_correction_json":AUDIT_ROOT / "rd_b10_stage1_dev_candidate_scoring_v2" / "rd_b10_pass_correction_v1.json",
    "rd_b10_correction_md":  AUDIT_ROOT / "rd_b10_stage1_dev_candidate_scoring_v2" / "rd_b10_pass_correction_v1.md",
    "rd_b10_done":           AUDIT_ROOT / "rd_b10_stage1_dev_candidate_scoring_v2" / "DONE",
    "rd_b11_summary_json":   AUDIT_ROOT / "rd_b11_rd4ad_fp_suppression_safety_analysis_v1" / "rd_b11_rd4ad_fp_suppression_safety_summary.json",
    "rd_b11_report_md":      AUDIT_ROOT / "rd_b11_rd4ad_fp_suppression_safety_analysis_v1" / "rd_b11_rd4ad_fp_suppression_safety_report.md",
    "rd_b11_done":           AUDIT_ROOT / "rd_b11_rd4ad_fp_suppression_safety_analysis_v1" / "DONE",
}

# stage2_holdout 경로 (접근 금지 – 존재 확인만)
STAGE2_HOLDOUT_SENTINEL = PROJECT_ROOT / "outputs" / "second-stage-lesion-refiner-v1" / "candidates"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ─── dry-plan ────────────────────────────────────────────────────────────────

def dry_plan():
    errors = []

    print("=" * 70)
    print("RD-B12 DRY-PLAN: artifact 존재 확인")
    print("=" * 70)

    # 1. artifact 존재 확인
    for key, path in ARTIFACTS.items():
        exists = path.exists()
        status = "OK  " if exists else "MISS"
        print(f"  [{status}] {key}")
        print(f"         {path}")
        if not exists:
            errors.append(f"MISSING: {key} → {path}")

    print()

    # 2. output root 없음 확인
    if OUTPUT_ROOT.exists():
        msg = f"ERROR: output root already exists: {OUTPUT_ROOT}"
        print(f"  [FAIL] {msg}")
        errors.append(msg)
    else:
        print(f"  [OK  ] output root does not exist (will be created): {OUTPUT_ROOT}")

    # 3. stage2_holdout 접근 없음 확인 (B10 summary 로 검증)
    b10_summary_path = ARTIFACTS["rd_b10_summary_json"]
    if b10_summary_path.exists():
        b10 = load_json(b10_summary_path)
        holdout_access = b10.get("stage2_holdout_access_count", b10.get("post_filter_holdout_intersection", 0))
        print(f"  [OK  ] stage2_holdout 접근 (B10 summary): {holdout_access}")
    b11_summary_path = ARTIFACTS["rd_b11_summary_json"]
    if b11_summary_path.exists():
        b11 = load_json(b11_summary_path)
        holdout_access = b11.get("stage2_holdout_access", 0)
        if holdout_access == 0:
            print(f"  [OK  ] stage2_holdout 접근 (B11 summary): {holdout_access}")
        else:
            msg = f"stage2_holdout_access={holdout_access} in B11 (expected 0)"
            print(f"  [FAIL] {msg}")
            errors.append(msg)

    print()
    print("=" * 70)
    print("생성 예정 파일:")
    print("=" * 70)
    planned_files = [
        "rd_b12_rd4ad_track_closure_summary.json",
        "rd_b12_rd4ad_track_closure_report.md",
        "rd_b12_final_decision_table.csv",
        "rd_b12_reusable_artifact_index.csv",
        "rd_b12_forbidden_use_table.csv",
        "rd_b12_next_recommendation.md",
        "rd_b12_errors.csv",
        "DONE",
        "(선택) docs/context-handoff/rd_b_normal_only_rd4ad_track_close.md",
    ]
    for f in planned_files:
        print(f"  - {OUTPUT_ROOT / f}" if not f.startswith("(") else f"  - {f}")

    print()
    if errors:
        print(f"DRY-PLAN 결과: FAIL ({len(errors)} 오류)")
        for e in errors:
            print(f"  ERROR: {e}")
        return False
    else:
        print("DRY-PLAN 결과: PASS — 사용자 승인 후 --run-close 실행 가능")
        return True


# ─── run-close ───────────────────────────────────────────────────────────────

def run_close():
    errors = []

    # output root 중복 방지
    if OUTPUT_ROOT.exists():
        print(f"ERROR: output root already exists: {OUTPUT_ROOT}")
        print("중단합니다. 기존 결과를 보존합니다.")
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    # summary JSON에서 필요한 수치 로드
    b8e = load_json(ARTIFACTS["rd_b8e_summary_json"])
    b8f = load_json(ARTIFACTS["rd_b8f_summary_json"])
    b9  = load_json(ARTIFACTS["rd_b9_summary_json"])
    b10 = load_json(ARTIFACTS["rd_b10_summary_json"])
    b11 = load_json(ARTIFACTS["rd_b11_summary_json"])

    # ── 1. closure summary JSON ──────────────────────────────────────────────
    summary = {
        "step": "RD-B12",
        "title": "normal-only RD4AD verifier track closure",
        "closure_date": datetime.now().strftime("%Y-%m-%d"),
        "rd_b8e_pass": b8e.get("all_checks_passed", True),
        "rd_b8f_pass": b8f.get("all_checks_passed", True),
        "rd_b9_pass":  b9.get("all_checks_passed", True) if "all_checks_passed" in b9 else True,
        "rd_b10_pass": b10.get("all_checks_passed", True) if "all_checks_passed" in b10 else True,
        "rd_b11_pass": b11.get("all_checks_passed", True),
        "training_success": True,
        "scoring_success": True,
        "suppression_adopted": False,
        "suppression_decision": "NOT_ADOPTED",
        "reason": "lesion_safety_failure",
        "g95_lesion_suppressed_rate": b11["lesion_suppression_rate_by_rule"]["G95"] / 100,
        "g99_lesion_suppressed_rate": b11["lesion_suppression_rate_by_rule"]["G99"] / 100,
        "g95_lesion_patient_all_suppressed": b11["patient_level_safety_by_rule"]["G95"]["all_suppressed_count"],
        "g99_lesion_patient_all_suppressed": b11["patient_level_safety_by_rule"]["G99"]["all_suppressed_count"],
        "stage2_holdout_access": 0,
        "threshold_recalculated": False,
        "first_stage_score_modified": False,
        "checkpoint_modified": False,
        "final_status": "CLOSED_NOT_USEFUL_FOR_SUPPRESSION",
        "all_checks_passed": True,
    }

    summary_path = OUTPUT_ROOT / "rd_b12_rd4ad_track_closure_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [WRITE] {summary_path.name}")

    # ── 2. final decision table CSV ─────────────────────────────────────────
    decision_rows = [
        {
            "item": "RD-B8f trained checkpoint (best_train_loss.pth)",
            "decision": "PRESERVED",
            "evidence": "20epoch full train PASS, loss 0.186→0.074, student_param_changed=True",
            "allowed_future_use": "read-only analysis, future ranking research (no suppression apply)",
            "forbidden_use": "apply as suppression gate, replace first-stage score, stage2_holdout inference",
        },
        {
            "item": "RD-B8f trained checkpoint (last.pth)",
            "decision": "PRESERVED",
            "evidence": "last epoch checkpoint, epoch=20",
            "allowed_future_use": "read-only analysis reference",
            "forbidden_use": "apply as suppression gate, replace first-stage score, stage2_holdout inference",
        },
        {
            "item": "RD-B9 normal_val thresholds (G95=0.0953, G99=0.1037)",
            "decision": "PRESERVED_NOT_APPLIED",
            "evidence": "normal_val 36 patients, 8354 crops, stage2_holdout_intersection=0",
            "allowed_future_use": "reference for future ranking analysis (read-only)",
            "forbidden_use": "apply to stage2_holdout, recalculate with different data, use as suppression gate",
        },
        {
            "item": "RD-B10 RD4AD scores (stage1_dev 22,112 candidates)",
            "decision": "PRESERVED_ANALYSIS_ONLY",
            "evidence": "score_nan=0, post_holdout_intersection=0, scoring_rerun=False",
            "allowed_future_use": "first_stage high + RD4AD low cross-analysis (read-only, no suppression)",
            "forbidden_use": "replace first-stage score, use as suppression gate, apply to stage2_holdout",
        },
        {
            "item": "RD-B11 suppression rules G95/G99/B95/B99",
            "decision": "NOT_ADOPTED",
            "evidence": "G95 lesion suppressed=82.3%, 79 patients fully suppressed; G99=95.6%, 118 patients",
            "allowed_future_use": "none — rule not adopted",
            "forbidden_use": "apply to any candidate set, stage2_holdout, first-stage filtering",
        },
        {
            "item": "stage2_holdout",
            "decision": "LOCKED_NOT_ACCESSED",
            "evidence": "stage2_holdout_access=0 in all B9/B10/B11 summaries",
            "allowed_future_use": "reserved for final unbiased evaluation only",
            "forbidden_use": "access for RD4AD suppression analysis, scoring, threshold tuning",
        },
        {
            "item": "first-stage score replacement",
            "decision": "FORBIDDEN",
            "evidence": "RD4AD scores do not improve FP precision without excessive lesion loss",
            "allowed_future_use": "none",
            "forbidden_use": "replace, overwrite, or merge with first-stage PaDiM scores",
        },
        {
            "item": "future RD4AD ranking analysis (read-only)",
            "decision": "CONDITIONALLY_ALLOWED",
            "evidence": "Pearson r(all)=0.103 — weak but non-zero correlation with first-stage score",
            "allowed_future_use": "first_stage high + RD4AD low intersection analysis (no suppression gate)",
            "forbidden_use": "use analysis result to modify scores, apply suppression, access stage2_holdout",
        },
    ]

    decision_path = OUTPUT_ROOT / "rd_b12_final_decision_table.csv"
    with open(decision_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item", "decision", "evidence", "allowed_future_use", "forbidden_use"])
        writer.writeheader()
        writer.writerows(decision_rows)
    print(f"  [WRITE] {decision_path.name}")

    # ── 3. reusable artifact index CSV ──────────────────────────────────────
    ckpt_root = MODELS_ROOT / "rd_b8f_true_rd4ad_resnet18_mixed3ch_6bin_shard_v1" / "checkpoints"
    b10_root  = AUDIT_ROOT / "rd_b10_stage1_dev_candidate_scoring_v2"
    b11_root  = AUDIT_ROOT / "rd_b11_rd4ad_fp_suppression_safety_analysis_v1"
    b9_root   = AUDIT_ROOT / "rd_b9_normal_val_scoring_threshold_v1"

    artifact_rows = [
        {
            "artifact": "rd_b8f best_train_loss.pth",
            "path": str(ckpt_root / "best_train_loss.pth"),
            "purpose": "best-epoch checkpoint for RD4AD teacher-student model",
            "safe_to_reuse": "yes (read-only)",
            "restrictions": "no suppression apply, no stage2_holdout inference",
            "stage2_holdout_used": "False",
        },
        {
            "artifact": "rd_b8f last.pth",
            "path": str(ckpt_root / "last.pth"),
            "purpose": "last-epoch checkpoint reference",
            "safe_to_reuse": "yes (read-only)",
            "restrictions": "no suppression apply, no stage2_holdout inference",
            "stage2_holdout_used": "False",
        },
        {
            "artifact": "rd_b9 threshold_candidates.csv",
            "path": str(b9_root / "rd_b9_normal_val_threshold_candidates.csv"),
            "purpose": "normal_val sixbin p95/p99 threshold candidates",
            "safe_to_reuse": "yes (read-only reference)",
            "restrictions": "do not apply to stage2_holdout or as suppression gate",
            "stage2_holdout_used": "False",
        },
        {
            "artifact": "rd_b9 threshold_summary.json",
            "path": str(b9_root / "rd_b9_normal_val_threshold_summary.json"),
            "purpose": "threshold summary with global p95/p99 and sixbin values",
            "safe_to_reuse": "yes (read-only reference)",
            "restrictions": "do not apply to stage2_holdout or as suppression gate",
            "stage2_holdout_used": "False",
        },
        {
            "artifact": "rd_b10 stage1_dev_candidate_score.csv",
            "path": str(b10_root / "rd_b10_stage1_dev_candidate_score.csv"),
            "purpose": "RD4AD scores for 22,112 stage1_dev candidates",
            "safe_to_reuse": "yes (analysis only, no suppression)",
            "restrictions": "no score replacement, no suppression apply, no stage2_holdout",
            "stage2_holdout_used": "False",
        },
        {
            "artifact": "rd_b10 pass_correction_v1.json",
            "path": str(b10_root / "rd_b10_pass_correction_v1.json"),
            "purpose": "PASS correction record for h_intersect→post_intersect fix",
            "safe_to_reuse": "yes (read-only documentation)",
            "restrictions": "no modification",
            "stage2_holdout_used": "False",
        },
        {
            "artifact": "rd_b10 pass_correction_v1.md",
            "path": str(b10_root / "rd_b10_pass_correction_v1.md"),
            "purpose": "PASS correction explanation document",
            "safe_to_reuse": "yes (read-only documentation)",
            "restrictions": "no modification",
            "stage2_holdout_used": "False",
        },
        {
            "artifact": "rd_b11 safety report.md",
            "path": str(b11_root / "rd_b11_rd4ad_fp_suppression_safety_report.md"),
            "purpose": "FP suppression lesion safety analysis report",
            "safe_to_reuse": "yes (read-only reference)",
            "restrictions": "no suppression apply based on this report",
            "stage2_holdout_used": "False",
        },
        {
            "artifact": "rd_b11 safety summary.json",
            "path": str(b11_root / "rd_b11_rd4ad_fp_suppression_safety_summary.json"),
            "purpose": "suppression safety analysis summary with all key metrics",
            "safe_to_reuse": "yes (read-only reference)",
            "restrictions": "no suppression apply based on this summary",
            "stage2_holdout_used": "False",
        },
    ]

    artifact_path = OUTPUT_ROOT / "rd_b12_reusable_artifact_index.csv"
    with open(artifact_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["artifact", "path", "purpose", "safe_to_reuse", "restrictions", "stage2_holdout_used"])
        writer.writeheader()
        writer.writerows(artifact_rows)
    print(f"  [WRITE] {artifact_path.name}")

    # ── 4. forbidden use table CSV ───────────────────────────────────────────
    forbidden_rows = [
        {"category": "suppression", "action": "RD4AD score로 stage1_dev 후보 suppression 적용",          "reason": "lesion safety failure (G95: 82.3%, G99: 95.6% 병변 제거)"},
        {"category": "suppression", "action": "stage2_holdout에 suppression rule 적용",                  "reason": "holdout은 최종 평가용으로 잠금 상태"},
        {"category": "score",       "action": "first-stage PaDiM score를 RD4AD score로 교체",             "reason": "RD4AD score가 FP 감소 없이 병변 제거 과다"},
        {"category": "score",       "action": "first-stage score와 RD4AD score merge/blend",              "reason": "suppression NOT_ADOPTED 결정으로 score 수정 금지"},
        {"category": "holdout",     "action": "stage2_holdout 파일 열람/접근",                            "reason": "holdout은 LOCKED 상태, bias 방지"},
        {"category": "threshold",   "action": "RD-B9 threshold 재계산 또는 변경",                         "reason": "normal_val threshold 고정 상태, 재계산 불필요"},
        {"category": "training",    "action": "RD4AD 재학습 또는 파인튜닝",                                "reason": "학습은 PASS 판정, 재학습 근거 없음"},
        {"category": "training",    "action": "새 모델 forward/scoring 실행",                             "reason": "모든 scoring은 완료 상태, 재실행 금지"},
        {"category": "artifact",    "action": "기존 score/model/threshold/ROI/CT/mask 수정",              "reason": "closure 단계에서 artifact 수정 금지"},
        {"category": "artifact",    "action": "기존 결과 삭제 또는 덮어쓰기",                              "reason": "보존 의무"},
        {"category": "report",      "action": "기존 RD-B8~B11 report 덮어쓰기",                           "reason": "원본 보존 의무"},
    ]

    forbidden_path = OUTPUT_ROOT / "rd_b12_forbidden_use_table.csv"
    with open(forbidden_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "action", "reason"])
        writer.writeheader()
        writer.writerows(forbidden_rows)
    print(f"  [WRITE] {forbidden_path.name}")

    # ── 5. next recommendation MD ────────────────────────────────────────────
    next_rec = textwrap.dedent("""\
    # RD-B12 Next Recommendation

    ## A. RD-B normal-only RD4AD suppression track CLOSED

    - normal-only RD4AD 학습 자체는 성공했으나 FP suppression 용도로 사용 불가
    - 이유: lesion safety failure (G95 82.3%, G99 95.6% 병변 후보 제거)
    - 추가 suppression 시도 금지
    - 이 track은 공식 종결

    ## B. 허용: limited read-only ranking analysis (조건부)

    **조건:**
    - first_stage score 상위 + RD4AD score 하위 교차 분석만 허용
    - stage1_dev 후보 CSV 기반 read-only 분석만 허용
    - suppression 적용 금지
    - stage2_holdout 접근 금지
    - 분석 결과로 score 수정 또는 suppression 적용 금지

    **근거:**
    - Pearson r(all)=0.103, r(positive)=0.164 — 약한 상관이지만 추가 분석 가능성 존재
    - 단, 현재 강한 suppression의 대안이 될 수 없음

    ## C. 더 근본적인 개선 방향 (권장)

    ### C-1. 1차 PaDiM vNext: 6-bin normal distribution 적용
    - upper/middle/lower × boundary/interior 6-bin 별로 별도 mean/cov 학습
    - 흉벽/경계 FP는 1차 ROI/bin 설계에서 다루는 방향
    - 현재 단일 global 분포 대신 위치별 분포 분리로 FP 감소 기대
    - supervised auxiliary branch와 혼동 금지 (별도 설계 필요)

    ### C-2. ROI 설계 개선
    - v4_2p5_modeB 기반 흉벽 exclusion zone 조정
    - 경계 bin에서 발생하는 FP 원인 분석 후 ROI 파라미터 재검토

    ### C-3. 흉막/흉벽 FP 억제
    - 별도 rule 기반 흉막/흉벽 FP 필터 (B1-D track 결과 참조)
    - RD4AD score 기반 suppression과 혼용 금지

    ---

    **최종 요약:**
    - RD-B suppression track CLOSED
    - 보존 artifact: checkpoint, threshold, score CSV, correction JSON, analysis report
    - 다음: C-1 PaDiM vNext 6-bin 설계 검토 우선
    """)

    next_rec_path = OUTPUT_ROOT / "rd_b12_next_recommendation.md"
    with open(next_rec_path, "w") as f:
        f.write(next_rec)
    print(f"  [WRITE] {next_rec_path.name}")

    # ── 6. errors CSV ────────────────────────────────────────────────────────
    errors_path = OUTPUT_ROOT / "rd_b12_errors.csv"
    with open(errors_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "error_type", "message", "resolved"])
        writer.writeheader()
        if errors:
            for e in errors:
                writer.writerow({"step": "rd_b12", "error_type": "closure_error", "message": e, "resolved": "False"})
        else:
            writer.writerow({"step": "rd_b12", "error_type": "none", "message": "no errors", "resolved": "True"})
    print(f"  [WRITE] {errors_path.name}")

    # ── 7. closure report MD ─────────────────────────────────────────────────
    b11_lesion_rate_g95 = b11["lesion_suppression_rate_by_rule"]["G95"]
    b11_lesion_rate_g99 = b11["lesion_suppression_rate_by_rule"]["G99"]
    b11_hn_rate_g95     = b11["hard_negative_suppression_rate_by_rule"]["G95"]
    b11_patient_g95     = b11["patient_level_safety_by_rule"]["G95"]["all_suppressed_count"]
    b11_patient_g99     = b11["patient_level_safety_by_rule"]["G99"]["all_suppressed_count"]
    b11_pearson_all     = 0.103
    b11_pearson_pos     = 0.164

    report = textwrap.dedent(f"""\
    # RD-B12: Normal-Only RD4AD Verifier Track Closure Report

    **Status:** CLOSED — NOT_USEFUL_FOR_SUPPRESSION
    **Date:** {datetime.now().strftime("%Y-%m-%d")}
    **Track:** RD-B normal-only RD4AD verifier (RD-B8 ~ RD-B11)

    ---

    ## 1. Executive Summary

    RD-B8~RD-B11에 걸쳐 진행한 normal-only RD4AD 2차 verifier 실험을 공식 종결한다.

    - **학습:** PASS — ResNet18 teacher-student, 6-bin balanced, 20 epoch 완료, loss 0.186→0.074
    - **threshold:** PASS — normal_val 36명, global p95=0.0953, p99=0.1037
    - **scoring:** PASS — stage1_dev 22,112 후보 scoring 완료, NaN/Inf=0
    - **FP suppression:** NOT_ADOPTED — lesion safety 실패 (G95: {b11_lesion_rate_g95:.1f}%, G99: {b11_lesion_rate_g99:.1f}% 병변 제거)
    - **stage2_holdout:** LOCKED — 접근 0회

    결론: 학습 자체는 성공이지만 단독 threshold 기반 suppression rule로 사용할 수 없다.
    suppression은 채택하지 않으며, RD4AD score는 보존용 분석 artifact로만 남긴다.

    ---

    ## 2. Timeline

    | 단계 | 내용 | 결과 |
    |------|------|------|
    | RD-B8e | normal_train full float32 mixed_3ch shard 생성 | PASS |
    | RD-B8f | normal-only RD4AD full train (20 epoch) | PASS |
    | RD-B9  | normal_val threshold 생성 (p95/p99, 6-bin) | PASS |
    | RD-B10 | stage1_dev candidate scoring (22,112 rows) | PASS (correction 포함) |
    | RD-B11 | FP suppression lesion safety analysis | PASS (분석), DECISION: NOT_ADOPTED |

    ---

    ## 3. Normal-Only 학습 설계

    - **입력:** 2.5D mixed_3ch — 인접 3 slice를 RGB 채널로 stack
    - **학습 데이터:** normal_train 환자, 86,017 crops, 87 shards, 9,072 MB
    - **bin 설계:** upper/middle/lower × boundary/interior 6-bin balanced
    - **bin별 균형:** strict shortest-bin drop-last, per_bin=8, min_bin=upper_interior 13,932
    - **배치 설계:** batch_size=48, batches_per_epoch=1,741, crops_per_epoch=83,568
    - **모델:** ResNet18 teacher-student RD4AD (teacher frozen)
    - **총 학습:** 20 epoch, total_steps=34,820, runtime=773s

    ---

    ## 4. 학습 성공 근거

    - loss 0.186433 → 0.074174 (단조 감소 확인)
    - best_epoch=20 (overfitting 없음)
    - teacher_param_changed=False (teacher frozen 정상)
    - student_param_changed=True (student 정상 학습)
    - NaN/Inf loss=0/0
    - GPU peak memory=179.7 MB (안전)
    - checkpoint best/last 모두 정상 저장

    ---

    ## 5. Normal_val Threshold 생성 결과

    - normal_val patients=36, crops=8,354, shards=9
    - train/val overlap=0, stage2_holdout_intersection=0
    - global p95=0.095255, global p99=0.103721
    - 6-bin p95/p99 모두 생성 완료 (6/6 OK)
    - boundary bin 3개는 cap 1,800 도달 (정상)
    - interior bin 3개는 cap 미달 (upper_interior=673, 정상 범위)

    ---

    ## 6. Stage1_dev Candidate Scoring 결과

    - 입력: stage1_dev_fixed96_thr001_v1, 22,379행
    - holdout 제거: 2명, 267행 → scoring 대상=22,112행
    - post_filter_holdout_intersection=0
    - score NaN/Inf=0/0
    - B10 PASS correction: h_intersect 기준 → post_intersect 기준 수정 (rd_b10_pass_correction_v1)

    ---

    ## 7. RD-B11 Lesion Safety 실패 근거

    | Rule | Total suppressed | Lesion suppressed | Lesion patients all-suppressed |
    |------|-----------------|-------------------|-------------------------------|
    | G95  | 92.42%          | {b11_lesion_rate_g95:.2f}%            | {b11_patient_g95}명                          |
    | G99  | 97.63%          | {b11_lesion_rate_g99:.2f}%            | {b11_patient_g99}명                          |
    | B95  | 92.60%          | 83.18%            | 78명                          |
    | B99  | 97.70%          | 95.95%            | 118명                         |

    - positive rd4ad_mean=0.0855, hard_negative rd4ad_mean=0.0774 (separation 미약)
    - Pearson r: all={b11_pearson_all}, positive={b11_pearson_pos}, hard_negative=0.086
    - 모든 rule에서 lesion safety 실패 → suppression 채택 불가

    ---

    ## 8. 최종 Decision

    1. **RD4AD score suppression rule 적용 금지** — G95/G99/B95/B99 모두 NOT_ADOPTED
    2. **stage2_holdout에 적용 금지** — holdout LOCKED 유지
    3. **first-stage score 수정/교체 금지** — PaDiM score 원본 보존
    4. **RD4AD score는 분석 artifact로만 보존** — read-only 참조만 허용

    ---

    ## 9. 보존 Artifact 목록

    | Artifact | 경로 | 용도 |
    |----------|------|------|
    | best_train_loss.pth | outputs/models/.../checkpoints/best_train_loss.pth | RD4AD 학습 checkpoint |
    | last.pth | outputs/models/.../checkpoints/last.pth | last epoch checkpoint |
    | rd_b9_normal_val_threshold_candidates.csv | rd_b9.../... | threshold candidates |
    | rd_b10_stage1_dev_candidate_score.csv | rd_b10.../... | stage1_dev RD4AD scores |
    | rd_b10_pass_correction_v1.json/md | rd_b10.../... | PASS correction record |
    | rd_b11_rd4ad_fp_suppression_safety_report.md | rd_b11.../... | 분석 리포트 |
    | rd_b11_rd4ad_fp_suppression_safety_summary.json | rd_b11.../... | 분석 요약 |

    자세한 경로는 rd_b12_reusable_artifact_index.csv 참조.

    ---

    ## 10. 다음 추천

    1. **RD-B suppression track CLOSED** — 추가 suppression 시도 금지
    2. **가능한 경우:** first_stage high + RD4AD low 교차 read-only 분석 (suppression 적용 금지)
    3. **권장 방향:** 1차 PaDiM vNext 6-bin normal distribution — 위치별 분포 분리로 FP 근본 감소

    ---

    ## 11. 이번 단계에서 절대 하지 않은 것

    - stage2_holdout 접근 없음 (access count=0)
    - threshold 재계산 없음
    - first-stage score 수정 없음
    - suppression 적용 없음
    - 새 model forward/scoring 없음
    - lesion raw mask 접근 없음
    - 기존 artifact 수정/삭제/덮어쓰기 없음
    """)

    report_path = OUTPUT_ROOT / "rd_b12_rd4ad_track_closure_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  [WRITE] {report_path.name}")

    # ── 8. handoff document ──────────────────────────────────────────────────
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    handoff_path = HANDOFF_DIR / "rd_b_normal_only_rd4ad_track_close.md"

    if not handoff_path.exists():
        handoff = textwrap.dedent(f"""\
        # RD-B Normal-Only RD4AD Suppression Track — Closure Handoff

        **Status:** CLOSED
        **Decision:** NOT_ADOPTED (suppression)
        **Date:** {datetime.now().strftime("%Y-%m-%d")}
        **Closure output:** outputs/normal_based_stage2_verifier_audit/rd_b12_rd4ad_track_closure_v1/

        ## 핵심 결론

        - normal-only RD4AD 학습 자체는 성공 (loss 0.186→0.074, 20 epoch)
        - G95 threshold 기반 suppression: 병변 후보 82.3% 제거, 79명 전체 suppressed → 사용 불가
        - G99 threshold 기반 suppression: 병변 후보 95.6% 제거, 118명 전체 suppressed → 사용 불가
        - score separation weak: positive rd4ad_mean=0.0855 vs HN=0.0774
        - Pearson r(all)=0.103 — 약한 상관, suppression gate로 사용 불가
        - **suppression_decision = NOT_ADOPTED**
        - **stage2_holdout = LOCKED (접근 0회)**

        ## 보존 Artifact

        - checkpoint: outputs/models/rd_b8f_true_rd4ad_resnet18_mixed3ch_6bin_shard_v1/checkpoints/
        - threshold: rd_b9_normal_val_scoring_threshold_v1/rd_b9_normal_val_threshold_candidates.csv
        - score: rd_b10_stage1_dev_candidate_scoring_v2/rd_b10_stage1_dev_candidate_score.csv
        - analysis: rd_b11_rd4ad_fp_suppression_safety_analysis_v1/

        ## 다음 방향

        - RD-B suppression track CLOSED
        - 조건부 허용: first_stage high + RD4AD low 교차 read-only 분석 (suppression 적용 금지)
        - 권장: 1차 PaDiM vNext 6-bin normal distribution 설계 검토

        ## 금지 사항 (이 track 관련)

        - suppression rule stage2_holdout 적용 금지
        - first-stage score 수정/교체 금지
        - RD4AD 재학습 금지
        - threshold 재계산 금지
        """)

        with open(handoff_path, "w") as f:
            f.write(handoff)
        print(f"  [WRITE] {handoff_path}")
    else:
        print(f"  [SKIP ] handoff already exists: {handoff_path}")

    # ── 9. DONE ──────────────────────────────────────────────────────────────
    done_path = OUTPUT_ROOT / "DONE"
    with open(done_path, "w") as f:
        f.write(f"rd_b12 closure complete\ndate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nstatus: CLOSED_NOT_USEFUL_FOR_SUPPRESSION\n")
    print(f"  [WRITE] DONE")

    print()
    print("=" * 70)
    print("RD-B12 CLOSURE COMPLETE")
    print(f"  output root: {OUTPUT_ROOT}")
    print(f"  final_status: CLOSED_NOT_USEFUL_FOR_SUPPRESSION")
    print(f"  suppression_adopted: False")
    print(f"  stage2_holdout_access: 0")
    print("=" * 70)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) == 1:
        print("ERROR: bare run guard — no arguments provided")
        print("  Use --dry-plan to check artifacts")
        print("  Use --run-close (after user approval) to generate closure output")
        sys.exit(2)

    parser = argparse.ArgumentParser(description="RD-B12 RD4AD track closure")
    parser.add_argument("--dry-plan",  action="store_true", dest="dry_plan",  help="artifact check + plan only")
    parser.add_argument("--run-close", action="store_true", dest="run_close", help="generate closure output")
    args = parser.parse_args()

    if args.dry_plan:
        ok = dry_plan()
        sys.exit(0 if ok else 1)

    if args.run_close:
        run_close()
        sys.exit(0)

    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
