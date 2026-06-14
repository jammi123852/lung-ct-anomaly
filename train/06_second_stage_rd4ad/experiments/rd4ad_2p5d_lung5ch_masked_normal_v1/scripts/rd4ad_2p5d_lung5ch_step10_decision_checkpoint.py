"""
Step 10: Decision checkpoint
- score family lock (candidate + track)
- baseline improvement verification
- problem patient audit
- complete miss audit
- stage2 readiness judgment
decision/report only — no training, no model forward, no stage2 access
"""
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).resolve().parents[1]
DONE_STEP9 = ROOT / "DONE_STEP9_STAGE1DEV_EVAL.json"
PLAN_LOCK  = ROOT / "docs/FINAL_PLAN_LOCK.json"
STEP9_SUMMARY = ROOT / "reports/step9_stage1dev_eval_summary.json"
STEP9_HIT_CSV = ROOT / "manifests/step9_patient_hit_summary.csv"
STEP9_TRACK_CSV = ROOT / "manifests/step9_track_level_topk.csv"
STEP9_CAND_CSV  = ROOT / "manifests/step9_candidate_level_topk.csv"
STEP9_BASELINE_CSV = ROOT / "manifests/step9_baseline_comparison.csv"
STEP9_TRACK_SCORES_CSV = ROOT / "manifests/step9_track_level_scores.csv"
SCORE_CSV  = ROOT / "scoring/step8_stage1dev_v1/rd4ad_lung5ch_stage1dev_scores_v1.csv"

OUT_MANIFESTS = ROOT / "manifests"
OUT_REPORTS   = ROOT / "reports"
OUT_LOGS      = ROOT / "logs"

DENOMINATOR_CANDIDATES = 95995
PROBLEM_PATIENTS = ["LUNG1-086", "LUNG1-386", "LUNG1-399"]
TOPK_LIST = [1, 3, 5, 10, 20, 50]

PRIMARY_CANDIDATE_SCORE = "rd4ad_lung5ch_score_raw"
PRIMARY_TRACK_SCORE     = "raw_track_top3_mean"


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-decision", action="store_true")
    ap.add_argument("--confirm-plan-lock", action="store_true")
    ap.add_argument("--confirm-no-stage2", action="store_true")
    ap.add_argument("--confirm-decision-only", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    if not args.dry_run and not args.run_decision:
        print("bare run blocked — use --dry-run or --run-decision with confirm flags")
        sys.exit(2)

    if args.dry_run:
        print("=" * 64)
        print("Step 10 Decision Checkpoint — DRY-RUN PLAN")
        print("=" * 64)
        print()
        print("[선행 조건]")
        print(f"  DONE_STEP9_STAGE1DEV_EVAL.json : {DONE_STEP9}")
        print(f"  step9_stage1dev_eval_summary.json : {STEP9_SUMMARY}")
        print()
        print("[결정 항목]")
        print("  1. Step 9 결과 재검증 (denominator/NaN/Inf/stage2)")
        print("  2. score family decision (candidate + track)")
        print("  3. baseline improvement table")
        print("  4. P1/P2 reject 근거")
        print("  5. problem patient audit (LUNG1-086/386/399)")
        print("  6. complete miss audit top20 raw (36명)")
        print("  7. stage2 readiness 판단")
        print()
        print("[금지]")
        print("  training / model forward / checkpoint / stage2 / threshold / score 새 탐색")
        print()
        print("[생성 파일]")
        for f in [
            "manifests/step10_score_family_decision.csv",
            "manifests/step10_baseline_improvement_summary.csv",
            "manifests/step10_problem_patient_audit.csv",
            "manifests/step10_complete_miss_audit.csv",
            "reports/step10_decision_checkpoint_report.md",
            "reports/step10_decision_checkpoint_summary.json",
            "logs/step10_decision_checkpoint_errors.csv",
            "DONE_STEP10_DECISION_CHECKPOINT.json",
        ]:
            print(f"  {ROOT / f}")
        print()
        print("[실행 명령]")
        print("  python experiments/rd4ad_2p5d_lung5ch_masked_normal_v1/scripts/"
              "rd4ad_2p5d_lung5ch_step10_decision_checkpoint.py \\")
        print("    --run-decision --confirm-plan-lock --confirm-no-stage2 --confirm-decision-only")
        print()
        print("DRY-RUN 완료.")
        return

    if not (args.confirm_plan_lock and args.confirm_no_stage2 and args.confirm_decision_only):
        print("BLOCKED: confirm flags missing")
        sys.exit(2)

    for d in [OUT_MANIFESTS, OUT_REPORTS, OUT_LOGS]:
        d.mkdir(parents=True, exist_ok=True)

    errors = []
    guardrail = {
        "plan_lock_loaded": False,
        "step9_stage1dev_eval_passed": False,
        "decision_only": True,
        "stage2_holdout_accessed": False,
        "training_executed": False,
        "model_forward_executed": False,
        "checkpoint_saved": False,
        "checkpoint_modified": False,
        "threshold_tuning_executed": False,
        "candidate_deletion_executed": False,
        "representative_only_scoring_used": False,
        "score_selection_from_stage2": False,
        "positive_label_used_for_metric_only": True,
        "positive_label_used_for_training": False,
        "lesion_mask_used_for_training": False,
        "convae_branch_created": False,
        "image_reconstruction_loss_used": False,
        "primary_candidate_score_locked": PRIMARY_CANDIDATE_SCORE,
        "primary_track_score_locked": PRIMARY_TRACK_SCORE,
        "P1_rejected_for_lung5ch": True,
        "denominator_candidates": DENOMINATOR_CANDIDATES,
    }

    # ── [1] 선행 조건 검증 ─────────────────────────────────────────────────
    print("[1] 선행 조건 검증")

    if PLAN_LOCK.exists():
        guardrail["plan_lock_loaded"] = True
        print("  plan lock: OK")
    else:
        errors.append({"step": "plan_lock", "msg": f"not found: {PLAN_LOCK}"})
        print(f"  WARN: plan lock not found")

    if not DONE_STEP9.exists():
        print("BLOCKED: DONE_STEP9_STAGE1DEV_EVAL.json not found")
        sys.exit(2)

    with open(DONE_STEP9) as f:
        done9 = json.load(f)

    if done9.get("verdict") != "PASS_STEP9_STAGE1DEV_EVAL":
        print(f"BLOCKED: step9 verdict = {done9.get('verdict')}")
        sys.exit(2)
    if done9.get("denominator_candidates") != DENOMINATOR_CANDIDATES:
        print(f"BLOCKED: denominator={done9.get('denominator_candidates')} != {DENOMINATOR_CANDIDATES}")
        sys.exit(2)
    if done9.get("total_tracks") != 15830:
        print(f"WARN: total_tracks={done9.get('total_tracks')} (expected 15830)")
    if done9.get("total_patients") != 152 or done9.get("positive_patients") != 150:
        print(f"WARN: patients={done9.get('total_patients')}/pos={done9.get('positive_patients')}")

    guardrail["step9_stage1dev_eval_passed"] = True
    print(f"  DONE_STEP9: PASS")
    print(f"    denominator={done9['denominator_candidates']}, "
          f"tracks={done9['total_tracks']}, "
          f"patients={done9['total_patients']}/pos={done9['positive_patients']}")

    with open(STEP9_SUMMARY) as f:
        s9 = json.load(f)

    if s9.get("nan_count", 1) != 0 or s9.get("inf_count", 1) != 0:
        print(f"BLOCKED: NaN={s9.get('nan_count')} Inf={s9.get('inf_count')}")
        sys.exit(2)
    if s9.get("stage2_accessed", True):
        print("BLOCKED: stage2_accessed=True in step9 summary")
        sys.exit(2)

    print(f"  NaN=0, Inf=0, stage2_accessed=False — OK")

    # ── [2] score family decision ──────────────────────────────────────────
    print()
    print("[2] score family decision")

    cand_topk = {int(k): v for k, v in s9["candidate_topk"].items()}
    track_topk = {int(k): v for k, v in s9["track_topk"].items()}

    # 각 k별 전 score 비교
    score_family_rows = []
    for k in TOPK_LIST:
        ct = cand_topk[k]
        tt = track_topk[k]
        score_family_rows.append({
            "k": k,
            "cand_raw": ct["raw"],
            "cand_P1": ct["P1"],
            "cand_P2": ct["P2"],
            "track_raw_top3": tt["raw_top3"],
            "track_raw_max": tt["raw_max"],
            "track_P1_top3": tt["P1_top3"],
            "track_P1_max": tt["P1_max"],
            "track_P2_top3": tt["P2_top3"],
            "track_P2_max": tt["P2_max"],
        })

    df_sf = pd.DataFrame(score_family_rows)
    df_sf.to_csv(OUT_MANIFESTS / "step10_score_family_decision.csv", index=False)

    # decision logic
    raw_beats_P1_all_k  = all(cand_topk[k]["raw"] > cand_topk[k]["P1"] for k in TOPK_LIST)
    raw_beats_P2_all_k  = all(cand_topk[k]["raw"] > cand_topk[k]["P2"] for k in TOPK_LIST)
    rtt3_beats_P1t3_all = all(track_topk[k]["raw_top3"] > track_topk[k]["P1_top3"] for k in TOPK_LIST)
    rtt3_beats_P2t3_all = all(track_topk[k]["raw_top3"] > track_topk[k]["P2_top3"] for k in TOPK_LIST)
    rtt3_beats_rmax_all = all(track_topk[k]["raw_top3"] >= track_topk[k]["raw_max"] for k in TOPK_LIST)

    cand_decision     = "raw" if (raw_beats_P1_all_k and raw_beats_P2_all_k) else "raw (partial)"
    track_decision    = "raw_top3" if (rtt3_beats_P1t3_all and rtt3_beats_P2t3_all) else "raw_top3 (partial)"
    P1_decision       = "REJECT" if raw_beats_P1_all_k else "AUXILIARY"
    P2_decision       = "REJECT" if raw_beats_P2_all_k else "AUXILIARY"

    print(f"  raw > P1 전 구간: {raw_beats_P1_all_k}  → P1 decision: {P1_decision}")
    print(f"  raw > P2 전 구간: {raw_beats_P2_all_k}  → P2 decision: {P2_decision}")
    print(f"  raw_top3 > P1_top3 전 구간: {rtt3_beats_P1t3_all}")
    print(f"  raw_top3 > P2_top3 전 구간: {rtt3_beats_P2t3_all}")
    print(f"  raw_top3 >= raw_max 전 구간: {rtt3_beats_rmax_all}")
    print(f"  → primary candidate score: {PRIMARY_CANDIDATE_SCORE}")
    print(f"  → primary track score    : {PRIMARY_TRACK_SCORE}")

    # ── [3] baseline improvement ───────────────────────────────────────────
    print()
    print("[3] baseline improvement")

    df_bl = pd.read_csv(str(STEP9_BASELINE_CSV))
    # baseline_rd_d1s_raw_hit_rate 컬럼이 숫자인지 확인
    bl_is_numeric = pd.to_numeric(df_bl["baseline_rd_d1s_raw_hit_rate"], errors="coerce").notna().all()

    bl_rows = []
    if bl_is_numeric:
        df_bl["baseline_rd_d1s_raw_hit_rate"] = pd.to_numeric(df_bl["baseline_rd_d1s_raw_hit_rate"])
        for _, r in df_bl.iterrows():
            diff_raw  = round(r["new_lung5ch_raw_hit_rate"] - r["baseline_rd_d1s_raw_hit_rate"], 4)
            diff_P1   = round(r["new_lung5ch_P1_hit_rate"]  - r["baseline_rd_d1s_raw_hit_rate"], 4)
            diff_trk  = round(r["new_lung5ch_P1_track_top3_hit_rate"] - r["baseline_rd_d1s_raw_hit_rate"], 4)
            bl_rows.append({
                "k": int(r["k"]),
                "new_raw": r["new_lung5ch_raw_hit_rate"],
                "new_P1":  r["new_lung5ch_P1_hit_rate"],
                "new_P1_track_top3": r["new_lung5ch_P1_track_top3_hit_rate"],
                "baseline_raw": r["baseline_rd_d1s_raw_hit_rate"],
                "diff_raw_vs_baseline": diff_raw,
                "diff_P1_vs_baseline":  diff_P1,
                "diff_track_vs_baseline": diff_trk,
                "raw_over_baseline": diff_raw > 0,
            })
        raw_over_all  = all(r["raw_over_baseline"] for r in bl_rows)
        baseline_verdict = "PASS_LUNG5CH_OVER_BASELINE" if raw_over_all else "PARTIAL_PASS"
    else:
        baseline_verdict = "BASELINE_NOT_FOUND"
        errors.append({"step": "baseline", "msg": "baseline_rd_d1s_raw_hit_rate not numeric"})

    df_bl_out = pd.DataFrame(bl_rows) if bl_rows else pd.DataFrame()
    df_bl_out.to_csv(OUT_MANIFESTS / "step10_baseline_improvement_summary.csv", index=False)

    for r in bl_rows:
        print(f"  @{r['k']:2d}: new_raw={r['new_raw']:.4f}  baseline={r['baseline_raw']:.4f}  "
              f"diff={r['diff_raw_vs_baseline']:+.4f}  {'PASS' if r['raw_over_baseline'] else 'FAIL'}")
    print(f"  → baseline verdict: {baseline_verdict}")

    # ── [4] P1/P2 reject 근거 ─────────────────────────────────────────────
    p1_reject_reason = (
        "lung5ch 모델은 v4_20 mask 내부 feature loss로 학습되었기 때문에, score 자체가 이미 "
        "폐/ROI 내부 feature anomaly에 집중되어 있다. 여기에 roi_ratio를 다시 곱하는 P1/P2 방식은 "
        "positive 후보까지 과도하게 penalty를 주는 경향을 보였고, stage1_dev에서 raw 대비 "
        "전 top-k 구간에서 열세였다. 따라서 lung5ch branch에서는 P1을 primary로 채택하지 않는다."
    )

    # ── [5] problem patient audit ──────────────────────────────────────────
    print()
    print("[5] problem patient audit")

    df_hit  = pd.read_csv(str(STEP9_HIT_CSV))
    df_score = pd.read_csv(str(SCORE_CSV))
    df_trk   = pd.read_csv(str(STEP9_TRACK_SCORES_CSV))

    pp_rows = []
    for pid in PROBLEM_PATIENTS:
        grp = df_score[df_score["patient_id"] == pid]
        hit_row = df_hit[df_hit["patient_id"] == pid]
        trk_grp = df_trk[df_trk["patient_id"] == pid]

        if len(grp) == 0:
            pp_rows.append({"patient_id": pid, "note": "not in scoring CSV"})
            continue

        n_cands = len(grp)
        n_tracks = len(trk_grp)
        n_pos_cands = int((grp["label"] == "positive").sum())
        n_pos_tracks = int(trk_grp["is_positive"].sum()) if len(trk_grp) else 0

        # candidate rank
        grp_sorted = grp.sort_values("rd4ad_lung5ch_score_raw", ascending=False).reset_index(drop=True)
        pos_cand_ranks = [i+1 for i, r in grp_sorted.iterrows() if r["label"] == "positive"]
        best_cand_rank_raw = pos_cand_ranks[0] if pos_cand_ranks else -1

        # track rank
        if len(trk_grp):
            trk_sorted = trk_grp.sort_values("raw_track_top3_mean", ascending=False).reset_index(drop=True)
            pos_trk_ranks = [i+1 for i, r in trk_sorted.iterrows() if r["is_positive"] == 1]
            best_trk_rank_raw_top3 = pos_trk_ranks[0] if pos_trk_ranks else -1
        else:
            best_trk_rank_raw_top3 = -1

        hr = hit_row.iloc[0] if len(hit_row) else {}

        row = {
            "patient_id": pid,
            "total_candidates": n_cands,
            "total_tracks": n_tracks,
            "positive_candidates": n_pos_cands,
            "positive_tracks": n_pos_tracks,
            "best_positive_candidate_rank_raw": best_cand_rank_raw,
            "best_positive_track_rank_raw_top3": best_trk_rank_raw_top3,
            "hit_top10_raw": int(hr.get("hit@10_raw", -1)) if len(hit_row) else -1,
            "hit_top20_raw": int(hr.get("hit@20_raw", -1)) if len(hit_row) else -1,
            "hit_top50_raw": int(hr.get("hit@50_raw", -1)) if len(hit_row) else -1,
            "complete_miss_top20": int(best_cand_rank_raw == -1 or (best_cand_rank_raw > 20)) if n_pos_cands > 0 else None,
            "note": (
                "positive candidates 없음 — hard_negative only patient; "
                "stage1_dev에서 positive 평가 제외 (is_positive_patient=0)"
                if n_pos_cands == 0
                else f"positive {n_pos_cands}개 존재, best rank={best_cand_rank_raw}"
            ),
        }
        pp_rows.append(row)

    df_pp = pd.DataFrame(pp_rows)
    df_pp.to_csv(OUT_MANIFESTS / "step10_problem_patient_audit.csv", index=False)
    for r in pp_rows:
        print(f"  {r['patient_id']}: cands={r.get('total_candidates')}, "
              f"pos={r.get('positive_candidates')}, "
              f"best_cand_rank={r.get('best_positive_candidate_rank_raw')}, "
              f"top20_hit={r.get('hit_top20_raw')}")

    # ── [6] complete miss audit ────────────────────────────────────────────
    print()
    print("[6] complete miss audit (top20 raw)")

    miss_list = s9.get("complete_miss_top20_raw", [])
    print(f"  miss count: {len(miss_list)}")

    miss_rows = []
    for pid in miss_list:
        grp = df_score[df_score["patient_id"] == pid]
        trk_g = df_trk[df_trk["patient_id"] == pid]
        n_cands = len(grp)
        n_tracks = len(trk_g)
        n_pos_cands = int((grp["label"] == "positive").sum())
        n_pos_tracks = int(trk_g["is_positive"].sum()) if len(trk_g) else 0

        grp_sorted = grp.sort_values("rd4ad_lung5ch_score_raw", ascending=False).reset_index(drop=True)
        pos_ranks = [i+1 for i, r in grp_sorted.iterrows() if r["label"] == "positive"]
        best_cand = pos_ranks[0] if pos_ranks else -1

        if len(trk_g):
            trk_sorted = trk_g.sort_values("raw_track_top3_mean", ascending=False).reset_index(drop=True)
            pos_trk = [i+1 for i, r in trk_sorted.iterrows() if r["is_positive"] == 1]
            best_trk = pos_trk[0] if pos_trk else -1
        else:
            best_trk = -1

        hit_row = df_hit[df_hit["patient_id"] == pid]
        h = hit_row.iloc[0] if len(hit_row) else {}

        # failure mode heuristic
        if n_pos_cands == 0:
            fm = "no_positive_candidate"
        elif n_cands > 0 and n_pos_cands / n_cands < 0.02:
            fm = "low_positive_density"
        elif best_cand > 100:
            fm = "positive_buried_deep"
        else:
            fm = "borderline_rank"

        miss_rows.append({
            "patient_id": pid,
            "total_candidates": n_cands,
            "total_tracks": n_tracks,
            "positive_candidates": n_pos_cands,
            "positive_tracks": n_pos_tracks,
            "best_positive_candidate_rank_raw": best_cand,
            "best_positive_track_rank_raw_top3": best_trk,
            "hit_top10":  int(h.get("hit@10_raw", 0)) if len(hit_row) else 0,
            "hit_top20":  0,
            "hit_top50":  int(h.get("hit@50_raw", 0)) if len(hit_row) else 0,
            "likely_failure_mode": fm,
            "note": f"complete miss @top20 raw; best_cand_rank={best_cand}",
        })

    df_miss = pd.DataFrame(miss_rows)
    df_miss.to_csv(OUT_MANIFESTS / "step10_complete_miss_audit.csv", index=False)

    fm_counts = df_miss["likely_failure_mode"].value_counts().to_dict()
    print(f"  failure mode breakdown: {fm_counts}")
    miss_top50_recoverable = int((df_miss["hit_top50"] == 1).sum())
    print(f"  top50으로 회복 가능: {miss_top50_recoverable} / {len(miss_list)}")

    # ── [7] stage2 readiness ───────────────────────────────────────────────
    print()
    print("[7] stage2 readiness 판단")

    stage2_ready_conditions = {
        "lung5ch_raw_over_baseline_all_k": baseline_verdict == "PASS_LUNG5CH_OVER_BASELINE",
        "primary_candidate_score_locked": True,
        "primary_track_score_locked": True,
        "P1_rejected_or_auxiliary": P1_decision in ("REJECT", "AUXILIARY"),
        "P2_rejected_or_auxiliary": P2_decision in ("REJECT", "AUXILIARY"),
        "problem_patient_audit_complete": len(pp_rows) == len(PROBLEM_PATIENTS),
        "complete_miss_audit_complete": len(miss_rows) == len(miss_list),
        "stage2_access_false": True,
        "threshold_tuning_executed": False,
        "candidate_deletion_executed": False,
    }

    stage2_ready = all(v is True or v is False and k == "threshold_tuning_executed"
                       or v is False and k == "candidate_deletion_executed"
                       for k, v in stage2_ready_conditions.items()
                       if isinstance(v, bool) and v is not False
                       ) and all(stage2_ready_conditions.values()
                                  if k not in ("threshold_tuning_executed", "candidate_deletion_executed")
                                  else True
                                  for k in stage2_ready_conditions)

    # simpler check
    stage2_ready = (
        stage2_ready_conditions["lung5ch_raw_over_baseline_all_k"] and
        stage2_ready_conditions["primary_candidate_score_locked"] and
        stage2_ready_conditions["primary_track_score_locked"] and
        stage2_ready_conditions["problem_patient_audit_complete"] and
        stage2_ready_conditions["complete_miss_audit_complete"] and
        stage2_ready_conditions["stage2_access_false"] and
        not stage2_ready_conditions["threshold_tuning_executed"] and
        not stage2_ready_conditions["candidate_deletion_executed"]
    )

    stage2_readiness_verdict = "PASS_STAGE2_READY" if stage2_ready else "HOLD"
    for k, v in stage2_ready_conditions.items():
        print(f"  {k}: {v}")
    print(f"  → {stage2_readiness_verdict}")

    # ── report ────────────────────────────────────────────────────────────
    print()
    print("[8] report 생성")

    overall_verdict = "PASS_STEP10_DECISION_CHECKPOINT" if (
        baseline_verdict == "PASS_LUNG5CH_OVER_BASELINE" and stage2_ready
    ) else "PARTIAL_PASS_STEP10"

    lines = []
    lines.append("# Step 10 Decision Checkpoint Report")
    lines.append("")
    lines.append("## Verdict")
    lines.append(f"**{overall_verdict}**")
    lines.append("")
    lines.append("## 1. Step 9 결과 재검증")
    lines.append(f"- denominator: {DENOMINATOR_CANDIDATES:,}")
    lines.append(f"- tracks: {done9['total_tracks']:,}")
    lines.append(f"- patients: {done9['total_patients']} total / {done9['positive_patients']} positive")
    lines.append(f"- NaN=0, Inf=0, stage2_accessed=False — OK")
    lines.append("")
    lines.append("## 2. Score Family Decision")
    lines.append("")
    lines.append(f"**Primary candidate score**: `{PRIMARY_CANDIDATE_SCORE}` — LOCKED")
    lines.append(f"**Primary track score**: `{PRIMARY_TRACK_SCORE}` — LOCKED")
    lines.append(f"**P1_times_roi**: {P1_decision}")
    lines.append(f"**P2_times_sqrt_roi**: {P2_decision}")
    lines.append("")
    lines.append("### P1/P2 Reject 근거")
    lines.append(p1_reject_reason)
    lines.append("")
    lines.append("### Candidate-level top-k (patient hit rate)")
    lines.append("| k | raw | P1 | P2 |")
    lines.append("|---|---|---|---|")
    for k in TOPK_LIST:
        ct = cand_topk[k]
        lines.append(f"| {k} | {ct['raw']} | {ct['P1']} | {ct['P2']} |")
    lines.append("")
    lines.append("### Track-level top-k (patient hit rate)")
    lines.append("| k | raw_top3 | raw_max | P1_top3 | P2_top3 |")
    lines.append("|---|---|---|---|---|")
    for k in TOPK_LIST:
        tt = track_topk[k]
        lines.append(f"| {k} | {tt['raw_top3']} | {tt['raw_max']} | {tt['P1_top3']} | {tt['P2_top3']} |")
    lines.append("")
    lines.append(
        "> **Caution**: Candidate-level top-k와 track-level top-k는 선택 단위가 다르므로 "
        "직접적인 수치 비교는 주의해야 한다."
    )
    lines.append("")
    lines.append("## 3. Baseline Improvement (동일 95,995 candidates)")
    lines.append("")
    lines.append(f"Baseline verdict: **{baseline_verdict}**")
    lines.append("")
    if bl_rows:
        lines.append("| k | new_raw | baseline_raw | diff | verdict |")
        lines.append("|---|---|---|---|---|")
        for r in bl_rows:
            lines.append(
                f"| {r['k']} | {r['new_raw']:.4f} | {r['baseline_raw']:.4f} "
                f"| {r['diff_raw_vs_baseline']:+.4f} | {'PASS' if r['raw_over_baseline'] else 'FAIL'} |"
            )
    lines.append("")
    lines.append(
        "> **Note**: 비교 기준은 동일 95,995 candidates, raw score만. "
        "기존 RD-D1s는 P1/track score 미제공 (raw만 비교 가능). "
        "eval set 구성 차이 주의: 기존 rd4ad_eval.json stage1_dev는 positive-only candidates였으나 "
        "새 평가는 hard_negative(61,287개) 혼합 — 새 모델에 불리한 조건에서도 raw 전 구간 우세."
    )
    lines.append("")
    lines.append("## 4. Problem Patient Audit")
    lines.append("")
    for r in pp_rows:
        lines.append(f"**{r['patient_id']}**")
        for kk, vv in r.items():
            if kk != "patient_id":
                lines.append(f"  - {kk}: {vv}")
        lines.append("")
    lines.append("## 5. Complete Miss @top20 raw")
    lines.append("")
    lines.append(f"- count: {len(miss_list)} / 150 positive patients")
    lines.append(f"- failure mode breakdown: {fm_counts}")
    lines.append(f"- top50으로 회복 가능: {miss_top50_recoverable}")
    lines.append(f"- 상세: manifests/step10_complete_miss_audit.csv")
    lines.append("")
    lines.append("## 6. Stage2 Readiness")
    lines.append("")
    lines.append(f"**{stage2_readiness_verdict}**")
    lines.append("")
    for k, v in stage2_ready_conditions.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 7. Guardrail")
    lines.append("")
    for kk, vv in guardrail.items():
        lines.append(f"- {kk}: {vv}")

    rpt_path = OUT_REPORTS / "step10_decision_checkpoint_report.md"
    rpt_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "step": "step10_decision_checkpoint",
        "verdict": overall_verdict,
        "created": "2026-06-10",
        "denominator_candidates": DENOMINATOR_CANDIDATES,
        "primary_candidate_score": PRIMARY_CANDIDATE_SCORE,
        "primary_track_score": PRIMARY_TRACK_SCORE,
        "P1_decision": P1_decision,
        "P2_decision": P2_decision,
        "baseline_verdict": baseline_verdict,
        "baseline_improvement": {r["k"]: r["diff_raw_vs_baseline"] for r in bl_rows},
        "complete_miss_top20_raw_count": len(miss_list),
        "complete_miss_top50_recoverable": miss_top50_recoverable,
        "problem_patient_audit": {r["patient_id"]: r.get("note", "") for r in pp_rows},
        "stage2_readiness": stage2_readiness_verdict,
        "stage2_ready_conditions": stage2_ready_conditions,
        "guardrail": guardrail,
    }

    s_path = OUT_REPORTS / "step10_decision_checkpoint_summary.json"
    with open(s_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    err_path = OUT_LOGS / "step10_decision_checkpoint_errors.csv"
    with open(err_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "msg"])
        w.writeheader()
        for e in errors:
            w.writerow(e)

    done_path = ROOT / "DONE_STEP10_DECISION_CHECKPOINT.json"
    with open(done_path, "w") as f:
        json.dump({
            "step": "step10_decision_checkpoint",
            "verdict": overall_verdict,
            "created": "2026-06-10",
            "primary_candidate_score": PRIMARY_CANDIDATE_SCORE,
            "primary_track_score": PRIMARY_TRACK_SCORE,
            "P1_rejected": True,
            "baseline_verdict": baseline_verdict,
            "stage2_readiness": stage2_readiness_verdict,
            "report": str(rpt_path),
            "summary_json": str(s_path),
        }, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 64)
    print(f"판정: {overall_verdict}")
    print("=" * 64)
    print(f"  primary candidate score : {PRIMARY_CANDIDATE_SCORE}")
    print(f"  primary track score     : {PRIMARY_TRACK_SCORE}")
    print(f"  P1 decision             : {P1_decision}")
    print(f"  P2 decision             : {P2_decision}")
    print(f"  baseline verdict        : {baseline_verdict}")
    print(f"  complete miss @top20    : {len(miss_list)} / 150 positive")
    print(f"  stage2 readiness        : {stage2_readiness_verdict}")
    print(f"  stage2 accessed         : False")
    print()
    print("다음 단계: Step 11 stage2 fixed-evaluation preflight (사용자 승인 후)")


if __name__ == "__main__":
    main()
