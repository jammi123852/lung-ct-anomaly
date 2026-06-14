"""
B1-D4k: Rule-B3 stage1_dev candidate-level preview validation checkpoint
- B1-D4j output CSV를 read-only로 검증
- metric/FROC/AUROC 계산 없음, threshold 재계산 없음, 기존 score CSV 수정 없음
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

B1D4J_CSV     = ROOT / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1/b1d4j_rule_b3_soft_penalty_0_5_stage1_dev_candidate_preview.csv"
B1D4J_SUMMARY = ROOT / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1/b1d4j_rule_b3_stage1_dev_candidate_preview_summary.json"
B1D4J_REPORT  = ROOT / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1/b1d4j_rule_b3_stage1_dev_candidate_preview_report.md"
PATCH_CSV     = ROOT / "outputs/position-aware-padim-v1/candidates/padim_v2_roi0_0_explanation_candidates_v1/patch_candidates.csv"
SENTINEL_CSV  = ROOT / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1/b1d4i_rule_b3_stage1_dev_full_preflight_safety_sentinels.csv"

OUT_DIR       = ROOT / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"
OUT_SUMMARY   = OUT_DIR / "b1d4k_rule_b3_stage1_dev_candidate_preview_validation_summary.json"
OUT_REPORT    = OUT_DIR / "b1d4k_rule_b3_stage1_dev_candidate_preview_validation_report.md"
OUT_PENALIZED = OUT_DIR / "b1d4k_rule_b3_penalized_rows_validation.csv"
OUT_RANKSHIFT = OUT_DIR / "b1d4k_rule_b3_top_rank_shift_preview.csv"

CHUNK_SIZE = 50_000
EXPECTED_ROWS = 761_206
PENALIZED_RIDS = {"R001", "R015", "R016", "R028"}
REQUIRED_COLS = [
    "original_score", "adjusted_score_preview", "soft_penalty_applied",
    "review_id", "patient_id", "candidate_id", "local_z", "y0", "x0",
    "holdout_flag", "stage_split_safety_flag",
]
FORBIDDEN_COLS = ["adjusted_score", "suppression_weight", "refined_score"]
SENTINEL_GROUPS = [
    "boundary_hard_case_must_keep", "lesion_kept", "lesion_risk_partial",
    "unreviewed_hold", "gate_candidate", "observation", "AD_wall_med", "AD_other",
]


def main():
    t_start = time.time()
    fail_reasons = []

    # ── STEP 0: output collision precheck ────────────────────────────────────
    print("[STEP 0] output collision precheck...")
    for p in [OUT_SUMMARY, OUT_REPORT, OUT_PENALIZED, OUT_RANKSHIFT]:
        if p.exists():
            print(f"[BLOCKED] 출력 파일 이미 존재: {p}", file=sys.stderr)
            sys.exit(1)
    print("[OK] 출력 파일 없음 확인")

    # ── STEP 1: 입력 mtime 기록 ───────────────────────────────────────────────
    print("[STEP 1] 입력 파일 mtime 기록...")
    input_files = [B1D4J_CSV, B1D4J_SUMMARY, PATCH_CSV, SENTINEL_CSV]
    for f in input_files:
        if not f.exists():
            print(f"[ERROR] 입력 파일 없음: {f}", file=sys.stderr)
            sys.exit(1)
    mtime_before = {str(f): os.path.getmtime(f) for f in input_files}
    print("[OK] mtime 기록 완료")

    # ── STEP 2: b1d4j summary 검증 ───────────────────────────────────────────
    print("[STEP 2] b1d4j summary 검증...")
    with open(B1D4J_SUMMARY) as fh:
        b1d4j_sum = json.load(fh)

    checks = [
        (b1d4j_sum.get("verdict") == "PASS",              "b1d4j_verdict != PASS"),
        (b1d4j_sum.get("total_rows") == EXPECTED_ROWS,    f"total_rows={b1d4j_sum.get('total_rows')} != {EXPECTED_ROWS}"),
        (b1d4j_sum.get("output_rows") == EXPECTED_ROWS,   f"output_rows={b1d4j_sum.get('output_rows')} != {EXPECTED_ROWS}"),
        (b1d4j_sum.get("penalized_rows") == 4,            f"penalized_rows={b1d4j_sum.get('penalized_rows')} != 4"),
        (set(b1d4j_sum.get("match_found_rids", [])) == PENALIZED_RIDS,
                                                          f"match_found_rids={b1d4j_sum.get('match_found_rids')} != {PENALIZED_RIDS}"),
        (b1d4j_sum.get("stage2_holdout_access") == 0,     "stage2_holdout_access != 0"),
        (b1d4j_sum.get("fail_count") == 0,                f"fail_count={b1d4j_sum.get('fail_count')} != 0"),
    ]
    for ok, msg in checks:
        if not ok:
            fail_reasons.append(f"STEP2: {msg}")
            print(f"  [FAIL] {msg}")
        else:
            print(f"  [OK] {msg.split('!=')[0].strip() if '!=' in msg else msg}")

    # ── STEP 3: 컬럼 무결성 ──────────────────────────────────────────────────
    print("[STEP 3] output CSV 컬럼 무결성...")
    header_df = pd.read_csv(B1D4J_CSV, nrows=0)
    actual_cols = set(header_df.columns.tolist())

    missing_required = [c for c in REQUIRED_COLS if c not in actual_cols]
    present_forbidden = [c for c in FORBIDDEN_COLS if c in actual_cols]

    if missing_required:
        msg = f"필수 컬럼 없음: {missing_required}"
        fail_reasons.append(f"STEP3: {msg}")
        print(f"  [FAIL] {msg}")
    else:
        print(f"  [OK] 필수 컬럼 {len(REQUIRED_COLS)}개 모두 존재")

    if present_forbidden:
        msg = f"금지 컬럼 존재: {present_forbidden}"
        fail_reasons.append(f"STEP3: {msg}")
        print(f"  [FAIL] {msg}")
    else:
        print(f"  [OK] 금지 컬럼 없음: {FORBIDDEN_COLS}")

    # ── STEP 4: streaming 1:1 대응 검증 ─────────────────────────────────────
    print("[STEP 4] streaming 1:1 대응 검증...")
    total_rows_b1d4j = 0
    total_rows_patch = 0
    mismatch_patient_id_count = 0
    mismatch_candidate_id_count = 0
    mismatch_coord_count = 0
    mismatch_original_score_count = 0
    holdout_flag_1_count = 0
    nan_original_score_count = 0
    nan_adjusted_score_preview_count = 0

    b1d4j_iter = pd.read_csv(
        B1D4J_CSV, chunksize=CHUNK_SIZE, low_memory=False,
        dtype={"holdout_flag": str, "stage_split_safety_flag": str},
    )
    patch_iter = pd.read_csv(
        PATCH_CSV, chunksize=CHUNK_SIZE, low_memory=False,
    )

    chunk_idx = 0
    for b_chunk, p_chunk in zip(b1d4j_iter, patch_iter):
        chunk_idx += 1
        n_b = len(b_chunk)
        n_p = len(p_chunk)
        total_rows_b1d4j += n_b
        total_rows_patch += n_p

        # holdout_flag 처리
        hf = b_chunk["holdout_flag"].fillna("0").replace("", "0")
        hf_int = pd.to_numeric(hf, errors="coerce").fillna(0).astype(int)
        holdout_flag_1_count += int((hf_int == 1).sum())

        # NaN 검사
        nan_original_score_count += int(b_chunk["original_score"].isna().sum())
        nan_adjusted_score_preview_count += int(b_chunk["adjusted_score_preview"].isna().sum())

        # patient_id
        mis_pid = int((b_chunk["patient_id"].values != p_chunk["patient_id"].values).sum())
        mismatch_patient_id_count += mis_pid

        # candidate_id vs candidate_patch_id
        mis_cid = int((b_chunk["candidate_id"].values != p_chunk["candidate_patch_id"].values).sum())
        mismatch_candidate_id_count += mis_cid

        # coord: local_z, y0, x0
        mis_coord = int(
            ((b_chunk["local_z"].values != p_chunk["local_z"].values) |
             (b_chunk["y0"].values != p_chunk["y0"].values) |
             (b_chunk["x0"].values != p_chunk["x0"].values)).sum()
        )
        mismatch_coord_count += mis_coord

        # score 비교
        b_score = b_chunk["original_score"].values.astype(float)
        p_score = p_chunk["padim_score"].values.astype(float)
        close = np.isclose(b_score, p_score, rtol=1e-5, atol=1e-6)
        mismatch_original_score_count += int((~close).sum())

        if chunk_idx % 4 == 0:
            print(f"  processed {total_rows_b1d4j:,} rows... ({int(time.time()-t_start)}s)")

    print(f"  b1d4j rows: {total_rows_b1d4j:,}")
    print(f"  patch rows: {total_rows_patch:,}")
    print(f"  mismatch patient_id: {mismatch_patient_id_count}")
    print(f"  mismatch candidate_id: {mismatch_candidate_id_count}")
    print(f"  mismatch coord: {mismatch_coord_count}")
    print(f"  mismatch original_score: {mismatch_original_score_count}")
    print(f"  holdout_flag=1: {holdout_flag_1_count}")

    input_output_row_match = (total_rows_b1d4j == total_rows_patch == EXPECTED_ROWS)
    original_score_integrity = (mismatch_original_score_count == 0)

    if not input_output_row_match:
        fail_reasons.append(f"STEP4: row 수 불일치 b1d4j={total_rows_b1d4j} patch={total_rows_patch}")
    if mismatch_patient_id_count > 0:
        fail_reasons.append(f"STEP4: patient_id 불일치 {mismatch_patient_id_count}행")
    if mismatch_candidate_id_count > 0:
        fail_reasons.append(f"STEP4: candidate_id 불일치 {mismatch_candidate_id_count}행")
    if mismatch_coord_count > 0:
        fail_reasons.append(f"STEP4: coord(local_z/y0/x0) 불일치 {mismatch_coord_count}행")
    if not original_score_integrity:
        fail_reasons.append(f"STEP4: original_score != patch.padim_score {mismatch_original_score_count}행")
    if holdout_flag_1_count > 0:
        fail_reasons.append(f"STEP4: holdout_flag=1 행 존재 {holdout_flag_1_count}행")
    if nan_original_score_count > 0:
        fail_reasons.append(f"STEP4: original_score NaN {nan_original_score_count}행")
    if nan_adjusted_score_preview_count > 0:
        fail_reasons.append(f"STEP4: adjusted_score_preview NaN {nan_adjusted_score_preview_count}행")

    # ── STEP 5: penalized row 검증 ───────────────────────────────────────────
    print("[STEP 5] penalized row 검증...")
    penalized_rows = []
    non_penalized_count = 0
    non_penalized_unchanged_count = 0
    non_penalized_changed_count = 0

    for chunk in pd.read_csv(B1D4J_CSV, chunksize=CHUNK_SIZE, low_memory=False,
                              dtype={"holdout_flag": str}):
        pen_mask = chunk["soft_penalty_applied"].astype(str).str.lower() == "true"
        penalized_rows.append(chunk[pen_mask])

        non_pen = chunk[~pen_mask]
        non_penalized_count += len(non_pen)
        unchanged = np.isclose(
            non_pen["original_score"].values.astype(float),
            non_pen["adjusted_score_preview"].values.astype(float),
            rtol=1e-5, atol=1e-8,
        )
        non_penalized_unchanged_count += int(unchanged.sum())
        non_penalized_changed_count += int((~unchanged).sum())

    penalized_df = pd.concat(penalized_rows, ignore_index=True) if penalized_rows else pd.DataFrame()

    pen_count = len(penalized_df)
    pen_rids = set(penalized_df["review_id"].dropna().tolist()) if pen_count > 0 else set()

    pen_val_rows = []
    if pen_count != 4:
        fail_reasons.append(f"STEP5: penalized count={pen_count} != 4")
    if pen_rids != PENALIZED_RIDS:
        fail_reasons.append(f"STEP5: penalized review_ids={pen_rids} != {PENALIZED_RIDS}")

    for _, row in penalized_df.iterrows():
        rid = row.get("review_id", "")
        orig = float(row["original_score"])
        adj = float(row["adjusted_score_preview"])
        delta_pct = float(row.get("score_delta_percent", 0))
        hf_val = str(row.get("holdout_flag", "0") or "0")
        hf_int = int(float(hf_val)) if hf_val.replace(".", "").isdigit() else 0
        cause = str(row.get("cause_class", ""))

        status_parts = []
        if not np.isclose(adj, orig * 0.5, rtol=1e-4):
            status_parts.append(f"FAIL_adj_score(expected={orig*0.5:.4f} got={adj:.4f})")
        if not np.isclose(delta_pct, -50.0, atol=0.01):
            status_parts.append(f"FAIL_delta_pct({delta_pct:.2f})")
        if hf_int != 0:
            status_parts.append(f"FAIL_holdout_flag({hf_int})")
        if cause.lower() not in ("b_boundary", "b boundary"):
            status_parts.append(f"FAIL_cause_class({cause})")

        validation_status = "PASS" if not status_parts else "_".join(status_parts)
        if status_parts:
            fail_reasons.append(f"STEP5: {rid} {validation_status}")

        pen_val_rows.append({
            "review_id": rid,
            "patient_id": row.get("patient_id", ""),
            "candidate_id": row.get("candidate_id", ""),
            "original_score": orig,
            "adjusted_score_preview": adj,
            "score_delta": float(row.get("score_delta", orig - adj)),
            "score_delta_percent": delta_pct,
            "cause_class": cause,
            "visual_label": row.get("visual_label", ""),
            "local_z": row.get("local_z", ""),
            "y0": row.get("y0", ""),
            "x0": row.get("x0", ""),
            "validation_status": validation_status,
        })

    pen_val_df = pd.DataFrame(pen_val_rows)
    pen_val_df.to_csv(OUT_PENALIZED, index=False)
    print(f"  penalized rows: {pen_count}, rids: {pen_rids}")
    print(f"  non_penalized_count: {non_penalized_count:,}")
    print(f"  non_penalized_unchanged: {non_penalized_unchanged_count:,}")
    print(f"  non_penalized_changed: {non_penalized_changed_count}")

    if non_penalized_changed_count > 0:
        fail_reasons.append(f"STEP6: 비감점 행 중 adjusted_score_preview != original_score: {non_penalized_changed_count}행")

    # ── STEP 7: safety sentinel 검증 ─────────────────────────────────────────
    print("[STEP 7] safety sentinel 검증...")
    sent_df = pd.read_csv(SENTINEL_CSV)
    sent_df["must_not_penalize"] = sent_df["must_not_penalize"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False)

    must_protect = sent_df[sent_df["must_not_penalize"] == True]["review_id"].dropna().tolist()
    overlap = set(must_protect) & pen_rids
    sentinel_overlap_fail = len(overlap)
    if sentinel_overlap_fail > 0:
        fail_reasons.append(f"STEP7: sentinel 감점 overlap: {overlap}")
        print(f"  [FAIL] sentinel 보호 대상이 감점됨: {overlap}")
    else:
        print(f"  [OK] sentinel overlap 없음")

    sentinel_group_counts = {}
    for grp in SENTINEL_GROUPS:
        grp_rids = sent_df[sent_df["protection_group"] == grp]["review_id"].dropna().tolist()
        cnt = len(set(grp_rids) & pen_rids)
        sentinel_group_counts[grp] = cnt
        if cnt > 0:
            fail_reasons.append(f"STEP7: sentinel group '{grp}' 감점 {cnt}건")
            print(f"  [FAIL] {grp}: penalty_count={cnt}")

    # ── STEP 8: rank shift preview ───────────────────────────────────────────
    print("[STEP 8] rank shift preview...")
    rank_df = pd.read_csv(
        B1D4J_CSV,
        usecols=["original_score", "adjusted_score_preview", "soft_penalty_applied", "review_id"],
        low_memory=False,
    )
    rank_df["soft_penalty_applied"] = rank_df["soft_penalty_applied"].astype(str).str.lower() == "true"
    rank_df["original_rank_all"] = rank_df["original_score"].rank(method="min", ascending=False).astype(int)
    rank_df["adjusted_rank_all"] = rank_df["adjusted_score_preview"].rank(method="min", ascending=False).astype(int)
    rank_df["rank_worse_by"] = rank_df["adjusted_rank_all"] - rank_df["original_rank_all"]

    pen_rank = rank_df[rank_df["soft_penalty_applied"]].copy()
    rank_shift_summary = []
    for _, row in pen_rank.iterrows():
        rank_shift_summary.append({
            "review_id": row["review_id"],
            "original_score": round(float(row["original_score"]), 6),
            "adjusted_score_preview": round(float(row["adjusted_score_preview"]), 6),
            "original_rank_all": int(row["original_rank_all"]),
            "adjusted_rank_all": int(row["adjusted_rank_all"]),
            "rank_worse_by": int(row["rank_worse_by"]),
        })

    topk_shift_summary = {}
    for k in [50, 100, 500]:
        topk_shift_summary[f"top{k}_original_penalized_count"] = int((pen_rank["original_rank_all"] <= k).sum())
        topk_shift_summary[f"top{k}_adjusted_penalized_count"] = int((pen_rank["adjusted_rank_all"] <= k).sum())

    rankshift_out_df = pen_rank[["review_id", "original_score", "adjusted_score_preview",
                                  "original_rank_all", "adjusted_rank_all", "rank_worse_by"]]
    rankshift_out_df.to_csv(OUT_RANKSHIFT, index=False)
    print(f"  rank shift 계산 완료 (penalized {len(pen_rank)}개)")
    for r in rank_shift_summary:
        print(f"  {r['review_id']}: rank {r['original_rank_all']} → {r['adjusted_rank_all']} (+{r['rank_worse_by']})")

    # ── STEP 9: mtime 무수정 확인 ────────────────────────────────────────────
    print("[STEP 9] 입력 파일 mtime 무수정 확인...")
    mtime_after = {str(f): os.path.getmtime(f) for f in input_files}
    for k, v_before in mtime_before.items():
        v_after = mtime_after[k]
        if abs(v_after - v_before) > 0.001:
            msg = f"입력 파일 mtime 변경: {Path(k).name}"
            fail_reasons.append(f"STEP9: {msg}")
            print(f"  [FAIL] {msg}")
        else:
            print(f"  [OK] {Path(k).name}")

    # ── STEP 10: summary JSON ────────────────────────────────────────────────
    print("[STEP 10] summary JSON 생성...")
    fail_count = len(fail_reasons)
    verdict = "PASS" if fail_count == 0 else "FAIL"

    recommended_next_step = (
        "B1-D4l candidate-level threshold-effect preview preflight"
        if verdict == "PASS"
        else "fail_reasons 확인 후 재검증"
    )

    summary = {
        "stage": "B1-D4k",
        "stage2_holdout_access": 0,
        "input_output_row_match": input_output_row_match,
        "original_score_integrity": original_score_integrity,
        "output_rows": total_rows_b1d4j,
        "original_rows": total_rows_patch,
        "soft_penalty_applied_count": pen_count,
        "penalized_review_ids": sorted(pen_rids),
        "non_penalized_unchanged_count": non_penalized_unchanged_count,
        "non_penalized_changed_count": non_penalized_changed_count,
        "holdout_rows": holdout_flag_1_count,
        "nan_original_score_count": nan_original_score_count,
        "nan_adjusted_score_preview_count": nan_adjusted_score_preview_count,
        "forbidden_columns_present": present_forbidden,
        "missing_required_columns": missing_required,
        "mismatch_patient_id_count": mismatch_patient_id_count,
        "mismatch_candidate_id_count": mismatch_candidate_id_count,
        "mismatch_coord_count": mismatch_coord_count,
        "mismatch_original_score_count": mismatch_original_score_count,
        "safety_sentinel_validation": {
            "sentinel_overlap_fail": sentinel_overlap_fail,
            "must_protect_count": len(must_protect),
            "group_penalty_counts": sentinel_group_counts,
        },
        "rank_shift_summary": rank_shift_summary,
        "topk_shift_summary": topk_shift_summary,
        "score_modified": False,
        "threshold_recomputed": False,
        "adjusted_score_created": False,
        "adjusted_score_preview_created": True,
        "suppression_weight_created": False,
        "refined_score_created": False,
        "fail_count": fail_count,
        "fail_reasons": fail_reasons,
        "recommended_next_step": recommended_next_step,
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    with open(OUT_SUMMARY, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[OK] summary JSON: {OUT_SUMMARY}")

    # ── STEP 11: report MD ───────────────────────────────────────────────────
    print("[STEP 11] report MD 생성...")

    pen_table_rows = "\n".join(
        f"| {r['review_id']} | {r['original_score']:.4f} | {r['adjusted_score_preview']:.4f} "
        f"| {r['score_delta_percent']:.1f}% | {r['cause_class']} | {r['validation_status']} |"
        for r in pen_val_rows
    )
    rank_table_rows = "\n".join(
        f"| {r['review_id']} | {r['original_score']:.4f} | {r['adjusted_score_preview']:.4f} "
        f"| {r['original_rank_all']:,} | {r['adjusted_rank_all']:,} | +{r['rank_worse_by']:,} |"
        for r in rank_shift_summary
    )
    topk_lines = "\n".join(
        f"| top-{k} | "
        f"{topk_shift_summary.get(f'top{k}_original_penalized_count',0)} | "
        f"{topk_shift_summary.get(f'top{k}_adjusted_penalized_count',0)} |"
        for k in [50, 100, 500]
    )
    sentinel_grp_lines = "\n".join(
        f"| {g} | {sentinel_group_counts.get(g, 0)} |"
        for g in SENTINEL_GROUPS
    )
    fail_section = (
        "전체 검증 통과. fail_reasons 없음."
        if not fail_reasons
        else "\n".join(f"- {r}" for r in fail_reasons)
    )

    report_md = f"""# B1-D4k Rule-B3 stage1_dev candidate-level preview validation checkpoint

**판정: {verdict}**
**일시:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**소요 시간:** {summary['elapsed_seconds']}s

---

## 1. B1-D4j 실행 결과 요약

| 항목 | 값 |
|------|----|
| verdict | {b1d4j_sum.get('verdict')} |
| total_rows | {b1d4j_sum.get('total_rows'):,} |
| penalized_rows | {b1d4j_sum.get('penalized_rows')} |
| stage2_holdout_access | {b1d4j_sum.get('stage2_holdout_access')} |
| fail_count | {b1d4j_sum.get('fail_count')} |
| elapsed_seconds | {b1d4j_sum.get('elapsed_seconds')}s |

---

## 2. output CSV 무결성 검증

- 필수 컬럼 {len(REQUIRED_COLS)}개: {'모두 존재' if not missing_required else f'누락 {missing_required}'}
- 금지 컬럼: {'없음' if not present_forbidden else f'존재 → FAIL: {present_forbidden}'}
- adjusted_score 컬럼: {'없음 (OK)' if 'adjusted_score' not in present_forbidden else '존재 → FAIL'}
- suppression_weight 컬럼: {'없음 (OK)' if 'suppression_weight' not in present_forbidden else '존재 → FAIL'}
- refined_score 컬럼: {'없음 (OK)' if 'refined_score' not in present_forbidden else '존재 → FAIL'}

---

## 3. 원본 patch_candidates 1:1 대응 검증

| 항목 | 값 |
|------|----|
| b1d4j rows | {total_rows_b1d4j:,} |
| patch_candidates rows | {total_rows_patch:,} |
| row 수 일치 | {'OK' if input_output_row_match else 'FAIL'} |
| patient_id 불일치 | {mismatch_patient_id_count} |
| candidate_id 불일치 | {mismatch_candidate_id_count} |
| coord(local_z/y0/x0) 불일치 | {mismatch_coord_count} |
| original_score 불일치 | {mismatch_original_score_count} |
| holdout_flag=1 rows | {holdout_flag_1_count} |
| NaN original_score | {nan_original_score_count} |
| NaN adjusted_score_preview | {nan_adjusted_score_preview_count} |
| original_score 무결성 | {'OK' if original_score_integrity else 'FAIL'} |

---

## 4. 감점 4개 row 검증

| review_id | original_score | adjusted_score_preview | delta% | cause_class | status |
|-----------|---------------|----------------------|--------|------------|--------|
{pen_table_rows}

---

## 5. 비감점 761,202행 무변경 확인

| 항목 | 값 |
|------|----|
| 비감점 row 수 | {non_penalized_count:,} |
| adjusted_score_preview == original_score | {non_penalized_unchanged_count:,} |
| adjusted_score_preview != original_score | {non_penalized_changed_count} |
| 판정 | {'OK (전체 무변경)' if non_penalized_changed_count == 0 else 'FAIL'} |

---

## 6. safety sentinel 검증

**sentinel overlap fail: {sentinel_overlap_fail}**

| protection_group | penalty_count |
|-----------------|--------------|
{sentinel_grp_lines}

- must_protect 대상 수: {len(must_protect)}
- 감점된 sentinel: {'없음 (OK)' if sentinel_overlap_fail == 0 else f'FAIL: {overlap}'}

---

## 7. rank shift preview

| review_id | original_score | adjusted_score_preview | original_rank | adjusted_rank | rank_worse_by |
|-----------|---------------|----------------------|--------------|--------------|--------------|
{rank_table_rows}

### top-K 이동 (단순 이동 확인, 성능 해석 아님)

| 구간 | original top-K 내 penalized 수 | adjusted top-K 내 penalized 수 |
|------|-------------------------------|-------------------------------|
{topk_lines}

---

## 8. 한계

- 이 결과는 **candidate-level preview**이며 실제 성능 지표(AUROC/FROC)가 아니다.
- threshold 재계산을 수행하지 않았다.
- stage2_holdout 데이터를 사용하지 않았다.
- adjusted_score_preview는 실제 PaDiM score를 수정하지 않는다.
- rank shift는 단순 순위 이동 확인이며 성능 개선으로 해석하지 않는다.

---

## 9. fail 사유

{fail_section}

---

## 10. 다음 단계 권고

{'**B1-D4l** candidate-level threshold-effect preview preflight' if verdict == 'PASS' else '**fail_reasons 확인 후 재검증**'}
"""

    with open(OUT_REPORT, "w") as fh:
        fh.write(report_md)
    print(f"[OK] report MD: {OUT_REPORT}")

    # ── 최종 ─────────────────────────────────────────────────────────────────
    elapsed = round(time.time() - t_start, 1)
    print(f"\n[{verdict}] B1-D4k 완료 elapsed={elapsed}s fail_count={fail_count}")
    if fail_reasons:
        for r in fail_reasons:
            print(f"  - {r}")


if __name__ == "__main__":
    main()
