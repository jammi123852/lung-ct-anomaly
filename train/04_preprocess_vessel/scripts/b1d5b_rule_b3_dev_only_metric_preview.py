"""
B1-D5c Rule-B3 Dev-Only Metric Preview
stage1_dev + normal only, fixed threshold, threshold 재계산 없음
stage2_holdout 접근 금지, FROC/AUROC 금지

Usage:
  --dry-run   : 입력/schema/join/collision 검증만 (metric 계산 없음)
  --run       : 실제 metric preview 계산 (ALLOW_REAL_PROCESSING=True 필요)

importlib 방식 real 실행:
  import importlib.util, sys
  spec = importlib.util.spec_from_file_location("m", "scripts/b1d5b_rule_b3_dev_only_metric_preview.py")
  m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
  m.ALLOW_REAL_PROCESSING = True
  sys.argv = ["b1d5b_rule_b3_dev_only_metric_preview.py", "--run"]
  m.main()
"""

import sys
import os
import json
import csv
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# ─── 실행 차단 플래그 (importlib로만 True 허용) ──────────────────────────────
ALLOW_REAL_PROCESSING = False  # B1-D5c 실행 시 importlib 방식으로만 True

# ─── 경로/상수 ────────────────────────────────────────────────────────────────
BASE     = Path("outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1")
SCRIPTS  = Path("scripts")

B1D4J_CSV      = BASE / "b1d4j_rule_b3_soft_penalty_0_5_stage1_dev_candidate_preview.csv"
PATCH_CANDS    = Path("outputs/position-aware-padim-v1/candidates/padim_v2_roi0_0_explanation_candidates_v1/patch_candidates.csv")
THRESHOLD_JSON = Path("outputs/position-aware-padim-v1/evaluation/normal_v2_roi0_0/normal_v2_threshold.json")
SENTINEL_CSV   = BASE / "b1d4i_rule_b3_stage1_dev_full_preflight_safety_sentinels.csv"

OUT_SUMMARY      = BASE / "b1d5c_rule_b3_dev_only_metric_preview_summary.json"
OUT_REPORT       = BASE / "b1d5c_rule_b3_dev_only_metric_preview_report.md"
OUT_PATCH_CSV    = BASE / "b1d5c_rule_b3_patch_level_metric_preview.csv"
OUT_SLICE_CSV    = BASE / "b1d5c_rule_b3_slice_level_metric_preview.csv"
OUT_SENTINEL_CSV = BASE / "b1d5c_rule_b3_sentinel_metric_preview.csv"

EXPECTED_P95  = 14.092057666455288
EXPECTED_TOTAL = 761206
CHUNKSIZE     = 50000

# ─── 공통 유틸 ────────────────────────────────────────────────────────────────
def check_inputs():
    errors = []
    for p in [B1D4J_CSV, PATCH_CANDS, THRESHOLD_JSON, SENTINEL_CSV]:
        if not p.exists():
            errors.append(f"입력 없음: {p}")
    return errors

def load_threshold():
    with open(THRESHOLD_JSON) as f:
        d = json.load(f)
    p95 = d["threshold_p95"]
    p99 = d["threshold_p99"]
    assert abs(p95 - EXPECTED_P95) < 1e-9, f"p95 불일치: {p95}"
    return p95, p99

def _all_output_paths():
    finals = [OUT_SUMMARY, OUT_REPORT, OUT_PATCH_CSV, OUT_SLICE_CSV, OUT_SENTINEL_CSV]
    tmps   = [Path(str(p) + ".tmp") for p in finals]
    return finals + tmps

def check_output_collision():
    """최종 파일 + .tmp 파일 모두 검사."""
    collisions = [str(p) for p in _all_output_paths() if p.exists()]
    return collisions

def check_schema():
    issues = []
    df4j = pd.read_csv(B1D4J_CSV, nrows=0, low_memory=False)
    required_4j = {"patient_id","candidate_id","original_score","adjusted_score_preview",
                   "soft_penalty_applied","review_id","local_z","y0","x0",
                   "stage_split_safety_flag","holdout_flag"}
    missing = required_4j - set(df4j.columns)
    if missing: issues.append(f"b1d4j 컬럼 없음: {missing}")

    dfc = pd.read_csv(PATCH_CANDS, nrows=0)
    required_pc = {"candidate_patch_id","patient_id","label","group",
                   "local_z","y0","x0","stage_split_safety_flag"}
    missing_pc = required_pc - set(dfc.columns)
    if missing_pc: issues.append(f"patch_candidates 컬럼 없음: {missing_pc}")
    return issues

def check_join_sample(n=1000):
    """candidate_id == candidate_patch_id join 샘플 검증."""
    df4j = pd.read_csv(B1D4J_CSV, usecols=["candidate_id","stage_split_safety_flag","holdout_flag"],
                       nrows=n, low_memory=False)
    dfc  = pd.read_csv(PATCH_CANDS, usecols=["candidate_patch_id","label","group"], nrows=n)
    merged = df4j.merge(dfc, left_on="candidate_id", right_on="candidate_patch_id", how="left")
    unmatched = merged["candidate_patch_id"].isnull().sum()
    holdout_nonzero = (df4j["holdout_flag"] != 0).sum()
    labels = merged["label"].dropna().unique().tolist()
    stage_flags = merged["stage_split_safety_flag"].unique().tolist()
    return {
        "sample_n": n,
        "unmatched_count": int(unmatched),
        "holdout_nonzero": int(holdout_nonzero),
        "label_values": labels,
        "stage_flags": stage_flags,
        "join_ok": unmatched == 0 and holdout_nonzero == 0,
    }

def atomic_write(path, content_fn):
    """tmp 파일로 쓰고 성공 시 rename."""
    tmp = Path(str(path) + ".tmp")
    content_fn(tmp)
    tmp.rename(path)

def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    elif isinstance(obj, bool):
        return bool(obj)
    return obj

# ─── main() ──────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) == 1:
        print("bare-run 차단: --dry-run 또는 --run 필요")
        sys.exit(2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if args.run and not ALLOW_REAL_PROCESSING:
        print("BLOCKED: ALLOW_REAL_PROCESSING=False. 실제 실행 차단.")
        sys.exit(2)

    # ─── DRY-RUN ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print("[dry-run] 시작")
        result = {}

        # 1. 입력 파일
        errors = check_inputs()
        result["input_files_ok"] = len(errors) == 0
        result["input_errors"] = errors

        # 2. threshold
        try:
            p95, p99 = load_threshold()
            result["threshold_loaded"] = True
            result["p95"] = p95
            result["p99"] = p99
            result["threshold_recomputed"] = False
        except Exception as e:
            result["threshold_loaded"] = False
            result["threshold_error"] = str(e)

        # 3. schema
        schema_issues = check_schema()
        result["schema_ok"] = len(schema_issues) == 0
        result["schema_issues"] = schema_issues
        result["join_schema_ready"] = result["schema_ok"]

        # 4. join sample
        try:
            join_result = check_join_sample(1000)
            result["join_sample"] = join_result
            result["join_ok"] = join_result["join_ok"]
        except Exception as e:
            result["join_ok"] = False
            result["join_error"] = str(e)

        # 5. output collision (최종 + tmp 포함)
        collisions = check_output_collision()
        result["output_collision"] = collisions
        result["output_collision_ok"] = len(collisions) == 0

        # 6. holdout 전수
        df_flag = pd.read_csv(B1D4J_CSV, usecols=["stage_split_safety_flag","holdout_flag"], low_memory=False)
        holdout_nonzero = int((df_flag["holdout_flag"] != 0).sum())
        flag_counts = df_flag["stage_split_safety_flag"].value_counts().to_dict()
        result["holdout_nonzero_full"] = holdout_nonzero
        result["stage_flag_counts"] = {str(k): int(v) for k, v in flag_counts.items()}
        result["holdout_guard_ok"] = holdout_nonzero == 0

        # 7. slice grouping
        df_slice = pd.read_csv(B1D4J_CSV, usecols=["patient_id","local_z"], low_memory=False)
        slice_null = int(df_slice["local_z"].isnull().sum())
        slice_unique = int(df_slice.groupby(["patient_id","local_z"]).ngroups)
        result["slice_grouping_ok"] = slice_null == 0
        result["local_z_null"] = slice_null
        result["slice_unique_count"] = slice_unique

        # 8. chunk plan
        total_rows = len(df_flag)
        n_chunks = (total_rows + CHUNKSIZE - 1) // CHUNKSIZE
        result["total_rows"] = total_rows
        result["chunksize"] = CHUNKSIZE
        result["n_chunks"] = n_chunks
        result["chunk_plan_ok"] = total_rows == EXPECTED_TOTAL

        dry_pass = (
            result["input_files_ok"]
            and result.get("threshold_loaded", False)
            and result["schema_ok"]
            and result.get("join_ok", False)
            and result["output_collision_ok"]
            and result["holdout_guard_ok"]
            and result["slice_grouping_ok"]
            and result["chunk_plan_ok"]
        )
        result["dry_run_pass"] = dry_pass

        print(json.dumps(to_serializable(result), indent=2, ensure_ascii=False))
        sys.exit(0 if dry_pass else 1)

    # ─── REAL RUN ─────────────────────────────────────────────────────────────
    if args.run:
        # collision guard (최종 + tmp 포함)
        collisions = check_output_collision()
        if collisions:
            print(f"BLOCKED: 출력 파일 이미 존재 (tmp 포함): {collisions}")
            sys.exit(1)

        errors = check_inputs()
        if errors:
            print(f"BLOCKED: {errors}")
            sys.exit(1)

        p95, p99 = load_threshold()
        print(f"threshold: p95={p95}, p99={p99}")

        # sentinel 로드
        df_sent = pd.read_csv(SENTINEL_CSV)
        sentinel_review_ids = set(df_sent["review_id"].dropna().unique())

        # patch_candidates label map 로드
        print("patch_candidates 로드 ...")
        df_pc = pd.read_csv(PATCH_CANDS,
                            usecols=["candidate_patch_id","label","group"],
                            dtype={"candidate_patch_id": str, "label": str, "group": str})
        pc_map = df_pc.set_index("candidate_patch_id")["label"].to_dict()
        print(f"  label map size: {len(pc_map)}")

        # streaming 처리 카운터
        patch_counts = {
            "normal":     {"orig_p95":0,"adj_p95":0,"orig_p99":0,"adj_p99":0,"total":0},
            "stage1_dev": {"orig_p95":0,"adj_p95":0,"orig_p99":0,"adj_p99":0,"total":0},
        }
        non_pen_cross_down       = 0
        non_pen_cross_up         = 0
        non_penalized_changed_count = 0  # soft_penalty_applied=false인 row에서 adj != orig (exact)

        # full join consistency
        full_join_match_count   = 0
        full_join_unmatch_count = 0
        label_set               = set()
        stage_label_mismatch    = 0

        # slice-level 누적 dict: (patient_id, local_z, stage_flag) -> max
        slice_orig = {}
        slice_adj  = {}

        sentinel_rows       = []
        total_rows          = 0
        penalized_lesion_found = 0

        print(f"b1d4j 청크 처리 (chunksize={CHUNKSIZE}) ...")
        for chunk in pd.read_csv(
            B1D4J_CSV, chunksize=CHUNKSIZE,
            usecols=["candidate_id","patient_id","original_score","adjusted_score_preview",
                     "soft_penalty_applied","review_id","stage_split_safety_flag",
                     "holdout_flag","local_z","y0","x0"],
            low_memory=False,
        ):
            total_rows += len(chunk)

            # holdout guard
            if (chunk["holdout_flag"] != 0).any():
                print("BLOCKED: holdout_flag != 0 발견")
                sys.exit(1)

            # stage2_holdout row 발견 시 차단
            if chunk["stage_split_safety_flag"].isin(["stage2_holdout","holdout","test_holdout"]).any():
                print("BLOCKED: stage2_holdout row 발견")
                sys.exit(1)

            orig  = chunk["original_score"].values
            adj   = chunk["adjusted_score_preview"].values
            sflag = chunk["stage_split_safety_flag"].values
            pen   = chunk["soft_penalty_applied"].astype(str).str.lower().values
            cids  = chunk["candidate_id"].astype(str).values

            # patch-level 카운트
            for flag_key in ["normal", "stage1_dev"]:
                mask = sflag == flag_key
                if mask.sum() == 0:
                    continue
                o = orig[mask]; a = adj[mask]
                patch_counts[flag_key]["total"]    += int(mask.sum())
                patch_counts[flag_key]["orig_p95"] += int((o >= p95).sum())
                patch_counts[flag_key]["adj_p95"]  += int((a >= p95).sum())
                patch_counts[flag_key]["orig_p99"] += int((o >= p99).sum())
                patch_counts[flag_key]["adj_p99"]  += int((a >= p99).sum())

            # non-penalized stability (p95 crossing + exact equality)
            non_pen_mask = pen != "true"
            np_orig = orig[non_pen_mask]; np_adj = adj[non_pen_mask]
            non_pen_cross_down          += int(((np_orig >= p95) & (np_adj < p95)).sum())
            non_pen_cross_up            += int(((np_orig < p95)  & (np_adj >= p95)).sum())
            non_penalized_changed_count += int((np_orig != np_adj).sum())

            # penalized lesion 검사
            pen_mask = pen == "true"
            if pen_mask.sum() > 0:
                pen_flags = sflag[pen_mask]
                penalized_lesion_found += int((pen_flags == "stage1_dev").sum())

            # full join consistency (전수 검사)
            mapped_list = [pc_map.get(cid) for cid in cids]
            match_arr   = np.array([v is not None for v in mapped_list])
            mapped_arr  = np.array(mapped_list, dtype=object)
            full_join_match_count   += int(match_arr.sum())
            full_join_unmatch_count += int((~match_arr).sum())

            valid_labels = [v for v in mapped_list if v is not None]
            label_set.update(valid_labels)

            # stage_split_safety_flag vs label 일관성
            valid_sf  = sflag[match_arr]
            valid_lb  = mapped_arr[match_arr].astype(str)
            normal_v  = valid_sf == "normal"
            dev_v     = valid_sf == "stage1_dev"
            if normal_v.any():
                stage_label_mismatch += int((valid_lb[normal_v] != "normal").sum())
            if dev_v.any():
                stage_label_mismatch += int((valid_lb[dev_v] != "lesion_test").sum())

            # slice-level 누적
            for _, row in chunk[["patient_id","local_z","stage_split_safety_flag",
                                  "original_score","adjusted_score_preview"]].iterrows():
                key = (str(row["patient_id"]), int(row["local_z"]), str(row["stage_split_safety_flag"]))
                o_s = float(row["original_score"]); a_s = float(row["adjusted_score_preview"])
                if key not in slice_orig:
                    slice_orig[key] = o_s; slice_adj[key] = a_s
                else:
                    if o_s > slice_orig[key]: slice_orig[key] = o_s
                    if a_s > slice_adj[key]:  slice_adj[key]  = a_s

            # sentinel 수집
            if "review_id" in chunk.columns:
                sent_rows_chunk = chunk[chunk["review_id"].isin(sentinel_review_ids)]
                if len(sent_rows_chunk) > 0:
                    sentinel_rows.append(sent_rows_chunk[
                        ["patient_id","review_id","stage_split_safety_flag",
                         "original_score","adjusted_score_preview","soft_penalty_applied"]
                    ].copy())

        print(f"처리 완료: total_rows={total_rows}")

        if total_rows != EXPECTED_TOTAL:
            print(f"FAIL: total_rows={total_rows} != {EXPECTED_TOTAL}")
            sys.exit(1)
        if penalized_lesion_found > 0:
            print(f"FAIL: penalized_lesion_count={penalized_lesion_found}")
            sys.exit(1)

        # full join consistency 검증 결과 집계
        join_consistency = {
            "full_join_match_count": full_join_match_count,
            "full_join_unmatch_count": full_join_unmatch_count,
            "label_values_found": sorted(list(label_set)),
            "stage_label_mismatch_count": stage_label_mismatch,
            "full_join_ok": full_join_unmatch_count == 0 and stage_label_mismatch == 0,
        }

        # 집계
        total_orig_p95 = patch_counts["normal"]["orig_p95"] + patch_counts["stage1_dev"]["orig_p95"]
        total_adj_p95  = patch_counts["normal"]["adj_p95"]  + patch_counts["stage1_dev"]["adj_p95"]
        total_orig_p99 = patch_counts["normal"]["orig_p99"] + patch_counts["stage1_dev"]["orig_p99"]
        total_adj_p99  = patch_counts["normal"]["adj_p99"]  + patch_counts["stage1_dev"]["adj_p99"]

        # slice-level 집계
        sl_rows = []
        sl_n_orig_p95 = sl_a_p95 = sl_n_orig_p99 = sl_a_p99 = 0
        sl_lesion_cross_down_p95 = sl_lesion_cross_down_p99 = 0

        for (pid, lz, sflag_k), o_max in slice_orig.items():
            a_max = slice_adj[(pid, lz, sflag_k)]
            orig_p95 = o_max >= p95; adj_p95 = a_max >= p95
            orig_p99 = o_max >= p99; adj_p99 = a_max >= p99
            p95_cross = orig_p95 and not adj_p95
            p99_cross = orig_p99 and not adj_p99

            sl_rows.append({
                "patient_id": pid, "local_z": lz, "stage_split_safety_flag": sflag_k,
                "max_score_original": round(o_max, 6), "max_score_adjusted": round(a_max, 6),
                "orig_above_p95": orig_p95, "adj_above_p95": adj_p95, "p95_cross_down": p95_cross,
                "orig_above_p99": orig_p99, "adj_above_p99": adj_p99, "p99_cross_down": p99_cross,
            })
            if orig_p95: sl_n_orig_p95 += 1
            if adj_p95:  sl_a_p95  += 1
            if orig_p99: sl_n_orig_p99 += 1
            if adj_p99:  sl_a_p99  += 1
            if sflag_k == "stage1_dev" and p95_cross: sl_lesion_cross_down_p95 += 1
            if sflag_k == "stage1_dev" and p99_cross: sl_lesion_cross_down_p99 += 1

        # sentinel 집계 (p95 + p99 모두)
        sentinel_pen       = 0
        sentinel_cross_p95 = 0
        sentinel_cross_p99 = 0
        sent_out_rows = []
        if sentinel_rows:
            df_sent_merged = pd.concat(sentinel_rows, ignore_index=True)
            for _, row in df_sent_merged.iterrows():
                is_pen  = str(row.get("soft_penalty_applied","")).lower() == "true"
                o_s     = float(row["original_score"]); a_s = float(row["adjusted_score_preview"])
                cross_p95 = bool((o_s >= p95 and a_s < p95) or (o_s < p95 and a_s >= p95))
                cross_p99 = bool((o_s >= p99 and a_s < p99) or (o_s < p99 and a_s >= p99))
                if is_pen:    sentinel_pen       += 1
                if cross_p95: sentinel_cross_p95 += 1
                if cross_p99: sentinel_cross_p99 += 1
                sent_out_rows.append({
                    "review_id": row["review_id"],
                    "stage_split_safety_flag": row["stage_split_safety_flag"],
                    "original_score": round(o_s, 6), "adjusted_score_preview": round(a_s, 6),
                    "soft_penalty_applied": is_pen,
                    "p95_crossing": cross_p95,
                    "p99_crossing": cross_p99,
                    "match": "FAIL" if (is_pen or cross_p95 or cross_p99) else "OK",
                })

        fail_reasons = []
        if non_pen_cross_down > 0:
            fail_reasons.append(f"non_pen_cross_down={non_pen_cross_down}")
        if non_pen_cross_up > 0:
            fail_reasons.append(f"non_pen_cross_up={non_pen_cross_up}")
        if non_penalized_changed_count > 0:
            fail_reasons.append(f"non_penalized_changed_count={non_penalized_changed_count}")
        if penalized_lesion_found > 0:
            fail_reasons.append(f"penalized_lesion_count={penalized_lesion_found}")
        if sentinel_pen > 0:
            fail_reasons.append(f"sentinel_penalty_count={sentinel_pen}")
        if sentinel_cross_p95 > 0:
            fail_reasons.append(f"sentinel_p95_crossing_count={sentinel_cross_p95}")
        if sentinel_cross_p99 > 0:
            fail_reasons.append(f"sentinel_p99_crossing_count={sentinel_cross_p99}")
        if not join_consistency["full_join_ok"]:
            fail_reasons.append(
                f"full_join_unmatch={full_join_unmatch_count}, stage_label_mismatch={stage_label_mismatch}"
            )

        verdict = "PASS" if len(fail_reasons) == 0 else "NEEDS_FIX"

        summary = {
            "stage": "B1-D5c",
            "stage2_holdout_access": 0,
            "threshold_source": str(THRESHOLD_JSON),
            "p95_threshold": p95, "p99_threshold": p99, "threshold_recomputed": False,
            "total_rows": total_rows,
            "normal_rows": patch_counts["normal"]["total"],
            "lesion_rows": patch_counts["stage1_dev"]["total"],
            "patch_level_original_p95_count": total_orig_p95,
            "patch_level_adjusted_p95_count": total_adj_p95,
            "patch_level_p95_delta": total_adj_p95 - total_orig_p95,
            "patch_level_original_p99_count": total_orig_p99,
            "patch_level_adjusted_p99_count": total_adj_p99,
            "patch_level_p99_delta": total_adj_p99 - total_orig_p99,
            "normal_patch_orig_p95":  patch_counts["normal"]["orig_p95"],
            "normal_patch_adj_p95":   patch_counts["normal"]["adj_p95"],
            "normal_patch_p95_delta": patch_counts["normal"]["adj_p95"] - patch_counts["normal"]["orig_p95"],
            "normal_patch_orig_p99":  patch_counts["normal"]["orig_p99"],
            "normal_patch_adj_p99":   patch_counts["normal"]["adj_p99"],
            "lesion_patch_orig_p95":  patch_counts["stage1_dev"]["orig_p95"],
            "lesion_patch_adj_p95":   patch_counts["stage1_dev"]["adj_p95"],
            "lesion_patch_p95_delta": patch_counts["stage1_dev"]["adj_p95"] - patch_counts["stage1_dev"]["orig_p95"],
            "lesion_patch_orig_p99":  patch_counts["stage1_dev"]["orig_p99"],
            "lesion_patch_adj_p99":   patch_counts["stage1_dev"]["adj_p99"],
            "slice_level_total_slices":      len(sl_rows),
            "slice_level_original_p95_count": sl_n_orig_p95,
            "slice_level_adjusted_p95_count": sl_a_p95,
            "slice_level_p95_delta":          sl_a_p95 - sl_n_orig_p95,
            "slice_level_original_p99_count": sl_n_orig_p99,
            "slice_level_adjusted_p99_count": sl_a_p99,
            "slice_level_p99_delta":          sl_a_p99 - sl_n_orig_p99,
            "lesion_slice_cross_down_p95":    sl_lesion_cross_down_p95,
            "lesion_slice_cross_down_p99":    sl_lesion_cross_down_p99,
            "non_penalized_cross_down":       non_pen_cross_down,
            "non_penalized_cross_up":         non_pen_cross_up,
            "non_penalized_changed_count":    non_penalized_changed_count,
            "penalized_lesion_count":         penalized_lesion_found,
            "sentinel_penalty_count":         sentinel_pen,
            "sentinel_p95_crossing_count":    sentinel_cross_p95,
            "sentinel_p99_crossing_count":    sentinel_cross_p99,
            "full_join_consistency":          join_consistency,
            "score_modified": False, "adjusted_score_created": False,
            "metric_scope": "dev_only_preview",
            "froc_computed": False, "auroc_computed": False,
            "fail_count": len(fail_reasons),
            "fail_reasons": fail_reasons,
            "verdict": verdict,
        }

        # ── output atomicity: CSV/report 먼저, summary 마지막 ──────────────────

        # 1. patch-level CSV
        patch_rows = []
        for flag_key, lbl in [("normal","normal"),("stage1_dev","stage1_dev_lesion"),("all","all")]:
            for thr_lbl, ok_field, aj_field in [("p95","orig_p95","adj_p95"),("p99","orig_p99","adj_p99")]:
                if flag_key == "all":
                    orig_c = patch_counts["normal"][ok_field] + patch_counts["stage1_dev"][ok_field]
                    adj_c  = patch_counts["normal"][aj_field] + patch_counts["stage1_dev"][aj_field]
                else:
                    orig_c = patch_counts[flag_key][ok_field]
                    adj_c  = patch_counts[flag_key][aj_field]
                patch_rows.append({
                    "split": lbl, "threshold_level": thr_lbl,
                    "original_exceed": orig_c, "adjusted_exceed": adj_c,
                    "delta": adj_c - orig_c,
                })

        def write_patch_csv(path):
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["split","threshold_level","original_exceed","adjusted_exceed","delta"])
                writer.writeheader(); writer.writerows(patch_rows)
        atomic_write(OUT_PATCH_CSV, write_patch_csv)
        print(f"[OUT] {OUT_PATCH_CSV}")

        # 2. slice-level CSV
        sl_df = pd.DataFrame(sl_rows)
        atomic_write(OUT_SLICE_CSV, lambda p: sl_df.to_csv(p, index=False))
        print(f"[OUT] {OUT_SLICE_CSV} ({len(sl_df)}행)")

        # 3. sentinel CSV (p95 + p99 crossing 포함)
        sent_fieldnames = ["review_id","stage_split_safety_flag","original_score",
                           "adjusted_score_preview","soft_penalty_applied",
                           "p95_crossing","p99_crossing","match"]

        def write_sentinel_csv(path):
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sent_fieldnames)
                writer.writeheader()
                if sent_out_rows:
                    writer.writerows(sent_out_rows)
        atomic_write(OUT_SENTINEL_CSV, write_sentinel_csv)
        print(f"[OUT] {OUT_SENTINEL_CSV}")

        # 4. report MD
        report = f"""# B1-D5c Rule-B3 Dev-Only Metric Preview

## 설정

- threshold source: `{THRESHOLD_JSON}`
- p95 = {p95}, p99 = {p99}
- threshold_recomputed = False
- stage2_holdout_access = 0
- metric_scope = dev_only_preview

## Patch-Level 결과

| split | threshold | original | adjusted | delta |
|---|---|---|---|---|
| normal | p95 | {patch_counts['normal']['orig_p95']:,} | {patch_counts['normal']['adj_p95']:,} | {patch_counts['normal']['adj_p95']-patch_counts['normal']['orig_p95']:+d} |
| normal | p99 | {patch_counts['normal']['orig_p99']:,} | {patch_counts['normal']['adj_p99']:,} | {patch_counts['normal']['adj_p99']-patch_counts['normal']['orig_p99']:+d} |
| stage1_dev (lesion) | p95 | {patch_counts['stage1_dev']['orig_p95']:,} | {patch_counts['stage1_dev']['adj_p95']:,} | {patch_counts['stage1_dev']['adj_p95']-patch_counts['stage1_dev']['orig_p95']:+d} |
| stage1_dev (lesion) | p99 | {patch_counts['stage1_dev']['orig_p99']:,} | {patch_counts['stage1_dev']['adj_p99']:,} | {patch_counts['stage1_dev']['adj_p99']-patch_counts['stage1_dev']['orig_p99']:+d} |
| **all** | p95 | {total_orig_p95:,} | {total_adj_p95:,} | **{total_adj_p95-total_orig_p95:+d}** |
| **all** | p99 | {total_orig_p99:,} | {total_adj_p99:,} | **{total_adj_p99-total_orig_p99:+d}** |

## Slice-Level 결과

- total unique slices: {len(sl_rows):,}
- p95 exceedance: {sl_n_orig_p95:,} → {sl_a_p95:,} (delta={sl_a_p95-sl_n_orig_p95:+d})
- p99 exceedance: {sl_n_orig_p99:,} → {sl_a_p99:,} (delta={sl_a_p99-sl_n_orig_p99:+d})
- lesion slice p95 cross_down: {sl_lesion_cross_down_p95}
- lesion slice p99 cross_down: {sl_lesion_cross_down_p99}

## Lesion Recall Risk

- penalized_lesion_count: **{penalized_lesion_found}**
- lesion_patch_p95_delta: {patch_counts['stage1_dev']['adj_p95']-patch_counts['stage1_dev']['orig_p95']:+d}
- lesion_slice_cross_down_p95: **{sl_lesion_cross_down_p95}**
- B1-D4j preview 기준 직접 감점된 lesion candidate는 0개. lesion score 직접 변경 risk = 0으로 확인.
- 실제 lesion recall은 이 결과로 단정하지 않는다.

## Safety

- non_penalized_cross_down: {non_pen_cross_down}
- non_penalized_cross_up: {non_pen_cross_up}
- non_penalized_changed_count: {non_penalized_changed_count}
- sentinel_penalty_count: {sentinel_pen}
- sentinel_p95_crossing_count: {sentinel_cross_p95}
- sentinel_p99_crossing_count: {sentinel_cross_p99}

## Full Join Consistency

- full_join_match_count: {full_join_match_count:,}
- full_join_unmatch_count: {full_join_unmatch_count}
- label_values_found: {sorted(list(label_set))}
- stage_label_mismatch_count: {stage_label_mismatch}
- full_join_ok: {join_consistency['full_join_ok']}

## 한계

- candidate pool 내부 count (전체 patch metric 아님)
- FROC/AUROC 미계산
- recall/precision 미계산
- threshold 재계산 없음
- stage2_holdout 미사용
- R015 여전히 p95/p99 위

## 판정

**{verdict}**
"""

        def write_report(path):
            with open(path, "w") as f:
                f.write(report)
        atomic_write(OUT_REPORT, write_report)
        print(f"[OUT] {OUT_REPORT}")

        # 5. summary JSON (마지막)
        def write_summary(path):
            with open(path, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        atomic_write(OUT_SUMMARY, write_summary)
        print(f"[OUT] {OUT_SUMMARY}")

        print(f"\n=== 판정: {verdict} ===")
        if fail_reasons:
            for r in fail_reasons: print(f"  FAIL: {r}")


if __name__ == "__main__":
    main()
