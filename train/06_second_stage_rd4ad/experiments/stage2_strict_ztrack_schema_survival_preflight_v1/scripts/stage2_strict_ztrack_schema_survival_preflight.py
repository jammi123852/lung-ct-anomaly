"""
stage2_strict_ztrack_schema_survival_preflight.py

목적: stage2_holdout candidate manifest에 대해 strict same-position z-track
      schema 검증 및 min_run_len=2 z-track survival preflight 수행.

실행:
  dry-run:    python ... --dry-run
  preflight:  python ... --run-preflight --confirm-readonly --confirm-stage2-holdout-eval-only
  bare run:   exit 2
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT     = Path(__file__).resolve().parents[1]

# ── 입력
INPUT_CSV = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/datasets"
    / "s6a_stage2_holdout_candidate_coordinate_manifest_v1.csv"
)
CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)
MASK_ROOT = (
    PROJECT_ROOT
    / "outputs/mip-postprocess-research-v1/masks"
    / "refined_roi_v4_20_modeB_all_v1"
)

# ── 출력
MANIFESTS_DIR = EXP_ROOT / "manifests"
REPORTS_DIR   = EXP_ROOT / "reports"
LOGS_DIR      = EXP_ROOT / "logs"

CAND_SURVIVAL_CSV   = MANIFESTS_DIR / "stage2_ztrack_candidate_survival_minrun2.csv"
TRACK_MANIFEST_CSV  = MANIFESTS_DIR / "stage2_ztrack_manifest_minrun2.csv"
VARIANT_SUMMARY_CSV = MANIFESTS_DIR / "stage2_ztrack_variant_summary.csv"
PATIENT_SUMMARY_CSV = MANIFESTS_DIR / "stage2_ztrack_patient_survival_summary.csv"
REPORT_MD           = REPORTS_DIR   / "stage2_strict_ztrack_schema_survival_preflight_report.md"
SUMMARY_JSON        = REPORTS_DIR   / "stage2_strict_ztrack_schema_survival_preflight_summary.json"
ERRORS_CSV          = LOGS_DIR      / "errors.csv"
DONE_JSON           = EXP_ROOT      / "DONE.json"

GUARDRAILS = {
    "stage2_holdout_accessed":                   True,
    "stage2_holdout_used_for_method_tuning":      False,
    "model_forward_executed":                     False,
    "checkpoint_loaded":                          False,
    "crop_generation_executed":                   False,
    "rd4ad_scoring_executed":                     False,
    "threshold_recalculated":                     False,
    "score_original_used_for_candidate_deletion": False,
    "score_original_used_for_topz_selection":     False,
    "label_used_for_evaluation_only":             True,
    "label_used_for_track_creation":              False,
    "xy_radius_grouping_used":                    False,
    "representative_only_scoring_used":           False,
    "hard_filter_applied":                        False,
    "vessel_mask_used":                           False,
    "existing_artifact_modified":                 False,
    "existing_script_modified":                   False,
    "output_overwrite":                           False,
}

MIN_RUN_LEN_PRIMARY = 2
CROP_SIZE           = 96
CT_SAMPLE_N         = 5

REQUIRED_INPUT_COLS = [
    "patient_id", "safe_id", "local_z",
    "y0", "x0", "y1", "x1",
    "label", "stage_split",
]


# ── z-track 알고리즘 ──────────────────────────────────────────────────────────

def consecutive_runs(sorted_z):
    """정렬된 z 배열에서 연속 run의 (z_start, z_end) 리스트 반환."""
    if len(sorted_z) == 0:
        return []
    runs = []
    start = prev = int(sorted_z[0])
    for z in sorted_z[1:]:
        z = int(z)
        if z == prev + 1:
            prev = z
        else:
            runs.append((start, prev))
            start = prev = z
    runs.append((start, prev))
    return runs


def build_track_index(df, min_run_len=2):
    """
    strict same-position (patient_id, y0, x0, y1, x1) 기반 z-track 생성.
    Returns:
        survived_idx : set of DataFrame index values that survived
        track_id_map : DataFrame index → track_id
        tracks       : list of track-level dicts
    """
    pos_z    = defaultdict(list)   # poskey → [z, ...]
    pos_rows = defaultdict(list)   # poskey → [df_idx, ...]
    pos_meta = {}                  # poskey → safe_id

    for idx, row in df.iterrows():
        key = (str(row["patient_id"]),
               int(row["y0"]), int(row["x0"]),
               int(row["y1"]), int(row["x1"]))
        pos_z[key].append(int(row["local_z"]))
        pos_rows[key].append(idx)
        if key not in pos_meta:
            pos_meta[key] = str(row["safe_id"])

    survived_idx = set()
    track_id_map = {}
    tracks       = []

    def _is_pos(val):
        return val in (1, "positive")
    def _is_hn(val):
        return val in (0, "hard_negative")

    for key, zlist in pos_z.items():
        pid, y0, x0, y1, x1 = key
        safe_id = pos_meta[key]

        z_idx_pairs = sorted(zip(zlist, pos_rows[key]))
        sorted_z    = [p[0] for p in z_idx_pairs]
        z_to_dfidx  = {p[0]: p[1] for p in z_idx_pairs}

        runs = consecutive_runs(sorted_z)

        for (z_start, z_end) in runs:
            run_len = z_end - z_start + 1
            if run_len < min_run_len:
                continue

            track_z   = list(range(z_start, z_end + 1))
            track_idx = [z_to_dfidx[z] for z in track_z]

            tid = f"{pid}|{y0}_{x0}_{y1}_{x1}|{z_start}_{z_end}"
            survived_idx.update(track_idx)
            for i in track_idx:
                track_id_map[i] = tid

            labels    = df.loc[track_idx, "label"].tolist()
            n_pos     = sum(1 for l in labels if _is_pos(l))
            n_hn      = sum(1 for l in labels if _is_hn(l))

            tracks.append({
                "track_id":        tid,
                "patient_id":      pid,
                "safe_id":         safe_id,
                "position_source": "y0_x0_y1_x1_as_crop_coords",
                "y0": y0, "x0": x0, "y1": y1, "x1": x1,
                "crop_y0": y0, "crop_x0": x0, "crop_y1": y1, "crop_x1": x1,
                "z_start":         z_start,
                "z_end":           z_end,
                "track_len":       run_len,
                "n_candidates":    run_len,
                "n_positive":      n_pos,
                "n_hard_negative": n_hn,
                "has_positive":    n_pos > 0,
                "positive_ratio":  round(n_pos / run_len, 4),
            })

    return survived_idx, track_id_map, tracks


# ── CT readiness 샘플 확인 ────────────────────────────────────────────────────

def check_ct_readiness(df, n_sample=CT_SAMPLE_N):
    results = []
    sample_sids = df["safe_id"].unique()[:n_sample]
    for sid in sample_sids:
        ct_path = CT_ROOT / sid / "ct_hu.npy"
        if not ct_path.exists():
            results.append({"safe_id": sid, "ct_exists": False,
                            "z_ok": False, "hw_ok": False,
                            "error": "ct_hu.npy not found"})
            continue
        try:
            ct  = np.load(ct_path, mmap_mode="r")
            Z, H, W = ct.shape
            sub = df[df["safe_id"] == sid]
            max_z = int(sub["local_z"].max())
            min_z = int(sub["local_z"].min())
            max_y1 = int(sub["y1"].max())
            max_x1 = int(sub["x1"].max())
            z_ok  = (min_z >= 1) and (max_z + 1 <= Z - 1)  # medi3ch: z-1..z+1
            hw_ok = (max_y1 <= H) and (max_x1 <= W)
            results.append({
                "safe_id":    sid,
                "ct_exists":  True,
                "ct_shape":   f"{Z}×{H}×{W}",
                "z_range":    f"{min_z}-{max_z}",
                "z_ok":       bool(z_ok),
                "hw_ok":      bool(hw_ok),
                "error":      None,
            })
        except Exception as e:
            results.append({"safe_id": sid, "ct_exists": True,
                            "z_ok": False, "hw_ok": False, "error": str(e)})
    return results


# ── dry-run ───────────────────────────────────────────────────────────────────

def run_dry_run():
    errs = []

    print("[DRY-RUN] 입력 파일 확인...")
    if not INPUT_CSV.exists():
        errs.append(f"입력 없음: {INPUT_CSV}")
    else:
        print(f"  ✓ {INPUT_CSV.name}")

    print("[DRY-RUN] 컬럼 확인...")
    if INPUT_CSV.exists():
        df5 = pd.read_csv(INPUT_CSV, nrows=5)
        missing = [c for c in REQUIRED_INPUT_COLS if c not in df5.columns]
        if missing:
            errs.append(f"필수 컬럼 누락: {missing}")
        else:
            print(f"  ✓ 필수 컬럼 모두 존재")
        if "candidate_id" not in df5.columns:
            print("  [INFO] candidate_id 없음 → row_id 로 대체 예정")

    print("[DRY-RUN] 출력 충돌 확인...")
    for p in [CAND_SURVIVAL_CSV, TRACK_MANIFEST_CSV, SUMMARY_JSON, DONE_JSON]:
        if p.exists():
            errs.append(f"출력 충돌: {p}")
    if not errs:
        print("  ✓ 출력 충돌 없음")

    print("[DRY-RUN] CT root 확인...")
    if not CT_ROOT.exists():
        errs.append(f"CT root 없음: {CT_ROOT}")
    else:
        print(f"  ✓ {CT_ROOT}")

    print("[DRY-RUN] guardrail 확인...")
    for k in ("model_forward_executed", "checkpoint_loaded", "crop_generation_executed",
              "rd4ad_scoring_executed", "existing_artifact_modified"):
        v = GUARDRAILS[k]
        ok = "✓" if not v else "✗"
        print(f"  {ok} {k}: {v}")

    if errs:
        print(f"\n[DRY-RUN] FAIL ({len(errs)} errors)")
        for e in errs:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("\n[DRY-RUN] PASS → --run-preflight 실행 가능")


# ── actual preflight ──────────────────────────────────────────────────────────

def run_preflight():
    errs = []

    for d in [MANIFESTS_DIR, REPORTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("[1] 입력 로드...")
    df_raw = pd.read_csv(INPUT_CSV)
    print(f"  전체: {len(df_raw):,}행  {df_raw['patient_id'].nunique()}명")

    # candidate_id 처리
    if "candidate_id" not in df_raw.columns:
        if "row_id" in df_raw.columns:
            df_raw["candidate_id"] = df_raw["row_id"].astype(str)
            print("  [INFO] candidate_id → row_id 사용")
        else:
            df_raw["candidate_id"] = df_raw.index.astype(str)
            print("  [INFO] candidate_id → index 사용")

    # stage2_holdout only
    n_before = len(df_raw)
    df = df_raw[df_raw["stage_split"] == "stage2_holdout"].copy().reset_index(drop=True)
    print(f"  stage2_holdout 필터: {n_before:,} → {len(df):,}행")

    print("\n[2] Schema 확인...")
    missing_cols = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing_cols:
        errs.append(f"필수 컬럼 누락: {missing_cols}")
        print(f"  ✗ 누락: {missing_cols}")
    else:
        print("  ✓ 필수 컬럼 모두 존재")

    label_dist = df["label"].value_counts().to_dict()
    print(f"  label 분포: {label_dist}")
    print(f"  환자 수: {df['patient_id'].nunique()}")

    # score_original 통계
    score_stats = {}
    if "score_original" in df.columns:
        s = df["score_original"].dropna()
        score_stats = {
            "count": int(len(s)),
            "mean": round(float(s.mean()), 4),
            "min":  round(float(s.min()),  4),
            "p25":  round(float(s.quantile(0.25)), 4),
            "p50":  round(float(s.quantile(0.50)), 4),
            "p75":  round(float(s.quantile(0.75)), 4),
            "p95":  round(float(s.quantile(0.95)), 4),
            "max":  round(float(s.max()),  4),
        }
        print(f"  score_original: mean={score_stats['mean']}, "
              f"min={score_stats['min']}, max={score_stats['max']}")
        print(f"  [확인] score_original = EfficientNet 1차 PaDiM 이상점수 (기록만, 삭제/필터 금지)")

    print("\n[3] Crop size 검증...")
    df["y_size"] = df["y1"] - df["y0"]
    df["x_size"] = df["x1"] - df["x0"]
    y_valid   = int((df["y_size"] == CROP_SIZE).sum())
    x_valid   = int((df["x_size"] == CROP_SIZE).sum())
    y_invalid = int((df["y_size"] != CROP_SIZE).sum())
    x_invalid = int((df["x_size"] != CROP_SIZE).sum())
    print(f"  y_size==96: {y_valid:,} / y_size!=96: {y_invalid:,}")
    print(f"  x_size==96: {x_valid:,} / x_size!=96: {x_invalid:,}")
    if y_invalid > 0 or x_invalid > 0:
        bad = df[df["y_size"] != CROP_SIZE][["y0","y1","y_size"]].head(3)
        print(f"  [WARNING] 비정상 샘플:\n{bad.to_string()}")

    # position 반복 확인
    pos_z_counts = df.groupby(["patient_id","y0","x0","y1","x1"])["local_z"].nunique()
    multi_z_count = int((pos_z_counts >= 2).sum())
    print(f"\n[4] Position 반복 (z≥2): {multi_z_count:,} / {len(pos_z_counts):,}")

    # candidate_id 중복
    dup_cid = int(df["candidate_id"].duplicated().sum())
    print(f"  candidate_id 중복: {dup_cid}")
    if dup_cid > 0:
        errs.append(f"candidate_id 중복: {dup_cid}")

    print("\n[5] CT readiness (샘플 5명)...")
    ct_results = check_ct_readiness(df, n_sample=CT_SAMPLE_N)
    for r in ct_results:
        ok = r.get("ct_exists") and r.get("z_ok") and r.get("hw_ok") and not r.get("error")
        print(f"  {'✓' if ok else '✗'} {r['safe_id']}: "
              f"exists={r.get('ct_exists')}  shape={r.get('ct_shape','-')}  "
              f"z_ok={r.get('z_ok')}  hw_ok={r.get('hw_ok')}  err={r.get('error')}")
    ct_fail = sum(1 for r in ct_results if not r.get("ct_exists") or r.get("error"))

    print("\n[6] crop_y0/x0/y1/x1 alias 생성...")
    df["crop_y0"] = df["y0"]
    df["crop_x0"] = df["x0"]
    df["crop_y1"] = df["y1"]
    df["crop_x1"] = df["x1"]
    df["position_source"] = "y0_x0_y1_x1_as_crop_coords"
    print("  ✓ position_source = y0_x0_y1_x1_as_crop_coords")

    print("\n[7] strict z-track (min_run_len=2) 생성...")
    survived_idx, track_id_map, tracks = build_track_index(df, min_run_len=MIN_RUN_LEN_PRIMARY)
    print(f"  survived candidates: {len(survived_idx):,} / {len(df):,}")
    print(f"  tracks: {len(tracks):,}")

    df["survived"]          = df.index.isin(survived_idx)
    df["ztrack_min_run_len"] = MIN_RUN_LEN_PRIMARY
    df["track_id"]          = df.index.map(lambda i: track_id_map.get(i, None))

    survived_df = df[df["survived"]].copy()

    print("\n[8] 지표 계산...")

    def is_pos(v): return v in (1, "positive")
    def is_hn(v):  return v in (0, "hard_negative")

    orig_pos = int(df["label"].apply(is_pos).sum())
    orig_hn  = int(df["label"].apply(is_hn).sum())
    surv_pos = int(survived_df["label"].apply(is_pos).sum())
    surv_hn  = int(survived_df["label"].apply(is_hn).sum())

    orig_pos_pts = set(df[df["label"].apply(is_pos)]["patient_id"])
    surv_pos_pts = set(survived_df[survived_df["label"].apply(is_pos)]["patient_id"])
    miss_pts     = orig_pos_pts - surv_pos_pts

    orig_pos_slices = int(df[df["label"].apply(is_pos)].groupby("patient_id")["local_z"].nunique().sum())
    surv_pos_slices = int(survived_df[survived_df["label"].apply(is_pos)].groupby("patient_id")["local_z"].nunique().sum())

    track_lens = [t["track_len"] for t in tracks]

    metrics = {
        "original_candidate_count":          len(df),
        "survived_candidate_count":          len(survived_df),
        "candidate_reduction_rate":          round(1 - len(survived_df)/len(df), 4),
        "survived_track_count":              len(tracks),
        "track_len_min":                     int(min(track_lens)) if track_lens else 0,
        "track_len_median":                  float(np.median(track_lens)) if track_lens else 0,
        "track_len_mean":                    round(float(np.mean(track_lens)), 2) if track_lens else 0,
        "track_len_max":                     int(max(track_lens)) if track_lens else 0,
        "original_positive_candidate":       orig_pos,
        "survived_positive_candidate":       surv_pos,
        "positive_candidate_retention":      round(surv_pos/orig_pos, 4) if orig_pos > 0 else 0,
        "original_hard_negative_candidate":  orig_hn,
        "survived_hard_negative_candidate":  surv_hn,
        "hard_negative_reduction_rate":      round(1 - surv_hn/orig_hn, 4) if orig_hn > 0 else 0,
        "original_positive_slice_count":     orig_pos_slices,
        "survived_positive_slice_count":     surv_pos_slices,
        "positive_slice_coverage":           round(surv_pos_slices/orig_pos_slices, 4) if orig_pos_slices > 0 else 0,
        "original_positive_patient_count":   len(orig_pos_pts),
        "survived_positive_patient_count":   len(surv_pos_pts),
        "positive_patient_coverage":         round(len(surv_pos_pts)/len(orig_pos_pts), 4) if orig_pos_pts else 0,
        "complete_miss_patient_count":       len(miss_pts),
        "complete_miss_patients":            sorted(miss_pts),
        "label_distribution":                {str(k): int(v) for k,v in label_dist.items()},
        "crop_y_valid_count":                y_valid,
        "crop_y_invalid_count":              y_invalid,
        "crop_x_valid_count":                x_valid,
        "crop_x_invalid_count":              x_invalid,
        "coordinate_source":                 "y0_x0_y1_x1_as_crop_coords",
        "ct_readiness_sample":               ct_results,
        "score_original_stats":              score_stats,
        "stage2_holdout_patient_count":      df["patient_id"].nunique(),
        "multi_z_position_count":            multi_z_count,
        "candidate_id_duplicates":           dup_cid,
    }

    for k, v in metrics.items():
        if not isinstance(v, (list, dict)):
            print(f"  {k}: {v}")

    print("\n[9] variant summary (참고용, primary=2 고정)...")
    variant_rows = []
    for R in [2, 3, 4, 5]:
        si_r, _, tr_r = build_track_index(df, min_run_len=R)
        surv_r   = df.index.isin(si_r)
        surv_pos_r = int(df[surv_r]["label"].apply(is_pos).sum())
        variant_rows.append({
            "min_run_len":        R,
            "survived_candidates": len(si_r),
            "survived_tracks":     len(tr_r),
            "survived_positive":   surv_pos_r,
            "positive_retention":  round(surv_pos_r/orig_pos, 4) if orig_pos > 0 else 0,
            "is_primary":          R == MIN_RUN_LEN_PRIMARY,
            "note":                "primary" if R == MIN_RUN_LEN_PRIMARY
                                   else "reference_only_not_for_stage2_tuning",
        })
        print(f"  min_run={R}: survived={len(si_r):,}  tracks={len(tr_r):,}  "
              f"pos_retention={variant_rows[-1]['positive_retention']}")
    variant_df = pd.DataFrame(variant_rows)

    # patient survival summary
    patient_rows = []
    for pid in sorted(df["patient_id"].unique()):
        psub  = df[df["patient_id"] == pid]
        psurv = survived_df[survived_df["patient_id"] == pid]
        pos_s    = int(psub["label"].apply(is_pos).sum())
        pos_surv = int(psurv["label"].apply(is_pos).sum()) if len(psurv) > 0 else 0
        patient_rows.append({
            "patient_id":          pid,
            "total_candidates":    len(psub),
            "survived_candidates": len(psurv),
            "positive_candidates": pos_s,
            "survived_positive":   pos_surv,
            "positive_retention":  round(pos_surv/pos_s, 4) if pos_s > 0 else None,
            "is_complete_miss":    pid in miss_pts,
        })
    patient_df = pd.DataFrame(patient_rows)

    print("\n[10] 저장...")

    out_cols = [
        "candidate_id", "patient_id", "safe_id", "local_z",
        "y0", "x0", "y1", "x1",
        "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "label", "survived", "ztrack_min_run_len", "track_id", "position_source",
    ]
    if "score_original" in df.columns:
        out_cols.insert(out_cols.index("label"), "score_original")

    df[out_cols].to_csv(CAND_SURVIVAL_CSV, index=False)
    print(f"  → {CAND_SURVIVAL_CSV} ({len(df):,}행)")

    track_df = pd.DataFrame(tracks)
    track_df.to_csv(TRACK_MANIFEST_CSV, index=False)
    print(f"  → {TRACK_MANIFEST_CSV} ({len(track_df):,}행)")

    variant_df.to_csv(VARIANT_SUMMARY_CSV, index=False)
    print(f"  → {VARIANT_SUMMARY_CSV}")

    patient_df.to_csv(PATIENT_SUMMARY_CSV, index=False)
    print(f"  → {PATIENT_SUMMARY_CSV}")

    pd.DataFrame([{"error": e} for e in errs]).to_csv(ERRORS_CSV, index=False)
    print(f"  → {ERRORS_CSV} ({len(errs)} errors)")

    # ── 판정
    crop_ok = (y_invalid == 0 and x_invalid == 0)
    ct_ok   = (ct_fail == 0)

    if (not missing_cols and crop_ok and dup_cid == 0 and ct_ok
            and len(miss_pts) == 0 and len(tracks) > 0):
        verdict = "PASS"
    elif not missing_cols and len(tracks) > 0:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "FAIL"

    print(f"\n판정: {verdict}")
    if errs:
        for e in errs:
            print(f"  ✗ {e}")

    # ── summary JSON
    summary = {
        "verdict":    verdict,
        "metrics":    metrics,
        "guardrails": GUARDRAILS,
        "errors":     errs,
        "next_step":  "stage2_strict_ztrack_rd4ad_scoring_v1"
                      if verdict in ("PASS", "PARTIAL_PASS") else None,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  → {SUMMARY_JSON}")

    with open(DONE_JSON, "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, "guardrails": GUARDRAILS}, f, indent=2)
    print(f"  → {DONE_JSON}")

    # ── report.md
    md = [
        f"# Stage2 Strict Z-Track Schema/Survival Preflight v1\n\n",
        f"**판정: {verdict}**\n\n",
        "## Guardrail\n\n",
        "| key | value |\n|-----|-------|\n",
    ]
    for k, v in GUARDRAILS.items():
        md.append(f"| {k} | {v} |\n")
    md += ["\n## 지표 요약\n\n", "| 항목 | 값 |\n|------|----|\n"]
    for k, v in metrics.items():
        if not isinstance(v, (list, dict)):
            md.append(f"| {k} | {v} |\n")
    md += ["\n## Variant Summary (참고용)\n\n",
           "primary=min_run_len=2 고정, stage2에서 이 수치로 primary를 재선택하지 않는다.\n\n"]
    md.append(variant_df.to_markdown(index=False) + "\n")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.writelines(md)
    print(f"  → {REPORT_MD}")

    print(f"\n[완료] 판정={verdict}")
    return verdict


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--run-preflight", action="store_true")
    parser.add_argument("--confirm-readonly", action="store_true")
    parser.add_argument("--confirm-stage2-holdout-eval-only", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not args.run_preflight:
        print("[BLOCKED] bare run 금지. --dry-run 또는 --run-preflight --confirm-readonly "
              "--confirm-stage2-holdout-eval-only 로 실행하세요.")
        sys.exit(2)

    if args.run_preflight:
        if not args.confirm_readonly or not args.confirm_stage2_holdout_eval_only:
            print("[BLOCKED] --confirm-readonly --confirm-stage2-holdout-eval-only 둘 다 필요합니다.")
            sys.exit(2)

    if args.dry_run:
        run_dry_run()
    else:
        run_preflight()


if __name__ == "__main__":
    main()
