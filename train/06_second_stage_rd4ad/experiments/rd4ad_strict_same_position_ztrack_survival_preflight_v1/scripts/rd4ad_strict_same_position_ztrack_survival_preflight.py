"""
RD4AD strict same-position z-track survival preflight v1

목적:
  1차 PaDiM candidate(stage1_dev 전체 113,447) 중에서
  "같은 환자 + 같은 patch 위치(strict)" 에서 local_z 가 한 slice 도 끊기지 않고
  연속(consecutive)된 run(track) 만 남겼을 때
  병변 candidate / 병변 slice / 병변 patient 가 얼마나 살아남는지 read-only 로 확인한다.

이전 group-level 방식과의 차이:
  - 이전: z_gap=1, xy_radius=24 fuzzy connected component → 대표 1개만 scoring (low top-k 실패)
  - 이번: strict 동일 position key + z 연속 run 만 track 인정, track 안 candidate 전부 보존
          (이번 preflight 는 scoring 안 함, survival 만 계산)

엄격 금지:
  model forward / checkpoint load / crop 생성 / full scoring /
  first_stage_score threshold / top-z 선택 / xy_radius grouping /
  representative 1개 선택 / 후보 삭제 실제 적용 / stage2_holdout 접근 /
  기존 artifact 수정.

label 은 survival 평가용으로만 사용한다(track 생성에 사용 금지).

실행:
  bare run: exit 2
  dry-run:       --dry-run
  preflight:     --run-preflight --confirm-readonly --confirm-stage1dev-only
"""
import argparse
import csv
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

# =============================================================================
# 경로
# =============================================================================

PROJECT_ROOT    = Path("/home/jinhy/project/lung-ct-anomaly")
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/rd4ad_strict_same_position_ztrack_survival_preflight_v1"

# 입력 (read-only)
CANDIDATE_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c2_effb0_v420_candidate_rd4ad_retest_v1"
    / "rd_c2_effb0_v420_candidate_manifest.csv"
)

# 출력 (새 폴더에만)
MANIFEST_DIR = EXPERIMENT_ROOT / "manifests"
REPORT_DIR   = EXPERIMENT_ROOT / "reports"
LOG_DIR      = EXPERIMENT_ROOT / "logs"

VARIANT_SUMMARY_CSV   = MANIFEST_DIR / "ztrack_variant_summary.csv"
TRACK_MANIFEST_CSV    = {2: MANIFEST_DIR / "ztrack_manifest_minrun2.csv",
                         3: MANIFEST_DIR / "ztrack_manifest_minrun3.csv"}
CAND_SURVIVAL_CSV     = {2: MANIFEST_DIR / "ztrack_candidate_survival_minrun2.csv",
                         3: MANIFEST_DIR / "ztrack_candidate_survival_minrun3.csv"}
PATIENT_SURVIVAL_CSV  = MANIFEST_DIR / "ztrack_patient_survival_summary.csv"
PROBLEM_AUDIT_CSV     = MANIFEST_DIR / "ztrack_problem_patient_audit.csv"
ERROR_CSV             = LOG_DIR / "errors.csv"
REPORT_MD             = REPORT_DIR / "rd4ad_strict_same_position_ztrack_survival_preflight_report.md"
SUMMARY_JSON          = REPORT_DIR / "rd4ad_strict_same_position_ztrack_survival_preflight_summary.json"
DONE_JSON             = EXPERIMENT_ROOT / "DONE.json"

# 분석 파라미터
MIN_RUN_LENS = [2, 3, 4, 5]
VARIANT_NAME = {2: "T1_minrun2", 3: "T2_minrun3", 4: "T3_minrun4", 5: "T4_minrun5"}
PRIMARY_RUNS = [2, 3]  # track manifest / candidate survival CSV 생성 대상
PROBLEM_PATIENT_IDS = ["LUNG1-086", "LUNG1-386", "LUNG1-399"]

# position key 컬럼 우선순위 (candidate manifest 기준)
POSITION_COLUMN_SETS = [
    ["y0", "x0", "y1", "x1"],
    ["patch_y0", "patch_x0", "patch_y1", "patch_x1"],
    ["stage1_patch_y0", "stage1_patch_x0", "stage1_patch_y1", "stage1_patch_x1"],
    ["crop_y0", "crop_x0", "crop_y1", "crop_x1"],
]

# RD4AD forward 예상 비교 기준
PATCH_BASELINE_COUNT = 113447
GROUP_REP_COUNT      = 20216
# shard 0 실측: 4,913 group 대표 forward ≈ 84.2s (estimate only)
SEC_PER_FORWARD = 84.2 / 4913.0

# 판정 임계
MEANINGFUL_REDUCTION = 0.05  # candidate reduction "의미 있게 존재" 하한

# =============================================================================
# guardrail
# =============================================================================

GUARDRAILS = {
    "stage2_holdout_accessed":                  False,
    "model_forward_executed":                   False,
    "checkpoint_loaded":                        False,
    "crop_generation_executed":                 False,
    "training_executed":                        False,
    "backward_executed":                        False,
    "optimizer_created":                        False,
    "checkpoint_saved":                         False,
    "full_scoring_executed":                    False,
    "threshold_recalculated":                   False,
    "existing_artifact_modified":               False,
    "existing_script_modified":                 False,
    "output_overwrite":                         False,
    "first_stage_score_used_for_candidate_deletion": False,
    "label_used_for_evaluation_only":           True,
    "label_used_for_track_creation":            False,
    "xy_radius_grouping_used":                  False,
    "representative_only_scoring_used":         False,
}

errors = []

# =============================================================================
# 안전 경로 / IO
# =============================================================================

def assert_path_safe(p: Path):
    s = str(p).lower()
    if "stage2_holdout" in s or ("stage2" in s and "holdout" in s):
        GUARDRAILS["stage2_holdout_accessed"] = True
        raise RuntimeError(f"[ABORT] stage2_holdout 경로 접근 차단: {p}")


def ensure_output_path_safe(p: Path):
    rp = Path(p).resolve()
    exp_root = EXPERIMENT_ROOT.resolve()
    if not str(rp).startswith(str(exp_root)):
        GUARDRAILS["existing_artifact_modified"] = True
        raise RuntimeError(f"[ABORT] 실험 폴더 외부 쓰기 차단: {p}")


def read_csv(path: Path):
    assert_path_safe(path)
    rows = []
    with open(str(path), encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def read_csv_header(path: Path):
    assert_path_safe(path)
    with open(str(path), encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


def write_csv(path: Path, fieldnames, rows):
    ensure_output_path_safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  saved: {path} ({len(rows)} rows)")


def log_error(msg, exc=None):
    errors.append(msg)
    print(f"  [ERROR] {msg}")


def save_error_csv():
    ensure_output_path_safe(ERROR_CSV)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(ERROR_CSV), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "message"])
        if not errors:
            w.writerow(["", "no errors"])
        else:
            for i, m in enumerate(errors):
                w.writerow([i, m])


# =============================================================================
# 통계 유틸
# =============================================================================

def percentiles(values):
    """min, p25, p50, p75, p90, p95, max (numpy 사용). 빈 입력은 모두 None."""
    if not values:
        return {k: None for k in ["min", "p25", "p50", "p75", "p90", "p95", "max", "mean", "n"]}
    import numpy as np
    arr = np.asarray(values, dtype=float)
    return {
        "min":  float(arr.min()),
        "p25":  float(np.percentile(arr, 25)),
        "p50":  float(np.percentile(arr, 50)),
        "p75":  float(np.percentile(arr, 75)),
        "p90":  float(np.percentile(arr, 90)),
        "p95":  float(np.percentile(arr, 95)),
        "max":  float(arr.max()),
        "mean": float(arr.mean()),
        "n":    int(arr.size),
    }


def consecutive_runs(sorted_z):
    """정렬된 unique z 리스트 -> [(start_z, end_z, length), ...] (연속 run)."""
    runs = []
    if not sorted_z:
        return runs
    start = prev = sorted_z[0]
    for z in sorted_z[1:]:
        if z == prev + 1:
            prev = z
            continue
        runs.append((start, prev, prev - start + 1))
        start = prev = z
    runs.append((start, prev, prev - start + 1))
    return runs


# =============================================================================
# 핵심: candidate 로드 + strict position z-track 구성
# =============================================================================

def detect_position_columns(header):
    for cols in POSITION_COLUMN_SETS:
        if all(c in header for c in cols):
            src = "patch_coords" if cols[0] in ("y0", "patch_y0", "stage1_patch_y0") else "crop_coords"
            return cols, src
    return None, None


def load_candidates(position_cols):
    """stage1_dev candidate 로드. 각 candidate: id, patient, z, poskey, label."""
    rows = read_csv(CANDIDATE_MANIFEST_CSV)
    cands = []
    for r in rows:
        if r.get("stage_split") != "stage1_dev":
            continue
        try:
            z = int(r["local_z"])
            pos = tuple(int(r[c]) for c in position_cols)
        except Exception as e:
            log_error(f"row parse fail cid={r.get('candidate_id')}: {e}")
            continue
        cands.append({
            "candidate_id": r.get("candidate_id", ""),
            "patient_id":   r.get("patient_id", ""),
            "local_z":      z,
            "poskey":       pos,
            "label":        r.get("label", ""),
        })
    return cands


def build_runlen_index(cands):
    """
    (patient, poskey) 별 z->해당 z 가 속한 연속 run 의 길이 매핑,
    그리고 (patient, poskey) 별 run 목록 반환.
    """
    by_pos = defaultdict(set)  # (patient, poskey) -> set(z)
    for c in cands:
        by_pos[(c["patient_id"], c["poskey"])].add(c["local_z"])

    z_runlen = {}      # (patient, poskey) -> {z: runlen}
    pos_runs = {}      # (patient, poskey) -> [(start,end,len), ...]
    for key, zset in by_pos.items():
        sorted_z = sorted(zset)
        runs = consecutive_runs(sorted_z)
        pos_runs[key] = runs
        m = {}
        for (s, e, L) in runs:
            for z in range(s, e + 1):
                m[z] = L
        z_runlen[key] = m
    return z_runlen, pos_runs


# =============================================================================
# 한 min_run_len 에 대한 survival 분석
# =============================================================================

def analyze_variant(cands, z_runlen, pos_runs, R):
    """R(min_run_len) 기준 survival 통계 dict 반환."""
    # candidate survival flag
    survived_flag = {}  # candidate_id -> bool
    for c in cands:
        key = (c["patient_id"], c["poskey"])
        runlen = z_runlen[key].get(c["local_z"], 1)
        survived_flag[c["candidate_id"]] = (runlen >= R)

    n_orig = len(cands)
    n_surv = sum(1 for c in cands if survived_flag[c["candidate_id"]])

    # label 기준
    orig_pos = sum(1 for c in cands if c["label"] == "positive")
    orig_hn  = sum(1 for c in cands if c["label"] == "hard_negative")
    surv_pos = sum(1 for c in cands if c["label"] == "positive" and survived_flag[c["candidate_id"]])
    surv_hn  = sum(1 for c in cands if c["label"] == "hard_negative" and survived_flag[c["candidate_id"]])

    # slice survival: (patient, z)
    orig_pos_slice = set()
    surv_pos_slice = set()
    for c in cands:
        if c["label"] != "positive":
            continue
        sl = (c["patient_id"], c["local_z"])
        orig_pos_slice.add(sl)
        if survived_flag[c["candidate_id"]]:
            surv_pos_slice.add(sl)

    # patient 기준
    orig_pos_pat = set(c["patient_id"] for c in cands if c["label"] == "positive")
    surv_pos_pat = set(c["patient_id"] for c in cands
                       if c["label"] == "positive" and survived_flag[c["candidate_id"]])

    # per-patient: positive slice coverage, survived candidate count, track count
    pat_orig_pos_slice = defaultdict(set)
    pat_surv_pos_slice = defaultdict(set)
    pat_surv_cand = defaultdict(int)
    for c in cands:
        if survived_flag[c["candidate_id"]]:
            pat_surv_cand[c["patient_id"]] += 1
        if c["label"] == "positive":
            pat_orig_pos_slice[c["patient_id"]].add(c["local_z"])
            if survived_flag[c["candidate_id"]]:
                pat_surv_pos_slice[c["patient_id"]].add(c["local_z"])

    # tracks (len>=R) — track 단위 통계
    # track id: (patient, poskey, start, end)
    pos_cand = defaultdict(lambda: defaultdict(list))  # (patient,poskey) -> z -> [cand,...]
    for c in cands:
        pos_cand[(c["patient_id"], c["poskey"])][c["local_z"]].append(c)

    tracks = []  # dict per surviving track
    pat_track_count = defaultdict(int)
    for key, runs in pos_runs.items():
        patient, poskey = key
        for (s, e, L) in runs:
            if L < R:
                continue
            zlist = list(range(s, e + 1))
            tcands = []
            for z in zlist:
                tcands.extend(pos_cand[key].get(z, []))
            n_pos = sum(1 for tc in tcands if tc["label"] == "positive")
            n_hn  = sum(1 for tc in tcands if tc["label"] == "hard_negative")
            has_pos = n_pos > 0
            pat_track_count[patient] += 1
            tracks.append({
                "track_id":      f"{patient}|{poskey[0]}_{poskey[1]}_{poskey[2]}_{poskey[3]}|{s}_{e}",
                "patient_id":    patient,
                "pos_y0": poskey[0], "pos_x0": poskey[1], "pos_y1": poskey[2], "pos_x1": poskey[3],
                "z_start": s, "z_end": e, "track_len": L,
                "n_candidates": len(tcands),
                "n_positive": n_pos, "n_hard_negative": n_hn,
                "has_positive": has_pos,
                "positive_ratio": round(n_pos / len(tcands), 4) if tcands else 0.0,
            })

    # track-level positive 분석
    has_pos_tracks = [t for t in tracks if t["has_positive"]]
    hn_only_tracks = [t for t in tracks if not t["has_positive"]]
    mixed_tracks   = [t for t in tracks if t["n_positive"] > 0 and t["n_hard_negative"] > 0]

    # per-patient coverage 분포
    cov_list = []
    cov_lt50 = cov_lt80 = cov_lt95 = cov_eq100 = complete_miss = 0
    for pid in orig_pos_pat:
        o = len(pat_orig_pos_slice[pid])
        sv = len(pat_surv_pos_slice[pid])
        cov = sv / o if o else 0.0
        cov_list.append(cov)
        if sv == 0:
            complete_miss += 1
        if cov < 0.50:
            cov_lt50 += 1
        if cov < 0.80:
            cov_lt80 += 1
        if cov < 0.95:
            cov_lt95 += 1
        if cov >= 1.0:
            cov_eq100 += 1

    forward_count = n_surv
    return {
        "variant": VARIANT_NAME[R],
        "min_run_len": R,
        # overall
        "original_candidate_count": n_orig,
        "survived_candidate_count": n_surv,
        "candidate_reduction_rate": round(1 - n_surv / n_orig, 4) if n_orig else 0.0,
        "survived_track_count": len(tracks),
        "track_len_dist": percentiles([t["track_len"] for t in tracks]),
        "patient_track_count_dist": percentiles(list(pat_track_count.values())),
        "patient_survived_candidate_dist": percentiles(list(pat_surv_cand.values())),
        # lesion candidate
        "original_positive_candidate": orig_pos,
        "survived_positive_candidate": surv_pos,
        "positive_candidate_retention": round(surv_pos / orig_pos, 4) if orig_pos else 0.0,
        "original_hard_negative_candidate": orig_hn,
        "survived_hard_negative_candidate": surv_hn,
        "hard_negative_retention": round(surv_hn / orig_hn, 4) if orig_hn else 0.0,
        "hard_negative_reduction_rate": round(1 - surv_hn / orig_hn, 4) if orig_hn else 0.0,
        # lesion slice
        "original_positive_slice": len(orig_pos_slice),
        "survived_positive_slice": len(surv_pos_slice),
        "positive_slice_coverage": round(len(surv_pos_slice) / len(orig_pos_slice), 4) if orig_pos_slice else 0.0,
        "patient_pos_slice_coverage_dist": percentiles(cov_list),
        "patients_coverage_lt50": cov_lt50,
        "patients_coverage_lt80": cov_lt80,
        "patients_coverage_lt95": cov_lt95,
        "patients_coverage_eq100": cov_eq100,
        "complete_miss_patient": complete_miss,
        # lesion patient
        "original_positive_patient": len(orig_pos_pat),
        "survived_positive_patient": len(surv_pos_pat),
        "positive_patient_coverage": round(len(surv_pos_pat) / len(orig_pos_pat), 4) if orig_pos_pat else 0.0,
        # track-level positive
        "has_positive_track_count": len(has_pos_tracks),
        "all_hard_negative_track_count": len(hn_only_tracks),
        "positive_containing_track_ratio": round(len(has_pos_tracks) / len(tracks), 4) if tracks else 0.0,
        "positive_track_len_dist": percentiles([t["track_len"] for t in has_pos_tracks]),
        "hn_only_track_len_dist": percentiles([t["track_len"] for t in hn_only_tracks]),
        "positive_track_posratio_dist": percentiles([t["positive_ratio"] for t in has_pos_tracks]),
        "mixed_track_ratio": round(len(mixed_tracks) / len(tracks), 4) if tracks else 0.0,
        # forward estimate
        "forward_candidate_count": forward_count,
        "forward_reduction_vs_patch_baseline": round(1 - forward_count / PATCH_BASELINE_COUNT, 4),
        "forward_vs_group_rep_delta": forward_count - GROUP_REP_COUNT,
        "forward_vs_group_rep_ratio": round(forward_count / GROUP_REP_COUNT, 4),
        "estimated_runtime_sec": round(forward_count * SEC_PER_FORWARD, 1),
        "estimated_runtime_min": round(forward_count * SEC_PER_FORWARD / 60.0, 1),
        # internal (CSV 생성용)
        "_survived_flag": survived_flag,
        "_tracks": tracks,
        "_pat_orig_pos_slice": pat_orig_pos_slice,
        "_pat_surv_pos_slice": pat_surv_pos_slice,
        "_pat_surv_cand": pat_surv_cand,
        "_pat_track_count": pat_track_count,
        "_orig_pos_pat": orig_pos_pat,
    }


# =============================================================================
# dry-run
# =============================================================================

def run_dry():
    print("=" * 70)
    print("[DRY-RUN] strict same-position z-track survival preflight v1")
    print("=" * 70)
    issues = []

    print("\n[1] 입력 파일 확인 (read-only)")
    ok = CANDIDATE_MANIFEST_CSV.exists()
    print(f"  [{'OK' if ok else 'MISSING'}] candidate manifest: {CANDIDATE_MANIFEST_CSV}")
    if not ok:
        issues.append("candidate manifest 없음")

    print("\n[2] position key 컬럼 확인")
    if ok:
        header = read_csv_header(CANDIDATE_MANIFEST_CSV)
        cols, src = detect_position_columns(header)
        if cols is None:
            issues.append("REQUIRED_POSITION_COLUMNS_MISSING")
            print("  [FAIL] position key 컬럼 없음 (y0/patch_/stage1_patch_/crop_ 모두 부재)")
        else:
            print(f"  [OK] position columns = {cols}  (source={src})")
        for need in ["local_z", "stage_split", "label", "patient_id", "candidate_id"]:
            print(f"  [{'OK' if need in header else 'MISSING'}] {need}")
            if need not in header:
                issues.append(f"필수 컬럼 없음: {need}")

    print("\n[3] stage2_holdout 접근 없음")
    print(f"  stage2_holdout_accessed: {GUARDRAILS['stage2_holdout_accessed']}")

    print("\n[4] 출력 충돌 확인 (생성 예정 파일)")
    for p in [VARIANT_SUMMARY_CSV, PATIENT_SURVIVAL_CSV, PROBLEM_AUDIT_CSV, REPORT_MD, SUMMARY_JSON, DONE_JSON]:
        print(f"  {'WARN exists' if p.exists() else 'OK'}: {p.name}")

    print("\n[5] guardrail (dry-run: forward/checkpoint/crop/파일생성 없음)")
    for k in ["model_forward_executed", "checkpoint_loaded", "crop_generation_executed",
              "full_scoring_executed", "existing_artifact_modified", "output_overwrite"]:
        print(f"  {k}: {GUARDRAILS[k]}")

    print("\n" + "=" * 70)
    if issues:
        print("[DRY-RUN] 이슈:")
        for it in issues:
            print(f"  - {it}")
        print("판정: NEEDS_FIX")
    else:
        print("[DRY-RUN] 입력/컬럼/경로 OK. preflight 실행 준비됨.")
        print("판정: READY_TO_RUN_PREFLIGHT")
    print("=" * 70)


# =============================================================================
# preflight
# =============================================================================

def run_preflight():
    print("=" * 70)
    print("[PREFLIGHT] strict same-position z-track survival v1")
    print("=" * 70)
    t0 = time.perf_counter()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    critical_error = False

    # 1. position 컬럼
    print("\n[1] position key 컬럼 감지")
    header = read_csv_header(CANDIDATE_MANIFEST_CSV)
    position_cols, position_source = detect_position_columns(header)
    if position_cols is None:
        log_error("REQUIRED_POSITION_COLUMNS_MISSING")
        critical_error = True
        _finalize_fail(t0, "REQUIRED_POSITION_COLUMNS_MISSING")
        return
    print(f"  position columns = {position_cols} (source={position_source})")

    # 2. candidate 로드
    print("\n[2] candidate 로드 (stage1_dev)")
    cands = load_candidates(position_cols)
    print(f"  candidates: {len(cands):,}")
    if not cands:
        log_error("no stage1_dev candidates")
        critical_error = True
        _finalize_fail(t0, "NO_CANDIDATES")
        return

    # 3. run-length index
    print("\n[3] strict (patient, position) z-run index 구성")
    z_runlen, pos_runs = build_runlen_index(cands)
    print(f"  unique (patient, position) keys: {len(pos_runs):,}")

    # 4. variant 분석
    print("\n[4] min_run_len 2/3/4/5 survival 분석")
    variants = {}
    for R in MIN_RUN_LENS:
        v = analyze_variant(cands, z_runlen, pos_runs, R)
        variants[R] = v
        print(f"  {v['variant']}: surv_cand={v['survived_candidate_count']:,} "
              f"reduction={v['candidate_reduction_rate']:.4f} "
              f"pos_cand_ret={v['positive_candidate_retention']:.4f} "
              f"pos_slice_cov={v['positive_slice_coverage']:.4f} "
              f"pos_pat_cov={v['positive_patient_coverage']:.4f} "
              f"complete_miss={v['complete_miss_patient']}")

    # 5. 출력 CSV
    print("\n[5] 출력 CSV 생성")
    _write_variant_summary(variants)
    for R in PRIMARY_RUNS:
        _write_track_manifest(variants[R], R)
        _write_candidate_survival(cands, variants[R], R)
    _write_patient_survival(variants)
    problem_summary = _write_problem_audit(cands, variants, pos_runs)

    # 6. 판정
    verdict, reasons = _decide_verdict(variants, problem_summary)

    # 7. summary / report / DONE
    elapsed = time.perf_counter() - t0
    _write_summary_json(variants, problem_summary, position_cols, position_source,
                        verdict, reasons, elapsed, critical_error)
    _write_report_md(variants, problem_summary, position_cols, position_source,
                     verdict, reasons, elapsed)
    save_error_csv()
    _write_done(verdict)

    # 8. 최종 보고
    print("\n" + "=" * 70)
    print(f"[PREFLIGHT] 완료 ({elapsed:.1f}s)  판정: {verdict}")
    for r in reasons:
        print(f"  - {r}")
    print("=" * 70)


def _decide_verdict(variants, problem_summary):
    reasons = []
    best_verdict = "FAIL"
    for R in PRIMARY_RUNS:
        v = variants[R]
        pat_cov = v["positive_patient_coverage"]
        slice_cov = v["positive_slice_coverage"]
        cmiss = v["complete_miss_patient"]
        reduction = v["candidate_reduction_rate"]
        prob_ok = all(problem_summary[v["variant"]][pid]["positive_slice_coverage"] > 0
                      for pid in PROBLEM_PATIENT_IDS)

        if (pat_cov >= 1.0 and slice_cov >= 0.95 and cmiss == 0 and prob_ok
                and reduction >= MEANINGFUL_REDUCTION):
            reasons.append(f"{v['variant']}: PASS 조건 충족 (pat_cov=1.0, slice_cov={slice_cov:.4f}, "
                           f"complete_miss=0, problem 3명 cov>0, reduction={reduction:.4f})")
            best_verdict = "PASS_CANDIDATE"
            break
        elif pat_cov >= 1.0 and 0.90 <= slice_cov < 0.95:
            reasons.append(f"{v['variant']}: PARTIAL (pat_cov=1.0, slice_cov={slice_cov:.4f} in [0.90,0.95))")
            if best_verdict != "PASS_CANDIDATE":
                best_verdict = "PARTIAL_PASS_EXPLORATORY"
        elif pat_cov >= 1.0 and not prob_ok:
            reasons.append(f"{v['variant']}: PARTIAL (pat_cov=1.0 이나 problem 환자 일부 미회복)")
            if best_verdict != "PASS_CANDIDATE":
                best_verdict = "PARTIAL_PASS_EXPLORATORY"
        elif pat_cov >= 1.0 and reduction < MEANINGFUL_REDUCTION:
            reasons.append(f"{v['variant']}: PARTIAL (reduction 약함 {reduction:.4f} < {MEANINGFUL_REDUCTION})")
            if best_verdict != "PASS_CANDIDATE":
                best_verdict = "PARTIAL_PASS_EXPLORATORY"
        else:
            reasons.append(f"{v['variant']}: FAIL 신호 (pat_cov={pat_cov:.4f}, slice_cov={slice_cov:.4f}, "
                           f"complete_miss={cmiss})")
    if GUARDRAILS["stage2_holdout_accessed"] or GUARDRAILS["existing_artifact_modified"]:
        reasons.append("guardrail 위반 → FAIL")
        best_verdict = "FAIL"
    return best_verdict, reasons


# =============================================================================
# 출력 작성기
# =============================================================================

def _flat_dist(prefix, dist):
    out = {}
    for k, v in dist.items():
        out[f"{prefix}_{k}"] = v
    return out


def _write_variant_summary(variants):
    rows = []
    for R in MIN_RUN_LENS:
        v = variants[R]
        row = {k: v[k] for k in v if not k.startswith("_") and not k.endswith("_dist")}
        # 분포는 p50/p90/max 만 평탄화해 함께 기록
        for distkey in ["track_len_dist", "patient_track_count_dist", "patient_survived_candidate_dist",
                        "patient_pos_slice_coverage_dist", "positive_track_len_dist",
                        "hn_only_track_len_dist", "positive_track_posratio_dist"]:
            d = v[distkey]
            for stat in ["p50", "p90", "max"]:
                row[f"{distkey}_{stat}"] = d.get(stat)
        rows.append(row)
    fields = list(rows[0].keys())
    write_csv(VARIANT_SUMMARY_CSV, fields, rows)


def _write_track_manifest(v, R):
    fields = ["track_id", "patient_id", "pos_y0", "pos_x0", "pos_y1", "pos_x1",
              "z_start", "z_end", "track_len", "n_candidates",
              "n_positive", "n_hard_negative", "has_positive", "positive_ratio"]
    write_csv(TRACK_MANIFEST_CSV[R], fields, v["_tracks"])


def _write_candidate_survival(cands, v, R):
    flag = v["_survived_flag"]
    rows = []
    for c in cands:
        rows.append({
            "candidate_id": c["candidate_id"],
            "patient_id":   c["patient_id"],
            "local_z":      c["local_z"],
            "pos_y0": c["poskey"][0], "pos_x0": c["poskey"][1],
            "pos_y1": c["poskey"][2], "pos_x1": c["poskey"][3],
            "label":        c["label"],
            "survived":     flag[c["candidate_id"]],
        })
    fields = ["candidate_id", "patient_id", "local_z", "pos_y0", "pos_x0",
              "pos_y1", "pos_x1", "label", "survived"]
    write_csv(CAND_SURVIVAL_CSV[R], fields, rows)


def _write_patient_survival(variants):
    rows = []
    for R in MIN_RUN_LENS:
        v = variants[R]
        all_pids = set(v["_pat_surv_cand"].keys()) | set(v["_orig_pos_pat"]) | set(v["_pat_track_count"].keys())
        for pid in sorted(all_pids):
            o_slice = len(v["_pat_orig_pos_slice"].get(pid, set()))
            s_slice = len(v["_pat_surv_pos_slice"].get(pid, set()))
            rows.append({
                "patient_id": pid,
                "variant": v["variant"],
                "min_run_len": R,
                "is_positive_patient": pid in v["_orig_pos_pat"],
                "survived_candidate_count": v["_pat_surv_cand"].get(pid, 0),
                "survived_track_count": v["_pat_track_count"].get(pid, 0),
                "original_positive_slice": o_slice,
                "survived_positive_slice": s_slice,
                "positive_slice_coverage": round(s_slice / o_slice, 4) if o_slice else "",
            })
    fields = ["patient_id", "variant", "min_run_len", "is_positive_patient",
              "survived_candidate_count", "survived_track_count",
              "original_positive_slice", "survived_positive_slice", "positive_slice_coverage"]
    write_csv(PATIENT_SURVIVAL_CSV, fields, rows)


def _write_problem_audit(cands, variants, pos_runs):
    """문제 환자 3명 × variant 상세. 반환: {variant: {pid: metrics}}."""
    # candidate 인덱스: patient -> label -> [(z, cid)]
    by_pat = defaultdict(list)
    for c in cands:
        by_pat[c["patient_id"]].append(c)

    summary = {VARIANT_NAME[R]: {} for R in MIN_RUN_LENS}
    rows = []
    for R in MIN_RUN_LENS:
        v = variants[R]
        flag = v["_survived_flag"]
        for pid in PROBLEM_PATIENT_IDS:
            pcands = by_pat.get(pid, [])
            o_pos = sum(1 for c in pcands if c["label"] == "positive")
            s_pos = sum(1 for c in pcands if c["label"] == "positive" and flag[c["candidate_id"]])
            o_slice = len(set(c["local_z"] for c in pcands if c["label"] == "positive"))
            s_slice = len(set(c["local_z"] for c in pcands
                              if c["label"] == "positive" and flag[c["candidate_id"]]))
            surv_tracks = [t for t in v["_tracks"] if t["patient_id"] == pid]
            pos_tracks = [t for t in surv_tracks if t["has_positive"]]
            longest_pos = max([t["track_len"] for t in pos_tracks], default=0)
            metrics = {
                "original_positive_candidate": o_pos,
                "survived_positive_candidate": s_pos,
                "positive_candidate_retention": round(s_pos / o_pos, 4) if o_pos else 0.0,
                "original_positive_slice": o_slice,
                "survived_positive_slice": s_slice,
                "positive_slice_coverage": round(s_slice / o_slice, 4) if o_slice else 0.0,
                "survived_track_count": len(surv_tracks),
                "longest_positive_track_len": longest_pos,
            }
            summary[v["variant"]][pid] = metrics
            row = {"patient_id": pid, "variant": v["variant"], "min_run_len": R}
            row.update(metrics)
            rows.append(row)
    fields = ["patient_id", "variant", "min_run_len",
              "original_positive_candidate", "survived_positive_candidate", "positive_candidate_retention",
              "original_positive_slice", "survived_positive_slice", "positive_slice_coverage",
              "survived_track_count", "longest_positive_track_len"]
    write_csv(PROBLEM_AUDIT_CSV, fields, rows)
    return summary


def _clean_variant_for_json(v):
    return {k: val for k, val in v.items() if not k.startswith("_")}


def _write_summary_json(variants, problem_summary, position_cols, position_source,
                        verdict, reasons, elapsed, critical_error):
    summary = {
        "verdict": verdict,
        "reasons": reasons,
        "position_columns": position_cols,
        "position_source": position_source,
        "original_candidate_count": variants[2]["original_candidate_count"],
        "min_run_lens": MIN_RUN_LENS,
        "variants": {VARIANT_NAME[R]: _clean_variant_for_json(variants[R]) for R in MIN_RUN_LENS},
        "problem_patient_audit": problem_summary,
        "critical_error": critical_error,
        "elapsed_sec": round(elapsed, 1),
        "guardrails": GUARDRAILS,
        "note": "estimated_runtime 은 shard0 실측(4913 forward≈84.2s) 단순비례 estimate only.",
    }
    ensure_output_path_safe(SUMMARY_JSON)
    with open(str(SUMMARY_JSON), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  saved: {SUMMARY_JSON}")


def _write_report_md(variants, problem_summary, position_cols, position_source,
                     verdict, reasons, elapsed):
    L = []
    L.append("# strict same-position z-track survival preflight v1 report")
    L.append("")
    L.append(f"**판정: {verdict}**")
    L.append("")
    L.append(f"- position columns: {position_cols} (source={position_source})")
    L.append(f"- original candidates: {variants[2]['original_candidate_count']:,}")
    L.append(f"- elapsed: {elapsed:.1f}s")
    L.append("")
    L.append("## min_run_len 별 survival")
    L.append("")
    L.append("| variant | surv_cand | reduction | pos_cand_ret | pos_slice_cov | pos_pat_cov | complete_miss | fwd_cand | est_min |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for R in MIN_RUN_LENS:
        v = variants[R]
        L.append(f"| {v['variant']} | {v['survived_candidate_count']:,} | {v['candidate_reduction_rate']:.4f} | "
                 f"{v['positive_candidate_retention']:.4f} | {v['positive_slice_coverage']:.4f} | "
                 f"{v['positive_patient_coverage']:.4f} | {v['complete_miss_patient']} | "
                 f"{v['forward_candidate_count']:,} | {v['estimated_runtime_min']} |")
    L.append("")
    L.append("## 병변 patient coverage 세부")
    L.append("")
    for R in MIN_RUN_LENS:
        v = variants[R]
        L.append(f"- {v['variant']}: orig_pos_patient={v['original_positive_patient']} "
                 f"surv_pos_patient={v['survived_positive_patient']} "
                 f"cov<50={v['patients_coverage_lt50']} cov<80={v['patients_coverage_lt80']} "
                 f"cov<95={v['patients_coverage_lt95']} cov==100={v['patients_coverage_eq100']}")
    L.append("")
    L.append("## track-level positive 분석")
    L.append("")
    for R in MIN_RUN_LENS:
        v = variants[R]
        L.append(f"- {v['variant']}: tracks={v['survived_track_count']:,} "
                 f"has_pos={v['has_positive_track_count']:,} "
                 f"hn_only={v['all_hard_negative_track_count']:,} "
                 f"pos_track_ratio={v['positive_containing_track_ratio']:.4f} "
                 f"mixed_ratio={v['mixed_track_ratio']:.4f}")
    L.append("")
    L.append("## RD4AD forward 예상량 (estimate only)")
    L.append("")
    L.append("| variant | forward_cand | vs113447 reduction | vs20216 group_rep | est_runtime_min |")
    L.append("|---|---|---|---|---|")
    for R in MIN_RUN_LENS:
        v = variants[R]
        L.append(f"| {v['variant']} | {v['forward_candidate_count']:,} | "
                 f"{v['forward_reduction_vs_patch_baseline']:.4f} | "
                 f"{v['forward_vs_group_rep_ratio']:.2f}x ({v['forward_vs_group_rep_delta']:+,}) | "
                 f"{v['estimated_runtime_min']} |")
    L.append("")
    L.append("## 문제 환자 3명 survival")
    L.append("")
    for pid in PROBLEM_PATIENT_IDS:
        L.append(f"### {pid}")
        for R in MIN_RUN_LENS:
            m = problem_summary[VARIANT_NAME[R]][pid]
            L.append(f"- {VARIANT_NAME[R]}: pos_cand {m['survived_positive_candidate']}/{m['original_positive_candidate']} "
                     f"(ret={m['positive_candidate_retention']:.4f}), "
                     f"pos_slice {m['survived_positive_slice']}/{m['original_positive_slice']} "
                     f"(cov={m['positive_slice_coverage']:.4f}), "
                     f"tracks={m['survived_track_count']}, longest_pos_track={m['longest_positive_track_len']}")
        L.append("")
    L.append("## 판정 사유")
    L.append("")
    for r in reasons:
        L.append(f"- {r}")
    L.append("")
    L.append("## guardrails")
    L.append("")
    for k, val in GUARDRAILS.items():
        L.append(f"- {k}: {val}")
    ensure_output_path_safe(REPORT_MD)
    with open(str(REPORT_MD), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  saved: {REPORT_MD}")


def _write_done(verdict):
    ensure_output_path_safe(DONE_JSON)
    with open(str(DONE_JSON), "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, "stage": "strict_ztrack_survival_preflight",
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
    print(f"  saved: {DONE_JSON}")


def _finalize_fail(t0, reason):
    save_error_csv()
    elapsed = time.perf_counter() - t0
    ensure_output_path_safe(SUMMARY_JSON)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(SUMMARY_JSON), "w", encoding="utf-8") as f:
        json.dump({"verdict": "FAIL", "reason": reason, "elapsed_sec": round(elapsed, 1),
                   "guardrails": GUARDRAILS}, f, indent=2, ensure_ascii=False)
    _write_done("FAIL")
    print(f"\n[PREFLIGHT] FAIL: {reason}")


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="strict same-position z-track survival preflight v1")
    parser.add_argument("--dry-run",                action="store_true")
    parser.add_argument("--run-preflight",          action="store_true")
    parser.add_argument("--confirm-readonly",       action="store_true")
    parser.add_argument("--confirm-stage1dev-only", action="store_true")
    args = parser.parse_args()

    if not any([args.dry_run, args.run_preflight]):
        print("[ABORT] bare run 차단. --dry-run / --run-preflight 사용.", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        run_dry()
        return

    if args.run_preflight:
        if not args.confirm_readonly or not args.confirm_stage1dev_only:
            print("[ABORT] --confirm-readonly --confirm-stage1dev-only 필요", file=sys.stderr)
            sys.exit(2)
        run_preflight()
        return


if __name__ == "__main__":
    main()
