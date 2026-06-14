"""
RD4AD strict same-position z-track actual scoring MERGE v1

shard_0~7 strict_ztrack_scores_shard_{id}.csv 를 병합·검증하고,
candidate-level 및 track-level top-k 평가 / report / summary 를 생성한다.

실행 방식:
  bare run (인자 없음) : exit 2 차단
  dry-run    : python <script> --dry-run
  merge-shards: python <script> --merge-shards --confirm-readonly
                    --confirm-stage1dev-only

입력 (read-only):
  shards/shard_{id}/strict_ztrack_scores_shard_{id}.csv  (0~7)
  shards/shard_{id}/DONE.json
  shards/shard_{id}/shard_{id}_summary.json
  manifests/strict_ztrack_scoring_candidate_manifest.csv
  manifests/strict_ztrack_scoring_shard_plan.csv
  (옵션) RD-D1s stage1dev candidate score CSV  (patch baseline)
  (옵션) group-level full scoring merged CSV    (fuzzy group baseline)

출력 (생성):
  manifests/strict_ztrack_scores_full_merged.csv
  manifests/strict_ztrack_patient_candidate_topk.csv
  manifests/strict_ztrack_track_score_summary.csv
  manifests/strict_ztrack_patient_track_topk.csv
  manifests/strict_ztrack_problem_patient_audit.csv
  reports/rd4ad_strict_ztrack_actual_scoring_report.md
  reports/rd4ad_strict_ztrack_actual_scoring_summary.json
  DONE_FULL_MERGE.json

금지:
  model forward / training / stage2_holdout 접근 /
  입력 manifest·shard·preflight 덮어쓰기 /
  기존 외부 artifact 수정.

주의:
  track-level top-k 는 단위(track)가 candidate-level 과 다르므로
  직접 수치 비교 시 caution 을 report 에 기록한다.
"""
import csv
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

# =============================================================================
# 경로 상수
# =============================================================================

PROJECT_ROOT    = Path("/home/jinhy/project/lung-ct-anomaly")
EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "experiments/rd4ad_strict_same_position_ztrack_actual_scoring_v1"
)

MANIFEST_DIR = EXPERIMENT_ROOT / "manifests"
REPORT_DIR   = EXPERIMENT_ROOT / "reports"
SHARDS_DIR   = EXPERIMENT_ROOT / "shards"

# 입력 manifest (read-only)
CANDIDATE_MANIFEST_CSV = MANIFEST_DIR / "strict_ztrack_scoring_candidate_manifest.csv"
SHARD_PLAN_CSV         = MANIFEST_DIR / "strict_ztrack_scoring_shard_plan.csv"

# preflight 산출물 (read-only)
PREFLIGHT_SUMMARY_JSON = (
    EXPERIMENT_ROOT / "reports/rd4ad_strict_ztrack_actual_scoring_preflight_summary.json"
)

# RD-D1s patch baseline (optional, read-only)
RD_D1S_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1s_medi3ch_true_rd4ad_shard_run_v1"
    / "rd_d1s_stage1dev_candidate_score.csv"
)

# group-level baseline (optional, read-only)
GROUP_MERGED_CSV = (
    PROJECT_ROOT
    / "experiments/rd4ad_z_continuity_group_full_scoring_v1"
    / "manifests/group_scores_full_merged.csv"
)

# 출력
MERGED_CSV             = MANIFEST_DIR / "strict_ztrack_scores_full_merged.csv"
CANDIDATE_TOPK_CSV     = MANIFEST_DIR / "strict_ztrack_patient_candidate_topk.csv"
TRACK_SUMMARY_CSV      = MANIFEST_DIR / "strict_ztrack_track_score_summary.csv"
TRACK_TOPK_CSV         = MANIFEST_DIR / "strict_ztrack_patient_track_topk.csv"
PROBLEM_AUDIT_CSV      = MANIFEST_DIR / "strict_ztrack_problem_patient_audit.csv"
REPORT_MD              = REPORT_DIR   / "rd4ad_strict_ztrack_actual_scoring_report.md"
SUMMARY_JSON           = REPORT_DIR   / "rd4ad_strict_ztrack_actual_scoring_summary.json"
DONE_FULL_MERGE_JSON   = EXPERIMENT_ROOT / "DONE_FULL_MERGE.json"

# 상수
SHARD_COUNT               = 8
EXPECTED_TOTAL_CANDIDATES = 92342
TOPK_VALS                 = [1, 3, 5, 10, 20, 50]
PRIMARY_SCORE_COL         = "rd4ad_ztrack_score_raw"
PREVIEW_SCORE_COLS        = ["P1_times_roi", "P2_times_sqrt_roi"]
PRIMARY_TRACK_SCORE_COL   = "track_score_max"
PROBLEM_PATIENT_IDS       = ["LUNG1-086", "LUNG1-386", "LUNG1-399"]

# =============================================================================
# guardrail
# =============================================================================

GUARDRAILS = {
    "stage2_holdout_accessed":                       False,
    "checkpoint_loaded":                             False,
    "model_forward_executed":                        False,
    "training_executed":                             False,
    "backward_executed":                             False,
    "optimizer_created":                             False,
    "checkpoint_saved":                              False,
    "crop_generation_executed":                      False,
    "full_scoring_executed":                         "merge_only",
    "threshold_recalculated":                        False,
    "existing_artifact_modified":                    False,
    "existing_script_modified":                      False,
    "output_overwrite":                              False,
    "first_stage_score_used_for_candidate_deletion": False,
    "label_used_for_evaluation_only":                True,
    "label_used_for_scoring_selection":              False,
    "xy_radius_grouping_used":                       False,
    "representative_only_scoring_used":              False,
    "all_survived_track_candidates_scored":          True,
    "raw_rd4ad_primary_score":                       True,
    "adjusted_score_preview_only":                   True,
}

# =============================================================================
# 안전 경로 검사
# =============================================================================

_PROTECTED_INPUTS = [
    CANDIDATE_MANIFEST_CSV,
    SHARD_PLAN_CSV,
    PREFLIGHT_SUMMARY_JSON,
    RD_D1S_SCORE_CSV,
    GROUP_MERGED_CSV,
]

_ALLOWED_OUTPUTS = {
    MERGED_CSV,
    CANDIDATE_TOPK_CSV,
    TRACK_SUMMARY_CSV,
    TRACK_TOPK_CSV,
    PROBLEM_AUDIT_CSV,
    REPORT_MD,
    SUMMARY_JSON,
    DONE_FULL_MERGE_JSON,
}


def assert_path_safe(p: Path) -> None:
    s = str(p).lower()
    if "stage2_holdout" in s or ("stage2" in s and "holdout" in s):
        GUARDRAILS["stage2_holdout_accessed"] = True
        raise RuntimeError(f"[ABORT] stage2_holdout 경로 접근 차단: {p}")


def ensure_output_path_safe(p: Path) -> None:
    rp = Path(p).resolve()
    for pi in _PROTECTED_INPUTS:
        try:
            if rp == pi.resolve():
                GUARDRAILS["existing_artifact_modified"] = True
                raise RuntimeError(f"[ABORT] 입력/preflight 파일 덮어쓰기 차단: {p}")
        except RuntimeError:
            raise
        except Exception:
            pass
    # shard 입력 CSV 도 보호
    shards_resolved = str(SHARDS_DIR.resolve())
    if str(rp).startswith(shards_resolved):
        GUARDRAILS["existing_artifact_modified"] = True
        raise RuntimeError(f"[ABORT] shard 입력 파일 덮어쓰기 차단: {p}")
    allowed_resolved = {o.resolve() for o in _ALLOWED_OUTPUTS}
    if rp not in allowed_resolved:
        GUARDRAILS["existing_artifact_modified"] = True
        raise RuntimeError(f"[ABORT] 허용되지 않은 출력 경로: {p}")


# =============================================================================
# CSV 유틸
# =============================================================================

def read_csv(path: Path) -> list:
    assert_path_safe(path)
    rows = []
    with open(str(path), encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def write_csv(path: Path, fieldnames: list, rows: list) -> None:
    ensure_output_path_safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  saved: {path} ({len(rows)} rows)")


def _to_float(v, default=None):
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v, default=0):
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


# =============================================================================
# patch baseline top-k (RD-D1s stage1dev candidate score)
# =============================================================================

def patch_topk_baseline() -> dict:
    if not RD_D1S_SCORE_CSV.exists():
        return {}
    assert_path_safe(RD_D1S_SCORE_CSV)
    rows = read_csv(RD_D1S_SCORE_CSV)
    pat: dict = defaultdict(list)
    for r in rows:
        if r.get("stage_split", "") != "stage1_dev":
            continue
        pat[r["patient_id"]].append(r)
    base: dict = {}
    for k in TOPK_VALS:
        hit = total = 0
        for pid, rs in pat.items():
            if not any(r.get("label") == "positive" for r in rs):
                continue
            total += 1
            scored = sorted(
                rs,
                key=lambda r: _to_float(r.get("rd_d1s_medi3ch_rd4ad_score"), -1e9),
                reverse=True,
            )
            if any(r.get("label") == "positive" for r in scored[:k]):
                hit += 1
        base[k] = round(hit / total, 4) if total else 0.0
    return base


# =============================================================================
# group-level baseline top-k (optional, fuzzy group)
# =============================================================================

def group_topk_baseline() -> dict:
    """group scoring merge CSV 에서 group-level retention 을 로드한다 (optional)."""
    if not GROUP_MERGED_CSV.exists():
        return {}
    try:
        assert_path_safe(GROUP_MERGED_CSV)
        rows = read_csv(GROUP_MERGED_CSV)
        pat: dict = defaultdict(list)
        for r in rows:
            pat[r["patient_id"]].append(r)
        base: dict = {}
        for k in TOPK_VALS:
            hit = total = 0
            for pid, rs in pat.items():
                if not any(r.get("has_positive") == "True" for r in rs):
                    continue
                total += 1
                scored = sorted(
                    rs,
                    key=lambda r: _to_float(r.get("rd4ad_group_score_raw"), -1e9),
                    reverse=True,
                )
                if any(r.get("has_positive") == "True" for r in scored[:k]):
                    hit += 1
            base[k] = round(hit / total, 4) if total else 0.0
        return base
    except Exception as e:
        print(f"  [WARN] group baseline 로드 실패: {e}")
        return {}


# =============================================================================
# candidate-level top-k 평가
# =============================================================================

def eval_candidate_topk(merged_rows: list, score_col: str) -> dict:
    """환자별 candidate top-k 에서 positive retention 집계."""
    pat: dict = defaultdict(list)
    for r in merged_rows:
        pat[r["patient_id"]].append(r)

    total_pos_cands = sum(1 for r in merged_rows if r.get("label") == "positive")
    total_pos_slices = set(
        (r["patient_id"], r["local_z"])
        for r in merged_rows
        if r.get("label") == "positive"
    )

    out: dict = {}
    for k in TOPK_VALS:
        n_pos_patients = 0
        n_hit          = 0
        topk_pos_cands  = 0
        topk_pos_slices: set = set()

        for pid, rows in pat.items():
            has_pos_patient = any(r.get("label") == "positive" for r in rows)
            scored = sorted(
                rows,
                key=lambda r: _to_float(r.get(score_col), -1e9),
                reverse=True,
            )
            topk = scored[:k]

            if not has_pos_patient:
                continue
            n_pos_patients += 1
            if any(r.get("label") == "positive" for r in topk):
                n_hit += 1
            for r in topk:
                if r.get("label") == "positive":
                    topk_pos_cands += 1
                    topk_pos_slices.add((r["patient_id"], r["local_z"]))

        retention = round(n_hit / n_pos_patients, 4) if n_pos_patients else 0.0
        out[k] = {
            "lesion_candidate_retention":   retention,
            "patient_hit_rate":             retention,
            "n_patients_with_positive":     n_pos_patients,
            "positive_candidate_coverage":  (
                round(topk_pos_cands / total_pos_cands, 4)
                if total_pos_cands else 0.0
            ),
            "positive_slice_coverage":      (
                round(len(topk_pos_slices) / len(total_pos_slices), 4)
                if total_pos_slices else 0.0
            ),
            "pat_all_sup_proxy":            round(1.0 - retention, 4),
        }
    return out


# =============================================================================
# track-level score 집계
# =============================================================================

def aggregate_track_scores(merged_rows: list) -> list:
    """track_id 별로 survived candidate score 를 집계한다."""
    track_cands: dict = defaultdict(list)
    for r in merged_rows:
        tid = r.get("track_id", "")
        if tid:
            track_cands[tid].append(r)

    track_rows: list = []
    for tid, cands in track_cands.items():
        scores = []
        for c in cands:
            v = _to_float(c.get("rd4ad_ztrack_score_raw"), None)
            if v is not None and math.isfinite(v):
                scores.append(v)
        if not scores:
            continue

        scores_desc = sorted(scores, reverse=True)
        track_score_max      = scores_desc[0]
        track_score_top2     = sum(scores_desc[:2]) / min(2, len(scores_desc))
        track_score_top3     = sum(scores_desc[:3]) / min(3, len(scores_desc))
        track_score_mean     = sum(scores) / len(scores)

        patient_id     = cands[0]["patient_id"]
        positive_count = sum(1 for c in cands if c.get("label") == "positive")
        hn_count       = sum(1 for c in cands if c.get("label") != "positive")
        has_positive   = positive_count > 0
        pos_slices     = set(
            (c["patient_id"], c["local_z"])
            for c in cands
            if c.get("label") == "positive"
        )

        track_rows.append({
            "track_id":              tid,
            "patient_id":            patient_id,
            "n_candidates":          len(cands),
            "positive_count":        positive_count,
            "hard_negative_count":   hn_count,
            "has_positive":          str(has_positive),
            "pos_slice_count":       len(pos_slices),
            "track_score_max":       round(track_score_max, 6),
            "track_score_top2_mean": round(track_score_top2, 6),
            "track_score_top3_mean": round(track_score_top3, 6),
            "track_score_mean":      round(track_score_mean, 6),
        })

    return track_rows


TRACK_SUMMARY_FIELDS = [
    "track_id", "patient_id", "n_candidates",
    "positive_count", "hard_negative_count", "has_positive", "pos_slice_count",
    "track_score_max", "track_score_top2_mean", "track_score_top3_mean", "track_score_mean",
]


# =============================================================================
# track-level top-k 평가
# =============================================================================

def eval_track_topk(
    track_rows: list,
    merged_rows: list,
    score_col: str = PRIMARY_TRACK_SCORE_COL,
) -> dict:
    """환자별 track top-k 에서 positive-containing track 의 retention 집계."""
    # track_id → candidate rows (for slice coverage)
    track_cand_map: dict = defaultdict(list)
    for r in merged_rows:
        tid = r.get("track_id", "")
        if tid:
            track_cand_map[tid].append(r)

    total_pos_slices = set(
        (r["patient_id"], r["local_z"])
        for r in merged_rows
        if r.get("label") == "positive"
    )
    total_pos_cands = sum(1 for r in merged_rows if r.get("label") == "positive")

    pat_tracks: dict = defaultdict(list)
    for t in track_rows:
        pat_tracks[t["patient_id"]].append(t)

    out: dict = {}
    for k in TOPK_VALS:
        n_pos_patients  = 0
        n_hit           = 0
        topk_pos_slices: set = set()
        topk_pos_cands  = 0

        for pid, tracks in pat_tracks.items():
            has_pos_patient = any(t.get("has_positive") == "True" for t in tracks)
            scored = sorted(
                tracks,
                key=lambda t: _to_float(t.get(score_col), -1e9),
                reverse=True,
            )
            topk_tracks    = scored[:k]
            topk_track_ids = {t["track_id"] for t in topk_tracks}

            if not has_pos_patient:
                continue
            n_pos_patients += 1
            if any(t.get("has_positive") == "True" for t in topk_tracks):
                n_hit += 1

            for tid in topk_track_ids:
                for c in track_cand_map.get(tid, []):
                    if c.get("label") == "positive":
                        topk_pos_slices.add((c["patient_id"], c["local_z"]))
                        topk_pos_cands += 1

        retention = round(n_hit / n_pos_patients, 4) if n_pos_patients else 0.0
        out[k] = {
            "positive_containing_track_hit_rate": retention,
            "patient_hit_rate":                   retention,
            "n_patients_with_positive_track":     n_pos_patients,
            "positive_candidate_coverage_by_tracks": (
                round(topk_pos_cands / total_pos_cands, 4) if total_pos_cands else 0.0
            ),
            "positive_slice_coverage_by_tracks": (
                round(len(topk_pos_slices) / len(total_pos_slices), 4)
                if total_pos_slices else 0.0
            ),
            "pat_all_sup_proxy": round(1.0 - retention, 4),
        }
    return out


# =============================================================================
# problem patient audit
# =============================================================================

def build_problem_audit(
    merged_rows: list,
    track_rows: list,
) -> tuple:
    """candidate-level + track-level 양쪽으로 문제 환자 3명 audit."""
    # candidate-level
    pat_cands: dict = defaultdict(list)
    for r in merged_rows:
        pat_cands[r["patient_id"]].append(r)

    # track-level
    pat_tracks: dict = defaultdict(list)
    for t in track_rows:
        pat_tracks[t["patient_id"]].append(t)

    audit_rows: list  = []
    audit_summary: dict = {}

    for pid in PROBLEM_PATIENT_IDS:
        cands  = pat_cands.get(pid, [])
        tracks = pat_tracks.get(pid, [])

        n_pos_cands  = sum(1 for c in cands if c.get("label") == "positive")
        n_pos_tracks = sum(1 for t in tracks if t.get("has_positive") == "True")

        # candidate-level
        cands_scored = sorted(
            cands,
            key=lambda r: _to_float(r.get(PRIMARY_SCORE_COL), -1e9),
            reverse=True,
        )
        # track-level
        tracks_scored = sorted(
            tracks,
            key=lambda t: _to_float(t.get(PRIMARY_TRACK_SCORE_COL), -1e9),
            reverse=True,
        )

        for k in TOPK_VALS:
            cand_hit  = any(
                r.get("label") == "positive" for r in cands_scored[:k]
            )
            track_hit = any(
                t.get("has_positive") == "True" for t in tracks_scored[:k]
            )
            audit_rows.append({
                "patient_id":         pid,
                "n_candidates":       len(cands),
                "n_pos_candidates":   n_pos_cands,
                "n_tracks":           len(tracks),
                "n_pos_tracks":       n_pos_tracks,
                "k":                  k,
                "cand_topk_has_positive":  str(cand_hit),
                "track_topk_has_positive": str(track_hit),
            })
            audit_summary[f"{pid}_cand_top{k}"]  = str(cand_hit)
            audit_summary[f"{pid}_track_top{k}"] = str(track_hit)

        print(
            f"  {pid}: "
            f"cands={len(cands)} pos_cands={n_pos_cands} "
            f"tracks={len(tracks)} pos_tracks={n_pos_tracks} "
            f"cand_top20={audit_summary.get(f'{pid}_cand_top20')} "
            f"cand_top50={audit_summary.get(f'{pid}_cand_top50')} "
            f"track_top20={audit_summary.get(f'{pid}_track_top20')} "
            f"track_top50={audit_summary.get(f'{pid}_track_top50')}"
        )

    return audit_rows, audit_summary


AUDIT_FIELDS = [
    "patient_id", "n_candidates", "n_pos_candidates",
    "n_tracks", "n_pos_tracks",
    "k", "cand_topk_has_positive", "track_topk_has_positive",
]


# =============================================================================
# dry-run
# =============================================================================

def run_dry() -> None:
    print("=" * 70)
    print("[DRY-RUN] RD4AD strict z-track actual scoring MERGE v1")
    print("  파일 생성 없음")
    print("=" * 70)
    issues: list = []

    print("\n[1] shard 출력 / DONE 확인")
    total_expected_from_summaries = 0
    for sid in range(SHARD_COUNT):
        d      = SHARDS_DIR / f"shard_{sid}"
        csv_p  = d / f"strict_ztrack_scores_shard_{sid}.csv"
        done_p = d / "DONE.json"
        summ_p = d / f"shard_{sid}_summary.json"

        ok_csv  = csv_p.exists()
        ok_done = done_p.exists()
        print(
            f"  shard {sid}: "
            f"csv={'OK' if ok_csv else 'MISSING'} "
            f"done={'OK' if ok_done else 'MISSING'}"
        )
        if not ok_csv:
            issues.append(f"shard {sid} csv 없음")
        if not ok_done:
            issues.append(f"shard {sid} DONE.json 없음")

        if summ_p.exists():
            try:
                with open(str(summ_p), encoding="utf-8") as f:
                    s = json.load(f)
                exp      = s.get("expected_candidate_count", 0)
                scored   = s.get("actual_scored_candidate_count", "?")
                failed   = s.get("failed_candidate_count", "?")
                verdict  = s.get("verdict", "?")
                total_expected_from_summaries += _to_int(exp)
                print(
                    f"           expected={exp} scored={scored} "
                    f"failed={failed} verdict={verdict}"
                )
                if verdict not in ("PASS",):
                    issues.append(f"shard {sid} verdict={verdict}")
            except Exception as e:
                issues.append(f"shard {sid} summary 읽기 실패: {e}")

    print(f"\n[2] expected candidate count 합계 (summaries): "
          f"{total_expected_from_summaries:,} (목표 {EXPECTED_TOTAL_CANDIDATES:,})")
    if (total_expected_from_summaries > 0
            and total_expected_from_summaries != EXPECTED_TOTAL_CANDIDATES):
        issues.append(
            f"expected 합계 {total_expected_from_summaries} != "
            f"{EXPECTED_TOTAL_CANDIDATES}"
        )

    print("\n[3] 출력 overwrite 위험")
    for p in [
        MERGED_CSV, CANDIDATE_TOPK_CSV, TRACK_SUMMARY_CSV,
        TRACK_TOPK_CSV, PROBLEM_AUDIT_CSV, REPORT_MD, SUMMARY_JSON,
    ]:
        status = "WARN exists" if p.exists() else "OK"
        print(f"  [{status}] {p.name}")
    if DONE_FULL_MERGE_JSON.exists():
        print(f"  [주의] {DONE_FULL_MERGE_JSON.name} 이미 존재 → 덮어씀")

    print("\n[4] 입력 manifest read-only 확인")
    for p in [CANDIDATE_MANIFEST_CSV, SHARD_PLAN_CSV]:
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status}] {p.name}")

    print("\n[5] 옵션 baseline 파일")
    print(f"  RD-D1s patch baseline: {'OK' if RD_D1S_SCORE_CSV.exists() else 'MISSING (생략)'}")
    print(f"  group-level baseline:  {'OK' if GROUP_MERGED_CSV.exists() else 'MISSING (생략)'}")

    print("\n" + "=" * 70)
    if issues:
        print("[DRY-RUN] 이슈:")
        for it in issues:
            print(f"  - {it}")
        print("판정: NEEDS_FIX (shard 미완료 또는 이슈 존재)")
        sys.exit(1)
    else:
        print("[DRY-RUN] shard 출력/DONE OK, merge 준비됨.")
        print("판정: READY_TO_MERGE")
    print("=" * 70)


# =============================================================================
# merge-shards
# =============================================================================

def run_merge() -> None:
    print("=" * 70)
    print("[MERGE] RD4AD strict z-track actual scoring MERGE v1")
    print("=" * 70)
    t0           = time.perf_counter()
    fail_reasons: list = []

    # ── 1. shard CSV / summary 로드 ──────────────────────────────────────────
    print("\n[1] shard CSV 로드")
    merged_rows: list  = []
    shard_row_counts: dict = {}
    shard_summaries: dict  = {}

    for sid in range(SHARD_COUNT):
        d      = SHARDS_DIR / f"shard_{sid}"
        csv_p  = d / f"strict_ztrack_scores_shard_{sid}.csv"
        done_p = d / "DONE.json"
        summ_p = d / f"shard_{sid}_summary.json"

        if not csv_p.exists():
            fail_reasons.append(f"shard {sid} csv 없음")
            continue
        if not done_p.exists():
            fail_reasons.append(f"shard {sid} DONE.json 없음")

        assert_path_safe(csv_p)
        rows = read_csv(csv_p)
        shard_row_counts[sid] = len(rows)
        merged_rows.extend(rows)
        print(f"  shard {sid}: {len(rows):,} rows")

        if summ_p.exists():
            with open(str(summ_p), encoding="utf-8") as f:
                shard_summaries[sid] = json.load(f)

    if fail_reasons:
        print("  [FAIL] shard 입력 누락:")
        for r in fail_reasons:
            print(f"    - {r}")
        _finalize(
            merged_rows=[], topk_cand={}, topk_track={}, track_rows=[],
            patch_base={}, group_base={},
            audit_rows=[], audit_summary={},
            shard_row_counts=shard_row_counts,
            shard_summaries=shard_summaries,
            verdict="FAIL", fail_reasons=fail_reasons,
            elapsed=time.perf_counter() - t0,
        )
        sys.exit(1)

    merged_row_count = len(merged_rows)
    print(f"  merged rows: {merged_row_count:,}")

    # ── 2. 무결성 검증 ────────────────────────────────────────────────────────
    print("\n[2] 무결성 검증")
    cids        = [r["candidate_id"] for r in merged_rows]
    cid_set     = set(cids)
    dup_count   = len(cids) - len(cid_set)

    # shard plan 기대 candidate count
    plan_shard_counts: dict = {}
    if SHARD_PLAN_CSV.exists():
        for row in read_csv(SHARD_PLAN_CSV):
            plan_shard_counts[int(row["shard_id"])] = int(row["candidate_count"])

    shard_count_ok = all(
        shard_row_counts.get(s, 0) == plan_shard_counts.get(s, 0)
        for s in range(SHARD_COUNT)
    ) if plan_shard_counts else True

    # candidate manifest 와 대조
    manifest_cids: set = set()
    if CANDIDATE_MANIFEST_CSV.exists():
        for row in read_csv(CANDIDATE_MANIFEST_CSV):
            manifest_cids.add(row["candidate_id"])
    missing_vs_manifest = manifest_cids - cid_set
    extra_vs_manifest   = cid_set - manifest_cids

    # NaN / Inf
    nan_count = inf_count = 0
    for r in merged_rows:
        v = _to_float(r.get(PRIMARY_SCORE_COL), None)
        if v is None:
            continue
        if math.isnan(v):
            nan_count += 1
        elif math.isinf(v):
            inf_count += 1

    print(f"  merged_row_count         : {merged_row_count:,}")
    print(f"  duplicate_candidate_id   : {dup_count}")
    print(f"  missing_vs_manifest      : {len(missing_vs_manifest)}")
    print(f"  extra_vs_manifest        : {len(extra_vs_manifest)}")
    print(f"  shard_count_ok           : {shard_count_ok}")
    print(f"  score NaN / Inf          : {nan_count} / {inf_count}")

    if merged_row_count != EXPECTED_TOTAL_CANDIDATES:
        fail_reasons.append(
            f"merged row count {merged_row_count} != {EXPECTED_TOTAL_CANDIDATES}"
        )
    if dup_count != 0:
        fail_reasons.append(f"duplicate candidate_id {dup_count}")
    if missing_vs_manifest:
        fail_reasons.append(f"missing vs manifest {len(missing_vs_manifest)}")
    if extra_vs_manifest:
        fail_reasons.append(f"extra vs manifest {len(extra_vs_manifest)}")
    if not shard_count_ok:
        fail_reasons.append("shard row count != shard plan")
    if nan_count or inf_count:
        fail_reasons.append(f"NaN/Inf score {nan_count}/{inf_count}")

    total_failed = sum(
        _to_int(s.get("failed_candidate_count", 0))
        for s in shard_summaries.values()
    )
    any_stage2  = any(
        s.get("stage2_holdout_accessed", False) for s in shard_summaries.values()
    )
    any_mod     = any(
        s.get("existing_artifact_modified", False) for s in shard_summaries.values()
    )
    if total_failed != 0:
        fail_reasons.append(f"shard failed_candidate_count 합계 {total_failed}")
    if any_stage2:
        fail_reasons.append("shard 에서 stage2_holdout 접근")
        GUARDRAILS["stage2_holdout_accessed"] = True
    if any_mod:
        fail_reasons.append("shard 에서 existing_artifact_modified")

    # ── 3. candidate-level top-k 평가 ────────────────────────────────────────
    print("\n[3] candidate-level top-k 평가")
    score_cols_cand = [PRIMARY_SCORE_COL] + [
        c for c in PREVIEW_SCORE_COLS
        if any(r.get(c, "") not in ("", None) for r in merged_rows[:100])
    ]
    topk_cand: dict = {}
    for sc in score_cols_cand:
        topk_cand[sc] = eval_candidate_topk(merged_rows, sc)

    for k in TOPK_VALS:
        m  = topk_cand[PRIMARY_SCORE_COL][k]
        print(
            f"  raw cand top{k:2d}: "
            f"retention={m['lesion_candidate_retention']:.4f} "
            f"pos_cand_cov={m['positive_candidate_coverage']:.4f} "
            f"pos_slice_cov={m['positive_slice_coverage']:.4f} "
            f"pat_all_sup_proxy={m['pat_all_sup_proxy']:.4f}"
        )

    # patch baseline
    print("\n  patch (RD-D1s) baseline top-k:")
    patch_base = patch_topk_baseline()
    for k in TOPK_VALS:
        if k in patch_base:
            print(f"    top{k:2d}: {patch_base[k]:.4f}")
    if not patch_base:
        print("    (데이터 없음)")

    # group baseline
    print("\n  fuzzy group-level baseline top-k:")
    group_base = group_topk_baseline()
    for k in TOPK_VALS:
        if k in group_base:
            print(f"    top{k:2d}: {group_base[k]:.4f}")
    if not group_base:
        print("    (데이터 없음 또는 그룹 merge 미완료)")

    # patch baseline 대비 candidate top10 악화 판정
    raw_cand_top10  = topk_cand[PRIMARY_SCORE_COL].get(10, {}).get("lesion_candidate_retention", 0.0)
    patch_top10     = patch_base.get(10, None)
    if patch_top10 is not None and raw_cand_top10 < patch_top10:
        fail_reasons.append(
            f"candidate top10 retention {raw_cand_top10} < patch baseline {patch_top10}"
        )

    # ── 4. track-level score 집계 ────────────────────────────────────────────
    print("\n[4] track-level score 집계")
    track_rows = aggregate_track_scores(merged_rows)
    total_tracks      = len(track_rows)
    total_pos_tracks  = sum(1 for t in track_rows if t.get("has_positive") == "True")
    print(f"  unique tracks: {total_tracks:,}  positive-containing: {total_pos_tracks:,}")

    # ── 5. track-level top-k 평가 ────────────────────────────────────────────
    print("\n[5] track-level top-k 평가")
    topk_track = eval_track_topk(track_rows, merged_rows, PRIMARY_TRACK_SCORE_COL)
    for k in TOPK_VALS:
        m = topk_track[k]
        print(
            f"  track top{k:2d}: "
            f"hit_rate={m['positive_containing_track_hit_rate']:.4f} "
            f"pos_cand_cov={m['positive_candidate_coverage_by_tracks']:.4f} "
            f"pos_slice_cov={m['positive_slice_coverage_by_tracks']:.4f} "
            f"pat_all_sup_proxy={m['pat_all_sup_proxy']:.4f}"
        )
    print("  [주의] track top-k 단위는 candidate top-k 와 다름 — 직접 수치 비교 caution")

    # ── 6. problem patient audit ─────────────────────────────────────────────
    print("\n[6] problem patient audit")
    audit_rows, audit_summary = build_problem_audit(merged_rows, track_rows)

    # problem patient 회복 판정 (top20 or top50 에서 cand hit)
    problem_recovery_ok = True
    for pid in PROBLEM_PATIENT_IDS:
        top20 = audit_summary.get(f"{pid}_cand_top20", "False")
        top50 = audit_summary.get(f"{pid}_cand_top50", "False")
        if top20 not in ("True",) and top50 not in ("True",):
            problem_recovery_ok = False
            fail_reasons.append(f"{pid}: cand top20/top50 모두 miss")

    # ── 7. 판정 ──────────────────────────────────────────────────────────────
    hard_fail_keys = [
        "merged row count", "duplicate", "missing", "extra",
        "NaN/Inf", "stage2_holdout", "existing_artifact", "baseline",
        "shard row count", "csv 없음", "DONE.json 없음",
    ]
    is_hard = any(
        any(k in r for k in hard_fail_keys) for r in fail_reasons
    )
    if not fail_reasons:
        verdict = "PASS"
    elif is_hard:
        verdict = "FAIL"
    else:
        verdict = "PARTIAL_PASS"

    # ── 8. 출력 파일 생성 ────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    _finalize(
        merged_rows=merged_rows,
        topk_cand=topk_cand,
        topk_track=topk_track,
        track_rows=track_rows,
        patch_base=patch_base,
        group_base=group_base,
        audit_rows=audit_rows,
        audit_summary=audit_summary,
        shard_row_counts=shard_row_counts,
        shard_summaries=shard_summaries,
        verdict=verdict,
        fail_reasons=fail_reasons,
        elapsed=elapsed,
        integrity={
            "merged_row_count":          merged_row_count,
            "duplicate_candidate_id":    dup_count,
            "missing_vs_manifest":       len(missing_vs_manifest),
            "extra_vs_manifest":         len(extra_vs_manifest),
            "shard_count_ok":            shard_count_ok,
            "score_nan_count":           nan_count,
            "score_inf_count":           inf_count,
            "total_failed_candidate_count": total_failed,
            "total_unique_tracks":       total_tracks,
            "total_positive_tracks":     total_pos_tracks,
        },
    )

    print("\n" + "=" * 70)
    print(f"[MERGE] 완료 ({elapsed:.1f}s)  판정: {verdict}")
    print("=" * 70)
    if verdict == "FAIL":
        sys.exit(1)


# =============================================================================
# 출력 파일 작성
# =============================================================================

def _finalize(
    merged_rows: list,
    topk_cand: dict,
    topk_track: dict,
    track_rows: list,
    patch_base: dict,
    group_base: dict,
    audit_rows: list,
    audit_summary: dict,
    shard_row_counts: dict,
    shard_summaries: dict,
    verdict: str,
    fail_reasons: list,
    elapsed: float,
    integrity: dict = None,
) -> None:
    integrity = integrity or {}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # merged CSV
    print("\n[A] merged CSV 저장")
    if merged_rows:
        fields   = list(merged_rows[0].keys())
        m_sorted = sorted(merged_rows, key=lambda r: r.get("candidate_id", ""))
        write_csv(MERGED_CSV, fields, m_sorted)

    # candidate top-k CSV
    print("\n[B] candidate top-k CSV 저장")
    cand_topk_rows: list = []
    for sc, kmap in topk_cand.items():
        for k in TOPK_VALS:
            m = kmap[k]
            cand_topk_rows.append({
                "score_col":                   sc,
                "k":                           k,
                "is_primary":                  str(sc == PRIMARY_SCORE_COL),
                "lesion_candidate_retention":  m.get("lesion_candidate_retention"),
                "patient_hit_rate":            m.get("patient_hit_rate"),
                "n_patients_with_positive":    m.get("n_patients_with_positive"),
                "positive_candidate_coverage": m.get("positive_candidate_coverage"),
                "positive_slice_coverage":     m.get("positive_slice_coverage"),
                "pat_all_sup_proxy":           m.get("pat_all_sup_proxy"),
                "patch_baseline_retention":    (
                    patch_base.get(k, "") if sc == PRIMARY_SCORE_COL else ""
                ),
                "group_baseline_retention":    (
                    group_base.get(k, "") if sc == PRIMARY_SCORE_COL else ""
                ),
            })
    if cand_topk_rows:
        write_csv(
            CANDIDATE_TOPK_CSV,
            [
                "score_col", "k", "is_primary",
                "lesion_candidate_retention", "patient_hit_rate",
                "n_patients_with_positive",
                "positive_candidate_coverage", "positive_slice_coverage",
                "pat_all_sup_proxy",
                "patch_baseline_retention", "group_baseline_retention",
            ],
            cand_topk_rows,
        )

    # track score summary CSV
    print("\n[C] track score summary CSV 저장")
    if track_rows:
        write_csv(TRACK_SUMMARY_CSV, TRACK_SUMMARY_FIELDS, track_rows)

    # track top-k CSV
    print("\n[D] track top-k CSV 저장")
    track_topk_rows: list = []
    for k in TOPK_VALS:
        m = topk_track.get(k, {})
        track_topk_rows.append({
            "k":                                      k,
            "positive_containing_track_hit_rate":     m.get("positive_containing_track_hit_rate"),
            "patient_hit_rate":                       m.get("patient_hit_rate"),
            "n_patients_with_positive_track":         m.get("n_patients_with_positive_track"),
            "positive_candidate_coverage_by_tracks":  m.get("positive_candidate_coverage_by_tracks"),
            "positive_slice_coverage_by_tracks":      m.get("positive_slice_coverage_by_tracks"),
            "pat_all_sup_proxy":                      m.get("pat_all_sup_proxy"),
            "caution":                                "track 단위 top-k, candidate 직접비교 금지",
        })
    if track_topk_rows:
        write_csv(
            TRACK_TOPK_CSV,
            [
                "k", "positive_containing_track_hit_rate", "patient_hit_rate",
                "n_patients_with_positive_track",
                "positive_candidate_coverage_by_tracks",
                "positive_slice_coverage_by_tracks",
                "pat_all_sup_proxy", "caution",
            ],
            track_topk_rows,
        )

    # problem patient audit CSV
    print("\n[E] problem patient audit CSV 저장")
    if audit_rows:
        write_csv(PROBLEM_AUDIT_CSV, AUDIT_FIELDS, audit_rows)

    # summary JSON
    print("\n[F] summary JSON 저장")
    raw_cand = topk_cand.get(PRIMARY_SCORE_COL, {})
    summary = {
        "verdict":                    verdict,
        "fail_reasons":               fail_reasons,
        "expected_total_candidates":  EXPECTED_TOTAL_CANDIDATES,
        "shard_row_counts":           shard_row_counts,
        "integrity":                  integrity,
        "candidate_topk_raw": {
            f"top{k}": raw_cand.get(k, {}) for k in TOPK_VALS
        },
        "track_topk_raw": {
            f"top{k}": topk_track.get(k, {}) for k in TOPK_VALS
        },
        "patch_baseline_topk": {
            f"top{k}": patch_base.get(k) for k in TOPK_VALS
        },
        "group_baseline_topk": {
            f"top{k}": group_base.get(k) for k in TOPK_VALS
        },
        "problem_patient_audit": audit_summary,
        "primary_score_col":          PRIMARY_SCORE_COL,
        "primary_track_score_col":    PRIMARY_TRACK_SCORE_COL,
        "preview_score_cols":         PREVIEW_SCORE_COLS,
        "track_topk_caution":         "track 단위 top-k 는 candidate 단위와 직접 비교 금지",
        "elapsed_sec":                round(elapsed, 1),
        "guardrails":                 GUARDRAILS,
        "shard_summaries":            shard_summaries,
    }
    ensure_output_path_safe(SUMMARY_JSON)
    with open(str(SUMMARY_JSON), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  saved: {SUMMARY_JSON}")

    # report MD
    print("\n[G] report MD 저장")
    lines = [
        "# RD4AD strict same-position z-track actual scoring report v1",
        "",
        f"**판정: {verdict}**",
        "",
        "## 무결성",
        "",
    ]
    for k, v in integrity.items():
        lines.append(f"- {k}: {v}")

    lines += ["", "## candidate-level top-k (primary: rd4ad_ztrack_score_raw)", ""]
    header = (
        "| k | retention | pos_cand_cov | pos_slice_cov "
        "| pat_all_sup | patch_base | group_base |"
    )
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|")
    for k in TOPK_VALS:
        m   = raw_cand.get(k, {})
        pb  = patch_base.get(k, "N/A")
        gb  = group_base.get(k, "N/A")
        lines.append(
            f"| top{k} "
            f"| {m.get('lesion_candidate_retention')} "
            f"| {m.get('positive_candidate_coverage')} "
            f"| {m.get('positive_slice_coverage')} "
            f"| {m.get('pat_all_sup_proxy')} "
            f"| {pb} "
            f"| {gb} |"
        )

    lines += [
        "",
        "## track-level top-k (primary: track_score_max)",
        "",
        "> **caution**: track 단위 top-k 는 candidate 단위와 직접 수치 비교 금지.",
        "",
        "| k | track_hit_rate | pos_cand_cov_by_tracks | pos_slice_cov_by_tracks | pat_all_sup |",
        "|---|---|---|---|---|",
    ]
    for k in TOPK_VALS:
        m = topk_track.get(k, {})
        lines.append(
            f"| top{k} "
            f"| {m.get('positive_containing_track_hit_rate')} "
            f"| {m.get('positive_candidate_coverage_by_tracks')} "
            f"| {m.get('positive_slice_coverage_by_tracks')} "
            f"| {m.get('pat_all_sup_proxy')} |"
        )

    lines += ["", "## problem patient audit", ""]
    for pid in PROBLEM_PATIENT_IDS:
        cand_cells  = [
            f"cand_top{k}={audit_summary.get(f'{pid}_cand_top{k}', 'N/A')}"
            for k in TOPK_VALS
        ]
        track_cells = [
            f"track_top{k}={audit_summary.get(f'{pid}_track_top{k}', 'N/A')}"
            for k in TOPK_VALS
        ]
        lines.append(f"- {pid}: " + "  ".join(cand_cells))
        lines.append(f"         " + "  ".join(track_cells))

    lines += ["", "## fail reasons", ""]
    if fail_reasons:
        for r in fail_reasons:
            lines.append(f"- {r}")
    else:
        lines.append("- 없음")

    lines += ["", "## guardrails", ""]
    for k, v in GUARDRAILS.items():
        lines.append(f"- {k}: {v}")

    lines += [
        "",
        "## 비고",
        "- P1_times_roi / P2_times_sqrt_roi 는 preview only (primary 는 rd4ad_ztrack_score_raw).",
        "- track_score_max 가 track primary score.",
        "- track-level top-k 는 단위가 다르므로 candidate top-k 와 직접 수치 비교 금지.",
        "- label 은 evaluation 전용. scoring selection / candidate 삭제에 사용 금지.",
    ]
    ensure_output_path_safe(REPORT_MD)
    with open(str(REPORT_MD), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  saved: {REPORT_MD}")

    # DONE_FULL_MERGE.json
    print("\n[H] DONE_FULL_MERGE.json 저장")
    ensure_output_path_safe(DONE_FULL_MERGE_JSON)
    with open(str(DONE_FULL_MERGE_JSON), "w", encoding="utf-8") as f:
        json.dump(
            {
                "verdict":    verdict,
                "stage":      "strict_ztrack_actual_scoring_merge",
                "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
                "fail_count": len(fail_reasons),
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"  saved: {DONE_FULL_MERGE_JSON}")


# =============================================================================
# main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RD4AD strict z-track actual scoring MERGE v1"
    )
    parser.add_argument("--dry-run",                action="store_true")
    parser.add_argument("--merge-shards",           action="store_true")
    parser.add_argument("--confirm-readonly",       action="store_true")
    parser.add_argument("--confirm-stage1dev-only", action="store_true")
    args = parser.parse_args()

    if not any([args.dry_run, args.merge_shards]):
        print(
            "[ABORT] bare run 차단. --dry-run 또는 --merge-shards 를 사용하세요.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.dry_run:
        run_dry()
        return

    if args.merge_shards:
        if not args.confirm_readonly:
            print("[ABORT] --confirm-readonly 필요", file=sys.stderr)
            sys.exit(2)
        if not args.confirm_stage1dev_only:
            print("[ABORT] --confirm-stage1dev-only 필요", file=sys.stderr)
            sys.exit(2)
        run_merge()
        return


if __name__ == "__main__":
    main()
