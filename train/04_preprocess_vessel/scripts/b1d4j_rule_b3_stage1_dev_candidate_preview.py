"""
B1-D4j: Rule-B3 x0.5 stage1_dev candidate-level preview
감점 대상: R001/R015/R016/R028 (4개, B_boundary overlap artifact)
나머지 761,202행: adjusted_score_preview = original_score

실행 방법:
  --dry-run    헤더/스키마/row count/join target 검증만, full output 생성 없음
  --sample-run join target 4개 매칭 검증 + 샘플 output 생성 (full output 생성 없음)
  --run        실제 full preview 실행 (ALLOW_REAL_PROCESSING=True 필요)

ALLOW_REAL_PROCESSING 원본 수정 금지.
real 실행은 importlib runtime override 방식으로만 가능:

  python - <<'PY'
  import importlib.util, sys
  from pathlib import Path

  script = Path("scripts/b1d4j_rule_b3_stage1_dev_candidate_preview.py")
  spec = importlib.util.spec_from_file_location("b1d4j", script)
  m = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(m)

  m.ALLOW_REAL_PROCESSING = True
  sys.argv = [str(script), "--run", "--chunk-size", "50000"]
  m.main()
  PY

실행 후 원본 파일 무수정 확인:
  grep "ALLOW_REAL_PROCESSING = False" scripts/b1d4j_rule_b3_stage1_dev_candidate_preview.py
"""
import argparse
import csv
import json
import os
import sys
import time

# =====================================================================
# 안전 차단: ALLOW_REAL_PROCESSING = False 동안 --run 모드 실행 불가
# =====================================================================
ALLOW_REAL_PROCESSING = False

# =====================================================================
# 경로 상수
# =====================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATCH_CANDIDATES_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs/position-aware-padim-v1/candidates/"
    "padim_v2_roi0_0_explanation_candidates_v1/patch_candidates.csv"
)
B1D1_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1/b1d1_fp_cause_diagnostic.csv"
)
B1D2_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1/b1d2_candidate_groups_preview.csv"
)
SAFETY_SENTINELS_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1/"
    "b1d4i_rule_b3_stage1_dev_full_preflight_safety_sentinels.csv"
)
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"
)
OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "b1d4j_rule_b3_soft_penalty_0_5_stage1_dev_candidate_preview.csv"
)
OUTPUT_SAMPLE_CSV = os.path.join(
    OUTPUT_DIR,
    "b1d4j0_dryrun_sample_validation.csv"
)
OUTPUT_SUMMARY_JSON = os.path.join(
    OUTPUT_DIR,
    "b1d4j_rule_b3_stage1_dev_candidate_preview_summary.json"
)
OUTPUT_REPORT_MD = os.path.join(
    OUTPUT_DIR,
    "b1d4j_rule_b3_stage1_dev_candidate_preview_report.md"
)

# =====================================================================
# 감점 파라미터
# =====================================================================
PENALTY_ALPHA = 0.5
PENALIZED_REVIEW_IDS = {"R001", "R015", "R016", "R028"}
EXPECTED_JOIN_MATCH_COUNT = 4

# stage2/holdout 차단 값
BLOCKED_STAGE_FLAGS = {"stage2", "holdout", "test_split"}

# =====================================================================
# B1D1에서 join target dict 구성
# key: (patient_id, local_z, y0, x0) -> dict with review_id, cause_class 등
# =====================================================================
def load_join_targets():
    """R001/R015/R016/R028 4개의 join key를 b1d1 CSV에서 로드."""
    targets = {}
    with open(B1D1_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row["review_id"]
            if rid not in PENALIZED_REVIEW_IDS:
                continue
            key = (
                row["patient_id"],
                row["candidate_local_z"],
                row["candidate_y0"],
                row["candidate_x0"],
            )
            targets[key] = {
                "review_id": rid,
                "cause_class": row.get("cause_class", ""),
                "refined_roi_ratio": row.get("refined_roi_ratio", ""),
                "center_in_refined_roi": row.get("center_in_refined_roi", ""),
                "safety_role": row.get("safety_role", ""),
            }
    return targets


def load_b1d2_labels():
    """b1d2에서 review_id -> best_visual_label 매핑 로드."""
    labels = {}
    with open(B1D2_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("review_id", "")
            if rid:
                labels[rid] = row.get("best_visual_label", "")
    return labels


def load_sentinel_set():
    """sentinel review_id 집합 로드."""
    sentinels = set()
    with open(SAFETY_SENTINELS_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentinels.add(row["review_id"])
    return sentinels


# =====================================================================
# 스키마 검증
# =====================================================================
REQUIRED_COLUMNS = {
    "patient_id", "local_z", "y0", "x0",
    "padim_score", "stage_split_safety_flag",
    "candidate_patch_id", "group", "label",
    "roi_0_0_patch_ratio", "position_bin", "z_level", "z_ratio",
}

def validate_schema(header):
    missing = REQUIRED_COLUMNS - set(header)
    if missing:
        print(f"[ERROR] 필수 컬럼 누락: {missing}", file=sys.stderr)
        return False
    return True


# =====================================================================
# 출력 컬럼 정의 (B1-D4i b1d4j_output_schema 준수)
# =====================================================================
OUTPUT_COLUMNS = [
    "row_id", "source_file", "patient_id", "candidate_id",
    "original_score", "adjusted_score_preview",
    "score_delta", "score_delta_percent",
    "rule_b3_candidate", "soft_penalty_applied", "penalty_alpha",
    "cause_class", "roi_0_0_patch_ratio", "refined_roi_ratio",
    "center_in_refined_roi", "visual_label",
    "safety_role", "review_id",
    "protected_hard_case", "protected_lesion_safety",
    "protected_gate_candidate", "protected_observation",
    "protected_unreviewed_boundary", "protected_ad_wall_med", "protected_ad_other",
    "fail_condition_hit", "fail_reason",
    "stage_split_safety_flag", "holdout_flag",
    "local_z", "y0", "x0",
]

HARD_CASE_RIDS = {"R018", "R024"}
UNREVIEWED_HOLD_RIDS = {"R020", "R021", "R025", "R027", "R029"}
GATE_CANDIDATE_RIDS = {"R002", "R005", "R006", "R009", "R012", "R030"}


def build_output_row(
    row_id, row, join_info, b1d2_label,
    is_penalized, fail_condition_hit, fail_reason,
    sentinel_set
):
    pid = row["patient_id"]
    orig_score = float(row["padim_score"])
    stage_flag = row["stage_split_safety_flag"]
    review_id = join_info.get("review_id", "") if join_info else ""
    cause_class = join_info.get("cause_class", "") if join_info else ""
    refined_roi_ratio = join_info.get("refined_roi_ratio", "") if join_info else ""
    center_in_refined_roi = join_info.get("center_in_refined_roi", "") if join_info else ""
    safety_role = join_info.get("safety_role", "") if join_info else ""
    visual_label = b1d2_label if b1d2_label else ""

    adj_score = orig_score * PENALTY_ALPHA if is_penalized else orig_score
    delta = adj_score - orig_score if is_penalized else 0.0
    delta_pct = (delta / orig_score * 100) if (is_penalized and orig_score != 0) else 0.0

    protected_hard_case = "true" if review_id in HARD_CASE_RIDS else "false"
    protected_unreviewed = "true" if review_id in UNREVIEWED_HOLD_RIDS else "false"
    protected_gate = "true" if review_id in GATE_CANDIDATE_RIDS else "false"
    protected_lesion = "true" if (review_id and review_id.startswith("R") and
                                   review_id not in PENALIZED_REVIEW_IDS and
                                   review_id not in HARD_CASE_RIDS and
                                   review_id not in UNREVIEWED_HOLD_RIDS and
                                   review_id not in GATE_CANDIDATE_RIDS) else "false"
    protected_obs = "false"
    protected_ad_wall_med = "false"
    protected_ad_other = "false"

    holdout_flag = "1" if stage_flag in BLOCKED_STAGE_FLAGS else "0"

    return {
        "row_id": row_id,
        "source_file": "patch_candidates.csv",
        "patient_id": pid,
        "candidate_id": row.get("candidate_patch_id", ""),
        "original_score": f"{orig_score:.6f}",
        "adjusted_score_preview": f"{adj_score:.6f}",
        "score_delta": f"{delta:.6f}",
        "score_delta_percent": f"{delta_pct:.2f}",
        "rule_b3_candidate": "true" if is_penalized else "false",
        "soft_penalty_applied": "true" if is_penalized else "false",
        "penalty_alpha": PENALTY_ALPHA if is_penalized else "",
        "cause_class": cause_class,
        "roi_0_0_patch_ratio": row.get("roi_0_0_patch_ratio", ""),
        "refined_roi_ratio": refined_roi_ratio,
        "center_in_refined_roi": center_in_refined_roi,
        "visual_label": visual_label,
        "safety_role": safety_role,
        "review_id": review_id,
        "protected_hard_case": protected_hard_case,
        "protected_lesion_safety": protected_lesion,
        "protected_gate_candidate": protected_gate,
        "protected_observation": protected_obs,
        "protected_unreviewed_boundary": protected_unreviewed,
        "protected_ad_wall_med": protected_ad_wall_med,
        "protected_ad_other": protected_ad_other,
        "fail_condition_hit": "true" if fail_condition_hit else "false",
        "fail_reason": fail_reason,
        "stage_split_safety_flag": stage_flag,
        "holdout_flag": holdout_flag,
        "local_z": row.get("local_z", ""),
        "y0": row.get("y0", ""),
        "x0": row.get("x0", ""),
    }


# =====================================================================
# dry-run 모드
# =====================================================================
def run_dry(chunk_size):
    print("[dry-run] 입력 파일 및 스키마 검증 시작...")
    t0 = time.time()

    # 1. 입력 파일 존재 확인
    for path in [PATCH_CANDIDATES_PATH, B1D1_PATH, B1D2_PATH, SAFETY_SENTINELS_PATH]:
        if not os.path.exists(path):
            print(f"[ERROR] 파일 없음: {path}", file=sys.stderr)
            sys.exit(1)
    print("[OK] 모든 입력 파일 존재")

    # 2. 출력 collision guard
    if os.path.exists(OUTPUT_CSV):
        print(f"[BLOCKED] 출력 파일 이미 존재: {OUTPUT_CSV}", file=sys.stderr)
        sys.exit(1)
    print("[OK] output collision guard 통과")

    # 3. join target 로드 및 검증
    join_targets = load_join_targets()
    print(f"[OK] join target dict 로드: {len(join_targets)}개")
    if len(join_targets) != EXPECTED_JOIN_MATCH_COUNT:
        print(f"[ERROR] join target 개수 불일치: {len(join_targets)} != {EXPECTED_JOIN_MATCH_COUNT}", file=sys.stderr)
        sys.exit(1)
    for key, val in join_targets.items():
        print(f"  -> {val['review_id']}: patient=...{key[0][-20:]}, z={key[1]}, y0={key[2]}, x0={key[3]}")

    # 4. sentinel 로드 및 overlap 확인
    sentinel_set = load_sentinel_set()
    overlap = PENALIZED_REVIEW_IDS & sentinel_set
    if overlap:
        print(f"[ERROR] penalized target이 sentinel과 겹침: {overlap}", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] sentinel({len(sentinel_set)}개) overlap 없음")

    # 5. patch_candidates 헤더 확인
    with open(PATCH_CANDIDATES_PATH, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    if not validate_schema(header):
        sys.exit(1)
    print(f"[OK] schema 검증 통과 ({len(header)} 컬럼)")

    # 6. patch_candidates 전체 streaming scan (join match count 확인)
    print(f"[dry-run] patch_candidates.csv 전체 scan 시작 (chunk_size={chunk_size})...")
    size_mb = os.path.getsize(PATCH_CANDIDATES_PATH) / (1024 * 1024)
    print(f"  파일 크기: {size_mb:.1f} MB")

    total_rows = 0
    holdout_rows = 0
    nan_score_count = 0
    join_match_count = 0
    match_found_rids = set()
    duplicate_match = False

    with open(PATCH_CANDIDATES_PATH, newline="") as f:
        reader = csv.DictReader(f)
        chunk = []
        for row in reader:
            total_rows += 1
            chunk.append(row)
            if len(chunk) >= chunk_size:
                # chunk 처리 (dry-run: 카운팅만)
                for r in chunk:
                    sf = r["stage_split_safety_flag"]
                    if sf in BLOCKED_STAGE_FLAGS:
                        holdout_rows += 1
                    try:
                        score = float(r["padim_score"])
                        if score != score:  # NaN check
                            nan_score_count += 1
                    except (ValueError, TypeError):
                        nan_score_count += 1
                    key = (r["patient_id"], r["local_z"], r["y0"], r["x0"])
                    if key in join_targets:
                        rid = join_targets[key]["review_id"]
                        if rid in match_found_rids:
                            duplicate_match = True
                        match_found_rids.add(rid)
                        join_match_count += 1
                chunk = []

        # 나머지 처리
        for r in chunk:
            sf = r["stage_split_safety_flag"]
            if sf in BLOCKED_STAGE_FLAGS:
                holdout_rows += 1
            try:
                score = float(r["padim_score"])
                if score != score:
                    nan_score_count += 1
            except (ValueError, TypeError):
                nan_score_count += 1
            key = (r["patient_id"], r["local_z"], r["y0"], r["x0"])
            if key in join_targets:
                rid = join_targets[key]["review_id"]
                if rid in match_found_rids:
                    duplicate_match = True
                match_found_rids.add(rid)
                join_match_count += 1

    elapsed = time.time() - t0
    print(f"\n[dry-run scan 완료] elapsed={elapsed:.1f}s")
    print(f"  total_rows: {total_rows}")
    print(f"  holdout_rows: {holdout_rows}")
    print(f"  nan_score_count: {nan_score_count}")
    print(f"  join_match_count: {join_match_count}")
    print(f"  match_found_rids: {match_found_rids}")
    print(f"  duplicate_match: {duplicate_match}")

    # 검증
    if holdout_rows > 0:
        print(f"[ERROR] holdout/stage2 행 발견: {holdout_rows}행", file=sys.stderr)
        sys.exit(1)
    print("[OK] holdout 행 없음")

    if join_match_count != EXPECTED_JOIN_MATCH_COUNT:
        print(f"[BLOCKED] join_match_count={join_match_count} != {EXPECTED_JOIN_MATCH_COUNT}", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] join_match_count = {join_match_count}")

    if duplicate_match:
        print("[ERROR] 중복 매칭 발생", file=sys.stderr)
        sys.exit(1)
    print("[OK] 중복 매칭 없음")

    missing_rids = PENALIZED_REVIEW_IDS - match_found_rids
    if missing_rids:
        print(f"[BLOCKED] 일부 penalized target 미매칭: {missing_rids}", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] 4개 penalized target 모두 매칭: {match_found_rids}")

    print("\n[dry-run] 모든 검증 통과")
    return {
        "total_rows": total_rows,
        "holdout_rows": holdout_rows,
        "join_match_count": join_match_count,
        "nan_score_count": nan_score_count,
        "duplicate_match": duplicate_match,
        "match_found_rids": sorted(match_found_rids),
        "elapsed_seconds": round(elapsed, 1),
    }


# =====================================================================
# sample-run 모드 (penalized 4개 + 주변 행 샘플 출력, full output 금지)
# =====================================================================
def run_sample(chunk_size):
    print("[sample-run] join target 4개 매칭 검증 및 샘플 출력...")

    # 출력 파일 collision guard
    if os.path.exists(OUTPUT_SAMPLE_CSV):
        print(f"[BLOCKED] 샘플 파일 이미 존재: {OUTPUT_SAMPLE_CSV}", file=sys.stderr)
        sys.exit(1)
    if os.path.exists(OUTPUT_CSV):
        print(f"[BLOCKED] full output 파일 이미 존재: {OUTPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    join_targets = load_join_targets()
    b1d2_labels = load_b1d2_labels()
    sentinel_set = load_sentinel_set()

    sample_rows = []
    penalized_count = 0
    match_found_rids = set()
    total_rows = 0
    holdout_rows = 0

    with open(PATCH_CANDIDATES_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row_id, row in enumerate(reader, start=1):
            total_rows += 1
            sf = row["stage_split_safety_flag"]
            if sf in BLOCKED_STAGE_FLAGS:
                holdout_rows += 1
                print(f"[ERROR] holdout 행 발견 row_id={row_id}", file=sys.stderr)
                sys.exit(1)

            key = (row["patient_id"], row["local_z"], row["y0"], row["x0"])
            join_info = join_targets.get(key)
            is_penalized = join_info is not None
            fail_condition_hit = False
            fail_reason = ""

            # safety sentinel 감점 방지
            if is_penalized:
                rid = join_info["review_id"]
                if rid in sentinel_set:
                    print(f"[ERROR] sentinel review_id가 penalized target: {rid}", file=sys.stderr)
                    sys.exit(1)
                match_found_rids.add(rid)
                penalized_count += 1

            if is_penalized or len(sample_rows) < 10:
                b1d2_label = ""
                if join_info:
                    b1d2_label = b1d2_labels.get(join_info["review_id"], "")
                out_row = build_output_row(
                    row_id, row, join_info, b1d2_label,
                    is_penalized, fail_condition_hit, fail_reason,
                    sentinel_set
                )
                sample_rows.append(out_row)

    print(f"\n[sample-run] total_rows={total_rows}")
    print(f"  penalized_count={penalized_count}")
    print(f"  match_found_rids={match_found_rids}")
    print(f"  holdout_rows={holdout_rows}")

    if penalized_count != EXPECTED_JOIN_MATCH_COUNT:
        print(f"[BLOCKED] penalized_count={penalized_count} != {EXPECTED_JOIN_MATCH_COUNT}", file=sys.stderr)
        sys.exit(1)
    print("[OK] join_match_count = 4")

    # 샘플 출력 저장 (penalized 4개 + 일반 최대 10개)
    with open(OUTPUT_SAMPLE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for r in sample_rows:
            writer.writerow(r)
    print(f"[OK] 샘플 파일 생성: {OUTPUT_SAMPLE_CSV} ({len(sample_rows)}행)")
    print("\n[sample-run] 완료")
    return {
        "total_rows": total_rows,
        "penalized_count": penalized_count,
        "holdout_rows": holdout_rows,
        "match_found_rids": sorted(match_found_rids),
        "sample_rows": len(sample_rows),
    }


# =====================================================================
# full run 후 summary JSON / report MD 생성
# =====================================================================
def write_summary_json(counters, match_found_rids, duplicate_match, chunk_size, elapsed, fail_reasons):
    if os.path.exists(OUTPUT_SUMMARY_JSON):
        print(f"[BLOCKED] summary JSON 이미 존재: {OUTPUT_SUMMARY_JSON}", file=sys.stderr)
        sys.exit(1)
    summary = {
        "stage": "B1-D4j",
        "stage2_holdout_access": 0,
        "total_rows": counters["total_rows"],
        "output_rows": counters["total_rows"],
        "penalized_rows": counters["penalized_rows"],
        "penalized_review_ids": sorted(PENALIZED_REVIEW_IDS),
        "match_found_rids": sorted(match_found_rids),
        "duplicate_match": duplicate_match,
        "holdout_rows": counters["holdout_rows"],
        "nan_score_count": counters["nan_score_count"],
        "invalid_coord_count": counters["invalid_coord_count"],
        "sentinel_overlap_fail": counters["sentinel_overlap_fail"],
        "chunk_size": chunk_size,
        "elapsed_seconds": round(elapsed, 1),
        "score_modified": False,
        "threshold_recomputed": False,
        "adjusted_score_created": False,
        "adjusted_score_preview_created": True,
        "suppression_weight_created": False,
        "refined_score_created": False,
        "output_csv_path": OUTPUT_CSV,
        "fail_count": len(fail_reasons),
        "fail_reasons": fail_reasons,
        "verdict": "PASS" if not fail_reasons else "FAIL",
    }
    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] summary JSON 생성: {OUTPUT_SUMMARY_JSON}")
    return summary


def write_report_md(summary):
    if os.path.exists(OUTPUT_REPORT_MD):
        print(f"[BLOCKED] report MD 이미 존재: {OUTPUT_REPORT_MD}", file=sys.stderr)
        sys.exit(1)
    verdict = summary["verdict"]
    lines = [
        f"# B1-D4j Rule-B3 ×0.5 stage1_dev candidate-level preview",
        f"",
        f"**판정: {verdict}**",
        f"",
        f"---",
        f"",
        f"## 입력 파일",
        f"",
        f"| 파일 | 경로 |",
        f"|------|------|",
        f"| patch_candidates.csv | {PATCH_CANDIDATES_PATH} |",
        f"| b1d1_fp_cause_diagnostic.csv | {B1D1_PATH} |",
        f"| b1d2_candidate_groups_preview.csv | {B1D2_PATH} |",
        f"| safety_sentinels.csv | {SAFETY_SENTINELS_PATH} |",
        f"",
        f"## 생성 파일",
        f"",
        f"| 파일 | 설명 |",
        f"|------|------|",
        f"| b1d4j_rule_b3_soft_penalty_0_5_stage1_dev_candidate_preview.csv | 761,206행 full preview |",
        f"| b1d4j_rule_b3_stage1_dev_candidate_preview_summary.json | 실행 결과 요약 |",
        f"| b1d4j_rule_b3_stage1_dev_candidate_preview_report.md | 이 파일 |",
        f"",
        f"---",
        f"",
        f"## 실행 결과 요약",
        f"",
        f"| 항목 | 값 |",
        f"|------|-----|",
        f"| stage2_holdout 접근 여부 | **없음 (0행)** |",
        f"| patch_candidates rows | {summary['total_rows']:,} |",
        f"| output rows | {summary['output_rows']:,} |",
        f"| chunk size | {summary['chunk_size']:,} |",
        f"| penalized rows | **{summary['penalized_rows']}** |",
        f"| penalized review_ids | {', '.join(summary['penalized_review_ids'])} |",
        f"| match_found_rids | {', '.join(summary['match_found_rids'])} |",
        f"| duplicate match | {summary['duplicate_match']} |",
        f"| holdout rows | {summary['holdout_rows']} |",
        f"| NaN score count | {summary['nan_score_count']} |",
        f"| sentinel overlap fail | {summary['sentinel_overlap_fail']} |",
        f"| elapsed (sec) | {summary['elapsed_seconds']} |",
        f"",
        f"## score / threshold / ROI 무수정 확인",
        f"",
        f"| 항목 | 값 |",
        f"|------|-----|",
        f"| score_modified | {summary['score_modified']} |",
        f"| threshold_recomputed | {summary['threshold_recomputed']} |",
        f"| adjusted_score 실제 생성 | {summary['adjusted_score_created']} |",
        f"| adjusted_score_preview 생성 | {summary['adjusted_score_preview_created']} (출력 CSV 내부 전용) |",
        f"| suppression_weight_created | {summary['suppression_weight_created']} |",
        f"| refined_score_created | {summary['refined_score_created']} |",
        f"",
        f"original_score 컬럼은 patch_candidates.csv 원본 값 그대로 유지.",
        f"adjusted_score_preview는 이 출력 CSV 내부 전용 컬럼이며 실제 adjusted_score가 아님.",
        f"",
        f"## 한계",
        f"",
        f"- **candidate-level preview일 뿐**: 4개 감점이 실제 임상 성능에 미치는 효과는 별도 검증 필요.",
        f"- **threshold 재계산 없음**: 감점 적용 후 threshold를 재산출하지 않았다.",
        f"- **FROC/AUROC 아님**: 이 파일은 rank/score preview 전용이며 성능 지표가 아니다.",
        f"- **stage2_holdout 미사용**: 검증용 holdout 데이터는 접근하지 않았다.",
        f"",
        f"## 다음 단계",
        f"",
        f"B1-D4k candidate-level preview validation/checkpoint",
        f"",
        f"---",
        f"",
        f"*generated by b1d4j_rule_b3_stage1_dev_candidate_preview.py*",
    ]
    with open(OUTPUT_REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] report MD 생성: {OUTPUT_REPORT_MD}")


# =====================================================================
# full run 모드
# =====================================================================
def run_full(chunk_size):
    if not ALLOW_REAL_PROCESSING:
        print("[BLOCKED] ALLOW_REAL_PROCESSING=False. 원본 파일 수정 금지.", file=sys.stderr)
        print("  importlib runtime override 방식으로만 실행 가능:", file=sys.stderr)
        print("    python - <<'PY'", file=sys.stderr)
        print("    import importlib.util, sys; from pathlib import Path", file=sys.stderr)
        print("    script = Path('scripts/b1d4j_rule_b3_stage1_dev_candidate_preview.py')", file=sys.stderr)
        print("    spec = importlib.util.spec_from_file_location('b1d4j', script)", file=sys.stderr)
        print("    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)", file=sys.stderr)
        print("    m.ALLOW_REAL_PROCESSING = True", file=sys.stderr)
        print("    sys.argv = [str(script), '--run', '--chunk-size', '50000']; m.main()", file=sys.stderr)
        print("    PY", file=sys.stderr)
        sys.exit(2)

    # 출력 collision precheck (처리 시작 전 4개 파일 전부 확인)
    OUTPUT_CSV_TMP = OUTPUT_CSV + ".tmp"
    _collision_targets = [
        ("OUTPUT_CSV",          OUTPUT_CSV),
        ("OUTPUT_CSV_TMP",      OUTPUT_CSV_TMP),
        ("OUTPUT_SUMMARY_JSON", OUTPUT_SUMMARY_JSON),
        ("OUTPUT_REPORT_MD",    OUTPUT_REPORT_MD),
    ]
    _collisions = [(name, path) for name, path in _collision_targets if os.path.exists(path)]
    if _collisions:
        print("[BLOCKED] run_full 시작 전 출력 파일 충돌 감지. 기존 파일 삭제/덮어쓰기 금지.", file=sys.stderr)
        for name, path in _collisions:
            print(f"  {name}: {path}", file=sys.stderr)
        print("  새 버전명을 사용하거나 기존 파일을 수동으로 정리 후 재실행하세요.", file=sys.stderr)
        sys.exit(1)
    print("[OK] output collision precheck 통과 (4개 파일 모두 미존재)")

    join_targets = load_join_targets()
    b1d2_labels = load_b1d2_labels()
    sentinel_set = load_sentinel_set()

    counters = {
        "total_rows": 0,
        "penalized_rows": 0,
        "protected_penalty_count": 0,
        "holdout_rows": 0,
        "nan_score_count": 0,
        "invalid_coord_count": 0,
        "sentinel_overlap_fail": 0,
    }
    match_found_rids = set()
    duplicate_match = False
    fail_triggered = False

    t0 = time.time()
    try:
        with open(OUTPUT_CSV_TMP, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()

            with open(PATCH_CANDIDATES_PATH, newline="") as f:
                reader = csv.DictReader(f)
                chunk = []
                row_id = 0

                def process_chunk(chunk):
                    nonlocal fail_triggered, duplicate_match
                    rows_out = []
                    for row in chunk:
                        counters["total_rows"] += 1
                        row_id_local = counters["total_rows"]

                        sf = row["stage_split_safety_flag"]
                        if sf in BLOCKED_STAGE_FLAGS:
                            counters["holdout_rows"] += 1
                            fail_triggered = True
                            print(f"[ERROR] holdout 행 발견 row={row_id_local}", file=sys.stderr)
                            sys.exit(1)

                        try:
                            score = float(row["padim_score"])
                            if score != score:
                                counters["nan_score_count"] += 1
                        except (ValueError, TypeError):
                            counters["nan_score_count"] += 1
                            score = 0.0

                        key = (row["patient_id"], row["local_z"], row["y0"], row["x0"])
                        join_info = join_targets.get(key)
                        is_penalized = join_info is not None

                        if is_penalized:
                            rid = join_info["review_id"]
                            if rid in sentinel_set:
                                counters["sentinel_overlap_fail"] += 1
                                fail_triggered = True
                                print(f"[ERROR] sentinel overlap: {rid}", file=sys.stderr)
                                sys.exit(1)
                            if rid in match_found_rids:
                                duplicate_match = True
                                fail_triggered = True
                                print(f"[ERROR] 중복 매칭: {rid}", file=sys.stderr)
                                sys.exit(1)
                            match_found_rids.add(rid)
                            counters["penalized_rows"] += 1

                        b1d2_label = ""
                        if join_info:
                            b1d2_label = b1d2_labels.get(join_info["review_id"], "")

                        out_row = build_output_row(
                            row_id_local, row, join_info, b1d2_label,
                            is_penalized, False, "",
                            sentinel_set
                        )
                        rows_out.append(out_row)
                    return rows_out

                for row in reader:
                    chunk.append(row)
                    if len(chunk) >= chunk_size:
                        out_rows = process_chunk(chunk)
                        for r in out_rows:
                            writer.writerow(r)
                        chunk = []
                        elapsed = time.time() - t0
                        print(f"  processed {counters['total_rows']:,} rows... ({elapsed:.0f}s)")

                if chunk:
                    out_rows = process_chunk(chunk)
                    for r in out_rows:
                        writer.writerow(r)

    except Exception as e:
        if os.path.exists(OUTPUT_CSV_TMP):
            os.remove(OUTPUT_CSV_TMP)
        print(f"[ERROR] 처리 중 오류: {e}", file=sys.stderr)
        sys.exit(1)

    # 최종 검증
    if counters["penalized_rows"] != EXPECTED_JOIN_MATCH_COUNT:
        if os.path.exists(OUTPUT_CSV_TMP):
            os.remove(OUTPUT_CSV_TMP)
        print(f"[BLOCKED] penalized_rows={counters['penalized_rows']} != {EXPECTED_JOIN_MATCH_COUNT}", file=sys.stderr)
        sys.exit(1)

    # 완료 후 rename
    os.rename(OUTPUT_CSV_TMP, OUTPUT_CSV)
    elapsed = time.time() - t0
    print(f"\n[full-run 완료] elapsed={elapsed:.1f}s")
    print(f"  total_rows: {counters['total_rows']}")
    print(f"  penalized_rows: {counters['penalized_rows']}")
    print(f"  holdout_rows: {counters['holdout_rows']}")
    print(f"  output: {OUTPUT_CSV}")

    # summary JSON + report MD 생성
    summary = write_summary_json(counters, match_found_rids, duplicate_match, chunk_size, elapsed, [])
    write_report_md(summary)

    # 원본 파일 ALLOW_REAL_PROCESSING 무수정 확인
    script_path = os.path.join(PROJECT_ROOT, "scripts", "b1d4j_rule_b3_stage1_dev_candidate_preview.py")
    with open(script_path, encoding="utf-8") as f:
        src = f.read()
    if "ALLOW_REAL_PROCESSING = False" not in src:
        print("[WARNING] 원본 파일에서 ALLOW_REAL_PROCESSING = False를 찾을 수 없음!", file=sys.stderr)
    else:
        print("[OK] 원본 파일 ALLOW_REAL_PROCESSING = False 유지 확인")

    return counters


# =====================================================================
# main
# =====================================================================
def main():
    # bare-run 차단 (인자 없이 실행하면 exit 2)
    if len(sys.argv) < 2:
        print("[BLOCKED] 인자 없이 실행 불가. --dry-run / --sample-run / --run 중 하나를 지정하세요.", file=sys.stderr)
        sys.exit(2)

    parser = argparse.ArgumentParser(description="B1-D4j Rule-B3 stage1_dev candidate preview")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="스키마/join/chunk plan 검증만 (output 생성 없음)")
    group.add_argument("--sample-run", action="store_true", help="join 매칭 4개 검증 + 샘플 output 생성")
    group.add_argument("--run", action="store_true", help="full preview 실행 (ALLOW_REAL_PROCESSING=True 필요)")
    parser.add_argument("--chunk-size", type=int, default=50000, help="chunk 크기 (기본 50000)")
    args = parser.parse_args()

    chunk_size = args.chunk_size

    if args.dry_run:
        result = run_dry(chunk_size)
        print(f"\n[DRY-RUN PASS] 결과: {json.dumps(result, indent=2, ensure_ascii=False)}")
    elif args.sample_run:
        result = run_sample(chunk_size)
        print(f"\n[SAMPLE-RUN PASS] 결과: {json.dumps(result, indent=2, ensure_ascii=False)}")
    elif args.run:
        result = run_full(chunk_size)
        print(f"\n[FULL-RUN PASS] 결과: {json.dumps(result, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
