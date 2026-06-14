#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explanation Card S2 : padim_v2_roi0_0 candidate 추출기

기준:
  - docs/explanation_card_plan_v1.md (S2)
  - reports/explanation_cards/s2_v2_candidate_extraction_preflight_v1.md (NEEDS_FIX 2건)
  - S1 reference bank full 완료(reference_bank_v1/full) — 본 단계는 존재 확인만, 수정 금지.

설계 결정:
  - 입력 score = padim_v2_roi0_0 (frame roi_0_0). threshold = 정상 val p95 = 14.0921 (고정, 재계산 금지).
  - 추출 대상 scope = 정상 72 + stage1_dev lesion 154 = 총 226. stage2_holdout 154 제외. 전체 lesion 308 금지.
  - NEEDS_FIX 1) stage2_holdout hard exclude:
      * lesion score 파일명 = patient_id. holdout 파일도 같은 폴더에 patient_id 이름으로 섞여 있어
        경로 토큰만으로 구분 불가. -> split allowlist(stage1_dev patient_id 154)로만 경로를 만든다.
      * lesion_v2_by_patient/ 폴더 전체 순회(listdir/glob) 금지. allowlist 기반으로만 open.
      * 처리 대상 patient_id/safe_id ∩ stage2_holdout = 0 을 plan/run 양쪽에서 assert. 1개라도 있으면 BLOCKED.
  - NEEDS_FIX 2) normal label 합성:
      * normal score CSV 에는 label 컬럼 없음 -> output 에 label="normal" 합성.
      * lesion score CSV 의 label 은 그대로 보존(없으면 "lesion" 합성 + 보고).

가드:
  - 플래그 없으면 BLOCKED. --selftest/--dry-run/--plan-scope-only 는 read-only.
  - --run-extract 는 --confirm-generate 동반 + 사용자 승인 필요. DONE.json/잔여 산출물 있으면 BLOCKED.
  - --overwrite 없음. model forward/score 재계산/threshold 재계산/CT·mask npy 로드 없음.
  - 본 단계(스크립트 작성+정적검사)에서는 --run-extract --confirm-generate 미실행.
"""

import argparse
import csv
import inspect
import json
import math
import os
import sys
from datetime import datetime

# ----------------------------------------------------------------------------
# 확정 상수
# ----------------------------------------------------------------------------
MODEL_TAG = "padim_v2_roi0_0"
FRAME = "roi_0_0"
THRESHOLD_P95 = 14.0921           # 고정 사용값 (재계산 금지)
THRESHOLD_TYPE = "p95"
THRESHOLD_JSON_TOL = 1e-3         # JSON p95 와 14.0921 일치 허용오차(확인용, 재계산 아님)

SPLIT_STAGE1 = "stage1_dev"
SPLIT_HOLDOUT = "stage2_holdout"

POSITION_BINS = (
    "upper_central", "upper_peripheral",
    "middle_central", "middle_peripheral",
    "lower_central", "lower_peripheral",
)

# 경로 토큰 가드: holdout 계열 차단. (lesion 폴더는 stage1_dev 정당 사용이므로 lesion 토큰은 차단하지 않음)
FORBIDDEN_PATH_TOKENS = ("stage2_holdout", "holdout")

PLANNED_ARTIFACTS = (
    "patch_candidates.csv",
    "component_candidates.csv",
    "patient_candidate_summary.csv",
    "runtime_summary.json",
    "errors.csv",
    "DONE.json",
)

# 출력 스키마 (작업 지시 고정)
PATCH_FIELDS = (
    "candidate_patch_id", "group", "patient_id", "safe_id", "label", "source_score_csv",
    "slice_index", "local_z", "y0", "x0", "y1", "x1", "patch_size",
    "roi_0_0_patch_ratio", "position_bin", "z_level", "z_ratio",
    "central_peripheral", "central_distance_ratio_mean", "left_right_metadata",
    "padim_score", "threshold", "threshold_type", "extraction_scope", "stage_split_safety_flag",
)

COMPONENT_FIELDS = (
    "component_id", "group", "patient_id", "safe_id", "label", "rank_in_patient",
    "position_bin", "slice_index_min", "slice_index_max", "z_span",
    "y0", "x0", "y1", "x1", "patch_count",
    "max_padim_score", "mean_padim_score", "max_score_slice_index",
    "threshold", "threshold_type", "roi_0_0_patch_ratio_mean",
    "central_peripheral", "central_distance_ratio_mean", "left_right_metadata",
    "extraction_scope", "stage_split_safety_flag",
)

PATIENT_FIELDS = (
    "group", "patient_id", "safe_id", "label",
    "n_patch_candidates", "n_component_candidates", "max_padim_score",
    "top_component_id", "top_component_position_bin", "top_component_z_span",
    "extraction_scope", "stage_split_safety_flag",
)

ERROR_FIELDS = ("scope", "key", "stage", "detail")

# 추출에 필요한 필수 score 컬럼 (label 은 normal 에 없을 수 있어 별도 처리)
REQUIRED_SCORE_COLS = (
    "group", "patient_id", "safe_id", "slice_index", "local_z",
    "y0", "x0", "y1", "x1", "patch_size", "roi_0_0_patch_ratio",
    "position_bin", "z_level", "z_ratio", "central_peripheral",
    "central_distance_ratio_mean", "left_right_metadata", "padim_score",
)

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_BLOCKED = 2

# ----------------------------------------------------------------------------
# 경로
# ----------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NORMAL_SCORE_DIR = os.path.join(
    REPO, "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/normal_by_patient")
LESION_SCORE_DIR = os.path.join(
    REPO, "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/lesion_v2_by_patient")
SPLIT_CSV = os.path.join(
    REPO, "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv")
THRESHOLD_JSON = os.path.join(
    REPO, "outputs/position-aware-padim-v1/evaluation/normal_v2_roi0_0/normal_v2_threshold.json")
REFERENCE_BANK_FULL = os.path.join(
    REPO, "outputs/position-aware-padim-v1/visualizations/candidate_cards/reference_bank_v1/full")

OUT_DIR = os.path.join(
    REPO, "outputs/position-aware-padim-v1/candidates/padim_v2_roi0_0_explanation_candidates_v1")


# ----------------------------------------------------------------------------
# 가드
# ----------------------------------------------------------------------------
def safe_path(path):
    low = str(path).replace("\\", "/").lower()
    for tok in FORBIDDEN_PATH_TOKENS:
        if tok in low:
            raise RuntimeError("FORBIDDEN path token '%s' in: %s" % (tok, path))
    return path


def is_file(path):
    return os.path.isfile(safe_path(path))


def is_dir(path):
    return os.path.isdir(safe_path(path))


# ----------------------------------------------------------------------------
# split allowlist / holdout denylist
# ----------------------------------------------------------------------------
def parse_split_records(records):
    """순수 함수(selftest 대상): split dict 리스트 -> (stage1 pid set, holdout pid set, holdout sid set)."""
    stage1_pids, holdout_pids, holdout_sids = set(), set(), set()
    for r in records:
        pid = (r.get("patient_id") or "").strip()
        sid = (r.get("safe_id") or "").strip()
        st = (r.get("stage_split") or "").strip()
        if st == SPLIT_STAGE1:
            if pid:
                stage1_pids.add(pid)
        elif st == SPLIT_HOLDOUT:
            if pid:
                holdout_pids.add(pid)
            if sid:
                holdout_sids.add(sid)
    return stage1_pids, holdout_pids, holdout_sids


def load_split_allowlist():
    with open(safe_path(SPLIT_CSV), "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        records = list(reader)
    return parse_split_records(records)


def assert_no_holdout(processed_pids, processed_sids, holdout_pids, holdout_sids):
    """처리 대상 ∩ holdout = 0 검증. 1개라도 있으면 RuntimeError(BLOCKED)."""
    inter_p = set(processed_pids) & set(holdout_pids)
    inter_s = set(processed_sids) & set(holdout_sids)
    if inter_p or inter_s:
        raise RuntimeError(
            "HOLDOUT LEAK detected -> BLOCKED. pid_inter=%s sid_inter=%s"
            % (sorted(inter_p)[:5], sorted(inter_s)[:5]))
    return True


# ----------------------------------------------------------------------------
# 대상 목록 구성 (read-only)
# ----------------------------------------------------------------------------
def list_normal_score_csvs():
    if not is_dir(NORMAL_SCORE_DIR):
        return []
    return [os.path.join(NORMAL_SCORE_DIR, n)
            for n in sorted(os.listdir(safe_path(NORMAL_SCORE_DIR))) if n.lower().endswith(".csv")]


def build_normal_targets():
    return [{"scope": "normal", "patient_id": None, "score_csv": p}
            for p in list_normal_score_csvs()]


def build_lesion_targets(stage1_pids, holdout_pids):
    """stage1_dev allowlist(patient_id)로만 lesion 경로 생성. 폴더 전체 순회 금지."""
    targets, missing = [], []
    for pid in sorted(stage1_pids):
        if pid in holdout_pids:                      # 방어: stage1 과 holdout 교차 pid 즉시 차단
            raise RuntimeError("stage1_dev ∩ holdout patient_id: %s -> BLOCKED" % pid)
        path = os.path.join(LESION_SCORE_DIR, pid + ".csv")
        safe_path(path)
        if os.path.isfile(path):
            targets.append({"scope": "stage1_dev", "patient_id": pid, "score_csv": path})
        else:
            missing.append(pid)
    return targets, missing


def read_score_header(csv_path):
    with open(safe_path(csv_path), "r", encoding="utf-8-sig", newline="") as f:
        return [h.strip() for h in next(csv.reader(f), [])]


def iter_score_rows(csv_path):
    with open(safe_path(csv_path), "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        for row in reader:
            yield row


# ----------------------------------------------------------------------------
# 순수 계산 (selftest 대상)
# ----------------------------------------------------------------------------
def _to_float(v):
    try:
        return float(str(v).strip())
    except Exception:
        return None


def _to_int(v):
    f = _to_float(v)
    if f is None or not math.isfinite(f):
        return None
    return int(round(f))


def parse_valid_patch(row, scope):
    """추출 기준 적용 후 유효 patch dict 반환. 부적격이면 (None, reason)."""
    grp = (row.get("group") or "").strip()
    sid = (row.get("safe_id") or "").strip()
    pid = (row.get("patient_id") or "").strip()
    if not (grp and sid and pid):
        return None, "missing_group_safe_patient"

    score = _to_float(row.get("padim_score"))
    if score is None or not math.isfinite(score):
        return None, "score_nan_inf"
    if score < THRESHOLD_P95:
        return None, "below_threshold"

    roi = _to_float(row.get("roi_0_0_patch_ratio"))
    if roi is None or not (roi > 0):
        return None, "roi_not_positive"

    pbin = (row.get("position_bin") or "").strip()
    if pbin not in POSITION_BINS:
        return None, "bad_position_bin"

    y0 = _to_int(row.get("y0")); x0 = _to_int(row.get("x0"))
    y1 = _to_int(row.get("y1")); x1 = _to_int(row.get("x1"))
    if None in (y0, x0, y1, x1) or not (0 <= y0 < y1 and 0 <= x0 < x1):
        return None, "bad_bbox"

    sl = _to_int(row.get("slice_index"))
    if sl is None:
        return None, "bad_slice_index"

    psize = _to_int(row.get("patch_size"))
    if psize is None or psize <= 0:
        # bbox 로 보정 (정사각 가정)
        psize = max(y1 - y0, x1 - x0)

    # label 처리: normal 합성, lesion 보존(없으면 "lesion")
    if scope == "normal":
        label = "normal"
    else:
        label = (row.get("label") or "").strip() or "lesion"

    rec = {
        "group": grp, "patient_id": pid, "safe_id": sid, "label": label,
        "slice_index": sl, "local_z": (row.get("local_z") or "").strip(),
        "y0": y0, "x0": x0, "y1": y1, "x1": x1, "patch_size": psize,
        "roi_0_0_patch_ratio": roi, "position_bin": pbin,
        "z_level": (row.get("z_level") or "").strip(),
        "z_ratio": _to_float(row.get("z_ratio")),
        "central_peripheral": (row.get("central_peripheral") or "").strip(),
        "central_distance_ratio_mean": _to_float(row.get("central_distance_ratio_mean")),
        "left_right_metadata": (row.get("left_right_metadata") or "").strip(),
        "padim_score": score,
    }
    return rec, None


def _bbox_iou(a, b):
    iy0, ix0 = max(a["y0"], b["y0"]), max(a["x0"], b["x0"])
    iy1, ix1 = min(a["y1"], b["y1"]), min(a["x1"], b["x1"])
    ih, iw = iy1 - iy0, ix1 - ix0
    if ih <= 0 or iw <= 0:
        return 0.0
    inter = ih * iw
    area_a = (a["y1"] - a["y0"]) * (a["x1"] - a["x0"])
    area_b = (b["y1"] - b["y0"]) * (b["x1"] - b["x0"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_distance(a, b):
    acy, acx = (a["y0"] + a["y1"]) / 2.0, (a["x0"] + a["x1"]) / 2.0
    bcy, bcx = (b["y0"] + b["y1"]) / 2.0, (b["x0"] + b["x1"]) / 2.0
    return math.hypot(acy - bcy, acx - bcx)


def patches_adjacent(a, b):
    """같은 component 연결 조건: |Δslice|<=1 AND (bbox IoU>0 OR center_dist<=patch_size).

    Δslice=0 -> 2D 동일 slice, Δslice=1 -> 2.5D 인접 slice 를 모두 포함."""
    if abs(a["slice_index"] - b["slice_index"]) > 1:
        return False
    if _bbox_iou(a, b) > 0.0:
        return True
    psize = max(a.get("patch_size") or 0, b.get("patch_size") or 0)
    return _center_distance(a, b) <= psize


def _uf_find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def cluster_patches(patches):
    """같은 safe_id + 같은 position_bin patch 묶음을 받아 component 리스트(list of list)로 반환.
    slice 정렬 후 |Δslice|<=1 창에서만 비교(시간 bound)."""
    ps = sorted(patches, key=lambda p: (p["slice_index"], p["y0"], p["x0"]))
    n = len(ps)
    parent = list(range(n))
    for i in range(n):
        si = ps[i]["slice_index"]
        for j in range(i + 1, n):
            if ps[j]["slice_index"] - si > 1:
                break
            if patches_adjacent(ps[i], ps[j]):
                ri, rj = _uf_find(parent, i), _uf_find(parent, j)
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for idx in range(n):
        groups.setdefault(_uf_find(parent, idx), []).append(ps[idx])
    return list(groups.values())


def _mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return (sum(vals) / len(vals)) if vals else ""


def component_metrics(patches):
    """component(같은 safe_id+bin patch 묶음) -> 지표 dict (component_id/rank 제외)."""
    top = max(patches, key=lambda p: (p["padim_score"], -p["slice_index"]))
    scores = [p["padim_score"] for p in patches]
    smin = min(p["slice_index"] for p in patches)
    smax = max(p["slice_index"] for p in patches)
    return {
        "group": top["group"], "patient_id": top["patient_id"], "safe_id": top["safe_id"],
        "label": top["label"], "position_bin": top["position_bin"],
        "slice_index_min": smin, "slice_index_max": smax, "z_span": smax - smin,
        "y0": min(p["y0"] for p in patches), "x0": min(p["x0"] for p in patches),
        "y1": max(p["y1"] for p in patches), "x1": max(p["x1"] for p in patches),
        "patch_count": len(patches),
        "max_padim_score": max(scores), "mean_padim_score": sum(scores) / len(scores),
        "max_score_slice_index": top["slice_index"],
        "roi_0_0_patch_ratio_mean": _mean([p["roi_0_0_patch_ratio"] for p in patches]),
        "central_peripheral": top["central_peripheral"],
        "central_distance_ratio_mean": _mean([p["central_distance_ratio_mean"] for p in patches]),
        "left_right_metadata": top["left_right_metadata"],
    }


def rank_components(components):
    """patient 내 component 정렬: max_padim_score desc -> patch_count desc
    (그 뒤 z_span desc, slice_index_min asc 로 완전 결정화). rank_in_patient 부여."""
    ordered = sorted(
        components,
        key=lambda c: (-c["max_padim_score"], -c["patch_count"], -c["z_span"], c["slice_index_min"]))
    for i, c in enumerate(ordered):
        c["rank_in_patient"] = i + 1
    return ordered


# ----------------------------------------------------------------------------
# 한 patient(score CSV) 처리 (selftest 대상; npy 미로드)
# ----------------------------------------------------------------------------
def process_patient_rows(rows, scope):
    """score row 리스트 -> (patch_records, component_dicts). bbox/필터/클러스터/랭킹 적용."""
    patches, n_filtered = [], 0
    for row in rows:
        rec, _reason = parse_valid_patch(row, scope)
        if rec is None:
            n_filtered += 1
            continue
        patches.append(rec)

    # 같은 safe_id + position_bin 단위 클러스터
    by_key = {}
    for p in patches:
        by_key.setdefault((p["safe_id"], p["position_bin"]), []).append(p)
    components_raw = []
    for _key, group_patches in by_key.items():
        for comp in cluster_patches(group_patches):
            components_raw.append(component_metrics(comp))

    # patient(safe_id) 별 랭킹
    by_sid = {}
    for c in components_raw:
        by_sid.setdefault(c["safe_id"], []).append(c)
    components = []
    for sid, comps in by_sid.items():
        ranked = rank_components(comps)
        for c in ranked:
            c["component_id"] = "%s__cmp%03d" % (sid, c["rank_in_patient"])
        components.extend(ranked)

    return patches, components, n_filtered


# ----------------------------------------------------------------------------
# scope 감사 (plan/run 공통)
# ----------------------------------------------------------------------------
def build_scope_audit():
    audit = {"blocked": False, "block_reason": None}
    stage1_pids, holdout_pids, holdout_sids = load_split_allowlist()
    audit["stage1_dev_count"] = len(stage1_pids)
    audit["holdout_count"] = len(holdout_pids)

    normal_targets = build_normal_targets()
    lesion_targets, missing = build_lesion_targets(stage1_pids, holdout_pids)
    audit["normal_target_count"] = len(normal_targets)
    audit["lesion_target_count"] = len(lesion_targets)
    audit["lesion_missing_files"] = missing
    audit["total_target_count"] = len(normal_targets) + len(lesion_targets)

    # 처리 대상 patient_id 집합(lesion) ∩ holdout = 0 검증 (normal 은 lesion split 밖)
    lesion_pids = set(t["patient_id"] for t in lesion_targets)
    try:
        assert_no_holdout(lesion_pids, set(), holdout_pids, holdout_sids)
        audit["holdout_intersection"] = 0
    except RuntimeError as e:
        audit["blocked"] = True
        audit["block_reason"] = str(e)
        audit["holdout_intersection"] = -1

    audit["_stage1_pids"] = stage1_pids
    audit["_holdout_pids"] = holdout_pids
    audit["_holdout_sids"] = holdout_sids
    audit["_normal_targets"] = normal_targets
    audit["_lesion_targets"] = lesion_targets
    return audit


# ----------------------------------------------------------------------------
# 실제 추출 (--run-extract --confirm-generate; 본 단계 미실행)
# ----------------------------------------------------------------------------
def _generate_candidates(out_dir):
    patch_path = os.path.join(out_dir, "patch_candidates.csv")
    comp_path = os.path.join(out_dir, "component_candidates.csv")
    patient_path = os.path.join(out_dir, "patient_candidate_summary.csv")
    runtime_path = os.path.join(out_dir, "runtime_summary.json")
    errors_path = os.path.join(out_dir, "errors.csv")
    done_path = os.path.join(out_dir, "DONE.json")

    # 가드: DONE 또는 잔여 산출물 -> BLOCKED (덮어쓰기/삭제 금지)
    if os.path.exists(safe_path(done_path)):
        sys.stderr.write("[BLOCKED] DONE.json 존재: %s\n  새 버전 경로 사용.\n" % done_path)
        return EXIT_BLOCKED
    if os.path.isdir(safe_path(out_dir)):
        leftovers = [p for p in (patch_path, comp_path, patient_path, runtime_path, errors_path)
                     if os.path.exists(p)]
        if leftovers:
            sys.stderr.write("[BLOCKED] 미완료 이전 산출물 존재: %s\n  삭제/덮어쓰기 금지 -> 새 버전 경로.\n"
                             % leftovers)
            return EXIT_BLOCKED

    # threshold JSON 확인만 (재계산 아님)
    with open(safe_path(THRESHOLD_JSON), "r", encoding="utf-8") as f:
        thj = json.load(f)
    if abs(float(thj.get("threshold_p95")) - THRESHOLD_P95) > THRESHOLD_JSON_TOL:
        sys.stderr.write("[BLOCKED] threshold 불일치: json=%s const=%s\n"
                         % (thj.get("threshold_p95"), THRESHOLD_P95))
        return EXIT_BLOCKED

    audit = build_scope_audit()
    if audit.get("blocked"):
        sys.stderr.write("[BLOCKED] %s\n" % audit.get("block_reason"))
        return EXIT_BLOCKED
    holdout_pids = audit["_holdout_pids"]
    holdout_sids = audit["_holdout_sids"]
    targets = list(audit["_normal_targets"]) + list(audit["_lesion_targets"])
    if not targets:
        sys.stderr.write("[BLOCKED] 추출 대상 0건\n")
        return EXIT_BLOCKED

    os.makedirs(safe_path(out_dir), exist_ok=True)

    processed_pids, processed_sids = set(), set()
    errors = []
    n_patch_total, n_comp_total = 0, 0
    patient_summaries = []
    t0 = datetime.now()

    pf = open(safe_path(patch_path), "w", encoding="utf-8", newline="")
    cf = open(safe_path(comp_path), "w", encoding="utf-8", newline="")
    pw = csv.DictWriter(pf, fieldnames=list(PATCH_FIELDS)); pw.writeheader()
    cw = csv.DictWriter(cf, fieldnames=list(COMPONENT_FIELDS)); cw.writeheader()
    try:
        for tgt in targets:
            scope = tgt["scope"]
            csv_path = tgt["score_csv"]
            try:
                rows = list(iter_score_rows(csv_path))
            except Exception as e:
                errors.append({"scope": scope, "key": os.path.basename(csv_path),
                               "stage": "load", "detail": str(e)[:300]})
                continue
            patches, components, _nf = process_patient_rows(rows, scope)

            # 이 CSV 의 safe_id/patient_id 수집 후 holdout 재검증
            sids = set(p["safe_id"] for p in patches) | set(c["safe_id"] for c in components)
            pids = set(p["patient_id"] for p in patches) | set(c["patient_id"] for c in components)
            try:
                assert_no_holdout(pids, sids, holdout_pids, holdout_sids)
            except RuntimeError as e:
                pf.close(); cf.close()
                sys.stderr.write("[BLOCKED] %s (file=%s)\n" % (e, os.path.basename(csv_path)))
                return EXIT_BLOCKED
            processed_sids |= sids
            processed_pids |= pids

            scope_flag = "normal" if scope == "normal" else "stage1_dev"

            # patch 기록
            patch_idx = 0
            for p in patches:
                patch_idx += 1
                pw.writerow({
                    "candidate_patch_id": "%s__p%05d" % (p["safe_id"], patch_idx),
                    "group": p["group"], "patient_id": p["patient_id"], "safe_id": p["safe_id"],
                    "label": p["label"], "source_score_csv": os.path.relpath(csv_path, REPO),
                    "slice_index": p["slice_index"], "local_z": p["local_z"],
                    "y0": p["y0"], "x0": p["x0"], "y1": p["y1"], "x1": p["x1"],
                    "patch_size": p["patch_size"], "roi_0_0_patch_ratio": p["roi_0_0_patch_ratio"],
                    "position_bin": p["position_bin"], "z_level": p["z_level"], "z_ratio": p["z_ratio"],
                    "central_peripheral": p["central_peripheral"],
                    "central_distance_ratio_mean": p["central_distance_ratio_mean"],
                    "left_right_metadata": p["left_right_metadata"], "padim_score": p["padim_score"],
                    "threshold": THRESHOLD_P95, "threshold_type": THRESHOLD_TYPE,
                    "extraction_scope": scope_flag, "stage_split_safety_flag": scope_flag,
                })
            n_patch_total += len(patches)

            # component 기록
            for c in components:
                cw.writerow({
                    "component_id": c["component_id"], "group": c["group"],
                    "patient_id": c["patient_id"], "safe_id": c["safe_id"], "label": c["label"],
                    "rank_in_patient": c["rank_in_patient"], "position_bin": c["position_bin"],
                    "slice_index_min": c["slice_index_min"], "slice_index_max": c["slice_index_max"],
                    "z_span": c["z_span"], "y0": c["y0"], "x0": c["x0"], "y1": c["y1"], "x1": c["x1"],
                    "patch_count": c["patch_count"], "max_padim_score": c["max_padim_score"],
                    "mean_padim_score": c["mean_padim_score"],
                    "max_score_slice_index": c["max_score_slice_index"],
                    "threshold": THRESHOLD_P95, "threshold_type": THRESHOLD_TYPE,
                    "roi_0_0_patch_ratio_mean": c["roi_0_0_patch_ratio_mean"],
                    "central_peripheral": c["central_peripheral"],
                    "central_distance_ratio_mean": c["central_distance_ratio_mean"],
                    "left_right_metadata": c["left_right_metadata"],
                    "extraction_scope": scope_flag, "stage_split_safety_flag": scope_flag,
                })
            n_comp_total += len(components)

            # patient summary (safe_id 단위)
            by_sid = {}
            for p in patches:
                by_sid.setdefault(p["safe_id"], {"patches": [], "comps": []})["patches"].append(p)
            for c in components:
                by_sid.setdefault(c["safe_id"], {"patches": [], "comps": []})["comps"].append(c)
            for sid, agg in by_sid.items():
                comps = agg["comps"]
                top = min(comps, key=lambda c: c["rank_in_patient"]) if comps else None
                base = (agg["patches"] or comps)[0]
                patient_summaries.append({
                    "group": base["group"], "patient_id": base["patient_id"], "safe_id": sid,
                    "label": base["label"], "n_patch_candidates": len(agg["patches"]),
                    "n_component_candidates": len(comps),
                    "max_padim_score": max((p["padim_score"] for p in agg["patches"]), default=""),
                    "top_component_id": top["component_id"] if top else "",
                    "top_component_position_bin": top["position_bin"] if top else "",
                    "top_component_z_span": top["z_span"] if top else "",
                    "extraction_scope": scope_flag, "stage_split_safety_flag": scope_flag,
                })
    finally:
        pf.close(); cf.close()

    # patient summary 기록
    with open(safe_path(patient_path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(PATIENT_FIELDS)); w.writeheader()
        for r in patient_summaries:
            w.writerow(r)

    # errors.csv
    with open(safe_path(errors_path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ERROR_FIELDS)); w.writeheader()
        for e in errors:
            w.writerow(e)

    # 최종 holdout 안전 재검증
    assert_no_holdout(processed_pids, processed_sids, holdout_pids, holdout_sids)

    summary = {
        "report": "S2 candidate extraction runtime summary", "model_tag": MODEL_TAG, "frame": FRAME,
        "threshold": THRESHOLD_P95, "threshold_type": THRESHOLD_TYPE,
        "scope": "normal72 + stage1_dev154 (226)",
        "normal_target_count": audit["normal_target_count"],
        "lesion_target_count": audit["lesion_target_count"],
        "lesion_missing_files": audit["lesion_missing_files"],
        "processed_patient_ids": len(processed_pids), "processed_safe_ids": len(processed_sids),
        "holdout_intersection": 0,
        "n_patch_candidates": n_patch_total, "n_component_candidates": n_comp_total,
        "n_patients_with_candidates": len(patient_summaries),
        "n_errors": len(errors),
        "started": t0.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    with open(safe_path(runtime_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(safe_path(done_path), "w", encoding="utf-8") as f:
        json.dump({"done": True, "summary": summary}, f, ensure_ascii=False, indent=2)

    print("[run-extract] 완료. patch=%d component=%d patients=%d errors=%d -> %s"
          % (n_patch_total, n_comp_total, len(patient_summaries), len(errors), out_dir))
    return EXIT_OK


# ----------------------------------------------------------------------------
# 모드: dry-run / plan-scope-only / selftest / run-extract
# ----------------------------------------------------------------------------
def mode_dry_run():
    print("[MODE] --dry-run (입력 read-only 점검 + 출력 계획)")
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  [%s] %s %s" % ("OK" if cond else "MISS", name, extra))

    chk("normal score dir", is_dir(NORMAL_SCORE_DIR))
    chk("lesion score dir", is_dir(LESION_SCORE_DIR))
    chk("split csv", is_file(SPLIT_CSV))
    chk("threshold json", is_file(THRESHOLD_JSON))
    chk("reference bank full(존재만)", is_dir(REFERENCE_BANK_FULL))
    chk("출력 DONE 부재", not os.path.exists(os.path.join(OUT_DIR, "DONE.json")), "(존재시 BLOCKED)")
    print("  [PLAN] 출력 경로:", os.path.relpath(OUT_DIR, REPO))
    for a in PLANNED_ARTIFACTS:
        print("     -", a)
    print("  [PLAN] threshold=%s (%s), 재계산 없음" % (THRESHOLD_P95, THRESHOLD_TYPE))
    return EXIT_OK if ok else EXIT_FAIL


def mode_plan_scope_only():
    print("[MODE] --plan-scope-only (대상 집합 계획만. score row/npy 미로드)")
    audit = build_scope_audit()
    print("  normal target      :", audit["normal_target_count"])
    print("  stage1_dev allowlist:", audit["stage1_dev_count"])
    print("  lesion target(존재) :", audit["lesion_target_count"],
          "(missing=%d)" % len(audit["lesion_missing_files"]))
    print("  total target       :", audit["total_target_count"])
    print("  holdout denylist   :", audit["holdout_count"])
    print("  holdout 교집합      :", audit["holdout_intersection"], "(0 이어야 함)")
    if audit.get("blocked"):
        print("  [BLOCKED]", audit.get("block_reason"))
        return EXIT_BLOCKED
    ok = (audit["normal_target_count"] == 72
          and audit["stage1_dev_count"] == 154
          and audit["holdout_count"] == 154
          and audit["holdout_intersection"] == 0)
    print("  [%s] normal72 + stage1_dev154 = 226, holdout 0 검증" % ("PASS" if ok else "CHECK"))
    return EXIT_OK if ok else EXIT_FAIL


def mode_selftest():
    print("[MODE] --selftest (순수 로직 + 소스 정적 검토)")
    results = []

    def expect(name, cond):
        results.append(bool(cond)); print("  [%s] %s" % ("PASS" if cond else "FAIL", name))

    # forbidden guard: holdout 계열 차단, 정상 lesion 경로 허용
    g_ok = True
    for p in ("a/stage2_holdout/x", "b/some_holdout.csv"):
        try:
            safe_path(p); g_ok = False
        except RuntimeError:
            pass
    expect("forbidden guard blocks holdout 계열", g_ok)
    try:
        safe_path("x/lesion_v2_by_patient/LUNG1-001.csv"); l_ok = True
    except RuntimeError:
        l_ok = False
    expect("forbidden guard allows stage1_dev lesion 경로", l_ok)

    # threshold
    expect("threshold 상수 14.0921", THRESHOLD_P95 == 14.0921)
    expect("threshold_type p95", THRESHOLD_TYPE == "p95")
    src_gen = inspect.getsource(_generate_candidates)
    src_proc = inspect.getsource(process_patient_rows)
    src_parse = inspect.getsource(parse_valid_patch)
    expect("threshold 재계산 함수 없음(percentile/quantile 미사용)",
           not any(k in (src_gen + src_proc + src_parse)
                   for k in ("percentile", "quantile", "np.percentile")))
    expect("position_bin == 6", len(POSITION_BINS) == 6)

    # split 파서 (순수)
    recs = [
        {"patient_id": "P1", "safe_id": "S1", "stage_split": "stage1_dev"},
        {"patient_id": "P2", "safe_id": "S2", "stage_split": "stage1_dev"},
        {"patient_id": "H1", "safe_id": "HS1", "stage_split": "stage2_holdout"},
    ]
    s1, hp, hs = parse_split_records(recs)
    expect("split allowlist stage1_dev 추출", s1 == {"P1", "P2"})
    expect("split holdout pid/sid 추출", hp == {"H1"} and hs == {"HS1"})

    # holdout 교집합 차단
    blocked = False
    try:
        assert_no_holdout({"H1"}, set(), {"H1"}, {"HS1"})
    except RuntimeError:
        blocked = True
    expect("assert_no_holdout 교집합 -> 차단", blocked)
    expect("assert_no_holdout 무교집합 -> 통과",
           assert_no_holdout({"P1"}, {"S1"}, {"H1"}, {"HS1"}) is True)

    # normal label 합성 / lesion label 보존
    nrow = {"group": "test", "patient_id": "n1", "safe_id": "ns1", "padim_score": "20.0",
            "roi_0_0_patch_ratio": "0.9", "position_bin": "upper_central",
            "y0": "0", "x0": "0", "y1": "32", "x1": "32", "patch_size": "32",
            "slice_index": "10", "central_distance_ratio_mean": "0.5"}
    nrec, _ = parse_valid_patch(nrow, "normal")
    expect("normal label 합성 'normal'", nrec is not None and nrec["label"] == "normal")
    lrow = dict(nrow); lrow["label"] = "lesion_test"
    lrec, _ = parse_valid_patch(lrow, "stage1_dev")
    expect("lesion label 보존", lrec is not None and lrec["label"] == "lesion_test")
    lrow2 = dict(nrow)  # lesion 인데 label 없음 -> 합성
    lrec2, _ = parse_valid_patch(lrow2, "stage1_dev")
    expect("lesion label 없으면 'lesion' 합성", lrec2 is not None and lrec2["label"] == "lesion")

    # 추출 필터: threshold/roi/bbox/score
    expect("below-threshold 제외", parse_valid_patch(dict(nrow, padim_score="10.0"), "normal")[0] is None)
    expect("roi<=0 제외", parse_valid_patch(dict(nrow, roi_0_0_patch_ratio="0"), "normal")[0] is None)
    expect("NaN score 제외", parse_valid_patch(dict(nrow, padim_score="nan"), "normal")[0] is None)
    expect("bad bbox 제외", parse_valid_patch(dict(nrow, y1="0"), "normal")[0] is None)
    expect("bad position_bin 제외", parse_valid_patch(dict(nrow, position_bin="zzz"), "normal")[0] is None)

    # clustering: 같은 bin 인접 slice overlap -> 1 component
    def mk(sl, y0, x0, sc=20.0):
        return {"group": "g", "patient_id": "pp", "safe_id": "ss", "label": "normal",
                "slice_index": sl, "local_z": "", "y0": y0, "x0": x0, "y1": y0 + 32, "x1": x0 + 32,
                "patch_size": 32, "roi_0_0_patch_ratio": 0.9, "position_bin": "upper_central",
                "z_level": "upper", "z_ratio": 0.1, "central_peripheral": "central",
                "central_distance_ratio_mean": 0.5, "left_right_metadata": "L", "padim_score": sc}
    near = [mk(10, 0, 0), mk(11, 4, 4)]
    expect("인접 slice + overlap -> 1 component", len(cluster_patches(near)) == 1)
    far = [mk(10, 0, 0), mk(10, 200, 200)]
    expect("멀리 떨어진 bbox -> 2 component", len(cluster_patches(far)) == 2)
    far_z = [mk(10, 0, 0), mk(20, 0, 0)]
    expect("slice 멀면 -> 2 component", len(cluster_patches(far_z)) == 2)

    # process_patient_rows: 다른 safe_id / 다른 bin -> 다른 component (grouping)
    rows_two_sid = [
        dict(nrow, safe_id="A", slice_index="10"),
        dict(nrow, safe_id="B", slice_index="10"),
    ]
    _pa, comps_sid, _ = process_patient_rows(rows_two_sid, "normal")
    expect("다른 safe_id -> 다른 component", len(comps_sid) == 2)
    rows_two_bin = [
        dict(nrow, safe_id="A", position_bin="upper_central"),
        dict(nrow, safe_id="A", position_bin="lower_central"),
    ]
    _pb, comps_bin, _ = process_patient_rows(rows_two_bin, "normal")
    expect("다른 position_bin -> 다른 component", len(comps_bin) == 2)

    # rank_in_patient 정렬
    comps = [
        {"max_padim_score": 18.0, "patch_count": 1, "z_span": 0, "slice_index_min": 5},
        {"max_padim_score": 25.0, "patch_count": 3, "z_span": 2, "slice_index_min": 1},
        {"max_padim_score": 25.0, "patch_count": 5, "z_span": 1, "slice_index_min": 2},
    ]
    ranked = rank_components([dict(c) for c in comps])
    order = [(c["max_padim_score"], c["patch_count"]) for c in sorted(ranked, key=lambda x: x["rank_in_patient"])]
    expect("rank: score desc -> patch_count desc", order == [(25.0, 5), (25.0, 3), (18.0, 1)])

    # 스키마 정합
    expect("PATCH schema 25열", len(PATCH_FIELDS) == 25 and PATCH_FIELDS[0] == "candidate_patch_id")
    expect("COMPONENT schema 26열", len(COMPONENT_FIELDS) == 26 and COMPONENT_FIELDS[0] == "component_id")
    expect("PATIENT schema 12열", len(PATIENT_FIELDS) == 12 and PATIENT_FIELDS[0] == "group")

    # 소스 정적: holdout open 금지 / lesion 전체순회 금지 / 실제연결 / DONE 가드
    src_lt = inspect.getsource(build_lesion_targets)
    expect("lesion 경로는 allowlist(pid)로만 생성", "pid + \".csv\"" in src_lt and "stage1_pids" in src_lt)
    expect("lesion 폴더 전체순회 안함(listdir/glob 미사용)",
           "listdir" not in src_lt and "glob" not in src_lt)
    src_run = inspect.getsource(mode_run_extract)
    expect("run-extract -> 실제 생성함수 연결", "_generate_candidates(OUT_DIR)" in src_run)
    expect("run-extract confirm 없으면 BLOCKED", "confirm_generate" in src_run and "EXIT_BLOCKED" in src_run)
    expect("run-extract 단순 print/EXIT_OK 아님", "return _generate_candidates(OUT_DIR)" in src_run)
    expect("placeholder 없음", not any(b in src_gen for b in ("placeholder", "TODO", "구현 자리")))
    expect("DONE 존재시 BLOCKED 로직", "DONE.json 존재" in src_gen and "EXIT_BLOCKED" in src_gen)
    expect("run 내 holdout 재검증 호출", "assert_no_holdout(" in src_gen)
    expect("CT/mask npy 미로드(np.load 없음)", "np.load" not in src_gen and "mmap_mode" not in src_gen)
    expect("model forward 없음(torch 미사용)", "torch" not in (src_gen + src_proc))
    expect("실제 기록(csv writeheader)", "writeheader" in src_gen)
    expect("전체 산출물 기록", all(a in src_gen for a in PLANNED_ARTIFACTS))

    n_pass = sum(1 for ok in results if ok)
    print("\n[SELFTEST] %d/%d PASS" % (n_pass, len(results)))
    return EXIT_OK if n_pass == len(results) else EXIT_FAIL


def mode_run_extract(confirm_generate):
    if not confirm_generate:
        sys.stderr.write("[BLOCKED] --run-extract 은 --confirm-generate 동반 + 사용자 승인 필요.\n")
        return EXIT_BLOCKED
    return _generate_candidates(OUT_DIR)


def build_parser():
    p = argparse.ArgumentParser(description="Explanation Card S2 v2_roi0_0 candidate 추출기 (가드 필수).")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plan-scope-only", action="store_true")
    p.add_argument("--run-extract", action="store_true")
    p.add_argument("--confirm-generate", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest:
        return mode_selftest()
    if args.dry_run:
        return mode_dry_run()
    if args.plan_scope_only:
        return mode_plan_scope_only()
    if args.run_extract:
        return mode_run_extract(args.confirm_generate)
    sys.stderr.write(
        "[BLOCKED] 가드 플래그가 필요합니다.\n"
        "  허용: --selftest | --dry-run | --plan-scope-only\n"
        "  (--run-extract 은 --confirm-generate + 승인 필요)\n")
    return EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main())
