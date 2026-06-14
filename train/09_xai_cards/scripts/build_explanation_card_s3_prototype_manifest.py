#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explanation Card S3 : prototype candidate manifest 빌더 (안 3)

기준:
  - reports/explanation_cards/s2_component_continuity_filter_analysis_v1.md (안 3 추천)
  - reports/explanation_cards/s2_s3_prototype_candidate_review_samples_v1.csv (대상 8명 권위 safe_id)
  - S2 candidate 산출물(component_candidates.csv) read-only

설계 결정 (안 3):
  - 대상 = normal 3(control) + stage1_dev lesion 5 = 8명. ★권위 키는 sample CSV의 safe_id
    (subset9/subset8 은 patient_id prefix 중복이므로 safe_id로만 식별).
  - 환자별 최대 top3 component:
      1) patch_count>=2 OR z_span>=1 충족분을 rank_in_patient 순(+상위창 내 position_bin 다양성)으로 우선
      2) 3개 미만이면 max_padim_score 순으로 보충(selection_rule=score_fill)
  - normal=normal_control / structural FP review, stage1_dev=lesion_candidate.
  - holdout 절대 금지. CT/mask/PNG/카드/score/threshold 미사용(좌표·점수 메타만 복사).

가드:
  - 플래그 없으면 BLOCKED. --selftest/--dry-run 은 read-only.
  - --run-build 은 --confirm-generate 동반 필요. DONE.json/잔여 산출물 있으면 BLOCKED. --overwrite 없음.
"""

import argparse
import csv
import inspect
import json
import os
import sys
from datetime import datetime

csv.field_size_limit(10 ** 9)

# ----------------------------------------------------------------------------
# 상수
# ----------------------------------------------------------------------------
THRESHOLD_P95 = 14.0921
THRESHOLD_TYPE = "p95"
TOPK = 3
MAX_CARDS = 24
DIVERSITY_WINDOW = 5            # 다양성 탐색 창(rank 너무 낮은 후보 억지 방지)

POSITION_BINS = ("upper_central", "upper_peripheral", "middle_central",
                 "middle_peripheral", "lower_central", "lower_peripheral")
FORBIDDEN_PATH_TOKENS = ("stage2_holdout", "holdout")

ROLE_NORMAL = "normal_control"
ROLE_LESION = "lesion_candidate"

MANIFEST_FIELDS = (
    "prototype_case_id", "prototype_patient_id", "prototype_role",
    "group", "patient_id", "safe_id", "label", "component_id", "rank_in_patient",
    "position_bin", "slice_index_min", "slice_index_max", "z_span",
    "y0", "x0", "y1", "x1", "patch_count", "max_padim_score", "mean_padim_score",
    "max_score_slice_index", "roi_0_0_patch_ratio_mean",
    "central_peripheral", "central_distance_ratio_mean", "left_right_metadata",
    "threshold", "threshold_type", "selection_rule", "selection_reason",
    "stage_split_safety_flag", "source_component_csv", "source_candidate_root",
)

PATIENT_FIELDS = (
    "prototype_patient_id", "prototype_role", "group", "patient_id", "safe_id", "label",
    "n_selected_components", "selected_component_ids", "top_selected_score",
    "position_bins_selected", "selection_reason", "stage_split_safety_flag",
)

ERROR_FIELDS = ("scope", "key", "stage", "detail")

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_BLOCKED = 2

# ----------------------------------------------------------------------------
# 경로
# ----------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATE_ROOT = os.path.join(
    REPO, "outputs/position-aware-padim-v1/candidates/padim_v2_roi0_0_explanation_candidates_v1")
COMPONENT_CSV = os.path.join(CANDIDATE_ROOT, "component_candidates.csv")
SAMPLE_CSV = os.path.join(
    REPO, "outputs/position-aware-padim-v1/reports/explanation_cards/s2_s3_prototype_candidate_review_samples_v1.csv")
SPLIT_CSV = os.path.join(
    REPO, "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv")
REFERENCE_BANK_FULL = os.path.join(
    REPO, "outputs/position-aware-padim-v1/visualizations/candidate_cards/reference_bank_v1/full")
OUT_DIR = os.path.join(
    REPO, "outputs/position-aware-padim-v1/candidates/s3_prototype_manifest_v1")

PLANNED_ARTIFACTS = (
    "s3_prototype_candidate_manifest_v1.csv",
    "s3_prototype_patient_summary_v1.csv",
    "runtime_summary.json",
    "errors.csv",
    "DONE.json",
)


# ----------------------------------------------------------------------------
# 가드
# ----------------------------------------------------------------------------
def safe_path(path):
    low = str(path).replace("\\", "/").lower()
    for tok in FORBIDDEN_PATH_TOKENS:
        if tok in low:
            raise RuntimeError("FORBIDDEN path token '%s' in: %s" % (tok, path))
    return path


# ----------------------------------------------------------------------------
# read-only 로더
# ----------------------------------------------------------------------------
def load_holdout_denylist():
    hp, hs = set(), set()
    with open(safe_path(SPLIT_CSV), "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f); r.fieldnames = [c.strip() for c in r.fieldnames]
        for row in r:
            if (row.get("stage_split") or "").strip() == "stage2_holdout":
                if row.get("patient_id"): hp.add(row["patient_id"].strip())
                if row.get("safe_id"): hs.add(row["safe_id"].strip())
    return hp, hs


def load_targets():
    """sample CSV에서 대상 8명(권위 safe_id, scope, reason) 로드."""
    targets = []
    with open(safe_path(SAMPLE_CSV), "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            scope = (row["scope"] or "").strip()
            targets.append({
                "safe_id": (row["safe_id"] or "").strip(),
                "patient_id": (row["patient_id"] or "").strip(),
                "label": (row["label"] or "").strip(),
                "scope": scope,
                "role": ROLE_NORMAL if scope == "normal" else ROLE_LESION,
                "sample_reason": (row["reason"] or "").strip(),
            })
    return targets


def load_components_for(safe_ids):
    """component_candidates.csv 1회 순회로 대상 safe_id 들의 component만 수집."""
    want = set(safe_ids)
    out = {sid: [] for sid in want}
    with open(safe_path(COMPONENT_CSV), "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = row["safe_id"].strip()
            if sid in want:
                out[sid].append({
                    "component_id": row["component_id"], "group": row["group"],
                    "patient_id": row["patient_id"].strip(), "safe_id": sid,
                    "label": row["label"], "rank": int(row["rank_in_patient"]),
                    "position_bin": row["position_bin"],
                    "slice_index_min": int(row["slice_index_min"]),
                    "slice_index_max": int(row["slice_index_max"]),
                    "z_span": int(row["z_span"]),
                    "y0": int(row["y0"]), "x0": int(row["x0"]), "y1": int(row["y1"]), "x1": int(row["x1"]),
                    "patch_count": int(row["patch_count"]),
                    "max_padim_score": float(row["max_padim_score"]),
                    "mean_padim_score": float(row["mean_padim_score"]),
                    "max_score_slice_index": int(row["max_score_slice_index"]),
                    "roi_0_0_patch_ratio_mean": row["roi_0_0_patch_ratio_mean"],
                    "central_peripheral": row["central_peripheral"],
                    "central_distance_ratio_mean": row["central_distance_ratio_mean"],
                    "left_right_metadata": row["left_right_metadata"],
                    "extraction_scope": row["extraction_scope"],
                    "stage_split_safety_flag": row["stage_split_safety_flag"],
                })
    return out


# ----------------------------------------------------------------------------
# 선택 로직 (순수, selftest 대상)
# ----------------------------------------------------------------------------
def select_components(comps, topk=TOPK):
    """안 3 선택: 연속성(pc>=2 OR z_span>=1) 우선 top3(+상위창 bin 다양성), 부족시 score 보충.
    반환: [(component_dict, selection_rule), ...] 최대 topk."""
    by_rank = sorted(comps, key=lambda c: c["rank"])
    eligible = [c for c in by_rank if c["patch_count"] >= 2 or c["z_span"] >= 1]
    chosen, chosen_ids, used_bins = [], set(), set()

    # 1) 상위창 내 position_bin 다양성 우선
    for c in eligible[:DIVERSITY_WINDOW]:
        if len(chosen) >= topk:
            break
        if c["position_bin"] not in used_bins:
            chosen.append((c, "continuity_top3_bindiv"))
            chosen_ids.add(c["component_id"]); used_bins.add(c["position_bin"])
    # 2) 남은 슬롯을 연속성 eligible rank 순으로 채움(bin 중복 허용)
    for c in eligible:
        if len(chosen) >= topk:
            break
        if c["component_id"] not in chosen_ids:
            chosen.append((c, "continuity_top3")); chosen_ids.add(c["component_id"])
    # 3) 그래도 부족하면 비-eligible 을 score 순 보충
    if len(chosen) < topk:
        rest = [c for c in by_rank if c["component_id"] not in chosen_ids]
        for c in sorted(rest, key=lambda c: -c["max_padim_score"]):
            if len(chosen) >= topk:
                break
            chosen.append((c, "score_fill")); chosen_ids.add(c["component_id"])
    # 최종 rank 순 정렬(가독성)
    chosen.sort(key=lambda t: t[0]["rank"])
    return chosen[:topk]


def assert_no_holdout(pids, sids, hp, hs):
    ip, is_ = set(pids) & set(hp), set(sids) & set(hs)
    if ip or is_:
        raise RuntimeError("HOLDOUT LEAK -> BLOCKED pid=%s sid=%s" % (sorted(ip)[:5], sorted(is_)[:5]))
    return True


# ----------------------------------------------------------------------------
# 빌드 (--run-build --confirm-generate)
# ----------------------------------------------------------------------------
def _generate_manifest(out_dir):
    manifest_path = os.path.join(out_dir, "s3_prototype_candidate_manifest_v1.csv")
    patient_path = os.path.join(out_dir, "s3_prototype_patient_summary_v1.csv")
    runtime_path = os.path.join(out_dir, "runtime_summary.json")
    errors_path = os.path.join(out_dir, "errors.csv")
    done_path = os.path.join(out_dir, "DONE.json")

    if os.path.exists(safe_path(done_path)):
        sys.stderr.write("[BLOCKED] DONE.json 존재: %s\n" % done_path); return EXIT_BLOCKED
    if os.path.isdir(safe_path(out_dir)):
        leftovers = [p for p in (manifest_path, patient_path, runtime_path, errors_path) if os.path.exists(p)]
        if leftovers:
            sys.stderr.write("[BLOCKED] 잔여 산출물 존재: %s\n" % leftovers); return EXIT_BLOCKED

    hp, hs = load_holdout_denylist()
    targets = load_targets()
    comps_by_sid = load_components_for([t["safe_id"] for t in targets])

    errors = []
    manifest_rows, patient_rows = [], []
    started = datetime.now()
    n_requested = len(targets)
    sel_pids, sel_sids = set(), set()

    for t in targets:
        sid = t["safe_id"]
        comps = comps_by_sid.get(sid, [])
        if not comps:
            errors.append({"scope": t["scope"], "key": sid, "stage": "missing",
                           "detail": "no component for safe_id in component_candidates.csv"})
            continue
        chosen = select_components(comps)
        if len(manifest_rows) + len(chosen) > MAX_CARDS:
            chosen = chosen[:max(0, MAX_CARDS - len(manifest_rows))]
        sel_ids, bins, top_score, reasons = [], [], None, set()
        for i, (c, rule) in enumerate(chosen, 1):
            reason = ("정상 FP control / structural FP review 목적"
                      if t["role"] == ROLE_NORMAL else "stage1_dev lesion 후보 설명 대상")
            reason += " | %s | %s" % (rule, t["sample_reason"])
            reasons.add(rule)
            flag = "normal" if t["scope"] == "normal" else "stage1_dev"
            manifest_rows.append({
                "prototype_case_id": "%s__c%d" % (t["patient_id"], i),
                "prototype_patient_id": t["patient_id"], "prototype_role": t["role"],
                "group": c["group"], "patient_id": c["patient_id"], "safe_id": sid, "label": c["label"],
                "component_id": c["component_id"], "rank_in_patient": c["rank"],
                "position_bin": c["position_bin"], "slice_index_min": c["slice_index_min"],
                "slice_index_max": c["slice_index_max"], "z_span": c["z_span"],
                "y0": c["y0"], "x0": c["x0"], "y1": c["y1"], "x1": c["x1"],
                "patch_count": c["patch_count"], "max_padim_score": c["max_padim_score"],
                "mean_padim_score": c["mean_padim_score"], "max_score_slice_index": c["max_score_slice_index"],
                "roi_0_0_patch_ratio_mean": c["roi_0_0_patch_ratio_mean"],
                "central_peripheral": c["central_peripheral"],
                "central_distance_ratio_mean": c["central_distance_ratio_mean"],
                "left_right_metadata": c["left_right_metadata"],
                "threshold": THRESHOLD_P95, "threshold_type": THRESHOLD_TYPE,
                "selection_rule": rule, "selection_reason": reason,
                "stage_split_safety_flag": flag,
                "source_component_csv": os.path.relpath(COMPONENT_CSV, REPO),
                "source_candidate_root": os.path.relpath(CANDIDATE_ROOT, REPO),
            })
            sel_ids.append(c["component_id"]); bins.append(c["position_bin"])
            top_score = c["max_padim_score"] if top_score is None else max(top_score, c["max_padim_score"])
        if not sel_ids:
            continue
        sel_pids.add(t["patient_id"]); sel_sids.add(sid)
        patient_rows.append({
            "prototype_patient_id": t["patient_id"], "prototype_role": t["role"],
            "group": chosen[0][0]["group"], "patient_id": t["patient_id"], "safe_id": sid,
            "label": t["label"], "n_selected_components": len(sel_ids),
            "selected_component_ids": ";".join(sel_ids), "top_selected_score": top_score,
            "position_bins_selected": ";".join(sorted(set(bins))),
            "selection_reason": ("normal_control" if t["role"] == ROLE_NORMAL else "lesion_candidate")
                                + " | rules=" + ";".join(sorted(reasons)),
            "stage_split_safety_flag": "normal" if t["scope"] == "normal" else "stage1_dev",
        })

    # holdout 최종 assert
    assert_no_holdout(sel_pids, sel_sids, hp, hs)

    os.makedirs(safe_path(out_dir), exist_ok=True)
    with open(safe_path(manifest_path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(MANIFEST_FIELDS)); w.writeheader()
        for r in manifest_rows: w.writerow(r)
    with open(safe_path(patient_path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(PATIENT_FIELDS)); w.writeheader()
        for r in patient_rows: w.writerow(r)
    with open(safe_path(errors_path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ERROR_FIELDS)); w.writeheader()
        for e in errors: w.writerow(e)

    summary = {
        "mode": "s3_prototype_manifest",
        "source_candidate_root": os.path.relpath(CANDIDATE_ROOT, REPO),
        "source_component_csv": os.path.relpath(COMPONENT_CSV, REPO),
        "n_requested_patients": n_requested, "n_selected_patients": len(patient_rows),
        "n_selected_components": len(manifest_rows),
        "normal_selected_patients": sum(1 for p in patient_rows if p["prototype_role"] == ROLE_NORMAL),
        "stage1_dev_selected_patients": sum(1 for p in patient_rows if p["prototype_role"] == ROLE_LESION),
        "normal_selected_components": sum(1 for r in manifest_rows if r["prototype_role"] == ROLE_NORMAL),
        "stage1_dev_selected_components": sum(1 for r in manifest_rows if r["prototype_role"] == ROLE_LESION),
        "threshold": THRESHOLD_P95, "threshold_type": THRESHOLD_TYPE,
        "selection_rule": "ans3: (pc>=2 OR z_span>=1) top%d + bindiv window%d + score_fill" % (TOPK, DIVERSITY_WINDOW),
        "holdout_intersection": 0, "errors_count": len(errors), "done": True,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(safe_path(runtime_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(safe_path(done_path), "w", encoding="utf-8") as f:
        json.dump({"done": True, "summary": summary}, f, ensure_ascii=False, indent=2)

    print("[run-build] 완료. patients=%d components=%d errors=%d -> %s"
          % (len(patient_rows), len(manifest_rows), len(errors), out_dir))
    return EXIT_OK


# ----------------------------------------------------------------------------
# 모드
# ----------------------------------------------------------------------------
def mode_dry_run():
    print("[MODE] --dry-run (입력 read-only + 출력 계획)")
    ok = True
    def chk(n, c, e=""):
        nonlocal ok; ok = ok and bool(c); print("  [%s] %s %s" % ("OK" if c else "MISS", n, e))
    chk("component_candidates.csv", os.path.isfile(safe_path(COMPONENT_CSV)))
    chk("sample CSV", os.path.isfile(safe_path(SAMPLE_CSV)))
    chk("split CSV", os.path.isfile(safe_path(SPLIT_CSV)))
    chk("reference bank full(존재만)", os.path.isdir(safe_path(REFERENCE_BANK_FULL)))
    chk("출력 DONE 부재", not os.path.exists(os.path.join(OUT_DIR, "DONE.json")), "(존재시 BLOCKED)")
    t = load_targets()
    print("  [PLAN] 대상 %d명(normal %d + stage1_dev %d), 환자당 top%d, 최대 %d장"
          % (len(t), sum(1 for x in t if x["scope"] == "normal"),
             sum(1 for x in t if x["scope"] == "stage1_dev"), TOPK, MAX_CARDS))
    for a in PLANNED_ARTIFACTS: print("     -", a)
    return EXIT_OK if ok else EXIT_FAIL


def mode_selftest():
    print("[MODE] --selftest")
    results = []
    def expect(n, c):
        results.append(bool(c)); print("  [%s] %s" % ("PASS" if c else "FAIL", n))

    g_ok = True
    for p in ("a/stage2_holdout/x", "b/holdout.csv"):
        try: safe_path(p); g_ok = False
        except RuntimeError: pass
    expect("forbidden guard blocks holdout", g_ok)
    expect("threshold 14.0921", THRESHOLD_P95 == 14.0921)
    expect("TOPK==3 / MAX==24", TOPK == 3 and MAX_CARDS == 24)
    expect("MANIFEST schema 32열", len(MANIFEST_FIELDS) == 32)
    expect("PATIENT schema 12열", len(PATIENT_FIELDS) == 12)
    expect("role 상수", ROLE_NORMAL == "normal_control" and ROLE_LESION == "lesion_candidate")

    def mk(rank, bin_, pc, z, sc):
        return {"component_id": "c%d" % rank, "group": "g", "patient_id": "p", "safe_id": "s",
                "label": "x", "rank": rank, "position_bin": bin_, "slice_index_min": 0,
                "slice_index_max": z, "z_span": z, "y0": 0, "x0": 0, "y1": 32, "x1": 32,
                "patch_count": pc, "max_padim_score": sc, "mean_padim_score": sc - 1,
                "max_score_slice_index": 0, "roi_0_0_patch_ratio_mean": "1.0",
                "central_peripheral": "central", "central_distance_ratio_mean": "0.5",
                "left_right_metadata": "L", "extraction_scope": "normal", "stage_split_safety_flag": "normal"}
    # 연속성 충분 -> top3 continuity, bin 다양성 반영
    comps = [mk(1, "upper_central", 5, 3, 30), mk(2, "upper_central", 4, 2, 25),
             mk(3, "middle_central", 3, 1, 20), mk(4, "lower_central", 1, 0, 18)]
    sel = select_components(comps)
    expect("최대 3 선택", len(sel) == 3)
    expect("연속성 우선(pc=1/z=0 비선택)", all(c["component_id"] != "c4" for c, _ in sel))
    bins = [c["position_bin"] for c, _ in sel]
    expect("bin 다양성(>=2종)", len(set(bins)) >= 2)
    expect("rule continuity", all(r.startswith("continuity") for _, r in sel))
    # 연속성 부족 -> score_fill
    comps2 = [mk(1, "upper_central", 1, 0, 30), mk(2, "upper_central", 1, 0, 25), mk(3, "upper_central", 1, 0, 28)]
    sel2 = select_components(comps2)
    expect("연속성 0 -> score_fill 보충", len(sel2) == 3 and any(r == "score_fill" for _, r in sel2))
    expect("score_fill 최고점 우선", sel2 and max(c["max_padim_score"] for c, _ in sel2) == 30)

    blocked = False
    try: assert_no_holdout({"H"}, set(), {"H"}, set())
    except RuntimeError: blocked = True
    expect("holdout 교집합 차단", blocked)
    expect("holdout 무교집합 통과", assert_no_holdout({"P"}, {"S"}, {"H"}, {"HS"}) is True)

    src = inspect.getsource(_generate_manifest)
    src_run = inspect.getsource(mode_run_build)
    expect("run-build 실제 연결", "_generate_manifest(OUT_DIR)" in src_run)
    expect("run-build confirm 가드", "confirm_generate" in src_run and "EXIT_BLOCKED" in src_run)
    expect("DONE/잔여 가드", "DONE.json 존재" in src and "잔여 산출물" in src)
    expect("holdout assert 호출", "assert_no_holdout(sel_pids" in src)
    expect("CT/mask 미로드", "np.load" not in src and "ct_hu" not in src and "refined_roi.npy" not in src)
    expect("PNG 미생성", "imsave" not in src and ".png" not in src)
    expect("전체 산출물 기록", all(a in src for a in PLANNED_ARTIFACTS))

    n = sum(1 for x in results if x)
    print("\n[SELFTEST] %d/%d PASS" % (n, len(results)))
    return EXIT_OK if n == len(results) else EXIT_FAIL


def mode_run_build(confirm_generate):
    if not confirm_generate:
        sys.stderr.write("[BLOCKED] --run-build 은 --confirm-generate 동반 + 사용자 승인 필요.\n")
        return EXIT_BLOCKED
    return _generate_manifest(OUT_DIR)


def build_parser():
    p = argparse.ArgumentParser(description="Explanation Card S3 prototype manifest 빌더 (가드 필수).")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-build", action="store_true")
    p.add_argument("--confirm-generate", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest: return mode_selftest()
    if args.dry_run: return mode_dry_run()
    if args.run_build: return mode_run_build(args.confirm_generate)
    sys.stderr.write("[BLOCKED] 가드 플래그 필요: --selftest | --dry-run | (--run-build --confirm-generate)\n")
    return EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main())
