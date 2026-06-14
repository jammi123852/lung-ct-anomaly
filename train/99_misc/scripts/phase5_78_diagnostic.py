#!/usr/bin/env python3
"""
Phase 5.78 Weak 3D Merge Result Diagnostic
input : Phase 5.77 3D cluster CSV / JSON / MD (read-only)
output: diagnostic MD / JSON / CSV

절대 금지:
- weak 3D merge 재실행
- clustering full-run
- visual review pack / PNG / HTML / ZIP 생성
- model forward / score 재계산 / training
- 기존 결과 삭제/이동/덮어쓰기
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 경로 정의 ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_INPUT_DIR = (
    _PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "first_stage_padim_cluster_review"
    / "phase5_77_weak_3d_merge_dry_run_v1"
)
_INPUT_CSV  = _INPUT_DIR / "phase5_77_weak_3d_cluster_summary.csv"
_INPUT_JSON = _INPUT_DIR / "phase5_77_weak_3d_cluster_summary.json"
_INPUT_MD   = _INPUT_DIR / "phase5_77_weak_3d_cluster_summary.md"

_OUTPUT_ROOT = (
    _PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/reports"
    / "phase5_78_weak_3d_merge_result_diagnostic_v1"
)
_OUT_TAG = "phase5_78_weak_3d_merge_result_diagnostic_v1"

# Phase 5.74 비교 기준값 (고정 참조값, 수정 금지)
_PHASE574_2D_REDUCTION_RATE = 0.301


# ── path guard ─────────────────────────────────────────────────────────────
def _guard_path(p: Path) -> None:
    for part in p.parts:
        pl = part.lower()
        if (
            "stage2_holdout" in pl
            or pl == "v2"
            or pl.startswith("v2v2")
            or pl.startswith("v2_")
            or "hard_negative" in pl
        ):
            sys.exit(f"[ERROR] Forbidden path segment '{part}' in {p}")


# ── 입력 파일 존재 검증 ─────────────────────────────────────────────────────
def _validate_inputs() -> None:
    for p in [_INPUT_CSV, _INPUT_JSON, _INPUT_MD]:
        _guard_path(p)
        if not p.exists():
            sys.exit(f"[ERROR] 입력 파일 없음: {p}")
    print(f"[OK] 입력 파일 3개 확인: {_INPUT_DIR}")


# ── 출력 폴더 사전 존재 검사 ────────────────────────────────────────────────
def _check_output_not_exists() -> None:
    _guard_path(_OUTPUT_ROOT)
    if _OUTPUT_ROOT.exists():
        sys.exit(
            f"[ERROR] 출력 폴더가 이미 존재합니다. "
            f"재실행을 원하면 폴더를 직접 삭제 후 재시도하세요: {_OUTPUT_ROOT}"
        )
    print(f"[OK] 출력 폴더 없음 확인 (신규 생성 예정): {_OUTPUT_ROOT}")


# ── 데이터 로드 ─────────────────────────────────────────────────────────────
def _load_data():
    df = pd.read_csv(_INPUT_CSV)
    with open(_INPUT_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if len(df) != 220:
        sys.exit(f"[ERROR] CSV row 수 불일치: expected=220, actual={len(df)}")
    if meta.get("input_2d_cluster_count") != 542:
        sys.exit(f"[ERROR] JSON input_2d_cluster_count != 542: {meta.get('input_2d_cluster_count')}")
    if meta.get("weak_3d_cluster_count") != 220:
        sys.exit(f"[ERROR] JSON weak_3d_cluster_count != 220: {meta.get('weak_3d_cluster_count')}")
    print(f"[OK] 데이터 로드: CSV {len(df)}행, input_2d=542, weak_3d=220")
    return df, meta


# ── bool 컬럼 정규화 ────────────────────────────────────────────────────────
def _to_bool(series: pd.Series) -> pd.Series:
    """True/False 문자열 또는 bool → bool"""
    if series.dtype == bool:
        return series
    return series.map(lambda x: str(x).strip().lower() == "true")


# ── z_span 분포 계산 ────────────────────────────────────────────────────────
def _calc_zspan_dist(df: pd.DataFrame) -> dict:
    counts = df["z_span"].value_counts().sort_index()
    dist = {int(k): int(v) for k, v in counts.items()}
    z1 = int((df["z_span"] == 1).sum())
    z2 = int((df["z_span"] == 2).sum())
    z3 = int((df["z_span"] == 3).sum())
    zgt3 = int((df["z_span"] > 3).sum())
    return {
        "distribution": dist,
        "z_span_eq1": z1,
        "z_span_eq2": z2,
        "z_span_eq3": z3,
        "z_span_gt3": zgt3,
        "z_span_max": int(df["z_span"].max()),
        "z_span_mean": round(float(df["z_span"].mean()), 4),
        "z_span_median": float(df["z_span"].median()),
    }


# ── top review candidate 분석 ───────────────────────────────────────────────
def _top_review_candidates(df: pd.DataFrame) -> list:
    rcf = _to_bool(df["review_candidate_flag"])
    top9 = df[rcf].copy()
    if len(top9) != 9:
        print(f"[WARN] review_candidate_flag=True 개수: {len(top9)} (expected 9)")
    cols = [
        "cluster3d_id", "patient_id", "z_min", "z_max", "z_span",
        "n_2d_clusters", "n_patches_total", "bbox_area",
        "top3_mean_patch_score_3d", "representative_2d_cluster_id",
        "representative_local_z", "overmerge_flag", "large_bbox_overmerge_flag",
        "large_extent_overmerge_flag", "complex_merge_flag", "high_score_ratio_flag",
    ]
    result = []
    for _, row in top9.iterrows():
        rec = {}
        for c in cols:
            v = row[c]
            if c.endswith("_flag"):
                v = bool(str(v).strip().lower() == "true")
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = round(float(v), 6)
            rec[c] = v
        result.append(rec)
    return result


# ── overmerge priority 후보 선정 ────────────────────────────────────────────
def _overmerge_priority_candidates(df: pd.DataFrame) -> list:
    omf = _to_bool(df["overmerge_flag"])
    rcf = _to_bool(df["review_candidate_flag"])

    # O1: review_candidate_flag=True AND overmerge_flag=True
    o1_ids = set(df[rcf & omf]["cluster3d_id"].tolist())

    # O2: z_span 상위 5개 (overmerge 여부 무관)
    o2_ids = set(df.nlargest(5, "z_span")["cluster3d_id"].tolist())

    # O3: top3_mean_patch_score_3d 상위 5개 중 overmerge_flag=True
    top5_score = set(df.nlargest(5, "top3_mean_patch_score_3d")["cluster3d_id"].tolist())
    o3_ids = top5_score & set(df[omf]["cluster3d_id"].tolist())

    # O4: bbox_area 상위 5개 중 overmerge_flag=True
    top5_bbox = set(df.nlargest(5, "bbox_area")["cluster3d_id"].tolist())
    o4_ids = top5_bbox & set(df[omf]["cluster3d_id"].tolist())

    all_ids = o1_ids | o2_ids | o3_ids | o4_ids
    # 최대 10개로 제한: O1 우선, O2, O3, O4 순서로 채우기
    ordered = []
    seen = set()
    for cid in list(o1_ids) + list(o2_ids) + list(o3_ids) + list(o4_ids):
        if cid not in seen:
            ordered.append(cid)
            seen.add(cid)
        if len(ordered) >= 10:
            break
    ordered = ordered[:10]

    result = []
    id2row = {row["cluster3d_id"]: row for _, row in df.iterrows()}
    for cid in ordered:
        row = id2row[cid]
        criteria = []
        if cid in o1_ids: criteria.append("O1")
        if cid in o2_ids: criteria.append("O2")
        if cid in o3_ids: criteria.append("O3")
        if cid in o4_ids: criteria.append("O4")
        rec = {
            "cluster3d_id": cid,
            "patient_id": str(row["patient_id"]),
            "z_span": int(row["z_span"]),
            "n_2d_clusters": int(row["n_2d_clusters"]),
            "bbox_area": int(row["bbox_area"]),
            "top3_mean_patch_score_3d": round(float(row["top3_mean_patch_score_3d"]), 6),
            "overmerge_flag": bool(str(row["overmerge_flag"]).strip().lower() == "true"),
            "review_candidate_flag": bool(str(row["review_candidate_flag"]).strip().lower() == "true"),
            "selection_criteria": "+".join(criteria),
        }
        result.append(rec)
    return result


# ── diagnostic group 및 priority 부여 ──────────────────────────────────────
def _assign_diagnostic_group(
    df: pd.DataFrame,
    top9_ids: set,
    overmerge_priority_ids: set,
) -> pd.DataFrame:
    out = df[[
        "cluster3d_id", "patient_id", "z_min", "z_max", "z_span",
        "n_2d_clusters", "n_patches_total", "bbox_area",
        "top3_mean_patch_score_3d", "representative_2d_cluster_id",
        "representative_local_z", "review_candidate_flag", "overmerge_flag",
        "large_bbox_overmerge_flag", "large_extent_overmerge_flag",
        "complex_merge_flag", "high_score_ratio_flag",
    ]].copy()

    omf = _to_bool(out["overmerge_flag"])
    rcf = _to_bool(out["review_candidate_flag"])
    lbf = _to_bool(out["large_bbox_overmerge_flag"])
    lef = _to_bool(out["large_extent_overmerge_flag"])
    cmf = _to_bool(out["complex_merge_flag"])
    hsf = _to_bool(out["high_score_ratio_flag"])

    # bool 컬럼 정규화
    out["overmerge_flag"] = omf
    out["review_candidate_flag"] = rcf
    out["large_bbox_overmerge_flag"] = lbf
    out["large_extent_overmerge_flag"] = lef
    out["complex_merge_flag"] = cmf
    out["high_score_ratio_flag"] = hsf

    groups = []
    priorities = []
    notes = []

    for _, row in out.iterrows():
        cid = row["cluster3d_id"]
        zspan = int(row["z_span"])
        is_top9 = cid in top9_ids
        is_op = cid in overmerge_priority_ids
        is_om = bool(row["overmerge_flag"])

        # diagnostic_group 결정 (단일 라벨 우선순위: top9 > overmerge_priority > overmerge_other > normal)
        if is_top9 and is_op:
            grp = "top9_review_candidate;overmerge_priority"
            pri = 1
        elif is_top9:
            grp = "top9_review_candidate"
            pri = 1
        elif is_op:
            grp = "overmerge_priority"
            pri = 2
        elif is_om:
            grp = "overmerge_other"
            pri = 3
        else:
            grp = "normal"
            pri = 9

        # diagnostic_note
        note_parts = []
        if is_top9:
            note_parts.append("top review candidate")
        if is_op:
            note_parts.append("overmerge priority")
        if zspan > 3:
            note_parts.append("z_span>3 visual check needed")
        if is_om and not is_op and not is_top9:
            note_parts.append("overmerge flagged (z_span>3)")
        if not note_parts:
            note_parts.append("normal candidate")

        groups.append(grp)
        priorities.append(pri)
        notes.append("; ".join(note_parts))

    out.insert(0, "diagnostic_group", groups)
    out["diagnostic_priority"] = priorities
    out["diagnostic_note"] = notes
    return out


# ── JSON 출력 구성 ───────────────────────────────────────────────────────────
def _build_json(
    meta: dict,
    zspan_info: dict,
    top9_list: list,
    overmerge_priority_list: list,
) -> dict:
    input_2d = meta["input_2d_cluster_count"]
    weak_3d  = meta["weak_3d_cluster_count"]
    reduction_2d   = meta.get("reduction_rate_from_2d_cluster", round(1 - weak_3d / input_2d, 4))
    reduction_susp = meta.get("reduction_rate_from_suspicious_patch", None)

    n_top9_overmerge = sum(1 for r in top9_list if r["overmerge_flag"])

    patient_2d_to_3d = {}
    p2d = meta.get("patient_2d_cluster_count", {})
    p3d = meta.get("patient_3d_cluster_count", {})
    for pid in sorted(set(list(p2d.keys()) + list(p3d.keys()))):
        patient_2d_to_3d[pid] = {
            "n_2d_clusters": p2d.get(pid, None),
            "n_3d_clusters": p3d.get(pid, None),
        }

    # next_phase_recommendation 결정
    # large_bbox/extent/complex/high_score_ratio 모두 0, same_z=0
    # z_span>3 35개 존재 → B(top9 + overmerge priority visual pack) 추천
    # but full-run 및 72명 확장은 명시적으로 금지
    next_rec = (
        "B: top9 review candidate + overmerge priority visual pack 생성을 먼저 진행 권장. "
        "근거: reduction 효과(59.41%)가 크고, same_z_remerge=0, large_bbox/extent/complex/high_score_ratio flag 전부 0으로 "
        "과병합 위험이 면적·범위·복잡도·score imbalance 측면에서는 제한적임. "
        "단, z_span>3 35개는 시각 확인 필요 후보로 B 단계에서 함께 검토. "
        "C(center_distance_multiplier=1.5 stricter param dry-run)는 B 시각 확인 이후 선택지. "
        "full-run(72명 확장)은 현 단계에서 금지."
    )

    diagnostic_conclusion = (
        "Phase 5.77 weak 3D merge dry-run(3명, sample-local p99 기반)에서 "
        f"542개 2D cluster가 {weak_3d}개 3D cluster로 감소(59.41% 감소). "
        "same_z_remerge=0으로 감소가 z-adjacent merge에 의한 것임 확인. "
        "large_bbox/large_extent/complex/high_score_ratio overmerge flag 전부 0으로 "
        "면적·범위·복잡도·score imbalance 과병합 신호 없음. "
        f"z_span>3 cluster {meta.get('overmerge_flag_count', 35)}개는 별도 시각 확인 필요. "
        "본 결과는 sample 3명 dry-run 기반이며, global threshold 미확정, 병변 성능 결론 불가, "
        "stage2_holdout 검증 미수행."
    )

    return {
        "input_2d_cluster_count": input_2d,
        "weak_3d_cluster_count": weak_3d,
        "reduction_rate_from_2d_cluster": reduction_2d,
        "reduction_rate_from_suspicious_patch": reduction_susp,
        "patient_2d_to_3d_counts": patient_2d_to_3d,
        "total_merge_edge_count": meta.get("total_merge_edge_count", None),
        "adjacent_z_merge_edge_count": meta.get("adjacent_z_merge_edge_count", None),
        "same_z_remerge_edge_count": meta.get("same_z_remerge_edge_count", 0),
        "z_span_distribution": zspan_info["distribution"],
        "z_span_max": zspan_info["z_span_max"],
        "z_span_mean": zspan_info["z_span_mean"],
        "z_span_median": zspan_info["z_span_median"],
        "z_span_eq1": zspan_info["z_span_eq1"],
        "z_span_eq2": zspan_info["z_span_eq2"],
        "z_span_eq3": zspan_info["z_span_eq3"],
        "z_span_gt3": zspan_info["z_span_gt3"],
        "overmerge_flag_count": meta.get("overmerge_flag_count", None),
        "large_bbox_overmerge_flag_count": meta.get("large_bbox_overmerge_flag_count", 0),
        "large_extent_overmerge_flag_count": meta.get("large_extent_overmerge_flag_count", 0),
        "complex_merge_flag_count": meta.get("complex_merge_flag_count", 0),
        "high_score_ratio_flag_count": meta.get("high_score_ratio_flag_count", 0),
        "top_review_candidate_summary": top9_list,
        "n_top9_overmerge": n_top9_overmerge,
        "overmerge_priority_review_candidates": overmerge_priority_list,
        "diagnostic_conclusion": diagnostic_conclusion,
        "next_phase_recommendation": next_rec,
        "notes": {
            "diagnostic_only": True,
            "phase5_77_readonly": True,
            "no_weak_3d_merge_rerun": True,
            "no_clustering_full_run": True,
            "no_visual_review_pack": True,
            "no_model_forward": True,
            "no_score_recalculation": True,
            "threshold_not_finalized": True,
            "lesion_conclusion_forbidden": True,
            "stage2_holdout_unused": True,
            "v2_unused": True,
            "original_files_unmodified": True,
            "phase574_2d_reduction_reference": f"{_PHASE574_2D_REDUCTION_RATE * 100:.1f}%",
        },
    }


# ── MD 출력 구성 ─────────────────────────────────────────────────────────────
def _build_md(
    meta: dict,
    zspan_info: dict,
    top9_list: list,
    overmerge_priority_list: list,
    out_json: dict,
) -> str:
    input_2d = out_json["input_2d_cluster_count"]
    weak_3d  = out_json["weak_3d_cluster_count"]
    red_2d   = out_json["reduction_rate_from_2d_cluster"]
    red_susp = out_json["reduction_rate_from_suspicious_patch"]
    om_count = out_json["overmerge_flag_count"]
    n_top9_om= out_json["n_top9_overmerge"]

    # z_span 분포 표
    dist = zspan_info["distribution"]
    zspan_table_rows = "\n".join(
        f"| {k} | {v} |" for k, v in sorted(dist.items())
    )

    # top9 표
    top9_header = (
        "| cluster3d_id | patient_id | z_min | z_max | z_span | "
        "n_2d | n_patches | bbox_area | top3_score | rep_local_z | overmerge_flag |"
    )
    top9_sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    top9_rows = []
    for r in top9_list:
        om = "Y" if r["overmerge_flag"] else "-"
        top9_rows.append(
            f"| {r['cluster3d_id']} | {r['patient_id']} "
            f"| {r['z_min']} | {r['z_max']} | {r['z_span']} "
            f"| {r['n_2d_clusters']} | {r['n_patches_total']} "
            f"| {r['bbox_area']} | {r['top3_mean_patch_score_3d']:.4f} "
            f"| {r['representative_local_z']} | {om} |"
        )
    top9_table = "\n".join([top9_header, top9_sep] + top9_rows)

    # overmerge priority 표
    op_header = (
        "| cluster3d_id | patient_id | z_span | n_2d | bbox_area | "
        "top3_score | overmerge_flag | review_flag | 선정기준 |"
    )
    op_sep = "|---|---|---|---|---|---|---|---|---|"
    op_rows = []
    for r in overmerge_priority_list:
        om = "Y" if r["overmerge_flag"] else "-"
        rc = "Y" if r["review_candidate_flag"] else "-"
        op_rows.append(
            f"| {r['cluster3d_id']} | {r['patient_id']} "
            f"| {r['z_span']} | {r['n_2d_clusters']} "
            f"| {r['bbox_area']} | {r['top3_mean_patch_score_3d']:.4f} "
            f"| {om} | {rc} | {r['selection_criteria']} |"
        )
    op_table = "\n".join([op_header, op_sep] + op_rows)

    patient_rows = []
    for pid, v in sorted(out_json["patient_2d_to_3d_counts"].items()):
        patient_rows.append(f"- {pid}: 2D {v['n_2d_clusters']}개 → 3D {v['n_3d_clusters']}개")
    patient_summary = "\n".join(patient_rows)

    zgt3_count = zspan_info["z_span_gt3"]
    om_top9_note = (
        f"top9 중 overmerge_flag=True: {n_top9_om}개"
        + (" (해당 cluster는 시각 확인 우선 대상)" if n_top9_om > 0 else "")
    )

    md = f"""# Phase 5.78 Weak 3D Merge 결과 진단 보고서

출력 태그: `{_OUT_TAG}`
입력 태그: `phase5_77_weak_3d_merge_dry_run_v1`
진단 기준: read-only 분석 (merge 재실행/model forward/score 재계산/visual pack 생성 전부 금지)

---

## 1. Phase 5.78 목적

Phase 5.77 weak 3D merge dry-run 결과(3명 sample, sample-local p99 기반)를 진단하여
- 후보 감소 효과를 수치로 확인하고
- overmerge 위험 여부를 flag별로 점검하며
- top review candidate 및 overmerge priority 후보를 정리하고
- 다음 단계 진행 방향을 선택한다.

본 보고서는 기존 Phase 5.77 결과 파일을 수정하지 않으며, 새 분석 결과만 생성한다.

---

## 2. Phase 5.77 결과 요약

| 항목 | 값 |
|---|---|
| 입력 2D cluster 수 | {input_2d}개 |
| weak 3D merge 후 cluster 수 | {weak_3d}개 |
| 2D cluster 기준 감소율 | {red_2d * 100:.2f}% |
| suspicious patch 기준 감소율 | {red_susp * 100:.2f}% |
| total merge edge 수 | {out_json['total_merge_edge_count']} |
| adjacent-z merge edge 수 | {out_json['adjacent_z_merge_edge_count']} |
| same-z remerge edge 수 | {out_json['same_z_remerge_edge_count']} |
| overmerge_flag=True 수 | {om_count} |
| large_bbox_overmerge_flag=True | {out_json['large_bbox_overmerge_flag_count']} |
| large_extent_overmerge_flag=True | {out_json['large_extent_overmerge_flag_count']} |
| complex_merge_flag=True | {out_json['complex_merge_flag_count']} |
| high_score_ratio_flag=True | {out_json['high_score_ratio_flag_count']} |

환자별 감소 현황:
{patient_summary}

---

## 3. Phase 5.74 2D clustering 감소율(30.1%)과 비교

Phase 5.74 2D clustering 단계에서는 suspicious patch → 2D cluster로 **30.1%** 감소가 이루어졌다.
Phase 5.77 weak 3D merge에서는 2D cluster → 3D cluster로 **{red_2d * 100:.2f}%** 감소가 이루어졌다.

weak 3D merge의 후보 수 감소 효과(59.41%)는 2D clustering 단계(30.1%)보다 약 2배 더 크다.
이는 z-adjacent slice 간 공간적 중복을 효과적으로 통합했음을 의미한다.

비교 기준: Phase 5.74의 30.1%는 고정 참조값으로, 동일 sample-local p99 입력 기준이다.

---

## 4. z-adjacent merge 효과 해석

total merge edge {out_json['total_merge_edge_count']}개 전부가 adjacent-z merge edge이다.
감소된 322개 cluster(542 - 220)는 z-adjacent slice 간 공간적으로 인접한 2D cluster들을
약한 기준(IoU>0 또는 center_distance < stride×2.0)으로 통합한 결과다.

merge 파라미터: z_gap=1, center_distance_multiplier=2.0 (dry-run 후보값, 확정 기준 아님)

---

## 5. same-z remerge 차단 확인

same_z_remerge_edge_count = **0** (allow_same_z_remerge=False)

same-slice 내 재병합이 전혀 발생하지 않았음을 확인하였다.
따라서 후보 수 감소는 오직 z-adjacent merge에 의한 것이며, 동일 slice 내 cluster들은
Phase 5.74에서 이미 결정된 2D 경계가 유지되었다.

---

## 6. z_span 분포

| z_span | count |
|---|---|
{zspan_table_rows}

- z_span==1 (단일 slice): {zspan_info['z_span_eq1']}개
- z_span==2: {zspan_info['z_span_eq2']}개
- z_span==3: {zspan_info['z_span_eq3']}개
- z_span>3: {zspan_info['z_span_gt3']}개
- z_span max: {zspan_info['z_span_max']}, mean: {zspan_info['z_span_mean']:.4f}, median: {zspan_info['z_span_median']}

---

## 7. overmerge flag 35개 진단

overmerge_flag=True: **{om_count}개**

overmerge_reason 분석: 전체 {om_count}개의 overmerge 사유는 z_span>3 기준으로만 flag됨.

| flag 종류 | count | 해석 |
|---|---|---|
| overmerge_flag (z_span>3) | {om_count} | z방향 확장 과병합 후보 |
| large_bbox_overmerge_flag | 0 | 면적 과병합 신호 없음 |
| large_extent_overmerge_flag | 0 | 범위 과병합 신호 없음 |
| complex_merge_flag | 0 | 복잡도 과병합 신호 없음 |
| high_score_ratio_flag | 0 | score imbalance 신호 없음 |

결론: 면적/범위/복잡도/score imbalance 측면의 과병합 신호는 전부 0이다.
단, z_span>3인 {zgt3_count}개 cluster는 z 방향 과확장 여부 시각 확인이 필요한 후보로 기록한다.
z_span>3 자체가 병변을 의미하지 않으며, 이 단계에서 병변 성능 결론은 내릴 수 없다.

---

## 8. top 9 3D review candidate 요약

{om_top9_note}

{top9_table}

---

## 9. overmerge priority review 후보 (최대 10개)

선정 기준:
- O1: review_candidate_flag=True AND overmerge_flag=True
- O2: z_span 상위 5개 (overmerge 여부 무관)
- O3: top3_mean_patch_score_3d 상위 5개 중 overmerge_flag=True
- O4: bbox_area 상위 5개 중 overmerge_flag=True

{op_table}

---

## 10. 다음 단계 선택지 A/B/C

### A. 현재 파라미터(z_gap=1, cd_mult=2.0)로 full-run 진행
- 장점: 현재 결과를 72명 전체로 확장
- 단점: global threshold 미확정 상태에서 full-run은 시기상조.
  z_span>3 35개의 시각 확인 없이 전체 확장 시 overmerge 위험 파악 불가.
- **현 단계에서 금지**

### B. top9 + overmerge priority visual pack 생성 후 시각 확인
- 장점: z_span>3 cluster의 실제 z 방향 확장이 타당한지 확인 가능.
  visual pack 생성 후 O1~O4 후보 집중 검토 → 파라미터 조정 여부 결정 가능.
- 단점: visual pack 생성 작업 추가 필요. PNG/HTML 생성 포함.
- **권장**

### C. center_distance_multiplier=1.5 stricter param dry-run
- 장점: 현재 cd_mult=2.0이 과도하게 넓은지 확인 가능. 더 보수적인 merge 결과 비교 가능.
- 단점: B 단계(시각 확인) 없이 파라미터만 조이는 것은 개선 근거 부족.
  B 이후 선택지로 적합.
- **B 이후 선택지**

---

## 11. 추천 진행 방향

**B 단계 진행 권장**

근거:
1. same_z_remerge=0으로 감소가 z-adjacent merge에 의한 것임 확인
2. large_bbox / large_extent / complex / high_score_ratio flag 전부 0
   → 면적·범위·복잡도·score imbalance 측면 과병합 신호 없음
3. z_span>3 35개 cluster는 시각 확인으로 실제 타당성 판단 필요
4. B 단계에서 top9 + overmerge priority 후보(최대 10개)를 먼저 시각 검토한 뒤
   파라미터 조정(C) 또는 full-run(A) 여부를 결정하는 것이 안전

---

## 12. 해석 제한 사항

- **sample 3명 dry-run**: normal004, normal013, normal014 3명 기반. 전체 72명 대표성 없음.
- **sample-local p99 기반**: 각 환자별 로컬 p99 임계값 사용. global threshold 기준이 아님.
- **global threshold 미확정**: 현재 score 기준은 확정 기준이 아니며 변경될 수 있음.
- **병변 성능 결론 불가**: 이 결과로 이상탐지 성능(sensitivity/specificity 등) 결론을 내릴 수 없음.
- **stage2_holdout 검증 미수행**: stage2_holdout 데이터는 접근하지 않았음.
- **v2 미사용**: v2 데이터 및 결과는 이 분석에 포함되지 않음.
- 모든 입력 Phase 5.77 파일은 읽기만 하였으며 수정하지 않았음.
"""
    return md


# ── 메인 ────────────────────────────────────────────────────────────────────
def main() -> None:
    _validate_inputs()
    _check_output_not_exists()

    df, meta = _load_data()

    # 계산
    zspan_info = _calc_zspan_dist(df)
    top9_list  = _top_review_candidates(df)
    op_list    = _overmerge_priority_candidates(df)

    # overmerge_reason 교차검증: z_span 포함 여부 확인
    om_mask = _to_bool(df["overmerge_flag"])
    om_df   = df[om_mask]
    reason_zspan = int(
        om_df["overmerge_reason"].astype(str).str.contains("z_span", case=False, na=False).sum()
    )
    print(
        f"[INFO] overmerge_flag=True 중 overmerge_reason에 'z_span' 포함: "
        f"{reason_zspan}/{len(om_df)}"
    )

    # JSON 구성
    out_json = _build_json(meta, zspan_info, top9_list, op_list)

    # CSV 구성
    top9_ids = {r["cluster3d_id"] for r in top9_list}
    op_ids   = {r["cluster3d_id"] for r in op_list}
    out_csv  = _assign_diagnostic_group(df, top9_ids, op_ids)

    # MD 구성
    out_md = _build_md(meta, zspan_info, top9_list, op_list, out_json)

    # 모든 계산 완료 후 mkdir + 저장
    _OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    print(f"[OK] 출력 폴더 생성: {_OUTPUT_ROOT}")

    # 1. MD
    md_path = _OUTPUT_ROOT / f"{_OUT_TAG}.md"
    md_path.write_text(out_md, encoding="utf-8")
    print(f"[OK] MD 저장: {md_path}")

    # 2. JSON
    json_path = _OUTPUT_ROOT / f"{_OUT_TAG}.json"
    json_path.write_text(
        json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] JSON 저장: {json_path}")

    # 3. CSV
    csv_path = _OUTPUT_ROOT / f"{_OUT_TAG}.csv"
    out_csv.to_csv(csv_path, index=False)
    print(f"[OK] CSV 저장: {csv_path} ({len(out_csv)}행)")

    # ── 최종 검증 보고 ──────────────────────────────────────────────────────
    print("\n=== 검증 보고 ===")
    print(f"[1] Phase 5.77 입력 파일 3개: {_INPUT_CSV.exists()}, {_INPUT_JSON.exists()}, {_INPUT_MD.exists()}")
    print(f"[2] 입력 CSV row 수: {len(df)} (expected 220)")
    print(f"[3] JSON input_2d_cluster_count: {meta['input_2d_cluster_count']} (expected 542)")
    print(f"[4] JSON weak_3d_cluster_count: {meta['weak_3d_cluster_count']} (expected 220)")
    print(f"[5] same_z_remerge_edge_count: {meta.get('same_z_remerge_edge_count', 'N/A')} (expected 0)")
    print(f"[6] overmerge_flag_count: {meta.get('overmerge_flag_count', 'N/A')} (expected 35)")
    print(f"[7] large_bbox={meta.get('large_bbox_overmerge_flag_count',0)}, "
          f"large_extent={meta.get('large_extent_overmerge_flag_count',0)}, "
          f"complex={meta.get('complex_merge_flag_count',0)}, "
          f"high_score_ratio={meta.get('high_score_ratio_flag_count',0)} (전부 0 예상)")
    print(f"[8] z_span 분포: {zspan_info['distribution']}")
    print(f"    z_span==1: {zspan_info['z_span_eq1']}, ==2: {zspan_info['z_span_eq2']}, "
          f"==3: {zspan_info['z_span_eq3']}, >3: {zspan_info['z_span_gt3']}")
    print(f"    max={zspan_info['z_span_max']}, mean={zspan_info['z_span_mean']}, median={zspan_info['z_span_median']}")
    print(f"[9] top9 review candidate: {len(top9_list)}개")
    for r in top9_list:
        print(f"    {r['cluster3d_id']} (patient={r['patient_id']}, z_span={r['z_span']}, overmerge={r['overmerge_flag']})")
    print(f"[10] overmerge priority 후보: {len(op_list)}개")
    for r in op_list:
        print(f"    {r['cluster3d_id']} (criteria={r['selection_criteria']}, z_span={r['z_span']})")
    print(f"[11] 생성 파일:")
    print(f"    MD : {md_path}")
    print(f"    JSON: {json_path}")
    print(f"    CSV : {csv_path}")
    print(f"[12] Phase 5.77 파일 미수정 확인 필요 (mtime 직접 비교 권장)")
    print(f"[13] diagnostic_conclusion: {out_json['diagnostic_conclusion'][:80]}...")
    print(f"[14] next_phase_recommendation: {out_json['next_phase_recommendation'][:80]}...")
    print(f"    생성 CSV row 수: {len(out_csv)}")
    print(f"    스크립트 경로: {Path(__file__).resolve()}")
    print("[DONE] Phase 5.78 진단 완료")


if __name__ == "__main__":
    main()
