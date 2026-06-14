#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1-D1.3_wall_mediastinum_fp_overlay_target_selection

B1-D1.2 원인 진단 CSV/JSON 을 read-only 로 분석하여 overlay 눈검증 대상만 선정한다.
- overlay PNG 를 생성하지 않는다 (B1-D1.4 별도 승인).
- 기존 CSV/JSON/mask/score 를 수정하지 않는다.
- AD_wall_med_inside 를 A/D 로 단정하지 않으며, LESION_RISK_partial 을 실제 병변손실로 단정하지 않는다.
- 출력 파일이 이미 있으면 즉시 중단(덮어쓰기 금지).
"""
import csv
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"
IN_CSV = DIR / "b1d1_fp_cause_diagnostic.csv"
IN_JSON = DIR / "b1d1_cause_summary.json"
OUT_CSV = DIR / "b1d1_overlay_target_selection.csv"
OUT_JSON = DIR / "b1d1_overlay_target_selection_summary.json"
OUT_MD = DIR / "b1d1_overlay_target_selection_report.md"

REQ_COLS = ["review_id", "patient_id", "safety_role", "human_label", "candidate_local_z",
            "candidate_y0", "candidate_x0", "candidate_score", "roi_0_0_patch_ratio",
            "refined_roi_ratio", "center_in_refined_roi", "cause_class"]

OVERLAY_Q = {
    "AD_wall_med_inside": "이 patch는 더 깎아도 되는 흉벽/종격동 잔여부(A)인가, 아니면 폐실질/병변 보존 때문에 남길 수밖에 없는 경계 정상구조(D)인가?",
    "B_boundary": "이 patch의 고점수 원인은 patch가 refined ROI 경계를 걸쳐 흉벽/종격동을 포함하기 때문인가?",
    "LESION_RISK_partial": "이 병변 보호 후보는 refined ROI 경계에서 병변 일부가 잘릴 위험이 있는가?",
    "AD_other_inside": "이 ROI 내부 고점수는 흉벽/종격동 문제가 아니라 vessel/diaphragm 등 다른 구조물 FP인가?",
}


def fail(msg):
    print(f"[B1-D1.3][중단] {msg}", file=sys.stderr)
    sys.exit(2)


def main():
    # ---- collision guard (덮어쓰기 금지) ----
    for p in (OUT_CSV, OUT_JSON, OUT_MD):
        if p.exists():
            fail(f"출력 파일이 이미 존재함(덮어쓰기 금지): {p}")

    # ---- 입력/검증 ----
    if not IN_CSV.exists():
        fail(f"입력 CSV 없음: {IN_CSV}")
    if not IN_JSON.exists():
        fail(f"입력 JSON 없음: {IN_JSON}")
    rows = list(csv.DictReader(open(IN_CSV, encoding="utf-8")))
    j = json.load(open(IN_JSON, encoding="utf-8"))
    if len(rows) != 54:
        fail(f"CSV row 수 {len(rows)} != 54")
    if j.get("n_rows") != 54:
        fail(f"JSON n_rows {j.get('n_rows')} != 54")
    if j.get("stage2_holdout_access") != 0:
        fail(f"stage2_holdout_access {j.get('stage2_holdout_access')} != 0")
    miss = [c for c in REQ_COLS if c not in rows[0]]
    if miss:
        fail(f"필수 컬럼 누락: {miss}")

    def rr(r):
        return float(r["refined_roi_ratio"])

    def sc(r):
        return float(r["candidate_score"])

    g_wall = [r for r in rows if r["cause_class"] == "AD_wall_med_inside"]
    g_bound = [r for r in rows if r["cause_class"] == "B_boundary"]
    g_risk = [r for r in rows if r["cause_class"] == "LESION_RISK_partial"]
    g_other = [r for r in rows if r["cause_class"] == "AD_other_inside"]

    selected = []  # (row, group, reason)

    # A. AD_wall_med_inside : 전부 포함 (PatchCore 필요성 판단의 핵심 분기점)
    for r in g_wall:
        selected.append((r, "AD_wall_med_inside", "core_branch_A_or_D_all_included"))

    # B. B_boundary : ratio 낮음2/중간2/높음2 (구간 내 score desc tie-break)
    bs = sorted(g_bound, key=lambda r: (rr(r), -sc(r)))
    n = len(bs)
    if n >= 6:
        idx = sorted({0, 1, n // 2 - 1, n // 2, n - 2, n - 1})
    else:
        idx = list(range(n))
    for i in idx:
        r = bs[i]
        band = "ratio_low" if i <= 1 else ("ratio_high" if i >= n - 2 else "ratio_mid")
        selected.append((r, "B_boundary", f"{band}_ratio{rr(r):.3f}_score{sc(r):.0f}"))

    # C. LESION_RISK_partial : ratio asc + score desc, 환자별 최대 2개 (분산), 6개 목표
    rs = sorted(g_risk, key=lambda r: (rr(r), -sc(r)))
    pcap = defaultdict(int)
    risk_sel = []
    for r in rs:
        if pcap[r["patient_id"]] >= 2:
            continue
        risk_sel.append(r)
        pcap[r["patient_id"]] += 1
        if len(risk_sel) >= 6:
            break
    if len(risk_sel) < 6:  # cap 으로 부족하면 잔여 채움
        for r in rs:
            if r in risk_sel:
                continue
            risk_sel.append(r)
            if len(risk_sel) >= 6:
                break
    for r in risk_sel:
        selected.append((r, "LESION_RISK_partial", f"low_ratio{rr(r):.3f}_score{sc(r):.0f}"))

    # D. AD_other_inside : score 상위 2개 (흉벽/종격동 문제와 혼입 확인용 보조)
    for r in sorted(g_other, key=lambda r: -sc(r))[:2]:
        selected.append((r, "AD_other_inside", f"score_top{sc(r):.0f}"))

    # ---- CSV ----
    out_cols = ["selection_id"] + REQ_COLS + ["selection_group", "selection_reason", "overlay_question"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for i, (r, grp, reason) in enumerate(selected, 1):
            w.writerow({"selection_id": f"SEL{i:03d}",
                        **{c: r[c] for c in REQ_COLS},
                        "selection_group": grp,
                        "selection_reason": reason,
                        "overlay_question": OVERLAY_Q[r["cause_class"]]})

    # ---- summary JSON ----
    by_group = dict(Counter(grp for _, grp, _ in selected))
    by_cause = dict(Counter(r["cause_class"] for r, _, _ in selected))
    by_patient = dict(Counter(r["patient_id"] for r, _, _ in selected))
    summary = {
        "input_csv": str(IN_CSV),
        "input_json": str(IN_JSON),
        "n_input_rows": len(rows),
        "n_selected_rows": len(selected),
        "selected_by_group": by_group,
        "selected_by_cause_class": by_cause,
        "selected_by_patient": by_patient,
        "stage2_holdout_access": 0,
        "png_generated": False,
        "patchcore_implemented": False,
        "score_modified": False,
        "next_step_recommendation": "B1-D1.4 selected target overlay PNG 생성 preflight/승인",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- report MD ----
    md = f"""# B1-D1.3 Overlay Target Selection Report

## B1-D1.2 핵심 분포 (입력 {len(rows)}행)
- fp_candidate(30): B_boundary 11, AD_wall_med_inside 10, AD_other_inside 9
- lesion_protect(24): lesion_kept 17, LESION_RISK_partial 7
- C_outside_roi: 0

## 왜 C_outside_roi=0 이 중요한가
refined_roi_ratio 최소가 0.396 으로 ROI 밖 후보가 없다. 따라서 "ranking 에 ROI 밖 patch 가 포함됨(C)" 원인은 배제된다.
즉 FP 는 ranking 단계의 ROI 누수가 아니라, refined ROI **경계(B)** 또는 refined ROI **내부 잔존(AD)** 에서 발생한다.

## 왜 AD_wall_med_inside 가 PatchCore 판단의 핵심인가
흉벽/종격동인데 refined ROI 안에 거의 다 남은 10개다. 이들이
(A) 더 깎아도 되는 흉벽/종격동 잔여인지, (D) 폐실질/병변 보존 때문에 남길 수밖에 없는 경계 정상구조인지에 따라 해법이 갈린다.
- A 가 많으면 mask 정밀화로 충분 → PatchCore 불필요 가능.
- D 가 많으면 ROI 로 뺄 수 없는 정상 고점수 → 표현력 향상(PatchCore 등) 검토 가치.
overlay 눈검증 전까지 A/D 는 단정하지 않는다.

## 왜 LESION_RISK_partial 을 같이 봐야 하는가
병변 보호 후보 중 refined ROI overlap 0.10~0.90 인 7개다. 흉벽/종격동을 "더 깎는"(A 해법) 방향이
병변 일부 절단으로 이어질 위험을 보여준다. AD 를 A 로 처리해 ROI 를 확장 절제하면 이들이 같이 깎일 수 있으므로,
A 해법의 안전성을 함께 평가해야 한다. (실제 병변 손실로 단정하지 않으며, overlay 확인 대상이다.)

## 선정된 overlay 대상
- 총 {len(selected)}개
- selected_by_group: {by_group}
- selected_by_cause_class: {by_cause}

## 다음 단계
B1-D1.4 selected target overlay PNG 생성 preflight / 승인.
"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    # ---- 콘솔 요약 ----
    print(f"selected rows : {len(selected)}")
    print(f"by_group      : {by_group}")
    print(f"by_cause_class: {by_cause}")
    print(f"by_patient    : {by_patient}")
    print(f"생성: {OUT_CSV.name}, {OUT_JSON.name}, {OUT_MD.name}")


if __name__ == "__main__":
    main()
